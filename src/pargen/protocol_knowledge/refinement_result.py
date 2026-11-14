"""Publish refinements with an explicit status; never let a partial attempt clobber a validated one."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path


class IncompleteRefinementError(RuntimeError):
    def __init__(self, diagnostic_path: Path, status: dict):
        self.diagnostic_path = diagnostic_path
        self.status = status
        super().__init__(f"Refinement incomplete; diagnostic retained at {diagnostic_path}. "
                         "No validated output was published.")


def refinement_status_path(path: Path) -> Path:
    # Keep status and diagnostic JSON out of legacy fixed_protocol_tree_*.json
    # auto-discovery. Neither is itself a protocol-tree candidate.
    return Path(path).parent / "refinement_status" / f"{Path(path).name}.status.json"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_refinement(path: Path, payload: dict, report, *, stage: str,
                       require_traffic: bool = False, extra_failures=(), metadata=None) -> Path:
    """Never overwrite a previous validated output on an unsuccessful attempt.

    This is a repair completion check, NOT an accuracy/knowledge-F1 filter.
    Known incomplete artifacts remain available for explicitly scoped analysis.
    """
    issues = getattr(report, "issues", {}) or {}
    errors = [str(getattr(i, "description", i)) for i in issues.values()
              if str(getattr(getattr(i, "severity", None), "name", getattr(i, "severity", ""))).upper() == "ERROR"]
    errors.extend(str(e) for e in (getattr(report, "errors", []) or []))
    errors.extend(str(e) for e in extra_failures)
    processed = int(getattr(report, "traffic_total_samples", 0) or 0)
    successful = int(getattr(report, "traffic_successful_samples", 0) or 0)
    if require_traffic and (processed <= 0 or successful != processed):
        errors.append(f"Traffic validation incomplete: {successful}/{processed}")
    if not bool(getattr(report, "ok", False)):
        errors.append("Final validator did not pass")
    errors = list(dict.fromkeys(errors))
    complete = not errors
    canonical = Path(path)
    # Default: remaining issues are kept as
    # diagnostics, the partial tree is published in place labelled "partial",
    # and the pipeline continues. STEP2_PUBLISH_STRICT=1 keeps an incomplete
    # repair aside and reports it as a failure. Either way a previously
    # validated output at this path is never overwritten.
    strict = os.getenv("STEP2_PUBLISH_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}
    prior_validated = False
    prior_status = refinement_status_path(canonical)
    if not complete and canonical.exists() and prior_status.exists():
        try:
            prior_validated = bool(json.loads(prior_status.read_text(encoding="utf-8")).get("validation_passed"))
        except Exception:
            prior_validated = False
    publish_in_place = complete or (not strict and not prior_validated)
    target = canonical if publish_in_place else canonical.parent / "incomplete" / f"{canonical.stem}.{uuid.uuid4().hex}.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    repair_status = "validated" if complete else ("partial" if publish_in_place else "incomplete")
    status = {"schema": 1, "stage": stage, "process_completed": True,
              "repair_status": repair_status,
              "diagnostic_only": not publish_in_place, "validation_passed": complete,
              "unresolved_errors": errors, "traffic_total_samples": processed,
              "traffic_successful_samples": successful,
              "artifact_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
              "metadata": dict(metadata or {})}
    _atomic_text(target, text)
    _atomic_text(refinement_status_path(target), json.dumps(status, indent=2, ensure_ascii=False))
    if not complete and strict:
        raise IncompleteRefinementError(target, status)
    return target
