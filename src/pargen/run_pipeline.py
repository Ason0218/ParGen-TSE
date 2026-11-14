"""One-click orchestrator to run steps A → B → C for a given protocol dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
PARENT = PACKAGE_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


def _default_step2_cache_namespace(argv: Sequence[str]) -> Optional[str]:
    """The dataset named by ``--protocol`` keys the Stage B fix cache.

    ``pargen.paths`` derives ``STEP2_FIX_CACHE_DIR`` from
    ``PARGEN_STEP2_CACHE_NAMESPACE`` at import time. Without it every run
    shares one directory and concurrent or repeated runs replay each other's
    MCTS patch responses (``patch_section_*_response.json``). An explicit
    environment value still wins.
    """
    for index, arg in enumerate(argv):
        if arg == "--protocol" and index + 1 < len(argv):
            return argv[index + 1].strip() or None
        if arg.startswith("--protocol="):
            return arg.split("=", 1)[1].strip() or None
    return None


if __name__ == "__main__" and not os.getenv("PARGEN_STEP2_CACHE_NAMESPACE"):
    _namespace = _default_step2_cache_namespace(sys.argv[1:])
    if _namespace:
        os.environ["PARGEN_STEP2_CACHE_NAMESPACE"] = _namespace

from pargen.parser_realization.llm_client import LLMConfig as StageCLLMConfig
from pargen.parser_realization.stage_c_pipeline import (
    StageCConfig,
    build_stage_c_backend_config,
    build_stage_c_input_signature,
    run_stage_c as run_stage_c_codegen,
)
from pargen.parser_realization.profile_utils import strip_protocol_hints
from pargen.config_manager import load_api_keys
from pargen.logging_utils import setup_logging
from pargen.paths import DATA_DIR, DEFAULT_API_CONFIG, STEP2_CACHE_DIR

DEFAULT_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash-0731"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"


def _sanitize_variant_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = []
    for ch in str(value).strip():
        if ch.isalnum() or ch in "._-":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("._-")
    return out or None


def _normalize_protocol_name(value: Optional[str]) -> Optional[str]:
    normalized = _sanitize_variant_name(value)
    return normalized.lower() if normalized is not None else None


def _exp_constraint_repair_requires_rebuild(host: Optional[str]) -> bool:
    return str(host or "").strip().lower() in {"wireshark_c", "wireshark_lua"}


def _exp_constraint_repair_validation_action(host: Optional[str]) -> Optional[str]:
    normalized = str(host or "").strip().lower()
    if normalized == "wireshark_c":
        return "wireshark_build"
    if normalized == "wireshark_lua":
        return "lua_syntax"
    if normalized == "scapy":
        return "scapy_syntax"
    if normalized == "zeek_spicy":
        return "zeek_spicy_syntax"
    return None


def _stage_c_exp_constraint_gate_accepts(old_res: Any, new_res: Any) -> tuple[bool, str]:
    """Accept code repair when explicit constraint checks improve.
    
    Normal seeds contribute TP/FN and mutated constraint violations contribute
    FP/TN. Acceptance requires the applicable positive and negative checks."""

    def _maybe_int(obj: Any, name: str) -> Optional[int]:
        raw = getattr(obj, name, None)
        if raw is None:
            return None
        try:
            return int(raw)
        except Exception:
            return None

    if getattr(old_res, "scoring_policy", None) != getattr(new_res, "scoring_policy", None):
        return False, "incompatible_conformance_metrics"
    if (getattr(old_res, "acceptance_semantics", None) == "exact_target_diagnostic"
            or getattr(new_res, "acceptance_semantics", None) == "exact_target_diagnostic"):
        if (getattr(old_res, "negative_isolation_unverified", 0)
                or getattr(new_res, "negative_isolation_unverified", 0)):
            return False, "conformance_test_qualification_required"

    old_tp = _maybe_int(old_res, "tp")
    old_fn = _maybe_int(old_res, "fn")
    new_tp = _maybe_int(new_res, "tp")
    new_fn = _maybe_int(new_res, "fn")
    if old_tp is None or old_fn is None or new_tp is None or new_fn is None:
        return False, "missing_positive_seed_confusion"

    old_fp = _maybe_int(old_res, "fp")
    old_tn = _maybe_int(old_res, "tn")
    new_fp = _maybe_int(new_res, "fp")
    new_tn = _maybe_int(new_res, "tn")
    if old_fp is None or old_tn is None or new_fp is None or new_tn is None:
        return False, "missing_mutant_confusion"

    if new_fn > old_fn or new_tp < old_tp:
        return False, "positive_seed_regression"
    if new_tn < old_tn or new_fp > old_fp:
        return False, "mutant_reject_regression"
    if getattr(new_res, "acceptance_semantics", None) == "exact_target_diagnostic":
        old_target = _maybe_int(old_res, "negative_target_diagnostic_success")
        new_target = _maybe_int(new_res, "negative_target_diagnostic_success")
        if old_target is None or new_target is None:
            return False, "missing_target_attribution_metrics"
        if new_target < old_target:
            return False, "target_attribution_regression"
        if new_target > old_target:
            return True, "target_attribution_improved"
    if new_tn > old_tn or new_fp < old_fp:
        return True, "explicit_mutant_reject_improved"
    if new_fn < old_fn or new_tp > old_tp:
        return True, "positive_seed_recovered"
    return False, "no_explicit_improvement"


def _stage_c_confusion_regression(old_res: Any, new_res: Any) -> Optional[str]:
    """A lost accepted positive (TP/FN) or a lost detected violation (TN/FP) on the same suite, else None."""
    counts: dict[str, int] = {}
    for name in ("tp", "fn", "fp", "tn"):
        for tag, res in (("old", old_res), ("new", new_res)):
            raw = getattr(res, name, None)
            if raw is None:
                return "missing_positive_seed_confusion" if name in ("tp", "fn") else "missing_mutant_confusion"
            counts[tag + name] = int(raw)
    if counts["newfn"] > counts["oldfn"] or counts["newtp"] < counts["oldtp"]:
        return "positive_seed_regression"
    if counts["newtn"] < counts["oldtn"] or counts["newfp"] > counts["oldfp"]:
        return "mutant_reject_regression"
    return None


def _stage_c_code_repair_accepts(
    old_res: Any, new_res: Any, *, old_conflicts: int, new_conflicts: int, field_check_lost: bool = False,
) -> tuple[bool, str]:
    """AcceptCodeRepair: executable, no lost accepted positive or detected violation, fewer conflicts.

    The conflict counts cover field observations and validity outcomes on the same PKG-constructed suite. The
    validity part compares the confusion counts directly. Suite-level conditions of the older constraint gate
    (unverified isolation of some negatives, target attribution) are the same for every candidate and are not
    acceptance criteria in Sec. IV-D, where unqualified cases are left out of the repair evidence instead.
    """
    if field_check_lost:
        return False, "candidate_field_check_unavailable_or_reduced"
    if getattr(old_res, "scoring_policy", None) != getattr(new_res, "scoring_policy", None):
        return False, "incompatible_conformance_metrics"
    reason = _stage_c_confusion_regression(old_res, new_res)
    if reason is not None:
        return False, reason
    if int(new_conflicts) < int(old_conflicts):
        return True, "conflicts_reduced"
    return False, "no_conflict_reduction"


def _stage_c_field_check_gaps(result: Any, error: Optional[str]) -> List[str]:
    if error:
        return [f"field_check_error: {error}"]
    if result is None:
        return ["field_check_unavailable"]
    return list(result.coverage_gaps)


def _stage_c_pkg_status(*, failures: int, evaluation_gaps: Sequence[Any],
                        preflight_issues: Sequence[Any], field_result: Any, field_error: Optional[str]) -> str:
    if failures > 0 or (field_result is not None and field_result.count > 0):
        return "failed"
    if evaluation_gaps or preflight_issues or _stage_c_field_check_gaps(field_result, field_error):
        return "partial"
    return "complete"




@dataclass
class ProtocolPaths:
    root: Path
    sections_override: Optional[Path] = None
    raw_override: Optional[Path] = None
    output_variant: Optional[str] = None

    def __post_init__(self) -> None:
        # Standardized layout under data/<protocol>/
        self.output_variant = _sanitize_variant_name(self.output_variant)
        self.inputs_dir = self.root / "inputs"
        if self.output_variant:
            self.outputs_dir = self.root / "outputs" / self.output_variant
            self.logs_dir = self.root / "logs" / self.output_variant
            self.cache_dir = self.root / "cache" / self.output_variant
        else:
            self.outputs_dir = self.root / "outputs"
            self.logs_dir = self.root / "logs"
            self.cache_dir = self.root / "cache"

        self.stage_a_dir = self.outputs_dir / "stage_a"
        self.stage_b_dir = self.outputs_dir / "stage_b"
        self.stage_c_dir = self.outputs_dir / "stage_c"

        self._preferred_step2_results_dir = self.stage_b_dir / "step2_results"
        self._preferred_step2_logs_dir = self.logs_dir / "stage_b"
        self.stage_c_logs_dir = self.logs_dir / "stage_c"

        self.pdfs_dir = self.inputs_dir / "pdfs"
        self.traffic_dir = self.inputs_dir / "traffic"
        self.packet_raw_dir = self.inputs_dir / "packet_raw"
        self.host_docs_root = self.inputs_dir / "host_docs"

        for path in (
            self.inputs_dir,
            self.outputs_dir,
            self.logs_dir,
            self.cache_dir,
            self.stage_a_dir,
            self.stage_b_dir,
            self.stage_c_dir,
            self._preferred_step2_results_dir,
            self._preferred_step2_logs_dir,
            self.stage_c_logs_dir,
            self.pdfs_dir,
            self.traffic_dir,
            self.packet_raw_dir,
            self.host_docs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # --- Legacy locations kept for backward compatibility -----------------
    @property
    def _legacy_pdf_dir(self) -> Path:
        return self.root / "pdfs"

    @property
    def _legacy_sections_dir(self) -> Path:
        return self.root / "sections"

    @property
    def _legacy_step2_results_dir(self) -> Path:
        return self.root / "step2_results"

    @property
    def _legacy_step2_logs_dir(self) -> Path:
        return self.root / "step2_logs"

    @property
    def _legacy_stage_c_dir(self) -> Path:
        return self.root / "stage_c_outputs"

    @property
    def _legacy_format_tree_file(self) -> Path:
        return self.root / "format_tree.json"

    # --- Helper utilities --------------------------------------------------
    @staticmethod
    def _has_files(path: Path, pattern: str = "*") -> bool:
        try:
            return path.exists() and any(path.glob(pattern))
        except OSError:
            return False

    def _prefer_dir(self, preferred: Path, legacy: Path, pattern: str = "*") -> Path:
        """Pick preferred dir unless legacy already holds useful files."""
        if self._has_files(preferred, pattern):
            return preferred
        if self._has_files(legacy, pattern):
            return legacy
        return preferred

    def _first_existing(self, *candidates: Optional[Path]) -> Optional[Path]:
        for cand in candidates:
            if cand and Path(cand).exists():
                return Path(cand)
        return next((c for c in candidates if c is not None), None)

    @property
    def pdf_dir(self) -> Path:
        return self._prefer_dir(self.pdfs_dir, self._legacy_pdf_dir, pattern="*.pdf")

    @property
    def sections_dir(self) -> Path:
        return self.stage_a_dir

    @property
    def sections_file(self) -> Path:
        return self.sections_override or (self.sections_dir / "document_sections.json")

    @property
    def legacy_sections_file(self) -> Path:
        return self._legacy_sections_dir / "document_sections.json"

    @property
    def raw_sections_file(self) -> Path:
        return self.raw_override or (self.sections_dir / "document_sections_raw.json")

    @property
    def legacy_raw_sections_file(self) -> Path:
        return self._legacy_sections_dir / "document_sections_raw.json"

    @property
    def sections_subset_file(self) -> Path:
        return self.sections_dir / f"{self.root.name}_document_sections_subset.json"

    @property
    def legacy_sections_subset_file(self) -> Path:
        return self._legacy_sections_dir / f"{self.root.name}_document_sections_subset.json"

    @property
    def format_tree_file(self) -> Path:
        return self.stage_b_dir / "format_tree.json"

    @property
    def legacy_format_tree_file(self) -> Path:
        return self._legacy_format_tree_file

    @property
    def step2_results_dir(self) -> Path:
        return self._preferred_step2_results_dir

    @property
    def step2_logs_dir(self) -> Path:
        return self._preferred_step2_logs_dir

    @property
    def stage_c_output_dir(self) -> Path:
        return self.stage_c_dir

    @property
    def traffic_fixed_tree_file(self) -> Path:
        preferred = self.step2_results_dir / "traffic_fixed_tree.json"
        if self.output_variant:
            return preferred
        legacy = self._legacy_step2_results_dir / "traffic_fixed_tree.json"
        existing = self._first_existing(preferred, legacy)
        return existing or preferred

    @property
    def syntax_fixed_tree_file(self) -> Path:
        return self.step2_results_dir / "syntax_fixed_tree.json"

    @property
    def traffic_file(self) -> Path:
        candidates = [
            self.traffic_dir / f"traffic_{self.root.name}.txt",
            self.traffic_dir / "traffic.txt",
            self.root / "traffic.txt",
            self.root / f"traffic_{self.root.name}.txt",
        ]
        found = self._first_existing(*candidates)
        return found or candidates[0]

    def host_docs_dir_for(self, host_name: str) -> Path:
        """Prefer protocol-scoped host docs, fall back to shared host docs."""
        preferred = self.host_docs_root / host_name
        shared = DATA_DIR / "host_docs" / host_name
        if preferred.exists() or not shared.exists():
            return preferred
        return shared

    @property
    def parser_output(self) -> Path:
        return self.root / "generated_parser.py"


@contextmanager
def _record_stage_time(timing: Optional[dict], stage_key: str):
    if timing is None:
        yield
        return
    stages = timing.setdefault("stages", {})
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = float(time.perf_counter() - t0)
        stages[stage_key] = float(stages.get(stage_key, 0.0) or 0.0) + elapsed


def _sum_section_processing_time(sections_path: Path) -> Optional[float]:
    try:
        payload = json.loads(sections_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    sections = payload.get("sections") if isinstance(payload, dict) else payload
    if not isinstance(sections, list):
        return None

    total = 0.0
    seen = 0
    for section in sections:
        if not isinstance(section, dict):
            continue
        cost_info = section.get("cost_info")
        if not isinstance(cost_info, dict):
            continue
        if "processing_time" not in cost_info:
            continue
        try:
            total += float(cost_info["processing_time"])
            seen += 1
        except Exception:
            continue
    return total if seen else None


def _write_timing_report(paths: ProtocolPaths, timing: dict) -> Optional[Path]:
    try:
        out_dir = paths.outputs_dir / "timing"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(timing.get("run_id") or time.strftime("%Y%m%d_%H%M%S"))
        host = str(timing.get("host") or "unknown_host")
        report_path = out_dir / f"pipeline_{run_id}_{host}.json"
        report_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report_path
    except Exception:
        return None


def _find_tshark(ws_build_dir: Path, override: Optional[Path] = None) -> Path:
    """Locate tshark under the Wireshark build/run directory or use an override."""
    if override:
        return override
    run_dir = ws_build_dir / "run"
    candidates = [
        run_dir / "tshark",
        run_dir / "tshark.exe",
        run_dir / "Wireshark.app" / "Contents" / "MacOS" / "tshark",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"tshark not found under {run_dir}")


def _ensure_tshark(
    ws_build_dir: Path,
    *,
    logs_root: Path,
    jobs: int = 4,
) -> Path:
    """Ensure tshark exists under ws_build_dir/run, building it if needed."""
    try:
        return _find_tshark(ws_build_dir)
    except FileNotFoundError:
        pass

    logs_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_root / f"wireshark_build_tshark_{ts}.log"

    cmd = [
        "cmake",
        "--build",
        ".",
        "--target",
        "tshark",
        "--parallel",
        str(int(jobs)),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ws_build_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        log_path.write_text(
            "$ " + " ".join(cmd) + "\n" + (proc.stdout or "") + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    if proc.returncode != 0:
        raise RuntimeError(
            f"tshark not found under {ws_build_dir / 'run'} and building it failed (rc={proc.returncode}). "
            f"See log: {log_path}"
        )

    return _find_tshark(ws_build_dir)


def _load_sections_from_file(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "sections" in payload:
        return list(payload["sections"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"Unrecognised sections format in {path}")


def _write_sections(path: Path, sections: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"sections": list(sections)}, handle, ensure_ascii=False, indent=2)


def _write_raw_sections(path: Path, raw_sections: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"sections": list(raw_sections)}, handle, ensure_ascii=False, indent=2)


def _ensure_no_spec_stub_files(paths: ProtocolPaths) -> Tuple[Path, Path]:
    sections_path = paths.stage_a_dir / "document_sections_no_spec.json"
    raw_path = paths.stage_a_dir / "document_sections_raw_no_spec.json"
    if not sections_path.exists():
        _write_sections(sections_path, [])
    if not raw_path.exists():
        _write_raw_sections(raw_path, [])
    return sections_path, raw_path


def _stage_b_cache_root(paths: ProtocolPaths) -> Path:
    namespace = _stage_b_cache_namespace(paths)
    return STEP2_CACHE_DIR / namespace


def _stage_b_cache_namespace(paths: ProtocolPaths) -> str:
    if paths.output_variant:
        return f"{paths.root.name}/{paths.output_variant}"
    return str(paths.root.name)


def _validate_no_spec_args(args: argparse.Namespace) -> None:
    incompatible: List[str] = []
    if bool(getattr(args, "run_doc_stage", False)):
        incompatible.append("--run-doc-stage")
    if bool(getattr(args, "stage_a_only", False)):
        incompatible.append("--stage-a-only")
    if getattr(args, "sections_file", None) is not None:
        incompatible.append("--sections-file")
    if getattr(args, "raw_sections_file", None) is not None:
        incompatible.append("--raw-sections-file")
    if getattr(args, "sections_subset", None) is not None:
        incompatible.append("--sections-subset")
    if bool(getattr(args, "stage_c_doc_summary", False)):
        incompatible.append("--stage-c-doc-summary")
    if getattr(args, "stage_c_doc_summary_source", None) is not None:
        incompatible.append("--stage-c-doc-summary-source")
    if incompatible:
        joined = ", ".join(incompatible)
        raise SystemExit(f"--ablation-no-spec cannot be combined with: {joined}")


def _resolve_base_stage_a_sections(protocol_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    base_paths = ProtocolPaths(protocol_dir)
    sections_candidate = base_paths._first_existing(
        base_paths.outputs_dir / "stage_a" / "document_sections.json",
        protocol_dir / "outputs" / "a_bak" / "document_sections.json",
        base_paths.legacy_sections_file,
    )
    raw_candidate = base_paths._first_existing(
        base_paths.outputs_dir / "stage_a" / "document_sections_raw.json",
        protocol_dir / "outputs" / "a_bak" / "document_sections_raw.json",
        base_paths.legacy_raw_sections_file,
    )
    sections_ok = sections_candidate if sections_candidate and sections_candidate.exists() else None
    raw_ok = raw_candidate if raw_candidate and raw_candidate.exists() else None
    return sections_ok, raw_ok


def _seed_protocol_name_from_dataset(payload: Any, dataset_name: str) -> Any:
    """Fill protocol_name when the builder left its default.

    The initial-tree prompt only receives a protocol hint under --ablation-no-spec,
    so the tree otherwise carries "Unknown Protocol". The dataset directory is <protocol>[_variant...],
    which is the same convention the worker path uses.
    """
    base = str(dataset_name or "").split("_")[0].strip().lower()
    if not base:
        return payload
    tree = payload.get("protocol_tree") if isinstance(payload, dict) and isinstance(payload.get("protocol_tree"), dict) else payload
    if isinstance(tree, dict) and str(tree.get("protocol_name") or "").strip() in {"", "Unknown Protocol"}:
        tree["protocol_name"] = base
    return payload


def _normalize_protocol_tree_payload(payload: Any) -> Any:
    def normalize_tree(tree: Any) -> Any:
        if not isinstance(tree, dict):
            return tree
        tree = copy.deepcopy(tree)
        nodes = tree.get("nodes")
        if isinstance(nodes, dict):
            tree["nodes"] = list(nodes.values())
        edges = tree.get("edges")
        if isinstance(edges, dict):
            tree["edges"] = list(edges.values())
        for node in tree.get("nodes", []):
            if isinstance(node, dict):
                children = node.get("children_ids")
                if isinstance(children, dict):
                    node["children_ids"] = list(children.values())
        return tree

    if isinstance(payload, dict):
        payload = copy.deepcopy(payload)
        if "protocol_tree" in payload:
            payload["protocol_tree"] = normalize_tree(payload["protocol_tree"])
        else:
            payload = normalize_tree(payload)
    return payload


def run_stage_a(paths: ProtocolPaths, pdfs: Sequence[Path], claude_key: str, deepseek_key: Optional[str]) -> None:
    print("\n[Stage A] Document extraction")
    # Lazy import: Stage A requires optional deps (e.g., anthropic), but Stage B/C should run
    # on prebuilt datasets without forcing Stage A deps to be installed.
    from pargen.specification_processing.pipeline import DocumentPipeline, DocumentPipelineConfig

    config = DocumentPipelineConfig(
        claude_api_key=claude_key,
        deepseek_api_key=deepseek_key or None,
        target_protocol=paths.root.name,
    )
    pipeline = DocumentPipeline(config)

    aggregated_sections: List[dict] = []
    aggregated_raw: List[dict] = []

    for pdf in pdfs:
        print(f"  Processing PDF: {pdf}")
        result = pipeline.process_pdf(pdf)
        if not isinstance(result, dict) or not bool(result.get("success")):
            err = None
            if isinstance(result, dict):
                err = result.get("error") or result.get("message")
            raise SystemExit(f"Stage A failed for {pdf}: {err or result}")

        outputs = result.get("output_files") or {}
        llm_results = outputs.get("llm_results")
        raw_data = outputs.get("raw_data")
        if not llm_results or not raw_data:
            raise SystemExit(f"Stage A did not produce output files for {pdf} (output_files={list(outputs.keys())}).")

        llm_path = Path(str(llm_results))
        raw_path = Path(str(raw_data))
        if not llm_path.exists():
            raise SystemExit(f"Stage A expected llm_results at {llm_path} but file is missing.")
        if not raw_path.exists():
            raise SystemExit(f"Stage A expected raw_data at {raw_path} but file is missing.")
        aggregated_sections.extend(_load_sections_from_file(llm_path))
        aggregated_raw.extend(_load_sections_from_file(raw_path))

    _write_sections(paths.sections_file, aggregated_sections)
    _write_raw_sections(paths.raw_sections_file, aggregated_raw)
    print(f"  Saved merged sections to {paths.sections_file}")
    print(f"  Saved merged raw sections to {paths.raw_sections_file}")


def _load_cached_format_tree(
    cache_path: Path,
    target_path: Path,
    cache_copy: Optional[Path] = None,
    expected_signature: Optional[dict[str, Any]] = None,
) -> bool:
    try:
        cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  Failed to load cached tree ({cache_path}): {exc}. Rebuilding...")
        return False

    if expected_signature is not None:
        expected = str(expected_signature.get("sha256") or "")
        actual_sig = cached_payload.get("input_signature") if isinstance(cached_payload, dict) else None
        actual = str((actual_sig or {}).get("sha256") or "")
        if actual != expected:
            allow_unsigned = os.getenv("PARGEN_STAGE_B_ALLOW_UNSIGNED_CACHE", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not allow_unsigned:
                print(
                    "  Cached Stage B tree input signature mismatch "
                    f"(expected={expected[:12]}, actual={actual[:12] if actual else 'missing'}). Rebuilding..."
                )
                return False
            print("  WARNING: reusing unsigned/stale Stage B cache because PARGEN_STAGE_B_ALLOW_UNSIGNED_CACHE=1")

    protocol_tree_payload = cached_payload.get("protocol_tree") if isinstance(cached_payload, dict) else cached_payload
    protocol_tree_payload = _normalize_protocol_tree_payload(protocol_tree_payload)

    if not isinstance(protocol_tree_payload, dict) or not protocol_tree_payload.get("nodes"):
        print("  Cached tree invalid (missing nodes). Rebuilding Stage B...")
        return False
    if not protocol_tree_payload.get("edges"):
        print("  Cached tree missing edges. Rebuilding Stage B...")
        return False

    final_payload: dict[str, Any] = {"protocol_tree": protocol_tree_payload}
    if isinstance(cached_payload, dict):
        for key in ("protocol_name", "generated_at", "input_signature", "validation"):
            if key in cached_payload:
                final_payload[key] = cached_payload[key]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Also keep a cache-local copy to avoid overwriting the generation output downstream.
    if cache_copy is None:
        cache_copy = cache_path.parent / "format_tree.json"
    cache_copy.parent.mkdir(parents=True, exist_ok=True)
    cache_copy.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[Stage B] Format tree construction (cached)")
    print(f"  Loaded format tree from cache: {cache_path}")
    return True


def _stage_b_llm_options(args: Optional[argparse.Namespace]) -> dict[str, Any]:
    explicit_initial_provider = getattr(args, "stage_b_initial_provider", None)
    explicit_initial_model = getattr(args, "stage_b_initial_model", None)
    explicit_initial_temperature = getattr(args, "stage_b_initial_temperature", None)
    explicit_initial_max_tokens = getattr(args, "stage_b_initial_max_tokens", None)

    default_claude_model = os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
    default_deepseek_model = os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    stage_b_refine_provider = getattr(args, "stage_b_provider", None) or DEFAULT_PROVIDER
    stage_b_refine_provider_key = (stage_b_refine_provider or "").strip().lower()
    stage_b_refine_model = getattr(args, "stage_b_model", None)
    if stage_b_refine_model is None:
        if stage_b_refine_provider_key in {"anthropic", "claude"}:
            stage_b_refine_model = default_claude_model
        elif stage_b_refine_provider_key == "deepseek":
            stage_b_refine_model = default_deepseek_model
    stage_b_refine_temperature = getattr(args, "stage_b_temperature", None)
    stage_b_refine_max_tokens = getattr(args, "stage_b_max_tokens", None)
    stage_b_initial_provider = explicit_initial_provider or DEFAULT_PROVIDER
    initial_matches_refine = (
        (stage_b_initial_provider or "").strip().lower()
        == (stage_b_refine_provider or "").strip().lower()
    )
    stage_b_initial_model = (
        explicit_initial_model
        if explicit_initial_model is not None
        else (stage_b_refine_model if initial_matches_refine else None)
    )
    stage_b_initial_temperature = (
        explicit_initial_temperature
        if explicit_initial_temperature is not None
        else (stage_b_refine_temperature if initial_matches_refine else None)
    )
    stage_b_initial_max_tokens = (
        explicit_initial_max_tokens
        if explicit_initial_max_tokens is not None
        else (stage_b_refine_max_tokens if initial_matches_refine else None)
    )
    return {
        "initial_provider": stage_b_initial_provider,
        "initial_model": stage_b_initial_model,
        "initial_temperature": stage_b_initial_temperature,
        "initial_max_tokens": stage_b_initial_max_tokens,
        "refine_provider": stage_b_refine_provider,
        "refine_model": stage_b_refine_model,
        "refine_temperature": stage_b_refine_temperature,
        "refine_max_tokens": stage_b_refine_max_tokens,
    }


def _stage_b_traffic_file_for_signature(paths: ProtocolPaths, args: Optional[argparse.Namespace]) -> Optional[Path]:
    if bool(getattr(args, "ablation_no_traffic", False)):
        return None
    traffic = getattr(args, "traffic", None) or paths.traffic_file
    traffic_path = Path(traffic) if traffic is not None else None
    return traffic_path if traffic_path and traffic_path.exists() else None


_STAGE_B_ARTIFACT_SIGNATURE_SCHEMA = 1


def _sha256_file_for_signature(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _exp_constraint_mapping_matches_tree(mapping_path: Optional[Path], tree_path: Optional[Path]) -> tuple[bool, str]:
    if mapping_path is None or tree_path is None:
        return True, ""
    mapping = Path(mapping_path)
    tree = Path(tree_path)
    if not mapping.exists() or not tree.exists():
        return True, ""
    try:
        payload = json.loads(mapping.read_text(encoding="utf-8"))
    except Exception as exc:
        return True, f"mapping unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return True, ""

    expected_sha = _sha256_file_for_signature(tree)
    recorded_sha = payload.get("tree_sha256")
    if isinstance(recorded_sha, str) and recorded_sha.strip() and expected_sha:
        if recorded_sha.strip() != expected_sha:
            return False, f"mapping tree_sha256={recorded_sha.strip()[:12]} current={expected_sha[:12]}"
        return True, ""

    recorded_tree = payload.get("tree")
    if not isinstance(recorded_tree, str) or not recorded_tree.strip():
        return True, ""

    candidates = [Path(recorded_tree)]
    if not Path(recorded_tree).is_absolute():
        candidates.append(PARENT / recorded_tree)
        candidates.append(mapping.parent / recorded_tree)

    current_resolved = tree.resolve()
    for cand in candidates:
        try:
            if cand.exists() and cand.resolve() == current_resolved:
                return True, ""
        except Exception:
            pass

    for cand in candidates:
        recorded_tree_sha = _sha256_file_for_signature(cand)
        if recorded_tree_sha and expected_sha:
            if recorded_tree_sha == expected_sha:
                return True, ""
            return False, f"mapping tree={recorded_tree} sha={recorded_tree_sha[:12]} current={expected_sha[:12]}"

    return False, f"mapping tree={recorded_tree} does not match current tree={tree}"


def _stable_signature_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_b_artifact_meta_path(tree_path: Path) -> Path:
    return tree_path.with_name(f"{tree_path.stem}.meta.json")


def _stage_b_fix_sections_path(paths: ProtocolPaths, args: Optional[argparse.Namespace]) -> Optional[Path]:
    subset = getattr(args, "sections_subset", None)
    if subset is not None and Path(subset).exists():
        return Path(subset)
    if paths.sections_file.exists():
        return paths.sections_file
    return None


def _stage_b_active_scope_sections_path(paths: ProtocolPaths, sections_path: Optional[Path]) -> Optional[Path]:
    if sections_path is None:
        return None
    if str(os.getenv("STEP2_STAGEB_ACTIVE_SCOPE_SECTIONS", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return None
    scope_plan_path = _stage_b_cache_root(paths) / "traffic_locate_scope_plan.json"
    if not scope_plan_path.exists():
        return None
    try:
        scope_plan = json.loads(scope_plan_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(scope_plan, dict):
        return None
    active_indices_raw = scope_plan.get("active_core_indices")
    if not isinstance(active_indices_raw, list):
        return None
    active_indices: List[int] = []
    seen: set[int] = set()
    for value in active_indices_raw:
        try:
            idx = int(value)
        except Exception:
            return None
        if idx in seen:
            continue
        seen.add(idx)
        active_indices.append(idx)
    sections = _load_stage_b_sections_for_scaffold(sections_path)
    if not sections:
        return None
    if not active_indices or any(idx < 0 or idx >= len(sections) for idx in active_indices):
        return None
    if len(active_indices) >= len(sections):
        return None
    selected_sections = [sections[idx] for idx in active_indices]
    output_path = paths.stage_b_dir / "active_core_sections_for_stage_b_fix.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "sections": selected_sections,
                "scope": "active_core_only",
                "source_sections_path": str(sections_path),
                "traffic_locate_scope_plan": str(scope_plan_path),
                "active_core_indices": active_indices,
                "coverage_only_indices": scope_plan.get("coverage_only_indices") or [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def _stage_b_effective_fix_sections_path(paths: ProtocolPaths, args: Optional[argparse.Namespace]) -> Optional[Path]:
    base_path = _stage_b_fix_sections_path(paths, args)
    scoped_path = _stage_b_active_scope_sections_path(paths, base_path)
    return scoped_path or base_path


def _stage_b_int_arg(args: Optional[argparse.Namespace], name: str, default: int) -> int:
    try:
        return int(getattr(args, name, default))
    except Exception:
        return int(default)


def _stage_b_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return int(default)


def _build_stage_b_expected_signature(
    paths: ProtocolPaths,
    args: Optional[argparse.Namespace],
    *,
    traffic_file: Optional[Path] = None,
) -> dict[str, Any]:
    from pargen.protocol_knowledge.construction.step2_format_graph_builder import (
        GenerationLLMAgent,
        build_stage2_input_signature,
        _infer_step2_cache_namespace,
    )

    opts = _stage_b_llm_options(args)
    initial_agent = GenerationLLMAgent(
        provider=opts["initial_provider"],
        model=opts["initial_model"],
        temperature=opts["initial_temperature"],
        max_tokens=opts["initial_max_tokens"],
    )
    refine_agent = GenerationLLMAgent(
        provider=opts["refine_provider"],
        model=opts["refine_model"],
        temperature=opts["refine_temperature"],
        max_tokens=opts["refine_max_tokens"],
    )
    cache_namespace = (
        _stage_b_cache_namespace(paths)
        if paths.output_variant
        else _infer_step2_cache_namespace(str(paths.sections_file), str(paths.raw_sections_file))
    )
    return build_stage2_input_signature(
        sections_file=paths.sections_file,
        raw_file=paths.raw_sections_file,
        traffic_file=traffic_file,
        protocol_hint=(paths.root.name if bool(getattr(args, "ablation_no_spec", False)) else None),
        cache_namespace=cache_namespace,
        initial_llm=initial_agent.signature_config(),
        refine_llm=refine_agent.signature_config(),
    )


def _build_stage_b_artifact_signature(
    paths: ProtocolPaths,
    args: Optional[argparse.Namespace],
    *,
    artifact_kind: str,
    traffic_file: Optional[Path] = None,
) -> dict[str, Any]:
    from pargen.protocol_knowledge.semantic_integrity import REFINEMENT_CONTRACT_VERSION

    traffic_file = traffic_file if traffic_file is not None else _stage_b_traffic_file_for_signature(paths, args)
    stage2_signature = _build_stage_b_expected_signature(paths, args, traffic_file=traffic_file)
    fix_sections_path = _stage_b_effective_fix_sections_path(paths, args)
    target_message_type = str(getattr(args, "target_message_type", "") or "").strip().lower()
    payload: dict[str, Any] = {
        "schema": _STAGE_B_ARTIFACT_SIGNATURE_SCHEMA,
        "artifact_kind": str(artifact_kind or ""),
        "refinement_contract": REFINEMENT_CONTRACT_VERSION,
        "stage2_input_signature": stage2_signature,
        "files": {
            "sections_sha256": _sha256_file_for_signature(paths.sections_file),
            "raw_sha256": _sha256_file_for_signature(paths.raw_sections_file),
            "fix_sections_sha256": _sha256_file_for_signature(fix_sections_path),
            "traffic_sha256": _sha256_file_for_signature(traffic_file),
        },
        "stage_b2": {
            "enable_fix_stage": bool(getattr(args, "enable_fix_stage", False)),
            "fix_provider": str(getattr(args, "fix_provider", "") or ""),
            "fix_max_llm_calls": _stage_b_int_arg(args, "fix_max_llm_calls", 0),
        },
        "stage_b3": {
            "enable_semantic_fix": bool(getattr(args, "enable_semantic_fix", False)),
            "ablation_no_traffic": bool(getattr(args, "ablation_no_traffic", False)),
            "semantic_max_packets": _stage_b_int_arg(args, "semantic_max_packets", 0),
            "target_message_type": target_message_type,
            "traffic_max_llm_calls": _stage_b_env_int("STEP2_TRAFFIC_MAX_LLM_CALLS", 20),
            "traffic_accept_max_failed_increase": _stage_b_env_int(
                "STEP2_TRAFFIC_ACCEPT_MAX_FAILED_INCREASE", 0
            ),
            "traffic_mcts_rounds": os.getenv("STEP2_TRAFFIC_MCTS_ROUNDS", ""),
            "traffic_section_batch_size": os.getenv("STEP2_TRAFFIC_SECTION_BATCH_SIZE", ""),
        },
    }
    return {
        "schema": _STAGE_B_ARTIFACT_SIGNATURE_SCHEMA,
        "sha256": _stable_signature_hash(payload),
        "payload": payload,
    }


def _write_stage_b_artifact_metadata(
    tree_path: Path,
    *,
    input_signature: dict[str, Any],
) -> Path:
    meta_path = _stage_b_artifact_meta_path(tree_path)
    metadata = {
        "schema": _STAGE_B_ARTIFACT_SIGNATURE_SCHEMA,
        "artifact": {
            "path": str(tree_path),
            "sha256": _sha256_file_for_signature(tree_path),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "input_signature": input_signature,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def _load_stage_b_sections_for_scaffold(sections_path: Optional[Path]) -> list[dict[str, Any]]:
    if sections_path is None:
        return []
    try:
        payload = json.loads(Path(sections_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [section for section in payload if isinstance(section, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sections"), list):
        return [section for section in payload["sections"] if isinstance(section, dict)]
    return []


def _write_stage_b_coverage_only_scaffold_artifact(
    tree_path: Path,
    *,
    paths: ProtocolPaths,
    sections_path: Optional[Path],
) -> dict[str, Any]:
    """Write a coverage-only scaffold tree without replacing the active Stage B artifact."""

    summary: dict[str, Any] = {
        "applied": False,
        "reason": None,
        "tree_path": str(tree_path),
        "sections_path": str(sections_path) if sections_path is not None else None,
        "coverage_tree_path": None,
        "coverage_sections_path": None,
        "coverage_only_indices": [],
        "changes_count": 0,
        "before_counts": None,
        "after_counts": None,
    }
    if str(os.getenv("STEP2_STAGEB_COVERAGE_ONLY_ARTIFACT", "1")).strip().lower() in {"0", "false", "no", "off"}:
        summary["reason"] = "disabled"
        return summary

    scope_plan_path = _stage_b_cache_root(paths) / "traffic_locate_scope_plan.json"
    no_scope_fallback = False
    if not scope_plan_path.exists():
        if _env_bool("STEP2_STAGEB_COVERAGE_OBLIGATION", False) and _env_bool(
            "STEP2_STAGEB_COVERAGE_ONLY_ALLOW_FULL_FALLBACK",
            True,
        ):
            scope_plan = {}
            no_scope_fallback = True
        else:
            summary["reason"] = "scope_plan_missing"
            return summary
    else:
        try:
            scope_plan = json.loads(scope_plan_path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary["reason"] = f"scope_plan_unreadable:{type(exc).__name__}"
            return summary
        if not isinstance(scope_plan, dict):
            summary["reason"] = "scope_plan_not_object"
            return summary

    sections = _load_stage_b_sections_for_scaffold(sections_path)
    if not sections:
        summary["reason"] = "sections_missing_or_empty"
        return summary
    active_indices: List[int] = []
    active_seen: set[int] = set()
    for value in scope_plan.get("active_core_indices") or []:
        try:
            idx = int(value)
        except Exception:
            continue
        if idx in active_seen or idx < 0 or idx >= len(sections):
            continue
        active_seen.add(idx)
        active_indices.append(idx)
    locator_coverage_indices: List[int] = []
    locator_seen: set[int] = set()
    for value in scope_plan.get("coverage_only_indices") or []:
        try:
            idx = int(value)
        except Exception:
            continue
        if idx in locator_seen or idx < 0 or idx >= len(sections):
            continue
        locator_seen.add(idx)
        locator_coverage_indices.append(idx)
    mode = str(os.getenv("STEP2_STAGEB_COVERAGE_ONLY_MODE", "non_active_fixed_subset")).strip().lower()
    if no_scope_fallback:
        coverage_indices = list(range(len(sections)))
        mode = "full_fixed_subset_no_scope"
    elif mode in {"explicit", "locator", "traffic_locator"}:
        coverage_indices = list(locator_coverage_indices)
    elif active_indices:
        coverage_indices = [idx for idx in range(len(sections)) if idx not in active_seen]
    else:
        coverage_indices = list(locator_coverage_indices)
    summary["coverage_only_indices"] = coverage_indices
    summary["locator_coverage_only_indices"] = locator_coverage_indices
    summary["active_core_indices"] = active_indices
    summary["coverage_only_mode"] = mode
    if not coverage_indices:
        summary["reason"] = "coverage_only_empty"
        return summary

    path = Path(tree_path)
    if not path.exists():
        summary["reason"] = "tree_missing"
        return summary
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary["reason"] = f"tree_unreadable:{type(exc).__name__}"
        return summary
    tree_payload = payload.get("protocol_tree") if isinstance(payload, dict) and isinstance(payload.get("protocol_tree"), dict) else payload
    if not isinstance(tree_payload, dict):
        summary["reason"] = "tree_payload_not_object"
        return summary

    coverage_sections = [sections[idx] for idx in coverage_indices]
    coverage_dir = paths.stage_b_dir / "coverage_only"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    coverage_sections_path = coverage_dir / "coverage_only_sections.json"
    coverage_sections_path.write_text(
        json.dumps(
            {
                "sections": coverage_sections,
                "scope": "coverage_only",
                "source_sections_path": str(sections_path) if sections_path is not None else None,
                "traffic_locate_scope_plan": str(scope_plan_path),
                "active_core_indices": active_indices,
                "locator_coverage_only_indices": locator_coverage_indices,
                "coverage_only_indices": coverage_indices,
                "coverage_only_mode": mode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    from pargen.protocol_knowledge.construction.step2_format_graph_builder import (
        _source_structure_gap_counts,
    )

    baseline_tree = copy.deepcopy(tree_payload)
    before_counts = _source_structure_gap_counts(baseline_tree, coverage_sections)
    coverage_tree, changes = baseline_tree, []
    after_counts = _source_structure_gap_counts(coverage_tree, coverage_sections)
    coverage_tree_path = coverage_dir / "coverage_scaffold_tree.json"
    if isinstance(payload, dict) and isinstance(payload.get("protocol_tree"), dict):
        output_payload = copy.deepcopy(payload)
        output_payload["protocol_tree"] = coverage_tree
    else:
        output_payload = coverage_tree
    coverage_tree_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary.update(
        {
            "applied": True,
            "reason": "ok",
            "coverage_tree_path": str(coverage_tree_path),
            "coverage_sections_path": str(coverage_sections_path),
            "changes_count": len(changes),
            "before_counts": list(before_counts),
            "after_counts": list(after_counts),
            "changes_sample": [str(change) for change in changes[:50]],
            "active_tree_unchanged": True,
        }
    )
    (coverage_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _load_protocol_tree_payload(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("protocol_tree"), dict):
        return payload["protocol_tree"]
    if isinstance(payload, dict):
        return payload
    return None


def _coverage_obligation_strip_synthetic_obligation_sections(
    sections: Sequence[Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        if section.get("doc_cluster_obligations") or section.get("constraint_fn_obligations"):
            continue
        out.append(dict(section))
    return out


def _run_stage_b_coverage_obligation_driver(
    tree_path: Path,
    *,
    paths: ProtocolPaths,
    coverage_scaffold_summary: Mapping[str, Any],
    traffic_path: Optional[Path],
    provider: str,
) -> dict[str, Any]:
    """Repair document obligations on a separate coverage tree.
    
    Use the Step2 fixer and promote the candidate when promotion is enabled
    and the observed traffic anchor is preserved."""

    coverage_dir = paths.stage_b_dir / "coverage_only"
    obligation_dir = coverage_dir / "obligation_driver"
    summary_path = coverage_dir / "obligation_driver_summary.json"
    summary: dict[str, Any] = {
        "attempted": False,
        "accepted": False,
        "promoted": False,
        "reason": None,
        "tree_path": str(tree_path),
        "coverage_scaffold_tree_path": coverage_scaffold_summary.get("coverage_tree_path"),
        "coverage_sections_path": coverage_scaffold_summary.get("coverage_sections_path"),
        "coverage_obligation_tree_path": None,
        "max_llm_calls": None,
        "anchor_gate": None,
        "active_tree_unchanged": True,
    }
    if not _env_bool("STEP2_STAGEB_COVERAGE_OBLIGATION", False):
        summary["reason"] = "disabled"
        return summary
    max_llm_calls = _env_int("STEP2_STAGEB_COVERAGE_OBLIGATION_MAX_CALLS", 0)
    summary["max_llm_calls"] = max_llm_calls
    if max_llm_calls <= 0:
        summary["reason"] = "max_llm_calls_zero"
        return summary
    if not coverage_scaffold_summary.get("applied"):
        summary["reason"] = f"coverage_scaffold_not_applied:{coverage_scaffold_summary.get('reason')}"
        return summary
    coverage_tree_path = Path(str(coverage_scaffold_summary.get("coverage_tree_path") or ""))
    coverage_sections_path = Path(str(coverage_scaffold_summary.get("coverage_sections_path") or ""))
    if not coverage_tree_path.exists():
        summary["reason"] = "coverage_tree_missing"
        return summary
    if not coverage_sections_path.exists():
        summary["reason"] = "coverage_sections_missing"
        return summary

    obligation_dir.mkdir(parents=True, exist_ok=True)
    return _run_stage_b_llm_source_review(
        Path(tree_path),
        summary=summary,
        summary_path=summary_path,
        coverage_dir=coverage_dir,
        coverage_tree_path=coverage_tree_path,
        coverage_sections_path=coverage_sections_path,
        traffic_path=traffic_path,
        provider=provider,
    )


def _promote_stage_b_coverage_tree(
    tree_path: Path,
    coverage_obligation_tree_path: Path,
    summary: dict[str, Any],
    *,
    driver_meta: Mapping[str, Any],
) -> None:
    """Replace the active Stage-B tree with the coverage pass result and rebind its receipts."""

    shutil.copy2(coverage_obligation_tree_path, tree_path)
    summary["promoted"] = True
    summary["active_tree_unchanged"] = False
    meta_path = _stage_b_artifact_meta_path(Path(tree_path))
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        artifact = metadata.setdefault("artifact", {})
        if isinstance(artifact, dict):
            artifact["path"] = str(tree_path)
            artifact["sha256"] = _sha256_file_for_signature(Path(tree_path))
            artifact["coverage_obligation_promoted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        metadata["coverage_obligation_driver"] = dict(driver_meta)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # The published status binds its label to the artifact hash; after the
    # copy it still named the B3 tree, so consumers read the promoted tree
    # as a checksum mismatch. Rebind it and record what was replaced.
    from pargen.protocol_knowledge.refinement_result import refinement_status_path
    status_path = refinement_status_path(Path(tree_path))
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["coverage_obligation_promoted"] = {
                "replaced_sha256": status.get("artifact_sha256"),
                "coverage_obligation_tree_path": str(coverage_obligation_tree_path),
                "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            status["artifact_sha256"] = _sha256_file_for_signature(Path(tree_path))
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"[Stage B] Could not rebind refinement status after promotion: {exc}")


def _run_stage_b_llm_source_review(
    tree_path: Path,
    *,
    summary: dict[str, Any],
    summary_path: Path,
    coverage_dir: Path,
    coverage_tree_path: Path,
    coverage_sections_path: Path,
    traffic_path: Optional[Path],
    provider: str,
) -> dict[str, Any]:
    """Source review: an LLM reads each specification section against the PKG and proposes the missing
    content; a proposal is kept only if it removes nothing and adds no error."""

    from pargen.protocol_knowledge.repair.source_review import ReviewLLM, load_sections, run_source_review

    review_dir = coverage_dir / "source_review"
    tree = json.loads(coverage_tree_path.read_text(encoding="utf-8"))
    sections = load_sections(coverage_sections_path)
    llm = ReviewLLM(
        log_path=review_dir / "calls.jsonl",
        provider=provider,
        model=os.getenv("STEP2_SOURCE_REVIEW_MODEL") or None,
        max_calls=_env_int("STEP2_SOURCE_REVIEW_MAX_CALLS", 40),
        budget_usd=float(os.getenv("STEP2_SOURCE_REVIEW_BUDGET_USD", "1.0") or 1.0),
    )
    protocol = str((_load_protocol_tree_payload(coverage_tree_path) or {}).get("protocol_name") or "")
    reviewed, review = run_source_review(
        tree,
        sections,
        llm=llm,
        protocol=protocol,
        traffic_path=Path(traffic_path) if traffic_path else None,
    )
    review_summary_path = review_dir / "summary.json"
    review_summary_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary.update(
        attempted=True,
        detector="llm_judge",
        source_review_summary_path=str(review_summary_path),
        source_review={"counts": review.get("counts"), "stop_reason": review.get("stop_reason"), "llm": review.get("llm")},
    )
    if not review.get("resolved"):
        summary["reason"] = f"source_review_no_accepted_patch:{review.get('stop_reason')}"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary
    coverage_obligation_tree_path = coverage_dir / "coverage_obligation_tree.json"
    coverage_obligation_tree_path.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary.update(accepted=True, reason="ok", coverage_obligation_tree_path=str(coverage_obligation_tree_path))
    if _env_bool("STEP2_STAGEB_COVERAGE_OBLIGATION_PROMOTE", False):
        _promote_stage_b_coverage_tree(
            Path(tree_path),
            coverage_obligation_tree_path,
            summary,
            driver_meta={
                "detector": "llm_judge",
                "coverage_obligation_tree_path": str(coverage_obligation_tree_path),
                "coverage_sections_path": str(coverage_sections_path),
                "source_review_summary_path": str(review_summary_path),
            },
        )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _stage_b_artifact_signature_matches(tree_path: Path, expected_signature: dict[str, Any]) -> Tuple[bool, str]:
    meta_path = _stage_b_artifact_meta_path(tree_path)
    if not meta_path.exists():
        return False, f"metadata missing ({meta_path.name})"
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"metadata unreadable ({exc})"
    actual_sig = metadata.get("input_signature") if isinstance(metadata, dict) else None
    expected = str(expected_signature.get("sha256") or "")
    actual = str((actual_sig or {}).get("sha256") or "")
    if actual != expected:
        return False, f"signature mismatch (expected={expected[:12]} actual={actual[:12] if actual else 'missing'})"
    recorded_artifact = metadata.get("artifact") if isinstance(metadata, dict) else None
    recorded_sha = str((recorded_artifact or {}).get("sha256") or "")
    current_sha = _sha256_file_for_signature(tree_path) or ""
    if recorded_sha and recorded_sha != current_sha:
        return False, "artifact hash mismatch"
    return True, "signature matched"


def run_stage_b(
    paths: ProtocolPaths,
    force_rebuild: bool = False,
    args: Optional[argparse.Namespace] = None,
) -> Path:
    cache_root = _stage_b_cache_root(paths)
    cache_path = cache_root / "final_complete_protocol_tree.json"
    stage_b_opts = _stage_b_llm_options(args)
    traffic_for_signature = _stage_b_traffic_file_for_signature(paths, args)
    expected_signature = _build_stage_b_expected_signature(paths, args, traffic_file=traffic_for_signature)
    if not force_rebuild and cache_path.exists():
        if _load_cached_format_tree(
            cache_path,
            paths.format_tree_file,
            cache_copy=cache_root / "format_tree.json",
            expected_signature=expected_signature,
        ):
            return cache_path

    print("\n[Stage B] Format tree construction")
    from pargen.protocol_knowledge.pipeline import FormatTreeConfig, FormatTreePipeline

    config = FormatTreeConfig(
        sections_file=paths.sections_file,
        raw_file=paths.raw_sections_file,
        traffic_file=traffic_for_signature,
        protocol_hint=(paths.root.name if bool(getattr(args, "ablation_no_spec", False)) else None),
        cache_namespace=(_stage_b_cache_namespace(paths) if paths.output_variant else None),
        initial_llm_provider=stage_b_opts["initial_provider"],
        initial_llm_model=stage_b_opts["initial_model"],
        initial_llm_temperature=stage_b_opts["initial_temperature"],
        initial_llm_max_tokens=stage_b_opts["initial_max_tokens"],
        refine_llm_provider=stage_b_opts["refine_provider"],
        refine_llm_model=stage_b_opts["refine_model"],
        refine_llm_temperature=stage_b_opts["refine_temperature"],
        refine_llm_max_tokens=stage_b_opts["refine_max_tokens"],
    )
    pipeline = FormatTreePipeline(config)
    tree = pipeline.build()

    if tree is None:
        raise RuntimeError("Format tree generation failed.")

    if hasattr(tree, "to_dict"):
        payload = tree.to_dict()
    elif hasattr(tree, "__dict__"):
        payload = json.loads(json.dumps(tree, default=lambda o: getattr(o, "__dict__", str(o))))
    else:
        payload = tree

    payload = _normalize_protocol_tree_payload(payload)
    payload = _seed_protocol_name_from_dataset(payload, paths.root.name)

    paths.format_tree_file.parent.mkdir(parents=True, exist_ok=True)
    paths.format_tree_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved format tree to {paths.format_tree_file}")

    # Also cache a copy under step2_cache to preserve the generation output separately.
    cache_format_path = cache_root / "format_tree.json"
    cache_format_path.parent.mkdir(parents=True, exist_ok=True)
    cache_format_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Cached format tree copy: {cache_format_path}")

    protocol_tree_payload = payload.get("protocol_tree") if isinstance(payload, dict) and "protocol_tree" in payload else payload
    cache_payload = {
        "protocol_tree": protocol_tree_payload,
        "protocol_name": payload.get("protocol_name") or protocol_tree_payload.get("name") if isinstance(protocol_tree_payload, dict) else "unknown",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_signature": expected_signature,
    }
    if isinstance(payload, dict) and "validation" in payload:
        cache_payload["validation"] = payload["validation"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Cached tree for fixer: {cache_path}")
    return cache_path


def _load_doc_summaries_from_sections(sections_path: Path, *, max_chars: int = 20000) -> str:
    if max_chars <= 0:
        return ""
    try:
        payload = json.loads(sections_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""

    # Handle both formats: list[...] or {"sections": [...]}
    sections: list[dict] = []
    if isinstance(payload, list):
        sections = [sec for sec in payload if isinstance(sec, dict)]
    elif isinstance(payload, dict):
        raw = payload.get("sections")
        if isinstance(raw, list):
            sections = [sec for sec in raw if isinstance(sec, dict)]

    out_lines: list[str] = [f"[DocSummary] source={sections_path.name}"]
    used = 0
    for sec in sections:
        number = str(sec.get("number") or "").strip()
        title = str(sec.get("title") or "").strip()
        summary = str(sec.get("summary") or "").strip()
        if not summary:
            continue
        header = f"\n[Section {number}] {title}".strip()
        block = header + "\n" + summary + "\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 0:
                out_lines.append(block[:remaining])
            break
        out_lines.append(block)
        used += len(block)
    return "\n".join(out_lines).strip()


def run_stage_c(
    format_tree_path: Path,
    paths: ProtocolPaths,
    args: argparse.Namespace,
    *,
    timing: Optional[dict] = None,
    doc_summary_text: Optional[str] = None,
) -> None:
    doc_summary_enabled = bool(str(doc_summary_text or "").strip())
    if format_tree_path is None and not doc_summary_enabled:
        raise SystemExit(
            "No format tree available for Stage C. Please run Stage B or provide --reuse-b-stage-output with an existing traffic_fixed_tree.json."
        )

    output_dir = args.stage_c_output_dir or paths.stage_c_output_dir
    plugin_dir = output_dir / "plugin" / args.stage_c_host
    report_path = output_dir / "stage_c_summary.json"

    codegen_profile_path = args.stage_c_codegen_profile
    if codegen_profile_path is None:
        host_candidate = paths.inputs_dir / f"stage_c_codegen_profile_{args.stage_c_host}.json"
        if host_candidate.exists():
            codegen_profile_path = host_candidate
        else:
            candidate = paths.inputs_dir / "stage_c_codegen_profile.json"
            if candidate.exists():
                codegen_profile_path = candidate

    reuse_stage_c = bool(getattr(args, "reuse_c_stage_output", False))
    requested_constraints_mode = "codegen"
    effective_constraints_mode = requested_constraints_mode
    host_docs_dir = args.stage_c_host_docs or paths.host_docs_dir_for(args.stage_c_host)
    llm_cfg = StageCLLMConfig(
        provider=args.stage_c_provider,
        model=args.stage_c_model,
        temperature=args.stage_c_temperature,
        max_tokens=args.stage_c_max_tokens,
    )
    cfg = StageCConfig(
        format_tree_path=format_tree_path,
        host_name=args.stage_c_host,
        protocol_family=args.stage_c_protocol_family or args.protocol,
        host_docs_dir=host_docs_dir,
        output_dir=output_dir,
        host_ir_cache_path=args.stage_c_host_ir_cache,
        codegen_profile_path=codegen_profile_path,
        doc_summary_text=(doc_summary_text if doc_summary_enabled else None),
        verify=bool(args.stage_c_verify),
        llm_config=llm_cfg,
        constraints_mode="codegen",
        contract_repair_max_rounds=int(getattr(args, "stage_c_contract_repair_max_rounds", 2) or 2),
        allow_missing_contract_tags=bool(getattr(args, "stage_c_allow_missing_contract_tags", False)),
    )
    if reuse_stage_c and plugin_dir.exists():
        try:
            has_files = any(p.is_file() for p in plugin_dir.rglob("*"))
        except OSError:
            has_files = True
        if has_files:
            expected_signature = build_stage_c_input_signature(
                cfg,
                input_spec=("doc_summary" if doc_summary_enabled else "format_tree"),
                constraints_mode=effective_constraints_mode,
                constraints_mode_requested=requested_constraints_mode,
            )
            rebuild_reasons: List[str] = []
            existing_constraints_mode: Optional[str] = None
            meta: dict = {}
            meta_path = plugin_dir / "_pargen_stage_c_meta.json"
            if meta_path.exists():
                try:
                    loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta = loaded_meta if isinstance(loaded_meta, dict) else {}
                    if isinstance(meta, dict) and meta.get("constraints_mode"):
                        existing_constraints_mode = str(meta.get("constraints_mode") or "").strip().lower() or None
                except Exception:
                    rebuild_reasons.append("metadata file is unreadable")
                    existing_constraints_mode = None
            else:
                rebuild_reasons.append("metadata file is missing")
            if existing_constraints_mode is None and report_path.exists():
                try:
                    loaded = json.loads(report_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        loaded_host = str(loaded.get("host") or "").strip().lower()
                        if loaded_host == str(args.stage_c_host or "").strip().lower() and loaded.get("constraints_mode"):
                            existing_constraints_mode = str(loaded.get("constraints_mode") or "").strip().lower() or None
                except Exception:
                    existing_constraints_mode = None

            if existing_constraints_mode != effective_constraints_mode:
                rebuild_reasons.append(
                    f"constraints_mode mismatch: existing={existing_constraints_mode!r}, requested={effective_constraints_mode!r}"
                )

            existing_signature = meta.get("input_signature") if isinstance(meta, dict) else None
            if existing_signature != expected_signature:
                rebuild_reasons.append("input signature is missing or does not match current Stage C inputs")

            summary: dict = {
                "host": args.stage_c_host,
                "protocol": "Unknown Protocol",
                "target_protocol": args.stage_c_protocol_family or args.protocol,
                "plugin_dir": str(plugin_dir),
                "files": sorted([p.relative_to(plugin_dir).as_posix() for p in plugin_dir.rglob("*") if p.is_file()]),
                "docs_loaded": 0,
            }
            if codegen_profile_path:
                summary["codegen_profile_path"] = str(codegen_profile_path)

            # Best-effort merge with the existing Stage C summary file (may be for a different host).
            if report_path.exists():
                try:
                    loaded = json.loads(report_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        for k, v in loaded.items():
                            if k not in summary:
                                summary[k] = v
                except Exception:
                    pass

            if not doc_summary_enabled:
                try:
                    from pargen.parser_realization.constraint_contract import build_contract, find_missing_tags
                    from pargen.parser_realization.parse_ir import build_parse_ir

                    if format_tree_path is None:
                        raise ValueError("format_tree_path is required for constraint reuse validation")
                    tree_json = json.loads(format_tree_path.read_text(encoding="utf-8"))
                    parse_ir = build_parse_ir(tree_json)
                    contract_items = build_contract(parse_ir)

                    src_files: dict[str, str] = {}
                    for path in plugin_dir.rglob("*"):
                        if not path.is_file():
                            continue
                        rel = path.relative_to(plugin_dir).as_posix()
                        try:
                            src_files[rel] = path.read_text(encoding="utf-8")
                        except Exception:
                            continue

                    missing_tags = find_missing_tags(
                        files=src_files,
                        items=contract_items,
                        host_name=str(args.stage_c_host or ""),
                    )
                    if missing_tags and not (
                        bool(getattr(args, "stage_c_allow_missing_contract_tags", False))
                        or bool(getattr(args, "stage_c_exp_constraint_fix", False))
                    ):
                        sample = ", ".join(missing_tags[:10])
                        rebuild_reasons.append(
                            f"constraint contract missing tags: missing={len(missing_tags)} sample=[{sample}]"
                        )
                except Exception as exc:
                    rebuild_reasons.append(f"constraint contract validation failed: {exc}")

            if rebuild_reasons:
                if bool(getattr(args, "stage_c_reuse_ignore_signature", False)):
                    print("\n[Stage C] Reusing existing Stage C output despite signature mismatch")
                    for reason in rebuild_reasons[:8]:
                        print(f"  - ignored: {reason}")
                    print(f"  plugin_dir: {plugin_dir}")
                    with _record_stage_time(timing, "stage_c_post_tools_s"):
                        run_stage_c_post_tools(
                            summary,
                            paths,
                            args,
                            format_tree_path=(Path(format_tree_path) if not doc_summary_enabled else None),
                            timing=timing,
                        )
                    return
                else:
                    print("\n[Stage C] Existing Stage C output is not reusable; rebuilding")
                    for reason in rebuild_reasons[:8]:
                        print(f"  - {reason}")
            else:
                print("\n[Stage C] Reusing existing Stage C output; skipping code generation")
                print(f"  plugin_dir: {plugin_dir}")
                print("  (use --no-reuse-c-stage-output to rebuild Stage C)")
                with _record_stage_time(timing, "stage_c_post_tools_s"):
                    run_stage_c_post_tools(
                        summary,
                        paths,
                        args,
                        format_tree_path=(Path(format_tree_path) if not doc_summary_enabled else None),
                        timing=timing,
                    )
                return

    print("\n[Stage C] Host-IR driven code generation")
    with _record_stage_time(timing, "stage_c_codegen_s"):
        summary = run_stage_c_codegen(cfg)
    print(f"  Stage C completed. Summary written to {report_path}")
    for key, val in summary.items():
        print(f"    - {key}: {val}")
    with _record_stage_time(timing, "stage_c_post_tools_s"):
        run_stage_c_post_tools(
            summary,
            paths,
            args,
            format_tree_path=(Path(format_tree_path) if not doc_summary_enabled else None),
            timing=timing,
        )


def _base_protocol_name(protocol: str) -> str:
    """A rerun dataset is named <protocol>[_variant...] (ipv4_gtm1_k004_seed0 -> ipv4)."""
    return str(protocol or "").split("_")[0]


def _dataset_gt_roots(*, root: Path, inputs_dir: Path, protocol: str) -> list[Path]:
    """Dataset GT folders: the dataset's own, then its base protocol's (a rerun dataset copies
    inputs/ only, and some protocols keep their GT under <protocol>/gt/)."""
    roots = [root / "gt", inputs_dir / "gt"]
    base = _base_protocol_name(protocol)
    if base and base != str(protocol or ""):
        roots += [DATA_DIR / base / "gt", DATA_DIR / base / "inputs" / "gt"]
    return roots


def _resolve_default_stage_c_pcap_path(*, inputs_dir: Path, protocol: str) -> Path:
    """Resolve the default PCAP path for Stage C (with backward-compatible fallbacks).

    Historically some datasets used a misspelled `inputs/pacap/` folder. Prefer the
    canonical `inputs/pcap/` when available, but fall back to `inputs/pacap/` if
    that's where the dataset lives. A rerun dataset is named <protocol>[_variant...]
    and keeps the base protocol's capture name (ipv4_gtm1_k004_seed0 has ipv4_100.pcap).
    """

    base = _base_protocol_name(protocol)
    candidates = [
        inputs_dir / folder / f"{name}{tail}"
        for name in dict.fromkeys([protocol, base])
        for tail in ("_100.pcap", ".pcap")
        for folder in ("pcap", "pacap")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _stage_c_eval_frames(
    *, inputs_dir: Path, pcap_path: Path, codegen_profile: Optional[Mapping[str, Any]], host: str
) -> tuple[Optional[List[int]], Optional[List[int]]]:
    """Frames Stage C evaluates on (inputs/pcap_frame_index.json): (frames, raw index frames).

    Strategy B: evaluate on reassembled datagrams/streams where applicable. For IP-payload
    protocols (e.g., ICMP), collapse fragmented datagrams to the reassembled unit (keyed by
    the last fragment's frame number, matching tshark ip.defragment behavior).
    """
    from pargen.parser_realization.semantic_stage.frame_index import load_frame_index

    frame_index = load_frame_index(inputs_dir / "pcap_frame_index.json", pcap_stem=pcap_path.stem)
    semantic_frames: Optional[List[int]] = None
    if frame_index is not None:
        semantic_frames = list(frame_index.include_frames)
    semantic_frames_raw = list(semantic_frames) if semantic_frames is not None else None
    try:
        from pargen.parser_realization.target_platforms.extraction_profile import infer_ip_proto, infer_pcap_layer

        if semantic_frames is not None and pcap_path.exists():
            pcap_layer = infer_pcap_layer(codegen_profile, host=host)
            if pcap_layer == "ip_payload":
                ip_proto = infer_ip_proto(codegen_profile, host=host)
                if ip_proto is not None:
                    from pargen.parser_realization.semantic_stage.gt_from_tree import extract_ip_payloads_from_pcap

                    payloads = extract_ip_payloads_from_pcap(pcap_path, semantic_frames, ip_proto=ip_proto)
                    semantic_frames = sorted(payloads.keys())
    except Exception:
        semantic_frames = semantic_frames_raw
    return semantic_frames, semantic_frames_raw


def _find_dataset_gt_paths(
    *, root: Path, inputs_dir: Path, protocol: str, stem: str
) -> tuple[Optional[Path], Optional[Path]]:
    """Dataset GT (Level-1/Level-2 pair) for a capture <stem> of dataset <protocol>."""

    def _gt_key(path: Path, *, level: int) -> str:
        suffix = f"_level{level}_gt.json"
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
        return path.stem

    def _select_pair(gt_level1_candidates: list[Path], gt_level2_candidates: list[Path]) -> tuple[Optional[Path], Optional[Path]]:
        if not gt_level1_candidates or not gt_level2_candidates:
            return None, None

        l1_by_key: dict[str, Path] = {}
        for p in gt_level1_candidates:
            l1_by_key.setdefault(_gt_key(p, level=1), p)

        l2_by_key: dict[str, Path] = {}
        for p in gt_level2_candidates:
            l2_by_key.setdefault(_gt_key(p, level=2), p)

        common = set(l1_by_key) & set(l2_by_key)
        if not common:
            return None, None

        def _score(key: str) -> tuple[int, int, str]:
            score = 0
            if key == stem:
                score += 100
            elif key.startswith(stem):
                score += 50
            protocol_name = str(protocol) if key.startswith(str(protocol)) else _base_protocol_name(protocol)
            if key == protocol_name:
                score += 40
            elif key.startswith(protocol_name):
                score += 20
            # Prefer a more specific key if ties remain.
            return score, len(key), key

        best_key = max(common, key=_score)
        return l1_by_key[best_key], l2_by_key[best_key]

    # Search order:
    # 1) data/<protocol>/gt/<pcap_stem>/*_level{1,2}_gt.json (legacy)
    # 2) data/<protocol>/inputs/gt/*_level{1,2}_gt.json (used by some datasets like BFD)
    # 3) the same under the base protocol's dataset, for a rerun dataset <protocol>_<variant>
    base_protocol = _base_protocol_name(protocol)
    for gt_root in _dataset_gt_roots(root=root, inputs_dir=inputs_dir, protocol=protocol):
        if not gt_root.exists():
            continue

        gt_dir = gt_root / stem
        if gt_dir.exists():
            gt_level1, gt_level2 = _select_pair(
                sorted(gt_dir.glob("*_level1_gt.json")),
                sorted(gt_dir.glob("*_level2_gt.json")),
            )
            if gt_level1 and gt_level2:
                return gt_level1, gt_level2

        gt_level1, gt_level2 = _select_pair(
            sorted(gt_root.rglob(f"{stem}*_level1_gt.json")),
            sorted(gt_root.rglob(f"{stem}*_level2_gt.json")),
        )
        if gt_level1 and gt_level2:
            return gt_level1, gt_level2

        for name in dict.fromkeys([str(protocol), base_protocol]):
            gt_level1, gt_level2 = _select_pair(
                sorted(gt_root.rglob(f"{name}*_level1_gt.json")),
                sorted(gt_root.rglob(f"{name}*_level2_gt.json")),
            )
            if gt_level1 and gt_level2:
                return gt_level1, gt_level2

        l1_all = sorted(gt_root.rglob("*_level1_gt.json"))
        l2_all = sorted(gt_root.rglob("*_level2_gt.json"))
        if len(l1_all) == 1 and len(l2_all) == 1:
            return l1_all[0], l2_all[0]

    return None, None


def _wireshark_builtin_gt(
    *,
    protocol: str,
    pcap_path: Path,
    codegen_profile: Optional[Mapping[str, Any]],
    host: str,
    tshark: Optional[Any],
    out_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """Export a tshark-derived reference for diagnostics.
    
    Write outputs under <out_dir>/metric/_gt_wireshark_builtin/."""
    from pargen.parser_realization.target_platforms.wireshark_export import export_wireshark_tooljson

    try:
        from pargen.parser_realization.target_platforms.extraction_profile import infer_pcap_layer

        pcap_layer = infer_pcap_layer(codegen_profile, host=host)
    except Exception:
        pcap_layer = ""

    force = str(os.getenv("STAGE_C_DATASET_GT_SOURCE") or "").strip().lower() in {"tshark", "wireshark"}
    if not force and pcap_layer != "ip_payload":
        return None, None

    tshark_path: Optional[Path] = None
    if tshark is not None:
        try:
            tshark_path = Path(str(tshark)).expanduser()
        except Exception:
            tshark_path = None
    if tshark_path is None:
        tshark_path_str = shutil.which("tshark")
        if tshark_path_str:
            tshark_path = Path(tshark_path_str)
    if tshark_path is None or not tshark_path.exists():
        return None, None

    builtin_filter = str(os.getenv("STAGE_C_DATASET_GT_FILTER") or "").strip() or _base_protocol_name(protocol)
    if not builtin_filter:
        return None, None

    gt_dir = out_dir / "metric" / "_gt_wireshark_builtin" / pcap_path.stem / builtin_filter
    gt_dir.mkdir(parents=True, exist_ok=True)

    tool_out = gt_dir / "tooljson"
    tool_res = export_wireshark_tooljson(
        tshark_path=tshark_path,
        pcap_path=pcap_path,
        filter_name=builtin_filter,
        out_dir=tool_out,
        export_jsonraw=True,
        export_texttree=False,
        max_packets=0,
        extra_tshark_args=[
            "-o",
            "ip.defragment:TRUE",
            "-o",
            "ipv6.defragment:TRUE",
            "-o",
            "tcp.desegment_tcp_streams:TRUE",
        ],
        codegen_profile=None,
    )

    try:
        recs = json.loads(tool_res.tooljson_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(recs, list) or not recs:
        return None, None

    packets: list[dict] = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        raw_no = rec.get("packet_no")
        try:
            frame_no = int(raw_no) if raw_no is not None else None
        except Exception:
            frame_no = None
        if frame_no is None or frame_no <= 0:
            continue
        pkt = {
            "frame_no": int(frame_no),
            "hex": str(rec.get("hex") or ""),
            "fields": rec.get("fields") if isinstance(rec.get("fields"), list) else [],
        }
        packets.append(pkt)

    if not packets:
        return None, None
    packets.sort(key=lambda p: int(p.get("frame_no", 0) or 0))
    payload = {
        "source": "tshark_builtin",
        "filter": builtin_filter,
        "pcap": str(pcap_path),
        "packets": packets,
    }

    l1 = gt_dir / f"{pcap_path.stem}_{builtin_filter}_level1_gt.json"
    l2 = gt_dir / f"{pcap_path.stem}_{builtin_filter}_level2_gt.json"
    l1.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    l2.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return l1, l2


def _stage_c_dataset_gt(
    *,
    root: Path,
    inputs_dir: Path,
    protocol: str,
    pcap_path: Path,
    codegen_profile: Optional[Mapping[str, Any]],
    host: str,
    tshark: Optional[Any],
    out_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """Use the checked-in dataset annotations consistently across target hosts.

    Built-in dissector exports may be produced separately for diagnostics, but must
    not silently replace frozen annotations according to the protocol or environment.
    """
    return _find_dataset_gt_paths(root=root, inputs_dir=inputs_dir, protocol=protocol, stem=pcap_path.stem)


def _merge_stage_c_codegen_profiles(
    base_profile: Optional[Mapping[str, Any]],
    host_profile: Optional[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Deep-merge a host profile without dropping base transport registration."""

    if not isinstance(base_profile, Mapping) and not isinstance(host_profile, Mapping):
        return None

    def _merge(base: Any, override: Any, *, key: str = "") -> Any:
        if isinstance(base, Mapping) and isinstance(override, Mapping):
            merged = {str(k): copy.deepcopy(v) for k, v in base.items()}
            for child_key, child_value in override.items():
                child_key_s = str(child_key)
                if child_key_s in merged:
                    merged[child_key_s] = _merge(
                        merged[child_key_s],
                        child_value,
                        key=child_key_s,
                    )
                else:
                    merged[child_key_s] = copy.deepcopy(child_value)
            if key == "registration":
                for registration_key in (
                    "transport",
                    "dissector_table",
                    "port",
                    "ports",
                    "port_range",
                    "selector",
                    "selector_type",
                ):
                    if merged.get(registration_key) in (None, "", [], ()) and base.get(registration_key) not in (
                        None,
                        "",
                        [],
                        (),
                    ):
                        merged[registration_key] = copy.deepcopy(base[registration_key])
            return merged
        if key == "registration" and isinstance(base, Mapping):
            return copy.deepcopy(dict(base))
        return copy.deepcopy(override)

    base = dict(base_profile) if isinstance(base_profile, Mapping) else {}
    override = dict(host_profile) if isinstance(host_profile, Mapping) else {}
    return _merge(base, override)


def run_stage_c_post_tools(
    summary: dict,
    paths: ProtocolPaths,
    args: argparse.Namespace,
    *,
    format_tree_path: Optional[Path],
    timing: Optional[dict] = None,
) -> None:
    """Optional post-Stage-C steps: build Wireshark + run tshark export + semantic loop."""
    stage_c_out_dir = Path(args.stage_c_output_dir) if getattr(args, "stage_c_output_dir", None) else paths.stage_c_output_dir
    semantic_enabled = bool(getattr(args, "stage_c_semantic_verify", False) or getattr(args, "stage_c_semantic_fix", False))
    exp_constraints_enabled = bool(getattr(args, "stage_c_exp_constraint_verify", False))
    requested_suite_source = str(
        getattr(args, "stage_c_exp_constraint_suite_source", "pkg") or "pkg"
    ).strip().lower()
    if bool(getattr(args, "stage_c_exp_constraint_generate_from_traffic", False)):
        requested_suite_source = "traffic"
    pkg_tree_only_mode = exp_constraints_enabled and requested_suite_source == "pkg"
    if (
        pkg_tree_only_mode
        and bool(getattr(args, "stage_c_semantic_fix", False))
    ):
        raise SystemExit(
            "--stage-c-semantic-fix cannot be combined with the PKG message suite; "
            "use --no-stage-c-semantic-fix"
        )
    if (
        pkg_tree_only_mode
        and str(getattr(args, "stage_c_host", "") or "").strip().lower() == "zeek_spicy"
        and bool(getattr(args, "stage_c_zeek_runtime_check", False))
    ):
        raise SystemExit(
            "PKG tree-only conformance is incompatible with Zeek dataset runtime repair: "
            "use --no-stage-c-zeek-runtime-check"
        )
    if not (args.stage_c_build_ws or args.stage_c_pcap_to_texttree or semantic_enabled or exp_constraints_enabled):
        return

    from pargen.parser_realization.syntax_stage import SyntaxRepairLoopConfig, run_syntax_with_auto_repair
    from pargen.parser_realization.syntax_stage.backends.scapy import ScapyBackend
    from pargen.parser_realization.syntax_stage.backends.wireshark_c import WiresharkCBackend
    from pargen.parser_realization.syntax_stage.backends.wireshark_lua import WiresharkLuaBackend
    from pargen.parser_realization.syntax_stage.backends.zeek_spicy import ZeekSpicyBackend
    from pargen.parser_realization.target_platforms.scapy_export import export_scapy_tooljson
    from pargen.parser_realization.target_platforms.wireshark_export import export_wireshark_tooljson
    from pargen.parser_realization.target_platforms.wireshark_lua_syntax_tester import resolve_lua_entry
    from pargen.parser_realization.target_platforms.wireshark_syntax_tester import WIRESHARK_DEFAULT_URL

    plugin_dir = Path(summary.get("plugin_dir") or (stage_c_out_dir / "plugin" / args.stage_c_host))
    ws_src_root = args.stage_c_ws_src_root or (DATA_DIR / "tools" / "wireshark-4.4.11")
    ws_build_dir = args.stage_c_ws_build_dir or (ws_src_root / "build")
    plugin_id = args.stage_c_plugin_id or f"{args.protocol}_bs"
    ws_filter = args.stage_c_ws_filter or args.protocol
    codegen_profile: Optional[dict] = None
    profile_path = args.stage_c_codegen_profile
    if profile_path is None:
        try:
            from_summary = summary.get("codegen_profile_path")
            if from_summary:
                profile_path = Path(str(from_summary))
        except Exception:
            profile_path = None
    if profile_path is None:
        host_candidate = paths.inputs_dir / f"stage_c_codegen_profile_{args.stage_c_host}.json"
        if host_candidate.exists():
            profile_path = host_candidate
        else:
            candidate = paths.inputs_dir / "stage_c_codegen_profile.json"
            if candidate.exists():
                profile_path = candidate
    if profile_path and Path(profile_path).exists():
        try:
            payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                codegen_profile = strip_protocol_hints(payload)
        except Exception:
            codegen_profile = None

    pkg_codegen_profile = codegen_profile
    if pkg_tree_only_mode:
        base_profile: Optional[dict] = None
        base_profile_path = paths.inputs_dir / "stage_c_codegen_profile.json"
        if base_profile_path.exists():
            try:
                loaded_base_profile = json.loads(base_profile_path.read_text(encoding="utf-8"))
                if isinstance(loaded_base_profile, dict):
                    base_profile = strip_protocol_hints(loaded_base_profile)
            except Exception:
                base_profile = None
        pkg_codegen_profile = _merge_stage_c_codegen_profiles(base_profile, codegen_profile)

    # Prefer the host registration.filter_name from the codegen profile if present.
    try:
        reg = (codegen_profile or {}).get(args.stage_c_host, {}).get("registration", {})  # type: ignore[union-attr]
        if isinstance(reg, dict) and reg.get("filter_name"):
            ws_filter = str(reg.get("filter_name"))
    except Exception:
        pass

    def _lua_tshark_extra_args(entry: Path) -> List[str]:
        extra: List[str] = []
        try:
            reg = (codegen_profile or {}).get("wireshark_lua", {}).get("registration", {})  # type: ignore[union-attr]
        except Exception:
            reg = {}
        raw_disable: Any = None
        if isinstance(reg, dict):
            raw_disable = (
                reg.get("disable_builtin_protocols")
                or reg.get("disable_builtin_protocol")
                or reg.get("disable_protocols")
                or reg.get("disable_protocol")
            )
        if isinstance(raw_disable, str):
            disable_protocols = [raw_disable]
        elif isinstance(raw_disable, (list, tuple, set)):
            disable_protocols = [str(item) for item in raw_disable]
        else:
            disable_protocols = []
        for proto_name in disable_protocols:
            proto_name = str(proto_name or "").strip()
            if proto_name:
                extra.extend(["--disable-protocol", proto_name])
        extra.extend(["-X", f"lua_script:{entry}"])
        return extra

    def _wireshark_c_tshark_extra_args() -> List[str]:
        extra: List[str] = []
        try:
            reg = (codegen_profile or {}).get("wireshark_c", {}).get("registration", {})  # type: ignore[union-attr]
        except Exception:
            reg = {}
        raw_disable: Any = None
        if isinstance(reg, dict):
            raw_disable = (
                reg.get("runtime_disable_protocols")
                or reg.get("runtime_disable_protocol")
                or reg.get("disable_protocols")
                or reg.get("disable_protocol")
            )
        if isinstance(raw_disable, str):
            disable_protocols = [raw_disable]
        elif isinstance(raw_disable, (list, tuple, set)):
            disable_protocols = [str(item) for item in raw_disable]
        else:
            disable_protocols = []
        for proto_name in disable_protocols:
            proto_name = str(proto_name or "").strip()
            if proto_name:
                extra.extend(["--disable-protocol", proto_name])
        return extra

    build_result: Optional[Any] = None
    lua_syntax_result: Optional[Any] = None
    lua_entry: Optional[Path] = None
    scapy_syntax_result: Optional[Any] = None
    semantic_result: Optional[Any] = None
    exp_constraints_result: Optional[Any] = None
    generated_exp_constraint_suite: Optional[Any] = None
    ws_plugin_needs_rebuild = False
    pcap_path = args.stage_c_pcap or _resolve_default_stage_c_pcap_path(inputs_dir=paths.inputs_dir, protocol=args.protocol)


    # Constraint validation refresh: honor Stage C constraints mode when reusing existing outputs.
    constraints_mode = "codegen"
    allow_missing_contract_tags = bool(getattr(args, "stage_c_allow_missing_contract_tags", False))


    if plugin_dir.exists():
        try:
            from pargen.parser_realization.constraint_contract import build_contract, find_missing_tags
            from pargen.parser_realization.parse_ir import build_parse_ir

            contract_items = []
            parse_ir = None
            if format_tree_path is not None and format_tree_path.exists():
                tree_json = json.loads(format_tree_path.read_text(encoding="utf-8"))
                parse_ir = build_parse_ir(tree_json)
                contract_items = build_contract(parse_ir)

            src_files: dict[str, str] = {}
            for path in plugin_dir.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(plugin_dir).as_posix()
                try:
                    src_files[rel] = path.read_text(encoding="utf-8")
                except Exception:
                    continue

            missing_tags = find_missing_tags(files=src_files, items=contract_items, host_name=args.stage_c_host) if contract_items else []
            if not contract_items:
                missing_tags = []
            elif constraints_mode == "codegen":
                if missing_tags and not allow_missing_contract_tags:
                    sample = ", ".join(missing_tags[:10])
                    raise RuntimeError(
                        "Stage C constraint contract tags missing after codegen "
                        f"(missing={len(missing_tags)} sample=[{sample}])"
                    )
        except Exception as exc:
            raise RuntimeError(f"Stage C constraint validation refresh failed: {exc}") from exc

    if (
        semantic_enabled
        or args.stage_c_pcap_to_texttree
        or bool(getattr(args, "stage_c_exp_constraint_generate_from_traffic", False))
    ) and not pcap_path.exists():
        raise SystemExit(f"PCAP not found: {pcap_path}")

    frame_index_path = paths.inputs_dir / "pcap_frame_index.json"
    semantic_frames, semantic_frames_raw = _stage_c_eval_frames(
        inputs_dir=paths.inputs_dir, pcap_path=pcap_path, codegen_profile=codegen_profile, host=str(args.stage_c_host or "")
    )

    def _maybe_compute_dataset_metrics(*, pred_tooljson: Path) -> Optional[dict]:
        # Best-effort dataset GT evaluation (independent from semantic oracle).
        try:
            gt_level1: Optional[Path] = None
            gt_level2: Optional[Path] = None

            gt_level1, gt_level2 = _stage_c_dataset_gt(
                root=paths.root,
                inputs_dir=paths.inputs_dir,
                protocol=str(args.protocol),
                pcap_path=pcap_path,
                codegen_profile=codegen_profile,
                host=str(args.stage_c_host or ""),
                tshark=args.stage_c_tshark,
                out_dir=stage_c_out_dir,
            )
            if not gt_level1 or not gt_level2:
                return None

            from pargen.parser_realization.target_platforms.tooljson_metrics import compute_tooljson_metrics

            metrics_out_dir = stage_c_out_dir / "metric" / args.stage_c_host
            relax_keywords = args.stage_c_metric_relax_payload_keyword
            if (
                bool(args.stage_c_metric_relax_payload_split_merge)
                and relax_keywords is None
                and _base_protocol_name(args.protocol).strip().lower() == "modbus"
            ):
                # Modbus/TCP often represents payload-like data as per-register fields; treat split/merge as equivalent.
                relax_keywords = ["data", "payload", "register"]
            metrics = compute_tooljson_metrics(
                pred_tooljson=pred_tooljson,
                gt_level1=gt_level1,
                gt_level2=gt_level2,
                out_dir=metrics_out_dir,
                frame_index=semantic_frames,
                l2_bert_enable=bool(args.stage_c_metric_l2_bert),
                l2_bert_threshold=float(args.stage_c_metric_l2_bert_threshold),
                l2_bert_model_type=args.stage_c_metric_l2_bert_model_type,
                l2_bert_device=str(args.stage_c_metric_l2_bert_device),
                l2_bert_batch_size=int(args.stage_c_metric_l2_bert_batch_size),
                l2_bert_rescale_with_baseline=bool(args.stage_c_metric_l2_bert_rescale_with_baseline),
                l2_bert_idf=bool(args.stage_c_metric_l2_bert_idf),
                l2_bert_use_fast_tokenizer=bool(args.stage_c_metric_l2_bert_use_fast_tokenizer),
                l2_judge_enable=bool(args.stage_c_metric_l2_judge),
                l2_judge_protocol=_base_protocol_name(args.protocol),
                l2_judge_provider=str(args.stage_c_metric_l2_judge_provider or args.stage_c_provider or "deepseek"),
                l2_judge_model=(args.stage_c_metric_l2_judge_model or args.stage_c_model),
                l2_judge_temperature=float(args.stage_c_metric_l2_judge_temperature),
                l2_judge_max_tokens=int(args.stage_c_metric_l2_judge_max_tokens),
                l2_judge_batch_size=int(args.stage_c_metric_l2_judge_batch_size),
                l2_judge_cache_path=args.stage_c_metric_l2_judge_cache,
                primary_l2_metric=str(args.stage_c_metric_primary_l2 or "auto"),
                human_review_path=getattr(args, "stage_c_metric_human_review", None),
                evaluation_metadata={
                    "gt_selection": "frozen_dataset_files",
                    "frame_index_path": str(frame_index_path),
                    "raw_index_frame_ids": semantic_frames_raw,
                    "index": json.loads(frame_index_path.read_text(encoding="utf-8")) if frame_index_path.exists() else None,
                    "evaluation_unit": "host_extraction_unit_after_reassembly",
                },
                relax_payload_split_merge=bool(args.stage_c_metric_relax_payload_split_merge),
                relax_payload_keywords=relax_keywords,
            )

            from pargen.parser_realization.target_platforms.tooljson_metrics import stage_c_metric_summary

            msg, counts = stage_c_metric_summary(metrics)

            print(msg)
            print(counts)
            if semantic_frames is not None:
                if semantic_frames_raw is not None and len(semantic_frames) != len(semantic_frames_raw):
                    print(
                        f"  dataset metrics evaluated_frames={len(semantic_frames)} "
                        f"(reassembled; raw_index={len(semantic_frames_raw)} from {frame_index_path})"
                    )
                else:
                    print(f"  dataset metrics evaluated_frames={len(semantic_frames)} (from {frame_index_path})")
            print(f"  wrote dataset metrics to {metrics_out_dir}")
            return metrics
        except Exception as exc:
            if getattr(args, "stage_c_metric_human_review", None) or bool(getattr(args, "stage_c_metric_l2_bert", False)) or bool(
                getattr(args, "stage_c_metric_l2_judge", False)
            ):
                raise SystemExit(str(exc)) from exc
            print(f"  [Stage C] dataset metric computation skipped/failed: {exc}")
            return None

    output_path: Optional[Path] = None
    json_combined_path: Optional[Path] = None
    if args.stage_c_pcap_to_texttree:
        # Default to host-specific filenames to avoid overwriting results when
        # evaluating multiple Stage C hosts (wireshark_c/wireshark_lua/scapy/zeek_spicy).
        if args.stage_c_pcap_output is None:
            args.stage_c_pcap_output = stage_c_out_dir / f"tool_{args.stage_c_host}.txt"
        output_path = args.stage_c_pcap_output or (stage_c_out_dir / "tool.txt")
        json_combined_path = output_path.with_suffix(".json")

        def _finalize_export(export_result: Any) -> Optional[dict]:
            assert output_path is not None
            assert json_combined_path is not None

            # Keep legacy output paths stable (custom --stage-c-pcap-output paths).
            if export_result.texttree_path and export_result.texttree_path != output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(export_result.texttree_path.read_text(encoding="utf-8"), encoding="utf-8")

            if export_result.tooljson_path != json_combined_path:
                json_combined_path.parent.mkdir(parents=True, exist_ok=True)
                json_combined_path.write_text(export_result.tooljson_path.read_text(encoding="utf-8"), encoding="utf-8")

            print(f"  Wrote protocol tree to {output_path}")
            print(f"  wrote {export_result.packets} packets to {json_combined_path}")

            # If ground-truth (GT) exists for this PCAP, compute metrics automatically.
            # This is best-effort and will not fail the pipeline.
            return _maybe_compute_dataset_metrics(pred_tooljson=json_combined_path)

    def _run_stage_c_export(*, reason: str) -> Optional[dict]:
        if not args.stage_c_pcap_to_texttree:
            return None
        assert output_path is not None

        if args.stage_c_host == "wireshark_c":
            # If the user explicitly provided --stage-c-tshark, respect it (no auto build / no plugin injection).
            if args.stage_c_tshark is not None:
                tshark_path = _find_tshark(ws_build_dir, args.stage_c_tshark)
            else:
                # For tshark export we MUST ensure the current generated plugin is built and loadable.
                # Otherwise we may end up running an old tshark build without the newly generated plugin.
                if build_result is None:
                    _build_ws_with_plugin(reason=f"{reason} requires plugin")
                tshark_path = None
                if build_result and build_result.artifacts and build_result.artifacts.tool_bin:
                    tshark_path = build_result.artifacts.tool_bin
                if tshark_path is None:
                    try:
                        tshark_path = _ensure_tshark(ws_build_dir, logs_root=paths.stage_c_logs_dir)
                    except Exception as exc:
                        raise SystemExit(str(exc)) from exc

            print(f"\n[Stage C] Exporting protocol tree + tool.json via tshark to {output_path.parent}")

            def _run_export() -> Any:
                return export_wireshark_tooljson(
                    tshark_path=Path(tshark_path),
                    pcap_path=pcap_path,
                    filter_name=ws_filter,
                    out_dir=output_path.parent,
                    export_jsonraw=True,
                    export_texttree=True,
                    max_packets=0,
                    extra_tshark_args=_wireshark_c_tshark_extra_args(),
                    codegen_profile=codegen_profile,
                    ws_isolate_plugins=bool(getattr(args, "stage_c_ws_isolate_plugins", True)),
                    ws_plugin_id=(str(plugin_id) if plugin_id else None),
                )

            try:
                export_result = _run_export()
            except Exception as exc:
                stderr = str(exc)
                rebuild_triggers = (
                    "Duplicate protocol filter_name",
                    "is not a valid protocol",
                    "dissector table",
                    "OOPS:",
                )
                if args.stage_c_tshark is None and any(tok in stderr for tok in rebuild_triggers):
                    print("\n[Stage C] tshark export failed; attempting to rebuild Wireshark with auto-repair and retry...")
                    _build_ws_with_plugin(reason="tshark export retry")
                    if build_result and build_result.artifacts and build_result.artifacts.tool_bin:
                        tshark_path = build_result.artifacts.tool_bin
                    else:
                        tshark_path = _ensure_tshark(ws_build_dir, logs_root=paths.stage_c_logs_dir)
                    export_result = _run_export()
                else:
                    raise SystemExit(stderr)

            # _finalize_export is defined only when stage_c_pcap_to_texttree is enabled.
            return _finalize_export(export_result)  # type: ignore[name-defined]

        if args.stage_c_host == "wireshark_lua":
            if args.stage_c_tshark is not None:
                tshark_path = Path(args.stage_c_tshark)
            else:
                tshark_path_str = shutil.which("tshark")
                if not tshark_path_str:
                    raise SystemExit("tshark not found on PATH; pass --stage-c-tshark for wireshark_lua export.")
                tshark_path = Path(tshark_path_str)

            entry = _get_lua_entry()
            extra_args = _lua_tshark_extra_args(entry)

            print(f"\n[Stage C] Exporting protocol tree + tool.json via tshark to {output_path.parent}")
            try:
                export_result = export_wireshark_tooljson(
                    tshark_path=Path(tshark_path),
                    pcap_path=pcap_path,
                    filter_name=ws_filter,
                    out_dir=output_path.parent,
                    export_jsonraw=True,
                    export_texttree=True,
                    max_packets=0,
                    extra_tshark_args=extra_args,
                    codegen_profile=codegen_profile,
                    ws_isolate_plugins=bool(getattr(args, "stage_c_ws_isolate_plugins", True)),
                    ws_plugin_id=None,
                )
                return _finalize_export(export_result)  # type: ignore[name-defined]
            except Exception as exc:
                raise SystemExit(str(exc)) from exc

        if args.stage_c_host == "scapy":
            scapy_ports = _parse_scapy_ports_arg()
            scapy_transport = getattr(args, "stage_c_scapy_transport", "auto")
            entry_override = getattr(args, "stage_c_scapy_entry", None)

            print(f"\n[Stage C] Exporting protocol tree + tool.json via scapy to {output_path.parent}")
            try:
                export_result = export_scapy_tooljson(
                    plugin_src_dir=plugin_dir,
                    pcap_path=pcap_path,
                    filter_name=ws_filter,
                    out_dir=output_path.parent,
                    entry_override=entry_override,
                    ports=scapy_ports,
                    transport=scapy_transport,
                    max_packets=0,
                    include_payload_layers=True,
                    codegen_profile=codegen_profile,
                )
                return _finalize_export(export_result)  # type: ignore[name-defined]
            except Exception as exc:
                raise SystemExit(str(exc)) from exc

        if args.stage_c_host == "zeek_spicy":
            print(f"\n[Stage C] Exporting protocol tree + tool.json via spicy-dump to {output_path.parent}")
            try:
                from pargen.parser_realization.target_platforms.zeek_spicy_export import export_zeek_spicy_tooljson

                export_result = export_zeek_spicy_tooljson(
                    plugin_src_dir=plugin_dir,
                    pcap_path=pcap_path,
                    filter_name=ws_filter,
                    out_dir=output_path.parent,
                    codegen_profile=codegen_profile,
                    frame_index=(semantic_frames if semantic_enabled else None),
                    max_packets=0,
                    spicyz_path=args.stage_c_spicyz,
                    disable_runtime_constraints_for_export=bool(semantic_enabled),
                )
                return _finalize_export(export_result)  # type: ignore[name-defined]
            except Exception as exc:
                raise SystemExit(str(exc)) from exc

        print(
            f"\n[Stage C] stage_c_pcap_to_texttree requested, but host='{args.stage_c_host}'. "
            "Skipping tshark export (unsupported host)."
        )
        return None

    def _stage_c_dataset_metrics_path() -> Path:
        return stage_c_out_dir / "metric" / args.stage_c_host / "metrics.json"

    def _read_stage_c_dataset_metrics() -> Optional[dict]:
        metrics_path = _stage_c_dataset_metrics_path()
        if not metrics_path.exists():
            return None
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _metric_float(metrics: dict, *keys: str) -> float:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _dataset_metric_score(metrics: dict) -> tuple[float, float, float, float]:
        from pargen.parser_realization.target_platforms.tooljson_metrics import _stage_c_metric_reporting_view
        semantic = _stage_c_metric_reporting_view(metrics)["paper_metrics"]["primary_level2"]["score"]
        return (
            _metric_float(metrics, "field_f1_level1", "L1_f1", "l1_f1"),
            float(semantic) if semantic is not None else -1.0,
            -_metric_float(metrics, "fp_level1", "L1_fp", "fp_level1"),
            -_metric_float(metrics, "fn_level1", "L1_fn", "fn_level1"),
        )

    def _dataset_metric_regressed(candidate: dict, baseline: dict, *, eps: float = 1e-9) -> bool:
        cand_score = _dataset_metric_score(candidate)
        base_score = _dataset_metric_score(baseline)
        for cand_value, base_value in zip(cand_score, base_score):
            if cand_value < base_value - eps:
                return True
            if cand_value > base_value + eps:
                return False
        return False

    def _dataset_metric_brief(metrics: Optional[dict]) -> str:
        if not metrics:
            return "<missing>"
        from pargen.parser_realization.target_platforms.tooljson_metrics import stage_c_metric_summary
        return stage_c_metric_summary(metrics)[0].strip()

    def _snapshot_semantic_gate_state(snapshot_dir: Path) -> None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if plugin_dir.exists():
            shutil.copytree(plugin_dir, snapshot_dir / "plugin_src")

        metric_dir = stage_c_out_dir / "metric" / args.stage_c_host
        if metric_dir.exists():
            shutil.copytree(metric_dir, snapshot_dir / "metric")

        artifacts_dir = snapshot_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for path in (stage_c_out_dir.iterdir() if stage_c_out_dir.exists() else ()):
            if not path.is_file():
                continue
            if path.name.startswith("tool") or path.name in {"bitfield_map.json", "tshark_export.log"}:
                shutil.copy2(path, artifacts_dir / path.name)

    def _restore_semantic_gate_state(snapshot_dir: Path) -> None:
        plugin_snapshot = snapshot_dir / "plugin_src"
        if plugin_snapshot.exists():
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
            plugin_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(plugin_snapshot, plugin_dir)

        metric_snapshot = snapshot_dir / "metric"
        metric_dir = stage_c_out_dir / "metric" / args.stage_c_host
        if metric_snapshot.exists():
            if metric_dir.exists():
                shutil.rmtree(metric_dir)
            metric_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(metric_snapshot, metric_dir)

        artifacts_snapshot = snapshot_dir / "artifacts"
        if artifacts_snapshot.exists():
            for path in artifacts_snapshot.iterdir():
                if path.is_file():
                    shutil.copy2(path, stage_c_out_dir / path.name)

    def _write_semantic_gate_summary(payload: dict) -> None:
        try:
            out = stage_c_out_dir / "semantic_gate_summary.json"
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _get_lua_entry() -> Path:
        nonlocal lua_entry
        if lua_entry is not None:
            return lua_entry
        if not plugin_dir.exists():
            raise SystemExit(f"Stage C Lua entry resolution failed: plugin_dir does not exist: {plugin_dir}")
        try:
            lua_entry = resolve_lua_entry(
                plugin_dir,
                ws_filter,
                lua_entry=args.stage_c_lua_entry,
            )
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        return lua_entry

    def _build_ws_with_plugin(reason: str) -> Any:
        """Build (or rebuild) Wireshark with the generated plugin, with bounded auto-repair."""
        nonlocal build_result
        if args.stage_c_host != "wireshark_c":
            raise SystemExit(
                f"Stage C Wireshark build requested ({reason}), but host='{args.stage_c_host}'. "
                "Only wireshark_c is supported for build/tshark export."
            )
        if not plugin_dir.exists():
            raise SystemExit(
                f"Cannot build Wireshark ({reason}) because generated plugin_dir does not exist: {plugin_dir}"
            )

        print(f"\n[Stage C] Building Wireshark with generated plugin ({reason})")
        backend = WiresharkCBackend(
            ws_src_root=ws_src_root,
            ws_build_dir=ws_build_dir,
            plugin_id=plugin_id,
            plugin_c_mode=getattr(args, "stage_c_ws_plugin_c_mode", "generated"),
            download_url=args.stage_c_ws_download_url or WIRESHARK_DEFAULT_URL,
        )
        repair_llm_cfg = StageCLLMConfig(
            provider=args.stage_c_provider,
            model=args.stage_c_model,
            temperature=0.0,
            max_tokens=args.stage_c_max_tokens,
        )
        with _record_stage_time(timing, "stage_c_syntax_s"):
            build_result = run_syntax_with_auto_repair(
                backend=backend,
                plugin_src_dir=plugin_dir,
                protocol_filter_name=ws_filter,
                file_prefix=args.protocol,
                logs_root=paths.stage_c_logs_dir,
                codegen_profile=codegen_profile,
                llm_config=repair_llm_cfg,
                cfg=SyntaxRepairLoopConfig(),
                progress_printer=print,
            )

        print(f"  syntax_ok: {bool(build_result.ok)}")
        if build_result.ok:
            print(f"  runtime_ok: {build_result.runtime_ok}")
        else:
            print("  runtime_ok: None (build failed)")
        if build_result.build_log:
            print(f"  build_log: {build_result.build_log}")
        if build_result.runtime_log:
            print(f"  runtime_log: {build_result.runtime_log}")

        if not build_result.ok:
            details = f"build_rc={build_result.build_rc}"
            raise SystemExit(
                f"Wireshark build failed ({details}). See build log: {build_result.build_log or '<missing>'}"
            )
        if build_result.runtime_ok is False:
            raise SystemExit(
                "Wireshark build succeeded but tshark failed to load/register the plugin "
                f"(runtime_rc={build_result.runtime_rc}). See runtime log: {build_result.runtime_log or '<missing>'}"
            )
        return build_result

    def _run_lua_syntax_check(reason: str) -> Any:
        """Run Lua syntax check (luac/lua) with bounded auto-repair."""
        nonlocal lua_syntax_result
        if args.stage_c_host != "wireshark_lua":
            raise SystemExit(
                f"Stage C Lua syntax requested ({reason}), but host='{args.stage_c_host}'. "
                "Only wireshark_lua is supported for Lua syntax checks."
            )
        if not plugin_dir.exists():
            raise SystemExit(
                f"Cannot run Lua syntax ({reason}) because generated plugin_dir does not exist: {plugin_dir}"
            )

        entry = _get_lua_entry()
        print(f"\n[Stage C] Running Lua syntax check ({reason})")
        backend = WiresharkLuaBackend(
            lua_entry=entry,
            luac_path=getattr(args, "stage_c_luac", None),
            lua_path=getattr(args, "stage_c_lua", None),
        )
        repair_llm_cfg = StageCLLMConfig(
            provider=args.stage_c_provider,
            model=args.stage_c_model,
            temperature=0.0,
            max_tokens=args.stage_c_max_tokens,
        )
        with _record_stage_time(timing, "stage_c_syntax_s"):
            lua_extra: dict[str, Any] = {"pcap_path": str(pcap_path)}
            if bool(getattr(args, "stage_c_lua_load_check", False)):
                load_pcap = pcap_path
                if pkg_tree_only_mode:
                    # Build/load check only: an empty capture (pcap header, no packets) loads the plugin without
                    # handing evaluation packets to repair.
                    import struct

                    load_pcap = stage_c_out_dir / "lua_load_check_empty.pcap"
                    load_pcap.parent.mkdir(parents=True, exist_ok=True)
                    load_pcap.write_bytes(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
                lua_extra["tshark_load_check"] = {"pcap_path": str(load_pcap), "tshark": "tshark"}
            lua_backend_config = build_stage_c_backend_config(
                format_tree_path=format_tree_path,
                codegen_profile=codegen_profile,
                extra=lua_extra,
            )
            lua_syntax_result = run_syntax_with_auto_repair(
                backend=backend,
                plugin_src_dir=plugin_dir,
                protocol_filter_name=ws_filter,
                file_prefix=args.protocol,
                logs_root=paths.stage_c_logs_dir,
                codegen_profile=codegen_profile,
                backend_config=lua_backend_config,
                llm_config=repair_llm_cfg,
                cfg=SyntaxRepairLoopConfig(),
                source_globs=("*.lua",),
                progress_printer=print,
            )

        print(f"  syntax_ok: {bool(lua_syntax_result.ok)}")
        if lua_syntax_result.build_log:
            print(f"  build_log: {lua_syntax_result.build_log}")
        if not lua_syntax_result.ok:
            raise SystemExit(
                "Lua syntax check failed. See build log: "
                f"{lua_syntax_result.build_log or '<missing>'}"
            )
        return lua_syntax_result

    def _parse_scapy_ports_arg() -> Optional[List[int]]:
        raw = getattr(args, "stage_c_scapy_ports", None)
        if raw is None:
            return None
        items: List[str] = []
        if isinstance(raw, str):
            items = [raw]
        elif isinstance(raw, list):
            items = [str(v) for v in raw if v is not None]
        else:
            items = [str(raw)]

        ports: List[int] = []
        for item in items:
            for part in str(item).split(","):
                tok = part.strip()
                if not tok:
                    continue
                try:
                    p = int(tok)
                except Exception:
                    continue
                if 0 < p <= 65535:
                    ports.append(p)
        ports = sorted(set(ports))
        return ports or None

    def _run_scapy_syntax_check(reason: str) -> Any:
        """Run Python compile+import checks for scapy plugin with bounded auto-repair."""
        nonlocal scapy_syntax_result
        if args.stage_c_host != "scapy":
            raise SystemExit(
                f"Stage C Scapy syntax requested ({reason}), but host='{args.stage_c_host}'. "
                "Only scapy is supported for Scapy syntax checks."
            )
        if not plugin_dir.exists():
            raise SystemExit(
                f"Cannot run Scapy syntax ({reason}) because generated plugin_dir does not exist: {plugin_dir}"
            )

        print(f"\n[Stage C] Running Scapy syntax check ({reason})")
        backend = ScapyBackend(entry_override=getattr(args, "stage_c_scapy_entry", None))
        repair_llm_cfg = StageCLLMConfig(
            provider=args.stage_c_provider,
            model=args.stage_c_model,
            temperature=0.0,
            max_tokens=args.stage_c_max_tokens,
        )
        dataset_runtime_check = not pkg_tree_only_mode
        with _record_stage_time(timing, "stage_c_syntax_s"):
            scapy_extra: dict[str, Any] = {
                "runtime_check": dataset_runtime_check,
                "max_runtime_packets": 3,
                "scapy_transport": getattr(args, "stage_c_scapy_transport", "auto"),
                "scapy_ports": _parse_scapy_ports_arg(),
            }
            if dataset_runtime_check:
                scapy_extra["pcap_path"] = str(pcap_path)
            scapy_backend_config = build_stage_c_backend_config(
                format_tree_path=format_tree_path,
                codegen_profile=codegen_profile,
                extra=scapy_extra,
            )
            scapy_syntax_result = run_syntax_with_auto_repair(
                backend=backend,
                plugin_src_dir=plugin_dir,
                protocol_filter_name=ws_filter,
                file_prefix=args.protocol,
                logs_root=paths.stage_c_logs_dir,
                codegen_profile=codegen_profile,
                backend_config=scapy_backend_config,
                llm_config=repair_llm_cfg,
                cfg=SyntaxRepairLoopConfig(),
                source_globs=("*.py",),
                progress_printer=print,
            )
        print(f"  syntax_ok: {bool(scapy_syntax_result.ok)}")
        if scapy_syntax_result.build_log:
            print(f"  build_log: {scapy_syntax_result.build_log}")
        if not scapy_syntax_result.ok:
            raise SystemExit(
                "Scapy syntax check failed. See build log: "
                f"{scapy_syntax_result.build_log or '<missing>'}"
            )
        return scapy_syntax_result

    def _run_zeek_spicy_syntax_check(reason: str) -> Any:
        """Run spicyz compilation (and optional zeek runtime checks) with bounded auto-repair."""
        if args.stage_c_host != "zeek_spicy":
            raise SystemExit(
                f"Stage C Zeek+Spicy syntax requested ({reason}), but host='{args.stage_c_host}'. "
                "Only zeek_spicy is supported for this syntax check."
            )
        if not plugin_dir.exists():
            raise SystemExit(
                f"Cannot run Zeek+Spicy syntax ({reason}) because generated plugin_dir does not exist: {plugin_dir}"
            )

        print(f"\n[Stage C] Running Zeek+Spicy syntax check ({reason})")
        backend = ZeekSpicyBackend(
            zeek_path=getattr(args, "stage_c_zeek", None),
            spicyz_path=getattr(args, "stage_c_spicyz", None),
        )
        repair_llm_cfg = StageCLLMConfig(
            provider=args.stage_c_provider,
            model=args.stage_c_model,
            temperature=0.0,
            max_tokens=args.stage_c_max_tokens,
        )
        dataset_runtime_check = bool(getattr(args, "stage_c_zeek_runtime_check", False))
        with _record_stage_time(timing, "stage_c_syntax_s"):
            zeek_extra: dict[str, Any] = {"runtime_check": dataset_runtime_check}
            if dataset_runtime_check:
                zeek_extra["pcap_path"] = str(pcap_path)
            result = run_syntax_with_auto_repair(
                backend=backend,
                plugin_src_dir=plugin_dir,
                protocol_filter_name=ws_filter,
                file_prefix=args.protocol,
                logs_root=paths.stage_c_logs_dir,
                codegen_profile=codegen_profile,
                llm_config=repair_llm_cfg,
                cfg=SyntaxRepairLoopConfig(),
                backend_config=build_stage_c_backend_config(
                    format_tree_path=format_tree_path,
                    codegen_profile=codegen_profile,
                    extra=zeek_extra,
                ),
                source_globs=("**/*.spicy", "**/*.evt", "**/*.zeek", "**/*.sig", "**/*.meta"),
                progress_printer=print,
            )

        print(f"  syntax_ok: {bool(result.ok)}")
        if result.runtime_ok is not None:
            print(f"  runtime_ok: {result.runtime_ok}")
        if result.build_log:
            print(f"  build_log: {result.build_log}")
        if result.runtime_log:
            print(f"  runtime_log: {result.runtime_log}")
        if not result.ok:
            raise SystemExit(
                "Zeek+Spicy syntax check failed. See build log: "
                f"{result.build_log or '<missing>'}"
            )
        if dataset_runtime_check and result.runtime_ok is False:
            raise SystemExit(
                "Zeek+Spicy compilation succeeded but runtime checks failed. See runtime log: "
                f"{result.runtime_log or '<missing>'}"
            )
        return result

    if args.stage_c_build_ws:
        if args.stage_c_host == "wireshark_c":
            build_result = _build_ws_with_plugin(reason="explicit request")
            ws_plugin_needs_rebuild = False
        elif args.stage_c_host == "wireshark_lua":
            lua_syntax_result = _run_lua_syntax_check(reason="explicit request")
        elif args.stage_c_host == "scapy":
            scapy_syntax_result = _run_scapy_syntax_check(reason="explicit request")
        elif args.stage_c_host == "zeek_spicy":
            _run_zeek_spicy_syntax_check(reason="explicit request")
        else:
            raise SystemExit(
                f"Stage C build requested, but host='{args.stage_c_host}' does not support syntax checks."
            )

    defer_dataset_export = exp_constraints_enabled and bool(getattr(args, "stage_c_exp_constraint_fix", False))
    if args.stage_c_pcap_to_texttree and not semantic_enabled and not defer_dataset_export:
        # With code refinement, evaluate only the retained parser after refinement.
        _run_stage_c_export(reason="post-stage-c")

    normal_dataset_parse_summary_done = False

    def _run_normal_dataset_parse_summary(*, reason: str) -> None:
        nonlocal normal_dataset_parse_summary_done
        if normal_dataset_parse_summary_done:
            return
        if bool(args.stage_c_pcap_to_texttree):
            return
        if semantic_enabled:
            # Semantic verification already runs the parser on the normal dataset PCAP, so avoid
            # repeating a potentially expensive exporter pass here.
            print("\n[Stage C] NOTE: normal dataset export is disabled (--no-stage-c-pcap-to-texttree).")
            print("  Normal dataset parsing is already exercised by Stage C semantic validation/repair.")
            normal_dataset_parse_summary_done = True
            return
        if not pcap_path.exists():
            print(f"\n[Stage C] normal dataset parse summary skipped (pcap missing): {pcap_path}")
            normal_dataset_parse_summary_done = True
            return

        out_dir = stage_c_out_dir / "exp_constraint" / args.stage_c_host
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "normal_dataset_parse_summary.json"
        summary_payload: dict[str, Any] = {
            "protocol": str(args.protocol),
            "host": str(args.stage_c_host),
            "pcap": str(pcap_path),
            "frames_indexed": (len(semantic_frames) if semantic_frames is not None else None),
        }

        print(f"\n[Stage C] Normal dataset parse summary (reason={reason})")
        try:
            ok = 0
            fail = 0
            timeouts = 0
            exported = 0
            error_samples: list[dict[str, Any]] = []

            with tempfile.TemporaryDirectory(prefix="pargen_stage_c_normal_dataset_") as td:
                tmp_dir = Path(td)

                if args.stage_c_host == "zeek_spicy":
                    from pargen.parser_realization.target_platforms.zeek_spicy_export import export_zeek_spicy_tooljson

                    export_zeek_spicy_tooljson(
                        plugin_src_dir=plugin_dir,
                        pcap_path=pcap_path,
                        filter_name=ws_filter,
                        out_dir=tmp_dir,
                        codegen_profile=codegen_profile,
                        frame_index=semantic_frames,
                        max_packets=0,
                        spicyz_path=args.stage_c_spicyz,
                        fail_on_all_parse_failures=False,
                    )
                    raw_timeout = os.getenv("PARGEN_ZEEK_SPICY_TIMEOUT_S")
                    timeout_s = 10.0
                    if raw_timeout is not None:
                        try:
                            timeout_s = float(str(raw_timeout).strip())
                        except Exception:
                            timeout_s = 10.0
                    summary_payload["timeout_s"] = timeout_s

                elif args.stage_c_host == "scapy":
                    scapy_ports = _parse_scapy_ports_arg()
                    scapy_transport = getattr(args, "stage_c_scapy_transport", "auto")
                    entry_override = getattr(args, "stage_c_scapy_entry", None)
                    export_scapy_tooljson(
                        plugin_src_dir=plugin_dir,
                        pcap_path=pcap_path,
                        filter_name=ws_filter,
                        out_dir=tmp_dir,
                        entry_override=entry_override,
                        ports=scapy_ports,
                        transport=scapy_transport,
                        max_packets=0,
                        include_payload_layers=True,
                        codegen_profile=codegen_profile,
                    )

                elif args.stage_c_host in {"wireshark_c", "wireshark_lua"}:
                    # Export parsed fields directly through tshark.
                    tshark_path = None
                    extra_args = None
                    ws_plugin_id = None

                    if args.stage_c_host == "wireshark_c":
                        if args.stage_c_tshark is not None:
                            tshark_path = _find_tshark(ws_build_dir, args.stage_c_tshark)
                        else:
                            try:
                                tshark_path = _find_tshark(ws_build_dir)
                            except Exception:
                                tshark_path = None
                            if tshark_path is None:
                                tshark_path = _ensure_tshark(ws_build_dir, logs_root=paths.stage_c_logs_dir)
                        ws_plugin_id = (str(plugin_id) if plugin_id else None)

                    else:
                        if args.stage_c_tshark is not None:
                            tshark_path = Path(args.stage_c_tshark)
                        else:
                            tshark_path_str = shutil.which("tshark")
                            if not tshark_path_str:
                                raise RuntimeError("tshark not found on PATH; pass --stage-c-tshark.")
                            tshark_path = Path(tshark_path_str)
                        entry = _get_lua_entry()
                        extra_args = _lua_tshark_extra_args(entry)

                    export_wireshark_tooljson(
                        tshark_path=Path(tshark_path),
                        pcap_path=pcap_path,
                        filter_name=ws_filter,
                        out_dir=tmp_dir,
                        export_jsonraw=True,
                        export_texttree=False,
                        max_packets=0,
                        extra_tshark_args=extra_args,
                        codegen_profile=codegen_profile,
                        ws_isolate_plugins=bool(getattr(args, "stage_c_ws_isolate_plugins", True)),
                        ws_plugin_id=ws_plugin_id,
                    )

                else:
                    raise RuntimeError(f"normal dataset parse summary unsupported host={args.stage_c_host}")

                tooljson_path = tmp_dir / "tool.json"
                try:
                    tool_payload = json.loads(tooljson_path.read_text(encoding="utf-8"))
                except Exception:
                    tool_payload = []

                if not isinstance(tool_payload, list):
                    tool_payload = []

                exported = len(tool_payload)
                for pkt in tool_payload:
                    dbg = pkt.get("debug") if isinstance(pkt, dict) else None
                    dbg = dbg if isinstance(dbg, dict) else {}
                    rc = dbg.get("rc")
                    err = dbg.get("error")

                    parse_ok = True
                    rc_int: Optional[int] = None
                    if rc is not None:
                        try:
                            rc_int = int(rc)
                            if rc_int != 0:
                                parse_ok = False
                        except Exception:
                            parse_ok = False
                    if err:
                        parse_ok = False

                    if parse_ok:
                        ok += 1
                    else:
                        fail += 1
                        if rc_int == -1:
                            timeouts += 1
                        if len(error_samples) < 5:
                            error_samples.append(
                                {
                                    "packet_no": pkt.get("packet_no") if isinstance(pkt, dict) else None,
                                    "rc": rc,
                                    "error": err,
                                    "stderr": dbg.get("stderr"),
                                }
                            )

                # If GT exists, also compute dataset-level metrics (L1/L2/etc) without writing
                # the full Stage C tool_<host>.json to disk.
                print("\n[Stage C] Dataset metrics (normal dataset; temporary tool.json)")
                dataset_metrics = _maybe_compute_dataset_metrics(pred_tooljson=tooljson_path)

            summary_payload.update(
                {
                    "exported_packets": exported,
                    "parse_ok": ok,
                    "parse_fail": fail,
                    "parse_timeouts": timeouts,
                    "error_samples": error_samples,
                    "dataset_metrics": dataset_metrics,
                }
            )
            summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"  exported={exported} parse_ok={ok} parse_fail={fail} timeouts={timeouts} "
                f"(wrote {summary_path})"
            )
            if not bool(args.stage_c_pcap_to_texttree):
                print("  Full tool.json export is disabled (--no-stage-c-pcap-to-texttree).")
                print("  Re-run without it to save tool_<host>.txt/.json under outputs/stage_c/.")
        except Exception as exc:
            summary_payload["error"] = f"{type(exc).__name__}: {exc}"
            try:
                summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass
            print(f"  [Stage C] normal dataset parse summary failed: {exc}")
        finally:
            normal_dataset_parse_summary_done = True

    def _run_exp_constraint_verify_once(*, reason: str) -> None:
        nonlocal exp_constraints_result, generated_exp_constraint_suite
        if not exp_constraints_enabled:
            return
        if exp_constraints_result is not None:
            return

        out_dir = stage_c_out_dir / "exp_constraint" / args.stage_c_host
        out_dir.mkdir(parents=True, exist_ok=True)
        suite_dir = paths.inputs_dir / "exp_constraint"
        mapping_path: Optional[Path] = None
        pkg_suite_unavailable_reason: Optional[str] = None
        pkg_generation_complete: Optional[bool] = None
        pkg_generation_incomplete_reasons: Tuple[str, ...] = ()
        pkg_preflight_issues: list[str] = []
        suite_source = str(getattr(args, "stage_c_exp_constraint_suite_source", "pkg") or "pkg").strip().lower()
        if bool(getattr(args, "stage_c_exp_constraint_generate_from_traffic", False)):
            suite_source = "traffic"
        if suite_source == "traffic":
            if format_tree_path is None:
                print("\n[Stage C] traffic-derived exp_constraint suite skipped (tree-free Stage C mode)")
            else:
                traces_path = paths.step2_results_dir / "traffic_parsing_traces.jsonl"
                if not traces_path.exists():
                    print(f"\n[Stage C] traffic-derived exp_constraint suite skipped (no traces at {traces_path})")
                else:
                    generated_suite_dir = (
                        Path(args.stage_c_exp_constraint_generated_suite_dir)
                        if getattr(args, "stage_c_exp_constraint_generated_suite_dir", None) is not None
                        else (stage_c_out_dir / "exp_constraint_generated_suite")
                    )
                    try:
                        print(f"\n[Stage C] generating traffic-derived exp_constraint suite (reason={reason})")
                        from pargen.parser_realization.traffic_exp_constraint_generator import (
                            generate_exp_constraint_suite_from_traffic,
                        )

                        generated_exp_constraint_suite = generate_exp_constraint_suite_from_traffic(
                            protocol=args.protocol,
                            tree_path=Path(format_tree_path),
                            traces_path=traces_path,
                            base_pcap=pcap_path,
                            out_dir=generated_suite_dir,
                            codegen_profile=codegen_profile,
                            max_negatives_per_constraint=int(
                                getattr(args, "stage_c_exp_constraint_max_negatives_per_constraint", 1) or 1
                            ),
                        )
                        if int(getattr(generated_exp_constraint_suite, "cases_generated", 0) or 0) > 0:
                            suite_dir = Path(generated_exp_constraint_suite.suite_dir)
                            mapping_path = Path(generated_exp_constraint_suite.mapping_path)
                            print(
                                "  generated suite: "
                                f"cases={generated_exp_constraint_suite.cases_generated} "
                                f"constraints_seen={generated_exp_constraint_suite.constraints_seen_in_traffic} "
                                f"dir={suite_dir}"
                            )
                        else:
                            print("  generated suite is empty; falling back to static suite if available")
                    except Exception as exc:
                        print(f"  traffic-derived exp_constraint suite generation failed: {type(exc).__name__}: {exc}")
        elif suite_source == "pkg":
            if format_tree_path is None:
                print("\n[Stage C] PKG-derived exp_constraint suite skipped (tree-free Stage C mode)")
                pkg_suite_unavailable_reason = "tree-free Stage C mode"
            else:
                generated_suite_dir = (
                    Path(args.stage_c_exp_constraint_generated_suite_dir)
                    if getattr(args, "stage_c_exp_constraint_generated_suite_dir", None) is not None
                    else (stage_c_out_dir / "exp_constraint_generated_suite")
                )
                try:
                    print(f"\n[Stage C] generating PKG-derived exp_constraint suite (reason={reason})")
                    from pargen.parser_realization.pkg_exp_constraint_suite import (
                        generate_exp_constraint_suite_from_pkg_paths,
                    )

                    generated_exp_constraint_suite = generate_exp_constraint_suite_from_pkg_paths(
                        protocol=args.protocol,
                        tree_path=Path(format_tree_path),
                        out_dir=generated_suite_dir,
                        codegen_profile=pkg_codegen_profile,
                        max_negatives_per_positive=int(
                            getattr(args, "stage_c_exp_constraint_max_negatives_per_constraint", 1) or 1
                        ),
                        max_paths=getattr(args, "stage_c_exp_constraint_pkg_max_paths", None),
                    )
                    raw_generation_complete = getattr(generated_exp_constraint_suite, "generation_complete", None)
                    pkg_generation_complete = (
                        raw_generation_complete if isinstance(raw_generation_complete, bool) else None
                    )
                    raw_incomplete_reasons = getattr(
                        generated_exp_constraint_suite,
                        "generation_incomplete_reasons",
                        (),
                    )
                    if isinstance(raw_incomplete_reasons, (list, tuple)):
                        pkg_generation_incomplete_reasons = tuple(
                            str(item) for item in raw_incomplete_reasons if str(item).strip()
                        )
                    if pkg_generation_complete is not True:
                        if pkg_generation_complete is None:
                            pkg_preflight_issues.append("generation completeness provenance is unknown")
                        elif pkg_generation_incomplete_reasons:
                            pkg_preflight_issues.append(
                                "generation is incomplete: " + "; ".join(pkg_generation_incomplete_reasons)
                            )
                        else:
                            pkg_preflight_issues.append("generation is incomplete")
                    if int(getattr(generated_exp_constraint_suite, "cases_generated", 0) or 0) > 0:
                        suite_dir = Path(generated_exp_constraint_suite.suite_dir)
                        mapping_path = Path(generated_exp_constraint_suite.mapping_path)
                        print(
                            "  generated PKG suite: "
                            f"cases={generated_exp_constraint_suite.cases_generated} "
                            f"positives={generated_exp_constraint_suite.positive_samples_usable} "
                            f"skipped_paths={generated_exp_constraint_suite.skipped_paths} "
                            f"dir={suite_dir}"
                        )
                    else:
                        pkg_suite_unavailable_reason = (
                            "tree-derived suite has no replay-valid counterexamples "
                            f"(paths={getattr(generated_exp_constraint_suite, 'paths_total', 0)}, "
                            f"skipped={getattr(generated_exp_constraint_suite, 'skipped_paths', 0)})"
                        )
                        print(f"  generated PKG suite is empty; {pkg_suite_unavailable_reason}")
                except Exception as exc:
                    pkg_suite_unavailable_reason = f"generation failed: {type(exc).__name__}: {exc}"
                    print(f"  PKG-derived exp_constraint suite generation failed: {type(exc).__name__}: {exc}")

        pkg_preflight_payload = {
            "generation_complete": pkg_generation_complete,
            "generation_incomplete_reasons": list(pkg_generation_incomplete_reasons),
            "issues": list(pkg_preflight_issues),
        }

        def _record_pkg_runtime_unavailable(unavailable_reason: str) -> None:
            nonlocal exp_constraints_result
            exp_constraints_result = False
            if suite_source != "pkg":
                return
            unavailable_summary_path = out_dir / "summary.json"
            unavailable_summary_path.write_text(
                json.dumps(
                    {
                        "protocol": args.protocol,
                        "host": args.stage_c_host,
                        "mode": str(getattr(args, "stage_c_exp_constraint_mode", "binary") or "binary"),
                        "status": "unavailable",
                        "success": 0,
                        "fail": 1,
                        "total": 0,
                        "coverage": 0.0,
                        "reason": unavailable_reason,
                        "runner_executed": False,
                        "static_fallback_used": False,
                        "pkg_preflight": pkg_preflight_payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if bool(getattr(args, "stage_c_exp_constraint_strict", False)):
                raise SystemExit(
                    "Stage C PKG exp_constraint verification unavailable: "
                    f"{unavailable_reason}"
                )
        if suite_source == "pkg" and pkg_suite_unavailable_reason is None and pkg_preflight_issues:
            print("  PKG conformance preflight is partial:")
            for issue in pkg_preflight_issues:
                print(f"    - {issue}")
            if bool(getattr(args, "stage_c_exp_constraint_strict", False)):
                incomplete_summary_path = out_dir / "summary.json"
                incomplete_summary_path.write_text(
                    json.dumps(
                        {
                            "protocol": args.protocol,
                            "host": args.stage_c_host,
                            "mode": str(getattr(args, "stage_c_exp_constraint_mode", "binary") or "binary"),
                            "status": "incomplete",
                            "success": 0,
                            "fail": 1,
                            "total": 0,
                            "coverage": 0.0,
                            "reason": "; ".join(pkg_preflight_issues),
                            "runner_executed": False,
                            "static_fallback_used": False,
                            "pkg_preflight": pkg_preflight_payload,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise SystemExit(
                    "Stage C PKG strict completeness/provenance preflight failed: "
                    + "; ".join(pkg_preflight_issues)
                )

        if suite_source == "pkg" and pkg_suite_unavailable_reason is not None:
            exp_constraints_result = False
            unavailable_summary_path = out_dir / "summary.json"
            unavailable_summary_path.write_text(
                json.dumps(
                    {
                        "protocol": args.protocol,
                        "host": args.stage_c_host,
                        "mode": str(getattr(args, "stage_c_exp_constraint_mode", "binary") or "binary"),
                        "status": "unavailable",
                        "success": 0,
                        "fail": 1,
                        "total": 0,
                        "coverage": 0.0,
                        "reason": pkg_suite_unavailable_reason,
                        "static_fallback_used": False,
                        "pkg_preflight": pkg_preflight_payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                "\n[Stage C] PKG exp_constraint verification skipped; "
                f"static/dataset fallback is forbidden ({pkg_suite_unavailable_reason}; wrote {unavailable_summary_path})"
            )
            if bool(getattr(args, "stage_c_exp_constraint_strict", False)):
                raise SystemExit(
                    "Stage C PKG exp_constraint verification unavailable: "
                    f"{pkg_suite_unavailable_reason}"
                )
            _run_normal_dataset_parse_summary(reason="post-stage-c")
            return

        if not suite_dir.exists():
            unavailable_reason = f"suite directory is missing: {suite_dir}"
            _record_pkg_runtime_unavailable(unavailable_reason)
            print(f"\n[Stage C] exp_constraint verification skipped ({unavailable_reason})")
            _run_normal_dataset_parse_summary(reason="post-stage-c")
            return

        # Host-specific runtime requirements.
        tshark_path: Optional[Path] = None
        lua_entry_path: Optional[Path] = None
        plugin_dir_for_suite: Optional[Path] = plugin_dir

        if args.stage_c_host == "wireshark_c":
            if args.stage_c_tshark is not None:
                tshark_path = _find_tshark(ws_build_dir, args.stage_c_tshark)
            else:
                # Prefer an existing Wireshark build if available; only rebuild when the plugin library
                # is missing (keeps "reuse Stage C outputs" fast).
                try:
                    tshark_path = _find_tshark(ws_build_dir)
                except Exception:
                    tshark_path = None
                if tshark_path is None:
                    tshark_path = _ensure_tshark(ws_build_dir, logs_root=paths.stage_c_logs_dir)

                needs_plugin_build = False
                if ws_plugin_needs_rebuild:
                    needs_plugin_build = True
                if bool(getattr(args, "stage_c_ws_isolate_plugins", True)) and plugin_id:
                    try:
                        from pargen.parser_realization.target_platforms.wireshark_export import _ws_find_plugin_library

                        run_plugins_root = Path(tshark_path).parent / "plugins"
                        lib = _ws_find_plugin_library(run_plugins_root=run_plugins_root, plugin_id=str(plugin_id))
                        if lib is None:
                            needs_plugin_build = True
                    except Exception:
                        needs_plugin_build = True

                if needs_plugin_build:
                    _build_ws_with_plugin(reason=f"{reason} requires plugin build")
                    if build_result and getattr(build_result, "artifacts", None) and getattr(build_result.artifacts, "tool_bin", None):
                        tshark_path = Path(build_result.artifacts.tool_bin)
                    if tshark_path is None:
                        tshark_path = _ensure_tshark(ws_build_dir, logs_root=paths.stage_c_logs_dir)
        elif args.stage_c_host == "wireshark_lua":
            lua_entry_path = _get_lua_entry()
            if args.stage_c_tshark is not None:
                tshark_path = Path(args.stage_c_tshark)
            else:
                tshark_path_str = shutil.which("tshark")
                if not tshark_path_str:
                    unavailable_reason = "tshark not found for wireshark_lua"
                    _record_pkg_runtime_unavailable(unavailable_reason)
                    print(f"\n[Stage C] exp_constraint verification skipped ({unavailable_reason}).")
                    return
                tshark_path = Path(tshark_path_str)
        elif args.stage_c_host == "scapy":
            # scapy suite does not require tshark.
            pass
        elif args.stage_c_host == "zeek_spicy":
            # zeek_spicy suite uses spicy-dump exporter and does not require tshark.
            pass
        else:
            unavailable_reason = f"unsupported host={args.stage_c_host}"
            _record_pkg_runtime_unavailable(unavailable_reason)
            print(f"\n[Stage C] exp_constraint verification skipped ({unavailable_reason})")
            return

        from pargen.parser_realization.exp_constraint_suite import _default_mapping_path, run_exp_constraint_suite

        resolved_mapping_path: Optional[Path] = mapping_path
        if resolved_mapping_path is None:
            try:
                resolved_mapping_path = _default_mapping_path(protocol=args.protocol, suite_dir=suite_dir)
            except Exception:
                resolved_mapping_path = None
        if generated_exp_constraint_suite is None and resolved_mapping_path is not None and format_tree_path is not None:
            matches_tree, mismatch_reason = _exp_constraint_mapping_matches_tree(
                resolved_mapping_path,
                Path(format_tree_path),
            )
            if not matches_tree:
                exp_constraints_result = False
                print(
                    "\n[Stage C] exp_constraint verification skipped "
                    f"(mapping/tree mismatch: {mismatch_reason}; mapping={resolved_mapping_path})"
                )
                _run_normal_dataset_parse_summary(reason="post-stage-c")
                return
        mapping_path = resolved_mapping_path

        def _mapping_has_positive_seed(payload: Any) -> bool:
            if not isinstance(payload, dict):
                return False
            try:
                if int(payload.get("seed_frame_no") or 0) > 0:
                    return True
            except Exception:
                pass
            cases = payload.get("cases")
            if not isinstance(cases, dict):
                return False
            for case in cases.values():
                if not isinstance(case, dict):
                    continue
                try:
                    if int(case.get("seed_frame_no") or 0) > 0:
                        return True
                except Exception:
                    continue
            return False

        def _mapping_with_default_positive_seed(path: Optional[Path]) -> Optional[Path]:
            if str(getattr(args, "stage_c_exp_constraint_mode", "binary") or "binary") != "binary":
                return path
            resolved = path
            if resolved is None:
                try:
                    resolved = _default_mapping_path(protocol=args.protocol, suite_dir=suite_dir)
                except Exception:
                    resolved = None
            if resolved is None or not Path(resolved).exists():
                return path
            try:
                payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
            except Exception:
                return path
            if not isinstance(payload, dict) or not payload.get("base_pcap"):
                return resolved
            if _mapping_has_positive_seed(payload):
                return resolved
            seed_frame_no = int(getattr(args, "stage_c_exp_constraint_default_seed_frame_no", 1) or 1)
            if seed_frame_no <= 0:
                return resolved
            patched = dict(payload)
            patched["seed_frame_no"] = seed_frame_no
            patched_path = out_dir / "mapping_with_default_positive_seed.json"
            patched_path.write_text(json.dumps(patched, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                "  exp_constraint: added default positive seed "
                f"frame={seed_frame_no} for runtime confusion ({patched_path})"
            )
            return patched_path

        mapping_path = _mapping_with_default_positive_seed(mapping_path)

        def _exp_constraint_source_globs() -> Tuple[str, ...]:
            if args.stage_c_host == "wireshark_c":
                return ("*.c", "*.h")
            if args.stage_c_host == "wireshark_lua":
                return ("*.lua",)
            if args.stage_c_host == "scapy":
                return ("*.py",)
            if args.stage_c_host == "zeek_spicy":
                return ("**/*.spicy", "**/*.evt", "**/*.zeek", "**/*.sig", "**/*.meta")
            return ("*",)

        def _scan_exp_constraint_sources() -> dict[str, str]:
            files: dict[str, str] = {}
            if plugin_dir_for_suite is None or not Path(plugin_dir_for_suite).exists():
                return files
            for pattern in _exp_constraint_source_globs():
                for path in sorted(Path(plugin_dir_for_suite).glob(pattern)):
                    if not path.is_file():
                        continue
                    try:
                        rel = path.relative_to(Path(plugin_dir_for_suite)).as_posix()
                        files[rel] = path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
            return files

        def _write_exp_constraint_sources(patched: dict[str, str]) -> None:
            if plugin_dir_for_suite is None:
                raise RuntimeError("plugin_dir is unavailable for exp_constraint repair")
            root = Path(plugin_dir_for_suite)
            for name, content in patched.items():
                target = (root / name).resolve()
                try:
                    target.relative_to(root.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"refusing to write outside plugin_dir: {name}") from exc
                target.write_text(content, encoding="utf-8")

        def _summarize_backend_contract_findings(
            sources: Optional[dict[str, str]] = None,
            *,
            max_items: int = 12,
        ) -> tuple[tuple[str, ...], str]:
            try:
                from pargen.parser_realization.backend_contract import (
                    format_backend_contract_violations,
                    validate_backend_realization,
                )
            except Exception:
                return tuple(), ""

            srcs = sources if sources is not None else _scan_exp_constraint_sources()
            if not srcs:
                return tuple(), ""
            try:
                violations = validate_backend_realization(
                    str(args.stage_c_host),
                    srcs,
                    max_violations=max(1, int(max_items) * 2),
                )
            except Exception:
                return tuple(), ""
            if not violations:
                return tuple(), ""
            errors = tuple(v.format_for_repair() for v in list(violations)[: max(0, int(max_items))])
            summary = format_backend_contract_violations(violations, max_items=max_items)
            return errors, summary

        def _summarize_exp_constraint_mapping(
            target_ids: set[str],
            *,
            suite_res: Optional[Any] = None,
            max_items: int = 20,
        ) -> str:
            if not target_ids:
                return ""
            raw_mapping = None
            suite_mapping = None
            if suite_res is not None and getattr(suite_res, "mapping_path", None):
                suite_mapping = Path(str(getattr(suite_res, "mapping_path")))
            for candidate in (mapping_path, suite_mapping):
                if candidate is None:
                    continue
                try:
                    if Path(candidate).exists():
                        raw_mapping = json.loads(Path(candidate).read_text(encoding="utf-8", errors="ignore"))
                        break
                except Exception:
                    continue
            if not isinstance(raw_mapping, dict):
                return ""
            catalog = raw_mapping.get("constraints_from_tree")
            if not isinstance(catalog, dict):
                catalog = raw_mapping.get("constraints_catalog")
            if not isinstance(catalog, dict):
                catalog = raw_mapping.get("derived_constraints")
            if not isinstance(catalog, dict):
                return ""
            lines = ["Constraint specs referenced by failed cases:"]
            for cid in sorted(target_ids)[: max(0, int(max_items))]:
                spec = catalog.get(cid)
                if spec is None:
                    continue
                try:
                    spec_s = json.dumps(spec, ensure_ascii=False, sort_keys=True)
                except Exception:
                    spec_s = str(spec)
                if len(spec_s) > 1200:
                    spec_s = spec_s[:1200] + "...(truncated)"
                lines.append(f"- {cid}: {spec_s}")
            return "\n".join(lines) if len(lines) > 1 else ""

        def _generated_suite_summary_text() -> str:
            if generated_exp_constraint_suite is None:
                return ""
            try:
                return json.dumps(generated_exp_constraint_suite.to_dict(), ensure_ascii=False, indent=2)
            except Exception:
                return str(generated_exp_constraint_suite)

        def _summarize_positive_seed_failures(suite_res: Any, *, max_cases: int) -> tuple[tuple[str, ...], str]:
            errors: list[str] = []
            for cr in list(getattr(suite_res, "positive_results", ()) or ()):
                if getattr(cr, "ok", False):
                    continue
                if len(errors) >= max_cases:
                    break
                errors.append(
                    "backend-realization positive seed failure: "
                    f"case={getattr(cr, 'case', '')} frame={getattr(cr, 'frame_no', '')} "
                    f"reason={getattr(cr, 'reason', '')}"
                )
            tail_parts: list[str] = []
            suite_summary = _generated_suite_summary_text()
            if suite_summary:
                tail_parts.append("Generated conformance-counterexample suite summary:\n" + suite_summary)
            return tuple(errors), "\n\n".join(tail_parts)

        def _summarize_negative_constraint_failures(suite_res: Any, *, max_cases: int) -> tuple[tuple[str, ...], str]:
            errors: list[str] = []
            target_ids: set[str] = set()
            for cr in list(getattr(suite_res, "results", ()) or ()):
                if getattr(cr, "ok", False):
                    continue
                ids = tuple(str(x) for x in (getattr(cr, "target_constraints", ()) or ()))
                target_ids.update(ids)
                if len(errors) < max_cases:
                    errors.append(
                        "constraint-violating sample was accepted or under-reported: "
                        f"case={getattr(cr, 'case', '')} frame={getattr(cr, 'frame_no', '')} "
                        f"constraints={list(ids)} reason={getattr(cr, 'reason', '')}"
                        f" observations={json.dumps(getattr(cr, 'observations', None), ensure_ascii=False)}"
                    )
            mapping_summary = _summarize_exp_constraint_mapping(target_ids, suite_res=suite_res)
            suite_summary = _generated_suite_summary_text()
            tail_parts = []
            if suite_summary:
                tail_parts.append("Generated conformance-counterexample suite summary:\n" + suite_summary)
            if mapping_summary:
                tail_parts.append(mapping_summary)
            return tuple(errors), "\n\n".join(tail_parts)

        def _qualified_for_repair(suite_res: Any) -> Any:
            """Drop counterexamples whose intended violation was not independently qualified (PKG suite)."""
            if suite_source != "pkg":
                return suite_res
            from pargen.parser_realization.conformance_observations import evaluation_gaps
            unqualified = {
                gap.get("case") for gap in evaluation_gaps(suite_res) if gap.get("target_constraints")
            }
            if not unqualified:
                return suite_res
            kept = [case for case in suite_res.results if getattr(case, "case", None) not in unqualified]
            import copy as _copy
            import dataclasses as _dc
            if _dc.is_dataclass(suite_res) and not isinstance(suite_res, type):
                try:
                    return _dc.replace(suite_res, results=kept)
                except Exception:
                    pass
            view = _copy.copy(suite_res)
            try:
                view.results = kept
            except Exception:
                return suite_res
            return view

        def _select_exp_constraint_repair_feedback(
            suite_res: Any,
            *,
            max_cases: int,
            gate: bool,
            field_res: Any = None,
        ) -> tuple[str, tuple[str, ...], str]:
            """
            Keep Stage C repair phases separated:
            1) backend realization / positive seed stability
            2) IR constraint realization on already-stable backend code

            This prevents the LLM from satisfying counterexamples by broad rejection
            or backend-breaking changes that regress positive seed parsing.
            """
            backend_errors, backend_summary = _summarize_backend_contract_findings(max_items=max_cases)
            positive_errors, positive_tail = _summarize_positive_seed_failures(suite_res, max_cases=max_cases)
            backend_phase_errors = list(backend_errors)
            for item in positive_errors:
                if len(backend_phase_errors) >= max_cases:
                    break
                backend_phase_errors.append(item)
            if backend_phase_errors:
                tail_parts = []
                if backend_summary:
                    tail_parts.append(backend_summary)
                if positive_tail:
                    tail_parts.append(positive_tail)
                phase = "backend_gate" if gate else "backend_realization"
                return phase, tuple(backend_phase_errors), "\n\n".join(tail_parts)

            constraint_errors, constraint_tail = _summarize_negative_constraint_failures(
                _qualified_for_repair(suite_res), max_cases=max_cases
            )
            if field_res is not None and getattr(field_res, "count", 0):
                # Localize(Q, G): field conflicts carry their PKG identity and rules; validity conflicts their rule ids.
                from pargen.parser_realization.pkg_field_conformance import describe_field_conflicts
                field_lines = describe_field_conflicts(field_res, _pkg_tree_payload(), max_items=max_cases)
                phase = "conformance_gate" if gate else "conformance"
                return phase, tuple(field_lines) + tuple(constraint_errors), constraint_tail
            phase = "constraint_gate" if gate else "constraint_realization"
            return phase, constraint_errors, constraint_tail

        def _exp_constraint_improved(old_res: Any, new_res: Any) -> bool:
            accepted, _reason = _stage_c_exp_constraint_gate_accepts(old_res, new_res)
            return accepted

        def _run_exp_constraint_suite_round(round_idx: int) -> Any:
            print(f"\n[Stage C] exp_constraint verification (reason={reason}, round={round_idx})")
            with _record_stage_time(timing, "stage_c_exp_constraint_s"):
                return run_exp_constraint_suite(
                    protocol=args.protocol,
                    host=args.stage_c_host,
                    mode=str(getattr(args, "stage_c_exp_constraint_mode", "binary") or "binary"),
                    suite_dir=suite_dir,
                    mapping_path=mapping_path,
                    tshark_path=tshark_path,
                    plugin_id=str(plugin_id) if plugin_id else None,
                    stage_c_out_dir=stage_c_out_dir,
                    lua_entry=lua_entry_path,
                    scapy_entry=getattr(args, "stage_c_scapy_entry", None),
                    plugin_dir=plugin_dir_for_suite,
                    filter_name=ws_filter,
                    ws_isolate_plugins=bool(getattr(args, "stage_c_ws_isolate_plugins", True)),
                    # zeek_spicy can take longer per case (spicy-dump JIT); show progress by default.
                    verbose=bool(args.stage_c_host == "zeek_spicy"),
                    packet_logs_dir=out_dir / f"round_{round_idx:02d}_packets",
                    binary_positive_policy=str(
                        getattr(args, "stage_c_exp_constraint_binary_positive_policy", "unique_seed")
                        or "unique_seed"
                    ),
                    binary_require_bs_constraint=bool(
                        getattr(args, "stage_c_exp_constraint_binary_require_bs_constraint", False)
                    ),
                )

        def _validate_exp_constraint_repair_candidate(reason: str) -> None:
            nonlocal tshark_path
            action = _exp_constraint_repair_validation_action(args.stage_c_host)
            if action == "wireshark_build":
                _build_ws_with_plugin(reason=reason)
                if build_result and getattr(build_result, "artifacts", None) and getattr(build_result.artifacts, "tool_bin", None):
                    tshark_path = Path(build_result.artifacts.tool_bin)
            elif action == "lua_syntax":
                _run_lua_syntax_check(reason=reason)
            elif action == "scapy_syntax":
                _run_scapy_syntax_check(reason=reason)
            elif action == "zeek_spicy_syntax":
                _run_zeek_spicy_syntax_check(reason=reason)

        def _append_pkg_realization_trace(row: Mapping[str, Any]) -> None:
            trace_path = out_dir / "pkg_realization_trace.tsv"
            fields = [
                "round",
                "phase",
                "compile_load_issues",
                "syntax_issues",
                "semantic_issues",
                "constraint_issues",
                "remaining_issues",
                "l1_f1",
                "l2_f1",
                "l2_aligned_f1",
                "export_error",
                "trace_dir",
            ]
            if not trace_path.exists():
                trace_path.write_text("\t".join(fields) + "\n", encoding="utf-8")
            values = []
            for key in fields:
                val = row.get(key, "")
                if val is None:
                    val = ""
                values.append(str(val).replace("\t", " ").replace("\n", " "))
            with trace_path.open("a", encoding="utf-8") as f:
                f.write("\t".join(values) + "\n")

        def _run_pkg_realization_check(round_idx: int, *, phase: str, suite_res: Optional[Any]) -> None:
            # Code refinement with the PKG message suite uses only messages
            # constructed from the PKG.
            if pkg_tree_only_mode:
                return
            if format_tree_path is None or not Path(format_tree_path).exists() or not pcap_path.exists():
                return

            trace_dir = out_dir / "pkg_realization" / f"round_{round_idx:02d}_{phase}"
            trace_dir.mkdir(parents=True, exist_ok=True)

            export_error = ""
            compile_load_issues = 0
            metrics_payload: dict[str, Any] = {}
            syntax_issues = 0
            semantic_issues = 0
            l1_f1 = 0.0
            l2_f1 = 0.0
            l2_aligned_f1 = 0.0

            try:
                if args.stage_c_host == "wireshark_lua":
                    entry = lua_entry_path or _get_lua_entry()
                    local_tshark_path = Path(tshark_path) if tshark_path is not None else None
                    if local_tshark_path is None:
                        tshark_path_str = shutil.which("tshark")
                        if not tshark_path_str:
                            raise RuntimeError("tshark not found for wireshark_lua realization check")
                        local_tshark_path = Path(tshark_path_str)
                    export_result = export_wireshark_tooljson(
                        tshark_path=local_tshark_path,
                        pcap_path=pcap_path,
                        filter_name=ws_filter,
                        out_dir=trace_dir,
                        export_jsonraw=True,
                        export_texttree=True,
                        max_packets=int(getattr(args, "stage_c_semantic_max_packets", 0) or 0),
                        extra_tshark_args=_lua_tshark_extra_args(entry),
                        codegen_profile=codegen_profile,
                        ws_isolate_plugins=bool(getattr(args, "stage_c_ws_isolate_plugins", True)),
                        ws_plugin_id=None,
                    )
                elif args.stage_c_host == "wireshark_c":
                    if tshark_path is None:
                        raise RuntimeError("tshark path missing for wireshark_c realization check")
                    export_result = export_wireshark_tooljson(
                        tshark_path=Path(tshark_path),
                        pcap_path=pcap_path,
                        filter_name=ws_filter,
                        out_dir=trace_dir,
                        export_jsonraw=True,
                        export_texttree=True,
                        max_packets=int(getattr(args, "stage_c_semantic_max_packets", 0) or 0),
                        extra_tshark_args=_wireshark_c_tshark_extra_args(),
                        codegen_profile=codegen_profile,
                        ws_isolate_plugins=bool(getattr(args, "stage_c_ws_isolate_plugins", True)),
                        ws_plugin_id=(str(plugin_id) if plugin_id else None),
                    )
                elif args.stage_c_host == "scapy":
                    export_result = export_scapy_tooljson(
                        plugin_src_dir=plugin_dir,
                        pcap_path=pcap_path,
                        filter_name=ws_filter,
                        out_dir=trace_dir,
                        entry_override=getattr(args, "stage_c_scapy_entry", None),
                        ports=_parse_scapy_ports_arg(),
                        transport=getattr(args, "stage_c_scapy_transport", "auto"),
                        max_packets=int(getattr(args, "stage_c_semantic_max_packets", 0) or 0),
                        include_payload_layers=True,
                        codegen_profile=codegen_profile,
                    )
                else:
                    return

                log_path = trace_dir / "tshark_export.log"
                if log_path.exists():
                    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
                    if any(tok in log_text for tok in ("Lua Error", "Traceback", "Dissector bug", "Range is out of bounds")):
                        compile_load_issues = 1

                from pargen.parser_realization.semantic_stage.validator import run_semantic_validation
                from pargen.parser_realization.target_platforms.extraction_profile import infer_ip_proto, infer_pcap_layer

                host_key = str(args.stage_c_host or "")
                pcap_layer = infer_pcap_layer(codegen_profile, host=host_key)
                pcap_ip_proto = infer_ip_proto(codegen_profile, host=host_key) if pcap_layer == "ip_payload" else None
                validation_dir = trace_dir / "pkg_reference"
                validation = run_semantic_validation(
                    tree_path=Path(format_tree_path),
                    pred_tooljson_path=Path(export_result.tooljson_path),
                    pcap_path=pcap_path,
                    pcap_layer=pcap_layer,
                    pcap_ip_proto=pcap_ip_proto,
                    frame_index=semantic_frames,
                    out_dir=validation_dir,
                    require_full_parse=bool(getattr(args, "stage_c_semantic_require_full_parse", True)),
                    max_packets=int(getattr(args, "stage_c_semantic_max_packets", 0) or 0),
                    worst_k=int(getattr(args, "stage_c_exp_constraint_fix_worst_k", 8) or 8),
                    min_f1_l1=0.0,
                )
                if validation.metrics_json and Path(validation.metrics_json).exists():
                    metrics_payload = json.loads(Path(validation.metrics_json).read_text(encoding="utf-8"))
                    syntax_issues = int(metrics_payload.get("fp_level1", 0) or 0) + int(
                        metrics_payload.get("fn_level1", 0) or 0
                    )
                    semantic_issues = int(
                        metrics_payload.get("name_mismatch_level2_aligned")
                        or metrics_payload.get("mismatch_level2_semantic_only_exact")
                        or 0
                    )
                    l1_f1 = float(metrics_payload.get("field_f1_level1", 0.0) or 0.0)
                    l2_f1 = float(metrics_payload.get("field_f1_level2", 0.0) or 0.0)
                    l2_aligned_f1 = float(metrics_payload.get("field_f1_level2_aligned", l2_f1) or 0.0)
            except Exception as exc:
                compile_load_issues = 1
                export_error = f"{type(exc).__name__}: {exc}"
                try:
                    (trace_dir / "realization_error.txt").write_text(export_error + "\n", encoding="utf-8")
                except Exception:
                    pass

            constraint_issues = int(getattr(suite_res, "fail", 0) or 0) if suite_res is not None else 0
            positive_seed_issues = int(getattr(suite_res, "fn", 0) or 0) if suite_res is not None else 0
            remaining_issues = int(compile_load_issues) + int(syntax_issues) + int(semantic_issues) + int(
                constraint_issues
            ) + int(positive_seed_issues)
            row = {
                "round": int(round_idx),
                "phase": str(phase),
                "compile_load_issues": int(compile_load_issues),
                "syntax_issues": int(syntax_issues),
                "semantic_issues": int(semantic_issues),
                "constraint_issues": int(constraint_issues),
                "positive_seed_issues": int(positive_seed_issues),
                "remaining_issues": int(remaining_issues),
                "l1_f1": f"{float(l1_f1):.6f}",
                "l2_f1": f"{float(l2_f1):.6f}",
                "l2_aligned_f1": f"{float(l2_aligned_f1):.6f}",
                "export_error": export_error,
                "trace_dir": str(trace_dir),
            }
            (trace_dir / "pkg_realization_summary.json").write_text(
                json.dumps(
                    {
                        **row,
                        "metrics": metrics_payload,
                        "constraint_summary": {
                            "success": int(getattr(suite_res, "success", 0) or 0) if suite_res is not None else None,
                            "fail": int(getattr(suite_res, "fail", 0) or 0) if suite_res is not None else None,
                            "total": int(getattr(suite_res, "total", 0) or 0) if suite_res is not None else None,
                            "coverage": float(getattr(suite_res, "coverage", 0.0) or 0.0)
                            if suite_res is not None
                            else None,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            _append_pkg_realization_trace(row)
            print(
                "  pkg-realization trace: "
                f"round={round_idx} issues={remaining_issues} "
                f"(compile={compile_load_issues}, syntax={syntax_issues}, "
                f"semantic={semantic_issues}, constraint={constraint_issues}, "
                f"positive_seed={positive_seed_issues}) "
                f"L1={l1_f1:.4f} L2={l2_f1:.4f}"
            )

        _pkg_tree_cache: dict[str, Any] = {}

        def _pkg_tree_payload() -> Any:
            if "tree" not in _pkg_tree_cache:
                _pkg_tree_cache["tree"] = json.loads(Path(format_tree_path).read_text(encoding="utf-8"))
            return _pkg_tree_cache["tree"]

        def _export_parser_tooljson(pcap: Path, export_dir: Path, *, frames: Sequence[int]) -> Path:
            """Platform adapter: the generated parser's field output on `pcap` as tool.json."""
            host = str(args.stage_c_host or "")
            ws_isolate = bool(getattr(args, "stage_c_ws_isolate_plugins", True))
            if host in {"wireshark_lua", "wireshark_c"}:
                local_tshark = Path(tshark_path) if tshark_path is not None else None
                if local_tshark is None and host == "wireshark_lua" and shutil.which("tshark"):
                    local_tshark = Path(str(shutil.which("tshark")))
                if local_tshark is None:
                    raise RuntimeError(f"tshark not available for the {host} field check")
                extra = (_lua_tshark_extra_args(lua_entry_path or _get_lua_entry()) if host == "wireshark_lua"
                         else _wireshark_c_tshark_extra_args())
                exported = export_wireshark_tooljson(
                    tshark_path=local_tshark, pcap_path=pcap, filter_name=ws_filter, out_dir=export_dir,
                    export_jsonraw=True, export_texttree=False, max_packets=0, extra_tshark_args=extra,
                    codegen_profile=codegen_profile, ws_isolate_plugins=ws_isolate,
                    ws_plugin_id=(str(plugin_id) if plugin_id and host == "wireshark_c" else None),
                )
            elif host == "scapy":
                exported = export_scapy_tooljson(
                    plugin_src_dir=plugin_dir, pcap_path=pcap, filter_name=ws_filter, out_dir=export_dir,
                    entry_override=getattr(args, "stage_c_scapy_entry", None), ports=_parse_scapy_ports_arg(),
                    transport=getattr(args, "stage_c_scapy_transport", "auto"), max_packets=0,
                    include_payload_layers=True, codegen_profile=codegen_profile,
                )
            elif host == "zeek_spicy":
                from pargen.parser_realization.target_platforms.zeek_spicy_export import export_zeek_spicy_tooljson
                exported = export_zeek_spicy_tooljson(
                    plugin_src_dir=plugin_dir, pcap_path=pcap, filter_name=ws_filter, out_dir=export_dir,
                    codegen_profile=codegen_profile, frame_index=list(frames), max_packets=0,
                    spicyz_path=args.stage_c_spicyz, disable_runtime_constraints_for_export=True,
                )
            else:
                raise RuntimeError(f"no field export for host {host}")
            return Path(exported.tooljson_path)

        def _run_pkg_field_check(round_idx: int) -> tuple[Any, Optional[str]]:
            """CheckParser, field part: parser vs PKG interpreter on the suite's positive messages (by PKG identity)."""
            if suite_source != "pkg":
                return None, None
            if format_tree_path is None or mapping_path is None:
                return None, "PKG tree or suite mapping unavailable"
            if str(os.getenv("STAGE_C_FIELD_CHECK", "1")).strip().lower() in {"0", "false", "no", "off"}:
                return None, "field check disabled by STAGE_C_FIELD_CHECK"
            try:
                mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
            except Exception as exc:
                return None, f"mapping unreadable: {type(exc).__name__}: {exc}"
            base = str(mapping.get("base_pcap") or "")
            frames = sorted({int(item["frame_no"]) for item in (mapping.get("positive_seeds") or [])
                             if isinstance(item, dict) and item.get("frame_no")})
            candidates = [Path(suite_dir) / base, Path(mapping_path).parent / base] if base else []
            positives_pcap = next((path for path in candidates if path.is_file()), None)
            if positives_pcap is None or not frames:
                return None, "positive PCAP or positive frame list unavailable"
            from pargen.parser_realization.pkg_field_conformance import check_parser_fields
            try:
                with tempfile.TemporaryDirectory(prefix="stage_c_field_export_") as tmp:
                    tooljson = _export_parser_tooljson(positives_pcap, Path(tmp), frames=frames)
                    result = check_parser_fields(
                        tree_path=Path(format_tree_path), positives_pcap=positives_pcap, parser_tooljson=tooljson,
                        pcap_layer=str(mapping.get("pcap_layer") or "transport_payload"),
                        ip_proto=mapping.get("ip_proto"), frames=frames, filter_name=ws_filter,
                        out_path=out_dir / "field_check" / f"round_{round_idx:02d}.json",
                    )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"  field check unavailable: {message}")
                return None, message
            print(
                f"  field check: conflicts={result.count} {result.kinds()} "
                f"matched={result.matched_fields}/{result.reference_fields} frames={result.frames_checked}"
            )
            return result, None

        def _parser_conflicts(suite_res: Any, field_res: Any) -> int:
            """|Q|: rejected positives, undetected qualified violations, and field conflicts."""
            qualified = _qualified_for_repair(suite_res)
            negatives = sum(1 for case in (getattr(qualified, "results", ()) or ()) if not getattr(case, "ok", False))
            positives = int(getattr(suite_res, "fn", 0) or 0)
            return negatives + positives + (int(field_res.count) if field_res is not None else 0)

        def _read_text_tail(path: Optional[Path], *, max_chars: int = 12000) -> str:
            if path is None:
                return ""
            try:
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""
            if len(text) <= max_chars:
                return text
            return text[-max_chars:]

        def _load_exp_constraint_host_docs() -> str:
            try:
                from pargen.parser_realization.host_docs import load_host_docs

                host_dir = DATA_DIR / "host_docs" / str(args.stage_c_host)
                chunks = load_host_docs(host_dir)
            except Exception:
                return ""
            parts: list[str] = []
            for chunk in sorted(chunks, key=lambda c: (c.kind, c.source)):
                try:
                    source = Path(chunk.source).resolve().relative_to(DATA_DIR).as_posix()
                except Exception:
                    source = str(chunk.source)
                parts.append(f"--- {chunk.kind}: {source} ---\n{chunk.content.rstrip()}")
            return "\n\n".join(parts).strip()

        def _summarize_exp_constraint_build_failure(exc: BaseException) -> tuple[tuple[str, ...], str]:
            lines: list[str] = [f"generated parser failed to compile/load after exp_constraint repair: {exc}"]
            log_tail = ""
            br = build_result
            if br is not None:
                for attr in ("errors", "runtime_errors"):
                    vals = getattr(br, attr, None)
                    if vals:
                        try:
                            lines.extend(str(v) for v in list(vals)[:12])
                        except Exception:
                            lines.append(str(vals))
                for attr in ("build_log", "runtime_log"):
                    log_path = getattr(br, attr, None)
                    tail = _read_text_tail(Path(log_path) if log_path else None, max_chars=12000)
                    if tail:
                        log_tail += f"\n\n--- {attr}: {log_path} ---\n{tail}"
            backend_errors, backend_summary = _summarize_backend_contract_findings(max_items=12)
            lines.extend(backend_errors)
            if backend_summary:
                log_tail += f"\n\n--- backend realization contract ---\n{backend_summary}"
            return tuple(dict.fromkeys([ln for ln in lines if ln.strip()])), log_tail.strip()

        def _request_exp_constraint_repair(
            *,
            sources: dict[str, str],
            errors: tuple[str, ...],
            log_tail: str,
            round_idx: int,
            phase: str,
        ) -> Optional[dict[str, str]]:
            if not sources:
                return None
            if not errors:
                return None
            print(f"  exp_constraint repair: prompting LLM phase={phase} with {len(errors)} failure signals")
            try:
                from pargen.parser_realization.backend_contract import backend_contract_repair_rules
                from pargen.parser_realization.code_repair_agent import CodeRepairAgent, RepairDiagnostics
                from pargen.parser_realization.llm_client import create_llm_client

                repair_llm_cfg = StageCLLMConfig(
                    provider=args.stage_c_provider,
                    model=args.stage_c_model,
                    temperature=0.0,
                    max_tokens=args.stage_c_max_tokens,
                )
                repair_logs_dir = paths.stage_c_logs_dir / "exp_constraint_repair" / args.stage_c_host
                repair_logs_dir.mkdir(parents=True, exist_ok=True)
                agent = CodeRepairAgent(create_llm_client(repair_llm_cfg))
                host_docs = _load_exp_constraint_host_docs() if args.stage_c_host == "wireshark_c" else ""
                backend_rules = backend_contract_repair_rules(str(args.stage_c_host))
                phase_norm = str(phase or "").strip().lower()
                phase_specific_rules = ""
                if phase_norm.startswith("backend"):
                    phase_specific_rules = (
                        "Backend-realization phase:\n"
                        "- Repair only target-backend API misuse, compile/load/runtime crashes, and positive seed parsing regressions.\n"
                        "- Do not add broad packet rejection to satisfy negative counterexamples in this phase.\n"
                        "- Preserve existing constraint checks and BS_CONSTRAINT tags unless they are causing backend/runtime failure."
                    )
                elif phase_norm.startswith("conformance"):
                    phase_specific_rules = (
                        "Conformance phase (fields and validity on PKG-constructed messages):\n"
                        "- A field conflict names the PKG node id and its output key: the parser misses the field, emits "
                        "an additional one, or reads a different span (bit offset or width) or decoded value than the PKG "
                        "interpreter. Repair the implicated reads, offsets, lengths, branch guards, byte order or "
                        "signedness, and emit the field under its output key.\n"
                        "- A validity conflict names the PKG rule (BS_CONSTRAINT <id>): a violated rule that is not "
                        "reported, or a rule reported on a valid message.\n"
                        "- Keep backend registration, and keep the diagnostics of rules that already behave correctly."
                    )
                elif phase_norm.startswith("constraint"):
                    phase_specific_rules = (
                        "Constraint-realization phase:\n"
                        "- Assume backend registration/API usage is stable; focus only on missing IR constraint checks and BS_CONSTRAINT diagnostics.\n"
                        "- Do not change dissector registration, field layout, or normal positive-seed parsing except for the minimal code needed to report a violated IR constraint.\n"
                        "- Normal seed packets must remain free of BS_CONSTRAINT diagnostics."
                    )
                elif phase_norm == "compile":
                    phase_specific_rules = (
                        "Compile/load repair phase:\n"
                        "- Repair only build, load, or backend API failures introduced by the previous patch.\n"
                        "- Keep protocol parsing and constraint behavior unchanged unless required to compile or load."
                    )
                return agent.repair_files(
                    files=sources,
                    diagnostics=RepairDiagnostics(
                        tool_name=f"exp_constraint_conformance_suite:{phase}",
                        protocol_name=args.protocol,
                        errors=errors,
                        log_tail=log_tail,
                        log_tail_max_chars=20000,
                        host_docs=host_docs,
                        host_docs_max_chars=18000,
                    ),
                    codegen_profile=codegen_profile,
                    extra_rules=(
                        "Exp-constraint repair rules:\n"
                        "- This is protocol conformance repair for our generated parser, not security or attack detection.\n"
                        "- You are repairing a generated parser against BOTH positive seed packets and constraint-violating counterexamples.\n"
                        "- A patch is accepted only if it reduces the reported conflicts (field observations and validity outcomes) without losing accepted positive messages or detected violations.\n"
                        "- Normal seed packets must keep parsing normally; never satisfy failures by rejecting broad packet classes.\n"
                        "- Do not emit BS_CONSTRAINT for a valid selector/value merely because it selects a branch.\n"
                        "- For condition_on constraints, report a violation only when the selected branch's required field/length relation is missing or inconsistent.\n"
                        "- Constraint-violating counterexamples should trigger the parser's existing constraint-reporting path.\n"
                        "- For Wireshark hosts, emit/retain the expected BS_CONSTRAINT diagnostic tags when a violated IR constraint is observed.\n"
                        + (backend_rules + "\n" if backend_rules else "")
                        + (phase_specific_rules + "\n" if phase_specific_rules else "")
                        + "- Prefer implementing checks from existing decoded fields and the codegen profile/IR; do not hard-code pcap bytes or case names.\n"
                        "- Keep changes minimal and localized to the generated parser sources."
                    ),
                    prompt_path=repair_logs_dir / f"round_{round_idx:02d}_{phase}_prompt.txt",
                    response_path=repair_logs_dir / f"round_{round_idx:02d}_{phase}_response.txt",
                    allow_new_files=False,
                )
            except Exception as exc:
                print(f"  exp_constraint repair failed before patching: {type(exc).__name__}: {exc}")
                return None

        out_dir = stage_c_out_dir / "exp_constraint" / args.stage_c_host
        out_dir.mkdir(parents=True, exist_ok=True)
        repair_enabled = bool(getattr(args, "stage_c_exp_constraint_fix", False))
        max_rounds = int(getattr(args, "stage_c_exp_constraint_fix_max_rounds", 3) or 3) if repair_enabled else 1
        max_rounds = max(1, min(5, max_rounds))
        worst_k = max(1, int(getattr(args, "stage_c_exp_constraint_fix_worst_k", 8) or 8))
        repair_history: list[dict[str, Any]] = []
        pending_restore_sources: Optional[dict[str, str]] = None
        pending_baseline_res: Any = None
        pending_baseline_fields: Any = None
        pending_baseline_field_error: Optional[str] = None
        field_res: Any = None
        field_err: Optional[str] = None
        rebuild_before_next_round = False
        res = None
        for round_idx in range(1, max_rounds + 1):
            if rebuild_before_next_round and _exp_constraint_repair_requires_rebuild(args.stage_c_host):
                try:
                    _validate_exp_constraint_repair_candidate(reason=f"exp_constraint repair round {round_idx}")
                    rebuild_before_next_round = False
                except SystemExit as exc:
                    errors, log_tail = _summarize_exp_constraint_build_failure(exc)
                    repair_history.append(
                        {
                            "round": round_idx,
                            "accepted": False,
                            "reason": "build_failed_after_repair",
                            "errors": list(errors)[:12],
                        }
                    )
                    if pending_restore_sources is not None and repair_enabled and round_idx < max_rounds:
                        sources = _scan_exp_constraint_sources()
                        patched = _request_exp_constraint_repair(
                            sources=sources,
                            errors=errors,
                            log_tail=log_tail,
                            round_idx=round_idx,
                            phase="compile",
                        )
                        if patched:
                            _write_exp_constraint_sources(patched)
                            repair_history.append(
                                {
                                    "round": round_idx,
                                    "patched_files": sorted(patched.keys()),
                                    "accepted": None,
                                    "reason": "pending_compile_repair_retry",
                                }
                            )
                            rebuild_before_next_round = True
                            continue
                    if pending_restore_sources is not None:
                        print("  exp_constraint repair build failed; rolling back generated parser files")
                        _write_exp_constraint_sources(pending_restore_sources)
                        try:
                            _validate_exp_constraint_repair_candidate(reason="exp_constraint repair rollback")
                        except SystemExit as rollback_exc:
                            print(f"  exp_constraint rollback build failed: {rollback_exc}")
                        res = pending_baseline_res
                        field_res = pending_baseline_fields
                        field_err = pending_baseline_field_error
                        pending_restore_sources = None
                        pending_baseline_res = None
                    break

            res = _run_exp_constraint_suite_round(round_idx)
            if res is None:
                if pending_restore_sources is not None:
                    print("  exp_constraint repair produced an unusable suite result; rolling back generated parser files")
                    _write_exp_constraint_sources(pending_restore_sources)
                    repair_history.append(
                        {
                            "round": round_idx,
                            "accepted": False,
                            "reason": "suite_result_unusable",
                        }
                    )
                    if _exp_constraint_repair_requires_rebuild(args.stage_c_host):
                        _validate_exp_constraint_repair_candidate(reason="exp_constraint repair rollback")
                    res = pending_baseline_res
                    field_res = pending_baseline_fields
                    field_err = pending_baseline_field_error
                    break
                break

            field_res, field_err = _run_pkg_field_check(round_idx)
            round_summary_path = out_dir / f"round_{round_idx:02d}_summary.json"
            from pargen.parser_realization.conformance_observations import evaluation_gaps, metric_summary
            qualification_gaps = evaluation_gaps(res) if suite_source == "pkg" else []
            if qualification_gaps:
                # Unqualified counterexamples stay in the results and the coverage record but are not
                # used as repair evidence (see _qualified_for_repair); repair continues on qualified cases.
                repair_history.append({"round": round_idx, "accepted": None,
                    "reason": "unqualified_cases_excluded_from_repair", "evaluation_gaps": qualification_gaps})
            round_summary_path.write_text(
                json.dumps(
                    {
                        "round": round_idx,
                        "success": res.success,
                        "fail": res.fail,
                        "total": res.total,
                        "coverage": res.coverage,
                        "conformance_metrics": metric_summary(res),
                        "evaluation_gaps": qualification_gaps,
                        "field_conflicts": field_res.count if field_res is not None else None,
                        "field_conflict_kinds": field_res.kinds() if field_res is not None else None,
                        "field_check_error": field_err,
                        "field_check_gaps": _stage_c_field_check_gaps(field_res, field_err) if suite_source == "pkg" else [],
                        "conflicts_total": _parser_conflicts(res, field_res),
                        "confusion_matrix": {
                            "tp": int(getattr(res, "tp") or 0),
                            "fn": int(getattr(res, "fn") or 0),
                            "fp": int(getattr(res, "fp") or 0),
                            "tn": int(getattr(res, "tn") or 0),
                        }
                        if getattr(res, "tp", None) is not None
                        else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            _run_pkg_realization_check(round_idx, phase="post_suite", suite_res=res)

            if pending_restore_sources is not None and pending_baseline_res is not None:
                gate_accepted, gate_reason = _stage_c_code_repair_accepts(
                    pending_baseline_res,
                    res,
                    old_conflicts=_parser_conflicts(pending_baseline_res, pending_baseline_fields),
                    new_conflicts=_parser_conflicts(res, field_res),
                    field_check_lost=suite_source == "pkg" and (
                        field_err is not None or field_res is None or pending_baseline_fields is None
                        or field_res.frames_checked < pending_baseline_fields.frames_checked
                        or field_res.reference_fields < pending_baseline_fields.reference_fields
                        or field_res.unverified_values > pending_baseline_fields.unverified_values
                        or field_res.reference_unparsed_frames > pending_baseline_fields.reference_unparsed_frames
                        or bool(set(field_res.reference_missing_frames) - set(pending_baseline_fields.reference_missing_frames))
                    ),
                )
                if not gate_accepted:
                    print(f"  code repair not accepted ({gate_reason})")
                    gate_phase, gate_errors, gate_log_tail = _select_exp_constraint_repair_feedback(
                        res,
                        max_cases=worst_k,
                        gate=True,
                        field_res=field_res,
                    )
                    if (repair_enabled and round_idx < max_rounds and gate_errors
                            and (suite_source != "pkg" or (field_err is None and field_res is not None))):
                        print(
                            "  exp_constraint repair: using "
                            f"{gate_phase} feedback for another bounded repair round"
                        )
                        sources = _scan_exp_constraint_sources()
                        patched = _request_exp_constraint_repair(
                            sources=sources,
                            errors=gate_errors,
                            log_tail=gate_log_tail,
                            round_idx=round_idx,
                            phase=gate_phase,
                        )
                        if patched:
                            _write_exp_constraint_sources(patched)
                            repair_history.append(
                                {
                                    "round": round_idx,
                                    "patched_files": sorted(patched.keys()),
                                    "accepted": None,
                                    "reason": f"pending_{gate_phase}_repair_retry",
                                    "baseline_success": int(getattr(pending_baseline_res, "success", 0) or 0),
                                    "new_success": int(getattr(res, "success", 0) or 0),
                                    "baseline_fn": int(getattr(pending_baseline_res, "fn", 0) or 0),
                                    "new_fn": int(getattr(res, "fn", 0) or 0),
                                    "gate_reason": str(gate_reason),
                                }
                            )
                            if _exp_constraint_repair_requires_rebuild(args.stage_c_host):
                                rebuild_before_next_round = True
                            continue
                    _write_exp_constraint_sources(pending_restore_sources)
                    print("  exp_constraint repair did not recover before the round limit; rolling back generated parser files")
                    repair_history.append(
                        {
                            "round": round_idx,
                            "accepted": False,
                            "reason": "no_regression_gate",
                            "baseline_success": int(getattr(pending_baseline_res, "success", 0) or 0),
                            "new_success": int(getattr(res, "success", 0) or 0),
                            "baseline_fn": int(getattr(pending_baseline_res, "fn", 0) or 0),
                            "new_fn": int(getattr(res, "fn", 0) or 0),
                            "gate_reason": str(gate_reason),
                        }
                    )
                    if _exp_constraint_repair_requires_rebuild(args.stage_c_host):
                        _validate_exp_constraint_repair_candidate(reason="exp_constraint repair rollback")
                    res = pending_baseline_res
                    field_res = pending_baseline_fields
                    field_err = pending_baseline_field_error
                    break
                repair_history.append(
                    {
                        "round": round_idx,
                        "accepted": True,
                        "success": int(getattr(res, "success", 0) or 0),
                        "fail": int(getattr(res, "fail", 0) or 0),
                        "fn": int(getattr(res, "fn", 0) or 0),
                        "field_conflicts": field_res.count if field_res is not None else None,
                        "gate_reason": str(gate_reason),
                    }
                )
                pending_restore_sources = None
                pending_baseline_res = None
                pending_baseline_fields = None

            round_failures = _parser_conflicts(res, field_res)
            if suite_source == "pkg" and (field_res is None or field_err):
                repair_history.append({"round": round_idx, "accepted": None, "reason": "field_check_unavailable"})
                break
            if round_failures <= 0:
                break
            if not repair_enabled or round_idx >= max_rounds:
                break

            sources = _scan_exp_constraint_sources()
            if not sources:
                print("  exp_constraint repair skipped: no editable generated parser sources found")
                break

            repair_phase, errors, log_tail = _select_exp_constraint_repair_feedback(
                res,
                max_cases=worst_k,
                gate=False,
                field_res=field_res,
            )
            if not errors:
                print("  exp_constraint repair skipped: no failed cases were extractable")
                break

            patched = _request_exp_constraint_repair(
                sources=sources,
                errors=errors,
                log_tail=log_tail,
                round_idx=round_idx,
                phase=repair_phase,
            )

            if not patched:
                print("  exp_constraint repair made no usable code changes")
                break

            pending_restore_sources = sources
            pending_baseline_res = res
            pending_baseline_fields = field_res
            pending_baseline_field_error = field_err
            _write_exp_constraint_sources(patched)
            repair_history.append(
                {
                    "round": round_idx,
                    "patched_files": sorted(patched.keys()),
                    "accepted": None,
                    "reason": f"pending_{repair_phase}_next_round",
                }
            )
            if _exp_constraint_repair_requires_rebuild(args.stage_c_host):
                rebuild_before_next_round = True
        if res is None:
            exp_constraints_result = False
            unavailable_summary_path = out_dir / "summary.json"
            unavailable_summary_path.write_text(
                json.dumps(
                    {
                        "protocol": args.protocol,
                        "host": args.stage_c_host,
                        "mode": str(getattr(args, "stage_c_exp_constraint_mode", "binary") or "binary"),
                        "status": "unavailable",
                        "success": 0,
                        "fail": 1,
                        "total": 0,
                        "coverage": 0.0,
                        "reason": "suite runner returned no result",
                        "static_fallback_used": False,
                        "pkg_preflight": pkg_preflight_payload if suite_source == "pkg" else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"  exp_constraint: unavailable (suite runner returned no result; wrote {unavailable_summary_path})")
            if suite_source == "pkg" and bool(getattr(args, "stage_c_exp_constraint_strict", False)):
                raise SystemExit("Stage C PKG exp_constraint verification unavailable: suite runner returned no result")
            _run_normal_dataset_parse_summary(reason="post-stage-c")
            return
        if suite_source == "pkg" and int(getattr(res, "total", 0) or 0) <= 0:
            exp_constraints_result = False
            unavailable_summary_path = out_dir / "summary.json"
            unavailable_summary_path.write_text(
                json.dumps(
                    {
                        "protocol": args.protocol,
                        "host": args.stage_c_host,
                        "mode": getattr(res, "mode", "binary"),
                        "status": "unavailable",
                        "success": 0,
                        "fail": 1,
                        "total": 0,
                        "coverage": 0.0,
                        "reason": "suite runner returned zero evaluated cases",
                        "static_fallback_used": False,
                        "pkg_preflight": pkg_preflight_payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"  exp_constraint: unavailable (zero evaluated cases; wrote {unavailable_summary_path})")
            if bool(getattr(args, "stage_c_exp_constraint_strict", False)):
                raise SystemExit("Stage C PKG exp_constraint verification unavailable: zero evaluated cases")
            _run_normal_dataset_parse_summary(reason="post-stage-c")
            return

        summary_path = out_dir / "summary.json"
        summary_payload = {
            "protocol": res.protocol,
            "host": res.host,
            "mode": getattr(res, "mode", "binary"),
            "success": res.success,
            "fail": res.fail,
            "total": res.total,
            "coverage": res.coverage,
            "suite_dir": res.suite_dir,
            "mapping_path": res.mapping_path,
            "cases": [cr.__dict__ for cr in res.results],
            "repair_enabled": repair_enabled,
            "repair_history": repair_history,
            "conformance_metrics": metric_summary(res),
            "evaluation_gaps": evaluation_gaps(res) if suite_source == "pkg" else [],
            "field_check": field_res.to_dict(max_conflicts=100) if field_res is not None else None,
            "field_check_error": field_err,
            "field_check_gaps": _stage_c_field_check_gaps(field_res, field_err) if suite_source == "pkg" else [],
        }
        if suite_source == "pkg":
            negative_failures = int(getattr(res, "fail", 0) or 0)
            positive_seed_failures = int(getattr(res, "fn", 0) or 0)
            summary_payload["status"] = _stage_c_pkg_status(
                failures=negative_failures + positive_seed_failures,
                evaluation_gaps=summary_payload["evaluation_gaps"], preflight_issues=pkg_preflight_issues,
                field_result=field_res, field_error=field_err,
            )
            summary_payload["pkg_preflight"] = pkg_preflight_payload
        if generated_exp_constraint_suite is not None:
            summary_payload["generated_suite"] = generated_exp_constraint_suite.to_dict()
        if getattr(res, "tp", None) is not None:
            summary_payload["confusion_matrix"] = {
                # Positive = normal (seed); Negative = mutated.
                "tp": int(getattr(res, "tp") or 0),
                "fn": int(getattr(res, "fn") or 0),
                "fp": int(getattr(res, "fp") or 0),
                "tn": int(getattr(res, "tn") or 0),
                "positive_total": int(getattr(res, "positive_total") or 0),
                "negative_total": int(getattr(res, "negative_total") or 0),
            }
            summary_payload["positive_cases"] = [cr.__dict__ for cr in getattr(res, "positive_results", ())]
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Print failures only (keep logs compact).
        for cr in res.results:
            if not cr.ok:
                print(f"  [FAIL] {cr.case}: {cr.reason}")
        for cr in getattr(res, "positive_results", ()):
            if not cr.ok:
                print(f"  [FAIL] {cr.case}: {cr.reason}")

        print(
            f"  exp_constraint summary: success={res.success} fail={res.fail} total={res.total} "
            f"coverage={res.coverage:.4f} (wrote {summary_path})"
        )
        if getattr(res, "tp", None) is not None:
            print(
                "  exp_constraint confusion: "
                f"TP={int(getattr(res, 'tp') or 0)} FP={int(getattr(res, 'fp') or 0)} "
                f"FN={int(getattr(res, 'fn') or 0)} TN={int(getattr(res, 'tn') or 0)}"
            )
        exp_constraints_result = res
        negative_failures = int(getattr(res, "fail", 0) or 0)
        positive_seed_failures = int(getattr(res, "fn", 0) or 0)
        if bool(getattr(args, "stage_c_exp_constraint_strict", False)) and (
            negative_failures > 0 or positive_seed_failures > 0
            or (suite_source == "pkg" and summary_payload["status"] != "complete")
        ):
            raise SystemExit(
                "Stage C exp_constraint verification failed: "
                f"negative_fail={negative_failures}/{res.total}, "
                f"positive_seed_fail={positive_seed_failures}/{int(getattr(res, 'positive_total', 0) or 0)}, "
                f"status={summary_payload.get('status')}, field_check_gaps={summary_payload['field_check_gaps']}"
            )

        # Also export parsing results on the normal dataset when exp_constraint is enabled.
        # This helps interpret "binary abnormal" scores (e.g., avoid the degenerate case where the
        # parser always fails to parse).
        _run_normal_dataset_parse_summary(reason="post-exp_constraint")
        return

    # --- Stage C semantic verify/fix (optional) ---------------------------
    if semantic_enabled:
        if not plugin_dir.exists():
            raise SystemExit(f"Stage C semantic requested, but plugin_dir does not exist: {plugin_dir}")
        if format_tree_path is None or not format_tree_path.exists():
            raise SystemExit("Stage C semantic requested, but no format tree is available for semantic GT.")

        semantic_out_dir = args.stage_c_semantic_out_dir or (stage_c_out_dir / "semantic" / args.stage_c_host)
        semantic_logs_dir = paths.stage_c_logs_dir / "semantic" / args.stage_c_host

        repair_llm_cfg = StageCLLMConfig(
            provider=args.stage_c_provider,
            model=args.stage_c_model,
            temperature=0.0,
            max_tokens=args.stage_c_max_tokens,
        )

        enable_fix = bool(getattr(args, "stage_c_semantic_fix", False))
        from pargen.parser_realization.semantic_stage import (
            SemanticRepairLoopConfig,
            run_stage_c_semantic_with_auto_repair,
        )

        semantic_cfg = SemanticRepairLoopConfig(
            max_rounds=(int(args.stage_c_semantic_max_rounds) if enable_fix else 1),
            min_f1_l1=float(args.stage_c_semantic_min_f1),
            max_packets=int(args.stage_c_semantic_max_packets),
            require_full_parse=bool(args.stage_c_semantic_require_full_parse),
            enable_llm_repair=enable_fix,
            mutants_per_packet=int(getattr(args, "stage_c_semantic_mutants_per_packet", 0) or 0),
        )

        if args.stage_c_host == "wireshark_c":
            backend = WiresharkCBackend(
                ws_src_root=ws_src_root,
                ws_build_dir=ws_build_dir,
                plugin_id=plugin_id,
                plugin_c_mode=getattr(args, "stage_c_ws_plugin_c_mode", "generated"),
                download_url=args.stage_c_ws_download_url or WIRESHARK_DEFAULT_URL,
            )
            tshark_override = args.stage_c_tshark
            source_globs = None
            tshark_extra_args = _wireshark_c_tshark_extra_args()
        elif args.stage_c_host == "wireshark_lua":
            entry = _get_lua_entry()
            backend = WiresharkLuaBackend(
                lua_entry=entry,
                luac_path=getattr(args, "stage_c_luac", None),
                lua_path=getattr(args, "stage_c_lua", None),
            )
            if args.stage_c_tshark is not None:
                tshark_override = Path(args.stage_c_tshark)
            else:
                tshark_path_str = shutil.which("tshark")
                if not tshark_path_str:
                    raise SystemExit("tshark not found on PATH; pass --stage-c-tshark for wireshark_lua semantic.")
                tshark_override = Path(tshark_path_str)
            source_globs = ("*.lua",)
            tshark_extra_args = ["-X", f"lua_script:{entry}"]
        elif args.stage_c_host == "scapy":
            backend = ScapyBackend(entry_override=getattr(args, "stage_c_scapy_entry", None))
            tshark_override = None
            source_globs = ("*.py",)
            tshark_extra_args = None
        elif args.stage_c_host == "zeek_spicy":
            backend = ZeekSpicyBackend(
                zeek_path=getattr(args, "stage_c_zeek", None),
                spicyz_path=getattr(args, "stage_c_spicyz", None),
            )
            tshark_override = None
            source_globs = ("**/*.spicy", "**/*.evt", "**/*.zeek", "**/*.sig", "**/*.meta")
            tshark_extra_args = None
        else:
            print(
                f"\n[Stage C] semantic requested, but host='{args.stage_c_host}'. "
                "Skipping semantic loop (unsupported host)."
            )
            return

        print("\n[Stage C] Semantic validation/repair (oracle=Stage B traffic_fixed_tree)")
        if semantic_frames is not None:
            if semantic_frames_raw is not None and len(semantic_frames) != len(semantic_frames_raw):
                print(
                    f"  [SemanticRepair] using dataset frame index: {frame_index_path} "
                    f"(frames={len(semantic_frames)} reassembled; raw_index={len(semantic_frames_raw)})"
                )
            else:
                print(f"  [SemanticRepair] using dataset frame index: {frame_index_path} (frames={len(semantic_frames)})")

        semantic_gate_baseline_metrics: Optional[dict] = None
        semantic_gate_snapshot_dir: Optional[Path] = None
        if enable_fix and bool(args.stage_c_pcap_to_texttree) and bool(getattr(args, "stage_c_semantic_dataset_gate", False)):
            print("\n[Stage C] Semantic no-regression gate: capturing pre-repair dataset baseline")
            try:
                semantic_gate_baseline_metrics = _run_stage_c_export(reason="pre-semantic-baseline")
            except SystemExit as exc:
                semantic_gate_baseline_metrics = None
                print(f"  [Stage C] semantic gate baseline export unavailable: {exc}")
            if semantic_gate_baseline_metrics is None:
                semantic_gate_baseline_metrics = _read_stage_c_dataset_metrics()
            if semantic_gate_baseline_metrics is not None:
                semantic_gate_snapshot_dir = Path(tempfile.mkdtemp(prefix="stagec_semantic_gate_"))
                _snapshot_semantic_gate_state(semantic_gate_snapshot_dir)
                print(f"  semantic gate baseline: {_dataset_metric_brief(semantic_gate_baseline_metrics)}")
            else:
                print("  semantic gate baseline: <missing>; no dataset no-regression check will be applied")

        with _record_stage_time(timing, "stage_c_semantic_s"):
            semantic_result = run_stage_c_semantic_with_auto_repair(
                syntax_backend=backend,
                plugin_src_dir=plugin_dir,
                tree_path=format_tree_path,
                pcap_path=pcap_path,
                frame_index=semantic_frames,
                protocol_filter_name=ws_filter,
                file_prefix=args.protocol,
                out_dir=semantic_out_dir,
                logs_root=semantic_logs_dir,
                codegen_profile=codegen_profile,
                constraints_mode=constraints_mode,
                llm_config=repair_llm_cfg,
                semantic_cfg=semantic_cfg,
                tshark_override=tshark_override,
                source_globs=source_globs,
                tshark_extra_args=tshark_extra_args,
                ws_isolate_plugins=bool(getattr(args, "stage_c_ws_isolate_plugins", True)),
                ws_plugin_id=(str(plugin_id) if args.stage_c_host == "wireshark_c" else None),
                scapy_entry_override=getattr(args, "stage_c_scapy_entry", None),
                scapy_ports=_parse_scapy_ports_arg(),
                scapy_transport=getattr(args, "stage_c_scapy_transport", "auto"),
                progress_printer=print,
            )
        metrics_payload: Optional[dict] = None
        if semantic_result.metrics_json and Path(semantic_result.metrics_json).exists():
            try:
                metrics_payload = json.loads(Path(semantic_result.metrics_json).read_text(encoding="utf-8"))
            except Exception:
                metrics_payload = None

        if args.stage_c_host == "zeek_spicy" and metrics_payload is not None:
            thr = float(metrics_payload.get("level2_soft_threshold", 0.0) or 0.0)
            l2s_f = float(metrics_payload.get("field_f1_level2_soft", 0.0) or 0.0)
            print(
                f"  semantic: ok={bool(semantic_result.ok)} L1_F1={float(semantic_result.f1_l1):.4f} "
                f"L2_soft@{thr:.2f} F1={l2s_f:.4f} skipped={int(semantic_result.skipped_packets)}"
            )
        else:
            print(
                f"  semantic: ok={bool(semantic_result.ok)} L1_F1={float(semantic_result.f1_l1):.4f} "
                f"L2_F1={float(semantic_result.f1_l2):.4f} skipped={int(semantic_result.skipped_packets)}"
            )
        if semantic_result.metrics_json:
            print(f"  semantic.metrics_json: {semantic_result.metrics_json}")
        if semantic_result.diff_report:
            print(f"  semantic.diff_report: {semantic_result.diff_report}")
        if semantic_result.metrics_json:
            try:
                gt_report = Path(semantic_result.metrics_json).parent / "semantic_gt_build_report.json"
                if gt_report.exists():
                    print(f"  semantic.gt_build_report: {gt_report}")
            except Exception:
                pass

        if metrics_payload is not None:
            try:
                tp1 = int(metrics_payload.get("tp_level1", 0) or 0)
                fp1 = int(metrics_payload.get("fp_level1", 0) or 0)
                fn1 = int(metrics_payload.get("fn_level1", 0) or 0)
                if args.stage_c_host == "zeek_spicy":
                    tp2s = int(metrics_payload.get("tp_level2_soft", 0) or 0)
                    fp2s = int(metrics_payload.get("fp_level2_soft", 0) or 0)
                    fn2s = int(metrics_payload.get("fn_level2_soft", 0) or 0)
                    dtype_counts = ""
                    if "field_f1_level2_dtype" in metrics_payload:
                        tp2d = int(metrics_payload.get("tp_level2_dtype", 0) or 0)
                        fp2d = int(metrics_payload.get("fp_level2_dtype", 0) or 0)
                        fn2d = int(metrics_payload.get("fn_level2_dtype", 0) or 0)
                        dtype_counts = f" | L2_dtype TP={tp2d} FP={fp2d} FN={fn2d}"
                    print(
                        f"  semantic field_counts: L1 TP={tp1} FP={fp1} FN={fn1} | "
                        f"L2_soft TP={tp2s} FP={fp2s} FN={fn2s}{dtype_counts}"
                    )
                else:
                    tp2 = int(metrics_payload.get("tp_level2", 0) or 0)
                    fp2 = int(metrics_payload.get("fp_level2", 0) or 0)
                    fn2 = int(metrics_payload.get("fn_level2", 0) or 0)
                    tp2a = int(metrics_payload.get("tp_level2_aligned", 0) or 0)
                    fp2a = int(metrics_payload.get("fp_level2_aligned", 0) or 0)
                    fn2a = int(metrics_payload.get("fn_level2_aligned", 0) or 0)
                    mm2a = int(metrics_payload.get("name_mismatch_level2_aligned", 0) or 0)
                    dtype_counts = ""
                    if "field_f1_level2_dtype" in metrics_payload:
                        tp2d = int(metrics_payload.get("tp_level2_dtype", 0) or 0)
                        fp2d = int(metrics_payload.get("fp_level2_dtype", 0) or 0)
                        fn2d = int(metrics_payload.get("fn_level2_dtype", 0) or 0)
                        dtype_counts = f" | L2_dtype TP={tp2d} FP={fp2d} FN={fn2d}"
                    aligned_counts = ""
                    if "field_f1_level2_aligned" in metrics_payload:
                        aligned_counts = f" | L2_aligned TP={tp2a} FP={fp2a} FN={fn2a} MISMATCH={mm2a}"
                    print(
                        f"  semantic field_counts: L1 TP={tp1} FP={fp1} FN={fn1} | "
                        f"L2 TP={tp2} FP={fp2} FN={fn2}{aligned_counts}{dtype_counts}"
                    )
            except Exception:
                pass

        # Dataset-level GT evaluation (reference dissector), when Stage C export won't run.
        # - If semantic is hard-failing, compute before raising.
        # - If --no-stage-c-pcap-to-texttree, compute here since there will be no post-semantic export.
        pred_tooljson_from_semantic: Optional[Path] = None
        if semantic_result.metrics_json:
            try:
                candidate = Path(semantic_result.metrics_json).parent / "tool.json"
                if candidate.exists():
                    pred_tooljson_from_semantic = candidate
            except Exception:
                pred_tooljson_from_semantic = None

        will_raise = (not bool(semantic_result.ok)) and (not bool(getattr(args, "stage_c_semantic_soft_fail", False)))
        export_supported = args.stage_c_host in {"wireshark_c", "wireshark_lua", "scapy", "zeek_spicy"}
        should_compute_dataset_metrics = will_raise or (not bool(args.stage_c_pcap_to_texttree)) or (not export_supported)
        if pred_tooljson_from_semantic and should_compute_dataset_metrics:
            _maybe_compute_dataset_metrics(pred_tooljson=pred_tooljson_from_semantic)

        # Run exp_constraint verification even when semantic will hard-fail (helps debugging).
        if will_raise:
            _run_exp_constraint_verify_once(reason="pre-semantic-fail")

        if not bool(semantic_result.ok):
            min_f1 = float(getattr(args, "stage_c_semantic_min_f1", 0.99) or 0.99)
            gt_report = "<missing>"
            if semantic_result.metrics_json:
                try:
                    candidate = Path(semantic_result.metrics_json).parent / "semantic_gt_build_report.json"
                    if candidate.exists():
                        gt_report = str(candidate)
                except Exception:
                    gt_report = "<missing>"
            msg = (
                "Stage C semantic validation/repair failed: "
                f"L1_F1={float(semantic_result.f1_l1):.4f} < min={min_f1:.4f}. "
                f"metrics_json={str(semantic_result.metrics_json or '<missing>')}, "
                f"diff_report={str(semantic_result.diff_report or '<missing>')}, "
                f"gt_build_report={gt_report}"
            )
            if bool(getattr(args, "stage_c_semantic_soft_fail", False)):
                print(f"  [Stage C] WARNING: {msg}")
            else:
                raise SystemExit(msg)

        # After semantic repair, export stable Stage C artifacts for dataset-level evaluation.
        if args.stage_c_pcap_to_texttree:
            final_dataset_metrics: Optional[dict] = None
            post_semantic_export_failed: Optional[str] = None
            try:
                final_dataset_metrics = _run_stage_c_export(reason="post-semantic")
            except SystemExit as exc:
                post_semantic_export_failed = str(exc)
                if semantic_gate_snapshot_dir is None or semantic_gate_baseline_metrics is None:
                    raise
                print(
                    "  [Stage C] Semantic no-regression gate: post-repair export failed; "
                    "restoring pre-repair dataset baseline."
                )
                _restore_semantic_gate_state(semantic_gate_snapshot_dir)
                final_dataset_metrics = semantic_gate_baseline_metrics

            if final_dataset_metrics is None:
                final_dataset_metrics = _read_stage_c_dataset_metrics()

            gate_summary = {
                "enabled": bool(semantic_gate_baseline_metrics is not None and semantic_gate_snapshot_dir is not None),
                "baseline": semantic_gate_baseline_metrics,
                "candidate": final_dataset_metrics,
                "post_semantic_export_failed": post_semantic_export_failed,
                "accepted": True,
                "reason": "no_baseline" if semantic_gate_baseline_metrics is None else "candidate_not_worse",
            }
            if (
                semantic_gate_baseline_metrics is not None
                and final_dataset_metrics is not None
                and semantic_gate_snapshot_dir is not None
                and _dataset_metric_regressed(final_dataset_metrics, semantic_gate_baseline_metrics)
            ):
                print(
                    "  [Stage C] Semantic no-regression gate: rejecting repaired parser "
                    f"({ _dataset_metric_brief(final_dataset_metrics) }) below baseline "
                    f"({ _dataset_metric_brief(semantic_gate_baseline_metrics) }); restoring baseline."
                )
                _restore_semantic_gate_state(semantic_gate_snapshot_dir)
                gate_summary["accepted"] = False
                gate_summary["reason"] = "dataset_metric_regression"
                gate_summary["restored"] = "pre_semantic_baseline"
                final_dataset_metrics = semantic_gate_baseline_metrics

            if semantic_gate_baseline_metrics is not None:
                gate_summary["final"] = final_dataset_metrics
                _write_semantic_gate_summary(gate_summary)
            if semantic_gate_snapshot_dir is not None:
                shutil.rmtree(semantic_gate_snapshot_dir, ignore_errors=True)

    # Run exp_constraint verification when semantic is disabled or passes.
    _run_exp_constraint_verify_once(reason="post-stage-c")

    # Evaluate the retained parser after code refinement has ended.
    if (
        args.stage_c_pcap_to_texttree
        and not semantic_enabled
        and exp_constraints_enabled
        and bool(getattr(args, "stage_c_exp_constraint_fix", False))
    ):
        _run_stage_c_export(reason="post-refinement")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pargen pipeline (A → B → C)")
    parser.add_argument("--protocol", default="modbus", help="Protocol dataset under pargen/data (default: modbus)")
    parser.add_argument(
        "--modbus",
        action="store_true",
        help="Shortcut: equivalent to --protocol modbus",
    )
    parser.add_argument(
        "--pdf",
        dest="pdfs",
        action="append",
        type=Path,
        help="Explicit PDF path (can repeat). Defaults to all PDFs under data/<protocol>/pdfs/",
    )
    parser.add_argument("--claude-key", default=None, help="API key for Stage A (overrides config/env)")
    parser.add_argument("--deepseek-key", default=None, help="Optional DeepSeek API key (overrides config/env)")
    parser.add_argument("--config", type=Path, default=DEFAULT_API_CONFIG, help="Path to API config JSON")
    parser.add_argument("--sections-file", type=Path, help="Pre-extracted sections JSON (Stage B input)")
    parser.add_argument("--raw-sections-file", type=Path, help="Pre-extracted raw sections JSON")
    parser.add_argument("--sections-subset", type=Path, help="Subset/merged sections JSON for Step2 fix")
    parser.add_argument(
        "--spec-sections",
        action="store_true",
        default=False,
        help=(
            "Stage A without a model: cut every input document (inputs/pdfs: *.pdf, *.txt, *.docx) at its own "
            "headings and feed all sections, verbatim and unselected, to Stage B (pargen/spec_sections.py). "
            "Writes outputs/stage_a/spec_sections.json with the input documents' SHA-256; a file whose receipt "
            "matches the documents is reused."
        ),
    )
    parser.add_argument(
        "--spec-sections-only",
        action="store_true",
        default=False,
        help="Write the --spec-sections input and exit (implies --spec-sections).",
    )
    parser.add_argument(
        "--spec-select",
        choices=["all", "toc_llm"],
        default="all",
        help=(
            "Which sections --spec-sections keeps: all (default), or toc_llm: the Stage B initial model reads each "
            "document's table of contents (headings only) and picks the sections a parser needs "
            "(pargen/spec_sections.py TOC_SELECT_INSTRUCTIONS). Writes "
            "outputs/stage_a/spec_sections_toc.json; the GT is never read."
        ),
    )
    parser.add_argument(
        "--ablation-no-spec",
        action="store_true",
        default=False,
        help=(
            "Ablation: disable specification/document inputs for Stage B. "
            "Uses empty section stubs and writes outputs/logs under a separate no_spec variant directory."
        ),
    )
    parser.add_argument(
        "--ablation-no-traffic",
        action="store_true",
        default=False,
        help=(
            "Ablation: disable traffic-driven repair inputs. Reuses the default Stage A document sections, "
            "skips Stage B3 traffic fix and Stage C semantic repair/verify, and writes outputs/logs under no_traffic."
        ),
    )
    parser.add_argument("--run-doc-stage", action="store_true", help="Re-run Stage A even if sections already exist")
    parser.add_argument(
        "--stage-a-only",
        action="store_true",
        help="Run Stage A (document extraction) and exit (implies --run-doc-stage).",
    )
    parser.add_argument(
        "--stage-b-only",
        action="store_true",
        help="Run Stages A/B and exit before Stage C.",
    )
    parser.add_argument("--traffic", type=Path, help="Optional override for Stage B3 semantic fix")
    parser.add_argument(
        "--semantic-max-packets",
        type=int,
        default=0,
        help="Max traffic samples to use during hybrid validation (0 = no limit)",
    )
    parser.add_argument(
        "--target-message-type",
        type=str,
        default=None,
        help="Optional message_type filter for hybrid validation (request/response)",
    )
    parser.add_argument("--force-format-stage", action="store_true", help="Re-run Stage B even if cached tree exists")
    parser.add_argument(
        "--force-format-stage-llm",
        action="store_true",
        help="Force Stage B to ignore cached LLM responses (sets PARGEN_STEP2_FORCE_REBUILD=1).",
    )
    parser.add_argument(
        "--stage-b-initial-provider",
        default=None,
        help="LLM provider for Stage B initial tree generation (default: deepseek).",
    )
    parser.add_argument(
        "--stage-b-initial-model",
        default=None,
        help="LLM model for Stage B initial tree generation (default: --stage-b-model when providers match, else provider default).",
    )
    parser.add_argument(
        "--stage-b-initial-temperature",
        type=float,
        default=None,
        help="LLM temperature for Stage B initial tree generation (default: --stage-b-temperature when providers match).",
    )
    parser.add_argument(
        "--stage-b-initial-max-tokens",
        type=int,
        default=None,
        help="LLM max tokens for Stage B initial tree generation (default: --stage-b-max-tokens when providers match).",
    )
    parser.add_argument(
        "--stage-b-provider",
        default=None,
        help="LLM provider for Stage B refinement/inline-fix (default: deepseek).",
    )
    parser.add_argument(
        "--stage-b-model",
        default=None,
        help="LLM model for Stage B refinement/inline-fix (provider default if unset).",
    )
    parser.add_argument(
        "--stage-b-temperature",
        type=float,
        default=None,
        help="LLM temperature for Stage B refinement/inline-fix (provider default if unset).",
    )
    parser.add_argument(
        "--stage-b-max-tokens",
        type=int,
        default=None,
        help="LLM max tokens for Stage B refinement/inline-fix (provider default if unset).",
    )
    parser.add_argument("--no-fix-stage", dest="enable_fix_stage", action="store_false", default=True, help="Disable legacy Step2 fix stage (enabled by default)")
    parser.add_argument("--fix-provider", default=DEFAULT_PROVIDER, help="LLM provider for Step2 fix (default: deepseek)")
    parser.add_argument(
        "--fix-max-llm-calls",
        type=int,
        default=None,
        help="Max LLM calls for Stage B2 Step2 fix (default: $STEP2_MCTS_MAX_CALLS or 20).",
    )
    parser.add_argument("--fix-output-dir", type=Path, help="Directory to store Step2 fixed trees")
    parser.add_argument("--fix-log-dir", type=Path, help="Directory for Step2 fix logs")
    parser.add_argument("--no-semantic-fix", dest="enable_semantic_fix", action="store_false", default=True, help="Disable traffic-aware semantic fixing (enabled by default)")
    parser.add_argument("--semantic-fix-output", type=Path, help="Path to save semantic fixed tree")
    parser.add_argument(
        "--stage-c-tree",
        type=Path,
        default=None,
        help=(
            "Override the Stage C input protocol tree JSON and skip Stages A/B. "
            "Useful for ablations (e.g., use step2_cache/<protocol>/inline_initial_tree.json)."
        ),
    )
    parser.add_argument(
        "--stage-c-doc-summary",
        action="store_true",
        help=(
            "Ablation: run Stage C codegen using ONLY document summaries (no format tree / Parse-IR). "
            "Summaries are extracted from Stage A section JSON (prefer --sections-subset when provided). "
            "Skips Stages A/B tree construction."
        ),
    )
    parser.add_argument(
        "--stage-c-doc-summary-source",
        type=Path,
        default=None,
        help=(
            "Optional sections JSON (list or {\"sections\": [...]}) to extract 'summary' fields from when "
            "using --stage-c-doc-summary. Defaults to --sections-subset/--sections-file if available."
        ),
    )
    parser.add_argument("--reuse-b-stage-output", action="store_true", help="If traffic_fixed_tree.json already exists, skip Stage B and reuse it for Stage C")
    parser.add_argument(
        "--no-auto-reuse-b-stage-output",
        action="store_true",
        help="Disable automatic reuse of existing traffic_fixed_tree.json (still allows --reuse-b-stage-output).",
    )
    parser.add_argument(
        "--reuse-c-stage-output",
        dest="reuse_c_stage_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If Stage C outputs already exist under outputs/stage_c/plugin/<host>, reuse them and skip Stage C codegen "
            "(disabled by default; reuse is allowed only when metadata matches current Stage C inputs)"
        ),
    )
    parser.add_argument(
        "--stage-c-reuse-ignore-signature",
        action="store_true",
        help=(
            "Expert/debug mode: with --reuse-c-stage-output, run Stage C post-tools on an existing "
            "artifact even when its recorded input signature differs."
        ),
    )
    parser.add_argument("--stage-c-host", default="wireshark_c", help="Stage C host target (e.g., wireshark_c)")
    parser.add_argument(
        "--wireshark_c",
        "--wireshark-c",
        dest="wireshark_c_shortcut",
        action="store_true",
        help="Shortcut: equivalent to --stage-c-host wireshark_c",
    )
    parser.add_argument(
        "--wireshark_lua",
        "--wireshark-lua",
        dest="wireshark_lua_shortcut",
        action="store_true",
        help="Shortcut: equivalent to --stage-c-host wireshark_lua",
    )
    parser.add_argument(
        "--scapy",
        dest="scapy_shortcut",
        action="store_true",
        help="Shortcut: equivalent to --stage-c-host scapy",
    )
    parser.add_argument(
        "--zeek-spicy",
        "--zeek_spicy",
        "--zeek",
        dest="zeek_spicy_shortcut",
        action="store_true",
        help="Shortcut: equivalent to --stage-c-host zeek_spicy",
    )
    parser.add_argument("--stage-c-protocol-family", help="Protocol family for Stage C (default: --protocol)")
    parser.add_argument("--stage-c-host-docs", type=Path, help="Directory containing host docs/examples")
    parser.add_argument("--stage-c-host-ir-cache", type=Path, help="Cache file for learned Host-IR")
    parser.add_argument(
        "--stage-c-codegen-profile",
        type=Path,
        default=None,
        help="Optional JSON profile injected into the Stage C codegen prompt (e.g., port/direction rules, implicit fields).",
    )
    parser.add_argument(
        "--stage-c-contract-repair-max-rounds",
        dest="stage_c_contract_repair_max_rounds",
        type=int,
        default=2,
        help="Max LLM repair rounds for the constraint contract (default: 2).",
    )
    parser.add_argument(
        "--stage-c-allow-missing-contract-tags",
        action="store_true",
        default=False,
        help=(
            "Allow Stage C to write/evaluate a parser even when constraint contract tags remain missing. "
            "Intended for ablation variants that evaluate partially realized parsers instead of gating them."
        ),
    )
    parser.add_argument("--stage-c-output-dir", type=Path, help="Output directory for Stage C artifacts")
    parser.add_argument("--stage-c-verify", action="store_true", help="Run verification hooks after codegen")
    parser.add_argument("--stage-c-provider", default=DEFAULT_PROVIDER, help="LLM provider for Stage C codegen (default: deepseek)")
    parser.add_argument("--stage-c-model", default=None, help="LLM model for Stage C codegen (default: the provider's configured model)")
    parser.add_argument("--stage-c-temperature", type=float, default=0.0, help="LLM temperature for Stage C codegen")
    parser.add_argument("--stage-c-max-tokens", type=int, default=64000, help="LLM max tokens for Stage C codegen")
    parser.add_argument(
        "--stage-c-build-ws",
        dest="stage_c_build_ws",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After Stage C, run host syntax checks (Wireshark build or Lua syntax) (enabled by default)",
    )
    parser.add_argument(
        "--stage-c-ws-src-root",
        type=Path,
        help="Wireshark source root; default: pargen/data/tools/wireshark-4.4.11",
    )
    parser.add_argument(
        "--stage-c-ws-build-dir",
        type=Path,
        help="Wireshark build dir; default: <ws-src-root>/build",
    )
    parser.add_argument(
        "--stage-c-ws-download-url",
        type=str,
        help="Optional Wireshark source URL; defaults to WIRESHARK_DEFAULT_URL when auto-downloading",
    )
    parser.add_argument(
        "--stage-c-plugin-id",
        type=str,
        help="Plugin directory name under plugins/epan/<plugin_id>; default: <protocol>_bs",
    )
    parser.add_argument(
        "--stage-c-ws-isolate-plugins",
        dest="stage_c_ws_isolate_plugins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When running tshark exports, isolate Wireshark C plugin loading to the plugin under test "
            "(prevents cross-protocol plugin interference; enabled by default)"
        ),
    )
    parser.add_argument(
        "--stage-c-ws-plugin-c-mode",
        type=str,
        choices=("generated", "from_stage_c"),
        default="generated",
        help=(
            "Source of plugin.c for the Wireshark build: "
            "'generated' (default) = plugin.c is produced by the Wireshark CMake macro and the Stage C plugin.c is not copied; "
            "'from_stage_c' = use the plugin.c written by Stage C (copied into the source plugin directory and compiled as is)."
        ),
    )
    parser.add_argument(
        "--no-stage-c-pcap-to-texttree",
        dest="stage_c_pcap_to_texttree",
        action="store_false",
        default=True,
        help="Skip exporting tshark tree/tool.json after Stage C (enabled by default)",
    )
    parser.add_argument(
        "--stage-c-pcap",
        type=Path,
        help=(
            "PCAP file to feed into tshark for text tree export; default: "
            "data/<protocol>/inputs/pcap/<protocol>_100.pcap "
            "(fallback: data/<protocol>/inputs/pacap/<protocol>_100.pcap)"
        ),
    )
    parser.add_argument(
        "--stage-c-pcap-output",
        type=Path,
        help="Output path for text tree (default: <stage_c_output>/tool_<host>.txt)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-bert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Stage C dataset metric: L2-bert (BERTScore field-name similarity) (default: enabled)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-bert-threshold",
        type=float,
        default=0.85,
        help="BERTScore-F1 threshold for L2-bert (default: 0.85)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-bert-model-type",
        type=str,
        default=None,
        help="Optional bert-score model_type (default: bert-score's default for lang=en)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-bert-device",
        type=str,
        default="auto",
        help="Device for L2-bert: auto/cpu/cuda/mps/cuda:0/... (default: auto)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-bert-batch-size",
        type=int,
        default=64,
        help="BERTScore batch size for L2-bert (default: 64)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-bert-rescale-with-baseline",
        action="store_true",
        default=False,
        help="Enable bert-score rescale_with_baseline for L2-bert (default: disabled)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-bert-idf",
        action="store_true",
        default=False,
        help="Enable IDF weighting for L2-bert (default: disabled)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-bert-use-fast-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fast tokenizer for L2-bert when supported (default: false)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate context-aware LLM suggestions for human semantic review",
    )
    parser.add_argument(
        "--stage-c-metric-l2-judge-provider",
        type=str,
        default=None,
        help="LLM provider for L2 judge (default: --stage-c-provider)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-judge-model",
        type=str,
        default=None,
        help="LLM model for L2 judge (default: --stage-c-model/provider default)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-judge-temperature",
        type=float,
        default=0.0,
        help="LLM temperature for L2 judge (default: 0.0)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-judge-max-tokens",
        type=int,
        default=4096,
        help="LLM max tokens for L2 judge (default: 4096)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-judge-batch-size",
        type=int,
        default=40,
        help="Context-bound field pairs per L2 judge call (default: 40)",
    )
    parser.add_argument(
        "--stage-c-metric-l2-judge-cache",
        type=Path,
        default=None,
        help="Optional L2 judge JSON cache (default: <metric-out-dir>/l2_judge_cache.json)",
    )
    parser.add_argument(
        "--stage-c-metric-primary-l2",
        choices=["auto", "human", "judge"],
        default="human",
        help=(
            "Primary semantic F1 uses human review; auto/judge remain compatibility aliases. "
            "LLM suggestions are diagnostic and cannot supply final decisions."
        ),
    )
    parser.add_argument("--stage-c-metric-human-review", type=Path,
                        help="Human semantic review bound to the final exported parser output")
    parser.add_argument(
        "--stage-c-metric-relax-payload-split-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Relax Stage C dataset metric to treat payload-like fields (data/payload) as split/merge equivalent "
            "(e.g., Wireshark may split a payload into derived subfields while another host outputs one raw data field)."
        ),
    )
    parser.add_argument(
        "--stage-c-metric-relax-payload-keyword",
        action="append",
        default=None,
        help=(
            "Keyword token(s) that mark a field as payload-like for split/merge relaxation "
            "(default: data,payload). Can be repeated."
        ),
    )
    parser.add_argument(
        "--stage-c-ws-filter",
        type=str,
        default=None,
        help="Display filter for tshark export (default: <protocol>)",
    )
    parser.add_argument(
        "--stage-c-lua-entry",
        type=Path,
        help="Optional Lua entry file for wireshark_lua host (defaults to auto-detection).",
    )
    parser.add_argument(
        "--stage-c-scapy-entry",
        type=Path,
        help="Optional Scapy entry .py file for scapy host (defaults to auto-detection).",
    )
    parser.add_argument(
        "--stage-c-scapy-ports",
        action="append",
        default=None,
        help="Override Scapy registration ports (comma-separated; can repeat).",
    )
    parser.add_argument(
        "--stage-c-scapy-transport",
        type=str,
        choices=("auto", "tcp", "udp"),
        default="auto",
        help="Transport selection for Scapy export (default: auto).",
    )
    parser.add_argument(
        "--stage-c-zeek",
        type=Path,
        default=None,
        help="Optional zeek binary path for zeek_spicy runtime checks.",
    )
    parser.add_argument(
        "--stage-c-spicyz",
        type=Path,
        default=None,
        help="Optional spicyz binary path for zeek_spicy compilation.",
    )
    parser.add_argument(
        "--stage-c-zeek-runtime-check",
        action="store_true",
        default=False,
        help="Enable zeek -N / zeek -r runtime checks after spicyz compilation (default: disabled).",
    )
    parser.add_argument(
        "--stage-c-luac",
        type=Path,
        help="Optional luac path for wireshark_lua syntax checks.",
    )
    parser.add_argument(
        "--stage-c-lua",
        type=Path,
        help="Optional lua interpreter path for wireshark_lua syntax checks.",
    )
    parser.add_argument(
        "--stage-c-tshark",
        type=Path,
        help="Optional explicit tshark path; otherwise use build/run/tshark",
    )
    parser.add_argument(
        "--stage-c-semantic-verify",
        action="store_true",
        help="Run Stage C semantic validation (oracle=Stage B traffic_fixed_tree, gating uses Level1 F1)",
    )
    parser.add_argument(
        "--stage-c-semantic-fix",
        dest="stage_c_semantic_fix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run Stage C semantic repair loop (LLM patches Stage C plugin until Level1 F1 passes threshold) "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--stage-c-lua-load-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --stage-c-build-ws on wireshark_lua (default on, the build/load check of Sec. IV-C): after luac, "
            "load the plugin in tshark and dissect the first packets; Lua errors at load time or while dissecting "
            "(API misuse, dissector bugs) are treated as syntax errors for the bounded syntax repair. With the PKG "
            "suite (--stage-c-exp-constraint-suite-source pkg) the plugin is loaded on an empty capture, so the "
            "evaluation PCAP stays out of repair; dissection is exercised by the PKG-constructed messages. "
            "Needs no format tree, so it also works in --stage-c-doc-summary mode."
        ),
    )
    parser.add_argument(
        "--stage-c-semantic-dataset-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the semantic-repaired parser only if its metrics on the dataset GT (the evaluation "
            "reference) do not regress, else restore the pre-repair parser. Off by default: it selects on the "
            "evaluation GT, which generation and repair must not see."
        ),
    )
    parser.add_argument(
        "--stage-c-semantic-soft-fail",
        action="store_true",
        help=(
            "Do not fail the pipeline when Stage C semantic validation/repair does not reach the required threshold; "
            "print reports only (default: strict / fail on semantic mismatch)"
        ),
    )
    parser.add_argument(
        "--stage-c-semantic-max-rounds",
        type=int,
        default=5,
        help="Max semantic repair rounds (default: 5)",
    )
    parser.add_argument(
        "--stage-c-semantic-min-f1",
        type=float,
        default=0.99,
        help="Minimum Level1 F1 required for semantic pass (default: 0.99)",
    )
    parser.add_argument(
        "--stage-c-semantic-max-packets",
        type=int,
        default=0,
        help="Max packets to evaluate in semantic loop (0 = no limit)",
    )
    parser.add_argument(
        "--stage-c-semantic-require-full-parse",
        dest="stage_c_semantic_require_full_parse",
        action="store_true",
        default=True,
        help="Require full tree parse success when building GT (default: True)",
    )
    parser.add_argument(
        "--no-stage-c-semantic-require-full-parse",
        dest="stage_c_semantic_require_full_parse",
        action="store_false",
        help="Allow partial parse when building GT (not recommended; default is full-parse-only)",
    )
    parser.add_argument(
        "--stage-c-semantic-mutants-per-packet",
        type=int,
        default=0,
        help="(Optional) generate valid mutants per packet for extra semantic coverage (default: 0)",
    )
    parser.add_argument(
        "--stage-c-semantic-out-dir",
        type=Path,
        default=None,
        help="Output directory for semantic loop artifacts (default: <stage_c_output>/semantic/<host>)",
    )
    parser.add_argument(
        "--stage-c-exp-constraint-verify",
        dest="stage_c_exp_constraint_verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run exp_constraint suite validation after Stage C when available (enabled by default)",
    )
    parser.add_argument(
        "--stage-c-exp-constraint-generate-from-traffic",
        dest="stage_c_exp_constraint_generate_from_traffic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Generate a fresh exp_constraint conformance-counterexample suite from Stage B traffic traces and the base PCAP, "
            "then evaluate that suite instead of only relying on the static dataset suite (default: disabled)"
        ),
    )
    parser.add_argument(
        "--stage-c-exp-constraint-suite-source",
        choices=("static", "traffic", "pkg"),
        default="pkg",
        help=(
            "Source for Stage-C exp_constraint verification: static uses the checked-in suite, "
            "traffic generates from Stage-B traffic traces, and pkg generates from enumerated PKG paths "
            "(default: pkg). The --stage-c-exp-constraint-generate-from-traffic flag still selects traffic."
        ),
    )
    parser.add_argument(
        "--stage-c-exp-constraint-generated-suite-dir",
        type=Path,
        default=None,
        help="Optional output dir for the traffic-derived exp_constraint suite (default: <stage_c_output>/exp_constraint_generated_suite)",
    )
    parser.add_argument(
        "--stage-c-exp-constraint-pkg-max-paths",
        type=int,
        default=None,
        help="Optional cap on PKG parsing paths when --stage-c-exp-constraint-suite-source=pkg (default: no cap)",
    )
    parser.add_argument(
        "--stage-c-exp-constraint-max-negatives-per-constraint",
        type=int,
        default=2,
        help="Max constraint-violating cases per rule, one per boundary side (default: 2)",
    )
    parser.add_argument(
        "--stage-c-exp-constraint-fix",
        dest="stage_c_exp_constraint_fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After exp_constraint failures, ask the Stage C repair agent to patch the generated parser and "
            "rerun the suite with a no-regression gate (default: enabled)"
        ),
    )
    parser.add_argument(
        "--stage-c-exp-constraint-fix-max-rounds",
        type=int,
        default=5,
        help="Max exp_constraint verify/repair rounds when --stage-c-exp-constraint-fix is enabled (default: 5, B_code)",
    )
    parser.add_argument(
        "--stage-c-exp-constraint-fix-worst-k",
        type=int,
        default=8,
        help="Max failed exp_constraint cases to include in each repair prompt (default: 8)",
    )
    parser.add_argument(
        "--stage-c-exp-constraint-mode",
        choices=("binary", "per_constraint"),
        default="binary",
        help="exp_constraint evaluation mode: binary=normal/abnormal, per_constraint=tag coverage (default: binary)",
    )
    parser.add_argument(
        "--stage-c-exp-constraint-binary-positive-policy",
        choices=("unique_seed", "paired_seed"),
        default="unique_seed",
        help=(
            "How binary exp_constraint counts normal seed packets when reporting runtime confusion: "
            "unique_seed de-duplicates seed frames; paired_seed counts one normal counterpart per negative case "
            "(default: unique_seed)"
        ),
    )
    parser.add_argument(
        "--stage-c-exp-constraint-default-seed-frame-no",
        type=int,
        default=1,
        help=(
            "Default normal seed frame for binary exp_constraint mappings that define base_pcap but no seed_frame_no; "
            "set <=0 to disable automatic positive-seed confusion logging (default: 1)"
        ),
    )
    parser.add_argument(
        "--stage-c-exp-constraint-binary-require-bs-constraint",
        dest="stage_c_exp_constraint_binary_require_bs_constraint",
        action="store_true",
        default=False,
        help=(
            "In exp_constraint mode=binary for Wireshark hosts, require the output to contain 'BS_CONSTRAINT' "
            "(do not count generic expert-info / dissector bugs as success; default: disabled)"
        ),
    )
    parser.add_argument(
        "--stage-c-exp-constraint-strict",
        action="store_true",
        default=False,
        help="Fail the pipeline when exp_constraint suite has any failures (default: disabled)",
    )
    return parser.parse_args()


def _select_precomputed_tree_for_reuse(paths: ProtocolPaths, args: argparse.Namespace) -> Tuple[Optional[Path], str]:
    """
    Select a traffic-fixed tree to reuse as the starting point for Stage C.

    The traffic-fixed tree is the last Stage B artifact (after syntax + traffic repairs) and is a
    much better default than rebuilding Stage B from document sections every run.
    """
    stage_c_tree = getattr(args, "stage_c_tree", None)
    if stage_c_tree is not None:
        tree_path = Path(stage_c_tree)
        if not tree_path.exists():
            raise SystemExit(f"--stage-c-tree not found: {tree_path}")
        return tree_path, "stage_c_tree"

    if bool(getattr(args, "ablation_no_traffic", False)):
        precomputed_tree = args.semantic_fix_output or paths.syntax_fixed_tree_file
    else:
        precomputed_tree = args.semantic_fix_output or paths.traffic_fixed_tree_file
    if not precomputed_tree.exists():
        return None, ""
    if bool(getattr(args, "force_format_stage", False)):
        return None, ""
    if bool(getattr(args, "reuse_b_stage_output", False)):
        return precomputed_tree, "explicit"
    if bool(getattr(args, "no_auto_reuse_b_stage_output", False)):
        return None, ""

    # Auto-reuse unless the user explicitly requests a doc-stage rerun or provides direct Stage-B
    # inputs (which usually indicates intent to rebuild).
    if bool(getattr(args, "run_doc_stage", False)):
        return None, ""
    if getattr(args, "sections_file", None) or getattr(args, "raw_sections_file", None):
        return None, ""
    artifact_kind = "syntax_fixed" if bool(getattr(args, "ablation_no_traffic", False)) else "traffic_fixed"
    expected_signature = _build_stage_b_artifact_signature(paths, args, artifact_kind=artifact_kind)
    signature_ok, reason = _stage_b_artifact_signature_matches(precomputed_tree, expected_signature)
    if not signature_ok:
        print(
            f"\n[Stage B] Existing {precomputed_tree.name} is not auto-reusable: {reason}. "
            "Rebuilding Stage B. Use --reuse-b-stage-output to force reuse.",
            flush=True,
        )
        return None, ""
    return precomputed_tree, "auto"


def main() -> None:
    args = parse_args()
    if getattr(args, "spec_sections_only", False):
        args.spec_sections = True
    if getattr(args, "stage_a_only", False):
        args.run_doc_stage = True
    if getattr(args, "modbus", False):
        args.protocol = "modbus"
    if getattr(args, "wireshark_c_shortcut", False):
        args.stage_c_host = "wireshark_c"
    if getattr(args, "wireshark_lua_shortcut", False):
        args.stage_c_host = "wireshark_lua"
    if getattr(args, "scapy_shortcut", False):
        args.stage_c_host = "scapy"
    if getattr(args, "zeek_spicy_shortcut", False):
        args.stage_c_host = "zeek_spicy"

    load_api_keys(args.config, set_env=True)
    args.protocol = _normalize_protocol_name(args.protocol) or str(args.protocol)
    protocol_dir = DATA_DIR / args.protocol
    if not protocol_dir.exists():
        raise SystemExit(f"Protocol directory not found: {protocol_dir}")

    if bool(getattr(args, "ablation_no_spec", False)):
        _validate_no_spec_args(args)
    if bool(getattr(args, "spec_sections", False)):
        clashing = [flag for flag, on in (
            ("--sections-file", args.sections_file is not None),
            ("--raw-sections-file", args.raw_sections_file is not None),
            ("--sections-subset", args.sections_subset is not None),
            ("--run-doc-stage", bool(getattr(args, "run_doc_stage", False))),
            ("--ablation-no-spec", bool(getattr(args, "ablation_no_spec", False))),
        ) if on]
        if clashing:
            raise SystemExit(f"--spec-sections cannot be combined with: {', '.join(clashing)}")
    if bool(getattr(args, "ablation_no_traffic", False)):
        args.enable_semantic_fix = False
        args.stage_c_semantic_fix = False
        args.stage_c_semantic_verify = False

    if bool(getattr(args, "ablation_no_spec", False)) and bool(getattr(args, "ablation_no_traffic", False)):
        output_variant = "no_spec_no_traffic"
    elif bool(getattr(args, "ablation_no_spec", False)):
        output_variant = "no_spec"
    elif bool(getattr(args, "ablation_no_traffic", False)):
        output_variant = "no_traffic"
    else:
        output_variant = None
    paths = ProtocolPaths(protocol_dir, output_variant=output_variant)
    setup_logging(
        console_level=logging.WARNING,
        file_path=paths.logs_dir / "pipeline.log",
        replace_existing=True,
    )
    if bool(getattr(args, "ablation_no_spec", False)):
        sections_stub, raw_stub = _ensure_no_spec_stub_files(paths)
        paths.sections_override = sections_stub
        paths.raw_override = raw_stub
        print(
            f"\n[Ablation] no-spec enabled: using empty section stubs at {sections_stub} "
            f"and isolated outputs under {paths.outputs_dir}",
            flush=True,
        )
    elif bool(getattr(args, "ablation_no_traffic", False)):
        base_sections, base_raw = _resolve_base_stage_a_sections(protocol_dir)
        if args.sections_file is None and args.sections_subset is None and not bool(getattr(args, "run_doc_stage", False)):
            if base_sections is not None:
                paths.sections_override = base_sections
            if args.raw_sections_file is None and base_raw is not None:
                paths.raw_override = base_raw
        print(
            f"\n[Ablation] no-traffic enabled: reusing default Stage A sections"
            f"{f' from {paths.sections_override}' if paths.sections_override else ''} "
            f"and isolated outputs under {paths.outputs_dir}. "
            "Stage B3 traffic fix and Stage C semantic repair/verify are disabled.",
            flush=True,
        )
    if args.sections_file:
        paths.sections_override = args.sections_file
    if args.raw_sections_file:
        paths.raw_override = args.raw_sections_file
    # If the user explicitly supplies a sections subset, treat it as the Stage-B input
    # unless they also supplied explicit Stage-A section files via --sections-file/--raw-sections-file.
    # Explicit section inputs take precedence over cached document_sections.json.
    if args.sections_subset and not args.sections_file:
        paths.sections_override = args.sections_subset
    if args.sections_subset and not args.raw_sections_file:
        paths.raw_override = args.sections_subset
    spec_receipt = None
    if bool(getattr(args, "spec_sections", False)):
        from pargen.spec_sections import build as build_spec_sections

        if getattr(args, "spec_select", "all") == "toc_llm":
            from pargen.spec_sections import chat_completer

            toc_model = args.stage_b_initial_model or os.getenv("DEEPSEEK_MODEL") or ""
            toc_complete = chat_completer(
                os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
                os.getenv("DEEPSEEK_API_KEY", ""),
                toc_model,
                temperature=float(args.stage_b_initial_temperature if args.stage_b_initial_temperature is not None else 0.0),
                max_tokens=int(os.getenv("DEEPSEEK_PRO_MAX_TOKENS_CAP") or args.stage_b_initial_max_tokens or 65536),
                timeout=int(os.getenv("AI_READ_TIMEOUT", "3600")),
                log_dir=paths.sections_dir,
            )
            spec_path, spec_receipt, spec_reused = build_spec_sections(
                paths.pdfs_dir, paths.sections_dir / "spec_sections_toc.json", select="toc_llm",
                protocol=paths.root.name.split("_", 1)[0], complete=toc_complete, model=toc_model,
            )
        else:
            spec_path, spec_receipt, spec_reused = build_spec_sections(
                paths.pdfs_dir, paths.sections_dir / "spec_sections.json"
            )
        paths.sections_override = spec_path
        paths.raw_override = spec_path
        print(
            f"\n[Stage A] {'Reused' if spec_reused else 'Wrote'} {spec_path}: {spec_receipt['sections']} sections, "
            f"{spec_receipt['chars']} chars, cut at the documents' headings ({spec_receipt['selection']}).",
            flush=True,
        )
        if getattr(args, "spec_sections_only", False):
            return

    pdfs = args.pdfs or sorted(paths.pdf_dir.glob("*.pdf"))
    claude_key = args.claude_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    deepseek_key = args.deepseek_key or os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        os.environ.setdefault("DEEPSEEK_API_KEY", deepseek_key)

    timing: dict = {
        "protocol": str(args.protocol),
        "host": str(getattr(args, "stage_c_host", "") or "unknown_host"),
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "started_at": float(time.time()),
        "status": "running",
        "stages": {},
        "flags": {
            "run_doc_stage": bool(getattr(args, "run_doc_stage", False)),
            "stage_a_only": bool(getattr(args, "stage_a_only", False)),
            "force_format_stage": bool(getattr(args, "force_format_stage", False)),
            "enable_fix_stage": bool(getattr(args, "enable_fix_stage", False)),
            "enable_semantic_fix": bool(getattr(args, "enable_semantic_fix", False)),
            "ablation_no_spec": bool(getattr(args, "ablation_no_spec", False)),
            "ablation_no_traffic": bool(getattr(args, "ablation_no_traffic", False)),
            "reuse_b_stage_output": bool(getattr(args, "reuse_b_stage_output", False)),
            "reuse_c_stage_output": bool(getattr(args, "reuse_c_stage_output", False)),
            "stage_c_build_ws": bool(getattr(args, "stage_c_build_ws", False)),
            "stage_c_pcap_to_texttree": bool(getattr(args, "stage_c_pcap_to_texttree", False)),
            "stage_c_semantic_fix": bool(getattr(args, "stage_c_semantic_fix", False)),
            "stage_c_semantic_verify": bool(getattr(args, "stage_c_semantic_verify", False)),
            "stage_c_exp_constraint_verify": bool(getattr(args, "stage_c_exp_constraint_verify", False)),
            "stage_c_exp_constraint_suite_source": str(
                getattr(args, "stage_c_exp_constraint_suite_source", "pkg") or "pkg"
            ),
            "stage_c_exp_constraint_generate_from_traffic": bool(
                getattr(args, "stage_c_exp_constraint_generate_from_traffic", False)
            ),
            "stage_c_exp_constraint_fix": bool(getattr(args, "stage_c_exp_constraint_fix", False)),
            "stage_c_exp_constraint_fix_max_rounds": int(getattr(args, "stage_c_exp_constraint_fix_max_rounds", 3) or 3),
        },
    }

    if spec_receipt is not None:
        timing.setdefault("inputs", {})["spec_sections"] = dict(spec_receipt, path=str(paths.sections_file))
    subset_path = args.sections_subset or paths.sections_subset_file
    legacy_subset_path = paths.legacy_sections_subset_file

    doc_summary_mode = bool(getattr(args, "stage_c_doc_summary", False))
    if doc_summary_mode:
        # Doc-summary ablation is tree-free; ignore any Stage-B reuse requests.
        format_tree_path, reuse_mode = None, ""
        if getattr(args, "stage_c_tree", None) is not None:
            print("\n[Stage C] NOTE: --stage-c-doc-summary ignores --stage-c-tree (doc-summary mode is tree-free).")
        if getattr(args, "reuse_b_stage_output", False):
            print("\n[Stage C] NOTE: --stage-c-doc-summary ignores --reuse-b-stage-output (doc-summary mode is tree-free).")
    else:
        format_tree_path, reuse_mode = _select_precomputed_tree_for_reuse(paths, args)
    timing["doc_summary_mode"] = bool(doc_summary_mode)
    timing["reuse_mode"] = str(reuse_mode or "")
    timing["output_variant"] = str(paths.output_variant or "")
    timing.setdefault("inputs", {})["format_tree_path"] = str(format_tree_path) if format_tree_path is not None else None
    if format_tree_path is not None:
        if reuse_mode == "stage_c_tree":
            print(
                f"\n[Stage B] Skipped (using --stage-c-tree={format_tree_path} (mode={reuse_mode})); "
                "skipping Stages A/B."
            )
        else:
            print(
                f"\n[Stage B] Reusing existing traffic-fixed tree at {format_tree_path} (mode={reuse_mode}); "
                "skipping Stages A/B. Use --force-format-stage to rebuild."
            )

    if format_tree_path is None:
        if args.run_doc_stage:
            if not args.claude_key and not claude_key:
                raise SystemExit("Stage A requested, but no Claude API key provided (--claude-key or configure pargen/config/api_config.json).")
            with _record_stage_time(timing, "stage_a_wall_s"):
                run_stage_a(paths, pdfs, claude_key, deepseek_key)
            paths.sections_override = paths.sections_file
            paths.raw_override = paths.raw_sections_file
            stage_a_sum = _sum_section_processing_time(paths.sections_file)
            if stage_a_sum is not None:
                timing.setdefault("derived", {})["stage_a_sections_processing_time_s"] = float(stage_a_sum)
            if getattr(args, "stage_a_only", False):
                print("\n[Stage A] Completed (--stage-a-only); exiting before Stage B/C.")
                timing["status"] = "ok"
                timing["ended_at"] = float(time.time())
                report_path = _write_timing_report(paths, timing)
                if report_path is not None:
                    print(f"\n[Timing] Wrote timing report: {report_path}")
                return
        else:
            sections_candidate = paths._first_existing(
                paths.sections_override,
                paths.sections_dir / "document_sections.json",
                subset_path,
                paths.legacy_sections_file,
                legacy_subset_path,
            )
            raw_candidate = paths._first_existing(
                paths.raw_override,
                paths.sections_dir / "document_sections_raw.json",
                subset_path,
                paths.legacy_raw_sections_file,
                legacy_subset_path,
            )

            missing_sections = not (sections_candidate and sections_candidate.exists())
            missing_raw = not (raw_candidate and raw_candidate.exists())
            if missing_sections or missing_raw:
                subset_candidate = paths._first_existing(subset_path, legacy_subset_path)
                if subset_candidate and subset_candidate.exists():
                    paths.sections_override = subset_candidate
                    paths.raw_override = subset_candidate
                    print(f"[Stage A] Using fallback subset file: {subset_candidate}")
                else:
                    raise SystemExit(
                        "Sections files missing. Re-run with --run-doc-stage (and provide API keys) "
                        "or supply paths via --sections-file / --raw-sections-file."
                    )
            else:
                if sections_candidate:
                    paths.sections_override = sections_candidate
                if raw_candidate:
                    paths.raw_override = raw_candidate
                print("\n[Stage A] Skipped (using existing sections).")

        stage_a_sum = _sum_section_processing_time(paths.sections_file)
        if stage_a_sum is not None:
            timing.setdefault("derived", {})["stage_a_sections_processing_time_s"] = float(stage_a_sum)

        if doc_summary_mode:
            summary_source = args.stage_c_doc_summary_source or paths.sections_file
            if not Path(summary_source).exists():
                raise SystemExit(f"--stage-c-doc-summary source not found: {summary_source}")
            doc_summary_text = _load_doc_summaries_from_sections(Path(summary_source))
            if not doc_summary_text.strip():
                raise SystemExit(f"--stage-c-doc-summary produced empty summaries from: {summary_source}")
            with _record_stage_time(timing, "stage_c_total_s"):
                run_stage_c(Path(summary_source), paths, args, timing=timing, doc_summary_text=doc_summary_text)
            print("\nPipeline completed.")
            timing["status"] = "ok"
            timing["ended_at"] = float(time.time())
            report_path = _write_timing_report(paths, timing)
            if report_path is not None:
                print(f"\n[Timing] Wrote timing report: {report_path}")
            return

        if getattr(args, "force_format_stage_llm", False):
            os.environ["PARGEN_STEP2_FORCE_REBUILD"] = "1"
        with _record_stage_time(timing, "stage_b1_build_tree_s"):
            tree_cache_path = run_stage_b(paths, force_rebuild=args.force_format_stage, args=args)

        format_tree_path = paths.format_tree_file
        if args.enable_fix_stage:
            fix_sections_path = _stage_b_effective_fix_sections_path(paths, args)
            if fix_sections_path is None:
                fix_sections_path = args.sections_subset or subset_path
            if not fix_sections_path.exists():
                fix_sections_path = paths.sections_file
            if not fix_sections_path.exists():
                raise SystemExit("No sections file available for Step2 fix. Provide --sections-subset or --sections-file.")
            base_fix_sections_path = _stage_b_fix_sections_path(paths, args)
            if base_fix_sections_path is not None and Path(fix_sections_path) != Path(base_fix_sections_path):
                print(f"[Stage B2] Using active/core scoped sections for fix: {fix_sections_path}")

            fix_output_dir = args.fix_output_dir or paths.step2_results_dir
            fix_log_dir = args.fix_log_dir or paths.step2_logs_dir
            fix_output_dir.mkdir(parents=True, exist_ok=True)
            fix_log_dir.mkdir(parents=True, exist_ok=True)

            # Stage B2: Syntax-only fixing (no traffic)
            from pargen.protocol_knowledge.repair.step2_fix import run_step2_fix

            with _record_stage_time(timing, "stage_b2_syntax_fix_s"):
                fixed_tree_path = run_step2_fix(
                    cache_path=tree_cache_path,
                    sections_path=fix_sections_path,
                    agent_provider=args.fix_provider,
                    output_dir=fix_output_dir,
                    log_dir=fix_log_dir,
                    max_llm_calls=args.fix_max_llm_calls,
                    traffic_path=None,
                    max_packets=0,
                    target_message_type=None,
                    use_hybrid_validator=False,
                )
            print(f"[Stage B2] Syntax-only fixed tree saved to {fixed_tree_path}")

            # Re-validate the saved tree. Refinement keeps issues left after the static rounds as internal
            # diagnostics and continues with the message-instantiated rounds (Sec. IV-B);
            # STEP2_ABORT_ON_RESIDUAL_STATIC_ERRORS=1 restores the earlier stop before Stage B3.
            try:
                from pargen.protocol_knowledge.validation.syntax_validator import validate_protocol_tree

                saved_payload = json.loads(fixed_tree_path.read_text(encoding="utf-8"))
                tree_payload = saved_payload.get("protocol_tree") if isinstance(saved_payload, dict) else saved_payload
                static_report = validate_protocol_tree(json.dumps(tree_payload))
                static_errors = [i for i in static_report.issues.values() if i.severity.name == "ERROR"]
            except Exception as exc:
                raise RuntimeError("[Stage B2] Saved-tree validation could not complete; refusing to proceed.") from exc
            if static_errors:
                abort = _env_bool("STEP2_ABORT_ON_RESIDUAL_STATIC_ERRORS", False)
                print(f"[Stage B2] Post-save static validation found {len(static_errors)} errors; "
                      + ("aborting before traffic fix." if abort else "kept as diagnostics, continuing with Stage B3."))
                for iss in static_errors:
                    print(f"  - {iss.description}")
                if abort:
                    raise SystemExit(1)
                timing.setdefault("stage_b", {})["b2_residual_static_errors"] = sorted(
                    f"{getattr(i.type, 'name', i.type)}:{getattr(i, 'code', '') or ''}" for i in static_errors
                )

            format_tree_path = fixed_tree_path
        else:
            print("\n[Stage B2] Skipped (inline fixes enabled during Stage B).")
            fix_sections_path = _stage_b_effective_fix_sections_path(paths, args)
            if fix_sections_path is None:
                fix_sections_path = _stage_b_fix_sections_path(paths, args) or subset_path

        # Stage B3: Traffic semantic fixing (separate MCTS pass, per packet)
        if bool(getattr(args, "ablation_no_traffic", False)):
            if format_tree_path is not None:
                stable_syntax_tree = paths.syntax_fixed_tree_file
                stable_syntax_tree.parent.mkdir(parents=True, exist_ok=True)
                src_tree = Path(format_tree_path)
                if src_tree.resolve() != stable_syntax_tree.resolve():
                    shutil.copy2(src_tree, stable_syntax_tree)
                syntax_signature = _build_stage_b_artifact_signature(
                    paths,
                    args,
                    artifact_kind="syntax_fixed",
                    traffic_file=None,
                )
                syntax_meta = _write_stage_b_artifact_metadata(
                    stable_syntax_tree,
                    input_signature=syntax_signature,
                )
                print(f"[Stage B2] Syntax-fixed metadata saved to {syntax_meta}")
                format_tree_path = stable_syntax_tree
            print("\n[Stage B3] Skipped (--ablation-no-traffic; traffic semantic fixing disabled).")
        elif args.enable_semantic_fix:
            semantic_traffic_path = args.traffic or paths.traffic_file
            if not semantic_traffic_path.exists():
                print(f"\n[Stage B3] Skipped (traffic file not found at {semantic_traffic_path})")
            else:
                b3_output = args.semantic_fix_output or (paths.step2_results_dir / "traffic_fixed_tree.json")
                b3_output.parent.mkdir(parents=True, exist_ok=True)
                print("\n[Stage B3] Traffic semantic fixing (separate pass)...")
                # Traffic semantic fixing reuses the same document sections as context for the fix agent.
                # `batch_size` here refers to how many *document sections* are included per LLM prompt
                # batch (NOT packet batching). Default: include all sections in a single batch.
                try:
                    _sections_payload = json.loads(fix_sections_path.read_text(encoding="utf-8"))
                    if isinstance(_sections_payload, dict):
                        _sections_payload = _sections_payload.get("sections", [])
                    section_count = len(_sections_payload) if isinstance(_sections_payload, list) else None
                except Exception:
                    section_count = None
                raw_section_batch = os.getenv("STEP2_TRAFFIC_SECTION_BATCH_SIZE", "").strip()
                if raw_section_batch:
                    try:
                        traffic_section_batch_size = int(raw_section_batch)
                    except Exception:
                        traffic_section_batch_size = section_count or 999999
                else:
                    traffic_section_batch_size = section_count or 999999
                print(
                    f"[Stage B3] section_batch_size={traffic_section_batch_size} (all_sections={section_count}; set STEP2_TRAFFIC_SECTION_BATCH_SIZE to override)",
                    flush=True,
                )
                from pargen.protocol_knowledge.run_traffic_fix import run_traffic_fix

                with _record_stage_time(timing, "stage_b3_traffic_fix_s"):
                    traffic_fixed = run_traffic_fix(
                        tree_path=Path(format_tree_path),
                        traffic_path=semantic_traffic_path,
                        sections_path=fix_sections_path,
                        provider=args.fix_provider,
                        output_path=b3_output,
                        max_llm_calls=int(os.getenv("STEP2_TRAFFIC_MAX_LLM_CALLS", "20")),
                        max_packets=args.semantic_max_packets,
                        target_message_type=args.target_message_type,
                        per_issue_mcts=True,
                        batch_size=traffic_section_batch_size,
                    )
                traffic_signature = _build_stage_b_artifact_signature(
                    paths,
                    args,
                    artifact_kind="traffic_fixed",
                    traffic_file=semantic_traffic_path,
                )
                traffic_meta = _write_stage_b_artifact_metadata(
                    traffic_fixed,
                    input_signature=traffic_signature,
                )
                format_tree_path = traffic_fixed
                print(f"[Stage B3] Traffic-fixed tree saved to {traffic_fixed}")
                print(f"[Stage B3] Traffic-fixed metadata saved to {traffic_meta}")
        else:
            print("\n[Stage B3] Skipped (semantic fixing disabled).")

        if format_tree_path is not None:
            coverage_base_sections_path = _stage_b_fix_sections_path(paths, args) or subset_path or paths.sections_file
            coverage_scaffold_summary = _write_stage_b_coverage_only_scaffold_artifact(
                Path(format_tree_path),
                paths=paths,
                sections_path=coverage_base_sections_path,
            )
            timing.setdefault("stage_b", {})["coverage_only_scaffold"] = coverage_scaffold_summary
            if coverage_scaffold_summary.get("applied"):
                print(
                    "[Stage B] Coverage-only scaffold artifact written "
                    f"({coverage_scaffold_summary.get('changes_count', 0)} changes; "
                    f"indices {coverage_scaffold_summary.get('coverage_only_indices')}; "
                    f"path {coverage_scaffold_summary.get('coverage_tree_path')})."
                )
            coverage_obligation_summary = _run_stage_b_coverage_obligation_driver(
                Path(format_tree_path),
                paths=paths,
                coverage_scaffold_summary=coverage_scaffold_summary,
                traffic_path=args.traffic or paths.traffic_file,
                provider=args.fix_provider,
            )
            timing.setdefault("stage_b", {})["coverage_obligation_driver"] = coverage_obligation_summary
            if coverage_obligation_summary.get("attempted"):
                status = "accepted" if coverage_obligation_summary.get("accepted") else "rejected"
                print(
                    "[Stage B] Coverage-obligation driver "
                    f"{status}; promoted={coverage_obligation_summary.get('promoted')} "
                    f"reason={coverage_obligation_summary.get('reason')} "
                    f"path={coverage_obligation_summary.get('coverage_obligation_tree_path')}"
                )

    if getattr(args, "stage_b_only", False):
        print(f"\n[Stage B] Completed (--stage-b-only); exiting before Stage C. Tree: {format_tree_path}")
        timing["status"] = "ok"
        timing["ended_at"] = float(time.time())
        report_path = _write_timing_report(paths, timing)
        if report_path is not None:
            print(f"\n[Timing] Wrote timing report: {report_path}")
        return

    with _record_stage_time(timing, "stage_c_total_s"):
        run_stage_c(format_tree_path, paths, args, timing=timing)

    print("\nPipeline completed.")
    if "derived" not in timing or "stage_a_sections_processing_time_s" not in (timing.get("derived") or {}):
        stage_a_sum = _sum_section_processing_time(paths.sections_file)
        if stage_a_sum is not None:
            timing.setdefault("derived", {})["stage_a_sections_processing_time_s"] = float(stage_a_sum)
    timing["status"] = "ok"
    timing["ended_at"] = float(time.time())
    report_path = _write_timing_report(paths, timing)
    if report_path is not None:
        print(f"\n[Timing] Wrote timing report: {report_path}")


if __name__ == "__main__":
    main()
