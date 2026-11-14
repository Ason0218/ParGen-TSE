#!/usr/bin/env python3
"""
Traffic-focused refinement entry point.
Runs hybrid validation (static + dynamic traffic parsing) through the MCTS fix loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import hashlib
import logging
import os
import signal
import sys
import re
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Sequence, Iterable
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pargen.paths import STEP2_CACHE_DIR, STEP2_FIX_CACHE_DIR, DATA_DIR, LOGS_DIR
from pargen.protocol_knowledge.tree_utils import normalize_protocol_tree
from pargen.protocol_knowledge.semantic_integrity import binding_scope, scope_preserved, layout_binding_losses
from pargen.protocol_knowledge.refinement_result import publish_refinement
from pargen.protocol_knowledge.repair.agent import EnhancedPureAIAgent
from pargen.protocol_knowledge.repair.refinement import mcts_fix_tree, run_full_validation
from pargen.protocol_knowledge.repair.modes import get_mode
from pargen.protocol_knowledge.validation.syntax_validator import (
    validate_protocol_tree,
    Issue,
    ValidationReport,
    IssueType,
    Severity,
)
from pargen.protocol_knowledge.repair.traffic_length_inference import (
    infer_size_bits_candidates_from_report,
    infer_size_formula_for_node,
    infer_payload_fill_candidates_from_report,
)
from pargen.protocol_knowledge.message_validation.semantic_validator import (
    SemanticValidator,
    _load_packets_from_path,
    export_parsing_traces,
)
from pargen.protocol_knowledge.message_validation.interpreter import (
    ConstraintViolationError,
    DynamicTreeInterpreter,
    LengthMismatchError,
    NodeParseError,
    RoutingError,
)
from pargen.logging_utils import setup_logging

from pargen.protocol_knowledge.repair.evidence import retrieve_evidence_sections

logger = logging.getLogger(__name__)


def _traffic_mcts_round_timeout_sec() -> float:
    raw = (
        os.getenv("STEP2_TRAFFIC_MCTS_ROUND_TIMEOUT_SEC")
        or os.getenv("STEP2_TRAFFIC_MCTS_WALL_TIMEOUT_SEC")
        or "900"
    )
    try:
        value = float(str(raw).strip())
    except Exception:
        value = 900.0
    return max(0.0, value)


class _TrafficMCTSDeadline(BaseException):
    """Cancel the whole search, bypassing nested provider/patch retry handlers."""


@contextmanager
def _traffic_mcts_wall_timeout(timeout_sec: float, label: str):
    if (
        timeout_sec <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(_signum, _frame):
        raise _TrafficMCTSDeadline(f"{label} exceeded wall-clock timeout {timeout_sec:g}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer and previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _build_node_lookup(tree: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    nodes = tree.get("nodes", []) if isinstance(tree, dict) else []
    lookup: Dict[Any, Dict[str, Any]] = {}
    for node in nodes:
        nid = node.get("node_id")
        if nid is None:
            continue
        lookup[nid] = node
        lookup[str(nid)] = node
    return lookup


def _is_aggregator_node(node: Optional[Dict[str, Any]], lookup: Dict[Any, Dict[str, Any]]) -> bool:
    if not node:
        return False
    ntype = str(node.get("node_type", "") or "").lower()
    if ntype in {"protocol", "message"}:
        return True
    children = node.get("children_ids") or []
    for cid in children:
        child = lookup.get(cid) or lookup.get(str(cid))
        if child and str(child.get("node_type", "")).lower() == "variant":
            return True
    return False


def _setup_logging() -> None:
    """Attach a debug file handler for traffic fixing runs."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / "traffic_fix_debug.log"
        setup_logging(
            console_level=logging.WARNING,
            file_path=log_path,
            file_level=logging.DEBUG,
            replace_existing=False,
        )
        logger.debug("Debug logging initialized at %s", log_path)
    except Exception:
        logger.warning("Failed to initialize debug file logging", exc_info=True)


def _smoke_test_interpreter(tree: Dict[str, Any], samples: List[bytes], max_samples: int = 5) -> Optional[str]:
    """
    Run a quick parse on a handful of packets to catch interpreter-level issues
    (e.g., node-id mismatches) before invoking MCTS.
    """
    if not samples:
        return None
    interp = DynamicTreeInterpreter(tree)
    limit = min(max_samples, len(samples))
    for idx in range(limit):
        payload = samples[idx]
        try:
            success, ctx, err = interp.parse(payload)
        except Exception as exc:  # implementation/runtime errors
            return f"Interpreter exception on packet {idx}: {exc!r}"
        if err:
            msg = str(err)
            if "not found" in msg.lower():
                return f"Interpreter structural error on packet {idx}: {msg}"
        # semantic failures are fine; only structural/implementation errors abort
    return None


def _select_diverse_sample_payloads(
    *,
    tree: Dict[str, Any],
    packets: Sequence[bytes],
    sample_budget: int,
    target_message_type: Optional[str] = None,
) -> Tuple[List[bytes], Dict[str, Any]]:
    """
    Pick up to `sample_budget` packets while maximizing diversity by parse-path signature.

    This avoids the previous "first N packets" bias, which can miss variants and
    cause traffic_fix to overfit or repeatedly chase the wrong failure mode.
    """
    total = len(packets)
    if total == 0:
        return [], {"strategy": "none", "selected": [], "groups": 0, "total": 0}

    if sample_budget <= 0 or sample_budget >= total:
        return list(packets), {"strategy": "all", "selected": list(range(total)), "groups": total, "total": total}

    interp = DynamicTreeInterpreter(tree, target_message_type=target_message_type)
    prefix_len = max(1, int(os.getenv("STEP2_TRAFFIC_SAMPLE_PATH_PREFIX_LEN", "16")))

    def _coerce_path(ctx: Any) -> Tuple[int, ...]:
        raw = getattr(ctx, "parsing_path", None) or []
        out: List[int] = []
        for nid in raw:
            try:
                out.append(int(nid))
            except Exception:
                continue
        return tuple(out)

    def _failure_kind(err: Any) -> str:
        if err is None:
            return "ok"
        if isinstance(err, RoutingError):
            return "routing"
        if isinstance(err, ConstraintViolationError):
            return "constraint"
        if isinstance(err, LengthMismatchError):
            return "length_mismatch"
        if isinstance(err, NodeParseError):
            return "node_error"
        if isinstance(err, str):
            if err.startswith("Incomplete parse:"):
                return "coverage_tail_gap"
            return "error"
        return type(err).__name__.lower()

    PacketKey = Tuple[str, Optional[int], Optional[int], Tuple[int, ...], Tuple[int, ...]]
    # key = (failure_kind, error_node_id, routing_selector_id, routing_candidates, path_prefix)
    groups: Dict[PacketKey, List[int]] = defaultdict(list)
    ok_count = 0
    fail_count = 0

    for idx, payload in enumerate(packets):
        success, ctx, err = interp.parse(payload)
        kind = _failure_kind(err)
        if success and err is None:
            ok_count += 1
        else:
            fail_count += 1
        path = _coerce_path(ctx)
        path_prefix = path[:prefix_len]

        node_id: Optional[int] = None
        selector_id: Optional[int] = None
        candidates: Tuple[int, ...] = tuple()

        if isinstance(err, RoutingError):
            try:
                node_id = int(getattr(err, "node_id", None))
            except Exception:
                node_id = None
            try:
                selector_id_raw = getattr(err, "selector_id", None)
                selector_id = int(selector_id_raw) if selector_id_raw is not None else None
            except Exception:
                selector_id = None
            raw_cands = getattr(err, "candidate_variants", None) or []
            try:
                candidates = tuple(int(x) for x in raw_cands)
            except Exception:
                candidates = tuple()
        else:
            try:
                node_id_raw = getattr(err, "node_id", None)
                node_id = int(node_id_raw) if node_id_raw is not None else None
            except Exception:
                node_id = None

        groups[(kind, node_id, selector_id, candidates, path_prefix)].append(idx)

    # Prioritize failing groups, then large groups (coverage).
    group_items = sorted(
        groups.items(),
        key=lambda kv: (
            0 if kv[0][0] != "ok" else 1,
            -len(kv[1]),
            kv[1][0],
        ),
    )

    selected_indices: List[int] = []
    for _key, indices in group_items:
        selected_indices.append(indices[0])
        if len(selected_indices) >= sample_budget:
            break

    selected_indices = sorted(set(selected_indices))
    selected_payloads = [packets[i] for i in selected_indices]

    debug = {
        "strategy": "diverse_parse_path",
        "total": total,
        "ok": ok_count,
        "failed": fail_count,
        "groups": len(groups),
        "selected": selected_indices,
        "prefix_len": prefix_len,
    }
    return selected_payloads, debug


def _coerce_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return int(candidate, 0)
        except ValueError:
            return None
    return None


def _traffic_failed_samples(report: Any) -> int:
    try:
        total = int(getattr(report, "traffic_total_samples", 0) or 0)
        succ = int(getattr(report, "traffic_successful_samples", 0) or 0)
        if total:
            return max(0, total - succ)
    except Exception:
        pass
    try:
        return len(getattr(report, "traffic_failures", []) or [])
    except Exception:
        return 0


def _traffic_coverage_ratio(report: Any) -> float:
    ratios = getattr(report, "traffic_content_coverage_ratio_per_sample", None) or ()
    if ratios:
        try:
            vals = [float(x) for x in ratios if x is not None]
            if vals:
                return sum(vals) / float(len(vals))
        except Exception:
            pass
    content_bits = getattr(report, "traffic_content_covered_bits_per_sample", None) or ()
    total_bits = getattr(report, "traffic_total_bits_per_sample", None) or ()
    if content_bits and total_bits:
        cov_sum = 0.0
        count = 0
        for cb, tb in zip(content_bits, total_bits):
            try:
                cb_val = float(cb)
                tb_val = float(tb)
            except Exception:
                continue
            if tb_val <= 0:
                continue
            cov_sum += min(1.0, cb_val / tb_val)
            count += 1
        if count:
            return cov_sum / float(count)
    return 0.0


def _traffic_error_issue_count(report: Any) -> int:
    try:
        return sum(
            1
            for issue in (getattr(report, "issues", {}) or {}).values()
            if getattr(issue, "severity", None) == Severity.ERROR
        )
    except Exception:
        return 0


def _traffic_accept_score(report: Any) -> Tuple[int, float, int, int, int, int]:
    failed = _traffic_failed_samples(report)
    coverage = _traffic_coverage_ratio(report)
    try:
        gap_bits = int(getattr(report, "traffic_total_coverage_gap_bits", 0) or 0)
    except Exception:
        gap_bits = 0
    try:
        len_err = int(getattr(report, "traffic_length_total_abs_error_bits", 0) or 0)
    except Exception:
        len_err = 0
    try:
        overflow = int(getattr(report, "traffic_overflow_length_bits", 0) or 0)
    except Exception:
        overflow = 0
    return (failed, -coverage, gap_bits, len_err, overflow, _traffic_error_issue_count(report))

def _traffic_scalar_score(report: Any) -> float:
    """
    Scalarize traffic metrics into a single value (higher is better).

    This is used to accept intermediate trees during traffic fixing even when
    `failed` temporarily increases (multi-step repairs), as long as global
    coverage/length metrics improve substantially.
    """
    try:
        total = int(getattr(report, "traffic_total_samples", 0) or 0)
    except Exception:
        total = 0
    failed = _traffic_failed_samples(report)
    succ = max(0, total - failed)
    cov = _traffic_coverage_ratio(report)
    try:
        gap_bits = float(getattr(report, "traffic_total_coverage_gap_bits", 0) or 0)
    except Exception:
        gap_bits = 0.0
    try:
        len_err = float(getattr(report, "traffic_length_total_abs_error_bits", 0) or 0)
    except Exception:
        len_err = 0.0
    try:
        overflow = float(getattr(report, "traffic_overflow_length_bits", 0) or 0)
    except Exception:
        overflow = 0.0

    def _w(env: str, default: float) -> float:
        raw = os.getenv(env)
        if raw is None or not str(raw).strip():
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    # Defaults match pargen/protocol_knowledge/repair/mcts.py:MCTSConfig
    w_success = _w("STEP2_MCTS_TRAFFIC_WEIGHT_SUCCESS", 5.0)
    w_cov = _w("STEP2_MCTS_TRAFFIC_WEIGHT_COVERAGE", 1.0)
    w_gap = _w("STEP2_MCTS_TRAFFIC_WEIGHT_GAP", 0.02)
    w_len = _w("STEP2_MCTS_TRAFFIC_WEIGHT_LENGTH_ERROR", 0.05)
    w_new = _w("STEP2_MCTS_TRAFFIC_WEIGHT_NEW", 0.5)
    w_overflow = _w("STEP2_MCTS_TRAFFIC_WEIGHT_OVERFLOW", 0.0)

    return (
        (w_success * float(succ))
        + (w_cov * float(cov))
        - (w_gap * float(gap_bits))
        - (w_len * float(len_err))
        - (w_new * float(failed))
        - (w_overflow * float(overflow))
    )


def _format_traffic_accept_score(score: Tuple[int, float, int, int, int, int]) -> str:
    failed, neg_cov, gap_bits, len_err, overflow, issue_cnt = score
    return (
        f"failed={failed} cov={-neg_cov:.3f} gap_bits={gap_bits} "
        f"len_abs_err_bits={len_err} overflow_bits={overflow} error_issues={issue_cnt}"
    )


def _traffic_accept_max_failed_increase() -> int:
    try:
        return max(0, int(os.getenv("STEP2_TRAFFIC_ACCEPT_MAX_FAILED_INCREASE", "0").strip()))
    except Exception:
        return 0


def _traffic_no_worse_than(
    candidate_score: Tuple[int, float, int, int, int, int],
    candidate_scalar: float,
    baseline_score: Tuple[int, float, int, int, int, int],
    baseline_scalar: float,
    *,
    max_failed_increase: int = 0,
) -> bool:
    if candidate_score[0] > baseline_score[0] + max(0, int(max_failed_increase)):
        return False
    return candidate_score <= baseline_score or candidate_scalar >= baseline_scalar - 1e-9


def _traffic_preserve_node_constraints_enabled() -> bool:
    raw = os.getenv("STEP2_TRAFFIC_PRESERVE_NODE_CONSTRAINTS", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _normalize_constraint_for_guard(constraint: Any) -> str:
    if isinstance(constraint, str):
        text = (
            constraint.replace("：", ":")
            .replace("≤", "<=")
            .replace("≥", ">=")
            .replace("\u00a0", " ")
        )
    else:
        text = json.dumps(constraint, ensure_ascii=True, sort_keys=True)
    return re.sub(r"\s+", "", str(text).strip().lower())


def _constraint_guard_kind(norm_constraint: str) -> str:
    text = str(norm_constraint or "").strip().lower()
    if text.startswith("enum:"):
        return "enum"
    if text.startswith("range:"):
        return "range"
    if text.startswith("formula:"):
        return "formula"
    if text.startswith(("value==", "value!=", "value<", "value>", "val(")):
        return "formula"
    return "unsupported"


def _constraint_guard_inventory(tree: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    inventory: Dict[str, Dict[str, Any]] = {}
    if not isinstance(tree, dict):
        return inventory
    for node in tree.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        nid = node.get("node_id")
        if nid is None:
            continue
        node_key = str(nid)
        name = str(node.get("name") or "")
        constraints = node.get("constraints")
        if not isinstance(constraints, list):
            continue
        slot = inventory.setdefault(node_key, {"label": f"{nid}:{name}", "constraints": {}})
        for constraint in constraints:
            norm = _normalize_constraint_for_guard(constraint)
            if not norm or norm in {"null", "\"\""}:
                continue
            slot["constraints"].setdefault(norm, constraint)
    return inventory


def _node_constraint_signature(tree: Dict[str, Any]) -> set[str]:
    """Fingerprint node-level constraints for tests/debugging."""
    sig: set[str] = set()
    for node_key, info in _constraint_guard_inventory(tree).items():
        label = info.get("label") or node_key
        for norm in (info.get("constraints") or {}).keys():
            sig.add(f"{label}:{norm}")
    return sig


def _constraint_guard_has_supported_refinement(lost_norm: str, added_norms_same_node: set[str]) -> bool:
    lost_kind = _constraint_guard_kind(lost_norm)
    if lost_kind == "unsupported":
        return True
    for added in added_norms_same_node:
        added_kind = _constraint_guard_kind(added)
        if added_kind == "formula":
            # Conditional formulas are the preferred way to repair an over-broad
            # "must be zero" or mode-dependent node constraint without erasing it.
            return True
        if added_kind in {"enum", "range"} and os.getenv(
            "STEP2_TRAFFIC_ALLOW_SAME_KIND_CONSTRAINT_REWRITE",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _tree_conditioned_variant_ids(tree: Dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for edge in tree.get("edges", []) or []:
        if not isinstance(edge, dict) or str(edge.get("rel") or "") != "condition_on":
            continue
        dst = _coerce_int_or_none(edge.get("dst"))
        if dst is not None:
            out.add(int(dst))
    return out


def _tree_children_by_parent(tree: Dict[str, Any]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = defaultdict(list)
    for node in tree.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        nid = _coerce_int_or_none(node.get("node_id"))
        pid = _coerce_int_or_none(node.get("parent_id"))
        if nid is None or pid is None:
            continue
        children[int(pid)].append(int(nid))
    return children


def _subtree_ids(root_id: int, children_by_parent: Dict[int, List[int]]) -> List[int]:
    out: List[int] = []
    seen: set[int] = {int(root_id)}
    stack = [int(root_id)]
    while stack:
        current = stack.pop()
        for child_id in children_by_parent.get(current, []) or []:
            if child_id in seen:
                continue
            seen.add(child_id)
            out.append(child_id)
            stack.append(child_id)
    return out


def _node_in_selector_body_common_scope(tree: Dict[str, Any], node_id: int) -> bool:
    """
    Return True for scalar fields in the common prefix of a selector body.

    Traffic may prove that an enum/range on such a shared scalar is too narrow
    for the observed backbone. The right repair is to remove the shared
    executable constraint and let document-obligation repair reattach it under
    guarded variants, not to force the observed packet into one branch's range.
    """

    lookup = _build_node_lookup(tree)
    node = lookup.get(int(node_id)) or lookup.get(str(int(node_id)))
    if not isinstance(node, dict):
        return False
    if str(node.get("node_type") or "").strip().lower() not in {"field", "selector", "type", "length"}:
        return False

    conditioned_variants = _tree_conditioned_variant_ids(tree)
    children_by_parent = _tree_children_by_parent(tree)
    current = node
    seen: set[int] = set()
    while isinstance(current, dict):
        current_id = _coerce_int_or_none(current.get("node_id"))
        if current_id is None or int(current_id) in seen:
            break
        seen.add(int(current_id))
        parent_id = _coerce_int_or_none(current.get("parent_id"))
        if parent_id is None:
            break
        parent = lookup.get(int(parent_id)) or lookup.get(str(int(parent_id)))
        if not isinstance(parent, dict):
            break

        variant_ids: List[int] = []
        for descendant_id in _subtree_ids(int(parent_id), children_by_parent):
            descendant = lookup.get(descendant_id) or lookup.get(str(descendant_id))
            if not isinstance(descendant, dict):
                continue
            if str(descendant.get("node_type") or "").strip().lower() != "variant":
                continue
            if conditioned_variants and int(descendant_id) not in conditioned_variants:
                continue
            variant_ids.append(int(descendant_id))
        if len(set(variant_ids)) >= 2:
            return True
        current = parent
    return False


def _traffic_constraint_guard_exemptions(
    *,
    baseline_tree: Dict[str, Any],
    baseline_report: Optional[Any],
) -> set[Tuple[str, str]]:
    """
    Constraints that traffic has directly disproven on selector-body common fields.

    This is deliberately narrow: only enum/range constraints that appear as the
    concrete failing constraint in the baseline traffic report are exempted.
    """

    exemptions: set[Tuple[str, str]] = set()
    if baseline_report is None:
        return exemptions
    failures = getattr(baseline_report, "traffic_failures", None) or []
    for failure in failures:
        node_id = _coerce_int_or_none(getattr(failure, "node_id", None))
        if node_id is None:
            continue
        kind = str(getattr(failure, "constraint_kind", "") or "").strip().lower()
        if kind not in {"enum", "range"}:
            continue
        raw_constraint = getattr(failure, "constraint_text", None)
        norm = _normalize_constraint_for_guard(raw_constraint)
        if not norm or _constraint_guard_kind(norm) not in {"enum", "range"}:
            continue
        if not _node_in_selector_body_common_scope(baseline_tree, int(node_id)):
            continue
        exemptions.add((str(int(node_id)), norm))
    return exemptions


def _traffic_constraint_guard_violations(
    *,
    baseline_tree: Dict[str, Any],
    candidate_tree: Dict[str, Any],
    baseline_report: Optional[Any] = None,
) -> List[str]:
    base = _constraint_guard_inventory(baseline_tree)
    cand = _constraint_guard_inventory(candidate_tree)
    traffic_exemptions = _traffic_constraint_guard_exemptions(
        baseline_tree=baseline_tree,
        baseline_report=baseline_report,
    )
    candidate_all = {
        (norm, binding_scope(candidate_tree, key))
        for key, info in cand.items()
        for norm in (info.get("constraints") or {}).keys()
    }
    violations: List[str] = layout_binding_losses(baseline_tree, candidate_tree)
    for node_key, base_info in sorted(base.items()):
        base_constraints = set((base_info.get("constraints") or {}).keys())
        cand_info = cand.get(node_key) or {"constraints": {}}
        cand_constraints = set((cand_info.get("constraints") or {}).keys())
        if base_constraints and not scope_preserved(baseline_tree, candidate_tree, node_key):
            violations.append(f"{node_key}:constraint_scope_changed")
        lost = base_constraints - cand_constraints
        added_same_node = cand_constraints - base_constraints
        for lost_norm in sorted(lost):
            if (str(node_key), lost_norm) in traffic_exemptions:
                continue
            if (lost_norm, binding_scope(baseline_tree, node_key)) in candidate_all:
                # Constraint was moved to another node instead of deleted.
                continue
            if _constraint_guard_has_supported_refinement(lost_norm, added_same_node):
                continue
            label = str(base_info.get("label") or node_key)
            violations.append(f"{label}:{lost_norm}")
    return violations


def _traffic_constraint_guard_allows(
    *,
    baseline_tree: Dict[str, Any],
    candidate_tree: Dict[str, Any],
    baseline_report: Optional[Any] = None,
    context: str,
    log: bool = True,
) -> bool:
    if not _traffic_preserve_node_constraints_enabled():
        return True
    violations = _traffic_constraint_guard_violations(
        baseline_tree=baseline_tree,
        candidate_tree=candidate_tree,
        baseline_report=baseline_report,
    )
    if not violations:
        return True
    if log:
        preview = "; ".join(sorted(violations)[:5])
        print(
            "[TrafficFix][ConstraintGuard] "
            f"reject context={context} lost_node_constraints={len(violations)} preview={preview}",
            flush=True,
        )
    return False


def _traffic_report_with_constraint_guard(
    report: ValidationReport,
    *,
    baseline_tree: Dict[str, Any],
    candidate_tree: Dict[str, Any],
    baseline_report: Optional[Any] = None,
    context: str,
) -> ValidationReport:
    if not _traffic_preserve_node_constraints_enabled():
        return report
    violations = _traffic_constraint_guard_violations(
        baseline_tree=baseline_tree,
        candidate_tree=candidate_tree,
        baseline_report=baseline_report,
    )
    if not violations:
        return report

    guarded = copy.deepcopy(report)
    preview = "; ".join(sorted(violations)[:5])
    message = (
        "Constraint preservation gate rejected erased bindings or changed execution scope "
        f"context={context} lost_node_constraints={len(violations)} preview={preview}"
    )
    guarded.ok = False
    guarded.errors = list(getattr(guarded, "errors", []) or []) + [message]
    issues = dict(getattr(guarded, "issues", {}) or {})
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
    issue_id = f"traffic_constraint_preservation_{digest}"
    issues[issue_id] = Issue(
        id=issue_id,
        type=IssueType.STRUCTURE,
        severity=Severity.ERROR,
        description=message,
        code="CONSTRAINT_PRESERVATION",
    )
    guarded.issues = issues
    return guarded


def _find_latest_fixed_tree() -> Tuple[Optional[Path], str]:
    """
    Locate the most recent fixed_protocol_tree_* file under data/modbus/outputs/stage_b/step2_results.
    Returns (path or None, source_description).
    """
    candidate_dirs = [
        DATA_DIR / "modbus" / "outputs" / "stage_b" / "step2_results",
        DATA_DIR / "modbus" / "step2_results",
    ]
    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        fixed_files = sorted(
            candidate_dir.glob("fixed_protocol_tree_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if fixed_files:
            return fixed_files[0], f"latest_step2_results@{candidate_dir}"
    return None, "fallback_final_cache"


def _load_tree(tree_path: Path) -> Dict[str, Any]:
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "protocol_tree" in payload:
        return payload["protocol_tree"]
    return payload


def _load_sections(sections_path: Path | None) -> List[dict]:
    def _try_load(path: Path) -> Optional[List[dict]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict) and "sections" in payload:
            sections = payload.get("sections") or []
            return list(sections) if isinstance(sections, list) else []
        if isinstance(payload, list):
            return list(payload)
        return None

    # Primary path
    if sections_path and sections_path.exists():
        parsed = _try_load(sections_path)
        if parsed is not None:
            return parsed
        logger.warning("Unrecognized sections format in %s (using fallback search)", sections_path)

    # Fallback candidates (align with parse_args defaults)
    candidates = [
        DATA_DIR / "modbus" / "outputs" / "stage_a" / "modbus_document_sections_subset.json",
        DATA_DIR / "modbus" / "sections" / "modbus_document_sections_subset.json",
    ]
    for cand in candidates:
        if cand.exists():
            parsed = _try_load(cand)
            if parsed is not None:
                logger.warning("Sections file not found/invalid at %s; using fallback %s", sections_path, cand)
                return parsed

    logger.error("Sections file not found or invalid: %s", sections_path)
    raise FileNotFoundError(f"Sections file not found or invalid: {sections_path}")


def _build_dfs_order_map(tree: Dict[str, Any]) -> Dict[str, int]:
    """Assign DFS visit order to every node for upstream-first prioritization."""
    nodes_map: Dict[str, Dict[str, Any]] = {}
    for node in tree.get("nodes", []):
        if not isinstance(node, dict):
            continue
        nid = node.get("node_id")
        if nid is None:
            continue
        nodes_map[str(nid)] = node

    order_map: Dict[str, int] = {}
    counter = [0]

    def _dfs(nid: str) -> None:
        if nid in order_map:
            return
        order_map[nid] = counter[0]
        counter[0] += 1
        node = nodes_map.get(nid)
        if not node:
            return
        for child_id in node.get("children_ids", []):
            _dfs(str(child_id))

    root_id = tree.get("root_node_id")
    root_key = str(root_id) if root_id is not None else None
    if root_key and root_key in nodes_map:
        _dfs(root_key)

    # Assign any disconnected nodes to keep ordering total.
    for leftover in nodes_map:
        if leftover not in order_map:
            _dfs(leftover)

    return order_map


def parse_args() -> argparse.Namespace:
    auto_tree, auto_source = _find_latest_fixed_tree()
    fallback_tree = STEP2_CACHE_DIR / "final_complete_protocol_tree.json"
    default_tree = auto_tree or fallback_tree

    traffic_candidates = [
        DATA_DIR / "modbus" / "inputs" / "traffic" / "traffic_modbus.txt",
        DATA_DIR / "modbus" / "traffic_modbus.txt",
        DATA_DIR / "modbus" / "traffic.txt",
    ]
    default_traffic = next((p for p in traffic_candidates if p.exists()), traffic_candidates[0])

    sections_candidates = [
        DATA_DIR / "modbus" / "outputs" / "stage_a" / "modbus_document_sections_subset.json",
        DATA_DIR / "modbus" / "sections" / "modbus_document_sections_subset.json",
    ]
    default_sections = next((p for p in sections_candidates if p.exists()), sections_candidates[0])

    parser = argparse.ArgumentParser(description="Traffic-aware MCTS fix loop")
    parser.add_argument(
        "--tree",
        type=Path,
        default=default_tree,
        help=(
            "Path to protocol tree JSON. "
            "Defaults to latest fixed_protocol_tree_*.json under data/modbus/outputs/stage_b/step2_results if present, "
            "otherwise falls back to data/_artifacts/step2_cache/final_complete_protocol_tree.json"
        ),
    )
    parser.add_argument(
        "--traffic",
        type=Path,
        default=default_traffic,
        help="Path to traffic hex dump (one frame per line, default: modbus traffic_modbus.txt)",
    )
    parser.add_argument(
        "--sections",
        type=Path,
        default=default_sections,
        help="Optional sections file for prompt context (default: modbus subset if present)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("AI_PROVIDER", "deepseek"),
        help="LLM provider (openai|deepseek|claude). Default: deepseek or $AI_PROVIDER if set",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for MCTS refinement (traffic fix prefers 1)")
    parser.add_argument("--max-llm-calls", type=int, default=20, help="Maximum LLM calls for the fix loop")
    parser.add_argument("--max-packets", type=int, default=0, help="Max traffic samples to evaluate (0 = no limit)")
    parser.add_argument(
        "--target-message-type",
        type=str,
        default=None,
        help="Optional message_type filter for traffic parsing (for example request or response)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STEP2_FIX_CACHE_DIR / "traffic_fixed_tree.json",
        help="Where to save the refined tree",
    )
    parser.add_argument(
        "--per-issue-mcts",
        dest="per_issue_mcts",
        action="store_true",
        default=True,
        help="Run a full MCTS pass per highest-priority issue (default: on for traffic fix)",
    )
    parser.add_argument(
        "--no-per-issue-mcts",
        dest="per_issue_mcts",
        action="store_false",
        help="Process all issues in one MCTS run (legacy behaviour)",
    )
    args = parser.parse_args()
    # Track how the default was chosen for logging.
    args._auto_tree_source = auto_source
    args._auto_tree_default = default_tree
    return args


def run_traffic_fix(
    tree_path: Path,
    traffic_path: Path,
    sections_path: Path | None,
    provider: str,
    output_path: Path,
    max_llm_calls: int = 20,
    max_packets: int = 0,
    target_message_type: Optional[str] = None,
    per_issue_mcts: bool = True,
    batch_size: int = 1,
) -> Path:
    """
    Run the traffic-aware semantic fixing stage.
    Returns the path to the fixed tree.
    """
    _setup_logging()
    tree = normalize_protocol_tree(_load_tree(tree_path))
    input_tree_snapshot = copy.deepcopy(tree)
    digest_before = hashlib.sha256(
        json.dumps(tree, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    sections = _load_sections(sections_path)
    from pargen.protocol_knowledge.validation.doc_constraint_coverage import augment_report_with_doc_constraint_coverage

    def with_doc_summary(report, candidate):
        # Supply the existing MCTS regression gate without turning all missing
        # document claims into new traffic repair errors.
        return augment_report_with_doc_constraint_coverage(
            report, candidate, sections, max_issues=0, max_structure_issues=0, focus_kind="constraint")
    logger.info("Loaded %s sections for traffic fix (source=%s)", len(sections), sections_path)
    if not sections:
        logger.warning("No sections available for traffic fix; proceeding with traffic-only context.")

    # B3 can run static fixes too; prompt_mode alone cannot separate it from B2.
    agent = EnhancedPureAIAgent(provider=provider, cache_dir=STEP2_FIX_CACHE_DIR / "traffic_stage")
    # Load full traffic first; sampling is handled separately to avoid "first N packets" bias.
    packets = _load_packets_from_path(traffic_path, max_packets=0)
    total_packets = len(packets)
    logger.info(
        "Starting traffic-focused refinement with traffic=%s (packets=%s target_message_type=%s)",
        traffic_path,
        total_packets,
        target_message_type or "<all>",
    )
    smoke_error = _smoke_test_interpreter(tree, packets)
    if smoke_error:
        logger.error("Aborting traffic fixing: %s", smoke_error)
        raise RuntimeError(smoke_error)
    shared_fix_history: List[Dict[str, Any]] = []

    # Build a diverse packet sample for traffic validation. `max_packets=0` means "no limit".
    sample_budget = total_packets if max_packets <= 0 else min(total_packets, max_packets)
    sample_payloads, sample_debug = _select_diverse_sample_payloads(
        tree=tree,
        packets=packets,
        sample_budget=sample_budget,
        target_message_type=target_message_type,
    )
    logger.info(
        "Traffic sample selection: strategy=%s selected=%s/%s groups=%s ok=%s failed=%s prefix_len=%s",
        sample_debug.get("strategy"),
        len(sample_payloads),
        total_packets,
        sample_debug.get("groups"),
        sample_debug.get("ok"),
        sample_debug.get("failed"),
        sample_debug.get("prefix_len"),
    )

    def _static_validator(candidate_tree: Dict[str, Any]) -> ValidationReport:
        """Static-only validation for syntax stage."""
        try:
            serialized = json.dumps(candidate_tree, ensure_ascii=False)
        except TypeError:
            serialized = json.dumps({"protocol_tree": candidate_tree}, ensure_ascii=False)
        report = with_doc_summary(validate_protocol_tree(serialized), candidate_tree)
        # Mark that traffic validation was intentionally skipped for this evaluation.
        extras = list(report.extras or [])
        extras.append("TRAFFIC_VALIDATION_SKIPPED")
        report.extras = extras
        return _traffic_report_with_constraint_guard(
            report, baseline_tree=input_tree_snapshot, candidate_tree=candidate_tree,
            context="traffic_initial_static_fix",
        )

    def _needs_static_fix(report: ValidationReport) -> bool:
        return any(iss.severity == Severity.ERROR for iss in report.issues.values())

    def _has_fatal_structure_error(report: ValidationReport) -> bool:
        for issue in report.issues.values():
            if issue.type == IssueType.STRUCTURE and issue.severity == Severity.ERROR:
                return True
        return False

    def _traffic_candidates(report: ValidationReport, current_tree: Dict[str, Any], payloads: Sequence[bytes]) -> Dict[int, Any]:
        try:
            raw = infer_size_bits_candidates_from_report(report, current_tree)
            lookup = _build_node_lookup(current_tree)
            root_id = current_tree.get("root_node_id")
            filtered: Dict[int, Any] = {}
            for nid, cands in raw.items():
                node = lookup.get(nid) or lookup.get(str(nid))
                if root_id is not None and (nid == root_id or str(nid) == str(root_id)):
                    continue
                if _is_aggregator_node(node, lookup):
                    continue
                filtered[nid] = cands
            # Augment with doc+traffic inferred expressions for failing nodes
            for failure in getattr(report, "traffic_failures", []) or []:
                nid = failure.node_id
                if _is_aggregator_node(lookup.get(nid) or lookup.get(str(nid)), lookup):
                    continue
                inferred = infer_size_formula_for_node(
                    current_tree,
                    nid,
                    sections,
                    payloads,
                    interpreter_factory=DynamicTreeInterpreter,
                )
                if inferred:
                    entry = SimpleNamespace(expression=inferred, controlling_field_id=None)
                    filtered.setdefault(nid, []).insert(0, entry)
            return filtered
        except Exception as exc:
            logger.debug("Failed to derive traffic length candidates: %s", exc, exc_info=True)
            return {}

    def _traffic_payload_fill_candidates(report: ValidationReport, current_tree: Dict[str, Any]) -> Dict[int, Any]:
        try:
            return infer_payload_fill_candidates_from_report(report, current_tree)
        except Exception as exc:
            logger.debug("Failed to derive traffic payload fill candidates: %s", exc, exc_info=True)
            return {}

    # 1) The static refinement loop runs in the static stage; an extra static search here is opt-in.
    static_report = _static_validator(tree)
    initial_static_fix = str(os.getenv("STEP2_TRAFFIC_INITIAL_STATIC_FIX", "0")).strip().lower() in {
        "1", "true", "yes", "on"}
    if initial_static_fix and _needs_static_fix(static_report):
        tree = mcts_fix_tree(
            agent,
            tree,
            retrieve_evidence_sections(tree, static_report, sections),
            raw_sections=None,
            batch_size=max(1, batch_size),
            max_llm_calls=max_llm_calls,
            validator_fn=_static_validator,
            prompt_mode="fix",
            fix_history=shared_fix_history,
        )
        static_report = _static_validator(tree)

    def _static_error_count(report: ValidationReport) -> int:
        return sum(1 for issue in report.issues.values() if issue.severity == Severity.ERROR)


    if _needs_static_fix(static_report):
        publish_refinement(
            output_path, tree, static_report, stage="traffic_initial_static_fix",
            extra_failures=["Static fixing still has errors; traffic validation was not run."],
            metadata={"input_tree": str(tree_path), "validation_scope": "static_only"},
        )

    logger.info("======= TRAFFIC SEMANTIC FIXING (sampled packets=%s/%s) =======", len(sample_payloads), total_packets)

    def packet_validator(candidate_tree: Dict[str, Any]) -> ValidationReport:
        """Validate a tree against the selected traffic sample (syntax + semantic)."""
        try:
            serialized = json.dumps(candidate_tree, ensure_ascii=False)
        except TypeError:
            serialized = json.dumps({"protocol_tree": candidate_tree}, ensure_ascii=False)

        static_report = with_doc_summary(validate_protocol_tree(serialized), candidate_tree)
        has_static_errors = any(iss.severity == Severity.ERROR for iss in static_report.issues.values())

        if _has_fatal_structure_error(static_report) or has_static_errors:
            try:
                extras = list(static_report.extras or [])
                extras.append("TRAFFIC_VALIDATION_SKIPPED_STATIC_ERRORS")
                static_report.extras = extras
                static_report.errors = list(static_report.errors) + [
                    "TRAFFIC_VALIDATION_SKIPPED_STATIC_ERRORS"
                ]
            except Exception:
                pass
            # Important: if we skip traffic evaluation, make that visible to the
            # accept/reward logic by marking all samples as failed.
            try:
                setattr(static_report, "traffic_total_samples", int(len(sample_payloads)))
                setattr(static_report, "traffic_successful_samples", 0)
                setattr(static_report, "traffic_coverage_gap_samples", int(len(sample_payloads)))
                setattr(static_report, "traffic_total_coverage_gap_bits", 0)
                setattr(static_report, "traffic_length_total_abs_error_bits", 0)
                setattr(static_report, "traffic_overflow_length_bits", 0)
                setattr(static_report, "traffic_repair_hints", [])
            except Exception:
                pass
            return static_report

        traffic_validator = SemanticValidator(
            candidate_tree,
            max_packets=len(sample_payloads),
            target_message_type=target_message_type,
            stop_on_first_failure=False,
            coverage_issue_grouping=os.getenv("STEP2_TRAFFIC_COVERAGE_ISSUE_GROUPING", "prefix"),
        )
        dynamic_issues, dynamic_extras, traffic_failures, traffic_stats = traffic_validator.validate_packets(
            list(sample_payloads)
        )

        merged_issues = {
            issue_id: issue
            for issue_id, issue in static_report.issues.items()
            if issue.severity == Severity.ERROR
        }
        blocking_dynamic = [
            ctx
            for ctx in dynamic_issues
            if getattr(ctx.issue, "severity", Severity.ERROR) == Severity.ERROR
        ]
        warning_dynamic = [
            ctx
            for ctx in dynamic_issues
            if getattr(ctx.issue, "severity", Severity.ERROR) != Severity.ERROR
        ]
        for ctx in blocking_dynamic:
            merged_issues[ctx.issue.id] = ctx.issue

        ok = static_report.ok and len(blocking_dynamic) == 0
        errors = list(static_report.errors)
        warnings = list(getattr(static_report, "warnings", []))
        if blocking_dynamic:
            errors.extend([ctx.message for ctx in blocking_dynamic])
        if warning_dynamic:
            warnings.extend([f"WARNING: {ctx.message}" for ctx in warning_dynamic])

        extras = list(static_report.extras) if static_report.extras else []
        extras.extend(dynamic_extras)
        extras.append(f"traffic_total_packets={total_packets}")
        extras.append(f"traffic_sample_packets={len(sample_payloads)}")
        extras.append(f"traffic_sample_strategy={sample_debug.get('strategy')}")
        extras.append(f"traffic_sample_groups={sample_debug.get('groups')}")
        if target_message_type:
            extras.append(f"traffic_target_message_type={target_message_type}")

        return with_doc_summary(ValidationReport(
            ok=ok,
            errors=errors,
            warnings=warnings,
            extras=extras,
            issues=merged_issues,
            traffic_failures=traffic_failures,
            traffic_global_summary=traffic_stats.get("summary"),
            traffic_repair_hints=list(traffic_stats.get("repair_hints", []) or []),
            traffic_successful_samples=traffic_stats.get("success_count", 0),
            traffic_total_samples=traffic_stats.get("processed", 0),
            traffic_max_bit_reached=traffic_stats.get("max_bits"),
            traffic_total_bits_per_sample=traffic_stats.get("total_bits"),
            traffic_length_total_abs_error_bits=int(
                traffic_stats.get("length_error_total_bits", 0) or 0
            ),
            traffic_wire_length_total_abs_error_bits=int(
                traffic_stats.get("wire_length_error_total_bits", 0) or 0
            ),
            traffic_total_coverage_gap_bits=int(
                traffic_stats.get("coverage_gap_bits", 0) or 0
            ),
            traffic_coverage_gap_samples=int(
                traffic_stats.get("coverage_gap_samples", 0) or 0
            ),
            traffic_overflow_length_bits=int(
                traffic_stats.get("overflow_length_bits", 0) or 0
            ),
            traffic_content_covered_bits_per_sample=list(
                traffic_stats.get("content_bits", []) or []
            ),
            traffic_content_coverage_ratio_per_sample=list(
                traffic_stats.get("content_ratio", []) or []
            ),
        ), candidate_tree)

    report = packet_validator(tree)
    try:
        baseline_traffic_ok = f"{int(getattr(report, 'traffic_successful_samples', 0) or 0)}/{int(getattr(report, 'traffic_total_samples', 0) or 0)}"
    except Exception:
        baseline_traffic_ok = "?"
    try:
        baseline_error_issues = sum(
            1 for iss in (getattr(report, "issues", {}) or {}).values() if getattr(iss, "severity", None) == Severity.ERROR
        )
    except Exception:
        baseline_error_issues = 0
    print(
        "[TrafficFix] "
        f"baseline sample_ok={bool(getattr(report, 'ok', False))} "
        f"traffic_ok={baseline_traffic_ok} "
        f"error_issues={baseline_error_issues} "
        f"digest={digest_before[:8]}",
        flush=True,
    )
    try:
        hints = list(getattr(report, "traffic_repair_hints", []) or [])
        kind_counts = Counter(str(h.get("kind", "")) for h in hints if isinstance(h, dict)) if hints else Counter()
        print(
            "[TrafficFix] "
            f"repair_hints={len(hints)} kinds_top={dict(kind_counts.most_common(6))}",
            flush=True,
        )
        try:
            preview_n = int(os.getenv("STEP2_TRAFFIC_HINTS_PREVIEW", "3").strip())
        except Exception:
            preview_n = 3
        if hints and preview_n > 0:
            preview: List[str] = []
            for hint in hints[:preview_n]:
                if not isinstance(hint, dict):
                    continue
                kind = str(hint.get("kind", "") or "")
                if kind == "add_length_of":
                    preview.append(
                        f"add_length_of src={hint.get('src_id')} dst={hint.get('dst_id')} formula={hint.get('formula')}"
                    )
                elif kind == "set_variant_size_bits":
                    preview.append(
                        f"set_variant_size_bits variant_id={hint.get('variant_id')} size_bits={hint.get('suggested_size_bits')}"
                    )
                elif kind == "shift_variant_subtree":
                    preview.append(
                        f"shift_variant_subtree variant_id={hint.get('variant_id')} shift_bits={hint.get('shift_bits')}"
                    )
                else:
                    preview.append(f"{kind} target={hint.get('target')}")
            if preview:
                print("[TrafficFix] hints_preview=" + " | ".join(preview), flush=True)
    except Exception:
        pass
    if not report.ok:
        logger.info("Traffic sample failed; invoking traffic-focused MCTS.")
        # Traffic fixing usually needs multi-step edits (e.g., shift + size_bits + length_of).
        # When per_issue_mcts is enabled (default), iterate until the traffic score stops improving.
        try:
            env_rounds = os.getenv("STEP2_TRAFFIC_MCTS_ROUNDS")
            max_rounds = int(env_rounds) if env_rounds is not None else (5 if per_issue_mcts else 1)  # five message rounds
        except Exception:
            max_rounds = 5 if per_issue_mcts else 1
        max_rounds = max(1, max_rounds)
        mcts_round_timeout_sec = _traffic_mcts_round_timeout_sec()
        baseline_score = _traffic_accept_score(report)
        baseline_scalar = _traffic_scalar_score(report)
        for round_idx in range(max_rounds):
            if report.ok:
                break
            try:
                print(
                    "[TrafficFix] "
                    f"mcts_round={round_idx + 1}/{max_rounds} "
                    f"baseline=({_format_traffic_accept_score(baseline_score)})",
                    flush=True,
                )
            except Exception:
                pass
            size_candidates_by_node = _traffic_candidates(report, tree, sample_payloads)
            payload_fill_candidates = _traffic_payload_fill_candidates(report, tree)

            def guarded_packet_validator(candidate_tree: Dict[str, Any]) -> ValidationReport:
                candidate_report_inner = packet_validator(candidate_tree)
                return _traffic_report_with_constraint_guard(
                    candidate_report_inner,
                    baseline_tree=tree,
                    candidate_tree=candidate_tree,
                    baseline_report=report,
                    context=f"mcts_round_{round_idx + 1}",
                )

            try:
                with _traffic_mcts_wall_timeout(
                    mcts_round_timeout_sec,
                    f"traffic MCTS round {round_idx + 1}",
                ):
                    candidate_tree = mcts_fix_tree(
                        agent,
                        tree,
                        retrieve_evidence_sections(tree, report, sections),
                        raw_sections=None,
                        batch_size=max(1, batch_size),
                        max_llm_calls=max(0, int(max_llm_calls)),
                        validator_fn=guarded_packet_validator,
                        prompt_mode="traffic_fix",
                        fix_history=shared_fix_history,
                        size_bits_candidates=size_candidates_by_node,
                        payload_fill_candidates=payload_fill_candidates,
                    )
                    candidate_report = guarded_packet_validator(candidate_tree)
            except (_TrafficMCTSDeadline, TimeoutError) as exc:
                # A slow provider round must not throw away the run: keep the
                # best tree so far and let the normal publish path label it.
                message = f"Traffic MCTS timed out during round {round_idx + 1}: {exc}; keeping the best tree so far"
                logger.error(message)
                print(f"[TrafficFix] mcts_round_timeout {message}", flush=True)
                break
            candidate_score = _traffic_accept_score(candidate_report)
            candidate_scalar = _traffic_scalar_score(candidate_report)
            improved = (candidate_score < baseline_score) or (candidate_scalar > baseline_scalar + 1e-9)
            max_failed_increase = _traffic_accept_max_failed_increase()
            within_failed_slack = candidate_score[0] <= (baseline_score[0] + max_failed_increase)
            improved = (
                improved
                and within_failed_slack
                and _traffic_constraint_guard_allows(
                    baseline_tree=tree,
                    candidate_tree=candidate_tree,
                    baseline_report=report,
                    context=f"mcts_round_{round_idx + 1}",
                )
            )
            try:
                print(
                    "[TrafficFix] "
                    f"mcts_round_result accept={improved} "
                    f"baseline=({_format_traffic_accept_score(baseline_score)}) "
                    f"candidate=({_format_traffic_accept_score(candidate_score)}) "
                    f"baseline_scalar={baseline_scalar:.3f} candidate_scalar={candidate_scalar:.3f}",
                    flush=True,
                )
            except Exception:
                pass
            if not improved:
                break
            tree = candidate_tree
            report = candidate_report
            baseline_score = candidate_score
            baseline_scalar = candidate_scalar

    # Optional syntax clean-up after traffic patches. Keep the traffic-accepted
    # baseline so syntax repairs cannot silently publish a worse parser.
    pre_syntax_tree = copy.deepcopy(tree)
    pre_syntax_report = copy.deepcopy(report)
    pre_syntax_score = _traffic_accept_score(report)
    pre_syntax_scalar = _traffic_scalar_score(report)

    def syntax_preservation_validator(candidate_tree):
        return _traffic_report_with_constraint_guard(
            run_full_validation(candidate_tree), baseline_tree=pre_syntax_tree,
            candidate_tree=candidate_tree, baseline_report=pre_syntax_report,
            context="traffic_final_syntax_cleanup",
        )

    syntax_cleanup_attempted = False
    syntax_rounds = 0
    max_syntax_rounds = 3
    while syntax_rounds < max_syntax_rounds:
        syntax_report = syntax_preservation_validator(tree)
        syntax_errors = [iss for iss in syntax_report.issues.values() if iss.severity == Severity.ERROR]
        if not syntax_errors:
            break
        syntax_cleanup_attempted = True
        logger.warning(
            "Syntax errors detected after traffic fix (round %s/%s); invoking syntax fix agent.",
            syntax_rounds + 1,
            max_syntax_rounds,
        )
        llm_budget = max(0, int(max_llm_calls))
        if llm_budget <= 0:
            logger.warning(
                "Skipping syntax fix agent because traffic LLM budget is 0 (set STEP2_TRAFFIC_MAX_LLM_CALLS to enable)."
            )
            break
        tree = mcts_fix_tree(
            agent,
            tree,
            sections,
            raw_sections=None,
            batch_size=max(1, batch_size),
            max_llm_calls=llm_budget,
            validator_fn=syntax_preservation_validator,
            prompt_mode="fix",
            fix_history=shared_fix_history,
        )
        # Re-check traffic after syntax fixes (syntax fixes may perturb parsing).
        post_report = packet_validator(tree)
        if not post_report.ok:
            logger.warning("Traffic sample failed again after syntax fixes; re-running traffic fix.")
            size_candidates_by_node = _traffic_candidates(post_report, tree, sample_payloads)
            payload_fill_candidates = _traffic_payload_fill_candidates(post_report, tree)
            try:
                with _traffic_mcts_wall_timeout(
                    _traffic_mcts_round_timeout_sec(),
                    f"traffic MCTS syntax-cleanup round {syntax_rounds + 1}",
                ):
                    tree = mcts_fix_tree(
                        agent,
                        tree,
                        sections,
                        raw_sections=None,
                        batch_size=max(1, batch_size),
                        max_llm_calls=llm_budget,
                        validator_fn=packet_validator,
                        prompt_mode="traffic_fix",
                        fix_history=shared_fix_history,
                        size_bits_candidates=size_candidates_by_node,
                        payload_fill_candidates=payload_fill_candidates,
                    )
            except (_TrafficMCTSDeadline, TimeoutError) as exc:
                message = f"Traffic MCTS timed out after syntax cleanup round {syntax_rounds + 1}: {exc}"
                logger.error(message)
                print(f"[TrafficFix] mcts_round_timeout {message}", flush=True)
                raise RuntimeError(message) from exc
        syntax_rounds += 1

    if syntax_cleanup_attempted:
        final_syntax_report = syntax_preservation_validator(tree)
        final_syntax_errors = [
            iss for iss in final_syntax_report.issues.values() if iss.severity == Severity.ERROR
        ]
        final_traffic_report = packet_validator(tree)
        final_score = _traffic_accept_score(final_traffic_report)
        final_scalar = _traffic_scalar_score(final_traffic_report)
        if final_syntax_errors or not _traffic_no_worse_than(
            final_score,
            final_scalar,
            pre_syntax_score,
            pre_syntax_scalar,
            max_failed_increase=_traffic_accept_max_failed_increase(),
        ):
            detail = (
                f"syntax_errors={len(final_syntax_errors)} "
                f"pre_syntax=({_format_traffic_accept_score(pre_syntax_score)}) "
                f"final=({_format_traffic_accept_score(final_score)})"
            )
            if os.getenv("STEP2_TRAFFIC_FINAL_GATE_ROLLBACK", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                logger.warning("Rolling back syntax cleanup after final gate failure: %s", detail)
                tree = pre_syntax_tree
                report = packet_validator(tree)
            else:
                publish_refinement(
                    output_path, tree, final_traffic_report, stage="traffic_final_syntax_cleanup",
                    extra_failures=["Final syntax/traffic non-regression gate failed: " + detail],
                    metadata={"input_tree": str(tree_path), "validation_scope": "sample"},
                )

    tree = normalize_protocol_tree(tree)
    digest_after = hashlib.sha256(
        json.dumps(tree, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    tree_changed = digest_after != digest_before
    # Compute final success rate on the loaded packet set. When max_packets is set,
    # this is intentionally the selected repair sample, not the full trace.
    final_eval_payloads: List[bytes] = list(packets) if max_packets <= 0 else list(sample_payloads)
    final_eval_scope = "full" if max_packets <= 0 else "sample"
    final_validator = SemanticValidator(
        tree,
        max_packets=len(final_eval_payloads),
        target_message_type=target_message_type,
        stop_on_first_failure=False,
        coverage_issue_grouping=os.getenv("STEP2_TRAFFIC_COVERAGE_ISSUE_GROUPING", "prefix"),
    )
    _final_issues, _final_extras, _final_failures, final_stats = final_validator.validate_packets(final_eval_payloads)
    ok_packets = int(final_stats.get("success_count", 0) or 0)
    processed = int(final_stats.get("processed", 0) or 0)
    # Validate AFTER final scaffold/normalization and BEFORE publishing. A
    # completed search or a non-regression alone does not mean repair passed.
    final_report = packet_validator(tree)
    publish_refinement(
        output_path, tree, final_report, stage="stage_b_traffic_fix", require_traffic=True,
        extra_failures=layout_binding_losses(input_tree_snapshot, tree) +
            _traffic_constraint_guard_violations(
                baseline_tree=pre_syntax_tree, candidate_tree=tree, baseline_report=pre_syntax_report,
            ) +
            ([f"Final traffic pass incomplete: {ok_packets}/{processed}"]
             if processed <= 0 or ok_packets != processed else []),
        metadata={"input_tree": str(tree_path), "validation_scope": final_eval_scope,
                  "input_sha256": hashlib.sha256(tree_path.read_bytes()).hexdigest(),
                  "selected_packets": len(final_eval_payloads)},
    )
    logger.info("Traffic-aware refinement complete. Saved to %s (traffic_ok=%s/%s)", output_path, ok_packets, processed)
    try:
        coverage_gap_samples = int(final_stats.get("coverage_gap_samples", 0) or 0)
    except Exception:
        coverage_gap_samples = 0
    try:
        coverage_gap_bits = int(final_stats.get("coverage_gap_bits", 0) or 0)
    except Exception:
        coverage_gap_bits = 0
    try:
        length_error_bits = int(final_stats.get("length_error_total_bits", 0) or 0)
    except Exception:
        length_error_bits = 0
    try:
        overflow_bits = int(final_stats.get("overflow_length_bits", 0) or 0)
    except Exception:
        overflow_bits = 0
    print(
        "[TrafficFix] "
        f"{final_eval_scope}_traffic_ok={ok_packets}/{processed} "
        f"coverage_gap_samples={coverage_gap_samples} "
        f"coverage_gap_bits={coverage_gap_bits} "
        f"length_error_bits={length_error_bits} "
        f"overflow_bits={overflow_bits}"
    )
    strict_cache = STEP2_FIX_CACHE_DIR / "strict_validator_loop.json"
    print(
        "[TrafficFix] "
        f"tree_changed={tree_changed} digest_before={digest_before[:8]} digest_after={digest_after[:8]} "
        f"strict_cache={strict_cache if strict_cache.exists() else '<none>'}",
        flush=True,
    )
    traces_path = output_path.with_name("traffic_parsing_traces.jsonl")
    try:
        logger.info("Exporting parsing traces to %s", traces_path)
        export_parsing_traces(
            tree=tree,
            traffic_path=traffic_path,
            output_path=traces_path,
            max_packets=max_packets,
            payloads=sample_payloads,
        )
        logger.info("Parsing traces exported to %s", traces_path)
        print(f"[TrafficFix] traces={traces_path}")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to export parsing traces to %s: %s", traces_path, exc, exc_info=True)
    return output_path


def main() -> None:
    args = parse_args()
    _setup_logging()
    source_label = "user-specified"
    if args.tree == getattr(args, "_auto_tree_default", None):
        source_label = getattr(args, "_auto_tree_source", "auto")
    logger.info("Loading protocol tree from %s (source=%s)", args.tree, source_label)
    run_traffic_fix(
        tree_path=args.tree,
        traffic_path=args.traffic,
        sections_path=args.sections,
        provider=args.provider,
        output_path=args.output,
        max_llm_calls=args.max_llm_calls,
        max_packets=args.max_packets,
        target_message_type=args.target_message_type,
        per_issue_mcts=args.per_issue_mcts,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
