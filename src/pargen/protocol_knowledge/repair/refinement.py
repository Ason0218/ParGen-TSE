"""Patch refinement workflow utilities for step2."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from .message import build_patch_refinement_message, summarize_sections_for_patch
from .mcts import MCTSConfig, SearchOutcome, ValidationSummary, log_mcts_event, search_for_batch
from ..constraint_ir import constraint_ir_to_text
from ..tree_utils import normalize_protocol_tree, synchronize_parent_child_links
from ..construction.core_ir import canonicalize_protocol_tree, prune_duplicate_scalar_leaf_children
from ..validation.syntax_validator import (
    validate_protocol_tree,
    ValidationReport,
    Issue,
    IssueType,
    Severity,
)

from ...paths import STEP2_FIX_CACHE_DIR

if TYPE_CHECKING:  # pragma: no cover
    from .agent import EnhancedPureAIAgent
    from .mcts import PatchEvaluation, TreeState


logger = logging.getLogger(__name__)


def _parent_child_sync_enabled() -> bool:
    raw = os.getenv("STEP2_PARENT_CHILD_SYNC", "1")
    return str(raw or "").strip().lower() not in {"0", "false", "no", "off"}


def _normalize_refinement_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_protocol_tree(tree)
    normalized = prune_duplicate_scalar_leaf_children(normalized)
    if _parent_child_sync_enabled():
        synchronize_parent_child_links(normalized)
    return normalized


def _unwrap_cached_patch(payload: Any) -> Optional[Dict[str, Any]]:
    patch = payload
    for _ in range(4):
        if not isinstance(patch, dict):
            return None
        if isinstance(patch.get("patch"), dict):
            patch = patch["patch"]
            continue
        if isinstance(patch.get("patch_snapshot"), dict):
            patch = patch["patch_snapshot"]
            continue
        return patch
    return patch if isinstance(patch, dict) else None


def _extract_cached_patch_entries(cached_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not isinstance(cached_data, dict):
        return entries

    cached_patches = cached_data.get("patches")
    if isinstance(cached_patches, list):
        for item in cached_patches:
            if isinstance(item, dict):
                patch = _unwrap_cached_patch(item)
                if isinstance(patch, dict):
                    entries.append(
                        {
                            "patch": patch,
                            "source": item.get("source", "cache"),
                            "summary": item.get("summary") or item.get("patch_summary"),
                            "messages": item.get("messages"),
                        }
                    )

    else:
        patch = _unwrap_cached_patch(cached_data)
        if not isinstance(patch, dict):
            return entries
        entries.append({
            "patch": patch,
            "source": cached_data.get("source", "cache"),
            "summary": cached_data.get("patch_summary"),
        })

    return entries


def _hash_patch(patch: Dict[str, Any]) -> str:
    serialized = json.dumps(patch, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _persist_batch_tree(batch_index: int, tree: Dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "batch_index": batch_index,
        "protocol_tree": tree,
    }
    filename = f"intermediate_tree_batch_{batch_index:03d}.json"
    path = STEP2_FIX_CACHE_DIR / filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to persist intermediate tree for batch %s: %s", batch_index + 1, exc)


def _validator_disabled() -> bool:
    return os.getenv("STEP2_DISABLE_VALIDATOR", "0") == "1"


def _strict_validator_loop_enabled() -> bool:
    """Read STEP2_STRICT_VALIDATOR_LOOP to enable validation-guided repair."""

    val = os.getenv("STEP2_STRICT_VALIDATOR_LOOP", "0")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _traffic_static_no_regression(
    *,
    baseline_struct: int,
    candidate_struct: int,
    baseline_sem: int,
    candidate_sem: int,
    baseline_invariant: int,
    candidate_invariant: int,
) -> bool:
    return (
        candidate_struct <= baseline_struct
        and candidate_sem <= baseline_sem
        and candidate_invariant <= baseline_invariant
    )


def _filter_error_issues(issues: Dict[str, Issue]) -> Dict[str, Issue]:
    """Restrict issues to ERROR severity for fixing/metrics."""
    if not issues:
        return {}
    return {
        issue_id: issue
        for issue_id, issue in issues.items()
        if getattr(issue, "severity", None) == Severity.ERROR
    }


def _patch_new_edges(patch: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return condition/length edges from both supported patch spellings.

    Step2 refine prompts historically used ``edge_adds`` while the fix-agent
    patch schema uses ``new_edges``. Treat them as aliases at the application
    boundary so valid refine patches do not silently lose selector routes.
    """

    if not isinstance(patch, dict):
        return []
    edges: List[Dict[str, Any]] = []
    for key in ("new_edges", "edge_adds"):
        value = patch.get(key)
        if not isinstance(value, list):
            continue
        edges.extend(edge for edge in value if isinstance(edge, dict))
    return edges


def _is_noop_patch(patch: Optional[Dict[str, Any]]) -> bool:
    if not patch:
        return True

    metadata = patch.get("patch_metadata")
    if isinstance(metadata, dict):
        intent = metadata.get("intent") or metadata.get("action")
        if isinstance(intent, str) and intent.strip().lower() in {"noop", "no-op", "skip"}:
            return True

    meaningful_keys = (
        "new_nodes",
        "node_updates",
        "new_edges",
        "edge_adds",
        "edge_updates",
        "edge_removes",
        "nodes_to_remove",
    )
    for key in meaningful_keys:
        value = patch.get(key)
        if isinstance(value, list) and value:
            return False
    # Treat dict updates specially: empty dict should not count as change
    if patch.get("protocol_tree"):
        return False
    return True


def _merge_sections_with_raw(
    sections: Sequence[Dict[str, Any]],
    raw_sections: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for index, section in enumerate(sections):
        base = copy.deepcopy(section)
        if raw_sections and index < len(raw_sections):
            raw_entry = raw_sections[index]
            if isinstance(raw_entry, dict):
                if "content" in raw_entry and "raw_content" not in base:
                    base["raw_content"] = raw_entry.get("content")
                if "summary" in raw_entry and "raw_summary" not in base:
                    base["raw_summary"] = raw_entry.get("summary")
                if "packet_formats" in raw_entry and "raw_packet_formats" not in base:
                    base["raw_packet_formats"] = raw_entry.get("packet_formats")
        merged.append(base)
    return merged


class BatchPatchSupplier:
    """Patch supplier that feeds the MCTS loop while tracking feedback and cache."""

    def __init__(
        self,
        agent: "EnhancedPureAIAgent",
        sections: Sequence[Dict[str, Any]],
        batch_index: int,
        batch_start: int,
        batch_size: int,
        *,
        initial_feedback: Optional[str],
        cached_entries: Optional[Sequence[Dict[str, Any]]],
        max_calls: int,
        cache_filename: str,
        prompt_mode: str = "fix",
        fix_history: Optional[List[Dict[str, Any]]] = None,
        size_bits_candidates: Optional[Dict[int, List[Any]]] = None,
        payload_fill_candidates: Optional[Dict[int, List[Any]]] = None,
    ) -> None:
        self.agent = agent
        self.sections = sections
        self.batch_index = batch_index
        self.batch_start = batch_start
        self.batch_size = batch_size
        self.batch_end = batch_start + batch_size
        self.feedback: Optional[str] = initial_feedback
        self.previous_patch_summary: Optional[str] = None
        self.cached_archive: List[Dict[str, Any]] = [copy.deepcopy(entry) for entry in (cached_entries or [])]
        # Cached patches are kept only for audit and analysis by default because
        # stale failures can pollute exploration diversity.  Replaying them is
        # useful for controlled reruns where the cache namespace is fixed.
        self.replay_cached_patches = os.getenv("STEP2_REPLAY_CACHED_PATCHES", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.remaining_cached: List[Dict[str, Any]] = (
            [copy.deepcopy(entry) for entry in self.cached_archive] if self.replay_cached_patches else []
        )
        self.max_calls = max_calls
        self.cache_filename = cache_filename
        self.calls_made = 0
        self._total_emitted = 0
        self._llm_first_toggle = False

        self.generated_log: List[Dict[str, Any]] = []
        self.action_info: Dict[str, Dict[str, Any]] = {}
        self.evaluation_records: List[Dict[str, Any]] = []
        self.patch_dir: Path = Path(getattr(agent, "cache_dir", STEP2_FIX_CACHE_DIR)) / ("traffic_fix" if prompt_mode == "traffic_fix" else "fix") / "patches" / f"batch_{batch_index:02d}"
        self.patch_dir.mkdir(parents=True, exist_ok=True)
        self.patch_files: Dict[str, Path] = {}
        if self.replay_cached_patches and not self.remaining_cached:
            self.remaining_cached.extend(self._load_patch_file_cache())
        # Queue for atomicized patches: when enabled, a large LLM patch can be
        # split into smaller patches to force multi-step search and better
        # demonstrate MCTS branching.
        self._atomic_queue: List[Dict[str, Any]] = []
        self.prompt_mode = (prompt_mode or "fix").strip().lower()
        self._traffic_debug_logged = False
        # Shared across MCTS runs to let LLM reuse prior strategies
        self.fix_history: List[Dict[str, Any]] = fix_history if fix_history is not None else []
        self._size_bits_candidates = size_bits_candidates or {}
        self._payload_fill_candidates = payload_fill_candidates or {}
        self._served_algorithmic: Set[Tuple[int, str]] = set()
        self._served_payload_fill: Set[Tuple[Any, Any, Any]] = set()
        self._served_repair_hints: Set[Tuple[Any, ...]] = set()
        self._next_new_node_id_base: Optional[int] = None

    def __call__(self, state: "TreeState", count: int, avoid_summaries: Optional[List[str]] = None) -> Sequence[Dict[str, Any]]:
        patches: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()
        if self.prompt_mode != "traffic_fix":
            while len(patches) < count and self.calls_made < self.max_calls:
                patch = self._produce_patch(state, avoid_summaries)
                if patch is None:
                    continue
                try:
                    h = _hash_patch(patch)
                except Exception:
                    h = None
                if h and h in seen_hashes:
                    logger.info(
                        "Batch %s suppressed duplicate patch in same iteration: %s",
                        self.batch_index + 1,
                        h[:8],
                    )
                    continue
                if h:
                    seen_hashes.add(h)
                patches.append(patch)
            return patches

        # traffic_fix: all patches should be LLM-generated by default. We still include
        # validator-derived candidates (repair_hints / size_bits_candidates / payload_fill_candidates)
        # in the prompt as guidance, but we do not auto-apply them unless explicitly opted in.
        llm_only = os.getenv("STEP2_TRAFFIC_LLM_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
        sources = ["llm"] if llm_only else ["algo", "llm"]
        if not self._traffic_debug_logged:
            self._traffic_debug_logged = True
            try:
                from collections import Counter

                hints = tuple(
                    getattr(getattr(state, "validation", None), "traffic_repair_hints", ()) or ()
                )
                failures = tuple(
                    getattr(getattr(state, "validation", None), "traffic_failures", ()) or ()
                )
                kind_counts = Counter(
                    str(h.get("kind", "")) for h in hints if isinstance(h, dict)
                )
                print(
                    "[TrafficFix][MCTS] "
                    f"patch_sources={sources} llm_only={llm_only} "
                    f"hints_in_state={len(hints)} failures_in_state={len(failures)} "
                    f"kinds_top={dict(kind_counts.most_common(6))}",
                    flush=True,
                )
            except Exception:
                pass
        max_attempts = max(10, int(count) * 6)
        attempts = 0
        # A failed request (a truncated or unparseable response) is not an answer.
        # Returning nothing makes MCTS mark the node terminal, which ended the whole
        # traffic search on the first truncation with the repair budget unspent.
        # Retry a bounded number of times; a truncation has already raised the
        # output budget for the next request.
        failure_retries = max(0, int(os.getenv("STEP2_TRAFFIC_PATCH_FAILURE_RETRIES", "2")))
        failures = 0
        while len(patches) < count and attempts < max_attempts:
            attempts += 1
            patch = self._request_patch(sources, state, avoid_summaries)
            if patch is None:
                failures += 1
                if failures > failure_retries or self.calls_made >= self.max_calls:
                    break
                logger.warning("Batch %s traffic patch request produced no patch; retrying (%s/%s)",
                               self.batch_index + 1, failures, failure_retries)
                continue
            try:
                h = _hash_patch(patch)
            except Exception:
                h = None
            if h and h in seen_hashes:
                logger.info(
                    "Batch %s suppressed duplicate patch in same iteration: %s",
                    self.batch_index + 1,
                    h[:8],
                )
                continue
            if h:
                seen_hashes.add(h)
            patches.append(patch)
        return patches

    def _next_attempt_number(self) -> int:
        return self._total_emitted + 1

    def _bump_counters(self, *, llm_used: bool) -> None:
        self._total_emitted += 1
        if llm_used:
            self.calls_made += 1

    @staticmethod
    def _patch_has_effective_actions(patch: Any) -> bool:
        if not isinstance(patch, dict):
            return False
        for key in ("new_nodes", "node_updates", "new_edges", "edge_updates", "edge_removes", "nodes_to_remove"):
            value = patch.get(key)
            if isinstance(value, list) and value:
                return True
        return False

    def _has_algorithmic_candidates(self) -> bool:
        if self.prompt_mode != "traffic_fix":
            return False
        if self._payload_fill_candidates:
            for raw_nid, cands in self._payload_fill_candidates.items():
                for cand in cands or []:
                    key = (
                        getattr(cand, "dst_id", None) or getattr(cand, "target_node_id", raw_nid),
                        getattr(cand, "bit_start_expr", None),
                        getattr(cand, "size_bits_expr", None),
                    )
                    if key not in self._served_payload_fill:
                        return True
        if not self._size_bits_candidates:
            return False
        def _coerce_node_id(raw: Any) -> Optional[int]:
            try:
                return int(raw)
            except Exception:
                return None
        for raw_nid, cands in self._size_bits_candidates.items():
            nid = _coerce_node_id(raw_nid)
            if nid is None:
                continue
            for cand in cands or []:
                key = (nid, getattr(cand, "expression", None))
                if key not in self._served_algorithmic:
                    return True
        return False

    def _request_patch(self, sources: List[str], state: "TreeState", avoid_summaries: Optional[List[str]]) -> Optional[Dict[str, Any]]:
        if self._atomic_queue:
            return self._atomic_queue.pop(0)
        for source in sources:
            patch: Optional[Dict[str, Any]] = None
            if source == "algo":
                patch = self._next_algorithmic_patch(state)
            elif source == "llm":
                if self.calls_made < self.max_calls:
                    patch = self._generate_patch_via_llm(state, avoid_summaries)
            elif source == "cache":
                patch = self._consume_cached_patch()
            if patch is not None:
                return patch
        return None

    def _compute_next_node_id(self, state: "TreeState") -> int:
        if self._next_new_node_id_base is None:
            max_id = 0
            try:
                for node in state.tree.get("nodes", []) or []:
                    try:
                        nid = int(node.get("node_id"))
                        if nid > max_id:
                            max_id = nid
                    except Exception:
                        continue
            except Exception:
                max_id = 0
            self._next_new_node_id_base = max_id + 1
        next_id = self._next_new_node_id_base
        self._next_new_node_id_base += 1
        return next_id

    def _produce_patch(self, state: "TreeState", avoid_summaries: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        # Serve any pending atomicized patches first
        if self._atomic_queue:
            return self._atomic_queue.pop(0)

        if self.replay_cached_patches and self.remaining_cached:
            return self._consume_cached_patch()

        if self.calls_made >= self.max_calls:
            return None

        algo_patch = self._next_algorithmic_patch(state)
        if algo_patch is not None:
            return algo_patch

        # Cached patches are intentionally ignored to avoid reintroducing
        # previously rejected actions into the search frontier.
        return self._generate_patch_via_llm(state, avoid_summaries)

    def _load_patch_file_cache(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for path in sorted(self.patch_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Failed to load cached patch file: %s", path, exc_info=True)
                continue
            patch = _unwrap_cached_patch(payload)
            if not isinstance(patch, dict) or _is_noop_patch(patch):
                continue
            entries.append(
                {
                    "patch": patch,
                    "source": "patch_file_cache",
                    "summary": _summarize_patch_for_prompt(patch),
                    "messages": payload.get("messages") if isinstance(payload, dict) else None,
                }
            )
        if entries:
            logger.info(
                "Batch %s loaded %d cached patch file(s) for replay from %s",
                self.batch_index + 1,
                len(entries),
                self.patch_dir,
            )
        return entries

    def _consume_cached_patch(self) -> Optional[Dict[str, Any]]:
        entry = self.remaining_cached.pop(0)
        patch = copy.deepcopy(entry.get("patch", {}))
        if not isinstance(patch, dict):
            logger.warning(
                "Cached patch for batch %s is not a dict, skipping",
                self.batch_index + 1,
            )
            self._bump_counters(llm_used=True)
            return None

        self._ensure_patch_defaults(patch)
        attempt_number = self._next_attempt_number()
        summary = entry.get("summary") or _summarize_patch_for_prompt(patch)
        patch_hash = _hash_patch(patch)
        source = entry.get("source", "cache")

        self._record_patch(patch, patch_hash, summary, source, attempt_number, entry.get("messages"))
        self.action_info[patch_hash] = {
            "summary": summary,
            "source": source,
            "attempt": attempt_number,
        }
        self.previous_patch_summary = summary
        self._bump_counters(llm_used=True)

        logger.info(
            "Batch %s using cached patch (attempt %s): hash=%s",
            self.batch_index + 1,
            attempt_number,
            patch_hash[:8],
        )
        return patch

    def _next_algorithmic_patch(self, state: "TreeState") -> Optional[Dict[str, Any]]:
        """Emit an algorithmic patch derived from traffic length inference."""
        if self.prompt_mode != "traffic_fix":
            return None

        hint_patch = self._next_repair_hint_patch(state)
        if hint_patch is not None:
            return hint_patch

        fill_patch = self._next_payload_fill_patch(state)
        if fill_patch is not None:
            return fill_patch

        if not self._size_bits_candidates:
            return None

        def _coerce_node_id(raw: Any) -> Optional[int]:
            try:
                return int(raw)
            except Exception:
                return None

        for issue in (state.pending_issues or {}).values():
            target = getattr(issue, "target", None)
            if not target or getattr(target, "kind", None) != "node":
                continue
            nid = _coerce_node_id(getattr(target, "identifier", None))
            if nid is None:
                continue
            candidates = self._size_bits_candidates.get(nid) or self._size_bits_candidates.get(str(nid)) or []
            for cand in candidates:
                key = (nid, getattr(cand, "expression", None))
                if key in self._served_algorithmic:
                    continue
                summary = f"Set node {nid} size_bits={cand.expression} (traffic-derived)"
                patch: Dict[str, Any] = {
                    "patch_metadata": {
                        "description": summary,
                        "intent": "traffic_length_fit",
                        "source": "traffic_inference",
                    },
                    "node_updates": [
                        {
                            "node_id": nid,
                            "property": "size_bits",
                            "value": cand.expression,
                        }
                    ],
                    "new_edges": [],
                    "edge_updates": [],
                    "edge_removes": [],
                    "new_nodes": [],
                    "nodes_to_remove": [],
                }
                if getattr(cand, "controlling_field_id", None) is not None:
                    patch["new_edges"].append(
                        {
                            "src": cand.controlling_field_id,
                            "dst": nid,
                            "rel": "length_of",
                            "formula": cand.expression,
                        }
                    )

                self._ensure_patch_defaults(patch)
                attempt_number = self._next_attempt_number()
                patch_hash = _hash_patch(patch)
                self._record_patch(patch, patch_hash, summary, "traffic_inference", attempt_number, None)
                self.action_info[patch_hash] = {
                    "summary": summary,
                    "source": "traffic_inference",
                    "attempt": attempt_number,
                }
                self.previous_patch_summary = summary
                self._bump_counters(llm_used=False)
                self._served_algorithmic.add(key)
                return patch

        # Fallback: if no issue-targeted candidate was served, try remaining inferred candidates.
        for raw_nid, candidates in self._size_bits_candidates.items():
            nid = _coerce_node_id(raw_nid)
            if nid is None:
                continue
            for cand in candidates:
                key = (nid, getattr(cand, "expression", None))
                if key in self._served_algorithmic:
                    continue
                summary = f"Set node {nid} size_bits={cand.expression} (traffic-derived)"
                patch: Dict[str, Any] = {
                    "patch_metadata": {
                        "description": summary,
                        "intent": "traffic_length_fit",
                        "source": "traffic_inference",
                    },
                    "node_updates": [
                        {
                            "node_id": nid,
                            "property": "size_bits",
                            "value": cand.expression,
                        }
                    ],
                    "new_edges": [],
                    "edge_updates": [],
                    "edge_removes": [],
                    "new_nodes": [],
                    "nodes_to_remove": [],
                }
                if getattr(cand, "controlling_field_id", None) is not None:
                    patch["new_edges"].append(
                        {
                            "src": cand.controlling_field_id,
                            "dst": nid,
                            "rel": "length_of",
                            "formula": cand.expression,
                        }
                    )
                self._ensure_patch_defaults(patch)
                attempt_number = self._next_attempt_number()
                patch_hash = _hash_patch(patch)
                self._record_patch(patch, patch_hash, summary, "traffic_inference", attempt_number, None)
                self.action_info[patch_hash] = {
                    "summary": summary,
                    "source": "traffic_inference",
                    "attempt": attempt_number,
                }
                self.previous_patch_summary = summary
                self._bump_counters(llm_used=False)
                self._served_algorithmic.add(key)
                return patch
        return None

    def _next_repair_hint_patch(self, state: "TreeState") -> Optional[Dict[str, Any]]:
        if self.prompt_mode != "traffic_fix":
            return None
        hints = list(getattr(getattr(state, "validation", None), "traffic_repair_hints", None) or [])
        if not hints:
            return None

        tree = state.tree
        nodes = tree.get("nodes", []) if isinstance(tree, dict) else []
        edges = tree.get("edges", []) if isinstance(tree, dict) else []
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []
        nodes_by_id: Dict[int, Dict[str, Any]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            try:
                nid = int(node.get("node_id"))
            except Exception:
                continue
            nodes_by_id[nid] = node

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except Exception:
                return None

        # If both `set_variant_size_bits` and `shift_variant_subtree` hints exist for the same
        # variant, apply them together as one patch. This avoids intermediate states where a
        # tightened `size_bits` makes existing descendants overflow the variant boundary.
        shift_hint_by_variant: Dict[int, int] = {}
        for raw_hint in hints:
            if not isinstance(raw_hint, dict):
                continue
            if str(raw_hint.get("kind", "") or "").strip() != "shift_variant_subtree":
                continue
            vid_hint = _coerce_int(raw_hint.get("variant_id"))
            shift_bits_hint = _coerce_int(raw_hint.get("shift_bits"))
            if vid_hint is None or shift_bits_hint is None or shift_bits_hint == 0:
                continue
            shift_hint_by_variant.setdefault(int(vid_hint), int(shift_bits_hint))

        def _edge_exists(src_id: int, dst_id: int, rel: str) -> Optional[Dict[str, Any]]:
            for edge in edges or []:
                if not isinstance(edge, dict):
                    continue
                if str(edge.get("rel") or "") != rel:
                    continue
                if _coerce_int(edge.get("src")) != src_id:
                    continue
                if _coerce_int(edge.get("dst")) != dst_id:
                    continue
                return edge
            return None

        def _descendants(root_id: int) -> List[int]:
            out: List[int] = []
            stack: List[int] = [root_id]
            seen: Set[int] = {root_id}
            while stack:
                cur = stack.pop()
                node = nodes_by_id.get(cur)
                if not node:
                    continue
                for cid in node.get("children_ids") or []:
                    child_id = _coerce_int(cid)
                    if child_id is None or child_id in seen:
                        continue
                    seen.add(child_id)
                    out.append(child_id)
                    stack.append(child_id)
            return out

        for hint in hints:
            if not isinstance(hint, dict):
                continue
            kind = str(hint.get("kind", "") or "").strip()

            if kind == "add_length_of":
                src_id = _coerce_int(hint.get("src_id"))
                dst_id = _coerce_int(hint.get("dst_id"))
                formula = hint.get("formula")
                if src_id is None or dst_id is None or not formula:
                    continue
                key = ("add_length_of", src_id, dst_id, str(formula))
                if key in self._served_repair_hints:
                    continue
                existing = _edge_exists(src_id, dst_id, "length_of")
                if existing is not None:
                    self._served_repair_hints.add(key)
                    continue
                src_node = nodes_by_id.get(src_id) or {}
                dst_node = nodes_by_id.get(dst_id) or {}
                msg_type = (
                    dst_node.get("message_type")
                    or src_node.get("message_type")
                    or "bidirectional"
                )
                summary = f"Add length_of edge {src_id}->{dst_id} formula={formula} (traffic hint)"
                patch: Dict[str, Any] = {
                    "patch_metadata": {
                        "description": summary,
                        "intent": "traffic_repair_hint",
                        "source": "traffic_repair_hint",
                    },
                    "new_edges": [
                        {
                            "src": src_id,
                            "dst": dst_id,
                            "rel": "length_of",
                            "formula": str(formula),
                            "message_type": msg_type,
                        }
                    ],
                    "node_updates": [],
                    "edge_updates": [],
                    "edge_removes": [],
                    "new_nodes": [],
                    "nodes_to_remove": [],
                }
                self._ensure_patch_defaults(patch)
                attempt_number = self._next_attempt_number()
                patch_hash = _hash_patch(patch)
                self._record_patch(patch, patch_hash, summary, "traffic_repair_hint", attempt_number, None)
                self.action_info[patch_hash] = {"summary": summary, "source": "traffic_repair_hint", "attempt": attempt_number}
                self.previous_patch_summary = summary
                self._served_repair_hints.add(key)
                self._bump_counters(llm_used=False)
                return patch

            if kind == "set_variant_size_bits":
                vid = _coerce_int(hint.get("variant_id"))
                suggested = hint.get("suggested_size_bits")
                if vid is None or not suggested:
                    continue
                key = ("set_variant_size_bits", vid, str(suggested))
                if key in self._served_repair_hints:
                    continue
                cur_node = nodes_by_id.get(vid) or {}
                if str(cur_node.get("size_bits") or "") == str(suggested):
                    self._served_repair_hints.add(key)
                    continue
                node_updates: List[Dict[str, Any]] = [
                    {"node_id": vid, "property": "size_bits", "value": str(suggested)}
                ]

                # Combine with an unserved subtree shift if available for this variant.
                shift_bits = shift_hint_by_variant.get(int(vid))
                key_shift: Optional[Tuple[Any, ...]] = None
                if shift_bits is not None:
                    key_shift = ("shift_variant_subtree", int(vid), int(shift_bits))
                    if key_shift in self._served_repair_hints:
                        key_shift = None
                if key_shift is not None:
                    for nid in _descendants(int(vid)):
                        node = nodes_by_id.get(nid) or {}
                        raw = node.get("bit_start")
                        if raw is None:
                            continue
                        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                            new_val = int(raw) + int(shift_bits)
                        elif isinstance(raw, str):
                            try:
                                new_val = int(raw.strip(), 0) + int(shift_bits)
                            except Exception:
                                continue
                        else:
                            continue
                        node_updates.append({"node_id": nid, "property": "bit_start", "value": new_val})

                if key_shift is not None and len(node_updates) <= 1:
                    # Shift hint exists but nothing was editable (likely expression bit_start);
                    # fall back to size_bits-only and keep the shift hint unserved.
                    key_shift = None
                    node_updates = [{"node_id": vid, "property": "size_bits", "value": str(suggested)}]

                if key_shift is not None:
                    summary = (
                        f"Shift variant {vid} descendants by {shift_bits} bits and set size_bits={suggested} "
                        "(traffic hint)"
                    )
                else:
                    summary = f"Set variant {vid} size_bits={suggested} (traffic hint)"

                patch = {
                    "patch_metadata": {
                        "description": summary,
                        "intent": "traffic_repair_hint",
                        "source": "traffic_repair_hint",
                    },
                    "node_updates": node_updates,
                    "new_edges": [],
                    "edge_updates": [],
                    "edge_removes": [],
                    "new_nodes": [],
                    "nodes_to_remove": [],
                }
                self._ensure_patch_defaults(patch)
                attempt_number = self._next_attempt_number()
                patch_hash = _hash_patch(patch)
                self._record_patch(patch, patch_hash, summary, "traffic_repair_hint", attempt_number, None)
                self.action_info[patch_hash] = {"summary": summary, "source": "traffic_repair_hint", "attempt": attempt_number}
                self.previous_patch_summary = summary
                self._served_repair_hints.add(key)
                if key_shift is not None:
                    self._served_repair_hints.add(key_shift)
                self._bump_counters(llm_used=False)
                return patch

            if kind == "shift_variant_subtree":
                vid = _coerce_int(hint.get("variant_id"))
                shift_bits = _coerce_int(hint.get("shift_bits"))
                if vid is None or shift_bits is None or shift_bits == 0:
                    continue
                key = ("shift_variant_subtree", vid, shift_bits)
                if key in self._served_repair_hints:
                    continue
                updates: List[Dict[str, Any]] = []
                for nid in _descendants(vid):
                    node = nodes_by_id.get(nid) or {}
                    raw = node.get("bit_start")
                    if raw is None:
                        continue
                    # Only shift simple numeric bit_start values; skip expressions.
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                        new_val = int(raw) + int(shift_bits)
                    elif isinstance(raw, str):
                        try:
                            new_val = int(raw.strip(), 0) + int(shift_bits)
                        except Exception:
                            continue
                    else:
                        continue
                    updates.append({"node_id": nid, "property": "bit_start", "value": new_val})
                if not updates:
                    self._served_repair_hints.add(key)
                    continue
                summary = f"Shift variant {vid} descendants by {shift_bits} bits (traffic hint)"
                patch = {
                    "patch_metadata": {
                        "description": summary,
                        "intent": "traffic_repair_hint",
                        "source": "traffic_repair_hint",
                    },
                    "node_updates": updates,
                    "new_edges": [],
                    "edge_updates": [],
                    "edge_removes": [],
                    "new_nodes": [],
                    "nodes_to_remove": [],
                }
                self._ensure_patch_defaults(patch)
                attempt_number = self._next_attempt_number()
                patch_hash = _hash_patch(patch)
                self._record_patch(patch, patch_hash, summary, "traffic_repair_hint", attempt_number, None)
                self.action_info[patch_hash] = {"summary": summary, "source": "traffic_repair_hint", "attempt": attempt_number}
                self.previous_patch_summary = summary
                self._served_repair_hints.add(key)
                self._bump_counters(llm_used=False)
                return patch

        return None

    def _next_payload_fill_patch(self, state: "TreeState") -> Optional[Dict[str, Any]]:
        """Emit a controlled payload fill patch for coverage gaps."""
        if not self._payload_fill_candidates or self.prompt_mode != "traffic_fix":
            return None

        for dst, candidates in self._payload_fill_candidates.items():
            for cand in candidates or []:
                key = (
                    getattr(cand, "dst_id", None) or getattr(cand, "target_node_id", dst),
                    getattr(cand, "parent_id", None),
                    getattr(cand, "bit_start_expr", None),
                    getattr(cand, "size_bits_expr", None),
                )
                if key in self._served_payload_fill:
                    continue
                try:
                    dst_id = int(getattr(cand, "dst_id", None) or getattr(cand, "target_node_id", dst) or dst)
                except Exception:
                    dst_id = dst
                try:
                    parent_id = int(getattr(cand, "parent_id", None) or dst_id)
                except Exception:
                    parent_id = getattr(cand, "parent_id", None) or dst_id
                new_node_id = self._compute_next_node_id(state)
                bit_start_expr = getattr(cand, "bit_start_expr", None) or ""
                size_bits_expr = getattr(cand, "size_bits_expr", None) or str(
                    getattr(cand, "gap_bits", "") or ""
                )
                summary = f"Add opaque payload under node {parent_id} (container {dst_id}) to fill coverage gap"
                patch: Dict[str, Any] = {
                    "patch_metadata": {
                        "description": summary,
                        "intent": "traffic_payload_fill",
                        "source": "traffic_payload_fill",
                    },
                    "new_nodes": [
                        {
                            "node_id": new_node_id,
                            "name": "opaque_payload",
                            "node_type": "field",
                            "data_type": "bytes",
                            "bit_start": bit_start_expr,
                            "size_bits": size_bits_expr,
                            "parent_id": parent_id,
                        }
                    ],
                    "node_updates": [],
                    "new_edges": [],
                    "edge_updates": [],
                    "edge_removes": [],
                    "nodes_to_remove": [],
                }
                self._ensure_patch_defaults(patch)
                attempt_number = self._next_attempt_number()
                patch_hash = _hash_patch(patch)
                self._record_patch(patch, patch_hash, summary, "traffic_payload_fill", attempt_number, None)
                self.action_info[patch_hash] = {
                    "summary": summary,
                    "source": "traffic_payload_fill",
                    "attempt": attempt_number,
                }
                self.previous_patch_summary = summary
                self._served_payload_fill.add(key)
                self._bump_counters(llm_used=False)
                return patch
        return None

    def _should_use_relation_compact_prompt(self, state: "TreeState") -> bool:
        raw = os.getenv("STEP2_RELATIONAL_COMPACT_PROMPT")
        if raw is not None:
            if raw.strip().lower() not in {"1", "true", "yes", "on"}:
                return False
        issues = getattr(state, "pending_issues", {}) or {}
        return any(self._is_detail_relation_issue_id(issue_id) for issue_id in issues)

    @staticmethod
    def _is_detail_relation_issue_id(issue_id: Any) -> bool:
        text = str(issue_id)
        return text.startswith("doc_relational_obligation_") and text != "doc_relational_obligation_total_uncovered"

    def _should_use_doc_constraint_compact_prompt(self, state: "TreeState") -> bool:
        raw = os.getenv("STEP2_DOC_CONSTRAINT_COMPACT_PROMPT")
        if raw is not None and raw.strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        issues = getattr(state, "pending_issues", {}) or {}
        return any(self._is_detail_doc_constraint_issue_id(issue_id) for issue_id in issues)

    @staticmethod
    def _is_detail_doc_constraint_issue_id(issue_id: Any) -> bool:
        text = str(issue_id)
        return text.startswith("doc_constraint_") and text != "doc_constraint_coverage_total_uncovered"

    @staticmethod
    def _doc_structure_compact_disabled_by_active_focus() -> bool:
        active = os.getenv("STEP2_STAGEB_COVERAGE_OBLIGATION_ACTIVE", "").strip().lower()
        if active not in {"1", "true", "yes", "on"}:
            return False
        focus = os.getenv("STEP2_DOC_COVERAGE_FOCUS_KIND", "").strip().lower()
        if focus in {
            "relational",
            "relation",
            "relations",
            "relation_constraint",
            "relation_constraints",
            "constraint_relation",
            "constraint_relations",
            "constraint",
            "constraints",
            "non_structure",
            "non_structural",
        }:
            return True
        structure_limit = os.getenv("STEP2_DOC_STRUCTURE_MAX_ISSUES")
        if structure_limit is not None:
            try:
                return int(str(structure_limit).strip() or "0") <= 0
            except ValueError:
                return False
        return False

    def _should_use_doc_structure_compact_prompt(self, state: "TreeState") -> bool:
        if self._doc_structure_compact_disabled_by_active_focus():
            return False
        raw = os.getenv("STEP2_DOC_STRUCTURE_COMPACT_PROMPT")
        if raw is not None and raw.strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        issues = getattr(state, "pending_issues", {}) or {}
        return any(self._is_detail_doc_structure_issue_id(issue_id) for issue_id in issues)

    @staticmethod
    def _is_detail_doc_structure_issue_id(issue_id: Any) -> bool:
        return str(issue_id).startswith("doc_structure_alignment_")

    @staticmethod
    def _doc_structure_selector_seed_ids(
        repair: Dict[str, Any],
        nodes_by_id: Dict[int, Dict[str, Any]],
    ) -> Set[int]:
        """Seed compact prompts for selector-table issues without a variant id.

        TLV abstraction issues often point to a documented selector name such as
        "Code" while the current tree uses a generic name such as "Tag".  The
        prompt still needs the selector neighborhood, so fall back to generic
        selector/control nodes instead of expanding the whole tree.
        """

        kind = str(repair.get("kind") or "")
        if kind == "missing_selector_variants":
            seeds: Set[int] = set()
            for value in repair.get("matched_selector_node_ids") or []:
                try:
                    seeds.add(int(value))
                except Exception:
                    continue
            selector_tokens = BatchPatchSupplier._relation_name_tokens(
                repair.get("selector_name") or repair.get("guard_field")
            )
            for nid, node in nodes_by_id.items():
                node_tokens = BatchPatchSupplier._relation_name_tokens(node.get("name"))
                if selector_tokens and selector_tokens <= node_tokens:
                    seeds.add(int(nid))
            return seeds

        if kind != "missing_tlv_abstraction":
            return set()
        profile = repair.get("tlv_profile") if isinstance(repair.get("tlv_profile"), dict) else {}
        guard_tokens = BatchPatchSupplier._relation_name_tokens(profile.get("guard_field"))
        selector_tokens = {
            "code",
            "function",
            "kind",
            "message",
            "opcode",
            "option",
            "selector",
            "tag",
            "type",
        }
        seeds: Set[int] = set()
        for nid, node in nodes_by_id.items():
            node_tokens = BatchPatchSupplier._relation_name_tokens(node.get("name"))
            node_type = str(node.get("node_type") or "").strip().lower()
            if guard_tokens and guard_tokens <= node_tokens:
                seeds.add(int(nid))
                continue
            if node_type == "selector" and (node_tokens & selector_tokens):
                seeds.add(int(nid))
        return seeds

    @staticmethod
    def _doc_constraint_issue_metadata(issue: Issue) -> Dict[str, Any]:
        target = getattr(issue, "target", None)
        metadata = getattr(target, "metadata", None) if target is not None else None
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _doc_structure_issue_metadata(issue: Issue) -> Dict[str, Any]:
        target = getattr(issue, "target", None)
        metadata = getattr(target, "metadata", None) if target is not None else None
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _doc_structure_safe_repair(metadata: Dict[str, Any]) -> Dict[str, Any]:
        repair = metadata.get("safe_repair")
        return repair if isinstance(repair, dict) else {}

    @staticmethod
    def _rank_doc_structure_issue_for_compact_prompt(item: Tuple[str, Issue]) -> Tuple[int, int, int, str]:
        issue_id, issue = item
        repair = BatchPatchSupplier._doc_structure_safe_repair(
            BatchPatchSupplier._doc_structure_issue_metadata(issue)
        )
        kind = str(repair.get("kind") or "")
        rank = {
            "missing_repeated_format": 0,
            "missing_selector_variants": 0,
            "missing_tlv_abstraction": 1,
            "opaque_variant_expansion": 2,
            "missing_guarded_variant": 3,
            "missing_scoped_field": 4,
            "weak_name_alignment": 5,
        }.get(kind, 6)
        missing_fields = [
            str(value)
            for value in (repair.get("missing_expected_fields") or repair.get("expected_body_fields") or [])
            if str(value or "").strip()
        ]
        expected_specs = [spec for spec in (repair.get("expected_field_specs") or []) if isinstance(spec, dict)]
        constrained_specs = sum(1 for spec in expected_specs if spec.get("direct_constraint") or spec.get("size_bits") is not None)
        # Within the same generic structure class, fix the most informative
        # local body first. This avoids spending calls on one-field umbrella
        # placeholders while richer source-backed fields remain opaque.
        return rank, -len(missing_fields), -constrained_specs, str(issue_id)

    @staticmethod
    def _doc_structure_issue_kind_for_compact_prompt(item: Tuple[str, Issue]) -> str:
        _issue_id, issue = item
        repair = BatchPatchSupplier._doc_structure_safe_repair(
            BatchPatchSupplier._doc_structure_issue_metadata(issue)
        )
        return str(repair.get("kind") or "")

    @staticmethod
    def _condition_formula_covers_selector_value(formula: Any, selector_id: int, value: Any) -> bool:
        try:
            target_value = int(str(value).strip(), 0)
        except Exception:
            return False
        text = str(formula or "")
        if not text:
            return False
        selector_ref = rf"val\(\s*{re.escape(str(selector_id))}\s*\)"
        number = r"(0x[0-9A-Fa-f]+|\d+)"
        patterns = [
            re.compile(selector_ref + rf"\s*==\s*{number}"),
            re.compile(rf"{number}\s*==\s*" + selector_ref),
        ]
        for pattern in patterns:
            for match in pattern.finditer(text):
                raw = match.group(1)
                try:
                    if int(raw, 0) == target_value:
                        return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _condition_variant_for_selector_values(
        *,
        edges: Sequence[Dict[str, Any]],
        selector_id: int,
        values: Sequence[Any],
        by_id: Dict[int, Dict[str, Any]],
    ) -> Optional[int]:
        for value in values:
            for edge in edges:
                if not isinstance(edge, dict) or edge.get("rel") != "condition_on":
                    continue
                if str(edge.get("src")) != str(selector_id):
                    continue
                if not BatchPatchSupplier._condition_formula_covers_selector_value(
                    edge.get("formula"), selector_id, value
                ):
                    continue
                try:
                    dst_id = int(edge.get("dst"))
                except Exception:
                    continue
                if dst_id in by_id:
                    return dst_id
        return None

    @staticmethod
    def _doc_constraint_repairs(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        repairs = metadata.get("safe_repairs")
        if not isinstance(repairs, list):
            return []
        return [repair for repair in repairs if isinstance(repair, dict) and repair.get("append_constraint")]

    @staticmethod
    def _doc_constraint_repair_node_id(repair: Dict[str, Any]) -> Optional[int]:
        key = "variant_id" if str(repair.get("kind") or "") in {"variant_formula", "guarded_variant_formula"} else "node_id"
        try:
            value = repair.get(key)
            return int(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _doc_constraint_patch_skeleton(
        *,
        issue_id: str,
        issue: Issue,
        by_id: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        metadata = BatchPatchSupplier._doc_constraint_issue_metadata(issue)
        repairs = BatchPatchSupplier._doc_constraint_repairs(metadata)
        skeleton: Dict[str, Any] = {
            "patch_metadata": {
                "description": f"Encode documentation constraint {issue_id}",
                "intent": "Doc constraint coverage",
                "focused_issue_id": issue_id,
            },
            "new_nodes": [],
            "nodes_to_remove": [],
            "node_updates": [],
            "new_edges": [],
            "edge_updates": [],
            "edge_removes": [],
            "validation_notes": [
                f"Section: {str(metadata.get('section') or '')[:160]}",
                f"Field: {str(metadata.get('field') or '')[:120]}",
                "Preserve all existing constraints on updated nodes.",
            ],
        }
        for repair in repairs[:3]:
            update_node_id = BatchPatchSupplier._doc_constraint_repair_node_id(repair)
            append_constraint = str(repair.get("append_constraint") or "").strip()
            if update_node_id is None or not append_constraint:
                continue
            node = by_id.get(update_node_id, {})
            existing_constraints = repair.get("preserve_existing_constraints")
            if not isinstance(existing_constraints, list):
                existing_constraints = node.get("constraints") if isinstance(node, dict) else []
            if not isinstance(existing_constraints, list):
                existing_constraints = []
            value = list(existing_constraints)
            if append_constraint not in {str(item) for item in value}:
                value.append(append_constraint)
            skeleton["node_updates"].append(
                {
                    "node_id": update_node_id,
                    "property": "constraints",
                    "value": value,
                    "reason": (
                        f"Apply {repair.get('kind')} safe repair for documented "
                        f"{metadata.get('field') or 'field'} constraint."
                    ),
                }
            )
        return skeleton

    @staticmethod
    def _doc_constraint_group_patch_skeleton(
        doc_issues: Dict[str, Issue],
        by_id: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        skeleton: Dict[str, Any] = {
            "patch_metadata": {
                "description": "Encode grouped documentation constraints",
                "intent": "Doc constraint coverage",
                "focused_issue_ids": sorted(str(issue_id) for issue_id in doc_issues),
            },
            "new_nodes": [],
            "nodes_to_remove": [],
            "node_updates": [],
            "new_edges": [],
            "edge_updates": [],
            "edge_removes": [],
            "validation_notes": [
                "Grouped safe repairs share the same scoped target node.",
                "Preserve all existing constraints on updated nodes.",
            ],
        }
        updates_by_node: Dict[int, Dict[str, Any]] = {}
        reasons_by_node: Dict[int, List[str]] = {}
        for issue_id, issue in doc_issues.items():
            metadata = BatchPatchSupplier._doc_constraint_issue_metadata(issue)
            for repair in BatchPatchSupplier._doc_constraint_repairs(metadata)[:1]:
                update_node_id = BatchPatchSupplier._doc_constraint_repair_node_id(repair)
                append_constraint = str(repair.get("append_constraint") or "").strip()
                if update_node_id is None or not append_constraint:
                    continue
                node = by_id.get(update_node_id, {})
                existing_constraints = repair.get("preserve_existing_constraints")
                if not isinstance(existing_constraints, list):
                    existing_constraints = node.get("constraints") if isinstance(node, dict) else []
                if not isinstance(existing_constraints, list):
                    existing_constraints = []
                update = updates_by_node.setdefault(
                    update_node_id,
                    {
                        "node_id": update_node_id,
                        "property": "constraints",
                        "value": list(existing_constraints),
                        "reason": "",
                    },
                )
                current = {str(item) for item in update["value"]}
                if append_constraint not in current:
                    update["value"].append(append_constraint)
                reasons_by_node.setdefault(update_node_id, []).append(
                    f"{issue_id}: {repair.get('kind')} for {metadata.get('field') or 'field'}"
                )
        for node_id, update in updates_by_node.items():
            update["reason"] = "; ".join(reasons_by_node.get(node_id, [])[:6])
            skeleton["node_updates"].append(update)
        return skeleton

    @staticmethod
    def _doc_structure_repair_variant_id(repair: Dict[str, Any]) -> Optional[int]:
        try:
            value = repair.get("variant_id")
            return int(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _doc_structure_is_length_node(node: Dict[str, Any]) -> bool:
        tokens = BatchPatchSupplier._relation_name_tokens(node.get("name"))
        return bool(tokens & {"len", "length"})

    @staticmethod
    def _doc_structure_is_payload_node(node: Dict[str, Any]) -> bool:
        tokens = BatchPatchSupplier._relation_name_tokens(node.get("name"))
        node_type = str(node.get("node_type") or "").strip().lower()
        return node_type in {"payload", "bytes"} or bool(
            tokens
            & {
                "address",
                "addresses",
                "body",
                "content",
                "data",
                "payload",
                "route",
                "routes",
                "value",
                "values",
            }
        )

    @staticmethod
    def _doc_structure_tlv_child_ids(
        variant_id: int,
        by_id: Dict[int, Dict[str, Any]],
    ) -> Tuple[Optional[int], Optional[int]]:
        variant = by_id.get(int(variant_id), {})
        length_id: Optional[int] = None
        payload_id: Optional[int] = None
        for child in variant.get("children_ids") or []:
            try:
                child_id = int(child)
            except Exception:
                continue
            node = by_id.get(child_id, {})
            if length_id is None and BatchPatchSupplier._doc_structure_is_length_node(node):
                length_id = child_id
            if payload_id is None and BatchPatchSupplier._doc_structure_is_payload_node(node):
                payload_id = child_id
        return length_id, payload_id

    @staticmethod
    def _doc_structure_looks_like_length_delimited_parent(node: Dict[str, Any]) -> bool:
        node_type = str(node.get("node_type") or "").strip().lower()
        tokens = BatchPatchSupplier._relation_name_tokens(node.get("name"))
        item_tokens = {
            "attribute",
            "element",
            "entry",
            "ie",
            "item",
            "option",
            "options",
            "parameter",
            "record",
            "rr",
            "tlv",
        }
        return node_type in {"option", "record", "tlv", "tlv_seq"} or bool(tokens & item_tokens)

    @staticmethod
    def _doc_structure_length_scoped(
        *,
        length_id: int,
        parent_id: int,
        following_ids: Sequence[int],
        by_id: Dict[int, Dict[str, Any]],
        edges: Sequence[Dict[str, Any]],
    ) -> bool:
        scoped_targets = {str(parent_id), *(str(node_id) for node_id in following_ids)}
        for edge in edges:
            if not isinstance(edge, dict) or str(edge.get("rel") or "").strip().lower() != "length_of":
                continue
            if str(edge.get("src")) == str(length_id) and str(edge.get("dst")) in scoped_targets:
                return True
        val_pattern = re.compile(rf"val\(\s*{re.escape(str(length_id))}\s*\)")
        for node_id in [parent_id, *following_ids]:
            node = by_id.get(int(node_id), {})
            exprs: List[Any] = [node.get("size_bits"), node.get("bit_start")]
            exprs.extend(node.get("constraints") or [])
            if any(val_pattern.search(str(expr or "")) for expr in exprs):
                return True
        return False

    @staticmethod
    def _doc_structure_add_length_scope_constraint(
        *,
        skeleton: Dict[str, Any],
        parent_id: int,
        selector_id: int,
        by_id: Dict[int, Dict[str, Any]],
        edges: Sequence[Dict[str, Any]],
        reason: str,
    ) -> None:
        parent = by_id.get(int(parent_id), {})
        if not BatchPatchSupplier._doc_structure_looks_like_length_delimited_parent(parent):
            return
        children: List[int] = []
        for raw_child_id in parent.get("children_ids") or []:
            try:
                child_id = int(raw_child_id)
            except Exception:
                continue
            if child_id in by_id:
                children.append(child_id)
        if len(children) < 2 or selector_id not in children:
            return
        selector_pos = children.index(selector_id)
        length_id: Optional[int] = None
        for child_id in children[selector_pos + 1 :]:
            if BatchPatchSupplier._doc_structure_is_length_node(by_id.get(child_id, {})):
                length_id = child_id
                break
        if length_id is None:
            for child_id in children:
                if BatchPatchSupplier._doc_structure_is_length_node(by_id.get(child_id, {})):
                    length_id = child_id
                    break
        if length_id is None:
            return
        try:
            length_pos = children.index(length_id)
        except ValueError:
            return
        following = [child_id for child_id in children[length_pos + 1 :] if child_id in by_id]
        if not following:
            return
        if BatchPatchSupplier._doc_structure_length_scoped(
            length_id=length_id,
            parent_id=parent_id,
            following_ids=following,
            by_id=by_id,
            edges=edges,
        ):
            return
        body_candidates = [
            child_id
            for child_id in following
            if BatchPatchSupplier._doc_structure_is_payload_node(by_id.get(child_id, {}))
        ]
        anchor_id = body_candidates[0] if body_candidates else parent_id
        length_node = by_id.get(length_id, {})
        anchor_node = by_id.get(anchor_id, {})
        parent_name = str(parent.get("name") or f"node_{parent_id}")
        length_name = str(length_node.get("name") or f"node_{length_id}")
        anchor_name = str(anchor_node.get("name") or f"node_{anchor_id}")
        constraint = constraint_ir_to_text(
            kind="length_relation",
            expr=f"val({length_id}) scopes {parent_name} body",
            field_ids=[length_id, anchor_id],
            field_names=[length_name, anchor_name],
            source=reason,
        )

        for update in skeleton.get("node_updates") or []:
            if update.get("node_id") == parent_id and update.get("property") == "constraints":
                values = update.get("value")
                if not isinstance(values, list):
                    values = []
                    update["value"] = values
                if constraint not in {str(item) for item in values}:
                    values.append(constraint)
                    update["reason"] = "; ".join(
                        part
                        for part in [str(update.get("reason") or "").strip(), f"{reason}: bind Length to body scope before selector shell expansion"]
                        if part
                    )
                return

        existing_constraints = parent.get("constraints") if isinstance(parent.get("constraints"), list) else []
        values = list(existing_constraints)
        if constraint not in {str(item) for item in values}:
            values.append(constraint)
        skeleton["node_updates"].append(
            {
                "node_id": parent_id,
                "property": "constraints",
                "value": values,
                "reason": f"{reason}: bind Length to body scope before selector shell expansion",
            }
        )
        skeleton["validation_notes"].append(
            f"{reason}: added relation-only length scope on {parent_name}(ID:{parent_id}) using {length_name}(ID:{length_id}); no parser length formula was guessed."
        )

    @staticmethod
    def _doc_structure_formula_excludes_selector_value(formula: Any, selector_id: int, value: Any) -> bool:
        try:
            target_value = int(str(value).strip(), 0)
        except Exception:
            return False
        text = str(formula or "")
        selector_ref = rf"val\(\s*{re.escape(str(selector_id))}\s*\)"
        number = r"(0x[0-9A-Fa-f]+|\d+)"
        patterns = [
            re.compile(selector_ref + rf"\s*!=\s*{number}"),
            re.compile(rf"{number}\s*!=\s*" + selector_ref),
        ]
        for pattern in patterns:
            for match in pattern.finditer(text):
                raw = match.group(1)
                try:
                    if int(raw, 0) == target_value:
                        return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _doc_structure_add_selector_fallback_exclusions(
        *,
        skeleton: Dict[str, Any],
        selector_id: int,
        parent_id: int,
        exact_values: Sequence[Any],
        by_id: Dict[int, Dict[str, Any]],
        edges: Sequence[Dict[str, Any]],
        reason: str,
    ) -> None:
        values: List[int] = []
        for value in exact_values or []:
            try:
                parsed = int(str(value).strip(), 0)
            except Exception:
                continue
            if parsed not in values:
                values.append(parsed)
        if not values:
            return
        for edge in edges:
            if not isinstance(edge, dict) or edge.get("rel") != "condition_on":
                continue
            if str(edge.get("src")) != str(selector_id):
                continue
            try:
                dst_id = int(edge.get("dst"))
            except Exception:
                continue
            dst_node = by_id.get(dst_id)
            if not dst_node or str(dst_node.get("parent_id")) != str(parent_id):
                continue
            formula = str(edge.get("formula") or "").strip()
            if not formula:
                continue
            if any(
                BatchPatchSupplier._condition_formula_covers_selector_value(formula, selector_id, value)
                for value in values
            ):
                continue
            new_formula = formula
            added: List[int] = []
            for value in values:
                if BatchPatchSupplier._doc_structure_formula_excludes_selector_value(new_formula, selector_id, value):
                    continue
                new_formula = f"({new_formula}) and val({selector_id}) != {value}"
                added.append(value)
            if not added or new_formula == formula:
                continue
            edge_identifier = {
                "src": selector_id,
                "dst": dst_id,
                "rel": "condition_on",
                "formula": formula,
            }
            for update in skeleton.get("edge_updates") or []:
                ident = update.get("edge_identifier") if isinstance(update, dict) else None
                if not isinstance(ident, dict):
                    continue
                if (
                    str(ident.get("src")) == str(selector_id)
                    and str(ident.get("dst")) == str(dst_id)
                    and ident.get("rel") == "condition_on"
                    and ident.get("formula") == formula
                ):
                    update.setdefault("new_properties", {})["formula"] = new_formula
                    break
            else:
                skeleton["edge_updates"].append(
                    {
                        "edge_identifier": edge_identifier,
                        "update_type": "modify",
                        "new_properties": {"formula": new_formula},
                        "reason": (
                            f"{reason}: keep generic selector fallback mutually exclusive with exact "
                            f"selector-coded variant value(s) {added}."
                        ),
                    }
                )
            skeleton["validation_notes"].append(
                f"{reason}: refined fallback condition on {dst_node.get('name')}(ID:{dst_id}) to exclude selector value(s) {added} before adding exact variants."
            )

    @staticmethod
    def _doc_structure_selector_ids_for_repair(
        repair: Dict[str, Any],
        by_id: Dict[int, Dict[str, Any]],
    ) -> Set[int]:
        metadata_seed_ids = BatchPatchSupplier._doc_structure_selector_seed_ids(repair, by_id)
        selector_ids = {
            nid
            for nid in metadata_seed_ids
            if str((by_id.get(nid) or {}).get("node_type") or "").strip().lower() == "selector"
        }
        if selector_ids:
            return selector_ids
        return {
            nid
            for nid, node in by_id.items()
            if str(node.get("node_type") or "").strip().lower() == "selector"
        }

    @staticmethod
    def _doc_structure_excluded_selector_values(formula: Any, selector_id: int) -> Set[int]:
        text = str(formula or "")
        escaped = re.escape(str(selector_id))
        values: Set[int] = set()
        patterns = [
            rf"val\(\s*{escaped}\s*\)\s*!=\s*(0x[0-9a-fA-F]+|\d+)",
            rf"(0x[0-9a-fA-F]+|\d+)\s*!=\s*val\(\s*{escaped}\s*\)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                try:
                    values.add(int(match.group(1), 0))
                except Exception:
                    continue
        return values

    @staticmethod
    def _doc_structure_relaxed_tlv_formula(
        *,
        existing_formula: Any,
        selector_id: int,
        length_values: Set[int],
        zero_body_values: Set[int],
    ) -> str:
        excluded = BatchPatchSupplier._doc_structure_excluded_selector_values(existing_formula, selector_id)
        if excluded:
            excluded = excluded - set(length_values)
        else:
            excluded = set(zero_body_values)
        if not excluded and zero_body_values:
            excluded = set(zero_body_values)
        selector = f"val({selector_id})"
        if excluded:
            return " and ".join(f"{selector} != {value}" for value in sorted(excluded))
        return f"{selector} >= 0"

    @staticmethod
    def _doc_structure_tlv_reuse_skeleton(
        *,
        issue_id: str,
        repair: Dict[str, Any],
        by_id: Dict[int, Dict[str, Any]],
        edges: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if str(repair.get("kind") or "") != "missing_tlv_abstraction":
            return None
        profile = repair.get("tlv_profile") if isinstance(repair.get("tlv_profile"), dict) else {}
        length_values: Set[int] = set()
        zero_body_values: Set[int] = set()
        BatchPatchSupplier._collect_numeric_values(profile.get("length_values"), length_values)
        BatchPatchSupplier._collect_numeric_values(profile.get("zero_body_values"), zero_body_values)
        selector_ids = BatchPatchSupplier._doc_structure_selector_ids_for_repair(repair, by_id)

        candidates: List[Tuple[int, int, Dict[str, Any], Optional[int], Optional[int]]] = []
        for edge in edges:
            if str(edge.get("rel") or "") != "condition_on":
                continue
            try:
                src = int(edge.get("src"))
                dst = int(edge.get("dst"))
            except Exception:
                continue
            if selector_ids and src not in selector_ids:
                continue
            variant = by_id.get(dst)
            if not isinstance(variant, dict):
                continue
            length_id, payload_id = BatchPatchSupplier._doc_structure_tlv_child_ids(dst, by_id)
            if length_id is None or payload_id is None:
                continue
            name_tokens = BatchPatchSupplier._relation_name_tokens(variant.get("name"))
            name_score = 10 if name_tokens & {"generic", "normal", "option", "other", "tlv"} else 0
            excluded = BatchPatchSupplier._doc_structure_excluded_selector_values(edge.get("formula"), src)
            focus_overlap = len(excluded & length_values)
            candidates.append((name_score + focus_overlap, dst, edge, length_id, payload_id))

        if not candidates:
            return None
        _score, variant_id, edge, length_id, payload_id = sorted(candidates, reverse=True)[0]
        selector_id = int(edge.get("src"))
        old_formula = edge.get("formula")
        new_formula = BatchPatchSupplier._doc_structure_relaxed_tlv_formula(
            existing_formula=old_formula,
            selector_id=selector_id,
            length_values=length_values,
            zero_body_values=zero_body_values,
        )
        variant = by_id.get(variant_id, {})
        length_node = by_id.get(length_id or -1, {})
        payload_node = by_id.get(payload_id or -1, {})
        return {
            "patch_metadata": {
                "description": f"Reuse existing generic TLV branch for {issue_id}",
                "intent": "Doc structure alignment",
                "focused_issue_id": issue_id,
            },
            "new_nodes": [],
            "nodes_to_remove": [],
            "node_updates": [],
            "new_edges": [],
            "edge_updates": [
                {
                    "edge_identifier": {
                        "src": selector_id,
                        "dst": variant_id,
                        "rel": "condition_on",
                        "formula": old_formula,
                    },
                    "new_properties": {
                        "formula": new_formula,
                    },
                    "reason": (
                        f"{issue_id}: reuse existing generic TLV variant "
                        f"{variant.get('name')}(ID:{variant_id}) for documented length-bearing values "
                        f"{sorted(length_values)} instead of creating a duplicate TLV_Option branch."
                    ),
                }
            ],
            "edge_removes": [],
            "validation_notes": [
                f"Reused existing TLV variant {variant.get('name')}(ID:{variant_id}).",
                f"Existing length node: {length_node.get('name')}(ID:{length_id}); preserve its current constraints unless documentation explicitly requires a change.",
                f"Existing payload node: {payload_node.get('name')}(ID:{payload_id}); preserve its current size_bits={payload_node.get('size_bits')!r} unless documentation explicitly defines different length semantics.",
                "Do not add a parallel TLV_Option/Other_Option branch when this edge update can cover the issue.",
            ],
        }

    @staticmethod
    def _selector_coded_doc_cluster_contracts(
        marked_sections: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        contracts: List[Dict[str, Any]] = []
        for section in marked_sections or []:
            if not isinstance(section, dict):
                continue
            raw_items = section.get("doc_cluster_obligations") or []
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                continue
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                cluster_kind = str(raw_item.get("cluster_kind") or "").strip().lower()
                selector = str(raw_item.get("selector") or "").strip()
                if cluster_kind != "selector_coded_tlv":
                    continue
                raw_values = raw_item.get("guard_values") or []
                if isinstance(raw_values, (str, bytes)):
                    raw_values = [raw_values]
                guard_values: List[int] = []
                for value in raw_values:
                    try:
                        parsed = int(str(value).strip(), 0)
                    except Exception:
                        continue
                    if parsed not in guard_values:
                        guard_values.append(parsed)
                raw_specs = raw_item.get("field_specs") or []
                specs_by_value: Dict[int, List[Dict[str, Any]]] = {value: [] for value in guard_values}
                if isinstance(raw_specs, list):
                    for spec in raw_specs:
                        if not isinstance(spec, dict):
                            continue
                        spec_values = spec.get("guard_values") or []
                        if isinstance(spec_values, (str, bytes)):
                            spec_values = [spec_values]
                        for value in spec_values:
                            try:
                                parsed = int(str(value).strip(), 0)
                            except Exception:
                                continue
                            if parsed not in specs_by_value:
                                continue
                            field_name = str(spec.get("field_name") or spec.get("canonical_name") or "").strip()
                            if not field_name:
                                continue
                            entry: Dict[str, Any] = {"field_name": field_name}
                            for key in ("bit_position", "byte_position", "size_bits", "direct_constraint", "format_name"):
                                if spec.get(key) not in (None, ""):
                                    entry[key] = spec.get(key)
                            specs_by_value[parsed].append(entry)
                required_variants = [
                    {
                        "selector_value": value,
                        "fields": specs_by_value.get(value) or [],
                    }
                    for value in guard_values
                ]
                contracts.append(
                    {
                        "id": str(raw_item.get("id") or "").strip(),
                        "cluster_kind": cluster_kind,
                        "selector": selector,
                        "required_guard_values": guard_values,
                        "required_variants": required_variants,
                        "instruction": (
                            "Complete this selector-coded TLV cluster in one patch. Cover every required_guard_value "
                            "with its selector-conditioned typed body fields and scalar constraints, preserve the "
                            "generic fallback, and return an empty patch if you cannot cover the whole cluster without "
                            "introducing new errors. Do not add new length_of or relation edges in this structural "
                            "cluster pass; length/count/formula repairs are handled by the relation compact pass."
                        ),
                    }
                )
        return contracts

    @staticmethod
    def _doc_structure_patch_skeleton(
        *,
        structure_issues: Dict[str, Issue],
        by_id: Dict[int, Dict[str, Any]],
        edges: Optional[Sequence[Dict[str, Any]]] = None,
        doc_cluster_contracts: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        edges = list(edges or [])
        next_node_id = max(by_id.keys(), default=0) + 1
        selector_coded_cluster_active = os.getenv("STEP2_STAGEB_SELECTOR_CODED_CLUSTER", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        selector_guidance = (
            "For selector-coded cluster repairs, create or reuse selector-conditioned variants and add typed "
            "variant body fields when the focused documentation lists body fields. Preserve the existing "
            "generic fallback/TLV branch and train-packet anchor. Do not add new length_of, count/repeat, or "
            "cross-field relation edges in this structural pass; use leaf constraints or leave relation binding "
            "for the relation compact pass."
            if selector_coded_cluster_active
            else "For selector_shell repairs, create only empty condition_on variant shells first; do not add body fields until a body_expansion issue is focused."
        )
        skeleton: Dict[str, Any] = {
            "patch_metadata": {
                "description": "Align documented scoped structure",
                "intent": "Doc structure alignment",
                "focused_issue_ids": sorted(str(issue_id) for issue_id in structure_issues),
                "next_available_node_id": next_node_id,
            },
            "new_nodes": [],
            "nodes_to_remove": [],
            "node_updates": [],
            "new_edges": [],
            "edge_updates": [],
            "edge_removes": [],
            "validation_notes": [
                "Use this as a local scaffold only; preserve existing sibling fields and constraints.",
                f"Any new_nodes must use node_id >= {next_node_id}; lower IDs may exist outside the compact prompt.",
                "Do not move or delete existing selectors unless the safe repair explicitly asks for rename/alias.",
                selector_guidance,
            ],
        }
        if selector_coded_cluster_active and doc_cluster_contracts:
            clean_contracts = [dict(contract) for contract in doc_cluster_contracts if isinstance(contract, dict)]
            if clean_contracts:
                skeleton["patch_metadata"]["doc_cluster_contracts"] = clean_contracts
                selector_coded_blueprints: List[Dict[str, Any]] = []

                def _contract_selector_node_ids(contract: Dict[str, Any]) -> List[int]:
                    selector_tokens = BatchPatchSupplier._relation_name_tokens(contract.get("selector"))
                    candidate_ids: List[int] = []
                    for issue in structure_issues.values():
                        metadata = BatchPatchSupplier._doc_structure_issue_metadata(issue)
                        repair = BatchPatchSupplier._doc_structure_safe_repair(metadata)
                        repair_selector = repair.get("selector_name") or repair.get("guard_field")
                        repair_tokens = BatchPatchSupplier._relation_name_tokens(repair_selector)
                        if selector_tokens and repair_tokens and not (selector_tokens & repair_tokens):
                            continue
                        for raw_id in repair.get("matched_selector_node_ids") or []:
                            try:
                                node_id = int(raw_id)
                            except Exception:
                                continue
                            if node_id in by_id and node_id not in candidate_ids:
                                candidate_ids.append(node_id)
                    if candidate_ids:
                        return candidate_ids
                    for node_id, node in sorted(by_id.items()):
                        node_type = str(node.get("node_type") or node.get("type") or "").strip().lower()
                        if node_type not in {"selector", "type", "field"}:
                            continue
                        node_tokens = BatchPatchSupplier._relation_name_tokens(node.get("name"))
                        if selector_tokens and selector_tokens <= node_tokens:
                            candidate_ids.append(node_id)
                    return candidate_ids

                def _variant_name_for_contract_value(
                    contract: Dict[str, Any],
                    variant: Dict[str, Any],
                ) -> str:
                    for field in variant.get("fields") or []:
                        if not isinstance(field, dict):
                            continue
                        format_name = str(field.get("format_name") or "").strip()
                        if format_name:
                            return format_name
                    selector_name = str(contract.get("selector") or "Selector").strip() or "Selector"
                    selector_value = str(variant.get("selector_value") or "Value").strip() or "Value"
                    clean_selector = re.sub(r"[^A-Za-z0-9]+", "_", selector_name).strip("_") or "Selector"
                    clean_value = re.sub(r"[^A-Za-z0-9]+", "_", selector_value).strip("_") or "Value"
                    return f"{clean_selector}_{clean_value}_Variant"

                for contract in clean_contracts:
                    selector_node_ids = _contract_selector_node_ids(contract)
                    selector_parent_id: Optional[int] = None
                    selector_bit_start: Optional[Any] = None
                    selector_size_bits: Optional[Any] = None
                    if selector_node_ids:
                        selector_node = by_id.get(selector_node_ids[0], {})
                        try:
                            raw_parent = selector_node.get("parent_id")
                            selector_parent_id = int(raw_parent) if raw_parent is not None else None
                        except Exception:
                            selector_parent_id = None
                        selector_bit_start = selector_node.get("bit_start")
                        selector_size_bits = selector_node.get("size_bits")
                    variants: List[Dict[str, Any]] = []
                    for variant in contract.get("required_variants") or []:
                        if not isinstance(variant, dict):
                            continue
                        body_fields: List[Dict[str, Any]] = []
                        for field in variant.get("fields") or []:
                            if not isinstance(field, dict):
                                continue
                            field_entry: Dict[str, Any] = {
                                "field_name": field.get("field_name") or "Documented_Field",
                            }
                            for key in ("size_bits", "bit_position", "byte_position", "direct_constraint", "format_name"):
                                if field.get(key) not in (None, ""):
                                    field_entry[key] = field.get(key)
                            body_fields.append(field_entry)
                        variants.append(
                            {
                                "selector_value": variant.get("selector_value"),
                                "variant_name": _variant_name_for_contract_value(contract, variant),
                                "selector_node_ids": selector_node_ids,
                                "parent_id": selector_parent_id,
                                "variant_bit_start_hint": (
                                    f"{selector_node_ids[0]}.bit_start + {selector_size_bits}"
                                    if selector_node_ids and selector_size_bits not in (None, "")
                                    else selector_bit_start
                                ),
                                "body_fields": body_fields,
                                "required_actions": [
                                    "add_or_reuse_condition_on_variant",
                                    "add_typed_body_fields",
                                    "add_scalar_constraints_only",
                                    "preserve_generic_fallback",
                                    "do_not_add_length_of_or_relation_edges",
                                ],
                            }
                        )
                    selector_coded_blueprints.append(
                        {
                            "cluster_id": contract.get("id"),
                            "selector": contract.get("selector"),
                            "required_guard_values": list(contract.get("required_guard_values") or []),
                            "variants": variants,
                        }
                    )
                if selector_coded_blueprints:
                    skeleton["patch_metadata"]["selector_coded_cluster_blueprint"] = selector_coded_blueprints
                skeleton["validation_notes"].append(
                    "COMPLETE SELECTOR-CODED CLUSTER CONTRACT: cover every required_guard_value in doc_cluster_contracts in this patch, or return {}. Do not resolve only a subset and leave adjacent selector values as new doc_structure/doc_constraint errors."
                )
                skeleton["validation_notes"].append(
                    "Use patch_metadata.selector_coded_cluster_blueprint as the exact write plan: for each listed variant, add or reuse the condition_on variant under parent_id, add the listed body_fields, preserve the generic fallback, and do not substitute a generic payload or placeholder for a typed body."
                )
                skeleton["validation_notes"].append(
                    "Do not add new length_of or other relation edges in selector-coded cluster patches. Add selector-conditioned variant bodies and scalar constraints only; length/count/formula binding belongs to relation compact."
                )
                for contract in clean_contracts[:3]:
                    values = contract.get("required_guard_values") or []
                    skeleton["validation_notes"].append(
                        f"Cluster {contract.get('id') or '<unnamed>'}: required_selector_values={values}; selector={contract.get('selector') or '<selector>'}."
                    )
                    for variant in (contract.get("required_variants") or [])[:12]:
                        fields = []
                        for field in variant.get("fields") or []:
                            if not isinstance(field, dict):
                                continue
                            detail = str(field.get("field_name") or "")
                            if field.get("size_bits") not in (None, ""):
                                detail += f"[{field.get('size_bits')}b]"
                            if field.get("direct_constraint"):
                                detail += f"{{{field.get('direct_constraint')}}}"
                            if detail:
                                fields.append(detail)
                        skeleton["validation_notes"].append(
                            f"  value {variant.get('selector_value')}: required_fields={fields or ['<documented typed body>']}."
                        )
        children_updates: Dict[int, List[Any]] = {}
        append_offsets: Dict[int, int] = {}
        selected_kinds: Set[str] = set()
        for issue_id, issue in structure_issues.items():
            metadata = BatchPatchSupplier._doc_structure_issue_metadata(issue)
            repair = BatchPatchSupplier._doc_structure_safe_repair(metadata)
            kind = str(repair.get("kind") or "")
            if kind:
                selected_kinds.add(kind)
            if kind == "missing_tlv_abstraction":
                reuse_skeleton = BatchPatchSupplier._doc_structure_tlv_reuse_skeleton(
                    issue_id=str(issue_id),
                    repair=repair,
                    by_id=by_id,
                    edges=edges,
                )
                if reuse_skeleton:
                    skeleton["edge_updates"].extend(reuse_skeleton.get("edge_updates") or [])
                    skeleton["validation_notes"].extend(reuse_skeleton.get("validation_notes") or [])
                else:
                    skeleton["validation_notes"].append(
                        f"{issue_id}: no existing generic TLV Len/Data branch was found; only then consider adding a new TLV abstraction."
                    )
                continue
            if kind == "missing_selector_variants":
                selector_name = repair.get("selector_name") or repair.get("guard_field")
                selector_values = repair.get("selector_values") or repair.get("guard_values") or []
                expected_variants = list(repair.get("expected_variants") or [])
                selector_node_ids = list(repair.get("matched_selector_node_ids") or [])
                if selector_node_ids:
                    skeleton["validation_notes"].append(
                        f"{issue_id}: reuse existing selector node(s) {selector_node_ids} for {selector_name}; keep them as leaf fields."
                    )
                else:
                    skeleton["validation_notes"].append(
                        f"{issue_id}: no matching selector node was found for {selector_name}; add one only if the packet layout evidence shows it."
                    )
                variant_bits = []
                for item in expected_variants[:10]:
                    name = item.get("canonical_variant_name") or item.get("variant_name")
                    values = item.get("selector_values") or []
                    variant_bits.append(f"{name} for {selector_name}={values}")
                if selector_coded_cluster_active:
                    skeleton["validation_notes"].append(
                        f"{issue_id}: selector-coded cluster phase. Add or reuse condition_on variant(s) for selector values {selector_values}; documented variants: {', '.join(variant_bits)}."
                    )
                    skeleton["validation_notes"].append(
                        f"{issue_id}: if the focused documentation lists fields for these selector values, add typed variant body fields under the selected variant now. Preserve the existing generic TLV fallback and length binding."
                    )
                else:
                    skeleton["validation_notes"].append(
                        f"{issue_id}: selector_shell phase. Add or rename condition_on variant shell(s) for selector values {selector_values}; documented variants: {', '.join(variant_bits)}."
                    )
                    skeleton["validation_notes"].append(
                        f"{issue_id}: preserve the existing generic body/TLV sequence. Do not add body fields in this patch; follow-up body_expansion issues will expand scoped fields under the shell."
                    )
                selector_id = None
                for raw_id in selector_node_ids:
                    try:
                        candidate_id = int(raw_id)
                    except Exception:
                        continue
                    if candidate_id in by_id:
                        selector_id = candidate_id
                        break
                selector_node = by_id.get(selector_id) if selector_id is not None else None
                parent_id = None
                shell_bit_start = None
                selector_parent: Optional[Dict[str, Any]] = None
                if selector_node is not None:
                    try:
                        raw_parent = selector_node.get("parent_id")
                        if raw_parent is not None:
                            parent_id = int(raw_parent)
                    except Exception:
                        parent_id = None
                if selector_id is not None and parent_id is not None and parent_id in by_id:
                    selector_parent = by_id[parent_id]
                    parent_tokens = BatchPatchSupplier._relation_name_tokens(selector_parent.get("name"))
                    parent_size = str(selector_parent.get("size_bits") or "").strip().lower()
                    parent_looks_like_fixed_header = (
                        bool(parent_tokens & {"header", "hdr", "fixed", "common"})
                        and parent_size not in {"", "variable", "var", "*"}
                        and selector_parent.get("parent_id") is not None
                    )
                    if parent_looks_like_fixed_header:
                        try:
                            grandparent_id = int(selector_parent.get("parent_id"))
                        except Exception:
                            grandparent_id = None
                        if grandparent_id is not None and grandparent_id in by_id:
                            parent_id = grandparent_id
                            shell_bit_start = f"{selector_parent.get('node_id')}.bit_start + {selector_parent.get('node_id')}.size_bits"
                    existing_variant_starts: List[str] = []
                    for edge in edges:
                        if not isinstance(edge, dict) or edge.get("rel") != "condition_on":
                            continue
                        if str(edge.get("src")) != str(selector_id):
                            continue
                        try:
                            dst_id = int(edge.get("dst"))
                        except Exception:
                            continue
                        dst_node = by_id.get(dst_id)
                        if not dst_node or str(dst_node.get("parent_id")) != str(parent_id):
                            continue
                        raw_start = dst_node.get("bit_start")
                        if raw_start is not None and str(raw_start).strip():
                            existing_variant_starts.append(str(raw_start).strip())
                    if existing_variant_starts:
                        shell_bit_start = Counter(existing_variant_starts).most_common(1)[0][0]
                    if (
                        shell_bit_start is None
                        and selector_parent is not None
                        and str(parent_id) == str(selector_parent.get("node_id"))
                    ):
                        for sibling in by_id.values():
                            if str(sibling.get("parent_id")) != str(selector_parent.get("node_id")):
                                continue
                            sibling_tokens = BatchPatchSupplier._relation_name_tokens(sibling.get("name"))
                            if not sibling_tokens & {"len", "length", "size"}:
                                continue
                            raw_start = sibling.get("bit_start")
                            raw_size = sibling.get("size_bits")
                            if raw_start is None or raw_size is None:
                                continue
                            shell_bit_start = f"{sibling.get('node_id')}.bit_start + {sibling.get('node_id')}.size_bits"
                            break
                    if shell_bit_start is None:
                        shell_bit_start = f"{selector_id}.bit_start + {selector_id}.size_bits"
                if selector_id is not None and parent_id is not None and parent_id in by_id:
                    parent = by_id[parent_id]
                    children = children_updates.setdefault(parent_id, list(parent.get("children_ids") or []))
                    existing_edges = {
                        (str(edge.get("src")), str(edge.get("dst")), str(edge.get("formula") or ""))
                        for edge in edges
                        if isinstance(edge, dict) and edge.get("rel") == "condition_on"
                    }
                    BatchPatchSupplier._doc_structure_add_length_scope_constraint(
                        skeleton=skeleton,
                        parent_id=parent_id,
                        selector_id=selector_id,
                        by_id=by_id,
                        edges=edges,
                        reason=str(issue_id),
                    )
                    if selector_coded_cluster_active:
                        exact_selector_values: List[Any] = []
                        for variant_item in expected_variants[:10]:
                            exact_selector_values.extend(variant_item.get("selector_values") or [])
                        BatchPatchSupplier._doc_structure_add_selector_fallback_exclusions(
                            skeleton=skeleton,
                            selector_id=selector_id,
                            parent_id=parent_id,
                            exact_values=exact_selector_values,
                            by_id=by_id,
                            edges=edges,
                            reason=str(issue_id),
                        )
                    sibling_nodes = [
                        node
                        for node in by_id.values()
                        if str(node.get("parent_id")) == str(parent_id)
                    ]
                    for item in expected_variants[:10]:
                        name = str(item.get("canonical_variant_name") or item.get("variant_name") or "Documented_Variant")
                        values = item.get("selector_values") or []
                        if not values:
                            continue
                        name_tokens = BatchPatchSupplier._relation_name_tokens(name)
                        existing_variant_id = BatchPatchSupplier._condition_variant_for_selector_values(
                            edges=edges,
                            selector_id=selector_id,
                            values=values,
                            by_id=by_id,
                        )
                        if existing_variant_id is not None:
                            existing_node = by_id.get(existing_variant_id, {})
                            existing_name = str(existing_node.get("name") or "")
                            if existing_name and name and existing_name != name:
                                field_role_tokens = {
                                    "count",
                                    "data",
                                    "field",
                                    "fields",
                                    "kind",
                                    "len",
                                    "length",
                                    "payload",
                                    "size",
                                    "type",
                                    "value",
                                }
                                if name_tokens and name_tokens <= field_role_tokens:
                                    skeleton["validation_notes"].append(
                                        f"{issue_id}: selector value(s) {values} already route to variant ID "
                                        f"{existing_variant_id}; skip unsafe rename to field-role name {name!r}."
                                    )
                                else:
                                    skeleton["node_updates"].append(
                                        {
                                            "node_id": existing_variant_id,
                                            "property": "name",
                                            "value": name,
                                            "reason": (
                                                f"{issue_id}: reuse existing condition_on branch for selector value(s) "
                                                f"{values}; align variant name instead of adding a duplicate shell."
                                            ),
                                        }
                                    )
                            skeleton["validation_notes"].append(
                                f"{issue_id}: selector value(s) {values} already route to variant ID {existing_variant_id}; reuse/rename it, do not add a duplicate condition_on branch."
                            )
                        for sibling in sibling_nodes:
                            if existing_variant_id is not None:
                                break
                            sibling_tokens = BatchPatchSupplier._relation_name_tokens(sibling.get("name"))
                            if name_tokens and name_tokens <= sibling_tokens:
                                try:
                                    existing_variant_id = int(sibling.get("node_id"))
                                except Exception:
                                    existing_variant_id = None
                                if existing_variant_id is not None:
                                    break
                        if existing_variant_id is None:
                            existing_variant_id = next_node_id
                            next_node_id += 1
                            skeleton["new_nodes"].append(
                                {
                                    "node_id": existing_variant_id,
                                    "name": name,
                                    "node_type": "variant",
                                    "parent_id": parent_id,
                                    "bit_start": shell_bit_start,
                                    "size_bits": 0,
                                    "data_type": "container",
                                    "message_type": parent.get("message_type") or "bidirectional",
                                    "children_ids": [],
                                    "description": f"{issue_id}: documented selector variant for {selector_name}",
                                }
                            )
                            if existing_variant_id not in children:
                                children.append(existing_variant_id)
                        for value in values:
                            formula = f"val({selector_id}) == {value}"
                            edge_key = (str(selector_id), str(existing_variant_id), formula)
                            if edge_key in existing_edges:
                                continue
                            skeleton["new_edges"].append(
                                {
                                    "src": selector_id,
                                    "dst": existing_variant_id,
                                    "rel": "condition_on",
                                    "formula": formula,
                                    "reason": f"{issue_id}: documented selector value {value} maps to {name}.",
                                }
                            )
                    skeleton["validation_notes"].append(
                        f"{issue_id}: patch_skeleton provides exact parent_id={parent_id}, bit_start={shell_bit_start}, zero-size shell variants, and condition_on edges; do not reparent existing Body/TLV nodes."
                    )
                continue
            if kind == "opaque_variant_expansion":
                variant_id = BatchPatchSupplier._doc_structure_repair_variant_id(repair)
                expected_specs = list(repair.get("expected_field_specs") or [])
                missing_fields = [
                    str(value)
                    for value in (repair.get("missing_expected_fields") or repair.get("expected_body_fields") or [])
                    if str(value or "").strip()
                ]
                opaque_child_ids = list(repair.get("opaque_child_ids") or [])
                skeleton["validation_notes"].append(
                    f"{issue_id}: expand existing variant ID {variant_id} in place; preserve its condition_on edge and length binding."
                )
                if opaque_child_ids:
                    opaque_labels = []
                    for raw_child_id in opaque_child_ids:
                        try:
                            child_id = int(raw_child_id)
                        except Exception:
                            continue
                        child = by_id.get(child_id)
                        if child:
                            opaque_labels.append(f"{child.get('name')}(ID:{child_id})")
                    skeleton["validation_notes"].append(
                        f"{issue_id}: replace/refine opaque payload child node(s) {opaque_child_ids}; if keeping one residual payload, move it after the documented fields and subtract their fixed widths from its size."
                    )
                    if opaque_labels:
                        skeleton["validation_notes"].append(
                            f"{issue_id}: opaque child context: {', '.join(opaque_labels)}. If an opaque child is an existing TLV/IE/option sequence/container, add the missing documented body field inside that existing child instead of rewriting the focused message variant's children_ids."
                        )
                skeleton["validation_notes"].append(
                    f"{issue_id}: do not invent undocumented spare/reserved/padding prefix fields; use only the documented expected_field_specs unless the issue explicitly lists such a field."
                )
                if missing_fields:
                    skeleton["validation_notes"].append(
                        f"{issue_id}: documented internal fields to model: {', '.join(missing_fields[:8])}."
                    )
                    skeleton["validation_notes"].append(
                        f"{issue_id}: missing_expected_fields is the authoritative write set for this patch. Treat expected_field_specs as context only; do not add sibling wrapper nodes for already-covered fields."
                    )
                elif expected_specs:
                    names = [str(spec.get("canonical_name") or spec.get("field_name")) for spec in expected_specs[:8]]
                    skeleton["validation_notes"].append(
                        f"{issue_id}: documented internal fields to model: {', '.join(names)}."
                    )
                if expected_specs:
                    spec_bits = []
                    for spec in expected_specs[:8]:
                        name = str(spec.get("canonical_name") or spec.get("field_name") or "Documented_Field")
                        size = spec.get("size_bits")
                        constraint = spec.get("direct_constraint")
                        details = [name]
                        if size is not None:
                            details.append(f"size_bits={size}")
                        if constraint:
                            details.append(f"constraint={constraint}")
                        spec_bits.append("(" + ", ".join(details) + ")")
                    skeleton["validation_notes"].append(
                        f"{issue_id}: expected_field_specs detail: {'; '.join(spec_bits)}."
                    )
                continue
            if kind == "weak_name_alignment":
                try:
                    node_id = int(repair.get("existing_node_id"))
                except Exception:
                    continue
                node = by_id.get(node_id, {})
                skeleton["node_updates"].append(
                    {
                        "node_id": node_id,
                        "property": "name",
                        "value": repair.get("canonical_name") or repair.get("field_name") or node.get("name"),
                        "reason": f"{issue_id}: align documented field name without duplicating wire bytes.",
                    }
                )
                direct_constraint = str(repair.get("direct_constraint") or "").strip()
                if direct_constraint:
                    constraints = list(repair.get("preserve_existing_constraints") or node.get("constraints") or [])
                    if direct_constraint not in {str(item) for item in constraints}:
                        constraints.append(direct_constraint)
                    skeleton["node_updates"].append(
                        {
                            "node_id": node_id,
                            "property": "constraints",
                            "value": constraints,
                            "reason": f"{issue_id}: preserve existing constraints and add documented field constraint.",
                        }
                    )
                continue
            if kind != "missing_scoped_field":
                continue
            variant_id = BatchPatchSupplier._doc_structure_repair_variant_id(repair)
            if variant_id is None or variant_id not in by_id:
                continue
            parent = by_id.get(variant_id, {})
            size_bits_raw = repair.get("size_bits")
            if str(size_bits_raw or "").strip().lower() in {"variable", "var", "*"} and parent.get("children_ids"):
                skeleton["validation_notes"].append(
                    f"{issue_id}: skipped local skeleton because a variable-size field cannot be safely inserted into a populated variant."
                )
                continue
            new_node_id = next_node_id
            next_node_id += 1
            new_node: Dict[str, Any] = {
                "node_id": new_node_id,
                "name": repair.get("canonical_name") or repair.get("field_name") or "Documented_Field",
                "node_type": "field",
                "parent_id": variant_id,
                "children_ids": [],
                "description": f"Documented by {repair.get('section') or 'source documentation'}",
            }
            if size_bits_raw is not None:
                new_node["size_bits"] = size_bits_raw
                try:
                    size_int = int(str(size_bits_raw).strip())
                except Exception:
                    size_int = None
            else:
                size_int = None
            if not parent.get("children_ids"):
                offset = append_offsets.setdefault(variant_id, 0)
                new_node["bit_start"] = f"{variant_id}.bit_start + {offset}"
                if size_int is not None:
                    append_offsets[variant_id] = offset + size_int
            direct_constraint = str(repair.get("direct_constraint") or "").strip()
            if direct_constraint:
                new_node["constraints"] = [direct_constraint]
            skeleton["new_nodes"].append(new_node)
            children = children_updates.setdefault(variant_id, list(parent.get("children_ids") or []))
            if new_node_id not in children:
                children.append(new_node_id)
        for variant_id, children in children_updates.items():
            skeleton["node_updates"].append(
                {
                    "node_id": variant_id,
                    "property": "children_ids",
                    "value": children,
                    "reason": "Attach newly added documented child fields or selector-shell variants to the focused parent.",
                }
            )
        if (
            selected_kinds
            and selected_kinds <= {"missing_selector_variants", "missing_tlv_abstraction"}
            and not selector_coded_cluster_active
        ):
            body_node_seen = False
            for node in skeleton.get("new_nodes") or []:
                if not isinstance(node, dict):
                    continue
                node_type = str(node.get("node_type") or "").strip().lower()
                size_bits = str(node.get("size_bits") if node.get("size_bits") is not None else "").strip()
                if node_type != "variant" or size_bits not in {"0", "0.0"}:
                    body_node_seen = True
                    break
            if not body_node_seen and BatchPatchSupplier._patch_has_effective_actions(skeleton):
                skeleton["patch_metadata"]["deterministic_safe"] = True
                skeleton["patch_metadata"]["deterministic_reason"] = (
                    "selector/TLV shell repair only updates condition_on shells or existing TLV edges; "
                    "it does not add body fields"
                )
        return skeleton

    @staticmethod
    def _doc_structure_compact_section_evidence(
        marked_sections: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep structured section evidence visible in doc-structure compact prompts."""

        evidence_sections: List[Dict[str, Any]] = []
        structured_keys = (
            "packet_formats",
            "field_definitions",
            "selector_obligations",
            "coverage_obligations",
            "doc_cluster_obligations",
        )
        for raw_section in marked_sections or []:
            if not isinstance(raw_section, dict):
                continue
            section: Dict[str, Any] = {
                "title": raw_section.get("title")
                or raw_section.get("section_title")
                or raw_section.get("number")
                or "Focused documentation section",
            }
            content = str(raw_section.get("content") or raw_section.get("summary") or "").strip()
            if content:
                section["content"] = content
            for key in structured_keys:
                value = raw_section.get(key)
                if value:
                    section[key] = copy.deepcopy(value)
            if len(section) > 1:
                evidence_sections.append(section)
        return evidence_sections[:3]

    @staticmethod
    def _merge_referenced_patch_skeleton_dependencies(
        tree: Dict[str, Any],
        patch: Dict[str, Any],
        patch_skeleton: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Restore skeleton nodes when an LLM patch references them as if present."""

        if not isinstance(patch, dict) or not isinstance(patch_skeleton, dict):
            return patch
        skeleton_nodes = {
            _node_id_key(node.get("node_id")): node
            for node in (patch_skeleton.get("new_nodes") or [])
            if isinstance(node, dict) and _node_id_key(node.get("node_id")) is not None
        }
        if not skeleton_nodes:
            return patch

        existing_ids = _tree_node_id_keys(tree)
        patch_new_ids = _patch_new_node_id_keys(patch)
        skeleton_ids = set(skeleton_nodes)
        referenced: Set[str] = set()

        def _add_ref(value: Any) -> None:
            key = _node_id_key(value)
            if key is not None:
                referenced.add(key)

        for node in patch.get("new_nodes") or []:
            if not isinstance(node, dict):
                continue
            _add_ref(node.get("parent_id"))
            for child_id in node.get("children_ids") or []:
                _add_ref(child_id)
            referenced.update(_collect_expression_node_refs(node))
        for update in patch.get("node_updates") or []:
            if not isinstance(update, dict):
                continue
            _add_ref(update.get("node_id"))
            for field_name, value in _iter_update_field_values(update):
                if field_name in {"parent_id", "children_ids"}:
                    if isinstance(value, list):
                        for item in value:
                            _add_ref(item)
                    else:
                        _add_ref(value)
                referenced.update(_collect_expression_node_refs(value))
        for edge in _patch_new_edges(patch) + list(patch.get("edge_updates") or []):
            if not isinstance(edge, dict):
                continue
            for key in ("src", "dst"):
                _add_ref(edge.get(key))
            edge_identifier = edge.get("edge_identifier")
            if isinstance(edge_identifier, dict):
                for key in ("src", "dst"):
                    _add_ref(edge_identifier.get(key))
            referenced.update(_collect_expression_node_refs(edge))

        needed = (referenced - existing_ids - patch_new_ids) & skeleton_ids
        if not needed:
            return patch

        # Pull in parent skeleton nodes transitively.
        changed = True
        while changed:
            changed = False
            for node_id in list(needed):
                parent_key = _node_id_key((skeleton_nodes.get(node_id) or {}).get("parent_id"))
                if (
                    parent_key
                    and parent_key in skeleton_ids
                    and parent_key not in existing_ids
                    and parent_key not in patch_new_ids
                    and parent_key not in needed
                ):
                    needed.add(parent_key)
                    changed = True

        completed = copy.deepcopy(patch)

        def _merge_items(key: str, items: Sequence[Any], *, prepend: bool = False) -> None:
            current = list(completed.get(key) or [])
            seen = {json.dumps(item, sort_keys=True, ensure_ascii=True, default=str) for item in current}
            additions: List[Any] = []
            for item in items:
                marker = json.dumps(item, sort_keys=True, ensure_ascii=True, default=str)
                if marker in seen:
                    continue
                seen.add(marker)
                additions.append(copy.deepcopy(item))
            completed[key] = additions + current if prepend else current + additions

        skeleton_new_nodes = [
            node for node in (patch_skeleton.get("new_nodes") or [])
            if isinstance(node, dict) and _node_id_key(node.get("node_id")) in needed
        ]
        _merge_items("new_nodes", skeleton_new_nodes, prepend=True)

        def _touches_needed_node(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            for key in ("node_id", "src", "dst"):
                if _node_id_key(item.get(key)) in needed:
                    return True
            edge_identifier = item.get("edge_identifier")
            if isinstance(edge_identifier, dict):
                if _node_id_key(edge_identifier.get("src")) in needed:
                    return True
                if _node_id_key(edge_identifier.get("dst")) in needed:
                    return True
            for _field_name, value in _iter_update_field_values(item):
                if isinstance(value, list) and any(_node_id_key(entry) in needed for entry in value):
                    return True
                if _node_id_key(value) in needed:
                    return True
            return bool(_collect_expression_node_refs(item) & needed)

        for key in ("node_updates", "new_edges", "edge_updates", "edge_removes"):
            _merge_items(
                key,
                [
                    item
                    for item in (patch_skeleton.get(key) or [])
                    if _touches_needed_node(item)
                ],
                prepend=False,
            )
        notes = list(completed.get("validation_notes") or [])
        notes.append(
            "Merged referenced patch_skeleton dependencies for node(s): "
            + ", ".join(sorted(needed, key=lambda value: int(value) if value.isdigit() else value))
        )
        completed["validation_notes"] = notes
        return completed

    @staticmethod
    def _collect_numeric_values(value: Any, out: Set[int]) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            out.add(int(value))
            return
        if isinstance(value, str):
            try:
                out.add(int(value.strip(), 0))
            except Exception:
                for match in re.finditer(r"\b\d+\b", value):
                    try:
                        out.add(int(match.group(0)))
                    except Exception:
                        pass
            return
        if isinstance(value, dict):
            for item in value.values():
                BatchPatchSupplier._collect_numeric_values(item, out)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                BatchPatchSupplier._collect_numeric_values(item, out)

    @staticmethod
    def _collect_relation_node_ids_from_metadata(metadata: Any, out: Set[int]) -> None:
        if not isinstance(metadata, dict):
            return
        for key, value in metadata.items():
            key_text = str(key or "").lower()
            if key_text in {
                "node_id",
                "node_ids",
                "target_node_ids",
                "field_node_ids",
                "variant_id",
                "existing_node_id",
                "src",
                "dst",
            }:
                BatchPatchSupplier._collect_numeric_values(value, out)
            elif isinstance(value, dict):
                BatchPatchSupplier._collect_relation_node_ids_from_metadata(value, out)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict):
                        BatchPatchSupplier._collect_relation_node_ids_from_metadata(item, out)

    @staticmethod
    def _issue_description_node_ids(text: Any) -> Set[int]:
        ids: Set[int] = set()
        for match in re.finditer(r"\bID:(\d+)\b|\bnode[_\s-]?id['\"]?\s*[:=]\s*['\"]?(\d+)", str(text or ""), re.IGNORECASE):
            raw = match.group(1) or match.group(2)
            try:
                ids.add(int(raw))
            except Exception:
                pass
        return ids

    @staticmethod
    def _relation_name_tokens(value: Any) -> Set[str]:
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value or ""))
        tokens = {
            tok
            for tok in re.sub(r"[^a-z0-9]+", "_", text.lower()).split("_")
            if tok
        }
        for token in list(tokens):
            if len(token) > 4 and token.endswith("ies"):
                tokens.add(token[:-3] + "y")
            elif len(token) > 4 and token.endswith("es"):
                tokens.add(token[:-2])
            elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                tokens.add(token[:-1])
            hex_match = re.fullmatch(r"0x([0-9a-f]+)", token)
            if hex_match:
                try:
                    value_int = int(hex_match.group(1), 16)
                except ValueError:
                    value_int = None
                if value_int is not None:
                    tokens.add(str(value_int))
                    tokens.add(f"{value_int:02d}")
                    tokens.add(f"fc{value_int:02d}")
            suffix_match = re.fullmatch(r"([a-z]+)(\d+)", token)
            if suffix_match:
                digits = suffix_match.group(2)
                tokens.add(digits)
                try:
                    value_int = int(digits, 10)
                except ValueError:
                    value_int = None
                if value_int is not None:
                    tokens.add(str(value_int))
                    tokens.add(f"{value_int:02d}")
                    tokens.add(f"0x{value_int:02x}")
        words = [tok for tok in tokens if tok.isalpha() and len(tok) > 1]
        if len(words) > 1:
            tokens.add("".join(word[0] for word in words))
        return tokens

    @staticmethod
    def _relation_name_match_score(field_tokens: Set[str], node_tokens: Set[str]) -> int:
        if not field_tokens or not node_tokens:
            return 0
        overlap = field_tokens & node_tokens
        if not overlap:
            return 0
        if field_tokens <= node_tokens:
            return 100 + len(field_tokens)
        if node_tokens <= field_tokens:
            return 80 + len(node_tokens)
        if len(overlap) >= 2:
            return 50 + len(overlap)
        token = next(iter(overlap))
        # Single-token matches are intentionally allowed only for substantial
        # field words. This keeps the prompt compact while still finding local
        # neighborhoods for phrases like "subsequent option" or "fragment offset".
        if len(token) >= 4:
            return 20
        return 0

    @staticmethod
    def _relation_context_tokens(metadata: Dict[str, Any]) -> Set[str]:
        text = f"{metadata.get('section') or ''} {metadata.get('relation_text') or ''}"
        tokens = BatchPatchSupplier._relation_name_tokens(text)
        generic = {
            "and",
            "byte",
            "bytes",
            "code",
            "count",
            "data",
            "field",
            "function",
            "including",
            "length",
            "not",
            "of",
            "quantity",
            "read",
            "register",
            "registers",
            "request",
            "response",
            "status",
            "statuses",
            "the",
            "value",
            "write",
        }
        return {token for token in tokens if len(token) > 2 and token not in generic}

    @staticmethod
    def _relation_node_is_payload_like(node: Dict[str, Any]) -> bool:
        name_tokens = BatchPatchSupplier._relation_name_tokens(node.get("name"))
        if name_tokens & {
            "bits",
            "body",
            "data",
            "input",
            "inputs",
            "output",
            "outputs",
            "payload",
            "register",
            "registers",
            "status",
            "value",
            "values",
        }:
            return True
        size_bits = str(node.get("size_bits") or "").strip().lower()
        return size_bits in {"variable", "var", "*", "payload"}

    @staticmethod
    def _packed_bits_count_hint(node_id: int, by_id: Dict[int, Dict[str, Any]]) -> str:
        node = by_id.get(int(node_id), {})
        try:
            parent_id = int(node.get("parent_id")) if isinstance(node, dict) and node.get("parent_id") is not None else None
        except Exception:
            parent_id = None
        if parent_id is None:
            return "external"
        best: Optional[int] = None
        for sibling_id, sibling in by_id.items():
            if int(sibling_id) == int(node_id):
                continue
            try:
                if int(sibling.get("parent_id")) != parent_id:
                    continue
            except Exception:
                continue
            tokens = BatchPatchSupplier._relation_name_tokens(sibling.get("name"))
            if "byte" in tokens or "bytes" in tokens:
                continue
            if tokens & {"quantity", "count", "number"}:
                best = int(sibling_id)
                break
        return f"val({best})" if best is not None else "external"

    @staticmethod
    def _relation_node_path_tokens(node_id: int, by_id: Dict[int, Dict[str, Any]]) -> Set[str]:
        tokens: Set[str] = set()
        node = by_id.get(int(node_id))
        seen: Set[int] = set()
        for _ in range(8):
            if not isinstance(node, dict):
                break
            try:
                nid = int(node.get("node_id"))
            except Exception:
                break
            if nid in seen:
                break
            seen.add(nid)
            tokens.update(BatchPatchSupplier._relation_name_tokens(node.get("name")))
            try:
                parent_id = int(node.get("parent_id")) if node.get("parent_id") is not None else None
            except Exception:
                parent_id = None
            if parent_id is None:
                break
            node = by_id.get(parent_id)
        return tokens

    @staticmethod
    def _relation_metadata_field_names(metadata: Any) -> List[str]:
        if not isinstance(metadata, dict):
            return []
        names: List[str] = []
        for key in ("field_names", "related_fields", "primary_field"):
            raw_value = metadata.get(key)
            if isinstance(raw_value, str):
                names.append(raw_value)
            elif isinstance(raw_value, (list, tuple)):
                names.extend(str(item) for item in raw_value if item is not None)
        return names

    @staticmethod
    def _relation_metadata_node_ids(metadata: Any) -> Set[int]:
        ids: Set[int] = set()
        BatchPatchSupplier._collect_relation_node_ids_from_metadata(metadata, ids)
        return ids

    @staticmethod
    def _relation_issue_metadata(issue: Issue) -> Dict[str, Any]:
        target = getattr(issue, "target", None)
        metadata = getattr(target, "metadata", None) if target is not None else None
        return metadata if isinstance(metadata, dict) else {}

    def _rank_relation_issue(self, item: Tuple[str, Issue]) -> Tuple[int, int, str]:
        issue_id, issue = item
        metadata = self._relation_issue_metadata(issue)
        ids = self._relation_metadata_node_ids(metadata)
        description = str(getattr(issue, "description", "") or "").lower()
        missing_penalty = 1 if "no matching tree node" in description else 0
        kind = str(metadata.get("kind") or "")
        kind_priority = {"conditional_zero": 0, "conditional_value": 0, "length_relation": 1, "mod_alignment": 2}.get(kind, 3)
        # Prefer actionable obligations with a concrete existing field node; a
        # second call can then handle structure-creating obligations after the
        # simpler local patch has been validated.
        return (kind_priority, missing_penalty, -len(ids), str(issue_id))

    @staticmethod
    def _pick_relation_target_node_id(
        metadata: Dict[str, Any],
        target_node_ids: Sequence[int],
        by_id: Dict[int, Dict[str, Any]],
    ) -> Optional[int]:
        field_node_ids = metadata.get("field_node_ids")
        if not isinstance(field_node_ids, dict):
            return int(target_node_ids[0]) if target_node_ids else None

        def _coerce_ids(values: Any) -> List[int]:
            ids: List[int] = []
            if isinstance(values, (str, int)):
                values = [values]
            if isinstance(values, (list, tuple, set)):
                for value in values:
                    try:
                        ids.append(int(value))
                    except Exception:
                        continue
            return ids

        primary_field = metadata.get("primary_field")
        primary_ids = _coerce_ids(field_node_ids.get(primary_field)) if primary_field is not None else []
        if not primary_ids:
            for values in field_node_ids.values():
                primary_ids.extend(_coerce_ids(values))
                if primary_ids:
                    break
        related_ids: List[int] = []
        for field, values in field_node_ids.items():
            if primary_field is not None and str(field) == str(primary_field):
                continue
            related_ids.extend(_coerce_ids(values))
        related_parents = {
            by_id[rid].get("parent_id")
            for rid in related_ids
            if rid in by_id and isinstance(by_id.get(rid), dict)
        }
        if related_parents:
            for nid in primary_ids:
                node = by_id.get(nid)
                if isinstance(node, dict) and node.get("parent_id") in related_parents:
                    return nid
        if primary_ids:
            return primary_ids[0]
        return int(target_node_ids[0]) if target_node_ids else None

    @staticmethod
    def _relation_related_node_ids(metadata: Dict[str, Any]) -> Set[int]:
        field_node_ids = metadata.get("field_node_ids")
        if not isinstance(field_node_ids, dict):
            return set()
        primary_field = metadata.get("primary_field")
        related: Set[int] = set()
        for field, values in field_node_ids.items():
            if primary_field is not None and str(field) == str(primary_field):
                continue
            if isinstance(values, (str, int)):
                values = [values]
            if isinstance(values, (list, tuple, set)):
                for value in values:
                    try:
                        related.add(int(value))
                    except Exception:
                        continue
        return related

    @staticmethod
    def _relation_patch_skeleton(
        *,
        issue_id: str,
        issue: Issue,
        target_node_ids: Sequence[int],
        by_id: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        metadata = BatchPatchSupplier._relation_issue_metadata(issue)
        source_relation = str(metadata.get("relation_text") or getattr(issue, "description", "") or "")
        obligation_kind = str(metadata.get("kind") or "")
        skeleton: Dict[str, Any] = {
            "patch_metadata": {
                "description": f"Encode documentation relation {issue_id}",
                "intent": "Doc relational obligation",
                "focused_issue_id": issue_id,
            },
            "new_nodes": [],
            "nodes_to_remove": [],
            "node_updates": [],
            "new_edges": [],
            "edge_updates": [],
            "edge_removes": [],
            "validation_notes": [
                f"Source relation: {source_relation[:240]}",
                "Preserve all existing constraints on updated nodes.",
            ],
        }
        if target_node_ids:
            if obligation_kind == "count_relation":
                field_node_ids = metadata.get("field_node_ids")
                primary_field = metadata.get("primary_field")

                def _coerce_ids(values: Any) -> List[int]:
                    if isinstance(values, (str, int)):
                        values = [values]
                    ids: List[int] = []
                    if isinstance(values, (list, tuple, set)):
                        for value in values:
                            try:
                                node_id = int(value)
                            except Exception:
                                continue
                            if node_id in by_id:
                                ids.append(node_id)
                    return ids

                primary_ids: List[int] = []
                if isinstance(field_node_ids, dict) and primary_field is not None:
                    primary_ids = _coerce_ids(field_node_ids.get(primary_field))
                related_ids = sorted(BatchPatchSupplier._relation_related_node_ids(metadata))
                related_ids = [node_id for node_id in related_ids if node_id in by_id]
                if primary_ids and related_ids:
                    src_id = int(primary_ids[0])
                    dst_id = int(related_ids[0])
                    skeleton["patch_metadata"]["source"] = "deterministic_count_relation_repair"
                    skeleton["patch_metadata"]["deterministic_safe"] = True
                    skeleton["new_edges"].append(
                        {
                            "src": src_id,
                            "dst": dst_id,
                            "rel": "repeat_count",
                            "formula": f"val({src_id})",
                        }
                    )
                    skeleton["validation_notes"].append(
                        "Added a generic repeat_count edge from the documented count field to the repeated entry/container."
                    )
                    return skeleton

            picked_node_id = BatchPatchSupplier._pick_relation_target_node_id(metadata, target_node_ids, by_id)
            if picked_node_id is None:
                return skeleton
            node_id = int(picked_node_id)
            node = by_id.get(node_id, {})
            existing_constraints = node.get("constraints") if isinstance(node, dict) else []
            if not isinstance(existing_constraints, list):
                existing_constraints = []
            relation_field_ids = sorted(BatchPatchSupplier._relation_metadata_node_ids(metadata))
            relation_field_names = BatchPatchSupplier._relation_metadata_field_names(metadata)
            constraint_ir_hint = constraint_ir_to_text(
                kind=obligation_kind or "relational_obligation",
                expr=source_relation,
                field_ids=relation_field_ids,
                field_names=relation_field_names,
                source=source_relation,
            )

            def _constraints_with_ir(*extra: str) -> List[Any]:
                values = list(existing_constraints)
                if constraint_ir_hint not in values:
                    values.append(constraint_ir_hint)
                for item in extra:
                    if item and item not in values:
                        values.append(item)
                return values

            numeric_values = metadata.get("numeric_values")
            if obligation_kind == "length_relation" and isinstance(numeric_values, list) and numeric_values:
                unit = numeric_values[-1]
                relation_lower = source_relation.lower()
                if "multiple" in relation_lower or "modulo" in relation_lower or "aligned" in relation_lower:
                    skeleton["node_updates"] = [
                        {
                            "node_id": node_id,
                            "property": "constraints",
                            "value": _constraints_with_ir(f"formula: val({node_id}) % {unit} == 0"),
                            "reason": "Encode the documented multiple/alignment relation as a local value constraint.",
                        }
                    ]
                else:
                    skeleton["node_updates"] = [
                        {
                            "node_id": node_id,
                            "property": "value_expr",
                            "value": f"val({node_id}) * {unit}",
                            "reason": "Encode the documented semantic unit/value relation without changing packet layout.",
                        },
                        {
                            "node_id": node_id,
                            "property": "constraints",
                            "value": _constraints_with_ir(),
                            "reason": "Record the documented relation as protocol knowledge without forcing an unsafe parser predicate.",
                        },
                    ]
            elif obligation_kind == "mod_alignment" and isinstance(numeric_values, list) and numeric_values:
                unit = numeric_values[0]
                relation_lower = source_relation.lower()
                if re.search(r"\b(?:number\s+of|measured\s+in\s+units?\s+of|words?)\b", relation_lower):
                    skeleton["node_updates"] = [
                        {
                            "node_id": node_id,
                            "property": "value_expr",
                            "value": f"val({node_id}) * {unit}",
                            "reason": "Encode the documented semantic unit/value relation without changing packet layout.",
                        },
                        {
                            "node_id": node_id,
                            "property": "constraints",
                            "value": _constraints_with_ir(),
                            "reason": "Record the documented unit/alignment relation as protocol knowledge.",
                        },
                    ]
                    return skeleton
                formula_hint = f"formula: val({node_id}) % {unit} == 0"
                skeleton["node_updates"] = [
                    {
                        "node_id": node_id,
                        "property": "constraints",
                        "value": _constraints_with_ir(formula_hint),
                        "reason": "Encode the focused documentation relation as an explicit local formula.",
                    }
                ]
            elif obligation_kind == "post_end_zero_padding":
                count_hint = BatchPatchSupplier._packed_bits_count_hint(node_id, by_id)
                constraint = f"packed_bits: unit_bits=1; order=lsb_first; pad=zero; count={count_hint}"
                skeleton["node_updates"] = [
                    {
                        "node_id": node_id,
                        "property": "constraints",
                        "value": _constraints_with_ir(constraint),
                        "reason": (
                            "Encode the documented packed-bit final-byte padding rule. "
                            "Use count=val(ID) only when the bit-count field is in this same message; otherwise keep count=external."
                        ),
                    }
                ]
            else:
                skeleton["node_updates"] = [
                    {
                        "node_id": node_id,
                        "property": "constraints",
                        "value": _constraints_with_ir(),
                        "reason": (
                            "Record the focused documentation relation as explicit protocol knowledge. "
                            "Add a formula only if it can be expressed safely with shown node IDs."
                        ),
                    }
                ]
        return skeleton

    @staticmethod
    def _relation_section_text(section: Any) -> str:
        if isinstance(section, dict):
            parts: List[str] = []
            for key in ("title", "heading", "content", "text", "summary"):
                value = section.get(key)
                if value:
                    parts.append(str(value))
            for key in ("pages", "packet_formats", "fields"):
                value = section.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            for subkey in ("title", "format_name", "field_name", "content", "text"):
                                subvalue = item.get(subkey)
                                if subvalue:
                                    parts.append(str(subvalue))
                        elif item is not None:
                            parts.append(str(item))
            if parts:
                return "\n".join(parts)
        return json.dumps(section, ensure_ascii=True, default=str)

    def _relation_source_windows(
        self,
        relation_issues: Dict[str, Issue],
        *,
        radius: int = 700,
        max_windows: int = 3,
    ) -> List[str]:
        windows: List[str] = []
        for issue_id, issue in relation_issues.items():
            metadata = self._relation_issue_metadata(issue)
            relation_text = str(metadata.get("relation_text") or "").strip()
            section_title = str(metadata.get("section") or "").strip()
            if not relation_text:
                continue
            matched_window: Optional[str] = None
            section_pool = list(self.sections or [])
            for title_pass in (True, False):
                for section in section_pool:
                    if title_pass and section_title and isinstance(section, dict):
                        raw_title = str(section.get("title") or section.get("heading") or "")
                        if raw_title and raw_title != section_title:
                            continue
                    text = self._relation_section_text(section)
                    needles = [relation_text, relation_text[:160], relation_text[:96], relation_text[:48]]
                    idx = -1
                    for needle in needles:
                        needle = needle.strip()
                        if len(needle) < 12:
                            continue
                        idx = text.find(needle)
                        if idx >= 0:
                            break
                    if idx < 0:
                        continue
                    start = max(0, idx - radius)
                    end = min(len(text), idx + len(relation_text) + radius)
                    matched_window = re.sub(r"\s+", " ", text[start:end]).strip()
                    break
                if matched_window:
                    break
            if matched_window:
                windows.append(f"- {issue_id} local source window: {matched_window}")
                if len(windows) >= max_windows:
                    break
        return windows

    def _relation_compact_prompt_context(
        self,
        state: "TreeState",
        marked_sections: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]], Optional[str]]:
        issues = getattr(state, "pending_issues", {}) or {}
        relation_issues = {
            issue_id: issue
            for issue_id, issue in issues.items()
            if self._is_detail_relation_issue_id(issue_id)
        }
        if not relation_issues:
            return state.tree, marked_sections, None
        focus_single = os.getenv("STEP2_RELATIONAL_COMPACT_SINGLE_ISSUE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if focus_single and len(relation_issues) > 1:
            focused_id, focused_issue = sorted(relation_issues.items(), key=self._rank_relation_issue)[0]
            relation_issues = {focused_id: focused_issue}

        tree = normalize_protocol_tree(copy.deepcopy(state.tree))
        nodes = [node for node in tree.get("nodes", []) or [] if isinstance(node, dict)]
        edges = [edge for edge in tree.get("edges", []) or [] if isinstance(edge, dict)]
        by_id: Dict[int, Dict[str, Any]] = {}
        children_by_parent: Dict[int, Set[int]] = {}
        for node in nodes:
            try:
                nid = int(node.get("node_id"))
            except Exception:
                continue
            by_id[nid] = node
            parent = node.get("parent_id")
            try:
                if parent is not None:
                    children_by_parent.setdefault(int(parent), set()).add(nid)
            except Exception:
                pass
            for child in node.get("children_ids") or []:
                try:
                    children_by_parent.setdefault(nid, set()).add(int(child))
                except Exception:
                    pass

        seed_ids: Set[int] = set()
        evidence_lines: List[str] = []
        relation_field_names: List[str] = []
        relation_context_tokens: Set[str] = set()
        metadata_has_node_ids = False
        has_post_end_issue = False
        for issue_id, issue in relation_issues.items():
            desc = str(getattr(issue, "description", "") or "")
            evidence_lines.append(f"- {issue_id}: {desc}")
            seed_ids.update(self._issue_description_node_ids(desc))
            target = getattr(issue, "target", None)
            metadata = getattr(target, "metadata", None) if target is not None else None
            if isinstance(metadata, dict):
                has_post_end_issue = has_post_end_issue or str(metadata.get("kind") or "") == "post_end_zero_padding"
                self._collect_relation_node_ids_from_metadata(metadata, seed_ids)
                relation_field_names.extend(self._relation_metadata_field_names(metadata))
                relation_context_tokens.update(self._relation_context_tokens(metadata))
                field_node_ids = metadata.get("field_node_ids")
                if isinstance(field_node_ids, dict):
                    metadata_has_node_ids = any(bool(values) for values in field_node_ids.values())
            ident = getattr(target, "identifier", None) if target is not None else None
            try:
                seed_ids.add(int(str(ident)))
            except Exception:
                pass

        stop = {
            "a",
            "an",
            "and",
            "beginning",
            "field",
            "fields",
            "for",
            "in",
            "of",
            "subsequent",
            "the",
            "to",
            "value",
        }
        name_match_candidates: List[Tuple[int, int, int]] = []
        for field_name in relation_field_names:
            field_tokens = self._relation_name_tokens(field_name) - stop
            if not field_tokens:
                continue
            for nid, node in by_id.items():
                node_tokens = self._relation_name_tokens(node.get("name")) - stop
                if not node_tokens:
                    continue
                score = self._relation_name_match_score(field_tokens, node_tokens)
                if score > 0:
                    name_match_candidates.append((score, -len(node_tokens), nid))

        try:
            name_match_limit = max(0, int(os.getenv("STEP2_RELATIONAL_COMPACT_NAME_MATCH_LIMIT", "16")))
        except Exception:
            name_match_limit = 16
        name_matched_ids: Set[int] = set()
        for _score, _specificity, nid in sorted(name_match_candidates, reverse=True):
            if nid in name_matched_ids:
                continue
            name_matched_ids.add(nid)
            if len(name_matched_ids) >= name_match_limit:
                break
        context_matched_ids: Set[int] = set()
        context_preferred_id: Optional[int] = None
        if relation_context_tokens:
            context_candidates: List[Tuple[int, int, int]] = []
            for field_name in relation_field_names:
                field_tokens = self._relation_name_tokens(field_name) - stop
                if not field_tokens:
                    continue
                for nid, node in by_id.items():
                    node_tokens = self._relation_name_tokens(node.get("name")) - stop
                    field_overlap = field_tokens & node_tokens
                    context_overlap = relation_context_tokens & node_tokens
                    if field_overlap and context_overlap:
                        score = len(field_overlap) * 4 + len(context_overlap) * 3
                        context_candidates.append((score, -len(node_tokens), nid))
            if has_post_end_issue:
                for nid, node in by_id.items():
                    node_tokens = self._relation_name_tokens(node.get("name")) - stop
                    context_overlap = relation_context_tokens & node_tokens
                    if not context_overlap or not self._relation_node_is_payload_like(node):
                        continue
                    path_overlap = relation_context_tokens & self._relation_node_path_tokens(nid, by_id)
                    size_bits = str(node.get("size_bits") or "").strip().lower()
                    variable_bonus = 5 if size_bits in {"variable", "var", "*", "payload"} else 0
                    score = len(context_overlap) * 3 + len(path_overlap - context_overlap) * 4 + variable_bonus
                    context_candidates.append((score, -len(node_tokens), nid))
            for _score, _specificity, nid in sorted(context_candidates, reverse=True):
                if context_preferred_id is None:
                    context_preferred_id = nid
                context_matched_ids.add(nid)
                if len(context_matched_ids) >= max(1, name_match_limit):
                    break
        if context_matched_ids and not metadata_has_node_ids:
            name_matched_ids = set(context_matched_ids)
        seed_ids.update(name_matched_ids)

        preferred_target_id: Optional[int] = None
        if len(relation_issues) == 1:
            focused_issue = next(iter(relation_issues.values()))
            metadata = self._relation_issue_metadata(focused_issue)
            if context_preferred_id is not None and not metadata_has_node_ids:
                picked_id = int(context_preferred_id)
            else:
                picked_id = self._pick_relation_target_node_id(metadata, sorted(seed_ids), by_id)
            related_ids = self._relation_related_node_ids(metadata)
            preferred_target_id = picked_id
            if (
                picked_id is not None
                and not related_ids
                and str(metadata.get("kind") or "") == "length_relation"
            ):
                picked_node = by_id.get(int(picked_id))
                try:
                    parent_id = int(picked_node.get("parent_id")) if isinstance(picked_node, dict) and picked_node.get("parent_id") is not None else None
                except Exception:
                    parent_id = None
                if parent_id is not None:
                    sibling_ids = sorted(children_by_parent.get(parent_id, set()) - {int(picked_id)})
                    related_ids.update(sibling_ids[:8])
            if picked_id is not None and (metadata.get("field_node_ids") or metadata.get("primary_field")):
                seed_ids = {int(picked_id)} | {rid for rid in related_ids if rid in by_id}
                name_matched_ids = {nid for nid in name_matched_ids if nid in seed_ids}

        if not seed_ids:
            return state.tree, marked_sections, None

        focus_node_ids = sorted(seed_ids)
        if preferred_target_id is not None and int(preferred_target_id) in seed_ids:
            focus_node_ids = [int(preferred_target_id)] + [
                nid for nid in focus_node_ids if nid != int(preferred_target_id)
            ]
        include_ids: Set[int] = set(seed_ids)
        frontier = set(seed_ids)
        radius = max(0, int(os.getenv("STEP2_RELATIONAL_COMPACT_RADIUS", "1")))
        for _ in range(radius):
            next_frontier: Set[int] = set()
            for nid in frontier:
                node = by_id.get(nid)
                if node:
                    try:
                        parent = node.get("parent_id")
                        if parent is not None:
                            next_frontier.add(int(parent))
                    except Exception:
                        pass
                    next_frontier.update(children_by_parent.get(nid, set()))
                for edge in edges:
                    try:
                        src = int(edge.get("src"))
                        dst = int(edge.get("dst"))
                    except Exception:
                        continue
                    if src == nid or dst == nid:
                        next_frontier.add(src)
                        next_frontier.add(dst)
            next_frontier -= include_ids
            include_ids.update(next_frontier)
            frontier = next_frontier

        root_id = tree.get("root_node_id")
        try:
            if root_id is not None:
                include_ids.add(int(root_id))
        except Exception:
            pass

        compact_nodes = [copy.deepcopy(node) for node in nodes if int(node.get("node_id", -1)) in include_ids]
        compact_edges = []
        for edge in edges:
            try:
                src = int(edge.get("src"))
                dst = int(edge.get("dst"))
            except Exception:
                continue
            if src in include_ids and dst in include_ids:
                compact_edges.append(copy.deepcopy(edge))

        compact_tree = {
            "protocol_name": tree.get("protocol_name"),
            "root_node_id": tree.get("root_node_id"),
            "nodes": compact_nodes,
            "edges": compact_edges,
            "_prompt_context": {
                "type": "doc_relational_compact",
                "note": "Partial prompt-only tree view. Node IDs are real; apply patches to the full tree by ID.",
                "full_node_count": len(nodes),
                "full_edge_count": len(edges),
                "shown_node_ids": sorted(include_ids),
                "seed_node_ids": sorted(seed_ids),
                "name_matched_node_ids": sorted(name_matched_ids),
                "focused_issue_ids": sorted(str(issue_id) for issue_id in relation_issues),
                "patch_skeleton": self._relation_patch_skeleton(
                    issue_id=next(iter(relation_issues.keys())),
                    issue=next(iter(relation_issues.values())),
                    target_node_ids=focus_node_ids,
                    by_id=by_id,
                )
                if len(relation_issues) == 1
                else None,
            },
        }
        source_windows = self._relation_source_windows(relation_issues)
        if source_windows:
            evidence_lines.append("")
            evidence_lines.append("LOCAL SOURCE WINDOWS")
            evidence_lines.extend(source_windows)
        compact_sections = [
            {
                "title": "Relational obligation evidence",
                "content": "\n".join(evidence_lines),
            }
        ]
        feedback = (
            "RELATION COMPACT CONTEXT: the CURRENT TREE block is a prompt-only neighborhood, not the full tree. "
            "Patch only real node IDs shown in this compact context, or add a local constraint/edge using those IDs. "
            "Do not delete or rewrite unseen nodes."
        )
        return compact_tree, compact_sections, feedback

    def _doc_constraint_compact_prompt_context(
        self,
        state: "TreeState",
        marked_sections: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]], Optional[str]]:
        issues = getattr(state, "pending_issues", {}) or {}
        doc_issues = {
            issue_id: issue
            for issue_id, issue in issues.items()
            if self._is_detail_doc_constraint_issue_id(issue_id)
        }
        if not doc_issues:
            return state.tree, marked_sections, None
        has_safe_repairs = any(
            self._doc_constraint_repairs(self._doc_constraint_issue_metadata(issue))
            for issue in doc_issues.values()
        )
        focus_single = os.getenv("STEP2_DOC_CONSTRAINT_COMPACT_SINGLE_ISSUE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if focus_single and len(doc_issues) > 1:
            try:
                group_limit = max(1, int(os.getenv("STEP2_DOC_CONSTRAINT_COMPACT_GROUP_LIMIT", "8")))
            except Exception:
                group_limit = 8

            def _item_priority(item: Tuple[str, Issue]) -> Tuple[int, str]:
                repairs = self._doc_constraint_repairs(self._doc_constraint_issue_metadata(item[1]))
                formula_priority = (
                    0
                    if any(
                        str(repair.get("kind") or "") in {"guarded_variant_formula", "variant_formula"}
                        for repair in repairs
                    )
                    else 1
                )
                return formula_priority, str(item[0])

            ranked_issues = sorted(doc_issues.items(), key=_item_priority)
            groups: Dict[int, List[Tuple[str, Issue]]] = {}
            for issue_id, issue in ranked_issues:
                repairs = self._doc_constraint_repairs(self._doc_constraint_issue_metadata(issue))
                if not repairs:
                    continue
                node_id = self._doc_constraint_repair_node_id(repairs[0])
                if node_id is None:
                    continue
                groups.setdefault(int(node_id), []).append((issue_id, issue))
            if groups:
                best_node_id, best_items = sorted(
                    groups.items(),
                    key=lambda item: (
                        -len(item[1]),
                        _item_priority(item[1][0]),
                        item[0],
                    ),
                )[0]
                del best_node_id
                doc_issues = {issue_id: issue for issue_id, issue in best_items[:group_limit]}
            else:
                focused_id, focused_issue = ranked_issues[0]
                doc_issues = {focused_id: focused_issue}

        tree = normalize_protocol_tree(copy.deepcopy(state.tree))
        nodes = [node for node in tree.get("nodes", []) or [] if isinstance(node, dict)]
        edges = [edge for edge in tree.get("edges", []) or [] if isinstance(edge, dict)]
        by_id: Dict[int, Dict[str, Any]] = {}
        children_by_parent: Dict[int, Set[int]] = {}
        for node in nodes:
            try:
                nid = int(node.get("node_id"))
            except Exception:
                continue
            by_id[nid] = node
            parent = node.get("parent_id")
            try:
                if parent is not None:
                    children_by_parent.setdefault(int(parent), set()).add(nid)
            except Exception:
                pass
            for child in node.get("children_ids") or []:
                try:
                    children_by_parent.setdefault(nid, set()).add(int(child))
                except Exception:
                    pass

        seed_ids: Set[int] = set()
        evidence_lines: List[str] = []
        for issue_id, issue in doc_issues.items():
            desc = str(getattr(issue, "description", "") or "")
            metadata = self._doc_constraint_issue_metadata(issue)
            repairs = self._doc_constraint_repairs(metadata)
            evidence_lines.append(f"- {issue_id}: {desc}")
            evidence_lines.append(
                "  safe_repairs: "
                + json.dumps(repairs[:3], ensure_ascii=True, sort_keys=True)
            )
            for key in ("target_node_ids", "guard_node_ids", "candidate_variant_ids", "forbidden_node_ids"):
                values = metadata.get(key)
                if isinstance(values, (list, tuple, set)):
                    for value in values:
                        try:
                            seed_ids.add(int(value))
                        except Exception:
                            pass
            for repair in repairs:
                for key in ("node_id", "variant_id", "target_node_id", "guard_node_id"):
                    try:
                        value = repair.get(key)
                        if value is not None:
                            seed_ids.add(int(value))
                    except Exception:
                        pass

        if not seed_ids:
            return state.tree, marked_sections, None

        include_ids: Set[int] = set(seed_ids)
        frontier = set(seed_ids)
        radius = max(0, int(os.getenv("STEP2_DOC_CONSTRAINT_COMPACT_RADIUS", "1")))
        for _ in range(radius):
            next_frontier: Set[int] = set()
            for nid in frontier:
                node = by_id.get(nid)
                if node:
                    try:
                        parent = node.get("parent_id")
                        if parent is not None:
                            next_frontier.add(int(parent))
                    except Exception:
                        pass
                    next_frontier.update(children_by_parent.get(nid, set()))
                for edge in edges:
                    try:
                        src = int(edge.get("src"))
                        dst = int(edge.get("dst"))
                    except Exception:
                        continue
                    if src == nid or dst == nid:
                        next_frontier.add(src)
                        next_frontier.add(dst)
            next_frontier -= include_ids
            include_ids.update(next_frontier)
            frontier = next_frontier

        root_id = tree.get("root_node_id")
        try:
            if root_id is not None:
                include_ids.add(int(root_id))
        except Exception:
            pass

        compact_nodes = [copy.deepcopy(node) for node in nodes if int(node.get("node_id", -1)) in include_ids]
        compact_edges = []
        for edge in edges:
            try:
                src = int(edge.get("src"))
                dst = int(edge.get("dst"))
            except Exception:
                continue
            if src in include_ids and dst in include_ids:
                compact_edges.append(copy.deepcopy(edge))

        compact_tree = {
            "protocol_name": tree.get("protocol_name"),
            "root_node_id": tree.get("root_node_id"),
            "nodes": compact_nodes,
            "edges": compact_edges,
            "_prompt_context": {
                "type": "doc_constraint_compact",
                "note": "Partial prompt-only tree view. Node IDs are real; apply patches to the full tree by ID.",
                "full_node_count": len(nodes),
                "full_edge_count": len(edges),
                "shown_node_ids": sorted(include_ids),
                "seed_node_ids": sorted(seed_ids),
                "focused_issue_ids": sorted(str(issue_id) for issue_id in doc_issues),
                "has_safe_repairs": has_safe_repairs,
                "skip_llm_if_no_safe_repairs": not has_safe_repairs,
                "patch_skeleton": (
                    self._doc_constraint_patch_skeleton(
                        issue_id=next(iter(doc_issues.keys())),
                        issue=next(iter(doc_issues.values())),
                        by_id=by_id,
                    )
                    if len(doc_issues) == 1
                    else self._doc_constraint_group_patch_skeleton(doc_issues, by_id)
                ),
            },
        }
        compact_sections = [
            {
                "title": "Documentation constraint evidence",
                "content": "\n".join(evidence_lines),
            }
        ]
        feedback = (
            "DOC CONSTRAINT COMPACT CONTEXT: the CURRENT TREE block is a prompt-only neighborhood, not the full tree. "
            "Patch only real node IDs shown in this compact context. Use guarded formulas for catch-all/generic variants. "
            "If the skeleton has no node_updates, the validator found no safe local repair; return {} unless the shown "
            "neighborhood proves a scoped repair is safe."
        )
        return compact_tree, compact_sections, feedback

    def _doc_structure_compact_prompt_context(
        self,
        state: "TreeState",
        marked_sections: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Sequence[Dict[str, Any]], Optional[str]]:
        issues = getattr(state, "pending_issues", {}) or {}
        structure_issues = {
            issue_id: issue
            for issue_id, issue in issues.items()
            if self._is_detail_doc_structure_issue_id(issue_id)
        }
        if not structure_issues:
            return state.tree, marked_sections, None
        focus_single = os.getenv("STEP2_DOC_STRUCTURE_COMPACT_SINGLE_GROUP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if focus_single and len(structure_issues) > 1:
            try:
                group_limit = max(1, int(os.getenv("STEP2_DOC_STRUCTURE_COMPACT_GROUP_LIMIT", "4")))
            except Exception:
                group_limit = 4
            ranked_all = sorted(structure_issues.items(), key=self._rank_doc_structure_issue_for_compact_prompt)
            if ranked_all and self._rank_doc_structure_issue_for_compact_prompt(ranked_all[0])[0] == 0:
                best_items = [
                    item
                    for item in ranked_all
                    if self._rank_doc_structure_issue_for_compact_prompt(item)[0] == 0
                ][:group_limit]
                structure_issues = {issue_id: issue for issue_id, issue in best_items}
            elif ranked_all and self._doc_structure_issue_kind_for_compact_prompt(ranked_all[0]) in {
                "opaque_variant_expansion",
                "missing_scoped_field",
            }:
                top_kind = self._doc_structure_issue_kind_for_compact_prompt(ranked_all[0])
                best_items = [
                    item
                    for item in ranked_all
                    if self._doc_structure_issue_kind_for_compact_prompt(item) == top_kind
                ][:group_limit]
                structure_issues = {issue_id: issue for issue_id, issue in best_items}
            else:
                groups: Dict[int, List[Tuple[str, Issue]]] = {}
                fallback: List[Tuple[str, Issue]] = []
                for issue_id, issue in ranked_all:
                    repair = self._doc_structure_safe_repair(self._doc_structure_issue_metadata(issue))
                    variant_id = self._doc_structure_repair_variant_id(repair)
                    if variant_id is None:
                        fallback.append((issue_id, issue))
                        continue
                    groups.setdefault(variant_id, []).append((issue_id, issue))
                if groups:
                    _variant_id, best_items = sorted(
                        groups.items(),
                        key=lambda item: (-len(item[1]), self._rank_doc_structure_issue_for_compact_prompt(item[1][0]), item[0]),
                    )[0]
                    structure_issues = {issue_id: issue for issue_id, issue in best_items[:group_limit]}
                elif fallback:
                    structure_issues = {fallback[0][0]: fallback[0][1]}

        tree = normalize_protocol_tree(copy.deepcopy(state.tree))
        nodes = [node for node in tree.get("nodes", []) or [] if isinstance(node, dict)]
        edges = [edge for edge in tree.get("edges", []) or [] if isinstance(edge, dict)]
        by_id: Dict[int, Dict[str, Any]] = {}
        children_by_parent: Dict[int, Set[int]] = {}
        for node in nodes:
            try:
                nid = int(node.get("node_id"))
            except Exception:
                continue
            by_id[nid] = node
            parent = node.get("parent_id")
            try:
                if parent is not None:
                    children_by_parent.setdefault(int(parent), set()).add(nid)
            except Exception:
                pass
            for child in node.get("children_ids") or []:
                try:
                    children_by_parent.setdefault(nid, set()).add(int(child))
                except Exception:
                    pass

        seed_ids: Set[int] = set()
        evidence_lines: List[str] = []
        for issue_id, issue in structure_issues.items():
            desc = str(getattr(issue, "description", "") or "")
            metadata = self._doc_structure_issue_metadata(issue)
            repair = self._doc_structure_safe_repair(metadata)
            evidence_lines.append(f"- {issue_id}: {desc}")
            evidence_lines.append(
                "  safe_repair: "
                + json.dumps(repair, ensure_ascii=True, sort_keys=True)
            )
            for key in ("variant_id", "existing_node_id"):
                try:
                    value = repair.get(key)
                    if value is not None:
                        seed_ids.add(int(value))
                except Exception:
                    pass
            seed_ids.update(self._doc_structure_selector_seed_ids(repair, by_id))
            seed_ids.update(self._issue_description_node_ids(desc))

        if not seed_ids:
            return state.tree, marked_sections, None

        include_ids: Set[int] = set(seed_ids)
        frontier = set(seed_ids)
        radius = max(0, int(os.getenv("STEP2_DOC_STRUCTURE_COMPACT_RADIUS", "1")))
        for _ in range(radius):
            next_frontier: Set[int] = set()
            for nid in frontier:
                node = by_id.get(nid)
                if node:
                    try:
                        parent = node.get("parent_id")
                        if parent is not None:
                            next_frontier.add(int(parent))
                    except Exception:
                        pass
                    next_frontier.update(children_by_parent.get(nid, set()))
                for edge in edges:
                    try:
                        src = int(edge.get("src"))
                        dst = int(edge.get("dst"))
                    except Exception:
                        continue
                    if src == nid or dst == nid:
                        next_frontier.add(src)
                        next_frontier.add(dst)
            next_frontier -= include_ids
            include_ids.update(next_frontier)
            frontier = next_frontier

        root_id = tree.get("root_node_id")
        try:
            if root_id is not None:
                include_ids.add(int(root_id))
        except Exception:
            pass

        compact_nodes = [copy.deepcopy(node) for node in nodes if int(node.get("node_id", -1)) in include_ids]
        compact_edges = []
        for edge in edges:
            try:
                src = int(edge.get("src"))
                dst = int(edge.get("dst"))
            except Exception:
                continue
            if src in include_ids and dst in include_ids:
                compact_edges.append(copy.deepcopy(edge))

        doc_cluster_contracts = self._selector_coded_doc_cluster_contracts(marked_sections)
        compact_tree = {
            "protocol_name": tree.get("protocol_name"),
            "root_node_id": tree.get("root_node_id"),
            "nodes": compact_nodes,
            "edges": compact_edges,
            "_prompt_context": {
                "type": "doc_structure_compact",
                "note": "Partial prompt-only tree view. Node IDs are real; apply patches to the full tree by ID.",
                "full_node_count": len(nodes),
                "full_edge_count": len(edges),
                "next_available_node_id": max(by_id.keys(), default=0) + 1,
                "shown_node_ids": sorted(include_ids),
                "seed_node_ids": sorted(seed_ids),
                "focused_issue_ids": sorted(str(issue_id) for issue_id in structure_issues),
                "doc_cluster_contracts": doc_cluster_contracts,
                "patch_skeleton": self._doc_structure_patch_skeleton(
                    structure_issues=structure_issues,
                    by_id=by_id,
                    edges=edges,
                    doc_cluster_contracts=doc_cluster_contracts,
                ),
            },
        }
        selected_kinds = {
            str(
                self._doc_structure_safe_repair(self._doc_structure_issue_metadata(issue)).get("kind")
                or ""
            )
            for issue in structure_issues.values()
        }
        compact_sections = [
            {
                "title": "Documentation structure evidence",
                "content": "\n".join(evidence_lines),
            }
        ]
        compact_sections.extend(self._doc_structure_compact_section_evidence(marked_sections))
        feedback = (
            "DOC STRUCTURE COMPACT CONTEXT: the CURRENT TREE block is a prompt-only neighborhood, not the full tree. "
            "Patch only real node IDs shown in this compact context, or add new child fields/variants under shown parent IDs. "
            "For any new_nodes, use node_id >= _prompt_context.next_available_node_id because lower IDs may exist outside this compact view. "
            "For TLV abstraction issues, prefer the patch_skeleton edge_updates that reuse an existing generic TLV branch; "
            "do not create a parallel TLV_Option branch unless the skeleton says no reusable Len/Data branch exists."
        )
        if "missing_selector_variants" in selected_kinds:
            selector_coded_cluster_active = os.getenv(
                "STEP2_STAGEB_SELECTOR_CODED_CLUSTER",
                "",
            ).strip().lower() in {"1", "true", "yes", "on"}
            if selector_coded_cluster_active:
                feedback += (
                    " For selector-coded cluster issues, do not stop at zero-size selector shells. Add or reuse "
                    "condition_on variants for the focused selector values and add typed body fields when the "
                    "focused documentation lists them. Preserve the generic fallback/TLV branch, length binding, "
                    "and active train-packet anchor. If _prompt_context.doc_cluster_contracts is present, every "
                    "required_guard_value in that contract must be covered in the same patch; otherwise return an "
                    "empty patch instead of a partial repair."
                )
            else:
                feedback += (
                    " For selector-map issues, this pass is SELECTOR SHELL ONLY: add empty zero-size variant shells "
                    "with condition_on edges. If the selector sits inside a fixed/common header, place shells under "
                    "the header parent at the header end; otherwise use the selector's enclosing container. Do not "
                    "expand body fields."
                )
        if "opaque_variant_expansion" in selected_kinds or "missing_scoped_field" in selected_kinds:
            feedback += (
                " For body-expansion issues, refine the existing selected variant in place; do not create a new "
                "selector map or duplicate variant for the same selector value. Every new child node must have "
                "parent_id equal to the variant whose children_ids you update; do not update children_ids on an "
                "unfocused parent, and do not reference a child from two different parents. If the safe repair "
                "names opaque_child_ids and one of those children is an existing TLV/IE/option sequence, put the "
                "missing fields inside that sequence rather than replacing the selected message variant with new "
                "variable wrapper nodes. When missing_expected_fields is non-empty, add only those missing fields "
                "in this patch; expected_field_specs is context, not permission to duplicate already-covered fields."
            )
        return compact_tree, compact_sections, feedback

    @staticmethod
    def _filter_previous_attempt_for_focus(
        previous_attempt: Optional[Dict[str, Any]],
        focus_issue_ids: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if not previous_attempt or not focus_issue_ids:
            return previous_attempt
        focus = {str(issue_id) for issue_id in focus_issue_ids}
        filtered = copy.deepcopy(previous_attempt)
        pending = filtered.get("pending_issues")
        if isinstance(pending, list):
            filtered["pending_issues"] = [
                issue
                for issue in pending
                if isinstance(issue, dict) and str(issue.get("id")) in focus
            ]
        def _keep_line(line: Any) -> bool:
            text = str(line)
            return any(issue_id in text for issue_id in focus)
        for key in ("validator_errors", "validator_extras"):
            values = filtered.get(key)
            if isinstance(values, list):
                kept = [value for value in values if _keep_line(value)]
                # Keep the compact prompt focused. The exact evidence is also
                # included in the documentation block, so dropping unrelated
                # validator lines is safe.
                filtered[key] = kept
        return filtered

    def _generate_patch_via_llm(self, state: "TreeState", avoid_summaries: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        logger.info(f"DEBUG: BatchPatchSupplier agent type: {type(self.agent)}")
        attempt_number = self._next_attempt_number()
        marked_sections = summarize_sections_for_patch(
            self.sections,
            self.batch_start,
            self.batch_size,
            focus_only=True,
        )
        feedback_hint = self._merge_feedback_with_failures(self.feedback, state)
        previous_attempt_context = self._build_previous_attempt_context(state)
        prompt_tree = state.tree
        prompt_sections: Sequence[Dict[str, Any]] = marked_sections
        active_patch_skeleton: Optional[Dict[str, Any]] = None
        if self._should_use_doc_structure_compact_prompt(state):
            prompt_tree, prompt_sections, compact_feedback = self._doc_structure_compact_prompt_context(state, marked_sections)
            if compact_feedback:
                feedback_hint = (feedback_hint + "\n" if feedback_hint else "") + compact_feedback
                try:
                    context = prompt_tree.get("_prompt_context", {}) if isinstance(prompt_tree, dict) else {}
                    focus_ids = context.get("focused_issue_ids") or []
                    previous_attempt_context = self._filter_previous_attempt_for_focus(previous_attempt_context, focus_ids)
                    logger.info(
                        "Using doc structure compact prompt: nodes=%s/%s edges=%s/%s focus=%s seeds=%s",
                        len(prompt_tree.get("nodes", []) or []) if isinstance(prompt_tree, dict) else "?",
                        context.get("full_node_count"),
                        len(prompt_tree.get("edges", []) or []) if isinstance(prompt_tree, dict) else "?",
                        context.get("full_edge_count"),
                        focus_ids,
                        context.get("seed_node_ids"),
                    )
                except Exception:
                    pass
                context = prompt_tree.get("_prompt_context", {}) if isinstance(prompt_tree, dict) else {}
                patch_skeleton = context.get("patch_skeleton") if isinstance(context, dict) else None
                active_patch_skeleton = patch_skeleton if isinstance(patch_skeleton, dict) else None
                deterministic_enabled = os.getenv("STEP2_RELATION_DETERMINISTIC_REPAIR", "1").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                if (
                    deterministic_enabled
                    and isinstance(patch_skeleton, dict)
                    and isinstance(patch_skeleton.get("patch_metadata"), dict)
                    and patch_skeleton["patch_metadata"].get("deterministic_safe")
                    and self._patch_has_effective_actions(patch_skeleton)
                ):
                    patch = copy.deepcopy(patch_skeleton)
                    self._ensure_patch_defaults(patch)
                    attempt_number = self._next_attempt_number()
                    patch_hash = _hash_patch(patch)
                    summary = _summarize_patch_for_prompt(patch)
                    self._record_patch(
                        patch,
                        patch_hash,
                        summary,
                        "deterministic_structure",
                        attempt_number,
                        None,
                    )
                    self.action_info[patch_hash] = {
                        "summary": summary,
                        "source": "deterministic_structure",
                        "attempt": attempt_number,
                    }
                    self.previous_patch_summary = summary
                    self._bump_counters(llm_used=False)
                    return patch
        elif self._should_use_relation_compact_prompt(state):
            prompt_tree, prompt_sections, compact_feedback = self._relation_compact_prompt_context(state, marked_sections)
            if compact_feedback:
                feedback_hint = (feedback_hint + "\n" if feedback_hint else "") + compact_feedback
                try:
                    context = prompt_tree.get("_prompt_context", {}) if isinstance(prompt_tree, dict) else {}
                    focus_ids = context.get("focused_issue_ids") or []
                    previous_attempt_context = self._filter_previous_attempt_for_focus(previous_attempt_context, focus_ids)
                    logger.info(
                        "Using relation compact prompt: nodes=%s/%s edges=%s/%s focus=%s seeds=%s",
                        len(prompt_tree.get("nodes", []) or []) if isinstance(prompt_tree, dict) else "?",
                        context.get("full_node_count"),
                        len(prompt_tree.get("edges", []) or []) if isinstance(prompt_tree, dict) else "?",
                        context.get("full_edge_count"),
                        focus_ids,
                        context.get("seed_node_ids"),
                    )
                except Exception:
                    pass
                context = prompt_tree.get("_prompt_context", {}) if isinstance(prompt_tree, dict) else {}
                patch_skeleton = context.get("patch_skeleton") if isinstance(context, dict) else None
                deterministic_enabled = os.getenv("STEP2_RELATION_DETERMINISTIC_REPAIR", "1").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                if (
                    deterministic_enabled
                    and isinstance(patch_skeleton, dict)
                    and isinstance(patch_skeleton.get("patch_metadata"), dict)
                    and patch_skeleton["patch_metadata"].get("deterministic_safe")
                    and self._patch_has_effective_actions(patch_skeleton)
                ):
                    patch = copy.deepcopy(patch_skeleton)
                    self._ensure_patch_defaults(patch)
                    attempt_number = self._next_attempt_number()
                    patch_hash = _hash_patch(patch)
                    summary = _summarize_patch_for_prompt(patch)
                    self._record_patch(
                        patch,
                        patch_hash,
                        summary,
                        "deterministic_relation",
                        attempt_number,
                        None,
                    )
                    self.action_info[patch_hash] = {
                        "summary": summary,
                        "source": "deterministic_relation",
                        "attempt": attempt_number,
                    }
                    self.previous_patch_summary = summary
                    self._bump_counters(llm_used=False)
                    return patch
        elif self._should_use_doc_constraint_compact_prompt(state):
            prompt_tree, prompt_sections, compact_feedback = self._doc_constraint_compact_prompt_context(state, marked_sections)
            if compact_feedback:
                feedback_hint = (feedback_hint + "\n" if feedback_hint else "") + compact_feedback
                try:
                    context = prompt_tree.get("_prompt_context", {}) if isinstance(prompt_tree, dict) else {}
                    focus_ids = context.get("focused_issue_ids") or []
                    previous_attempt_context = self._filter_previous_attempt_for_focus(previous_attempt_context, focus_ids)
                    logger.info(
                        "Using doc constraint compact prompt: nodes=%s/%s edges=%s/%s focus=%s seeds=%s",
                        len(prompt_tree.get("nodes", []) or []) if isinstance(prompt_tree, dict) else "?",
                        context.get("full_node_count"),
                        len(prompt_tree.get("edges", []) or []) if isinstance(prompt_tree, dict) else "?",
                        context.get("full_edge_count"),
                        focus_ids,
                        context.get("seed_node_ids"),
                    )
                except Exception:
                    pass
                if (
                    isinstance(prompt_tree, dict)
                    and isinstance(prompt_tree.get("_prompt_context"), dict)
                    and prompt_tree["_prompt_context"].get("skip_llm_if_no_safe_repairs")
                    and os.getenv("STEP2_DOC_CONSTRAINT_LLM_ON_UNSAFE", "0").strip().lower()
                    not in {"1", "true", "yes", "on"}
                ):
                    logger.info(
                        "Skipping doc constraint LLM request: focused issues have no validator-safe local repairs; "
                        "returning no-op patch to preserve budget."
                    )
                    patch: Dict[str, Any] = {
                        "patch_metadata": {
                            "description": "No safe local doc-constraint repair available",
                            "intent": "Doc constraint coverage",
                            "source": "skipped_no_safe_repair",
                        },
                        "new_nodes": [],
                        "nodes_to_remove": [],
                        "node_updates": [],
                        "new_edges": [],
                        "edge_updates": [],
                        "edge_removes": [],
                        "validation_notes": [
                            "Focused doc-constraint issues require unavailable scope/structure; no LLM call was made."
                        ],
                    }
                    self._bump_counters(llm_used=True)
                    return patch
        messages = build_patch_refinement_message(
            prompt_tree,
            prompt_sections,
            self.batch_start,
            self.batch_size,
            feedback_hint,
            self.previous_patch_summary,
            previous_attempt_context,
            mode=self.prompt_mode,
            avoid_summaries=avoid_summaries,
            experience=self.fix_history,
        )

        logger.info(
            "Batch %s requesting patch via %s (attempt %s/%s)",
            self.batch_index + 1,
            self.agent.provider,
            attempt_number,
            self.max_calls,
        )

        try:
            base_temp = 0.4
            temp_increment = 0.05
            dynamic_temp = min(0.7, base_temp + (self.calls_made * temp_increment))
            # A profile can fix the candidate-patch temperature (the MCTS setting
            # is 0.6); unset keeps the 0.4 -> 0.7 ramp.
            fixed_patch_temp = os.getenv("STEP2_PATCH_TEMPERATURE")
            temperature_fix = float(fixed_patch_temp) if fixed_patch_temp else 0.6  # candidate patches at 0.6

            temperature_expand = float(os.getenv("STEP2_EXPAND_TEMPERATURE", "0.6"))
            temperature = temperature_fix if self.prompt_mode == "fix" else temperature_expand
            try:
                deepseek_patch_max_tokens = int(
                    os.getenv("STEP2_PATCH_MAX_TOKENS")
                    or os.getenv("DEEPSEEK_PATCH_MAX_TOKENS")
                    or "8000"
                )
            except Exception:
                deepseek_patch_max_tokens = 8000
            deepseek_patch_max_tokens = max(1024, deepseek_patch_max_tokens)
            deepseek_patch_max_tokens = max(deepseek_patch_max_tokens,
                                           getattr(self, '_repair_retry_token_floor', 0))

            if self.agent.provider == "openai":
                payload = {
                    "model": self.agent.default_model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                }
                if self.agent.default_model.startswith(("gpt-5", "o1", "o3", "o4")):
                    payload.update(max_completion_tokens=32_768, reasoning_effort="medium")
                else:
                    # OpenAI-compatible local models do not take GPT reasoning parameters.
                    payload.update(max_tokens=deepseek_patch_max_tokens, temperature=temperature)
            elif self.agent.provider == "deepseek":
                payload = {
                    "model": self.agent.default_model,
                    "messages": messages,
                    "max_tokens": deepseek_patch_max_tokens,
                    "temperature": temperature,
                    "stream": False,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
                response_format_raw = (
                    os.getenv("STEP2_PATCH_RESPONSE_FORMAT")
                    or os.getenv("DEEPSEEK_PATCH_RESPONSE_FORMAT")
                    or "json_object"
                )
                if response_format_raw.strip().lower() not in {"0", "false", "no", "none", "off"}:
                    payload["response_format"] = {"type": "json_object"}
            else:
                payload = {
                    "model": self.agent.default_model,
                    "system": messages[0]["content"],
                    "messages": [{"role": "user", "content": messages[1]["content"]}],
                    "max_tokens": 64_000,
                }

            fallback_model = os.getenv("STEP2_PATCH_FALLBACK_MODEL")
            empty_retry_limit = max(1, int(os.getenv("STEP2_EMPTY_RESPONSE_RETRIES", "2")))
            if os.getenv("STEP2_EMPTY_RESPONSE_RETRIES") is None:
                for budget_env in ("STEP2_MCTS_MAX_CALLS", "STEP2_MCTS_MAX_LLM_CALLS", "INLINE_FIX_MAX_CALLS"):
                    raw_budget = os.getenv(budget_env)
                    if raw_budget is None:
                        continue
                    try:
                        if int(raw_budget.strip()) <= 1:
                            empty_retry_limit = 1
                            break
                    except Exception:
                        continue
            quick_timeout = int(os.getenv("STEP2_EMPTY_RESPONSE_TIMEOUT", "45"))
            original_timeout = self.agent.read_timeout

            active_payload = dict(payload)
            clean_response: Optional[str] = None
            raw_response: Optional[str] = None

            try:
                for quick_attempt in range(empty_retry_limit):
                    if quick_attempt > 0 and quick_timeout > 0:
                        self.agent.read_timeout = min(original_timeout, quick_timeout)

                    result = self.agent._call_api_with_retry(active_payload)
                    logger.info(f"DEBUG: API Result keys: {list(result.keys())}")

                    choice0 = (result.get('choices') or [{}])[0]
                    finish = choice0.get('finish_reason') or result.get('stop_reason')
                    if finish in {'length', 'max_tokens'}:
                        current_tokens = int(active_payload.get('max_tokens') or 0)
                        cap = current_tokens
                        if self.agent.provider == 'deepseek':
                            cap_env = ('DEEPSEEK_FLASH_MAX_TOKENS_CAP'
                                       if 'flash' in str(active_payload.get('model', '')).lower()
                                       else 'DEEPSEEK_MAX_TOKENS_CAP')
                            try:
                                cap = max(current_tokens, int(os.getenv(cap_env, str(current_tokens))))
                            except (TypeError, ValueError):
                                pass
                        increased = min(cap, max(current_tokens + 1, current_tokens * 2))
                        if current_tokens and increased > current_tokens:
                            self._repair_retry_token_floor = increased
                            active_payload['max_tokens'] = increased
                            logger.warning('Repair response truncated; increasing configured output budget %s -> %s',
                                           current_tokens, increased)
                            if quick_attempt + 1 < empty_retry_limit:
                                continue
                        # Even syntactically valid partial output is not a completed
                        # patch. Never reinterpret reasoning_content as final JSON.
                        raise ValueError(f'Repair response truncated at {current_tokens} tokens')

                    if "content" in result and isinstance(result["content"], list):
                        first_item = result["content"][0]
                        if isinstance(first_item, dict):
                            raw_response = first_item.get("text", "") or first_item.get("content", "")
                        else:
                            raw_response = str(first_item)
                    elif "choices" in result:
                        choices = result["choices"]
                        if not choices:
                             raise ValueError("Empty choices in API response")
                        message = choices[0]["message"]
                        if isinstance(message, str):
                             logger.warning(f"DEBUG: message is string: {message}")
                             raw_response = message
                        else:
                            raw_response = message.get("content", "") or message.get("text", "")
                    else:
                        raise ValueError(f"Unexpected API response format: {list(result.keys())}")

                    clean_response = self.agent._clean_raw_response(raw_response or "")
                    if clean_response:
                        break

                    if quick_attempt == empty_retry_limit - 1:
                        try:
                            choice0 = (result.get("choices") or [{}])[0]
                            message0 = choice0.get("message") if isinstance(choice0, dict) else None
                            logger.warning(
                                "Empty response diagnostics: finish_reason=%s message_keys=%s usage=%s",
                                choice0.get("finish_reason") if isinstance(choice0, dict) else None,
                                sorted(message0.keys()) if isinstance(message0, dict) else type(message0).__name__,
                                result.get("usage"),
                            )
                        except Exception:
                            pass
                        raise ValueError("LLM returned empty patch response")

                    logger.warning(
                        "Empty response for batch %s attempt %s; retrying immediately (quick retry %s/%s)",
                        self.batch_index + 1,
                        attempt_number,
                        quick_attempt + 1,
                        empty_retry_limit - 1,
                    )

                    if fallback_model and active_payload.get("model") != fallback_model:
                        active_payload = dict(active_payload)
                        active_payload["model"] = fallback_model
                        logger.info(
                            "Switching to fallback model %s for batch %s due to empty response",
                            fallback_model,
                            self.batch_index + 1,
                        )
                else:  # pragma: no cover
                    raise ValueError("LLM failed to provide patch response")
            finally:
                self.agent.read_timeout = original_timeout

            if not clean_response:
                raise ValueError("Received empty patch response after retries")

            logger.info(f"DEBUG: LLM Response Patch: {clean_response}")

            patch = json.loads(clean_response)
            if not isinstance(patch, dict):
                raise ValueError("LLM response did not decode to an object")

            self._ensure_patch_defaults(patch)
            patch = self._merge_referenced_patch_skeleton_dependencies(
                state.tree,
                patch,
                active_patch_skeleton,
            )
            self._ensure_patch_defaults(patch)
            meta = patch.get("patch_metadata")
            if not isinstance(meta, dict):
                meta = {}
                patch["patch_metadata"] = meta
            # Make patch provenance visible to the MCTS logger (and downstream analysis).
            meta["source"] = "llm"
            patch = _strip_payload_fill_child_link_updates(state.tree, patch)
            # Optionally split a big patch into atomic patches to encourage
            # deeper MCTS exploration. Controlled by env:
            #   STEP2_PATCH_ATOMIC_SPLIT=1 (enable)
            #   STEP2_PATCH_MAX_CHANGES=N (bucket size, default 1)
            self._maybe_enqueue_atomic_splits(patch)
            summary = _summarize_patch_for_prompt(patch)
            patch_hash = _hash_patch(patch)

            truncated_messages = [
                {"role": msg.get("role", ""), "content": msg.get("content", "")[:500]}
                for msg in messages
            ]

            self._record_patch(patch, patch_hash, summary, "llm", attempt_number, truncated_messages)
            self.action_info[patch_hash] = {
                "summary": summary,
                "source": "llm",
                "attempt": attempt_number,
            }
            self.previous_patch_summary = summary
            self._bump_counters(llm_used=True)

            return patch

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse patch JSON on batch %s attempt %s: %s", self.batch_index + 1, attempt_number, exc)
        except Exception as exc:
            logger.error("Patch request failed on batch %s attempt %s: %s", self.batch_index + 1, attempt_number, exc)

        self._bump_counters(llm_used=True)
        return None

    def _count_patch_changes(self, patch: Dict[str, Any]) -> int:
        total = 0
        for key in (
            "new_nodes",
            "node_updates",
            "new_edges",
            "edge_updates",
            "edge_removes",
            "nodes_to_remove",
        ):
            items = patch.get(key)
            if isinstance(items, list):
                total += len(items)
        return total

    def _maybe_enqueue_atomic_splits(self, patch: Dict[str, Any]) -> None:
        try:
            from os import getenv
            enabled = getenv("STEP2_PATCH_ATOMIC_SPLIT", "0").lower() in {"1", "true"}
            if not enabled:
                return
            limit = int(getenv("STEP2_PATCH_MAX_CHANGES", "1"))
            if limit < 1:
                limit = 1
        except Exception:
            return

        total = self._count_patch_changes(patch)
        if total <= limit:
            return

        # Preserve optional metadata
        meta_keys = {"patch_metadata"}
        shared: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in patch.items() if k in meta_keys}

        # Flatten to atomic units, ordered to remove unsafe edges/nodes first
        units: List[Tuple[str, Any]] = []
        for key in (
            "edge_removes",
            "nodes_to_remove",
            "edge_updates",
            "new_edges",
            "node_updates",
            "new_nodes",
        ):
            items = patch.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                units.append((key, copy.deepcopy(item)))

        # Bucket into patches of size 'limit'
        bucket: List[Tuple[str, Any]] = []
        atomic_patches: List[Dict[str, Any]] = []
        for entry in units:
            bucket.append(entry)
            if len(bucket) >= limit:
                atomic_patches.append(self._build_atomic_patch(shared, bucket))
                bucket = []
        if bucket:
            atomic_patches.append(self._build_atomic_patch(shared, bucket))

        # Enqueue for subsequent calls
        self._atomic_queue.extend(atomic_patches)

    def _build_atomic_patch(self, shared: Dict[str, Any], entries: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        atomic: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in shared.items()}
        # Initialize empty lists to keep schema consistent
        for key in (
            "new_nodes",
            "node_updates",
            "new_edges",
            "edge_updates",
            "edge_removes",
            "nodes_to_remove",
        ):
            atomic[key] = []
        for key, item in entries:
            atomic[key].append(copy.deepcopy(item))
        return atomic

    def _derive_followup_tasks(self, state: "TreeState") -> List[str]:
        tasks: List[str] = []
        task_set: set[str] = set()

        def add_task(text: str) -> None:
            cleaned = text.strip()
            if cleaned and cleaned not in task_set:
                task_set.add(cleaned)
                tasks.append(cleaned)
        last_action = getattr(state, "last_action", None)
        patch = last_action.patch if last_action is not None else None
        if not isinstance(patch, dict):
            return tasks

        tree = state.tree
        nodes_index = {node.get('node_id'): node for node in tree.get('nodes', []) if isinstance(node, dict)}

        def node_label(node_dict: Dict[str, Any]) -> str:
            node_id = node_dict.get('node_id')
            name = node_dict.get('name', f'node_{node_id}')
            return f"{name}(ID:{node_id})"

        new_nodes = patch.get('new_nodes', []) or []
        for entry in new_nodes:
            if not isinstance(entry, dict):
                continue
            label = node_label(entry)
            parent_id = entry.get('parent_id')
            parent = nodes_index.get(parent_id)
            parent_label = node_label(parent) if parent else f'node_{parent_id}'
            node_type = (entry.get('node_type') or '').lower()
            if node_type == 'selector':
                add_task(f"Define condition_on edges for selector {label} and ensure each branch is represented by variants.")
            else:
                add_task(
                    f"Review newly added {label} under {parent_label}: add required condition_on/length_of edges and adjust sibling bit_start expressions so ranges do not overlap."
                )

        for update in patch.get('node_updates', []) or []:
            if not isinstance(update, dict):
                continue
            changes = update.get('changes') or {}
            if 'children_ids' in changes:
                parent_id = update.get('node_id')
                parent = nodes_index.get(parent_id)
                parent_label = node_label(parent) if parent else f'node_{parent_id}'
                add_task(
                    f"Parent {parent_label} had children_ids rewritten; ensure every listed child has the correct structural edges and ordering."
                )

        # Global tree scans (protocol-agnostic)
        nodes = tree.get('nodes', []) or []
        edges = tree.get('edges', []) or []

        condition_edges: Dict[str, List[Dict[str, Any]]] = {}
        length_edge_rels = {"length_of", "repeat_count", "offset_of"}
        length_edges: Dict[str, List[Dict[str, Any]]] = {}

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = edge.get('src')
            rel = edge.get('rel')
            if src is None:
                continue
            key = str(src)
            if rel == 'condition_on':
                condition_edges.setdefault(key, []).append(edge)
            if rel in length_edge_rels:
                length_edges.setdefault(key, []).append(edge)

        def parse_enum_values(node: Dict[str, Any]) -> List[str]:
            enums: List[str] = []
            for constraint in node.get('constraints', []) or []:
                if not isinstance(constraint, str):
                    continue
                lowered = constraint.lower()
                if lowered.startswith('enum:'):
                    payload = constraint.split(':', 1)[1]
                    enums = [item.strip() for item in payload.split('|') if item.strip()]
                    break
            return enums

        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get('node_id')
            label = node_label(node)
            node_type = (node.get('node_type') or '').lower()

            if node_type == 'selector':
                cond = condition_edges.get(str(node_id)) or []
                if not cond:
                    add_task(f"Selector {label} lacks condition_on edges; add variants and boolean condition_on mappings covering its value constraints.")
                enums = parse_enum_values(node)
                if enums:
                    formatted = ', '.join(enums[:6]) + ('…' if len(enums) > 6 else '')
                    add_task(f"Ensure selector {label} has variants/condition_on edges for enumerated values ({formatted}).")

            def looks_like_length(n: Dict[str, Any]) -> bool:
                name = (n.get('name') or '').lower()
                keywords = ('length', 'count', 'size', 'number', 'total', 'cnt')
                return any(keyword in name for keyword in keywords)

            if looks_like_length(node):
                token = f"val({node_id})"
                referenced = False
                for other in nodes:
                    if not isinstance(other, dict):
                        continue
                    for key in ('size_bits', 'bit_start'):
                        expr = other.get(key)
                        if isinstance(expr, str) and token in expr:
                            referenced = True
                            break
                    if referenced:
                        break
                    for constraint in other.get('constraints') or []:
                        if isinstance(constraint, str) and token in constraint:
                            referenced = True
                            break
                if not referenced:
                    for edge in edges:
                        if not isinstance(edge, dict):
                            continue
                        formula = edge.get('formula')
                        if isinstance(formula, str) and token in formula:
                            referenced = True
                            break
                if length_edges.get(str(node_id)) or referenced:
                    add_task(f"Connect length/count field {label} to dependent payloads using length_of/repeat_count/offset_of edges.")

            if node_type not in {'selector', 'variant'}:
                enums = parse_enum_values(node)
                if enums:
                    add_task(f"Field {label} defines enumerated values ({', '.join(enums[:6])}{'…' if len(enums) > 6 else ''}); consider modelling dedicated variants or gating edges for these cases.")

        def resolve_label(node_id: Optional[int], text: str = "") -> str:
            if node_id is not None:
                node = nodes_index.get(node_id)
                if node:
                    return node_label(node)
            match = re.search(r"([A-Za-z0-9_]+\(ID:\d+\))", text)
            if match:
                return match.group(1)
            return "the affected node"

        def apply_issue_templates(text: str, *, node_id: Optional[int] = None) -> None:
            if not text:
                return
            lowered = text.lower()
            label = resolve_label(node_id, text)

            if "multiple root nodes" in lowered:
                add_task("Ensure the protocol has exactly one root node; reparent or remove extra roots and update children_ids accordingly.")
            if "no root node found" in lowered:
                add_task("Designate a single root node and attach any previously rootless nodes beneath it, keeping parent_id/children_ids consistent.")
            if "parent node" in lowered and "does not exist" in lowered:
                add_task(f"Attach {label} to a valid parent and update both parent_id and children_ids so the relationship is consistent.")
            if "child" in lowered and "does not exist" in lowered:
                add_task(f"Either create the missing child referenced by {label} or remove it from children_ids to keep the tree consistent.")
            if "not listed in parent's children_ids" in lowered:
                add_task(f"Update {label}'s parent children_ids to include it, or adjust parent_id so the hierarchy stays consistent.")
            if "missing required field" in lowered:
                add_task(f"Populate the required fields (name/node_type/message_type/bit_start/size_bits) for {label} so it is fully defined.")
            if "size_bits references" in lowered or "bit_start references" in lowered:
                add_task(f"Rewrite the expressions on {label} so they only reference existing sibling nodes or literal values.")
            if "children may exceed parent size" in lowered:
                add_task(f"Recalculate bit_start and size_bits for each child under {label}, adjusting the parent size_bits so the layout fits without overflow.")
            if "invalid overlap with sibling" in lowered:
                add_task(
                    f"Check whether {label} and the sibling should be mutually-exclusive variants under ONE shared selector. "
                    "Prefer merging duplicate/shadow selectors and adding selector->variant condition_on edges; do not fix true alternatives by merely shifting bit_start."
                )
            if "expression references sibling" in lowered:
                add_task(f"Duplicate shared data into {label}'s branch or change its expressions so they reference nodes within the same selector path instead of sibling branches.")
            if "non-exclusive conditions" in lowered:
                add_task(
                    f"Provide mutually exclusive condition_on formulas from a single shared selector so {label} only activates when that selector value matches the intended branch."
                )
            if "variable-length field lacks explicit length binding" in lowered:
                add_task(f"Add explicit length_of edges (and matching formulas) so {label} ties clearly to the associated length/count field.")
            if "formula constraint cannot be satisfied" in lowered:
                add_task(f"Rewrite the constraint on {label} so it matches the documented range and can be satisfied alongside existing selector/domain limits.")
            if "documentation constraint appears uncovered" in lowered:
                add_task(
                    f"Add the documented constraint for {label} using the existing tree schema: prefer a node-level enum/range when globally true, "
                    "or a scoped variant `formula:` / condition_on branch when the documentation makes it mode-dependent. Keep the repair grounded "
                    "in the provided section text, not protocol-specific assumptions."
                )
            if "scoped repair" in lowered and "unconditional constraint" in lowered:
                add_task(
                    "This doc claim is scoped by context. Do not add or re-add a global enum/range constraint to a shared fixed-header field; "
                    "only encode it under an existing matching variant/message scope, otherwise leave the claim unresolved."
                )
            if "documentation structure appears uncovered" in lowered:
                add_task(
                    "If several documentation-structure issues share the same guard/selector field, repair them as one coherent selector-table group: "
                    "reuse the existing selector, add the visible missing guarded variants/fields from the pending issues, and update parent children_ids "
                    "and condition_on edges consistently. Do not stop after the first guarded value when the prompt lists multiple sibling gaps."
                )
            if "has no condition_on variant for" in lowered:
                add_task(
                    "For missing guarded variants, keep the selector as a leaf under its enclosing container; add sibling variant containers for the listed "
                    "guard value(s), include the documented body fields, set each variant size_bits to cover its children, and add mutually exclusive "
                    "condition_on edges from the shared selector."
                )
            if "unable to interpret formula constraint" in lowered or "malformed formula constraint" in lowered:
                add_task(f"Restate the constraint on {label} using supported arithmetic or If(...) syntax so the validator can evaluate it.")
            if "formula constraint references" in lowered and "should only depend on \"value\"" in lowered:
                add_task(f"Rewrite or remove the constraint on {label} so it only references 'value'. Use length_of edges or size_bits formulas to relate to other fields.")
            if "formula may underflow" in lowered or "formula may overflow" in lowered:
                add_task(f"Clamp or reformulate the constraint on {label} so its value stays within the data type bounds reported by the validator.")
            if "multiple variants share identical condition" in lowered:
                add_task(f"Give each variant under the relevant selector unique condition_on formulas so the branches cannot trigger simultaneously.")
            if "lacks activation conditions" in lowered:
                add_task(f"Add condition_on edges for the selector controlling {label}, covering every documented request/response variant value.")
            if "edge source" in lowered and "does not exist" in lowered:
                add_task("Remove or fix edges whose source node is missing so every graph edge connects valid nodes.")
            if "edge destination" in lowered and "does not exist" in lowered:
                add_task("Repair or delete edges that target missing nodes to keep the edge graph consistent.")
            if "node is not reachable from any root" in lowered:
                add_task(f"Either attach {label} to a reachable parent or remove it if the documentation no longer references it.")
            if "part of circular dependency" in lowered:
                add_task(f"Break the circular dependency involving {label} by rewriting expressions so they no longer reference each other recursively.")
            if "variant condition for" in lowered and "cannot be satisfied" in lowered:
                add_task(f"Adjust condition_on logic so the activation constraint for {label} is satisfiable within the selector's documented range.")
            # Suppress coverage guidance during fixing: do not auto-suggest adding request/response coverage.
            # This avoids introducing symmetric edges not grounded in documentation (e.g., exception codes on request side).
            # if "coverage" in lowered and "only mapped to request" in lowered:
            #     add_task(f"Add response or exception variants for the selector values highlighted in the coverage warning so both directions are covered.")
            # if "coverage" in lowered and "only mapped to response" in lowered:
            #     add_task(f"Add request-side handling for the selector values mentioned so request coverage matches the documentation.")
            # if "no covered values" in lowered and "coverage matrix" in lowered:
            #     add_task("Model variants and condition_on edges for the selector so the documented codes or values are represented.")

        for issue in (state.pending_issues or {}).values():
            if not isinstance(issue, Issue):
                continue
            target_id: Optional[int] = None
            if issue.target and issue.target.kind == "node":
                try:
                    target_id = int(issue.target.identifier) if issue.target.identifier is not None else None
                except (TypeError, ValueError):
                    target_id = None
            apply_issue_templates(issue.description, node_id=target_id)

        for raw_error in state.validation.errors:
            apply_issue_templates(raw_error)

        for extra in state.validation.extras:
            apply_issue_templates(extra)

        return tasks

    def _build_previous_attempt_context(self, state: "TreeState") -> Optional[Dict[str, Any]]:
        context: Dict[str, Any] = {}

        if state.last_action is not None:
            context["raw_patch"] = copy.deepcopy(state.last_action.patch)
            context["last_patch_hash"] = state.last_action.hash
            if state.history:
                last_record = state.history[-1]
                context["last_reward"] = last_record.reward
                if last_record.introduced:
                    context["issues_introduced"] = self._summarize_issues(last_record.introduced)
                if last_record.resolved:
                    context["issues_resolved"] = self._summarize_issues(last_record.resolved)

        if state.validation.errors:
            context["validator_errors"] = list(state.validation.errors)

        if state.validation.extras:
            context["validator_extras"] = list(state.validation.extras)

        if state.pending_issues:
            context["pending_issues"] = self._summarize_issues(state.pending_issues)

        # In strict validator loop mode, we do not add any heuristic follow-up
        # tasks derived from tree scans or patch side effects. The LLM must be
        # guided only by validator output + the current tree.
        if not _strict_validator_loop_enabled():
            followup_tasks = self._derive_followup_tasks(state)
            if followup_tasks:
                context["followup_tasks"] = followup_tasks

        if state.history:
            recent: List[Dict[str, Any]] = []
            for record in reversed(state.history[-3:]):
                recent.append(
                    {
                        "hash": record.action.hash,
                        "summary": record.action.summary,
                        "reward": record.reward,
                        "introduced": self._summarize_issues(record.introduced) if record.introduced else [],
                        "resolved": self._summarize_issues(record.resolved) if record.resolved else [],
                        "timestamp": record.created_at,
                    }
                )
            context["recent_records"] = list(recent)

        if not context:
            return None

        return context

    def _record_patch(
        self,
        patch: Dict[str, Any],
        patch_hash: str,
        summary: Optional[str],
        source: str,
        attempt_number: int,
        messages: Optional[Sequence[Dict[str, Any]]],
    ) -> None:
        sequence_index = len(self.generated_log) + 1

        locations = self._summarize_patch_locations(patch)

        record: Dict[str, Any] = {
            "patch_snapshot": copy.deepcopy(patch),
            "patch_hash": patch_hash,
            "summary": summary,
            "source": source,
            "attempt": attempt_number,
            "timestamp": datetime.now().isoformat(),
            "sequence_index": sequence_index,
            "batch_index": self.batch_index,
            "locations": locations,
        }
        if messages:
            record["messages"] = list(messages)
        self.generated_log.append(record)
        self.patch_files[patch_hash] = self._write_patch_record(record)

    def on_candidate_evaluated(self, evaluation: "PatchEvaluation") -> None:
        info = self.action_info.get(evaluation.action.hash, {})
        entry = {
            "batch": self.batch_index,
            "source": info.get("source", "llm"),
            "attempt": info.get("attempt"),
            "reward": evaluation.reward,
            "errors": list(evaluation.validation_after.errors),
            "extras": list(evaluation.validation_after.extras),
            "patch_summary": info.get("summary"),
            "issues_after": self._summarize_issues(evaluation.validation_after.issues)
            if evaluation.validation_after.issues
            else [],
            "introduced": self._summarize_issues(evaluation.delta.introduced)
            if evaluation.delta.introduced
            else [],
            "resolved": self._summarize_issues(evaluation.delta.resolved)
            if evaluation.delta.resolved
            else [],
        }
        self.evaluation_records.append(entry)

        for record in self.generated_log:
            if record.get("patch_hash") == evaluation.action.hash:
                record["errors"] = list(evaluation.validation_after.errors)
                record["extras"] = list(evaluation.validation_after.extras)
                record["reward"] = evaluation.reward
                if evaluation.delta.introduced:
                    record["issues_introduced"] = self._summarize_issues(evaluation.delta.introduced)
                if evaluation.delta.resolved:
                    record["issues_resolved"] = self._summarize_issues(evaluation.delta.resolved)
                path = self.patch_files.get(evaluation.action.hash)
                if path is None:
                    path = self._write_patch_record(record)
                    self.patch_files[evaluation.action.hash] = path
                else:
                    self._write_patch_record(record, path)
                break

        self.feedback = _build_patch_feedback(
            list(evaluation.validation_after.errors),
            list(evaluation.validation_after.extras),
            evaluation.tree_after,
        )
        if info.get("summary"):
            self.previous_patch_summary = info["summary"]

    def persist_cache(self) -> None:
        if not self.generated_log:
            return

        payload = {
            "analysis_only": True,
            "notice": "Cached records are stored for offline review only. The refinement loop ignores these entries during future runs.",
            "patches": self.generated_log,
            "timestamp": datetime.now().isoformat(),
            "batch_info": f"Sections {self.batch_start} to {self.batch_end - 1}",
        }
        self.agent._save_to_cache(self.cache_filename, payload)

    def get_action_info(self, action_hash: Optional[str]) -> Optional[Dict[str, Any]]:
        if not action_hash:
            return None
        return self.action_info.get(action_hash)

    def _write_patch_record(self, record: Dict[str, Any], destination: Optional[Path] = None) -> Path:
        patch_hash = record.get("patch_hash") or "unknown"
        if destination is None:
            attempt = record.get("attempt") or 0
            filename = f"{attempt:03d}_{patch_hash[:8]}.json"
            destination = self.patch_dir / filename

        snapshot = copy.deepcopy(record)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
        return destination

    @staticmethod
    def _ensure_patch_defaults(patch: Dict[str, Any]) -> None:
        for key in [
            "new_nodes",
            "node_updates",
            "new_edges",
            "edge_updates",
            "edge_removes",
            "nodes_to_remove",
        ]:
            if key not in patch or not isinstance(patch[key], list):
                patch[key] = []

    @staticmethod
    def _summarize_patch_locations(patch: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        def _node_info(entry: Dict[str, Any]) -> Dict[str, Any]:
            info: Dict[str, Any] = {}
            node_id = entry.get("node_id")
            if node_id is not None:
                info["node_id"] = node_id
            name = entry.get("name")
            if name:
                info["name"] = name
            parent_id = entry.get("parent_id")
            if parent_id is not None:
                info["parent_id"] = parent_id
            return info

        new_nodes: List[Dict[str, Any]] = []
        for entry in patch.get("new_nodes", []) or []:
            if isinstance(entry, dict):
                new_nodes.append(_node_info(entry))
        if new_nodes:
            summary["new_nodes"] = new_nodes

        updated_nodes: List[Dict[str, Any]] = []
        for entry in patch.get("node_updates", []) or []:
            if not isinstance(entry, dict):
                continue
            info = _node_info(entry)
            if not info and entry.get("node_id") is not None:
                info["node_id"] = entry.get("node_id")
            if entry.get("updates"):
                info["fields"] = sorted(entry["updates"].keys())
            else:
                fields = [key for key in entry.keys() if key not in {"node_id", "name", "parent_id"}]
                if fields:
                    info["fields"] = sorted(fields)
            if info:
                updated_nodes.append(info)
        if updated_nodes:
            summary["updated_nodes"] = updated_nodes

        new_edges: List[Dict[str, Any]] = []
        for entry in _patch_new_edges(patch):
            if not isinstance(entry, dict):
                continue
            info = {key: entry.get(key) for key in ("src", "dst", "rel") if entry.get(key) is not None}
            if entry.get("formula"):
                info["formula"] = entry["formula"]
            if info:
                new_edges.append(info)
        if new_edges:
            summary["new_edges"] = new_edges

        edge_updates: List[Dict[str, Any]] = []
        for entry in patch.get("edge_updates", []) or []:
            if not isinstance(entry, dict):
                continue
            info = {key: entry.get(key) for key in ("src", "dst", "rel") if entry.get(key) is not None}
            if entry.get("updates") and isinstance(entry["updates"], dict):
                info["fields"] = sorted(entry["updates"].keys())
            if info:
                edge_updates.append(info)
        if edge_updates:
            summary["edge_updates"] = edge_updates

        edge_removes: List[Dict[str, Any]] = []
        for entry in patch.get("edge_removes", []) or []:
            if not isinstance(entry, dict):
                continue
            info = {key: entry.get(key) for key in ("src", "dst", "rel") if entry.get(key) is not None}
            if info:
                edge_removes.append(info)
        if edge_removes:
            summary["edge_removes"] = edge_removes

        removed_nodes: List[int] = []
        for entry in patch.get("nodes_to_remove", []) or []:
            if isinstance(entry, dict):
                node_id = entry.get("node_id")
            else:
                node_id = entry
            if isinstance(node_id, int):
                removed_nodes.append(node_id)
        if removed_nodes:
            summary["nodes_to_remove"] = sorted(set(removed_nodes))

        touched = set()
        for group in (summary.get("new_nodes", []), summary.get("updated_nodes", [])):
            if not group:
                continue
            for item in group:
                node_id = item.get("node_id")
                if isinstance(node_id, int):
                    touched.add(node_id)
        for node_id in summary.get("nodes_to_remove", []) or []:
            touched.add(node_id)

        if touched:
            summary["touched_node_ids"] = sorted(touched)

        return summary

    @staticmethod
    def _summarize_issues(issues: Dict[str, Issue]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for issue_id, issue in sorted(issues.items()):
            target = issue.target.identifier if issue.target else None
            metadata = getattr(issue.target, "metadata", None) if issue.target else None
            summary.append(
                {
                    "id": issue_id,
                    "type": issue.type.value,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "target": target,
                    "target_kind": issue.target.kind if issue.target else None,
                    "metadata": copy.deepcopy(metadata) if isinstance(metadata, dict) else None,
                }
            )
        return summary

    def _merge_feedback_with_failures(self, base_feedback: Optional[str], state: Any) -> Optional[str]:
        """Attach structured traffic failures to the feedback so the LLM sees them."""
        validation = getattr(state, "validation", None)
        if validation is None:
            return base_feedback
        failures = getattr(validation, "traffic_failures", ()) or ()
        if not failures:
            return base_feedback

        summary = getattr(validation, "traffic_global_summary", None)
        total_samples = getattr(validation, "traffic_total_samples", 0)
        success_samples = getattr(validation, "traffic_successful_samples", 0)
        failed_samples = total_samples - success_samples if total_samples else getattr(summary, "failed_samples", 0) or len(failures)

        # Strict closed-loop mode: do not add any heuristic "analysis" beyond
        # emitting raw, structured validator evidence. This keeps the LLM
        # guidance exclusively validator-driven.
        if _strict_validator_loop_enabled():
            max_groups = max(1, int(os.getenv("STEP2_TRAFFIC_FAILURES_IN_PROMPT", "8")))
            examples_per_group = max(1, int(os.getenv("STEP2_TRAFFIC_GROUP_EXAMPLES", "2")))
            prefix_len = max(1, int(os.getenv("STEP2_TRAFFIC_GROUP_PREFIX_LEN", "16")))

            def _env_limit(name: str, default: Optional[int], *, min_value: int = 1) -> Optional[int]:
                raw = os.getenv(name)
                if raw is None:
                    return default
                s = str(raw).strip().lower()
                if s in {"inf", "infty", "infinite", "unlimited", "none", "null", "no_limit", "nolimit"}:
                    return None
                try:
                    val = int(s)
                except Exception:
                    return default
                if val <= 0:
                    return None
                return max(min_value, val)

            hex_max_bytes = _env_limit("STEP2_TRAFFIC_PACKET_HEX_MAX_BYTES", 128, min_value=1)
            max_failures_per_sample = _env_limit("STEP2_TRAFFIC_GROUP_SAMPLE_FAILURES", 4, min_value=1)
            max_variant_errors = max(1, int(os.getenv("STEP2_TRAFFIC_ROUTING_VARIANT_ERRORS", "6")))

            def _coerce_int(raw: Any) -> Optional[int]:
                try:
                    return int(raw)
                except Exception:
                    return None

            def _path_prefix_tuple(f: Any) -> Tuple[int, ...]:
                raw = getattr(f, "path_node_ids", None) or []
                out: List[int] = []
                for nid in raw[:prefix_len]:
                    coerced = _coerce_int(nid)
                    if coerced is None:
                        continue
                    out.append(coerced)
                return tuple(out)

            def _infer_group_signature(f: Any) -> str:
                sig = getattr(f, "group_signature", None)
                if sig:
                    sig_str = str(sig)
                    if len(sig_str) > 200:
                        sig_str = sig_str[:197] + "..."
                    return sig_str

                selector_id = _coerce_int(getattr(f, "routing_selector_id", None))
                candidates_raw = getattr(f, "routing_candidate_variant_ids", None) or []
                candidates: List[int] = []
                if isinstance(candidates_raw, list):
                    for item in candidates_raw[:10]:
                        coerced = _coerce_int(item)
                        if coerced is None:
                            continue
                        candidates.append(int(coerced))
                parts: List[str] = []
                if selector_id is not None:
                    parts.append(f"selector={selector_id}")
                if candidates:
                    parts.append(f"candidates={candidates}")
                if parts:
                    return "routing:" + " ".join(parts)

                path_prefix = _path_prefix_tuple(f)
                if path_prefix:
                    return f"path_prefix={list(path_prefix)}"
                return "group=unknown"

            def _format_packet_hex(hex_str: Optional[str]) -> str:
                if not hex_str:
                    return "-"
                if hex_max_bytes is None:
                    return hex_str
                max_chars = int(hex_max_bytes) * 2
                if len(hex_str) <= max_chars:
                    return hex_str
                return hex_str[:max_chars] + "..."

            node_lookup: Dict[Any, Dict[str, Any]] = {}
            try:
                raw_nodes = getattr(state, "tree", {}).get("nodes", []) if hasattr(state, "tree") else []
                if not raw_nodes and isinstance(state, SimpleNamespace):
                    raw_nodes = state.tree.get("nodes", [])  # type: ignore[assignment]
                if isinstance(raw_nodes, list):
                    for n in raw_nodes:
                        if not isinstance(n, dict):
                            continue
                        nid = n.get("node_id")
                        if nid is None:
                            continue
                        node_lookup[nid] = n
                        node_lookup[str(nid)] = n
            except Exception:
                node_lookup = {}

            def _node_name(nid: Any) -> str:
                node = node_lookup.get(nid) or node_lookup.get(str(nid)) or {}
                name = node.get("name")
                return str(name) if name else ""

            def _format_ctx_fields(values: Dict[int, int], *, limit: int = 14) -> str:
                if not values:
                    return "{}"
                interesting: List[Tuple[int, int]] = []
                rest: List[Tuple[int, int]] = []
                for k, v in values.items():
                    name = _node_name(k).lower()
                    if any(tok in name for tok in ("length", "byte_count", "function", "unit", "quantity", "address")):
                        interesting.append((k, v))
                    else:
                        rest.append((k, v))
                interesting.sort(key=lambda kv: kv[0])
                rest.sort(key=lambda kv: kv[0])
                items = interesting + rest
                shown = items if limit <= 0 else items[:limit]
                parts: List[str] = []
                for k, v in shown:
                    name = _node_name(k)
                    if name:
                        parts.append(f"{k}={v}({name})")
                    else:
                        parts.append(f"{k}={v}")
                suffix = ", ..." if limit > 0 and len(items) > limit else ""
                return "{" + ", ".join(parts) + suffix + "}"

            grouped: Dict[str, List[Any]] = {}
            for f in failures:
                grouped.setdefault(_infer_group_signature(f), []).append(f)

            pending_issues = getattr(state, "pending_issues", None) or {}
            issue_desc_blob = ""
            issue_target_ids: set[int] = set()
            try:
                descs: List[str] = []
                for iss in pending_issues.values():
                    desc = getattr(iss, "description", None)
                    if desc:
                        descs.append(str(desc))
                    target = getattr(iss, "target", None)
                    ident = getattr(target, "identifier", None) if target is not None else None
                    if ident is None:
                        continue
                    try:
                        issue_target_ids.add(int(str(ident)))
                    except Exception:
                        continue
                issue_desc_blob = "\n".join(descs)
            except Exception:
                issue_desc_blob = ""
                issue_target_ids = set()

            def _is_relevant_group(sig: str, items: List[Any]) -> bool:
                if sig:
                    try:
                        if issue_desc_blob and sig in issue_desc_blob:
                            return True
                        if sig.startswith("routing:"):
                            sig2 = sig[len("routing:") :].strip()
                            if sig2 and issue_desc_blob and sig2 in issue_desc_blob:
                                return True
                    except Exception:
                        pass
                if issue_target_ids:
                    for rec in items:
                        nid = _coerce_int(getattr(rec, "node_id", None))
                        if nid is not None and int(nid) in issue_target_ids:
                            return True
                return False

            def _group_order(item: Tuple[str, List[Any]]) -> Tuple[int, int, int, str]:
                sig, items = item
                pkt_set: set[int] = set()
                for rec in items:
                    try:
                        pkt_set.add(int(getattr(rec, "packet_index", 0) or 0))
                    except Exception:
                        continue
                pkt_count = len(pkt_set)
                min_pkt = min(pkt_set) if pkt_set else 0
                relevant = _is_relevant_group(sig, items)
                # Prefer groups that correspond to current pending issues so the LLM doesn't
                # "fix" unrelated traffic patterns first.
                return (0 if relevant else 1, -pkt_count, min_pkt, sig)

            top_groups = sorted(grouped.items(), key=_group_order)[:max_groups]
            lines: List[str] = []
            lines.append("=== TRAFFIC FAILURE GROUPS (SAMPLES) ===")
            lines.append(f"samples: total={total_samples} success={success_samples} failed={failed_samples}")
            lines.append(
                "grouping=group_signature "
                f"group_prefix_len={prefix_len} groups_total={len(grouped)} groups_shown={len(top_groups)} "
                f"examples_per_group={examples_per_group} hex_max_bytes={hex_max_bytes}"
            )

            for idx, (sig, items) in enumerate(top_groups, 1):
                packet_indices = sorted(
                    {
                        int(getattr(f, "packet_index", 0) or 0)
                        for f in items
                        if getattr(f, "packet_index", None) is not None
                    }
                )
                kind_counts = Counter()
                for f in items:
                    kind = str(getattr(f, "failure_kind", "") or "")
                    if kind:
                        kind_counts[kind] += 1
                kind_summary = ", ".join(
                    f"{kind}={count}" for kind, count in kind_counts.most_common(8)
                )

                header = (
                    f"- group#{idx}: sig={sig} | packets={len(packet_indices)} | records={len(items)}"
                )
                if kind_summary:
                    header += f" | kinds: {kind_summary}"
                lines.append(header)

                for pkt in packet_indices[:examples_per_group]:
                    rep = next((f for f in items if int(getattr(f, "packet_index", -1) or -1) == pkt), None)
                    pkt_len = getattr(rep, "packet_len_bytes", None) if rep is not None else None
                    hex_str = getattr(rep, "packet_hex", None) if rep is not None else None
                    if pkt_len is None and isinstance(hex_str, str) and hex_str:
                        pkt_len = len(hex_str) // 2
                    lines.append(
                        f"  * pkt#{pkt} len_bytes={pkt_len} hex={_format_packet_hex(hex_str)}"
                    )
                    ctx_fields = getattr(rep, "context_field_values", None) if rep is not None else None
                    if isinstance(ctx_fields, dict) and ctx_fields:
                        try:
                            ctx_fields_int: Dict[int, int] = {}
                            for k, v in ctx_fields.items():
                                try:
                                    ctx_fields_int[int(k)] = int(v)
                                except Exception:
                                    continue
                            if ctx_fields_int:
                                lines.append(f"    ctx_fields: {_format_ctx_fields(ctx_fields_int)}")
                        except Exception:
                            pass

                    pkt_failures = [f for f in items if int(getattr(f, "packet_index", -1) or -1) == pkt]
                    pkt_failures.sort(
                        key=lambda f: (
                            str(getattr(f, "failure_kind", "") or ""),
                            _coerce_int(getattr(f, "node_id", None)) or -1,
                        )
                    )
                    sample_failures = pkt_failures if max_failures_per_sample is None else pkt_failures[: int(max_failures_per_sample)]
                    for f in sample_failures:
                        kind = str(getattr(f, "failure_kind", "") or "")
                        nid = _coerce_int(getattr(f, "node_id", None))
                        bit_start = getattr(f, "bit_start", None)
                        size_eval = getattr(f, "size_bits_eval", None)
                        msg = str(getattr(f, "message", "") or "").strip()
                        if len(msg) > 260:
                            msg = msg[:257] + "..."
                        lines.append(
                            f"    - {kind} node={nid} bit_start={bit_start} size_bits_eval={size_eval} :: {msg}"
                        )

                        selector_id = getattr(f, "routing_selector_id", None)
                        if selector_id is not None:
                            cands_raw = getattr(f, "routing_candidate_variant_ids", None)
                            if cands_raw and isinstance(cands_raw, list):
                                cands_raw = cands_raw[:10]
                            lines.append(f"      · routing: selector={selector_id} candidates={cands_raw}")
                            variant_errors = getattr(f, "routing_variant_errors", None)
                            if variant_errors and isinstance(variant_errors, list):
                                for detail in variant_errors[:max_variant_errors]:
                                    if not isinstance(detail, dict):
                                        continue
                                    vid = detail.get("variant_id")
                                    vname = str(detail.get("variant_name") or "")
                                    etype = detail.get("error_type")
                                    enid = detail.get("node_id")
                                    emsg = str(detail.get("message") or "").strip()
                                    if len(emsg) > 240:
                                        emsg = emsg[:237] + "..."
                                    label = (
                                        f"{vname}({vid})"
                                        if vname and vid is not None
                                        else str(vid)
                                        if vid is not None
                                        else "?"
                                    )
                                    lines.append(f"        · {label} err={etype} node={enid} :: {emsg}")

            # Optional: include protocol-agnostic repair hints derived by the traffic semantic validator.
            repair_hints = getattr(validation, "traffic_repair_hints", None) or []
            if repair_hints:
                raw_limit = os.getenv("STEP2_TRAFFIC_HINTS_IN_PROMPT", "8")
                try:
                    max_hints_in_prompt = int(str(raw_limit).strip())
                except Exception:
                    max_hints_in_prompt = 8
                if max_hints_in_prompt <= 0:
                    max_hints_in_prompt = 0
                    repair_hints = []
            if repair_hints:
                lines.append("=== TRAFFIC REPAIR HINTS (AUTO-INFERRED) ===")
                try:
                    hint_kind_counts = Counter(str(h.get("kind", "")) for h in repair_hints if isinstance(h, dict))
                    lines.append(
                        f"hints_total={len(repair_hints)} hints_shown={min(len(repair_hints), max_hints_in_prompt)} "
                        f"kinds_top={dict(hint_kind_counts.most_common(6))}"
                    )
                except Exception:
                    lines.append(
                        f"hints_total={len(repair_hints)} hints_shown={min(len(repair_hints), max_hints_in_prompt)}"
                    )
                for i, hint in enumerate(repair_hints[:max_hints_in_prompt], 1):
                    if not isinstance(hint, dict):
                        continue
                    kind = str(hint.get("kind", "") or "")
                    score = hint.get("score", None)
                    conf = hint.get("confidence", None)
                    target = hint.get("target", None)
                    desc = hint.get("description", None)
                    header = f"- hint#{i}: kind={kind}"
                    if score is not None:
                        header += f" score={score}"
                    if conf is not None:
                        header += f" confidence={conf}"
                    if target:
                        header += f" target={target}"
                    lines.append(header)
                    if desc:
                        lines.append(f"  {str(desc)}")
                    if kind == "add_length_of":
                        lines.append(
                            f"  suggest: add length_of src={hint.get('src')} -> dst={hint.get('target')} formula={hint.get('formula')}"
                        )
                    elif kind == "set_size_bits":
                        lines.append(
                            f"  suggest: set {hint.get('target')} size_bits={hint.get('suggested_size_bits')}"
                        )
                    elif kind == "set_variant_size_bits":
                        lines.append(
                            f"  suggest: set {hint.get('target')} size_bits={hint.get('suggested_size_bits')}"
                        )
                    elif kind == "shift_variant_subtree":
                        lines.append(
                            f"  suggest: shift descendants of {hint.get('target')} by {hint.get('shift_bits')} bits"
                        )
                    evidence = hint.get("evidence", None)
                    if isinstance(evidence, dict) and evidence:
                        try:
                            packets = evidence.get("example_packets", None)
                            if packets:
                                lines.append(f"  evidence: example_packets={packets}")
                            matches = evidence.get("tail_gap_matches", None)
                            if matches:
                                lines.append(f"  evidence: tail_gap_matches={matches}")
                        except Exception:
                            pass

            merged = "\n".join(lines)
            if base_feedback:
                return base_feedback + "\n\n" + merged
            return merged
        if total_samples == 0 and summary is not None:
            total_samples = getattr(summary, "total_samples", 0) or total_samples
        if success_samples == 0 and summary is not None:
            try:
                success_samples = total_samples - (getattr(summary, "failed_samples", 0) or 0)
            except Exception:
                success_samples = 0

        node_lookup: Dict[Any, Dict[str, Any]] = {}
        try:
            for n in state.tree.get("nodes", []) if isinstance(state, SimpleNamespace) else state.tree.get("nodes", []):
                nid = n.get("node_id")
                if nid is None:
                    continue
                node_lookup[nid] = n
                node_lookup[str(nid)] = n
        except Exception:
            try:
                for n in getattr(state, "tree", {}).get("nodes", []):
                    nid = n.get("node_id")
                    if nid is None:
                        continue
                    node_lookup[nid] = n
                    node_lookup[str(nid)] = n
            except Exception:
                node_lookup = {}

        def _node_label(nid: Any) -> str:
            node = node_lookup.get(nid) or node_lookup.get(str(nid), {})
            name = node.get("name")
            ntype = node.get("node_type")
            if name and ntype:
                return f'{name} ({ntype})'
            if name:
                return str(name)
            if ntype:
                return f"{ntype}"
            return ""

        def _format_ctx_fields(values: Dict[int, int], limit: int = 6) -> str:
            if not values:
                return "{}"
            items = sorted(values.items(), key=lambda kv: kv[0])[:limit]
            parts: List[str] = []
            for k, v in items:
                label = _node_label(k)
                if label:
                    parts.append(f"{k}:{v}({label})")
                else:
                    parts.append(f"{k}:{v}")
            payload = ", ".join(parts)
            if len(values) > limit:
                payload += ", ..."
            return "{" + payload + "}"

        def _format_path(path: List[int], limit: int = 12) -> str:
            if not path:
                return "[]"
            trimmed = path[:limit]
            suffix = "..." if len(path) > limit else ""
            return "[" + ", ".join(str(p) for p in trimmed) + suffix + "]"

        node_counts = Counter()
        if summary and getattr(summary, "per_node_failure_counts", None):
            try:
                node_counts.update(getattr(summary, "per_node_failure_counts"))
            except Exception:
                pass
        for f in failures:
            try:
                nid_int = int(getattr(f, "node_id", -1))
            except Exception:
                continue
            node_counts[nid_int] += 1

        max_failures = max(1, int(os.getenv("STEP2_TRAFFIC_FAILURES_IN_PROMPT", "8")))
        ordered_failures = sorted(
            failures,
            key=lambda f: (
                -node_counts.get(getattr(f, "node_id", None), 0),
                getattr(f, "packet_index", 0),
            ),
        )[:max_failures]

        lines: List[str] = []
        lines.append("=== TRAFFIC FAILURES (STRUCTURED) ===")
        lines.append(f"samples: total={total_samples} success={success_samples} failed={failed_samples}")
        if node_counts:
            top_nodes = node_counts.most_common(5)
            lines.append(f"top_fail_nodes: {[(nid, cnt) for nid, cnt in top_nodes]}")

        # Identify variable-sized leaf fields without a length binding on paths.
        # These nodes will parse as size_bits=0 and create coverage gaps/tail gaps.
        length_of_targets: Set[int] = set()
        try:
            for e in (state.tree.get("edges") or []):
                if not isinstance(e, dict):
                    continue
                if (e.get("rel") or "").strip().lower() != "length_of":
                    continue
                dst = e.get("dst")
                if dst is None:
                    continue
                try:
                    length_of_targets.add(int(dst))
                except Exception:
                    continue
        except Exception:
            length_of_targets = set()

        def _is_variable_leaf_without_length_binding(node_id: Any) -> bool:
            try:
                nid_int = int(node_id)
            except Exception:
                return False
            node = node_lookup.get(nid_int) or node_lookup.get(str(nid_int))
            if not isinstance(node, dict):
                return False
            node_type = str(node.get("node_type") or "").lower()
            children = node.get("children_ids") or []
            is_leaf_like = node_type in {"field", "selector", "type", "length", "checksum"} or not children
            if not is_leaf_like:
                return False
            size_bits = node.get("size_bits")
            if size_bits is None:
                return nid_int not in length_of_targets
            if isinstance(size_bits, str) and size_bits.strip().lower() in {"variable", "unknown", "dynamic", ""}:
                return nid_int not in length_of_targets
            return False

        variable_leaf_counts: Counter[int] = Counter()
        for f in failures:
            for nid in (getattr(f, "path_node_ids", None) or [])[:64]:
                if _is_variable_leaf_without_length_binding(nid):
                    try:
                        variable_leaf_counts[int(nid)] += 1
                    except Exception:
                        continue
        if variable_leaf_counts:
            top_variable = []
            for nid, cnt in variable_leaf_counts.most_common(5):
                top_variable.append((nid, cnt, _node_label(nid)))
            lines.append(f"top_variable_leaves_without_length_of: {top_variable}")

        # Group routing failures to avoid getting distracted by a few packets.
        routing_groups: Counter[Tuple[Any, Any, Tuple[int, ...], Tuple[int, ...]]] = Counter()
        for f in failures:
            if (getattr(f, "failure_kind", "") or "").lower() != "routing":
                continue
            try:
                node = int(getattr(f, "node_id", -1))
            except Exception:
                node = getattr(f, "node_id", None)
            selector_id = getattr(f, "routing_selector_id", None)
            raw_cands = getattr(f, "routing_candidate_variant_ids", None) or []
            try:
                cands = tuple(int(x) for x in raw_cands)
            except Exception:
                cands = tuple()
            prefix_raw = (getattr(f, "path_node_ids", None) or [])[:8]
            try:
                prefix = tuple(int(x) for x in prefix_raw)
            except Exception:
                prefix = tuple()
            routing_groups[(node, selector_id, cands, prefix)] += 1
        if routing_groups:
            top_groups = routing_groups.most_common(3)
            rendered = []
            for (node, selector_id, cands, prefix), cnt in top_groups:
                rendered.append(
                    {
                        "count": cnt,
                        "node": node,
                        "selector": selector_id,
                        "candidates": list(cands),
                        "prefix": list(prefix),
                    }
                )
            lines.append(f"top_routing_groups: {rendered}")

        # ------------------------------------------------------------------
        # Protocol-agnostic structural heuristics (NO tree mutation here).
        #
        # Goal: help the LLM choose the right *kind* of patch when traffic
        # failures are driven by common alignment/length-link patterns:
        #   - selector double-counting inside variant size_bits
        #   - children bit_start shifted by selector width inside variant
        #   - count-prefixed variable leaf fields missing a length_of binding
        #
        # These are hints only; the LLM must emit the actual patch.
        # ------------------------------------------------------------------

        def _as_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                try:
                    return int(value)
                except Exception:
                    return None
            if isinstance(value, str):
                candidate = value.strip()
                if not candidate:
                    return None
                # Only accept plain integers; skip expressions like "(val(5)-1)*8".
                if candidate.isdigit() or (candidate.startswith("-") and candidate[1:].isdigit()):
                    try:
                        return int(candidate, 10)
                    except Exception:
                        return None
            return None

        def _node_type(nid: Any) -> str:
            node = node_lookup.get(nid) or node_lookup.get(str(nid))
            if not isinstance(node, dict):
                return ""
            return str(node.get("node_type") or "").lower()

        # Unique mapping: variant_id -> selector_id (condition_on).
        selector_for_variant: Dict[int, int] = {}
        ambiguous_variants: Set[int] = set()
        try:
            for e in (state.tree.get("edges") or []):
                if not isinstance(e, dict):
                    continue
                if (e.get("rel") or "").strip().lower() != "condition_on":
                    continue
                src = _as_int(e.get("src"))
                dst = _as_int(e.get("dst"))
                if src is None or dst is None:
                    continue
                if dst in selector_for_variant and selector_for_variant[dst] != src:
                    ambiguous_variants.add(dst)
                else:
                    selector_for_variant[dst] = src
        except Exception:
            selector_for_variant = {}
            ambiguous_variants = set()
        for dst in ambiguous_variants:
            selector_for_variant.pop(dst, None)

        # Count how often each variant appears on failing paths (prioritize).
        variant_hits: Counter[int] = Counter()
        for f in failures:
            for nid in (getattr(f, "path_node_ids", None) or [])[:64]:
                try:
                    nid_int = int(nid)
                except Exception:
                    continue
                if _node_type(nid_int) == "variant":
                    variant_hits[nid_int] += 1

        selector_double_count: List[Dict[str, Any]] = []
        selector_child_shift: List[Dict[str, Any]] = []
        for var_id, sel_id in selector_for_variant.items():
            var = node_lookup.get(var_id) or node_lookup.get(str(var_id))
            sel = node_lookup.get(sel_id) or node_lookup.get(str(sel_id))
            if not isinstance(var, dict) or not isinstance(sel, dict):
                continue
            if str(var.get("node_type") or "").lower() != "variant":
                continue
            v_start = _as_int(var.get("bit_start"))
            v_size = _as_int(var.get("size_bits"))
            s_start = _as_int(sel.get("bit_start"))
            s_size = _as_int(sel.get("size_bits"))
            if v_start is None or v_size is None or s_start is None or s_size is None:
                continue
            if v_start != s_start + s_size:
                continue
            child_ids = var.get("children_ids") or []
            if not isinstance(child_ids, list) or not child_ids:
                continue
            min_child_start: Optional[int] = None
            max_child_end: Optional[int] = None
            child_nodes: List[int] = []
            unknown = False
            for cid_raw in child_ids:
                cid = _as_int(cid_raw)
                if cid is None:
                    continue
                child = node_lookup.get(cid) or node_lookup.get(str(cid))
                if not isinstance(child, dict):
                    continue
                c_start = _as_int(child.get("bit_start"))
                c_size = _as_int(child.get("size_bits"))
                if c_start is None or c_size is None:
                    unknown = True
                    break
                child_nodes.append(cid)
                end = c_start + c_size
                min_child_start = c_start if min_child_start is None else min(min_child_start, c_start)
                max_child_end = end if max_child_end is None else max(max_child_end, end)
            if unknown or max_child_end is None or max_child_end <= v_start:
                continue
            span = max_child_end - v_start

            # A) Selector double-counting: variant starts after selector, but size includes selector width.
            if span > 0 and v_size == span + s_size:
                selector_double_count.append(
                    {
                        "variant": var_id,
                        "selector": sel_id,
                        "selector_bits": s_size,
                        "size_bits": v_size,
                        "children_span": span,
                        "suggested_size_bits": span,
                        "hits": int(variant_hits.get(var_id, 0)),
                    }
                )
                continue

            # B) Children shifted by selector width inside the variant (leading gap == selector width).
            if (
                min_child_start is not None
                and min_child_start == v_start + s_size
                and v_size == span
                and span > s_size
            ):
                selector_child_shift.append(
                    {
                        "variant": var_id,
                        "selector": sel_id,
                        "shift_bits": s_size,
                        "size_bits": v_size,
                        "children": child_nodes[:8],
                        "hits": int(variant_hits.get(var_id, 0)),
                    }
                )

        # C) Count-prefixed variable leaf fields: suggest nearby prefix candidates (no naming assumptions).
        count_prefixed: List[Dict[str, Any]] = []
        for dst_id, hit_count in variable_leaf_counts.most_common(5):
            if not _is_variable_leaf_without_length_binding(dst_id):
                continue
            dst_node = node_lookup.get(dst_id) or node_lookup.get(str(dst_id))
            if not isinstance(dst_node, dict):
                continue
            # Only suggest "count-prefixed" length_of for leaf *fields* (avoid variants/selectors).
            if str(dst_node.get("node_type") or "").lower() != "field":
                continue
            dst_start = _as_int(dst_node.get("bit_start"))
            if dst_start is None:
                continue
            parent_id = _as_int(dst_node.get("parent_id"))
            parent = node_lookup.get(parent_id) or node_lookup.get(str(parent_id)) if parent_id is not None else None
            if not isinstance(parent, dict):
                continue
            siblings = parent.get("children_ids") or []
            if not isinstance(siblings, list):
                continue
            prefix_candidates: List[int] = []
            for sid_raw in siblings:
                sid = _as_int(sid_raw)
                if sid is None or sid == dst_id:
                    continue
                sib = node_lookup.get(sid) or node_lookup.get(str(sid))
                if not isinstance(sib, dict):
                    continue
                stype = str(sib.get("node_type") or "").lower()
                if stype not in {"field", "length"}:
                    continue
                s_start = _as_int(sib.get("bit_start"))
                s_size = _as_int(sib.get("size_bits"))
                if s_start is None or s_size is None:
                    continue
                if s_size <= 0 or s_size > 32:
                    continue
                if s_start + s_size != dst_start:
                    continue
                prefix_candidates.append(sid)
            if prefix_candidates:
                count_prefixed.append(
                    {
                        "dst": int(dst_id),
                        "prefix": prefix_candidates[:3],
                        "units_bits": [1, 8, 16, 32],
                        "hits": int(hit_count),
                    }
                )

        max_show = max(1, int(os.getenv("STEP2_TRAFFIC_HEURISTIC_HINTS", "4")))
        if selector_double_count:
            selector_double_count.sort(key=lambda d: (-int(d.get("hits", 0)), -int(d.get("selector_bits", 0))))
            shown = selector_double_count[:max_show]
            lines.append(f"heuristic_selector_double_counting: {shown}")
        if selector_child_shift:
            selector_child_shift.sort(key=lambda d: (-int(d.get("hits", 0)), -int(d.get("shift_bits", 0))))
            shown = selector_child_shift[:max_show]
            lines.append(f"heuristic_selector_child_offset: {shown}")
        if count_prefixed:
            count_prefixed.sort(key=lambda d: -int(d.get("hits", 0)))
            shown = count_prefixed[:max_show]
            lines.append(f"heuristic_count_prefixed_variable_field: {shown}")

        # Error-kind specific guidance: still LLM-only, but help the model pick
        # the right patch *type* based on the observed failure kinds.
        kind_counts = Counter()
        for f in failures:
            kind_counts[(getattr(f, "failure_kind", "") or "unknown").lower()] += 1

        guidance: List[str] = []
        if variable_leaf_counts:
            guidance.append(
                "- variable_size_leaf(no_length_of): some leaf fields have size_bits missing/variable and no length_of binding, so they parse 0 bytes and create coverage gaps. Consult DOCUMENTATION CONTEXT to identify the controlling length/count field, then add a length_of edge src=<length_field> -> dst=<variable_leaf> with a bits formula (often *8 for bytes). Keep size_bits as \"variable\"; do NOT guess a fixed size."
            )
        if kind_counts.get("coverage_gap"):
            guidance.append(
                "- coverage_gap: parsed content is smaller than the length-controlled region. FIRST check for structural problems (oob_seek/oob_read candidate variants, selector double-counting, child offset hints) and missing length_of bindings for variable leaves; fixing those often eliminates large gaps. Use traffic_payload_fill only as a LAST RESORT for genuinely opaque/undocumented bytes, and anchor formulas only to nodes on the active parse path."
            )
        if kind_counts.get("coverage_tail_gap") or kind_counts.get("coverage_internal_gap"):
            guidance.append(
                "- coverage_tail_gap/internal_gap: parser did not cover all bytes on the visited path. Prefer adding missing leaf fields (or a trailing bytes field) under the nearest container on the active parse path; derive size from an existing length/byte_count field when available (avoid circular refs like child size depending on parent.size_bits)."
            )
        if kind_counts.get("length_mismatch") or kind_counts.get("length_overflow"):
            guidance.append(
                "- length_mismatch/overflow: length binding or size_bits is wrong on-wire. Prefer adjusting length_of formula and/or dst.size_bits (check bits vs bytes: *8). Avoid fixing by adding padding unless the issue is explicitly coverage_gap."
            )
        if kind_counts.get("routing"):
            guidance.append(
                "- routing: selector/condition_on logic is wrong. Add/adjust condition_on edges or variant constraints so exactly one variant matches; keep variants mutually exclusive on the same selector."
            )
        if kind_counts.get("constraint"):
            guidance.append(
                "- constraint: observed values violate enum/range/expr. Treat this as ambiguous evidence: first check bit_start/size_bits, variant routing, message_type, and length bindings. Do NOT silently delete a node constraint. If traffic and docs show the constraint is mode-dependent, rewrite it as a conditional formula or move it to the correct variant; unsupported constraint syntax may be removed."
            )
        if kind_counts.get("oob_read") or kind_counts.get("oob_seek"):
            guidance.append(
                "- oob_read/oob_seek: attempted to read/seek past end. Prefer fixing size_bits/length_of or variant selection so fields don't exceed the packet; avoid making containers larger than frame."
            )
        if kind_counts.get("node_error") or kind_counts.get("unknown"):
            guidance.append(
                "- node_error/unknown: often caused by invalid bit_start/size_bits formulas (e.g., referencing missing node.attr). Prefer rewriting formulas to reference earlier nodes on the path, and keep arithmetic simple and monotonic."
            )

        if guidance:
            max_guidance = int(os.getenv("STEP2_TRAFFIC_GUIDANCE_LINES", "4"))
            lines.append("guidance:")
            lines.extend(guidance[: max(1, max_guidance)])

        lines.append(f"failures (up to {max_failures}):")

        for f in ordered_failures:
            nid = getattr(f, "node_id", None)
            label = _node_label(nid)
            name_part = f' name="{label}"' if label else ""
            node_type = ""
            node_obj = node_lookup.get(nid) or node_lookup.get(str(nid), {})
            if node_obj:
                node_type_val = node_obj.get("node_type")
                if node_type_val:
                    node_type = f" type={node_type_val}"
            lines.append(
                f"- pkt#{getattr(f, 'packet_index', '?')} node={nid}{name_part}{node_type} kind={getattr(f, 'failure_kind', 'unknown')}"
            )

            failure_kind = (getattr(f, "failure_kind", "") or "").lower()
            if failure_kind == "length_mismatch":
                src = getattr(f, "length_mismatch_src", None) or getattr(f, "length_src_node_id", None)
                dst = getattr(f, "length_mismatch_dst", None) or nid
                lines.append(
                    f"  edge: {src}->{dst} formula=\"{getattr(f, 'length_formula', None)}\" expected={getattr(f, 'length_expected_bits', None)} actual={getattr(f, 'length_actual_bits', None)}"
                )
            elif failure_kind == "coverage_gap":
                src = getattr(f, "length_mismatch_src", None) or getattr(f, "length_src_node_id", None)
                dst = getattr(f, "length_mismatch_dst", None) or nid
                lines.append(
                    f"  edge: {src}->{dst} gap_bits={getattr(f, 'length_gap_bits', None)} content={getattr(f, 'length_content_bits', None)} wire={getattr(f, 'length_wire_bits', None)} expected={getattr(f, 'length_expected_bits', None)}"
                )
            elif failure_kind == "routing":
                selector_id = getattr(f, "routing_selector_id", None)
                cands = getattr(f, "routing_candidate_variant_ids", None)
                if selector_id is not None or cands:
                    lines.append(f"  routing: selector={selector_id} candidates={cands}")
                verrs = getattr(f, "routing_variant_errors", None) or []
                if verrs:
                    try:
                        counts = Counter((d.get("error_type") or "unknown") for d in verrs if isinstance(d, dict))
                        lines.append(f"  variant_errors_summary: {dict(counts)}")
                    except Exception:
                        pass
                    max_show = max(1, int(os.getenv("STEP2_TRAFFIC_ROUTING_VARIANT_ERRORS_IN_PROMPT", "3")))
                    for d in verrs[:max_show]:
                        if not isinstance(d, dict):
                            continue
                        var_id = d.get("variant_id")
                        var_name = d.get("variant_name") or _node_label(var_id) if var_id is not None else None
                        var_txt = f" var={var_id}" + (f' name="{var_name}"' if var_name else "")
                        err_type = d.get("error_type") or "unknown"
                        err_node = d.get("node_id")
                        msg = str(d.get("message") or "")
                        msg = msg.replace("\n", " ").strip()
                        if len(msg) > 200:
                            msg = msg[:200] + "..."
                        nested_sel = d.get("routing_selector_id")
                        nested_cands = d.get("routing_candidate_variant_ids")
                        nested = ""
                        if nested_sel is not None or nested_cands:
                            nested = f" nested_selector={nested_sel} nested_candidates={nested_cands}"
                        lines.append(f"    -{var_txt} err={err_type} node={err_node}{nested} msg=\"{msg}\"")
            elif failure_kind == "constraint":
                lines.append(
                    f"  constraint: {getattr(f, 'constraint_text', None)} value={getattr(f, 'constraint_value', None)} kind={getattr(f, 'constraint_kind', None)}"
                )
            elif getattr(f, "message", None):
                lines.append(f"  message: {getattr(f, 'message', '')}")

            lines.append(
                f"  bit_start={getattr(f, 'bit_start', None)} size_bits_eval={getattr(f, 'size_bits_eval', None)} total_bits={getattr(f, 'total_bits', None)} max_bit_reached={getattr(f, 'max_bit_reached', None)}"
            )
            if failure_kind in {"oob_seek", "oob_read"}:
                try:
                    bs = getattr(f, "bit_start", None)
                    sz = getattr(f, "size_bits_eval", None)
                    tb = getattr(f, "total_bits", None)
                    if bs is not None and sz is not None and tb is not None:
                        bs_i = int(bs)
                        sz_i = int(sz)
                        tb_i = int(tb)
                        overshoot = (bs_i + sz_i) - tb_i
                        if overshoot > 0:
                            if overshoot % 8 == 0:
                                lines.append(
                                    f"  oob_overshoot_bits={overshoot} (={overshoot // 8} bytes)"
                                )
                            else:
                                lines.append(f"  oob_overshoot_bits={overshoot}")
                except Exception:
                    pass
            path_val = getattr(f, "path_node_ids", None) or []
            if path_val:
                lines.append(f"  path: {_format_path(path_val)}")
                variable_on_path: List[Tuple[int, str]] = []
                for pid in path_val:
                    if _is_variable_leaf_without_length_binding(pid):
                        try:
                            pid_int = int(pid)
                        except Exception:
                            continue
                        variable_on_path.append((pid_int, _node_label(pid_int)))
                    if len(variable_on_path) >= 3:
                        break
                if variable_on_path:
                    lines.append(f"  variable_leaves_without_length_of_on_path: {variable_on_path}")
            ctx_vals = getattr(f, "context_field_values", None) or {}
            if ctx_vals:
                lines.append(f"  ctx_fields: {_format_ctx_fields(ctx_vals)}")

            cand_exprs: List[str] = []
            if self._size_bits_candidates:
                candidates = self._size_bits_candidates.get(nid) or self._size_bits_candidates.get(str(nid)) or []
                for cand in candidates:
                    expr = getattr(cand, "expression", None) if hasattr(cand, "expression") else None
                    if expr is None and isinstance(cand, str):
                        expr = cand
                    if expr:
                        cand_exprs.append(str(expr))
            if cand_exprs:
                lines.append(f"  size_bits_candidates(node {nid}): {cand_exprs}")
            fill_exprs: List[str] = []
            if self._payload_fill_candidates:
                fills = self._payload_fill_candidates.get(nid) or self._payload_fill_candidates.get(str(nid)) or []
                for cand in fills:
                    size_expr = getattr(cand, "size_bits_expr", None)
                    gap_bits = getattr(cand, "gap_bits", None)
                    bit_start_expr = getattr(cand, "bit_start_expr", None)
                    parent_id = getattr(cand, "parent_id", None)
                    fill_exprs.append(
                        f"parent_id={parent_id} size_bits={size_expr} gap_bits={gap_bits} bit_start={bit_start_expr}"
                    )
            if fill_exprs:
                lines.append(f"  payload_fill_candidates(node {nid}): {fill_exprs}")

        failure_block = "\n".join(lines)
        if base_feedback:
            return f"{base_feedback}\n\n{failure_block}"
        return failure_block

def add_attention_markers(sections: Sequence[Dict[str, Any]], batch_start: int, batch_size: int) -> List[Dict[str, Any]]:
    """Highlight the focused sections for the current batch."""

    marked_sections: List[Dict[str, Any]] = []
    focus_end = batch_start + batch_size

    for index, section in enumerate(sections):
        section_copy = copy.deepcopy(section)
        content = section_copy.get("content", "")

        if batch_start <= index < focus_end:
            section_copy["content"] = f"<attention priority=\"high\">\n{content}\n</attention>"
            section_copy["is_focused"] = True
        else:
            section_copy["content"] = content
            section_copy["is_focused"] = False

        marked_sections.append(section_copy)
    return marked_sections




def _normalize_formula_text(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.replace('||', ' or ').replace('&&', ' and ')
        return normalized
    return value


def _is_non_executable_node_property_formula_constraint(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text.lower().startswith("formula:"):
        return False
    expr = text.split(":", 1)[1].strip()
    if not expr:
        return True
    if re.search(r"\bval\s*\(", expr):
        return False
    return bool(re.search(r"\b(size_bits|bit_start|remaining_bits)\b", expr))


def _normalize_constraints_value(value: Any) -> Any:
    if not isinstance(value, list):
        return _normalize_patch_value(value)
    normalized: List[Any] = []
    for item in value:
        item = _normalize_formula_text(item)
        if _is_non_executable_node_property_formula_constraint(item):
            continue
        normalized.append(item)
    return normalized


def _normalize_patch_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_patch_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_patch_value(item) for item in value]
    return _normalize_formula_text(value)


def _normalize_node_fields(node: Dict[str, Any]) -> None:
    if not isinstance(node, dict):
        return
    if 'constraints' in node and isinstance(node['constraints'], list):
        node['constraints'] = _normalize_constraints_value(node['constraints'])
    if 'bit_start' in node:
        node['bit_start'] = _normalize_formula_text(node['bit_start'])
    if 'size_bits' in node:
        node['size_bits'] = _normalize_formula_text(node['size_bits'])

def _lookup_node(node_index: Dict[str, Dict[str, Any]], tree: Dict[str, Any], node_id: Any) -> Optional[Dict[str, Any]]:
    if node_id is None:
        return None
    key = str(node_id)
    node = node_index.get(key)
    if node is not None:
        return node
    for candidate in tree.get("nodes", []) or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("node_id")) == key:
            node_index[key] = candidate
            return candidate
    return None


def _ensure_child_reference(parent: Dict[str, Any], child_id: Any) -> None:
    if parent is None or child_id is None:
        return
    children = parent.get("children_ids")
    if not isinstance(children, list):
        children = []
        parent["children_ids"] = children
    if child_id not in children:
        children.append(child_id)


def _remove_child_reference(parent: Optional[Dict[str, Any]], child_id: Any) -> None:
    if not parent or child_id is None:
        return
    children = parent.get("children_ids")
    if not isinstance(children, list):
        return
    try:
        while child_id in children:
            children.remove(child_id)
    except ValueError:
        pass


def _reassign_parent(
    node_index: Dict[str, Dict[str, Any]],
    tree: Dict[str, Any],
    child_id: Any,
    old_parent_id: Any,
    new_parent_id: Any,
) -> None:
    if old_parent_id == new_parent_id:
        return
    child = _lookup_node(node_index, tree, child_id)
    if child is None:
        return
    _remove_child_reference(_lookup_node(node_index, tree, old_parent_id), child_id)
    child["parent_id"] = new_parent_id
    _ensure_child_reference(_lookup_node(node_index, tree, new_parent_id), child_id)


def _sync_children_parent_links(
    node_index: Dict[str, Dict[str, Any]],
    tree: Dict[str, Any],
    parent_node: Dict[str, Any],
) -> None:
    if not parent_node:
        return
    parent_id = parent_node.get("node_id")
    if parent_id is None:
        return
    children = parent_node.get("children_ids")
    if not isinstance(children, list):
        parent_node["children_ids"] = []
        children = parent_node["children_ids"]
    desired = {str(child_id) for child_id in children if child_id is not None}
    for node in tree.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        if str(node.get("parent_id")) == str(parent_id) and str(node.get("node_id")) not in desired:
            node["parent_id"] = None
    for child_id in children:
        child = _lookup_node(node_index, tree, child_id)
        if child is not None:
            child["parent_id"] = parent_id


def _node_lookup(tree: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    lookup: Dict[Any, Dict[str, Any]] = {}
    for node in tree.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        nid = node.get("node_id")
        if nid is None:
            continue
        lookup[nid] = node
        lookup[str(nid)] = node
    return lookup


_NODE_ID_EXPR_RE = re.compile(r"(?:val\(\s*(\d+)\s*\)|\b(\d+)\.(?:bit_start|size_bits)\b)")
_FORBIDDEN_EXPRESSION_FUNCTION_RE = re.compile(r"\blength_of\s*\(", re.IGNORECASE)
def _repair_contract_enabled() -> bool:
    raw = os.getenv("STEP2_REPAIR_CONTRACT", "1")
    return str(raw or "").strip().lower() not in {"0", "false", "no", "off"}


def _node_id_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
    except Exception:
        pass
    return str(value)


def _bit_start_is_nonzero(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip()
    if not text:
        return False
    return text not in {"0", "0.0"}


def _tree_node_id_keys(tree: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    for node in tree.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        key = _node_id_key(node.get("node_id"))
        if key is not None:
            keys.add(key)
    return keys


def _patch_new_node_id_keys(patch: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    for node in patch.get("new_nodes", []) or []:
        if not isinstance(node, dict):
            continue
        key = _node_id_key(node.get("node_id"))
        if key is not None:
            keys.add(key)
    return keys


def _patch_removed_node_id_keys(patch: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    for entry in patch.get("nodes_to_remove", []) or []:
        raw = entry.get("node_id") if isinstance(entry, dict) else entry
        key = _node_id_key(raw)
        if key is not None:
            keys.add(key)
    return keys


def _iter_update_field_values(update: Dict[str, Any]) -> List[Tuple[str, Any]]:
    values: List[Tuple[str, Any]] = []
    if "field" in update and "new_value" in update:
        values.append((str(update.get("field")), update.get("new_value")))
    if "property" in update and "value" in update:
        values.append((str(update.get("property")), update.get("value")))

    changes = update.get("changes")
    if isinstance(changes, dict):
        for field_name, change_info in changes.items():
            if isinstance(change_info, dict):
                value = change_info.get("new_value", change_info.get("value"))
            else:
                value = change_info
            values.append((str(field_name), value))

    updates_block = update.get("updates")
    if isinstance(updates_block, dict):
        for field_name, value in updates_block.items():
            values.append((str(field_name), value))

    reserved = {
        "node_id",
        "update_type",
        "reason",
        "field",
        "new_value",
        "changes",
        "updates",
        "property",
        "value",
    }
    for field_name, value in update.items():
        if field_name in reserved:
            continue
        values.append((str(field_name), value))
    return values


def _collect_expression_node_refs(value: Any) -> Set[str]:
    refs: Set[str] = set()
    if isinstance(value, str):
        for match in _NODE_ID_EXPR_RE.finditer(value):
            refs.add(match.group(1) or match.group(2))
    elif isinstance(value, list):
        for item in value:
            refs.update(_collect_expression_node_refs(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.update(_collect_expression_node_refs(item))
    return refs


def _patch_with_inferred_new_node_bit_starts(
    tree: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """Infer local offsets for new nodes inserted under an offset parent.

    Obligation patches often know the structural placement but omit bit_start for
    a new field inside a variant body. The contract still needs an explicit local
    offset so downstream static/traffic validators can reason about the patch.
    """

    if not isinstance(patch, dict):
        return patch
    new_nodes = patch.get("new_nodes")
    if not isinstance(new_nodes, list) or not new_nodes:
        return patch

    existing_lookup = _node_lookup(tree)
    new_lookup: Dict[str, Dict[str, Any]] = {}
    for node in new_nodes:
        if not isinstance(node, dict):
            continue
        node_key = _node_id_key(node.get("node_id"))
        if node_key is not None:
            new_lookup[node_key] = node

    if not new_lookup:
        return patch

    children_by_parent: Dict[str, List[str]] = {}
    for node in tree.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_key = _node_id_key(node.get("node_id"))
        if node_key is None:
            continue
        raw_children = node.get("children_ids")
        if not isinstance(raw_children, list):
            children_by_parent[node_key] = []
            continue
        children_by_parent[node_key] = [
            child_key
            for child_id in raw_children
            if (child_key := _node_id_key(child_id)) is not None
        ]

    for update in patch.get("node_updates", []) or []:
        if not isinstance(update, dict):
            continue
        update_node_key = _node_id_key(update.get("node_id"))
        if update_node_key is None:
            continue
        for field_name, value in _iter_update_field_values(update):
            if field_name != "children_ids" or not isinstance(value, list):
                continue
            children_by_parent[update_node_key] = [
                child_key
                for child_id in value
                if (child_key := _node_id_key(child_id)) is not None
            ]

    for node_key, node in new_lookup.items():
        parent_key = _node_id_key(node.get("parent_id"))
        if parent_key is None:
            continue
        children = children_by_parent.setdefault(parent_key, [])
        if node_key not in children:
            children.append(node_key)

    def _projected_node(node_key: Optional[str]) -> Optional[Dict[str, Any]]:
        if node_key is None:
            return None
        return new_lookup.get(node_key) or existing_lookup.get(node_key)

    inferred_patch: Optional[Dict[str, Any]] = None
    for idx, node in enumerate(new_nodes):
        if not isinstance(node, dict) or "bit_start" in node:
            continue
        node_key = _node_id_key(node.get("node_id"))
        parent_key = _node_id_key(node.get("parent_id"))
        if node_key is None or parent_key is None:
            continue
        parent_node = _projected_node(parent_key)
        if not isinstance(parent_node, dict) or not _bit_start_is_nonzero(parent_node.get("bit_start")):
            continue
        node_type = str(node.get("node_type") or "").strip().lower()
        if node_type in {"variant", "choice"}:
            continue
        siblings = children_by_parent.get(parent_key) or []
        try:
            sibling_index = siblings.index(node_key)
        except ValueError:
            sibling_index = len(siblings)
        if sibling_index <= 0:
            inferred_bit_start = f"{parent_key}.bit_start + 0"
        else:
            prev_key = siblings[sibling_index - 1]
            inferred_bit_start = f"{prev_key}.bit_start + {prev_key}.size_bits"
        if inferred_patch is None:
            inferred_patch = copy.deepcopy(patch)
            new_nodes = inferred_patch.get("new_nodes") or []
        if isinstance(new_nodes, list) and idx < len(new_nodes) and isinstance(new_nodes[idx], dict):
            new_nodes[idx]["bit_start"] = inferred_bit_start

    return inferred_patch if inferred_patch is not None else patch


def _reject_unsupported_expression_functions(value: Any, path: str) -> None:
    if isinstance(value, str):
        if _FORBIDDEN_EXPRESSION_FUNCTION_RE.search(value):
            raise ValueError(
                f"{path} uses unsupported expression function length_of(); "
                "use a rel=length_of edge or constraint_ir instead"
            )
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _reject_unsupported_expression_functions(item, f"{path}[{idx}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_unsupported_expression_functions(item, f"{path}.{key}")


def _collect_tree_reference_errors(tree: Dict[str, Any]) -> Set[str]:
    errors: Set[str] = set()
    node_ids = _tree_node_id_keys(tree)
    seen_ids: Set[str] = set()
    duplicate_ids: Set[str] = set()

    for node in tree.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_key = _node_id_key(node.get("node_id"))
        if node_key is None:
            errors.add("node missing node_id")
            continue
        if node_key in seen_ids:
            duplicate_ids.add(node_key)
        seen_ids.add(node_key)

        parent_key = _node_id_key(node.get("parent_id"))
        if parent_key is not None and parent_key not in node_ids:
            errors.add(f"node {node_key} parent_id references unknown node {parent_key}")

        children = node.get("children_ids")
        if isinstance(children, list):
            for child_id in children:
                child_key = _node_id_key(child_id)
                if child_key is not None and child_key not in node_ids:
                    errors.add(f"node {node_key} children_ids references unknown node {child_key}")

        for ref in _collect_expression_node_refs(node):
            if ref not in node_ids:
                errors.add(f"node {node_key} expression has unknown node reference {ref}")

    for duplicate_id in duplicate_ids:
        errors.add(f"duplicate node_id {duplicate_id}")

    for edge in tree.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        src_key = _node_id_key(edge.get("src"))
        dst_key = _node_id_key(edge.get("dst"))
        if src_key is not None and src_key not in node_ids:
            errors.add(f"edge {src_key}->{dst_key} references unknown src node {src_key}")
        if dst_key is not None and dst_key not in node_ids:
            errors.add(f"edge {src_key}->{dst_key} references unknown dst node {dst_key}")
        for ref in _collect_expression_node_refs(edge):
            if ref not in node_ids:
                errors.add(f"edge {src_key}->{dst_key} expression has unknown node reference {ref}")

    root_key = _node_id_key(tree.get("root_node_id"))
    if root_key is not None and root_key not in node_ids:
        errors.add(f"root_node_id references unknown node {root_key}")

    return errors


def _validate_patch_repair_contract(tree: Dict[str, Any], patch: Dict[str, Any]) -> None:
    if not isinstance(patch, dict):
        raise ValueError("patch must be a JSON object")

    existing_ids = _tree_node_id_keys(tree)
    new_ids = _patch_new_node_id_keys(patch)
    removed_ids = _patch_removed_node_id_keys(patch)
    available_ids = (existing_ids | new_ids) - removed_ids
    existing_lookup = _node_lookup(tree)
    explicit_reparent_targets: Dict[str, str] = {}
    final_children_updates: Dict[str, Set[str]] = {}
    for update in patch.get("node_updates", []) or []:
        if not isinstance(update, dict):
            continue
        update_node_key = _node_id_key(update.get("node_id"))
        if update_node_key is None:
            continue
        for field_name, value in _iter_update_field_values(update):
            if field_name == "parent_id":
                new_parent_key = _node_id_key(value)
                existing_node = existing_lookup.get(update_node_key)
                old_parent_key = (
                    _node_id_key(existing_node.get("parent_id"))
                    if isinstance(existing_node, dict)
                    else None
                )
                if new_parent_key is not None and new_parent_key != old_parent_key:
                    explicit_reparent_targets[update_node_key] = new_parent_key
            elif field_name == "children_ids" and isinstance(value, list):
                final_children_updates[update_node_key] = {
                    child_key
                    for child_id in value
                    if (child_key := _node_id_key(child_id)) is not None
                }

    def _children_drop_is_explicit_reparent(old_parent_key: str, child_key: str) -> bool:
        new_parent_key = explicit_reparent_targets.get(child_key)
        if new_parent_key is None or new_parent_key == old_parent_key:
            return False
        if new_parent_key not in available_ids:
            return False
        final_new_parent_children = final_children_updates.get(new_parent_key)
        if final_new_parent_children is not None and child_key not in final_new_parent_children:
            return False
        return True

    for node_id in removed_ids:
        if node_id not in existing_ids:
            raise ValueError(f"nodes_to_remove references unknown node {node_id}")

    for idx, node in enumerate(patch.get("new_nodes", []) or []):
        if not isinstance(node, dict):
            raise ValueError(f"new_nodes[{idx}] must be an object")
        _reject_unsupported_expression_functions(node, f"new_nodes[{idx}]")
        node_key = _node_id_key(node.get("node_id"))
        if node_key is None:
            raise ValueError(f"new_nodes[{idx}] missing node_id")
        parent_key = _node_id_key(node.get("parent_id"))
        if parent_key is not None and parent_key not in available_ids:
            raise ValueError(f"new_nodes[{idx}] parent_id references unknown node {parent_key}")
        parent_node = existing_lookup.get(parent_key) if parent_key is not None else None
        node_type = str(node.get("node_type") or "").strip().lower()
        if (
            parent_node is not None
            and _bit_start_is_nonzero(parent_node.get("bit_start"))
            and "bit_start" not in node
            and node_type not in {"variant", "choice"}
        ):
            raise ValueError(f"new_nodes[{idx}] missing bit_start under offset parent {parent_key}")
        children = node.get("children_ids")
        if isinstance(children, list):
            for child_id in children:
                child_key = _node_id_key(child_id)
                if child_key is not None and child_key not in available_ids:
                    raise ValueError(f"new_nodes[{idx}] children_ids references unknown node {child_key}")
        for ref in _collect_expression_node_refs(node):
            if ref not in available_ids:
                raise ValueError(f"new_nodes[{idx}] expression has unknown node reference {ref}")

    for idx, update in enumerate(patch.get("node_updates", []) or []):
        if not isinstance(update, dict):
            raise ValueError(f"node_updates[{idx}] must be an object")
        node_key = _node_id_key(update.get("node_id"))
        if node_key is None:
            raise ValueError(f"node_updates[{idx}] missing node_id")
        if node_key not in available_ids:
            raise ValueError(f"node_updates[{idx}] references unknown node {node_key}")

        for field_name, value in _iter_update_field_values(update):
            _reject_unsupported_expression_functions(value, f"node_updates[{idx}].{field_name}")
            if field_name == "parent_id":
                parent_key = _node_id_key(value)
                if parent_key is not None and parent_key not in available_ids:
                    raise ValueError(f"node_updates[{idx}].parent_id references unknown node {parent_key}")
            if field_name == "children_ids" and isinstance(value, list):
                existing_node = existing_lookup.get(node_key)
                existing_children = existing_node.get("children_ids") if isinstance(existing_node, dict) else None
                if isinstance(existing_children, list):
                    new_child_keys = {
                        child_key
                        for child_id in value
                        if (child_key := _node_id_key(child_id)) is not None
                    }
                    for old_child_id in existing_children:
                        old_child_key = _node_id_key(old_child_id)
                        if old_child_key is None or old_child_key in removed_ids:
                            continue
                        if old_child_key not in new_child_keys:
                            if _children_drop_is_explicit_reparent(node_key, old_child_key):
                                continue
                            raise ValueError(
                                f"node_updates[{idx}].children_ids drops existing child {old_child_key}"
                            )
                for child_id in value:
                    child_key = _node_id_key(child_id)
                    if child_key is not None and child_key not in available_ids:
                        raise ValueError(f"node_updates[{idx}].children_ids references unknown node {child_key}")
            for ref in _collect_expression_node_refs(value):
                if ref not in available_ids:
                    raise ValueError(f"node_updates[{idx}].{field_name} has unknown node reference {ref}")

    for idx, edge in enumerate(_patch_new_edges(patch)):
        if not isinstance(edge, dict):
            raise ValueError(f"new_edges[{idx}] must be an object")
        _reject_unsupported_expression_functions(edge, f"new_edges[{idx}]")
        for endpoint in ("src", "dst"):
            endpoint_key = _node_id_key(edge.get(endpoint))
            if endpoint_key is not None and endpoint_key not in available_ids:
                raise ValueError(f"new_edges[{idx}].{endpoint} references unknown node {endpoint_key}")
        for ref in _collect_expression_node_refs(edge):
            if ref not in available_ids:
                raise ValueError(f"new_edges[{idx}] expression has unknown node reference {ref}")


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


def _is_protected_node(tree: Dict[str, Any], node_id: Any) -> bool:
    lookup = _node_lookup(tree)
    node = lookup.get(node_id) or lookup.get(str(node_id))
    root_id = tree.get("root_node_id")
    if root_id is not None and (node_id == root_id or str(node_id) == str(root_id)):
        return True
    return _is_aggregator_node(node, lookup)


def _is_payload_fill_patch(patch: Dict[str, Any]) -> bool:
    """Detect controlled payload fill patches allowed in traffic_fix."""
    try:
        meta = patch.get("patch_metadata") or {}
        intent = str(meta.get("intent") or meta.get("description") or "").strip().lower()
    except Exception:
        intent = ""
    if intent != "traffic_payload_fill":
        return False
    new_nodes = patch.get("new_nodes") or []
    if not isinstance(new_nodes, list) or not new_nodes:
        return False

    # Payload fill patches must be additive-only; apply_patch will auto-link the
    # new node(s) via parent_id, so do not rewrite layout or graph structure.
    if (
        patch.get("nodes_to_remove")
        or patch.get("edge_removes")
        or patch.get("edge_updates")
        or patch.get("new_edges")
        or patch.get("edge_adds")
        or patch.get("node_updates")
    ):
        return False

    # Safety: prevent known-bad formulations that always fail SyntaxValidator.
    # - No max()/min() helpers (keep expression language simple)
    # - No references to the *parent's* size_bits (circular dependency)
    for node in new_nodes:
        if not isinstance(node, dict):
            return False
        if str(node.get("node_type", "")).lower() != "field":
            return False
        dtype = str(node.get("data_type", "") or "").lower()
        if "byte" not in dtype:
            return False
        parent_id = node.get("parent_id")
        if parent_id is None:
            return False
        for expr_key in ("bit_start", "size_bits"):
            expr = node.get(expr_key)
            if not isinstance(expr, str):
                continue
            lowered = expr.lower()
            if "max(" in lowered or "min(" in lowered:
                return False
            try:
                if re.search(rf"\b{re.escape(str(parent_id))}\.size_bits\b", expr):
                    return False
            except Exception:
                # If regex fails for any reason, fall back to conservative blocking.
                return False
    return True


def _strip_payload_fill_child_link_updates(tree: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    traffic_payload_fill patches are required to be additive-only (new_nodes only).
    Some LLMs redundantly include parent.children_ids node_updates; apply_patch
    already auto-links new nodes via parent_id, so safely drop those updates when
    they only append the newly introduced node(s).
    """
    if not isinstance(patch, dict):
        return patch
    meta = patch.get("patch_metadata") or {}
    intent = str(meta.get("intent") or "").strip().lower() if isinstance(meta, dict) else ""
    if intent != "traffic_payload_fill":
        return patch

    new_nodes = patch.get("new_nodes") or []
    if not isinstance(new_nodes, list) or not new_nodes:
        return patch

    node_updates = patch.get("node_updates") or []
    if not isinstance(node_updates, list) or not node_updates:
        return patch

    # Only strip if the patch is otherwise additive-only; keep bundled edits visible.
    for key in ("new_edges", "edge_updates", "edge_removes", "nodes_to_remove"):
        if patch.get(key):
            return patch

    new_ids_by_parent: Dict[str, List[str]] = {}
    for node in new_nodes:
        if not isinstance(node, dict):
            return patch
        node_id = node.get("node_id")
        parent_id = node.get("parent_id")
        if node_id is None or parent_id is None:
            return patch
        new_ids_by_parent.setdefault(str(parent_id), []).append(str(node_id))

    lookup = _node_lookup(tree)

    def _as_id_list(value: Any) -> Optional[List[str]]:
        if not isinstance(value, list):
            return None
        return [str(item) for item in value if item is not None]

    for update in node_updates:
        if not isinstance(update, dict):
            return patch
        if update.get("property") != "children_ids":
            return patch
        if "node_id" not in update or "value" not in update:
            return patch
        parent_id = str(update.get("node_id"))
        expected_new = new_ids_by_parent.get(parent_id) or []
        if not expected_new:
            return patch
        parent = lookup.get(parent_id)
        if not isinstance(parent, dict):
            return patch
        old_children = _as_id_list(parent.get("children_ids") or []) or []
        new_children = _as_id_list(update.get("value"))
        if new_children is None:
            return patch
        if new_children[: len(old_children)] != old_children:
            return patch
        appended = new_children[len(old_children) :]
        if len(appended) != len(expected_new) or set(appended) != set(expected_new):
            return patch

    sanitized = copy.deepcopy(patch)
    sanitized["node_updates"] = []
    return sanitized


def _replace_node_property_reference(value: Any, old_node_id: str, new_node_id: str) -> Any:
    if isinstance(value, str):
        pattern = re.compile(rf"\b{re.escape(old_node_id)}\.(bit_start|size_bits)\b")
        return pattern.sub(lambda match: f"{new_node_id}.{match.group(1)}", value)
    if isinstance(value, list):
        return [_replace_node_property_reference(item, old_node_id, new_node_id) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_node_property_reference(item, old_node_id, new_node_id)
            for key, item in value.items()
        }
    return value


def _explicitly_updates_field(update: Dict[str, Any], field_name: str) -> bool:
    return any(name == field_name for name, _ in _iter_update_field_values(update))


def _patch_parent_id_update(update: Dict[str, Any]) -> Optional[Any]:
    for field_name, value in _iter_update_field_values(update):
        if field_name == "parent_id":
            return value
    return None


def _compact_node_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _removed_node_looks_like_replaced_parent(node: Dict[str, Any], removed_node: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(node, dict) or not isinstance(removed_node, dict):
        return False
    node_name = _compact_node_name(node.get("name"))
    removed_name = _compact_node_name(removed_node.get("name"))
    if not node_name or not removed_name:
        return False
    return node_name == removed_name or node_name.startswith(f"{removed_name}_")


def _rewrite_removed_parent_refs_for_reparent_updates(tree: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    If a patch moves a child away from a parent that the same patch removes,
    keep the moved child's layout expressions consistent with its new parent.
    """
    if not isinstance(patch, dict):
        return patch
    removed_ids = _patch_removed_node_id_keys(patch)
    if not removed_ids:
        return patch

    lookup = _node_lookup(tree)
    node_updates = patch.get("node_updates") or []
    if not isinstance(node_updates, list) or not node_updates:
        return patch

    existing_update_fields: Set[Tuple[str, str]] = set()
    reparent_updates: List[Tuple[str, str, str]] = []
    for update in node_updates:
        if not isinstance(update, dict):
            continue
        node_key = _node_id_key(update.get("node_id"))
        if node_key is None:
            continue
        for field_name, _ in _iter_update_field_values(update):
            existing_update_fields.add((node_key, field_name))
        new_parent_key = _node_id_key(_patch_parent_id_update(update))
        if new_parent_key is None or new_parent_key in removed_ids:
            continue
        node = lookup.get(node_key)
        if not isinstance(node, dict):
            continue
        old_parent_key = _node_id_key(node.get("parent_id"))
        if old_parent_key is not None and old_parent_key in removed_ids:
            reparent_updates.append((node_key, old_parent_key, new_parent_key))
            continue
        for field_name in ("bit_start", "size_bits"):
            for ref in _collect_expression_node_refs(node.get(field_name)):
                if ref not in removed_ids:
                    continue
                if not _removed_node_looks_like_replaced_parent(node, lookup.get(ref)):
                    continue
                reparent_updates.append((node_key, ref, new_parent_key))
                break

    if not reparent_updates:
        return patch

    added_updates: List[Dict[str, Any]] = []
    for node_key, old_parent_key, new_parent_key in reparent_updates:
        node = lookup.get(node_key)
        if not isinstance(node, dict):
            continue
        for field_name in ("bit_start", "size_bits"):
            if (node_key, field_name) in existing_update_fields:
                continue
            value = node.get(field_name)
            rewritten = _replace_node_property_reference(value, old_parent_key, new_parent_key)
            if rewritten == value:
                continue
            added_updates.append(
                {
                    "node_id": node.get("node_id"),
                    "property": field_name,
                    "value": rewritten,
                    "reason": (
                        f"Retarget {field_name} from removed parent {old_parent_key} "
                        f"to new parent {new_parent_key}"
                    ),
                }
            )
            existing_update_fields.add((node_key, field_name))

    if not added_updates:
        return patch
    rewritten_patch = copy.deepcopy(patch)
    normalized_updates = rewritten_patch.get("node_updates")
    if not isinstance(normalized_updates, list):
        normalized_updates = []
        rewritten_patch["node_updates"] = normalized_updates
    normalized_updates.extend(added_updates)
    return rewritten_patch


def _is_structural_traffic_patch(patch: Dict[str, Any], tree: Dict[str, Any]) -> bool:
    """
    For traffic_fix mode, allow small semantic/layout tweaks (length/constraints/conditions)
    but block large structural rewrites unless explicitly allowed via env.
    """
    if _is_payload_fill_patch(patch):
        return False
    allow_struct = os.getenv("STEP2_TRAFFIC_ALLOW_STRUCT", "0").lower() in {"1", "true", "yes"}
    if not isinstance(patch, dict):
        return True
    if not allow_struct and (patch.get("new_nodes") or patch.get("nodes_to_remove")):
        return True
    allowed_node_props = {"size_bits", "bit_start", "constraints", "message_type", "data_type", "needs_length_binding"}
    for upd in patch.get("node_updates", []) or []:
        nid = upd.get("node_id")
        if _is_protected_node(tree, nid) and not allow_struct:
            return True
        prop = upd.get("property") or upd.get("field")
        updates_block = upd.get("updates") if isinstance(upd, dict) else None
        props_to_check = set()
        if prop:
            props_to_check.add(prop)
        if isinstance(updates_block, dict):
            props_to_check.update(updates_block.keys())
        if not props_to_check:
            continue
        if not props_to_check.issubset(allowed_node_props):
            return True
        if not allow_struct and ("parent_id" in props_to_check or "children_ids" in props_to_check):
            return True
    allowed_rels = {"length_of", "condition_on", "depends_on"}
    for edge in _patch_new_edges(patch):
        if edge.get("rel") not in allowed_rels:
            return True
        dst = edge.get("dst")
        if _is_protected_node(tree, dst) and not allow_struct:
            return True
    for edge in patch.get("edge_updates", []) or []:
        rel = None
        ident = edge.get("edge_identifier") or {}
        rel = ident.get("rel") or edge.get("rel")
        if rel and rel not in allowed_rels:
            return True
    for edge in patch.get("edge_removes", []) or []:
        if edge.get("rel") not in allowed_rels and not allow_struct:
            return True
    return False


def reparent_node(tree: Dict[str, Any], child_id: Any, new_parent_id: Optional[Any]) -> None:
    """
    Safely move a node to a new parent while keeping children_ids/parent_id in sync.
    """
    node_index = {
        str(node.get("node_id")): node
        for node in (tree.get("nodes", []) or [])
        if isinstance(node, dict) and node.get("node_id") is not None
    }
    child = _lookup_node(node_index, tree, child_id)
    if child is None:
        return
    old_parent_id = child.get("parent_id")
    if str(old_parent_id) == str(new_parent_id):
        return
    _remove_child_reference(_lookup_node(node_index, tree, old_parent_id), child_id)
    child["parent_id"] = new_parent_id
    new_parent = _lookup_node(node_index, tree, new_parent_id)
    _ensure_child_reference(new_parent, child_id)


def remove_node(tree: Dict[str, Any], node_id: Any) -> None:
    """
    Remove a node and clean up associated edges/child references.
    """
    node_list = tree.get("nodes", [])
    if not isinstance(node_list, list):
        return
    node_index = {
        str(node.get("node_id")): node
        for node in node_list
        if isinstance(node, dict) and node.get("node_id") is not None
    }
    target = _lookup_node(node_index, tree, node_id)
    if target is None:
        return
    target_id_str = str(target.get("node_id"))
    # Detach from parent
    parent = _lookup_node(node_index, tree, target.get("parent_id"))
    _remove_child_reference(parent, target.get("node_id"))
    # Detach children
    for child in list(node_list):
        if not isinstance(child, dict):
            continue
        if str(child.get("parent_id")) == target_id_str:
            child["parent_id"] = None
    # Remove node entry
    tree["nodes"] = [
        n
        for n in node_list
        if not isinstance(n, dict) or str(n.get("node_id")) != target_id_str
    ]
    # Remove edges touching this node
    edges = tree.get("edges", [])
    if isinstance(edges, list):
        tree["edges"] = [
            e for e in edges
            if not isinstance(e, dict)
            or (str(e.get("src")) != target_id_str and str(e.get("dst")) != target_id_str)
        ]


def apply_patch(tree: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a patch to a protocol tree structure."""

    patched_tree = copy.deepcopy(tree)
    if not isinstance(patch, dict):
        return patched_tree
    patch = _rewrite_removed_parent_refs_for_reparent_updates(patched_tree, patch)
    patch = _patch_with_inferred_new_node_bit_starts(patched_tree, patch)
    repair_contract = _repair_contract_enabled()
    baseline_contract_errors = _collect_tree_reference_errors(patched_tree) if repair_contract else set()
    if repair_contract:
        _validate_patch_repair_contract(patched_tree, patch)
    strict_apply = os.getenv("STEP2_STRICT_APPLY_PATCH", "0").lower() in {"1", "true", "yes", "on"}

    nodes_list = patched_tree.get("nodes")
    if not isinstance(nodes_list, list):
        nodes_list = []
        patched_tree["nodes"] = nodes_list

    node_index = {
        str(node.get("node_id")): node
        for node in nodes_list
        if isinstance(node, dict) and node.get("node_id") is not None
    }

    removed_nodes_raw = patch.get("nodes_to_remove", []) or []

    # Remove nodes (and clean up references/edges) before applying additions/updates
    if removed_nodes_raw:
        removed_ids = []
        for entry in removed_nodes_raw:
            node_id = entry.get("node_id") if isinstance(entry, dict) else entry
            if node_id is None:
                continue
            removed_ids.append(str(node_id))
        if removed_ids:
            removed_set = set(removed_ids)
            logger.debug("Removing nodes via nodes_to_remove: %s", sorted(removed_set))

            for rid in removed_ids:
                remove_node(patched_tree, rid)
            # Rebuild index after removal
            node_index = {
                str(node.get("node_id")): node
                for node in (patched_tree.get("nodes", []) or [])
                if isinstance(node, dict) and node.get("node_id") is not None
            }

    pending_links: List[Tuple[Any, Any]] = []

    new_nodes_raw = patch.get("new_nodes") or []
    if not isinstance(new_nodes_raw, list):
        new_nodes_raw = []
    for new_node in new_nodes_raw:
        if not isinstance(new_node, dict):
            continue
        node_id_value = new_node.get("node_id")
        if node_id_value is None:
            continue
        node_id = str(node_id_value)
        if node_id not in node_index:
            normalized_node = copy.deepcopy(new_node)
            _normalize_node_fields(normalized_node)
            patched_tree.setdefault("nodes", []).append(normalized_node)
            node_index[node_id] = normalized_node
            logger.debug("Added new node: %s (ID: %s)", new_node.get("name"), node_id)

            # Auto-link to parent.children_ids if parent exists and link is missing
            parent_id = normalized_node.get("parent_id")
            if parent_id is not None:
                parent = _lookup_node(node_index, patched_tree, parent_id)
                if parent is not None:
                    _ensure_child_reference(parent, normalized_node.get("node_id"))
                    logger.debug(
                        "Linked new node %s to parent %s.children_ids",
                        node_id,
                        str(parent.get("node_id")),
                    )
                else:
                    pending_links.append((parent_id, normalized_node.get("node_id")))
        else:
            logger.debug("Node %s already exists, skipping", node_id)

    # Retry linking for nodes whose parents appeared later in the patch
    for parent_id, child_id in pending_links:
        parent = _lookup_node(node_index, patched_tree, parent_id)
        child = _lookup_node(node_index, patched_tree, child_id)
        if parent is None or child is None:
            continue
        _ensure_child_reference(parent, child_id)
        logger.debug(
            "Deferred link: new node %s attached to parent %s.children_ids",
            str(child_id),
            str(parent.get("node_id")),
        )

    node_updates_raw = patch.get("node_updates") or []
    if not isinstance(node_updates_raw, list):
        node_updates_raw = []
    for update in node_updates_raw:
        if not isinstance(update, dict):
            continue
        node_id_value = update.get("node_id")
        if node_id_value is None:
            continue
        node_id = str(node_id_value)
        node = node_index.get(node_id)
        if not node:
            if repair_contract:
                raise ValueError(f"node_updates references unknown node {node_id}")
            placeholder: Dict[str, Any] = {
                "node_id": int(node_id) if str(node_id).isdigit() else node_id,
            }
            patched_tree.setdefault("nodes", []).append(placeholder)
            node_index[node_id] = placeholder
            node = placeholder

        update_type = update.get("update_type", "single_field")
        if "field" in update and "new_value" in update:
            field = update["field"]
            old_parent_id = node.get("parent_id")
            old_value = node.get(field)
            node[field] = _normalize_formula_text(update["new_value"])
            if field == "parent_id":
                _reassign_parent(node_index, patched_tree, node.get("node_id"), old_parent_id, node.get("parent_id"))
            logger.debug("Updated node %s.%s: %s -> %s (%s)", node_id, field, old_value, update["new_value"], update.get("reason", "no reason"))
            continue

        # Support {"property": "<field>", "value": <any>} shorthand used by some patches
        if "property" in update and "value" in update:
            prop = update.get("property")
            new_value = (
                _normalize_constraints_value(update.get("value"))
                if prop == "constraints"
                else _normalize_patch_value(update.get("value"))
            )
            old_parent_id = node.get("parent_id")
            old_value = node.get(prop)
            node[prop] = new_value
            if prop == "parent_id":
                _reassign_parent(node_index, patched_tree, node.get("node_id"), old_parent_id, node.get("parent_id"))
            if prop == "children_ids":
                _sync_children_parent_links(node_index, patched_tree, node)
            logger.debug(
                "Updated node %s.%s via property/value: %s -> %s (%s)",
                node_id,
                prop,
                old_value,
                new_value,
                update.get("reason", "no reason"),
            )
            continue

        if "changes" in update:
            changes = update["changes"]
            updated_fields: List[str] = []
            for field_name, change_info in changes.items():
                old_parent_id = node.get("parent_id")
                if isinstance(change_info, dict) and "new_value" in change_info:
                    old_value = node.get(field_name)
                    node[field_name] = (
                        _normalize_constraints_value(change_info["new_value"])
                        if field_name == "constraints"
                        else _normalize_formula_text(change_info["new_value"])
                    )
                    updated_fields.append(f"{field_name}: {old_value} -> {change_info['new_value']}")
                else:
                    old_value = node.get(field_name)
                    node[field_name] = (
                        _normalize_constraints_value(change_info)
                        if field_name == "constraints"
                        else _normalize_formula_text(change_info)
                    )
                    updated_fields.append(f"{field_name}: {old_value} -> {change_info}")
                if field_name == "parent_id":
                    _reassign_parent(node_index, patched_tree, node.get("node_id"), old_parent_id, node.get("parent_id"))
                if field_name == "children_ids":
                    _sync_children_parent_links(node_index, patched_tree, node)
            logger.debug("Updated node %s (%s): %s (%s)", node_id, update_type, ", ".join(updated_fields), update.get("reason", "no reason"))
            continue

        if "updates" in update and isinstance(update["updates"], dict):
            updates_block = update["updates"]
            updated_fields: List[str] = []
            for field_name, new_value in updates_block.items():
                normalized_value = (
                    _normalize_constraints_value(new_value)
                    if field_name == "constraints"
                    else _normalize_patch_value(new_value)
                )
                old_parent_id = node.get("parent_id")
                old_value = node.get(field_name)
                node[field_name] = normalized_value
                updated_fields.append(f"{field_name}: {old_value} -> {normalized_value}")
                if field_name == "parent_id":
                    _reassign_parent(node_index, patched_tree, node.get("node_id"), old_parent_id, node.get("parent_id"))
                if field_name == "children_ids":
                    _sync_children_parent_links(node_index, patched_tree, node)
            logger.debug(
                "Updated node %s (%s updates): %s",
                node_id,
                update_type,
                ", ".join(updated_fields),
            )
            continue

        direct_updates = {
            key: value
            for key, value in update.items()
            if key not in {"node_id", "update_type", "reason", "field", "new_value", "changes", "updates", "property", "value"}
        }
        if direct_updates:
            updated_fields: List[str] = []
            for field_name, new_value in direct_updates.items():
                normalized_value = (
                    _normalize_constraints_value(new_value)
                    if field_name == "constraints"
                    else _normalize_patch_value(new_value)
                )
                old_parent_id = node.get("parent_id")
                old_value = node.get(field_name)
                node[field_name] = normalized_value
                updated_fields.append(f"{field_name}: {old_value} -> {normalized_value}")
                if field_name == "parent_id":
                    _reassign_parent(node_index, patched_tree, node.get("node_id"), old_parent_id, node.get("parent_id"))
                if field_name == "children_ids":
                    _sync_children_parent_links(node_index, patched_tree, node)
            logger.debug(
                "Updated node %s (%s direct fields): %s",
                node_id,
                update_type,
                ", ".join(updated_fields),
            )
            continue

        logger.debug("Invalid update format for node %s", node_id)

    edges_list = patched_tree.get("edges")
    if not isinstance(edges_list, list):
        edges_list = []
        patched_tree["edges"] = edges_list

    for removal in patch.get("edge_removes", []) or []:
        if not isinstance(removal, dict):
            continue
        src = removal.get("src")
        dst = removal.get("dst")
        rel = removal.get("rel")
        if src is None or dst is None or not rel:
            logger.debug("edge_removes entry missing src/dst/rel: %s", removal)
            continue

        message_type = removal.get("message_type")
        formula = removal.get("formula")
        src_str, dst_str = str(src), str(dst)

        if rel == "condition_on" and formula is None and strict_apply:
            logger.debug("edge_removes condition_on requires formula; skipping: %s", removal)
            continue

        removed_count = 0
        for index in range(len(edges_list) - 1, -1, -1):
            edge = edges_list[index]
            if str(edge.get("src")) != src_str or str(edge.get("dst")) != dst_str or edge.get("rel") != rel:
                continue
            if message_type is not None and edge.get("message_type") != message_type:
                continue
            if rel == "condition_on":
                if formula is not None and edge.get("formula") != formula:
                    continue
            elif formula is not None and edge.get("formula") != formula:
                continue
            removed_edge = edges_list.pop(index)
            removed_count += 1
            logger.debug(
                "Removed edge via edge_removes: %s -> %s (%s)%s%s",
                src_str,
                dst_str,
                rel,
                f", message_type={removed_edge.get('message_type')}" if removed_edge.get("message_type") else "",
                f", formula={removed_edge.get('formula')}" if removed_edge.get("formula") else "",
            )

        if removed_count == 0:
            logger.debug("edge_removes entry did not match any edges: %s", removal)

    existing_edges = {
        (
            str(edge.get("src")),
            str(edge.get("dst")),
            edge.get("rel"),
            str(edge.get("formula") or ""),
            str(edge.get("message_type") or ""),
        )
        for edge in edges_list
    }
    for new_edge in _patch_new_edges(patch):
        if not isinstance(new_edge, dict):
            continue
        # length_of edges should be unique per target and should not target protected aggregators
        if new_edge.get("rel") == "length_of":
            dst = new_edge.get("dst")
            if strict_apply and _is_protected_node(patched_tree, dst):
                logger.debug("Blocking length_of targeting protected/aggregator node %s", dst)
                continue
            # remove existing length_of edges to the same dst to maintain uniqueness
            for idx in range(len(edges_list) - 1, -1, -1):
                edge = edges_list[idx]
                if edge.get("rel") == "length_of" and str(edge.get("dst")) == str(dst):
                    edges_list.pop(idx)
            # refresh existing_edges after potential removals
            existing_edges = {
                (
                    str(edge.get("src")),
                    str(edge.get("dst")),
                    edge.get("rel"),
                    str(edge.get("formula") or ""),
                    str(edge.get("message_type") or ""),
                )
                for edge in edges_list
            }
        edge_key = (
            str(new_edge.get("src")),
            str(new_edge.get("dst")),
            new_edge.get("rel"),
            str(new_edge.get("formula") or ""),
            str(new_edge.get("message_type") or ""),
        )
        if edge_key not in existing_edges:
            edge_copy = copy.deepcopy(new_edge)
            if 'formula' in edge_copy:
                edge_copy['formula'] = _normalize_formula_text(edge_copy['formula'])
            patched_tree.setdefault("edges", []).append(edge_copy)
            existing_edges.add(edge_key)
            # Warn if endpoints do not exist (helps diagnose dangling edges)
            if str(new_edge.get("src")) not in node_index or str(new_edge.get("dst")) not in node_index:
                logger.debug(
                    "Edge references non-existent node(s): %s -> %s (%s)",
                    new_edge.get("src"),
                    new_edge.get("dst"),
                    new_edge.get("rel"),
                )
            logger.debug("Added new edge: %s -> %s (%s)", new_edge.get("src"), new_edge.get("dst"), new_edge.get("rel"))

    edge_updates_raw = patch.get("edge_updates") or []
    if not isinstance(edge_updates_raw, list):
        edge_updates_raw = []
    for edge_update in edge_updates_raw:
        if not isinstance(edge_update, dict):
            continue
        update_type = edge_update.get("update_type", "modify")
        edge_id = edge_update.get("edge_identifier", {})
        if not isinstance(edge_id, dict):
            edge_id = {}
        src, dst, rel = str(edge_id.get("src", "")), str(edge_id.get("dst", "")), edge_id.get("rel", "")
        target_formula = edge_id.get("formula")
        target_msg_type = edge_id.get("message_type")

        for index, edge in enumerate(patched_tree.get("edges", []) or []):
            if not isinstance(edge, dict):
                continue
            if str(edge.get("src")) != src or str(edge.get("dst")) != dst or edge.get("rel") != rel:
                continue
            
            # Check optional discriminators for precise targeting
            if target_formula is not None and edge.get("formula") != target_formula:
                continue
            if target_msg_type is not None and edge.get("message_type") != target_msg_type:
                continue

            # Found the matching edge
            if update_type == "remove":
                del patched_tree["edges"][index]
                logger.debug("Removed edge: %s -> %s (%s)", src, dst, rel)
            elif update_type == "modify":
                for prop, value in edge_update.get("new_properties", {}).items():
                    old_value = edge.get(prop)
                    edge[prop] = _normalize_formula_text(value)
                    logger.debug("Updated edge %s->%s(%s).%s: %s -> %s", src, dst, rel, prop, old_value, value)
            break
        else:
            logger.debug("Edge to update not found: %s -> %s (%s)", src, dst, rel)

    if repair_contract:
        contract_errors = _collect_tree_reference_errors(patched_tree)
        new_errors = sorted(contract_errors - baseline_contract_errors)
        if new_errors:
            preview = "; ".join(new_errors[:5])
            if len(new_errors) > 5:
                preview += f"; ... ({len(new_errors)} total)"
            raise ValueError(f"patch violates StageB repair contract: {preview}")

    return patched_tree


def _summarize_patch_for_prompt(patch: Dict[str, Any]) -> str:
    """Produce a short description of the previous patch for prompt feedback."""

    if not isinstance(patch, dict):
        return "unavailable"

    summary_parts: List[str] = []
    metadata = patch.get("patch_metadata")
    if isinstance(metadata, dict):
        description = metadata.get("description")
        if isinstance(description, str) and description.strip():
            summary_parts.append(description.strip())

    counts: List[str] = []
    for key, label in (
        ("new_nodes", "new nodes"),
        ("node_updates", "node updates"),
        ("new_edges", "new edges"),
        ("edge_adds", "new edges"),
        ("edge_updates", "edge updates"),
        ("nodes_to_remove", "nodes removed"),
    ):
        items = patch.get(key)
        if isinstance(items, list) and items:
            counts.append(f"{len(items)} {label}")

    if counts:
        summary_parts.append(", ".join(counts))

    if summary_parts:
        return "; ".join(summary_parts)
    return "without metadata"


def _is_boolean_expression(expr: str) -> bool:
    lowered = expr.lower()
    return any(token in lowered for token in ["==", "!=", " and ", " or ", "&&", "||", " not ", " xor "])


def _is_patch_empty(patch: Dict[str, Any]) -> bool:
    if not isinstance(patch, dict):
        return False
    for key in ("new_nodes", "node_updates", "new_edges", "edge_adds", "edge_updates", "nodes_to_remove"):
        items = patch.get(key)
        if isinstance(items, list) and items:
            return False
    return True


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _issue_specific_templates(raw: str) -> Optional[List[str]]:
    if not raw:
        return None

    templates: List[str] = []

    # length_of: formula references wrong node ids
    m_len_ref = re.search(
        r"Graph: .*\(ID:(?P<dst>\d+)\): length_of formula references \[(?P<refs>[^\]]+)\] but src is (?P<src>\d+)",
        raw,
    )
    if m_len_ref:
        dst = m_len_ref.group("dst")
        src = m_len_ref.group("src")
        refs = m_len_ref.group("refs").strip()
        templates.extend(
            [
                (
                    f"For node ID {dst}, rewrite the length_of edge formula or update src node id in edge array, so formula reference use the edge src {src} only (e.g., val({src})*8 if units are bytes)."
                ),
                (
                    f"If the intended sizing field is one of {refs}, then change the length_of edge src to that node instead of {src} to keep src and formula consistent."
                ),
            ]
        )

    # length_of: duplicate bindings to same dst
    m_len_multi = re.search(
        r"Graph: .*\(ID:(?P<dst>\d+)\): Multiple length_of bindings to node (?P=dst)",
        raw,
    )
    if m_len_multi:
        dst = m_len_multi.group("dst")
        templates.extend(
            [
                (
                    f"Collapse multiple length_of edges targeting node ID {dst} into a single binding."
                ),
                (
                    f"If multiple sources are required, fold their formulas into one canonical length_of edge (e.g., val(a) + val(b)) instead of duplicating bindings."
                ),
            ]
        )

    length_missing = re.search(
        r"Semantics: (?P<name>[\w]+)\(ID:(?P<id>\d+)\): Variable-length field lacks explicit length binding",
        raw,
    )
    if length_missing:
        node_name = length_missing.group("name")
        node_id = length_missing.group("id")
        templates.extend(
            [
                f"For {node_name}(ID:{node_id}), connect the governing prefix field with a length_of edge and mirror that arithmetic in size_bits (e.g., val(prefix) - payload_overhead).",
                f"If a prefix field already carries the size, add a length_of edge from that prefix to {node_name}(ID:{node_id}) so the dependency is explicit.",
                "Ensure the length_of formula and the target node's size_bits stay in sync so the validator sees a single source of truth.",
            ]
        )

    missing_len_edge = re.search(
        r"references val\((?P<src>\d+)\) but no length_of edge exists.*\(id=(?P<dst>\d+)\)",
        raw,
    )
    if missing_len_edge:
        src = missing_len_edge.group("src")
        dst = missing_len_edge.group("dst")
        templates.extend(
            [
                (
                    f"Add a length_of edge from node ID {src} to node ID {dst} so the size_bits reference is backed by an explicit dependency."
                ),
                (
                    f"Mirror the same expression (e.g., val({src})) on both the new length_of edge formula and the target size_bits to keep them aligned."
                ),
            ]
        )

    parent_overflow = re.search(
        r"Layout: (?P<name>[\w_]+)\(ID:(?P<id>\d+)\): Children may exceed parent size",
        raw,
    )
    if parent_overflow:
        label = parent_overflow.group("name")
        pid = parent_overflow.group("id")
        templates.extend(
            [
                (
                    f"Re-evaluate {label}(ID:{pid}) children ordering: bump each child's bit_start by the previous child's size_bits and recompute the parent size_bits."
                ),
                (
                    f"If a mis-parented field is causing overflow, detach it to the documented container before recalculating offsets."
                ),
            ]
        )

    multiple_roots = re.search(r"Structure: Multiple root nodes found", raw)
    if multiple_roots:
        templates.extend(
            [
                "Pick the canonical protocol root (typically Protocol_Message/Protocol_Root) and reparent every other root beneath it, then delete the redundant root stub.",
                "Confirm children_ids/parent_id fields are updated for each moved node so only one root remains.",
            ]
        )

    overlap = re.search(
        r"(overlap|boundary|conflict)", raw, re.IGNORECASE)
    if overlap:
        templates.extend(
            [
                "Adjust bit_start expressions to advance by the exact size of the preceding node so no ranges intersect.",
                "Recompute any derived size_bits formulas that still assume the old layout after shifting nodes.",
            ]
        )

    selector_issue = re.search(r"selector|variant|coverage", raw, re.IGNORECASE)
    if selector_issue:
        templates.extend(
            [
                "Ensure each selector lists concrete enumerated outcomes and add condition_on edges mapping those values to variants.",
                "Back-fill missing request/response variants so every documented branch is reachable.",
            ]
        )

    return _dedupe_preserve_order(templates) if templates else None


_SEQUENTIAL_OVERLAP_HINTS = {
    "Adjust bit_start expressions to advance by the exact size of the preceding node so no ranges intersect.",
    "Recompute any derived size_bits formulas that still assume the old layout after shifting nodes.",
}


def _selector_routing_overlap_hints(errors: Sequence[str]) -> List[str]:
    """Detect dense sibling-overlap cliques that are more likely variants than layout."""

    pairs: List[Tuple[str, str]] = []
    for raw in errors:
        text = str(raw or "")
        if "invalid overlap with sibling" not in text.lower():
            continue
        match = re.search(
            r"node=(?P<node>[A-Za-z0-9_]+)\(ID:(?P<id>\d+)\).*?"
            r"Invalid overlap with sibling (?P<sibling>[A-Za-z0-9_]+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            continue
        pairs.append((match.group("node"), match.group("sibling")))

    if len(pairs) < 3:
        return []

    endpoints = Counter()
    for left, right in pairs:
        endpoints[left] += 1
        endpoints[right] += 1

    repeated_endpoint = any(count >= 3 for count in endpoints.values())
    dense_group = len(endpoints) >= 4 and len(pairs) >= 4
    if not (repeated_endpoint or dense_group):
        return []

    top = ", ".join(name for name, _count in endpoints.most_common(4))
    suffix = f" Examples: {top}." if top else ""
    return [
        "Repeated sibling overlaps look like mutually exclusive alternatives sharing one wire offset, not sequential fields." + suffix,
        "Treat this as selector-routing: reuse the common tag/type/opcode/function control field at the same parent layer, convert alternatives into variant containers, and add mutually exclusive condition_on edges.",
        "Do not fix these alternatives by shifting bit_start or serializing sibling bodies; return {} if no shared selector evidence is present.",
        "Do not mark on-wire variants/containers as semantic-only or set their size_bits to 0 to hide overlaps.",
    ]


def _summarize_error_categories(errors: Sequence[str]) -> List[str]:
    if not errors:
        return []

    summaries: List[str] = []
    selector_overlap_hints = _selector_routing_overlap_hints(errors)
    if selector_overlap_hints:
        summaries.extend(selector_overlap_hints)

    for raw in errors[:10]:
        templates = _issue_specific_templates(raw)
        if templates:
            if selector_overlap_hints:
                templates = [hint for hint in templates if hint not in _SEQUENTIAL_OVERLAP_HINTS]
            summaries.extend(templates)
        else:
            summaries.append("Clear remaining structural violations reported by the validator before proceeding.")

    summaries = _dedupe_preserve_order(summaries)
    if len(errors) > 10:
        summaries.append("Additional validator findings remain; address them after the primary fixes above.")
    return summaries


def _summarize_coverage_hints(extras: Sequence[str]) -> List[str]:
    if not extras:
        return []

    hints: List[str] = []
    for entry in extras:
        if not isinstance(entry, str):
            continue
        lowered = entry.lower()
        if not lowered.startswith("coverage matrix"):
            continue
        if "(request)" in lowered:
            hints.append("Exercise additional request-side selector outcomes to satisfy coverage matrices.")
        elif "(response)" in lowered:
            hints.append("Exercise additional response-side selector outcomes to satisfy coverage matrices.")
        else:
            hints.append("Improve selector coverage across message flows so every branch receives an example.")
    return _dedupe_preserve_order(hints)


def _derive_structure_hints(tree: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(tree, dict):
        return []

    try:
        normalized = normalize_protocol_tree(copy.deepcopy(tree))
    except Exception:  # pragma: no cover
        normalized = tree

    protocol_tree = normalized if isinstance(normalized, dict) and "nodes" in normalized else normalized.get("protocol_tree", {})
    if not isinstance(protocol_tree, dict):
        return []

    nodes = protocol_tree.get("nodes") or []
    edges = protocol_tree.get("edges") or []
    if not isinstance(nodes, list):
        return []

    node_by_id: Dict[int, Dict[str, Any]] = {}
    for node in nodes:
        if isinstance(node, dict):
            node_id = node.get("node_id")
            if isinstance(node_id, int):
                node_by_id[node_id] = node

    length_bound_targets: Set[int] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if edge.get("rel") == "length_of":
            dst_id = edge.get("dst")
            if isinstance(dst_id, int):
                length_bound_targets.add(dst_id)

    unbound_variable_nodes: Set[int] = set()
    bound_variable_nodes: Set[int] = set()

    for node_id, node in node_by_id.items():
        size_bits = node.get("size_bits")
        if isinstance(size_bits, str) and size_bits.strip().lower() == "variable":
            if node_id in length_bound_targets:
                bound_variable_nodes.add(node_id)
            else:
                unbound_variable_nodes.add(node_id)

    hints: List[str] = []
    if unbound_variable_nodes:
        hints.append("Attach explicit length_of bindings from prefix/count fields to each variable-sized segment.")

    if bound_variable_nodes:
        hints.append("Replace placeholder 'variable' size_bits with explicit formulas tied to their controlling length fields.")

    unresolved_offset_refs = False
    if unbound_variable_nodes:
        pattern = re.compile(r"(\d+)\.size_bits")
        for node in nodes:
            if not isinstance(node, dict):
                continue
            bit_start = node.get("bit_start")
            if not isinstance(bit_start, str):
                continue
            for candidate in pattern.findall(bit_start):
                try:
                    ref_id = int(candidate)
                except ValueError:
                    continue
                if ref_id in unbound_variable_nodes:
                    unresolved_offset_refs = True
                    break
            if unresolved_offset_refs:
                break

    if unresolved_offset_refs:
        hints.append("Revisit bit_start expressions that rely on unresolved variable-sized siblings to keep offsets deterministic.")

    return _dedupe_preserve_order(hints)


def _build_patch_feedback(errors: List[str], extras: List[str], tree: Optional[Dict[str, Any]] = None) -> Optional[str]:
    error_hints = _summarize_error_categories(errors)
    # Extras (validator "extras" like coverage matrices / evaluation traces) proved too noisy,
    # so we only pass through explicit gate summaries that explain why the
    # previous candidate was rejected.
    gate_hints: List[str] = []
    for raw_extra in extras or []:
        text = str(raw_extra or "").strip()
        if text.startswith("[STATIC_REGRESSION_GATE]"):
            gate_hints.append(text)
    coverage_hints: List[str] = []
    # In strict validator-loop mode, do not derive additional structure hints
    # from the tree (heuristics). Keep guidance sourced from validator output.
    structure_hints = [] if _strict_validator_loop_enabled() else _derive_structure_hints(tree)

    segments: List[str] = []

    if error_hints:
        segments.append("Address outstanding validator findings:")
        segments.extend(f"- {hint}" for hint in error_hints)

    if gate_hints:
        if segments:
            segments.append("")
        segments.append("Do not repeat the previous candidate regression:")
        segments.extend(f"- {hint}" for hint in gate_hints[:3])

    if coverage_hints:
        if segments:
            segments.append("")
        segments.append("Coverage improvements to target:")
        segments.extend(f"- {hint}" for hint in coverage_hints)

    if structure_hints:
        if segments:
            segments.append("")
        segments.append("Structural refinement opportunities:")
        segments.extend(f"- {hint}" for hint in structure_hints)

    if not segments:
        return None

    return "\n".join(segments)


def validate_patch_consistency(tree: Dict[str, Any], patch: Dict[str, Any]) -> bool:
    strict = os.getenv("STEP2_STRICT_PATCH_CONSISTENCY", "0").lower() in {"1", "true", "yes", "on"}
    if not isinstance(patch, dict):
        logger.warning("Patch is not a dict; skipping consistency check")
        return True

    if "protocol_tree" in patch:
        logger.warning("Patch contains entire protocol_tree; skipping consistency check")
        return True

    required_fields = {"node_id", "name", "node_type"}
    has_failure = False

    new_nodes = patch.get("new_nodes") or []
    if not isinstance(new_nodes, list):
        new_nodes = []
    for idx, new_node in enumerate(new_nodes):
        if not isinstance(new_node, dict):
            logger.warning("new_nodes[%s] is not an object", idx)
            has_failure = True
            continue
        missing = required_fields - new_node.keys()
        if missing:
            logger.warning("new_nodes[%s] missing required fields: %s", idx, ", ".join(missing))
            has_failure = True
            if strict:
                return False
        bit_start = new_node.get("bit_start")
        if bit_start and not _validate_position_expression(bit_start, new_node.get("parent_id")):
            logger.warning("new_nodes[%s] has invalid bit_start expression: %s", idx, bit_start)
            has_failure = True
            if strict:
                return False

        size_bits = new_node.get("size_bits")
        if isinstance(size_bits, str) and _is_boolean_expression(size_bits):
            logger.warning("new_nodes[%s] has non-numeric size_bits expression: %s", idx, size_bits)
            has_failure = True
            if strict:
                return False

    existing_nodes = {
        str(node.get("node_id")): node
        for node in (tree.get("nodes", []) or [])
        if isinstance(node, dict) and node.get("node_id") is not None
    }

    node_updates = patch.get("node_updates") or []
    if not isinstance(node_updates, list):
        node_updates = []
    for idx, update in enumerate(node_updates):
        if not isinstance(update, dict):
            logger.warning("node_updates[%s] is not an object", idx)
            has_failure = True
            if strict:
                return False
            continue
        node_id = str(update.get("node_id"))
        target = existing_nodes.get(node_id)
        if not target:
            logger.warning("node_updates[%s] references non-existent node ID: %s", idx, node_id)
            has_failure = True
            if strict:
                return False

        if "field" in update and update["field"] == "bit_start":
            new_value = update.get("new_value")
            if new_value and not _validate_position_expression(new_value, target.get("parent_id")):
                logger.warning("node_updates[%s] invalid bit_start expression: %s", idx, new_value)
                has_failure = True
                if strict:
                    return False
        if "field" in update and update["field"] == "size_bits":
            new_value = update.get("new_value")
            if isinstance(new_value, str) and _is_boolean_expression(new_value):
                logger.warning("node_updates[%s] invalid size_bits expression: %s", idx, new_value)
                has_failure = True
                if strict:
                    return False

        for field_name, change_info in update.get("changes", {}).items():
            if field_name == "bit_start":
                new_value = change_info.get("new_value") if isinstance(change_info, dict) else change_info
                if new_value and not _validate_position_expression(new_value, target.get("parent_id")):
                    logger.warning("node_updates[%s].changes invalid bit_start expression: %s", idx, new_value)
                    has_failure = True
                    if strict:
                        return False
            if field_name == "size_bits":
                new_value = change_info.get("new_value") if isinstance(change_info, dict) else change_info
                if isinstance(new_value, str) and _is_boolean_expression(new_value):
                    logger.warning("node_updates[%s].changes invalid size_bits expression: %s", idx, new_value)
                    has_failure = True
                    if strict:
                        return False

    nodes_to_remove = patch.get("nodes_to_remove") or []
    if not isinstance(nodes_to_remove, list):
        nodes_to_remove = []
    for idx, node_id in enumerate(nodes_to_remove):
        node_id_str = str(node_id.get("node_id") if isinstance(node_id, dict) else node_id)
        if node_id_str not in existing_nodes:
            logger.warning("nodes_to_remove[%s] references non-existent node ID: %s", idx, node_id)
            has_failure = True
            if strict:
                return False

    return not has_failure if strict else True


def _validate_position_expression(bit_start_expr: Any, parent_id: Optional[Any]) -> bool:
    if isinstance(bit_start_expr, int):
        return bit_start_expr >= 0
    if isinstance(bit_start_expr, str):
        import re

        expr = bit_start_expr.strip()
        if expr.isdigit():
            return int(expr) >= 0

        allowed_pattern = re.compile(
            r"^(?:\d+\.bit_start|val\(\d+\))(?:\s*[+\-]\s*(?:\d+|\d+\.size_bits|val\(\d+\)))*$"
        )
        if allowed_pattern.match(expr):
            return True

        logger.warning("Unsupported bit_start expression: %s (parent: %s)", bit_start_expr, parent_id)
        return False
    return False


def run_full_validation(tree: Dict[str, Any]) -> ValidationReport:
    if _validator_disabled():
        return ValidationReport(
            ok=True,
            errors=[],
            warnings=[],
            extras=[],
            issues={},
        )

    normalized = normalize_protocol_tree(copy.deepcopy(tree))
    try:
        serialized = json.dumps(normalized, ensure_ascii=False)
    except TypeError:
        serialized = json.dumps({"protocol_tree": normalized}, ensure_ascii=False)
    report = validate_protocol_tree(serialized)

    self_edge_errors: List[str] = []
    protocol_tree = normalized if "nodes" in normalized else normalized.get("protocol_tree", {})
    edges = protocol_tree.get("edges", []) if isinstance(protocol_tree, dict) else []
    for edge in edges or []:
        if not isinstance(edge, dict):
            continue
        src = edge.get("src")
        dst = edge.get("dst")
        rel = edge.get("rel")
        if src is None or dst is None:
            continue
        if str(src) == str(dst):
            self_edge_errors.append(f"Edge {src}->{dst} ({rel}) is self-referential; selectors/relationships must target distinct nodes.")

    if not self_edge_errors:
        return ValidationReport(
            ok=report.ok,
            errors=list(report.errors),
            warnings=list(getattr(report, "warnings", [])),
            extras=list(report.extras),
            issues=_filter_error_issues(report.issues),
            traffic_failures=list(getattr(report, "traffic_failures", [])),
            traffic_global_summary=getattr(report, "traffic_global_summary", None),
            traffic_successful_samples=getattr(report, "traffic_successful_samples", 0),
            traffic_total_samples=getattr(report, "traffic_total_samples", 0),
            traffic_max_bit_reached=list(getattr(report, "traffic_max_bit_reached", []) or []),
            traffic_total_bits_per_sample=list(getattr(report, "traffic_total_bits_per_sample", []) or []),
            traffic_length_total_abs_error_bits=int(
                getattr(report, "traffic_length_total_abs_error_bits", 0) or 0
            ),
            traffic_wire_length_total_abs_error_bits=int(
                getattr(report, "traffic_wire_length_total_abs_error_bits", 0) or 0
            ),
            traffic_total_coverage_gap_bits=int(
                getattr(report, "traffic_total_coverage_gap_bits", 0) or 0
            ),
            traffic_coverage_gap_samples=int(
                getattr(report, "traffic_coverage_gap_samples", 0) or 0
            ),
            traffic_overflow_length_bits=int(getattr(report, "traffic_overflow_length_bits", 0) or 0),
            traffic_content_covered_bits_per_sample=list(
                getattr(report, "traffic_content_covered_bits_per_sample", []) or []
            ),
            traffic_content_coverage_ratio_per_sample=list(
                getattr(report, "traffic_content_coverage_ratio_per_sample", []) or []
            ),
        )

    errors = list(report.errors) + self_edge_errors
    warnings = list(getattr(report, "warnings", []))
    issues = _filter_error_issues(report.issues)
    issue_offset = len(issues)
    for idx, message in enumerate(self_edge_errors):
        issue_id = f"self_edge_{idx + issue_offset}"
        issues[issue_id] = Issue(
            id=issue_id,
            type=IssueType.STRUCTURE,
            severity=Severity.ERROR,
            description=message,
        )

    return ValidationReport(
        ok=False,
        errors=errors,
        warnings=warnings,
        extras=report.extras,
        issues=issues,
        traffic_failures=list(getattr(report, "traffic_failures", [])),
        traffic_global_summary=getattr(report, "traffic_global_summary", None),
        traffic_successful_samples=getattr(report, "traffic_successful_samples", 0),
        traffic_total_samples=getattr(report, "traffic_total_samples", 0),
        traffic_max_bit_reached=list(getattr(report, "traffic_max_bit_reached", []) or []),
        traffic_total_bits_per_sample=list(getattr(report, "traffic_total_bits_per_sample", []) or []),
        traffic_length_total_abs_error_bits=int(
            getattr(report, "traffic_length_total_abs_error_bits", 0) or 0
        ),
        traffic_wire_length_total_abs_error_bits=int(
            getattr(report, "traffic_wire_length_total_abs_error_bits", 0) or 0
        ),
        traffic_total_coverage_gap_bits=int(getattr(report, "traffic_total_coverage_gap_bits", 0) or 0),
        traffic_coverage_gap_samples=int(getattr(report, "traffic_coverage_gap_samples", 0) or 0),
        traffic_overflow_length_bits=int(getattr(report, "traffic_overflow_length_bits", 0) or 0),
        traffic_content_covered_bits_per_sample=list(
            getattr(report, "traffic_content_covered_bits_per_sample", []) or []
        ),
        traffic_content_coverage_ratio_per_sample=list(
            getattr(report, "traffic_content_coverage_ratio_per_sample", []) or []
        ),
    )


def sequential_refine_tree(
    agent: "EnhancedPureAIAgent",
    initial_tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    raw_sections: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    max_section_attempts: int = 3,
) -> Dict[str, Any]:
    """Sequentially refine the protocol tree section by section before MCTS fixes."""

    tree = _normalize_refinement_tree(copy.deepcopy(initial_tree) if isinstance(initial_tree, dict) else initial_tree)
    enriched_sections = [copy.deepcopy(s) for s in (sections or [])]
    num_sections = len(enriched_sections)

    current_report = run_full_validation(tree)
    feedback = _build_patch_feedback(current_report.errors, current_report.extras, tree)
    audit_records: List[Dict[str, Any]] = []

    for idx, section in enumerate(enriched_sections):
        section_label = section.get("number") or section.get("title") or f"section_{idx}"
        log_prefix = f"Section {idx + 1}/{num_sections} ({section_label})"
        logger.info("%s: starting sequential refinement", log_prefix)

        baseline_issue_count = len(current_report.issues)
        baseline_ok = current_report.ok
        best_issue_count = baseline_issue_count
        best_ok = baseline_ok
        best_tree: Optional[Dict[str, Any]] = None
        best_report: Optional[ValidationReport] = None
        best_patch: Optional[Dict[str, Any]] = None
        best_summary: Optional[str] = None
        attempts_made = 0
        reused_cache = False

        cache_filename = f"sequential_section_{idx:03d}.json"
        cached_payload = agent._load_from_cache(cache_filename)
        if isinstance(cached_payload, dict):
            cached_patch = cached_payload.get("patch")
            if isinstance(cached_patch, dict) and not _is_noop_patch(cached_patch):
                try:
                    candidate_tree = _normalize_refinement_tree(apply_patch(tree, cached_patch))
                except ValueError as exc:
                    logger.warning("%s: cached patch failed repair contract: %s", log_prefix, exc)
                    candidate_tree = None
                if candidate_tree is None:
                    cached_patch = None
                else:
                    candidate_report = run_full_validation(candidate_tree)
                    issue_count = len(candidate_report.issues)
                    candidate_ok = candidate_report.ok

                    better = False
                    if issue_count < best_issue_count:
                        better = True
                    elif issue_count == best_issue_count:
                        if candidate_ok and not best_ok:
                            better = True
                        elif candidate_ok == best_ok:
                            better = True

                    if better:
                        best_tree = candidate_tree
                        best_report = candidate_report
                        best_patch = cached_patch
                        best_summary = cached_payload.get("summary")
                        best_issue_count = issue_count
                        best_ok = candidate_ok
                        reused_cache = True
                        logger.info(
                            "%s: reused cached patch (issues %s -> %s)",
                            log_prefix,
                            baseline_issue_count,
                            best_issue_count,
                        )

        supplier: Optional[BatchPatchSupplier] = None

        if best_tree is None:
            supplier = BatchPatchSupplier(
                agent,
                enriched_sections,
                batch_index=idx,
                batch_start=idx,
                batch_size=1,
                initial_feedback=feedback,
                cached_entries=None,
                max_calls=max(1, max_section_attempts),
                cache_filename=f"sequential_section_{idx:03d}_llm.json",
                payload_fill_candidates=payload_fill_candidates,
            )

            max_calls = max(1, max_section_attempts)
            for attempt in range(max_calls):
                attempts_made = attempt + 1
                state = SimpleNamespace(
                    tree=tree,
                    pending_issues=dict(current_report.issues),
                    validation=ValidationSummary.from_report(current_report),
                    history=[],
                    last_action=None,
                    empty_patch_stop=False,
                )

                patch = supplier._generate_patch_via_llm(state)
                if patch is None:
                    logger.info("%s: no patch generated on attempt %s", log_prefix, attempts_made)
                    break
                if _is_noop_patch(patch):
                    logger.info("%s: noop patch on attempt %s; stopping further attempts", log_prefix, attempts_made)
                    break

                if not validate_patch_consistency(tree, patch):
                    logger.warning("%s: patch failed consistency checks on attempt %s", log_prefix, attempts_made)
                    continue

                try:
                    candidate_tree = _normalize_refinement_tree(apply_patch(tree, patch))
                except ValueError as exc:
                    logger.warning(
                        "%s: patch failed repair contract on attempt %s: %s",
                        log_prefix,
                        attempts_made,
                        exc,
                    )
                    continue
                candidate_report = run_full_validation(candidate_tree)
                issue_count = len(candidate_report.issues)
                candidate_ok = candidate_report.ok

                better = False
                if issue_count < best_issue_count:
                    better = True
                elif issue_count == best_issue_count:
                    if candidate_ok and not best_ok:
                        better = True
                    elif candidate_ok == best_ok:
                        better = True

                if better:
                    best_tree = candidate_tree
                    best_report = candidate_report
                    best_patch = copy.deepcopy(patch)
                    best_summary = _summarize_patch_for_prompt(patch)
                    best_issue_count = issue_count
                    best_ok = candidate_ok
                    logger.info(
                        "%s: issues improved %s -> %s on attempt %s",
                        log_prefix,
                        baseline_issue_count,
                        best_issue_count,
                        attempts_made,
                    )
                    if best_issue_count == 0:
                        logger.info("%s: validator reports zero outstanding issues", log_prefix)
                        supplier.feedback = None
                        break

                supplier.feedback = _build_patch_feedback(
                    candidate_report.errors,
                    candidate_report.extras,
                    candidate_tree,
                )

            if supplier.calls_made and supplier.generated_log:
                supplier.persist_cache()

        if supplier is not None:
            attempts_made = supplier.calls_made

        meaningful_patch = best_patch if isinstance(best_patch, dict) else None
        applied = False
        if best_tree is not None and (
            best_issue_count < baseline_issue_count
            or best_ok != baseline_ok
            or (meaningful_patch is not None and not _is_noop_patch(meaningful_patch))
        ):
            tree = best_tree
            current_report = best_report if best_report is not None else current_report
            feedback = _build_patch_feedback(current_report.errors, current_report.extras, tree)
            applied = True
            agent._save_to_cache(
                cache_filename,
                {
                    "patch": best_patch,
                    "summary": best_summary,
                    "section_index": idx,
                    "section_number": section.get("number"),
                    "section_title": section.get("title"),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        audit_records.append(
            {
                "section_index": idx,
                "section_number": section.get("number"),
                "section_title": section.get("title"),
                "attempts": supplier.calls_made if supplier else attempts_made,
                "baseline_issues": baseline_issue_count,
                "result_issues": best_issue_count if applied else baseline_issue_count,
                "applied": applied,
                "used_cache": reused_cache,
                "patch_summary": best_summary,
            }
        )

        if applied:
            logger.info(
                "%s: applied patch; outstanding issues now %s",
                log_prefix,
                best_issue_count,
            )
        else:
            logger.info("%s: no acceptable patch applied", log_prefix)

        # Refresh feedback even if no patch was applied to keep guidance aligned with latest report
        feedback = _build_patch_feedback(current_report.errors, current_report.extras, tree)

    agent._save_to_cache(
        "sequential_refinement_audit.json",
        {
            "timestamp": datetime.now().isoformat(),
            "sections": audit_records,
        },
    )

    return tree


def strict_validator_llm_fix_tree(
    agent: "EnhancedPureAIAgent",
    initial_tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]] = (),
    raw_sections: Optional[Sequence[Dict[str, Any]]] = None,
    batch_size: int = 1,
    max_llm_calls: int = 50,
    node_snapshot_dir: Optional[str] = None,
    *,
    validator_fn=run_full_validation,
    prompt_mode: str = "traffic_fix",
    fix_history: Optional[List[Dict[str, Any]]] = None,
    size_bits_candidates: Optional[Dict[int, List[Any]]] = None,
    payload_fill_candidates: Optional[Dict[int, List[Any]]] = None,
) -> Dict[str, Any]:
    """Run the validation, patch-generation, application, and revalidation loop.
    
    STEP2_STRICT_VALIDATOR_LOOP enables the mode. STEP2_STRICT_USE_SECTIONS
    includes specification sections in the patch prompt. Validation metrics
    determine candidate acceptance."""

    tree = copy.deepcopy(initial_tree) if isinstance(initial_tree, dict) else initial_tree
    tree = _normalize_refinement_tree(tree)

    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on", "y"}

    def _shorten(text: Optional[str], limit: int = 160) -> str:
        if not text:
            return ""
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)] + "..."

    def _report_traffic_ok(report: ValidationReport) -> str:
        total_samples = int(getattr(report, "traffic_total_samples", 0) or 0)
        success_samples = int(getattr(report, "traffic_successful_samples", 0) or 0)
        if total_samples <= 0:
            return "?"
        return f"{success_samples}/{total_samples}"

    def _score(report: ValidationReport) -> Tuple[int, int, int, int, int]:
        """A protocol-agnostic objective: lexicographically minimize."""
        issues = _filter_error_issues(getattr(report, "issues", {}) or {})
        issue_count = len(issues)
        total_samples = int(getattr(report, "traffic_total_samples", 0) or 0)
        success_samples = int(getattr(report, "traffic_successful_samples", 0) or 0)
        failed_samples = (total_samples - success_samples) if total_samples else len(getattr(report, "traffic_failures", []) or [])
        coverage_gap_bits = int(getattr(report, "traffic_total_coverage_gap_bits", 0) or 0)
        length_abs_err_bits = int(getattr(report, "traffic_length_total_abs_error_bits", 0) or 0)
        overflow_bits = int(getattr(report, "traffic_overflow_length_bits", 0) or 0)
        return (
            issue_count,
            int(failed_samples),
            coverage_gap_bits,
            length_abs_err_bits,
            overflow_bits,
        )

    strict_use_sections = os.getenv("STEP2_STRICT_USE_SECTIONS", "0").strip().lower() in {"1", "true", "yes", "on"}
    prompt_sections: Sequence[Dict[str, Any]] = [copy.deepcopy(s) for s in (sections or [])] if strict_use_sections else []

    prompt_mode_norm = (prompt_mode or "traffic_fix").strip().lower()

    max_calls = max(1, int(os.getenv("STEP2_STRICT_LOOP_MAX_CALLS", str(max_llm_calls))))
    avoid_keep = max(0, int(os.getenv("STEP2_STRICT_AVOID_SUMMARIES_KEEP", "8")))
    snapshot_rejected = os.getenv("STEP2_STRICT_SNAPSHOT_REJECTED", "0").strip().lower() in {"1", "true", "yes", "on"}

    print_progress = _env_flag("STEP2_STRICT_LOOP_PRINT", default=(prompt_mode_norm == "traffic_fix"))

    supplier = BatchPatchSupplier(
        agent,
        prompt_sections,
        batch_index=0,
        batch_start=0,
        batch_size=max(1, len(prompt_sections) or 1),
        initial_feedback=None,
        cached_entries=None,
        max_calls=max_calls,
        cache_filename="strict_validator_loop.json",
        prompt_mode=prompt_mode_norm,
        fix_history=fix_history,
        size_bits_candidates=size_bits_candidates,
        payload_fill_candidates=payload_fill_candidates,
    )

    strict_cache_path: Optional[Path]
    try:
        strict_cache_path = Path(getattr(agent, "cache_dir", "")).joinpath(supplier.cache_filename)
    except Exception:
        strict_cache_path = None

    best_tree = copy.deepcopy(tree)
    best_report = validator_fn(tree)
    best_score = _score(best_report)

    avoid_summaries: List[str] = []
    noop_streak = 0
    last_action = None
    accepted_count = 0
    rejected_count = 0

    if print_progress:
        cache_hint = f" cache={strict_cache_path}" if strict_cache_path else ""
        print(
            "[StrictFix] "
            f"mode={prompt_mode_norm} max_calls={max_calls} score={best_score} "
            f"traffic_ok={_report_traffic_ok(best_report)}"
            f"{cache_hint}",
            flush=True,
        )

    def _persist_snapshot(iter_idx: int, current_tree: Dict[str, Any], report: ValidationReport, patch: Optional[Dict[str, Any]]) -> None:
        if not node_snapshot_dir:
            return
        try:
            out_dir = Path(node_snapshot_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            patch_hash = _hash_patch(patch)[:8] if isinstance(patch, dict) else "nopatch"
            path = out_dir / f"strict_iter_{iter_idx:04d}_{patch_hash}.json"
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "iter": iter_idx,
                "patch_hash": patch_hash,
                "score": list(_score(report)),
                "validation": {
                    "ok": bool(getattr(report, "ok", False)),
                    "errors": list(getattr(report, "errors", []) or []),
                    "extras": list(getattr(report, "extras", []) or []),
                    "issue_ids": list(_filter_error_issues(getattr(report, "issues", {}) or {}).keys()),
                    "traffic_total_samples": int(getattr(report, "traffic_total_samples", 0) or 0),
                    "traffic_successful_samples": int(getattr(report, "traffic_successful_samples", 0) or 0),
                    "traffic_total_coverage_gap_bits": int(getattr(report, "traffic_total_coverage_gap_bits", 0) or 0),
                    "traffic_length_total_abs_error_bits": int(getattr(report, "traffic_length_total_abs_error_bits", 0) or 0),
                    "traffic_overflow_length_bits": int(getattr(report, "traffic_overflow_length_bits", 0) or 0),
                },
                "protocol_tree": current_tree,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Strict loop: failed to persist snapshot: %s", exc)

    for attempt in range(max_calls):
        report_before = validator_fn(tree)
        issues_before = _filter_error_issues(getattr(report_before, "issues", {}) or {})
        if not issues_before:
            logger.info("Strict loop: validator reports zero ERROR issues at attempt %s; stopping.", attempt + 1)
            if print_progress:
                print(f"[StrictFix] stop attempt={attempt + 1}: no ERROR issues", flush=True)
            supplier.feedback = None
            break

        # Build the prompt state *only* from validator output + current tree.
        validation_summary = ValidationSummary.from_report(report_before)
        state = SimpleNamespace(
            tree=tree,
            pending_issues=issues_before,
            validation=validation_summary,
            history=[],
            last_action=last_action,
            empty_patch_stop=False,
        )

        score_before = _score(report_before)
        if print_progress:
            print(
                "[StrictFix] "
                f"attempt={attempt + 1}/{max_calls} score={score_before} "
                f"issues={len(issues_before)} traffic_ok={_report_traffic_ok(report_before)}",
                flush=True,
            )

        patches = supplier(state, 1, avoid_summaries=avoid_summaries)
        if not patches:
            logger.info("Strict loop: LLM returned no patch at attempt %s; stopping.", attempt + 1)
            if print_progress:
                print(f"[StrictFix] stop attempt={attempt + 1}: LLM returned no patch", flush=True)
            break
        patch = patches[0]

        if patch is None or _is_noop_patch(patch):
            noop_streak += 1
            logger.info("Strict loop: noop patch at attempt %s (streak=%s)", attempt + 1, noop_streak)
            if print_progress:
                print(f"[StrictFix] noop attempt={attempt + 1} streak={noop_streak}", flush=True)
            if noop_streak >= 2:
                break
            continue
        noop_streak = 0

        if not isinstance(patch, dict):
            logger.warning("Strict loop: patch is not a dict at attempt %s; skipping.", attempt + 1)
            if print_progress:
                print(f"[StrictFix] skip attempt={attempt + 1}: patch is not a dict", flush=True)
            continue

        try:
            candidate_tree = _normalize_refinement_tree(apply_patch(tree, patch))
        except ValueError as exc:
            logger.warning("Strict loop: patch failed repair contract at attempt %s: %s", attempt + 1, exc)
            if print_progress:
                print(f"[StrictFix] skip attempt={attempt + 1}: repair contract failed: {exc}", flush=True)
            continue
        candidate_report = validator_fn(candidate_tree)

        score_after = _score(candidate_report)

        patch_hash = _hash_patch(patch)[:8]
        patch_summary = _shorten(_summarize_patch_for_prompt(patch), 140)

        if score_after <= score_before:
            # Accept non-worsening patches; best-tree tracking keeps monotonicity.
            tree = candidate_tree
            last_action = SimpleNamespace(
                patch=patch,
                hash=_hash_patch(patch)[:16],
                summary=_summarize_patch_for_prompt(patch),
            )
            _persist_snapshot(attempt, tree, candidate_report, patch)

            if score_after < best_score:
                best_score = score_after
                best_tree = copy.deepcopy(tree)
                best_report = candidate_report
                avoid_summaries = []
            else:
                # Keep avoid list bounded.
                if avoid_keep > 0:
                    avoid_summaries = avoid_summaries[-avoid_keep:]

            logger.info(
                "Strict loop: accepted patch %s score %s -> %s",
                last_action.hash[:8],
                score_before,
                score_after,
            )
            accepted_count += 1
            if print_progress:
                print(
                    "[StrictFix] "
                    f"ACCEPT patch={patch_hash} score={score_before}->{score_after}"
                    f"{' summary=' + patch_summary if patch_summary else ''}",
                    flush=True,
                )

            if not _filter_error_issues(getattr(candidate_report, "issues", {}) or {}):
                logger.info("Strict loop: validator reports zero ERROR issues after attempt %s; stopping.", attempt + 1)
                if print_progress:
                    print(f"[StrictFix] stop attempt={attempt + 1}: zero ERROR issues", flush=True)
                supplier.feedback = None
                break
        else:
            # Reject and tell the LLM to avoid repeating this patch summary.
            summary_text = _summarize_patch_for_prompt(patch)
            if summary_text:
                avoid_summaries.append(summary_text)
                if avoid_keep > 0:
                    avoid_summaries = avoid_summaries[-avoid_keep:]

            logger.info(
                "Strict loop: rejected patch score %s -> %s; retrying.",
                score_before,
                score_after,
            )
            rejected_count += 1
            if print_progress:
                print(
                    "[StrictFix] "
                    f"REJECT patch={patch_hash} score={score_before}->{score_after}"
                    f"{' summary=' + patch_summary if patch_summary else ''}",
                    flush=True,
                )

            if snapshot_rejected:
                _persist_snapshot(attempt, candidate_tree, candidate_report, patch)

    if supplier.calls_made and supplier.generated_log:
        supplier.persist_cache()

    # Final safety: return the best (lexicographically smallest score) tree.
    try:
        final_report = validator_fn(tree)
        if _score(final_report) > best_score:
            tree = best_tree
    except Exception:
        tree = best_tree

    if print_progress:
        try:
            final_report = validator_fn(tree)
            final_score = _score(final_report)
            traffic_ok = _report_traffic_ok(final_report)
        except Exception:
            final_score = best_score
            traffic_ok = "?"
        cache_hint = f" cache={strict_cache_path}" if strict_cache_path else ""
        print(
            "[StrictFix] "
            f"done calls={supplier.calls_made} accepted={accepted_count} rejected={rejected_count} "
            f"final_score={final_score} traffic_ok={traffic_ok}"
            f"{cache_hint}",
            flush=True,
        )

    return tree


def mcts_fix_tree(
    agent: "EnhancedPureAIAgent",
    initial_tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    raw_sections: Optional[Sequence[Dict[str, Any]]] = None,
    batch_size: int = 5,
    max_llm_calls: int = 100000,
    node_snapshot_dir: Optional[str] = None,
    *,
    validator_fn=run_full_validation,
    prompt_mode: str = "fix",
    fix_history: Optional[List[Dict[str, Any]]] = None,
    size_bits_candidates: Optional[Dict[int, List[Any]]] = None,
    payload_fill_candidates: Optional[Dict[int, List[Any]]] = None,
) -> Dict[str, Any]:
    """Run MCTS with one repair-attempt budget shared across document batches.

    Provider transport retries are accounted in usage logs separately. Explicit
    nested syntax-followup invocations have their own configured repair budget.
    """

    tree = copy.deepcopy(initial_tree) if isinstance(initial_tree, dict) else initial_tree
    tree = _normalize_refinement_tree(tree)

    strict_enabled = _strict_validator_loop_enabled()
    prompt_mode_norm = (prompt_mode or "").strip().lower()
    traffic_patch_sanitize = os.getenv("STEP2_TRAFFIC_SANITIZE_PATCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if prompt_mode_norm == "traffic_fix":
        try:
            print(f"[TrafficFix][MCTS] patch_sanitize={traffic_patch_sanitize}", flush=True)
        except Exception:
            pass

    # NOTE: Strict mode is meant to make fixing "validator-driven", not to disable MCTS.
    # If you really want the sequential strict loop (no MCTS), opt-in explicitly.
    strict_bypass_mcts = os.getenv("STEP2_STRICT_TRAFFIC_BYPASS_MCTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict_enabled and prompt_mode_norm == "traffic_fix":
        # Honor strict-loop call budget as an optional cap for traffic fixing even when using MCTS.
        try:
            env_cap = os.getenv("STEP2_STRICT_LOOP_MAX_CALLS")
            if env_cap is not None:
                cap_int = int(env_cap)
                if cap_int > 0:
                    max_llm_calls = min(max_llm_calls, cap_int)
        except Exception:
            pass

        if strict_bypass_mcts:
            try:
                print(
                    "[StrictFix] strict traffic_fix: bypassing MCTS (STEP2_STRICT_TRAFFIC_BYPASS_MCTS=1)",
                    flush=True,
                )
            except Exception:
                pass
            logger.info(
                "Strict traffic_fix: bypassing MCTS (STEP2_STRICT_TRAFFIC_BYPASS_MCTS=1) and running strict loop"
            )
            return strict_validator_llm_fix_tree(
                agent,
                tree,
                sections=sections,
                raw_sections=raw_sections,
                batch_size=batch_size,
                max_llm_calls=max_llm_calls,
                node_snapshot_dir=node_snapshot_dir,
                validator_fn=validator_fn,
                prompt_mode=prompt_mode,
                fix_history=fix_history,
                size_bits_candidates=size_bits_candidates,
                payload_fill_candidates=payload_fill_candidates,
            )

        try:
            print(
                "[StrictFix] strict traffic_fix: running MCTS search (set STEP2_STRICT_TRAFFIC_BYPASS_MCTS=1 to bypass)",
                flush=True,
            )
        except Exception:
            pass

    enriched_sections = [copy.deepcopy(s) for s in (sections or [])]
    num_sections = len(enriched_sections)
    batch_count = (num_sections + batch_size - 1) // batch_size
    all_patch_paths: List[List[Dict[str, Any]]] = []
    batch_path_records: List[Dict[str, Any]] = []
    patch_audit: List[Dict[str, Any]] = []

    base_report = validator_fn(tree)
    best_tree = copy.deepcopy(tree)
    from ..validation.syntax_validator import Severity as _Severity

    def _is_traffic_issue_id(issue_id: Any) -> bool:
        return str(issue_id).startswith("traffic_")

    def _is_doc_issue_id(issue_id: Any) -> bool:
        text = str(issue_id)
        return (
            text == "doc_constraint_coverage_total_uncovered"
            or text.startswith("doc_constraint_")
            or text.startswith("doc_structure_alignment_")
            or text == "doc_relational_obligation_total_uncovered"
            or text.startswith("doc_relational_obligation_")
            or text.startswith("node_constraint_preservation_")
        )

    def _error_count(report, *, only_static: bool = False) -> int:
        try:
            issues = getattr(report, "issues", {}) or {}
            if only_static:
                return sum(
                    1
                    for issue_id, issue in issues.items()
                    if (not _is_traffic_issue_id(issue_id))
                    and (not _is_doc_issue_id(issue_id))
                    and getattr(issue, "severity", None) == _Severity.ERROR
                )
            return sum(
                1
                for issue in issues.values()
                if getattr(issue, "severity", None) == _Severity.ERROR
            )
        except Exception:
            return len(report.errors or [])

    def _issue_breakdown(report, *, only_static: bool = False) -> Tuple[int, int, int, int]:
        struct_err = 0
        sem_err = 0
        invariant_err = 0
        issues = getattr(report, "issues", {}) or {}
        for issue_id, issue in issues.items():
            if only_static and (_is_traffic_issue_id(issue_id) or _is_doc_issue_id(issue_id)):
                continue
            if getattr(issue, "severity", None) != _Severity.ERROR:
                continue
            if getattr(issue, "type", None) == IssueType.STRUCTURE:
                struct_err += 1
                if getattr(issue, "code", None) == "INVARIANT_VIOLATION":
                    invariant_err += 1
            elif getattr(issue, "type", None) == IssueType.SEMANTICS:
                sem_err += 1
        total_err = _error_count(report, only_static=only_static)
        return struct_err, sem_err, total_err, invariant_err

    def _doc_accept_score(report: Any) -> Tuple[int, int, int]:
        """Lower is better: uncovered doc constraints, structure requirements, relational obligations."""

        summary = ValidationSummary.from_report(report)
        constraint_uncovered = int(getattr(summary, "doc_constraint_uncovered_total", 0) or 0)
        if int(getattr(summary, "doc_structure_total_requirements", 0) or 0) > 0:
            structure_uncovered = int(getattr(summary, "doc_structure_uncovered_requirements", 0) or 0)
        else:
            structure_uncovered = int(getattr(summary, "doc_structure_total_issues", 0) or 0)
        relational_uncovered = int(getattr(summary, "doc_relational_uncovered_total", 0) or 0)
        return constraint_uncovered, structure_uncovered, relational_uncovered

    def _doc_structure_issue_kind(issue: Issue) -> str:
        target = getattr(issue, "target", None)
        metadata = getattr(target, "metadata", None) if target is not None else None
        if not isinstance(metadata, dict):
            return ""
        repair = metadata.get("safe_repair")
        if not isinstance(repair, dict):
            return ""
        return str(repair.get("kind") or "")

    selector_coded_doc_structure_kinds = {
        "missing_selector_variants",
        "missing_guarded_variant",
        "missing_tlv_abstraction",
        "opaque_variant_expansion",
        "missing_scoped_field",
        "missing_repeated_format",
    }

    def _is_selector_coded_doc_structure_issue(issue: Issue) -> bool:
        kind = _doc_structure_issue_kind(issue)
        if kind in selector_coded_doc_structure_kinds:
            return True
        target = getattr(issue, "target", None)
        metadata = getattr(target, "metadata", None) if target is not None else None
        if not isinstance(metadata, dict):
            return False
        repair = metadata.get("safe_repair")
        if not isinstance(repair, dict):
            return False
        phase = str(repair.get("repair_phase") or "").strip().lower()
        return phase in {"selector_shell", "exact_selector_variant", "body_expansion"}

    def _doc_structure_selector_value_count(issue: Issue) -> int:
        target = getattr(issue, "target", None)
        metadata = getattr(target, "metadata", None) if target is not None else None
        if not isinstance(metadata, dict):
            return 1
        repair = metadata.get("safe_repair")
        if not isinstance(repair, dict):
            return 1
        values = repair.get("selector_values") or repair.get("guard_values") or []
        try:
            count = len(list(values))
        except Exception:
            count = 0
        return max(1, count)

    def _is_doc_structure_alignment_issue(issue_id: Any, issue: Issue) -> bool:
        return str(issue_id).startswith("doc_structure_alignment_") or str(
            getattr(issue, "description", "") or ""
        ).lower().startswith("documentation structure")

    def _is_doc_relational_obligation_issue(issue_id: Any, issue: Issue) -> bool:
        return str(issue_id).startswith("doc_relational_obligation_") or str(
            getattr(issue, "description", "") or ""
        ).lower().startswith("documentation relational obligation")

    def _staged_selector_doc_progress(baseline_report: Any, candidate_report: Any) -> bool:
        """Allow selector-map repairs that reveal follow-up body obligations.

        Adding a documented selector branch can make previously unreachable
        child-body requirements visible. That is staged progress, not a true
        regression, as long as no non-doc issue or relational obligation appears.
        """

        before = getattr(baseline_report, "issues", {}) or {}
        after = getattr(candidate_report, "issues", {}) or {}
        before_ids = set(before.keys())
        after_ids = set(after.keys())
        introduced = {issue_id: after[issue_id] for issue_id in after_ids - before_ids}
        resolved = {issue_id: before[issue_id] for issue_id in before_ids - after_ids}
        resolved_selector = sum(
            1
            for issue_id, issue in resolved.items()
            if _is_doc_structure_alignment_issue(issue_id, issue)
            and _is_selector_coded_doc_structure_issue(issue)
        )
        resolved_selector_values = sum(
            _doc_structure_selector_value_count(issue)
            for issue_id, issue in resolved.items()
            if _is_doc_structure_alignment_issue(issue_id, issue)
            and _is_selector_coded_doc_structure_issue(issue)
        )
        if resolved_selector <= 0:
            return False
        introduced_doc_structure = sum(
            1
            for issue_id, issue in introduced.items()
            if _is_doc_structure_alignment_issue(issue_id, issue)
        )
        introduced_doc_relational = sum(
            1
            for issue_id, issue in introduced.items()
            if _is_doc_relational_obligation_issue(issue_id, issue)
        )
        introduced_effective = len(introduced) - introduced_doc_structure - introduced_doc_relational
        return (
            introduced_effective == 0
            and introduced_doc_relational == 0
            and introduced_doc_structure <= max(4, resolved_selector_values * 2)
        )

    def _traffic_failed_samples(report: Any) -> int:
        total = int(getattr(report, "traffic_total_samples", 0) or 0)
        succ = int(getattr(report, "traffic_successful_samples", 0) or 0)
        if total:
            return max(0, total - succ)
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
        max_bits = getattr(report, "traffic_max_bit_reached", None) or ()
        total_bits = total_bits or (getattr(report, "traffic_total_bits_per_sample", None) or ())
        if not max_bits or not total_bits:
            return 0.0
        cov_sum = 0.0
        count = 0
        for mb, tb in zip(max_bits, total_bits):
            try:
                mb_val = float(mb)
                tb_val = float(tb)
            except Exception:
                continue
            if tb_val <= 0:
                continue
            mb_eff = mb_val if mb_val <= tb_val else tb_val
            cov_sum += mb_eff / tb_val
            count += 1
        return (cov_sum / float(count)) if count else 0.0

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
        return (failed, -coverage, gap_bits, len_err, overflow, _error_count(report))

    def _format_traffic_score(score: Tuple[int, float, int, int, int, int]) -> str:
        failed, neg_cov, gap_bits, len_err, overflow, issue_cnt = score
        return (
            f"failed={failed} cov={-neg_cov:.3f} gap_bits={gap_bits} "
            f"len_abs_err_bits={len_err} overflow_bits={overflow} error_issues={issue_cnt}"
        )

    def _traffic_scalar_score(report: Any) -> float:
        """
        Scalarize traffic metrics into a single value (higher is better) so we can
        accept multi-step improvements that may temporarily trade success vs coverage.

        We intentionally reuse the same env-tunable weights as MCTS reward shaping.
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

        defaults = MCTSConfig()
        w_success = _w("STEP2_MCTS_TRAFFIC_WEIGHT_SUCCESS", defaults.weight_traffic_success)
        w_cov = _w("STEP2_MCTS_TRAFFIC_WEIGHT_COVERAGE", defaults.weight_traffic_fix)
        w_gap = _w("STEP2_MCTS_TRAFFIC_WEIGHT_GAP", defaults.weight_traffic_gap)
        w_len = _w("STEP2_MCTS_TRAFFIC_WEIGHT_LENGTH_ERROR", defaults.weight_traffic_length_error)
        w_new = _w("STEP2_MCTS_TRAFFIC_WEIGHT_NEW", defaults.weight_traffic_new)
        w_overflow = _w("STEP2_MCTS_TRAFFIC_WEIGHT_OVERFLOW", 0.0)

        return (
            (w_success * float(succ))
            + (w_cov * float(cov))
            - (w_gap * float(gap_bits))
            - (w_len * float(len_err))
            - (w_new * float(failed))
            - (w_overflow * float(overflow))
        )
    # Track only ERROR-grade issues for root stats/logs to align with gating logic
    best_issue_count = _error_count(base_report)
    best_error_count = best_issue_count
    feedback = _build_patch_feedback(base_report.errors, base_report.extras, tree)

    patch_audit.append(
        {
            "batch": -1,
            "source": "initial",
            "ok": base_report.ok,
            "errors": base_report.errors,
            "extras": base_report.extras,
        }
    )

    logger.info("Starting MCTS fix with %s sections (batch size=%s)", num_sections, batch_size)
    remaining_llm_calls = max(0, int(max_llm_calls))

    for batch_idx in range(batch_count):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_sections)
        logger.info("Processing batch %s/%s: sections %s to %s", batch_idx + 1, batch_count, batch_start, batch_end - 1)

        # Baseline validation metrics before attempting this batch
        baseline_report = validator_fn(tree)
        baseline_struct_err, baseline_sem_err, baseline_total_err, baseline_invariant_err = _issue_breakdown(baseline_report)
        baseline_static_struct_err = baseline_struct_err
        baseline_static_sem_err = baseline_sem_err
        baseline_static_total_err = baseline_total_err
        baseline_static_invariant_err = baseline_invariant_err
        if prompt_mode_norm != "traffic_fix":
            baseline_static_struct_err, baseline_static_sem_err, baseline_static_total_err, baseline_static_invariant_err = _issue_breakdown(
                baseline_report, only_static=True
            )
        baseline_traffic_score: Optional[Tuple[int, float, int, int, int, int]] = None
        if prompt_mode_norm == "traffic_fix":
            baseline_traffic_score = _traffic_accept_score(baseline_report)
            baseline_static_struct_err, baseline_static_sem_err, baseline_static_total_err, baseline_static_invariant_err = _issue_breakdown(
                baseline_report, only_static=True
            )

        legacy_cache_filename = f"patch_section_{batch_start}_to_{batch_end}_response.json"
        cache_filename = f"{prompt_mode_norm}_{legacy_cache_filename}"
        cached_data = agent._load_from_cache(cache_filename)
        if cached_data is None:
            cached_data = agent._load_from_cache(legacy_cache_filename)
        cached_entries = _extract_cached_patch_entries(cached_data)

        supplier = BatchPatchSupplier(
            agent,
            enriched_sections,
            batch_index=batch_idx,
            batch_start=batch_start,
            batch_size=batch_size,
            initial_feedback=feedback,
            cached_entries=cached_entries,
            max_calls=remaining_llm_calls,
            cache_filename=cache_filename,
            prompt_mode=prompt_mode,
            fix_history=fix_history,
            size_bits_candidates=size_bits_candidates,
            payload_fill_candidates=payload_fill_candidates,
        )

        remaining_batches = list(range(batch_idx + 1, batch_count))
        # Build MCTS config with environment-tunable knobs. We keep
        # `patches_per_iteration` fixed at 1 so that each expansion evaluates
        # a single candidate patch (classic MCTS behaviour).
        def _env_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.strip().lower() in {"1", "true", "yes", "on"}

        def _env_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default

        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return default

        default_ppi = 3 if prompt_mode == "traffic_fix" else 1
        mcts_config = MCTSConfig(
            exploration_constant=_env_float("STEP2_MCTS_C", MCTSConfig().exploration_constant),
            patches_per_iteration=_env_int("STEP2_MCTS_PATCHES_PER_ITER", default_ppi),
            max_depth=_env_int("STEP2_MCTS_MAX_DEPTH", 16),
            log_details=True,
            simplified_logs=False,
            log_uct=_env_bool("STEP2_MCTS_LOG_UCT", True),
            # Early stop after an empty patch only if issues are gone; can be
            # disabled to force continued exploration.
            empty_patch_early_stop=not _env_bool("STEP2_MCTS_NO_EARLY_STOP", False),
            # Optional reward stagnation cutoff; when set, search stops if no
            # reward improvement for N simulations.
            reward_stagnation_limit=(
                _env_int("STEP2_MCTS_STAGNATION", 5) or None
            ),
            patch_step_penalty=_env_float("STEP2_MCTS_PATCH_STEP_PENALTY", 0.05),
            weight_traffic_fix=_env_float("STEP2_MCTS_TRAFFIC_WEIGHT_COVERAGE", MCTSConfig().weight_traffic_fix),
            weight_traffic_new=_env_float("STEP2_MCTS_TRAFFIC_WEIGHT_NEW", MCTSConfig().weight_traffic_new),
            weight_traffic_success=_env_float("STEP2_MCTS_TRAFFIC_WEIGHT_SUCCESS", MCTSConfig().weight_traffic_success),
            weight_traffic_gap=_env_float("STEP2_MCTS_TRAFFIC_WEIGHT_GAP", MCTSConfig().weight_traffic_gap),
            repeat_expansion_on_visit=_env_bool(
                "STEP2_MCTS_REPEAT_EXPANSION",
                True if prompt_mode == "traffic_fix" else False,
            ),
            progressive_widening_k=_env_float("STEP2_MCTS_PW_K", 0.0),
            progressive_widening_alpha=_env_float("STEP2_MCTS_PW_ALPHA", 0.5),
            progressive_widening_min_children=_env_int("STEP2_MCTS_PW_MIN", 1),
            expand_non_improving_children=_env_bool(
                "STEP2_MCTS_EXPAND_NON_IMPROVING",
                True if prompt_mode == "traffic_fix" else False,
            ),
        )

        apply_fn = apply_patch
        if prompt_mode_norm == "traffic_fix" and traffic_patch_sanitize:
            def _safe_apply(tree_arg: Dict[str, Any], patch_arg: Dict[str, Any]) -> Dict[str, Any]:
                # In traffic_fix mode we still sanitize known-safe redundant edits,
                # but we do NOT reject patches (the validator + reward handles bad ones).
                try:
                    meta = patch_arg.get("patch_metadata") if isinstance(patch_arg, dict) else None
                    intent = str(meta.get("intent") or "").strip().lower() if isinstance(meta, dict) else ""
                except Exception:
                    intent = ""
                if not isinstance(patch_arg, dict):
                    return copy.deepcopy(tree_arg)
                candidate = copy.deepcopy(patch_arg)
                if intent == "traffic_payload_fill":
                    candidate = _strip_payload_fill_child_link_updates(tree_arg, candidate)
                return apply_patch(tree_arg, candidate)
            apply_fn = _safe_apply

        # In traffic_fix mode, allow candidate patches to temporarily violate static constraints.
        # If the patch introduces static ERRORs, run the normal syntax-fixing stage first, then
        # continue evaluation on the syntax-fixed tree.
        if prompt_mode_norm == "traffic_fix":
            run_followup = os.getenv("STEP2_TRAFFIC_CANDIDATE_SYNTAX_FIX", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if run_followup:
                base_apply_fn = apply_fn

                def _apply_with_syntax_followup(
                    tree_arg: Dict[str, Any],
                    patch_arg: Dict[str, Any],
                ) -> Dict[str, Any]:
                    patched_tree = base_apply_fn(tree_arg, patch_arg)
                    try:
                        static_report = run_full_validation(patched_tree)
                    except Exception:
                        return patched_tree
                    static_errors = [
                        iss
                        for iss in getattr(static_report, "issues", {}).values()
                        if getattr(iss, "severity", None) == Severity.ERROR
                    ]
                    if not static_errors:
                        return patched_tree

                    try:
                        max_calls = int(os.getenv("STEP2_TRAFFIC_CANDIDATE_SYNTAX_MAX_CALLS", "10"))
                    except Exception:
                        max_calls = 10
                    if max_calls <= 0:
                        return patched_tree

                    try:
                        default_bs = max(1, int(batch_size))
                    except Exception:
                        default_bs = 1
                    try:
                        followup_batch_size = int(
                            os.getenv("STEP2_TRAFFIC_CANDIDATE_SYNTAX_BATCH_SIZE", str(default_bs))
                        )
                    except Exception:
                        followup_batch_size = default_bs

                    try:
                        print(
                            "[TrafficFix][SyntaxFollowup] "
                            f"static_errors={len(static_errors)} "
                            f"batch_size={max(1, followup_batch_size)} "
                            f"max_calls={max_calls}",
                            flush=True,
                        )
                    except Exception:
                        pass

                    try:
                        fixed_tree = mcts_fix_tree(
                            agent,
                            patched_tree,
                            sections,
                            raw_sections=raw_sections,
                            batch_size=max(1, followup_batch_size),
                            max_llm_calls=max(1, max_calls),
                            node_snapshot_dir=node_snapshot_dir,
                            validator_fn=run_full_validation,
                            prompt_mode="fix",
                            fix_history=fix_history,
                        )
                    except Exception as exc:
                        try:
                            print(
                                "[TrafficFix][SyntaxFollowup] "
                                f"failed={type(exc).__name__}:{exc}",
                                flush=True,
                            )
                        except Exception:
                            pass
                        return patched_tree

                    try:
                        final_report = run_full_validation(fixed_tree)
                        final_errors = [
                            iss
                            for iss in getattr(final_report, "issues", {}).values()
                            if getattr(iss, "severity", None) == Severity.ERROR
                        ]
                        print(
                            "[TrafficFix][SyntaxFollowup] "
                            f"done static_errors_after={len(final_errors)}",
                            flush=True,
                        )
                    except Exception:
                        pass

                    return fixed_tree

                apply_fn = _apply_with_syntax_followup

        default_max_sims = 12 if prompt_mode_norm == "traffic_fix" else 5
        outcome, stats = search_for_batch(
            tree,
            batch_idx,
            patch_supplier=supplier,
            validator=validator_fn,
            apply_patch_fn=apply_fn,
            remaining_batches=remaining_batches,
            config=mcts_config,
            max_simulations=_env_int("STEP2_MCTS_MAX_SIMS", default_max_sims),
            evaluation_callback=supplier.on_candidate_evaluated,
            node_snapshot_dir=node_snapshot_dir,
            normalizer=canonicalize_protocol_tree if prompt_mode != "traffic_fix" else _normalize_refinement_tree,
        )

        remaining_llm_calls = max(0, remaining_llm_calls - supplier.calls_made)
        logger.info("Repair attempt budget: used=%s remaining=%s total=%s",
                    supplier.calls_made, remaining_llm_calls, max_llm_calls)
        supplier.persist_cache()
        patch_audit.extend(supplier.evaluation_records)

        if stats.simulations:
            logger.info(
                "Batch %s MCTS stats: simulations=%s, expansions=%s, best_reward=%.3f, best_issues=%s",
                batch_idx + 1,
                stats.simulations,
                stats.expansions,
                stats.best_reward,
                stats.best_issue_count,
            )

        if supplier.evaluation_records:
            total_candidates = len(supplier.evaluation_records)
            positive = sum(1 for record in supplier.evaluation_records if (record.get("reward") or 0) > 0)
            logger.info(
                "Batch %s evaluation summary: candidates=%s, positive_rewards=%s",
                batch_idx + 1,
                total_candidates,
                positive,
            )

        records = list(outcome.records if outcome else [])
        trimmed_negative_suffix_count = 0
        if (
            records
            and prompt_mode_norm != "traffic_fix"
            and _env_bool("STEP2_MCTS_DROP_NEGATIVE_SUFFIX", True)
        ):
            selected_records = list(records)
            while selected_records:
                last_record = selected_records[-1]
                if last_record.reward >= 0:
                    break
                if not last_record.issues_after:
                    break
                selected_records.pop()
                trimmed_negative_suffix_count += 1
            if trimmed_negative_suffix_count:
                logger.info(
                    "Batch %s trimmed %s negative-reward MCTS suffix patch(es) before final validation",
                    batch_idx + 1,
                    trimmed_negative_suffix_count,
                )
                records = selected_records
        path_hashes = [record.action.hash for record in records]
        if records:
            trace_lines = [
                f"[trace] root (issues={len(base_report.issues)}, reward=0.0)",
            ]
            for idx, record in enumerate(records, 1):
                trace_lines.append(f"  └─ [sim {idx}] patch {record.action.hash[:8]}")
                trace_lines.append(f"     • resolved: {len(record.resolved)}")
                trace_lines.append(f"     • introduced: {len(record.introduced)}")
                trace_lines.append(f"     • issues_after: {len(record.issues_after)}")
                total_reward = record.reward + (record.rollout_reward or 0.0)
                reward_line = f"     • reward: {record.reward:.3f}"
                if record.rollout_reward:
                    reward_line += f" (+{record.rollout_reward:.3f} rollout)"
                reward_line += f" = {total_reward:.3f}"
                trace_lines.append(reward_line)
            logger.info("\n".join(trace_lines))
        path_metadata: List[Dict[str, Any]] = []
        for record in records:
            info = supplier.get_action_info(record.action.hash) or {}
            summary = record.action.summary or info.get("summary")
            path_metadata.append(
                {
                    "hash": record.action.hash,
                    "summary": summary,
                    "source": info.get("source"),
                    "attempt": info.get("attempt"),
                    "reward": record.reward,
                    "introduced": len(record.introduced),
                    "resolved": len(record.resolved),
                }
            )
        path_summaries = [meta.get("summary") for meta in path_metadata]
        path_summary = " | ".join(filter(None, path_summaries)) or None
        path_patches = [copy.deepcopy(record.action.patch) for record in records]

        previous_best = best_issue_count
        # Track ERROR-only count to prefer fixes that reduce true errors even if warnings remain
        previous_best_errors = best_error_count

        if outcome is not None:
            if prompt_mode_norm != "traffic_fix":
                current_tree = _normalize_refinement_tree(copy.deepcopy(tree))
                # search_for_batch already applied and validated the selected
                # path.  Replaying patch records from the pre-search tree can
                # reject valid multi-step paths when normalizers changed the
                # intermediate structure, so treat the outcome tree as
                # authoritative.
                candidate_tree = _normalize_refinement_tree(copy.deepcopy(outcome.final_tree))
                applied = bool(records) or candidate_tree != current_tree
            else:
                candidate_tree = _normalize_refinement_tree(copy.deepcopy(outcome.final_tree))
                applied = True
        else:
            candidate_tree = _normalize_refinement_tree(copy.deepcopy(tree))
            applied = False

        candidate_report = validator_fn(candidate_tree)
        candidate_struct_err, candidate_sem_err, candidate_issue_count, candidate_invariant_err = _issue_breakdown(candidate_report)
        candidate_static_struct_err, candidate_static_sem_err, candidate_static_total_err, candidate_static_invariant_err = _issue_breakdown(
            candidate_report, only_static=True
        )
        candidate_error_count = candidate_issue_count
        baseline_doc_score = _doc_accept_score(baseline_report)
        candidate_doc_score = _doc_accept_score(candidate_report)
        baseline_doc_uncovered = sum(baseline_doc_score)
        candidate_doc_uncovered = sum(candidate_doc_score)
        staged_selector_progress = _staged_selector_doc_progress(baseline_report, candidate_report)
        doc_improved = candidate_doc_uncovered < baseline_doc_uncovered
        doc_no_regression = candidate_doc_uncovered <= baseline_doc_uncovered or staged_selector_progress
        candidate_changed = candidate_tree != _normalize_refinement_tree(copy.deepcopy(tree))
        mcts_positive_progress = bool(
            outcome is not None
            and candidate_changed
            and float(getattr(outcome, "reward", 0.0) or 0.0) > 0.0
            and candidate_issue_count <= baseline_total_err
        )

        # Accept if no invariant regression, no increase in struct/sem errors, and some improvement.
        # For traffic_fix, prefer traffic metrics (success/coverage) over coarse issue counts.
        if prompt_mode_norm == "traffic_fix":
            baseline_score = baseline_traffic_score or _traffic_accept_score(baseline_report)
            candidate_score = _traffic_accept_score(candidate_report)
            baseline_scalar = _traffic_scalar_score(baseline_report)
            candidate_scalar = _traffic_scalar_score(candidate_report)
            improved = (candidate_score < baseline_score) or (candidate_scalar > baseline_scalar + 1e-9)
            no_regression = _traffic_static_no_regression(
                baseline_struct=baseline_static_struct_err,
                candidate_struct=candidate_static_struct_err,
                baseline_sem=baseline_static_sem_err,
                candidate_sem=candidate_static_sem_err,
                baseline_invariant=baseline_static_invariant_err,
                candidate_invariant=candidate_static_invariant_err,
            )
            # Guardrail: allow temporary traffic trade-offs, but avoid accepting extreme
            # regressions in parse success that are hard to recover from.
            try:
                max_failed_increase = int(
                    os.getenv("STEP2_TRAFFIC_ACCEPT_MAX_FAILED_INCREASE", "0").strip()
                )
            except Exception:
                max_failed_increase = 0
            max_failed_increase = max(0, max_failed_increase)
            baseline_failed = int(baseline_score[0])
            candidate_failed = int(candidate_score[0])
            within_failed_slack = candidate_failed <= (baseline_failed + max_failed_increase)

            accept_candidate = applied and no_regression and improved and within_failed_slack
            try:
                print(
                    "[TrafficFix][MCTS] "
                    f"section_batch={batch_idx + 1}/{batch_count} "
                    f"sections={batch_start}-{batch_end - 1} "
                    f"accept={accept_candidate} "
                    f"baseline=({_format_traffic_score(baseline_score)}) "
                    f"candidate=({_format_traffic_score(candidate_score)}) "
                    f"baseline_scalar={baseline_scalar:.3f} candidate_scalar={candidate_scalar:.3f} "
                    f"reward={outcome.reward if outcome else None}",
                    flush=True,
                )
            except Exception:
                pass
        else:
            improved = (
                candidate_issue_count < baseline_total_err
                or candidate_struct_err < baseline_struct_err
                or candidate_sem_err < baseline_sem_err
                or doc_improved
                or staged_selector_progress
                or mcts_positive_progress
            )
            no_regression = (
                (candidate_struct_err <= baseline_struct_err or staged_selector_progress)
                and candidate_sem_err <= baseline_sem_err
                and candidate_static_struct_err <= baseline_static_struct_err
                and candidate_static_sem_err <= baseline_static_sem_err
                and candidate_invariant_err <= baseline_invariant_err
                and candidate_static_invariant_err <= baseline_static_invariant_err
                and (doc_no_regression or mcts_positive_progress)
            )
            accept_candidate = applied and no_regression and improved

        if accept_candidate:
            tree = copy.deepcopy(candidate_tree)
            best_tree = copy.deepcopy(candidate_tree)
            best_issue_count = candidate_issue_count
            best_error_count = candidate_error_count
            if prompt_mode_norm == "traffic_fix":
                logger.info(
                    "Batch %s candidate tree accepted (static_struct %s->%s, static_sem %s->%s, total %s->%s)",
                    batch_idx + 1,
                    baseline_static_struct_err,
                    candidate_static_struct_err,
                    baseline_static_sem_err,
                    candidate_static_sem_err,
                    baseline_total_err,
                    candidate_issue_count,
                )
            else:
                logger.info(
                    "Batch %s candidate tree accepted (struct %s->%s, sem %s->%s, total %s->%s)",
                    batch_idx + 1,
                    baseline_struct_err,
                    candidate_struct_err,
                    baseline_sem_err,
                    candidate_sem_err,
                    baseline_total_err,
                    candidate_issue_count,
                )
        else:
            if prompt_mode_norm == "traffic_fix":
                if not _traffic_static_no_regression(
                    baseline_struct=baseline_static_struct_err,
                    candidate_struct=candidate_static_struct_err,
                    baseline_sem=baseline_static_sem_err,
                    candidate_sem=candidate_static_sem_err,
                    baseline_invariant=baseline_static_invariant_err,
                    candidate_invariant=candidate_static_invariant_err,
                ):
                    logger.error(
                        "Batch %s candidate tree regressed static validation (struct %s->%s, sem %s->%s, invariant %s->%s); retaining previous tree",
                        batch_idx + 1,
                        baseline_static_struct_err,
                        candidate_static_struct_err,
                        baseline_static_sem_err,
                        candidate_static_sem_err,
                        baseline_static_invariant_err,
                        candidate_static_invariant_err,
                    )
                else:
                    logger.info(
                        "Batch %s candidate tree unchanged (static_struct %s->%s, static_sem %s->%s, static_invariant %s->%s); retaining previous tree",
                        batch_idx + 1,
                        baseline_static_struct_err,
                        candidate_static_struct_err,
                        baseline_static_sem_err,
                        candidate_static_sem_err,
                        baseline_static_invariant_err,
                        candidate_static_invariant_err,
                    )
            else:
                if (
                    candidate_invariant_err > baseline_invariant_err
                    or candidate_static_invariant_err > baseline_static_invariant_err
                    or (candidate_struct_err > baseline_struct_err and not staged_selector_progress)
                    or candidate_sem_err > baseline_sem_err
                    or candidate_static_struct_err > baseline_static_struct_err
                    or candidate_static_sem_err > baseline_static_sem_err
                ):
                    logger.error(
                        "Batch %s candidate tree regressed validation (struct %s->%s, sem %s->%s, static_struct %s->%s, static_sem %s->%s); retaining previous tree",
                        batch_idx + 1,
                        baseline_struct_err,
                        candidate_struct_err,
                        baseline_sem_err,
                        candidate_sem_err,
                        baseline_static_struct_err,
                        candidate_static_struct_err,
                        baseline_static_sem_err,
                        candidate_static_sem_err,
                    )
                else:
                    logger.info(
                        "Batch %s candidate tree unchanged (struct %s->%s, sem %s->%s, static_struct %s->%s, static_sem %s->%s); retaining previous tree",
                        batch_idx + 1,
                        baseline_struct_err,
                        candidate_struct_err,
                        baseline_sem_err,
                        candidate_sem_err,
                        baseline_static_struct_err,
                        candidate_static_struct_err,
                        baseline_static_sem_err,
                        candidate_static_sem_err,
                    )
            applied = False
            candidate_tree = _normalize_refinement_tree(copy.deepcopy(tree))
            candidate_report = baseline_report
            candidate_issue_count = baseline_total_err

        _persist_batch_tree(batch_idx, tree)

        feedback = _build_patch_feedback(candidate_report.errors, candidate_report.extras, candidate_tree)
        if candidate_issue_count == 0:
            feedback = None

        patch_audit.append(
            {
                "batch": batch_idx,
                "source": "mcts_best",
                "ok": candidate_report.ok,
                "errors": candidate_report.errors,
                "extras": candidate_report.extras,
                "patch_summary": path_summary,
                "path_summary": path_summary,
                "path_hashes": path_hashes,
                "simulations": stats.simulations,
                "best_reward": stats.best_reward,
                "best_issue_count": stats.best_issue_count,
                "terminal_found": stats.terminal_found,
                "applied": applied,
                "accepted": accept_candidate,
                "issue_count": candidate_issue_count,
                "empty_patch_stop": stats.empty_patch_stop,
                "trimmed_negative_suffix_count": trimmed_negative_suffix_count,
            }
        )

        record_patches = path_patches if accept_candidate else []
        all_patch_paths.append(record_patches)
        batch_path_records.append(
            {
                "batch": batch_idx,
                "applied": applied,
                "accepted": accept_candidate,
                "reward": outcome.reward if outcome else None,
                "issue_count": candidate_issue_count,
                "path_hashes": path_hashes,
                "summaries": [meta.get("summary") for meta in path_metadata],
                "path_length": len(records),
                "terminal": outcome.terminal if outcome else False,
                "patches": record_patches,
                "introduced_counts": [len(record.introduced) for record in records],
                "resolved_counts": [len(record.resolved) for record in records],
                "trimmed_negative_suffix_count": trimmed_negative_suffix_count,
            }
        )

        log_mcts_event(
            {
                "event": "batch_summary",
                "batch_index": batch_idx,
                "path_hashes": path_hashes,
                "path_length": len(records),
                "path_reward": outcome.reward if outcome else None,
                "path_summary": path_summary,
                "path_applied": applied,
                "path_accepted": accept_candidate,
                "issue_count": candidate_issue_count,
                "tree_node_count": len(candidate_tree.get("nodes", [])),
                "tree_edge_count": len(candidate_tree.get("edges", [])),
                "simulations": stats.simulations,
                "best_reward": stats.best_reward,
                "best_issues": stats.best_issue_count,
                "terminal": outcome.terminal if outcome else False,
                "empty_patch_stop": stats.empty_patch_stop,
            }
        )

        if stats.terminal_found and candidate_issue_count == 0:
            logger.info("Batch %s reached zero-issue state via MCTS search", batch_idx + 1)

    tree = copy.deepcopy(best_tree)

    total_applied_patches = sum(len(path) for path in all_patch_paths)
    logger.info("Refinement complete. Applied %s patches.", total_applied_patches)
    logger.info(
        "Final tree: %s nodes, %s edges",
        len(tree.get("nodes", [])),
        len(tree.get("edges", [])),
    )

    flattened_patches: List[Dict[str, Any]] = [patch for path in all_patch_paths for patch in path]
    agent._save_to_cache(
        "patch_history.json",
        {
            "paths": batch_path_records,
            "patches": flattened_patches,
            "timestamp": datetime.now().isoformat(),
            "total_batches": batch_count,
            "total_applied_patches": total_applied_patches,
            "repair_attempt_budget": max(0, int(max_llm_calls)),
            "repair_attempts_used": max(0, int(max_llm_calls)) - remaining_llm_calls,
        },
    )

    agent._save_to_cache(
        "patch_validation_report.json",
        {
            "timestamp": datetime.now().isoformat(),
            "entries": patch_audit,
            "applied_patches": total_applied_patches,
        },
    )

    log_mcts_event(
        {
            "event": "refinement_complete",
            "total_batches": batch_count,
            "applied_patches": total_applied_patches,
            "final_node_count": len(tree.get("nodes", [])),
            "final_edge_count": len(tree.get("edges", [])),
            "remaining_issue_count": best_issue_count,
        }
    )

    return tree


def refine_tree_with_raw_data(
    agent: "EnhancedPureAIAgent",
    initial_tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    raw_sections: Optional[Sequence[Dict[str, Any]]] = None,
    batch_size: int = 5,
    max_llm_calls: int = 100000,
    *,
    sequential_attempts: Optional[int] = None,
) -> Dict[str, Any]:
    """Run sequential per-section refinement followed by an optional MCTS fix stage."""

    enriched_sections = [copy.deepcopy(s) for s in (sections or [])]
    attempts = sequential_attempts
    if attempts is None:
        try:
            attempts = int(os.getenv("STEP2_SEQUENTIAL_ATTEMPTS", "3"))
        except ValueError:
            attempts = 3

    logger.info("Sequential refinement: max attempts per section set to %s", attempts)

    refined_tree = sequential_refine_tree(
        agent,
        initial_tree,
        enriched_sections,
        raw_sections=None,
        max_section_attempts=max(1, attempts),
    )

    post_report = run_full_validation(refined_tree)
    remaining_issues = len(post_report.issues)
    if remaining_issues == 0 and post_report.ok:
        logger.info("Sequential refinement resolved all validator issues; skipping MCTS fix stage")
        return refined_tree

    if os.getenv("STEP2_SKIP_MCTS_FIX", "0") == "1":
        logger.info("STEP2_SKIP_MCTS_FIX=1; returning tree after sequential refinement without MCTS fix")
        return refined_tree

    logger.info(
        "Sequential refinement complete with %s outstanding issues; invoking MCTS fix stage",
        remaining_issues,
    )

    fixed_tree = mcts_fix_tree(
        agent,
        refined_tree,
        enriched_sections,
        raw_sections=None,
        batch_size=batch_size,
        max_llm_calls=max_llm_calls,
    )

    return fixed_tree
