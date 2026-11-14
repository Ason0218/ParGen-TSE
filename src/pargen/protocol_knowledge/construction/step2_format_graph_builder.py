#!/usr/bin/env python3

import copy
import json
import os
import logging
import multiprocessing
import re
import requests
import time
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set, TYPE_CHECKING
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
import hashlib
import struct
from enum import Enum

from ...config_manager import load_api_keys
from ...paths import DEFAULT_API_CONFIG, STEP2_CACHE_DIR, STEP2_FIX_CACHE_DIR
from ..tree_utils import normalize_protocol_tree, synchronize_parent_child_links
from ..validation.syntax_validator import validate_protocol_tree, Severity
from ..validation.static_repair import normalize_parent_size_references
from ..validation.constraint_attachment import attach_constraint_obligations, build_compact_constraint_context
from ..repair.refinement import mcts_fix_tree, apply_patch as fix_apply_patch, run_full_validation
from ..message_validation.interpreter import DynamicTreeInterpreter
from ..message_validation.semantic_validator import _ctx_to_record, _load_packets_from_path, run_hybrid_validation
from ..static_layout import static_property
from .core_ir import canonicalize_protocol_tree, add_request_response_variants

if TYPE_CHECKING:
    # Only for type hints; avoid runtime circular import
    from ..repair.agent import EnhancedPureAIAgent as FixLLMAgent

_NODE_TYPE_ALIAS = {
    # Core canonical types
    "protocol": "protocol",
    "header": "header",
    "field": "field",
    "selector": "selector",
    "variant": "variant",
    "payload": "payload",
    "container": "container",
    # Common synonyms
    "body": "payload",
    "segment": "container",
    "composite": "container",
}

_STAGE2_INPUT_SIGNATURE_SCHEMA = 4
DEFAULT_AI_READ_TIMEOUT_SEC = 900
DEFAULT_AI_PROVIDER_WALL_TIMEOUT_SEC = 960.0
# Initial-tree output budget (nodes / edges). The defaults are the compact scaffold the
# refine batches then expand; a profile may raise them with STEP2_INITIAL_TREE_MAX_NODES /
# STEP2_INITIAL_TREE_MAX_EDGES (see _initial_tree_limits()).
_INITIAL_TREE_SCHEMA_MAX_NODES = 32
_INITIAL_TREE_SCHEMA_MAX_EDGES = 48

_ROUTING_SEMANTIC_FIX_CODES = {
    "NON_EXCLUSIVE_VARIANT_CONDITIONS",
    "DUPLICATE_VARIANT_CONDITIONS",
}


def _initial_tree_max_json_retries() -> int:
    raw = (os.getenv("STAGE_B_INITIAL_MAX_ATTEMPTS") or "").strip()
    if not raw:
        return 3
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _env_flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _provider_failure_fail_fast_enabled() -> bool:
    return _env_flag("STEP2_FAIL_FAST_ON_PROVIDER_FAILURE", "0")


def _initial_tree_limits() -> Tuple[int, int]:
    """(max nodes, max edges) for the initial tree: schema hard limits and prompt text."""
    return (_env_positive_int("STEP2_INITIAL_TREE_MAX_NODES", _INITIAL_TREE_SCHEMA_MAX_NODES),
            _env_positive_int("STEP2_INITIAL_TREE_MAX_EDGES", _INITIAL_TREE_SCHEMA_MAX_EDGES))


def _initial_tree_expands_tables() -> bool:
    """STEP2_INITIAL_TREE_EXPAND_TABLES=1: the initial tree may expand documented option/TLV
    tables instead of deferring them."""
    return (os.getenv("STEP2_INITIAL_TREE_EXPAND_TABLES") or "").strip().lower() in {"1", "true", "yes", "on"}


def _formula_implications_enabled() -> bool:
    """STEP2_FORMULA_IMPLICATIONS=1: prompts describe conditional rules `A => B`
    (cross-field rules); tree_utils.desugar_implication makes them executable."""
    return (os.getenv("STEP2_FORMULA_IMPLICATIONS") or "").strip().lower() in {"1", "true", "yes", "on"}


def _initial_tree_budget_block() -> str:
    """The initial-tree output budget section of the prompt.

    Default: the compact scaffold (option/code tables deferred to refinement or
    coverage-obligation passes). With STEP2_INITIAL_TREE_EXPAND_TABLES=1 the model expands
    documented option/TLV tables in the initial tree, within the (raised) node budget.
    """
    max_nodes, max_edges = _initial_tree_limits()
    head = (
        "## INITIAL TREE OUTPUT BUDGET\n"
        "The initial tree is a compact scaffold, not the final knowledge tree:\n"
        f"- Keep the initial tree at or below {max_nodes} nodes and {max_edges} edges; the response schema enforces these hard limits.\n"
        "- Include shared headers, selector/code fields, generic repeated containers, and named variant placeholders.\n"
    )
    if _initial_tree_expands_tables():
        return head + (
            "- Expand documented option/TLV/code tables that the documentation defines in this input: the tag/type selector, "
            "one variant per documented option or code with its length and data fields (`length_of` from the length octet), "
            "single-octet options as zero-body variants, and an unknown/fallback variant. Split documented bit fields "
            "(flags, reserved bits) into their own leaf fields when the text defines them one by one.\n"
            "- Attach every constraint the documentation states for a field you include (enum/range/formula), "
            "including conditional rules between fields.\n"
            "- Still prefer `size_bits=\"variable\"` over guessed sizes, and leave undocumented bodies minimal; do not invent "
            "fields the documentation does not define.\n"
            "- Do NOT collapse named selector values into generic broad-range variants such as `Normal_Request_Data`; broad "
            "range variants are only for explicitly documented reserved/unassigned/catch-all ranges.\n"
            "- Name variants from the documented selector value and section title, e.g. `OP03_Read_Status_Request`.\n"
        )
    return head + (
        "- Do NOT expand option/code tables into detailed body fields in the initial tree; detailed bodies belong to refinement or coverage-obligation passes.\n"
        "- For TLV/option/code tables, emit the selector/tag/code field, a compact typed variant shell for important observed or documented values, and an unknown/fallback variant.\n"
        "- Prefer `size_bits=\"variable\"` and empty/minimal variant bodies over large nested field expansions.\n"
        "Initial output MUST stay compact:\n"
        "- Prefer an empty/minimal variant body over a detailed per-variant subtree.\n"
        "- Do NOT expand every table row into leaf fields in the initial tree; leave that to the refinement batches.\n"
        "- Include only shared fixed headers, selector fields, variant containers, and the smallest fields needed to anchor layout.\n"
        "- If observed traffic is present, do NOT expand every unobserved sibling variant in the initial tree. If no traffic is present, create no more than 8 representative named placeholder variants across documented directions; later refinement receives the full section set and adds the deferred variants.\n"
        "- Do NOT collapse named selector values into generic broad-range variants such as `Normal_Request_Data` / `Normal_Response_Data`; broad range variants are only for explicitly documented reserved/unassigned/catch-all ranges.\n"
        "- Name placeholder variants from the documented selector value and section title, e.g. `OP03_Read_Status_Request`; avoid generic names that hide the selector value.\n"
        "- Never try to achieve complete selector coverage in the initial response. Omit deferred variants rather than merging them into an inaccurate broad-range variant; refinement is responsible for complete coverage.\n"
        "It is expected for variant bodies to be minimal/empty at this stage; the refinement stage adds detailed fields later.\n"
    )


def _formula_rules_block() -> str:
    """The `formula:` rules of the constraint DSL section of the prompts."""
    rules = ("- **Formula assignment**: Every `formula:` constraint must be written as `formula: value = <expression>`, "
             "for example `formula: value = min(val(37), 252 * 8)`.\n")
    if _formula_implications_enabled():
        rules = (
            "- **Formula assignment**: a `formula:` constraint that defines this field's value is written as "
            "`formula: value = <expression>`, for example `formula: value = min(val(37), 252 * 8)`.\n"
            "- **Conditional (cross-field) rules**: when the documentation states that one field's value restricts "
            "another (\"if DF is set, MF must be 0 and Fragment Offset must be 0\"; \"when MF is set the data length is a "
            "multiple of 8\"), write an implication over node ids: "
            "`formula: (val(DF_ID) == 1) => (val(MF_ID) == 0 and val(OFFSET_ID) == 0)`, "
            "`formula: (val(MF_ID) == 1) => ((val(LEN_ID) - 4 * val(IHL_ID)) % 8 == 0)`. "
            "Attach the rule to the referenced leaf field that comes LAST on the wire (the rule is checked when that field is "
            "parsed, so every val() it uses must already be parsed; a rule on an earlier field or on a container is not "
            "checked). Use `=>` once per rule, `val(ID)` for every field, and only fields that exist in the tree.\n"
        )
    return rules


def _env_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int((os.getenv(name) or str(default)).strip()))
    except Exception:
        return max(1, int(default))


def _env_positive_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or str(default)).strip()
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if value > 0 else float(default)


def _truncate_prompt_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 32)].rstrip() + " ... [truncated]"


_INITIAL_SECTION_KEEP_KEYS = (
    "section_index",
    "number",
    "title",
    "heading",
    "section_title",
    "name",
    "section_type",
    "content",
    "summary",
    "packet_formats",
    "field_definitions",
    "constraints",
)

_INITIAL_PROMPT_FORMAT_KEEP_KEYS = (
    "format_name",
    "name",
    "description",
    "structure_type",
    "total_size",
    "size",
    "selector_field",
    "selector_value",
    "type_value",
    "code",
    "kind",
    "fields",
    "constraints",
)

_INITIAL_PROMPT_FIELD_KEEP_KEYS = (
    "field_name",
    "name",
    "data_type",
    "bit_position",
    "byte_position",
    "bit_start",
    "size",
    "size_bits",
    "description",
    "valid_values",
    "enum_values",
    "constraints",
    "condition",
    "condition_on",
    "length_of",
    "repeat_count",
)


def _compact_prompt_scalar(value: Any, *, text_limit: int) -> Any:
    if isinstance(value, str):
        return _truncate_prompt_text(value, limit=text_limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate_prompt_text(value, limit=text_limit)


def _compact_prompt_value(value: Any, *, text_limit: int, list_limit: int) -> Any:
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, child in value.items():
            if key in {
                "cost_info",
                "api_used",
                "processing_method",
                "position",
                "source_file",
                "source_paragraphs",
                "curation_note",
                "children",
                "level",
            }:
                continue
            compact[key] = _compact_prompt_value(child, text_limit=text_limit, list_limit=list_limit)
        return compact
    if isinstance(value, list):
        return [
            _compact_prompt_value(item, text_limit=text_limit, list_limit=list_limit)
            for item in value[:list_limit]
        ]
    return _compact_prompt_scalar(value, text_limit=text_limit)


def _compact_prompt_field(field: Any, *, text_limit: int) -> Any:
    if not isinstance(field, dict):
        return _compact_prompt_scalar(field, text_limit=text_limit)
    compact: Dict[str, Any] = {}
    for key in _INITIAL_PROMPT_FIELD_KEEP_KEYS:
        if key not in field:
            continue
        compact[key] = _compact_prompt_value(field[key], text_limit=text_limit, list_limit=16)
    return compact


def _compact_prompt_packet_format(fmt: Any, *, text_limit: int, max_fields: int) -> Any:
    if not isinstance(fmt, dict):
        return _compact_prompt_scalar(fmt, text_limit=text_limit)
    compact: Dict[str, Any] = {}
    for key in _INITIAL_PROMPT_FORMAT_KEEP_KEYS:
        if key not in fmt:
            continue
        if key == "fields" and isinstance(fmt.get(key), list):
            compact[key] = [
                _compact_prompt_field(field, text_limit=text_limit)
                for field in fmt[key][:max_fields]
            ]
            if len(fmt[key]) > max_fields:
                compact["fields_truncated_count"] = len(fmt[key]) - max_fields
        else:
            compact[key] = _compact_prompt_value(fmt[key], text_limit=text_limit, list_limit=24)
    return compact


# Sections that carry verbatim source text: page chunks (scripts/protocol_source_corpus.py) and
# specification sections cut at their headings (tools/cut_spec_sections.py). They are shown
# whole, never reduced to a 180-character excerpt.
LOSSLESS_SOURCE_METHODS = ('pdf_text_pages_v1', 'spec_text_sections_v1')


def _compact_initial_prompt_sections(sections: Sequence[Any]) -> List[Dict[str, Any]]:
    """Render Stage-A sections for the initial tree prompt without audit metadata.

    The full fixed subset remains available to later refinement, scaffold, and
    coverage-obligation passes.  This compact view keeps only information that
    can affect on-wire tree construction and avoids sending Stage-A provenance,
    cost, page-position, and raw audit fields to the initial provider call.
    """

    text_limit = _env_positive_int("STEP2_INITIAL_PROMPT_TEXT_FIELD_CHARS", 180)
    max_formats = _env_positive_int("STEP2_INITIAL_PROMPT_PACKET_FORMATS_MAX", 4)
    max_format_fields = _env_positive_int("STEP2_INITIAL_PROMPT_PACKET_FORMAT_FIELDS_MAX", 12)
    max_field_defs = _env_positive_int("STEP2_INITIAL_PROMPT_FIELD_DEFS_MAX", 16)
    compacted: List[Dict[str, Any]] = []
    for idx, section in enumerate(sections or []):
        # Subset adapter validates origin before this view. Keep local citation
        # indices distinct from full-corpus indices and retain all source text.
        if isinstance(section, dict) and section.get('retrieval_scope_id'):
            if section.get('section_index') != idx:
                raise ValueError('Subset local citation index differs from prompt order')
            compacted.append({key: section[key] for key in
                ('section_index', 'source_section_index', 'source_file', 'source_page',
                 'source_block', 'title', 'content', 'retrieval_scope_id', 'retrieval_scope')
                if key in section})
            continue
        # Lossless source-text chunks are already bounded by the corpus builder.
        # Do not reduce each page to its 180-character running header.
        if isinstance(section, dict) and section.get('extraction_method') in LOSSLESS_SOURCE_METHODS:
            compacted.append({key: section[key] for key in
                              ('source_file', 'source_page', 'title', 'content')})
            continue
        if not isinstance(section, dict):
            compacted.append({"section_index": idx, "content": _truncate_prompt_text(section, limit=text_limit)})
            continue

        out: Dict[str, Any] = {"section_index": idx}
        for key in _INITIAL_SECTION_KEEP_KEYS:
            if key in {"section_index", "packet_formats", "field_definitions"}:
                continue
            if key not in section:
                continue
            value = section.get(key)
            if value in (None, "", [], {}):
                continue
            out[key] = _compact_prompt_value(value, text_limit=text_limit, list_limit=24)

        prompt_content = section.get("prompt_content")
        if isinstance(prompt_content, str) and prompt_content.strip():
            out["content"] = _truncate_prompt_text(prompt_content, limit=text_limit)
        elif "content" in out:
            out["content"] = _truncate_prompt_text(out["content"], limit=text_limit)
        elif "summary" in out:
            out["content"] = out["summary"]

        packet_formats = section.get("packet_formats")
        if isinstance(packet_formats, list) and packet_formats:
            out["packet_formats"] = [
                _compact_prompt_packet_format(fmt, text_limit=text_limit, max_fields=max_format_fields)
                for fmt in packet_formats[:max_formats]
            ]
            if len(packet_formats) > max_formats:
                out["packet_formats_truncated_count"] = len(packet_formats) - max_formats

        field_definitions = section.get("field_definitions")
        if isinstance(field_definitions, list) and field_definitions:
            out["field_definitions"] = [
                _compact_prompt_field(field, text_limit=text_limit)
                for field in field_definitions[:max_field_defs]
            ]
            if len(field_definitions) > max_field_defs:
                out["field_definitions_truncated_count"] = len(field_definitions) - max_field_defs

        compacted.append(out)
    return compacted


def _run_with_wall_timeout(fn, *, timeout_sec: float, label: str):
    if timeout_sec <= 0:
        return fn()
    result: Dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # propagate provider exceptions to caller
            result["exception"] = exc

    worker = threading.Thread(
        target=_target,
        name=f"provider-wall-timeout-{label}",
        daemon=True,
    )
    worker.start()
    worker.join(timeout_sec)
    if worker.is_alive():
        raise TimeoutError(f"{label} exceeded wall-clock timeout {timeout_sec:g}s")
    if "exception" in result:
        raise result["exception"]
    return result.get("value")


def _issue_code_name(issue: Any) -> str:
    code = getattr(issue, "code", None)
    if isinstance(code, Enum):
        return code.name
    return str(code or "")


def _blocking_static_issue_count(report: Any) -> int:
    issues = getattr(report, "issues", {}) or {}
    if not isinstance(issues, dict):
        return 0
    return sum(1 for issue in issues.values() if _issue_code_name(issue) in _ROUTING_SEMANTIC_FIX_CODES)


def _should_run_static_fix(report: Any) -> bool:
    if not getattr(report, "ok", False):
        return True
    return _blocking_static_issue_count(report) > 0


def _error_issue_ids(report: Any) -> Set[str]:
    """Ids of ERROR-severity issues; the key of report.issues is the stable issue id."""
    ids: Set[str] = set()
    for iid, issue in (getattr(report, "issues", {}) or {}).items():
        severity = getattr(issue, "severity", None)
        name = str(getattr(severity, "name", severity or "")).upper()
        if name == "ERROR":
            ids.add(str(iid))
    return ids


def _hard_error_codes(report: Any) -> List[str]:
    """Codes of the ERROR-severity issues, in report order; a rejection that only
    records a count cannot be diagnosed afterwards without re-running the model."""
    codes: List[str] = []
    for issue in getattr(report, "errors", []) or []:
        code = getattr(issue, "code", None)
        codes.append(str(code) if code else "UNCODED")
    return codes


_ANCHOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "bit",
    "bits",
    "byte",
    "bytes",
    "field",
    "fields",
    "format",
    "header",
    "in",
    "is",
    "of",
    "on",
    "packet",
    "section",
    "the",
    "to",
    "type",
    "value",
    "with",
}


def _protocol_tree_payload(tree: Dict[str, Any]) -> Dict[str, Any]:
    payload = tree.get("protocol_tree") if isinstance(tree, dict) else None
    return payload if isinstance(payload, dict) else tree


def _nodes_by_id(tree: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    payload = _protocol_tree_payload(tree)
    nodes = payload.get("nodes") if isinstance(payload, dict) else []
    lookup: Dict[str, Dict[str, Any]] = {}
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        lookup[str(node_id)] = node
    return lookup


def _anchor_tokens_from_text(text: Any) -> Set[str]:
    tokens = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+", str(text or ""))
        if len(tok) > 1
    }
    expanded: Set[str] = set()
    for tok in tokens:
        expanded.add(tok)
        if any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
            expanded.update(part for part in re.findall(r"[A-Za-z]+|[0-9]+", tok) if len(part) > 1)
    return {tok for tok in expanded if tok not in _ANCHOR_STOPWORDS}


def _traffic_anchor_tokens(tree: Dict[str, Any], anchor_records: Sequence[Dict[str, Any]]) -> Set[str]:
    node_lookup = _nodes_by_id(tree)
    tokens: Set[str] = set()
    for record in anchor_records or []:
        if not isinstance(record, dict) or not record.get("success"):
            continue
        ids: Set[str] = set(str(nid) for nid in record.get("parsing_path") or [])
        for selector_id, variants in (record.get("active_variants") or {}).items():
            ids.add(str(selector_id))
            ids.update(str(variant_id) for variant_id in variants or [])
        typed_values = record.get("typed_values") or {}
        for node_id in ids:
            node = node_lookup.get(str(node_id)) or {}
            tokens.update(_anchor_tokens_from_text(node.get("name")))
            value = typed_values.get(str(node_id))
            if value is not None:
                tokens.update(_anchor_tokens_from_text(value))
    return tokens


def _section_anchor_text(section: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("number", "title", "heading", "section_title", "name", "content", "summary"):
        value = section.get(key) if isinstance(section, dict) else None
        if value:
            parts.append(str(value))
    for key in ("packet_formats", "field_definitions", "selector_obligations", "coverage_obligations"):
        value = section.get(key) if isinstance(section, dict) else None
        if value:
            try:
                parts.append(json.dumps(value, ensure_ascii=False))
            except Exception:
                parts.append(str(value))
    return "\n".join(parts)


def _traffic_anchor_locate_order(
    sections: Sequence[Dict[str, Any]],
    tree: Dict[str, Any],
    anchor_records: Sequence[Dict[str, Any]],
) -> List[int]:
    """Rank sections by overlap with the observed traffic path and active variants."""

    anchor_tokens = _traffic_anchor_tokens(tree, anchor_records)
    if not anchor_tokens:
        return list(range(len(sections)))

    scored: List[Tuple[int, int]] = []
    for idx, section in enumerate(sections):
        text_tokens = _anchor_tokens_from_text(_section_anchor_text(section))
        title_tokens = _anchor_tokens_from_text(
            " ".join(
                str(section.get(key) or "")
                for key in ("title", "heading", "section_title", "name")
                if isinstance(section, dict)
            )
        )
        packet_tokens = _anchor_tokens_from_text(section.get("packet_formats") if isinstance(section, dict) else "")
        overlap = anchor_tokens.intersection(text_tokens)
        score = len(overlap) + 4 * len(anchor_tokens.intersection(title_tokens)) + 2 * len(anchor_tokens.intersection(packet_tokens))
        scored.append((score, idx))

    if not scored or max(score for score, _idx in scored) <= 0:
        return list(range(len(sections)))
    return [idx for _score, idx in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _reorder_by_indices(seq: Optional[Sequence[Dict[str, Any]]], order: Sequence[int]) -> Optional[List[Dict[str, Any]]]:
    if seq is None:
        return None
    if len(seq) != len(order):
        return list(seq)
    return [seq[idx] for idx in order]


def _traffic_anchor_reorder_remaining(
    sections: Sequence[Dict[str, Any]],
    raw_sections: Optional[Sequence[Dict[str, Any]]],
    tree: Dict[str, Any],
    anchor_records: Sequence[Dict[str, Any]],
    *,
    processed_count: int,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], List[int]]:
    """Reorder only unprocessed sections once an observed traffic anchor becomes available."""

    section_list = list(sections)
    raw_list = list(raw_sections) if raw_sections is not None else None
    remaining_start = min(max(0, processed_count), len(section_list))
    remaining_sections = section_list[remaining_start:]
    if not remaining_sections:
        return section_list, raw_list, []

    order = _traffic_anchor_locate_order(remaining_sections, tree, anchor_records)
    if order == list(range(len(remaining_sections))):
        return section_list, raw_list, order

    reordered_remaining = _reorder_by_indices(remaining_sections, order) or list(remaining_sections)
    section_list = section_list[:remaining_start] + reordered_remaining
    if raw_list is not None and len(raw_list) == len(section_list):
        remaining_raw = raw_list[remaining_start:]
        reordered_raw = _reorder_by_indices(remaining_raw, order) or list(remaining_raw)
        raw_list = raw_list[:remaining_start] + reordered_raw
    return section_list, raw_list, order


def _traffic_locate_payload(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    payload = report.get("traffic_locate")
    if isinstance(payload, dict):
        return payload
    return report


def _traffic_locate_indices(value: Any, *, section_count: int) -> List[int]:
    out: List[int] = []
    items = value if isinstance(value, list) else ([] if value is None else [value])
    for item in items:
        raw_idx = item
        if isinstance(item, dict):
            raw_idx = item.get("section_index", item.get("index", item.get("section")))
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if 0 <= idx < section_count and idx not in out:
            out.append(idx)
    return out


_TRAFFIC_LOCATE_ACTIVE_KEYS: Tuple[str, ...] = (
    "active_core_sections",
    "header_sections",
    "selector_sections",
    "active_variant_sections",
    "supporting_tlv_or_ie_sections",
    "active_supporting_sections",
)


_TRAFFIC_LOCATE_LEGACY_FALLBACK_KEYS: Tuple[str, ...] = (
    "supporting_sections",
    "relevant_sections",
)


_TRAFFIC_LOCATE_COVERAGE_KEYS: Tuple[str, ...] = (
    "coverage_only_sections",
    "deferred_variant_sections",
    "deferred_sections",
)

def _collect_traffic_locate_indices(
    payload: Dict[str, Any],
    keys: Sequence[str],
    *,
    section_count: int,
) -> List[int]:
    selected: List[int] = []
    for key in keys:
        for idx in _traffic_locate_indices(payload.get(key), section_count=section_count):
            if idx not in selected:
                selected.append(idx)
    return selected


def _traffic_locate_section_order(
    sections: Sequence[Dict[str, Any]],
    locate_report: Optional[Dict[str, Any]],
) -> List[int]:
    """Order initial sections using the LLM traffic-location report.
    
    Preserve document order when the report is absent or invalid."""

    section_count = len(sections)
    payload = _traffic_locate_payload(locate_report)
    if not payload:
        return list(range(section_count))

    ordered = _collect_traffic_locate_indices(
        payload,
        _TRAFFIC_LOCATE_ACTIVE_KEYS,
        section_count=section_count,
    )
    if not ordered:
        ordered = _collect_traffic_locate_indices(
            payload,
            _TRAFFIC_LOCATE_LEGACY_FALLBACK_KEYS,
            section_count=section_count,
        )

    if not ordered:
        return list(range(section_count))

    for idx in range(section_count):
        if idx not in ordered:
            ordered.append(idx)
    return ordered


def _traffic_locate_selected_indices(
    locate_report: Optional[Dict[str, Any]],
    *,
    section_count: int,
) -> List[int]:
    payload = _traffic_locate_payload(locate_report)
    if not payload:
        return []

    selected = _collect_traffic_locate_indices(
        payload,
        _TRAFFIC_LOCATE_ACTIVE_KEYS,
        section_count=section_count,
    )
    if selected:
        return selected
    # Backward compatibility for old cached locate reports that predate the
    # active/core vs coverage-only split.
    return _collect_traffic_locate_indices(
        payload,
        _TRAFFIC_LOCATE_LEGACY_FALLBACK_KEYS,
        section_count=section_count,
    )


def _traffic_locate_packet_evidence_requires_active(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    raw_bytes = str(item.get("raw_bytes") or "").strip()
    packet_offsets = item.get("packet_offsets")
    has_offsets = bool(packet_offsets) if isinstance(packet_offsets, list) else bool(str(packet_offsets or "").strip())
    if not raw_bytes and not has_offsets:
        return False
    text = str(item.get("reason") or item.get("spec_evidence") or item.get("note") or "").lower()
    if any(marker in text for marker in ("unobserved", "not present", "not in this packet", "deferred only")):
        return False
    return True


def _traffic_locate_section_title(section: Any) -> str:
    if not isinstance(section, dict):
        return str(section or "")
    for key in ("title", "heading", "section_title", "name", "number"):
        value = section.get(key)
        if value:
            return str(value)
    return str(section.get("content") or section.get("summary") or "")[:120]


def _traffic_locate_coverage_conflict_reason(
    payload: Dict[str, Any],
    *,
    sections: Optional[Sequence[Dict[str, Any]]],
    active_indices: Sequence[int],
    coverage_indices: Sequence[int],
) -> Optional[str]:
    if not sections or not coverage_indices:
        return None

    active = set(active_indices)
    coverage = {idx for idx in coverage_indices if idx not in active}
    if not coverage:
        return None

    evidence_items = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        idx = _traffic_locate_indices(item.get("section_index"), section_count=len(sections))
        if not idx:
            continue
        section_idx = idx[0]
        if section_idx not in active and _traffic_locate_packet_evidence_requires_active(item):
            title = _traffic_locate_section_title(sections[section_idx])
            return (
                f"packet evidence cites section {section_idx} ({title}) but it is not "
                "in the active/core section set"
            )
    return None


def _node_source_section_indices(node: Dict[str, Any], *, section_count: int) -> List[int]:
    values: List[Any] = []
    for key in ("source_section_ids", "source_sections", "source_section_id", "section_ids", "section_index"):
        if key in node:
            values.append(node.get(key))
    out: List[int] = []
    for value in values:
        for idx in _traffic_locate_indices(value, section_count=section_count):
            if idx not in out:
                out.append(idx)
    return out


def _scope_section_items(sections: Any) -> List[Any]:
    if not sections:
        return []
    if isinstance(sections, dict):
        for key in ("sections", "document_sections", "items"):
            value = sections.get(key)
            if isinstance(value, list):
                return list(value)
        return [sections]
    if isinstance(sections, (list, tuple)):
        return list(sections)
    return [sections]


def _section_direction_terms(sections: Any) -> Set[str]:
    section_items = _scope_section_items(sections)
    if not section_items:
        return set()
    roles: Set[str] = set()
    for section in section_items:
        if isinstance(section, dict):
            try:
                text = json.dumps(section, ensure_ascii=False)
            except Exception:
                text = " ".join(str(section.get(key) or "") for key in ("title", "heading", "content", "summary"))
        else:
            text = str(section or "")
        lowered = text.lower()
        if re.search(r"\brequests?\b|\brequest[-_ ]?pdu\b|\brequest[-_ ]?message\b", lowered):
            roles.add("request")
        if re.search(r"\bresponses?\b|\bresponse[-_ ]?pdu\b|\bresponse[-_ ]?message\b", lowered):
            roles.add("response")
        if roles == {"request", "response"}:
            break
    return roles


def _single_direction_scaffold_rejection_reason(
    tree: Optional[Dict[str, Any]],
    *,
    sections: Any,
) -> Optional[str]:
    if not _env_flag("STEP2_INITIAL_DIRECTION_SCOPE_GUARD", "1"):
        return None
    if not tree or _section_direction_terms(sections) != {"request", "response"}:
        return None
    tree_payload = _protocol_tree_payload(tree)
    nodes = tree_payload.get("nodes") if isinstance(tree_payload, dict) else []
    if not isinstance(nodes, list) or not nodes:
        return None

    scaffold_types = {"protocol", "message", "header", "container", "selector"}
    scaffold_message_types: Set[str] = set()
    variant_message_types: Set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = canonicalize_node_type(str(node.get("node_type") or ""))
        message_type = str(node.get("message_type") or "").strip().lower()
        if message_type not in {"request", "response", "bidirectional"}:
            continue
        if node_type in scaffold_types:
            scaffold_message_types.add(message_type)
        if node_type == "variant":
            variant_message_types.add(message_type)

    directional_scaffold = {mt for mt in scaffold_message_types if mt in {"request", "response"}}
    has_bidirectional_scaffold = "bidirectional" in scaffold_message_types
    if len(directional_scaffold) == 1 and not has_bidirectional_scaffold:
        direction = next(iter(directional_scaffold))
        return (
            "message_type scope collapsed to "
            f"{direction!r} even though the source sections describe both request and response wire layouts; "
            "keep the protocol root, shared headers, and shared selectors bidirectional unless an explicit "
            "single-direction target is supplied"
        )

    directional_variants = {mt for mt in variant_message_types if mt in {"request", "response"}}
    if (
        not has_bidirectional_scaffold
        and not directional_scaffold
        and len(directional_variants) == 1
        and len(variant_message_types) > 0
    ):
        direction = next(iter(directional_variants))
        return (
            "message_type scope collapsed to "
            f"{direction!r} variants with no bidirectional shared scaffold even though the source sections "
            "describe both request and response wire layouts"
        )
    return None


def _initial_tree_scope_rejection_reason(
    tree: Optional[Dict[str, Any]],
    locate_report: Optional[Dict[str, Any]],
    *,
    sections: Any,
) -> Optional[str]:
    section_items = _scope_section_items(sections)
    direction_scope_issue = _single_direction_scaffold_rejection_reason(tree, sections=sections)
    if direction_scope_issue:
        return direction_scope_issue

    payload = _traffic_locate_payload(locate_report)
    if not tree or not payload or not section_items:
        return None
    section_count = len(section_items)
    active_indices = _traffic_locate_selected_indices(locate_report, section_count=section_count)
    if not active_indices:
        return None
    coverage_indices = _traffic_locate_coverage_only_indices(
        locate_report,
        section_count=section_count,
        active_indices=active_indices,
    )

    tree_payload = _protocol_tree_payload(tree)
    nodes = tree_payload.get("nodes") if isinstance(tree_payload, dict) else []
    if not isinstance(nodes, list):
        return None

    active_set = set(active_indices)
    coverage_set = set(coverage_indices)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = canonicalize_node_type(str(node.get("node_type") or ""))
        parent_id = node.get("parent_id")
        if node_type == "protocol" or parent_id is None:
            continue
        node_name = str(node.get("name") or node.get("node_id") or "<unnamed>")
        source_indices = _node_source_section_indices(node, section_count=section_count)
        if not source_indices:
            return (
                f"initial tree node {node_name!r} is missing source_section_ids "
                "under traffic-locate active/core scope"
            )
        for section_idx in source_indices:
            title = _traffic_locate_section_title(section_items[section_idx])
            if section_idx in coverage_set:
                return (
                    f"initial tree node {node_name!r} references coverage-only "
                    f"source section {section_idx} ({title})"
                )
            if section_idx not in active_set:
                return (
                    f"initial tree node {node_name!r} references non-active "
                    f"source section {section_idx} ({title})"
                )
    return None


def _traffic_locate_scope_rejection_reason(
    locate_report: Optional[Dict[str, Any]],
    *,
    section_count: int,
    sections: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Validate one-packet active scopes in LLM traffic-location reports.
    
    Check active/core and coverage-only assignments for consistency with the
    reported packet evidence. Permit all-section scopes for small formats."""

    if section_count <= 1:
        return None
    payload = _traffic_locate_payload(locate_report)
    if not payload:
        return None

    active = _collect_traffic_locate_indices(
        payload,
        _TRAFFIC_LOCATE_ACTIVE_KEYS,
        section_count=section_count,
    )
    if not active:
        return None

    coverage = _collect_traffic_locate_indices(
        payload,
        _TRAFFIC_LOCATE_COVERAGE_KEYS,
        section_count=section_count,
    )
    irrelevant = _traffic_locate_indices(payload.get("irrelevant_sections"), section_count=section_count)
    min_broad_sections = _env_positive_int("STEP2_TRAFFIC_LOCATE_BROAD_ACTIVE_MIN_SECTIONS", 8)
    max_active_percent = _env_positive_int("STEP2_TRAFFIC_LOCATE_MAX_ACTIVE_PERCENT", 80)
    active_percent = int(round((len(active) * 100) / max(1, section_count)))

    if (
        section_count >= min_broad_sections
        and active_percent >= max_active_percent
        and not coverage
        and not irrelevant
        ):
        return (
            f"active/core selected {len(active)}/{section_count} sections "
            f"({active_percent}%) with no coverage-only/deferred/irrelevant split"
        )
    coverage_conflict = _traffic_locate_coverage_conflict_reason(
        payload,
        sections=sections,
        active_indices=active,
        coverage_indices=coverage,
    )
    if coverage_conflict:
        return coverage_conflict
    return None


def _traffic_locate_coverage_only_indices(
    locate_report: Optional[Dict[str, Any]],
    *,
    section_count: int,
    active_indices: Sequence[int],
) -> List[int]:
    payload = _traffic_locate_payload(locate_report)
    explicit = _collect_traffic_locate_indices(
        payload,
        _TRAFFIC_LOCATE_COVERAGE_KEYS,
        section_count=section_count,
    )
    if explicit:
        return [idx for idx in explicit if idx not in set(active_indices)]
    active = set(active_indices)
    irrelevant = set(_traffic_locate_indices(payload.get("irrelevant_sections"), section_count=section_count))
    return [idx for idx in range(section_count) if idx not in active and idx not in irrelevant]


def _sections_at_indices(
    sections: Optional[Sequence[Dict[str, Any]]],
    indices: Sequence[int],
) -> Optional[List[Dict[str, Any]]]:
    if sections is None:
        return None
    selected: List[Dict[str, Any]] = []
    for idx in indices:
        if not 0 <= idx < len(sections):
            continue
        section = sections[idx]
        if isinstance(section, dict):
            copied = copy.deepcopy(section)
            copied.setdefault("traffic_locate_original_index", idx)
            selected.append(copied)
        else:
            selected.append({"content": str(section), "traffic_locate_original_index": idx})
    return selected


def _sections_excluding_indices(
    sections: Optional[Sequence[Dict[str, Any]]],
    indices: Sequence[int],
) -> Optional[List[Dict[str, Any]]]:
    if sections is None:
        return None
    excluded = set(indices)
    return [section for idx, section in enumerate(sections) if idx not in excluded]


def _format_observed_traffic_context(
    traffic_file: Optional[Union[str, Path]],
    *,
    max_bytes: int = 48,
    max_packets: Optional[int] = None,
) -> str:
    if not traffic_file:
        return ""
    path = Path(str(traffic_file))
    if not path.exists():
        return ""
    if max_packets is None:
        max_packets = _env_positive_int("STEP2_TRAFFIC_CONTEXT_MAX_PACKETS", 1)
    max_packets = max(1, int(max_packets or 1))
    try:
        packets = _load_packets_from_path(path, max_packets=max_packets)
    except Exception as exc:
        logger.warning("Failed to load observed traffic context from %s: %s", path, exc)
        return ""
    if not packets:
        return ""
    lines = [
        "# OBSERVED RAW TRAFFIC",
        f"- packet_count_shown={len(packets)} source_file={path}",
    ]
    for packet_index, packet in enumerate(packets[:max_packets]):
        byte_lines = []
        for idx, byte in enumerate(packet[:max_bytes]):
            byte_lines.append(f"{idx}:0x{byte:02x}/{byte}")
        if len(packet) > max_bytes:
            byte_lines.append(f"... {len(packet) - max_bytes} more byte(s)")
        lines.extend(
            [
                f"- packet_index={packet_index} total_bytes={len(packet)} raw_hex_prefix={packet.hex()[:160]}",
                "  byte offset values: " + ", ".join(byte_lines),
            ]
        )
    lines.extend(
        [
            "- Use these raw bytes as evidence when reasoning about the documented header, selector fields, active selector value, and active variant.",
            "- Build the initial skeleton so the observed packet(s) have plausible main paths; also include documented sibling variants around the same selector when compact enough.",
            "- Do not overfit to only the observed packet(s): traffic anchors observed paths, while documentation defines the sibling branch set and optional structures.",
        ]
    )
    return "\n".join(lines)


def _traffic_locate_section_briefs(
    sections: Sequence[Dict[str, Any]],
    *,
    max_content_chars: int = 4500,
) -> List[Dict[str, Any]]:
    briefs: List[Dict[str, Any]] = []
    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            briefs.append({"section_index": idx, "content": str(section)[:max_content_chars]})
            continue
        content = section.get("prompt_content")
        if not isinstance(content, str) or not content.strip():
            content = section.get("content") or section.get("summary") or ""
        brief: Dict[str, Any] = {
            "section_index": idx,
            "number": section.get("number", ""),
            "title": section.get("title") or section.get("heading") or section.get("section_title") or section.get("name") or "",
            "source_file": section.get("source_file", ""),
            "content": str(content or "")[:max_content_chars],
        }
        if section.get("packet_formats"):
            packet_formats = section.get("packet_formats") or []
            if isinstance(packet_formats, list):
                brief["packet_formats"] = [
                    _summarize_packet_format(packet_format, max_fields=80)
                    for packet_format in packet_formats[:12]
                ]
            else:
                brief["packet_formats"] = packet_formats
        if section.get("field_definitions"):
            brief["field_definitions"] = section.get("field_definitions")
        if section.get("selector_obligations"):
            brief["selector_obligations"] = section.get("selector_obligations")
        if section.get("coverage_obligations"):
            brief["coverage_obligations"] = section.get("coverage_obligations")
        briefs.append(brief)
    return briefs


def _format_traffic_locate_report(locate_report: Optional[Dict[str, Any]]) -> str:
    payload = _traffic_locate_payload(locate_report)
    if not payload:
        return ""
    try:
        return json.dumps({"traffic_locate": payload}, indent=2, ensure_ascii=False)
    except Exception:
        return str(locate_report or "")


def _build_traffic_anchor_records(
    tree: Dict[str, Any],
    traffic_file: Optional[Union[str, Path]],
    *,
    max_packets: int = 1,
) -> List[Dict[str, Any]]:
    if not traffic_file:
        return []
    path = Path(str(traffic_file))
    if not path.exists():
        return []
    try:
        packets = _load_packets_from_path(path, max_packets=max(1, max_packets))
    except Exception as exc:
        logger.warning("Failed to load traffic anchors from %s: %s", path, exc)
        return []
    if not packets:
        return []
    payload = _protocol_tree_payload(tree)
    try:
        interpreter = DynamicTreeInterpreter(payload, target_message_type=None)
    except Exception as exc:
        logger.debug("Failed to initialize traffic anchor interpreter: %s", exc)
        return []
    node_lookup = _nodes_by_id(payload)
    records: List[Dict[str, Any]] = []
    for idx, packet in enumerate(packets[: max(1, max_packets)]):
        try:
            success, ctx, error = interpreter.parse(packet)
            records.append(_ctx_to_record(idx, packet, success, ctx, error, node_lookup))
        except Exception as exc:
            records.append(
                {
                    "packet_index": idx,
                    "success": False,
                    "error": str(exc),
                    "raw_hex": packet.hex(),
                    "visited_nodes": [],
                    "parsing_path": [],
                    "active_variants": {},
                    "values": {},
                    "typed_values": {},
                    "bit_starts": {},
                    "sizes": {},
                }
            )
    return records


def _format_traffic_anchor_context(
    tree: Dict[str, Any],
    anchor_records: Sequence[Dict[str, Any]],
    *,
    max_nodes: int = 40,
) -> str:
    successful = [record for record in anchor_records or [] if isinstance(record, dict) and record.get("success")]
    if not successful:
        return "- None: no observed packet currently parses successfully through this tree."

    node_lookup = _nodes_by_id(tree)
    lines = [
        "# OBSERVED TRAFFIC ANCHOR (LOCKED)",
        "- The packet below anchors parseability of the observed main path. You may add or restructure sibling doc-derived variants as long as this packet still parses.",
    ]
    for record in successful[:2]:
        raw_hex = str(record.get("raw_hex") or "")
        lines.append(f"- packet_index={record.get('packet_index')} raw_hex_prefix={raw_hex[:96]} total_hex_len={len(raw_hex)}")
        active_variants = record.get("active_variants") or {}
        typed_values = record.get("typed_values") or {}
        if active_variants:
            lines.append("  active selector routes:")
            for selector_id, variants in active_variants.items():
                selector = node_lookup.get(str(selector_id)) or {}
                selector_name = selector.get("name") or selector_id
                selector_value = typed_values.get(str(selector_id), (record.get("values") or {}).get(str(selector_id)))
                variant_names = []
                for variant_id in variants or []:
                    variant = node_lookup.get(str(variant_id)) or {}
                    variant_names.append(f"{variant.get('name') or variant_id}(ID:{variant_id})")
                lines.append(f"  - {selector_name}(ID:{selector_id}) value={selector_value} -> {', '.join(variant_names)}")
        path = [str(nid) for nid in record.get("parsing_path") or []]
        bit_starts = record.get("bit_starts") or {}
        sizes = record.get("sizes") or {}
        lines.append("  locked observed path/spans:")
        for node_id in path[:max_nodes]:
            node = node_lookup.get(node_id) or {}
            value = typed_values.get(node_id, (record.get("values") or {}).get(node_id))
            value_text = f" value={value}" if value is not None else ""
            lines.append(
                f"  - {node.get('name') or node_id}(ID:{node_id}) bit_start={bit_starts.get(node_id)} size_bits={sizes.get(node_id)}{value_text}"
            )
        if len(path) > max_nodes:
            lines.append(f"  - ... {len(path) - max_nodes} additional observed nodes omitted")
    return "\n".join(lines)


def _traffic_anchor_regression_reason(
    before_records: Sequence[Dict[str, Any]],
    after_records: Sequence[Dict[str, Any]],
    *,
    max_locked_nodes: int = 80,
) -> Optional[str]:
    before_success = [record for record in before_records or [] if isinstance(record, dict) and record.get("success")]
    if not before_success:
        return None
    after_by_index = {
        int(record.get("packet_index", idx)): record
        for idx, record in enumerate(after_records or [])
        if isinstance(record, dict)
    }
    for before in before_success:
        packet_index = int(before.get("packet_index", 0) or 0)
        after = after_by_index.get(packet_index)
        if not after:
            return f"observed packet {packet_index} is missing after patch"
        if not after.get("success"):
            return f"observed packet {packet_index} no longer parses: {after.get('error')}"

    return None


def _sha256_file(path: Optional[Union[str, Path]]) -> Optional[str]:
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


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_stage2_input_signature(
    *,
    sections_file: Union[str, Path],
    raw_file: Union[str, Path],
    traffic_file: Optional[Union[str, Path]] = None,
    protocol_hint: Optional[str] = None,
    cache_namespace: Optional[str] = None,
    initial_llm: Optional[Dict[str, Any]] = None,
    refine_llm: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a reproducible signature for Stage B cache reuse."""

    payload: Dict[str, Any] = {
        "schema": _STAGE2_INPUT_SIGNATURE_SCHEMA,
        "sections_sha256": _sha256_file(sections_file),
        "raw_sha256": _sha256_file(raw_file),
        "traffic_sha256": _sha256_file(traffic_file),
        "protocol_hint": _normalize_protocol_hint(protocol_hint),
        "cache_namespace": str(cache_namespace or ""),
        "initial_llm": initial_llm or {},
        "refine_llm": refine_llm or {},
        "prompt_contract": "stage2-format-tree-v2",
        "traffic_inline_contract": "traffic-llm-locate-v1" if traffic_file else "",
    }
    return {
        "schema": _STAGE2_INPUT_SIGNATURE_SCHEMA,
        "sha256": _stable_json_hash(payload),
        "payload": payload,
    }


def _stage2_signature_compare_payload(signature: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(signature, dict):
        return {}
    payload = copy.deepcopy(signature.get("payload") or {})
    if not isinstance(payload, dict):
        return {}
    for llm_key in ("initial_llm", "refine_llm"):
        llm_cfg = payload.get(llm_key)
        if not isinstance(llm_cfg, dict):
            continue
        if not bool(llm_cfg.get("reasoner_fallback_enabled")):
            llm_cfg.pop("reasoner_fallback_model", None)
            llm_cfg.pop("reasoner_fallback_max_tokens", None)
    return payload


def _stage2_signatures_equivalent(
    expected_signature: Optional[Dict[str, Any]],
    actual_signature: Optional[Dict[str, Any]],
) -> bool:
    if not expected_signature or not actual_signature:
        return False
    if str(expected_signature.get("sha256") or "") == str(actual_signature.get("sha256") or ""):
        return True
    expected_payload = _stage2_signature_compare_payload(expected_signature)
    actual_payload = _stage2_signature_compare_payload(actual_signature)
    return bool(expected_payload) and expected_payload == actual_payload


def canonicalize_node_type(node_type: Optional[str]) -> str:
    if not node_type:
        return "field"
    normalized = node_type.strip().lower()
    return _NODE_TYPE_ALIAS.get(normalized, normalized)


def _extract_fenced_json_blocks(text: str) -> List[str]:
    if not text:
        return []
    return [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]


def _extract_balanced_json_object(text: str) -> str:
    if not text:
        return ""

    starts = [idx for idx, ch in enumerate(text) if ch in "{["]
    for start in starts:
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == opening:
                depth += 1
                continue
            if ch == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1].strip()
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
    return ""


def _infer_step2_cache_namespace(sections_file: str, raw_file: str) -> str:
    """Infer a stable cache namespace (protocol name) from input file paths.

    Prefer `.../data/<protocol>/...` when present, otherwise fall back to a short
    hash of absolute paths to avoid cross-protocol cache contamination.
    """

    for cand in (sections_file, raw_file):
        try:
            parts = Path(cand).resolve().parts
        except Exception:
            continue
        for idx, part in enumerate(parts):
            if part != "data" or idx + 1 >= len(parts):
                continue
            protocol = parts[idx + 1]
            if protocol and protocol not in {"_artifacts", "logs", "host_docs"}:
                return protocol

    key = f"{Path(sections_file).resolve()}|{Path(raw_file).resolve()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _normalize_protocol_hint(protocol_hint: Optional[str]) -> str:
    text = str(protocol_hint or "").strip()
    return text


def _protocol_hint_root_name(protocol_hint: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", protocol_hint.strip()).strip("_")
    if not cleaned:
        return "Protocol_Message"
    parts = [part for part in cleaned.split("_") if part]
    if not parts:
        return "Protocol_Message"
    titled = "_".join(part[:1].upper() + part[1:] for part in parts)
    return f"{titled}_Message"


def _summarize_packet_format_field(field_obj: Any) -> Any:
    if not isinstance(field_obj, dict):
        return field_obj
    out: Dict[str, Any] = {}
    for key in (
        "field_name",
        "name",
        "byte_position",
        "bit_position",
        "size",
        "data_type",
        "description",
        "constraints",
        "encoding",
        "endianness",
        "source",
    ):
        if key in field_obj:
            out[key] = field_obj.get(key)
    if "field_name" not in out and "name" in out:
        out["field_name"] = out.pop("name")
    return out


def _summarize_packet_format(packet_format: Any, *, max_fields: int = 60) -> Any:
    if not isinstance(packet_format, dict):
        return packet_format
    out: Dict[str, Any] = {}
    for key in ("format_name", "description", "total_size", "structure_type"):
        if key in packet_format:
            out[key] = packet_format.get(key)
    fields = packet_format.get("fields")
    if isinstance(fields, list):
        out["fields"] = [_summarize_packet_format_field(f) for f in fields[:max_fields]]
    elif fields is not None:
        out["fields"] = fields
    return out


def _normalize_doc_field_name(text: Any) -> str:
    raw = str(text or "").replace("'", "").replace("’", "")
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", raw.lower())).strip()


def _field_constraint_required_false(field: Dict[str, Any]) -> bool:
    constraints = field.get("constraints")
    raw_required = constraints.get("required") if isinstance(constraints, dict) else field.get("required")
    if isinstance(raw_required, bool):
        return raw_required is False
    return str(raw_required or "").strip().lower() in {"false", "no", "optional"}


def _field_transport_envelope_dependent(field: Dict[str, Any]) -> bool:
    constraints = field.get("constraints")
    constraint_values: List[Any] = []
    if isinstance(constraints, dict):
        constraint_values.extend(constraints.values())
    else:
        constraint_values.append(constraints)
    text = " ".join(
        str(value or "")
        for value in (
            field.get("field_name"),
            field.get("name"),
            field.get("description"),
            field.get("encoding"),
            field.get("byte_position"),
            field.get("bit_position"),
            *constraint_values,
        )
    ).lower()
    return any(
        marker in text
        for marker in (
            "underlying communication layer",
            "underlying communication",
            "underlying protocol",
            "transport-specific",
            "transport specific",
            "protocol-dependent",
            "protocol_dependent",
            "protocol dependent",
            "varies by protocol",
            "varies by underlying",
            "replaced by",
        )
    )


def _field_function_dependent_payload_placeholder(field: Dict[str, Any]) -> bool:
    constraints = field.get("constraints")
    constraint_values: List[Any] = []
    if isinstance(constraints, dict):
        constraint_values.extend(constraints.values())
    else:
        constraint_values.append(constraints)
    name_norm = _normalize_doc_field_name(field.get("field_name") or field.get("name"))
    text = " ".join(
        str(value or "")
        for value in (
            field.get("field_name"),
            field.get("name"),
            field.get("description"),
            field.get("size"),
            field.get("data_type"),
            field.get("encoding"),
            field.get("parent_field"),
            *constraint_values,
        )
    ).lower()
    data_type = str(field.get("data_type") or "").lower()
    size = str(field.get("size") or "").lower()
    generic_payload_name = (
        name_norm in {"data", "request data", "response data", "payload", "body"}
        or name_norm.endswith(" data")
    )
    function_dependent = any(
        marker in text
        for marker in (
            "code dependent",
            "type dependent",
            "function dependent",
        )
    )
    variable_payload = any(marker in size for marker in ("n bytes", "0 to n", "variable")) or any(
        marker in data_type for marker in ("bytes", "payload", "binary")
    )
    return (
        generic_payload_name
        and _field_constraint_required_false(field)
        and function_dependent
        and variable_payload
    )


def _text_describes_variable_model_position(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "variable",
            "varies",
            "block-specific",
            "based on index",
            "implementation dependent",
            "not applicable",
        )
    )


def _strict_gap_packet_field_candidate(field: Dict[str, Any]) -> bool:
    """Return true when a Stage-A table row is a concrete parseable field gap."""

    if not isinstance(field, dict):
        return False
    if _field_transport_envelope_dependent(field):
        return False
    if _field_function_dependent_payload_placeholder(field):
        return False
    if _field_constraint_required_false(field) and (
        _text_describes_variable_model_position(field.get("byte_position"))
        or _text_describes_variable_model_position(field.get("bit_position"))
    ):
        return False
    return True


def _packet_format_is_strict_gap_source(packet_format: Dict[str, Any]) -> bool:
    """Return false for illustrative packet-format tables that are source context."""

    if not isinstance(packet_format, dict):
        return False
    text = " ".join(
        str(packet_format.get(key) or "")
        for key in ("format_name", "name", "description", "visual_representation")
    ).lower()
    if any(marker in text for marker in ("example", "sample", "illustrative")):
        return False
    return True


_SOURCE_SCAFFOLD_STOP_TOKENS = {
    "common",
    "core",
    "definition",
    "definitions",
    "element",
    "field",
    "fields",
    "frame",
    "information",
    "ie",
    "message",
    "of",
    "packet",
    "pdu",
    "target",
    "value",
    "values",
}
_SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS = {
    "address",
    "bit",
    "bits",
    "data",
    "description",
    "flag",
    "flags",
    "id",
    "interface",
    "length",
    "specific",
    "spare",
    "timestamp",
    "value",
}
_SOURCE_SCAFFOLD_SPLIT_BYTE_TOKENS = {
    "hi",
    "lo",
    "high",
    "low",
    "msb",
    "lsb",
}
_SOURCE_SCAFFOLD_BODY_TOKENS = {
    "bytes",
    "data",
    "payload",
    "value",
}


def _node_size_bits_int(node: Dict[str, Any]) -> Optional[int]:
    value = node.get("size_bits")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _packet_field_split_piece_represented(
    *,
    wanted: Set[str],
    child_tokens: Set[str],
    child: Dict[str, Any],
    parent_tokens: Set[str],
) -> bool:
    if not (wanted & _SOURCE_SCAFFOLD_SPLIT_BYTE_TOKENS):
        return False
    wanted_base = wanted - _SOURCE_SCAFFOLD_SPLIT_BYTE_TOKENS
    child_base = child_tokens - _SOURCE_SCAFFOLD_SPLIT_BYTE_TOKENS
    if not wanted_base or not child_base:
        return False
    missing_from_child = wanted_base - child_base
    allowed_missing = parent_tokens | _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
    if missing_from_child - allowed_missing:
        return False
    if not (wanted_base & child_base):
        return False
    size_bits = _node_size_bits_int(child)
    return size_bits is not None and size_bits >= 16


def _doc_name_tokens(text: Any) -> List[str]:
    raw = str(text or "").replace("'", "").replace("’", "")
    candidate = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", raw)
    candidate = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", candidate)
    tokens: List[str] = []
    for token in re.sub(r"[^0-9a-zA-Z]+", " ", candidate.lower()).split():
        if not token:
            continue
        tokens.append(token)
        if token == "v4":
            tokens.append("ipv4")
        elif token == "ipv4":
            tokens.append("v4")
        elif token == "v6":
            tokens.append("ipv6")
        elif token == "ipv6":
            tokens.append("v6")
        elif token == "id":
            tokens.append("identifier")
        elif token == "identifier":
            tokens.append("id")
        elif token == "auth":
            tokens.append("authentication")
        elif token == "authentication":
            tokens.append("auth")
    return tokens


def _doc_name_token_set(text: Any, *, drop_stop: bool = True) -> Set[str]:
    tokens = set(_doc_name_tokens(text))
    if "ie" in tokens:
        tokens.update({"information", "element"})
    if {"information", "element"} <= tokens:
        tokens.add("ie")
    if {"s", "d"} <= tokens:
        tokens.update({"source", "destination"})
    if {"source", "destination"} <= tokens:
        tokens.update({"s", "d"})
    if drop_stop:
        tokens = {token for token in tokens if token not in _SOURCE_SCAFFOLD_STOP_TOKENS and not token.isdigit()}
    return tokens


_SELECTOR_MATCH_SUFFIX_TOKENS = {
    "code",
    "codes",
    "field",
    "fields",
    "selector",
    "selectors",
    "type",
    "types",
    "value",
    "values",
}


def _selector_match_token_set(tokens: Set[str]) -> Set[str]:
    primary: Set[str] = set()
    for token in tokens:
        if token in _SELECTOR_MATCH_SUFFIX_TOKENS:
            continue
        if token.isdigit():
            continue
        if re.fullmatch(r"fc\d+", token):
            continue
        if re.fullmatch(r"0x[0-9a-f]+", token):
            continue
        primary.add(token)
    return primary or set(tokens)


def _has_exception_response_role_phrase(text: Any) -> bool:
    raw = str(text or "").lower()
    normalized = re.sub(r"[^0-9a-zA-Z]+", " ", raw)
    return bool(
        re.search(r"\bexception\s+(?:response|reply|rsp|pdu|message|packet|frame)\b", normalized)
    )


def _source_format_role(text: Any) -> str:
    tokens = _doc_name_token_set(text, drop_stop=False)
    if "request" in tokens or "req" in tokens:
        return "request"
    if "error" in tokens or _has_exception_response_role_phrase(text):
        return "error"
    if "response" in tokens or "reply" in tokens or "rsp" in tokens:
        return "response"
    return ""


def _source_format_roles_compatible(format_name: Any, node: Dict[str, Any]) -> bool:
    expected_role = _source_format_role(format_name)
    actual_role = _source_format_role(node.get("name"))
    return not (expected_role and actual_role and expected_role != actual_role)


def _normalize_message_type_scope(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"request", "response"}:
        return text
    return "bidirectional"


def _message_type_scope_compatible(edge_message_type: Any, expected_message_type: Any) -> bool:
    expected = _normalize_message_type_scope(expected_message_type)
    if expected == "bidirectional":
        return True
    edge_scope = _normalize_message_type_scope(edge_message_type)
    return edge_scope in {"bidirectional", expected}


def _selector_obligation_message_type(entry: Dict[str, Any], owner_node: Optional[Dict[str, Any]] = None) -> str:
    """Return the effective direction scope for selector-obligation metadata."""

    explicit = entry.get("message_type") if isinstance(entry, dict) else None
    if explicit:
        return _normalize_message_type_scope(explicit)
    if isinstance(owner_node, dict):
        return _normalize_message_type_scope(owner_node.get("message_type"))
    return "bidirectional"


def _packet_format_parent_match_score(node: Dict[str, Any], format_name: Any) -> int:
    node_type = str(node.get("node_type") or "").lower()
    if node_type not in {"variant", "container", "payload", "header", "field", "selector"}:
        return -1
    if not _source_format_roles_compatible(format_name, node):
        return -1
    fmt_tokens = _doc_name_token_set(format_name)
    node_tokens = _doc_name_token_set(node.get("name"))
    if not fmt_tokens or not node_tokens:
        return -1
    type_bonus = 10 if node_type in {"variant", "container", "payload", "header"} else 0
    if type_bonus and (fmt_tokens - {"type"}) == node_tokens:
        return 90 + type_bonus
    if fmt_tokens == node_tokens:
        return 100 + type_bonus
    common = fmt_tokens & node_tokens
    if len(fmt_tokens) >= 2 and len(common) >= 2 and (fmt_tokens <= node_tokens or node_tokens <= fmt_tokens):
        return 80 + type_bonus + len(common)
    if len(fmt_tokens) == 1 and fmt_tokens == node_tokens:
        return 90 + type_bonus
    return -1


def _find_packet_format_parent(tree: Dict[str, Any], format_name: Any) -> Optional[Dict[str, Any]]:
    best_node: Optional[Dict[str, Any]] = None
    best_score = -1
    for node in tree.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        score = _packet_format_parent_match_score(node, format_name)
        if score > best_score:
            best_score = score
            best_node = node
    return best_node if best_score >= 80 else None


def _find_packet_format_selector_variant_parent(
    tree: Dict[str, Any],
    packet_format: Dict[str, Any],
    section: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    guard = _packet_format_selector_guard(packet_format, section)
    if guard is None:
        return None
    _, selector_values = guard
    format_name = str(packet_format.get("format_name") or "").strip()
    expected_role = _source_format_role(format_name)
    expected_message_type = _infer_packet_format_message_type(packet_format)
    section_label = _source_section_label(section)

    best_node: Optional[Dict[str, Any]] = None
    best_score = -1
    for node in tree.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_type = canonicalize_node_type(str(node.get("node_type") or ""))
        if node_type not in {"variant", "container", "payload"}:
            continue
        if expected_role:
            if not _source_format_roles_compatible(format_name, node):
                continue
            message_type = str(node.get("message_type") or "").lower()
            name_role = _source_format_role(node.get("name"))
            if expected_role == "request" and message_type and message_type != "request":
                continue
            if expected_role in {"response", "error"} and message_type and message_type != "response":
                continue
            if expected_role == "error" and name_role != "error":
                continue
            if expected_role in {"request", "response"} and name_role and name_role != expected_role:
                continue
        for entry in node.get("selector_obligations") or []:
            if not isinstance(entry, dict):
                continue
            if _selector_obligation_values(entry) != tuple(selector_values):
                continue
            if not _message_type_scope_compatible(
                _selector_obligation_message_type(entry, node),
                expected_message_type,
            ):
                continue
            score = 100
            if str(entry.get("source") or "") == section_label:
                score += 100
            if expected_role and _source_format_role(node.get("name")) == expected_role:
                score += 25
            if score > best_score:
                best_score = score
                best_node = node
    return best_node


def _unique_node_by_canonical_type(
    tree: Dict[str, Any],
    wanted_types: Set[str],
) -> Optional[Dict[str, Any]]:
    candidates = [
        node
        for node in tree.get("nodes") or []
        if isinstance(node, dict)
        and canonicalize_node_type(str(node.get("node_type") or "")) in wanted_types
    ]
    return candidates[0] if len(candidates) == 1 else None


def _packet_format_structural_parent_candidate(node: Dict[str, Any]) -> bool:
    node_type = canonicalize_node_type(str(node.get("node_type") or ""))
    return node_type not in {"length", "selector"}


def _find_packet_format_structural_parent(
    tree: Dict[str, Any],
    packet_format: Dict[str, Any],
    section: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    format_name = str(packet_format.get("format_name") or "").strip()
    if _packet_format_is_generic_exception_alias(packet_format):
        parent = _find_generic_exception_response_parent(tree)
        if isinstance(parent, dict):
            return parent

    if _role_only_packet_format_name(format_name):
        parent = _find_packet_format_selector_variant_parent(tree, packet_format, section)
        if isinstance(parent, dict) and _packet_format_structural_parent_candidate(parent):
            return parent

    if _packet_format_selector_guard(packet_format, section) is not None:
        parent = _find_packet_format_selector_variant_parent(tree, packet_format, section)
        if isinstance(parent, dict) and _packet_format_structural_parent_candidate(parent):
            return parent

    parent = _find_packet_format_parent(tree, format_name)
    if isinstance(parent, dict) and _packet_format_structural_parent_candidate(parent):
        return parent

    parent = _find_packet_format_selector_variant_parent(tree, packet_format, section)
    if isinstance(parent, dict) and _packet_format_structural_parent_candidate(parent):
        return parent

    structure_type = str(packet_format.get("structure_type") or "").lower()
    format_tokens = _doc_name_token_set(format_name, drop_stop=False)
    if "header" in format_tokens:
        return _unique_node_by_canonical_type(tree, {"header"})
    if "header" in structure_type:
        parent = _unique_node_by_canonical_type(tree, {"header"})
        if isinstance(parent, dict):
            parent_tokens = _doc_name_token_set(parent.get("name"), drop_stop=False)
            common = format_tokens & parent_tokens
            if format_tokens and (format_tokens <= parent_tokens or len(common) >= 2):
                return parent
    return None


def _coerce_packet_field_size_bits(field: Dict[str, Any]) -> Any:
    size = str(field.get("size") or "").strip().lower()
    match = re.match(r"^(\d+)\s*bits?(?:\s*\(.*\))?$", size)
    if match:
        return int(match.group(1))
    match = re.match(r"^(\d+)\s*bytes?$", size)
    if match:
        return int(match.group(1)) * 8
    data_type = str(field.get("data_type") or "").lower()
    for label, bits in (
        ("uint64", 64),
        ("uint32", 32),
        ("uint16", 16),
        ("uint8", 8),
        ("uint4", 4),
        ("bool", 1),
        ("ipv4", 32),
        ("ipv6", 128),
    ):
        if label in data_type:
            return bits
    return "variable"


def _source_field_bit_start(field: Dict[str, Any]) -> Optional[int]:
    match = re.match(r"^(\d+)(?:\s*[-+]|$)", str(field.get("bit_position") or "").strip())
    return int(match[1]) if match else None


def _section_text_lines(section: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key in ("content", "prompt_content", "summary"):
        value = section.get(key)
        if isinstance(value, str):
            lines.extend(value.splitlines())
    for paragraph in section.get("source_paragraphs") or []:
        if isinstance(paragraph, dict) and isinstance(paragraph.get("text"), str):
            lines.append(paragraph["text"])
    return [line.strip() for line in lines if str(line or "").strip()]


def _clean_bitfield_flag_label(raw_label: Any) -> str:
    label = str(raw_label or "").replace("\u00a0", " ").strip()
    label = re.sub(r"\([^)]*\)", " ", label)
    label = re.split(r"\b(?:if|when|shall|is supported|are present)\b", label, maxsplit=1, flags=re.IGNORECASE)[0]
    label = re.sub(r"\s+", " ", label.strip(" :-–—,."))
    if not label:
        return ""
    aliases = {
        "s/d": "Source Destination",
        "s d": "Source Destination",
    }
    lowered = label.lower()
    if lowered in aliases:
        return aliases[lowered]
    return label


def _bitfield_flag_display_name(format_name: Any, label: str, bit_count: int = 1) -> str:
    label = _clean_bitfield_flag_label(label)
    if not label:
        return ""
    format_text = str(format_name or "").strip()
    label_tokens = _doc_name_token_set(label)
    format_tokens = _doc_name_token_set(format_text)
    if label_tokens & {"spare", "reserved"}:
        suffix = "Spare Bit" if bit_count <= 1 else "Spare Flags"
    elif "flag" in label_tokens:
        suffix = label
    else:
        suffix = f"{label} Flag"
    suffix_tokens = _doc_name_token_set(suffix)
    if format_text and format_tokens and not format_tokens <= suffix_tokens:
        return f"{format_text} {suffix}".strip()
    return suffix


def _spec_bit_number_range_to_msb0(start: int, end: Optional[int] = None) -> Tuple[int, int]:
    """Convert source bit numbering to a zero-based MSB0 bit offset.

    Many protocol specs number bits within an octet as 8..1, where bit 8 is
    the most significant bit.  Our IR and evaluators use MSB0 offsets, so
    source "Bit 8" maps to offset 0 and "Bit 1" maps to offset 7.
    """

    lo = min(int(start), int(end if end is not None else start))
    hi = max(int(start), int(end if end is not None else start))
    width = max(1, hi - lo + 1)
    if 1 <= lo <= 8 and 1 <= hi <= 8:
        return max(0, 8 - hi), width
    return max(0, lo - 1), width


def _extract_bitfield_flag_specs(section: Dict[str, Any], packet_format: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract source-backed bit/flag leaves from generic bitmask evidence."""

    format_name = str(packet_format.get("format_name") or "").strip()
    if not format_name:
        return []
    text_lines = _section_text_lines(section)
    if not text_lines:
        return []
    text_blob = "\n".join(text_lines).lower()
    has_octet_bit_rows = "bit" in text_blob and any(
        re.fullmatch(r"\d+\s*/\s*\d+", line)
        for line in text_lines
    )
    if (
        "bitmask" not in text_blob
        and not re.search(r"\b-?\s*bit\s+\d+", text_blob)
        and not has_octet_bit_rows
    ):
        return []

    fields = packet_format.get("fields") or []
    bitfield_fields = [
        field
        for field in fields
        if isinstance(field, dict)
        and (
            "bitfield" in str(field.get("data_type") or "").lower()
            or "flag" in _doc_name_token_set(field.get("field_name") or field.get("name"), drop_stop=False)
        )
    ]
    bitfield_parent_fields = [
        field
        for field in fields
        if isinstance(field, dict)
        and (
            "bitfield" in str(field.get("data_type") or "").lower()
            or "description" in _doc_name_token_set(field.get("field_name") or field.get("name"), drop_stop=False)
        )
    ]
    bitfield_parent_name = ""
    if bitfield_parent_fields:
        bitfield_parent_name = _scaffold_field_display_name(
            format_name,
            bitfield_parent_fields[0].get("field_name") or bitfield_parent_fields[0].get("name"),
        )
    else:
        bitfield_parent_name = _scaffold_field_display_name(format_name, "Flags")
    if not bitfield_fields:
        return []

    specs: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    occupied_offsets: Set[int] = set()

    def add_spec(label: str, bit_count: int = 1, bit_offset: Optional[int] = None) -> None:
        display = _bitfield_flag_display_name(format_name, label, bit_count)
        norm = _normalize_doc_field_name(display)
        if not display or not norm or norm in seen:
            return
        seen.add(norm)
        specs.append(
            {
                "field_name": display,
                "data_type": "uint" if "spare" in norm or "reserved" in norm or bit_count > 1 else "bool",
                "size": f"{max(1, int(bit_count))} bits",
                "constraints": {"valid_values": ["0"]} if "spare" in norm or "reserved" in norm else {},
                "_source_kind": "bitfield_flag",
                "_bit_count": max(1, int(bit_count)),
                "_bit_offset": bit_offset,
                "_bitfield_parent_name": bitfield_parent_name,
            }
        )
        if bit_offset is not None:
            occupied_offsets.update(
                range(int(bit_offset), int(bit_offset) + max(1, int(bit_count)))
            )

    # Prose form: "-Bit 1 - V6: ..." or "-Bit 3 to 8 are spare ..."
    current_prose_octet: Optional[int] = None
    base_prose_octet: Optional[int] = None
    for line in text_lines:
        octet_match = re.search(r"\boctet\s+(?P<octet>\d+)\b", line, flags=re.IGNORECASE)
        if octet_match:
            current_prose_octet = int(octet_match.group("octet"))
            if base_prose_octet is None:
                base_prose_octet = current_prose_octet
        match = re.search(
            r"-?\s*bit\s+(?P<start>\d+)(?:\s+to\s+(?P<end>\d+))?\s*(?:[-–—]\s*(?P<label>[^:.,]+)|(?P<spare>are\s+spare|spare\b))",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        local_bit_offset, bit_count = _spec_bit_number_range_to_msb0(start, end)
        label = match.group("label") or "Spare"
        octet_delta = 0
        if current_prose_octet is not None and base_prose_octet is not None:
            octet_delta = max(0, current_prose_octet - base_prose_octet) * 8
        add_spec(label, bit_count, max(0, octet_delta + local_bit_offset))

    # Table form: one line "5/1" followed by a label line.  Use the first
    # observed octet as the base so rows such as 6/1 do not collide with 5/1.
    table_rows: List[Tuple[int, int, int]] = []
    for idx, line in enumerate(text_lines[:-1]):
        bit_row = re.fullmatch(r"\d+\s*/\s*(\d+)", line)
        full_row = re.fullmatch(r"(\d+)\s*/\s*(\d+)", line)
        if not bit_row or not full_row:
            continue
        table_rows.append((idx, int(full_row.group(1)), int(full_row.group(2))))
    base_octet = min((octet for _, octet, _ in table_rows), default=0)
    for idx, octet, bit in table_rows:
        next_line = text_lines[idx + 1]
        if re.fullmatch(r"\d+\s*/\s*\d+", next_line):
            continue
        label = _clean_bitfield_flag_label(next_line)
        if not label:
            continue
        label_tokens = _doc_name_token_set(label)
        format_tokens = _doc_name_token_set(format_name)
        if (
            "definition" in _doc_name_tokens(label)
            or label_tokens == format_tokens
            or (format_tokens and label_tokens and label_tokens <= format_tokens)
        ):
            continue
        # Avoid expanding "Additional ..." extension bitmasks into unsupported
        # fine-grained rows unless the source field itself is the primary bitmask.
        if "additional" in label.lower() and "additional" not in _normalize_doc_field_name(format_name):
            continue
        local_bit_offset, bit_count = _spec_bit_number_range_to_msb0(bit, bit)
        add_spec(label, bit_count, max(0, (octet - base_octet) * 8 + local_bit_offset))

    parent_width = None
    if bitfield_parent_fields:
        parent_width = _coerce_packet_field_size_bits(bitfield_parent_fields[0])
    has_declared_spare_bits = bool(
        re.search(r"\bspare bits?\b|\bspare\s*,|\breserved\b", text_blob, flags=re.IGNORECASE)
    )
    if isinstance(parent_width, int) and parent_width > 0 and occupied_offsets and has_declared_spare_bits:
        offset = 0
        while offset < parent_width:
            if offset in occupied_offsets:
                offset += 1
                continue
            start = offset
            while offset < parent_width and offset not in occupied_offsets:
                offset += 1
            add_spec("Spare", offset - start, start)

    return specs


def _scaffold_field_display_name(format_name: Any, field_name: Any) -> str:
    raw_name = str(field_name or "").strip()
    if not raw_name:
        return raw_name
    format_tokens = _doc_name_token_set(format_name)
    field_tokens = _doc_name_token_set(raw_name)
    raw_tokens = set(_doc_name_tokens(raw_name))
    local_component_tokens = raw_tokens & _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
    # Concrete names such as "FAR ID" or "F-TEID V4 Flag" already carry enough
    # semantics.  Short/generic leaves such as "TEID", "Spare", and "Value"
    # need the owning format context to avoid becoming ambiguous in coverage.
    needs_context = (
        not field_tokens
        or len(field_tokens) <= 1
        or bool(field_tokens <= _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS)
        or bool(local_component_tokens and format_tokens and not format_tokens <= field_tokens)
        or (
            "value" in field_tokens
            and format_tokens
            and not format_tokens <= field_tokens
        )
    )
    if format_tokens and not format_tokens <= field_tokens and needs_context:
        owner_words = [
            token
            for token in re.sub(r"[^0-9a-zA-Z]+", " ", str(format_name or "").lower()).split()
            if token in format_tokens
        ]
        raw_words = re.findall(r"[0-9A-Za-z]+", raw_name)
        raw_norm = [
            re.sub(r"[^0-9a-zA-Z]+", "", word.lower())
            for word in raw_words
            if re.sub(r"[^0-9a-zA-Z]+", "", word.lower())
        ]
        while owner_words and raw_words and raw_norm and owner_words[-1] == raw_norm[0]:
            raw_words = raw_words[1:]
            raw_norm = raw_norm[1:]
        if raw_words and " ".join(raw_words) != raw_name:
            display_tail = " ".join(word[:1].upper() + word[1:] if word.islower() else word for word in raw_words)
            return f"{format_name} {display_tail}".strip()
        return f"{format_name} {raw_name}".strip()
    return raw_name


def _scaffold_field_display_name_for_field(format_name: Any, field_obj: Dict[str, Any]) -> str:
    raw_name = str(field_obj.get("field_name") or field_obj.get("name") or "").strip()
    if not raw_name:
        return ""
    raw_tokens = _doc_name_token_set(raw_name)
    if raw_tokens & {"spare", "reserved"} and not (raw_tokens & {"bit", "bits", "flag", "flags"}):
        size_bits = _coerce_packet_field_size_bits(field_obj)
        if isinstance(size_bits, int) and size_bits <= 1:
            raw_name = f"{raw_name} Bit"
    return _scaffold_field_display_name(format_name, raw_name)


def _scaffold_field_display_name_for_parent(
    format_name: Any,
    field_obj: Dict[str, Any],
    parent: Optional[Dict[str, Any]],
) -> str:
    parent_type = canonicalize_node_type(str((parent or {}).get("node_type") or ""))
    parent_norm = _normalize_doc_field_name((parent or {}).get("name"))
    format_norm = _normalize_doc_field_name(format_name)
    parent_tokens = _doc_name_token_set((parent or {}).get("name"))
    format_tokens = _doc_name_token_set(format_name)
    format_matches_parent = bool(
        parent_tokens
        and format_tokens
        and (
            parent_tokens == format_tokens
            or format_tokens <= parent_tokens
            or parent_tokens <= format_tokens
        )
    )
    raw_name = str(field_obj.get("field_name") or field_obj.get("name") or "").strip()
    raw_tokens = set(_doc_name_tokens(raw_name))
    raw_generic_tokens = raw_tokens & _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
    raw_specific_tokens = _doc_name_token_set(raw_name) - _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
    parent_specific_tokens = (
        parent_tokens
        - _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
        - {"section", "header", "format", "table"}
    )
    needs_format_context_under_matching_parent = (
        parent_type in {"variant", "container", "payload"}
        and format_matches_parent
        and format_tokens
        and raw_tokens
        and not format_tokens <= _doc_name_token_set(raw_name)
        and not _role_only_packet_format_name(format_name)
        and not _source_format_role(format_name)
        and bool(raw_generic_tokens)
        and not (len(raw_tokens) == 1 and raw_name.isupper() and raw_name.lower() == "type")
    )
    if (
        needs_format_context_under_matching_parent
        and raw_specific_tokens
        and parent_specific_tokens
        and raw_specific_tokens & parent_specific_tokens
    ):
        return _scaffold_field_display_name_for_field("", field_obj)
    if needs_format_context_under_matching_parent:
        return _scaffold_field_display_name_for_field(format_name, field_obj)
    if (
        parent_type in {"variant", "container", "payload"}
        and raw_specific_tokens
        and parent_specific_tokens
        and raw_specific_tokens & parent_specific_tokens
    ):
        return _scaffold_field_display_name_for_field("", field_obj)
    if (
        parent_type in {"field", "selector", "header"}
        or (parent_norm and parent_norm == format_norm)
        or format_matches_parent
    ):
        return _scaffold_field_display_name_for_field("", field_obj)
    return _scaffold_field_display_name_for_field(format_name, field_obj)


def _child_ids_for_parent(parent: Dict[str, Any], nodes_by_id: Dict[Any, Dict[str, Any]]) -> List[Any]:
    out: List[Any] = list(parent.get("children_ids") or [])
    parent_id = parent.get("node_id")
    if parent_id is None:
        return out
    seen: Set[str] = {str(child_id) for child_id in out}
    unique_nodes: Dict[str, Dict[str, Any]] = {}
    for node in nodes_by_id.values():
        if isinstance(node, dict) and node.get("node_id") is not None:
            unique_nodes[str(node.get("node_id"))] = node
    for node in unique_nodes.values():
        node_id = node.get("node_id")
        if str(node.get("parent_id")) == str(parent_id) and str(node_id) not in seen:
            out.append(node_id)
            seen.add(str(node_id))
    return out


def _ancestor_context_tokens_between(
    parent: Dict[str, Any],
    child: Dict[str, Any],
    nodes_by_id: Dict[Any, Dict[str, Any]],
) -> Set[str]:
    """Return branch/container tokens between a searched parent and a matched leaf."""

    parent_id = parent.get("node_id")
    if parent_id is None:
        return set()
    out: Set[str] = set()
    current = nodes_by_id.get(child.get("parent_id")) or nodes_by_id.get(str(child.get("parent_id")))
    seen: Set[str] = set()
    while isinstance(current, dict):
        current_id = current.get("node_id")
        if current_id is None:
            break
        if str(current_id) == str(parent_id):
            break
        if str(current_id) in seen:
            break
        seen.add(str(current_id))
        out.update(_doc_name_token_set(current.get("name")))
        current = nodes_by_id.get(current.get("parent_id")) or nodes_by_id.get(str(current.get("parent_id")))
    return out


def _packet_field_represented_under(parent: Dict[str, Any], nodes_by_id: Dict[Any, Dict[str, Any]], field_name: str) -> bool:
    wanted = _doc_name_token_set(field_name)
    if not wanted:
        return False
    parent_tokens = _doc_name_token_set(parent.get("name"))
    stack = _child_ids_for_parent(parent, nodes_by_id)
    seen_ids: Set[str] = set()
    while stack:
        child_id = stack.pop(0)
        if str(child_id) in seen_ids:
            continue
        seen_ids.add(str(child_id))
        child = nodes_by_id.get(child_id) or nodes_by_id.get(str(child_id))
        if not isinstance(child, dict):
            continue
        stack.extend(child.get("children_ids") or [])
        child_tokens = _doc_name_token_set(child.get("name"))
        if not child_tokens:
            continue
        wanted_body_tokens = wanted & _SOURCE_SCAFFOLD_BODY_TOKENS
        child_has_body_token = bool(child_tokens & _SOURCE_SCAFFOLD_BODY_TOKENS)
        ancestor_context_tokens = _ancestor_context_tokens_between(parent, child, nodes_by_id)
        contextual_child_tokens = child_tokens | ancestor_context_tokens
        if child_tokens == wanted:
            return True
        if ancestor_context_tokens and not (wanted_body_tokens and not child_has_body_token):
            context_missing = wanted - contextual_child_tokens
            if context_missing <= (parent_tokens | _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS):
                return True
        extra_child_tokens = child_tokens - wanted
        allowed_extra_child_tokens = parent_tokens | _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
        if len(wanted) >= 2 and wanted <= child_tokens and extra_child_tokens <= allowed_extra_child_tokens:
            return True
        if len(wanted) >= 2 and wanted <= child_tokens and not extra_child_tokens:
            return True
        child_specific = child_tokens - parent_tokens
        child_type = str(child.get("node_type") or "").lower()
        missing_from_child = wanted - child_tokens
        allowed_missing = parent_tokens | _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
        if (
            child_type in {"field", "length", "selector"}
            and len(child_tokens) >= 2
            and child_tokens <= wanted
            and child_specific
            and missing_from_child <= allowed_missing
            and not (wanted_body_tokens and not child_has_body_token)
        ):
            return True
        if (
            child_type in {"field", "length", "selector"}
            and _packet_field_split_piece_represented(
                wanted=wanted,
                child_tokens=child_tokens,
                child=child,
                parent_tokens=parent_tokens,
            )
        ):
            return True
    return False


def _packet_field_represented_anywhere(nodes_by_id: Dict[Any, Dict[str, Any]], field_name: str) -> bool:
    wanted = _doc_name_token_set(field_name)
    specific_wanted = wanted - _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
    if not wanted or not specific_wanted:
        return False

    unique_nodes: Dict[str, Dict[str, Any]] = {}
    for node in nodes_by_id.values():
        if isinstance(node, dict) and node.get("node_id") is not None:
            unique_nodes[str(node.get("node_id"))] = node

    for node in unique_nodes.values():
        node_type = str(node.get("node_type") or "").lower()
        if node_type not in {"field", "length", "selector"}:
            continue
        child_tokens = _doc_name_token_set(node.get("name"))
        if not child_tokens:
            continue
        child_specific = child_tokens - _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS
        if child_tokens == wanted:
            return True
        if specific_wanted <= child_tokens and child_specific:
            return True
        if (
            child_tokens <= wanted
            and child_specific
            and len(child_specific & specific_wanted) >= 2
        ):
            return True
        missing_from_child = wanted - child_tokens
        if child_tokens <= wanted and child_specific and missing_from_child <= _SOURCE_SCAFFOLD_GENERIC_LEAF_TOKENS:
            return True
        if _packet_field_split_piece_represented(
            wanted=wanted,
            child_tokens=child_tokens,
            child=node,
            parent_tokens=set(),
        ):
            return True
    return False


def _packet_field_exact_represented_under(parent: Dict[str, Any], nodes_by_id: Dict[Any, Dict[str, Any]], field_name: str) -> bool:
    wanted_norm = _normalize_doc_field_name(field_name)
    wanted_tokens = _doc_name_token_set(field_name)
    if not wanted_norm and not wanted_tokens:
        return False
    parent_tokens = _doc_name_token_set(parent.get("name"))
    stack = _child_ids_for_parent(parent, nodes_by_id)
    seen_ids: Set[str] = set()
    while stack:
        child_id = stack.pop(0)
        if str(child_id) in seen_ids:
            continue
        seen_ids.add(str(child_id))
        child = nodes_by_id.get(child_id) or nodes_by_id.get(str(child_id))
        if not isinstance(child, dict):
            continue
        child_norm = _normalize_doc_field_name(child.get("name"))
        child_tokens = _doc_name_token_set(child.get("name"))
        if (wanted_norm and child_norm == wanted_norm) or (wanted_tokens and child_tokens == wanted_tokens):
            return True
        if wanted_tokens and child_tokens and wanted_tokens <= child_tokens and (child_tokens - wanted_tokens) <= parent_tokens:
            return True
        stack.extend(child.get("children_ids") or [])
    return False


def _source_section_label(section: Optional[Dict[str, Any]]) -> str:
    if not isinstance(section, dict):
        return "source"
    return str(section.get("title") or section.get("section_title") or section.get("number") or "source")


def _role_only_packet_format_name(format_name: Any) -> bool:
    tokens = _doc_name_token_set(format_name)
    if not tokens:
        return False
    role_tokens = {"request", "response", "error", "exception", "req", "rsp"}
    return tokens <= role_tokens


def _packet_format_is_generic_exception_alias(packet_format: Dict[str, Any]) -> bool:
    text = " ".join(
        str(packet_format.get(key) or "")
        for key in ("format_name", "description", "structure_type")
    ).lower()
    if not ("error" in text or "exception" in text):
        return False

    fields = [field for field in packet_format.get("fields") or [] if isinstance(field, dict)]
    if not fields:
        return False

    has_exception_body = False
    for field in fields:
        tokens = _doc_name_token_set(field.get("field_name") or field.get("name"), drop_stop=False)
        if not tokens:
            continue
        if {"function", "code"} <= tokens or {"error", "code"} <= tokens:
            continue
        if {"exception", "code"} <= tokens:
            has_exception_body = True
            continue
        return False
    return has_exception_body


def _find_generic_exception_response_parent(tree: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    current = normalize_protocol_tree(tree)
    nodes = [node for node in current.get("nodes") or [] if isinstance(node, dict)]
    nodes_by_id: Dict[Any, Dict[str, Any]] = {
        node.get("node_id"): node
        for node in nodes
        if node.get("node_id") is not None
    }
    nodes_by_id.update({str(key): value for key, value in list(nodes_by_id.items())})
    edges = [edge for edge in current.get("edges") or [] if isinstance(edge, dict)]

    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for node in nodes:
        node_type = canonicalize_node_type(str(node.get("node_type") or ""))
        if node_type not in {"variant", "container", "payload"}:
            continue
        if str(node.get("message_type") or "").lower() not in {"", "response", "bidirectional"}:
            continue
        name = str(node.get("name") or "")
        name_tokens = _doc_name_token_set(name, drop_stop=False)
        if "exception" not in name_tokens:
            continue
        if not _has_exception_response_role_phrase(name):
            continue
        if re.search(r"\bfc[0-9a-f]+\b", name.lower()):
            continue
        has_exception_code = any(
            isinstance(nodes_by_id.get(child_id) or nodes_by_id.get(str(child_id)), dict)
            and {"exception", "code"} <= _doc_name_token_set(
                (nodes_by_id.get(child_id) or nodes_by_id.get(str(child_id))).get("name"),
                drop_stop=False,
            )
            for child_id in node.get("children_ids") or []
        )
        if not has_exception_code:
            continue
        score = 100
        for edge in edges:
            if (
                isinstance(edge, dict)
                and str(edge.get("rel")) == "condition_on"
                and str(edge.get("dst")) == str(node.get("node_id"))
                and _condition_edge_covers_value(edge, edge.get("src"), 128)
                and _condition_edge_covers_value(edge, edge.get("src"), 255)
            ):
                score += 100
                break
        if {"response", "exception"} <= name_tokens:
            score += 20
        candidates.append((score, node))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def _generic_exception_alias_field_covered_by_parent(
    parent_node: Dict[str, Any],
    packet_format: Dict[str, Any],
    field_obj: Dict[str, Any],
    edges: Sequence[Dict[str, Any]],
) -> bool:
    if not _packet_format_is_generic_exception_alias(packet_format):
        return False
    field_tokens = _doc_name_token_set(field_obj.get("field_name") or field_obj.get("name"), drop_stop=False)
    if not ({"error", "code"} <= field_tokens or {"function", "code"} <= field_tokens):
        return False
    parent_tokens = _doc_name_token_set(parent_node.get("name"), drop_stop=False)
    if "exception" not in parent_tokens:
        return False
    parent_id = parent_node.get("node_id")
    if parent_id is None:
        return False

    try:
        from ..validation.doc_constraint_coverage import _selector_values_from_field
    except Exception:
        values: Tuple[int, ...] = ()
    else:
        values = tuple(int(value) for value in _selector_values_from_field(field_obj))

    for edge in edges:
        if (
            not isinstance(edge, dict)
            or str(edge.get("rel")) != "condition_on"
            or str(edge.get("dst")) != str(parent_id)
        ):
            continue
        selector_id = edge.get("src")
        if values:
            if all(_condition_edge_covers_value(edge, selector_id, int(value)) for value in values):
                return True
        elif (
            _condition_edge_covers_value(edge, selector_id, 128)
            and _condition_edge_covers_value(edge, selector_id, 255)
        ):
            return True
    return False


def _packet_format_selector_guard(
    packet_format: Dict[str, Any],
    section: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, Tuple[int, ...]]]:
    try:
        from ..validation.doc_constraint_coverage import (
            _selector_name_from_context,
            _selector_values_from_field,
        )
    except Exception:
        return None

    fields = [field for field in packet_format.get("fields") or [] if isinstance(field, dict)]
    for field in fields:
        selector_name = _selector_name_from_context(packet_format, field, section or {})
        values = _selector_values_from_field(field)
        if not selector_name or not values:
            continue
        field_tokens = _doc_name_token_set(field.get("field_name") or field.get("name"), drop_stop=False)
        selector_tokens = _doc_name_token_set(selector_name, drop_stop=False)
        if not (field_tokens & selector_tokens or {"function", "code"} <= field_tokens):
            continue
        return selector_name, tuple(int(value) for value in values)
    return None


def _infer_packet_format_message_type(packet_format: Dict[str, Any]) -> str:
    name = str(packet_format.get("format_name") or "").lower()
    if ("reply" in name or "response" in name) and ("/" in name or " or " in name or "request" in name):
        return "bidirectional"
    name_role = _source_format_role(packet_format.get("format_name"))
    if name_role == "request":
        return "request"
    if name_role in {"response", "error"}:
        return "response"
    description_role = _source_format_role(packet_format.get("description"))
    if description_role == "request":
        return "request"
    if description_role in {"response", "error"}:
        return "response"
    structure_role = _source_format_role(packet_format.get("structure_type"))
    if structure_role == "request":
        return "request"
    if structure_role in {"response", "error"}:
        return "response"
    return "bidirectional"


def _field_has_generic_exception_selector_range(field: Dict[str, Any]) -> bool:
    tokens = _doc_name_token_set(field.get("field_name") or field.get("name"), drop_stop=False)
    if not ({"function", "code"} <= tokens or {"error", "code"} <= tokens):
        return False
    constraints = field.get("constraints") if isinstance(field.get("constraints"), dict) else {}

    def parse_int(value: Any) -> Optional[int]:
        text = str(value or "").strip()
        if not text:
            return None
        match = re.search(r"0x[0-9a-fA-F]+|\d+", text)
        if not match:
            return None
        raw = match.group(0)
        try:
            return int(raw, 16) if raw.lower().startswith("0x") else int(raw)
        except ValueError:
            return None

    min_value = parse_int(constraints.get("min_value"))
    max_value = parse_int(constraints.get("max_value"))
    if min_value is not None and max_value is not None and min_value >= 128 and max_value >= 255:
        return True

    values = constraints.get("valid_values")
    if isinstance(values, (str, int)):
        values = [values]
    for value in values or []:
        text = str(value or "").lower()
        nums = [
            int(raw, 16) if raw.lower().startswith("0x") else int(raw)
            for raw in re.findall(r"0x[0-9a-f]+|\d+", text)
        ]
        if len(nums) >= 2 and min(nums) >= 128 and max(nums) >= 255:
            return True
    return False


def _normalize_initial_tree_schema(tree: Any) -> Optional[str]:
    """Bring common key aliases in an initial tree to the node schema, in place.

    The model occasionally answers with its own keys (``id``/``field_name``,
    constraints as an object). Aliases are renamed; a tree whose nodes still
    lack ``node_id`` or ``name`` is reported so the attempt is retried instead
    of crashing later passes that index nodes by ``node_id``.
    """

    nodes = tree.get("nodes") if isinstance(tree, dict) else None
    if not isinstance(nodes, list):
        return "no node list"
    if not nodes:
        return None  # later passes already handle an empty tree
    for node in nodes:
        if not isinstance(node, dict):
            return "non-object node"
        if "node_id" not in node and "id" in node:
            node["node_id"] = node.pop("id")
        if "name" not in node:
            for alias in ("field_name", "node_name", "title"):
                if isinstance(node.get(alias), str) and node[alias].strip():
                    node["name"] = node.pop(alias)
                    break
        if isinstance(node.get("constraints"), dict):
            node["constraints"] = [f"{key}: {value}" for key, value in node["constraints"].items()
                                   if value not in (None, "", [], {})]
    missing = [i for i, node in enumerate(nodes) if "node_id" not in node or not str(node.get("name") or "").strip()]
    if missing:
        return f"{len(missing)} of {len(nodes)} nodes lack node_id or name"
    if tree.get("root_node_id") not in {node["node_id"] for node in nodes}:
        roots = [node["node_id"] for node in nodes if node.get("parent_id") in (None, "", -1)]
        if len(roots) != 1:
            return "root_node_id names no node and there is no single parentless node"
        tree["root_node_id"] = roots[0]
    return None


def _selector_tokens_match(selector_name: Any, node: Dict[str, Any]) -> bool:
    selector_tokens = _doc_name_token_set(selector_name, drop_stop=False)
    node_tokens = _doc_name_token_set(node.get("name"), drop_stop=False)
    if not selector_tokens or not node_tokens:
        return False
    # Type and presence/enable flags are distinct wire fields, even when they
    # share an "authentication" or other family name.
    if "type" in selector_tokens and "type" not in node_tokens:
        return False
    if ("sub" in selector_tokens) != ("sub" in node_tokens):
        return False
    selector_primary = _selector_match_token_set(selector_tokens)
    node_primary = _selector_match_token_set(node_tokens)
    if selector_primary == node_primary:
        return True
    if selector_primary <= node_primary:
        return True
    common = selector_primary & node_primary
    if len(common) >= max(2, min(len(selector_primary), len(node_primary))):
        return True
    if selector_tokens == node_tokens or selector_tokens <= node_tokens:
        return True
    return node_tokens <= selector_tokens and selector_primary <= node_primary


def _selector_node_match_score(selector_name: Any, node: Dict[str, Any]) -> int:
    if not _selector_tokens_match(selector_name, node):
        return -1
    selector_tokens = _doc_name_token_set(selector_name, drop_stop=False)
    node_tokens = _doc_name_token_set(node.get("name"), drop_stop=False)
    selector_primary = _selector_match_token_set(selector_tokens)
    node_primary = _selector_match_token_set(node_tokens)
    node_type = canonicalize_node_type(str(node.get("node_type") or ""))
    score = 0
    if node_type == "selector":
        score += 100
    elif node_type == "field":
        score += 40
    elif node_type in {"protocol", "header", "container", "payload", "variant", "tlv_seq"}:
        score -= 30
    if selector_primary == node_primary:
        score += 60
    elif selector_primary <= node_primary:
        score += 30
    elif node_primary < selector_primary:
        score -= 30
    score += len(selector_primary & node_primary) * 2
    if selector_tokens == node_tokens:
        score += 50
    elif selector_tokens <= node_tokens:
        score += 25
    elif node_tokens < selector_tokens:
        score -= 20
    score += len(selector_tokens & node_tokens)
    return score


def _node_parent(node: Dict[str, Any], nodes_by_id: Dict[Any, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    parent_id = node.get("parent_id")
    if parent_id is None:
        return None
    return nodes_by_id.get(parent_id) or nodes_by_id.get(str(parent_id))


def _find_selector_body_parent(
    selector_node: Dict[str, Any],
    nodes_by_id: Dict[Any, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    selector_parent = _node_parent(selector_node, nodes_by_id)
    if not isinstance(selector_parent, dict):
        return None
    selector_parent_type = canonicalize_node_type(str(selector_parent.get("node_type") or ""))
    if selector_parent_type == "header":
        grandparent = _node_parent(selector_parent, nodes_by_id)
        if isinstance(grandparent, dict):
            sibling_candidates: List[Tuple[int, Dict[str, Any]]] = []
            for sibling_id in grandparent.get("children_ids") or []:
                sibling = nodes_by_id.get(sibling_id) or nodes_by_id.get(str(sibling_id))
                if not isinstance(sibling, dict) or sibling.get("node_id") == selector_parent.get("node_id"):
                    continue
                node_type = canonicalize_node_type(str(sibling.get("node_type") or ""))
                if node_type not in {"payload", "container", "variant", "tlv_seq"}:
                    continue
                tokens = _doc_name_token_set(sibling.get("name"), drop_stop=False)
                score = 0
                if "body" in tokens:
                    score += 100
                if "payload" in tokens or "data" in tokens or "value" in tokens:
                    score += 50
                if "header" in tokens:
                    score -= 50
                if node_type in {"payload", "container", "tlv_seq"}:
                    score += 10
                if score > 0:
                    sibling_candidates.append((score, sibling))
            if sibling_candidates:
                return sorted(sibling_candidates, key=lambda item: item[0], reverse=True)[0][1]
    preferred_tokens = {"value", "body", "payload", "data"}
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for child_id in selector_parent.get("children_ids") or []:
        child = nodes_by_id.get(child_id) or nodes_by_id.get(str(child_id))
        if not isinstance(child, dict) or child.get("node_id") == selector_node.get("node_id"):
            continue
        node_type = str(child.get("node_type") or "").lower()
        if node_type not in {"payload", "container", "variant", "field"}:
            continue
        tokens = _doc_name_token_set(child.get("name"), drop_stop=False)
        score = 0
        if tokens & preferred_tokens:
            score += 50 + len(tokens & preferred_tokens)
        if "length" in tokens or "type" in tokens:
            score -= 20
        if node_type in {"payload", "container"}:
            score += 10
        elif node_type == "variant" and tokens & preferred_tokens:
            score += 10
        if score > 0:
            candidates.append((score, child))
    if candidates:
        return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
    return selector_parent


def _condition_edge_covers_value(edge: Dict[str, Any], selector_id: Any, value: int) -> bool:
    if edge.get("rel") != "condition_on" or str(edge.get("src")) != str(selector_id):
        return False
    formula = str(edge.get("formula") or "")
    if not formula:
        return False
    expression = formula.replace("&&", " and ").replace("||", " or ")
    unsupported_selector = False

    def replace_val(match: re.Match[str]) -> str:
        nonlocal unsupported_selector
        if str(match.group(1)) != str(selector_id):
            unsupported_selector = True
        return str(int(value))

    expression = re.sub(r"\bval\s*\(\s*([^)]+?)\s*\)", replace_val, expression)
    if not unsupported_selector and re.fullmatch(r"[0-9A-Za-z_\s<>=!()+\-*/%.]+", expression or ""):
        try:
            return bool(eval(expression, {"__builtins__": {}}, {}))
        except Exception:
            pass
    nums = {
        int(raw, 16) if raw.lower().startswith("0x") else int(raw)
        for raw in re.findall(r"0x[0-9a-fA-F]+|\d+", formula)
    }
    equality_numbers = {
        int(raw, 16) if raw.lower().startswith("0x") else int(raw)
        for raw in re.findall(r"==\s*(0x[0-9a-fA-F]+|\d+)", formula)
    }
    if equality_numbers:
        return int(value) in equality_numbers
    return int(value) in nums and "!=" not in formula


def _packet_field_covered_by_selector_route(
    parent_node: Dict[str, Any],
    packet_format: Dict[str, Any],
    field_obj: Dict[str, Any],
    section: Dict[str, Any],
    nodes: Sequence[Dict[str, Any]],
    nodes_by_id: Dict[Any, Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
) -> bool:
    if _generic_exception_alias_field_covered_by_parent(parent_node, packet_format, field_obj, edges):
        return True
    try:
        from ..validation.doc_constraint_coverage import (
            _selector_name_from_context,
            _selector_values_from_field,
        )
    except Exception:
        return False

    selector_name = _selector_name_from_context(packet_format, field_obj, section)
    values = _selector_values_from_field(field_obj)
    if not selector_name or not values:
        return False
    parent_id = parent_node.get("node_id")
    if parent_id is None:
        return False
    selector_nodes = [
        node
        for _, node in sorted(
            (
                (_selector_node_match_score(selector_name, node), node)
                for node in nodes
                if isinstance(node, dict) and node.get("node_id") is not None
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if _ >= 0
    ]
    for selector_node in selector_nodes:
        selector_id = selector_node.get("node_id")
        for edge in edges:
            if not isinstance(edge, dict) or edge.get("rel") != "condition_on":
                continue
            if str(edge.get("src")) != str(selector_id) or str(edge.get("dst")) != str(parent_id):
                continue
            if not _message_type_scope_compatible(edge.get("message_type"), _infer_packet_format_message_type(packet_format)):
                continue
            if any(_condition_edge_covers_value(edge, selector_id, int(value)) for value in values):
                return True
    return False


def _selector_obligation_values(entry: Dict[str, Any]) -> Tuple[int, ...]:
    existing_values = entry.get("selector_values")
    if existing_values is None:
        existing_values = [entry.get("selector_value")]
    elif not isinstance(existing_values, (list, tuple, set)):
        existing_values = [existing_values]
    try:
        return tuple(sorted({int(value) for value in existing_values if value is not None}))
    except (TypeError, ValueError):
        return ()


def _source_structure_gap_counts(
    tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    return (
        len(_missing_packet_format_names(tree, sections)),
        len(_missing_selector_obligation_names(tree, sections)),
    )


_STATIC_ERROR_COUNT_FAILURE = 10**9


def _static_error_count_inline(tree: Dict[str, Any]) -> int:
    previous_skip = os.environ.get("PARGEN_SKIP_SEMANTIC_OVERLAPS")
    previous_timeout = os.environ.get("PARGEN_Z3_TIMEOUT_MS")
    try:
        os.environ["PARGEN_SKIP_SEMANTIC_OVERLAPS"] = "1"
        os.environ["PARGEN_Z3_TIMEOUT_MS"] = os.getenv("PARGEN_SOURCE_SCAFFOLD_Z3_TIMEOUT_MS", "100")
        report = validate_protocol_tree(json.dumps({"protocol_tree": tree}))
        return sum(issue.severity == Severity.ERROR for issue in (getattr(report, "issues", {}) or {}).values())
    except Exception:
        return _STATIC_ERROR_COUNT_FAILURE
    finally:
        if previous_skip is None:
            os.environ.pop("PARGEN_SKIP_SEMANTIC_OVERLAPS", None)
        else:
            os.environ["PARGEN_SKIP_SEMANTIC_OVERLAPS"] = previous_skip
        if previous_timeout is None:
            os.environ.pop("PARGEN_Z3_TIMEOUT_MS", None)
        else:
            os.environ["PARGEN_Z3_TIMEOUT_MS"] = previous_timeout


def _static_error_count_worker(tree: Dict[str, Any], queue: Any) -> None:
    queue.put(_static_error_count_inline(tree))


def _static_error_count(tree: Dict[str, Any]) -> int:
    timeout_sec = _env_positive_float("STEP2_SOURCE_SCAFFOLD_STATIC_GATE_TIMEOUT_SEC", 10.0)
    context_name = (os.getenv("STEP2_SOURCE_SCAFFOLD_STATIC_GATE_MP_CONTEXT") or "fork").strip()
    try:
        ctx = multiprocessing.get_context(context_name)
    except Exception:
        ctx = multiprocessing.get_context()
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_static_error_count_worker, args=(copy.deepcopy(tree), queue))
    proc.daemon = True
    try:
        proc.start()
        proc.join(timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(1.0)
            return _STATIC_ERROR_COUNT_FAILURE
        if proc.exitcode not in (0, None):
            return _STATIC_ERROR_COUNT_FAILURE
        try:
            return int(queue.get_nowait())
        except Exception:
            return _STATIC_ERROR_COUNT_FAILURE
    except Exception:
        return _static_error_count_inline(tree)
    finally:
        try:
            queue.close()
        except Exception:
            pass


def _normalize_expr_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _refine_response_payload_object(response_payload: Any) -> Optional[Dict[str, Any]]:
    return response_payload if isinstance(response_payload, dict) else None


def _selector_value_from_observed_variant_name(name: Any, selector_node: Dict[str, Any]) -> Optional[int]:
    text = str(name or "")
    if not text:
        return None
    selector_tokens = _doc_name_token_set(selector_node.get("name"), drop_stop=False)
    fc_match = re.search(r"(?:^|[^A-Za-z0-9])FC\s*([0-9A-Fa-f]{1,4})(?:[^A-Za-z0-9]|$)", text)
    if fc_match and {"function", "code"} <= selector_tokens:
        raw = fc_match.group(1)
        try:
            return int(raw, 16) if re.search(r"[A-Fa-f]", raw) else int(raw, 10)
        except ValueError:
            return None

    name_tokens = _doc_name_token_set(text, drop_stop=False)
    selector_name_tokens = selector_tokens - {"code", "selector", "type", "kind", "id"}
    if selector_name_tokens and not (selector_name_tokens & name_tokens):
        return None
    numbers = [
        match.group(1)
        for match in re.finditer(
            r"(?:^|[^0-9A-Za-z])(0x[0-9A-Fa-f]+|\d{1,6})(?=$|[^0-9A-Za-z])",
            text,
        )
    ]
    if not numbers:
        return None
    raw = numbers[-1]
    try:
        return int(raw, 16) if raw.lower().startswith("0x") else int(raw, 10)
    except ValueError:
        return None


def _prune_or_guard_unrouted_observed_selector_variants(tree: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Fix LLM-created observed selector variants that lack selector routing.

    Initial traffic-grounded trees sometimes add placeholder branches such as
    ``Observed_Type_7_Response`` under the selector body but forget to attach a
    ``condition_on`` route.  If a source-backed sibling already covers the same
    selector value and direction, the observed placeholder is redundant and
    should be removed.  Otherwise the placeholder is kept but guarded by the
    shared selector.  The rule is generic: it uses the existing selector scope,
    message direction, and numeric value in the observed branch label.
    """

    candidate = copy.deepcopy(normalize_protocol_tree(tree))
    nodes = candidate.get("nodes")
    if not isinstance(nodes, list):
        return candidate, []
    edges = candidate.setdefault("edges", [])
    if not isinstance(edges, list):
        edges = []
        candidate["edges"] = edges

    nodes_by_id: Dict[Any, Dict[str, Any]] = {
        node.get("node_id"): node
        for node in nodes
        if isinstance(node, dict) and node.get("node_id") is not None
    }
    nodes_by_id.update({str(key): value for key, value in list(nodes_by_id.items())})

    def id_key(value: Any) -> str:
        return str(value)

    def incoming_condition_edges(node_id: Any) -> List[Dict[str, Any]]:
        return [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and edge.get("rel") == "condition_on"
            and id_key(edge.get("dst")) == id_key(node_id)
        ]

    selectors_by_body_parent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if canonicalize_node_type(str(node.get("node_type") or "")) != "selector":
            continue
        body_parent = _find_selector_body_parent(node, nodes_by_id)
        if isinstance(body_parent, dict) and body_parent.get("node_id") is not None:
            selectors_by_body_parent[id_key(body_parent.get("node_id"))].append(node)

    remove_ids: Set[str] = set()
    changes: List[str] = []
    existing_edge_keys = {
        (
            id_key(edge.get("src")),
            id_key(edge.get("dst")),
            str(edge.get("rel") or ""),
            _normalize_expr_text(edge.get("formula")),
            _normalize_message_type_scope(edge.get("message_type")),
        )
        for edge in edges
        if isinstance(edge, dict)
    }

    for node in list(nodes):
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if node_id is None or id_key(node_id) in remove_ids:
            continue
        if canonicalize_node_type(str(node.get("node_type") or "")) != "variant":
            continue
        name = str(node.get("name") or "")
        source = str(node.get("source") or "").lower()
        if "observed" not in name.lower() and "observed" not in source and "traffic" not in source:
            continue
        if incoming_condition_edges(node_id):
            continue
        parent_id = node.get("parent_id")
        selectors = selectors_by_body_parent.get(id_key(parent_id), [])
        if not selectors:
            continue
        message_type = _normalize_message_type_scope(
            node.get("message_type") or _source_format_role(name) or "bidirectional"
        )
        matched_selector: Optional[Dict[str, Any]] = None
        matched_value: Optional[int] = None
        for selector in sorted(
            selectors,
            key=lambda item: _selector_node_match_score(item.get("name"), node),
            reverse=True,
        ):
            value = _selector_value_from_observed_variant_name(name, selector)
            if value is None:
                continue
            matched_selector = selector
            matched_value = value
            break
        if matched_selector is None or matched_value is None:
            continue
        selector_id = matched_selector.get("node_id")
        has_guarded_sibling = False
        for edge in edges:
            if not isinstance(edge, dict) or edge.get("rel") != "condition_on":
                continue
            if id_key(edge.get("src")) != id_key(selector_id):
                continue
            if id_key(edge.get("dst")) == id_key(node_id):
                continue
            dst_node = nodes_by_id.get(edge.get("dst")) or nodes_by_id.get(id_key(edge.get("dst")))
            if not isinstance(dst_node, dict):
                continue
            if id_key(dst_node.get("parent_id")) != id_key(parent_id):
                continue
            if canonicalize_node_type(str(dst_node.get("node_type") or "")) != "variant":
                continue
            if not _message_type_scope_compatible(edge.get("message_type"), message_type):
                continue
            if _condition_edge_covers_value(edge, selector_id, int(matched_value)):
                has_guarded_sibling = True
                break
        if has_guarded_sibling:
            stack = [node_id]
            while stack:
                current_id = stack.pop()
                current_key = id_key(current_id)
                if current_key in remove_ids:
                    continue
                remove_ids.add(current_key)
                current_node = nodes_by_id.get(current_id) or nodes_by_id.get(current_key)
                if isinstance(current_node, dict):
                    stack.extend(current_node.get("children_ids") or [])
            changes.append(f"pruned unguarded observed variant {name}")
            continue

        formula = f"val({selector_id}) == {int(matched_value)}"
        edge_key = (
            id_key(selector_id),
            id_key(node_id),
            "condition_on",
            _normalize_expr_text(formula),
            message_type,
        )
        if edge_key not in existing_edge_keys:
            edges.append(
                {
                    "src": selector_id,
                    "dst": node_id,
                    "rel": "condition_on",
                    "formula": formula,
                    "message_type": message_type,
                }
            )
            existing_edge_keys.add(edge_key)
            changes.append(f"guarded unguarded observed variant {name}")

    if remove_ids:
        candidate["nodes"] = [
            node
            for node in nodes
            if isinstance(node, dict) and id_key(node.get("node_id")) not in remove_ids
        ]
        for node in candidate["nodes"]:
            if not isinstance(node, dict):
                continue
            node["children_ids"] = [
                child_id
                for child_id in node.get("children_ids") or []
                if id_key(child_id) not in remove_ids
            ]
        candidate["edges"] = [
            edge
            for edge in candidate.get("edges") or []
            if isinstance(edge, dict)
            and id_key(edge.get("src")) not in remove_ids
            and id_key(edge.get("dst")) not in remove_ids
        ]
    if changes:
        synchronize_parent_child_links(candidate)
    return candidate, changes


def _node_names(tree: Dict[str, Any]) -> List[str]:
    root = tree.get("protocol_tree") if isinstance(tree.get("protocol_tree"), dict) else tree
    names: List[str] = []
    for node in root.get("nodes", []) or []:
        if isinstance(node, dict):
            norm = _normalize_doc_field_name(node.get("name", ""))
            if norm:
                names.append(norm)
    return names


def _packet_field_is_shared_header(parent, packet_format, field, section, nodes, nodes_by_id):
    guard = _packet_format_selector_guard(packet_format, section)
    position = _source_field_bit_start(field)
    if not guard or position is None:
        return False
    selectors = sorted([n for n in nodes if _selector_node_match_score(guard[0], n) >= 0],
                       key=lambda n: _selector_node_match_score(guard[0], n), reverse=True)
    source_selector = next((f for f in packet_format.get("fields", []) if isinstance(f, dict)
                            and _selector_tokens_match(guard[0], {"name": f.get("field_name") or f.get("name")})), None)
    if not selectors or source_selector is None:
        return False
    selector_position = _source_field_bit_start(source_selector)
    selector_start = static_property(nodes_by_id, selectors[0]["node_id"], "bit_start")
    parent_start = static_property(nodes_by_id, parent["node_id"], "bit_start")
    if selector_position is None or selector_start is None or parent_start is None:
        return False
    start = selector_start - selector_position + position
    width = _coerce_packet_field_size_bits(field)
    if not isinstance(width, int) or start + width > parent_start:
        return False
    expected_name = field.get("field_name") or field.get("name")
    return any(_selector_tokens_match(expected_name, n)
               and static_property(nodes_by_id, n["node_id"], "bit_start") == start
               and static_property(nodes_by_id, n["node_id"], "size_bits") == width for n in nodes)


def _leading_source_position(text: Any) -> Optional[int]:
    match = re.match(r"^\s*(\d+)\s*(?:[-+]|to\b|$)", str(text or ""))
    return int(match[1]) if match else None


def _packet_format_documented_starts(fields: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    """Documented start bit per field (keyed by id(field)), relative to the format.

    Stage A tables either give format-relative bit ranges ("32-63") or give each
    field's bits inside its own bytes ("0-7" on every row) next to a byte offset.
    A column counts as a layout only when its starts strictly increase in row
    order; otherwise the byte column is tried; otherwise nothing is returned.
    """

    for key, scale in (("bit_position", 1), ("byte_position", 8)):
        known = [(_leading_source_position(field.get(key)), field) for field in fields]
        known = [(start, field) for start, field in known if start is not None]
        starts = [start for start, _ in known]
        if len(starts) >= 2 and all(a < b for a, b in zip(starts, starts[1:])):
            return {id(field): start * scale for start, field in known}
    return {}


def _packet_field_node_under(parent: Dict[str, Any], nodes_by_id: Dict[Any, Dict[str, Any]], field_name: Any) -> Optional[Dict[str, Any]]:
    """The one descendant of ``parent`` named exactly like the field, else None."""

    wanted = _doc_name_token_set(field_name)
    if not wanted:
        return None
    hits: List[Dict[str, Any]] = []
    stack = list(_child_ids_for_parent(parent, nodes_by_id))
    seen: Set[str] = set()
    while stack:
        child_id = stack.pop(0)
        if str(child_id) in seen:
            continue
        seen.add(str(child_id))
        child = nodes_by_id.get(child_id) or nodes_by_id.get(str(child_id))
        if not isinstance(child, dict):
            continue
        stack.extend(child.get("children_ids") or [])
        if _doc_name_token_set(child.get("name")) == wanted:
            hits.append(child)
    return hits[0] if len(hits) == 1 else None


def _packet_format_frame_offset(parent, packet_format, section, nodes, nodes_by_id, documented) -> Optional[int]:
    """Absolute bit where the format's documented position 0 lies in the tree.

    Formats routed by a selector are anchored on the selector node, which sits
    outside the variant body, so a body shifted by an oversized shared header
    is visible. Other formats are anchored on their first documented field.
    """

    fields = [field for field in packet_format.get("fields") or [] if isinstance(field, dict)]
    guard = _packet_format_selector_guard(packet_format, section)
    if guard:
        selectors = sorted([n for n in nodes if _selector_node_match_score(guard[0], n) >= 0],
                           key=lambda n: _selector_node_match_score(guard[0], n), reverse=True)
        source_selector = next((f for f in fields
                                if _selector_tokens_match(guard[0], {"name": f.get("field_name") or f.get("name")})), None)
        if not selectors or source_selector is None or id(source_selector) not in documented:
            return None
        selector_start = static_property(nodes_by_id, selectors[0]["node_id"], "bit_start")
        return None if selector_start is None else selector_start - documented[id(source_selector)]
    anchored = []
    for field in fields:
        if id(field) not in documented:
            continue
        node = _packet_field_node_under(parent, nodes_by_id, field.get("field_name") or field.get("name"))
        start = static_property(nodes_by_id, node["node_id"], "bit_start") if node else None
        if start is not None:
            anchored.append((documented[id(field)], start))
    if not anchored:
        return None
    position, start = min(anchored)
    return start - position


def _packet_field_offset_mismatch(parent, packet_format, field, display_name, frame, documented, nodes_by_id) -> Optional[str]:
    """Describe a represented field whose tree offset contradicts the document."""

    if frame is None or id(field) not in documented:
        return None
    node = (_packet_field_node_under(parent, nodes_by_id, display_name)
            or _packet_field_node_under(parent, nodes_by_id, field.get("field_name") or field.get("name")))
    start = static_property(nodes_by_id, node["node_id"], "bit_start") if node else None
    expected = documented[id(field)]
    if start is None or start - frame == expected:
        return None
    format_name = str(packet_format.get("format_name") or "").strip()
    return (f"{display_name} at wrong offset: {format_name} documents it at bit {expected}, "
            f"the tree places it at bit {start - frame} (node {node['node_id']}); "
            "fix its bit_start or the enclosing header/container layout")


def _missing_packet_format_names(tree: Dict[str, Any], sections: Sequence[Dict[str, Any]]) -> List[str]:
    """Return focused packet-format field names not represented by concrete tree nodes.

    This is a generic no-op guard for TLV/IE-heavy specs.  A generic TLV catch-all
    often satisfies syntax validation while still losing source-backed typed IE
    names; reject empty refinement patches when focused structured fields are
    absent from the current tree.
    """

    current = normalize_protocol_tree(tree)
    existing = _node_names(current)
    nodes_by_id: Dict[Any, Dict[str, Any]] = {
        node.get("node_id"): node
        for node in current.get("nodes") or []
        if isinstance(node, dict) and node.get("node_id") is not None
    }
    nodes_by_id.update({str(key): value for key, value in list(nodes_by_id.items())})
    nodes = [node for node in current.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in current.get("edges") or [] if isinstance(edge, dict)]
    generic_names = {
        "information element",
        "information element type",
        "information element length",
        "information element value",
        "ie type",
        "ie length",
        "ie value",
        "tlv type",
        "tlv length",
        "tlv value",
        "message type",
        "message length",
    }
    missing: List[str] = []
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        if _context_only_non_wire_section(section):
            continue
        for pf in section.get("packet_formats") or []:
            if not isinstance(pf, dict):
                continue
            if not _packet_format_is_strict_gap_source(pf):
                continue
            format_name = str(pf.get("format_name") or "").strip()
            parent = _find_packet_format_structural_parent(current, pf, section) if format_name else None
            fields = [
                field
                for field in pf.get("fields") or []
                if isinstance(field, dict)
            ]
            # Presence alone is not enough: a field that exists but sits at the
            # wrong documented offset (e.g. every body shifted by an oversized
            # shared header) passes traffic checks whose anchors never reach it.
            documented = _packet_format_documented_starts(fields)
            frame = (_packet_format_frame_offset(parent, pf, section, nodes, nodes_by_id, documented)
                     if isinstance(parent, dict) and documented else None)
            fields.extend(_extract_bitfield_flag_specs(section, pf))
            for field in fields:
                if not isinstance(field, dict):
                    continue
                if not _strict_gap_packet_field_candidate(field):
                    continue
                raw_name = field.get("field_name") or field.get("name")
                norm = _normalize_doc_field_name(raw_name)
                if not norm or norm in generic_names:
                    continue
                display_name = _scaffold_field_display_name_for_parent(format_name, field, parent)
                if isinstance(parent, dict):
                    if field.get("_source_kind") == "bitfield_flag":
                        represented = _packet_field_exact_represented_under(parent, nodes_by_id, display_name)
                    else:
                        represented = _packet_field_represented_under(parent, nodes_by_id, display_name)
                        if not represented:
                            represented = _packet_field_covered_by_selector_route(
                                parent,
                                pf,
                                field,
                                section,
                                nodes,
                                nodes_by_id,
                                edges,
                            )
                        if not represented:
                            represented = _packet_field_is_shared_header(parent, pf, field, section, nodes, nodes_by_id)
                else:
                    # A document with no identifiable structural parent may be
                    # a cross-cutting field table. Keep legacy alias resolution
                    # there; once a parent is known, sibling matches are forbidden.
                    display_norm = _normalize_doc_field_name(display_name)
                    represented = _packet_field_represented_anywhere(nodes_by_id, display_name)
                    if not represented:
                        represented = any(display_norm in name or name in display_norm for name in existing)
                if not represented:
                    missing.append(str(display_name or raw_name))
                elif isinstance(parent, dict) and field.get("_source_kind") != "bitfield_flag":
                    mismatch = _packet_field_offset_mismatch(
                        parent, pf, field, display_name, frame, documented, nodes_by_id)
                    if mismatch:
                        missing.append(mismatch)
    return sorted(set(missing))


def _packet_format_coverage_improved(
    before_tree: Dict[str, Any],
    after_tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
) -> bool:
    """Return true when a candidate patch reduces focused structured-field gaps.

    The refine stage is allowed to make small incremental patches, but when the
    focused source sections contain structured `packet_formats`, a patch that
    leaves the exact same source-backed field names missing should not be
    accepted as useful progress.  This gate is deliberately protocol-agnostic:
    it compares only document-derived field names against concrete tree nodes.
    """

    before_missing = _missing_packet_format_names(before_tree, sections)
    if not before_missing:
        return True
    after_missing = _missing_packet_format_names(after_tree, sections)
    return len(after_missing) < len(before_missing)


def _missing_selector_obligation_names(tree: Dict[str, Any], sections: Sequence[Dict[str, Any]]) -> List[str]:
    """Return source-backed selector variants not represented in the tree.

    This is deliberately protocol-agnostic.  The source document must expose a
    selector/value table or field constraint first; generic catch-all payloads
    such as IE_Value/TLV_Value do not satisfy a value-specific variant body.
    """

    try:
        from ..validation.doc_constraint_coverage import (
            find_uncovered_selector_variant_obligations,
        )
    except Exception:
        return []

    labels: List[str] = []
    try:
        issues = find_uncovered_selector_variant_obligations(tree, sections)
    except Exception:
        return []
    for issue in issues or []:
        for expected in issue.get("expected_variants") or []:
            if not isinstance(expected, dict):
                continue
            selector = (
                str(expected.get("selector_name") or issue.get("selector") or "").strip()
                or "selector"
            )
            values = expected.get("selector_values") or issue.get("missing_selector_values") or []
            if isinstance(values, (str, int)):
                value_text = str(values)
            else:
                value_text = ",".join(str(v) for v in values)
            variant = (
                str(expected.get("canonical_variant_name") or expected.get("variant_name") or "").strip()
            )
            if variant:
                labels.append(f"{selector}={value_text} -> {variant}")
    return sorted(set(labels))


def _selector_obligation_coverage_improved(
    before_tree: Dict[str, Any],
    after_tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
) -> bool:
    before_missing = set(_missing_selector_obligation_names(before_tree, sections))
    after_missing = set(_missing_selector_obligation_names(after_tree, sections))
    if not before_missing:
        return not after_missing
    if after_missing - before_missing:
        return False
    return len(after_missing) < len(before_missing)


_COVERAGE_RESIDUAL_CACHE: Dict[str, Tuple[float, Any]] = {}


def _traffic_source_backed_gap_guard_active(traffic_file: Any = None) -> bool:
    if not traffic_file:
        return False
    return _env_flag("STEP2_TRAFFIC_STRICT_SOURCE_BACKED_REFINE_GAPS", "1")


def _should_fail_source_backed_refine_gap(
    source_structure_gaps: Sequence[str],
    *,
    traffic_file: Any = None,
) -> bool:
    return bool(source_structure_gaps) and (
        _env_flag("STEP2_STRICT_SOURCE_BACKED_REFINE_GAPS", "1")
        or _traffic_source_backed_gap_guard_active(traffic_file)
    )


def _source_gap_unresolved_after_adjudication(
    source_structure_gaps: Sequence[str],
    adjudication_decisions: Sequence[Dict[str, Any]],
) -> List[str]:
    """Return only gaps that an adjudicator explicitly says are still missing.

    The deterministic detector is intentionally high-recall.  Its output must
    not directly become a hard failure after an LLM adjudicator says the current tree
    already covers a candidate gap, or that the match is ambiguous enough to
    avoid hard-failing the run.
    """

    if not source_structure_gaps:
        return []
    decision_by_gap: Dict[str, str] = {}
    decision_by_norm: Dict[str, str] = {}
    for decision in adjudication_decisions or []:
        if not isinstance(decision, dict):
            continue
        gap = str(decision.get("gap") or decision.get("candidate_gap") or "").strip()
        status = str(decision.get("status") or decision.get("decision") or "").strip().lower()
        if not gap or not status:
            continue
        normalized_status = status.replace("-", "_").replace(" ", "_")
        decision_by_gap[gap] = normalized_status
        decision_by_norm[_normalize_doc_field_name(gap)] = normalized_status

    unresolved: List[str] = []
    for gap in source_structure_gaps:
        gap_text = str(gap)
        status = decision_by_gap.get(gap_text) or decision_by_norm.get(_normalize_doc_field_name(gap_text))
        if not status:
            unresolved.append(gap_text)
            continue
        if status in {"missing", "uncovered", "not_covered", "needs_patch", "needs_repair"}:
            unresolved.append(gap_text)
    return unresolved


def _source_gap_candidate_nodes(
    tree: Dict[str, Any],
    source_structure_gaps: Sequence[str],
    *,
    max_nodes_per_gap: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    nodes = [node for node in tree.get("nodes", []) or [] if isinstance(node, dict)]
    by_id = {str(node.get("node_id")): node for node in nodes if node.get("node_id") is not None}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for gap in source_structure_gaps or []:
        gap_text = str(gap)
        wanted = _doc_name_token_set(gap_text)
        scored: List[Tuple[int, int, Dict[str, Any]]] = []
        for node in nodes:
            node_id = node.get("node_id")
            if node_id is None:
                continue
            node_tokens = _doc_name_token_set(node.get("name"))
            parent = by_id.get(str(node.get("parent_id"))) or {}
            parent_tokens = _doc_name_token_set(parent.get("name"))
            source_tokens = _doc_name_token_set(node.get("source"))
            desc_tokens = _doc_name_token_set(node.get("description"))
            overlap = len(wanted & (node_tokens | parent_tokens | source_tokens | desc_tokens))
            if overlap <= 0:
                continue
            type_bonus = 2 if str(node.get("node_type") or "").lower() in {"field", "length", "selector"} else 0
            scored.append((overlap + type_bonus, overlap, node))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        candidates: List[Dict[str, Any]] = []
        for _, _, node in scored[:max_nodes_per_gap]:
            parent = by_id.get(str(node.get("parent_id"))) or {}
            candidates.append(
                {
                    "node_id": node.get("node_id"),
                    "name": node.get("name"),
                    "node_type": node.get("node_type"),
                    "parent_id": node.get("parent_id"),
                    "parent_name": parent.get("name"),
                    "bit_start": node.get("bit_start"),
                    "size_bits": node.get("size_bits"),
                    "data_type": node.get("data_type"),
                    "source": node.get("source"),
                    "description": node.get("description"),
                    "constraints": node.get("constraints"),
                }
            )
        out[gap_text] = candidates
    return out


def _source_gap_section_evidence(sections: Sequence[Dict[str, Any]], *, max_chars: int = 9000) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    used = 0
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        entry: Dict[str, Any] = {}
        for key in ("source_file", "number", "title", "section_title", "summary", "content"):
            value = section.get(key)
            if isinstance(value, str) and value.strip():
                entry[key] = value[:1200]
        packet_formats = section.get("packet_formats")
        if isinstance(packet_formats, list):
            entry["packet_formats"] = [_summarize_packet_format(pf, max_fields=20) for pf in packet_formats[:8]]
        field_definitions = section.get("field_definitions")
        if isinstance(field_definitions, list):
            entry["field_definitions"] = [
                {
                    key: field.get(key)
                    for key in ("field_name", "name", "description", "bit_position", "byte_position", "size", "data_type", "constraints")
                    if isinstance(field, dict) and key in field
                }
                for field in field_definitions[:20]
                if isinstance(field, dict)
            ]
        text = json.dumps(entry, ensure_ascii=False)
        if used + len(text) > max_chars and evidence:
            break
        used += len(text)
        evidence.append(entry)
    return evidence


_CONTEXT_ONLY_DATA_SECTION_TITLE_RE = re.compile(
    r"\b(data\s+model|addressing\s+model|data\s+encoding|logical\s+table|object\s+model)\b",
    flags=re.IGNORECASE,
)


def _context_only_non_wire_section(section: Dict[str, Any]) -> bool:
    if not isinstance(section, dict):
        return False
    section_type = str(section.get("section_type") or "").strip().lower()
    if section_type != "data_format":
        return False
    text = " ".join(
        str(section.get(key) or "")
        for key in ("number", "title", "section_title", "heading", "summary", "content")
    )
    if not _CONTEXT_ONLY_DATA_SECTION_TITLE_RE.search(text):
        return False
    # Packet-format sections can still carry structured source context.  They
    # are skipped here only after the strict gap detector found no packet,
    # selector, or coverage obligations for the focused batch.
    return True


def _traffic_context_only_noop_allowed(
    focused_secs: Sequence[Dict[str, Any]],
    *,
    traffic_file: Any = None,
    has_source_structure_gaps: bool = False,
) -> bool:
    if not traffic_file or has_source_structure_gaps:
        return False
    focused = [sec for sec in (focused_secs or []) if isinstance(sec, dict)]
    if not focused:
        return False
    return all(_context_only_non_wire_section(sec) for sec in focused)


def _format_source_structure_gap_context(
    *,
    packet_fields: Sequence[str],
    selector_obligations: Sequence[str],
    max_items: int = 12,
) -> str:
    """Format detector-backed refine obligations for the focused batch."""

    groups = [
        ("packet/table fields", list(packet_fields or [])),
        ("selector variants", list(selector_obligations or [])),
    ]
    all_gaps = [gap for _, gaps in groups for gap in gaps]
    if not all_gaps:
        return "- None. Empty patch is allowed only if the focused source has no missing parseable structure."

    lines = [
        "## SOURCE-BACKED STRUCTURE GAPS (BEFORE THIS PATCH)",
        "The local detector found focused documentation rows or obligations that are NOT represented in the current snapshot.",
        "You MUST reduce at least one listed gap with a doc-grounded patch. You MUST NOT return an empty patch or no_change intent while any listed gap remains.",
        "A generic parent/header/payload/container or variant shell does not count as representing a listed leaf/body field; add a compatible leaf, typed body, selector route, or corrective source annotation.",
    ]
    emitted = 0
    for label, gaps in groups:
        if not gaps:
            continue
        lines.append(f"- Missing {label}:")
        for gap in gaps[: max(0, max_items - emitted)]:
            lines.append(f"  - {gap}")
            emitted += 1
            if emitted >= max_items:
                break
        if emitted >= max_items:
            break
    omitted = len(all_gaps) - emitted
    if omitted > 0:
        lines.append(f"  - ... {omitted} additional gap(s) omitted")
    return "\n".join(lines)


def _format_focused_source_labels(focused_secs: Sequence[Dict[str, Any]], *, max_items: int = 8) -> str:
    labels: List[str] = []
    for sec in focused_secs or []:
        if not isinstance(sec, dict):
            continue
        source_file = str(sec.get("source_file") or "").strip()
        number = str(sec.get("number") or "").strip()
        title = str(sec.get("title") or sec.get("section_title") or sec.get("heading") or "").strip()
        parts: List[str] = []
        if source_file and number:
            parts.append(f"{source_file}:{number}")
        elif source_file:
            parts.append(source_file)
        elif number:
            parts.append(number)
        if title:
            parts.append(title)
        label = " ".join(parts).strip()
        if label and label not in labels:
            labels.append(label)

    if not labels:
        return "- No focused source labels were available; use the focused section title or number in each new node's source."

    lines = [
        "## FOCUSED SOURCE LABELS",
        "For every new node in this patch, set `source` to one of these copyable focused-section labels:",
    ]
    for label in labels[:max_items]:
        lines.append(f"- {label}")
    omitted = len(labels) - min(len(labels), max_items)
    if omitted > 0:
        lines.append(f"- ... {omitted} additional label(s) omitted")
    return "\n".join(lines)


def _is_empty_refine_patch(patch: Any) -> bool:
    if not isinstance(patch, dict):
        return False
    meaningful_keys = (
        "new_nodes",
        "node_updates",
        "edge_adds",
        "edge_updates",
        "edge_removes",
        "nodes_to_remove",
        "new_edges",
    )
    return not any(bool(patch.get(key)) for key in meaningful_keys)


def _refine_section_digest(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loss-minimized documentation digest for refine-stage prompts.

    Refine patches need more than the free-text `content` summary; they must see
    the structured `packet_formats` tables to expand payload variants into leaf
    fields (e.g., Starting Address / Byte Count / data arrays).
    """

    digest: Dict[str, Any] = {
        "number": section.get("number", ""),
        "title": section.get("title", ""),
        "source_file": section.get("source_file", ""),
        "content": section.get("content", ""),
        "summary": section.get("summary", ""),
        "constraints": section.get("constraints", []) or [],
    }

    packet_formats = section.get("packet_formats") or []
    if isinstance(packet_formats, list):
        digest["packet_formats"] = [_summarize_packet_format(pf) for pf in packet_formats[:8]]
    else:
        digest["packet_formats"] = packet_formats

    field_defs = section.get("field_definitions") or []
    if isinstance(field_defs, list):
        digest["field_definitions"] = field_defs[:80]
    else:
        digest["field_definitions"] = field_defs

    return digest


def _validation_report_summary(report: Any) -> Dict[str, Any]:
    issues = getattr(report, "issues", {}) or {}
    error_issues = [
        issue
        for issue in issues.values()
        if str(getattr(getattr(issue, "severity", None), "name", getattr(issue, "severity", ""))) == "ERROR"
    ]
    return {
        "ok": bool(getattr(report, "ok", False)),
        "error_count": len(getattr(report, "errors", []) or []),
        "warning_count": len(getattr(report, "warnings", []) or []),
        "error_issue_count": len(error_issues),
        "errors": list(getattr(report, "errors", []) or [])[:20],
        "extras": list(getattr(report, "extras", []) or [])[:20],
        "traffic_successful_samples": int(getattr(report, "traffic_successful_samples", 0) or 0),
        "traffic_total_samples": int(getattr(report, "traffic_total_samples", 0) or 0),
        "traffic_total_coverage_gap_bits": int(getattr(report, "traffic_total_coverage_gap_bits", 0) or 0),
        "traffic_length_total_abs_error_bits": int(getattr(report, "traffic_length_total_abs_error_bits", 0) or 0),
        "traffic_overflow_length_bits": int(getattr(report, "traffic_overflow_length_bits", 0) or 0),
    }

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class TreeModificationAction(Enum):
    SPLIT_NODE = "split_node"
    MERGE_NODES = "merge_nodes"
    RESTRUCTURE_HIERARCHY = "restructure_hierarchy"
    ADD_CONSTRAINT = "add_constraint"
    ADJUST_POSITION = "adjust_position"
    CHANGE_TYPE = "change_type"
    ADD_DEPENDENCY = "add_dependency"
    REMOVE_NODE = "remove_node"
    INSERT_INTERMEDIATE = "insert_intermediate"

@dataclass
class FieldConstraint:
    constraint_type: str  # "value", "range", "enum", "pattern", "length", "dependency"
    constraint_value: Any
    description: str
    is_mandatory: bool = True
    source: str = ""
    confidence: float = 1.0
    validation_count: int = 0
    violation_count: int = 0

    def validate(self, value: Any) -> bool:
        try:
            if self.constraint_type == "value":
                return value == self.constraint_value
            elif self.constraint_type == "range":
                if isinstance(self.constraint_value, list) and len(self.constraint_value) == 2:
                    return self.constraint_value[0] <= value <= self.constraint_value[1]
            elif self.constraint_type == "enum":
                return value in self.constraint_value
            elif self.constraint_type == "pattern":
                return bool(re.match(self.constraint_value, str(value)))
            elif self.constraint_type == "length":
                return len(str(value)) == self.constraint_value
        except:
            pass
        return True

@dataclass
class FieldDependency:
    dependency_type: str
    target_field: str
    condition: Optional[str] = None
    formula: Optional[str] = None

@dataclass
class ProtocolNode:
    node_id: str
    name: str
    node_type: str
    description: str

    bit_start: Optional[Union[int, str]] = None
    size_bits: Optional[Union[int, str]] = None
    data_type: Optional[str] = None
    byte_order: str = "big"
    message_type: str = ""

    constraints: List[Any] = field(default_factory=list)
    dependencies: List[Any] = field(default_factory=list)

    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    source: str = ""
    source_section_ids: List[Any] = field(default_factory=list)
    extra_attributes: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    parse_success_count: int = 0
    parse_failure_count: int = 0
    value_distribution: Dict[str, int] = field(default_factory=dict)

    def calculate_effective_position(self, context: Dict[str, Any]) -> Tuple[int, int]:
        bit_start = self.bit_start
        size_bits = self.size_bits

        if isinstance(bit_start, str):
            try:
                if "." in bit_start and "+" in bit_start:
                    parts = bit_start.split("+")
                    field_ref = parts[0].strip()
                    offset = int(parts[1].strip())

                    field_name = field_ref.split(".")[0]
                    if field_name in context:
                        ref_end = context[field_name].get('end_bit', 0)
                        bit_start = ref_end + offset
                else:
                    bit_start = int(bit_start)
            except:
                bit_start = 0

        if isinstance(size_bits, str):
            try:
                if "val(" in size_bits:
                    match = re.search(r'val\((\w+)\)', size_bits)
                    if match:
                        ref_field = match.group(1)
                        if ref_field in context:
                            ref_value = context[ref_field].get('value', 0)
                            if "*8" in size_bits:
                                size_bits = ref_value * 8
                            else:
                                size_bits = ref_value
                else:
                    size_bits = int(size_bits)
            except:
                size_bits = 8

        return int(bit_start), int(size_bits)

@dataclass
class ProtocolTree:
    protocol_name: str
    root_node_id: str
    nodes: Dict[str, ProtocolNode]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: str

    _NODE_DROP_FIELDS = {
        "confidence_score",
        "parse_success_count",
        "parse_failure_count",
        "value_distribution",
    }

    def get_node(self, node_id: str) -> Optional[ProtocolNode]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[ProtocolNode]:
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]

    def get_all_fields(self) -> List[ProtocolNode]:
        return [node for node in self.nodes.values() if node.node_type == "field"]

    def get_tree_depth(self) -> int:
        def _get_depth(node_id: str, current_depth: int = 0) -> int:
            node = self.get_node(node_id)
            if not node or not node.children_ids:
                return current_depth
            return max(_get_depth(child_id, current_depth + 1) for child_id in node.children_ids)

        return _get_depth(self.root_node_id)

    def clone(self) -> 'ProtocolTree':
        return ProtocolTree(
            protocol_name=self.protocol_name,
            root_node_id=self.root_node_id,
            nodes=copy.deepcopy(self.nodes),
            edges=copy.deepcopy(self.edges),
            metadata=copy.deepcopy(self.metadata),
            created_at=self.created_at
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree into JSON-ready payload without extra telemetry fields."""

        def _node_sort_key(node: ProtocolNode) -> Any:
            node_id = getattr(node, "node_id", "")
            try:
                return (0, int(node_id))
            except Exception:
                return (1, str(node_id))

        def serialized_id(value):
            if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
                return int(value)
            return value

        serialized_nodes: List[Dict[str, Any]] = []
        for node in sorted(self.nodes.values(), key=_node_sort_key):
            node_dict = asdict(node)
            extras = node_dict.pop('extra_attributes', {})
            node_dict = {**extras, **node_dict}
            for drop_field in self._NODE_DROP_FIELDS:
                node_dict.pop(drop_field, None)
            # Ensure children_ids and similar collections always exist
            node_dict.setdefault("children_ids", [])
            node_dict.setdefault("constraints", [])
            node_dict.setdefault("dependencies", [])
            node_dict['node_id'] = serialized_id(node_dict['node_id'])
            node_dict['parent_id'] = serialized_id(node_dict.get('parent_id'))
            node_dict['children_ids'] = [serialized_id(value) for value in node_dict['children_ids']]
            serialized_nodes.append(node_dict)

        serialized_edges = copy.deepcopy(self.edges)
        for edge in serialized_edges:
            if isinstance(edge, dict):
                for key in ('src', 'dst'):
                    if key in edge:
                        edge[key] = serialized_id(edge[key])

        payload = {
            "protocol_name": self.protocol_name,
            "root_node_id": serialized_id(self.root_node_id),
            "nodes": serialized_nodes,
            "edges": serialized_edges,
        }
        if self.metadata:
            payload["metadata"] = copy.deepcopy(self.metadata)
        return payload


TAG_OPEN = '<attention priority="{priority}">'
TAG_CLOSE = "</attention>"
TAG_PATTERN = re.compile(r"</?attention(?:\s+[^>]*)?>", re.IGNORECASE)
def _wrap_with_attention(text: str, priority: str = "high", strip_existing: bool = False) -> str:
    """Wrap entire text with <attention> ... </attention> (optionally removing prior tags)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if strip_existing:
        text = TAG_PATTERN.sub("", text)
    # avoid double-wrapping if already has any attention tag
    if TAG_PATTERN.search(text):
        return text
    return f'{TAG_OPEN.format(priority=priority)}\n{text}\n{TAG_CLOSE}'

def mark_sections_by_batch(
    sections: List[Dict[str, Any]],
    batch_start: int,
    batch_size: int,
    *,
    text_key: str = "content",
    out_key: str = "content",
    priority: str = "high",
    strip_existing: bool = False,
    set_flag_key: str | None = "is_focused",
) -> List[Dict[str, Any]]:
    """
    Mark the current batch of sections by wrapping their entire text with <attention>...</attention>.
    No heuristics; just whole-section focus.

    Args:
      sections: list of section dicts
      batch_start: starting index of the focused batch (0-based)
      batch_size: number of sections to focus in this batch
      text_key: where to read raw text from (default: 'content')
      out_key: where to write marked text to (default: 'content', overwriting)
      priority: the 'priority' attribute value for the attention tag
      strip_existing: when True, remove any existing attention tags before wrapping
      set_flag_key: optional key to set boolean focus flag on each section (default: 'is_focused'); pass None to skip

    Returns:
      New list with updated sections (shallow-copied dicts).
    """
    n = len(sections or [])
    if n == 0 or batch_size <= 0:
        return list(sections or [])

    # Clamp focus range
    s = max(0, batch_start)
    e = min(n, batch_start + batch_size)

    out: List[Dict[str, Any]] = []
    for i, sec in enumerate(sections):
        # copy
        new_sec = dict(sec) if isinstance(sec, dict) else {"content": str(sec)}
        raw = new_sec.get(text_key, "")
        if s <= i < e:
            new_sec[out_key] = _wrap_with_attention(raw, priority=priority, strip_existing=strip_existing)
            if set_flag_key:
                new_sec[set_flag_key] = True
        else:
            # optionally strip tags from unfocused sections to keep prompt clean
            new_sec[out_key] = TAG_PATTERN.sub("", raw) if strip_existing else raw
            if set_flag_key:
                new_sec[set_flag_key] = False
        out.append(new_sec)
    return out

class GenerationLLMAgent:

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        # Keep Stage B aligned with Stage C's config mechanism: load keys from config
        # (if present) and prefer environment variables over baked-in defaults.
        load_api_keys(DEFAULT_API_CONFIG, set_env=True)

        raw_provider = (
            provider
            or os.getenv("PARGEN_STAGE_B_PROVIDER")
            or os.getenv("AI_PROVIDER")
            or os.getenv("LLM_PROVIDER")
            or "deepseek"
        )
        p = (raw_provider or "").strip().lower()
        if p in {"claude", "anthropic"}:
            p = "anthropic"
        elif p in {"openai"}:
            p = "openai"
        elif p in {"deepseek"}:
            p = "deepseek"
        self.provider = p or "deepseek"

        # Deterministic decoding unless overridden by pipeline args/env.
        self.temperature = float(temperature) if temperature is not None else 0.0
        # Refine patches the hard-error gate rolled back. The caller sees a
        # "no_change" tree in that case, so this is the only trace of what was lost.
        self.refine_rejections: List[Dict[str, Any]] = []
        # Source gaps the retry budget could not close. Recording them and moving
        # on is what lets a reproducible run finish; strict mode still raises.
        self.unresolved_source_gaps: List[Dict[str, Any]] = []
        # Cross-batch repair memory: issue id -> failed repair attempts. The same
        # unfixable issue otherwise gets re-attacked on every batch (observed
        # 62-76 hits of one id across a 41-batch run).
        self._issue_attempts: Dict[str, int] = {}
        self.deferred_issues: List[Dict[str, Any]] = []

        if self.provider == "openai":
            self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions").strip()
            self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            self.default_model = (model or os.getenv("OPENAI_MODEL") or "gpt-5-2025-08-07").strip()
            default_budget = 16384
        elif self.provider == "deepseek":
            self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
            self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions").strip()
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            self.default_model = (model or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash-0731").strip()
            default_budget = 8000 if self._is_deepseek_flash_model(self.default_model) else 64000
        else:
            # Claude / Anthropic
            self.provider = "anthropic"
            self.api_key = (api_key or os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
            self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages").strip()
            self.headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
                "anthropic-beta": os.getenv("ANTHROPIC_BETA", "prompt-caching-2024-07-31"),
            }
            self.default_model = (
                (model or os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL") or "claude-sonnet-4-6")
                .strip()
            )
            default_budget = 64000

        self.max_tokens = int(max_tokens) if max_tokens is not None else int(default_budget)

        self.connect_timeout = int(os.getenv("AI_CONNECT_TIMEOUT", "10"))
        self.read_timeout = int(os.getenv("AI_READ_TIMEOUT", str(DEFAULT_AI_READ_TIMEOUT_SEC)))
        self.provider_wall_timeout = _env_positive_float(
            "AI_PROVIDER_WALL_TIMEOUT_SEC",
            DEFAULT_AI_PROVIDER_WALL_TIMEOUT_SEC,
        )
        self.timeout = self.read_timeout
        self.max_retries = int(os.getenv('AI_MAX_RETRIES', '2'))
        self.retry_delay = int(os.getenv('AI_RETRY_DELAY', '3'))
        self.allow_deepseek_reasoner_fallback = os.getenv(
            "STEP2_ALLOW_DEEPSEEK_REASONER_FALLBACK", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

        self.cache_dir = str(STEP2_CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.input_signature: Optional[Dict[str, Any]] = None

    def _requested_max_tokens(self, requested: Optional[int] = None) -> int:
        try:
            return max(1, int(requested) if requested is not None else int(self.max_tokens))
        except Exception:
            return max(1, int(self.max_tokens or 1))

    def signature_config(self) -> Dict[str, Any]:
        requested = self._requested_max_tokens()
        model_name = str(self.default_model or "").strip()
        payload: Dict[str, Any] = {
            "provider": self.provider,
            "model": model_name,
            "temperature": self.temperature,
            "max_tokens_requested": requested,
        }
        if self.provider == "deepseek":
            cap = self._deepseek_token_cap(model_name)
            cap_env = self._deepseek_token_cap_env(model_name)
            payload.update(
                {
                    "max_tokens": self._clamp_deepseek_max_tokens(requested, model_name),
                    "max_tokens_cap": cap,
                    "max_tokens_cap_env": cap_env,
                    "reasoner_fallback_enabled": bool(self.allow_deepseek_reasoner_fallback),
                }
            )
            if self.allow_deepseek_reasoner_fallback:
                payload.update(
                    {
                        "reasoner_fallback_model": "deepseek-reasoner",
                        "reasoner_fallback_max_tokens": self._clamp_deepseek_max_tokens(
                            32000, "deepseek-reasoner"
                        ),
                    }
                )
        elif self.provider == "anthropic":
            cap = self._anthropic_token_cap(model_name)
            payload.update(
                {
                    "max_tokens": self._clamp_anthropic_max_tokens(requested, model_name),
                    "max_tokens_cap": cap,
                    "max_tokens_cap_env": "",
                }
            )
        else:
            payload["max_tokens"] = requested
            payload["max_tokens_cap"] = None
            payload["max_tokens_cap_env"] = ""
        return payload

    def _deepseek_token_cap(self, model: Optional[str] = None) -> int:
        model_name = str(model or self.default_model or "").strip()
        default_cap = 8192 if self._is_deepseek_flash_model(model_name) else 64000
        env_name = self._deepseek_token_cap_env(model_name)
        raw_cap = os.getenv(env_name) or os.getenv("DEEPSEEK_MAX_TOKENS_CAP")
        if raw_cap:
            try:
                cap = int(raw_cap)
                if cap > 0:
                    return cap
            except Exception:
                logger.warning("Ignoring invalid %s=%r", env_name, raw_cap)
        return default_cap

    def _is_deepseek_flash_model(self, model: Optional[str] = None) -> bool:
        model_name = str(model or "").strip().lower()
        return "flash" in model_name or model_name == "deepseek-chat"

    def _deepseek_token_cap_env(self, model: Optional[str] = None) -> str:
        model_name = str(model or "").strip().lower()
        if model_name == "deepseek-reasoner":
            return "DEEPSEEK_REASONER_MAX_TOKENS_CAP"
        if self._is_deepseek_flash_model(model_name):
            return "DEEPSEEK_FLASH_MAX_TOKENS_CAP"
        return "DEEPSEEK_PRO_MAX_TOKENS_CAP"

    def _clamp_deepseek_max_tokens(self, requested: Optional[int], model: Optional[str] = None) -> int:
        budget = self._requested_max_tokens(requested)
        return max(1, min(budget, self._deepseek_token_cap(model)))

    def _anthropic_token_cap(self, model: Optional[str] = None) -> int:
        _ = model
        return 64000

    def _clamp_anthropic_max_tokens(self, requested: Optional[int], model: Optional[str] = None) -> int:
        budget = self._requested_max_tokens(requested)
        return max(1, min(budget, self._anthropic_token_cap(model)))

    def _extract_response_metadata(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"provider": self.provider,
            "upstream_provider": response_data.get('provider'),
            "requested_provider": response_data.get('_requested_provider'),
            "generation_id": response_data.get('id')}
        model_name = response_data.get("model") or self.default_model
        if model_name:
            meta["model"] = model_name
        if response_data.get("stop_reason") is not None:
            meta["stop_reason"] = response_data.get("stop_reason")
        if response_data.get("stop_sequence") is not None:
            meta["stop_sequence"] = response_data.get("stop_sequence")

        finish_reason = None
        if isinstance(response_data.get("choices"), list) and response_data["choices"]:
            finish_reason = response_data["choices"][0].get("finish_reason")
        if finish_reason is not None:
            meta["finish_reason"] = finish_reason
        if response_data.get("_degenerate_retry_sampling") is not None:
            meta["degenerate_retry_sampling"] = response_data.get("_degenerate_retry_sampling")

        usage = response_data.get("usage")
        if isinstance(usage, dict):
            meta["usage"] = usage
            normalized: Dict[str, Any] = {}
            if "input_tokens" in usage or "output_tokens" in usage:
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
                cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
                normalized = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                    "effective_total_input_tokens": input_tokens + cache_creation + cache_read,
                }
            elif "prompt_tokens" in usage or "completion_tokens" in usage:
                normalized = {
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                }
            if normalized:
                meta["token_usage"] = normalized
        return meta

    def _log_response_metadata(self, response_data: Dict[str, Any], *, stage: str) -> Dict[str, Any]:
        meta = self._extract_response_metadata(response_data)
        token_usage = meta.get("token_usage")
        if token_usage:
            logger.info(
                "%s response meta: provider=%s model=%s stop_reason=%s finish_reason=%s token_usage=%s",
                stage,
                meta.get("provider"),
                meta.get("model"),
                meta.get("stop_reason"),
                meta.get("finish_reason"),
                token_usage,
            )
        else:
            logger.info(
                "%s response meta: provider=%s model=%s stop_reason=%s finish_reason=%s",
                stage,
                meta.get("provider"),
                meta.get("model"),
                meta.get("stop_reason"),
                meta.get("finish_reason"),
            )
        return meta

    def _append_usage_log(self, *, stage: str, meta: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
        if not meta:
            return
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "stop_reason": meta.get("stop_reason"),
            "finish_reason": meta.get("finish_reason"),
            "usage": meta.get("usage"),
            "token_usage": meta.get("token_usage"),
            "upstream_provider": meta.get('upstream_provider'),
            "requested_provider": meta.get('requested_provider'),
            "generation_id": meta.get('generation_id'),
        }
        if meta.get("degenerate_retry_sampling") is not None:
            entry["degenerate_retry_sampling"] = meta["degenerate_retry_sampling"]
        if extra:
            entry.update(extra)
        usage_log_path = os.path.join(self.cache_dir, "step2_usage_log.jsonl")
        try:
            with open(usage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to append step2 usage log %s: %s", usage_log_path, exc)

    def _ensure_api_key(self) -> None:
        if self.api_key:
            return
        try:
            load_api_keys(DEFAULT_API_CONFIG, set_env=True)
        except Exception:
            pass
        if self.provider == "openai":
            self.api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            self.headers["Authorization"] = f"Bearer {self.api_key}"
            if not self.api_key:
                raise RuntimeError("Missing OPENAI_API_KEY (set env or pargen/config/api_config.json).")
        elif self.provider == "deepseek":
            self.api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
            self.headers["Authorization"] = f"Bearer {self.api_key}"
            if not self.api_key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY (set env or pargen/config/api_config.json).")
        else:
            self.api_key = (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
            self.headers["x-api-key"] = self.api_key
            if not self.api_key:
                raise RuntimeError(
                    "Missing CLAUDE_API_KEY/ANTHROPIC_API_KEY (set env or pargen/config/api_config.json)."
                )

    def _filter_protocol_tree_content(self, data: dict) -> dict:
        if not isinstance(data, dict):
            return data

        if 'protocol_tree' in data:
            return data['protocol_tree']

        filtered_data = {
            'root_node_id': data.get('root_node_id', 'root'),
            'nodes': data.get('nodes', []) or [],
            'edges': data.get('edges', []) or []
        }

        logger.info(f"Filtered protocol tree: {len(filtered_data['nodes'])} nodes, {len(filtered_data['edges'])} edges")
        return filtered_data

    def _save_to_cache(self, filename: str, data: Any) -> None:
        cache_path = os.path.join(self.cache_dir, filename)

        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            output_data = None

            if isinstance(data, dict) and 'raw_response' in data:
                try:
                    raw_response = data['raw_response']
                    if isinstance(raw_response, str):
                        clean_response = self._clean_raw_response(raw_response)
                        clean_response = self._validate_and_fix_json_structure(clean_response)

                        if clean_response:
                            parsed_response = json.loads(clean_response)

                            if not isinstance(parsed_response, dict):
                                raise ValueError("raw_response JSON payload is not an object")

                            tree_like_response = (
                                "protocol_tree" in parsed_response
                                or "nodes" in parsed_response
                                or "root_node_id" in parsed_response
                            )
                            if tree_like_response:
                                filtered_response = self._filter_protocol_tree_content(parsed_response)
                                output_data = {
                                    "protocol_tree": filtered_response,
                                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                                    "section_info": data.get("section_info", None),
                                    "provider": data.get("provider"),
                                    "model": data.get("model"),
                                    "stop_reason": data.get("stop_reason"),
                                    "finish_reason": data.get("finish_reason"),
                                    "usage": data.get("usage"),
                                    "token_usage": data.get("token_usage"),
                                    "response_kind": data.get("response_kind"),
                                }
                            else:
                                output_data = {
                                    "raw_response": data["raw_response"],
                                    "parsed_response": parsed_response,
                                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                                    "section_info": data.get("section_info", None),
                                    "provider": data.get("provider"),
                                    "model": data.get("model"),
                                    "stop_reason": data.get("stop_reason"),
                                    "finish_reason": data.get("finish_reason"),
                                    "usage": data.get("usage"),
                                    "token_usage": data.get("token_usage"),
                                    "response_kind": data.get("response_kind"),
                                }
                            logger.info(f"Successfully parsed raw_response for {filename}")
                        else:
                            raise ValueError("Empty response after cleaning")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse raw_response in {filename}: {e}")
                    output_data = {
                        "raw_response": data['raw_response'],
                        "timestamp": data.get("timestamp", datetime.now().isoformat()),
                        "section_info": data.get("section_info", None),
                        "provider": data.get("provider"),
                        "model": data.get("model"),
                        "stop_reason": data.get("stop_reason"),
                        "finish_reason": data.get("finish_reason"),
                        "usage": data.get("usage"),
                        "token_usage": data.get("token_usage"),
                        "response_kind": data.get("response_kind"),
                    }

            elif isinstance(data, dict) and 'protocol_tree' in data:
                output_data = data

            elif isinstance(data, dict) and ('nodes' in data or 'root_node_id' in data):
                filtered_response = self._filter_protocol_tree_content(data)
                output_data = {
                    "protocol_tree": filtered_response,
                    "timestamp": datetime.now().isoformat(),
                    "section_info": None
                }

            else:
                output_data = data

            if isinstance(output_data, dict) and isinstance(data, dict):
                for meta_key in (
                    "prompt",
                    "context_hash",
                    "tree_hash",
                    "mode",
                    "attached_changes",
                    "json_retry_count",
                ):
                    if meta_key in data and meta_key not in output_data:
                        output_data[meta_key] = data[meta_key]

            if isinstance(output_data, dict) and self.input_signature:
                output_data["input_signature"] = self.input_signature
                output_data["cache_schema"] = _STAGE2_INPUT_SIGNATURE_SCHEMA

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            logger.info(f"Saved to cache: {cache_path}")

        except Exception as e:
            logger.error(f"Failed to save cache {filename}: {e}")
            raise

    def _merge_similar_nodes(self, tree: Dict) -> Dict:
        if not tree or 'nodes' not in tree:
            return tree

        nodes = tree['nodes']
        node_lookup = {
            node.get('node_id'): node
            for node in nodes
            if node.get('node_id') is not None
        }
        merged_nodes: List[Dict[str, Any]] = []
        seen_signatures: Dict[Tuple[str, ...], int] = {}
        merge_map: Dict[Any, Any] = {}  # old_id -> new_id

        for node in nodes:
            node_id = node.get('node_id')
            signature = self._node_context_signature(node, node_lookup)

            if signature in seen_signatures:
                existing_idx = seen_signatures[signature]
                existing_node = merged_nodes[existing_idx]
                existing_id = existing_node.get('node_id')
                
                # Map current node to existing node
                if node_id is not None:
                    merge_map[node_id] = existing_id
                    merge_map[str(node_id)] = existing_id

                existing_constraints = set(existing_node.get('constraints', []))
                new_constraints = set(node.get('constraints', []))
                existing_node['constraints'] = list(existing_constraints | new_constraints)

                if len(node.get('description', '') or '') > len(existing_node.get('description', '') or ''):
                    existing_node['description'] = node.get('description', '')

                logger.debug(f"Merged duplicate node {node_id} into {existing_id}")
            else:
                seen_signatures[signature] = len(merged_nodes)
                merged_nodes.append(node)
                if node_id is not None:
                    node_lookup[node_id] = node

        # Update relationships in merged nodes
        for node in merged_nodes:
            # Update parent_id
            pid = node.get('parent_id')
            if pid in merge_map:
                node['parent_id'] = merge_map[pid]
            
            # Update children_ids
            children = node.get('children_ids', [])
            new_children = []
            for child in children:
                if child in merge_map:
                    new_children.append(merge_map[child])
                else:
                    new_children.append(child)
            # Deduplicate children while preserving numeric node IDs when possible.
            # Using stringified IDs here can break downstream validators/interpreters that index nodes by int node_id.
            def _coerce_child_id(raw: Any) -> Any:
                if raw is None:
                    return None
                if isinstance(raw, bool):
                    return raw
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    try:
                        return int(raw)
                    except Exception:
                        return raw
                if isinstance(raw, str):
                    s = raw.strip()
                    if s and (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
                        try:
                            return int(s)
                        except Exception:
                            return raw
                    return raw
                return raw

            deduped: List[Any] = []
            seen: set = set()
            for child in new_children:
                coerced = _coerce_child_id(child)
                if coerced is None:
                    continue
                try:
                    key = coerced
                    if key in seen:
                        continue
                    seen.add(key)
                except TypeError:
                    key = str(coerced)
                    if key in seen:
                        continue
                    seen.add(key)
                deduped.append(coerced)

            def _sort_key(value: Any) -> Tuple[int, Any]:
                if isinstance(value, int) and not isinstance(value, bool):
                    return (0, value)
                return (1, str(value))

            node['children_ids'] = sorted(deduped, key=_sort_key)

        # Update edges
        new_edges = []
        for edge in tree.get('edges', []):
            src = edge.get('src')
            dst = edge.get('dst')
            
            # Update endpoints if they were merged
            # Check both raw ID and stringified ID
            if src in merge_map:
                src = merge_map[src]
            elif str(src) in merge_map:
                src = merge_map[str(src)]
                
            if dst in merge_map:
                dst = merge_map[dst]
            elif str(dst) in merge_map:
                dst = merge_map[str(dst)]
                
            edge['src'] = src
            edge['dst'] = dst
            new_edges.append(edge)

        tree['nodes'] = merged_nodes
        tree['edges'] = new_edges
        logger.info(f"Node merging: {len(nodes)} -> {len(merged_nodes)} nodes")
        return tree

    def _node_context_signature(self, node: Dict[str, Any], lookup: Dict[Any, Dict[str, Any]]) -> Tuple[str, ...]:
        name = str(node.get('name', '') or '')
        node_type = str(node.get('node_type', '') or '')
        message_type = str(node.get('message_type', '') or '')

        parent_id = node.get('parent_id')
        parent = lookup.get(parent_id) if parent_id is not None else None
        parent_name = str(parent.get('name', '') or '') if parent else ''
        parent_type = str(parent.get('node_type', '') or '') if parent else ''
        parent_message_type = str(parent.get('message_type', '') or '') if parent else ''

        source = str(node.get('source', '') or '')

        return (
            name,
            node_type,
            message_type,
            parent_name,
            parent_type,
            parent_message_type,
            source,
        )

    def _strip_js_single_line_comments_outside_strings(self, text: str) -> str:
        """Remove // comments only when they appear outside of quoted strings."""
        if not text:
            return text
        out: List[str] = []
        i = 0
        n = len(text)
        in_str = False
        escaped = False
        while i < n:
            ch = text[i]
            if in_str:
                out.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                i += 2
                while i < n and text[i] != "\n":
                    i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _clean_raw_response(self, raw_response: str) -> str:
        if not raw_response:
            return ""

        clean_text = raw_response.strip()

        import re
        clean_text = re.sub(r'极抱歉，.*?让我重新提供正确的JSON。?\s*', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'抱歉，.*?正确.*?JSON.*?\s*', '', clean_text, flags=re.DOTALL)

        if clean_text and not clean_text.startswith("{") and not clean_text.startswith("["):
            fenced_candidates = _extract_fenced_json_blocks(clean_text)
            for candidate in reversed(fenced_candidates):
                try:
                    json.loads(candidate)
                    clean_text = candidate
                    break
                except Exception:
                    continue
            else:
                extracted_json = _extract_balanced_json_object(clean_text)
                if extracted_json:
                    clean_text = extracted_json

        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]

        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        clean_text = clean_text.strip()
        if clean_text.startswith('"') and clean_text.endswith('"'):
            clean_text = clean_text[1:-1]
        elif clean_text.startswith("'") and clean_text.endswith("'"):
            clean_text = clean_text[1:-1]

        clean_text = re.sub(r'/\*.*?\*/', '', clean_text, flags=re.DOTALL)

        clean_text = self._strip_js_single_line_comments_outside_strings(clean_text)

        clean_text = self._validate_and_fix_json_structure(clean_text)

        clean_text = clean_text.strip()
        try:
            clean_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", clean_text)
        except Exception:
            pass

        return clean_text

    def _structured_response_format(self) -> Optional[Dict[str, Any]]:

        if self.provider == "deepseek":
            return {"type": "json_object"}

        if self.provider != "openai":
            return None

        # Keep this schema closed and bounded. A permissive initial schema lets
        # guided decoders satisfy required keys once, then nest intended nodes
        # in arbitrary property names until the output budget is exhausted.
        id_schema: Dict[str, Any] = {"type": "integer"}
        nullable_id_schema: Dict[str, Any] = {"type": ["integer", "null"]}

        node_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "node_id": id_schema,
                "name": {"type": "string", "maxLength": 128},
                "node_type": {"type": "string", "maxLength": 32},
                "message_type": {"type": "string", "maxLength": 32},
                "bit_start": {"type": ["string", "integer"], "maxLength": 256},
                "size_bits": {"type": ["string", "integer"], "maxLength": 256},
                "data_type": {"type": "string", "maxLength": 32},
                "byte_order": {"type": "string", "maxLength": 16},
                "parent_id": nullable_id_schema,
                "children_ids": {"type": "array", "items": id_schema, "maxItems": 16},
                "constraints": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 256},
                    "maxItems": 8,
                },
                "dependencies": {"type": "array", "maxItems": 0},
                "source": {"type": "string", "maxLength": 128},
                "source_section_ids": {"type": "array", "items": id_schema, "maxItems": 8},
            },
            "required": [
                "node_id", "name", "node_type", "message_type", "bit_start", "size_bits",
                "data_type", "byte_order", "parent_id", "children_ids", "constraints",
                "dependencies", "source", "source_section_ids",
            ],
            "additionalProperties": False,
        }

        edge_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "src": nullable_id_schema,
                "dst": nullable_id_schema,
                "rel": {"type": "string", "maxLength": 32},
                "formula": {"type": ["string", "null"], "maxLength": 512},
                "message_type": {"type": "string", "maxLength": 32},
            },
            "required": ["src", "dst", "rel", "formula", "message_type"],
            "additionalProperties": False,
        }

        protocol_tree_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "root_node_id": nullable_id_schema,
                "nodes": {
                    "type": "array", "items": node_schema, "minItems": 1,
                    "maxItems": _initial_tree_limits()[0],
                },
                "edges": {"type": "array", "items": edge_schema, "maxItems": _initial_tree_limits()[1]},
            },
            "required": ["root_node_id", "nodes", "edges"],
            "additionalProperties": False,
        }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "protocol_tree_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "protocol_tree": protocol_tree_schema,
                    },
                    "required": ["protocol_tree"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def _traffic_locate_json_schema(self) -> Dict[str, Any]:
        section_indices = {"type": "array", "items": {"type": "integer"}}
        evidence_schema = {
            "type": "object",
            "properties": {
                "section_index": {"type": ["integer", "null"]},
                "packet_offsets": {
                    "type": "array",
                    "items": {"type": ["integer", "string"]},
                },
                "raw_bytes": {"type": "string"},
                "spec_evidence": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["section_index", "packet_offsets", "raw_bytes", "spec_evidence", "reason"],
            "additionalProperties": True,
        }
        return {
            "type": "object",
            "properties": {
                "traffic_locate": {
                    "type": "object",
                    "properties": {
                        "active_core_sections": section_indices,
                        "header_sections": section_indices,
                        "selector_sections": section_indices,
                        "active_variant_sections": section_indices,
                        "supporting_tlv_or_ie_sections": section_indices,
                        "active_supporting_sections": section_indices,
                        "supporting_sections": section_indices,
                        "coverage_only_sections": section_indices,
                        "deferred_variant_sections": section_indices,
                        "relevant_sections": section_indices,
                        "irrelevant_sections": section_indices,
                        "evidence": {"type": "array", "items": evidence_schema},
                        "confidence": {"type": "string"},
                        "notes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "header_sections",
                        "selector_sections",
                        "active_variant_sections",
                        "supporting_tlv_or_ie_sections",
                        "relevant_sections",
                        "irrelevant_sections",
                        "evidence",
                    ],
                    "additionalProperties": True,
                }
            },
            "required": ["traffic_locate"],
            "additionalProperties": True,
        }

    def _traffic_locate_response_format(self) -> Optional[Dict[str, Any]]:
        if self.provider == "deepseek":
            return {"type": "json_object"}
        if self.provider != "openai":
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "traffic_locate_response",
                "schema": self._traffic_locate_json_schema(),
            },
        }

    def _normalize_traffic_locate_report(
        self,
        report: Any,
        *,
        section_count: int,
    ) -> Dict[str, Any]:
        payload = _traffic_locate_payload(report if isinstance(report, dict) else {})
        normalized: Dict[str, Any] = {}
        for key in (
            "header_sections",
            "selector_sections",
            "active_variant_sections",
            "supporting_tlv_or_ie_sections",
            "supporting_sections",
            "relevant_sections",
            "irrelevant_sections",
        ):
            normalized[key] = _traffic_locate_indices(payload.get(key), section_count=section_count)
        for key in (
            "active_core_sections",
            "active_supporting_sections",
            "coverage_only_sections",
            "deferred_variant_sections",
            "deferred_sections",
        ):
            normalized[key] = _traffic_locate_indices(payload.get(key), section_count=section_count)
        evidence = payload.get("evidence")
        normalized["evidence"] = evidence if isinstance(evidence, list) else []
        if payload.get("confidence") is not None:
            normalized["confidence"] = str(payload.get("confidence"))
        notes = payload.get("notes")
        if isinstance(notes, list):
            normalized["notes"] = [str(note) for note in notes]
        active_union = _collect_traffic_locate_indices(
            normalized,
            _TRAFFIC_LOCATE_ACTIVE_KEYS,
            section_count=section_count,
        )
        if active_union:
            normalized["active_core_sections"] = active_union
        return {"traffic_locate": normalized}

    def _extract_traffic_locate_raw_response(self, result: Dict[str, Any]) -> str:
        if isinstance(result.get("content"), list):
            text_parts: List[str] = []
            for block in result.get("content") or []:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
        if isinstance(result.get("choices"), list) and result["choices"]:
            message = result["choices"][0].get("message") or {}
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        raise ValueError(f"Unexpected traffic locate API response format: {result.keys()}")

    def locate_traffic_sections(
        self,
        sections: Sequence[Dict[str, Any]],
        *,
        protocol_hint: Optional[str] = None,
        traffic_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Use an LLM to locate spec sections relevant to observed traffic."""

        empty_report = {
            "traffic_locate": {
                "active_core_sections": [],
                "header_sections": [],
                "selector_sections": [],
                "active_variant_sections": [],
                "supporting_tlv_or_ie_sections": [],
                "active_supporting_sections": [],
                "supporting_sections": [],
                "coverage_only_sections": [],
                "deferred_variant_sections": [],
                "relevant_sections": [],
                "irrelevant_sections": [],
                "evidence": [],
            }
        }
        if not traffic_file:
            return empty_report

        cached = self._load_from_cache("traffic_locate_ai_response.json")
        if isinstance(cached, dict):
            cached_payload = cached.get("parsed_response") or cached.get("traffic_locate") or cached
            cached_report = self._normalize_traffic_locate_report(cached_payload, section_count=len(sections))
            if _traffic_locate_payload(cached_report):
                cached_scope_issue = _traffic_locate_scope_rejection_reason(
                    cached_report,
                    section_count=len(sections),
                    sections=sections,
                )
                if cached_scope_issue:
                    logger.warning(
                        "Ignoring cached LLM traffic locate report because active/core scope is invalid: %s",
                        cached_scope_issue,
                    )
                else:
                    logger.info("Using cached LLM traffic locate report from traffic_locate_ai_response.json")
                    return cached_report

        traffic_context = _format_observed_traffic_context(traffic_file, max_bytes=128)
        section_briefs = _traffic_locate_section_briefs(sections)
        protocol_hint_text = _normalize_protocol_hint(protocol_hint)
        prompt = f"""
## TASK

You are locating which specification sections are relevant to one or more observed raw traffic packets before building an initial protocol tree.

This is an LLM-driven locate step. You must decide relevance from the raw bytes, protocol documentation, wire-format layout, selector definitions, length fields, TLV/repeated-item rules, and message/body descriptions.

Do not assume that a section is relevant just because a number appears in both the packet bytes and text. Numeric overlap alone is insufficient evidence. Explain every relevant decision with packet-offset evidence and spec evidence.

## TARGET PROTOCOL

{protocol_hint_text or "Unknown"}

## OBSERVED TRAFFIC

{traffic_context}

## SPEC SECTIONS

{json.dumps(section_briefs, indent=2, ensure_ascii=False)}

## OUTPUT JSON

Return ONLY this JSON object:

{{
  "traffic_locate": {{
    "active_core_sections": [0, 1],
    "header_sections": [0],
    "selector_sections": [0],
    "active_variant_sections": [1],
    "supporting_tlv_or_ie_sections": [2],
    "active_supporting_sections": [],
    "coverage_only_sections": [3],
    "deferred_variant_sections": [3],
    "supporting_sections": [],
    "relevant_sections": [0, 1, 2, 3],
    "irrelevant_sections": [],
    "evidence": [
      {{
        "section_index": 0,
        "packet_offsets": [0, 1],
        "raw_bytes": "hex bytes from the packet",
        "spec_evidence": "short quote or paraphrase from the section",
        "reason": "why this section participates in parsing this packet"
      }}
    ],
    "confidence": "high|medium|low",
    "notes": []
  }}
}}

`active_core_sections` is the only section set that the ordinary one-packet StageB
build will consume. Include only sections required to parse the observed packet
now: shared fixed header, selector fields, selector tables that map the observed
packet value to a named variant, the active variant/body selected by the packet,
and directly required TLV/IE/option structure.

If packet evidence maps a selector value to a named variant section, both the
selector/mapping section and the selected variant section MUST be in
`active_core_sections` or `active_variant_sections`. Never put that selected
variant or its observed selector table only in `coverage_only_sections`.

Every section cited by an evidence item with `packet_offsets` or `raw_bytes`
must be in `active_core_sections`, `selector_sections`,
`active_variant_sections`, or `supporting_tlv_or_ie_sections`. Do not cite a
section with packet bytes and leave it only in `relevant_sections`;
`relevant_sections` is bookkeeping and is not consumed by the ordinary StageB
build.

Put unobserved sibling variants, optional message bodies, and full-protocol
coverage material in `coverage_only_sections` or
`deferred_variant_sections`. They are useful for a later coverage/scaffold run,
but they must not enter the active/core StageB build for this packet.
"""
        max_tokens = _env_positive_int("STEP2_TRAFFIC_LOCATE_MAX_TOKENS", self.max_tokens)
        if self.provider == "openai":
            token_key = "max_completion_tokens" if self.default_model.startswith("gpt-5") else "max_tokens"
            payload = {
                "model": self.default_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You locate traffic-relevant protocol specification sections. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                token_key: max_tokens,
                "temperature": self.temperature,
            }
            structured_format = self._traffic_locate_response_format()
            if structured_format:
                payload["response_format"] = structured_format
        elif self.provider == "deepseek":
            payload = {
                "model": self.default_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You locate traffic-relevant protocol specification sections. "
                            "Return valid JSON only; no markdown and no comments."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self._clamp_deepseek_max_tokens(max_tokens, self.default_model),
                "temperature": self.temperature,
                "top_p": 0.95,
                "stream": False,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
            structured_format = self._traffic_locate_response_format()
            if structured_format:
                payload["response_format"] = structured_format
        else:
            payload = {
                "model": self.default_model,
                "system": "You locate traffic-relevant protocol specification sections. Return valid JSON only.",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self._clamp_anthropic_max_tokens(max_tokens, self.default_model),
                "temperature": self.temperature,
            }

        max_locate_json_retries = _env_positive_int("STEP2_TRAFFIC_LOCATE_JSON_RETRIES", 3)
        parsed: Optional[Dict[str, Any]] = None
        cache_data: Dict[str, Any] = {}
        for attempt in range(max_locate_json_retries):
            result = self._call_api_with_retry(payload)
            response_meta = self._extract_response_metadata(result)
            self._append_usage_log(
                stage="traffic_locate",
                meta=response_meta,
                extra={"json_retry_count": attempt},
            )
            raw_response = self._extract_traffic_locate_raw_response(result)
            cache_data = {
                "prompt": prompt,
                "raw_response": raw_response,
                "timestamp": datetime.now().isoformat(),
                "provider": response_meta.get("provider"),
                "model": response_meta.get("model"),
                "stop_reason": response_meta.get("stop_reason"),
                "finish_reason": response_meta.get("finish_reason"),
                "usage": response_meta.get("usage"),
                "token_usage": response_meta.get("token_usage"),
                "json_retry_count": attempt,
            }
            try:
                parsed_candidate = json.loads(self._clean_raw_response(raw_response))
                if not isinstance(parsed_candidate, dict):
                    raise ValueError(f"traffic locate response is not a JSON object: {type(parsed_candidate).__name__}")
                normalized_candidate = self._normalize_traffic_locate_report(
                    parsed_candidate,
                    section_count=len(sections),
                )
                scope_issue = _traffic_locate_scope_rejection_reason(
                    normalized_candidate,
                    section_count=len(sections),
                    sections=sections,
                )
                if scope_issue:
                    raise ValueError(f"traffic locate active/core scope is invalid: {scope_issue}")
                parsed = normalized_candidate
                cache_data["parsed_response"] = parsed
                self._save_to_cache("traffic_locate_ai_response.json", cache_data)
                break
            except Exception as exc:
                is_scope_rejection = (
                    "traffic locate active/core scope is invalid" in str(exc)
                    or "traffic locate active/core scope is too broad" in str(exc)
                )
                cache_data["validation_error" if is_scope_rejection else "parse_error"] = str(exc)
                rejected_name = (
                    f"traffic_locate_ai_response.rejected_scope_attempt_{attempt + 1}.json"
                    if is_scope_rejection
                    else f"traffic_locate_ai_response.rejected_json_attempt_{attempt + 1}.json"
                )
                self._save_to_cache(rejected_name, cache_data)
                if attempt + 1 >= max_locate_json_retries:
                    raise
                if is_scope_rejection and payload.get("messages"):
                    payload["messages"][-1]["content"] = (
                        prompt
                        + "\n\n# PREVIOUS RESPONSE REJECTED\n"
                        + "The previous traffic locate response was rejected because "
                        + str(exc)
                        + ". Retry by putting packet-observed header, selector fields, observed selector "
                          "mapping/table sections, active variant/body sections, and direct TLV/option "
                          "sections in active_core_sections. Every section cited with packet_offsets or "
                          "raw_bytes must be active, not only relevant_sections. Put only unobserved sibling variants, optional "
                          "bodies, and broad coverage material in coverage_only_sections or "
                          "deferred_variant_sections."
                    )
                if is_scope_rejection:
                    logger.warning(
                        "Traffic locate scope validation failed (attempt %d/%d): %s; retrying",
                        attempt + 1,
                        max_locate_json_retries,
                        exc,
                    )
                else:
                    logger.warning(
                        "Traffic locate JSON parse failed (attempt %d/%d): %s; retrying",
                        attempt + 1,
                        max_locate_json_retries,
                        exc,
                    )
                continue
        if parsed is None:
            raise ValueError("Traffic locate did not return a parseable JSON object")
        report = self._normalize_traffic_locate_report(parsed, section_count=len(sections))
        logger.info(
            "LLM traffic locate selected sections: header=%s selector=%s active=%s support=%s relevant=%s",
            _traffic_locate_payload(report).get("header_sections"),
            _traffic_locate_payload(report).get("selector_sections"),
            _traffic_locate_payload(report).get("active_variant_sections"),
            _traffic_locate_payload(report).get("supporting_tlv_or_ie_sections"),
            _traffic_locate_payload(report).get("relevant_sections"),
        )
        return report

    def _constraint_localize_response_format(self) -> Optional[Dict[str, Any]]:
        if self.provider == "deepseek":
            return {"type": "json_object"}
        if self.provider != "openai":
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "constraint_localize_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "constraint_obligations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string"},
                                    "expr": {"type": "string"},
                                    "anchor_node_id": {"type": ["integer", "string", "null"]},
                                    "anchor_node_name": {"type": ["string", "null"]},
                                    "field_ids": {"type": "array", "items": {"type": ["integer", "string"]}},
                                    "field_names": {"type": "array", "items": {"type": "string"}},
                                    "scope_terms": {"type": "array", "items": {"type": "string"}},
                                    "source": {"type": "string"},
                                },
                                "required": ["kind", "expr"],
                                "additionalProperties": True,
                            },
                        }
                    },
                    "required": ["constraint_obligations"],
                    "additionalProperties": True,
                },
            },
        }

    def _constraint_tree_digest(self, tree: Dict[str, Any], *, max_nodes: int = 220) -> Dict[str, Any]:
        nodes = tree.get("nodes") if isinstance(tree, dict) else []
        edges = tree.get("edges") if isinstance(tree, dict) else []
        node_digest: List[Dict[str, Any]] = []
        for node in nodes[:max_nodes] if isinstance(nodes, list) else []:
            if not isinstance(node, dict):
                continue
            node_digest.append(
                {
                    "node_id": node.get("node_id"),
                    "name": node.get("name"),
                    "node_type": node.get("node_type"),
                    "message_type": node.get("message_type"),
                    "parent_id": node.get("parent_id"),
                    "children_ids": node.get("children_ids") or [],
                    "bit_start": node.get("bit_start"),
                    "size_bits": node.get("size_bits"),
                    "constraints": node.get("constraints") or [],
                }
            )
        edge_digest: List[Dict[str, Any]] = []
        for edge in edges[: max_nodes * 2] if isinstance(edges, list) else []:
            if not isinstance(edge, dict):
                continue
            rel = str(edge.get("rel") or "")
            if rel in {"condition_on", "length_of", "repeat_count"}:
                edge_digest.append(
                    {
                        "src": edge.get("src"),
                        "dst": edge.get("dst"),
                        "rel": rel,
                        "formula": edge.get("formula"),
                        "message_type": edge.get("message_type"),
                    }
                )
        return {"nodes": node_digest, "edges": edge_digest}

    def localize_and_attach_constraints(
        self,
        tree: Dict[str, Any],
        sections: Sequence[Dict[str, Any]],
        *,
        protocol_hint: Optional[str] = None,
        traffic_file: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Use an LLM to localize relation-only constraints, then attach as IR."""

        if not isinstance(tree, dict):
            return tree, []
        protocol_hint_text = _normalize_protocol_hint(protocol_hint)
        traffic_context = _format_observed_traffic_context(traffic_file, max_bytes=128) if traffic_file else ""

        def _full_context(source_tree: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "tree_digest": self._constraint_tree_digest(source_tree),
                "section_briefs": [_refine_section_digest(section) for section in sections[:28]],
            }

        def _compact_context(source_tree: Dict[str, Any]) -> Dict[str, Any]:
            return build_compact_constraint_context(
                source_tree,
                sections,
                max_nodes=_env_positive_int("STEP2_CONSTRAINT_COMPACT_MAX_NODES", 140),
                max_sections=_env_positive_int("STEP2_CONSTRAINT_COMPACT_MAX_SECTIONS", 28),
                max_section_chars=_env_positive_int("STEP2_CONSTRAINT_COMPACT_SECTION_CHARS", 1800),
            )

        def _run_localizer_once(
            source_tree: Dict[str, Any],
            *,
            context: Dict[str, Any],
            mode: str,
            cache_filename: str,
            rejected_prefix: str,
            max_tokens: int,
        ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
            context_hash = _stable_json_hash({"mode": mode, "context": context})
            cached = self._load_from_cache(cache_filename)
            if isinstance(cached, dict) and str(cached.get("context_hash") or cached.get("tree_hash") or "") == context_hash:
                parsed_cached = cached.get("parsed_response")
                obligations_cached = []
                if isinstance(parsed_cached, dict):
                    obligations_cached = parsed_cached.get("constraint_obligations") or []
                if isinstance(obligations_cached, list):
                    patched, changes = attach_constraint_obligations(source_tree, obligations_cached)
                    logger.info(
                        "Using cached %s constraint localization result: attached %d IR constraints",
                        mode,
                        len(changes),
                    )
                    return patched, changes

            prompt = f"""
## TASK

You are localizing protocol documentation constraints onto an existing Stage-B protocol tree.

Do not rewrite the tree. Do not output a patch. Identify only constraints that should be attached to existing nodes as relation-level `constraint_ir` annotations.

Focus on constraints that are often missed by structural refinement:
- cross-field formulas and unit relations
- selector-conditioned constraints
- variant-scoped constraints
- TLV/option/IE length and value constraints
- padding/end-marker rules

Use the current tree to choose the best anchor node. Prefer the controlling selector, length field, or scoped variant node. If an exact executable edge would be unsafe or requires structural changes, still output a relation-level obligation with clear field names and source evidence.

## TARGET PROTOCOL

{protocol_hint_text or "Unknown"}

## OBSERVED TRAFFIC ANCHOR

{traffic_context or "No observed traffic supplied."}

## CURRENT TREE DIGEST

{json.dumps(context.get("tree_digest") or {}, indent=2, ensure_ascii=False)}

## SPEC SECTION DIGESTS

{json.dumps(context.get("section_briefs") or [], indent=2, ensure_ascii=False)}

## OUTPUT JSON

Return ONLY this JSON object:

{{
  "constraint_obligations": [
    {{
      "kind": "length_relation|conditional_value|conditional_constraint|variant_map|post_end_zero_padding|unit_relation|mod_alignment|mutual_exclusion",
      "expr": "short mathematical or relation description with constants and units",
      "anchor_node_id": 12,
      "anchor_node_name": "Data_Offset",
      "field_ids": [12],
      "field_names": ["Data Offset", "TCP Header"],
      "scope_terms": ["TCP", "options"],
      "source": "brief source-backed evidence from the spec section"
    }}
  ]
}}
"""
            system_text = (
                "You localize protocol constraints onto existing tree nodes. "
                "Return valid JSON only; no markdown and no comments."
            )
            if self.provider == "openai":
                token_key = "max_completion_tokens" if self.default_model.startswith("gpt-5") else "max_tokens"
                payload = {
                    "model": self.default_model,
                    "messages": [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": prompt},
                    ],
                    token_key: max_tokens,
                    "temperature": self.temperature,
                }
                structured_format = self._constraint_localize_response_format()
                if structured_format:
                    payload["response_format"] = structured_format
            elif self.provider == "deepseek":
                payload = {
                    "model": self.default_model,
                    "messages": [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": self._clamp_deepseek_max_tokens(max_tokens, self.default_model),
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "stream": False,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
                structured_format = self._constraint_localize_response_format()
                if structured_format:
                    payload["response_format"] = structured_format
            else:
                payload = {
                    "model": self.default_model,
                    "system": system_text,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self._clamp_anthropic_max_tokens(max_tokens, self.default_model),
                    "temperature": self.temperature,
                }

            max_json_retries = _env_positive_int("STEP2_CONSTRAINT_LOCALIZE_JSON_RETRIES", 2)
            last_error: Optional[Exception] = None
            for attempt in range(max_json_retries):
                try:
                    result = self._call_api_with_retry(payload)
                    response_meta = self._extract_response_metadata(result)
                    self._append_usage_log(
                        stage=f"constraint_localize_{mode}",
                        meta=response_meta,
                        extra={"json_retry_count": attempt, "context_hash": context_hash},
                    )
                    raw_response = self._extract_traffic_locate_raw_response(result)
                    parsed = json.loads(self._clean_raw_response(raw_response))
                    if not isinstance(parsed, dict):
                        raise ValueError(f"constraint localize response is not a JSON object: {type(parsed).__name__}")
                    obligations = parsed.get("constraint_obligations") or []
                    if not isinstance(obligations, list):
                        raise ValueError("constraint_obligations must be a list")
                    patched, changes = attach_constraint_obligations(source_tree, obligations)
                    self._save_to_cache(
                        cache_filename,
                        {
                            "prompt": prompt,
                            "raw_response": raw_response,
                            "parsed_response": parsed,
                            "context_hash": context_hash,
                            "tree_hash": context_hash,
                            "mode": mode,
                            "attached_changes": changes,
                            "timestamp": datetime.now().isoformat(),
                            "provider": response_meta.get("provider"),
                            "model": response_meta.get("model"),
                            "stop_reason": response_meta.get("stop_reason"),
                            "finish_reason": response_meta.get("finish_reason"),
                            "usage": response_meta.get("usage"),
                            "token_usage": response_meta.get("token_usage"),
                            "json_retry_count": attempt,
                        },
                    )
                    logger.info("%s constraint localization attached %d IR constraints", mode, len(changes))
                    return patched, changes
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "%s constraint localization failed (attempt %d/%d): %s",
                        mode,
                        attempt + 1,
                        max_json_retries,
                        exc,
                    )
                    self._save_to_cache(
                        f"{rejected_prefix}.rejected_attempt_{attempt + 1}.json",
                        {
                            "prompt": prompt,
                            "context_hash": context_hash,
                            "mode": mode,
                            "parse_error": str(exc),
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
            if _env_flag("STEP2_CONSTRAINT_LOCALIZATION_STRICT", "0"):
                raise RuntimeError(f"{mode} constraint localization failed: {last_error}")
            return source_tree, []

        if _env_flag("STEP2_CONSTRAINT_LOCALIZATION_COMPACT", "0"):
            compact_max_tokens = _env_positive_int(
                "STEP2_CONSTRAINT_LOCALIZE_COMPACT_MAX_TOKENS",
                min(int(self.max_tokens or 8000), 7000),
            )
            compact_tree, compact_changes = _run_localizer_once(
                tree,
                context=_compact_context(tree),
                mode="compact",
                cache_filename="constraint_localize_compact_ai_response.json",
                rejected_prefix="constraint_localize_compact_ai_response",
                max_tokens=compact_max_tokens,
            )
            min_changes = _env_positive_int("STEP2_CONSTRAINT_LOCALIZATION_COMPACT_MIN_CHANGES", 4)
            fallback_enabled = _env_flag("STEP2_CONSTRAINT_LOCALIZATION_COMPACT_FALLBACK", "1")
            if len(compact_changes) >= min_changes or not fallback_enabled:
                return compact_tree, compact_changes
            logger.info(
                "Compact constraint localization attached %d < %d constraints; falling back to full context.",
                len(compact_changes),
                min_changes,
            )
            full_tree, full_changes = _run_localizer_once(
                compact_tree,
                context=_full_context(compact_tree),
                mode="full_after_compact",
                cache_filename="constraint_localize_full_after_compact_ai_response.json",
                rejected_prefix="constraint_localize_full_after_compact_ai_response",
                max_tokens=_env_positive_int("STEP2_CONSTRAINT_LOCALIZE_MAX_TOKENS", min(int(self.max_tokens or 8000), 12000)),
            )
            return full_tree, compact_changes + full_changes

        return _run_localizer_once(
            tree,
            context=_full_context(tree),
            mode="full",
            cache_filename="constraint_localize_ai_response.json",
            rejected_prefix="constraint_localize_ai_response",
            max_tokens=_env_positive_int("STEP2_CONSTRAINT_LOCALIZE_MAX_TOKENS", min(int(self.max_tokens or 8000), 12000)),
        )

    def _patch_json_schema(self) -> Dict[str, Any]:
        """Provider-independent JSON Schema for incremental tree patches."""
        id_schema: Dict[str, Any] = {"type": "integer"}
        nullable_id: Dict[str, Any] = {"type": ["integer", "null"]}
        patch_value: Dict[str, Any] = {
            "type": ["string", "integer", "number", "boolean", "array", "null"],
            "items": {},
        }
        patch_metadata_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "scope": {"type": "string"},
            },
            "required": ["intent", "scope"],
            "additionalProperties": False,
        }
        node_add = {
            "type": "object",
            "properties": {
                "node_id": id_schema,
                "name": {"type": "string"},
                "node_type": {"type": "string"},
                "message_type": {"type": "string"},
                "bit_start": {"type": ["string", "integer"]},
                "size_bits": {"type": ["string", "integer"]},
                "data_type": {"type": ["string", "null"]},
                "byte_order": {"type": "string"},
                "parent_id": nullable_id,
                "children_ids": {"type": "array", "items": id_schema},
                "constraints": {"type": "array", "items": {}},
                "dependencies": {"type": "array", "items": {}},
                "source": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
            },
            "required": [
                "node_id",
                "name",
                "node_type",
                "message_type",
                "bit_start",
                "size_bits",
                "data_type",
                "byte_order",
                "parent_id",
                "children_ids",
                "constraints",
                "dependencies",
                "source",
                "description",
            ],
            "additionalProperties": False,
        }
        node_update = {
            "type": "object",
            "properties": {
                "node_id": id_schema,
                "property": {"type": "string"},
                "value": patch_value,
            },
            "required": ["node_id", "property", "value"],
            "additionalProperties": False,
        }
        edge_add = {
            "type": "object",
            "properties": {
                "src": id_schema,
                "dst": id_schema,
                "rel": {"type": "string"},
                "formula": {"type": ["string", "null"]},
                "message_type": {"type": "string"},
            },
            "required": ["src", "dst", "rel", "formula", "message_type"],
            "additionalProperties": False,
        }
        edge_update = {
            "type": "object",
            "properties": {
                "src": id_schema,
                "dst": id_schema,
                "field": {"type": "string"},
                "value": patch_value,
            },
            "required": ["src", "dst", "field", "value"],
            "additionalProperties": False,
        }
        edge_remove = {
            "type": "object",
            "properties": {
                "src": id_schema,
                "dst": id_schema,
                "rel": {"type": ["string", "null"]},
                "formula": {"type": ["string", "null"]},
                "message_type": {"type": ["string", "null"]},
            },
            "required": ["src", "dst", "rel", "formula", "message_type"],
            "additionalProperties": False,
        }

        return {
            "type": "object",
            "properties": {
                "patch_metadata": patch_metadata_schema,
                "new_nodes": {"type": "array", "items": node_add},
                "node_updates": {"type": "array", "items": node_update},
                "edge_adds": {"type": "array", "items": edge_add},
                "new_edges": {"type": "array", "items": edge_add},
                "edge_updates": {"type": "array", "items": edge_update},
                "edge_removes": {"type": "array", "items": edge_remove},
                "nodes_to_remove": {"type": "array", "items": id_schema},
                "validation_notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "patch_metadata",
                "new_nodes",
                "node_updates",
                "edge_adds",
                "new_edges",
                "edge_updates",
                "edge_removes",
                "nodes_to_remove",
                "validation_notes",
            ],
            "additionalProperties": False,
        }

    def _patch_response_format(self) -> Optional[Dict[str, Any]]:
        """Structured response schema for incremental patches."""
        if self.provider == "deepseek":
            return {"type": "json_object"}
        if self.provider != "openai":
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "protocol_patch_response",
                "schema": self._patch_json_schema(),
            },
        }

    def _anthropic_patch_tool_schema(self) -> Dict[str, Any]:
        """Minimal strict tool schema; the patch itself is validated by our parser."""
        return {
            "type": "object",
            "properties": {
                "patch_json": {
                    "type": "string",
                    "description": (
                        "A raw JSON object string containing the incremental patch only. "
                        "Use {} when no safe improvement can be made."
                    ),
                }
            },
            "required": ["patch_json"],
            "additionalProperties": False,
        }

    def _attach_anthropic_patch_tool(self, payload: Dict[str, Any]) -> None:
        """Force Anthropic refine calls to return a schema-checked patch tool input."""
        if self.provider != "anthropic":
            return
        tool_name = "submit_protocol_patch"
        payload["tools"] = [
            {
                "name": tool_name,
                "description": (
                    "Submit the final incremental protocol-tree JSON patch for this Stage B refine batch. "
                "Call this tool exactly once. The patch_json argument must be a JSON object string, not "
                "markdown and not a full protocol_tree. Use {} only when no safe patch should be applied."
            ),
            "input_schema": self._anthropic_patch_tool_schema(),
            "strict": True,
            }
        ]
        payload["tool_choice"] = {"type": "tool", "name": tool_name}

    def _deepseek_patch_tool_use_enabled(self) -> bool:
        """Enable OpenAI-compatible patch tool calls by default on endpoints that support them."""
        raw = (os.getenv("STEP2_DEEPSEEK_PATCH_TOOL_USE", "auto") or "auto").strip().lower()
        if raw in {"1", "true", "yes", "on", "force"}:
            return True
        if raw in {"0", "false", "no", "off", "none"}:
            return False
        return self.provider == "deepseek" and "openrouter.ai" in (self.base_url or "").lower()

    def _attach_openai_compatible_patch_tool(self, payload: Dict[str, Any]) -> None:
        """Force OpenAI-compatible chat-completion APIs to return the refine patch via a function tool."""
        tool_name = "submit_protocol_patch"
        payload.pop("response_format", None)
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": (
                        "Submit the final incremental protocol-tree JSON patch for this Stage B refine batch. "
                        "Return an empty patch object only when no safe improvement can be made."
                    ),
                    "parameters": self._patch_json_schema(),
                },
            }
        ]
        payload["tool_choice"] = {"type": "function", "function": {"name": tool_name}}

    def _disable_openai_compatible_patch_tool(self, payload: Dict[str, Any]) -> None:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
        structured_format = self._patch_response_format()
        if structured_format:
            payload["response_format"] = structured_format

    def _extract_refine_raw_response(self, result: Dict[str, Any]) -> Tuple[str, str]:
        """Extract a refine response as JSON text, preferring provider tool-use input."""
        if self.provider == "anthropic" and isinstance(result.get("content"), list):
            text_parts: List[str] = []
            for block in result.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "submit_protocol_patch":
                    tool_input = block.get("input")
                    if isinstance(tool_input, dict):
                        patch_json = tool_input.get("patch_json")
                        if isinstance(patch_json, str):
                            return patch_json, "anthropic_tool_use"
                        return json.dumps(tool_input, ensure_ascii=False), "anthropic_tool_use"
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts), "anthropic_text"

        if isinstance(result.get("content"), list) and result["content"]:
            first_block = result["content"][0]
            if isinstance(first_block, dict) and isinstance(first_block.get("text"), str):
                return first_block["text"], "content_text"

        if isinstance(result.get("choices"), list) and result["choices"]:
            choice = result["choices"][0] if isinstance(result["choices"][0], dict) else {}
            message = choice.get("message") or {}
            if isinstance(message, dict):
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function_call = tool_call.get("function") or {}
                    if not isinstance(function_call, dict):
                        continue
                    if function_call.get("name") != "submit_protocol_patch":
                        continue
                    arguments = function_call.get("arguments")
                    if isinstance(arguments, str):
                        return arguments, "openai_tool_call"
                    if isinstance(arguments, dict):
                        return json.dumps(arguments, ensure_ascii=False), "openai_tool_call"
                content = message.get("content", "")
                if isinstance(content, str):
                    return content, "choice_text" if content.strip() else "choice_empty"
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                content = delta.get("content", "")
                if isinstance(content, str):
                    return content, "choice_delta" if content.strip() else "choice_empty"
            return "", "choice_empty"
        raise ValueError(f"Unexpected API response format: {result.keys()}")

    def _source_gap_adjudication_response_format(self) -> Optional[Dict[str, Any]]:
        if self.provider == "deepseek":
            return {"type": "json_object"}
        if self.provider != "openai":
            return None
        decision_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "gap": {"type": "string"},
                "status": {"type": "string", "enum": ["covered", "missing", "ambiguous"]},
                "covering_node_ids": {"type": "array", "items": {"type": ["integer", "string"]}},
                "reason": {"type": "string"},
            },
            "required": ["gap", "status", "covering_node_ids", "reason"],
            "additionalProperties": True,
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "source_gap_adjudication_response",
                "schema": {
                    "type": "object",
                    "properties": {"decisions": {"type": "array", "items": decision_schema}},
                    "required": ["decisions"],
                    "additionalProperties": True,
                },
            },
        }

    def _source_gap_adjudication_payload(
        self,
        tree: Dict[str, Any],
        focused_sections: Sequence[Dict[str, Any]],
        source_structure_gaps: Sequence[str],
    ) -> Dict[str, Any]:
        candidate_nodes = _source_gap_candidate_nodes(tree, source_structure_gaps)
        section_evidence = _source_gap_section_evidence(focused_sections)
        prompt = (
            "# TASK\n"
            "You are adjudicating high-recall source-backed gap detector candidates for a protocol tree.\n"
            "For each candidate gap, decide whether the current tree already covers it.\n\n"
            "# DECISION RULES\n"
            "- covered: the current tree has a semantically equivalent leaf, selector, variant route, or typed body. "
            "Naming aliases are allowed; for example, `Authentication_Present` may cover `Authentication (A) bit` "
            "when bit position and size agree.\n"
            "- missing: no compatible tree evidence exists, or only a generic shell/payload/container exists without "
            "the named field/route.\n"
            "- ambiguous: evidence is insufficient or the source/tree match is plausible but not certain. "
            "Ambiguous candidates are not hard failures.\n"
            "Judge semantics, layout/span, and source context. Do not require exact string equality.\n\n"
            "# CANDIDATE GAPS\n"
            f"{json.dumps(list(source_structure_gaps or []), ensure_ascii=False, indent=2)}\n\n"
            "# SOURCE EVIDENCE\n"
            f"{json.dumps(section_evidence, ensure_ascii=False, indent=2)}\n\n"
            "# CURRENT TREE CANDIDATE NODES BY GAP\n"
            f"{json.dumps(candidate_nodes, ensure_ascii=False, indent=2)}\n\n"
            "# OUTPUT\n"
            "Return JSON only:\n"
            "{\"decisions\":[{\"gap\":\"...\",\"status\":\"covered|missing|ambiguous\","
            "\"covering_node_ids\":[...],\"reason\":\"brief evidence\"}]}\n"
        )
        system = (
            "You are a precise protocol-tree coverage judge. Return strict JSON only. "
            "Do not propose patches; only adjudicate whether candidate gaps are actually missing."
        )
        max_tokens = _env_positive_int("STEP2_SOURCE_GAP_ADJUDICATION_MAX_TOKENS", 4096)
        if self.provider == "openai":
            token_key = "max_completion_tokens" if self.default_model.startswith("gpt-5") else "max_tokens"
            payload: Dict[str, Any] = {
                "model": self.default_model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                token_key: max_tokens,
                "temperature": 0.0,
            }
            structured_format = self._source_gap_adjudication_response_format()
            if structured_format:
                payload["response_format"] = structured_format
            return payload
        if self.provider == "deepseek":
            payload = {
                "model": self.default_model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "max_tokens": self._clamp_deepseek_max_tokens(max_tokens, self.default_model),
                "temperature": 0.0,
                "top_p": 0.95,
                "stream": False,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
            structured_format = self._source_gap_adjudication_response_format()
            if structured_format:
                payload["response_format"] = structured_format
            return payload
        return {
            "model": self.default_model,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._clamp_anthropic_max_tokens(max_tokens, self.default_model),
            "temperature": 0.0,
        }

    def _parse_source_gap_adjudication_decisions(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        decisions = payload.get("decisions")
        nested = payload.get("source_gap_adjudication")
        if decisions is None and isinstance(nested, dict):
            decisions = nested.get("decisions")
        if not isinstance(decisions, list):
            return []
        return [decision for decision in decisions if isinstance(decision, dict)]

    def _adjudicate_source_structure_gaps(
        self,
        tree: Dict[str, Any],
        focused_sections: Sequence[Dict[str, Any]],
        source_structure_gaps: Sequence[str],
        *,
        stage_label: str,
    ) -> List[Dict[str, Any]]:
        gaps = [str(gap) for gap in source_structure_gaps or [] if str(gap).strip()]
        if not gaps:
            return []
        fingerprint = _stable_json_hash(
            {
                "gaps": gaps,
                "candidate_nodes": _source_gap_candidate_nodes(tree, gaps),
                "sections": _source_gap_section_evidence(focused_sections),
            }
        )[:16]
        safe_stage = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(stage_label or "source_gap"))[:80]
        cache_filename = f"source_gap_adjudication_{safe_stage}_{fingerprint}.json"
        cached = self._load_from_cache(cache_filename)
        if isinstance(cached, dict):
            cached_payload = cached.get("parsed_response") if "parsed_response" in cached else cached
            cached_decisions = self._parse_source_gap_adjudication_decisions(cached_payload)
            if cached_decisions:
                logger.info(
                    "Loaded source-gap adjudication for %s from cache: %d decision(s)",
                    stage_label,
                    len(cached_decisions),
                )
                return cached_decisions

        payload = self._source_gap_adjudication_payload(tree, focused_sections, gaps)
        result = self._call_api_with_retry(payload)
        response_meta = self._extract_response_metadata(result)
        self._append_usage_log(
            stage="source_gap_adjudication",
            meta=response_meta,
            extra={"stage_label": stage_label, "gap_count": len(gaps)},
        )
        raw_response, response_kind = self._extract_refine_raw_response(result)
        parsed = json.loads(self._clean_raw_response(raw_response))
        decisions = self._parse_source_gap_adjudication_decisions(parsed)
        prompt_text = ""
        if isinstance(payload.get("messages"), list) and payload["messages"]:
            prompt_text = str(payload["messages"][-1].get("content") or "")
        self._save_to_cache(
            cache_filename,
            {
                "prompt": prompt_text,
                "raw_response": raw_response,
                "response_kind": response_kind,
                "timestamp": datetime.now().isoformat(),
                "section_info": stage_label,
                "provider": response_meta.get("provider"),
                "model": response_meta.get("model"),
                "stop_reason": response_meta.get("stop_reason"),
                "finish_reason": response_meta.get("finish_reason"),
                "usage": response_meta.get("usage"),
                "token_usage": response_meta.get("token_usage"),
                "parsed_response": parsed,
            },
        )
        return decisions

    def _source_gaps_for_hard_gate(
        self,
        tree: Dict[str, Any],
        focused_sections: Sequence[Dict[str, Any]],
        source_structure_gaps: Sequence[str],
        *,
        traffic_file: Any = None,
        stage_label: str = "source_gap",
    ) -> List[str]:
        gaps = [str(gap) for gap in source_structure_gaps or [] if str(gap).strip()]
        if not _should_fail_source_backed_refine_gap(gaps, traffic_file=traffic_file):
            return []
        if not _env_flag("STEP2_SOURCE_GAP_LLM_ADJUDICATION", "1"):
            return gaps
        try:
            decisions = self._adjudicate_source_structure_gaps(
                tree,
                focused_sections,
                gaps,
                stage_label=stage_label,
            )
        except Exception as exc:
            logger.warning(
                "Source-gap LLM adjudication failed for %s; keeping deterministic hard gaps: %s",
                stage_label,
                exc,
            )
            return gaps
        unresolved = _source_gap_unresolved_after_adjudication(gaps, decisions)
        waived = [gap for gap in gaps if gap not in unresolved]
        if waived:
            logger.info(
                "Source-gap LLM adjudication waived %d/%d candidate gap(s) for %s: %s",
                len(waived),
                len(gaps),
                stage_label,
                waived[:12],
            )
        return unresolved

    def _clip_patch_budget(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Limit expansion budget to keep patches manageable."""
        if not isinstance(patch, dict):
            return patch
        max_new_nodes = 20
        max_edge_adds = 20

        new_nodes = patch.get("new_nodes") or []
        edge_adds = patch.get("edge_adds") or []
        if len(new_nodes) > max_new_nodes:
            patch["new_nodes"] = new_nodes[:max_new_nodes]
        if len(edge_adds) > max_edge_adds:
            patch["edge_adds"] = edge_adds[:max_edge_adds]
        for node in patch.get("new_nodes", []):
            node.setdefault("children_ids", [])
            node.setdefault("byte_order", "big")
            node.setdefault("message_type", "bidirectional")
        return patch

    def _is_patch_doc_grounded(self, patch: Dict[str, Any], focused_secs: List[Dict[str, Any]]) -> bool:
        """Verify new nodes reference the focused documentation sections via source strings."""
        if not isinstance(patch, dict):
            return True
        new_nodes = patch.get("new_nodes") or []
        if not new_nodes:
            return True

        labels: Set[str] = set()
        for sec in focused_secs or []:
            for key in ("number", "title", "source_file"):
                raw = sec.get(key)
                val = str(raw if raw is not None else "").strip()
                if val:
                    labels.add(val.lower())
        for sec in focused_secs or []:
            raw_number = sec.get("number")
            num = str(raw_number if raw_number is not None else "").strip()
            if num:
                lowered = num.lower()
                labels.add(lowered)
                labels.add(re.sub(r"\s+", "", lowered))

        if not labels:
            return True

        for node in new_nodes:
            source = node.get("source")
            if not isinstance(source, str):
                return False
            source_val = source.strip().lower()
            if not source_val:
                return False
            if not any(label and label in source_val for label in labels):
                return False
        return True

    def _normalize_selector_like_name(self, name: Any) -> str:
        raw = re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())
        # Normalize common direction/role suffixes so shadow selectors such as
        # Opcode_Request / Opcode_Response are treated as the same
        # shared selector family as Opcode.
        changed = True
        while changed and raw:
            changed = False
            for suffix in (
                "requestvariant",
                "responsevariant",
                "errorvariant",
                "exceptionvariant",
                "request",
                "response",
                "variant",
                "error",
                "exception",
                "req",
                "resp",
                "rsp",
            ):
                if raw.endswith(suffix) and len(raw) > len(suffix):
                    raw = raw[: -len(suffix)]
                    changed = True
                    break
        aliases = {
            "functioncode": "functioncode",
            "opcode": "opcode",
            "commandcode": "commandcode",
            "commandid": "commandid",
            "messagetype": "messagetype",
            "subfunctioncode": "subfunctioncode",
            "subfunction": "subfunctioncode",
            "type": "type",
            "kind": "kind",
        }
        return aliases.get(raw, raw)

    def _looks_like_selector_like_node(self, node: Dict[str, Any]) -> bool:
        if not isinstance(node, dict):
            return False
        node_type = canonicalize_node_type(str(node.get("node_type") or ""))
        if node_type == "selector":
            return True
        name_norm = self._normalize_selector_like_name(node.get("name"))
        if name_norm not in {
            "functioncode",
            "opcode",
            "commandcode",
            "commandid",
            "messagetype",
            "subfunctioncode",
            "type",
            "kind",
        }:
            return False
        size_bits = node.get("size_bits")
        if isinstance(size_bits, (int, float)) and not isinstance(size_bits, bool):
            return int(size_bits) <= 32
        if isinstance(size_bits, str) and size_bits.strip().isdigit():
            return int(size_bits.strip()) <= 32
        return node_type == "field"

    def _detect_invalid_refine_patch(self, current_tree: Dict[str, Any], patch: Dict[str, Any]) -> Optional[str]:
        if not isinstance(current_tree, dict) or not isinstance(patch, dict):
            return None

        def _coerce_id(raw: Any) -> Any:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                try:
                    return int(raw)
                except Exception:
                    return raw
            if isinstance(raw, str):
                text = raw.strip()
                if text and (text.isdigit() or (text.startswith("-") and text[1:].isdigit())):
                    try:
                        return int(text)
                    except Exception:
                        return raw
            return raw

        current_nodes = current_tree.get("nodes") or []
        new_nodes = patch.get("new_nodes") or []
        edge_adds = patch.get("edge_adds") or []
        if not new_nodes:
            return None

        all_existing: Dict[Any, Dict[str, Any]] = {}
        for node in current_nodes:
            node_id = _coerce_id(node.get("id", node.get("node_id")))
            if node_id is not None:
                all_existing[node_id] = node

        all_new: Dict[Any, Dict[str, Any]] = {}
        for node in new_nodes:
            node_id = _coerce_id(node.get("id", node.get("node_id")))
            if node_id is not None:
                all_new[node_id] = node

        current_edges = current_tree.get("edges") or []

        def _get_parent_id(node: Dict[str, Any]) -> Any:
            return _coerce_id(node.get("parent_id"))

        combined_nodes: List[Dict[str, Any]] = list(current_nodes) + list(new_nodes)

        def _variant_local_selector_names(variant_node: Dict[str, Any]) -> Set[str]:
            variant_id = _coerce_id(variant_node.get("id", variant_node.get("node_id")))
            if variant_id is None:
                return set()
            names: Set[str] = set()
            for candidate in combined_nodes:
                if _get_parent_id(candidate) != variant_id:
                    continue
                if not self._looks_like_selector_like_node(candidate):
                    continue
                names.add(self._normalize_selector_like_name(candidate.get("name")))
            return names

        new_variants: List[Dict[str, Any]] = []
        for node in new_nodes:
            if canonicalize_node_type(str(node.get("node_type") or "")) == "variant":
                new_variants.append(node)
        if not new_variants:
            return None

        local_selectors_by_variant: Dict[Any, List[Tuple[Any, str]]] = {}
        local_selector_ids: Set[Any] = set()
        for variant in new_variants:
            variant_id = _coerce_id(variant.get("id", variant.get("node_id")))
            if variant_id is None:
                continue
            matches: List[Tuple[Any, str]] = []
            for child in new_nodes:
                if _get_parent_id(child) != variant_id:
                    continue
                if not self._looks_like_selector_like_node(child):
                    continue
                child_id = _coerce_id(child.get("id", child.get("node_id")))
                if child_id is None:
                    continue
                matches.append((child_id, self._normalize_selector_like_name(child.get("name"))))
                local_selector_ids.add(child_id)
            if matches:
                local_selectors_by_variant[variant_id] = matches

        if not local_selectors_by_variant:
            return None

        def _iter_siblings(parent_id: Any) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for node in current_nodes:
                if _coerce_id(node.get("id", node.get("node_id"))) in all_new:
                    continue
                if _get_parent_id(node) == parent_id:
                    out.append(node)
            for node in new_nodes:
                if _get_parent_id(node) == parent_id:
                    out.append(node)
            return out

        reasons: List[str] = []

        for variant in new_variants:
            variant_id = _coerce_id(variant.get("id", variant.get("node_id")))
            if variant_id not in local_selectors_by_variant:
                continue
            parent_id = _get_parent_id(variant)
            siblings = _iter_siblings(parent_id)
            for _local_selector_id, selector_name in local_selectors_by_variant[variant_id]:
                for sibling in siblings:
                    sibling_id = _coerce_id(sibling.get("id", sibling.get("node_id")))
                    if sibling_id == variant_id:
                        continue
                    if _get_parent_id(sibling) != parent_id:
                        continue
                    if canonicalize_node_type(str(sibling.get("node_type") or "")) == "variant":
                        sibling_message_type = str(sibling.get("message_type") or "bidirectional")
                        variant_message_type = str(variant.get("message_type") or "bidirectional")
                        if sibling_message_type == variant_message_type:
                            sibling_selector_names = _variant_local_selector_names(sibling)
                            if selector_name in sibling_selector_names:
                                reasons.append(
                                    f"variant {variant.get('name')} and sibling {sibling.get('name')} each embed local selector {selector_name} under parent {parent_id}"
                                )
                                break
                    if not self._looks_like_selector_like_node(sibling):
                        continue
                    if self._normalize_selector_like_name(sibling.get("name")) != selector_name:
                        continue
                    reasons.append(
                        f"variant {variant.get('name')} duplicates shared selector {sibling.get('name')} under parent {parent_id}"
                    )
                    break

        grouped_variants: Dict[Tuple[Any, str, str], List[str]] = defaultdict(list)
        for variant in new_variants:
            variant_id = _coerce_id(variant.get("id", variant.get("node_id")))
            parent_id = _get_parent_id(variant)
            message_type = str(variant.get("message_type") or "bidirectional")
            for _local_selector_id, selector_name in local_selectors_by_variant.get(variant_id, []):
                grouped_variants[(parent_id, message_type, selector_name)].append(str(variant.get("name") or variant_id))
        for (parent_id, message_type, selector_name), variant_names in grouped_variants.items():
            if len(variant_names) >= 2:
                joined = ", ".join(sorted(variant_names))
                reasons.append(
                    f"multiple {message_type} variants under parent {parent_id} each embed local selector {selector_name}: {joined}"
                )

        for edge in edge_adds:
            if str(edge.get("rel") or "").strip().lower() != "condition_on":
                continue
            src = _coerce_id(edge.get("src"))
            dst = _coerce_id(edge.get("dst"))
            if src in local_selector_ids and dst in local_selectors_by_variant:
                reasons.append(
                    f"condition_on edge {src}->{dst} routes from a local selector inside the destination variant instead of a shared selector"
                )

        def _normalize_formula(formula: Any) -> str:
            return re.sub(r"\s+", "", str(formula or ""))

        existing_condition_routes: Dict[Tuple[Any, str, str], List[Any]] = defaultdict(list)
        for edge in current_edges:
            if str(edge.get("rel") or "").strip().lower() != "condition_on":
                continue
            src = _coerce_id(edge.get("src"))
            dst = _coerce_id(edge.get("dst"))
            formula = _normalize_formula(edge.get("formula"))
            message_type = str(edge.get("message_type") or "bidirectional")
            existing_condition_routes[(src, message_type, formula)].append(dst)

        new_condition_routes: Dict[Tuple[Any, str, str], List[Any]] = defaultdict(list)
        for edge in edge_adds:
            if str(edge.get("rel") or "").strip().lower() != "condition_on":
                continue
            src = _coerce_id(edge.get("src"))
            dst = _coerce_id(edge.get("dst"))
            formula = _normalize_formula(edge.get("formula"))
            message_type = str(edge.get("message_type") or "bidirectional")
            if dst in all_new:
                new_condition_routes[(src, message_type, formula)].append(dst)

        for route_key, dsts in new_condition_routes.items():
            src, message_type, formula = route_key
            if len(dsts) >= 2:
                reasons.append(
                    f"multiple new {message_type} variants reuse identical condition_on from selector {src}: {formula or '<empty>'}"
                )
            existing_dsts = existing_condition_routes.get(route_key) or []
            if existing_dsts:
                reasons.append(
                    f"new {message_type} variant duplicates existing condition_on from selector {src}: {formula or '<empty>'}"
                )

        if reasons:
            return "; ".join(dict.fromkeys(reasons))
        return None

    def _validate_and_fix_json_structure(self, text: str) -> str:
        if not text:
            return text

        try:
            import json
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}, attempting repair...")

            repaired = text

            import re
            repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

            repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)

            repaired_control = self._escape_control_chars_in_json_strings(repaired)

            candidates = []
            if repaired_control != repaired:
                candidates.append((repaired_control, "after escaping control characters"))
            candidates.append((repaired, "after structural fixes"))

            for candidate, description in candidates:
                try:
                    json.loads(candidate)
                    logger.info(f"Successfully repaired JSON {description}")
                    return candidate
                except json.JSONDecodeError as repair_error:
                    logger.debug(f"JSON still invalid {description}: {repair_error}")

            logger.warning("Could not repair JSON, returning original text")
            return text

    def _looks_like_truncated_json(self, text: str) -> bool:
        """Heuristic for incomplete model outputs that ended before JSON closed."""
        if not text or not text.strip():
            return False
        stripped = text.rstrip()
        if stripped[-1] not in {'}', ']'}:
            return True

        stack: List[str] = []
        in_string = False
        escape = False
        for ch in stripped:
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in '{[':
                stack.append(ch)
            elif ch == '}':
                if not stack or stack[-1] != '{':
                    return True
                stack.pop()
            elif ch == ']':
                if not stack or stack[-1] != '[':
                    return True
                stack.pop()
        return in_string or bool(stack)

    def _escape_control_chars_in_json_strings(self, text: str) -> str:
        if not text:
            return text

        result: List[str] = []
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    result.append(ch)
                    escape = False
                else:
                    if ch == '\\':
                        escape = True
                        result.append(ch)
                    elif ch == '"':
                        in_string = False
                        result.append(ch)
                    elif ord(ch) < 0x20:
                        if ch == '\n':
                            result.append('\\n')
                        elif ch == '\r':
                            result.append('\\r')
                        elif ch == '\t':
                            result.append('\\t')
                        else:
                            result.append(f'\\u{ord(ch):04x}')
                    else:
                        result.append(ch)
            else:
                result.append(ch)
                if ch == '"':
                    backslash_count = 0
                    j = idx - 1
                    while j >= 0 and text[j] == '\\':
                        backslash_count += 1
                        j -= 1
                    if backslash_count % 2 == 0:
                        in_string = True
                        escape = False

        return ''.join(result)

    def _load_from_cache(self, filename: str) -> Optional[Any]:
        if os.getenv("PARGEN_STEP2_FORCE_REBUILD", "").strip().lower() in {"1", "true", "yes", "on"}:
            return None
        cache_path = os.path.join(self.cache_dir, filename)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("finish_reason") in {"error", "content_filter", "cancelled"}:
                    logger.warning("Ignoring failed provider response cache %s", cache_path)
                    return None
                if self.input_signature:
                    expected = str(self.input_signature.get("sha256") or "")
                    actual_sig = data.get("input_signature") if isinstance(data, dict) else None
                    actual = str((actual_sig or {}).get("sha256") or "")
                    if actual != expected and not _stage2_signatures_equivalent(self.input_signature, actual_sig):
                        allow_unsigned = os.getenv(
                            "PARGEN_STAGE2_ALLOW_UNSIGNED_CACHE", "0"
                        ).strip().lower() in {"1", "true", "yes", "on"}
                        if not allow_unsigned:
                            logger.info(
                                "Ignoring stale Stage B cache %s: input signature mismatch (expected=%s actual=%s)",
                                cache_path,
                                expected[:12],
                                actual[:12] if actual else "missing",
                            )
                            return None
                        logger.warning(
                            "Using unsigned/stale Stage B cache because PARGEN_STAGE2_ALLOW_UNSIGNED_CACHE=1: %s",
                            cache_path,
                        )
                logger.info(f"Loaded from cache: {cache_path}")
                return data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None

    def _call_api_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ...openrouter_routing import routed_payload
        from ... import degenerate_retry
        payload = routed_payload(payload, self.base_url)
        built_payload = payload
        # A retry of a request whose greedy reply was degenerate is sampled (STEP2_DEGENERATE_RETRY_SAMPLING).
        payload, retry_sampling = degenerate_retry.apply(built_payload)
        api_start_time = time.time()
        model_name = payload.get("model", "unknown")
        for attempt in range(self.max_retries):
            try:
                self._ensure_api_key()
                try:
                    approx_chars = len(json.dumps(payload, ensure_ascii=False))
                except Exception:
                    approx_chars = -1
                logger.info(
                    "Calling provider %s (attempt %s/%s), timeouts=(%ss connect, %ss read, %ss wall), payload≈%s chars",
                    model_name,
                    attempt + 1,
                    self.max_retries,
                    self.connect_timeout,
                    self.read_timeout,
                    self.provider_wall_timeout,
                    approx_chars,
                )
                response = _run_with_wall_timeout(
                    lambda: requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=(self.connect_timeout, self.read_timeout),
                    ),
                    timeout_sec=self.provider_wall_timeout,
                    label=f"provider {model_name} request",
                )

                if response.status_code == 200:
                    response_data = response.json()
                    response_data['_requested_provider'] = payload.get('provider')
                    if retry_sampling is not None:
                        response_data['_degenerate_retry_sampling'] = retry_sampling
                    degenerate_retry.note(built_payload, response_data, retry_sampling, log_dir=self.cache_dir)
                    api_duration = time.time() - api_start_time
                    choices = response_data.get("choices") or []
                    if choices and choices[0].get("finish_reason") in {"error", "content_filter", "cancelled"}:
                        meta = self._extract_response_metadata(response_data)
                        self._append_usage_log(stage="provider_failure", meta=meta, extra={"attempt": attempt + 1})
                        raise RuntimeError(f"Provider failed: finish_reason={choices[0].get('finish_reason')}")
                    if isinstance(response_data, dict) and response_data.get("error") is not None:
                        error_payload = response_data.get("error")
                        if isinstance(error_payload, dict):
                            error_text = (
                                error_payload.get("message")
                                or error_payload.get("code")
                                or json.dumps(error_payload, ensure_ascii=False)
                            )
                        else:
                            error_text = str(error_payload)
                        logger.warning("API returned error envelope: %s", error_text[:1000])
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                        raise RuntimeError(f"API returned error envelope: {error_text[:1000]}")
                    logger.info(
                        "API response successful: %s (%.2fs, attempt %s), response keys: %s",
                        model_name,
                        api_duration,
                        attempt + 1,
                        response_data.keys(),
                    )
                    self._log_response_metadata(response_data, stage="API")
                    if self.provider == "deepseek" and 'choices' in response_data:
                        if response_data['choices'] and 'message' in response_data['choices'][0]:
                            message = response_data['choices'][0]['message']
                            content = message.get('content') or ''
                            # Some servers name the field `reasoning`, others `reasoning_content`.
                            reasoning_content = message.get('reasoning_content') or message.get('reasoning') or ''
                            logger.info(f"DeepSeek content preview: {content[:200]}...")
                            logger.info(f"DeepSeek reasoning preview: {reasoning_content[:200]}...")
                            if not content.strip() and not (message.get("tool_calls") or []):
                                logger.warning(
                                    "DeepSeek returned empty content%s",
                                    "" if reasoning_content.strip() else " and reasoning",
                                )
                                # Keep the whole reply for diagnosis (one JSONL per cache namespace):
                                # the caller only sees a JSON parse error.
                                try:
                                    dump = Path(self.cache_dir) / "empty_responses.jsonl"  # namespaced by analyze_protocol
                                    dump.parent.mkdir(parents=True, exist_ok=True)
                                    with dump.open("a", encoding="utf-8") as fh:
                                        fh.write(json.dumps({"at": time.time(), "model": model_name,
                                                             "response": response_data}, ensure_ascii=False) + "\n")
                                except Exception as exc:
                                    logger.warning("Could not save the empty response: %s", exc)
                    return response_data
                else:
                    logger.warning(f"API error: {response.status_code}")
                    logger.warning(f"API error response: {response.text[:1000]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

        raise Exception("API call failed after all retries")

    def _refine_tree_with_raw_data(self, tree, raw_sections, sections, traffic_file=None):

        if not sections:
            logger.info("No sections provided, skipping refinement")
            return tree

        logger.info(f"Starting CONTROLLED section-by-section refinement with {len(sections)} sections")

        batch_size = 1
        num_sections = len(sections)
        batch_count = (num_sections + batch_size - 1) // batch_size
        logger.info(f"Global+Local Attention: processing ALL {num_sections} sections in {batch_count} batches")
        logger.info(f"Batch size: {batch_size} sections per batch for local context focus")

        batch_cache_files = []

        cumulative_tree = tree.copy() if isinstance(tree, dict) else tree
        cumulative_tree = self._inline_validate_and_fix(
            cumulative_tree,
            sections,
            raw_sections,
            "inline_initial_tree.json",
            "initial_phase",
            traffic_file=traffic_file,
        )
        traffic_anchor_records = _build_traffic_anchor_records(
            cumulative_tree,
            traffic_file,
            max_packets=_env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 1),
        )
        anchor_success_count = sum(1 for record in traffic_anchor_records if isinstance(record, dict) and record.get("success"))
        if anchor_success_count:
            first_anchor = next(record for record in traffic_anchor_records if isinstance(record, dict) and record.get("success"))
            active_selector_count = len(first_anchor.get("active_variants") or {})
            logger.info(
                "Traffic-anchor locate/gate active after initial validation: %d/%d observed packet(s) parse; first_path_nodes=%d active_selectors=%d",
                anchor_success_count,
                len(traffic_anchor_records),
                len(first_anchor.get("parsing_path") or []),
                active_selector_count,
            )
        elif traffic_file:
            logger.warning("Traffic-anchor gate has no successful anchors after initial validation; parseability protection is inactive.")
        if traffic_anchor_records and _env_flag("STEP2_TRAFFIC_ANCHOR_LOCATE", "1"):
            locate_order = _traffic_anchor_locate_order(sections, cumulative_tree, traffic_anchor_records)
            if locate_order != list(range(len(sections))):
                logger.info(
                    "Traffic-anchor locate reordered refine sections; top focused indices=%s",
                    locate_order[: min(8, len(locate_order))],
                )
                sections = _reorder_by_indices(sections, locate_order) or list(sections)
                raw_sections = _reorder_by_indices(raw_sections, locate_order)

        def _obligation_handoff_after_anchor_batches() -> Optional[int]:
            raw = os.getenv("STEP2_STAGEB_OBLIGATION_HANDOFF_AFTER_ANCHOR_BATCHES", "").strip()
            if not raw:
                return None
            try:
                value = int(raw)
            except Exception:
                logger.warning(
                    "Ignoring invalid STEP2_STAGEB_OBLIGATION_HANDOFF_AFTER_ANCHOR_BATCHES=%r",
                    raw,
                )
                return None
            return max(0, value)

        def _traffic_anchor_parseable(candidate_tree: Dict[str, Any]) -> bool:
            if not traffic_file:
                return False
            records = _build_traffic_anchor_records(
                candidate_tree,
                traffic_file,
                max_packets=_env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 1),
            )
            return any(isinstance(record, dict) and record.get("success") for record in records)

        obligation_handoff_after_batches = _obligation_handoff_after_anchor_batches()

        def _should_handoff_to_obligation_driver(completed_batches: int) -> bool:
            if obligation_handoff_after_batches is None:
                return False
            if completed_batches < obligation_handoff_after_batches:
                return False
            if not _env_flag("STEP2_STAGEB_COVERAGE_OBLIGATION", "0"):
                logger.warning(
                    "Ignoring StageB obligation handoff request because STEP2_STAGEB_COVERAGE_OBLIGATION is disabled."
                )
                return False
            if _traffic_source_backed_gap_guard_active(traffic_file):
                gap_counts = _source_structure_gap_counts(cumulative_tree, sections)
                if any(gap_counts):
                    logger.info(
                        "Deferring StageB obligation handoff because source-backed gaps remain "
                        "(packet=%d selector=%d).",
                        gap_counts[0],
                        gap_counts[1],
                    )
                    return False
            return _traffic_anchor_parseable(cumulative_tree)

        def _maybe_dynamic_traffic_anchor_locate(batch_idx: int, reason: str) -> None:
            nonlocal sections, raw_sections
            if not traffic_file or not _env_flag("STEP2_TRAFFIC_ANCHOR_LOCATE", "1"):
                return
            processed_count = min((batch_idx + 1) * batch_size, num_sections)
            if processed_count >= num_sections:
                return
            records = _build_traffic_anchor_records(
                cumulative_tree,
                traffic_file,
                max_packets=_env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 1),
            )
            successful = [record for record in records if isinstance(record, dict) and record.get("success")]
            if not successful:
                return
            first_anchor = successful[0]
            logger.info(
                "Traffic-anchor dynamic locate active after %s: %d/%d observed packet(s) parse; processed_sections=%d remaining_sections=%d first_path_nodes=%d active_selectors=%d",
                reason,
                len(successful),
                len(records),
                processed_count,
                max(0, num_sections - processed_count),
                len(first_anchor.get("parsing_path") or []),
                len(first_anchor.get("active_variants") or {}),
            )
            new_sections, new_raw_sections, tail_order = _traffic_anchor_reorder_remaining(
                sections,
                raw_sections,
                cumulative_tree,
                records,
                processed_count=processed_count,
            )
            if tail_order and tail_order != list(range(len(tail_order))):
                logger.info(
                    "Traffic-anchor dynamic locate reordered remaining sections after %s; processed=%d tail_order=%s",
                    reason,
                    processed_count,
                    tail_order[: min(8, len(tail_order))],
                )
                sections = new_sections
                raw_sections = new_raw_sections

        def _cache_candidate_is_usable(candidate_tree: Dict[str, Any], batch_number: int) -> bool:
            try:
                min_node_retention_ratio = float(
                    os.getenv("STEP2_REFINEMENT_MIN_NODE_RETENTION_RATIO", "0.7").strip()
                )
            except Exception:
                min_node_retention_ratio = 0.7
            min_node_retention_ratio = min(1.0, max(0.0, min_node_retention_ratio))
            prev_nodes = len(cumulative_tree.get("nodes", []) or [])
            candidate_nodes = len(candidate_tree.get("nodes", []) or [])
            min_candidate_nodes = int(prev_nodes * min_node_retention_ratio)
            if prev_nodes > 0 and candidate_nodes < min_candidate_nodes:
                logger.warning(
                    "Ignoring cached refine tree for batch %s: node count %d -> %d below retention threshold %d.",
                    batch_number,
                    prev_nodes,
                    candidate_nodes,
                    min_candidate_nodes,
                )
                return False

            try:
                max_error_increase = int(os.getenv("STEP2_REFINEMENT_MAX_HARD_ERROR_INCREASE", "0").strip())
            except Exception:
                max_error_increase = 0
            try:
                max_error_multiplier = float(
                    os.getenv("STEP2_REFINEMENT_MAX_HARD_ERROR_MULTIPLIER", "5").strip()
                )
            except Exception:
                max_error_multiplier = 5.0
            max_error_increase = max(0, max_error_increase)
            max_error_multiplier = max(1.0, max_error_multiplier)
            try:
                prev_report = validate_protocol_tree(json.dumps({"protocol_tree": cumulative_tree}))
                candidate_report = validate_protocol_tree(json.dumps({"protocol_tree": candidate_tree}))
                prev_errors = len(getattr(prev_report, "errors", []) or [])
                candidate_errors = len(getattr(candidate_report, "errors", []) or [])
                allowed_errors = prev_errors + max_error_increase
                if prev_errors > 0:
                    allowed_errors = max(allowed_errors, int(prev_errors * max_error_multiplier))
                if candidate_errors > allowed_errors:
                    logger.warning(
                        "Ignoring cached refine tree for batch %s: hard_errors %d -> %d exceeds allowed %d.",
                        batch_number,
                        prev_errors,
                        candidate_errors,
                        allowed_errors,
                    )
                    return False
            except Exception as gate_exc:
                logger.warning("Cached refine regression gate failed for batch %s: %s", batch_number, gate_exc)
            if traffic_file and _env_flag("STEP2_TRAFFIC_ANCHOR_GATE", "1"):
                before_anchor = _build_traffic_anchor_records(
                    cumulative_tree,
                    traffic_file,
                    max_packets=_env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 1),
                )
                if before_anchor:
                    after_anchor = _build_traffic_anchor_records(
                        candidate_tree,
                        traffic_file,
                        max_packets=_env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 1),
                    )
                    anchor_regression = _traffic_anchor_regression_reason(before_anchor, after_anchor)
                    if anchor_regression:
                        logger.warning(
                            "Ignoring cached refine tree for batch %s: observed traffic anchor regression: %s",
                            batch_number,
                            anchor_regression,
                        )
                        return False
            return True

        for batch_idx in range(batch_count):
            if _should_handoff_to_obligation_driver(batch_idx):
                logger.info(
                    "StageB obligation handoff: skipping ordinary refine after %d/%d completed batch(es); "
                    "observed traffic anchor remains parseable and coverage-obligation driver is enabled.",
                    batch_idx,
                    batch_count,
                )
                break
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_sections)

            cache_filename = f"refine_section_{start_idx}_to_{end_idx}_ai_response.json"
            batch_cache_files.append(cache_filename)
            cached_data = self._load_from_cache(cache_filename)

            sections_with_marks = mark_sections_by_batch(
                sections,
                batch_start=start_idx,
                batch_size=batch_size,
                text_key="content",
                out_key="content",
                strip_existing=True,
                priority="high",
                set_flag_key="is_focused",
            )
            batch_sections = sections_with_marks
            if cached_data and 'protocol_tree' in cached_data:
                candidate_tree = cached_data['protocol_tree']
                if not _cache_candidate_is_usable(candidate_tree, batch_idx + 1):
                    cached_data = None
                    logger.warning("Cached protocol_tree for batch %s rejected; regenerating...", batch_idx + 1)
                else:
                    logger.info(f"✓ Batch {batch_idx+1}/{batch_count} loaded from cache: {cache_filename}")
                    cumulative_tree = candidate_tree
                    is_last_batch = (batch_idx == batch_count - 1)
                    cumulative_tree = self._inline_validate_and_fix(
                        cumulative_tree,
                        sections,
                        raw_sections,
                        cache_filename,
                        f"refine_batch_{batch_idx + 1}",
                        force_full_fix=is_last_batch,
                        traffic_file=traffic_file,
                    )
                    _maybe_dynamic_traffic_anchor_locate(batch_idx, f"cached refine batch {batch_idx + 1}")
                    continue
            if cached_data and 'protocol_tree' in cached_data:
                logger.info(f"✓ Batch {batch_idx+1}/{batch_count} loaded from cache: {cache_filename}")
                cumulative_tree = cached_data['protocol_tree']
                is_last_batch = (batch_idx == batch_count - 1)
                cumulative_tree = self._inline_validate_and_fix(
                    cumulative_tree,
                    sections,
                    raw_sections,
                    cache_filename,
                    f"refine_batch_{batch_idx + 1}",
                    force_full_fix=is_last_batch,
                    traffic_file=traffic_file,
                )
                _maybe_dynamic_traffic_anchor_locate(batch_idx, f"cached refine batch {batch_idx + 1}")
                continue
            elif cached_data and 'raw_response' in cached_data:
                try:
                    raw_response = cached_data['raw_response']
                    cleaned_response = self._clean_raw_response(raw_response)
                    batch_result = json.loads(cleaned_response)
                    tree_obj = batch_result.get('protocol_tree') if isinstance(batch_result, dict) else None
                    if isinstance(tree_obj, dict):
                        if not _cache_candidate_is_usable(tree_obj, batch_idx + 1):
                            logger.warning("Cached raw_response tree for batch %s rejected; regenerating...", batch_idx + 1)
                            raise ValueError("Cached response failed regression gate")
                        cumulative_tree = tree_obj
                    else:
                        logger.warning("Cached raw_response missing protocol_tree; skipping cache reuse.")
                        raise ValueError("Invalid cached response")
                    logger.info(f"✓ Batch {batch_idx+1}/{batch_count} parsed from cached raw_response")
                    cumulative_tree = self._inline_validate_and_fix(
                        cumulative_tree,
                        sections,
                        raw_sections,
                        cache_filename,
                        f"refine_batch_{batch_idx + 1}",
                        traffic_file=traffic_file,
                    )
                    _maybe_dynamic_traffic_anchor_locate(batch_idx, f"cached raw refine batch {batch_idx + 1}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to parse cached raw_response for batch {batch_idx+1}: {e}, regenerating...")
            else:
                logger.warning(f"No valid cache found for batch {batch_idx+1}, regenerating...")
            batch_sections = sections_with_marks
            tree_snippet_json = json.dumps(
                self._filter_protocol_tree_content(cumulative_tree),
                ensure_ascii=False,
                indent=2,
                cls=NumpyEncoder,
            )
            focused_secs = [s for i, s in enumerate(sections_with_marks) if start_idx <= i < end_idx]
            marked_summary = [_refine_section_digest(sec) for sec in focused_secs]
            validator_report = validate_protocol_tree(json.dumps({"protocol_tree": cumulative_tree}))
            topk_errors = (getattr(validator_report, "errors", []) or [])[:5]
            marked_sections_summary_text = (
                json.dumps(marked_summary, ensure_ascii=False, indent=2)
                if marked_summary else "[]"
            )
            prompt_missing_packet_fields = _missing_packet_format_names(cumulative_tree, focused_secs)
            prompt_missing_selector_obligations = _missing_selector_obligation_names(
                cumulative_tree,
                focused_secs,
            )
            source_gap_context_text = _format_source_structure_gap_context(
                packet_fields=prompt_missing_packet_fields,
                selector_obligations=prompt_missing_selector_obligations,
            )
            focused_source_labels_text = _format_focused_source_labels(focused_secs)
            has_source_structure_gaps = bool(
                prompt_missing_packet_fields
                or prompt_missing_selector_obligations
            )
            if _traffic_context_only_noop_allowed(
                focused_secs,
                traffic_file=traffic_file,
                has_source_structure_gaps=has_source_structure_gaps,
            ):
                logger.info(
                    "Skipping refine batch %d/%d for traffic context-only non-wire section(s); "
                    "source-backed gap detector is empty.",
                    batch_idx + 1,
                    batch_count,
                )
                is_last_batch = (batch_idx == batch_count - 1)
                cumulative_tree = self._inline_validate_and_fix(
                    cumulative_tree,
                    sections,
                    raw_sections,
                    cache_filename,
                    f"refine_batch_{batch_idx + 1}",
                    force_full_fix=is_last_batch,
                    traffic_file=traffic_file,
                )
                try:
                    self._save_to_cache(
                        cache_filename,
                        {
                            "protocol_tree": cumulative_tree,
                            "timestamp": datetime.now().isoformat(),
                            "stage_label": f"refine_batch_{batch_idx + 1}",
                            "source": "traffic_context_only_noop_skip",
                            "skip_reason": "context_only_non_wire_section_without_source_backed_gaps",
                            "focused_sections": _format_focused_source_labels(focused_secs),
                        },
                    )
                except Exception as save_exc:
                    logger.warning(
                        "Failed to persist context-only no-op refine skip for batch %s: %s",
                        batch_idx + 1,
                        save_exc,
                    )
                _maybe_dynamic_traffic_anchor_locate(
                    batch_idx,
                    f"context-only skipped refine batch {batch_idx + 1}",
                )
                continue
            empty_patch_policy = (
                "The detector listed source-backed structure gaps for this batch. "
                "Returning `{}` or `patch_metadata.intent = no_change` is INVALID; produce a minimal patch "
                "that reduces at least one listed gap."
                if has_source_structure_gaps
                else "If the current snapshot already contains the focused batch's structure, return an empty patch (`{}`) exactly."
            )
            validator_errors_lines = [f"- {str(err)}" for err in topk_errors if err]
            validator_errors_text = (
                "\n".join(validator_errors_lines)
                if validator_errors_lines else "- None (validator clean in this snapshot)"
            )
            avoid_block_text = "- None (no rejected strategies recorded for this batch)."
            previous_patch_brief_text = "- None (first attempt for this batch)."
            current_anchor_records = _build_traffic_anchor_records(
                cumulative_tree,
                traffic_file,
                max_packets=_env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 1),
            )
            traffic_anchor_text = _format_traffic_anchor_context(cumulative_tree, current_anchor_records)
            batch_range_start = start_idx + 1
            batch_range_end = max(batch_range_start, end_idx)
            context_block = (
                "# CONTEXT\n"
                "- Current sub-tree snapshot (IDs are STABLE across this patch):\n"
                f"{tree_snippet_json}\n\n"
                "- Documentation sections (structured digest; includes packet_formats tables; focus only on these ranges for this batch "
                f"{batch_range_start}–{batch_range_end}):\n"
                f"{marked_sections_summary_text}\n\n"
                "- Focused source labels for grounding new nodes:\n"
                f"{focused_source_labels_text}\n\n"
                "- Source-backed structure gap detector (BEFORE this patch):\n"
                f"{source_gap_context_text}\n\n"
                "- Known validator issues (BEFORE this patch):\n"
                f"{validator_errors_text}\n\n"
                "- Observed traffic anchor / message validation context:\n"
                f"{traffic_anchor_text}\n\n"
                "- Recently tried & rejected strategies (do NOT repeat):\n"
                f"{avoid_block_text}\n\n"
                "- Previous patch summary (if any):\n"
                f"{previous_patch_brief_text}\n"
            )
            nodes_in_snapshot = cumulative_tree.get("nodes") or []
            used_ids = [
                int(node.get("node_id"))
                for node in nodes_in_snapshot
                if str(node.get("node_id")).isdigit()
            ]
            max_id = max(used_ids) if used_ids else 100
            reserve_start = max_id + 1
            reserve_end = max_id + 500
            instruction_block = f"""
	# OBJECTIVE — DOC‑GROUNDED DIFF (no mode switching)
	Your job is to incorporate **missing** parseable, on-wire structure described by the focused documentation in THIS batch.
	{empty_patch_policy}

    ## OBSERVED TRAFFIC ANCHOR RULE (PARSEABILITY ONLY)
	When an "Observed traffic anchor" is present:
	1) Treat it as evidence that the observed packet must still parse after your patch.
	2) You MAY add or restructure sibling variants, typed containers, and doc-derived leaves around the observed route.
	3) You MAY replace minimal observed-route containers/variants when needed to add documented siblings; do not preserve old node IDs just because traffic observed them.
	4) If the focused documentation describes the active variant's selector family, add missing sibling variants under the same shared selector instead of duplicating the selector inside each variant.
	5) A patch that makes the observed packet unparseable will be rejected by message validation.

## FIRST: DECIDE IF ANY CHANGE IS NEEDED (STOP-EARLY)
Before proposing any patch:
1) Check whether every focused parseable on-wire `packet_formats[].fields` item is already represented by an existing leaf in the snapshot (allow minor naming variants).
   - IMPORTANT: treat a table row as "represented" only if the matching leaf also has a compatible `bit_start` + `size_bits` (after base-offset alignment). If the name matches but the position/size does not, it is NOT represented — you must fix the misalignment.
   - IMPORTANT for TLV/IE protocols: a generic `TLV_Value` / `Information Element Value` / catch-all payload DOES NOT represent a named typed IE, grouped IE, or message-core row listed in the documentation.  It represents only the generic bytes fallback.  If the focused section names concrete on-wire IE/message fields, create or update typed variant/container nodes for those names instead of returning `{{}}`.
2) Treat data-model/catalog rows and `field_definitions` with variable/index-based positions, implementation-dependent placement, or `required=false` as source context for later packet sections, not as immediate wire leaves.
3) If the snapshot is validator-clean AND nothing is missing/misaligned for this batch, return `{{}}` exactly.

## MUST: EXPAND PACKET FORMAT TABLES INTO LEAF FIELDS (ONLY WHEN MISSING)
The focused documentation may include structured tables like `packet_formats[].fields` (with `byte_position`, `bit_position`, `size`, `data_type`).
When those are present:
1) You MUST create **leaf field nodes only for missing table rows** (do NOT duplicate existing fields; at most, add constraints/source/description).
2) You MUST align `byte_position` semantics (absolute vs relative) before emitting `bit_start`:
   - Many specs use byte positions **absolute from the protocol message start** , while others use offsets **relative to the PDU/variant start**.
   - Compute a `base_offset_bits` for EACH `packet_format` using ANCHOR alignment:
       * Find one or more anchor fields in the table that also exist in the current snapshot (by name match, e.g., Opcode / Opcode / Length / Type).
       * For each anchor, compute `candidate_base = <existing_node.bit_start> - <anchor.byte_position>*8`.
       * If multiple anchors exist, choose the MOST COMMON `candidate_base` (majority vote).
   - Then emit `bit_start = base_offset_bits + byte_position*8` (plus `bit_position` if provided).
   - Fallback only if no anchor exists: use the parent container base (`bit_start = parent.bit_start + byte_position*8`).
3) Parse sizes:
   - "1 Byte" → `size_bits = 8`
   - "2 Bytes" → `size_bits = 16`
   - "N Bytes" → model as `size_bits = val(<Byte_Count_ID>) * 8` and add a `length_of` edge if needed.

## MUST: DO NOT DROP RESERVED/PADDING TABLE ROWS (LAYOUT PLACEHOLDERS)
`packet_formats[].fields` rows named "Reserved"/"Padding"/"Spare"/"Unused" (or similar) are real on-wire bits/bytes.
- Always model them as leaf fields with the documented span; missing them causes coverage gaps and systematic offset drift.
- Only add hard constraints like `== 0` when the text explicitly mandates it ("MUST be zero"); otherwise keep them unconstrained.

## NOTE: `reserved_values` ARE NOT HARD CONSTRAINTS
`field_definitions[].constraints.reserved_values` (e.g., "9-31 reserved/unassigned") is descriptive.
- Do NOT translate it into a `range:` constraint and do NOT AND it with `enum` into a contradiction.
- The allowed numeric domain should remain the full bit-width or documented min/max (e.g., 5-bit -> 0..31). Put reserved ranges in description only unless explicitly forbidden.
4) Direction: leaf nodes under a request-only variant MUST use `message_type = "request"`, likewise for response.
5) No overlaps: if you keep a generic fallback payload variant (e.g., Request_Payload), its `condition_on` must be the **complement** of the modeled selector values to avoid overlap with specific variants.
	6) Every new leaf node MUST include `"source"` copied from the Focused source labels block for THIS batch (e.g., `source_file:number title`). A generic value like `"doc"` is invalid.
7) TLV specialization rule (general): if the current snapshot contains a generic TLV/IE sequence and the focused `packet_formats` lists named IE/grouped-IE/message rows, you MUST specialize the generic sequence by adding:
   - a shared Type/Tag selector if one is not already present,
   - typed `variant`/`container` nodes for the documented IE/message names,
   - typed body leaves for fields with documented spans, and placeholder typed containers for grouped IEs whose nested layout is in another focused section.
   Do NOT duplicate the generic Type/Length fields inside every typed variant unless the selector is physically inside that variant; reuse the existing shared Type/Length where possible.
8) SELECTOR RULE (GENERAL):
   - Do NOT duplicate an existing selector field inside each variant (e.g., avoid per-variant leaves like `Opcode_0F_Response` when a shared `Opcode` selector already exists).
   - Treat any small fixed-width control field with enum/range constraints as a selector candidate (e.g., OpCode, Message Type, Command ID, Type, Sub-Code).
   - If a matching selector already exists in the current snapshot, you MUST reuse it:
       * encode the constant in the `condition_on` edge (e.g., `val(<selector_id>) == 15`),
       * and the variant's `bit_start`/children MUST start AFTER the selector field (do not re-parse the selector again).
       * returning a variant that contains its own local `OpCode`/`Type` child in this case is INVALID; return `{{}}` instead if you cannot safely link to the shared selector.
    - If the spec defines a SECONDARY selector (e.g., Sub-Code / Subcommand):
       * you MAY create an additional selector node, but place it INSIDE the correct parent variant (usually after the primary selector),
       * then create nested sub-variants under that parent variant controlled by the secondary selector (one selector per sibling variant-group).
   - Same-direction variants controlled by one selector (e.g., multiple response layouts) should normally share the SAME `bit_start` immediately after the shared selector. Do NOT serialize them by shifting later variants after earlier ones just to avoid overlap.

## CRITICAL: CONTEXT LINKING STRATEGY
The "Current sub-tree snapshot" contains **Selectors** (Opcodes, Message Types) defined in previous batches.
When the focused documentation in THIS batch describes a specific **Variant** (e.g., "Response for Opcode 0x01"):
1. **SEARCH** the `Current sub-tree snapshot` for the existing Selector node ID (e.g., the "Opcode" field).
2. **CREATE** a `condition_on` edge where:
   - `src`: The ID of the EXISTING Selector node (from the snapshot).
   - `dst`: The ID of your NEW Variant node (being created now).
   - `formula`: The value constraint (e.g., `val(src_id) == 1`).
   - `message_type`: "request" or "response" as appropriate.
**DO NOT** create a new duplicate Selector node. You MUST link to the existing one.
If multiple new variants in this batch belong to the same selector family, they MUST all route from that ONE shared selector node.

	Each new node MUST include a "source" pointing to the exact section (e.g., `file:number` or section title).
	If the focused sections contain **no new parseable structure**, return either a **small corrective patch** (e.g., add `length_of`, fix illegal expressions) or **empty patch (`{{}}`)**.
	If the source-backed structure gap detector lists any gap above, this stop-early rule does NOT apply; `{{}}` / `no_change` is invalid until at least one listed gap is reduced.

# HARD LIMITS (do not exceed)
- new_nodes ≤ 20
- edge_adds ≤ 20
- Any NEW node_id MUST be within [{reserve_start}, {reserve_end}].

# INVARIANTS (strict)
1) Single root; do NOT create a second root.
2) Positioning: `bit_start = parent.bit_start + offset_within_parent`; no forward/cross-container/self references.
3) Sizing: numeric-only arithmetic; variable-size fields pair with a **`length_of`** edge (avoid boolean in `size_bits`).
4) Selector/Variant: `condition_on` edges from a selector MUST target `node_type="variant"` nodes only (do NOT attach per-field `condition_on` leaves). Variants exclude shared headers/selectors; each variant MUST have exactly one `condition_on` driven by its selector.
4a) INVALID PATCH PATTERN: adding siblings `Variant_A`, `Variant_B`, ... and also adding local selector children inside each of those variants (e.g., each variant contains its own `Opcode`). Reuse the shared selector from the snapshot instead.
5) Every edge MUST include `message_type` and numeric `src`/`dst`.
6) Patch only: NEVER return a full `protocol_tree`.
7) Boolean OR must be explicit: use `or` and repeat the full comparison each time (e.g., `val(7)==1 or val(7)==2 or val(7)==4 or val(7)==15`). Do NOT use `||` or shorthand like `val(7)==1 || 2`.
8) Keep formula syntax aligned with validator/interpreter/Z3 (STRICT): allowed identifiers val(<ID>), <ID>.size_bits, <ID>.bit_start; operators + - * / ( ) and/or/not with == != >= > <= < only (NO &&, ||, &, |); no chained comparisons—write `val(x) >= 1 and val(x) <= 4`; size_bits is arithmetic-only; condition_on is selector-only (no message_type inside); OR must be fully expanded (GOOD: `val(7)==1 or val(7)==2 or val(7)==4 or val(7)==15`; BAD: `val(7)==1 || 2 || 4 || 15`).
9) No self-loop edges: never create edges where src == dst (length_of/condition_on or any other rel).
10) Each variant must have exactly ONE condition_on from a single selector; do not attach multiple selectors or duplicate formulas to the same variant. Formulas across variants must be mutually exclusive. If you cannot give one clean condition_on, return {{}} instead of emitting multiple edges.

# OUTPUT — STRICT JSON PATCH (NOT a full tree)
{{ "patch_metadata": {{"intent":"expand","scope":"atomic"}},
   "new_nodes":[...], "node_updates":[...], "edge_adds":[...],
   "edge_updates":[...], "edge_removes":[...], "nodes_to_remove":[...], "validation_notes":[...] }}
"""
            direction_rules = """
# DIRECTIONALITY RULES (STRICT, REFINE-ONLY)

## A. Variants & edges must carry direction
- Every *variant* node must set `message_type` to **"request"** or **"response"**. Shared selectors/headers/length fields can remain "bidirectional".
- Never embed direction checks inside `condition_on` formulas. Formulas may reference selector values only (e.g., `val(SEL_ID) == 2`, `val(SEL_ID) in {1,2}`).

## B. Condition edges must inherit direction from dst
- For every `condition_on` edge, set `message_type` to exactly match the destination variant’s `message_type`.
  - If `dst.message_type == "request"`, the edge must also be `"request"`.
  - If `dst.message_type == "response"`, the edge must be `"response"`.
  - Only when `dst` itself is `"bidirectional"` may the edge be `"bidirectional"`.
- Do **not** use a single `"bidirectional"` `condition_on` to cover both request and response variants. Split them into two directional edges.

## C. Dedup & mutual exclusivity
- Do not emit duplicate edges with the same `src` and `formula` pointing to different variants. Distinguish them via direction or adjust the selector predicates.
- Variant formulas must be disjoint and collectively cover the selector domain; overlapping predicates are rejected.

## D. Examples
GOOD:
  {"src": 6, "dst": 8,  "rel": "condition_on", "formula": "val(6) == 2", "message_type": "request"}
  {"src": 6, "dst": 11, "rel": "condition_on", "formula": "val(6) == 2", "message_type": "response"}

BAD:
  {"src": 6, "dst": 8, "rel": "condition_on", "formula": "val(6)==2 and message_type=='request'", "message_type": "bidirectional"}
  {"src": 6, "dst": 9, "rel": "condition_on", "formula": "val(6) == 2", "message_type": "bidirectional"}  # if dst is request-only, edge must be request
  {"node_id": 201, "name": "Write_Single_Coil_Response", "node_type": "variant", "children_ids": [202,203,204]}
  {"node_id": 202, "name": "Opcode", "node_type": "selector", "parent_id": 201}  # BAD: local shadow selector inside variant
"""
            instruction_block = instruction_block + "\n" + direction_rules
            prompt = context_block + "\n" + instruction_block
            patch_system_content = (
                "You are a protocol tree patching assistant. You MUST return an incremental JSON patch only — "
                "no markdown, no comments. If the user prompt lists source-backed structure gaps, you MUST "
                "produce a minimal patch that reduces at least one listed gap and MUST NOT return {} or no_change."
                if has_source_structure_gaps
                else (
                    "You are a protocol tree patching assistant. You MUST return an incremental JSON patch only — "
                    "no markdown, no comments. If no safe improvement can be made, return {} exactly."
                )
            )
            if self.provider == "openai":
                payload = {
                    "model": self.default_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": patch_system_content,
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                }
                if self.default_model.startswith(("gpt-5", "o1", "o3", "o4")):
                    payload["max_completion_tokens"] = 16384
                else:
                    # Local/OpenAI-compatible models use the standard token
                    # parameter and must not receive GPT-only request fields.
                    payload["max_tokens"] = self.max_tokens
                structured_format = self._patch_response_format()
                if structured_format:
                    payload["response_format"] = structured_format
            elif self.provider == "deepseek":
                logger.info(f"refiment prompt: {prompt[:500]}...")
                payload = {
                    "model": self.default_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": patch_system_content,
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self._clamp_deepseek_max_tokens(self.max_tokens, self.default_model),
                    # The configured refine temperature (--stage-b-temperature); 0.0 when unset.
                    "temperature": self.temperature,
                    "stream": False,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
                if self._deepseek_patch_tool_use_enabled():
                    self._attach_openai_compatible_patch_tool(payload)
                else:
                    structured_format = self._patch_response_format()
                    if structured_format:
                        payload["response_format"] = structured_format
            else:
                claude_system_content = (
                    "You are a network protocol parser synthesizer. FOCUS: Model ONLY binary packet structures that appear on the network wire. EXCLUDE: Configuration parameters, socket options, implementation details, IP addresses, buffer sizes, timeouts, or API settings. INCLUDE: Headers, fields with bit_start positions, payload structures, selector values, data lengths, checksums, and conditional packet variants. Refine the tree using only sections that describe parseable binary packet formats."
                )
                if has_source_structure_gaps:
                    claude_system_content += (
                        " If the user prompt lists source-backed structure gaps, return a minimal JSON patch that "
                        "reduces at least one listed gap; do not return an empty patch or no_change."
                    )
                payload = {
                    "model": self.default_model,
                    "system": claude_system_content,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self._clamp_anthropic_max_tokens(self.max_tokens, self.default_model),
                }
                self._attach_anthropic_patch_tool(payload)
            max_json_retries = _env_positive_int("STEP2_REFINE_MAX_ATTEMPTS", 3)
            json_retry_count = 0
            tree_parsed = False
            fallback_used = False
            deepseek_tool_fallback_used = False

            def _handle_refine_json_failure(exc: json.JSONDecodeError, reason: str) -> str:
                nonlocal json_retry_count, fallback_used
                logger.warning(
                    "JSON decode failed on batch %d attempt %d: %s",
                    batch_idx + 1,
                    json_retry_count + 1,
                    exc,
                )
                json_retry_count += 1
                if json_retry_count >= max_json_retries:
                    if (
                        self.provider == "deepseek"
                        and self.allow_deepseek_reasoner_fallback
                        and not fallback_used
                        and payload.get("model") != "deepseek-reasoner"
                    ):
                        fallback_used = True
                        json_retry_count = 0
                        payload["model"] = "deepseek-reasoner"
                        logger.warning(
                            "Switching to deepseek-reasoner after repeated %s refine responses",
                            reason,
                        )
                        return "retry"
                    if not _env_flag("STEP2_SOURCE_GAP_FAIL_ON_EXHAUST", "1"):
                        # Same rule as source gaps under the relaxed profile: a
                        # reproducible run must finish. Keep the cumulative tree,
                        # record the batch, and treat the answer as an empty patch.
                        logger.warning(
                            "Giving up on refine batch %d after %d %s responses; keeping the previous tree.",
                            batch_idx + 1,
                            json_retry_count,
                            reason,
                        )
                        self.unresolved_source_gaps.append({
                            "batch": batch_idx + 1,
                            "reason": f"refine responses stayed {reason} after the retry budget",
                        })
                        return "give_up"
                    raise exc
                logger.warning("Retrying batch %d (attempt %d)", batch_idx + 1, json_retry_count + 1)
                return "retry"

            while json_retry_count < max_json_retries and not tree_parsed:
                try:
                    result = self._call_api_with_retry(payload)
                except Exception as api_exc:
                    api_exc_type = type(api_exc).__name__
                    api_exc_text = str(api_exc).lower()
                    is_transient_provider_failure = isinstance(api_exc, TimeoutError) or any(
                        marker in api_exc_text
                        for marker in (
                            "timeout",
                            "timed out",
                            "connection",
                            "connectionpool",
                            "name resolution",
                            "dns",
                            "temporarily unavailable",
                            "remote disconnected",
                        )
                    )
                    if (
                        self.provider == "deepseek"
                        and payload.get("tools")
                        and not deepseek_tool_fallback_used
                        and not is_transient_provider_failure
                    ):
                        deepseek_tool_fallback_used = True
                        logger.warning(
                            "DeepSeek patch tool-use call failed (%s); falling back to JSON response_format.",
                            api_exc_type,
                        )
                        self._disable_openai_compatible_patch_tool(payload)
                        continue
                    provider_failure_gaps = (
                        prompt_missing_packet_fields
                        + prompt_missing_selector_obligations
                    )
                    provider_failure_hard_gaps = self._source_gaps_for_hard_gate(
                        cumulative_tree,
                        focused_secs,
                        provider_failure_gaps,
                        traffic_file=traffic_file,
                        stage_label=f"refine_batch_{batch_idx + 1}_provider_failure",
                    )
                    if provider_failure_hard_gaps:
                        try:
                            self._save_to_cache(
                                cache_filename,
                                {
                                    "protocol_tree": cumulative_tree,
                                    "timestamp": datetime.now().isoformat(),
                                    "stage_label": f"refine_batch_{batch_idx + 1}",
                                    "source": "refine_provider_failure_source_gap_fail",
                                    "failure_reason": "provider_failure_with_source_backed_gaps",
                                    "skip_error_type": api_exc_type,
                                    "source_backed_gaps": provider_failure_hard_gaps[:50],
                                },
                            )
                        except Exception as save_exc:
                            logger.warning(
                                "Failed to persist provider-failure source-gap failure for batch %s: %s",
                                batch_idx + 1,
                                save_exc,
                            )
                        raise RuntimeError(
                            f"Refine batch {batch_idx + 1}/{batch_count} provider call failed "
                            f"while source-backed gaps remain ({api_exc_type}); "
                            "failing instead of deferring source-backed gaps: "
                            + ", ".join(provider_failure_hard_gaps[:12])
                        ) from api_exc
                    if _provider_failure_fail_fast_enabled():
                        raise RuntimeError(
                            f"Refine batch {batch_idx + 1}/{batch_count} provider call failed "
                            f"after configured retries ({api_exc_type}); fail-fast is enabled."
                        ) from api_exc
                    logger.warning(
                        "Skipping refine batch %d/%d after provider call failed after configured retries (%s); "
                        "retaining current cumulative tree and continuing.",
                        batch_idx + 1,
                        batch_count,
                        api_exc_type,
                    )
                    is_last_batch = (batch_idx == batch_count - 1)
                    cumulative_tree = self._inline_validate_and_fix(
                        cumulative_tree,
                        sections,
                        raw_sections,
                        cache_filename,
                        f"refine_batch_{batch_idx + 1}",
                        force_full_fix=is_last_batch,
                        traffic_file=traffic_file,
                    )
                    try:
                        self._save_to_cache(
                            cache_filename,
                            {
                                "protocol_tree": cumulative_tree,
                                "timestamp": datetime.now().isoformat(),
                                "stage_label": f"refine_batch_{batch_idx + 1}",
                                "source": "refine_provider_failure_skip",
                                "skip_reason": "provider_failure_after_configured_retries",
                                "skip_error_type": api_exc_type,
                            },
                        )
                    except Exception as save_exc:
                        logger.warning(
                            "Failed to persist provider-failure refine skip for batch %s: %s",
                            batch_idx + 1,
                            save_exc,
                        )
                    _maybe_dynamic_traffic_anchor_locate(
                        batch_idx,
                        f"provider-failure skipped refine batch {batch_idx + 1}",
                    )
                    tree_parsed = True
                    logger.info(
                        "✓ Batch %d/%d skipped after provider failure; cumulative tree remains at %d nodes",
                        batch_idx + 1,
                        batch_count,
                        len(cumulative_tree.get("nodes", []) or []),
                    )
                    break
                response_meta = self._extract_response_metadata(result)
                self._append_usage_log(
                    stage="refine",
                    meta=response_meta,
                    extra={"section_info": f"Section {start_idx} to {end_idx}", "json_retry_count": json_retry_count},
                )
                raw_response, response_kind = self._extract_refine_raw_response(result)

                self._save_to_cache(f"refine_section_{start_idx}_to_{end_idx}_ai_response.json", {
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "response_kind": response_kind,
                    "timestamp": datetime.now().isoformat(),
                    "section_info": f"Section {start_idx} to {end_idx}",
                    "json_retry_count": json_retry_count,
                    "provider": response_meta.get("provider"),
                    "model": response_meta.get("model"),
                    "stop_reason": response_meta.get("stop_reason"),
                    "finish_reason": response_meta.get("finish_reason"),
                    "usage": response_meta.get("usage"),
                    "token_usage": response_meta.get("token_usage"),
                })

                if not raw_response or not raw_response.strip():
                    exc = json.JSONDecodeError("Empty response", raw_response or "", 0)
                    action = _handle_refine_json_failure(exc, "empty")
                    if action == "retry":
                        continue
                    raw_response = "{}"

                logger.info(f"Raw response length: {len(raw_response)}, first 100 chars: {raw_response[:100]}")

                clean_response = self._clean_raw_response(raw_response)
                if not clean_response.strip():
                    exc = json.JSONDecodeError("Empty response", raw_response, 0)
                    action = _handle_refine_json_failure(exc, "cleaned-empty")
                    if action == "retry":
                        continue
                    clean_response = "{}"

                try:
                    response_payload = json.loads(clean_response)
                except json.JSONDecodeError as exc:
                    action = _handle_refine_json_failure(exc, "invalid")
                    if action == "retry":
                        continue
                    response_payload = {}

                def _retry_source_backed_gap_failure(gaps: Sequence[str], reason: str) -> bool:
                    nonlocal json_retry_count
                    stage_reason = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(reason or "source_gap")).strip("_")
                    hard_gaps = self._source_gaps_for_hard_gate(
                        cumulative_tree,
                        focused_batch_sections,
                        gaps,
                        traffic_file=traffic_file,
                        stage_label=f"refine_batch_{batch_idx + 1}_{stage_reason}_{json_retry_count}",
                    )
                    if not hard_gaps:
                        logger.info(
                            "Source-gap adjudication waived refine batch %d/%d %s retry/fail for detector gaps: %s",
                            batch_idx + 1,
                            batch_count,
                            reason,
                            list(gaps or [])[:12],
                        )
                        return False
                    gap_text = ", ".join(str(gap) for gap in hard_gaps[:12])
                    if payload.get("messages"):
                        payload["messages"][-1]["content"] = (
                            prompt
                            + "\n\n# NOTE\nYour previous patch was "
                            + reason
                            + " while source-backed structure gaps remain: "
                            + gap_text
                            + ". Retry with a minimal doc-grounded patch that reduces at least one listed gap. "
                              "Do not defer these fields to traffic-anchor or coverage-obligation repair."
                        )
                    json_retry_count += 1
                    if json_retry_count < max_json_retries:
                        logger.warning(
                            "Retrying refine batch %d/%d after %s while source-backed gaps remain: %s",
                            batch_idx + 1,
                            batch_count,
                            reason,
                            gap_text,
                        )
                        return True
                    message = (
                        "Refine LLM returned " + reason
                        + " while source-backed structure gaps remain: " + gap_text
                    )
                    if _env_flag("STEP2_SOURCE_GAP_FAIL_ON_EXHAUST", "1"):
                        raise ValueError(message)
                    # A reproducible run must finish. Keep the cumulative tree, leave
                    # the gap on record for the checkpoint and dashboard, and let the
                    # caller move to the next batch.
                    logger.warning("Deferring source gap after retry budget: %s", message)
                    self.unresolved_source_gaps.append({
                        "batch": batch_idx + 1,
                        "reason": reason,
                        "gaps": [str(g) for g in hard_gaps[:24]],
                        "retries": json_retry_count,
                    })
                    return False

                patch_payload = None
                refined_tree = None
                response_payload = _refine_response_payload_object(response_payload)
                if response_payload is None:
                    action = _handle_refine_json_failure(
                        ValueError("LLM response must be a JSON object containing either protocol_tree or patch data"),
                        "non-object",
                    )
                    if action == "retry":
                        continue
                    response_payload = {}

                has_tree = "protocol_tree" in response_payload and isinstance(response_payload["protocol_tree"], dict)
                has_patch = "patch" in response_payload and isinstance(response_payload["patch"], dict)

                if has_tree and not has_patch:
                    logger.warning(
                        "Refine batch %s returned a complete protocol_tree despite patch-only instructions; retrying.",
                        batch_idx + 1,
                    )
                    json_retry_count += 1
                    if json_retry_count >= max_json_retries:
                        if (
                            self.provider == "deepseek"
                            and self.allow_deepseek_reasoner_fallback
                            and not fallback_used
                            and payload.get("model") != "deepseek-reasoner"
                        ):
                            fallback_used = True
                            json_retry_count = 0
                            payload["model"] = "deepseek-reasoner"
                            logger.warning(
                                "Switching to deepseek-reasoner after repeated full-tree refine responses"
                            )
                            continue
                        raise ValueError("Refine LLM kept returning full protocol_tree instead of patch.")
                    logger.warning(
                        "Retrying batch %d (attempt %d) after full-tree response",
                        batch_idx + 1,
                        json_retry_count + 1,
                    )
                    continue

                if has_patch:
                    patch_payload = response_payload["patch"]
                elif not has_tree:
                    patch_payload = response_payload
                else:
                    refined_tree = response_payload["protocol_tree"]

                if patch_payload is None:
                    raise ValueError("Refine response missing patch payload after validation.")

                patch_payload = self._clip_patch_budget(patch_payload)
                focused_batch_sections = focused_secs
                missing_packet_fields = _missing_packet_format_names(cumulative_tree, focused_batch_sections)
                missing_selector_obligations = _missing_selector_obligation_names(
                    cumulative_tree,
                    focused_batch_sections,
                )
                source_structure_gaps = (
                    missing_packet_fields
                    + missing_selector_obligations
                )
                force_retain_refined_tree = False
                if _is_empty_refine_patch(patch_payload) and source_structure_gaps:
                    if _should_fail_source_backed_refine_gap(source_structure_gaps, traffic_file=traffic_file):
                        if _retry_source_backed_gap_failure(source_structure_gaps, "empty patch"):
                            continue
                        refined_tree = cumulative_tree
                        force_retain_refined_tree = True
                    else:
                        logger.warning(
                            "Skipping empty refine patch without LLM retry; deferring source-backed gaps for later "
                            "coverage/obligation repair: %s",
                            source_structure_gaps[:12],
                        )
                        refined_tree = cumulative_tree
                        force_retain_refined_tree = True
                invalid_patch_reason = (
                    None
                    if force_retain_refined_tree
                    else self._detect_invalid_refine_patch(cumulative_tree, patch_payload)
                )
                if invalid_patch_reason:
                    if payload.get("messages"):
                        payload["messages"][-1]["content"] = (
                            prompt
                            + "\n\n# NOTE\nYour previous patch was rejected because it duplicated selector routing in an invalid way: "
                            + invalid_patch_reason
                            + ". Retry with ONE shared selector at the common parent layer; do not create local Opcode/Type children inside each variant, and do not serialize mutually-exclusive variants by shifting bit_start."
                        )
                    json_retry_count += 1
                    if json_retry_count < max_json_retries:
                        continue
                    logger.warning("Rejecting invalid refine patch (selector/variant anti-pattern): %s", invalid_patch_reason)
                    refined_tree = cumulative_tree
                elif force_retain_refined_tree:
                    pass
                elif not self._is_patch_doc_grounded(patch_payload, focused_batch_sections):
                    if payload.get("messages"):
                        payload["messages"][-1]["content"] = (
                            prompt
                            + "\n\n# NOTE\nYour previous patch added nodes without valid 'source' pointing to the focused sections. "
                              "Retry: for EACH new node, copy a 'source' value from the Focused source labels block of THIS batch. "
                              "Do not use generic values like 'doc'."
                        )
                    json_retry_count += 1
                    if json_retry_count < max_json_retries:
                        continue
                    hard_source_structure_gaps = self._source_gaps_for_hard_gate(
                        cumulative_tree,
                        focused_batch_sections,
                        source_structure_gaps,
                        traffic_file=traffic_file,
                        stage_label=f"refine_batch_{batch_idx + 1}_ungrounded_patch_{json_retry_count}",
                    )
                    if hard_source_structure_gaps:
                        message = (
                            "Refine LLM returned only ungrounded patches while source-backed structure gaps remain: "
                            + ", ".join(hard_source_structure_gaps[:12])
                        )
                        if _env_flag("STEP2_SOURCE_GAP_FAIL_ON_EXHAUST", "1"):
                            raise ValueError(message)
                        logger.warning("Deferring source gap after retry budget: %s", message)
                        self.unresolved_source_gaps.append({
                            "batch": batch_idx + 1,
                            "reason": "ungrounded patch",
                            "gaps": [str(g) for g in hard_source_structure_gaps[:24]],
                            "retries": json_retry_count,
                        })
                    logger.warning("Rejecting ungrounded expansion (no valid sources). Skipping this batch.")
                    refined_tree = cumulative_tree
                else:
                    logger.info("Applying refine patch response (keys=%s)", list(patch_payload.keys()))
                    try:
                        candidate_tree = fix_apply_patch(cumulative_tree, patch_payload)
                    except Exception as patch_exc:
                        if payload.get("messages"):
                            payload["messages"][-1]["content"] = (
                                prompt
                                + "\n\n# NOTE\nYour previous patch could not be applied: "
                                + str(patch_exc)
                                + ". Retry with a minimal patch that references only node IDs present in the current "
                                  "snapshot or node IDs introduced in the same patch. Do not update children_ids to "
                                  "include unknown, removed, or never-created nodes."
                            )
                        json_retry_count += 1
                        if json_retry_count < max_json_retries:
                            continue
                        logger.warning(
                            "Rejecting refine patch after apply failure on batch %s: %s; retaining previous cumulative tree.",
                            batch_idx + 1,
                            patch_exc,
                        )
                        refined_tree = cumulative_tree
                    else:
                        packet_gap_still_open = bool(missing_packet_fields) and not _packet_format_coverage_improved(
                            cumulative_tree,
                            candidate_tree,
                            focused_batch_sections,
                        )
                        selector_gap_still_open = not _selector_obligation_coverage_improved(
                            cumulative_tree,
                            candidate_tree,
                            focused_batch_sections,
                        )
                        if packet_gap_still_open or selector_gap_still_open:
                            candidate_gaps = (
                                _missing_packet_format_names(candidate_tree, focused_batch_sections)
                                + _missing_selector_obligation_names(candidate_tree, focused_batch_sections)
                            )
                            if _should_fail_source_backed_refine_gap(candidate_gaps, traffic_file=traffic_file):
                                if _retry_source_backed_gap_failure(candidate_gaps, "no source-backed progress"):
                                    continue
                                refined_tree = candidate_tree
                            else:
                                logger.warning(
                                    "Skipping no-progress refine patch without LLM retry; deferring source-backed gaps "
                                    "for later coverage/obligation repair: %s",
                                    candidate_gaps[:12],
                                )
                                refined_tree = cumulative_tree
                        else:
                            refined_tree = candidate_tree

                if traffic_file and _env_flag("STEP2_TRAFFIC_ANCHOR_GATE", "1"):
                    before_anchor_records = current_anchor_records or _build_traffic_anchor_records(
                        cumulative_tree,
                        traffic_file,
                        max_packets=_env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 1),
                    )
                    if before_anchor_records:
                        after_anchor_records = _build_traffic_anchor_records(
                            refined_tree,
                            traffic_file,
                            max_packets=_env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 1),
                        )
                        anchor_regression = _traffic_anchor_regression_reason(
                            before_anchor_records,
                            after_anchor_records,
                        )
                        if anchor_regression:
                            if payload.get("messages"):
                                payload["messages"][-1]["content"] = (
                                    prompt
                                    + "\n\n# NOTE\nYour previous patch broke observed traffic parseability: "
                                    + anchor_regression
                                    + ". Retry with a patch that keeps the observed packet parseable while adding "
                                      "the doc-derived branch or fields."
                                )
                            json_retry_count += 1
                            if json_retry_count < max_json_retries:
                                continue
                            logger.warning(
                                "Rejecting refine patch for batch %s after observed traffic anchor regression: %s",
                                batch_idx + 1,
                                anchor_regression,
                            )
                            refined_tree = cumulative_tree
                        elif any(
                            isinstance(record, dict) and record.get("success")
                            for record in before_anchor_records
                        ):
                            logger.info(
                                "Traffic-anchor gate accepted refine batch %s without observed-path regression.",
                                batch_idx + 1,
                            )

                node_count = len(refined_tree.get("nodes", []))
                try:
                    merge_threshold = int(os.getenv("STEP2_NODE_MERGE_THRESHOLD", "0").strip())
                except Exception:
                    merge_threshold = 0
                merge_threshold = max(0, merge_threshold)
                if merge_threshold and node_count > merge_threshold:
                    logger.warning(
                        "Refined tree has %d nodes (>%d), applying node merging strategy",
                        node_count,
                        merge_threshold,
                    )
                    refined_tree = self._merge_similar_nodes(refined_tree)
                    node_count = len(refined_tree.get("nodes", []))
                    logger.info("After merging: %d nodes", node_count)

                try:
                    max_error_increase = int(os.getenv("STEP2_REFINEMENT_MAX_HARD_ERROR_INCREASE", "0").strip())
                except Exception:
                    max_error_increase = 0
                try:
                    max_error_multiplier = float(
                        os.getenv("STEP2_REFINEMENT_MAX_HARD_ERROR_MULTIPLIER", "5").strip()
                    )
                except Exception:
                    max_error_multiplier = 5.0
                try:
                    min_node_retention_ratio = float(
                        os.getenv("STEP2_REFINEMENT_MIN_NODE_RETENTION_RATIO", "0.7").strip()
                    )
                except Exception:
                    min_node_retention_ratio = 0.7
                max_error_increase = max(0, max_error_increase)
                max_error_multiplier = max(1.0, max_error_multiplier)
                min_node_retention_ratio = min(1.0, max(0.0, min_node_retention_ratio))
                try:
                    prev_nodes = len(cumulative_tree.get("nodes", []) or [])
                    candidate_nodes = len(refined_tree.get("nodes", []) or [])
                    min_candidate_nodes = int(prev_nodes * min_node_retention_ratio)
                    if prev_nodes > 0 and candidate_nodes < min_candidate_nodes:
                        logger.warning(
                            "Rejecting refine patch for batch %s: node count %d -> %d below retention threshold %d; "
                            "retaining previous cumulative tree.",
                            batch_idx + 1,
                            prev_nodes,
                            candidate_nodes,
                            min_candidate_nodes,
                        )
                        refined_tree = cumulative_tree
                        node_count = len(refined_tree.get("nodes", []))
                        raise RuntimeError("refine patch rejected by node retention gate")
                    prev_report = validate_protocol_tree(json.dumps({"protocol_tree": cumulative_tree}))
                    candidate_report = validate_protocol_tree(json.dumps({"protocol_tree": refined_tree}))
                    prev_errors = len(getattr(prev_report, "errors", []) or [])
                    candidate_errors = len(getattr(candidate_report, "errors", []) or [])
                    allowed_errors = prev_errors + max_error_increase
                    if prev_errors > 0:
                        allowed_errors = max(allowed_errors, int(prev_errors * max_error_multiplier))
                    if candidate_errors > allowed_errors:
                        repaired_tree, repaired_errors, repair_attempted = self._try_repair_regressed_refine_candidate(
                            refined_tree,
                            sections=sections,
                            raw_sections=raw_sections,
                            cache_filename=cache_filename,
                            stage_label=f"refine_batch_{batch_idx + 1}",
                            candidate_report=candidate_report,
                            allowed_errors=allowed_errors,
                        )
                        if repair_attempted and repaired_errors <= allowed_errors:
                            refined_tree = repaired_tree
                            node_count = len(refined_tree.get("nodes", []))
                            logger.info(
                                "Accepted repaired refine candidate for batch %s: hard_errors %d -> %d within allowed %d.",
                                batch_idx + 1,
                                candidate_errors,
                                repaired_errors,
                                allowed_errors,
                            )
                        else:
                            rejected_codes = _hard_error_codes(candidate_report)
                            logger.warning(
                                "Rejecting refine patch for batch %s: hard_errors %d -> %d exceeds allowed %d "
                                "(repair_attempted=%s, codes=%s); retaining previous cumulative tree.",
                                batch_idx + 1,
                                prev_errors,
                                candidate_errors,
                                allowed_errors,
                                repair_attempted,
                                rejected_codes,
                            )
                            self.refine_rejections.append({
                                "batch": batch_idx + 1,
                                "prev_errors": prev_errors,
                                "candidate_errors": candidate_errors,
                                "allowed_errors": allowed_errors,
                                "repair_attempted": bool(repair_attempted),
                                "repaired_errors": repaired_errors if repair_attempted else None,
                                "candidate_nodes": len(refined_tree.get("nodes", []) or []),
                                "hard_error_codes": rejected_codes,
                            })
                            refined_tree = cumulative_tree
                            node_count = len(refined_tree.get("nodes", []))
                except RuntimeError as gate_exc:
                    if str(gate_exc) != "refine patch rejected by node retention gate":
                        logger.warning("Refine patch regression gate failed for batch %s: %s", batch_idx + 1, gate_exc)
                except Exception as gate_exc:
                    logger.warning("Refine patch regression gate failed for batch %s: %s", batch_idx + 1, gate_exc)

                cumulative_tree = refined_tree
                is_last_batch = (batch_idx == batch_count - 1)
                cumulative_tree = self._inline_validate_and_fix(
                    cumulative_tree,
                    sections,
                    raw_sections,
                    cache_filename,
                    f"refine_batch_{batch_idx + 1}",
                    force_full_fix=is_last_batch,
                    traffic_file=traffic_file
                )
                try:
                    self._save_to_cache(
                        cache_filename,
                        {
                            "protocol_tree": cumulative_tree,
                            "timestamp": datetime.now().isoformat(),
                            "stage_label": f"refine_batch_{batch_idx + 1}",
                            "source": "refine_patch",
                        },
                    )
                except Exception as save_exc:
                    logger.warning("Failed to persist refined tree for batch %s: %s", batch_idx + 1, save_exc)
                _maybe_dynamic_traffic_anchor_locate(batch_idx, f"refine batch {batch_idx + 1}")
                tree_parsed = True
                logger.info(f"✓ Batch {batch_idx+1}/{batch_count} AI response parsed successfully (attempt {json_retry_count + 1}), {node_count} nodes")
                logger.info(f"✓ Cumulative tree now has {len(cumulative_tree.get('nodes', []))} total nodes")

        self._persist_refinement_status(cumulative_tree, sections, "refine_complete")

        final_merged_tree = cumulative_tree
        # Patches are merged node-by-node and drop the top-level name, after which
        # _parse_tree_response falls back to "Unknown Protocol". Carry the caller's
        # name through so the output tree matches the input on this field.
        if isinstance(tree, dict) and isinstance(final_merged_tree, dict):
            input_name = str(tree.get("protocol_name") or "").strip()
            if input_name and input_name != "Unknown Protocol" and \
                    str(final_merged_tree.get("protocol_name") or "").strip() in {"", "Unknown Protocol"}:
                final_merged_tree = dict(final_merged_tree, protocol_name=input_name)

        # Optional variant splitting toggle
        enable_variants = os.getenv("GEN_ENABLE_MESSAGE_VARIANTS", "0") == "1"
        if enable_variants:
            final_merged_tree = add_request_response_variants(final_merged_tree)

        # Canonicalize into Core IR subset before caching/consuming downstream
        final_merged_tree = canonicalize_protocol_tree(final_merged_tree)

        final_validation: Dict[str, Any] = {}
        try:
            final_static_report = validate_protocol_tree(json.dumps({"protocol_tree": final_merged_tree}))
            final_validation["static"] = _validation_report_summary(final_static_report)
        except Exception as exc:
            final_validation["static"] = {"ok": False, "errors": [str(exc)]}

        if traffic_file:
            traffic_path = Path(str(traffic_file))
            if traffic_path.exists():
                try:
                    max_packets = max(1, int(os.getenv("STEP2_FINAL_TRAFFIC_MAX_PACKETS", "20")))
                except Exception:
                    max_packets = 20
                try:
                    final_hybrid_report = run_hybrid_validation(
                        final_merged_tree,
                        traffic_path=traffic_path,
                        max_packets=max_packets,
                    )
                    final_validation["hybrid"] = _validation_report_summary(final_hybrid_report)
                except Exception as exc:
                    final_validation["hybrid"] = {"ok": False, "errors": [str(exc)]}
            else:
                final_validation["hybrid"] = {
                    "ok": False,
                    "errors": [f"traffic file not found: {traffic_path}"],
                }

        final_output = {
            "protocol_tree": final_merged_tree,
            "timestamp": datetime.now().isoformat(),
            "input_signature": self.input_signature,
            "cache_schema": _STAGE2_INPUT_SIGNATURE_SCHEMA,
            "validation": final_validation,
            "merge_info": {
                "total_batches": len(batch_cache_files),
                "total_sections": num_sections,
                "batch_files": batch_cache_files,
                "cumulative_learning": True
            }
        }

        final_cache_path = os.path.join(self.cache_dir, "final_complete_protocol_tree.json")
        with open(final_cache_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info(f"FINAL COMPLETE PROTOCOL TREE SAVED: {final_cache_path}")
        logger.info(f"Total nodes: {len(final_merged_tree['nodes'])}")
        logger.info(f"Total edges: {len(final_merged_tree['edges'])}")
        logger.info("="*60)

        # Convert the final merged tree to a proper ProtocolTree object
        protocol_tree = self._parse_tree_response(final_merged_tree)
        return protocol_tree


    def _try_repair_regressed_refine_candidate(
        self,
        candidate_tree: Dict[str, Any],
        *,
        sections: Sequence[Dict[str, Any]],
        raw_sections: Optional[Sequence[Dict[str, Any]]],
        cache_filename: str,
        stage_label: str,
        candidate_report: Any,
        allowed_errors: int,
    ) -> Tuple[Dict[str, Any], int, bool]:
        """Give a hard-error-regressed refine candidate one static-fix chance before rollback."""

        candidate_errors = len(getattr(candidate_report, "errors", []) or [])
        if candidate_errors <= allowed_errors:
            return candidate_tree, candidate_errors, False
        if not _env_flag("STEP2_REFINEMENT_REPAIR_REGRESSED_CANDIDATE", "1"):
            return candidate_tree, candidate_errors, False

        max_calls = _env_positive_int("STEP2_REFINEMENT_CANDIDATE_FIX_MAX_CALLS", 4)
        fix_batch_size = _env_positive_int("STEP2_REFINEMENT_CANDIDATE_FIX_BATCH_SIZE", 5)
        repair_stage_label = f"{stage_label}_candidate_repair"
        logger.warning(
            "Refine candidate %s introduced hard_errors=%s above allowed=%s (codes=%s); attempting static "
            "candidate repair (max_llm_calls=%s).",
            stage_label,
            candidate_errors,
            allowed_errors,
            _hard_error_codes(candidate_report),
            max_calls,
        )

        # STEP2_INLINE_STATIC_FIX governs the per-batch inline pass, which the
        # budgeted workers switch off. This path exists precisely to spend a
        # bounded repair on a regressed candidate, so it must not inherit that
        # switch; otherwise the "repair" returns the candidate untouched.
        repaired_tree = self._inline_validate_and_fix(
            candidate_tree,
            sections,
            raw_sections,
            cache_filename,
            repair_stage_label,
            force_full_fix=False,
            traffic_file=None,
            fix_batch_size=fix_batch_size,
            max_calls=max_calls,
            force_static_fix=True,
        )
        try:
            repaired_report = validate_protocol_tree(json.dumps({"protocol_tree": repaired_tree}))
            repaired_errors = len(getattr(repaired_report, "errors", []) or [])
        except Exception as exc:
            logger.warning("Candidate repair validation failed for %s: %s", stage_label, exc)
            return candidate_tree, candidate_errors, True

        if repaired_errors <= allowed_errors:
            return repaired_tree, repaired_errors, True

        logger.warning(
            "Candidate repair for %s still exceeds hard-error gate: %s > allowed %s.",
            stage_label,
            repaired_errors,
            allowed_errors,
        )
        return candidate_tree, candidate_errors, True


    def _persist_refinement_status(self, tree, sections, stage_label):
        payload = {"stage_label": stage_label, "tree_sha256": _stable_json_hash(tree),
                   "remaining_packet_fields": _missing_packet_format_names(tree, sections),
                   "remaining_selector_obligations": _missing_selector_obligation_names(tree, sections),
                   "unresolved_source_gaps": self.unresolved_source_gaps,
                   "deferred_issues": self.deferred_issues}
        self._save_to_cache("source_gap_status.json", payload)

    def _inline_validate_and_fix(
        self,
        tree: Dict[str, Any],
        sections: Sequence[Dict[str, Any]],
        raw_sections: Optional[Sequence[Dict[str, Any]]],
        cache_filename: str,
        stage_label: str,
        force_full_fix: bool = False,
        traffic_file: Optional[str] = None,
        fix_batch_size: Optional[int] = None,
        max_calls: Optional[int] = None,
        force_static_fix: bool = False,
    ) -> Dict[str, Any]:
        """Run validator (and optional inline fix) after each refine batch.

        ``force_static_fix`` runs the fix agent even when STEP2_INLINE_STATIC_FIX is
        off; callers that have already decided to spend a repair use it.
        """

        incoming_tree = copy.deepcopy(tree)
        anchor_limit = _env_positive_int("STEP2_TRAFFIC_ANCHOR_MAX_PACKETS", 4)
        incoming_anchors = _build_traffic_anchor_records(incoming_tree, traffic_file, max_packets=anchor_limit) if traffic_file else []
        if traffic_file and not any(r.get("success") for r in incoming_anchors):
            logger.warning("Inline repair (%s) has no successful baseline anchors; traffic non-regression is unverified", stage_label)
        tree = copy.deepcopy(tree)
        from ..semantic_integrity import reconcile_expanded_placeholders
        tree, placeholder_changes = reconcile_expanded_placeholders(tree)
        if placeholder_changes:
            logger.info('Inline refinement route normalization: %s', placeholder_changes)
        try:
            sanitation_changes = normalize_parent_size_references(tree)
            if sanitation_changes:
                logger.info(
                    "Inline validation (%s) normalized parent-size references before validation: %s",
                    stage_label,
                    "; ".join(sanitation_changes[:20]),
                )
        except Exception as exc:
            logger.warning("Inline parent-size reference sanitation failed for %s: %s", stage_label, exc)
        try:
            if _env_flag("STEP2_PRUNE_UNROUTED_OBSERVED_VARIANTS", "1"):
                tree, observed_route_changes = _prune_or_guard_unrouted_observed_selector_variants(tree)
                if observed_route_changes:
                    logger.info(
                        "Inline validation (%s) normalized unguarded observed selector variants before validation: %s",
                        stage_label,
                        "; ".join(observed_route_changes[:20]),
                    )
        except Exception as exc:
            logger.warning("Inline observed selector variant sanitation failed for %s: %s", stage_label, exc)

        from ..validation.structure_repair import repair_redundant_ancestor_links
        tree, hierarchy_changes = repair_redundant_ancestor_links(tree)
        if hierarchy_changes:
            logger.info("Inline hierarchy repair (%s): %s", stage_label, hierarchy_changes)

        payload = {"protocol_tree": tree}
        report = validate_protocol_tree(json.dumps(payload))
        issue_count = len(getattr(report, "issues", {}) or {})
        error_count = len(report.errors)
        blocking_issue_count = _blocking_static_issue_count(report)
        logger.info(
            "Inline validation (%s): %s hard errors, %s total issues, %s routing semantic issues",
            stage_label,
            error_count,
            issue_count,
            blocking_issue_count,
        )
        snapshot_dir = Path(STEP2_FIX_CACHE_DIR) / "inline_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{stage_label}.json"
        snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        current_tree = tree
        current_report = report
        inline_cache_payload: Optional[Dict[str, Any]] = None
        inline_static_fix_enabled = os.getenv("STEP2_INLINE_STATIC_FIX", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        should_fix_static = (inline_static_fix_enabled or force_static_fix) and _should_run_static_fix(report)

        deferral_limit = _env_positive_int("STEP2_ISSUE_DEFER_AFTER", 3)
        error_issue_ids = _error_issue_ids(report)
        deferred_now = {iid for iid in error_issue_ids if self._issue_attempts.get(iid, 0) >= deferral_limit}
        if should_fix_static and error_issue_ids and error_issue_ids <= deferred_now:
            # Every remaining hard issue has already beaten the repair agent
            # deferral_limit times; spending another call on it buys nothing.
            logger.warning(
                "Inline validation (%s): all %d remaining hard issues are deferred after %d attempts each; skipping fix agent.",
                stage_label,
                len(deferred_now),
                deferral_limit,
            )
            should_fix_static = False

        if not report.ok and error_count == 0 and issue_count == 0:
            logger.info(
                "Inline validation (%s) returned ok=false without concrete issues; treating as pass.",
                stage_label,
            )
            should_fix_static = False

        if should_fix_static:
            logger.warning(
                "Inline validation failed for batch %s: hard_errors=%s total_issues=%s (sum=%s); invoking standard fix agent (syntax validator).",
                stage_label,
                error_count,
                issue_count,
                error_count + issue_count,
            )

            fix_batch_size = max(1, int(fix_batch_size if fix_batch_size is not None else os.getenv("INLINE_FIX_BATCH_SIZE", "5")))
            max_calls = max(1, int(max_calls if max_calls is not None else os.getenv("INLINE_FIX_MAX_CALLS", "20")))

            inline_fix_root = Path(STEP2_FIX_CACHE_DIR) / "inline_runs"
            inline_fix_dir = inline_fix_root / stage_label.replace("/", "_")
            inline_fix_dir.mkdir(parents=True, exist_ok=True)
            node_snapshot_dir = inline_fix_dir / "node_snapshots"
            node_snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Lazy import to avoid circular dependency at module import time
            from ..repair.agent import EnhancedPureAIAgent as FixLLMAgent
            repair = FixLLMAgent(api_key=self.api_key, provider=self.provider)
            repair.base_url = self.base_url
            repair.headers = dict(self.headers)
            repair.cache_dir = str(inline_fix_dir)

            current_tree = mcts_fix_tree(
                repair,
                tree,
                sections=sections,
                raw_sections=raw_sections or sections,
                batch_size=fix_batch_size,
                max_llm_calls=max_calls,
                node_snapshot_dir=str(node_snapshot_dir),
                validator_fn=run_full_validation,
                prompt_mode="fix",
            )

            current_report = run_full_validation({"protocol_tree": current_tree})
            remaining_blocking_issues = _blocking_static_issue_count(current_report)
            survivors = error_issue_ids & _error_issue_ids(current_report)
            for iid in sorted(survivors):
                self._issue_attempts[iid] = self._issue_attempts.get(iid, 0) + 1
                if self._issue_attempts[iid] == deferral_limit:
                    issue = (getattr(current_report, "issues", {}) or {}).get(iid)
                    description = str(getattr(issue, "description", "") or "")[:200]
                    self.deferred_issues.append({
                        "issue_id": iid,
                        "stage": stage_label,
                        "attempts": deferral_limit,
                        "description": description,
                    })
                    logger.warning(
                        "Deferring issue %s after %d failed repair attempts: %s",
                        iid, deferral_limit, description[:120],
                    )
            if current_report.ok and remaining_blocking_issues == 0:
                logger.info("Inline fix resolved validator issues for %s", stage_label)
            else:
                logger.warning(
                    "Inline fix still has hard_errors=%s routing_semantic_issues=%s total_error_issues=%s for batch %s — continuing with best-effort tree.",
                    len(current_report.errors),
                    remaining_blocking_issues,
                    len(getattr(current_report, "issues", {}) or {}),
                    stage_label,
                )

            inline_cache_payload = {
                "protocol_tree": current_tree,
                "timestamp": datetime.now().isoformat(),
                "source": "inline_fix",
                "stage_label": stage_label,
                "validation_ok": current_report.ok,
                "validation": {"static": _validation_report_summary(current_report)},
                "errors": current_report.errors,
                "issues": list(getattr(current_report, "issues", {}).keys()),
            }

        run_traffic = bool(traffic_file) and (
            force_full_fix
            or os.getenv("STEP2_INLINE_TRAFFIC_EVERY_BATCH", "0").strip().lower() in {"1", "true", "yes", "on"}
        )
        if run_traffic:
            traffic_path = Path(str(traffic_file))
            if traffic_path.exists():
                try:
                    max_packets = max(1, int(os.getenv("STEP2_INLINE_TRAFFIC_MAX_PACKETS", "5")))
                except Exception:
                    max_packets = 5
                hybrid_report = run_hybrid_validation(
                    current_tree,
                    traffic_path=traffic_path,
                    max_packets=max_packets,
                )
                logger.info(
                    "Inline traffic validation (%s): ok=%s traffic=%s/%s errors=%s",
                    stage_label,
                    hybrid_report.ok,
                    getattr(hybrid_report, "traffic_successful_samples", 0),
                    getattr(hybrid_report, "traffic_total_samples", 0),
                    len(getattr(hybrid_report, "errors", []) or []),
                )


                try:
                    traffic_fix_calls = max(0, int(os.getenv("INLINE_TRAFFIC_FIX_MAX_CALLS", "0")))
                except Exception:
                    traffic_fix_calls = 0
                if not hybrid_report.ok and traffic_fix_calls > 0:
                    inline_fix_root = Path(STEP2_FIX_CACHE_DIR) / "inline_runs"
                    inline_fix_dir = inline_fix_root / f"{stage_label.replace('/', '_')}_traffic"
                    inline_fix_dir.mkdir(parents=True, exist_ok=True)
                    node_snapshot_dir = inline_fix_dir / "node_snapshots"
                    node_snapshot_dir.mkdir(parents=True, exist_ok=True)

                    from ..repair.agent import EnhancedPureAIAgent as FixLLMAgent
                    repair = FixLLMAgent(api_key=self.api_key, provider=self.provider)
                    repair.base_url = self.base_url
                    repair.headers = dict(self.headers)
                    repair.cache_dir = str(inline_fix_dir)

                    def _hybrid_validator(candidate_tree: Dict[str, Any]) -> Any:
                        return run_hybrid_validation(
                            candidate_tree,
                            traffic_path=traffic_path,
                            max_packets=max_packets,
                        )

                    current_tree = mcts_fix_tree(
                        repair,
                        current_tree,
                        sections=sections,
                        raw_sections=raw_sections or sections,
                        batch_size=max(
                            1,
                            int(os.getenv("INLINE_TRAFFIC_FIX_BATCH_SIZE", "5"))
                            if str(os.getenv("INLINE_TRAFFIC_FIX_BATCH_SIZE", "5")).strip().isdigit()
                            else 5,
                        ),
                        max_llm_calls=traffic_fix_calls,
                        node_snapshot_dir=str(node_snapshot_dir),
                        validator_fn=_hybrid_validator,
                        prompt_mode="traffic_fix",
                    )
                    hybrid_report = _hybrid_validator(current_tree)

                inline_cache_payload = inline_cache_payload or {
                    "protocol_tree": current_tree,
                    "timestamp": datetime.now().isoformat(),
                    "source": "inline_validation",
                    "stage_label": stage_label,
                    "validation": {},
                }
                inline_cache_payload.setdefault("validation", {})
                inline_cache_payload["validation"]["hybrid"] = _validation_report_summary(hybrid_report)
                inline_cache_payload["validation_ok"] = bool(
                    inline_cache_payload.get("validation_ok", current_report.ok)
                ) and bool(hybrid_report.ok)
            else:
                logger.warning("Inline traffic validation requested but traffic file is missing: %s", traffic_path)

        if incoming_anchors:
            candidate_anchors = _build_traffic_anchor_records(current_tree, traffic_file, max_packets=anchor_limit)
            regression = _traffic_anchor_regression_reason(incoming_anchors, candidate_anchors)
            if regression:
                logger.warning("Rejecting inline repair (%s): %s; restoring incoming tree", stage_label, regression)
                self.deferred_issues.append({"stage": stage_label, "reason": "inline_traffic_regression", "description": regression})
                current_tree = incoming_tree
                current_report = run_full_validation({"protocol_tree": current_tree})
                inline_cache_payload = {"protocol_tree": current_tree, "source": "inline_repair_rollback",
                                        "stage_label": stage_label, "rejection_reason": regression,
                                        "validation_ok": current_report.ok,
                                        "validation": {"static": _validation_report_summary(current_report)}}
        self._persist_refinement_status(current_tree, sections, stage_label)
        if inline_cache_payload is not None:
            inline_cache_payload["protocol_tree"] = current_tree
            self._save_to_cache(cache_filename, inline_cache_payload)
            snapshot_path.write_text(
                json.dumps({"protocol_tree": current_tree}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return current_tree

    @staticmethod
    def _merge_full_tree(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge a newly returned full protocol_tree into the running cumulative tree
        so that prior batches are not discarded when an LLM returns a full tree.
        Preference is given to fields from the new tree; nodes/edges absent in the
        new tree are preserved from the base.
        """
        if not isinstance(base, dict):
            return copy.deepcopy(new)
        if not isinstance(new, dict):
            return copy.deepcopy(base)

        merged = copy.deepcopy(base)
        merged["root_node_id"] = new.get("root_node_id", merged.get("root_node_id"))

        # merge nodes by node_id
        base_nodes = {n.get("node_id"): n for n in merged.get("nodes", []) if isinstance(n, dict)}
        new_nodes = {n.get("node_id"): n for n in new.get("nodes", []) if isinstance(n, dict)}
        base_nodes.update(new_nodes)  # new overrides base on same id
        merged["nodes"] = list(base_nodes.values())

        # edges: if new provides edges, use them; otherwise keep base
        if isinstance(new.get("edges"), list) and new.get("edges"):
            merged["edges"] = copy.deepcopy(new["edges"])

        return merged


    def build_initial_tree(
        self,
        sections: List[Dict],
        protocol_hint: Optional[str] = None,
        traffic_file: Optional[Union[str, Path]] = None,
        traffic_locate_report: Optional[Dict[str, Any]] = None,
        traffic_locate_scope_sections: Optional[Sequence[Dict[str, Any]]] = None,
        request_only: bool = False,
    ) -> Optional[ProtocolTree]:
        """Build an initial tree or render the provider request.
        
        request_only returns the provider payload and bypasses tree generation."""

        force_rebuild = request_only or os.getenv("PARGEN_STEP2_FORCE_REBUILD", "").strip().lower() in {"1", "true", "yes", "on"}
        scope_sections = traffic_locate_scope_sections or sections

        def _cached_initial_scope_issue(tree_data: Any) -> Optional[str]:
            tree_payload = tree_data.get("protocol_tree", tree_data) if isinstance(tree_data, dict) else tree_data
            return _initial_tree_scope_rejection_reason(
                tree_payload,
                traffic_locate_report,
                sections=scope_sections,
            )

        if not force_rebuild:
            cached_initial = self._load_from_cache("initial_tree_ai_response.json")
            if cached_initial:
                if 'protocol_tree' in cached_initial:
                    logger.info("Using cached initial tree from initial_tree_ai_response.json")
                    tree_data = (cached_initial['protocol_tree'])
                    scope_issue = _cached_initial_scope_issue(tree_data)
                    if scope_issue:
                        logger.warning("Ignoring cached initial tree because scope validation failed: %s", scope_issue)
                    else:
                        return self._finalize_initial_tree(tree_data, _normalize_protocol_hint(protocol_hint))
                if 'raw_response' in cached_initial:
                    try:
                        cleaned = self._clean_raw_response(cached_initial['raw_response'])
                        tree_data = (json.loads(cleaned))
                        scope_issue = _cached_initial_scope_issue(tree_data)
                        if scope_issue:
                            logger.warning("Ignoring cached raw initial tree because scope validation failed: %s", scope_issue)
                        else:
                            return self._finalize_initial_tree(tree_data, _normalize_protocol_hint(protocol_hint))
                    except Exception as exc:
                        logger.warning(f"Failed to parse cached raw_response for initial tree: {exc}")
            cached_inline_initial = self._load_from_cache("inline_initial_tree.json")
            if cached_inline_initial and isinstance(cached_inline_initial.get("protocol_tree"), dict):
                logger.info("Using cached initial tree from inline_initial_tree.json")
                tree_data = (cached_inline_initial["protocol_tree"])
                scope_issue = _cached_initial_scope_issue(tree_data)
                if scope_issue:
                    logger.warning("Ignoring cached inline initial tree because scope validation failed: %s", scope_issue)
                else:
                    return self._finalize_initial_tree(tree_data, _normalize_protocol_hint(protocol_hint))

        # Fallback: reuse previously saved Stage B artifacts *within the same cache namespace*.
        # Cross-protocol reuse is intentionally avoided (see `_infer_step2_cache_namespace`).
        if not force_rebuild:
            cache_root = Path(self.cache_dir)
            fallback_candidates = [
                cache_root / "final_complete_protocol_tree.json",
                cache_root / "format_tree.json",
            ]
            if os.getenv("PARGEN_ALLOW_GLOBAL_STEP2_CACHE_FALLBACK", "0") == "1":
                fallback_candidates.extend(
                    [
                        STEP2_CACHE_DIR / "final_complete_protocol_tree.json",
                        STEP2_CACHE_DIR / "format_tree.json",
                    ]
                )
            for fb in fallback_candidates:
                try:
                    if fb.exists():
                        with fb.open("r", encoding="utf-8") as f:
                            logger.info(f"Using fallback initial tree from {fb}")
                            tree = json.load(f)
                            tree = (tree)
                            scope_issue = _cached_initial_scope_issue(tree)
                            if scope_issue:
                                logger.warning(
                                    "Ignoring fallback initial tree from %s because scope validation failed: %s",
                                    fb,
                                    scope_issue,
                                )
                                continue
                            return tree
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"Failed to load fallback initial tree from {fb}: {exc}")

        # Keep initial generation grounded in fixed-summary evidence without
        # dumping Stage-A audit metadata or oversized nested artifacts into the
        # provider call. Later refinement/scaffold/coverage passes still receive
        # the full fixed subset.
        enhanced_sections = _compact_initial_prompt_sections(sections)

        protocol_hint_text = _normalize_protocol_hint(protocol_hint)
        protocol_hint_block = ""
        if protocol_hint_text:
            protocol_hint_block = f"""
## TARGET PROTOCOL

- The target protocol family for this tree is **{protocol_hint_text}**.
- Even if the documentation context is empty or incomplete, build the best **protocol-specific**
  on-wire skeleton you can for **{protocol_hint_text}** rather than falling back to a generic
  placeholder protocol.
- Prefer standard field/control names that are well-established for **{protocol_hint_text}**
  (for example common header identifiers, control codes, request/response directions, and length
  fields) when they are part of the protocol's canonical wire format.
- Do NOT rename the protocol to a generic placeholder such as "Generic_Protocol_Message" or
  "Unknown Protocol" when the target protocol is known.
"""
        preinitial_traffic_context = _format_observed_traffic_context(traffic_file)
        preinitial_traffic_block = ""
        if preinitial_traffic_context:
            preinitial_traffic_block = f"""
{preinitial_traffic_context}

## PACKET-GROUNDED FULL-SUMMARY INITIAL REQUIREMENT

- There is no separate active-section selection step. Use the complete fixed documentation summary in `DOCUMENTATION TO ANALYZE` together with the observed raw packet.
- The initial tree should contain enough shared header/selector/body structure for this packet to become parseable as early as possible.
- Before choosing any observed selector branch, decode the raw packet using the same field layout that you output: compute each selector field's byte/bit span from its `bit_start` and `size_bits`, extract that exact value from the raw packet, and use only that extracted value for the initial `condition_on` branch.
- Do not infer an observed selector value from any other byte with the same numeric value. length/count/quantity/payload bytes cannot select variants unless your output tree explicitly models the selector field at those bytes.
- If a selector table maps the layout-verified observed selector value to one named variant, instantiate that observed variant path first. Do not enumerate every unobserved sibling variant in the initial tree just to cover the whole protocol; coverage expansion belongs to later refinement/scaffold work.
- If you cannot prove the observed selector value from the output layout and raw bytes, keep the shared selector plus a compact `Observed_Body_Placeholder` rather than guessing a named variant.
- If the observed variant's detailed body appears later in the documentation, create the compact body skeleton now using that section's evidence.
- Do not choose sections by keyword or numeric overlap alone. Tie nodes to packet bytes, layout evidence, and documentation evidence.
- Every non-root node MUST include `source_section_ids`: section indices from the provided documentation summary that justify this node.
- Unless this run explicitly targets only one message direction, do NOT make the protocol root,
  shared header containers, or shared selector fields request-only or response-only. Raw traffic can
  contain unlabeled requests and responses; keep shared scaffolding `message_type="bidirectional"`
  and put direction-specific structure under variant/body nodes.
"""

        prompt = protocol_hint_block + preinitial_traffic_block + """

## CRITICAL NON-NEGOTIABLE GUIDELINES

- **Selector coverage must respect the current task scope**: when observed traffic is provided, instantiate the selector value(s) observed in the packet and keep unobserved sibling variants compact or defer them to later refinement/scaffold work. Do not let exhaustive selector enumeration prevent the observed packet from parsing.
- **Do not collapse bidirectional documentation into one direction**: if the documentation describes both request and response wire layouts, the root protocol/message node, shared headers, and shared selectors must remain `message_type="bidirectional"` unless an explicit single-direction target is supplied. Direction-specific request/response details belong under variant/body nodes; a response-only or request-only root scaffold is invalid.
- **Numeric selector labels**: if documentation writes `22 (0x16)` or `0x16 (22)`, this is one selector value with decimal value 22. Use `val(selector)==22`; do not name it as `OP16` unless the documented decimal value is actually 16. Prefer names like `OP22_0x16` when both forms are shown.
- **Mutually-exclusive layouts must be modeled as variants**: when a control field selects between multiple on-wire layouts that begin at the same offset (different format tables / alternatives), model the control field as `node_type="selector"` and create `node_type="variant"` containers for each alternative, routed via `condition_on`. IMPORTANT: do **NOT** set `parent_id=selector` for those variants unless the entire variant is physically contained within the selector's own bit range (rare, e.g., sub-bitfields). In most protocols the selector is a fixed-size field (e.g., 8 bits) and variants describe the *following* bytes, so variants should be siblings under the same parent container and start at `selector.bit_start + selector.size_bits`. Fields across different variants are allowed to share the same `bit_start` (overlap is expected); do NOT shift offsets to avoid overlap.
- **Selector routing must target variants (not fields)**: for a `node_type="selector"`, every outgoing `condition_on` edge MUST point to a `node_type="variant"` node (never directly to `field` leaves). If multiple leaf fields share the same selector predicate, group them under ONE variant container and attach a single `condition_on` edge to that variant.
- **Overlapping alternatives MUST share one selector**: if normal/error/request/response alternatives are distinguished by the same opcode/function/type byte (including derived values such as the high bit set or value + constant), model ONE shared selector at that wire offset and branch all sibling variants from it. Do NOT create per-variant selector copies.
- **condition_on formulas must reference ONLY the controlling selector**: for an edge `src -> dst` with `rel="condition_on"`, the formula may reference ONLY `val(src)` (plus constants and boolean ops). Do NOT reference `val(other_id)`; if multiple control fields are involved, introduce a nested selector/variant structure instead of mixing them in one formula.
- **condition_on formulas must be valid booleans**: NEVER concatenate clauses (e.g., `A (B)` or `A B`). If you need multiple cases, use explicit `or`/`and` with parentheses.
- **Model nested selectors (don’t flatten)**: if a present/flag-gated section begins with a "Type/Kind/Opcode" field (enumerated values), model it as a nested `node_type="selector"` inside the "section present" variant, then route to auth/option variants from that selector. Do NOT create per-variant shadow selector fields like `Type_A`, `Type_B` and then gate variants on those.
- **Reserved/Padding rows are on-wire**: fields named `Reserved`, `Padding`, `Spare`, `Unused` (or similar) in format tables occupy real bits/bytes. Include them as real leaf fields with the documented span so later offsets stay aligned. Only add hard constraints like `== 0` when the spec explicitly mandates it ("MUST be zero").
- **`reserved_values` are descriptive, not restrictive**: documentation `reserved_values` / "reserved/unassigned" ranges do NOT mean "value must be in that range". Do NOT translate them into `range:` constraints. Keep the allowed numeric domain as the full bit-width or documented min/max; treat reserved ranges as "unknown semantics" (record in description) unless the spec explicitly forbids them.
- **Bytes/payload length constraints use `size_bits`, not `value`**: for nodes with `data_type="bytes"` or `node_type="payload"`, NEVER write `range: ... <= value <= ...` to express byte length. Either omit the range and rely on `size_bits`/`length_of`, or constrain length via `formula: size_bits >= min and size_bits <= max` (in bits).
- **Represent exception patterns explicitly**: when the specification describes bit-derived or offset-based exceptions (e.g., high bit indicates error, value + constant), create separate variants with matching predicates.
- **Length bindings must respect the spec**: if a length field includes shared headers or auxiliary bytes, subtract them before binding to downstream payloads. Use `length_of` edges instead of embedding arithmetic in `size_bits`.
- **Length expressions are in bits**: treat all size formulas as bit-based; if a count field counts bytes convert it as `val(byte_field) * 8`, and if it already counts bits use `val(field)` directly; do not write inverse or rounding formulas.
- **length_of direction and typing**: a `length_of` edge goes from the length/count field to the payload, and the `message_type` of src and dst must agree; put checks and ranges in the `constraints` of the length field, and do not invert the relation inside `length_of`.
- **Parent sizing rule**: a parent container's size_bits should equal the sum of its children or be driven by length_of; do not use max(variant.size_bits) or a fixed constant to cover a variable child, and every variable child must have a length_of binding.
- **Avoid duplicate payloads**: do not keep two equivalent payloads at the same level (such as *_Bytes next to *_Response); keep the one whose length chain is clear.
- **HARD OUTPUT CONSTRAINTS**:
  1. node_type MUST be one of: "protocol", "header", "payload", "field", "selector", "variant", "container", "message", "tlv_seq". If unsure, use "field".
  2. message_type MUST be one of: "bidirectional", "request", "response". If unsure, use "bidirectional".
  3. size_bits MUST be either an integer (e.g., 8) or an SMT-safe expression using ONLY:
     - `val(<id>)` for field values
     - `<id>.bit_start` / `<id>.size_bits` for already-parsed node attributes
     - basic arithmetic `+ - * / %` and parentheses.
     Examples: `val(124)*8`, `0.size_bits - 100.bit_start`, `(10.bit_start + 10.size_bits) - 20.bit_start`.
     If unknown, set size_bits to "variable".
  4. bit_start MUST be non-negative. For siblings that can co-exist in one frame, keep bit_start non-decreasing and avoid overlap. For nodes gated by mutually-exclusive `condition_on` (selector routing), overlapping wire ranges are acceptable (prefer modeling them under variant containers).
  5. Do NOT invent extra node_type values or arbitrary fields; output the minimal schema-compliant JSON.
- **Variants contain only selector-specific bodies**: do not copy shared headers/selectors into variants; start variant `bit_start` immediately after the shared fields to avoid overlap.
- **Forbidden anti-pattern**: never build `Variant_A` with its own local `Opcode_A` and `Variant_B` with `Opcode_B`, then attach separate `condition_on` edges from each local selector. This fails exclusivity checks; use one shared selector instead.
- **SMT-safe expressions are mandatory**: follow the numeric-expression and constraint rules below so every formula is Z3 compatible.
- **Exactly one protocol root**: create a single root node (e.g., `Protocol_Message`) with `parent_id = null`. Every other node must attach beneath that root; never introduce multiple parentless nodes.
- **Selector hierarchy must be unique**: for any selector (opcode, message type, etc.), attach all of its variants beneath a single selector chain. The same variant body must never be referenced by multiple selectors or by both the root and a selector simultaneously.
- **Do not delete documented nodes**: re-parent or restructure them instead of removing them from the tree.
- **Initial-tree minimalism**: if some details are unknown, leave them as `size_bits="variable"` / empty `constraints` rather than inventing undocumented fields or constraints.

## STRUCTURAL ANTI-PATTERNS (AVOID THESE)

1. **Fixed-Header Overflow**: Do NOT place variable or selector fields inside a `header` node if they cause the total size to exceed the header's documented fixed length.
   - *Anti-Pattern*: Packing a 1-byte Selector into a fixed-size Header that is already full.
   - *Solution*: Move the Selector to be a sibling following the Header.

2. **Ambiguous Variants**: When a single Selector value (e.g., 0x01) is valid for BOTH Request and Response messages:
   - You MUST explicitly distinguish them using `message_type`.
   - Set `message_type="request"` for the Request variant and its `condition_on` edge.
   - Set `message_type="response"` for the Response variant and its `condition_on` edge.
   - *Reasoning*: Without this distinction, the validator will reject them as "duplicate conditions".

## UNIVERSAL NODE TYPES (Protocol-Agnostic)

```yaml
protocol:     # Root container for entire protocol message
header:       # Fixed-structure container with sub-fields  
field:        # Concrete data field (leaf node)
selector:     # Control field that determines message variants
variant:      # Alternative message structure based on selector
payload:      # Variable-content data section
container:    # Generic grouping of related fields
```

## POSITIONING STRATEGY (Universal)

CRITICAL: Always Use Numeric Node IDs in Expressions

### PROHIBITED Positioning Patterns:
**Self-references**: `node.bit_start + node.size_bits` (e.g., "22.bit_start + 22.size_bits")
**Parent container references**: Child referencing its direct parent container (e.g., Node 41 with parent_id=40 using "40.bit_start + 0")
**Forward references**: Referencing nodes that don't exist or come later in sequence
**Cross-container references** without proper hierarchy
**Circular dependencies** between nodes

**Key Principle**: bit_start = parent.bit_start + offset_within_parent

## SMT-COMPATIBLE EXPRESSION RULES (STRICT)

- `bit_start` / `size_bits` expressions must remain numeric. Use only literals, additions, or `val(NODE_ID) * constant` forms.
- NEVER embed boolean expressions inside arithmetic (no `(val(X) == 2)` inside `size_bits`).
- Conditions must remain boolean expressions (comparisons + `And`/`Or`/`Not`).
- Reference other nodes via `val(NODE_ID)` or `NODE_ID.bit_start/size_bits`; do not introduce field names. `condition_on` formulas MUST be expressed with `val(<selector_id>)` and may reference only the selector that drives the variants.
- For length prefixes, express the relationship via `length_of` edges rather than inline conditional arithmetic.

## MESSAGE TYPE CLASSIFICATION (Mandatory)

Every node MUST have exactly one message_type:

- **"bidirectional"**: Used in both directions (headers, control fields)
- **"request"**: Client-to-server/caller-to-callee structures  
- **"response"**: Server-to-client/callee-to-caller structures

## CONTROL FIELD DETECTION & VARIANT CREATION
When you find a field that determines message structure:

1. **Identify the selector field** (command, type, opcode, operation, etc.).
2. **Choose branch scope from traffic availability**: when observed traffic is provided, instantiate only selector values that your output layout extracts from the observed packet bytes; when no traffic is provided, include at most 8 representative named variant shells, balanced across documented request/response/exception directions. Defer all remaining documented values to refinement; do not invent a broad condition that pretends to cover them.
3. **Do not duplicate selector fields** inside variants—variants only contain the selector-specific payload.
4. **Add condition_on edges** that mirror the selector constraints exactly, including any bitwise predicates required for exceptions.
5. **Verify coverage according to scope**: with observed traffic, initial coverage means the observed packet path is covered and unobserved siblings are deferred; without observed traffic, initial coverage means representative direction/selector anchors are present and all omitted values remain explicitly deferred to refinement.
6. **Combine multiple selector values in a SINGLE condition_on formula** using `or` (e.g., `val(ID) == 1 or val(ID) == 2`). Do NOT emit multiple condition_on edges for the same variant from the same selector.
7. **Direction-specific opcodes are still modeled under the same selector**: if request/response/exception messages occupy different selector values, create separate variants branching from the same selector (do not mirror the entire payload under a second selector elsewhere).
8. **Exception/derived branches still reuse the original selector**: if an error branch uses a transformed value such as `opcode | 0x80` or `opcode + constant`, keep the original selector at the shared offset and create an additional variant from that same selector. Do NOT insert another selector inside the response/error variant.
```

### ANTI-PATTERN - Avoid This:
```json
{
  "node_id": "variant1_id",
  "name": "Command_A_Variant",
  "bit_start": "same_as_selector",  // POSITION CONFLICT!
  "children_ids": ["duplicate_selector", "field1", "field2"]
},
{
  "node_id": "duplicate_selector", 
  "name": "Command_A_Control_Field",  // DUPLICATE SELECTOR!
  "bit_start": "same_as_original_selector"
}
```

**Universal Rules for Variant Creation:**
- **One selector per protocol layer** - never duplicate control fields
- **Variants contain only command-specific data** - no repeated headers/selectors
- **Sequential positioning** - variants start after shared fields
- **Conditional edges** - link only the selector values represented in this compact initial scaffold; refinement completes the selector domain
 
**Initial tree scope (skeleton-first):** Build a stable universal skeleton. CRITICAL: If you identify fields that act as **Selectors** (e.g., opcodes, message types, auth type, flags that enable/disable blocks), you MUST:
1) set the field `node_type` to `"selector"`, and
2) when observed traffic is provided, create **placeholder** `node_type="variant"` children first for the observed selector values; when no traffic is available, create at most 8 representative named placeholders across directions and defer the rest to refinement,
3) connect selector → variant via `condition_on` edges (**Selector routing must target variants**, not leaf fields).
""" + _initial_tree_budget_block() + """
## CONSTRAINT EXTRACTION (SMT-Compatible Only)

Extract constraints as PURE mathematical expressions:
- "must be 0" → `"enum: 0"`
- "values 1-255" → `"range: 1..255"`
- "one of literal codes 1, 2, 3" → `"enum: 1|2|3"`
- "one of literal codes 0x01, 0x02" → `"enum: 0x01|0x02"`
- If the text lists only names/descriptions without literal numeric/hex/binary codes, do NOT emit `enum:`; keep it in `description` or variant names.
- "length field maximum 64 (bytes)" → `"range: 0..64"`
- "payload maximum 64 bytes" → `"formula: size_bits >= 0 and size_bits <= 512"`  *(or omit and rely on `length_of`)*
- "aligned to 2 bytes" → `"formula: value % 2 == 0"`
- "multiple of 8" → `"formula: value % 8 == 0"`
- "depends on field X" → `"formula: value = val(X_NODE_ID)"`
""" + _formula_rules_block() + """
**CRITICAL: SMT Solver Compatibility**
All constraints will be processed by an SMT solver for traffic generation.
FORBIDDEN constraint formats (natural language):
- "encoding: hexadecimal"
- "alignment: 2-byte boundary"
- "padding: zero bits"
- "endianness: big"
- "enum: Structured option data as defined in sections 2-9"
- "enum: A|B|C"

REQUIRED constraint formats (pure math only):
- Use "range: min..max" for scalar value ranges
- Use "enum: val1|val2|val3" only for literal numeric/hex/binary values, never symbolic names or prose
- Use "formula: mathematical_expression" for complex conditions
- Always use val(NODE_ID) to reference other nodes, NEVER field names

CRITICAL: NO natural language in constraints - only mathematical expressions!

## EDGE MESSAGE TYPE RULES

Every edge MUST have a message_type attribute:
- **"bidirectional"**: Edge applies to both request and response messages
- **"request"**: Edge only applies in request messages
- **"response"**: Edge only applies in response messages

For condition_on edges:
- If the condition determines a request variant → message_type = "request"
- If the condition determines a response variant → message_type = "response"
- If the condition applies to both → message_type = "bidirectional"

For length_of, offset_of, crc_of edges:
- Usually message_type = "bidirectional" unless specific to one direction
- `src` / `dst` MUST be existing numeric node IDs (write them as numbers, not quoted strings).

## OUTPUT SCHEMA (Strict JSON)

## TLV-SEQUENCE MODELLING (protocol-agnostic)

If the protocol contains a repeated TLV list (Tag/Code + optional Length + Value/Data repeated until end-of-list),
model it using a dedicated sequence node:
- Use a node with `node_type: "tlv_seq"` to represent the repeated list.
- The `tlv_seq` node's children define the *item template* to be parsed repeatedly from the current cursor.
- Stop conditions should be expressed via `stop_condition` on the tlv_seq node (e.g., `val(CODE_ID) == END_CODE`)
  and/or by bounding the tlv_seq via its declared `size_bits` (end-of-container).
- Within the item template, use a `selector` + `variant` with `condition_on` edges to model special codes such as
  PAD/END vs normal TLVs. IMPORTANT: `parent_id`/`children_ids` represent *physical containment* in this IR, so
  variants MUST NOT be children of the `selector` node unless they are sub-bitfields within the selector byte.
  Instead, introduce an item container (e.g., `TLV_Item`) under `tlv_seq`, put the tag selector under it, and
  attach all variants as siblings under that item container.
- IMPORTANT: If the TLV tag is modeled as a `selector` (consuming the tag byte), then variants that start at
  `selector.bit_start + selector.size_bits` MUST NOT include a redundant `Code_*`/`Tag_*` field at offset 0.
  Either (A) omit the code field and begin the body with Length/Value, or (B) if you insist on a code field,
  set the variant's `bit_start` to `selector.bit_start` (not after it). Mixing both shifts the TLV body by 1 byte.
- **Typed TLV/IE coverage is required**: a generic `Normal_TLV` / `TLV_Value` catch-all is only a fallback.  When
  documentation names concrete typed items (for example a table of named information elements), create typed variant/container placeholders under the
  shared TLV item selector for those documented types.  Do this even if their detailed body is refined later.
  The generic catch-all must not be treated as satisfying those named typed structures.

### CLASSIC TLV-SEQ PATTERN (copy & adapt; IDs/values are examples)

This example shows the intended modelling shape that avoids common failure modes:
  - missing stop_condition → tlv_seq scans into padding/next layer
  - treating byte length as bit length → leftover_bits != 0, gap_bits=8, OOB reads
  - modelling END as both a TLV variant and an extra trailing field → double-consumption

	```json
	{
	  "root_node_id": 0,
	  "nodes": [
    {"node_id": 0, "name": "PDU", "node_type": "protocol", "message_type": "bidirectional",
     "bit_start": 0, "size_bits": "variable", "data_type": "binary", "byte_order": "big",
     "parent_id": null, "children_ids": [100], "constraints": [], "source": "doc", "dependencies": []},

    {"node_id": 100, "name": "Options_Area", "node_type": "container", "message_type": "bidirectional",
     "bit_start": "0.bit_start + 0", "size_bits": "0.size_bits - 100.bit_start", "data_type": "binary",
     "byte_order": "big", "parent_id": 0, "children_ids": [110, 190], "constraints": [], "source": "doc", "dependencies": []},

	    {"node_id": 110, "name": "Options", "node_type": "tlv_seq", "message_type": "bidirectional",
	     "bit_start": "100.bit_start + 0", "size_bits": "variable",
	     "stop_condition": "val(120) == 127",
	     "parent_id": 100, "children_ids": [115], "constraints": [], "source": "doc", "dependencies": []},

	    {"node_id": 115, "name": "TLV_Item", "node_type": "container", "message_type": "bidirectional",
	     "bit_start": "110.bit_start + 0", "size_bits": "variable", "data_type": "binary",
	     "byte_order": "big", "parent_id": 110, "children_ids": [120, 121, 122, 123], "constraints": [], "source": "doc", "dependencies": []},

	    {"node_id": 120, "name": "TLV_Tag", "node_type": "selector", "message_type": "bidirectional",
	     "bit_start": "115.bit_start + 0", "size_bits": 8, "data_type": "uint8",
	     "byte_order": "big", "parent_id": 115, "children_ids": [], "constraints": [], "source": "doc", "dependencies": []},

	    {"node_id": 121, "name": "Pad_Option", "node_type": "variant", "message_type": "bidirectional",
	     "bit_start": "120.bit_start + 120.size_bits", "size_bits": 0, "data_type": "binary", "byte_order": "big",
	     "parent_id": 115, "children_ids": [], "constraints": [], "source": "doc", "dependencies": []},

	    {"node_id": 122, "name": "End_Option", "node_type": "variant", "message_type": "bidirectional",
	     "bit_start": "120.bit_start + 120.size_bits", "size_bits": 0, "data_type": "binary", "byte_order": "big",
	     "parent_id": 115, "children_ids": [], "constraints": [], "source": "doc", "dependencies": []},

	    {"node_id": 123, "name": "Normal_TLV", "node_type": "variant", "message_type": "bidirectional",
	     "bit_start": "120.bit_start + 120.size_bits", "size_bits": "variable", "data_type": "binary", "byte_order": "big",
	     "parent_id": 115, "children_ids": [124, 125], "constraints": [], "source": "doc", "dependencies": []},

    {"node_id": 124, "name": "TLV_Length_Bytes", "node_type": "field", "message_type": "bidirectional",
     "bit_start": "123.bit_start + 0", "size_bits": 8, "data_type": "uint8", "byte_order": "big",
     "parent_id": 123, "children_ids": [], "constraints": [], "source": "doc", "dependencies": []},

    {"node_id": 125, "name": "TLV_Value", "node_type": "payload", "message_type": "bidirectional",
     "bit_start": "124.bit_start + 124.size_bits", "size_bits": "variable", "data_type": "bytes", "byte_order": "big",
     "parent_id": 123, "children_ids": [], "constraints": [], "source": "doc", "dependencies": []},

    {"node_id": 190, "name": "Padding", "node_type": "payload", "message_type": "bidirectional",
     "bit_start": "110.bit_start + 110.size_bits", "size_bits": "0.size_bits - 190.bit_start", "data_type": "bytes",
     "byte_order": "big", "parent_id": 100, "children_ids": [], "constraints": [], "source": "doc", "dependencies": []}
  ],
  "edges": [
    {"src": 120, "dst": 121, "rel": "condition_on", "formula": "val(120) == 0", "message_type": "bidirectional"},
    {"src": 120, "dst": 122, "rel": "condition_on", "formula": "val(120) == 127", "message_type": "bidirectional"},
    {"src": 120, "dst": 123, "rel": "condition_on", "formula": "(val(120) != 0) and (val(120) != 127)", "message_type": "bidirectional"},
    {"src": 124, "dst": 125, "rel": "length_of", "formula": "val(124)*8", "message_type": "bidirectional"}
  ]
}
```

Notes:
- If the options/TLV area is bounded by a length field (not "to end of message"), do NOT use `0.size_bits - ...`;
  bind the container or tlv_seq via `length_of` from that length field instead.
- `Padding` is optional: only include when the spec allows trailing pad bytes after end-of-list.

{
  "protocol_tree": {
    "root_node_id": 0,
    "nodes": [
      {
        "node_id": 0,
        "name": "Protocol_Root",
        "node_type": "protocol|header|field|selector|variant|payload|container|tlv_seq",
        "message_type": "bidirectional|request|response",
        "bit_start": "positioning_expression (e.g., 0 or 'PARENT.bit_start + 8')",
        "size_bits": "size_expression_or_integer",
        "data_type": "binary|uint8|uint16|uint32|uint64|string|bytes",
        "byte_order": "big|little",
        "parent_id": "parent_node_id_or_null",
        "children_ids": ["array_of_child_node_ids"],
        "constraints": ["constraint_strings"],
        "source": "documentation_reference",
        "source_section_ids": ["section indices from the provided documentation summary"],
        "dependencies": []
      }
    ],
    "edges": [
      {
        "src": source_node_id,
        "dst": destination_node_id,
        "rel": "length_of|condition_on|offset_of|repeat_count|crc_of",
        "formula": "relationship_expression",
        "message_type": "bidirectional|request|response"
      }
    ]
  }
}


## EXAMPLE OUTPUT (Valid JSON; selector → variants skeleton)

{
  "protocol_tree": {
    "root_node_id": 0,
    "nodes": [
      {
        "node_id": 0,
        "name": "PDU",
        "node_type": "protocol",
        "message_type": "bidirectional",
        "bit_start": 0,
        "size_bits": "variable",
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": null,
        "children_ids": [1, 2],
        "constraints": [],
        "source": "doc",
        "source_section_ids": [0],
        "dependencies": []
      },
      {
        "node_id": 1,
        "name": "Header",
        "node_type": "header",
        "message_type": "bidirectional",
        "bit_start": "0.bit_start + 0",
        "size_bits": 16,
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": 0,
        "children_ids": [10],
        "constraints": [],
        "source": "doc",
        "source_section_ids": [0],
        "dependencies": []
      },
      {
        "node_id": 10,
        "name": "Type",
        "node_type": "selector",
        "message_type": "bidirectional",
        "bit_start": "1.bit_start + 0",
        "size_bits": 8,
        "data_type": "uint8",
        "byte_order": "big",
        "parent_id": 1,
        "children_ids": [],
        "constraints": ["enum: 1|2|3"],
        "source": "doc",
        "source_section_ids": [0],
        "dependencies": []
      },
      {
        "node_id": 2,
        "name": "Type_Variants",
        "node_type": "container",
        "message_type": "bidirectional",
        "bit_start": "1.bit_start + 16",
        "size_bits": "variable",
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": 0,
        "children_ids": [20, 21],
        "constraints": [],
        "source": "doc",
        "source_section_ids": [1],
        "dependencies": []
      },
      {
        "node_id": 20,
        "name": "Type_1_Body",
        "node_type": "variant",
        "message_type": "bidirectional",
        "bit_start": "2.bit_start + 0",
        "size_bits": "variable",
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": 2,
        "children_ids": [],
        "constraints": [],
        "source": "doc",
        "source_section_ids": [1],
        "dependencies": []
      },
      {
        "node_id": 21,
        "name": "Type_2_or_3_Body",
        "node_type": "variant",
        "message_type": "bidirectional",
        "bit_start": "2.bit_start + 0",
        "size_bits": "variable",
        "data_type": "binary",
        "byte_order": "big",
        "parent_id": 2,
        "children_ids": [],
        "constraints": [],
        "source": "doc",
        "source_section_ids": [1],
        "dependencies": []
      }
    ],
    "edges": [
      {
        "src": 10,
        "dst": 20,
        "rel": "condition_on",
        "formula": "val(10) == 1",
        "message_type": "bidirectional"
      },
      {
        "src": 10,
        "dst": 21,
        "rel": "condition_on",
        "formula": "val(10) == 2 or val(10) == 3",
        "message_type": "bidirectional"
      }
    ]
  }
}

## CRITICAL SUCCESS FACTORS

1. **Protocol Agnostic**: No hardcoded assumptions about specific protocols
2. **Structure First**: Focus on binary layout over semantic meaning
3. **Concrete Only**: Extract only fields with clear bit positions/sizes
4. **Relationship Driven**: Use edges to express field interdependencies
5. **Variant Aware**: Detect and model conditional message structures
6. **Position Safe**: Avoid circular dependencies and cross-container references
7. **Hierarchy Complete**: Whenever you add a node, append its ID to the parent `children_ids` so the tree stays connected.
8. **Format Consistency**: Ensure the protocol tree format stays consistent and aligned with the protocol semantics, constraints, and specifications described.

## QUALITY CHECKLIST

Before returning your result, verify:
- [ ] Every node has unique numeric node_id
- [ ] Every node has required message_type field
- [ ] No self-referential positioning expressions
- [ ] All selector fields have corresponding variant nodes for the observed traffic path, or documented placeholder variants when no traffic is provided
- [ ] Container hierarchy makes logical sense
- [ ] Size and position expressions are well-formed

Build a **minimal but correct** structure. Better to have a simple, accurate skeleton than a complex, incorrect tree.

## DOCUMENTATION TO ANALYZE:
""" + json.dumps(enhanced_sections, ensure_ascii=False, separators=(",", ":")) + """

Please analyze the above documentation and extract a protocol tree structure following the guidelines above.
Output ONLY the JSON object, no other text."""

        if self.provider == "openai":
            token_key = "max_completion_tokens" if self.default_model.startswith("gpt-5") else "max_tokens"
            payload = {
                "model": self.default_model,
                "messages": [
                    {"role": "system", "content": "You are building a protocol format tree. Be specific and comprehensive. CRITICAL: You must respond with valid JSON only. No markdown, no comments, no extra text."},
                    {"role": "user", "content": prompt + "\n\nIMPORTANT: Respond with ONLY the JSON object, no markdown formatting, no comments."}
                ],
                token_key: self.max_tokens,
                "temperature": self.temperature,
            }
            structured_format = self._structured_response_format()
            if structured_format:
                payload["response_format"] = structured_format
        elif self.provider == "deepseek":
            payload = {
                "model": self.default_model,
                "messages": [
                    {"role": "system", "content": "You are building a protocol format tree. Be specific and comprehensive. You must respond with valid JSON only. No markdown, no comments, no extra text."},
                    {"role": "user", "content": prompt + "\n\nIMPORTANT: Respond with ONLY the JSON object, no markdown formatting, no comments."}
                ],
                "max_tokens": self._clamp_deepseek_max_tokens(self.max_tokens, self.default_model),
                "temperature": self.temperature,
                "top_p": 0.95,
                "stream": False,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            structured_format = self._structured_response_format()
            if structured_format:
                payload["response_format"] = structured_format
            # Off unless set. Some providers serve the model without thinking
            # unless asked.
            initial_reasoning = (os.getenv("STEP2_INITIAL_REASONING_EFFORT") or "").strip().lower()
            if initial_reasoning:
                payload["reasoning"] = {"effort": initial_reasoning}
        else:
            payload = {
                "model": self.default_model,
                "system": "You are building a protocol format tree. Be specific and comprehensive.",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self._clamp_anthropic_max_tokens(self.max_tokens, self.default_model),
                "temperature": self.temperature,
            }

        if request_only:
            return payload

        max_json_retries = _initial_tree_max_json_retries()
        json_retry_count = 0
        fallback_used = False
        initial_retry_budget_ceiling = (
            self._deepseek_token_cap(payload.get("model")) if self.provider == "deepseek"
            else self._anthropic_token_cap(payload.get("model")) if self.provider == "anthropic"
            else max(1, int(payload.get("max_tokens") or self.max_tokens or 1))
        )

        while True:
            try:
                result = self._call_api_with_retry(payload)
                response_meta = self._extract_response_metadata(result)
                self._append_usage_log(
                    stage="initial",
                    meta=response_meta,
                    extra={"json_retry_count": json_retry_count},
                )

                if 'content' in result and isinstance(result['content'], list):
                    raw_response = result['content'][0]['text']
                elif 'choices' in result:
                    message = result['choices'][0]['message']
                    if self.provider == "deepseek" and payload.get("model") == "deepseek-reasoner":
                        raw_response = message.get('content', '')
                    else:
                        raw_response = message['content']
                else:
                    raise ValueError(f"Unexpected API response format: {result.keys()}")

                cache_data = {
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "timestamp": datetime.now().isoformat(),
                    "provider": response_meta.get("provider"),
                    "model": response_meta.get("model"),
                    "stop_reason": response_meta.get("stop_reason"),
                    "finish_reason": response_meta.get("finish_reason"),
                    "usage": response_meta.get("usage"),
                    "token_usage": response_meta.get("token_usage"),
                }

                raw_text = self._validate_and_fix_json_structure(self._clean_raw_response(raw_response))
                tree_data = json.loads(raw_text)
                logger.info("✓ Initial tree parsed directly from API response")
                tree_payload = tree_data.get("protocol_tree", tree_data) if isinstance(tree_data, dict) else tree_data
                tree_payload = (tree_payload)
                schema_issue = _normalize_initial_tree_schema(tree_payload)
                if schema_issue:
                    cache_data["validation_error"] = f"initial tree does not follow the node schema: {schema_issue}"
                    self._save_to_cache(
                        f"initial_tree_ai_response.rejected_schema_attempt_{json_retry_count + 1}.json",
                        cache_data,
                    )
                    raise ValueError(cache_data["validation_error"])
                scope_issue = _initial_tree_scope_rejection_reason(
                    tree_payload,
                    traffic_locate_report,
                    sections=traffic_locate_scope_sections or sections,
                )
                if scope_issue:
                    cache_data["validation_error"] = f"initial tree active/core scope is invalid: {scope_issue}"
                    self._save_to_cache(
                        f"initial_tree_ai_response.rejected_scope_attempt_{json_retry_count + 1}.json",
                        cache_data,
                    )
                    raise ValueError(cache_data["validation_error"])

                cache_data["protocol_tree"] = tree_payload
                self._save_to_cache("initial_tree_ai_response.json", cache_data)
                return self._finalize_initial_tree(tree_payload, protocol_hint_text)

            except Exception as e:
                json_retry_count += 1
                logger.error(f"Initial tree building failed (attempt {json_retry_count}/{max_json_retries}): {e}")

                is_scope_rejection = "initial tree active/core scope is invalid" in str(e)
                if not is_scope_rejection:
                    try:
                        cached_result = self._load_from_cache("initial_tree_ai_response.json")
                        if cached_result and 'protocol_tree' in cached_result:
                            logger.info("✓ Recovered initial tree from cache")
                            return self._finalize_initial_tree(cached_result['protocol_tree'], protocol_hint_text)
                    except Exception:
                        pass

                raw_candidate = ""
                try:
                    if isinstance(locals().get("raw_response"), str):
                        raw_candidate = self._clean_raw_response(locals().get("raw_response") or "")
                except Exception:
                    raw_candidate = ""

                truncated_response = self._looks_like_truncated_json(raw_candidate) or (
                    locals().get("response_meta", {}).get("finish_reason") == "length"
                    or locals().get("response_meta", {}).get("stop_reason") == "max_tokens"
                )

                if self.provider == "deepseek":
                    model_name = payload.get("model") or self.default_model
                    current_budget = self._clamp_deepseek_max_tokens(
                        payload.get("max_tokens") or self.max_tokens or 8000,
                        model_name,
                    )
                    if truncated_response:
                        floor_budget = 16000 if str(model_name).strip() == "deepseek-reasoner" else 8192
                        budget_cap = min(self._deepseek_token_cap(model_name), initial_retry_budget_ceiling)
                        # A model can spend the whole budget reasoning; doubling then fails all
                        # three attempts. The last attempt therefore gets the full cap.
                        if json_retry_count + 1 >= max_json_retries:
                            new_budget = budget_cap
                        else:
                            new_budget = min(max(current_budget * 2, floor_budget), budget_cap)
                        if new_budget > current_budget:
                            payload["max_tokens"] = new_budget
                            logger.warning(
                                "Initial tree response appears truncated; retrying with larger max_tokens=%s",
                                new_budget,
                            )
                        else:
                            logger.warning(
                                "Initial tree response appears truncated; retrying within existing max_tokens=%s",
                                current_budget,
                            )
                elif self.provider == "anthropic" and truncated_response:
                    model_name = payload.get("model") or self.default_model
                    current_budget = self._clamp_anthropic_max_tokens(
                        payload.get("max_tokens") or self.max_tokens or 16000,
                        model_name,
                    )
                    new_budget = min(
                        max(current_budget * 2, 24000),
                        self._anthropic_token_cap(model_name),
                        initial_retry_budget_ceiling,
                    )
                    if new_budget > current_budget:
                        payload["max_tokens"] = new_budget
                        logger.warning(
                            "Initial tree response appears truncated; retrying Anthropic with larger max_tokens=%s",
                            new_budget,
                        )
                    else:
                        logger.warning(
                            "Initial tree response appears truncated; retrying Anthropic within existing max_tokens=%s",
                            current_budget,
                        )

                if truncated_response and payload.get("messages"):
                    retry_instruction = (
                        prompt
                        + "\n\n# PREVIOUS INITIAL TREE RESPONSE TRUNCATED\n"
                        + "The previous initial tree response was truncated or invalid because it was too large. "
                          "Retry with a much smaller initial tree: keep only the stable protocol skeleton, shared fixed "
                          "headers, selector/code fields, generic repeated containers, minimal typed variant shells, "
                          "and unknown/fallback variants. Keep the initial tree under 40 nodes if possible. "
                          "Do NOT expand option/code tables into detailed body fields; leave detailed bodies, "
                          "coverage-only variants, and relation constraints to refinement or coverage-obligation passes."
                    )
                    if isinstance(payload.get("messages"), list) and payload["messages"]:
                        payload["messages"][-1]["content"] = (
                            retry_instruction
                            + "\n\nIMPORTANT: Respond with ONLY the JSON object, no markdown formatting, no comments."
                        )

                if is_scope_rejection and payload.get("messages"):
                    collapsed_direction_scope = "message_type scope collapsed" in str(e)
                    if collapsed_direction_scope:
                        retry_tail = (
                            ". Retry with shared protocol/message, header, container, and selector nodes marked "
                            "message_type=\"bidirectional\". Keep direction-specific request/response bodies as "
                            "variants beneath the shared selector/body boundary. Do not publish a request-only or "
                            "response-only root scaffold unless this run explicitly targets one direction."
                        )
                    else:
                        retry_tail = (
                            ". Retry with an active-path initial tree only: keep shared headers, observed selector fields, "
                            "observed selector table/mapping fields, and the selected active variant/body. Do not instantiate "
                            "coverage-only or deferred sibling variants in the initial tree."
                        )
                    retry_instruction = (
                        prompt
                        + "\n\n# PREVIOUS INITIAL TREE RESPONSE REJECTED\n"
                        + "The previous initial tree response was rejected because "
                        + str(e)
                        + retry_tail
                    )
                    if isinstance(payload.get("messages"), list) and payload["messages"]:
                        payload["messages"][-1]["content"] = (
                            retry_instruction
                            + "\n\nIMPORTANT: Respond with ONLY the JSON object, no markdown formatting, no comments."
                        )

                if json_retry_count < max_json_retries:
                    logger.warning("Retrying initial tree generation (attempt %d)", json_retry_count + 1)
                    continue

                if (
                    self.provider == "deepseek"
                    and self.allow_deepseek_reasoner_fallback
                    and not fallback_used
                    and payload.get("model") != "deepseek-reasoner"
                ):
                    fallback_used = True
                    json_retry_count = 0
                    payload["model"] = "deepseek-reasoner"
                    payload["max_tokens"] = self._clamp_deepseek_max_tokens(32000, "deepseek-reasoner")
                    logger.warning(
                        "Switching initial tree generation to deepseek-reasoner after repeated JSON failures"
                    )
                    continue
                break

        return None

    def _finalize_initial_tree(self, tree_data: Any, protocol_hint_text: str) -> Any:
        if not isinstance(tree_data, dict):
            return tree_data

        tree_data = normalize_protocol_tree(tree_data)
        if protocol_hint_text:
            tree_data.setdefault("protocol_name", protocol_hint_text)
            nodes = tree_data.get("nodes")
            if isinstance(nodes, list) and nodes:
                root_name = str((nodes[0] or {}).get("name") or "").strip()
                if root_name.lower() in {"", "generic_protocol_message", "unknown protocol", "protocol_root"}:
                    nodes[0]["name"] = _protocol_hint_root_name(protocol_hint_text)
            if synchronize_parent_child_links(tree_data):
                logger.info(
                    "Synchronized parent/children links in initial tree for protocol hint '%s'",
                    protocol_hint_text,
                )
        return tree_data

    def _parse_tree_response(self, tree_data: Dict[str, Any]) -> Optional[ProtocolTree]:
        try:
            nodes = {}
            for node_data in tree_data.get('nodes', []):
                children = node_data.get('children_ids', []) or []
                normalized_children = [str(child) for child in children]
                parent_id = node_data.get('parent_id')
                if parent_id is not None:
                    parent_id = str(parent_id)
                node_type = canonicalize_node_type(node_data.get('node_type'))
                node = ProtocolNode(
                    node_id=str(node_data.get('node_id')),
                    name=node_data.get('name', 'Unknown'),
                    node_type=node_type,
                    description=node_data.get('description', ''),
                    bit_start=node_data.get('bit_start'),
                    size_bits=node_data.get('size_bits'),
                    data_type=node_data.get('data_type'),
                    byte_order=node_data.get('byte_order', 'big'),
                    message_type=node_data.get('message_type', '') or '',
                    constraints=node_data.get('constraints', []) or [],
                    dependencies=node_data.get('dependencies', []) or [],
                    parent_id=parent_id,
                    children_ids=normalized_children,
                    source=node_data.get('source', '') or '',
                    source_section_ids=copy.deepcopy(node_data.get('source_section_ids', []) or []),
                    extra_attributes={key: copy.deepcopy(value) for key, value in node_data.items()
                                      if key not in ProtocolNode.__dataclass_fields__},
                    confidence_score=node_data.get('confidence_score', 0.7)
                )
                nodes[node.node_id] = node

            tree = ProtocolTree(
                protocol_name=tree_data.get('protocol_name', 'Unknown Protocol'),
                root_node_id=str(tree_data.get('root_node_id', 'root')),
                nodes=nodes,
                edges=list(tree_data.get('edges', [])),
                metadata=copy.deepcopy(tree_data.get('metadata') or {'initial_nodes': len(nodes)}),
                created_at=datetime.now().isoformat()
            )

            return tree

        except Exception as e:
            logger.error(f"Tree parsing failed: {e}")
            return None

class UniversalProtocolAnalyzer:

    def __init__(
        self,
        *,
        initial_provider: Optional[str] = None,
        initial_model: Optional[str] = None,
        initial_temperature: Optional[float] = None,
        initial_max_tokens: Optional[int] = None,
        refine_provider: Optional[str] = None,
        refine_model: Optional[str] = None,
        refine_temperature: Optional[float] = None,
        refine_max_tokens: Optional[int] = None,
    ):
        # Use separate LLM configs for Stage B initial generation vs refinement.
        # Initial generation and section-by-section refinement use the same default unless the caller overrides them.
        self.ai_agent = GenerationLLMAgent(
            provider=initial_provider,
            model=initial_model,
            temperature=initial_temperature,
            max_tokens=initial_max_tokens,
        )
        self.refine_agent = GenerationLLMAgent(
            provider=refine_provider or "anthropic",
            model=refine_model,
            temperature=refine_temperature,
            max_tokens=refine_max_tokens,
        )

    def analyze_protocol(
        self,
        sections_file: str,
        raw_file: str,
        traffic_file: Optional[str] = None,
        *,
        protocol_hint: Optional[str] = None,
        cache_namespace: Optional[str] = None,
    ):

        namespace = cache_namespace or os.getenv("PARGEN_STEP2_CACHE_NAMESPACE") or _infer_step2_cache_namespace(sections_file, raw_file)
        if namespace:
            cache_dir = str(Path(STEP2_CACHE_DIR) / namespace)
            self.ai_agent.cache_dir = cache_dir
            self.refine_agent.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(
                "Step2 cache namespace: %s (%s) initial_provider=%s refine_provider=%s",
                namespace,
                cache_dir,
                getattr(self.ai_agent, "provider", "?"),
                getattr(self.refine_agent, "provider", "?"),
            )

        input_signature = build_stage2_input_signature(
            sections_file=sections_file,
            raw_file=raw_file,
            traffic_file=traffic_file,
            protocol_hint=protocol_hint,
            cache_namespace=namespace,
            initial_llm=self.ai_agent.signature_config(),
            refine_llm=self.refine_agent.signature_config(),
        )
        self.ai_agent.input_signature = input_signature
        self.refine_agent.input_signature = input_signature

        with open(sections_file, 'r', encoding='utf-8') as f:
            sections_data = json.load(f)
            # Handle both formats: direct array or {"sections": [...]}
            if isinstance(sections_data, list):
                sections = sections_data
            else:
                sections = sections_data.get('sections', [])

        with open(raw_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            # Handle both formats: direct array or {"sections": [...]}
            if isinstance(raw_data, list):
                raw_sections = raw_data
            else:
                raw_sections = raw_data.get('sections', [])

        traffic_locate_report: Optional[Dict[str, Any]] = None
        initial_sections = sections
        refine_sections = sections
        refine_raw_sections = raw_sections
        stageb_scope_sections = sections
        if traffic_file and _env_flag("STEP2_TRAFFIC_LLM_LOCATE", "0"):
            logger.info(
                "Ignoring deprecated STEP2_TRAFFIC_LLM_LOCATE request; StageB now builds the initial tree from the full fixed summary plus observed traffic."
            )

        logger.info("Phase 1: Building initial protocol tree...")
        initial_tree_data = self.ai_agent.build_initial_tree(
            initial_sections,
            protocol_hint=protocol_hint,
            traffic_file=traffic_file,
            traffic_locate_report=traffic_locate_report,
            traffic_locate_scope_sections=None,
        )
        initial_tree_data = (initial_tree_data)
        protocol_hint_text = _normalize_protocol_hint(protocol_hint)
        if protocol_hint_text and isinstance(initial_tree_data, dict):
            initial_tree_data.setdefault("protocol_name", protocol_hint_text)

        if not initial_tree_data:
            raise ValueError("Failed to build initial tree")

        # Phase 2: refine protocol tree
        logger.info("Phase 2: Refine protocol tree...")
        final_tree = self.refine_agent._refine_tree_with_raw_data(
            initial_tree_data,
            refine_raw_sections,
            refine_sections,
            traffic_file=traffic_file,
        )
        if _env_flag("STEP2_CONSTRAINT_LOCALIZATION", "0"):
            try:
                final_tree, constraint_changes = self.refine_agent.localize_and_attach_constraints(
                    final_tree,
                    stageb_scope_sections,
                    protocol_hint=protocol_hint,
                    traffic_file=traffic_file,
                )
                if constraint_changes:
                    logger.info(
                        "Applied post-refine constraint localization (%d additions): %s",
                        len(constraint_changes),
                        constraint_changes[:12],
                    )
            except Exception as exc:
                if _env_flag("STEP2_CONSTRAINT_LOCALIZATION_STRICT", "0"):
                    raise
                logger.warning("Post-refine constraint localization skipped after failure: %s", exc)
        if protocol_hint_text and isinstance(final_tree, dict):
            final_tree = normalize_protocol_tree(final_tree)
            final_tree.setdefault("protocol_name", protocol_hint_text)
            if synchronize_parent_child_links(final_tree):
                logger.info(
                    "Synchronized parent/children links in refined tree for protocol hint '%s'",
                    protocol_hint_text,
                )
        return final_tree

    def _display_node(self, tree: ProtocolTree, node_id: str, depth: int, max_depth: int = 4):
        if depth > max_depth:
            return

        node = tree.get_node(node_id)
        if not node:
            return

        indent = "  " * depth
        icon = {
            'protocol': '🌐',
            'header': '📋',
            'payload': '📦',
            'field': '🔢',
            'composite': '📁'
        }.get(node.node_type, '📄')

        info = f"{icon} {node.name}"
        if node.node_type == "field":
            info += f" [{node.size_bits} bits"
            if isinstance(node.bit_start, int):
                info += f" @{node.bit_start}"
            info += f", {node.data_type or 'unknown'}]"

        print(f"{indent}{info}")

        if node.parse_success_count > 0 or node.parse_failure_count > 0:
            total = node.parse_success_count + node.parse_failure_count
            success_rate = node.parse_success_count / total * 100
            print(f"{indent}  └─ Parse: {success_rate:.0f}% success ({total} attempts)")

        for child_id in node.children_ids:
            self._display_node(tree, child_id, depth + 1, max_depth)
