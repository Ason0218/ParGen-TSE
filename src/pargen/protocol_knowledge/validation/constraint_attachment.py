"""Attach localized documentation constraints to a Stage-B tree.

The localizer is allowed to identify relation-level obligations that are useful
for knowledge evaluation and Stage-C prompts but risky to force into executable
parser predicates.  This module keeps the first pass conservative: it only adds
`constraint_ir:` annotations to the best matching existing node.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..constraint_ir import constraint_ir_to_text


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_CONSTRAINT_ANCHOR_TOKENS = {
    "alignment",
    "checksum",
    "code",
    "count",
    "data",
    "eol",
    "flag",
    "flags",
    "fragment",
    "header",
    "ihl",
    "ie",
    "kind",
    "len",
    "length",
    "mss",
    "offset",
    "option",
    "padding",
    "payload",
    "pointer",
    "reserved",
    "selector",
    "size",
    "syn",
    "tag",
    "tlv",
    "type",
    "urg",
    "value",
    "variant",
}


def _tokens(value: Any) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(str(value or ""))}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _node_id_text(node: Mapping[str, Any]) -> str:
    return str(node.get("node_id"))


def _find_node_by_id(nodes: Sequence[Mapping[str, Any]], node_id: Any) -> Optional[Mapping[str, Any]]:
    if node_id is None:
        return None
    wanted = str(node_id)
    for node in nodes:
        if _node_id_text(node) == wanted:
            return node
    return None


def _best_matching_node(
    nodes: Sequence[Mapping[str, Any]],
    *,
    anchor_name: Any = None,
    field_names: Iterable[Any] = (),
    scope_terms: Iterable[Any] = (),
) -> Optional[Mapping[str, Any]]:
    wanted_groups = [_tokens(anchor_name)]
    wanted_groups.extend(_tokens(name) for name in field_names)
    wanted_groups.append(_tokens(" ".join(str(term) for term in scope_terms)))
    wanted_groups = [group for group in wanted_groups if group]
    if not wanted_groups:
        return None

    best: Optional[Mapping[str, Any]] = None
    best_score = 0
    for node in nodes:
        node_tokens = _tokens(node.get("name"))
        if not node_tokens:
            continue
        score = 0
        for group in wanted_groups:
            overlap = node_tokens & group
            if not overlap:
                continue
            score += len(overlap)
            if node_tokens == group or group.issubset(node_tokens) or node_tokens.issubset(group):
                score += 3
        if score > best_score:
            best = node
            best_score = score
    return best


def _root_node(nodes: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for node in nodes:
        if node.get("parent_id") is None:
            return node
    return nodes[0] if nodes else None


def _node_score(node: Mapping[str, Any], endpoint_ids: set[str]) -> int:
    score = 0
    node_id = str(node.get("node_id"))
    node_type = str(node.get("node_type") or "").lower()
    if node_id in endpoint_ids:
        score += 3 if node_type == "payload" else 8
    if node_type in {"selector", "variant", "tlv_seq"}:
        score += 6
    elif node_type in {"field", "payload", "container", "header"}:
        score += 2
    score += len(_tokens(node.get("name")) & _CONSTRAINT_ANCHOR_TOKENS) * 3
    if node.get("constraints"):
        score += 2
    if "length" in _tokens(node.get("name")) or "len" in _tokens(node.get("name")):
        score += 4
    if "offset" in _tokens(node.get("name")):
        score += 4
    return score


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _compact_section_brief(section: Mapping[str, Any], *, max_chars: int) -> Dict[str, Any]:
    brief: Dict[str, Any] = {}
    for key in ("number", "title", "heading", "section_title", "name"):
        if section.get(key):
            brief[key] = _truncate_text(section.get(key), 180)
    for key in ("content", "summary", "packet_formats", "field_definitions", "selector_obligations", "coverage_obligations"):
        value = section.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False)
            except Exception:
                text = str(value)
        if text.strip():
            brief[key] = _truncate_text(text, max_chars)
    return brief


def build_compact_constraint_context(
    tree: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
    *,
    max_nodes: int = 90,
    max_sections: int = 28,
    max_section_chars: int = 900,
) -> Dict[str, Any]:
    """Build a smaller context for constraint localization prompts.

    The compact view preserves high-value anchors for cross-field and scoped
    constraints: relation endpoints, selectors, variants, TLV nodes, length/code
    fields, and existing constraints.  It is intentionally lossy and should be
    used with full-prompt fallback when metric preservation matters.
    """

    nodes = tree.get("nodes") if isinstance(tree, Mapping) else []
    edges = tree.get("edges") if isinstance(tree, Mapping) else []
    endpoint_ids: set[str] = set()
    edge_digest: List[Dict[str, Any]] = []
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            rel = str(edge.get("rel") or "")
            if rel not in {"condition_on", "length_of", "repeat_count"}:
                continue
            endpoint_ids.add(str(edge.get("src")))
            endpoint_ids.add(str(edge.get("dst")))
            edge_digest.append(
                {
                    "src": edge.get("src"),
                    "dst": edge.get("dst"),
                    "rel": rel,
                    "formula": edge.get("formula"),
                    "message_type": edge.get("message_type"),
                }
            )

    scored_nodes: List[tuple[int, int, Mapping[str, Any]]] = []
    if isinstance(nodes, list):
        for idx, node in enumerate(nodes):
            if isinstance(node, Mapping):
                scored_nodes.append((_node_score(node, endpoint_ids), idx, node))
    selected = [
        node
        for score, _idx, node in sorted(scored_nodes, key=lambda item: (-item[0], item[1]))
        if score > 0
    ][:max_nodes]
    if not selected and isinstance(nodes, list):
        selected = [node for node in nodes if isinstance(node, Mapping)][:max_nodes]

    node_digest: List[Dict[str, Any]] = []
    for node in selected:
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

    section_briefs = [
        _compact_section_brief(section, max_chars=max_section_chars)
        for section in list(sections or [])[:max_sections]
        if isinstance(section, Mapping)
    ]
    return {
        "tree_digest": {
            "nodes": node_digest,
            "edges": edge_digest[: max_nodes * 2],
        },
        "section_briefs": section_briefs,
    }


def _canonical_field_ids(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(int(text) if text.isdigit() else text)
    return result


def _canonical_field_names(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = " ".join(sorted(_tokens(text))) or text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _constraint_ir_from_obligation(obligation: Mapping[str, Any], anchor_id: Any) -> Optional[str]:
    kind = str(obligation.get("kind") or obligation.get("constraint_type") or obligation.get("type") or "").strip()
    if not kind:
        return None
    expr = str(
        obligation.get("expr")
        or obligation.get("expression")
        or obligation.get("formula")
        or obligation.get("description")
        or ""
    ).strip()
    field_ids = _canonical_field_ids([anchor_id, *_as_list(obligation.get("field_ids")), *_as_list(obligation.get("node_ids"))])
    field_names = _canonical_field_names(
        [
            obligation.get("anchor_node_name"),
            *_as_list(obligation.get("field_names")),
            *_as_list(obligation.get("target_field_names")),
        ]
    )
    source = str(obligation.get("source") or obligation.get("spec_evidence") or "").strip()
    return constraint_ir_to_text(
        kind=kind,
        expr=expr,
        field_ids=field_ids or None,
        field_names=field_names or None,
        source=source,
    )


def attach_constraint_obligations(
    tree: Mapping[str, Any],
    obligations: Iterable[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return a copy of *tree* with localized obligations attached as IR.

    Obligations are intentionally small JSON objects produced by an LLM locate
    pass.  The safest first behavior is relation-only attachment, not structural
    rewrites, so this function never removes nodes/edges and never changes
    parser-executable fields such as `size_bits`.
    """

    patched: Dict[str, Any] = copy.deepcopy(dict(tree))
    nodes = patched.get("nodes")
    if not isinstance(nodes, list):
        patched["nodes"] = []
        nodes = patched["nodes"]

    changes: List[Dict[str, Any]] = []
    for raw in obligations or []:
        if not isinstance(raw, Mapping):
            continue
        anchor = _find_node_by_id(nodes, raw.get("anchor_node_id") or raw.get("node_id"))
        if anchor is None:
            anchor = _best_matching_node(
                nodes,
                anchor_name=raw.get("anchor_node_name"),
                field_names=[*_as_list(raw.get("field_names")), *_as_list(raw.get("target_field_names"))],
                scope_terms=_as_list(raw.get("scope_terms")),
            )
        if anchor is None:
            anchor = _root_node(nodes)
        if anchor is None:
            continue

        constraints = anchor.get("constraints")
        if not isinstance(constraints, list):
            constraints = []
            anchor["constraints"] = constraints

        text = _constraint_ir_from_obligation(raw, anchor.get("node_id"))
        if not text or text in constraints:
            continue
        constraints.append(text)
        changes.append(
            {
                "node_id": anchor.get("node_id"),
                "kind": str(raw.get("kind") or raw.get("constraint_type") or raw.get("type") or "").strip(),
                "constraint": text,
            }
        )

    return patched, changes
