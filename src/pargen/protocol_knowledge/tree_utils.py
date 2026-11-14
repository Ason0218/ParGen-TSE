"""Shared helpers for protocol tree normalization."""

from __future__ import annotations

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


def _shared_subtree_clone_limit() -> int:
    raw = os.getenv("STEP2_SHARED_SUBTREE_CLONE_MAX_NODES", "50")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 50
    return max(0, value)


def _coerce_int(value: Any) -> Any:
    """Attempt to coerce stringified integers (including hex) to int."""

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return value
        try:
            return int(candidate, 0)
        except ValueError:
            return value
    return value


_NUMERIC_LITERAL_PATTERN = r"[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)"
_ENUM_VALUE_RE = re.compile(rf"^{_NUMERIC_LITERAL_PATTERN}$")


def _parse_numeric_literal(value: str) -> int:
    text = str(value).strip()
    try:
        return int(text, 0)
    except ValueError:
        return int(text, 10)


def parse_numeric_enum_constraint_text(value: Any) -> Optional[Tuple[int, ...]]:
    """Parse the executable numeric enum DSL; return None for prose/malformed input."""

    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"enum:\s*(.+)", value.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    body = match.group(1).strip()
    if len(body) >= 2 and (body[0], body[-1]) in {
        ("{", "}"),
        ("[", "]"),
        ("(", ")"),
    }:
        body = body[1:-1].strip()
    parts = [part.strip() for part in re.split(r"[|,]", body) if part.strip()]
    if not parts or not all(_ENUM_VALUE_RE.fullmatch(part) for part in parts):
        return None
    return tuple(_parse_numeric_literal(part) for part in parts)


def parse_numeric_range_constraint_text(value: Any) -> Optional[Tuple[int, int]]:
    """Parse supported numeric range DSL without mining incidental prose numbers."""

    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"range:\s*(.+)", value.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    body = match.group(1).strip().replace("≤", "<=").replace("≥", ">=")
    number = _NUMERIC_LITERAL_PATTERN
    byte_unit = r"(?:\s*bytes?)?"
    patterns = (
        rf"(?P<lo>{number}){byte_unit}\s*\.\.\s*(?P<hi>{number}){byte_unit}",
        rf"(?P<lo>{number})\s*(?:-|~)\s*(?P<hi>{number})",
        rf"(?P<lo>{number})\s*,\s*(?P<hi>{number})",
        rf"(?P<lo>{number})\s+to\s+(?P<hi>{number})",
    )
    for pattern in patterns:
        parsed = re.fullmatch(pattern, body, flags=re.IGNORECASE)
        if parsed is not None:
            lo = _parse_numeric_literal(parsed.group("lo"))
            hi = _parse_numeric_literal(parsed.group("hi"))
            return (lo, hi) if lo <= hi else None

    comparison = re.fullmatch(
        rf"(?P<lo>{number})\s*(?P<lop><=|<)\s*"
        rf"(?:value|size_bits|bit_start)\s*(?P<hip><=|<)\s*(?P<hi>{number})",
        body,
        flags=re.IGNORECASE,
    )
    if comparison is not None:
        lo = _parse_numeric_literal(comparison.group("lo"))
        hi = _parse_numeric_literal(comparison.group("hi"))
        if comparison.group("lop") == "<":
            lo += 1
        if comparison.group("hip") == "<":
            hi -= 1
        return (lo, hi) if lo <= hi else None
    return None


def parse_numeric_calculated_upper_range_constraint_text(value: Any) -> Optional[int]:
    """Parse the legacy ``range: <min>..calculated <name>`` lower-bound form."""

    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        rf"range:\s*(?P<lo>{_NUMERIC_LITERAL_PATTERN})\s*\.\.\s*"
        r"calculated\s+[A-Za-z_][A-Za-z0-9_]*",
        value.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return _parse_numeric_literal(match.group("lo"))


def is_valid_enum_constraint_text(value: Any) -> bool:
    """Return true only for executable enum constraints with literal values."""

    return parse_numeric_enum_constraint_text(value) is not None


_IMPLICATION_RE = re.compile(r"\s*(?:=>|⇒|\bimplies\b)\s*", re.IGNORECASE)


def desugar_implication(expr: str) -> str:
    """Rewrite a conditional rule `A => B` as the boolean `(not (A)) or (B)`.

    Validity rules include conditional (cross-field) rules such as DF = 1 ⇒ MF = 0; every
    evaluator here (the syntax validator's z3 check, the traffic interpreter, Stage C
    constraint injection) understands Python-style and/or/not but not `=>`, so the
    implication is desugared once, at ingestion. `A => B => C` reads as `A => (B => C)`.
    """
    if not isinstance(expr, str):
        return expr
    parts = _IMPLICATION_RE.split(expr, maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return expr
    return f"(not ({parts[0].strip()})) or ({desugar_implication(parts[1].strip())})"


def desugar_formula_constraint(constraint: Any) -> Any:
    if isinstance(constraint, str) and constraint.strip().lower().startswith("formula:"):
        head, body = constraint.split(":", 1)
        return f"{head}: {desugar_implication(body.strip())}"
    return constraint


_BARE_NODE_PROPERTY_RE = re.compile(r"(?<![\w.])(size_bits|bit_start)\b")


def bind_node_properties(constraint: Any, node_id: Any) -> Any:
    """Bind a bare `size_bits` / `bit_start` in a node's formula to that node: `52.size_bits`.

    The generation prompt recommends `formula: size_bits >= min and size_bits <= max` for a
    byte-length bound, but the syntax validator only accepts `<node_id>.size_bits` (a bare name
    is EXPRESSION_UNSUPPORTED_SYNTAX). The traffic interpreter evaluates `<node_id>.size_bits` too.
    """
    if node_id is None or not isinstance(constraint, str) or not constraint.strip().lower().startswith("formula:"):
        return constraint
    head, body = constraint.split(":", 1)
    return f"{head}:{_BARE_NODE_PROPERTY_RE.sub(lambda m: f'{node_id}.{m.group(1)}', body)}"


def sanitize_executable_constraints(constraints: Any) -> tuple[list[Any], int]:
    """Drop executable-looking constraints that are not valid executable DSL."""

    if not isinstance(constraints, list):
        return [], 0
    out: list[Any] = []
    dropped = 0
    for constraint in constraints:
        constraint = desugar_formula_constraint(constraint)
        if isinstance(constraint, str):
            lowered = constraint.strip().lower()
            if lowered.startswith("enum:") and not is_valid_enum_constraint_text(constraint):
                dropped += 1
                continue
            if (
                lowered.startswith("range:")
                and parse_numeric_range_constraint_text(constraint) is None
                and parse_numeric_calculated_upper_range_constraint_text(constraint) is None
            ):
                dropped += 1
                continue
        out.append(constraint)
    return out, dropped


def _node_name_text(node: Dict[str, Any]) -> str:
    parts = [
        node.get("name"),
        node.get("field_name"),
        node.get("label"),
        node.get("description"),
    ]
    return " ".join(str(part) for part in parts if part is not None).strip().lower()


def _looks_like_repeat_count_source(node: Dict[str, Any]) -> bool:
    text = _node_name_text(node)
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not normalized:
        return False
    length_tokens = {"length", "len", "size", "byte", "bytes", "octet", "octets", "bit", "bits"}
    tokens = [token for token in normalized.split("_") if token]
    if any(token in length_tokens for token in tokens):
        return False
    if "count" in tokens or normalized.endswith("count"):
        return True
    if "cnt" in tokens or normalized.endswith("cnt"):
        return True
    if "number" in tokens or normalized.startswith("num_"):
        return True
    return False


def _looks_like_repeated_target(node: Dict[str, Any]) -> bool:
    node_type = str(node.get("node_type") or "").strip().lower()
    if node_type in {"tlv_seq", "sequence", "repeated", "list"}:
        return True
    if node_type in LEAF_NODE_TYPES:
        return False
    if not (node.get("children_ids") or []):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", _node_name_text(node)).strip("_")
    tokens = [token for token in normalized.split("_") if token]
    repeated_tokens = {
        "entries",
        "entry",
        "items",
        "item",
        "records",
        "record",
        "questions",
        "question",
        "answers",
        "answer",
        "options",
        "option",
        "parameters",
        "parameter",
        "values",
        "value",
        "rrs",
        "ies",
    }
    return any(token in repeated_tokens for token in tokens)


def _repeat_count_formula_from_length_formula(formula: Any, src_id: Any) -> Any:
    if formula is None:
        return f"val({src_id})"
    if not isinstance(formula, str):
        return None
    src = re.escape(str(src_id))
    compact = re.sub(r"\s+", "", formula)
    val_src = rf"(?:val\({src}\)|\(val\({src}\)\))"
    if re.fullmatch(val_src, compact):
        return f"val({src_id})"
    if re.fullmatch(rf"{val_src}\*\d+", compact):
        return f"val({src_id})"
    if re.fullmatch(rf"\d+\*{val_src}", compact):
        return f"val({src_id})"
    return None


LEAF_NODE_TYPES = {"field", "selector", "type", "length", "checksum", "payload"}
SEQUENCE_NODE_TYPES = {"tlv_seq", "sequence", "repeated", "list"}


def _repeat_count_target_for_node(node: Dict[str, Any], nodes_by_id: Dict[Any, Dict[str, Any]]) -> Any:
    node_id = _coerce_int(node.get("node_id"))
    node_type = str(node.get("node_type") or "").strip().lower()
    if node_type in SEQUENCE_NODE_TYPES:
        children = [_coerce_int(child_id) for child_id in (node.get("children_ids") or [])]
        if len(children) == 1 and children[0] in nodes_by_id:
            return children[0]
    return node_id


def normalize_count_length_edges(tree_like: Dict[str, Any]) -> bool:
    """Convert count-as-length edges on repeated structures into repeat_count edges.
    
    Recognize length_of expressions of the form val(N) * item_width on repeated
    containers and preserve all other edges."""

    if not isinstance(tree_like, dict):
        return False
    tree = tree_like
    seen: set[int] = set()
    while isinstance(tree, dict) and "protocol_tree" in tree and isinstance(tree["protocol_tree"], dict):
        identifier = id(tree)
        if identifier in seen:
            break
        seen.add(identifier)
        tree = tree["protocol_tree"]
    if not isinstance(tree, dict):
        return False

    nodes = tree.get("nodes")
    edges = tree.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False

    nodes_by_id: Dict[Any, Dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if node_id is None:
            continue
        coerced = _coerce_int(node_id)
        nodes_by_id[node_id] = node
        nodes_by_id[coerced] = node
        nodes_by_id[str(coerced)] = node

    changed = False
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        rel = str(edge.get("rel") or "").strip().lower()
        if rel not in {"length_of", "repeat_count"}:
            continue
        src = _coerce_int(edge.get("src"))
        dst = _coerce_int(edge.get("dst"))
        src_node = nodes_by_id.get(src) or nodes_by_id.get(edge.get("src"))
        dst_node = nodes_by_id.get(dst) or nodes_by_id.get(edge.get("dst"))
        if rel == "repeat_count" and isinstance(dst_node, dict):
            repeat_dst = _repeat_count_target_for_node(dst_node, nodes_by_id)
            if repeat_dst != dst:
                edge["src"] = src
                edge["dst"] = repeat_dst
                changed = True
            continue

        repeat_formula = _repeat_count_formula_from_length_formula(edge.get("formula"), src)
        if (
            isinstance(src_node, dict)
            and isinstance(dst_node, dict)
            and repeat_formula is not None
            and _looks_like_repeat_count_source(src_node)
            and _looks_like_repeated_target(dst_node)
        ):
            repeat_dst = _repeat_count_target_for_node(dst_node, nodes_by_id)
            edge["src"] = src
            edge["dst"] = repeat_dst
            edge["rel"] = "repeat_count"
            edge["formula"] = repeat_formula
            changed = True
    return changed


def normalize_protocol_tree(tree_like: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a protocol tree dict: flatten wrappers, coerce IDs, ensure edges list."""

    if not isinstance(tree_like, dict):
        return tree_like

    # Unwrap nested protocol_tree keys
    tree = tree_like
    seen: set[int] = set()
    while isinstance(tree, dict) and "protocol_tree" in tree and isinstance(tree["protocol_tree"], dict):
        identifier = id(tree)
        if identifier in seen:
            break
        seen.add(identifier)
        tree = tree["protocol_tree"]

    if not isinstance(tree, dict):
        return tree_like

    # Ensure nodes and edges containers exist
    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
        tree["nodes"] = nodes

    edges = tree.get("edges")
    if not isinstance(edges, list):
        edges = []
        tree["edges"] = edges

    # Root node id coercion
    if "root_node_id" in tree:
        tree["root_node_id"] = _coerce_int(tree.get("root_node_id"))

    # Coerce node IDs and relationships
    for node in nodes:
        if not isinstance(node, dict):
            continue

        if "node_id" in node:
            node["node_id"] = _coerce_int(node.get("node_id"))

        constraints = node.get("constraints")
        if isinstance(constraints, list):
            sanitized_constraints, dropped_constraints = sanitize_executable_constraints(constraints)
            sanitized_constraints = [bind_node_properties(c, node.get("node_id")) for c in sanitized_constraints]
            if dropped_constraints or sanitized_constraints != constraints:
                node["constraints"] = sanitized_constraints

        if "parent_id" in node:
            node["parent_id"] = _coerce_int(node.get("parent_id"))

        children = node.get("children_ids")
        if isinstance(children, list):
            node["children_ids"] = [_coerce_int(child) for child in children]
        elif children is None:
            node["children_ids"] = []
        # Drop legacy length_strategy field so downstream validator only sees
        # canonical length_of edges.
        if "length_strategy" in node:
            node.pop("length_strategy", None)


    # Coerce edge endpoints
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if "src" in edge:
            edge["src"] = _coerce_int(edge.get("src"))
        if "dst" in edge:
            edge["dst"] = _coerce_int(edge.get("dst"))

    node_ids = {
        node.get("node_id")
        for node in nodes
        if isinstance(node, dict) and node.get("node_id") is not None
    }
    filtered_edges = []
    dropped_edges = 0
    for edge in edges:
        if not isinstance(edge, dict):
            dropped_edges += 1
            continue
        src = edge.get("src")
        dst = edge.get("dst")
        if src not in node_ids or dst not in node_ids:
            dropped_edges += 1
            continue
        filtered_edges.append(edge)
    if dropped_edges:
        logger.debug("Dropped %d protocol-tree edges with missing endpoints", dropped_edges)
    tree["edges"] = filtered_edges
    normalize_count_length_edges(tree)

    return tree


def synchronize_parent_child_links(tree_like: Dict[str, Any]) -> bool:
    """Repair parent/child link drift by reconciling children_ids with parent_id.

    In the common case, ``parent_id`` is the source of truth because it is
    singular and tends to survive LLM generation/refinement even when a
    parent's ``children_ids`` list accidentally drops an entry.  When an LLM
    lists the same child under both a generic parent and one of that parent's
    descendants, however, the descendant is the more precise physical container;
    prefer it.  If sibling candidates disagree, keep the current parent.
    """

    tree = normalize_protocol_tree(tree_like)
    if not isinstance(tree, dict):
        return False

    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        return False

    lookup: Dict[Any, Dict[str, Any]] = {}
    declared_children: Dict[Any, List[Any]] = defaultdict(list)
    listed_parents: Dict[Any, List[Any]] = defaultdict(list)

    def _index_tree() -> None:
        lookup.clear()
        listed_parents.clear()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("node_id")
            if node_id is None:
                continue
            lookup[node_id] = node
            lookup[str(node_id)] = node
            children = node.get("children_ids")
            if not isinstance(children, list):
                node["children_ids"] = []
                children = node["children_ids"]
            normalized_children: List[Any] = []
            seen_children = set()
            for child_id in children:
                child_id = _coerce_int(child_id)
                if child_id is None or child_id in seen_children:
                    continue
                seen_children.add(child_id)
                normalized_children.append(child_id)
                listed_parents[child_id].append(node_id)
            node["children_ids"] = normalized_children

    def _rewrite_refs(value: Any, mapping: Dict[Any, Any]) -> Any:
        if isinstance(value, str):
            rewritten = value
            for old_id, new_id in sorted(mapping.items(), key=lambda item: len(str(item[0])), reverse=True):
                old_s = re.escape(str(old_id))
                new_s = str(new_id)
                rewritten = re.sub(rf"\b{old_s}\b(?=\.)", new_s, rewritten)
                rewritten = re.sub(rf"(?<=val\(){old_s}(?=\))", new_s, rewritten)
            return rewritten
        if isinstance(value, list):
            return [_rewrite_refs(item, mapping) for item in value]
        if isinstance(value, dict):
            return {key: _rewrite_refs(item, mapping) for key, item in value.items()}
        return value

    def _collect_subtree_ids(root_id: Any) -> Set[Any]:
        root_id = _coerce_int(root_id)
        collected: Set[Any] = set()
        pending = [root_id]
        while pending:
            node_id = _coerce_int(pending.pop())
            if node_id in collected:
                continue
            node = lookup.get(node_id) or lookup.get(str(node_id))
            if not isinstance(node, dict):
                continue
            collected.add(node_id)
            for child_id in node.get("children_ids") or []:
                pending.append(child_id)
        return collected

    def _next_node_id() -> int:
        numeric_ids = [
            int(node.get("node_id"))
            for node in nodes
            if isinstance(node, dict) and isinstance(_coerce_int(node.get("node_id")), int)
        ]
        return (max(numeric_ids) if numeric_ids else 0) + 1

    def _parents_have_specificity_relation(parents: List[Any]) -> bool:
        def _is_ancestor_local(ancestor_id: Any, descendant_id: Any) -> bool:
            ancestor_id = _coerce_int(ancestor_id)
            current_id = _coerce_int(descendant_id)
            seen = set()
            while current_id is not None and current_id not in seen:
                seen.add(current_id)
                node = lookup.get(current_id) or lookup.get(str(current_id))
                if not isinstance(node, dict):
                    return False
                parent_id = _coerce_int(node.get("parent_id"))
                if parent_id == ancestor_id:
                    return True
                current_id = parent_id
            return False

        for left in parents:
            for right in parents:
                if left == right:
                    continue
                if _is_ancestor_local(left, right) or _is_ancestor_local(right, left):
                    return True
        return False

    def _clone_shared_reusable_subtrees() -> bool:
        changed_local = False
        next_id = _next_node_id()
        edges = tree.get("edges")
        if not isinstance(edges, list):
            edges = []
            tree["edges"] = edges

        for child_id, raw_parents in list(listed_parents.items()):
            child_id = _coerce_int(child_id)
            child_node = lookup.get(child_id) or lookup.get(str(child_id))
            if not isinstance(child_node, dict):
                continue
            parents: List[Any] = []
            seen_parents = set()
            for parent_id in raw_parents:
                parent_id = _coerce_int(parent_id)
                if parent_id in seen_parents:
                    continue
                if parent_id not in lookup and str(parent_id) not in lookup:
                    continue
                seen_parents.add(parent_id)
                parents.append(parent_id)
            if len(parents) <= 1:
                continue

            subtree_ids = _collect_subtree_ids(child_id)
            if len(subtree_ids) <= 1:
                continue
            if len(subtree_ids) > _shared_subtree_clone_limit():
                logger.debug(
                    "Skipped cloning shared subtree rooted at %s with %d nodes; exceeds STEP2_SHARED_SUBTREE_CLONE_MAX_NODES",
                    child_id,
                    len(subtree_ids),
                )
                continue
            if _parents_have_specificity_relation(parents):
                continue

            current_parent = _coerce_int(child_node.get("parent_id"))
            if current_parent is not None and current_parent not in parents:
                continue
            first_parent = current_parent if current_parent in parents else parents[0]
            child_node["parent_id"] = first_parent
            first_parent_node = lookup.get(first_parent) or lookup.get(str(first_parent))
            if isinstance(first_parent_node, dict):
                first_children = first_parent_node.get("children_ids") or []
                first_parent_node["children_ids"] = [
                    child_id if _coerce_int(item) == child_id else item for item in first_children
                ]

            original_nodes = [node for node in nodes if isinstance(node, dict) and _coerce_int(node.get("node_id")) in subtree_ids]
            original_edges = [
                edge
                for edge in edges
                if isinstance(edge, dict)
                and _coerce_int(edge.get("src")) in subtree_ids
                and _coerce_int(edge.get("dst")) in subtree_ids
            ]

            for parent_id in parents[1:]:
                mapping: Dict[Any, Any] = {}
                for old_node in original_nodes:
                    old_id = _coerce_int(old_node.get("node_id"))
                    mapping[old_id] = next_id
                    next_id += 1

                root_old_parent = _coerce_int(child_node.get("parent_id"))
                ref_mapping = dict(mapping)
                if root_old_parent is not None:
                    ref_mapping[root_old_parent] = parent_id
                for candidate_parent_id in parents:
                    ref_mapping[_coerce_int(candidate_parent_id)] = parent_id

                cloned_nodes: List[Dict[str, Any]] = []
                for old_node in original_nodes:
                    old_id = _coerce_int(old_node.get("node_id"))
                    clone = copy.deepcopy(old_node)
                    clone = _rewrite_refs(clone, ref_mapping)
                    clone["node_id"] = mapping[old_id]
                    old_parent = _coerce_int(old_node.get("parent_id"))
                    clone["parent_id"] = parent_id if old_id == child_id else mapping.get(old_parent, old_parent)
                    clone["children_ids"] = [
                        mapping.get(_coerce_int(grandchild_id), _coerce_int(grandchild_id))
                        for grandchild_id in old_node.get("children_ids") or []
                    ]
                    cloned_nodes.append(clone)
                nodes.extend(cloned_nodes)

                cloned_root_id = mapping[child_id]
                parent_node = lookup.get(parent_id) or lookup.get(str(parent_id))
                if isinstance(parent_node, dict):
                    parent_children = parent_node.get("children_ids") or []
                    parent_node["children_ids"] = [
                        cloned_root_id if _coerce_int(item) == child_id else item for item in parent_children
                    ]

                for old_edge in original_edges:
                    edge = copy.deepcopy(old_edge)
                    edge = _rewrite_refs(edge, ref_mapping)
                    edge["src"] = mapping[_coerce_int(old_edge.get("src"))]
                    edge["dst"] = mapping[_coerce_int(old_edge.get("dst"))]
                    edges.append(edge)

            changed_local = True

        return changed_local

    _index_tree()
    changed = _clone_shared_reusable_subtrees()
    if changed:
        _index_tree()

    def _is_ancestor(ancestor_id: Any, descendant_id: Any) -> bool:
        ancestor_id = _coerce_int(ancestor_id)
        current_id = _coerce_int(descendant_id)
        seen = set()
        while current_id is not None and current_id not in seen:
            seen.add(current_id)
            node = lookup.get(current_id) or lookup.get(str(current_id))
            if not isinstance(node, dict):
                return False
            parent_id = _coerce_int(node.get("parent_id"))
            if parent_id == ancestor_id:
                return True
            current_id = parent_id
        return False

    def _choose_parent(node_id: Any, current_parent_id: Any) -> Any:
        current_parent_id = _coerce_int(current_parent_id)
        raw_candidates = listed_parents.get(node_id, [])
        candidates: List[Any] = []
        seen_candidates = set()
        for candidate in raw_candidates:
            candidate = _coerce_int(candidate)
            if candidate in seen_candidates:
                continue
            if candidate not in lookup and str(candidate) not in lookup:
                continue
            seen_candidates.add(candidate)
            candidates.append(candidate)
        if not candidates:
            return current_parent_id

        if current_parent_id in seen_candidates:
            descendant_candidates = [
                candidate
                for candidate in candidates
                if candidate != current_parent_id and _is_ancestor(current_parent_id, candidate)
            ]
            if len(descendant_candidates) == 1:
                return descendant_candidates[0]
            return current_parent_id

        if len(candidates) == 1:
            return candidates[0]

        deepest_candidates = [
            candidate
            for candidate in candidates
            if not any(
                candidate != other and _is_ancestor(candidate, other)
                for other in candidates
            )
        ]
        if len(deepest_candidates) == 1:
            return deepest_candidates[0]
        return current_parent_id

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if node_id is None:
            continue
        parent_id = _choose_parent(node_id, node.get("parent_id"))
        if node.get("parent_id") != parent_id:
            changed = True
        node["parent_id"] = parent_id
        if parent_id is None:
            continue
        parent = lookup.get(parent_id) or lookup.get(str(parent_id))
        if not isinstance(parent, dict):
            continue
        declared_children[parent.get("node_id")].append(node_id)

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if node_id is None:
            continue

        expected = declared_children.get(node_id, [])
        current = list(node.get("children_ids") or [])
        if not expected and not current:
            continue

        expected_set = set(expected)
        repaired: List[Any] = []
        seen = set()

        for child_id in current:
            child_id = _coerce_int(child_id)
            if child_id is None or child_id not in expected_set or child_id in seen:
                continue
            repaired.append(child_id)
            seen.add(child_id)

        for child_id in expected:
            if child_id in seen:
                continue
            repaired.append(child_id)
            seen.add(child_id)

        if repaired != current:
            node["children_ids"] = repaired
            changed = True

    return changed


_SELECTOR_ALLOWED_PARENT_TYPES = {"container", "protocol", "message"}
_SELECTOR_FALLBACK_PARENT_TYPE = "container"
_SELECTOR_PARENT_SKIP_TYPES = {"variant", "selector"}


def auto_fix_selector_parent_types(tree: Dict[str, Any]) -> bool:
    """Auto-upgrade parents of selector nodes to containers when necessary."""

    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        return False

    lookup: Dict[Any, Dict[str, Any]] = {}

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        candidates = {node_id, _coerce_int(node_id)}
        for candidate in list(candidates):
            if candidate is None:
                candidates.discard(candidate)
                continue
            candidates.add(str(candidate))
        for candidate in candidates:
            if candidate is None:
                continue
            lookup[candidate] = node

    changed = False

    for node in nodes:
        if not isinstance(node, dict):
            continue
        child_type = (node.get("node_type") or "").strip().lower()
        if child_type != "selector":
            continue

        parent_id = node.get("parent_id")
        parent = None
        for key in (parent_id, _coerce_int(parent_id), str(parent_id) if parent_id is not None else None):
            if key is None:
                continue
            parent = lookup.get(key)
            if parent:
                break
        if not parent:
            continue

        parent_type_raw = parent.get("node_type")
        parent_type = (parent_type_raw or "").strip().lower()

        if parent_type in _SELECTOR_ALLOWED_PARENT_TYPES:
            continue
        if parent_type in _SELECTOR_PARENT_SKIP_TYPES:
            logger.debug(
                "Auto-fix skipped for selector %s (ID:%s) with parent type %s",
                node.get("name", f"node_{node.get('node_id')}") or "selector",
                node.get("node_id"),
                parent_type or "<none>",
            )
            continue

        parent_name = parent.get("name", f"node_{parent.get('node_id')}")
        child_name = node.get("name", f"node_{node.get('node_id')}")

        logger.info(
            "Auto-fix: updated parent %s (ID:%s) node_type %s -> %s to host selector %s (ID:%s)",
            parent_name,
            parent.get("node_id"),
            parent_type_raw or "<unset>",
            _SELECTOR_FALLBACK_PARENT_TYPE,
            child_name,
            node.get("node_id"),
        )

        parent["node_type"] = _SELECTOR_FALLBACK_PARENT_TYPE
        changed = True

    return changed
