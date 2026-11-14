from __future__ import annotations

"""
Core IR canonicalization utilities.

This module normalizes protocol tree dictionaries emitted by LLMs into a
stable, constrained subset that downstream validators and fixers can rely on.
"""

from typing import Dict, Any, List
import copy
import re
import logging

from pargen.protocol_knowledge.tree_utils import normalize_protocol_tree

# Allowed enums
# NOTE: `tlv_seq` is a container-like node_type representing repeated TLV items.
# It must be preserved during canonicalization; otherwise Stage B will silently
# downgrade it to `field` and lose the intended sequence semantics.
_ALLOWED_NODE_TYPES = {
    "protocol",
    "header",
    "payload",
    "field",
    "selector",
    "variant",
    "container",
    "message",
    "tlv_seq",
}
_ALLOWED_MESSAGE_TYPES = {"bidirectional", "request", "response"}
_SIMPLE_SIZE_RE = re.compile(r"^val\([^)]+\)\s*\*\s*8$")
# A conservative whitelist for size_bits arithmetic expressions (used by both the interpreter and Z3).
# NOTE: Use single backslashes so `\s` is interpreted as whitespace by the regex engine.
_SAFE_EXPR_RE = re.compile(r"^[0-9A-Za-z_\s\(\)\.\+\-\*/%<>=!&|,]+$")
_VAL_EQ_CONST_RE = re.compile(
    r"^val\(\s*(?P<id>-?\d+)\s*\)\s*==\s*(?P<const>0x[0-9a-fA-F]+|\d+)\s*$"
)
_END_NAME_RE = re.compile(r"(?:^|[^a-z])(end|terminat|eom|eof|stop)(?:[^a-z]|$)", re.IGNORECASE)
_PAD_NAME_RE = re.compile(r"(?:^|[^a-z])(pad|padding)(?:[^a-z]|$)", re.IGNORECASE)
_BANNED_SIZE_SYMBOLS = {"total_bits", "parent.", "remaining_bits"}
_SCALAR_NODE_TYPES = {"field", "selector", "length"}


def _coerce_int(value: Any) -> Any:
    try:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return int(candidate, 0)
    except Exception:
        return value
    return value


def _normalize_span_expr(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", "", value.strip())
    return str(value)


def _same_static_size(a: Any, b: Any) -> bool:
    a_int = _coerce_int(a)
    b_int = _coerce_int(b)
    if isinstance(a_int, int) and isinstance(b_int, int):
        return int(a_int) == int(b_int)
    return _normalize_span_expr(a) == _normalize_span_expr(b)


def _same_scalar_span(parent: Dict[str, Any], child: Dict[str, Any]) -> bool:
    parent_id = parent.get("node_id")
    parent_start = parent.get("bit_start")
    child_start = child.get("bit_start")
    if _normalize_span_expr(parent_start) == _normalize_span_expr(child_start):
        return True
    if parent_id is None:
        return False
    child_norm = _normalize_span_expr(child_start)
    parent_refs = {
        f"{parent_id}.bit_start",
        f"{parent_id}.bit_start+0",
        f"({parent_id}.bit_start)",
        f"({parent_id}.bit_start)+0",
    }
    return child_norm in parent_refs


def prune_duplicate_scalar_leaf_children(protocol_tree: Dict[str, Any]) -> Dict[str, Any]:
    """Remove scalar leaf aliases that exactly duplicate their scalar parent span.

    A scalar field cannot physically contain another full-width scalar field at
    the same offset. LLM-generated trees sometimes create this as an alias node
    (for example `Magic_Cookie` containing `magic_cookie` at `parent.bit_start`).
    Keeping both nodes makes the runtime parser consume the same wire span twice.
    """

    nodes = protocol_tree.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return protocol_tree

    node_by_id: Dict[Any, Dict[str, Any]] = {
        n.get("node_id"): n for n in nodes if isinstance(n, dict) and n.get("node_id") is not None
    }
    node_by_id.update({str(k): v for k, v in list(node_by_id.items())})

    edges = protocol_tree.get("edges")
    edge_refs: set[Any] = set()
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            for key in ("src", "dst"):
                ref = edge.get(key)
                if ref is not None:
                    edge_refs.add(ref)
                    edge_refs.add(str(ref))

    remove_ids: set[Any] = set()
    for parent in list(nodes):
        if not isinstance(parent, dict):
            continue
        parent_type = str(parent.get("node_type") or "").strip().lower()
        if parent_type not in _SCALAR_NODE_TYPES:
            continue
        children = parent.get("children_ids")
        if not isinstance(children, list) or not children:
            continue

        kept_children: List[Any] = []
        changed = False
        for child_id in children:
            child = node_by_id.get(child_id) or node_by_id.get(str(child_id))
            if not isinstance(child, dict):
                kept_children.append(child_id)
                continue
            canonical_child_id = child.get("node_id")
            if canonical_child_id in edge_refs or str(canonical_child_id) in edge_refs:
                kept_children.append(child_id)
                continue
            child_type = str(child.get("node_type") or "").strip().lower()
            child_children = child.get("children_ids") or []
            if (
                child_type in _SCALAR_NODE_TYPES
                and not child_children
                and _same_scalar_span(parent, child)
                and _same_static_size(parent.get("size_bits"), child.get("size_bits"))
            ):
                parent_constraints = parent.setdefault("constraints", [])
                if not isinstance(parent_constraints, list):
                    parent_constraints = [parent_constraints]
                    parent["constraints"] = parent_constraints
                for constraint in child.get("constraints") or []:
                    if constraint not in parent_constraints:
                        parent_constraints.append(constraint)
                remove_ids.add(canonical_child_id)
                remove_ids.add(str(canonical_child_id))
                changed = True
                continue
            kept_children.append(child_id)

        if changed:
            parent["children_ids"] = kept_children

    if remove_ids:
        protocol_tree["nodes"] = [
            n
            for n in nodes
            if not (isinstance(n, dict) and (n.get("node_id") in remove_ids or str(n.get("node_id")) in remove_ids))
        ]
        if isinstance(edges, list):
            protocol_tree["edges"] = [
                edge
                for edge in edges
                if not (
                    isinstance(edge, dict)
                    and (
                        edge.get("src") in remove_ids
                        or str(edge.get("src")) in remove_ids
                        or edge.get("dst") in remove_ids
                        or str(edge.get("dst")) in remove_ids
                    )
                )
            ]

    return protocol_tree


def canonicalize_protocol_tree(protocol_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an LLM-generated protocol tree into a constrained Core IR subset.

    This function may modify the input dict in place; callers that require
    immutability should pass a deep copy before invoking.
    """

    logger = logging.getLogger(__name__)
    try:
        tree = normalize_protocol_tree(copy.deepcopy(protocol_tree))
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "canonicalize_protocol_tree failed to normalize tree; returning input as-is: %s",
            exc,
            exc_info=True,
        )
        tree = copy.deepcopy(protocol_tree)
    nodes: List[Dict[str, Any]] = tree.get("nodes", [])

    # Ensure a valid root
    node_by_id: Dict[Any, Dict[str, Any]] = {
        n.get("node_id"): n for n in nodes if isinstance(n, dict) and n.get("node_id") is not None
    }

    root_id = tree.get("root_node_id")
    if root_id not in node_by_id:
        # Pick the first node with no parent; fallback to first node.
        candidate = next((n for n in nodes if n.get("parent_id") in (None, -1)), nodes[0] if nodes else None)
        if candidate:
            root_id = candidate.get("node_id")
            tree["root_node_id"] = root_id
    if root_id in node_by_id:
        node_by_id[root_id]["node_type"] = "protocol"

    # Node normalization
    for node in nodes:
        # node_type normalization
        raw_type = (node.get("node_type") or "").strip().lower()
        if raw_type not in _ALLOWED_NODE_TYPES:
            name_desc = f"{node.get('name','')} {node.get('description','')}".lower()
            if "header" in name_desc:
                node["node_type"] = "header"
            elif "payload" in name_desc or "pdu" in name_desc:
                node["node_type"] = "payload"
            elif "message" in name_desc:
                node["node_type"] = "message"
            else:
                node["node_type"] = "field"
        else:
            node["node_type"] = raw_type

        # message_type normalization
        if "message_type" in node:
            mt = (node.get("message_type") or "").strip().lower()
            node["message_type"] = mt if mt in _ALLOWED_MESSAGE_TYPES else "bidirectional"

        # size_bits normalization
        size_bits = node.get("size_bits")
        mark_needs_binding = False
        if isinstance(size_bits, int):
            pass
        elif isinstance(size_bits, str):
            stripped = size_bits.strip()
            if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
                node["size_bits"] = stripped
            elif _SIMPLE_SIZE_RE.match(stripped.replace(" ", "")):
                # keep normalized spacing
                node["size_bits"] = re.sub(r"\\s+", " ", stripped)
            elif (
                _SAFE_EXPR_RE.match(stripped)
                and ":" not in stripped
                and not any(sym in stripped.lower() for sym in _BANNED_SIZE_SYMBOLS)
            ):
                # Preserve general arithmetic expressions for size_bits (e.g., "0.size_bits - 20.bit_start").
                node["size_bits"] = re.sub(r"\\s+", " ", stripped)
            else:
                node["size_bits"] = "variable"
                mark_needs_binding = True
        else:
            node["size_bits"] = "variable"
            mark_needs_binding = True
        if mark_needs_binding:
            node["needs_length_binding"] = True

    tree = prune_duplicate_scalar_leaf_children(tree)
    nodes = tree.get("nodes", [])
    node_by_id = {
        n.get("node_id"): n for n in nodes if isinstance(n, dict) and n.get("node_id") is not None
    }

    def _coerce_int_or_none(val: Any) -> Any:
        try:
            if isinstance(val, bool):
                return None
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.strip():
                return int(val.strip(), 0)
        except Exception:
            return None
        return None

    def _node_name(nid: Any) -> str:
        node = node_by_id.get(nid) or node_by_id.get(str(nid))
        if not isinstance(node, dict):
            return ""
        return str(node.get("name") or "")

    def _node_type(nid: Any) -> str:
        node = node_by_id.get(nid) or node_by_id.get(str(nid))
        if not isinstance(node, dict):
            return ""
        return str(node.get("node_type") or "").strip().lower()

    def _node_children(nid: Any) -> List[Any]:
        node = node_by_id.get(nid) or node_by_id.get(str(nid))
        if not isinstance(node, dict):
            return []
        kids = node.get("children_ids") or []
        return list(kids) if isinstance(kids, list) else []

    def _is_zero_size_variant(nid: Any) -> bool:
        node = node_by_id.get(nid) or node_by_id.get(str(nid))
        if not isinstance(node, dict):
            return False
        if str(node.get("node_type") or "").strip().lower() != "variant":
            return False
        if node.get("children_ids"):
            return False
        sb = node.get("size_bits")
        if sb == 0:
            return True
        if isinstance(sb, str) and sb.strip() in {"0", "+0", "-0"}:
            return True
        return False

    def _parse_val_eq_const(formula: Any) -> Any:
        if not isinstance(formula, str):
            return None
        stripped = formula.strip()
        m = _VAL_EQ_CONST_RE.match(stripped)
        if not m:
            return None
        try:
            selector_id = int(m.group("id"), 0)
            const_val = int(m.group("const"), 0)
        except Exception:
            return None
        return selector_id, const_val, stripped

    def _collect_descendants(root: Any, children_map: Dict[Any, List[Dict[str, Any]]]) -> set[Any]:
        seen: set[Any] = set()
        stack: List[Any] = [root]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for child in children_map.get(cur, []) or []:
                cid = child.get("node_id")
                if cid is None:
                    continue
                stack.append(cid)
        return seen

    def _infer_stop_condition_for_tlv_seq(tlv_id: Any, descendants: set[Any], edges: List[Dict[str, Any]]) -> Any:
        # Find "end"-like condition_on edges inside tlv_seq subtree.
        candidates: List[tuple[int, int, str, str]] = []  # (score, const, formula, variant_name)
        for e in edges:
            if not isinstance(e, dict):
                continue
            if str(e.get("rel") or "").strip() != "condition_on":
                continue
            dst = e.get("dst")
            if dst is None or dst not in descendants:
                continue
            if not _is_zero_size_variant(dst):
                continue
            parsed = _parse_val_eq_const(e.get("formula"))
            if not parsed:
                continue
            _sel, const_val, formula_str = parsed
            vname = _node_name(dst)
            score = 0
            if _END_NAME_RE.search(vname) and not _PAD_NAME_RE.search(vname):
                score += 100
            if const_val != 0:
                score += 10
            candidates.append((score, const_val, formula_str, vname))

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best = candidates[0]
        if best[0] >= 100:
            return best[2]

        # Fallback: if we have both a 0-valued marker and some other marker, pick the max.
        consts = {c[1] for c in candidates}
        if 0 in consts and len(consts) >= 2:
            best = max(candidates, key=lambda t: t[1])
            return best[2]

        return None

    edges_raw = tree.get("edges", [])
    edges_list: List[Dict[str, Any]] = edges_raw if isinstance(edges_raw, list) else []
    root_id = tree.get("root_node_id")

    # -------- selector/variant physical containment fix (protocol-agnostic) --------
    # Some generated trees incorrectly attach variable-size variants as children of the selector byte.
    # Since `parent_id`/`children_ids` model physical containment, a selector (e.g., 8-bit tag) cannot
    # contain a variant whose `bit_start` is *after* the selector. When we detect the common pattern
    # (variant.bit_start == selector.bit_start + selector.size_bits), reparent the variant under the
    # selector's parent container to avoid "may extend beyond parent selector" static errors.
    for e in edges_list:
        if not isinstance(e, dict) or str(e.get("rel") or "").strip() != "condition_on":
            continue
        sid = _coerce_int_or_none(e.get("src"))
        vid = _coerce_int_or_none(e.get("dst"))
        if sid is None or vid is None:
            continue
        sel = node_by_id.get(sid) or node_by_id.get(str(sid))
        var = node_by_id.get(vid) or node_by_id.get(str(vid))
        if not isinstance(sel, dict) or not isinstance(var, dict):
            continue
        if str(sel.get("node_type") or "").strip().lower() != "selector":
            continue
        if str(var.get("node_type") or "").strip().lower() != "variant":
            continue
        if _coerce_int_or_none(var.get("parent_id")) != int(sid):
            continue

        sel_size = _coerce_int_or_none(sel.get("size_bits"))
        if sel_size is None or int(sel_size) <= 0:
            continue
        var_bs = var.get("bit_start")
        if not isinstance(var_bs, str) or not var_bs.strip():
            continue
        var_bs_norm = var_bs.strip().replace(" ", "")
        after_sel_norm = f"{int(sid)}.bit_start+{int(sel_size)}"
        if var_bs_norm != after_sel_norm:
            continue

        parent_id = _coerce_int_or_none(sel.get("parent_id"))
        if parent_id is None:
            continue
        parent = node_by_id.get(parent_id) or node_by_id.get(str(parent_id))
        if not isinstance(parent, dict):
            continue

        # Remove from selector children_ids.
        sel_children = sel.get("children_ids")
        if isinstance(sel_children, list) and sel_children:
            sel["children_ids"] = [c for c in sel_children if _coerce_int_or_none(c) != int(vid)]

        # Add to parent children_ids (prefer right-after selector for stability).
        parent_children = parent.get("children_ids")
        if not isinstance(parent_children, list):
            parent_children = []
            parent["children_ids"] = parent_children
        if int(vid) not in {_coerce_int_or_none(c) for c in parent_children}:
            try:
                idx = next(i for i, c in enumerate(parent_children) if _coerce_int_or_none(c) == int(sid))
                parent_children.insert(idx + 1, int(vid))
            except StopIteration:
                parent_children.append(int(vid))

        var["parent_id"] = int(parent_id)

    # Bit start monotonic fix within each parent
    children_map: Dict[Any, List[Dict[str, Any]]] = {}
    for node in nodes:
        parent_id = node.get("parent_id")
        children_map.setdefault(parent_id, []).append(node)

    # -------- tlv_seq stop_condition + trailing padding modelling (protocol-agnostic) --------
    tlv_nodes = [
        n
        for n in nodes
        if isinstance(n, dict) and str(n.get("node_type") or "").strip().lower() == "tlv_seq"
    ]
    if tlv_nodes:
        # Compute max numeric node_id for deterministic new node ids.
        max_node_id = -1
        for n in nodes:
            nid_int = _coerce_int_or_none(n.get("node_id") if isinstance(n, dict) else None)
            if isinstance(nid_int, int) and nid_int > max_node_id:
                max_node_id = nid_int

        for tlv in tlv_nodes:
            tlv_id = tlv.get("node_id")
            if tlv_id is None:
                continue
            stop_formula = tlv.get("stop_condition") or tlv.get("repeat_until")
            if not stop_formula and edges_list:
                descendants = _collect_descendants(tlv_id, children_map)
                inferred = _infer_stop_condition_for_tlv_seq(tlv_id, descendants, edges_list)
                if inferred:
                    stop_formula = inferred
                    tlv["stop_condition"] = inferred

            # Only add trailing padding when the sequence stops on an explicit terminator.
            if not stop_formula:
                continue

            # If the tlv_seq is the last child, and we can bound the parent size, add a trailing Padding payload.
            parent_id = tlv.get("parent_id")
            if parent_id is None:
                continue
            parent = node_by_id.get(parent_id) or node_by_id.get(str(parent_id))
            if not isinstance(parent, dict):
                continue
            parent_children = parent.get("children_ids")
            if not isinstance(parent_children, list) or tlv_id not in parent_children:
                continue
            if parent_children and parent_children[-1] != tlv_id:
                continue

            # Avoid duplicating an existing padding-like sibling.
            sibling_nodes = [_node_name(cid).lower() for cid in parent_children if cid != tlv_id]
            if any(_PAD_NAME_RE.search(name) for name in sibling_nodes):
                continue

            # Only add padding when the parent has a resolvable size (root is always resolvable at runtime).
            parent_size = parent.get("size_bits")
            parent_size_is_variable = isinstance(parent_size, str) and parent_size.strip().lower() in {
                "variable",
                "unknown",
                "dynamic",
                "",
            }
            if parent_id != root_id and parent_size_is_variable:
                continue

            max_node_id += 1
            padding_id = max_node_id
            padding_node = {
                "node_id": int(padding_id),
                "name": "Padding",
                "node_type": "payload",
                "description": "Trailing bytes after TLV end marker (padding)",
                "bit_start": f"{tlv_id}.bit_start + {tlv_id}.size_bits",
                "size_bits": f"({parent_id}.bit_start + {parent_id}.size_bits) - {padding_id}.bit_start",
                "data_type": "binary",
                "byte_order": parent.get("byte_order") or tlv.get("byte_order") or "big",
                "message_type": parent.get("message_type") or tlv.get("message_type") or "bidirectional",
                "constraints": [],
                "dependencies": [],
                "parent_id": parent_id,
                "children_ids": [],
                "source": "",
            }
            nodes.append(padding_node)
            node_by_id[padding_id] = padding_node
            node_by_id[str(padding_id)] = padding_node
            parent_children.append(padding_id)
            children_map.setdefault(parent_id, []).append(padding_node)

    # `tlv_seq` models a repeated item template, where child `bit_start`/layout is
    # interpreted relative to the runtime cursor on each iteration. Any static
    # "monotonic layout" rewrite here would incorrectly freeze those children to
    # a single absolute offset (and can cause the parser to rewind/stuck).
    tlv_seq_ids = {
        n.get("node_id")
        for n in nodes
        if isinstance(n, dict) and str(n.get("node_type") or "").strip().lower() == "tlv_seq"
    }
    tlv_scoped_nodes: set[Any] = set(tlv_seq_ids)
    if tlv_seq_ids:
        stack: List[Any] = list(tlv_seq_ids)
        while stack:
            current_id = stack.pop()
            for child in children_map.get(current_id, []) or []:
                child_id = child.get("node_id")
                if child_id is None or child_id in tlv_scoped_nodes:
                    continue
                tlv_scoped_nodes.add(child_id)
                stack.append(child_id)

    for parent_id, kids in children_map.items():
        if parent_id is not None and parent_id in tlv_scoped_nodes:
            # Skip the monotonic layout fix for `tlv_seq` subtrees.
            continue
        parent = node_by_id.get(parent_id)
        base = 0
        if parent is not None:
            p_bs = parent.get("bit_start")
            if isinstance(p_bs, int) and p_bs >= 0:
                base = p_bs

        # sort by current bit_start if present
        def _key(n: Dict[str, Any]) -> int:
            bs = n.get("bit_start")
            return bs if isinstance(bs, int) and bs >= 0 else 10**12

        # IMPORTANT: Variants are mutually-exclusive branches and are allowed to
        # overlap with each other (and should not be laid out sequentially).
        # We therefore apply monotonic layout only to non-variant siblings, and
        # then clamp variants to start *after* the shared prefix (selectors/headers)
        # without letting one variant shift the others.

        non_variants = [k for k in kids if str(k.get("node_type") or "").lower() != "variant"]
        variants = [k for k in kids if str(k.get("node_type") or "").lower() == "variant"]

        current = base
        for child in sorted(non_variants, key=_key):
            bs = child.get("bit_start")
            if isinstance(bs, int):
                if bs < current:
                    child["bit_start"] = current
            elif isinstance(bs, str):
                # Preserve non-numeric expressions. Only coerce purely numeric strings.
                stripped = bs.strip()
                if stripped and (stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit())):
                    try:
                        bs_int = int(stripped, 0)
                    except Exception:
                        bs_int = None
                    if bs_int is None or bs_int < current:
                        child["bit_start"] = current
                    else:
                        child["bit_start"] = bs_int
            else:
                child["bit_start"] = current

            sb = child.get("size_bits")
            sb_int = None
            if isinstance(sb, int):
                sb_int = sb
            elif isinstance(sb, str) and sb.strip().lstrip("-").isdigit():
                try:
                    sb_int = int(sb.strip(), 0)
                except Exception:
                    sb_int = None

            if sb_int is not None and sb_int >= 0:
                current += sb_int

        for child in sorted(variants, key=_key):
            bs = child.get("bit_start")
            if isinstance(bs, int):
                if bs < current:
                    child["bit_start"] = current
            elif isinstance(bs, str):
                stripped = bs.strip()
                if stripped and (stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit())):
                    try:
                        bs_int = int(stripped, 0)
                    except Exception:
                        bs_int = None
                    if bs_int is None or bs_int < current:
                        child["bit_start"] = current
                    else:
                        child["bit_start"] = bs_int
                # Non-numeric expressions are preserved.
            else:
                child["bit_start"] = current

    return tree


def add_request_response_variants(protocol_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for splitting payload into request/response variants.

    Currently returns the tree unchanged; kept for optional feature toggling.
    """

    return protocol_tree
