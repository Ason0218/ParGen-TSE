"""Detect erased bindings and ambiguous routing in protocol representations."""
from __future__ import annotations

import ast
import copy
import re
from typing import Any


REFINEMENT_CONTRACT_VERSION = "lossless-bindings-explicit-routing-v1"


def canonical_formula(value: Any) -> str:
    """Normalize syntax only: whitespace, parentheses and numeric bases."""
    text = str(value if value is not None else "").strip()
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"\b(AND|OR|NOT)\b", lambda m: m[0].lower(), text)
    # Numeric node-property references are DSL syntax, not Python decimals.
    text = re.sub(r"\b(\d+)\.(size_bits|bit_start)\b", r"node_\1_\2", text)
    try:
        return ast.dump(ast.parse(text.strip(), mode="eval"), include_attributes=False)
    except (SyntaxError, ValueError):
        return re.sub(r"\s+", "", text)


def explicit_layout(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {
        "", "variable", "var", "unknown", "dynamic", "none", "*",
    }


def reconcile_expanded_placeholders(tree: dict) -> tuple[dict, list[str]]:
    """Remove only empty duplicate routes fully replaced by populated siblings.

    Predicate, selector, parent and start must match exactly; both directions
    must be covered when replacing a bidirectional shell. No protocol names,
    external rules or packet values participate in this normalization.
    """
    result = copy.deepcopy(tree)
    nodes, edges = result.get('nodes', []), result.get('edges', [])
    removed, changes = set(), []

    def directions(value):
        return {'request', 'response'} if value in (None, '', 'bidirectional') else {value}

    def routes(node):
        return [edge for edge in edges if edge.get('rel') == 'condition_on'
                and str(edge.get('dst')) == str(node.get('node_id'))]

    for node in nodes:
        nid = str(node.get('node_id'))
        if (node.get('node_type') != 'variant' or node.get('children_ids')
                or node.get('constraints') or node.get('dependencies')
                or explicit_layout(node.get('size_bits'))
                or any(str(other.get('parent_id')) == nid for other in nodes)):
            continue
        incoming = routes(node)
        if len(incoming) != 1:
            continue
        old = incoming[0]
        if not old.get('formula') or old.get('src') is None:
            continue
        if any(str(edge.get('src')) == nid or
               (str(edge.get('dst')) == nid and edge is not old) for edge in edges):
            continue
        reference = re.compile(r'(?<![\w])' + re.escape(nid) + r'\.(?:bit_start|size_bits)|val\(\s*' + re.escape(nid) + r'\s*\)')
        if any(reference.search(str(other.get(key, ''))) for other in nodes if other is not node
               for key in ('bit_start', 'size_bits', 'constraints', 'dependencies')):
            continue
        if any(reference.search(str(edge.get('formula', ''))) for edge in edges):
            continue
        needed = directions(old.get('message_type')) & directions(node.get('message_type'))
        covered, replacements = set(), []
        for other in nodes:
            if (other is node or other.get('node_type') != 'variant' or not other.get('children_ids')
                    or str(other.get('parent_id')) != str(node.get('parent_id'))
                    or canonical_formula(other.get('bit_start')) != canonical_formula(node.get('bit_start'))):
                continue
            candidate_routes = routes(other)
            if len(candidate_routes) != 1:
                continue
            edge = candidate_routes[0]
            if (str(edge.get('src')) != str(old.get('src'))
                    or canonical_formula(edge.get('formula')) != canonical_formula(old.get('formula'))):
                continue
            covered |= directions(edge.get('message_type')) & directions(other.get('message_type'))
            replacements.append(other)
        if not needed or not needed <= covered:
            continue
        removed.add(nid)
        for other in replacements:
            other.setdefault('replaced_placeholder_sources', []).append(dict(
                node_id=node.get('node_id'), source=node.get('source'),
                source_section_ids=copy.deepcopy(node.get('source_section_ids') or [])))
        changes.append('replaced empty route %s with %s' %
                       (nid, ','.join(str(other['node_id']) for other in replacements)))
    if removed:
        result['nodes'] = [node for node in nodes if str(node.get('node_id')) not in removed]
        for node in result['nodes']:
            node['children_ids'] = [value for value in node.get('children_ids', []) if str(value) not in removed]
        result['edges'] = [edge for edge in edges if str(edge.get('src')) not in removed
                           and str(edge.get('dst')) not in removed]
    return result, changes


def _nodes(tree):
    return {str(n["node_id"]): n for n in tree.get("nodes", []) if isinstance(n, dict) and "node_id" in n}


def binding_scope(tree: dict, node_id: Any) -> tuple:
    """Execution context, including guarded ancestors and message direction."""
    nodes = _nodes(tree)
    current = str(node_id)
    path, seen = [], set()
    while current in nodes and current not in seen:
        seen.add(current)
        node = nodes[current]
        guards = sorted(
            (str(e.get("src")), canonical_formula(e.get("formula")),
             str(e.get("message_type") or node.get("message_type") or "bidirectional").lower())
            for e in tree.get("edges", [])
            if e.get("rel") == "condition_on" and str(e.get("dst")) == current
        )
        direction = str(node.get("message_type") or "bidirectional").lower()
        if guards or direction != "bidirectional":
            path.append((tuple(guards), direction))
        parent = node.get("parent_id")
        if parent is None:
            break
        current = str(parent)
    if current in seen and nodes.get(current, {}).get("parent_id") is not None:
        path.append(((("cycle", current, "bidirectional"),), "bidirectional"))
    return tuple(reversed(path))


def scope_preserved(baseline: dict, candidate: dict, node_id: Any) -> bool:
    before, after = binding_scope(baseline, node_id), binding_scope(candidate, node_id)
    if before == after:
        return True
    # Binding an explicitly diagnosed orphan is a repair, not moving a valid
    # field to a different branch. Existing guards/directions must still hold.
    orphan_ids = {str(i["node_id"]) for i in unbound_variant_issues(baseline) if i["code"] == "UNBOUND_SELECTOR_VARIANT"}
    nodes, current, seen = _nodes(baseline), str(node_id), set()
    under_orphan = False
    while current in nodes and current not in seen:
        seen.add(current)
        under_orphan |= current in orphan_ids
        current = str(nodes[current].get("parent_id"))
    if not under_orphan:
        return False
    old_guards = {g for guards, _direction in before for g in guards}
    new_guards = {g for guards, _direction in after for g in guards}
    old_directions = {d for _guards, d in before if d != "bidirectional"}
    new_directions = {d for _guards, d in after if d != "bidirectional"}
    return old_guards <= new_guards and old_directions == new_directions


def _rerouted_empty_variant(nid: str, baseline: dict, candidate: dict) -> bool:
    before = _nodes(baseline).get(nid, {})
    if before.get("node_type") != "variant" or before.get("children_ids") or before.get("constraints"):
        return False
    if any(str(n.get("parent_id")) == nid for n in _nodes(baseline).values()):
        return False
    incoming = [e for e in baseline.get("edges", []) if str(e.get("dst")) == nid]
    if not incoming or any(e.get("rel") != "condition_on" for e in incoming):
        return False
    nodes = _nodes(candidate)
    for old in incoming:
        if not any(e.get("rel") == "condition_on" and str(e.get("src")) == str(old.get("src"))
                   and canonical_formula(e.get("formula")) == canonical_formula(old.get("formula"))
                   and str(e.get("message_type") or "bidirectional") == str(old.get("message_type") or "bidirectional")
                   and str(e.get("dst")) != nid and nodes.get(str(e.get("dst")), {}).get("children_ids")
                   and str(nodes[str(e["dst"])].get("parent_id")) == str(before.get("parent_id"))
                   for e in candidate.get("edges", [])):
            return False
    return True


def layout_binding_losses(baseline: dict, candidate: dict) -> list[str]:
    """Reject erasure, not all edits: explicit formula corrections remain possible.

    A size expression may migrate to an equivalent length edge in the same
    execution scope. Replacing a dynamic binding by a sampled constant is loss.
    Arbitrary node-ID renumbering is deliberately not guessed from field names.
    """
    base_nodes, cand_nodes = _nodes(baseline), _nodes(candidate)
    edges = candidate.get("edges", [])
    losses = []
    for nid, before in base_nodes.items():
        after = cand_nodes.get(nid, {})
        if not after and _rerouted_empty_variant(nid, baseline, candidate):
            continue
        for field in ("size_bits", "bit_start"):
            original = before.get(field)
            inferred = (before.get("inferred_layout_fields") or {}).get(field, {})
            if inferred.get("origin") == "traffic_hint_prepass" and inferred.get("value") == original:
                continue
            if not explicit_layout(original):
                continue
            replacement = after.get(field)
            dynamic = bool(re.search(r"val\s*\(|\d+\.(?:size_bits|bit_start)", str(original)))
            replacement_dynamic = bool(re.search(r"val\s*\(|\d+\.(?:size_bits|bit_start)", str(replacement)))
            preserved = explicit_layout(replacement) and (not dynamic or replacement_dynamic)
            if field == "size_bits" and not preserved:
                preserved = any(
                    e.get("rel") == "length_of" and str(e.get("dst")) == nid
                    and canonical_formula(e.get("formula")) == canonical_formula(original)
                    for e in edges
                )
            if not preserved:
                losses.append(f"{nid}:{field}:binding_erased:{original}")
            elif not scope_preserved(baseline, candidate, nid):
                losses.append(f"{nid}:{field}:binding_scope_changed")

    for edge in baseline.get("edges", []):
        if edge.get("origin") == "traffic_hint_prepass":
            continue
        rel, src, dst = edge.get("rel"), str(edge.get("src")), str(edge.get("dst"))
        if rel not in {"length_of", "condition_on", "repeat_count"}:
            continue
        formula = canonical_formula(edge.get("formula"))
        direction = str(edge.get("message_type") or "bidirectional").lower()
        matching = [e for e in edges if e.get("rel") == rel and str(e.get("src")) == src
                    and str(e.get("message_type") or "bidirectional").lower() == direction]
        if rel == "condition_on":
            # Explicit routing may be reconnected from an EMPTY placeholder to
            # a populated branch. Do not allow dropping/changing its guard.
            before = base_nodes.get(dst, {})
            empty = not before.get("children_ids") and not before.get("constraints")
            kept = any(canonical_formula(e.get("formula")) == formula and
                       (str(e.get("dst")) == dst or (empty and
                        str(e.get("dst")) in cand_nodes and
                        str(cand_nodes[str(e.get("dst"))].get("parent_id")) == str(before.get("parent_id"))))
                       for e in matching)
        else:
            kept = any(str(e.get("dst")) == dst and explicit_layout(e.get("formula")) for e in matching)
            if not kept and rel == "length_of":
                kept = dst in cand_nodes and canonical_formula(cand_nodes[dst].get("size_bits")) == formula
            if kept and (not scope_preserved(baseline, candidate, dst) or not scope_preserved(baseline, candidate, src)):
                losses.append(f"edge:{rel}:{src}->{dst}:binding_scope_changed")
        if not kept:
            losses.append(f"edge:{rel}:{src}->{dst}:binding_erased")
    return sorted(set(losses))


def unbound_variant_issues(tree: dict) -> list[dict]:
    """Find implicit branches inside explicitly routed sibling groups.

    Empty protocol branches are legal. A default branch is legal when encoded
    as an explicit condition_on edge with an empty predicate. We never guess
    which selector value belongs to an unbound branch from its display name.
    """
    nodes = _nodes(tree)
    controlled = {str(e.get("dst")) for e in tree.get("edges", []) if e.get("rel") == "condition_on"}
    routed_parents = {str(nodes[nid].get("parent_id")) for nid in controlled if nid in nodes}
    issues = []
    for edge in tree.get("edges", []):
        if edge.get("rel") != "condition_on" or str(edge.get("dst")) not in nodes:
            continue
        target = nodes[str(edge["dst"])]
        direction = str(edge.get("message_type") or "bidirectional").lower()
        target_direction = str(target.get("message_type") or "bidirectional").lower()
        if direction != "bidirectional" and target_direction != "bidirectional" and direction != target_direction:
            issues.append({"node_id": target["node_id"], "code": "SELECTOR_DIRECTION_MISMATCH",
                           "description": "condition_on message direction conflicts with the selected branch direction. "
                           "Resolve request/response routing from the source; do not infer it from display names."})
    for nid, node in nodes.items():
        parent = nodes.get(str(node.get("parent_id")), {})
        if str(node.get("node_type", "")).lower() != "variant" or nid in controlled:
            continue
        if str(node.get("parent_id")) not in routed_parents and parent.get("node_type") != "selector":
            continue
        issues.append({"node_id": node["node_id"], "code": "UNBOUND_SELECTOR_VARIANT",
                       "description": "Variant has no condition_on edge in a selector-routed sibling group. "
                       "Use the source document to connect the correct selector, numeric guard and message direction; "
                       "an intended default needs an explicit empty guard. Do not infer a guard from the node name "
                       "or delete the unbound fields to silence this issue."})
    return issues
