import argparse
import hashlib
import os
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Dict, Any, List, Tuple, Optional, Set
from copy import deepcopy
from collections import defaultdict, Counter

from pargen.logging_utils import setup_logging
from .agent import EnhancedPureAIAgent
from .refinement import mcts_fix_tree, run_full_validation
from ...paths import STEP2_CACHE_DIR, STEP2_FIX_CACHE_DIR
from ..message_validation.semantic_validator import run_hybrid_validation
from ..validation.doc_constraint_coverage import (
    augment_report_with_doc_constraint_coverage,
    find_doc_structure_alignment_issues,
    find_uncovered_doc_constraints,
)
from ..validation.syntax_validator import is_hard_error, Severity
from ..semantic_integrity import canonical_formula, explicit_layout, layout_binding_losses
from ..refinement_result import publish_refinement
from ..static_layout import static_property


def _coerce_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _adaptive_doc_structure_issue_limit(
    tree_payload: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    *,
    max_doc_issues: int,
) -> int:
    """Choose a small repair window for large guarded-variant tables.

    When a doc table exposes many missing variants, one big prompt tends to
    produce a broad, invalid JSON patch.  Keeping only 1-2 variants visible per
    validator round gives the LLM a smaller, repeatable edit target while the
    reward still uses full uncovered counts.
    """

    default_limit = min(12, max(1, int(max_doc_issues)))
    try:
        issues = find_doc_structure_alignment_issues(tree_payload, sections)
    except Exception:
        return default_limit
    missing_variants = sum(1 for issue in issues if issue.get("kind") == "missing_guarded_variant")
    if missing_variants >= 6:
        return min(1, default_limit)
    if missing_variants >= 4:
        return min(3, default_limit)
    return default_limit


def _strip_outer_parens(text: str) -> str:
    if not text:
        return text
    stripped = text.strip()
    while stripped.startswith('(') and stripped.endswith(')'):
        inner = stripped[1:-1].strip()
        if not inner:
            break
        stripped = inner
    return stripped


def _rewrite_assignment_formula(text: str, src_id: int) -> Tuple[bool, str]:
    """
    Handle patterns like 'val(src) = 1 + (dst.size_bits / 8)' and convert them into
    expressions that do not use assignment syntax or dst.size_bits references.
    Returns (handled, rewritten_text).
    """
    assignment_pattern = re.compile(rf"^\s*val\(\s*{src_id}\s*\)\s*=\s*(.+)$", re.IGNORECASE)
    match = assignment_pattern.match(text)
    if not match:
        return False, text

    rhs = _strip_outer_parens(match.group(1))

    # Case 1: constant + (dst.size_bits / divisor)
    pattern_a = re.compile(
        r"^\s*(?P<const>-?\d+)\s*\+\s*\(\s*(?P<dst>\d+)\.size_bits\s*/\s*(?P<div>\d+)\s*\)\s*$",
        re.IGNORECASE,
    )
    m_a = pattern_a.match(rhs)
    if m_a:
        const = int(m_a.group('const'))
        divisor = int(m_a.group('div'))
        return True, f"(val({src_id}) - {const}) * {divisor}"

    # Case 2: (dst.size_bits / divisor) + constant
    pattern_b = re.compile(
        r"^\s*\(\s*(?P<dst>\d+)\.size_bits\s*/\s*(?P<div>\d+)\s*\)\s*\+\s*(?P<const>-?\d+)\s*$",
        re.IGNORECASE,
    )
    m_b = pattern_b.match(rhs)
    if m_b:
        const = int(m_b.group('const'))
        divisor = int(m_b.group('div'))
        return True, f"(val({src_id}) - {const}) * {divisor}"

    # Case 3: (dst.size_bits / divisor)
    pattern_c = re.compile(
        r"^\s*\(?\s*(?P<dst>\d+)\.size_bits\s*/\s*(?P<div>\d+)\s*\)?\s*$",
        re.IGNORECASE,
    )
    m_c = pattern_c.match(rhs)
    if m_c:
        divisor = int(m_c.group('div'))
        return True, f"val({src_id}) * {divisor}"

    # Fall back to val(src) when pattern is unknown
    return True, f"val({src_id})"


def _normalize_length_formula(formula: str, src_id: int) -> str:
    if not formula:
        return f"(val({src_id})) * 8"

    text = _strip_outer_parens(str(formula).strip())
    handled_assignment, rewritten = _rewrite_assignment_formula(text, src_id)
    if handled_assignment:
        text = rewritten

    # If other val() references remain, align them to src_id
    if re.search(r"val\(\s*(?!%d)\d+\s*\)" % src_id, text, re.IGNORECASE):
        text = f"val({src_id})"

    if '.size_bits' in text or '=' in text:
        text = f"val({src_id})"

    text = _align_formula_with_src(text, src_id)

    if not re.search(r"val\(\s*\d+\s*\)", text):
        text = f"val({src_id})"

    if '*' not in text:
        base = _strip_outer_parens(text)
        if re.search(r"[+\-*/\s]", base):
            return f"({base}) * 8"
        return f"{base} * 8"
    return text


_VAL_REF_PATTERN = re.compile(r"val\(\s*(\d+)\s*\)", re.IGNORECASE)
_VAL_EQ_PATTERN = re.compile(r"^val\(\s*(\d+)\s*\)\s*==\s*(.+)$", re.IGNORECASE)


def _align_formula_with_src(formula: str, src_id: int) -> str:
    """
    Ensure length_of formulas reference the actual source node.
    If other node ids are referenced, rewrite them to use src_id.
    """
    if not formula or src_id is None:
        return formula

    def _replace(match: re.Match) -> str:
        try:
            ref_id = int(match.group(1))
        except Exception:
            return match.group(0)
        if ref_id == src_id:
            return match.group(0)
        return f"val({src_id})"

    return _VAL_REF_PATTERN.sub(_replace, formula)


def _build_node_map(nodes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for node in nodes or []:
        nid = _coerce_int(node.get('node_id'))
        if nid is None:
            continue
        mapping[nid] = node
    return mapping


def _truthy(flag: Any) -> bool:
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (int, float)):
        return bool(flag)
    if isinstance(flag, str):
        return flag.strip().lower() in {'1', 'true', 'yes', 'on'}
    return False


def _is_optional_variant(node: Dict[str, Any]) -> bool:
    annotations = node.get('annotations') or node.get('metadata')
    if isinstance(annotations, dict):
        if _truthy(annotations.get('optional_variant')) or _truthy(annotations.get('allow_variant_removal')):
            return True
    if _truthy(node.get('optional_variant')) or _truthy(node.get('allow_variant_removal')):
        return True
    return False


def _size_repr(value: Any) -> str:
    if isinstance(value, (int, float)):
        try:
            return str(int(value))
        except Exception:
            return str(value)
    return (value or '').strip() if isinstance(value, str) else ''


def _bit_start_repr(value: Any) -> str:
    if isinstance(value, (int, float)):
        try:
            return str(int(value))
        except Exception:
            return str(value)
    return (value or '').strip() if isinstance(value, str) else ''


def _node_signature_basic(node: Dict[str, Any]) -> Tuple[Any, ...]:
    name = (node.get('name', '') or '').strip()
    node_type = str(node.get('node_type', '') or '').lower()
    message_type = str(node.get('message_type', '') or '').lower()
    data_type = str(node.get('data_type', '') or '').lower()
    size_bits = _size_repr(node.get('size_bits'))
    bit_start = _bit_start_repr(node.get('bit_start'))
    return (name, node_type, message_type, data_type, size_bits, bit_start)


def _create_node_resolver(original_tree: Dict[str, Any], final_tree: Dict[str, Any]):
    orig_nodes = [n for n in original_tree.get('nodes', []) if isinstance(n, dict)]
    final_nodes = [n for n in final_tree.get('nodes', []) if isinstance(n, dict)]

    orig_nodes_map = _build_node_map(orig_nodes)
    final_nodes_map = _build_node_map(final_nodes)

    final_signature_index: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    name_type_index: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    name_index: Dict[str, List[int]] = defaultdict(list)

    for nid, node in final_nodes_map.items():
        sig = _node_signature_basic(node)
        final_signature_index[sig].append(nid)
        name = (node.get('name', '') or '').strip()
        node_type = str(node.get('node_type', '') or '').lower()
        msg_type = str(node.get('message_type', '') or '').lower()
        name_type_index[(name, node_type, msg_type)].append(nid)
        name_index[name].append(nid)

    def _map_node_id(old_id: Optional[int]) -> Optional[int]:
        if old_id is None:
            return None
        if old_id in final_nodes_map:
            return old_id
        orig_node = orig_nodes_map.get(old_id)
        if not orig_node:
            return None
        sig = _node_signature_basic(orig_node)
        candidates = list(final_signature_index.get(sig, []))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            target_size = _size_repr(orig_node.get('size_bits'))
            filtered = [cid for cid in candidates if _size_repr(final_nodes_map[cid].get('size_bits')) == target_size]
            if len(filtered) == 1:
                return filtered[0]
            candidates = filtered
        if len(candidates) != 1:
            name = (orig_node.get('name', '') or '').strip()
            node_type = str(orig_node.get('node_type', '') or '').lower()
            msg_type = str(orig_node.get('message_type', '') or '').lower()
            candidates = list(name_type_index.get((name, node_type, msg_type), []))
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                target_size = _size_repr(orig_node.get('size_bits'))
                filtered = [cid for cid in candidates if _size_repr(final_nodes_map[cid].get('size_bits')) == target_size]
                if len(filtered) == 1:
                    return filtered[0]
                candidates = filtered
        if len(candidates) != 1:
            name = (orig_node.get('name', '') or '').strip()
            candidates = list(name_index.get(name, []))
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                target_size = _size_repr(orig_node.get('size_bits'))
                filtered = [cid for cid in candidates if _size_repr(final_nodes_map[cid].get('size_bits')) == target_size]
                if len(filtered) == 1:
                    return filtered[0]
        return candidates[0] if len(candidates) == 1 else None

    return _map_node_id, orig_nodes_map, final_nodes_map


def _collect_variant_signatures(tree: Dict[str, Any]) -> Counter:
    nodes_map = _build_node_map(tree.get('nodes', []))
    counter: Counter = Counter()
    for nid, node in nodes_map.items():
        if str(node.get('node_type', '')).lower() != 'variant':
            continue
        if _is_optional_variant(node):
            continue
        parent = nodes_map.get(_coerce_int(node.get('parent_id')))
        parent_name = (parent.get('name', '') or '').strip() if parent else ''
        parent_type = str(parent.get('node_type', '') or '').lower() if parent else ''
        signature = (
            (node.get('name', '') or '').strip(),
            str(node.get('message_type', '') or '').lower(),
            parent_name,
            parent_type,
        )
        counter[signature] += 1
    return counter


def _collect_selector_condition_map(tree: Dict[str, Any]) -> Dict[Tuple[Any, ...], set]:
    nodes_map = _build_node_map(tree.get('nodes', []))
    selector_map: Dict[Tuple[Any, ...], set] = defaultdict(set)
    for edge in tree.get('edges', []) or []:
        if str(edge.get('rel', '')).lower() != 'condition_on':
            continue
        src_id = _coerce_int(edge.get('src'))
        if src_id is None:
            continue
        node = nodes_map.get(src_id)
        if not node:
            continue
        if _is_optional_variant(node):
            continue
        parent = nodes_map.get(_coerce_int(node.get('parent_id')))
        parent_name = (parent.get('name', '') or '').strip() if parent else ''
        parent_type = str(parent.get('node_type', '') or '').lower() if parent else ''
        signature = (
            (node.get('name', '') or '').strip(),
            str(node.get('message_type', '') or '').lower(),
            parent_name,
            parent_type,
        )
        formula_norm = _normalize_condition_formula_text(str(edge.get('formula') or ''))
        if formula_norm:
            selector_map[signature].add(formula_norm)
    return selector_map


def _validate_structural_invariants(original_tree: Dict[str, Any], final_tree: Dict[str, Any]) -> None:
    orig_variants = _collect_variant_signatures(original_tree)
    final_variants = _collect_variant_signatures(final_tree)
    missing_variants = orig_variants - final_variants
    extra_variants = final_variants - orig_variants
    if missing_variants:
        raise RuntimeError(f"Missing variant definitions after fix: {dict(missing_variants)}")
    if extra_variants:
        raise RuntimeError(f"Unexpected new variants introduced: {dict(extra_variants)}")

    orig_selector_map = _collect_selector_condition_map(original_tree)
    final_selector_map = _collect_selector_condition_map(final_tree)
    missing_selectors = set(orig_selector_map) - set(final_selector_map)
    if missing_selectors:
        raise RuntimeError(f"Missing selector conditions after fix: {missing_selectors}")
    for signature, formulas in orig_selector_map.items():
        candidate_formulas = final_selector_map.get(signature, set())
        missing_formulas = formulas - candidate_formulas
        if missing_formulas:
            raise RuntimeError(
                f"Selector {signature} lost condition(s): {missing_formulas}"
            )

def _split_top_level_args(expr: str) -> List[str]:
    args: List[str] = []
    depth = 0
    current: List[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            if depth > 0:
                depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            arg = ''.join(current).strip()
            if arg:
                args.append(arg)
            current = []
        else:
            current.append(ch)
        i += 1
    tail = ''.join(current).strip()
    if tail:
        args.append(tail)
    return args


def _rewrite_logic_function_calls(expr: str) -> str:
    if not expr:
        return expr

    pattern = re.compile(r'\b(OR|AND)\s*\(', re.IGNORECASE)

    result = expr
    while True:
        match = pattern.search(result)
        if not match:
            break
        func_name = match.group(1).lower()
        joiner = ' or ' if func_name == 'or' else ' and '
        start = match.start()
        args_start = match.end()
        depth = 1
        idx = args_start
        while idx < len(result) and depth > 0:
            ch = result[idx]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            idx += 1
        if depth != 0:
            break
        args_end = idx
        args_blob = result[args_start:args_end]
        args = _split_top_level_args(args_blob)
        normalized = [a.strip() for a in args if a.strip()]
        if not normalized:
            replacement = ''
        elif len(normalized) == 1:
            replacement = f"({normalized[0]})"
        else:
            replacement = '(' + joiner.join(normalized) + ')'
        result = result[:start] + replacement + result[args_end + 1:]
    return result


def _normalize_condition_formula_text(formula: str) -> str:
    if not isinstance(formula, str):
        return formula
    text = formula.strip()
    if not text:
        return text

    text = text.replace('&&', ' and ').replace('||', ' or ')
    text = _rewrite_logic_function_calls(text)
    text = re.sub(r'\bAND\b', ' and ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bOR\b', ' or ', text, flags=re.IGNORECASE)
    # LLM patches sometimes concatenate two parenthesized boolean clauses:
    # `(val(1) >= 1 and val(1) <= 4) (val(1) != 3)`.
    # condition_on formulas are boolean predicates, so the only sensible
    # protocol-agnostic repair is to insert an explicit conjunction.
    text = re.sub(r'\)\s+\(', ') and (', text)
    text = re.sub(
        r'((?:val\s*\(\s*\d+\s*\)|[A-Za-z_][A-Za-z0-9_]*)\s*(?:==|!=|<=|>=|<|>)\s*[-+]?(?:0x[0-9a-fA-F]+|\d+))\s+\(',
        r'\1 and (',
        text,
    )
    text = re.sub(r"(?<![<>=!])=(?!=)", "==", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _split_top_level_disjunctions(expr: str) -> List[str]:
    clauses: List[str] = []
    if not expr:
        return clauses
    depth = 0
    start = 0
    i = 0
    length = len(expr)
    while i < length:
        ch = expr[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            if depth > 0:
                depth -= 1
        elif depth == 0 and expr[i:i + 4].lower() == ' or ':
            clause = expr[start:i].strip()
            if clause:
                clauses.append(clause)
            start = i + 4
            i += 3
        i += 1
    tail = expr[start:].strip()
    if tail:
        clauses.append(tail)
    return clauses


def _extract_simple_disjunction_clauses(formula: str, src_id: Optional[int]) -> Optional[List[str]]:
    normalized = _normalize_condition_formula_text(str(formula or ''))
    if not normalized:
        return None
    stripped = _strip_outer_parens(normalized)
    if ' or ' not in stripped.lower():
        return None

    clauses = _split_top_level_disjunctions(stripped)
    if len(clauses) <= 1:
        return None

    simple_clauses: List[str] = []
    for clause in clauses:
        clause_text = _strip_outer_parens(_normalize_condition_formula_text(clause))
        clause_lower = clause_text.lower()
        if ' or ' in clause_lower or ' and ' in clause_lower or '||' in clause_text or '&&' in clause_text:
            return None
        match = _VAL_EQ_PATTERN.match(clause_text)
        if not match:
            return None
        try:
            clause_src = int(match.group(1))
        except Exception:
            return None
        if src_id is not None and clause_src != src_id:
            return None
        simple_clauses.append(clause_text)

    unique_clauses = []
    seen: Set[str] = set()
    for clause in simple_clauses:
        trimmed = clause.strip()
        if trimmed in seen:
            continue
        seen.add(trimmed)
        unique_clauses.append(trimmed)

    if len(unique_clauses) <= 1:
        return None
    return unique_clauses


def _expand_condition_on_disjunctions(tree: Dict[str, Any]) -> None:
    if not isinstance(tree, dict):
        return
    edges = tree.get('edges')
    if not isinstance(edges, list):
        return

    new_edges: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[Any, Any, Any, Any, Any]] = set()

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        rel = str(edge.get('rel') or '').lower()
        if rel != 'condition_on':
            key = (
                edge.get('src'),
                edge.get('dst'),
                edge.get('rel'),
                edge.get('formula'),
                edge.get('message_type'),
            )
            if key not in seen_keys:
                new_edges.append(edge)
                seen_keys.add(key)
            continue

        src_id = _coerce_int(edge.get('src'))
        formula = edge.get('formula')
        clauses = _extract_simple_disjunction_clauses(formula, src_id)
        if not clauses:
            normalized = _normalize_condition_formula_text(str(formula or ''))
            if normalized != formula:
                edge = dict(edge)
                edge['formula'] = normalized
            key = (
                edge.get('src'),
                edge.get('dst'),
                edge.get('rel'),
                edge.get('formula'),
                edge.get('message_type'),
            )
            if key not in seen_keys:
                new_edges.append(edge)
                seen_keys.add(key)
            continue

        for clause in clauses:
            new_edge = dict(edge)
            new_edge['formula'] = clause
            key = (
                new_edge.get('src'),
                new_edge.get('dst'),
                new_edge.get('rel'),
                new_edge.get('formula'),
                new_edge.get('message_type'),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            new_edges.append(new_edge)

    tree['edges'] = new_edges


def _normalize_condition_formulas(tree: Dict[str, Any]) -> None:
    if not isinstance(tree, dict):
        return
    edges = tree.get('edges', [])
    if not isinstance(edges, list):
        return
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if str(edge.get('rel')).lower() != 'condition_on':
            continue
        formula = edge.get('formula')
        if formula is None:
            continue
        normalized = _normalize_condition_formula_text(str(formula))
        if normalized:
            edge['formula'] = normalized


def _restore_length_formulas(original_tree: Dict[str, Any], final_tree: Dict[str, Any]) -> None:
    if not isinstance(original_tree, dict) or not isinstance(final_tree, dict):
        return
    orig_edges = original_tree.get('edges', []) or []
    final_edges = final_tree.get('edges')
    if not isinstance(final_edges, list):
        final_edges = []
        final_tree['edges'] = final_edges

    map_node_id, _, final_nodes_map = _create_node_resolver(original_tree, final_tree)

    final_edge_index: Dict[tuple, Dict[str, Any]] = {}
    for edge in final_edges:
        if not isinstance(edge, dict):
            continue
        if str(edge.get('rel')) != 'length_of':
            continue
        key = (_coerce_int(edge.get('src')), _coerce_int(edge.get('dst')))
        if None in key:
            continue
        final_edge_index[key] = edge

    for edge in orig_edges:
        if str(edge.get('rel')) != 'length_of':
            continue
        src = map_node_id(_coerce_int(edge.get('src')))
        dst = map_node_id(_coerce_int(edge.get('dst')))
        if src is None or dst is None:
            continue
        if dst not in final_nodes_map:
            continue
        orig_formula = str(edge.get('formula') or '').strip()
        if not orig_formula:
            continue
        normalized_formula = orig_formula
        key = (src, dst)
        target_edge = final_edge_index.get(key)
        if target_edge is None:
            target_edge = {
                'src': src,
                'dst': dst,
                'rel': 'length_of',
                'formula': normalized_formula,
            }
            message_type = edge.get('message_type')
            if message_type:
                target_edge['message_type'] = message_type
            final_edges.append(target_edge)
            final_edge_index[key] = target_edge
        else:
            target_edge['src'] = src
            target_edge['dst'] = dst
            target_edge['formula'] = normalized_formula
            if edge.get('message_type') and not target_edge.get('message_type'):
                target_edge['message_type'] = edge.get('message_type')

    # Do this once after collecting ALL bindings: selecting the first edge
    # would erase conflicting/direction-dependent length evidence.
    _rewrite_problematic_size_bits(final_tree)


def _rewrite_problematic_size_bits(final_tree: Dict[str, Any]) -> None:
    """
    Fill unspecified sizes only when length edges agree; never erase a formula.

    A val() or node-property expression is already canonical tree knowledge,
    even without a duplicate length_of edge. Unsupported/conflicting formulas
    must reach the validator unchanged, not disappear into a variable fallback.
    """
    if not isinstance(final_tree, dict):
        return

    nodes: List[Dict[str, Any]] = [n for n in final_tree.get('nodes', []) if isinstance(n, dict)]
    edges: List[Dict[str, Any]] = [e for e in final_tree.get('edges', []) if isinstance(e, dict)]

    length_edges_by_dst: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if str(edge.get('rel')) != 'length_of':
            continue
        dst = _coerce_int(edge.get('dst'))
        if dst is None:
            continue
        length_edges_by_dst[dst].append(edge)

    for node in nodes:
        node_id = _coerce_int(node.get('node_id'))
        if node_id is None:
            continue
        if explicit_layout(node.get('size_bits')):
            continue
        dst_edges = length_edges_by_dst.get(node_id, [])
        formulas = [str(e.get('formula') or '').strip() for e in dst_edges]
        # Keep branch-dependent/malformed edges for the validator to resolve.
        if formulas and all(explicit_layout(f) for f in formulas) and len({canonical_formula(f) for f in formulas}) == 1:
            node['size_bits'] = formulas[0]


_MALFORMED_NODE_SIZE_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<node_id>[0-9]+)\.bit\.size_bits\b"
)


def _make_default_variants_exclusive(tree: Dict[str, Any]) -> List[str]:
    """Turn a catch-all selector branch into the "otherwise" branch.

    A tree can route specific values by ``val(s) == c`` and a generic branch by a
    formula such as ``(val(s) != a) and (val(s) != b)``. The generic branch then also
    matches every code a sibling claims, and the validator reports
    NON_EXCLUSIVE_VARIANT_CONDITIONS. The intended reading is
    "any other code"; this conjoins ``val(s) != c`` for each constant a sibling selects by
    plain equality and the catch-all also accepts. Only formulas that reference nothing but
    this selector are touched.
    """
    import re

    def _single(sid, formula):
        refs = set(re.findall(r"val\(\s*(\d+)\s*\)", str(formula or "")))
        return refs == {str(sid)}

    def _accepts(sid, formula, value):
        expr = re.sub(rf"val\(\s*{re.escape(str(sid))}\s*\)", str(int(value)), str(formula))
        expr = expr.replace("&&", " and ").replace("||", " or ")
        expr = re.sub(r"!(?!=)", " not ", expr)
        # Only integer comparisons joined by and/or/not are evaluated; anything else is left alone.
        if not re.fullmatch(r"(?:\s|\d|0x[0-9a-fA-F]+|[()<>=!-]|\band\b|\bor\b|\bnot\b)*", expr):
            return None
        try:
            return bool(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return None

    edges = [e for e in tree.get("edges") or []
             if isinstance(e, dict) and str(e.get("rel") or "").lower() == "condition_on"]
    by_selector: Dict[Any, List[Dict[str, Any]]] = {}
    for edge in edges:
        by_selector.setdefault(edge.get("src"), []).append(edge)
    changes: List[str] = []
    for sid, group in by_selector.items():
        equal: Dict[int, Any] = {}
        for edge in group:
            m = re.fullmatch(rf"\s*\(?\s*val\(\s*{re.escape(str(sid))}\s*\)\s*==\s*(-?(?:0x[0-9a-fA-F]+|\d+))\s*\)?\s*",
                             str(edge.get("formula") or ""))
            if m:
                equal[int(m.group(1), 0)] = edge.get("dst")
        if not equal:
            continue
        for edge in group:
            formula = edge.get("formula")
            if not isinstance(formula, str) or not _single(sid, formula):
                continue
            if re.fullmatch(rf"\s*\(?\s*val\(\s*{re.escape(str(sid))}\s*\)\s*==\s*[-0-9xa-fA-F]+\s*\)?\s*", formula):
                continue
            claimed = sorted(c for c, dst in equal.items()
                             if dst != edge.get("dst") and _accepts(sid, formula, c) is True)
            if not claimed:
                continue
            edge["formula"] = f"({formula}) and " + " and ".join(f"(val({sid}) != {c})" for c in claimed)
            changes.append(f"selector {sid} catch-all -> {edge.get('dst')} excludes {claimed}")
    return changes


def _widen_selector_enums_to_condition_values(tree: Dict[str, Any]) -> List[str]:
    """Add to a selector's enum the constants its own condition_on edges select.

    The tree already holds a variant for such a value while the enum does not
    list it, and the validator reports ENUM_EXCLUDES_CONDITION_VALUES. Widening only
    admits values the tree already routes, so no documented branch is lost.
    """
    import re

    nodes = {n.get("node_id"): n for n in tree.get("nodes") or [] if isinstance(n, dict)}
    referenced: Dict[Any, set] = {}
    for edge in tree.get("edges") or []:
        if not isinstance(edge, dict) or str(edge.get("rel") or "").lower() != "condition_on":
            continue
        sid = edge.get("src")
        formula = edge.get("formula")
        if not isinstance(formula, str):
            continue
        for m in re.finditer(rf"val\(\s*{re.escape(str(sid))}\s*\)\s*==\s*(-?(?:0x[0-9a-fA-F]+|\d+))", formula):
            try:
                referenced.setdefault(sid, set()).add(int(m.group(1), 0))
            except ValueError:
                continue
    changes: List[str] = []
    for sid, values in referenced.items():
        node = nodes.get(sid)
        if not node or str(node.get("node_type") or "").lower() != "selector":
            continue
        constraints = node.get("constraints")
        if not isinstance(constraints, list):
            continue
        for i, raw in enumerate(constraints):
            if not isinstance(raw, str):
                continue
            m = re.match(r"\s*enum\s*:\s*(.+)$", raw, flags=re.IGNORECASE)
            if not m:
                continue
            sep = "|" if "|" in m.group(1) else ", "
            allowed = []
            for token in re.split(r"[|,]", m.group(1)):
                token = token.strip()
                try:
                    allowed.append(int(token, 0))
                except ValueError:
                    allowed = None
                    break
            if not allowed:
                continue
            missing = sorted(values - set(allowed))
            if missing:
                constraints[i] = "enum: " + sep.join(str(v) for v in sorted(set(allowed) | set(missing)))
                changes.append(f"selector {sid} enum += {missing}")
    return changes


def _normalize_malformed_layout_references(tree: Dict[str, Any]) -> List[str]:
    """Repair an unambiguous property-path typo without changing semantics.

    Generated layout expressions occasionally spell ``N.size_bits`` as
    ``N.bit.size_bits``.  The latter is not part of the expression language and
    cannot denote any other valid value, so correcting it is a safe,
    protocol-independent normalization.  Other unsupported expressions remain
    untouched and must still be handled by validation/refinement.
    """
    if not isinstance(tree, dict):
        return []

    changes: List[str] = []
    for node in tree.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = _coerce_int(node.get("node_id"))
        for property_name in ("bit_start", "size_bits"):
            value = node.get(property_name)
            if not isinstance(value, str):
                continue
            normalized = _MALFORMED_NODE_SIZE_REFERENCE_RE.sub(
                lambda match: f"{match.group('node_id')}.size_bits",
                value,
            )
            if normalized == value:
                continue
            node[property_name] = normalized
            changes.append(
                f"normalize node {node_id}.{property_name} malformed size reference"
            )
    return changes


def _infer_container_sizes(final_tree: Dict[str, Any]) -> None:
    """
    Infer deterministic sizes for container-like nodes when all children have
    fixed-width definitions. This keeps logic protocol-agnostic while allowing
    headers with constant fields to expose explicit size_bits values.
    """
    if not isinstance(final_tree, dict):
        return

    nodes: List[Dict[str, Any]] = [n for n in final_tree.get('nodes', []) if isinstance(n, dict)]
    nodes_by_id = {_coerce_int(n.get('node_id')): n for n in nodes}

    for node in nodes:
        node_id = _coerce_int(node.get('node_id'))
        if node_id is None:
            continue
        node_type = str(node.get('node_type', '')).lower()
        if node_type not in {'container', 'header', 'variant'}:
            continue

        size_bits = node.get('size_bits')
        if isinstance(size_bits, int):
            continue
        if isinstance(size_bits, str) and size_bits.strip().lower() not in {'', 'none', 'variable'}:
            continue

        children_ids = node.get('children_ids') or []
        if not children_ids:
            continue

        # A length-bound/repeated container is dynamic even if its current
        # example happens to contain only fixed-size children.
        if any(e.get('dst') == node_id and e.get('rel') in {'length_of', 'repeat_count'}
               for e in final_tree.get('edges', []) if isinstance(e, dict)):
            continue
        start = static_property(nodes_by_id, node_id, 'bit_start')
        origin = node_id if start is None else None
        start = 0 if start is None else start
        child_ends: List[int] = []
        all_static = True
        for child_id in children_ids:
            child_node = nodes_by_id.get(_coerce_int(child_id))
            if not child_node:
                all_static = False
                break
            child_size = static_property(nodes_by_id, child_id, 'size_bits', origin=origin)
            child_start = static_property(nodes_by_id, child_id, 'bit_start', origin=origin)
            if child_size is None or child_start is None or child_size < 0 or child_start < start:
                all_static = False
                break
            child_ends.append(child_start + child_size)

        if not all_static or not child_ends:
            continue

        inferred = max(child_ends) - start
        if inferred <= 0:
            continue

        node['size_bits'] = inferred


def _normalize_value_constraints(final_tree: Dict[str, Any]) -> None:
    """
    Ensure constraints expressed as pseudo-range (e.g., 'range: value % 2 == 0')
    are rewritten back to formula constraints so downstream stages can handle them.
    """
    if not isinstance(final_tree, dict):
        return

    nodes: List[Dict[str, Any]] = [n for n in final_tree.get('nodes', []) if isinstance(n, dict)]
    for node in nodes:
        constraints = node.get('constraints')
        if not isinstance(constraints, list):
            continue
        normalized: List[Any] = []
        changed = False
        for item in constraints:
            if isinstance(item, str) and item.strip().lower().startswith('range:'):
                body = item.split(':', 1)[1].strip()
                if re.search(r'value\s*[%*/+-]', body):
                    normalized.append(f"formula: {body}")
                    changed = True
                    continue
            normalized.append(item)
        if changed:
            node['constraints'] = normalized
logger = logging.getLogger(__name__)


def ensure_logging(log_filename: str) -> None:
    setup_logging(
        console_level=logging.ERROR,
        file_path=log_filename,
        file_level=logging.INFO,
        replace_existing=False,
    )


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_sections(payload: Any) -> Sequence[Dict[str, Any]]:
    if isinstance(payload, dict) and "sections" in payload:
        payload = payload["sections"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Sections file must be a list or contain a 'sections' array")


def _env_flag_true(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


from pargen.protocol_knowledge.repair.evidence import retrieve_evidence_sections


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip() or default)
    except Exception:
        return int(default)


def _issue_location(issue) -> str:
    target = getattr(issue, "target", None)
    if target is not None and getattr(target, "identifier", None) is not None:
        return f"{getattr(target, 'kind', '')}:{target.identifier}"
    return ""


def _issue_keys(report, predicate) -> "Counter[tuple[str, str, str]]":
    return Counter(
        (str(getattr(getattr(issue, "type", None), "name", getattr(issue, "type", ""))),
         str(getattr(issue, "code", "") or ""), _issue_location(issue))
        for issue in (getattr(report, "issues", {}) or {}).values() if predicate(issue)
    )


def _hard_error_keys(report) -> "Counter[tuple[str, str, str]]":
    return _issue_keys(report, is_hard_error)


def _reported_conflict_keys(report) -> "Counter[tuple[str, str, str]]":
    """Reported conflicts: every ERROR the check reports, structural or not (e.g. a documented rule
    the PKG does not represent is a SEMANTICS error, not a hard one)."""
    return _issue_keys(report, lambda issue: getattr(issue, "severity", None) == Severity.ERROR)


def accept_graph_repair(old_report, new_report) -> tuple[bool, str]:
    """AcceptGraphRepair: the edit resolves reported conflicts and passes the structural and semantic gates.

    Resolved: fewer reported conflicts (ERROR issues) than the baseline. Structural gate: no structural
    hard error that the baseline did not have. Semantic gate: no erased binding or predicate (the
    preservation guard reports these as CONSTRAINT_PRESERVATION).
    """
    old_conflicts, new_conflicts = _reported_conflict_keys(old_report), _reported_conflict_keys(new_report)
    if sum(new_conflicts.values()) >= sum(old_conflicts.values()):
        return False, "no_resolved_conflict"
    if any(code == "CONSTRAINT_PRESERVATION" for _type, code, _node in new_conflicts - old_conflicts):
        return False, "semantic_regression"
    if any(kind == "STRUCTURE" for kind, _code, _node in _hard_error_keys(new_report) - _hard_error_keys(old_report)):
        return False, "structural_regression"
    return True, "resolved_conflicts"


def _issue_stats(report) -> tuple[int, int]:
    issues_raw = getattr(report, "issues", {}) or {}
    issues = {
        iid: iss for iid, iss in issues_raw.items() if getattr(iss, "severity", None) == Severity.ERROR
    }
    hard = sum(1 for issue in issues.values() if is_hard_error(issue))
    return hard, len(issues)


def run_step2_fix(
    cache_path: Optional[Path] = None,
    sections_path: Optional[Path] = None,
    agent_provider: str = "deepseek",
    output_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    *,
    traffic_path: Optional[Path] = None,
    max_packets: int = 0,
    target_message_type: Optional[str] = None,
    use_hybrid_validator: bool = False,
    max_llm_calls: Optional[int] = None,
) -> Path:
    base_dir = Path(__file__).resolve().parent
    cache_path = Path(cache_path) if cache_path else STEP2_CACHE_DIR / "final_complete_protocol_tree.json"
    sections_path = Path(sections_path) if sections_path else base_dir / "modbus_document_sections_subset.json"
    output_dir = Path(output_dir) if output_dir else (base_dir / "results")
    log_dir = Path(log_dir) if log_dir else (base_dir / "logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = log_dir / f"log_{int(time.time())}.txt"
    os.environ.setdefault("STEP2_MCTS_LOG_UCT", "1")
    os.environ.setdefault("STEP2_MCTS_LOG_FILE", str(log_filename))
    os.environ.setdefault("STEP2_ENABLE_MCTS_LOG", "1")
    ensure_logging(str(log_filename))

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            tree_data = json.load(handle)
        tree = tree_data.get("protocol_tree", tree_data)
        logger.info("Loaded protocol tree from cache: %s", cache_path)
        print(f"Successfully loaded protocol tree with {len(tree.get('nodes', []))} nodes")
    except Exception as exc:
        logger.error("Failed to load protocol tree %s: %s", cache_path, exc)
        raise

    try:
        sections_data = load_json(str(sections_path))
        sections: Sequence[Dict[str, Any]] = _coerce_sections(sections_data)
        logger.info("Loaded %d sections from: %s", len(sections), sections_path)
        print(f"Successfully loaded {len(sections)} document sections")
    except Exception as exc:
        logger.error("Failed to load sections %s: %s", sections_path, exc)
        raise

    try:
        agent = EnhancedPureAIAgent(agent_provider, cache_dir=STEP2_FIX_CACHE_DIR)
        logger.info("Initialized AI agent with provider: %s", agent_provider)
        print(f"Initialized AI agent with provider: {agent_provider}")
    except Exception as exc:
        logger.error("Failed to initialize AI agent: %s", exc)
        raise

    original_tree_snapshot = deepcopy(tree)
    preprocessed_tree = deepcopy(tree)

    try:
        changes = _normalize_malformed_layout_references(preprocessed_tree)
        if changes:
            logger.info("Pre-MCTS layout reference normalization: %s", "; ".join(changes[:20]))
    except Exception as exc:
        logger.warning("Pre-MCTS layout reference normalization failed: %s", exc)


    strict_validator_loop = os.getenv("STEP2_STRICT_VALIDATOR_LOOP", "0").strip().lower() in {"1", "true", "yes", "on"}
    if strict_validator_loop:
        logger.info(
            "STEP2_STRICT_VALIDATOR_LOOP enabled: skipping pre-MCTS heuristic normalizations; "
            "traffic fixing will be driven only by validator results"
        )
    else:
        try:
            _restore_length_formulas(original_tree_snapshot, preprocessed_tree)
        except Exception as exc:
            logger.warning("Pre-MCTS length formula normalization failed: %s", exc)

        try:
            # if _env_flag_true("STEP2_SPLIT_CONDITION_ON", False):
            #     _expand_condition_on_disjunctions(preprocessed_tree)
            logger.info("Skipping _expand_condition_on_disjunctions to preserve complex formulas.")
        except Exception as exc:
            logger.warning("Pre-MCTS condition split failed: %s", exc)

        try:
            _normalize_condition_formulas(preprocessed_tree)
        except Exception as exc:
            logger.warning("Pre-MCTS condition formula normalization failed: %s", exc)


        try:
            _normalize_value_constraints(preprocessed_tree)
        except Exception as exc:
            logger.warning("Pre-MCTS constraint normalization failed: %s", exc)

    validator_fn = run_full_validation
    prompt_mode = "fix"
    hybrid_path: Optional[Path] = None
    if use_hybrid_validator and traffic_path is not None:
        hybrid_path = Path(traffic_path)
        if not hybrid_path.exists():
            logger.warning("Hybrid validator enabled but traffic file missing: %s; falling back to syntax-only", hybrid_path)
            hybrid_path = None
    if use_hybrid_validator and hybrid_path is not None:
        prompt_mode = "traffic_fix"

        def _hybrid_validator(tree_payload: Dict[str, Any]):
            return run_hybrid_validation(
                tree_payload,
                traffic_path=hybrid_path,
                max_packets=max_packets,
                target_message_type=target_message_type,
            )

        validator_fn = _hybrid_validator

    if _env_flag("STEP2_DOC_CONSTRAINT_COVERAGE"):
        base_validator_fn = validator_fn
        try:
            max_doc_issues = int(os.getenv("STEP2_DOC_CONSTRAINT_MAX_ISSUES", "12").strip() or "12")
        except Exception:
            max_doc_issues = 12
        max_doc_issues = max(1, max_doc_issues)
        structure_limit_raw = os.getenv("STEP2_DOC_STRUCTURE_MAX_ISSUES")
        if structure_limit_raw is None:
            max_doc_structure_issues = _adaptive_doc_structure_issue_limit(
                preprocessed_tree,
                sections,
                max_doc_issues=max_doc_issues,
            )
        else:
            try:
                max_doc_structure_issues = int(structure_limit_raw.strip() or "0")
            except Exception:
                max_doc_structure_issues = min(12, max_doc_issues)
        max_doc_structure_issues = max(0, max_doc_structure_issues)
        include_weak_name_alignment = _env_flag("STEP2_DOC_STRUCTURE_INCLUDE_WEAK_NAMES")
        doc_coverage_focus_kind = (os.getenv("STEP2_DOC_COVERAGE_FOCUS_KIND") or "all").strip().lower() or "all"
        fixed_doc_claim_ids: Optional[set[str]] = None

        def _doc_constraint_validator(tree_payload: Dict[str, Any]):
            nonlocal fixed_doc_claim_ids
            base_report = base_validator_fn(tree_payload)
            if fixed_doc_claim_ids is None:
                initial_uncovered = find_uncovered_doc_constraints(tree_payload, sections)
                fixed_doc_claim_ids = {
                    claim.claim_id for claim, _ in initial_uncovered[:max_doc_issues]
                }
            return augment_report_with_doc_constraint_coverage(
                base_report,
                tree_payload,
                sections,
                max_issues=max_doc_issues,
                claim_ids=fixed_doc_claim_ids,
                baseline_tree=preprocessed_tree,
                max_structure_issues=max_doc_structure_issues,
                include_weak_name_alignment=include_weak_name_alignment,
                focus_kind=doc_coverage_focus_kind,
            )

        validator_fn = _doc_constraint_validator
        logger.info(
            "STEP2_DOC_CONSTRAINT_COVERAGE enabled: adding protocol-agnostic documentation constraint "
            "coverage issues (focus_kind=%s, constraint_window=%d, structure_window=%d, include_weak_names=%s)",
            doc_coverage_focus_kind,
            max_doc_issues,
            max_doc_structure_issues,
            include_weak_name_alignment,
        )

    # Give search the same preservation feedback used by the publication
    # gate, so erasing a binding cannot improve a candidate's validation score.
    from pargen.protocol_knowledge.run_traffic_fix import _traffic_report_with_constraint_guard

    semantic_validator_fn = validator_fn

    def preservation_validator(tree_payload):
        return _traffic_report_with_constraint_guard(
            semantic_validator_fn(tree_payload), baseline_tree=original_tree_snapshot,
            candidate_tree=tree_payload, context="stage_b_static_fix",
        )

    validator_fn = preservation_validator
    initial_report = validator_fn(preprocessed_tree)
    initial_hard_errors, initial_total_issues = _issue_stats(initial_report)

    try:
        if strict_validator_loop:
            print("Starting strict validator-driven fix loop...")
            logger.info("Starting strict validator-driven fix loop")
        else:
            print("Starting MCTS fix process...")
            logger.info("Starting MCTS fix process")
        if max_llm_calls is None:
            raw_max_calls = os.getenv("STEP2_MCTS_MAX_CALLS", "").strip()
            if not raw_max_calls:
                raw_max_calls = os.getenv("STEP2_MCTS_MAX_LLM_CALLS", "20").strip()
            try:
                max_llm_calls = int(raw_max_calls)
            except Exception:
                max_llm_calls = 20
        max_llm_calls = max(0, int(max_llm_calls))
        try:
            print(
                f"[Step2Fix] max_llm_calls={max_llm_calls} (override via STEP2_MCTS_MAX_CALLS)",
                flush=True,
            )
        except Exception:
            pass

        node_snapshot_dir = STEP2_FIX_CACHE_DIR / "mcts_node_snapshots"
        node_snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Static refinement: CheckPKG -> RetrieveEvidence -> SearchGraphRepair -> AcceptGraphRepair, for the static round budget.
        static_rounds = max(1, _env_int("STEP2_STATIC_REPAIR_ROUNDS", 5))
        current_tree, current_report = preprocessed_tree, initial_report
        for round_idx in range(1, static_rounds + 1):
            if not _reported_conflict_keys(current_report):  # R_G is empty
                break
            # RetrieveEvidence: repair sees the fragments linked to the implicated objects.
            evidence_sections = retrieve_evidence_sections(current_tree, current_report, sections)
            try:
                batch_size_override = max(1, int(os.getenv("STEP2_MCTS_BATCH_SIZE", str(len(evidence_sections)))))
            except Exception:
                batch_size_override = max(1, len(evidence_sections))
            candidate_tree = mcts_fix_tree(
                agent,
                current_tree,
                sections=evidence_sections,
                raw_sections=evidence_sections,
                batch_size=batch_size_override,
                max_llm_calls=max_llm_calls,
                node_snapshot_dir=str(node_snapshot_dir),
                validator_fn=validator_fn,
                prompt_mode=prompt_mode,
            )
            candidate_report = validator_fn(candidate_tree)
            accepted, accept_reason = accept_graph_repair(current_report, candidate_report)
            logger.info("Static repair round %d/%d: accepted=%s (%s), hard errors %d -> %d", round_idx,
                        static_rounds, accepted, accept_reason, _issue_stats(current_report)[0],
                        _issue_stats(candidate_report)[0])
            print(f"[Step2Fix] static round {round_idx}/{static_rounds}: accepted={accepted} ({accept_reason})",
                  flush=True)
            if not accepted:
                break
            current_tree, current_report = candidate_tree, candidate_report
        final_tree = current_tree
        logger.info("Fix search ended; final validation is still required")

        # Keep post-MCTS cleanup narrow.  Size/length inference is intentionally
        # disabled here because it can overwrite the tree selected by search, but
        # formula token cleanup is text-local and prevents syntactically invalid
        # condition_on guards from being emitted.
        try:
            _normalize_condition_formulas(final_tree)
        except Exception as exc:
            logger.warning("Post-MCTS condition formula normalization failed: %s", exc)


        final_report = validator_fn(final_tree)
        final_hard_errors, final_total_issues = _issue_stats(final_report)

        output_path = output_dir / f"fixed_protocol_tree_{int(time.time())}.json"
        output_payload = {
            "protocol_tree": final_tree,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fix_info": {
                "agent_provider": agent_provider,
                "original_nodes": len(tree.get('nodes', [])),
                "fixed_nodes": len(final_tree.get('nodes', [])) if isinstance(final_tree, dict) else 0,
            },
        }

        summary = {
            "protocol_name": final_tree.get("protocol_name") or tree.get("protocol_name") or cache_path.stem,
            "validator": "hybrid" if prompt_mode == "traffic_fix" else "static",
            "initial_hard_errors": initial_hard_errors,
            "initial_total_issues": initial_total_issues,
            "after_fix_hard_errors": final_hard_errors,
            "after_fix_total_issues": final_total_issues,
            "initial_ok": bool(getattr(initial_report, "ok", False)),
            "after_fix_ok": bool(getattr(final_report, "ok", False)) and not any(
                issue.severity == Severity.ERROR for issue in final_report.issues.values()),
            "traffic_successful_samples": int(getattr(final_report, "traffic_successful_samples", 0) or 0),
            "traffic_total_samples": int(getattr(final_report, "traffic_total_samples", 0) or 0),
            "traffic_total_coverage_gap_bits": int(getattr(final_report, "traffic_total_coverage_gap_bits", 0) or 0),
            "traffic_length_total_abs_error_bits": int(getattr(final_report, "traffic_length_total_abs_error_bits", 0) or 0),
            "traffic_overflow_length_bits": int(getattr(final_report, "traffic_overflow_length_bits", 0) or 0),
            "mcts_simulations": None,
            "best_issue_count": final_total_issues,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Step2 summary saved to: %s", summary_path)
        publish_refinement(
            output_path, output_payload, final_report, stage="stage_b_static_fix",
            require_traffic=prompt_mode == "traffic_fix",
            extra_failures=layout_binding_losses(original_tree_snapshot, final_tree),
            metadata={"input_tree": str(cache_path), "sections": str(sections_path),
                      "input_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest()},
        )
        logger.info("Validated fixed tree saved to: %s", output_path)
        print(f"Validated fixed tree saved to: {output_path}")
        return output_path
    except Exception as exc:
        logger.error("MCTS fix process failed: %s", exc)
        raise


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Step2 fix pipeline")
    parser.add_argument("--tree", type=Path, default=None, help="Path to final_complete_protocol_tree.json")
    parser.add_argument("--sections", type=Path, default=None, help="Path to sections JSON")
    parser.add_argument("--provider", default="deepseek", help="LLM provider for the fixer")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write fixed trees")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store log files")
    parser.add_argument("--traffic", type=Path, default=None, help="Optional traffic hex dump for hybrid validation")
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=None,
        help="Max LLM calls for Step2 fix (default: $STEP2_MCTS_MAX_CALLS or 20).",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="Limit traffic samples used in hybrid validation (0 = no limit)",
    )
    parser.add_argument(
        "--target-message-type",
        type=str,
        default=None,
        help="Optional message_type filter for hybrid validation (request/response)",
    )
    parser.add_argument(
        "--use-hybrid-validator",
        action="store_true",
        default=False,
        help="Use hybrid (syntax + traffic) validation inside the fixer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    try:
        result = run_step2_fix(
            cache_path=cli_args.tree,
            sections_path=cli_args.sections,
            agent_provider=cli_args.provider,
            output_dir=cli_args.output_dir,
            log_dir=cli_args.log_dir,
            traffic_path=cli_args.traffic,
            max_llm_calls=cli_args.max_llm_calls,
            max_packets=cli_args.max_packets,
            target_message_type=cli_args.target_message_type,
            use_hybrid_validator=cli_args.use_hybrid_validator,
        )
        print(f"Fixed tree saved to: {result}")
    except Exception as exc:
        print(f"Error during MCTS fix process: {exc}")
        raise
