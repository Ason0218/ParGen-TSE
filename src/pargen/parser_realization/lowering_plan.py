from __future__ import annotations

"""Stage C lowering-plan extraction from Parse-IR.

This module is intentionally host-neutral.  It turns Parse-IR relations into a
small plan that backend emitters can consume before asking the LLM for code.
"""

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .constraint_contract import ContractItem, build_contract, edge_constraint_id
from .parse_ir import Edge, Node, ParseIR


@dataclass(frozen=True)
class ValueTerm:
    node_id: int
    node_name: str
    scale_bits: int


@dataclass(frozen=True)
class LoweredField:
    node_id: int
    name: str
    path: str
    data_type: str
    bit_start: Any
    size_bits: Any
    base_bit_offset: Optional[int] = None
    byte_offset: Optional[int] = None
    bit_offset: Optional[int] = None
    value_terms: List[ValueTerm] = field(default_factory=list)
    flag_node_id: Optional[int] = None
    flag_name: Optional[str] = None
    flag_base_bit_offset: Optional[int] = None
    flag_byte_offset: Optional[int] = None
    flag_bit_offset: Optional[int] = None
    flag_mask: Optional[str] = None
    flag_contract_id: Optional[str] = None


@dataclass(frozen=True)
class SelectorDispatch:
    selector_id: int
    selector_name: str
    selector_value: int
    variant_id: int
    variant_name: str
    fields: List[LoweredField] = field(default_factory=list)
    flag_controlled_fields: List[LoweredField] = field(default_factory=list)


@dataclass(frozen=True)
class LengthCheck:
    src: int
    src_name: str
    dst: int
    dst_name: str
    rel: str
    formula: str
    contract_id: Optional[str] = None


@dataclass(frozen=True)
class StageCLoweringPlan:
    selector_dispatches: List[SelectorDispatch] = field(default_factory=list)
    length_checks: List[LengthCheck] = field(default_factory=list)
    constraint_diagnostics: List[ContractItem] = field(default_factory=list)


_CONDITION_EQ_RE = re.compile(r"val\((?P<selector>\d+)\)\s*==\s*(?P<value>0x[0-9a-fA-F]+|\d+)")


def _tokens(value: object) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").replace("_", " ").replace("-", " ").lower())


def _aliases(value: object) -> set[str]:
    toks = set(_tokens(value))
    aliases = set(toks)
    if {"v4"} & toks or "ipv4" in toks or "ip4" in toks:
        aliases.update({"v4", "ipv4", "ip4"})
    if {"v6"} & toks or "ipv6" in toks or "ip6" in toks:
        aliases.update({"v6", "ipv6", "ip6"})
    return aliases


def _path_for(node_id: int, nodes: Dict[int, Node]) -> str:
    parts: List[str] = []
    current = nodes.get(int(node_id))
    seen: set[int] = set()
    while current is not None and int(current.id) not in seen:
        seen.add(int(current.id))
        parts.append(str(current.name))
        if current.parent_id is None:
            break
        current = nodes.get(int(current.parent_id))
    return ".".join(reversed(parts))


def _children(parse_ir: ParseIR) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for node in parse_ir.nodes.values():
        if node.parent_id is not None:
            out[int(node.parent_id)].append(int(node.id))
    for parent, child_ids in list(out.items()):
        child_ids.sort()
    return out


def _leaf_ids(root_id: int, parse_ir: ParseIR, children: Dict[int, List[int]]) -> List[int]:
    leaves: List[int] = []

    def visit(node_id: int) -> None:
        child_ids = children.get(int(node_id), [])
        node = parse_ir.nodes.get(int(node_id))
        if node is None:
            return
        if child_ids:
            for child_id in child_ids:
                visit(child_id)
            return
        if str(node.kind or "").strip().lower() in {"field", "selector", "length", "payload"}:
            leaves.append(int(node_id))

    visit(int(root_id))
    return leaves


def _condition_value(edge: Edge) -> Optional[int]:
    match = _CONDITION_EQ_RE.search(str(edge.formula or ""))
    if not match:
        return None
    try:
        if int(match.group("selector")) != int(edge.src):
            return None
        return int(match.group("value"), 0)
    except Exception:
        return None


def _int_size_bits(node: Optional[Node]) -> Optional[int]:
    if node is None:
        return None
    raw = node.size_bits
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int) and raw >= 0:
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if re.fullmatch(r"-?\d+", s):
            value = int(s)
            return value if value >= 0 else None
    return None


def _split_linear_terms(expr: str) -> List[str]:
    text = str(expr or "").replace(" ", "")
    if not text:
        return []
    terms = re.findall(r"[+-]?[^+-]+", text)
    return [term for term in terms if term not in {"", "+", "-"}]


def _strip_wrapping_parens(term: str) -> str:
    text = term.strip()
    while text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        if inner.count("(") != inner.count(")"):
            break
        text = inner
    return text


def _parse_scaled_val(term: str) -> Optional[tuple[int, int]]:
    text = _strip_wrapping_parens(term)
    match = re.fullmatch(r"val\((\d+)\)\*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.fullmatch(r"(\d+)\*val\((\d+)\)", text)
    if match:
        return int(match.group(2)), int(match.group(1))
    match = re.fullmatch(r"val\((\d+)\)", text)
    if match:
        return int(match.group(1)), 1
    return None


def _merge_terms(dst: Dict[int, int], src: Dict[int, int], *, coefficient: int = 1) -> None:
    for node_id, scale in src.items():
        dst[int(node_id)] = dst.get(int(node_id), 0) + int(coefficient) * int(scale)
        if dst[int(node_id)] == 0:
            del dst[int(node_id)]


def _compile_bit_expr_relative_to(
    parse_ir: ParseIR,
    expr: Any,
    *,
    root_id: int,
    seen: Optional[set[int]] = None,
) -> Optional[tuple[int, Dict[int, int]]]:
    """Compile a simple Stage-B bit expression relative to `root_id`.

    Supported forms are intentionally conservative and backend-neutral:
    integer constants, `<node>.bit_start`, `<node>.size_bits`, and
    `val(<node>) * N` terms combined with `+` / `-`. Runtime value terms are
    preserved as scale factors instead of guessed at codegen time.
    """

    if isinstance(expr, bool) or expr is None:
        return None
    if isinstance(expr, int):
        root = parse_ir.nodes.get(int(root_id))
        root_start = root.bit_start if root is not None else None
        if isinstance(root_start, int):
            return int(expr) - int(root_start), {}
        return int(expr), {}
    text = str(expr).strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+", text):
        return _compile_bit_expr_relative_to(parse_ir, int(text), root_id=root_id, seen=seen)

    const_bits = 0
    value_terms: Dict[int, int] = {}
    seen_ids = set(seen or set())

    for raw_term in _split_linear_terms(text):
        sign = -1 if raw_term.startswith("-") else 1
        term = raw_term[1:] if raw_term[:1] in {"+", "-"} else raw_term
        term = _strip_wrapping_parens(term)
        if not term:
            continue

        val_term = _parse_scaled_val(term)
        if val_term is not None:
            node_id, scale = val_term
            value_terms[node_id] = value_terms.get(node_id, 0) + sign * scale
            continue

        match = re.fullmatch(r"(\d+)\.bit_start", term)
        if match:
            ref_id = int(match.group(1))
            if ref_id == int(root_id):
                continue
            if ref_id in seen_ids:
                return None
            ref = parse_ir.nodes.get(ref_id)
            if ref is None:
                return None
            compiled = _compile_bit_expr_relative_to(
                parse_ir,
                ref.bit_start,
                root_id=int(root_id),
                seen=seen_ids | {ref_id},
            )
            if compiled is None:
                return None
            ref_const, ref_terms = compiled
            const_bits += sign * ref_const
            _merge_terms(value_terms, ref_terms, coefficient=sign)
            continue

        match = re.fullmatch(r"(\d+)\.size_bits", term)
        if match:
            size = _int_size_bits(parse_ir.nodes.get(int(match.group(1))))
            if size is None:
                return None
            const_bits += sign * int(size)
            continue

        if re.fullmatch(r"\d+", term):
            const_bits += sign * int(term)
            continue

        return None

    value_terms = {node_id: scale for node_id, scale in value_terms.items() if scale != 0}
    return const_bits, value_terms


def _value_terms_to_objects(parse_ir: ParseIR, terms: Dict[int, int]) -> List[ValueTerm]:
    out: List[ValueTerm] = []
    for node_id in sorted(terms):
        node = parse_ir.nodes.get(int(node_id))
        out.append(
            ValueTerm(
                node_id=int(node_id),
                node_name=str(node.name) if node is not None else f"node_{node_id}",
                scale_bits=int(terms[node_id]),
            )
        )
    return out


def _subbyte_mask(bit_offset: int, size_bits: int) -> Optional[str]:
    if bit_offset < 0 or size_bits <= 0 or bit_offset + size_bits > 8:
        return None
    mask = 0
    for bit in range(bit_offset, bit_offset + size_bits):
        mask |= 1 << (7 - bit)
    return f"0x{mask:02X}"


def _flag_override_matches(
    override: Dict[str, Any],
    *,
    variant_path: str,
    field: Node,
    flag: Node,
) -> bool:
    group_aliases = _aliases(override.get("group"))
    if group_aliases and not (group_aliases & _aliases(variant_path)):
        return False

    field_aliases = _aliases(override.get("field"))
    if field_aliases and not (field_aliases & _aliases(field.name)):
        return False

    flag_aliases = _aliases(override.get("flag"))
    if flag_aliases and not (flag_aliases & _aliases(flag.name)):
        return False

    for version_aliases in ({"v4", "ipv4", "ip4"}, {"v6", "ipv6", "ip6"}):
        if field_aliases & version_aliases and not (_aliases(field.name) & version_aliases):
            return False
        if flag_aliases & version_aliases and not (_aliases(flag.name) & version_aliases):
            return False
    return True


def _profile_flag_overrides(codegen_profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(codegen_profile, dict):
        return []
    raw = codegen_profile.get("flag_presence_overrides")
    if not isinstance(raw, list):
        raw = (codegen_profile.get("wireshark_lua") or {}).get("flag_presence_overrides")
    return [item for item in raw or [] if isinstance(item, dict)]


def _flag_for_field(
    field_node: Node,
    *,
    variant_path: str,
    flag_nodes: Iterable[Node],
    codegen_profile: Optional[Dict[str, Any]],
) -> Optional[Node]:
    if str(field_node.size_bits or "").strip() == "1" or "flag" in str(field_node.name or "").lower():
        return None
    overrides = _profile_flag_overrides(codegen_profile)
    for override in overrides:
        for flag in flag_nodes:
            if _flag_override_matches(override, variant_path=variant_path, field=field_node, flag=flag):
                return flag

    evidence = " ".join(
        str(value or "")
        for value in list(field_node.constraints or []) + [field_node.description or ""]
    ).lower()
    if not evidence or "present" not in evidence or not ({"when", "if"} & set(_tokens(evidence))):
        return None
    field_aliases = _aliases(field_node.name)
    for flag in flag_nodes:
        if field_aliases & _aliases(flag.name):
            return flag
    return None


def _is_single_bit_condition_source(node: Optional[Node]) -> bool:
    if node is None:
        return False
    if str(node.kind or "").strip().lower() == "selector":
        return False
    size = _int_size_bits(node)
    if size == 1:
        return True
    name = str(node.name or "").lower()
    return "flag" in name or "presence" in name


def _flag_gates_by_leaf(
    parse_ir: ParseIR,
    children: Dict[int, List[int]],
) -> Dict[int, Tuple[Node, Optional[str]]]:
    gates: Dict[int, Tuple[Node, Optional[str]]] = {}
    for edge in parse_ir.edges:
        if str(edge.rel or "").strip().lower() != "condition_on":
            continue
        flag = parse_ir.nodes.get(int(edge.src))
        if not _is_single_bit_condition_source(flag):
            continue
        if _condition_value(edge) != 1:
            continue
        dst = parse_ir.nodes.get(int(edge.dst))
        if dst is None:
            continue
        contract_id = edge_constraint_id(edge)
        for leaf_id in _leaf_ids(int(dst.id), parse_ir, children):
            gates[int(leaf_id)] = (flag, contract_id)
    return gates


def _lowered_field(
    node: Node,
    parse_ir: ParseIR,
    *,
    root_id: int,
    flag: Optional[Node] = None,
    flag_contract_id: Optional[str] = None,
) -> LoweredField:
    compiled = _compile_bit_expr_relative_to(parse_ir, node.bit_start, root_id=int(root_id))
    base_bit_offset: Optional[int] = None
    byte_offset: Optional[int] = None
    bit_offset: Optional[int] = None
    value_terms: List[ValueTerm] = []
    if compiled is not None:
        base_bit_offset, terms = compiled
        if base_bit_offset >= 0:
            byte_offset = base_bit_offset // 8
            bit_offset = base_bit_offset % 8
        value_terms = _value_terms_to_objects(parse_ir, terms)

    flag_base_bit_offset: Optional[int] = None
    flag_byte_offset: Optional[int] = None
    flag_bit_offset: Optional[int] = None
    flag_mask: Optional[str] = None
    if flag is not None:
        flag_compiled = _compile_bit_expr_relative_to(parse_ir, flag.bit_start, root_id=int(root_id))
        flag_size = _int_size_bits(flag)
        if flag_compiled is not None:
            flag_base_bit_offset, _flag_terms = flag_compiled
            if flag_base_bit_offset >= 0:
                flag_byte_offset = flag_base_bit_offset // 8
                flag_bit_offset = flag_base_bit_offset % 8
                if flag_size is not None:
                    flag_mask = _subbyte_mask(flag_bit_offset, int(flag_size))

    if base_bit_offset is None and flag_base_bit_offset is not None and flag is not None:
        flag_size = _int_size_bits(flag)
        if flag_size is not None and f"{int(flag.id)}.bit_start" in str(node.bit_start or ""):
            base_bit_offset = flag_base_bit_offset + int(flag_size)
            if base_bit_offset >= 0:
                byte_offset = base_bit_offset // 8
                bit_offset = base_bit_offset % 8

    return LoweredField(
        node_id=int(node.id),
        name=str(node.name),
        path=_path_for(int(node.id), parse_ir.nodes),
        data_type=str(node.data_type),
        bit_start=node.bit_start,
        size_bits=node.size_bits,
        base_bit_offset=base_bit_offset,
        byte_offset=byte_offset,
        bit_offset=bit_offset,
        value_terms=value_terms,
        flag_node_id=int(flag.id) if flag is not None else None,
        flag_name=str(flag.name) if flag is not None else None,
        flag_base_bit_offset=flag_base_bit_offset,
        flag_byte_offset=flag_byte_offset,
        flag_bit_offset=flag_bit_offset,
        flag_mask=flag_mask,
        flag_contract_id=flag_contract_id,
    )


def build_stage_c_lowering_plan(
    parse_ir: ParseIR,
    *,
    codegen_profile: Optional[Dict[str, Any]] = None,
) -> StageCLoweringPlan:
    children = _children(parse_ir)
    explicit_flag_gates = _flag_gates_by_leaf(parse_ir, children)
    selector_dispatches: List[SelectorDispatch] = []
    length_checks: List[LengthCheck] = []
    contract_items = build_contract(parse_ir)
    edge_contract_ids = {
        (int(item.src), int(item.dst), str(item.rel or "").strip().lower()): item.id
        for item in contract_items
        if item.kind == "edge" and item.src is not None and item.dst is not None
    }

    for edge in parse_ir.edges:
        rel = str(edge.rel or "").strip().lower()
        if rel == "condition_on":
            selector = parse_ir.nodes.get(int(edge.src))
            variant = parse_ir.nodes.get(int(edge.dst))
            selector_value = _condition_value(edge)
            if _is_single_bit_condition_source(selector):
                continue
            if selector is None or variant is None or selector_value is None:
                continue
            leaves = [parse_ir.nodes[node_id] for node_id in _leaf_ids(int(variant.id), parse_ir, children)]
            flag_nodes = [
                node
                for node in leaves
                if str(node.kind or "").strip().lower() == "field"
                and str(node.size_bits or "").strip() == "1"
                and "flag" in str(node.name or "").lower()
            ]
            variant_path = _path_for(int(variant.id), parse_ir.nodes)
            lowered_fields: List[LoweredField] = []
            flag_fields: List[LoweredField] = []
            for node in leaves:
                flag_pair = explicit_flag_gates.get(int(node.id))
                flag = flag_pair[0] if flag_pair is not None else None
                flag_contract_id = flag_pair[1] if flag_pair is not None else None
                if flag is None:
                    flag = _flag_for_field(
                        node,
                        variant_path=variant_path,
                        flag_nodes=flag_nodes,
                        codegen_profile=codegen_profile,
                    )
                lowered = _lowered_field(
                    node,
                    parse_ir,
                    root_id=int(variant.id),
                    flag=flag,
                    flag_contract_id=flag_contract_id,
                )
                lowered_fields.append(lowered)
                if flag is not None:
                    flag_fields.append(lowered)
            selector_dispatches.append(
                SelectorDispatch(
                    selector_id=int(selector.id),
                    selector_name=str(selector.name),
                    selector_value=int(selector_value),
                    variant_id=int(variant.id),
                    variant_name=str(variant.name),
                    fields=lowered_fields,
                    flag_controlled_fields=flag_fields,
                )
            )
        elif rel == "length_of":
            src = parse_ir.nodes.get(int(edge.src))
            dst = parse_ir.nodes.get(int(edge.dst))
            if src is None or dst is None:
                continue
            length_checks.append(
                LengthCheck(
                    src=int(edge.src),
                    src_name=str(src.name),
                    dst=int(edge.dst),
                    dst_name=str(dst.name),
                    rel=rel,
                    formula=str(edge.formula or ""),
                    contract_id=edge_contract_ids.get((int(edge.src), int(edge.dst), rel)),
                )
            )

    selector_dispatches.sort(key=lambda item: (item.selector_id, item.selector_value, item.variant_id))
    length_checks.sort(key=lambda item: (item.src, item.dst, item.formula))
    return StageCLoweringPlan(
        selector_dispatches=selector_dispatches,
        length_checks=length_checks,
        constraint_diagnostics=contract_items,
    )


def stage_c_lowering_plan_to_dict(plan: StageCLoweringPlan) -> Dict[str, Any]:
    return asdict(plan)

