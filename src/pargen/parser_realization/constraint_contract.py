from __future__ import annotations

"""
Constraint contract helpers for Stage C codegen.

Goal:
- Make Parse-IR constraints *portable* across host backends by defining a stable ID/tag
  for each constraint that must be validated at runtime.
- Enable deterministic coverage checks without using exp_constraint during codegen.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .parse_ir import Edge, ParseIR

_WS_RE = re.compile(r"\\s+")
_DYNAMIC_TAG_HINT_RE = re.compile(
    r"(BS_CONSTRAINT\s*\{)|(BS_CONSTRAINT\s*%[^%])|(BS_CONSTRAINT\s*\+)|(BS_CONSTRAINT\s*\.format\s*\()"
    # String concatenation after the literal: Lua `"BS_CONSTRAINT " .. id`, `'BS_CONSTRAINT ' + id`.
    r"|(BS_CONSTRAINT\s*[\"']\s*(?:\.\.|\+))",
    flags=re.IGNORECASE,
)


def _normalize_text(value: object) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    s = (
        s.replace("：", ":")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("\\u00a0", " ")
    )
    return _WS_RE.sub(" ", s).strip()


def _hash8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _maybe_single_enum_value(clause: str) -> Optional[int]:
    normalized = _normalize_text(clause)
    if not normalized:
        return None
    head, sep, tail = normalized.partition(":")
    if sep != ":":
        return None
    if head.strip().lower() != "enum":
        return None
    tail = tail.strip()
    if not tail or "|" in tail:
        return None
    try:
        return int(tail, 0)
    except Exception:
        return None


def _bit_start_is_parent_zero_offset(*, node_bit_start: object, parent_id: int) -> bool:
    s = str(node_bit_start or "").replace(" ", "")
    if not s:
        return False
    # Typical Stage B encoding: "<parent_id>.bit_start + 0".
    return s.startswith(f"{int(parent_id)}.bit_start") and s.endswith("+0")


def _is_redundant_selector_code_enum(*, parse_ir: ParseIR, node_id: int, clause: str) -> bool:
    """
    Some Stage B trees represent a `condition_on` selector constant twice:

    - the selector (e.g. `Option_Tag`) decides a variant via `condition_on: val(sel) == K`
    - inside the chosen variant, there is a synthetic `*_Code` field with `enum: K`

    That inner `enum: K` is redundant (and sometimes does not correspond to real bytes),
    so we do not require a standalone contract tag for it.
    """
    value = _maybe_single_enum_value(clause)
    if value is None:
        return False
    node = parse_ir.nodes.get(int(node_id))
    if node is None or node.parent_id is None:
        return False
    parent = parse_ir.nodes.get(int(node.parent_id))
    if parent is None or parent.kind != "variant":
        return False
    if not _bit_start_is_parent_zero_offset(node_bit_start=node.bit_start, parent_id=int(node.parent_id)):
        return False

    target = f"=={int(value)}"
    for edge in parse_ir.edges:
        if str(getattr(edge, "rel", "") or "").strip().lower() != "condition_on":
            continue
        if int(getattr(edge, "dst", -1)) != int(node.parent_id):
            continue
        src_node = parse_ir.nodes.get(int(getattr(edge, "src", -1)))
        if src_node is not None and src_node.kind != "selector":
            continue
        formula = str(getattr(edge, "formula", "") or "").replace(" ", "")
        if target in formula:
            return True
    return False


def _tokens(value: object) -> List[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.replace("_", " ").replace("-", " "))


_GENERIC_FLAG_EDGE_TOKENS = {
    "common",
    "field",
    "flag",
    "header",
    "ie",
    "information",
    "element",
    "item",
    "length",
    "message",
    "present",
    "type",
    "value",
    "variant",
    "variants",
}

_FLAG_EDGE_ALIASES = {
    "v4": {"ipv4"},
    "ipv4": {"v4"},
    "v6": {"ipv6"},
    "ipv6": {"v6"},
}


def _meaningful_flag_tokens(value: object) -> set[str]:
    return {
        tok
        for tok in _tokens(value)
        if tok and tok not in _GENERIC_FLAG_EDGE_TOKENS and not tok.isdigit()
    }


def _condition_flag_name_relevant(*, parse_ir: ParseIR, edge: Edge) -> bool:
    src_node = parse_ir.nodes.get(int(getattr(edge, "src", -1)))
    dst_node = parse_ir.nodes.get(int(getattr(edge, "dst", -1)))
    if src_node is None or dst_node is None:
        return False
    src_name = str(getattr(src_node, "name", "") or "")
    if "flag" not in src_name.lower():
        return True
    src_tokens = _meaningful_flag_tokens(src_name)
    dst_tokens = _meaningful_flag_tokens(getattr(dst_node, "name", ""))
    if src_tokens & dst_tokens:
        return True
    expanded = set(src_tokens)
    for tok in list(src_tokens):
        expanded.update(_FLAG_EDGE_ALIASES.get(tok, set()))
    return bool(expanded & dst_tokens)


def _endswith_tokens(longer: Sequence[str], shorter: Sequence[str]) -> bool:
    if not longer or not shorter or len(longer) <= len(shorter):
        return False
    return list(longer[-len(shorter) :]) == list(shorter)


def _contains_tokens(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not haystack or not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    for idx in range(0, len(haystack) - n + 1):
        if list(haystack[idx : idx + n]) == list(needle):
            return True
    return False


def _same_static_size(a: object, b: object) -> bool:
    try:
        return int(str(a).strip(), 0) == int(str(b).strip(), 0)
    except Exception:
        return _normalize_text(a) == _normalize_text(b) and bool(_normalize_text(a))


def _is_redundant_prefixed_container_alias_constraint(*, parse_ir: ParseIR, node_id: int, clause: str) -> bool:
    """Skip duplicate node obligations introduced by prefixed alias fields.

    Stage-B repair can carry both a canonical field (`Version`) and a
    container-prefixed alias (`Common Header Version`) with the same constraint.
    Requiring two runtime tags for the same semantic obligation makes Stage C
    chase unstable alias offsets.  This only collapses aliases when the longer
    field name explicitly contains its parent/container name and ends with a
    shorter field name that has the same width and normalized constraint.
    """

    node = parse_ir.nodes.get(int(node_id))
    if node is None or node.parent_id is None:
        return False
    parent = parse_ir.nodes.get(int(node.parent_id))
    if parent is None:
        return False

    node_tokens = _tokens(node.name)
    parent_tokens = _tokens(parent.name)
    if not node_tokens or not parent_tokens:
        return False
    if not _contains_tokens(node_tokens, parent_tokens):
        return False

    normalized_clause = _normalize_text(clause)
    for other_id, other in parse_ir.nodes.items():
        if int(other_id) == int(node_id):
            continue
        other_tokens = _tokens(other.name)
        if not _endswith_tokens(node_tokens, other_tokens):
            continue
        if not _same_static_size(getattr(node, "size_bits", None), getattr(other, "size_bits", None)):
            continue
        if any(_normalize_text(raw) == normalized_clause for raw in (other.constraints or [])):
            return True
    return False


def node_constraint_id(*, node_id: int, clause: str) -> Optional[str]:
    normalized = _normalize_text(clause)
    if not normalized:
        return None
    kind = normalized.split(":", 1)[0].strip().lower()
    if kind not in {"enum", "range", "complement", "constraint_ir", "formula"}:
        return None
    basis = f"node|{int(node_id)}|{kind}|{normalized}"
    return f"N{int(node_id)}_{kind}_{_hash8(basis)}"


def _node_obligation_type(clause: str) -> str:
    normalized = _normalize_text(clause)
    kind = normalized.split(":", 1)[0].strip().lower()
    if kind == "constraint_ir":
        match = re.search(r'"kind"\s*:\s*"([^"]+)"', normalized)
        if match:
            ir_kind = match.group(1).strip().lower()
            if ir_kind:
                return f"constraint_ir:{ir_kind}"
        return "constraint_ir"
    if kind:
        return kind
    return "node_constraint"


def _is_unlowerable_constraint_ir_clause(clause: str) -> bool:
    normalized = _normalize_text(clause)
    head, sep, tail = normalized.partition(":")
    if sep != ":" or head.strip().lower() != "constraint_ir":
        return False
    try:
        payload = json.loads(tail.strip())
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    ir_kind = str(payload.get("kind") or "").strip().lower()
    # This Stage-B hint is descriptive unless a separate length_of or
    # executable length_relation encodes the actual formula.
    return ir_kind == "conditional_field"


def edge_constraint_id(edge: Edge) -> Optional[str]:
    rel = str(getattr(edge, "rel", "") or "").strip().lower()
    if rel == "condition_on":
        # A valid selector choosing this branch is not a violation.  The
        # runtime obligation is branch consistency: if this branch is selected,
        # its required body/fields/length relations must be present and
        # consumed consistently; if it is not selected, its body must not be
        # consumed as this variant.
        return f"E{int(edge.src)}_{int(edge.dst)}_condition_on"
    if rel not in {"length_of"}:
        return None
    mt = _normalize_text(getattr(edge, "message_type", "") or "").lower() or "bidirectional"
    formula = _normalize_text(getattr(edge, "formula", "") or "")
    basis = f"edge|{int(edge.src)}|{int(edge.dst)}|{rel}|{mt}|{formula}"
    return f"E{int(edge.src)}_{int(edge.dst)}_{rel}_{_hash8(basis)}"


def _is_fallback_condition_on_edge(*, parse_ir: ParseIR, edge: Edge) -> bool:
    rel = str(getattr(edge, "rel", "") or "").strip().lower()
    if rel != "condition_on":
        return False
    formula = _normalize_text(getattr(edge, "formula", "") or "").lower()
    if "!=" not in formula or "==" in formula:
        return False
    dst_node = parse_ir.nodes.get(int(getattr(edge, "dst", -1)))
    if dst_node is None:
        return False
    tokens = set(_tokens(getattr(dst_node, "name", "")) + _tokens(getattr(dst_node, "description", "")))
    return bool(tokens & {"fallback", "unknown", "other", "unrecognized", "unsupported"})


@dataclass(frozen=True)
class ContractItem:
    id: str
    kind: str  # node|edge
    obligation_type: Optional[str] = None
    node_id: Optional[int] = None
    node_name: Optional[str] = None
    clause: Optional[str] = None
    src: Optional[int] = None
    src_name: Optional[str] = None
    dst: Optional[int] = None
    dst_name: Optional[str] = None
    rel: Optional[str] = None
    formula: Optional[str] = None
    message_type: Optional[str] = None


def build_contract(parse_ir: ParseIR) -> List[ContractItem]:
    items: List[ContractItem] = []

    for nid in sorted(parse_ir.nodes.keys()):
        node = parse_ir.nodes[nid]
        for raw in node.constraints or []:
            if _is_redundant_selector_code_enum(parse_ir=parse_ir, node_id=int(node.id), clause=str(raw)):
                continue
            if _is_redundant_prefixed_container_alias_constraint(
                parse_ir=parse_ir,
                node_id=int(node.id),
                clause=str(raw),
            ):
                continue
            if _is_unlowerable_constraint_ir_clause(str(raw)):
                continue
            cid = node_constraint_id(node_id=int(node.id), clause=str(raw))
            if not cid:
                continue
            items.append(
                ContractItem(
                    id=cid,
                    kind="node",
                    obligation_type=_node_obligation_type(str(raw)),
                    node_id=int(node.id),
                    node_name=str(node.name),
                    clause=_normalize_text(raw),
                )
            )

    for edge in parse_ir.edges:
        if _is_fallback_condition_on_edge(parse_ir=parse_ir, edge=edge):
            continue
        if str(getattr(edge, "rel", "") or "").strip().lower() == "condition_on" and not _condition_flag_name_relevant(
            parse_ir=parse_ir, edge=edge
        ):
            continue
        cid = edge_constraint_id(edge)
        if not cid:
            continue
        src_node = parse_ir.nodes.get(int(edge.src))
        dst_node = parse_ir.nodes.get(int(edge.dst))
        items.append(
            ContractItem(
                id=cid,
                kind="edge",
                obligation_type=str(edge.rel),
                src=int(edge.src),
                src_name=str(src_node.name) if src_node is not None else None,
                dst=int(edge.dst),
                dst_name=str(dst_node.name) if dst_node is not None else None,
                rel=str(edge.rel),
                formula=_normalize_text(edge.formula),
                message_type=_normalize_text(edge.message_type).lower() or "bidirectional",
            )
        )

    items.sort(key=lambda x: x.id)
    return items


def tag_for(contract_id: str) -> str:
    return f"BS_CONSTRAINT {str(contract_id).strip()}"


def find_missing_tags(*, files: Dict[str, str], items: Sequence[ContractItem], host_name: Optional[str] = None) -> List[str]:
    if not items:
        return []
    host = str(host_name or "").strip().lower().replace("-", "_")

    # Host-specific sanity constraints beyond "tag is present in code".
    #
    # For Scapy, constraint checks must run on *parsing* (pcap dissection), not only on packet build.
    # The most portable hook is `Packet.post_dissect()`. If it is missing, treat the contract as missing
    # so Stage C codegen/repair adds the runtime hook.
    if host == "scapy":
        def _has_scapy_post_dissect_validation(text: str) -> bool:
            if not text or "post_dissect" not in text:
                return False
            for m in re.finditer(r"(?m)^(?P<indent>[ \t]*)def\s+post_dissect\s*\(", text):
                indent = m.group("indent") or ""
                # Find the next method at the same indentation level to bound the function body.
                body_start = m.end()
                next_def = re.search(rf"(?m)^{re.escape(indent)}def\s+\w+\s*\(", text[body_start:])
                end = len(text) if next_def is None else body_start + next_def.start()
                block = text[m.start() : end]
                # Evidence of actual constraint/violation logic running on parse.
                if "BS_CONSTRAINT" in block or "_pargen_constraint_violations" in block or "violations.append" in block:
                    return True
            return False

        def _has_scapy_injected_parse_wrapper(text: str) -> bool:
            if not text or "__PARGEN_CONSTRAINT_VALIDATION__" not in text:
                return False
            # The deterministic Scapy injector wraps each generated Packet
            # class by assigning `cls.post_dissect = _wrapped`. That is a
            # parse-time validation hook even though the generated source does
            # not contain a literal `def post_dissect(...)` for every class.
            return "_bs_wrap" in text and "post_dissect" in text and "_BS_CONSTRAINTS" in text

        if not any(
            _has_scapy_post_dissect_validation(v or "") or _has_scapy_injected_parse_wrapper(v or "")
            for v in files.values()
        ):
            return [it.id for it in items]

    haystack = "\\n".join([v or "" for v in files.values()])
    # Some hosts build the tag dynamically, e.g. `f\"BS_CONSTRAINT {constraint_id}\"`.
    # In that case, require:
    # - the contract id appears in the code (as a literal), and
    # - there is evidence the implementation emits `BS_CONSTRAINT ...` at runtime.
    has_dynamic_tag_template = bool(_DYNAMIC_TAG_HINT_RE.search(haystack))
    missing: List[str] = []
    for item in items:
        tag = tag_for(item.id)
        if tag not in haystack:
            if has_dynamic_tag_template and (str(item.id) in haystack):
                continue
            missing.append(item.id)
    return missing


def format_contract_for_prompt(*, host_name: str, items: Sequence[ContractItem]) -> str:
    host = str(host_name or "").strip().lower().replace("-", "_")
    if not items:
        return "(no constraints in contract)"

    rules = [
        "ParGen constraint contract (STRICT):",
        "- Contract items are backend-independent typed runtime obligations derived from the Parse-IR/PKG, not protocol-specific hard-coded cases.",
        "- You MUST implement runtime validation checks for every contract item below.",
        "- For each check, you MUST emit a deterministic tag EXACTLY: `BS_CONSTRAINT <id>`.",
        "- Only emit `BS_CONSTRAINT ...` when the constraint is VIOLATED (do not emit tags on success).",
        "- For Wireshark-C/Wireshark-Lua/Scapy: put the tag inside the emitted diagnostic string.",
        "- For Zeek+Spicy: if the backend cannot attach messages to parse failures, put the tag in a nearby comment line.",
        "- The generated parser will be gated with normal positive-seed packets and mutated constraint-violating packets.",
        "- Normal positive-seed packets MUST keep parsing without BS_CONSTRAINT diagnostics; do not satisfy a counterexample by broad packet rejection.",
        "- Mutated constraint-violating packets MUST be rejected by an explicit BS_CONSTRAINT diagnostic for the violated contract item, not by a generic crash, missing-layer parse failure, or unrelated exception.",
        "- On truncation/missing-field situations relevant to a constraint, emit the tag as a violation (do NOT silently skip).",
        "- For `length_of` edges: enforce exact byte consumption. If the expected length does not match the bytes available/consumed (truncation or leftover), emit the tag.",
        "- For `condition_on` edges: implement branch consistency. Do NOT report a violation merely because a valid selector chooses a branch; report a violation only when the selected branch's required fields/body/length relation is missing or inconsistent, or when bytes are consumed as a variant not selected by the condition.",
        "- Branch-scoped checks matter: if a contract item belongs to a selected variant, put its check on the code path that parses that variant. Do not satisfy the contract by placing the tag in a generic helper, unreachable code, comments, or a different selector branch.",
        "- If the Stage C lowering plan lists selector dispatches or length checks, treat those entries as executable obligations: the selector branch must bounds-check the branch body before reading fields, and the length check must run after the length source is decoded but before accepting the packet as normal.",
        "- For `constraint_ir` and `formula` node clauses: treat the clause as a typed runtime obligation (for example conditional_zero, bitmask_zero, bounded length relation, count/repeat relation, pointer/bounds relation, or derived-length minimum) and lower it using host APIs documented in Host-IR/API docs.",
        "- Do NOT hide violations behind guards like `if len > 0` / `if remaining >= ...`: if you cannot parse due to missing bytes, or a computed length is invalid (including 0 when a range/size requires >0), emit the tag before returning/skipping.",
    ]

    host_hint = ""
    if host in {"wireshark_c", "wireshark-c"}:
        host_hint = "Host note (wireshark_c): use Expert Info (PI_MALFORMED/PI_ERROR) and include the tag in the message."
    elif host in {"wireshark_lua", "wireshark-lua"}:
        host_hint = (
            "Host note (wireshark_lua): use subtree:add_expert_info(PI_MALFORMED, PI_ERROR, ...) and include the tag "
            "(emit it even when the buffer is truncated / a length check fails before parsing any bytes)."
        )
    elif host == "scapy":
        host_hint = (
            "Host note (scapy): append violations to `pkt._pargen_constraint_violations` (list[str]) and include the tag in the string. "
            "IMPORTANT: run validation on parsing by implementing `post_dissect()` (and optionally also `post_build()`). "
            "Do NOT rely on `post_build()` only (it is not called when dissecting PCAP bytes). "
            "Avoid parse-time exceptions: do not write ConditionalField conditions that can throw (e.g., `pkt.kind >= 6` when the field may be absent/None). "
            "Prefer checking parsed presence via `'<field>' in pkt.fields` and guard comparisons with `is not None`. "
            "For truncation/missing bytes, do NOT attempt to read past available data; record the violation tag and stop cleanly."
        )
    elif host in {"zeek_spicy", "zeek-spicy"}:
        host_hint = (
            "Host note (zeek_spicy): use &requires for local field checks and `on %done { throw \"BS_CONSTRAINT <id> ...\"; }` "
            "for cross-field checks. In binary exp_constraint mode, the malformed packet should fail parsing; include tags as comments "
            "only when Spicy cannot attach a runtime message."
        )
    if host_hint:
        rules.append(host_hint)

    lines: List[str] = []
    for item in items:
        if item.kind == "node":
            lines.append(
                f"- id={item.id} kind=node obligation_type={item.obligation_type} "
                f"node_id={item.node_id} node_name={item.node_name} clause={item.clause}"
            )
        else:
            lines.append(
                f"- id={item.id} kind=edge obligation_type={item.obligation_type} rel={item.rel} src={item.src} dst={item.dst} "
                f"src_name={item.src_name} dst_name={item.dst_name} message_type={item.message_type} "
                f"formula={item.formula}"
            )

    return "\\n".join(rules + ["", "Contract items:", *lines])


def iter_tags(items: Sequence[ContractItem]) -> Iterable[str]:
    for item in items:
        yield tag_for(item.id)
