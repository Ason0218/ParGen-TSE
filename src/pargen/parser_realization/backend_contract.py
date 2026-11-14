from __future__ import annotations

"""Backend contracts for Stage C realization.

The checks in this module are intentionally backend-scoped, not protocol-scoped.
They describe API and runtime contracts of a target parser backend (for example
Wireshark C), and must not encode protocol facts such as field values or option semantics.
"""

import re
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BackendContractViolation:
    host: str
    rule_id: str
    severity: str
    file: str
    line: int
    message: str
    suggestion: str = ""

    def format_for_repair(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line > 0 else self.file
        out = f"backend contract violation [{self.rule_id}] {loc}: {self.message}"
        if self.suggestion:
            out += f" Suggested fix: {self.suggestion}"
        return out


REQUIRED_LOWERING_PRIMITIVES: Tuple[str, ...] = (
    "bitfield",
    "selector_dispatch",
    "condition_on",
    "length_of",
    "constraint_marker",
)


@dataclass(frozen=True)
class BackendAdapterContract:
    """Host-neutral primitive support declared by one Stage C backend.

    This is a conformance surface for backend adapters. It deliberately names
    parser-construction primitives, not protocol facts. A backend can satisfy a
    primitive with deterministic lowering, backend API rules, or a repair/gate
    hook, but it must declare where that responsibility lives.
    """

    host: str
    supported_primitives: Tuple[str, ...]
    primitive_emitters: Mapping[str, str]


_BACKEND_ADAPTER_CONTRACTS: Mapping[str, BackendAdapterContract] = {
    "wireshark_lua": BackendAdapterContract(
        host="wireshark_lua",
        supported_primitives=REQUIRED_LOWERING_PRIMITIVES,
        primitive_emitters={
            "bitfield": "ProtoField.uint*/bool with explicit mask over containing tvb range",
            "selector_dispatch": "TLV/variant materialization scaffold plus bounded fallback",
            "condition_on": "runtime constraint injection and gated variant/body parsing",
            "length_of": "tvb length checks and exact consumed-length runtime checks",
            "constraint_marker": "TreeItem:add_expert_info message containing BS_CONSTRAINT",
        },
    ),
    "wireshark_c": BackendAdapterContract(
        host="wireshark_c",
        supported_primitives=REQUIRED_LOWERING_PRIMITIVES,
        primitive_emitters={
            "bitfield": "hf_register_info bitmask or proto_tree_add_bits with mask-free hf",
            "selector_dispatch": "bounded switch/if dispatch over selector field",
            "condition_on": "variant consistency checks with registered Expert Info",
            "length_of": "tvb_reported_length_remaining checks and bounded subtree reads",
            "constraint_marker": "expert_add_info_format through registered ei_* field with BS_CONSTRAINT",
        },
    ),
    "scapy": BackendAdapterContract(
        host="scapy",
        supported_primitives=REQUIRED_LOWERING_PRIMITIVES,
        primitive_emitters={
            "bitfield": "BitField/BitEnumField groups aligned to byte boundaries",
            "selector_dispatch": "PacketLenField/PacketListField/custom Field dispatch for known selector values",
            "condition_on": "ConditionalField or custom Field guarded by selector/flag predicate",
            "length_of": "FieldLenField, length_from/count_from, and explicit trailing-byte validation",
            "constraint_marker": "_pargen_constraint_violations entries containing BS_CONSTRAINT",
        },
    ),
    "zeek_spicy": BackendAdapterContract(
        host="zeek_spicy",
        supported_primitives=REQUIRED_LOWERING_PRIMITIVES,
        primitive_emitters={
            "bitfield": "Spicy byte-multiple bitfield(N) with explicit bit-order",
            "selector_dispatch": "switch over selector with bounded value_bytes and &parse-from",
            "condition_on": "&requires or on %done checks for selector/flag predicates",
            "length_of": "bytes &size bindings and on %done exact-size checks",
            "constraint_marker": 'throw "BS_CONSTRAINT <id> ..." or &requires diagnostic',
        },
    ),
}


_WIRESHARK_C_ALIASES = {"wireshark_c", "wireshark-c", "wireshark"}
_WIRESHARK_LUA_ALIASES = {"wireshark_lua", "wireshark-lua"}
_ZEEK_SPICY_ALIASES = {"zeek_spicy", "zeek-spicy", "zeek"}
_SCAPY_ALIASES = {"scapy"}


def normalize_backend_host(host: str) -> str:
    h = str(host or "").strip().lower().replace("-", "_")
    if h in {x.replace("-", "_") for x in _WIRESHARK_C_ALIASES}:
        return "wireshark_c"
    if h in {x.replace("-", "_") for x in _WIRESHARK_LUA_ALIASES}:
        return "wireshark_lua"
    if h in {x.replace("-", "_") for x in _ZEEK_SPICY_ALIASES}:
        return "zeek_spicy"
    if h in _SCAPY_ALIASES:
        return "scapy"
    return h


def backend_contract_prompt(host: str, *, context: str = "parse_ir") -> str:
    """Return backend-contract guidance for codegen or repair prompts."""

    host_norm = normalize_backend_host(host)
    prompt = ""
    if host_norm == "wireshark_c":
        prompt = _wireshark_c_prompt(context=context)
    elif host_norm == "wireshark_lua":
        prompt = _wireshark_lua_prompt(context=context)
    elif host_norm == "zeek_spicy":
        prompt = _zeek_spicy_prompt(context=context)
    elif host_norm == "scapy":
        prompt = _scapy_prompt(context=context)
    if not prompt:
        return ""
    primitive_contract = _format_backend_adapter_contract(backend_adapter_contract(host_norm))
    if primitive_contract:
        return prompt.rstrip() + "\n\n" + primitive_contract
    return prompt


def backend_adapter_contract(host: str) -> BackendAdapterContract:
    """Return the primitive-support contract for a Stage C backend."""

    host_norm = normalize_backend_host(host)
    contract = _BACKEND_ADAPTER_CONTRACTS.get(host_norm)
    if contract is not None:
        return contract
    return BackendAdapterContract(host=host_norm, supported_primitives=tuple(), primitive_emitters={})


def validate_backend_adapter_contract(
    host: str,
    *,
    required_primitives: Sequence[str] = REQUIRED_LOWERING_PRIMITIVES,
) -> Tuple[BackendContractViolation, ...]:
    """Check that a backend declares a lowering strategy for every required primitive."""

    contract = backend_adapter_contract(host)
    supported = set(contract.supported_primitives)
    violations: List[BackendContractViolation] = []
    for primitive in required_primitives:
        emitter = str(contract.primitive_emitters.get(str(primitive), "") or "").strip()
        if primitive not in supported:
            violations.append(
                BackendContractViolation(
                    host=contract.host,
                    rule_id="backend_adapter.missing_primitive",
                    severity="error",
                    file="<backend-adapter-contract>",
                    line=0,
                    message=f"backend does not declare lowering primitive {primitive!r}",
                    suggestion="Declare a platform-specific emitter or repair/gate strategy for this primitive.",
                )
            )
        elif not emitter:
            violations.append(
                BackendContractViolation(
                    host=contract.host,
                    rule_id="backend_adapter.missing_emitter",
                    severity="error",
                    file="<backend-adapter-contract>",
                    line=0,
                    message=f"backend declares {primitive!r} but does not name an emitter strategy",
                    suggestion="Add a concrete backend API pattern, deterministic lowering hook, or repair/gate hook.",
                )
            )
    return tuple(violations)


def _format_backend_adapter_contract(contract: BackendAdapterContract) -> str:
    if not contract.supported_primitives:
        return ""
    lines = [
        "Host-neutral lowering primitives required by this backend adapter:",
    ]
    for primitive in contract.supported_primitives:
        emitter = str(contract.primitive_emitters.get(primitive, "") or "").strip()
        suffix = f": {emitter}" if emitter else ""
        lines.append(f"- {primitive}{suffix}")
    return "\n".join(lines)


def backend_contract_repair_rules(host: str) -> str:
    rules = backend_contract_prompt(host, context="repair").strip()
    if not rules:
        return ""
    return "Backend realization contract rules:\n" + rules


def validate_backend_realization(
    host: str,
    files: Mapping[str, str],
    *,
    max_violations: int = 64,
) -> Tuple[BackendContractViolation, ...]:
    """Check generated source for backend API patterns associated with build and runtime failures."""

    host_norm = normalize_backend_host(host)
    if host_norm == "wireshark_c":
        return tuple(_validate_wireshark_c(files, max_violations=max_violations))
    return tuple()


def format_backend_contract_violations(
    violations: Sequence[BackendContractViolation],
    *,
    max_items: int = 12,
) -> str:
    if not violations:
        return ""
    lines = ["Backend realization contract findings:"]
    for violation in list(violations)[: max(0, int(max_items))]:
        lines.append(f"- {violation.format_for_repair()}")
    remaining = len(violations) - max(0, int(max_items))
    if remaining > 0:
        lines.append(f"- ... ({remaining} more)")
    return "\n".join(lines)


def _wireshark_c_prompt(*, context: str) -> str:
    base = """
Wireshark-C backend contract:
- Register with Wireshark's declared dissector APIs and field arrays; use `hf_register_info` and `ei_register_info[]`.
- Always bounds-check before reading: use `tvb_reported_length_remaining()` / `tvb_captured_length_remaining()`.
- Report constraint violations through registered Expert Info fields; include `BS_CONSTRAINT` in violation messages when the IR contract asks for it.
- Never call proto_tree/proto_item add APIs with zero byte length or zero bit width. Skip absent optional fields instead of adding zero-length placeholders.
- Do not use scalar field handles (`FT_UINT8`, `FT_UINT16`, `FT_BOOLEAN`, etc.) as variable-length containers. Add scalar leaves at their exact wire width and use `FT_BYTES` or an existing parent item for variable-length subtrees.
- Do not use `proto_tree_add_bits*` with an `hf_register_info` bitmask; either register that bit-field with mask `0x0` when using bit offsets/widths, or add the containing byte/word with the field mask.
- Do not pass `NULL` as the registered expert field to `expert_add_info_format`.
"""
    if str(context or "").strip() == "parse_ir":
        base += """
- Request/response direction must follow Parse-IR `message_type`; for port-based backends classify by `destport == registered_port` for requests and `srcport == registered_port` for responses.
- When Parse-IR provides `bit_start` / `size_bits`, make offsets match those byte/bit positions instead of assuming all fields are sequential.
"""
    if str(context or "").strip() == "repair":
        base += """
- Apply the smallest source-level change that satisfies the backend contract and preserves positive seed parsing.
- Backend-contract fixes must not hard-code protocol-specific constants unless they are already present in the IR/codegen profile/source being repaired.
"""
    return textwrap.dedent(base).strip()


def _wireshark_lua_prompt(*, context: str) -> str:
    base = """
Wireshark-Lua backend contract:
- Bounds-check buffer slices before reading them.
- Use Wireshark Lua expert info for malformed/constraint violations when parsing can continue safely; prefer subtree:add_expert_info(...).
- Do not silently accept malformed lengths or selector/variant mismatches.
- Preserve Parse-IR bit spans exactly. For any field whose `size_bits` is not a whole number of bytes or whose
  `bit_start` is not byte-aligned, define the ProtoField with an explicit bitmask on the containing byte/word
  (for example `ProtoField.uint8(..., nil, 0xF0)` for a high-nibble field) and add that containing range to
  the tree. Do not expose sub-byte leaves as unmasked whole-byte fields.
- A computed local variable does not satisfy a Parse-IR field. Every leaf field in Parse-IR, including bitfields
  such as Version/Diagnostic/State/Flags, must have its own ProtoField and its own `subtree:add(field, tvb(...))`
  call. A combined byte field is allowed only as an extra display aid, never as a replacement for leaf fields.
- Use absolute tvb ranges that correspond to resolved Parse-IR bit positions. When a subtree starts at offset X,
  do not add X again to child field offsets; child `tvb(...)` slices should still point to the absolute packet
  offsets implied by the child bit_start. A variant body whose first child is at `body.bit_start + 0` starts at
  the body offset, not at body offset plus another header length.
- Use MSB-first bit numbering within network-order bytes unless the Parse-IR says otherwise.
- Prefer Lua `bit32.*` operations with the project compatibility shim over relying on a global `bit` module.
"""
    if str(context or "").strip() == "parse_ir":
        base += """
- Request/response direction must follow Parse-IR `message_type`; for port-based protocols classify by destination/source port.
"""
    return textwrap.dedent(base).strip()


def _zeek_spicy_prompt(*, context: str) -> str:
    base = """
Zeek+Spicy backend contract:
- Prefer failing a single packet cleanly over producing silently wrong fields.
- Use `&requires` for hard invariants and `&size` / length bindings for variable-length regions.
- Respect bit-level fields exactly; use Spicy `bitfield(N)` for sub-byte fields instead of approximating them as whole bytes.
- Spicy bitfield containers must use byte-multiple widths accepted by `spicyz` (8, 16, 24, 32, ...), not standalone `bitfield(1)`..`bitfield(7)`. If a sub-byte selector is followed immediately by a value field, parse the enclosing byte-multiple region as one bitfield wrapper and expose the selector/value as bit ranges inside that wrapper.
- Use unique names for distinct Spicy bitfield containers; reusing a generic name such as `flags` for several different layouts makes exported sub-byte fields ambiguous.
- For TLV/tag-length-value protocols, do not stop at a generic raw Value field when the Parse-IR/profile names child fields or variants. Consume the length-delimited value into `value_bytes: bytes &size=self.<len>` and parse a variant-specific sub-unit from that buffer with `&parse-from=self.value_bytes`.
- Dispatch TLV bodies with Spicy `switch (self.<tag>) { value -> body: BodyType &parse-from=self.value_bytes; * -> raw_value: bytes &parse-from=self.value_bytes; };`. This is a backend API pattern, not a protocol-specific rule.
- For grouped/nested TLVs, parse the nested item array from the group's value buffer, bounded by the value length; never use `&eod` at the top packet scope for a grouped IE body unless the enclosing length is exactly that scope.
- Expand only grouped variants whose child fields are defined in the Parse-IR; keep other grouped values as bounded bytes to avoid unsupported helper fields.
- Runtime constraints must reject malformed packets. Use `: void &requires=(...)` for simple field predicates, and use `on %done { if (...) throw "BS_CONSTRAINT <id> ..."; }` for cross-field constraints such as selector/variant consistency, length relations, and reserved-zero bits.
"""
    if str(context or "").strip() == "repair":
        base += """
- Apply the smallest valid Spicy change. Prefer introducing a missing sub-unit and connecting it with `&parse-from` over rewriting the whole analyzer.
- Backend-contract fixes must not hard-code protocol-specific constants unless they already appear in the Parse-IR, codegen profile, or generated source being repaired.
"""
    return textwrap.dedent(base).strip()


def _scapy_prompt(*, context: str) -> str:
    return textwrap.dedent(
        """
Scapy backend contract:
- Implement on-wire structure as Scapy `Packet` layers with `fields_desc`.
- Represent length and selector fields as standalone fields; use them to bind variable-length fields and conditional variants.
- For repeated TLV/tag-length-value regions, do not collapse the whole region into one raw `StrLenField`
  when the Parse-IR/profile names item fields or variants. Model the repeated item with `PacketListField`
  bounded by the enclosing length/remaining bytes, and expose item tag/type, length, and value as public Scapy fields.
- For one TLV item, expose the length field with `FieldLenField`/`ShortField` and bind the value with
  `PacketLenField`, `StrLenField`, or `ConditionalField` using `length_from=lambda pkt: pkt.<length>`.
- For selector-dispatched TLV bodies, prefer guarded `ConditionalField(PacketLenField(...), lambda pkt: pkt.<tag> == K)`
  or a small custom Field/Packet that consumes exactly the value length. Do not let a generic raw value consume bytes
  before typed child fields that are present in the Parse-IR.
- Never pass a `Packet` class directly to `ConditionalField`; Scapy requires the first argument to be a Field instance
  such as `PacketLenField(...)`, `PacketListField(...)`, or `PacketField(...)`.
- For grouped/nested TLVs, parse nested TLV items from the group's bounded value bytes; do not parse until packet end
  unless the enclosing group length is exactly the remaining packet length.
- If the Parse-IR/profile has a generic TLV/Information-Element item with tag/type, length, and value, and also names
  variant body fields, parse every body element through the TLV wrapper first. Do not place variant body fields directly
  in the enclosing message body unless the Parse-IR says they are not tag-length-value encoded.
- Keep import side effects limited to class definitions and explicit `bind_layers` registration.
- For `Packet` classes used as `PacketListField` elements, define `extract_padding(self, s): return b"", s`
  unless the class has an explicit, schema-defined Scapy payload layer. This prevents one list element from
  absorbing the bytes that belong to later list elements or following sections.
- For fixed/nested headers represented with `PacketField`, define `extract_padding(self, s): return b"", s` on the
  nested `Packet` class unless it intentionally owns a payload layer. Otherwise Scapy may hide following fields as the
  child packet's Raw payload.
- Consecutive `BitField`/`BitEnumField` declarations must reach byte boundaries before any non-bit field and at the
  end of `fields_desc`; add explicit reserved/spare bits when a selector occupies only part of an octet.
- Reportable fields must be exposed through public Scapy Packet/Field APIs. ParGen-specific reporting hooks are diagnostic
  only and are disabled in main parser-quality exports.
        """
    ).strip()


_SCALAR_FT_WIDTHS = {
    "FT_UINT8": 1,
    "FT_INT8": 1,
    "FT_UINT16": 2,
    "FT_INT16": 2,
    "FT_UINT24": 3,
    "FT_INT24": 3,
    "FT_UINT32": 4,
    "FT_INT32": 4,
    "FT_FLOAT": 4,
    "FT_UINT64": 8,
    "FT_INT64": 8,
    "FT_DOUBLE": 8,
}


@dataclass(frozen=True)
class _WiresharkFieldInfo:
    ft_type: str
    bitmask: Optional[int] = None


def _validate_wireshark_c(
    files: Mapping[str, str],
    *,
    max_violations: int,
) -> List[BackendContractViolation]:
    violations: List[BackendContractViolation] = []
    hf_info: Dict[str, _WiresharkFieldInfo] = {}
    for text in files.values():
        hf_info.update(_collect_wireshark_hf_info(text))

    for name, text in files.items():
        if not str(name).endswith((".c", ".h")):
            continue
        _add_wireshark_c_file_violations(
            host="wireshark_c",
            file_name=str(name),
            text=text,
            hf_info=hf_info,
            violations=violations,
            max_violations=max_violations,
        )
        if len(violations) >= max_violations:
            break
    return violations[:max_violations]


def _collect_wireshark_hf_info(text: str) -> Dict[str, _WiresharkFieldInfo]:
    out: Dict[str, _WiresharkFieldInfo] = {}
    # Matches the common hf_register_info entry layout:
    # { &hf_name, { "Label", "filter.name", FT_UINT8, BASE_DEC, NULL, 0x0, ... } }
    entry_re = re.compile(
        r"\{\s*&\s*(?P<hf>hf_[A-Za-z0-9_]+)\s*,\s*\{\s*"
        r"\"(?:\\.|[^\"])*\"\s*,\s*\"(?:\\.|[^\"])*\"\s*,\s*"
        r"(?P<ft>FT_[A-Za-z0-9_]+)\s*,\s*"
        r"(?P<display>[^,]+)\s*,\s*(?P<strings>[^,]+)\s*,\s*"
        r"(?P<mask>0x[0-9A-Fa-f]+|\d+)\s*,",
        flags=re.DOTALL,
    )
    for match in entry_re.finditer(text or ""):
        hf = match.group("hf")
        ft_type = match.group("ft")
        bitmask = _safe_int(match.group("mask"))
        out[hf] = _WiresharkFieldInfo(ft_type=ft_type, bitmask=bitmask)
    return out


def _add_wireshark_c_file_violations(
    *,
    host: str,
    file_name: str,
    text: str,
    hf_info: Mapping[str, _WiresharkFieldInfo],
    violations: List[BackendContractViolation],
    max_violations: int,
) -> None:
    def add(rule_id: str, severity: str, pos: int, message: str, suggestion: str = "") -> None:
        if len(violations) >= max_violations:
            return
        violations.append(
            BackendContractViolation(
                host=host,
                rule_id=rule_id,
                severity=severity,
                file=file_name,
                line=_line_no(text, pos),
                message=message,
                suggestion=suggestion,
            )
        )

    if re.search(r"#include\s+[<\"]wireshark/config\.h[>\"]", text):
        add(
            "wireshark_c.config_include",
            "error",
            text.find("wireshark/config.h"),
            "Do not include <wireshark/config.h> from generated plugins.",
            "Use #include \"config.h\" plus Wireshark public headers such as <epan/packet.h>.",
        )

    for call in _iter_c_calls(text, ("proto_tree_add_item", "proto_tree_add_bits_item", "proto_tree_add_bits_ret_val")):
        args = _split_c_args(call.args)
        if call.name == "proto_tree_add_item" and len(args) >= 6:
            hf = _strip_expr(args[1])
            length_expr = _strip_expr(args[4])
            length_val = _safe_int(length_expr)
            if length_val == 0:
                add(
                    "wireshark_c.zero_length_item",
                    "error",
                    call.pos,
                    f"`{call.name}` is called for `{hf}` with zero byte length.",
                    "Skip absent optional fields instead of adding a zero-length item.",
                )
            info = hf_info.get(hf)
            if info is not None and info.ft_type in _SCALAR_FT_WIDTHS:
                expected_width = _SCALAR_FT_WIDTHS[info.ft_type]
                if length_val is None:
                    add(
                        "wireshark_c.scalar_variable_container",
                        "error",
                        call.pos,
                        f"`{hf}` is registered as {info.ft_type} but is added with non-literal/variable length `{length_expr}`.",
                        f"Add `{hf}` with exactly {expected_width} byte(s), and use FT_BYTES or a parent item for variable-length subtrees.",
                    )
                elif length_val != expected_width:
                    add(
                        "wireshark_c.scalar_width_mismatch",
                        "error",
                        call.pos,
                        f"`{hf}` is registered as {info.ft_type} but is added with length {length_val}; expected {expected_width}.",
                        f"Use length {expected_width} for this scalar field.",
                    )
        elif call.name in {"proto_tree_add_bits_item", "proto_tree_add_bits_ret_val"} and len(args) >= 5:
            hf = _strip_expr(args[1])
            bit_width = _strip_expr(args[4])
            bit_width_val = _safe_int(bit_width)
            if bit_width_val == 0:
                add(
                    "wireshark_c.zero_width_bits",
                    "error",
                    call.pos,
                    f"`{call.name}` is called for `{hf}` with zero bit width.",
                    "Skip absent optional bit fields instead of adding a zero-width item.",
                )
            info = hf_info.get(hf)
            if info is not None and (info.bitmask or 0) != 0:
                add(
                    "wireshark_c.bits_api_with_hf_bitmask",
                    "error",
                    call.pos,
                    f"`{call.name}` is used with `{hf}`, whose hf_register_info bitmask is non-zero.",
                    "Either register this field with mask 0x0 for bit-offset APIs, or add the containing byte/word with the field mask.",
                )

    for call in _iter_c_calls(text, ("expert_add_info_format",)):
        args = _split_c_args(call.args)
        if len(args) >= 3 and (_strip_expr(args[1]) == "NULL" or _strip_expr(args[2]) == "NULL"):
            add(
                "wireshark_c.null_expert_field",
                "error",
                call.pos,
                "`expert_add_info_format` is called with NULL proto_item or expert field.",
                "Pass the relevant proto_item and a registered expert field such as `&ei_<name>`.",
            )


@dataclass(frozen=True)
class _CCall:
    name: str
    args: str
    pos: int


def _iter_c_calls(text: str, names: Iterable[str]) -> Iterable[_CCall]:
    name_set = tuple(names)
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in name_set) + r")\s*\(")
    for match in pattern.finditer(text or ""):
        open_pos = text.find("(", match.end() - 1)
        close_pos = _find_matching_paren(text, open_pos)
        if close_pos is None:
            continue
        yield _CCall(name=match.group(1), args=text[open_pos + 1 : close_pos], pos=match.start())


def _find_matching_paren(text: str, open_pos: int) -> Optional[int]:
    if open_pos < 0 or open_pos >= len(text) or text[open_pos] != "(":
        return None
    depth = 0
    in_string: Optional[str] = None
    escape = False
    i = open_pos
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in {"\"", "'"}:
            in_string = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _split_c_args(args: str) -> List[str]:
    out: List[str] = []
    start = 0
    depth = 0
    in_string: Optional[str] = None
    escape = False
    for i, ch in enumerate(args or ""):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            continue
        if ch in {"\"", "'"}:
            in_string = ch
            continue
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            continue
        if ch == "," and depth == 0:
            out.append(args[start:i].strip())
            start = i + 1
    tail = (args[start:] or "").strip()
    if tail:
        out.append(tail)
    return out


def _strip_expr(expr: str) -> str:
    return re.sub(r"\s+", " ", str(expr or "").strip())


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Avoid treating expressions like `len - 2` as an integer.
    if not re.fullmatch(r"[+-]?(?:0x[0-9A-Fa-f]+|\d+)", s):
        return None
    try:
        return int(s, 0)
    except Exception:
        return None


def _line_no(text: str, pos: int) -> int:
    try:
        return (text or "").count("\n", 0, max(0, int(pos))) + 1
    except Exception:
        return 0
