"""Check constraint coverage using Stage A section records."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..constraint_ir import (
    constraint_ir_blob,
    constraint_ir_field_ids,
    constraint_ir_to_text,
    parse_constraint_ir,
)
from .syntax_validator import (
    FixHint,
    Issue,
    IssueType,
    Severity,
    TargetRef,
    ValidationReport,
)


_CONTROL_FIELD_RE = re.compile(
    r"\b("
    r"type|message\s*type|msg\s*type|kind|opcode|op\s*code|operation|"
    r"function|function\s*code|command|command\s*code|tag|selector"
    r")\b",
    re.IGNORECASE,
)
_NON_SELECTOR_FIELD_NAMES = {
    "type_of_service",
    "tos",
    "traffic_class",
    "class_of_service",
}
_ZERO_MANDATE_RE = re.compile(r"\b(must|shall|should|required|always)\b.*\b(zero|0)\b|\b(zero|0)\b.*\b(must|shall|should|required|always)\b", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?0x[0-9a-fA-F]+|[-+]?\d+")
_BIT_SPAN_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\d+")
_LEADING_NUM_RE = re.compile(r"\s*([-+]?0x[0-9a-fA-F]+|[-+]?\d+)")
_VAL_REF_RE = re.compile(r"\bval\s*\(\s*([^)]*?)\s*\)", re.IGNORECASE)
_DOTTED_DECIMAL_RE = re.compile(r"^\s*\d+(?:\.\d+){2,}\s*$")
_RANGE_WORD_RE = re.compile(r"\b(?:to|through|thru)\b|\.{2,}|[–—]", re.IGNORECASE)
_EXAMPLE_FORMAT_RE = re.compile(r"\b(example|sample|illustrat(?:e|ion|ive)|comparison)\b", re.IGNORECASE)
_REQUEST_WORD_RE = re.compile(r"\b(?:request|req)\b", re.IGNORECASE)
_RESPONSE_WORD_RE = re.compile(r"\b(?:response|rsp|reply)\b", re.IGNORECASE)
_SCOPED_CONTEXT_RE = re.compile(
    r"\b(?:request|response|clients?|servers?|senders?|receivers?|transmit|receive|tx|rx|when|if|only|unless)\b",
    re.IGNORECASE,
)
_FIELD_MATCH_QUALIFIER_TOKENS = {
    "field",
    "fields",
    "value",
    "values",
    "data",
    "length",
    "len",
    "count",
    "counts",
    "code",
    "codes",
    "type",
    "types",
    "kind",
    "kinds",
    "tag",
    "tags",
    "selector",
    "selectors",
    "number",
    "list",
    "block",
    "blocks",
    "payload",
    "size",
    "sizes",
    "message",
    "msg",
    "packet",
    "pdu",
}
_FIELD_MATCH_NODE_IGNORABLE_TOKENS = {
    "gen",
    "generic",
    "tlv",
    "variant",
    "variants",
    "item",
    "seq",
    "area",
}
_FIELD_MATCH_VARIANT_SUFFIX_TOKENS = {
    "req",
    "request",
    "resp",
    "response",
    "rsp",
}
_SELECTOR_ALIAS_NAMES = {
    "code",
    "message_type",
    "msg_type",
    "option_code",
    "tag",
    "option_tag",
    "kind",
    "option_kind",
    "type",
    "option_type",
    "ie_type",
    "information_element_type",
}
_SELECTOR_ALIAS_TOKENS = {
    frozenset({"code"}),
    frozenset({"message", "type"}),
    frozenset({"msg", "type"}),
    frozenset({"option", "code"}),
    frozenset({"tag"}),
    frozenset({"option", "tag"}),
    frozenset({"kind"}),
    frozenset({"option", "kind"}),
    frozenset({"type"}),
    frozenset({"option", "type"}),
    frozenset({"ie", "type"}),
    frozenset({"information", "element", "type"}),
}
_SELECTOR_NAME_PATTERNS: Tuple[Tuple[frozenset[str], str], ...] = (
    (frozenset({"message", "type"}), "Message Type"),
    (frozenset({"msg", "type"}), "Message Type"),
    (frozenset({"sub", "function", "code"}), "Sub Function Code"),
    (frozenset({"sub", "function"}), "Sub Function Code"),
    (frozenset({"function", "code"}), "Function Code"),
    (frozenset({"command", "code"}), "Command Code"),
    (frozenset({"opcode"}), "Opcode"),
    (frozenset({"op", "code"}), "Opcode"),
    (frozenset({"option", "code"}), "Option Code"),
    (frozenset({"option", "type"}), "Option Type"),
    (frozenset({"option", "kind"}), "Option Kind"),
    (frozenset({"information", "element", "type"}), "IE Type"),
    (frozenset({"ie", "type"}), "IE Type"),
    (frozenset({"selector"}), "Selector"),
    (frozenset({"kind"}), "Kind"),
    (frozenset({"tag"}), "Tag"),
    (frozenset({"type"}), "Type"),
    (frozenset({"code"}), "Code"),
)
_GENERIC_SELECTOR_PATTERN_NAMES = {"Code", "Kind", "Selector", "Tag", "Type"}
_SELECTOR_VARIANT_SUFFIX_TOKENS = {
    "code",
    "codes",
    "field",
    "fields",
    "kind",
    "kinds",
    "message",
    "msg",
    "selector",
    "selectors",
    "tag",
    "tags",
    "type",
    "types",
    "value",
    "values",
}
_CONDITIONAL_GUARD_RE = re.compile(
    r"\b(?P<field>[A-Za-z][A-Za-z0-9 _/-]{0,48}?"
    r"(?:type|kind|code|tag|opcode|op\s*code|function|command)"
    r"[A-Za-z0-9 _/-]{0,48}?)\s*"
    r"(?:==|=|is|in)\s*"
    r"(?:\{(?P<brace>[^}]+)\}|(?P<value>[-+]?0x[0-9a-fA-F]+|[-+]?\d+))",
    re.IGNORECASE,
)
_RAW_SELECTOR_ASSIGNMENT_RE = re.compile(
    r"\b(?P<guard>type|code|kind|tag|opcode|op\s*code|function(?:\s+code)?)\s*=\s*"
    r"(?P<value>[-+]?0x[0-9a-fA-F]+|[-+]?\d+)\b",
    re.IGNORECASE,
)
_RAW_LENGTH_ASSIGNMENT_RE = re.compile(
    r"\blength\s*=\s*(?P<value>[-+]?0x[0-9a-fA-F]+|[-+]?\d+)\b",
    re.IGNORECASE,
)
_RAW_BINARY_CELL_RE = re.compile(r"^[01\s]{4,}$")
_RELATION_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_RELATION_EXPR_NAME = r"[A-Za-z][A-Za-z0-9_/-]*(?:\s+[A-Za-z][A-Za-z0-9_/-]*){0,5}"
_RELATION_ASSIGNMENT_RE = re.compile(
    rf"\b(?P<field>{_RELATION_EXPR_NAME})\s*(?:(?:==|=)|\b(?:equals?|is equal to|is)\b)\s*"
    r"(?P<value>[-+]?0x[0-9a-fA-F]+|[-+]?\d+)\b",
    re.IGNORECASE,
)
_RELATION_LENGTH_EXPR_RE = re.compile(
    rf"\b(?P<lhs>{_RELATION_EXPR_NAME})"
    r"\s*(?:(?:==|=)|\b(?:equals?|is equal to|is)\b)\s*"
    r"(?P<expr>[^.;,\n]{1,96})",
    re.IGNORECASE,
)
_RELATION_FIELD_KEYWORDS = {
    "address",
    "auth",
    "checksum",
    "code",
    "count",
    "data",
    "field",
    "final",
    "flag",
    "fragment",
    "ihl",
    "kind",
    "len",
    "length",
    "list",
    "mask",
    "message",
    "offset",
    "option",
    "padding",
    "payload",
    "poll",
    "reserved",
    "sack",
    "selector",
    "size",
    "syn",
    "tag",
    "type",
    "value",
}
_RELATION_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "been",
    "both",
    "byte",
    "bytes",
    "can",
    "clear",
    "cleared",
    "contain",
    "contains",
    "from",
    "if",
    "in",
    "is",
    "may",
    "must",
    "not",
    "of",
    "octet",
    "octets",
    "only",
    "or",
    "remaining",
    "remainder",
    "reserved",
    "set",
    "shall",
    "should",
    "the",
    "then",
    "to",
    "when",
    "zero",
}


@dataclass(frozen=True)
class ConstraintClaim:
    """A generic constraint claim extracted from one documentation section."""

    claim_id: str
    section_title: str
    field_name: str
    kind: str
    values: Tuple[int, ...] = tuple()
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    guard_field: Optional[str] = None
    guard_values: Tuple[int, ...] = tuple()
    guard_source: str = ""
    message_type: Optional[str] = None
    format_name: str = ""
    source: str = ""

    def short_expected(self) -> str:
        if self.kind == "enum":
            values = "|".join(str(v) for v in self.values)
            return f"{self.field_name} in {{{values}}}"
        if self.kind == "range":
            return f"{self.min_value} <= {self.field_name} <= {self.max_value}"
        return f"{self.field_name} {self.kind}"


@dataclass(frozen=True)
class RelationalObligation:
    """A prose-level relation between documented fields.

    These obligations are intentionally syntax-oriented and protocol-agnostic:
    they capture relations such as conditional zeroing, mutual exclusion,
    length formulas, alignment, and end-padding rules without knowing specific
    protocol names.
    """

    obligation_id: str
    section_title: str
    kind: str
    relation_text: str
    field_names: Tuple[str, ...] = tuple()
    primary_field: Optional[str] = None
    related_fields: Tuple[str, ...] = tuple()
    numeric_values: Tuple[int, ...] = tuple()
    message_type: Optional[str] = None
    format_name: str = ""
    source: str = ""

    def short_expected(self) -> str:
        fields = ", ".join(self.field_names)
        if self.kind == "conditional_zero":
            return f"condition implies zero value for {fields}"
        if self.kind == "mutual_exclusion":
            return f"mutual exclusion among {fields}"
        if self.kind == "length_relation":
            return f"length/value relation for {fields}"
        if self.kind == "mod_alignment":
            values = ", ".join(str(v) for v in self.numeric_values)
            return f"alignment/modulo relation for {fields}" + (f" using {values}" if values else "")
        if self.kind == "post_end_zero_padding":
            return f"post-end padding bytes are zero for {fields}"
        return f"{self.kind} relation for {fields}"


@dataclass(frozen=True)
class StructuralFieldRequirement:
    """A guarded on-wire field that should be represented in a variant body."""

    requirement_id: str
    section_title: str
    field_name: str
    canonical_name: str
    guard_field: str
    guard_values: Tuple[int, ...]
    guard_source: str = ""
    message_type: Optional[str] = None
    format_name: str = ""
    bit_position: str = ""
    byte_position: str = ""
    size_bits: Any = None
    direct_constraint: Optional[str] = None
    source: str = ""


@dataclass(frozen=True)
class SelectorVariantObligation:
    """A documented selector value that should be represented as a variant."""

    obligation_id: str
    section_title: str
    selector_name: str
    selector_values: Tuple[int, ...]
    variant_name: str
    canonical_variant_name: str
    message_type: Optional[str] = None
    format_name: str = ""
    source: str = ""
    expected_field_names: Tuple[str, ...] = tuple()


def _stable_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _split_identifier_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text


def _field_tokens(value: Any) -> set[str]:
    tokens = {part for part in _norm_text(_split_identifier_text(value)).split("_") if part}
    expanded = set(tokens)
    for token in tokens:
        if token == "ie":
            expanded.update({"information", "element"})
            continue
        if token == "values":
            expanded.add("value")
            continue
        if len(token) > 4 and token.endswith("ies"):
            expanded.add(token[:-3] + "y")
        elif len(token) > 4 and token.endswith("es"):
            expanded.add(token[:-2])
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            expanded.add(token[:-1])
        hex_match = re.fullmatch(r"0x([0-9a-f]+)", token)
        if hex_match:
            try:
                value_int = int(hex_match.group(1), 16)
            except ValueError:
                value_int = None
            if value_int is not None:
                expanded.add(str(value_int))
                expanded.add(f"{value_int:02d}")
                expanded.add(f"fc{value_int:02d}")
        suffix_match = re.fullmatch(r"([a-z]+)(\d+)", token)
        if suffix_match:
            digits = suffix_match.group(2)
            expanded.add(digits)
            try:
                value_int = int(digits, 10)
            except ValueError:
                value_int = None
            if value_int is not None:
                expanded.add(str(value_int))
                expanded.add(f"{value_int:02d}")
                expanded.add(f"0x{value_int:02x}")
    return expanded


def _drop_redundant_plural_tokens(tokens: set[str]) -> set[str]:
    compact = set(tokens)
    for token in list(tokens):
        if len(token) > 4 and token.endswith("ies") and token[:-3] + "y" in tokens:
            compact.discard(token)
        elif len(token) > 4 and token.endswith("es") and token[:-2] in tokens:
            compact.discard(token)
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss") and token[:-1] in tokens:
            compact.discard(token)
    return compact


def _field_acronym(value: Any) -> str:
    words = [word for word in re.findall(r"[A-Za-z0-9]+", str(value or "")) if re.search(r"[A-Za-z]", word)]
    if len(words) < 2:
        return ""
    return "".join(word[0] for word in words).lower()


def _strip_generated_variant_suffix_tokens(tokens: set[str]) -> set[str]:
    stripped = set(tokens)
    has_numeric_suffix = any(re.fullmatch(r"\d{1,3}", token) for token in tokens)
    for token in list(tokens):
        if token in _FIELD_MATCH_VARIANT_SUFFIX_TOKENS:
            stripped.discard(token)
            continue
        if re.fullmatch(r"(?:fc)?\d{1,3}", token):
            stripped.discard(token)
            continue
        if re.fullmatch(r"0x[0-9a-f]{1,4}", token):
            stripped.discard(token)
            continue
        if re.fullmatch(r"(?=.*\d)[0-9a-f]{2,4}", token):
            stripped.discard(token)
            continue
        if has_numeric_suffix and token in {"a", "b", "c", "d", "e", "f"}:
            stripped.discard(token)
    return stripped or set(tokens)


def _is_selector_alias_name(value: Any) -> bool:
    norm = _norm_text(_split_identifier_text(value))
    if norm in _SELECTOR_ALIAS_NAMES:
        return True
    return frozenset(_field_tokens(value)) in _SELECTOR_ALIAS_TOKENS


def _selector_alias_matching_enabled() -> bool:
    return os.getenv("STEP2_DOC_SELECTOR_ALIAS_MATCH", "0").strip().lower() in {"1", "true", "yes", "on"}


def _is_control_field_name(value: Any) -> bool:
    if _norm_text(_split_identifier_text(value)) in _NON_SELECTOR_FIELD_NAMES:
        return False
    text = re.sub(r"[_/-]+", " ", _split_identifier_text(value))
    if _CONTROL_FIELD_RE.search(text):
        return True
    return _selector_alias_matching_enabled() and _is_selector_alias_name(text)


def _guard_allows_structural_selector_fallback(value: Any) -> bool:
    tokens = _field_tokens(value)
    if not tokens:
        return False
    if _is_selector_alias_name(value):
        return True
    generic_selector_tokens = {
        "code",
        "command",
        "error",
        "exception",
        "function",
        "kind",
        "message",
        "msg",
        "opcode",
        "op",
        "option",
        "selector",
        "tag",
        "type",
    }
    return tokens <= generic_selector_tokens


def _canonical_tree_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\+", " plus ", text, flags=re.IGNORECASE)
    # Stage-A prose names often carry explanatory suffixes ("... of Original
    # Data Datagram").  Keep the wire-field head as the canonical tree name.
    text = re.split(r"\bof\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", text) if part]
    if not parts:
        return _norm_text(value)
    if len(parts) == 1 and parts[0].islower():
        return parts[0]
    return "_".join(part[:1].upper() + part[1:] for part in parts)


def _parse_bit_span_size(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("+"):
        return "variable"
    nums = [_parse_int(item) for item in _BIT_SPAN_NUM_RE.findall(text)]
    nums = [num for num in nums if num is not None]
    if len(nums) >= 2:
        lo, hi = int(nums[0]), int(nums[1])
        if hi >= lo:
            return hi - lo + 1
    return None


def _parse_declared_size_bits(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value or "").strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {"variable", "var", "variable length"} or "variable" in lower:
        return "variable"
    bit_match = re.search(r"\b(\d+)\s*bits?\b", lower)
    if bit_match:
        return int(bit_match.group(1))
    byte_match = re.search(r"\b(\d+)\s*(?:bytes?|octets?)\b", lower)
    if byte_match:
        return int(byte_match.group(1)) * 8
    if re.search(r"\b(?:len|length|bytes?|octets?)\b", lower):
        return "variable"
    if re.fullmatch(r"\d+", lower):
        return int(lower)
    return None


def _field_size_bits(field: Dict[str, Any]) -> Any:
    if field.get("size_bits") is not None:
        return field.get("size_bits")
    span_size = _parse_bit_span_size(field.get("bit_position"))
    if span_size is not None:
        return span_size
    return _parse_declared_size_bits(field.get("size") or field.get("length") or field.get("width"))


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "n/a", "na", "null", "any", "variable"}:
        return None
    match = _NUM_RE.search(text)
    if not match:
        return None
    token = match.group(0)
    try:
        return int(token, 16) if token.lower().startswith(("0x", "+0x", "-0x")) else int(token, 10)
    except Exception:
        return None


def _parse_numeric_values(values: Any, *, max_values: int = 32) -> Tuple[int, ...]:
    if values is None:
        return tuple()
    if not isinstance(values, list):
        values = [values]
    parsed: List[int] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        # Dotted decimal byte sequences/IP-like values are not scalar integers.
        # When a section also provides a hex scalar (e.g. a magic cookie), that
        # separate entry will be parsed below.
        if _DOTTED_DECIMAL_RE.match(text):
            continue
        # Numeric enum entries normally start with the number ("3" or
        # "3 = foo").  Avoid extracting incidental numbers from prose such as
        # "complete header + 64 bits of data".
        if not _LEADING_NUM_RE.match(text):
            continue
        # Treat "0x0000-0xFFFF" as a range, not an enum value list.
        if "-" in text and len(_NUM_RE.findall(text)) >= 2:
            continue
        # Treat "0 to 65535" / "0x0000 to 0xFFFF" as ranges, not singleton
        # enum values whose first token happens to be numeric.
        if _RANGE_WORD_RE.search(text) and len(_NUM_RE.findall(text)) >= 2:
            continue
        # Arithmetic/cardinality expressions such as "2+8*N" are not singleton
        # enum values.  They should be represented by richer formula checks, not
        # as the leading constant.
        if re.search(r"[+*/]", text) and len(_NUM_RE.findall(text)) >= 2:
            continue
        # Dynamic cardinalities such as "2 x N*" or "N* x 1 byte" are length
        # formulas, not singleton enum constraints whose first token is fixed.
        if re.search(r"\b\d+\s*[xX]\s*[A-Za-z]|\b[A-Za-z]\w*\s*[xX]\s*\d+|\b[A-Za-z]\w*\*", text):
            continue
        label_matches = list(re.finditer(r"(?<![A-Za-z0-9])((?:0x)?[0-9A-Fa-f]+)\s*=", text))
        if len(label_matches) >= 2:
            for match in label_matches:
                number = _parse_int(match.group(1))
                if number is not None:
                    parsed.append(number)
            continue
        number = _parse_int(text)
        if number is not None:
            parsed.append(number)
    unique = sorted(set(parsed))
    if not unique or len(unique) > max_values:
        return tuple()
    return tuple(unique)


def _parse_guard_from_text(text: Any) -> Tuple[Optional[str], Tuple[int, ...]]:
    """Parse control-field names and numeric guard values from Stage A sections."""

    raw = str(text or "")
    if not raw.strip():
        return None, tuple()
    for match in _CONDITIONAL_GUARD_RE.finditer(raw):
        field = re.sub(r"\b(only|when|if|for)\b", " ", match.group("field"), flags=re.IGNORECASE)
        field = re.sub(r"[^A-Za-z0-9 _/-]+", " ", field).strip(" _/-")
        if not field or not _is_control_field_name(field):
            continue
        values_raw = match.group("brace") if match.group("brace") is not None else match.group("value")
        pieces = re.split(r"[,|/]\s*|\s+or\s+", str(values_raw or ""), flags=re.IGNORECASE)
        values = _parse_numeric_values([piece.strip() for piece in pieces if piece.strip()], max_values=32)
        if values:
            return field, values
    return None, tuple()


def _field_constraints(field: Dict[str, Any]) -> Dict[str, Any]:
    constraints = field.get("constraints")
    return constraints if isinstance(constraints, dict) else {}


def _format_text(fmt: Dict[str, Any], section: Optional[Dict[str, Any]] = None) -> str:
    parts = [
        fmt.get("format_name"),
        fmt.get("description"),
        fmt.get("structure_type"),
        fmt.get("visual_representation"),
    ]
    if section is not None:
        parts.extend([section.get("title"), section.get("section_title")])
    return " ".join(str(part or "") for part in parts)


def _is_example_packet_format(fmt: Dict[str, Any], section: Optional[Dict[str, Any]] = None) -> bool:
    return bool(_EXAMPLE_FORMAT_RE.search(_format_text(fmt, section)))


def _explicit_required_state(field: Dict[str, Any]) -> Optional[bool]:
    required = _field_constraints(field).get("required")
    if isinstance(required, bool):
        return required
    if isinstance(required, str):
        normalized = required.strip().lower()
        if normalized in {"true", "yes", "required", "must", "shall"}:
            return True
        if normalized in {"false", "no", "optional", "may", "not required"}:
            return False
    return None


def _looks_like_data_model_position(field: Dict[str, Any]) -> bool:
    text = f"{field.get('bit_position') or ''} {field.get('byte_position') or ''}".lower()
    return "variable based on index" in text or "based on index" in text


def _packet_format_field_is_claim_candidate(
    fmt: Dict[str, Any],
    field: Dict[str, Any],
    section: Optional[Dict[str, Any]] = None,
) -> bool:
    if _is_example_packet_format(fmt, section):
        return False
    if _explicit_required_state(field) is False:
        return False
    if _looks_like_data_model_position(field):
        return False
    return True


def _packet_format_is_wire_layout_table(fmt: Dict[str, Any], fields: Sequence[Dict[str, Any]]) -> bool:
    """Return true for packet-format tables that describe byte/bit layout rows.

    A prose description such as "two field table showing error code and
    exception code" is still a wire-layout table when rows have positions/sizes;
    it is not an enum/selector table whose rows should become variants.
    """

    structure_type = str(fmt.get("structure_type") or "").lower()
    if "enum" in structure_type or "selector" in structure_type:
        return False
    for field in fields or []:
        if not isinstance(field, dict):
            continue
        if (
            str(field.get("bit_position") or field.get("byte_position") or "").strip()
            or _field_size_bits(field) is not None
        ):
            return True
    return False


def _infer_message_type_from_text(value: Any) -> Optional[str]:
    text = re.sub(r"[^a-z0-9]+", " ", _split_identifier_text(value).lower())
    if not text:
        return None
    has_request = bool(_REQUEST_WORD_RE.search(text))
    has_response = bool(_RESPONSE_WORD_RE.search(text))
    if has_request and not has_response:
        return "request"
    if "error response" in text or "exception response" in text:
        return "response"
    if has_response and not has_request:
        return "response"
    return None


def _infer_format_message_type(fmt: Dict[str, Any], section: Dict[str, Any]) -> Optional[str]:
    # Format names are usually more precise than section titles.  A section
    # such as "Read Coils" can contain request, response, and error response
    # packet formats sharing the same function-code guard.
    for value in (fmt.get("format_name"), fmt.get("description")):
        direct = _infer_message_type_from_text(value)
        if direct:
            return direct
    return _infer_message_type_from_text(f"{section.get('title') or ''} {section.get('section_title') or ''}")


def _node_message_type(node: Dict[str, Any]) -> Optional[str]:
    raw = _infer_message_type_from_text(node.get("message_type"))
    if raw:
        return raw
    return _infer_message_type_from_text(node.get("name"))


def _message_type_compatible(expected: Optional[str], node: Optional[Dict[str, Any]]) -> bool:
    if not expected or not node:
        return True
    actual_raw = str(node.get("message_type") or "").strip().lower()
    if actual_raw == "bidirectional":
        return True
    actual = _node_message_type(node)
    if actual is None:
        # Do not reject older trees that did not annotate message_type.
        return True
    return actual == expected


def _edge_message_type_compatible(expected: Optional[str], edge: Optional[Dict[str, Any]]) -> bool:
    if not expected or not edge:
        return True
    actual = str(edge.get("message_type") or "bidirectional").strip().lower()
    return actual in {"", "bidirectional", expected}


def _field_source(section: Dict[str, Any], field: Dict[str, Any]) -> str:
    title = section.get("title") or section.get("section_title") or section.get("number") or "section"
    source = field.get("source") or field.get("field_name") or "field"
    return f"{title} / {source}"


def _section_raw_text(section: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("content", "text", "summary", "combined_text", "raw_text", "document_text"):
        value = section.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    pages = section.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            text = page.get("text") or page.get("content")
            if isinstance(text, str) and text.strip():
                parts.append(text)
    return "\n".join(parts)


def _raw_title_candidate(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if not text[0].isupper():
        return False
    if re.match(r"^(?:This|The|A|An|Each|If|May|Must|Shall|Should)\b", text):
        return False
    if text.startswith("[Page") or text.startswith("+") or "|" in text:
        return False
    if _RAW_SELECTOR_ASSIGNMENT_RE.search(text):
        return False
    if text.endswith(":") or text.endswith("."):
        return False
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words or len(words) > 10:
        return False
    return any(re.search(r"[A-Za-z]", word) for word in words)


def _raw_title_before(lines: Sequence[str], marker_idx: int) -> str:
    for idx in range(marker_idx - 1, max(-1, marker_idx - 36), -1):
        text = lines[idx].strip()
        if _raw_title_candidate(text):
            return text
    return "Raw selector-defined format"


def _clean_raw_diagram_cell(cell: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(cell or "").replace("//", " ")).strip(" .")
    if not text or _RAW_BINARY_CELL_RE.match(text):
        return None
    if set(text) <= {"-", "+", "."}:
        return None
    if not re.search(r"[A-Za-z]", text):
        return None
    return text


def _raw_diagram_cells_near(lines: Sequence[str], marker_idx: int) -> List[str]:
    cells: List[str] = []
    seen: set[str] = set()
    for idx in range(max(0, marker_idx - 14), min(len(lines), marker_idx + 4)):
        line = lines[idx]
        if "|" not in line or line.lstrip().startswith("+"):
            continue
        for piece in line.split("|")[1:-1]:
            cell = _clean_raw_diagram_cell(piece)
            if not cell:
                continue
            key = _norm_text(cell)
            if key in seen:
                continue
            seen.add(key)
            cells.append(cell)
    return cells


def _raw_field_name(cell: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", cell)
    if not words:
        return "Documented Field"
    return " ".join(word.upper() if word.isupper() and len(word) <= 4 else word.capitalize() for word in words)


def _raw_option_packet_formats(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Synthesize packet-format rows from raw ASCII diagrams with selector values.

    Stage-A sometimes keeps RFC option diagrams only in raw page text.  This
    parser is deliberately generic: a diagram plus ``Type=...`` / ``Code=...`` /
    ``Kind=...`` is treated as a selector-guarded format, and any named diagram
    cells become required body fields.
    """

    raw_text = _section_raw_text(section)
    if "|" not in raw_text or not _RAW_SELECTOR_ASSIGNMENT_RE.search(raw_text):
        return []

    lines = raw_text.splitlines()
    formats: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, str]] = set()
    for idx, line in enumerate(lines):
        for match in _RAW_SELECTOR_ASSIGNMENT_RE.finditer(line):
            guard_name = re.sub(r"\s+", " ", match.group("guard")).strip().title()
            value = _parse_int(match.group("value"))
            if value is None:
                continue
            title = _raw_title_before(lines, idx)
            key = (_norm_text(guard_name), value, _norm_text(title))
            if key in seen:
                continue
            seen.add(key)

            cells = _raw_diagram_cells_near(lines, idx)
            fields: List[Dict[str, Any]] = [
                {
                    "field_name": guard_name,
                    "bit_position": "0-7",
                    "size_bits": 8,
                    "constraints": {"valid_values": [str(value)], "required": "true"},
                    "source": f"raw diagram selector {guard_name}={value}",
                }
            ]

            start_bit = 8
            body_fields = 0
            length_assignment = _RAW_LENGTH_ASSIGNMENT_RE.search(line)
            if length_assignment and not any("length" in cell.lower() for cell in cells):
                length_value = _parse_int(length_assignment.group("value"))
                constraints: Dict[str, Any] = {"required": "true"}
                if length_value is not None:
                    constraints.update(
                        {
                            "valid_values": [str(length_value)],
                            "min_value": str(length_value),
                            "max_value": str(length_value),
                        }
                    )
                fields.append(
                    {
                        "field_name": "Length",
                        "bit_position": f"{start_bit}-{start_bit + 7}",
                        "size_bits": 8,
                        "constraints": constraints,
                        "source": "raw selector line length assignment",
                    }
                )
                body_fields += 1
                start_bit += 8
            for cell in cells:
                if _norm_text(cell) == _norm_text(guard_name):
                    continue
                field_name = _raw_field_name(cell)
                constraints: Dict[str, Any] = {"required": "true"}
                lower = cell.lower()
                bit_position = f"{start_bit}-{start_bit + 7}"
                size_bits: Any = 8
                if any(token in lower for token in ("data", "address", "timestamp", "payload", "route")):
                    bit_position = f"{start_bit}+"
                    size_bits = "variable"
                if "length" in lower and length_assignment:
                    length_value = _parse_int(length_assignment.group("value"))
                    if length_value is not None:
                        constraints.update(
                            {
                                "valid_values": [str(length_value)],
                                "min_value": str(length_value),
                                "max_value": str(length_value),
                            }
                        )
                fields.append(
                    {
                        "field_name": field_name,
                        "bit_position": bit_position,
                        "size_bits": size_bits,
                        "constraints": constraints,
                        "source": f"raw diagram cell {cell}",
                    }
                )
                body_fields += 1
                if isinstance(size_bits, int):
                    start_bit += size_bits
                else:
                    start_bit += 8

            if body_fields == 0:
                fields.append(
                    {
                        "field_name": "Option Payload",
                        "bit_position": "8-7",
                        "size_bits": "0",
                        "description": "Selector-defined format has no documented body fields.",
                        "constraints": {"required": "true"},
                        "source": "raw single-octet selector diagram",
                    }
                )

            formats.append(
                {
                    "format_name": title,
                    "description": f"Synthesized from raw document diagram near {guard_name}={value}",
                    "fields": fields,
                    "_synthetic_raw_option": True,
                }
            )
    return formats


def _collect_fields(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    def add(field: Any, *, from_field_definition: bool = False) -> None:
        if not isinstance(field, dict):
            return
        name = str(field.get("field_name") or field.get("name") or "").strip()
        if not name:
            return
        if from_field_definition:
            parent_field = str(field.get("parent_field") or "").strip().lower()
            # Stage-A often represents enum table rows as field_definitions
            # whose parent_field points at the real wire field.  Those rows are
            # labels for one selector value, not additional packet fields.
            if parent_field and parent_field not in {"none", "null", "n/a", "na"}:
                return
        constraints = _field_constraints(field)
        key = (
            _norm_text(name),
            str(field.get("bit_position") or field.get("byte_position") or ""),
            json.dumps(constraints, ensure_ascii=True, sort_keys=True),
        )
        if key in seen:
            return
        seen.add(key)
        fields.append(field)

    for fmt in section.get("packet_formats") or []:
        if isinstance(fmt, dict):
            for field in fmt.get("fields") or []:
                add(field)
    for field in section.get("field_definitions") or []:
        add(field, from_field_definition=True)
    return fields


def _collect_packet_format_fields(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for fmt in section.get("packet_formats") or []:
        if not isinstance(fmt, dict):
            continue
        for field in fmt.get("fields") or []:
            if not isinstance(field, dict):
                continue
            name = str(field.get("field_name") or field.get("name") or "").strip()
            if not name:
                continue
            key = (
                _norm_text(name),
                str(field.get("bit_position") or ""),
                str(field.get("byte_position") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            fields.append(field)
    return fields


def _iter_packet_format_field_groups(section: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    for fmt in section.get("packet_formats") or []:
        if not isinstance(fmt, dict):
            continue
        fields = [field for field in (fmt.get("fields") or []) if isinstance(field, dict)]
        if fields:
            yield fmt, fields
    for fmt in _raw_option_packet_formats(section):
        fields = [field for field in (fmt.get("fields") or []) if isinstance(field, dict)]
        if fields:
            yield fmt, fields


def _field_order_key(field: Dict[str, Any]) -> int:
    for key in ("bit_position", "byte_position"):
        number = _parse_int(field.get(key))
        if number is not None:
            return number
    return 10**9


def _format_group_key(fmt: Dict[str, Any], fields: Sequence[Dict[str, Any]]) -> Tuple[Any, ...]:
    """Stable key for structured and synthesized packet-format rows."""

    field_signature = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("field_name") or field.get("name") or "").strip()
        constraints = _field_constraints(field)
        field_signature.append(
            (
                _norm_text(name),
                str(field.get("bit_position") or ""),
                str(field.get("byte_position") or ""),
                str(field.get("size_bits") or ""),
                json.dumps(constraints, ensure_ascii=True, sort_keys=True),
            )
        )
    return (
        _norm_text(fmt.get("format_name") or fmt.get("description") or ""),
        tuple(field_signature),
    )


def _singleton_field_values(field: Dict[str, Any]) -> Tuple[int, ...]:
    constraints = _field_constraints(field)
    values = _parse_numeric_values(constraints.get("valid_values"))
    min_v = _parse_int(constraints.get("min_value"))
    max_v = _parse_int(constraints.get("max_value"))
    if not values and min_v is not None and max_v is not None and min_v == max_v:
        values = (min_v,)
    if len(values) == 1:
        return tuple(values)
    return tuple()


def _field_slot_signature(field: Dict[str, Any]) -> Tuple[str, str, Any]:
    return (
        str(field.get("bit_position") or "").strip().lower(),
        str(field.get("byte_position") or "").strip().lower(),
        _field_size_bits(field),
    )


def _is_length_field_name(value: Any) -> bool:
    tokens = _field_tokens(value)
    norm = _norm_text(_split_identifier_text(value))
    return "length" in tokens or "len" in tokens or norm.endswith("_len")


def _is_payloadish_field_name(value: Any) -> bool:
    tokens = _field_tokens(value)
    norm = _norm_text(_split_identifier_text(value))
    payload_tokens = {
        "address",
        "addresses",
        "body",
        "content",
        "data",
        "payload",
        "route",
        "routes",
        "timestamp",
        "timestamps",
        "value",
        "values",
    }
    payload_norms = {
        "option_payload",
        "option_data",
        "tlv_data",
        "tlv_value",
        "route_data",
    }
    return bool(tokens & payload_tokens) or norm in payload_norms


def _is_zero_sized_requirement(requirement: StructuralFieldRequirement) -> bool:
    size = str(requirement.size_bits or "").strip().lower()
    return size == "0" or (
        requirement.size_bits == 0 and _is_payloadish_field_name(requirement.field_name)
    )


def _requirement_tlv_group_key(requirement: StructuralFieldRequirement) -> Tuple[str, str, Optional[str]]:
    return (
        requirement.section_title,
        _norm_text(requirement.guard_field),
        requirement.message_type,
    )


def _tlv_profiles_for_requirements(
    requirements: Sequence[StructuralFieldRequirement],
) -> Dict[Tuple[str, str, Optional[str]], Dict[str, Any]]:
    """Infer repeated selector tables that are better repaired as one TLV shape.

    The detector is protocol-agnostic: it only uses document table structure.
    A profile needs repeated selector values, at least two rows with a length-
    like field, and some payload/value-like field evidence.
    """

    grouped: Dict[Tuple[str, str, Optional[str]], List[StructuralFieldRequirement]] = {}
    for requirement in requirements:
        if requirement.guard_source not in {"table", "name"} or not requirement.guard_values:
            continue
        grouped.setdefault(_requirement_tlv_group_key(requirement), []).append(requirement)

    profiles: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]] = {}
    for key, group in grouped.items():
        values = sorted({int(value) for req in group for value in req.guard_values})
        if len(values) < 3:
            continue

        by_value: Dict[int, List[StructuralFieldRequirement]] = {}
        for req in group:
            for value in req.guard_values:
                by_value.setdefault(int(value), []).append(req)

        length_values = sorted(
            value
            for value, rows in by_value.items()
            if any(_is_length_field_name(row.field_name) for row in rows)
        )
        payload_values = sorted(
            value
            for value, rows in by_value.items()
            if any(_is_payloadish_field_name(row.field_name) for row in rows)
        )
        zero_body_values = sorted(
            value
            for value, rows in by_value.items()
            if rows and all(_is_zero_sized_requirement(row) for row in rows)
        )

        if len(length_values) < 2 or not payload_values:
            continue

        representative = next(
            (
                req
                for req in group
                if any(int(value) in length_values for value in req.guard_values)
                and _is_length_field_name(req.field_name)
            ),
            group[0],
        )
        length_names = sorted({_canonical_tree_name(req.field_name) for req in group if _is_length_field_name(req.field_name)})
        payload_names = sorted({_canonical_tree_name(req.field_name) for req in group if _is_payloadish_field_name(req.field_name)})
        profiles[key] = {
            "key": key,
            "representative": representative,
            "section": representative.section_title,
            "guard_field": representative.guard_field,
            "message_type": representative.message_type,
            "guard_values": values,
            "length_values": length_values,
            "payload_values": payload_values,
            "zero_body_values": zero_body_values,
            "length_field_names": length_names or ["Option_Length"],
            "payload_field_names": payload_names or ["Option_Data"],
            "requirements": group,
        }
    zero_values_by_guard: Dict[Tuple[str, Optional[str]], set[int]] = {}
    for requirement in requirements:
        if not requirement.guard_values or not _is_zero_sized_requirement(requirement):
            continue
        guard_key = (_norm_text(requirement.guard_field), requirement.message_type)
        zero_values_by_guard.setdefault(guard_key, set()).update(int(value) for value in requirement.guard_values)

    for profile in profiles.values():
        guard_key = (_norm_text(profile.get("guard_field")), profile.get("message_type"))
        related_zero_values = zero_values_by_guard.get(guard_key, set())
        if related_zero_values:
            profile["zero_body_values"] = sorted(set(profile.get("zero_body_values") or []) | related_zero_values)
    return profiles


def _requirement_tlv_profile(
    requirement: StructuralFieldRequirement,
    profiles: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    profile = profiles.get(_requirement_tlv_group_key(requirement))
    if not profile:
        return None
    profile_ids = {req.requirement_id for req in profile.get("requirements") or []}
    return profile if requirement.requirement_id in profile_ids else None


def _first_singleton_field(fields: Sequence[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], Tuple[int, ...]]]:
    ordered = sorted(
        [
            field
            for field in fields
            if isinstance(field, dict)
            and str(field.get("field_name") or field.get("name") or "").strip()
            and _packet_format_field_is_claim_candidate({}, field)
        ],
        key=_field_order_key,
    )
    if not ordered:
        return None
    first = ordered[0]
    values = _singleton_field_values(first)
    if values:
        return first, values
    return None


def _table_guard_candidates(section: Dict[str, Any]) -> Dict[Tuple[Any, ...], Tuple[str, Tuple[int, ...]]]:
    """Infer guarded rows from repeated packet-format tables.

    If several packet formats in one section start with a singleton-valued
    field at the same wire slot, that leading field is treated as the row
    selector.  This uses document structure, not protocol-specific field names.
    """

    rows: List[Tuple[Tuple[Any, ...], Dict[str, Any], Tuple[int, ...], Tuple[str, str, Any]]] = []
    for fmt, fields in _iter_packet_format_field_groups(section):
        first = _first_singleton_field(fields)
        if first is None:
            continue
        field, values = first
        rows.append((_format_group_key(fmt, fields), field, values, _field_slot_signature(field)))

    slot_counts: Dict[Tuple[str, str, Any], int] = {}
    for _fmt_id, _field, _values, slot in rows:
        slot_counts[slot] = slot_counts.get(slot, 0) + 1

    out: Dict[Tuple[Any, ...], Tuple[str, Tuple[int, ...]]] = {}
    for fmt_key, field, values, slot in rows:
        if slot_counts.get(slot, 0) < 2:
            continue
        name = str(field.get("field_name") or field.get("name") or "").strip()
        if name:
            out[fmt_key] = (name, values)
    return out


def _select_guard_field(fields: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Tuple[int, ...]]:
    candidates: List[Tuple[int, str, Tuple[int, ...]]] = []
    for field in fields:
        name = str(field.get("field_name") or field.get("name") or "").strip()
        if not name or _norm_text(name) in _NON_SELECTOR_FIELD_NAMES or not _is_control_field_name(name):
            continue
        values = _parse_numeric_values(_field_constraints(field).get("valid_values"))
        if not values:
            values = _singleton_field_values(field)
        if values and len(values) <= 8:
            candidates.append((_field_order_key(field), name, values))
    if not candidates:
        return None, tuple()
    _, name, values = sorted(candidates, key=lambda item: item[0])[0]
    return name, values


def _selector_values_from_field(field: Dict[str, Any]) -> Tuple[int, ...]:
    constraints = _field_constraints(field)
    value_sources = [
        constraints.get("valid_values"),
        constraints.get("enum_values"),
        constraints.get("value"),
        constraints.get("type"),
        constraints.get("code"),
        constraints.get("kind"),
        constraints.get("tag"),
        constraints.get("opcode"),
        constraints.get("op_code"),
        constraints.get("function_code"),
        constraints.get("command_code"),
        constraints.get("message_type"),
        constraints.get("option_type"),
        constraints.get("option_code"),
        constraints.get("option_kind"),
        constraints.get("ie_type"),
        constraints.get("information_element_type"),
    ]
    for source in value_sources:
        values = _parse_numeric_values(source, max_values=16)
        if values:
            return values
    return _singleton_field_values(field)


def _selector_name_from_context(
    fmt: Dict[str, Any],
    field: Dict[str, Any],
    section: Dict[str, Any],
) -> Optional[str]:
    constraints = _field_constraints(field)
    explicit = str(constraints.get("selector") or "").strip()
    explicit_tokens = _field_tokens(explicit)
    if explicit:
        if {"information", "element", "type"} <= explicit_tokens:
            return "IE Type"
        return explicit

    name = str(field.get("field_name") or field.get("name") or "").strip()
    name_tokens = _field_tokens(name)
    if name_tokens == {"type"}:
        return "Type"
    for pattern_tokens, selector_name in _SELECTOR_NAME_PATTERNS:
        if pattern_tokens <= name_tokens and selector_name not in _GENERIC_SELECTOR_PATTERN_NAMES:
            return selector_name
    if "code" in name_tokens and name_tokens <= {"code", "error", "exception"}:
        return "Code"
    if _is_control_field_name(name) and name.strip().title() not in _GENERIC_SELECTOR_PATTERN_NAMES:
        return name

    text_parts = [
        fmt.get("format_name"),
        fmt.get("description"),
        fmt.get("structure_type"),
        section.get("title"),
        section.get("section_title"),
    ]
    text = " ".join(str(part or "") for part in text_parts)
    tokens = _field_tokens(text)
    for pattern_tokens, selector_name in _SELECTOR_NAME_PATTERNS:
        if pattern_tokens <= tokens:
            return selector_name
    return None


def _selector_variant_label(selector_name: str, field_name: str, fmt: Dict[str, Any]) -> str:
    raw = str(field_name or "").strip() or str(fmt.get("format_name") or "").strip()
    selector_tokens = _field_tokens(selector_name)
    parts = [
        part
        for part in re.split(r"[^A-Za-z0-9]+", _split_identifier_text(raw))
        if part and part.lower() not in selector_tokens and part.lower() not in _SELECTOR_VARIANT_SUFFIX_TOKENS
    ]
    if parts:
        return " ".join(parts)
    fmt_name = str(fmt.get("format_name") or fmt.get("description") or "").strip()
    return fmt_name or raw


def _selector_variant_label_degenerate(selector_name: str, variant_name: str, format_name: str) -> bool:
    if str(format_name or "").strip():
        return False
    selector_tokens = _field_tokens(selector_name)
    variant_tokens = _field_tokens(variant_name)
    if not selector_tokens or not variant_tokens:
        return False
    return variant_tokens <= selector_tokens or _norm_text(variant_name) == _norm_text(selector_name)


def _packet_format_is_selector_table(fmt: Dict[str, Any], section: Dict[str, Any]) -> bool:
    structure_type = str(fmt.get("structure_type") or "").lower()
    if "enum" in structure_type or "selector" in structure_type:
        return True
    text = _format_text(fmt, section)
    tokens = _field_tokens(text)
    if "table" in tokens and any(pattern <= tokens for pattern, _name in _SELECTOR_NAME_PATTERNS):
        return True
    return False


def _first_wire_layout_control_field(
    fmt: Dict[str, Any],
    fields: Sequence[Dict[str, Any]],
    section: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not _packet_format_is_wire_layout_table(fmt, fields):
        return None
    text_tokens = _field_tokens(_format_text(fmt, section))
    candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for index, field in enumerate(fields or []):
        if not isinstance(field, dict) or not _packet_format_field_is_claim_candidate(fmt, field, section):
            continue
        field_name = str(field.get("field_name") or field.get("name") or "").strip()
        if not field_name or not _is_control_field_name(field_name):
            continue
        field_tokens = _field_tokens(field_name)
        packet_selector_tokens = (
            {"function", "code"} <= field_tokens
            or {"command", "code"} <= field_tokens
            or {"op", "code"} <= field_tokens
            or "opcode" in field_tokens
            or {"message", "type"} <= field_tokens
            or {"msg", "type"} <= field_tokens
        )
        if not packet_selector_tokens:
            continue
        if {"error", "response"} <= text_tokens or {"exception", "response"} <= text_tokens:
            if "error" in field_tokens or "exception" in field_tokens:
                continue
        if not _selector_values_from_field(field):
            continue
        candidates.append((_field_order_key(field), index, field))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]


def extract_selector_variant_obligations(sections: Sequence[Dict[str, Any]]) -> List[SelectorVariantObligation]:
    """Extract selector-value variant obligations from documented enum tables.

    This is intentionally table-driven: it only emits obligations when Stage A
    already exposed a singleton numeric selector value.  It also treats the first
    singleton control field in a concrete wire-layout table as a selector route
    obligation.  The validator is not inventing protocol knowledge; it turns
    explicit source rows into repairable coverage targets.
    """

    obligations: List[SelectorVariantObligation] = []
    seen: set[Tuple[Any, ...]] = set()
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or section.get("section_title") or section.get("number") or "section")
        for fmt, fields in _iter_packet_format_field_groups(section):
            if _is_example_packet_format(fmt, section):
                continue
            is_selector_table = _packet_format_is_selector_table(fmt, section)
            if is_selector_table and _packet_format_is_wire_layout_table(fmt, fields):
                is_selector_table = False
            wire_layout_control_field = None
            if not is_selector_table:
                wire_layout_control_field = _first_wire_layout_control_field(fmt, fields, section)
            format_name = str(fmt.get("format_name") or fmt.get("description") or "")
            message_type = _infer_format_message_type(fmt, section)
            for field in fields:
                if not isinstance(field, dict) or not _packet_format_field_is_claim_candidate(fmt, field, section):
                    continue
                values = _selector_values_from_field(field)
                if not values:
                    continue
                field_name = str(field.get("field_name") or field.get("name") or "").strip()
                if not field_name:
                    continue
                selector_name = _selector_name_from_context(fmt, field, section)
                if not selector_name:
                    continue
                constraints = _field_constraints(field)
                is_wire_layout_selector = wire_layout_control_field is field
                if not (
                    is_selector_table
                    or constraints.get("selector")
                    or constraints.get("ie_type") is not None
                    or is_wire_layout_selector
                ):
                    continue
                variant_name = _selector_variant_label(selector_name, field_name, fmt)
                if is_wire_layout_selector and _selector_variant_label_degenerate(selector_name, variant_name, format_name):
                    continue
                canonical = _canonical_tree_name(variant_name)
                if not canonical:
                    continue
                key = (
                    section_title,
                    format_name,
                    message_type,
                    _norm_text(selector_name),
                    tuple(values),
                    _norm_text(canonical),
                )
                if key in seen:
                    continue
                seen.add(key)
                obligation_id = "selector_variant_" + _stable_id(
                    {
                        "section": section_title,
                        "format": format_name,
                        "selector": selector_name,
                        "values": list(values),
                        "variant": canonical,
                    }
                )
                obligations.append(
                    SelectorVariantObligation(
                        obligation_id=obligation_id,
                        section_title=section_title,
                        selector_name=selector_name,
                        selector_values=tuple(int(value) for value in values),
                        variant_name=variant_name,
                        canonical_variant_name=canonical,
                        message_type=message_type,
                        format_name=format_name,
                        source=f"{section_title}: {format_name or field_name}",
                        expected_field_names=tuple(
                            str(spec.get("field_name") or spec.get("canonical_name") or "")
                            for spec in _format_expected_field_specs(fmt, section)
                            if str(spec.get("field_name") or spec.get("canonical_name") or "").strip()
                        ),
                    )
                )
    return obligations


def _structural_direct_constraint(field: Dict[str, Any]) -> Optional[str]:
    constraints = _field_constraints(field)
    values = _parse_numeric_values(constraints.get("valid_values"))
    min_value = _parse_int(constraints.get("min_value"))
    max_value = _parse_int(constraints.get("max_value"))
    description = f"{field.get('description') or ''} {constraints.get('conditional') or ''}"

    if not values and min_value is not None and max_value is not None and min_value == max_value:
        values = (min_value,)
    if not values and str(constraints.get("default_value") or "").strip() in {"0", "0x0", "0x00"} and _ZERO_MANDATE_RE.search(description):
        values = (0,)
    if values:
        return "enum: " + "|".join(str(value) for value in values)
    if min_value is not None and max_value is not None and min_value <= max_value:
        return f"range: {min_value}..{max_value}"
    return None


def _field_required_by_format(field: Dict[str, Any]) -> bool:
    required_state = _explicit_required_state(field)
    if required_state is not None:
        return required_state
    # Packet-format rows are normally on-wire fields even when the extractor
    # omitted an explicit required flag.  A bit/byte position is enough evidence
    # to make this a structural coverage requirement.
    return bool(str(field.get("bit_position") or field.get("byte_position") or "").strip()) or _field_size_bits(field) is not None


def extract_structural_field_requirements(sections: Sequence[Dict[str, Any]]) -> List[StructuralFieldRequirement]:
    """Extract guarded body fields that should appear below selected variants."""

    requirements: List[StructuralFieldRequirement] = []
    seen: set[Tuple[Any, ...]] = set()
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or section.get("section_title") or section.get("number") or "section")
        table_guards = _table_guard_candidates(section)
        for fmt, fields in _iter_packet_format_field_groups(section):
            if _is_example_packet_format(fmt, section):
                continue
            if _packet_format_is_selector_table(fmt, section):
                continue
            table_guard = table_guards.get(_format_group_key(fmt, fields))
            if table_guard is not None:
                guard_name, guard_values = table_guard
                guard_source = "table"
            else:
                guard_name, guard_values = _select_guard_field(fields)
                guard_source = "name" if guard_name else ""
            if not guard_name or not guard_values:
                continue
            format_name = str(fmt.get("format_name") or fmt.get("description") or "")
            message_type = _infer_format_message_type(fmt, section)
            for field in fields:
                field_name = str(field.get("field_name") or field.get("name") or "").strip()
                if not field_name or _norm_text(field_name) == _norm_text(guard_name):
                    continue
                if not _packet_format_field_is_claim_candidate(fmt, field, section) or not _field_required_by_format(field):
                    continue
                canonical = _canonical_tree_name(field_name)
                if not canonical:
                    continue
                key = (
                    section_title,
                    format_name,
                    message_type,
                    _norm_text(field_name),
                    tuple(guard_values),
                    str(field.get("bit_position") or ""),
                    str(field.get("byte_position") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                payload = {
                    "section": section_title,
                    "format": format_name,
                    "field": field_name,
                    "guard": guard_name,
                    "guard_values": tuple(guard_values),
                    "bit_position": field.get("bit_position"),
                    "byte_position": field.get("byte_position"),
                }
                requirements.append(
                    StructuralFieldRequirement(
                        requirement_id="doc_structure_" + _stable_id(payload),
                        section_title=section_title,
                        field_name=field_name,
                        canonical_name=canonical,
                        guard_field=guard_name,
                        guard_values=tuple(guard_values),
                        guard_source=guard_source,
                        message_type=message_type,
                        format_name=format_name,
                        bit_position=str(field.get("bit_position") or ""),
                        byte_position=str(field.get("byte_position") or ""),
                        size_bits=_field_size_bits(field),
                        direct_constraint=_structural_direct_constraint(field),
                        source=f"{_field_source(section, field)} / {format_name}" if format_name else _field_source(section, field),
                    )
                )
    return requirements


def _iter_constraint_claim_fields(section: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Optional[str], Tuple[int, ...], str, Optional[str], str]]:
    table_guards = _table_guard_candidates(section)
    for fmt, fields in _iter_packet_format_field_groups(section):
        if _is_example_packet_format(fmt, section):
            continue
        if _packet_format_is_selector_table(fmt, section):
            # Enum/selector tables define dispatch variants.  Treating each row
            # as a normal enum constraint makes the guard inference pick an
            # arbitrary row label when the row name happens to contain words like
            # "Function" or "Type".  Selector coverage is handled by
            # extract_selector_variant_obligations/find_doc_structure_alignment_issues.
            continue
        table_guard = table_guards.get(_format_group_key(fmt, fields))
        if table_guard is not None:
            guard_name, guard_values = table_guard
            guard_source = "table"
        else:
            guard_name, guard_values = _select_guard_field(fields)
            guard_source = "name" if guard_name else ""
        format_name = str(fmt.get("format_name") or fmt.get("description") or "")
        message_type = _infer_format_message_type(fmt, section)
        for field in fields:
            if not _packet_format_field_is_claim_candidate(fmt, field, section):
                continue
            yield field, guard_name, guard_values, format_name, message_type, guard_source
    for field in section.get("field_definitions") or []:
        if not isinstance(field, dict):
            continue
        name = str(field.get("field_name") or field.get("name") or "").strip()
        if not name:
            continue
        parent_field = str(field.get("parent_field") or "").strip().lower()
        if parent_field and parent_field not in {"none", "null", "n/a", "na"}:
            continue
        constraints = _field_constraints(field)
        guard_name, guard_values = _parse_guard_from_text(
            f"{constraints.get('conditional') or ''} {field.get('description') or ''}"
        )
        if not guard_name or not guard_values:
            # Standalone field_definitions are often glossary/table rows rather
            # than concrete on-wire claims.  Keep doc-coverage repair high
            # precision by only using definitions that carry explicit guarded
            # context.
            continue
        yield field, guard_name, guard_values, "field_definitions", None, "name"


def extract_constraint_claims(sections: Sequence[Dict[str, Any]]) -> List[ConstraintClaim]:
    """Extract generic enum/range claims from Stage-A structured and raw sections."""

    claims: List[ConstraintClaim] = []
    seen: set[Tuple[Any, ...]] = set()
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or section.get("section_title") or section.get("number") or "section")
        for field, guard_name, guard_values, format_name, message_type, guard_source in _iter_constraint_claim_fields(section):
            field_name = str(field.get("field_name") or field.get("name") or "").strip()
            if not field_name:
                continue
            constraints = _field_constraints(field)
            values = _parse_numeric_values(constraints.get("valid_values"))
            min_value = _parse_int(constraints.get("min_value"))
            max_value = _parse_int(constraints.get("max_value"))
            description = f"{field.get('description') or ''} {constraints.get('conditional') or ''}"

            if not values and min_value is not None and max_value is not None and min_value == max_value:
                values = (min_value,)
            if not values and str(constraints.get("default_value") or "").strip() in {"0", "0x0", "0x00"} and _ZERO_MANDATE_RE.search(description):
                values = (0,)

            kind: Optional[str] = None
            if values:
                kind = "enum"
            elif min_value is not None and max_value is not None and min_value <= max_value:
                kind = "range"

            if not kind:
                continue

            # The first fixed control field in a section is context, not a
            # requirement that every protocol packet globally equals that value.
            is_guard_field = guard_name is not None and _norm_text(field_name) == _norm_text(guard_name)
            if is_guard_field:
                continue
            claim_guard_name = None if is_guard_field else guard_name
            claim_guard_values = tuple() if is_guard_field else guard_values
            key = (
                section_title,
                format_name,
                _norm_text(field_name),
                kind,
                values,
                min_value,
                max_value,
                _norm_text(claim_guard_name),
                claim_guard_values,
                guard_source,
                message_type,
            )
            if key in seen:
                continue
            seen.add(key)
            claim_payload = {
                "section": section_title,
                "field": field_name,
                "kind": kind,
                "values": values,
                "min": min_value,
                "max": max_value,
                "guard": claim_guard_name,
                "guard_values": claim_guard_values,
                "guard_source": guard_source if claim_guard_name else "",
                "message_type": message_type,
            }
            claims.append(
                ConstraintClaim(
                    claim_id="doc_constraint_" + _stable_id(claim_payload),
                    section_title=section_title,
                    field_name=field_name,
                    kind=kind,
                    values=tuple(values),
                    min_value=min_value,
                    max_value=max_value,
                    guard_field=claim_guard_name,
                    guard_values=tuple(claim_guard_values),
                    guard_source=guard_source if claim_guard_name else "",
                    message_type=message_type,
                    format_name=format_name,
                    source=f"{_field_source(section, field)} / {format_name}" if format_name else _field_source(section, field),
                )
            )
    return claims


def _tree_nodes(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = tree.get("protocol_tree") if isinstance(tree.get("protocol_tree"), dict) else tree
    return list(payload.get("nodes") or []) if isinstance(payload, dict) else []


def _tree_edges(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = tree.get("protocol_tree") if isinstance(tree.get("protocol_tree"), dict) else tree
    return list(payload.get("edges") or []) if isinstance(payload, dict) else []


def _node_label(node: Dict[str, Any]) -> str:
    return f"{node.get('name')}(ID:{node.get('node_id')})"


_SCALAR_VAL_NODE_TYPES = {"field", "selector", "type", "length", "checksum"}
_STRUCTURAL_VAL_NODE_TYPES = {
    "array",
    "body",
    "bytes",
    "container",
    "header",
    "list",
    "message",
    "payload",
    "record",
    "sequence",
    "struct",
    "tlv_seq",
    "variant",
}


def _node_type_lower(node: Dict[str, Any]) -> str:
    return str(node.get("node_type") or node.get("type") or "").strip().lower()


def _node_looks_like_structural_payload_value(node: Dict[str, Any]) -> bool:
    node_type = _node_type_lower(node)
    if node_type in _STRUCTURAL_VAL_NODE_TYPES:
        return True
    if node_type != "field" or not _is_payloadish_field_name(node.get("name")):
        return False
    if node.get("children_ids"):
        return True
    data_text = " ".join(
        str(node.get(key) or "")
        for key in ("data_type", "encoding", "description")
    ).lower()
    if re.search(r"\b(bytes?|octets?|opaque|payload|binary|blob|raw|string)\b", data_text):
        return True
    size_text = str(node.get("size_bits") or "").strip().lower()
    if not size_text or size_text in {"var", "variable", "dynamic", "unknown", "n/a", "none"}:
        return True
    return "val(" in size_text


def _node_is_scalar_val_reference_target(node: Dict[str, Any]) -> bool:
    node_type = _node_type_lower(node)
    if node_type not in _SCALAR_VAL_NODE_TYPES:
        return False
    return not _node_looks_like_structural_payload_value(node)


def _match_field_nodes(nodes: Sequence[Dict[str, Any]], field_name: str) -> List[Dict[str, Any]]:
    wanted = _field_tokens(field_name)
    if not wanted:
        return []
    wanted_norm = _norm_text(_split_identifier_text(field_name))
    wanted_acronym = _field_acronym(field_name)
    exact = [n for n in nodes if _norm_text(_split_identifier_text(n.get("name"))) == wanted_norm]
    if exact:
        return exact
    if wanted_acronym:
        acronym_matches = [
            n
            for n in nodes
            if _norm_text(n.get("name")) == wanted_acronym
            or wanted_acronym in _field_tokens(n.get("name"))
        ]
        if acronym_matches:
            return acronym_matches
    if _selector_alias_matching_enabled() and _is_selector_alias_name(field_name):
        selector_alias_matches = [
            n
            for n in nodes
            if _is_selector_alias_name(n.get("name"))
            and _norm_text(n.get("name")) not in _NON_SELECTOR_FIELD_NAMES
        ]
        if selector_alias_matches:
            return selector_alias_matches
    matches: List[Dict[str, Any]] = []
    for node in nodes:
        node_norm = _norm_text(node.get("name"))
        if wanted_norm in _NON_SELECTOR_FIELD_NAMES and node_norm in {"type", "message_type", "msg_type"}:
            continue
        tokens = _field_tokens(node.get("name"))
        if wanted and wanted <= tokens:
            matches.append(node)
            continue
        if wanted and tokens and tokens <= wanted:
            missing = wanted - tokens
            if missing and missing <= _FIELD_MATCH_QUALIFIER_TOKENS:
                matches.append(node)
                continue
        compact_tokens = _drop_redundant_plural_tokens(tokens)
        if wanted and compact_tokens and compact_tokens <= wanted:
            missing = wanted - compact_tokens
            if missing and missing <= _FIELD_MATCH_QUALIFIER_TOKENS:
                matches.append(node)
                continue
        comparable_tokens = tokens - _FIELD_MATCH_NODE_IGNORABLE_TOKENS
        if wanted and comparable_tokens and comparable_tokens <= wanted:
            missing = wanted - comparable_tokens
            if missing and missing <= _FIELD_MATCH_QUALIFIER_TOKENS and len(comparable_tokens & wanted) >= 2:
                matches.append(node)
    return matches


def _match_guard_nodes(nodes: Sequence[Dict[str, Any]], field_name: Optional[str]) -> List[Dict[str, Any]]:
    if not field_name:
        return []
    matches = list(_match_field_nodes(nodes, field_name))
    if not _selector_alias_matching_enabled() or not _is_selector_alias_name(field_name):
        return matches
    seen = {str(node.get("node_id")) for node in matches}
    for node in nodes:
        node_id = str(node.get("node_id"))
        if node_id in seen:
            continue
        if _is_selector_alias_name(node.get("name")) and _norm_text(node.get("name")) not in _NON_SELECTOR_FIELD_NAMES:
            matches.append(node)
            seen.add(node_id)
    return matches


def _exact_field_name_match(node: Dict[str, Any], *names: str) -> bool:
    node_norm = _norm_text(node.get("name"))
    node_tokens = _field_tokens(node.get("name"))
    node_tokens_compact = _drop_redundant_plural_tokens(node_tokens)
    node_tokens_stripped = _strip_generated_variant_suffix_tokens(node_tokens_compact)
    for name in names:
        if not str(name or "").strip():
            continue
        if node_norm == _norm_text(name):
            return True
        name_tokens = _field_tokens(name)
        name_tokens_compact = _drop_redundant_plural_tokens(name_tokens)
        if node_tokens and node_tokens == name_tokens:
            return True
        if node_tokens_compact and node_tokens_compact == name_tokens_compact:
            return True
        if node_tokens_stripped and node_tokens_stripped == name_tokens_compact:
            return True
    return False


def _field_name_similarity(node_name: Any, field_name: Any, canonical_name: Any = "") -> float:
    node_norm = _norm_text(node_name)
    wanted_norms = [_norm_text(field_name)]
    canonical_norm = _norm_text(canonical_name)
    if canonical_norm and canonical_norm not in wanted_norms:
        wanted_norms.append(canonical_norm)
    if not node_norm or not any(wanted_norms):
        return 0.0
    if node_norm in wanted_norms:
        return 1.0
    if any(wanted and (wanted in node_norm or node_norm in wanted) for wanted in wanted_norms):
        return 0.92
    node_tokens = _field_tokens(node_name)
    node_tokens_stripped = _strip_generated_variant_suffix_tokens(node_tokens)
    best = 0.0
    for wanted in (field_name, canonical_name):
        wanted_tokens = _field_tokens(wanted)
        if not wanted_tokens or not node_tokens:
            continue
        if node_tokens_stripped and node_tokens_stripped == wanted_tokens:
            best = max(best, 0.98)
            continue
        overlap = wanted_tokens & node_tokens
        if not overlap:
            continue
        best = max(best, len(overlap) / min(len(wanted_tokens), len(node_tokens)))
    return best


def _weak_field_node_matches(
    nodes: Sequence[Dict[str, Any]],
    field_name: str,
    canonical_name: str,
    *,
    min_score: float = 0.55,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for node in nodes:
        score = _field_name_similarity(node.get("name"), field_name, canonical_name)
        if score >= min_score:
            scored.append((score, node))
    return [node for _, node in sorted(scored, key=lambda item: (-item[0], str(item[1].get("node_id"))))]


def _enum_values_from_constraint(text: Any) -> Tuple[int, ...]:
    raw = str(text or "").strip()
    if not raw.lower().startswith("enum:"):
        return tuple()
    payload = raw.split(":", 1)[1]
    pieces = re.split(r"[,|/]\s*|\s+or\s+", payload, flags=re.IGNORECASE)
    return _parse_numeric_values([piece.strip() for piece in pieces if piece.strip()], max_values=128)


def _range_from_constraint(text: Any) -> Optional[Tuple[int, int]]:
    raw = str(text or "").strip().lower()
    if not raw.startswith("range:"):
        return None
    nums = [_parse_int(item) for item in _NUM_RE.findall(raw)]
    nums = [n for n in nums if n is not None]
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None


def _expression_payload_for_targets(text: str, target_ids: Optional[set[str]]) -> Optional[str]:
    raw = str(text or "").strip()
    lowered = raw.lower()
    if not (lowered.startswith("formula:") or lowered.startswith("expression:")):
        return None
    payload = raw.split(":", 1)[1]
    refs = {str(match.group(1)).strip() for match in _VAL_REF_RE.finditer(payload)}
    if target_ids is not None and refs.isdisjoint(target_ids):
        return None
    # Remove field references before numeric extraction so node ids are not
    # mistaken for expected enum/range values.
    return _VAL_REF_RE.sub("FIELD", payload)


def _range_from_expression_payload(payload: str) -> Optional[Tuple[int, int]]:
    nums = [_parse_int(item) for item in _NUM_RE.findall(payload)]
    nums = [n for n in nums if n is not None]
    if len(nums) >= 2 and any(op in payload for op in ("<=", ">=", "<", ">")):
        return min(nums), max(nums)
    return None


def _term_sets_field_to_value(term: str, field_id: Any, value: int) -> bool:
    field = str(field_id).strip()
    val_ref = re.escape(f"val({field})")
    patterns = [
        re.compile(rf"{val_ref}\s*={{2,3}}\s*{re.escape(str(int(value)))}\b", re.IGNORECASE),
        re.compile(rf"\b{re.escape(str(int(value)))}\s*={{2,3}}\s*{val_ref}", re.IGNORECASE),
    ]
    return any(pattern.search(term) for pattern in patterns)


def _term_excludes_field_value(term: str, field_id: Any, value: int) -> bool:
    field = str(field_id).strip()
    val_ref = re.escape(f"val({field})")
    patterns = [
        re.compile(rf"{val_ref}\s*!=\s*{re.escape(str(int(value)))}\b", re.IGNORECASE),
        re.compile(rf"\b{re.escape(str(int(value)))}\s*!=\s*{val_ref}", re.IGNORECASE),
    ]
    return any(pattern.search(term) for pattern in patterns)


def _payload_targets_claim(payload: str, claim: ConstraintClaim, target_ids: Optional[set[str]]) -> bool:
    if not target_ids:
        return False
    if claim.kind == "enum" and claim.values:
        for target_id in target_ids:
            if all(_term_sets_field_to_value(payload, target_id, int(value)) for value in claim.values):
                return True
        return False
    if claim.kind == "range" and claim.min_value is not None and claim.max_value is not None:
        for target_id in target_ids:
            target = re.escape(f"val({str(target_id).strip()})")
            ge = re.search(rf"{target}\s*>=\s*{re.escape(str(int(claim.min_value)))}\b", payload, re.IGNORECASE)
            le = re.search(rf"{target}\s*<=\s*{re.escape(str(int(claim.max_value)))}\b", payload, re.IGNORECASE)
            if ge and le:
                return True
        return False
    return False


def _implication_formula_satisfies_claim(
    payload: str,
    claim: ConstraintClaim,
    *,
    target_ids: Optional[set[str]],
    guard_ids: Optional[set[str]],
) -> bool:
    if not claim.guard_field or not claim.guard_values or not guard_ids:
        return False
    if not _payload_targets_claim(payload, claim, target_ids):
        return False
    for guard_value in claim.guard_values:
        if not any(
            _term_excludes_field_value(payload, guard_id, int(guard_value))
            for guard_id in guard_ids
        ):
            return False
    return True


def _conditional_formula_satisfies_claim(
    constraint: Any,
    claim: ConstraintClaim,
    *,
    target_ids: Optional[set[str]],
    guard_ids: Optional[set[str]],
) -> bool:
    raw = str(constraint or "").strip()
    lowered = raw.lower()
    if not (lowered.startswith("formula:") or lowered.startswith("expression:")):
        return False
    if not target_ids:
        return False
    payload = raw.split(":", 1)[1]
    if _implication_formula_satisfies_claim(
        payload,
        claim,
        target_ids=target_ids,
        guard_ids=guard_ids,
    ):
        return True
    if not claim.values:
        return False
    terms = re.split(r"\bor\b|\|\|", payload, flags=re.IGNORECASE)
    for term in terms:
        stripped = term.strip().strip("()")
        target_ok = any(
            _term_sets_field_to_value(stripped, target_id, int(value))
            for target_id in target_ids
            for value in claim.values
        )
        if not target_ok:
            continue
        if not claim.guard_field or not claim.guard_values:
            return True
        if not guard_ids:
            continue
        guard_ok = any(
            _formula_covers_selector_value(stripped, guard_id, int(guard_value))
            for guard_id in guard_ids
            for guard_value in claim.guard_values
        )
        if guard_ok:
            return True
    return False


def _constraint_satisfies_claim(
    constraint: Any,
    claim: ConstraintClaim,
    *,
    target_ids: Optional[set[str]] = None,
    guard_ids: Optional[set[str]] = None,
    allow_bare_constraint: bool = True,
) -> bool:
    text = str(constraint or "")
    if _conditional_formula_satisfies_claim(
        text,
        claim,
        target_ids=target_ids,
        guard_ids=guard_ids,
    ):
        return True
    if claim.kind == "enum":
        if allow_bare_constraint:
            enum_values = set(_enum_values_from_constraint(text))
            if enum_values and enum_values == set(claim.values):
                return True
        # A formula/expression can be used on a variant to express a scoped
        # value constraint, e.g. expression: val(11) == 0.
        payload = _expression_payload_for_targets(text, target_ids)
        if payload and claim.values:
            nums = {_parse_int(item) for item in _NUM_RE.findall(payload)}
            nums.discard(None)
            values = {int(n) for n in nums if n is not None}
            if "==" in payload and values == set(claim.values):
                return True
            expr_range = _range_from_expression_payload(payload)
            if expr_range:
                lo, hi = expr_range
                return len(claim.values) == 1 and lo == hi == claim.values[0]
        if allow_bare_constraint:
            rng = _range_from_constraint(text)
            if rng and claim.values:
                lo, hi = rng
                return len(claim.values) == 1 and lo == hi == claim.values[0]
    if claim.kind == "range":
        if allow_bare_constraint:
            rng = _range_from_constraint(text)
            if rng and claim.min_value is not None and claim.max_value is not None:
                lo, hi = rng
                return lo == claim.min_value and hi == claim.max_value
        payload = _expression_payload_for_targets(text, target_ids)
        expr_range = _range_from_expression_payload(payload or "")
        if expr_range and claim.min_value is not None and claim.max_value is not None:
            lo, hi = expr_range
            return lo == claim.min_value and hi == claim.max_value
        if allow_bare_constraint:
            enum_values = set(_enum_values_from_constraint(text))
            if enum_values and claim.min_value is not None and claim.max_value is not None:
                return claim.min_value == claim.max_value and enum_values == {claim.min_value}
    return False


def _node_has_packed_bits_constraint(node: Dict[str, Any]) -> bool:
    return any(
        "packed_bits" in str(constraint or "").lower()
        for constraint in node.get("constraints") or []
    )


def _node_is_binary_payload(node: Dict[str, Any]) -> bool:
    text = " ".join(
        str(node.get(key) or "")
        for key in ("name", "node_type", "data_type", "encoding", "description")
    ).lower()
    return bool(re.search(r"\b(bitfield|bit[-_ ]?packed|binary|bytes?|payload|status)\b", text))


def _node_is_selector_like(node: Dict[str, Any]) -> bool:
    node_type = _norm_text(node.get("node_type"))
    if node_type in {"selector", "discriminator", "switch", "control"}:
        return True
    name_norm = _norm_text(_split_identifier_text(node.get("name")))
    if name_norm in _NON_SELECTOR_FIELD_NAMES:
        return False
    return _is_control_field_name(node.get("name"))


def _selector_enum_subset_covers_range(node: Dict[str, Any], claim: ConstraintClaim) -> bool:
    if claim.kind != "range" or claim.min_value is None or claim.max_value is None:
        return False
    if not _node_is_selector_like(node):
        return False
    enum_values: set[int] = set()
    for constraint in node.get("constraints") or []:
        enum_values.update(_enum_values_from_constraint(constraint))
    if not enum_values:
        return False
    return all(claim.min_value <= value <= claim.max_value for value in enum_values)


def _selector_enum_covers_enum_values(node: Dict[str, Any], claim: ConstraintClaim) -> bool:
    if claim.kind != "enum" or not claim.values:
        return False
    if not _node_is_selector_like(node):
        return False
    enum_values: set[int] = set()
    for constraint in node.get("constraints") or []:
        enum_values.update(_enum_values_from_constraint(constraint))
    return bool(enum_values) and set(claim.values) <= enum_values


def _selector_node_covers_domain_claim(node: Dict[str, Any], claim: ConstraintClaim) -> bool:
    return _selector_enum_covers_enum_values(node, claim) or _selector_enum_subset_covers_range(node, claim)


def _packed_or_binary_node_satisfies_claim(node: Dict[str, Any], claim: ConstraintClaim) -> bool:
    if not _node_has_packed_bits_constraint(node) and not _node_is_binary_payload(node):
        return False
    if claim.kind == "enum" and set(claim.values) in ({0, 1}, {0}, {1}):
        return _node_has_packed_bits_constraint(node)
    if (
        claim.kind == "range"
        and claim.min_value == 0
        and claim.max_value == 255
        and _node_is_binary_payload(node)
    ):
        return True
    return False


def _direct_constraint_for_claim(claim: ConstraintClaim) -> Optional[str]:
    if claim.kind == "enum" and claim.values:
        values = "|".join(str(value) for value in claim.values)
        return f"enum: {values}"
    if claim.kind == "range" and claim.min_value is not None and claim.max_value is not None:
        return f"range: {claim.min_value}..{claim.max_value}"
    return None


def _formula_constraint_for_claim(claim: ConstraintClaim, target_id: Any) -> Optional[str]:
    target = str(target_id)
    if claim.kind == "enum" and claim.values:
        expr = " or ".join(f"val({target}) == {value}" for value in claim.values)
        return f"formula: {expr}"
    if claim.kind == "range" and claim.min_value is not None and claim.max_value is not None:
        return f"formula: val({target}) >= {claim.min_value} and val({target}) <= {claim.max_value}"
    return None


def _guarded_formula_constraint_for_claim(
    claim: ConstraintClaim,
    target_id: Any,
    guard_id: Any,
) -> Optional[str]:
    guard = str(guard_id)
    guard_values = [int(value) for value in claim.guard_values]
    if not guard_values:
        return _formula_constraint_for_claim(claim, target_id)
    if len(guard_values) == 1:
        guard_clause = f"val({guard}) != {guard_values[0]}"
    else:
        guard_clause = " and ".join(f"val({guard}) != {value}" for value in guard_values)
    target = str(target_id)
    if claim.kind == "enum" and claim.values:
        target_clause = " or ".join(f"val({target}) == {int(value)}" for value in claim.values)
        return f"formula: {guard_clause} or ({target_clause})"
    if claim.kind == "range" and claim.min_value is not None and claim.max_value is not None:
        return f"formula: {guard_clause} or (val({target}) >= {claim.min_value} and val({target}) <= {claim.max_value})"
    return None


def _variant_guard_is_exact_for_claim(
    variant_id: Any,
    edges: Sequence[Dict[str, Any]],
    guard_nodes: Sequence[Dict[str, Any]],
    claim: ConstraintClaim,
) -> bool:
    guard_ids = {str(node.get("node_id")) for node in guard_nodes if node.get("node_id") is not None}
    if not guard_ids:
        return False
    wanted = {int(value) for value in claim.guard_values}
    if not wanted:
        return False
    for edge in edges:
        if edge.get("rel") != "condition_on" or str(edge.get("dst")) != str(variant_id):
            continue
        if str(edge.get("src")) not in guard_ids:
            continue
        explicit = _formula_values_for_selector(edge.get("formula"), edge.get("src"))
        if explicit and explicit == wanted:
            return True
    return False


def _first_guard_id_for_claim(guard_nodes: Sequence[Dict[str, Any]]) -> Optional[Any]:
    for node in guard_nodes:
        if node.get("node_id") is not None:
            return node.get("node_id")
    return None


def _descendant_ids(root_id: Any, children_by_parent: Dict[str, List[Any]]) -> set[str]:
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        current = stack.pop()
        key = str(current)
        for child in children_by_parent.get(key, []):
            child_key = str(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            stack.append(child)
    return seen


def _formula_values_for_selector(formula: Any, selector_id: Any) -> set[int]:
    text = str(formula or "")
    selector = str(selector_id).strip()
    if selector_id is not None and f"val({selector})" not in text:
        return set()
    # Only explicit positive equality/set-membership formulas prove that an
    # edge covers a concrete selector value.  Catch-all formulas such as
    # ``val(7) >= 1 and val(7) <= 127 and val(7) != 1`` mention many numeric
    # constants, but they do not positively cover those excluded values.
    val_ref = re.escape(f"val({selector})")
    values: set[int] = set()
    equality_patterns = [
        re.compile(rf"{val_ref}\s*={{2,3}}\s*({_NUM_RE.pattern})", re.IGNORECASE),
        re.compile(rf"({_NUM_RE.pattern})\s*={{2,3}}\s*{val_ref}", re.IGNORECASE),
    ]
    for pattern in equality_patterns:
        for match in pattern.finditer(text):
            number = _parse_int(match.group(1))
            if number is not None:
                values.add(int(number))

    membership_re = re.compile(
        rf"{val_ref}\s+in\s*(?:\{{(?P<brace>[^}}]+)\}}|\[(?P<bracket>[^\]]+)\])",
        re.IGNORECASE,
    )
    for match in membership_re.finditer(text):
        raw = match.group("brace") if match.group("brace") is not None else match.group("bracket")
        pieces = re.split(r"[,|/]\s*|\s+or\s+", str(raw or ""), flags=re.IGNORECASE)
        for number in _parse_numeric_values([piece.strip() for piece in pieces if piece.strip()], max_values=128):
            values.add(int(number))
    return values


def _formula_covers_selector_value(formula: Any, selector_id: Any, value: int) -> bool:
    """Return true when a condition_on predicate proves it covers ``value``.

    Exact equalities are handled by _formula_values_for_selector.  This helper
    additionally accepts simple catch-all predicates such as
    ``val(type) != 0 and val(type) != 1`` for a concrete documented table row.
    It deliberately ignores disjunctions to avoid treating broad prose-like
    formulas as proof.
    """

    explicit = _formula_values_for_selector(formula, selector_id)
    if explicit:
        return int(value) in explicit
    selector = str(selector_id).strip()
    if selector_id is None or f"val({selector})" not in str(formula or ""):
        return False
    text = str(formula or "")
    if re.search(r"\bor\b|\|\|", text, re.IGNORECASE):
        return False
    val_ref = re.escape(f"val({selector})")
    checks: List[bool] = []

    inequality_patterns = [
        (re.compile(rf"{val_ref}\s*!=\s*({_NUM_RE.pattern})", re.IGNORECASE), lambda n: int(value) != n),
        (re.compile(rf"({_NUM_RE.pattern})\s*!=\s*{val_ref}", re.IGNORECASE), lambda n: n != int(value)),
        (re.compile(rf"{val_ref}\s*>=\s*({_NUM_RE.pattern})", re.IGNORECASE), lambda n: int(value) >= n),
        (re.compile(rf"{val_ref}\s*>\s*({_NUM_RE.pattern})", re.IGNORECASE), lambda n: int(value) > n),
        (re.compile(rf"{val_ref}\s*<=\s*({_NUM_RE.pattern})", re.IGNORECASE), lambda n: int(value) <= n),
        (re.compile(rf"{val_ref}\s*<\s*({_NUM_RE.pattern})", re.IGNORECASE), lambda n: int(value) < n),
    ]
    for pattern, predicate in inequality_patterns:
        for match in pattern.finditer(text):
            number = _parse_int(match.group(1))
            if number is None:
                continue
            checks.append(bool(predicate(int(number))))

    return bool(checks) and all(checks)


def _tree_relationships(tree: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Any]], set[str]]:
    nodes = _tree_nodes(tree)
    edges = _tree_edges(tree)
    nodes_by_id = {str(n.get("node_id")): n for n in nodes}
    children_by_parent: Dict[str, List[Any]] = {}
    for node in nodes:
        parent = node.get("parent_id")
        if parent is not None:
            children_by_parent.setdefault(str(parent), []).append(node.get("node_id"))
        for child in node.get("children_ids") or []:
            children_by_parent.setdefault(str(node.get("node_id")), []).append(child)
    condition_variant_ids = {
        str(edge.get("dst"))
        for edge in edges
        if edge.get("rel") == "condition_on" and edge.get("dst") is not None
    }
    condition_variant_ids.update(
        str(node.get("node_id"))
        for node in nodes
        if str(node.get("node_type") or "").lower() == "variant" and node.get("node_id") is not None
    )
    return nodes_by_id, children_by_parent, condition_variant_ids


def _ancestor_ids(node: Dict[str, Any], nodes_by_id: Dict[str, Dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    parent = node.get("parent_id")
    while parent is not None:
        parent_id = str(parent)
        if parent_id in seen:
            break
        seen.add(parent_id)
        parent_node = nodes_by_id.get(parent_id)
        if not parent_node:
            break
        parent = parent_node.get("parent_id")
    return seen


def _condition_variants_for_guard(
    *,
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    guard_field: Optional[str],
    guard_values: Sequence[int],
    message_type: Optional[str],
    allow_structural_selector_fallback: bool = False,
) -> Tuple[set[str], set[int], List[Dict[str, Any]]]:
    """Find condition_on variants for a documented guard.

    Name matches are preferred.  If a document row says the guard is "Code" but
    the tree already has a selector with branches for the same documented
    values, the structural selector candidate is also considered.  This avoids
    baking in protocol-specific Code/Tag aliases while still using observable
    tree structure.
    """

    matched_guard_nodes = list(_match_guard_nodes(nodes, guard_field))
    guard_ids = {str(node.get("node_id")) for node in matched_guard_nodes if node.get("node_id") is not None}
    candidate_variant_ids: set[str] = set()
    covered_guard_values: set[int] = set()

    def consider_edge(edge: Dict[str, Any], *, require_named_guard: bool) -> None:
        if edge.get("rel") != "condition_on":
            return
        if not _edge_message_type_compatible(message_type, edge):
            return
        src = edge.get("src")
        if src is None:
            return
        if require_named_guard and str(src) not in guard_ids:
            return
        matched_values = {
            int(value)
            for value in guard_values
            if _formula_covers_selector_value(edge.get("formula"), edge.get("src"), int(value))
        }
        if matched_values:
            dst = str(edge.get("dst"))
            if not _message_type_compatible(message_type, nodes_by_id.get(dst)):
                return
            candidate_variant_ids.add(dst)
            covered_guard_values.update(matched_values)

    for edge in edges:
        consider_edge(edge, require_named_guard=True)

    missing_values = set(guard_values) - covered_guard_values
    if missing_values and allow_structural_selector_fallback and _guard_allows_structural_selector_fallback(guard_field):
        for edge in edges:
            consider_edge(edge, require_named_guard=False)
        for edge in edges:
            if edge.get("rel") != "condition_on":
                continue
            src = edge.get("src")
            if src is None or str(src) in guard_ids:
                continue
            matched_values = {
                int(value)
                for value in guard_values
                if _formula_covers_selector_value(edge.get("formula"), edge.get("src"), int(value))
            }
            if matched_values:
                node = nodes_by_id.get(str(src))
                if node is not None and str(src) not in {str(n.get("node_id")) for n in matched_guard_nodes}:
                    matched_guard_nodes.append(node)

    return candidate_variant_ids, set(guard_values) - covered_guard_values, matched_guard_nodes


def _selector_variant_comparison_tokens(value: Any) -> set[str]:
    tokens = _strip_generated_variant_suffix_tokens(_drop_redundant_plural_tokens(_field_tokens(value)))
    generic = _SELECTOR_VARIANT_SUFFIX_TOKENS | {
        "ie",
        "information",
        "element",
        "message",
        "msg",
        "option",
        "tlv",
        "variant",
        "variants",
    }
    compact: set[str] = set()
    for token in tokens:
        if token in generic:
            continue
        if re.fullmatch(r"(?:fc)?\d{1,5}", token):
            continue
        if re.fullmatch(r"0x[0-9a-f]{1,6}", token):
            continue
        compact.add(token)
    singles = {token for token in compact if len(token) == 1 and token.isalpha()}
    longs = {token for token in compact if len(token) > 2 and token.isalnum()}
    for prefix in singles:
        for tail in longs:
            compact.add(prefix + tail)
    return compact or tokens


def _selector_variant_name_matches(obligation: SelectorVariantObligation, node: Optional[Dict[str, Any]]) -> bool:
    if not node:
        return False
    node_name = str(node.get("name") or "")
    expected_norms = {
        _norm_text(obligation.variant_name),
        _norm_text(obligation.canonical_variant_name),
    }
    node_norm = _norm_text(node_name)
    if node_norm in expected_norms:
        return True
    expected = _selector_variant_comparison_tokens(obligation.canonical_variant_name or obligation.variant_name)
    actual = _selector_variant_comparison_tokens(node_name)
    if expected and expected <= actual:
        return True
    if len(actual) == 1 and (actual & expected) and any(len(token) > 3 for token in actual):
        return True
    long_expected = {token for token in expected if len(token) > 2}
    if long_expected and long_expected <= actual:
        return True
    if expected and actual and len(expected & actual) >= max(1, min(len(expected), 2)):
        # Accept short aliases such as F-TEID vs IE_Type_21_F_TEID, but not a
        # purely generic TLV/Data branch that lacks the documented variant label.
        return len(expected & actual) / max(1, len(expected)) >= 0.75
    return False


def _selector_variant_expected_fields_match(
    obligation: SelectorVariantObligation,
    variant: Optional[Dict[str, Any]],
    *,
    nodes_by_id: Dict[str, Dict[str, Any]],
    children_by_parent: Dict[str, List[Any]],
) -> bool:
    if not variant or variant.get("node_id") is None:
        return False
    expected_fields = tuple(
        field for field in getattr(obligation, "expected_field_names", tuple()) if str(field or "").strip()
    )
    if not expected_fields:
        return False
    scoped_nodes = [
        nodes_by_id[node_id]
        for node_id in _descendant_ids(variant.get("node_id"), children_by_parent)
        if node_id in nodes_by_id
    ]
    if not scoped_nodes:
        return False
    for expected in expected_fields:
        if not any(_exact_field_name_match(node, expected, _canonical_tree_name(expected)) for node in scoped_nodes):
            return False
    return True


def _compact_bit_expr(value: Any) -> str:
    text = "" if value is None else re.sub(r"\s+", "", str(value))
    while text.endswith("+0") or text.endswith("+0.0"):
        text = re.sub(r"\+0(?:\.0)?$", "", text)
    return text


def _expr_is_node_start(expr: str, node_id: Any) -> bool:
    if node_id is None:
        return False
    return _compact_bit_expr(expr) == f"{node_id}.bit_start"


def _expr_is_node_end(expr: str, node: Optional[Dict[str, Any]]) -> bool:
    if not node or node.get("node_id") is None:
        return False
    node_id = node.get("node_id")
    compact = _compact_bit_expr(expr)
    if compact == f"{node_id}.bit_start+{node_id}.size_bits":
        return True
    node_start = _compact_bit_expr(node.get("bit_start"))
    node_size = _parse_int(node.get("size_bits"))
    if node_start and node_size is not None and compact == f"{node_start}+{int(node_size)}":
        return True
    return False


def _selector_variant_body_boundary_compatible(
    obligation: SelectorVariantObligation,
    selector: Optional[Dict[str, Any]],
    variant: Optional[Dict[str, Any]],
    *,
    nodes_by_id: Dict[str, Dict[str, Any]],
) -> bool:
    """Reject source-backed variant bodies that overlap their selector field.

    A documented packet-format row with body fields means the guarded variant is
    the body after the selector, not another view starting at the selector byte.
    This is protocol-agnostic: it compares tree topology and bit expressions,
    never selector names or numeric value aliases.
    """

    expected_fields = tuple(
        field for field in getattr(obligation, "expected_field_names", tuple()) if str(field or "").strip()
    )
    if not expected_fields or not selector or not variant:
        return True

    selector_start = _compact_bit_expr(selector.get("bit_start"))
    variant_start = _compact_bit_expr(variant.get("bit_start"))
    if selector_start and variant_start and selector_start == variant_start:
        return False

    selector_parent_id = selector.get("parent_id")
    variant_parent_id = variant.get("parent_id")
    if selector_parent_id is not None and str(selector_parent_id) == str(variant_parent_id):
        if _expr_is_node_start(selector_start, selector_parent_id) and _expr_is_node_start(variant_start, selector_parent_id):
            return False

    if _expr_is_node_end(variant_start, selector):
        return True

    variant_parent = nodes_by_id.get(str(variant_parent_id)) if variant_parent_id is not None else None
    parent_start = _compact_bit_expr(variant_parent.get("bit_start")) if variant_parent else ""
    if variant_parent and _expr_is_node_end(parent_start, selector):
        return True
    if variant_parent and selector_start and parent_start == selector_start and _expr_is_node_start(variant_start, variant_parent_id):
        return False

    return True


def _selector_variant_obligation_coverage(
    tree: Dict[str, Any],
    obligation: SelectorVariantObligation,
) -> Tuple[bool, set[int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes_by_id, children_by_parent, _condition_variant_ids = _tree_relationships(tree)
    nodes = list(nodes_by_id.values())
    edges = _tree_edges(tree)
    matched_guard_nodes = _match_guard_nodes(nodes, obligation.selector_name)
    guard_ids = {str(node.get("node_id")) for node in matched_guard_nodes if node.get("node_id") is not None}
    covered_values: set[int] = set()
    matched_variants: List[Dict[str, Any]] = []

    def nested_condition_targets(selector_id: Any) -> List[Dict[str, Any]]:
        """Return body variants reached through a selector target.

        A source table can describe an outer selector value whose body starts
        with another selector.  In the tree that is represented as
        outer-selector -> inner-selector -> body-variant.  The outer selector
        obligation should be considered covered when one of the inner branches
        matches the documented body name or fields.
        """

        if selector_id is None:
            return []
        selector_key = str(selector_id)
        direct = nodes_by_id.get(selector_key)
        if not direct or str(direct.get("node_type") or "").lower() != "selector":
            return []
        targets: List[Dict[str, Any]] = []
        for nested_edge in edges:
            if nested_edge.get("rel") != "condition_on" or str(nested_edge.get("src")) != selector_key:
                continue
            target = nodes_by_id.get(str(nested_edge.get("dst")))
            if target is not None:
                targets.append(target)
        return targets

    for edge in edges:
        if edge.get("rel") != "condition_on":
            continue
        if not _edge_message_type_compatible(obligation.message_type, edge):
            continue
        src = edge.get("src")
        if src is None or str(src) not in guard_ids:
            continue
        dst = str(edge.get("dst"))
        matching_values = {
            int(value)
            for value in obligation.selector_values
            if _formula_covers_selector_value(edge.get("formula"), edge.get("src"), int(value))
        }
        if not matching_values:
            continue
        selector = nodes_by_id.get(str(src))
        candidates = [node for node in [nodes_by_id.get(dst)] if node is not None]
        candidates.extend(nested_condition_targets(dst))
        for variant in candidates:
            if not _message_type_compatible(obligation.message_type, variant):
                continue
            if not (
                _selector_variant_name_matches(obligation, variant)
                or _selector_variant_expected_fields_match(
                    obligation,
                    variant,
                    nodes_by_id=nodes_by_id,
                    children_by_parent=children_by_parent,
                )
            ):
                continue
            if not _selector_variant_body_boundary_compatible(
                obligation,
                selector,
                variant,
                nodes_by_id=nodes_by_id,
            ):
                continue
            covered_values.update(matching_values)
            matched_variants.append(variant)
            break

    missing_values = set(int(value) for value in obligation.selector_values) - covered_values
    return not missing_values, missing_values, matched_guard_nodes, matched_variants


def _selector_obligation_to_requirement(obligation: SelectorVariantObligation) -> StructuralFieldRequirement:
    return StructuralFieldRequirement(
        requirement_id=obligation.obligation_id,
        section_title=obligation.section_title,
        field_name=obligation.variant_name,
        canonical_name=obligation.canonical_variant_name,
        guard_field=obligation.selector_name,
        guard_values=obligation.selector_values,
        guard_source="selector_table",
        message_type=obligation.message_type,
        format_name=obligation.format_name,
        source=obligation.source,
    )


def find_uncovered_selector_variant_obligations(
    tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    *,
    max_issues: Optional[int] = None,
) -> List[Dict[str, Any]]:
    obligations = extract_selector_variant_obligations(sections)
    grouped: Dict[Tuple[str, str, str, Optional[str]], Dict[str, Any]] = {}
    for obligation in obligations:
        covered, missing_values, guard_nodes, _matched_variants = _selector_variant_obligation_coverage(tree, obligation)
        if covered:
            continue
        key = (
            obligation.section_title,
            obligation.selector_name,
            obligation.format_name,
            obligation.message_type,
        )
        group = grouped.setdefault(
            key,
            {
                "obligations": [],
                "missing_values": set(),
                "guard_nodes": {},
            },
        )
        group["obligations"].append(obligation)
        group["missing_values"].update(missing_values or set(obligation.selector_values))
        for node in guard_nodes:
            node_id = node.get("node_id")
            if node_id is not None:
                group["guard_nodes"][str(node_id)] = node

    issues: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        obligations_for_issue: List[SelectorVariantObligation] = group["obligations"]
        if not obligations_for_issue:
            continue
        representative = obligations_for_issue[0]
        missing_values = sorted(int(value) for value in group["missing_values"])
        guard_nodes = list(group["guard_nodes"].values())
        payload = {
            "section": representative.section_title,
            "selector": representative.selector_name,
            "format": representative.format_name,
            "values": missing_values,
            "variants": [ob.canonical_variant_name for ob in obligations_for_issue],
        }
        issues.append(
            {
                "id": "doc_structure_alignment_" + _stable_id(payload),
                "kind": "missing_selector_variants",
                "requirement": _selector_obligation_to_requirement(representative),
                "selector_obligations": obligations_for_issue,
                "missing_selector_values": missing_values,
                "expected_variants": [
                    {
                        "selector_name": ob.selector_name,
                        "selector_values": list(ob.selector_values),
                        "variant_name": ob.variant_name,
                        "canonical_variant_name": ob.canonical_variant_name,
                        "source": ob.source,
                    }
                    for ob in obligations_for_issue
                ],
                "matched_selector_node_ids": [
                    node.get("node_id") for node in guard_nodes if node.get("node_id") is not None
                ],
            }
        )
        if max_issues is not None and len(issues) >= max_issues:
            return issues
    return issues


def selector_variant_coverage_counts(tree: Dict[str, Any], sections: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    total = 0
    uncovered = 0
    for obligation in extract_selector_variant_obligations(sections):
        total += 1
        covered, _missing_values, _guard_nodes, _matched_variants = _selector_variant_obligation_coverage(tree, obligation)
        if not covered:
            uncovered += 1
    return total, uncovered


def _formula_is_catchall_selector_coverage(formula: Any, selector_id: Any) -> bool:
    if _formula_values_for_selector(formula, selector_id):
        return False
    text = str(formula or "")
    selector = str(selector_id).strip()
    if selector_id is None or f"val({selector})" not in text:
        return False
    if re.search(r"\bor\b|\|\|", text, re.IGNORECASE):
        return False
    return bool(re.search(r"!=|>=|<=|>|<", text))


def _node_is_tlv_length(node: Dict[str, Any]) -> bool:
    return _is_length_field_name(node.get("name"))


def _node_is_tlv_payload(node: Dict[str, Any]) -> bool:
    if _is_payloadish_field_name(node.get("name")):
        return True
    text = " ".join(str(node.get(key) or "") for key in ("node_type", "data_type", "type"))
    return bool(re.search(r"\b(bytes?|octets?|binary|payload|data|opaque)\b", text, re.IGNORECASE))


def _variant_has_tlv_envelope(
    variant_id: Any,
    *,
    nodes_by_id: Dict[str, Dict[str, Any]],
    children_by_parent: Dict[str, List[Any]],
) -> bool:
    search_ids = _descendant_ids(variant_id, children_by_parent)
    scoped_nodes = [nodes_by_id[nid] for nid in search_ids if nid in nodes_by_id]
    has_length = any(_node_is_tlv_length(node) for node in scoped_nodes)
    has_named_payload = any(_node_is_tlv_payload(node) for node in scoped_nodes)
    has_body_after_length = any(
        not _node_is_tlv_length(node)
        and str(node.get("node_type") or "").lower() in {"field", "payload", "container", "bytes"}
        for node in scoped_nodes
    )
    direct_nodes = [
        nodes_by_id[str(child_id)]
        for child_id in children_by_parent.get(str(variant_id), []) or []
        if str(child_id) in nodes_by_id
    ]
    direct_field_like = [
        node
        for node in direct_nodes
        if str(node.get("node_type") or "").strip().lower() in {"field", "payload", "container", "bytes"}
    ]
    has_second_body_field = len(direct_field_like) >= 2 and any(_node_is_tlv_length(node) for node in direct_field_like)
    has_payload = has_named_payload or has_body_after_length or has_second_body_field
    return has_length and has_payload


def _variant_has_zero_body(
    variant_id: Any,
    *,
    nodes_by_id: Dict[str, Dict[str, Any]],
    children_by_parent: Dict[str, List[Any]],
) -> bool:
    variant = nodes_by_id.get(str(variant_id))
    if not variant:
        return False
    size = _parse_int(variant.get("size_bits"))
    if size == 0:
        return True
    search_ids = _descendant_ids(variant_id, children_by_parent)
    if not search_ids:
        return False
    scoped_nodes = [nodes_by_id[nid] for nid in search_ids if nid in nodes_by_id]
    field_like = [
        node
        for node in scoped_nodes
        if str(node.get("node_type") or "").strip().lower() in {"field", "payload", "bytes"}
    ]
    return bool(field_like) and all(_parse_int(node.get("size_bits")) == 0 for node in field_like)


_TLV_ENVELOPE_FIELD_TOKENS = {
    "body",
    "data",
    "element",
    "ie",
    "information",
    "len",
    "length",
    "option",
    "payload",
    "tlv",
    "type",
    "value",
}
_FORMAT_VARIANT_IGNORABLE_TOKENS = _FIELD_MATCH_NODE_IGNORABLE_TOKENS | {
    "body",
    "common",
    "core",
    "definition",
    "definitions",
    "element",
    "field",
    "fields",
    "format",
    "ie",
    "information",
    "message",
    "messages",
    "option",
    "payload",
    "request",
    "response",
    "tlv",
    "value",
    "variant",
}


def _strip_format_variant_tokens(tokens: set[str]) -> set[str]:
    return _drop_redundant_plural_tokens(tokens) - _FORMAT_VARIANT_IGNORABLE_TOKENS


def _format_is_generic_tlv_envelope(fmt: Dict[str, Any], fields: Sequence[Dict[str, Any]]) -> bool:
    text_tokens = _field_tokens(_format_text(fmt))
    if not ({"tlv"} & text_tokens or {"information", "element"} <= text_tokens or {"option"} & text_tokens):
        return False
    core_rows: set[frozenset[str]] = set()
    for field in fields:
        name = str(field.get("field_name") or field.get("name") or "").strip()
        if not name:
            continue
        tokens = _field_tokens(name) - {"information", "element", "ie", "option", "tlv"}
        if not tokens:
            continue
        if not tokens <= _TLV_ENVELOPE_FIELD_TOKENS:
            return False
        core_rows.add(frozenset(tokens))
    has_selector = any(row in {frozenset({"type"})} for row in core_rows)
    has_length = any(row in {frozenset({"length"}), frozenset({"len"})} for row in core_rows)
    has_payload = any(row in {frozenset({"value"}), frozenset({"payload"}), frozenset({"data"}), frozenset({"body"})} for row in core_rows)
    return has_selector and has_length and has_payload


def _format_expected_field_specs(fmt: Dict[str, Any], section: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields = [field for field in (fmt.get("fields") or []) if isinstance(field, dict)]
    if not fields or _format_is_generic_tlv_envelope(fmt, fields):
        return []
    specs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for field in fields:
        name = str(field.get("field_name") or field.get("name") or "").strip()
        if not name or not _packet_format_field_is_claim_candidate(fmt, field, section):
            continue
        if _is_control_field_name(name) and _singleton_field_values(field):
            continue
        canonical = _canonical_tree_name(name)
        if not canonical:
            continue
        size_bits = _field_size_bits(field)
        direct_constraint = _structural_direct_constraint(field)
        required_state = _explicit_required_state(field)
        if required_state is False:
            continue
        if size_bits is None and direct_constraint is None and required_state is not True:
            continue
        norm = _norm_text(canonical)
        if norm in seen:
            continue
        seen.add(norm)
        specs.append(
            {
                "field_name": name,
                "canonical_name": canonical,
                "size_bits": size_bits,
                "direct_constraint": direct_constraint,
                "source": _field_source(section, field),
            }
        )
    return specs


def _expected_specs_need_variant_expansion(specs: Sequence[Dict[str, Any]]) -> bool:
    for spec in specs:
        size_bits = spec.get("size_bits")
        size_text = str(size_bits).strip().lower()
        if _is_payloadish_field_name(spec.get("field_name") or spec.get("canonical_name")) and size_text in {"0", "0.0"}:
            continue
        return True
    return False


def _format_has_guarded_selector(section: Dict[str, Any], fmt: Dict[str, Any], fields: Sequence[Dict[str, Any]]) -> bool:
    table_guard = _table_guard_candidates(section).get(_format_group_key(fmt, fields))
    if table_guard is not None and table_guard[1]:
        return True
    _guard_name, guard_values = _select_guard_field(fields)
    return bool(guard_values)


def _variant_matches_format(variant: Dict[str, Any], fmt: Dict[str, Any]) -> bool:
    format_name = str(fmt.get("format_name") or fmt.get("description") or "").strip()
    if not format_name:
        return False
    variant_tokens = _strip_format_variant_tokens(_field_tokens(variant.get("name")))
    format_tokens = _strip_format_variant_tokens(_field_tokens(format_name))
    if not variant_tokens or not format_tokens:
        return False
    if variant_tokens == format_tokens:
        return True
    overlap = variant_tokens & format_tokens
    if len(overlap) >= 2 and (overlap == variant_tokens or overlap == format_tokens):
        return True
    return bool(overlap) and len(overlap) / max(1, min(len(variant_tokens), len(format_tokens))) >= 0.8


def _variant_scoped_nodes(
    variant_id: Any,
    *,
    nodes_by_id: Dict[str, Dict[str, Any]],
    children_by_parent: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    search_ids = _descendant_ids(variant_id, children_by_parent)
    return [nodes_by_id[nid] for nid in search_ids if nid in nodes_by_id]


def _variant_opaque_child_ids(
    variant: Dict[str, Any],
    *,
    nodes_by_id: Dict[str, Dict[str, Any]],
    children_by_parent: Dict[str, List[Any]],
) -> List[Any]:
    direct_nodes = [
        nodes_by_id[str(child_id)]
        for child_id in children_by_parent.get(str(variant.get("node_id")), []) or []
        if str(child_id) in nodes_by_id
    ]
    if not direct_nodes:
        return []
    field_like = [
        node
        for node in direct_nodes
        if str(node.get("node_type") or "").strip().lower() in {"field", "payload", "bytes", "container"}
    ]
    if not field_like:
        return []
    opaque = [
        node
        for node in field_like
        if _node_is_tlv_payload(node)
        or bool(re.search(r"\b(?:opaque|payload|value|data|bytes?)\b", str(node.get("name") or ""), re.IGNORECASE))
    ]
    if len(opaque) != len(field_like):
        return []
    out: List[Any] = []
    seen: set[str] = set()
    for node in opaque:
        node_id = node.get("node_id")
        key = str(node_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(node_id)
    return out


def _variant_is_opaque_or_empty(
    variant: Dict[str, Any],
    *,
    nodes_by_id: Dict[str, Dict[str, Any]],
    children_by_parent: Dict[str, List[Any]],
) -> bool:
    direct_children = children_by_parent.get(str(variant.get("node_id")), []) or []
    return not direct_children or bool(
        _variant_opaque_child_ids(variant, nodes_by_id=nodes_by_id, children_by_parent=children_by_parent)
    )


def _format_missing_specs_for_variant(
    variant: Dict[str, Any],
    specs: Sequence[Dict[str, Any]],
    *,
    nodes_by_id: Dict[str, Dict[str, Any]],
    children_by_parent: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    scoped_nodes = _variant_scoped_nodes(
        variant.get("node_id"),
        nodes_by_id=nodes_by_id,
        children_by_parent=children_by_parent,
    )
    missing: List[Dict[str, Any]] = []
    for spec in specs:
        if any(_exact_field_name_match(node, spec["field_name"], spec["canonical_name"]) for node in scoped_nodes):
            continue
        missing.append(dict(spec))
    return missing


def _opaque_variant_expansion_issue_payload(
    *,
    section: Dict[str, Any],
    fmt: Dict[str, Any],
    variant: Dict[str, Any],
    expected_specs: Sequence[Dict[str, Any]],
    missing_specs: Sequence[Dict[str, Any]],
    opaque_child_ids: Sequence[Any],
) -> Dict[str, Any]:
    section_title = str(section.get("title") or section.get("section_title") or section.get("number") or "section")
    format_name = str(fmt.get("format_name") or fmt.get("description") or "")
    first = dict(missing_specs[0] if missing_specs else expected_specs[0])
    payload = {
        "kind": "opaque_variant_expansion",
        "section": section_title,
        "format": format_name,
        "variant_id": variant.get("node_id"),
        "missing": [spec.get("canonical_name") for spec in missing_specs],
    }
    requirement = StructuralFieldRequirement(
        requirement_id="doc_structure_" + _stable_id(payload),
        section_title=section_title,
        field_name=str(first.get("field_name") or ""),
        canonical_name=str(first.get("canonical_name") or _canonical_tree_name(first.get("field_name"))),
        guard_field="",
        guard_values=tuple(),
        guard_source="format_name",
        message_type=_infer_format_message_type(fmt, section),
        format_name=format_name,
        size_bits=first.get("size_bits"),
        direct_constraint=first.get("direct_constraint"),
        source=f"{first.get('source') or _field_source(section, {})} / {format_name}" if format_name else str(first.get("source") or ""),
    )
    return {
        "id": "doc_structure_alignment_" + _stable_id(payload),
        "kind": "opaque_variant_expansion",
        "requirement": requirement,
        "variant": variant,
        "expected_body_fields": [str(spec.get("canonical_name") or spec.get("field_name") or "") for spec in expected_specs],
        "expected_field_specs": [dict(spec) for spec in expected_specs],
        "missing_expected_fields": [str(spec.get("canonical_name") or spec.get("field_name") or "") for spec in missing_specs],
        "opaque_child_ids": list(opaque_child_ids),
    }


def _find_opaque_variant_expansion_issues(tree: Dict[str, Any], sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes_by_id, children_by_parent, condition_variant_ids = _tree_relationships(tree)
    variants = [
        node
        for node_id, node in nodes_by_id.items()
        if node_id in condition_variant_ids or str(node.get("node_type") or "").strip().lower() == "variant"
    ]
    issues: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        for fmt, fields in _iter_packet_format_field_groups(section):
            if _is_example_packet_format(fmt, section):
                continue
            if _format_has_guarded_selector(section, fmt, fields):
                continue
            expected_specs = _format_expected_field_specs(fmt, section)
            if not expected_specs or not _expected_specs_need_variant_expansion(expected_specs):
                continue
            expected_message_type = _infer_format_message_type(fmt, section)
            for variant in variants:
                variant_id = str(variant.get("node_id"))
                if not _message_type_compatible(expected_message_type, variant):
                    continue
                if not _variant_matches_format(variant, fmt):
                    continue
                missing_specs = _format_missing_specs_for_variant(
                    variant,
                    expected_specs,
                    nodes_by_id=nodes_by_id,
                    children_by_parent=children_by_parent,
                )
                if not missing_specs or not _variant_is_opaque_or_empty(
                    variant,
                    nodes_by_id=nodes_by_id,
                    children_by_parent=children_by_parent,
                ):
                    continue
                key = (variant_id, _norm_text(fmt.get("format_name") or fmt.get("description")))
                if key in seen:
                    continue
                seen.add(key)
                issues.append(
                    _opaque_variant_expansion_issue_payload(
                        section=section,
                        fmt=fmt,
                        variant=variant,
                        expected_specs=expected_specs,
                        missing_specs=missing_specs,
                        opaque_child_ids=_variant_opaque_child_ids(
                            variant,
                            nodes_by_id=nodes_by_id,
                            children_by_parent=children_by_parent,
                        ),
                    )
                )
    return issues


_REPEATED_FORMAT_IGNORABLE_TOKENS = {
    "answer",
    "answers",
    "authority",
    "additional",
    "entry",
    "entries",
    "message",
    "messages",
    "section",
    "sections",
}


def _repeated_format_key(value: Any) -> frozenset[str]:
    return frozenset(
        token
        for token in _strip_format_variant_tokens(_field_tokens(value))
        if token not in _REPEATED_FORMAT_IGNORABLE_TOKENS
    )


def _node_matches_repeated_format(node: Dict[str, Any], target_name: str) -> bool:
    target_tokens = _repeated_format_key(target_name)
    node_tokens = _repeated_format_key(node.get("name"))
    if not target_tokens or not node_tokens:
        return False
    if target_tokens <= node_tokens:
        return True
    overlap = target_tokens & node_tokens
    return len(overlap) >= 2 and overlap == node_tokens


def _format_matches_repeated_target(fmt: Dict[str, Any], target_name: str) -> bool:
    target_tokens = _repeated_format_key(target_name)
    format_tokens = _repeated_format_key(fmt.get("format_name") or fmt.get("description"))
    if not target_tokens or not format_tokens:
        return False
    if target_tokens <= format_tokens or format_tokens <= target_tokens:
        return True
    overlap = target_tokens & format_tokens
    return len(overlap) >= 2 and (overlap == target_tokens or overlap == format_tokens)


def _find_repeated_format_issues(tree: Dict[str, Any], sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes = _tree_nodes(tree)
    edges = _tree_edges(tree)
    nodes_by_id: Dict[str, Dict[str, Any]] = {
        str(node.get("node_id")): node for node in nodes if isinstance(node, dict) and node.get("node_id") is not None
    }
    children_by_parent: Dict[str, List[str]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        parent_id = node.get("node_id")
        if parent_id is None:
            continue
        children_by_parent[str(parent_id)] = [str(child) for child in (node.get("children_ids") or []) if child is not None]

    def _descendant_nodes(root_id: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        stack = [str(root_id)]
        seen_ids: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            node = nodes_by_id.get(nid)
            if node:
                out.append(node)
            stack.extend(reversed(children_by_parent.get(nid) or []))
        return out

    def _subtree_covers_repeated_format(
        subtree_nodes: Sequence[Dict[str, Any]],
        format_name: str,
        expected_fields: Sequence[str],
    ) -> bool:
        if any(_node_matches_repeated_format(node, format_name) for node in subtree_nodes):
            return True
        expected = [name for name in expected_fields if str(name or "").strip()]
        if expected:
            present = set()
            for field_name in expected:
                if _match_field_nodes(subtree_nodes, field_name):
                    present.add(field_name)
            if len(expected) <= 3 and len(present) == len(expected):
                return True
            if len(expected) > 3 and len(present) >= max(3, min(len(expected), 4)):
                return True
        return False

    def _repeat_targets_for_counter(counter_name: str) -> List[str]:
        counter_nodes = _match_field_nodes(nodes, counter_name)
        counter_ids = {str(node.get("node_id")) for node in counter_nodes if node.get("node_id") is not None}
        targets: List[str] = []
        for edge in edges:
            if edge.get("rel") != "repeat_count":
                continue
            if str(edge.get("src")) not in counter_ids:
                continue
            dst = edge.get("dst")
            if dst is not None:
                targets.append(str(dst))
        return targets

    issues: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or section.get("section_title") or section.get("number") or "section")
        formats = [
            (fmt, fields)
            for fmt, fields in _iter_packet_format_field_groups(section)
            if not _is_example_packet_format(fmt, section)
        ]
        all_formats: List[Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]] = []
        for source_section in sections or []:
            if not isinstance(source_section, dict):
                continue
            all_formats.extend(
                (fmt, fields)
                for fmt, fields in _iter_packet_format_field_groups(source_section)
                if not _is_example_packet_format(fmt, source_section)
            )
        for _fmt, fields in formats:
            for field in fields:
                if not isinstance(field, dict):
                    continue
                constraints = _field_constraints(field)
                if not (
                    constraints.get("repeat_count")
                    or constraints.get("repeat_count_for")
                    or constraints.get("entry_format")
                ):
                    continue
                field_name = str(field.get("field_name") or field.get("name") or "").strip()
                counter_name = str(constraints.get("repeat_count") or field_name).strip()
                target_name = str(
                    constraints.get("entry_format")
                    or constraints.get("repeat_count_for")
                    or field_name
                    or ""
                ).strip()
                if not counter_name or not target_name:
                    continue
                matched_format: Optional[Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]] = None
                for candidate_fmt, candidate_fields in all_formats:
                    if _format_matches_repeated_target(candidate_fmt, target_name):
                        matched_format = (candidate_fmt, candidate_fields)
                        break
                if matched_format is None:
                    continue
                fmt, fmt_fields = matched_format
                format_name = str(fmt.get("format_name") or fmt.get("description") or target_name)
                expected_specs = _format_expected_field_specs(fmt, section)
                expected_fields = [
                    str(spec.get("canonical_name") or spec.get("field_name") or "")
                    for spec in expected_specs
                    if str(spec.get("canonical_name") or spec.get("field_name") or "").strip()
                ]
                documented_fields = [
                    _canonical_tree_name(str(item.get("field_name") or item.get("name") or ""))
                    for item in fmt_fields
                    if isinstance(item, dict) and str(item.get("field_name") or item.get("name") or "").strip()
                ]
                for name in documented_fields:
                    if name and name not in expected_fields:
                        expected_fields.append(name)
                repeat_targets = _repeat_targets_for_counter(counter_name)
                if repeat_targets:
                    if any(
                        _subtree_covers_repeated_format(_descendant_nodes(target_id), format_name, expected_fields)
                        for target_id in repeat_targets
                    ):
                        continue
                elif any(_node_matches_repeated_format(node, format_name) for node in nodes):
                    continue
                key = (section_title, counter_name, format_name)
                if key in seen:
                    continue
                seen.add(key)
                requirement = StructuralFieldRequirement(
                    requirement_id="doc_structure_" + _stable_id(
                        {
                            "section": section_title,
                            "counter": counter_name,
                            "format": format_name,
                            "target": target_name,
                        }
                    ),
                    section_title=section_title,
                    field_name=format_name,
                    canonical_name=_canonical_tree_name(format_name),
                    guard_field=counter_name,
                    guard_values=tuple(),
                    guard_source="repeat_count",
                    message_type=_infer_format_message_type(fmt, section),
                    format_name=format_name,
                    source=f"{section_title}: {counter_name} repeats {format_name}",
                )
                issues.append(
                    {
                        "id": "doc_structure_alignment_"
                        + _stable_id(
                            {
                                "kind": "missing_repeated_format",
                                "section": section_title,
                                "counter": counter_name,
                                "format": format_name,
                            }
                        ),
                        "kind": "missing_repeated_format",
                        "requirement": requirement,
                        "expected_body_fields": expected_fields,
                        "expected_field_specs": [dict(spec) for spec in expected_specs],
                        "repeat_count_field": counter_name,
                        "repeated_format_name": format_name,
                    }
                )
    return issues


def _opaque_variant_expansion_counts(tree: Dict[str, Any], sections: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    nodes_by_id, children_by_parent, condition_variant_ids = _tree_relationships(tree)
    variants = [
        node
        for node_id, node in nodes_by_id.items()
        if node_id in condition_variant_ids or str(node.get("node_type") or "").strip().lower() == "variant"
    ]
    total = 0
    uncovered = 0
    seen: set[Tuple[str, str]] = set()
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        for fmt, fields in _iter_packet_format_field_groups(section):
            if _is_example_packet_format(fmt, section):
                continue
            if _format_has_guarded_selector(section, fmt, fields):
                continue
            expected_specs = _format_expected_field_specs(fmt, section)
            if not expected_specs or not _expected_specs_need_variant_expansion(expected_specs):
                continue
            expected_message_type = _infer_format_message_type(fmt, section)
            for variant in variants:
                if not _message_type_compatible(expected_message_type, variant):
                    continue
                if not _variant_matches_format(variant, fmt):
                    continue
                key = (str(variant.get("node_id")), _norm_text(fmt.get("format_name") or fmt.get("description")))
                if key in seen:
                    continue
                seen.add(key)
                total += 1
                missing_specs = _format_missing_specs_for_variant(
                    variant,
                    expected_specs,
                    nodes_by_id=nodes_by_id,
                    children_by_parent=children_by_parent,
                )
                if missing_specs:
                    uncovered += 1
    return total, uncovered


def _tree_has_tlv_abstraction(tree: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    nodes = _tree_nodes(tree)
    edges = _tree_edges(tree)
    nodes_by_id, children_by_parent, _condition_variant_ids = _tree_relationships(tree)
    guard_nodes = _match_guard_nodes(nodes, profile.get("guard_field"))
    guard_ids = {str(node.get("node_id")) for node in guard_nodes if node.get("node_id") is not None}
    if _guard_allows_structural_selector_fallback(profile.get("guard_field")):
        for node in nodes:
            if str(node.get("node_type") or "").strip().lower() != "selector":
                continue
            if _is_selector_alias_name(node.get("name")) and _norm_text(node.get("name")) not in _NON_SELECTOR_FIELD_NAMES:
                node_id = node.get("node_id")
                if node_id is not None:
                    guard_ids.add(str(node_id))
    length_values = {int(value) for value in profile.get("length_values") or []}
    if not length_values:
        return False
    zero_body_values = {int(value) for value in profile.get("zero_body_values") or []}

    catchall_covered_values: set[int] = set()
    exact_covered_values: set[int] = set()
    zero_body_covered_values: set[int] = set()
    for edge in edges:
        if edge.get("rel") != "condition_on":
            continue
        src = edge.get("src")
        if src is None:
            continue
        if guard_ids and str(src) not in guard_ids:
            continue
        zero_values = {
            value
            for value in zero_body_values
            if _formula_covers_selector_value(edge.get("formula"), src, value)
        }
        if zero_values:
            dst = str(edge.get("dst"))
            variant = nodes_by_id.get(dst)
            if _message_type_compatible(profile.get("message_type"), variant) and _variant_has_zero_body(
                dst,
                nodes_by_id=nodes_by_id,
                children_by_parent=children_by_parent,
            ):
                zero_body_covered_values.update(zero_values)
            continue
        covered_values = {
            value
            for value in length_values
            if _formula_covers_selector_value(edge.get("formula"), src, value)
        }
        if not covered_values:
            continue
        dst = str(edge.get("dst"))
        variant = nodes_by_id.get(dst)
        if not _message_type_compatible(profile.get("message_type"), variant):
            continue
        if not _variant_has_tlv_envelope(dst, nodes_by_id=nodes_by_id, children_by_parent=children_by_parent):
            continue
        if _formula_is_catchall_selector_coverage(edge.get("formula"), src):
            catchall_covered_values.update(covered_values)
        else:
            exact_covered_values.update(covered_values)
    return (
        length_values.issubset(catchall_covered_values | exact_covered_values)
        and zero_body_values.issubset(zero_body_covered_values)
    )


def _tlv_abstraction_covers_requirement(
    tree: Dict[str, Any],
    requirement: StructuralFieldRequirement,
    profiles: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]],
) -> bool:
    profile = _requirement_tlv_profile(requirement, profiles)
    if not profile or not _tree_has_tlv_abstraction(tree, profile):
        return False
    return _tlv_profile_represents_requirement(requirement, profile)


def _tlv_profile_represents_requirement(
    requirement: StructuralFieldRequirement,
    profile: Dict[str, Any],
) -> bool:
    """Return whether a single generic TLV profile should cover this row.

    This mirrors the abstraction used for repair prompts: once a repeated
    selector table is represented as one TLV envelope, length/payload-like row
    fields should not be double-counted as separate structural obligations.
    """

    length_values = {int(value) for value in profile.get("length_values") or []}
    zero_body_values = {int(value) for value in profile.get("zero_body_values") or []}
    guard_values = {int(value) for value in requirement.guard_values or []}

    if guard_values & zero_body_values and _is_zero_sized_requirement(requirement):
        return True
    if guard_values & length_values:
        if _is_length_field_name(requirement.field_name) or _is_payloadish_field_name(requirement.field_name):
            return True
    else:
        return False

    payload_abstraction = os.getenv("STEP2_DOC_STRUCTURE_TLV_PAYLOAD_ABSTRACTION", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        payload_abstraction
        and not requirement.direct_constraint
        and _is_payloadish_field_name(requirement.field_name)
        and not _is_control_field_name(requirement.field_name)
    ):
        return True
    return False


def _variant_ids_and_guard_nodes_for_requirement(
    tree: Dict[str, Any],
    requirement: StructuralFieldRequirement,
) -> Tuple[set[str], set[int], List[Dict[str, Any]]]:
    nodes = _tree_nodes(tree)
    edges = _tree_edges(tree)
    nodes_by_id = {str(n.get("node_id")): n for n in nodes}
    candidate_variant_ids, missing_guard_values, _guard_nodes = _condition_variants_for_guard(
        nodes=nodes,
        edges=edges,
        nodes_by_id=nodes_by_id,
        guard_field=requirement.guard_field,
        guard_values=requirement.guard_values,
        message_type=requirement.message_type,
        allow_structural_selector_fallback=requirement.guard_source == "table",
    )
    return candidate_variant_ids, missing_guard_values, _guard_nodes


def _variant_ids_for_requirement(tree: Dict[str, Any], requirement: StructuralFieldRequirement) -> Tuple[set[str], set[int]]:
    candidate_variant_ids, missing_guard_values, _guard_nodes = _variant_ids_and_guard_nodes_for_requirement(
        tree,
        requirement,
    )
    return candidate_variant_ids, missing_guard_values


def _variant_guard_is_exact_for_requirement(
    variant_id: Any,
    edges: Sequence[Dict[str, Any]],
    guard_nodes: Sequence[Dict[str, Any]],
    requirement: StructuralFieldRequirement,
) -> bool:
    guard_ids = {str(node.get("node_id")) for node in guard_nodes if node.get("node_id") is not None}
    if not guard_ids:
        return False
    wanted = {int(value) for value in requirement.guard_values}
    if not wanted:
        return False
    for edge in edges:
        if edge.get("rel") != "condition_on" or str(edge.get("dst")) != str(variant_id):
            continue
        if str(edge.get("src")) not in guard_ids:
            continue
        explicit = _formula_values_for_selector(edge.get("formula"), edge.get("src"))
        if explicit and explicit == wanted:
            return True
    return False


def _variant_guard_is_catchall_for_requirement(
    variant_id: Any,
    edges: Sequence[Dict[str, Any]],
    guard_nodes: Sequence[Dict[str, Any]],
    requirement: StructuralFieldRequirement,
) -> bool:
    guard_ids = {str(node.get("node_id")) for node in guard_nodes if node.get("node_id") is not None}
    if not guard_ids:
        return False
    for edge in edges:
        if edge.get("rel") != "condition_on" or str(edge.get("dst")) != str(variant_id):
            continue
        if str(edge.get("src")) not in guard_ids:
            continue
        if not _formula_is_catchall_selector_coverage(edge.get("formula"), edge.get("src")):
            continue
        if any(
            _formula_covers_selector_value(edge.get("formula"), edge.get("src"), int(value))
            for value in requirement.guard_values
        ):
            return True
    return False


def _requirement_needs_exact_selector_variant(requirement: StructuralFieldRequirement) -> bool:
    if _is_zero_sized_requirement(requirement):
        return False
    if _is_payloadish_field_name(requirement.field_name) and not requirement.direct_constraint:
        return False
    return True


def _has_shared_exact_field(
    nodes: Sequence[Dict[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    variant_ids: set[str],
    requirement: StructuralFieldRequirement,
) -> bool:
    for node in nodes:
        if not _exact_field_name_match(node, requirement.field_name, requirement.canonical_name):
            continue
        ancestors = _ancestor_ids(node, nodes_by_id)
        if str(node.get("node_id")) not in variant_ids and ancestors.isdisjoint(variant_ids):
            return True
    return False


def _structure_issue_payload(
    kind: str,
    requirement: StructuralFieldRequirement,
    *,
    variant: Optional[Dict[str, Any]] = None,
    existing_node: Optional[Dict[str, Any]] = None,
    missing_guard_values: Optional[Iterable[int]] = None,
    expected_body_fields: Optional[Sequence[str]] = None,
    tlv_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "kind": kind,
        "section": requirement.section_title,
        "field": requirement.field_name,
        "canonical_name": requirement.canonical_name,
        "guard_field": requirement.guard_field,
        "guard_values": list(requirement.guard_values),
        "guard_source": requirement.guard_source,
        "message_type": requirement.message_type,
        "format_name": requirement.format_name,
        "variant_id": variant.get("node_id") if variant else None,
        "variant_name": variant.get("name") if variant else None,
        "existing_node_id": existing_node.get("node_id") if existing_node else None,
        "existing_node_name": existing_node.get("name") if existing_node else None,
    }
    return {
        "id": "doc_structure_alignment_" + _stable_id(payload),
        "kind": kind,
        "requirement": requirement,
        "variant": variant,
        "existing_node": existing_node,
        "missing_guard_values": sorted(set(missing_guard_values or [])),
        "expected_body_fields": list(expected_body_fields or []),
        "tlv_profile": tlv_profile,
    }


def _tlv_structure_issue_payload(profile: Dict[str, Any]) -> Dict[str, Any]:
    requirement: StructuralFieldRequirement = profile["representative"]
    payload = {
        "kind": "missing_tlv_abstraction",
        "section": profile.get("section"),
        "guard_field": profile.get("guard_field"),
        "message_type": profile.get("message_type"),
        "length_values": list(profile.get("length_values") or []),
        "zero_body_values": list(profile.get("zero_body_values") or []),
    }
    expected_fields = ["Option_Length", "Option_Data"]
    return {
        "id": "doc_structure_alignment_" + _stable_id(payload),
        "kind": "missing_tlv_abstraction",
        "requirement": requirement,
        "tlv_profile": profile,
        "missing_guard_values": list(profile.get("length_values") or []),
        "expected_body_fields": expected_fields,
    }


def find_doc_structure_alignment_issues(
    tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    *,
    max_issues: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Find guarded fields that are structurally missing or weakly named."""

    requirements = extract_structural_field_requirements(sections)
    nodes = _tree_nodes(tree)
    nodes_by_id, children_by_parent, condition_variant_ids = _tree_relationships(tree)
    tlv_profiles = _tlv_profiles_for_requirements(requirements)
    missing_guarded_keys: set[Tuple[str, Tuple[int, ...], Optional[str], str]] = set()
    for requirement in requirements:
        profile = _requirement_tlv_profile(requirement, tlv_profiles)
        if profile and _requirement_tlv_group_key(requirement) in {
            key for key, candidate in tlv_profiles.items() if _tree_has_tlv_abstraction(tree, candidate)
        }:
            continue
        candidate_variant_ids, missing_guard_values, guard_nodes = _variant_ids_and_guard_nodes_for_requirement(
            tree,
            requirement,
        )
        exact_variant_ids = {
            variant_id
            for variant_id in candidate_variant_ids
            if _variant_guard_is_exact_for_requirement(variant_id, _tree_edges(tree), guard_nodes, requirement)
        }
        catchall_variant_ids = {
            variant_id
            for variant_id in candidate_variant_ids
            if _variant_guard_is_catchall_for_requirement(variant_id, _tree_edges(tree), guard_nodes, requirement)
        }
        if missing_guard_values or (
            (profile and not _tlv_profile_represents_requirement(requirement, profile))
            or (catchall_variant_ids and _requirement_needs_exact_selector_variant(requirement))
        ) and not exact_variant_ids:
            missing_guarded_keys.add(
                (requirement.section_title, tuple(requirement.guard_values), requirement.message_type, requirement.format_name)
            )
    issues: List[Dict[str, Any]] = []
    suppressed_tlv_keys: set[Tuple[str, str, Optional[str]]] = set()
    for key, profile in sorted(tlv_profiles.items(), key=lambda item: str(item[0])):
        if _tree_has_tlv_abstraction(tree, profile):
            continue
        suppressed_tlv_keys.add(key)
        issues.append(_tlv_structure_issue_payload(profile))
        if max_issues is not None and len(issues) >= max_issues:
            return issues

    for issue in _find_repeated_format_issues(tree, sections):
        issues.append(issue)
        if max_issues is not None and len(issues) >= max_issues:
            return issues

    for issue in find_uncovered_selector_variant_obligations(tree, sections):
        requirement = issue.get("requirement")
        if isinstance(requirement, StructuralFieldRequirement):
            key = (
                requirement.section_title,
                tuple(requirement.guard_values),
                requirement.message_type,
                requirement.format_name,
            )
            if key in missing_guarded_keys:
                continue
        issues.append(issue)
        if max_issues is not None and len(issues) >= max_issues:
            return issues

    for issue in _find_opaque_variant_expansion_issues(tree, sections):
        issues.append(issue)
        if max_issues is not None and len(issues) >= max_issues:
            return issues

    emitted_missing_variants: set[Tuple[str, Tuple[int, ...], Optional[str], str]] = set()
    body_fields_by_section: Dict[Tuple[str, Tuple[int, ...], Optional[str], str], List[str]] = {}
    for requirement in requirements:
        profile = _requirement_tlv_profile(requirement, tlv_profiles)
        if profile and _requirement_tlv_group_key(requirement) in suppressed_tlv_keys:
            continue
        if _tlv_abstraction_covers_requirement(tree, requirement, tlv_profiles):
            continue
        key = (requirement.section_title, tuple(requirement.guard_values), requirement.message_type, requirement.format_name)
        if not _has_shared_exact_field(nodes, nodes_by_id, condition_variant_ids, requirement):
            body_fields_by_section.setdefault(key, []).append(requirement.canonical_name)

    for requirement in requirements:
        profile = _requirement_tlv_profile(requirement, tlv_profiles)
        if profile and _requirement_tlv_group_key(requirement) in suppressed_tlv_keys:
            continue
        if _tlv_abstraction_covers_requirement(tree, requirement, tlv_profiles):
            continue
        candidate_variant_ids, missing_guard_values, guard_nodes = _variant_ids_and_guard_nodes_for_requirement(
            tree,
            requirement,
        )
        exact_variant_ids = {
            variant_id
            for variant_id in candidate_variant_ids
            if _variant_guard_is_exact_for_requirement(variant_id, _tree_edges(tree), guard_nodes, requirement)
        }
        catchall_variant_ids = {
            variant_id
            for variant_id in candidate_variant_ids
            if _variant_guard_is_catchall_for_requirement(variant_id, _tree_edges(tree), guard_nodes, requirement)
        }
        if (
            (profile and not _tlv_profile_represents_requirement(requirement, profile))
            or (catchall_variant_ids and _requirement_needs_exact_selector_variant(requirement))
        ):
            if not exact_variant_ids:
                key = (requirement.section_title, tuple(requirement.guard_values), requirement.message_type, requirement.format_name)
                if key not in emitted_missing_variants:
                    emitted_missing_variants.add(key)
                    issue = _structure_issue_payload(
                        "missing_guarded_variant",
                        requirement,
                        missing_guard_values=requirement.guard_values,
                        expected_body_fields=body_fields_by_section.get(key, []),
                        tlv_profile=profile,
                    )
                    issues.append(issue)
                if max_issues is not None and len(issues) >= max_issues:
                    return issues
                continue
            candidate_variant_ids = exact_variant_ids
        elif exact_variant_ids and catchall_variant_ids:
            candidate_variant_ids = exact_variant_ids
        if missing_guard_values:
            key = (requirement.section_title, tuple(requirement.guard_values), requirement.message_type, requirement.format_name)
            if key not in emitted_missing_variants:
                emitted_missing_variants.add(key)
                issue = _structure_issue_payload(
                    "missing_guarded_variant",
                    requirement,
                    missing_guard_values=missing_guard_values,
                    expected_body_fields=body_fields_by_section.get(key, []),
                    tlv_profile=profile,
                )
                issues.append(issue)
            if max_issues is not None and len(issues) >= max_issues:
                return issues
            continue

        if _has_shared_exact_field(nodes, nodes_by_id, condition_variant_ids, requirement):
            continue

        for variant_id in sorted(candidate_variant_ids):
            variant = nodes_by_id.get(str(variant_id))
            if not variant:
                continue
            search_ids = {str(variant_id)} | _descendant_ids(variant_id, children_by_parent)
            scoped_nodes = [nodes_by_id[nid] for nid in search_ids if nid in nodes_by_id]
            exact_nodes = [
                node
                for node in scoped_nodes
                if _exact_field_name_match(node, requirement.field_name, requirement.canonical_name)
            ]
            if exact_nodes:
                continue
            weak_nodes = _weak_field_node_matches(scoped_nodes, requirement.field_name, requirement.canonical_name)
            if weak_nodes:
                issue = _structure_issue_payload(
                    "weak_name_alignment",
                    requirement,
                    variant=variant,
                    existing_node=weak_nodes[0],
                )
            else:
                issue = _structure_issue_payload(
                    "missing_scoped_field",
                    requirement,
                    variant=variant,
                )
            issues.append(issue)
            if max_issues is not None and len(issues) >= max_issues:
                return issues
    return issues


def doc_structure_coverage_counts(
    tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    """Return ``(total_requirements, uncovered_requirements)``.

    Issue counts are intentionally capped and sometimes aggregated for prompt
    readability.  Reward shaping needs a stable denominator, so this counts the
    extracted structural requirements directly instead of counting emitted
    repair issues.
    """

    requirements = extract_structural_field_requirements(sections)
    nodes = _tree_nodes(tree)
    nodes_by_id, children_by_parent, condition_variant_ids = _tree_relationships(tree)
    tlv_profiles = _tlv_profiles_for_requirements(requirements)
    total = 0
    uncovered = 0
    opaque_total, opaque_uncovered = _opaque_variant_expansion_counts(tree, sections)
    total += opaque_total
    uncovered += opaque_uncovered
    repeated_issues = _find_repeated_format_issues(tree, sections)
    total += len(repeated_issues)
    uncovered += len(repeated_issues)
    selector_total, selector_uncovered = selector_variant_coverage_counts(tree, sections)
    total += selector_total
    uncovered += selector_uncovered

    for profile in tlv_profiles.values():
        total += 1
        if not _tree_has_tlv_abstraction(tree, profile):
            uncovered += 1

    for requirement in requirements:
        profile = _requirement_tlv_profile(requirement, tlv_profiles)
        if profile and _tlv_profile_represents_requirement(requirement, profile):
            continue
        total += 1
        if _tlv_abstraction_covers_requirement(tree, requirement, tlv_profiles):
            continue
        candidate_variant_ids, missing_guard_values = _variant_ids_for_requirement(tree, requirement)
        if missing_guard_values:
            uncovered += 1
            continue
        if _has_shared_exact_field(nodes, nodes_by_id, condition_variant_ids, requirement):
            continue

        covered = False
        for variant_id in sorted(candidate_variant_ids):
            search_ids = {str(variant_id)} | _descendant_ids(variant_id, children_by_parent)
            scoped_nodes = [nodes_by_id[nid] for nid in search_ids if nid in nodes_by_id]
            if any(
                _exact_field_name_match(node, requirement.field_name, requirement.canonical_name)
                for node in scoped_nodes
            ):
                covered = True
                break
        if not covered:
            uncovered += 1

    return total, uncovered


def _clean_relation_field_name(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;()[]{}")
    text = re.sub(r"\s+[-+]?\d+\s*$", "", text)
    text = re.sub(r"^(?:if|when|where|unless|then|and|or|the|a|an|this|that)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s+\b(?:number|list|length|size|count)\s+of\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:must|shall|should|may|can|is|are|be|been|set|cleared|clear|zero|only|then|if|when)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" .,:;()[]{}")
    if len(_field_tokens(text)) > 1:
        text = re.sub(r"\s+(?:field|fields|flag|flags|bit|bits|value|values)$", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" .,:;()[]{}")


def _relation_field_phrase_ok(value: Any) -> bool:
    text = _clean_relation_field_name(value)
    if not text:
        return False
    tokens = _field_tokens(text)
    if not tokens or tokens <= _RELATION_STOP_TOKENS:
        return False
    if len(tokens) > 6:
        return False
    if any(token in _RELATION_FIELD_KEYWORDS for token in tokens):
        return True
    # Short uppercase labels such as DF/MF/P/F are common selector or flag
    # names.  Treat them as field-like without knowing the protocol.
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    return bool(compact and compact.upper() == compact and 1 <= len(compact) <= 8)


def _dedupe_relation_fields(values: Iterable[Any]) -> Tuple[str, ...]:
    fields: List[str] = []
    seen: set[str] = set()
    for value in values:
        field = _clean_relation_field_name(value)
        key = _norm_text(field)
        if not field or key in seen or not _relation_field_phrase_ok(field):
            continue
        seen.add(key)
        fields.append(field)
    return tuple(fields)


def _dedupe_count_relation_fields(values: Iterable[Any]) -> Tuple[str, ...]:
    fields: List[str] = []
    seen: set[str] = set()
    for value in values:
        field = _clean_relation_field_name(value)
        raw = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;()[]{}")
        raw = re.sub(r"\s+[-+]?\d+\s*$", "", raw)
        if (
            raw
            and re.search(r"\bvalues?$", raw, re.IGNORECASE)
            and _norm_text(field)
            and _norm_text(raw).startswith(_norm_text(field))
            and len(_field_tokens(raw)) <= 8
        ):
            field = raw
        key = _norm_text(field)
        tokens = _field_tokens(field)
        if not field or key in seen or not tokens or tokens <= _RELATION_STOP_TOKENS or len(tokens) > 8:
            continue
        seen.add(key)
        fields.append(field)
    return tuple(fields)


def _candidate_relation_fields(text: Any) -> Tuple[str, ...]:
    raw = str(text or "")
    raw = re.sub(r"[-_/]+", " ", raw)
    descriptor_match = re.search(
        r"^\s*(?P<field>[A-Za-z][A-Za-z0-9_/-]{0,24})\s+"
        r"(?:length|len|size|count|number)\s+of\s+"
        r"(?P<desc>[A-Za-z][A-Za-z0-9 _/-]{0,64}?)(?:\bmust\b|\bshall\b|\bis\b|$)",
        raw,
        re.IGNORECASE,
    )
    descriptor_candidates: List[str] = []
    if descriptor_match:
        descriptor_candidates.append(descriptor_match.group("field"))
        descriptor_candidates.append(descriptor_match.group("desc"))
    pieces = re.split(r"\b(?:and|or|,|;|/)\b", raw, flags=re.IGNORECASE)
    candidates: List[str] = list(descriptor_candidates)
    for piece in pieces:
        piece = re.sub(r"^\s*(?:[-+]?0x[0-9a-fA-F]+|[-+]?\d+)\s*(?:[+*/-]\s*)?", "", piece)
        piece = re.sub(
            r"\b(?:must|shall|should|may|can|is|are|be|been|set|cleared|clear|zero|only|then)\b.*$",
            "",
            piece,
            flags=re.IGNORECASE,
        )
        piece = _clean_relation_field_name(piece)
        if _relation_field_phrase_ok(piece):
            candidates.append(piece)
    if candidates:
        return _dedupe_relation_fields(candidates)

    matches = re.findall(_RELATION_EXPR_NAME, raw)
    return _dedupe_relation_fields(matches)


def _relation_sentences_from_section(section: Dict[str, Any]) -> List[Tuple[str, str]]:
    section_title = str(section.get("title") or section.get("section_title") or section.get("number") or "section")
    texts: List[Tuple[str, str]] = []
    raw = _section_raw_text(section)
    if raw.strip():
        texts.append((section_title, raw))
    for fmt in section.get("packet_formats") or []:
        if not isinstance(fmt, dict) or _is_example_packet_format(fmt, section):
            continue
        fmt_name = str(fmt.get("format_name") or fmt.get("name") or "").strip()
        fmt_source = f"{section_title} / {fmt_name}" if fmt_name else section_title
        fmt_text = " ".join(str(fmt.get(key) or "") for key in ("format_name", "description", "notes"))
        if fmt_text.strip():
            texts.append((fmt_source, fmt_text))
        for field in fmt.get("fields") or []:
            if not isinstance(field, dict):
                continue
            field_text = " ".join(
                str(field.get(key) or "")
                for key in ("field_name", "name", "description", "notes", "semantic_meaning")
            )
            constraints = field.get("constraints")
            if isinstance(constraints, dict):
                relation_values: List[str] = []
                for value in constraints.values():
                    if not isinstance(value, str):
                        continue
                    if re.search(
                        r"\b(?:if|when|must|shall|equal|equals|length|multiple|modulo|aligned|alignment|boundary|padding|zero)\b",
                        value,
                        re.IGNORECASE,
                    ):
                        relation_values.append(value)
                if relation_values:
                    field_text += " " + " ".join(relation_values)
            if field_text.strip():
                field_name = field.get("field_name") or field.get("name") or "field"
                texts.append((f"{fmt_source} / {field_name}", field_text))

    sentences: List[Tuple[str, str]] = []
    for source, text in texts:
        parts = _RELATION_SENTENCE_SPLIT_RE.split(str(text))
        for idx, sentence in enumerate(parts):
            normalized = re.sub(r"\s+", " ", sentence).strip()
            if normalized and normalized[0].islower() and idx > 0:
                previous = re.sub(r"\s+", " ", parts[idx - 1]).strip()
                if _is_optional_relation_sentence(previous):
                    normalized = f"{previous} {normalized}".strip()
            if len(normalized) < 12:
                continue
            if _EXAMPLE_FORMAT_RE.search(normalized):
                continue
            sentences.append((source, normalized))
    return sentences


def _build_relation_obligation(
    section_title: str,
    kind: str,
    relation_text: str,
    *,
    field_names: Sequence[str],
    primary_field: Optional[str] = None,
    related_fields: Sequence[str] = (),
    numeric_values: Sequence[int] = (),
    source: str = "",
    format_name: str = "",
    message_type: Optional[str] = None,
) -> RelationalObligation:
    if kind == "count_relation":
        fields = _dedupe_count_relation_fields(field_names)
        primary_key = _norm_text(primary_field or "")
        related = tuple(
            field
            for field in _dedupe_count_relation_fields(related_fields)
            if _norm_text(field) != primary_key
        )
    else:
        fields = _dedupe_relation_fields(field_names)
        related = tuple(field for field in _dedupe_relation_fields(related_fields) if field not in fields)
    payload = {
        "section": section_title,
        "kind": kind,
        "text": relation_text,
        "fields": fields,
        "primary": primary_field,
        "related": related,
        "numbers": list(numeric_values),
        "source": source,
    }
    return RelationalObligation(
        obligation_id="doc_relational_obligation_" + _stable_id(payload),
        section_title=section_title,
        kind=kind,
        relation_text=relation_text,
        field_names=fields,
        primary_field=primary_field,
        related_fields=related,
        numeric_values=tuple(sorted(set(int(value) for value in numeric_values))),
        message_type=message_type,
        format_name=format_name,
        source=source,
    )


def _is_optional_relation_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    # "may not both be set" is a prohibition, not an optional capability.
    if re.search(r"\b(?:may|can|could)\s+not\b", lower):
        return False
    if re.search(r"\b(?:must|shall|required|requires?|always|ensure|ensures|ensured)\b", lower):
        return False
    return bool(
        re.search(r"\b(?:may|can|could|optional|optionally)\b", lower)
        or "for example" in lower
        or "e.g." in lower
    )


_STRUCTURED_PRESENT_WHEN_RE = re.compile(
    r"(?P<target>[A-Za-z0-9][A-Za-z0-9_ /\-+]{0,96}?)\s+"
    r"(?:field\s+)?(?:shall\s+be\s+|is\s+)?present\s+when\b",
    re.IGNORECASE,
)


def _structured_relation_field_names(field_name: str, relation_text: Any) -> Tuple[str, ...]:
    names: List[str] = []
    if str(field_name or "").strip():
        names.append(str(field_name).strip())
    text = str(relation_text or "").strip()
    match = _STRUCTURED_PRESENT_WHEN_RE.search(text)
    if match:
        target = _clean_relation_field_name(match.group("target"))
        if target and _relation_field_phrase_ok(target):
            names.append(target)
    return _dedupe_relation_fields(names)


def _structured_relation_kind(relation_text: Any) -> str:
    text = str(relation_text or "").lower()
    if re.search(r"\b(?:length|size|bytes?|octets?|payload|excludes?|includes?)\b", text):
        return "length_relation"
    if re.search(r"\b(?:present|presence|if|when|only\s+if|otherwise|shall\s+not)\b", text):
        return "conditional_value"
    return "conditional_value"


_COUNT_PHRASE_RE = re.compile(
    r"\b(?:quantity|number|count)\s+of\s+"
    r"[A-Za-z][A-Za-z0-9_ /\-]{0,64}?(?=\s*(?:[.,;:)]|$|\b(?:x|times|multiplied|bytes?|octets?|bits?)\b))",
    re.IGNORECASE,
)


def _count_phrases_from_relation_text(text: Any) -> Tuple[str, ...]:
    candidates = [match.group(0) for match in _COUNT_PHRASE_RE.finditer(str(text or ""))]
    return _dedupe_count_relation_fields(candidates)


def _field_text_blob(field: Dict[str, Any]) -> str:
    pieces = [
        str(field.get(key) or "")
        for key in ("field_name", "name", "size", "description", "notes", "semantic_meaning")
    ]
    constraints = field.get("constraints")
    if isinstance(constraints, dict):
        pieces.extend(str(value or "") for value in constraints.values())
    return " ".join(piece for piece in pieces if piece).strip()


def _field_has_dynamic_count_size(field: Dict[str, Any]) -> bool:
    text = _field_text_blob(field)
    return bool(
        re.search(r"\bN\s*\*", text, re.IGNORECASE)
        or re.search(r"\b(?:quantity|count|number)\s+of\b", text, re.IGNORECASE)
        or re.search(r"\brepeat(?:s|ed|_count)?\b", text, re.IGNORECASE)
    ) and bool(re.search(r"\b(?:bytes?|octets?|bits?|registers?|words?)\b", text, re.IGNORECASE))


def _format_relation_field_candidates(
    relation_text: Any,
    *,
    fields: Sequence[Dict[str, Any]],
    primary_field_name: str,
) -> Tuple[str, ...]:
    text = str(relation_text or "")
    text_key = _norm_text(text)
    candidates: List[str] = []
    primary_key = _norm_text(primary_field_name)
    primary_tokens = _field_tokens(primary_field_name)

    candidates.extend(_count_phrases_from_relation_text(text))

    relation_mentions_size = bool(
        re.search(
            r"\b(?:count|quantity|number|length|size|bytes?|octets?|bits?|registers?|words?)\b|[x*]",
            text,
            re.IGNORECASE,
        )
    )
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("field_name") or field.get("name") or "").strip()
        if not name or _norm_text(name) == primary_key:
            continue
        name_key = _norm_text(name)
        if name_key and name_key in text_key:
            candidates.append(name)
            continue
        if (
            relation_mentions_size
            and (primary_tokens & {"count", "length", "len", "size"})
            and _field_has_dynamic_count_size(field)
        ):
            candidates.append(name)

    return _dedupe_count_relation_fields(candidates)


def _extract_structured_relation_obligations(
    section: Dict[str, Any],
    section_title: str,
) -> List[RelationalObligation]:
    obligations: List[RelationalObligation] = []
    for fmt, fields in _iter_packet_format_field_groups(section):
        if _is_example_packet_format(fmt, section):
            continue
        if _packet_format_is_selector_table(fmt, section):
            continue
        format_name = str(fmt.get("format_name") or fmt.get("description") or "")
        message_type = _infer_format_message_type(fmt, section)
        for field in fields:
            if not isinstance(field, dict):
                continue
            field_name = str(field.get("field_name") or field.get("name") or "").strip()
            if not field_name:
                continue
            constraints = _field_constraints(field)
            relation_values = [
                ("conditional", constraints.get("conditional")),
                ("formula", constraints.get("formula")),
                ("length_relation", constraints.get("length_relation")),
                ("relation", constraints.get("relation")),
                ("repeat_count", constraints.get("repeat_count") or constraints.get("repeat_count_for")),
            ]
            for relation_key, raw_relation in relation_values:
                relation_text = str(raw_relation or "").strip()
                if not relation_text:
                    continue
                if relation_key == "repeat_count":
                    field_names = _dedupe_relation_fields((field_name,))
                    target_name = _clean_relation_field_name(relation_text)
                    if target_name:
                        target_key = _norm_text(target_name)
                        existing_keys = {_norm_text(value) for value in field_names}
                        if target_key and target_key not in existing_keys:
                            field_names = tuple(list(field_names) + [target_name])
                    field_names = field_names or (field_name,)
                    source_text = f"{field_name}: repeat count controlled by {relation_text}"
                    kind = "count_relation"
                else:
                    field_names = _structured_relation_field_names(field_name, relation_text) or (field_name,)
                    source_text = f"{field_name}: {relation_text}"
                    kind = _structured_relation_kind(relation_text)
                    if relation_key in {"formula", "length_relation"}:
                        kind = "length_relation"
                    elif relation_key == "relation" and re.search(
                        r"\b(?:count|number\s+of|quantity\s+of|entries?|records?|items?)\b",
                        relation_text,
                        re.IGNORECASE,
                    ):
                        kind = "count_relation"
                    elif re.search(
                        r"\b(?:count|number\s+of|quantity\s+of|entries?|records?|items?)\b",
                        relation_text,
                        re.IGNORECASE,
                    ) and re.search(
                        r"\b(?:equal|equals|determined|based|per|times|multiplied)\b|[x*]",
                        relation_text,
                        re.IGNORECASE,
                    ):
                        kind = "count_relation"
                    format_related = _format_relation_field_candidates(
                        relation_text,
                        fields=fields,
                        primary_field_name=field_name,
                    )
                    if format_related:
                        field_names = tuple(list(field_names) + list(format_related))

                if kind == "count_relation":
                    field_names = _dedupe_count_relation_fields(field_names) or (field_name,)
                else:
                    field_names = _dedupe_relation_fields(field_names) or (field_name,)
                obligations.append(
                    _build_relation_obligation(
                        section_title,
                        kind,
                        source_text,
                        field_names=field_names,
                        primary_field=field_name,
                        related_fields=[
                            name for name in field_names if _norm_text(name) != _norm_text(field_name)
                        ],
                        source=f"{_field_source(section, field)} / {format_name}" if format_name else _field_source(section, field),
                        format_name=format_name,
                        message_type=message_type,
                    )
                )
    return obligations


def _extract_conditional_zero_obligations(
    section_title: str,
    sentence: str,
    source: str,
) -> List[RelationalObligation]:
    if _is_optional_relation_sentence(sentence):
        return []
    if not re.search(r"\b(?:if|when|where|unless)\b", sentence, re.IGNORECASE):
        return []
    if not (
        re.search(r"\bzero\b|0x0", sentence, re.IGNORECASE)
        or re.search(r"(?:==|=|is|be)\s*0\b", sentence, re.IGNORECASE)
    ):
        return []
    split_match = re.search(r"\b(?:then|must|shall|should|required|requires?)\b", sentence, re.IGNORECASE)
    if split_match:
        left = sentence[: split_match.start()]
        right = sentence[split_match.end() :]
        if "," in left:
            trigger_part, target_prefix = left.rsplit(",", 1)
            if target_prefix.strip() and not re.search(r"\b(?:if|when|where|unless)\b", target_prefix, re.IGNORECASE):
                left = trigger_part
                right = f"{target_prefix} {right}"
        pieces = [left, right]
    else:
        pieces = re.split(r"[,;]", sentence, maxsplit=1)
    if len(pieces) < 2:
        return []
    if not (
        re.search(r"\bzero\b|0x0", pieces[1], re.IGNORECASE)
        or re.search(r"(?:==|=|is|be)\s*0\b", pieces[1], re.IGNORECASE)
    ):
        return []
    trigger_fields = _candidate_relation_fields(pieces[0])
    target_fields = _candidate_relation_fields(pieces[1])
    target_fields = tuple(field for field in target_fields if _norm_text(field) not in {_norm_text(v) for v in trigger_fields})
    obligations: List[RelationalObligation] = []
    for target in target_fields[:4]:
        fields = tuple(trigger_fields[:2]) + (target,)
        if len(fields) < 2:
            continue
        obligations.append(
            _build_relation_obligation(
                section_title,
                "conditional_zero",
                sentence,
                field_names=fields,
                primary_field=target,
                related_fields=trigger_fields,
                numeric_values=(0, 1),
                source=source,
            )
        )
    return obligations


def _extract_assignment_obligations(
    section_title: str,
    sentence: str,
    source: str,
) -> List[RelationalObligation]:
    if _is_optional_relation_sentence(sentence):
        return []
    assignments = list(_RELATION_ASSIGNMENT_RE.finditer(sentence))
    if len(assignments) < 2:
        return []
    if not re.search(r"\b(?:implies|requires?|then)\b|=>|->", sentence, re.IGNORECASE):
        return []
    trigger = assignments[0]
    obligations: List[RelationalObligation] = []
    trigger_field = _clean_relation_field_name(trigger.group("field"))
    trigger_value = _parse_int(trigger.group("value"))
    if not _relation_field_phrase_ok(trigger_field) or trigger_value is None:
        return []
    for assignment in assignments[1:]:
        target_field = _clean_relation_field_name(assignment.group("field"))
        target_value = _parse_int(assignment.group("value"))
        if not _relation_field_phrase_ok(target_field) or target_value is None:
            continue
        obligations.append(
            _build_relation_obligation(
                section_title,
                "conditional_value",
                sentence,
                field_names=(trigger_field, target_field),
                primary_field=target_field,
                related_fields=(trigger_field,),
                numeric_values=(trigger_value, target_value),
                source=source,
            )
        )
    return obligations


def _extract_mutual_exclusion_obligations(
    section_title: str,
    sentence: str,
    source: str,
) -> List[RelationalObligation]:
    if _is_optional_relation_sentence(sentence):
        return []
    lower = sentence.lower()
    if not (
        "not both" in lower
        or "must not both" in lower
        or "shall not both" in lower
        or "not simultaneously" in lower
        or "mutually exclusive" in lower
    ):
        return []
    head = re.split(
        r"\b(?:must|shall|should|may|can)?\s*not\s+(?:both|simultaneously)\b|\bmutually\s+exclusive\b",
        sentence,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    fields = _candidate_relation_fields(head)
    if len(fields) < 2:
        fields = _candidate_relation_fields(sentence)
    if len(fields) < 2:
        return []
    return [
        _build_relation_obligation(
            section_title,
            "mutual_exclusion",
            sentence,
            field_names=fields[:4],
            primary_field=fields[0],
            related_fields=fields[1:4],
            numeric_values=(1,),
            source=source,
        )
    ]


def _extract_length_relation_obligations(
    section_title: str,
    sentence: str,
    source: str,
) -> List[RelationalObligation]:
    if _is_optional_relation_sentence(sentence):
        return []
    obligations: List[RelationalObligation] = []
    for match in _RELATION_LENGTH_EXPR_RE.finditer(sentence):
        lhs = _clean_relation_field_name(match.group("lhs"))
        expr = match.group("expr")
        expr_clean = re.sub(r"\s+", " ", str(expr or "")).strip(" .,:;()[]{}")
        if re.fullmatch(r"(?:ok|yes|no|supported|unsupported|error|exception|valid|invalid)", expr_clean, re.IGNORECASE):
            continue
        if not (_field_tokens(lhs) & {"length", "len", "size", "offset", "count"}):
            continue
        if not _relation_field_phrase_ok(lhs):
            continue
        rhs_fields = [
            field
            for field in _candidate_relation_fields(expr)
            if _norm_text(field) != _norm_text(lhs)
        ]
        numbers = [_parse_int(value) for value in _NUM_RE.findall(expr)]
        numbers = [int(value) for value in numbers if value is not None]
        if re.search(r"\b(?:octets?|bytes?)\b", expr, re.IGNORECASE) and 8 not in numbers:
            numbers.append(8)
        relation_cue = bool(
            re.search(
                r"[+*/%]|\b(?:plus|minus|times|multiplied|multiple|modulo|measured\s+in|units?\s+of|aligned|alignment)\b",
                expr,
                re.IGNORECASE,
            )
        )
        if not numbers and re.search(r"\b(?:limited\s+by|dependent\s+on|depends\s+on)\b", expr, re.IGNORECASE) and not relation_cue:
            continue
        if re.search(r"\bfixed\s+length\b", expr, re.IGNORECASE) and not relation_cue:
            continue
        if not rhs_fields and not relation_cue:
            continue
        if not rhs_fields and not numbers:
            continue
        obligations.append(
            _build_relation_obligation(
                section_title,
                "length_relation",
                sentence,
                field_names=(lhs, *rhs_fields[:4]),
                primary_field=lhs,
                related_fields=rhs_fields[:4],
                numeric_values=numbers[:6],
                source=source,
            )
        )
    return obligations


def _extract_alignment_obligations(
    section_title: str,
    sentence: str,
    source: str,
) -> List[RelationalObligation]:
    if _is_optional_relation_sentence(sentence):
        return []
    if not re.search(r"\b(?:multiple|modulo|mod|aligned|alignment|boundary|boundaries)\b", sentence, re.IGNORECASE):
        return []
    if not re.search(
        r"\b(?:multiple\s+of|multiple\s+\d|measured\s+in\s+units?\s+of|aligned\s+(?:on|to)|boundary|boundaries|modulo|mod)\b",
        sentence,
        re.IGNORECASE,
    ):
        return []
    if re.search(r"\bmultiple\b", sentence, re.IGNORECASE) and not re.search(
        r"\b(?:multiple\s+of|multiple\s+\d|multiple\s+[A-Za-z]+\s+bits?|multiple\s+[A-Za-z]+\s+bytes?|modulo|mod|aligned|alignment|boundary|boundaries)\b",
        sentence,
        re.IGNORECASE,
    ):
        return []
    fields = _candidate_relation_fields(sentence)
    numbers = [_parse_int(value) for value in _NUM_RE.findall(sentence)]
    numbers = [int(value) for value in numbers if value is not None and int(value) > 1]
    if not fields or not numbers:
        return []
    return [
        _build_relation_obligation(
            section_title,
            "mod_alignment",
            sentence,
            field_names=fields[:4],
            primary_field=fields[0],
            related_fields=fields[1:4],
            numeric_values=numbers[:3],
            source=source,
        )
    ]


def _extract_post_end_zero_obligations(
    section_title: str,
    sentence: str,
    source: str,
) -> List[RelationalObligation]:
    if _is_optional_relation_sentence(sentence):
        return []
    if not re.search(r"\b(?:remaining|remainder|pad|padding|eol|end)\b", sentence, re.IGNORECASE):
        return []
    if not re.search(r"\bzero\b|0x00", sentence, re.IGNORECASE):
        return []
    fields = _candidate_relation_fields(sentence)
    if not fields:
        fields = ("Padding",)
    return [
        _build_relation_obligation(
            section_title,
            "post_end_zero_padding",
            sentence,
            field_names=fields[:4],
            primary_field=fields[0],
            related_fields=fields[1:4],
            numeric_values=(0,),
            source=source,
        )
    ]


def extract_relational_obligations(sections: Sequence[Dict[str, Any]]) -> List[RelationalObligation]:
    """Extract relation obligations from Stage A prose.
    
    Recognized forms include conditionals, mutual exclusion, length formulas,
    modulo and alignment rules, and post-end zero padding."""

    obligations: List[RelationalObligation] = []
    seen: set[str] = set()
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or section.get("section_title") or section.get("number") or "section")
        for obligation in _extract_structured_relation_obligations(section, section_title):
            if obligation.obligation_id in seen or len(obligation.field_names) < 1:
                continue
            seen.add(obligation.obligation_id)
            obligations.append(obligation)
        for source, sentence in _relation_sentences_from_section(section):
            extracted: List[RelationalObligation] = []
            extracted.extend(_extract_assignment_obligations(section_title, sentence, source))
            extracted.extend(_extract_conditional_zero_obligations(section_title, sentence, source))
            extracted.extend(_extract_mutual_exclusion_obligations(section_title, sentence, source))
            extracted.extend(_extract_length_relation_obligations(section_title, sentence, source))
            extracted.extend(_extract_alignment_obligations(section_title, sentence, source))
            extracted.extend(_extract_post_end_zero_obligations(section_title, sentence, source))
            for obligation in extracted:
                if obligation.obligation_id in seen or len(obligation.field_names) < 1:
                    continue
                seen.add(obligation.obligation_id)
                obligations.append(obligation)
    return obligations


def _constraint_texts_for_tree(tree: Dict[str, Any]) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
    texts: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
    nodes = _tree_nodes(tree)
    by_id = {str(node.get("node_id")): node for node in nodes if node.get("node_id") is not None}
    node_labels = {str(node.get("node_id")): _node_label(node) for node in nodes}

    def _node_with_relation_path(node: Dict[str, Any]) -> Dict[str, Any]:
        names = [str(node.get("name") or "")]
        seen: set[str] = set()
        parent_id = node.get("parent_id")
        for _ in range(8):
            key = str(parent_id) if parent_id is not None else ""
            if not key or key in seen or key not in by_id:
                break
            seen.add(key)
            parent = by_id[key]
            names.append(str(parent.get("name") or ""))
            parent_id = parent.get("parent_id")
        contextual = dict(node)
        contextual["_relation_path_name"] = " ".join(name for name in names if name)
        return contextual

    for node in nodes:
        node_id = node.get("node_id")
        label = _node_label(node)
        contextual_node = _node_with_relation_path(node)
        for constraint in node.get("constraints") or []:
            if str(constraint or "").strip():
                texts.append(("node_constraint", str(constraint), contextual_node))
        for key in ("size_bits", "length_expr", "value_expr"):
            value = node.get(key)
            if value is not None and str(value).strip():
                texts.append((key, f"{label} {key} {value}", contextual_node))
    for edge in _tree_edges(tree):
        formula = edge.get("formula") or edge.get("condition") or edge.get("expr")
        if formula is not None and str(formula).strip():
            src_label = node_labels.get(str(edge.get("src")), str(edge.get("src") or ""))
            dst_label = node_labels.get(str(edge.get("dst")), str(edge.get("dst") or ""))
            texts.append(("edge_formula", f"{src_label} -> {dst_label} {edge.get('rel') or ''} {formula}", None))
    return texts


def _ids_for_relation_field(nodes: Sequence[Dict[str, Any]], field_name: str) -> set[str]:
    return {str(node.get("node_id")) for node in _match_field_nodes(nodes, field_name) if node.get("node_id") is not None}


def _relation_context_tokens_for_obligation(obligation: RelationalObligation) -> set[str]:
    text = f"{obligation.section_title or ''} {obligation.relation_text or ''}"
    tokens = _field_tokens(text)
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


def _ids_for_relation_field_with_context(
    nodes: Sequence[Dict[str, Any]],
    field_name: str,
    obligation: RelationalObligation,
) -> set[str]:
    direct = _ids_for_relation_field(nodes, field_name)
    if direct:
        return direct
    field_tokens = _field_tokens(field_name)
    context_tokens = _relation_context_tokens_for_obligation(obligation)
    if not field_tokens or not context_tokens:
        return set()
    if len(field_tokens) == 1:
        flag_matches: set[str] = set()
        for node in nodes:
            node_id = node.get("node_id")
            if node_id is None:
                continue
            node_tokens = _field_tokens(node.get("name"))
            if not node_tokens or "flag" not in node_tokens:
                continue
            if field_tokens & node_tokens:
                flag_matches.add(str(node_id))
                continue
            context_specific = {
                token
                for token in context_tokens
                if token not in _FIELD_MATCH_QUALIFIER_TOKENS and len(token) > 2
            }
            if context_specific & node_tokens:
                flag_matches.add(str(node_id))
        if flag_matches:
            return flag_matches
    scored: List[Tuple[int, int, str]] = []
    for node in nodes:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        node_tokens = _field_tokens(node.get("name"))
        field_overlap = field_tokens & node_tokens
        context_overlap = context_tokens & node_tokens
        if not field_overlap or not context_overlap:
            continue
        score = len(field_overlap) * 4 + len(context_overlap) * 3
        scored.append((score, -len(node_tokens), str(node_id)))
    if not scored:
        return set()
    scored.sort(reverse=True)
    best = scored[0][0]
    return {node_id for score, _specificity, node_id in scored if score == best}


def _constraint_mentions_field(text: str, ids: set[str], field_name: str) -> bool:
    if ids:
        refs = {str(ref).strip() for ref in _VAL_REF_RE.findall(text)}
        if refs & ids:
            return True
    wanted = _field_tokens(field_name)
    if not wanted:
        return False
    tokens = _field_tokens(text)
    if wanted <= tokens:
        return True
    wanted_norm = _norm_text(field_name)
    return bool(wanted_norm and wanted_norm in _norm_text(text))


def _constraint_mentions_number(text: str, value: int) -> bool:
    numbers = {_parse_int(match.group(0)) for match in _NUM_RE.finditer(text)}
    return value in {num for num in numbers if num is not None}


def _node_matches_relation_context(node: Optional[Dict[str, Any]], obligation: RelationalObligation) -> bool:
    if node is None:
        return False
    node_tokens = _field_tokens(node.get("name"))
    if not node_tokens:
        return False
    context_tokens = _relation_context_tokens_for_obligation(obligation)
    if not context_tokens:
        return True
    node_overlap = node_tokens & context_tokens
    if not node_overlap:
        return False
    weak_context_tokens = {
        "input",
        "inputs",
        "output",
        "outputs",
        "register",
        "registers",
        "status",
        "statuses",
        "value",
        "values",
    }
    if node_overlap - weak_context_tokens:
        return True
    path_tokens = node_tokens | _field_tokens(node.get("_relation_path_name"))
    return bool((path_tokens & context_tokens) - weak_context_tokens)


def _is_packed_bits_zero_padding_constraint(text: str) -> bool:
    lower = text.lower()
    if "packed_bits" not in lower:
        return False
    if not re.search(r"\b(?:pad|padding|zero_pad|unused)\b", lower):
        return False
    return bool(re.search(r"(?:pad|padding|zero_pad|unused)\s*=\s*zero|zero", lower))


def _normalize_relation_ir_kind(kind: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(kind or "").lower()).strip("_")
    aliases = {
        "alignment": "mod_alignment",
        "alignment_modulo": "mod_alignment",
        "modulo_alignment": "mod_alignment",
        "multiple_of": "mod_alignment",
        "length": "length_relation",
        "length_binding": "length_relation",
        "length_count_relation": "length_relation",
        "count_relation": "length_relation",
        "size_relation": "length_relation",
        "conditional": "conditional_value",
        "conditional_constraint": "conditional_value",
        "conditional_relation": "conditional_value",
        "implication": "conditional_value",
        "if_then": "conditional_value",
        "must_be_zero_when": "conditional_zero",
        "zero_when": "conditional_zero",
        "not_both": "mutual_exclusion",
        "mutual_exclusive": "mutual_exclusion",
        "exclusive_flags": "mutual_exclusion",
        "padding_zero": "post_end_zero_padding",
        "zero_padding": "post_end_zero_padding",
        "post_end_padding_zero": "post_end_zero_padding",
        "padding_zero_after_terminator": "post_end_zero_padding",
        "packed_bits_padding": "post_end_zero_padding",
        "packed_bit_padding": "post_end_zero_padding",
    }
    return aliases.get(normalized, normalized)


def _relation_ir_kind_matches(ir_kind: str, obligation_kind: str) -> bool:
    ir_norm = _normalize_relation_ir_kind(ir_kind)
    obligation_norm = _normalize_relation_ir_kind(obligation_kind)
    if ir_norm == obligation_norm:
        return True
    if obligation_norm in {"conditional_zero", "conditional_value"}:
        return ir_norm in {"conditional_zero", "conditional_value"}
    return False


def _constraint_ir_covers_relation(
    text: str,
    obligation: RelationalObligation,
    ids_by_field: Dict[str, set[str]],
    node: Optional[Dict[str, Any]],
) -> Optional[bool]:
    ir = parse_constraint_ir(text)
    if ir is None:
        return None
    if not _relation_ir_kind_matches(str(ir.get("kind") or ""), obligation.kind):
        return False

    blob = constraint_ir_blob(ir)
    ir_ids = constraint_ir_field_ids(ir)
    fields = list(obligation.field_names)
    mentioned = [
        field
        for field in fields
        if (ids_by_field.get(field, set()) & ir_ids)
        or _constraint_mentions_field(blob, ids_by_field.get(field, set()), field)
    ]

    if obligation.kind in {"conditional_zero", "conditional_value"}:
        if len(mentioned) < min(2, len(fields)):
            return False
        if obligation.kind == "conditional_zero" and not _constraint_mentions_number(blob, 0):
            return False
        if obligation.kind == "conditional_value" and obligation.numeric_values and not all(
            _constraint_mentions_number(blob, number) for number in obligation.numeric_values
        ):
            return False
        return True

    if obligation.kind == "mutual_exclusion":
        return len(mentioned) >= min(2, len(fields))

    if obligation.kind == "length_relation":
        if obligation.primary_field and obligation.primary_field not in mentioned:
            return False
        related = [field for field in obligation.related_fields if field in mentioned]
        has_number = not obligation.numeric_values or any(
            _constraint_mentions_number(blob, number) for number in obligation.numeric_values
        )
        return has_number and (bool(related) or bool(obligation.numeric_values))

    if obligation.kind == "count_relation":
        if obligation.primary_field and obligation.primary_field not in mentioned:
            return False
        related = [field for field in obligation.related_fields if field in mentioned]
        lower = blob.lower()
        has_count_semantics = any(
            token in lower
            for token in ("count_relation", "repeat_count", "count", "number", "entry", "record", "item")
        )
        return has_count_semantics and bool(related)

    if obligation.kind == "mod_alignment":
        if not mentioned:
            return False
        return not obligation.numeric_values or any(
            _constraint_mentions_number(blob, number) for number in obligation.numeric_values
        )

    if obligation.kind == "post_end_zero_padding":
        if node is not None:
            return _node_matches_relation_context(node, obligation)
        return bool(
            "zero" in blob.lower()
            and any(token in blob.lower() for token in ("padding", "pad", "eol", "end", "remaining"))
        )

    return False


def _relation_formula_covers(
    text: str,
    obligation: RelationalObligation,
    ids_by_field: Dict[str, set[str]],
    node: Optional[Dict[str, Any]] = None,
) -> bool:
    ir_covered = _constraint_ir_covers_relation(text, obligation, ids_by_field, node)
    if ir_covered is not None:
        return ir_covered

    lower = text.lower()
    fields = list(obligation.field_names)
    mentioned = [
        field
        for field in fields
        if _constraint_mentions_field(text, ids_by_field.get(field, set()), field)
    ]
    if obligation.kind in {"conditional_zero", "conditional_value"}:
        if len(mentioned) < min(2, len(fields)):
            return False
        if obligation.kind == "conditional_zero":
            if not _constraint_mentions_number(text, 0):
                return False
        elif obligation.numeric_values and not all(
            _constraint_mentions_number(text, number) for number in obligation.numeric_values
        ):
            return False
        return bool(re.search(r"==|=|<=|>=|!=", text))
    if obligation.kind == "mutual_exclusion":
        if len(mentioned) < min(2, len(fields)):
            return False
        return bool(
            "not" in lower
            or "!" in lower
            or "<= 1" in lower
            or "<=1" in lower
            or re.search(r"\+\s*val\s*\([^)]*\)\s*<=\s*1", lower)
        )
    if obligation.kind == "length_relation":
        if obligation.primary_field and obligation.primary_field not in mentioned:
            return False
        related = [field for field in obligation.related_fields if field in mentioned]
        has_relation_operator = bool(re.search(r"==|=|\+|-|\*|/", text))
        has_number = not obligation.numeric_values or any(
            _constraint_mentions_number(text, number) for number in obligation.numeric_values
        )
        return has_relation_operator and has_number and (
            bool(related)
            or bool(obligation.numeric_values)
            or "length_of" in lower
            or "size_bits" in lower
        )
    if obligation.kind == "count_relation":
        if obligation.primary_field and obligation.primary_field not in mentioned:
            return False
        related = [field for field in obligation.related_fields if field in mentioned]
        has_count_relation = any(
            token in lower
            for token in ("repeat_count", "count_relation", "count", "number of", "entries", "records", "items")
        )
        return has_count_relation and bool(related)
    if obligation.kind == "mod_alignment":
        if not mentioned:
            return False
        has_mod = "%" in text or "mod" in lower or "multiple" in lower or "align" in lower or "boundary" in lower
        if obligation.numeric_values:
            for number in obligation.numeric_values:
                if re.search(rf"(?:\*\s*{number}\b|\b{number}\s*\*)", text):
                    has_mod = True
                    break
        has_number = any(_constraint_mentions_number(text, number) for number in obligation.numeric_values)
        return has_mod and has_number
    if obligation.kind == "post_end_zero_padding":
        if _is_packed_bits_zero_padding_constraint(text):
            return _node_matches_relation_context(node, obligation)
        if not ("zero" in lower or "0x00" in lower or re.search(r"(^|[^0-9])0([^0-9]|$)", lower)):
            return False
        if any(token in lower for token in ("padding", "pad", "eol", "end", "remaining", "remainder")):
            if node is not None:
                return _node_matches_relation_context(node, obligation)
            return True
        return bool(mentioned)
    return False


def _relational_obligation_covered(tree: Dict[str, Any], obligation: RelationalObligation) -> Tuple[bool, str]:
    nodes = _tree_nodes(tree)
    ids_by_field = {
        field: _ids_for_relation_field_with_context(nodes, field, obligation)
        for field in obligation.field_names
    }
    missing_fields = [field for field, ids in ids_by_field.items() if not ids]
    if obligation.kind == "length_relation" and obligation.primary_field:
        primary_ids = ids_by_field.get(obligation.primary_field, set())
        if not primary_ids:
            return False, f"no matching tree node(s) for primary relational field: {obligation.primary_field!r}"
        missing_fields = [field for field in missing_fields if field == obligation.primary_field]
    if missing_fields and obligation.kind not in {"post_end_zero_padding"}:
        return False, f"no matching tree node(s) for relational field(s): {missing_fields}"

    for kind, text, node in _constraint_texts_for_tree(tree):
        if _relation_formula_covers(text, obligation, ids_by_field, node):
            target = _node_label(node) if node else kind
            return True, f"covered by {target} {kind}: {text!r}"
    return False, f"no tree formula/constraint covers documented relation: {obligation.relation_text!r}"


def find_uncovered_relational_obligations(
    tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    *,
    max_issues: Optional[int] = None,
) -> List[Tuple[RelationalObligation, str]]:
    uncovered: List[Tuple[RelationalObligation, str]] = []
    for obligation in extract_relational_obligations(sections):
        covered, evidence = _relational_obligation_covered(tree, obligation)
        if covered:
            continue
        uncovered.append((obligation, evidence))
        if max_issues is not None and len(uncovered) >= max_issues:
            break
    return uncovered


def relational_obligation_coverage_counts(
    tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    obligations = extract_relational_obligations(sections)
    uncovered = 0
    for obligation in obligations:
        covered, _evidence = _relational_obligation_covered(tree, obligation)
        if not covered:
            uncovered += 1
    return len(obligations), uncovered


def _relational_safe_repair_guidance(
    tree: Dict[str, Any],
    obligation: RelationalObligation,
) -> Tuple[str, Dict[str, Any]]:
    nodes = _tree_nodes(tree)
    nodes_by_id = {str(node.get("node_id")): node for node in nodes if node.get("node_id") is not None}
    raw_ids_by_field = {
        field: sorted(_ids_for_relation_field_with_context(nodes, field, obligation), key=str)
        for field in obligation.field_names
    }
    ids_by_field: Dict[str, List[str]] = {}
    structural_ids_by_field: Dict[str, List[str]] = {}
    for field, node_ids in raw_ids_by_field.items():
        safe_ids: List[str] = []
        structural_ids: List[str] = []
        for node_id in node_ids:
            node = nodes_by_id.get(str(node_id))
            if node is not None and _node_is_scalar_val_reference_target(node):
                safe_ids.append(str(node_id))
            elif str(node_id).strip():
                structural_ids.append(str(node_id))
        ids_by_field[field] = safe_ids
        structural_ids_by_field[field] = structural_ids
    flat_field_ids = [
        node_id
        for field in obligation.field_names
        for node_id in ids_by_field.get(field, [])
        if str(node_id).strip()
    ]
    append_constraint_ir = constraint_ir_to_text(
        kind=obligation.kind,
        expr=obligation.relation_text,
        field_ids=flat_field_ids,
        field_names=obligation.field_names,
        source=obligation.relation_text,
    )
    guidance = (
        "RELATIONAL REPAIR: represent this documented prose relation explicitly with `constraint_ir:` on the "
        "smallest relevant node/variant, and add an executable `formula:` or length edge only when it is safe. "
        "Use existing node ids when possible, preserve "
        "existing constraints, and create missing structure first if one of the relation fields is absent. "
        "Keep the repair generic: do not inject protocol-specific hidden constants unless they appear in the "
        "quoted documentation evidence. `val(ID)` may reference only scalar parsed-value nodes "
        "(selector, type, length, checksum, or fixed scalar field). Do not write `val(ID)` for structural "
        "payload/body/container/data nodes; bind formulas to the selector/length/scalar child instead, or keep "
        "the relation as `constraint_ir:` without an executable `formula:` until scalar children exist. "
        "Suggested IR constraint: "
        f"{append_constraint_ir!r}."
    )
    if obligation.kind == "post_end_zero_padding":
        guidance += (
            " For packed bit arrays with final-byte padding, add a node-level schema constraint on the bit-array field: "
            "`packed_bits: unit_bits=1; order=lsb_first|msb_first|unknown; pad=zero; count=val(ID)|external`. "
            "Use count=external when the bit count is only known from another packet/message context."
        )
    elif obligation.kind == "mod_alignment":
        guidance += " For alignment rules, add an explicit modulo/multiple constraint on the documented length/offset field or variant size."
    elif obligation.kind == "mutual_exclusion":
        guidance += " For mutual exclusion, add one formula equivalent to not(A && B) or A + B <= 1."
    elif obligation.kind in {"conditional_zero", "conditional_value"}:
        guidance += " For conditional rules, guard the consequent with the documented trigger instead of making the target constraint unconditional."
    elif obligation.kind == "length_relation":
        guidance += " For length relations, bind the size/value expression to the referenced length/count fields rather than using a free variable."
    elif obligation.kind == "count_relation":
        guidance += " For count relations, add a repeat_count edge from the documented count field to the repeated entry/container."
    return guidance, {
        "safe_repairs": [
            {
                "kind": "relational_constraint_ir",
                "obligation_kind": obligation.kind,
                "field_node_ids": ids_by_field,
                "structural_field_node_ids": structural_ids_by_field,
                "source_relation": obligation.relation_text,
                "numeric_values": list(obligation.numeric_values),
                "append_constraint_ir": append_constraint_ir,
            }
        ],
        "field_node_ids": ids_by_field,
        "structural_field_node_ids": structural_ids_by_field,
    }


def _claim_covered(tree: Dict[str, Any], claim: ConstraintClaim) -> Tuple[bool, str]:
    nodes = _tree_nodes(tree)
    edges = _tree_edges(tree)
    nodes_by_id = {str(n.get("node_id")): n for n in nodes}
    children_by_parent: Dict[str, List[Any]] = {}
    for node in nodes:
        parent = node.get("parent_id")
        if parent is not None:
            children_by_parent.setdefault(str(parent), []).append(node.get("node_id"))
        for child in node.get("children_ids") or []:
            children_by_parent.setdefault(str(node.get("node_id")), []).append(child)

    target_nodes = _match_field_nodes(nodes, claim.field_name)

    if not claim.guard_field:
        if not target_nodes:
            return True, f"ignored: target field {claim.field_name!r} is not present in the current tree"
        for node in target_nodes:
            if _packed_or_binary_node_satisfies_claim(node, claim):
                return True, f"covered by {_node_label(node)} packed/binary representation"
            if _selector_node_covers_domain_claim(node, claim):
                return True, f"covered by {_node_label(node)} selector enum domain"
            for constraint in node.get("constraints") or []:
                if _constraint_satisfies_claim(constraint, claim):
                    return True, f"covered by {_node_label(node)} constraint {constraint!r}"
        return False, f"no matching node constraint for {claim.short_expected()}"

    # A shared/header target constraint applies in every guarded branch.  This
    # covers common layouts where a field such as Code or Checksum is parsed
    # once in the header, while per-message body variants are selected by Type.
    for node in target_nodes:
        if _packed_or_binary_node_satisfies_claim(node, claim):
            return True, f"covered by shared {_node_label(node)} packed/binary representation"
        if _selector_node_covers_domain_claim(node, claim):
            return True, f"covered by shared {_node_label(node)} selector enum domain"
        for constraint in node.get("constraints") or []:
            if _constraint_satisfies_claim(constraint, claim):
                return True, f"covered by shared {_node_label(node)} constraint {constraint!r}"

    candidate_variant_ids, missing_guard_values, guard_nodes = _condition_variants_for_guard(
        nodes=nodes,
        edges=edges,
        nodes_by_id=nodes_by_id,
        guard_field=claim.guard_field,
        guard_values=claim.guard_values,
        message_type=claim.message_type,
        allow_structural_selector_fallback=claim.guard_source == "table",
    )
    if not guard_nodes:
        return True, f"ignored: guard field {claim.guard_field!r} is not present in the current tree"
    if not candidate_variant_ids:
        return False, f"no condition_on branch for {claim.guard_field} values {list(claim.guard_values)}"
    if missing_guard_values:
        return False, f"missing condition_on branch for {claim.guard_field} values {sorted(missing_guard_values)}"

    guard_ids = {str(node.get("node_id")) for node in guard_nodes if node.get("node_id") is not None}
    for variant_id in candidate_variant_ids:
        variant = nodes_by_id.get(variant_id)
        search_ids = {variant_id} | _descendant_ids(variant_id, children_by_parent)
        scoped_nodes = [nodes_by_id[nid] for nid in search_ids if nid in nodes_by_id]
        scoped_targets = _match_field_nodes(scoped_nodes, claim.field_name)
        scoped_target_ids = {str(node.get("node_id")) for node in scoped_targets if node.get("node_id") is not None}
        for node in scoped_targets:
            if _packed_or_binary_node_satisfies_claim(node, claim):
                return True, f"covered by scoped {_node_label(node)} packed/binary representation"
            if _selector_node_covers_domain_claim(node, claim):
                return True, f"covered by scoped {_node_label(node)} selector enum domain"
            for constraint in node.get("constraints") or []:
                if _constraint_satisfies_claim(
                    constraint,
                    claim,
                    target_ids=scoped_target_ids,
                    guard_ids=guard_ids,
                ):
                    return True, f"covered by scoped {_node_label(node)} constraint {constraint!r}"
        if _is_length_field_name(claim.field_name):
            # TLV option tables often document row-specific long names such as
            # "Window Scale Option Length", while the tree uses a shorter
            # branch-local length field such as "WS_Option_Length".  Once the
            # selector branch is proven, any scoped length-like node with the
            # documented enum/range constraint is sufficient evidence.
            for node in scoped_nodes:
                if not _node_is_tlv_length(node):
                    continue
                for constraint in node.get("constraints") or []:
                    if _constraint_satisfies_claim(constraint, claim):
                        return True, f"covered by scoped TLV length {_node_label(node)} constraint {constraint!r}"
        if variant:
            # Variant-level constraints often reference selector/header fields
            # outside the variant subtree, e.g. expression: val(Code) == 0.
            # Use the globally matched target ids to keep those expressions
            # target-specific without requiring duplicated per-variant fields.
            target_ids = {str(node.get("node_id")) for node in (scoped_targets or target_nodes)}
            for constraint in variant.get("constraints") or []:
                if _constraint_satisfies_claim(
                    constraint,
                    claim,
                    target_ids=target_ids,
                    guard_ids=guard_ids,
                    allow_bare_constraint=False,
                ):
                    return True, f"covered by variant {_node_label(variant)} constraint {constraint!r}"
    return (
        False,
        f"condition_on branch exists, but no scoped constraint covers {claim.short_expected()}",
    )


def _safe_repair_guidance(tree: Dict[str, Any], claim: ConstraintClaim) -> Tuple[str, Dict[str, Any]]:
    nodes = _tree_nodes(tree)
    edges = _tree_edges(tree)
    nodes_by_id = {str(n.get("node_id")): n for n in nodes}
    children_by_parent: Dict[str, List[Any]] = {}
    for node in nodes:
        parent = node.get("parent_id")
        if parent is not None:
            children_by_parent.setdefault(str(parent), []).append(node.get("node_id"))
        for child in node.get("children_ids") or []:
            children_by_parent.setdefault(str(node.get("node_id")), []).append(child)

    target_nodes = _match_field_nodes(nodes, claim.field_name)
    target_ids = [node.get("node_id") for node in target_nodes if node.get("node_id") is not None]
    direct_constraint = _direct_constraint_for_claim(claim)

    safe_repairs: List[Dict[str, Any]] = []
    notes: List[str] = []
    forbidden_node_ids: List[Any] = []

    if not claim.guard_field:
        scoped_context = bool(
            claim.message_type
            or _SCOPED_CONTEXT_RE.search(
                f"{claim.section_title} {claim.format_name} {claim.guard_source} {claim.source}"
            )
        )
        if scoped_context:
            notes.append(
                "SCOPED REPAIR: this claim appears context-scoped (for example request/response/client/server/conditional text). "
                "Do NOT append it as an unconditional constraint to a shared field in a bidirectional tree. "
                "Add it only under a matching message/variant scope; if no such scope exists, leave this doc claim unresolved instead of making traffic-invalid global constraints."
            )
            return " ".join(notes), {
                "safe_repairs": safe_repairs,
                "target_node_ids": target_ids,
                "forbidden_node_ids": forbidden_node_ids,
                "requires_scope": True,
            }
        for node in target_nodes[:4]:
            safe_repairs.append(
                {
                    "kind": "target_node_constraint",
                    "node_id": node.get("node_id"),
                    "node_name": node.get("name"),
                    "append_constraint": direct_constraint,
                    "preserve_existing_constraints": list(node.get("constraints") or []),
                }
            )
        if safe_repairs:
            notes.append(
                "SAFE REPAIR: append the direct constraint to the matching target field node; "
                "preserve all existing constraints on that node."
            )
        return " ".join(notes), {
            "safe_repairs": safe_repairs,
            "target_node_ids": target_ids,
            "forbidden_node_ids": forbidden_node_ids,
        }

    candidate_variant_ids, _missing_guard_values, guard_nodes = _condition_variants_for_guard(
        nodes=nodes,
        edges=edges,
        nodes_by_id=nodes_by_id,
        guard_field=claim.guard_field,
        guard_values=claim.guard_values,
        message_type=claim.message_type,
        allow_structural_selector_fallback=claim.guard_source == "table",
    )

    if _is_length_field_name(claim.field_name) and claim.guard_field and claim.guard_values:
        expected = _direct_constraint_for_claim(claim) or claim.short_expected()
        notes.append(
            "TLV-SAFE REPAIR: when this length constraint belongs to a selector/length/data table, "
            "prefer a conditional formula on the generic length field using the selector and length node ids "
            f"(for example: {claim.guard_field} in {list(claim.guard_values)} AND {claim.field_name} satisfies {expected}). "
            "Do not create exact per-value option variants solely to attach length constants; keep row-specific payload "
            "inside the generic data/payload field unless the documentation gives separate constrained subfields."
        )

    for variant_id in sorted(candidate_variant_ids):
        variant = nodes_by_id.get(variant_id)
        if not variant:
            continue
        search_ids = {variant_id} | _descendant_ids(variant_id, children_by_parent)
        scoped_nodes = [nodes_by_id[nid] for nid in search_ids if nid in nodes_by_id]
        scoped_targets = _match_field_nodes(scoped_nodes, claim.field_name)
        if scoped_targets:
            for node in scoped_targets[:3]:
                if _variant_guard_is_exact_for_claim(variant_id, edges, guard_nodes, claim):
                    safe_repairs.append(
                        {
                            "kind": "scoped_target_node_constraint",
                            "variant_id": variant.get("node_id"),
                            "variant_name": variant.get("name"),
                            "node_id": node.get("node_id"),
                            "node_name": node.get("name"),
                            "append_constraint": direct_constraint,
                            "preserve_existing_constraints": list(node.get("constraints") or []),
                        }
                    )
                    continue
                guard_id = _first_guard_id_for_claim(guard_nodes)
                guarded_formula = (
                    _guarded_formula_constraint_for_claim(claim, node.get("node_id"), guard_id)
                    if guard_id is not None
                    else None
                )
                if guarded_formula:
                    safe_repairs.append(
                        {
                            "kind": "guarded_variant_formula",
                            "variant_id": variant.get("node_id"),
                            "variant_name": variant.get("name"),
                            "target_node_id": node.get("node_id"),
                            "target_node_name": node.get("name"),
                            "guard_node_id": guard_id,
                            "guard_values": list(claim.guard_values),
                            "append_constraint": guarded_formula,
                            "preserve_existing_constraints": list(variant.get("constraints") or []),
                        }
                    )
        elif target_nodes:
            target = target_nodes[0]
            guard_id = _first_guard_id_for_claim(guard_nodes)
            if _variant_guard_is_exact_for_claim(variant_id, edges, guard_nodes, claim):
                formula_constraint = _formula_constraint_for_claim(claim, target.get("node_id"))
            else:
                formula_constraint = (
                    _guarded_formula_constraint_for_claim(claim, target.get("node_id"), guard_id)
                    if guard_id is not None
                    else _formula_constraint_for_claim(claim, target.get("node_id"))
                )
            safe_repairs.append(
                {
                    "kind": "variant_formula",
                    "variant_id": variant.get("node_id"),
                    "variant_name": variant.get("name"),
                    "target_node_id": target.get("node_id"),
                    "target_node_name": target.get("name"),
                    "guard_node_id": guard_id,
                    "guard_values": list(claim.guard_values),
                    "append_constraint": formula_constraint,
                    "preserve_existing_constraints": list(variant.get("constraints") or []),
                }
            )
            forbidden_node_ids.append(target.get("node_id"))

    if safe_repairs:
        first = safe_repairs[0]
        if first.get("kind") in {"variant_formula", "guarded_variant_formula"}:
            notes.append(
                "SAFE REPAIR: append the exact target-specific formula to the listed variant constraints, "
                f"e.g. add {first.get('append_constraint')!r} to variant "
                f"{first.get('variant_name')}(ID:{first.get('variant_id')}); preserve existing variant constraints."
            )
            if forbidden_node_ids:
                notes.append(
                    "Do NOT widen/delete the shared target field constraints on "
                    + ", ".join(str(node_id) for node_id in sorted(set(forbidden_node_ids), key=str))
                    + "; if a shared constraint is too narrow, replace it with equivalent scoped formulas for the branches where it still applies."
                )
        else:
            notes.append(
                "SAFE REPAIR: append the direct enum/range constraint to the scoped child field listed in safe_repairs; "
                "preserve existing node constraints."
            )
        notes.append("Do NOT add condition_on edges to leaf fields; selector routing must target variants only.")

    return " ".join(notes), {
        "safe_repairs": safe_repairs[:8],
        "target_node_ids": target_ids,
        "guard_node_ids": [node.get("node_id") for node in guard_nodes if node.get("node_id") is not None],
        "candidate_variant_ids": [int(v) if str(v).isdigit() else v for v in sorted(candidate_variant_ids)],
        "forbidden_node_ids": sorted(set(forbidden_node_ids), key=str),
        "tlv_length_constraint_hint": (
            bool(_is_length_field_name(claim.field_name) and claim.guard_field and claim.guard_values)
        ),
    }


def _select_claim_window(
    claims: Sequence[ConstraintClaim],
    *,
    max_claims: Optional[int] = None,
    claim_ids: Optional[Iterable[str]] = None,
    claim_offset: int = 0,
) -> List[ConstraintClaim]:
    if claim_ids is not None:
        wanted = {str(claim_id) for claim_id in claim_ids}
        return [claim for claim in claims if claim.claim_id in wanted]
    if max_claims is None:
        return list(claims)
    start = max(0, int(claim_offset or 0))
    limit = max(0, int(max_claims))
    return list(claims[start : start + limit])


def _find_uncovered_in_claims(
    tree: Dict[str, Any],
    claims: Sequence[ConstraintClaim],
) -> List[Tuple[ConstraintClaim, str]]:
    uncovered: List[Tuple[ConstraintClaim, str]] = []
    for claim in claims:
        covered, evidence = _claim_covered(tree, claim)
        if not covered:
            uncovered.append((claim, evidence))
    return uncovered


def find_uncovered_doc_constraints(
    tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    *,
    max_claims: Optional[int] = None,
    claim_ids: Optional[Iterable[str]] = None,
    claim_offset: int = 0,
) -> List[Tuple[ConstraintClaim, str]]:
    claims = extract_constraint_claims(sections)
    selected = _select_claim_window(
        claims,
        max_claims=max_claims,
        claim_ids=claim_ids,
        claim_offset=claim_offset,
    )
    return _find_uncovered_in_claims(tree, selected)


def _norm_constraint(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _constraint_preserved_by_candidate(baseline_constraint: Any, candidate_constraints: Sequence[Any]) -> bool:
    baseline_norm = _norm_constraint(baseline_constraint)
    candidate_norms = {_norm_constraint(value) for value in candidate_constraints}
    if baseline_norm in candidate_norms:
        return True

    baseline_enum = set(_enum_values_from_constraint(baseline_constraint))
    baseline_range = _range_from_constraint(baseline_constraint)
    for candidate in candidate_constraints:
        candidate_enum = set(_enum_values_from_constraint(candidate))
        candidate_range = _range_from_constraint(candidate)
        if baseline_enum:
            if candidate_enum and candidate_enum.issubset(baseline_enum):
                return True
            if candidate_range and all(candidate_range[0] <= value <= candidate_range[1] for value in baseline_enum):
                # A range is only a preservation of enum evidence when it
                # collapses to exactly the same singleton.
                if len(baseline_enum) == 1 and candidate_range[0] == candidate_range[1] == next(iter(baseline_enum)):
                    return True
        if baseline_range:
            blo, bhi = baseline_range
            if candidate_range:
                clo, chi = candidate_range
                if clo >= blo and chi <= bhi:
                    return True
            if candidate_enum and all(blo <= value <= bhi for value in candidate_enum):
                return True
    return False


def _constraint_replacement_is_doc_supported(
    node: Dict[str, Any],
    baseline_constraint: Any,
    candidate_constraints: Sequence[Any],
    claims: Sequence[ConstraintClaim],
) -> bool:
    """Allow replacing stale node constraints when the replacement matches docs.

    The preservation gate should reject silent deletion/broadening, but it
    should not prevent the repair loop from correcting an earlier wrong enum
    or range when the documentation gives exact evidence for the same field.
    """

    if not candidate_constraints:
        return False
    for claim in claims:
        if not _match_field_nodes([node], claim.field_name):
            continue
        baseline_satisfies = _constraint_satisfies_claim(baseline_constraint, claim)
        candidate_satisfies = any(
            _constraint_satisfies_claim(candidate_constraint, claim)
            for candidate_constraint in candidate_constraints
        )
        if candidate_satisfies and not baseline_satisfies:
            return True
    return False


def find_node_constraint_preservation_violations(
    baseline_tree: Dict[str, Any],
    candidate_tree: Dict[str, Any],
    *,
    max_issues: Optional[int] = None,
    claims: Sequence[ConstraintClaim] = (),
) -> List[Dict[str, Any]]:
    """Find generic node-level constraint removals or broadening regressions."""

    baseline_nodes = _tree_nodes(baseline_tree)
    candidate_nodes = _tree_nodes(candidate_tree)
    candidate_by_id = {str(node.get("node_id")): node for node in candidate_nodes}
    violations: List[Dict[str, Any]] = []
    for node in baseline_nodes:
        node_id = node.get("node_id")
        constraints = [value for value in (node.get("constraints") or []) if str(value or "").strip()]
        if node_id is None or not constraints:
            continue
        candidate = candidate_by_id.get(str(node_id))
        candidate_constraints = list(candidate.get("constraints") or []) if isinstance(candidate, dict) else []
        for constraint in constraints:
            if _constraint_preserved_by_candidate(constraint, candidate_constraints):
                continue
            if _constraint_replacement_is_doc_supported(node, constraint, candidate_constraints, claims):
                continue
            payload = {
                "node_id": node_id,
                "node_name": node.get("name"),
                "constraint": str(constraint),
            }
            violations.append(
                {
                    "id": "node_constraint_preservation_" + _stable_id(payload),
                    "node_id": node_id,
                    "node_name": node.get("name"),
                    "baseline_constraint": str(constraint),
                    "candidate_constraints": [str(value) for value in candidate_constraints],
                }
            )
            if max_issues is not None and len(violations) >= max_issues:
                return violations
    return violations


def _structure_repair_priority(issue: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
    """Sort doc-structure repair prompts from smallest useful variant bodies first.

    Large option/variant tables can produce many missing guarded variants.  A
    zero-length or opaque payload row is usually less useful to repair first
    than a row with concrete documented body fields or direct constraints.
    This ordering stays protocol-agnostic: it looks only at extracted table
    structure, not protocol names or values.
    """

    requirement = issue.get("requirement")
    expected_fields = [str(value or "") for value in issue.get("expected_body_fields") or []]
    expected_norms = {_norm_text(value) for value in expected_fields}
    zero_or_opaque_only = bool(expected_fields) and expected_norms.issubset(
        {"payload", "option_payload", "data", "body", "content"}
    )
    if getattr(requirement, "size_bits", None) not in {0, "0", None}:
        zero_or_opaque_only = False
    has_direct_constraint = bool(getattr(requirement, "direct_constraint", None))
    kind_order = {
        "missing_repeated_format": 0,
        "missing_tlv_abstraction": 0,
        "missing_selector_variants": 1,
        "opaque_variant_expansion": 2,
        "missing_guarded_variant": 3,
        "missing_scoped_field": 4,
        "weak_name_alignment": 5,
    }
    kind_rank = kind_order.get(str(issue.get("kind") or ""), 5)
    opaque_rank = 1 if zero_or_opaque_only else 0
    constraint_rank = 0 if has_direct_constraint else 1
    # Prefer a small concrete batch over a giant variant body, but keep
    # variants with no fields last only via opaque_rank above.
    field_count = len(expected_fields)
    label = ":".join(
        [
            str(getattr(requirement, "section_title", "")),
            str(getattr(requirement, "format_name", "")),
            str(getattr(requirement, "guard_values", "")),
        ]
    )
    return (kind_rank, opaque_rank, constraint_rank, field_count, label)


def augment_report_with_doc_constraint_coverage(
    report: ValidationReport,
    tree: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    *,
    max_issues: int = 24,
    claim_ids: Optional[Iterable[str]] = None,
    claim_offset: int = 0,
    baseline_tree: Optional[Dict[str, Any]] = None,
    max_structure_issues: Optional[int] = None,
    include_weak_name_alignment: bool = False,
    focus_kind: Optional[str] = None,
) -> ValidationReport:
    """Return a report with protocol-agnostic doc-constraint coverage issues."""

    def _csv_env(name: str) -> set[str]:
        raw = os.getenv(name, "")
        values: set[str] = set()
        for piece in str(raw).replace(";", ",").split(","):
            value = piece.strip().lower()
            if value:
                values.add(value)
        return values

    def _claim_allowed_by_filter(claim: ConstraintClaim, allowed: set[str]) -> bool:
        if not allowed:
            return True
        kind = str(getattr(claim, "kind", "") or "").strip().lower()
        if kind in allowed:
            return True
        if "value_domain" in allowed and kind in {"enum", "range"}:
            return True
        return False

    focus_raw = (focus_kind or os.getenv("STEP2_DOC_COVERAGE_FOCUS_KIND") or "all").strip().lower()
    focus_aliases = {
        "constraints": "constraint",
        "doc_constraint": "constraint",
        "doc_constraints": "constraint",
        "relations": "relational",
        "relation": "relational",
        "relational_obligation": "relational",
        "relational_obligations": "relational",
        "relation_constraints": "relation_constraint",
        "relations_constraints": "relation_constraint",
        "constraint_relation": "relation_constraint",
        "constraint_relations": "relation_constraint",
        "non_structure": "relation_constraint",
        "non_structural": "relation_constraint",
        "structures": "structure",
        "structural": "structure",
        "structure_alignment": "structure",
    }
    focus = focus_aliases.get(focus_raw, focus_raw)
    if focus not in {"all", "constraint", "structure", "relational", "relation_constraint"}:
        focus = "all"
    include_constraints = focus in {"all", "constraint", "relation_constraint"}
    include_structure = focus in {"all", "structure"}
    include_relational = focus in {"all", "relational", "relation_constraint"}

    claims = extract_constraint_claims(sections)
    constraint_kind_filter = _csv_env("STEP2_DOC_CONSTRAINT_KIND_FILTER")
    if constraint_kind_filter:
        claims = [claim for claim in claims if _claim_allowed_by_filter(claim, constraint_kind_filter)]
    full_uncovered = _find_uncovered_in_claims(tree, claims)
    full_uncovered_by_id = {claim.claim_id: (claim, evidence) for claim, evidence in full_uncovered}
    baseline_uncovered_ids: set[str] = set()
    if baseline_tree is not None:
        baseline_uncovered = _find_uncovered_in_claims(baseline_tree, claims)
        baseline_uncovered_ids = {claim.claim_id for claim, _ in baseline_uncovered}
    newly_uncovered_doc_ids = set(full_uncovered_by_id) - baseline_uncovered_ids
    selected_claims = _select_claim_window(
        claims,
        max_claims=max_issues,
        claim_ids=claim_ids,
        claim_offset=claim_offset,
    )
    uncovered = (
        [
            full_uncovered_by_id[claim.claim_id]
            for claim in selected_claims
            if claim.claim_id in full_uncovered_by_id
        ]
        if include_constraints
        else []
    )

    issues = dict(report.issues or {})
    errors = list(report.errors or [])
    extras = list(report.extras or [])
    extras.append(
        "[DOC_COVERAGE_FOCUS] "
        f"focus_kind={focus} "
        f"include_constraints={include_constraints} "
        f"include_structure={include_structure} "
        f"include_relational={include_relational}"
    )
    if include_constraints:
        extras.append(
            "[DOC_CONSTRAINT_COVERAGE_SUMMARY] "
            f"total_claims={len(claims)} "
            f"uncovered_total={len(full_uncovered)} "
            f"emitted={len(uncovered)} "
            f"claim_window={len(selected_claims)}"
        )

    if include_constraints and full_uncovered:
        sample_ids = ",".join(claim.claim_id for claim, _ in full_uncovered[:8])
        summary_description = (
            "Documentation constraint coverage has uncovered claims: "
            f"{len(full_uncovered)}/{len(claims)} claims are not represented in the tree. "
            "Detailed repair issues are capped to the configured stable claim window; "
            f"sample_uncovered_ids={sample_ids}."
        )
        summary_issue = Issue(
            id="doc_constraint_coverage_total_uncovered",
            type=IssueType.SEMANTICS,
            severity=Severity.WARN,
            description=summary_description,
            target=TargetRef(
                kind="doc_constraint_summary",
                identifier="doc_constraint_coverage_total_uncovered",
                metadata={
                    "total_claims": len(claims),
                    "uncovered_total": len(full_uncovered),
                    "emitted": len(uncovered),
                    "claim_window": len(selected_claims),
                },
            ),
            suggestions=(
                FixHint(
                    action="reduce_doc_constraint_uncovered_count",
                    details={
                        "uncovered_total": len(full_uncovered),
                        "emitted": len(uncovered),
                    },
                ),
            ),
        )
        issues[summary_issue.id] = summary_issue
        warnings = list(getattr(report, "warnings", []) or [])
        warnings.append(summary_description)
        extras.append(f"[DOC_CONSTRAINT_COVERAGE] {summary_description}")
    else:
        warnings = list(getattr(report, "warnings", []) or [])

    preservation_violations: List[Dict[str, Any]] = []
    if baseline_tree is not None and newly_uncovered_doc_ids:
        preservation_violations = find_node_constraint_preservation_violations(
            baseline_tree,
            tree,
            max_issues=max_issues,
            claims=claims,
        )
    for violation in preservation_violations:
        description = (
            "Existing node constraint was removed or broadened during doc-constraint repair: "
            f"{violation['node_name']}(ID:{violation['node_id']}) had "
            f"{violation['baseline_constraint']!r}; candidate constraints are "
            f"{violation['candidate_constraints']!r}. Preserve existing constraints and add "
            "scoped child constraints or target-specific variant formulas instead of broadening them."
        )
        issue = Issue(
            id=str(violation["id"]),
            type=IssueType.SEMANTICS,
            severity=Severity.ERROR,
            description=description,
            target=TargetRef(
                kind="node_constraint_preservation",
                identifier=str(violation["node_id"]),
                metadata=dict(violation),
            ),
            suggestions=(
                FixHint(
                    action="preserve_existing_node_constraint",
                    details={
                        "node_id": violation["node_id"],
                        "node_name": violation["node_name"],
                        "baseline_constraint": violation["baseline_constraint"],
                    },
                ),
            ),
        )
        issues[issue.id] = issue
        errors.append(description)
        extras.append(f"[DOC_CONSTRAINT_PRESERVATION] {description}")

    full_structure_issues = find_doc_structure_alignment_issues(tree, sections) if include_structure else []
    if include_structure:
        total_structure_requirements, uncovered_structure_requirements = doc_structure_coverage_counts(tree, sections)
    else:
        total_structure_requirements, uncovered_structure_requirements = 0, 0
    repairable_structure_issues = [
        issue
        for issue in full_structure_issues
        if include_weak_name_alignment or issue.get("kind") != "weak_name_alignment"
    ]
    structure_kind_filter = _csv_env("STEP2_DOC_STRUCTURE_KIND_FILTER")
    if structure_kind_filter:
        repairable_structure_issues = [
            issue
            for issue in repairable_structure_issues
            if str(issue.get("kind") or "").strip().lower() in structure_kind_filter
        ]
    repairable_structure_issues = sorted(repairable_structure_issues, key=_structure_repair_priority)
    if max_structure_issues is None:
        structure_limit = len(repairable_structure_issues) if max_issues is None else min(8, max(0, int(max_issues)))
    else:
        structure_limit = max(0, int(max_structure_issues))
    structure_issues = repairable_structure_issues[:structure_limit]
    if full_structure_issues or total_structure_requirements:
        suppressed_weak = sum(1 for issue in full_structure_issues if issue.get("kind") == "weak_name_alignment") - (
            sum(1 for issue in structure_issues if issue.get("kind") == "weak_name_alignment")
            if include_weak_name_alignment
            else 0
        )
        extras.append(
            "[DOC_STRUCTURE_ALIGNMENT_SUMMARY] "
            f"total_issues={len(full_structure_issues)} "
            f"total_requirements={total_structure_requirements} "
            f"uncovered_requirements={uncovered_structure_requirements} "
            f"repairable={len(repairable_structure_issues)} "
            f"emitted={len(structure_issues)} "
            f"weak_name_suppressed={max(0, suppressed_weak)}"
        )
    for structure_issue in structure_issues:
        requirement: StructuralFieldRequirement = structure_issue["requirement"]
        variant = structure_issue.get("variant")
        existing_node = structure_issue.get("existing_node")
        safe_repair: Dict[str, Any] = {
            "kind": structure_issue["kind"],
            "section": requirement.section_title,
            "field_name": requirement.field_name,
            "canonical_name": requirement.canonical_name,
            "guard_field": requirement.guard_field,
            "guard_values": list(requirement.guard_values),
            "guard_source": requirement.guard_source,
            "message_type": requirement.message_type,
            "format_name": requirement.format_name,
            "size_bits": requirement.size_bits,
            "direct_constraint": requirement.direct_constraint,
            "bit_position": requirement.bit_position,
            "byte_position": requirement.byte_position,
            "selector_variant_parenting": (
                "Keep the selector as a leaf field. Add selected variants as sibling containers under "
                "the selector's enclosing container, not as children of the selector leaf. If adding a "
                "new selector field for a table row, put that selector and its selected variants under "
                "the same enclosing container, and give each variant a size_bits expression that covers "
                "its documented children."
            ),
            "tlv_abstraction_hint": (
                "If many guarded rows share a TLV-style length/data envelope, first reuse an existing "
                "generic/catch-all TLV branch and its existing Length/Data semantics. Only add a new "
                "TLV_Option/Other_Option branch if no generic TLV branch exists. The catch-all "
                "condition should cover the documented length-bearing values and exclude documented "
                "single-byte/no-body values or unrelated exact variants as needed."
            ),
        }
        if variant:
            safe_repair.update(
                {
                    "variant_id": variant.get("node_id"),
                    "variant_name": variant.get("name"),
                }
            )
        if existing_node:
            safe_repair.update(
                {
                    "existing_node_id": existing_node.get("node_id"),
                    "existing_node_name": existing_node.get("name"),
                    "preserve_existing_constraints": list(existing_node.get("constraints") or []),
                    "preserve_existing_size_bits": existing_node.get("size_bits"),
                }
            )

        if structure_issue["kind"] == "missing_repeated_format":
            expected_specs = list(structure_issue.get("expected_field_specs") or [])
            expected_fields = list(structure_issue.get("expected_body_fields") or [])
            repeated_format_name = str(structure_issue.get("repeated_format_name") or requirement.format_name or requirement.field_name)
            repeat_count_field = str(structure_issue.get("repeat_count_field") or requirement.guard_field)
            safe_repair.update(
                {
                    "expected_field_specs": expected_specs,
                    "expected_body_fields": expected_fields,
                    "repeat_count_field": repeat_count_field,
                    "repeated_format_name": repeated_format_name,
                    "repair_phase": "repeated_section_shell",
                    "recommended_shape": (
                        "Add a distinct repeated section/container for the documented format, include only the "
                        "documented body fields in order, and connect the count field to that section with a "
                        "repeat_count or constraint_ir count relation. Do not reuse a different repeated format "
                        "whose envelope has extra fields."
                    ),
                }
            )
            field_summary = ", ".join(str(value) for value in expected_fields[:8])
            description = (
                "Documentation repeated section appears uncovered: "
                f"[{requirement.section_title}] {repeat_count_field} controls repeated format "
                f"{repeated_format_name!r}, but the tree has no distinct matching section/container. "
                "SAFE STRUCTURAL REPAIR: add that repeated section/container under the message payload, "
                f"include documented body fields{(': ' + field_summary) if field_summary else ''}, "
                f"and add a count relation from {repeat_count_field} to the new section. "
                "Do not satisfy this by reusing a different record format with additional fields."
            )
            target_kind = "doc_structure_repeated_format"
            identifier = f"{requirement.section_title}:{repeat_count_field}:{repeated_format_name}"
        elif structure_issue["kind"] == "missing_tlv_abstraction":
            profile = structure_issue.get("tlv_profile") or {}
            zero_values = sorted(int(value) for value in profile.get("zero_body_values") or [])
            length_values = sorted(int(value) for value in profile.get("length_values") or [])
            length_name = "Option_Length"
            payload_name = "Option_Data"
            catchall_hint = (
                "exclude single-byte/no-body selector values "
                f"{zero_values}"
                if zero_values
                else "cover the length-bearing selector values without enumerating each row"
            )
            length_formula_hint = (
                "preserve the existing Length/Data size semantics if a generic TLV branch already exists; "
                "only derive a new payload size formula from documentation evidence, because some protocols "
                "define length as payload bytes while others define total option bytes"
            )
            safe_repair.update(
                {
                    "tlv_profile": {
                        "guard_field": profile.get("guard_field"),
                        "message_type": profile.get("message_type"),
                        "guard_values": list(profile.get("guard_values") or []),
                        "length_values": length_values,
                        "zero_body_values": zero_values,
                        "length_field_name": length_name,
                        "payload_field_name": payload_name,
                        "documented_length_field_names": list(profile.get("length_field_names") or []),
                        "documented_payload_field_names": list(profile.get("payload_field_names") or []),
                    },
                    "recommended_shape": (
                        "First reuse an existing generic/catch-all TLV branch if present by updating its "
                        f"condition_on formula so it covers length-bearing values {length_values}. "
                        f"Preserve its existing Length/Data nodes and size semantics. If no such branch exists, "
                        f"add exact zero-size variants for {zero_values} when present and add one "
                        f"TLV_Option/Other_Option variant containing {length_name} and {payload_name}. "
                        f"{length_formula_hint}."
                    ),
                }
            )
            description = (
                "Documentation structure appears to be a repeated selector table with a TLV envelope: "
                f"[{requirement.section_title}] uses {requirement.guard_field} with length-bearing values "
                f"{length_values}. SAFE STRUCTURAL REPAIR: model this as a generic TLV abstraction first, "
                "not as many near-duplicate per-value variants. If the tree already has a generic TLV branch "
                "with Length/Data fields, prefer updating that existing condition_on edge so it covers these "
                f"length-bearing values; do not create a parallel TLV_Option branch. If no generic branch exists, "
                f"create zero-size/single-byte variants for {zero_values or 'documented no-body values'}; "
                f"create one TLV_Option/Other_Option branch whose condition {catchall_hint}, and include "
                f"generic {length_name} plus {payload_name}. {length_formula_hint}. "
                "Preserve any already modeled exact variants and do not make the catch-all merely exclude rows "
                "already modeled in this patch."
            )
            target_kind = "doc_structure_tlv_abstraction"
            identifier = f"{requirement.section_title}:{requirement.guard_field}:tlv"
        elif structure_issue["kind"] == "opaque_variant_expansion":
            expected_specs = list(structure_issue.get("expected_field_specs") or [])
            missing_fields = list(structure_issue.get("missing_expected_fields") or structure_issue.get("expected_body_fields") or [])
            opaque_child_ids = list(structure_issue.get("opaque_child_ids") or [])
            field_summary = ", ".join(str(value) for value in missing_fields[:8])
            safe_repair.update(
                {
                    "expected_field_specs": expected_specs,
                    "missing_expected_fields": missing_fields,
                    "opaque_child_ids": opaque_child_ids,
                    "repair_phase": "body_expansion",
                    "recommended_shape": (
                        "Expand the existing variant body in place: keep the selector condition and variant id, "
                        "replace or refine opaque payload children with the documented internal fields, and keep "
                        "a residual payload only when the documented fixed fields do not consume the whole length. "
                        "Use only fields listed in expected_field_specs; do not invent undocumented spare/reserved "
                        "prefix fields from model prior knowledge."
                    ),
                }
            )
            description = (
                "Documentation structure is more detailed than the current TLV variant body: "
                f"[{requirement.section_title}] format {requirement.format_name!r} matches existing variant "
                f"{variant.get('name')!r}(ID:{variant.get('node_id')}) but the variant is empty or only has an opaque "
                f"payload child. SAFE STRUCTURAL REPAIR: expand that existing variant in place with documented "
                f"fields{(': ' + field_summary) if field_summary else ''}; preserve the current condition_on edge "
                "and length binding. Do not add a parallel duplicate variant for the same selector value. Do not "
                "invent undocumented spare/reserved/padding fields; if a residual payload is needed, start it "
                "immediately after the documented fields and subtract the documented field widths from its size."
            )
            target_kind = "doc_structure_variant_expansion"
            identifier = str(variant.get("node_id"))
        elif structure_issue["kind"] == "missing_selector_variants":
            expected_variants = list(structure_issue.get("expected_variants") or [])
            missing_values = [
                int(value)
                for value in (structure_issue.get("missing_selector_values") or requirement.guard_values)
                if _parse_int(value) is not None
            ]
            variant_summary = ", ".join(
                f"{item.get('canonical_variant_name')}={item.get('selector_values')}"
                for item in expected_variants[:8]
            )
            selector_coded_cluster_active = os.getenv("STEP2_STAGEB_SELECTOR_CODED_CLUSTER", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if selector_coded_cluster_active:
                repair_phase = "selector_coded_cluster"
                recommended_shape = (
                    "Add or reuse condition_on variants for the documented selector values, and expand the "
                    "typed variant body when the focused documentation lists body fields. Preserve the existing "
                    "selector field, generic fallback/TLV branch, active train-packet branch, and existing length "
                    "binding. Do not add new length_of, count/repeat, or cross-field relation edges in this "
                    "selector-coded structural pass; add selector-conditioned typed body fields and scalar "
                    "constraints only. Length/count/formula binding belongs to the relation compact pass."
                )
                description_suffix = (
                    "In selector-coded cluster mode, include documented typed body fields in the focused variant "
                    "instead of stopping at zero-size shells. Preserve the existing selector field and generic "
                    "fallback/TLV branch; do not create a duplicate catch-all branch. Do not add new length_of, "
                    "count/repeat, or cross-field relation edges in this structural pass; add typed variant body "
                    "fields and scalar constraints only, then let relation compact bind length/count/formula "
                    "constraints after the body exists."
                )
            else:
                repair_phase = "selector_shell"
                recommended_shape = (
                    "First add or rename condition_on variant shells for the documented selector values. Keep "
                    "the selector as a leaf field, parent the shells under the selector's enclosing container, "
                    "and do not expand body fields in this phase. A later body_expansion issue should refine "
                    "each selected variant using source-backed scoped field definitions."
                )
                description_suffix = (
                    "Add condition_on variant shells for these documented selector values, named from the documented "
                    "variant labels. Preserve the existing selector field and existing generic body/TLV sequence; do "
                    "not expand body fields in this selector-shell phase. Do not create or edit Length/Data/TLV body "
                    "nodes in this selector-shell patch; if a length-delimited body is needed, it must be handled by a "
                    "later body-expansion/TLV patch that first binds the parent/body span to the documented Length field."
                )
            safe_repair.update(
                {
                    "selector_name": requirement.guard_field,
                    "selector_values": missing_values,
                    "expected_variants": expected_variants,
                    "matched_selector_node_ids": list(structure_issue.get("matched_selector_node_ids") or []),
                    "repair_phase": repair_phase,
                    "recommended_shape": recommended_shape,
                }
            )
            description = (
                "Documentation selector map appears uncovered: "
                f"[{requirement.section_title}] maps {requirement.guard_field} values {missing_values} "
                f"to variants {variant_summary or requirement.canonical_name}. SAFE STRUCTURAL REPAIR: "
                f"{description_suffix}"
            )
            target_kind = "doc_structure_selector_variants"
            identifier = f"{requirement.section_title}:{requirement.guard_field}:{'|'.join(str(v) for v in missing_values)}"
        elif structure_issue["kind"] == "missing_guarded_variant":
            missing_values = "|".join(str(value) for value in structure_issue.get("missing_guard_values") or requirement.guard_values)
            expected_fields = ", ".join(structure_issue.get("expected_body_fields") or [])
            tlv_profile = structure_issue.get("tlv_profile") or {}
            if not tlv_profile:
                tlv_tokens = (
                    _field_tokens(requirement.guard_field)
                    | _field_tokens(requirement.section_title)
                    | _field_tokens(requirement.format_name)
                )
                if {"option", "tlv"} & tlv_tokens:
                    tlv_profile = {
                        "guard_field": requirement.guard_field,
                        "message_type": requirement.message_type,
                        "guard_values": list(requirement.guard_values),
                        "length_values": list(requirement.guard_values),
                        "zero_body_values": [],
                        "length_field_names": ["Option_Length"],
                        "payload_field_names": ["Option_Data"],
                    }
            tlv_guidance = ""
            if tlv_profile:
                safe_repair.update(
                    {
                        "tlv_profile": {
                            "guard_field": tlv_profile.get("guard_field"),
                            "message_type": tlv_profile.get("message_type"),
                            "guard_values": list(tlv_profile.get("guard_values") or []),
                            "length_values": list(tlv_profile.get("length_values") or []),
                            "zero_body_values": list(tlv_profile.get("zero_body_values") or []),
                            "length_field_names": list(tlv_profile.get("length_field_names") or []),
                            "payload_field_names": list(tlv_profile.get("payload_field_names") or []),
                        },
                        "preserve_generic_tlv_branch": True,
                        "repair_phase": "exact_selector_variant",
                        "recommended_shape": (
                            "Add the exact selector-coded variant as a sibling of the existing generic TLV "
                            "branch. Preserve the existing generic TLV branch, catch-all condition, active "
                            "train-packet branch, Length/Data field ids, and existing length binding unless "
                            "the patch also proves the active packet still parses. Add the typed variant body "
                            "fields and scalar constraints only. Do not add new length_of, count/repeat, or "
                            "cross-field relation edges in this structural pass; length/count/formula binding "
                            "belongs to the relation compact pass. Do not change a generic payload formula into "
                            "an absolute packet-sized span."
                        ),
                    }
                )
                tlv_guidance = (
                    " This is a TLV-style selector-coded variant: preserve the existing generic TLV branch, "
                    "its catch-all condition, active train-packet branch, Length/Data field ids, and length "
                    "binding. Add typed variant body fields and scalar constraints only in this structural "
                    "pass. Do not add new length_of, count/repeat, or cross-field relation edges here; relation "
                    "compact binds length/count/formula constraints after the body exists. Add the exact variant "
                    "as a sibling/fallback refinement only; do not edit the generic payload span or convert the "
                    "option length into an absolute packet-sized span."
                )
            description = (
                "Documentation structure appears uncovered: "
                f"[{requirement.section_title}] has no condition_on variant for "
                f"{requirement.guard_field} in {{{missing_values}}}. "
                "SAFE STRUCTURAL REPAIR: add a variant selected by the documented guard value(s), "
                f"name it from the section title, and include documented body fields"
                f"{(': ' + expected_fields) if expected_fields else ''}. "
                "Keep the selector as a leaf field; parent the variant under the selector's enclosing container, "
                "start the variant body immediately after the selector field, and set the variant/container "
                "size_bits to the sum of its children so static validation can prove the layout. "
                "Do not use only a variant-level formula when the documentation lists on-wire fields."
                f"{tlv_guidance}"
            )
            target_kind = "doc_structure_variant"
            identifier = f"{requirement.section_title}:{missing_values}"
        elif structure_issue["kind"] == "weak_name_alignment":
            description = (
                "Documentation structure has weak field-name alignment: "
                f"[{requirement.section_title}] field {requirement.field_name!r} under "
                f"{requirement.guard_field} in {list(requirement.guard_values)} appears to be "
                f"{existing_node.get('name')!r}(ID:{existing_node.get('node_id')}) in variant "
                f"{variant.get('name')!r}(ID:{variant.get('node_id')}). "
                f"SAFE STRUCTURAL REPAIR: rename or alias that existing node to {requirement.canonical_name!r}; "
                "preserve node id, size_bits, children, and existing constraints. "
                + (
                    f"Also preserve/add the documented constraint {requirement.direct_constraint!r} on that field. "
                    if requirement.direct_constraint
                    else ""
                )
                + "Do not create a duplicate field unless the existing node is not the same wire bytes."
            )
            target_kind = "doc_structure_name_alignment"
            identifier = str(existing_node.get("node_id"))
        else:
            description = (
                "Documentation structure appears uncovered: "
                f"[{requirement.section_title}] variant {variant.get('name')!r}(ID:{variant.get('node_id')}) "
                f"does not contain documented field {requirement.field_name!r}. "
                f"SAFE STRUCTURAL REPAIR: add a child field named {requirement.canonical_name!r}"
                + (f" with size_bits={requirement.size_bits!r}" if requirement.size_bits is not None else "")
                + (f" and constraint {requirement.direct_constraint!r}" if requirement.direct_constraint else "")
                + "; preserve existing sibling order/offsets. "
                "A variant-level formula alone does not represent the missing on-wire field."
            )
            target_kind = "doc_structure_field"
            identifier = f"{variant.get('node_id')}:{requirement.canonical_name}"

        metadata = {
            "source": requirement.source,
            "safe_repair": safe_repair,
            "expected_body_fields": list(structure_issue.get("expected_body_fields") or []),
        }
        issue = Issue(
            id=str(structure_issue["id"]),
            type=IssueType.STRUCTURE,
            severity=Severity.ERROR,
            description=description,
            target=TargetRef(
                kind=target_kind,
                identifier=identifier,
                metadata=metadata,
            ),
            suggestions=(
                FixHint(
                    action="align_documented_structure",
                    details=safe_repair,
                ),
            ),
        )
        issues[issue.id] = issue
        errors.append(description)
        extras.append(f"[DOC_STRUCTURE_ALIGNMENT] {description}")

    if include_relational:
        total_relational_obligations, uncovered_relational_count = relational_obligation_coverage_counts(tree, sections)
        full_relational_uncovered = find_uncovered_relational_obligations(tree, sections)
        relational_kind_filter = _csv_env("STEP2_DOC_RELATIONAL_KIND_FILTER")
        if relational_kind_filter:
            full_relational_uncovered = [
                (obligation, evidence)
                for obligation, evidence in full_relational_uncovered
                if str(getattr(obligation, "kind", "") or "").strip().lower() in relational_kind_filter
            ]
    else:
        total_relational_obligations, uncovered_relational_count = 0, 0
        full_relational_uncovered = []
    env_rel_limit = os.getenv("STEP2_DOC_RELATIONAL_MAX_ISSUES")
    if env_rel_limit:
        try:
            relational_limit = max(0, int(env_rel_limit))
        except ValueError:
            relational_limit = 6
    elif max_issues is None:
        relational_limit = len(full_relational_uncovered)
    else:
        relational_limit = min(6, max(0, int(max_issues)))
    relational_uncovered = full_relational_uncovered[:relational_limit]
    if total_relational_obligations:
        extras.append(
            "[DOC_RELATIONAL_OBLIGATION_SUMMARY] "
            f"total_obligations={total_relational_obligations} "
            f"uncovered_total={uncovered_relational_count} "
            f"emitted={len(relational_uncovered)}"
        )
    if full_relational_uncovered:
        sample_ids = ",".join(obligation.obligation_id for obligation, _ in full_relational_uncovered[:8])
        summary_description = (
            "Documentation relational-obligation coverage has uncovered prose relations: "
            f"{uncovered_relational_count}/{total_relational_obligations} obligations are not represented in the tree. "
            "These are generic if/when/must/shall/length/alignment/padding obligations; "
            f"sample_uncovered_ids={sample_ids}."
        )
        summary_issue = Issue(
            id="doc_relational_obligation_total_uncovered",
            type=IssueType.SEMANTICS,
            severity=Severity.WARN,
            description=summary_description,
            target=TargetRef(
                kind="doc_relational_obligation_summary",
                identifier="doc_relational_obligation_total_uncovered",
                metadata={
                    "total_obligations": total_relational_obligations,
                    "uncovered_total": uncovered_relational_count,
                    "emitted": len(relational_uncovered),
                },
            ),
            suggestions=(
                FixHint(
                    action="reduce_doc_relational_obligation_uncovered_count",
                    details={
                        "uncovered_total": uncovered_relational_count,
                        "emitted": len(relational_uncovered),
                    },
                ),
            ),
        )
        issues[summary_issue.id] = summary_issue
        warnings.append(summary_description)
        extras.append(f"[DOC_RELATIONAL_OBLIGATION] {summary_description}")

    for obligation, evidence in relational_uncovered:
        repair_guidance, repair_metadata = _relational_safe_repair_guidance(tree, obligation)
        description = (
            "Documentation relational obligation appears uncovered: "
            f"[{obligation.section_title}] {obligation.short_expected()}. "
            f"Evidence: {evidence}. Source relation: {obligation.relation_text!r}. "
            f"{repair_guidance}"
        )
        metadata = {
            "section": obligation.section_title,
            "kind": obligation.kind,
            "field_names": list(obligation.field_names),
            "primary_field": obligation.primary_field,
            "related_fields": list(obligation.related_fields),
            "numeric_values": list(obligation.numeric_values),
            "source": obligation.source,
            "relation_text": obligation.relation_text,
        }
        metadata.update(repair_metadata)
        issue = Issue(
            id=obligation.obligation_id,
            type=IssueType.SEMANTICS,
            severity=Severity.ERROR,
            description=description,
            target=TargetRef(
                kind="doc_relational_obligation",
                identifier=obligation.obligation_id,
                metadata=metadata,
            ),
            suggestions=(
                FixHint(
                    action="add_or_scope_documented_relational_constraint",
                    details={
                        "obligation_kind": obligation.kind,
                        "field_names": list(obligation.field_names),
                        "numeric_values": list(obligation.numeric_values),
                        "source_relation": obligation.relation_text,
                        **repair_metadata,
                    },
                ),
            ),
        )
        issues[issue.id] = issue
        errors.append(description)
        extras.append(f"[DOC_RELATIONAL_OBLIGATION] {description}")

    for claim, evidence in uncovered:
        repair_guidance, repair_metadata = _safe_repair_guidance(tree, claim)
        guard = ""
        if claim.guard_field and claim.guard_values:
            values = "|".join(str(v) for v in claim.guard_values)
            guard = f" under {claim.guard_field} in {{{values}}}"
        description = (
            "Documentation constraint appears uncovered: "
            f"[{claim.section_title}] {claim.short_expected()}{guard}. "
            f"Evidence: {evidence}. Add a documented node constraint, scoped variant formula, "
            "or condition_on branch; do not hard-code protocol-specific rules. "
            f"{repair_guidance}"
        )
        metadata = {
            "section": claim.section_title,
            "field": claim.field_name,
            "kind": claim.kind,
            "values": list(claim.values),
            "min_value": claim.min_value,
            "max_value": claim.max_value,
            "guard_field": claim.guard_field,
            "guard_values": list(claim.guard_values),
            "guard_source": claim.guard_source,
            "source": claim.source,
        }
        metadata.update(repair_metadata)
        issue = Issue(
            id=claim.claim_id,
            type=IssueType.SEMANTICS,
            severity=Severity.ERROR,
            description=description,
            target=TargetRef(
                kind="doc_constraint",
                identifier=claim.claim_id,
                metadata=metadata,
            ),
            suggestions=(
                FixHint(
                    action="add_or_scope_documented_constraint",
                    details={
                        "field": claim.field_name,
                        "guard_field": claim.guard_field,
                        "guard_values": list(claim.guard_values),
                        "expected": claim.short_expected(),
                        **repair_metadata,
                    },
                ),
            ),
        )
        issues[issue.id] = issue
        errors.append(description)
        extras.append(f"[DOC_CONSTRAINT_COVERAGE] {description}")

    return ValidationReport(
        ok=bool(
            report.ok
            and not (full_uncovered if include_constraints else [])
            and not preservation_violations
            and not full_structure_issues
            and not full_relational_uncovered
        ),
        errors=errors,
        warnings=warnings,
        extras=extras,
        issues=issues,
        traffic_failures=list(getattr(report, "traffic_failures", []) or []),
        traffic_global_summary=getattr(report, "traffic_global_summary", None),
        traffic_repair_hints=list(getattr(report, "traffic_repair_hints", []) or []),
        traffic_successful_samples=int(getattr(report, "traffic_successful_samples", 0) or 0),
        traffic_total_samples=int(getattr(report, "traffic_total_samples", 0) or 0),
        traffic_max_bit_reached=copy.deepcopy(getattr(report, "traffic_max_bit_reached", None)),
        traffic_total_bits_per_sample=copy.deepcopy(getattr(report, "traffic_total_bits_per_sample", None)),
        traffic_length_total_abs_error_bits=int(getattr(report, "traffic_length_total_abs_error_bits", 0) or 0),
        traffic_wire_length_total_abs_error_bits=int(getattr(report, "traffic_wire_length_total_abs_error_bits", 0) or 0),
        traffic_total_coverage_gap_bits=int(getattr(report, "traffic_total_coverage_gap_bits", 0) or 0),
        traffic_coverage_gap_samples=int(getattr(report, "traffic_coverage_gap_samples", 0) or 0),
        traffic_overflow_length_bits=int(getattr(report, "traffic_overflow_length_bits", 0) or 0),
        traffic_content_covered_bits_per_sample=copy.deepcopy(
            getattr(report, "traffic_content_covered_bits_per_sample", None)
        ),
        traffic_content_coverage_ratio_per_sample=copy.deepcopy(
            getattr(report, "traffic_content_coverage_ratio_per_sample", None)
        ),
    )
