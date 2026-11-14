from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


_PROTOCOL_ALIASES: dict[str, tuple[str, ...]] = {
    "bfd": ("bfd", "bidirectional forwarding detection"),
    "dhcp": ("dhcp", "dynamic host configuration protocol"),
    "dns": ("dns", "domain name system"),
    "icmp": ("icmp", "internet control message protocol"),
    "ipv4": ("ipv4", "ip", "internet protocol"),
    "modbus": ("modbus",),
    "pfcp": ("pfcp", "packet forwarding control protocol"),
    "tcp": ("tcp", "transmission control protocol"),
}

_PARSER_RELEVANCE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bheader\b",
        r"\bformat\b",
        r"\bfield(s)?\b",
        r"\boption(s)?\b",
        r"\bmessage(s)?\b",
        r"\bpacket(s)?\b",
        r"\bprotocol\s+description\b",
        r"\bframe(s)?\b",
        r"\bdatagram(s)?\b",
        r"\bsegment(s)?\b",
        r"\bpdu(s)?\b",
        r"\bfunction\s+code(s)?\b",
        r"\bcommand(s)?\b",
        r"\boperation(s)?\b",
        r"\btransaction(s)?\b",
        r"\btype(s)?\b",
        r"\blength\b",
        r"\bsize\b",
        r"\bflag(s)?\b",
        r"\bbit(s)?\b",
        r"\bbyte(s)?\b",
        r"\boctet(s)?\b",
        r"\bencoding\b",
        r"\bencoded\b",
        r"\bvalid value(s)?\b",
        r"\bvalue(s)?\b",
        r"\benum(eration)?(s)?\b",
        r"\b0x[0-9a-f]+\b",
        r"\btable(s)?\b",
        r"\btlv(s)?\b",
        r"\bie(s)?\b",
        r"\binformation element(s)?\b",
        r"\bidentifier(s)?\b",
        r"\baddress(es)?\b",
        r"\baddressing\s+model\b",
        r"\bdata\s+model\b",
        r"\bchecksum\b",
        r"\bpayload\b",
        r"\bcompression\b",
        r"\bselector(s)?\b",
        r"\bvariant(s)?\b",
        r"\brequest\b",
        r"\bresponse(s)?\b",
        r"\bresource record(s)?\b",
        r"\bquestion\b",
        r"\banswer\b",
        r"\brdata\b",
    )
)


def protocol_aliases(target_protocol: str | None, extra_aliases: Iterable[str] = ()) -> tuple[str, ...]:
    if not target_protocol:
        return tuple(_normalize(alias) for alias in extra_aliases if _normalize(alias))
    normalized = _normalize(target_protocol)
    base_protocol = normalized.split(" ", 1)[0]
    aliases = list(_PROTOCOL_ALIASES.get(normalized) or _PROTOCOL_ALIASES.get(base_protocol) or (normalized,))
    aliases.extend(str(alias) for alias in extra_aliases)
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        normalized_alias = _normalize(alias)
        if normalized_alias and normalized_alias not in seen:
            seen.add(normalized_alias)
            out.append(normalized_alias)
    return tuple(out)


def is_parser_relevant_section(title_or_path: str) -> bool:
    normalized = _normalize(title_or_path)
    return any(pattern.search(normalized) for pattern in _PARSER_RELEVANCE_PATTERNS)


def select_target_protocol_sections(
    sections: Sequence[T],
    *,
    target_protocol: str | None,
    target_protocol_aliases: Iterable[str] = (),
) -> list[T]:
    """Select parser-relevant sections conditioned on the target protocol.

    The selector works on TOC-like objects with ``number``, ``title``, and
    ``level`` attributes.  If the TOC names the target protocol, it keeps
    parser-relevant leaves under that target subtree and drops parser-looking
    leaves under other protocol subtrees.  If the target name is absent, it
    falls back to generic parser-relevance so single-protocol RFCs with generic
    titles such as "Header Format" are still usable.
    """

    entries = list(sections)
    if not entries:
        return []

    leaves = _leaf_entries(entries)
    aliases = protocol_aliases(target_protocol, target_protocol_aliases)
    if not aliases:
        return leaves

    target_subtree_leaves = _target_protocol_subtree_leaves(entries, leaves, aliases)
    if target_subtree_leaves:
        target_relevant = [
            entry
            for entry in target_subtree_leaves
            if is_parser_relevant_section(_section_path_text(entry, entries))
        ]
        return target_relevant or target_subtree_leaves

    parser_hits = []
    for entry in leaves:
        path_text = _section_path_text(entry, entries)
        is_relevant = is_parser_relevant_section(path_text)
        if is_relevant:
            parser_hits.append(entry)

    if parser_hits:
        return parser_hits
    return leaves


def selection_cache_matches(
    cache_payload: object,
    *,
    section_strategy: str,
    target_protocol: str | None,
    target_protocol_aliases: Iterable[str] = (),
) -> bool:
    """Return whether a Stage-A section-selection cache matches current inputs."""

    normalized_strategy = str(section_strategy or "leaf_only")
    normalized_target = _normalize(target_protocol)
    normalized_aliases = tuple(_normalize(alias) for alias in target_protocol_aliases if _normalize(alias))

    if not isinstance(cache_payload, dict):
        return normalized_strategy == "leaf_only" and not normalized_target and not normalized_aliases

    cached_strategy = str(cache_payload.get("section_strategy") or "leaf_only")
    cached_target = _normalize(cache_payload.get("target_protocol"))
    cached_aliases = tuple(
        _normalize(alias)
        for alias in (cache_payload.get("target_protocol_aliases") or [])
        if _normalize(alias)
    )
    return (
        cached_strategy == normalized_strategy
        and cached_target == normalized_target
        and cached_aliases == normalized_aliases
    )


def _leaf_entries(entries: Sequence[T]) -> list[T]:
    numbers_with_children = {str(_entry_value(entry, "parent_number") or "") for entry in entries}
    for candidate in entries:
        number = str(_entry_value(candidate, "number") or "")
        level = _level(candidate)
        for other in entries:
            if other is candidate:
                continue
            other_number = str(_entry_value(other, "number") or "")
            if _level(other) == level + 1 and other_number.startswith(number + "."):
                numbers_with_children.add(number)
                break
    return [entry for entry in entries if str(_entry_value(entry, "number") or "") not in numbers_with_children]


def _target_protocol_subtree_leaves(entries: Sequence[T], leaves: Sequence[T], aliases: Sequence[str]) -> list[T]:
    selected: list[T] = []
    seen: set[int] = set()
    for entry in entries:
        if not _contains_any_alias(_section_path_text(entry, entries), aliases):
            continue
        descendants = [leaf for leaf in leaves if leaf is not entry and _is_descendant_of(leaf, entry, entries)]
        if not descendants:
            continue
        for leaf in descendants:
            identity = id(leaf)
            if identity not in seen:
                seen.add(identity)
                selected.append(leaf)
    return selected


def _is_descendant_of(entry: T, ancestor: T, entries: Sequence[T]) -> bool:
    ancestor_number = str(_entry_value(ancestor, "number") or "")
    if not ancestor_number:
        return False

    by_number = {str(_entry_value(item, "number") or ""): item for item in entries}
    parent_number = str(_entry_value(entry, "parent_number") or "")
    while parent_number:
        if parent_number == ancestor_number:
            return True
        parent = by_number.get(parent_number)
        if parent is None:
            break
        parent_number = str(_entry_value(parent, "parent_number") or "")

    number = str(_entry_value(entry, "number") or "")
    return _level(entry) > _level(ancestor) and number.startswith(ancestor_number + ".")


def _section_path_text(entry: T, entries: Sequence[T]) -> str:
    parts = [str(_entry_value(entry, "title") or "")]
    parent_number = str(_entry_value(entry, "parent_number") or "")
    by_number = {str(_entry_value(item, "number") or ""): item for item in entries}
    current = entry
    while parent_number:
        parent = by_number.get(parent_number)
        if parent is None:
            break
        parts.append(str(_entry_value(parent, "title") or ""))
        current = parent
        parent_number = str(_entry_value(current, "parent_number") or "")
    if len(parts) == 1:
        number = str(_entry_value(entry, "number") or "")
        level = _level(entry)
        for parent in entries:
            parent_number_candidate = str(_entry_value(parent, "number") or "")
            if (
                parent is not entry
                and _level(parent) < level
                and number.startswith(parent_number_candidate + ".")
            ):
                parts.append(str(_entry_value(parent, "title") or ""))
    return " ".join(reversed(parts))


def _level(entry: T) -> int:
    raw = _entry_value(entry, "level")
    try:
        return int(raw)
    except Exception:
        number = str(_entry_value(entry, "number") or "")
        return number.count(".") + 1 if number else 1


def _contains_any_alias(text: str, aliases: Sequence[str]) -> bool:
    normalized_text = _normalize(text)
    return any(_contains_alias(normalized_text, alias) for alias in aliases)


def _contains_alias(normalized_text: str, alias: str) -> bool:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", normalized_text) is not None


def _normalize(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _entry_value(entry: object, key: str) -> object:
    if isinstance(entry, Mapping):
        return entry.get(key)
    return getattr(entry, key, None)
