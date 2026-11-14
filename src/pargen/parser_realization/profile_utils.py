from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _list_of_str(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def profile_forbidden_required(profile: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    if not isinstance(profile, dict) or not profile:
        return [], []
    forbidden = _list_of_str(profile.get("forbidden_substrings"))
    required = _list_of_str(profile.get("required_substrings"))
    return forbidden, required


def profile_max_attempts(
    profile: Optional[Dict[str, Any]],
    *,
    default: int = 2,
    key: str = "max_codegen_attempts",
    min_attempts: int = 1,
    max_attempts: int = 3,
) -> int:
    if not isinstance(profile, dict) or not profile:
        return default
    raw = profile.get(key, default)
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(min_attempts, min(max_attempts, value))


def profile_violations(files: Dict[str, str], profile: Optional[Dict[str, Any]]) -> List[str]:
    forbidden, required = profile_forbidden_required(profile)
    if not forbidden and not required:
        return []

    violations: List[str] = []

    for token in forbidden:
        hits = [name for name, content in files.items() if token in (content or "")]
        if hits:
            violations.append(f"forbidden_substring={token!r} present in files={sorted(hits)}")

    for token in required:
        present = any(token in (content or "") for content in files.values())
        if not present:
            violations.append(f"required_substring={token!r} missing from all output files")

    return violations



# Keys of a codegen profile that carry protocol knowledge rather than platform integration.  Code
# generation receives only the PKG and the platform context (registration, reads, field emission, diagnostics).
PROTOCOL_HINT_KEYS = frozenset({
    "semantic_rule_hints",
    "constraint_rule_hints",
    "field_label_overrides",
    "bitfield_order_overrides",
    "flag_presence_overrides",
    "canonical_field_labels",
    "field_display_overrides",
    "bitfield_orders",
    "flag_controlled_fields",
    "optional_body_overrides",
    "stage_c_scaffold_first",
    "scaffold_first",
    "scaffold_first_model_free",
})


def strip_protocol_hints(profile: Any) -> Any:
    """Return a copy of a codegen profile without protocol-knowledge hints (recursively)."""
    if isinstance(profile, dict):
        return {k: strip_protocol_hints(v) for k, v in profile.items() if k not in PROTOCOL_HINT_KEYS}
    if isinstance(profile, list):
        return [strip_protocol_hints(v) for v in profile]
    return profile
