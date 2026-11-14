"""Machine-readable protocol constraint annotations.

The Stage-B tree already has an executable constraint DSL (`enum:`, `range:`,
`formula:`).  Some documented protocol obligations are relational knowledge
that is useful for evaluation and later code generation but not safe to force
into the traffic parser as an executable predicate.  `constraint_ir:` keeps
those obligations explicit without changing the tree schema.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional, Set


CONSTRAINT_IR_PREFIX = "constraint_ir:"
RELATION_IR_PREFIX = "relation_ir:"
_VAL_REF_RE = re.compile(r"\bval\s*\(\s*([^)]*?)\s*\)", re.IGNORECASE)


def parse_constraint_ir(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse a `constraint_ir:` string into a normalized dict.

    Returns None for non-IR constraints or malformed IR.  Callers that need to
    report malformed entries can use `is_constraint_ir_text` first.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    lower = text.lower()
    if lower.startswith(CONSTRAINT_IR_PREFIX):
        payload = text.split(":", 1)[1].strip()
    elif lower.startswith(RELATION_IR_PREFIX):
        payload = text.split(":", 1)[1].strip()
    else:
        return None
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    kind = str(parsed.get("kind") or parsed.get("type") or "").strip().lower()
    if not kind:
        return None
    parsed = dict(parsed)
    parsed["kind"] = kind
    return parsed


def is_constraint_ir_text(raw: Any) -> bool:
    if not isinstance(raw, str):
        return False
    lower = raw.strip().lower()
    return lower.startswith(CONSTRAINT_IR_PREFIX) or lower.startswith(RELATION_IR_PREFIX)


def constraint_ir_to_text(
    *,
    kind: str,
    expr: str = "",
    field_ids: Optional[Iterable[Any]] = None,
    field_names: Optional[Iterable[Any]] = None,
    source: str = "",
) -> str:
    payload: Dict[str, Any] = {"kind": str(kind).strip().lower()}
    if expr:
        payload["expr"] = str(expr)
    if field_ids is not None:
        payload["field_ids"] = [int(value) if str(value).isdigit() else str(value) for value in field_ids]
    if field_names is not None:
        payload["field_names"] = [str(value) for value in field_names if str(value).strip()]
    if source:
        payload["source"] = str(source)
    return f"{CONSTRAINT_IR_PREFIX} {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"


def constraint_ir_field_ids(ir: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for key in ("field_ids", "fields", "nodes", "node_ids"):
        values = ir.get(key)
        if values is None:
            continue
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        for value in values:
            text = str(value).strip()
            if re.fullmatch(r"\d+", text):
                ids.add(text)
    for key in ("expr", "expression", "formula", "applies_when", "guard"):
        text = str(ir.get(key) or "")
        for match in _VAL_REF_RE.finditer(text):
            token = match.group(1).strip()
            if re.fullmatch(r"\d+", token):
                ids.add(token)
    return ids


def constraint_ir_blob(ir: Dict[str, Any]) -> str:
    parts = [
        str(ir.get("kind") or ""),
        str(ir.get("expr") or ir.get("expression") or ir.get("formula") or ""),
        str(ir.get("applies_when") or ir.get("guard") or ""),
        str(ir.get("source") or ""),
        " ".join(str(value) for value in ir.get("field_names") or [] if str(value).strip()),
    ]
    try:
        parts.append(json.dumps(ir, ensure_ascii=True, sort_keys=True))
    except Exception:
        pass
    return " ".join(part for part in parts if part)
