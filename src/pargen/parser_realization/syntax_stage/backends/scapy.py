from __future__ import annotations

import importlib.util
import ast
import json
import py_compile
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backend_base import SyntaxBackend
from ..schema import SyntaxArtifacts, SyntaxResult


_SCAPY_EXTRA_LLM_RULES = """
Scapy repair rules (must follow):
- Output must be valid Python 3 code and must be importable (no ImportError/SyntaxError).
- Define protocol layers as `class Xxx(Packet)` with `fields_desc = [...]` using Scapy Field objects.
- Do not do file IO or network operations at import time; only class definitions + bind_layers.
- Do not import or delegate to Scapy's built-in/contrib parser for the target protocol (for example
  `scapy.layers.<protocol>` or `scapy.contrib.<protocol>`). Use only generic Scapy Packet/Field APIs.
- If you register on TCP/UDP ports, register BOTH directions using bind_layers(..., dport=PORT) and bind_layers(..., sport=PORT).
- Any custom `Field.getfield(self, pkt, s)` implementation MUST return a tuple `(remaining_bytes, parsed_value)` (not the other way around).
- For Packet classes used as `PacketListField` elements, define `extract_padding(self, s): return b"", s` unless there is an
  explicit schema-defined Scapy payload layer. Otherwise one list element may absorb bytes for later elements/sections.
- For Packet classes embedded with `PacketField` before later fields, also define `extract_padding(self, s): return b"", s`.
  Otherwise Scapy stores the remaining bytes as the child packet payload and starves later fields.
- Never pass a `Packet` class directly to `ConditionalField`. The first argument must be a Scapy Field object, for example
  `ConditionalField(PacketLenField("body", None, BodyPacket, length_from=...), lambda pkt: ...)` or
  `ConditionalField(PacketListField("items", [], ItemPacket, length_from=...), lambda pkt: ...)`.
- Reportable Scapy parser fields must be exposed through public Scapy Packet/Field APIs. Do not rely on
  ParGen-specific reporting hooks.
- If a message body is defined as a TLV/IE/item list, parse the body as repeated TLV wrappers first. Do not place
  named IE bodies directly in the message body unless the Parse-IR says those fields are not length-tagged.
- Every `self.setfieldval("name", ...)` target must be declared as a public Scapy field in that packet's
  `fields_desc`; setting undeclared fields causes Scapy dissection to fail and fall back to Raw.
- Do not call `super(...).do_dissect()` on a sliced/truncated buffer such as `s[:4]`; Scapy will parse every
  declared field in `fields_desc` and can fail before the custom parser reaches the body.
- Do not attach message bodies as generic `Packet(...)` or `Raw(...)` payloads when the Parse-IR names
  body fields or TLV variants. Use a concrete Packet/PacketListField/custom Field parser.
- When the Parse-IR has structured repeated bodies (TLV/IE/options/resource-records), do not expose the
  body as one generic `body`/`payload`/`data`/`value` string field. Parse it into PacketListField/custom Field
  elements that expose the repeated item and variant leaf fields.
- For selector-controlled variants from the Parse-IR, dispatch known selector values to concrete generated Packet
  classes (for example by a custom Field.getfield parser, MultipleTypeField, or guess_payload_class). Generic fallback
  bytes are allowed only for unknown selector values and must not swallow known variant fields.
- Do not invent selector numbers or IE/option type mappings that are not present in the Parse-IR context. If the
  Parse-IR selector is a message type, dispatch the message body by that message type instead of guessing nested
  TLV/IE type codes.
- If you use Scapy `MultipleTypeField`, each choice tuple must be `(FieldObjectWithSameName, condition_callable)`.
  Do not put the lambda/condition first. When in doubt, prefer a small custom Field/PacketLenField subclass.
- Consecutive `BitField`/`BitEnumField` declarations must end on a byte boundary before any non-bit field, and at
  the end of `fields_desc`. Add explicit reserved/spare bit fields instead of leaving a partial byte open.
- In `post_dissect(self, s)`, `s` is the residual bytes left after declared fields have consumed input; do not use
  `len(s)` to validate the byte length of a `StrLenField`/payload field that has already been parsed. Compare the
  expected length against the parsed field value length instead.
- Do not use `self.underlayer.payload.original` to validate this layer's declared packet length; during Scapy
  dissection it is layer-order dependent. Use this packet's own `self.original` bytes or a declared parsed field.
""".strip()


def _resolve_scapy_entry(
    plugin_src_dir: Path,
    filter_name: str,
    *,
    entry_override: Optional[Path] = None,
) -> Path:
    if entry_override:
        override = Path(entry_override)
        if not override.is_absolute():
            override = plugin_src_dir / override
        if override.exists() and override.is_file():
            return override.resolve()
        raise FileNotFoundError(f"Scapy entry override does not exist: {override}")

    candidates = [
        "dissector.py",
        "scapy_layer.py",
        "plugin.py",
        "init.py",
        f"{str(filter_name or '').strip()}.py",
    ]
    for name in candidates:
        if not name or name == ".py":
            continue
        path = plugin_src_dir / name
        if path.exists() and path.is_file():
            return path.resolve()

    py_files = sorted([p for p in plugin_src_dir.glob("*.py") if p.is_file()])
    if len(py_files) == 1:
        return py_files[0].resolve()

    listing = ", ".join([p.name for p in py_files]) if py_files else "<none>"
    raise RuntimeError(
        "Unable to resolve Scapy entry file. "
        f"Found Python files under {plugin_src_dir}: {listing}. "
        "Pass an explicit entry via --stage-c-scapy-entry."
    )


def _validate_scapy_packet_fields(mod: Any) -> List[str]:
    """
    Catch Scapy constructs that import successfully but cannot actually dissect packets.

    A common LLM mistake is `ConditionalField(PacketSubclass, condition)`. Scapy expects
    `ConditionalField` to wrap a Field instance, not a Packet class. The module imports, but
    dissection later fails or silently falls back to Raw, which makes syntax look healthy while
    export metrics collapse.
    """

    errors: List[str] = []
    try:
        from scapy.fields import Field  # type: ignore[import-not-found]
        from scapy.packet import Packet  # type: ignore[import-not-found]
    except Exception:
        return errors

    def _is_packet_class(value: Any) -> bool:
        try:
            return isinstance(value, type) and issubclass(value, Packet) and value is not Packet
        except Exception:
            return False

    for class_name, obj in sorted(vars(mod).items()):
        if not _is_packet_class(obj):
            continue
        for idx, desc in enumerate(list(getattr(obj, "fields_desc", []) or [])):
            if _is_packet_class(desc):
                errors.append(
                    f"[SCAPY_API] {class_name}.fields_desc[{idx}] is Packet class "
                    f"{desc.__name__}; wrap nested packets in PacketField, PacketLenField, "
                    "PacketListField, or bind_layers/guess_payload_class."
                )
                continue
            inner = getattr(desc, "fld", None)
            if _is_packet_class(inner):
                errors.append(
                    f"[SCAPY_API] {class_name}.fields_desc[{idx}] wraps Packet class "
                    f"{inner.__name__} in {desc.__class__.__name__}; ConditionalField and "
                    "similar wrappers must wrap a Scapy Field instance such as "
                    "PacketLenField(...), PacketListField(...), or PacketField(...)."
                )
                continue
            if inner is not None and not isinstance(inner, Field):
                wrapper_name = desc.__class__.__name__
                if wrapper_name == "ConditionalField":
                    errors.append(
                        f"[SCAPY_API] {class_name}.fields_desc[{idx}] has ConditionalField "
                        f"inner object {inner!r}, not a Scapy Field instance."
                    )
    return errors


_SCAPY_RAW_BODY_FIELD_NAMES = {
    "body",
    "payload",
    "data",
    "raw",
    "value",
    "values",
    "ie_data",
    "iedata",
    "ie_value",
    "ievalue",
    "information_elements",
    "ies",
    "options",
    "rdata",
    "rrdata",
}

_SCAPY_RAW_BODY_FIELD_CALLS = {
    "StrLenField",
    "StrFixedLenField",
    "StrField",
    "XStrLenField",
    "XStrFixedLenField",
    "XStrField",
    "PacketLenField",
}

_STRUCTURED_REMAINDER_MARKERS = (
    "tlv",
    "type_length_value",
    "typelengthvalue",
    "information_element",
    "informationelement",
    "informationelements",
    "ie_body",
    "iebody",
    "option",
    "options",
    "resource_record",
    "resourcerecord",
    "question",
    "answer",
    "authority",
    "additional",
)

_SCAPY_GENERATED_TAIL_MARKERS = (
    "# __PARGEN_SCAPY_RUNTIME_HELPERS__: begin",
    "def _pargen_scapy_append_violation",
    "import os as _bs_os",
    "class ParGenConstraintWarning",
    "_BS_CONSTRAINTS =",
)


def _norm_scapy_name(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _unwrap_protocol_tree_payload(payload: Any) -> Any:
    """Return the graph-like tree object regardless of local wrapper shape."""

    cur = payload
    for _ in range(4):
        if not isinstance(cur, dict):
            break
        for key in ("protocol_tree", "format_tree", "tree"):
            if isinstance(cur.get(key), dict):
                cur = cur.get(key)
                break
        else:
            break
    return cur


def _scapy_call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _scapy_unwrap_field_call(node: ast.AST) -> Optional[ast.Call]:
    if not isinstance(node, ast.Call):
        return None
    if _scapy_call_name(node) == "ConditionalField" and node.args and isinstance(node.args[0], ast.Call):
        return node.args[0]
    return node


def _scapy_packet_arg_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _tree_requires_structured_scapy_body(tree_path: Any) -> bool:
    if not tree_path:
        return False
    try:
        path = Path(str(tree_path)).expanduser()
        payload = _unwrap_protocol_tree_payload(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return False

    names: Dict[int, str] = {}
    child_count: Dict[int, int] = {}
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
        for node in payload.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            raw_id = node.get("node_id", node.get("id"))
            try:
                node_id = int(raw_id)
            except Exception:
                continue
            label = node.get("name") or node.get("field_name") or node.get("label") or ""
            names[node_id] = str(label)
            children = node.get("children_ids") or node.get("children") or []
            if isinstance(children, list):
                child_count[node_id] = max(child_count.get(node_id, 0), len(children))
        for edge in payload.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            raw_src = edge.get("src", edge.get("source"))
            try:
                src = int(raw_src)
            except Exception:
                continue
            child_count[src] = child_count.get(src, 0) + 1
    else:
        def _walk(obj: Any) -> None:
            if isinstance(obj, dict):
                label = obj.get("name") or obj.get("field_name") or obj.get("label")
                raw_id = obj.get("node_id", obj.get("id"))
                if label is not None and raw_id is not None:
                    try:
                        node_id = int(raw_id)
                        names[node_id] = str(label)
                        children = obj.get("children") or obj.get("children_ids") or []
                        if isinstance(children, list):
                            child_count[node_id] = max(child_count.get(node_id, 0), len(children))
                    except Exception:
                        pass
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        _walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)

        _walk(payload)

    for node_id, label in names.items():
        norm = _norm_scapy_name(label)
        if any(marker in norm for marker in _STRUCTURED_REMAINDER_MARKERS):
            if child_count.get(node_id, 0) > 0:
                return True
    return False


def _load_tree_graph(tree_path: Any) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], List[Dict[str, Any]]]:
    if not tree_path:
        return {}, {}, []
    try:
        path = Path(str(tree_path)).expanduser()
        payload = _unwrap_protocol_tree_payload(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}, {}, []

    nodes: Dict[str, Dict[str, Any]] = {}
    children: Dict[str, List[str]] = {}
    edges: List[Dict[str, Any]] = []

    def _node_id(obj: Any) -> str:
        if not isinstance(obj, dict):
            return ""
        raw_id = obj.get("node_id", obj.get("id"))
        if raw_id is None:
            return ""
        return str(raw_id)

    if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
        for node in payload.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            nid = _node_id(node)
            if not nid:
                continue
            nodes[nid] = node
            raw_children = node.get("children_ids") or node.get("children") or []
            if isinstance(raw_children, list):
                children[nid] = [str(x) for x in raw_children]
        for edge in payload.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            src = edge.get("src", edge.get("source"))
            dst = edge.get("dst", edge.get("target"))
            if src is None or dst is None:
                continue
            edge = dict(edge)
            edge["src"] = str(src)
            edge["dst"] = str(dst)
            edges.append(edge)
            if edge.get("rel") in {"contains", "child", "parent_child"}:
                children.setdefault(str(src), []).append(str(dst))
    else:
        def _walk(obj: Any, parent: str = "") -> None:
            if isinstance(obj, dict):
                nid = _node_id(obj)
                if nid:
                    nodes[nid] = obj
                    raw_children = obj.get("children") or obj.get("children_ids") or []
                    if isinstance(raw_children, list):
                        for child in raw_children:
                            cid = _node_id(child) if isinstance(child, dict) else str(child)
                            if cid:
                                children.setdefault(nid, []).append(cid)
                            if isinstance(child, dict):
                                _walk(child, nid)
                    if parent:
                        children.setdefault(parent, []).append(nid)
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        _walk(value, nid or parent)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item, parent)

        _walk(payload)

    deduped: Dict[str, List[str]] = {}
    for src, dsts in children.items():
        seen: set[str] = set()
        out: List[str] = []
        for dst in dsts:
            if dst in nodes and dst not in seen and dst != src:
                out.append(dst)
                seen.add(dst)
        if out:
            deduped[src] = out
    return nodes, deduped, edges


def _tree_selector_condition_value(edge: Dict[str, Any], selector_id: str) -> Optional[int]:
    formula = str(edge.get("formula") or edge.get("condition") or "").strip()
    if not formula:
        return None
    selector_pat = re.escape(str(selector_id))
    match = re.search(rf"\bval\s*\(\s*{selector_pat}\s*\)\s*==\s*(0x[0-9a-fA-F]+|\d+)\b", formula)
    if not match:
        return None
    try:
        return int(match.group(1), 0)
    except Exception:
        return None


def _tree_selector_maps(
    backend_config: Optional[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    cfg = backend_config if isinstance(backend_config, dict) else {}
    plan_maps = _selector_maps_from_stage_c_lowering_plan(cfg)
    if plan_maps:
        return plan_maps
    tree_path = cfg.get("format_tree_path") or cfg.get("tree_path") or cfg.get("parse_ir_path")
    nodes, children, edges = _load_tree_graph(tree_path)
    if not nodes:
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        if str(edge.get("rel") or "").strip() != "condition_on":
            continue
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src not in nodes or dst not in nodes:
            continue
        value = _tree_selector_condition_value(edge, src)
        if value is None:
            continue
        selector = nodes[src]
        variant = nodes[dst]
        if _norm_scapy_name(_tree_node_kind(selector)) != "selector":
            continue
        if _norm_scapy_name(_tree_node_kind(variant)) != "variant":
            continue
        out.setdefault(src, []).append(
            {
                "selector_id": src,
                "selector_name": _tree_node_label(selector),
                "value": int(value),
                "variant_id": dst,
                "variant_name": _tree_node_label(variant),
                "variant_children": len(children.get(dst, [])),
                "path": _tree_path_to(dst, nodes),
            }
        )
    for entries in out.values():
        entries.sort(key=lambda x: (int(x.get("value", 0)), str(x.get("variant_name") or "")))
    return out


def _selector_maps_from_stage_c_lowering_plan(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    plan = cfg.get("stage_c_lowering_plan")
    if not isinstance(plan, dict):
        return {}
    raw_dispatches = plan.get("selector_dispatches")
    if not isinstance(raw_dispatches, list):
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for raw in raw_dispatches:
        if not isinstance(raw, dict):
            continue
        selector_id = raw.get("selector_id")
        selector_name = str(raw.get("selector_name") or "").strip() or "selector"
        variant_id = raw.get("variant_id")
        variant_name = str(raw.get("variant_name") or "").strip()
        raw_value = raw.get("selector_value")
        try:
            selector_key = str(int(selector_id))
            value = int(raw_value)
        except Exception:
            continue
        fields = raw.get("fields")
        if not isinstance(fields, list):
            fields = []
        childful_fields = [
            field
            for field in fields
            if isinstance(field, dict)
            and str(field.get("name") or "").strip()
            and str(field.get("size_bits") or "").strip() not in {"", "0", "None"}
        ]
        path = str(raw.get("path") or "").strip()
        if not path:
            field_paths = [str(field.get("path") or "").strip() for field in childful_fields if isinstance(field, dict)]
            field_paths = [field_path for field_path in field_paths if field_path]
            if field_paths:
                path = field_paths[0].rsplit(".", 1)[0]
        out.setdefault(selector_key, []).append(
            {
                "selector_id": selector_key,
                "selector_name": selector_name,
                "value": value,
                "variant_id": str(variant_id) if variant_id is not None else str(value),
                "variant_name": variant_name or f"{selector_name}_{value}",
                "variant_children": len(childful_fields),
                "path": path,
            }
        )
    for entries in out.values():
        entries.sort(key=lambda x: (int(x.get("value", 0)), str(x.get("variant_name") or "")))
    return out


def _tree_variant_search_terms(*, selector_name: str, variant_name: str, value: int) -> List[str]:
    raw = str(variant_name or "").strip()
    if not raw:
        return []
    terms = {_norm_scapy_name(raw)}
    words = [w for w in re.split(r"[^0-9A-Za-z]+", raw) if w]
    selector_words = [w for w in re.split(r"[^0-9A-Za-z]+", str(selector_name or "")) if w]
    if words and selector_words:
        i = 0
        while i < len(words) and i < len(selector_words) and words[i].lower() == selector_words[i].lower():
            i += 1
        if i < len(words) and words[i].isdigit() and int(words[i]) == int(value):
            i += 1
        tail = "".join(words[i:])
        if tail:
            terms.add(_norm_scapy_name(tail))
    for idx, word in enumerate(words):
        if word.isdigit() and int(word) == int(value):
            tail = "".join(words[idx + 1 :])
            if tail:
                terms.add(_norm_scapy_name(tail))
            break
    return sorted(t for t in terms if len(t) >= 3)


def _scapy_attr_or_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _scapy_selector_eq_values(expr: ast.AST) -> List[tuple[str, int]]:
    out: List[tuple[str, int]] = []
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        if not isinstance(node.ops[0], ast.Eq):
            continue
        left = node.left
        right = node.comparators[0]
        field = _scapy_attr_or_name(left)
        value_node = right
        if isinstance(left, ast.Constant) and isinstance(left.value, int):
            field = _scapy_attr_or_name(right)
            value_node = left
        if not field or not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, int):
            continue
        out.append((field, int(value_node.value)))
    return out


def _scapy_module_int_class_maps(tree: ast.Module) -> Dict[str, set[int]]:
    out: Dict[str, set[int]] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Dict):
            continue
        map_names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        if not map_names:
            continue
        values: set[int] = set()
        for key, val in zip(stmt.value.keys, stmt.value.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, int):
                continue
            if not isinstance(val, (ast.Name, ast.Attribute)):
                continue
            values.add(int(key.value))
        if values:
            for name in map_names:
                out[name] = set(values)
    return out


def _extract_scapy_selector_dispatch_values(tree: ast.Module) -> Dict[str, set[int]]:
    dispatch_values: Dict[str, set[int]] = {}
    module_maps = _scapy_module_int_class_maps(tree)
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        call_name = _scapy_call_name(call)
        if call_name == "MultipleTypeField":
            if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
                continue
            for choice in call.args[0].elts:
                if not isinstance(choice, ast.Tuple) or len(choice.elts) < 2:
                    continue
                condition = choice.elts[1]
                if isinstance(condition, ast.Lambda):
                    condition = condition.body
                for field, value in _scapy_selector_eq_values(condition):
                    dispatch_values.setdefault(_norm_scapy_name(field), set()).add(int(value))
        elif call_name == "ParGenSelectorDispatchField":
            selector_from = ""
            values: set[int] = set()
            for kw in call.keywords:
                if kw.arg == "selector_from" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    selector_from = kw.value.value
                elif kw.arg == "dispatch" and isinstance(kw.value, ast.Dict):
                    for key in kw.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, int):
                            values.add(int(key.value))
                elif kw.arg == "dispatch_from" and isinstance(kw.value, ast.Lambda):
                    body = kw.value.body
                    if isinstance(body, ast.Name):
                        values.update(module_maps.get(body.id, set()))
            if selector_from and values:
                dispatch_values.setdefault(_norm_scapy_name(selector_from), set()).update(values)
    return dispatch_values


def _tree_node_label(node: Dict[str, Any]) -> str:
    return str(node.get("name") or node.get("field_name") or node.get("label") or "").strip()


def _tree_node_kind(node: Dict[str, Any]) -> str:
    return str(node.get("node_type") or node.get("type") or "").strip()


def _tree_node_brief(node_id: str, nodes: Dict[str, Dict[str, Any]]) -> str:
    node = nodes.get(str(node_id), {})
    label = _tree_node_label(node) or f"node_{node_id}"
    kind = _tree_node_kind(node)
    size = node.get("size_bits", node.get("size", ""))
    constraints = node.get("constraints") or []
    parts = [f"{label}#{node_id}"]
    if kind:
        parts.append(kind)
    if size not in ("", None):
        parts.append(f"size={size}")
    if isinstance(constraints, list) and constraints:
        parts.append("constraints=" + "; ".join(str(x) for x in constraints[:3]))
    return " ".join(parts)


def _tree_path_to(node_id: str, nodes: Dict[str, Dict[str, Any]]) -> str:
    parent_by_child: Dict[str, str] = {}
    for nid, node in nodes.items():
        parent = node.get("parent_id", node.get("parent"))
        if parent is not None:
            parent_by_child[str(nid)] = str(parent)
    chain: List[str] = []
    cur = str(node_id)
    seen: set[str] = set()
    while cur and cur not in seen and cur in nodes and len(chain) < 8:
        seen.add(cur)
        chain.append(_tree_node_label(nodes[cur]) or cur)
        cur = parent_by_child.get(cur, "")
    return "/".join(reversed(chain))


def _append_tree_subtree_lines(
    *,
    lines: List[str],
    node_id: str,
    nodes: Dict[str, Dict[str, Any]],
    children: Dict[str, List[str]],
    depth: int,
    max_depth: int,
    max_lines: int,
) -> None:
    if len(lines) >= max_lines or str(node_id) not in nodes:
        return
    prefix = "  " * depth
    lines.append(f"{prefix}- {_tree_node_brief(str(node_id), nodes)}")
    if depth >= max_depth:
        return
    for child_id in children.get(str(node_id), [])[:10]:
        _append_tree_subtree_lines(
            lines=lines,
            node_id=str(child_id),
            nodes=nodes,
            children=children,
            depth=depth + 1,
            max_depth=max_depth,
            max_lines=max_lines,
        )
        if len(lines) >= max_lines:
            return


def _scapy_structured_body_context(backend_config: Optional[Dict[str, Any]], *, max_chars: int = 9000) -> str:
    cfg = backend_config if isinstance(backend_config, dict) else {}
    selector_maps = _tree_selector_maps(backend_config)
    tree_path = cfg.get("format_tree_path") or cfg.get("tree_path") or cfg.get("parse_ir_path")
    nodes, children, edges = _load_tree_graph(tree_path)
    if not nodes and not selector_maps:
        return ""

    marker_ids: List[str] = []
    for node_id, node in nodes.items():
        label_norm = _norm_scapy_name(_tree_node_label(node))
        kind_norm = _norm_scapy_name(_tree_node_kind(node))
        if any(marker in label_norm for marker in _STRUCTURED_REMAINDER_MARKERS) or kind_norm in {
            "variant",
            "selector",
        }:
            if children.get(str(node_id)) or kind_norm in {"selector", "variant"}:
                marker_ids.append(str(node_id))

    selector_edges = [
        edge
        for edge in edges
        if str(edge.get("rel") or "").strip() == "condition_on"
        and str(edge.get("src", "")) in nodes
        and str(edge.get("dst", "")) in nodes
    ]
    if not marker_ids and not selector_edges and not selector_maps:
        return ""

    lines: List[str] = [
        "Scapy structured Parse-IR context (derived from Stage B tree; use it as repair guidance, not protocol-specific code):",
        "- Structured bodies must parse known selector/variant leaf fields through public Scapy fields.",
        "- Backend lowering contract: each concrete selector map below must become a length-bounded dispatch from the selector field to a concrete Packet class. Unknown selector values may use a raw fallback, but known selector values must not be swallowed by the fallback.",
    ]
    if selector_maps:
        lines.append("- Concrete selector maps available for backend lowering:")
        for entries in sorted(
            selector_maps.values(),
            key=lambda xs: (-len(xs), str(xs[0].get("selector_name") if xs else "")),
        )[:8]:
            if not entries:
                continue
            selector_name = str(entries[0].get("selector_name") or "selector")
            sample = ", ".join(
                f"{int(e['value'])}->{e['variant_name']}" for e in entries[:18] if e.get("variant_name")
            )
            suffix = f", ... total={len(entries)}" if len(entries) > 18 else f" total={len(entries)}"
            lines.append(f"  * {selector_name}: {sample}{suffix}")
    if selector_edges:
        lines.append("- Selector obligations:")
        for edge in selector_edges[:18]:
            src = str(edge.get("src"))
            dst = str(edge.get("dst"))
            formula = str(edge.get("formula") or edge.get("condition") or "").strip()
            msg_type = str(edge.get("message_type") or "").strip()
            tail = f" [{msg_type}]" if msg_type else ""
            lines.append(
                f"  * {_tree_node_brief(src, nodes)} --{formula or 'condition'}--> "
                f"{_tree_node_brief(dst, nodes)} path={_tree_path_to(dst, nodes)}{tail}"
            )

    structured_roots: List[str] = []
    for node_id in marker_ids:
        norm = _norm_scapy_name(_tree_node_label(nodes[node_id]))
        if any(marker in norm for marker in _STRUCTURED_REMAINDER_MARKERS) and children.get(node_id):
            structured_roots.append(node_id)
    if structured_roots:
        lines.append("- Structured body neighborhoods:")
        seen_roots: set[str] = set()
        for root_id in structured_roots[:8]:
            if root_id in seen_roots:
                continue
            seen_roots.add(root_id)
            lines.append(f"  path={_tree_path_to(root_id, nodes)}")
            _append_tree_subtree_lines(
                lines=lines,
                node_id=root_id,
                nodes=nodes,
                children=children,
                depth=2,
                max_depth=7,
                max_lines=140,
            )
            if len(lines) >= 140:
                break

    context = "\n".join(lines).strip()
    if len(context) > max_chars:
        context = context[:max_chars].rstrip() + "\n... [structured context truncated]"
    return context


def _tree_selector_variant_names(backend_config: Optional[Dict[str, Any]]) -> List[str]:
    cfg = backend_config if isinstance(backend_config, dict) else {}
    tree_path = cfg.get("format_tree_path") or cfg.get("tree_path") or cfg.get("parse_ir_path")
    nodes, _, edges = _load_tree_graph(tree_path)
    if not nodes:
        return []
    names: List[str] = []
    seen: set[str] = set()
    for edge in edges:
        if str(edge.get("rel") or "").strip() != "condition_on":
            continue
        dst = str(edge.get("dst", ""))
        node = nodes.get(dst)
        if not node:
            continue
        if _norm_scapy_name(_tree_node_kind(node)) != "variant":
            continue
        name = _tree_node_label(node)
        norm = _norm_scapy_name(name)
        if not name or norm.startswith("other") or norm in seen:
            continue
        seen.add(norm)
        names.append(name)
    return names


def _split_scapy_generated_tail(source: str) -> tuple[str, str]:
    text = str(source or "")
    cuts = [idx for marker in _SCAPY_GENERATED_TAIL_MARKERS if (idx := text.find(marker)) >= 0]
    if not cuts:
        return text.rstrip() + ("\n" if text.strip() else ""), ""
    cut = min(cuts)
    head = text[:cut].rstrip()
    tail = text[cut:].lstrip("\n")
    return (head + "\n" if head else ""), tail


def make_scapy_structured_repair_view(
    files: Dict[str, str],
    errors: List[str] | tuple[str, ...],
) -> Optional[Dict[str, str]]:
    """Return compact Scapy sources for structured-body repair prompts.

    Constraint/runtime injection can make generated Scapy files very large. For
    TLV/IE/repeated-body errors the useful repair target is the public Scapy
    parser surface before ParGen's generated helper tail. Sending only that
    head keeps the LLM focused while merge_scapy_structured_repair_view() puts
    the original generated tail back.
    """

    if not any("SCAPY_STRUCTURED_BODY" in str(err) for err in (errors or ())):
        return None
    compact: Dict[str, str] = {}
    changed = False
    for name, content in (files or {}).items():
        if not str(name).endswith(".py"):
            compact[name] = content
            continue
        head, tail = _split_scapy_generated_tail(content or "")
        compact[name] = head or (content or "")
        if tail:
            changed = True
    return compact if changed else None


def merge_scapy_structured_repair_view(
    *,
    original_files: Dict[str, str],
    patched_files: Dict[str, str],
) -> Dict[str, str]:
    """Merge compact Scapy repair output back with generated helper tails."""

    merged: Dict[str, str] = {}
    for name, content in (patched_files or {}).items():
        original = original_files.get(name, "")
        if not str(name).endswith(".py") or not original:
            merged[name] = content
            continue
        _, original_tail = _split_scapy_generated_tail(original)
        patched_head, patched_tail = _split_scapy_generated_tail(content or "")
        if patched_tail:
            # The model returned a complete generated-tail-bearing file; keep it.
            merged[name] = (patched_head.rstrip() + "\n\n" + patched_tail.rstrip() + "\n").lstrip("\n")
        elif original_tail:
            merged[name] = (patched_head.rstrip() + "\n\n" + original_tail.rstrip() + "\n").lstrip("\n")
        else:
            merged[name] = content
    return merged


def _scapy_disallow_generic_body_fields(backend_config: Optional[Dict[str, Any]]) -> bool:
    cfg = backend_config if isinstance(backend_config, dict) else {}
    if bool(cfg.get("disallow_generic_body_fields")):
        return True
    tree_path = cfg.get("format_tree_path") or cfg.get("tree_path") or cfg.get("parse_ir_path")
    return _tree_requires_structured_scapy_body(tree_path)


def _validate_scapy_source_api(
    tree: ast.Module,
    *,
    path_name: str,
    backend_config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    errors: List[str] = []
    disallow_generic_body = _scapy_disallow_generic_body_fields(backend_config)
    selector_variant_names = _tree_selector_variant_names(backend_config) if disallow_generic_body else []
    selector_maps = _tree_selector_maps(backend_config) if disallow_generic_body else {}
    try:
        source_norm = _norm_scapy_name(ast.unparse(tree))
    except Exception:
        source_norm = ""
    selector_dispatch_values = _extract_scapy_selector_dispatch_values(tree)
    class_defs: Dict[str, str] = {}
    used_names: set[str] = set()
    for item in ast.walk(tree):
        if isinstance(item, ast.ClassDef):
            class_defs[_norm_scapy_name(item.name)] = item.name
            continue
        if isinstance(item, ast.Name):
            used_names.add(item.id)
        elif isinstance(item, ast.Call) and _scapy_call_name(item) in {"PacketField", "PacketLenField", "PacketListField"}:
            if len(item.args) >= 3:
                packet_arg = _scapy_packet_arg_name(item.args[2])
                if packet_arg:
                    used_names.add(packet_arg)

    def _is_call_to(node: ast.AST, func_name: str) -> bool:
        if not isinstance(node, ast.Call):
            return False
        f = node.func
        if isinstance(f, ast.Name):
            return f.id == func_name
        if isinstance(f, ast.Attribute):
            return f.attr == func_name
        return False

    def _first_str_arg(call: ast.Call) -> Optional[str]:
        if not call.args:
            return None
        arg0 = call.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            return arg0.value
        return None

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        declared: set[str] = set()
        set_targets: set[str] = set()
        field_lists: list[ast.List] = []
        for item in node.body:
            if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "fields_desc" for t in item.targets):
                if isinstance(item.value, ast.List):
                    field_lists.append(item.value)
                    for elt in item.value.elts:
                        if isinstance(elt, ast.Call):
                            name = _first_str_arg(elt)
                            if name:
                                declared.add(name)
                        elif isinstance(elt, ast.Call) and _is_call_to(elt, "ConditionalField"):
                            pass
                for call in ast.walk(item.value):
                    if isinstance(call, ast.Call) and _is_call_to(call, "ConditionalField") and call.args:
                        inner = call.args[0]
                        if isinstance(inner, ast.Name) and _norm_scapy_name(inner.id) in class_defs:
                            errors.append(
                                f"[SCAPY_API] {path_name}:{node.name}.fields_desc wraps Packet class "
                                f"{inner.id} in ConditionalField; ConditionalField must wrap a Scapy Field "
                                "instance such as PacketLenField(...), PacketListField(...), or PacketField(...)."
                            )
                        if isinstance(inner, ast.Call):
                            name = _first_str_arg(inner)
                            if name:
                                declared.add(name)
        for fields_list in field_lists:
            bit_run = 0
            for idx, elt in enumerate(fields_list.elts):
                call = elt if isinstance(elt, ast.Call) else None
                if call is not None and _is_call_to(call, "ConditionalField") and call.args and isinstance(call.args[0], ast.Call):
                    call = call.args[0]
                if call is not None and (_is_call_to(call, "BitField") or _is_call_to(call, "BitEnumField")):
                    width: Optional[int] = None
                    if len(call.args) >= 3 and isinstance(call.args[2], ast.Constant) and isinstance(call.args[2].value, int):
                        width = int(call.args[2].value)
                    for kw in call.keywords:
                        if kw.arg == "size" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                            width = int(kw.value.value)
                            break
                    if width is not None:
                        bit_run += int(width)
                    continue
                if bit_run % 8 != 0:
                    errors.append(
                        f"[SCAPY_API] {path_name}:{node.name}.fields_desc[{idx}] follows a partial "
                        f"{bit_run % 8}-bit BitField run; add explicit reserved/spare bits to reach a byte boundary."
                    )
                bit_run = 0
            if bit_run % 8 != 0:
                errors.append(
                    f"[SCAPY_API] {path_name}:{node.name}.fields_desc ends with a partial "
                    f"{bit_run % 8}-bit BitField run; add explicit reserved/spare bits to reach a byte boundary."
                )
            for idx, elt in enumerate(fields_list.elts):
                call = _scapy_unwrap_field_call(elt)
                if call is None or _scapy_call_name(call) != "MultipleTypeField":
                    continue
                if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
                    continue
                choices = call.args[0].elts
                for choice_idx, choice in enumerate(choices[:8]):
                    if not isinstance(choice, ast.Tuple) or not choice.elts:
                        continue
                    first = choice.elts[0]
                    if not isinstance(first, ast.Call):
                        errors.append(
                            f"[SCAPY_API] {path_name}:{node.name}.fields_desc[{idx}] uses MultipleTypeField "
                            f"choice[{choice_idx}] with a non-Field first element. Scapy expects "
                            "(FieldObjectWithSameName, condition_callable), not (condition, FieldObject)."
                        )
                        continue
                    first_name = _scapy_call_name(first)
                    if not first_name.endswith("Field"):
                        errors.append(
                            f"[SCAPY_API] {path_name}:{node.name}.fields_desc[{idx}] uses MultipleTypeField "
                            f"choice[{choice_idx}] whose first element is {first_name!r}, not a Scapy Field object."
                        )
            if disallow_generic_body:
                for idx, elt in enumerate(fields_list.elts):
                    call = _scapy_unwrap_field_call(elt)
                    if call is None:
                        continue
                    callee = _scapy_call_name(call)
                    field_name = _first_str_arg(call)
                    norm_field = _norm_scapy_name(field_name)
                    if callee in _SCAPY_RAW_BODY_FIELD_CALLS and norm_field in _SCAPY_RAW_BODY_FIELD_NAMES:
                        errors.append(
                            f"[SCAPY_API][SCAPY_STRUCTURED_BODY] {path_name}:{node.name}.fields_desc[{idx}] uses generic "
                            f"{callee}('{field_name}', ...) for a structured body. "
                            "The Stage B tree contains TLV/IE/option/resource-record structure; parse that body "
                            "with PacketListField/custom Field elements and expose repeated item/variant leaf fields "
                            "instead of one raw catch-all field."
                        )
                    if callee == "PacketListField" and norm_field in _SCAPY_RAW_BODY_FIELD_NAMES:
                        elem_name = _scapy_packet_arg_name(call.args[2]) if len(call.args) >= 3 else ""
                        if _norm_scapy_name(elem_name) in {_norm_scapy_name(node.name), "packet", "raw"}:
                            errors.append(
                                f"[SCAPY_API][SCAPY_STRUCTURED_BODY] {path_name}:{node.name}.fields_desc[{idx}] uses recursive/generic "
                                f"PacketListField('{field_name}', ..., {elem_name!r}) for a structured body. "
                                "Do not model TLV/IE values as another copy of the same generic wrapper; dispatch known "
                                "selector values to concrete generated Packet classes and expose their leaf fields."
                            )
        for item in node.body:
            for call in ast.walk(item):
                if not isinstance(call, ast.Call):
                    continue
                if not isinstance(call.func, ast.Attribute) or call.func.attr != "setfieldval":
                    continue
                if not isinstance(call.func.value, ast.Name) or call.func.value.id != "self":
                    continue
                target = _first_str_arg(call)
                if target:
                    set_targets.add(target)
                continue
            for call in ast.walk(item):
                if not isinstance(call, ast.Call):
                    continue
                if isinstance(call.func, ast.Attribute) and call.func.attr == "do_dissect":
                    if (
                        isinstance(call.func.value, ast.Call)
                        and isinstance(call.func.value.func, ast.Name)
                        and call.func.value.func.id == "super"
                        and call.args
                        and isinstance(call.args[0], ast.Subscript)
                    ):
                        errors.append(
                            f"[SCAPY_API] {path_name}:{node.name}.do_dissect calls super().do_dissect() on a sliced buffer; "
                            "parse with fields_desc on the full buffer or manually parse without delegating to Scapy's fixed field parser."
                        )
                if isinstance(call.func, ast.Attribute) and call.func.attr == "add_payload":
                    if not call.args:
                        continue
                    arg = call.args[0]
                    if isinstance(arg, ast.Call):
                        callee = arg.func
                        callee_name = callee.id if isinstance(callee, ast.Name) else callee.attr if isinstance(callee, ast.Attribute) else ""
                        if callee_name in {"Packet", "Raw"}:
                            errors.append(
                                f"[SCAPY_API] {path_name}:{node.name}.do_dissect attaches generic {callee_name}(...) payload; "
                                "use a concrete generated Packet/PacketListField/custom Field for body bytes."
                            )
        for target in sorted(set_targets - declared):
            errors.append(
                f"[SCAPY_API] {path_name}:{node.name}.setfieldval('{target}', ...) targets an undeclared field; "
                "declare it in fields_desc or stop using setfieldval for parser-visible data."
            )
    for variant_name in selector_variant_names:
        cls_name = class_defs.get(_norm_scapy_name(variant_name))
        if not cls_name or cls_name in used_names:
            continue
        errors.append(
            f"[SCAPY_API][SCAPY_STRUCTURED_BODY] {path_name}: generated selector variant class "
            f"{cls_name!r} for Parse-IR variant {variant_name!r} is never referenced by any Scapy "
            "PacketField/PacketListField/custom dispatch path. Wire selector-controlled bodies through "
            "concrete Packet classes; do not leave known variants as unreachable definitions behind a generic body."
        )
    for entries in selector_maps.values():
        if len(entries) < 4:
            continue
        childful_entries = [e for e in entries if int(e.get("variant_children") or 0) > 0]
        # Only force selector lowering when the Stage B tree actually gives
        # variant body fields to lower. Selector maps used only as semantic
        # labels (no variant children) should not fail Scapy syntax by
        # themselves; they are checked by semantic metrics instead.
        required = min(6, len(childful_entries))
        if required <= 0:
            continue
        selector_name = str(entries[0].get("selector_name") or "selector")
        selector_norm = _norm_scapy_name(selector_name)
        dispatched_values: set[int] = set()
        for field_norm, values in selector_dispatch_values.items():
            if field_norm == selector_norm or field_norm.endswith(selector_norm) or selector_norm.endswith(field_norm):
                dispatched_values.update(values)
        if dispatched_values:
            implemented = [entry for entry in entries if int(entry.get("value") or 0) in dispatched_values]
        else:
            implemented = []
            for entry in entries:
                terms = _tree_variant_search_terms(
                    selector_name=selector_name,
                    variant_name=str(entry.get("variant_name") or ""),
                    value=int(entry.get("value") or 0),
                )
                if any(term and term in source_norm for term in terms):
                    implemented.append(entry)
        if len(implemented) >= required:
            continue
        sample_missing = [
            f"{int(e.get('value') or 0)}->{e.get('variant_name')}"
            for e in entries
            if e not in implemented
        ][:8]
        errors.append(
            f"[SCAPY_API][SCAPY_STRUCTURED_BODY] {path_name}: Parse-IR has a concrete "
            f"{selector_name} selector map with {len(entries)} variants "
            f"({len(childful_entries)} carrying child fields), but this Scapy source exposes only "
            f"{len(implemented)} recognizable variant Packet classes/dispatch targets. "
            "This under-lowers structured selector knowledge into a low-coverage parser. "
            "Generate length-bounded dispatch entries for known selector values before the raw fallback; "
            f"missing examples: {', '.join(sample_missing)}."
        )
    return errors


class ScapyBackend(SyntaxBackend):
    name = "scapy"

    def __init__(self, *, entry_override: Optional[Path] = None) -> None:
        self.entry_override = entry_override

    def check(
        self,
        *,
        plugin_src_dir: Path,
        protocol_filter_name: str,
        file_prefix: str,
        logs_root: Path,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> SyntaxResult:
        logs_root.mkdir(parents=True, exist_ok=True)
        log_path = logs_root / "scapy_syntax.log"

        errors: List[str] = []
        compile_ok = True
        import_ok = True
        runtime_ok: Optional[bool] = None

        entry: Optional[Path] = None
        try:
            entry = _resolve_scapy_entry(
                plugin_src_dir,
                protocol_filter_name,
                entry_override=self.entry_override,
            )
        except Exception as exc:
            errors.append(f"[ENTRY] {type(exc).__name__}: {exc}")
            compile_ok = False
            import_ok = False

        py_files = sorted([p for p in plugin_src_dir.glob("*.py") if p.is_file()])

        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[ScapySyntax] plugin_src_dir={plugin_src_dir}\n")
            logf.write(f"[ScapySyntax] entry={entry}\n")
            logf.write(f"[ScapySyntax] files={[p.name for p in py_files]}\n")

            if not py_files:
                errors.append(f"[COMPILE] No Python files found under: {plugin_src_dir}")
                compile_ok = False

            # (0) Reject target-protocol built-in Scapy imports. A generated parser may use
            # generic Scapy Packet/Field APIs, but importing scapy.layers.<target> or
            # scapy.contrib.<target> delegates parsing to an external deployed parser and can
            # also steal the transport binding from the generated layer.
            target_mod = str(protocol_filter_name or file_prefix or "").strip().lower().replace("-", "_")
            target_mod = "".join(ch for ch in target_mod if ch.isalnum() or ch == "_")
            if target_mod:
                forbidden_imports = {
                    f"scapy.layers.{target_mod}",
                    f"scapy.contrib.{target_mod}",
                }
                for path in py_files:
                    try:
                        source = path.read_text(encoding="utf-8", errors="ignore")
                        tree = ast.parse(source, filename=str(path))
                    except Exception:
                        continue
                    for node in ast.walk(tree):
                        mod_name = None
                        if isinstance(node, ast.ImportFrom):
                            mod_name = node.module
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                name = str(alias.name or "")
                                if name in forbidden_imports or any(name.startswith(f"{x}.") for x in forbidden_imports):
                                    msg = (
                                        f"[IMPORT_POLICY] {path.name}: target protocol built-in Scapy import is forbidden: "
                                        f"{name}. Implement the parser with generic Scapy Packet/Field APIs instead."
                                    )
                                    errors.append(msg)
                                    logf.write(msg + "\n")
                                    compile_ok = False
                                    import_ok = False
                        if mod_name and (
                            mod_name in forbidden_imports
                            or any(str(mod_name).startswith(f"{x}.") for x in forbidden_imports)
                        ):
                            msg = (
                                f"[IMPORT_POLICY] {path.name}: target protocol built-in Scapy import is forbidden: "
                                f"{mod_name}. Implement the parser with generic Scapy Packet/Field APIs instead."
                            )
                            errors.append(msg)
                            logf.write(msg + "\n")
                            compile_ok = False
                            import_ok = False
                    api_errors = _validate_scapy_source_api(
                        tree,
                        path_name=path.name,
                        backend_config=backend_config,
                    )
                    if api_errors:
                        errors.extend(api_errors)
                        for msg in api_errors:
                            logf.write(msg + "\n")
                        compile_ok = False
                        import_ok = False

            # (1) py_compile all *.py
            for path in py_files:
                try:
                    py_compile.compile(str(path), doraise=True)
                except Exception as exc:
                    compile_ok = False
                    msg = f"[COMPILE] {path.name}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    logf.write(msg + "\n")
                    logf.write(traceback.format_exc() + "\n")

            # (2) import scapy + import entry module
            try:
                __import__("scapy")
            except Exception as exc:
                import_ok = False
                hint = "Scapy is not installed. Install it with: pip install scapy"
                msg = f"[IMPORT] {hint} ({type(exc).__name__}: {exc})"
                errors.append(msg)
                logf.write(msg + "\n")
                logf.write(traceback.format_exc() + "\n")

            if entry is not None and compile_ok and import_ok:
                module_name = f"pargen_stage_c_scapy_{protocol_filter_name}_syntax_{time.time_ns()}"
                try:
                    sys.path.insert(0, str(plugin_src_dir))
                    spec = importlib.util.spec_from_file_location(module_name, entry)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"spec_from_file_location failed for entry: {entry}")
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules.pop(module_name, None)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
                    logf.write(f"[IMPORT] ok module={module_name}\n")
                    api_errors = _validate_scapy_packet_fields(mod)
                    if api_errors:
                        import_ok = False
                        errors.extend(api_errors)
                        for msg in api_errors:
                            logf.write(msg + "\n")
                except Exception as exc:
                    import_ok = False
                    msg = f"[IMPORT] {entry.name}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    logf.write(msg + "\n")
                    logf.write(traceback.format_exc() + "\n")
                finally:
                    try:
                        if sys.path and sys.path[0] == str(plugin_src_dir):
                            sys.path.pop(0)
                    except Exception:
                        pass

            # (3) Optional runtime smoke check. Python import success is too
            # weak for Scapy: a layer can import but still miss bind_layers(),
            # select the wrong root Packet, or fail when field replay sees real
            # bytes. Keep the check bounded and feed actionable errors into the
            # same syntax repair loop.
            runtime_cfg = backend_config if isinstance(backend_config, dict) else {}
            runtime_check = bool(runtime_cfg.get("runtime_check") or runtime_cfg.get("scapy_runtime_check"))
            pcap_raw = runtime_cfg.get("pcap_path") or runtime_cfg.get("pcap")
            if compile_ok and import_ok and runtime_check:
                runtime_ok = False
                if not pcap_raw:
                    msg = "[RUNTIME_SMOKE] runtime_check enabled but backend_config has no pcap_path"
                    errors.append(msg)
                    logf.write(msg + "\n")
                else:
                    pcap_path = Path(str(pcap_raw)).expanduser()
                    if not pcap_path.exists():
                        msg = f"[RUNTIME_SMOKE] pcap not found: {pcap_path}"
                        errors.append(msg)
                        logf.write(msg + "\n")
                    else:
                        try:
                            from pargen.parser_realization.target_platforms.scapy_export import export_scapy_tooljson

                            smoke_dir = logs_root / "scapy_runtime_smoke"
                            max_packets = int(
                                runtime_cfg.get("max_runtime_packets")
                                or runtime_cfg.get("scapy_max_runtime_packets")
                                or 3
                            )
                            export_res = export_scapy_tooljson(
                                plugin_src_dir=plugin_src_dir,
                                pcap_path=pcap_path,
                                filter_name=protocol_filter_name,
                                out_dir=smoke_dir,
                                entry_override=entry,
                                ports=runtime_cfg.get("scapy_ports"),
                                transport=str(runtime_cfg.get("scapy_transport") or "auto"),
                                max_packets=max(1, max_packets),
                                include_payload_layers=True,
                                include_plugin_extra_fields=False,
                                codegen_profile=runtime_cfg,
                            )
                            records = json.loads(export_res.tooljson_path.read_text(encoding="utf-8"))
                            bad: List[str] = []
                            if not records:
                                bad.append(
                                    "no sample packets were exported; check pcap filter, transport, and bind_layers registration"
                                )
                            for rec in records[: max(1, max_packets)]:
                                if not isinstance(rec, dict):
                                    continue
                                pkt_no = rec.get("packet_no")
                                dbg = rec.get("debug") if isinstance(rec.get("debug"), dict) else {}
                                fields = rec.get("fields") if isinstance(rec.get("fields"), list) else []
                                hex_s = str(rec.get("hex") or "")
                                strategy = str(dbg.get("strategy") or "")
                                parse_cls = str(dbg.get("parse_cls") or "")
                                dbg_errors = dbg.get("errors")
                                dbg_error = dbg.get("error")
                                if strategy == "missing_plugin_layer":
                                    bad.append(
                                        f"packet {pkt_no}: generated layer missing from Scapy layer chain; "
                                        "bind the root Packet to the declared transport/ports at module import time"
                                    )
                                if dbg_error:
                                    bad.append(f"packet {pkt_no}: {dbg_error}")
                                if dbg_errors:
                                    bad.append(f"packet {pkt_no}: field replay errors: {dbg_errors}")
                                if not hex_s:
                                    bad.append(
                                        f"packet {pkt_no}: empty protocol bytes (strategy={strategy}, parse_cls={parse_cls})"
                                    )
                                if not fields:
                                    bad.append(
                                        f"packet {pkt_no}: parsed zero fields (strategy={strategy}, parse_cls={parse_cls})"
                                    )
                            if bad:
                                for msg in bad[:8]:
                                    full = f"[RUNTIME_SMOKE] {msg}"
                                    errors.append(full)
                                    logf.write(full + "\n")
                                runtime_ok = False
                            else:
                                runtime_ok = True
                                logf.write(
                                    f"[RUNTIME_SMOKE] ok packets={len(records)} tooljson={export_res.tooljson_path}\n"
                                )
                        except Exception as exc:
                            msg = f"[RUNTIME_SMOKE] {type(exc).__name__}: {exc}"
                            errors.append(msg)
                            logf.write(msg + "\n")
                            logf.write(traceback.format_exc() + "\n")
                            runtime_ok = False

        ok = bool(compile_ok and import_ok and (runtime_ok is not False))
        artifacts = SyntaxArtifacts(
            tool_name=self.name,
            plugin_install_dir=plugin_src_dir,
            extra={
                "entry": str(entry) if entry else None,
                "python": sys.executable,
            },
        )

        return SyntaxResult(
            ok=ok,
            runtime_ok=(runtime_ok if runtime_ok is not None else (bool(import_ok) if bool(compile_ok) else None)),
            errors=errors,
            runtime_errors=[],
            build_log=log_path,
            runtime_log=None,
            artifacts=artifacts,
            build_rc=(0 if ok else 1),
            runtime_rc=None,
        )


    def extra_llm_rules(
        self,
        *,
        protocol_filter_name: str,
        file_prefix: str,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = _scapy_structured_body_context(backend_config)
        if not context:
            return _SCAPY_EXTRA_LLM_RULES
        return _SCAPY_EXTRA_LLM_RULES + "\n\n" + context
