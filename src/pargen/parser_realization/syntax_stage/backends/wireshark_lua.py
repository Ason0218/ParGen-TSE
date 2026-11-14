from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..backend_base import SyntaxBackend
from ..schema import SyntaxArtifacts, SyntaxResult
from ...target_platforms.wireshark_lua_syntax_tester import (
    WiresharkLuaSyntaxTester,
)


_WIRESHARK_LUA_EXTRA_LLM_RULES = """
Wireshark-Lua repair rules (must follow):
- Proto abbrev/filter_name must exactly match the protocol under test (filter_name).
- Define ProtoFields and assign them to proto.fields.
- In dissector(tvbuf, pinfo, tree), check tvbuf:len() before every read.
- Parse fields using valid tvbuf APIs (uint/le_uint/string, etc.).
- Register to the correct DissectorTable (tcp.port/udp.port) using the configured port(s).
- Use Proto("abbrev", "desc") to create protos (Proto.create(...) is NOT a valid API).
- Do NOT use TreeItem:add_be(...). Wireshark Lua has TreeItem:add(...) (big-endian/default) and TreeItem:add_le(...).
- Do NOT require("expert"). Use built-in PI_* group/severity constants with TreeItem:add_expert_info(...).
- Do NOT call DissectorTable.get("tcp"):add_heuristic(...). For heuristics use proto:register_heuristic("tcp", func),
  or skip heuristic registration if tcp.port/udp.port registration is already present.
- For TLV or typed value protocols, do not stop at a generic bytes/value field when the Stage B tree names
  selector-specific child fields. Parse known selector variants into bounded leaf ProtoFields and keep any raw
  fallback gated to unknown selector values.
""".strip()


def _tshark_load_errors(entry: Path, load_cfg: Dict[str, Any], logs_root: Path) -> list[str]:
    """Load the plugin in tshark and dissect the first packets (isolated plugin dir); return Lua errors
    raised at load time ("Lua: Error during loading") or while dissecting ("Lua Error: ..." dissector bug)."""
    import os
    import subprocess
    import tempfile

    pcap = load_cfg.get("pcap_path")
    tshark = load_cfg.get("tshark") or "tshark"
    if not pcap:
        return []
    with tempfile.TemporaryDirectory(prefix="bs_lua_load_") as isolated:
        env = dict(os.environ, WIRESHARK_PLUGIN_DIR=isolated)
        proc = subprocess.run([str(tshark), "-X", f"lua_script:{entry}", "-r", str(pcap), "-c", "5", "-V"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out = (proc.stderr or "") + "\n" + (proc.stdout or "")
    try:
        logs_root.mkdir(parents=True, exist_ok=True)
        with (logs_root / "wireshark_lua_load_check.log").open("a", encoding="utf-8") as handle:
            handle.write(f"$ {tshark} -X lua_script:{entry} -r {pcap} -c 5 -V\n{out}\n")
    except Exception:
        pass
    if "Lua: Error" not in out and "Lua Error:" not in out:
        return []
    return [line.strip() for line in out.splitlines() if "Lua" in line and "Error" in line][:20]


def _tree_id(value: Any) -> str:
    return str(value).strip()


def _tree_node_label(node: Dict[str, Any]) -> str:
    return str(node.get("name") or node.get("label") or node.get("field_name") or node.get("node_id") or "").strip()


def _tree_node_kind(node: Dict[str, Any]) -> str:
    return str(node.get("node_type") or node.get("type") or "").strip().lower()


def _load_lua_tree_graph(tree_path: Any) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    if not tree_path:
        return {}, {}, []
    try:
        path = Path(str(tree_path))
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}, []
    if not isinstance(payload, dict):
        return {}, {}, []
    for _ in range(4):
        if isinstance(payload.get("nodes"), list):
            break
        nested = next(
            (
                payload.get(key)
                for key in ("protocol_tree", "format_tree", "tree")
                if isinstance(payload.get(key), dict)
            ),
            None,
        )
        if nested is None:
            break
        payload = nested
    nodes_raw = payload.get("nodes")
    if not isinstance(nodes_raw, list):
        return {}, {}, []
    nodes: dict[str, dict[str, Any]] = {}
    for node in nodes_raw:
        if not isinstance(node, dict):
            continue
        raw_id = node.get("node_id", node.get("id"))
        if raw_id is None:
            continue
        nodes[_tree_id(raw_id)] = node

    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node_id, node in nodes.items():
        raw_children = node.get("children_ids") or node.get("children") or []
        if isinstance(raw_children, list):
            children[node_id] = [_tree_id(cid) for cid in raw_children if _tree_id(cid) in nodes]

    edges_raw = payload.get("edges") or payload.get("relations") or []
    edges = [edge for edge in edges_raw if isinstance(edge, dict)] if isinstance(edges_raw, list) else []
    return nodes, children, edges


def _lua_tree_path_to(node_id: str, nodes: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    current = str(node_id)
    while current in nodes and current not in seen and len(parts) < 24:
        seen.add(current)
        node = nodes[current]
        label = _tree_node_label(node)
        if label:
            parts.append(label)
        parent = node.get("parent_id")
        if parent is None:
            break
        current = _tree_id(parent)
    return ".".join(reversed(parts))


def _append_lua_tree_subtree_lines(
    *,
    lines: list[str],
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    children: dict[str, list[str]],
    depth: int,
    max_depth: int,
    max_lines: int,
) -> None:
    if len(lines) >= max_lines or depth > max_depth or node_id not in nodes:
        return
    node = nodes[node_id]
    indent = "  " * max(0, depth)
    size = node.get("size_bits")
    size_s = f" size_bits={size}" if size is not None else ""
    lines.append(f"{indent}- {_tree_node_label(node)} [{_tree_node_kind(node) or 'node'}]{size_s}")
    for child_id in children.get(node_id, [])[:12]:
        _append_lua_tree_subtree_lines(
            lines=lines,
            node_id=str(child_id),
            nodes=nodes,
            children=children,
            depth=depth + 1,
            max_depth=max_depth,
            max_lines=max_lines,
        )


def _lua_structured_body_context(backend_config: Optional[Dict[str, Any]], *, max_lines: int = 150) -> str:
    cfg = backend_config if isinstance(backend_config, dict) else {}
    tree_path = cfg.get("format_tree_path") or cfg.get("tree_path") or cfg.get("parse_ir_path")
    nodes, children, edges = _load_lua_tree_graph(tree_path)
    if not nodes:
        return ""

    selector_edges = [
        edge
        for edge in edges
        if str(edge.get("rel") or "").strip() == "condition_on"
        and _tree_id(edge.get("src")) in nodes
        and _tree_id(edge.get("dst")) in nodes
    ]
    structured_roots = [
        node_id
        for node_id, node in nodes.items()
        if _tree_node_kind(node) == "variant" and children.get(node_id)
    ]
    if not selector_edges and not structured_roots:
        return ""

    lines: list[str] = [
        "Wireshark-Lua structured Parse-IR context (derived from Stage B tree; use as repair guidance, not protocol-specific code):",
        "- Known selector variants must become bounded Lua parsing branches; do not leave known variant bodies as only a raw value/bytes field.",
        "- For each leaf under a selected variant, define a ProtoField, add it to proto.fields, and emit subtree:add(...) at the leaf span.",
        "- Unknown selector values may keep a raw fallback, but the fallback must not swallow bytes for known selector values.",
    ]
    if selector_edges:
        lines.append("- Selector obligations:")
        for edge in selector_edges[:20]:
            src = _tree_id(edge.get("src"))
            dst = _tree_id(edge.get("dst"))
            formula = str(edge.get("formula") or edge.get("condition") or "").strip()
            lines.append(
                f"  * {_tree_node_label(nodes[src])} --{formula or 'condition'}--> "
                f"{_tree_node_label(nodes[dst])} path={_lua_tree_path_to(dst, nodes)}"
            )
            if len(lines) >= max_lines:
                break

    if structured_roots and len(lines) < max_lines:
        lines.append("- Structured variant/body neighborhoods:")
        seen: set[str] = set()
        for root_id in structured_roots[:12]:
            if root_id in seen:
                continue
            seen.add(root_id)
            lines.append(f"  path={_lua_tree_path_to(root_id, nodes)}")
            _append_lua_tree_subtree_lines(
                lines=lines,
                node_id=root_id,
                nodes=nodes,
                children=children,
                depth=2,
                max_depth=7,
                max_lines=max_lines,
            )
            if len(lines) >= max_lines:
                break

    return "\n".join(lines[:max_lines]).strip()


def _find_matching_paren(text: str, open_idx: int) -> Optional[int]:
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


class WiresharkLuaBackend(SyntaxBackend):
    name = "wireshark_lua"

    def __init__(
        self,
        *,
        lua_entry: Optional[Path] = None,
        luac_path: Optional[Path] = None,
        lua_path: Optional[Path] = None,
    ) -> None:
        self.lua_entry = lua_entry
        self.luac_path = luac_path
        self.lua_path = lua_path

    def check(
        self,
        *,
        plugin_src_dir: Path,
        protocol_filter_name: str,
        file_prefix: str,
        logs_root: Path,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> SyntaxResult:
        tester = WiresharkLuaSyntaxTester(
            plugin_src_dir=plugin_src_dir,
            lua_entry=self.lua_entry,
            filter_name=protocol_filter_name,
            luac_path=self.luac_path,
            lua_path=self.lua_path,
            logs_root=logs_root,
        )
        result = tester.run_syntax_check()

        artifacts = SyntaxArtifacts(
            tool_name=self.name,
            plugin_install_dir=plugin_src_dir,
            extra={
                "lua_entry": str(result.entry),
                "lua_tool": result.used_tool,
            },
        )

        errors = list(result.errors or [])
        ok = bool(result.ok)
        load_cfg = (backend_config or {}).get("tshark_load_check")
        if ok and load_cfg:
            # luac only compiles; ProtoField/API misuse surfaces when tshark loads the plugin.
            # A load error is reported like a syntax error so the same bounded repair loop fixes it.
            load_errors = _tshark_load_errors(result.entry, load_cfg, logs_root)
            if load_errors:
                ok = False
                errors = load_errors

        return SyntaxResult(
            ok=ok,
            runtime_ok=None,
            errors=errors,
            runtime_errors=[],
            build_log=result.log_path,
            runtime_log=None,
            artifacts=artifacts,
            build_rc=result.rc,
            runtime_rc=None,
        )


    def extra_llm_rules(
        self,
        *,
        protocol_filter_name: str,
        file_prefix: str,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = _lua_structured_body_context(backend_config)
        if not context:
            return _WIRESHARK_LUA_EXTRA_LLM_RULES
        return _WIRESHARK_LUA_EXTRA_LLM_RULES + "\n\n" + context
