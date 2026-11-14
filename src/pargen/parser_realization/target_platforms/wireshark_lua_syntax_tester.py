from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LuaSyntaxTestResult:
    ok: bool
    rc: int
    errors: List[str]
    log_path: Optional[Path]
    entry: Path
    used_tool: str


def resolve_lua_entry(
    plugin_src_dir: Path,
    filter_name: Optional[str] = None,
    *,
    lua_entry: Optional[Path] = None,
) -> Path:
    if lua_entry is not None:
        entry = Path(lua_entry)
        if not entry.is_absolute():
            entry = plugin_src_dir / entry
        if not entry.exists():
            raise FileNotFoundError(f"Lua entry not found: {entry}")
        return entry.resolve()

    candidates: List[str] = []
    if filter_name:
        name = str(filter_name).strip()
        if name:
            if name.endswith(".lua"):
                candidates.append(name)
            else:
                candidates.append(f"{name}.lua")
            # Common ParGen convention: filter_name ends with "_bs" but the entry file is
            # named after the protocol (e.g., filter_name="tcp_bs" -> "tcp.lua").
            if name.endswith("_bs"):
                base = name[:-3]
                candidates.extend([f"{base}.lua", f"packet-{base}.lua", f"packet_{base}.lua"])

    candidates.extend(["dissector.lua", "plugin.lua", "init.lua"])

    for name in candidates:
        path = plugin_src_dir / name
        if path.exists() and path.is_file():
            return path.resolve()

    lua_files = sorted([p for p in plugin_src_dir.glob("*.lua") if p.is_file()])
    # Stage C may emit helper/build scripts (e.g., build.lua) alongside the dissector.
    # Prefer a single non-build Lua file when possible.
    non_entry_names = {"build.lua"}
    non_entry_suffixes = ("_utils.lua", "_util.lua", "_helpers.lua", "_helper.lua")
    entry_candidates = [
        p
        for p in lua_files
        if p.name not in non_entry_names and not any(p.name.endswith(suf) for suf in non_entry_suffixes)
    ]
    if len(entry_candidates) == 1:
        return entry_candidates[0].resolve()

    if not lua_files:
        raise FileNotFoundError(
            f"No Lua sources found under {plugin_src_dir}. Provide --stage-c-lua-entry to select an entry file."
        )
    if not entry_candidates:
        names = ", ".join(p.name for p in lua_files)
        ignored = ", ".join(sorted(non_entry_names))
        raise FileNotFoundError(
            f"No Lua dissector entry found under {plugin_src_dir} (found: {names}; ignored: {ignored}). "
            "Provide --stage-c-lua-entry to select the entry file."
        )
    names = ", ".join(p.name for p in lua_files)
    raise ValueError(
        f"Multiple Lua sources found under {plugin_src_dir} ({names}). "
        "Provide --stage-c-lua-entry to select the entry file."
    )


class WiresharkLuaSyntaxTester:
    def __init__(
        self,
        *,
        plugin_src_dir: Path,
        lua_entry: Optional[Path] = None,
        filter_name: Optional[str] = None,
        luac_path: Optional[Path] = None,
        lua_path: Optional[Path] = None,
        logs_root: Optional[Path] = None,
    ) -> None:
        self.plugin_src_dir = plugin_src_dir
        self.lua_entry = lua_entry
        self.filter_name = filter_name
        self.luac_path = luac_path
        self.lua_path = lua_path
        self.logs_root = logs_root

    def run_syntax_check(self) -> LuaSyntaxTestResult:
        entry = resolve_lua_entry(
            self.plugin_src_dir,
            self.filter_name,
            lua_entry=self.lua_entry,
        )

        luac = self.luac_path or shutil.which("luac")
        lua = self.lua_path or shutil.which("lua")

        used_tool = None
        cmd: List[str] = []
        if luac:
            used_tool = str(luac)
            cmd = [str(luac), "-p", str(entry)]
        elif lua:
            used_tool = str(lua)
            cmd = [str(lua), "-e", "assert(loadfile(arg[1]))", str(entry)]
        else:
            raise RuntimeError("Missing luac/lua; install Lua or pass --stage-c-luac/--stage-c-lua.")

        logs_root = self.logs_root or (self.plugin_src_dir / "logs")
        logs_root.mkdir(parents=True, exist_ok=True)
        log_path = logs_root / "wireshark_lua_syntax.log"

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"$ {' '.join(cmd)}\n")
                if proc.stdout:
                    handle.write(proc.stdout)
                if proc.stderr:
                    handle.write("\n[stderr]\n")
                    handle.write(proc.stderr)
                handle.write("\n")
        except Exception:
            pass

        rc = int(proc.returncode)
        ok = rc == 0
        errors: List[str] = []
        if not ok:
            err_lines = (proc.stderr or "").splitlines()
            if not err_lines:
                err_lines = (proc.stdout or "").splitlines()
            if err_lines:
                errors = err_lines[-20:]
            else:
                errors = [f"{used_tool} syntax check failed with rc={rc}"]

        return LuaSyntaxTestResult(
            ok=ok,
            rc=rc,
            errors=errors,
            log_path=log_path,
            entry=entry,
            used_tool=str(used_tool or ""),
        )
