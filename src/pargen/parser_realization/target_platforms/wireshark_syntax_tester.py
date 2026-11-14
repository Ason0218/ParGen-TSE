from __future__ import annotations

import datetime
import os
import re
import shutil
import signal
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from pargen.parser_realization.fence_utils import split_fenced_files

WIRESHARK_DEFAULT_URL = "https://www.wireshark.org/download/src/wireshark-4.4.11.tar.xz"


def allocated_build_parallelism() -> int:
    """Respect the scheduler allocation instead of hard-coding four builders."""
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None:
        return 1 if os.environ.get("SLURM_JOB_ID") else 4
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


@dataclass
class SyntaxTestResult:

    ok: bool
    cmake_rc: int
    make_rc: int
    errors: List[str]
    log_path: Optional[Path]

    runtime_ok: Optional[bool] = None
    runtime_rc: Optional[int] = None
    runtime_errors: List[str] = field(default_factory=list)
    runtime_log_path: Optional[Path] = None


class WiresharkSyntaxTester:

    def __init__(
        self,
        ws_src_root: Path,
        ws_build_dir: Path,
        plugin_id: str,
        download_url: str = WIRESHARK_DEFAULT_URL,
        *,
        plugin_c_mode: str = "generated",
    ) -> None:
        self.ws_src_root = ws_src_root
        self.ws_build_dir = ws_build_dir
        self.plugin_id = plugin_id
        mode = str(plugin_c_mode or "generated").strip().lower()
        if mode not in {"generated", "from_stage_c"}:
            raise ValueError(f"Unsupported plugin_c_mode={plugin_c_mode!r} (expected 'generated' or 'from_stage_c')")
        self.plugin_c_mode = mode
        self.download_url = download_url

    def run_full_syntax_test(
        self,
        plugin_src_dir: Path,
        protocol_name: str,
        logs_root: Optional[Path] = None,
        *,
        check_runtime: bool = False,
        disable_builtin: bool = True,
    ) -> SyntaxTestResult:
        self._ensure_ws_source()
        self._ensure_build_dir()

        self._install_plugin_sources(plugin_src_dir)
        self._write_plugin_cmakelists()
        self._patch_top_cmakelists()
        if disable_builtin:
            self._disable_builtin_dissector(protocol_name)

        result = self._run_cmake_and_make(protocol_name, logs_root=logs_root)

        if result.ok and check_runtime:
            runtime_info = self._run_runtime_compatibility_test(protocol_name, logs_root=logs_root)
            result.runtime_ok = runtime_info["ok"]
            result.runtime_rc = runtime_info["rc"]
            result.runtime_errors = runtime_info["errors"]
            result.runtime_log_path = runtime_info["log_path"]
        else:
            result.runtime_ok = None
            result.runtime_rc = None
            result.runtime_errors = []
            result.runtime_log_path = None

        return result

    def _ensure_ws_source(self) -> None:
        if (self.ws_src_root / "CMakeLists.txt").exists():
            return

        self.ws_src_root.parent.mkdir(parents=True, exist_ok=True)
        tar_path = self.ws_src_root.parent / "wireshark_src.tar.xz"

        print(f"[WiresharkSyntaxTester] Wireshark source not found, downloading to {tar_path} ...")
        urllib.request.urlretrieve(self.download_url, tar_path)

        print(f"[WiresharkSyntaxTester] Extracting {tar_path} into {self.ws_src_root.parent} ...")
        with tarfile.open(tar_path, "r:xz") as tf:
            tf.extractall(self.ws_src_root.parent)

        if not (self.ws_src_root / "CMakeLists.txt").exists():
            print(
                f"[WiresharkSyntaxTester] Warning: {self.ws_src_root}/CMakeLists.txt not found after extraction. "
                f"Please ensure ws_src_root points to the actual Wireshark source directory."
            )

    def _ensure_build_dir(self) -> None:
        self.ws_build_dir.mkdir(parents=True, exist_ok=True)

    def _install_plugin_sources(self, plugin_src_dir: Path) -> None:
        ws_plugin_dir = self.ws_src_root / "plugins" / "epan" / self.plugin_id
        ws_plugin_dir.mkdir(parents=True, exist_ok=True)

        self._recover_fenced_plugin_outputs(plugin_src_dir)

        try:
            entries = sorted([p for p in plugin_src_dir.iterdir() if p.is_file() and p.suffix in {".c", ".h"}])
        except OSError:
            entries = []

        # Some Stage C outputs (or manual edits) may leave backup files with whitespace in the filename,
        # e.g., "packet-foo 2.c". CMake treats such entries as multiple tokens unless quoted, and we also
        # do not want to compile accidental duplicates. Ignore them here to keep the build deterministic.
        entries = [p for p in entries if not any(ch.isspace() for ch in p.name)]

        if self.plugin_c_mode == "generated":
            stage_c_sources = [p for p in entries if p.name != "plugin.c"]
        else:
            stage_c_sources = list(entries)
            if not any(p.name == "plugin.c" for p in stage_c_sources):
                raise FileNotFoundError(
                    f"plugin_c_mode='from_stage_c' requires plugin.c, but it was not found under {plugin_src_dir}"
                )

        if not any(p.suffix == ".c" and p.name != "plugin.c" for p in stage_c_sources):
            raise FileNotFoundError(
                f"No dissector C sources found under {plugin_src_dir} (expected at least one *.c besides plugin.c)"
            )

        # Avoid stale sources from previous runs by removing old *.c/*.h (and generated CMakeLists.txt).
        try:
            for existing in ws_plugin_dir.iterdir():
                if not existing.is_file():
                    continue
                if existing.name == "CMakeLists.txt" or existing.suffix in {".c", ".h"}:
                    try:
                        existing.unlink()
                    except OSError:
                        continue
        except OSError:
            pass

        for src in stage_c_sources:
            dst = ws_plugin_dir / src.name
            shutil.copy2(src, dst)

    def _recover_fenced_plugin_outputs(self, plugin_src_dir: Path) -> None:
        try:
            has_sources = any(
                p.is_file() and p.suffix in {".c", ".h"} and not any(ch.isspace() for ch in p.name)
                for p in plugin_src_dir.iterdir()
            )
        except OSError:
            return
        if has_sources:
            return

        raw_candidates = [plugin_src_dir / "plugin.txt", plugin_src_dir / "response.txt"]
        for raw_path in raw_candidates:
            if not raw_path.exists():
                continue
            try:
                raw_text = raw_path.read_text(encoding="utf-8")
            except OSError:
                continue
            recovered = split_fenced_files(raw_text)
            if not recovered:
                continue
            for name, content in recovered.items():
                target = plugin_src_dir / Path(name).name
                try:
                    target.write_text(content, encoding="utf-8")
                except OSError:
                    continue
            break

    def _write_plugin_cmakelists(self) -> None:
        ws_plugin_dir = self.ws_src_root / "plugins" / "epan" / self.plugin_id
        cmake_path = ws_plugin_dir / "CMakeLists.txt"

        c_sources = sorted([p.name for p in ws_plugin_dir.glob("*.c") if p.name != "plugin.c"])
        h_headers = sorted([p.name for p in ws_plugin_dir.glob("*.h")])
        if not c_sources:
            raise RuntimeError(
                f"No dissector C sources found under {ws_plugin_dir} (expected at least one *.c besides plugin.c)"
            )
        if self.plugin_c_mode == "from_stage_c":
            if not (ws_plugin_dir / "plugin.c").exists():
                raise FileNotFoundError(
                    f"plugin_c_mode='from_stage_c' requires plugin.c under {ws_plugin_dir}, but it is missing"
                )

        c_list = "\n".join(f"    {name}" for name in c_sources)
        h_list = "\n".join(f"    {name}" for name in h_headers)

        # NOTE: Do NOT list plugin.c inside DISSECTOR_SRC. Doing so duplicates plugin.c in the
        # generated build graph and can trigger circular dependencies such as:
        #   "Circular plugins/epan/<id>/plugin.c <- plugins/epan/<id>/plugin.c dependency dropped."
        # The canonical Wireshark plugin CMake layout keeps plugin.c separate from dissector sources.
        if self.plugin_c_mode == "generated":
            plugin_c_block = "register_plugin_files(plugin.c plugin ${DISSECTOR_SRC})"
        else:
            # In this mode, plugin.c is provided by Stage C (source tree) instead of generated by Wireshark macros.
            plugin_c_block = "# plugin.c provided by Stage C (plugin_c_mode=from_stage_c); do not auto-generate."
        content = f"""# Auto-generated by WiresharkSyntaxTester for pargen.

include(WiresharkPlugin)

set(DISSECTOR_SRC
{c_list}
)

set(PLUGIN_FILES
    plugin.c
    ${{DISSECTOR_SRC}}
)

set(DISSECTOR_SUPPORT_HEADERS
{h_list}
)

set_module_info({self.plugin_id} 1 0 0 0)

{plugin_c_block}

add_wireshark_plugin_library({self.plugin_id} epan)
target_link_libraries({self.plugin_id} epan)

install_plugin({self.plugin_id} epan)
"""

        cmake_path.write_text(content, encoding="utf-8")

    def _find_cmake_command_span(
        self,
        text: str,
        command: str,
        first_arg: str,
    ) -> Optional[Tuple[int, int]]:
        """
        Locate the span [start, end) of a CMake command like: command(first_arg ... ).
        Uses parenthesis depth (with basic quoted-string handling) to find the matching ')'.
        """
        pattern = re.compile(rf"(?im)^[ \t]*{re.escape(command)}\(\s*{re.escape(first_arg)}\b")
        match = pattern.search(text)
        if not match:
            return None

        start = match.start()
        open_idx = text.find("(", match.start(), match.end())
        if open_idx == -1:
            return None

        depth = 0
        in_dquote = False
        escaped = False

        for idx in range(open_idx, len(text)):
            ch = text[idx]

            if in_dquote:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_dquote = False
                continue

            if ch == '"':
                in_dquote = True
                continue

            if ch == "(":
                depth += 1
                continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    return (start, end)

        return None

    def _patch_top_cmakelists(self) -> None:
        top_cmake = self.ws_src_root / "CMakeLists.txt"
        if not top_cmake.exists():
            print(f"[WiresharkSyntaxTester] Warning: top-level CMakeLists.txt not found at {top_cmake}")
            return

        token = f"plugins/epan/{self.plugin_id}"
        original = top_cmake.read_text(encoding="utf-8", errors="ignore")

        # (A) Remove any existing occurrences of the token anywhere in the file.
        cleaned = re.sub(rf"(?m)^[ \t]*{re.escape(token)}[ \t]*\r?\n", "", original)
        cleaned = cleaned.replace(token, "")

        # (B) Locate the set(PLUGIN_SRC_DIRS ...) command span precisely.
        span = self._find_cmake_command_span(cleaned, "set", "PLUGIN_SRC_DIRS")
        if span is None:
            print("[WiresharkSyntaxTester] Warning: set(PLUGIN_SRC_DIRS ...) not found; cannot inject plugin dir.")
            return

        start, end = span
        cmd_text = cleaned[start:end]

        if token in cmd_text:
            top_cmake.write_text(cleaned, encoding="utf-8")
            return

        newline = "\r\n" if "\r\n" in cleaned else "\n"

        # (C) Insert token inside the command, handling single-line vs multi-line.
        if ("\n" not in cmd_text) and ("\r\n" not in cmd_text):
            close_pos = cmd_text.rfind(")")
            if close_pos == -1:
                print("[WiresharkSyntaxTester] Warning: malformed set(PLUGIN_SRC_DIRS ...) without ')'")
                return
            before = cmd_text[:close_pos].rstrip()
            after = cmd_text[close_pos:]
            patched_cmd = f"{before} {token}{after}"
        else:
            lines = cmd_text.splitlines(keepends=True)
            close_idx = None
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].lstrip().startswith(")"):
                    close_idx = i
                    break
            if close_idx is None:
                print("[WiresharkSyntaxTester] Warning: malformed multi-line set(PLUGIN_SRC_DIRS ...) without closing ')'")
                return

            item_indent: Optional[str] = None
            for line in lines[1:]:
                stripped = line.strip()
                if not stripped or stripped.startswith(")"):
                    continue
                item_indent = line[: len(line) - len(line.lstrip(" \t"))]
                break
            if item_indent is None:
                item_indent = "    "

            inject_line = f"{item_indent}{token}{newline}"
            lines.insert(close_idx, inject_line)
            patched_cmd = "".join(lines)

        new_text = cleaned[:start] + patched_cmd + cleaned[end:]

        # (D) Invariants: token appears exactly once and inside PLUGIN_SRC_DIRS command.
        if new_text.count(token) != 1:
            raise RuntimeError(
                f"WiresharkSyntaxTester patch invariant failed: token count != 1 (count={new_text.count(token)})"
            )

        span_after = self._find_cmake_command_span(new_text, "set", "PLUGIN_SRC_DIRS")
        if span_after is None:
            raise RuntimeError("WiresharkSyntaxTester patch invariant failed: cannot re-locate set(PLUGIN_SRC_DIRS ...)")

        s2, e2 = span_after
        tok_pos = new_text.find(token)
        if not (s2 <= tok_pos < e2):
            raise RuntimeError("WiresharkSyntaxTester patch invariant failed: token injected outside set(PLUGIN_SRC_DIRS ...)")

        top_cmake.write_text(new_text, encoding="utf-8")

    def _disable_builtin_dissector(self, builtin_filter_name: str) -> None:
        """
        Disable a built-in dissector that registers the given protocol filter_name.

        Note: Wireshark's dissector source filename may not match the protocol filter_name
        (e.g., "modbus" is registered in packet-mbtcp.c), so we locate the source file via
        proto_register_protocol(..., ..., "<filter_name>") and then remove that source from
        epan/dissectors/CMakeLists.txt.
        """
        filter_name = builtin_filter_name
        dissectors_dir = self.ws_src_root / "epan" / "dissectors"
        cmake_path = dissectors_dir / "CMakeLists.txt"
        if not cmake_path.exists():
            print(
                f"[WiresharkSyntaxTester] Warning: {cmake_path} not found; cannot disable builtin dissector."
            )
            return

        candidates: List[Path] = []
        proto_pat = re.compile(
            rf'proto_register_protocol\(\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*"{re.escape(filter_name)}"\s*\)'
        )
        if dissectors_dir.exists():
            known_builtin_sources = {
                "modbus": ["packet-mbtcp.c"],
            }
            for name in [f"packet-{filter_name}.c", *known_builtin_sources.get(filter_name, [])]:
                path = dissectors_dir / name
                if not path.exists() or path in candidates:
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if proto_pat.search(content):
                    candidates.append(path)

            rg = shutil.which("rg")
            if rg and not candidates:
                try:
                    proc = subprocess.run(
                        [
                            rg,
                            "-n",
                            "-l",
                            "--glob",
                            "*.c",
                            rf'proto_register_protocol\([^)]*"{re.escape(filter_name)}"',
                            str(dissectors_dir),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=30,
                    )
                    if proc.returncode == 0:
                        for line in proc.stdout.splitlines():
                            path = Path(line.strip())
                            if path.suffix == ".c":
                                candidates.append(path)
                except subprocess.TimeoutExpired:
                    candidates = []

            if not candidates:
                for path in dissectors_dir.rglob("*.c"):
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    if '"%s"' % filter_name not in content or "proto_register_protocol" not in content:
                        continue
                    if proto_pat.search(content):
                        candidates.append(path)

        if not candidates:
            fallback = dissectors_dir / f"packet-{filter_name}.c"
            if fallback.exists():
                candidates.append(fallback)

        if not candidates:
            print(
                f"[WiresharkSyntaxTester] No builtin dissector source found for filter_name='{filter_name}'."
            )
            return

        # Also disable any built-in dissectors that register themselves on dissector tables owned by
        # the disabled protocol. Otherwise, those dissectors can emit runtime "OOPS: dissector table
        # '<filter>.*' doesn't exist" at startup (because plugins are loaded after built-in handoff).
        #
        # Example (Wireshark 4.4.x):
        #   - packet-opensafety.c does dissector_add_string("modbus.data", ...)
        #   - packet-mbtcp.c registers "modbus.data"
        # If we disable packet-mbtcp.c but keep packet-opensafety.c, tshark -G protocols prints OOPS.
        dependent_basenames: set[str] = set()
        try:
            table_pat = re.compile(r'register_dissector_table\(\s*"([^"]+)"')
            registered_tables: set[str] = set()
            for src in candidates:
                try:
                    content = src.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for m in table_pat.finditer(content):
                    name = (m.group(1) or "").strip()
                    # Keep this conservative: only treat tables under the protocol namespace.
                    if name and name.startswith(f"{filter_name}."):
                        registered_tables.add(name)

            if registered_tables and dissectors_dir.exists():
                rg = shutil.which("rg")
                if rg:
                    for table in sorted(registered_tables):
                        try:
                            proc = subprocess.run(
                                [
                                    rg,
                                    "-n",
                                    "-l",
                                    "--glob",
                                    "*.c",
                                    rf'dissector_add_\w+\(\s*"{re.escape(table)}"',
                                    str(dissectors_dir),
                                ],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                text=True,
                                timeout=30,
                            )
                        except subprocess.TimeoutExpired:
                            continue
                        if proc.returncode != 0:
                            continue
                        for line in proc.stdout.splitlines():
                            path = Path(line.strip())
                            if path.suffix != ".c":
                                continue
                            if path.name in {p.name for p in candidates}:
                                continue
                            dependent_basenames.add(path.name)
                else:
                    add_pat_by_table = [
                        re.compile(rf'dissector_add_\w+\(\s*"{re.escape(table)}"')
                        for table in sorted(registered_tables)
                    ]
                    for path in dissectors_dir.rglob("*.c"):
                        if path.name in {p.name for p in candidates}:
                            continue
                        try:
                            content = path.read_text(encoding="utf-8", errors="ignore")
                        except OSError:
                            continue
                        if any(pat.search(content) for pat in add_pat_by_table):
                            dependent_basenames.add(path.name)
        except Exception:
            dependent_basenames = set()

        source_basenames = sorted({p.name for p in candidates} | dependent_basenames)

        text = cmake_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines(keepends=True)
        new_lines: List[str] = []
        removed_any = False

        def _should_drop_line(stripped_line: str, basename: str) -> bool:
            if stripped_line == basename:
                return True
            if stripped_line.endswith("/" + basename):
                return True
            if stripped_line.endswith("\\" + basename):
                return True
            return False

        for line in lines:
            stripped = line.strip()
            if any(_should_drop_line(stripped, bn) for bn in source_basenames):
                removed_any = True
                continue
            new_lines.append(line)

        if removed_any:
            backup = cmake_path.with_suffix(".txt.bak")
            if not backup.exists():
                backup.write_text(text, encoding="utf-8")
            cmake_path.write_text("".join(new_lines), encoding="utf-8")
            cmake_cache = self.ws_build_dir / "CMakeCache.txt"
            if cmake_cache.exists():
                cmake_cache.unlink()
            print(
                f"[WiresharkSyntaxTester] Disabled builtin dissector(s) for filter_name='{filter_name}': "
                + ", ".join(source_basenames)
            )
        else:
            # If none of the candidate basenames exist in the file, they were likely disabled in a
            # previous run; avoid spamming warnings on repeated checks.
            if all(bn not in text for bn in source_basenames):
                return
            print(
                f"[WiresharkSyntaxTester] Builtin dissector source(s) not listed in dissectors/CMakeLists for "
                f"filter_name='{filter_name}': {', '.join(source_basenames)}"
            )

    def _run_cmake_and_make(
        self,
        protocol_name: str,
        logs_root: Optional[Path] = None,
    ) -> SyntaxTestResult:
        if logs_root is None:
            logs_root = self.ws_build_dir / "pargen_logs"
        logs_root.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_root / f"wireshark_build_{self.plugin_id}_{protocol_name}_{ts}.log"

        def _read_tail(*, max_bytes: int = 20000) -> str:
            try:
                with log_path.open("rb") as f:
                    f.seek(0, 2)
                    end = f.tell()
                    start = max(0, end - int(max_bytes))
                    f.seek(start)
                    data = f.read(end - start)
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return ""

        print(
            f"[WiresharkSyntaxTester] Building plugin_id='{self.plugin_id}' protocol='{protocol_name}' "
            f"(build_dir={self.ws_build_dir}; log={log_path})"
        )

        # This builder is only used for generated C plugins. Disable Wireshark's Lua
        # subsystem explicitly so runtime checks do not depend on an environment-specific
        # wslua build state.
        cmake_cmd = [
            "cmake",
            "..",
            "-DENABLE_WERROR=OFF",
            "-DENABLE_APPLICATION_BUNDLE=OFF",
            "-DBUILD_wireshark=OFF",
            "-DENABLE_QT6=OFF",
            "-DENABLE_LUA=OFF",
        ]
        cmake_cache = self.ws_build_dir / "CMakeCache.txt"
        if not cmake_cache.exists() and shutil.which("ninja"):
            cmake_cmd.extend(["-G", "Ninja"])

        def _build_timeout_s() -> Optional[float]:
            raw = os.getenv("PARGEN_WIRESHARK_BUILD_TIMEOUT_S")
            if raw is None:
                return 900.0
            try:
                value = float(str(raw).strip())
            except (TypeError, ValueError):
                return 900.0
            if value <= 0:
                return None
            return value

        def _run(cmd: List[str]) -> Tuple[int, str]:
            # Stream output directly into the log file so users can `tail -f` during long builds.
            timeout_s = _build_timeout_s()
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"$ {' '.join(cmd)}\n")
                if timeout_s is not None:
                    f.write(f"[WiresharkSyntaxTester] timeout_s={timeout_s:.1f}\n")
                f.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=self.ws_build_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    rc = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        proc.wait()
                    f.write(f"\n[WiresharkSyntaxTester] ERROR: command timed out after {timeout_s:.1f}s\n")
                    f.flush()
                    return 124, _read_tail()
                f.write("\n")
                f.flush()
            return rc, _read_tail()

        cmake_rc = 0
        if not cmake_cache.exists():
            cmake_rc, _ = _run(cmake_cmd)

        build_cmd = ["cmake", "--build", ".", "--target", self.plugin_id, "--parallel", str(allocated_build_parallelism())]
        make_rc, make_out = _run(build_cmd)

        # If the build dir was configured before we injected a new plugin target,
        # the first targeted build fails immediately because the target doesn't exist
        # in the generated build graph. In that case, force a (re)configure and retry once.
        lower_make_out = (make_out or "").lower()
        if make_rc != 0 and ("no rule to make target" in lower_make_out or "unknown target" in lower_make_out):
            cmake_rc, _ = _run(cmake_cmd)
            if cmake_rc == 0:
                make_rc, _ = _run(build_cmd)

        ok = (cmake_rc == 0 and make_rc == 0)

        errors: List[str] = []
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            for line in log_text.splitlines():
                lower = line.lower()
                if " error:" in lower or "fatal error" in lower:
                    errors.append(line.strip())
                if "no rule to make target" in lower or "cmake error" in lower:
                    errors.append(line.strip())
        except Exception:
            pass

        if not ok and not errors:
            errors.append("Compilation failed, see log for details.")

        return SyntaxTestResult(
            ok=ok,
            cmake_rc=cmake_rc,
            make_rc=make_rc,
            errors=errors,
            log_path=log_path,
        )

    def _run_runtime_compatibility_test(
        self,
        protocol_name: str,
        logs_root: Optional[Path] = None,
    ) -> dict:
        if logs_root is None:
            logs_root = self.ws_build_dir / "pargen_logs"
        logs_root.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_root / f"wireshark_runtime_{self.plugin_id}_{protocol_name}_{ts}.log"

        run_dir = self.ws_build_dir / "run"
        candidates = [
            run_dir / "tshark",
            run_dir / "tshark.exe",
            run_dir / "Wireshark.app" / "Contents" / "MacOS" / "tshark",
        ]
        tshark_path = None
        for candidate in candidates:
            if candidate.exists():
                tshark_path = candidate.resolve()
                break

        errors: List[str] = []

        if tshark_path is None:
            errors.append(f"tshark not found under {run_dir}")
            return {
                "ok": False,
                "rc": -1,
                "errors": errors,
                "log_path": log_path,
            }

        cmd = [str(tshark_path), "-G", "plugins"]
        try:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(f"$ {' '.join(cmd)}\n")
                proc = subprocess.run(
                    cmd,
                    cwd=run_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                f.write(proc.stdout)
                f.write(proc.stderr or "")
                f.write("\n")
                rc = proc.returncode
        except FileNotFoundError:
            errors.append(f"tshark not found at {tshark_path}")
            return {
                "ok": False,
                "rc": -1,
                "errors": errors,
                "log_path": log_path,
            }

        if rc != 0:
            errors.append(f"tshark exited with code {rc}")
            try:
                for line in ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines():
                    low = line.lower()
                    if any(marker in low for marker in ("error", "failed", "oops", "load plugin", "undefined symbol")):
                        errors.append(line.strip())
            except Exception:
                pass
            return {
                "ok": False,
                "rc": rc,
                "errors": errors,
                "log_path": log_path,
            }

        plugins_out = proc.stdout or ""
        plugins_err = proc.stderr or ""

        found_plugin = False
        try:
            for line in plugins_out.splitlines():
                cols = line.split("\t")
                # A loader error mentioning the filename is NOT a plugin row.
                if len(cols) >= 4 and cols[0].strip() in {
                    self.plugin_id + ext for ext in (".so", ".dll", ".dylib")
                }:
                    found_plugin = True
                    break
        except Exception:
            pass

        if not found_plugin:
            errors.append(f"plugin_id '{self.plugin_id}' not found in tshark -G plugins output")

        # Also validate the protocol filter_name actually exists.
        cmd2 = [str(tshark_path), "-G", "protocols"]
        protocols_out = ""
        rc2 = 0
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n$ {' '.join(cmd2)}\n")
                proc2 = subprocess.run(
                    cmd2,
                    cwd=run_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                protocols_out = proc2.stdout or ""
                protocols_err = proc2.stderr or ""
                rc2 = proc2.returncode
                f.write(protocols_out)
                f.write(protocols_err)
                f.write("\n")
        except Exception as exc:
            errors.append(f"tshark -G protocols failed: {type(exc).__name__}: {exc}")
            return {
                "ok": False,
                "rc": rc2 or rc,
                "errors": errors,
                "log_path": log_path,
            }

        if rc2 != 0:
            errors.append(f"tshark -G protocols exited with code {rc2}")
            errors.extend(line.strip() for line in protocols_err.splitlines() if line.strip())
            return {
                "ok": False,
                "rc": rc2,
                "errors": errors,
                "log_path": log_path,
            }

        target = (protocol_name or "").strip().lower()
        found_protocol = False
        try:
            for line in protocols_out.splitlines():
                if not line.strip():
                    continue
                cols = line.split("\t")
                # Column 3 is the filter. A display name such as "CIP Modbus
                # Object" must not stand in for the generated "modbus" filter.
                if len(cols) >= 3 and cols[2].strip().lower() == target:
                    found_protocol = True
                    break
        except Exception:
            found_protocol = False

        if not found_protocol and target:
            errors.append(
                f"protocol filter_name '{protocol_name}' not found in tshark -G protocols output"
            )

        # tshark can exit zero while dlopen fails. Reject loader errors in both
        # streams, including when a separate/built-in protocol happens to match.
        fatal_markers = ("oops", "doesn't exist", "couldn't load plugin", "could not load plugin",
                         "cannot load plugin", "failed to load plugin", "undefined symbol",
                         "symbol not found", "cannot open shared object")
        for out in (plugins_out, plugins_err, protocols_out, protocols_err):
            for line in (out or "").splitlines():
                if any(marker in line.lower() for marker in fatal_markers):
                    errors.append(line.strip())

        ok = bool(
            found_plugin
            and found_protocol
            and not errors
        )
        return {
            "ok": ok,
            "rc": rc,
            "errors": errors,
            "log_path": log_path,
        }
