from __future__ import annotations

"""Zeek + Spicy syntax backend (Stage C).

This backend compiles Spicy source (*.spicy, *.evt) into a Zeek-loadable HLTO
artifact via `spicyz`. Optionally it can run a lightweight runtime check using
`zeek` to validate that the generated HLTO can be loaded.
"""

import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..backend_base import SyntaxBackend
from ..schema import SyntaxArtifacts, SyntaxResult


_ZEEK_SPICY_EXTRA_LLM_RULES = """
Zeek+Spicy repair rules (must follow):
- Output must include at least one *.spicy file under the plugin directory.
- If you are implementing a Zeek Spicy protocol analyzer, you MUST include an *.evt file and it MUST declare a protocol analyzer, e.g.:
    protocol analyzer spicy::<AnalyzerName> over TCP:
        parse with <SpicyUnit> , port <PORT> ;
  or the UDP variant. Use parse originator/parse responder only when required and valid.
- Keep names consistent:
  * Spicy module/namespace, unit names, and the analyzer name must match (avoid mismatched module paths).
  * The analyzer name should be stable and derived from the protocol under test (file_prefix/filter_name).
- IMPORTANT Spicy syntax constraints (compiler will reject common "guessed" syntax):
  * Spicy does NOT support an attribute named '&length'. Use '&size' only on 'bytes', and for length-delimited sub-units use '&parse-from' with a bytes buffer:
      public type Message = unit {
          hdr: Header;
          body_bytes: bytes &size=(hdr.len);
          body: Body &parse-from=body_bytes;
      };
  * For bit-level parsing, you MUST use `bitfield(N) { ... } &bit-order=...;` (Spicy does NOT support `uint8 &bit-order=... { ... }` nor `uint8(4)` style members).
      first_byte: bitfield(8) {
          kind: 0..2;
          code: 3..7;
      } &bit-order=spicy::BitOrder::MSB0;
    `N` must be a byte-multiple width accepted by spicyz (8, 16, 24, 32, ...). Do NOT emit standalone `bitfield(1)`..`bitfield(7)`;
    for a sub-byte selector followed by another field, model the byte-multiple enclosing bytes and expose the subfields inside that wrapper.
  * Do NOT use the deprecated `&transient` attribute (spicyz reports it as removed). If you need direction-specific parsing, use EVT analyzer declarations (`parse originator with` / `parse responder with`) instead of trying to encode direction as a field in the Spicy grammar.
  * Spicy 'switch' is not C-style. Do NOT use 'case ...:' or 'case ... of { ... }'.
    Correct form (note '->' and the terminating '};'). IMPORTANT: 'switch' is a parsing construct (a statement) and cannot be used as an expression or as `field: switch (...) { ... }`.
      switch (self.kind) {
          1 -> a: TypeA;
          2, 3 -> b: TypeB;
          * -> raw: bytes &eod;
      };
  * Prefer `throw "<message>"` or `: void &requires=(...)` for constraints; do NOT invent exception types like `InvalidValue(...)`.
  * When referencing earlier fields inside a unit, use `self.<field>` (e.g., `bytes &size=self.len` and `switch (self.kind) { ... }`).
- Platform API pattern for TLV/tag-length-value protocols:
  * Do not leave a TLV item as only `{ type; length; value: bytes &size=length; }` when the Parse-IR or codegen_profile names child fields/variants inside the value. That loses parser knowledge.
  * Correct pattern: first capture the bounded value bytes, then parse a selected body unit from that buffer with `&parse-from=self.value_bytes`.
      type TLV = unit {
          kind: uint16;
          len: uint16;
          value_bytes: bytes &size=self.len;
          switch (self.kind) {
              1 -> body_a: BodyA &parse-from=self.value_bytes;
              2 -> body_b: BodyB &parse-from=self.value_bytes;
              * -> raw_value: bytes &parse-from=self.value_bytes;
          };
      };
  * For grouped/nested TLVs, parse nested TLV arrays from the group's value buffer, not from the top-level packet stream.
  * Expand only grouped variants whose child fields are defined in the Parse-IR; keep other grouped values as bounded bytes instead of emitting unsupported helper fields.
  * Use unique names for distinct bitfield containers (e.g. Header_Flags, Option_Flags, Variant_Flags). Reusing a generic field name like `flags` for multiple layouts makes export-time sub-bit recovery ambiguous.
  * Enforce variant consistency and reserved/zero/length invariants with `&requires` or `on %done { if (...) throw "BS_CONSTRAINT <id> ..."; }`. A malformed negative seed should cause a packet-level parse failure instead of being silently accepted.
- Do NOT introduce load-time side effects:
  * Zeek scripts (*.zeek) should only define events/logging/hooks, not do file IO at load time.
  * Spicy code should not perform external IO; parsing should be deterministic.
- Avoid `var x: T = self.y` at unit scope: it can compile but fail at runtime with `hilti::rt::UnsetOptional`.
  Prefer computing inline, or declare `var x: T;` and assign in a hook (`on <field> { self.x = ...; }` / `on %done`).
- Never use `hilti::BitOrder` / `hilti::ByteOrder`. Use `spicy::BitOrder::{LSB0,MSB0}` / `spicy::ByteOrder::{Big,Little}` and add `import spicy;`.
- Do not use `.size()` (it is not a Spicy method). For bytes length, use `|bytes_expr|` (or `|$$|` inside a bytes field).
- Keep the package structure compatible with zkg/Zeek Spicy conventions:
  analyzer/*.spicy, analyzer/*.evt, scripts/__load__.zeek, scripts/main.zeek, zkg.meta.
""".strip()


def _zeek_spicy_lowering_context(backend_config: Optional[Dict[str, Any]], *, max_dispatches: int = 12) -> str:
    cfg = backend_config if isinstance(backend_config, dict) else {}
    plan = cfg.get("stage_c_lowering_plan")
    if not isinstance(plan, dict):
        return ""

    dispatches = plan.get("selector_dispatches")
    length_checks = plan.get("length_checks")
    if not isinstance(dispatches, list):
        dispatches = []
    if not isinstance(length_checks, list):
        length_checks = []
    if not dispatches and not length_checks:
        return ""

    lines: List[str] = [
        "Stage C lowering plan (host-neutral contract; implement this in Spicy, do not reinterpret it as prose):",
        "- Selector dispatches must become switch arms over the selector field, with bounded value bytes parsed by &parse-from.",
        "- bit_start/size_bits entries describe parser-visible fields relative to the selector variant body.",
        "- condition_on and flag-controlled fields must preserve selector/flag consistency; length_of entries must become exact consumed-length checks.",
    ]
    for raw in dispatches[:max_dispatches]:
        if not isinstance(raw, dict):
            continue
        selector_name = str(raw.get("selector_name") or "selector").strip()
        selector_value = raw.get("selector_value")
        variant_name = str(raw.get("variant_name") or "variant").strip()
        fields = raw.get("fields")
        if not isinstance(fields, list):
            fields = []
        field_bits: List[str] = []
        for field in fields[:8]:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name") or "field").strip()
            size = str(field.get("size_bits") or "?").strip()
            offset = field.get("base_bit_offset")
            offset_txt = f"@bit+{offset}" if isinstance(offset, int) else ""
            flag_name = str(field.get("flag_name") or "").strip()
            flag_txt = ""
            if flag_name:
                flag_mask = str(field.get("flag_mask") or "").strip()
                flag_offset = field.get("flag_base_bit_offset")
                flag_txt = f" if flag {flag_name}"
                if isinstance(flag_offset, int) or flag_mask:
                    flag_txt += f"({flag_mask or '?'}@bit+{flag_offset if isinstance(flag_offset, int) else '?'})"
            field_bits.append(f"{name}:{size}b{offset_txt}{flag_txt}")
        suffix = ", ..." if len(fields) > 8 else ""
        lines.append(f"  * {selector_name}={selector_value} -> {variant_name}: {', '.join(field_bits)}{suffix}")
    if len(dispatches) > max_dispatches:
        lines.append(f"  * ... {len(dispatches) - max_dispatches} more selector dispatches omitted")

    if length_checks:
        lines.append("- Length checks:")
        for raw in length_checks[:8]:
            if not isinstance(raw, dict):
                continue
            src = str(raw.get("src_name") or raw.get("src") or "length").strip()
            dst = str(raw.get("dst_name") or raw.get("dst") or "body").strip()
            formula = str(raw.get("formula") or "").strip()
            tail = f" ({formula})" if formula else ""
            lines.append(f"  * {src} length_of {dst}{tail}")
        if len(length_checks) > 8:
            lines.append(f"  * ... {len(length_checks) - 8} more length checks omitted")
    return "\n".join(lines)


def _sanitize_prefix(file_prefix: str) -> str:
    s = (file_prefix or "").strip()
    if not s:
        return "stage_c"
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("._-")
    return s or "stage_c"


def _sorted_existing(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        try:
            if p.is_file():
                out.append(p)
        except OSError:
            continue
    out.sort(key=lambda x: str(x))
    return out


def _which_or_none(name: str) -> Optional[Path]:
    found = shutil.which(name)
    if not found:
        return None
    try:
        return Path(found).resolve()
    except OSError:
        return Path(found)


class ZeekSpicyBackend(SyntaxBackend):
    """Compile Spicy sources to HLTO; optionally validate runtime loading with Zeek."""

    name = "zeek_spicy"

    def __init__(
        self,
        *,
        zeek_path: Optional[Path] = None,
        spicyz_path: Optional[Path] = None,
    ) -> None:
        self.zeek_path = Path(zeek_path) if zeek_path is not None else None
        self.spicyz_path = Path(spicyz_path) if spicyz_path is not None else None

    def _resolve_spicyz(self) -> Optional[Path]:
        if self.spicyz_path is not None:
            try:
                if self.spicyz_path.exists():
                    return self.spicyz_path.resolve()
            except OSError:
                return self.spicyz_path
            return None
        return _which_or_none("spicyz")

    def _resolve_zeek(self) -> Optional[Path]:
        if self.zeek_path is not None:
            try:
                if self.zeek_path.exists():
                    return self.zeek_path.resolve()
            except OSError:
                return self.zeek_path
            return None
        return _which_or_none("zeek")

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
        prefix = _sanitize_prefix(file_prefix)

        build_log = logs_root / f"{prefix}_zeek_spicy_build.log"
        runtime_log = logs_root / f"{prefix}_zeek_spicy_runtime.log"
        hlto_path = logs_root / f"{prefix}.hlto"

        runtime_check = False
        pcap_path: Optional[Path] = None
        if isinstance(backend_config, dict):
            runtime_check = bool(backend_config.get("runtime_check", False))
            raw_pcap = backend_config.get("pcap_path")
            if raw_pcap:
                try:
                    pcap_path = Path(str(raw_pcap)).expanduser()
                except Exception:
                    pcap_path = None

        errors: List[str] = []
        runtime_errors: List[str] = []

        spicy_files = _sorted_existing(plugin_src_dir.rglob("*.spicy"))
        evt_files = _sorted_existing(plugin_src_dir.rglob("*.evt"))

        spicyz = self._resolve_spicyz()
        if not spicy_files:
            errors.append(f"[BUILD] No *.spicy files found under: {plugin_src_dir}")
        if spicyz is None:
            hint = "spicyz not found. Install Zeek Spicy tooling or pass --stage-c-spicyz <path>."
            errors.append(f"[TOOL] {hint}")

        build_rc: Optional[int] = None
        runtime_rc: Optional[int] = None
        runtime_ok: Optional[bool] = None

        def _write_build_log(text: str) -> None:
            try:
                build_log.parent.mkdir(parents=True, exist_ok=True)
                with build_log.open("a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                return

        try:
            build_log.write_text("", encoding="utf-8")
        except Exception:
            pass

        _write_build_log(f"[ZeekSpicyBuild] plugin_src_dir={plugin_src_dir}\n")
        _write_build_log(f"[ZeekSpicyBuild] protocol_filter_name={protocol_filter_name}\n")
        _write_build_log(f"[ZeekSpicyBuild] file_prefix={file_prefix} sanitized_prefix={prefix}\n")
        _write_build_log(f"[ZeekSpicyBuild] spicy_files={[str(p) for p in spicy_files]}\n")
        _write_build_log(f"[ZeekSpicyBuild] evt_files={[str(p) for p in evt_files]}\n")
        _write_build_log(f"[ZeekSpicyBuild] spicyz={str(spicyz) if spicyz else None}\n")
        _write_build_log(f"[ZeekSpicyBuild] hlto_out={hlto_path}\n")

        if errors:
            artifacts = SyntaxArtifacts(
                tool_name=self.name,
                build_dir=logs_root,
                tool_bin=spicyz,
                plugin_install_dir=plugin_src_dir,
                extra={
                    "hlto": str(hlto_path),
                    "spicy_files": [str(p) for p in spicy_files],
                    "evt_files": [str(p) for p in evt_files],
                    "spicyz": str(spicyz) if spicyz else None,
                    "zeek": str(self.zeek_path) if self.zeek_path else None,
                },
            )
            return SyntaxResult(
                ok=False,
                runtime_ok=None,
                errors=errors,
                runtime_errors=[],
                build_log=build_log,
                runtime_log=None,
                artifacts=artifacts,
                build_rc=1,
                runtime_rc=None,
            )

        # Remove stale output if present to avoid false positives on failure.
        try:
            if hlto_path.exists():
                hlto_path.unlink()
        except Exception:
            pass

        # The compiler runs with cwd=plugin_src_dir so generated imports resolve
        # locally. Source arguments must still be absolute; otherwise a repo-root
        # relative plugin path becomes invalid after the cwd switch.
        cmd = [str(spicyz), "-o", str(hlto_path)] + [str(p.resolve()) for p in spicy_files] + [
            str(p.resolve()) for p in evt_files
        ]
        _write_build_log(f"[ZeekSpicyBuild] cmd={' '.join(cmd)}\n")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(plugin_src_dir),
                capture_output=True,
                text=True,
            )
            build_rc = int(proc.returncode)
            if proc.stdout:
                _write_build_log("\n[stdout]\n" + proc.stdout + ("\n" if not proc.stdout.endswith("\n") else ""))
            if proc.stderr:
                _write_build_log("\n[stderr]\n" + proc.stderr + ("\n" if not proc.stderr.endswith("\n") else ""))
        except Exception as exc:
            build_rc = 127
            errors.append(f"[BUILD] subprocess failed: {type(exc).__name__}: {exc}")
            _write_build_log(f"\n[exception] {type(exc).__name__}: {exc}\n")
            _write_build_log(traceback.format_exc() + "\n")

        if build_rc != 0:
            errors.append(f"[BUILD] spicyz failed (rc={build_rc}); see build log: {build_log}")
        else:
            try:
                if not hlto_path.exists():
                    errors.append(
                        "[BUILD] spicyz reported success but HLTO output does not exist: "
                        f"{hlto_path} (see build log: {build_log})"
                    )
            except OSError:
                errors.append(f"[BUILD] unable to stat HLTO output: {hlto_path}")

        ok = not errors

        zeek = self._resolve_zeek() if (runtime_check and ok) else None
        if runtime_check and ok:
            runtime_ok = True
            try:
                runtime_log.write_text("", encoding="utf-8")
            except Exception:
                pass

            def _write_runtime_log(text: str) -> None:
                try:
                    runtime_log.parent.mkdir(parents=True, exist_ok=True)
                    with runtime_log.open("a", encoding="utf-8") as f:
                        f.write(text)
                except Exception:
                    return

            _write_runtime_log(f"[ZeekSpicyRuntime] zeek={str(zeek) if zeek else None}\n")
            _write_runtime_log(f"[ZeekSpicyRuntime] hlto={hlto_path}\n")
            _write_runtime_log(f"[ZeekSpicyRuntime] pcap={str(pcap_path) if pcap_path else None}\n")
            _write_runtime_log(f"[ZeekSpicyRuntime] python={sys.executable}\n")

            if zeek is None:
                runtime_ok = False
                runtime_errors.append(
                    "[RUNTIME] zeek not found. Install Zeek or pass --stage-c-zeek <path> to enable runtime checks."
                )
            else:
                # (1) Check Zeek built-in Spicy support.
                cmd_n = [str(zeek), "-N", "Zeek::Spicy"]
                _write_runtime_log(f"[ZeekSpicyRuntime] cmd={' '.join(cmd_n)}\n")
                try:
                    proc_n = subprocess.run(cmd_n, capture_output=True, text=True)
                    if proc_n.stdout:
                        _write_runtime_log("\n[stdout]\n" + proc_n.stdout + ("\n" if not proc_n.stdout.endswith("\n") else ""))
                    if proc_n.stderr:
                        _write_runtime_log("\n[stderr]\n" + proc_n.stderr + ("\n" if not proc_n.stderr.endswith("\n") else ""))
                    runtime_rc = int(proc_n.returncode)
                    if proc_n.returncode != 0:
                        runtime_ok = False
                        runtime_errors.append(f"[RUNTIME] zeek -N Zeek::Spicy failed (rc={proc_n.returncode})")
                except Exception as exc:
                    runtime_ok = False
                    runtime_rc = 127
                    runtime_errors.append(f"[RUNTIME] subprocess failed: {type(exc).__name__}: {exc}")
                    _write_runtime_log(f"\n[exception] {type(exc).__name__}: {exc}\n")
                    _write_runtime_log(traceback.format_exc() + "\n")

                # (2) If a PCAP is provided, attempt to load and run the analyzer.
                if runtime_ok and pcap_path is not None:
                    try:
                        if pcap_path.exists():
                            cmd_r = [str(zeek), "-r", str(pcap_path), str(hlto_path)]
                            _write_runtime_log(f"\n[ZeekSpicyRuntime] cmd={' '.join(cmd_r)}\n")
                            proc_r = subprocess.run(cmd_r, capture_output=True, text=True)
                            if proc_r.stdout:
                                _write_runtime_log("\n[stdout]\n" + proc_r.stdout + ("\n" if not proc_r.stdout.endswith("\n") else ""))
                            if proc_r.stderr:
                                _write_runtime_log("\n[stderr]\n" + proc_r.stderr + ("\n" if not proc_r.stderr.endswith("\n") else ""))
                            runtime_rc = int(proc_r.returncode)
                            if proc_r.returncode != 0:
                                runtime_ok = False
                                runtime_errors.append(f"[RUNTIME] zeek -r failed (rc={proc_r.returncode}); see runtime log: {runtime_log}")
                        else:
                            _write_runtime_log(f"[ZeekSpicyRuntime] pcap not found; skipping: {pcap_path}\n")
                    except Exception as exc:
                        runtime_ok = False
                        runtime_errors.append(f"[RUNTIME] zeek -r failed: {type(exc).__name__}: {exc}")
                        _write_runtime_log(f"\n[exception] {type(exc).__name__}: {exc}\n")
                        _write_runtime_log(traceback.format_exc() + "\n")

        artifacts = SyntaxArtifacts(
            tool_name=self.name,
            build_dir=logs_root,
            tool_bin=spicyz,
            plugin_install_dir=plugin_src_dir,
            extra={
                "hlto": str(hlto_path),
                "spicy_files": [str(p) for p in spicy_files],
                "evt_files": [str(p) for p in evt_files],
                "spicyz": str(spicyz) if spicyz else None,
                "zeek": str(zeek) if zeek else None,
                "runtime_check": bool(runtime_check),
                "pcap": str(pcap_path) if pcap_path else None,
            },
        )

        return SyntaxResult(
            ok=bool(ok),
            runtime_ok=runtime_ok,
            errors=errors,
            runtime_errors=runtime_errors,
            build_log=build_log,
            runtime_log=(runtime_log if (runtime_check and ok) else None),
            artifacts=artifacts,
            build_rc=build_rc,
            runtime_rc=runtime_rc,
        )


    def extra_llm_rules(
        self,
        *,
        protocol_filter_name: str,
        file_prefix: str,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = _zeek_spicy_lowering_context(backend_config)
        if not context:
            return _ZEEK_SPICY_EXTRA_LLM_RULES
        return _ZEEK_SPICY_EXTRA_LLM_RULES + "\n\n" + context
