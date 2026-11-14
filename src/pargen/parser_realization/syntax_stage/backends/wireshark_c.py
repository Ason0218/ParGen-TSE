from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..backend_base import SyntaxBackend
from ..schema import SyntaxArtifacts, SyntaxResult

from ...target_platforms.wireshark_syntax_tester import (
    WIRESHARK_DEFAULT_URL,
    WiresharkSyntaxTester,
)


_WIRESHARK_EXTRA_LLM_RULES = """
Wireshark-C repair rules (must follow):
- Make the MINIMAL changes needed to fix compilation/runtime registration errors.
- Do NOT change protocol parsing semantics unless required for correctness.
- If you use expert info (ei_register_info / expert_register_field_array), you MUST:
  * include <epan/expert.h>
  * declare every referenced expert_field (ei_*) as: static expert_field <name> = EI_INIT;
  * ensure the ei_register_info array is a COMPLETE type before array_length(ei) usage
    (no forward decl like 'static ei_register_info ei[];' that makes it incomplete).
- Prefer tvb_get_uint8 over deprecated tvb_get_guint8.
- If a function has warn_unused_result, store the return value or cast to (void).
- The plugin MUST register the protocol filter_name used by the Stage C run. If the codegen
  profile contains wireshark_c.registration.filter_name, use that exact isolated filter name
  instead of the target protocol's built-in filter name.
- If you call dissector_add_uint/dissector_add_uint_with_preference on a dissector table,
  ensure the table exists (register_dissector_table) OR use the correct built-in table
  like "tcp.port" for TCP-port registration.
""".strip()


def _read_log_tail(path: Optional[Path], *, max_chars: int = 12000) -> str:
    if path is None:
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(data) <= max_chars:
        return data
    return data[-max_chars:]


def _find_tshark_bin(ws_build_dir: Path) -> Optional[Path]:
    run_dir = ws_build_dir / "run"
    candidates = [
        run_dir / "tshark",
        run_dir / "tshark.exe",
        run_dir / "Wireshark.app" / "Contents" / "MacOS" / "tshark",
    ]
    for path in candidates:
        try:
            if path.exists():
                return path.resolve()
        except OSError:
            continue
    return None


def _disable_builtin_from_config(backend_config: Optional[Dict[str, Any]]) -> bool:
    """Whether the syntax tester should remove a colliding built-in dissector.

    Most generated Wireshark-C plugins intentionally shadow the built-in protocol filter during
    isolated builds. Some protocols, however, expose helper symbols used by other built-ins; those
    plugins should instead use an isolated filter name and keep the built-in dissector enabled.
    The codegen profile can express that with wireshark_c.registration.disable_builtin_protocol.
    """

    if not isinstance(backend_config, dict):
        return True
    ws_cfg = backend_config.get("wireshark_c")
    if not isinstance(ws_cfg, dict):
        return True
    reg = ws_cfg.get("registration")
    if not isinstance(reg, dict):
        return True
    value = reg.get("disable_builtin_protocol")
    if value is None and "disable_builtin_protocol" not in reg:
        value = reg.get("disable_builtin")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"0", "false", "no", "off", "keep"}:
            return False
        if v in {"1", "true", "yes", "on", "disable"}:
            return True
    return True


class WiresharkCBackend(SyntaxBackend):
    name = "wireshark_c"

    def __init__(
        self,
        *,
        ws_src_root: Path,
        ws_build_dir: Path,
        plugin_id: str,
        plugin_c_mode: str = "generated",
        download_url: str = WIRESHARK_DEFAULT_URL,
    ) -> None:
        self.tester = WiresharkSyntaxTester(
            ws_src_root=ws_src_root,
            ws_build_dir=ws_build_dir,
            plugin_id=plugin_id,
            plugin_c_mode=plugin_c_mode,
            download_url=download_url,
        )

    def check(
        self,
        *,
        plugin_src_dir: Path,
        protocol_filter_name: str,
        file_prefix: str,
        logs_root: Path,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> SyntaxResult:
        result = self.tester.run_full_syntax_test(
            plugin_src_dir=plugin_src_dir,
            protocol_name=protocol_filter_name,
            logs_root=logs_root,
            disable_builtin=_disable_builtin_from_config(backend_config),
            check_runtime=bool((backend_config or {}).get('source_only_registration_check', False)),
        )

        artifacts = SyntaxArtifacts(
            tool_name=self.name,
            build_dir=self.tester.ws_build_dir,
            tool_bin=_find_tshark_bin(self.tester.ws_build_dir),
            plugin_install_dir=self.tester.ws_src_root / "plugins" / "epan" / self.tester.plugin_id,
            extra={
                "ws_src_root": str(self.tester.ws_src_root),
                "plugin_id": self.tester.plugin_id,
                "ws_build_dir": str(self.tester.ws_build_dir),
            },
        )

        return SyntaxResult(
            ok=bool(result.ok),
            runtime_ok=result.runtime_ok,
            errors=list(result.errors or []),
            runtime_errors=list(result.runtime_errors or []),
            build_log=result.log_path,
            runtime_log=result.runtime_log_path,
            artifacts=artifacts,
            build_rc=result.make_rc,
            runtime_rc=result.runtime_rc,
        )


    def extra_llm_rules(
        self,
        *,
        protocol_filter_name: str,
        file_prefix: str,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        return _WIRESHARK_EXTRA_LLM_RULES
