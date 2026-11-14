from __future__ import annotations

"""Auto-repair loop for Wireshark-C plugin builds.

Stage C codegen is LLM-driven and may occasionally emit code that fails to compile
against a specific Wireshark version. In practice, most failures are local and
repairable by either:
  (1) deterministic fixups (e.g., missing expert_field declarations), or
  (2) feeding compiler diagnostics back to a code agent for minimal edits.

This module integrates both approaches in a bounded loop.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backend_contract import (
    backend_contract_repair_rules,
    format_backend_contract_violations,
    validate_backend_realization,
)
from ..code_repair_agent import CodeRepairAgent, RepairDiagnostics
from ..llm_client import LLMClient, LLMConfig, create_llm_client
from .wireshark_syntax_tester import SyntaxTestResult, WiresharkSyntaxTester


@dataclass
class RepairLoopConfig:
    """Configuration for bounded build-and-repair."""

    # Total rounds includes the initial build attempt.
    max_rounds: int = 4


    # Enable LLM-based repair (requires API key). If False, only deterministic fixups run.
    enable_llm_repair: bool = True

    # Limit log tail injected into the repair prompt.
    log_tail_chars: int = 12000


def run_wireshark_build_with_auto_repair(
    tester: WiresharkSyntaxTester,
    plugin_src_dir: Path,
    protocol_name: str,
    logs_root: Path,
    codegen_profile: Optional[Dict[str, Any]] = None,
    llm_config: Optional[LLMConfig] = None,
    cfg: Optional[RepairLoopConfig] = None,
) -> SyntaxTestResult:
    """Run WiresharkSyntaxTester with an integrated auto-repair loop.

    Behavior:
      - Attempt build once.
      - If build fails, apply deterministic fixups and retry.
      - If still failing, invoke an LLM repair step (bounded) and retry.

    Returns the final SyntaxTestResult (ok or not).
    """

    cfg = cfg or RepairLoopConfig()
    logs_root.mkdir(parents=True, exist_ok=True)

    llm: Optional[LLMClient] = None
    if cfg.enable_llm_repair:
        try:
            llm = create_llm_client(llm_config)
        except Exception:
            # If API keys are missing, fall back to deterministic-only repairs.
            llm = None

    last_result: Optional[SyntaxTestResult] = None

    for round_idx in range(cfg.max_rounds):
        last_result = tester.run_full_syntax_test(
            plugin_src_dir=plugin_src_dir,
            protocol_name=protocol_name,
            logs_root=logs_root,
        )
        # Success requires BOTH compilation and runtime registration of the protocol.
        if last_result.ok and last_result.runtime_ok is not False:
            return last_result

        # --- repair path ---
        phase = "build" if not last_result.ok else "runtime"
        if phase == "build":
            errors = list(last_result.errors or [])
            log_tail = _read_log_tail(last_result.log_path, max_chars=cfg.log_tail_chars)
            print(
                f"[WiresharkAutoRepair] Build failed (round {round_idx + 1}/{cfg.max_rounds}). "
                f"cmake_rc={last_result.cmake_rc}, make_rc={last_result.make_rc}"
            )
        else:
            errors = list(last_result.runtime_errors or [])
            log_tail = _read_log_tail(last_result.runtime_log_path, max_chars=cfg.log_tail_chars)
            print(
                f"[WiresharkAutoRepair] Runtime failed (round {round_idx + 1}/{cfg.max_rounds}). "
                f"runtime_rc={last_result.runtime_rc}"
            )

        if errors:
            # Print at most a few lines; full context is in the logs.
            for line in errors[:8]:
                print("  ", line)
            if len(errors) > 8:
                print(f"  ... ({len(errors) - 8} more error lines)")

        changed = False


        if changed:
            print("[WiresharkAutoRepair] Applied deterministic fixups. Retrying build...")

        # If deterministic fixes changed files, retry build in next loop iteration.
        if changed:
            continue

        if cfg.enable_llm_repair and llm is not None:
            print("[WiresharkAutoRepair] Invoking code agent for minimal compile/runtime-fix patch...")
            changed |= _apply_llm_repair(
                llm=llm,
                plugin_src_dir=plugin_src_dir,
                protocol_name=protocol_name,
                errors=errors,
                log_tail=log_tail,
                codegen_profile=codegen_profile,
                logs_root=logs_root,
                round_idx=round_idx,
            )
            if changed:
                print("[WiresharkAutoRepair] Applied LLM patch. Retrying build...")
                continue

        # Nothing changed; no point looping further.
        break

    return last_result or SyntaxTestResult(
        ok=False,
        cmake_rc=-1,
        make_rc=-1,
        errors=["Auto-repair failed: no build result."],
        log_path=None,
    )


# ------------------------
# Deterministic fixups
# ------------------------


# ------------------------
# Runtime-focused deterministic fixups
# ------------------------


# ------------------------
# LLM repair
# ------------------------


def _apply_llm_repair(
    llm: LLMClient,
    plugin_src_dir: Path,
    protocol_name: str,
    errors: List[str],
    log_tail: str,
    codegen_profile: Optional[Dict[str, Any]],
    logs_root: Path,
    round_idx: int,
) -> bool:
    files = _read_plugin_files(plugin_src_dir, protocol_name)
    if not files:
        return False

    backend_violations = validate_backend_realization("wireshark_c", files, max_violations=24)
    backend_errors = [v.format_for_repair() for v in backend_violations[:12]]
    backend_summary = format_backend_contract_violations(backend_violations, max_items=12)
    errors_for_prompt = tuple(list(errors or []) + backend_errors)
    log_tail_for_prompt = (log_tail or "").strip()
    if backend_summary:
        log_tail_for_prompt += "\n\nBackend realization contract findings:\n" + backend_summary

    extra_rules = """
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
- The plugin MUST register the filter_name requested by the run. If codegen_profile contains
  wireshark_c.registration.filter_name, use that exact isolated filter name instead of a
  built-in protocol filter name.
- If you call dissector_add_uint/dissector_add_uint_with_preference on a dissector table,
  ensure the table exists (register_dissector_table) OR use the correct built-in table
  like "tcp.port" for TCP-port registration.
""".strip()
    backend_rules = backend_contract_repair_rules("wireshark_c")
    if backend_rules:
        extra_rules += "\n\n" + backend_rules

    prompt_path = logs_root / f"stage_c_wireshark_repair_round{round_idx:02d}_prompt.txt"
    resp_path = logs_root / f"stage_c_wireshark_repair_round{round_idx:02d}_response.txt"
    agent = CodeRepairAgent(llm)
    patched = agent.repair_files(
        files=files,
        diagnostics=RepairDiagnostics(
            tool_name="wireshark_c",
            protocol_name=protocol_name,
            errors=errors_for_prompt,
            log_tail=log_tail_for_prompt,
        ),
        codegen_profile=codegen_profile,
        extra_rules=extra_rules,
        prompt_path=prompt_path,
        response_path=resp_path,
        allow_new_files=False,
    )
    if not patched:
        return False

    changed = False
    for name, content in patched.items():
        out_path = plugin_src_dir / name
        if not out_path.exists():
            # Be strict: do not allow the LLM to create arbitrary new files.
            continue
        try:
            old = out_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            old = ""
        if (content or "").strip() and content.strip() != old.strip():
            try:
                out_path.write_text(content.rstrip() + "\n", encoding="utf-8")
                changed = True
            except Exception:
                pass

    return changed


def _read_plugin_files(plugin_src_dir: Path, protocol_name: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    try:
        entries = sorted(plugin_src_dir.iterdir())
    except OSError:
        entries = []
    for path in entries:
        if not path.is_file() or path.suffix not in {".c", ".h"}:
            continue
        try:
            files[path.name] = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
    return files


def _read_log_tail(path: Optional[Path], max_chars: int = 12000) -> str:
    if path is None:
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(data) <= max_chars:
        return data
    return data[-max_chars:]
