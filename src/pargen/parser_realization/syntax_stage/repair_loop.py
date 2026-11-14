from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from pargen.paths import DATA_DIR

from .backend_base import SyntaxBackend
from .schema import SyntaxResult
from ..code_repair_agent import CodeRepairAgent, RepairDiagnostics
from ..host_docs import load_host_docs
from ..llm_client import LLMClient, LLMConfig, create_llm_client
from ..profile_utils import profile_violations


@dataclass
class SyntaxRepairLoopConfig:
    max_rounds: int = 5  # B_build = 5
    enable_llm_repair: bool = True
    log_tail_chars: int = 12000
    include_host_docs: bool = True
    host_docs_max_chars: int = 20000


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


def _extract_error_lines_from_log_tail(*, log_tail: str, max_lines: int = 25, max_chars: int = 4000) -> Sequence[str]:
    """Best-effort extraction of actionable error lines from a tool log tail.

    Many toolchains print the real compilation errors only in stdout/stderr, while the backend
    may return a short high-level error like "tool failed". This helper promotes a small set of
    relevant log lines into the "primary" error list for the LLM to act on.
    """
    tail = (log_tail or "").strip("\n")
    if not tail:
        return tuple()

    lines = [ln.rstrip() for ln in tail.splitlines() if ln.strip()]

    def _looks_like_error(line: str) -> bool:
        lower = line.lower()
        if "[error]" in lower or "[fatal]" in lower:
            return True
        if "syntax error" in lower:
            return True
        if re.search(r"\b(fatal error|error:)\b", lower):
            return True
        if "traceback (most recent call last)" in lower:
            return True
        if re.search(r"\b(module not found|modulenotfounderror|importerror)\b", lower):
            return True
        if re.search(r"\b(syntaxerror|typeerror|valueerror|keyerror|attributeerror)\b", lower):
            return True
        # Common compiler location format: file:line:col[-line:col]:
        if re.search(r":[0-9]+:[0-9]+(?:-[0-9]+:[0-9]+)?:", line):
            return True
        return False

    picked: list[str] = []
    seen: set[str] = set()
    for ln in reversed(lines):
        if not _looks_like_error(ln):
            continue
        if ln in seen:
            continue
        seen.add(ln)
        picked.append(ln)
        if len(picked) >= max_lines:
            break
    picked.reverse()

    if not picked:
        return tuple()

    # Enforce a character budget while preserving the most recent messages.
    joined = "\n".join(picked)
    if len(joined) <= max_chars:
        return tuple(picked)

    trimmed: list[str] = []
    total = 0
    for ln in reversed(picked):
        add = len(ln) + 1
        if total + add > max_chars:
            break
        trimmed.append(ln)
        total += add
    trimmed.reverse()
    return tuple(trimmed)


def _scan_source_files(plugin_src_dir: Path, source_globs: Sequence[str]) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for pattern in source_globs:
        for path in sorted(plugin_src_dir.glob(pattern)):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(plugin_src_dir)
                files[rel.as_posix()] = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return files


def _load_host_docs_text(*, host_name: str) -> str:
    host_dir = DATA_DIR / "host_docs" / host_name
    chunks = load_host_docs(host_dir)
    if not chunks:
        return ""

    def _rel_source(source: str) -> str:
        try:
            p = Path(source).resolve()
            return p.relative_to(DATA_DIR).as_posix()
        except Exception:
            return source

    parts: list[str] = []
    for chunk in sorted(chunks, key=lambda c: (c.kind, c.source)):
        parts.append(f"--- {chunk.kind}: {_rel_source(chunk.source)} ---\n{chunk.content.rstrip()}")
    return "\n\n".join(parts).strip()


def run_syntax_with_auto_repair(
    *,
    backend: SyntaxBackend,
    plugin_src_dir: Path,
    protocol_filter_name: str,
    file_prefix: str,
    logs_root: Path,
    codegen_profile: Optional[Dict[str, Any]] = None,
    llm_config: Optional[LLMConfig] = None,
    backend_config: Optional[Dict[str, Any]] = None,
    cfg: Optional[SyntaxRepairLoopConfig] = None,
    source_globs: Optional[Sequence[str]] = None,
    progress_printer: Optional[Callable[[str], None]] = None,
) -> SyntaxResult:
    logs_root.mkdir(parents=True, exist_ok=True)
    cfg = cfg or SyntaxRepairLoopConfig()
    source_globs = source_globs or ("*.c", "*.h")
    if backend_config is None and isinstance(codegen_profile, dict):
        backend_config = codegen_profile
    host_docs_text = ""
    if cfg.include_host_docs:
        try:
            host_docs_text = _load_host_docs_text(host_name=backend.name)
        except Exception:
            host_docs_text = ""

    def _log(message: str) -> None:
        if progress_printer is None:
            return
        try:
            progress_printer(message)
        except Exception:
            return

    llm: Optional[LLMClient] = None
    if cfg.enable_llm_repair:
        try:
            llm = create_llm_client(llm_config)
        except Exception:
            llm = None


    last: Optional[SyntaxResult] = None
    for round_idx in range(cfg.max_rounds):
        _log(f"  [SyntaxRepair] tool={backend.name} round={round_idx:02d}: running syntax check")
        last = backend.check(
            plugin_src_dir=plugin_src_dir,
            protocol_filter_name=protocol_filter_name,
            file_prefix=file_prefix,
            logs_root=logs_root,
            backend_config=backend_config,
        )
        if last.ok:


            # Profile constraints (required/forbidden substrings) are not part of the syntax
            # pass criteria, but we can still attempt a best-effort repair here so that later
            # stages (export/semantic) start from a more compliant baseline.
            if cfg.enable_llm_repair and llm is not None and isinstance(codegen_profile, dict) and codegen_profile:
                files_now = _scan_source_files(plugin_src_dir, source_globs)
                violations = profile_violations(files_now, codegen_profile)
                if violations and files_now:
                    prompt_path = logs_root / f"stage_c_syntax_{backend.name}_profile_round{round_idx:02d}_prompt.txt"
                    resp_path = logs_root / f"stage_c_syntax_{backend.name}_profile_round{round_idx:02d}_response.txt"
                    _log(
                        f"  [SyntaxRepair] tool={backend.name} round={round_idx:02d}: "
                        f"profile repair (prompt={prompt_path}, response={resp_path})"
                    )
                    agent = CodeRepairAgent(llm)
                    patched = agent.repair_files(
                        files=files_now,
                        diagnostics=RepairDiagnostics(
                            tool_name=f"{backend.name}_profile",
                            protocol_name=protocol_filter_name,
                            errors=tuple([f"[PROFILE] {v}" for v in violations]),
                            log_tail=_read_log_tail(last.build_log, max_chars=cfg.log_tail_chars),
                            host_docs=host_docs_text,
                            host_docs_max_chars=int(cfg.host_docs_max_chars),
                        ),
                        codegen_profile=codegen_profile,
                        extra_rules=backend.extra_llm_rules(
                            protocol_filter_name=protocol_filter_name,
                            file_prefix=file_prefix,
                            backend_config=backend_config,
                        ),
                        prompt_path=prompt_path,
                        response_path=resp_path,
                        allow_new_files=False,
                    )
                    if patched:
                        changed_ok = False
                        for name, content in patched.items():
                            path = plugin_src_dir / name
                            if not path.exists() or not path.is_file():
                                continue
                            try:
                                old = path.read_text(encoding="utf-8", errors="ignore")
                            except Exception:
                                old = ""
                            if (content or "").strip() and content.strip() != (old or "").strip():
                                try:
                                    path.write_text(content.rstrip() + "\n", encoding="utf-8")
                                    changed_ok = True
                                except Exception:
                                    continue
                        if changed_ok:
                            _log(
                                f"  [SyntaxRepair] tool={backend.name} round={round_idx:02d}: "
                                f"profile patched_files={sorted(patched.keys())}"
                            )
                            if round_idx == cfg.max_rounds - 1:
                                last = backend.check(
                                    plugin_src_dir=plugin_src_dir,
                                    protocol_filter_name=protocol_filter_name,
                                    file_prefix=file_prefix,
                                    logs_root=logs_root,
                                    backend_config=backend_config,
                                )
                                return last
                            continue

            _log(f"  [SyntaxRepair] tool={backend.name} round={round_idx:02d}: syntax_ok=True")
            return last

        errors = list(last.errors)
        log_path = last.build_log
        log_tail = _read_log_tail(log_path, max_chars=cfg.log_tail_chars)
        for line in _extract_error_lines_from_log_tail(log_tail=log_tail):
            if line not in errors:
                errors.append(line)

        changed = False


        if cfg.enable_llm_repair and llm is not None:
            files = _scan_source_files(plugin_src_dir, source_globs)
            if files:
                repair_files = files
                scapy_structured_view = False
                if backend.name == "scapy":
                    try:
                        from .backends.scapy import make_scapy_structured_repair_view

                        compact = make_scapy_structured_repair_view(files, errors)
                        if compact:
                            repair_files = compact
                            scapy_structured_view = True
                    except Exception:
                        repair_files = files
                        scapy_structured_view = False
                prompt_path = logs_root / f"stage_c_syntax_{backend.name}_round{round_idx:02d}_prompt.txt"
                resp_path = logs_root / f"stage_c_syntax_{backend.name}_round{round_idx:02d}_response.txt"
                _log(
                    f"  [SyntaxRepair] tool={backend.name} round={round_idx:02d}: "
                    f"LLM repair (prompt={prompt_path}, response={resp_path})"
                )
                agent = CodeRepairAgent(llm)
                patched = agent.repair_files(
                    files=repair_files,
                    diagnostics=RepairDiagnostics(
                        tool_name=backend.name,
                        protocol_name=protocol_filter_name,
                        errors=tuple(errors),
                        log_tail=log_tail,
                        host_docs=host_docs_text,
                        host_docs_max_chars=int(cfg.host_docs_max_chars),
                    ),
                    codegen_profile=codegen_profile,
                    extra_rules=backend.extra_llm_rules(
                        protocol_filter_name=protocol_filter_name,
                        file_prefix=file_prefix,
                        backend_config=backend_config,
                    ),
                    prompt_path=prompt_path,
                    response_path=resp_path,
                    allow_new_files=False,
                )
                if patched and scapy_structured_view:
                    try:
                        from .backends.scapy import merge_scapy_structured_repair_view

                        patched = merge_scapy_structured_repair_view(original_files=files, patched_files=patched)
                    except Exception:
                        patched = patched
                if patched:
                    _log(
                        f"  [SyntaxRepair] tool={backend.name} round={round_idx:02d}: "
                        f"patched_files={sorted(patched.keys())}"
                    )
                    for name, content in patched.items():
                        path = plugin_src_dir / name
                        if not path.exists() or not path.is_file():
                            continue
                        try:
                            old = path.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            old = ""
                        if (content or "").strip() and content.strip() != (old or "").strip():
                            try:
                                path.write_text(content.rstrip() + "\n", encoding="utf-8")
                                changed = True
                            except Exception:
                                continue
                    if changed:
                        if round_idx == cfg.max_rounds - 1:
                            last = backend.check(
                                plugin_src_dir=plugin_src_dir,
                                protocol_filter_name=protocol_filter_name,
                                file_prefix=file_prefix,
                                logs_root=logs_root,
                                backend_config=backend_config,
                            )
                            return last
                        continue


        if not changed:
            _log(f"  [SyntaxRepair] tool={backend.name} round={round_idx:02d}: no changes; stopping")
            break

    return last or SyntaxResult(ok=False, runtime_ok=None)
