from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .fence_utils import split_fenced_files
from .llm_client import LLMClient
from .profile_utils import profile_max_attempts, profile_violations
from .source_sanity import generated_source_violations


@dataclass(frozen=True)
class RepairDiagnostics:
    tool_name: str
    protocol_name: str
    errors: Tuple[str, ...] = tuple()
    log_tail: str = ""
    log_tail_max_chars: int = 12000
    host_docs: str = ""
    host_docs_max_chars: int = 20000


class CodeRepairAgent:
    """
    A generic code-repair agent that edits existing files based on tool-provided diagnostics.

    Different tools (C build, Python runtime, etc.) can provide different feedback, but the repair
    framework (prompting, parsing file fences, profile constraints) remains shared.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.last_prompt: Optional[str] = None
        self.last_raw_response: Optional[str] = None
        self.last_violations: List[str] = []

    def repair_files(
        self,
        *,
        files: Dict[str, str],
        diagnostics: RepairDiagnostics,
        codegen_profile: Optional[Dict[str, Any]] = None,
        extra_rules: Optional[str] = None,
        max_attempts: Optional[int] = None,
        prompt_path: Optional[Path] = None,
        response_path: Optional[Path] = None,
        allow_new_files: bool = False,
        source_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Returns a mapping of filename -> updated file content for changed files only.
        If the LLM output is unusable or makes no changes, returns None.
        """
        if not files:
            return None

        allowed_names = set(files.keys())
        attempts = max_attempts
        if attempts is None:
            attempts = profile_max_attempts(codegen_profile, default=2, key="max_repair_attempts")
            # If profile doesn't specify repair attempts, fall back to the codegen attempts key.
            if attempts == 2 and isinstance(codegen_profile, dict) and "max_repair_attempts" not in codegen_profile:
                attempts = profile_max_attempts(codegen_profile, default=2, key="max_codegen_attempts")
            attempts = max(1, min(3, int(attempts)))

        base_prompt = self._build_prompt(
            files=files,
            diagnostics=diagnostics,
            codegen_profile=codegen_profile,
            extra_rules=extra_rules,
        )

        last_problem: List[str] = []
        for attempt in range(1, attempts + 1):
            prompt = base_prompt
            if last_problem:
                prompt += (
                    "\n\nREPAIR VIOLATIONS (fix and regenerate ALL provided files; use one of the accepted output formats):\n- "
                    + "\n- ".join(last_problem)
                )

            self.last_prompt = prompt
            if prompt_path:
                self._try_write(prompt_path, prompt, attempt=attempt)

            try:
                if callable(getattr(type(self.llm), 'repair_files_structured', None)):
                    if source_context is None:
                        raw = self.llm.repair_files_structured(prompt, files)
                    else:
                        raw = self.llm.repair_files_structured(prompt, files, source_context=source_context)
                else:
                    raw = self.llm.complete(prompt)
            except Exception as exc:  # noqa: BLE001
                # A bounded client's deferral/failure must reach the job
                # finalizer, not be mistaken for an unsuccessful semantic fix.
                if callable(getattr(type(self.llm), 'repair_files_structured', None)):
                    raise
                self.last_raw_response = f"[LLM ERROR] {type(exc).__name__}: {exc}"
                if response_path:
                    self._try_write(response_path, self.last_raw_response, attempt=attempt)
                return None
            self.last_raw_response = raw
            if response_path:
                self._try_write(response_path, raw, attempt=attempt)

            raw_text = str(raw or "")
            has_diff_output = bool(re.search(r"```(?:diff|patch)\s*\n", raw_text, flags=re.IGNORECASE)) or (
                "\n--- " in f"\n{raw_text}" and "\n+++ " in f"\n{raw_text}"
            )
            patched = _recover_unified_diff_patch(raw_text, files=files, allowed_names=allowed_names)
            if not patched and not has_diff_output:
                patched = split_fenced_files(raw_text)
            if not patched and not has_diff_output:
                patched = _recover_single_unnamed_code_fence(raw_text, allowed_names=allowed_names)
            if not patched:
                if has_diff_output:
                    last_problem = [
                        "the unified diff was present but could not be applied to the current file contents",
                        "do not use ellipses or abbreviated hunks",
                        "prefer complete updated ```file:<filename>``` fences for each changed file",
                    ]
                else:
                    last_problem = ["no usable ```diff``` patch or ```file:...``` blocks found in LLM output"]
                continue

            if not allow_new_files:
                patched = {k: v for k, v in patched.items() if k in allowed_names}
            if not patched:
                last_problem = ["LLM output did not include any allowed filenames"]
                continue

            changed: Dict[str, str] = {}
            merged = dict(files)
            for name, content in patched.items():
                if not (content or "").strip():
                    continue
                if content.strip() != (files.get(name) or "").strip():
                    merged[name] = content.rstrip() + "\n"
                    changed[name] = content.rstrip() + "\n"

            if not changed:
                last_problem = ["LLM output made no changes to any allowed files"]
                continue

            violations = list(profile_violations(merged, codegen_profile))
            violations.extend(generated_source_violations(merged))
            if violations:
                self.last_violations = violations
                last_problem = violations
                continue

            self.last_violations = []
            return changed

        return None

    def _build_prompt(
        self,
        *,
        files: Dict[str, str],
        diagnostics: RepairDiagnostics,
        codegen_profile: Optional[Dict[str, Any]],
        extra_rules: Optional[str],
    ) -> str:
        err_block = "\n".join(diagnostics.errors).strip() or "(no extracted error lines)"
        tail = (diagnostics.log_tail or "").strip()
        try:
            tail_max = int(getattr(diagnostics, "log_tail_max_chars", 12000) or 0)
        except Exception:
            tail_max = 12000
        if tail_max > 0 and len(tail) > tail_max:
            tail = tail[-tail_max:]

        host_docs = (getattr(diagnostics, "host_docs", "") or "").strip()
        try:
            docs_max = int(getattr(diagnostics, "host_docs_max_chars", 20000) or 0)
        except Exception:
            docs_max = 20000
        if docs_max > 0 and len(host_docs) > docs_max:
            head_n = max(0, docs_max // 2)
            tail_n = max(0, docs_max - head_n)
            host_docs = (
                host_docs[:head_n]
                + "\n\n...(host docs truncated)...\n\n"
                + (host_docs[-tail_n:] if tail_n else "")
            ).strip()

        profile_json = "(none)"
        if isinstance(codegen_profile, dict) and codegen_profile:
            profile_json = json.dumps(codegen_profile, ensure_ascii=False, indent=2)

        structured = callable(getattr(type(self.llm), 'repair_files_structured', None))
        # The structured hook owns the exact source view. Including files here
        # as well doubles the request (the production failure was 142168 tokens).
        file_blocks = []
        if not structured:
            for name, content in files.items():
                file_blocks.append(f"```file:{name}\n{(content or '').rstrip()}\n```")
        output_rule = (
            '- Return only the independently complete JSON edits unit requested by the caller. '
            'Use its exact schema; do not emit diffs or whole-file replacements.'
            if structured else
            '- Output ONLY one of these formats; no prose, no extra text:\n'
            '  1. Prefer a small unified diff in a ```diff``` fence when only a few regions change.\n'
            '  2. Otherwise output complete updated files as ```file:filename``` fences.'
        )
        output_format = ('Return only the JSON edits unit using the caller-supplied schema.' if structured else
            '```diff\n--- a/filename\n+++ b/filename\n@@ ...\n-old line\n+new line\n```\n\nor:\n\n'
            '```file:filename\n<complete file content>\n```')

        rules = textwrap.dedent(
            f"""
You are a code repair agent.

Goals:
- Make the code compile/run under the given toolchain.
- Apply the MINIMAL changes needed for correctness.

Hard constraints:
- Do NOT rename files.
- Do NOT invent new files unless explicitly allowed.
- Only edit the existing files provided below.
{output_rule}
            """
        ).strip()

        extra = (extra_rules or "").strip()
        if extra:
            rules += "\n\n" + extra

        files_block = ('See the exact source view in CURRENT UNIT; unshown source is preserved.'
                       if structured else "\n\n".join(file_blocks))
        docs_block = ""
        if host_docs:
            docs_block = f"\n\nHost API docs (reference):\n<host_docs>\n{host_docs}\n</host_docs>\n"
        return textwrap.dedent(
            f"""
{rules}

Tool: {diagnostics.tool_name}
Protocol under test: {diagnostics.protocol_name}

<codegen_profile>
{profile_json}

{docs_block}
Tool errors (primary signal):
{err_block}

Tool log tail (context):
{tail}

Current files:
{files_block}

Output format (strict):
{output_format}
            """
        ).strip()

    def _try_write(self, path: Path, text: str, *, attempt: int) -> None:
        try:
            out_path = path
            if attempt > 1:
                out_path = Path(str(path).replace(".txt", f"_attempt{attempt:02d}.txt"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
        except Exception:
            return


def _recover_single_unnamed_code_fence(raw: str, *, allowed_names: set[str]) -> Dict[str, str]:
    """
    Recover common LLM repair output that uses ```lua / ```python fences without `file:`.

    The prompt still requires file fences, but accepting an unnamed fence is safe when the
    editable target is unambiguous. This avoids wasting repair rounds on formatting-only
    failures.
    """

    if not raw or not allowed_names:
        return {}
    blocks: List[Tuple[str, str]] = []
    for match in re.finditer(r"```(?P<lang>[A-Za-z0-9_+.-]*)\s*\n(?P<body>.*?)\n```", str(raw), flags=re.DOTALL):
        lang = str(match.group("lang") or "").strip().lower()
        if lang.startswith("file:"):
            continue
        body = str(match.group("body") or "").strip()
        if body:
            blocks.append((lang, body))
    if len(blocks) != 1:
        return {}

    lang, body = blocks[0]
    if len(allowed_names) == 1:
        return {next(iter(allowed_names)): body}

    ext_by_lang = {
        "lua": ".lua",
        "python": ".py",
        "py": ".py",
        "c": ".c",
        "cpp": ".cpp",
        "c++": ".cpp",
        "spicy": ".spicy",
        "zeek": ".evt",
    }
    wanted = ext_by_lang.get(lang)
    if not wanted:
        return {}
    candidates = [name for name in allowed_names if str(name).lower().endswith(wanted)]
    if len(candidates) == 1:
        return {candidates[0]: body}
    return {}


_APPLY_PATCH_ENVELOPE_RE = re.compile(r"^\*\*\* (?:Begin Patch|End Patch|End of File|(?:Update|Add|Delete) File:.*)\s*$")


def _strip_apply_patch_envelope(diff_text: str) -> str:
    """Drop apply_patch envelope lines (`*** Begin Patch`, `*** End Patch`, `*** Update File: x`) that
    OpenAI-family models wrap around, or append to, a unified diff; they are not diff content, and one
    of them makes the whole patch unparseable."""
    return "\n".join(line for line in str(diff_text).split("\n") if not _APPLY_PATCH_ENVELOPE_RE.match(line))


def _recover_unified_diff_patch(raw: str, *, files: Dict[str, str], allowed_names: set[str]) -> Dict[str, str]:
    """Apply unified diff output from the repair model to existing files only."""

    if not raw or not files or not allowed_names:
        return {}
    diff_texts: List[str] = []
    for match in re.finditer(r"```(?P<lang>[A-Za-z0-9_+.-]*)\s*\n(?P<body>.*?)\n```", str(raw), flags=re.DOTALL):
        lang = str(match.group("lang") or "").strip().lower()
        if lang not in {"diff", "patch"}:
            continue
        body = str(match.group("body") or "").strip("\n")
        if body:
            diff_texts.append(body)
    if not diff_texts and ("\n--- " in f"\n{raw}" and "\n+++ " in f"\n{raw}"):
        diff_texts.append(str(raw).strip())

    patched: Dict[str, str] = {}
    for diff_text in diff_texts:
        diff_text = _strip_apply_patch_envelope(diff_text)
        applied = _apply_unified_diff_text(diff_text, files=files, allowed_names=allowed_names)
        if not applied:
            applied = _apply_unified_diff_text_by_unique_context(
                diff_text,
                files=files,
                allowed_names=allowed_names,
            )
        if applied:
            patched.update(applied)
    return patched


def _apply_unified_diff_text(diff_text: str, *, files: Dict[str, str], allowed_names: set[str]) -> Dict[str, str]:
    lines = diff_text.splitlines()
    patched: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1
            continue
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            return {}

        old_name = _normalize_diff_path(lines[i][4:].strip())
        new_name = _normalize_diff_path(lines[i + 1][4:].strip())
        name = new_name if new_name in allowed_names else old_name
        if name not in allowed_names or name not in files:
            return {}
        i += 2

        original_text = files.get(name) or ""
        original_had_newline = original_text.endswith("\n")
        original = original_text.splitlines()
        rendered: List[str] = []
        orig_idx = 0
        changed = False

        while i < len(lines):
            line = lines[i]
            if line.startswith("--- "):
                break
            if not line.startswith("@@"):
                i += 1
                continue
            m_hunk = re.match(r"@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@", line)
            if not m_hunk:
                return {}
            hunk_start = max(0, int(m_hunk.group(1)) - 1)
            if hunk_start < orig_idx or hunk_start > len(original):
                return {}
            rendered.extend(original[orig_idx:hunk_start])
            orig_idx = hunk_start
            i += 1

            while i < len(lines):
                hline = lines[i]
                if hline.startswith("@@") or hline.startswith("--- "):
                    break
                if hline.startswith("\\"):
                    i += 1
                    continue
                if not hline:
                    prefix = " "
                    payload = ""
                else:
                    prefix = hline[0]
                    payload = hline[1:]
                if prefix == " ":
                    if orig_idx >= len(original) or original[orig_idx] != payload:
                        return {}
                    rendered.append(payload)
                    orig_idx += 1
                elif prefix == "-":
                    if orig_idx >= len(original) or original[orig_idx] != payload:
                        return {}
                    orig_idx += 1
                    changed = True
                elif prefix == "+":
                    rendered.append(payload)
                    changed = True
                else:
                    return {}
                i += 1

        rendered.extend(original[orig_idx:])
        if changed:
            text = "\n".join(rendered)
            if original_had_newline or text:
                text += "\n"
            patched[name] = text
    return patched


def _apply_unified_diff_text_by_unique_context(
    diff_text: str,
    *,
    files: Dict[str, str],
    allowed_names: set[str],
) -> Dict[str, str]:
    """Apply a unified diff by matching each hunk's old/context block uniquely.

    Repair models often emit correct hunks with stale line numbers after deterministic
    preflight rewrites. This fallback ignores the hunk line numbers, but only applies a
    hunk when the exact old/context block occurs at a single location in the current file.
    """

    lines = diff_text.splitlines()
    patched: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1
            continue
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            return {}

        old_name = _normalize_diff_path(lines[i][4:].strip())
        new_name = _normalize_diff_path(lines[i + 1][4:].strip())
        name = new_name if new_name in allowed_names else old_name
        if name not in allowed_names or name not in files:
            return {}
        i += 2

        current = (files.get(name) or "").splitlines()
        changed = False
        while i < len(lines):
            line = lines[i]
            if line.startswith("--- "):
                break
            if not line.startswith("@@"):
                i += 1
                continue
            i += 1

            old_block: List[str] = []
            new_block: List[str] = []
            while i < len(lines):
                hline = lines[i]
                if hline.startswith("@@") or hline.startswith("--- "):
                    break
                if hline.startswith("\\"):
                    i += 1
                    continue
                if not hline:
                    prefix = " "
                    payload = ""
                else:
                    prefix = hline[0]
                    payload = hline[1:]
                if prefix == " ":
                    old_block.append(payload)
                    new_block.append(payload)
                elif prefix == "-":
                    old_block.append(payload)
                    changed = True
                elif prefix == "+":
                    new_block.append(payload)
                    changed = True
                else:
                    return {}
                i += 1

            if not old_block:
                return {}
            matches: List[int] = []
            span = len(old_block)
            for idx in range(0, len(current) - span + 1):
                if current[idx : idx + span] == old_block:
                    matches.append(idx)
                    if len(matches) > 1:
                        break
            if len(matches) != 1:
                return {}
            pos = matches[0]
            current = current[:pos] + new_block + current[pos + span :]

        if changed:
            text = "\n".join(current)
            if (files.get(name) or "").endswith("\n") or text:
                text += "\n"
            patched[name] = text
    return patched


def _normalize_diff_path(raw: str) -> str:
    path = (raw or "").split("\t", 1)[0].split(" ", 1)[0].strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path
