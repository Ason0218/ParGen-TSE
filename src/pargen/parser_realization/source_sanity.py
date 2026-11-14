from __future__ import annotations

import re
from typing import List, Mapping, Optional, Tuple


def generated_source_violations(files: Mapping[str, str]) -> List[str]:
    """Return static completeness violations for generated source files.

    This is intentionally lightweight. It catches common LLM truncation failures
    before expensive host builds, without trying to replace a C compiler.
    """

    violations: List[str] = []
    if not isinstance(files, dict):
        return violations

    for name, content in files.items():
        if not _is_c_source_name(str(name or "")):
            continue
        if not isinstance(content, str) or not content.strip():
            violations.append(f"source_sanity:{name}: empty C source")
            continue
        violations.extend(f"source_sanity:{name}: {msg}" for msg in _scan_c_like_source(content))

    return violations


def _is_c_source_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith((".c", ".h", ".cc", ".cpp", ".hpp"))


def _scan_c_like_source(text: str) -> List[str]:
    messages: List[str] = []
    stack: List[Tuple[str, int, int]] = []
    line = 1
    col = 0
    i = 0
    state = "code"
    escaped = False
    comment_start: Optional[Tuple[int, int]] = None
    string_start: Optional[Tuple[int, int]] = None
    char_start: Optional[Tuple[int, int]] = None
    line_has_code = False
    preprocessor = False

    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")": "(", "]": "[", "}": "{"}

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        col += 1

        if ch == "\n":
            if state in {"string", "char"} and not escaped:
                messages.append(f"unescaped newline in {state} literal at line {line}")
            continued = i > 0 and text[i - 1] == "\\"
            line += 1
            col = 0
            if state == "line_comment":
                state = "code"
            i += 1
            escaped = False
            if not continued:
                line_has_code = False
                preprocessor = False
            continue

        if state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                col += 1
                continue
            i += 1
            continue

        if state == "line_comment":
            i += 1
            continue

        if state == "string":
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                state = "code"
            i += 1
            continue

        if state == "char":
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                state = "code"
            i += 1
            continue

        if ch == "/" and nxt == "*":
            state = "block_comment"
            comment_start = (line, col)
            i += 2
            col += 1
            continue
        if ch == "/" and nxt == "/":
            state = "line_comment"
            i += 2
            col += 1
            continue
        if ch == '"':
            line_has_code = True
            state = "string"
            string_start = (line, col)
            escaped = False
            i += 1
            continue
        if ch == "'":
            line_has_code = True
            state = "char"
            char_start = (line, col)
            escaped = False
            i += 1
            continue

        if ch == "#":
            directive = re.match(r"#\s*(include|define|if|ifdef|ifndef|endif|else|elif|pragma)\b", text[i:])
            if directive and line_has_code:
                messages.append(f"preprocessor directive must start a logical line at line {line}, column {col}")
            if not line_has_code:
                preprocessor = True
            include = re.match(r'#\s*include\s*(?:<[^>\n]+>|"[^"\n]+")([^\n]*)', text[i:])
            if include:
                trailing = re.sub(r"/\*.*?\*/", "", include.group(1))
                trailing = trailing.split("//", 1)[0].strip()
                if trailing and not trailing.startswith("/*"):
                    messages.append(f"code or another directive after #include at line {line}")
        if not ch.isspace() and not (preprocessor and ch == "\\"):
            line_has_code = True
        if ch in pairs:
            stack.append((ch, line, col))
        elif ch in closing:
            expected_open = closing[ch]
            if not stack:
                messages.append(f"unmatched closing {ch!r} at line {line}, column {col}")
            else:
                got, got_line, got_col = stack.pop()
                if got != expected_open:
                    messages.append(
                        f"mismatched closing {ch!r} at line {line}, column {col}; "
                        f"opened with {got!r} at line {got_line}, column {got_col}"
                    )
        i += 1

    if state == "block_comment":
        lno, cno = comment_start or (line, col)
        messages.append(f"unterminated block comment opened at line {lno}, column {cno}")
    elif state == "string":
        lno, cno = string_start or (line, col)
        messages.append(f"unterminated string literal opened at line {lno}, column {cno}")
    elif state == "char":
        lno, cno = char_start or (line, col)
        messages.append(f"unterminated char literal opened at line {lno}, column {cno}")

    for opener, lno, cno in reversed(stack[-10:]):
        messages.append(f"unclosed {opener!r} opened at line {lno}, column {cno}")
    if len(stack) > 10:
        messages.append(f"{len(stack) - 10} additional unclosed delimiters")

    return messages
