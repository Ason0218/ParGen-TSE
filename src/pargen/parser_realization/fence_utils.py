from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Dict


def validate_fenced_filename(name: str) -> str:
    """Return a normalized relative plugin filename or raise ValueError."""
    raw = str(name or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("empty fenced filename")
    if "\x00" in raw or any(ord(ch) < 32 for ch in raw):
        raise ValueError(f"invalid control character in fenced filename: {name!r}")
    if ":" in raw:
        raise ValueError(f"invalid ':' in fenced filename: {name!r}")

    path = PurePosixPath(raw)
    if path.is_absolute():
        raise ValueError(f"absolute fenced filename is not allowed: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"path traversal in fenced filename is not allowed: {name!r}")
    return path.as_posix()


def resolve_plugin_file_path(plugin_dir: Path, name: str) -> Path:
    """Resolve a fenced filename under plugin_dir and guard against traversal."""
    normalized = validate_fenced_filename(name)
    root = plugin_dir.resolve()
    path = (plugin_dir / normalized).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"fenced filename escapes plugin_dir: {name!r}") from exc
    return path


_GENERIC_FENCE_FILENAME_RE = re.compile(
    r"^(?:(?://|#|--|;)\s*|/\*\s*)(?:file\s*:\s*)?([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)(?:\s*\*/)?\s*$",
    re.IGNORECASE,
)


def _filename_from_generic_fence_content(content: str) -> str | None:
    """Recover a filename from the first comment line in a plain code fence."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _GENERIC_FENCE_FILENAME_RE.match(stripped)
        if not match:
            return None
        candidate = match.group(1).strip()
        if candidate != "CMakeLists.txt" and "." not in PurePosixPath(candidate).name:
            return None
        return validate_fenced_filename(candidate)
    return None


def _split_generic_fenced_files(raw: str) -> Dict[str, str]:
    """Fallback for ```c blocks whose first line names the intended file."""
    files: Dict[str, str] = {}
    current_lines: list[str] | None = None

    for line in raw.splitlines(keepends=True):
        stripped = line.strip()
        if current_lines is None:
            if stripped.startswith("```") and not stripped.startswith("```file:"):
                current_lines = []
            continue

        if stripped == "```":
            content = "".join(current_lines).strip()
            filename = _filename_from_generic_fence_content(content)
            if filename is not None:
                files[filename] = content
            current_lines = None
            continue

        current_lines.append(line)

    if current_lines is not None:
        content = "".join(current_lines).strip()
        filename = _filename_from_generic_fence_content(content)
        if filename is not None:
            files[filename] = content

    return files


def split_fenced_files(raw: str) -> Dict[str, str]:
    """Extract ```file:<name>``` fenced blocks into a dict of {filename: content}.

    Be tolerant of malformed LLM output:
    - allow leading spaces before the opening fence,
    - allow leading/trailing spaces around the closing ``` line,
    - if the final fenced file is not closed, consume until EOF.
    - if no explicit file fences exist, recover plain language fences whose
      first non-empty line is a filename comment, such as // packet-demo.c.
    """
    files: Dict[str, str] = {}
    if not raw:
        return files

    current_name: str | None = None
    current_lines: list[str] = []
    saw_fence = False

    for line in raw.splitlines(keepends=True):
        stripped = line.strip()
        if current_name is None:
            if stripped.startswith("```file:"):
                current_name = validate_fenced_filename(stripped[len("```file:") :].strip())
                current_lines = []
                saw_fence = True
            continue

        if stripped == "```":
            files[current_name] = "".join(current_lines).strip()
            current_name = None
            current_lines = []
            continue

        current_lines.append(line)

    if current_name is not None:
        files[current_name] = "".join(current_lines).strip()

    if saw_fence and files:
        return files

    return _split_generic_fenced_files(raw)
