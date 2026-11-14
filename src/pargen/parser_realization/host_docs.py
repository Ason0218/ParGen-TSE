from __future__ import annotations

"""Host documentation loader for Stage C."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class HostDocChunk:
    source: str  # file path or section name
    kind: str  # "doc" / "header" / "example_code"
    content: str


def load_host_docs(host_docs_dir: Path) -> List[HostDocChunk]:
    """
    Recursively load host documentation/examples into prompt-ready chunks.
    """
    allowed_suffixes = {
        # Docs
        ".md",
        ".txt",
        ".rst",
        # Common C/C++ code
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hh",
        ".hpp",
        # Scripting / codegen targets
        ".py",
        ".lua",
        # Zeek + Spicy
        ".spicy",
        ".evt",
        ".zeek",
        ".sig",
        ".meta",
        # Build/config snippets
        ".cmake",
        ".in",
    }
    chunks: List[HostDocChunk] = []
    if not host_docs_dir.exists():
        return chunks
    for path in sorted(host_docs_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in allowed_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        kind = "doc" if suffix in {".md", ".txt", ".rst"} else "example_code"
        chunks.append(HostDocChunk(source=str(path), kind=kind, content=text))
    return chunks
