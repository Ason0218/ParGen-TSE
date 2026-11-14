from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SyntaxArtifacts:
    tool_name: str
    build_dir: Optional[Path] = None
    tool_bin: Optional[Path] = None
    plugin_install_dir: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntaxResult:
    ok: bool
    runtime_ok: Optional[bool] = None

    errors: List[str] = field(default_factory=list)
    runtime_errors: List[str] = field(default_factory=list)

    build_log: Optional[Path] = None
    runtime_log: Optional[Path] = None

    artifacts: Optional[SyntaxArtifacts] = None

    build_rc: Optional[int] = None
    runtime_rc: Optional[int] = None

