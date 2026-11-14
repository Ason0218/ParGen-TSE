"""Stage C syntax checking + auto repair.

Note: `python -m unittest discover -s pargen` treats this package as a
top-level module (`import parser_realization.syntax_stage`). Parent-relative imports used by
Stage C are only valid under `pargen.parser_realization.*`, so keep this module
import-light unless imported via the full package path.
"""

from __future__ import annotations

if __package__ and __package__.startswith("pargen."):
    from .backend_base import SyntaxBackend
    from .repair_loop import SyntaxRepairLoopConfig, run_syntax_with_auto_repair
    from .schema import SyntaxArtifacts, SyntaxResult

    __all__ = [
        "SyntaxArtifacts",
        "SyntaxResult",
        "SyntaxBackend",
        "SyntaxRepairLoopConfig",
        "run_syntax_with_auto_repair",
    ]
else:  # pragma: no cover
    __all__: list[str] = []
