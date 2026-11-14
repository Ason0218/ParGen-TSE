"""Stage C semantic validation + auto repair.

Note: `python -m unittest discover -s pargen` treats this package as a
top-level module (`import parser_realization.semantic_stage`). Parent-relative imports used
by Stage C are only valid under `pargen.parser_realization.*`, so keep this module
import-light unless imported via the full package path.
"""

from __future__ import annotations

__all__ = [
    "SemanticRoundArtifacts",
    "SemanticRepairLoopConfig",
    "SemanticValidationResult",
    "TreeGTBuildStats",
    "run_semantic_validation",
    "run_stage_c_semantic_with_auto_repair",
]


if not (__package__ and __package__.startswith("pargen.")):  # pragma: no cover
    # Keep import-light when discovered as a top-level package (parser_realization.*).
    __all__ = []


def __getattr__(name: str):  # pragma: no cover
    # Lazy exports to avoid circular imports with tools importing semantic submodules.
    if not (__package__ and __package__.startswith("pargen.")):
        raise AttributeError(name)

    if name in {"SemanticRoundArtifacts", "SemanticValidationResult", "TreeGTBuildStats"}:
        from .schema import SemanticRoundArtifacts, SemanticValidationResult, TreeGTBuildStats

        return {
            "SemanticRoundArtifacts": SemanticRoundArtifacts,
            "SemanticValidationResult": SemanticValidationResult,
            "TreeGTBuildStats": TreeGTBuildStats,
        }[name]

    if name in {"SemanticRepairLoopConfig", "run_stage_c_semantic_with_auto_repair"}:
        from .repair_loop import SemanticRepairLoopConfig, run_stage_c_semantic_with_auto_repair

        return {
            "SemanticRepairLoopConfig": SemanticRepairLoopConfig,
            "run_stage_c_semantic_with_auto_repair": run_stage_c_semantic_with_auto_repair,
        }[name]

    if name == "run_semantic_validation":
        from .validator import run_semantic_validation

        return run_semantic_validation

    raise AttributeError(name)
