"""Minimal facade package regrouping the core pargen pipeline steps.

Note: Imports are lazy to avoid pulling heavy optional deps (e.g., anthropic)
when only running downstream stages like traffic fixing.
"""

__all__ = [
    "DocumentPipeline",
    "DocumentPipelineConfig",
    "FormatTreePipeline",
    "FormatTreeConfig",
    "StageCConfig",
    "run_stage_c",
]


def __getattr__(name):
    if name in {"DocumentPipeline", "DocumentPipelineConfig"}:
        from .specification_processing.pipeline import DocumentPipeline, DocumentPipelineConfig
        return {"DocumentPipeline": DocumentPipeline, "DocumentPipelineConfig": DocumentPipelineConfig}[name]
    if name in {"FormatTreePipeline", "FormatTreeConfig"}:
        from .protocol_knowledge.pipeline import FormatTreePipeline, FormatTreeConfig
        return {"FormatTreePipeline": FormatTreePipeline, "FormatTreeConfig": FormatTreeConfig}[name]
    if name in {"StageCConfig", "run_stage_c"}:
        from .parser_realization.stage_c_pipeline import StageCConfig, run_stage_c
        return {"StageCConfig": StageCConfig, "run_stage_c": run_stage_c}[name]
    raise AttributeError(f"module 'pargen' has no attribute '{name}'")
