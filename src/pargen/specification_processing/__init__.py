"""Document processing stage (document ingestion and cleanup).

Note: `python -m unittest discover -s pargen` treats subpackages as top-level
modules (e.g., `import specification_processing`). In that context, parent-relative imports used by
the full pipeline are not valid. Keep this module import-light unless we are
imported as `pargen.specification_processing`.
"""

from __future__ import annotations

if __package__ and __package__.startswith("pargen."):
    from .pipeline import DocumentPipeline, DocumentPipelineConfig

    __all__ = ["DocumentPipeline", "DocumentPipelineConfig"]
else:  # pragma: no cover
    __all__: list[str] = []
