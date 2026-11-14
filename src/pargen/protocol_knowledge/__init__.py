"""Format tree construction stage (structure induction).

Note: `python -m unittest discover -s pargen` treats subpackages as top-level
modules (e.g., `import protocol_knowledge`). In that context, parent-relative imports used
by the full pipeline are not valid. Keep this module import-light unless we are
imported as `pargen.protocol_knowledge`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if __package__ and __package__.startswith("pargen."):
    __all__ = ["FormatTreePipeline", "FormatTreeConfig"]

    if TYPE_CHECKING:
        from .pipeline import FormatTreeConfig, FormatTreePipeline

    def __getattr__(name: str) -> Any:
        if name == "FormatTreePipeline":
            from .pipeline import FormatTreePipeline

            return FormatTreePipeline
        if name == "FormatTreeConfig":
            from .pipeline import FormatTreeConfig

            return FormatTreeConfig
        raise AttributeError(name)

    def __dir__() -> list[str]:
        return sorted(list(globals().keys()) + list(__all__))
else:  # pragma: no cover
    __all__: list[str] = []
