"""Standalone tools for platform-level validation of Stage C outputs.

Note: `python -m unittest discover -s pargen` treats this package as a
top-level module (`import parser_realization.target_platforms`). Parent-relative imports used
by Stage C are only valid under `pargen.parser_realization.*`, so keep this module
import-light unless imported via the full package path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if __package__ and __package__.startswith("pargen."):
    __all__ = [
        "SyntaxTestResult",
        "WIRESHARK_DEFAULT_URL",
        "WiresharkSyntaxTester",
        "RepairLoopConfig",
        "run_wireshark_build_with_auto_repair",
    ]

    if TYPE_CHECKING:
        from .wireshark_auto_repair import RepairLoopConfig, run_wireshark_build_with_auto_repair
        from .wireshark_syntax_tester import SyntaxTestResult, WIRESHARK_DEFAULT_URL, WiresharkSyntaxTester

    _EXPORTS: dict[str, tuple[str, str]] = {
        "RepairLoopConfig": ("wireshark_auto_repair", "RepairLoopConfig"),
        "run_wireshark_build_with_auto_repair": ("wireshark_auto_repair", "run_wireshark_build_with_auto_repair"),
        "SyntaxTestResult": ("wireshark_syntax_tester", "SyntaxTestResult"),
        "WIRESHARK_DEFAULT_URL": ("wireshark_syntax_tester", "WIRESHARK_DEFAULT_URL"),
        "WiresharkSyntaxTester": ("wireshark_syntax_tester", "WiresharkSyntaxTester"),
    }

    def __getattr__(name: str) -> Any:
        spec = _EXPORTS.get(name)
        if spec is None:
            raise AttributeError(name)
        mod_name, attr = spec
        module = __import__(f"{__name__}.{mod_name}", fromlist=[attr])
        return getattr(module, attr)

    def __dir__() -> list[str]:
        return sorted(list(globals().keys()) + list(__all__))
else:  # pragma: no cover
    __all__: list[str] = []
