"""Syntax backends for Stage C.

Note: `python -m unittest discover -s pargen` treats this package as a
top-level module (`import parser_realization.syntax_stage.backends`). Parent-relative
imports used by Stage C are only valid under `pargen.parser_realization.*`, so keep this
module import-light unless imported via the full package path.
"""

from __future__ import annotations

if __package__ and __package__.startswith("pargen."):
    from .scapy import ScapyBackend
    from .wireshark_c import WiresharkCBackend
    from .wireshark_lua import WiresharkLuaBackend
    from .zeek_spicy import ZeekSpicyBackend

    __all__ = ["ScapyBackend", "WiresharkCBackend", "WiresharkLuaBackend", "ZeekSpicyBackend"]
else:  # pragma: no cover
    __all__: list[str] = []
