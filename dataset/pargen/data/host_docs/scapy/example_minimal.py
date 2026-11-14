"""Minimal Scapy dissector example (host docs for ParGen Stage C).

This file is intended as a reference for Host-IR learning and should remain small and readable.
"""

from __future__ import annotations

from scapy.fields import ByteField, FieldLenField, ShortField, StrLenField
from scapy.layers.inet import TCP
from scapy.packet import Packet, bind_layers


class ExampleProto(Packet):
    name = "ExampleProto"
    fields_desc = [
        ByteField("msg_type", 0),
        FieldLenField("data_len", None, length_of="data", fmt="B"),
        StrLenField("data", b"", length_from=lambda pkt: int(pkt.data_len or 0)),
        ShortField("checksum", 0),
    ]


# Register BOTH directions on a known port.
_PORT = 5555
bind_layers(TCP, ExampleProto, dport=_PORT)
bind_layers(TCP, ExampleProto, sport=_PORT)

