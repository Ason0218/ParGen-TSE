from __future__ import annotations

"""Shared data models for the parser_realization stage."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FieldBoundary:
    """
    Boundary of a parsed field.

    Bit semantics (wire order):
    - start_bit counts from the beginning of the packet, starting at 0.
    - Bit 0 is the MSB of the first byte on the wire; bits increase in wire order.
    - bit_length is the number of consecutive bits.
    - Non-byte-aligned fields are allowed.

    Aligns with protocol_knowledge output where start positions may be expressed in bits
    (start_bit/bit_length) or bytes (start_byte/byte_length). All values here
    are normalized to bit-based offsets.
    """

    start_bit: int
    bit_length: int

    def __post_init__(self) -> None:
        if self.start_bit < 0 or self.bit_length < 0:
            raise ValueError("FieldBoundary offsets must be non-negative")

    @property
    def start_byte(self) -> int:
        return self.start_bit // 8

    @property
    def byte_length(self) -> int:
        return (self.bit_length + 7) // 8

    def to_dict(self) -> Dict[str, int]:
        return {"start_bit": int(self.start_bit), "bit_length": int(self.bit_length)}

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "FieldBoundary":
        if not isinstance(payload, dict):
            raise TypeError("FieldBoundary payload must be a mapping")
        start_bit = payload.get("start_bit")
        bit_length = payload.get("bit_length")
        if start_bit is None or bit_length is None:
            start_byte = payload.get("start_byte")
            byte_length = payload.get("byte_length") or payload.get("length")
            if start_byte is None or byte_length is None:
                raise ValueError("Missing boundary offsets")
            start_bit = int(start_byte) * 8
            bit_length = int(byte_length) * 8
        return cls(start_bit=int(start_bit), bit_length=int(bit_length))


@dataclass
class PositiveSample:
    """
    Ground-truth parsed result for a real packet from protocol_knowledge.

    packet_id: stable identifier for the frame/packet.
    raw_bytes: original bytes as captured (not mutated).
    path_id / variant_id: routing info for the activated path/variant in the tree.
    field_values: GT values keyed by field name.
    field_boundaries: GT boundaries keyed by field name.
    activated_constraints: optional list of constraint IDs hit by this packet.
    """

    packet_id: str
    raw_bytes: bytes
    path_id: str
    variant_id: Optional[str] = None
    field_values: Dict[str, Any] = field(default_factory=dict)
    field_boundaries: Dict[str, FieldBoundary] = field(default_factory=dict)
    activated_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "raw_hex": self.raw_bytes.hex(),
            "path_id": self.path_id,
            "variant_id": self.variant_id,
            "field_values": self.field_values,
            "field_boundaries": {k: v.to_dict() for k, v in self.field_boundaries.items()},
            "activated_constraints": list(self.activated_constraints),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PositiveSample":
        if not isinstance(payload, dict):
            raise TypeError("PositiveSample payload must be a mapping")
        raw_hex = payload.get("raw_hex") or payload.get("raw_bytes_hex") or payload.get("raw") or payload.get("raw_bytes")
        if isinstance(raw_hex, bytes):
            raw_bytes = raw_hex
        else:
            raw_bytes = bytes.fromhex(str(raw_hex) or "")
        boundaries_payload = payload.get("field_boundaries") or {}
        boundaries: Dict[str, FieldBoundary] = {}
        for name, entry in boundaries_payload.items():
            try:
                boundaries[name] = FieldBoundary.from_mapping(entry)
            except Exception:
                continue
        return cls(
            packet_id=str(payload.get("packet_id") or payload.get("id") or ""),
            raw_bytes=raw_bytes,
            path_id=str(payload.get("path_id") or payload.get("path") or ""),
            variant_id=payload.get("variant_id"),
            field_values=dict(payload.get("field_values") or {}),
            field_boundaries=boundaries,
            activated_constraints=list(payload.get("activated_constraints") or []),
        )


@dataclass
class NegativeSample:
    """
    Mutated packet derived from a real positive sample to violate specific constraints.

    base_packet_id: ID of the source positive sample.
    raw_bytes: mutated payload.
    expected_violated_constraints: constraint IDs intentionally broken.
    note: human-readable mutation description.
    """

    base_packet_id: str
    raw_bytes: bytes
    expected_violated_constraints: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_packet_id": self.base_packet_id,
            "raw_hex": self.raw_bytes.hex(),
            "expected_violated_constraints": list(self.expected_violated_constraints),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NegativeSample":
        if not isinstance(payload, dict):
            raise TypeError("NegativeSample payload must be a mapping")
        raw_hex = payload.get("raw_hex") or payload.get("raw") or payload.get("raw_bytes")
        raw_bytes = raw_hex if isinstance(raw_hex, bytes) else bytes.fromhex(str(raw_hex) or "")
        return cls(
            base_packet_id=str(payload.get("base_packet_id") or payload.get("packet_id") or ""),
            raw_bytes=raw_bytes,
            expected_violated_constraints=list(payload.get("expected_violated_constraints") or []),
            note=str(payload.get("note") or ""),
        )


@dataclass
class ParseResultProxy:
    """
    Normalized view of a parser output, independent of the generated module's internals.
    """

    ok: bool
    fields: Dict[str, Any]
    boundaries: Dict[str, FieldBoundary]
    errors: List[str]
    path_id: Optional[str] = None
    variant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "fields": self.fields,
            "boundaries": {k: v.to_dict() for k, v in self.boundaries.items()},
            "errors": list(self.errors),
            "path_id": self.path_id,
            "variant_id": self.variant_id,
        }


@dataclass
class EvalMetrics:
    """
    Aggregate metrics for parser quality.
    """

    positive_boundary_loss: float = 0.0
    positive_field_loss: float = 0.0
    negative_miss_penalty: float = 0.0
    anomaly_detection_rate: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive_boundary_loss": float(self.positive_boundary_loss),
            "positive_field_loss": float(self.positive_field_loss),
            "negative_miss_penalty": float(self.negative_miss_penalty),
            "anomaly_detection_rate": float(self.anomaly_detection_rate),
            "details": self.details,
        }
