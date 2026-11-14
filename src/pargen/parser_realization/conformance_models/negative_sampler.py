from __future__ import annotations

"""Constraint-guided negative sample generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constraint_utils import constraint_id_for_edge, constraint_id_for_node, is_checksum_field
from .models import FieldBoundary, NegativeSample, PositiveSample


@dataclass
class _ConstraintSpec:
    constraint_id: str
    kind: str  # range | enum | length | condition | checksum | equality | unknown
    field_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    endianness: str = "big"  # "big" or "little"

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


class ConstraintGuidedNegativeSampler:
    """Generate negative samples by mutating real packets to break constraints."""

    def __init__(self, format_tree: Dict[str, Any]) -> None:
        self.format_tree = format_tree
        self.node_name_map, self.node_meta_map = self._build_node_maps(format_tree)
        self.constraints = self._collect_constraints(format_tree)
        self.constraints_by_id = {c.constraint_id: c for c in self.constraints}

    def generate_for_positive(
        self,
        sample: PositiveSample,
        max_per_constraint: int = 1,
    ) -> List[NegativeSample]:
        """
        For each activated constraint (or every known constraint if not specified),
        create up to max_per_constraint mutated packets that intentionally violate it.
        """

        target_ids: Sequence[str]
        if sample.activated_constraints:
            target_ids = sample.activated_constraints
        else:
            target_ids = [c.constraint_id for c in self.constraints]

        output: List[NegativeSample] = []
        for cid in target_ids:
            spec = self.constraints_by_id.get(cid)
            if not spec:
                # Unknown constraint id; fall back to generic flip
                spec = _ConstraintSpec(constraint_id=cid, kind="unknown", field_name=None, params={})
            # One candidate per boundary side, in order, until the per-rule limit is reached.
            seen: set = set()
            for choice in range(max(1, int(max_per_constraint))):
                mutated = self._mutate_for_constraint(sample, spec, choice=choice)
                if mutated is None:
                    break
                if mutated.raw_bytes in seen:
                    continue
                seen.add(mutated.raw_bytes)
                output.append(mutated)
        return output

    # --- Constraint collection helpers -------------------------------------------------
    def _build_node_maps(self, format_tree: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        tree = format_tree.get("protocol_tree") or format_tree
        name_map: Dict[str, str] = {}
        meta_map: Dict[str, Dict[str, Any]] = {}
        for node in tree.get("nodes", []):
            node_id = str(node.get("node_id"))
            name = str(node.get("name") or f"node_{node_id}")
            endianness = self._get_endianness(node)
            name_map[node_id] = name
            meta_map[node_id] = {"name": name, "endianness": endianness, "raw": node}
        return name_map, meta_map

    def _collect_constraints(self, format_tree: Dict[str, Any]) -> List[_ConstraintSpec]:
        tree = format_tree.get("protocol_tree") or format_tree
        nodes = tree.get("nodes", [])
        edges = tree.get("edges", [])
        specs: List[_ConstraintSpec] = []

        for node in nodes:
            node_id = str(node.get("node_id"))
            field = str(node.get("name") or f"node_{node_id}")
            endianness = self._get_endianness(node)
            constraints = node.get("constraints") or []
            for raw in constraints:
                parsed = self._parse_node_constraint(raw)
                if not parsed:
                    continue
                cid, kind, params = parsed
                constraint_id = constraint_id_for_node(field, raw)
                specs.append(_ConstraintSpec(constraint_id=constraint_id or f"{field}_{kind}", kind=kind, field_name=field, params=params, endianness=endianness))

            if is_checksum_field(field):
                cid = constraint_id_for_node(field, "checksum")
                specs.append(_ConstraintSpec(constraint_id=cid, kind="checksum", field_name=field, params={}, endianness=endianness))

        for edge in edges:
            rel = str(edge.get("rel") or "").lower()
            src = str(edge.get("src"))
            dst = str(edge.get("dst"))
            cid = edge.get("id") or edge.get("relation_id") or f"{rel}_{src}_{dst}"
            src_name = self.node_name_map.get(src, src)
            dst_name = self.node_name_map.get(dst, dst)
            endianness = self.node_meta_map.get(src, {}).get("endianness", "big")
            cid = constraint_id_for_edge(rel, src_name, dst_name, edge.get("formula")) or str(cid)
            if rel == "length_of":
                specs.append(
                    _ConstraintSpec(
                        constraint_id=str(cid),
                        kind="length",
                        field_name=src_name,
                        params={"length_src": src_name, "target": dst_name, "formula": edge.get("formula")},
                        endianness=endianness,
                    )
                )
            elif rel == "condition_on":
                specs.append(
                    _ConstraintSpec(
                        constraint_id=str(cid),
                        kind="condition",
                        field_name=src_name,
                        params={"selector": src_name, "target": dst_name, "formula": edge.get("formula")},
                        endianness=endianness,
                    )
                )
        return specs

    def _parse_node_constraint(self, raw: Any) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        lower = text.lower()
        if lower.startswith("range"):
            bounds = self._parse_range(text)
            return None if bounds is None else (text, "range", {"range": bounds})
        if lower.startswith("enum"):
            enums = self._parse_enum(text)
            return None if enums is None else (text, "enum", {"enum": enums})
        if lower.startswith("formula"):
            return (text, "equality", {"formula": text})
        return None

    def _parse_range(self, text: str) -> Optional[Tuple[int, int]]:
        from pargen.protocol_knowledge.tree_utils import parse_numeric_range_constraint_text

        return parse_numeric_range_constraint_text(text)

    def _parse_enum(self, text: str) -> Optional[List[int]]:
        from pargen.protocol_knowledge.tree_utils import parse_numeric_enum_constraint_text

        parsed = parse_numeric_enum_constraint_text(text)
        return list(parsed) if parsed is not None else None

    def _get_endianness(self, node: Dict[str, Any]) -> str:
        if not isinstance(node, dict):
            return "big"
        enc = node.get("encoding") or {}
        for key in ("endianness", "byte_order"):
            if key in node and node.get(key):
                return str(node.get(key)).lower()
            if key in enc and enc.get(key):
                return str(enc.get(key)).lower()
        return "big"

    # --- Mutation helpers --------------------------------------------------------------
    def _mutate_for_constraint(
        self, sample: PositiveSample, spec: _ConstraintSpec, choice: int = 0
    ) -> Optional[NegativeSample]:
        """Violate `spec`; `choice` selects the boundary side (range, enum, length) and is 0 for other kinds."""
        kind = spec.kind
        if kind == "range":
            mutated = self._mutate_range(spec, sample, choice=choice)
        elif kind == "enum":
            mutated = self._mutate_enum(spec, sample, choice=choice)
        elif kind == "length":
            mutated = self._mutate_length(spec, sample, choice=choice)
        elif choice > 0:
            return None
        elif kind == "condition":
            mutated = self._mutate_condition(spec, sample)
        elif kind == "checksum":
            mutated = self._mutate_checksum(spec, sample)
        elif kind == "equality":
            mutated = self._mutate_equality(spec, sample)
        else:
            mutated = self._mutate_generic(sample)

        if mutated is None or mutated == sample.raw_bytes:
            return None
        note = f"violated {spec.kind} constraint {spec.constraint_id}"
        return NegativeSample(
            base_packet_id=sample.packet_id,
            raw_bytes=mutated,
            expected_violated_constraints=[spec.constraint_id] if spec.constraint_id else [],
            note=note,
        )

    def _mutate_range(self, spec: _ConstraintSpec, sample: PositiveSample, choice: int = 0) -> Optional[bytes]:
        field = spec.field_name
        boundary = sample.field_boundaries.get(field) if field else None
        if not boundary:
            return self._mutate_generic(sample) if choice == 0 else None

        lo, hi = spec.params.get("range", (None, None))
        lo_v: Optional[int]
        hi_v: Optional[int]
        try:
            lo_v = int(lo) if lo is not None else None
            hi_v = int(hi) if hi is not None else None
        except Exception:
            lo_v, hi_v = None, None

        raw_bits = self._extract_int(sample.raw_bytes, boundary)
        if raw_bits is None:
            return self._mutate_generic(sample) if choice == 0 else None

        bit_len = boundary.bit_length
        domain_hi = (1 << bit_len) - 1
        current_val = self._interpret_value(raw_bits, bit_len, spec.endianness)

        # Just outside each boundary of [lo, hi] that the field width can represent.
        if lo_v is not None and hi_v is not None:
            if lo_v <= 0 and hi_v >= domain_hi:
                return None
            candidates = [v for v in (hi_v + 1, lo_v - 1) if 0 <= v <= domain_hi]
        elif hi_v is not None:
            candidates = [hi_v + 1] if hi_v + 1 <= domain_hi else []
        elif lo_v is not None:
            candidates = [lo_v - 1] if lo_v - 1 >= 0 else []
        else:
            candidates = [current_val ^ 0xFF]
        if choice >= len(candidates):
            return None
        new_bits = self._encode_value(candidates[choice], bit_len, spec.endianness)
        return self._patch_value(sample.raw_bytes, boundary, new_bits)

    def _mutate_enum(self, spec: _ConstraintSpec, sample: PositiveSample, choice: int = 0) -> Optional[bytes]:
        boundary = sample.field_boundaries.get(spec.field_name or "")
        if not boundary:
            return self._mutate_generic(sample) if choice == 0 else None
        enum_vals = spec.params.get("enum") or []
        raw_bits = self._extract_int(sample.raw_bytes, boundary)
        if raw_bits is None:
            return self._mutate_generic(sample) if choice == 0 else None
        current = self._interpret_value(raw_bits, boundary.bit_length, spec.endianness)
        domain_hi = (1 << boundary.bit_length) - 1 if boundary.bit_length > 0 else 0
        enum_set = {int(v) for v in enum_vals}
        if boundary.bit_length > 0 and len(enum_set) == domain_hi + 1 and min(enum_set, default=0) == 0 \
                and max(enum_set, default=-1) == domain_hi:
            return None
        # Just outside the enumerated set: above its largest member, in its first interior gap, below its smallest.
        if enum_set:
            lo, hi = min(enum_set), max(enum_set)
            gap = next((v for v in range(lo + 1, hi) if v not in enum_set), None)
            ordered = [hi + 1, gap if gap is not None else lo - 1]
            if gap is not None:
                ordered.append(lo - 1)
        else:
            ordered = [current + 1]
        candidates = list(dict.fromkeys(v for v in ordered if v is not None and 0 <= v <= domain_hi and v not in enum_set))
        if choice >= len(candidates):
            return None
        new_bits = self._encode_value(candidates[choice], boundary.bit_length, spec.endianness)
        return self._patch_value(sample.raw_bytes, boundary, new_bits)

    def _mutate_length(self, spec: _ConstraintSpec, sample: PositiveSample, choice: int = 0) -> Optional[bytes]:
        length_field = spec.params.get("length_src") or spec.field_name
        boundary = sample.field_boundaries.get(length_field or "")
        if boundary:
            raw_bits = self._extract_int(sample.raw_bytes, boundary)
            if raw_bits is None:
                return self._mutate_generic(sample) if choice == 0 else None
            current = self._interpret_value(raw_bits, boundary.bit_length, spec.endianness)
            domain_hi = (1 << boundary.bit_length) - 1
            # One past the stated length on each side.
            candidates = [v for v in (current + 1, current - 1) if 0 <= v <= domain_hi]
            if choice >= len(candidates):
                return None
            new_bits = self._encode_value(candidates[choice], boundary.bit_length, spec.endianness)
            return self._patch_value(sample.raw_bytes, boundary, new_bits)
        # fallback: truncate payload
        if choice == 0 and len(sample.raw_bytes) > 1:
            return sample.raw_bytes[:-1]
        return None

    def _mutate_condition(self, spec: _ConstraintSpec, sample: PositiveSample) -> Optional[bytes]:
        selector = spec.params.get("selector") or spec.field_name
        boundary = sample.field_boundaries.get(selector or "")
        if not boundary:
            return self._mutate_generic(sample)

        raw_bits = self._extract_int(sample.raw_bytes, boundary)
        if raw_bits is None:
            return self._mutate_generic(sample)

        bit_len = boundary.bit_length
        current = self._interpret_value(raw_bits, bit_len, spec.endianness)
        formula = (spec.params.get("formula") or "").strip()

        import re

        m = re.search(r"val\((\d+)\)\s*==\s*([01])", formula)
        if m:
            bit_index = int(m.group(1))
            expected = int(m.group(2))

            if bit_index >= bit_len:
                return self._mutate_generic(sample)

            mask = 1 << bit_index
            if expected == 1:
                new_val = current & ~mask
            else:
                new_val = current | mask

            new_bits = self._encode_value(new_val, bit_len, spec.endianness)
            return self._patch_value(sample.raw_bytes, boundary, new_bits)

        target_val = None
        for token in ("==", "in"):
            if token in formula:
                try:
                    rhs = formula.split(token, 1)[1]
                    rhs = rhs.strip().strip("[](){} ")
                    if rhs:
                        parts = rhs.split(",")
                        target_val = int(parts[0])
                except Exception:
                    pass

        new_val = current + 1 if target_val is None else target_val + 1
        new_bits = self._encode_value(new_val, bit_len, spec.endianness)
        return self._patch_value(sample.raw_bytes, boundary, new_bits)

    def _mutate_checksum(self, spec: _ConstraintSpec, sample: PositiveSample) -> Optional[bytes]:
        boundary = sample.field_boundaries.get(spec.field_name or "")
        if boundary:
            raw_bits = self._extract_int(sample.raw_bytes, boundary)
            if raw_bits is None:
                return self._mutate_generic(sample)
            current = self._interpret_value(raw_bits, boundary.bit_length, spec.endianness)
            new_bits = self._encode_value(current ^ 0xFF, boundary.bit_length, spec.endianness)
            return self._patch_value(sample.raw_bytes, boundary, new_bits)
        return self._mutate_generic(sample)

    def _mutate_equality(self, spec: _ConstraintSpec, sample: PositiveSample) -> Optional[bytes]:
        # Simple strategy: flip the referenced field if known
        field = spec.field_name or next(iter(sample.field_boundaries.keys()), None)
        if not field:
            return self._mutate_generic(sample)
        boundary = sample.field_boundaries.get(field)
        if not boundary:
            return self._mutate_generic(sample)
        raw_bits = self._extract_int(sample.raw_bytes, boundary)
        if raw_bits is None:
            return self._mutate_generic(sample)
        current = self._interpret_value(raw_bits, boundary.bit_length, spec.endianness)
        new_bits = self._encode_value(current + 1, boundary.bit_length, spec.endianness)
        return self._patch_value(sample.raw_bytes, boundary, new_bits)

    def _mutate_generic(self, sample: PositiveSample) -> bytes:
        if not sample.raw_bytes:
            return b""
        mutated = bytearray(sample.raw_bytes)
        mutated[0] = mutated[0] ^ 0xFF
        return bytes(mutated)

    # --- Bit utilities ----------------------------------------------------------------
    def _patch_value(self, raw: bytes, boundary: FieldBoundary, new_value: int) -> Optional[bytes]:
        start_bit = boundary.start_bit
        bit_len = boundary.bit_length

        if bit_len <= 0:
            return None

        total_bits = len(raw) * 8
        if start_bit + bit_len > total_bits:
            return None

        data_int = int.from_bytes(raw, byteorder="big", signed=False)
        new_value &= (1 << bit_len) - 1
        shift = total_bits - (start_bit + bit_len)
        mask = ((1 << bit_len) - 1) << shift

        new_int = (data_int & ~mask) | (new_value << shift)
        return new_int.to_bytes(len(raw), byteorder="big", signed=False)

    def _extract_int(self, raw: bytes, boundary: FieldBoundary) -> Optional[int]:
        start_bit = boundary.start_bit
        bit_len = boundary.bit_length

        if bit_len <= 0:
            return None

        total_bits = len(raw) * 8
        if start_bit + bit_len > total_bits:
            return None

        data_int = int.from_bytes(raw, byteorder="big", signed=False)
        shift = total_bits - (start_bit + bit_len)
        mask = (1 << bit_len) - 1
        return (data_int >> shift) & mask

    def _interpret_value(self, raw_bits: int, bit_length: int, endianness: str) -> int:
        """Convert wire-order bits to semantic value according to endianness."""
        if endianness == "big":
            return raw_bits
        if endianness == "little" and bit_length % 8 == 0:
            byte_len = bit_length // 8
            b = raw_bits.to_bytes(byte_len, "big", signed=False)
            rev = b[::-1]
            return int.from_bytes(rev, "big", signed=False)
        return raw_bits

    def _encode_value(self, semantic_val: int, bit_length: int, endianness: str) -> int:
        """Convert semantic value to wire-order bits according to endianness."""
        semantic_val &= (1 << bit_length) - 1
        if endianness == "big":
            return semantic_val
        if endianness == "little" and bit_length % 8 == 0:
            byte_len = bit_length // 8
            b = semantic_val.to_bytes(byte_len, "big", signed=False)
            rev = b[::-1]
            return int.from_bytes(rev, "big", signed=False) & ((1 << bit_length) - 1)
        return semantic_val
