from __future__ import annotations

import json
import re
import struct
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pargen.parser_realization.conformance_models.constraint_utils import (
    constraint_id_for_edge,
    constraint_id_for_node,
    is_checksum_field,
)
from pargen.parser_realization.conformance_models.models import FieldBoundary, NegativeSample, PositiveSample
from pargen.parser_realization.conformance_models.negative_sampler import ConstraintGuidedNegativeSampler
from pargen.parser_realization.target_platforms.extraction_profile import infer_ip_proto, infer_pcap_layer
from pargen.protocol_knowledge.tree_utils import normalize_protocol_tree

_PCAP_MAGIC_PCAPNG = b"\x0a\x0d\x0d\x0a"
_PCAP_MAGIC_LE = {b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"}
_PCAP_MAGIC_BE = {b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"}

_PCAPNG_SHB = 0x0A0D0D0A
_PCAPNG_IDB = 0x00000001
_PCAPNG_SPB = 0x00000003
_PCAPNG_EPB = 0x00000006
_PCAPNG_BOM_BE = b"\x1a\x2b\x3c\x4d"
_PCAPNG_BOM_LE = b"\x4d\x3c\x2b\x1a"

_DLT_EN10MB = 1
_DLT_RAW = 101
_DLT_LINUX_SLL = 113
_DLT_IPV4 = 228
_DLT_IPV6 = 229
_DLT_LINUX_SLL2 = 276

_ETHERTYPE_IPV4 = 0x0800
_ETHERTYPE_IPV6 = 0x86DD
_ETHERTYPE_VLAN = {0x8100, 0x88A8, 0x9100}

_IPPROTO_TCP = 6
_IPPROTO_UDP = 17


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


@dataclass(frozen=True)
class TrafficPositiveExample:
    sample: PositiveSample
    frame_no: int


@dataclass(frozen=True)
class GeneratedConstraintSuite:
    protocol: str
    suite_dir: Path
    mapping_path: Path
    pcap_layer: str
    ip_proto: Optional[int]
    positive_samples_total: int
    positive_samples_usable: int
    constraints_total: int
    constraints_seen_in_traffic: int
    cases_generated: int
    skipped_no_positive: int
    skipped_no_mutation: int
    skipped_no_frame: int
    skipped_unsupported_frame: int
    skipped_tautological: int = 0
    skipped_inactive_condition: int = 0
    skipped_invalid_mutation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "suite_dir": str(self.suite_dir),
            "mapping_path": str(self.mapping_path),
            "pcap_layer": self.pcap_layer,
            "ip_proto": self.ip_proto,
            "positive_samples_total": self.positive_samples_total,
            "positive_samples_usable": self.positive_samples_usable,
            "constraints_total": self.constraints_total,
            "constraints_seen_in_traffic": self.constraints_seen_in_traffic,
            "cases_generated": self.cases_generated,
            "skipped_no_positive": self.skipped_no_positive,
            "skipped_no_mutation": self.skipped_no_mutation,
            "skipped_no_frame": self.skipped_no_frame,
            "skipped_unsupported_frame": self.skipped_unsupported_frame,
            "skipped_tautological": self.skipped_tautological,
            "skipped_inactive_condition": self.skipped_inactive_condition,
            "skipped_invalid_mutation": self.skipped_invalid_mutation,
        }


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _build_node_lookup(format_tree: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    tree = format_tree.get("protocol_tree") or format_tree
    lookup: Dict[str, str] = {}
    constraint_map: Dict[str, List[str]] = {}
    for node in tree.get("nodes", []):
        if not isinstance(node, dict):
            continue
        nid = str(node.get("node_id"))
        lookup[nid] = str(node.get("name") or nid)
        raw_constraints = node.get("constraints") or []
        if isinstance(raw_constraints, list):
            constraint_map[nid] = [str(c) for c in raw_constraints if isinstance(c, str)]
    return lookup, constraint_map


def _compute_activated_constraints(
    record: Dict[str, Any],
    node_map: Dict[str, str],
    constraint_map: Dict[str, List[str]],
    edges: List[Dict[str, Any]],
) -> List[str]:
    constraints: List[str] = []
    visited_nodes = {str(n) for n in (record.get("visited_nodes") or [])}
    values = record.get("values") or {}
    if isinstance(values, Mapping):
        visited_nodes.update(str(n) for n in values.keys())
    else:
        values = {}
    for nid_str in values.keys():
        name = node_map.get(str(nid_str), str(nid_str))
        node_constraints = constraint_map.get(str(nid_str), [])
        for raw in node_constraints:
            constraints.append(constraint_id_for_node(name, raw))
        if is_checksum_field(name):
            constraints.append(constraint_id_for_node(name, "checksum"))

    for edge in edges:
        rel = str(edge.get("rel") or "").lower()
        src = str(edge.get("src"))
        dst = str(edge.get("dst"))
        if rel == "condition_on":
            if src not in visited_nodes or dst not in visited_nodes:
                continue
        elif src not in visited_nodes and dst not in visited_nodes:
            continue
        src_name = node_map.get(src, src)
        dst_name = node_map.get(dst, dst)
        constraints.append(constraint_id_for_edge(rel, src_name, dst_name, edge.get("formula")))
    return list(dict.fromkeys(constraints))


def _load_success_positive_examples(
    *,
    format_tree: Dict[str, Any],
    traces_path: Path,
) -> List[TrafficPositiveExample]:
    node_map, constraint_map = _build_node_lookup(format_tree)
    edges = (format_tree.get("protocol_tree") or format_tree).get("edges", [])
    out: List[TrafficPositiveExample] = []
    for record in _iter_jsonl(traces_path):
        success = record.get("success")
        if success is False:
            continue
        raw_hex = str(record.get("raw_hex") or "").strip()
        if not raw_hex:
            continue
        try:
            raw_bytes = bytes.fromhex(raw_hex)
        except Exception:
            continue
        packet_index = record.get("packet_index")
        try:
            frame_no = int(packet_index) + 1
        except Exception:
            continue

        parsing_path = record.get("parsing_path") or []
        path_names = [node_map.get(str(nid), str(nid)) for nid in parsing_path]
        path_id = "->".join(path_names) if path_names else "unknown_path"

        variant_id = None
        active_variants = record.get("active_variants") or []
        try:
            if isinstance(active_variants, (list, tuple)) and active_variants:
                variant_id = str(active_variants[0])
            elif isinstance(active_variants, dict) and active_variants:
                variant_id = str(next(iter(active_variants.values())))
            elif isinstance(active_variants, str):
                variant_id = active_variants
        except Exception:
            variant_id = None

        field_values: Dict[str, Any] = {}
        boundaries: Dict[str, FieldBoundary] = {}
        values = record.get("typed_values") or record.get("values") or {}
        bit_starts = record.get("bit_starts") or {}
        sizes = record.get("sizes") or {}
        if not isinstance(values, Mapping):
            values = {}
        if not isinstance(bit_starts, Mapping):
            bit_starts = {}
        if not isinstance(sizes, Mapping):
            sizes = {}
        for nid_str, val in values.items():
            name = node_map.get(str(nid_str), str(nid_str))
            field_values[name] = val
            b_start = bit_starts.get(str(nid_str))
            b_size = sizes.get(str(nid_str))
            if b_start is None or b_size is None:
                continue
            try:
                boundaries[name] = FieldBoundary(start_bit=int(b_start), bit_length=int(b_size))
            except Exception:
                continue

        activated_constraints = _compute_activated_constraints(record, node_map, constraint_map, edges)
        sample = PositiveSample(
            packet_id=str(packet_index),
            raw_bytes=raw_bytes,
            path_id=path_id,
            variant_id=variant_id,
            field_values=field_values,
            field_boundaries=boundaries,
            activated_constraints=activated_constraints,
        )
        out.append(TrafficPositiveExample(sample=sample, frame_no=frame_no))
    return out


def _build_constraints_catalog(format_tree: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tree = format_tree.get("protocol_tree") or format_tree
    nodes = tree.get("nodes", []) or []
    edges = tree.get("edges", []) or []
    catalog: Dict[str, Dict[str, Any]] = {}
    node_name_map: Dict[str, str] = {}
    node_endianness_map: Dict[str, str] = {}

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id"))
        node_name = str(node.get("name") or node_id)
        node_name_map[node_id] = node_name
        endianness = _get_node_endianness(node)
        node_endianness_map[node_id] = endianness
        size_bits = _safe_int(node.get("size_bits") if node.get("size_bits") is not None else node.get("bit_length"))
        bit_start = _safe_int(node.get("bit_start"))
        for raw in node.get("constraints") or []:
            if not isinstance(raw, str):
                continue
            cid = constraint_id_for_node(node_name, raw)
            constraint_ir = _parse_constraint_ir(raw)
            related_field_names: List[str] = []
            if constraint_ir:
                for field_id in constraint_ir.get("field_ids") or []:
                    name = node_name_map.get(str(field_id))
                    if name and name not in related_field_names:
                        related_field_names.append(name)
                for name in constraint_ir.get("field_names") or []:
                    if isinstance(name, str) and name and name in node_name_map.values() and name not in related_field_names:
                        related_field_names.append(name)
            entry = {
                "kind": "node_constraint",
                "node_id": int(node_id) if node_id.isdigit() else node_id,
                "node_name": node_name,
                "constraint": raw,
                "endianness": endianness,
                "size_bits": size_bits,
                "bit_start": bit_start,
                "present_in_tree": True,
            }
            if constraint_ir:
                entry["constraint_ir"] = constraint_ir
                entry["related_field_names"] = related_field_names
            catalog[cid] = entry
        if is_checksum_field(node_name):
            cid = constraint_id_for_node(node_name, "checksum")
            catalog[cid] = {
                "kind": "node_constraint",
                "node_id": int(node_id) if node_id.isdigit() else node_id,
                "node_name": node_name,
                "constraint": "checksum",
                "endianness": endianness,
                "size_bits": size_bits,
                "bit_start": bit_start,
                "present_in_tree": True,
            }

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        rel = str(edge.get("rel") or "").lower().strip()
        if rel not in {"length_of", "condition_on"}:
            continue
        src = str(edge.get("src"))
        dst = str(edge.get("dst"))
        src_name = node_name_map.get(src, src)
        dst_name = node_name_map.get(dst, dst)
        cid = constraint_id_for_edge(rel, src_name, dst_name, edge.get("formula"))
        catalog[cid] = {
            "kind": "edge_constraint",
            "src": int(src) if src.isdigit() else src,
            "dst": int(dst) if dst.isdigit() else dst,
            "src_name": src_name,
            "dst_name": dst_name,
            "rel": rel,
            "formula": edge.get("formula"),
            "endianness": node_endianness_map.get(src, "big"),
            "present_in_tree": True,
        }
    return catalog


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _get_node_endianness(node: Mapping[str, Any]) -> str:
    enc = node.get("encoding") if isinstance(node.get("encoding"), Mapping) else {}
    for key in ("endianness", "byte_order"):
        value = node.get(key) or enc.get(key)
        if value:
            text = str(value).strip().lower()
            if text in {"little", "le", "little_endian", "little-endian"}:
                return "little"
            return "big"
    return "big"


def _parse_ints(text: str) -> List[int]:
    return [int(v) for v in re.findall(r"-?\d+", str(text or ""))]


def _parse_range_constraint(text: str) -> Optional[Tuple[int, int]]:
    values = _parse_ints(text)
    if len(values) < 2:
        return None
    lo, hi = values[0], values[1]
    if lo >= 0 and hi < 0:
        hi = abs(hi)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parse_enum_constraint(text: str) -> Optional[List[int]]:
    values = _parse_ints(text)
    return values or None


def _parse_constraint_ir(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw.lower().startswith("constraint_ir"):
        return None
    payload = raw.split(":", 1)[1].strip() if ":" in raw else ""
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_int(raw: bytes, boundary: FieldBoundary) -> Optional[int]:
    bit_len = int(boundary.bit_length)
    start_bit = int(boundary.start_bit)
    if bit_len <= 0:
        return None
    total_bits = len(raw) * 8
    if start_bit < 0 or start_bit + bit_len > total_bits:
        return None
    data_int = int.from_bytes(raw, byteorder="big", signed=False)
    shift = total_bits - (start_bit + bit_len)
    return (data_int >> shift) & ((1 << bit_len) - 1)


def _interpret_value(raw_bits: int, bit_length: int, endianness: str) -> int:
    if str(endianness or "big").lower().startswith("little") and bit_length % 8 == 0:
        byte_len = bit_length // 8
        return int.from_bytes(raw_bits.to_bytes(byte_len, "big", signed=False)[::-1], "big", signed=False)
    return int(raw_bits)


def _field_value_from_bytes(raw: bytes, boundary: FieldBoundary, endianness: str) -> Optional[int]:
    raw_bits = _extract_int(raw, boundary)
    if raw_bits is None:
        return None
    return _interpret_value(raw_bits, int(boundary.bit_length), endianness)


def _encode_value(semantic_value: int, bit_length: int, endianness: str) -> int:
    semantic_value = int(semantic_value) & ((1 << int(bit_length)) - 1)
    if str(endianness or "big").lower().startswith("little") and int(bit_length) % 8 == 0:
        byte_len = int(bit_length) // 8
        return int.from_bytes(semantic_value.to_bytes(byte_len, "big", signed=False)[::-1], "big", signed=False)
    return semantic_value


def _patch_field_value(raw: bytes, boundary: FieldBoundary, semantic_value: int, endianness: str) -> Optional[bytes]:
    bit_len = int(boundary.bit_length)
    start_bit = int(boundary.start_bit)
    if bit_len <= 0:
        return None
    total_bits = len(raw) * 8
    if start_bit < 0 or start_bit + bit_len > total_bits:
        return None
    new_bits = _encode_value(int(semantic_value), bit_len, endianness)
    data_int = int.from_bytes(raw, byteorder="big", signed=False)
    shift = total_bits - (start_bit + bit_len)
    mask = ((1 << bit_len) - 1) << shift
    new_int = (data_int & ~mask) | ((new_bits & ((1 << bit_len) - 1)) << shift)
    return new_int.to_bytes(len(raw), byteorder="big", signed=False)


def _field_value_for_name(sample: PositiveSample, field_name: str, endianness: str) -> Optional[int]:
    boundary = sample.field_boundaries.get(str(field_name or ""))
    if boundary is None:
        return None
    return _field_value_from_bytes(sample.raw_bytes, boundary, endianness)


def _formula_rhs_values(formula: Any) -> List[int]:
    return [int(v) for v in re.findall(r"val\(\s*\d+\s*\)\s*(?:==|!=|<=|>=|<|>)\s*(-?\d+)", str(formula or ""))]


def _selector_value_to_make_formula_false(formula: Any, current: int, bit_length: int) -> Optional[int]:
    domain_hi = _domain_hi(int(bit_length))
    if domain_hi is None:
        return None
    candidates: List[int] = []
    rhs_values = _formula_rhs_values(formula)
    for value in rhs_values:
        candidates.extend([value, value + 1, value - 1])
    candidates.extend([0, 1, int(current) + 1, int(current) - 1, int(domain_hi)])
    seen: set[int] = set()
    for candidate in candidates:
        if candidate in seen or candidate < 0 or candidate > domain_hi:
            continue
        seen.add(candidate)
        active = _eval_selector_formula(formula, int(candidate))
        if active is False:
            return int(candidate)
    return None


def _mutate_field_by_name(
    *,
    sample: PositiveSample,
    field_name: str,
    endianness: str,
    prefer_value: Optional[int] = None,
) -> Optional[bytes]:
    boundary = sample.field_boundaries.get(str(field_name or ""))
    if boundary is None:
        return None
    before = _field_value_from_bytes(sample.raw_bytes, boundary, endianness)
    if before is None:
        return None
    domain_hi = _domain_hi(int(boundary.bit_length))
    if domain_hi is None:
        return None
    if prefer_value is not None:
        candidate = int(prefer_value) & int(domain_hi)
        if candidate != int(before):
            return _patch_field_value(sample.raw_bytes, boundary, candidate, endianness)
    candidate = (int(before) + 1) & int(domain_hi)
    if candidate == int(before):
        candidate = int(before) ^ int(domain_hi)
    if candidate == int(before):
        return None
    return _patch_field_value(sample.raw_bytes, boundary, candidate, endianness)


def _changed_inside_boundary(before: bytes, after: bytes, boundary: FieldBoundary) -> bool:
    if len(before) != len(after):
        return True
    start = int(boundary.start_bit)
    end = start + int(boundary.bit_length)
    total_bits = len(before) * 8
    for bit_idx in range(max(0, start), min(end, total_bits)):
        byte_idx = bit_idx // 8
        bit_in_byte = 7 - (bit_idx % 8)
        mask = 1 << bit_in_byte
        if (before[byte_idx] & mask) != (after[byte_idx] & mask):
            return True
    return False


def _domain_hi(bit_length: int) -> Optional[int]:
    if bit_length <= 0:
        return None
    return (1 << int(bit_length)) - 1


def _enum_covers_domain(values: Sequence[int], *, bit_length: int) -> bool:
    domain_hi = _domain_hi(bit_length)
    if domain_hi is None:
        return False
    unique = {int(v) for v in values}
    return bool(unique) and len(unique) == domain_hi + 1 and min(unique) == 0 and max(unique) == domain_hi


def _range_covers_domain(bounds: Tuple[int, int], *, bit_length: int) -> bool:
    domain_hi = _domain_hi(bit_length)
    if domain_hi is None:
        return False
    lo, hi = bounds
    return int(lo) <= 0 and int(hi) >= int(domain_hi)


def _eval_selector_formula(formula: Any, selector_value: int) -> Optional[bool]:
    text = str(formula or "").strip()
    if not text:
        return None

    def eval_atom(atom: str) -> Optional[bool]:
        m = re.search(r"val\(\s*\d+\s*\)\s*(==|!=|<=|>=|<|>)\s*(-?\d+)", atom.strip())
        if not m:
            return None
        op, rhs_s = m.group(1), m.group(2)
        rhs = int(rhs_s)
        if op == "==":
            return selector_value == rhs
        if op == "!=":
            return selector_value != rhs
        if op == "<=":
            return selector_value <= rhs
        if op == ">=":
            return selector_value >= rhs
        if op == "<":
            return selector_value < rhs
        if op == ">":
            return selector_value > rhs
        return None

    saw_any = False
    for disj in re.split(r"\bor\b", text, flags=re.IGNORECASE):
        atoms = [a for a in re.split(r"\band\b", disj, flags=re.IGNORECASE) if a.strip()]
        if not atoms:
            continue
        atom_values: List[bool] = []
        for atom in atoms:
            value = eval_atom(atom)
            if value is None:
                return None
            atom_values.append(value)
        saw_any = True
        if all(atom_values):
            return True
    return False if saw_any else None


def _mutation_validation_for_constraint(
    *,
    cid: str,
    catalog_entry: Mapping[str, Any],
    sample: PositiveSample,
    mutated_raw: bytes,
) -> Tuple[bool, str, Dict[str, Any]]:
    kind = str(catalog_entry.get("kind") or "")
    if kind == "node_constraint":
        field = str(catalog_entry.get("node_name") or "")
        raw_constraint = str(catalog_entry.get("constraint") or "")
        constraint_lower = raw_constraint.lower().strip()
        boundary = sample.field_boundaries.get(field)
        if boundary is None:
            return False, "missing_field_boundary", {"field": field}
        endianness = str(catalog_entry.get("endianness") or "big")
        before = _field_value_from_bytes(sample.raw_bytes, boundary, endianness)
        after = _field_value_from_bytes(mutated_raw, boundary, endianness)
        details: Dict[str, Any] = {
            "constraint_kind": "node",
            "field": field,
            "constraint": raw_constraint,
            "field_boundary": boundary.to_dict(),
            "endianness": endianness,
            "before_value": before,
            "after_value": after,
            "bytes_changed": sample.raw_bytes != mutated_raw,
        }
        if before is None or after is None:
            return False, "unreadable_field_value", details
        if constraint_lower.startswith("enum"):
            values = _parse_enum_constraint(raw_constraint)
            if not values:
                return False, "unparseable_enum", details
            details["allowed_values"] = [int(v) for v in values]
            if _enum_covers_domain(values, bit_length=int(boundary.bit_length)):
                return False, "tautological_enum_domain", details
            if int(after) in {int(v) for v in values}:
                return False, "mutation_did_not_violate_enum", details
            return True, "ok", details
        if constraint_lower.startswith("range"):
            bounds = _parse_range_constraint(raw_constraint)
            if bounds is None:
                return False, "unparseable_range", details
            lo, hi = bounds
            details["range"] = {"lo": int(lo), "hi": int(hi)}
            if _range_covers_domain(bounds, bit_length=int(boundary.bit_length)):
                return False, "tautological_range_domain", details
            if int(lo) <= int(after) <= int(hi):
                return False, "mutation_did_not_violate_range", details
            return True, "ok", details
        if constraint_lower == "checksum":
            if before == after:
                return False, "mutation_did_not_change_checksum", details
            return True, "ok", details
        constraint_ir = catalog_entry.get("constraint_ir")
        if constraint_ir is None and constraint_lower.startswith("constraint_ir"):
            constraint_ir = _parse_constraint_ir(raw_constraint)
        if isinstance(constraint_ir, Mapping):
            ir_kind = str(constraint_ir.get("kind") or "").strip().lower()
            details["constraint_ir"] = dict(constraint_ir)
            if ir_kind == "conditional":
                for related in catalog_entry.get("related_field_names") or []:
                    related_boundary = sample.field_boundaries.get(str(related))
                    if related_boundary is None:
                        continue
                    related_before = _field_value_from_bytes(sample.raw_bytes, related_boundary, endianness)
                    related_after = _field_value_from_bytes(mutated_raw, related_boundary, endianness)
                    before_active = _eval_selector_formula(constraint_ir.get("expr"), int(related_before)) if related_before is not None else None
                    after_active = _eval_selector_formula(constraint_ir.get("expr"), int(related_after)) if related_after is not None else None
                    details["related_selector"] = {
                        "field": str(related),
                        "boundary": related_boundary.to_dict(),
                        "before_value": related_before,
                        "after_value": related_after,
                        "before_formula_active": before_active,
                        "after_formula_active": after_active,
                    }
                    if before_active is True and after_active is False:
                        return True, "ok", details
                if before != after:
                    return True, "ok", details
                return False, "constraint_ir_conditional_not_violated", details
            if ir_kind == "conditional_zero":
                if before == 0 and after != 0:
                    return True, "ok", details
                if before == after:
                    return False, "constraint_ir_zero_field_unchanged", details
                return True, "ok", details
            if ir_kind == "length_relation":
                if before == after:
                    return False, "constraint_ir_length_field_unchanged", details
                return True, "ok", details
            if before == after:
                return False, "constraint_ir_field_unchanged", details
            return True, "ok", details
        if before == after:
            return False, "mutation_did_not_change_field", details
        return True, "ok", details

    if kind == "edge_constraint" and str(catalog_entry.get("rel") or "").lower() == "condition_on":
        selector = str(catalog_entry.get("src_name") or "")
        target = str(catalog_entry.get("dst_name") or "")
        boundary = sample.field_boundaries.get(selector)
        if boundary is None:
            return False, "missing_selector_boundary", {"selector": selector, "target": target}
        target_boundary = sample.field_boundaries.get(target)
        endianness = str(catalog_entry.get("endianness") or "big")
        before = _field_value_from_bytes(sample.raw_bytes, boundary, endianness)
        after = _field_value_from_bytes(mutated_raw, boundary, endianness)
        before_active = _eval_selector_formula(catalog_entry.get("formula"), int(before)) if before is not None else None
        after_active = _eval_selector_formula(catalog_entry.get("formula"), int(after)) if after is not None else None
        details = {
            "constraint_kind": "condition_on",
            "selector": selector,
            "target": target,
            "formula": catalog_entry.get("formula"),
            "selector_boundary": boundary.to_dict(),
            "target_boundary": target_boundary.to_dict() if target_boundary is not None else None,
            "endianness": endianness,
            "before_value": before,
            "after_value": after,
            "before_formula_active": before_active,
            "after_formula_active": after_active,
            "bytes_changed": sample.raw_bytes != mutated_raw,
        }
        if before_active is False:
            return False, "inactive_condition_seed", details
        if sample.raw_bytes == mutated_raw:
            return False, "mutation_did_not_change_bytes", details
        if after_active is False:
            return True, "selector_deselected_target", details
        if target_boundary is None:
            return False, "missing_condition_target_boundary", details
        if not _changed_inside_boundary(sample.raw_bytes, mutated_raw, target_boundary):
            return False, "condition_target_not_mutated", details
        if after_active is None:
            return False, "unverified_condition_formula", details
        return True, "ok", details

    if sample.raw_bytes == mutated_raw:
        return False, "mutation_did_not_change_bytes", {"constraint_kind": kind}
    return True, "ok", {"constraint_kind": kind, "bytes_changed": True}


def _precheck_constraint_for_mutation(
    *,
    catalog_entry: Mapping[str, Any],
    sample: PositiveSample,
) -> Tuple[bool, str, Dict[str, Any]]:
    kind = str(catalog_entry.get("kind") or "")
    if kind == "node_constraint":
        raw_constraint = str(catalog_entry.get("constraint") or "")
        lower = raw_constraint.lower().strip()
        if not (lower.startswith("enum") or lower.startswith("range")):
            return True, "ok", {}
        field = str(catalog_entry.get("node_name") or "")
        boundary = sample.field_boundaries.get(field)
        if boundary is None:
            return False, "missing_field_boundary", {"field": field}
        details = {"field": field, "field_boundary": boundary.to_dict(), "constraint": raw_constraint}
        if lower.startswith("enum"):
            values = _parse_enum_constraint(raw_constraint)
            if values and _enum_covers_domain(values, bit_length=int(boundary.bit_length)):
                details["allowed_values"] = [int(v) for v in values]
                return False, "tautological_enum_domain", details
        if lower.startswith("range"):
            bounds = _parse_range_constraint(raw_constraint)
            if bounds and _range_covers_domain(bounds, bit_length=int(boundary.bit_length)):
                lo, hi = bounds
                details["range"] = {"lo": int(lo), "hi": int(hi)}
                return False, "tautological_range_domain", details
        return True, "ok", details

    if kind == "edge_constraint" and str(catalog_entry.get("rel") or "").lower() == "condition_on":
        selector = str(catalog_entry.get("src_name") or "")
        boundary = sample.field_boundaries.get(selector)
        if boundary is None:
            return False, "missing_selector_boundary", {"selector": selector, "target": catalog_entry.get("dst_name")}
        target = str(catalog_entry.get("dst_name") or "")
        before = _field_value_from_bytes(sample.raw_bytes, boundary, str(catalog_entry.get("endianness") or "big"))
        active = _eval_selector_formula(catalog_entry.get("formula"), int(before)) if before is not None else None
        details = {
            "selector": selector,
            "target": catalog_entry.get("dst_name"),
            "formula": catalog_entry.get("formula"),
            "target_boundary_present": bool(target in sample.field_boundaries),
            "before_value": before,
            "before_formula_active": active,
        }
        if active is False:
            return False, "inactive_condition_seed", details
        return True, "ok", details

    return True, "ok", {}


def _fallback_mutation_for_constraint(
    *,
    cid: str,
    catalog_entry: Mapping[str, Any],
    sample: PositiveSample,
) -> Optional[NegativeSample]:
    kind = str(catalog_entry.get("kind") or "")
    endianness = str(catalog_entry.get("endianness") or "big")
    mutated: Optional[bytes] = None
    note = f"fallback mutation for {cid}"

    if kind == "edge_constraint" and str(catalog_entry.get("rel") or "").lower() == "condition_on":
        selector = str(catalog_entry.get("src_name") or "")
        boundary = sample.field_boundaries.get(selector)
        if boundary is None:
            return None
        before = _field_value_from_bytes(sample.raw_bytes, boundary, endianness)
        if before is None:
            return None
        candidate = _selector_value_to_make_formula_false(catalog_entry.get("formula"), int(before), int(boundary.bit_length))
        if candidate is None:
            return None
        mutated = _patch_field_value(sample.raw_bytes, boundary, candidate, endianness)
        note = f"selector deselection fallback for {cid}: {selector} {before}->{candidate}"

    elif kind == "node_constraint":
        field = str(catalog_entry.get("node_name") or "")
        raw_constraint = str(catalog_entry.get("constraint") or "")
        constraint_ir = catalog_entry.get("constraint_ir")
        if constraint_ir is None and raw_constraint.lower().strip().startswith("constraint_ir"):
            constraint_ir = _parse_constraint_ir(raw_constraint)
        if isinstance(constraint_ir, Mapping):
            ir_kind = str(constraint_ir.get("kind") or "").strip().lower()
            if ir_kind in {"length_relation", "conditional_zero"}:
                mutated = _mutate_field_by_name(sample=sample, field_name=field, endianness=endianness)
                note = f"{ir_kind} field fallback for {cid}: {field}"
            elif ir_kind == "conditional":
                for related in catalog_entry.get("related_field_names") or []:
                    related_boundary = sample.field_boundaries.get(str(related))
                    if related_boundary is None:
                        continue
                    before = _field_value_from_bytes(sample.raw_bytes, related_boundary, endianness)
                    if before is None:
                        continue
                    candidate = _selector_value_to_make_formula_false(constraint_ir.get("expr"), int(before), int(related_boundary.bit_length))
                    if candidate is None:
                        continue
                    mutated = _patch_field_value(sample.raw_bytes, related_boundary, candidate, endianness)
                    note = f"conditional selector fallback for {cid}: {related} {before}->{candidate}"
                    if mutated and mutated != sample.raw_bytes:
                        break
                if mutated is None:
                    mutated = _mutate_field_by_name(sample=sample, field_name=field, endianness=endianness)
                    note = f"conditional field fallback for {cid}: {field}"

    if mutated is None or mutated == sample.raw_bytes:
        return None
    return NegativeSample(
        base_packet_id=sample.packet_id,
        raw_bytes=mutated,
        expected_violated_constraints=[cid],
        note=note,
    )


def _sanitize_case_fragment(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip())
    return text.strip("_")[:48] or "constraint"


def _case_name_for_constraint(constraint_id: str, *, frame_no: int, seq: int) -> str:
    if constraint_id.startswith("node:"):
        parts = constraint_id.split(":", 2)
        field = parts[1] if len(parts) > 1 else "node"
        raw = parts[2] if len(parts) > 2 else "constraint"
        kind = raw.split(":", 1)[0] if ":" in raw else raw
        return f"auto_F{frame_no}_{_sanitize_case_fragment(field)}_{_sanitize_case_fragment(kind)}_{seq:02d}"
    if constraint_id.startswith("edge:"):
        parts = constraint_id.split(":", 3)
        rel = parts[1] if len(parts) > 1 else "edge"
        tail = parts[2] if len(parts) > 2 else "constraint"
        return f"auto_F{frame_no}_{_sanitize_case_fragment(rel)}_{_sanitize_case_fragment(tail)}_{seq:02d}"
    return f"auto_F{frame_no}_{_sanitize_case_fragment(constraint_id)}_{seq:02d}"


def _iter_pcap_frames(path: Path) -> Iterable[Tuple[int, int, bytes]]:
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"Invalid PCAP header: {path}")
        magic = header[:4]
        if magic == _PCAP_MAGIC_PCAPNG:
            handle.seek(0)
            yield from _iter_pcapng_frames(handle)
            return
        if magic in _PCAP_MAGIC_LE:
            endian = "<"
        elif magic in _PCAP_MAGIC_BE:
            endian = ">"
        else:
            raise ValueError(f"Unrecognized PCAP magic {magic!r}: {path}")
        linktype = int(struct.unpack(endian + "I", header[20:24])[0])
        frame_no = 0
        while True:
            rec = handle.read(16)
            if not rec:
                break
            if len(rec) < 16:
                break
            _ts_sec, _ts_sub, incl_len, _orig_len = struct.unpack(endian + "IIII", rec)
            if incl_len < 0:
                break
            frame = handle.read(int(incl_len))
            if len(frame) < int(incl_len):
                break
            frame_no += 1
            yield frame_no, linktype, frame


def _iter_pcapng_frames(handle: Any) -> Iterable[Tuple[int, int, bytes]]:
    interfaces: List[int] = []
    endian = "<"
    frame_no = 0

    def _read_exact(n: int) -> bytes:
        data = handle.read(int(n))
        return data if data is not None else b""

    hdr = _read_exact(12)
    if len(hdr) < 12:
        raise ValueError("Invalid PCAPNG header")
    block_type = int(struct.unpack("<I", hdr[:4])[0])
    if block_type != _PCAPNG_SHB:
        raise ValueError("Invalid PCAPNG: first block is not SHB")
    bom = hdr[8:12]
    if bom == _PCAPNG_BOM_BE:
        endian = ">"
    elif bom == _PCAPNG_BOM_LE:
        endian = "<"
    total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
    if total_len < 12:
        raise ValueError("Invalid PCAPNG SHB length")
    _ = _read_exact(total_len - 12)

    while True:
        block_hdr = _read_exact(8)
        if not block_hdr:
            break
        if len(block_hdr) < 8:
            break
        btype = int(struct.unpack(endian + "I", block_hdr[:4])[0])
        blen = int(struct.unpack(endian + "I", block_hdr[4:8])[0])
        if blen < 12:
            break
        body = _read_exact(blen - 12)
        trailer = _read_exact(4)
        if len(body) < blen - 12 or len(trailer) < 4:
            break

        if btype == _PCAPNG_IDB:
            if len(body) >= 8:
                linktype = int(struct.unpack(endian + "H", body[0:2])[0])
                interfaces.append(linktype)
            continue

        if btype == _PCAPNG_EPB:
            if len(body) < 20:
                continue
            iface_id = int(struct.unpack(endian + "I", body[0:4])[0])
            cap_len = int(struct.unpack(endian + "I", body[12:16])[0])
            pkt_off = 20
            pkt_end = pkt_off + max(0, cap_len)
            if iface_id < 0 or iface_id >= len(interfaces) or pkt_end > len(body):
                continue
            frame_no += 1
            yield frame_no, interfaces[iface_id], body[pkt_off:pkt_end]
            continue

        if btype == _PCAPNG_SPB:
            if len(body) < 4 or not interfaces:
                continue
            pkt_len = int(struct.unpack(endian + "I", body[0:4])[0])
            cap_len = min(max(0, pkt_len), max(0, len(body) - 4))
            frame_no += 1
            yield frame_no, interfaces[0], body[4 : 4 + cap_len]


def _extract_frames_from_pcap(path: Path, frame_nos: Sequence[int]) -> Dict[int, Tuple[int, bytes]]:
    wanted = {int(x) for x in frame_nos if int(x) > 0}
    out: Dict[int, Tuple[int, bytes]] = {}
    if not wanted:
        return out
    for frame_no, linktype, frame in _iter_pcap_frames(path):
        if frame_no not in wanted:
            continue
        out[frame_no] = (linktype, frame)
        if len(out) >= len(wanted):
            break
    return out


def _write_single_frame_pcap(path: Path, *, linktype: int, frame: bytes) -> None:
    global_hdr = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, int(linktype))
    pkt_hdr = struct.pack("<IIII", 0, 0, len(frame), len(frame))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(global_hdr + pkt_hdr + frame)


def _extract_l3_packet_with_offset(frame: bytes, linktype: int) -> Optional[Tuple[int, bytes]]:
    if linktype == _DLT_IPV4:
        if not frame or ((frame[0] >> 4) & 0x0F) != 4:
            return None
        return 0, frame
    if linktype == _DLT_IPV6:
        if not frame or ((frame[0] >> 4) & 0x0F) != 6:
            return None
        return 0, frame
    if linktype == _DLT_RAW:
        return 0, frame
    if linktype == _DLT_EN10MB:
        if len(frame) < 14:
            return None
        ethertype = int(struct.unpack("!H", frame[12:14])[0])
        off = 14
        while ethertype in _ETHERTYPE_VLAN:
            if len(frame) < off + 4:
                return None
            ethertype = int(struct.unpack("!H", frame[off + 2 : off + 4])[0])
            off += 4
        if ethertype not in {_ETHERTYPE_IPV4, _ETHERTYPE_IPV6}:
            return None
        return off, frame[off:]
    if linktype == _DLT_LINUX_SLL:
        if len(frame) < 16:
            return None
        proto = int(struct.unpack("!H", frame[14:16])[0])
        if proto not in {_ETHERTYPE_IPV4, _ETHERTYPE_IPV6}:
            return None
        return 16, frame[16:]
    if linktype == _DLT_LINUX_SLL2:
        if len(frame) < 20:
            return None
        proto = int(struct.unpack("!H", frame[0:2])[0])
        if proto not in {_ETHERTYPE_IPV4, _ETHERTYPE_IPV6}:
            return None
        return 20, frame[20:]
    return None


def _internet_checksum(data: bytes) -> int:
    buf = data
    if len(buf) % 2:
        buf += b"\x00"
    total = 0
    for i in range(0, len(buf), 2):
        total += (buf[i] << 8) | buf[i + 1]
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _rewrite_ipv4_header_checksum(ip_pkt: bytearray, ihl: int) -> None:
    if len(ip_pkt) < ihl or ihl < 20:
        return
    ip_pkt[10:12] = b"\x00\x00"
    csum = _internet_checksum(bytes(ip_pkt[:ihl]))
    ip_pkt[10:12] = struct.pack("!H", csum)


def _rewrite_transport_checksum_ipv4(ip_pkt: bytearray, ihl: int, proto: int, l4_len: int) -> None:
    if proto not in {_IPPROTO_TCP, _IPPROTO_UDP}:
        return
    l4_off = ihl
    if len(ip_pkt) < l4_off + l4_len:
        return
    seg = bytearray(ip_pkt[l4_off : l4_off + l4_len])
    if proto == _IPPROTO_TCP:
        if len(seg) < 20:
            return
        seg[16:18] = b"\x00\x00"
    else:
        if len(seg) < 8:
            return
        seg[6:8] = b"\x00\x00"
    pseudo = bytearray()
    pseudo.extend(ip_pkt[12:20])
    pseudo.extend(struct.pack("!BBH", 0, int(proto), int(l4_len)))
    csum = _internet_checksum(bytes(pseudo) + bytes(seg))
    if proto == _IPPROTO_TCP:
        seg[16:18] = struct.pack("!H", csum)
    else:
        if csum == 0:
            csum = 0xFFFF
        seg[6:8] = struct.pack("!H", csum)
    ip_pkt[l4_off : l4_off + l4_len] = seg


def _mutate_ipv4_frame(
    *,
    frame: bytes,
    l3_off: int,
    ip_pkt: bytes,
    pcap_layer: str,
    new_layer_bytes: bytes,
) -> bytes:
    if len(ip_pkt) < 20:
        raise ValueError("IPv4 packet too short")
    ihl = int(ip_pkt[0] & 0x0F) * 4
    if ihl < 20 or len(ip_pkt) < ihl:
        raise ValueError("Invalid IPv4 IHL")
    total_len = int(struct.unpack("!H", ip_pkt[2:4])[0])
    if total_len <= 0 or total_len > len(ip_pkt):
        total_len = len(ip_pkt)
    frag = int(struct.unpack("!H", ip_pkt[6:8])[0])
    if (frag & 0x1FFF) != 0 or (frag & 0x2000) != 0:
        raise ValueError("Fragmented IPv4 packet is unsupported")
    proto = int(ip_pkt[9])
    prefix = frame[:l3_off]
    suffix = frame[l3_off + total_len :]
    ip_view = ip_pkt[:total_len]

    if pcap_layer == "l3_packet":
        return prefix + bytes(new_layer_bytes) + suffix

    if pcap_layer == "ip_payload":
        rebuilt = bytearray(ip_view[:ihl] + bytes(new_layer_bytes))
        rebuilt[2:4] = struct.pack("!H", len(rebuilt))
        _rewrite_ipv4_header_checksum(rebuilt, ihl)
        return prefix + bytes(rebuilt) + suffix

    l4 = ip_view[ihl:total_len]
    if proto == _IPPROTO_TCP:
        if len(l4) < 20:
            raise ValueError("TCP segment too short")
        data_offset = (l4[12] >> 4) & 0x0F
        l4_hdr_len = int(data_offset) * 4
        if l4_hdr_len < 20 or l4_hdr_len > len(l4):
            raise ValueError("Invalid TCP header length")
    elif proto == _IPPROTO_UDP:
        l4_hdr_len = 8
        if len(l4) < l4_hdr_len:
            raise ValueError("UDP segment too short")
    else:
        raise ValueError(f"Unsupported IPv4 transport proto={proto}")

    if pcap_layer == "transport_payload":
        rebuilt_l4 = bytearray(l4[:l4_hdr_len] + bytes(new_layer_bytes))
        if proto == _IPPROTO_UDP and len(rebuilt_l4) >= 8:
            rebuilt_l4[4:6] = struct.pack("!H", len(rebuilt_l4))
        rebuilt_ip = bytearray(ip_view[:ihl] + bytes(rebuilt_l4))
        rebuilt_ip[2:4] = struct.pack("!H", len(rebuilt_ip))
        _rewrite_ipv4_header_checksum(rebuilt_ip, ihl)
        _rewrite_transport_checksum_ipv4(rebuilt_ip, ihl, proto, len(rebuilt_l4))
        return prefix + bytes(rebuilt_ip) + suffix

    if pcap_layer == "tcp_header":
        if proto != _IPPROTO_TCP:
            raise ValueError(f"tcp_header mode requires TCP, got proto={proto}")
        rebuilt_l4 = bytearray(bytes(new_layer_bytes) + l4[l4_hdr_len:])
        rebuilt_ip = bytearray(ip_view[:ihl] + bytes(rebuilt_l4))
        rebuilt_ip[2:4] = struct.pack("!H", len(rebuilt_ip))
        _rewrite_ipv4_header_checksum(rebuilt_ip, ihl)
        return prefix + bytes(rebuilt_ip) + suffix

    raise ValueError(f"Unsupported pcap_layer={pcap_layer!r}")


def _frame_protocol_bytes(*, frame: bytes, linktype: int, pcap_layer: str, ip_proto: Optional[int]) -> Optional[bytes]:
    """The protocol-layer bytes of a capture frame, located as _mutate_frame_protocol_bytes replaces them."""
    try:
        l3 = _extract_l3_packet_with_offset(frame, linktype)
        if l3 is None:
            return None
        _l3_off, ip_pkt = l3
        if len(ip_pkt) < 20 or (ip_pkt[0] >> 4) & 0x0F != 4:
            return None
        ihl = int(ip_pkt[0] & 0x0F) * 4
        total_len = int(struct.unpack("!H", ip_pkt[2:4])[0])
        if total_len <= 0 or total_len > len(ip_pkt):
            total_len = len(ip_pkt)
        frag = int(struct.unpack("!H", ip_pkt[6:8])[0])
        if (frag & 0x1FFF) != 0 or (frag & 0x2000) != 0:
            return None
        ip_view = ip_pkt[:total_len]
        if pcap_layer == "l3_packet":
            return bytes(ip_view)
        if pcap_layer == "ip_payload":
            if ip_proto is not None and int(ip_view[9]) != int(ip_proto):
                return None
            return bytes(ip_view[ihl:])
        proto = int(ip_view[9])
        l4 = ip_view[ihl:]
        if proto == _IPPROTO_TCP:
            if len(l4) < 20:
                return None
            l4_hdr_len = ((l4[12] >> 4) & 0x0F) * 4
        elif proto == _IPPROTO_UDP:
            l4_hdr_len = 8
        else:
            return None
        if pcap_layer == "transport_payload":
            return bytes(l4[l4_hdr_len:])
        if pcap_layer == "tcp_header" and proto == _IPPROTO_TCP:
            return bytes(l4[:l4_hdr_len])
    except Exception:
        return None
    return None


def _locate_anchor_frames(
    positives: Sequence[TrafficPositiveExample],
    *,
    base_pcap: Path,
    pcap_layer: str,
    ip_proto: Optional[int],
) -> Tuple[List[TrafficPositiveExample], Dict[str, Optional[int]]]:
    """Each traffic anchor's own frame in the capture, found by its protocol-layer bytes.

    A trace's packet_index is the anchor's index among the k Stage-B anchors, not its capture
    frame; using index + 1 as the frame put mutated anchors into the envelopes of frames 1..k and
    checked positives on frames 1..k, which include the evaluation suite's seed frame. Anchors
    whose bytes match no single unfragmented frame are dropped.
    """
    wanted = {bytes(entry.sample.raw_bytes) for entry in positives}
    first_frame: Dict[bytes, int] = {}
    for frame_no, linktype, frame in _iter_pcap_frames(base_pcap):
        layer = _frame_protocol_bytes(frame=frame, linktype=linktype, pcap_layer=pcap_layer, ip_proto=ip_proto)
        if layer is not None and layer in wanted and layer not in first_frame:
            first_frame[layer] = int(frame_no)
    located: List[TrafficPositiveExample] = []
    frames: Dict[str, Optional[int]] = {}
    for entry in positives:
        frame_no = first_frame.get(bytes(entry.sample.raw_bytes))
        frames[str(entry.sample.packet_id)] = frame_no
        if frame_no is not None:
            located.append(TrafficPositiveExample(sample=entry.sample, frame_no=frame_no))
    return located, frames


def _mutate_frame_protocol_bytes(
    *,
    frame: bytes,
    linktype: int,
    pcap_layer: str,
    ip_proto: Optional[int],
    new_layer_bytes: bytes,
) -> bytes:
    l3 = _extract_l3_packet_with_offset(frame, linktype)
    if l3 is None:
        raise ValueError(f"Unsupported linktype={linktype}")
    l3_off, ip_pkt = l3
    if not ip_pkt:
        raise ValueError("Empty L3 packet")
    version = (ip_pkt[0] >> 4) & 0x0F
    if version != 4:
        raise ValueError(f"Unsupported IP version={version}")
    if pcap_layer == "ip_payload" and ip_proto is not None and len(ip_pkt) >= 10:
        observed = int(ip_pkt[9])
        if observed != int(ip_proto):
            raise ValueError(f"IPv4 proto mismatch: expected {ip_proto}, got {observed}")
    return _mutate_ipv4_frame(
        frame=frame,
        l3_off=l3_off,
        ip_pkt=ip_pkt,
        pcap_layer=pcap_layer,
        new_layer_bytes=bytes(new_layer_bytes),
    )


def generate_exp_constraint_suite_from_traffic(
    *,
    protocol: str,
    tree_path: Path,
    traces_path: Path,
    base_pcap: Path,
    out_dir: Path,
    codegen_profile: Optional[Mapping[str, Any]] = None,
    max_negatives_per_constraint: int = 1,
) -> GeneratedConstraintSuite:
    protocol_norm = str(protocol or "").strip().lower()
    if not protocol_norm:
        raise ValueError("protocol is required")
    if max_negatives_per_constraint <= 0:
        raise ValueError("max_negatives_per_constraint must be > 0")

    format_tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if isinstance(format_tree, dict) and isinstance(format_tree.get("protocol_tree"), dict):
        format_tree = format_tree["protocol_tree"]
    if not isinstance(format_tree, dict):
        raise ValueError(f"Invalid format tree at {tree_path}")
    format_tree = normalize_protocol_tree(format_tree)

    positives_all = _load_success_positive_examples(format_tree={"protocol_tree": format_tree}, traces_path=traces_path)
    positives = [entry for entry in positives_all if entry.sample.raw_bytes and entry.sample.field_boundaries]
    sampler = ConstraintGuidedNegativeSampler({"protocol_tree": format_tree})
    constraints_catalog = _build_constraints_catalog({"protocol_tree": format_tree})
    pcap_layer = infer_pcap_layer(codegen_profile, host=None)
    ip_proto = infer_ip_proto(codegen_profile, host=None) if pcap_layer == "ip_payload" else None
    positives, anchor_frames = _locate_anchor_frames(
        positives, base_pcap=base_pcap, pcap_layer=pcap_layer, ip_proto=ip_proto
    )

    by_constraint: Dict[str, List[TrafficPositiveExample]] = {}
    for entry in positives:
        for cid in entry.sample.activated_constraints:
            by_constraint.setdefault(str(cid), []).append(entry)

    needed_frames = sorted({entry.frame_no for entry in positives})
    frames_by_no = _extract_frames_from_pcap(base_pcap, needed_frames)

    suite_dir = out_dir.resolve()
    samples_dir = suite_dir / "samples"
    suite_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    cases: Dict[str, Any] = {}
    case_idx = 0
    skipped_no_positive = 0
    skipped_no_mutation = 0
    skipped_no_frame = 0
    skipped_unsupported_frame = 0
    skipped_tautological_constraints: set[str] = set()
    skipped_inactive_condition_constraints: set[str] = set()
    skipped_invalid_mutation_constraints: set[str] = set()
    generated_per_constraint: Dict[str, int] = {}

    for cid in sorted(constraints_catalog.keys()):
        catalog_entry = constraints_catalog.get(cid) or {}
        positives_for_cid = by_constraint.get(cid) or []
        if not positives_for_cid:
            skipped_no_positive += 1
            continue
        for entry in positives_for_cid:
            if int(generated_per_constraint.get(cid, 0)) >= int(max_negatives_per_constraint):
                break
            frame_record = frames_by_no.get(int(entry.frame_no))
            if frame_record is None:
                skipped_no_frame += 1
                continue
            sample = entry.sample
            reduced = PositiveSample(
                packet_id=sample.packet_id,
                raw_bytes=sample.raw_bytes,
                path_id=sample.path_id,
                variant_id=sample.variant_id,
                field_values=dict(sample.field_values),
                field_boundaries=dict(sample.field_boundaries),
                activated_constraints=[cid],
            )
            pre_ok, pre_reason, _pre_details = _precheck_constraint_for_mutation(
                catalog_entry=catalog_entry,
                sample=reduced,
            )
            if not pre_ok:
                if pre_reason.startswith("tautological"):
                    skipped_tautological_constraints.add(cid)
                    break
                elif pre_reason == "inactive_condition_seed":
                    skipped_inactive_condition_constraints.add(cid)
                else:
                    skipped_no_mutation += 1
                continue
            negs = sampler.generate_for_positive(reduced, max_per_constraint=1)
            if not negs:
                fallback = _fallback_mutation_for_constraint(cid=cid, catalog_entry=catalog_entry, sample=reduced)
                if fallback is None:
                    skipped_no_mutation += 1
                    continue
                neg = fallback
            else:
                neg = negs[0]
            mutation_ok, mutation_reason, mutation_details = _mutation_validation_for_constraint(
                cid=cid,
                catalog_entry=catalog_entry,
                sample=reduced,
                mutated_raw=neg.raw_bytes,
            )
            if not mutation_ok:
                fallback = _fallback_mutation_for_constraint(cid=cid, catalog_entry=catalog_entry, sample=reduced)
                if fallback is not None:
                    fallback_ok, fallback_reason, fallback_details = _mutation_validation_for_constraint(
                        cid=cid,
                        catalog_entry=catalog_entry,
                        sample=reduced,
                        mutated_raw=fallback.raw_bytes,
                    )
                    if fallback_ok:
                        neg = fallback
                        mutation_ok = True
                        mutation_reason = fallback_reason
                        mutation_details = fallback_details
                if not mutation_ok:
                    if mutation_reason.startswith("tautological"):
                        skipped_tautological_constraints.add(cid)
                        break
                    elif mutation_reason == "inactive_condition_seed":
                        skipped_inactive_condition_constraints.add(cid)
                    else:
                        skipped_invalid_mutation_constraints.add(cid)
                    continue
            linktype, frame_bytes = frame_record
            try:
                mutated_frame = _mutate_frame_protocol_bytes(
                    frame=frame_bytes,
                    linktype=linktype,
                    pcap_layer=pcap_layer,
                    ip_proto=ip_proto,
                    new_layer_bytes=neg.raw_bytes,
                )
            except Exception:
                skipped_unsupported_frame += 1
                continue

            case_idx += 1
            seq = int(generated_per_constraint.get(cid, 0)) + 1
            case_name = _case_name_for_constraint(cid, frame_no=int(entry.frame_no), seq=seq)
            pcap_name = f"{case_name}.pcap"
            meta_name = f"{case_name}.json"
            pcap_path = samples_dir / pcap_name
            meta_path = samples_dir / meta_name
            _write_single_frame_pcap(pcap_path, linktype=linktype, frame=mutated_frame)
            meta_payload = {
                "protocol": protocol_norm,
                "constraint_id": cid,
                "seed_frame_no": int(entry.frame_no),
                "seed_packet_id": str(sample.packet_id),
                "packet_index": int(sample.packet_id) if str(sample.packet_id).isdigit() else None,
                "note": neg.note,
                "pcap_layer": pcap_layer,
                "ip_proto": ip_proto,
                "mutation": mutation_details,
            }
            meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            cases[case_name] = {
                "pcap": f"samples/{pcap_name}",
                "json": f"samples/{meta_name}",
                "target_constraints": [cid],
                "seed_frame_no": int(entry.frame_no),
                "mutation_note": neg.note,
                "mutation": mutation_details,
            }
            generated_per_constraint[cid] = seq

    mapping_path = suite_dir / f"{protocol_norm}_negpcaps_constraints_map_generated.json"
    mapping_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol_norm,
        "generator": "traffic_conformance_counterexample_v1",
        "base_pcap": str(base_pcap),
        "tree": str(tree_path),
        "tree_sha256": _sha256_file(tree_path),
        "traces": str(traces_path),
        "anchor_frames": anchor_frames,
        "pcap_layer": pcap_layer,
        "ip_proto": ip_proto,
        "constraints_from_tree": constraints_catalog,
        "cases": cases,
    }
    mapping_path.write_text(json.dumps(mapping_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = GeneratedConstraintSuite(
        protocol=protocol_norm,
        suite_dir=suite_dir,
        mapping_path=mapping_path,
        pcap_layer=pcap_layer,
        ip_proto=ip_proto,
        positive_samples_total=len(positives_all),
        positive_samples_usable=len(positives),
        constraints_total=len(constraints_catalog),
        constraints_seen_in_traffic=len(by_constraint),
        cases_generated=len(cases),
        skipped_no_positive=skipped_no_positive,
        skipped_no_mutation=skipped_no_mutation,
        skipped_no_frame=skipped_no_frame,
        skipped_unsupported_frame=skipped_unsupported_frame,
        skipped_tautological=len(skipped_tautological_constraints),
        skipped_inactive_condition=len(skipped_inactive_condition_constraints),
        skipped_invalid_mutation=len(skipped_invalid_mutation_constraints),
    )
    (suite_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
