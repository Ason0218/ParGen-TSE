"""Field observations of CheckParser (Sec. IV-D): PKG reference view versus parser view on the constructed suite.

For each PKG-constructed message, the PKG interpreter yields the reference field view F_G(x): the PKG node of every
field occurrence, its span, its emitted name and its decoded value (declared byte order and signedness).  The
generated parser's platform output is mapped back to PKG identities through output keys, which are determined by
the PKG (``output_keys``) and given to code generation.  Comparison reports missing and additional field
observations and span or value differences; each conflict carries the PKG identity used for localization.

Validity outcomes V_G/V_P are compared by the conformance suite through BS_CONSTRAINT rule identifiers; this module
covers the field part.  Platform records that no output key connects to a PKG object (for example aggregate items
that exporters add for display) are listed as unmapped diagnostics and are not counted as conflicts.
"""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_KEY_RE = re.compile(r"[^a-z0-9]+")
_LEAF_TYPES = {"field", "selector", "length", "type", "checksum", "count"}


def _norm_key(text: Any) -> str:
    return _KEY_RE.sub("_", str(text or "").strip().lower()).strip("_")


def output_keys(nodes: Iterable[Tuple[Any, Any]]) -> Dict[str, str]:
    """Output key of every PKG node: its normalized name, with the node id appended when names collide."""
    base: List[Tuple[str, str]] = []
    for nid, name in nodes:
        nid_s = str(nid)
        key = _norm_key(name) or f"node_{_norm_key(nid_s)}"
        if key[0].isdigit():
            key = f"f_{key}"
        base.append((nid_s, key))
    counts = Counter(key for _nid, key in base)
    return {nid: (key if counts[key] == 1 else f"{key}_n{_norm_key(nid)}") for nid, key in base}


def _payload(tree: Mapping[str, Any]) -> Mapping[str, Any]:
    inner = tree.get("protocol_tree") if isinstance(tree, Mapping) else None
    return inner if isinstance(inner, Mapping) else tree


def _nodes(tree: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(n.get("node_id")): n for n in (_payload(tree).get("nodes") or []) if isinstance(n, Mapping)}


def tree_output_keys(tree: Mapping[str, Any]) -> Dict[str, str]:
    return output_keys((nid, node.get("name") or f"node_{nid}") for nid, node in _nodes(tree).items())


def _is_leaf(node: Mapping[str, Any]) -> bool:
    return str(node.get("node_type") or "").lower() in _LEAF_TYPES or not (node.get("children_ids") or [])


@dataclass(frozen=True)
class FieldObservation:
    frame: int
    node_id: Optional[str]
    key: str
    name: str
    bit_start: int
    size_bits: int
    value: Optional[int]

    @property
    def end(self) -> int:
        return self.bit_start + self.size_bits

    def view(self) -> Dict[str, Any]:
        return {"bit_start": self.bit_start, "size_bits": self.size_bits, "value": self.value, "key": self.key}


@dataclass
class FieldCheckResult:
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    frames_checked: int = 0
    reference_fields: int = 0
    matched_fields: int = 0
    unmapped_observations: Dict[str, int] = field(default_factory=dict)
    reference_unparsed_frames: int = 0
    reference_missing_frames: List[int] = field(default_factory=list)
    unverified_values: int = 0

    @property
    def coverage_gaps(self) -> List[str]:
        gaps = []
        if self.frames_checked <= 0 or self.reference_fields <= 0:
            gaps.append("no_reference_fields")
        if self.reference_unparsed_frames:
            gaps.append("reference_parse_incomplete")
        if self.reference_missing_frames:
            gaps.append("reference_frames_missing")
        if self.unverified_values:
            gaps.append("reference_values_unavailable")
        return gaps

    @property
    def count(self) -> int:
        return len(self.conflicts)

    def kinds(self) -> Dict[str, int]:
        return dict(Counter(c["kind"] for c in self.conflicts))

    def to_dict(self, *, max_conflicts: int = 500) -> Dict[str, Any]:
        return {
            "conflicts_total": self.count,
            "conflict_kinds": self.kinds(),
            "frames_checked": self.frames_checked,
            "reference_fields": self.reference_fields,
            "matched_fields": self.matched_fields,
            "reference_unparsed_frames": self.reference_unparsed_frames,
            "reference_missing_frames": self.reference_missing_frames,
            "unverified_values": self.unverified_values,
            "coverage_gaps": self.coverage_gaps,
            "unmapped_observations": self.unmapped_observations,
            "conflicts": self.conflicts[:max_conflicts],
            "conflicts_truncated": max(0, self.count - max_conflicts),
        }


def reference_observations(tree: Mapping[str, Any], gt_packets: Sequence[Mapping[str, Any]]) -> Dict[int, List[FieldObservation]]:
    """F_G per frame from interpreter spans, with values decoded under the declared byte order and signedness."""
    from pargen.protocol_knowledge.message_validation.interpreter import decode_typed_value

    nodes, keys = _nodes(tree), tree_output_keys(tree)
    out: Dict[int, List[FieldObservation]] = {}
    for pkt in gt_packets:
        frame = int(pkt.get("frame_no") or 0)
        rows: List[FieldObservation] = []
        for f in pkt.get("fields") or []:
            nid = str(f.get("node_id"))
            node = nodes.get(nid)
            if node is None:
                continue
            size = int(f.get("size_bits") or 0)
            value: Optional[int] = None
            if 0 < size <= 64 and f.get("value_int") is not None:
                value = decode_typed_value(int(f["value_int"]), size, dict(node))
            rows.append(FieldObservation(frame=frame, node_id=nid, key=keys.get(nid, nid), name=str(node.get("name") or nid),
                                         bit_start=int(f.get("bit_start") or 0), size_bits=size, value=value))
        out[frame] = rows
    return out


def _record_span(rec: Mapping[str, Any]) -> Tuple[int, int]:
    if rec.get("bit_start") is not None and rec.get("size_bits") is not None:
        return int(rec["bit_start"]), int(rec["size_bits"])
    length = rec.get("length") if rec.get("length") is not None else rec.get("size")
    return int(rec.get("offset") or 0) * 8, int(length or 0) * 8


def _record_value(rec: Mapping[str, Any]) -> Optional[int]:
    from pargen.parser_realization.target_platforms.tooljson_metrics import _parse_numeric_value_loose

    raw = rec.get("value")
    value = _parse_numeric_value_loose(raw) if not isinstance(raw, (list, dict)) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def parser_observations(
    tool_packets: Sequence[Mapping[str, Any]],
    tree: Mapping[str, Any],
    *,
    filter_name: Optional[str] = None,
) -> Tuple[Dict[int, List[FieldObservation]], Counter]:
    """F_P per frame; output keys (or, after normalization, the emitted node name) connect records to PKG nodes."""
    nodes, keys = _nodes(tree), tree_output_keys(tree)
    by_key = {key: nid for nid, key in keys.items()}
    name_counts = Counter(_norm_key(n.get("name") or nid) for nid, n in nodes.items())
    by_name = {_norm_key(n.get("name") or nid): nid for nid, n in nodes.items()
               if name_counts[_norm_key(n.get("name") or nid)] == 1}
    proto_key = _norm_key(filter_name) if filter_name else ""
    out: Dict[int, List[FieldObservation]] = defaultdict(list)
    unmapped: Counter = Counter()
    for idx, pkt in enumerate(tool_packets, start=1):
        frame = idx
        for name in ("frame_no", "packet_no", "frame_index"):
            if pkt.get(name) is not None:
                try:
                    frame = int(pkt[name])
                    break
                except Exception:
                    continue
        for rec in pkt.get("fields") or []:
            if not isinstance(rec, Mapping):
                continue
            abbrev, name = str(rec.get("abbrev") or ""), str(rec.get("name") or "")
            if proto_key and _norm_key(abbrev) == proto_key:
                continue  # the protocol item itself
            candidates = [_norm_key(abbrev.split(".")[-1]) if abbrev else "", _norm_key(name)]
            nid = next((by_key[c] for c in candidates if c and c in by_key), None)
            if nid is None:
                nid = next((by_name[c] for c in candidates if c and c in by_name), None)
            bit_start, size = _record_span(rec)
            if size <= 0:
                continue
            key = candidates[0] or candidates[1]
            if nid is None:
                unmapped[key or "<unnamed>"] += 1
                continue
            out[frame].append(FieldObservation(frame=frame, node_id=nid, key=key, name=name, bit_start=bit_start,
                                               size_bits=size, value=_record_value(rec)))
    return dict(out), unmapped


def compare_field_observations(
    reference: Mapping[int, Sequence[FieldObservation]],
    observed: Mapping[int, Sequence[FieldObservation]],
    tree: Mapping[str, Any],
) -> FieldCheckResult:
    nodes = _nodes(tree)
    result = FieldCheckResult(frames_checked=len(reference))
    for frame in sorted(reference):
        refs = sorted(reference[frame], key=lambda r: (r.bit_start, str(r.node_id)))
        result.reference_fields += len(refs)
        obs_by_node: Dict[str, List[FieldObservation]] = defaultdict(list)
        for o in observed.get(frame, ()):
            if o.node_id is not None and o.node_id in nodes and _is_leaf(nodes[o.node_id]):
                obs_by_node[o.node_id].append(o)
        used: set[int] = set()
        for r in refs:
            cands = [o for o in obs_by_node.get(str(r.node_id), []) if id(o) not in used]
            match = next((o for o in cands if o.bit_start == r.bit_start), None)
            if match is None:
                overlapping = [o for o in cands if o.bit_start < r.end and r.bit_start < o.end]
                match = min(overlapping, key=lambda o: abs(o.bit_start - r.bit_start)) if overlapping else None
            base = {"frame": frame, "node_id": r.node_id, "output_key": r.key, "name": r.name, "expected": r.view()}
            if match is None:
                result.conflicts.append({"kind": "missing_field", **base})
                continue
            used.add(id(match))
            if (match.bit_start, match.size_bits) != (r.bit_start, r.size_bits):
                result.conflicts.append({"kind": "span_mismatch", **base, "observed": match.view()})
            elif r.value is None:
                # The interpreter does not currently provide decoded values for every
                # field type/width. A matching span alone is not a verified value.
                result.unverified_values += 1
            elif match.value is None:
                result.conflicts.append({"kind": "missing_or_invalid_value", **base, "observed": match.view()})
            elif match.value != r.value:
                result.conflicts.append({"kind": "value_mismatch", **base, "observed": match.view()})
            else:
                result.matched_fields += 1
        for nid, rows in obs_by_node.items():
            for o in rows:
                if id(o) not in used:
                    node = nodes.get(nid) or {}
                    result.conflicts.append({"kind": "additional_field", "frame": frame, "node_id": nid,
                                             "output_key": o.key, "name": str(node.get("name") or nid),
                                             "observed": o.view()})
    return result


def check_parser_fields(
    *,
    tree_path: Path,
    positives_pcap: Path,
    parser_tooljson: Path,
    pcap_layer: str,
    ip_proto: Optional[int],
    frames: Sequence[int],
    filter_name: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> FieldCheckResult:
    """Compare the parser's field view with the PKG reference view on the suite's positive messages."""
    from pargen.parser_realization.semantic_stage.gt_from_tree import build_gt_from_tree_and_tooljson

    tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="pkg_field_ref_") as tmp:
        gt_l1, _gt_l2, stats = build_gt_from_tree_and_tooljson(
            tree_path=Path(tree_path),
            pred_tooljson_path=Path(parser_tooljson),
            pcap_path=Path(positives_pcap),
            pcap_layer=pcap_layer,
            pcap_ip_proto=ip_proto,
            frame_index=list(frames),
            out_dir=Path(tmp),
            require_full_parse=True,
            emit_tail_data_field=False,
            max_packets=0,
            target_message_type=None,
        )
        gt_payload = json.loads(Path(gt_l1).read_text(encoding="utf-8"))
        gt_packets = gt_payload.get("packets", []) if isinstance(gt_payload, dict) else gt_payload
    tool_packets = json.loads(Path(parser_tooljson).read_text(encoding="utf-8"))
    if not isinstance(tool_packets, list):
        tool_packets = []
    reference = reference_observations(tree, gt_packets)
    observed, unmapped = parser_observations(tool_packets, tree, filter_name=filter_name)
    result = compare_field_observations(reference, observed, tree)
    result.unmapped_observations = dict(unmapped.most_common(50))
    result.reference_unparsed_frames = int(stats.packets_skipped_parse_fail + stats.packets_skipped_no_hex)
    result.reference_missing_frames = sorted(set(int(f) for f in frames) - set(reference))
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def describe_field_conflicts(result: FieldCheckResult, tree: Mapping[str, Any], *, max_items: int = 8) -> Tuple[str, ...]:
    """Localized conflicts for a repair request: grouped by kind and PKG node, with the node's PKG rules."""
    nodes = _nodes(tree)
    edges = [e for e in (_payload(tree).get("edges") or []) if isinstance(e, Mapping)]
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in result.conflicts:
        grouped[(c["kind"], str(c.get("node_id")))].append(c)
    lines: List[str] = []
    for (kind, nid), items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(lines) >= max_items:
            break
        node = nodes.get(nid) or {}
        first = items[0]
        related = [f"{e.get('rel')} {e.get('src')}->{e.get('dst')}" + (f" [{e.get('formula')}]" if e.get("formula") else "")
                   for e in edges if str(e.get("src")) == nid or str(e.get("dst")) == nid][:6]
        rule = (f"type={node.get('node_type')} data_type={node.get('data_type')} byte_order={node.get('byte_order')} "
                f"bit_start={node.get('bit_start')!r} size_bits={node.get('size_bits')!r} parent={node.get('parent_id')} "
                f"constraints={list(node.get('constraints') or [])[:4]} relations={related}")
        expected = first.get("expected")
        observed = first.get("observed")
        lines.append(
            f"field conflict {kind}: PKG node {nid} (output key `{first.get('output_key')}`, name {first.get('name')!r}) "
            f"in {len(items)} message(s), first at frame {first.get('frame')}: "
            + (f"PKG interpreter observes {expected}" if expected else "PKG interpreter has no occurrence here")
            + (f"; parser observes {observed}" if observed else "; parser emits no field with this identity")
            + f". PKG rules: {rule}"
        )
    return tuple(lines)
