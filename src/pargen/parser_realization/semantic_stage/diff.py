from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .gt_from_tree import _normalize_hex
from pargen.parser_realization.target_platforms.bit_utils import extract_bits_msb0


def _packet_id(pkt: Dict[str, Any], fallback: int) -> int:
    for key in ("packet_no", "frame_no"):
        if key in pkt:
            try:
                return int(pkt.get(key))  # type: ignore[arg-type]
            except Exception:
                continue
    return fallback


def _packet_bytes(pkt: Dict[str, Any]) -> Optional[bytes]:
    raw = _normalize_hex(pkt.get("hex"))
    if not raw:
        return None
    try:
        return bytes.fromhex(raw)
    except Exception:
        return None


def _l1_key(field: Dict[str, Any], *, pkt_bytes: Optional[bytes]) -> Tuple[int, int, int]:
    bit_start_raw = field.get("bit_start")
    size_bits_raw = field.get("size_bits")
    if bit_start_raw is None or size_bits_raw is None:
        try:
            offset = int(field.get("offset") or 0)
        except Exception:
            offset = 0
        try:
            length = int(field.get("length") or 0)
        except Exception:
            length = 0
        bit_start = int(offset) * 8
        size_bits = int(length) * 8
    else:
        try:
            bit_start = int(bit_start_raw)
        except Exception:
            bit_start = 0
        try:
            size_bits = int(size_bits_raw)
        except Exception:
            size_bits = 0

    value_int_raw = field.get("value_int")
    if value_int_raw is not None:
        try:
            value_int = int(value_int_raw)
        except Exception:
            value_int = 0
    else:
        value_int = 0
        if pkt_bytes is not None and bit_start >= 0 and size_bits > 0 and bit_start + size_bits <= len(pkt_bytes) * 8:
            try:
                value_int = int(extract_bits_msb0(pkt_bytes, bit_start, size_bits))
            except Exception:
                value_int = 0
        elif size_bits > 0 and (bit_start % 8 == 0) and (size_bits % 8 == 0):
            raw_hex = _normalize_hex(field.get("raw"))
            if raw_hex:
                try:
                    value_int = int(raw_hex, 16)
                except Exception:
                    value_int = 0

    return (int(bit_start), int(size_bits), int(value_int))


def _range_for_key(key: Tuple[int, int, int]) -> Tuple[int, int]:
    bit_start, size_bits, _value = key
    return (int(bit_start), int(bit_start) + int(size_bits))


def build_semantic_diff_report(
    *,
    pred_tooljson_path: Path,
    gt_level1_path: Path,
    out_path: Path,
    worst_k: int = 5,
    max_fields_per_packet: int = 40,
) -> Path:
    pred = json.loads(pred_tooljson_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_level1_path.read_text(encoding="utf-8"))
    pred_packets = pred if isinstance(pred, list) else []
    gt_packets = (gt.get("packets") if isinstance(gt, dict) else None) or []

    pred_by_id: Dict[int, Dict[str, Any]] = {}
    for idx, pkt in enumerate(pred_packets, start=1):
        if isinstance(pkt, dict):
            pred_by_id[_packet_id(pkt, idx)] = pkt

    gt_by_id: Dict[int, Dict[str, Any]] = {}
    for idx, pkt in enumerate(gt_packets, start=1):
        if isinstance(pkt, dict):
            gt_by_id[_packet_id(pkt, idx)] = pkt

    gt_frame_ids = set(gt_by_id.keys())
    pred_frame_ids = set(pred_by_id.keys())
    # Use the union so catastrophic exporter/parser failures still produce
    # actionable feedback instead of an empty diff. If the prediction has no
    # packets, every GT field should be reported as missing; if prediction has
    # packets outside the GT set, those fields should be reported as extra.
    frame_ids = sorted(gt_frame_ids | pred_frame_ids)

    missing_total = 0
    extra_total = 0
    wrong_raw_total = 0
    merged_total = 0

    worst: List[Dict[str, Any]] = []
    shift_diffs: List[int] = []

    for frame_id in frame_ids:
        gt_pkt = gt_by_id.get(frame_id, {})
        pred_pkt = pred_by_id.get(frame_id, {})

        gt_fields = gt_pkt.get("fields") or []
        pred_fields = pred_pkt.get("fields") or []
        if not isinstance(gt_fields, list):
            gt_fields = []
        if not isinstance(pred_fields, list):
            pred_fields = []

        gt_bytes = _packet_bytes(gt_pkt)
        pred_bytes = _packet_bytes(pred_pkt)

        gt_keys = {_l1_key(f, pkt_bytes=gt_bytes) for f in gt_fields if isinstance(f, dict)}
        pred_keys = {_l1_key(f, pkt_bytes=pred_bytes) for f in pred_fields if isinstance(f, dict)}

        missing = sorted(gt_keys - pred_keys)
        extra = sorted(pred_keys - gt_keys)

        # wrong_value: same bit_start/size_bits but different value_int.
        gt_by_bs: Dict[Tuple[int, int], int] = {}
        pred_by_bs: Dict[Tuple[int, int], int] = {}
        for k in gt_keys:
            gt_by_bs[(k[0], k[1])] = int(k[2])
        for k in pred_keys:
            pred_by_bs[(k[0], k[1])] = int(k[2])

        wrong_raw: List[Dict[str, Any]] = []
        for bs in set(gt_by_bs.keys()) & set(pred_by_bs.keys()):
            if int(gt_by_bs[bs]) != int(pred_by_bs[bs]):
                bit_start, size_bits = bs
                wrong_raw.append(
                    {
                        "bit_start": int(bit_start),
                        "size_bits": int(size_bits),
                        "start_byte": int(bit_start // 8),
                        "bit_offset_in_first_byte": int(bit_start % 8),
                        "gt_value_int": int(gt_by_bs[bs]),
                        "pred_value_int": int(pred_by_bs[bs]),
                    }
                )

        # merged: a predicted field fully covers a missing field range.
        merged: List[Dict[str, Any]] = []
        extra_ranges = [(_range_for_key(k), k) for k in extra]
        for miss in missing:
            miss_range = _range_for_key(miss)
            for (rng, k) in extra_ranges:
                if rng[0] <= miss_range[0] and rng[1] >= miss_range[1] and rng != miss_range:
                    merged.append(
                        {
                            "missing": {"bit_start": miss[0], "size_bits": miss[1], "value_int": miss[2]},
                            "covering_pred": {"bit_start": k[0], "size_bits": k[1], "value_int": k[2]},
                        }
                    )
                    break

        # Collect global shift hints by matching (size_bits, value_int) irrespective of bit_start.
        gt_by_sv: Dict[Tuple[int, int], List[int]] = {}
        pred_by_sv: Dict[Tuple[int, int], List[int]] = {}
        for bs, sz, val in gt_keys:
            gt_by_sv.setdefault((sz, int(val)), []).append(int(bs))
        for bs, sz, val in pred_keys:
            pred_by_sv.setdefault((sz, int(val)), []).append(int(bs))
        for sv in set(gt_by_sv.keys()) & set(pred_by_sv.keys()):
            best: Optional[int] = None
            for gb in gt_by_sv[sv]:
                for pb in pred_by_sv[sv]:
                    diff = int(pb) - int(gb)
                    if best is None or abs(diff) < abs(best):
                        best = diff
            if best is not None:
                shift_diffs.append(int(best))

        missing_total += len(missing)
        extra_total += len(extra)
        wrong_raw_total += len(wrong_raw)
        merged_total += len(merged)

        score = len(missing) + len(extra) + len(wrong_raw)
        if score <= 0:
            continue

        gt_meta: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for f in gt_fields:
            if not isinstance(f, dict):
                continue
            k = _l1_key(f, pkt_bytes=gt_bytes)
            gt_meta[k] = {
                "name": f.get("name"),
                "node_id": f.get("node_id"),
                "format_path": f.get("format_path"),
            }

        def _fmt_item(k: Tuple[int, int, int], *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            bit_start, size_bits, value_int = k
            out: Dict[str, Any] = {
                "bit_start": int(bit_start),
                "size_bits": int(size_bits),
                "value_int": int(value_int),
                "start_byte": int(bit_start // 8),
                "bit_offset_in_first_byte": int(bit_start % 8),
            }
            if meta:
                out.update({kk: vv for kk, vv in meta.items() if vv is not None})
            return out

        missing_items = [_fmt_item(k, meta=gt_meta.get(k)) for k in missing[:max_fields_per_packet]]
        extra_items = [_fmt_item(k) for k in extra[:max_fields_per_packet]]

        worst.append(
            {
                "frame_no": frame_id,
                "missing": missing_items,
                "extra": extra_items,
                "wrong_raw": wrong_raw[:max_fields_per_packet],
                "merged": merged[:max_fields_per_packet],
                "counts": {
                    "missing": len(missing),
                    "extra": len(extra),
                    "wrong_raw": len(wrong_raw),
                    "merged": len(merged),
                },
                "pred_fields_total": len(pred_fields),
                "gt_fields_total": len(gt_fields),
            }
        )

    worst.sort(
        key=lambda p: (
            -int(p.get("counts", {}).get("missing", 0)),
            -int(p.get("counts", {}).get("extra", 0)),
            -int(p.get("counts", {}).get("wrong_raw", 0)),
        )
    )
    worst = worst[: max(0, int(worst_k))]

    shift_hint: Dict[str, Any] = {}
    if shift_diffs:
        ctr = Counter(shift_diffs)
        mode, mode_count = ctr.most_common(1)[0]
        median = int(statistics.median(shift_diffs))
        shift_hint = {
            "samples": len(shift_diffs),
            "mode": int(mode),
            "mode_count": int(mode_count),
            "median": int(median),
        }

    summary = {
        "pred_tooljson": str(pred_tooljson_path),
        "gt_level1": str(gt_level1_path),
        "frames_evaluated": len(frame_ids),
        "frames_with_gt": len(gt_frame_ids),
        "frames_with_pred": len(pred_frame_ids),
        "frames_missing_pred": len(gt_frame_ids - pred_frame_ids),
        "frames_extra_pred": len(pred_frame_ids - gt_frame_ids),
        "missing_total": int(missing_total),
        "extra_total": int(extra_total),
        "wrong_raw_total": int(wrong_raw_total),
        "merged_total": int(merged_total),
        "global_shift_hint": shift_hint or None,
    }

    out_payload = {"summary": summary, "worst_packets": worst}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path
