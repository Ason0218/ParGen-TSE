"""Human semantic decisions over saved parser output; no model calls or parser runs."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "declared_source": payload.get("source") if isinstance(payload, dict) else None,
    }


def evaluate_human_semantics(
    *, gt_by_id: Mapping[int, Mapping[str, Any]], pred_by_id: Mapping[int, Mapping[str, Any]],
    frame_ids: Sequence[int], inputs: Mapping[str, Any], out_dir: Path,
    review_path: Optional[Path] = None, suggestions: Sequence[Mapping[str, Any]] = (),
) -> dict:
    """Align exact spans, then aggregate human name-and-decoded-value decisions.

    Every field occurrence counts. All same-span candidates are presented when spans
    repeat; accepted matches must be one-to-one. Unmatched fields contribute FP/FN.
    LLM suggestions never fill the human decision column.
    """
    from .tooljson_metrics import _field_bit_start_and_size, _iter_fields

    policy = "exact_span_human_name_and_decoded_value_v1"
    binding = {"policy": policy, "inputs": inputs, "evaluated_frame_ids": list(frame_ids)}
    fingerprint = _digest(binding)
    pairs = []
    packets = []
    total_gt = total_pred = 0
    suggestion_index = defaultdict(list)
    for suggestion in suggestions:
        for example in suggestion.get("examples") or []:
            key = (example.get("frame_index"), example.get("bit_start"), example.get("size_bits"),
                   suggestion.get("gt_name"), suggestion.get("pred_name"))
            suggestion_index[key].append(dict(suggestion))
    for frame in frame_ids:
        gt_pkt, pred_pkt = gt_by_id[frame], pred_by_id.get(frame, {})
        gt_fields = [dict(f) for f in _iter_fields(gt_pkt) if _field_bit_start_and_size(f)[1] > 0]
        pred_fields = [dict(f) for f in _iter_fields(pred_pkt) if _field_bit_start_and_size(f)[1] > 0]
        total_gt += len(gt_fields)
        total_pred += len(pred_fields)
        packets.append({"frame_id": frame, "gt_hex": gt_pkt.get("hex"), "pred_hex": pred_pkt.get("hex"),
                        "gt_fields": gt_fields, "pred_fields": pred_fields})
        by_span = defaultdict(list)
        for pi, pred in enumerate(pred_fields):
            by_span[_field_bit_start_and_size(pred)].append((pi, pred))
        for gi, gt in enumerate(gt_fields):
            span = _field_bit_start_and_size(gt)
            for pi, pred in by_span[span]:
                identity = {"frame_id": frame, "gt_index": gi, "pred_index": pi, "gt": gt, "pred": pred}
                key = (frame, *span, gt.get("name"), pred.get("name"))
                pairs.append({"id": _digest(identity), **identity,
                              "llm_suggestions": suggestion_index.get(key, []),
                              "equivalent": None, "notes": ""})

    template = {
        "schema_version": 1, "fingerprint": fingerprint, **binding, "reviewer": "",
        "instructions": (
            "Check the specification and packet context. Set equivalent to true only when both the field's "
            "meaning and its emitted decoded value are correct. Raw bytes alone do not establish decoded-value "
            "correctness. Review every candidate; accepted pairs must be one-to-one. LLM suggestions are aids. "
            "Record the reviewer and any adjudication notes."
        ),
        "total_gt_fields": total_gt, "total_pred_fields": total_pred, "packets": packets, "pairs": pairs,
    }
    # Read before writing: a reviewed template may itself be supplied as the input.
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path is not None else None
    decisions = {}
    reviewer = ""
    if review is not None:
        if review.get("schema_version") != 1 or review.get("fingerprint") != fingerprint:
            raise ValueError("Human semantic review does not match these inputs and evaluation frames")
        reviewer = str(review.get("reviewer") or "").strip()
        known = {pair["id"] for pair in pairs}
        for row in review.get("pairs", []):
            pair_id = row.get("id")
            if pair_id not in known or pair_id in decisions:
                raise ValueError("Human semantic review contains an unknown or duplicate pair id")
            decision = row.get("equivalent")
            if decision is not None and type(decision) is not bool:
                raise ValueError("Human equivalent decisions must be true, false or null")
            decisions[pair_id] = decision
        if any(value is not None for value in decisions.values()) and not reviewer:
            raise ValueError("Human semantic decisions require a reviewer identifier")

    matched_gt, matched_pred = set(), set()
    packet_tp = defaultdict(int)
    for pair in pairs:
        if decisions.get(pair["id"]) is not True:
            continue
        if pair["pred"].get("value") is None:
            raise ValueError("A semantic match requires an emitted decoded value; raw bytes are insufficient")
        gt_key = (pair["frame_id"], pair["gt_index"])
        pred_key = (pair["frame_id"], pair["pred_index"])
        if gt_key in matched_gt or pred_key in matched_pred:
            raise ValueError("Accepted human semantic matches must be one-to-one")
        matched_gt.add(gt_key)
        matched_pred.add(pred_key)
        packet_tp[pair["frame_id"]] += 1
    reviewed = sum(decisions.get(pair["id"]) is not None for pair in pairs)
    complete = reviewed == len(pairs) and bool(frame_ids) and total_gt > 0
    tp = len(matched_gt)
    fp, fn = total_pred - tp, total_gt - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    exact_packets = sum(packet_tp[p["frame_id"]] == len(p["gt_fields"]) == len(p["pred_fields"]) for p in packets)
    result = {
        "name": "level2_end_to_end_human", "method": "human_review", "metric_type": "end_to_end_f1",
        "status": "complete" if complete else ("no_reference_fields" if not total_gt else "pending_review"),
        "score": f1 if complete else None, "f1": f1 if complete else None,
        "precision": (tp / total_pred if total_pred else 0.0) if complete else None,
        "recall": (tp / total_gt if total_gt else 0.0) if complete else None,
        "packet_accuracy": (exact_packets / len(frame_ids)) if complete else None,
        "tp": tp if complete else None, "fp": fp if complete else None, "fn": fn if complete else None,
        "total_gt_fields": total_gt, "total_pred_fields": total_pred,
        "support": len(pairs), "reviewed_pairs": reviewed, "pending_pairs": len(pairs) - reviewed,
        "reviewer": reviewer or None, "fingerprint": fingerprint,
        "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest() if review_path else None,
        "policy": policy,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    template_path = out_dir / "human_semantic_review.template.json"
    if review_path is None or template_path.resolve() != review_path.resolve():
        template_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "human_semantic_review.result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Preserve the actual reviewed decisions next to the aggregate, including notes.
    if review is not None:
        (out_dir / "human_semantic_review.applied.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        (out_dir / "human_semantic_review.applied.json").write_text(
            json.dumps({"status": "not_applied", "fingerprint": fingerprint}, indent=2) + "\n", encoding="utf-8")
    return result
