from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from pargen.parser_realization.target_platforms.tooljson_metrics import compute_tooljson_metrics

from .diff import build_semantic_diff_report
from .gt_from_tree import build_gt_from_tree_and_tooljson
from .schema import SemanticValidationResult


def run_semantic_validation(
    *,
    tree_path: Path,
    pred_tooljson_path: Path,
    pcap_path: Path,
    pcap_layer: str = "transport_payload",
    pcap_ip_proto: Optional[int] = None,
    frame_index: Optional[Sequence[int]] = None,
    out_dir: Path,
    require_full_parse: bool = True,
    max_packets: int = 0,
    worst_k: int = 5,
    min_f1_l1: float = 0.99,
    max_skipped_packets: int = 0,
    max_skipped_ratio: Optional[float] = None,
) -> SemanticValidationResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_l1_path, gt_l2_path, stats = build_gt_from_tree_and_tooljson(
        tree_path=tree_path,
        pred_tooljson_path=pred_tooljson_path,
        pcap_path=pcap_path,
        pcap_layer=pcap_layer,
        pcap_ip_proto=pcap_ip_proto,
        frame_index=frame_index,
        out_dir=out_dir,
        require_full_parse=require_full_parse,
        emit_tail_data_field=True,
        max_packets=max_packets,
        target_message_type=None,
    )

    metrics = compute_tooljson_metrics(
        pred_tooljson=pred_tooljson_path,
        gt_level1=gt_l1_path,
        gt_level2=gt_l2_path,
        out_dir=out_dir,
        l2_bert_enable=False,
    )
    f1_l1 = float(metrics.get("field_f1_level1", 0.0) or 0.0)
    f1_l2 = float(metrics.get("field_f1_level2", 0.0) or 0.0)
    f1_l1_byte = float(metrics.get("field_f1_level1_byte", 0.0) or 0.0)

    diff_path = build_semantic_diff_report(
        pred_tooljson_path=pred_tooljson_path,
        gt_level1_path=gt_l1_path,
        out_path=out_dir / "semantic_diff.json",
        worst_k=worst_k,
    )

    skipped = int(stats.packets_skipped_no_hex + stats.packets_skipped_parse_fail)

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or not str(raw).strip():
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not str(raw).strip():
            return int(default)
        try:
            return int(float(raw))
        except Exception:
            return int(default)

    min_f1_l1 = _env_float("STAGE_C_SEMANTIC_MIN_F1_L1", float(min_f1_l1))
    max_skipped = _env_int("STAGE_C_SEMANTIC_MAX_SKIPPED", int(max_skipped_packets))

    ratio_default = max_skipped_ratio
    if ratio_default is None:
        ratio_default = 0.25 if str(pcap_layer or "").strip().lower() == "ip_payload" else 0.0
    max_skipped_ratio = _env_float("STAGE_C_SEMANTIC_MAX_SKIPPED_RATIO", float(ratio_default))
    max_skipped_ratio = max(0.0, min(1.0, float(max_skipped_ratio)))
    max_skipped_by_ratio = int(float(stats.packets_total) * float(max_skipped_ratio)) if stats.packets_total else 0

    allowed_skipped = max(int(max_skipped), int(max_skipped_by_ratio))
    ok = bool(f1_l1 >= float(min_f1_l1) and int(stats.packets_used) > 0 and skipped <= allowed_skipped)

    return SemanticValidationResult(
        ok=ok,
        f1_l1=f1_l1,
        f1_l2=f1_l2,
        f1_l1_bit=f1_l1,
        f1_l1_byte=f1_l1_byte,
        skipped_packets=skipped,
        gt_level1=gt_l1_path,
        gt_level2=gt_l2_path,
        metrics_json=out_dir / "metrics.json",
        diff_report=diff_path,
        stats=stats,
    )
