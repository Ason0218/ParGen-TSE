from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TreeGTBuildStats:
    packets_total: int = 0
    packets_used: int = 0
    packets_skipped_no_hex: int = 0
    packets_skipped_parse_fail: int = 0
    packets_truncated_to_pred_fields_end: int = 0
    fields_emitted: int = 0
    fields_skipped_non_byte_aligned: int = 0
    fields_emitted_non_byte_aligned: int = 0


@dataclass
class SemanticValidationResult:
    ok: bool
    f1_l1: float
    f1_l2: float
    f1_l1_bit: float = 0.0
    f1_l1_byte: float = 0.0
    skipped_packets: int = 0
    gt_level1: Optional[Path] = None
    gt_level2: Optional[Path] = None
    metrics_json: Optional[Path] = None
    diff_report: Optional[Path] = None
    stats: TreeGTBuildStats = field(default_factory=TreeGTBuildStats)


@dataclass
class SemanticRoundArtifacts:
    round_dir: Path
    tooljson_path: Path
    gt_level1: Path
    gt_level2: Path
    metrics_json: Path
    diff_report: Path
