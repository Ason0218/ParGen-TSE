from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FrameIndex:
    """Dataset frame index (1-based frame numbers).

    This is used to restrict semantic evaluation to a stable subset of frames.
    """

    include_frames: Tuple[int, ...]
    source_path: Path


def _coerce_positive_ints(values: Sequence[Any]) -> List[int]:
    out: List[int] = []
    for v in values:
        try:
            i = int(v)
        except Exception:
            continue
        if i <= 0:
            continue
        out.append(i)
    return out


def _normalize_frames(frames: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(set(int(x) for x in frames if int(x) > 0)))


def _extract_frames_from_dict(payload: Dict[str, Any], *, pcap_stem: Optional[str]) -> Optional[Sequence[Any]]:
    # Per-PCAP mapping format: { "<pcap_stem>": [1,2,3] }
    if pcap_stem:
        per = payload.get(pcap_stem)
        if isinstance(per, list):
            return per

    # If the index file declares a specific PCAP, only apply it when stems match.
    if pcap_stem:
        try:
            declared = payload.get("pcap")
            if isinstance(declared, str) and declared.strip():
                declared_stem = Path(declared.strip()).stem
                if declared_stem and declared_stem != pcap_stem:
                    return None
        except Exception:
            # If we cannot interpret the declared PCAP, fall back to the generic keys below.
            pass

    # Explicit list format in a dict.
    for key in ("include_frames", "frames", "frame_numbers", "frame_nos"):
        v = payload.get(key)
        if isinstance(v, list):
            return v
    return None


def load_frame_index(path: Path, *, pcap_stem: Optional[str] = None) -> Optional[FrameIndex]:
    """Load a dataset frame index from JSON.

    Supported formats:
    - List[int]: [1,2,3]
    - Dict: { "include_frames": [1,2,3], ... }
    - Dict mapping: { "<pcap_stem>": [1,2,3], ... }
    """

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    raw_frames: Optional[Sequence[Any]] = None
    if isinstance(payload, list):
        raw_frames = payload
    elif isinstance(payload, dict):
        raw_frames = _extract_frames_from_dict(payload, pcap_stem=pcap_stem)
    else:
        raw_frames = None

    if not raw_frames:
        return None

    frames = _coerce_positive_ints(raw_frames)
    include = _normalize_frames(frames)
    if not include:
        return None
    return FrameIndex(include_frames=include, source_path=path)
