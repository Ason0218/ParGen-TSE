from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pargen.parser_realization.code_repair_agent import CodeRepairAgent, RepairDiagnostics
from pargen.parser_realization.llm_client import LLMClient, LLMConfig, create_llm_client
from pargen.parser_realization.syntax_stage.backend_base import SyntaxBackend
from pargen.parser_realization.syntax_stage.repair_loop import SyntaxRepairLoopConfig, run_syntax_with_auto_repair
from pargen.parser_realization.target_platforms.extraction_profile import infer_ip_proto, infer_pcap_layer
from pargen.parser_realization.target_platforms.scapy_export import export_scapy_tooljson
from pargen.parser_realization.target_platforms.wireshark_export import export_wireshark_tooljson
from pargen.parser_realization.target_platforms.zeek_spicy_export import export_zeek_spicy_tooljson

from .schema import SemanticValidationResult, TreeGTBuildStats
from .validator import run_semantic_validation


@dataclass
class SemanticRepairLoopConfig:
    max_rounds: int = 3
    min_f1_l1: float = 0.99
    max_packets: int = 0
    require_full_parse: bool = True
    worst_k: int = 5
    enable_llm_repair: bool = True
    log_tail_chars: int = 12000
    mutants_per_packet: int = 0
    enable_negative_mutants: bool = False
    negative_mode: str = "N0"


def _semantic_selection_score(result: SemanticValidationResult) -> tuple[float, float, float]:
    """Score semantic repair candidates without letting L1 hide semantic regressions."""

    l1 = float(getattr(result, "f1_l1", 0.0) or 0.0)
    l2 = float(getattr(result, "f1_l2", 0.0) or 0.0)
    return (min(l1, l2), l1, l2)


def _scan_plugin_sources(plugin_src_dir: Path, source_globs: Sequence[str]) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for pattern in source_globs:
        for path in sorted(plugin_src_dir.glob(pattern)):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(plugin_src_dir)
                files[rel.as_posix()] = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return files


def _summarize_tree_nodes(tree_path: Path, node_ids: set[int], *, max_nodes: int = 25) -> str:
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(tree, dict):
        return ""

    nodes = tree.get("nodes") or []
    if not isinstance(nodes, list):
        return ""
    by_id: Dict[int, Dict[str, Any]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        try:
            nid = int(n.get("node_id"))
        except Exception:
            continue
        by_id[nid] = n

    lines: list[str] = []
    keep = sorted(node_ids)[: max(0, int(max_nodes))]
    for nid in keep:
        node = by_id.get(nid)
        if not node:
            continue
        lines.append(
            f"- node_id={nid} name={node.get('name')!r} type={node.get('node_type')!r} "
            f"bit_start={node.get('bit_start')!r} size_bits={node.get('size_bits')!r} "
            f"parent_id={node.get('parent_id')!r} children={node.get('children_ids')!r}"
        )
    if not lines:
        return ""
    return "Relevant tree nodes:\n" + "\n".join(lines)


def _read_log_tail(path: Optional[Path], *, max_chars: int = 12000) -> str:
    if path is None:
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(data) <= max_chars:
        return data
    return data[-max_chars:]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _find_tshark_under_ws_build_dir(ws_build_dir: Path) -> Optional[Path]:
    run_dir = ws_build_dir / "run"
    candidates = [
        run_dir / "tshark",
        run_dir / "tshark.exe",
        run_dir / "Wireshark.app" / "Contents" / "MacOS" / "tshark",
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _ensure_tshark_under_ws_build_dir(
    ws_build_dir: Path,
    *,
    log_path: Path,
    jobs: int = 4,
) -> Path:
    """Build `tshark` under a Wireshark build dir if it is missing."""
    existing = _find_tshark_under_ws_build_dir(ws_build_dir)
    if existing is not None:
        return existing

    cmd = [
        "cmake",
        "--build",
        ".",
        "--target",
        "tshark",
        "--parallel",
        str(int(jobs)),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            cmd,
            cwd=ws_build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(proc.stdout or "")
        handle.write("\n")

    if proc.returncode != 0:
        raise RuntimeError(
            f"tshark not found under {ws_build_dir / 'run'} and auto-build failed (rc={proc.returncode}). "
            f"See log: {log_path}"
        )

    built = _find_tshark_under_ws_build_dir(ws_build_dir)
    if built is None:
        raise RuntimeError(
            f"tshark build reported success but tshark is still missing under {ws_build_dir / 'run'}. "
            f"See log: {log_path}"
        )
    return built


_SEMANTIC_EXTRA_RULES = """
Semantic repair rules (must follow):
- Oracle: the Stage B traffic_fixed_tree (tree_path) is the authoritative spec for this stage;.
- Keep protocol filter_name EXACTLY equal to the protocol under test (do NOT change registration filter_name).
- Zeek+Spicy semantic mismatches are almost never fixed by renaming the module, analyzer, EVT hooks, or protocol registration. Do NOT rename Spicy modules/units/analyzers/EVT declarations during semantic repair unless the syntax log explicitly reports a missing module/unit name.
- Do NOT merge multiple leaf fields into a single big "payload" field unless the tree also models it as one field.
- Field spans must match the Stage B traffic_fixed_tree leaf boundaries. If a leaf is not byte-aligned, still model it as a leaf and preserve the bit-level boundary.
- Zeek+Spicy: when the tree has sub-byte leaves (`uint1`, `uint2`, `uint3`, `uint4`, bit_start with `+ 4`, etc.), do not parse them as a full `uint8`; use a `bitfield(N)` wrapper or bounded bytes wrapper so leaf spans preserve the bit-level boundary.
- Length fields must be dissected as standalone leaf fields and then used to bound subsequent payload parsing.
- Selector/condition_on fields must be dissected as standalone leaf fields; choose variants based on the selector value.
- If fields appear globally shifted, first check starting offset (tvb offset/base_offset) and ensure you start at the protocol layer payload.
- Prefer a small ```diff``` patch over rewriting the whole source file; semantic repair prompts often include large generated parsers, and full-file rewrites are likely to be truncated.
- Zeek+Spicy: if you use `bitfield(...)`, ALWAYS add `&bit-order=spicy::BitOrder::MSB0` (ParGen canonical) and `import spicy;` (never `spicy::BitOrder::MSB`/`LSB`).
- Zeek+Spicy: `bitfield(N)` containers must use a byte-multiple width accepted by spicyz (8, 16, 24, 32, ...). Do NOT emit standalone `bitfield(1)`..`bitfield(7)`. For a 4-bit selector followed by a 32-bit value at bit+4, parse the byte-multiple enclosing bytes and expose both leaves from that wrapper instead of using `bitfield(4)`.
- Scapy: avoid leaving protocol payload as a single `Raw.load` blob; ensure PacketLenField/ConditionalField choices produce leaf fields.
- Scapy: ParGen export extracts leaves by replaying `fields_desc` via `Field.getfield(...)`; do NOT rely on `guess_payload_class` or the `.payload` chain to expose on-wire leaf fields.
- Scapy: avoid using `PacketField` for fixed-length headers that precede other fields (it can consume the remainder as payload and starve later fields). Prefer inline header fields, `PacketLenField` (fixed length), or override `extract_padding` to stop at the header boundary.
- Scapy: any generic fallback payload field (e.g., `StrLenField(...Payload...)`, `PacketLenField`, `Raw`) MUST be last and MUST be gated to unknown selector values; otherwise it will swallow bytes and prevent later leaf fields from parsing.
- Scapy: direction (request/response) based on sport/dport is most reliable at the ROOT layer where `pkt.underlayer` is TCP/UDP. If you branch inside a nested Packet stored in PacketField/PacketLenField, `pkt.underlayer` is the parent Packet; access transport via `pkt.underlayer.underlayer` (if available) or move the branch outward.
- Wireshark-Lua: for TLV/typed value protocols, do not leave known selector variants as a generic bytes/value field; define ProtoFields and bounded subtree:add(...) calls for each Stage-B leaf under known variants.
""".strip()

_HEX_RE = re.compile(r"[^0-9a-fA-F]")
_GETFIELD_FAIL_RE = re.compile(r"getfield\\(([^)]+)\\)")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _short_hex(value: Any, *, max_chars: int = 96) -> str:
    s = str(value or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "..."


def _load_tooljson_packets(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def _packet_id(packet: Mapping[str, Any], fallback: int) -> int:
    # pred uses packet_no; gt uses frame_no.
    for key in ("frame_no", "packet_no", "frame_index"):
        if key in packet:
            try:
                return int(packet.get(key) or 0)
            except Exception:
                continue
    return fallback


def _tooljson_packet_coverage(packet: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    hex_str = str(packet.get("hex") or "")
    hex_len = 0
    if hex_str:
        try:
            hex_len = len(hex_str) // 2
        except Exception:
            hex_len = 0

    max_end = 0
    field_count = 0
    fields = packet.get("fields") or []
    if isinstance(fields, list):
        for f in fields:
            if not isinstance(f, dict):
                continue
            field_count += 1
            off = _safe_int(f.get("offset"), default=0)
            ln = _safe_int(f.get("length"), default=0)
            if off < 0 or ln <= 0:
                continue
            max_end = max(max_end, int(off) + int(ln))

    uncovered = 0
    if hex_len > 0 and max_end >= 0 and max_end < hex_len:
        uncovered = hex_len - max_end

    return hex_len, max_end, uncovered, field_count


def _normalize_hex(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    s = _HEX_RE.sub("", s)
    return s.lower()


def _field_l1_key(field: Mapping[str, Any]) -> Tuple[int, int, str]:
    off = _safe_int(field.get("offset"), default=0)
    ln = _safe_int(field.get("length"), default=0)
    raw = _normalize_hex(field.get("raw"))
    return int(off), int(ln), raw


def _scapy_common_getfield_failures(pred_by_id: Mapping[int, Dict[str, Any]], frame_ids: Sequence[int]) -> List[str]:
    counts: Dict[str, int] = {}
    frames_with_errors = 0
    for fid in frame_ids:
        pkt = pred_by_id.get(int(fid))
        if not pkt:
            continue
        dbg = pkt.get("debug") if isinstance(pkt, dict) else None
        if not isinstance(dbg, dict):
            continue
        errs = dbg.get("errors")
        if not isinstance(errs, list) or not errs:
            continue
        frames_with_errors += 1
        for e in errs:
            m = _GETFIELD_FAIL_RE.search(str(e))
            if not m:
                continue
            name = str(m.group(1)).strip()
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1

    if not counts:
        return []

    denom = max(1, frames_with_errors)
    top = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:3]
    return [f"{name} ({cnt}/{denom})" for name, cnt in top]


def _scapy_merge_culprit_hints(
    diff_json: Mapping[str, Any],
    *,
    pred_by_id: Mapping[int, Dict[str, Any]],
    max_candidates: int = 2,
) -> List[str]:
    worst = diff_json.get("worst_packets") if isinstance(diff_json.get("worst_packets"), list) else []
    if not worst:
        return []

    candidates: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}

    def _range(off: int, ln: int) -> Tuple[int, int]:
        return int(off), int(off) + max(0, int(ln))

    for pkt in worst:
        if not isinstance(pkt, dict):
            continue
        frame_no = _safe_int(pkt.get("frame_no"), default=0)
        if frame_no <= 0:
            continue
        pred_pkt = pred_by_id.get(frame_no)
        if not pred_pkt:
            continue

        pred_fields = pred_pkt.get("fields") if isinstance(pred_pkt, dict) else None
        if not isinstance(pred_fields, list):
            pred_fields = []
        pred_fields = [f for f in pred_fields if isinstance(f, dict)]

        pred_field_by_key: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        pred_field_by_ol: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for f in pred_fields:
            k = _field_l1_key(f)
            pred_field_by_key[k] = f
            pred_field_by_ol.setdefault((k[0], k[1]), []).append(f)

        missing_items = pkt.get("missing") if isinstance(pkt.get("missing"), list) else []
        extra_items = pkt.get("extra") if isinstance(pkt.get("extra"), list) else []

        missing: List[Dict[str, Any]] = []
        for m in missing_items:
            if not isinstance(m, dict):
                continue
            off = _safe_int(m.get("offset"), default=-1)
            ln = _safe_int(m.get("length"), default=0)
            if off < 0 or ln <= 0:
                continue
            missing.append(m)

        for ex in extra_items:
            if not isinstance(ex, dict):
                continue
            ex_off = _safe_int(ex.get("offset"), default=-1)
            ex_ln = _safe_int(ex.get("length"), default=0)
            if ex_off < 0 or ex_ln <= 0:
                continue
            ex_raw = _normalize_hex(ex.get("raw"))
            pred_field = pred_field_by_key.get((ex_off, ex_ln, ex_raw))
            if pred_field is None:
                options = pred_field_by_ol.get((ex_off, ex_ln)) or []
                pred_field = options[0] if options else None
            if pred_field is None:
                continue

            ex_rng = _range(ex_off, ex_ln)
            covered: List[Dict[str, Any]] = []
            for m in missing:
                m_off = _safe_int(m.get("offset"), default=0)
                m_ln = _safe_int(m.get("length"), default=0)
                m_rng = _range(m_off, m_ln)
                if ex_rng[0] <= m_rng[0] and ex_rng[1] >= m_rng[1] and ex_rng != m_rng:
                    covered.append(m)
            if not covered:
                continue

            name = str(pred_field.get("name") or "").strip()
            abbrev = str(pred_field.get("abbrev") or "").strip()
            key = (name or "<unknown>", abbrev or "<unknown>", int(ex_off), int(ex_ln))
            st = candidates.setdefault(
                key,
                {
                    "frames": set(),
                    "missing_covered": 0,
                    "examples": [],
                },
            )
            st["frames"].add(frame_no)
            st["missing_covered"] += len(covered)
            if len(st["examples"]) < 2:
                ex_names: List[str] = []
                for m in covered[:4]:
                    nm = str(m.get("name") or m.get("format_path") or "").strip()
                    if nm:
                        ex_names.append(nm)
                st["examples"].append({"frame_no": frame_no, "missing": ex_names})

    if not candidates:
        return []

    ranked = sorted(
        candidates.items(),
        key=lambda kv: (
            -int(kv[1].get("missing_covered") or 0),
            -len(kv[1].get("frames") or set()),
            -int(kv[0][3]),
            str(kv[0][0]),
        ),
    )[: max(1, int(max_candidates))]

    lines: List[str] = []
    for (name, abbrev, off, ln), st in ranked:
        frames = sorted([int(x) for x in (st.get("frames") or set())])[:6]
        missing_covered = int(st.get("missing_covered") or 0)
        examples = st.get("examples") or []
        example_missing: List[str] = []
        if isinstance(examples, list) and examples:
            for ex in examples:
                if not isinstance(ex, dict):
                    continue
                for nm in ex.get("missing") or []:
                    if isinstance(nm, str) and nm and nm not in example_missing:
                        example_missing.append(nm)
                if len(example_missing) >= 6:
                    break
        hint = (
            f"[SCAPY][HINT] Pred field {name!r} (abbrev={abbrev!r} off={off} len={ln}) "
            f"covers {missing_covered} GT-missing leaf fields across worst frames {frames}. "
            "This strongly suggests a broad length-bounded field (StrLenField/PacketLenField/PacketField/Raw) is swallowing "
            "bytes that should be parsed as separate leaf fields. Fix by gating that field to unknown selector values only "
            "and/or moving it to the END of fields_desc so known variants parse their leaf fields first."
        )
        if example_missing:
            hint += f" Example GT-missing leaves: {example_missing[:6]!r}."
        lines.append(hint)
    return lines


def _summarize_semantic_diff(diff_json: Dict[str, Any], *, max_packets: int = 5, max_items: int = 6) -> str:
    lines: List[str] = []
    summary = diff_json.get("summary") if isinstance(diff_json.get("summary"), dict) else {}
    if isinstance(summary, dict) and summary:
        frames = _safe_int(summary.get("frames_evaluated"), default=0)
        frames_with_gt = _safe_int(summary.get("frames_with_gt"), default=0)
        frames_with_pred = _safe_int(summary.get("frames_with_pred"), default=0)
        frames_missing_pred = _safe_int(summary.get("frames_missing_pred"), default=0)
        frames_extra_pred = _safe_int(summary.get("frames_extra_pred"), default=0)
        missing = _safe_int(summary.get("missing_total"), default=0)
        extra = _safe_int(summary.get("extra_total"), default=0)
        merged = _safe_int(summary.get("merged_total"), default=0)
        shift = summary.get("global_shift_hint")
        lines.append(
            "Semantic diff summary: "
            f"frames={frames} gt_frames={frames_with_gt} pred_frames={frames_with_pred} "
            f"missing_pred_frames={frames_missing_pred} extra_pred_frames={frames_extra_pred} "
            f"missing_total={missing} extra_total={extra} merged_total={merged} shift_hint={shift}"
        )

    worst = diff_json.get("worst_packets") or []
    if not isinstance(worst, list):
        worst = []
    keep = worst[: max(0, int(max_packets))]
    for pkt in keep:
        if not isinstance(pkt, dict):
            continue
        frame_no = _safe_int(pkt.get("frame_no"), default=0)
        counts = pkt.get("counts") if isinstance(pkt.get("counts"), dict) else {}
        missing_cnt = _safe_int((counts or {}).get("missing"), default=0)
        extra_cnt = _safe_int((counts or {}).get("extra"), default=0)
        wrong_raw_cnt = _safe_int((counts or {}).get("wrong_raw"), default=0)
        lines.append(f"worst frame={frame_no} missing={missing_cnt} extra={extra_cnt} wrong_raw={wrong_raw_cnt}")

        missing_items = pkt.get("missing") or []
        if not isinstance(missing_items, list):
            missing_items = []
        for item in missing_items[: max(0, int(max_items))]:
            if not isinstance(item, dict):
                continue
            off = _safe_int(item.get("offset"), default=0)
            ln = _safe_int(item.get("length"), default=0)
            name = str(item.get("name") or "").strip()
            raw = _short_hex(item.get("raw"))
            lines.append(f"  GT missing off={off} len={ln} name={name!r} raw={raw}")

        extra_items = pkt.get("extra") or []
        if not isinstance(extra_items, list):
            extra_items = []
        for item in extra_items[: max(0, int(max_items))]:
            if not isinstance(item, dict):
                continue
            off = _safe_int(item.get("offset"), default=0)
            ln = _safe_int(item.get("length"), default=0)
            raw = _short_hex(item.get("raw"))
            lines.append(f"  Pred extra off={off} len={ln} raw={raw}")

    return "\n".join(lines).strip()


def _summarize_pred_packets_for_frames(
    pred_tooljson_path: Path,
    frame_ids: Sequence[int],
    *,
    max_fields_per_packet: int = 20,
    gt_min_missing_offsets: Optional[Mapping[int, int]] = None,
) -> str:
    packets = _load_tooljson_packets(pred_tooljson_path)
    by_id: Dict[int, Dict[str, Any]] = {}
    for idx, pkt in enumerate(packets, start=1):
        by_id[_packet_id(pkt, idx)] = pkt

    lines: List[str] = []
    for fid in frame_ids:
        pkt = by_id.get(int(fid))
        if not pkt:
            continue
        hex_len, max_end, uncovered, field_count = _tooljson_packet_coverage(pkt)
        debug = pkt.get("debug") if isinstance(pkt.get("debug"), dict) else {}
        gt_min_missing = None
        if gt_min_missing_offsets is not None:
            try:
                gt_min_missing = int(gt_min_missing_offsets.get(int(fid)) or 0) or None
            except Exception:
                gt_min_missing = None
        lines.append(
            f"Pred packet frame_no={fid} hex_len={hex_len} parsed_end={max_end} uncovered_tail={uncovered} "
            f"field_count={field_count} gt_min_missing_offset={gt_min_missing} debug={debug}"
        )

        fields = pkt.get("fields") or []
        if not isinstance(fields, list):
            fields = []

        def key(item: Any) -> Tuple[int, int, str]:
            if not isinstance(item, dict):
                return (0, 0, "")
            return (
                _safe_int(item.get("offset"), default=0),
                _safe_int(item.get("length"), default=0),
                str(item.get("abbrev") or item.get("name") or ""),
            )

        kept = 0
        for f in sorted([x for x in fields if isinstance(x, dict)], key=key):
            off = _safe_int(f.get("offset"), default=0)
            ln = _safe_int(f.get("length"), default=0)
            name = str(f.get("name") or "").strip()
            abbrev = str(f.get("abbrev") or "").strip()
            raw = _short_hex(f.get("raw"))
            lines.append(f"  pred off={off} len={ln} name={name!r} abbrev={abbrev!r} raw={raw}")
            kept += 1
            if kept >= int(max_fields_per_packet):
                break

    return "\n".join(lines).strip()


def _count_load_like_fields(packets: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    packets_with_blob = 0
    blob_fields = 0
    for pkt in packets:
        fields = pkt.get("fields") or []
        if not isinstance(fields, list):
            continue
        has_blob = False
        for f in fields:
            if not isinstance(f, dict):
                continue
            name = str(f.get("name") or "").strip().lower()
            abbrev = str(f.get("abbrev") or "").strip().lower()
            if name == "load" or abbrev.endswith(".load"):
                blob_fields += 1
                has_blob = True
        if has_blob:
            packets_with_blob += 1
    return packets_with_blob, blob_fields


def _mode_int(values: Sequence[int]) -> Tuple[Optional[int], int]:
    counts: Dict[int, int] = {}
    for v in values:
        try:
            i = int(v)
        except Exception:
            continue
        counts[i] = counts.get(i, 0) + 1
    if not counts:
        return None, 0
    mode, cnt = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0]
    return int(mode), int(cnt)


def _scapy_top_root_classes_from_tooljson(
    packets: Sequence[Dict[str, Any]], *, top_k: int = 3
) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for pkt in packets:
        dbg = pkt.get("debug") if isinstance(pkt, dict) else None
        if not isinstance(dbg, dict):
            continue
        name = str(dbg.get("parse_cls") or "").strip()
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    out = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return out[: max(0, int(top_k))]


def _scapy_static_root_fields_desc(files: Mapping[str, str], root_class_name: str) -> Optional[Dict[str, Any]]:
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return str(node.id)
        if isinstance(node, ast.Attribute):
            return str(node.attr)
        return ""

    def _const_str(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return ""

    for fname, content in files.items():
        try:
            tree = ast.parse(content or "")
        except Exception:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if str(node.name) != str(root_class_name):
                continue

            fields_desc_calls: List[str] = []
            packetfield_before_end = False
            has_guess_payload_class = False

            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and str(stmt.name) == "guess_payload_class":
                    has_guess_payload_class = True

                value = None
                if isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "fields_desc":
                            value = stmt.value
                            break
                elif isinstance(stmt, ast.AnnAssign):
                    if isinstance(stmt.target, ast.Name) and stmt.target.id == "fields_desc":
                        value = stmt.value

                if value is None:
                    continue
                if not isinstance(value, (ast.List, ast.Tuple)):
                    continue

                elts = list(value.elts)
                for idx, elt in enumerate(elts):
                    if not isinstance(elt, ast.Call):
                        fields_desc_calls.append(type(elt).__name__)
                        continue
                    callee = _call_name(elt.func) or "<call>"
                    first_arg = ""
                    if elt.args:
                        first_arg = _const_str(elt.args[0])
                    if first_arg:
                        fields_desc_calls.append(f"{callee}({first_arg})")
                    else:
                        fields_desc_calls.append(callee)
                    if callee == "PacketField" and idx < len(elts) - 1:
                        packetfield_before_end = True

            return {
                "file": fname,
                "class": root_class_name,
                "fields_desc": fields_desc_calls,
                "packetfield_before_end": packetfield_before_end,
                "has_guess_payload_class": has_guess_payload_class,
            }

    return None


def _scapy_static_conditional_underlayer_port_checks(files: Mapping[str, str]) -> List[Dict[str, Any]]:
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return str(node.id)
        if isinstance(node, ast.Attribute):
            return str(node.attr)
        return ""


    findings: List[Dict[str, Any]] = []

    for fname, content in files.items():
        try:
            tree = ast.parse(content or "")
        except Exception:
            continue

        class Visitor(ast.NodeVisitor):
            def __init__(self, *, file_name: str) -> None:
                self.file_name = file_name
                self.class_name: Optional[str] = None
                self.items: List[Dict[str, Any]] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
                prev = self.class_name
                self.class_name = str(node.name)
                self.generic_visit(node)
                self.class_name = prev


        v = Visitor(file_name=fname)
        v.visit(tree)
        findings.extend(v.items)

    return findings


def run_stage_c_semantic_with_auto_repair(
    *,
    syntax_backend: SyntaxBackend,
    plugin_src_dir: Path,
    tree_path: Path,
    pcap_path: Path,
    frame_index: Optional[Sequence[int]] = None,
    protocol_filter_name: str,
    file_prefix: str,
    out_dir: Path,
    logs_root: Path,
    codegen_profile: Optional[Dict[str, Any]] = None,
    constraints_mode: str = "codegen",
    llm_config: Optional[LLMConfig] = None,
    syntax_cfg: Optional[SyntaxRepairLoopConfig] = None,
    semantic_cfg: Optional[SemanticRepairLoopConfig] = None,
    tshark_override: Optional[Path] = None,
    source_globs: Optional[Sequence[str]] = None,
    tshark_extra_args: Optional[List[str]] = None,
    ws_isolate_plugins: bool = True,
    ws_plugin_id: Optional[str] = None,
    scapy_entry_override: Optional[Path] = None,
    scapy_ports: Optional[Sequence[int]] = None,
    scapy_transport: str = "auto",
    progress_printer: Optional[Callable[[str], None]] = None,
) -> SemanticValidationResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    syntax_cfg = syntax_cfg or SyntaxRepairLoopConfig()
    semantic_cfg = semantic_cfg or SemanticRepairLoopConfig()
    source_globs = source_globs or ("*.c", "*.h")

    def _log(message: str) -> None:
        if progress_printer is None:
            return
        try:
            progress_printer(message)
        except Exception:
            return

    llm: Optional[LLMClient] = None
    if semantic_cfg.enable_llm_repair:
        try:
            llm = create_llm_client(llm_config)
        except Exception:
            llm = None

    last_validation: Optional[SemanticValidationResult] = None
    last_good_files: Optional[Dict[str, str]] = None
    last_good_validation: Optional[SemanticValidationResult] = None
    best_files: Optional[Dict[str, str]] = None
    best_validation: Optional[SemanticValidationResult] = None

    host_key = str(getattr(syntax_backend, "name", "") or "").strip() or None
    pcap_layer = infer_pcap_layer(codegen_profile, host=host_key)
    pcap_ip_proto = infer_ip_proto(codegen_profile, host=host_key) if pcap_layer == "ip_payload" else None

    def _restore_plugin_sources(snapshot: Mapping[str, str]) -> None:
        for rel, content in snapshot.items():
            if not isinstance(rel, str) or not rel:
                continue
            path = plugin_src_dir / rel
            try:
                if not path.exists() or not path.is_file():
                    continue
            except OSError:
                continue
            try:
                path.write_text((content or "").rstrip() + "\n", encoding="utf-8")
            except Exception:
                continue

    for round_idx in range(int(semantic_cfg.max_rounds)):
        _log(f"  [SemanticRepair] round={round_idx:02d}: start")
        round_dir = out_dir / f"round{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        syntax_logs = logs_root / f"syntax_round{round_idx:02d}"
        syntax_logs.mkdir(parents=True, exist_ok=True)


        syntax_result = run_syntax_with_auto_repair(
            backend=syntax_backend,
            plugin_src_dir=plugin_src_dir,
            protocol_filter_name=protocol_filter_name,
            file_prefix=file_prefix,
            logs_root=syntax_logs,
            codegen_profile=codegen_profile,
            backend_config=(
                {
                    **(codegen_profile or {}),
                    "pcap_path": str(pcap_path),
                    "runtime_check": True,
                    "max_runtime_packets": min(3, int(semantic_cfg.max_packets) if semantic_cfg.max_packets else 3),
                    "scapy_transport": scapy_transport,
                    "scapy_ports": list(scapy_ports or []),
                    "format_tree_path": str(tree_path),
                }
                if getattr(syntax_backend, "name", "") == "scapy"
                else None
            ),
            llm_config=llm_config,
            cfg=syntax_cfg,
            source_globs=source_globs,
            progress_printer=progress_printer,
        )
        # Syntax repair rewrites complete source files. Re-apply deterministic
        # constraint validation hooks afterwards so repair cannot silently drop
        # the runtime-checking loop.

        if not syntax_result.ok:
            _log(f"  [SemanticRepair] round={round_idx:02d}: syntax_ok=False; stopping")
            last_validation = SemanticValidationResult(
                ok=False,
                f1_l1=0.0,
                f1_l2=0.0,
                skipped_packets=0,
                stats=TreeGTBuildStats(),
            )
            if last_good_files is not None:
                _restore_plugin_sources(last_good_files)
                if last_good_validation is not None:
                    last_validation = last_good_validation
                _log(
                    f"  [SemanticRepair] round={round_idx:02d}: syntax failure after a prior valid round; "
                    "restored last good sources"
                )
                return last_validation
            if best_files is not None and best_validation is not None:
                _restore_plugin_sources(best_files)
                _log(
                    f"  [SemanticRepair] round={round_idx:02d}: syntax failure; restored best sources "
                    f"(best_L1_F1={float(best_validation.f1_l1):.4f}, best_L2_F1={float(best_validation.f1_l2):.4f})"
                )
                return best_validation
            return last_validation

        export_log_name = "tshark_export.log"
        if getattr(syntax_backend, "name", "") == "scapy":
            export_log_name = "scapy_export.log"
        if getattr(syntax_backend, "name", "") == "zeek_spicy":
            export_log_name = "zeek_spicy_export.log"
        export_log_path = round_dir / export_log_name

        try:
            if getattr(syntax_backend, "name", "") == "scapy":
                _log(
                    f"  [SemanticRepair] round={round_idx:02d}: exporting with scapy "
                    f"(transport={scapy_transport}, ports={list(scapy_ports or [])})"
                )
                export_result = export_scapy_tooljson(
                    plugin_src_dir=plugin_src_dir,
                    pcap_path=pcap_path,
                    filter_name=protocol_filter_name,
                    out_dir=round_dir,
                    entry_override=scapy_entry_override,
                    ports=scapy_ports,
                    transport=scapy_transport,
                    max_packets=int(semantic_cfg.max_packets) if semantic_cfg.max_packets else 0,
                    include_payload_layers=True,
                    codegen_profile=codegen_profile,
                )
            elif getattr(syntax_backend, "name", "") == "zeek_spicy":
                hlto_path = None
                spicyz_path = None
                if syntax_result.artifacts is not None:
                    spicyz_path = syntax_result.artifacts.tool_bin
                    extra = syntax_result.artifacts.extra or {}
                    raw_hlto = extra.get("hlto")
                    if raw_hlto:
                        try:
                            hlto_path = Path(str(raw_hlto)).expanduser()
                        except Exception:
                            hlto_path = None

                _log(f"  [SemanticRepair] round={round_idx:02d}: exporting with spicy-dump (zeek_spicy)")
                export_result = export_zeek_spicy_tooljson(
                    plugin_src_dir=plugin_src_dir,
                    pcap_path=pcap_path,
                    filter_name=protocol_filter_name,
                    out_dir=round_dir,
                    codegen_profile=codegen_profile,
                    frame_index=frame_index,
                    max_packets=int(semantic_cfg.max_packets) if semantic_cfg.max_packets else 0,
                    hlto_path=hlto_path,
                    spicyz_path=spicyz_path,
                    fail_on_all_parse_failures=True,
                    disable_runtime_constraints_for_export=True,
                )
            else:
                if tshark_override is not None:
                    tshark_path = tshark_override
                else:
                    tshark_path = None
                    if syntax_result.artifacts and syntax_result.artifacts.tool_bin:
                        tshark_path = syntax_result.artifacts.tool_bin
                    if tshark_path is None and getattr(syntax_backend, "name", "") == "wireshark_c":
                        ws_build_dir = None
                        if syntax_result.artifacts is not None and syntax_result.artifacts.build_dir is not None:
                            ws_build_dir = syntax_result.artifacts.build_dir
                        if ws_build_dir is not None:
                            tshark_path = _ensure_tshark_under_ws_build_dir(
                                ws_build_dir,
                                log_path=export_log_path,
                            )
                            if syntax_result.artifacts is not None:
                                syntax_result.artifacts.tool_bin = tshark_path
                if tshark_path is None:
                    raise RuntimeError("Syntax stage succeeded but tshark path is missing from syntax artifacts.")

                _log(
                    f"  [SemanticRepair] round={round_idx:02d}: exporting with tshark (extra_args={tshark_extra_args or []})"
                )
                export_result = export_wireshark_tooljson(
                    tshark_path=tshark_path,
                    pcap_path=pcap_path,
                    filter_name=protocol_filter_name,
                    out_dir=round_dir,
                    export_jsonraw=True,
                    export_texttree=True,
                    max_packets=int(semantic_cfg.max_packets) if semantic_cfg.max_packets else 0,
                    extra_tshark_args=tshark_extra_args,
                    codegen_profile=codegen_profile,
                    ws_isolate_plugins=bool(ws_isolate_plugins),
                    ws_plugin_id=(str(ws_plugin_id) if (host_key == "wireshark_c" and ws_plugin_id) else None),
                )
        except Exception as exc:
            # Export failures (including plugin load/runtime errors) must surface here and be repaired
            # through the existing semantic auto-repair loop.
            _log(f"  [SemanticRepair] round={round_idx:02d}: export failed: {type(exc).__name__}: {exc}")
            _log(f"  [SemanticRepair] round={round_idx:02d}: export_log={export_log_path}")
            last_validation = SemanticValidationResult(
                ok=False,
                f1_l1=0.0,
                f1_l2=0.0,
                skipped_packets=0,
                stats=TreeGTBuildStats(),
            )

            if last_good_files is not None:
                _restore_plugin_sources(last_good_files)
                if last_good_validation is not None:
                    last_validation = last_good_validation
                _log(
                    f"  [SemanticRepair] round={round_idx:02d}: rolled back to last good sources "
                    f"(round={int(round_idx) - 1:02d}); skipping export-repair for this round"
                )
                if (round_idx + 1) >= int(semantic_cfg.max_rounds):
                    return last_validation
                continue

            if (round_idx + 1) >= int(semantic_cfg.max_rounds):
                _log(f"  [SemanticRepair] round={round_idx:02d}: reached max_rounds; stopping without applying unvalidated patch")
                return last_validation

            non_repairable = (
                "tshark path is missing from syntax artifacts",
                "tshark not found under",
                "tshark build reported success but tshark is still missing",
                "auto-build failed",
            )
            if any(tok in str(exc) for tok in non_repairable):
                _log(f"  [SemanticRepair] round={round_idx:02d}: export failure is not repairable by editing sources; stopping")
                return last_validation
            if not semantic_cfg.enable_llm_repair or llm is None:
                _log(f"  [SemanticRepair] round={round_idx:02d}: LLM repair disabled/unavailable; stopping")
                return last_validation

            errors: list[str] = [
                f"[EXPORT] {type(exc).__name__}: {exc}",
                f"[EXPORT] {export_log_name} tail follows.",
            ]
            log_tail = _read_log_tail(
                export_log_path,
                max_chars=int(semantic_cfg.log_tail_chars),
            )
            files = _scan_plugin_sources(plugin_src_dir, source_globs)
            agent = CodeRepairAgent(llm)
            prompt_path = logs_root / f"round{round_idx:02d}_export_prompt.txt"
            resp_path = logs_root / f"round{round_idx:02d}_export_response.txt"
            _log(
                f"  [SemanticRepair] round={round_idx:02d}: LLM export-repair "
                f"(prompt={prompt_path}, response={resp_path})"
            )
            patched = agent.repair_files(
                files=files,
                diagnostics=RepairDiagnostics(
                    tool_name="semantic_export",
                    protocol_name=protocol_filter_name,
                    errors=tuple(errors),
                    log_tail=log_tail,
                ),
                codegen_profile=codegen_profile,
                extra_rules=_SEMANTIC_EXTRA_RULES,
                prompt_path=prompt_path,
                response_path=resp_path,
                allow_new_files=False,
            )
            if not patched:
                _log(f"  [SemanticRepair] round={round_idx:02d}: no LLM patch; stopping")
                break

            changed_any = False
            for name, content in patched.items():
                path = plugin_src_dir / name
                if not path.exists() or not path.is_file():
                    continue
                try:
                    old = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    old = ""
                if (content or "").strip() and content.strip() != (old or "").strip():
                    path.write_text(content.rstrip() + "\n", encoding="utf-8")
                    changed_any = True

            if not changed_any:
                _log(f"  [SemanticRepair] round={round_idx:02d}: patch produced no file changes; stopping")
                break
            _log(f"  [SemanticRepair] round={round_idx:02d}: patched_files={sorted(patched.keys())}")
            continue

        last_validation = run_semantic_validation(
            tree_path=tree_path,
            pred_tooljson_path=export_result.tooljson_path,
            pcap_path=pcap_path,
            pcap_layer=pcap_layer,
            pcap_ip_proto=pcap_ip_proto,
            frame_index=frame_index,
            out_dir=round_dir,
            require_full_parse=bool(semantic_cfg.require_full_parse),
            max_packets=int(semantic_cfg.max_packets) if semantic_cfg.max_packets else 0,
            worst_k=int(semantic_cfg.worst_k),
            min_f1_l1=float(semantic_cfg.min_f1_l1),
        )
        _log(
            f"  [SemanticRepair] round={round_idx:02d}: "
            f"L1_F1={float(last_validation.f1_l1):.4f} L2_F1={float(last_validation.f1_l2):.4f} "
            f"skipped={int(last_validation.skipped_packets)}"
        )
        last_good_files = _scan_plugin_sources(plugin_src_dir, source_globs)
        last_good_validation = last_validation

        if best_validation is None or _semantic_selection_score(last_validation) > _semantic_selection_score(best_validation):
            best_validation = last_validation
            best_files = dict(last_good_files)
        elif best_files is not None and best_validation is not None:
            _restore_plugin_sources(best_files)
            _log(
                f"  [SemanticRepair] round={round_idx:02d}: no-regression gate restored best sources "
                f"(best_L1_F1={float(best_validation.f1_l1):.4f}, best_L2_F1={float(best_validation.f1_l2):.4f})"
            )
            return best_validation

        if int(last_validation.stats.packets_total) > 0 and int(last_validation.skipped_packets) >= int(last_validation.stats.packets_total):
            gt_report = round_dir / "semantic_gt_build_report.json"
            _log(
                f"  [SemanticRepair] round={round_idx:02d}: all frames skipped (pcap_layer={pcap_layer}, ip_proto={pcap_ip_proto}); "
                f"no_hex={int(last_validation.stats.packets_skipped_no_hex)} parse_fail={int(last_validation.stats.packets_skipped_parse_fail)}; "
                f"gt_build_report={gt_report}; semantic LLM repair has no signal; stopping"
            )
            return last_validation

        if bool(last_validation.ok):
            _log(f"  [SemanticRepair] round={round_idx:02d}: success (>= min_f1)")
            return last_validation

        if not semantic_cfg.enable_llm_repair or llm is None:
            _log(f"  [SemanticRepair] round={round_idx:02d}: below min_f1 and LLM repair disabled/unavailable; stopping")
            break
        if (round_idx + 1) >= int(semantic_cfg.max_rounds):
            if best_files is not None and best_validation is not None:
                _restore_plugin_sources(best_files)
                _log(
                    f"  [SemanticRepair] round={round_idx:02d}: reached max_rounds; restored best sources "
                    f"(best_L1_F1={float(best_validation.f1_l1):.4f}, best_L2_F1={float(best_validation.f1_l2):.4f})"
                )
                return best_validation
            _log(f"  [SemanticRepair] round={round_idx:02d}: reached max_rounds; stopping without applying unvalidated patch")
            break

        # Build LLM diagnostics from metrics + diff report.
        errors: list[str] = [
            f"[ORACLE] tree_path={tree_path} is the GT spec for this stage (traffic_fixed_tree); "
            "make exported tool.json match the tree-derived GT exactly.",
            f"[SEMANTIC] L1_F1={float(last_validation.f1_l1):.4f} (min={float(semantic_cfg.min_f1_l1):.4f})",
            f"[SEMANTIC] L2_F1={float(last_validation.f1_l2):.4f}",
            f"[SEMANTIC] skipped_packets={int(last_validation.skipped_packets)} "
            f"(no_hex={int(last_validation.stats.packets_skipped_no_hex)}, parse_fail={int(last_validation.stats.packets_skipped_parse_fail)})",
        ]

        diff_tail = ""
        pred_tail = ""
        static_tail = ""
        node_ids: set[int] = set()
        worst_frame_ids: List[int] = []
        gt_min_missing_offsets: Dict[int, int] = {}
        diff_json_obj: Optional[Dict[str, Any]] = None
        if last_validation.diff_report and last_validation.diff_report.exists():
            try:
                diff_json = json.loads(last_validation.diff_report.read_text(encoding="utf-8"))
                if isinstance(diff_json, dict):
                    diff_json_obj = diff_json
                    diff_tail = _summarize_semantic_diff(diff_json, max_packets=int(semantic_cfg.worst_k))
                worst = (diff_json.get("worst_packets") if isinstance(diff_json, dict) else None) or []
                for p in worst[: int(semantic_cfg.worst_k)]:
                    counts = (p.get("counts") or {}) if isinstance(p, dict) else {}
                    frame_no = 0
                    try:
                        frame_no = int(p.get("frame_no"))
                        worst_frame_ids.append(frame_no)
                    except Exception:
                        pass
                    errors.append(
                        f"[SEMANTIC] worst frame={p.get('frame_no')} missing={counts.get('missing', 0)} "
                        f"extra={counts.get('extra', 0)} wrong_raw={counts.get('wrong_raw', 0)}"
                    )
                    missing = p.get("missing") or []
                    if isinstance(missing, list):
                        min_off: Optional[int] = None
                        for item in missing:
                            if isinstance(item, dict) and "offset" in item:
                                off = _safe_int(item.get("offset"), default=-1)
                                if off >= 0:
                                    if min_off is None or off < min_off:
                                        min_off = off
                            if isinstance(item, dict) and "node_id" in item:
                                try:
                                    node_ids.add(int(item.get("node_id")))
                                except Exception:
                                    continue
                        if frame_no > 0 and min_off is not None:
                            gt_min_missing_offsets[frame_no] = int(min_off)
            except Exception:
                diff_tail = ""
                diff_json_obj = None

        files = _scan_plugin_sources(plugin_src_dir, source_globs)

        # Add Scapy-specific signals from the predicted tool.json to help the LLM localize failures.
        try:
            if getattr(syntax_backend, "name", "") == "scapy":
                pred_packets = _load_tooljson_packets(export_result.tooljson_path)
                packets_with_blob, blob_fields = _count_load_like_fields(pred_packets)
                if pred_packets:
                    errors.append(
                        f"[SCAPY] packets_with_blob_load={packets_with_blob}/{len(pred_packets)} "
                        f"blob_fields_total={blob_fields}"
                    )

                    cov_hex_lens: List[int] = []
                    cov_parsed_ends: List[int] = []
                    cov_uncovered: List[int] = []
                    cov_field_counts: List[int] = []
                    pred_by_id: Dict[int, Dict[str, Any]] = {}
                    for idx, pkt in enumerate(pred_packets, start=1):
                        pred_by_id[_packet_id(pkt, idx)] = pkt
                        hex_len, max_end, uncovered, field_count = _tooljson_packet_coverage(pkt)
                        if hex_len <= 0:
                            continue
                        cov_hex_lens.append(hex_len)
                        cov_parsed_ends.append(max_end)
                        cov_uncovered.append(uncovered)
                        cov_field_counts.append(field_count)

                    if cov_hex_lens:
                        uncovered_packets = sum(1 for u in cov_uncovered if int(u) > 0)
                        max_uncovered = max([int(u) for u in cov_uncovered] or [0])
                        avg_uncovered = float(sum([int(u) for u in cov_uncovered] or [0])) / float(len(cov_uncovered))
                        mode_end, mode_end_cnt = _mode_int(cov_parsed_ends)
                        mode_fields, mode_fields_cnt = _mode_int(cov_field_counts)
                        errors.append(
                            f"[SCAPY] coverage uncovered_tail_packets={uncovered_packets}/{len(cov_hex_lens)} "
                            f"max_uncovered={max_uncovered} avg_uncovered={avg_uncovered:.2f} "
                            f"mode_parsed_end={mode_end} (n={mode_end_cnt}) "
                            f"mode_field_count={mode_fields} (n={mode_fields_cnt})"
                        )
                        if uncovered_packets >= max(1, int(len(cov_hex_lens) * 0.6)):
                            errors.append(
                                "[SCAPY] hypothesis: parser likely stops early and leaves an uncovered tail. "
                                "A common cause is PacketField consuming the remainder as payload or a wrong length_from; "
                                "ensure fields_desc continues parsing leaf fields beyond the boundary."
                            )

                    if diff_json_obj is not None:
                        summary = diff_json_obj.get("summary") if isinstance(diff_json_obj.get("summary"), dict) else {}
                        missing_total = _safe_int((summary or {}).get("missing_total"), default=0)
                        merged_total = _safe_int((summary or {}).get("merged_total"), default=0)
                        extra_total = _safe_int((summary or {}).get("extra_total"), default=0)
                        if missing_total > 0 and merged_total >= int(missing_total * 0.9):
                            errors.append(
                                f"[SCAPY] diff indicates GT leaves are merged into larger pred fields: "
                                f"missing_total={missing_total} merged_total={merged_total} extra_total={extra_total}. "
                                "This usually means a broad length-bounded field (StrLenField/PacketLenField/PacketField/Raw) "
                                "is swallowing multiple leaf fields."
                            )

                    boundary_total = 0
                    boundary_match = 0
                    for fid in worst_frame_ids:
                        if fid <= 0:
                            continue
                        if fid not in pred_by_id:
                            continue
                        if fid not in gt_min_missing_offsets:
                            continue
                        _, pred_end, _, _ = _tooljson_packet_coverage(pred_by_id[fid])
                        boundary_total += 1
                        if int(pred_end) == int(gt_min_missing_offsets[fid]):
                            boundary_match += 1
                    if boundary_total:
                        errors.append(
                            f"[SCAPY] parsed_end_equals_GT_first_missing_offset={boundary_match}/{boundary_total} "
                            f"(worst frames)"
                        )
                        if boundary_match >= max(1, int(boundary_total * 0.6)):
                            errors.append(
                                "[SCAPY] hypothesis: parsed_end matches the first GT-missing offset on many worst frames, "
                                "so parsing stops exactly at that boundary. Check the field immediately before that offset "
                                "(often PacketField/PacketLenField/StrLenField bounds) and ensure it does not swallow the PDU."
                            )

                    if worst_frame_ids:
                        common_getfield = _scapy_common_getfield_failures(pred_by_id, worst_frame_ids)
                        if common_getfield:
                            errors.append(
                                "[SCAPY] common getfield(...) failures on worst frames: " + ", ".join(common_getfield)
                            )
                    if diff_json_obj is not None:
                        errors.extend(_scapy_merge_culprit_hints(diff_json_obj, pred_by_id=pred_by_id))

                    root_classes = _scapy_top_root_classes_from_tooljson(pred_packets, top_k=3)
                    root_names = [n for n, _ in root_classes]
                    static_lines: List[str] = []
                    if root_classes:
                        static_lines.append("Static plugin analysis:")
                        static_lines.append(f"- top_root_classes={root_classes!r}")
                        for root_name, cnt in root_classes:
                            info = _scapy_static_root_fields_desc(files, root_name)
                            if not info:
                                static_lines.append(f"- class={root_name!r} count={cnt} (class not found in sources)")
                                continue
                            raw_calls = info.get("fields_desc") if isinstance(info.get("fields_desc"), list) else []
                            calls = [str(x) for x in raw_calls if isinstance(x, str)]
                            if len(calls) > 12:
                                calls = calls[:12] + ["..."]
                            static_lines.append(
                                f"- class={root_name!r} count={cnt} file={info.get('file')!r} "
                                f"packetfield_before_end={bool(info.get('packetfield_before_end'))} "
                                f"has_guess_payload_class={bool(info.get('has_guess_payload_class'))} "
                                f"fields_desc={calls!r}"
                            )
                            if bool(info.get("packetfield_before_end")):
                                errors.append(
                                    f"[SCAPY] static: class {root_name!r} has PacketField before end in fields_desc "
                                    "(it may swallow the remainder and prevent later fields from parsing)"
                                )
                            if bool(info.get("has_guess_payload_class")):
                                errors.append(
                                    f"[SCAPY] static: class {root_name!r} defines guess_payload_class "
                                    "(ParGen export ignores payload chain; ensure fields_desc exposes all leaves)"
                                )

                    checks = _scapy_static_conditional_underlayer_port_checks(files)
                    if checks:
                        unsafe = [
                            c
                            for c in checks
                            if int(c.get("underlayer_depth") or 0) == 1
                            and str(c.get("class") or "") not in set(root_names)
                        ]
                        examples = []
                        for c in unsafe[:5]:
                            cls = str(c.get("class") or "?")
                            attr = str(c.get("attr") or "?")
                            examples.append(f"{cls}.{attr}")
                        static_lines.append(
                            f"- conditional_underlayer_port_checks_total={len(checks)} "
                            f"unsafe_nonroot_depth1={len(unsafe)} examples={examples!r}"
                        )
                        if unsafe:
                            ex = ", ".join(examples) if examples else "<unknown>"
                            errors.append(
                                f"[SCAPY] static: ConditionalField lambdas use pkt.underlayer.sport/dport in non-root classes "
                                f"(examples: {ex}). Underlayer is not TCP/UDP there; move direction checks to the root layer "
                                "or use pkt.underlayer.underlayer.<sport/dport>."
                            )

                    static_tail = "\n".join(static_lines).strip()
                if worst_frame_ids:
                    pred_tail = _summarize_pred_packets_for_frames(
                        export_result.tooljson_path,
                        [fid for fid in worst_frame_ids if int(fid) > 0],
                        gt_min_missing_offsets=gt_min_missing_offsets,
                    )
        except Exception:
            pred_tail = ""

        tree_summary = _summarize_tree_nodes(tree_path, node_ids)
        parse_ir_block = ""
        backend_name = str(getattr(syntax_backend, "name", "") or "")
        if backend_name == "scapy":
            try:
                from pargen.parser_realization.syntax_stage.backends.scapy import _scapy_structured_body_context

                scapy_tree_context = _scapy_structured_body_context(
                    {"format_tree_path": str(tree_path)},
                    max_chars=max(4000, min(12000, int(semantic_cfg.log_tail_chars))),
                ).strip()
            except Exception:
                scapy_tree_context = ""
            if scapy_tree_context:
                parse_ir_block = "<parse_ir_structured_summary>\n" + scapy_tree_context + "\n</parse_ir_structured_summary>"
        elif backend_name == "wireshark_lua":
            try:
                from pargen.parser_realization.syntax_stage.backends.wireshark_lua import _lua_structured_body_context

                lua_tree_context = _lua_structured_body_context(
                    {"format_tree_path": str(tree_path)},
                    max_lines=150,
                ).strip()
            except Exception:
                lua_tree_context = ""
            if lua_tree_context:
                parse_ir_block = "<parse_ir_structured_summary>\n" + lua_tree_context + "\n</parse_ir_structured_summary>"
        if not parse_ir_block:
            parse_ir_text = _read_text(tree_path).strip()
            if parse_ir_text:
                parse_ir_block = "<parse_ir_json>\n" + parse_ir_text + "\n</parse_ir_json>"

        context_summary = "\n\n".join(
            [t for t in (diff_tail.strip(), pred_tail.strip(), static_tail.strip(), tree_summary.strip()) if t]
        ).strip()

        # Hard cap ONLY the derived context summary (diff/pred/static/tree). Parse-IR is always included in full.
        cap = int(semantic_cfg.log_tail_chars)
        ellipsis = "\n...\n"
        if cap > 0 and len(context_summary) > cap:
            keep = max(0, cap - len(ellipsis))
            head_len = keep // 2
            tail_len = keep - head_len
            context_summary = (
                context_summary[:head_len].rstrip() + ellipsis + context_summary[-tail_len:].lstrip()
            ).strip()

        log_tail = "\n\n".join([t for t in (context_summary.strip(), parse_ir_block.strip()) if t]).strip()

        agent = CodeRepairAgent(llm)
        prompt_path = logs_root / f"round{round_idx:02d}_prompt.txt"
        resp_path = logs_root / f"round{round_idx:02d}_response.txt"
        _log(f"  [SemanticRepair] round={round_idx:02d}: LLM semantic-repair (prompt={prompt_path}, response={resp_path})")
        repair_files = files
        scapy_structured_view = False
        if str(getattr(syntax_backend, "name", "") or "") == "scapy":
            try:
                from pargen.parser_realization.syntax_stage.backends.scapy import make_scapy_structured_repair_view

                compact = make_scapy_structured_repair_view(files, ["SCAPY_STRUCTURED_BODY semantic repair"])
                if compact:
                    repair_files = compact
                    scapy_structured_view = True
            except Exception:
                repair_files = files
                scapy_structured_view = False
        patched = agent.repair_files(
            files=repair_files,
            diagnostics=RepairDiagnostics(
                tool_name="semantic_traffic",
                protocol_name=protocol_filter_name,
                errors=tuple(errors),
                log_tail=log_tail,
                log_tail_max_chars=0,
            ),
            codegen_profile=codegen_profile,
            extra_rules=_SEMANTIC_EXTRA_RULES,
            prompt_path=prompt_path,
            response_path=resp_path,
            allow_new_files=False,
        )
        if patched and scapy_structured_view:
            try:
                from pargen.parser_realization.syntax_stage.backends.scapy import merge_scapy_structured_repair_view

                patched = merge_scapy_structured_repair_view(original_files=files, patched_files=patched)
            except Exception:
                patched = patched

        if not patched:
            _log(f"  [SemanticRepair] round={round_idx:02d}: no LLM patch; stopping")
            break

        changed_any = False
        for name, content in patched.items():
            path = plugin_src_dir / name
            if not path.exists() or not path.is_file():
                continue
            try:
                old = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                old = ""
            if (content or "").strip() and content.strip() != (old or "").strip():
                path.write_text(content.rstrip() + "\n", encoding="utf-8")
                changed_any = True

        if not changed_any:
            _log(f"  [SemanticRepair] round={round_idx:02d}: patch produced no file changes; stopping")
            break
        _log(f"  [SemanticRepair] round={round_idx:02d}: patched_files={sorted(patched.keys())}")

    if best_files is not None and best_validation is not None:
        _restore_plugin_sources(best_files)
        return best_validation
    return last_validation or SemanticValidationResult(ok=False, f1_l1=0.0, f1_l2=0.0, stats=TreeGTBuildStats())
