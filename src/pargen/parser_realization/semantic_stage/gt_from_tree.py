from __future__ import annotations

import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import TreeGTBuildStats

from pargen.protocol_knowledge.tree_utils import normalize_protocol_tree
from pargen.protocol_knowledge.message_validation.interpreter import DynamicTreeInterpreter, LEAF_TYPES
from pargen.parser_realization.target_platforms.bit_utils import ceil_div, safe_value_int_msb0


_HEX_RE = re.compile(r"[^0-9a-fA-F]")

_PCAP_MAGIC_PCAPNG = b"\x0a\x0d\x0d\x0a"
_PCAP_MAGIC_LE = {b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"}
_PCAP_MAGIC_BE = {b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"}

# DLT values (pcap link-layer types)
_DLT_EN10MB = 1
_DLT_RAW = 101
_DLT_IPV4 = 228
_DLT_IPV6 = 229
_DLT_LINUX_SLL = 113
_DLT_LINUX_SLL2 = 276

# pcapng block types
_PCAPNG_SHB = 0x0A0D0D0A
_PCAPNG_IDB = 0x00000001
_PCAPNG_SPB = 0x00000003
_PCAPNG_EPB = 0x00000006

_PCAPNG_BOM_BE = b"\x1a\x2b\x3c\x4d"
_PCAPNG_BOM_LE = b"\x4d\x3c\x2b\x1a"

_ETHERTYPE_IPV4 = 0x0800
_ETHERTYPE_IPV6 = 0x86DD
_ETHERTYPE_VLAN = {0x8100, 0x88A8, 0x9100}

_IPPROTO_TCP = 6
_IPPROTO_UDP = 17


def _normalize_hex(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    s = _HEX_RE.sub("", s)
    return s.lower()


def _coerce_node_id(raw: Any) -> Any:
    try:
        return int(raw)
    except Exception:
        return raw


def _lookup(map_obj: Dict[Any, Any], node_id: Any) -> Optional[Any]:
    if node_id in map_obj:
        return map_obj.get(node_id)
    s = str(node_id)
    if s in map_obj:
        return map_obj.get(s)
    return None


def _build_nodes_by_id(tree: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    out: Dict[Any, Dict[str, Any]] = {}
    for node in tree.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        nid = _coerce_node_id(node.get("node_id"))
        out[nid] = node
        out[str(nid)] = node
    return out


def _format_path_for_node(
    *,
    node_id: Any,
    nodes_by_id: Dict[Any, Dict[str, Any]],
    root_id: Any,
) -> str:
    parts: list[str] = []
    seen: set[Any] = set()
    cur = node_id
    for _ in range(256):
        if cur in seen:
            break
        seen.add(cur)
        node = nodes_by_id.get(cur) or nodes_by_id.get(str(cur))
        if not node:
            parts.append(str(cur))
            break
        name = (node.get("name") or str(cur)).strip() or str(cur)
        parts.append(name)
        if cur == root_id:
            break
        parent = node.get("parent_id")
        if parent is None:
            break
        cur = _coerce_node_id(parent)
    return ".".join(reversed(parts))


def build_gt_from_tree_and_tooljson(
    *,
    tree_path: Path,
    pred_tooljson_path: Path,
    pcap_path: Path,
    pcap_layer: str = "transport_payload",
    pcap_ip_proto: Optional[int] = None,
    frame_index: Optional[Sequence[int]] = None,
    out_dir: Path,
    require_full_parse: bool = True,
    emit_tail_data_field: bool = False,
    max_packets: int = 0,
    target_message_type: Optional[str] = None,
) -> Tuple[Path, Path, TreeGTBuildStats]:
    """
    Build GT (level1/level2) by parsing each packet payload with DynamicTreeInterpreter.

    This uses Stage B (traffic-fixed) protocol tree as the oracle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if not isinstance(tree, dict):
        raise ValueError(f"Invalid tree JSON at {tree_path}")
    # Some Stage B artifacts (e.g., outputs/stage_b/format_tree.json) wrap the actual protocol
    # tree under a top-level `protocol_tree` key. The traffic-fixed trees are typically
    # unwrapped already. Support both to keep semantic validation usable for ablations.
    if isinstance(tree.get("protocol_tree"), dict):
        tree = tree["protocol_tree"]
    tree = normalize_protocol_tree(tree)

    nodes_by_id = _build_nodes_by_id(tree)
    root_id = _coerce_node_id(tree.get("root_node_id", 0))

    pred_packets = json.loads(pred_tooljson_path.read_text(encoding="utf-8"))
    if not isinstance(pred_packets, list):
        pred_packets = []

    interpreter = DynamicTreeInterpreter(tree, target_message_type=target_message_type)

    stats = TreeGTBuildStats()
    report_path = out_dir / "semantic_gt_build_report.json"
    parse_fail_errs: Counter[str] = Counter()
    parse_fail_samples: list[dict[str, Any]] = []
    no_hex_frames_sample: list[int] = []
    parse_fail_ctx_none = 0
    parse_fail_partial_ctx = 0
    limit = int(max_packets) if max_packets and max_packets > 0 else 0
    eval_frames: List[int] = []
    seen: set[int] = set()

    if frame_index is not None:
        for raw in frame_index:
            try:
                frame_no = int(raw)
            except Exception:
                continue
            if frame_no <= 0 or frame_no in seen:
                continue
            seen.add(frame_no)
            eval_frames.append(frame_no)
        eval_frames.sort()
    else:
        for idx, pkt in enumerate(pred_packets, start=1):
            if not isinstance(pkt, dict):
                continue
            packet_no = pkt.get("packet_no")
            try:
                frame_no = int(packet_no) if packet_no is not None else idx
            except Exception:
                frame_no = idx
            if frame_no <= 0 or frame_no in seen:
                continue
            seen.add(frame_no)
            eval_frames.append(frame_no)

    layer = str(pcap_layer or "").strip().lower()
    if limit and eval_frames and layer not in {"ip_payload", "l3_packet"}:
        eval_frames = eval_frames[:limit]
    if layer == "tcp_header":
        payload_by_frame = _extract_tcp_headers_from_pcap(pcap_path, eval_frames)
    elif layer == "ip_payload":
        payload_by_frame = _extract_ip_payloads_from_pcap(pcap_path, eval_frames, ip_proto=pcap_ip_proto)
        eval_frames = sorted(payload_by_frame.keys())
        if limit and eval_frames:
            eval_frames = eval_frames[:limit]
            payload_by_frame = {int(fid): payload_by_frame[int(fid)] for fid in eval_frames if int(fid) in payload_by_frame}
    elif layer == "l3_packet":
        payload_by_frame = _extract_l3_packets_from_pcap(pcap_path, eval_frames)
        eval_frames = sorted(payload_by_frame.keys())
        if limit and eval_frames:
            eval_frames = eval_frames[:limit]
            payload_by_frame = {int(fid): payload_by_frame[int(fid)] for fid in eval_frames if int(fid) in payload_by_frame}
    else:
        payload_by_frame = _extract_tcp_udp_payloads_from_pcap(pcap_path, eval_frames)

    stats.packets_total = len(eval_frames)

    allowed_node_ids: Optional[set[Any]] = None
    if layer == "tcp_header":
        # When validating TCP itself, we extract only the TCP header(+options) bytes.
        # In this mode, restrict oracle fields to the header subtree to avoid expecting
        # application payload ("Data") fields that are not present in the extracted bytes.
        header_root_ids: list[Any] = []
        for node in tree.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            if _coerce_node_id(node.get("parent_id")) != root_id:
                continue
            if str(node.get("node_type") or "").strip().lower() != "header":
                continue
            header_root_ids.append(_coerce_node_id(node.get("node_id")))

        if header_root_ids:
            allowed: set[Any] = set()
            stack = list(header_root_ids)
            while stack:
                nid = stack.pop()
                if nid in allowed:
                    continue
                allowed.add(nid)
                node = nodes_by_id.get(nid) or nodes_by_id.get(str(nid))
                if not node:
                    continue
                for child_id in node.get("children_ids") or []:
                    stack.append(_coerce_node_id(child_id))
            allowed_node_ids = allowed

    gt_packets_l1: list[dict] = []
    gt_packets_l2: list[dict] = []

    for frame_no in eval_frames:
        payload_full = payload_by_frame.get(frame_no)
        if payload_full is None:
            stats.packets_skipped_no_hex += 1
            if len(no_hex_frames_sample) < 10:
                no_hex_frames_sample.append(int(frame_no))
            continue

        payload = payload_full
        if frame_index is None:
            # Oracle input is ALWAYS bytes extracted from the PCAP.
            # By default this is the TCP/UDP payload ("transport_payload"), but for some
            # protocols (e.g., validating TCP itself) we may extract the TCP header
            # ("tcp_header") instead.
            # When a single TCP/UDP payload contains multiple protocol units, evaluate ONLY the
            # FIRST unit (e.g., the first Modbus ADU). We infer the first-unit boundary using an
            # oracle probe (tree-based) and then parse only that prefix.
            end_bytes = _infer_first_unit_len_bytes_or_0(payload_full, interpreter, root_id=root_id)
            if 0 < end_bytes < len(payload_full):
                payload = payload_full[:end_bytes]
                # Historically this stat referred to truncation using pred tool.json fields end.
                # Oracle-based semantics now truncate to the first protocol unit length.
                stats.packets_truncated_to_pred_fields_end += 1

        success, ctx, err = interpreter.parse(payload)
        if ctx is None:
            stats.packets_skipped_parse_fail += 1
            parse_fail_ctx_none += 1
            err_str = str(err or "").strip() or "<no_error>"
            parse_fail_errs[err_str] += 1
            if len(parse_fail_samples) < 10:
                parse_fail_samples.append(
                    {
                        "frame_no": int(frame_no),
                        "skipped_because": "ctx_none",
                        "success": bool(success),
                        "payload_len": int(len(payload)),
                        "payload_hex_head": payload[:64].hex(),
                        "err": err_str[:500],
                    }
                )
            continue
        if not success and require_full_parse:
            stats.packets_skipped_parse_fail += 1
            parse_fail_partial_ctx += 1
            err_str = str(err or "").strip() or "<no_error>"
            parse_fail_errs[err_str] += 1
            if len(parse_fail_samples) < 10:
                parse_fail_samples.append(
                    {
                        "frame_no": int(frame_no),
                        "skipped_because": "require_full_parse",
                        "success": bool(success),
                        "payload_len": int(len(payload)),
                        "payload_hex_head": payload[:64].hex(),
                        "err": err_str[:500],
                    }
                )
            continue

        fields: list[dict] = []
        seen_keys: set[tuple[int, int, int, str]] = set()

        span_records = getattr(ctx, "field_spans", None)
        if isinstance(span_records, list) and span_records:
            span_iter = [
                r
                for r in span_records
                if isinstance(r, dict) and r.get("node_id") is not None and r.get("bit_start") is not None
            ]
        else:
            span_iter = []

        if span_iter:
            def _span_key(rec: Dict[str, Any]) -> Tuple[int, Any, int]:
                nid_any = _coerce_node_id(rec.get("node_id"))
                try:
                    nid_i = int(nid_any)
                    nid_key: Tuple[int, Any] = (0, nid_i)
                except Exception:
                    nid_key = (1, str(nid_any))
                try:
                    bs = int(rec.get("bit_start") or 0)
                except Exception:
                    bs = 0
                return (nid_key[0], nid_key[1], bs)

            span_iter.sort(key=_span_key)

        for raw in span_iter or [{"node_id": rid} for rid in sorted(ctx.visited_nodes, key=lambda x: str(x))]:
            nid = _coerce_node_id(raw.get("node_id"))
            if allowed_node_ids is not None and nid not in allowed_node_ids:
                continue
            node = nodes_by_id.get(nid) or nodes_by_id.get(str(nid))
            if not node:
                continue

            node_type = str(node.get("node_type") or "")
            children_ids = node.get("children_ids") or []
            is_leaf = node_type in LEAF_TYPES or not children_ids
            if not is_leaf:
                continue

            bit_start = raw.get("bit_start") if isinstance(raw, dict) and raw.get("bit_start") is not None else None
            size_bits = None
            if isinstance(raw, dict):
                size_bits = raw.get("wire_size_bits") or raw.get("size_bits")
            if bit_start is None:
                bit_start = _lookup(ctx.bit_starts, nid)
            if size_bits is None:
                size_bits = _lookup(ctx.wire_sizes, nid)
            if size_bits is None:
                size_bits = _lookup(ctx.actual_sizes, nid)
            if size_bits is None:
                size_bits = _lookup(ctx.sizes, nid)
            if bit_start is None or size_bits is None:
                continue

            try:
                bit_start_i = int(bit_start)
                size_bits_i = int(size_bits)
            except Exception:
                continue

            if bit_start_i < 0 or size_bits_i <= 0:
                continue
            end_bits = bit_start_i + size_bits_i
            if end_bits > len(payload) * 8:
                continue

            if bit_start_i % 8 != 0 or size_bits_i % 8 != 0:
                stats.fields_emitted_non_byte_aligned += 1

            try:
                value_int = int(safe_value_int_msb0(payload, bit_start_i, size_bits_i))
            except Exception:
                continue

            # Backward-compatible cover bytes (not necessarily byte-aligned).
            start_byte = bit_start_i // 8
            end_byte = int(ceil_div(end_bits, 8))
            length = end_byte - start_byte
            if length <= 0:
                continue
            if start_byte < 0 or end_byte > len(payload) or end_byte < start_byte:
                continue

            raw = payload[start_byte:end_byte].hex()
            name = str(node.get("name") or nid)
            fmt_path = _format_path_for_node(node_id=nid, nodes_by_id=nodes_by_id, root_id=root_id)
            key = (bit_start_i, size_bits_i, value_int, fmt_path)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            fields.append(
                {
                    "name": name,
                    "format_path": fmt_path,
                    "node_id": _coerce_node_id(node.get("node_id", nid)),
                    "bit_start": bit_start_i,
                    "size_bits": size_bits_i,
                    "value_int": value_int,
                    "bit_order": "MSB0",
                    "bit_offset_in_first_byte": int(bit_start_i % 8),
                    # Legacy byte-level spans (cover bytes).
                    "offset": int(start_byte),
                    "length": length,
                    "raw": raw,
                }
            )

        if emit_tail_data_field:
            total_bits = len(payload) * 8
            max_end = 0
            for f in fields:
                try:
                    end = int(f.get("bit_start", 0)) + int(f.get("size_bits", 0))
                except Exception:
                    continue
                if end > max_end:
                    max_end = end

            tail_bits = int(total_bits) - int(max_end)
            if tail_bits > 0 and max_end >= 0 and max_end + tail_bits <= total_bits:
                start_byte = int(max_end) // 8
                end_byte = int(ceil_div(total_bits, 8))
                raw_tail = payload[start_byte:end_byte].hex() if 0 <= start_byte <= end_byte <= len(payload) else ""
                try:
                    value_tail = int(safe_value_int_msb0(payload, int(max_end), int(tail_bits)))
                except Exception:
                    value_tail = 0

                fields.append(
                    {
                        "name": "Data",
                        "format_path": "__tail_data__",
                        "node_id": "__tail_data__",
                        "bit_start": int(max_end),
                        "size_bits": int(tail_bits),
                        "value_int": int(value_tail),
                        "bit_order": "MSB0",
                        "bit_offset_in_first_byte": int(int(max_end) % 8),
                        "offset": int(start_byte),
                        "length": int(end_byte - start_byte),
                        "raw": str(raw_tail),
                    }
                )

        fields.sort(key=lambda f: (int(f.get("bit_start", 0)), int(f.get("size_bits", 0)), str(f.get("name", ""))))
        stats.fields_emitted += len(fields)

        gt_packets_l1.append({"frame_no": frame_no, "fields": fields})
        gt_packets_l2.append({"frame_no": frame_no, "fields": fields})
        stats.packets_used += 1

    gt_l1_path = out_dir / "semantic_level1_gt.json"
    gt_l2_path = out_dir / "semantic_level2_gt.json"

    gt_l1_payload = {
        "pcap": str(pcap_path),
        "gt_level": 1,
        "description": "GT derived from Stage B traffic_fixed_tree via DynamicTreeInterpreter",
        "packets": gt_packets_l1,
    }
    gt_l2_payload = {
        "pcap": str(pcap_path),
        "gt_level": 2,
        "description": "GT derived from Stage B traffic_fixed_tree via DynamicTreeInterpreter",
        "packets": gt_packets_l2,
    }

    gt_l1_path.write_text(json.dumps(gt_l1_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gt_l2_path.write_text(json.dumps(gt_l2_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        parse_fail_top = [
            {"err": err, "count": int(count)}
            for err, count in parse_fail_errs.most_common(20)
            if err and count
        ]
        report_payload = {
            "tree_path": str(tree_path),
            "pcap_path": str(pcap_path),
            "pcap_layer": str(pcap_layer),
            "pcap_ip_proto": int(pcap_ip_proto) if pcap_ip_proto is not None else None,
            "require_full_parse": bool(require_full_parse),
            "target_message_type": target_message_type,
            "packets_total": int(stats.packets_total),
            "packets_used": int(stats.packets_used),
            "packets_skipped_no_hex": int(stats.packets_skipped_no_hex),
            "packets_skipped_parse_fail": int(stats.packets_skipped_parse_fail),
            "parse_fail_ctx_none": int(parse_fail_ctx_none),
            "parse_fail_partial_ctx": int(parse_fail_partial_ctx),
            "no_hex_frames_sample": no_hex_frames_sample,
            "parse_fail_err_top": parse_fail_top,
            "parse_fail_samples": parse_fail_samples,
        }
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    return gt_l1_path, gt_l2_path, stats


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return (a + b - 1) // b


def _infer_first_unit_len_bytes_or_0(payload_full: bytes, interpreter: DynamicTreeInterpreter, *, root_id: Any) -> int:
    total = len(payload_full)
    if total <= 0:
        return 0

    # Probe with bounded prefixes so that we do not parse the whole transport payload when multiple
    # protocol units are coalesced. The oracle (tree) should be able to infer the unit length from
    # early header fields (e.g., length fields) and expose it via ctx.{sizes,wire_sizes,...}.
    probes = [64, 128, 256, 512, 1024, 2048, 4096, total]
    for n in probes:
        probe_len = min(total, int(n))
        if probe_len <= 0:
            continue
        try:
            _ok, ctx, _err = interpreter.parse(payload_full[:probe_len])
        except Exception:
            ctx = None
        if ctx is None:
            continue
        # For unit boundary inference we only trust the root node's declared/known size,
        # not "how far we managed to parse" in a truncated probe.
        end_bits = _infer_root_end_bits_or_0(ctx, root_id=root_id)
        if end_bits <= 0:
            continue
        end_bytes = _ceil_div(int(end_bits), 8)
        if end_bytes <= 0 or end_bytes > total:
            continue

        # If the inferred end is exactly the probe boundary (and we're not probing the full payload),
        # it's usually just the root fallback to `stream.total_bits` (i.e., "unknown length"), not a
        # real first-unit boundary. Returning it would incorrectly truncate protocols with large fixed
        # headers (e.g., DHCP) and make the oracle parse fail.
        if end_bytes == probe_len and probe_len < total:
            continue

        return int(end_bytes)

    return 0


def _infer_root_end_bits_or_0(ctx: Any, *, root_id: Any) -> int:
    try:
        bit_start = ctx.bit_starts.get(root_id) if getattr(ctx, "bit_starts", None) is not None else None
    except Exception:
        bit_start = None
    if not isinstance(bit_start, int):
        bit_start = 0

    size_bits = None
    try:
        if getattr(ctx, "wire_sizes", None) is not None:
            size_bits = ctx.wire_sizes.get(root_id)
    except Exception:
        size_bits = None
    if size_bits is None:
        try:
            if getattr(ctx, "actual_sizes", None) is not None:
                size_bits = ctx.actual_sizes.get(root_id)
        except Exception:
            size_bits = None
    if size_bits is None:
        try:
            if getattr(ctx, "sizes", None) is not None:
                size_bits = ctx.sizes.get(root_id)
        except Exception:
            size_bits = None

    if not isinstance(size_bits, int) or size_bits <= 0:
        return 0
    return int(bit_start) + int(size_bits)


def _extract_tcp_udp_payloads_from_pcap(pcap_path: Path, frame_nos: Sequence[int]) -> Dict[int, bytes]:
    wanted = sorted({int(n) for n in frame_nos if int(n) > 0})
    if not wanted:
        return {}
    max_frame_no = max(wanted)
    wanted_set = set(wanted)

    payload_by_frame: Dict[int, bytes] = {}
    for frame_no, payload in _iter_pcap_tcp_udp_payloads(pcap_path, max_frame_no=max_frame_no):
        if frame_no not in wanted_set:
            continue
        payload_by_frame[int(frame_no)] = payload
        if len(payload_by_frame) >= len(wanted_set):
            break
    return payload_by_frame


def _extract_tcp_headers_from_pcap(pcap_path: Path, frame_nos: Sequence[int]) -> Dict[int, bytes]:
    """Extract TCP header(+options) bytes (starting at TCP header) for selected frames."""
    wanted = sorted({int(n) for n in frame_nos if int(n) > 0})
    if not wanted:
        return {}
    max_frame_no = max(wanted)
    wanted_set = set(wanted)

    header_by_frame: Dict[int, bytes] = {}
    for frame_no, header in _iter_pcap_tcp_headers(pcap_path, max_frame_no=max_frame_no):
        if frame_no not in wanted_set:
            continue
        header_by_frame[int(frame_no)] = header
        if len(header_by_frame) >= len(wanted_set):
            break
    return header_by_frame


def _extract_ip_payloads_from_pcap(
    pcap_path: Path,
    frame_nos: Sequence[int],
    *,
    ip_proto: Optional[int] = None,
) -> Dict[int, bytes]:
    """Extract IP payload (upper-layer) bytes for selected frames."""
    wanted = sorted({int(n) for n in frame_nos if int(n) > 0})
    if not wanted:
        return {}
    max_frame_no = max(wanted)
    wanted_set = set(wanted)

    payload_by_frame: Dict[int, bytes] = {}
    for frame_no, payload in _iter_pcap_ip_payloads(pcap_path, max_frame_no=max_frame_no, ip_proto=ip_proto):
        if frame_no not in wanted_set:
            continue
        payload_by_frame[int(frame_no)] = payload
        if len(payload_by_frame) >= len(wanted_set):
            break
    return payload_by_frame


def _extract_l3_packets_from_pcap(pcap_path: Path, frame_nos: Sequence[int]) -> Dict[int, bytes]:
    """Extract L3 packet bytes (IPv4/IPv6 header+payload) for selected frames."""
    wanted = sorted({int(n) for n in frame_nos if int(n) > 0})
    if not wanted:
        return {}
    max_frame_no = max(wanted)
    wanted_set = set(wanted)

    pkt_by_frame: Dict[int, bytes] = {}
    for frame_no, pkt in _iter_pcap_l3_packets(pcap_path, max_frame_no=max_frame_no):
        if frame_no not in wanted_set:
            continue
        pkt_by_frame[int(frame_no)] = pkt
        if len(pkt_by_frame) >= len(wanted_set):
            break
    return pkt_by_frame


def extract_tcp_headers_from_pcap(pcap_path: Path, frame_nos: Sequence[int]) -> Dict[int, bytes]:
    """Extract TCP header(+options) bytes (starting at TCP header) for selected frames."""
    return _extract_tcp_headers_from_pcap(pcap_path, frame_nos)


def extract_ip_payloads_from_pcap(pcap_path: Path, frame_nos: Sequence[int], *, ip_proto: Optional[int] = None) -> Dict[int, bytes]:
    """Extract IP payload (upper-layer) bytes for selected frames."""
    return _extract_ip_payloads_from_pcap(pcap_path, frame_nos, ip_proto=ip_proto)


def extract_l3_packets_from_pcap(pcap_path: Path, frame_nos: Sequence[int]) -> Dict[int, bytes]:
    """Extract L3 packet bytes (IPv4/IPv6 datagrams) for selected frames."""
    return _extract_l3_packets_from_pcap(pcap_path, frame_nos)


def extract_tcp_udp_payloads_with_ports_from_pcap(
    pcap_path: Path, frame_nos: Sequence[int]
) -> Dict[int, Tuple[bytes, str, int, int]]:
    """
    Extract TCP/UDP payload bytes and transport ports for selected frames.

    Returns mapping:
      frame_no -> (payload_bytes, transport, sport, dport)
    where transport is "tcp" or "udp".
    """
    wanted = sorted({int(n) for n in frame_nos if int(n) > 0})
    if not wanted:
        return {}
    max_frame_no = max(wanted)
    wanted_set = set(wanted)

    payload_by_frame: Dict[int, Tuple[bytes, str, int, int]] = {}
    for frame_no, payload, transport, sport, dport in _iter_pcap_tcp_udp_payloads_with_ports(pcap_path, max_frame_no=max_frame_no):
        if frame_no not in wanted_set:
            continue
        payload_by_frame[int(frame_no)] = (payload, transport, int(sport), int(dport))
        if len(payload_by_frame) >= len(wanted_set):
            break
    return payload_by_frame


def iter_pcap_tcp_udp_payloads_with_ports(
    pcap_path: Path, *, max_frame_no: int = 0
) -> Iterable[Tuple[int, bytes, str, int, int]]:
    """Iterate TCP/UDP payload bytes with (transport, sport, dport) from a PCAP/PCAPNG."""
    yield from _iter_pcap_tcp_udp_payloads_with_ports(pcap_path, max_frame_no=max_frame_no)


def iter_pcap_tcp_headers(pcap_path: Path, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes]]:
    """Iterate TCP header(+options) bytes from a PCAP/PCAPNG."""
    yield from _iter_pcap_tcp_headers(pcap_path, max_frame_no=max_frame_no)


def iter_pcap_ip_payloads(
    pcap_path: Path, *, max_frame_no: int = 0, ip_proto: Optional[int] = None
) -> Iterable[Tuple[int, bytes]]:
    """Iterate IP payload (upper-layer) bytes from a PCAP/PCAPNG."""
    yield from _iter_pcap_ip_payloads(pcap_path, max_frame_no=max_frame_no, ip_proto=ip_proto)


def iter_pcap_l3_packets(pcap_path: Path, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes]]:
    """Iterate L3 packet bytes (IPv4/IPv6 datagrams) from a PCAP/PCAPNG."""
    yield from _iter_pcap_l3_packets(pcap_path, max_frame_no=max_frame_no)


def _pcap_frame_ipv4_fragment_payload(
    frame: bytes, linktype: int, *, ip_proto: Optional[int] = None
) -> Optional[Tuple[Tuple[bytes, bytes, int, int], int, bool, bytes]]:
    """Return IPv4 fragment tuple: ((src,dst,proto,ident), offset_bytes, more, fragment_payload)."""
    ip_pkt = _extract_l3_packet(frame, linktype)
    if ip_pkt is None or not ip_pkt:
        return None
    if len(ip_pkt) < 20 or ((ip_pkt[0] >> 4) & 0xF) != 4:
        return None

    ver_ihl = ip_pkt[0]
    ihl = int(ver_ihl & 0x0F) * 4
    if ihl < 20 or len(ip_pkt) < ihl:
        return None

    try:
        total_len = int(struct.unpack("!H", ip_pkt[2:4])[0])
    except Exception:
        total_len = len(ip_pkt)
    if total_len <= 0 or total_len > len(ip_pkt):
        total_len = len(ip_pkt)

    try:
        frag = int(struct.unpack("!H", ip_pkt[6:8])[0])
    except Exception:
        frag = 0
    off_units = int(frag & 0x1FFF)
    more = bool(frag & 0x2000)
    if off_units == 0 and not more:
        return None

    proto = int(ip_pkt[9])
    if ip_proto is not None and int(proto) != int(ip_proto):
        return None

    try:
        ident = int(struct.unpack("!H", ip_pkt[4:6])[0])
    except Exception:
        ident = 0

    src = bytes(ip_pkt[12:16])
    dst = bytes(ip_pkt[16:20])
    offset_bytes = int(off_units) * 8
    frag_payload = bytes(ip_pkt[ihl:total_len])
    key = (src, dst, int(proto), int(ident))
    return key, offset_bytes, bool(more), frag_payload


def _try_emit_ipv4_reassembly(
    buffers: Dict[Tuple[bytes, bytes, int, int], Dict[str, Any]],
    key: Tuple[bytes, bytes, int, int],
) -> Optional[Tuple[int, bytes]]:
    st = buffers.get(key)
    if not st:
        return None
    total_len = st.get("total_len")
    last_frame = st.get("last_frame")
    frags = st.get("frags")
    if not isinstance(total_len, int) or total_len <= 0:
        return None
    if not isinstance(last_frame, int) or last_frame <= 0:
        return None
    if not isinstance(frags, dict) or not frags:
        return None

    ranges: List[Tuple[int, int]] = []
    for off, chunk in frags.items():
        try:
            off_i = int(off)
        except Exception:
            continue
        if off_i < 0:
            continue
        if not isinstance(chunk, (bytes, bytearray)) or not chunk:
            continue
        end_i = off_i + len(chunk)
        if end_i <= 0:
            continue
        ranges.append((off_i, end_i))
    if not ranges:
        return None

    ranges.sort()
    cur = 0
    for start, end in ranges:
        if start > cur:
            return None
        if end > cur:
            cur = end
        if cur >= int(total_len):
            break
    if cur < int(total_len):
        return None

    out = bytearray(int(total_len))
    for off, chunk in frags.items():
        try:
            off_i = int(off)
        except Exception:
            continue
        if off_i < 0 or off_i >= int(total_len):
            continue
        if not isinstance(chunk, (bytes, bytearray)):
            continue
        part = bytes(chunk[: max(0, int(total_len) - off_i)])
        out[off_i : off_i + len(part)] = part

    del buffers[key]
    return int(last_frame), bytes(out)


def _iter_pcap_ip_payloads(
    pcap_path: Path, *, max_frame_no: int = 0, ip_proto: Optional[int] = None
) -> Iterable[Tuple[int, bytes]]:
    with pcap_path.open("rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"Invalid PCAP header (too short): {pcap_path}")
        magic = header[:4]
        if magic == _PCAP_MAGIC_PCAPNG:
            handle.seek(0)
            yield from _iter_pcapng_ip_payloads(handle, max_frame_no=max_frame_no, ip_proto=ip_proto)
            return
        if magic in _PCAP_MAGIC_LE:
            endian = "<"
        elif magic in _PCAP_MAGIC_BE:
            endian = ">"
        else:
            raise ValueError(f"Unrecognized PCAP magic {magic!r}: {pcap_path}")

        try:
            linktype = int(struct.unpack(endian + "I", header[20:24])[0])
        except Exception as exc:
            raise ValueError(f"Failed to parse PCAP linktype: {pcap_path}") from exc

        frame_no = 0
        ipv4_reassembly: Dict[Tuple[bytes, bytes, int, int], Dict[str, Any]] = {}
        while True:
            if max_frame_no and frame_no >= int(max_frame_no):
                break
            rec = handle.read(16)
            if not rec:
                break
            if len(rec) < 16:
                break
            try:
                _ts_sec, _ts_sub, incl_len, _orig_len = struct.unpack(endian + "IIII", rec)
            except Exception:
                break
            if incl_len <= 0:
                frame_no += 1
                continue
            frame = handle.read(int(incl_len))
            if len(frame) < int(incl_len):
                break

            frame_no += 1
            frag = _pcap_frame_ipv4_fragment_payload(frame, linktype, ip_proto=ip_proto)
            if frag is not None:
                key, offset_bytes, more, frag_payload = frag
                st = ipv4_reassembly.get(key)
                if st is None:
                    st = {"frags": {}, "total_len": None, "last_frame": None}
                    ipv4_reassembly[key] = st
                frags = st.get("frags")
                if not isinstance(frags, dict):
                    frags = {}
                    st["frags"] = frags
                prev = frags.get(int(offset_bytes))
                if prev is None or (isinstance(prev, (bytes, bytearray)) and len(frag_payload) > len(prev)):
                    frags[int(offset_bytes)] = frag_payload
                if not bool(more):
                    st["total_len"] = int(offset_bytes) + int(len(frag_payload))
                    st["last_frame"] = int(frame_no)
                emitted = _try_emit_ipv4_reassembly(ipv4_reassembly, key)
                if emitted is not None:
                    yield emitted
                continue

            payload = _pcap_frame_ip_payload(frame, linktype, ip_proto=ip_proto)
            if payload is None:
                continue
            yield frame_no, payload


def _iter_pcap_l3_packets(pcap_path: Path, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes]]:
    with pcap_path.open("rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"Invalid PCAP header (too short): {pcap_path}")
        magic = header[:4]
        if magic == _PCAP_MAGIC_PCAPNG:
            handle.seek(0)
            yield from _iter_pcapng_l3_packets(handle, max_frame_no=max_frame_no)
            return
        if magic in _PCAP_MAGIC_LE:
            endian = "<"
        elif magic in _PCAP_MAGIC_BE:
            endian = ">"
        else:
            raise ValueError(f"Unrecognized PCAP magic {magic!r}: {pcap_path}")

        try:
            linktype = int(struct.unpack(endian + "I", header[20:24])[0])
        except Exception as exc:
            raise ValueError(f"Failed to parse PCAP linktype: {pcap_path}") from exc

        frame_no = 0
        while True:
            if max_frame_no and frame_no >= int(max_frame_no):
                break
            rec = handle.read(16)
            if not rec:
                break
            if len(rec) < 16:
                break
            try:
                _ts_sec, _ts_sub, incl_len, _orig_len = struct.unpack(endian + "IIII", rec)
            except Exception:
                break
            if incl_len <= 0:
                frame_no += 1
                continue
            frame = handle.read(int(incl_len))
            if len(frame) < int(incl_len):
                break

            frame_no += 1
            pkt = _pcap_frame_l3_packet(frame, linktype)
            if pkt is None:
                continue
            yield frame_no, pkt


def _iter_pcap_tcp_udp_payloads(pcap_path: Path, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes]]:
    with pcap_path.open("rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"Invalid PCAP header (too short): {pcap_path}")
        magic = header[:4]
        if magic == _PCAP_MAGIC_PCAPNG:
            handle.seek(0)
            yield from _iter_pcapng_tcp_udp_payloads(handle, max_frame_no=max_frame_no)
            return
        if magic in _PCAP_MAGIC_LE:
            endian = "<"
        elif magic in _PCAP_MAGIC_BE:
            endian = ">"
        else:
            raise ValueError(f"Unrecognized PCAP magic {magic!r}: {pcap_path}")

        try:
            linktype = int(struct.unpack(endian + "I", header[20:24])[0])
        except Exception as exc:
            raise ValueError(f"Failed to parse PCAP linktype: {pcap_path}") from exc

        frame_no = 0
        while True:
            if max_frame_no and frame_no >= int(max_frame_no):
                break
            rec = handle.read(16)
            if not rec:
                break
            if len(rec) < 16:
                break
            try:
                _ts_sec, _ts_sub, incl_len, _orig_len = struct.unpack(endian + "IIII", rec)
            except Exception:
                break
            if incl_len <= 0:
                frame_no += 1
                continue
            frame = handle.read(int(incl_len))
            if len(frame) < int(incl_len):
                break

            frame_no += 1
            payload = _pcap_frame_tcp_udp_payload(frame, linktype)
            if payload is None:
                continue
            yield frame_no, payload


def _iter_pcap_tcp_headers(pcap_path: Path, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes]]:
    with pcap_path.open("rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"Invalid PCAP header (too short): {pcap_path}")
        magic = header[:4]
        if magic == _PCAP_MAGIC_PCAPNG:
            handle.seek(0)
            yield from _iter_pcapng_tcp_headers(handle, max_frame_no=max_frame_no)
            return
        if magic in _PCAP_MAGIC_LE:
            endian = "<"
        elif magic in _PCAP_MAGIC_BE:
            endian = ">"
        else:
            raise ValueError(f"Unrecognized PCAP magic {magic!r}: {pcap_path}")

        try:
            linktype = int(struct.unpack(endian + "I", header[20:24])[0])
        except Exception as exc:
            raise ValueError(f"Failed to parse PCAP linktype: {pcap_path}") from exc

        frame_no = 0
        while True:
            if max_frame_no and frame_no >= int(max_frame_no):
                break
            rec = handle.read(16)
            if not rec:
                break
            if len(rec) < 16:
                break
            try:
                _ts_sec, _ts_sub, incl_len, _orig_len = struct.unpack(endian + "IIII", rec)
            except Exception:
                break
            if incl_len <= 0:
                frame_no += 1
                continue
            frame = handle.read(int(incl_len))
            if len(frame) < int(incl_len):
                break

            frame_no += 1
            hdr = _pcap_frame_tcp_header(frame, linktype)
            if hdr is None:
                continue
            yield frame_no, hdr


def _iter_pcap_tcp_udp_payloads_with_ports(
    pcap_path: Path, *, max_frame_no: int = 0
) -> Iterable[Tuple[int, bytes, str, int, int]]:
    with pcap_path.open("rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            raise ValueError(f"Invalid PCAP header (too short): {pcap_path}")
        magic = header[:4]
        if magic == _PCAP_MAGIC_PCAPNG:
            handle.seek(0)
            yield from _iter_pcapng_tcp_udp_payloads_with_ports(handle, max_frame_no=max_frame_no)
            return
        if magic in _PCAP_MAGIC_LE:
            endian = "<"
        elif magic in _PCAP_MAGIC_BE:
            endian = ">"
        else:
            raise ValueError(f"Unrecognized PCAP magic {magic!r}: {pcap_path}")

        try:
            linktype = int(struct.unpack(endian + "I", header[20:24])[0])
        except Exception as exc:
            raise ValueError(f"Failed to parse PCAP linktype: {pcap_path}") from exc

        frame_no = 0
        while True:
            if max_frame_no and frame_no >= int(max_frame_no):
                break
            rec = handle.read(16)
            if not rec:
                break
            if len(rec) < 16:
                break
            try:
                _ts_sec, _ts_sub, incl_len, _orig_len = struct.unpack(endian + "IIII", rec)
            except Exception:
                break
            if incl_len <= 0:
                frame_no += 1
                continue
            frame = handle.read(int(incl_len))
            if len(frame) < int(incl_len):
                break

            frame_no += 1
            parsed = _pcap_frame_tcp_udp_payload_and_ports(frame, linktype)
            if parsed is None:
                continue
            payload, transport, sport, dport = parsed
            yield frame_no, payload, transport, sport, dport


def _iter_pcapng_ip_payloads(
    handle: Any, *, max_frame_no: int = 0, ip_proto: Optional[int] = None
) -> Iterable[Tuple[int, bytes]]:
    # Minimal pcapng reader: supports SHB/IDB/EPB/SPB. Frame numbering follows packet blocks.
    interfaces: List[int] = []
    frame_no = 0
    endian = "<"
    ipv4_reassembly: Dict[Tuple[bytes, bytes, int, int], Dict[str, Any]] = {}

    def _read_exact(n: int) -> bytes:
        data = handle.read(int(n))
        return data if data is not None else b""

    # First block must be SHB.
    hdr = _read_exact(12)
    if len(hdr) < 12:
        raise ValueError("Invalid PCAPNG header (too short)")
    block_type = int(struct.unpack("<I", hdr[:4])[0])
    if block_type != _PCAPNG_SHB:
        raise ValueError(f"Invalid PCAPNG: first block is not SHB (type={block_type})")

    bom = hdr[8:12]
    if bom == _PCAPNG_BOM_LE:
        endian = "<"
    elif bom == _PCAPNG_BOM_BE:
        endian = ">"
    else:
        endian = "<"

    try:
        total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
    except Exception:
        total_len = 0
    if total_len < 12 or total_len % 4 != 0:
        raise ValueError(f"Invalid PCAPNG SHB total_len={total_len}")
    _ = _read_exact(total_len - 12)

    def _parse_shb(block: bytes) -> str:
        if len(block) < 12:
            return "<"
        bom2 = block[8:12]
        if bom2 == _PCAPNG_BOM_LE:
            return "<"
        if bom2 == _PCAPNG_BOM_BE:
            return ">"
        return "<"

    while True:
        if max_frame_no and frame_no >= int(max_frame_no):
            break
        hdr = _read_exact(8)
        if not hdr:
            break
        if len(hdr) < 8:
            break
        block_type = int(struct.unpack(endian + "I", hdr[:4])[0])
        total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
        if total_len < 12 or total_len % 4 != 0:
            break
        body = _read_exact(total_len - 8)
        if len(body) < total_len - 8:
            break
        block = hdr + body

        if block_type == _PCAPNG_SHB:
            endian = _parse_shb(block)
            continue

        if block_type == _PCAPNG_IDB:
            if len(block) >= 8 + 8:
                try:
                    linktype = int(struct.unpack(endian + "H", block[8:10])[0])
                    interfaces.append(linktype)
                except Exception:
                    continue
            continue

        if block_type == _PCAPNG_EPB:
            if len(block) < 8 + 20 + 4:
                continue
            try:
                iface_id = int(struct.unpack(endian + "I", block[8:12])[0])
                cap_len = int(struct.unpack(endian + "I", block[20:24])[0])
            except Exception:
                iface_id = -1
                cap_len = 0

            pkt_off = 8 + 20
            pkt_end = pkt_off + max(0, int(cap_len))
            pkt_bytes = block[pkt_off:pkt_end] if pkt_end <= len(block) else b""
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if iface_id < 0 or iface_id >= len(interfaces):
                continue
            frag = _pcap_frame_ipv4_fragment_payload(pkt_bytes, interfaces[iface_id], ip_proto=ip_proto)
            if frag is not None:
                key, offset_bytes, more, frag_payload = frag
                st = ipv4_reassembly.get(key)
                if st is None:
                    st = {"frags": {}, "total_len": None, "last_frame": None}
                    ipv4_reassembly[key] = st
                frags = st.get("frags")
                if not isinstance(frags, dict):
                    frags = {}
                    st["frags"] = frags
                prev = frags.get(int(offset_bytes))
                if prev is None or (isinstance(prev, (bytes, bytearray)) and len(frag_payload) > len(prev)):
                    frags[int(offset_bytes)] = frag_payload
                if not bool(more):
                    st["total_len"] = int(offset_bytes) + int(len(frag_payload))
                    st["last_frame"] = int(frame_no)
                emitted = _try_emit_ipv4_reassembly(ipv4_reassembly, key)
                if emitted is not None:
                    yield emitted
                continue

            payload = _pcap_frame_ip_payload(pkt_bytes, interfaces[iface_id], ip_proto=ip_proto)
            if payload is None:
                continue
            yield frame_no, payload
            continue

        if block_type == _PCAPNG_SPB:
            if len(block) < 8 + 4 + 4:
                continue
            try:
                pkt_len = int(struct.unpack(endian + "I", block[8:12])[0])
            except Exception:
                pkt_len = 0
            pkt_off = 8 + 4
            max_cap = max(0, len(block) - (pkt_off + 4))
            cap_len = min(max_cap, max(0, int(pkt_len)))
            pkt_bytes = block[pkt_off : pkt_off + cap_len]
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if not interfaces:
                continue
            frag = _pcap_frame_ipv4_fragment_payload(pkt_bytes, interfaces[0], ip_proto=ip_proto)
            if frag is not None:
                key, offset_bytes, more, frag_payload = frag
                st = ipv4_reassembly.get(key)
                if st is None:
                    st = {"frags": {}, "total_len": None, "last_frame": None}
                    ipv4_reassembly[key] = st
                frags = st.get("frags")
                if not isinstance(frags, dict):
                    frags = {}
                    st["frags"] = frags
                prev = frags.get(int(offset_bytes))
                if prev is None or (isinstance(prev, (bytes, bytearray)) and len(frag_payload) > len(prev)):
                    frags[int(offset_bytes)] = frag_payload
                if not bool(more):
                    st["total_len"] = int(offset_bytes) + int(len(frag_payload))
                    st["last_frame"] = int(frame_no)
                emitted = _try_emit_ipv4_reassembly(ipv4_reassembly, key)
                if emitted is not None:
                    yield emitted
                continue

            payload = _pcap_frame_ip_payload(pkt_bytes, interfaces[0], ip_proto=ip_proto)
            if payload is None:
                continue
            yield frame_no, payload
            continue

        continue


def _iter_pcapng_l3_packets(handle: Any, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes]]:
    # Minimal pcapng reader: supports SHB/IDB/EPB/SPB. Frame numbering follows packet blocks.
    interfaces: List[int] = []
    frame_no = 0
    endian = "<"

    def _read_exact(n: int) -> bytes:
        data = handle.read(int(n))
        return data if data is not None else b""

    # First block must be SHB.
    hdr = _read_exact(12)
    if len(hdr) < 12:
        raise ValueError("Invalid PCAPNG header (too short)")
    block_type = int(struct.unpack("<I", hdr[:4])[0])
    if block_type != _PCAPNG_SHB:
        raise ValueError(f"Invalid PCAPNG: first block is not SHB (type={block_type})")

    bom = hdr[8:12]
    if bom == _PCAPNG_BOM_LE:
        endian = "<"
    elif bom == _PCAPNG_BOM_BE:
        endian = ">"
    else:
        endian = "<"

    try:
        total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
    except Exception:
        total_len = 0
    if total_len < 12 or total_len % 4 != 0:
        raise ValueError(f"Invalid PCAPNG SHB total_len={total_len}")
    _ = _read_exact(total_len - 12)

    def _parse_shb(block: bytes) -> str:
        if len(block) < 12:
            return "<"
        bom2 = block[8:12]
        if bom2 == _PCAPNG_BOM_LE:
            return "<"
        if bom2 == _PCAPNG_BOM_BE:
            return ">"
        return "<"

    while True:
        if max_frame_no and frame_no >= int(max_frame_no):
            break
        hdr = _read_exact(8)
        if not hdr:
            break
        if len(hdr) < 8:
            break
        block_type = int(struct.unpack(endian + "I", hdr[:4])[0])
        total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
        if total_len < 12 or total_len % 4 != 0:
            break
        body = _read_exact(total_len - 8)
        if len(body) < total_len - 8:
            break
        block = hdr + body

        if block_type == _PCAPNG_SHB:
            endian = _parse_shb(block)
            continue

        if block_type == _PCAPNG_IDB:
            if len(block) >= 8 + 8:
                try:
                    linktype = int(struct.unpack(endian + "H", block[8:10])[0])
                    interfaces.append(linktype)
                except Exception:
                    continue
            continue

        if block_type == _PCAPNG_EPB:
            if len(block) < 8 + 20 + 4:
                continue
            try:
                iface_id = int(struct.unpack(endian + "I", block[8:12])[0])
                cap_len = int(struct.unpack(endian + "I", block[20:24])[0])
            except Exception:
                iface_id = -1
                cap_len = 0

            pkt_off = 8 + 20
            pkt_end = pkt_off + max(0, int(cap_len))
            pkt_bytes = block[pkt_off:pkt_end] if pkt_end <= len(block) else b""
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if iface_id < 0 or iface_id >= len(interfaces):
                continue
            pkt = _pcap_frame_l3_packet(pkt_bytes, interfaces[iface_id])
            if pkt is None:
                continue
            yield frame_no, pkt
            continue

        if block_type == _PCAPNG_SPB:
            if len(block) < 8 + 4 + 4:
                continue
            try:
                pkt_len = int(struct.unpack(endian + "I", block[8:12])[0])
            except Exception:
                pkt_len = 0
            pkt_off = 8 + 4
            max_cap = max(0, len(block) - (pkt_off + 4))
            cap_len = min(max_cap, max(0, int(pkt_len)))
            pkt_bytes = block[pkt_off : pkt_off + cap_len]
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if not interfaces:
                continue
            pkt = _pcap_frame_l3_packet(pkt_bytes, interfaces[0])
            if pkt is None:
                continue
            yield frame_no, pkt
            continue

        continue


def _iter_pcapng_tcp_udp_payloads(handle: Any, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes]]:
    # Minimal pcapng reader: supports SHB/IDB/EPB/SPB. Frame numbering follows packet blocks.
    interfaces: List[int] = []
    frame_no = 0
    endian = "<"

    def _read_exact(n: int) -> bytes:
        data = handle.read(int(n))
        return data if data is not None else b""

    # First block must be SHB.
    hdr = _read_exact(12)
    if len(hdr) < 12:
        raise ValueError("Invalid PCAPNG header (too short)")
    block_type = int(struct.unpack("<I", hdr[:4])[0])
    if block_type != _PCAPNG_SHB:
        raise ValueError(f"Invalid PCAPNG: first block is not SHB (type={block_type})")

    bom = hdr[8:12]
    if bom == _PCAPNG_BOM_LE:
        endian = "<"
    elif bom == _PCAPNG_BOM_BE:
        endian = ">"
    else:
        # Fall back to little-endian (common) but keep best-effort parsing.
        endian = "<"

    total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
    if total_len < 12 or total_len % 4 != 0:
        raise ValueError(f"Invalid PCAPNG SHB total_len={total_len}")
    rest = _read_exact(total_len - 12)
    if len(rest) < total_len - 12:
        raise ValueError("Truncated PCAPNG SHB block")

    def _parse_shb(block: bytes) -> str:
        if len(block) < 12:
            return "<"
        bom2 = block[8:12]
        if bom2 == _PCAPNG_BOM_LE:
            return "<"
        if bom2 == _PCAPNG_BOM_BE:
            return ">"
        return "<"

    while True:
        if max_frame_no and frame_no >= int(max_frame_no):
            break
        hdr = _read_exact(8)
        if not hdr:
            break
        if len(hdr) < 8:
            break
        block_type = int(struct.unpack(endian + "I", hdr[:4])[0])
        total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
        if total_len < 12 or total_len % 4 != 0:
            break
        body = _read_exact(total_len - 8)
        if len(body) < total_len - 8:
            break
        block = hdr + body

        # Section header can appear again; update endianness.
        if block_type == _PCAPNG_SHB:
            endian = _parse_shb(block)
            continue

        if block_type == _PCAPNG_IDB:
            # Interface Description Block: linktype (u16), reserved (u16), snaplen (u32)
            if len(block) >= 8 + 8:
                try:
                    linktype = int(struct.unpack(endian + "H", block[8:10])[0])
                    interfaces.append(linktype)
                except Exception:
                    continue
            continue

        if block_type == _PCAPNG_EPB:
            # Enhanced Packet Block.
            if len(block) < 8 + 20 + 4:
                continue
            try:
                iface_id = int(struct.unpack(endian + "I", block[8:12])[0])
                cap_len = int(struct.unpack(endian + "I", block[20:24])[0])
            except Exception:
                iface_id = -1
                cap_len = 0

            pkt_off = 8 + 20
            pkt_end = pkt_off + max(0, int(cap_len))
            pkt_bytes = block[pkt_off:pkt_end] if pkt_end <= len(block) else b""
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if iface_id < 0 or iface_id >= len(interfaces):
                continue
            payload = _pcap_frame_tcp_udp_payload(pkt_bytes, interfaces[iface_id])
            if payload is None:
                continue
            yield frame_no, payload
            continue

        if block_type == _PCAPNG_SPB:
            # Simple Packet Block.
            if len(block) < 8 + 4 + 4:
                continue
            try:
                pkt_len = int(struct.unpack(endian + "I", block[8:12])[0])
            except Exception:
                pkt_len = 0
            pkt_off = 8 + 4
            # payload bytes are padded to 32-bit; total_len already includes padding and trailer.
            max_cap = max(0, len(block) - (pkt_off + 4))
            cap_len = min(max_cap, max(0, int(pkt_len)))
            pkt_bytes = block[pkt_off : pkt_off + cap_len]
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if not interfaces:
                continue
            payload = _pcap_frame_tcp_udp_payload(pkt_bytes, interfaces[0])
            if payload is None:
                continue
            yield frame_no, payload
            continue

        # Ignore other block types.
        continue


def _iter_pcapng_tcp_headers(handle: Any, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes]]:
    # Minimal pcapng reader: supports SHB/IDB/EPB/SPB. Frame numbering follows packet blocks.
    interfaces: List[int] = []
    frame_no = 0
    endian = "<"

    def _read_exact(n: int) -> bytes:
        data = handle.read(int(n))
        return data if data is not None else b""

    # First block must be SHB.
    hdr = _read_exact(12)
    if len(hdr) < 12:
        raise ValueError("Invalid PCAPNG header (too short)")
    block_type = int(struct.unpack("<I", hdr[:4])[0])
    if block_type != _PCAPNG_SHB:
        raise ValueError(f"Invalid PCAPNG: first block is not SHB (type={block_type})")

    bom = hdr[8:12]
    if bom == _PCAPNG_BOM_LE:
        endian = "<"
    elif bom == _PCAPNG_BOM_BE:
        endian = ">"
    else:
        endian = "<"

    total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
    if total_len < 12 or total_len % 4 != 0:
        raise ValueError(f"Invalid PCAPNG SHB total_len={total_len}")
    rest = _read_exact(total_len - 12)
    if len(rest) < total_len - 12:
        raise ValueError("Truncated PCAPNG SHB block")

    def _parse_shb(block: bytes) -> str:
        if len(block) < 12:
            return "<"
        bom2 = block[8:12]
        if bom2 == _PCAPNG_BOM_LE:
            return "<"
        if bom2 == _PCAPNG_BOM_BE:
            return ">"
        return "<"

    while True:
        if max_frame_no and frame_no >= int(max_frame_no):
            break
        hdr = _read_exact(8)
        if not hdr:
            break
        if len(hdr) < 8:
            break
        block_type = int(struct.unpack(endian + "I", hdr[:4])[0])
        total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
        if total_len < 12 or total_len % 4 != 0:
            break
        body = _read_exact(total_len - 8)
        if len(body) < total_len - 8:
            break
        block = hdr + body

        if block_type == _PCAPNG_SHB:
            endian = _parse_shb(block)
            continue

        if block_type == _PCAPNG_IDB:
            if len(block) >= 8 + 8:
                try:
                    linktype = int(struct.unpack(endian + "H", block[8:10])[0])
                    interfaces.append(linktype)
                except Exception:
                    continue
            continue

        if block_type == _PCAPNG_EPB:
            if len(block) < 8 + 20 + 4:
                continue
            try:
                iface_id = int(struct.unpack(endian + "I", block[8:12])[0])
                cap_len = int(struct.unpack(endian + "I", block[20:24])[0])
            except Exception:
                iface_id = -1
                cap_len = 0

            pkt_off = 8 + 20
            pkt_end = pkt_off + max(0, int(cap_len))
            pkt_bytes = block[pkt_off:pkt_end] if pkt_end <= len(block) else b""
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if iface_id < 0 or iface_id >= len(interfaces):
                continue
            hdr_bytes = _pcap_frame_tcp_header(pkt_bytes, interfaces[iface_id])
            if hdr_bytes is None:
                continue
            yield frame_no, hdr_bytes
            continue

        if block_type == _PCAPNG_SPB:
            if len(block) < 8 + 4 + 4:
                continue
            try:
                pkt_len = int(struct.unpack(endian + "I", block[8:12])[0])
            except Exception:
                pkt_len = 0
            pkt_off = 8 + 4
            max_cap = max(0, len(block) - (pkt_off + 4))
            cap_len = min(max_cap, max(0, int(pkt_len)))
            pkt_bytes = block[pkt_off : pkt_off + cap_len]
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if not interfaces:
                continue
            hdr_bytes = _pcap_frame_tcp_header(pkt_bytes, interfaces[0])
            if hdr_bytes is None:
                continue
            yield frame_no, hdr_bytes
            continue

        continue


def _iter_pcapng_tcp_udp_payloads_with_ports(
    handle: Any, *, max_frame_no: int = 0
) -> Iterable[Tuple[int, bytes, str, int, int]]:
    # Minimal pcapng reader: supports SHB/IDB/EPB/SPB. Frame numbering follows packet blocks.
    interfaces: List[int] = []
    frame_no = 0
    endian = "<"

    def _read_exact(n: int) -> bytes:
        data = handle.read(int(n))
        return data if data is not None else b""

    # First block must be SHB.
    hdr = _read_exact(12)
    if len(hdr) < 12:
        raise ValueError("Invalid PCAPNG header (too short)")
    block_type = int(struct.unpack("<I", hdr[:4])[0])
    if block_type != _PCAPNG_SHB:
        raise ValueError(f"Invalid PCAPNG: first block is not SHB (type={block_type})")

    bom = hdr[8:12]
    if bom == _PCAPNG_BOM_LE:
        endian = "<"
    elif bom == _PCAPNG_BOM_BE:
        endian = ">"
    else:
        endian = "<"

    total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
    if total_len < 12 or total_len % 4 != 0:
        raise ValueError(f"Invalid PCAPNG SHB total_len={total_len}")
    rest = _read_exact(total_len - 12)
    if len(rest) < total_len - 12:
        raise ValueError("Truncated PCAPNG SHB block")

    def _parse_shb(block: bytes) -> str:
        if len(block) < 12:
            return "<"
        bom2 = block[8:12]
        if bom2 == _PCAPNG_BOM_LE:
            return "<"
        if bom2 == _PCAPNG_BOM_BE:
            return ">"
        return "<"

    while True:
        if max_frame_no and frame_no >= int(max_frame_no):
            break
        hdr = _read_exact(8)
        if not hdr:
            break
        if len(hdr) < 8:
            break
        block_type = int(struct.unpack(endian + "I", hdr[:4])[0])
        total_len = int(struct.unpack(endian + "I", hdr[4:8])[0])
        if total_len < 12 or total_len % 4 != 0:
            break
        body = _read_exact(total_len - 8)
        if len(body) < total_len - 8:
            break
        block = hdr + body

        if block_type == _PCAPNG_SHB:
            endian = _parse_shb(block)
            continue

        if block_type == _PCAPNG_IDB:
            if len(block) >= 8 + 8:
                try:
                    linktype = int(struct.unpack(endian + "H", block[8:10])[0])
                    interfaces.append(linktype)
                except Exception:
                    continue
            continue

        if block_type == _PCAPNG_EPB:
            if len(block) < 8 + 20 + 4:
                continue
            try:
                iface_id = int(struct.unpack(endian + "I", block[8:12])[0])
                cap_len = int(struct.unpack(endian + "I", block[20:24])[0])
            except Exception:
                iface_id = -1
                cap_len = 0

            pkt_off = 8 + 20
            pkt_end = pkt_off + max(0, int(cap_len))
            pkt_bytes = block[pkt_off:pkt_end] if pkt_end <= len(block) else b""
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if iface_id < 0 or iface_id >= len(interfaces):
                continue
            parsed = _pcap_frame_tcp_udp_payload_and_ports(pkt_bytes, interfaces[iface_id])
            if parsed is None:
                continue
            payload, transport, sport, dport = parsed
            yield frame_no, payload, transport, sport, dport
            continue

        if block_type == _PCAPNG_SPB:
            if len(block) < 8 + 4 + 4:
                continue
            try:
                pkt_len = int(struct.unpack(endian + "I", block[8:12])[0])
            except Exception:
                pkt_len = 0
            pkt_off = 8 + 4
            max_cap = max(0, len(block) - (pkt_off + 4))
            cap_len = min(max_cap, max(0, int(pkt_len)))
            pkt_bytes = block[pkt_off : pkt_off + cap_len]
            frame_no += 1
            if max_frame_no and frame_no > int(max_frame_no):
                break
            if not interfaces:
                continue
            parsed = _pcap_frame_tcp_udp_payload_and_ports(pkt_bytes, interfaces[0])
            if parsed is None:
                continue
            payload, transport, sport, dport = parsed
            yield frame_no, payload, transport, sport, dport
            continue

        continue


def _pcap_frame_tcp_udp_payload(frame: bytes, linktype: int) -> Optional[bytes]:
    ip_pkt = _extract_l3_packet(frame, linktype)
    if ip_pkt is None:
        return None
    if not ip_pkt:
        return b""
    ver = (ip_pkt[0] >> 4) & 0xF
    if ver == 4:
        return _ipv4_tcp_udp_payload(ip_pkt)
    if ver == 6:
        return _ipv6_tcp_udp_payload(ip_pkt)
    return None


def _pcap_frame_ip_payload(frame: bytes, linktype: int, *, ip_proto: Optional[int] = None) -> Optional[bytes]:
    ip_pkt = _extract_l3_packet(frame, linktype)
    if ip_pkt is None:
        return None
    if not ip_pkt:
        return b""
    ver = (ip_pkt[0] >> 4) & 0xF
    if ver == 4:
        return _ipv4_ip_payload(ip_pkt, ip_proto=ip_proto)
    if ver == 6:
        return _ipv6_ip_payload(ip_pkt, ip_proto=ip_proto)
    return None


def _clip_ip_datagram_len(ip_pkt: bytes) -> bytes:
    """Best-effort trim of IP datagram bytes to the declared header length.

    This strips common L2 padding artifacts (e.g., Ethernet minimum-frame padding).
    """
    if not ip_pkt:
        return ip_pkt
    ver = (ip_pkt[0] >> 4) & 0xF
    if ver == 4:
        if len(ip_pkt) < 4:
            return ip_pkt
        try:
            total_len = int(struct.unpack("!H", ip_pkt[2:4])[0])
        except Exception:
            total_len = 0
        if 0 < total_len <= len(ip_pkt):
            return ip_pkt[: int(total_len)]
        return ip_pkt
    if ver == 6:
        if len(ip_pkt) < 6:
            return ip_pkt
        try:
            payload_len = int(struct.unpack("!H", ip_pkt[4:6])[0])
        except Exception:
            payload_len = 0
        total_len = 40 + int(payload_len)
        if 40 <= total_len <= len(ip_pkt):
            return ip_pkt[: int(total_len)]
        return ip_pkt
    return ip_pkt


def _pcap_frame_l3_packet(frame: bytes, linktype: int) -> Optional[bytes]:
    """Return IPv4/IPv6 datagram bytes (header+payload) clipped to declared length."""
    ip_pkt = _extract_l3_packet(frame, linktype)
    if ip_pkt is None:
        return None
    if not ip_pkt:
        return b""
    ver = (ip_pkt[0] >> 4) & 0xF
    if ver not in {4, 6}:
        return None
    return _clip_ip_datagram_len(ip_pkt)


def _pcap_frame_tcp_header(frame: bytes, linktype: int) -> Optional[bytes]:
    ip_pkt = _extract_l3_packet(frame, linktype)
    if ip_pkt is None:
        return None
    if not ip_pkt:
        return b""
    ver = (ip_pkt[0] >> 4) & 0xF
    if ver == 4:
        return _ipv4_tcp_header(ip_pkt)
    if ver == 6:
        return _ipv6_tcp_header(ip_pkt)
    return None


def _pcap_frame_tcp_udp_payload_and_ports(frame: bytes, linktype: int) -> Optional[Tuple[bytes, str, int, int]]:
    ip_pkt = _extract_l3_packet(frame, linktype)
    if ip_pkt is None:
        return None
    if not ip_pkt:
        return (b"", "tcp", 0, 0)
    ver = (ip_pkt[0] >> 4) & 0xF
    if ver == 4:
        return _ipv4_tcp_udp_payload_and_ports(ip_pkt)
    if ver == 6:
        return _ipv6_tcp_udp_payload_and_ports(ip_pkt)
    return None


def _extract_l3_packet(frame: bytes, linktype: int) -> Optional[bytes]:
    if linktype == _DLT_RAW:
        return frame
    if linktype == _DLT_IPV4:
        return frame if frame and ((frame[0] >> 4) & 0xF) == 4 else None
    if linktype == _DLT_IPV6:
        return frame if frame and ((frame[0] >> 4) & 0xF) == 6 else None
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
        return frame[off:]
    if linktype == _DLT_LINUX_SLL:
        # Linux cooked capture v1: 16-byte header, protocol at offset 14.
        if len(frame) < 16:
            return None
        proto = int(struct.unpack("!H", frame[14:16])[0])
        if proto not in {_ETHERTYPE_IPV4, _ETHERTYPE_IPV6}:
            return None
        return frame[16:]
    if linktype == _DLT_LINUX_SLL2:
        # Linux cooked capture v2: 20-byte header, protocol at offset 0.
        if len(frame) < 20:
            return None
        proto = int(struct.unpack("!H", frame[0:2])[0])
        if proto not in {_ETHERTYPE_IPV4, _ETHERTYPE_IPV6}:
            return None
        return frame[20:]
    return None


def _ipv4_tcp_udp_payload(ip_pkt: bytes) -> Optional[bytes]:
    if len(ip_pkt) < 20:
        return None
    ver_ihl = ip_pkt[0]
    if ((ver_ihl >> 4) & 0xF) != 4:
        return None
    ihl = int(ver_ihl & 0x0F) * 4
    if ihl < 20 or len(ip_pkt) < ihl:
        return None

    try:
        total_len = int(struct.unpack("!H", ip_pkt[2:4])[0])
    except Exception:
        total_len = len(ip_pkt)
    if total_len <= 0 or total_len > len(ip_pkt):
        total_len = len(ip_pkt)

    # Reject fragmented IPv4 packets (no reassembly).
    try:
        frag = int(struct.unpack("!H", ip_pkt[6:8])[0])
    except Exception:
        frag = 0
    if (frag & 0x1FFF) != 0 or (frag & 0x2000) != 0:
        return None

    proto = int(ip_pkt[9])
    l4 = ip_pkt[ihl:total_len]
    if proto == _IPPROTO_TCP:
        return _tcp_payload(l4)
    if proto == _IPPROTO_UDP:
        return _udp_payload(l4)
    return None


def _ipv4_ip_payload(ip_pkt: bytes, *, ip_proto: Optional[int] = None) -> Optional[bytes]:
    if len(ip_pkt) < 20:
        return None
    ver_ihl = ip_pkt[0]
    if ((ver_ihl >> 4) & 0xF) != 4:
        return None
    ihl = int(ver_ihl & 0x0F) * 4
    if ihl < 20 or len(ip_pkt) < ihl:
        return None

    try:
        total_len = int(struct.unpack("!H", ip_pkt[2:4])[0])
    except Exception:
        total_len = len(ip_pkt)
    if total_len <= 0 or total_len > len(ip_pkt):
        total_len = len(ip_pkt)

    # Reject fragmented IPv4 packets (no reassembly).
    try:
        frag = int(struct.unpack("!H", ip_pkt[6:8])[0])
    except Exception:
        frag = 0
    if (frag & 0x1FFF) != 0 or (frag & 0x2000) != 0:
        return None

    proto = int(ip_pkt[9])
    if ip_proto is not None and int(proto) != int(ip_proto):
        return None

    return ip_pkt[ihl:total_len]


def _ipv4_tcp_header(ip_pkt: bytes) -> Optional[bytes]:
    if len(ip_pkt) < 20:
        return None
    ver_ihl = ip_pkt[0]
    if ((ver_ihl >> 4) & 0xF) != 4:
        return None
    ihl = int(ver_ihl & 0x0F) * 4
    if ihl < 20 or len(ip_pkt) < ihl:
        return None

    try:
        total_len = int(struct.unpack("!H", ip_pkt[2:4])[0])
    except Exception:
        total_len = len(ip_pkt)
    if total_len <= 0 or total_len > len(ip_pkt):
        total_len = len(ip_pkt)

    # Reject fragmented IPv4 packets (no reassembly).
    try:
        frag = int(struct.unpack("!H", ip_pkt[6:8])[0])
    except Exception:
        frag = 0
    if (frag & 0x1FFF) != 0 or (frag & 0x2000) != 0:
        return None

    proto = int(ip_pkt[9])
    if proto != _IPPROTO_TCP:
        return None
    l4 = ip_pkt[ihl:total_len]
    return _tcp_header(l4)


def _ipv4_tcp_udp_payload_and_ports(ip_pkt: bytes) -> Optional[Tuple[bytes, str, int, int]]:
    if len(ip_pkt) < 20:
        return None
    ver_ihl = ip_pkt[0]
    if ((ver_ihl >> 4) & 0xF) != 4:
        return None
    ihl = int(ver_ihl & 0x0F) * 4
    if ihl < 20 or len(ip_pkt) < ihl:
        return None

    try:
        total_len = int(struct.unpack("!H", ip_pkt[2:4])[0])
    except Exception:
        total_len = len(ip_pkt)
    if total_len <= 0 or total_len > len(ip_pkt):
        total_len = len(ip_pkt)

    # Reject fragmented IPv4 packets (no reassembly).
    try:
        frag = int(struct.unpack("!H", ip_pkt[6:8])[0])
    except Exception:
        frag = 0
    if (frag & 0x1FFF) != 0 or (frag & 0x2000) != 0:
        return None

    proto = int(ip_pkt[9])
    l4 = ip_pkt[ihl:total_len]
    if proto == _IPPROTO_TCP:
        parsed = _tcp_payload_and_ports(l4)
        if parsed is None:
            return None
        payload, sport, dport = parsed
        return payload, "tcp", sport, dport
    if proto == _IPPROTO_UDP:
        parsed = _udp_payload_and_ports(l4)
        if parsed is None:
            return None
        payload, sport, dport = parsed
        return payload, "udp", sport, dport
    return None


def _ipv6_tcp_udp_payload(ip_pkt: bytes) -> Optional[bytes]:
    if len(ip_pkt) < 40:
        return None
    if ((ip_pkt[0] >> 4) & 0xF) != 6:
        return None
    try:
        payload_len = int(struct.unpack("!H", ip_pkt[4:6])[0])
    except Exception:
        payload_len = len(ip_pkt) - 40
    next_header = int(ip_pkt[6])
    off = 40
    end = min(len(ip_pkt), off + max(0, int(payload_len)))

    for _ in range(16):
        if next_header in {_IPPROTO_TCP, _IPPROTO_UDP}:
            break
        if next_header in {0, 43, 60}:  # Hop-by-Hop, Routing, Destination Options
            if off + 2 > end:
                return None
            nh = int(ip_pkt[off])
            hdr_len = (int(ip_pkt[off + 1]) + 1) * 8
            if hdr_len <= 0 or off + hdr_len > end:
                return None
            next_header = nh
            off += hdr_len
            continue
        if next_header == 44:  # Fragment
            if off + 8 > end:
                return None
            # Only accept the first fragment with no more fragments.
            frag_off_res_m = int(struct.unpack("!H", ip_pkt[off + 2 : off + 4])[0])
            frag_off = (frag_off_res_m >> 3) & 0x1FFF
            more = frag_off_res_m & 0x1
            if frag_off != 0 or more != 0:
                return None
            next_header = int(ip_pkt[off])
            off += 8
            continue
        if next_header == 51:  # AH
            if off + 2 > end:
                return None
            nh = int(ip_pkt[off])
            hdr_len = (int(ip_pkt[off + 1]) + 2) * 4
            if hdr_len <= 0 or off + hdr_len > end:
                return None
            next_header = nh
            off += hdr_len
            continue
        # ESP or unknown extensions: unsupported.
        return None

    l4 = ip_pkt[off:end]
    if next_header == _IPPROTO_TCP:
        return _tcp_payload(l4)
    if next_header == _IPPROTO_UDP:
        return _udp_payload(l4)
    return None


def _ipv6_ip_payload(ip_pkt: bytes, *, ip_proto: Optional[int] = None) -> Optional[bytes]:
    if len(ip_pkt) < 40:
        return None
    if ((ip_pkt[0] >> 4) & 0xF) != 6:
        return None
    try:
        payload_len = int(struct.unpack("!H", ip_pkt[4:6])[0])
    except Exception:
        payload_len = len(ip_pkt) - 40
    next_header = int(ip_pkt[6])
    off = 40
    end = min(len(ip_pkt), off + max(0, int(payload_len)))

    for _ in range(16):
        if ip_proto is not None and next_header == int(ip_proto):
            break
        if next_header in {0, 43, 60}:  # Hop-by-Hop, Routing, Destination Options
            if off + 2 > end:
                return None
            nh = int(ip_pkt[off])
            hdr_len = (int(ip_pkt[off + 1]) + 1) * 8
            if hdr_len <= 0 or off + hdr_len > end:
                return None
            next_header = nh
            off += hdr_len
            continue
        if next_header == 44:  # Fragment
            if off + 8 > end:
                return None
            # Only accept the first fragment with no more fragments.
            frag_off_res_m = int(struct.unpack("!H", ip_pkt[off + 2 : off + 4])[0])
            frag_off = (frag_off_res_m >> 3) & 0x1FFF
            more = frag_off_res_m & 0x1
            if frag_off != 0 or more != 0:
                return None
            next_header = int(ip_pkt[off])
            off += 8
            continue
        if next_header == 51:  # AH
            if off + 2 > end:
                return None
            nh = int(ip_pkt[off])
            hdr_len = (int(ip_pkt[off + 1]) + 2) * 4
            if hdr_len <= 0 or off + hdr_len > end:
                return None
            next_header = nh
            off += hdr_len
            continue
        if ip_proto is None:
            break
        return None

    if ip_proto is not None and next_header != int(ip_proto):
        return None
    return ip_pkt[off:end]


def _ipv6_tcp_header(ip_pkt: bytes) -> Optional[bytes]:
    if len(ip_pkt) < 40:
        return None
    if ((ip_pkt[0] >> 4) & 0xF) != 6:
        return None
    try:
        payload_len = int(struct.unpack("!H", ip_pkt[4:6])[0])
    except Exception:
        payload_len = len(ip_pkt) - 40
    next_header = int(ip_pkt[6])
    off = 40
    end = min(len(ip_pkt), off + max(0, int(payload_len)))

    for _ in range(16):
        if next_header in {_IPPROTO_TCP, _IPPROTO_UDP}:
            break
        if next_header in {0, 43, 60}:  # Hop-by-Hop, Routing, Destination Options
            if off + 2 > end:
                return None
            nh = int(ip_pkt[off])
            hdr_len = (int(ip_pkt[off + 1]) + 1) * 8
            if hdr_len <= 0 or off + hdr_len > end:
                return None
            next_header = nh
            off += hdr_len
            continue
        if next_header == 44:  # Fragment
            if off + 8 > end:
                return None
            frag_off_res_m = int(struct.unpack("!H", ip_pkt[off + 2 : off + 4])[0])
            frag_off = (frag_off_res_m >> 3) & 0x1FFF
            more = frag_off_res_m & 0x1
            if frag_off != 0 or more != 0:
                return None
            next_header = int(ip_pkt[off])
            off += 8
            continue
        if next_header == 51:  # AH
            if off + 2 > end:
                return None
            nh = int(ip_pkt[off])
            hdr_len = (int(ip_pkt[off + 1]) + 2) * 4
            if hdr_len <= 0 or off + hdr_len > end:
                return None
            next_header = nh
            off += hdr_len
            continue
        return None

    if next_header != _IPPROTO_TCP:
        return None
    l4 = ip_pkt[off:end]
    return _tcp_header(l4)


def _ipv6_tcp_udp_payload_and_ports(ip_pkt: bytes) -> Optional[Tuple[bytes, str, int, int]]:
    if len(ip_pkt) < 40:
        return None
    if ((ip_pkt[0] >> 4) & 0xF) != 6:
        return None
    try:
        payload_len = int(struct.unpack("!H", ip_pkt[4:6])[0])
    except Exception:
        payload_len = len(ip_pkt) - 40
    next_header = int(ip_pkt[6])
    off = 40
    end = min(len(ip_pkt), off + max(0, int(payload_len)))

    for _ in range(16):
        if next_header in {_IPPROTO_TCP, _IPPROTO_UDP}:
            break
        if next_header in {0, 43, 60}:  # Hop-by-Hop, Routing, Destination Options
            if off + 2 > end:
                return None
            nh = int(ip_pkt[off])
            hdr_len = (int(ip_pkt[off + 1]) + 1) * 8
            if hdr_len <= 0 or off + hdr_len > end:
                return None
            next_header = nh
            off += hdr_len
            continue
        if next_header == 44:  # Fragment
            if off + 8 > end:
                return None
            frag_off_res_m = int(struct.unpack("!H", ip_pkt[off + 2 : off + 4])[0])
            frag_off = (frag_off_res_m >> 3) & 0x1FFF
            more = frag_off_res_m & 0x1
            if frag_off != 0 or more != 0:
                return None
            next_header = int(ip_pkt[off])
            off += 8
            continue
        if next_header == 51:  # AH
            if off + 2 > end:
                return None
            nh = int(ip_pkt[off])
            hdr_len = (int(ip_pkt[off + 1]) + 2) * 4
            if hdr_len <= 0 or off + hdr_len > end:
                return None
            next_header = nh
            off += hdr_len
            continue
        return None

    l4 = ip_pkt[off:end]
    if next_header == _IPPROTO_TCP:
        parsed = _tcp_payload_and_ports(l4)
        if parsed is None:
            return None
        payload, sport, dport = parsed
        return payload, "tcp", sport, dport
    if next_header == _IPPROTO_UDP:
        parsed = _udp_payload_and_ports(l4)
        if parsed is None:
            return None
        payload, sport, dport = parsed
        return payload, "udp", sport, dport
    return None


def _tcp_payload(seg: bytes) -> Optional[bytes]:
    if len(seg) < 20:
        return None
    data_offset = (seg[12] >> 4) & 0xF
    hdr_len = int(data_offset) * 4
    if hdr_len < 20 or hdr_len > len(seg):
        return None
    return seg[hdr_len:]


def _tcp_header(seg: bytes) -> Optional[bytes]:
    if len(seg) < 20:
        return None
    data_offset = (seg[12] >> 4) & 0xF
    hdr_len = int(data_offset) * 4
    if hdr_len < 20:
        hdr_len = 20
    if hdr_len > len(seg):
        hdr_len = len(seg)
    return seg[:hdr_len]


def _tcp_payload_and_ports(seg: bytes) -> Optional[Tuple[bytes, int, int]]:
    if len(seg) < 20:
        return None
    try:
        sport = int(struct.unpack("!H", seg[0:2])[0])
        dport = int(struct.unpack("!H", seg[2:4])[0])
    except Exception:
        return None
    data_offset = (seg[12] >> 4) & 0xF
    hdr_len = int(data_offset) * 4
    if hdr_len < 20 or hdr_len > len(seg):
        return None
    return seg[hdr_len:], sport, dport


def _udp_payload(seg: bytes) -> Optional[bytes]:
    if len(seg) < 8:
        return None
    try:
        udp_len = int(struct.unpack("!H", seg[4:6])[0])
    except Exception:
        udp_len = len(seg)
    if udp_len < 8 or udp_len > len(seg):
        udp_len = len(seg)
    return seg[8:udp_len]


def _udp_payload_and_ports(seg: bytes) -> Optional[Tuple[bytes, int, int]]:
    if len(seg) < 8:
        return None
    try:
        sport = int(struct.unpack("!H", seg[0:2])[0])
        dport = int(struct.unpack("!H", seg[2:4])[0])
    except Exception:
        return None
    try:
        udp_len = int(struct.unpack("!H", seg[4:6])[0])
    except Exception:
        udp_len = len(seg)
    if udp_len < 8 or udp_len > len(seg):
        udp_len = len(seg)
    return seg[8:udp_len], sport, dport
