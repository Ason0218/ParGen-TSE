from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PCAP_MAGIC_PCAPNG = b"\x0a\x0d\x0d\x0a"
_PCAP_MAGIC_LE = {b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"}
_PCAP_MAGIC_BE = {b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"}

# DLT values (pcap link-layer types)
_DLT_EN10MB = 1
_DLT_RAW = 101
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
    for frame_no, payload, transport, sport, dport in iter_pcap_tcp_udp_payloads_with_ports(pcap_path, max_frame_no=max_frame_no):
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


def _iter_pcapng_tcp_udp_payloads_with_ports(handle: Any, *, max_frame_no: int = 0) -> Iterable[Tuple[int, bytes, str, int, int]]:
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
    if total_len < 12:
        raise ValueError("Invalid PCAPNG SHB (bad block length)")
    _ = _read_exact(total_len - 12)

    while True:
        if max_frame_no and frame_no >= int(max_frame_no):
            break

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
            if len(body) < 8:
                continue
            linktype = int(struct.unpack(endian + "H", body[0:2])[0])
            interfaces.append(linktype)
            continue

        if btype == _PCAPNG_EPB:
            if len(body) < 20:
                continue
            if_id = int(struct.unpack(endian + "I", body[0:4])[0])
            if if_id < 0 or if_id >= len(interfaces):
                continue
            cap_len = int(struct.unpack(endian + "I", body[12:16])[0])
            pkt_off = 20
            if cap_len <= 0 or pkt_off + cap_len > len(body):
                continue
            pkt_bytes = body[pkt_off : pkt_off + cap_len]
            frame_no += 1
            parsed = _pcap_frame_tcp_udp_payload_and_ports(pkt_bytes, interfaces[if_id])
            if parsed is None:
                continue
            payload, transport, sport, dport = parsed
            yield frame_no, payload, transport, sport, dport
            continue

        if btype == _PCAPNG_SPB:
            if len(interfaces) != 1:
                continue
            if len(body) < 4:
                continue
            cap_len = int(struct.unpack(endian + "I", body[0:4])[0])
            if cap_len <= 0 or 4 + cap_len > len(body):
                continue
            pkt_bytes = body[4 : 4 + cap_len]
            frame_no += 1
            parsed = _pcap_frame_tcp_udp_payload_and_ports(pkt_bytes, interfaces[0])
            if parsed is None:
                continue
            payload, transport, sport, dport = parsed
            yield frame_no, payload, transport, sport, dport
            continue

        continue


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
        if len(frame) < 16:
            return None
        proto = int(struct.unpack("!H", frame[14:16])[0])
        if proto not in {_ETHERTYPE_IPV4, _ETHERTYPE_IPV6}:
            return None
        return frame[16:]
    if linktype == _DLT_LINUX_SLL2:
        if len(frame) < 20:
            return None
        proto = int(struct.unpack("!H", frame[0:2])[0])
        if proto not in {_ETHERTYPE_IPV4, _ETHERTYPE_IPV6}:
            return None
        return frame[20:]
    return None


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

