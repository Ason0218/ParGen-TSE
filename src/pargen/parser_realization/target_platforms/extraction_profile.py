from __future__ import annotations

from typing import Any, Mapping, Optional


_KNOWN_HOST_KEYS = ("scapy", "wireshark_c", "wireshark_lua", "zeek_spicy")


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    try:
        return int(str(value).strip(), 0)
    except Exception:
        return None


def _get_extraction_section(codegen_profile: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    try:
        extraction = (codegen_profile or {}).get("extraction", {})
    except Exception:
        extraction = {}
    return extraction if isinstance(extraction, Mapping) else {}


def _get_host_section(codegen_profile: Optional[Mapping[str, Any]], host: Optional[str]) -> Mapping[str, Any]:
    if not host:
        return {}
    section = (codegen_profile or {}).get(str(host), {})
    return section if isinstance(section, Mapping) else {}


def _get_registration_section(codegen_profile: Optional[Mapping[str, Any]], host: Optional[str]) -> Mapping[str, Any]:
    section = _get_host_section(codegen_profile, host)
    reg = section.get("registration", {}) if isinstance(section, Mapping) else {}
    return reg if isinstance(reg, Mapping) else {}


def infer_ip_proto(codegen_profile: Optional[Mapping[str, Any]], *, host: Optional[str] = None) -> Optional[int]:
    """
    Infer IP protocol number (IPv4 proto / IPv6 next-header) from a codegen profile.

    Priority:
      1) explicit `extraction.ip_proto`
      2) host.registration.selector_type == "ip_proto" + host.registration.selector
      3) scan other host sections for selector_type == "ip_proto"
    """
    extraction = _get_extraction_section(codegen_profile)
    proto = _as_int(extraction.get("ip_proto"))
    if proto is not None and 0 <= proto <= 255:
        return int(proto)

    reg = _get_registration_section(codegen_profile, host)
    if reg:
        if str(reg.get("selector_type") or "").strip().lower() == "ip_proto":
            proto = _as_int(reg.get("selector"))
            if proto is not None and 0 <= proto <= 255:
                return int(proto)
        # Scapy profiles sometimes encode this as `transport: ip` + `proto`.
        proto = _as_int(reg.get("ip_proto") or reg.get("proto"))
        if proto is not None and 0 <= proto <= 255:
            return int(proto)

    for key in _KNOWN_HOST_KEYS:
        if key == host:
            continue
        reg2 = _get_registration_section(codegen_profile, key)
        if not reg2:
            continue
        if str(reg2.get("selector_type") or "").strip().lower() != "ip_proto":
            continue
        proto = _as_int(reg2.get("selector"))
        if proto is not None and 0 <= proto <= 255:
            return int(proto)

    return None


def infer_pcap_layer(codegen_profile: Optional[Mapping[str, Any]], *, host: Optional[str] = None) -> str:
    """
    Infer which bytes in the dataset PCAP correspond to the protocol-under-test.

    Supported values:
      - transport_payload: TCP/UDP payload bytes (default)
      - tcp_header: TCP header (+options) bytes
      - ip_payload: IP payload bytes (e.g., ICMP over IPv4/IPv6)
      - l3_packet: L3 packet bytes (e.g., IPv4 datagram under Ethernet)
    """
    extraction = _get_extraction_section(codegen_profile)
    raw = extraction.get("pcap_layer")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()

    # IP protocols (e.g., ICMP) are carried as the IP payload, not TCP/UDP payload.
    if infer_ip_proto(codegen_profile, host=host) is not None:
        return "ip_payload"

    reg = _get_registration_section(codegen_profile, host)
    transport = str(reg.get("transport") or "").strip().lower()
    if transport in {"ip", "ipv4", "ipv6", "l3"}:
        return "ip_payload"

    selector_type = str(reg.get("selector_type") or "").strip().lower()
    dissector_table = str(reg.get("dissector_table") or "").strip().lower()
    if selector_type == "ethertype" or dissector_table == "ethertype" or transport in {"ethernet", "eth", "l2"}:
        return "l3_packet"

    # Scan other host sections when the requested host doesn't provide enough info.
    for key in _KNOWN_HOST_KEYS:
        if key == host:
            continue
        reg2 = _get_registration_section(codegen_profile, key)
        if not reg2:
            continue
        selector_type2 = str(reg2.get("selector_type") or "").strip().lower()
        dissector_table2 = str(reg2.get("dissector_table") or "").strip().lower()
        transport2 = str(reg2.get("transport") or "").strip().lower()
        if selector_type2 == "ethertype" or dissector_table2 == "ethertype" or transport2 in {"ethernet", "eth", "l2"}:
            return "l3_packet"

    return "transport_payload"
