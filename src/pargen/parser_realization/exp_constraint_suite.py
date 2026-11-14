from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from pargen.paths import DATA_DIR
from .conformance_observations import POLICY as CONFORMANCE_SCORING_POLICY

_SCAPY_EXP_CONSTRAINT_PLUGIN_CACHE: Dict[Tuple[str, str], str] = {}


def _tshark_supports_lua(tshark: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(tshark), "-X", "lua_script:/dev/null", "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    return "not built with support for Lua scripting" not in text


@dataclass(frozen=True)
class ConstraintExpectation:
    constraint_id: str
    patterns_any: Tuple[str, ...]
    uncovered_reason: Optional[str] = None


@dataclass(frozen=True)
class CaseExpectation:
    case: str
    pcap: Path
    frame_no: int
    constraints: Tuple[ConstraintExpectation, ...]


@dataclass(frozen=True)
class CaseResult:
    case: str
    ok: bool
    reason: str
    pcap: str
    frame_no: int
    target_constraints: Tuple[str, ...]
    anomaly_detected: Optional[bool] = None
    target_diagnostic_matched: Optional[bool] = None
    independently_verified: bool = False
    observations: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SuiteResult:
    protocol: str
    host: str
    mode: str
    success: int
    fail: int
    total: int
    coverage: float
    suite_dir: str
    mapping_path: str
    results: Tuple[CaseResult, ...]
    # Optional binary-mode confusion matrix over (seed normal, mutated negative) samples.
    positive_results: Tuple[CaseResult, ...] = ()
    tp: Optional[int] = None
    fp: Optional[int] = None
    tn: Optional[int] = None
    fn: Optional[int] = None
    positive_total: Optional[int] = None
    negative_total: Optional[int] = None
    scoring_policy: str = CONFORMANCE_SCORING_POLICY
    acceptance_semantics: str = "configured_gate"
    confusion_semantics: str = "binary_detection_runtime_failures_are_non_detection"
    negative_detection_success: int = 0
    negative_target_diagnostic_success: int = 0
    negative_independently_verified: int = 0
    negative_isolation_unverified: int = 0


@dataclass(frozen=True)
class ZeekBuiltinDatasetParseSummary:
    protocol: str
    host: str
    pcap: str
    frames_indexed: int
    filter_name: str
    timeout_s: float
    zeek_bin: str
    tshark_bin: str
    pcap_filtered: bool
    logs_dir: Optional[str]
    zeek_rc: int
    parse_ok: bool
    reason: str
    weird_rows: int
    log_rows: int
    duration_s: float


@dataclass(frozen=True)
class ScapyBuiltinDatasetParseSummary:
    protocol: str
    host: str
    pcap: str
    frames_indexed: int
    filter_name: str
    parse_ok: int
    parse_fail: int
    errors: int
    duration_s: float
    error_samples: Tuple[Dict[str, Any], ...]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_suite_dir(protocol: str) -> Path:
    return DATA_DIR / protocol / "inputs" / "exp_constraint"


def _resolve_zeek_bin() -> Path:
    raw_bin = str(os.getenv("PARGEN_ZEEK_BIN") or "").strip()
    if raw_bin:
        zeek = Path(raw_bin).expanduser()
    else:
        found = shutil.which("zeek")
        zeek = Path(found) if found else None

    if zeek is None or not zeek.exists():
        raise RuntimeError("zeek not found on PATH; set PARGEN_ZEEK_BIN=/path/to/zeek")
    return zeek


def _resolve_zeek_timeout_s(default: float = 60.0) -> float:
    raw_timeout = os.getenv("PARGEN_ZEEK_TIMEOUT_S")
    if raw_timeout is None:
        return float(default)
    try:
        return float(str(raw_timeout).strip())
    except Exception:
        return float(default)


def _resolve_tshark_exp_constraint_timeout_s(default: float = 30.0) -> float:
    raw_timeout = os.getenv("PARGEN_TSHARK_EXP_CONSTRAINT_TIMEOUT_S")
    if raw_timeout is None:
        raw_timeout = os.getenv("PARGEN_WIRESHARK_EXP_CONSTRAINT_TIMEOUT_S")
    if raw_timeout is None:
        return float(default)
    try:
        return float(str(raw_timeout).strip())
    except Exception:
        return float(default)


def _resolve_tshark_bin(tshark_path: Optional[Path] = None) -> Path:
    if tshark_path is not None:
        cand = Path(tshark_path).expanduser()
        if cand.exists():
            return cand
    bundled = DATA_DIR / "tools" / "wireshark-4.4.11" / "build" / "run" / "tshark"
    if bundled.exists():
        return bundled
    found = shutil.which("tshark")
    if found:
        return Path(found)
    raise RuntimeError("tshark not found; install wireshark or pass --tshark")


def _log_row_count(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    try:
        cnt = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                cnt += 1
        return cnt
    except Exception:
        return 0


def _default_mapping_path(*, protocol: str, suite_dir: Path) -> Optional[Path]:
    # Prefer suite root; fall back to suite_dir/samples (some datasets store the
    # mapping alongside the generated negative pcaps).
    for root in (suite_dir, suite_dir / "samples"):
        cand = root / f"{protocol}_negpcaps_constraints_map.json"
        if cand.exists():
            return cand
        alt = sorted(root.glob("*_negpcaps_constraints_map.json"))
        if len(alt) == 1:
            return alt[0]
        alt2 = sorted(root.glob("*constraints*map*.json"))
        if len(alt2) == 1:
            return alt2[0]
    return None


def _iter_case_names(mapping: Mapping[str, Any]) -> List[str]:
    suite = mapping.get("suite") if isinstance(mapping, Mapping) else None
    if isinstance(suite, Mapping):
        suite_cases = suite.get("cases")
        if isinstance(suite_cases, list) and suite_cases:
            return [str(x) for x in suite_cases if str(x).strip()]

    cases = mapping.get("cases") if isinstance(mapping, Mapping) else None
    if isinstance(cases, Mapping):
        return sorted([str(k) for k in cases.keys() if str(k).strip()])
    if isinstance(cases, list):
        out: List[str] = []
        for item in cases:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return sorted(out)
    return []


def _coerce_constraints_catalog(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Support multiple exp_constraint suite schemas.

    Legacy schema:
      - constraints_catalog: {constraint_id -> spec}

    Lightweight hand-audited schema:
      - constraints: {constraint_id -> spec}

    Mutator-derived schema (from real traffic):
      - constraints_from_tree: {constraint_id -> spec}
      - derived_constraints:   {constraint_id -> spec}
    """
    raw = mapping.get("constraints_catalog")
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}

    catalog: Dict[str, Any] = {}
    lightweight = mapping.get("constraints")
    if isinstance(lightweight, Mapping):
        for cid, spec in lightweight.items():
            if isinstance(spec, Mapping):
                spec_dict = dict(spec)
                spec_dict.setdefault("present_in_tree", True)
                catalog[str(cid)] = spec_dict
            else:
                catalog[str(cid)] = {"description": str(spec), "present_in_tree": True}

    from_tree = mapping.get("constraints_from_tree")
    if isinstance(from_tree, Mapping):
        for cid, spec in from_tree.items():
            if not isinstance(spec, Mapping):
                continue
            spec_dict: Dict[str, Any] = dict(spec)
            spec_dict.setdefault("present_in_tree", True)
            kind = str(spec_dict.get("kind") or "")
            if kind == "node":
                spec_dict["kind"] = "node_constraint"
            elif kind == "edge":
                spec_dict["kind"] = "edge_constraint"
            catalog[str(cid)] = spec_dict

    derived = mapping.get("derived_constraints")
    if isinstance(derived, Mapping):
        for cid, spec in derived.items():
            if not isinstance(spec, Mapping):
                continue
            spec_dict = dict(spec)
            spec_dict.setdefault("present_in_tree", False)
            catalog[str(cid)] = spec_dict

    return catalog


def _mapping_disable_protocols(mapping: Mapping[str, Any]) -> List[str]:
    raw: Any = (
        mapping.get("disable_builtin_protocols")
        or mapping.get("disable_builtin_protocol")
        or mapping.get("disable_protocols")
        or mapping.get("disable_protocol")
    )
    tshark_cfg = mapping.get("tshark")
    if raw is None and isinstance(tshark_cfg, Mapping):
        raw = (
            tshark_cfg.get("disable_builtin_protocols")
            or tshark_cfg.get("disable_builtin_protocol")
            or tshark_cfg.get("disable_protocols")
            or tshark_cfg.get("disable_protocol")
        )
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item) for item in raw]
    else:
        values = []
    return [value.strip() for value in values if str(value or "").strip()]


def _suite_path_candidates(*, suite_dir: Path, relpath: str) -> List[Path]:
    rel = str(relpath or "").strip()
    if not rel:
        return []
    return [(suite_dir / rel), (suite_dir / "samples" / rel)]


def _resolve_case_pcap(*, suite_dir: Path, case_name: str, case_info: Mapping[str, Any]) -> Path:
    for key in ("pcap", "pcapng", "pcap_path", "pcap_file"):
        v = case_info.get(key)
        if isinstance(v, str) and v.strip():
            for cand in _suite_path_candidates(suite_dir=suite_dir, relpath=v.strip()):
                if cand.exists():
                    return cand.resolve()
            return (suite_dir / v.strip()).resolve()
    # Best-effort fallback.
    for rel in (f"{case_name}.pcap", f"{case_name}.pcapng"):
        for cand in _suite_path_candidates(suite_dir=suite_dir, relpath=rel):
            if cand.exists():
                return cand.resolve()
    return (suite_dir / f"{case_name}.pcapng").resolve()


def _resolve_base_pcap(
    *,
    protocol: str,
    suite_dir: Path,
    base_pcap: Any,
    require_suite_local: bool = False,
) -> Optional[Path]:
    """
    Resolve the seed/base (normal) pcap path from a mapping payload.

    Some mappings store an absolute path from another machine; fall back to searching
    within the current repo's DATA_DIR.
    """
    if require_suite_local:
        if not isinstance(base_pcap, str) or not base_pcap.strip():
            return None
        raw = Path(base_pcap.strip()).expanduser()
        candidates = [raw] if raw.is_absolute() else [suite_dir / raw, suite_dir / "samples" / raw]
        if raw.is_absolute():
            candidates.extend([suite_dir / raw.name, suite_dir / "samples" / raw.name])
        suite_root = suite_dir.resolve()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(suite_root)
            except Exception:
                continue
            if resolved.exists():
                return resolved
        return None

    if isinstance(base_pcap, str) and base_pcap.strip():
        raw = Path(base_pcap.strip()).expanduser()
        if raw.exists():
            return raw.resolve()
        basename = raw.name
        if basename:
            proto = str(protocol or "").strip().lower()
            candidates = [
                DATA_DIR / proto / "inputs" / "pcap" / basename,
                suite_dir / basename,
                suite_dir / "samples" / basename,
                suite_dir.parent / "pcap" / basename,
            ]
            for cand in candidates:
                try:
                    if cand.exists():
                        return cand.resolve()
                except Exception:
                    continue

    # Best-effort: pick the only pcap in inputs/pcap when unambiguous.
    proto = str(protocol or "").strip().lower()
    pcap_dir = DATA_DIR / proto / "inputs" / "pcap"
    if pcap_dir.exists():
        pcaps = sorted(list(pcap_dir.glob("*.pcap")) + list(pcap_dir.glob("*.pcapng")))
        if len(pcaps) == 1:
            return pcaps[0].resolve()
    return None


def _resolve_seed_frame_no(case_info: Mapping[str, Any]) -> Optional[int]:
    """
    Resolve the frame number of the unmutated (seed) packet for a negative case.

    Different suite schemas use different keys; return None when unavailable.
    """
    for key in ("seed_frame_no", "seed_frame", "seed_frame_index"):
        v = case_info.get(key)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            continue
    meta = case_info.get("meta")
    if isinstance(meta, Mapping):
        for key in ("seed_frame_no", "seed_frame", "seed_frame_index", "frame_no"):
            v = meta.get(key)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
    seed = case_info.get("seed")
    if isinstance(seed, Mapping):
        # Mutator-derived schemas commonly store a 0-based index of the seed packet
        # inside a nested `seed` object (e.g., BFD).
        idx = seed.get("frame_index")
        if idx is not None:
            try:
                return int(idx) + 1
            except Exception:
                pass
        for key in ("frame_no", "frame", "frame_index_1based"):
            v = seed.get(key)
            if v is None:
                continue
            try:
                return int(v)
            except Exception:
                continue
    return None


def _resolve_case_frame_no(*, suite_dir: Path, case_name: str, case_info: Mapping[str, Any]) -> int:
    meta = case_info.get("meta")
    if isinstance(meta, Mapping):
        idx = meta.get("target_packet_index")
        if idx is not None:
            try:
                return int(idx) + 1
            except Exception:
                pass
        frame_no = meta.get("frame_no")
        if frame_no is not None:
            try:
                return int(frame_no)
            except Exception:
                pass

    # Fall back to per-case JSON file.
    meta_paths: List[Path] = []
    v = case_info.get("json")
    if isinstance(v, str) and v.strip():
        meta_paths.extend(_suite_path_candidates(suite_dir=suite_dir, relpath=v.strip()))
    meta_paths.extend(_suite_path_candidates(suite_dir=suite_dir, relpath=f"{case_name}.json"))

    for meta_path in meta_paths:
        if not meta_path.exists():
            continue
        try:
            payload = _load_json(meta_path)
            if isinstance(payload, Mapping):
                idx = payload.get("target_packet_index")
                if idx is not None:
                    return int(idx) + 1
        except Exception:
            continue

    # Most negative suites store a single mutated packet per pcap.
    return 1


def _case_target_constraints(case_info: Mapping[str, Any]) -> List[str]:
    for key in ("target_constraints", "constraint_ids", "constraint_id"):
        v = case_info.get(key)
        if isinstance(v, list) and v:
            return [str(x) for x in v if str(x).strip()]
    meta = case_info.get("meta")
    if isinstance(meta, Mapping):
        v = meta.get("target_constraints")
        if isinstance(v, list) and v:
            return [str(x) for x in v if str(x).strip()]
    cid = case_info.get("constraint_id")
    if isinstance(cid, str) and cid.strip():
        return [cid.strip()]
    return []


def _host_normalize(host: str) -> str:
    h = str(host or "").strip().lower().replace("-", "_")
    if h == "zeek":
        return "zeek_spicy"
    return h


def _scapy_external_constraint_patterns(cid: str, spec: Mapping[str, Any]) -> Tuple[str, ...]:
    field = str(spec.get("field") or "")
    constraint = str(spec.get("constraint") or "")
    hay = f"{cid} {field} {constraint}".lower()
    patterns: List[str] = []

    def add(value: str) -> None:
        value = str(value or "").strip()
        if value and value not in patterns:
            patterns.append(value)

    raw_field = field.strip()
    if raw_field:
        add(raw_field)
        add(raw_field.replace(" ", "_"))
        for sep in (".", " vs ", "/", ":"):
            if sep in raw_field:
                for part in raw_field.replace(" vs ", ".").replace("/", ".").replace(":", ".").split("."):
                    part = part.strip()
                    if part:
                        add(part)
                        add(part.replace(" ", "_"))

    if "version" in hay:
        add("Version")
    if "ihl" in hay or "header length" in hay:
        add("IHL")
        add("Options must be present")
    if "total_length" in hay or "total length" in hay:
        add("Total_Length")
    if "reserved" in hay:
        add("Reserved")
    if "checksum" in hay:
        add("Checksum")
        add("checksum")
    if "captured" in hay or "capture" in hay or "available" in hay:
        add("length_relation")
        add("too small for payload")
    if "range" in hay:
        add("range")
    if "enum" in hay:
        add("enum")

    # Avoid turning a generic BS tag into a pass for every external-suite case.
    patterns = [p for p in patterns if p not in {"IPv4", "IP", "BS_CONSTRAINT"}]
    return tuple(patterns)


_SCAPY_BUILTIN_MODBUS_LAYER_TYPES: Optional[Tuple[type, ...]] = None
_SCAPY_BUILTIN_BFD_LAYER_TYPES: Optional[Tuple[type, ...]] = None
_SCAPY_BUILTIN_TCP_LAYER_TYPES: Optional[Tuple[type, ...]] = None
_SCAPY_BUILTIN_DHCP_LAYER_TYPES: Optional[Tuple[type, ...]] = None
_SCAPY_BUILTIN_IPV4_LAYER_TYPES: Optional[Tuple[type, ...]] = None
_SCAPY_BUILTIN_ICMP_LAYER_TYPES: Optional[Tuple[type, ...]] = None
_SCAPY_BUILTIN_DNS_LAYER_TYPES: Optional[Tuple[type, ...]] = None


def _scapy_builtin_modbus_layers() -> Tuple[type, ...]:
    """
    Return Scapy's official Modbus/TCP layer types.

    Scapy ships Modbus under scapy.contrib.modbus (official contrib module).
    """

    global _SCAPY_BUILTIN_MODBUS_LAYER_TYPES
    if _SCAPY_BUILTIN_MODBUS_LAYER_TYPES is not None:
        return _SCAPY_BUILTIN_MODBUS_LAYER_TYPES

    try:
        from scapy.all import load_contrib  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    load_contrib("modbus")

    try:
        from scapy.contrib.modbus import ModbusADURequest, ModbusADUResponse  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy contrib modbus is unavailable (scapy.contrib.modbus import failed)") from exc

    _SCAPY_BUILTIN_MODBUS_LAYER_TYPES = (ModbusADURequest, ModbusADUResponse)
    return _SCAPY_BUILTIN_MODBUS_LAYER_TYPES


def _scapy_builtin_bfd_layers() -> Tuple[type, ...]:
    """
    Return Scapy's official BFD control-packet layer type.

    Scapy ships BFD under scapy.contrib.bfd (official contrib module).
    """

    global _SCAPY_BUILTIN_BFD_LAYER_TYPES
    if _SCAPY_BUILTIN_BFD_LAYER_TYPES is not None:
        return _SCAPY_BUILTIN_BFD_LAYER_TYPES

    try:
        from scapy.all import load_contrib  # type: ignore[import-not-found]
        from scapy.layers.inet import UDP  # type: ignore[import-not-found]
        from scapy.packet import bind_layers  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    load_contrib("bfd")

    try:
        from scapy.contrib.bfd import BFD  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy contrib bfd is unavailable (scapy.contrib.bfd import failed)") from exc

    # Ensure BFD binds to its well-known UDP ports for `haslayer()` on packets read from PCAP.
    for port in (3784, 3785, 4784):
        try:
            bind_layers(UDP, BFD, dport=int(port))
            bind_layers(UDP, BFD, sport=int(port))
        except Exception:
            continue

    _SCAPY_BUILTIN_BFD_LAYER_TYPES = (BFD,)
    return _SCAPY_BUILTIN_BFD_LAYER_TYPES


def _scapy_builtin_tcp_layers() -> Tuple[type, ...]:
    """Return the Scapy TCP layer class used by the binary acceptance check."""

    global _SCAPY_BUILTIN_TCP_LAYER_TYPES
    if _SCAPY_BUILTIN_TCP_LAYER_TYPES is not None:
        return _SCAPY_BUILTIN_TCP_LAYER_TYPES

    try:
        from scapy.layers.inet import TCP  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    _SCAPY_BUILTIN_TCP_LAYER_TYPES = (TCP,)
    return _SCAPY_BUILTIN_TCP_LAYER_TYPES


def _scapy_builtin_ipv4_layers() -> Tuple[type, ...]:
    """Return the Scapy IPv4 layer class used by the binary acceptance check."""

    global _SCAPY_BUILTIN_IPV4_LAYER_TYPES
    if _SCAPY_BUILTIN_IPV4_LAYER_TYPES is not None:
        return _SCAPY_BUILTIN_IPV4_LAYER_TYPES

    try:
        from scapy.layers.inet import IP  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    _SCAPY_BUILTIN_IPV4_LAYER_TYPES = (IP,)
    return _SCAPY_BUILTIN_IPV4_LAYER_TYPES


def _scapy_builtin_icmp_layers() -> Tuple[type, ...]:
    """Return the Scapy ICMP layer class used by the binary acceptance check."""

    global _SCAPY_BUILTIN_ICMP_LAYER_TYPES
    if _SCAPY_BUILTIN_ICMP_LAYER_TYPES is not None:
        return _SCAPY_BUILTIN_ICMP_LAYER_TYPES

    try:
        from scapy.layers.inet import ICMP, IP  # type: ignore[import-not-found]
        from scapy.packet import bind_layers  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    # Ensure ICMP binds under IPv4 for `haslayer()` on packets read from PCAP.
    try:
        bind_layers(IP, ICMP, proto=1)
    except Exception:
        pass

    _SCAPY_BUILTIN_ICMP_LAYER_TYPES = (ICMP,)
    return _SCAPY_BUILTIN_ICMP_LAYER_TYPES


def _scapy_builtin_dhcp_layers() -> Tuple[type, ...]:
    """Return the Scapy BOOTP and DHCP layer classes used by the binary acceptance check."""

    global _SCAPY_BUILTIN_DHCP_LAYER_TYPES
    if _SCAPY_BUILTIN_DHCP_LAYER_TYPES is not None:
        return _SCAPY_BUILTIN_DHCP_LAYER_TYPES

    try:
        from scapy.layers.dhcp import BOOTP, DHCP  # type: ignore[import-not-found]
        from scapy.layers.inet import UDP  # type: ignore[import-not-found]
        from scapy.packet import bind_layers  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    # Ensure BOOTP binds to its well-known UDP ports for `haslayer()` on packets read from PCAP.
    for port in (67, 68):
        try:
            bind_layers(UDP, BOOTP, dport=int(port))
            bind_layers(UDP, BOOTP, sport=int(port))
        except Exception:
            continue

    # Ensure DHCP decodes as payload of BOOTP when possible.
    try:
        bind_layers(BOOTP, DHCP)
    except Exception:
        pass

    _SCAPY_BUILTIN_DHCP_LAYER_TYPES = (BOOTP, DHCP)
    return _SCAPY_BUILTIN_DHCP_LAYER_TYPES


def _scapy_builtin_dns_layers() -> Tuple[type, ...]:
    """Return Scapy's builtin DNS layer type and bind it to UDP/53."""

    global _SCAPY_BUILTIN_DNS_LAYER_TYPES
    if _SCAPY_BUILTIN_DNS_LAYER_TYPES is not None:
        return _SCAPY_BUILTIN_DNS_LAYER_TYPES

    try:
        from scapy.layers.dns import DNS  # type: ignore[import-not-found]
        from scapy.layers.inet import UDP  # type: ignore[import-not-found]
        from scapy.packet import bind_layers  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    for port in (53,):
        try:
            bind_layers(UDP, DNS, dport=int(port))
            bind_layers(UDP, DNS, sport=int(port))
        except Exception:
            continue

    _SCAPY_BUILTIN_DNS_LAYER_TYPES = (DNS,)
    return _SCAPY_BUILTIN_DNS_LAYER_TYPES


def _dns_payload_bytes(pkt: Any) -> Optional[bytes]:
    try:
        from scapy.layers.dns import DNS  # type: ignore[import-not-found]
        from scapy.layers.inet import UDP  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        if pkt.haslayer(DNS):  # type: ignore[arg-type]
            layer = pkt.getlayer(DNS)
            if layer is not None:
                return bytes(layer)
    except Exception:
        pass
    try:
        if pkt.haslayer(UDP):  # type: ignore[arg-type]
            udp = pkt.getlayer(UDP)
            if udp is not None and (int(getattr(udp, "sport", 0)) == 53 or int(getattr(udp, "dport", 0)) == 53):
                return bytes(udp.payload)
    except Exception:
        pass
    return None


def _dns_structural_abnormal(pkt: Any, *, value_checks: bool) -> Optional[str]:
    data = _dns_payload_bytes(pkt)
    if data is None:
        return "missing DNS layer"
    if len(data) < 12:
        return "DNS header shorter than 12 bytes"

    flags = int.from_bytes(data[2:4], "big")
    qd = int.from_bytes(data[4:6], "big")
    an = int.from_bytes(data[6:8], "big")
    ns = int.from_bytes(data[8:10], "big")
    ar = int.from_bytes(data[10:12], "big")

    def skip_name(pos: int, depth: int = 0) -> Optional[int]:
        seen: set[int] = set()
        cur = pos
        while cur < len(data) and depth < 16:
            if cur in seen:
                return None
            seen.add(cur)
            ln = data[cur]
            if ln == 0:
                return cur + 1
            if (ln & 0xC0) == 0xC0:
                if cur + 1 >= len(data):
                    return None
                ptr = ((ln & 0x3F) << 8) | data[cur + 1]
                if ptr >= len(data):
                    return None
                if skip_name(ptr, depth + 1) is None:
                    return None
                return cur + 2
            if (ln & 0xC0) != 0:
                return None
            cur += 1 + int(ln)
        return None

    cur = 12
    for _ in range(qd):
        nxt = skip_name(cur)
        if nxt is None:
            return "DNS question name invalid"
        cur = nxt
        if cur + 4 > len(data):
            return "DNS question truncated"
        cur += 4

    for _ in range(an + ns + ar):
        nxt = skip_name(cur)
        if nxt is None:
            return "DNS RR name invalid"
        cur = nxt
        if cur + 10 > len(data):
            return "DNS RR header truncated"
        rr_type = int.from_bytes(data[cur:cur + 2], "big")
        rdlen = int.from_bytes(data[cur + 8:cur + 10], "big")
        cur += 10
        if cur + rdlen > len(data):
            return "DNS RDATA truncated"
        if value_checks and rr_type == 1 and rdlen != 4:
            return "DNS A RDATA length != 4"
        if value_checks and rr_type == 28 and rdlen != 16:
            return "DNS AAAA RDATA length != 16"
        cur += rdlen

    if value_checks and ((flags >> 4) & 0x7) != 0:
        return "DNS reserved Z bits nonzero"
    return None


def _scapy_builtin_observable_abnormal(protocol: str, pkt: Any, layer_types: Sequence[type], *, policy: str) -> Optional[str]:
    """Detect incomplete or inconsistent state exposed by decoded Scapy layers."""

    proto = str(protocol or "").strip().lower()
    mode = str(policy or "observable").strip().lower().replace("-", "_")

    if proto == "dns":
        # For deployed Scapy DNS, keep the baseline strictly parser-observable:
        # Scapy exposing a DNS layer is an accept; Scapy itself does not emit
        # constraint-violation tags for reserved bits, RDLENGTH, or compression
        # pointer obligations.  Counting those here would evaluate our validator,
        # not the official Scapy parser.
        return None

    if proto == "dhcp":
        try:
            from scapy.layers.dhcp import BOOTP, DHCP  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

        if not pkt.haslayer(BOOTP):  # type: ignore[arg-type]
            return "missing BOOTP layer"
        bootp = pkt.getlayer(BOOTP)
        if bootp is None:
            return "missing BOOTP layer"

        # If Scapy cannot build the DHCP options layer, the DHCP magic/options area was not accepted.
        if not pkt.haslayer(DHCP):  # type: ignore[arg-type]
            return "missing DHCP options layer"
        dhcp = pkt.getlayer(DHCP)
        try:
            options = list(getattr(dhcp, "options", []) or [])
        except Exception:
            return "invalid DHCP options"
        for item in options:
            if isinstance(item, (bytes, bytearray)):
                return "raw DHCP option bytes"
        if mode == "structural":
            return None

        # Value-domain checks are parser-observable but stricter than Scapy's default accept/reject behavior.
        try:
            if int(getattr(bootp, "op")) not in {1, 2}:
                return "invalid BOOTP op"
        except Exception:
            return "invalid BOOTP op"
        try:
            if int(getattr(bootp, "htype")) != 1:
                return "unexpected BOOTP hardware type for Ethernet DHCP"
        except Exception:
            return "invalid BOOTP htype"
        try:
            if int(getattr(bootp, "hlen")) != 6:
                return "unexpected BOOTP hardware address length for Ethernet DHCP"
        except Exception:
            return "invalid BOOTP hlen"
        return None

    if proto == "ipv4":
        try:
            from scapy.layers.inet import IP  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

        if not pkt.haslayer(IP):  # type: ignore[arg-type]
            return "missing IPv4 layer"
        ip = pkt.getlayer(IP)
        if ip is None:
            return "missing IPv4 layer"
        try:
            if int(getattr(ip, "version")) != 4:
                return "invalid IPv4 version"
        except Exception:
            return "invalid IPv4 version"
        try:
            ihl = int(getattr(ip, "ihl"))
            if ihl < 5:
                return "IPv4 IHL < 5"
        except Exception:
            return "invalid IPv4 IHL"
        try:
            total_len = int(getattr(ip, "len"))
            header_len = int(getattr(ip, "ihl")) * 4
            actual_len = len(bytes(ip))
            if total_len < header_len:
                return "IPv4 total length shorter than header"
            if total_len > actual_len:
                return "IPv4 total length exceeds captured bytes"
        except Exception:
            pass
        if mode == "structural":
            return None
        try:
            flags = getattr(ip, "flags")
            if int(flags) & 0x4:
                return "IPv4 reserved flag set"
        except Exception:
            pass
        try:
            from scapy.utils import checksum  # type: ignore[import-not-found]
            raw = bytearray(bytes(ip)[: int(getattr(ip, "ihl")) * 4])
            if len(raw) >= 20:
                raw[10] = 0
                raw[11] = 0
                if int(getattr(ip, "chksum")) != int(checksum(bytes(raw))):
                    return "IPv4 checksum mismatch"
        except Exception:
            pass
        return None

    if proto == "tcp":
        try:
            from scapy.layers.inet import TCP  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

        if not pkt.haslayer(TCP):  # type: ignore[arg-type]
            return "missing TCP layer"
        tcp = pkt.getlayer(TCP)
        if tcp is None:
            return "missing TCP layer"
        try:
            dataofs = int(getattr(tcp, "dataofs"))
            if dataofs < 5:
                return "TCP data offset < 5"
            actual_len = len(bytes(tcp))
            if dataofs * 4 > actual_len:
                return "TCP header length exceeds captured bytes"
        except Exception:
            return "invalid TCP data offset"
        if mode == "structural":
            return None
        try:
            if int(getattr(tcp, "reserved", 0)) != 0:
                return "TCP reserved bits nonzero"
        except Exception:
            pass
        return None

    if proto == "icmp":
        try:
            from scapy.layers.inet import ICMP  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

        if not pkt.haslayer(ICMP):  # type: ignore[arg-type]
            return "missing ICMP layer"
        icmp = pkt.getlayer(ICMP)
        if icmp is None:
            return "missing ICMP layer"
        if mode == "structural":
            return None
        try:
            icmp_type = int(getattr(icmp, "type"))
            code = int(getattr(icmp, "code"))
            if icmp_type in {0, 8, 9, 10, 13, 14, 15, 16, 17, 18} and code != 0:
                return "ICMP code nonzero for fixed-code type"
        except Exception:
            pass
        try:
            from scapy.utils import checksum  # type: ignore[import-not-found]
            raw = bytearray(bytes(icmp))
            if len(raw) >= 4:
                raw[2] = 0
                raw[3] = 0
                if int(getattr(icmp, "chksum")) != int(checksum(bytes(raw))):
                    return "ICMP checksum mismatch"
        except Exception:
            pass
        return None

    if proto == "bfd":
        try:
            layers = [pkt.getlayer(t) for t in layer_types if pkt.haslayer(t)]  # type: ignore[arg-type]
        except Exception:
            return "missing BFD layer"
        bfd = next((layer for layer in layers if layer is not None), None)
        if bfd is None:
            return "missing BFD layer"
        try:
            if int(getattr(bfd, "version", 1)) != 1:
                return "invalid BFD version"
        except Exception:
            pass
        try:
            length = int(getattr(bfd, "len"))
            if length > len(bytes(bfd)):
                return "BFD length exceeds captured bytes"
        except Exception:
            pass
        return None

    # Keep Modbus in parse-only mode for Scapy's contrib parser: its contrib layer is
    # intentionally permissive and does not expose stable malformed indicators for the
    # richer MBAP/PDU constraint suite.
    return None


def _run_scapy_builtin_case_with_meta(
    *,
    protocol: str,
    filter_name: str,
    pcap: Path,
    frame_no: int,
    case_name: Optional[str] = None,
    log_path: Optional[Path] = None,
    abnormal_policy: str = "layer",
) -> Tuple[str, bool]:
    """
    Return ("", has_layer) using Scapy's builtin contrib dissector.

    Binary exp_constraint mode uses has_layer to decide normal/abnormal:
    - missing layer => abnormal
    - present layer => normal
    """

    try:
        from scapy.utils import PcapReader  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    proto = str(protocol or "").strip().lower()
    fn = str(filter_name or "").strip()
    if not proto:
        raise ValueError("protocol is required")

    if proto == "modbus":
        layer_types = _scapy_builtin_modbus_layers()
    elif proto == "bfd":
        layer_types = _scapy_builtin_bfd_layers()
    elif proto == "tcp":
        layer_types = _scapy_builtin_tcp_layers()
    elif proto == "ipv4":
        layer_types = _scapy_builtin_ipv4_layers()
    elif proto == "icmp":
        layer_types = _scapy_builtin_icmp_layers()
    elif proto == "dhcp":
        layer_types = _scapy_builtin_dhcp_layers()
    elif proto == "dns":
        layer_types = _scapy_builtin_dns_layers()
    else:
        raise ValueError(f"scapy_builtin exp_constraint runner does not support protocol={proto!r}")

    idx = max(0, int(frame_no) - 1)
    pkt: Any = None
    with PcapReader(str(pcap)) as reader:
        for i, p in enumerate(reader):
            if i == idx:
                pkt = p
                break
    if pkt is None:
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "case": str(case_name or ""),
                    "pcap": str(pcap),
                    "frame_no": int(frame_no),
                    "protocol": proto,
                    "filter_name": fn,
                    "parsed": False,
                    "reason": "missing packet at requested frame_no",
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
        return "", False

    has_layer = False
    layer_name: Optional[str] = None
    fields_primary: Dict[str, Any] = {}
    fields_secondary: Dict[str, Any] = {}
    secondary_layer_name: Optional[str] = None
    parse_error: Optional[str] = None
    abnormal_reason: Optional[str] = None

    try:
        has_layer = any(pkt.haslayer(t) for t in layer_types)  # type: ignore[arg-type]
    except Exception:
        has_layer = False
        parse_error = "haslayer failed"

    policy_norm = str(abnormal_policy or "layer").strip().lower().replace("-", "_")
    if has_layer and policy_norm in {"observable", "structural"}:
        try:
            abnormal_reason = _scapy_builtin_observable_abnormal(proto, pkt, layer_types, policy=policy_norm)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

    if has_layer:
        try:
            for t in layer_types:
                if pkt.haslayer(t):  # type: ignore[arg-type]
                    layer = pkt.getlayer(t)
                    if layer is None:
                        continue
                    layer_name = type(layer).__name__
                    try:
                        fields = getattr(layer, "fields", None)
                    except Exception:
                        fields = None
                    if isinstance(fields, dict):
                        for k, v in list(fields.items())[:100]:
                            if isinstance(v, (str, int, float, bool)) or v is None:
                                fields_primary[str(k)] = v
                            else:
                                fields_primary[str(k)] = repr(v)

                    # Protocol-specific extra logging.
                    if proto == "modbus":
                        for attr in ("transId", "protoId", "len", "unitId"):
                            try:
                                if hasattr(layer, attr):
                                    fields_secondary[attr] = getattr(layer, attr)
                            except Exception:
                                continue
                        try:
                            pdu = getattr(layer, "payload", None)
                        except Exception:
                            pdu = None
                        if pdu is not None:
                            secondary_layer_name = type(pdu).__name__
                            try:
                                pdu_fields = getattr(pdu, "fields", None)
                            except Exception:
                                pdu_fields = None
                            if isinstance(pdu_fields, dict):
                                for k, v in list(pdu_fields.items())[:100]:
                                    if isinstance(v, (str, int, float, bool)) or v is None:
                                        fields_secondary[str(k)] = v
                                    else:
                                        fields_secondary[str(k)] = repr(v)
                    break
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw = getattr(pkt, "original", None)
            except Exception:
                raw = None
            if not isinstance(raw, (bytes, bytearray)):
                try:
                    raw = bytes(pkt)
                except Exception:
                    raw = b""
            raw_hex = raw.hex() if isinstance(raw, (bytes, bytearray)) else ""
            payload = {
                "case": str(case_name or ""),
                "pcap": str(pcap),
                "frame_no": int(frame_no),
                "protocol": proto,
                "filter_name": fn,
                "has_layer": bool(has_layer),
                "layer": layer_name,
                "fields": fields_primary,
                "secondary_layer": secondary_layer_name,
                "secondary_fields": fields_secondary,
                "raw_len": int(len(raw)) if isinstance(raw, (bytes, bytearray)) else None,
                "raw_hex_prefix": raw_hex[:512],
                "error": parse_error,
                "observable_abnormal": abnormal_reason,
                "summary": (pkt.summary() if hasattr(pkt, "summary") else repr(pkt))[:2000],
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
    out_text = f"SCAPY_BUILTIN_ABNORMAL: {abnormal_reason}" if abnormal_reason else ""
    return out_text, bool(has_layer)


def _expectations_for_case(
    *,
    protocol: str,
    host: str,
    suite_dir: Path,
    case_name: str,
    case_info: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> CaseExpectation:
    pcap = _resolve_case_pcap(suite_dir=suite_dir, case_name=case_name, case_info=case_info)
    frame_no = _resolve_case_frame_no(suite_dir=suite_dir, case_name=case_name, case_info=case_info)

    targets = _case_target_constraints(case_info)
    expectations: List[ConstraintExpectation] = []
    for cid in targets:
        cid_s = str(cid)
        spec = catalog.get(cid_s)
        if cid_s.startswith("X_"):
            expectations.append(
                ConstraintExpectation(constraint_id=cid_s, patterns_any=(), uncovered_reason=f"{cid_s} (not present in tree)")
            )
            continue
        if not isinstance(spec, Mapping):
            expectations.append(
                ConstraintExpectation(constraint_id=cid_s, patterns_any=(), uncovered_reason=f"{cid_s} (missing from constraints_catalog)")
            )
            continue
        present = spec.get("present_in_tree", True)
        if str(present).lower() in {"false", "0"}:
            expectations.append(
                ConstraintExpectation(constraint_id=cid_s, patterns_any=(), uncovered_reason=f"{cid_s} (not present in tree)")
            )
            continue

        rel = spec.get("rel")
        kind = spec.get("kind")

        def _tag_prefix_for_spec() -> Optional[str]:
            exact_contract_tag = str(spec.get("contract_tag") or "").strip()
            if exact_contract_tag:
                return exact_contract_tag
            if str(kind) == "node_constraint" or (rel is None and cid_s.startswith("N")):
                node_id = spec.get("node_id")
                raw_constraint = str(spec.get("constraint") or "").strip().lower()
                clause_kind = raw_constraint.split(":", 1)[0].strip().lower() if raw_constraint else ""
                if node_id is not None and clause_kind in {"enum", "range", "complement"}:
                    return f"BS_CONSTRAINT N{int(node_id)}_{clause_kind}"
                if node_id is not None:
                    return f"BS_CONSTRAINT N{int(node_id)}"
                return None
            if str(rel) == "length_of":
                src = spec.get("src")
                dst = spec.get("dst")
                if src is not None and dst is not None:
                    return f"BS_CONSTRAINT E{int(src)}_{int(dst)}_length_of"
                return None
            if str(rel) == "condition_on":
                src = spec.get("src")
                dst = spec.get("dst")
                if src is not None and dst is not None:
                    return f"BS_CONSTRAINT E{int(src)}_{int(dst)}_condition_on"
                return None
            return None

        if bool(spec.get("strict_contract_tag")):
            exact_tag = _tag_prefix_for_spec()
            if exact_tag:
                expectations.append(
                    ConstraintExpectation(constraint_id=cid_s, patterns_any=(exact_tag,))
                )
            else:
                expectations.append(
                    ConstraintExpectation(
                        constraint_id=cid_s,
                        patterns_any=(),
                        uncovered_reason=f"{cid_s} (missing exact Stage-C contract tag)",
                    )
                )
            continue

        # Host-specific alert patterns.
        if str(kind) == "node_constraint" or (rel is None and cid_s.startswith("N")):
            # Prefer stable contract tags; fall back to legacy messages.
            tag_pat = _tag_prefix_for_spec() or f"BS_CONSTRAINT {cid_s}"
            node_name = str(spec.get("node_name") or "").strip()
            node_id = int(spec.get("node_id") or 0)
            raw_constraint = str(spec.get("constraint") or "").strip().lower()
            is_value_constraint = raw_constraint.startswith("enum:") or raw_constraint.startswith("range:")
            if not is_value_constraint:
                # Non-value constraints (e.g., complement/layout) are treated as satisfied by any
                # deterministic violation/malformed indicator in the host output.
                if host == "scapy":
                    patterns = (tag_pat, "violated")
                else:
                    patterns = (tag_pat, "Constraint violation", "Malformed Packet", "Group: Malformed")
                expectations.append(
                    ConstraintExpectation(constraint_id=cid_s, patterns_any=tuple(p for p in patterns if p))
                )
                continue
            if host == "wireshark_c":
                patterns = (tag_pat, f"Constraint violation: {node_name} (node {node_id})", f"Constraint violation: {node_name}")
            elif host == "wireshark_lua":
                patterns = (tag_pat, f"Constraint violation: {node_name}(#{node_id})", f"Constraint violation: {node_name}")
            elif host == "scapy":
                patterns = (tag_pat, f"{node_name}#{node_id}", str(node_name))
            else:
                patterns = (tag_pat, "Constraint violation", node_name)
            expectations.append(ConstraintExpectation(constraint_id=cid_s, patterns_any=tuple(p for p in patterns if p)))
            continue

        if str(rel) == "length_of":
            src_name = str(spec.get("src_name") or "").strip()
            dst_name = str(spec.get("dst_name") or "").strip()
            tag_pat = _tag_prefix_for_spec() or f"BS_CONSTRAINT {cid_s}"
            patterns = (tag_pat, f"Constraint violation: length_of {dst_name} from {src_name}")
            expectations.append(ConstraintExpectation(constraint_id=cid_s, patterns_any=patterns))
            continue

        if str(rel) == "condition_on":
            # Layout mismatch is satisfied by either our deterministic alert or Wireshark's malformed packet.
            tag_pat = _tag_prefix_for_spec() or f"BS_CONSTRAINT {cid_s}"
            patterns = (tag_pat, "missing length field", "Malformed Packet", "Group: Malformed", "Constraint violation")
            expectations.append(ConstraintExpectation(constraint_id=cid_s, patterns_any=patterns))
            continue

        if host == "scapy":
            patterns = _scapy_external_constraint_patterns(cid_s, spec) or ("Constraint violation",)
            expectations.append(ConstraintExpectation(constraint_id=cid_s, patterns_any=patterns))
        else:
            expectations.append(ConstraintExpectation(constraint_id=cid_s, patterns_any=("Constraint violation",)))

    return CaseExpectation(case=case_name, pcap=pcap, frame_no=frame_no, constraints=tuple(expectations))


def _run_tshark(
    *,
    tshark: Path,
    pcap: Path,
    frame_no: int,
    env: Mapping[str, str],
    extra_args: Sequence[str] = (),
    filter_name: str = "exp_constraint",
    ws_plugin_id: Optional[str] = None,
    isolate_generated_plugins: bool = False,
) -> Tuple[int, str]:
    from pargen.parser_realization.target_platforms.wireshark_export import (
        _stage_root_lua_scripts_for_command,
        _ws_temporarily_hide_generated_plugins,
    )

    timeout_s = _resolve_tshark_exp_constraint_timeout_s()
    with tempfile.TemporaryDirectory(prefix="pargen_exp_constraint_frame_") as td:
        selected_pcap = Path(td) / "selected_frame.pcapng"
        filter_hide_plugins = (
            _ws_temporarily_hide_generated_plugins(tshark_path=tshark, keep_plugin_id=ws_plugin_id)
            if bool(isolate_generated_plugins)
            else nullcontext()
        )
        try:
            with filter_hide_plugins:
                _filter_pcap_to_frames(
                    tshark=tshark,
                    src_pcap=pcap,
                    include_frames=[int(frame_no)],
                    dst_pcap=selected_pcap,
                )
        except Exception as exc:
            return 125, f"tshark frame filter failed: {type(exc).__name__}: {exc}"

        cmd = [str(tshark), "-r", str(selected_pcap), "-V", *list(extra_args)]
        with _stage_root_lua_scripts_for_command(tshark_path=tshark, cmd=cmd, filter_name=str(filter_name or "exp_constraint")) as staged_cmd:
            hide_plugins = (
                _ws_temporarily_hide_generated_plugins(tshark_path=tshark, keep_plugin_id=ws_plugin_id)
                if bool(isolate_generated_plugins)
                else nullcontext()
            )
            try:
                with hide_plugins:
                    proc = subprocess.run(
                        staged_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=dict(env),
                        timeout=(float(timeout_s) if timeout_s > 0 else None),
                    )
            except subprocess.TimeoutExpired as exc:
                def _coerce_timeout_text(value: Any) -> str:
                    if value is None:
                        return ""
                    if isinstance(value, bytes):
                        return value.decode("utf-8", errors="replace")
                    return str(value)

                parts = [f"tshark timeout after {float(exc.timeout or timeout_s):.1f}s"]
                stdout = _coerce_timeout_text(getattr(exc, "output", None) or getattr(exc, "stdout", None))
                stderr = _coerce_timeout_text(getattr(exc, "stderr", None))
                if stdout:
                    parts.append(stdout)
                if stderr:
                    parts.append(stderr)
                return 124, "\n".join(parts)
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return int(proc.returncode), combined


def _iter_scapy_layers(pkt: Any) -> Iterable[Any]:
    try:
        from scapy.packet import NoPayload, Packet  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return []

    seen: set[int] = set()

    def _iter_value(value: Any) -> Iterable[Any]:
        if isinstance(value, Packet):
            yield from _walk_packet(value)
            return
        if isinstance(value, Mapping):
            for nested in value.values():
                yield from _iter_value(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                yield from _iter_value(nested)

    def _walk_packet(cur: Any) -> Iterable[Any]:
        if not isinstance(cur, Packet) or isinstance(cur, NoPayload):
            return
        obj_id = id(cur)
        if obj_id in seen:
            return
        seen.add(obj_id)

        yield cur

        # Scapy generated backends may represent protocol variants as nested
        # PacketField/PacketLenField values rather than payload-chain layers.
        try:
            field_values = getattr(cur, "fields", {}) or {}
        except Exception:
            field_values = {}
        if isinstance(field_values, Mapping):
            for value in field_values.values():
                yield from _iter_value(value)

        try:
            payload = cur.payload
        except Exception:
            payload = None
        if isinstance(payload, Packet) and not isinstance(payload, NoPayload):
            yield from _walk_packet(payload)

    yield from _walk_packet(pkt)


def _ensure_scapy_plugin_loaded(
    *,
    plugin_dir: Path,
    filter_name: str,
    entry_override: Optional[Path] = None,
) -> Path:
    """
    Load the Stage C Scapy plugin once per process.

    Scapy's `bind_layers` is global and order-sensitive (first match wins). Reloading the plugin per case
    leads to multiple duplicate bindings and makes layer provenance checks brittle. For exp_constraint
    evaluation we only need the plugin registered once.
    """
    from pargen.parser_realization.target_platforms.scapy_export import resolve_scapy_entry

    entry = resolve_scapy_entry(plugin_dir, filter_name, entry_override=entry_override).resolve()
    key = (str(plugin_dir.resolve()), str(entry))
    cached = _SCAPY_EXP_CONSTRAINT_PLUGIN_CACHE.get(key)
    if cached and cached in sys.modules:
        return entry

    module_name = f"pargen_exp_constraint_scapy_{filter_name}_{uuid4().hex}"
    import importlib.util

    sys.path.insert(0, str(plugin_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, entry)
        if spec is None or spec.loader is None:
            raise ImportError(f"spec_from_file_location failed for entry: {entry}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules.pop(module_name, None)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        _scapy_fix_builtin_binding_precedence(filter_name=filter_name)
    finally:
        try:
            if sys.path and sys.path[0] == str(plugin_dir):
                sys.path.pop(0)
        except Exception:
            pass

    _SCAPY_EXP_CONSTRAINT_PLUGIN_CACHE[key] = module_name
    return entry


def _scapy_fix_builtin_binding_precedence(*, filter_name: str) -> None:
    """
    Ensure Stage C Scapy plugins take precedence over Scapy builtin layers in exp_constraint evaluation.

    Scapy's payload guessing is order-sensitive. For protocols where Scapy already has builtin dissectors
    (e.g., TCP), simply importing the generated plugin is not enough: the builtin binding may win, so the
    generated layer never appears and exp_constraint evaluation becomes meaningless.
    """

    fn = str(filter_name or "").strip().lower()
    if fn != "tcp":
        return

    try:
        from scapy.layers.inet import IP, TCP  # type: ignore[import-not-found]
        from scapy.packet import split_layers  # type: ignore[import-not-found]
    except Exception:
        return

    # The generated TCP plugin typically binds `bind_layers(IP, tcpPacket, proto=6)`.
    # Remove the builtin `IP->TCP` binding so our plugin layer can be selected.
    try:
        split_layers(IP, TCP, proto=6)  # type: ignore[arg-type]
    except Exception:
        pass


def _scapy_class_name_score(cls: Any, *, filter_name: str) -> int:
    name = "".join(ch for ch in str(getattr(cls, "__name__", "")).lower() if ch.isalnum())
    filt = "".join(ch for ch in str(filter_name or "").lower() if ch.isalnum())
    score = 0
    if filt and filt in name:
        score += 1000
    try:
        score += len(getattr(cls, "fields_desc", []) or [])
    except Exception:
        pass
    return score


def _scapy_generated_packet_classes(
    *,
    plugin_dir: Path,
    entry_path: Path,
    filter_name: str,
) -> List[Any]:
    try:
        from scapy.packet import Packet  # type: ignore[import-not-found]
    except Exception:
        return []

    plugin_root = plugin_dir.resolve()
    entry_resolved = entry_path.resolve()
    key = (str(plugin_root), str(entry_resolved))
    module_names: List[str] = []
    cached = _SCAPY_EXP_CONSTRAINT_PLUGIN_CACHE.get(key)
    if cached:
        module_names.append(cached)

    for name, mod in list(sys.modules.items()):
        if name in module_names:
            continue
        mod_file = getattr(mod, "__file__", None)
        if not isinstance(mod_file, str) or not mod_file:
            continue
        try:
            mod_path = Path(mod_file).resolve()
        except Exception:
            continue
        try:
            if mod_path == entry_resolved or plugin_root in mod_path.parents:
                module_names.append(name)
        except Exception:
            continue

    classes: List[Any] = []
    seen: set[int] = set()
    for module_name in module_names:
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        for obj in vars(mod).values():
            if not isinstance(obj, type):
                continue
            if id(obj) in seen:
                continue
            try:
                if obj is Packet or not issubclass(obj, Packet):
                    continue
            except Exception:
                continue
            try:
                if getattr(obj, "__module__", None) != getattr(mod, "__name__", None):
                    continue
            except Exception:
                pass
            seen.add(id(obj))
            classes.append(obj)

    classes.sort(key=lambda c: _scapy_class_name_score(c, filter_name=filter_name), reverse=True)
    return classes


def _scapy_protocol_byte_candidates_from_profile(
    *,
    pcap: Path,
    frame_no: int,
    codegen_profile: Optional[Dict[str, Any]],
) -> List[Tuple[str, bytes]]:
    try:
        from pargen.parser_realization.target_platforms.extraction_profile import infer_ip_proto, infer_pcap_layer
        from pargen.parser_realization.semantic_stage.gt_from_tree import (
            extract_ip_payloads_from_pcap,
            extract_l3_packets_from_pcap,
            extract_tcp_headers_from_pcap,
            extract_tcp_udp_payloads_with_ports_from_pcap,
        )
    except Exception:
        return []

    layer = infer_pcap_layer(codegen_profile, host="scapy")
    ip_proto = infer_ip_proto(codegen_profile, host="scapy") if layer == "ip_payload" else None
    frame = int(frame_no)
    out: List[Tuple[str, bytes]] = []
    try:
        if layer == "l3_packet":
            data = _scapy_relaxed_l3_packet_from_pcap(pcap=pcap, frame_no=frame)
            if not data:
                data = extract_l3_packets_from_pcap(pcap, [frame]).get(frame)
            if data:
                out.append(("profile:l3_packet", bytes(data)))
        elif layer == "ip_payload":
            data = extract_ip_payloads_from_pcap(pcap, [frame], ip_proto=ip_proto).get(frame)
            if data:
                out.append(("profile:ip_payload", bytes(data)))
        elif layer == "tcp_header":
            data = extract_tcp_headers_from_pcap(pcap, [frame]).get(frame)
            if data:
                out.append(("profile:tcp_header", bytes(data)))
        else:
            item = extract_tcp_udp_payloads_with_ports_from_pcap(pcap, [frame]).get(frame)
            if item:
                payload, transport, sport, dport = item
                out.append((f"profile:transport_payload:{transport}:{sport}->{dport}", bytes(payload)))
    except Exception:
        return out
    return out


def _scapy_relaxed_l3_packet_from_pcap(*, pcap: Path, frame_no: int) -> Optional[bytes]:
    try:
        from scapy.utils import RawPcapReader  # type: ignore[import-not-found]
        from pargen.parser_realization.traffic_exp_constraint_generator import _extract_l3_packet_with_offset
    except Exception:
        return None

    idx = max(0, int(frame_no) - 1)
    reader: Any = None
    try:
        reader = RawPcapReader(str(pcap))
        linktype = int(getattr(reader, "linktype", 0))
        for i, item in enumerate(reader):
            if i != idx:
                continue
            raw = item[0] if isinstance(item, tuple) else item
            if not isinstance(raw, (bytes, bytearray)):
                return None
            extracted = _extract_l3_packet_with_offset(bytes(raw), linktype)
            if extracted is None:
                return None
            _off, payload = extracted
            return bytes(payload)
    except Exception:
        return None
    finally:
        try:
            if reader is not None:
                reader.close()
        except Exception:
            pass
    return None


def _scapy_fallback_parse_profile_bytes(
    *,
    plugin_dir: Path,
    entry_path: Path,
    filter_name: str,
    pcap: Path,
    frame_no: int,
) -> Tuple[List[str], bool, Optional[Dict[str, Any]]]:
    codegen_profile = _load_codegen_profile_from_stage_c_meta(plugin_dir=plugin_dir)
    candidates = _scapy_protocol_byte_candidates_from_profile(
        pcap=pcap,
        frame_no=frame_no,
        codegen_profile=codegen_profile,
    )
    if not candidates:
        return [], False, None

    classes = _scapy_generated_packet_classes(
        plugin_dir=plugin_dir,
        entry_path=entry_path,
        filter_name=filter_name,
    )
    if not classes:
        return [], False, None

    old_silence = os.environ.get("PARGEN_CONSTRAINT_SILENCE")
    os.environ["PARGEN_CONSTRAINT_SILENCE"] = "1"
    best: Optional[Tuple[Tuple[int, int, int, int], List[str], Dict[str, Any]]] = None
    try:
        for cand_idx, (source, data) in enumerate(candidates):
            if not data:
                continue
            for cls_idx, cls in enumerate(classes[:8]):
                try:
                    parsed = cls(bytes(data))
                except Exception:
                    continue
                msgs: List[str] = []
                for layer in _iter_scapy_layers(parsed):
                    v = getattr(layer, "_pargen_constraint_violations", None)
                    if isinstance(v, list):
                        msgs.extend([str(x) for x in v if str(x).strip()])
                meta = {
                    "source": source,
                    "parser_class": getattr(cls, "__name__", str(cls)),
                    "bytes_len": len(data),
                    "violations_count": len(msgs),
                }
                score = (
                    _scapy_class_name_score(cls, filter_name=filter_name),
                    1 if msgs else 0,
                    -len(msgs),
                    -cand_idx * 100 - cls_idx,
                )
                if best is None or score > best[0]:
                    best = (score, msgs, meta)
    finally:
        if old_silence is None:
            os.environ.pop("PARGEN_CONSTRAINT_SILENCE", None)
        else:
            os.environ["PARGEN_CONSTRAINT_SILENCE"] = old_silence

    if best is None:
        return [], False, None
    return best[1], True, best[2]


def _run_scapy_case(
    *,
    plugin_dir: Path,
    filter_name: str,
    pcap: Path,
    frame_no: int,
    entry_override: Optional[Path] = None,
) -> str:
    out_text, _has_layer = _run_scapy_case_with_meta(
        plugin_dir=plugin_dir,
        filter_name=filter_name,
        pcap=pcap,
        frame_no=frame_no,
        entry_override=entry_override,
    )
    return out_text


def _run_scapy_case_with_meta(
    *,
    plugin_dir: Path,
    filter_name: str,
    pcap: Path,
    frame_no: int,
    case_name: Optional[str] = None,
    log_path: Optional[Path] = None,
    entry_override: Optional[Path] = None,
) -> Tuple[str, bool]:
    """
    Return (violations_text, has_target_layer).

    has_target_layer is True when Scapy builds at least one layer class coming from the Stage C plugin module.
    This helps binary exp_constraint mode treat "protocol layer missing" as abnormal.
    """

    try:
        from scapy.utils import PcapReader  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    entry = _ensure_scapy_plugin_loaded(
        plugin_dir=plugin_dir,
        filter_name=filter_name,
        entry_override=entry_override,
    )
    entry_path = Path(entry).resolve()
    plugin_root = plugin_dir.resolve()

    idx = max(0, int(frame_no) - 1)
    pkt: Any = None

    old_silence = os.environ.get("PARGEN_CONSTRAINT_SILENCE")
    os.environ["PARGEN_CONSTRAINT_SILENCE"] = "1"
    try:
        with PcapReader(str(pcap)) as reader:
            for i, p in enumerate(reader):
                if i == idx:
                    pkt = p
                    break
    finally:
        if old_silence is None:
            os.environ.pop("PARGEN_CONSTRAINT_SILENCE", None)
        else:
            os.environ["PARGEN_CONSTRAINT_SILENCE"] = old_silence

    if pkt is None:
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "case": str(case_name or ""),
                    "pcap": str(pcap),
                    "frame_no": int(frame_no),
                    "has_target_layer": False,
                    "reason": "missing packet at requested frame_no",
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
        return "", False

    msgs: List[str] = []
    has_target_layer = False
    layer_meta: List[Dict[str, Any]] = []
    fallback_meta: Optional[Dict[str, Any]] = None
    for layer in _iter_scapy_layers(pkt):
        try:
            mod_name = getattr(layer.__class__, "__module__", None)
            mod_obj = sys.modules.get(str(mod_name)) if mod_name else None
            mod_file = getattr(mod_obj, "__file__", None) if mod_obj else None
            mod_file_resolved: Optional[str] = None
            if isinstance(mod_file, str) and mod_file:
                try:
                    mod_file_resolved = str(Path(mod_file).resolve())
                except Exception:
                    mod_file_resolved = mod_file
            if mod_file_resolved:
                try:
                    if Path(mod_file_resolved) == entry_path or plugin_root in Path(mod_file_resolved).parents:
                        has_target_layer = True
                except Exception:
                    pass
            layer_meta.append(
                {
                    "layer": layer.__class__.__name__,
                    "module": mod_name,
                    "module_file": mod_file_resolved,
                }
            )
        except Exception:
            pass
        v = getattr(layer, "_pargen_constraint_violations", None)
        if isinstance(v, list):
            msgs.extend([str(x) for x in v if str(x).strip()])

    if not has_target_layer:
        fallback_msgs, fallback_has_layer, fallback_meta = _scapy_fallback_parse_profile_bytes(
            plugin_dir=plugin_root,
            entry_path=entry_path,
            filter_name=filter_name,
            pcap=pcap,
            frame_no=frame_no,
        )
        if fallback_has_layer:
            has_target_layer = True
            if fallback_msgs:
                msgs.extend(fallback_msgs)

    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                raw = getattr(pkt, "original", None)
            except Exception:
                raw = None
            if not isinstance(raw, (bytes, bytearray)):
                try:
                    raw = bytes(pkt)
                except Exception:
                    raw = b""
            raw_hex = raw.hex() if isinstance(raw, (bytes, bytearray)) else ""
            payload = {
                "case": str(case_name or ""),
                "pcap": str(pcap),
                "frame_no": int(frame_no),
                "has_target_layer": bool(has_target_layer),
                "layers": layer_meta,
                "violations_count": len(msgs),
                "violations": list(msgs)[:50],
                "raw_hex": raw_hex,
                "entry": str(entry_path),
            }
            if fallback_meta is not None:
                payload["fallback_target_parse"] = fallback_meta
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    return "\n".join(msgs), has_target_layer


def _load_codegen_profile_from_stage_c_meta(*, plugin_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = plugin_dir / "_pargen_stage_c_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = _load_json(meta_path)
    except Exception:
        return None
    if not isinstance(meta, Mapping):
        return None
    raw = meta.get("codegen_profile_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _prepare_zeek_spicy_plugin_for_exp_constraint(
    *,
    plugin_dir: Path,
    protocol_filter_name: str,
) -> Tuple[tempfile.TemporaryDirectory, Path, bool]:
    """Return a temporary Zeek+Spicy plugin copy with backend static fixes applied."""

    src_dir = Path(plugin_dir)
    tmp = tempfile.TemporaryDirectory(prefix="pargen_exp_constraint_zeek_spicy_plugin_")
    dst_dir = Path(tmp.name) / src_dir.name
    shutil.copytree(src_dir, dst_dir)

    changed = False
    return tmp, dst_dir, changed


def _run_zeek_spicy_case(
    *,
    plugin_dir: Path,
    filter_name: str,
    pcap: Path,
    frame_no: int,
    codegen_profile: Optional[Dict[str, Any]],
) -> Tuple[str, bool]:
    from pargen.parser_realization.target_platforms.zeek_spicy_export import export_zeek_spicy_tooljson

    with tempfile.TemporaryDirectory(prefix="pargen_exp_constraint_zeek_spicy_") as td:
        out_dir = Path(td)
        export_zeek_spicy_tooljson(
            plugin_src_dir=plugin_dir,
            pcap_path=pcap,
            filter_name=str(filter_name),
            out_dir=out_dir,
            codegen_profile=codegen_profile,
            frame_index=[int(frame_no)],
            max_packets=1,
            fail_on_all_parse_failures=False,
        )

        tooljson_path = out_dir / "tool.json"
        payload: Any = None
        try:
            payload = _load_json(tooljson_path)
        except Exception:
            payload = None

        ok = True
        reason = "ok"
        stderr = ""

        if not isinstance(payload, list) or not payload:
            ok = False
            reason = "no packets exported"
        else:
            pkt0 = payload[0] if isinstance(payload[0], dict) else {}
            dbg = pkt0.get("debug") if isinstance(pkt0, dict) else {}
            dbg = dbg if isinstance(dbg, dict) else {}
            raw_err = dbg.get("error")
            if raw_err:
                ok = False
                reason = str(raw_err)
            else:
                rc = dbg.get("rc")
                if rc is not None:
                    try:
                        if int(rc) != 0:
                            ok = False
                            reason = f"rc={int(rc)}"
                    except Exception:
                        ok = False
                        reason = f"rc={rc}"
            stderr = str(dbg.get("stderr") or "").strip()

        out_text = f"zeek_spicy parse_ok={ok} reason={reason}"
        if stderr:
            out_text += "\n" + stderr
        return out_text, ok


def _log_has_rows(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                return True
    except Exception:
        return False
    return False


def _run_zeek_builtin_case(
    *,
    pcap: Path,
    filter_name: str,
) -> Tuple[str, bool]:
    """
    Run the system Zeek (builtin analyzers) on a PCAP and return (out_text, parse_ok).

    Binary rule (parse_ok):
    - zeek exits with rc==0
    - AND weird.log has no non-comment rows
    - AND <filter_name>.log exists with at least one non-comment row (best-effort signal that the
      target analyzer produced output).
    """

    zeek = _resolve_zeek_bin()
    timeout_s = _resolve_zeek_timeout_s()

    fn = str(filter_name or "").strip()
    if not fn:
        raise ValueError("filter_name is required for zeek_builtin (expected <protocol>.log basename)")

    with tempfile.TemporaryDirectory(prefix="pargen_exp_constraint_zeek_builtin_") as td:
        out_dir = Path(td)
        cmd = [str(zeek), "-C", "-r", str(pcap)]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(out_dir),
            timeout=(timeout_s if timeout_s > 0 else None),
        )
        combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

        ok = True
        reason = "ok"
        rc = int(proc.returncode or 0)
        if rc != 0:
            ok = False
            reason = f"zeek_rc={rc}"
        elif _log_has_rows(out_dir / "weird.log"):
            ok = False
            reason = "weird.log"
        elif not _log_has_rows(out_dir / f"{fn}.log"):
            ok = False
            reason = f"missing {fn}.log"

        out_text = f"zeek_builtin parse_ok={ok} reason={reason}"
        if combined.strip():
            out_text += "\n" + combined.strip()
        return out_text, ok


def _filter_pcap_to_frames(
    *,
    tshark: Path,
    src_pcap: Path,
    include_frames: Sequence[int],
    dst_pcap: Path,
) -> None:
    frames = sorted({int(x) for x in include_frames if int(x) > 0})
    if not frames:
        raise ValueError("include_frames is empty")

    filt = " || ".join([f"frame.number=={n}" for n in frames])
    cmd = [str(tshark), "-r", str(src_pcap), "-Y", filt, "-w", str(dst_pcap)]
    timeout_s = _resolve_tshark_exp_constraint_timeout_s()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=(float(timeout_s) if timeout_s > 0 else None),
        )
    except subprocess.TimeoutExpired as exc:
        msg = getattr(exc, "stderr", None) or getattr(exc, "output", None) or ""
        if isinstance(msg, bytes):
            msg = msg.decode("utf-8", errors="replace")
        msg_s = str(msg).strip()
        tail = f": {msg_s}" if msg_s else ""
        raise RuntimeError(f"tshark frame filter timed out after {float(exc.timeout or timeout_s):.1f}s{tail}") from exc
    if int(proc.returncode or 0) != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"tshark failed to filter frames (rc={proc.returncode}): {msg}")
    if not dst_pcap.exists():
        raise RuntimeError(f"tshark did not write filtered pcap: {dst_pcap}")


def run_zeek_builtin_dataset_parse_summary(
    *,
    protocol: str,
    filter_name: str,
    frame_index_path: Optional[Path] = None,
    tshark_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    save_logs_dir: Optional[Path] = None,
) -> ZeekBuiltinDatasetParseSummary:
    """Run Zeek's built-in analyzers on the selected PCAP frames.
    
    Report whether the protocol log is produced and whether weird.log contains
    malformed-input observations."""

    proto = str(protocol or "").strip().lower()
    if not proto:
        raise ValueError("protocol is required")
    fn = str(filter_name or "").strip()
    if not fn:
        raise ValueError("filter_name is required (expected <protocol>.log basename)")

    proto_dir = (DATA_DIR / proto).resolve()
    index_path = (frame_index_path or (proto_dir / "inputs" / "pcap_frame_index.json")).resolve()
    if not index_path.exists():
        raise RuntimeError(f"pcap_frame_index.json not found: {index_path}")

    payload = _load_json(index_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"pcap_frame_index.json is not an object: {index_path}")

    raw_pcap = payload.get("pcap")
    if not isinstance(raw_pcap, str) or not raw_pcap.strip():
        raise ValueError(f"pcap_frame_index.json missing pcap path: {index_path}")
    pcap_path = (proto_dir / raw_pcap.strip()).resolve()
    if not pcap_path.exists():
        raise RuntimeError(f"dataset pcap not found: {pcap_path}")

    include_frames: List[int] = []
    raw_frames = payload.get("include_frames")
    if isinstance(raw_frames, list):
        for v in raw_frames:
            try:
                i = int(v)
            except Exception:
                continue
            if i > 0:
                include_frames.append(i)
    frames_indexed = len(sorted(set(include_frames)))

    zeek = _resolve_zeek_bin()
    tshark = _resolve_tshark_bin(tshark_path)
    timeout_s = _resolve_zeek_timeout_s()

    filtered_pcap: Optional[Path] = None
    pcap_for_run = pcap_path

    start = time.time()
    zeek_rc = -1
    ok = False
    reason = "unknown"
    weird_rows = 0
    log_rows = 0

    try:
        with tempfile.TemporaryDirectory(prefix="pargen_zeek_builtin_dataset_") as td:
            tmp = Path(td)
            if frames_indexed > 0:
                filtered_pcap = tmp / f"{proto}_{fn}_include_frames.pcapng"
                _filter_pcap_to_frames(
                    tshark=tshark,
                    src_pcap=pcap_path,
                    include_frames=include_frames,
                    dst_pcap=filtered_pcap,
                )
                pcap_for_run = filtered_pcap

            out_dir = tmp / "zeek_out"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [str(zeek), "-C", "-r", str(pcap_for_run)]
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(out_dir),
                    timeout=(timeout_s if timeout_s > 0 else None),
                )
                zeek_rc = int(proc.returncode or 0)
            except subprocess.TimeoutExpired:
                zeek_rc = -1
                ok = False
                reason = "timeout"
            else:
                weird_rows = _log_row_count(out_dir / "weird.log")
                log_rows = _log_row_count(out_dir / f"{fn}.log")

                if zeek_rc != 0:
                    ok = False
                    reason = f"zeek_rc={zeek_rc}"
                elif log_rows <= 0:
                    ok = False
                    reason = f"missing {fn}.log"
                else:
                    ok = True
                    # Note: unlike exp_constraint binary verdicts, we do NOT treat weird.log as a
                    # failure signal for the normal dataset parse summary. The TCP normal dataset
                    # can legitimately trigger Zeek TCP weirds (e.g., incomplete flows, midstream
                    # captures) while still parsing and producing the target log.
                    reason = "ok"

            # Optionally persist Zeek logs for inspection.
            if save_logs_dir is not None:
                dst = Path(save_logs_dir)
                dst.mkdir(parents=True, exist_ok=True)
                for log_path in sorted(out_dir.glob("*.log")):
                    if log_path.is_file():
                        try:
                            shutil.copy2(log_path, dst / log_path.name)
                        except Exception:
                            pass
    finally:
        duration_s = max(0.0, float(time.time() - start))

    summary = ZeekBuiltinDatasetParseSummary(
        protocol=proto,
        host="zeek_builtin",
        pcap=str(pcap_path),
        frames_indexed=int(frames_indexed),
        filter_name=str(fn),
        timeout_s=float(timeout_s),
        zeek_bin=str(zeek),
        tshark_bin=str(tshark),
        pcap_filtered=bool(filtered_pcap is not None),
        logs_dir=str(Path(save_logs_dir)) if save_logs_dir is not None else None,
        zeek_rc=int(zeek_rc),
        parse_ok=bool(ok),
        reason=str(reason),
        weird_rows=int(weird_rows),
        log_rows=int(log_rows),
        duration_s=float(duration_s),
    )

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def run_scapy_builtin_dataset_parse_summary(
    *,
    protocol: str,
    filter_name: str,
    frame_index_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    packet_log_path: Optional[Path] = None,
) -> ScapyBuiltinDatasetParseSummary:
    """
    Run Scapy's official contrib dissector on the protocol's normal-dataset PCAP and emit a parse summary.

    This is a coarse baseline (binary parse-only):
    - parse_ok counts frames where Scapy builds the target protocol layer.
    - parse_fail counts frames where the layer is missing.
    """

    proto = str(protocol or "").strip().lower()
    if not proto:
        raise ValueError("protocol is required")
    fn = str(filter_name or "").strip()
    if not fn:
        raise ValueError("filter_name is required")

    proto_dir = (DATA_DIR / proto).resolve()
    index_path = (frame_index_path or (proto_dir / "inputs" / "pcap_frame_index.json")).resolve()
    if not index_path.exists():
        raise RuntimeError(f"pcap_frame_index.json not found: {index_path}")

    payload = _load_json(index_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"pcap_frame_index.json is not an object: {index_path}")

    raw_pcap = payload.get("pcap")
    if not isinstance(raw_pcap, str) or not raw_pcap.strip():
        raise ValueError(f"pcap_frame_index.json missing pcap path: {index_path}")
    pcap_path = (proto_dir / raw_pcap.strip()).resolve()
    if not pcap_path.exists():
        raise RuntimeError(f"dataset pcap not found: {pcap_path}")

    include_frames: List[int] = []
    raw_frames = payload.get("include_frames")
    if isinstance(raw_frames, list):
        for v in raw_frames:
            try:
                i = int(v)
            except Exception:
                continue
            if i > 0:
                include_frames.append(i)
    include_set = set(include_frames)
    frames_indexed = len(sorted(include_set)) if include_set else 0

    try:
        from scapy.utils import PcapReader  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Scapy is not installed. Install it with: pip install scapy") from exc

    if proto == "modbus":
        layer_types = _scapy_builtin_modbus_layers()
    elif proto == "bfd":
        layer_types = _scapy_builtin_bfd_layers()
    elif proto == "tcp":
        layer_types = _scapy_builtin_tcp_layers()
    elif proto == "ipv4":
        layer_types = _scapy_builtin_ipv4_layers()
    elif proto == "icmp":
        layer_types = _scapy_builtin_icmp_layers()
    elif proto == "dhcp":
        layer_types = _scapy_builtin_dhcp_layers()
    elif proto == "dns":
        layer_types = _scapy_builtin_dns_layers()
    else:
        raise ValueError(f"scapy_builtin dataset parse summary does not support protocol={proto!r}")

    ok = 0
    fail = 0
    errors = 0
    error_samples: List[Dict[str, Any]] = []
    log_handle = None
    start = time.time()
    try:
        if packet_log_path is not None:
            packet_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = packet_log_path.open("w", encoding="utf-8")
        with PcapReader(str(pcap_path)) as reader:
            for idx, pkt in enumerate(reader, 1):
                if include_set and idx not in include_set:
                    continue
                try:
                    has_layer = any(pkt.haslayer(t) for t in layer_types)  # type: ignore[arg-type]
                except Exception as exc:
                    errors += 1
                    has_layer = False
                    if len(error_samples) < 5:
                        error_samples.append({"frame_no": int(idx), "error": f"{type(exc).__name__}: {exc}"})
                if has_layer:
                    ok += 1
                else:
                    fail += 1
                if log_handle is not None:
                    try:
                        raw = getattr(pkt, "original", None)
                    except Exception:
                        raw = None
                    if not isinstance(raw, (bytes, bytearray)):
                        try:
                            raw = bytes(pkt)
                        except Exception:
                            raw = b""
                    raw_hex = raw.hex() if isinstance(raw, (bytes, bytearray)) else ""

                    layer_name: Optional[str] = None
                    fields_primary: Dict[str, Any] = {}
                    fields_secondary: Dict[str, Any] = {}
                    secondary_layer_name: Optional[str] = None
                    if has_layer:
                        try:
                            for t in layer_types:
                                if pkt.haslayer(t):  # type: ignore[arg-type]
                                    layer = pkt.getlayer(t)
                                    if layer is None:
                                        continue
                                    layer_name = type(layer).__name__
                                    try:
                                        fields = getattr(layer, "fields", None)
                                    except Exception:
                                        fields = None
                                    if isinstance(fields, dict):
                                        for k, v in list(fields.items())[:100]:
                                            if isinstance(v, (str, int, float, bool)) or v is None:
                                                fields_primary[str(k)] = v
                                            else:
                                                fields_primary[str(k)] = repr(v)

                                    if proto == "modbus":
                                        for attr in ("transId", "protoId", "len", "unitId"):
                                            try:
                                                if hasattr(layer, attr):
                                                    fields_secondary[attr] = getattr(layer, attr)
                                            except Exception:
                                                continue
                                        try:
                                            pdu = getattr(layer, "payload", None)
                                        except Exception:
                                            pdu = None
                                        if pdu is not None:
                                            secondary_layer_name = type(pdu).__name__
                                            try:
                                                pdu_fields = getattr(pdu, "fields", None)
                                            except Exception:
                                                pdu_fields = None
                                            if isinstance(pdu_fields, dict):
                                                for k, v in list(pdu_fields.items())[:100]:
                                                    if isinstance(v, (str, int, float, bool)) or v is None:
                                                        fields_secondary[str(k)] = v
                                                    else:
                                                        fields_secondary[str(k)] = repr(v)
                                    break
                        except Exception:
                            pass

                    entry = {
                        "frame_no": int(idx),
                        "protocol": proto,
                        "filter_name": str(fn),
                        "has_layer": bool(has_layer),
                        "layer": layer_name,
                        "fields": fields_primary,
                        "secondary_layer": secondary_layer_name,
                        "secondary_fields": fields_secondary,
                        "raw_len": int(len(raw)) if isinstance(raw, (bytes, bytearray)) else None,
                        "raw_hex_prefix": raw_hex[:512],
                        "summary": (pkt.summary() if hasattr(pkt, "summary") else repr(pkt))[:2000],
                    }
                    log_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
        duration_s = max(0.0, float(time.time() - start))

    summary = ScapyBuiltinDatasetParseSummary(
        protocol=proto,
        host="scapy_builtin",
        pcap=str(pcap_path),
        frames_indexed=int(frames_indexed),
        filter_name=str(fn),
        parse_ok=int(ok),
        parse_fail=int(fail),
        errors=int(errors),
        duration_s=float(duration_s),
        error_samples=tuple(error_samples),
    )

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def run_exp_constraint_suite(
    *,
    protocol: str,
    host: str,
    mode: str = "binary",
    suite_dir: Optional[Path] = None,
    mapping_path: Optional[Path] = None,
    tshark_path: Optional[Path] = None,
    plugin_id: Optional[str] = None,
    stage_c_out_dir: Optional[Path] = None,
    lua_entry: Optional[Path] = None,
    scapy_entry: Optional[Path] = None,
    plugin_dir: Optional[Path] = None,
    filter_name: Optional[str] = None,
    ws_isolate_plugins: bool = True,
    verbose: bool = False,
    packet_logs_dir: Optional[Path] = None,
    binary_require_bs_constraint: bool = False,
    binary_positive_policy: str = "unique_seed",
    scapy_builtin_abnormal_policy: str = "layer",
) -> Optional[SuiteResult]:
    """
    Validate exp_constraint negative PCAPs.

    Modes:
    - per_constraint: a case passes only if every target constraint emits the expected alert tag.
    - binary: a case passes if the host output indicates "abnormal" (cannot parse normally), e.g.
              any `BS_CONSTRAINT` tag, `Malformed Packet`, or missing target protocol layer.

    Binary strictness:
    - binary_require_bs_constraint=True requires `BS_CONSTRAINT` to appear in the host output for generated hosts.
      This avoids counting generic expert-info, parse failures, dissector crashes ("Lua Error"/"Dissector bug"), or
      missing-layer outcomes as successful abnormal detection.
    - A mapping with binary_require_exact_target_tags=true additionally requires every case's exact target tag
      for the acceptance gate (success/fail), but never rewrites the observed binary detection outcome.
      Separate metrics expose detection, target attribution and independently verified target coverage.

    Coverage semantics:
    - per_constraint: NOT present in tree / missing catalog entry => counts as FAIL (no SKIP).
    - binary: normally checks only normal/abnormal behavior.  Tree-only mappings opt into exact
      target tags and strict runtime-health checks with binary_require_exact_target_tags=true.

    Binary confusion matrix:
    - When the mapping provides `base_pcap` and per-case `seed_frame_no`, binary mode also evaluates the
      corresponding seed (normal) packets and reports a confusion matrix:
        TP: seed -> normal, FN: seed -> abnormal, FP: mutated -> normal, TN: mutated -> abnormal.
    - binary_positive_policy controls how normal seeds are counted:
        unique_seed de-duplicates repeated seed frame numbers (legacy repair-gate behavior).
        paired_seed counts one positive counterpart per negative case (paired tests).
    - scapy_builtin_abnormal_policy controls deployed Scapy baseline abnormal indicators:
        layer uses only target-layer presence (legacy parse-only baseline).
        structural also counts incomplete Scapy structure, e.g. missing DHCP options or impossible header lengths.
        observable additionally counts stricter value/checksum abnormalities exposed by Scapy-decoded fields.
    """
    proto = str(protocol or "").strip().lower()
    if not proto:
        raise ValueError("protocol is required")

    mode_norm = str(mode or "").strip().lower().replace("-", "_")
    if mode_norm in {"constraint", "constraints", "perconstraint"}:
        mode_norm = "per_constraint"
    if mode_norm not in {"per_constraint", "binary"}:
        raise ValueError(f"Unsupported exp_constraint mode={mode!r} (expected 'per_constraint' or 'binary')")

    host_norm = _host_normalize(host)
    if host_norm == "zeek_builtin" and mode_norm != "binary":
        raise ValueError("zeek_builtin supports mode=binary only (no per-constraint tag coverage)")
    if host_norm == "scapy_builtin" and mode_norm != "binary":
        raise ValueError("scapy_builtin supports mode=binary only (parse-only baseline; no per-constraint tag coverage)")
    positive_policy = str(binary_positive_policy or "unique_seed").strip().lower().replace("-", "_")
    if positive_policy not in {"unique_seed", "paired_seed"}:
        raise ValueError(
            "binary_positive_policy must be one of {'unique_seed', 'paired_seed'} "
            f"(got {binary_positive_policy!r})"
        )
    scapy_policy = str(scapy_builtin_abnormal_policy or "layer").strip().lower().replace("-", "_")
    if scapy_policy not in {"layer", "structural", "observable"}:
        raise ValueError(
            "scapy_builtin_abnormal_policy must be one of {'layer', 'structural', 'observable'} "
            f"(got {scapy_builtin_abnormal_policy!r})"
        )
    suite = (suite_dir or _default_suite_dir(proto)).resolve()
    if not suite.exists():
        return None

    mapping = (mapping_path or _default_mapping_path(protocol=proto, suite_dir=suite))
    if mapping is None:
        return None
    mapping = mapping.resolve()
    if not mapping.exists():
        return None

    payload = _load_json(mapping)
    if not isinstance(payload, Mapping):
        raise ValueError(f"mapping is not a JSON object: {mapping}")

    cases = payload.get("cases") or {}
    if not isinstance(cases, Mapping):
        raise ValueError(f"mapping missing cases: {mapping}")
    binary_require_exact_target_tags = bool(payload.get("binary_require_exact_target_tags", False))

    catalog = _coerce_constraints_catalog(payload)
    if (mode_norm == "per_constraint" or binary_require_exact_target_tags) and not catalog:
        raise ValueError(f"mapping missing constraints_catalog/constraints_from_tree: {mapping}")

    case_names = _iter_case_names(payload)
    if not case_names:
        return None

    expectations: List[CaseExpectation] = []
    seed_cases: List[Tuple[str, int]] = []
    explicit_positive_seeds: List[Tuple[str, int]] = []
    raw_positive_seeds = payload.get("positive_seeds")
    if isinstance(raw_positive_seeds, list):
        for index, item in enumerate(raw_positive_seeds, start=1):
            if not isinstance(item, Mapping):
                continue
            try:
                frame_no = int(item.get("frame_no") or 0)
            except Exception:
                continue
            if frame_no <= 0:
                continue
            packet_id = str(item.get("packet_id") or f"frame_{frame_no}_{index}").strip()
            explicit_positive_seeds.append((packet_id or f"frame_{frame_no}_{index}", frame_no))
    mapping_seed_frame_no: Optional[int] = None
    raw_mapping_seed = payload.get("seed_frame_no")
    if raw_mapping_seed is not None:
        try:
            mapping_seed_frame_no = int(raw_mapping_seed)
        except Exception:
            mapping_seed_frame_no = None
    for case_name in case_names:
        case_info = cases.get(case_name) or {}
        if not isinstance(case_info, Mapping):
            continue
        seed_frame_no = _resolve_seed_frame_no(case_info)
        if seed_frame_no is None:
            seed_frame_no = mapping_seed_frame_no
        if seed_frame_no is not None and seed_frame_no > 0:
            seed_cases.append((str(case_name), int(seed_frame_no)))
        expectations.append(
            _expectations_for_case(
                protocol=proto,
                host=host_norm,
                suite_dir=suite,
                case_name=case_name,
                case_info=case_info,
                catalog=catalog,
            )
        )

    if not expectations:
        return None

    require_suite_local_base = str(payload.get("base_pcap_scope") or "").strip().lower() == "suite_local_required"
    positive_seed_scope = str(payload.get("positive_seed_scope") or "").strip().lower()
    if positive_seed_scope == "all_replay_valid_representative_paths":
        coverage_payload = payload.get("coverage")
        representative_payload = (
            coverage_payload.get("representative_paths")
            if isinstance(coverage_payload, Mapping)
            else None
        )
        try:
            # One seed per materialized path, plus the upper-boundary seeds of those paths.
            expected_positive_count = int(representative_payload.get("materialized")) + int(  # type: ignore[union-attr]
                representative_payload.get("boundary_positives") or 0  # type: ignore[union-attr]
            )
        except Exception as exc:
            raise ValueError(
                "tree-only conformance mapping is missing representative-path materialization count"
            ) from exc
        frame_numbers = [frame_no for _packet_id, frame_no in explicit_positive_seeds]
        packet_ids = [packet_id for packet_id, _frame_no in explicit_positive_seeds]
        if (
            expected_positive_count <= 0
            or len(explicit_positive_seeds) != expected_positive_count
            or len(set(frame_numbers)) != expected_positive_count
            or len(set(packet_ids)) != expected_positive_count
        ):
            raise ValueError(
                "tree-only conformance mapping has an incomplete or duplicate representative positive seed index: "
                f"expected={expected_positive_count} indexed={len(explicit_positive_seeds)}"
            )
    if require_suite_local_base:
        suite_root = suite.resolve()
        for expectation in expectations:
            try:
                expectation.pcap.resolve().relative_to(suite_root)
            except Exception as exc:
                raise ValueError(
                    "tree-only conformance case PCAP must remain inside the generated suite: "
                    f"case={expectation.case!r} pcap={expectation.pcap}"
                ) from exc

    # Resolve host runtime.
    tshark: Optional[Path] = None
    if host_norm in {"wireshark_c", "wireshark_lua"}:
        if tshark_path is not None:
            tshark = Path(tshark_path).expanduser()
        else:
            # Prefer the repo-bundled Wireshark build, except for Wireshark-Lua when
            # that build lacks Lua support. In that case, fall back to the system
            # tshark rather than silently evaluating no Lua parser at all.
            bundled = DATA_DIR / "tools" / "wireshark-4.4.11" / "build" / "run" / "tshark"
            candidates: List[Path] = []
            if bundled.exists():
                candidates.append(bundled)
            found = shutil.which("tshark")
            if found:
                candidates.append(Path(found))
            for candidate in candidates:
                if host_norm == "wireshark_lua" and not _tshark_supports_lua(candidate):
                    continue
                tshark = candidate
                break
        if tshark is None or not tshark.exists():
            raise RuntimeError("tshark not found; build wireshark or pass --stage-c-tshark")

    pid = str(plugin_id or f"{proto}_bs").strip()
    out_dir = (stage_c_out_dir or (DATA_DIR / proto / "outputs" / "stage_c")).resolve()

    env = dict(os.environ)
    extra_args: List[str] = []

    zeek_codegen_profile: Optional[Dict[str, Any]] = None
    zeek_plugin_tmp: Optional[tempfile.TemporaryDirectory] = None
    if host_norm == "zeek_spicy":
        resolved_plugin_dir = Path(plugin_dir) if plugin_dir is not None else (out_dir / "plugin" / "zeek_spicy")
        try:
            resolved_plugin_dir = resolved_plugin_dir.resolve()
        except OSError:
            resolved_plugin_dir = Path(resolved_plugin_dir)
        if not resolved_plugin_dir.exists():
            raise RuntimeError(f"zeek_spicy plugin_dir does not exist: {resolved_plugin_dir}")
        plugin_dir = resolved_plugin_dir
        zeek_codegen_profile = _load_codegen_profile_from_stage_c_meta(plugin_dir=resolved_plugin_dir)
        if zeek_codegen_profile is None:
            cand = DATA_DIR / proto / "inputs" / "stage_c_codegen_profile_zeek_spicy.json"
            if cand.exists():
                try:
                    loaded = _load_json(cand)
                    if isinstance(loaded, dict):
                        zeek_codegen_profile = loaded
                except Exception:
                    zeek_codegen_profile = None
        zeek_plugin_tmp, prepared_plugin_dir, _zeek_static_changed = _prepare_zeek_spicy_plugin_for_exp_constraint(
            plugin_dir=resolved_plugin_dir,
            protocol_filter_name=str(filter_name or proto),
        )
        plugin_dir = prepared_plugin_dir

    if host_norm == "wireshark_c":
        for proto_name in _mapping_disable_protocols(payload):
            extra_args.extend(["--disable-protocol", proto_name])
        if ws_isolate_plugins:
            from pargen.parser_realization.target_platforms.wireshark_export import _ws_prepare_isolated_plugin_root

            plugins_root = _ws_prepare_isolated_plugin_root(tshark_path=tshark, out_dir=out_dir, plugin_id=pid)  # type: ignore[arg-type]
            if plugins_root is not None:
                env["WIRESHARK_PLUGIN_DIR"] = str(plugins_root)
    elif host_norm == "wireshark_lua":
        if lua_entry is None:
            raise ValueError("lua_entry is required for wireshark_lua exp_constraint validation")
        entry = Path(lua_entry)
        extra_args = []
        for proto_name in _mapping_disable_protocols(payload):
            extra_args.extend(["--disable-protocol", proto_name])
        extra_args.extend(["-X", f"lua_script:{entry}"])
        if ws_isolate_plugins:
            from pargen.parser_realization.target_platforms.wireshark_export import _ws_prepare_isolated_plugin_root

            plugins_root = _ws_prepare_isolated_plugin_root(tshark_path=tshark, out_dir=out_dir, plugin_id=None)  # type: ignore[arg-type]
            if plugins_root is not None:
                env["WIRESHARK_PLUGIN_DIR"] = str(plugins_root)

    success = 0
    fail = 0
    results: List[CaseResult] = []
    positive_expectations: List[CaseExpectation] = []
    if mode_norm == "binary" and (explicit_positive_seeds or seed_cases):
        base_pcap = _resolve_base_pcap(
            protocol=proto,
            suite_dir=suite,
            base_pcap=payload.get("base_pcap"),
            require_suite_local=require_suite_local_base,
        )
        if require_suite_local_base and base_pcap is None:
            raise ValueError(
                "tree-only conformance suite is missing its suite-local synthetic positive PCAP; "
                "dataset fallback is forbidden"
            )
        if base_pcap is not None:
            if explicit_positive_seeds:
                positive_seed_items = explicit_positive_seeds
            elif positive_policy == "paired_seed":
                positive_seed_items = seed_cases
            else:
                positive_seed_items = [(f"frame_{seed_frame_no}", seed_frame_no) for seed_frame_no in sorted({f for _, f in seed_cases})]
            for case_name, seed_frame_no in positive_seed_items:
                positive_expectations.append(
                    CaseExpectation(
                        case=f"POS::{case_name}",
                        pcap=base_pcap,
                        frame_no=int(seed_frame_no),
                        constraints=(),
                    )
                )

    def _extract_protocols_in_frame(text: str) -> List[str]:
        for line in (text or "").splitlines():
            s = line.strip()
            if not s.startswith("[Protocols in frame:"):
                continue
            # Example: [Protocols in frame: eth:ethertype:ip:tcp:mbtcp:modbus]
            try:
                inner = s.split(":", 1)[1]
                inner = inner.strip()
                if inner.endswith("]"):
                    inner = inner[:-1]
                inner = inner.strip()
                if not inner:
                    return []
                return [tok.strip() for tok in inner.split(":") if tok.strip()]
            except Exception:
                return []
        return []

    binary_require_tag = bool(binary_require_bs_constraint)
    abnormal_filter_name = str(filter_name) if filter_name is not None else ""

    def _ws_has_abnormal_expert_info(text: str) -> Optional[str]:
        """
        Best-effort binary abnormal indicator for Wireshark:
        treat expert warnings/errors/malformed reports as abnormal parsing outcomes.
        """
        for line in (text or "").splitlines():
            lower = line.lower()
            if "expert info" not in lower:
                continue
            # Example: [Expert Info (Warning/Protocol): ...]
            if "(warn" in lower or "(warning" in lower or "(error" in lower or "malformed" in lower:
                return f"expert info: {line.strip()[:320]}"
        return None

    def _ws_has_dissector_bug(text: str) -> bool:
        lower = (text or "").lower()
        return (
            ("dissector bug" in lower)
            or ("lua error" in lower)
            or ("unregistered expert info" in lower)
            or ("unhandled exception" in lower)
            or ("failed assertion" in lower)
        )

    def _ws_runtime_diagnostic(text: str, *, max_chars: int = 320) -> str:
        """
        Return a compact Wireshark runtime diagnostic for repair prompts.

        tshark often prints useful assertion/plugin-load failures after a long
        partial text tree. Keeping only the diagnostic line makes LLM repair much
        more actionable without dumping packet output into every failure signal.
        """
        priority_needles = (
            "tshark timeout",
            "unhandled exception",
            "failed assertion",
            "dissector bug",
            "unregistered expert info",
            "lua error",
            "malformed packet",
        )
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        for needle in priority_needles:
            for line in lines:
                if needle in line.lower():
                    return line[:max_chars]
        return (lines[-1][:max_chars] if lines else "")

    def _ws_tshark_timed_out(out_text: str, tshark_rc: Optional[int]) -> bool:
        try:
            if tshark_rc is not None and int(tshark_rc) == 124:
                return True
        except Exception:
            pass
        return "tshark timeout" in (out_text or "").lower()

    def _binary_is_abnormal(
        *,
        host: str,
        out_text: str,
        tshark_rc: Optional[int] = None,
        filter_name: Optional[str] = None,
        scapy_has_layer: Optional[bool] = None,
        zeek_parse_ok: Optional[bool] = None,
        strict_tag: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        strict = binary_require_tag if strict_tag is None else bool(strict_tag)

        # Strict binary mode: require explicit ParGen tags and disallow generic
        # parse failures, missing protocol layers, crashes, or host fallback behavior
        # from being counted as successful constraint rejection.
        if strict:
            if host in {"wireshark_c", "wireshark_lua"}:
                if _ws_tshark_timed_out(out_text or "", tshark_rc):
                    diag = _ws_runtime_diagnostic(out_text or "")
                    return False, diag or "tshark timeout"
                if _ws_has_dissector_bug(out_text or ""):
                    diag = _ws_runtime_diagnostic(out_text or "")
                    return False, f"dissector bug: {diag}" if diag else "dissector bug"
                if tshark_rc is not None:
                    try:
                        if int(tshark_rc) != 0:
                            diag = _ws_runtime_diagnostic(out_text or "")
                            return False, f"tshark_rc={tshark_rc}; {diag}" if diag else f"tshark_rc={tshark_rc}"
                    except Exception:
                        return False, "tshark_rc=unknown"
            elif host == "scapy" and scapy_has_layer is not True:
                return False, "missing target protocol layer"
            elif host in {"zeek_spicy", "zeek_builtin"} and zeek_parse_ok is not True:
                # Generated Spicy checks intentionally use `throw` to abort an
                # invalid packet, so parse_ok=False is expected when the exact
                # BS_CONSTRAINT exception is observable.  Keep infrastructure
                # failures distinct: a tag printed before a crash/timeout must
                # not turn that unhealthy run into a conformance success.
                zeek_text = (out_text or "").lower()
                fatal_markers = (
                    "segmentation fault",
                    "core dumped",
                    "timed out",
                    "timeout",
                    "assertion failed",
                    "internal error",
                    "fatal error",
                    "no packets exported",
                    "failed to load",
                    "compilation failed",
                )
                expected_constraint_throw = (
                    zeek_parse_ok is False
                    and "BS_CONSTRAINT" in (out_text or "")
                    and not any(marker in zeek_text for marker in fatal_markers)
                )
                if not expected_constraint_throw:
                    return False, "Zeek parse/runtime failure"
            if "BS_CONSTRAINT" in (out_text or ""):
                return True, "BS_CONSTRAINT"
            return False, "missing BS_CONSTRAINT"

        if host in {"wireshark_c", "wireshark_lua"} and _ws_tshark_timed_out(out_text or "", tshark_rc):
            diag = _ws_runtime_diagnostic(out_text or "")
            return False, diag or "tshark timeout"

        # Any explicit ParGen constraint tag indicates abnormal input.
        if "BS_CONSTRAINT" in (out_text or ""):
            return True, "BS_CONSTRAINT"

        # Wireshark texttree includes this marker on bounds/exception parsing failures.
        if host in {"wireshark_c", "wireshark_lua"} and "Malformed Packet" in (out_text or ""):
            return True, "Malformed Packet"

        if host in {"wireshark_c", "wireshark_lua"}:
            expert = _ws_has_abnormal_expert_info(out_text or "")
            if expert:
                return True, expert

            # Treat internal dissector errors as abnormal (cannot parse normally).
            if _ws_has_dissector_bug(out_text or ""):
                diag = _ws_runtime_diagnostic(out_text or "")
                return True, f"dissector error: {diag}" if diag else "dissector error"
            if tshark_rc is not None and int(tshark_rc) != 0:
                diag = _ws_runtime_diagnostic(out_text or "")
                return True, f"tshark_rc={tshark_rc}; {diag}" if diag else f"tshark_rc={tshark_rc}"

            # Missing target protocol layer => cannot parse normally.
            fn = str(filter_name or "").strip()
            if fn:
                protos = _extract_protocols_in_frame(out_text)
                if protos and fn not in protos:
                    return True, f"missing protocol layer: {fn}"

        if host == "scapy":
            # Scapy host emits constraint violations via `_pargen_constraint_violations` (captured into out_text).
            if (out_text or "").strip():
                return True, "constraint violations"
            if scapy_has_layer is False:
                return True, "missing protocol layer"
            return False, "parsed"
        if host == "scapy_builtin":
            if "SCAPY_BUILTIN_ABNORMAL" in (out_text or ""):
                reason = (out_text or "").strip().split(":", 1)
                return True, reason[1].strip() if len(reason) > 1 else "Scapy observable abnormal"
            if scapy_has_layer is False:
                return True, "missing protocol layer"
            return False, "parsed"

        if host in {"zeek_spicy", "zeek_builtin"}:
            if zeek_parse_ok is False:
                return True, "parse failure"
            return False, "parsed"

        return False, "no abnormal indicator"

    want_confusion = bool(mode_norm == "binary" and positive_expectations)
    tp = fp = tn = fn = 0
    positive_results: List[CaseResult] = []

    observations_by_case: Dict[str, Dict[str, Any]] = {}
    for exp in expectations:
        target_ids = tuple(e.constraint_id for e in exp.constraints)
        uncovered = [e.uncovered_reason for e in exp.constraints if e.uncovered_reason]
        if uncovered and mode_norm == "per_constraint":
            fail += 1
            results.append(
                CaseResult(
                    case=exp.case,
                    ok=False,
                    reason=f"uncovered constraints: {', '.join([u for u in uncovered if u])}",
                    pcap=str(exp.pcap),
                    frame_no=int(exp.frame_no),
                    target_constraints=target_ids,
                )
            )
            if verbose:
                print(f"[FAIL] {exp.case}: {results[-1].reason}")
            continue

        if not exp.pcap.exists():
            fail += 1
            if want_confusion:
                # Infrastructure failures are not successful mutant rejection.
                # Count them conservatively as missed negative detections so the
                # confusion matrix cannot be inflated by a broken/missing suite.
                fp += 1
            results.append(
                CaseResult(
                    case=exp.case,
                    ok=False,
                    reason=f"missing pcap {exp.pcap}",
                    pcap=str(exp.pcap),
                    frame_no=int(exp.frame_no),
                    target_constraints=target_ids,
                )
            )
            if verbose:
                print(f"[FAIL] {exp.case}: {results[-1].reason}")
            continue

        out_text = ""
        tshark_rc: Optional[int] = None
        scapy_has_layer: Optional[bool] = None
        zeek_parse_ok: Optional[bool] = None
        try:
            if host_norm in {"wireshark_c", "wireshark_lua"}:
                tshark_rc, out_text = _run_tshark(
                    tshark=tshark,  # type: ignore[arg-type]
                    pcap=exp.pcap,
                    frame_no=exp.frame_no,
                    env=env,
                    extra_args=extra_args,
                    filter_name=str(filter_name or proto),
                    ws_plugin_id=(pid if host_norm == "wireshark_c" else None),
                    isolate_generated_plugins=bool(ws_isolate_plugins),
                )
            elif host_norm == "scapy":
                if plugin_dir is None:
                    raise ValueError("plugin_dir is required for scapy exp_constraint validation")
                if mode_norm == "binary":
                    case_log_path = None
                    if packet_logs_dir is not None:
                        case_log_path = Path(packet_logs_dir) / "cases" / f"{exp.case}.jsonl"
                    out_text, scapy_has_layer = _run_scapy_case_with_meta(
                        plugin_dir=Path(plugin_dir),
                        filter_name=str(filter_name or proto),
                        pcap=exp.pcap,
                        frame_no=exp.frame_no,
                        case_name=exp.case,
                        log_path=case_log_path,
                        entry_override=scapy_entry,
                    )
                else:
                    out_text = _run_scapy_case(
                        plugin_dir=Path(plugin_dir),
                        filter_name=str(filter_name or proto),
                        pcap=exp.pcap,
                        frame_no=exp.frame_no,
                        entry_override=scapy_entry,
                    )
            elif host_norm == "scapy_builtin":
                case_log_path = None
                if packet_logs_dir is not None:
                    case_log_path = Path(packet_logs_dir) / "cases" / f"{exp.case}.jsonl"
                out_text, scapy_has_layer = _run_scapy_builtin_case_with_meta(
                    protocol=proto,
                    filter_name=str(filter_name or proto),
                    pcap=exp.pcap,
                    frame_no=exp.frame_no,
                    case_name=exp.case,
                    log_path=case_log_path,
                    abnormal_policy=scapy_policy,
                )
            elif host_norm == "zeek_spicy":
                if plugin_dir is None:
                    raise ValueError("plugin_dir is required for zeek_spicy exp_constraint validation")
                out_text, zeek_parse_ok = _run_zeek_spicy_case(
                    plugin_dir=Path(plugin_dir),
                    filter_name=str(filter_name or proto),
                    pcap=exp.pcap,
                    frame_no=exp.frame_no,
                    codegen_profile=zeek_codegen_profile,
                )
            elif host_norm == "zeek_builtin":
                out_text, zeek_parse_ok = _run_zeek_builtin_case(
                    pcap=exp.pcap,
                    filter_name=str(filter_name or proto),
                )
            else:
                raise ValueError(f"Unsupported host for exp_constraint validation: {host_norm}")
        except Exception as exc:
            fail += 1
            if want_confusion:
                fp += 1
            results.append(
                CaseResult(
                    case=exp.case,
                    ok=False,
                    reason=f"runner error: {type(exc).__name__}: {exc}",
                    pcap=str(exp.pcap),
                    frame_no=int(exp.frame_no),
                    target_constraints=target_ids,
                )
            )
            if verbose:
                print(f"[FAIL] {exp.case}: {results[-1].reason}")
            continue

        from .conformance_observations import observe_case, matches_expectation
        evidence = observe_case(exp, cases.get(exp.case) or {}, out_text, tshark_rc,
            logs_dir=packet_logs_dir, pcap_layer=payload.get("pcap_layer"))
        observed_abnormal, observed_why = _binary_is_abnormal(
            host=host_norm, out_text=out_text, tshark_rc=tshark_rc,
            filter_name=abnormal_filter_name, scapy_has_layer=scapy_has_layer,
            zeek_parse_ok=zeek_parse_ok,
            strict_tag=(binary_require_tag or binary_require_exact_target_tags or (mode_norm != "binary" and any(
                bool((catalog.get(ce.constraint_id) or {}).get("strict_contract_tag")) for ce in exp.constraints))))
        target_matched = bool(exp.constraints) and observed_abnormal and all(
            matches_expectation(ce, out_text, catalog) for ce in exp.constraints)
        evidence.update(anomaly_detected=observed_abnormal, detection_reason=observed_why,
                        target_diagnostic_matched=target_matched)
        observations_by_case[exp.case] = evidence

        if mode_norm == "binary":
            abnormal, why = observed_abnormal, observed_why
            accepted = abnormal
            if binary_require_exact_target_tags:
                missing_exact: List[str] = []
                for ce in exp.constraints:
                    if not ce.patterns_any:
                        missing_exact.append(f"{ce.constraint_id}: <no exact tag>")
                        continue
                    if not matches_expectation(ce, out_text, catalog):
                        missing_exact.append(f"{ce.constraint_id}: {ce.patterns_any[0]}")
                if missing_exact:
                    accepted = False
                    why += f"; missing exact target tag(s): {', '.join(missing_exact[:3])}"
            if want_confusion:
                if abnormal:
                    tn += 1
                else:
                    fp += 1
            if accepted:
                success += 1
                results.append(
                    CaseResult(
                        case=exp.case,
                        ok=True,
                        reason=f"abnormal ({why})",
                        pcap=str(exp.pcap),
                        frame_no=int(exp.frame_no),
                        target_constraints=target_ids,
                    )
                )
                if verbose:
                    print(f"[PASS] {exp.case} (abnormal: {why})")
            else:
                fail += 1
                extra = ""
                if uncovered:
                    extra = f"; uncovered={', '.join([u for u in uncovered if u])}"
                results.append(
                    CaseResult(
                        case=exp.case,
                        ok=False,
                        reason=f"{'abnormal_with_other_diagnostic' if abnormal else 'not_detected'} ({why}){extra}",
                        pcap=str(exp.pcap),
                        frame_no=int(exp.frame_no),
                        target_constraints=target_ids,
                    )
                )
                if verbose:
                    print(f"[FAIL] {exp.case}: {results[-1].reason}")
            continue

        strict_runtime = any(
            bool((catalog.get(ce.constraint_id) or {}).get("strict_contract_tag"))
            for ce in exp.constraints
        )
        if strict_runtime:
            runtime_ok, runtime_why = _binary_is_abnormal(
                host=host_norm,
                out_text=out_text,
                tshark_rc=tshark_rc,
                filter_name=abnormal_filter_name,
                scapy_has_layer=scapy_has_layer,
                zeek_parse_ok=zeek_parse_ok,
                strict_tag=True,
            )
            if not runtime_ok:
                fail += 1
                results.append(
                    CaseResult(
                        case=exp.case,
                        ok=False,
                        reason=f"runtime health check failed ({runtime_why})",
                        pcap=str(exp.pcap),
                        frame_no=int(exp.frame_no),
                        target_constraints=target_ids,
                    )
                )
                if verbose:
                    print(f"[FAIL] {exp.case}: {results[-1].reason}")
                continue

        if "Dissector bug" in out_text or "Unregistered expert info" in out_text:
            fail += 1
            results.append(
                CaseResult(
                    case=exp.case,
                    ok=False,
                    reason="wireshark dissector bug detected (Dissector bug / Unregistered expert info)",
                    pcap=str(exp.pcap),
                    frame_no=int(exp.frame_no),
                    target_constraints=target_ids,
                )
            )
            if verbose:
                print(f"[FAIL] {exp.case}: {results[-1].reason}")
            continue

        missing: List[str] = []
        ok = True
        for ce in exp.constraints:
            if not ce.patterns_any:
                ok = False
                missing.append(f"{ce.constraint_id}: <no patterns>")
                continue
            if not matches_expectation(ce, out_text, catalog):
                ok = False
                missing.append(f"{ce.constraint_id}: {ce.patterns_any[0]}")

        if ok:
            success += 1
            results.append(
                CaseResult(
                    case=exp.case,
                    ok=True,
                    reason="ok",
                    pcap=str(exp.pcap),
                    frame_no=int(exp.frame_no),
                    target_constraints=target_ids,
                )
            )
            if verbose:
                print(f"[PASS] {exp.case} (frame={exp.frame_no})")
        else:
            fail += 1
            results.append(
                CaseResult(
                    case=exp.case,
                    ok=False,
                    reason=f"missing expected alert(s): {', '.join(missing[:3])}",
                    pcap=str(exp.pcap),
                    frame_no=int(exp.frame_no),
                    target_constraints=target_ids,
                )
            )
            if verbose:
                print(f"[FAIL] {exp.case}: {results[-1].reason}")

    if want_confusion:
        for exp in positive_expectations:
            if not exp.pcap.exists():
                fn += 1
                positive_results.append(
                    CaseResult(
                        case=exp.case,
                        ok=False,
                        reason=f"abnormal (missing pcap {exp.pcap})",
                        pcap=str(exp.pcap),
                        frame_no=int(exp.frame_no),
                        target_constraints=(),
                    )
                )
                continue

            out_text = ""
            tshark_rc = None
            scapy_has_layer = None
            zeek_parse_ok = None
            try:
                if host_norm in {"wireshark_c", "wireshark_lua"}:
                    tshark_rc, out_text = _run_tshark(
                        tshark=tshark,  # type: ignore[arg-type]
                        pcap=exp.pcap,
                        frame_no=exp.frame_no,
                        env=env,
                        extra_args=extra_args,
                        filter_name=str(filter_name or proto),
                        ws_plugin_id=(pid if host_norm == "wireshark_c" else None),
                        isolate_generated_plugins=bool(ws_isolate_plugins),
                    )
                elif host_norm == "scapy":
                    if plugin_dir is None:
                        raise ValueError("plugin_dir is required for scapy exp_constraint validation")
                    case_log_path = None
                    if packet_logs_dir is not None:
                        case_log_path = Path(packet_logs_dir) / "cases" / f"{exp.case}.jsonl"
                    out_text, scapy_has_layer = _run_scapy_case_with_meta(
                        plugin_dir=Path(plugin_dir),
                        filter_name=str(filter_name or proto),
                        pcap=exp.pcap,
                        frame_no=exp.frame_no,
                        case_name=exp.case,
                        log_path=case_log_path,
                        entry_override=scapy_entry,
                    )
                elif host_norm == "scapy_builtin":
                    case_log_path = None
                    if packet_logs_dir is not None:
                        case_log_path = Path(packet_logs_dir) / "cases" / f"{exp.case}.jsonl"
                    out_text, scapy_has_layer = _run_scapy_builtin_case_with_meta(
                        protocol=proto,
                        filter_name=str(filter_name or proto),
                        pcap=exp.pcap,
                        frame_no=exp.frame_no,
                        case_name=exp.case,
                        log_path=case_log_path,
                        abnormal_policy=scapy_policy,
                    )
                elif host_norm == "zeek_spicy":
                    if plugin_dir is None:
                        raise ValueError("plugin_dir is required for zeek_spicy exp_constraint validation")
                    out_text, zeek_parse_ok = _run_zeek_spicy_case(
                        plugin_dir=Path(plugin_dir),
                        filter_name=str(filter_name or proto),
                        pcap=exp.pcap,
                        frame_no=exp.frame_no,
                        codegen_profile=zeek_codegen_profile,
                    )
                elif host_norm == "zeek_builtin":
                    out_text, zeek_parse_ok = _run_zeek_builtin_case(
                        pcap=exp.pcap,
                        filter_name=str(filter_name or proto),
                    )
                else:
                    raise ValueError(f"Unsupported host for exp_constraint validation: {host_norm}")
            except Exception as exc:
                fn += 1
                positive_results.append(
                    CaseResult(
                        case=exp.case,
                        ok=False,
                        reason=f"abnormal (runner error: {type(exc).__name__}: {exc})",
                        pcap=str(exp.pcap),
                        frame_no=int(exp.frame_no),
                        target_constraints=(),
                    )
                )
                continue

            if host_norm in {"wireshark_c", "wireshark_lua"}:
                runtime_failure = ""
                if _ws_tshark_timed_out(out_text or "", tshark_rc):
                    runtime_failure = _ws_runtime_diagnostic(out_text or "") or "tshark timeout"
                elif _ws_has_dissector_bug(out_text or ""):
                    diagnostic = _ws_runtime_diagnostic(out_text or "")
                    runtime_failure = f"dissector bug: {diagnostic}" if diagnostic else "dissector bug"
                elif tshark_rc is not None:
                    try:
                        if int(tshark_rc) != 0:
                            diagnostic = _ws_runtime_diagnostic(out_text or "")
                            runtime_failure = (
                                f"tshark_rc={tshark_rc}; {diagnostic}" if diagnostic else f"tshark_rc={tshark_rc}"
                            )
                    except Exception:
                        runtime_failure = "tshark_rc=unknown"
                if runtime_failure:
                    fn += 1
                    positive_results.append(
                        CaseResult(
                            case=exp.case,
                            ok=False,
                            reason=f"abnormal (runtime failure: {runtime_failure})",
                            pcap=str(exp.pcap),
                            frame_no=int(exp.frame_no),
                            target_constraints=(),
                        )
                    )
                    continue

            abnormal, why = _binary_is_abnormal(
                host=host_norm,
                out_text=out_text,
                tshark_rc=tshark_rc,
                filter_name=abnormal_filter_name,
                scapy_has_layer=scapy_has_layer,
                zeek_parse_ok=zeek_parse_ok,
                strict_tag=False,
            )
            if abnormal:
                fn += 1
                positive_results.append(
                    CaseResult(
                        case=exp.case,
                        ok=False,
                        reason=f"abnormal ({why})",
                        pcap=str(exp.pcap),
                        frame_no=int(exp.frame_no),
                        target_constraints=(),
                    )
                )
            else:
                tp += 1
                positive_results.append(
                    CaseResult(
                        case=exp.case,
                        ok=True,
                        reason=f"normal ({why})",
                        pcap=str(exp.pcap),
                        frame_no=int(exp.frame_no),
                        target_constraints=(),
                    )
                )

    results = [replace(case,
        anomaly_detected=observations_by_case[case.case]["anomaly_detected"],
        target_diagnostic_matched=observations_by_case[case.case]["target_diagnostic_matched"],
        independently_verified=(observations_by_case[case.case]["target_diagnostic_matched"]
            and observations_by_case[case.case]["counterexample_qualification"].get("independent_target_test") is True),
        observations=observations_by_case[case.case]) if case.case in observations_by_case else case
        for case in results]
    total = len(expectations)
    coverage = (float(success) / float(total)) if total else 0.0
    suite_result = SuiteResult(
        protocol=proto,
        host=host_norm,
        mode=mode_norm,
        success=success,
        fail=fail,
        total=total,
        coverage=coverage,
        suite_dir=str(suite),
        mapping_path=str(mapping),
        results=tuple(results),
        positive_results=tuple(positive_results),
        tp=int(tp) if want_confusion else None,
        fp=int(fp) if want_confusion else None,
        tn=int(tn) if want_confusion else None,
        fn=int(fn) if want_confusion else None,
        positive_total=len(positive_expectations) if want_confusion else None,
        negative_total=len(expectations) if want_confusion else None,
        acceptance_semantics="exact_target_diagnostic" if binary_require_exact_target_tags or mode_norm == "per_constraint"
            else "binary_detection",
        negative_detection_success=sum(case.anomaly_detected is True for case in results),
        negative_target_diagnostic_success=sum(case.target_diagnostic_matched is True for case in results),
        negative_independently_verified=sum(case.independently_verified for case in results),
        negative_isolation_unverified=sum((case.observations or {}).get("counterexample_qualification", {}).get(
            "independent_target_test") is not True for case in results),
    )
    if zeek_plugin_tmp is not None:
        zeek_plugin_tmp.cleanup()
    return suite_result
