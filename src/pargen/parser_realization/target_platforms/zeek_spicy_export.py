from __future__ import annotations

"""Stage C exporter for Zeek+Spicy (zeek_spicy host).

This module exports a ParGen-compatible `tool.json` by running `spicy-dump`
over bytes extracted from the dataset PCAP (e.g., TCP/UDP payload, or TCP header
bytes when the codegen profile requests `pcap_layer=tcp_header`).

We rely on `spicy-dump -J -Q` which emits JSON plus a `__offsets` map of
stream positions, enabling deterministic reconstruction of leaf field spans.
"""

import json
import hashlib
import os
import re
import shutil
import signal
import subprocess
import time
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pargen.parser_realization.semantic_stage.gt_from_tree import (
    extract_ip_payloads_from_pcap,
    extract_l3_packets_from_pcap,
    extract_tcp_headers_from_pcap,
    extract_tcp_udp_payloads_with_ports_from_pcap,
    iter_pcap_ip_payloads,
    iter_pcap_l3_packets,
    iter_pcap_tcp_headers,
    iter_pcap_tcp_udp_payloads_with_ports,
)

from .bit_utils import ceil_div, safe_value_int_msb0
from .extraction_profile import infer_ip_proto, infer_pcap_layer
from .wireshark_export import ExportTooljsonResult

_LIST_PARSERS_CACHE: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], List[str]] = {}
_LIST_PARSERS_LOCK = threading.Lock()
_HLTO_COMPILE_LOCK = threading.Lock()
_SPICYC_VERSION: Optional[str] = None


def _strip_bs_constraint_throws_for_semantic_export(text: str) -> str:
    """Remove ParGen runtime constraint throws for semantic-only exports.

    Semantic field extraction and exp_constraint validation are separate checks.
    A generated runtime constraint may be misplaced while the field parser is
    still useful; in semantic mode we therefore neutralize only tagged
    `BS_CONSTRAINT` throws and leave other parser errors intact.
    """

    if "BS_CONSTRAINT" not in text:
        return text

    # if (...) { throw "BS_CONSTRAINT ..."; }
    text = re.sub(
        r'(?m)^[ \t]*if\s*\([^\n]*\)\s*\{\s*throw\s+"BS_CONSTRAINT[^"]*";\s*\}[ \t]*(?://[^\n]*)?\n?',
        "",
        text,
    )
    # if (...)\n    throw "BS_CONSTRAINT ...";
    text = re.sub(
        r'(?m)^[ \t]*if\s*\([^\n]*\)\s*\n[ \t]*throw\s+"BS_CONSTRAINT[^"]*";[ \t]*(?://[^\n]*)?\n?',
        "",
        text,
    )
    # Standalone tagged throws. Keep non-ParGen throws untouched.
    text = re.sub(
        r'(?m)^[ \t]*throw\s+"BS_CONSTRAINT[^"]*";[ \t]*(?://[^\n]*)?\n?',
        "",
        text,
    )
    return text


def _copy_plugin_without_bs_constraint_throws(src_dir: Path) -> Tuple[tempfile.TemporaryDirectory, Path, int]:
    tmp = tempfile.TemporaryDirectory(prefix="pargen_zeek_spicy_semantic_")
    dst_dir = Path(tmp.name) / src_dir.name
    shutil.copytree(src_dir, dst_dir)

    changed = 0
    for path in sorted(dst_dir.rglob("*.spicy")):
        if not path.is_file():
            continue
        try:
            old = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        new = _strip_bs_constraint_throws_for_semantic_export(old)
        if new != old:
            path.write_text(new, encoding="utf-8")
            changed += 1
    return tmp, dst_dir, changed


def _run_capture(
    cmd: Sequence[str],
    *,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Tuple[int, str, str, bool]:
    """
    Run a command and capture stdout/stderr.

    Returns: (rc, stdout, stderr, timed_out).
    Uses a new process group/session so we can reliably kill the whole tree on timeout.

    NOTE: Kept as a module-level helper so tests can patch it.
    """

    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return int(proc.returncode or 0), stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except Exception:
            stdout, stderr = "", ""
        return -1, stdout or "", stderr or "", True


def _spicyc_version() -> str:
    global _SPICYC_VERSION
    if _SPICYC_VERSION is not None:
        return _SPICYC_VERSION
    try:
        proc = subprocess.run(
            ["spicyc", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        _SPICYC_VERSION = out.strip().splitlines()[0].strip() if out.strip() else "spicyc"
    except Exception:
        _SPICYC_VERSION = "spicyc"
    return _SPICYC_VERSION


def _hlto_cache_key(*, inputs: Sequence[Path], library_paths: Sequence[Path]) -> str:
    """
    Compute a stable cache key for precompiled HLTO artifacts.

    We hash the spicyc version, library path list, and the contents of each input.
    """
    h = hashlib.sha256()
    h.update(_spicyc_version().encode("utf-8", errors="ignore") + b"\0")
    for lp in library_paths:
        h.update(str(lp).encode("utf-8", errors="ignore") + b"\0")
    for p in inputs:
        h.update(str(p).encode("utf-8", errors="ignore") + b"\0")
        try:
            h.update(p.read_bytes())
        except Exception:
            # If we can't read a file, keep the cache conservative by mixing in a marker.
            h.update(b"<unreadable>\0")
    return h.hexdigest()


def _infer_ports_from_profile(codegen_profile: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    try:
        reg = (codegen_profile or {}).get("zeek_spicy", {}).get("registration", {})
    except Exception:
        reg = {}
    if not isinstance(reg, dict):
        return None

    ports_raw = reg.get("ports")
    if ports_raw is None and reg.get("port") is not None:
        ports_raw = [reg.get("port")]

    ports: List[int] = []
    if isinstance(ports_raw, (list, tuple)):
        for item in ports_raw:
            try:
                p = int(str(item).strip())
            except Exception:
                continue
            if 0 < p <= 65535:
                ports.append(p)
    elif isinstance(ports_raw, (int, str)):
        try:
            p = int(str(ports_raw).strip())
            if 0 < p <= 65535:
                ports.append(p)
        except Exception:
            ports = []

    ports = sorted(set(ports))
    return ports or None


def _infer_transport_from_profile(codegen_profile: Optional[Dict[str, Any]]) -> str:
    try:
        reg = (codegen_profile or {}).get("zeek_spicy", {}).get("registration", {})
    except Exception:
        reg = {}
    if not isinstance(reg, dict):
        return "auto"
    t = str(reg.get("transport") or "auto").strip().lower()
    if t not in {"tcp", "udp", "auto"}:
        return "auto"
    return t


def _infer_module_name_from_profile(codegen_profile: Optional[Dict[str, Any]]) -> Optional[str]:
    try:
        reg = (codegen_profile or {}).get("zeek_spicy", {}).get("registration", {})
    except Exception:
        reg = {}
    if not isinstance(reg, dict):
        return None
    for key in ("module_name", "analyzer_name", "namespace", "module"):
        v = reg.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _infer_direction_units_from_profile(codegen_profile: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    try:
        direction = (codegen_profile or {}).get("zeek_spicy", {}).get("direction", {})
    except Exception:
        direction = {}
    if not isinstance(direction, dict):
        return None, None
    strategy = str(direction.get("strategy") or "").strip().lower()
    if strategy not in {"originator_responder", "originator-responder", "originator/responder"}:
        return None, None
    o = direction.get("originator_entry_unit")
    r = direction.get("responder_entry_unit")
    o_name = o.strip() if isinstance(o, str) and o.strip() else None
    r_name = r.strip() if isinstance(r, str) and r.strip() else None
    return o_name, r_name


def _infer_entry_unit_from_profile(codegen_profile: Optional[Dict[str, Any]]) -> Optional[str]:
    try:
        reg = (codegen_profile or {}).get("zeek_spicy", {}).get("registration", {})
    except Exception:
        reg = {}
    if isinstance(reg, dict):
        for key in ("entry_unit", "entry", "parser", "unit", "entry_type"):
            v = reg.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    try:
        extraction = (codegen_profile or {}).get("extraction", {})
    except Exception:
        extraction = {}
    if isinstance(extraction, dict):
        for key in ("entry_unit", "entry", "parser", "unit", "entry_type"):
            v = extraction.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _normalize_unit(name: Optional[str]) -> Optional[str]:
    s = (name or "").strip()
    return s or None


def _ports_from_tcp_header(buf: bytes) -> Tuple[int, int]:
    if len(buf) < 4:
        return 0, 0
    try:
        sport = int.from_bytes(buf[0:2], "big", signed=False)
        dport = int.from_bytes(buf[2:4], "big", signed=False)
    except Exception:
        return 0, 0
    return int(sport), int(dport)


def _resolve_spicy_dump(*, spicy_dump_path: Optional[Path], spicyz_path: Optional[Path]) -> Optional[Path]:
    if spicy_dump_path is not None:
        try:
            if spicy_dump_path.exists():
                return spicy_dump_path.resolve()
        except OSError:
            return spicy_dump_path
        return None

    if spicyz_path is not None:
        try:
            cand = spicyz_path.with_name("spicy-dump")
            if cand.exists():
                return cand.resolve()
        except OSError:
            pass

    found = shutil.which("spicy-dump")
    if not found:
        return None
    try:
        return Path(found).resolve()
    except OSError:
        return Path(found)


def _sorted_existing(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        try:
            if p.is_file():
                out.append(p)
        except OSError:
            continue
    out.sort(key=lambda x: str(x))
    return out


def _parse_list_parsers(stdout: str) -> List[str]:
    parsers: List[str] = []
    for line in (stdout or "").splitlines():
        s = line.strip()
        if not s or s.lower().startswith("available parsers"):
            continue
        parsers.append(s)
    # Preserve order but de-dup.
    seen: set[str] = set()
    out: List[str] = []
    for p in parsers:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _select_parser_name(
    *,
    available: Sequence[str],
    module_name: Optional[str],
    unit_name: Optional[str],
) -> Optional[str]:
    unit = _normalize_unit(unit_name)
    if not unit:
        return None

    candidates: List[str] = []
    mod = (module_name or "").strip()
    if mod:
        candidates.append(f"{mod}::{unit}")

    for p in available:
        if p == unit or p.endswith(f"::{unit}"):
            candidates.append(p)

    for cand in candidates:
        if cand in available:
            return cand
    return None


def _select_default_parser_name(
    *,
    available: Sequence[str],
    module_name: Optional[str],
    filter_name: str,
    entry_unit: Optional[str] = None,
) -> Optional[str]:
    if not available:
        return None

    explicit = _select_parser_name(available=available, module_name=module_name, unit_name=entry_unit)
    if explicit:
        return explicit

    def _local(p: str) -> str:
        return str(p or "").split("::")[-1]

    proto = re.sub(r"[^a-z0-9]+", "", str(filter_name or "").lower())
    mod = re.sub(r"[^a-z0-9]+", "", str(module_name or "").lower())
    desired_names = {
        "message",
        "packet",
        "datagram",
        "frame",
        "segment",
        "pdu",
    }
    for base in [proto, mod]:
        if not base:
            continue
        desired_names.update(
            {
                base,
                f"{base}message",
                f"{base}packet",
                f"{base}datagram",
                f"{base}frame",
                f"{base}segment",
                f"{base}pdu",
            }
        )

    child_markers = (
        "header",
        "label",
        "labels",
        "option",
        "options",
        "field",
        "record",
        "rdata",
        "question",
        "answer",
        "section",
        "sections",
        "body",
        "payload",
        "value",
        "values",
        "item",
        "entry",
    )
    root_suffixes = ("message", "packet", "datagram", "frame", "segment", "pdu")

    scored: List[Tuple[int, int, str]] = []
    for idx, parser in enumerate(available):
        local_raw = _local(parser)
        local = re.sub(r"[^a-z0-9]+", "", local_raw.lower())
        score = 0
        if module_name and str(parser).startswith(f"{module_name}::"):
            score += 10
        if local in desired_names:
            score += 100
        if any(local.endswith(suffix) for suffix in root_suffixes):
            score += 45
        if any(marker in local for marker in child_markers):
            score -= 30
        # Generated entry units are usually aggregate roots and tend to have
        # longer names than helper units such as Header/Label/Option.
        score += min(len(local_raw), 32)
        scored.append((score, -idx, parser))

    scored.sort(reverse=True)
    return scored[0][2] if scored else available[0]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in sorted(value.items(), key=lambda kv: str(kv[0])):
            # Drop nested offsets from the value payload to keep tool.json compact.
            if str(k) == "__offsets":
                continue
            out[str(k)] = _json_safe(v)
        return out
    try:
        return repr(value)
    except Exception:
        return str(type(value))


@dataclass(frozen=True)
class _FieldSpan:
    name: str
    abbrev: str
    offset: int
    length: int
    raw: str
    value: Any


@dataclass(frozen=True)
class _BitfieldDef:
    size_bits: int
    bit_order: str  # MSB0 | LSB0 (Spicy semantics)
    fields: Dict[str, Tuple[int, int]]  # inclusive bit ranges in Spicy numbering


_BITFIELD_DEF_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*bitfield\(\s*(?P<size>\d+)\s*\)\s*\{(?P<body>.*?)\}\s*(?P<attrs>(?:\s*&[^;]+)*)\s*;",
    re.DOTALL,
)
_BITFIELD_SUB_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<start>\d+)(?:\s*\.\.\s*(?P<end>\d+))?(?:\s*&[^;]+)*\s*;",
    re.MULTILINE,
)
_BIT_ORDER_ATTR_RE = re.compile(r"&bit-order\s*=\s*spicy::BitOrder::(MSB0|LSB0)")


def _strip_spicy_line_comments(text: str) -> str:
    # Spicy uses `#` for line comments.
    return re.sub(r"#.*$", "", text or "", flags=re.MULTILINE)


def _parse_spicy_bitfields_from_text(text: str) -> Dict[str, _BitfieldDef]:
    cleaned = _strip_spicy_line_comments(text)
    defs: Dict[str, _BitfieldDef] = {}
    duplicates: set[str] = set()
    for m in _BITFIELD_DEF_RE.finditer(cleaned):
        name = str(m.group("name") or "").strip()
        if not name:
            continue
        try:
            size_bits = int(m.group("size"))
        except Exception:
            continue
        if size_bits <= 0:
            continue

        attrs = str(m.group("attrs") or "")
        bit_order = "LSB0"
        mo = _BIT_ORDER_ATTR_RE.search(attrs)
        if mo:
            bit_order = mo.group(1)

        body = str(m.group("body") or "")
        fields: Dict[str, Tuple[int, int]] = {}
        for fm in _BITFIELD_SUB_RE.finditer(body):
            sub = str(fm.group("name") or "").strip()
            if not sub:
                continue
            try:
                start = int(fm.group("start"))
                end_raw = fm.group("end")
                end = int(end_raw) if end_raw is not None else start
            except Exception:
                continue
            if end < start:
                start, end = end, start
            fields[sub] = (int(start), int(end))

        if not fields:
            continue

        if name in defs:
            duplicates.add(name)
        else:
            defs[name] = _BitfieldDef(size_bits=int(size_bits), bit_order=str(bit_order), fields=fields)

    for d in duplicates:
        defs.pop(d, None)
    return defs


def _parse_spicy_bitfields(paths: Sequence[Path]) -> Dict[str, _BitfieldDef]:
    out: Dict[str, _BitfieldDef] = {}
    duplicates: set[str] = set()
    for p in paths:
        try:
            if p.suffix != ".spicy" or not p.is_file():
                continue
        except OSError:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        defs = _parse_spicy_bitfields_from_text(text)
        for k, v in defs.items():
            if k in out:
                duplicates.add(k)
                continue
            out[k] = v
    for d in duplicates:
        out.pop(d, None)
    return out


def _bit_range_to_msb0(*, size_bits: int, bit_order: str, start: int, end: int) -> Tuple[int, int]:
    """Convert an inclusive bit range in Spicy numbering to MSB0 numbering (big-endian)."""
    s = int(start)
    e = int(end)
    if e < s:
        s, e = e, s
    n = int(size_bits)
    if n <= 0:
        return s, e
    bo = str(bit_order or "LSB0").upper()
    if bo == "MSB0":
        return s, e
    if bo == "LSB0":
        # In big-endian byte order, LSB0 bit i corresponds to MSB0 bit (n-1-i).
        return (n - 1 - e), (n - 1 - s)
    return s, e


def _cover_bytes_for_bits(bit_start: int, size_bits: int) -> Tuple[int, int]:
    start_byte = int(bit_start) // 8
    end_bit = int(bit_start) + int(size_bits)
    end_byte = int(ceil_div(end_bit, 8))
    return start_byte, max(0, int(end_byte - start_byte))


def _fields_from_spans(
    spans: Sequence[_FieldSpan],
    payload: bytes,
    *,
    bitfield_defs: Dict[str, _BitfieldDef],
    field_name_mode: str = "tail",
) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []

    name_mode = str(field_name_mode or "tail").strip().lower()
    if name_mode not in {"full", "tail"}:
        name_mode = "tail"

    def _emit_name(name: str) -> str:
        raw = str(name or "").strip()
        if not raw or name_mode == "full":
            return raw
        # Default: keep only the suffix after the last dot, to make field naming
        # consistent with hosts that do not expose hierarchical paths.
        return raw.split(".")[-1]

    for s in spans:
        base_name = str(s.name).split(".")[-1]
        bf = bitfield_defs.get(base_name)
        if bf is not None:
            # Named bitfield: spicy-dump may report either:
            #   1) a single byte/word span with a dict of labeled bit values; OR
            #   2) a single byte/word span with an int value (older versions / some builds).
            #
            # In both cases we can deterministically expand into leaf bit spans using the
            # bitfield layout parsed from the Spicy source.
            sub_value: Dict[str, Any] = s.value if isinstance(s.value, dict) else {}

            container_bit_start = int(s.offset) * 8
            # Deterministic order: sort by MSB0 start bit, then name.
            order: List[Tuple[int, str, int, int]] = []
            for sub, (st, en) in bf.fields.items():
                ms, me = _bit_range_to_msb0(size_bits=bf.size_bits, bit_order=bf.bit_order, start=st, end=en)
                order.append((int(ms), str(sub), int(ms), int(me)))
            order.sort(key=lambda x: (x[0], x[1]))

            for _, sub, ms, me in order:
                size_bits = int(me - ms + 1)
                if size_bits <= 0:
                    continue
                bit_start = int(container_bit_start + ms)
                off, ln = _cover_bytes_for_bits(bit_start, size_bits)
                raw_hex = payload[off : off + ln].hex() if ln > 0 else ""
                try:
                    value_int = int(safe_value_int_msb0(payload, bit_start, size_bits))
                except Exception:
                    value_int = None

                fields.append(
                    {
                        "name": _emit_name(f"{s.name}.{sub}"),
                        "abbrev": f"{s.abbrev}.{sub}",
                        "offset": int(off),
                        "length": int(ln),
                        "raw": raw_hex,
                        "bit_start": int(bit_start),
                        "size_bits": int(size_bits),
                        "bit_order": "MSB0",
                        "value_int": value_int,
                        "value": _json_safe(sub_value.get(sub, value_int)),
                    }
                )
            continue

        # Default: byte-span field.
        bit_start = int(s.offset) * 8
        size_bits = int(s.length) * 8
        try:
            value_int = int(safe_value_int_msb0(payload, bit_start, size_bits))
        except Exception:
            value_int = None

        fields.append(
            {
                "name": _emit_name(str(s.name)),
                "abbrev": s.abbrev,
                "offset": int(s.offset),
                "length": int(s.length),
                "raw": s.raw,
                "bit_start": int(bit_start),
                "size_bits": int(size_bits),
                "bit_order": "MSB0",
                "value_int": value_int,
                "value": s.value,
            }
        )

    # De-dup for stability.
    seen: set[Tuple[int, int, int, str]] = set()
    out: List[Dict[str, Any]] = []
    for f in sorted(fields, key=lambda x: (int(x.get("bit_start") or 0), int(x.get("size_bits") or 0), str(x.get("name") or ""))):
        try:
            key = (
                int(f.get("bit_start") or 0),
                int(f.get("size_bits") or 0),
                int(f.get("value_int") or 0),
                str(f.get("name") or ""),
            )
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _extract_leaf_spans(obj: Any, payload: bytes, *, prefix: str = "", base_offset: int = 0) -> List[_FieldSpan]:
    if not isinstance(obj, dict):
        return []
    offsets = obj.get("__offsets")
    if not isinstance(offsets, dict):
        return []

    def _descendant_relative_end(value: Any) -> int:
        best = 0
        if isinstance(value, dict):
            child_offsets = value.get("__offsets")
            if isinstance(child_offsets, dict):
                for ck, cv in child_offsets.items():
                    if str(ck) == "self" or not isinstance(cv, dict):
                        continue
                    try:
                        cs = int(cv.get("start"))
                        ce = int(cv.get("end"))
                    except Exception:
                        continue
                    if ce > cs:
                        best = max(best, int(ce))
            for ck, cv in value.items():
                if str(ck) == "__offsets":
                    continue
                best = max(best, _descendant_relative_end(cv))
        elif isinstance(value, list):
            for item in value:
                best = max(best, _descendant_relative_end(item))
        return int(best)

    def _range(name: str) -> Optional[Tuple[int, int]]:
        off_obj = offsets.get(name)
        if not isinstance(off_obj, dict):
            return None
        try:
            start = int(off_obj.get("start"))
            end = int(off_obj.get("end"))
        except Exception:
            return None
        return start, end

    def _lookup_offset_key(name: str) -> Optional[str]:
        if name in offsets:
            return name
        want = str(name or "").lower()
        for k in offsets.keys():
            if str(k).lower() == want:
                return str(k)
        return None

    # Suppress intermediate bytes buffers used only for &parse-from patterns.
    # Common pattern:
    #   pdu_bytes: bytes &size=...;
    #   pdu: PDU &parse-from=self.pdu_bytes;
    # Offsets from spicy-dump for `pdu` are relative to the buffer, so exporting `pdu_bytes`
    # as an additional leaf creates systematic false positives.
    suppress_bytes: set[str] = set()
    for k in offsets.keys():
        key = str(k)
        if key == "self":
            continue

        base_key: Optional[str] = None
        if key.endswith("_bytes"):
            base_key = key[: -len("_bytes")]
        else:
            # Common variant: `<name>_bytes2`, `<name>_bytes3`, ...
            pos = key.rfind("_bytes")
            if pos != -1 and pos + len("_bytes") < len(key) and key[pos + len("_bytes") :].isdigit():
                base_key = key[:pos]

        if not base_key:
            continue
        child = obj.get(base_key)
        if not isinstance(child, dict):
            for obj_key, obj_value in obj.items():
                if str(obj_key).lower() == base_key.lower():
                    child = obj_value
                    break
        buf_rng = _range(key)
        if buf_rng is None:
            continue
        buf_len = int(buf_rng[1] - buf_rng[0])

        child_len: Optional[int] = None
        if isinstance(child, dict) and isinstance(child.get("__offsets"), dict):
            child_offsets = child.get("__offsets")
            if isinstance(child_offsets, dict):
                self_obj = child_offsets.get("self")
                if isinstance(self_obj, dict):
                    try:
                        ss = int(self_obj.get("start"))
                        se = int(self_obj.get("end"))
                        if se > ss:
                            child_len = int(se - ss)
                    except Exception:
                        child_len = None
        else:
            child_key = _lookup_offset_key(str(base_key))
            if child_key is not None:
                try:
                    cr = _range(child_key)
                    if cr is not None:
                        child_len = int(cr[1] - cr[0])
                except Exception:
                    child_len = None
        if child_len is None:
            child_rng = _range(base_key)
            if child_rng is None:
                continue
            child_len = int(child_rng[1] - child_rng[0])

        if buf_len > 0 and child_len > 0 and buf_len == child_len:
            suppress_bytes.add(key)

    # Some generated grammars use structural names that do not share the buffer
    # prefix, e.g. `body_bytes` parsed by `ie_seq`, or `payload_bytes` parsed by
    # `items`.  These buffers are implementation plumbing; unlike `value_bytes`
    # in TLV grammars they are not protocol fields and should not create FPs.
    structural_buffer_names = {"body_bytes", "payload_bytes", "pdu_bytes", "message_bytes", "packet_bytes", "seid_bytes"}
    for key in list(offsets.keys()):
        key_s = str(key)
        if key_s in suppress_bytes or key_s.lower() not in structural_buffer_names:
            continue
        rng = _range(key_s)
        if rng is None:
            continue
        buf_len = int(rng[1] - rng[0])
        if buf_len <= 0:
            continue
        for child_key, child_value in obj.items():
            if str(child_key) in {"__offsets", key_s}:
                continue
            if not isinstance(child_value, dict) or not isinstance(child_value.get("__offsets"), dict):
                continue
            child_offsets = child_value.get("__offsets")
            if not isinstance(child_offsets, dict):
                continue
            self_obj = child_offsets.get("self")
            if not isinstance(self_obj, dict):
                child_len = _descendant_relative_end(child_value)
            else:
                try:
                    child_len = int(self_obj.get("end")) - int(self_obj.get("start"))
                except Exception:
                    child_len = _descendant_relative_end(child_value)
                if child_len <= 0:
                    child_len = _descendant_relative_end(child_value)
            if child_len > 0 and buf_len >= child_len:
                suppress_bytes.add(key_s)
                break

    spans: List[_FieldSpan] = []
    # Sort keys for determinism; Spicy itself is deterministic, but JSON dict ordering is not a contract.
    for key in sorted(offsets.keys(), key=lambda x: str(x)):
        name = str(key)
        if name == "self":
            continue
        if name.startswith("_anon"):
            # Spicy emits anonymous helper ranges for some arrays/switches. They
            # are parser implementation artifacts, not protocol fields.
            continue
        if name in suppress_bytes:
            continue
        off_obj = offsets.get(key)
        if not isinstance(off_obj, dict):
            continue
        try:
            start = int(off_obj.get("start"))
            end = int(off_obj.get("end"))
        except Exception:
            continue
        child = obj.get(key)
        if end <= start and isinstance(child, dict) and isinstance(child.get("__offsets"), dict):
            self_obj = child.get("__offsets", {}).get("self")
            if isinstance(self_obj, dict):
                try:
                    child_start = int(self_obj.get("start"))
                    child_end = int(self_obj.get("end"))
                    if child_end > child_start:
                        start = child_start
                        end = child_end
                except Exception:
                    pass
            if end <= start:
                # Spicy represents switch arms parsed through `&parse-from=value_bytes`
                # as a zero-width wrapper (`body: { self: 0..0, variant: 0..N }`).
                # Treat the wrapper as structural and recurse into the descendants
                # instead of dropping the already parsed protocol fields.
                desc_end = _descendant_relative_end(child)
                if desc_end > 0:
                    start = 0
                    end = int(desc_end)
        if start < 0 or end <= start:
            continue

        # Primitive fields can also be parsed from a bytes buffer:
        #   seid_bytes: bytes &size=8;
        #   SEID: uint64 &parse-from=self.seid_bytes;
        # spicy-dump may then report `SEID` at 0..8 relative to the buffer
        # while the backing buffer is at the true packet offset.  Anchor such
        # fields to their case-insensitive `<name>_bytes` sibling.
        primitive_src_key = _lookup_offset_key(f"{name}_bytes")
        if primitive_src_key is not None:
            rng = _range(primitive_src_key)
            if rng is not None:
                src_len = int(rng[1] - rng[0])
                if int(start) == 0 and int(end - start) == src_len:
                    start, end = int(rng[0]), int(rng[1])

        abs_start = int(base_offset) + int(start)
        abs_end = int(base_offset) + int(end)
        if abs_start < 0 or abs_end <= abs_start:
            continue
        if abs_start > len(payload):
            continue
        abs_end = min(abs_end, len(payload))
        length = int(abs_end - abs_start)
        if length <= 0:
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        # Recurse into nested units to emit leaf-only spans.
        if isinstance(child, dict) and isinstance(child.get("__offsets"), dict):
            child_offsets = child.get("__offsets")
            child_self_start = 0
            child_len = int(end - start)
            if isinstance(child_offsets, dict):
                self_obj = child_offsets.get("self")
                if isinstance(self_obj, dict):
                    try:
                        child_self_start = int(self_obj.get("start"))
                        child_self_end = int(self_obj.get("end"))
                        if child_self_end > child_self_start:
                            child_len = int(child_self_end - child_self_start)
                    except Exception:
                        child_self_start = 0

            # Spicy offsets for nested objects are usually in the same coordinate system as
            # the parent payload. However, when `&parse-from` is used, the nested unit's
            # offsets can be relative to the parse-from buffer (starting at 0) while the
            # nested `self` range may still be reported in stream coordinates.
            #
            # We infer whether child offsets are relative by checking if any child field's
            # start is < child_self_start.
            child_origin = int(child_self_start)
            if isinstance(child_offsets, dict) and child_origin > 0:
                min_child_start: Optional[int] = None
                for ck, cv in child_offsets.items():
                    if str(ck) == "self" or not isinstance(cv, dict):
                        continue
                    try:
                        cs = int(cv.get("start"))
                        ce = int(cv.get("end"))
                    except Exception:
                        continue
                    if ce <= cs:
                        continue
                    if min_child_start is None or cs < min_child_start:
                        min_child_start = cs
                if min_child_start is not None and min_child_start < child_origin:
                    child_origin = 0

            # Default: align child-origin with this field's start.
            nested_base_offset = int(base_offset) + int(start) - int(child_origin)

            # If the child was parsed via `&parse-from`, spicy-dump offsets are relative to the
            # buffer, so the parent may show child.start==0. Prefer an explicit `<name>_bytes`
            # source buffer when available, and as a fallback try to infer a unique bytes-like
            # sibling of the same length.
            src_rng: Optional[Tuple[int, int]] = None
            src_base = f"{name}_bytes"
            candidates: List[str] = []
            for k2 in offsets.keys():
                s2 = str(k2)
                s2_l = s2.lower()
                src_base_l = src_base.lower()
                if s2_l == src_base_l:
                    candidates.append(s2)
                    continue
                if s2_l.startswith(src_base_l) and s2_l[len(src_base_l) :].isdigit():
                    candidates.append(s2)

            # Common TLV shape: `value_bytes: bytes &size=...; body:
            # Variant(...) &parse-from=self.value_bytes;`. The child is not
            # named `value`, so `<name>_bytes` would miss the backing buffer.
            if name.lower() in {"body", "variant", "variants", "parsed", "payload"}:
                for alt_base in ("value_bytes", "body_bytes", "payload_bytes", "data_bytes"):
                    for k2 in offsets.keys():
                        s2 = str(k2)
                        s2_l = s2.lower()
                        alt_l = alt_base.lower()
                        if s2_l == alt_l or (s2_l.startswith(alt_l) and s2_l[len(alt_l) :].isdigit()):
                            candidates.append(s2)

            if not candidates and name.lower() not in {"ie_seq", "items", "sequence"}:
                # Conservative generic fallback for `&parse-from`: when there
                # is exactly one structural bytes sibling large enough to back
                # the child, use it as the coordinate origin.  Exclude
                # `value_bytes` here because TLV value spans are often real
                # protocol fields and can coexist with a selected body.
                fallback: List[str] = []
                for k2 in offsets.keys():
                    s2 = str(k2)
                    if not s2.lower().endswith("_bytes") or s2.lower() == "value_bytes":
                        continue
                    rng = _range(s2)
                    if rng is None:
                        continue
                    if int(rng[1] - rng[0]) >= int(child_len):
                        fallback.append(s2)
                if len(fallback) == 1:
                    candidates.extend(fallback)

            def _cand_key(s: str) -> Tuple[int, int]:
                if s.lower() == src_base.lower():
                    return (0, 0)
                suffix = s[len(src_base) :]
                try:
                    n = int(suffix) if suffix else 0
                except Exception:
                    n = 0
                return (1, n)

            for cand in sorted(candidates, key=_cand_key):
                rng = _range(cand)
                if rng is None:
                    continue
                # The backing `value_bytes` buffer may be larger than the
                # selected child unit when the generated parser only decodes
                # the fixed prefix of a variable/grouped IE body. It is still
                # the correct coordinate origin for that child.
                if int(rng[1] - rng[0]) >= int(child_len):
                    src_rng = rng
                    break

            if src_rng is not None:
                nested_base_offset = int(base_offset) + int(src_rng[0]) - int(child_origin)

            nested = _extract_leaf_spans(child, payload, prefix=full_name, base_offset=nested_base_offset)
            if nested:
                spans.extend(nested)
                continue

        # Recurse into arrays of nested units when spicy-dump includes per-element __offsets.
        # Example (DHCP options):
        #   options: Option_Item[] &until=...
        # where each Option_Item element is a dict with its own __offsets (including self).
        if isinstance(child, list):

            nested_any = False
            for elem in child:
                if not isinstance(elem, dict) or not isinstance(elem.get("__offsets"), dict):
                    continue
                # For array elements, spicy-dump typically reports absolute offsets in stream coordinates
                # (element.__offsets.self.start is the element's position in the parent payload), so we
                # can recurse without shifting. If an element appears to use a relative coordinate
                # system (self.start == 0 while the array itself starts later), shift by the array's
                # start offset as a fallback.
                elem_offsets = elem.get("__offsets")
                child_self_start = 0
                if isinstance(elem_offsets, dict):
                    self_obj = elem_offsets.get("self")
                    if isinstance(self_obj, dict):
                        try:
                            child_self_start = int(self_obj.get("start"))
                        except Exception:
                            child_self_start = 0
                nested_base_offset = int(base_offset)
                if child_self_start == 0 and int(start) > 0:
                    nested_base_offset = int(base_offset) + int(start)
                nested = _extract_leaf_spans(elem, payload, prefix=full_name, base_offset=nested_base_offset)
                if nested:
                    spans.extend(nested)
                    nested_any = True
            if nested_any:
                continue

        raw_hex = payload[abs_start:abs_end].hex()
        spans.append(
            _FieldSpan(
                name=full_name,
                abbrev=full_name,
                offset=int(abs_start),
                length=int(length),
                raw=raw_hex,
                value=_json_safe(child),
            )
        )

    # De-dup for stability.
    seen: set[Tuple[int, int, str, str]] = set()
    out: List[_FieldSpan] = []
    for s in sorted(spans, key=lambda x: (x.offset, x.length, x.name, x.raw)):
        k = (int(s.offset), int(s.length), str(s.raw), str(s.name))
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def export_zeek_spicy_tooljson(
    *,
    plugin_src_dir: Path,
    pcap_path: Path,
    filter_name: str,
    out_dir: Path,
    codegen_profile: Optional[Dict[str, Any]] = None,
    frame_index: Optional[Sequence[int]] = None,
    max_packets: int = 0,
    hlto_path: Optional[Path] = None,
    spicyz_path: Optional[Path] = None,
    spicy_dump_path: Optional[Path] = None,
    fail_on_all_parse_failures: bool = False,
    disable_runtime_constraints_for_export: bool = False,
) -> ExportTooljsonResult:
    """Export Zeek+Spicy parsing results as tool.json + tool.txt.

    `packet_no` equals the dataset PCAP frame number; `hex` is the extracted byte
    buffer (e.g., TCP/UDP payload, or TCP header bytes).

    Field offsets/lengths are relative to the extracted bytes start (base_offset=0).
    """
    try:
        plugin_src_dir = plugin_src_dir.resolve()
    except OSError:
        plugin_src_dir = Path(plugin_src_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    tooljson_path = out_dir / "tool.json"
    texttree_path = out_dir / "tool.txt"
    log_path = out_dir / "zeek_spicy_export.log"

    def _write_log(text: str) -> None:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            return

    try:
        log_path.write_text("", encoding="utf-8")
    except Exception:
        pass

    semantic_tmp: Optional[tempfile.TemporaryDirectory] = None
    if bool(disable_runtime_constraints_for_export):
        try:
            semantic_tmp, semantic_plugin_dir, changed = _copy_plugin_without_bs_constraint_throws(plugin_src_dir)
            plugin_src_dir = semantic_plugin_dir.resolve()
            _write_log(
                "[ZeekSpicyExport] semantic_constraint_mode=disabled "
                f"sanitized_spicy_files={int(changed)} temp_plugin_dir={plugin_src_dir}\n"
            )
        except Exception as exc:
            if semantic_tmp is not None:
                semantic_tmp.cleanup()
                semantic_tmp = None
            _write_log(f"[ZeekSpicyExport] WARNING: failed to disable runtime constraints for export: {exc}\n")

    spicy_dump = _resolve_spicy_dump(spicy_dump_path=spicy_dump_path, spicyz_path=spicyz_path)
    if spicy_dump is None:
        msg = "spicy-dump not found. Install Zeek/Spicy tooling or ensure spicy-dump is on PATH."
        _write_log(f"[ZeekSpicyExport] ERROR: {msg}\n")
        raise RuntimeError(msg)

    field_name_mode = str(os.getenv("PARGEN_ZEEK_SPICY_FIELD_NAME_MODE", "tail") or "tail").strip().lower()
    if field_name_mode not in {"full", "tail"}:
        field_name_mode = "tail"
    _write_log(f"[ZeekSpicyExport] field_name_mode={field_name_mode}\n")

    # Prefer compiling from *.spicy sources so we do not depend on any prebuilt HLTO artifacts.
    # However, `spicy-dump` will JIT-compile sources on *every invocation* (no cross-process cache),
    # which can be prohibitively slow for dataset-level evaluation.
    #
    # To keep exports fast and deterministic across protocols, we opportunistically precompile the
    # plugin sources to a cached `.hlto` (via `spicyc -j`) and then run `spicy-dump` against that.
    # If precompilation fails, we fall back to the original source-based behavior.
    # Prefer top-level generated grammars under plugin_src_dir/*.spicy.
    #
    # The Zeek+Spicy plugin skeleton may also ship an `analyzer/<protocol>.spicy` (e.g., a
    # baseline analyzer grammar). Including both can accidentally compile/execute the wrong
    # module due to duplicate module names. For tool.json export we want the Stage C generated
    # grammar first; fall back to recursive discovery only when no top-level grammars exist.
    top_level = [p.resolve() for p in _sorted_existing(plugin_src_dir.glob("*.spicy"))]
    if top_level:
        inputs_spicy = top_level
    else:
        # Refuse fallback to recursive discovery: `analyzer/*.spicy` may contain a baseline
        # grammar with the same module name, which can silently override the Stage C output.
        if hlto_path is None:
            raise RuntimeError(
                "No top-level '*.spicy' found under plugin_src_dir. "
                "Refusing to fallback to analyzer/. Ensure Stage C outputs its generated grammar "
                "at plugin_src_dir/<protocol>.spicy (or pass an explicit hlto_path). "
                f"plugin_src_dir={plugin_src_dir}"
            )
        inputs_spicy = []
    inputs: List[Path] = list(inputs_spicy)
    input_mode = "spicy_src"

    # Explicit HLTO override still takes precedence.
    if (not inputs) and hlto_path is not None:
        try:
            if hlto_path.exists() and hlto_path.is_file():
                inputs = [hlto_path.resolve()]
                input_mode = "hlto"
        except OSError:
            inputs = [Path(hlto_path)]
            input_mode = "hlto"
    if not inputs:
        msg = f"No *.spicy or *.hlto inputs found under: {plugin_src_dir}"
        _write_log(f"[ZeekSpicyExport] ERROR: {msg}\n")
        raise RuntimeError(msg)

    bitfield_defs: Dict[str, _BitfieldDef] = {}
    if input_mode == "spicy_src":
        bitfield_defs = _parse_spicy_bitfields(inputs)
        if bitfield_defs:
            _write_log(f"[ZeekSpicyExport] bitfield_defs={sorted(bitfield_defs.keys())}\n")

    library_paths: List[Path] = []
    if input_mode == "spicy_src":
        library_paths.append(plugin_src_dir)
        analyzer_dir = plugin_src_dir / "analyzer"
        if analyzer_dir.exists():
            try:
                library_paths.append(analyzer_dir.resolve())
            except OSError:
                library_paths.append(analyzer_dir)
        # Also include parent dirs of all inputs, in case sources live under analyzer/.
        for p in inputs:
            try:
                library_paths.append(p.parent)
            except Exception:
                continue
        # De-dup with stable order.
        seen_lp: set[str] = set()
        uniq: List[Path] = []
        for lp in library_paths:
            s = str(lp)
            if s in seen_lp:
                continue
            seen_lp.add(s)
            uniq.append(lp)
        library_paths = uniq

    # Opportunistic HLTO precompilation (performance).
    #
    # `spicy-dump` JIT-compiles `.spicy` sources on *every invocation*, which makes dataset-scale
    # exports unusably slow. We precompile the sources once to a cached `.hlto` via `spicyc -j`,
    # and then run `spicy-dump` against the `.hlto` (fast, no per-packet compile).
    #
    # Disable via PARGEN_ZEEK_SPICY_USE_HLTO=0 if needed.
    if input_mode == "spicy_src" and inputs_spicy:
        raw_use_hlto = os.getenv("PARGEN_ZEEK_SPICY_USE_HLTO")
        use_hlto = True
        if raw_use_hlto is not None:
            use_hlto = str(raw_use_hlto).strip().lower() not in {"", "0", "false", "no", "off"}
        if use_hlto:
            spicyc_path = shutil.which("spicyc")
            if spicyc_path:
                cache_dir = plugin_src_dir / ".pargen_hlto_cache"
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    cache_dir = plugin_src_dir
                key = _hlto_cache_key(inputs=list(inputs_spicy), library_paths=library_paths)
                hlto_cached = cache_dir / f"{key}.hlto"
                try:
                    hlto_cached = hlto_cached.resolve()
                except OSError:
                    hlto_cached = Path(hlto_cached)

                if not hlto_cached.exists():
                    compile_timeout_s = 60.0
                    raw_compile_timeout = os.getenv("PARGEN_ZEEK_SPICY_COMPILE_TIMEOUT_S")
                    if raw_compile_timeout is not None:
                        try:
                            compile_timeout_s = float(str(raw_compile_timeout).strip())
                        except Exception:
                            compile_timeout_s = 60.0

                    with _HLTO_COMPILE_LOCK:
                        if not hlto_cached.exists():
                            tmp_hlto = hlto_cached.with_suffix(".tmp.hlto")
                            # IMPORTANT: compile with `-Q` so the resulting HLTO retains offset
                            # instrumentation (required for `spicy-dump -Q` to emit `__offsets`).
                            cmd = [str(spicyc_path), "-j", "-Q", "-o", str(tmp_hlto)]
                            if library_paths:
                                for lp in library_paths:
                                    cmd += ["--library-path", str(lp)]
                            cmd += [str(p) for p in inputs_spicy]
                            _write_log(f"[ZeekSpicyExport] precompile_hlto=1 cache_key={key}\n")
                            _write_log(f"[ZeekSpicyExport] spicyc_cmd={' '.join(cmd)}\n")
                            rc, out, err, timed_out = _run_capture(
                                cmd,
                                cwd=str(plugin_src_dir),
                                timeout_s=(float(compile_timeout_s) if compile_timeout_s > 0 else None),
                            )
                            if timed_out:
                                _write_log(
                                    f"[ZeekSpicyExport] WARNING: spicyc precompile timed out after {compile_timeout_s:.1f}s\n"
                                )
                            if out:
                                _write_log("\n[spicyc stdout]\n" + out + ("\n" if not out.endswith("\n") else ""))
                            if err:
                                _write_log("\n[spicyc stderr]\n" + err + ("\n" if not err.endswith("\n") else ""))
                            if int(rc) == 0 and tmp_hlto.exists():
                                try:
                                    tmp_hlto.replace(hlto_cached)
                                except Exception:
                                    try:
                                        shutil.copyfile(tmp_hlto, hlto_cached)
                                        tmp_hlto.unlink(missing_ok=True)  # type: ignore[arg-type]
                                    except Exception:
                                        pass
                            else:
                                try:
                                    tmp_hlto.unlink()
                                except Exception:
                                    pass

                if hlto_cached.exists():
                    inputs = [hlto_cached]
                    input_mode = "hlto_cache"
                    _write_log(f"[ZeekSpicyExport] using_hlto_cache={hlto_cached}\n")
            else:
                _write_log("[ZeekSpicyExport] precompile_hlto skipped: spicyc not found on PATH\n")

    profile_transport = _infer_transport_from_profile(codegen_profile)
    profile_ports = _infer_ports_from_profile(codegen_profile)
    module_name = _infer_module_name_from_profile(codegen_profile)
    entry_unit = _infer_entry_unit_from_profile(codegen_profile)
    originator_unit, responder_unit = _infer_direction_units_from_profile(codegen_profile)
    pcap_layer = infer_pcap_layer(codegen_profile, host="zeek_spicy")
    profile_ip_proto = infer_ip_proto(codegen_profile, host="zeek_spicy") if pcap_layer == "ip_payload" else None

    _write_log(f"[ZeekSpicyExport] plugin_src_dir={plugin_src_dir}\n")
    _write_log(f"[ZeekSpicyExport] pcap_path={pcap_path}\n")
    _write_log(f"[ZeekSpicyExport] filter_name={filter_name}\n")
    _write_log(f"[ZeekSpicyExport] spicy_dump={spicy_dump}\n")
    _write_log(f"[ZeekSpicyExport] input_mode={input_mode}\n")
    _write_log(f"[ZeekSpicyExport] inputs={[str(p) for p in inputs]}\n")
    if library_paths:
        _write_log(f"[ZeekSpicyExport] library_paths={[str(p) for p in library_paths]}\n")
    _write_log(f"[ZeekSpicyExport] profile_transport={profile_transport}\n")
    _write_log(f"[ZeekSpicyExport] profile_ports={profile_ports}\n")
    _write_log(f"[ZeekSpicyExport] pcap_layer={pcap_layer or '<default>'}\n")
    _write_log(f"[ZeekSpicyExport] profile_ip_proto={profile_ip_proto}\n")
    _write_log(f"[ZeekSpicyExport] module_name={module_name}\n")
    _write_log(f"[ZeekSpicyExport] entry_unit={entry_unit}\n")
    _write_log(f"[ZeekSpicyExport] originator_unit={originator_unit} responder_unit={responder_unit}\n")
    frame_index_count: Optional[int] = None
    if frame_index is not None:
        try:
            frame_index_count = len(frame_index)
        except Exception:
            frame_index_count = None
    _write_log(f"[ZeekSpicyExport] frame_index={'<none>' if frame_index is None else frame_index_count}\n")

    # List parsers once (helps map unit names to fully-qualified parser names).
    parsers: List[str] = []
    try:
        cache_key = (
            str(spicy_dump),
            tuple(str(lp) for lp in library_paths),
            tuple(str(p) for p in inputs),
        )
        with _LIST_PARSERS_LOCK:
            cached = _LIST_PARSERS_CACHE.get(cache_key)
        if cached is not None:
            parsers = list(cached)
            if parsers:
                _write_log(f"[ZeekSpicyExport] parsers(cache)={parsers}\n")
            else:
                _write_log("[ZeekSpicyExport] parsers(cache)=<empty>\n")
        else:
            cmd = [str(spicy_dump), "--list-parsers"]
            if library_paths:
                for lp in library_paths:
                    cmd += ["--library-path", str(lp)]
            cmd += [str(p) for p in inputs]
            # Apply the same per-invocation timeout used for packet parsing to avoid
            # hanging on `--list-parsers` for some protocol plugins/toolchains.
            list_timeout_s = 10.0
            raw_timeout = os.getenv("PARGEN_ZEEK_SPICY_TIMEOUT_S")
            if raw_timeout is not None:
                try:
                    list_timeout_s = float(str(raw_timeout).strip())
                except Exception:
                    list_timeout_s = 10.0
            rc, out, err, timed_out = _run_capture(
                cmd,
                cwd=str(plugin_src_dir),
                timeout_s=(float(list_timeout_s) if list_timeout_s > 0 else None),
            )
            _write_log(f"[ZeekSpicyExport] cmd={' '.join(cmd)}\n")
            if timed_out:
                _write_log(f"[ZeekSpicyExport] WARNING: list-parsers timed out after {list_timeout_s:.1f}s\n")
            if out:
                _write_log("\n[list-parsers stdout]\n" + out + ("\n" if not out.endswith("\n") else ""))
            if err:
                _write_log("\n[list-parsers stderr]\n" + err + ("\n" if not err.endswith("\n") else ""))
            if int(rc) == 0:
                parsers = _parse_list_parsers(out or "")
            with _LIST_PARSERS_LOCK:
                _LIST_PARSERS_CACHE[cache_key] = list(parsers)
    except Exception as exc:
        _write_log(f"\n[list-parsers exception] {type(exc).__name__}: {exc}\n")
        _write_log(traceback.format_exc() + "\n")

    if not parsers:
        # Fall back to no explicit parser; spicy-dump will pick the only parser if available.
        _write_log("[ZeekSpicyExport] WARNING: no parsers discovered via --list-parsers; continuing without -p.\n")
    else:
        _write_log(f"[ZeekSpicyExport] parsers={parsers}\n")

    originator_parser = _select_parser_name(available=parsers, module_name=module_name, unit_name=originator_unit)
    responder_parser = _select_parser_name(available=parsers, module_name=module_name, unit_name=responder_unit)
    default_parser = _select_default_parser_name(
        available=parsers,
        module_name=module_name,
        filter_name=filter_name,
        entry_unit=entry_unit,
    )

    _write_log(
        f"[ZeekSpicyExport] originator_parser={originator_parser} responder_parser={responder_parser} "
        f"default_parser={default_parser}\n"
    )

    def _choose_parser(transport: str, sport: int, dport: int) -> Optional[str]:
        if not profile_ports:
            return default_parser
        # When the caller provides an explicit frame index, do not hard-filter by ports.
        # Still use port direction (if possible) to select request/response entry units.
        if originator_parser and dport in profile_ports:
            return originator_parser
        if responder_parser and sport in profile_ports:
            return responder_parser
        return default_parser

    spicy_cwd_local = threading.local()
    spicy_cwd_paths: List[Path] = []
    spicy_cwd_lock = threading.Lock()

    def _spicy_cwd(jobs: int) -> str:
        # Spicy JIT compilation can write temporary build artifacts into the current working
        # directory. Parallel invocations sharing the same cwd can race and intermittently
        # fail with "C++ compilation failed". Isolate the cwd per worker thread when jobs>1.
        if int(jobs) <= 1:
            return str(plugin_src_dir)
        existing = getattr(spicy_cwd_local, "cwd", None)
        if isinstance(existing, str) and existing:
            return existing
        path = Path(tempfile.mkdtemp(prefix="pargen_spicy_cwd_"))
        setattr(spicy_cwd_local, "cwd", str(path))
        with spicy_cwd_lock:
            spicy_cwd_paths.append(path)
        return str(path)

    def _run_spicy_dump(
        payload: bytes,
        *,
        parser_name: Optional[str],
        cwd: str,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "spicy_dump": str(spicy_dump),
            "parser": parser_name,
            "cwd": str(cwd),
        }
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(prefix="pargen_spicy_", suffix=".bin", delete=False) as f:
                f.write(payload)
                tmp_path = Path(f.name)

            cmd = [str(spicy_dump), "-J", "-Q", "-f", str(tmp_path)]
            if parser_name:
                cmd += ["-p", str(parser_name)]
            if library_paths:
                for lp in library_paths:
                    cmd += ["--library-path", str(lp)]
            cmd += [str(p) for p in inputs]
            rc, out, err, timed_out = _run_capture(
                cmd,
                cwd=str(cwd),
                timeout_s=(float(spicy_timeout_s) if spicy_timeout_s > 0 else None),
            )
            if timed_out:
                debug["rc"] = -1
                debug["error"] = f"spicy-dump timeout after {spicy_timeout_s:.1f}s"
                if out:
                    debug["stdout"] = out.strip()
                if err:
                    debug["stderr"] = err.strip()
                return None, debug

            debug["rc"] = int(rc)
            if err:
                debug["stderr"] = err.strip()
            if rc != 0:
                return None, debug

            raw_out = (out or "").strip()
            if not raw_out:
                debug["stderr"] = (debug.get("stderr") or "") + "\n<empty stdout>"
                return None, debug

            try:
                obj = json.loads(raw_out)
            except Exception:
                # Some versions may include stray newlines; attempt to parse the last JSON-looking line.
                lines = [ln.strip() for ln in raw_out.splitlines() if ln.strip()]
                obj = json.loads(lines[-1]) if lines else None
            if not isinstance(obj, dict):
                debug["stderr"] = (debug.get("stderr") or "") + "\n<non-dict JSON output>"
                return None, debug
            return obj, debug
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    # Select frames to export.
    want_frames: Optional[List[int]] = None
    if frame_index is not None:
        seen: set[int] = set()
        want_frames = []
        for raw in frame_index:
            try:
                fid = int(raw)
            except Exception:
                continue
            if fid <= 0 or fid in seen:
                continue
            seen.add(fid)
            want_frames.append(fid)
        want_frames.sort()
        if max_packets and max_packets > 0:
            want_frames = want_frames[: int(max_packets)]

    packets: List[Dict[str, Any]] = []
    packets_failed = 0
    failure_stderr_samples: List[str] = []
    failure_error_samples: List[str] = []

    def _emit_packet(
        *,
        frame_no: int,
        payload: bytes,
        transport: str,
        sport: int,
        dport: int,
    ) -> None:
        nonlocal packets_failed
        parser_name = _choose_parser(transport, sport, dport)
        obj, dbg = _run_spicy_dump(payload, parser_name=parser_name, cwd=_spicy_cwd(jobs))
        debug: Dict[str, Any] = {
            "transport": str(transport),
            "sport": int(sport),
            "dport": int(dport),
        }
        debug.update(dbg)

        fields: List[Dict[str, Any]] = []
        if obj is None:
            packets_failed += 1
            debug.setdefault("error", "spicy-dump failed")
            stderr = str(debug.get("stderr") or "").strip()
            if stderr:
                failure_stderr_samples.append(stderr)
            else:
                failure_error_samples.append(str(debug.get("error") or "spicy-dump failed"))
        else:
            spans = _extract_leaf_spans(obj, payload, prefix="")
            fields = _fields_from_spans(spans, payload, bitfield_defs=bitfield_defs, field_name_mode=field_name_mode)

        packets.append(
            {
                "packet_no": int(frame_no),
                "hex": payload.hex(),
                "base_offset": 0,
                "fields": fields,
                "debug": debug,
            }
        )

    def _env_jobs(default: int) -> int:
        raw = os.getenv("PARGEN_ZEEK_SPICY_JOBS")
        if raw is None:
            return int(default)
        try:
            v = int(str(raw).strip())
        except Exception:
            return int(default)
        return v

    jobs_default = min(4, os.cpu_count() or 1)
    jobs = _env_jobs(jobs_default)
    if jobs <= 0:
        jobs = 1

    def _env_timeout_s(default: float) -> float:
        raw = os.getenv("PARGEN_ZEEK_SPICY_TIMEOUT_S")
        if raw is None:
            return float(default)
        try:
            return float(str(raw).strip())
        except Exception:
            return float(default)

    def _env_progress_every(default: int) -> int:
        raw = os.getenv("PARGEN_ZEEK_SPICY_PROGRESS_EVERY")
        if raw is None:
            return int(default)
        try:
            return int(str(raw).strip())
        except Exception:
            return int(default)

    def _env_progress_enabled() -> bool:
        raw = os.getenv("PARGEN_ZEEK_SPICY_PROGRESS")
        if raw is None:
            return False
        return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}

    spicy_timeout_s = _env_timeout_s(10.0)
    if spicy_timeout_s <= 0:
        spicy_timeout_s = 0.0
    progress_every = max(0, _env_progress_every(10))
    progress_enabled = _env_progress_enabled()
    progress_lock = threading.Lock()
    start_ts = time.time()

    work_items: List[Tuple[int, bytes, str, int, int]] = []

    if pcap_layer == "tcp_header":
        if want_frames is not None:
            by_frame = extract_tcp_headers_from_pcap(pcap_path, want_frames)
            for fid in want_frames:
                hdr = by_frame.get(int(fid))
                if hdr is None:
                    packets_failed += 1
                    packets.append(
                        {
                            "packet_no": int(fid),
                            "hex": "",
                            "base_offset": 0,
                            "fields": [],
                            "debug": {
                                "transport": "tcp",
                                "sport": 0,
                                "dport": 0,
                                "rc": 1,
                                "error": "tcp header not found in pcap",
                            },
                        }
                    )
                    continue
                sport, dport = _ports_from_tcp_header(hdr)
                work_items.append((int(fid), hdr, "tcp", int(sport), int(dport)))
        else:
            # Best-effort full scan; apply profile filters to reduce noise.
            count = 0
            max_frame_no = 0
            for frame_no, hdr in iter_pcap_tcp_headers(pcap_path, max_frame_no=max_frame_no):
                transport = "tcp"
                sport, dport = _ports_from_tcp_header(hdr)
                if profile_transport in {"tcp", "udp"} and transport != profile_transport:
                    continue
                if profile_ports and int(sport) not in profile_ports and int(dport) not in profile_ports:
                    continue
                work_items.append((int(frame_no), hdr, transport, int(sport), int(dport)))
                count += 1
                if max_packets and max_packets > 0 and count >= int(max_packets):
                    break
    elif pcap_layer == "ip_payload":
        if want_frames is not None:
            by_frame = extract_ip_payloads_from_pcap(pcap_path, want_frames, ip_proto=profile_ip_proto)
            for fid in want_frames:
                payload = by_frame.get(int(fid))
                if payload is None:
                    packets_failed += 1
                    packets.append(
                        {
                            "packet_no": int(fid),
                            "hex": "",
                            "base_offset": 0,
                            "fields": [],
                            "debug": {
                                "transport": "ip",
                                "sport": 0,
                                "dport": 0,
                                "rc": 1,
                                "error": "ip payload not found in pcap",
                                "ip_proto": profile_ip_proto,
                            },
                        }
                    )
                    continue
                work_items.append((int(fid), payload, "ip", 0, 0))
        else:
            # Best-effort full scan; apply IP-proto filter when available.
            count = 0
            max_frame_no = 0
            for frame_no, payload in iter_pcap_ip_payloads(pcap_path, max_frame_no=max_frame_no, ip_proto=profile_ip_proto):
                work_items.append((int(frame_no), payload, "ip", 0, 0))
                count += 1
                if max_packets and max_packets > 0 and count >= int(max_packets):
                    break
    elif pcap_layer == "l3_packet":
        if want_frames is not None:
            by_frame = extract_l3_packets_from_pcap(pcap_path, want_frames)
            for fid in want_frames:
                datagram = by_frame.get(int(fid))
                if datagram is None:
                    packets_failed += 1
                    packets.append(
                        {
                            "packet_no": int(fid),
                            "hex": "",
                            "base_offset": 0,
                            "fields": [],
                            "debug": {
                                "transport": "ethernet",
                                "sport": 0,
                                "dport": 0,
                                "rc": 1,
                                "error": "l3 packet not found in pcap",
                            },
                        }
                    )
                    continue
                work_items.append((int(fid), datagram, "ethernet", 0, 0))
        else:
            count = 0
            max_frame_no = 0
            for frame_no, datagram in iter_pcap_l3_packets(pcap_path, max_frame_no=max_frame_no):
                work_items.append((int(frame_no), datagram, "ethernet", 0, 0))
                count += 1
                if max_packets and max_packets > 0 and count >= int(max_packets):
                    break
    else:
        if want_frames is not None:
            by_frame = extract_tcp_udp_payloads_with_ports_from_pcap(pcap_path, want_frames)
            for fid in want_frames:
                entry = by_frame.get(int(fid))
                if entry is None:
                    packets_failed += 1
                    packets.append(
                        {
                            "packet_no": int(fid),
                            "hex": "",
                            "base_offset": 0,
                            "fields": [],
                            "debug": {
                                "transport": "tcp",
                                "sport": 0,
                                "dport": 0,
                                "rc": 1,
                                "error": "payload not found in pcap",
                            },
                        }
                    )
                    continue
                payload, transport, sport, dport = entry
                work_items.append((int(fid), payload, str(transport), int(sport), int(dport)))
        else:
            # Best-effort full scan; apply profile filters to reduce noise.
            count = 0
            max_frame_no = 0
            for frame_no, payload, transport, sport, dport in iter_pcap_tcp_udp_payloads_with_ports(
                pcap_path, max_frame_no=max_frame_no
            ):
                if profile_transport in {"tcp", "udp"} and str(transport) != profile_transport:
                    continue
                if profile_ports and int(sport) not in profile_ports and int(dport) not in profile_ports:
                    continue
                work_items.append((int(frame_no), payload, str(transport), int(sport), int(dport)))
                count += 1
                if max_packets and max_packets > 0 and count >= int(max_packets):
                    break

    # For single-packet export, keep things simple and deterministic:
    # - avoid thread pools
    # - reuse plugin_src_dir as cwd (enables compiler cache reuse across calls)
    if len(work_items) <= 1:
        jobs = 1

    # Some Spicy toolchains produce HLTO artifacts that behave differently from source-based
    # JIT compilation. If we enabled the auto HLTO cache and observe that parsing a sample
    # packet fails with HLTO but succeeds with the original *.spicy sources, fall back to
    # source mode for the remainder of this export.
    #
    # This is protocol-agnostic and avoids per-protocol special cases (e.g., bitfield-related
    # HLTO bugs seen on some setups).
    raw_hlto_fallback = os.getenv("PARGEN_ZEEK_SPICY_HLTO_FALLBACK")
    hlto_fallback_enabled = True
    if raw_hlto_fallback is not None:
        hlto_fallback_enabled = str(raw_hlto_fallback).strip().lower() not in {"", "0", "false", "no", "off"}

    if hlto_fallback_enabled and input_mode == "hlto_cache" and inputs_spicy and work_items:
        try:
            _fid0, _payload0, _transport0, _sport0, _dport0 = work_items[0]
            _parser0 = _choose_parser(str(_transport0), int(_sport0), int(_dport0))
            obj_hlto, dbg_hlto = _run_spicy_dump(_payload0, parser_name=_parser0, cwd=str(plugin_src_dir))
            if obj_hlto is None:
                stderr_hlto = str((dbg_hlto or {}).get("stderr") or "").strip()
                saved_inputs = list(inputs)
                saved_mode = str(input_mode)
                inputs = list(inputs_spicy)
                input_mode = "spicy_src_fallback"
                obj_src, dbg_src = _run_spicy_dump(_payload0, parser_name=_parser0, cwd=str(plugin_src_dir))
                if obj_src is None:
                    # Fallback doesn't help; keep HLTO mode.
                    inputs = saved_inputs
                    input_mode = saved_mode
                else:
                    _write_log("[ZeekSpicyExport] WARNING: HLTO parse failed; falling back to spicy sources.\n")
                    if stderr_hlto:
                        _write_log("[ZeekSpicyExport] HLTO first-error:\n" + stderr_hlto + "\n")
        except Exception:
            # Best-effort: never fail export because of the HLTO sanity check.
            pass

    def _process_one(item: Tuple[int, bytes, str, int, int]) -> Tuple[Dict[str, Any], bool, str, str]:
        frame_no, payload, transport, sport, dport = item
        parser_name = _choose_parser(transport, sport, dport)
        obj, dbg = _run_spicy_dump(payload, parser_name=parser_name, cwd=_spicy_cwd(jobs))
        debug: Dict[str, Any] = {
            "transport": str(transport),
            "sport": int(sport),
            "dport": int(dport),
        }
        debug.update(dbg)

        fields: List[Dict[str, Any]] = []
        failed = False
        stderr_sample = ""
        error_sample = ""
        if obj is None:
            failed = True
            debug.setdefault("error", "spicy-dump failed")
            stderr = str(debug.get("stderr") or "").strip()
            if stderr:
                stderr_sample = stderr
            else:
                error_sample = str(debug.get("error") or "spicy-dump failed")
        else:
            spans = _extract_leaf_spans(obj, payload, prefix="")
            fields = _fields_from_spans(spans, payload, bitfield_defs=bitfield_defs, field_name_mode=field_name_mode)

        pkt = {
            "packet_no": int(frame_no),
            "hex": payload.hex(),
            "base_offset": 0,
            "fields": fields,
            "debug": debug,
        }
        return pkt, failed, stderr_sample, error_sample

    try:
        if work_items:
            total_items = len(work_items)
            done = 0
            if jobs == 1:
                for item in work_items:
                    pkt, failed, stderr_sample, error_sample = _process_one(item)
                    done += 1
                    if failed:
                        packets_failed += 1
                        if stderr_sample:
                            failure_stderr_samples.append(stderr_sample)
                        elif error_sample:
                            failure_error_samples.append(error_sample)
                    packets.append(pkt)
                    if progress_enabled and progress_every and (done % progress_every == 0 or done == total_items):
                        elapsed = max(0.001, time.time() - start_ts)
                        with progress_lock:
                            _write_log(
                                f"[ZeekSpicyExport] progress {done}/{total_items} failed={packets_failed} "
                                f"elapsed_s={elapsed:.1f}\n"
                            )
            else:
                with ThreadPoolExecutor(max_workers=int(jobs)) as executor:
                    futures = [executor.submit(_process_one, item) for item in work_items]
                    try:
                        for fut in as_completed(futures):
                            pkt, failed, stderr_sample, error_sample = fut.result()
                            done += 1
                            if failed:
                                packets_failed += 1
                                if stderr_sample:
                                    failure_stderr_samples.append(stderr_sample)
                                elif error_sample:
                                    failure_error_samples.append(error_sample)
                            packets.append(pkt)
                            if progress_enabled and progress_every and (done % progress_every == 0 or done == total_items):
                                elapsed = max(0.001, time.time() - start_ts)
                                with progress_lock:
                                    _write_log(
                                        f"[ZeekSpicyExport] progress {done}/{total_items} failed={packets_failed} "
                                        f"elapsed_s={elapsed:.1f}\n"
                                    )
                    except KeyboardInterrupt:
                        for fut in futures:
                            try:
                                fut.cancel()
                            except Exception:
                                continue
                        raise

            packets.sort(key=lambda p: int(p.get("packet_no") or 0))
    finally:
        # Clean up isolated CWDs (only created when jobs>1).
        with spicy_cwd_lock:
            paths = list(spicy_cwd_paths)
            spicy_cwd_paths.clear()
        for p in paths:
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

    # Write tool.json and tool.txt
    tooljson_path.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    for pkt in packets:
        fid = int(pkt.get("packet_no") or 0)
        hex_len = len(str(pkt.get("hex") or "")) // 2
        dbg = pkt.get("debug") if isinstance(pkt.get("debug"), dict) else {}
        lines.append(
            f"Packet {fid} len={hex_len} transport={dbg.get('transport')} sport={dbg.get('sport')} "
            f"dport={dbg.get('dport')} parser={dbg.get('parser')}"
        )
        fields = pkt.get("fields") if isinstance(pkt.get("fields"), list) else []
        def _k(f: Any) -> Tuple[int, int, str]:
            if not isinstance(f, dict):
                return (0, 0, "")
            return (int(f.get("offset") or 0), int(f.get("length") or 0), str(f.get("name") or ""))

        for f in sorted([x for x in fields if isinstance(x, dict)], key=_k):
            off = int(f.get("offset") or 0)
            ln = int(f.get("length") or 0)
            name = str(f.get("name") or "")
            raw = str(f.get("raw") or "")
            val = f.get("value")
            lines.append(f"  off={off} len={ln} name={name} raw={raw} value={val}")
        lines.append("")
    texttree_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    total = len(packets)
    _write_log(f"\n[ZeekSpicyExport] packets_total={total} parse_failures={packets_failed}\n")
    if packets_failed and total:
        unique_stderr: List[str] = []
        for item in failure_stderr_samples:
            s = str(item).strip()
            if not s:
                continue
            if s not in unique_stderr:
                unique_stderr.append(s)
            if len(unique_stderr) >= 3:
                break
        if unique_stderr:
            _write_log("[ZeekSpicyExport] sample_stderr:\n")
            for s in unique_stderr:
                _write_log("  " + s.replace("\n", "\\n") + "\n")
        elif failure_error_samples:
            _write_log("[ZeekSpicyExport] sample_errors:\n")
            for s in failure_error_samples[:3]:
                _write_log("  " + str(s).replace("\n", "\\n") + "\n")

    if fail_on_all_parse_failures and total and packets_failed >= total:
        detail = ""
        for s in failure_stderr_samples:
            s = str(s).strip()
            if s:
                detail = s
                break
        if not detail:
            detail = str((failure_error_samples[0] if failure_error_samples else "spicy-dump failed")).strip()
        raise RuntimeError(
            f"spicy-dump failed for all packets (parse_failures={packets_failed}/{total}). "
            f"First error: {detail}"
        )
    try:
        return ExportTooljsonResult(tooljson_path=tooljson_path, jsonraw_path=None, texttree_path=texttree_path, packets=len(packets))
    finally:
        if semantic_tmp is not None:
            semantic_tmp.cleanup()
