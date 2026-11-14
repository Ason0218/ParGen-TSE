from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
from uuid import uuid4

from .wireshark_export import ExportTooljsonResult
from .extraction_profile import infer_ip_proto, infer_pcap_layer
from pargen.parser_realization.target_platforms.bit_utils import ceil_div, safe_value_int_msb0


def resolve_scapy_entry(
    plugin_src_dir: Path,
    filter_name: str,
    *,
    entry_override: Optional[Path] = None,
) -> Path:
    """
    Resolve the Scapy entry file under `plugin_src_dir`.

    Resolution order:
      a) entry_override (relative to plugin_src_dir allowed)
      b) known candidate names
      c) if plugin_src_dir contains exactly one *.py, use it
      d) otherwise raise with a listing of *.py files
    """
    if entry_override:
        override = Path(entry_override)
        if not override.is_absolute():
            override = plugin_src_dir / override
        if override.exists() and override.is_file():
            return override.resolve()
        raise FileNotFoundError(f"Scapy entry override does not exist: {override}")

    candidates = [
        "dissector.py",
        "scapy_layer.py",
        "plugin.py",
        "init.py",
        f"{str(filter_name or '').strip()}.py",
    ]
    for name in candidates:
        if not name or name == ".py":
            continue
        path = plugin_src_dir / name
        if path.exists() and path.is_file():
            return path.resolve()

    py_files = sorted([p for p in plugin_src_dir.glob("*.py") if p.is_file()])
    if len(py_files) == 1:
        return py_files[0].resolve()

    listing = ", ".join([p.name for p in py_files]) if py_files else "<none>"
    raise RuntimeError(
        "Unable to resolve Scapy entry file. "
        f"Found Python files under {plugin_src_dir}: {listing}. "
        "Pass an explicit entry via --stage-c-scapy-entry."
    )


def _infer_ports_from_profile(codegen_profile: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    try:
        reg = (codegen_profile or {}).get("scapy", {}).get("registration", {})
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


def _normalize_ports(ports: Optional[Sequence[int]]) -> Optional[List[int]]:
    if not ports:
        return None
    out: List[int] = []
    for item in ports:
        try:
            p = int(item)
        except Exception:
            continue
        if 0 < p <= 65535:
            out.append(p)
    out = sorted(set(out))
    return out or None


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class FieldSpan:
    name: str
    abbrev: str
    offset: int
    length: int
    raw: str
    value: Any
    bit_start: int
    size_bits: int
    value_int: int
    bit_order: str = "MSB0"


def export_scapy_tooljson(
    *,
    plugin_src_dir: Path,
    pcap_path: Path,
    filter_name: str,
    out_dir: Path,
    entry_override: Optional[Path] = None,
    ports: Optional[Sequence[int]] = None,
    transport: str = "auto",
    max_packets: int = 0,
    include_payload_layers: bool = False,
    include_plugin_extra_fields: bool = False,
    codegen_profile: Optional[Dict[str, Any]] = None,
) -> ExportTooljsonResult:
    """
    Export Stage C Scapy plugin parsing results as tool.json + tool.txt.

    tool.json schema (per packet):
      {
        "packet_no": int,
        "hex": str,               # protocol-layer bytes (relative offsets start at 0)
        "base_offset": int|None,  # scapy uses 0
        "fields": [ { "name","abbrev","offset","length","raw","value" }, ... ],
        "debug": dict
      }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tooljson_path = out_dir / "tool.json"
    texttree_path = out_dir / "tool.txt"
    log_path = out_dir / "scapy_export.log"

    def _extraction_mode(profile: Optional[Dict[str, Any]]) -> str:
        try:
            extraction = (profile or {}).get("extraction", {})
        except Exception:
            extraction = {}
        if isinstance(extraction, dict):
            v = extraction.get("pcap_layer")
            if isinstance(v, str):
                return v.strip().lower()
        return ""

    extraction_mode = _extraction_mode(codegen_profile)
    profile_pcap_layer = infer_pcap_layer(codegen_profile, host="scapy")
    profile_ip_proto = infer_ip_proto(codegen_profile, host="scapy") if profile_pcap_layer == "ip_payload" else None
    allow_fallback = _env_flag("PARGEN_SCAPY_EXPORT_ALLOW_FALLBACK", default=False)

    def _tcp_header_len(segment_bytes: bytes) -> int:
        if not segment_bytes:
            return 0
        if len(segment_bytes) < 13:
            return len(segment_bytes)
        dataofs_words = (segment_bytes[12] >> 4) & 0x0F
        hdr_len = int(dataofs_words) * 4
        if hdr_len < 20:
            hdr_len = 20
        if hdr_len > len(segment_bytes):
            hdr_len = len(segment_bytes)
        return hdr_len

    # Import scapy lazily (repo does not depend on it).
    try:
        from scapy.fields import BitField  # type: ignore[import-not-found]
        from scapy.layers.inet import IP, TCP, UDP  # type: ignore[import-not-found]
        try:
            from scapy.layers.inet6 import IPv6  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover
            IPv6 = None  # type: ignore[assignment]
        from scapy.packet import NoPayload, Packet  # type: ignore[import-not-found]
        from scapy.utils import PcapReader  # type: ignore[import-not-found]
    except Exception as exc:
        msg = "Scapy is not installed. Install it with: pip install scapy"
        log_path.write_text(
            f"[ScapyExport] ERROR: {msg}\n{type(exc).__name__}: {exc}\n{traceback.format_exc()}\n",
            encoding="utf-8",
        )
        raise RuntimeError(msg) from exc

    entry = resolve_scapy_entry(plugin_src_dir, filter_name, entry_override=entry_override)

    ports_effective = _normalize_ports(ports) or _infer_ports_from_profile(codegen_profile)
    transport_effective = (transport or "auto").strip().lower()
    if transport_effective == "auto":
        if profile_pcap_layer == "ip_payload":
            transport_effective = "ip"
        else:
            try:
                reg = (codegen_profile or {}).get("scapy", {}).get("registration", {})
            except Exception:
                reg = {}
            if isinstance(reg, dict):
                inferred = str(reg.get("transport") or "").strip().lower()
                if inferred in {"tcp", "udp", "ip"}:
                    transport_effective = inferred
    if transport_effective not in {"auto", "tcp", "udp", "ip"}:
        transport_effective = "auto"

    module_name = f"pargen_stage_c_scapy_{filter_name}_{uuid4().hex}"
    entry_path = entry.resolve()
    plugin_root = plugin_src_dir.resolve()

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
                out[str(k)] = _json_safe(v)
            return out
        try:
            return repr(value)
        except Exception:
            return str(type(value))

    def _load_plugin() -> Any:
        sys.path.insert(0, str(plugin_src_dir))
        try:
            spec = importlib.util.spec_from_file_location(module_name, entry)
            if spec is None or spec.loader is None:
                raise ImportError(f"spec_from_file_location failed for entry: {entry}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules.pop(module_name, None)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            return mod
        finally:
            try:
                if sys.path and sys.path[0] == str(plugin_src_dir):
                    sys.path.pop(0)
            except Exception:
                pass

    def _iter_layers(pkt: Any) -> Iterable[Any]:
        cur = pkt
        seen: set[int] = set()
        while cur is not None and not isinstance(cur, NoPayload):
            oid = id(cur)
            if oid in seen:
                break
            seen.add(oid)
            yield cur
            cur = getattr(cur, "payload", None)

    def _find_plugin_layer(pkt: Any) -> Optional[Any]:
        for layer in _iter_layers(pkt):
            try:
                mod_name = getattr(layer.__class__, "__module__", None)
                if mod_name == module_name:
                    return layer
                if not mod_name:
                    continue
                mod_obj = sys.modules.get(str(mod_name))
                mod_file = getattr(mod_obj, "__file__", None) if mod_obj else None
                if not isinstance(mod_file, str) or not mod_file:
                    continue
                try:
                    mod_file_path = Path(mod_file).resolve()
                except Exception:
                    mod_file_path = Path(mod_file)
                if mod_file_path == entry_path or plugin_root in mod_file_path.parents:
                    return layer
            except Exception:
                continue
        return None

    def _select_root_packet_class(mod: Any) -> Optional[Type[Any]]:
        classes: List[Type[Any]] = []
        for obj in mod.__dict__.values():
            if not isinstance(obj, type):
                continue
            try:
                if issubclass(obj, Packet) and obj is not Packet and getattr(obj, "__module__", None) == module_name:
                    classes.append(obj)
            except Exception:
                continue
        if not classes:
            return None

        fn = (filter_name or "").strip().lower()
        fn_norm = "".join(ch for ch in fn if ch.isalnum())
        root_tokens = {fn_norm} if fn_norm else set()
        for suffix in ("_bs", "_pargen", "_proto", "_packet", "_dissector"):
            if fn.endswith(suffix):
                root_tokens.add("".join(ch for ch in fn[: -len(suffix)] if ch.isalnum()))
        if "_" in fn:
            root_tokens.add("".join(ch for ch in fn.split("_", 1)[0] if ch.isalnum()))
        root_tokens = {t for t in root_tokens if t}

        def score(cls: Type[Any]) -> Tuple[int, int, str]:
            name = str(getattr(cls, "__name__", "")).lower()
            name_norm = "".join(ch for ch in name if ch.isalnum())
            sc = 0
            if name_norm in root_tokens:
                sc += 100
            for token in root_tokens:
                if name_norm == f"{token}packet" or name_norm == f"{token}parser":
                    sc += 95
                elif token and token in name_norm:
                    sc += 50
            if fn and fn in name:
                sc += 10
            sc += min(9, len(getattr(cls, "fields_desc", []) or []))
            return (-sc, -len(getattr(cls, "fields_desc", []) or []), name)

        classes.sort(key=score)
        return classes[0]

    def _field_abbrev(prefix: str, field_name: str) -> str:
        base = (field_name or "").strip()
        if not base:
            base = "field"
        if prefix:
            return f"{prefix}.{base}"
        return base

    def _norm_field_name(value: Any) -> str:
        return "".join(ch for ch in str(value or "").lower() if ch.isalnum())

    def _preserve_length_delimited_container(name: str) -> bool:
        norm = _norm_field_name(name)
        return norm in {
            "value",
            "values",
            "ievalue",
            "iedata",
            "informationelementvalue",
            "optionvalue",
            "tlvvalue",
            "rdata",
            "rrdata",
        }

    def _stable_field_value(field: Any, pkt_obj: Any, val: Any) -> Any:
        try:
            s = field.i2repr(pkt_obj, val)
            if isinstance(s, str):
                return s
        except Exception:
            pass
        return _json_safe(val)

    def _dedupe_and_leaf_only(spans: List[FieldSpan]) -> List[FieldSpan]:
        spans = [s for s in spans if int(s.size_bits) > 0 and int(s.bit_start) >= 0]

        # Remove container-like spans: keep minimal intervals (leaf-only).
        by_len = sorted(spans, key=lambda s: (int(s.size_bits), int(s.bit_start), s.abbrev, s.raw))
        leaf: List[FieldSpan] = []
        for s in by_len:
            contains_other = False
            end = int(s.bit_start) + int(s.size_bits)
            if not _preserve_length_delimited_container(s.name):
                for k in leaf:
                    kend = int(k.bit_start) + int(k.size_bits)
                    if int(s.size_bits) > int(k.size_bits) and int(s.bit_start) <= int(k.bit_start) and end >= kend:
                        contains_other = True
                        break
            if not contains_other:
                leaf.append(s)

        # Dedupe by bit-level key and prefer the most specific abbrev.
        best: Dict[Tuple[int, int, int], FieldSpan] = {}
        for s in leaf:
            key = (int(s.bit_start), int(s.size_bits), int(s.value_int))
            prev = best.get(key)
            if prev is None:
                best[key] = s
                continue
            cand = s
            if (len(str(prev.abbrev)), str(prev.abbrev), str(prev.name)) < (
                len(str(cand.abbrev)),
                str(cand.abbrev),
                str(cand.name),
            ):
                best[key] = cand

        out = list(best.values())
        out.sort(key=lambda s: (int(s.bit_start), int(s.size_bits), s.abbrev, s.raw))
        return out

    def _plugin_extra_spans(
        mod: Any,
        raw: bytes,
        *,
        underlayer: Optional[Any],
        prefix: str,
        base_offset: int,
        pkt_debug: Dict[str, Any],
        packet_no: int,
    ) -> List[FieldSpan]:
        """Call an optional pargen_extra_fields hook for diagnostic exports.
        
        The hook returns additional field spans from the raw message. It is disabled
        by default; regular exports use Scapy Packet and Field structures."""
        if not include_plugin_extra_fields:
            return []
        hook = getattr(mod, "pargen_extra_fields", None)
        if not callable(hook):
            return []

        try:
            try:
                payload = hook(
                    raw=bytes(raw or b""),
                    underlayer=underlayer,
                    prefix=prefix,
                    base_offset=base_offset,
                    packet_no=packet_no,
                )
            except TypeError:
                payload = hook(bytes(raw or b""))
        except Exception as exc:
            pkt_debug.setdefault("warnings", []).append(
                f"pargen_extra_fields_failed:{type(exc).__name__}:{exc}"
            )
            pkt_debug.setdefault("tracebacks", []).append(traceback.format_exc())
            return []

        if payload is None:
            return []
        if isinstance(payload, dict):
            fields = payload.get("fields") or payload.get("spans") or []
            meta = payload.get("debug")
            if meta is not None:
                pkt_debug["plugin_extra_debug"] = _json_safe(meta)
        else:
            fields = payload
        if not isinstance(fields, (list, tuple)):
            pkt_debug.setdefault("warnings", []).append("pargen_extra_fields_return_not_list")
            return []

        out: List[FieldSpan] = []
        raw_b = bytes(raw or b"")
        for i, item in enumerate(fields):
            if not isinstance(item, dict):
                pkt_debug.setdefault("warnings", []).append(f"extra_field_not_dict:{i}")
                continue
            name = str(item.get("name") or item.get("field") or "field").strip() or "field"
            abbrev = str(item.get("abbrev") or _field_abbrev(prefix, name)).strip()
            try:
                if item.get("bit_start") is not None:
                    bit_start = int(item.get("bit_start"))
                else:
                    bit_start = (int(base_offset) + int(item.get("offset", 0))) * 8
                if item.get("size_bits") is not None:
                    size_bits = int(item.get("size_bits"))
                else:
                    size_bits = int(item.get("length", 0)) * 8
            except Exception:
                pkt_debug.setdefault("warnings", []).append(f"extra_field_bad_span:{name}")
                continue
            if size_bits <= 0 or bit_start < 0:
                continue

            try:
                offset = int(item.get("offset")) if item.get("offset") is not None else int(bit_start // 8)
            except Exception:
                offset = int(bit_start // 8)
            try:
                length = int(item.get("length")) if item.get("length") is not None else int(
                    max(0, ceil_div(bit_start + size_bits, 8) - offset)
                )
            except Exception:
                length = int(max(0, ceil_div(bit_start + size_bits, 8) - offset))

            raw_hex = item.get("raw")
            if isinstance(raw_hex, (bytes, bytearray)):
                raw_hex_s = bytes(raw_hex).hex()
            elif raw_hex is not None:
                raw_hex_s = str(raw_hex)
            else:
                local_start = int(offset - int(base_offset))
                local_end = int(local_start + length)
                raw_hex_s = raw_b[local_start:local_end].hex() if 0 <= local_start <= local_end <= len(raw_b) else ""

            value = item.get("value")
            try:
                if item.get("value_int") is not None:
                    value_int = int(item.get("value_int"))
                else:
                    value_int = int(safe_value_int_msb0(raw_b, int(bit_start - int(base_offset) * 8), int(size_bits)))
            except Exception:
                value_int = 0

            out.append(
                FieldSpan(
                    name=name.replace("_", " "),
                    abbrev=abbrev,
                    offset=int(offset),
                    length=int(length),
                    raw=str(raw_hex_s),
                    value=_json_safe(value),
                    bit_start=int(bit_start),
                    size_bits=int(size_bits),
                    value_int=int(value_int),
                    bit_order=str(item.get("bit_order") or "MSB0"),
                )
            )

        if out:
            pkt_debug["plugin_extra_field_count"] = len(out)
        return out

    def _remaining_state(remaining: Any) -> Tuple[bytes, int]:
        if isinstance(remaining, (bytes, bytearray)):
            return bytes(remaining), 0
        if isinstance(remaining, tuple) and remaining:
            first = remaining[0]
            if isinstance(first, (bytes, bytearray)):
                bn = 0
                if len(remaining) > 1:
                    try:
                        bn = int(remaining[1])
                    except Exception:
                        bn = 0
                return bytes(first), bn
        return b"", 0

    def _is_bitfield(field: Any) -> bool:
        try:
            return isinstance(field, BitField)
        except Exception:
            return "BitField" in type(field).__name__

    def _is_packet_list_field(field: Any) -> bool:
        return field.__class__.__name__ == "PacketListField" and getattr(field, "cls", None) is not None

    def _packet_payload_bytes(pkt_obj: Any) -> bytes:
        try:
            payload = getattr(pkt_obj, "payload", None)
            if payload is None or isinstance(payload, NoPayload):
                return b""
            return bytes(payload)
        except Exception:
            return b""

    def _parse_packet_list_remainder(field: Any, raw: bytes, pkt_debug: Dict[str, Any]) -> Tuple[List[Any], int]:
        elem_cls = getattr(field, "cls", None)
        try:
            is_packet_cls = isinstance(elem_cls, type) and issubclass(elem_cls, Packet) and elem_cls is not Packet
        except Exception:
            is_packet_cls = False
        if not is_packet_cls:
            return [], 0

        out: List[Any] = []
        pos = 0
        raw_b = bytes(raw or b"")
        limit = min(1024, max(1, len(raw_b)))
        while pos < len(raw_b) and len(out) < limit:
            chunk = raw_b[pos:]
            try:
                elem = elem_cls(chunk)
            except Exception as exc:
                pkt_debug.setdefault("warnings", []).append(
                    f"packet_list_remainder_parse_failed:{getattr(field,'name','?')}:{pos}:{type(exc).__name__}:{exc}"
                )
                break
            payload_bytes = _packet_payload_bytes(elem)
            consumed = len(chunk) - len(payload_bytes)
            if consumed <= 0:
                try:
                    consumed = len(bytes(elem))
                except Exception:
                    consumed = 0
            if consumed <= 0 or pos + consumed > len(raw_b):
                pkt_debug.setdefault("warnings", []).append(
                    f"packet_list_remainder_no_progress:{getattr(field,'name','?')}:{pos}:{consumed}"
                )
                break
            try:
                setattr(elem, "_pargen_consumed_len", int(consumed))
            except Exception:
                pass
            out.append(elem)
            pos += int(consumed)
        if pos < len(raw_b) and out:
            pkt_debug.setdefault("warnings", []).append(
                f"packet_list_remainder_leftover:{getattr(field,'name','?')}:{len(raw_b) - pos}"
            )
        return out, int(pos)

    def _replay_fields(
        cls: Type[Any],
        raw: bytes,
        *,
        underlayer: Optional[Any],
        prefix: str,
        base_offset: int,
        pkt_debug: Dict[str, Any],
        depth: int = 0,
    ) -> List[FieldSpan]:
        if depth > 16:
            pkt_debug.setdefault("warnings", []).append("max_depth_exceeded")
            return []

        pkt_obj = cls()
        try:
            if underlayer is not None:
                pkt_obj.underlayer = underlayer
        except Exception:
            pass

        spans: List[FieldSpan] = []
        bit_cursor = 0
        remaining: Any = raw
        fields_desc = list(getattr(cls, "fields_desc", []) or [])
        raw_b = bytes(raw or b"")

        for field_idx, field in enumerate(fields_desc):
            prev_cursor = int(bit_cursor)
            prev_bytes, prev_bn = _remaining_state(remaining)
            try:
                remain2, val = field.getfield(pkt_obj, remaining)
            except Exception as exc:
                pkt_debug.setdefault("errors", []).append(f"getfield({getattr(field,'name','?')}) failed: {exc}")
                pkt_debug.setdefault("tracebacks", []).append(traceback.format_exc())
                break

            next_bytes, next_bn = _remaining_state(remain2)
            consumed_bits = (len(prev_bytes) - len(next_bytes)) * 8 + (int(next_bn) - int(prev_bn))
            if consumed_bits < 0:
                pkt_debug.setdefault("warnings", []).append(
                    f"negative_bit_consumption:{getattr(field,'name','?')}:{consumed_bits}"
                )
                consumed_bits = 0

            if (
                consumed_bits == 0
                and _is_packet_list_field(field)
                and len(prev_bytes) > 0
                and field_idx == len(fields_desc) - 1
            ):
                parsed_list, consumed_bytes = _parse_packet_list_remainder(field, prev_bytes, pkt_debug)
                if parsed_list and consumed_bytes > 0:
                    val = parsed_list
                    remain2 = prev_bytes[consumed_bytes:]
                    next_bytes, next_bn = _remaining_state(remain2)
                    consumed_bits = int(consumed_bytes) * 8
                    pkt_debug.setdefault("warnings", []).append(
                        f"packet_list_remainder_replayed:{getattr(field,'name','?')}:{len(parsed_list)}:{consumed_bytes}"
                    )

            name = str(getattr(field, "name", "") or "").strip() or "field"

            # Update packet context for subsequent fields (length_from/count_from/cond).
            try:
                pkt_obj.fields[name] = val
            except Exception:
                try:
                    setattr(pkt_obj, name, val)
                except Exception:
                    pass

            size_bits = int(consumed_bits)
            if _is_bitfield(field):
                try:
                    size_bits = int(getattr(field, "size", 0) or 0)
                except Exception:
                    size_bits = int(consumed_bits)
            if size_bits < 0:
                size_bits = 0

            if size_bits > 0:
                cur_prefix = prefix
                abbrev = _field_abbrev(cur_prefix, name)
                global_bit_start = int(base_offset) * 8 + int(prev_cursor)

                if (not _is_bitfield(field)) and (prev_cursor % 8 != 0):
                    pkt_debug.setdefault("warnings", []).append(f"non_bitfield_not_byte_aligned:{abbrev}")

                start_byte = global_bit_start // 8
                end_byte = int(ceil_div(global_bit_start + size_bits, 8))
                length_bytes = int(max(0, end_byte - start_byte))
                local_start = int(start_byte - int(base_offset))
                local_end = int(end_byte - int(base_offset))
                raw_cover = raw_b[local_start:local_end].hex() if 0 <= local_start <= local_end <= len(raw_b) else ""

                value_int = 0
                try:
                    value_int = int(safe_value_int_msb0(raw_b, int(prev_cursor), int(size_bits)))
                except Exception:
                    value_int = 0

                # Nested Packet container: recurse and do not emit the container itself when byte-aligned.
                if isinstance(val, Packet) and (prev_cursor % 8 == 0) and (size_bits % 8 == 0):
                    sub_off = int(base_offset) + int(prev_cursor // 8)
                    sub_len = int(size_bits // 8)
                    sub_raw = raw_b[int(prev_cursor // 8) : int(prev_cursor // 8) + sub_len]
                    if _preserve_length_delimited_container(name):
                        spans.append(
                            FieldSpan(
                                name=name.replace("_", " "),
                                abbrev=abbrev,
                                offset=int(start_byte),
                                length=int(length_bytes),
                                raw=str(raw_cover),
                                value=_stable_field_value(field, pkt_obj, val),
                                bit_start=int(global_bit_start),
                                size_bits=int(size_bits),
                                value_int=int(value_int),
                                bit_order="MSB0",
                            )
                        )
                    spans.extend(
                        _replay_fields(
                            val.__class__,
                            sub_raw,
                            underlayer=pkt_obj,
                            prefix=abbrev,
                            base_offset=sub_off,
                            pkt_debug=pkt_debug,
                            depth=depth + 1,
                        )
                    )
                elif (
                    isinstance(val, list)
                    and val
                    and any(isinstance(v, Packet) for v in val)
                    and (prev_cursor % 8 == 0)
                    and (size_bits % 8 == 0)
                ):
                    base_off = int(base_offset) + int(prev_cursor // 8)
                    raw_slice = raw_b[int(prev_cursor // 8) : int(prev_cursor // 8) + int(size_bits // 8)]
                    pos = 0
                    for idx, elem in enumerate(val):
                        if not isinstance(elem, Packet):
                            continue
                        try:
                            elem_len = int(getattr(elem, "_pargen_consumed_len", 0) or 0)
                        except Exception:
                            elem_len = 0
                        if elem_len <= 0:
                            try:
                                elem_bytes = bytes(elem)
                            except Exception:
                                elem_bytes = b""
                            elem_len = len(elem_bytes)
                        if elem_len <= 0 or pos + elem_len > len(raw_slice):
                            pkt_debug.setdefault("warnings", []).append(f"packet_list_split_failed:{name}:{idx}")
                            break
                        spans.extend(
                            _replay_fields(
                                elem.__class__,
                                raw_slice[pos : pos + elem_len],
                                underlayer=pkt_obj,
                                prefix=f"{abbrev}[{idx}]",
                                base_offset=base_off + pos,
                                pkt_debug=pkt_debug,
                                depth=depth + 1,
                            )
                        )
                        pos += elem_len
                else:
                    spans.append(
                        FieldSpan(
                            name=name.replace("_", " "),
                            abbrev=abbrev,
                            offset=int(start_byte),
                            length=int(length_bytes),
                            raw=str(raw_cover),
                            value=_stable_field_value(field, pkt_obj, val),
                            bit_start=int(global_bit_start),
                            size_bits=int(size_bits),
                            value_int=int(value_int),
                            bit_order="MSB0",
                        )
                    )

            bit_cursor += int(size_bits)
            remaining = remain2

        # Optionally decode the payload as a *layer* (not a Field) and include its fields.
        # This is important for Scapy builtin/contrib dissectors that model nested protocols
        # as payload layers (e.g., ModbusADU* -> ModbusPDU*), rather than embedding them in
        # the parent layer's `fields_desc`.
        if include_payload_layers:
            rem_bytes, rem_bn = _remaining_state(remaining)
            if rem_bytes:
                if int(rem_bn) != 0 or int(bit_cursor) % 8 != 0:
                    pkt_debug.setdefault("warnings", []).append(
                        f"payload_not_byte_aligned:{getattr(cls,'__name__','<unknown>')}:{bit_cursor}:{rem_bn}"
                    )
                else:
                    payload_cls = None
                    try:
                        payload_cls = pkt_obj.guess_payload_class(rem_bytes)
                    except Exception as exc:
                        pkt_debug.setdefault("warnings", []).append(
                            f"guess_payload_class_failed:{getattr(cls,'__name__','<unknown>')}:{type(exc).__name__}:{exc}"
                        )
                        payload_cls = None

                    if (
                        payload_cls is not None
                        and payload_cls is not NoPayload
                        and payload_cls is not cls
                        and hasattr(payload_cls, "fields_desc")
                    ):
                        try:
                            consumed = max(0, int(len(raw_b)) - int(len(rem_bytes)))
                            payload_base_offset = int(base_offset) + int(consumed)
                            spans.extend(
                                _replay_fields(
                                    payload_cls,
                                    rem_bytes,
                                    underlayer=pkt_obj,
                                    prefix=f"{prefix}.payload" if prefix else "payload",
                                    base_offset=payload_base_offset,
                                    pkt_debug=pkt_debug,
                                    depth=depth + 1,
                                )
                            )
                        except Exception as exc:
                            pkt_debug.setdefault("warnings", []).append(
                                f"payload_replay_failed:{getattr(cls,'__name__','<unknown>')}:{type(exc).__name__}:{exc}"
                            )

        return spans

    def _protocol_bytes_from_packet(
        pkt: Any, root_cls: Optional[Type[Any]]
    ) -> Tuple[Optional[bytes], Optional[Any], str, Optional[Type[Any]]]:
        transport_layer: Optional[Any] = None
        if transport_effective in {"auto", "tcp"}:
            try:
                transport_layer = pkt.getlayer(TCP)
            except Exception:
                transport_layer = None
        if transport_layer is None and transport_effective in {"auto", "udp"}:
            try:
                transport_layer = pkt.getlayer(UDP)
            except Exception:
                transport_layer = None

        plugin_layer = _find_plugin_layer(pkt)
        if plugin_layer is not None:
            try:
                original = getattr(plugin_layer, "original", None)
                if isinstance(original, (bytes, bytearray)) and original:
                    return (
                        bytes(original),
                        getattr(plugin_layer, "underlayer", None) or transport_layer,
                        "layer_chain",
                        plugin_layer.__class__,
                    )
                return (
                    bytes(plugin_layer),
                    getattr(plugin_layer, "underlayer", None) or transport_layer,
                    "layer_chain",
                    plugin_layer.__class__,
                )
            except Exception:
                return None, transport_layer, "layer_chain_failed", None

        if transport_effective == "ip" or profile_pcap_layer == "ip_payload":
            ip_layer = None
            try:
                ip_layer = pkt.getlayer(IP)
            except Exception:
                ip_layer = None
            if ip_layer is None and IPv6 is not None:
                try:
                    ip_layer = pkt.getlayer(IPv6)
                except Exception:
                    ip_layer = None

            if ip_layer is not None and profile_ip_proto is not None:
                proto_val = None
                try:
                    if isinstance(ip_layer, IP):
                        proto_val = getattr(ip_layer, "proto", None)
                    else:
                        proto_val = getattr(ip_layer, "nh", None)
                except Exception:
                    proto_val = None
                try:
                    proto_i = int(proto_val) if proto_val is not None else None
                except Exception:
                    proto_i = None
                if proto_i is None or int(proto_i) != int(profile_ip_proto):
                    return None, ip_layer, "ip_proto_mismatch", None

            if ip_layer is not None:
                try:
                    payload = bytes(ip_layer.payload)
                except Exception:
                    payload = b""
                # Scapy includes Ethernet padding bytes (to the 60-byte minimum frame) in the
                # decoded IP payload. Trim to the IP header's declared length so offsets/hex are
                # relative to the protocol bytes only.
                try:
                    expected_len: Optional[int] = None
                    if isinstance(ip_layer, IP):
                        ihl = getattr(ip_layer, "ihl", None)
                        ihl_bytes = int(ihl) * 4 if ihl is not None else 20
                        total_len = getattr(ip_layer, "len", None)
                        if total_len is not None:
                            expected_len = max(0, int(total_len) - int(ihl_bytes))
                    else:
                        plen = getattr(ip_layer, "plen", None)
                        if plen is not None:
                            expected_len = max(0, int(plen))
                    if expected_len is not None and expected_len <= len(payload):
                        payload = payload[: int(expected_len)]
                except Exception:
                    pass
                if root_cls is not None:
                    return payload, ip_layer, "ip_payload", root_cls
                return payload, ip_layer, "ip_payload_no_root", None

        if transport_layer is not None:
            if extraction_mode == "tcp_header" and isinstance(transport_layer, TCP):
                try:
                    segment = bytes(transport_layer)
                except Exception:
                    segment = b""
                header = segment[: _tcp_header_len(segment)]
                under = getattr(transport_layer, "underlayer", None)
                if root_cls is not None:
                    return header, under, "tcp_header", root_cls
                return header, under, "tcp_header_no_root", None

            if profile_pcap_layer == "transport_payload" and not allow_fallback:
                # In transport_payload mode, we expect the generated Scapy layer to be present in
                # the packet's layer chain (via bind_layers). If it's missing, do NOT fall back to
                # decoding raw payload bytes, because that would hide integration failures and
                # artificially inflate dataset metrics.
                return b"", transport_layer, "missing_plugin_layer", None

            try:
                payload = bytes(transport_layer.payload)
            except Exception:
                payload = b""
            if root_cls is not None:
                return payload, transport_layer, "transport_payload", root_cls
            return payload, transport_layer, "transport_payload_no_root", None

        return None, None, "no_transport", None

    total = 0
    matched = 0
    exported = 0
    failed = 0

    records: List[Dict[str, Any]] = []
    observed_bitfields: Dict[str, Dict[str, Any]] = {}

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"[ScapyExport] plugin_src_dir={plugin_src_dir}\n")
        logf.write(f"[ScapyExport] entry={entry}\n")
        logf.write(f"[ScapyExport] module_name={module_name}\n")
        logf.write(f"[ScapyExport] pcap={pcap_path}\n")
        logf.write(f"[ScapyExport] filter_name={filter_name}\n")
        logf.write(f"[ScapyExport] transport={transport_effective}\n")
        logf.write(f"[ScapyExport] ports={ports_effective}\n")
        logf.write(f"[ScapyExport] profile_pcap_layer={profile_pcap_layer}\n")
        logf.write(f"[ScapyExport] profile_ip_proto={profile_ip_proto}\n")
        logf.write(f"[ScapyExport] allow_fallback={allow_fallback}\n")

        try:
            mod = _load_plugin()
        except Exception as exc:
            logf.write(f"[ScapyExport] ERROR loading plugin: {type(exc).__name__}: {exc}\n")
            logf.write(traceback.format_exc() + "\n")
            raise

        root_cls = _select_root_packet_class(mod)
        logf.write(f"[ScapyExport] root_cls={getattr(root_cls,'__name__',None)}\n")

        if profile_pcap_layer == "ip_payload":
            from pargen.parser_realization.semantic_stage.gt_from_tree import iter_pcap_ip_payloads

            for frame_no, payload in iter_pcap_ip_payloads(pcap_path, ip_proto=profile_ip_proto):
                total += 1
                if max_packets and exported >= int(max_packets):
                    break
                matched += 1

                pkt_debug: Dict[str, Any] = {
                    "transport": "ip",
                    "sport": None,
                    "dport": None,
                    "ports": None,
                    "strategy": "ip_payload_reassembled",
                    "parse_cls": getattr(root_cls, "__name__", None) if root_cls is not None else None,
                    "profile_ip_proto": profile_ip_proto,
                }

                spans: List[FieldSpan] = []
                try:
                    if root_cls is not None:
                        spans = _replay_fields(
                            root_cls,
                            payload,
                            underlayer=None,
                            prefix=str(filter_name or "").strip(),
                            base_offset=0,
                            pkt_debug=pkt_debug,
                        )
                    spans = _dedupe_and_leaf_only(spans)
                    spans.extend(
                        _plugin_extra_spans(
                            mod,
                            payload,
                            underlayer=None,
                            prefix=str(filter_name or "").strip(),
                            base_offset=0,
                            pkt_debug=pkt_debug,
                            packet_no=int(frame_no),
                        )
                    )
                    for s in spans:
                        if int(s.bit_start) % 8 != 0 or int(s.size_bits) % 8 != 0:
                            observed_bitfields.setdefault(
                                s.abbrev,
                                {
                                    "abbrev": s.abbrev,
                                    "size_bits": int(s.size_bits),
                                    "bit_offset_msb0_in_span": int(s.bit_start),
                                    "bit_order": "MSB0",
                                },
                            )
                except Exception as exc:
                    failed += 1
                    pkt_debug["error"] = f"{type(exc).__name__}: {exc}"
                    pkt_debug.setdefault("tracebacks", []).append(traceback.format_exc())
                    logf.write(f"[ScapyExport] packet_no={frame_no} ERROR: {type(exc).__name__}: {exc}\n")
                    logf.write(traceback.format_exc() + "\n")
                    spans = []

                fields_json = [
                    {
                        "name": s.name,
                        "abbrev": s.abbrev,
                        "offset": int(s.offset),
                        "length": int(s.length),
                        "raw": s.raw,
                        "value": _json_safe(s.value),
                        "bit_start": int(s.bit_start),
                        "size_bits": int(s.size_bits),
                        "value_int": int(s.value_int),
                        "bit_order": str(s.bit_order),
                    }
                    for s in spans
                ]

                records.append(
                    {
                        "packet_no": int(frame_no),
                        "hex": bytes(payload).hex(),
                        "base_offset": 0,
                        "fields": fields_json,
                        "debug": _json_safe(pkt_debug),
                    }
                )
                exported += 1
        elif profile_pcap_layer == "tcp_header":
            # Do not rely on Scapy's built-in IP->TCP bindings here: exp_constraint evaluation may
            # call split_layers(IP, TCP, proto=6) to allow Stage C plugins to bind, which would
            # make pkt.getlayer(TCP) unavailable in the same Python process. Extract TCP header
            # bytes directly from the PCAP and parse them with the Stage C class.
            from pargen.parser_realization.semantic_stage.gt_from_tree import iter_pcap_tcp_headers

            for frame_no, header in iter_pcap_tcp_headers(pcap_path):
                total += 1
                if max_packets and exported >= int(max_packets):
                    break
                matched += 1

                pkt_debug: Dict[str, Any] = {
                    "transport": "ip",
                    "sport": None,
                    "dport": None,
                    "ports": None,
                    "strategy": "tcp_header_extracted",
                    "parse_cls": getattr(root_cls, "__name__", None) if root_cls is not None else None,
                    "profile_ip_proto": 6,
                }

                spans: List[FieldSpan] = []
                try:
                    if root_cls is not None:
                        spans = _replay_fields(
                            root_cls,
                            header,
                            underlayer=None,
                            prefix=str(filter_name or "").strip(),
                            base_offset=0,
                            pkt_debug=pkt_debug,
                        )
                    spans = _dedupe_and_leaf_only(spans)
                    spans.extend(
                        _plugin_extra_spans(
                            mod,
                            header,
                            underlayer=None,
                            prefix=str(filter_name or "").strip(),
                            base_offset=0,
                            pkt_debug=pkt_debug,
                            packet_no=int(frame_no),
                        )
                    )
                    for s in spans:
                        if int(s.bit_start) % 8 != 0 or int(s.size_bits) % 8 != 0:
                            observed_bitfields.setdefault(
                                s.abbrev,
                                {
                                    "abbrev": s.abbrev,
                                    "size_bits": int(s.size_bits),
                                    "bit_offset_msb0_in_span": int(s.bit_start),
                                    "bit_order": "MSB0",
                                },
                            )
                except Exception as exc:
                    failed += 1
                    pkt_debug["error"] = f"{type(exc).__name__}: {exc}"
                    pkt_debug.setdefault("tracebacks", []).append(traceback.format_exc())
                    logf.write(f"[ScapyExport] packet_no={frame_no} ERROR: {type(exc).__name__}: {exc}\n")
                    logf.write(traceback.format_exc() + "\n")
                    spans = []

                fields_json = [
                    {
                        "name": s.name,
                        "abbrev": s.abbrev,
                        "offset": int(s.offset),
                        "length": int(s.length),
                        "raw": s.raw,
                        "value": _json_safe(s.value),
                        "bit_start": int(s.bit_start),
                        "size_bits": int(s.size_bits),
                        "value_int": int(s.value_int),
                        "bit_order": str(s.bit_order),
                    }
                    for s in spans
                ]

                records.append(
                    {
                        "packet_no": int(frame_no),
                        "hex": bytes(header).hex(),
                        "base_offset": 0,
                        "fields": fields_json,
                        "debug": _json_safe(pkt_debug),
                    }
                )
                exported += 1
        elif profile_pcap_layer == "l3_packet":
            from pargen.parser_realization.semantic_stage.gt_from_tree import iter_pcap_l3_packets

            for frame_no, datagram in iter_pcap_l3_packets(pcap_path):
                total += 1
                if max_packets and exported >= int(max_packets):
                    break
                matched += 1

                pkt_debug: Dict[str, Any] = {
                    "transport": "ethernet",
                    "sport": None,
                    "dport": None,
                    "ports": None,
                    "strategy": "l3_packet_clipped",
                    "parse_cls": getattr(root_cls, "__name__", None) if root_cls is not None else None,
                    "profile_ip_proto": None,
                }

                spans: List[FieldSpan] = []
                try:
                    if root_cls is not None:
                        spans = _replay_fields(
                            root_cls,
                            datagram,
                            underlayer=None,
                            prefix=str(filter_name or "").strip(),
                            base_offset=0,
                            pkt_debug=pkt_debug,
                        )
                    spans = _dedupe_and_leaf_only(spans)
                    spans.extend(
                        _plugin_extra_spans(
                            mod,
                            datagram,
                            underlayer=None,
                            prefix=str(filter_name or "").strip(),
                            base_offset=0,
                            pkt_debug=pkt_debug,
                            packet_no=int(frame_no),
                        )
                    )
                    for s in spans:
                        if int(s.bit_start) % 8 != 0 or int(s.size_bits) % 8 != 0:
                            observed_bitfields.setdefault(
                                s.abbrev,
                                {
                                    "abbrev": s.abbrev,
                                    "size_bits": int(s.size_bits),
                                    "bit_offset_msb0_in_span": int(s.bit_start),
                                    "bit_order": "MSB0",
                                },
                            )
                except Exception as exc:
                    failed += 1
                    pkt_debug["error"] = f"{type(exc).__name__}: {exc}"
                    pkt_debug.setdefault("tracebacks", []).append(traceback.format_exc())
                    logf.write(f"[ScapyExport] packet_no={frame_no} ERROR: {type(exc).__name__}: {exc}\n")
                    logf.write(traceback.format_exc() + "\n")
                    spans = []

                fields_json = [
                    {
                        "name": s.name,
                        "abbrev": s.abbrev,
                        "offset": int(s.offset),
                        "length": int(s.length),
                        "raw": s.raw,
                        "value": _json_safe(s.value),
                        "bit_start": int(s.bit_start),
                        "size_bits": int(s.size_bits),
                        "value_int": int(s.value_int),
                        "bit_order": str(s.bit_order),
                    }
                    for s in spans
                ]

                records.append(
                    {
                        "packet_no": int(frame_no),
                        "hex": bytes(datagram).hex(),
                        "base_offset": 0,
                        "fields": fields_json,
                        "debug": _json_safe(pkt_debug),
                    }
                )
                exported += 1
        else:
            try:
                reader = PcapReader(str(pcap_path))
            except Exception as exc:
                logf.write(f"[ScapyExport] ERROR opening pcap: {type(exc).__name__}: {exc}\n")
                logf.write(traceback.format_exc() + "\n")
                raise

            try:
                for idx, pkt in enumerate(reader, start=1):
                    total += 1
                    if max_packets and exported >= int(max_packets):
                        break

                    # Candidate filtering for TCP/UDP-based protocols; IP-based protocols skip this.
                    try:
                        tcp_layer = pkt.getlayer(TCP)
                    except Exception:
                        tcp_layer = None
                    try:
                        udp_layer = pkt.getlayer(UDP)
                    except Exception:
                        udp_layer = None

                    sport = None
                    dport = None
                    layer_for_ports = tcp_layer if tcp_layer is not None else udp_layer
                    if layer_for_ports is not None:
                        try:
                            sport = int(getattr(layer_for_ports, "sport", None))
                        except Exception:
                            sport = None
                        try:
                            dport = int(getattr(layer_for_ports, "dport", None))
                        except Exception:
                            dport = None

                    if ports_effective is not None:
                        if layer_for_ports is None:
                            continue
                        if sport not in ports_effective and dport not in ports_effective:
                            continue
                    if transport_effective == "tcp" and tcp_layer is None:
                        continue
                    if transport_effective == "udp" and udp_layer is None:
                        continue

                    proto_bytes, underlayer, strategy, parse_cls = _protocol_bytes_from_packet(pkt, root_cls)
                    if proto_bytes is None:
                        continue
                    matched += 1

                    pkt_debug: Dict[str, Any] = {
                        "transport": (
                            "tcp"
                            if tcp_layer is not None and transport_effective != "ip"
                            else "udp"
                            if udp_layer is not None and transport_effective != "ip"
                            else "ip"
                            if transport_effective == "ip" or strategy.startswith("ip_")
                            else None
                        ),
                        "sport": sport,
                        "dport": dport,
                        "ports": ports_effective,
                        "strategy": strategy,
                        "parse_cls": getattr(parse_cls, "__name__", None) if parse_cls is not None else None,
                        "profile_ip_proto": profile_ip_proto,
                    }

                    if strategy == "missing_plugin_layer":
                        failed += 1
                        pkt_debug["error"] = "missing plugin layer in Scapy layer chain (fallback disabled)"

                    spans: List[FieldSpan] = []
                    try:
                        if parse_cls is not None:
                            spans = _replay_fields(
                                parse_cls,
                                proto_bytes,
                                underlayer=underlayer,
                                prefix=str(filter_name or "").strip(),
                                base_offset=0,
                                pkt_debug=pkt_debug,
                            )
                        spans = _dedupe_and_leaf_only(spans)
                        spans.extend(
                            _plugin_extra_spans(
                                mod,
                                proto_bytes,
                                underlayer=underlayer,
                                prefix=str(filter_name or "").strip(),
                                base_offset=0,
                                pkt_debug=pkt_debug,
                                packet_no=int(idx),
                            )
                        )
                        for s in spans:
                            if int(s.bit_start) % 8 != 0 or int(s.size_bits) % 8 != 0:
                                observed_bitfields.setdefault(
                                    s.abbrev,
                                    {
                                        "abbrev": s.abbrev,
                                        "size_bits": int(s.size_bits),
                                        "bit_offset_msb0_in_span": int(s.bit_start),
                                        "bit_order": "MSB0",
                                    },
                                )
                    except Exception as exc:
                        failed += 1
                        pkt_debug["error"] = f"{type(exc).__name__}: {exc}"
                        pkt_debug.setdefault("tracebacks", []).append(traceback.format_exc())
                        logf.write(f"[ScapyExport] packet_no={idx} ERROR: {type(exc).__name__}: {exc}\n")
                        logf.write(traceback.format_exc() + "\n")
                        spans = []

                    if pkt_debug.get("errors") or pkt_debug.get("error"):
                        logf.write(
                            f"[ScapyExport] packet_no={idx} debug_errors={pkt_debug.get('errors') or pkt_debug.get('error')}\n"
                        )

                    fields_json = [
                        {
                            "name": s.name,
                            "abbrev": s.abbrev,
                            "offset": int(s.offset),
                            "length": int(s.length),
                            "raw": s.raw,
                            "value": _json_safe(s.value),
                            "bit_start": int(s.bit_start),
                            "size_bits": int(s.size_bits),
                            "value_int": int(s.value_int),
                            "bit_order": str(s.bit_order),
                        }
                        for s in spans
                    ]

                    records.append(
                        {
                            "packet_no": idx,
                            "hex": bytes(proto_bytes).hex(),
                            "base_offset": 0,
                            "fields": fields_json,
                            "debug": _json_safe(pkt_debug),
                        }
                    )
                    exported += 1
            finally:
                try:
                    reader.close()
                except Exception:
                    pass

        logf.write(f"[ScapyExport] total_packets={total}\n")
        logf.write(f"[ScapyExport] matched_packets={matched}\n")
        logf.write(f"[ScapyExport] exported_packets={exported}\n")
        logf.write(f"[ScapyExport] failed_packets={failed}\n")
        logf.write(f"[ScapyExport] tooljson_path={tooljson_path}\n")
        logf.write(f"[ScapyExport] texttree_path={texttree_path}\n")

    tooljson_path.write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: List[str] = []
    for rec in records:
        lines.append(f"Packet {rec.get('packet_no')}: hex_len={len(str(rec.get('hex') or ''))//2}")
        dbg = rec.get("debug") if isinstance(rec.get("debug"), dict) else {}
        if dbg:
            lines.append(f"  debug: {json.dumps(dbg, ensure_ascii=False, sort_keys=True)}")
        fields = rec.get("fields") if isinstance(rec.get("fields"), list) else []
        fields_sorted = []
        for f in fields:
            if isinstance(f, dict):
                fields_sorted.append(f)
        fields_sorted.sort(key=lambda f: (int(f.get("offset", 0) or 0), int(f.get("length", 0) or 0), str(f.get("abbrev") or "")))
        for f in fields_sorted:
            off = int(f.get("offset", 0) or 0)
            ln = int(f.get("length", 0) or 0)
            name = str(f.get("name") or "")
            raw = str(f.get("raw") or "")
            value = f.get("value")
            if value is None or value == "":
                lines.append(f"  [{off:04d}:{ln:04d}] {name} raw={raw}")
            else:
                lines.append(f"  [{off:04d}:{ln:04d}] {name} raw={raw} value={value}")
        lines.append("")

    texttree_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    bitfield_map_payload = {
        "filter_name": str(filter_name),
        "bit_order": "MSB0",
        "generated_by": "scapy_export",
        "entries": sorted(observed_bitfields.values(), key=lambda e: str(e.get("abbrev", ""))),
    }
    (out_dir / "bitfield_map.json").write_text(
        json.dumps(bitfield_map_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return ExportTooljsonResult(
        tooljson_path=tooljson_path,
        jsonraw_path=None,
        texttree_path=texttree_path,
        packets=int(exported),
    )
