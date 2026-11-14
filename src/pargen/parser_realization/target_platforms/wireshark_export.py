from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from pargen.parser_realization.target_platforms.bit_utils import safe_value_int_msb0


@dataclass(frozen=True)
class ExportTooljsonResult:
    tooljson_path: Path
    jsonraw_path: Optional[Path] = None
    texttree_path: Optional[Path] = None
    packets: int = 0


_HEX_ALLOWED = set("0123456789abcdefABCDEF")
_FRAME_LINE_RE = re.compile(r"^Frame\s+(\d+):", flags=re.MULTILINE)


def _parse_frame_numbers_from_texttree(text: str) -> List[int]:
    if not text:
        return []
    out: List[int] = []
    for m in _FRAME_LINE_RE.finditer(text):
        try:
            out.append(int(m.group(1)))
        except Exception:
            continue
    return out


def _parse_frame_numbers_from_fields(text: str) -> List[int]:
    out: List[int] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except Exception:
            continue
    return out


def _normalize_hex(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    s = "".join(ch for ch in s if ch in _HEX_ALLOWED)
    return s.lower()


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _pcap_layer_from_profile(codegen_profile: Optional[Mapping[str, Any]]) -> str:
    try:
        extraction = (codegen_profile or {}).get("extraction", {})
    except Exception:
        extraction = {}
    if isinstance(extraction, Mapping):
        v = extraction.get("pcap_layer")
        if isinstance(v, str):
            return v.strip().lower()
    return ""


def _tshark_pref_is_set(extra: Sequence[str], pref_key: str) -> bool:
    key = str(pref_key or "").strip().lower()
    if not key:
        return False
    for arg in extra:
        if key in str(arg).strip().lower():
            return True
    return False


def _maybe_inject_reassembly_prefs(
    extra: List[str],
    *,
    codegen_profile: Optional[Mapping[str, Any]],
) -> List[str]:
    # Strategy B (reassembled datagrams/streams): for IP-payload protocols (e.g., ICMP over IPv4),
    # ensure tshark runs with IP defragmentation enabled so frame numbering aligns with the
    # reassembled unit (Wireshark attaches dissector output to the last fragment).
    try:
        from pargen.parser_realization.target_platforms.extraction_profile import infer_pcap_layer

        pcap_layer = infer_pcap_layer(codegen_profile, host=None)
    except Exception:
        pcap_layer = _pcap_layer_from_profile(codegen_profile)

    if pcap_layer != "ip_payload":
        return extra

    out = list(extra)
    if not _tshark_pref_is_set(out, "ip.defragment:"):
        out.extend(["-o", "ip.defragment:TRUE"])
    if not _tshark_pref_is_set(out, "ipv6.defragment:"):
        out.extend(["-o", "ipv6.defragment:TRUE"])
    return out


_WS_PLUGIN_EXTS = (".so", ".dylib", ".dll")
_WS_PLUGIN_VERSION_RE = re.compile(r"^\d+\-\d+$")


def _ws_find_plugin_library(*, run_plugins_root: Path, plugin_id: str) -> Optional[Path]:
    pid = str(plugin_id or "").strip()
    if not pid:
        return None

    for ext in _WS_PLUGIN_EXTS:
        # Prefer the canonical run/plugins/<ver>/epan/<plugin_id>.<ext>
        for cand in run_plugins_root.glob(f"*/epan/{pid}{ext}"):
            try:
                if cand.is_file():
                    return cand
            except OSError:
                continue

    # Fallback: best-effort deep search.
    try:
        for cand in run_plugins_root.rglob(f"{pid}*"):
            try:
                if cand.is_file() and cand.suffix.lower() in _WS_PLUGIN_EXTS:
                    return cand
            except OSError:
                continue
    except Exception:
        return None

    return None


def _ws_prepare_isolated_plugin_root(
    *,
    tshark_path: Path,
    out_dir: Path,
    plugin_id: Optional[str],
) -> Optional[Path]:
    """
    Create an isolated Wireshark plugin root directory containing ONLY the requested plugin (if provided),
    suitable for use as WIRESHARK_PLUGIN_DIR.

    Notes:
    - WIRESHARK_PLUGIN_DIR expects a directory containing version subdirs (e.g., <root>/4-4/epan/*.so).
    - If plugin_id is None/empty, we return an empty root to disable all C plugins (useful for wireshark_lua).
    - If we cannot find the built plugin library for plugin_id, return None (caller should fall back to default).
    """
    root = out_dir / "_ws_plugins_isolated"
    try:
        if root.exists():
            shutil.rmtree(root)
    except Exception:
        pass
    root.mkdir(parents=True, exist_ok=True)

    pid = str(plugin_id or "").strip()
    if not pid:
        return root

    run_dir = Path(tshark_path).parent
    run_plugins_root = run_dir / "plugins"
    if not run_plugins_root.exists():
        return None

    lib_path = _ws_find_plugin_library(run_plugins_root=run_plugins_root, plugin_id=pid)
    if lib_path is None:
        return None

    # Infer version directory from the library path: run/plugins/<ver>/epan/<name>.<ext>
    version_dir: Optional[str] = None
    try:
        rel = lib_path.relative_to(run_plugins_root)
        if rel.parts:
            version_dir = rel.parts[0]
    except Exception:
        version_dir = None
    if not (isinstance(version_dir, str) and _WS_PLUGIN_VERSION_RE.match(version_dir or "")):
        try:
            version_dir = lib_path.parent.parent.name
        except Exception:
            version_dir = None
    if not version_dir:
        version_dir = "4-4"

    epan_dir = root / str(version_dir) / "epan"
    epan_dir.mkdir(parents=True, exist_ok=True)
    dest = epan_dir / lib_path.name
    target = lib_path
    try:
        target = lib_path.resolve()
    except Exception:
        target = lib_path
    try:
        dest.symlink_to(target)
    except Exception:
        try:
            shutil.copy2(target, dest)
        except Exception:
            return None
    return root


def _ws_is_generated_plugin_name(name: str) -> bool:
    stem = Path(str(name or "")).stem
    if not stem:
        return False
    return (
        stem.endswith("_bs")
        or "_bs_" in stem
        or "_exp_" in stem
        or stem.startswith("pargen_")
    )


@contextmanager
def _ws_temporarily_hide_generated_plugins(
    *,
    tshark_path: Path,
    keep_plugin_id: Optional[str],
) -> Iterator[None]:
    """Hide stale ParGen-generated Wireshark C plugins during one tshark run.

    WIRESHARK_PLUGIN_DIR adds an isolated plugin root, but the bundled tshark
    still loads plugins from its build/run/plugins directory. Prior experiments
    may leave plugins with names outside the ``*_bs`` convention (for example
    ``*_exp_*`` probes), so in isolated mode keep only the requested plugin.
    """

    run_plugins_root = Path(tshark_path).parent / "plugins"
    if not run_plugins_root.exists():
        yield
        return

    keep = str(keep_plugin_id or "").strip()
    keep_names = {f"{keep}{ext}" for ext in _WS_PLUGIN_EXTS if keep}
    quarantine = run_plugins_root / f".pargen_hidden_plugins_{uuid.uuid4().hex}"
    moved: List[Tuple[Path, Path]] = []
    try:
        for cand in sorted(run_plugins_root.glob("*/epan/*")):
            try:
                if not cand.is_file() or cand.suffix.lower() not in _WS_PLUGIN_EXTS:
                    continue
                if cand.name in keep_names:
                    continue
                if not _ws_is_generated_plugin_name(cand.name):
                    continue
                rel = cand.relative_to(run_plugins_root)
                dst = quarantine / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cand), str(dst))
                moved.append((dst, cand))
            except Exception:
                continue
        yield
    finally:
        for src, dst in reversed(moved):
            try:
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
            except Exception:
                continue
        try:
            shutil.rmtree(quarantine)
        except Exception:
            pass


def _tcp_header_len_from_layer_hex(layer_hex: str) -> Optional[int]:
    # TCP Data Offset is stored in the high nibble of the 13th byte (offset 12).
    # It encodes header size in 32-bit words.
    s = _normalize_hex(layer_hex)
    if len(s) < 26:
        return None
    if len(s) % 2 == 1:
        s = s[:-1]
    try:
        b12 = int(s[24:26], 16)
    except Exception:
        return None
    dataofs_words = (b12 >> 4) & 0x0F
    hdr_len = int(dataofs_words) * 4
    if hdr_len < 20:
        hdr_len = 20
    avail = len(s) // 2
    if hdr_len > avail:
        hdr_len = avail
    return hdr_len


def _json_loads_preserve_duplicates(text: str) -> Any:
    """
    Load JSON while preserving duplicate object keys.

    tshark `-T jsonraw` output for Lua dissectors can contain duplicate keys (e.g. multiple `_ws.lua.text`
    objects). Python's default `json.loads` keeps the last occurrence and silently drops earlier ones,
    which can discard fields. This loader aggregates duplicates into a list, preserving all values.
    """

    def _pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                prev = out[key]
                if isinstance(prev, list):
                    prev.append(value)
                else:
                    out[key] = [prev, value]
            else:
                out[key] = value
        return out

    def _safe_parse_int(s: str) -> Any:
        # jsonraw can contain extremely large integer values (e.g., bytes fields rendered as a
        # decimal integer). Python >=3.11 may raise when parsing these due to the max-int-string-
        # digits guard. We keep "normal" ints as int and represent huge ints as strings.
        try:
            raw = str(s)
            digits = raw.lstrip("+-")
            if len(digits) > 1024:
                return raw
            return int(raw)
        except Exception:
            return s

    return json.loads(text, object_pairs_hook=_pairs_hook, parse_int=_safe_parse_int)


def _select_layer_raw_triplet(value: Any) -> Optional[Tuple[str, Optional[int], Optional[int]]]:
    """
    Best-effort parse of a jsonraw `<filter>_raw` entry.

    Typical shape:
      [hex, base_offset, length]
    """
    if isinstance(value, list) and value:
        # When parsing jsonraw with duplicate keys preserved, values can become a list-of-lists.
        # Pick the first triplet-like entry.
        if isinstance(value[0], list):
            for item in value:
                triplet = _select_layer_raw_triplet(item)
                if triplet is not None:
                    return triplet
            return None
        # Sometimes tshark wraps values as a singleton list-of-list; unwrap it.
        if len(value) == 1 and isinstance(value[0], list):
            return _select_layer_raw_triplet(value[0])
        hex_str = _normalize_hex(value[0])
        base_off = _coerce_int(value[1]) if len(value) > 1 else None
        size = _coerce_int(value[2]) if len(value) > 2 else None
        return (hex_str, base_off, size)
    if isinstance(value, str):
        return (_normalize_hex(value), None, None)
    return None


def _is_child_tree(value: Any) -> bool:
    """A field's subtree: an object, or, for a field repeated within one parent (duplicate keys
    aggregated by _json_loads_preserve_duplicates), a list of objects."""
    return isinstance(value, dict) or (isinstance(value, list) and any(isinstance(item, dict) for item in value))


def _iter_field_raw_entries(layer_obj: Any) -> List[Dict[str, Any]]:
    """
    Extract raw field spans from a protocol layer object.

    Returns list entries:
      {"raw_name": str, "abs_offset": int|None, "size": int|None, "raw_hex": str, "bitmask_int": int|None}
    """
    out: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    walk(v)
                    continue
                if isinstance(v, list):
                    # Field raw entry: <name>_raw: [hex, abs_off, size]
                    if isinstance(k, str) and k.endswith("_raw") and v:
                        # tshark may emit a raw entry as either:
                        #   [hex, abs_off, size, ...]
                        # or, when a key repeats, as:
                        #   [[hex, abs_off, size, ...], [hex, abs_off, size, ...], ...]
                        # or, for some Lua dissectors, as a mixed form where the first entry is
                        # inlined (flattened) and subsequent entries are nested:
                        #   [hex, abs_off, size, ..., [hex, abs_off, size, ...], [hex, abs_off, size, ...], ...]
                        triplets: List[list[Any]] = []
                        if isinstance(v[0], str):
                            triplets = [v]
                            # Preserve additional occurrences that tshark encodes as nested lists.
                            for item in v[1:]:
                                if isinstance(item, list) and item and isinstance(item[0], str):
                                    triplets.append(item)
                        elif isinstance(v[0], list):
                            triplets = [item for item in v if isinstance(item, list) and item and isinstance(item[0], str)]
                        for t in triplets:
                            raw_hex = _normalize_hex(t[0])
                            abs_off = _coerce_int(t[1]) if len(t) > 1 else None
                            size = _coerce_int(t[2]) if len(t) > 2 else None
                            bitmask_int = _coerce_int(t[3]) if len(t) > 3 else None
                            if size is None and raw_hex:
                                size = max(1, len(raw_hex) // 2)
                            out.append(
                                {
                                    "raw_name": k,
                                    "raw_hex": raw_hex,
                                    "abs_offset": abs_off,
                                    "size": size,
                                    "bitmask_int": bitmask_int,
                                    # Containment alone does not establish a
                                    # container: malformed sibling bitmasks can
                                    # overlap perfectly valid scalar fields.
                                    "has_child_tree": _is_child_tree(obj.get(k[:-4]))
                                    or _is_child_tree(obj.get(k[:-4] + "_tree")),
                                }
                            )
                    for item in v:
                        if isinstance(item, dict):
                            walk(item)
                elif isinstance(v, (str, int, float, bool, type(None))):
                    continue
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(layer_obj)
    return out


def _iter_prefixed_field_raw_entries(obj: Any, filter_name: str) -> List[Dict[str, Any]]:
    """Extract raw field spans for a filter prefix from an arbitrary jsonraw subtree.

    Lua dissectors sometimes appear in tshark jsonraw under generic
    ``_ws.lua.text`` containers instead of a top-level protocol layer key.  In
    that shape, the field abbreviations are still stable
    (for example, ``dns_bs.id_raw``), so falling back to a prefix-filtered
    recursive scan preserves the platform contract without adding
    protocol-specific exporters.
    """

    prefix = f"{str(filter_name).strip()}."
    raw_layer_name = f"{str(filter_name).strip()}_raw"
    entries = _iter_field_raw_entries(obj)
    out: List[Dict[str, Any]] = []
    for entry in entries:
        raw_name = str(entry.get("raw_name") or "")
        if raw_name == raw_layer_name:
            continue
        if raw_name.startswith(prefix):
            out.append(entry)
    return out


def _name_from_raw_key(raw_key: str) -> str:
    # Backwards-compatible fallback used only when we can't resolve a better display name.
    base = raw_key.strip()
    if base.endswith("_raw"):
        base = base[: -len("_raw")]
    return base.split(".")[-1]


def _abbrev_from_raw_key(raw_key: str) -> str:
    """Convert a tshark jsonraw key like `modbus.foo_raw` to a field abbrev `modbus.foo`."""
    key = str(raw_key or "").strip()
    if key.endswith("_raw"):
        key = key[: -len("_raw")]
    return key


def _requested_lua_script(extra_tshark_args: Optional[List[str]]) -> bool:
    extra = extra_tshark_args or []
    for arg in extra:
        s = str(arg)
        if "lua_script:" in s:
            return True
    return False


def _stderr_has_lua_load_error(stderr: str) -> bool:
    s = (stderr or "").lower()
    return ("lua: error during loading" in s) or ("lua: error while loading" in s)


def _extract_lua_script_args(args: Sequence[str]) -> Tuple[List[Path], List[str]]:
    scripts: List[Path] = []
    adjusted: List[str] = []
    i = 0
    while i < len(args):
        arg = str(args[i])
        if arg == "-X" and i + 1 < len(args):
            value = str(args[i + 1])
            if value.startswith("lua_script:"):
                scripts.append(Path(value.split(":", 1)[1]))
                i += 2
                continue
            if re.fullmatch(r"lua_script\d+:.*", value):
                i += 2
                continue
        if arg.startswith("lua_script:"):
            scripts.append(Path(arg.split(":", 1)[1]))
            i += 1
            continue
        if re.fullmatch(r"lua_script\d+:.*", arg):
            i += 1
            continue
        adjusted.append(arg)
        i += 1
    return scripts, adjusted


def _running_as_root() -> bool:
    try:
        return bool(os.geteuid() == 0)
    except Exception:
        return False


def _global_lua_plugin_dir(tshark_path: Path) -> Optional[Path]:
    override = os.environ.get("PARGEN_WIRESHARK_GLOBAL_LUA_PLUGIN_DIR")
    if override:
        return Path(override)
    try:
        proc = subprocess.run(
            [str(tshark_path), "-G", "folders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        for line in (proc.stdout or "").splitlines():
            if line.startswith("Global Lua Plugins:"):
                raw = line.split(":", 1)[1].strip()
                if raw:
                    return Path(raw)
    except Exception:
        pass
    fallback = Path("/usr/lib/x86_64-linux-gnu/wireshark/plugins")
    return fallback if fallback.exists() else None


@contextmanager
def _stage_root_lua_scripts_for_command(
    *,
    tshark_path: Path,
    cmd: Sequence[str],
    filter_name: str,
) -> Iterator[List[str]]:
    cmd_list = [str(x) for x in cmd]
    scripts, adjusted = _extract_lua_script_args(cmd_list)
    if not scripts or not _running_as_root():
        yield cmd_list
        return

    plugin_dir = _global_lua_plugin_dir(tshark_path)
    if plugin_dir is None:
        yield cmd_list
        return
    try:
        plugin_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        yield cmd_list
        return
    if not os.access(plugin_dir, os.W_OK):
        yield cmd_list
        return

    lock_path = Path(tempfile.gettempdir()) / "pargen_wireshark_global_lua_plugin.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    staged: List[Path] = []
    try:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        safe_filter = re.sub(r"[^A-Za-z0-9_]+", "_", str(filter_name or "lua")).strip("_") or "lua"
        for script in scripts:
            dst = plugin_dir / f"pargen_{safe_filter}_{os.getpid()}_{uuid.uuid4().hex}.lua"
            shutil.copy2(script, dst)
            staged.append(dst)
        yield adjusted
    finally:
        for path in staged:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_handle.close()


def _tail_lines(text: str, *, max_lines: int = 40) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines).strip()
    return "\n".join(lines[-max_lines:]).strip()


@dataclass(frozen=True)
class WiresharkFieldInfo:
    display_name: str
    field_type: str
    bitmask_int: int


def _parse_bitmask_int(raw: str) -> int:
    s = str(raw or "").strip()
    if not s:
        return 0
    try:
        return int(s, 0)
    except Exception:
        return 0


def _trailing_zeros(n: int) -> int:
    x = int(n)
    if x <= 0:
        return 0
    return int((x & -x).bit_length() - 1)


def bitmask_to_msb0_offset_and_size(bitmask_int: int, size_bytes: int) -> Tuple[int, int, str]:
    """Convert a Wireshark bitmask into (bit_offset_msb0, size_bits, assumption)."""

    mask = int(bitmask_int)
    span_bytes = int(size_bytes)
    if mask <= 0 or span_bytes <= 0:
        return 0, max(0, span_bytes * 8), "not_bitfield"

    total_bits = int(span_bytes * 8)
    hi = int(mask.bit_length() - 1)
    if hi >= total_bits:
        # Bitmask refers to a wider storage than we observed; fall back to "whole span".
        return 0, total_bits, "mask_exceeds_span"
    lo = _trailing_zeros(mask)
    width = int(hi - lo + 1)
    contiguous = bool(mask == (((1 << width) - 1) << lo))

    bit_offset_msb0 = int((total_bits - 1) - hi)
    if contiguous:
        return bit_offset_msb0, width, "contiguous_mask"
    # Approximation for non-contiguous masks: cover the full bit range [hi..lo].
    return bit_offset_msb0, width, "noncontiguous_mask_approx"


def _guess_size_bytes_from_type(field_type: str) -> Optional[int]:
    ft = str(field_type or "").strip().upper()
    if not ft:
        return None
    if ft in {"FT_UINT8", "FT_INT8", "FT_BOOLEAN"}:
        return 1
    if ft in {"FT_UINT16", "FT_INT16"}:
        return 2
    if ft in {"FT_UINT24", "FT_INT24"}:
        return 3
    if ft in {"FT_UINT32", "FT_INT32", "FT_FLOAT"}:
        return 4
    if ft in {"FT_UINT40", "FT_INT40"}:
        return 5
    if ft in {"FT_UINT48", "FT_INT48"}:
        return 6
    if ft in {"FT_UINT56", "FT_INT56"}:
        return 7
    if ft in {"FT_UINT64", "FT_INT64", "FT_DOUBLE"}:
        return 8
    return None


def _load_field_info_map(
    *,
    tshark_path: Path,
    filter_name: str,
    extra_tshark_args: Optional[List[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, WiresharkFieldInfo]:
    """Return `field_abbrev -> WiresharkFieldInfo` for fields under `filter_name`.

    This is derived from `tshark -G fields`, which includes both built-in dissectors and loaded plugins.
    """
    extra = [str(arg) for arg in (extra_tshark_args or [])]
    cmd = [str(tshark_path), *extra, "-G", "fields"]
    with _stage_root_lua_scripts_for_command(tshark_path=tshark_path, cmd=cmd, filter_name=filter_name) as staged_cmd:
        proc = subprocess.Popen(staged_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        assert proc.stdout is not None
        assert proc.stderr is not None
        mapping: Dict[str, WiresharkFieldInfo] = {}
        prefix = f"{filter_name}."
        try:
            for line in proc.stdout:
                # Format: F <TAB> <display_name> <TAB> <abbrev> <TAB> ...
                if not line.startswith("F\t"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                display_name = parts[1].strip()
                abbrev = parts[2].strip()
                if not abbrev.startswith(prefix):
                    continue
                field_type = parts[3].strip() if len(parts) > 3 else ""
                bitmask = parts[6].strip() if len(parts) > 6 else "0x0"
                mapping[abbrev] = WiresharkFieldInfo(
                    display_name=display_name or abbrev,
                    field_type=field_type,
                    bitmask_int=_parse_bitmask_int(bitmask),
                )
        finally:
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
                proc.wait()
        stderr = ""
        try:
            stderr = proc.stderr.read()
        except Exception:
            stderr = ""
    if proc.returncode != 0:
        return {}
    if _requested_lua_script(extra_tshark_args) and _stderr_has_lua_load_error(stderr):
        raise RuntimeError(f"tshark -G fields reported Lua load error:\n{_tail_lines(stderr)}")
    return mapping


def flatten_jsonraw_to_tooljson_records(
    payload: Any,
    filter_name: str,
    *,
    field_infos: Optional[Mapping[str, WiresharkFieldInfo]] = None,
    respect_bitmask: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Flatten tshark -T jsonraw payload into the standard Stage C tool.json schema.

    Notes:
      - Prefer `<filter_name>_raw` for protocol-layer bytes and base_offset.
      - Field offsets are emitted relative to the protocol layer base_offset.
      - Only structurally identified containers are pruned by containment.
      - Container prune + dedupe use bit-level intervals to avoid merging bitfields.
    """
    records: List[Dict[str, Any]] = []
    bitfield_defs: Dict[str, Dict[str, Any]] = {}
    raw_packets = payload if isinstance(payload, list) else []

    for idx, pkt in enumerate(raw_packets, start=1):
        layers = pkt.get("_source", {}).get("layers", pkt) if isinstance(pkt, dict) else {}
        frame = layers.get("frame", {}) if isinstance(layers, dict) else {}
        number = frame.get("frame.number") if isinstance(frame, dict) else None
        if number in (None, "", [], {}):
            number = frame.get("frame.number_raw") if isinstance(frame, dict) else None
        packet_no: int
        try:
            packet_no = int(number[0]) if isinstance(number, list) else int(number)  # type: ignore[arg-type]
        except Exception:
            packet_no = idx

        debug: Dict[str, Any] = {}

        layer_obj = layers.get(filter_name) if isinstance(layers, dict) else None
        layer_raw_triplet = _select_layer_raw_triplet(layers.get(f"{filter_name}_raw")) if isinstance(layers, dict) else None
        frame_raw_triplet = _select_layer_raw_triplet(layers.get("frame_raw")) if isinstance(layers, dict) else None

        layer_hex = layer_raw_triplet[0] if layer_raw_triplet else ""
        base_offset = layer_raw_triplet[1] if layer_raw_triplet else None
        layer_len_bytes = layer_raw_triplet[2] if layer_raw_triplet else None

        # Gather field entries early so we can derive base_offset when missing.
        field_entries = _iter_field_raw_entries(layer_obj or {})
        if not field_entries and layer_obj is None and isinstance(layers, dict):
            field_entries = _iter_prefixed_field_raw_entries(layers, filter_name)
            if field_entries:
                debug["layer_obj_fallback"] = "recursive_prefixed_fields"
        abs_offsets = [e["abs_offset"] for e in field_entries if isinstance(e.get("abs_offset"), int)]

        if base_offset is None and abs_offsets:
            base_offset = min(abs_offsets)
            debug["base_offset_fallback"] = "min_field_abs_offset"

        # Determine protocol-layer bytes (hex) and ensure offsets align to it.
        layer_bytes: bytes = b""
        if layer_hex:
            norm = layer_hex
            if layer_len_bytes is not None and layer_len_bytes > 0:
                norm = norm[: layer_len_bytes * 2]
            try:
                layer_bytes = bytes.fromhex(norm)
            except ValueError:
                layer_bytes = b""
                debug["layer_hex_parse_error"] = True
            # IPv4 dissector quirk: `ip_raw` is header-only, but some evaluations
            # require the full IPv4 datagram (header + payload). When possible,
            # upgrade the layer bytes using `frame_raw` + IPv4 Total Length.
            if str(filter_name).strip().lower() == "ip" and frame_raw_triplet and base_offset is not None:
                total_len = 0
                try:
                    if len(layer_bytes) >= 4:
                        total_len = int.from_bytes(layer_bytes[2:4], "big", signed=False)
                except Exception:
                    total_len = 0
                if total_len > 0 and total_len > len(layer_bytes):
                    frame_hex = frame_raw_triplet[0]
                    if frame_hex:
                        try:
                            frame_bytes = bytes.fromhex(frame_hex)
                        except ValueError:
                            frame_bytes = b""
                            debug["frame_hex_parse_error"] = True
                        end = int(base_offset) + int(total_len)
                        if int(base_offset) >= 0 and end <= len(frame_bytes):
                            layer_bytes = frame_bytes[int(base_offset) : end]
                            debug["layer_hex_upgraded"] = "ipv4_total_length"
                            debug["ipv4_total_length_bytes"] = int(total_len)
            layer_hex = layer_bytes.hex()
        else:
            # Fallback to frame_raw: slice out the protocol layer bytes using base_offset + max_end.
            frame_hex = frame_raw_triplet[0] if frame_raw_triplet else ""
            if frame_hex and base_offset is not None:
                try:
                    frame_bytes = bytes.fromhex(frame_hex)
                except ValueError:
                    frame_bytes = b""
                    debug["frame_hex_parse_error"] = True
                max_end = None
                try:
                    ends = []
                    for e in field_entries:
                        off = e.get("abs_offset")
                        size = e.get("size")
                        if isinstance(off, int) and isinstance(size, int):
                            ends.append(off + size)
                    max_end = max(ends) if ends else None
                except Exception:
                    max_end = None
                if max_end is not None and max_end >= base_offset:
                    layer_bytes = frame_bytes[base_offset:max_end]
                    layer_hex = layer_bytes.hex()
                    debug["layer_hex_fallback"] = "frame_raw_sliced"
                else:
                    layer_bytes = frame_bytes
                    layer_hex = layer_bytes.hex()
                    base_offset = 0
                    debug["layer_hex_fallback"] = "frame_raw_full"
            else:
                layer_hex = ""
                base_offset = None
                debug["layer_hex_missing"] = True

        # Precompute bit-level spans for container prune.
        spans: List[Tuple[int, int, Dict[str, Any], int, int, str, WiresharkFieldInfo]] = []
        for e in field_entries:
            abs_off = e.get("abs_offset")
            size = e.get("size")
            if not isinstance(abs_off, int) or not isinstance(size, int) or size <= 0:
                continue
            if base_offset is None:
                continue
            rel = abs_off - base_offset
            if rel < 0:
                continue

            raw_name = str(e.get("raw_name") or "")
            abbrev = _abbrev_from_raw_key(raw_name)
            info = (field_infos or {}).get(abbrev) or WiresharkFieldInfo(display_name=_name_from_raw_key(raw_name), field_type="", bitmask_int=0)

            raw_bitmask_int = e.get("bitmask_int")
            bitmask_int = int(raw_bitmask_int) if (respect_bitmask and isinstance(raw_bitmask_int, int) and raw_bitmask_int) else 0
            if bitmask_int == 0:
                bitmask_int = int(getattr(info, "bitmask_int", 0) or 0) if respect_bitmask else 0
            bit_off_msb0 = 0
            size_bits = int(size * 8)
            assumption = "not_bitfield"
            if bitmask_int:
                bit_off_msb0, size_bits, assumption = bitmask_to_msb0_offset_and_size(bitmask_int, size)
                bitfield_defs.setdefault(
                    abbrev,
                    {
                        "abbrev": abbrev,
                        "display_name": str(getattr(info, "display_name", "") or ""),
                        "type": str(getattr(info, "field_type", "") or ""),
                        "bitmask": hex(int(bitmask_int)),
                        "bit_offset_msb0": int(bit_off_msb0),
                        "size_bits": int(size_bits),
                        "assumption": str(assumption),
                        "observed_size_bytes": int(size),
                    },
                )

            bit_start = int(rel * 8 + bit_off_msb0)
            spans.append((bit_start, bit_start + int(size_bits), e, int(rel), int(size), str(assumption), info))

        dropped_container = 0
        keep_entries: List[Dict[str, Any]] = []
        for bit_start, bit_end, e, _rel, _size, _assumption, _info in spans:
            is_container = False
            container_evidence = bool(e.get("has_child_tree")) or str(
                getattr(_info, "field_type", "")) in {"FT_NONE", "FT_PROTOCOL"}
            for other_start, other_end, _other_e, _r2, _s2, _a2, _i2 in spans if container_evidence else ():
                if (other_start, other_end) == (bit_start, bit_end):
                    continue
                if other_start >= bit_start and other_end <= bit_end:
                    is_container = True
                    break
            if is_container:
                dropped_container += 1
                continue
            keep_entries.append(e)

        # Build final field list with dedupe by bit-level key (bit_start, size_bits, value_int).
        dedup: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        dropped_negative = 0
        dropped_out_of_range = 0
        dropped_no_base = 0

        for e in keep_entries:
            abs_off = e.get("abs_offset")
            size = e.get("size")
            if not isinstance(abs_off, int) or not isinstance(size, int) or size <= 0:
                continue
            if base_offset is None:
                dropped_no_base += 1
                continue
            rel = abs_off - base_offset
            if rel < 0:
                dropped_negative += 1
                continue

            raw_bytes = None
            if layer_bytes:
                if rel + size <= len(layer_bytes):
                    raw_bytes = layer_bytes[rel : rel + size]
                else:
                    dropped_out_of_range += 1
                    continue
            else:
                raw_hex = _normalize_hex(e.get("raw_hex"))[: size * 2]
                if raw_hex:
                    try:
                        raw_bytes = bytes.fromhex(raw_hex)
                    except Exception:
                        raw_bytes = None

            raw_name = str(e.get("raw_name") or "")
            abbrev = _abbrev_from_raw_key(raw_name)
            info = (field_infos or {}).get(abbrev)
            name = str(getattr(info, "display_name", "") or "").strip() if info else ""
            if not name:
                name = _name_from_raw_key(raw_name)

            raw_bitmask_int = e.get("bitmask_int")
            bitmask_int = int(raw_bitmask_int) if (respect_bitmask and isinstance(raw_bitmask_int, int) and raw_bitmask_int) else 0
            if bitmask_int == 0:
                bitmask_int = int(getattr(info, "bitmask_int", 0) or 0) if (info and respect_bitmask) else 0
            bit_off_msb0 = 0
            size_bits = int(size * 8)
            if bitmask_int:
                bit_off_msb0, size_bits, _assumption = bitmask_to_msb0_offset_and_size(bitmask_int, size)
            bit_start = int(rel * 8 + bit_off_msb0)

            raw = raw_bytes.hex() if raw_bytes is not None else _normalize_hex(e.get("raw_hex"))[: size * 2]
            value_int = 0
            if raw_bytes is not None:
                try:
                    value_int = int(safe_value_int_msb0(raw_bytes, int(bit_off_msb0), int(size_bits)))
                except Exception:
                    try:
                        value_int = int(safe_value_int_msb0(raw_bytes, 0, int(len(raw_bytes) * 8)))
                    except Exception:
                        value_int = 0

            key = (bit_start, int(size_bits), int(value_int))

            entry = {
                "name": name,
                "abbrev": abbrev,
                "raw_name": raw_name,
                "offset": rel,
                "length": size,
                "raw": raw,
                "bit_start": int(bit_start),
                "size_bits": int(size_bits),
                "value_int": int(value_int),
                "bit_order": "MSB0",
                "value": int(value_int) if size_bits <= 64 else raw,
            }

            if key not in dedup:
                dedup[key] = entry
                continue

            # Prefer the more specific raw_name (deeper/longer) for the display name.
            prev = dedup[key]
            prev_name = str(prev.get("name") or "")
            prev_abbrev = str(prev.get("abbrev") or "")
            if len(abbrev) > len(prev_abbrev):
                prev.update(entry)
            elif len(abbrev) == len(prev_abbrev) and len(name) > len(prev_name):
                prev.update(entry)

        fields_out = sorted(dedup.values(), key=lambda x: (int(x.get("bit_start", 0)), int(x.get("size_bits", 0)), str(x.get("name", ""))))

        if debug:
            debug.update(
                {
                    "fields_raw_total": len(field_entries),
                    "fields_kept_after_container_prune": len(keep_entries),
                    "fields_dropped_container": dropped_container,
                    "fields_dropped_negative_offset": dropped_negative,
                    "fields_dropped_out_of_range": dropped_out_of_range,
                    "fields_dropped_no_base_offset": dropped_no_base,
                }
            )

        records.append(
            {
                "packet_no": packet_no,
                "hex": layer_hex,
                "base_offset": base_offset,
                "fields": fields_out,
                "debug": debug or {},
            }
        )

    return records, bitfield_defs


def _coalesce_repeated_raw_fields_in_place(record: Dict[str, Any]) -> None:
    fields_raw = record.get("fields")
    if not isinstance(fields_raw, list) or len(fields_raw) < 2:
        return

    pkt_hex = _normalize_hex(record.get("hex"))
    packet_bytes: Optional[bytes] = None
    if pkt_hex:
        try:
            packet_bytes = bytes.fromhex(pkt_hex)
        except Exception:
            packet_bytes = None

    # Group candidate fields by raw_name.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for f in fields_raw:
        if not isinstance(f, dict):
            continue
        raw_name = str(f.get("raw_name") or "").strip()
        if not raw_name:
            continue
        groups.setdefault(raw_name, []).append(f)

    if not groups:
        return

    def _field_span_bits(field: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        bs = _coerce_int(field.get("bit_start"))
        sz = _coerce_int(field.get("size_bits"))
        if bs is None or sz is None or sz <= 0:
            off = _coerce_int(field.get("offset"))
            ln = _coerce_int(field.get("length"))
            if off is None or ln is None or ln <= 0:
                return None
            return int(off) * 8, int(ln) * 8
        return int(bs), int(sz)

    # Build a map of field object ids to whether they should be dropped, and a list of merged replacements.
    to_drop: set[int] = set()
    replacements: List[Tuple[int, Dict[str, Any]]] = []  # (insert_bit_start, merged_field)

    for raw_name, group in groups.items():
        if len(group) < 2:
            continue

        spans: List[Tuple[int, int, Dict[str, Any]]] = []
        for f in group:
            span = _field_span_bits(f)
            if span is None:
                continue
            bs, sz = span
            if bs % 8 != 0 or sz % 8 != 0:
                continue
            spans.append((int(bs), int(sz), f))
        if len(spans) < 2:
            continue

        spans.sort(key=lambda t: (t[0], t[1]))

        i = 0
        while i < len(spans):
            start_bs, start_sz, start_f = spans[i]
            end_bs = start_bs + start_sz
            parts = [spans[i]]
            j = i + 1
            while j < len(spans):
                bs, sz, f = spans[j]
                if bs != end_bs:
                    break
                parts.append(spans[j])
                end_bs = bs + sz
                j += 1

            if len(parts) >= 2:
                merged_sz = end_bs - start_bs
                merged_offset = start_bs // 8
                merged_length = merged_sz // 8
                if merged_length > 0:
                    # Prefer a stable, GT-friendly merged name for known Wireshark Modbus expansions.
                    name = str(start_f.get("name") or "").strip() or _name_from_raw_key(str(start_f.get("raw_name") or ""))
                    raw_lower = raw_name.lower()
                    name_norm = re.sub(r"\s+", " ", str(name).strip().lower().replace("_", " ").replace("-", " "))
                    if "regval" in raw_lower or ("register" in name_norm and "value" in name_norm):
                        name = "Register Values"
                    elif name_norm == "bit value":
                        name = "Bit Values"

                    merged: Dict[str, Any] = {
                        "name": name,
                        "abbrev": str(start_f.get("abbrev") or ""),
                        "raw_name": str(start_f.get("raw_name") or raw_name),
                        "offset": int(merged_offset),
                        "length": int(merged_length),
                        "bit_start": int(start_bs),
                        "size_bits": int(merged_sz),
                        "bit_order": "MSB0",
                    }

                    raw_hex = ""
                    if packet_bytes is not None and 0 <= merged_offset <= len(packet_bytes) and merged_offset + merged_length <= len(packet_bytes):
                        raw_hex = packet_bytes[merged_offset : merged_offset + merged_length].hex()
                    else:
                        # Fall back to concatenating raw fragments when packet bytes are unavailable.
                        raws: List[str] = []
                        for _bs, _sz, f in parts:
                            r = _normalize_hex(f.get("raw"))
                            if not r:
                                raws = []
                                break
                            raws.append(r)
                        if raws:
                            raw_hex = "".join(raws)
                    if raw_hex:
                        merged["raw"] = raw_hex
                        try:
                            merged["value_int"] = int(safe_value_int_msb0(bytes.fromhex(raw_hex), 0, int(merged_sz)))
                        except Exception:
                            merged["value_int"] = 0

                    replacements.append((int(start_bs), merged))
                    for _bs, _sz, f in parts:
                        to_drop.add(id(f))

            i = j

    if not replacements:
        return

    # Replace the dropped fields with merged ones while keeping deterministic ordering by (bit_start, size_bits).
    kept: List[Dict[str, Any]] = []
    for f in fields_raw:
        if not isinstance(f, dict):
            continue
        if id(f) in to_drop:
            continue
        kept.append(f)

    merged_fields = kept + [mf for _bs, mf in replacements]
    merged_fields.sort(key=lambda f: (_coerce_int(f.get("bit_start")) or 0, _coerce_int(f.get("size_bits")) or 0, str(f.get("name") or "")))
    record["fields"] = merged_fields


def export_wireshark_tooljson(
    *,
    tshark_path: Path,
    pcap_path: Path,
    filter_name: str,
    out_dir: Path,
    export_jsonraw: bool = True,
    export_texttree: bool = True,
    max_packets: int = 0,
    extra_tshark_args: Optional[List[str]] = None,
    codegen_profile: Optional[Mapping[str, Any]] = None,
    ws_isolate_plugins: bool = False,
    ws_plugin_id: Optional[str] = None,
    base_layer: Optional[str] = None,
    field_layers: Optional[Sequence[str]] = None,
    respect_bitmask: bool = True,
    coalesce_repeated_raw_fields: bool = False,
) -> ExportTooljsonResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    tooljson_path = out_dir / "tool.json"
    jsonraw_path = out_dir / "tool.jsonraw" if export_jsonraw else None
    texttree_path = out_dir / "tool.txt" if export_texttree else None
    log_path = out_dir / "tshark_export.log"
    extra = [str(arg) for arg in (extra_tshark_args or [])]
    extra = _maybe_inject_reassembly_prefs(extra, codegen_profile=codegen_profile)
    frame_numbers: List[int] = []
    env: Optional[Dict[str, str]] = None
    if bool(ws_isolate_plugins):
        isolated_root = _ws_prepare_isolated_plugin_root(tshark_path=tshark_path, out_dir=out_dir, plugin_id=ws_plugin_id)
        if isolated_root is not None:
            env = dict(os.environ)
            try:
                env["WIRESHARK_PLUGIN_DIR"] = str(isolated_root.resolve())
            except Exception:
                env["WIRESHARK_PLUGIN_DIR"] = str(isolated_root)

    def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
        with _stage_root_lua_scripts_for_command(tshark_path=tshark_path, cmd=cmd, filter_name=filter_name) as staged_cmd:
            hide_plugins = (
                _ws_temporarily_hide_generated_plugins(tshark_path=tshark_path, keep_plugin_id=ws_plugin_id)
                if bool(ws_isolate_plugins)
                else nullcontext()
            )
            with hide_plugins:
                proc = subprocess.run(staged_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"$ {' '.join(staged_cmd)}\n")
                if env and env.get("WIRESHARK_PLUGIN_DIR"):
                    handle.write(f"[env] WIRESHARK_PLUGIN_DIR={env.get('WIRESHARK_PLUGIN_DIR')}\n")
                if proc.stdout:
                    handle.write(proc.stdout)
                if proc.stderr:
                    handle.write("\n[stderr]\n")
                    handle.write(proc.stderr)
                handle.write("\n")
        except Exception:
            pass
        if _requested_lua_script(extra_tshark_args) and proc.stderr and _stderr_has_lua_load_error(proc.stderr):
            raise RuntimeError(f"tshark reported Lua load error:\n{_tail_lines(proc.stderr)}")
        return proc

    if export_texttree:
        cmd: List[str] = [
            str(tshark_path),
            *extra,
            "-r",
            str(pcap_path),
            "-Y",
            filter_name,
            "-V",
        ]
        if max_packets and max_packets > 0:
            cmd.extend(["-c", str(int(max_packets))])
        proc = _run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"tshark texttree failed (rc={proc.returncode}): {proc.stderr.strip()}")
        assert texttree_path is not None
        texttree_path.write_text(proc.stdout, encoding="utf-8")
        frame_numbers = _parse_frame_numbers_from_texttree(proc.stdout)

    if not export_jsonraw:
        # Caller requested no JSON export; still return an empty tool.json for consistency.
        tooljson_path.write_text("[]\n", encoding="utf-8")
        return ExportTooljsonResult(tooljson_path=tooljson_path, jsonraw_path=None, texttree_path=texttree_path, packets=0)

    cmd_json: List[str] = [
        str(tshark_path),
        *extra,
        "-r",
        str(pcap_path),
        "-Y",
        filter_name,
        "-T",
        "jsonraw",
        "-x",
    ]
    if max_packets and max_packets > 0:
        cmd_json.extend(["-c", str(int(max_packets))])
    proc_json = _run(cmd_json)
    if proc_json.returncode != 0:
        raise RuntimeError(f"tshark jsonraw failed (rc={proc_json.returncode}): {proc_json.stderr.strip()}")

    if jsonraw_path is not None:
        jsonraw_path.write_text(proc_json.stdout, encoding="utf-8")

    try:
        payload = _json_loads_preserve_duplicates(proc_json.stdout)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse tshark jsonraw output: {exc}") from exc

    base_layer_name = str(base_layer or filter_name).strip()
    layers = [str(x).strip() for x in (field_layers or [filter_name]) if str(x).strip()]
    if not layers:
        layers = [filter_name]
    if base_layer_name not in layers:
        layers = [base_layer_name, *layers]

    field_infos: Dict[str, WiresharkFieldInfo] = {}
    for layer_name in dict.fromkeys(layers):
        layer_infos = _load_field_info_map(
            tshark_path=tshark_path,
            filter_name=layer_name,
            extra_tshark_args=extra_tshark_args,
            env=env,
        )
        # Abbreviations are namespaced by proto (e.g., mbtcp.len), so merging is safe.
        field_infos.update(layer_infos)

    records, observed_bitfields = flatten_jsonraw_to_tooljson_records(
        payload,
        base_layer_name,
        field_infos=field_infos,
        respect_bitmask=bool(respect_bitmask),
    )

    # Optional: merge in extra field layers on top of the base record offsets/bytes.
    if layers != [base_layer_name]:
        base_records = records
        base_offsets = [r.get("base_offset") for r in base_records if isinstance(r, dict)]
        for layer_name in layers:
            if layer_name == base_layer_name:
                continue
            layer_records, layer_bitfields = flatten_jsonraw_to_tooljson_records(
                payload,
                layer_name,
                field_infos=field_infos,
                respect_bitmask=bool(respect_bitmask),
            )
            observed_bitfields.update(layer_bitfields)

            if not layer_records:
                continue

            # Merge record-by-record; jsonraw order is stable for a fixed `-Y` filter.
            n = min(len(base_records), len(layer_records))
            for i in range(n):
                base_rec = base_records[i]
                layer_rec = layer_records[i]
                if not isinstance(base_rec, dict) or not isinstance(layer_rec, dict):
                    continue
                delta = 0
                bo = base_rec.get("base_offset")
                lo = layer_rec.get("base_offset")
                if isinstance(bo, int) and isinstance(lo, int):
                    delta = int(lo - bo)
                if not delta:
                    # If base offsets are missing, do not attempt to shift; still append.
                    delta = 0

                merged_fields: List[Dict[str, Any]] = []
                base_fields = base_rec.get("fields")
                if isinstance(base_fields, list):
                    merged_fields.extend([f for f in base_fields if isinstance(f, dict)])

                layer_fields = layer_rec.get("fields")
                if isinstance(layer_fields, list):
                    for f in layer_fields:
                        if not isinstance(f, dict):
                            continue
                        f2 = dict(f)
                        if isinstance(f2.get("offset"), int):
                            f2["offset"] = int(f2["offset"]) + int(delta)
                        if isinstance(f2.get("bit_start"), int):
                            f2["bit_start"] = int(f2["bit_start"]) + int(delta) * 8
                        merged_fields.append(f2)

                base_rec["fields"] = merged_fields
                dbg = base_rec.get("debug")
                if not isinstance(dbg, dict):
                    dbg = {}
                merged = dbg.get("merged_layers")
                if not isinstance(merged, list):
                    merged = []
                merged.append({"layer": layer_name, "delta_bytes": int(delta)})
                dbg["merged_layers"] = merged
                base_rec["debug"] = dbg

        # Reassign for clarity.
        records = base_records
        _ = base_offsets

    if bool(coalesce_repeated_raw_fields):
        for rec in records:
            if isinstance(rec, dict):
                _coalesce_repeated_raw_fields_in_place(rec)

    packet_no_source: Optional[str] = "tshark_texttree" if frame_numbers else None
    if not frame_numbers and records:
        # jsonraw output frequently omits `frame.number` (and `frame.number_raw` can be empty),
        # so when the caller disables `-V` export we still recover stable packet numbers via
        # a cheap `-T fields` query.
        cmd_frames: List[str] = [
            str(tshark_path),
            *extra,
            "-r",
            str(pcap_path),
            "-Y",
            filter_name,
            "-T",
            "fields",
            "-e",
            "frame.number",
        ]
        if max_packets and max_packets > 0:
            cmd_frames.extend(["-c", str(int(max_packets))])
        proc_frames = _run(cmd_frames)
        if proc_frames.returncode == 0:
            nums = _parse_frame_numbers_from_fields(proc_frames.stdout)
            if nums:
                frame_numbers = nums
                packet_no_source = "tshark_fields"

    if packet_no_source and frame_numbers and len(frame_numbers) == len(records):
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            rec["packet_no"] = int(frame_numbers[idx])
            debug = rec.get("debug")
            if not isinstance(debug, dict):
                debug = {}
            debug["packet_no_source"] = packet_no_source
            rec["debug"] = debug

    # Align tool outputs with the evaluation/extraction scope encoded in the codegen profile.
    # For `pcap_layer=tcp_header`, only the TCP header(+options) bytes are in-scope, so drop
    # fields that span beyond the header boundary (e.g., payload "Data").
    if _pcap_layer_from_profile(codegen_profile) == "tcp_header":
        for rec in records:
            if not isinstance(rec, dict):
                continue
            layer_hex = str(rec.get("hex") or "")
            hdr_len = _tcp_header_len_from_layer_hex(layer_hex)
            if hdr_len is None:
                continue
            rec["hex"] = _normalize_hex(layer_hex)[: hdr_len * 2]

            fields_raw = rec.get("fields") or []
            if isinstance(fields_raw, list):
                kept: List[Dict[str, Any]] = []
                for field in fields_raw:
                    if not isinstance(field, dict):
                        continue
                    off = _coerce_int(field.get("offset"))
                    ln = _coerce_int(field.get("length"))
                    if off is None or ln is None:
                        continue
                    if off < 0 or ln <= 0:
                        continue
                    if off + ln > hdr_len:
                        continue
                    kept.append(field)
                rec["fields"] = kept

            debug = rec.get("debug")
            if not isinstance(debug, dict):
                debug = {}
            debug["pcap_layer"] = "tcp_header"
            debug["tcp_header_len_bytes"] = int(hdr_len)
            rec["debug"] = debug


    tooljson_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Write bitfield_map.json for downstream bit-span upgrades/debugging.
    entries: List[Dict[str, Any]] = []
    for abbrev, info in sorted(field_infos.items(), key=lambda kv: kv[0]):
        if int(getattr(info, "bitmask_int", 0) or 0) == 0:
            continue
        entry = dict(observed_bitfields.get(abbrev) or {})
        if not entry:
            size_guess = _guess_size_bytes_from_type(getattr(info, "field_type", "") or "") or 1
            off, sz, assumption = bitmask_to_msb0_offset_and_size(int(info.bitmask_int), int(size_guess))
            entry = {
                "abbrev": abbrev,
                "display_name": str(getattr(info, "display_name", "") or ""),
                "type": str(getattr(info, "field_type", "") or ""),
                "bitmask": hex(int(info.bitmask_int)),
                "bit_offset_msb0": int(off),
                "size_bits": int(sz),
                "assumption": str(assumption),
                "observed_size_bytes": None,
            }
        entries.append(entry)

    bitfield_map_payload = {
        "filter_name": str(filter_name),
        "base_layer": str(base_layer_name),
        "field_layers": list(dict.fromkeys(layers)),
        "bit_order": "MSB0",
        "generated_by": "wireshark_export",
        "respect_bitmask": bool(respect_bitmask),
        "entries": entries,
    }
    (out_dir / "bitfield_map.json").write_text(
        json.dumps(bitfield_map_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return ExportTooljsonResult(
        tooljson_path=tooljson_path,
        jsonraw_path=jsonraw_path,
        texttree_path=texttree_path,
        packets=len(records),
    )
