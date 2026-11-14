"""Shared filesystem locations for the standalone pargen package."""

from __future__ import annotations

import os
from pathlib import Path


PARGEN_ROOT = Path(__file__).resolve().parent


def _default_data_dir() -> Path:
    env_value = os.getenv("PARGEN_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser()
    artifact_data = PARGEN_ROOT.parents[1] / "dataset" / "pargen" / "data"
    if artifact_data.exists():
        return artifact_data
    return PARGEN_ROOT / "data"


DATA_DIR = _default_data_dir()
CONFIG_DIR = PARGEN_ROOT / "config"

# Keep immutable input/configuration paths independent of writable runtime state.
# Existing local workflows retain their default locations when no override is set.
_runtime_setting = os.getenv("PARGEN_RUNTIME_ROOT")
RUNTIME_ROOT = Path(_runtime_setting) if _runtime_setting else None
if RUNTIME_ROOT is not None and not RUNTIME_ROOT.is_absolute():
    raise ValueError("PARGEN_RUNTIME_ROOT must be an absolute path")
ARTIFACTS_ROOT = RUNTIME_ROOT / "artifacts" if RUNTIME_ROOT is not None else DATA_DIR / "_artifacts"
LOGS_DIR = RUNTIME_ROOT / "logs" if RUNTIME_ROOT is not None else DATA_DIR / "logs"
DOC_CACHE_DIR = ARTIFACTS_ROOT / "doc_cache"
DOC_CHUNKS_DIR = ARTIFACTS_ROOT / "doc_chunks"
STEP2_CACHE_DIR = ARTIFACTS_ROOT / "step2_cache"
STEP2_RESULTS_DIR = ARTIFACTS_ROOT / "step2_results"


def _sanitize_cache_namespace(value: str) -> str:
    cleaned = []
    for ch in value.strip():
        if ch.isalnum() or ch in "._-":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    return out or "default"


_STEP2_FIX_CACHE_BASE_DIR = ARTIFACTS_ROOT / "step2_fix_cache"
_step2_namespace = os.getenv("PARGEN_STEP2_CACHE_NAMESPACE")
if _step2_namespace and RUNTIME_ROOT is None:
    LOGS_DIR = DATA_DIR / _sanitize_cache_namespace(_step2_namespace) / "logs"
STEP2_FIX_CACHE_DIR = (
    _STEP2_FIX_CACHE_BASE_DIR / _sanitize_cache_namespace(_step2_namespace)
    if _step2_namespace
    else _STEP2_FIX_CACHE_BASE_DIR
)
STEP3_LOG_DIR = LOGS_DIR / "step3_runs"
REALFLOW_REPORT_DIR = LOGS_DIR / "realflow_reports"
PCAP_DIR = ARTIFACTS_ROOT / "pcap_samples"
DEFAULT_API_CONFIG = CONFIG_DIR / "api_config.json"
EXAMPLE_API_CONFIG = CONFIG_DIR / "api_config.example.json"


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


_ensure_dirs(
    (
        ARTIFACTS_ROOT,
        LOGS_DIR,
        DOC_CACHE_DIR,
        DOC_CHUNKS_DIR,
        STEP2_CACHE_DIR,
        STEP2_RESULTS_DIR,
        STEP2_FIX_CACHE_DIR,
        STEP3_LOG_DIR,
        REALFLOW_REPORT_DIR,
        PCAP_DIR,
    )
)
if RUNTIME_ROOT is None:
    _ensure_dirs((CONFIG_DIR, DATA_DIR))


__all__ = [
    "PARGEN_ROOT",
    "RUNTIME_ROOT",
    "CONFIG_DIR",
    "ARTIFACTS_ROOT",
    "DOC_CACHE_DIR",
    "DOC_CHUNKS_DIR",
    "STEP2_CACHE_DIR",
    "STEP2_RESULTS_DIR",
    "STEP2_FIX_CACHE_DIR",
    "LOGS_DIR",
    "STEP3_LOG_DIR",
    "REALFLOW_REPORT_DIR",
    "PCAP_DIR",
    "DATA_DIR",
    "DEFAULT_API_CONFIG",
    "EXAMPLE_API_CONFIG",
]
