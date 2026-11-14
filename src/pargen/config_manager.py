"""Helpers for loading API keys and common settings from pargen/config."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .paths import DEFAULT_API_CONFIG, EXAMPLE_API_CONFIG


def load_api_keys(config_path: Optional[Path] = None, set_env: bool = True) -> Dict[str, str]:
    """
    Load API keys and provider endpoint settings from a JSON config file and
    optionally populate os.environ.

    The config file structure is expected to be::

        {
          "api_keys": {
            "DEEPSEEK_API_KEY": "...",
            ...
          },
          "base_urls": {
            "DEEPSEEK_BASE_URL": "..."
          },
          "models": {
            "DEEPSEEK_MODEL": "deepseek-v4-flash-0731"
          }
        }

    Empty strings and placeholder values are ignored.
    """

    def _is_placeholder(value: str) -> bool:
        raw = (value or "").strip().lower()
        if not raw:
            return True
        return raw.startswith("paste-your-") or raw in {"your-api-key", "changeme", "replace-me"}

    path = Path(config_path) if config_path else DEFAULT_API_CONFIG
    if not path.exists():
        # if the real file does not exist but the example is present, surface a hint
        if EXAMPLE_API_CONFIG.exists() and config_path is None:
            print(f"[config] API config not found at {path}. Copy {EXAMPLE_API_CONFIG} and fill in your keys.")
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    applied: Dict[str, str] = {}
    for section_name in ("api_keys", "base_urls", "models"):
        values = payload.get(section_name)
        if values is None and section_name == "api_keys":
            values = payload
        if not isinstance(values, dict):
            continue
        for env_name, value in values.items():
            if not value or _is_placeholder(str(value)):
                continue
            applied[env_name] = str(value)
            if set_env:
                os.environ.setdefault(env_name, str(value))
    return applied


__all__ = ["load_api_keys"]
