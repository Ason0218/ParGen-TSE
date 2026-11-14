"""LLM agent wrapper used by step2 for protocol tree construction."""

from __future__ import annotations

import json
import logging
import os
import time
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

import requests

from ..construction.step2_format_graph_builder import ProtocolNode, canonicalize_node_type
from ...paths import STEP2_CACHE_DIR


logger = logging.getLogger(__name__)

DEFAULT_AI_READ_TIMEOUT_SEC = 900
DEFAULT_AI_PROVIDER_WALL_TIMEOUT_SEC = 960.0


def _env_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _run_with_wall_timeout(fn, *, timeout_sec: float, label: str):
    if timeout_sec <= 0:
        return fn()
    result: Dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # propagate provider exceptions to caller
            result["exception"] = exc

    worker = threading.Thread(
        target=_target,
        name=f"provider-wall-timeout-{label}",
        daemon=True,
    )
    worker.start()
    worker.join(timeout_sec)
    if worker.is_alive():
        raise TimeoutError(f"{label} exceeded wall-clock timeout {timeout_sec:g}s")
    if "exception" in result:
        raise result["exception"]
    return result.get("value")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy types."""

    def default(self, obj):  # type: ignore[override]
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _extract_fenced_json_block(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_balanced_json_object(text: str) -> str:
    if not text:
        return ""

    starts = [idx for idx, ch in enumerate(text) if ch in "{["]
    for start in starts:
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == opening:
                depth += 1
                continue
            if ch == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1].strip()
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break
        # try next possible opening brace
    return ""


class EnhancedPureAIAgent:
    """Enhanced AI agent supporting multiple providers."""

    def __init__(
        self,
        provider: str = "openai",
        cache_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider

        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions").strip()
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self.default_model = (os.getenv("STEP2_AGENT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5").strip()
        elif self.provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                # Fall back to OPENAI_API_KEY to ease reuse of settings across stages if provided
                self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable must be set")
            self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions").strip()
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            model_override = (
                os.getenv("STEP2_AGENT_MODEL")
                or os.getenv("DEEPSEEK_MODEL")
                or os.getenv("AI_MODEL_OVERRIDE")
            )
            self.default_model = (model_override or "deepseek-v4-flash-0731").strip()
        else:
            self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
            if not self.api_key:
                raise ValueError("CLAUDE_API_KEY environment variable must be set")
            self.base_url = "https://api.anthropic.com/v1/messages"
            self.headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
            }
            model_override = (
                os.getenv("CLAUDE_MODEL")
                or os.getenv("ANTHROPIC_MODEL")
                or os.getenv("AI_MODEL_OVERRIDE")
            )
            self.default_model = (model_override or "claude-sonnet-4-6").strip()

        self.connect_timeout = int(os.getenv("AI_CONNECT_TIMEOUT", "10"))
        self.read_timeout = int(os.getenv("AI_READ_TIMEOUT", str(DEFAULT_AI_READ_TIMEOUT_SEC)))
        self.provider_wall_timeout = _env_positive_float(
            "AI_PROVIDER_WALL_TIMEOUT_SEC",
            DEFAULT_AI_PROVIDER_WALL_TIMEOUT_SEC,
        )
        self.timeout = self.read_timeout
        self.max_retries = int(os.getenv("AI_MAX_RETRIES", "2"))
        if os.getenv("AI_MAX_RETRIES") is None:
            for budget_env in ("STEP2_MCTS_MAX_CALLS", "STEP2_MCTS_MAX_LLM_CALLS", "INLINE_FIX_MAX_CALLS"):
                raw_budget = os.getenv(budget_env)
                if raw_budget is None:
                    continue
                try:
                    if int(raw_budget.strip()) <= 1:
                        self.max_retries = 1
                        break
                except Exception:
                    continue
        self.retry_delay = int(os.getenv("AI_RETRY_DELAY", "3"))

        cache_root = cache_dir or STEP2_CACHE_DIR
        self.cache_dir = str(cache_root)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.api_call_times: List[Dict[str, Any]] = []
        self.total_api_time = 0.0
        self.step2_start_time: Optional[float] = None

        self.structural_validator = None
        self.validator_integration = None
        self._current_learning_feedback_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Cache helpers
    def _clean_raw_response(self, raw_response: str) -> str:
        if not raw_response:
            return ""

        clean_text = raw_response.strip()
        clean_text = _strip_deepseek_prefixes(clean_text)
        fenced_json = _extract_fenced_json_block(clean_text)
        if fenced_json:
            clean_text = fenced_json
        else:
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]

        clean_text = clean_text.strip()
        if clean_text.startswith('"') and clean_text.endswith('"'):
            clean_text = clean_text[1:-1]
        elif clean_text.startswith("'") and clean_text.endswith("'"):
            clean_text = clean_text[1:-1]

        clean_text = _strip_js_comments(clean_text)
        clean_text = clean_text.strip()
        try:
            clean_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", clean_text)
        except Exception:
            pass
        if clean_text and not clean_text.startswith("{") and not clean_text.startswith("["):
            extracted_json = _extract_balanced_json_object(clean_text)
            if extracted_json:
                clean_text = extracted_json
        return clean_text

    def _structured_response_format(self) -> Optional[Dict[str, Any]]:
        """
        Build a loose JSON schema for protocol tree responses so that providers
        (OpenAI / DeepSeek) emit machine-parseable JSON objects.
        """

        if self.provider == "deepseek":
            return {"type": "json_object"}

        if self.provider != "openai":
            return None

        def _id_schema() -> Dict[str, Any]:
            return {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "string"},
                ]
            }

        node_schema = {
            "type": "object",
            "properties": {
                "node_id": _id_schema(),
                "name": {"type": ["string", "null"]},
                "node_type": {"type": "string"},
                "message_type": {"type": ["string", "null"]},
                "bit_start": {"type": ["string", "integer", "null"]},
                "size_bits": {"type": ["string", "integer", "null"]},
                "data_type": {"type": ["string", "null"]},
                "byte_order": {"type": ["string", "null"]},
                "parent_id": {
                    "anyOf": [
                        _id_schema(),
                        {"type": "null"},
                    ]
                },
                "children_ids": {
                    "type": "array",
                    "items": _id_schema(),
                },
                "constraints": {"type": "array"},
                "dependencies": {"type": "array"},
                "source": {"type": ["string", "null"]},
            },
            "required": ["node_id", "node_type", "children_ids"],
            "additionalProperties": True,
        }

        edge_schema = {
            "type": "object",
            "properties": {
                "src": _id_schema(),
                "dst": _id_schema(),
                "rel": {"type": "string"},
                "formula": {"type": ["string", "null"]},
                "message_type": {"type": ["string", "null"]},
            },
            "required": ["src", "dst", "rel"],
            "additionalProperties": True,
        }

        protocol_tree_schema = {
            "type": "object",
            "properties": {
                "root_node_id": _id_schema(),
                "nodes": {"type": "array", "items": node_schema},
                "edges": {"type": "array", "items": edge_schema},
            },
            "required": ["root_node_id", "nodes", "edges"],
            "additionalProperties": True,
        }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "protocol_tree_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "protocol_tree": protocol_tree_schema,
                    },
                    "required": ["protocol_tree"],
                    "additionalProperties": True,
                },
            },
        }

    def _load_from_cache(self, filename: str) -> Optional[Any]:
        if os.getenv("PARGEN_STEP2_FORCE_REBUILD", "").strip().lower() in {"1", "true", "yes", "on"}:
            return None
        cache_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            logger.info("Loaded from cache: %s", cache_path)
            return data
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load cache %s: %s", cache_path, exc)
            return None

    def _save_to_cache(self, filename: str, data: Any) -> None:
        cache_path = os.path.join(self.cache_dir, filename)
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            output_data: Any = data

            if isinstance(data, dict) and "raw_response" in data:
                raw_response = data["raw_response"]
                if isinstance(raw_response, str):
                    clean_response = self._clean_raw_response(raw_response)
                    try:
                        parsed_response = json.loads(clean_response) if clean_response else None
                    except (json.JSONDecodeError, ValueError):
                        parsed_response = None

                    if parsed_response is not None:
                        if isinstance(parsed_response, dict) and "protocol_tree" in parsed_response:
                            output_data = {
                                "protocol_tree": parsed_response["protocol_tree"],
                                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                                "section_info": data.get("section_info"),
                            }
                        else:
                            output_data = {
                                "protocol_tree": parsed_response,
                                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                                "section_info": data.get("section_info"),
                            }
                    else:
                        output_data = {
                            "raw_response": raw_response,
                            "timestamp": data.get("timestamp", datetime.now().isoformat()),
                            "section_info": data.get("section_info"),
                        }
            elif isinstance(data, dict) and ("nodes" in data or "root_node_id" in data):
                output_data = {
                    "protocol_tree": data,
                    "timestamp": datetime.now().isoformat(),
                    "section_info": None,
                }

            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(output_data, handle, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            logger.info("Saved to cache: %s", cache_path)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to save cache %s: %s", filename, exc)
            raise

    # ------------------------------------------------------------------
    # Validation helpers
    def _update_node(self, node: ProtocolNode, node_data: Dict[str, Any]) -> None:
        if "name" in node_data:
            node.name = node_data["name"]
        if "node_type" in node_data:
            node.node_type = canonicalize_node_type(node_data["node_type"])
        if "bit_start" in node_data:
            node.bit_start = node_data["bit_start"]
        if "size_bits" in node_data:
            node.size_bits = node_data["size_bits"]
        if "message_type" in node_data:
            node.message_type = node_data["message_type"] or ""
        if "data_type" in node_data:
            node.data_type = node_data["data_type"]
        if "byte_order" in node_data:
            node.byte_order = node_data["byte_order"]
        if "parent_id" in node_data:
            pid = node_data["parent_id"]
            node.parent_id = str(pid) if pid is not None else None
        if "children_ids" in node_data:
            node.children_ids = [str(child) for child in node_data["children_ids"] or []]
        if "source" in node_data:
            node.source = node_data["source"]

        if "constraints" in node_data:
            node.constraints = node_data["constraints"]

        if "dependencies" in node_data:
            node.dependencies = node_data["dependencies"]

    def _apply_validation_results_to_tree(self, cumulative_tree: Dict, validation_result: Dict) -> Dict:
        merged_tree = cumulative_tree.copy()
        existing_nodes = {str(node["node_id"]): node for node in merged_tree.get("nodes", [])}

        for conflict in validation_result.get("conflicts_detected", []):
            resolution = conflict.get("resolution", {})
            merged_node = resolution.get("merged_node")
            if merged_node:
                existing_nodes[str(merged_node["node_id"])] = merged_node

        for preserved in validation_result.get("learning_preserved", []):
            new_node = preserved.get("new_node")
            if not new_node:
                continue
            new_id = str(new_node["node_id"])
            while new_id in existing_nodes:
                new_id = f"{new_id}_preserved"
            new_node["node_id"] = new_id
            existing_nodes[new_id] = new_node

        merged_tree["nodes"] = list(existing_nodes.values())
        return merged_tree

    def _merge_batch_into_cumulative_tree(self, cumulative_tree: Dict, batch_result: Dict) -> Dict:
        from position_conflict_detector import PositionConflictDetector

        detector = PositionConflictDetector()
        result = detector.merge_trees_with_position_conflict_detection(cumulative_tree, batch_result)
        logger.info("Completed merge with position conflict detection")
        return result

    # ------------------------------------------------------------------
    # API calls
    def _call_api_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from ...openrouter_routing import routed_payload
        from ... import degenerate_retry
        from ...paths import STEP2_CACHE_DIR
        payload = routed_payload(payload, self.base_url)
        built_payload = payload
        # A retry of a request whose greedy reply was degenerate is sampled (STEP2_DEGENERATE_RETRY_SAMPLING).
        payload, retry_sampling = degenerate_retry.apply(built_payload)
        api_start_time = time.time()
        model_name = payload.get("model", "unknown")

        for attempt in range(self.max_retries):
            try:
                try:
                    approx_chars = len(json.dumps(payload))
                except Exception:
                    approx_chars = -1
                logger.info(
                    "Calling provider %s (attempt %s/%s), timeouts=(%ss connect, %ss read, %ss wall), payload≈%s chars",
                    model_name,
                    attempt + 1,
                    self.max_retries,
                    self.connect_timeout,
                    self.read_timeout,
                    self.provider_wall_timeout,
                    approx_chars,
                )

                response = _run_with_wall_timeout(
                    lambda: requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=(self.connect_timeout, self.read_timeout),
                    ),
                    timeout_sec=self.provider_wall_timeout,
                    label=f"provider {model_name} request",
                )

                if response.status_code == 200:
                    response_data = response.json()
                    choices = response_data.get("choices") or []
                    finish = choices[0].get("finish_reason") if choices else response_data.get("stop_reason")
                    from ...paths import LOGS_DIR
                    usage_path = Path(os.environ.get("PARGEN_USAGE_LOG", str(LOGS_DIR / "step2_usage_log.jsonl")))
                    usage_path.parent.mkdir(parents=True, exist_ok=True)
                    with usage_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                            "stage": "repair", "provider": self.provider, "model": response_data.get("model", model_name),
                            "finish_reason": finish, "usage": response_data.get("usage"),
                            "requested_provider": payload.get('provider'),
                            "upstream_provider": response_data.get('provider'),
                            "generation_id": response_data.get('id'),
                            "degenerate_retry_sampling": retry_sampling,
                            "attempt": attempt + 1}, ensure_ascii=False) + "\n")
                    degenerate_retry.note(built_payload, response_data, retry_sampling,
                                         log_dir=STEP2_CACHE_DIR / (os.getenv('PARGEN_STEP2_CACHE_NAMESPACE') or ''))
                    if response_data.get("error") or finish in {"error", "content_filter", "cancelled"}:
                        raise RuntimeError(f"Repair provider failure: finish_reason={finish!r}; error={response_data.get('error')}")
                    api_duration = time.time() - api_start_time
                    self.api_call_times.append(
                        {
                            "model": model_name,
                            "duration": api_duration,
                            "attempt": attempt + 1,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    self.total_api_time += api_duration
                    logger.info("API call succeeded: %s (%.2fs, attempt %s)", model_name, api_duration, attempt + 1)
                    return response_data

                logger.warning("API error %s: %s", response.status_code, response.text[:1000])
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

            except Exception as exc:
                api_duration = time.time() - api_start_time
                logger.warning("API call failed (attempt %s): %s (%.2fs)", attempt + 1, exc, api_duration)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.api_call_times.append(
                        {
                            "model": model_name,
                            "duration": api_duration,
                            "attempt": attempt + 1,
                            "timestamp": datetime.now().isoformat(),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    raise

        final_duration = time.time() - api_start_time
        self.api_call_times.append(
            {
                "model": model_name,
                "duration": final_duration,
                "attempt": self.max_retries,
                "timestamp": datetime.now().isoformat(),
                "status": "failed_all_retries",
            }
        )
        raise RuntimeError("API call failed after all retries")


def _strip_deepseek_prefixes(text: str) -> str:
    import re

    text = re.sub(r"极抱歉，.*?让我重新提供正确的JSON。?\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"抱歉，.*?正确.*?JSON.*?\s*", "", text, flags=re.DOTALL)
    return text


def _strip_js_comments(text: str) -> str:
    import re

    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    return text
