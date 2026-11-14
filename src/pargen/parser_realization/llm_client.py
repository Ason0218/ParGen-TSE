from __future__ import annotations

"""Minimal LLM client wrapper used by Stage C agents."""

import json
import logging
import os
import time
import urllib.request
from http.client import IncompleteRead
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

try:  # kept at module scope so tests can patch pargen.parser_realization.llm_client.httpx.Client
    import httpx
except Exception:  # pragma: no cover - requests fallback handles this path
    httpx = None  # type: ignore[assignment]

from ..config_manager import load_api_keys
from ..paths import DEFAULT_API_CONFIG

logger = logging.getLogger(__name__)


class EmptyCompletionError(RuntimeError):
    """The provider answered without any text, e.g. reasoning used the whole output budget."""


class _WallClockTimeout(RuntimeError):
    """One request attempt ran past PARGEN_LLM_WALL_TIMEOUT_S."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class LLMClient(Protocol):
    def complete(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - interface
        ...


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 8000


@dataclass
class LLMCompletion:
    content: str
    raw: Optional[dict[str, Any]] = None
    token_logprobs: Optional[list[dict[str, Any]]] = None


def resolve_effective_llm_request(config: Optional[LLMConfig] = None) -> dict[str, Any]:
    """Resolve provider/model and the actual max_tokens sent to the API."""

    cfg = config or LLMConfig()
    provider = _normalize_provider(getattr(cfg, "provider", None) or os.getenv("LLM_PROVIDER") or "deepseek")
    requested = int(getattr(cfg, "max_tokens", 0) or 0) or 8000

    if provider == "deepseek":
        model = (getattr(cfg, "model", None) or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash-0731").strip()
    elif provider == "anthropic":
        configured = (getattr(cfg, "model", None) or "").strip()
        if configured == "deepseek-chat":
            configured = ""
        model = configured or (os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL") or "claude-sonnet-4-6").strip()
    else:
        model = (getattr(cfg, "model", None) or os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

    effective = requested
    cap: Optional[int] = None
    cap_source: Optional[str] = None
    if provider == "deepseek":
        model_key = (model or "").strip().lower()
        if model_key == "deepseek-reasoner":
            cap = 64000
            cap_source = "default_deepseek_reasoner_cap"
            for env_name in ("DEEPSEEK_REASONER_MAX_TOKENS_CAP", "DEEPSEEK_MAX_TOKENS_CAP"):
                raw_cap = os.getenv(env_name)
                if not raw_cap:
                    continue
                try:
                    parsed_cap = int(raw_cap)
                except Exception:
                    logger.warning("Ignoring invalid %s=%r", env_name, raw_cap)
                    continue
                if parsed_cap > 0:
                    cap = parsed_cap
                    cap_source = env_name
                    break
            effective = min(requested, cap)
        elif "flash" in model_key or model_key == "deepseek-chat":
            cap = 8192
            cap_source = "default_deepseek_flash_cap"
            for env_name in ("DEEPSEEK_FLASH_MAX_TOKENS_CAP", "DEEPSEEK_MAX_TOKENS_CAP"):
                raw_cap = os.getenv(env_name)
                if not raw_cap:
                    continue
                try:
                    parsed_cap = int(raw_cap)
                except Exception:
                    logger.warning("Ignoring invalid %s=%r", env_name, raw_cap)
                    continue
                if parsed_cap > 0:
                    cap = parsed_cap
                    cap_source = env_name
                    break
            effective = min(requested, cap)
        else:
            cap = 64000
            cap_source = "default_deepseek_pro_cap"
            for env_name in ("DEEPSEEK_PRO_MAX_TOKENS_CAP", "DEEPSEEK_MAX_TOKENS_CAP"):
                raw_cap = os.getenv(env_name)
                if not raw_cap:
                    continue
                try:
                    parsed_cap = int(raw_cap)
                except Exception:
                    logger.warning("Ignoring invalid %s=%r", env_name, raw_cap)
                    continue
                if parsed_cap > 0:
                    cap = parsed_cap
                    cap_source = env_name
                    break
            effective = min(requested, cap)

    payload: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "temperature": getattr(cfg, "temperature", None),
        "max_tokens": effective,
        "max_tokens_requested": requested,
    }
    if cap is not None:
        payload["max_tokens_cap"] = cap
        payload["max_tokens_cap_source"] = cap_source
    return payload


def create_llm_client(config: Optional[LLMConfig] = None, ai_client: Optional[Any] = None) -> LLMClient:
    cfg = config or LLMConfig()
    if ai_client:
        return _AIClientWrapper(ai_client, max_tokens=cfg.max_tokens, temperature=cfg.temperature)
    load_api_keys(DEFAULT_API_CONFIG, set_env=True)
    _load_model_config_env()
    return _EnvLLMClient(cfg)


def _normalize_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in {"claude"}:
        return "anthropic"
    return p or "deepseek"


def _is_placeholder_secret(value: str) -> bool:
    raw = (value or "").strip().lower()
    return not raw or raw.startswith("paste-your-") or raw in {"your-api-key", "changeme", "replace-me"}


def _pick_env(*names: str) -> str:
    for name in names:
        val = (os.getenv(name) or "").strip()
        if val and not _is_placeholder_secret(val):
            return val
    return ""


class _AIClientWrapper:
    """Bridge unified_ai_client.AIClient-like interface to the .complete contract."""

    def __init__(self, client: Any, max_tokens: int = 8000, temperature: float = 0.2) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.usage_records: list[dict[str, Any]] = []

    def complete(self, prompt: str, **_: Any) -> str:
        resp = self.client.invoke_model(
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = resp.get("content", "")
        if isinstance(content, list):
            content = content[0].get("text", "") if content else ""
        if not content and isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
        if not content:
            raise EmptyCompletionError("LLM response has no text content")
        return str(content)

    def complete_with_metadata(self, prompt: str, **kwargs: Any) -> LLMCompletion:
        return LLMCompletion(content=self.complete(prompt, **kwargs), raw=None, token_logprobs=None)

    def usage_summary(self) -> dict[str, Any]:
        return _usage_summary_from_records(self.usage_records)


class _EnvLLMClient:
    """Simple HTTP client for OpenAI-compatible chat endpoints."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        provider = _normalize_provider(config.provider or os.getenv("LLM_PROVIDER") or "deepseek")
        self.provider = provider

        if provider == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY") or ""
            self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
            self.model = (config.model or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash-0731").strip()
        elif provider == "anthropic":
            # Prefer CLAUDE_API_KEY (used by Stage A) over ANTHROPIC_API_KEY.
            self.api_key = _pick_env("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")
            self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")
            configured = (config.model or "").strip()
            if configured == "deepseek-chat":
                configured = ""
            self.model = configured or (os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL") or "claude-sonnet-4-6").strip()
        else:
            self.api_key = os.getenv("OPENAI_API_KEY") or ""
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
            self.model = (config.model or os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

        if not self.api_key:
            _load_model_config_env()
            if provider == "anthropic":
                self.api_key = _pick_env("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")
            elif provider == "deepseek":
                self.api_key = _pick_env("DEEPSEEK_API_KEY")
            else:
                self.api_key = _pick_env("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(f"Missing API key for LLM provider '{provider}'. Set *_API_KEY env or update {DEFAULT_API_CONFIG}.")

        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.usage_records: list[dict[str, Any]] = []

    def _effective_max_tokens(self) -> int:
        requested = int(self.max_tokens or 0) or 8000
        resolved = resolve_effective_llm_request(
            LLMConfig(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=requested,
            )
        )
        effective = int(resolved.get("max_tokens") or requested)
        if effective < requested:
            logger.warning(
                "Capping %s max_tokens from %s to %s for model %s",
                self.provider,
                requested,
                effective,
                self.model,
            )
        return effective

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.complete_with_metadata(prompt, **kwargs).content

    def complete_with_metadata(self, prompt: str, **kwargs: Any) -> LLMCompletion:
        # A thinking model at T=0 can reason until the output cap and return no answer. Such a reply is
        # resent once with STEP2_DEGENERATE_RETRY_SAMPLING (pargen/degenerate_retry.py); the first request
        # keeps the configured decoding, and with the variable unset nothing changes.
        try:
            return self._complete_with_metadata_once(prompt, **kwargs)
        except EmptyCompletionError as first_exc:
            from .. import degenerate_retry

            sampling = degenerate_retry.settings() if self.provider != "anthropic" else None
            if not sampling:
                raise
            logger.warning("Degenerate reply (%s); resending once with sampling %s", first_exc, sampling)
            still_degenerate = True
            try:
                result = self._complete_with_metadata_once(prompt, _sampling=sampling, **kwargs)
                still_degenerate = False
                return result
            finally:
                log_path = os.getenv("PARGEN_DEGENERATE_RETRY_LOG")
                if log_path:
                    try:
                        with open(log_path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"at": time.time(), "model": self.model, "sampling": sampling,
                                                 "first_error": str(first_exc), "still_degenerate": still_degenerate,
                                                 "pid": os.getpid()}) + "\n")
                    except OSError:
                        pass

    def _complete_with_metadata_once(
        self, prompt: str, *, _sampling: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> LLMCompletion:
        effective_max_tokens = self._effective_max_tokens()
        request_logprobs = bool(kwargs.get("logprobs", False))
        request_top_logprobs = kwargs.get("top_logprobs")
        if self.provider == "anthropic":
            payload: dict[str, Any] = {
                "model": self.model,
                "max_tokens": effective_max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
                "Accept-Encoding": "identity",
                "Connection": "close",
            }
        else:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": effective_max_tokens,
            }
            # Explicit opt-in for locally served models; leave other providers unchanged.
            thinking = os.getenv("PARGEN_QWEN_ENABLE_THINKING")
            if thinking is not None and self.provider == "openai" and self.model.lower().startswith("qwen/"):
                if thinking not in {"0", "1"}:
                    raise ValueError("PARGEN_QWEN_ENABLE_THINKING must be 0 or 1")
                payload["chat_template_kwargs"] = {"enable_thinking": thinking == "1"}
            if request_logprobs:
                payload["logprobs"] = True
                if request_top_logprobs is not None:
                    payload["top_logprobs"] = int(request_top_logprobs)
            # OpenRouter provider routing (PARGEN_OPENROUTER_PROFILE), as in Stage B; no-op elsewhere.
            from pargen.openrouter_routing import routed_payload

            payload = routed_payload(payload, self.base_url)
            if _sampling:
                payload = {**payload, **_sampling}
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept-Encoding": "identity",
                "Connection": "close",
            }

        started = time.perf_counter()
        try:
            request_timeout = float(os.getenv("PARGEN_LLM_HTTP_TIMEOUT_S", "600"))
        except Exception:
            request_timeout = 600.0
        request_timeout = max(1.0, request_timeout)
        try:
            request_retries = int(os.getenv("PARGEN_LLM_HTTP_RETRIES", "5"))
        except Exception:
            request_retries = 5
        request_retries = max(1, request_retries)
        # The read timeout above is per chunk; an endpoint can keep a slow request alive with
        # whitespace, so a runaway request can run for a long time. Cap each attempt's total
        # time and drop the connection.
        wall_timeout = _env_float("PARGEN_LLM_WALL_TIMEOUT_S", 1800.0)

        last_exc: Exception | None = None
        body: str = ""
        wall_timeouts = 0
        # Prefer httpx client; fall back to requests session with retries.
        if httpx is None:
            last_exc = RuntimeError("httpx is not available")
        else:
            for attempt in range(request_retries):
                attempt_started = time.perf_counter()
                try:
                    with httpx.Client(http2=False, timeout=request_timeout) as client:
                        with client.stream("POST", self.base_url, json=payload, headers=headers) as resp:
                            if resp.status_code >= 400:
                                resp.read()
                                resp.raise_for_status()
                            chunks = []
                            for chunk in resp.iter_bytes():
                                chunks.append(chunk)
                                if wall_timeout > 0 and time.perf_counter() - attempt_started > wall_timeout:
                                    raise _WallClockTimeout(
                                        f"LLM request exceeded wall-clock timeout {wall_timeout:g}s"
                                    )
                            body = b"".join(chunks).decode("utf-8")
                    break
                except _WallClockTimeout as exc:
                    last_exc = exc
                    wall_timeouts += 1
                    logger.warning("%s (attempt %d/%d)", exc, attempt + 1, request_retries)
                    if attempt < request_retries - 1:
                        continue
                except httpx.HTTPStatusError as exc:
                    last_exc = RuntimeError(
                        f"HTTP {exc.response.status_code} from {self.base_url}: {exc.response.text[:1000]}"
                    )
                    if attempt < request_retries - 1:
                        time.sleep(1)
                        continue
                except httpx.HTTPError as exc:
                    last_exc = exc
                    if attempt < request_retries - 1:
                        time.sleep(1)
                        continue
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < request_retries - 1:
                        time.sleep(1)
                        continue

        if not body and not wall_timeouts:
            # Fallback to requests with retry if httpx did not succeed (a transport problem;
            # a wall-clock timeout is the provider, and requests has no total deadline).
            try:
                import requests
                from requests.adapters import HTTPAdapter
                from requests.exceptions import ChunkedEncodingError, RequestException
                from urllib3.exceptions import ProtocolError
                from urllib3.util import Retry
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"LLM HTTP client imports failed after httpx failure: {last_exc}") from exc

            session = requests.Session()
            retry_cfg = Retry(
                total=request_retries,
                connect=request_retries,
                read=request_retries,
                backoff_factor=1,
                allowed_methods=frozenset({"POST"}),
                status_forcelist=(502, 503, 504),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry_cfg)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            for attempt in range(request_retries):
                try:
                    resp = session.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        timeout=request_timeout,
                    )
                    resp.raise_for_status()
                    body = resp.text
                    break
                except requests.HTTPError as exc:
                    response_text = ""
                    if exc.response is not None:
                        response_text = exc.response.text[:1000]
                    last_exc = RuntimeError(
                        f"HTTP {exc.response.status_code if exc.response is not None else 'unknown'} "
                        f"from {self.base_url}: {response_text}"
                    )
                    if attempt < request_retries - 1:
                        time.sleep(1)
                        continue
                except (ChunkedEncodingError, RequestException, ProtocolError) as exc:
                    last_exc = exc
                    if attempt < request_retries - 1:
                        time.sleep(1)
                        continue
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < request_retries - 1:
                        time.sleep(1)
                        continue

        if not body and last_exc:
            raise RuntimeError(f"LLM request failed ({type(last_exc).__name__}): {last_exc}") from last_exc

        parsed = json.loads(body)
        elapsed_s = max(0.0, time.perf_counter() - started)

        def finish(content_text: str, token_logprobs: Optional[list[dict[str, Any]]] = None) -> LLMCompletion:
            self._record_usage(
                prompt=prompt,
                content=content_text,
                request_payload=payload,
                response_payload=parsed,
                elapsed_s=elapsed_s,
            )
            if not content_text.strip():
                # A null message content used to reach the caller as the text "None".
                usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
                raise EmptyCompletionError(
                    f"LLM response from {self.model} has no text content "
                    f"(finish_reason={_finish_reason(parsed)!r}, "
                    f"completion_tokens={usage.get('completion_tokens')}, max_tokens={payload.get('max_tokens')})"
                )
            return LLMCompletion(content=content_text, raw=parsed, token_logprobs=token_logprobs)

        if self.provider == "anthropic":
            content = parsed.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    elif "text" in item:
                        parts.append(str(item.get("text") or ""))
                return finish("".join(parts), None)
            return finish(str(content or ""), None)

        choices = parsed.get("choices") or []
        if choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                content_text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            else:
                content_text = str(content or "")
            token_logprobs = None
            logprobs = first_choice.get("logprobs")
            if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
                token_logprobs = [dict(item) for item in logprobs["content"] if isinstance(item, dict)]
            return finish(content_text, token_logprobs)

        content = parsed.get("content")
        if isinstance(content, list):
            return finish("".join(item.get("text", "") for item in content if isinstance(item, dict)), None)
        return finish(str(content or ""), None)

    def _record_usage(
        self,
        *,
        prompt: str,
        content: str,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any],
        elapsed_s: float,
    ) -> None:
        usage = response_payload.get("usage") if isinstance(response_payload, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else {}
        prompt_tokens = _safe_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        completion_tokens = _safe_int(usage.get("completion_tokens") or usage.get("output_tokens"))
        prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), Mapping) else {}
        prompt_cache_hit_tokens = _safe_int(
            usage.get("prompt_cache_hit_tokens")
            or prompt_details.get("cached_tokens")
            or usage.get("cache_read_input_tokens")
        )
        cache_creation_input_tokens = _safe_int(usage.get("cache_creation_input_tokens"))
        cache_read_input_tokens = _safe_int(usage.get("cache_read_input_tokens"))
        token_source = "api_usage"
        if prompt_tokens is None:
            prompt_tokens = _estimate_tokens(prompt)
            token_source = "char_estimate"
        if completion_tokens is None:
            completion_tokens = _estimate_tokens(content)
            token_source = "char_estimate"
        if prompt_cache_hit_tokens is None:
            prompt_cache_hit_tokens = 0
        prompt_cache_miss_tokens = _safe_int(usage.get("prompt_cache_miss_tokens"))
        if prompt_cache_miss_tokens is None:
            prompt_cache_miss_tokens = max(0, int(prompt_tokens or 0) - int(prompt_cache_hit_tokens or 0))
        total_tokens = _safe_int(usage.get("total_tokens"))
        if total_tokens is None:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

        cost_usd, cost_source = _estimate_usage_cost_usd(
            provider=self.provider,
            model=self.model,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            prompt_cache_hit_tokens=int(prompt_cache_hit_tokens or 0),
            prompt_cache_miss_tokens=int(prompt_cache_miss_tokens or 0),
            cache_creation_input_tokens=int(cache_creation_input_tokens or 0),
            cache_read_input_tokens=int(cache_read_input_tokens or 0),
        )
        self.usage_records.append(
            {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": request_payload.get("max_tokens"),
                "finish_reason": _finish_reason(response_payload),
                "prompt_chars": len(prompt or ""),
                "completion_chars": len(content or ""),
                "prompt_tokens": int(prompt_tokens or 0),
                "prompt_cache_hit_tokens": int(prompt_cache_hit_tokens or 0),
                "prompt_cache_miss_tokens": int(prompt_cache_miss_tokens or 0),
                "cache_creation_input_tokens": int(cache_creation_input_tokens or 0),
                "cache_read_input_tokens": int(cache_read_input_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or 0),
                "token_source": token_source,
                "elapsed_s": float(elapsed_s),
                "estimated_cost_usd": cost_usd,
                "cost_source": cost_source,
            }
        )

    def usage_summary(self) -> dict[str, Any]:
        return _usage_summary_from_records(getattr(self, "usage_records", []) or [])


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _estimate_tokens(text: str) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    return max(1, int((len(raw) + 3) // 4))


def _usage_summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    records = list(records or [])
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in records)
    prompt_cache_hit_tokens = sum(int(r.get("prompt_cache_hit_tokens") or 0) for r in records)
    prompt_cache_miss_tokens = sum(int(r.get("prompt_cache_miss_tokens") or 0) for r in records)
    cache_creation_input_tokens = sum(int(r.get("cache_creation_input_tokens") or 0) for r in records)
    cache_read_input_tokens = sum(int(r.get("cache_read_input_tokens") or 0) for r in records)
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in records)
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in records)
    known_costs = [r.get("estimated_cost_usd") for r in records if r.get("estimated_cost_usd") is not None]
    return {
        "calls": len(records),
        "prompt_tokens": int(prompt_tokens),
        "prompt_cache_hit_tokens": int(prompt_cache_hit_tokens),
        "prompt_cache_miss_tokens": int(prompt_cache_miss_tokens),
        "cache_creation_input_tokens": int(cache_creation_input_tokens),
        "cache_read_input_tokens": int(cache_read_input_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "elapsed_s": sum(float(r.get("elapsed_s") or 0.0) for r in records),
        "estimated_cost_usd": (sum(float(v) for v in known_costs) if known_costs else None),
        "token_sources": sorted({str(r.get("token_source") or "") for r in records if r.get("token_source")}),
        "cost_sources": sorted({str(r.get("cost_source") or "") for r in records if r.get("cost_source")}),
        "records": records,
    }


def _finish_reason(response_payload: Mapping[str, Any]) -> Optional[str]:
    try:
        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            reason = choices[0].get("finish_reason") or choices[0].get("stop_reason")
            return str(reason) if reason is not None else None
        reason = response_payload.get("stop_reason") or response_payload.get("finish_reason")
        return str(reason) if reason is not None else None
    except Exception:
        return None


def _estimate_usage_cost_usd(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cache_hit_tokens: int = 0,
    prompt_cache_miss_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> tuple[Optional[float], Optional[str]]:
    raw_cache_hit = os.getenv("PARGEN_LLM_CACHE_HIT_USD_PER_MTOK")
    raw_cache_miss = os.getenv("PARGEN_LLM_CACHE_MISS_USD_PER_MTOK")
    raw_input = os.getenv("PARGEN_LLM_INPUT_USD_PER_MTOK")
    raw_output = os.getenv("PARGEN_LLM_OUTPUT_USD_PER_MTOK")
    if raw_cache_hit and raw_cache_miss and raw_output:
        try:
            cache_hit_rate = float(raw_cache_hit)
            cache_miss_rate = float(raw_cache_miss)
            output_rate = float(raw_output)
        except Exception:
            logger.warning("Ignoring invalid PARGEN_LLM cache-aware USD_PER_MTOK settings")
        else:
            miss_tokens = int(prompt_cache_miss_tokens or max(0, prompt_tokens - prompt_cache_hit_tokens))
            cost = (float(prompt_cache_hit_tokens or 0) / 1_000_000.0) * cache_hit_rate
            cost += (float(miss_tokens) / 1_000_000.0) * cache_miss_rate
            cost += (float(completion_tokens) / 1_000_000.0) * output_rate
            return cost, "PARGEN_LLM_CACHE_HIT/MISS/OUTPUT_USD_PER_MTOK"

    if not raw_input or not raw_output:
        default = _default_llm_price(provider=provider, model=model)
        if default is None:
            return None, None
        source, rates = default
        miss_tokens = int(prompt_cache_miss_tokens or max(0, prompt_tokens - prompt_cache_hit_tokens))
        hit_tokens = int(prompt_cache_hit_tokens or 0)
        cost = (float(hit_tokens) / 1_000_000.0) * rates["cache_hit"]
        cost += (float(miss_tokens) / 1_000_000.0) * rates["cache_miss"]
        # Anthropic-style cache creation/read tokens are included only when providers add defaults.
        cost += (float(cache_creation_input_tokens or 0) / 1_000_000.0) * rates.get("cache_creation", rates["cache_miss"])
        cost += (float(cache_read_input_tokens or 0) / 1_000_000.0) * rates.get("cache_read", rates["cache_hit"])
        cost += (float(completion_tokens) / 1_000_000.0) * rates["output"]
        return cost, source
    try:
        input_rate = float(raw_input)
        output_rate = float(raw_output)
    except Exception:
        logger.warning("Ignoring invalid PARGEN_LLM_*_USD_PER_MTOK settings")
        return None, None
    cost = (float(prompt_tokens) / 1_000_000.0) * input_rate
    cost += (float(completion_tokens) / 1_000_000.0) * output_rate
    return cost, "PARGEN_LLM_INPUT_USD_PER_MTOK/PARGEN_LLM_OUTPUT_USD_PER_MTOK"


def _default_llm_price(provider: str, model: str) -> Optional[tuple[str, dict[str, float]]]:
    normalized_provider = _normalize_provider(provider)
    model_key = (model or "").strip().lower()
    # DeepSeek prices are the only defaults we rely on for current ParGen Stage C
    # experiments. Keep other providers opt-in via PARGEN_LLM_* env vars.
    if normalized_provider != "deepseek":
        return None
    if "pro" in model_key:
        return (
            "default_deepseek_v4_pro_usd_per_mtok_2026_05",
            {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
        )
    if "flash" in model_key or "chat" in model_key or not model_key:
        return (
            "default_deepseek_v4_flash_usd_per_mtok_2026_05",
            {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
        )
    return None


def _load_model_config_env(config_path: Optional[Path] = None) -> None:
    """
    Lightweight parser for parser_realization/model_config.yaml to mirror Stage B key loading without PyYAML.
    Extracts provider api_key/base_url/model and injects into env if not already set.
    """
    path = config_path or Path(__file__).with_name("model_config.yaml")
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return

    current = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith("api:"):
            key = line.rstrip(":").strip()
            if key in ("openai", "anthropic", "deepseek"):
                current = key
            else:
                current = None
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
        if current and "api_key" in line and ":" in line:
            _, val = line.split(":", 1)
            val = val.strip().strip('"').strip("'")
            if not val:
                continue
            env_name = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
            }.get(current)
            if env_name and not os.getenv(env_name):
                os.environ[env_name] = val
        if current and "base_url" in line and ":" in line:
            _, val = line.split(":", 1)
            val = val.strip().strip('"').strip("'")
            if not val:
                continue
            env_name = {
                "openai": "OPENAI_BASE_URL",
                "anthropic": "ANTHROPIC_BASE_URL",
                "deepseek": "DEEPSEEK_BASE_URL",
            }.get(current)
            if env_name and not os.getenv(env_name):
                os.environ[env_name] = val
        if current and "model" in line and ":" in line and "api_key" not in line:
            _, val = line.split(":", 1)
            val = val.strip().strip('"').strip("'")
            if not val:
                continue
            env_name = {
                "openai": "OPENAI_MODEL",
                "anthropic": "ANTHROPIC_MODEL",
                "deepseek": "DEEPSEEK_MODEL",
            }.get(current)
            if env_name and not os.getenv(env_name):
                os.environ[env_name] = val
