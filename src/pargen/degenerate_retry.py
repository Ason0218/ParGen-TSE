"""Sampling for the retry of a request whose greedy answer was degenerate.

At T=0 a thinking model can loop inside its reasoning and then stop, or reach the output cap,
with no answer, and every retry of the same request returns the same reply. With
STEP2_DEGENERATE_RETRY_SAMPLING set to a JSON object (for example
{"temperature": 0.6, "top_p": 0.95, "top_k": 20}), a request whose reply was degenerate - empty content without tool calls, or
finish_reason 'length' - is sent with those parameters when the same request is sent again.
The first request of every prompt keeps the profile's decoding (T=0), so non-degenerate calls
are unchanged. Unset (every other profile): nothing changes.

Each sampled retry is appended to <run's step2 cache dir>/degenerate_retries.jsonl (one file
per run), which the round receipt counts.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

ENV = 'STEP2_DEGENERATE_RETRY_SAMPLING'
_seen = set()
_lock = threading.Lock()


def settings():
    raw = (os.getenv(ENV) or '').strip()
    return json.loads(raw) if raw else None


def _key(payload):
    core = {k: payload.get(k) for k in ('model', 'messages', 'system', 'tools', 'tool_choice', 'response_format', 'max_tokens')}
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def degenerate(response_data):
    choices = (response_data or {}).get('choices') or []
    if not choices:
        return False
    choice = choices[0]
    message = choice.get('message') or {}
    if choice.get('finish_reason') == 'length':
        return True
    return not (message.get('content') or '').strip() and not (message.get('tool_calls') or [])


def apply(payload):
    """(payload to send, sampling applied or None)."""
    sampling = settings()
    if not sampling:
        return payload, None
    with _lock:
        seen = _key(payload) in _seen
    return ({**payload, **sampling}, sampling) if seen else (payload, None)


def note(payload, response_data, sampled, log_dir=None):
    """Remember a degenerate reply to `payload` (the request as built, before apply())."""
    if not settings():
        return
    key = _key(payload)
    if degenerate(response_data):
        with _lock:
            _seen.add(key)
    if sampled is not None and log_dir is not None:
        choice = ((response_data or {}).get('choices') or [{}])[0]
        try:
            path = Path(log_dir) / 'degenerate_retries.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps({'at': time.time(), 'request_sha256': key, 'model': payload.get('model'),
                                     'sampling': sampled, 'finish_reason': choice.get('finish_reason'),
                                     'still_degenerate': degenerate(response_data),
                                     'usage': (response_data or {}).get('usage')}, ensure_ascii=False) + '\n')
        except OSError:
            pass
