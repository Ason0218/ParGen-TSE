"""Optional request routing, separate from model and sampling configuration."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse


def routed_payload(payload, base_url):
    profile = os.getenv('PARGEN_OPENROUTER_PROFILE')
    if not profile:
        return payload
    # Other configured providers are unaffected by an OpenRouter-only profile.
    if urlparse(base_url).hostname != 'openrouter.ai':
        return payload
    options = json.loads(Path(profile).read_text()).get('provider', {})
    if not isinstance(options, dict) or set(options) - {'only', 'order', 'allow_fallbacks', 'require_parameters', 'sort', 'max_price'}:
        raise ValueError('Invalid OpenRouter provider routing options')
    if not options:
        return payload
    return {**payload, 'provider': {**payload.get('provider', {}), **options}}
