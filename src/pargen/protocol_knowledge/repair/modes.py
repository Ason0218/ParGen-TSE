"""
Mode helpers for repair.

Provides a single place to map high-level modes (syntax vs. semantic/traffic)
to their validator functions and prompt modes, so callers don't hardcode
strings scattered across the codebase.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

ValidatorFn = Callable[[dict], "ValidationReport"]  # quoted to avoid import issues


def get_mode(mode: str) -> Tuple[ValidatorFn, str]:
    """
    Return (validator_fn, prompt_mode) for the given mode keyword.

    Supported modes:
      - "syntax": static/spec-based validation (default)
      - "semantic": traffic-based hybrid validation
    """
    normalized = (mode or "syntax").strip().lower()
    if normalized == "semantic":
        from ..message_validation.semantic_validator import run_hybrid_validation as semantic_validator

        return semantic_validator, "traffic_fix"

    # Fallback: syntax/static validation
    from .refinement import run_full_validation as syntax_validator

    return syntax_validator, "fix"


# Export a simple mapping if callers prefer dictionary access
MODE_VALIDATORS: Dict[str, ValidatorFn] = {
    "syntax": lambda tree: get_mode("syntax")[0](tree),
    "semantic": lambda tree: get_mode("semantic")[0](tree),
}

MODE_PROMPTS: Dict[str, str] = {
    "syntax": "fix",
    "semantic": "traffic_fix",
}
