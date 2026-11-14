"""Command-line entry point for Stage C (Host-IR driven plugin codegen)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config_manager import load_api_keys
from .llm_client import LLMConfig, create_llm_client
from .stage_c_pipeline import StageCConfig, run_stage_c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C: Parse-IR + Host-IR codegen backend.")
    parser.add_argument("--format-tree", required=True, type=Path, help="Path to traffic_fixed_tree JSON")
    parser.add_argument("--host", default="wireshark_c", help="Host target (e.g., wireshark_c)")
    parser.add_argument("--protocol-family", required=True, help="Protocol family name (e.g., modbus)")
    parser.add_argument("--host-docs", type=Path, help="Directory containing host docs/examples")
    parser.add_argument("--output-dir", type=Path, default=Path("c_code_outputs"), help="Directory to place generated artifacts")
    parser.add_argument("--host-ir-cache", type=Path, help="Optional cache file for learned Host-IR (YAML)")
    parser.add_argument("--codegen-profile", type=Path, help="Optional host/codegen profile JSON")
    parser.add_argument("--verify", action="store_true", help="Enable verification hooks (placeholder)")

    parser.add_argument("--model-config", type=Path, help="Optional model config YAML/JSON for API keys")
    parser.add_argument("--provider", type=str, default="deepseek", help="LLM provider (openai/anthropic/deepseek)")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash-0731", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=64000, help="LLM max tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_config:
        try:
            load_api_keys(args.model_config, set_env=True)
        except Exception:
            pass  # Allow missing config in offline mode

    llm_cfg = LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    llm_client = create_llm_client(llm_cfg)
    host_docs_dir = args.host_docs or Path(__file__).resolve().parents[1] / "data" / "host_docs" / args.host
    cfg = StageCConfig(
        format_tree_path=args.format_tree,
        host_name=args.host,
        protocol_family=args.protocol_family,
        host_docs_dir=host_docs_dir,
        output_dir=args.output_dir,
        host_ir_cache_path=args.host_ir_cache,
        codegen_profile_path=args.codegen_profile,
        verify=args.verify,
        llm_config=llm_cfg,
    )

    summary = run_stage_c(cfg, llm_client=llm_client)
    print("Stage C codegen finished. Summary:")
    for key, val in summary.items():
        print(f"  - {key}: {val}")


if __name__ == "__main__":
    main()
