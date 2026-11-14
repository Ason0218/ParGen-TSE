"""Command-line entry point for the document processing stage (specification_processing)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..config_manager import load_api_keys
from ..paths import DEFAULT_API_CONFIG
from .pipeline import DocumentPipeline, DocumentPipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the document processing pipeline")
    parser.add_argument("pdf", type=Path, help="Path to the source specification PDF")
    parser.add_argument("--output", type=Path, default=Path("doc_results.json"), help="Where to store processed results JSON")
    parser.add_argument("--claude-key", default=None, help="Claude API key (overrides config/env)")
    parser.add_argument("--deepseek-key", default=None, help="DeepSeek API key (overrides config/env)")
    parser.add_argument("--config", type=Path, default=DEFAULT_API_CONFIG, help="Path to API config JSON")
    parser.add_argument("--section-strategy", choices=["leaf_only", "fine_grained"], default="leaf_only", help="Section selection strategy")
    parser.add_argument("--target-protocol", default=None, help="Target protocol name used to condition section selection")
    parser.add_argument(
        "--target-protocol-alias",
        action="append",
        default=[],
        help="Additional target protocol alias used during section selection; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_api_keys(args.config, set_env=True)

    claude_key = args.claude_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    deepseek_key = args.deepseek_key or os.environ.get("DEEPSEEK_API_KEY")

    if not claude_key:
        raise SystemExit("Claude API key is required (provide --claude-key or configure pargen/config/api_config.json)")

    config = DocumentPipelineConfig(
        claude_api_key=claude_key,
        deepseek_api_key=deepseek_key,
        section_strategy=args.section_strategy,
        target_protocol=args.target_protocol,
        target_protocol_aliases=tuple(args.target_protocol_alias),
    )
    pipeline = DocumentPipeline(config)
    result = pipeline.process_pdf(args.pdf)

    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = pipeline.summarize(result)
    print("Document processing completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
