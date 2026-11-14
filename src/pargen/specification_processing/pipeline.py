"""Reusable interface for PDF section extraction and cache inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .step1_pdf_extract import AutomatedPDFExtractor, TOCEntry


@dataclass
class DocumentPipelineConfig:
    """Configuration for the document processing stage."""

    claude_api_key: str
    deepseek_api_key: Optional[str] = None
    section_strategy: str = "leaf_only"
    target_protocol: Optional[str] = None
    target_protocol_aliases: tuple[str, ...] = ()


class DocumentPipeline:
    """Wrapper that makes the step1 document extraction easier to consume."""

    def __init__(self, config: DocumentPipelineConfig):
        self.config = config
        self._extractor = AutomatedPDFExtractor(
            claude_api_key=config.claude_api_key,
            deepseek_api_key=config.deepseek_api_key,
        )

    # --- Convenience helpers -------------------------------------------------
    def extract_toc(self, pdf_path: Path) -> List[TOCEntry]:
        """Return the table-of-contents entries detected in ``pdf_path``."""

        return self._extractor.extract_toc_with_regex(str(pdf_path))

    def select_sections(
        self,
        toc_entries: List[TOCEntry],
        strategy: Optional[str] = None,
        target_protocol: Optional[str] = None,
    ) -> List[TOCEntry]:
        """Filter TOC entries based on the SectionProcessor heuristics."""

        strategy = strategy or self.config.section_strategy
        target_protocol = target_protocol or self.config.target_protocol
        return self._extractor.section_processor.filter_sections_for_processing(
            toc_entries,
            strategy=strategy,
            target_protocol=target_protocol,
            target_protocol_aliases=list(self.config.target_protocol_aliases),
        )

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Run the full automated pipeline with caching and JSON repair."""

        return self._extractor.process_pdf_automated_enhanced(
            str(pdf_path),
            section_strategy=self.config.section_strategy,
            target_protocol=self.config.target_protocol,
            target_protocol_aliases=list(self.config.target_protocol_aliases),
        )

    # --- Introspection -------------------------------------------------------
    def cached_status(self, pdf_path: Path) -> Dict[str, Any]:
        """Expose cache inspection helpers from the underlying extractor."""

        pdf_name = pdf_path.stem
        return self._extractor.check_pdf_cache_status(pdf_name)

    def summarize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a light-weight summary from a processing result."""

        return {
            "from_cache": result.get("from_cache", False),
            "llm_sections": result.get("llm_processed_sections", 0),
            "raw_sections": result.get("raw_sections", 0),
            "cost_summary": result.get("cost_summary", {}),
        }
