"""Adapter that registers the VLM-based PDF parser (e.g., DeepSeek-VL)."""

from __future__ import annotations

from typing import Any, BinaryIO, Optional

from ..base import BaseParser, ParsedDocument, PdfParseOptions
from ..engines.pdf_vlm_engine import VlmPdfEngine
from ..registry import register_parser


class VlmPdfAdapter(BaseParser):
    content_type: str = "application/pdf"

    def __init__(self, engine: Optional[VlmPdfEngine] = None, **engine_kwargs: Any) -> None:
        self.engine = engine or VlmPdfEngine(**engine_kwargs)

    def parse(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        return self.engine.run(file, options=options)


# Register primary and backward-compatible ids
register_parser("pdf_vlm", VlmPdfAdapter)
register_parser("pdf_deepseek_vl", VlmPdfAdapter)

# Backward-compatible alias
DeepSeekVLPdfAdapter = VlmPdfAdapter

__all__ = ["VlmPdfAdapter", "DeepSeekVLPdfAdapter"]
