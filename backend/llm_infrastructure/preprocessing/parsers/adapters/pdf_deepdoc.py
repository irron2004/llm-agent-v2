"""Adapter that registers the DeepDoc PDF parser."""

from __future__ import annotations

from typing import BinaryIO, Optional

from ..base import BaseParser, ParsedDocument, PdfParseOptions
from ..engines.pdf_deepdoc_engine import DeepDocPdfEngine
from ..registry import register_parser


class DeepDocPdfAdapter(BaseParser):
    content_type: str = "application/pdf"

    def __init__(self, engine: Optional[DeepDocPdfEngine] = None) -> None:
        self.engine = engine or DeepDocPdfEngine()

    def parse(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        return self.engine.run(file, options=options)


register_parser("pdf_deepdoc", DeepDocPdfAdapter)

__all__ = ["DeepDocPdfAdapter"]
