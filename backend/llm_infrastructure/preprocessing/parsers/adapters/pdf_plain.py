"""Adapter that registers a plain-text PDF parser."""

from __future__ import annotations

from typing import BinaryIO, Optional

from ..base import BaseParser, ParsedDocument, PdfParseOptions
from ..registry import register_parser
from ..engines.pdf_plain_engine import PlainPdfEngine


class PlainPdfAdapter(BaseParser):
    content_type: str = "application/pdf"

    def __init__(self, engine: Optional[PlainPdfEngine] = None) -> None:
        self.engine = engine or PlainPdfEngine()

    def parse(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        return self.engine.run(file, options=options)


register_parser("pdf_plain", PlainPdfAdapter)

__all__ = ["PlainPdfAdapter"]
