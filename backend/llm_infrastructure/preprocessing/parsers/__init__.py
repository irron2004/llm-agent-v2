"""Parser implementations for common document formats."""

from .base import (
    BaseParser,
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
    PdfParseOptions,
)
from .adapters import DeepDocPdfAdapter, DeepSeekVLPdfAdapter, PlainPdfAdapter, VlmPdfAdapter
from .registry import get_parser, list_parsers, register_parser

__all__ = [
    "BaseParser",
    "BoundingBox",
    "ParsedBlock",
    "ParsedDocument",
    "ParsedFigure",
    "ParsedPage",
    "ParsedTable",
    "PdfParseOptions",
    "DeepDocPdfAdapter",
    "DeepSeekVLPdfAdapter",
    "PlainPdfAdapter",
    "VlmPdfAdapter",
    "get_parser",
    "list_parsers",
    "register_parser",
]
