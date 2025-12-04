"""Parser interfaces and shared dataclasses for document parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence


class DeepDocBackend(str, Enum):
    """Available DeepDoc PDF parser backends.

    RAGFLOW: 전체 기능 (OCR, 레이아웃 분석, 테이블 인식)
    PLAIN: 텍스트 추출만 (빠름, 모델 불필요)
    """

    RAGFLOW = "RAGFlowPdfParser"
    PLAIN = "PlainParser"

    @classmethod
    def choices(cls) -> list[str]:
        """Return all valid backend names for error messages."""
        return [e.value for e in cls]


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_sequence(cls, coords: Sequence[float] | None) -> Optional["BoundingBox"]:
        if coords is None:
            return None
        if len(coords) != 4:
            raise ValueError("Bounding boxes must contain four coordinates")
        x0, y0, x1, y1 = [float(value) for value in coords]
        return cls(x0=x0, y0=y0, x1=x1, y1=y1)


@dataclass
class ParsedPage:
    number: int
    text: str
    width: Optional[float] = None
    height: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedBlock:
    text: str
    page: int
    bbox: Optional[BoundingBox] = None
    label: str = "text"
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedTable:
    page: int
    bbox: Optional[BoundingBox] = None
    html: Optional[str] = None
    text: Optional[str] = None
    image_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedFigure:
    page: int
    bbox: Optional[BoundingBox] = None
    caption: Optional[str] = None
    image_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    pages: List[ParsedPage] = field(default_factory=list)
    blocks: List[ParsedBlock] = field(default_factory=list)
    tables: List[ParsedTable] = field(default_factory=list)
    figures: List[ParsedFigure] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    content_type: str = "application/pdf"

    def merged_text(self, separator: str = "\n\n") -> str:
        return separator.join(block.text for block in self.blocks if block.text)


@dataclass
class PdfParseOptions:
    """Options consumed by PDF parsers."""

    ocr: bool = True
    layout: bool = True
    tables: bool = True
    merge: bool = True
    scrap_filter: bool = True
    model_root: Optional[Path] = None
    device: str = "cpu"
    max_pages: Optional[int] = None
    fallback_to_plain: bool = True
    preserve_layout: bool = False
    # CV models (DeepDoc)
    ocr_model: Optional[str] = None
    layout_model: Optional[str] = None
    tsr_model: Optional[str] = None
    # VLM models (DeepSeek or similar)
    vlm_model: Optional[str] = None
    vlm_prompt: Optional[str] = None
    vlm_max_new_tokens: Optional[int] = None
    vlm_temperature: Optional[float] = None
    # Shared download/cache hints
    allow_download: bool = True
    hf_endpoint: Optional[str] = None
    # DeepDoc backend selection
    preferred_backend: Optional["DeepDocBackend"] = None


class BaseParser(ABC):
    content_type: str = "application/octet-stream"

    @abstractmethod
    def parse(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        """Parse a binary file into a structured ParsedDocument."""


__all__ = [
    "BaseParser",
    "BoundingBox",
    "DeepDocBackend",
    "ParsedBlock",
    "ParsedDocument",
    "ParsedFigure",
    "ParsedPage",
    "ParsedTable",
    "PdfParseOptions",
]
