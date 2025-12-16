"""Service-layer orchestration for PDF ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional

from backend.config.settings import deepdoc_settings, vlm_parser_settings
from backend.llm_infrastructure.preprocessing.parsers import ParsedDocument, PdfParseOptions
from backend.llm_infrastructure.preprocessing.parsers.registry import get_parser
from .normalizer import get_normalizer


@dataclass
class Section:
    title: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "metadata": self.metadata or {},
        }


class DocumentIngestService:
    """Parses PDFs then applies domain-level splitting/normalization."""

    def __init__(
        self,
        parser_id: str = "pdf_plain",
        parser_options: Optional[PdfParseOptions] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.parser_id = parser_id
        self.parser_options = parser_options or self._default_options(parser_id)
        self.parser = get_parser(parser_id, **(parser_kwargs or {}))

    @classmethod
    def for_deepdoc(cls) -> "DocumentIngestService":
        """Construct a service preconfigured with the DeepDoc parser."""
        return cls(parser_id="pdf_deepdoc")

    @classmethod
    def for_vlm(
        cls,
        *,
        vlm_client: Any | None = None,
        vlm_factory: Any | None = None,
        renderer: Any | None = None,
        options: Optional[PdfParseOptions] = None,
    ) -> "DocumentIngestService":
        """Construct a service preconfigured with the VLM (e.g., DeepSeek-VL) parser."""
        parser_kwargs: Dict[str, Any] = {}
        if vlm_client is not None:
            parser_kwargs["vlm_client"] = vlm_client
        if vlm_factory is not None:
            parser_kwargs["vlm_factory"] = vlm_factory
        if renderer is not None:
            parser_kwargs["renderer"] = renderer
        return cls(
            parser_id="pdf_vlm",
            parser_options=options or cls._default_vlm_options(),
            parser_kwargs=parser_kwargs,
        )

    def ingest_pdf(self, file: BinaryIO, doc_type: str = "generic") -> Dict[str, Any]:
        parsed = self.parser.parse(file, options=self.parser_options)
        sections = self._split_sections(parsed, doc_type)
        normalizer = get_normalizer(level="L3")
        for section in sections:
            section.text = normalizer(section.text)
        return {
            "parsed": parsed,
            "sections": [section.to_dict() for section in sections],
            "metadata": {"parser_id": self.parser_id, "doc_type": doc_type},
        }

    def _split_sections(self, parsed: ParsedDocument, doc_type: str) -> List[Section]:
        # 현재는 섹션 분리를 하지 않고 VLM 결과 전체를 하나의 섹션으로 반환
        if not parsed.blocks:
            return []

        merged_text = "\n".join(
            self._normalize_text(block.text) for block in parsed.blocks if block.text
        ).strip()

        return [
            Section(
                title="document",
                text=merged_text,
                page_start=parsed.blocks[0].page,
                page_end=parsed.blocks[-1].page,
                metadata={"source": "raw"},
            )
        ]

    @staticmethod
    def _normalize_heading(value: str) -> str:
        text = (value or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalize_text(value: str) -> str:
        text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _section_pattern(self, doc_type: str) -> re.Pattern[str]:
        token = (doc_type or "").lower()
        patterns: Dict[str, re.Pattern[str]] = {
            "sop": re.compile(r"^\d+\.\s*"),
            "ts": re.compile(r"^[A-D]-\d+\.\s*"),
            "guide": re.compile(r"^\d+\.\d*\s+"),
            # raw/none: 매칭되지 않는 패턴으로 섹션 분리를 비활성화
            "raw": re.compile(r"(?! )"),
            "none": re.compile(r"(?! )"),
        }
        return patterns.get(token, re.compile(r"^(\d+\.|\w+\s*[:\-])"))

    def _default_options(self, parser_id: str) -> PdfParseOptions:
        if parser_id.startswith("pdf_deepseek"):
            return PdfParseOptions(
                vlm_model=vlm_parser_settings.model_id,
                vlm_prompt=vlm_parser_settings.prompt,
                vlm_max_new_tokens=vlm_parser_settings.max_new_tokens,
                vlm_temperature=vlm_parser_settings.temperature,
                hf_endpoint=vlm_parser_settings.hf_endpoint or None,
                allow_download=vlm_parser_settings.allow_download,
                model_root=vlm_parser_settings.model_root,
                device=vlm_parser_settings.device,
            )
        return PdfParseOptions(
            model_root=deepdoc_settings.model_root,
            hf_endpoint=deepdoc_settings.hf_endpoint or None,
            ocr_model=deepdoc_settings.ocr_model,
            layout_model=deepdoc_settings.layout_model,
            tsr_model=deepdoc_settings.tsr_model,
            allow_download=deepdoc_settings.allow_download,
            device=deepdoc_settings.device,
        )

    @staticmethod
    def _default_vlm_options() -> PdfParseOptions:
        return PdfParseOptions(
            vlm_model=vlm_parser_settings.model_id,
            vlm_prompt=vlm_parser_settings.prompt,
            vlm_max_new_tokens=vlm_parser_settings.max_new_tokens,
            vlm_temperature=vlm_parser_settings.temperature,
            hf_endpoint=vlm_parser_settings.hf_endpoint or None,
            allow_download=vlm_parser_settings.allow_download,
            model_root=vlm_parser_settings.model_root,
            device=vlm_parser_settings.device,
        )
