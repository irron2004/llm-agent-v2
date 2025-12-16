"""Service-layer orchestration for PDF ingestion.

Pipeline:
1. VLM parsing: PDF → page texts (OCR)
2. Document metadata extraction: device_name, doc_description from first pages
3. Section splitting: page-by-page sections
4. Chapter assignment: rule-based with carry-forward
5. Summarization (optional): chunk-level summaries
6. Normalization: text cleanup
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, List, Optional

from backend.config.settings import deepdoc_settings, vlm_parser_settings
from backend.llm_infrastructure.preprocessing.parsers import ParsedDocument, PdfParseOptions
from backend.llm_infrastructure.preprocessing.parsers.registry import get_parser
from .normalizer import get_normalizer

if TYPE_CHECKING:
    from .metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


def _truncate(value: str, max_len: int) -> str:
    """Truncate string to max length, adding ellipsis if truncated."""
    if not value:
        return ""
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


@dataclass
class Section:
    title: str
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with a shallow copy of metadata to prevent mutation."""
        return {
            "title": self.title,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "metadata": dict(self.metadata) if self.metadata else {},
        }


class DocumentIngestService:
    """Parses PDFs then applies domain-level splitting/normalization.

    Enhanced pipeline with metadata extraction:
    - Document metadata: device_name, doc_description from first pages
    - Chapter assignment: rule-based with carry-forward
    - Chunk summarization: optional LLM-based summaries
    """

    def __init__(
        self,
        parser_id: str = "pdf_plain",
        parser_options: Optional[PdfParseOptions] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        *,
        metadata_extractor: Optional["MetadataExtractor"] = None,
        enable_doc_metadata: bool = True,
        enable_chapter_extraction: bool = True,
        enable_summarization: bool = False,
    ) -> None:
        """Initialize DocumentIngestService.

        Args:
            parser_id: Parser backend identifier.
            parser_options: Parser configuration options.
            parser_kwargs: Additional parser kwargs.
            metadata_extractor: Optional MetadataExtractor for enhanced extraction.
            enable_doc_metadata: Extract device_name/doc_description from first pages.
            enable_chapter_extraction: Assign chapter titles with carry-forward.
            enable_summarization: Generate chunk-level summaries (requires LLM).
        """
        self.parser_id = parser_id
        self.parser_options = parser_options or self._default_options(parser_id)
        self.parser = get_parser(parser_id, **(parser_kwargs or {}))

        # Metadata extraction settings
        self._metadata_extractor = metadata_extractor
        self.enable_doc_metadata = enable_doc_metadata
        self.enable_chapter_extraction = enable_chapter_extraction
        self.enable_summarization = enable_summarization

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
        metadata_extractor: Optional["MetadataExtractor"] = None,
        enable_doc_metadata: bool = True,
        enable_chapter_extraction: bool = True,
        enable_summarization: bool = False,
    ) -> "DocumentIngestService":
        """Construct a service preconfigured with the VLM (e.g., Qwen-VL) parser.

        Args:
            vlm_client: Pre-configured VLM client.
            vlm_factory: Factory for creating VLM client.
            renderer: PDF page renderer.
            options: Parser options.
            metadata_extractor: Optional MetadataExtractor for enhanced extraction.
            enable_doc_metadata: Extract device_name/doc_description.
            enable_chapter_extraction: Assign chapter titles.
            enable_summarization: Generate chunk summaries.

        Returns:
            Configured DocumentIngestService instance.
        """
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
            metadata_extractor=metadata_extractor,
            enable_doc_metadata=enable_doc_metadata,
            enable_chapter_extraction=enable_chapter_extraction,
            enable_summarization=enable_summarization,
        )

    def ingest_pdf(self, file: BinaryIO, doc_type: str = "generic") -> Dict[str, Any]:
        """Parse PDF and extract sections with metadata.

        Pipeline:
        1. VLM parsing: PDF → page texts
        2. Document metadata extraction (optional, best-effort)
        3. Section splitting: page-by-page
        4. Chapter assignment (optional, best-effort)
        5. Summarization (optional, best-effort)
        6. Text normalization

        Args:
            file: PDF file binary.
            doc_type: Document type for pattern matching.

        Returns:
            Dict with parsed document, sections, and metadata.
        """
        # Step 1: Parse PDF with VLM (critical - failure should propagate)
        parsed = self.parser.parse(file, options=self.parser_options)

        # Step 2: Extract document-level metadata (best-effort)
        doc_metadata: Dict[str, str] = {"device_name": "", "doc_description": ""}
        if self.enable_doc_metadata and self._metadata_extractor:
            try:
                pages_text = self._get_first_pages_text(parsed, max_pages=3)
                if pages_text:
                    extracted = self._metadata_extractor.extract_document_metadata(pages_text)
                    doc_metadata = {
                        "device_name": _truncate(extracted.device_name, 200),
                        "doc_description": _truncate(extracted.doc_description, 1000),
                    }
                    logger.debug(
                        "Extracted doc metadata: device=%s, source=%s",
                        extracted.device_name,
                        extracted.source,
                    )
            except Exception:
                logger.exception("Doc metadata extraction failed; continuing without metadata")

        # Step 3: Split into page-by-page sections
        sections = self._split_sections(parsed, doc_type)

        # Inject document-level metadata into all sections
        for section in sections:
            section.metadata.update(doc_metadata)

        # Step 4: Assign chapters with carry-forward (best-effort, page-key based)
        if self.enable_chapter_extraction and self._metadata_extractor:
            try:
                section_dicts = [s.to_dict() for s in sections]
                result_dicts = self._metadata_extractor.assign_chapters(section_dicts, doc_type)

                # Build page -> chapter mapping (safe against reordering/filtering)
                chapter_by_page: Dict[Optional[int], str] = {}
                for d in result_dicts:
                    page_key = d.get("page_start")
                    chapter = d.get("metadata", {}).get("chapter", "")
                    if page_key is not None:
                        chapter_by_page[page_key] = chapter

                # Apply to sections using page key
                for section in sections:
                    chapter = chapter_by_page.get(section.page_start, "")
                    section.metadata["chapter"] = _truncate(chapter, 500)
            except Exception:
                logger.exception("Chapter assignment failed; continuing without chapters")

        # Step 5: Generate chunk summaries (best-effort, page-key based)
        if self.enable_summarization and self._metadata_extractor:
            try:
                section_dicts = [s.to_dict() for s in sections]
                result_dicts = self._metadata_extractor.summarize_chunks(section_dicts)

                # Build page -> summary mapping
                summary_by_page: Dict[Optional[int], str] = {}
                for d in result_dicts:
                    page_key = d.get("page_start")
                    summary = d.get("metadata", {}).get("chunk_summary", "")
                    if page_key is not None:
                        summary_by_page[page_key] = summary

                # Apply to sections using page key
                for section in sections:
                    summary = summary_by_page.get(section.page_start, "")
                    section.metadata["chunk_summary"] = _truncate(summary, 500)
            except Exception:
                logger.exception("Summarization failed; continuing without summaries")

        # Step 6: Normalize text (for indexing, after chapter extraction)
        normalizer = get_normalizer(level="L3")
        for section in sections:
            section.text = normalizer(section.text)

        return {
            "sections": [section.to_dict() for section in sections],
            "metadata": {
                "parser_id": self.parser_id,
                "doc_type": doc_type,
                **doc_metadata,
            },
        }

    def _split_sections(self, parsed: ParsedDocument, doc_type: str) -> List[Section]:
        """페이지별로 섹션을 분리하여 반환.

        Fallback order:
        1. pages (VLM parser) - page-by-page sections
        2. blocks (DeepDoc parser) - group by page number
        3. merged_text - single section with all text
        """
        # Strategy 1: Use pages if available (VLM parser)
        if parsed.pages:
            sections: List[Section] = []
            for page in parsed.pages:
                text = self._normalize_text(page.text) if page.text else ""
                if not text:
                    continue
                sections.append(
                    Section(
                        title=f"page_{page.number}",
                        text=text,
                        page_start=page.number,
                        page_end=page.number,
                        metadata={"source": "vlm"},
                    )
                )
            if sections:
                return sections

        # Strategy 2: Use blocks grouped by page (DeepDoc/plain parser)
        if parsed.blocks:
            return self._sections_from_blocks(parsed.blocks)

        # Strategy 3: Fallback to merged_text as single section
        merged = parsed.merged_text()
        if merged:
            logger.warning("No pages or blocks found, using merged_text as single section")
            return [
                Section(
                    title="document",
                    text=self._normalize_text(merged),
                    page_start=1,
                    page_end=1,
                    metadata={"source": "fallback"},
                )
            ]

        return []

    def _sections_from_blocks(self, blocks: List[Any]) -> List[Section]:
        """Group blocks by page number into sections."""
        from collections import defaultdict

        page_texts: Dict[int, List[str]] = defaultdict(list)
        for block in blocks:
            page_num = getattr(block, "page", 1)
            text = getattr(block, "text", "")
            if text:
                page_texts[page_num].append(text)

        sections: List[Section] = []
        for page_num in sorted(page_texts.keys()):
            combined_text = "\n".join(page_texts[page_num])
            normalized = self._normalize_text(combined_text)
            if normalized:
                sections.append(
                    Section(
                        title=f"page_{page_num}",
                        text=normalized,
                        page_start=page_num,
                        page_end=page_num,
                        metadata={"source": "blocks"},
                    )
                )

        return sections

    def _get_first_pages_text(self, parsed: ParsedDocument, max_pages: int = 3) -> List[str]:
        """Get text from first N pages for metadata extraction.

        Fallback order: pages → blocks → merged_text
        """
        # Try pages first (VLM parser)
        if parsed.pages:
            return [p.text for p in parsed.pages[:max_pages] if p.text]

        # Try blocks grouped by page
        if parsed.blocks:
            from collections import defaultdict
            page_texts: Dict[int, List[str]] = defaultdict(list)
            for block in parsed.blocks:
                page_num = getattr(block, "page", 1)
                text = getattr(block, "text", "")
                if text:
                    page_texts[page_num].append(text)

            result = []
            for page_num in sorted(page_texts.keys())[:max_pages]:
                combined = "\n".join(page_texts[page_num])
                if combined:
                    result.append(combined)
            return result

        # Fallback to merged_text (split into chunks)
        merged = parsed.merged_text()
        if merged:
            # Return first ~3000 chars as single "page"
            return [merged[:3000]]

        return []

    @staticmethod
    def _normalize_heading(value: str) -> str:
        """Normalize heading text (single line)."""
        text = (value or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Light normalization preserving newlines for structure detection.

        Used during section splitting - preserves line structure for:
        - Chapter/heading detection (patterns like ^##, ^1., etc.)
        - Table/list structure
        - Paragraph boundaries

        Final aggressive normalization happens in Step 6 via get_normalizer().
        """
        if not value:
            return ""
        # Normalize line endings
        text = value.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse horizontal whitespace only (preserve newlines)
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive blank lines (3+ → 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_text_aggressive(value: str) -> str:
        """Aggressive normalization collapsing all whitespace.

        Use only when structure doesn't matter (e.g., final indexing).
        """
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
        """Get default parser options based on parser_id.

        VLM parsers: pdf_vlm, pdf_deepseek*, pdf_qwen*, etc.
        DeepDoc parsers: pdf_deepdoc, pdf_plain, etc.
        """
        # VLM-based parsers (vision-language models)
        vlm_parser_ids = ("pdf_vlm", "pdf_deepseek", "pdf_qwen")
        if parser_id.startswith(vlm_parser_ids):
            return self._default_vlm_options()

        # DeepDoc/plain parsers (OCR + layout)
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
