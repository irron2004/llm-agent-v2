"""Metadata extraction for document ingestion.

Extracts document-level and chunk-level metadata using:
1. Rule-based extraction (regex patterns) - primary
2. LLM-based extraction - fallback when rules fail

Usage:
    extractor = MetadataExtractor(llm=llm_instance)
    doc_meta = extractor.extract_document_metadata(first_pages_text)
    chapters = extractor.assign_chapters(sections, doc_type="sop")
    summary = extractor.summarize_chunk(chunk_text)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from backend.llm_infrastructure.llm.base import BaseLLM

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol for LLM interface (allows mocking/testing)
# =============================================================================
class LLMProtocol(Protocol):
    """Minimal LLM interface for metadata extraction."""

    def generate(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Any:
        """Generate response from messages."""
        ...


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class DocumentMetadata:
    """Document-level metadata extracted from first pages."""

    device_name: str = ""
    doc_description: str = ""
    confidence: float = 0.0
    source: str = "unknown"  # "rule" | "llm" | "unknown"


@dataclass
class ChapterInfo:
    """Chapter information for a section/page."""

    title: str = ""
    level: int = 1  # 1=main chapter, 2=sub-chapter, etc.
    source: str = "unknown"  # "rule" | "llm" | "carry_forward"


# =============================================================================
# Rule-based Patterns (doc_type별 heading 패턴)
# =============================================================================
HEADING_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    # SOP: "1. Title", "1.1 Title", "1.1.1 Title"
    "sop": [
        re.compile(r"^(\d+\.(?:\d+\.)*)\s*(.+)$", re.MULTILINE),
    ],
    # TS (Troubleshooting): "A-1. Title", "B-2.1 Title"
    "ts": [
        re.compile(r"^([A-Z]-\d+(?:\.\d+)*\.?)\s*(.+)$", re.MULTILINE),
    ],
    # Guide: "Chapter 1:", "Section 1.1", "1. Title"
    "guide": [
        re.compile(r"^(?:Chapter|Section|장|절)\s*(\d+(?:\.\d+)*)[:\s]+(.+)$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^(\d+\.(?:\d+\.)*)\s*(.+)$", re.MULTILINE),
    ],
    # Generic: Markdown headings, numbered sections
    "generic": [
        re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),  # Markdown headings
        re.compile(r"^(\d+\.(?:\d+\.)*)\s*(.+)$", re.MULTILINE),  # Numbered
    ],
}

# 장비명 추출 패턴 (rule-based)
DEVICE_NAME_PATTERNS: list[re.Pattern[str]] = [
    # "Model: SUPRA XP", "모델명: ABC-123"
    re.compile(r"(?:Model|모델|모델명|장비명|Equipment)\s*[:：]\s*([A-Z0-9][A-Z0-9\-_ ]+)", re.IGNORECASE),
    # "SUPRA XP Manual", "EFEM User Guide"
    re.compile(r"^([A-Z][A-Z0-9\-]+(?:\s+[A-Z0-9\-]+)*)\s+(?:Manual|Guide|매뉴얼|가이드|SOP)", re.MULTILINE | re.IGNORECASE),
    # 대문자로 된 장비명 패턴 (3글자 이상)
    re.compile(r"\b([A-Z]{2,}[A-Z0-9\-]*(?:\s+[A-Z0-9\-]+)*)\b"),
]


# =============================================================================
# LLM Prompts
# =============================================================================
DOC_METADATA_PROMPT = """다음은 기술 문서의 첫 페이지 내용입니다.
아래 정보를 JSON 형식으로 추출하세요:

1. device_name: 이 문서가 설명하는 장비/제품명 (예: "SUPRA XP", "EFEM", "RFID Module")
2. doc_description: 문서의 목적/내용을 1~2문장으로 요약

정보가 없으면 빈 문자열("")을 사용하세요.
반드시 JSON만 출력하세요.

문서 내용:
{page_text}

출력 형식:
{{"device_name": "...", "doc_description": "..."}}"""

CHAPTER_EXTRACT_PROMPT = """이 페이지에서 챕터/섹션 제목을 추출하세요.

이전 챕터: {prev_chapter}

페이지 내용:
{page_text}

규칙:
- 새로운 챕터/섹션 제목이 있으면 해당 제목을 반환
- 제목이 없으면 이전 챕터를 그대로 반환
- 제목만 반환 (설명 없이)

제목:"""

SUMMARIZE_PROMPT = """다음 내용을 1~2문장으로 요약하세요.
기술 문서이므로 핵심 정보를 정확히 전달해야 합니다.

내용:
{text}

요약:"""


# =============================================================================
# MetadataExtractor Class
# =============================================================================
class MetadataExtractor:
    """Extract metadata from document text using rules and LLM fallback.

    Architecture:
    - VLM is for OCR only (text extraction from images)
    - Text LLM is for metadata extraction and summarization
    - Rule-based extraction is tried first, LLM is fallback
    """

    def __init__(
        self,
        llm: LLMProtocol | None = None,
        *,
        use_llm_fallback: bool = True,
        summarizer: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize MetadataExtractor.

        Args:
            llm: Text LLM for fallback extraction (optional).
            use_llm_fallback: Whether to use LLM when rules fail.
            summarizer: Custom summarizer function (optional, uses LLM if not provided).
        """
        self.llm = llm
        self.use_llm_fallback = use_llm_fallback and (llm is not None)
        self._summarizer = summarizer

    # -------------------------------------------------------------------------
    # Document-level Metadata
    # -------------------------------------------------------------------------
    def extract_document_metadata(
        self,
        pages_text: list[str],
        *,
        max_pages: int = 3,
    ) -> DocumentMetadata:
        """Extract device_name and doc_description from first pages.

        Strategy:
        1. Rule-based: regex patterns on first N pages
        2. Frequency voting: if multiple candidates, pick most frequent
        3. LLM fallback: if rules fail and LLM available

        Args:
            pages_text: List of page texts (first few pages).
            max_pages: Maximum pages to analyze.

        Returns:
            DocumentMetadata with extracted info.
        """
        # Combine first N pages
        combined_text = "\n\n".join(pages_text[:max_pages])

        # 1. Rule-based extraction
        device_name = self._extract_device_name_rule(combined_text)
        doc_description = self._extract_doc_description_rule(combined_text)

        if device_name or doc_description:
            logger.debug(
                "Metadata extracted via rule: device=%s, desc_len=%d",
                device_name,
                len(doc_description),
            )
            return DocumentMetadata(
                device_name=device_name,
                doc_description=doc_description,
                confidence=0.8 if device_name else 0.5,
                source="rule",
            )

        # 2. LLM fallback (only if no rule-based results)
        if self.use_llm_fallback:
            return self._extract_doc_metadata_llm(combined_text)

        # No extraction possible - return empty with clear source
        logger.debug("No metadata extracted (rules failed, LLM disabled)")
        return DocumentMetadata(source="none")

    def _extract_device_name_rule(self, text: str) -> str:
        """Extract device name using regex patterns."""
        candidates: list[str] = []

        for pattern in DEVICE_NAME_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                name = match.strip() if isinstance(match, str) else match
                # Filter out common words
                if name and len(name) >= 3 and name.upper() not in {
                    "THE", "AND", "FOR", "WITH", "USER", "GUIDE", "MANUAL",
                    "SYSTEM", "MODULE", "UNIT", "SETUP", "PAGE", "TABLE",
                }:
                    candidates.append(name)

        if not candidates:
            return ""

        # Frequency voting (return most common)
        from collections import Counter
        counter = Counter(candidates)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _extract_doc_description_rule(self, text: str) -> str:
        """Extract document description using simple rules.

        Strategy: Find first meaningful sentence(s) that describe the document.
        Look for patterns like "이 문서는...", "This document...", "본 매뉴얼은..." etc.
        If not found, use the first 1-2 sentences after filtering headers/noise.
        """
        # Patterns that indicate document description
        desc_patterns = [
            re.compile(r"(?:이\s*문서|본\s*문서|본\s*매뉴얼|이\s*가이드)[은는이가]\s*(.+?)[.。]", re.IGNORECASE),
            re.compile(r"(?:This\s+document|This\s+manual|This\s+guide)\s+(?:is|describes|provides|covers)\s+(.+?)\.", re.IGNORECASE),
            re.compile(r"(?:목적|Purpose|Overview|개요)[:\s]+(.+?)[.。\n]", re.IGNORECASE),
        ]

        for pattern in desc_patterns:
            match = pattern.search(text[:2000])
            if match:
                desc = match.group(0).strip()
                # Limit length
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                return desc

        # Fallback: extract first meaningful sentence
        # Skip lines that look like headers (short, all caps, numbered)
        lines = text[:2000].split("\n")
        meaningful_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip header-like lines
            if len(line) < 10:
                continue
            if line.isupper() and len(line) < 50:
                continue
            if re.match(r"^[\d.]+\s*$", line):  # Just numbers
                continue
            if re.match(r"^(page|페이지|목차|table of contents)\s*\d*$", line, re.IGNORECASE):
                continue
            meaningful_lines.append(line)
            if len(meaningful_lines) >= 2:
                break

        if meaningful_lines:
            desc = " ".join(meaningful_lines)
            if len(desc) > 200:
                desc = desc[:200] + "..."
            return desc

        return ""

    def _extract_doc_metadata_llm(self, text: str) -> DocumentMetadata:
        """Extract metadata using LLM."""
        if not self.llm:
            return DocumentMetadata(source="unknown")

        try:
            prompt = DOC_METADATA_PROMPT.format(page_text=text[:4000])
            response = self.llm.generate([{"role": "user", "content": prompt}])
            result_text = response.text.strip()

            # Parse JSON
            # Handle markdown code blocks
            if "```" in result_text:
                result_text = re.search(r"```(?:json)?\s*(.*?)```", result_text, re.DOTALL)
                result_text = result_text.group(1) if result_text else "{}"

            data = json.loads(result_text)
            return DocumentMetadata(
                device_name=data.get("device_name", ""),
                doc_description=data.get("doc_description", ""),
                confidence=0.7,
                source="llm",
            )
        except Exception as e:
            logger.warning("LLM metadata extraction failed: %s", e)
            return DocumentMetadata(source="unknown")

    # -------------------------------------------------------------------------
    # Chapter Assignment
    # -------------------------------------------------------------------------
    def assign_chapters(
        self,
        sections: list[dict[str, Any]],
        doc_type: str = "generic",
    ) -> list[dict[str, Any]]:
        """Assign chapter titles to sections with carry-forward.

        Strategy:
        1. Check section title first (if it looks like a chapter heading)
        2. Rule-based: regex patterns on text based on doc_type
        3. Carry-forward: if no new chapter found, use previous

        Args:
            sections: List of section dicts with 'text', 'title', etc.
            doc_type: Document type for pattern selection.

        Returns:
            Sections with 'chapter' added to metadata.
        """
        patterns = HEADING_PATTERNS.get(doc_type, HEADING_PATTERNS["generic"])
        current_chapter = ""

        for section in sections:
            title = section.get("title", "")
            text = section.get("text", "")
            metadata = section.get("metadata", {})

            chapter = ""
            source = "none"

            # Step 1: Check section title first (if not just "page_N")
            if title and not re.match(r"^page_\d+$", title):
                chapter = self._extract_chapter_from_title(title, patterns)
                if chapter:
                    source = "title"

            # Step 2: Try rule-based chapter extraction from text
            if not chapter:
                chapter = self._extract_chapter_rule(text, patterns)
                if chapter:
                    source = "rule"

            # Step 3: Apply result or carry-forward
            if chapter:
                current_chapter = chapter
                metadata["chapter"] = chapter
                metadata["chapter_source"] = source
            elif current_chapter:
                # Carry-forward
                metadata["chapter"] = current_chapter
                metadata["chapter_source"] = "carry_forward"
            else:
                metadata["chapter"] = ""
                metadata["chapter_source"] = "none"

            section["metadata"] = metadata

        return sections

    def _extract_chapter_from_title(
        self,
        title: str,
        patterns: list[re.Pattern[str]],
    ) -> str:
        """Extract chapter from section title if it matches heading pattern."""
        title = title.strip()
        if not title:
            return ""

        # Check if title matches any heading pattern
        for pattern in patterns:
            match = pattern.match(title)
            if match:
                return title

        # Also accept titles that look like chapters (numbered, markdown headings)
        if re.match(r"^(\d+\.)+\s*.+", title):  # "1.2.3 Something"
            return title
        if re.match(r"^#{1,3}\s+.+", title):  # "## Something"
            return re.sub(r"^#+\s*", "", title)

        return ""

    def _extract_chapter_rule(
        self,
        text: str,
        patterns: list[re.Pattern[str]],
    ) -> str:
        """Extract chapter title from text using patterns."""
        # Look at first 500 chars for chapter headers
        header_text = text[:500]

        for pattern in patterns:
            match = pattern.search(header_text)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    # Format: "1.1 Title" -> "1.1 Title"
                    number, title = groups[0], groups[1]
                    return f"{number} {title}".strip()
                elif len(groups) == 1:
                    return groups[0].strip()

        return ""

    # -------------------------------------------------------------------------
    # Chunk Summarization
    # -------------------------------------------------------------------------
    def summarize_chunk(self, text: str, max_length: int = 200) -> str:
        """Summarize a chunk of text.

        Args:
            text: Text to summarize.
            max_length: Maximum summary length.

        Returns:
            Summary string (empty if summarization disabled/failed).
        """
        if not text or len(text) < 100:
            return ""

        # Use custom summarizer if provided
        if self._summarizer:
            try:
                return self._summarizer(text)[:max_length]
            except Exception as e:
                logger.warning("Custom summarizer failed: %s", e)
                return ""

        # Use LLM for summarization
        if not self.llm:
            return ""

        try:
            prompt = SUMMARIZE_PROMPT.format(text=text[:2000])
            response = self.llm.generate([{"role": "user", "content": prompt}])
            return response.text.strip()[:max_length]
        except Exception as e:
            logger.warning("LLM summarization failed: %s", e)
            return ""

    def summarize_chunks(
        self,
        sections: list[dict[str, Any]],
        *,
        min_text_length: int = 100,
    ) -> list[dict[str, Any]]:
        """Add summaries to all sections.

        Args:
            sections: List of section dicts.
            min_text_length: Minimum text length to summarize.

        Returns:
            Sections with 'chunk_summary' added to metadata.
        """
        for section in sections:
            text = section.get("text", "")
            metadata = section.get("metadata", {})

            if len(text) >= min_text_length:
                summary = self.summarize_chunk(text)
                metadata["chunk_summary"] = summary
            else:
                metadata["chunk_summary"] = ""

            section["metadata"] = metadata

        return sections


# =============================================================================
# Factory Function
# =============================================================================
def create_metadata_extractor(
    llm: LLMProtocol | None = None,
    *,
    use_llm_fallback: bool = True,
) -> MetadataExtractor:
    """Create MetadataExtractor with optional LLM.

    If no LLM provided, creates from settings.

    Args:
        llm: Pre-configured LLM instance.
        use_llm_fallback: Enable LLM fallback for extraction.

    Returns:
        Configured MetadataExtractor instance.
    """
    if llm is None and use_llm_fallback:
        try:
            from backend.llm_infrastructure.llm.registry import get_llm
            llm = get_llm("vllm")
        except Exception as e:
            logger.warning("Could not create LLM for metadata extraction: %s", e)
            llm = None

    return MetadataExtractor(llm=llm, use_llm_fallback=use_llm_fallback)


__all__ = [
    "MetadataExtractor",
    "DocumentMetadata",
    "ChapterInfo",
    "create_metadata_extractor",
]
