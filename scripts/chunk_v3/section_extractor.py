"""VLM 파싱된 페이지에서 section/chapter 경계를 추출하는 모듈.

doc_type별 전략:
- SOP/SETUP: TOC 파싱 → ## N. Title 헤더 매칭 → carry-forward
- TS: alpha TOC (A/B/C) 파싱 → X-N. 서브섹션 패턴 매칭
- PEMS: no-op (섹션 없음)

Public API:
    extract_sections(pages, doc_type) -> list[SectionInfo]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class SectionInfo:
    """페이지별 섹션 정보."""

    section_chapter: str  # e.g. "10. Work Procedure"
    section_number: int  # e.g. 10, -1 if unknown
    chapter_source: str  # "toc_match" | "title" | "rule" | "carry" | "none"
    chapter_ok: bool  # True if reliable


def extract_sections(
    pages: list[dict[str, Any]],
    doc_type: str,
) -> list[SectionInfo]:
    """페이지 리스트에서 섹션 경계를 추출.

    Args:
        pages: VLM parsed pages (각 page는 {"page": int, "text": str})
        doc_type: 문서 종류 (sop, ts, setup, pems 등)

    Returns:
        페이지별 SectionInfo 리스트 (len == len(pages))
    """
    dt = doc_type.strip().lower()
    if dt in ("sop", "setup"):
        return _extract_sop_sections(pages)
    elif dt == "ts":
        return _extract_ts_sections(pages)
    else:
        return _extract_noop_sections(pages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fence(text: str) -> str:
    """Remove ```markdown ... ``` fences wrapping VLM output."""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _is_toc_page(text: str) -> bool:
    """Check if a page is a Table of Contents page."""
    stripped = _strip_markdown_fence(text)
    lines = stripped.splitlines()
    for line in lines[:15]:
        low = line.strip().lower()
        if low in ("## contents", "contents", "## 목차", "목차"):
            return True
        if re.match(r"^#+\s*(contents|목차)\s*$", low):
            return True
    return False


def _parse_toc_sop(pages: list[dict[str, Any]]) -> tuple[dict[int, str], int]:
    """SOP/Setup TOC에서 번호→제목 매핑 추출.

    Returns:
        (toc_map, toc_page_idx) where toc_map = {1: "Safety", 2: "Safety Lab", ...}
        toc_page_idx = index of the TOC page (-1 if not found)
    """
    toc_map: dict[int, str] = {}
    toc_page_idx = -1

    for idx, page in enumerate(pages):
        text = _strip_markdown_fence(page.get("text", ""))
        if not _is_toc_page(text):
            continue

        toc_page_idx = idx
        for line in text.splitlines():
            line = line.strip()
            # Match patterns like "1. Safety" or "- 1. Safety" or "10. Work Procedure"
            m = re.match(r"^[-*]?\s*(\d{1,2})\.\s+(.+?)(?:\s*\.{2,}.*)?$", line)
            if m:
                num = int(m.group(1))
                title = m.group(2).strip()
                # Skip if it looks like a page reference only
                if title and not re.match(r"^\d+$", title):
                    toc_map[num] = title
        if toc_map:
            break  # Use first TOC page found

    return toc_map, toc_page_idx


def _match_header_in_page(text: str) -> tuple[int, str] | None:
    """페이지 텍스트에서 ## N. Title 형식 헤더 추출.

    Returns:
        (number, title) or None
    """
    stripped = _strip_markdown_fence(text)
    for line in stripped.splitlines():
        line = line.strip()
        # Match "## 1. Safety" or "## 10. Work Procedure"
        m = re.match(r"^#{1,3}\s+(\d{1,2})\.\s+(.+)", line)
        if m:
            return int(m.group(1)), m.group(2).strip()
    return None


# ---------------------------------------------------------------------------
# SOP / Setup extraction
# ---------------------------------------------------------------------------

def _extract_sop_sections(pages: list[dict[str, Any]]) -> list[SectionInfo]:
    """SOP/Setup 문서의 섹션 경계 추출.

    Strategy:
    1. TOC 파싱 → 번호→제목 맵
    2. 각 페이지에서 ## N. Title 헤더 매칭
    3. TOC에 없는 번호의 헤더는 무시 (noise 방지)
    4. TOC 페이지 및 그 이전 페이지는 스킵
    5. Carry-forward: 헤더 없는 페이지는 이전 섹션 계승
    """
    toc_map, toc_page_idx = _parse_toc_sop(pages)
    toc_nums_set = set(toc_map.keys())

    results: list[SectionInfo] = []
    current_section = ""
    current_number = -1
    current_source = "none"

    for idx, page in enumerate(pages):
        text = page.get("text", "").strip()

        # Skip TOC page and all pages before it
        if toc_page_idx >= 0 and idx <= toc_page_idx:
            results.append(SectionInfo(
                section_chapter="",
                section_number=-1,
                chapter_source="none",
                chapter_ok=False,
            ))
            continue

        # Skip if this page itself is a TOC page (safety check)
        if _is_toc_page(text):
            results.append(SectionInfo(
                section_chapter="",
                section_number=-1,
                chapter_source="none",
                chapter_ok=False,
            ))
            continue

        # Try to match header in page
        header = _match_header_in_page(text)

        # Filter out headers whose numbers aren't in TOC (if TOC exists)
        if header and toc_map and header[0] not in toc_nums_set:
            header = None

        if header:
            num, title = header
            # Use TOC title if available (more reliable than OCR'd header)
            if num in toc_map:
                title = toc_map[num]
                source = "toc_match"
            else:
                source = "title"

            current_section = f"{num}. {title}"
            current_number = num
            current_source = source

            results.append(SectionInfo(
                section_chapter=current_section,
                section_number=current_number,
                chapter_source=current_source,
                chapter_ok=True,
            ))
        elif current_section:
            # Carry forward
            results.append(SectionInfo(
                section_chapter=current_section,
                section_number=current_number,
                chapter_source="carry",
                chapter_ok=True,
            ))
        else:
            results.append(SectionInfo(
                section_chapter="",
                section_number=-1,
                chapter_source="none",
                chapter_ok=False,
            ))

    return results


# ---------------------------------------------------------------------------
# TS extraction
# ---------------------------------------------------------------------------

_TS_ALPHA_TOC_RE = re.compile(
    r"^[-*]?\s*([A-Z])\.\s+(.+?)(?:\s*\.{2,}.*)?$"
)
_TS_SUBSECTION_RE = re.compile(
    r"^(?:#{1,3}\s+)?([A-Z])-(\d+)\.\s+(.+)"
)


def _parse_toc_ts(pages: list[dict[str, Any]]) -> tuple[dict[str, str], int]:
    """TS alpha TOC 파싱 (A. topic, B. topic, ...).

    Returns:
        (toc_map, toc_page_idx) where toc_map = {"A": "FFU Pressure range error", ...}
    """
    toc_map: dict[str, str] = {}
    toc_page_idx = -1

    for idx, page in enumerate(pages):
        text = _strip_markdown_fence(page.get("text", ""))
        found_items: dict[str, str] = {}

        for line in text.splitlines():
            line = line.strip()
            m = _TS_ALPHA_TOC_RE.match(line)
            if m:
                letter = m.group(1)
                title = m.group(2).strip()
                if title and not re.match(r"^\d+$", title):
                    found_items[letter] = title

        if len(found_items) >= 2:
            toc_map = found_items
            toc_page_idx = idx
            break

    return toc_map, toc_page_idx


def _extract_ts_sections(pages: list[dict[str, Any]]) -> list[SectionInfo]:
    """TS 문서의 섹션 경계 추출.

    Strategy:
    1. Alpha TOC 파싱 (A/B/C...)
    2. 각 페이지에서 X-N. subsection 패턴 매칭
    3. Carry-forward
    """
    toc_map, toc_page_idx = _parse_toc_ts(pages)

    results: list[SectionInfo] = []
    current_section = ""
    current_number = -1
    current_source = "none"

    for idx, page in enumerate(pages):
        text = page.get("text", "").strip()

        # Skip TOC page and pages before it
        if toc_page_idx >= 0 and idx <= toc_page_idx:
            results.append(SectionInfo(
                section_chapter="",
                section_number=-1,
                chapter_source="none",
                chapter_ok=False,
            ))
            continue

        # Try to match X-N. subsection pattern
        stripped = _strip_markdown_fence(text)
        matched_sub = None
        for line in stripped.splitlines():
            line = line.strip()
            m = _TS_SUBSECTION_RE.match(line)
            if m:
                letter = m.group(1)
                sub_num = int(m.group(2))
                sub_title = m.group(3).strip()
                # Use TOC title for the letter if available
                parent_title = toc_map.get(letter, "")
                if parent_title:
                    matched_sub = (letter, sub_num, parent_title, "toc_match")
                else:
                    matched_sub = (letter, sub_num, sub_title, "rule")
                break

        if matched_sub:
            letter, sub_num, title, source = matched_sub
            current_section = f"{letter}. {title}"
            current_number = ord(letter) - ord("A") + 1
            current_source = source

            results.append(SectionInfo(
                section_chapter=current_section,
                section_number=current_number,
                chapter_source=current_source,
                chapter_ok=True,
            ))
        elif current_section:
            results.append(SectionInfo(
                section_chapter=current_section,
                section_number=current_number,
                chapter_source="carry",
                chapter_ok=True,
            ))
        else:
            results.append(SectionInfo(
                section_chapter="",
                section_number=-1,
                chapter_source="none",
                chapter_ok=False,
            ))

    return results


# ---------------------------------------------------------------------------
# No-op (PEMS, etc.)
# ---------------------------------------------------------------------------

def _extract_noop_sections(pages: list[dict[str, Any]]) -> list[SectionInfo]:
    """섹션 추출 없음 (PEMS 등 doc-mode 문서)."""
    return [
        SectionInfo(
            section_chapter="",
            section_number=-1,
            chapter_source="none",
            chapter_ok=False,
        )
        for _ in pages
    ]


__all__ = ["SectionInfo", "extract_sections"]
