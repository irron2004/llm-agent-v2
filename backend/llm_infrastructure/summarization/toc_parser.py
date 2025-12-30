"""Table of Contents (TOC) parsing utilities.

Supports both regex-based and LLM-based TOC extraction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from .schemas import TOCEntry, TOCParseResult

if TYPE_CHECKING:
    from backend.llm_infrastructure.llm.base import BaseLLM


# Prompt directory for TOC parsing
PROMPT_DIR = Path(__file__).parent / "prompts"


# -----------------------------------------------------------------------------
# Regex-based TOC Parsing
# -----------------------------------------------------------------------------
_TOC_LINE_RE = re.compile(r"^\s*(?P<title>.+?)\s*(?:\.{2,}\s*)?(?P<page>\d{1,4})\s*$")


def _looks_like_toc_line(line: str) -> bool:
    """Check if line looks like a TOC entry (has text + page number)."""
    has_text = bool(re.search(r"[A-Za-z가-힣]", line))
    has_page = bool(re.search(r"\d{1,4}\s*$", line))
    return has_text and has_page


def _infer_level_from_title(title: str) -> int:
    """Infer TOC hierarchy level from numbering pattern.

    Examples:
        "1" -> level 1
        "1.2" -> level 2
        "2.3.1" -> level 3
    """
    m = re.match(r"^\s*(\d+(?:\.\d+)*)\s*[\.\)]?\s+(.+)$", title)
    if not m:
        return 1
    numbering = m.group(1)
    return numbering.count(".") + 1


def parse_toc_regex(toc_text: str) -> list[TOCEntry]:
    """Parse TOC using regex patterns.

    Works well for standard TOC formats with dot leaders.

    Args:
        toc_text: Raw text of the TOC page.

    Returns:
        List of TOCEntry objects sorted by start_page.
    """
    entries: list[TOCEntry] = []

    for raw_line in toc_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not _looks_like_toc_line(line):
            continue

        m = _TOC_LINE_RE.match(line)
        if not m:
            # Fallback: "1 Introduction 3" (no dot leader)
            m2 = re.match(r"^\s*(?P<title>.+?)\s+(?P<page>\d{1,4})\s*$", line)
            if not m2:
                continue
            title = m2.group("title").strip()
            page = int(m2.group("page"))
        else:
            title = m.group("title").strip()
            page = int(m.group("page"))

        level = _infer_level_from_title(title)
        entries.append(TOCEntry(title=title, start_page=page, level=level))

    # Sort by page, then by level
    entries.sort(key=lambda x: (x.start_page, x.level))
    return entries


# -----------------------------------------------------------------------------
# LLM-based TOC Parsing
# -----------------------------------------------------------------------------
def _load_toc_prompt() -> dict[str, Any]:
    """Load TOC parsing prompt from YAML."""
    prompt_path = PROMPT_DIR / "toc_parse_v1.yaml"
    if not prompt_path.exists():
        # Fallback inline prompt
        return {
            "system": (
                "You extract table-of-contents entries from a technical SOP/guide.\n"
                "Return ONLY the TOC entries as structured data.\n"
                "- Do NOT invent entries.\n"
                "- start_page must be an integer (1-based) as written in the TOC.\n"
                "- If the TOC mixes Korean/English, keep titles as-is.\n"
            ),
            "user": (
                "TOC text:\n\n{toc_text}\n\n"
                "Extract TOC entries (chapter/section title, start page, level)."
            ),
        }
    with prompt_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {
        "system": data.get("system", ""),
        "user": next(
            (m["content"] for m in data.get("messages", []) if m.get("role") == "user"),
            "",
        ),
    }


class TOCParseResponse(BaseModel):
    """LLM response model for TOC parsing."""

    entries: list[TOCEntry] = Field(default_factory=list)


def parse_toc_llm(toc_text: str, llm: "BaseLLM") -> list[TOCEntry]:
    """Parse TOC using LLM with structured output.

    Args:
        toc_text: Raw text of the TOC page.
        llm: LLM instance for generation.

    Returns:
        List of TOCEntry objects sorted by start_page.
    """
    prompt = _load_toc_prompt()
    user_content = prompt["user"].format(toc_text=toc_text)

    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": user_content},
    ]

    # Generate with Pydantic response model
    try:
        result = llm.generate(messages, response_model=TOCParseResponse)
        entries = result.entries
    except Exception:
        # Fallback: return empty if parsing fails
        return []

    entries.sort(key=lambda x: (x.start_page, x.level))
    return entries


# -----------------------------------------------------------------------------
# Page Matching - Find actual pages for TOC entries
# -----------------------------------------------------------------------------
def match_toc_to_pages(
    toc_entries: list[TOCEntry],
    pages: list[str],
    threshold: int = 70,
    toc_page_indices: list = []

) -> list[TOCEntry]:
    """Find actual page numbers for TOC entries by searching page content.

    Searches all pages for each TOC title and updates start_page
    to the actual page where the title appears.

    Args:
        toc_entries: TOC entries (start_page may be dummy values).
        pages: List of page texts (0-indexed).
        threshold: Minimum fuzzy match score (0-100) to consider a match.

    Returns:
        TOC entries with corrected start_page values (1-based).
    """
    if not toc_entries or not pages:
        return toc_entries

    updated_entries: list[TOCEntry] = []

    for entry in toc_entries:
        best_page = -1
        best_score = 0

        # Search all pages for this title
        for page_idx, page_text in enumerate(pages):
            if page_idx in toc_page_indices:
                continue

            # Check first ~1500 chars where heading likely appears
            hay = page_text[:100]
            score = fuzz.partial_ratio(entry.title, hay)

            if score > best_score and score >= threshold:
                best_score = score
                best_page = page_idx

        # Update start_page if found (convert to 1-based)
        if best_page >= 0:
            updated_entries.append(
                TOCEntry(
                    title=entry.title,
                    start_page=best_page + 1,  # 1-based
                    level=entry.level,
                )
            )
        else:
            # Keep original if not found
            updated_entries.append(entry)

    # Sort by actual page
    updated_entries.sort(key=lambda x: (x.start_page, x.level))
    return updated_entries


def infer_page_offset(
    pages: list[str],
    toc_entries: list[TOCEntry],
    max_offset: int = 10,
    sample_k: int = 8,
) -> int:
    """Infer offset between TOC page numbers and actual page indices.

    DEPRECATED: Use match_toc_to_pages() instead for TOCs without page numbers.

    Args:
        pages: List of page texts (0-indexed).
        toc_entries: Parsed TOC entries with start_page (1-based).
        max_offset: Maximum offset to search.
        sample_k: Number of TOC entries to sample for matching.

    Returns:
        Offset value. toc.start_page + offset - 1 = pages index.
    """
    if not toc_entries:
        return 0

    entries = toc_entries[:sample_k]
    best_offset = 0
    best_score = -1.0

    for off in range(-max_offset, max_offset + 1):
        score = 0.0
        used = 0
        for e in entries:
            idx = (e.start_page + off) - 1
            if 0 <= idx < len(pages):
                # Check first ~1200 chars where heading likely appears
                hay = pages[idx][:1200]
                s = fuzz.partial_ratio(e.title, hay)
                score += s
                used += 1
        if used > 0:
            avg_score = score / used
            if avg_score > best_score:
                best_score = avg_score
                best_offset = off

    # If matching score is too low, default to 0
    if best_score < 55:
        return 0
    return best_offset


# -----------------------------------------------------------------------------
# Main Parser Function
# -----------------------------------------------------------------------------
def parse_toc(
    toc_text: str,
    *,
    llm: "BaseLLM | None" = None,
) -> list[TOCEntry]:
    """Parse table of contents from text using LLM.

    Args:
        toc_text: Raw text of the TOC page.
        llm: LLM instance for parsing (required for meaningful results).

    Returns:
        List of TOCEntry objects.
    """
    if llm is None:
        # Without LLM, return empty (regex is unreliable for varied formats)
        return []

    return parse_toc_llm(toc_text, llm)


__all__ = [
    "parse_toc",
    "parse_toc_regex",
    "parse_toc_llm",
    "match_toc_to_pages",
    "infer_page_offset",  # Deprecated
    "TOCEntry",
]
