"""Parser for GCB (Global Customer Bulletin) text files.

Parses structured GCB reports with header, title, question, resolution, and tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GCBReport:
    """Parsed GCB report."""

    meta: dict[str, Any]
    sections: dict[str, str]  # section_name -> text (question, resolution)
    full_text: str  # Combined text for embedding


def parse_gcb_txt(content: str) -> GCBReport:
    """Parse GCB text file.

    Expected format:
        ### GCB 344 | Status: Close | Model: TERA21 | Req:
        **Title**: Some title here

        **Question (원인/문의/초기 가설)**:
        Question text...

        **Confirmed Resolution (최종 확정 조치)**:
        Resolution text...

        **Tags**: model=TERA21; req=; os=N/A; patch=N/A; module=CHAMBER

    Args:
        content: Raw text content.

    Returns:
        GCBReport with parsed meta and sections.
    """
    # Parse header line
    meta = _parse_header(content)

    # Parse title
    title = _parse_title(content)
    if title:
        meta["title"] = title

    # Parse tags and merge into meta
    tags = _parse_tags(content)
    meta.update(tags)

    # Map model to device_name for consistency with myservice
    if "model" in meta:
        meta["device_name"] = meta["model"]

    # Parse question and resolution sections
    sections = _parse_sections(content)

    # Build full text for embedding
    full_text = _build_full_text(title, sections)

    return GCBReport(
        meta=meta,
        sections=sections,
        full_text=full_text,
    )


def _parse_header(content: str) -> dict[str, Any]:
    """Parse header line: ### GCB 344 | Status: Close | Model: TERA21 | Req:"""
    meta = {}

    # Match header pattern (ignore Req field as it's usually empty)
    header_pattern = r"###\s*GCB\s*(\d+)\s*\|\s*Status:\s*([^|]*)\|\s*Model:\s*([^|]*)"
    match = re.search(header_pattern, content, re.IGNORECASE)

    if match:
        meta["gcb_number"] = match.group(1).strip()
        meta["status"] = match.group(2).strip()
        meta["model"] = match.group(3).strip()

    return meta


def _parse_title(content: str) -> str:
    """Parse **Title**: ... line."""
    title_pattern = r"\*\*Title\*\*:\s*(.+?)(?:\n\n|\n\*\*|$)"
    match = re.search(title_pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()
    return ""


def _parse_tags(content: str) -> dict[str, Any]:
    """Parse **Tags**: key=value; key=value; ... line."""
    tags = {}

    tags_pattern = r"\*\*Tags\*\*:\s*(.+?)(?:\n|$)"
    match = re.search(tags_pattern, content)

    if match:
        tags_str = match.group(1).strip()
        # Parse semicolon-separated key=value pairs
        for pair in tags_str.split(";"):
            pair = pair.strip()
            if "=" in pair:
                key, value = pair.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                # Skip N/A or empty values
                if value and value.upper() != "N/A":
                    tags[key] = value

    return tags


def _parse_sections(content: str) -> dict[str, str]:
    """Parse Question and Confirmed Resolution sections."""
    sections = {}

    # Question section
    question_pattern = r"\*\*Question\s*\([^)]*\)\*\*:\s*(.+?)(?=\*\*Confirmed Resolution|\*\*Tags\*\*|$)"
    match = re.search(question_pattern, content, re.DOTALL)
    if match:
        sections["question"] = match.group(1).strip()
    else:
        sections["question"] = ""

    # Confirmed Resolution section
    resolution_pattern = r"\*\*Confirmed Resolution\s*\([^)]*\)\*\*:\s*(.+?)(?=\*\*Tags\*\*|$)"
    match = re.search(resolution_pattern, content, re.DOTALL)
    if match:
        sections["resolution"] = match.group(1).strip()
    else:
        sections["resolution"] = ""

    return sections


def _build_full_text(title: str, sections: dict[str, str]) -> str:
    """Build combined text from title and sections for embedding."""
    parts = []

    if title:
        parts.append(f"Title: {title}")
        parts.append("")

    if sections.get("question"):
        parts.append("[question]")
        parts.append(sections["question"])
        parts.append("")

    if sections.get("resolution"):
        parts.append("[resolution]")
        parts.append(sections["resolution"])

    return "\n".join(parts).strip()


__all__ = ["GCBReport", "parse_gcb_txt"]
