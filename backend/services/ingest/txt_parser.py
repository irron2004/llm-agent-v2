"""Parser for maintenance report text files.

Parses structured maintenance reports with [meta], [status], [action], [cause], [result] sections.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class MaintenanceReport:
    """Parsed maintenance report."""

    meta: dict[str, Any]
    sections: dict[str, str]  # section_name -> text
    full_text: str  # Combined text for embedding


def parse_maintenance_txt(content: str) -> MaintenanceReport:
    """Parse maintenance report text file.

    Expected format:
        >>>> [meta]
        {
          "Order No.": "...",
          ...
        }

        [status]
        # text

        [action]
        # text

        [cause]
        text

        [result]
        text

    Args:
        content: Raw text content.

    Returns:
        MaintenanceReport with parsed meta and sections.
    """
    # Parse meta section (JSON)
    meta = _parse_meta_section(content)

    # Parse text sections
    sections = _parse_text_sections(content)

    # Build full text for embedding (exclude meta)
    full_text = _build_full_text(sections)

    return MaintenanceReport(
        meta=meta,
        sections=sections,
        full_text=full_text,
    )


def _parse_meta_section(content: str) -> dict[str, Any]:
    """Extract and parse [meta] JSON section."""
    # Find [meta] block (with or without >>>>)
    # Match until next section header or end
    meta_pattern = r"(?:>>>>)?\s*\[meta\](.*?)(?=\[(?:status|action|cause|result)\]|\Z)"
    match = re.search(meta_pattern, content, re.DOTALL | re.IGNORECASE)

    if not match:
        return {}

    json_text = match.group(1).strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Try to extract JSON from potential surrounding text
        json_match = re.search(r"\{.*\}", json_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def _parse_text_sections(content: str) -> dict[str, str]:
    """Parse [status], [action], [cause], [result] sections."""
    sections = {}

    # Define section names
    section_names = ["status", "action", "cause", "result"]

    for section_name in section_names:
        # Pattern: [section_name] ... until next [section] or end
        # Use negative lookahead to stop before any new section header
        pattern = rf"\[{section_name}\](.*?)(?=\[(?:status|action|cause|result)\]|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            text = match.group(1).strip()
            sections[section_name] = text
        else:
            sections[section_name] = ""

    return sections


def _build_full_text(sections: dict[str, str]) -> str:
    """Build combined text from sections for embedding.

    Format:
        [status]
        text

        [action]
        text

        [cause]
        text

        [result]
        text
    """
    parts = []

    for section_name in ["status", "action", "cause", "result"]:
        if section_name in sections and sections[section_name]:
            parts.append(f"[{section_name}]")
            parts.append(sections[section_name])
            parts.append("")  # Empty line separator

    return "\n".join(parts).strip()


__all__ = ["MaintenanceReport", "parse_maintenance_txt"]
