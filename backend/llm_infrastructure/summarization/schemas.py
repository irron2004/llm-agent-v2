"""Pydantic schemas for document summarization.

These schemas are used for structured LLM output parsing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TOCEntry(BaseModel):
    """Single table-of-contents entry."""

    title: str = Field(..., description="Chapter/section title (may include numbering)")
    start_page: int = Field(..., ge=1, description="Start page number (1-based, as shown in TOC)")
    level: int = Field(1, ge=1, description="TOC hierarchy level (1=chapter, 2=subsection, ...)")


class TOCParseResult(BaseModel):
    """Result of TOC parsing (LLM structured output)."""

    entries: list[TOCEntry] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    """Summary of a single text chunk."""

    summary: str = Field(..., description="Concise factual summary (3-7 sentences)")
    keywords: list[str] = Field(
        default_factory=list,
        description="5-12 keywords/key phrases",
    )
    # SOP-specific extensions
    actions: list[str] = Field(
        default_factory=list,
        description="Action steps mentioned in the chunk",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings, cautions, or prohibited items",
    )


class ChapterSummary(BaseModel):
    """Summary of a chapter (aggregated from chunk summaries)."""

    chapter_title: str = Field(..., description="Chapter title")
    summary: str = Field(..., description="1-2 paragraph summary")
    key_points: list[str] = Field(
        default_factory=list,
        description="5-10 key bullet points",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="8-20 keywords/key phrases",
    )


class DocumentMetadata(BaseModel):
    """Document-level metadata extracted from content."""

    device_name: str | None = Field(
        default=None,
        description="Equipment/device name mentioned in the document",
    )
    doc_type: str | None = Field(
        default=None,
        description="Document type (e.g., SOP, Manual, Guide, Specification)",
    )
    doc_version: str | None = Field(
        default=None,
        description="Document version if mentioned",
    )
    doc_date: str | None = Field(
        default=None,
        description="Document date if mentioned",
    )


class DocumentSummary(BaseModel):
    """Summary of the entire document."""

    summary: str = Field(..., description="1-2 paragraph overview")
    key_points: list[str] = Field(
        default_factory=list,
        description="7-15 key bullet points",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="10-25 keywords/key phrases",
    )
    metadata: DocumentMetadata | None = Field(
        default=None,
        description="Extracted document-level metadata",
    )


__all__ = [
    "TOCEntry",
    "TOCParseResult",
    "ChunkSummary",
    "ChapterSummary",
    "DocumentMetadata",
    "DocumentSummary",
]
