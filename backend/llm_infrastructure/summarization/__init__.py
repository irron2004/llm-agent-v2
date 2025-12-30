"""Document summarization module.

Provides hierarchical summarization: chunk -> chapter -> document.
"""

from .base import BaseSummarizer, SummaryResult
from .registry import SummarizerRegistry, register_summarizer, get_summarizer
from .schemas import (
    TOCEntry,
    TOCParseResult,
    ChunkSummary,
    ChapterSummary,
    DocumentMetadata,
    DocumentSummary,
)

# Import adapters to trigger registration
from . import adapters  # noqa: F401

__all__ = [
    "BaseSummarizer",
    "SummaryResult",
    "SummarizerRegistry",
    "register_summarizer",
    "get_summarizer",
    "TOCEntry",
    "TOCParseResult",
    "ChunkSummary",
    "ChapterSummary",
    "DocumentMetadata",
    "DocumentSummary",
]
