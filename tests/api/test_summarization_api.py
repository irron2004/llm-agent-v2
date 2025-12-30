"""Tests for summarization API endpoints."""

import pytest
from unittest.mock import MagicMock, patch

from backend.llm_infrastructure.summarization.schemas import (
    ChunkSummary,
    ChapterSummary,
    DocumentMetadata,
    DocumentSummary,
)
from backend.llm_infrastructure.summarization.toc_parser import parse_toc_regex


class TestTOCParser:
    """Tests for TOC regex parsing."""

    def test_parse_toc_with_dot_leaders(self):
        """Test parsing TOC with dot leaders."""
        toc_text = """목차
1. 소개 .......... 2
2. 절차 .......... 5
3. 주의사항 ...... 12
"""
        entries = parse_toc_regex(toc_text)

        assert len(entries) == 3
        assert entries[0].title == "1. 소개"
        assert entries[0].start_page == 2
        assert entries[0].level == 1
        assert entries[2].start_page == 12

    def test_parse_toc_without_dot_leaders(self):
        """Test parsing TOC without dot leaders."""
        toc_text = """1 Introduction 3
2 Setup 5
3 Troubleshooting 10
"""
        entries = parse_toc_regex(toc_text)

        assert len(entries) == 3
        assert entries[0].title == "1 Introduction"
        assert entries[0].start_page == 3

    def test_parse_toc_with_hierarchy(self):
        """Test parsing TOC with hierarchical numbering."""
        toc_text = """1. Overview 2
1.1 Background 3
1.2 Scope 4
2. Procedure 5
2.1 Prerequisites 6
2.1.1 Tools 7
"""
        entries = parse_toc_regex(toc_text)

        assert len(entries) == 6
        assert entries[0].level == 1  # 1.
        assert entries[1].level == 2  # 1.1
        assert entries[5].level == 3  # 2.1.1

    def test_parse_empty_toc(self):
        """Test parsing empty TOC."""
        entries = parse_toc_regex("")
        assert entries == []

    def test_parse_toc_filters_non_toc_lines(self):
        """Test that non-TOC lines are filtered."""
        toc_text = """This is a header
Some random text without page numbers
1. Actual Entry 5
More random text
"""
        entries = parse_toc_regex(toc_text)

        assert len(entries) == 1
        assert entries[0].title == "1. Actual Entry"


class TestChunkOverlapValidation:
    """Tests for chunk overlap validation in API."""

    def test_overlap_less_than_size_valid(self):
        """Test that overlap < size is valid."""
        from pydantic import ValidationError
        from backend.api.routers.summarization import DocumentSummarizationRequest

        # Should not raise
        req = DocumentSummarizationRequest(
            pages=["page 1"],
            chunk_size=500,
            chunk_overlap=100,
        )
        assert req.chunk_overlap < req.chunk_size

    def test_overlap_equals_size_invalid(self):
        """Test that overlap == size raises validation error."""
        from pydantic import ValidationError
        from backend.api.routers.summarization import DocumentSummarizationRequest

        with pytest.raises(ValidationError) as exc_info:
            DocumentSummarizationRequest(
                pages=["page 1"],
                chunk_size=500,
                chunk_overlap=500,
            )

        assert "chunk_overlap" in str(exc_info.value)

    def test_overlap_greater_than_size_invalid(self):
        """Test that overlap > size raises validation error."""
        from pydantic import ValidationError
        from backend.api.routers.summarization import DocumentSummarizationRequest

        with pytest.raises(ValidationError) as exc_info:
            DocumentSummarizationRequest(
                pages=["page 1"],
                chunk_size=100,
                chunk_overlap=200,
            )

        assert "chunk_overlap" in str(exc_info.value)


class TestSummarizationSchemas:
    """Tests for summarization Pydantic schemas."""

    def test_chunk_summary_defaults(self):
        """Test ChunkSummary with defaults."""
        summary = ChunkSummary(summary="Test summary")

        assert summary.summary == "Test summary"
        assert summary.keywords == []
        assert summary.actions == []
        assert summary.warnings == []

    def test_chunk_summary_with_all_fields(self):
        """Test ChunkSummary with all fields."""
        summary = ChunkSummary(
            summary="Test summary",
            keywords=["kw1", "kw2"],
            actions=["action1"],
            warnings=["warning1"],
        )

        assert len(summary.keywords) == 2
        assert len(summary.actions) == 1
        assert len(summary.warnings) == 1

    def test_document_summary_with_metadata(self):
        """Test DocumentSummary with metadata."""
        metadata = DocumentMetadata(
            device_name="ABC-123",
            doc_type="SOP",
            doc_version="1.0",
        )
        summary = DocumentSummary(
            summary="Document overview",
            key_points=["point1"],
            keywords=["kw1"],
            metadata=metadata,
        )

        assert summary.metadata is not None
        assert summary.metadata.device_name == "ABC-123"
        assert summary.metadata.doc_type == "SOP"

    def test_document_metadata_all_optional(self):
        """Test that all DocumentMetadata fields are optional."""
        metadata = DocumentMetadata()

        assert metadata.device_name is None
        assert metadata.doc_type is None
        assert metadata.doc_version is None
        assert metadata.doc_date is None


class TestSummarizationService:
    """Tests for DocumentSummarizationService."""

    def test_chunk_text_prevents_infinite_loop(self):
        """Test that chunking handles overlap >= size gracefully."""
        from backend.services.document_summarization_service import DocumentSummarizationService

        # Create service with overlap == size (would cause infinite loop without fix)
        service = DocumentSummarizationService(
            chunk_size=100,
            chunk_overlap=100,  # Same as size
        )

        # Should not hang - the service clamps overlap internally
        # Use shorter text to avoid excessive chunks
        text = "A" * 150
        chunks = service._chunk_text(text)

        # Should produce chunks without infinite loop
        # With clamped overlap (99), each chunk advances 1 char minimum
        assert len(chunks) > 0
        # Verify the loop terminates (if it didn't, test would timeout)

    def test_chunk_text_small_text(self):
        """Test chunking text smaller than chunk_size."""
        from backend.services.document_summarization_service import DocumentSummarizationService

        service = DocumentSummarizationService(chunk_size=1000)

        text = "Short text"
        chunks = service._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        from backend.services.document_summarization_service import DocumentSummarizationService

        service = DocumentSummarizationService()

        chunks = service._chunk_text("")
        assert chunks == []

        chunks = service._chunk_text("   ")
        assert chunks == []
