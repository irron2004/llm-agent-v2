"""Tests for PlainPdfEngine."""

import io
from unittest.mock import MagicMock, patch

import pytest

from llm_infrastructure.preprocessing.parsers.base import PdfParseOptions
from llm_infrastructure.preprocessing.parsers.engines.pdf_plain_engine import PlainPdfEngine


class TestPlainPdfEngine:
    def test_import_error_when_pdfplumber_missing(self):
        """Test that ImportError is raised when pdfplumber is not available."""
        with patch("llm_infrastructure.preprocessing.parsers.engines.pdf_plain_engine.pdfplumber", None):
            engine = PlainPdfEngine()
            with pytest.raises(ImportError, match="pdfplumber is required"):
                engine.run(io.BytesIO(b"dummy"), PdfParseOptions())

    def test_parse_simple_pdf(self):
        """Test parsing a simple PDF with mocked pdfplumber."""
        # Create mock PDF with 2 pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page1.width = 612.0
        mock_page1.height = 792.0

        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_page2.width = 612.0
        mock_page2.height = 792.0

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch("llm_infrastructure.preprocessing.parsers.engines.pdf_plain_engine.pdfplumber", mock_pdfplumber):
            engine = PlainPdfEngine()
            result = engine.run(io.BytesIO(b"dummy"), PdfParseOptions())

        assert len(result.pages) == 2
        assert result.pages[0].number == 1
        assert result.pages[0].text == "Page 1 content"
        assert result.pages[0].width == 612.0
        assert result.pages[1].number == 2
        assert result.pages[1].text == "Page 2 content"

        assert len(result.blocks) == 2
        assert result.blocks[0].text == "Page 1 content"
        assert result.blocks[0].page == 1
        assert result.blocks[0].label == "page"

        assert result.tables == []
        assert result.figures == []
        assert result.metadata["parser"] == "pdf_plain"
        assert result.metadata["ocr"] is False
        assert result.metadata["layout"] is False

    def test_max_pages_limit(self):
        """Test that max_pages option limits the number of processed pages."""
        mock_pages = [MagicMock(extract_text=MagicMock(return_value=f"Page {i}")) for i in range(1, 11)]
        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch("llm_infrastructure.preprocessing.parsers.engines.pdf_plain_engine.pdfplumber", mock_pdfplumber):
            engine = PlainPdfEngine()
            result = engine.run(io.BytesIO(b"dummy"), PdfParseOptions(max_pages=3))

        assert len(result.pages) == 3
        assert len(result.blocks) == 3
        assert result.metadata["max_pages"] == 3

    def test_empty_page_text(self):
        """Test handling of pages with no text."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None  # pdfplumber returns None for empty pages
        mock_page.width = 612.0
        mock_page.height = 792.0

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch("llm_infrastructure.preprocessing.parsers.engines.pdf_plain_engine.pdfplumber", mock_pdfplumber):
            engine = PlainPdfEngine()
            result = engine.run(io.BytesIO(b"dummy"), PdfParseOptions())

        assert len(result.pages) == 1
        assert result.pages[0].text == ""
        assert result.blocks[0].text == ""

    def test_seek_before_parsing(self):
        """Test that engine seeks to beginning of file before parsing."""
        file = io.BytesIO(b"dummy content")
        file.read()  # Move cursor to end
        assert file.tell() > 0

        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch("llm_infrastructure.preprocessing.parsers.engines.pdf_plain_engine.pdfplumber", mock_pdfplumber):
            engine = PlainPdfEngine()
            engine.run(file, PdfParseOptions())

        assert file.tell() == 0  # Should be reset to beginning

    def test_preserve_layout_in_metadata(self):
        """Test that preserve_layout option is recorded in metadata."""
        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch("llm_infrastructure.preprocessing.parsers.engines.pdf_plain_engine.pdfplumber", mock_pdfplumber):
            engine = PlainPdfEngine()
            result = engine.run(io.BytesIO(b"dummy"), PdfParseOptions(preserve_layout=True))

        assert result.metadata["preserve_layout"] is True

    def test_parse_error_propagates(self):
        """Test that parsing errors are propagated."""
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.side_effect = Exception("PDF corrupted")

        with patch("llm_infrastructure.preprocessing.parsers.engines.pdf_plain_engine.pdfplumber", mock_pdfplumber):
            engine = PlainPdfEngine()
            with pytest.raises(Exception, match="PDF corrupted"):
                engine.run(io.BytesIO(b"dummy"), PdfParseOptions())

    def test_content_type(self):
        """Test that content_type is set correctly."""
        engine = PlainPdfEngine()
        assert engine.content_type == "application/pdf"