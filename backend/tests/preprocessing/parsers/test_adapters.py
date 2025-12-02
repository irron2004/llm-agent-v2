"""Tests for PDF parser adapters and registry integration."""

import io
from unittest.mock import MagicMock, patch

import pytest

from llm_infrastructure.preprocessing.parsers import get_parser, list_parsers
from llm_infrastructure.preprocessing.parsers.adapters.pdf_deepdoc import DeepDocPdfAdapter
from llm_infrastructure.preprocessing.parsers.adapters.pdf_plain import PlainPdfAdapter
from llm_infrastructure.preprocessing.parsers.adapters.pdf_vlm import VlmPdfAdapter, DeepSeekVLPdfAdapter
from llm_infrastructure.preprocessing.parsers.base import ParsedDocument, PdfParseOptions


class TestDeepDocPdfAdapter:
    def test_initialization_default_engine(self):
        """Test adapter initialization with default engine."""
        adapter = DeepDocPdfAdapter()
        assert adapter.content_type == "application/pdf"
        assert adapter.engine is not None

    def test_initialization_custom_engine(self):
        """Test adapter initialization with custom engine."""
        mock_engine = MagicMock()
        adapter = DeepDocPdfAdapter(engine=mock_engine)
        assert adapter.engine == mock_engine

    def test_parse_delegates_to_engine(self):
        """Test that parse() delegates to engine.run()."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = ParsedDocument(metadata={"test": "value"})

        adapter = DeepDocPdfAdapter(engine=mock_engine)
        file = io.BytesIO(b"dummy")
        opts = PdfParseOptions(max_pages=10)

        result = adapter.parse(file, opts)

        mock_engine.run.assert_called_once_with(file, options=opts)
        assert result.metadata["test"] == "value"

    def test_parse_without_options(self):
        """Test parse() with no options provided."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = ParsedDocument()

        adapter = DeepDocPdfAdapter(engine=mock_engine)
        adapter.parse(io.BytesIO(b"dummy"))

        mock_engine.run.assert_called_once()
        call_args = mock_engine.run.call_args
        assert call_args[1]["options"] is None


class TestPlainPdfAdapter:
    def test_initialization_default_engine(self):
        """Test adapter initialization with default engine."""
        adapter = PlainPdfAdapter()
        assert adapter.content_type == "application/pdf"
        assert adapter.engine is not None

    def test_initialization_custom_engine(self):
        """Test adapter initialization with custom engine."""
        mock_engine = MagicMock()
        adapter = PlainPdfAdapter(engine=mock_engine)
        assert adapter.engine == mock_engine

    def test_parse_delegates_to_engine(self):
        """Test that parse() delegates to engine.run()."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = ParsedDocument(metadata={"parser": "plain"})

        adapter = PlainPdfAdapter(engine=mock_engine)
        file = io.BytesIO(b"dummy")
        opts = PdfParseOptions()

        result = adapter.parse(file, opts)

        mock_engine.run.assert_called_once_with(file, options=opts)
        assert result.metadata["parser"] == "plain"


class TestVlmPdfAdapter:
    def test_initialization_default_engine(self):
        """Test adapter initialization with default engine."""
        adapter = VlmPdfAdapter()
        assert adapter.content_type == "application/pdf"
        assert adapter.engine is not None

    def test_initialization_custom_engine(self):
        """Test adapter initialization with custom engine."""
        mock_engine = MagicMock()
        adapter = VlmPdfAdapter(engine=mock_engine)
        assert adapter.engine == mock_engine

    def test_initialization_with_engine_kwargs(self):
        """Test adapter initialization with engine kwargs."""
        mock_vlm_client = MagicMock()
        adapter = VlmPdfAdapter(vlm_client=mock_vlm_client)
        assert adapter.engine.vlm_client == mock_vlm_client

    def test_parse_delegates_to_engine(self):
        """Test that parse() delegates to engine.run()."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = ParsedDocument(metadata={"parser": "pdf_vlm"})

        adapter = VlmPdfAdapter(engine=mock_engine)
        file = io.BytesIO(b"dummy")
        opts = PdfParseOptions(vlm_model="deepseek-vl2")

        result = adapter.parse(file, opts)

        mock_engine.run.assert_called_once_with(file, options=opts)
        assert result.metadata["parser"] == "pdf_vlm"

    def test_parse_without_options(self):
        """Test parse() with no options provided."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = ParsedDocument()

        adapter = VlmPdfAdapter(engine=mock_engine)
        adapter.parse(io.BytesIO(b"dummy"))

        mock_engine.run.assert_called_once()
        call_args = mock_engine.run.call_args
        assert call_args[1]["options"] is None

    def test_backward_compatible_alias(self):
        """Test that DeepSeekVLPdfAdapter is an alias for VlmPdfAdapter."""
        assert DeepSeekVLPdfAdapter is VlmPdfAdapter


class TestRegistryIntegration:
    def test_pdf_deepdoc_registered(self):
        """Test that pdf_deepdoc parser is registered."""
        parsers = list(list_parsers())
        assert "pdf_deepdoc" in parsers

    def test_pdf_plain_registered(self):
        """Test that pdf_plain parser is registered."""
        parsers = list(list_parsers())
        assert "pdf_plain" in parsers

    def test_pdf_vlm_registered(self):
        """Test that pdf_vlm parser is registered."""
        parsers = list(list_parsers())
        assert "pdf_vlm" in parsers

    def test_pdf_deepseek_vl_registered_as_alias(self):
        """Test that pdf_deepseek_vl is registered as backward-compatible alias."""
        parsers = list(list_parsers())
        assert "pdf_deepseek_vl" in parsers

    def test_get_pdf_deepdoc_parser(self):
        """Test retrieving pdf_deepdoc parser from registry."""
        parser = get_parser("pdf_deepdoc")
        assert isinstance(parser, DeepDocPdfAdapter)
        assert parser.content_type == "application/pdf"

    def test_get_pdf_plain_parser(self):
        """Test retrieving pdf_plain parser from registry."""
        parser = get_parser("pdf_plain")
        assert isinstance(parser, PlainPdfAdapter)
        assert parser.content_type == "application/pdf"

    def test_get_pdf_vlm_parser(self):
        """Test retrieving pdf_vlm parser from registry."""
        parser = get_parser("pdf_vlm")
        assert isinstance(parser, VlmPdfAdapter)
        assert parser.content_type == "application/pdf"

    def test_get_pdf_deepseek_vl_parser_alias(self):
        """Test retrieving pdf_deepseek_vl parser (backward-compatible alias)."""
        parser = get_parser("pdf_deepseek_vl")
        assert isinstance(parser, VlmPdfAdapter)
        assert parser.content_type == "application/pdf"

    def test_get_parser_with_custom_engine(self):
        """Test getting parser with custom engine via kwargs."""
        mock_engine = MagicMock()
        parser = get_parser("pdf_deepdoc", engine=mock_engine)
        assert parser.engine == mock_engine

    def test_end_to_end_parsing_with_registry(self):
        """Test end-to-end parsing using registry."""
        # Mock the engine to avoid dependencies
        mock_engine = MagicMock()
        mock_engine.run.return_value = ParsedDocument(
            blocks=[MagicMock(text="Sample text", page=1)], metadata={"parser": "pdf_deepdoc"}
        )

        parser = get_parser("pdf_deepdoc", engine=mock_engine)
        result = parser.parse(io.BytesIO(b"dummy"), PdfParseOptions())

        assert len(result.blocks) == 1
        assert result.metadata["parser"] == "pdf_deepdoc"
