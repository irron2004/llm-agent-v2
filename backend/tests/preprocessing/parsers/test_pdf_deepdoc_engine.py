"""Tests for DeepDocPdfEngine."""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from llm_infrastructure.preprocessing.parsers.base import BoundingBox, ParsedBlock, ParsedFigure, ParsedTable, PdfParseOptions
from llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine import DeepDocPdfEngine


class MockDeepDocParser:
    """Mock DeepDoc parser for testing."""

    def __init__(self):
        self.called_with = None

    def __call__(self, pdf_path, **kwargs):
        self.called_with = (pdf_path, kwargs)
        return {
            "chunks": [
                {"text": "Sample text", "page": 1, "bbox": {"x0": 10, "y0": 20, "x1": 100, "y1": 200}},
                {"text": "Another block", "page": 2, "bbox": [5, 5, 50, 50]},
            ],
            "tables": [{"page": 1, "html": "<table></table>", "text": "Table content"}],
            "figures": [{"page": 1, "caption": "Figure 1", "image": "/path/to/fig.png"}],
        }


class TestDeepDocPdfEngine:
    def test_load_backend_class_deepdoc_parser(self):
        """Test loading DeepDoc parser from deepdoc.parser.pdf_parser."""
        mock_parser = type("RAGFlowPdfParser", (), {})
        mock_module = MagicMock()
        mock_module.RAGFlowPdfParser = mock_parser

        with patch("importlib.import_module", return_value=mock_module) as mock_import:
            engine = DeepDocPdfEngine()
            backend_cls = engine._load_backend_class()

        assert backend_cls == mock_parser
        mock_import.assert_called()

    def test_load_backend_class_not_found(self):
        """Test when no DeepDoc backend is available."""
        with patch("importlib.import_module", side_effect=ImportError("No module named deepdoc")):
            engine = DeepDocPdfEngine()
            backend_cls = engine._load_backend_class()

        assert backend_cls is None

    def test_configure_hf_env(self):
        """Test HuggingFace environment configuration."""
        opts = PdfParseOptions(hf_endpoint="https://hf-mirror.com", model_root=Path("/tmp/models"))

        with patch.dict(os.environ, {}, clear=True):
            engine = DeepDocPdfEngine()
            engine._configure_hf_env(opts)

            assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
            assert os.environ["HF_HOME"] == "/tmp/models"
            assert os.environ["HUGGINGFACE_HUB_CACHE"] == "/tmp/models"
            assert os.environ["TRANSFORMERS_CACHE"] == "/tmp/models"

    def test_configure_hf_env_no_override(self):
        """Test that existing environment variables are not overridden."""
        with patch.dict(os.environ, {"HF_ENDPOINT": "existing_value"}, clear=True):
            opts = PdfParseOptions(hf_endpoint="https://new-value.com")
            engine = DeepDocPdfEngine()
            engine._configure_hf_env(opts)

            assert os.environ["HF_ENDPOINT"] == "existing_value"

    def test_maybe_download_models_disabled(self):
        """Test that model download is skipped when disabled."""
        opts = PdfParseOptions(allow_download=False, model_root=Path("/tmp/models"), ocr_model="model/ocr")

        # No mocking needed - just verify the method doesn't raise
        engine = DeepDocPdfEngine()
        engine._maybe_download_models(opts)  # Should return early without downloading

    def test_coerce_bbox_from_dict_x0_y0_x1_y1(self):
        """Test BBox conversion from dict with x0/y0/x1/y1 keys."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox({"x0": 10.5, "y0": 20.5, "x1": 100.5, "y1": 200.5})

        assert bbox is not None
        assert bbox.x0 == 10.5
        assert bbox.y0 == 20.5
        assert bbox.x1 == 100.5
        assert bbox.y1 == 200.5

    def test_coerce_bbox_from_dict_left_top_right_bottom(self):
        """Test BBox conversion from dict with left/top/right/bottom keys."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox({"left": 5, "top": 10, "right": 50, "bottom": 100})

        assert bbox is not None
        assert bbox.x0 == 5.0
        assert bbox.y0 == 10.0
        assert bbox.x1 == 50.0
        assert bbox.y1 == 100.0

    def test_coerce_bbox_from_list(self):
        """Test BBox conversion from list."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox([10, 20, 100, 200])

        assert bbox is not None
        assert bbox.x0 == 10.0
        assert bbox.y1 == 200.0

    def test_coerce_bbox_from_tuple(self):
        """Test BBox conversion from tuple."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox((10, 20, 100, 200))

        assert bbox is not None
        assert bbox.x0 == 10.0

    def test_coerce_bbox_none(self):
        """Test BBox conversion with None input."""
        engine = DeepDocPdfEngine()
        assert engine._coerce_bbox(None) is None

    def test_coerce_bbox_invalid_dict(self):
        """Test BBox conversion with incomplete dict."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox({"x0": 10, "y0": 20})  # Missing x1, y1
        assert bbox is None

    def test_iter_block_entries_from_dict(self):
        """Test iterating block entries from dict structure."""
        engine = DeepDocPdfEngine()
        raw = {"chunks": [{"text": "Block 1"}, {"text": "Block 2"}], "metadata": {"version": "1.0"}}

        entries = list(engine._iter_block_entries(raw))
        assert len(entries) == 2
        assert entries[0]["text"] == "Block 1"
        assert entries[1]["text"] == "Block 2"

    def test_iter_block_entries_from_list(self):
        """Test iterating block entries from list structure."""
        engine = DeepDocPdfEngine()
        raw = [{"text": "Block 1"}, {"text": "Block 2"}]

        entries = list(engine._iter_block_entries(raw))
        assert len(entries) == 2

    def test_iter_block_entries_none(self):
        """Test iterating with None input."""
        engine = DeepDocPdfEngine()
        entries = list(engine._iter_block_entries(None))
        assert entries == []

    def test_coerce_document_dict_structure(self):
        """Test document coercion from dict structure."""
        engine = DeepDocPdfEngine()
        backend_result = {
            "chunks": [
                {"text": "Block 1", "page": 1, "bbox": [10, 20, 100, 200], "label": "paragraph"},
                {"text": "Block 2", "page": 2, "confidence": 0.95},
            ],
            "tables": [{"page": 1, "html": "<table>...</table>", "text": "Table text", "bbox": [0, 0, 100, 100]}],
            "figures": [{"page": 1, "caption": "Figure 1", "image_path": "/tmp/fig1.png"}],
        }

        doc = engine._coerce_document(backend_result, PdfParseOptions())

        assert len(doc.blocks) == 2
        assert doc.blocks[0].text == "Block 1"
        assert doc.blocks[0].page == 1
        assert doc.blocks[0].label == "paragraph"
        assert doc.blocks[0].bbox is not None
        assert doc.blocks[1].confidence == 0.95

        assert len(doc.tables) == 1
        assert doc.tables[0].html == "<table>...</table>"
        assert doc.tables[0].text == "Table text"

        assert len(doc.figures) == 1
        assert doc.figures[0].caption == "Figure 1"
        assert doc.figures[0].image_ref == "/tmp/fig1.png"

        assert len(doc.pages) == 2
        assert doc.pages[0].number == 1
        assert doc.pages[0].text == "Block 1"
        assert doc.pages[1].number == 2

        assert doc.metadata["parser"] == "pdf_deepdoc"
        assert "chunks" in doc.metadata["backend_keys"]

    def test_coerce_document_list_structure(self):
        """Test document coercion from plain list structure."""
        engine = DeepDocPdfEngine()
        backend_result = [{"text": "Block 1", "page": 1}, {"text": "Block 2", "page": 1}]

        doc = engine._coerce_document(backend_result, PdfParseOptions())

        assert len(doc.blocks) == 2
        assert len(doc.pages) == 1
        assert doc.tables == []
        assert doc.figures == []

    def test_coerce_document_unknown_type(self):
        """Test document coercion with unsupported type."""
        engine = DeepDocPdfEngine()
        doc = engine._coerce_document("invalid", PdfParseOptions())

        assert doc.blocks == []
        assert doc.pages == []
        assert "raw_payload_type" in doc.metadata

    def test_run_backend_with_parse_method(self):
        """Test running backend that has a parse() method."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = {"chunks": [{"text": "Test", "page": 1}]}

        engine = DeepDocPdfEngine()
        result = engine._run_backend(lambda: mock_parser, "/tmp/test.pdf", PdfParseOptions(max_pages=10))

        mock_parser.parse.assert_called_once_with("/tmp/test.pdf", max_pages=10)

    def test_run_backend_callable(self):
        """Test running backend that is directly callable."""
        mock_parser = MockDeepDocParser()

        engine = DeepDocPdfEngine()
        result = engine._run_backend(lambda: mock_parser, "/tmp/test.pdf", PdfParseOptions(max_pages=5))

        assert mock_parser.called_with[0] == "/tmp/test.pdf"
        assert mock_parser.called_with[1] == {"max_pages": 5}

    def test_run_success(self):
        """Test successful PDF parsing with DeepDoc."""
        mock_parser_cls = MockDeepDocParser

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_parser_cls):
            engine = DeepDocPdfEngine()
            pdf_bytes = b"%PDF-1.4 dummy content"
            result = engine.run(io.BytesIO(pdf_bytes), PdfParseOptions())

        assert len(result.blocks) == 2
        assert result.blocks[0].text == "Sample text"
        assert result.blocks[0].page == 1
        assert len(result.tables) == 1
        assert len(result.figures) == 1
        assert result.metadata["backend"] == "MockDeepDocParser"

    def test_run_fallback_when_backend_not_available(self):
        """Test fallback to PlainPdfEngine when DeepDoc is not available."""
        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=None):
            mock_plain_engine = MagicMock()
            mock_plain_engine.run.return_value = MagicMock(metadata={})

            engine = DeepDocPdfEngine(plain_engine=mock_plain_engine)
            result = engine.run(io.BytesIO(b"dummy"), PdfParseOptions(fallback_to_plain=True))

        mock_plain_engine.run.assert_called_once()
        assert result.metadata["used_fallback"] is True
        assert "DeepDoc backend not available" in result.metadata["fallback_reason"]

    def test_run_raises_when_no_fallback(self):
        """Test that ImportError is raised when fallback is disabled."""
        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=None):
            engine = DeepDocPdfEngine()
            with pytest.raises(ImportError, match="DeepDoc backend is not available"):
                engine.run(io.BytesIO(b"dummy"), PdfParseOptions(fallback_to_plain=False))

    def test_run_fallback_on_parse_error(self):
        """Test fallback when DeepDoc parsing fails."""
        mock_parser_cls = MagicMock(side_effect=Exception("Parsing error"))

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_parser_cls):
            mock_plain_engine = MagicMock()
            mock_plain_engine.run.return_value = MagicMock(metadata={})

            engine = DeepDocPdfEngine(plain_engine=mock_plain_engine)
            result = engine.run(io.BytesIO(b"dummy"), PdfParseOptions(fallback_to_plain=True))

        mock_plain_engine.run.assert_called_once()
        assert result.metadata["used_fallback"] is True
        assert "Parsing error" in result.metadata["fallback_reason"]

    def test_run_temp_file_cleanup(self):
        """Test that temporary file is cleaned up after parsing."""
        mock_parser_cls = MockDeepDocParser

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_parser_cls):
            with patch("os.path.exists", return_value=True) as mock_exists:
                with patch("os.remove") as mock_remove:
                    engine = DeepDocPdfEngine()
                    engine.run(io.BytesIO(b"dummy"), PdfParseOptions())

                    # Verify temp file cleanup
                    mock_remove.assert_called_once()

    def test_run_temp_file_cleanup_on_error(self):
        """Test that temp file is cleaned up even when parsing fails."""
        mock_parser_cls = MagicMock(side_effect=Exception("Parse error"))

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_parser_cls):
            with patch("os.path.exists", return_value=True):
                with patch("os.remove") as mock_remove:
                    engine = DeepDocPdfEngine()
                    try:
                        engine.run(io.BytesIO(b"dummy"), PdfParseOptions(fallback_to_plain=False))
                    except Exception:
                        pass

                    mock_remove.assert_called_once()

    def test_file_seek_before_reading(self):
        """Test that file is seeked to beginning before reading."""
        file = io.BytesIO(b"dummy content")
        file.read()  # Move cursor to end

        mock_parser_cls = MockDeepDocParser

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_parser_cls):
            engine = DeepDocPdfEngine()
            engine.run(file, PdfParseOptions())

        # File should have been reset to beginning
        assert file.tell() == len(b"dummy content")  # After reading

    def test_content_type(self):
        """Test that content_type is set correctly."""
        engine = DeepDocPdfEngine()
        assert engine.content_type == "application/pdf"

    def test_metadata_includes_options(self):
        """Test that parsed document metadata includes parse options."""
        mock_parser_cls = MockDeepDocParser

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_parser_cls):
            engine = DeepDocPdfEngine()
            opts = PdfParseOptions(ocr=True, layout=False, tables=True, max_pages=5)
            result = engine.run(io.BytesIO(b"dummy"), opts)

        assert result.metadata["ocr"] is True
        assert result.metadata["layout"] is False
        assert result.metadata["tables"] is True
        assert result.metadata["max_pages"] == 5