"""Tests for base parser interfaces and dataclasses."""

import pytest

from llm_infrastructure.preprocessing.parsers.base import (
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
    PdfParseOptions,
)


class TestBoundingBox:
    def test_creation(self):
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
        assert bbox.x0 == 10.0
        assert bbox.y0 == 20.0
        assert bbox.x1 == 100.0
        assert bbox.y1 == 200.0

    def test_from_sequence_valid(self):
        bbox = BoundingBox.from_sequence([10, 20, 100, 200])
        assert bbox is not None
        assert bbox.x0 == 10.0
        assert bbox.y1 == 200.0

    def test_from_sequence_none(self):
        assert BoundingBox.from_sequence(None) is None

    def test_from_sequence_invalid_length(self):
        with pytest.raises(ValueError, match="must contain four coordinates"):
            BoundingBox.from_sequence([10, 20, 100])


class TestParsedPage:
    def test_creation_minimal(self):
        page = ParsedPage(number=1, text="Hello world")
        assert page.number == 1
        assert page.text == "Hello world"
        assert page.width is None
        assert page.height is None
        assert page.metadata == {}

    def test_creation_with_dimensions(self):
        page = ParsedPage(number=2, text="Page 2", width=612.0, height=792.0)
        assert page.number == 2
        assert page.width == 612.0
        assert page.height == 792.0

    def test_metadata(self):
        page = ParsedPage(number=1, text="", metadata={"custom": "data"})
        assert page.metadata["custom"] == "data"


class TestParsedBlock:
    def test_creation_minimal(self):
        block = ParsedBlock(text="Sample text", page=1)
        assert block.text == "Sample text"
        assert block.page == 1
        assert block.bbox is None
        assert block.label == "text"
        assert block.confidence is None

    def test_creation_with_bbox(self):
        bbox = BoundingBox(10, 20, 100, 200)
        block = ParsedBlock(text="Sample", page=1, bbox=bbox, label="title", confidence=0.95)
        assert block.bbox == bbox
        assert block.label == "title"
        assert block.confidence == 0.95

    def test_metadata(self):
        block = ParsedBlock(text="Text", page=1, metadata={"source": "ocr"})
        assert block.metadata["source"] == "ocr"


class TestParsedTable:
    def test_creation(self):
        bbox = BoundingBox(0, 0, 100, 100)
        table = ParsedTable(page=1, bbox=bbox, html="<table></table>", text="Header1|Header2", image_ref="/tmp/table.png")
        assert table.page == 1
        assert table.bbox == bbox
        assert table.html == "<table></table>"
        assert table.text == "Header1|Header2"
        assert table.image_ref == "/tmp/table.png"

    def test_minimal(self):
        table = ParsedTable(page=1)
        assert table.page == 1
        assert table.bbox is None
        assert table.html is None


class TestParsedFigure:
    def test_creation(self):
        bbox = BoundingBox(0, 0, 200, 200)
        figure = ParsedFigure(page=1, bbox=bbox, caption="Figure 1: Example", image_ref="/tmp/fig.png")
        assert figure.page == 1
        assert figure.caption == "Figure 1: Example"
        assert figure.image_ref == "/tmp/fig.png"


class TestParsedDocument:
    def test_creation_empty(self):
        doc = ParsedDocument()
        assert doc.pages == []
        assert doc.blocks == []
        assert doc.tables == []
        assert doc.figures == []
        assert doc.metadata == {}
        assert doc.errors == []
        assert doc.content_type == "application/pdf"

    def test_creation_with_content(self):
        page = ParsedPage(number=1, text="Page 1")
        block = ParsedBlock(text="Block 1", page=1)
        doc = ParsedDocument(pages=[page], blocks=[block])
        assert len(doc.pages) == 1
        assert len(doc.blocks) == 1

    def test_merged_text_default_separator(self):
        doc = ParsedDocument(
            blocks=[
                ParsedBlock(text="First paragraph.", page=1),
                ParsedBlock(text="Second paragraph.", page=1),
                ParsedBlock(text="Third paragraph.", page=2),
            ]
        )
        merged = doc.merged_text()
        assert merged == "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    def test_merged_text_custom_separator(self):
        doc = ParsedDocument(blocks=[ParsedBlock(text="Line 1", page=1), ParsedBlock(text="Line 2", page=1)])
        merged = doc.merged_text(separator=" | ")
        assert merged == "Line 1 | Line 2"

    def test_merged_text_empty_blocks(self):
        doc = ParsedDocument(blocks=[ParsedBlock(text="", page=1), ParsedBlock(text="Text", page=1), ParsedBlock(text="", page=1)])
        merged = doc.merged_text()
        assert merged == "Text"

    def test_metadata(self):
        doc = ParsedDocument(metadata={"parser": "test", "version": "1.0"})
        assert doc.metadata["parser"] == "test"
        assert doc.metadata["version"] == "1.0"

    def test_errors(self):
        doc = ParsedDocument(errors=["Error 1", "Error 2"])
        assert len(doc.errors) == 2
        assert "Error 1" in doc.errors


class TestPdfParseOptions:
    def test_defaults(self):
        opts = PdfParseOptions()
        assert opts.ocr is True
        assert opts.layout is True
        assert opts.tables is True
        assert opts.merge is True
        assert opts.scrap_filter is True
        assert opts.model_root is None
        assert opts.device == "cpu"
        assert opts.max_pages is None
        assert opts.fallback_to_plain is True
        assert opts.preserve_layout is False
        assert opts.ocr_model is None
        assert opts.layout_model is None
        assert opts.tsr_model is None
        assert opts.allow_download is True
        assert opts.hf_endpoint is None

    def test_custom_values(self):
        from pathlib import Path

        opts = PdfParseOptions(
            ocr=False,
            layout=False,
            tables=False,
            device="cuda",
            max_pages=10,
            model_root=Path("/tmp/models"),
            ocr_model="yolov8",
            layout_model="layoutlm",
            tsr_model="table-transformer",
            hf_endpoint="https://hf-mirror.com",
        )
        assert opts.ocr is False
        assert opts.layout is False
        assert opts.device == "cuda"
        assert opts.max_pages == 10
        assert opts.model_root == Path("/tmp/models")
        assert opts.ocr_model == "yolov8"
        assert opts.hf_endpoint == "https://hf-mirror.com"