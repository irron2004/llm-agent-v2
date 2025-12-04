from __future__ import annotations

import io
import types

import pytest

from llm_infrastructure.preprocessing.base import PdfParseOptions
from llm_infrastructure.preprocessing.registry import get_parser, list_parsers
from llm_infrastructure.preprocessing.parsers import deepdoc_pdf, pdf_plain
from services.ingest.document_service import DocumentIngestService


def _patch_pdfplumber(monkeypatch: pytest.MonkeyPatch, page_texts: list[str]) -> None:
    class DummyPage:
        def __init__(self, text: str) -> None:
            self._text = text
            self.width = 100
            self.height = 200

        def extract_text(self) -> str:
            return self._text

    class DummyPdf:
        def __init__(self) -> None:
            self.pages = [DummyPage(text) for text in page_texts]

        def __enter__(self) -> "DummyPdf":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(pdf_plain, "pdfplumber", types.SimpleNamespace(open=lambda _file: DummyPdf()))


def test_registry_has_builtin_parsers() -> None:
    available = set(list_parsers())
    assert "pdf_plain" in available
    assert "pdf_deepdoc" in available
    assert isinstance(get_parser("pdf_plain"), pdf_plain.PlainPdfParser)


def test_plain_parser_uses_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pdfplumber(monkeypatch, ["Hello world", "Second page"])
    parser = pdf_plain.PlainPdfParser()
    doc = parser.parse(io.BytesIO(b"demo"), options=PdfParseOptions(max_pages=None))
    assert [page.text for page in doc.pages] == ["Hello world", "Second page"]
    assert doc.blocks[0].page == 1
    assert doc.metadata["parser"] == "pdf_plain"


def test_deepdoc_fallbacks_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pdfplumber(monkeypatch, ["1. Intro", "2. Next section"])
    monkeypatch.setattr(deepdoc_pdf.DeepDocPdfParser, "_load_backend_class", lambda self: None)
    parser = deepdoc_pdf.DeepDocPdfParser()
    doc = parser.parse(io.BytesIO(b"demo"), options=PdfParseOptions(fallback_to_plain=True))
    assert doc.metadata.get("used_fallback") is True
    assert doc.metadata.get("parser") == "pdf_deepdoc"
    assert len(doc.blocks) == 2


def test_document_ingest_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pdfplumber(monkeypatch, ["1. Intro\nLine A", "2. Follow up\nLine B"])
    service = DocumentIngestService(parser_id="pdf_plain")
    result = service.ingest_pdf(io.BytesIO(b"demo"), doc_type="sop")
    sections = result["sections"]
    assert len(sections) == 2
    assert sections[0]["title"].startswith("1.")
    assert "Line A" in sections[0]["text"]
    assert sections[1]["title"].startswith("2.")
