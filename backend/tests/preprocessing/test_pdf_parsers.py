from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "httpx" not in sys.modules:
    class _DummyHttpxClient:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

    sys.modules["httpx"] = types.SimpleNamespace(Client=_DummyHttpxClient)

from backend.llm_infrastructure.preprocessing.parsers import PdfParseOptions  # noqa: E402
from backend.llm_infrastructure.preprocessing.parsers.adapters import pdf_deepdoc, pdf_plain, pdf_vlm  # noqa: E402
from backend.llm_infrastructure.preprocessing.parsers.engines import (  # noqa: E402
    pdf_plain_engine,
)
from backend.llm_infrastructure.preprocessing.parsers.registry import get_parser, list_parsers  # noqa: E402
from backend.services.ingest.document_ingest_service import DocumentIngestService  # noqa: E402


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

    monkeypatch.setattr(
        pdf_plain_engine, "pdfplumber", types.SimpleNamespace(open=lambda _file: DummyPdf())
    )


def test_registry_has_builtin_parsers() -> None:
    available = set(list_parsers())
    assert "pdf_plain" in available
    assert "pdf_deepdoc" in available
    assert "pdf_deepseek_vl" in available
    assert "pdf_vlm" in available
    assert isinstance(get_parser("pdf_plain"), pdf_plain.PlainPdfAdapter)


def test_plain_parser_uses_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pdfplumber(monkeypatch, ["Hello world", "Second page"])
    parser = pdf_plain.PlainPdfAdapter()
    doc = parser.parse(io.BytesIO(b"demo"), options=PdfParseOptions(max_pages=None))
    assert [page.text for page in doc.pages] == ["Hello world", "Second page"]
    assert doc.blocks[0].page == 1
    assert doc.metadata["parser"] == "pdf_plain"


def test_deepdoc_fallbacks_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pdfplumber(monkeypatch, ["1. Intro", "2. Next section"])
    monkeypatch.setattr(pdf_deepdoc.DeepDocPdfEngine, "_load_backend_class", lambda self: None)
    parser = pdf_deepdoc.DeepDocPdfAdapter()
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


def test_deepseek_vl_parser_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    images = [b"page1", b"page2"]

    def renderer(_pdf_bytes: bytes) -> list[bytes]:
        return images

    class DummyVLM:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def generate(self, image: bytes, prompt: str, **kwargs: object) -> str:
            self.calls.append((image, prompt, kwargs))
            return f"text-{len(self.calls)}"

    parser = pdf_vlm.VlmPdfAdapter(
        vlm_client=DummyVLM(),
        renderer=renderer,
    )
    doc = parser.parse(
        io.BytesIO(b"demo"),
        options=PdfParseOptions(vlm_prompt="prompt", vlm_max_new_tokens=128),
    )
    assert [page.text for page in doc.pages] == ["text-1", "text-2"]
    assert doc.blocks[0].page == 1
    assert doc.metadata["parser"] in {"pdf_deepseek_vl", "pdf_vlm"}


def test_document_ingest_service_for_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    images = [b"page1", b"page2"]

    def renderer(_pdf_bytes: bytes) -> list[bytes]:
        return images

    class DummyVLM:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def generate(self, image: bytes, prompt: str, **kwargs: object) -> str:
            self.calls.append((image, prompt, kwargs))
            return f"text-{len(self.calls)}"

    service = DocumentIngestService.for_vlm(vlm_client=DummyVLM(), renderer=renderer)
    result = service.ingest_pdf(io.BytesIO(b"demo"), doc_type="generic")
    parsed = result["parsed"]
    assert [page.text for page in parsed.pages] == ["text-1", "text-2"]
    assert parsed.metadata["parser"] in {"pdf_deepseek_vl", "pdf_vlm"}
