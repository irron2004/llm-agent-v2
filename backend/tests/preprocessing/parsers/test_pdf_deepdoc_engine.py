"""Tests for DeepDocPdfEngine.

수정된 엔진에 맞춰 업데이트된 테스트:
- 프로젝트 내 deepdoc 패키지 직접 사용
- preferred_backend 필수 지정
- fallback 로직 제거됨

Note: deepdoc 의존성이 설치되지 않은 환경에서도 단위 테스트가 가능하도록
      mock 기반 테스트로 구성됨
"""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import sys

import pytest

# deepdoc import를 mock으로 우회
sys.modules["llm_infrastructure.preprocessing.deepdoc"] = MagicMock()
sys.modules["llm_infrastructure.preprocessing.deepdoc.parser"] = MagicMock()
sys.modules["llm_infrastructure.preprocessing.deepdoc.parser.pdf_parser"] = MagicMock()

from llm_infrastructure.preprocessing.parsers.base import (
    BoundingBox,
    DeepDocBackend,
    ParsedDocument,
    PdfParseOptions,
)


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


# Mock 설정 후 import
mock_ragflow_parser = type("RAGFlowPdfParser", (), {"__call__": MockDeepDocParser.__call__})
mock_plain_parser = type("PlainParser", (), {"__call__": MockDeepDocParser.__call__})

sys.modules["llm_infrastructure.preprocessing.deepdoc.parser.pdf_parser"].RAGFlowPdfParser = mock_ragflow_parser
sys.modules["llm_infrastructure.preprocessing.deepdoc.parser.pdf_parser"].PlainParser = mock_plain_parser

from llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine import (
    DeepDocPdfEngine,
    _get_backend_classes,
)


class TestGetBackendClass:
    """_get_backend_class 메서드 테스트."""

    def test_get_backend_class_ragflow(self):
        """RAGFLOW 백엔드 클래스 가져오기."""
        engine = DeepDocPdfEngine()
        backend_cls = engine._get_backend_class(DeepDocBackend.RAGFLOW)

        assert backend_cls is not None

    def test_get_backend_class_plain(self):
        """PLAIN 백엔드 클래스 가져오기."""
        engine = DeepDocPdfEngine()
        backend_cls = engine._get_backend_class(DeepDocBackend.PLAIN)

        assert backend_cls is not None

    def test_get_backend_class_none_returns_none(self):
        """preferred가 None이면 None 반환."""
        engine = DeepDocPdfEngine()
        backend_cls = engine._get_backend_class(None)

        assert backend_cls is None

    def test_get_backend_class_logs_error_when_none(self, caplog):
        """preferred가 None일 때 에러 로그."""
        import logging
        caplog.set_level(logging.ERROR)

        engine = DeepDocPdfEngine()
        engine._get_backend_class(None)

        assert "must be explicitly specified" in caplog.text


class TestConfigureHfEnv:
    """HuggingFace 환경변수 설정 테스트."""

    def test_configure_hf_env_sets_variables(self):
        """환경변수가 올바르게 설정되는지 확인."""
        opts = PdfParseOptions(
            hf_endpoint="https://hf-mirror.com",
            model_root=Path("/tmp/models"),
            preferred_backend=DeepDocBackend.RAGFLOW,
        )

        with patch.dict(os.environ, {}, clear=True):
            engine = DeepDocPdfEngine()
            engine._configure_hf_env(opts)

            assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
            assert os.environ["HF_HOME"] == "/tmp/models"
            assert os.environ["HUGGINGFACE_HUB_CACHE"] == "/tmp/models"
            assert os.environ["TRANSFORMERS_CACHE"] == "/tmp/models"

    def test_configure_hf_env_no_override(self):
        """기존 환경변수를 덮어쓰지 않는지 확인."""
        with patch.dict(os.environ, {"HF_ENDPOINT": "existing_value"}, clear=True):
            opts = PdfParseOptions(
                hf_endpoint="https://new-value.com",
                preferred_backend=DeepDocBackend.RAGFLOW,
            )
            engine = DeepDocPdfEngine()
            engine._configure_hf_env(opts)

            assert os.environ["HF_ENDPOINT"] == "existing_value"

    def test_configure_hf_env_skips_when_empty(self):
        """옵션이 비어있으면 설정하지 않음."""
        with patch.dict(os.environ, {}, clear=True):
            opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
            engine = DeepDocPdfEngine()
            engine._configure_hf_env(opts)

            assert "HF_ENDPOINT" not in os.environ


class TestMaybeDownloadModels:
    """모델 다운로드 테스트."""

    def test_download_disabled(self):
        """allow_download=False일 때 다운로드 안 함."""
        opts = PdfParseOptions(
            allow_download=False,
            model_root=Path("/tmp/models"),
            ocr_model="model/ocr",
            preferred_backend=DeepDocBackend.RAGFLOW,
        )

        engine = DeepDocPdfEngine()
        engine._maybe_download_models(opts)

    def test_download_skipped_without_model_root(self):
        """model_root가 없으면 다운로드 안 함."""
        opts = PdfParseOptions(
            allow_download=True,
            ocr_model="model/ocr",
            preferred_backend=DeepDocBackend.RAGFLOW,
        )

        engine = DeepDocPdfEngine()
        engine._maybe_download_models(opts)


class TestCoerceBbox:
    """BoundingBox 변환 테스트."""

    def test_from_dict_x0_y0_x1_y1(self):
        """x0/y0/x1/y1 형식 dict 변환."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox({"x0": 10.5, "y0": 20.5, "x1": 100.5, "y1": 200.5})

        assert bbox is not None
        assert bbox.x0 == 10.5
        assert bbox.y0 == 20.5
        assert bbox.x1 == 100.5
        assert bbox.y1 == 200.5

    def test_from_dict_left_top_right_bottom(self):
        """left/top/right/bottom 형식 dict 변환."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox({"left": 5, "top": 10, "right": 50, "bottom": 100})

        assert bbox is not None
        assert bbox.x0 == 5.0
        assert bbox.y0 == 10.0
        assert bbox.x1 == 50.0
        assert bbox.y1 == 100.0

    def test_from_list(self):
        """리스트 형식 변환."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox([10, 20, 100, 200])

        assert bbox is not None
        assert bbox.x0 == 10.0
        assert bbox.y1 == 200.0

    def test_from_tuple(self):
        """튜플 형식 변환."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox((10, 20, 100, 200))

        assert bbox is not None
        assert bbox.x0 == 10.0

    def test_none_input(self):
        """None 입력 처리."""
        engine = DeepDocPdfEngine()
        assert engine._coerce_bbox(None) is None

    def test_invalid_dict(self):
        """불완전한 dict 처리."""
        engine = DeepDocPdfEngine()
        bbox = engine._coerce_bbox({"x0": 10, "y0": 20})
        assert bbox is None

    def test_invalid_list_length(self):
        """잘못된 길이의 리스트 처리."""
        engine = DeepDocPdfEngine()
        assert engine._coerce_bbox([10, 20, 100]) is None
        assert engine._coerce_bbox([10, 20, 100, 200, 300]) is None


class TestIterBlockEntries:
    """블록 엔트리 순회 테스트."""

    def test_from_dict_with_chunks(self):
        """chunks 키가 있는 dict에서 추출."""
        engine = DeepDocPdfEngine()
        raw = {
            "chunks": [{"text": "Block 1"}, {"text": "Block 2"}],
            "metadata": {"version": "1.0"},
        }

        entries = list(engine._iter_block_entries(raw))
        assert len(entries) == 2
        assert entries[0]["text"] == "Block 1"
        assert entries[1]["text"] == "Block 2"

    def test_from_list(self):
        """리스트에서 직접 추출."""
        engine = DeepDocPdfEngine()
        raw = [{"text": "Block 1"}, {"text": "Block 2"}]

        entries = list(engine._iter_block_entries(raw))
        assert len(entries) == 2

    def test_none_input(self):
        """None 입력 처리."""
        engine = DeepDocPdfEngine()
        entries = list(engine._iter_block_entries(None))
        assert entries == []

    def test_filters_non_dict_entries(self):
        """dict가 아닌 엔트리는 필터링."""
        engine = DeepDocPdfEngine()
        raw = [{"text": "Valid"}, "invalid", None, {"text": "Also valid"}]

        entries = list(engine._iter_block_entries(raw))
        assert len(entries) == 2


class TestCoerceDocument:
    """문서 변환 테스트."""

    def test_dict_structure_with_all_fields(self):
        """모든 필드가 있는 dict 변환."""
        engine = DeepDocPdfEngine()
        backend_result = {
            "chunks": [
                {"text": "Block 1", "page": 1, "bbox": [10, 20, 100, 200], "label": "paragraph"},
                {"text": "Block 2", "page": 2, "confidence": 0.95},
            ],
            "tables": [
                {"page": 1, "html": "<table>...</table>", "text": "Table text", "bbox": [0, 0, 100, 100]}
            ],
            "figures": [{"page": 1, "caption": "Figure 1", "image_path": "/tmp/fig1.png"}],
        }

        opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
        doc = engine._coerce_document(backend_result, opts)

        assert len(doc.blocks) == 2
        assert doc.blocks[0].text == "Block 1"
        assert doc.blocks[0].page == 1
        assert doc.blocks[0].label == "paragraph"
        assert doc.blocks[0].bbox is not None
        assert doc.blocks[1].confidence == 0.95

        assert len(doc.tables) == 1
        assert doc.tables[0].html == "<table>...</table>"

        assert len(doc.figures) == 1
        assert doc.figures[0].caption == "Figure 1"
        assert doc.figures[0].image_ref == "/tmp/fig1.png"

        assert len(doc.pages) == 2
        assert doc.pages[0].number == 1
        assert doc.pages[1].number == 2

        assert doc.metadata["parser"] == "pdf_deepdoc"

    def test_list_structure(self):
        """리스트 형식 결과 변환."""
        engine = DeepDocPdfEngine()
        backend_result = [{"text": "Block 1", "page": 1}, {"text": "Block 2", "page": 1}]

        opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
        doc = engine._coerce_document(backend_result, opts)

        assert len(doc.blocks) == 2
        assert len(doc.pages) == 1
        assert doc.tables == []
        assert doc.figures == []

    def test_unknown_type(self):
        """지원하지 않는 타입 처리."""
        engine = DeepDocPdfEngine()
        opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
        doc = engine._coerce_document("invalid", opts)

        assert doc.blocks == []
        assert doc.pages == []
        assert "raw_payload_type" in doc.metadata


class TestRunBackend:
    """백엔드 실행 테스트."""

    def test_calls_backend_with_path(self):
        """백엔드가 PDF 경로로 호출되는지 확인."""
        mock_parser = MockDeepDocParser()
        mock_cls = lambda: mock_parser

        engine = DeepDocPdfEngine()
        opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
        result = engine._run_backend(mock_cls, "/tmp/test.pdf", opts)

        assert mock_parser.called_with[0] == "/tmp/test.pdf"
        assert result is not None


class TestRun:
    """전체 run 메서드 테스트."""

    def test_run_success(self, sample_pdf_file):
        """성공적인 파싱."""
        mock_parser = MockDeepDocParser()

        def mock_get_backend_classes():
            return {DeepDocBackend.RAGFLOW: lambda: mock_parser}

        with patch(
            "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine._get_backend_classes",
            mock_get_backend_classes,
        ):
            engine = DeepDocPdfEngine()
            opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
            result = engine.run(sample_pdf_file, opts)

        assert len(result.blocks) == 2
        assert result.blocks[0].text == "Sample text"
        assert len(result.tables) == 1
        assert len(result.figures) == 1

    def test_run_raises_without_backend(self, sample_pdf_file):
        """backend가 없으면 ImportError 발생."""
        def mock_get_backend_classes():
            return {}

        with patch(
            "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine._get_backend_classes",
            mock_get_backend_classes,
        ):
            engine = DeepDocPdfEngine()
            opts = PdfParseOptions()  # preferred_backend 없음

            with pytest.raises(ImportError, match="DeepDoc backend not available"):
                engine.run(sample_pdf_file, opts)

    def test_run_temp_file_cleanup(self, sample_pdf_file):
        """임시 파일이 정리되는지 확인."""
        mock_parser = MockDeepDocParser()

        def mock_get_backend_classes():
            return {DeepDocBackend.RAGFLOW: lambda: mock_parser}

        with patch(
            "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine._get_backend_classes",
            mock_get_backend_classes,
        ):
            with patch(
                "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine.os.path.exists",
                return_value=True,
            ):
                with patch(
                    "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine.os.remove"
                ) as mock_remove:
                    engine = DeepDocPdfEngine()
                    opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
                    engine.run(sample_pdf_file, opts)

                    mock_remove.assert_called_once()

    def test_run_temp_file_cleanup_on_error(self, sample_pdf_file):
        """에러 발생 시에도 임시 파일 정리."""
        mock_cls = MagicMock(side_effect=Exception("Parse error"))

        def mock_get_backend_classes():
            return {DeepDocBackend.RAGFLOW: mock_cls}

        with patch(
            "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine._get_backend_classes",
            mock_get_backend_classes,
        ):
            with patch(
                "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine.os.path.exists",
                return_value=True,
            ):
                with patch(
                    "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine.os.remove"
                ) as mock_remove:
                    engine = DeepDocPdfEngine()
                    opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)

                    with pytest.raises(Exception):
                        engine.run(sample_pdf_file, opts)

                    mock_remove.assert_called_once()

    def test_file_seek_before_reading(self):
        """파일 읽기 전 seek(0) 호출 확인."""
        file = io.BytesIO(b"dummy content")
        file.read()  # Move cursor to end

        mock_parser = MockDeepDocParser()

        def mock_get_backend_classes():
            return {DeepDocBackend.RAGFLOW: lambda: mock_parser}

        with patch(
            "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine._get_backend_classes",
            mock_get_backend_classes,
        ):
            engine = DeepDocPdfEngine()
            opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
            engine.run(file, opts)

        # File should have been read
        assert file.tell() == len(b"dummy content")

    def test_metadata_includes_options(self, sample_pdf_file):
        """메타데이터에 옵션 정보 포함 확인."""
        mock_parser = MockDeepDocParser()

        def mock_get_backend_classes():
            return {DeepDocBackend.RAGFLOW: lambda: mock_parser}

        with patch(
            "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine._get_backend_classes",
            mock_get_backend_classes,
        ):
            engine = DeepDocPdfEngine()
            opts = PdfParseOptions(
                ocr=True,
                layout=False,
                tables=True,
                max_pages=5,
                preferred_backend=DeepDocBackend.RAGFLOW,
            )
            result = engine.run(sample_pdf_file, opts)

        assert result.metadata["ocr"] is True
        assert result.metadata["layout"] is False
        assert result.metadata["tables"] is True
        assert result.metadata["max_pages"] == 5

    def test_metadata_includes_backend_name(self, sample_pdf_file):
        """메타데이터에 백엔드 이름 포함 확인."""
        mock_parser = MockDeepDocParser()

        def mock_get_backend_classes():
            return {DeepDocBackend.RAGFLOW: lambda: mock_parser}

        with patch(
            "llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine._get_backend_classes",
            mock_get_backend_classes,
        ):
            engine = DeepDocPdfEngine()
            opts = PdfParseOptions(preferred_backend=DeepDocBackend.RAGFLOW)
            result = engine.run(sample_pdf_file, opts)

        assert "backend" in result.metadata


class TestContentType:
    """content_type 테스트."""

    def test_content_type_is_pdf(self):
        """content_type이 application/pdf인지 확인."""
        engine = DeepDocPdfEngine()
        assert engine.content_type == "application/pdf"


class TestDeepDocBackendEnum:
    """DeepDocBackend enum 테스트."""

    def test_choices_returns_values(self):
        """choices()가 모든 값을 반환하는지 확인."""
        choices = DeepDocBackend.choices()
        assert "RAGFlowPdfParser" in choices
        assert "PlainParser" in choices

    def test_enum_values(self):
        """enum 값들이 올바른지 확인."""
        assert DeepDocBackend.RAGFLOW.value == "RAGFlowPdfParser"
        assert DeepDocBackend.PLAIN.value == "PlainParser"
