"""DeepDoc PDF 파싱 엔진.

프로젝트 내 deepdoc 패키지의 PDF 파서를 사용하여 통일된 인터페이스를 제공합니다.
OCR, 레이아웃 분석, 테이블 구조 인식 등의 기능을 지원합니다.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional

from ..base import (
    BoundingBox,
    DeepDocBackend,
    ParsedBlock,
    ParsedDocument,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
    PdfParseOptions,
)

logger = logging.getLogger(__name__)


def _get_backend_classes() -> dict[DeepDocBackend, type]:
    """백엔드 클래스 매핑을 반환합니다 (lazy import).

    Returns:
        DeepDocBackend enum을 클래스로 매핑하는 dict
    """
    from ...deepdoc.parser.pdf_parser import RAGFlowPdfParser, PlainParser

    return {
        DeepDocBackend.RAGFLOW: RAGFlowPdfParser,
        DeepDocBackend.PLAIN: PlainParser,
    }


class DeepDocPdfEngine:
    """DeepDoc 기반 PDF 파싱 엔진.

    프로젝트 내 deepdoc 패키지를 사용하여 PDF 문서를 파싱합니다.
    OCR, 레이아웃 분석, 테이블/이미지 추출 기능을 제공합니다.

    Attributes:
        content_type: 처리 가능한 MIME 타입
    """

    content_type: str = "application/pdf"

    def __init__(self) -> None:
        """엔진 인스턴스를 초기화합니다."""
        pass

    def _get_backend_class(self, preferred: Optional[DeepDocBackend] = None) -> Optional[type]:
        """지정된 DeepDoc 백엔드 클래스를 반환합니다.

        Args:
            preferred: 사용할 백엔드 (DeepDocBackend enum). 필수 지정.

        Returns:
            백엔드 클래스. 찾지 못하면 None.
        """
        if not preferred:
            logger.error(
                f"DeepDoc backend must be explicitly specified. "
                f"Available options: {DeepDocBackend.choices()}"
            )
            return None

        backend_cls = _get_backend_classes().get(preferred)
        if backend_cls:
            logger.info(f"Using DeepDoc backend: {preferred.value}")
            return backend_cls

        logger.warning(
            f"Backend '{preferred.value}' not found. "
            f"Available options: {DeepDocBackend.choices()}"
        )
        return None

    def _configure_hf_env(self, opts: PdfParseOptions) -> None:
        """HuggingFace 관련 환경변수를 설정합니다.

        모델 다운로드 및 캐시 경로를 위한 환경변수를 구성합니다.
        HF_ENDPOINT, HF_HOME, HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE를 설정합니다.

        Args:
            opts: HF 엔드포인트 및 모델 루트 경로를 포함한 파싱 옵션
        """
        if opts.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", opts.hf_endpoint)
        if opts.model_root:
            root = str(Path(opts.model_root))
            os.environ.setdefault("HF_HOME", root)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", root)
            os.environ.setdefault("TRANSFORMERS_CACHE", root)
            Path(root).mkdir(parents=True, exist_ok=True)

    def _maybe_download_models(self, opts: PdfParseOptions) -> None:
        """필요한 모델을 HuggingFace에서 다운로드합니다.

        OCR, 레이아웃, 테이블 구조 인식(TSR) 모델을 다운로드합니다.
        이미 로컬에 존재하는 모델은 건너뜁니다.

        Args:
            opts: 모델 정보 및 다운로드 설정을 포함한 파싱 옵션

        Note:
            allow_download=False이거나 model_root가 없으면 아무 작업도 하지 않습니다.
            다운로드 실패는 예외를 발생시키지 않고 건너뜁니다.
        """
        if not opts.allow_download or not opts.model_root:
            return
        repos = [opts.ocr_model, opts.layout_model, opts.tsr_model]
        repos = [repo for repo in repos if repo]
        if not repos:
            return
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            return

        root = Path(opts.model_root)
        for repo in repos:
            target = root / repo.replace("/", "_")
            if target.exists():
                continue
            try:
                snapshot_download(
                    repo_id=repo,
                    local_dir=target,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            except Exception:
                continue

    def _coerce_bbox(self, payload: Any) -> Optional[BoundingBox]:
        """다양한 형식의 바운딩 박스를 BoundingBox 객체로 변환합니다.

        지원하는 형식:
        - dict: {"x0", "y0", "x1", "y1"} 또는 {"left", "top", "right", "bottom"}
        - list/tuple: [x0, y0, x1, y1]

        Args:
            payload: 바운딩 박스 데이터 (dict, list, tuple, 또는 None)

        Returns:
            BoundingBox 객체. 변환 불가능하면 None.
        """
        if payload is None:
            return None
        if isinstance(payload, dict):
            coords = [payload.get(key) for key in ("x0", "y0", "x1", "y1")]
            if any(coord is None for coord in coords):
                coords = [payload.get(key) for key in ("left", "top", "right", "bottom")]
            if all(coord is not None for coord in coords):
                return BoundingBox.from_sequence(coords)  # type: ignore[arg-type]
            return None
        if isinstance(payload, (list, tuple)) and len(payload) == 4:
            return BoundingBox.from_sequence(payload)  # type: ignore[arg-type]
        return None

    def _iter_block_entries(self, raw: Any) -> Iterable[dict]:
        """백엔드 결과에서 블록 엔트리들을 추출하여 순회합니다.

        DeepDoc 출력 형식의 다양한 구조를 처리합니다:
        - dict: {"pages": [...], "blocks": [...]} 형태에서 리스트 값들을 추출
        - list: 직접 엔트리들을 순회

        Args:
            raw: 백엔드에서 반환된 원시 데이터

        Yields:
            각 블록을 나타내는 dict 객체
        """
        if raw is None:
            return
        if isinstance(raw, dict):
            for value in raw.values():
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict):
                            yield entry
            return
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict):
                    yield entry
        return

    def _coerce_document(self, backend_result: Any, opts: PdfParseOptions) -> ParsedDocument:
        """백엔드 결과를 ParsedDocument 객체로 변환합니다.

        DeepDoc 백엔드의 다양한 출력 형식을 표준 ParsedDocument 구조로 정규화합니다.
        블록, 테이블, 이미지 정보를 추출하고 페이지별로 그룹화합니다.

        Args:
            backend_result: 백엔드 파서에서 반환된 원시 결과
            opts: 메타데이터에 포함할 파싱 옵션

        Returns:
            정규화된 ParsedDocument 객체
        """
        metadata: Dict[str, Any] = {"parser": "pdf_deepdoc"}
        raw_tables: List[Any] = []
        raw_figures: List[Any] = []
        raw_blocks: Any = []

        if isinstance(backend_result, dict):
            metadata["backend_keys"] = sorted(backend_result.keys())
            raw_blocks = (
                backend_result.get("chunks")
                or backend_result.get("blocks")
                or backend_result.get("pages")
                or backend_result.get("text")
                or []
            )
            raw_tables = backend_result.get("tables") or []
            raw_figures = backend_result.get("figures") or backend_result.get("images") or []
        elif isinstance(backend_result, list):
            raw_blocks = backend_result
        else:
            metadata["raw_payload_type"] = type(backend_result).__name__
            return ParsedDocument(metadata=metadata, content_type=self.content_type)

        blocks: List[ParsedBlock] = []
        pages_map: Dict[int, List[str]] = {}

        for entry in self._iter_block_entries(raw_blocks):
            text = entry.get("text") or entry.get("content") or ""
            page_no = int(entry.get("page", entry.get("page_num", 1)) or 1)
            bbox = self._coerce_bbox(entry.get("bbox") or entry.get("position") or entry.get("box"))
            label = entry.get("label") or entry.get("type") or "text"
            confidence = entry.get("confidence") or entry.get("score")
            blocks.append(
                ParsedBlock(
                    text=str(text),
                    page=page_no,
                    bbox=bbox,
                    label=str(label),
                    confidence=float(confidence) if confidence is not None else None,
                    metadata={
                        k: v
                        for k, v in entry.items()
                        if k
                        not in {
                            "text",
                            "content",
                            "page",
                            "page_num",
                            "bbox",
                            "position",
                            "box",
                            "type",
                            "label",
                            "confidence",
                            "score",
                        }
                    },
                )
            )
            pages_map.setdefault(page_no, []).append(str(text))

        pages: List[ParsedPage] = []
        for page_no in sorted(pages_map.keys()):
            pages.append(ParsedPage(number=page_no, text="\n".join(pages_map[page_no])))

        tables: List[ParsedTable] = []
        for entry in raw_tables or []:
            if not isinstance(entry, dict):
                continue
            tables.append(
                ParsedTable(
                    page=int(entry.get("page", 1)),
                    bbox=self._coerce_bbox(entry.get("bbox") or entry.get("position") or entry.get("box")),
                    html=entry.get("html") or entry.get("content_html"),
                    text=entry.get("text") or entry.get("content"),
                    image_ref=entry.get("image") or entry.get("image_path"),
                    metadata={
                        k: v
                        for k, v in entry.items()
                        if k
                        not in {
                            "page",
                            "bbox",
                            "position",
                            "box",
                            "html",
                            "content_html",
                            "text",
                            "content",
                            "image",
                            "image_path",
                        }
                    },
                )
            )

        figures: List[ParsedFigure] = []
        for entry in raw_figures or []:
            if not isinstance(entry, dict):
                continue
            figures.append(
                ParsedFigure(
                    page=int(entry.get("page", 1)),
                    bbox=self._coerce_bbox(entry.get("bbox") or entry.get("position") or entry.get("box")),
                    caption=entry.get("caption") or entry.get("text"),
                    image_ref=entry.get("image") or entry.get("image_path"),
                    metadata={
                        k: v
                        for k, v in entry.items()
                        if k
                        not in {
                            "page",
                            "bbox",
                            "position",
                            "box",
                            "caption",
                            "text",
                            "image",
                            "image_path",
                        }
                    },
                )
            )

        metadata.update(
            {
                "ocr": opts.ocr,
                "layout": opts.layout,
                "tables": opts.tables,
                "merge": opts.merge,
                "scrap_filter": opts.scrap_filter,
                "preserve_layout": opts.preserve_layout,
                "max_pages": opts.max_pages,
            }
        )

        return ParsedDocument(
            pages=pages,
            blocks=blocks,
            tables=tables,
            figures=figures,
            metadata=metadata,
            content_type=self.content_type,
        )

    def _run_backend(self, backend_cls: type, pdf_path: str, opts: PdfParseOptions) -> Any:
        """백엔드 파서를 실행합니다.

        백엔드 클래스의 인스턴스를 생성하고 PDF를 파싱합니다.

        Args:
            backend_cls: DeepDoc 파서 클래스 (RAGFlowPdfParser 또는 PlainParser)
            pdf_path: PDF 파일의 임시 경로
            opts: max_pages 등을 포함한 파싱 옵션

        Returns:
            백엔드 파서의 원시 결과
        """
        instance = backend_cls()
        # RAGFlowPdfParser, PlainParser는 __call__ 메서드 사용
        return instance(pdf_path)

    def run(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        """PDF 파일을 파싱하여 ParsedDocument를 반환합니다.

        전체 파싱 파이프라인을 실행합니다:
        1. 환경변수 설정
        2. 필요시 모델 다운로드
        3. 백엔드 클래스 로드
        4. PDF를 임시 파일로 저장
        5. 백엔드 실행 및 결과 정규화
        6. 임시 파일 정리

        Args:
            file: PDF 파일의 바이너리 스트림
            options: 파싱 옵션 (preferred_backend 필수 지정)

        Returns:
            파싱된 문서 정보를 담은 ParsedDocument 객체

        Raises:
            ImportError: 지정된 백엔드를 찾을 수 없는 경우

        Note:
            이 메서드는 BinaryIO를 입력으로 받습니다.
            파일 경로가 아닌 바이트로 전달되는 경우(HTTP 업로드, S3 다운로드,
            메모리 생성 PDF 등)를 지원하기 위함입니다.
            DeepDoc 백엔드는 파일 경로(str)만 받을 수 있으므로, 바이트를 임시 파일로
            저장한 뒤 경로를 전달합니다. 임시 파일은 처리 완료 후 자동 삭제됩니다.
        """
        opts = options or PdfParseOptions()
        self._configure_hf_env(opts)
        self._maybe_download_models(opts)
        backend_cls = self._get_backend_class(preferred=opts.preferred_backend)
        if backend_cls is None:
            raise ImportError(
                f"DeepDoc backend not available. "
                f"Specify preferred_backend from: {DeepDocBackend.choices()}"
            )

        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except Exception:
                pass
        pdf_bytes = file.read()

        temp_path = ""
        try:
            # 1. 임시 파일 생성
            with tempfile.NamedTemporaryFile(
                prefix="deepdoc_",      # 파일명 접두사: deepdoc_xxxxx.pdf
                suffix=".pdf",          # 확장자
                delete=False            # with 블록 끝나도 삭제하지 않음 (수동 삭제 예정)
            ) as handle:
                handle.write(pdf_bytes)  # PDF 바이트 쓰기
                handle.flush()           # 디스크에 확실히 기록
                temp_path = handle.name  # 파일 경로 저장 (예: /tmp/deepdoc_abc123.pdf)


            backend_result = self._run_backend(backend_cls, temp_path, opts)
            parsed = self._coerce_document(backend_result, opts)
            parsed.metadata["backend"] = backend_cls.__name__
            return parsed
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


__all__ = ["DeepDocPdfEngine"]
