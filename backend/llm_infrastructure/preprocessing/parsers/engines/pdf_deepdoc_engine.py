"""Engine that wraps DeepDoc/RAGFlow PDF parsing with graceful fallbacks."""

from __future__ import annotations

import importlib
import io
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
from .pdf_plain_engine import PlainPdfEngine

logger = logging.getLogger(__name__)


class DeepDocPdfEngine:
    content_type: str = "application/pdf"

    # 설치 방식에 따라 다를 수 있는 모듈 경로
    MODULE_CANDIDATES = [
        "deepdoc.parser.pdf_parser",
        "ragflow.deepdoc.parser.pdf_parser",
    ]

    def __init__(self, plain_engine: Optional[PlainPdfEngine] = None) -> None:
        self.plain_engine = plain_engine or PlainPdfEngine()

    def _load_backend_class(
        self, preferred: Optional[DeepDocBackend] = None
    ) -> Optional[type]:
        """Attempt to locate a DeepDoc-compatible parser class.

        Args:
            preferred: Explicitly requested backend. Must be specified.

        Returns:
            The backend class if found, None otherwise.
        """
        if not preferred:
            logger.error(
                f"DeepDoc backend must be explicitly specified. "
                f"Available options: {DeepDocBackend.choices()}"
            )
            return None

        target_class = preferred.value
        for module_name in self.MODULE_CANDIDATES:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            backend_cls = getattr(module, target_class, None)
            if backend_cls:
                logger.info(f"Using DeepDoc backend: {target_class}")
                return backend_cls

        # 원하는 백엔드를 찾지 못함
        logger.warning(
            f"Backend '{target_class}' not found. "
            f"Available options: {DeepDocBackend.choices()}"
        )
        return None

    def _configure_hf_env(self, opts: PdfParseOptions) -> None:
        if opts.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", opts.hf_endpoint)
        if opts.model_root:
            root = str(Path(opts.model_root))
            os.environ.setdefault("HF_HOME", root)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", root)
            os.environ.setdefault("TRANSFORMERS_CACHE", root)
            Path(root).mkdir(parents=True, exist_ok=True)

    def _maybe_download_models(self, opts: PdfParseOptions) -> None:
        if not opts.allow_download or not opts.model_root:
            return
        repos = [opts.ocr_model, opts.layout_model, opts.tsr_model]
        repos = [repo for repo in repos if repo]
        if not repos:
            return
        try:  # pragma: no cover - optional dependency
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
                # Download errors are non-fatal; fallback will handle missing assets.
                continue

    def _coerce_bbox(self, payload: Any) -> Optional[BoundingBox]:
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
        if raw is None:
            return
        if isinstance(raw, dict):
            # Common DeepDoc output shape: {"pages": [...], "blocks": [...]}
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
        instance = backend_cls()
        call_kwargs: Dict[str, Any] = {}
        if opts.max_pages is not None:
            call_kwargs["max_pages"] = opts.max_pages
        if hasattr(instance, "parse"):
            return instance.parse(pdf_path, **call_kwargs)
        if callable(instance):
            return instance(pdf_path, **call_kwargs)
        raise TypeError(f"Backend parser {backend_cls} is not callable")

    def run(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        opts = options or PdfParseOptions()
        self._configure_hf_env(opts)
        self._maybe_download_models(opts)
        backend_cls = self._load_backend_class(preferred=opts.preferred_backend)
        if backend_cls is None:
            if opts.fallback_to_plain:
                return self._fallback_to_plain(file, opts, reason="DeepDoc backend not available")
            raise ImportError("DeepDoc backend is not available; install ragflow/deepdoc to enable it.")

        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except Exception:
                pass
        pdf_bytes = file.read()

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="deepdoc_", suffix=".pdf", delete=False) as handle:
                handle.write(pdf_bytes)
                handle.flush()
                temp_path = handle.name

            backend_result = self._run_backend(backend_cls, temp_path, opts)
            parsed = self._coerce_document(backend_result, opts)
            parsed.metadata["backend"] = backend_cls.__name__
            return parsed
        except Exception as exc:
            if opts.fallback_to_plain:
                return self._fallback_to_plain(io.BytesIO(pdf_bytes), opts, reason=str(exc))
            raise
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def _fallback_to_plain(self, file: BinaryIO, opts: PdfParseOptions, *, reason: str) -> ParsedDocument:
        fallback_doc = self.plain_engine.run(file, options=opts)
        fallback_doc.metadata.update(
            {"parser": "pdf_deepdoc", "used_fallback": True, "fallback_reason": reason}
        )
        return fallback_doc


__all__ = ["DeepDocPdfEngine"]
