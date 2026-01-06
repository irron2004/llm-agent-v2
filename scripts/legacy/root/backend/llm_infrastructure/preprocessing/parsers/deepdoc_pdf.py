"""Wrapper around DeepDoc/RAGFlow PDF parsers with graceful fallbacks."""

from __future__ import annotations

import importlib
import io
import os
import tempfile
from typing import Any, BinaryIO, Dict, Iterable, List, Optional

from .base import (
    BaseParser,
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    ParsedFigure,
    ParsedPage,
    ParsedTable,
    PdfParseOptions,
)
from .pdf_plain import PlainPdfParser
from .registry import register_parser


class DeepDocPdfParser(BaseParser):
    content_type: str = "application/pdf"

    def _load_backend_class(self) -> Optional[type]:
        """Attempt to locate a DeepDoc-compatible parser class."""

        candidates = [
            ("deepdoc.parser.pdf_parser", ("RAGFlowPdfParser", "PdfParser", "PlainParser")),
            ("ragflow.deepdoc.parser.pdf_parser", ("RAGFlowPdfParser", "PdfParser", "PlainParser")),
        ]
        for module_name, class_names in candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for class_name in class_names:
                backend_cls = getattr(module, class_name, None)
                if backend_cls:
                    return backend_cls
        return None

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

    def parse(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        opts = options or PdfParseOptions()
        backend_cls = self._load_backend_class()
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
        fallback_doc = PlainPdfParser().parse(file, options=opts)
        fallback_doc.metadata.update(
            {"parser": "pdf_deepdoc", "used_fallback": True, "fallback_reason": reason}
        )
        return fallback_doc


register_parser("pdf_deepdoc", DeepDocPdfParser)

__all__ = ["DeepDocPdfParser"]
