"""Lightweight PDF engine that relies on pdfplumber for text extraction."""

from __future__ import annotations

from typing import BinaryIO, Optional

from ..base import ParsedBlock, ParsedDocument, ParsedPage, PdfParseOptions

try:  # pragma: no cover - optional dependency
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None


class PlainPdfEngine:
    content_type: str = "application/pdf"

    def run(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        opts = options or PdfParseOptions()
        if not pdfplumber:
            raise ImportError("pdfplumber is required for PlainPdfEngine")

        blocks: list[ParsedBlock] = []
        pages: list[ParsedPage] = []
        errors: list[str] = []

        try:
            if hasattr(file, "seek"):
                file.seek(0)
            with pdfplumber.open(file) as pdf:  # type: ignore[arg-type]
                for index, page in enumerate(pdf.pages):
                    if opts.max_pages is not None and index >= opts.max_pages:
                        break
                    text = page.extract_text() or ""
                    pages.append(
                        ParsedPage(
                            number=index + 1,
                            text=text,
                            width=getattr(page, "width", None),
                            height=getattr(page, "height", None),
                        )
                    )
                    blocks.append(ParsedBlock(text=text, page=index + 1, label="page"))
        except Exception as exc:  # pragma: no cover - propagate parse failures
            errors.append(str(exc))
            raise

        metadata = {
            "parser": "pdf_plain",
            "ocr": False,
            "layout": False,
            "tables": False,
            "merge": False,
            "preserve_layout": opts.preserve_layout,
            "max_pages": opts.max_pages,
        }

        return ParsedDocument(
            pages=pages,
            blocks=blocks,
            tables=[],
            figures=[],
            metadata=metadata,
            errors=errors,
            content_type=self.content_type,
        )


__all__ = ["PlainPdfEngine"]
