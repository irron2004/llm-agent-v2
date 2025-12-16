"""Engine that uses a vision-language model (e.g., DeepSeek-VL) to parse PDFs."""

from __future__ import annotations

from typing import Any, BinaryIO, Callable, Dict, List, Optional

from ..base import ParsedBlock, ParsedDocument, ParsedPage, PdfParseOptions

DEFAULT_PROMPT = (
    "Read this page and return all content as Markdown. Preserve tables and formulas. "
    "Do not summarize or omit text."
)


class VlmPdfEngine:
    """VLM-based PDF parser.

    Dependencies are injected to avoid hard coupling to any specific model or renderer:
      - vlm_client: object with generate(image=..., prompt=..., **kwargs) -> str
      - vlm_factory: callable that builds a vlm_client from PdfParseOptions
      - renderer: callable that turns PDF bytes into a list of image-like objects
                  (PIL images, bytes, etc. as expected by the VLM client)
    """

    content_type: str = "application/pdf"

    def __init__(
        self,
        *,
        vlm_client: Any | None = None,
        vlm_factory: Callable[[PdfParseOptions], Any] | None = None,
        renderer: Callable[[bytes], List[Any]] | None = None,
        default_prompt: str = DEFAULT_PROMPT,
    ) -> None:
        self.vlm_client = vlm_client
        self.vlm_factory = vlm_factory
        self.renderer = renderer
        self.default_prompt = default_prompt

    def _render_pdf(self, pdf_bytes: bytes) -> List[Any]:
        if self.renderer:
            return self.renderer(pdf_bytes)
        try:  # pragma: no cover - optional dependency
            from pdf2image import convert_from_bytes  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "pdf2image is required when no custom renderer is provided for VLM parsing"
            ) from exc
        return convert_from_bytes(pdf_bytes)

    def run(self, file: BinaryIO, options: Optional[PdfParseOptions] = None) -> ParsedDocument:
        opts = options or PdfParseOptions()
        if self.vlm_client is None and self.vlm_factory is not None:
            self.vlm_client = self.vlm_factory(opts)
        if self.vlm_client is None:
            raise ImportError("vlm_client is required for VlmPdfEngine")

        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except Exception:
                pass
        pdf_bytes = file.read()

        images = self._render_pdf(pdf_bytes)
        blocks: List[ParsedBlock] = []
        pages: List[ParsedPage] = []

        for index, image in enumerate(images, start=1):
            prompt = opts.vlm_prompt or self.default_prompt
            infer_kwargs: Dict[str, Any] = {}
            if opts.vlm_max_new_tokens is not None:
                infer_kwargs["max_tokens"] = opts.vlm_max_new_tokens
            if opts.vlm_temperature is not None:
                infer_kwargs["temperature"] = opts.vlm_temperature

            text = self.vlm_client.generate(image=image, prompt=prompt, **infer_kwargs)
            pages.append(ParsedPage(number=index, text=text))
            blocks.append(ParsedBlock(text=text, page=index, label="page"))

        metadata = {
            "parser": "pdf_vlm",
            "vlm_model": opts.vlm_model,
            "vlm_prompt": opts.vlm_prompt or self.default_prompt,
        }

        return ParsedDocument(
            pages=pages,
            blocks=blocks,
            tables=[],
            figures=[],
            metadata=metadata,
            errors=[],
            content_type=self.content_type,
        )


__all__ = ["VlmPdfEngine"]
