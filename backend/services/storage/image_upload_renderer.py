"""PDF renderer that uploads page images to MinIO.

This renderer wraps the default pdf2image rendering and uploads
each page image to MinIO storage while returning images for VLM processing.
"""

import io
import logging
from typing import Any, List

from PIL import Image

from .minio_client import upload_page_image

logger = logging.getLogger(__name__)


class ImageUploadRenderer:
    """PDF page renderer that uploads images to MinIO.

    Usage:
        renderer = ImageUploadRenderer(doc_id="doc_001")
        images = renderer(pdf_bytes)  # Renders, uploads, and returns images
    """

    def __init__(
        self,
        doc_id: str,
        *,
        dpi: int = 150,
        fmt: str = "PNG",
        upload_enabled: bool = True,
    ) -> None:
        """Initialize the renderer.

        Args:
            doc_id: Document ID for naming uploaded images
            dpi: DPI for rendering (default 150 for balance of quality/size)
            fmt: Image format (PNG recommended for quality)
            upload_enabled: Whether to upload images to MinIO
        """
        self.doc_id = doc_id
        self.dpi = dpi
        self.fmt = fmt
        self.upload_enabled = upload_enabled

    def __call__(self, pdf_bytes: bytes) -> List[Any]:
        """Render PDF pages and upload to MinIO.

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            List of PIL Image objects for VLM processing
        """
        try:
            from pdf2image import convert_from_bytes
        except ImportError as exc:
            raise ImportError(
                "pdf2image is required for PDF rendering. "
                "Install with: pip install pdf2image"
            ) from exc

        # Render PDF pages to images
        images = convert_from_bytes(pdf_bytes, dpi=self.dpi, fmt=self.fmt)
        logger.info("Rendered %d pages from PDF for doc_id=%s", len(images), self.doc_id)

        # Upload each page to MinIO if enabled
        if self.upload_enabled:
            for page_num, image in enumerate(images, start=1):
                try:
                    # Convert PIL Image to PNG bytes
                    img_bytes = self._image_to_bytes(image)
                    result = upload_page_image(self.doc_id, page_num, img_bytes)
                    if result:
                        logger.debug(
                            "Uploaded page %d for doc_id=%s", page_num, self.doc_id
                        )
                    else:
                        logger.warning(
                            "Failed to upload page %d for doc_id=%s",
                            page_num,
                            self.doc_id,
                        )
                except Exception as e:
                    logger.error(
                        "Error uploading page %d for doc_id=%s: %s",
                        page_num,
                        self.doc_id,
                        e,
                    )
                    # Continue with other pages even if one fails

        return images

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to PNG bytes.

        Args:
            image: PIL Image object

        Returns:
            PNG image as bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def create_renderer_for_doc(doc_id: str, **kwargs) -> ImageUploadRenderer:
    """Factory function to create a renderer for a specific document.

    Args:
        doc_id: Document ID
        **kwargs: Additional kwargs passed to ImageUploadRenderer

    Returns:
        Configured ImageUploadRenderer instance
    """
    return ImageUploadRenderer(doc_id=doc_id, **kwargs)
