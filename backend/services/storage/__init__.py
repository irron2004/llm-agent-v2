"""Storage service module for object storage operations."""

from .minio_client import (
    get_minio_client,
    ensure_bucket,
    upload_bytes,
    get_object,
    get_page_image,
    get_doc_page_count,
    upload_page_image,
    generate_image_path,
    sanitize_doc_id,
)
from .image_upload_renderer import (
    ImageUploadRenderer,
    create_renderer_for_doc,
)

__all__ = [
    "get_minio_client",
    "ensure_bucket",
    "upload_bytes",
    "get_object",
    "get_page_image",
    "get_doc_page_count",
    "upload_page_image",
    "generate_image_path",
    "sanitize_doc_id",
    "ImageUploadRenderer",
    "create_renderer_for_doc",
]
