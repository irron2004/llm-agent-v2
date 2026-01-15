"""MinIO client utilities for document image storage.

This module provides functions for:
- Connecting to MinIO
- Uploading document page images
- Retrieving images for API serving
"""

import io
import re
import logging
from functools import lru_cache
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from backend.config.settings import minio_settings

logger = logging.getLogger(__name__)


def sanitize_doc_id(doc_id: str) -> str:
    """Sanitize document ID for use in object paths.

    Replaces any characters that are not alphanumeric, dots, underscores,
    or hyphens with underscores.

    Args:
        doc_id: Original document ID

    Returns:
        Sanitized document ID safe for use in S3 paths
    """
    return re.sub(r"[^a-zA-Z0-9._-]", "_", doc_id)


def generate_image_path(doc_id: str, page: int) -> str:
    """Generate the object path for a document page image.

    Args:
        doc_id: Document ID
        page: Page number (1-indexed)

    Returns:
        Object path in format: documents/{safe_doc_id}/page_{page}.png
    """
    safe_doc_id = sanitize_doc_id(doc_id)
    return f"documents/{safe_doc_id}/page_{page}.png"


@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    """Get a cached MinIO client instance.

    Returns:
        Configured Minio client
    """
    # Parse endpoint to extract host and port
    parsed = urlparse(minio_settings.endpoint)
    endpoint = parsed.netloc or parsed.path

    logger.info(f"Creating MinIO client for endpoint: {endpoint}")

    return Minio(
        endpoint=endpoint,
        access_key=minio_settings.access_key,
        secret_key=minio_settings.secret_key,
        secure=minio_settings.secure,
    )


def ensure_bucket(bucket_name: str | None = None) -> bool:
    """Ensure the bucket exists, create if not.

    Args:
        bucket_name: Bucket name (defaults to settings.bucket)

    Returns:
        True if bucket exists or was created successfully
    """
    bucket = bucket_name or minio_settings.bucket
    client = get_minio_client()

    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info(f"Created bucket: {bucket}")
        return True
    except S3Error as e:
        logger.error(f"Failed to ensure bucket {bucket}: {e}")
        return False


def upload_bytes(
    data: bytes,
    object_name: str,
    content_type: str = "image/png",
    bucket_name: str | None = None,
) -> str | None:
    """Upload bytes data to MinIO.

    Args:
        data: Binary data to upload
        object_name: Object path/name in the bucket
        content_type: MIME type of the content
        bucket_name: Bucket name (defaults to settings.bucket)

    Returns:
        Object name if successful, None otherwise
    """
    bucket = bucket_name or minio_settings.bucket
    client = get_minio_client()

    try:
        # Ensure bucket exists
        ensure_bucket(bucket)

        # Upload the data
        data_stream = io.BytesIO(data)
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=data_stream,
            length=len(data),
            content_type=content_type,
        )
        logger.debug(f"Uploaded object: {bucket}/{object_name}")
        return object_name
    except S3Error as e:
        logger.error(f"Failed to upload {object_name}: {e}")
        return None


def get_object(
    object_name: str,
    bucket_name: str | None = None,
) -> bytes | None:
    """Get an object from MinIO.

    Args:
        object_name: Object path/name in the bucket
        bucket_name: Bucket name (defaults to settings.bucket)

    Returns:
        Object data as bytes if successful, None otherwise
    """
    bucket = bucket_name or minio_settings.bucket
    client = get_minio_client()

    try:
        response = client.get_object(bucket_name=bucket, object_name=object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except S3Error as e:
        if e.code == "NoSuchKey":
            logger.warning(f"Object not found: {bucket}/{object_name}")
        else:
            logger.error(f"Failed to get {object_name}: {e}")
        return None


def upload_page_image(doc_id: str, page: int, image_data: bytes) -> str | None:
    """Upload a document page image to MinIO.

    Args:
        doc_id: Document ID
        page: Page number (1-indexed)
        image_data: PNG image data

    Returns:
        Object path if successful, None otherwise
    """
    object_name = generate_image_path(doc_id, page)
    return upload_bytes(image_data, object_name, content_type="image/png")


def get_page_image(doc_id: str, page: int) -> bytes | None:
    """Get a document page image from MinIO.

    Args:
        doc_id: Document ID
        page: Page number (1-indexed)

    Returns:
        PNG image data if successful, None otherwise
    """
    object_name = generate_image_path(doc_id, page)
    return get_object(object_name)
