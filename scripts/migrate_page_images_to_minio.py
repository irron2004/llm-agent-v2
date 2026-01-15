#!/usr/bin/env python3
"""Migrate existing PDF page images to MinIO.

This script:
1. Finds all PDF files in the ingestions folder
2. Renders each page as PNG
3. Uploads to MinIO using the doc_id + page naming convention

Usage:
    # Dry run (no upload)
    python scripts/migrate_page_images_to_minio.py --dry-run

    # Migrate specific run folder
    python scripts/migrate_page_images_to_minio.py --run-id 20251211_084140

    # Migrate all
    python scripts/migrate_page_images_to_minio.py

    # With custom DPI
    python scripts/migrate_page_images_to_minio.py --dpi 200
"""

import argparse
import io
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2image import convert_from_path
from PIL import Image

from minio import Minio
from minio.error import S3Error

from backend.config.settings import minio_settings
from backend.services.storage.minio_client import (
    sanitize_doc_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Global MinIO client (set in main)
_minio_client: Minio | None = None
_bucket_name: str = ""


def get_client() -> Minio:
    """Get the configured MinIO client."""
    if _minio_client is None:
        raise RuntimeError("MinIO client not initialized")
    return _minio_client


def ensure_bucket() -> bool:
    """Ensure the bucket exists."""
    client = get_client()
    try:
        if not client.bucket_exists(_bucket_name):
            client.make_bucket(_bucket_name)
            logger.info(f"Created bucket: {_bucket_name}")
        return True
    except S3Error as e:
        logger.error(f"Failed to ensure bucket {_bucket_name}: {e}")
        return False


def upload_page_image(doc_id: str, page: int, image_data: bytes) -> str | None:
    """Upload a document page image to MinIO."""
    safe_doc_id = sanitize_doc_id(doc_id)
    object_name = f"documents/{safe_doc_id}/page_{page}.png"
    client = get_client()

    try:
        data_stream = io.BytesIO(image_data)
        client.put_object(
            bucket_name=_bucket_name,
            object_name=object_name,
            data=data_stream,
            length=len(image_data),
            content_type="image/png",
        )
        return object_name
    except S3Error as e:
        logger.error(f"Failed to upload {object_name}: {e}")
        return None


def find_pdf_files(ingestions_dir: Path, run_id: str | None = None) -> list[tuple[str, Path]]:
    """Find all PDF files in ingestions directory.

    Args:
        ingestions_dir: Path to ingestions directory
        run_id: Optional run ID to filter by

    Returns:
        List of (doc_id, pdf_path) tuples
    """
    pdf_files = []

    if run_id:
        run_dirs = [ingestions_dir / run_id]
    else:
        run_dirs = [d for d in ingestions_dir.iterdir() if d.is_dir()]

    for run_dir in run_dirs:
        if not run_dir.exists():
            logger.warning(f"Run directory not found: {run_dir}")
            continue

        # Each subdirectory is a document
        for doc_dir in run_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            pdf_path = doc_dir / "source.pdf"
            if pdf_path.exists():
                # Use folder name as doc_id
                doc_id = doc_dir.name
                pdf_files.append((doc_id, pdf_path))

    return pdf_files


def render_pdf_pages(pdf_path: Path, dpi: int = 150) -> list[bytes]:
    """Render PDF pages to PNG bytes.

    Args:
        pdf_path: Path to PDF file
        dpi: DPI for rendering

    Returns:
        List of PNG image bytes for each page
    """
    images = convert_from_path(str(pdf_path), dpi=dpi, fmt="PNG")

    png_bytes_list = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        png_bytes_list.append(buffer.getvalue())

    return png_bytes_list


def migrate_pdf_to_minio(
    doc_id: str,
    pdf_path: Path,
    dpi: int = 150,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate a single PDF's pages to MinIO.

    Args:
        doc_id: Document ID
        pdf_path: Path to PDF file
        dpi: DPI for rendering
        dry_run: If True, don't actually upload

    Returns:
        Tuple of (success_count, total_count)
    """
    safe_doc_id = sanitize_doc_id(doc_id)
    logger.info(f"Processing: {doc_id} -> {safe_doc_id}")

    try:
        png_bytes_list = render_pdf_pages(pdf_path, dpi=dpi)
    except Exception as e:
        logger.error(f"Failed to render PDF {pdf_path}: {e}")
        return 0, 0

    total = len(png_bytes_list)
    success = 0

    for page_num, png_bytes in enumerate(png_bytes_list, start=1):
        if dry_run:
            logger.info(f"  [DRY RUN] Would upload page {page_num}/{total} ({len(png_bytes)} bytes)")
            success += 1
        else:
            result = upload_page_image(safe_doc_id, page_num, png_bytes)
            if result:
                logger.debug(f"  Uploaded page {page_num}/{total}")
                success += 1
            else:
                logger.warning(f"  Failed to upload page {page_num}/{total}")

    logger.info(f"  Completed: {success}/{total} pages")
    return success, total


def main():
    global _minio_client, _bucket_name

    parser = argparse.ArgumentParser(description="Migrate PDF page images to MinIO")
    parser.add_argument(
        "--ingestions-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "ingestions",
        help="Path to ingestions directory",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to migrate (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for rendering (default: 150)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually upload, just show what would be done",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="MinIO endpoint (default: from settings, use 'localhost:9000' for local)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="MinIO bucket name (default: from settings)",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        default=None,
        help="MinIO access key (default: from env MINIO_ROOT_USER)",
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        default=None,
        help="MinIO secret key (default: from env MINIO_ROOT_PASSWORD)",
    )

    args = parser.parse_args()

    # Determine endpoint and bucket
    endpoint = args.endpoint or minio_settings.endpoint
    # Remove http:// prefix and extract host:port
    if endpoint.startswith("http://"):
        endpoint = endpoint[7:]
    elif endpoint.startswith("https://"):
        endpoint = endpoint[8:]

    _bucket_name = args.bucket or minio_settings.bucket

    # Get credentials (prefer args, then env vars, then settings)
    access_key = args.access_key or os.environ.get("MINIO_ROOT_USER") or minio_settings.access_key
    secret_key = args.secret_key or os.environ.get("MINIO_ROOT_PASSWORD") or minio_settings.secret_key

    # Check MinIO connection
    if not args.dry_run:
        logger.info("Checking MinIO connection...")
        logger.info(f"  Endpoint: {endpoint}")
        logger.info(f"  Bucket: {_bucket_name}")

        try:
            _minio_client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=minio_settings.secure,
            )
            if not ensure_bucket():
                logger.error("Failed to ensure bucket exists")
                sys.exit(1)
            logger.info("  MinIO connection OK")
        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            sys.exit(1)

    # Find PDF files
    logger.info(f"Scanning ingestions directory: {args.ingestions_dir}")
    pdf_files = find_pdf_files(args.ingestions_dir, args.run_id)
    logger.info(f"Found {len(pdf_files)} PDF files")

    if args.limit:
        pdf_files = pdf_files[:args.limit]
        logger.info(f"Limited to {len(pdf_files)} files")

    if not pdf_files:
        logger.warning("No PDF files found to migrate")
        return

    # Process each PDF
    total_success = 0
    total_pages = 0

    for i, (doc_id, pdf_path) in enumerate(pdf_files, start=1):
        logger.info(f"[{i}/{len(pdf_files)}] {doc_id}")
        success, total = migrate_pdf_to_minio(
            doc_id=doc_id,
            pdf_path=pdf_path,
            dpi=args.dpi,
            dry_run=args.dry_run,
        )
        total_success += success
        total_pages += total

    # Summary
    logger.info("=" * 60)
    logger.info("Migration Summary:")
    logger.info(f"  Documents processed: {len(pdf_files)}")
    logger.info(f"  Pages uploaded: {total_success}/{total_pages}")
    if args.dry_run:
        logger.info("  Mode: DRY RUN (no actual uploads)")


if __name__ == "__main__":
    main()
