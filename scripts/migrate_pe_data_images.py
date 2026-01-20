#!/usr/bin/env python3
"""Migrate PE Agent PDF page images to MinIO.

This script processes PDFs from pe_agent_data directories and uploads page images to MinIO.

Supported directories:
- pe_preprocess_data/set_up_manual/  -> Set_Up_Manual PDFs
- pe_preprocess_data/sop_pdfs/       -> Global SOP PDFs
- pe_preprocess_data/ts_pdfs/        -> Trouble Shooting Guide PDFs

Usage:
    # Dry run (no upload)
    python scripts/migrate_pe_data_images.py --dry-run

    # Migrate all PDFs
    python scripts/migrate_pe_data_images.py

    # Migrate specific directory
    python scripts/migrate_pe_data_images.py --source-dir /path/to/pdfs

    # With limit
    python scripts/migrate_pe_data_images.py --limit 10
"""

import argparse
import io
import logging
import os
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf2image import convert_from_path
from minio import Minio
from minio.error import S3Error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Global MinIO client
_minio_client: Minio | None = None
_bucket_name: str = ""


def sanitize_doc_id(doc_id: str) -> str:
    """Sanitize document ID for use in object paths."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", doc_id)


def pdf_name_to_doc_id(pdf_name: str) -> str:
    """Convert PDF filename to doc_id format matching ES index.

    Examples:
        Set_Up_Manual_SUPRA_N.pdf -> set_up_manual_supra_n
        Global SOP_ZEDIUS XP_ALL_PM.pdf -> global_sop_zedius_xp_all_pm
        SUPRA N_ALL_Trouble_Shooting_Guide_Trace TM Robot Abnormal.pdf
            -> supra_n_all_trouble_shooting_guide_trace_tm_robot_abnormal
    """
    # Remove .pdf extension
    name = pdf_name.replace(".pdf", "").replace(".PDF", "")
    # Convert to lowercase and replace spaces/special chars with underscore
    doc_id = re.sub(r"[^a-zA-Z0-9]+", "_", name).lower()
    # Remove leading/trailing underscores
    doc_id = doc_id.strip("_")
    # Replace multiple underscores with single
    doc_id = re.sub(r"_+", "_", doc_id)
    return doc_id


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


def check_existing(doc_id: str) -> bool:
    """Check if images already exist for this doc_id."""
    client = get_client()
    safe_doc_id = sanitize_doc_id(doc_id)
    prefix = f"documents/{safe_doc_id}/"
    try:
        objects = list(client.list_objects(_bucket_name, prefix=prefix))
        return len(objects) > 0
    except Exception:
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


def find_pdf_files(source_dirs: list[Path]) -> list[tuple[str, Path]]:
    """Find all PDF files in source directories.

    Returns:
        List of (doc_id, pdf_path) tuples
    """
    pdf_files = []

    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.warning(f"Directory not found: {source_dir}")
            continue

        # Find all PDFs recursively
        for pdf_path in source_dir.rglob("*.pdf"):
            doc_id = pdf_name_to_doc_id(pdf_path.name)
            pdf_files.append((doc_id, pdf_path))

        # Also check for .PDF extension
        for pdf_path in source_dir.rglob("*.PDF"):
            doc_id = pdf_name_to_doc_id(pdf_path.name)
            pdf_files.append((doc_id, pdf_path))

    return pdf_files


def render_pdf_pages(pdf_path: Path, dpi: int = 150) -> list[bytes]:
    """Render PDF pages to PNG bytes."""
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
    skip_existing: bool = True,
) -> tuple[int, int]:
    """Migrate a single PDF's pages to MinIO."""
    safe_doc_id = sanitize_doc_id(doc_id)

    # Check if already exists
    if skip_existing and not dry_run and check_existing(doc_id):
        logger.info(f"  Skipping (already exists): {safe_doc_id}")
        return 0, 0

    logger.info(f"Processing: {pdf_path.name} -> {safe_doc_id}")

    try:
        png_bytes_list = render_pdf_pages(pdf_path, dpi=dpi)
    except Exception as e:
        logger.error(f"Failed to render PDF {pdf_path}: {e}")
        return 0, 0

    total = len(png_bytes_list)
    success = 0

    for page_num, png_bytes in enumerate(png_bytes_list, start=1):
        if dry_run:
            logger.debug(f"  [DRY RUN] Would upload page {page_num}/{total}")
            success += 1
        else:
            result = upload_page_image(safe_doc_id, page_num, png_bytes)
            if result:
                success += 1
            else:
                logger.warning(f"  Failed to upload page {page_num}/{total}")

    logger.info(f"  Completed: {success}/{total} pages")
    return success, total


def main():
    global _minio_client, _bucket_name

    parser = argparse.ArgumentParser(description="Migrate PE Agent PDF images to MinIO")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Specific source directory (default: all pe_preprocess_data subdirs)",
    )
    parser.add_argument(
        "--pe-data-root",
        type=Path,
        default=Path("/home/llm-share/datasets/pe_agent_data"),
        help="PE agent data root directory",
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
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip documents that already have images (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-upload all documents even if they exist",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="localhost:9000",
        help="MinIO endpoint (default: localhost:9000)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="doc-images",
        help="MinIO bucket name (default: doc-images)",
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
    endpoint = args.endpoint
    if endpoint.startswith("http://"):
        endpoint = endpoint[7:]
    elif endpoint.startswith("https://"):
        endpoint = endpoint[8:]

    _bucket_name = args.bucket

    # Get credentials
    access_key = args.access_key or os.environ.get("MINIO_ROOT_USER", "minioadmin")
    secret_key = args.secret_key or os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin123")

    # Initialize MinIO client
    if not args.dry_run:
        logger.info("Connecting to MinIO...")
        logger.info(f"  Endpoint: {endpoint}")
        logger.info(f"  Bucket: {_bucket_name}")

        try:
            _minio_client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False,
            )
            if not ensure_bucket():
                logger.error("Failed to ensure bucket exists")
                sys.exit(1)
            logger.info("  MinIO connection OK")
        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            sys.exit(1)

    # Determine source directories
    if args.source_dir:
        source_dirs = [args.source_dir]
    else:
        pe_preprocess = args.pe_data_root / "pe_preprocess_data"
        source_dirs = [
            pe_preprocess / "set_up_manual",
            pe_preprocess / "sop_pdfs",
            pe_preprocess / "ts_pdfs",
        ]

    # Find PDF files
    logger.info("Scanning for PDF files...")
    pdf_files = find_pdf_files(source_dirs)
    logger.info(f"Found {len(pdf_files)} PDF files")

    if args.limit:
        pdf_files = pdf_files[:args.limit]
        logger.info(f"Limited to {len(pdf_files)} files")

    if not pdf_files:
        logger.warning("No PDF files found")
        return

    # Process each PDF
    total_success = 0
    total_pages = 0
    skipped = 0

    for i, (doc_id, pdf_path) in enumerate(pdf_files, start=1):
        logger.info(f"[{i}/{len(pdf_files)}] {doc_id}")
        success, total = migrate_pdf_to_minio(
            doc_id=doc_id,
            pdf_path=pdf_path,
            dpi=args.dpi,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
        if success == 0 and total == 0 and args.skip_existing:
            skipped += 1
        else:
            total_success += success
            total_pages += total

    # Summary
    logger.info("=" * 60)
    logger.info("Migration Summary:")
    logger.info(f"  Documents processed: {len(pdf_files)}")
    logger.info(f"  Documents skipped (existing): {skipped}")
    logger.info(f"  Pages uploaded: {total_success}/{total_pages}")
    if args.dry_run:
        logger.info("  Mode: DRY RUN (no actual uploads)")


if __name__ == "__main__":
    main()
