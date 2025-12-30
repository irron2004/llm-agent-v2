#!/usr/bin/env python3
"""Summarize all ES documents via the summarization API.

Usage:
  python scripts/es_summarize_all.py
  python scripts/es_summarize_all.py --base-url http://10.10.100.45:8001/api/summarization
  python scripts/es_summarize_all.py --limit 10 --sleep 0.2
  python scripts/es_summarize_all.py --force-regenerate
  python scripts/es_summarize_all.py --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx


async def fetch_total(client: httpx.AsyncClient, base_url: str) -> int:
    resp = await client.get(f"{base_url}/es/documents", params={"size": 1})
    resp.raise_for_status()
    data = resp.json()
    total = data.get("total")
    if total is None:
        raise ValueError("Response missing 'total' field")
    return int(total)


async def list_documents(client: httpx.AsyncClient, base_url: str, size: int) -> list[dict]:
    resp = await client.get(f"{base_url}/es/documents", params={"size": size})
    resp.raise_for_status()
    data = resp.json()
    docs = data.get("documents", [])
    if not isinstance(docs, list):
        raise ValueError("Response 'documents' is not a list")
    return docs


async def summarize_document(
    client: httpx.AsyncClient,
    base_url: str,
    doc_id: str,
    *,
    force_regenerate: bool,
    update_es: bool,
) -> dict:
    url = f"{base_url}/es/summarize/{quote(doc_id, safe='')}"
    payload = {
        "force_regenerate": force_regenerate,
        "update_es": update_es,
    }
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize all ES documents and update ES with summaries."
    )
    parser.add_argument(
        "--base-url",
        default="http://10.10.100.45:8001/api/summarization",
        help="Summarization API base URL",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Override size for /es/documents (defaults to total)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N documents",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between requests (seconds)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate summaries even if they exist",
    )
    parser.add_argument(
        "--no-update-es",
        action="store_true",
        help="Do not write summaries back to ES",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent requests (default: 4)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save log files (default: logs)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable log file (only console output)",
    )
    return parser.parse_args()


async def process_document(
    client: httpx.AsyncClient,
    base_url: str,
    doc_id: str,
    idx: int,
    total: int,
    *,
    force_regenerate: bool,
    update_es: bool,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
) -> tuple[str, bool, str | None, float]:
    """Process a single document with concurrency control.

    Returns:
        Tuple of (doc_id, success, error_message, duration_seconds)
    """
    async with semaphore:
        start_time = time.time()
        msg = f"[{idx}/{total}] Summarizing: {doc_id}"
        print(msg)
        logger.info(msg)

        try:
            await summarize_document(
                client,
                base_url,
                doc_id,
                force_regenerate=force_regenerate,
                update_es=update_es,
            )
            duration = time.time() - start_time
            success_msg = f"  ✓ Success (took {duration:.1f}s)"
            print(success_msg)
            logger.info(success_msg)
            return (doc_id, True, None, duration)
        except Exception as exc:
            import traceback
            duration = time.time() - start_time
            error_msg = f"{type(exc).__name__}: {str(exc)}"
            fail_msg = f"  ✗ ERROR: {error_msg} (after {duration:.1f}s)"
            print(fail_msg)
            logger.error(fail_msg)
            # Print traceback for debugging
            if "--debug" in sys.argv:
                traceback.print_exc()
            return (doc_id, False, error_msg, duration)


def setup_logger(args: argparse.Namespace) -> logging.Logger:
    """Setup logger with console and file handlers."""
    logger = logging.getLogger("es_summarize_all")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if not args.no_log_file:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"es_summarize_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        print(f"Log file: {log_file}")

    return logger


async def async_main(args: argparse.Namespace) -> int:
    logger = setup_logger(args)
    update_es = not args.no_update_es

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("ES Document Summarization Started")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Update ES: {update_es}")
    logger.info(f"Force regenerate: {args.force_regenerate}")
    logger.info("=" * 80)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        size = args.size
        if size is None:
            size = await fetch_total(client, args.base_url)

        docs = await list_documents(client, args.base_url, size=size)
        if args.limit is not None:
            docs = docs[: args.limit]

        total = len(docs)
        if total == 0:
            print("No documents found.")
            logger.warning("No documents found.")
            return 0

        logger.info(f"Processing {total} documents...")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(args.concurrency)

        # Create tasks for all documents
        tasks = []
        for idx, doc in enumerate(docs, start=1):
            doc_id = doc.get("doc_id")
            if not doc_id:
                msg = f"[{idx}/{total}] ERROR: Missing doc_id in response"
                print(msg)
                logger.error(msg)
                continue

            task = process_document(
                client,
                args.base_url,
                doc_id,
                idx,
                total,
                force_regenerate=args.force_regenerate,
                update_es=update_es,
                semaphore=semaphore,
                logger=logger,
            )
            tasks.append(task)

            # Sleep between task creation (optional rate limiting)
            if args.sleep > 0 and idx < total:
                await asyncio.sleep(args.sleep)

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success = 0
        failures: list[tuple[str, str]] = []
        total_duration = 0.0

        for result in results:
            if isinstance(result, Exception):
                failures.append(("<exception>", str(result)))
                logger.error(f"Exception: {result}")
            else:
                doc_id, succeeded, error, duration = result
                total_duration += duration
                if succeeded:
                    success += 1
                else:
                    failures.append((doc_id, error or "Unknown error"))

        elapsed = time.time() - start_time
        avg_duration = total_duration / len(results) if results else 0

        summary = f"\nDone. Success: {success}, Failed: {len(failures)}"
        print(summary)
        logger.info("=" * 80)
        logger.info(summary)
        logger.info(f"Total elapsed time: {elapsed:.1f}s")
        logger.info(f"Average time per document: {avg_duration:.1f}s")

        if failures:
            fail_summary = "Failed doc_ids:"
            print(fail_summary)
            logger.info(fail_summary)
            for doc_id, err in failures:
                fail_msg = f"  - {doc_id}: {err}"
                print(fail_msg)
                logger.info(fail_msg)

        logger.info("=" * 80)

    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
