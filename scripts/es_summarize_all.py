#!/usr/bin/env python3
"""Summarize all ES documents via the summarization API.

Usage:
  python scripts/es_summarize_all.py
  python scripts/es_summarize_all.py --base-url http://10.10.100.45:8001/api/summarization
  python scripts/es_summarize_all.py --limit 10 --sleep 0.2
  python scripts/es_summarize_all.py --force-regenerate
  python scripts/es_summarize_all.py --timeout 600 --concurrency 1
  python scripts/es_summarize_all.py --doc-id supra_n_all_trouble_shooting_guide_trace_tool_shut_down
"""

from __future__ import annotations

import argparse
import asyncio
import logging
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


def single_line(text: str) -> str:
    """Normalize arbitrary text into a single log-friendly line."""
    return " ".join(str(text).splitlines()).strip()


def is_transient_httpx_error(exc: Exception) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))


async def summarize_document_with_retry(
    client: httpx.AsyncClient,
    base_url: str,
    doc_id: str,
    *,
    force_regenerate: bool,
    update_es: bool,
    retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
) -> dict:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await summarize_document(
                client,
                base_url,
                doc_id,
                force_regenerate=force_regenerate,
                update_es=update_es,
            )
        except Exception as exc:  # noqa: BLE001
            if not is_transient_httpx_error(exc):
                raise
            last_exc = exc
            if attempt >= retries:
                break
            delay = min(retry_max_backoff, retry_backoff * (2**attempt))
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


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
        "--doc-id",
        action="append",
        default=None,
        help="Process only the given doc_id (repeatable)",
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
        default=600.0,
        help="Read timeout in seconds (per request)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Connect/write/pool timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries for transient network/timeout errors (default: 2)",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Retry backoff base seconds (default: 1.0)",
    )
    parser.add_argument(
        "--retry-max-backoff",
        type=float,
        default=30.0,
        help="Retry backoff cap seconds (default: 30.0)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full traceback on errors",
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
    retries: int,
    retry_backoff: float,
    retry_max_backoff: float,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    debug: bool,
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
            await summarize_document_with_retry(
                client,
                base_url,
                doc_id,
                force_regenerate=force_regenerate,
                update_es=update_es,
                retries=retries,
                retry_backoff=retry_backoff,
                retry_max_backoff=retry_max_backoff,
            )
            duration = time.time() - start_time
            success_msg = f"  ✓ Success: {doc_id} (took {duration:.1f}s)"
            print(success_msg)
            logger.info(success_msg)
            return (doc_id, True, None, duration)
        except Exception as exc:
            import traceback
            duration = time.time() - start_time
            if isinstance(exc, httpx.TimeoutException):
                error_msg = (
                    f"{type(exc).__name__}: timed out "
                    f"(connect={client.timeout.connect}s read={client.timeout.read}s) "
                    "hint: increase --timeout or lower --concurrency"
                )
            elif isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                url = exc.request.url
                error_msg = f"HTTP {status} for {url}"
            else:
                error_msg = f"{type(exc).__name__}: {single_line(str(exc) or repr(exc))}"
            fail_msg = f"  ✗ ERROR: {doc_id}: {error_msg} (after {duration:.1f}s)"
            print(fail_msg)
            logger.error(fail_msg)
            # Print traceback for debugging
            if debug:
                traceback.print_exc()
            return (doc_id, False, error_msg, duration)


def setup_logger(args: argparse.Namespace) -> logging.Logger:
    """Setup logger (file handler by default)."""
    logger = logging.getLogger("es_summarize_all")
    logger.setLevel(logging.INFO)
    logger.propagate = False

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

    print(
        f"Base URL: {args.base_url} | concurrency={args.concurrency} | "
        f"read_timeout={args.timeout}s | retries={args.retries}"
    )

    timeout = httpx.Timeout(
        connect=args.connect_timeout,
        read=args.timeout,
        write=args.connect_timeout,
        pool=args.connect_timeout,
    )
    limits = httpx.Limits(
        max_connections=max(1, args.concurrency),
        max_keepalive_connections=max(1, args.concurrency),
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        if args.doc_id:
            docs = [{"doc_id": doc_id} for doc_id in args.doc_id]
        else:
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

        # Create tasks for all documents (optionally staggered).
        created_tasks: list[asyncio.Task[tuple[str, bool, str | None, float]]] = []
        for idx, doc in enumerate(docs, start=1):
            doc_id = doc.get("doc_id")
            if not doc_id:
                msg = f"[{idx}/{total}] ERROR: Missing doc_id in response"
                print(msg)
                logger.error(msg)
                continue

            created_tasks.append(
                asyncio.create_task(
                    process_document(
                        client,
                        args.base_url,
                        doc_id,
                        idx,
                        total,
                        force_regenerate=args.force_regenerate,
                        update_es=update_es,
                        retries=args.retries,
                        retry_backoff=args.retry_backoff,
                        retry_max_backoff=args.retry_max_backoff,
                        semaphore=semaphore,
                        logger=logger,
                        debug=args.debug,
                    )
                )
            )

            # Optional rate limiting: stagger task creation / request start.
            if args.sleep > 0 and idx < total:
                await asyncio.sleep(args.sleep)

        # Run all tasks concurrently
        if args.fail_fast:
            results: list[tuple[str, bool, str | None, float] | Exception] = []
            for fut in asyncio.as_completed(created_tasks):
                try:
                    result = await fut
                except Exception as exc:  # noqa: BLE001
                    result = exc
                results.append(result)

                failed = isinstance(result, Exception) or not result[1]
                if failed:
                    cancel_msg = "Fail-fast enabled; cancelling remaining tasks."
                    print(cancel_msg)
                    logger.warning(cancel_msg)
                    for task in created_tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*created_tasks, return_exceptions=True)
                    break
        else:
            results = await asyncio.gather(*created_tasks, return_exceptions=True)

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
