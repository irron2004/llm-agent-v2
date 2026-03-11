#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import cast

DEFAULT_API_BASE_URL = "http://localhost:8011"


@dataclass(frozen=True)
class CliArgs:
    api_base_url: str
    input_path: str
    out_path: str
    timeout_seconds: float
    limit: int | None


@dataclass(frozen=True)
class QueryRow:
    qid: str
    query: str


def _coerce_str_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    mapped: dict[str, object] = {}
    for key, item in raw.items():
        mapped[str(key)] = item
    return mapped


def _build_endpoint(base_url: str, path: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/api"):
        return f"{stripped}{path}"
    return f"{stripped}/api{path}"


def _post_json(
    url: str, payload: dict[str, object], *, timeout_seconds: float
) -> tuple[dict[str, object], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with cast(
            http.client.HTTPResponse,
            urllib.request.urlopen(req, timeout=timeout_seconds),
        ) as resp:
            raw = resp.read()
    except TimeoutError as exc:
        raise RuntimeError(f"Request timed out for {url}") from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    try:
        parsed = cast(object, json.loads(raw.decode("utf-8")))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc

    mapped = _coerce_str_mapping(parsed)
    if mapped is None:
        raise RuntimeError(
            f"Unexpected response shape from {url}: JSON object expected"
        )
    return mapped, elapsed_ms


def _resolve_input_path(input_path: str) -> Path:
    candidate = Path(input_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / candidate).resolve()


def _load_queries(path: Path, limit: int | None) -> list[QueryRow]:
    rows: list[QueryRow] = []
    try:
        with path.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                payload = line.split("|", 1)[-1].strip()
                if not payload:
                    continue
                try:
                    parsed = cast(object, json.loads(payload))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON at {path}:{line_no}: {exc}"
                    ) from exc

                row = _coerce_str_mapping(parsed)
                if row is None:
                    raise RuntimeError(f"JSON object expected at {path}:{line_no}")

                qid = str(row.get("qid") or "").strip()
                if not qid:
                    raise RuntimeError(f"Missing qid at {path}:{line_no}")

                query = str(row.get("query") or "").strip()
                if not query:
                    raise RuntimeError(f"Missing query at {path}:{line_no}")

                rows.append(QueryRow(qid=qid, query=query))
                if limit is not None and len(rows) >= limit:
                    break
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    if not rows:
        raise RuntimeError(f"No queries loaded from {path}")
    return rows


def _build_thread_id(qid: str) -> str:
    digest = hashlib.sha1(qid.encode("utf-8")).hexdigest()[:12]
    return f"rrf-eval-{digest}"


def _extract_doc_subset(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    docs: list[dict[str, object]] = []
    for item in cast(list[object], value):
        doc_map = _coerce_str_mapping(item)
        if doc_map is None:
            continue

        metadata = _coerce_str_mapping(doc_map.get("metadata")) or {}
        metadata_subset: dict[str, object] = {}
        for key in ("doc_id", "page", "source", "file_name", "chunk_id"):
            if key in metadata:
                metadata_subset[key] = metadata[key]
        for key, val in metadata.items():
            if key.startswith("rrf_"):
                metadata_subset[key] = val

        doc_id = doc_map.get("id")
        if doc_id is None:
            doc_id = doc_map.get("doc_id")

        page = doc_map.get("page")
        if page is None:
            page = metadata.get("page")

        docs.append(
            {
                "doc_id": doc_id,
                "page": page,
                "metadata": metadata_subset,
            }
        )
    return docs


def _first_device_name(interrupt_payload: dict[str, object]) -> str | None:
    devices = interrupt_payload.get("devices")
    if not isinstance(devices, list):
        return None
    for item in devices:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Run chat-flow retrieval eval (retrieval interrupt only)"
    )
    _ = parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    _ = parser.add_argument(
        "--queries", dest="input", help="Input JSONL path (alias for --input)"
    )
    _ = parser.add_argument("--input", dest="input", help="Input JSONL path")
    _ = parser.add_argument("--out", required=True, help="Output JSONL path")
    _ = parser.add_argument("--timeout-seconds", type=float, default=120.0)
    _ = parser.add_argument("--limit", type=int, default=None)
    parsed = parser.parse_args()

    timeout_seconds = float(getattr(parsed, "timeout_seconds", 120.0))
    if timeout_seconds <= 0:
        raise RuntimeError("--timeout-seconds must be > 0")

    limit = cast(int | None, getattr(parsed, "limit", None))
    if limit is not None and limit <= 0:
        raise RuntimeError("--limit must be >= 1 when provided")

    input_path = str(getattr(parsed, "input", "")).strip()
    out_path = str(getattr(parsed, "out", "")).strip()
    if not input_path:
        raise RuntimeError("--input is required")
    if not out_path:
        raise RuntimeError("--out is required")

    return CliArgs(
        api_base_url=str(getattr(parsed, "api_base_url", DEFAULT_API_BASE_URL)),
        input_path=input_path,
        out_path=out_path,
        timeout_seconds=timeout_seconds,
        limit=limit,
    )


def main() -> int:
    args = _parse_args()
    input_path = _resolve_input_path(args.input_path)
    out_path = Path(args.out_path)

    if out_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {out_path}")
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_queries(input_path, limit=args.limit)
    run_url = _build_endpoint(args.api_base_url, "/agent/run")

    with out_path.open("w", encoding="utf-8") as out_fp:
        for row in rows:
            thread_id = _build_thread_id(row.qid)
            payload: dict[str, object] = {
                "message": row.query,
                "ask_user_after_retrieve": True,
                "auto_parse": False,
                "guided_confirm": False,
                "use_canonical_retrieval": False,
                "mq_mode": "off",
                "max_attempts": 0,
                "mode": "base",
                "thread_id": thread_id,
            }

            response, elapsed_ms = _post_json(
                run_url,
                payload,
                timeout_seconds=args.timeout_seconds,
            )

            interrupt_payload = (
                _coerce_str_mapping(response.get("interrupt_payload")) or {}
            )
            if (
                bool(response.get("interrupted") is True)
                and interrupt_payload.get("type") == "device_selection"
            ):
                first_device = _first_device_name(interrupt_payload)
                if first_device:
                    resume_payload: dict[str, object] = {
                        "message": row.query,
                        "ask_user_after_retrieve": True,
                        "auto_parse": False,
                        "guided_confirm": False,
                        "use_canonical_retrieval": False,
                        "mq_mode": "off",
                        "max_attempts": 0,
                        "mode": "base",
                        "thread_id": thread_id,
                        "resume_decision": {
                            "type": "device_selection",
                            "selected_devices": [first_device],
                        },
                    }
                    resumed, resumed_elapsed_ms = _post_json(
                        run_url,
                        resume_payload,
                        timeout_seconds=args.timeout_seconds,
                    )
                    response = resumed
                    elapsed_ms += resumed_elapsed_ms

            interrupt_payload = (
                _coerce_str_mapping(response.get("interrupt_payload")) or {}
            )
            metadata = _coerce_str_mapping(response.get("metadata")) or {}
            retrieval_debug = metadata.get("retrieval_debug")

            out_row: dict[str, object] = {
                "qid": row.qid,
                "query": row.query,
                "thread_id": thread_id,
                "elapsed_ms": elapsed_ms,
                "interrupted": bool(response.get("interrupted") is True),
                "interrupt_payload": interrupt_payload if interrupt_payload else None,
                "search_queries": response.get("search_queries"),
                "retrieved_docs": _extract_doc_subset(response.get("retrieved_docs")),
                "metadata": {
                    "retrieval_debug": retrieval_debug,
                },
            }
            _ = out_fp.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_fp.flush()

    print(f"Wrote rows: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
