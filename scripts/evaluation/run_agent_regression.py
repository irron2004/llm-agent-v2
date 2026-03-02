#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast


DEFAULT_API_BASE_URL = "http://localhost:8011"


@dataclass(frozen=True)
class CliArgs:
    api_base_url: str
    queries: str
    out_dir: str
    timeout_seconds: float
    top_k: int
    max_attempts: int
    mode: str
    auto_parse: bool
    use_canonical_retrieval: bool
    limit: int | None


@dataclass(frozen=True)
class QueryRow:
    qid: str
    group_id: str
    query: str


def _coerce_str_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    mapped: dict[str, object] = {}
    for key, item in raw.items():
        mapped[str(key)] = item
    return mapped


def _parse_bool_text(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value: {value}")


def _build_endpoint(base_url: str, path: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/api"):
        return f"{stripped}{path}"
    return f"{stripped}/api{path}"


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

                query = str(
                    row.get("query") or row.get("canonical_query") or ""
                ).strip()
                if not query:
                    raise RuntimeError(f"Missing query at {path}:{line_no}")
                qid = str(row.get("qid") or f"line_{line_no:06d}").strip()
                group_id = str(row.get("group_id") or "").strip()

                rows.append(QueryRow(qid=qid, group_id=group_id, query=query))
                if limit is not None and len(rows) >= limit:
                    break
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    if not rows:
        raise RuntimeError(f"No queries loaded from {path}")
    return rows


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


def _iter_sse_events(response: http.client.HTTPResponse) -> Iterable[dict[str, object]]:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip("\r\n")
        if not line:
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if not payload:
            continue
        try:
            parsed = cast(object, json.loads(payload))
        except json.JSONDecodeError:
            continue
        mapped = _coerce_str_mapping(parsed)
        if mapped is None:
            continue
        yield mapped


def _sanitize_agent_response_for_evidence(resp: dict[str, object]) -> dict[str, object]:
    cleaned = dict(resp)
    if "expanded_docs" in cleaned:
        cleaned["expanded_docs"] = None
    return cleaned


def _sanitize_stream_event_for_evidence(event: dict[str, object]) -> dict[str, object]:
    cleaned = dict(event)
    if cleaned.get("type") == "final":
        result = cleaned.get("result")
        result_map = _coerce_str_mapping(result)
        if result_map is not None:
            cleaned["result"] = _sanitize_agent_response_for_evidence(result_map)
    return cleaned


def _extract_retrieved_doc_ids(agent_response: dict[str, object]) -> list[str]:
    docs_raw = agent_response.get("retrieved_docs")
    if not isinstance(docs_raw, list):
        return []
    ids: list[str] = []
    for item in cast(list[object], docs_raw):
        doc_map = _coerce_str_mapping(item)
        if doc_map is None:
            continue
        doc_id = doc_map.get("id")
        if isinstance(doc_id, str):
            value = doc_id.strip()
        else:
            value = str(doc_id).strip() if doc_id is not None else ""
        if value:
            ids.append(value)
    return ids


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Run agent regression capture")
    _ = parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    _ = parser.add_argument("--queries", required=True)
    _ = parser.add_argument("--out-dir", required=True)
    _ = parser.add_argument("--timeout-seconds", type=float, default=120.0)
    _ = parser.add_argument("--top-k", type=int, default=10)
    _ = parser.add_argument("--max-attempts", type=int, default=3)
    _ = parser.add_argument("--mode", default="verified")
    _ = parser.add_argument("--auto-parse", default="true")
    _ = parser.add_argument("--use-canonical-retrieval", default="true")
    _ = parser.add_argument("--limit", type=int, default=None)
    parsed = parser.parse_args()

    timeout_seconds = float(getattr(parsed, "timeout_seconds", 120.0))
    if timeout_seconds <= 0:
        raise RuntimeError("--timeout-seconds must be > 0")
    top_k = int(getattr(parsed, "top_k", 10))
    if top_k <= 0:
        raise RuntimeError("--top-k must be >= 1")
    max_attempts = int(getattr(parsed, "max_attempts", 3))
    if max_attempts < 0:
        raise RuntimeError("--max-attempts must be >= 0")
    mode = str(getattr(parsed, "mode", "verified")).strip() or "verified"
    auto_parse = _parse_bool_text(str(getattr(parsed, "auto_parse", "true")))
    use_canonical_retrieval = _parse_bool_text(
        str(getattr(parsed, "use_canonical_retrieval", "true"))
    )
    limit = cast(int | None, getattr(parsed, "limit", None))
    if limit is not None and limit <= 0:
        raise RuntimeError("--limit must be >= 1 when provided")

    return CliArgs(
        api_base_url=str(getattr(parsed, "api_base_url", DEFAULT_API_BASE_URL)),
        queries=str(getattr(parsed, "queries", "")),
        out_dir=str(getattr(parsed, "out_dir", "")),
        timeout_seconds=timeout_seconds,
        top_k=top_k,
        max_attempts=max_attempts,
        mode=mode,
        auto_parse=auto_parse,
        use_canonical_retrieval=use_canonical_retrieval,
        limit=limit,
    )


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_out = out_dir / "run.jsonl"
    stream_out = out_dir / "stream_events.ndjson"
    errors_out = out_dir / "errors.jsonl"

    for path in (run_out, stream_out, errors_out):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing file: {path}")

    query_rows = _load_queries(Path(args.queries), limit=args.limit)
    run_url = _build_endpoint(args.api_base_url, "/agent/run")
    stream_url = _build_endpoint(args.api_base_url, "/agent/run/stream")

    with (
        run_out.open("w", encoding="utf-8") as run_fp,
        stream_out.open("w", encoding="utf-8") as stream_fp,
        errors_out.open("w", encoding="utf-8") as err_fp,
    ):
        for row in query_rows:
            payload: dict[str, object] = {
                "message": row.query,
                "mode": args.mode,
                "auto_parse": bool(args.auto_parse),
                "max_attempts": int(args.max_attempts),
                "use_canonical_retrieval": bool(args.use_canonical_retrieval),
                "top_k": int(args.top_k),
            }

            run_record: dict[str, object] = {
                "qid": row.qid,
                "group_id": row.group_id,
                "query": row.query,
                "run": None,
                "stream": None,
                "error": None,
            }

            try:
                body, elapsed_ms = _post_json(
                    run_url,
                    payload,
                    timeout_seconds=args.timeout_seconds,
                )
                run_record["run"] = {
                    "elapsed_ms": elapsed_ms,
                    "thread_id": body.get("thread_id"),
                    "metadata": body.get("metadata", {}),
                    "judge": body.get("judge", {}),
                    "answer": body.get("answer", ""),
                    "search_queries": body.get("search_queries"),
                    "retrieved_doc_ids": _extract_retrieved_doc_ids(body),
                }
            except Exception as exc:
                run_record["error"] = {"phase": "run", "detail": str(exc)}
                _ = err_fp.write(json.dumps(run_record, ensure_ascii=False) + "\n")
                err_fp.flush()
                continue

            try:
                body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    stream_url,
                    data=body_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                started = time.perf_counter()
                with cast(
                    http.client.HTTPResponse,
                    urllib.request.urlopen(req, timeout=args.timeout_seconds),
                ) as resp:
                    events: list[dict[str, object]] = []
                    for event in _iter_sse_events(resp):
                        enriched = dict(event)
                        enriched["qid"] = row.qid
                        enriched["group_id"] = row.group_id
                        enriched["query"] = row.query
                        sanitized = _sanitize_stream_event_for_evidence(enriched)
                        _ = stream_fp.write(
                            json.dumps(sanitized, ensure_ascii=False) + "\n"
                        )
                        stream_fp.flush()
                        events.append(event)
                stream_elapsed_ms = (time.perf_counter() - started) * 1000.0

                type_counts: dict[str, int] = {}
                node_end_sequence: list[str] = []
                final_present = False
                thread_id: str | None = None
                trace_id: str | None = None
                for event in events:
                    ev_type = str(event.get("type") or "")
                    if ev_type:
                        type_counts[ev_type] = type_counts.get(ev_type, 0) + 1
                    if ev_type == "open":
                        thread_id_raw = event.get("thread_id")
                        if isinstance(thread_id_raw, str):
                            thread_id = thread_id_raw
                        trace = _coerce_str_mapping(event.get("trace"))
                        if trace is not None:
                            trace_id_raw = trace.get("trace_id")
                            if isinstance(trace_id_raw, str):
                                trace_id = trace_id_raw
                    if ev_type == "node_end":
                        node_raw = event.get("node")
                        if isinstance(node_raw, str) and node_raw.strip():
                            node_end_sequence.append(node_raw.strip())
                    if ev_type == "final":
                        final_present = True

                run_record["stream"] = {
                    "elapsed_ms": stream_elapsed_ms,
                    "final_present": final_present,
                    "type_counts": type_counts,
                    "node_end_sequence": node_end_sequence,
                    "thread_id": thread_id,
                    "trace_id": trace_id,
                }
            except Exception as exc:
                run_record["error"] = {"phase": "stream", "detail": str(exc)}
                _ = err_fp.write(json.dumps(run_record, ensure_ascii=False) + "\n")
                err_fp.flush()
                continue

            _ = run_fp.write(json.dumps(run_record, ensure_ascii=False) + "\n")
            run_fp.flush()

    print(f"Wrote agent run rows: {run_out}")
    print(f"Wrote stream events: {stream_out}")
    print(f"Wrote errors: {errors_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
