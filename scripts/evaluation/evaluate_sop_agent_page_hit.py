#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AGENT_ENV_CAPTURE_KEYS = (
    "AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED",
    "AGENT_EARLY_PAGE_PENALTY_ENABLED",
    "AGENT_EARLY_PAGE_PENALTY_MAX_PAGE",
    "AGENT_EARLY_PAGE_PENALTY_FACTOR",
    "AGENT_SECOND_STAGE_MAX_DOC_IDS",
    "AGENT_SECOND_STAGE_TOP_K",
)


def _capture_env(keys: tuple[str, ...]) -> dict[str, str]:
    return {key: os.environ.get(key, "") for key in keys}


def _normalize_doc_id(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\.(pdf|docx|doc|txt)$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _parse_page_range(raw: str) -> tuple[int, int] | None:
    value = str(raw or "").strip()
    if not value:
        return None
    if "-" in value:
        left, right = value.split("-", 1)
        try:
            a = int(left.strip())
            b = int(right.strip())
        except ValueError:
            return None
        if a <= b:
            return (a, b)
        return (b, a)
    try:
        page = int(value)
    except ValueError:
        return None
    return (page, page)


def _in_range(page: int | None, expected: tuple[int, int] | None) -> bool:
    if page is None or expected is None:
        return False
    return expected[0] <= page <= expected[1]


def _extract_page(doc: dict[str, Any]) -> int | None:
    page_raw = doc.get("page")
    if isinstance(page_raw, int):
        return page_raw
    if isinstance(page_raw, str):
        s = page_raw.strip()
        if s.isdigit():
            return int(s)
    return None


def _build_url(base_url: str, path: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/api"):
        return f"{stripped}{path}"
    return f"{stripped}/api{path}"


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {exc}") from exc

    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("Unexpected response shape: object expected")
    return parsed


@dataclass
class EvalRow:
    idx: int
    query: str
    expected_doc: str
    expected_doc_norm: str
    expected_pages_raw: str
    expected_page_range: tuple[int, int] | None


def _load_rows(path: Path, limit: int | None) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open(encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for i, row in enumerate(reader, start=1):
            query = str(row.get("질문내용") or "").strip()
            expected_doc = str(row.get("정답문서") or "").strip()
            expected_pages = str(row.get("정답 페이지") or "").strip()
            if not query or not expected_doc:
                continue
            rows.append(
                EvalRow(
                    idx=i,
                    query=query,
                    expected_doc=expected_doc,
                    expected_doc_norm=_normalize_doc_id(expected_doc),
                    expected_pages_raw=expected_pages,
                    expected_page_range=_parse_page_range(expected_pages),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No valid rows found in {path}")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SOP page-hit via /api/agent/run")
    _ = parser.add_argument("--api-base-url", default="http://127.0.0.1:18021")
    _ = parser.add_argument(
        "--csv-path",
        default="data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv",
    )
    _ = parser.add_argument("--out-dir", required=True)
    _ = parser.add_argument("--top-k", type=int, default=10)
    _ = parser.add_argument("--timeout-seconds", type=float, default=120.0)
    _ = parser.add_argument("--sleep-seconds", type=float, default=0.0)
    _ = parser.add_argument("--limit", type=int, default=None)
    _ = parser.add_argument("--auto-parse", action="store_true", default=True)
    _ = parser.add_argument("--no-auto-parse", action="store_false", dest="auto_parse")
    _ = parser.add_argument("--use-canonical-retrieval", action="store_true", default=False)
    _ = parser.add_argument("--max-attempts", type=int, default=0)
    return parser.parse_args()


def _calc_page_hit(top_docs: list[dict[str, Any]], expected_doc_norm: str, expected_range: tuple[int, int] | None, k: int) -> bool:
    for doc in top_docs[:k]:
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id != expected_doc_norm:
            continue
        if _in_range(_extract_page(doc), expected_range):
            return True
    return False


def _calc_doc_hit(top_docs: list[dict[str, Any]], expected_doc_norm: str, k: int) -> bool:
    for doc in top_docs[:k]:
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id == expected_doc_norm:
            return True
    return False


def _first_match_rank(top_docs: list[dict[str, Any]], expected_doc_norm: str) -> int | None:
    for idx, doc in enumerate(top_docs, start=1):
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id == expected_doc_norm:
            return idx
    return None


def _first_page_match_rank(top_docs: list[dict[str, Any]], expected_doc_norm: str, expected_range: tuple[int, int] | None) -> int | None:
    for idx, doc in enumerate(top_docs, start=1):
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id != expected_doc_norm:
            continue
        if _in_range(_extract_page(doc), expected_range):
            return idx
    return None


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(Path(args.csv_path), args.limit)
    run_url = _build_url(args.api_base_url, "/agent/run")

    rows_out = out_dir / "rows.csv"
    raw_out = out_dir / "raw.jsonl"
    summary_out = out_dir / "summary.json"

    total = len(rows)
    page_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    doc_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    failures = 0

    with (
        rows_out.open("w", encoding="utf-8", newline="") as rows_fp,
        raw_out.open("w", encoding="utf-8") as raw_fp,
    ):
        writer = csv.writer(rows_fp)
        writer.writerow(
            [
                "idx",
                "query",
                "expected_doc",
                "expected_pages",
                "error",
                "doc_hit@1",
                "doc_hit@3",
                "doc_hit@5",
                "doc_hit@10",
                "page_hit@1",
                "page_hit@3",
                "page_hit@5",
                "page_hit@10",
                "doc_match_rank",
                "page_match_rank",
                "top_docs",
                "answer",
                "search_queries",
            ]
        )

        for row in rows:
            payload: dict[str, Any] = {
                "message": row.query,
                "top_k": int(args.top_k),
                "mode": "verified",
                "auto_parse": bool(args.auto_parse),
                "use_canonical_retrieval": bool(args.use_canonical_retrieval),
                "max_attempts": int(args.max_attempts),
            }

            error = ""
            response: dict[str, Any] = {}
            try:
                response = _post_json(run_url, payload, timeout_seconds=float(args.timeout_seconds))
            except Exception as exc:  # noqa: BLE001
                failures += 1
                error = str(exc)

            top_docs = response.get("retrieved_docs")
            if not isinstance(top_docs, list):
                top_docs = []

            top_docs_obj: list[dict[str, Any]] = []
            for doc in top_docs:
                if isinstance(doc, dict):
                    top_docs_obj.append(doc)

            doc_hit_flags = {k: _calc_doc_hit(top_docs_obj, row.expected_doc_norm, k) for k in doc_hits}
            page_hit_flags = {
                k: _calc_page_hit(top_docs_obj, row.expected_doc_norm, row.expected_page_range, k)
                for k in page_hits
            }
            for k, ok in doc_hit_flags.items():
                if ok:
                    doc_hits[k] += 1
            for k, ok in page_hit_flags.items():
                if ok:
                    page_hits[k] += 1

            top_repr = " | ".join(
                f"{_normalize_doc_id(str(doc.get('id') or ''))}(p{_extract_page(doc)})"
                for doc in top_docs_obj[:5]
            )
            answer = str(response.get("answer") or "").replace("\n", " ").strip()
            if len(answer) > 400:
                answer = answer[:400] + "..."

            search_queries = response.get("search_queries")
            search_queries_repr = ""
            if isinstance(search_queries, list):
                search_queries_repr = " | ".join(str(q) for q in search_queries)

            writer.writerow(
                [
                    row.idx,
                    row.query,
                    row.expected_doc,
                    row.expected_pages_raw,
                    error,
                    doc_hit_flags[1],
                    doc_hit_flags[3],
                    doc_hit_flags[5],
                    doc_hit_flags[10],
                    page_hit_flags[1],
                    page_hit_flags[3],
                    page_hit_flags[5],
                    page_hit_flags[10],
                    _first_match_rank(top_docs_obj, row.expected_doc_norm),
                    _first_page_match_rank(top_docs_obj, row.expected_doc_norm, row.expected_page_range),
                    top_repr,
                    answer,
                    search_queries_repr,
                ]
            )

            raw_payload = {
                "idx": row.idx,
                "query": row.query,
                "expected_doc": row.expected_doc,
                "expected_pages": row.expected_pages_raw,
                "request": payload,
                "response": response,
                "error": error,
            }
            raw_fp.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")

            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))

    summary = {
        "total": total,
        "failures": failures,
        "env": _capture_env(AGENT_ENV_CAPTURE_KEYS),
        "doc_hit": {f"@{k}": {"count": doc_hits[k], "ratio": doc_hits[k] / total} for k in sorted(doc_hits)},
        "page_hit": {f"@{k}": {"count": page_hits[k], "ratio": page_hits[k] / total} for k in sorted(page_hits)},
    }

    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"rows: {rows_out}")
    print(f"raw: {raw_out}")
    print(f"summary: {summary_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
