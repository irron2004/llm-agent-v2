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


def _coerce_str_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    mapped: dict[str, Any] = {}
    for key, item in value.items():
        mapped[str(key)] = item
    return mapped


def _coerce_retry_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return int(text)
            except ValueError:
                return 0
    return 0


def _extract_retry_count(response: dict[str, Any]) -> int:
    trace_top = _coerce_str_mapping(response.get("trace"))
    if trace_top is not None and "retry_count" in trace_top:
        return _coerce_retry_count(trace_top.get("retry_count"))

    metadata = _coerce_str_mapping(response.get("metadata"))
    if metadata is None:
        return _coerce_retry_count(response.get("attempts"))

    trace_meta = _coerce_str_mapping(metadata.get("trace"))
    if trace_meta is not None and "retry_count" in trace_meta:
        return _coerce_retry_count(trace_meta.get("retry_count"))

    if "attempts" in metadata:
        return _coerce_retry_count(metadata.get("attempts"))
    if "retry_count" in metadata:
        return _coerce_retry_count(metadata.get("retry_count"))
    return _coerce_retry_count(response.get("attempts"))


def _build_trace_payload(response: dict[str, Any], retry_count: int) -> dict[str, Any]:
    trace = _coerce_str_mapping(response.get("trace"))
    if trace is None:
        metadata = _coerce_str_mapping(response.get("metadata"))
        trace = (
            _coerce_str_mapping(metadata.get("trace")) if metadata is not None else None
        )
    out = dict(trace or {})
    out["retry_count"] = int(retry_count)
    return out


def _canonicalize_response(
    response: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    retry_count = _extract_retry_count(response)
    trace_payload = _build_trace_payload(response, retry_count)

    cleaned = dict(response)
    metadata = _coerce_str_mapping(cleaned.get("metadata"))
    if metadata is not None:
        metadata = dict(metadata)
        _ = metadata.pop("attempts", None)
        _ = metadata.pop("retry_count", None)
        cleaned["metadata"] = metadata
    _ = cleaned.pop("attempts", None)
    _ = cleaned.pop("retry_count", None)

    return cleaned, trace_payload


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


def _post_json(
    url: str, payload: dict[str, Any], timeout_seconds: float
) -> dict[str, Any]:
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


def _post_json_testclient(
    payload: dict[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    root_dir = Path(__file__).resolve().parents[2]
    root_dir_text = str(root_dir)
    if root_dir_text not in sys.path:
        sys.path.append(root_dir_text)

    from fastapi.testclient import TestClient

    from backend.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/api/agent/run", json=payload, timeout=timeout_seconds)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    parsed = resp.json()
    if not isinstance(parsed, dict):
        raise RuntimeError("Unexpected response shape: object expected")
    mapped: dict[str, Any] = {}
    for key, value in parsed.items():
        mapped[str(key)] = value
    return mapped


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


def _parse_mq_modes(raw: str) -> list[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    out: list[str] = []
    for token in value.split(","):
        mode = token.strip()
        if mode:
            out.append(mode)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SOP page-hit via /api/agent/run"
    )
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
    _ = parser.add_argument(
        "--use-canonical-retrieval", action="store_true", default=False
    )
    _ = parser.add_argument("--use-testclient", action="store_true", default=False)
    _ = parser.add_argument("--max-attempts", type=int, default=0)
    _ = parser.add_argument(
        "--mq-modes",
        default="",
        help="Comma-separated mq modes (for sweep), e.g. off,fallback,on",
    )
    return parser.parse_args()


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _write_mode_report(
    path: Path, summary: dict[str, Any], mq_mode: str | None
) -> None:
    doc_hit = summary.get("doc_hit") if isinstance(summary.get("doc_hit"), dict) else {}
    page_hit = (
        summary.get("page_hit") if isinstance(summary.get("page_hit"), dict) else {}
    )
    lines: list[str] = ["# Agent Eval Report", ""]
    if mq_mode is None:
        lines.append("- mq_mode: (default)")
    else:
        lines.append(f"- mq_mode: `{mq_mode}`")
    lines.append(f"- total: {summary.get('total', '-')}")
    lines.append(f"- failures: {summary.get('failures', '-')}")
    lines.append("")
    lines.append("## Doc Hit")
    for k in (1, 3, 5, 10):
        key = f"@{k}"
        cell = doc_hit.get(key) if isinstance(doc_hit, dict) else {}
        ratio = cell.get("ratio") if isinstance(cell, dict) else None
        count = cell.get("count") if isinstance(cell, dict) else "-"
        lines.append(f"- @{k}: {count} ({_fmt_ratio(ratio)})")
    lines.append("")
    lines.append("## Page Hit")
    for k in (1, 3, 5, 10):
        key = f"@{k}"
        cell = page_hit.get(key) if isinstance(page_hit, dict) else {}
        ratio = cell.get("ratio") if isinstance(cell, dict) else None
        count = cell.get("count") if isinstance(cell, dict) else "-"
        lines.append(f"- @{k}: {count} ({_fmt_ratio(ratio)})")
    _ = path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_mq_ablation_report(
    path: Path, by_mode: list[tuple[str, dict[str, Any]]]
) -> None:
    lines: list[str] = [
        "# MQ Ablation Report",
        "",
        "| mode | failures | doc@1 | doc@3 | doc@5 | doc@10 | page@1 | page@3 | page@5 | page@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in by_mode:
        doc_hit = _coerce_str_mapping(summary.get("doc_hit")) or {}
        page_hit = _coerce_str_mapping(summary.get("page_hit")) or {}

        def _ratio(hit: Any, key: str) -> str:
            mapped = _coerce_str_mapping(hit) or {}
            cell = _coerce_str_mapping(mapped.get(key)) or {}
            value = cell.get("ratio")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return _fmt_ratio(None)
            return _fmt_ratio(float(value))

        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    str(summary.get("failures", "-")),
                    _ratio(doc_hit, "@1"),
                    _ratio(doc_hit, "@3"),
                    _ratio(doc_hit, "@5"),
                    _ratio(doc_hit, "@10"),
                    _ratio(page_hit, "@1"),
                    _ratio(page_hit, "@3"),
                    _ratio(page_hit, "@5"),
                    _ratio(page_hit, "@10"),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append(
        "- Jaccard@k is omitted because this evaluator run does not include a paired baseline needed for overlap computation."
    )
    _ = path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_single_mode(
    args: argparse.Namespace,
    rows: list[EvalRow],
    run_url: str,
    out_dir: Path,
    mq_mode: str | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out = out_dir / "rows.csv"
    raw_out = out_dir / "agent_eval.jsonl"
    summary_out = out_dir / "summary.json"
    report_out = out_dir / "report.md"

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
            if mq_mode is not None:
                payload["mq_mode"] = mq_mode

            error = ""
            response: dict[str, Any] = {}
            try:
                if bool(args.use_testclient):
                    response = _post_json_testclient(
                        payload, timeout_seconds=float(args.timeout_seconds)
                    )
                else:
                    response = _post_json(
                        run_url, payload, timeout_seconds=float(args.timeout_seconds)
                    )
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

            doc_hit_flags = {
                k: _calc_doc_hit(top_docs_obj, row.expected_doc_norm, k)
                for k in doc_hits
            }
            page_hit_flags = {
                k: _calc_page_hit(
                    top_docs_obj, row.expected_doc_norm, row.expected_page_range, k
                )
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
            answer_preview = answer
            if len(answer_preview) > 400:
                answer_preview = answer_preview[:400] + "..."

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
                    _first_page_match_rank(
                        top_docs_obj, row.expected_doc_norm, row.expected_page_range
                    ),
                    top_repr,
                    answer_preview,
                    search_queries_repr,
                ]
            )

            canonical_response, trace_payload = _canonicalize_response(response)

            raw_payload = {
                "idx": row.idx,
                "query": row.query,
                "expected_doc": row.expected_doc,
                "expected_pages": row.expected_pages_raw,
                "request": payload,
                "response": canonical_response,
                "answer": answer,
                "trace": trace_payload,
                "error": error,
            }
            raw_fp.write(json.dumps(raw_payload, ensure_ascii=False) + "\n")

            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))

    summary = {
        "total": total,
        "failures": failures,
        "env": _capture_env(AGENT_ENV_CAPTURE_KEYS),
        "doc_hit": {
            f"@{k}": {"count": doc_hits[k], "ratio": doc_hits[k] / total}
            for k in sorted(doc_hits)
        },
        "page_hit": {
            f"@{k}": {"count": page_hits[k], "ratio": page_hits[k] / total}
            for k in sorted(page_hits)
        },
    }

    summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_mode_report(report_out, summary, mq_mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"rows: {rows_out}")
    print(f"raw: {raw_out}")
    print(f"summary: {summary_out}")
    print(f"report: {report_out}")
    return summary


def _calc_page_hit(
    top_docs: list[dict[str, Any]],
    expected_doc_norm: str,
    expected_range: tuple[int, int] | None,
    k: int,
) -> bool:
    for doc in top_docs[:k]:
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id != expected_doc_norm:
            continue
        if _in_range(_extract_page(doc), expected_range):
            return True
    return False


def _calc_doc_hit(
    top_docs: list[dict[str, Any]], expected_doc_norm: str, k: int
) -> bool:
    for doc in top_docs[:k]:
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id == expected_doc_norm:
            return True
    return False


def _first_match_rank(
    top_docs: list[dict[str, Any]], expected_doc_norm: str
) -> int | None:
    for idx, doc in enumerate(top_docs, start=1):
        doc_id = _normalize_doc_id(str(doc.get("id") or ""))
        if doc_id == expected_doc_norm:
            return idx
    return None


def _first_page_match_rank(
    top_docs: list[dict[str, Any]],
    expected_doc_norm: str,
    expected_range: tuple[int, int] | None,
) -> int | None:
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
    mq_modes = _parse_mq_modes(args.mq_modes)

    if not mq_modes:
        _ = _run_single_mode(args, rows, run_url, out_dir, mq_mode=None)
        return 0

    mode_summaries: list[tuple[str, dict[str, Any]]] = []
    for mode in mq_modes:
        mode_out_dir = out_dir / f"mq_{mode}"
        summary = _run_single_mode(args, rows, run_url, mode_out_dir, mq_mode=mode)
        mode_summaries.append((mode, summary))

    ablation_out = out_dir / "mq_ablation_report.md"
    _write_mq_ablation_report(ablation_out, mode_summaries)
    print(f"mq_ablation_report: {ablation_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
