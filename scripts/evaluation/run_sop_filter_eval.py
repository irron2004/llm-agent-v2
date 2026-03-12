#!/usr/bin/env python3
"""SOP filter evaluation: compare SOP-only vs 절차검색(SOP+setup) filter results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

DEFAULT_API_BASE_URL = "http://localhost:8001"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_ANSWER_PREVIEW_CHARS = 2000
DEFAULT_MAX_RETRIEVED_DOCS = 10
SCHEMA_VERSION = "sop_eval_v1"


@dataclass
class GoldRow:
    idx: int
    question: str
    device: str
    doc_type: str
    gold_doc: str
    gold_pages: str
    note: str


@dataclass
class EvalResult:
    idx: int
    question: str
    filter_label: str
    filter_doc_types: list[str]
    request_payload: dict[str, Any]
    response_metadata: dict[str, Any]
    gold_doc: str
    gold_pages: str
    retrieved_docs: list[dict[str, Any]]
    answer: str
    hit_gold_doc: bool
    hit_gold_page: bool
    hit_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    match_debug: dict[str, int | str | None]
    elapsed_ms: float
    error: str | None = None


def _post_json(
    url: str, payload: dict[str, Any], *, timeout: float
) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    try:
        with cast(
            http.client.HTTPResponse,
            urllib.request.urlopen(req, timeout=timeout),
        ) as resp:
            raw = resp.read()
    except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return {"error": str(exc)}, elapsed

    elapsed = (time.perf_counter() - started) * 1000
    try:
        return json.loads(raw.decode("utf-8")), elapsed
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}, elapsed


def _post_json_testclient(
    payload: dict[str, Any], *, timeout: float
) -> tuple[dict[str, Any], float]:
    root_dir = Path(__file__).resolve().parents[2]
    root_dir_text = str(root_dir)
    if root_dir_text not in sys.path:
        sys.path.append(root_dir_text)

    from fastapi.testclient import TestClient

    from backend.api.main import create_app

    started = time.perf_counter()
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/api/agent/run", json=payload, timeout=timeout)
    elapsed = (time.perf_counter() - started) * 1000
    if resp.status_code >= 400:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}, elapsed

    parsed = resp.json()
    if not isinstance(parsed, dict):
        return {"error": "Unexpected response shape: object expected"}, elapsed

    mapped: dict[str, Any] = {}
    for key, value in parsed.items():
        mapped[str(key)] = value
    return mapped, elapsed


def _build_testclient_poster() -> tuple[
    Callable[[dict[str, Any], float], tuple[dict[str, Any], float]],
    Callable[[], None],
]:
    root_dir = Path(__file__).resolve().parents[2]
    root_dir_text = str(root_dir)
    if root_dir_text not in sys.path:
        sys.path.append(root_dir_text)

    from fastapi.testclient import TestClient

    from backend.api.main import create_app

    client = TestClient(create_app())

    def _post(payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            resp = client.post("/api/agent/run", json=payload, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - started) * 1000
            return {"error": str(exc)}, elapsed
        elapsed = (time.perf_counter() - started) * 1000
        if resp.status_code >= 400:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}, elapsed

        parsed = resp.json()
        if not isinstance(parsed, dict):
            return {"error": "Unexpected response shape: object expected"}, elapsed

        mapped: dict[str, Any] = {}
        for key, value in parsed.items():
            mapped[str(key)] = value
        return mapped, elapsed

    return _post, client.close


def _build_url(base_url: str, path: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/api"):
        return f"{stripped}{path}"
    return f"{stripped}/api{path}"


def _coerce_doc_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(cast(dict[str, Any], item))
    return out


def _coerce_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        out = value.strip()
        return out or None
    out = str(value).strip()
    return out or None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _extract_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(_coerce_mapping(response.get("metadata")))
    for key in (
        "route",
        "search_queries",
        "detected_language",
        "target_language",
        "template_version",
    ):
        if key not in metadata and key in response:
            metadata[key] = response.get(key)
    return metadata


def _build_top_docs(
    retrieved_docs: list[dict[str, Any]], max_docs: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, doc in enumerate(retrieved_docs[: max(max_docs, 0)], start=1):
        metadata = _coerce_mapping(doc.get("metadata"))
        out.append(
            {
                "rank": rank,
                "doc_id": _safe_str(doc.get("doc_id") or doc.get("id")),
                "title": _safe_str(doc.get("title")),
                "source": _safe_str(
                    doc.get("source")
                    or metadata.get("source")
                    or metadata.get("file_name")
                ),
                "page": _safe_parse_page(doc.get("page") or metadata.get("page")),
                "score": _safe_float(doc.get("score") or metadata.get("score")),
                "doc_type": _safe_str(
                    doc.get("doc_type")
                    or metadata.get("doc_type")
                    or metadata.get("type")
                ),
                "device_name": _safe_str(
                    doc.get("device_name")
                    or metadata.get("device_name")
                    or metadata.get("device")
                ),
                "chunk_id": _safe_str(
                    doc.get("chunk_id")
                    or metadata.get("chunk_id")
                    or metadata.get("chunk")
                ),
            }
        )
    return out


def _thread_id(idx: int, label: str) -> str:
    h = hashlib.sha1(f"sop-eval-{idx}-{label}".encode()).hexdigest()[:12]
    return f"sop-eval-{h}"


def _normalize_doc_name(name: str) -> str:
    """Normalize doc name for comparison."""
    normalized = str(name or "").lower().strip()
    normalized = re.sub(r"\.(pdf|docx|doc|txt)$", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _safe_parse_page(page: object) -> int | None:
    if isinstance(page, bool):
        return None
    if isinstance(page, int):
        return page
    if isinstance(page, float):
        return int(page) if page.is_integer() else None
    if isinstance(page, str):
        text = page.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _parse_page_range(pages_str: str) -> set[int]:
    """Parse '6-14' or '28-45' into set of page numbers."""
    pages = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                for p in range(int(start), int(end) + 1):
                    pages.add(p)
            except ValueError:
                pass
        elif part.isdigit():
            pages.add(int(part))
    return pages


def _check_hit(
    retrieved_docs: list[dict[str, Any]], gold_doc: str, gold_pages: str
) -> tuple[bool, bool, int | None, dict[str, int | str | None]]:
    """Check if gold doc and page are in retrieved results."""
    gold_doc_norm = _normalize_doc_name(gold_doc)
    if not gold_doc_norm:
        return (
            False,
            False,
            None,
            {"matched_field": None, "matched_value": None, "rank": None},
        )

    gold_page_set = _parse_page_range(gold_pages)

    first_match_rank: int | None = None
    page_match_found = False
    match_debug: dict[str, int | str | None] = {
        "matched_field": None,
        "matched_value": None,
        "rank": None,
    }

    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_id = str(doc.get("doc_id") or doc.get("id") or "")
        title = str(doc.get("title") or "")
        metadata = doc.get("metadata")
        source = ""
        if isinstance(metadata, dict):
            source = str(metadata.get("source") or metadata.get("file_name") or "")

        candidates = [
            ("doc_id", doc_id),
            ("title", title),
            ("source", source),
        ]
        for field_name, candidate in candidates:
            candidate_norm = _normalize_doc_name(candidate)
            if candidate_norm and gold_doc_norm in candidate_norm:
                if first_match_rank is None:
                    first_match_rank = rank
                    match_debug = {
                        "matched_field": field_name,
                        "matched_value": candidate,
                        "rank": rank,
                    }

                parsed_page = _safe_parse_page(doc.get("page"))
                if parsed_page is not None and parsed_page in gold_page_set:
                    page_match_found = True
                break

        if page_match_found:
            break

    if first_match_rank is None:
        return False, False, None, match_debug
    return True, page_match_found, first_match_rank, match_debug


def run_eval_single(
    row: GoldRow,
    filter_label: str,
    filter_doc_types: list[str],
    *,
    api_base_url: str,
    timeout_seconds: float,
    post_json_testclient: Callable[
        [dict[str, Any], float], tuple[dict[str, Any], float]
    ]
    | None,
) -> EvalResult:
    """Run single evaluation query."""
    thread_id = _thread_id(row.idx, filter_label)
    run_url = _build_url(api_base_url, "/agent/run")

    # SUPRA XP와 ZEDIUS XP는 같은 장비 (ES에 ZEDIUS XP로 색인됨)
    DEVICE_ALIASES = {
        "SUPRA XP": ["SUPRA XP", "ZEDIUS XP"],
    }
    devices = DEVICE_ALIASES.get(row.device, [row.device]) if row.device else []

    payload: dict[str, Any] = {
        "message": row.question,
        "thread_id": thread_id,
        "auto_parse": False,
        "guided_confirm": False,
        "filter_doc_types": filter_doc_types,
        "filter_devices": devices,
        "target_language": "ko",
    }

    sent_payloads: list[dict[str, Any]] = [dict(payload)]
    if post_json_testclient is not None:
        response, elapsed = post_json_testclient(payload, timeout_seconds)
    else:
        response, elapsed = _post_json(run_url, payload, timeout=timeout_seconds)
    final_response = response

    if "error" in response and isinstance(response["error"], str):
        return EvalResult(
            idx=row.idx,
            question=row.question,
            filter_label=filter_label,
            filter_doc_types=filter_doc_types,
            request_payload={"requests": sent_payloads},
            response_metadata=_extract_response_metadata(final_response),
            gold_doc=row.gold_doc,
            gold_pages=row.gold_pages,
            retrieved_docs=[],
            answer="",
            hit_gold_doc=False,
            hit_gold_page=False,
            hit_rank=None,
            hit_at_1=False,
            hit_at_3=False,
            hit_at_5=False,
            hit_at_10=False,
            match_debug={"matched_field": None, "matched_value": None, "rank": None},
            elapsed_ms=elapsed,
            error=response["error"],
        )

    # Handle interrupt (retrieval review)
    interrupted = bool(response.get("interrupted", False))
    interrupt_payload = _coerce_mapping(response.get("interrupt_payload"))

    if interrupted and interrupt_payload.get("type") == "retrieval_review":
        retrieved_docs = _coerce_doc_list(interrupt_payload.get("docs"))
        # Resume to get answer - accept all docs
        doc_ids = [
            d.get("doc_id") or d.get("id")
            for d in retrieved_docs
            if d.get("doc_id") or d.get("id")
        ]
        resume_payload: dict[str, Any] = {
            "message": row.question,
            "thread_id": thread_id,
            "auto_parse": False,
            "guided_confirm": False,
            "filter_doc_types": filter_doc_types,
            "filter_devices": [row.device] if row.device else [],
            "target_language": "ko",
            "resume_action": "approve_docs",
            "resume_payload": {"doc_ids": doc_ids, "ranks": list(range(len(doc_ids)))},
        }
        sent_payloads.append(dict(resume_payload))
        if post_json_testclient is not None:
            resp2, elapsed2 = post_json_testclient(resume_payload, timeout_seconds)
        else:
            resp2, elapsed2 = _post_json(
                run_url, resume_payload, timeout=timeout_seconds
            )
        elapsed += elapsed2
        final_response = resp2
        answer = str(resp2.get("answer") or resp2.get("response") or "")
    elif interrupted and interrupt_payload.get("type") == "device_selection":
        # Device selection interrupt - use first device and retry
        first_device = row.device
        devices = interrupt_payload.get("devices")
        if isinstance(devices, list) and devices:
            first = devices[0]
            if isinstance(first, dict):
                selected_name = first.get("name")
                if isinstance(selected_name, str) and selected_name:
                    first_device = selected_name
        retry_payload: dict[str, Any] = {
            "message": row.question,
            "thread_id": thread_id,
            "auto_parse": False,
            "guided_confirm": False,
            "filter_doc_types": filter_doc_types,
            "filter_devices": [first_device],
            "target_language": "ko",
            "resume_action": "select_device",
            "resume_payload": {"device_name": first_device},
        }
        sent_payloads.append(dict(retry_payload))
        if post_json_testclient is not None:
            resp2, elapsed2 = post_json_testclient(retry_payload, timeout_seconds)
        else:
            resp2, elapsed2 = _post_json(
                run_url, retry_payload, timeout=timeout_seconds
            )
        elapsed += elapsed2
        final_response = resp2
        interrupt_payload2 = _coerce_mapping(resp2.get("interrupt_payload"))
        retrieved_docs = _coerce_doc_list(interrupt_payload2.get("docs"))
        answer = str(resp2.get("answer") or resp2.get("response") or "")
    else:
        retrieved_docs = _coerce_doc_list(response.get("retrieved_docs"))
        answer = str(response.get("answer") or response.get("response") or "")

    hit_doc, hit_page, hit_rank, match_debug = _check_hit(
        retrieved_docs, row.gold_doc, row.gold_pages
    )

    return EvalResult(
        idx=row.idx,
        question=row.question,
        filter_label=filter_label,
        filter_doc_types=filter_doc_types,
        request_payload={"requests": sent_payloads},
        response_metadata=_extract_response_metadata(final_response),
        gold_doc=row.gold_doc,
        gold_pages=row.gold_pages,
        retrieved_docs=retrieved_docs,
        answer=answer,
        hit_gold_doc=hit_doc,
        hit_gold_page=hit_page,
        hit_rank=hit_rank,
        hit_at_1=hit_rank is not None and hit_rank <= 1,
        hit_at_3=hit_rank is not None and hit_rank <= 3,
        hit_at_5=hit_rank is not None and hit_rank <= 5,
        hit_at_10=hit_rank is not None and hit_rank <= 10,
        match_debug=match_debug,
        elapsed_ms=elapsed,
    )


def load_csv(path: str) -> list[GoldRow]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(
                GoldRow(
                    idx=idx,
                    question=row["질문내용"].strip(),
                    device=row.get("장비", "").strip(),
                    doc_type=row.get("문서종류", "").strip(),
                    gold_doc=row.get("정답문서", "").strip(),
                    gold_pages=row.get("정답 페이지", "").strip(),
                    note=row.get("특이점", "").strip(),
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run SOP filter evaluation")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--answer-preview-chars", type=int, default=DEFAULT_ANSWER_PREVIEW_CHARS
    )
    parser.add_argument(
        "--max-retrieved-docs", type=int, default=DEFAULT_MAX_RETRIEVED_DOCS
    )
    parser.add_argument("--use-testclient", action="store_true", default=False)
    parser.add_argument(
        "--csv-path", default="data/eval_sop_question_list_박진우_변형.csv"
    )
    parser.add_argument(
        "--out-dir", default=".sisyphus/evidence/2026-03-11_sop_filter_eval"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validate", action="store_true", default=False)
    args = parser.parse_args()

    csv_path = args.csv_path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} questions")

    configs = [
        ("sop_only", ["SOP"]),
        ("절차검색", ["SOP", "set_up_manual"]),
    ]

    all_results: dict[str, list[EvalResult]] = {}

    post_json_testclient: (
        Callable[[dict[str, Any], float], tuple[dict[str, Any], float]] | None
    ) = None
    close_testclient: Callable[[], None] | None = None
    if args.use_testclient:
        post_json_testclient, close_testclient = _build_testclient_poster()

    try:
        for label, doc_types in configs:
            print(f"\n{'=' * 60}")
            print(f"Running: {label} (filter_doc_types={doc_types})")
            print(f"{'=' * 60}")
            results = []

            for i, row in enumerate(rows):
                print(
                    f"  [{i + 1}/{len(rows)}] {row.question[:50]}...",
                    end=" ",
                    flush=True,
                )
                result = run_eval_single(
                    row,
                    label,
                    doc_types,
                    api_base_url=args.api_base_url,
                    timeout_seconds=float(args.timeout_seconds),
                    post_json_testclient=post_json_testclient,
                )
                results.append(result)

                status = "✓" if result.hit_gold_doc else "✗"
                page_status = "p✓" if result.hit_gold_page else "p✗"
                n_docs = len(result.retrieved_docs)
                print(
                    f"{status} {page_status} docs={n_docs} {result.elapsed_ms:.0f}ms"
                    + (f" ERR:{result.error[:30]}" if result.error else "")
                )

            all_results[label] = results

            # Save per-config results
            thin_path = out_dir / f"{label}_results.jsonl"
            rich_path = out_dir / f"{label}_results.rich.jsonl"
            answer_preview_chars = max(int(args.answer_preview_chars), 0)
            max_retrieved_docs = max(int(args.max_retrieved_docs), 0)

            with thin_path.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(
                        json.dumps(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "idx": r.idx,
                                "question": r.question,
                                "filter": r.filter_label,
                                "filter_doc_types": r.filter_doc_types,
                                "gold_doc": r.gold_doc,
                                "gold_pages": r.gold_pages,
                                "hit_doc": r.hit_gold_doc,
                                "hit_page": r.hit_gold_page,
                                "hit_rank": r.hit_rank,
                                "hit_at_1": r.hit_at_1,
                                "hit_at_3": r.hit_at_3,
                                "hit_at_5": r.hit_at_5,
                                "hit_at_10": r.hit_at_10,
                                "match_debug": r.match_debug,
                                "n_docs": len(r.retrieved_docs),
                                "elapsed_ms": r.elapsed_ms,
                                "error": r.error,
                                "retrieved_doc_ids": [
                                    d.get("doc_id") or d.get("id")
                                    for d in r.retrieved_docs[:max_retrieved_docs]
                                ],
                                "answer_preview": r.answer[:answer_preview_chars],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            with rich_path.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(
                        json.dumps(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "idx": r.idx,
                                "question": r.question,
                                "filter": r.filter_label,
                                "filter_doc_types": r.filter_doc_types,
                                "gold_doc": r.gold_doc,
                                "gold_pages": r.gold_pages,
                                "request_payload": r.request_payload,
                                "response_metadata": r.response_metadata,
                                "top_docs": _build_top_docs(
                                    r.retrieved_docs, max_retrieved_docs
                                ),
                                "answer": r.answer,
                                "answer_preview": r.answer[:answer_preview_chars],
                                "hit_doc": r.hit_gold_doc,
                                "hit_page": r.hit_gold_page,
                                "hit_rank": r.hit_rank,
                                "hit_at_1": r.hit_at_1,
                                "hit_at_3": r.hit_at_3,
                                "hit_at_5": r.hit_at_5,
                                "hit_at_10": r.hit_at_10,
                                "match_debug": r.match_debug,
                                "elapsed_ms": r.elapsed_ms,
                                "error": r.error,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if close_testclient is not None:
            close_testclient()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    summary_lines = []
    for label, results in all_results.items():
        total = len(results)
        errors = sum(1 for r in results if r.error)
        valid = total - errors
        doc_hits = sum(1 for r in results if r.hit_gold_doc)
        page_hits = sum(1 for r in results if r.hit_gold_page)
        avg_elapsed = sum(r.elapsed_ms for r in results) / max(total, 1)

        line = (
            f"  {label:12s}: "
            f"doc_hit={doc_hits}/{valid} ({doc_hits / max(valid, 1) * 100:.1f}%) | "
            f"page_hit={page_hits}/{valid} ({page_hits / max(valid, 1) * 100:.1f}%) | "
            f"errors={errors} | avg={avg_elapsed:.0f}ms"
        )
        print(line)
        summary_lines.append(line)

    # Compare
    if len(configs) == 2:
        label_a, label_b = configs[0][0], configs[1][0]
        results_a, results_b = all_results[label_a], all_results[label_b]
        print(f"\n  Comparison ({label_a} vs {label_b}):")
        diff_doc = []
        diff_page = []
        for ra, rb in zip(results_a, results_b):
            if ra.hit_gold_doc != rb.hit_gold_doc:
                diff_doc.append(
                    (ra.idx, ra.question[:40], ra.hit_gold_doc, rb.hit_gold_doc)
                )
            if ra.hit_gold_page != rb.hit_gold_page:
                diff_page.append(
                    (ra.idx, ra.question[:40], ra.hit_gold_page, rb.hit_gold_page)
                )

        if diff_doc:
            print(f"  Doc hit differences ({len(diff_doc)}):")
            for idx, q, a_hit, b_hit in diff_doc[:20]:
                print(
                    f"    [{idx}] {q}  {label_a}={'✓' if a_hit else '✗'} {label_b}={'✓' if b_hit else '✗'}"
                )
        else:
            print("  No doc hit differences")

        if diff_page:
            print(f"  Page hit differences ({len(diff_page)}):")
            for idx, q, a_hit, b_hit in diff_page[:20]:
                print(
                    f"    [{idx}] {q}  {label_a}={'✓' if a_hit else '✗'} {label_b}={'✓' if b_hit else '✗'}"
                )

    # Save summary
    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        _ = f.write("\n".join(summary_lines))

    # Run validation if --validate flag is set
    if args.validate:
        import subprocess

        print(f"\n{'=' * 60}")
        print("Running validation...")
        print(f"{'=' * 60}")

        validation_results: dict[str, object] = {}
        for label in all_results.keys():
            thin_path = out_dir / f"{label}_results.jsonl"
            rich_path = out_dir / f"{label}_results.rich.jsonl"

            for jsonl_path in (thin_path, rich_path):
                if not jsonl_path.exists():
                    continue

                print(f"\nValidating {jsonl_path.name}...")
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/evaluation/validate_sop_eval_jsonl.py",
                        "--jsonl",
                        str(jsonl_path),
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.stdout:
                    try:
                        validation_results[jsonl_path.name] = json.loads(result.stdout)
                    except json.JSONDecodeError:
                        validation_results[jsonl_path.name] = {
                            "error": "failed to parse output"
                        }
                else:
                    validation_results[jsonl_path.name] = {
                        "error": result.stderr or "no output"
                    }

                status = "PASS" if result.returncode == 0 else "FAIL"
                print(f"  {status}: {jsonl_path.name}")

        report_path = out_dir / "report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)

        print(f"\nValidation report saved to {report_path}")

        all_passed = True
        for value in validation_results.values():
            if isinstance(value, dict):
                if not bool(value.get("ok", False)):
                    all_passed = False
            else:
                all_passed = False
        if not all_passed:
            print("\nWARNING: validation failed for one or more files")
            return 1

        print("\nOK: all validations passed")

    print(f"\nResults saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
