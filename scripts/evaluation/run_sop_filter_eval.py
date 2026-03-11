#!/usr/bin/env python3
"""SOP filter evaluation: compare SOP-only vs 절차검색(SOP+setup) filter results."""
from __future__ import annotations

import csv
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

API_BASE_URL = "http://localhost:8001"
TIMEOUT = 300.0


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
    gold_doc: str
    gold_pages: str
    retrieved_docs: list[dict]
    answer: str
    hit_gold_doc: bool
    hit_gold_page: bool
    elapsed_ms: float
    error: str | None = None


def _post_json(url: str, payload: dict, *, timeout: float) -> tuple[dict, float]:
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


def _thread_id(idx: int, label: str) -> str:
    h = hashlib.sha1(f"sop-eval-{idx}-{label}".encode()).hexdigest()[:12]
    return f"sop-eval-{h}"


def _normalize_doc_name(name: str) -> str:
    """Normalize doc name for comparison."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


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


def _check_hit(retrieved_docs: list[dict], gold_doc: str, gold_pages: str) -> tuple[bool, bool]:
    """Check if gold doc and page are in retrieved results."""
    gold_doc_norm = _normalize_doc_name(gold_doc)
    gold_page_set = _parse_page_range(gold_pages)

    hit_doc = False
    hit_page = False

    for doc in retrieved_docs:
        doc_id = str(doc.get("doc_id") or doc.get("id") or "")
        title = str(doc.get("title") or "")
        source = str(doc.get("metadata", {}).get("source") or doc.get("metadata", {}).get("file_name") or "")

        for candidate in [doc_id, title, source]:
            if gold_doc_norm in _normalize_doc_name(candidate):
                hit_doc = True
                # Check page
                page = doc.get("page")
                if page is not None and int(page) in gold_page_set:
                    hit_page = True
                break

    return hit_doc, hit_page


def run_eval_single(
    row: GoldRow,
    filter_label: str,
    filter_doc_types: list[str],
) -> EvalResult:
    """Run single evaluation query."""
    thread_id = _thread_id(row.idx, filter_label)
    run_url = f"{API_BASE_URL}/api/agent/run"

    # SUPRA XP와 ZEDIUS XP는 같은 장비 (ES에 ZEDIUS XP로 색인됨)
    DEVICE_ALIASES = {
        "SUPRA XP": ["SUPRA XP", "ZEDIUS XP"],
    }
    devices = DEVICE_ALIASES.get(row.device, [row.device]) if row.device else []

    payload: dict = {
        "message": row.question,
        "thread_id": thread_id,
        "auto_parse": False,
        "guided_confirm": False,
        "filter_doc_types": filter_doc_types,
        "filter_devices": devices,
        "target_language": "ko",
    }

    response, elapsed = _post_json(run_url, payload, timeout=TIMEOUT)

    if "error" in response and isinstance(response["error"], str):
        return EvalResult(
            idx=row.idx, question=row.question, filter_label=filter_label,
            filter_doc_types=filter_doc_types, gold_doc=row.gold_doc,
            gold_pages=row.gold_pages, retrieved_docs=[], answer="",
            hit_gold_doc=False, hit_gold_page=False, elapsed_ms=elapsed,
            error=response["error"],
        )

    # Handle interrupt (retrieval review)
    interrupted = response.get("interrupted", False)
    interrupt_payload = response.get("interrupt_payload") or {}

    if interrupted and interrupt_payload.get("type") == "retrieval_review":
        retrieved_docs = interrupt_payload.get("docs") or []
        # Resume to get answer - accept all docs
        doc_ids = [d.get("doc_id") or d.get("id") for d in retrieved_docs if d.get("doc_id") or d.get("id")]
        resume_payload: dict = {
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
        resp2, elapsed2 = _post_json(run_url, resume_payload, timeout=TIMEOUT)
        elapsed += elapsed2
        answer = str(resp2.get("answer") or resp2.get("response") or "")
    elif interrupted and interrupt_payload.get("type") == "device_selection":
        # Device selection interrupt - use first device and retry
        devices = interrupt_payload.get("devices") or []
        first_device = devices[0]["name"] if devices else row.device
        retry_payload: dict = {
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
        resp2, elapsed2 = _post_json(run_url, retry_payload, timeout=TIMEOUT)
        elapsed += elapsed2
        retrieved_docs = (resp2.get("interrupt_payload") or {}).get("docs") or []
        answer = str(resp2.get("answer") or resp2.get("response") or "")
    else:
        retrieved_docs = response.get("retrieved_docs") or []
        answer = str(response.get("answer") or response.get("response") or "")

    hit_doc, hit_page = _check_hit(retrieved_docs, row.gold_doc, row.gold_pages)

    return EvalResult(
        idx=row.idx, question=row.question, filter_label=filter_label,
        filter_doc_types=filter_doc_types, gold_doc=row.gold_doc,
        gold_pages=row.gold_pages, retrieved_docs=retrieved_docs,
        answer=answer[:500], hit_gold_doc=hit_doc, hit_gold_page=hit_page,
        elapsed_ms=elapsed,
    )


def load_csv(path: str) -> list[GoldRow]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(GoldRow(
                idx=idx,
                question=row["질문내용"].strip(),
                device=row.get("장비", "").strip(),
                doc_type=row.get("문서종류", "").strip(),
                gold_doc=row.get("정답문서", "").strip(),
                gold_pages=row.get("정답 페이지", "").strip(),
                note=row.get("특이점", "").strip(),
            ))
    return rows


def main():
    csv_path = "data/eval_sop_question_list_박진우_변형.csv"
    out_dir = Path(".sisyphus/evidence/2026-03-11_sop_filter_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} questions")

    configs = [
        ("sop_only", ["SOP"]),
        ("절차검색", ["SOP", "set_up_manual"]),
    ]

    all_results: dict[str, list[EvalResult]] = {}

    for label, doc_types in configs:
        print(f"\n{'='*60}")
        print(f"Running: {label} (filter_doc_types={doc_types})")
        print(f"{'='*60}")
        results = []

        for i, row in enumerate(rows):
            print(f"  [{i+1}/{len(rows)}] {row.question[:50]}...", end=" ", flush=True)
            result = run_eval_single(row, label, doc_types)
            results.append(result)

            status = "✓" if result.hit_gold_doc else "✗"
            page_status = "p✓" if result.hit_gold_page else "p✗"
            n_docs = len(result.retrieved_docs)
            print(f"{status} {page_status} docs={n_docs} {result.elapsed_ms:.0f}ms"
                  + (f" ERR:{result.error[:30]}" if result.error else ""))

        all_results[label] = results

        # Save per-config results
        out_path = out_dir / f"{label}_results.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({
                    "idx": r.idx, "question": r.question, "filter": r.filter_label,
                    "filter_doc_types": r.filter_doc_types,
                    "gold_doc": r.gold_doc, "gold_pages": r.gold_pages,
                    "hit_doc": r.hit_gold_doc, "hit_page": r.hit_gold_page,
                    "n_docs": len(r.retrieved_docs), "elapsed_ms": r.elapsed_ms,
                    "error": r.error,
                    "retrieved_doc_ids": [d.get("doc_id") or d.get("id") for d in r.retrieved_docs[:10]],
                    "answer_preview": r.answer[:200],
                }, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
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
            f"doc_hit={doc_hits}/{valid} ({doc_hits/max(valid,1)*100:.1f}%) | "
            f"page_hit={page_hits}/{valid} ({page_hits/max(valid,1)*100:.1f}%) | "
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
                diff_doc.append((ra.idx, ra.question[:40], ra.hit_gold_doc, rb.hit_gold_doc))
            if ra.hit_gold_page != rb.hit_gold_page:
                diff_page.append((ra.idx, ra.question[:40], ra.hit_gold_page, rb.hit_gold_page))

        if diff_doc:
            print(f"  Doc hit differences ({len(diff_doc)}):")
            for idx, q, a_hit, b_hit in diff_doc[:20]:
                print(f"    [{idx}] {q}  {label_a}={'✓' if a_hit else '✗'} {label_b}={'✓' if b_hit else '✗'}")
        else:
            print("  No doc hit differences")

        if diff_page:
            print(f"  Page hit differences ({len(diff_page)}):")
            for idx, q, a_hit, b_hit in diff_page[:20]:
                print(f"    [{idx}] {q}  {label_a}={'✓' if a_hit else '✗'} {label_b}={'✓' if b_hit else '✗'}")

    # Save summary
    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
