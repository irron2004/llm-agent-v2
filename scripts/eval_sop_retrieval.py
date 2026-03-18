"""SOP 질문 리스트 평가 스크립트.

CSV의 질문을 chat flow와 동일하게 API에 요청하고,
retrieved docs가 정답 문서에 포함되는지 체크한다.

Usage:
    uv run python -u backend/scripts/eval_sop_retrieval.py [--limit N] [--api-url URL]
"""
from __future__ import annotations

import functools
import builtins
# Force unbuffered print
builtins.print = functools.partial(builtins.print, flush=True)  # type: ignore[assignment]

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

API_URL = "http://localhost:8001/api/agent/run"
CSV_PATH = Path("data/eval_sop_question_list_박진우.csv")


def _normalize_doc_name(name: str) -> str:
    """Normalize document name for fuzzy matching.

    Strips extension, lowercases, removes extra whitespace,
    and replaces separators with underscores.
    """
    name = name.strip().lower()
    name = re.sub(r"\.pdf$", "", name)
    name = re.sub(r"[\s]+", " ", name)
    # "global sop_supra xp_all_pm_prism source 3100qc"
    # -> "global_sop_supra_xp_all_pm_prism_source_3100qc"
    name = name.replace(" ", "_")
    return name


def _doc_id_matches(retrieved_doc_id: str, ground_truth_name: str) -> bool:
    """Check if a retrieved doc_id matches the ground truth document name."""
    norm_retrieved = _normalize_doc_name(retrieved_doc_id)
    norm_gt = _normalize_doc_name(ground_truth_name)
    # Exact match or containment (doc_id may have extra prefixes)
    return norm_gt in norm_retrieved or norm_retrieved in norm_gt


def _parse_page_range(page_str: str) -> set[int]:
    """Parse '6-14' or '6' into a set of page numbers."""
    pages = set()
    for part in page_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                pages.update(range(int(lo), int(hi) + 1))
            except ValueError:
                pass
        elif part.isdigit():
            pages.add(int(part))
    return pages


def run_chat_flow(query: str, api_url: str, timeout: int = 300) -> dict[str, Any]:
    """Execute the 2-step guided confirm chat flow.

    Step 1: guided_confirm=true → get thread_id
    Step 2: resume with task_mode=sop
    """
    # Step 1: Initial request with guided_confirm
    payload_step1 = {
        "message": query,
        "guided_confirm": True,
        "auto_parse": True,
        "max_attempts": 2,
    }
    resp1 = requests.post(api_url, json=payload_step1, timeout=timeout)
    resp1.raise_for_status()
    data1 = resp1.json()

    thread_id = data1.get("thread_id")
    interrupted = data1.get("interrupted", False)

    if not interrupted or not thread_id:
        # No interrupt — direct answer (shouldn't happen with guided_confirm)
        return data1

    # Step 2: Resume with task_mode=sop (작업절차검색)
    payload_step2 = {
        "message": query,
        "guided_confirm": True,
        "thread_id": thread_id,
        "resume_decision": {
            "type": "auto_parse_confirm",
            "task_mode": "sop",
            "target_language": "ko",
        },
        "max_attempts": 2,
    }
    resp2 = requests.post(api_url, json=payload_step2, timeout=timeout)
    resp2.raise_for_status()
    return resp2.json()


def evaluate_single(
    query: str,
    ground_truth_doc: str,
    ground_truth_pages: str,
    api_url: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Evaluate a single question with retry on timeout/connection errors."""
    t0 = time.time()
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = run_chat_flow(query, api_url)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            print(f"  retry {attempt}/{max_retries}: {type(exc).__name__}: {str(exc)[:80]}")
            if attempt < max_retries:
                time.sleep(5)
    if last_exc is not None:
        return {
            "query": query,
            "ground_truth_doc": ground_truth_doc,
            "ground_truth_pages": ground_truth_pages,
            "error": str(last_exc),
            "retrieval_hit": False,
            "page_hit": False,
            "answer_empty": True,
            "answer": "",
            "answer_length": 0,
            "elapsed": time.time() - t0,
        }
    elapsed = time.time() - t0

    answer = result.get("answer", "")
    retrieved_docs = result.get("retrieved_docs", [])
    all_retrieved_docs = result.get("all_retrieved_docs") or retrieved_docs
    expanded_docs = result.get("expanded_docs") or []

    # Check retrieval hit: does any retrieved doc match the ground truth?
    gt_pages = _parse_page_range(ground_truth_pages)

    retrieval_hit = False
    page_hit = False
    matched_doc_ids: list[str] = []
    matched_pages: set[int] = set()

    for doc in all_retrieved_docs:
        doc_id = doc.get("id", "")
        if _doc_id_matches(doc_id, ground_truth_doc):
            retrieval_hit = True
            matched_doc_ids.append(doc_id)
            # Check page overlap
            doc_pages = set()
            if doc.get("expanded_pages"):
                doc_pages.update(doc["expanded_pages"])
            elif doc.get("page") is not None:
                doc_pages.add(doc["page"])
            overlap = doc_pages & gt_pages
            if overlap:
                page_hit = True
                matched_pages.update(overlap)

    # Also check expanded_docs for retrieval hit
    for edoc in expanded_docs:
        doc_id = edoc.get("doc_id", "")
        if _doc_id_matches(doc_id, ground_truth_doc):
            retrieval_hit = True
            if doc_id not in matched_doc_ids:
                matched_doc_ids.append(doc_id)

    # Check if answer is empty/failure
    answer_empty = (
        "찾지 못했습니다" in answer
        or "관련 절차 문서를" in answer
        or len(answer.strip()) < 50
    )

    return {
        "query": query,
        "ground_truth_doc": ground_truth_doc,
        "ground_truth_pages": ground_truth_pages,
        "retrieval_hit": retrieval_hit,
        "page_hit": page_hit,
        "answer_empty": answer_empty,
        "answer": answer,
        "answer_length": len(answer),
        "matched_doc_ids": matched_doc_ids,
        "matched_pages": sorted(matched_pages),
        "num_retrieved": len(all_retrieved_docs),
        "num_expanded": len(expanded_docs),
        "elapsed": round(elapsed, 1),
        "route": result.get("metadata", {}).get("route", ""),
        "judge": result.get("judge", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="SOP 질문 retrieval 평가")
    parser.add_argument("--limit", type=int, default=0, help="평가할 질문 수 (0=전체)")
    parser.add_argument("--api-url", default=API_URL, help="API URL")
    parser.add_argument("--output", default="data/eval_sop_results.json", help="결과 저장 경로")
    parser.add_argument("--start", type=int, default=0, help="시작 인덱스")
    parser.add_argument("--csv", default=str(CSV_PATH), help="입력 CSV 경로")
    args = parser.parse_args()

    csv_path = Path(args.csv)

    # Read CSV — 두 가지 형식 지원:
    # 형식1: 질문내용, 정답문서, 정답 페이지 (original)
    # 형식2: qid, question, expected_doc, expected_pages, variant
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or row.get("질문내용") or "").strip()
            gt_doc = (row.get("expected_doc") or row.get("정답문서") or "").strip()
            gt_pages = (row.get("expected_pages") or row.get("정답 페이지") or "").strip()
            variant = (row.get("variant") or "").strip()
            if q and gt_doc:
                rows.append({"query": q, "gt_doc": gt_doc, "gt_pages": gt_pages, "variant": variant})

    if args.start > 0:
        rows = rows[args.start:]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"총 {len(rows)}개 질문 평가 시작 (API: {args.api_url})")
    print("=" * 80)

    results: list[dict[str, Any]] = []
    retrieval_hits = 0
    page_hits = 0
    answer_ok = 0

    # Load existing results for resume support
    output_path = Path(args.output)
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            results = existing.get("results", [])
            retrieval_hits = sum(1 for r in results if r.get("retrieval_hit"))
            page_hits = sum(1 for r in results if r.get("page_hit"))
            answer_ok = sum(1 for r in results if not r.get("answer_empty") and not r.get("error"))
            print(f"기존 결과 {len(results)}개 로드됨, 이어서 진행")
        except Exception:
            pass

    def _save_results():
        total = len(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "total": total,
                        "retrieval_hits": retrieval_hits,
                        "retrieval_accuracy": round(retrieval_hits / total * 100, 1) if total else 0,
                        "page_hits": page_hits,
                        "page_accuracy": round(page_hits / total * 100, 1) if total else 0,
                        "answer_ok": answer_ok,
                        "answer_rate": round(answer_ok / total * 100, 1) if total else 0,
                    },
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    for i, row in enumerate(rows):
        idx = i + args.start + 1
        print(f"\n[{idx}/{len(rows) + args.start}] {row['query'][:70]}...")
        res = evaluate_single(row["query"], row["gt_doc"], row["gt_pages"], args.api_url)
        results.append(res)

        r_mark = "O" if res["retrieval_hit"] else "X"
        p_mark = "O" if res["page_hit"] else "X"
        a_mark = "X" if res.get("answer_empty") else "O"

        if res["retrieval_hit"]:
            retrieval_hits += 1
        if res["page_hit"]:
            page_hits += 1
        if not res.get("answer_empty") and not res.get("error"):
            answer_ok += 1

        print(
            f"  문서검색={r_mark} 페이지={p_mark} 답변={a_mark} "
            f"({res['elapsed']}s) matched={res.get('matched_doc_ids', [])}"
        )

        if res.get("error"):
            print(f"  ERROR: {res['error']}")

        # 매 질문마다 저장 (중간 중단 대비)
        _save_results()

    # Summary
    total = len(results)
    print("\n" + "=" * 80)
    print(f"총 {total}개 질문 평가 완료")
    print(f"  문서 검색 정확도: {retrieval_hits}/{total} ({retrieval_hits/total*100:.1f}%)")
    print(f"  페이지 적중률:    {page_hits}/{total} ({page_hits/total*100:.1f}%)")
    print(f"  답변 생성 성공:   {answer_ok}/{total} ({answer_ok/total*100:.1f}%)")
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
