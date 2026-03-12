#!/usr/bin/env python3
"""SOP retrieval-only evaluation: check if gold docs are retrieved (no LLM answer)."""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

API_BASE_URL = "http://localhost:8001"
TIMEOUT = 60.0

DEVICE_ALIASES = {
    "SUPRA XP": ["SUPRA XP", "ZEDIUS XP"],
}


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


def _normalize(name: str) -> str:
    normalized = str(name or "").lower().strip()
    normalized = re.sub(r"\.(pdf|docx|doc|txt)$", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _parse_pages(pages_str: str) -> set[int]:
    pages = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                s, e = part.split("-", 1)
                for p in range(int(s), int(e) + 1):
                    pages.add(p)
            except ValueError:
                pass
        elif part.isdigit():
            pages.add(int(part))
    return pages


def _check_hit(
    docs: list[dict], gold_doc: str, gold_pages: str
) -> tuple[bool, bool, int | None]:
    gold_norm = _normalize(gold_doc)
    gold_page_set = _parse_pages(gold_pages)

    first_match_rank: int | None = None
    page_match_found = False

    for rank, doc in enumerate(docs, start=1):
        doc_id = _normalize(str(doc.get("doc_id") or doc.get("id") or ""))
        title = _normalize(str(doc.get("title") or ""))
        source = ""
        metadata_obj = doc.get("metadata")
        if isinstance(metadata_obj, dict):
            source = _normalize(
                str(metadata_obj.get("source") or metadata_obj.get("file_name") or "")
            )

        matched = False
        for candidate in [doc_id, title, source]:
            if gold_norm and gold_norm in candidate:
                matched = True
                break

        if not matched:
            continue

        if first_match_rank is None:
            first_match_rank = rank

        page = doc.get("page")
        if page is None:
            continue

        try:
            page_int = int(page)
        except (ValueError, TypeError):
            continue

        if page_int in gold_page_set:
            page_match_found = True
            break

    if first_match_rank is None:
        return False, False, None
    return True, page_match_found, first_match_rank


def main():
    csv_path = "data/eval_sop_question_list_박진우_변형.csv"
    out_dir = Path(".sisyphus/evidence/2026-03-11_sop_retrieval_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            rows.append(row | {"_idx": idx})

    print(f"Loaded {len(rows)} questions")

    configs = [
        ("sop_only", ["SOP"]),
        ("절차검색", ["SOP", "set_up_manual"]),
    ]

    all_results: dict[str, list[dict]] = {}

    for label, doc_types in configs:
        print(f"\n{'=' * 60}")
        print(f"Running: {label} (filter_doc_types={doc_types})")
        print(f"{'=' * 60}")
        results = []

        for row in rows:
            idx = row["_idx"]
            question = row["질문내용"].strip()
            device = row.get("장비", "").strip()
            gold_doc = row.get("정답문서", "").strip()
            gold_pages = row.get("정답 페이지", "").strip()

            devices = DEVICE_ALIASES.get(device, [device]) if device else []
            thread_id = (
                f"ret-eval-{hashlib.sha1(f'{idx}-{label}'.encode()).hexdigest()[:10]}"
            )

            payload = {
                "message": question,
                "thread_id": thread_id,
                "auto_parse": False,
                "guided_confirm": False,
                "ask_user_after_retrieve": True,
                "filter_doc_types": doc_types,
                "filter_devices": devices,
                "target_language": "ko",
            }

            response, elapsed = _post_json(
                f"{API_BASE_URL}/api/agent/run", payload, timeout=TIMEOUT
            )

            error = None
            docs = []
            if "error" in response:
                error = str(response["error"])
            elif response.get("interrupted"):
                ip = response.get("interrupt_payload") or {}
                docs = ip.get("docs") or []
            else:
                docs = response.get("retrieved_docs") or []

            hit_doc, hit_page, hit_rank = _check_hit(docs, gold_doc, gold_pages)

            status = "✓" if hit_doc else "✗"
            pstatus = "p✓" if hit_page else "p✗"
            rank_str = f"@{hit_rank}" if hit_rank else ""
            err_str = f" ERR:{error[:40]}" if error else ""
            print(
                f"  [{idx + 1}/{len(rows)}] {question[:50]:50s} {status} {pstatus} {rank_str:4s} docs={len(docs)} {elapsed:.0f}ms{err_str}"
            )

            result = {
                "idx": idx,
                "question": question,
                "device": device,
                "filter": label,
                "filter_doc_types": doc_types,
                "gold_doc": gold_doc,
                "gold_pages": gold_pages,
                "hit_doc": hit_doc,
                "hit_page": hit_page,
                "hit_rank": hit_rank,
                "n_docs": len(docs),
                "elapsed_ms": elapsed,
                "error": error,
                "retrieved_doc_ids": [
                    d.get("doc_id") or d.get("id") for d in docs[:20]
                ],
                "retrieved_pages": [d.get("page") for d in docs[:20]],
            }
            results.append(result)

        all_results[label] = results

        out_path = out_dir / f"{label}_results.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for label, results in all_results.items():
        total = len(results)
        errors = sum(1 for r in results if r["error"])
        valid = total - errors
        doc_hits = sum(1 for r in results if r["hit_doc"])
        page_hits = sum(1 for r in results if r["hit_page"])
        avg_elapsed = sum(r["elapsed_ms"] for r in results) / max(total, 1)
        ranks = [r["hit_rank"] for r in results if r["hit_rank"]]
        avg_rank = sum(ranks) / len(ranks) if ranks else 0

        print(
            f"  {label:12s}: doc_hit={doc_hits}/{valid} ({doc_hits / max(valid, 1) * 100:.1f}%) | "
            f"page_hit={page_hits}/{valid} ({page_hits / max(valid, 1) * 100:.1f}%) | "
            f"avg_rank={avg_rank:.1f} | errors={errors} | avg={avg_elapsed:.0f}ms"
        )

    # Detailed comparison
    if len(configs) == 2:
        la, lb = configs[0][0], configs[1][0]
        ra, rb = all_results[la], all_results[lb]

        print(f"\n  Missed docs (both miss):")
        both_miss = [
            (a, b) for a, b in zip(ra, rb) if not a["hit_doc"] and not b["hit_doc"]
        ]
        for a, _ in both_miss[:15]:
            print(
                f"    [{a['idx'] + 1}] {a['question'][:50]} | gold={a['gold_doc'][:40]}"
            )

        print(f"\n  {la} only hits:")
        for a, b in zip(ra, rb):
            if a["hit_doc"] and not b["hit_doc"]:
                print(f"    [{a['idx'] + 1}] {a['question'][:50]}")

        print(f"\n  {lb} only hits:")
        for a, b in zip(ra, rb):
            if not a["hit_doc"] and b["hit_doc"]:
                print(f"    [{a['idx'] + 1}] {a['question'][:50]}")

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        for label, results in all_results.items():
            total = len(results)
            errors = sum(1 for r in results if r["error"])
            valid = total - errors
            doc_hits = sum(1 for r in results if r["hit_doc"])
            page_hits = sum(1 for r in results if r["hit_page"])
            f.write(
                f"{label}: doc_hit={doc_hits}/{valid} page_hit={page_hits}/{valid} errors={errors}\n"
            )

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
