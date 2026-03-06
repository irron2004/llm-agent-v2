from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chunk_v3.run_embedding import MODEL_CONFIGS, embed_queries
from scripts.chunk_v3.run_ingest import CONTENT_INDEX, _get_es_client


@dataclass
class EvalItem:
    qid: str
    question: str
    expected_doc_id: str
    expected_pages_raw: str
    expected_pages: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="chunk_v3 formal SOP evaluation")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()))
    parser.add_argument(
        "--input-csv",
        default="docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--content-index", default=CONTENT_INDEX)
    parser.add_argument("--embed-index-prefix", default="chunk_v3_embed_")
    parser.add_argument("--embed-index-suffix", default="_v1")
    parser.add_argument("--output-dir", default="data/chunks_v3/eval_formal")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _normalize_doc_id(value: str) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"\.(pdf|pptx|docx|doc|txt)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _parse_expected_pages(raw: str) -> set[int]:
    out: set[int] = set()
    text = str(raw or "").strip()
    if not text:
        return out
    for token in [t.strip() for t in text.split(",") if t.strip()]:
        if "-" in token:
            left, right = token.split("-", 1)
            if left.strip().isdigit() and right.strip().isdigit():
                a = int(left.strip())
                b = int(right.strip())
                lo, hi = (a, b) if a <= b else (b, a)
                for i in range(lo, hi + 1):
                    out.add(i)
        elif token.isdigit():
            out.add(int(token))
    return out


def _load_items(csv_path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = str(row.get("question", "") or "").strip()
            if not question:
                continue
            expected_doc_id = str(row.get("matched_doc_id", "") or "").strip()
            if not expected_doc_id:
                expected_doc_id = _normalize_doc_id(
                    str(row.get("expected_doc", "") or "")
                )
            expected_pages_raw = str(row.get("expected_pages", "") or "").strip()
            items.append(
                EvalItem(
                    qid=str(row.get("qid", "") or ""),
                    question=question,
                    expected_doc_id=expected_doc_id,
                    expected_pages_raw=expected_pages_raw,
                    expected_pages=_parse_expected_pages(expected_pages_raw),
                )
            )
    return items


def _knn(
    es, index_name: str, query_vector: list[float], top_k: int, num_candidates: int
) -> list[dict[str, Any]]:
    body = {
        "size": top_k,
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": num_candidates,
        },
        "_source": ["chunk_id", "doc_type", "chapter", "content_hash"],
    }
    resp = es.search(index=index_name, body=body)
    return list(resp.get("hits", {}).get("hits", []))


def _metric_at_k(values: list[bool], k: int) -> bool:
    return any(values[:k])


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def main() -> None:
    args = parse_args()
    es = _get_es_client()
    items = _load_items(Path(args.input_csv))
    if not items:
        raise ValueError("no evaluation rows found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    questions = [item.question for item in items]
    for model_key in args.models:
        if model_key not in MODEL_CONFIGS:
            continue

        embed_index = f"{args.embed_index_prefix}{model_key}{args.embed_index_suffix}"
        query_vecs = embed_queries(model_key, questions, device=args.device)

        hit1 = hit3 = hit5 = hit10 = 0
        phit1 = phit3 = phit5 = phit10 = 0

        for item, qvec in zip(items, query_vecs):
            hits = _knn(
                es,
                index_name=embed_index,
                query_vector=qvec.tolist(),
                top_k=args.top_k,
                num_candidates=max(args.top_k, args.num_candidates),
            )

            chunk_ids = [str(h.get("_id", "")) for h in hits if str(h.get("_id", ""))]
            content_map: dict[str, dict[str, Any]] = {}
            if chunk_ids:
                mget_resp = es.mget(index=args.content_index, body={"ids": chunk_ids})
                for doc in mget_resp.get("docs", []):
                    if doc.get("found"):
                        content_map[str(doc.get("_id", ""))] = doc.get("_source", {})

            doc_match_flags: list[bool] = []
            page_match_flags: list[bool] = []
            matched_rank = 0

            for rank, hit in enumerate(hits, start=1):
                cid = str(hit.get("_id", ""))
                src = content_map.get(cid, {})
                doc_id = _normalize_doc_id(str(src.get("doc_id", "") or ""))
                page = int(src.get("page", 0) or 0)
                doc_match = bool(doc_id and doc_id == item.expected_doc_id)
                page_match = doc_match and (
                    not item.expected_pages or page in item.expected_pages
                )
                doc_match_flags.append(doc_match)
                page_match_flags.append(page_match)
                if matched_rank == 0 and doc_match:
                    matched_rank = rank

                detail_rows.append(
                    {
                        "model": model_key,
                        "qid": item.qid,
                        "question": item.question,
                        "expected_doc_id": item.expected_doc_id,
                        "expected_pages": item.expected_pages_raw,
                        "rank": rank,
                        "chunk_id": cid,
                        "doc_id": doc_id,
                        "page": page,
                        "score": float(hit.get("_score", 0.0) or 0.0),
                        "doc_match": doc_match,
                        "page_match": page_match,
                    }
                )

            h1 = _metric_at_k(doc_match_flags, 1)
            h3 = _metric_at_k(doc_match_flags, 3)
            h5 = _metric_at_k(doc_match_flags, 5)
            h10 = _metric_at_k(doc_match_flags, 10)
            p1 = _metric_at_k(page_match_flags, 1)
            p3 = _metric_at_k(page_match_flags, 3)
            p5 = _metric_at_k(page_match_flags, 5)
            p10 = _metric_at_k(page_match_flags, 10)

            hit1 += int(h1)
            hit3 += int(h3)
            hit5 += int(h5)
            hit10 += int(h10)
            phit1 += int(p1)
            phit3 += int(p3)
            phit5 += int(p5)
            phit10 += int(p10)

            detail_rows.append(
                {
                    "model": model_key,
                    "qid": item.qid,
                    "question": item.question,
                    "expected_doc_id": item.expected_doc_id,
                    "expected_pages": item.expected_pages_raw,
                    "rank": 0,
                    "chunk_id": "",
                    "doc_id": "",
                    "page": 0,
                    "score": 0.0,
                    "doc_match": False,
                    "page_match": False,
                    "matched_rank": matched_rank,
                    "hit_at_1": h1,
                    "hit_at_3": h3,
                    "hit_at_5": h5,
                    "hit_at_10": h10,
                    "page_hit_at_1": p1,
                    "page_hit_at_3": p3,
                    "page_hit_at_5": p5,
                    "page_hit_at_10": p10,
                }
            )

        total = len(items)
        summary_rows.append(
            {
                "model": model_key,
                "queries": total,
                "hit_at_1": _pct(hit1, total),
                "hit_at_3": _pct(hit3, total),
                "hit_at_5": _pct(hit5, total),
                "hit_at_10": _pct(hit10, total),
                "page_hit_at_1": _pct(phit1, total),
                "page_hit_at_3": _pct(phit3, total),
                "page_hit_at_5": _pct(phit5, total),
                "page_hit_at_10": _pct(phit10, total),
            }
        )

    summary_md = output_dir / "sop_eval_summary.md"
    summary_json = output_dir / "sop_eval_summary.json"
    detail_csv = output_dir / "sop_eval_rows.csv"

    lines = [
        "# chunk_v3 SOP Evaluation",
        "",
        f"- input: `{args.input_csv}`",
        f"- queries: {len(items)}",
        "",
    ]
    lines.append(
        "| model | hit@1 | hit@3 | hit@5 | hit@10 | page-hit@1 | page-hit@3 | page-hit@5 | page-hit@10 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['hit_at_1']} | {row['hit_at_3']} | {row['hit_at_5']} | {row['hit_at_10']} | "
            f"{row['page_hit_at_1']} | {row['page_hit_at_3']} | {row['page_hit_at_5']} | {row['page_hit_at_10']} |"
        )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_json.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "qid",
            "question",
            "expected_doc_id",
            "expected_pages",
            "rank",
            "chunk_id",
            "doc_id",
            "page",
            "score",
            "doc_match",
            "page_match",
            "matched_rank",
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "hit_at_10",
            "page_hit_at_1",
            "page_hit_at_3",
            "page_hit_at_5",
            "page_hit_at_10",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"summary: {summary_md}")
    print(f"rows: {detail_csv}")


if __name__ == "__main__":
    main()
