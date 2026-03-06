from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chunk_v3.run_embedding import MODEL_CONFIGS, embed_queries
from scripts.chunk_v3.run_ingest import CONTENT_INDEX, _get_es_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="chunk_v3 smoke retrieval evaluation")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()))
    parser.add_argument(
        "--queries-csv",
        default="docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv",
    )
    parser.add_argument("--manual-query", action="append", default=[])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--content-index", default=CONTENT_INDEX)
    parser.add_argument("--embed-index-prefix", default="chunk_v3_embed_")
    parser.add_argument("--embed-index-suffix", default="_v1")
    parser.add_argument("--output-dir", default="data/chunks_v3/eval_smoke")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_queries(csv_path: Path, limit: int) -> list[str]:
    queries: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = str(row.get("question", "") or "").strip()
            if q:
                queries.append(q)
            if len(queries) >= limit:
                break
    return queries


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
        "_source": ["chunk_id", "doc_type", "device_name", "chapter", "content_hash"],
    }
    resp = es.search(index=index_name, body=body)
    return list(resp.get("hits", {}).get("hits", []))


def main() -> None:
    args = parse_args()
    es = _get_es_client()

    csv_queries = _load_queries(Path(args.queries_csv), args.limit)
    queries = csv_queries + [q for q in args.manual_query if str(q).strip()]
    if not queries:
        raise ValueError("no queries found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        if model_key not in MODEL_CONFIGS:
            print(f"skip unknown model: {model_key}")
            continue

        embed_index = f"{args.embed_index_prefix}{model_key}{args.embed_index_suffix}"
        vecs = embed_queries(model_key, queries, device=args.device)
        rows: list[dict[str, Any]] = []

        for query, vec in zip(queries, vecs):
            hits = _knn(
                es,
                index_name=embed_index,
                query_vector=vec.tolist(),
                top_k=args.top_k,
                num_candidates=max(args.top_k, args.num_candidates),
            )

            chunk_ids = [str(h.get("_id", "")) for h in hits if str(h.get("_id", ""))]
            content_map: dict[str, dict[str, Any]] = {}
            if chunk_ids:
                content_resp = es.mget(
                    index=args.content_index, body={"ids": chunk_ids}
                )
                for doc in content_resp.get("docs", []):
                    if doc.get("found"):
                        cid = str(doc.get("_id", ""))
                        content_map[cid] = doc.get("_source", {})

            for rank, hit in enumerate(hits, start=1):
                chunk_id = str(hit.get("_id", ""))
                src = hit.get("_source", {}) or {}
                content_src = content_map.get(chunk_id, {})
                rows.append(
                    {
                        "model": model_key,
                        "query": query,
                        "rank": rank,
                        "score": float(hit.get("_score", 0.0) or 0.0),
                        "chunk_id": chunk_id,
                        "doc_id": str(content_src.get("doc_id", "") or ""),
                        "page": int(content_src.get("page", 0) or 0),
                        "doc_type": str(
                            src.get("doc_type", "") or content_src.get("doc_type", "")
                        ),
                        "chapter": str(
                            src.get("chapter", "") or content_src.get("chapter", "")
                        ),
                    }
                )

        out_jsonl = output_dir / f"smoke_{model_key}.jsonl"
        out_csv = output_dir / f"smoke_{model_key}.csv"

        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "query",
                    "rank",
                    "score",
                    "chunk_id",
                    "doc_id",
                    "page",
                    "doc_type",
                    "chapter",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"{model_key}: wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
