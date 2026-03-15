"""Hybrid+Rerank masked query experiment for Paper A.

Conditions:
  B0      BM25  no filter   (original + masked)
  B1      Dense no filter   (original + masked)
  B2      Hybrid RRF no filter (original + masked)
  B3      Hybrid RRF + Rerank no filter (original + masked)
  B4      B3 + hard device filter (masked only)
  B4.5    B3 + device + shared filter (masked only)

Outputs:
  data/paper_a/masked_hybrid_results.json   -- full per-query results
  Console summary table
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from elasticsearch import Elasticsearch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import rag_settings, search_settings
from backend.llm_infrastructure.reranking.adapters.cross_encoder import (
    CrossEncoderReranker,
)
from backend.services.embedding_service import EmbeddingService

from scripts.paper_a._io import read_jsonl, write_json

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTENT_INDEX = "chunk_v3_content"
EMBED_INDEX = "chunk_v3_embed_bge_m3_v1"
TOP_K = 10
RRF_K = 60
FETCH_N = TOP_K * 4  # candidates fetched per leg before RRF

EVAL_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl"
DOC_SCOPE_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/doc_scope.jsonl"
SHARED_IDS_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/shared_doc_ids.txt"
OUT_PATH = ROOT / "data/paper_a/masked_hybrid_results.json"

# Conditions to run
ORIGINAL_CONDITIONS = ["B0_orig", "B1_orig", "B2_orig", "B3_orig"]
MASKED_CONDITIONS = ["B0_masked", "B1_masked", "B2_masked", "B3_masked", "B4_masked", "B4.5_masked"]
ALL_CONDITIONS = ORIGINAL_CONDITIONS + MASKED_CONDITIONS


# ---------------------------------------------------------------------------
# ES client
# ---------------------------------------------------------------------------
def _build_es() -> Elasticsearch:
    if search_settings.es_user and search_settings.es_password:
        return Elasticsearch(
            hosts=[search_settings.es_host],
            verify_certs=True,
            basic_auth=(search_settings.es_user, search_settings.es_password),
        )
    return Elasticsearch(hosts=[search_settings.es_host], verify_certs=True)


# ---------------------------------------------------------------------------
# Policy data loading
# ---------------------------------------------------------------------------
def load_doc_scope(path: Path) -> dict[str, str]:
    """Return {es_doc_id -> es_device_name}."""
    result: dict[str, str] = {}
    for row in read_jsonl(path):
        assert isinstance(row, dict)
        doc_id = str(row.get("es_doc_id") or "").strip()
        device = str(row.get("es_device_name") or "").strip()
        if doc_id:
            result[doc_id] = device
    return result


def load_shared_ids(path: Path) -> set[str]:
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            result.add(line)
    return result


def load_eval(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        assert isinstance(row, dict)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
_embed_svc: EmbeddingService | None = None


def get_embed_svc() -> EmbeddingService:
    global _embed_svc
    if _embed_svc is None:
        _embed_svc = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
            device=rag_settings.embedding_device,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )
    return _embed_svc


def embed_query(query: str) -> list[float]:
    vec = np.asarray(get_embed_svc().embed_query(query), dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(device=rag_settings.embedding_device)
    return _reranker


# ---------------------------------------------------------------------------
# Low-level ES retrieval helpers
# ---------------------------------------------------------------------------
def bm25_search(
    es: Elasticsearch,
    query: str,
    top_n: int,
    extra_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """BM25 over chunk_v3_content. Returns list of {doc_id, chunk_id, content, score, device_name}."""
    must_clause: dict[str, Any] = {
        "multi_match": {
            "query": query,
            "fields": ["search_text^1.0", "content^0.8"],
            "type": "best_fields",
        }
    }
    bool_q: dict[str, Any] = {"must": must_clause}
    if extra_filter:
        bool_q["filter"] = extra_filter

    body: dict[str, Any] = {
        "query": {"bool": bool_q},
        "size": top_n,
        "_source": ["doc_id", "chunk_id", "content", "device_name", "search_text"],
    }
    resp = es.search(index=CONTENT_INDEX, body=body)
    hits = []
    for h in resp["hits"]["hits"]:
        src = h.get("_source", {})
        hits.append(
            {
                "doc_id": str(src.get("doc_id") or ""),
                "chunk_id": str(src.get("chunk_id") or h["_id"]),
                "content": str(src.get("content") or src.get("search_text") or ""),
                "score": float(h["_score"] or 0.0),
                "device_name": str(src.get("device_name") or ""),
            }
        )
    return hits


def dense_search(
    es: Elasticsearch,
    query_vec: list[float],
    top_n: int,
    device_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """kNN over chunk_v3_embed_bge_m3_v1. Returns list of {chunk_id, score, device_name}.
    Device filter applied on device_name.keyword in the embed index.
    """
    knn_q: dict[str, Any] = {
        "field": "embedding",
        "query_vector": query_vec,
        "k": top_n,
        "num_candidates": top_n * 2,
    }
    if device_filter:
        knn_q["filter"] = {"terms": {"device_name": device_filter}}

    body: dict[str, Any] = {
        "knn": knn_q,
        "size": top_n,
        "_source": ["chunk_id", "device_name"],
    }
    resp = es.search(index=EMBED_INDEX, body=body)
    hits = []
    for h in resp["hits"]["hits"]:
        src = h.get("_source", {})
        hits.append(
            {
                "chunk_id": str(src.get("chunk_id") or h["_id"]),
                "score": float(h["_score"] or 0.0),
                "device_name": str(src.get("device_name") or ""),
            }
        )
    return hits


def _fetch_content_by_chunk_ids(
    es: Elasticsearch, chunk_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Fetch doc_id + content from chunk_v3_content for a list of chunk_ids."""
    if not chunk_ids:
        return {}
    body: dict[str, Any] = {
        "query": {"terms": {"chunk_id": chunk_ids}},
        "size": len(chunk_ids),
        "_source": ["doc_id", "chunk_id", "content", "search_text", "device_name"],
    }
    resp = es.search(index=CONTENT_INDEX, body=body)
    result: dict[str, dict[str, Any]] = {}
    for h in resp["hits"]["hits"]:
        src = h.get("_source", {})
        cid = str(src.get("chunk_id") or h["_id"])
        result[cid] = {
            "doc_id": str(src.get("doc_id") or ""),
            "content": str(src.get("content") or src.get("search_text") or ""),
            "device_name": str(src.get("device_name") or ""),
        }
    return result


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def rrf_fuse(
    *rank_lists: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk_ids.

    Returns list of (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# High-level retrieval per condition
# ---------------------------------------------------------------------------
def retrieve(
    *,
    es: Elasticsearch,
    query: str,
    mode: str,  # bm25 | dense | hybrid
    rerank: bool,
    top_k: int,
    bm25_filter: dict[str, Any] | None = None,
    dense_device_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run retrieval and return list of {doc_id, chunk_id, content, score, device_name}."""

    if mode == "bm25":
        hits = bm25_search(es, query, top_n=top_k, extra_filter=bm25_filter)
        result = hits[:top_k]

    elif mode == "dense":
        d_hits = dense_search(es, embed_query(query), top_n=top_k, device_filter=dense_device_filter)
        # Enrich with doc_id/content from content index
        chunk_ids = [h["chunk_id"] for h in d_hits]
        content_map = _fetch_content_by_chunk_ids(es, chunk_ids)
        result = []
        for h in d_hits:
            cid = h["chunk_id"]
            meta = content_map.get(cid, {})
            result.append(
                {
                    "doc_id": meta.get("doc_id") or "",
                    "chunk_id": cid,
                    "content": meta.get("content") or "",
                    "score": h["score"],
                    "device_name": meta.get("device_name") or h["device_name"],
                }
            )

    elif mode == "hybrid":
        fetch_n = top_k * 4
        bm25_hits = bm25_search(es, query, top_n=fetch_n, extra_filter=bm25_filter)
        d_hits = dense_search(es, embed_query(query), top_n=fetch_n, device_filter=dense_device_filter)

        bm25_ranked = [h["chunk_id"] for h in bm25_hits]
        dense_ranked = [h["chunk_id"] for h in d_hits]

        fused = rrf_fuse(bm25_ranked, dense_ranked, k=RRF_K)

        # Collect all chunk_ids needed
        fused_ids = [cid for cid, _ in fused[: top_k * 2]]
        content_map = _fetch_content_by_chunk_ids(es, fused_ids)

        # Build dense score lookup (for device_name fallback)
        dense_device_map = {h["chunk_id"]: h["device_name"] for h in d_hits}
        # BM25 content already has doc_id/device; build lookup
        bm25_content_map = {h["chunk_id"]: h for h in bm25_hits}

        result = []
        for cid, rrf_score in fused[:top_k]:
            meta = content_map.get(cid) or bm25_content_map.get(cid, {})
            result.append(
                {
                    "doc_id": meta.get("doc_id") or bm25_content_map.get(cid, {}).get("doc_id") or "",
                    "chunk_id": cid,
                    "content": meta.get("content") or "",
                    "score": rrf_score,
                    "device_name": meta.get("device_name") or dense_device_map.get(cid) or "",
                }
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Rerank if requested
    if rerank and result:
        reranker = get_reranker()
        texts = [r["content"] for r in result]
        pairs = [(query, t) for t in texts]
        scores = reranker.model.predict(pairs, batch_size=32, show_progress_bar=False)
        for r, s in zip(result, scores):
            r["original_score"] = r["score"]
            r["score"] = float(s)
        result.sort(key=lambda x: x["score"], reverse=True)

    return result[:top_k]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_contamination(
    top_docs: list[dict[str, Any]],
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
    k: int,
) -> float:
    """cont@k = |{d in top-k : device(d) != target AND d not in shared}| / k"""
    top_k = top_docs[:k]
    if not top_k:
        return 0.0
    contaminated = 0
    for doc in top_k:
        doc_id = doc["doc_id"]
        if not doc_id:
            continue
        doc_device = doc_device_map.get(doc_id, doc.get("device_name", "")).strip().upper()
        tgt = target_device.strip().upper()
        is_same_device = doc_device == tgt
        is_shared = doc_id in shared_ids
        if not is_same_device and not is_shared:
            contaminated += 1
    return contaminated / k


def compute_gold_hit(
    top_docs: list[dict[str, Any]],
    gold_ids: list[str],
    k: int,
) -> bool:
    gold_set = set(gold_ids)
    top_k_ids = {d["doc_id"] for d in top_docs[:k] if d["doc_id"]}
    return bool(gold_set & top_k_ids)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment() -> None:
    print("Loading policy data...")
    doc_device_map = load_doc_scope(DOC_SCOPE_PATH)
    shared_ids = load_shared_ids(SHARED_IDS_PATH)
    queries = load_eval(EVAL_PATH)
    print(f"  doc_scope: {len(doc_device_map)} entries, shared: {len(shared_ids)}, queries: {len(queries)}")

    es = _build_es()
    print(f"  ES host: {search_settings.es_host}")

    # Pre-warm embedding + reranker models
    print("Pre-warming models (embedding + reranker)...")
    _ = embed_query("warm up query")
    get_reranker()
    print("  Models ready.")

    # Per-query results storage
    per_query: list[dict[str, Any]] = []

    total = len(queries)
    for qi, q in enumerate(queries):
        q_id = str(q.get("q_id") or qi)
        question_orig = str(q.get("question") or "")
        question_masked = str(q.get("question_masked") or "")
        target_device = str(q.get("canonical_device_name") or "")
        allowed_devices = [str(d) for d in (q.get("allowed_devices") or [])]
        gold_ids_loose = [str(g) for g in (q.get("gold_doc_ids") or [])]
        gold_ids_strict = [str(g) for g in (q.get("gold_doc_ids_strict") or [])]
        scope_obs = str(q.get("scope_observability") or "")

        if qi % 50 == 0:
            print(f"  [{qi}/{total}] q_id={q_id} device={target_device}")

        if not target_device or not gold_ids_loose:
            continue

        # Build device filter for B4/B4.5
        # BM25 filter uses doc_id list (all docs for target device)
        target_doc_ids = [did for did, dev in doc_device_map.items() if dev.strip().upper() == target_device.strip().upper()]
        b4_bm25_filter: dict[str, Any] | None = None
        b4_dense_filter: list[str] | None = None
        if target_doc_ids:
            b4_bm25_filter = {"terms": {"doc_id": target_doc_ids}}
            b4_dense_filter = allowed_devices if allowed_devices else [target_device]

        # B4.5: shared OR device
        b45_bm25_filter: dict[str, Any] | None = None
        b45_dense_filter: list[str] | None = None
        if target_doc_ids:
            shared_list = sorted(shared_ids)
            should: list[dict[str, Any]] = [{"terms": {"doc_id": target_doc_ids}}]
            if shared_list:
                should.append({"terms": {"doc_id": shared_list}})
            b45_bm25_filter = {"bool": {"should": should, "minimum_should_match": 1}}
            b45_dense_filter = allowed_devices if allowed_devices else [target_device]

        conditions: list[tuple[str, str, str, bool, dict[str, Any] | None, list[str] | None]] = [
            # (cond_name, query_text, mode, rerank, bm25_filter, dense_device_filter)
            ("B0_orig",     question_orig,   "bm25",   False, None,            None),
            ("B1_orig",     question_orig,   "dense",  False, None,            None),
            ("B2_orig",     question_orig,   "hybrid", False, None,            None),
            ("B3_orig",     question_orig,   "hybrid", True,  None,            None),
            ("B0_masked",   question_masked, "bm25",   False, None,            None),
            ("B1_masked",   question_masked, "dense",  False, None,            None),
            ("B2_masked",   question_masked, "hybrid", False, None,            None),
            ("B3_masked",   question_masked, "hybrid", True,  None,            None),
            ("B4_masked",   question_masked, "hybrid", True,  b4_bm25_filter,  b4_dense_filter),
            ("B4.5_masked", question_masked, "hybrid", True,  b45_bm25_filter, b45_dense_filter),
        ]

        q_result: dict[str, Any] = {
            "q_id": q_id,
            "target_device": target_device,
            "scope_observability": scope_obs,
            "gold_ids_loose": gold_ids_loose,
            "gold_ids_strict": gold_ids_strict,
            "conditions": {},
        }

        for cond_name, query_text, mode, rerank, bm25_flt, dense_flt in conditions:
            if not query_text:
                continue
            try:
                docs = retrieve(
                    es=es,
                    query=query_text,
                    mode=mode,
                    rerank=rerank,
                    top_k=TOP_K,
                    bm25_filter=bm25_flt,
                    dense_device_filter=dense_flt,
                )
            except Exception as exc:
                logger.warning("Error for q_id=%s cond=%s: %s", q_id, cond_name, exc)
                q_result["conditions"][cond_name] = {"error": str(exc)}
                continue

            cont = compute_contamination(docs, target_device, doc_device_map, shared_ids, TOP_K)
            gold_hit_loose = compute_gold_hit(docs, gold_ids_loose, TOP_K)
            gold_hit_strict = compute_gold_hit(docs, gold_ids_strict, TOP_K)
            top_doc_ids = [d["doc_id"] for d in docs]

            q_result["conditions"][cond_name] = {
                "cont@10": cont,
                "gold_hit_loose": gold_hit_loose,
                "gold_hit_strict": gold_hit_strict,
                "top_doc_ids": top_doc_ids,
            }

        per_query.append(q_result)

    print(f"\nSaving results to {OUT_PATH} ...")
    write_json(OUT_PATH, per_query)  # type: ignore[arg-type]
    print("Done.")

    _print_summary(per_query)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _print_summary(per_query: list[dict[str, Any]]) -> None:
    conditions = ALL_CONDITIONS

    # Aggregate by condition
    agg: dict[str, dict[str, list[float]]] = {}
    for cond in conditions:
        agg[cond] = {"cont": [], "gold_strict": [], "gold_loose": []}

    for q in per_query:
        for cond in conditions:
            cdata = q.get("conditions", {}).get(cond)
            if not cdata or "error" in cdata:
                continue
            agg[cond]["cont"].append(float(cdata.get("cont@10") or 0.0))
            agg[cond]["gold_strict"].append(1.0 if cdata.get("gold_hit_strict") else 0.0)
            agg[cond]["gold_loose"].append(1.0 if cdata.get("gold_hit_loose") else 0.0)

    n_total = len(per_query)

    print(f"\n{'=' * 72}")
    print(f"--- ALL (n={n_total}) ---")
    print(f"{'condition':<20}  {'cont@10':>8}  {'gold_strict':>14}  {'gold_loose':>12}")
    print("-" * 60)
    for cond in conditions:
        data = agg[cond]
        n = len(data["cont"])
        if n == 0:
            continue
        cont_avg = sum(data["cont"]) / n
        gs = sum(data["gold_strict"])
        gl = sum(data["gold_loose"])
        print(
            f"  {cond:<20}  {cont_avg:>8.3f}  "
            f"{int(gs):>5}/{n_total:<5}    "
            f"{int(gl):>5}/{n_total}"
        )

    # By scope_observability
    scope_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in per_query:
        scope_groups[q.get("scope_observability") or "unknown"].append(q)

    for scope, rows in sorted(scope_groups.items()):
        ns = len(rows)
        print(f"\n--- scope={scope} (n={ns}) ---")
        print(f"{'condition':<20}  {'cont@10':>8}  {'gold_strict':>14}  {'gold_loose':>12}")
        print("-" * 60)
        for cond in conditions:
            conts, gss, gls = [], [], []
            for q in rows:
                cdata = q.get("conditions", {}).get(cond)
                if not cdata or "error" in cdata:
                    continue
                conts.append(float(cdata.get("cont@10") or 0.0))
                gss.append(1.0 if cdata.get("gold_hit_strict") else 0.0)
                gls.append(1.0 if cdata.get("gold_hit_loose") else 0.0)
            if not conts:
                continue
            print(
                f"  {cond:<20}  {sum(conts)/len(conts):>8.3f}  "
                f"{int(sum(gss)):>5}/{ns:<5}    "
                f"{int(sum(gls)):>5}/{ns}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--eval-set", default=None, help="Override EVAL_PATH")
    _parser.add_argument("--output", default=None, help="Override OUT_PATH")
    _args = _parser.parse_args()
    if _args.eval_set:
        EVAL_PATH = Path(_args.eval_set)
    if _args.output:
        OUT_PATH = Path(_args.output)
    run_experiment()
