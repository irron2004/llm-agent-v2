"""P9b: Margin-Gated Verifier — P7+ top-1 + conditional verification.

Algorithm:
  1. Compute P7+ device mass ranking (rank-decay weighted)
  2. If margin(top-1, top-2) > threshold → use top-1 (same as P9a)
  3. If margin ≤ threshold → run hard retrieval for top-M candidates,
     pick device with best evidence score
  4. Return results from selected device

This targets the 20 scope miss cases where target is in P7+ top-3
but P9a's unconditional top-1 picks the wrong device.

Conditions:
  P9b_m03     margin_threshold=0.3, verify_m=2
  P9b_m05     margin_threshold=0.5, verify_m=2
  P9b_m07     margin_threshold=0.7, verify_m=2
  P9b_m03_v3  margin_threshold=0.3, verify_m=3
  P9b_m05_v3  margin_threshold=0.5, verify_m=3

Baselines (from cached):
  B3_masked, B4_masked, P7plus_masked, P9a_masked

Output: data/paper_a/p9b_results.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

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
from scripts.paper_a._io import JsonValue, read_jsonl, write_json

LOG_FILE = ROOT / "data/paper_a/p9b_experiment.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTENT_INDEX = "chunk_v3_content"
EMBED_INDEX = "chunk_v3_embed_bge_m3_v1"
TOP_K = 10
RRF_K = 60

EVAL_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl"
DOC_SCOPE_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/doc_scope.jsonl"
SHARED_IDS_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/shared_doc_ids.txt"
MASKED_HYBRID_PATH = ROOT / "data/paper_a/masked_hybrid_results.json"
P6P7_RESULTS_PATH = ROOT / "data/paper_a/masked_p6p7_results.json"
P9A_RESULTS_PATH = ROOT / "data/paper_a/p9a_results.json"
OUT_PATH = ROOT / "data/paper_a/p9b_results.json"


def normalize_device_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.strip().upper())


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
# Policy data
# ---------------------------------------------------------------------------
def load_doc_scope(path: Path) -> dict[str, str]:
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


def build_device_to_doc_ids(doc_device_map: dict[str, str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for doc_id, device in doc_device_map.items():
        norm = normalize_device_name(device)
        if norm:
            result[norm].append(doc_id)
    return dict(result)


def build_allowed_devices_map(
    doc_device_map: dict[str, str],
    es: Elasticsearch,
) -> dict[str, list[str]]:
    upper_to_raw: dict[str, set[str]] = defaultdict(set)
    for _doc_id, device in doc_device_map.items():
        if device:
            raw = device.strip()
            upper_to_raw[normalize_device_name(raw)].add(raw)
    try:
        resp = es.search(
            index=EMBED_INDEX,
            body={
                "size": 0,
                "aggs": {"devices": {"terms": {"field": "device_name", "size": 200}}},
            },
        )
        for bucket in resp["aggregations"]["devices"]["buckets"]:
            raw = bucket["key"]
            norm = normalize_device_name(raw)
            if norm:
                upper_to_raw[norm].add(raw)
        logger.info("  ES device_name variants loaded (%d devices)", len(upper_to_raw))
    except Exception:
        logger.warning("Failed to load ES device_name variants")
    return {k: sorted(v) for k, v in upper_to_raw.items()}


# ---------------------------------------------------------------------------
# Embedding + Reranker
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


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(device=rag_settings.embedding_device)
    return _reranker


# ---------------------------------------------------------------------------
# Retrieval (same as P9a — chunk-level RRF + doc dedup)
# ---------------------------------------------------------------------------
def bm25_search(
    es: Elasticsearch,
    query: str,
    top_n: int,
    extra_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
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
        hits.append({
            "doc_id": str(src.get("doc_id") or ""),
            "chunk_id": str(src.get("chunk_id") or h["_id"]),
            "content": str(src.get("content") or src.get("search_text") or ""),
            "score": float(h["_score"] or 0.0),
            "device_name": str(src.get("device_name") or ""),
        })
    return hits


def dense_search(
    es: Elasticsearch,
    query_vec: list[float],
    top_n: int,
    device_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
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
        hits.append({
            "chunk_id": str(src.get("chunk_id") or h["_id"]),
            "score": float(h["_score"] or 0.0),
            "device_name": str(src.get("device_name") or ""),
        })
    return hits


def _fetch_content_by_chunk_ids(
    es: Elasticsearch, chunk_ids: list[str]
) -> dict[str, dict[str, Any]]:
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


def rrf_fuse(*rank_lists: list[str], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve_hybrid_rerank(
    *,
    es: Elasticsearch,
    query: str,
    top_k: int,
    bm25_filter: dict[str, Any] | None = None,
    dense_device_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval at chunk level with doc dedup."""
    fetch_n = top_k * 4
    bm25_hits = bm25_search(es, query, fetch_n, extra_filter=bm25_filter)
    qvec = embed_query(query)
    dense_hits = dense_search(es, qvec, fetch_n, device_filter=dense_device_filter)

    bm25_ranked = [h["chunk_id"] for h in bm25_hits]
    dense_ranked = [h["chunk_id"] for h in dense_hits]
    fused = rrf_fuse(bm25_ranked, dense_ranked)

    fused_ids = [cid for cid, _ in fused[:top_k * 2]]
    content_map = _fetch_content_by_chunk_ids(es, fused_ids)
    bm25_content_map = {h["chunk_id"]: h for h in bm25_hits}

    result: list[dict[str, Any]] = []
    for cid, rrf_score in fused[:top_k]:
        meta = content_map.get(cid) or bm25_content_map.get(cid, {})
        result.append({
            "doc_id": meta.get("doc_id") or bm25_content_map.get(cid, {}).get("doc_id") or "",
            "chunk_id": cid,
            "content": meta.get("content") or "",
            "score": rrf_score,
            "device_name": meta.get("device_name") or "",
        })

    # Rerank
    if result:
        reranker = get_reranker()
        reranker_impl = cast(Any, reranker)
        texts = [r["content"] for r in result]
        pairs = [(query, t) for t in texts]
        scores = reranker_impl.model.predict(
            pairs, batch_size=32, show_progress_bar=False
        )
        for r, s in zip(result, scores):
            r["original_score"] = r["score"]
            r["score"] = float(s)
        result.sort(key=lambda x: x["score"], reverse=True)

    # Doc-level dedup
    seen_docs: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in result:
        did = r["doc_id"]
        if did and did not in seen_docs:
            seen_docs.add(did)
            deduped.append(r)

    return deduped[:top_k]


# ---------------------------------------------------------------------------
# Device proposal — returns full ranking
# ---------------------------------------------------------------------------
def p7plus_device_ranking(
    p7plus_doc_ids: list[str],
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> list[tuple[str, float]]:
    """Get ranked device list from P7+ results using rank-decay weighting.

    Returns [(device_norm, score), ...] sorted by score desc.
    """
    scores: dict[str, float] = defaultdict(float)
    for rank, did in enumerate(p7plus_doc_ids):
        if did in shared_ids:
            continue
        device = doc_device_map.get(did, "")
        norm = normalize_device_name(device)
        if norm:
            w = 1.0 / np.log2(rank + 2)
            scores[norm] += w
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def evidence_score(hits: list[dict[str, Any]], mode: str = "mean_top3") -> float:
    """Compute evidence score for a device's retrieval results."""
    if not hits:
        return -999.0
    scores = [h["score"] for h in hits]
    if mode == "max":
        return max(scores)
    elif mode == "mean_top3":
        top3 = sorted(scores, reverse=True)[:3]
        return sum(top3) / len(top3)
    elif mode == "sum":
        return sum(scores)
    return sum(scores)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_gold_hit(doc_ids: list[str], gold_ids: list[str], k: int) -> bool:
    gold_set = set(gold_ids)
    return any(d in gold_set for d in doc_ids[:k])


def compute_contamination_hits(
    hits: list[dict[str, Any]],
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
    k: int,
) -> float:
    tgt_norm = normalize_device_name(target_device)
    contam = 0
    total = 0
    for h in hits[:k]:
        did = h["doc_id"]
        if did in shared_ids:
            continue
        device = doc_device_map.get(did, "")
        norm = normalize_device_name(device)
        if not norm:
            continue
        total += 1
        if norm != tgt_norm:
            contam += 1
    return contam / total if total > 0 else 0.0


def compute_mrr(doc_ids: list[str], gold_ids: list[str], k: int) -> float:
    gold_set = set(gold_ids)
    for rank, doc_id in enumerate(doc_ids[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# P9b conditions
# ---------------------------------------------------------------------------
P9B_CONDITIONS = [
    # (name, margin_threshold, verify_m, evidence_mode)
    ("P9b_m03", 0.3, 2, "mean_top3"),
    ("P9b_m05", 0.5, 2, "mean_top3"),
    ("P9b_m07", 0.7, 2, "mean_top3"),
    ("P9b_m03_v3", 0.3, 3, "mean_top3"),
    ("P9b_m05_v3", 0.5, 3, "mean_top3"),
]

ALL_CONDITIONS_ORDER = [
    "B3_masked",
    "B4_masked",
    "P7plus_masked",
    "P9a_masked",
    "P9b_m03",
    "P9b_m05",
    "P9b_m07",
    "P9b_m03_v3",
    "P9b_m05_v3",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    logger.info("P9b experiment starting (pid=%d)", os.getpid())

    # Load policy data
    doc_device_map = load_doc_scope(DOC_SCOPE_PATH)
    shared_ids = load_shared_ids(SHARED_IDS_PATH)
    device_doc_ids = build_device_to_doc_ids(doc_device_map)
    logger.info("  doc_scope=%d, shared=%d, devices=%d",
                len(doc_device_map), len(shared_ids), len(device_doc_ids))

    # Load cached results
    with MASKED_HYBRID_PATH.open(encoding="utf-8") as f:
        masked_hybrid: list[dict[str, Any]] = json.load(f)

    # Load P7+ results
    p7plus_map: dict[str, dict[str, Any]] = {}
    if P6P7_RESULTS_PATH.exists():
        with P6P7_RESULTS_PATH.open(encoding="utf-8") as f:
            p7data: list[dict[str, Any]] = json.load(f)
        for row in p7data:
            q_id = str(row.get("q_id") or "")
            if q_id:
                p7plus_map[q_id] = row

    # Load P9a results for comparison
    p9a_map: dict[str, dict[str, Any]] = {}
    if P9A_RESULTS_PATH.exists():
        with P9A_RESULTS_PATH.open(encoding="utf-8") as f:
            p9adata: list[dict[str, Any]] = json.load(f)
        for row in p9adata:
            q_id = str(row.get("q_id") or "")
            if q_id:
                p9a_map[q_id] = row

    # Load eval set
    eval_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(EVAL_PATH):
        assert isinstance(row, dict)
        q_id = str(row.get("q_id") or "")
        if q_id:
            eval_rows[q_id] = row
    logger.info("  eval=%d queries", len(eval_rows))

    # Connect ES
    es = _build_es()
    allowed_devices_map = build_allowed_devices_map(doc_device_map, es)

    # Warm models
    _ = embed_query("warm up")
    get_reranker()
    logger.info("Models ready.")

    # Run experiment
    per_query: list[dict[str, Any]] = []
    total = len(masked_hybrid)
    start_time = time.time()

    # Stats for margin-gated triggering
    verify_triggered: dict[str, int] = defaultdict(int)

    for qi, q in enumerate(masked_hybrid):
        q_id = str(q.get("q_id") or qi)
        target_device = str(q.get("target_device") or "")
        scope_obs = str(q.get("scope_observability") or "")
        gold_ids_loose = [str(g) for g in (q.get("gold_ids_loose") or [])]
        gold_ids_strict = [str(g) for g in (q.get("gold_ids_strict") or [])]

        if not target_device or not gold_ids_loose:
            continue

        eval_row = eval_rows.get(q_id, {})
        question_masked = str(eval_row.get("question_masked") or "")
        if not question_masked:
            continue

        elapsed = time.time() - start_time
        qps = (qi + 1) / elapsed if elapsed > 0 else 0
        if qi % 50 == 0:
            logger.info("[%d/%d] q_id=%s device=%s (%.1f q/s)",
                        qi, total, q_id, target_device, qps)

        tgt_norm = normalize_device_name(target_device)

        q_result: dict[str, Any] = {
            "q_id": q_id,
            "target_device": target_device,
            "scope_observability": scope_obs,
            "gold_ids_loose": gold_ids_loose,
            "gold_ids_strict": gold_ids_strict,
            "conditions": {},
        }

        # Copy baselines
        conditions_cached = q.get("conditions") or {}
        for cond_name in ["B3_masked", "B4_masked"]:
            cdata = conditions_cached.get(cond_name, {})
            if cdata and "error" not in cdata:
                cached_doc_ids = [str(d) for d in (cdata.get("top_doc_ids") or [])]
                q_result["conditions"][cond_name] = {
                    "cont@10": float(cdata.get("cont@10") or 0.0),
                    "gold_hit_strict": bool(cdata.get("gold_hit_strict") or False),
                    "gold_hit_loose": bool(cdata.get("gold_hit_loose") or False),
                    "mrr": compute_mrr(cached_doc_ids, gold_ids_strict, TOP_K),
                    "top_doc_ids": cached_doc_ids[:TOP_K],
                }

        # Copy P7+
        p7plus_row = p7plus_map.get(q_id, {})
        p7plus_cond = (p7plus_row.get("conditions") or {}).get("P7plus_masked", {})
        p7plus_doc_ids: list[str] = []
        if p7plus_cond and "error" not in p7plus_cond:
            p7plus_doc_ids = [str(d) for d in (p7plus_cond.get("top_doc_ids") or [])]
            q_result["conditions"]["P7plus_masked"] = {
                "cont@10": float(p7plus_cond.get("cont@10") or 0.0),
                "gold_hit_strict": bool(p7plus_cond.get("gold_hit_strict") or False),
                "gold_hit_loose": bool(p7plus_cond.get("gold_hit_loose") or False),
                "mrr": compute_mrr(p7plus_doc_ids, gold_ids_strict, TOP_K),
                "top_doc_ids": p7plus_doc_ids[:TOP_K],
            }

        # Copy P9a
        p9a_row = p9a_map.get(q_id, {})
        p9a_cond = (p9a_row.get("conditions") or {}).get("P9a_masked", {})
        if p9a_cond and "error" not in p9a_cond:
            p9a_doc_ids = [str(d) for d in (p9a_cond.get("top_doc_ids") or [])]
            q_result["conditions"]["P9a_masked"] = {
                "cont@10": float(p9a_cond.get("cont@10") or 0.0),
                "gold_hit_strict": bool(p9a_cond.get("gold_hit_strict") or False),
                "gold_hit_loose": bool(p9a_cond.get("gold_hit_loose") or False),
                "mrr": compute_mrr(p9a_doc_ids, gold_ids_strict, TOP_K),
                "top_doc_ids": p9a_doc_ids[:TOP_K],
            }

        # P7+ device ranking
        device_ranking = p7plus_device_ranking(p7plus_doc_ids, doc_device_map, shared_ids)

        # Compute margin
        margin = 0.0
        if len(device_ranking) >= 2:
            margin = device_ranking[0][1] - device_ranking[1][1]
        top1_device = device_ranking[0][0] if device_ranking else None

        # Cache retrieval results per device (avoid duplicate retrievals across conditions)
        retrieval_cache: dict[str, list[dict[str, Any]]] = {}

        def get_device_hits(device: str) -> list[dict[str, Any]]:
            if device in retrieval_cache:
                return retrieval_cache[device]
            dev_doc_ids = [
                did for did in device_doc_ids.get(device, [])
                if did not in shared_ids
            ]
            if not dev_doc_ids:
                retrieval_cache[device] = []
                return []
            bm25_filter: dict[str, Any] = {"terms": {"doc_id": dev_doc_ids}}
            dense_filter = allowed_devices_map.get(device, [device])
            hits = retrieve_hybrid_rerank(
                es=es,
                query=question_masked,
                top_k=TOP_K,
                bm25_filter=bm25_filter,
                dense_device_filter=dense_filter,
            )
            retrieval_cache[device] = hits
            return hits

        # Run P9b conditions
        for cond_name, margin_thresh, verify_m, ev_mode in P9B_CONDITIONS:
            if not top1_device:
                q_result["conditions"][cond_name] = {
                    "error": "no_proposal",
                    "proposed_device": "",
                    "scope_correct": False,
                }
                continue

            try:
                if margin > margin_thresh or len(device_ranking) < 2:
                    # High confidence: use top-1 directly (same as P9a)
                    selected_device = top1_device
                    verified = False
                else:
                    # Low confidence: verify top-M candidates
                    verify_triggered[cond_name] += 1
                    verified = True
                    candidates = [d for d, _ in device_ranking[:verify_m]]
                    ev_scores: dict[str, float] = {}
                    for cand in candidates:
                        cand_hits = get_device_hits(cand)
                        ev_scores[cand] = evidence_score(cand_hits, mode=ev_mode)
                    selected_device = max(ev_scores, key=ev_scores.get)  # type: ignore[arg-type]

                scope_correct = (selected_device == tgt_norm)
                hits = get_device_hits(selected_device)
                top_doc_ids = [h["doc_id"] for h in hits[:TOP_K]]

                q_result["conditions"][cond_name] = {
                    "cont@10": compute_contamination_hits(
                        hits, target_device, doc_device_map, shared_ids, TOP_K
                    ),
                    "gold_hit_strict": compute_gold_hit(top_doc_ids, gold_ids_strict, TOP_K),
                    "gold_hit_loose": compute_gold_hit(top_doc_ids, gold_ids_loose, TOP_K),
                    "mrr": compute_mrr(top_doc_ids, gold_ids_strict, TOP_K),
                    "top_doc_ids": top_doc_ids,
                    "proposed_device": selected_device,
                    "scope_correct": scope_correct,
                    "verified": verified,
                    "margin": round(margin, 4),
                }
            except Exception:
                logger.exception("P9b %s failed for q_id=%s", cond_name, q_id)
                q_result["conditions"][cond_name] = {"error": "exception"}

        per_query.append(q_result)

    elapsed_total = time.time() - start_time
    logger.info("Completed %d queries in %.1fs (%.1f q/s)",
                len(per_query), elapsed_total, len(per_query) / max(elapsed_total, 0.001))

    # Log verification trigger counts
    for cond_name, count in sorted(verify_triggered.items()):
        logger.info("  %s: verification triggered %d/%d (%.1f%%)",
                    cond_name, count, len(per_query), count / len(per_query) * 100)

    # Save
    logger.info("Saving results to %s ...", OUT_PATH)
    serializable: list[JsonValue] = [cast(JsonValue, row) for row in per_query]
    write_json(OUT_PATH, serializable)

    # Print summary
    _print_summary(per_query)


def _print_summary(per_query: list[dict[str, Any]]) -> None:
    n = len(per_query)

    for scope_label, group in [
        ("ALL", per_query),
        ("explicit_device", [q for q in per_query if q.get("scope_observability") == "explicit_device"]),
        ("explicit_equip", [q for q in per_query if q.get("scope_observability") == "explicit_equip"]),
    ]:
        ng = len(group)
        if not group:
            continue

        print(f"\n{'=' * 80}")
        print(f"--- {scope_label} (n={ng}) ---")
        print(f"  {'condition':<20} {'cont@10':>8} {'gold_strict':>14} {'MRR':>6} {'scope_acc':>10}")
        print("-" * 80)

        for cond in ALL_CONDITIONS_ORDER:
            cont, strict, mrr, scope_acc = [], [], [], []
            for q in group:
                c = q.get("conditions", {}).get(cond, {})
                if c and "error" not in c:
                    cont.append(c.get("cont@10", 0))
                    strict.append(1 if c.get("gold_hit_strict") else 0)
                    mrr.append(c.get("mrr", 0))
                    if "scope_correct" in c:
                        scope_acc.append(1 if c["scope_correct"] else 0)

            if not cont:
                continue
            nc = len(cont)
            gs = int(sum(strict))
            sa_str = f"{sum(scope_acc)/len(scope_acc):.3f}" if scope_acc else "N/A"
            print(
                f"  {cond:<20} {sum(cont)/nc:>8.3f} "
                f"{gs:>5}/{ng} ({gs/ng*100:.1f}%) "
                f"{sum(mrr)/nc:>6.3f} "
                f"{sa_str:>10}"
            )

        # Verification trigger rate
        p9b_conds = [c for c in ALL_CONDITIONS_ORDER if c.startswith("P9b")]
        if p9b_conds:
            print(f"\n  Verification trigger rate:")
            for cond in p9b_conds:
                triggered = sum(
                    1 for q in group
                    if q.get("conditions", {}).get(cond, {}).get("verified", False)
                )
                total_valid = sum(
                    1 for q in group
                    if q.get("conditions", {}).get(cond, {}) and "error" not in q["conditions"][cond]
                )
                if total_valid:
                    print(f"    {cond:<20} {triggered}/{total_valid} ({triggered/total_valid*100:.1f}%)")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("FATAL: run() crashed")
        sys.exit(1)
