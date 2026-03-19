"""P9c: TF-IDF Device Profile Proposal — independent device proposal + hard retrieval.

Algorithm:
  Stage 1: Device proposal via TF-IDF device profile
    - Build per-device pseudo-document from chapter titles + doc_id patterns
    - Compute TF-IDF cosine similarity between query and each device profile
    - Optionally combine with P7+ device mass
    - Select top-1 device

  Stage 2: Hard filter retrieval (same as P9a)

Conditions:
  P9c_tfidf        TF-IDF only proposal
  P9c_hybrid_03    0.3*P7plus + 0.7*TF-IDF
  P9c_hybrid_05    0.5*P7plus + 0.5*TF-IDF
  P9c_hybrid_07    0.7*P7plus + 0.3*TF-IDF

Baselines (from cached):
  B3_masked, B4_masked, P7plus_masked, P9a_masked

Output: data/paper_a/p9c_results.json
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import rag_settings, search_settings
from backend.llm_infrastructure.reranking.adapters.cross_encoder import (
    CrossEncoderReranker,
)
from backend.services.embedding_service import EmbeddingService
from scripts.paper_a._io import JsonValue, read_jsonl, write_json

LOG_FILE = ROOT / "data/paper_a/p9c_experiment.log"
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
OUT_PATH = ROOT / "data/paper_a/p9c_results.json"


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
    except Exception:
        pass
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
# Retrieval (same as P9a)
# ---------------------------------------------------------------------------
def bm25_search(es, query, top_n, extra_filter=None):
    must_clause = {
        "multi_match": {
            "query": query,
            "fields": ["search_text^1.0", "content^0.8"],
            "type": "best_fields",
        }
    }
    bool_q: dict[str, Any] = {"must": must_clause}
    if extra_filter:
        bool_q["filter"] = extra_filter
    body = {
        "query": {"bool": bool_q},
        "size": top_n,
        "_source": ["doc_id", "chunk_id", "content", "device_name", "search_text"],
    }
    resp = es.search(index=CONTENT_INDEX, body=body)
    return [
        {
            "doc_id": str(h["_source"].get("doc_id") or ""),
            "chunk_id": str(h["_source"].get("chunk_id") or h["_id"]),
            "content": str(h["_source"].get("content") or h["_source"].get("search_text") or ""),
            "score": float(h["_score"] or 0.0),
            "device_name": str(h["_source"].get("device_name") or ""),
        }
        for h in resp["hits"]["hits"]
    ]


def dense_search(es, query_vec, top_n, device_filter=None):
    knn_q: dict[str, Any] = {
        "field": "embedding",
        "query_vector": query_vec,
        "k": top_n,
        "num_candidates": top_n * 2,
    }
    if device_filter:
        knn_q["filter"] = {"terms": {"device_name": device_filter}}
    body = {"knn": knn_q, "size": top_n, "_source": ["chunk_id", "device_name"]}
    resp = es.search(index=EMBED_INDEX, body=body)
    return [
        {
            "chunk_id": str(h["_source"].get("chunk_id") or h["_id"]),
            "score": float(h["_score"] or 0.0),
            "device_name": str(h["_source"].get("device_name") or ""),
        }
        for h in resp["hits"]["hits"]
    ]


def _fetch_content_by_chunk_ids(es, chunk_ids):
    if not chunk_ids:
        return {}
    body = {
        "query": {"terms": {"chunk_id": chunk_ids}},
        "size": len(chunk_ids),
        "_source": ["doc_id", "chunk_id", "content", "search_text", "device_name"],
    }
    resp = es.search(index=CONTENT_INDEX, body=body)
    result = {}
    for h in resp["hits"]["hits"]:
        src = h["_source"]
        cid = str(src.get("chunk_id") or h["_id"])
        result[cid] = {
            "doc_id": str(src.get("doc_id") or ""),
            "content": str(src.get("content") or src.get("search_text") or ""),
            "device_name": str(src.get("device_name") or ""),
        }
    return result


def rrf_fuse(*rank_lists, k=60):
    scores: dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve_hybrid_rerank(*, es, query, top_k, bm25_filter=None, dense_device_filter=None):
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

    result = []
    for cid, rrf_score in fused[:top_k]:
        meta = content_map.get(cid) or bm25_content_map.get(cid, {})
        result.append({
            "doc_id": meta.get("doc_id") or bm25_content_map.get(cid, {}).get("doc_id") or "",
            "chunk_id": cid,
            "content": meta.get("content") or "",
            "score": rrf_score,
            "device_name": meta.get("device_name") or "",
        })

    if result:
        reranker = get_reranker()
        reranker_impl = cast(Any, reranker)
        texts = [r["content"] for r in result]
        pairs = [(query, t) for t in texts]
        scores = reranker_impl.model.predict(pairs, batch_size=32, show_progress_bar=False)
        for r, s in zip(result, scores):
            r["original_score"] = r["score"]
            r["score"] = float(s)
        result.sort(key=lambda x: x["score"], reverse=True)

    seen_docs: set[str] = set()
    deduped = []
    for r in result:
        did = r["doc_id"]
        if did and did not in seen_docs:
            seen_docs.add(did)
            deduped.append(r)
    return deduped[:top_k]


# ---------------------------------------------------------------------------
# TF-IDF Device Profile
# ---------------------------------------------------------------------------
def build_device_profiles(
    es: Elasticsearch,
    device_doc_ids: dict[str, list[str]],
    shared_ids: set[str],
) -> dict[str, str]:
    """Build per-device pseudo-document from chapter titles.

    Returns {device_norm -> profile_text}
    """
    logger.info("Building device profiles from ES...")
    profiles: dict[str, list[str]] = defaultdict(list)

    # Aggregate chapters per device
    for device_norm, doc_ids in device_doc_ids.items():
        non_shared = [d for d in doc_ids if d not in shared_ids]
        if not non_shared:
            continue

        # Get unique chapters for this device's documents
        body: dict[str, Any] = {
            "size": 0,
            "query": {"terms": {"doc_id": non_shared[:500]}},  # cap for ES
            "aggs": {
                "chapters": {"terms": {"field": "chapter.keyword", "size": 500}},
                "doc_types": {"terms": {"field": "doc_type.keyword", "size": 50}},
            },
        }
        try:
            resp = es.search(index=CONTENT_INDEX, body=body)
            for bucket in resp["aggregations"]["chapters"]["buckets"]:
                chapter = bucket["key"].strip()
                if chapter and len(chapter) > 2:
                    # Repeat by sqrt(count) to weight by frequency
                    repeat = max(1, int(bucket["doc_count"] ** 0.5))
                    for _ in range(repeat):
                        profiles[device_norm].append(chapter.lower())
        except Exception as exc:
            logger.warning("Failed to get chapters for %s: %s", device_norm, exc)

        # Also add doc_id patterns (they contain component names)
        for doc_id in non_shared:
            # Extract meaningful parts from doc_id
            # e.g., "global_sop_precia_rep_pm_device_net" → "precia pm device net"
            parts = doc_id.replace("global_sop_", "").replace("_", " ")
            profiles[device_norm].append(parts.lower())

    result = {}
    for device, texts in profiles.items():
        result[device] = " ".join(texts)
        logger.info("  %s: %d tokens, %d sources", device, len(result[device].split()), len(texts))

    return result


class TFIDFDeviceProposer:
    """Proposes device for a query using TF-IDF cosine similarity."""

    def __init__(self, device_profiles: dict[str, str]):
        self.devices = sorted(device_profiles.keys())
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[a-z0-9]+",
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        corpus = [device_profiles[d] for d in self.devices]
        self.profile_matrix = self.vectorizer.fit_transform(corpus)
        logger.info("TF-IDF vectorizer: %d features, %d devices",
                    len(self.vectorizer.vocabulary_), len(self.devices))

    def score_query(self, query: str) -> dict[str, float]:
        """Return {device_norm -> tfidf_similarity}."""
        query_vec = self.vectorizer.transform([query.lower()])
        sims = cosine_similarity(query_vec, self.profile_matrix)[0]
        return {d: float(s) for d, s in zip(self.devices, sims)}


# ---------------------------------------------------------------------------
# Device proposal
# ---------------------------------------------------------------------------
def p7plus_device_scores(
    p7plus_doc_ids: list[str],
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for rank, did in enumerate(p7plus_doc_ids):
        if did in shared_ids:
            continue
        device = doc_device_map.get(did, "")
        norm = normalize_device_name(device)
        if norm:
            w = 1.0 / np.log2(rank + 2)
            scores[norm] += w
    return dict(scores)


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1]."""
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_gold_hit(doc_ids, gold_ids, k):
    return bool(set(gold_ids) & set(doc_ids[:k]))


def compute_contamination_hits(hits, target_device, doc_device_map, shared_ids, k):
    tgt_norm = normalize_device_name(target_device)
    contam = total = 0
    for h in hits[:k]:
        did = h["doc_id"]
        if did in shared_ids:
            continue
        dev = normalize_device_name(doc_device_map.get(did, ""))
        if not dev:
            continue
        total += 1
        if dev != tgt_norm:
            contam += 1
    return contam / total if total > 0 else 0.0


def compute_mrr(doc_ids, gold_ids, k):
    gold_set = set(gold_ids)
    for rank, doc_id in enumerate(doc_ids[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
P9C_CONDITIONS = [
    # (name, p7_weight, tfidf_weight)
    ("P9c_tfidf", 0.0, 1.0),       # TF-IDF only
    ("P9c_hybrid_03", 0.3, 0.7),   # 30% P7+ + 70% TF-IDF
    ("P9c_hybrid_05", 0.5, 0.5),   # 50/50
    ("P9c_hybrid_07", 0.7, 0.3),   # 70% P7+ + 30% TF-IDF
    ("P9c_hybrid_09", 0.9, 0.1),   # 90% P7+ + 10% TF-IDF
]

ALL_CONDITIONS_ORDER = [
    "B3_masked", "B4_masked", "P7plus_masked", "P9a_masked",
    "P9c_tfidf", "P9c_hybrid_03", "P9c_hybrid_05", "P9c_hybrid_07", "P9c_hybrid_09",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    logger.info("P9c experiment starting (pid=%d)", os.getpid())

    # Load policy data
    doc_device_map = load_doc_scope(DOC_SCOPE_PATH)
    shared_ids = load_shared_ids(SHARED_IDS_PATH)
    device_doc_ids = build_device_to_doc_ids(doc_device_map)

    # Load cached data
    with MASKED_HYBRID_PATH.open(encoding="utf-8") as f:
        masked_hybrid: list[dict[str, Any]] = json.load(f)

    p7plus_map: dict[str, dict[str, Any]] = {}
    if P6P7_RESULTS_PATH.exists():
        with P6P7_RESULTS_PATH.open(encoding="utf-8") as f:
            p7data = json.load(f)
        for row in p7data:
            q_id = str(row.get("q_id") or "")
            if q_id:
                p7plus_map[q_id] = row

    p9a_map: dict[str, dict[str, Any]] = {}
    if P9A_RESULTS_PATH.exists():
        with P9A_RESULTS_PATH.open(encoding="utf-8") as f:
            p9adata = json.load(f)
        for row in p9adata:
            q_id = str(row.get("q_id") or "")
            if q_id:
                p9a_map[q_id] = row

    eval_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(EVAL_PATH):
        assert isinstance(row, dict)
        q_id = str(row.get("q_id") or "")
        if q_id:
            eval_rows[q_id] = row

    # Connect ES
    es = _build_es()
    allowed_devices_map = build_allowed_devices_map(doc_device_map, es)

    # Build TF-IDF profiles
    device_profiles = build_device_profiles(es, device_doc_ids, shared_ids)
    tfidf_proposer = TFIDFDeviceProposer(device_profiles)

    # Warm models
    _ = embed_query("warm up")
    get_reranker()
    logger.info("Models ready.")

    # Run experiment
    per_query: list[dict[str, Any]] = []
    total = len(masked_hybrid)
    start_time = time.time()

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

        if qi % 50 == 0:
            elapsed = time.time() - start_time
            qps = (qi + 1) / elapsed if elapsed > 0 else 0
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

        # Compute scores
        p7_scores = p7plus_device_scores(p7plus_doc_ids, doc_device_map, shared_ids)
        tfidf_scores = tfidf_proposer.score_query(question_masked)

        p7_norm = normalize_scores(p7_scores)
        tfidf_norm = normalize_scores(tfidf_scores)

        # Retrieval cache
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
            bm25_filter = {"terms": {"doc_id": dev_doc_ids}}
            dense_filter = allowed_devices_map.get(device, [device])
            hits = retrieve_hybrid_rerank(
                es=es, query=question_masked, top_k=TOP_K,
                bm25_filter=bm25_filter, dense_device_filter=dense_filter,
            )
            retrieval_cache[device] = hits
            return hits

        # Run P9c conditions
        for cond_name, p7_w, tfidf_w in P9C_CONDITIONS:
            try:
                # Combine scores
                all_devices = set(p7_norm.keys()) | set(tfidf_norm.keys())
                combined: dict[str, float] = {}
                for d in all_devices:
                    combined[d] = p7_w * p7_norm.get(d, 0.0) + tfidf_w * tfidf_norm.get(d, 0.0)

                if not combined:
                    q_result["conditions"][cond_name] = {"error": "no_scores"}
                    continue

                selected_device = max(combined, key=combined.get)  # type: ignore[arg-type]
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
                    "tfidf_top1": max(tfidf_scores, key=tfidf_scores.get) if tfidf_scores else "",  # type: ignore
                    "tfidf_score": round(tfidf_scores.get(selected_device, 0), 4),
                }
            except Exception:
                logger.exception("P9c %s failed for q_id=%s", cond_name, q_id)
                q_result["conditions"][cond_name] = {"error": "exception"}

        per_query.append(q_result)

    elapsed_total = time.time() - start_time
    logger.info("Completed %d queries in %.1fs (%.1f q/s)",
                len(per_query), elapsed_total, len(per_query) / max(elapsed_total, 0.001))

    # Save
    write_json(OUT_PATH, [cast(JsonValue, row) for row in per_query])
    _print_summary(per_query)


def _print_summary(per_query: list[dict[str, Any]]) -> None:
    for scope_label, group in [
        ("ALL", per_query),
        ("explicit_device", [q for q in per_query if q.get("scope_observability") == "explicit_device"]),
        ("explicit_equip", [q for q in per_query if q.get("scope_observability") == "explicit_equip"]),
    ]:
        ng = len(group)
        if not group:
            continue
        print(f"\n{'=' * 85}")
        print(f"--- {scope_label} (n={ng}) ---")
        print(f"  {'condition':<20} {'cont@10':>8} {'gold_strict':>14} {'MRR':>6} {'scope_acc':>10}")
        print("-" * 85)

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
            sa = f"{sum(scope_acc)/len(scope_acc):.3f}" if scope_acc else "N/A"
            print(f"  {cond:<20} {sum(cont)/nc:>8.3f} {gs:>5}/{ng} ({gs/ng*100:.1f}%) {sum(mrr)/nc:>6.3f} {sa:>10}")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("FATAL")
        sys.exit(1)
