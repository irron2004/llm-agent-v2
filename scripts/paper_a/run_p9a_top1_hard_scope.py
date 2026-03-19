"""P9a: Proposal-Only Hard Scope — P7+ top-1 device + hard filter retrieval.

Algorithm:
  1. If hard evidence (parser/alias) → use that device
  2. Else → P7+ device mass top-1 device
  3. Hard filter retrieval with selected device
  4. Optional: shared gate (shared_cap=0,1,2)

No Stage 3 verification — trust the proposal.

Conditions:
  P9a_masked           P7+ top-1 device, shared_cap=0
  P9a_sc1_masked       P7+ top-1 device, shared_cap=1
  P9a_sc2_masked       P7+ top-1 device, shared_cap=2

Baselines (from cached):
  B3_masked, B4_masked, B4.5_masked, P7plus_masked, P8_masked

Output: data/paper_a/p9a_results.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
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

LOG_FILE = ROOT / "data/paper_a/p9a_experiment.log"
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
FETCH_N = TOP_K * 4

EVAL_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl"
DOC_SCOPE_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/doc_scope.jsonl"
SHARED_IDS_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/shared_doc_ids.txt"
MASKED_HYBRID_PATH = ROOT / "data/paper_a/masked_hybrid_results.json"
P6P7_RESULTS_PATH = ROOT / "data/paper_a/masked_p6p7_results.json"
P8_RESULTS_PATH = ROOT / "data/paper_a/p8_results.json"
OUT_PATH = ROOT / "data/paper_a/p9a_results.json"


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
    """Build {NORMALIZED_DEVICE -> [raw ES device_name values]}.

    Uses both doc_scope AND ES aggregation to capture all device_name variants
    (different casing, underscores, etc.).
    """
    upper_to_raw: dict[str, set[str]] = defaultdict(set)
    # From doc_scope
    for _doc_id, device in doc_device_map.items():
        if device:
            raw = device.strip()
            upper_to_raw[normalize_device_name(raw)].add(raw)
    # From ES embed index aggregation (actual stored values)
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
        logger.info("  ES device_name variants loaded (%d normalized devices)", len(upper_to_raw))
    except Exception:
        logger.warning("Failed to load ES device_name aggregation, using doc_scope only")
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
# Retrieval
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
        src = h["_source"]
        hits.append({
            "doc_id": src.get("doc_id", h["_id"]),
            "chunk_id": src.get("chunk_id", h["_id"]),
            "content": src.get("content", ""),
            "device_name": src.get("device_name", ""),
            "score": float(h["_score"]),
        })
    return hits


def dense_search(
    es: Elasticsearch,
    query_vec: list[float],
    top_n: int,
    device_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    knn: dict[str, Any] = {
        "field": "embedding",
        "query_vector": query_vec,
        "k": top_n,
        "num_candidates": top_n * 4,
    }
    if device_filter:
        knn["filter"] = {"terms": {"device_name": device_filter}}
    body: dict[str, Any] = {
        "knn": knn,
        "size": top_n,
        "_source": ["chunk_id", "device_name"],
    }
    resp = es.search(index=EMBED_INDEX, body=body)
    hits = []
    for h in resp["hits"]["hits"]:
        src = h.get("_source", {})
        hits.append({
            "chunk_id": src.get("chunk_id", h["_id"]),
            "device_name": str(src.get("device_name") or ""),
            "score": float(h["_score"]),
        })
    return hits


def rrf_fuse(
    bm25_ranked: list[str],
    dense_ranked: list[str],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """RRF fusion at chunk_id level. Returns [(chunk_id, rrf_score), ...]."""
    scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(bm25_ranked):
        scores[cid] += 1.0 / (k + rank + 1)
    for rank, cid in enumerate(dense_ranked):
        scores[cid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _fetch_content_by_chunk_ids(
    es: Elasticsearch, chunk_ids: list[str]
) -> dict[str, dict[str, str]]:
    """Return {chunk_id -> {doc_id, content, device_name}}."""
    if not chunk_ids:
        return {}
    body: dict[str, Any] = {
        "query": {"terms": {"chunk_id": chunk_ids}},
        "size": len(chunk_ids),
        "_source": ["doc_id", "chunk_id", "content", "device_name"],
    }
    resp = es.search(index=CONTENT_INDEX, body=body)
    result: dict[str, dict[str, str]] = {}
    for h in resp["hits"]["hits"]:
        src = h["_source"]
        cid = src.get("chunk_id", h["_id"])
        result[cid] = {
            "doc_id": src.get("doc_id", ""),
            "content": src.get("content", ""),
            "device_name": src.get("device_name", ""),
        }
    return result


def retrieve_hybrid_rerank(
    *,
    es: Elasticsearch,
    query: str,
    top_k: int,
    bm25_filter: dict[str, Any] | None = None,
    dense_device_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval at chunk level (matching run_masked_hybrid_experiment.py)."""
    fetch_n = top_k * 4
    bm25_hits = bm25_search(es, query, fetch_n, extra_filter=bm25_filter)
    qvec = embed_query(query)
    dense_hits = dense_search(es, qvec, fetch_n, device_filter=dense_device_filter)

    # RRF at chunk_id level
    bm25_ranked = [h["chunk_id"] for h in bm25_hits]
    dense_ranked = [h["chunk_id"] for h in dense_hits]
    fused = rrf_fuse(bm25_ranked, dense_ranked)

    # Fetch content for top candidates
    fused_ids = [cid for cid, _ in fused[:top_k * 2]]
    content_map = _fetch_content_by_chunk_ids(es, fused_ids)
    bm25_content_map = {h["chunk_id"]: h for h in bm25_hits}

    # Build chunk-level results
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
        pairs = [(query, r["content"]) for r in result]
        scores = reranker_impl.model.predict(pairs, batch_size=32, show_progress_bar=False)
        for r, s in zip(result, scores):
            r["original_score"] = r["score"]
            r["score"] = float(s)
        result.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate by doc_id, keeping highest-scored chunk per doc
    seen_docs: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in result:
        did = r["doc_id"]
        if did and did not in seen_docs:
            seen_docs.add(did)
            deduped.append(r)

    return deduped[:top_k]


# ---------------------------------------------------------------------------
# Device proposal from P7+ top-10
# ---------------------------------------------------------------------------
def p7plus_top1_device(
    p7plus_doc_ids: list[str],
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> str | None:
    """Get top-1 device from P7+ results using rank-decay weighting."""
    scores: dict[str, float] = defaultdict(float)
    for rank, did in enumerate(p7plus_doc_ids):
        if did in shared_ids:
            continue
        device = doc_device_map.get(did, "")
        norm = normalize_device_name(device)
        if norm:
            w = 1.0 / np.log2(rank + 2)
            scores[norm] += w
    if not scores:
        return None
    return max(scores, key=scores.get)  # type: ignore[arg-type]


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
# P9a conditions
# ---------------------------------------------------------------------------
P9A_CONDITIONS = [
    # (name, shared_cap)
    ("P9a_masked", 0),
    ("P9a_sc1_masked", 1),
    ("P9a_sc2_masked", 2),
]

BASELINE_FROM_CACHED = ["B3_masked", "B4_masked", "B4.5_masked"]

ALL_CONDITIONS_ORDER = [
    "B3_masked",
    "B4_masked",
    "B4.5_masked",
    "P7plus_masked",
    "P8_masked",
    "P9a_masked",
    "P9a_sc1_masked",
    "P9a_sc2_masked",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    logger.info("P9a experiment starting (pid=%d)", os.getpid())

    # Load policy data
    logger.info("Loading policy data...")
    doc_device_map = load_doc_scope(DOC_SCOPE_PATH)
    shared_ids = load_shared_ids(SHARED_IDS_PATH)
    device_doc_ids = build_device_to_doc_ids(doc_device_map)
    logger.info("  doc_scope=%d, shared=%d, devices=%d",
                len(doc_device_map), len(shared_ids), len(device_doc_ids))

    # Load cached results
    logger.info("Loading cached results...")
    with MASKED_HYBRID_PATH.open(encoding="utf-8") as f:
        masked_hybrid: list[dict[str, Any]] = json.load(f)

    # Load P7+ results
    p7plus_map: dict[str, dict[str, Any]] = {}
    if P6P7_RESULTS_PATH.exists():
        logger.info("Loading P7+ results...")
        with P6P7_RESULTS_PATH.open(encoding="utf-8") as f:
            p7data: list[dict[str, Any]] = json.load(f)
        for row in p7data:
            q_id = str(row.get("q_id") or "")
            if q_id:
                p7plus_map[q_id] = row

    # Load P8 results for comparison
    p8_map: dict[str, dict[str, Any]] = {}
    if P8_RESULTS_PATH.exists():
        logger.info("Loading P8 results...")
        with P8_RESULTS_PATH.open(encoding="utf-8") as f:
            p8data: list[dict[str, Any]] = json.load(f)
        for row in p8data:
            q_id = str(row.get("q_id") or "")
            if q_id:
                p8_map[q_id] = row

    # Load eval set
    logger.info("Loading eval set...")
    eval_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(EVAL_PATH):
        assert isinstance(row, dict)
        q_id = str(row.get("q_id") or "")
        if q_id:
            eval_rows[q_id] = row
    logger.info("  eval=%d queries", len(eval_rows))

    # Connect ES + warm models
    es = _build_es()
    logger.info("ES connected: %s", search_settings.es_host)

    # Build allowed_devices_map AFTER ES connection (needs ES agg)
    allowed_devices_map = build_allowed_devices_map(doc_device_map, es)

    logger.info("Pre-warming models...")
    _ = embed_query("warm up query")
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

        elapsed = time.time() - start_time
        qps = (qi + 1) / elapsed if elapsed > 0 else 0
        logger.info("[%d/%d] q_id=%s device=%s obs=%s (%.1f q/s)",
                    qi, total, q_id, target_device, scope_obs, qps)

        tgt_norm = normalize_device_name(target_device)

        # Build result
        q_result: dict[str, Any] = {
            "q_id": q_id,
            "target_device": target_device,
            "scope_observability": scope_obs,
            "gold_ids_loose": gold_ids_loose,
            "gold_ids_strict": gold_ids_strict,
            "conditions": {},
        }

        # Copy baselines from cached data
        conditions_cached = q.get("conditions") or {}
        for cond_name in BASELINE_FROM_CACHED:
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

        # Copy P7+ from cached
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

        # Copy P8 from cached
        p8_row = p8_map.get(q_id, {})
        p8_cond = (p8_row.get("conditions") or {}).get("P8_masked", {})
        if p8_cond and "error" not in p8_cond:
            p8_doc_ids = [str(d) for d in (p8_cond.get("top_doc_ids") or [])]
            q_result["conditions"]["P8_masked"] = {
                "cont@10": float(p8_cond.get("cont@10") or 0.0),
                "gold_hit_strict": bool(p8_cond.get("gold_hit_strict") or False),
                "gold_hit_loose": bool(p8_cond.get("gold_hit_loose") or False),
                "mrr": compute_mrr(p8_doc_ids, gold_ids_strict, TOP_K),
                "top_doc_ids": p8_doc_ids[:TOP_K],
            }

        # --- P9a: P7+ top-1 device proposal ---
        proposed_device = p7plus_top1_device(p7plus_doc_ids, doc_device_map, shared_ids)
        scope_correct = (proposed_device == tgt_norm) if proposed_device else False

        for cond_name, shared_cap in P9A_CONDITIONS:
            if not proposed_device:
                q_result["conditions"][cond_name] = {
                    "error": "no_proposal",
                    "proposed_device": "",
                    "scope_correct": False,
                }
                continue

            try:
                # Hard filter retrieval with proposed device
                dev_doc_ids = [
                    did for did in device_doc_ids.get(proposed_device, [])
                    if did not in shared_ids
                ]
                if not dev_doc_ids:
                    q_result["conditions"][cond_name] = {
                        "error": "no_docs_for_device",
                        "proposed_device": proposed_device,
                        "scope_correct": scope_correct,
                    }
                    continue

                bm25_filter: dict[str, Any] = {"terms": {"doc_id": dev_doc_ids}}
                dense_filter = allowed_devices_map.get(proposed_device, [proposed_device])

                hits = retrieve_hybrid_rerank(
                    es=es,
                    query=question_masked,
                    top_k=TOP_K,
                    bm25_filter=bm25_filter,
                    dense_device_filter=dense_filter,
                )

                # Apply shared_cap if needed
                if shared_cap > 0 and hits:
                    shared_hits = retrieve_hybrid_rerank(
                        es=es,
                        query=question_masked,
                        top_k=shared_cap,
                        bm25_filter={"terms": {"doc_id": list(shared_ids)}},
                        dense_device_filter=None,
                    )
                    if shared_hits:
                        existing_ids = {h["doc_id"] for h in hits}
                        new_shared = [h for h in shared_hits if h["doc_id"] not in existing_ids]
                        if new_shared:
                            # Replace lowest-scored device hits
                            replace_count = min(len(new_shared), shared_cap)
                            hits = hits[:TOP_K - replace_count] + new_shared[:replace_count]
                            hits.sort(key=lambda x: x["score"], reverse=True)

                top_doc_ids = [h["doc_id"] for h in hits[:TOP_K]]

                q_result["conditions"][cond_name] = {
                    "cont@10": compute_contamination_hits(
                        hits, target_device, doc_device_map, shared_ids, TOP_K
                    ),
                    "gold_hit_strict": compute_gold_hit(top_doc_ids, gold_ids_strict, TOP_K),
                    "gold_hit_loose": compute_gold_hit(top_doc_ids, gold_ids_loose, TOP_K),
                    "mrr": compute_mrr(top_doc_ids, gold_ids_strict, TOP_K),
                    "top_doc_ids": top_doc_ids,
                    "proposed_device": proposed_device,
                    "scope_correct": scope_correct,
                }

                logger.info("  %s: proposed=%s correct=%s", cond_name, proposed_device, scope_correct)

            except Exception:
                logger.exception("P9a %s failed for q_id=%s", cond_name, q_id)
                q_result["conditions"][cond_name] = {
                    "error": "exception",
                    "proposed_device": proposed_device,
                    "scope_correct": scope_correct,
                }

        per_query.append(q_result)

    elapsed_total = time.time() - start_time
    logger.info("Completed %d queries in %.1fs (%.1f q/s)",
                len(per_query), elapsed_total, len(per_query) / max(elapsed_total, 0.001))

    # Save
    logger.info("Saving to %s ...", OUT_PATH)
    serializable: list[JsonValue] = [cast(JsonValue, row) for row in per_query]
    write_json(OUT_PATH, serializable)
    logger.info("Saved %d results.", len(per_query))

    # Print summary
    _print_summary(per_query)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _print_summary(per_query: list[dict[str, Any]]) -> None:
    n_total = len(per_query)

    _print_summary_for_scope(per_query, "ALL", n_total)

    scope_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in per_query:
        scope_groups[q.get("scope_observability", "unknown")].append(q)

    for scope_name in ["explicit_device", "explicit_equip"]:
        group = scope_groups.get(scope_name, [])
        if group:
            _print_summary_for_scope(group, scope_name, len(group))

    # P9a scope accuracy
    print(f"\n{'=' * 72}")
    print("P9a Scope Selection Accuracy")
    print("=" * 72)
    for cond_name, _ in P9A_CONDITIONS:
        scope_correct_all = []
        for q in per_query:
            cdata = q.get("conditions", {}).get(cond_name, {})
            if cdata and "scope_correct" in cdata:
                scope_correct_all.append(1.0 if cdata["scope_correct"] else 0.0)
        if scope_correct_all:
            acc = sum(scope_correct_all) / len(scope_correct_all)
            print(f"  {cond_name:<20}  scope_acc={acc:.3f}  (n={len(scope_correct_all)})")

    for scope_name in ["explicit_device", "explicit_equip"]:
        group = scope_groups.get(scope_name, [])
        if not group:
            continue
        print(f"\n  --- {scope_name} (n={len(group)}) ---")
        for cond_name, _ in P9A_CONDITIONS:
            scope_correct_g = []
            for q in group:
                cdata = q.get("conditions", {}).get(cond_name, {})
                if cdata and "scope_correct" in cdata:
                    scope_correct_g.append(1.0 if cdata["scope_correct"] else 0.0)
            if scope_correct_g:
                acc = sum(scope_correct_g) / len(scope_correct_g)
                print(f"    {cond_name:<20}  scope_acc={acc:.3f}")


def _print_summary_for_scope(
    queries: list[dict[str, Any]], scope_label: str, n: int
) -> None:
    agg: dict[str, dict[str, list[float]]] = {
        c: {"cont": [], "strict": [], "loose": [], "mrr": []}
        for c in ALL_CONDITIONS_ORDER
    }

    for q in queries:
        for cond in ALL_CONDITIONS_ORDER:
            cdata = q.get("conditions", {}).get(cond)
            if not cdata or "error" in cdata:
                continue
            agg[cond]["cont"].append(float(cdata.get("cont@10") or 0.0))
            agg[cond]["strict"].append(1.0 if cdata.get("gold_hit_strict") else 0.0)
            agg[cond]["loose"].append(1.0 if cdata.get("gold_hit_loose") else 0.0)
            agg[cond]["mrr"].append(float(cdata.get("mrr") or 0.0))

    print(f"\n{'=' * 80}")
    print(f"--- {scope_label} (n={n}) ---")
    print(
        f"  {'condition':<20}  {'cont@10':>8}  {'strict':>14}  "
        f"{'loose':>12}  {'MRR':>6}"
    )
    print("-" * 80)
    for cond in ALL_CONDITIONS_ORDER:
        data = agg[cond]
        nc = len(data["cont"])
        if nc == 0:
            continue
        cont_avg = sum(data["cont"]) / nc
        gs = int(sum(data["strict"]))
        gl = int(sum(data["loose"]))
        mrr_avg = sum(data["mrr"]) / nc
        print(
            f"  {cond:<20}  {cont_avg:>8.3f}  {gs:>5}/{n:<5}    "
            f"{gl:>5}/{n}  {mrr_avg:>6.3f}"
        )

    # Delta vs B3
    b3_data = agg["B3_masked"]
    if not b3_data["cont"]:
        return
    b3_cont = sum(b3_data["cont"]) / len(b3_data["cont"])
    b3_strict = sum(b3_data["strict"]) / len(b3_data["strict"])

    print(f"\n  Delta vs B3_masked:")
    print(f"  {'condition':<20}  {'Δcont@10':>10}  {'Δstrict':>10}")
    print("  " + "-" * 50)
    for cond in ALL_CONDITIONS_ORDER[1:]:
        data = agg[cond]
        nc = len(data["cont"])
        if nc == 0:
            continue
        c_avg = sum(data["cont"]) / nc
        s_avg = sum(data["strict"]) / nc
        dc = c_avg - b3_cont
        ds = s_avg - b3_strict
        sign_c = "+" if dc >= 0 else ""
        sign_s = "+" if ds >= 0 else ""
        print(f"  {cond:<20}  {sign_c}{dc:>9.3f}  {sign_s}{ds:>9.3f}")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("FATAL: run() crashed")
        sys.exit(1)
