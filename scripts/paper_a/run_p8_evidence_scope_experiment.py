"""P8: Evidence-Based Scope Selection experiment (HVSR-lite).

Algorithm:
  Stage 1 — Device Hypothesis Generation
    Extract device distribution from B3 unfiltered top-10 (cached).
    Take top-M devices as hypotheses.

  Stage 2 — Per-Hypothesis Hard Retrieval
    For each hypothesis device, run hybrid+rerank with hard device filter.

  Stage 3 — Evidence-Based Selection
    Score each hypothesis by total retrieval score mass.
    Select device with highest evidence.

  Stage 4 — Final Output
    Top-k from selected device + optional shared docs (capped).

Conditions:
  P8_masked           M=3, shared_cap=0
  P8_sc1_masked       M=3, shared_cap=1
  P8_sc2_masked       M=3, shared_cap=2
  P8_m5_masked        M=5, shared_cap=0

Baselines (from cached data):
  B3_masked, B4_masked, B4.5_masked, P7plus_masked

Outputs:
  data/paper_a/p8_results.json
  Console summary table
"""

from __future__ import annotations

import json
import logging
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTENT_INDEX = "chunk_v3_content"
EMBED_INDEX = "chunk_v3_embed_bge_m3_v1"
TOP_K = 10
RRF_K = 60
FETCH_N = TOP_K * 4
PROBE_TOP_K = 40

EVAL_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl"
DOC_SCOPE_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/doc_scope.jsonl"
SHARED_IDS_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/shared_doc_ids.txt"
MASKED_HYBRID_PATH = ROOT / "data/paper_a/masked_hybrid_results.json"
P6P7_RESULTS_PATH = ROOT / "data/paper_a/masked_p6p7_results.json"
OUT_PATH = ROOT / "data/paper_a/p8_results.json"


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


def build_device_to_doc_ids(doc_device_map: dict[str, str]) -> dict[str, list[str]]:
    """Build {device_name_upper -> [doc_id, ...]}."""
    result: dict[str, list[str]] = defaultdict(list)
    for doc_id, device in doc_device_map.items():
        if device:
            result[device.strip().upper()].append(doc_id)
    return dict(result)


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
# Low-level ES retrieval (from run_masked_hybrid_experiment.py)
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
    es: Elasticsearch,
    query: str,
    top_k: int,
    bm25_filter: dict[str, Any] | None = None,
    dense_device_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid+rerank retrieval. Returns [{doc_id, chunk_id, content, score, device_name}]."""
    fetch_n = top_k * 4
    bm25_hits = bm25_search(es, query, top_n=fetch_n, extra_filter=bm25_filter)
    d_hits = dense_search(
        es, embed_query(query), top_n=fetch_n, device_filter=dense_device_filter
    )

    bm25_ranked = [h["chunk_id"] for h in bm25_hits]
    dense_ranked = [h["chunk_id"] for h in d_hits]
    fused = rrf_fuse(bm25_ranked, dense_ranked, k=RRF_K)

    fused_ids = [cid for cid, _ in fused[: top_k * 2]]
    content_map = _fetch_content_by_chunk_ids(es, fused_ids)
    dense_device_map = {h["chunk_id"]: h["device_name"] for h in d_hits}
    bm25_content_map = {h["chunk_id"]: h for h in bm25_hits}

    result = []
    for cid, rrf_score in fused[:top_k]:
        meta = content_map.get(cid) or bm25_content_map.get(cid, {})
        result.append(
            {
                "doc_id": meta.get("doc_id")
                or bm25_content_map.get(cid, {}).get("doc_id")
                or "",
                "chunk_id": cid,
                "content": meta.get("content") or "",
                "score": rrf_score,
                "device_name": meta.get("device_name")
                or dense_device_map.get(cid)
                or "",
            }
        )

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

    return result[:top_k]


# ---------------------------------------------------------------------------
# P8 Algorithm
# ---------------------------------------------------------------------------
def generate_hypotheses(
    b3_doc_ids: list[str],
    doc_device_map: dict[str, str],
    shared_ids: set[str],
    m: int,
) -> list[tuple[str, int]]:
    """Stage 1: Extract top-M device hypotheses from B3 unfiltered results.

    Returns [(device_name_upper, count), ...] sorted by frequency desc.
    """
    device_counts: Counter[str] = Counter()
    for doc_id in b3_doc_ids:
        if doc_id in shared_ids:
            continue
        device = doc_device_map.get(doc_id, "").strip().upper()
        if device:
            device_counts[device] += 1
    return device_counts.most_common(m)


def generate_hypotheses_from_hits(
    probe_hits: list[dict[str, Any]],
    doc_device_map: dict[str, str],
    shared_ids: set[str],
    m: int,
) -> list[tuple[str, int]]:
    device_counts: Counter[str] = Counter()
    for hit in probe_hits:
        doc_id = str(hit.get("doc_id") or "")
        if doc_id in shared_ids:
            continue
        device_name = str(hit.get("device_name") or "").strip().upper()
        if not device_name and doc_id:
            device_name = doc_device_map.get(doc_id, "").strip().upper()
        if device_name:
            device_counts[device_name] += 1
    return device_counts.most_common(m)


def evidence_score(hits: list[dict[str, Any]], mode: str = "sum") -> float:
    """Stage 3: Compute evidence score for a hypothesis's retrieval results.

    Modes:
      sum   — total reranker score (default)
      max   — max reranker score
      top3  — sum of top-3 scores
    """
    if not hits:
        return -999.0
    scores = [h["score"] for h in hits]
    if mode == "sum":
        return sum(scores)
    elif mode == "max":
        return max(scores)
    elif mode == "top3":
        return sum(sorted(scores, reverse=True)[:3])
    return sum(scores)


def run_p8_for_query(
    *,
    es: Elasticsearch,
    query: str,
    b3_doc_ids: list[str],
    target_device: str,
    doc_device_map: dict[str, str],
    device_doc_ids: dict[str, list[str]],
    shared_ids: set[str],
    allowed_devices_map: dict[str, list[str]],
    probe_hits: list[dict[str, Any]] | None = None,
    m: int = 3,
    shared_cap: int = 0,
    score_mode: str = "sum",
) -> dict[str, Any]:
    """Run the full P8 pipeline for a single query.

    Returns:
        {
            "hypotheses": [(device, count), ...],
            "selected_device": str,
            "scope_correct": bool,
            "evidence_scores": {device: score},
            "target_hypothesis_rank": int | None,
            "top_doc_ids": [str],
            "hits": [{doc_id, score, device_name}],
        }
    """
    # Stage 1: Hypothesis generation
    hypotheses = generate_hypotheses(b3_doc_ids, doc_device_map, shared_ids, m)
    hypothesis_source = "cached"
    if not hypotheses and probe_hits:
        hypotheses = generate_hypotheses_from_hits(
            probe_hits, doc_device_map, shared_ids, m
        )
        hypothesis_source = "probe"

    if not hypotheses:
        # Fallback: no device info → return B3 results as-is
        return {
            "hypotheses": [],
            "selected_device": "",
            "scope_correct": False,
            "evidence_scores": {},
            "target_hypothesis_rank": None,
            "top_doc_ids": b3_doc_ids[:TOP_K],
            "hits": [],
            "fallback": True,
            "hypothesis_source": hypothesis_source,
        }

    # Stage 2: Per-hypothesis hard retrieval
    hypothesis_results: dict[str, list[dict[str, Any]]] = {}
    ev_scores: dict[str, float] = {}
    tgt_upper = target_device.strip().upper()

    for device, _count in hypotheses:
        # Build device filter
        dev_doc_ids = [
            doc_id
            for doc_id in device_doc_ids.get(device, [])
            if doc_id not in shared_ids
        ]
        if not dev_doc_ids:
            continue

        bm25_filter: dict[str, Any] = {"terms": {"doc_id": dev_doc_ids}}
        # For dense, use raw device names that map to this upper-case device
        dense_filter = allowed_devices_map.get(device, [device])

        try:
            hits = retrieve_hybrid_rerank(
                es=es,
                query=query,
                top_k=TOP_K,
                bm25_filter=bm25_filter,
                dense_device_filter=dense_filter,
            )
        except Exception as exc:
            logger.warning("P8 retrieval failed for device=%s: %s", device, exc)
            continue

        hypothesis_results[device] = hits
        ev_scores[device] = evidence_score(hits, mode=score_mode)

    if not hypothesis_results:
        return {
            "hypotheses": [(d, c) for d, c in hypotheses],
            "selected_device": "",
            "scope_correct": False,
            "evidence_scores": {},
            "target_hypothesis_rank": None,
            "top_doc_ids": b3_doc_ids[:TOP_K],
            "hits": [],
            "fallback": True,
            "hypothesis_source": hypothesis_source,
        }

    # Stage 3: Evidence-based selection
    selected_device = max(ev_scores.items(), key=lambda item: item[1])[0]
    scope_correct = selected_device == tgt_upper

    # Target device rank in hypotheses
    hyp_devices = [d for d, _ in hypotheses]
    target_rank = hyp_devices.index(tgt_upper) + 1 if tgt_upper in hyp_devices else None

    # Stage 4: Final output
    selected_hits = list(hypothesis_results[selected_device])  # copy to avoid mutation
    top_doc_ids = [h["doc_id"] for h in selected_hits]

    # Optional: replace lowest-ranked device docs with shared docs
    # (P8-01 fix: "replace" strategy instead of "append" to guarantee shared_cap works
    #  even when Stage 2 already returns TOP_K results)
    if shared_cap > 0:
        shared_list = sorted(shared_ids)
        if shared_list:
            try:
                shared_filter: dict[str, Any] = {"terms": {"doc_id": shared_list}}
                shared_hits = retrieve_hybrid_rerank(
                    es=es,
                    query=query,
                    top_k=shared_cap,
                    bm25_filter=shared_filter,
                    dense_device_filter=None,
                )
                # Filter out shared docs that are already in results
                existing_doc_ids = set(top_doc_ids)
                new_shared = [
                    sh for sh in shared_hits if sh["doc_id"] not in existing_doc_ids
                ]
                # Replace from the tail: remove lowest-scored device docs to make room
                for sh in new_shared:
                    if len(selected_hits) >= TOP_K:
                        # Remove the last (lowest-scored) device doc
                        removed = selected_hits.pop()
                        top_doc_ids.remove(removed["doc_id"])
                    selected_hits.append(sh)
                    top_doc_ids.append(sh["doc_id"])
                selected_hits.sort(
                    key=lambda hit: float(hit.get("score") or 0.0), reverse=True
                )
                top_doc_ids = [str(hit.get("doc_id") or "") for hit in selected_hits]
            except Exception as exc:
                logger.warning("Shared retrieval failed: %s", exc)

    return {
        "hypotheses": [(d, c) for d, c in hypotheses],
        "selected_device": selected_device,
        "scope_correct": scope_correct,
        "evidence_scores": {d: round(s, 4) for d, s in ev_scores.items()},
        "target_hypothesis_rank": target_rank,
        "top_doc_ids": top_doc_ids[:TOP_K],
        "hits": [
            {
                "doc_id": h["doc_id"],
                "score": round(h["score"], 4),
                "device_name": h["device_name"],
            }
            for h in selected_hits[:TOP_K]
        ],
        "fallback": False,
        "hypothesis_source": hypothesis_source,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def v_scope(
    doc_id: str,
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> int:
    if not doc_id:
        return 0
    if doc_id in shared_ids:
        return 0
    doc_dev = doc_device_map.get(doc_id, "").strip().upper()
    tgt = target_device.strip().upper()
    return 0 if doc_dev == tgt else 1


def compute_contamination(
    doc_ids: list[str],
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
    k: int,
) -> float:
    top_k = doc_ids[:k]
    if not top_k:
        return 0.0
    contaminated = sum(
        1 for d in top_k if v_scope(d, target_device, doc_device_map, shared_ids) == 1
    )
    return contaminated / k


def compute_contamination_hits(
    hits: list[dict[str, Any]],
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
    k: int,
) -> float:
    top_hits = hits[:k]
    if not top_hits:
        return 0.0
    tgt = target_device.strip().upper()
    contaminated = 0
    for hit in top_hits:
        doc_id = str(hit.get("doc_id") or "")
        if not doc_id:
            continue
        if doc_id in shared_ids:
            continue
        device_name = str(hit.get("device_name") or "").strip().upper()
        if not device_name:
            device_name = doc_device_map.get(doc_id, "").strip().upper()
        if device_name != tgt:
            contaminated += 1
    return contaminated / k


def compute_gold_hit(doc_ids: list[str], gold_ids: list[str], k: int) -> bool:
    return bool(set(gold_ids) & set(doc_ids[:k]))


def compute_mrr(doc_ids: list[str], gold_ids: list[str], k: int) -> float:
    gold_set = set(gold_ids)
    for rank, doc_id in enumerate(doc_ids[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Build allowed_devices map (upper -> [raw ES device names])
# ---------------------------------------------------------------------------
def build_allowed_devices_map(doc_device_map: dict[str, str]) -> dict[str, list[str]]:
    """Build {DEVICE_UPPER -> [raw_device_name_1, raw_device_name_2, ...]}."""
    upper_to_raw: dict[str, set[str]] = defaultdict(set)
    for _doc_id, device in doc_device_map.items():
        if device:
            raw = device.strip()
            upper_to_raw[raw.upper()].add(raw)
    return {k: sorted(v) for k, v in upper_to_raw.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
P8_CONDITIONS = [
    # (name, M, shared_cap)
    ("P8_masked", 3, 0),
    ("P8_sc1_masked", 3, 1),
    ("P8_sc2_masked", 3, 2),
    ("P8_m5_masked", 5, 0),
]

BASELINE_CONDITIONS = [
    "B3_masked",
    "B4_masked",
    "B4.5_masked",
    "P7plus_masked",
]


def run() -> None:
    print("=" * 72)
    print("P8: Evidence-Based Scope Selection Experiment")
    print("=" * 72)

    # Load policy data
    print("\nLoading policy data...")
    doc_device_map = load_doc_scope(DOC_SCOPE_PATH)
    shared_ids = load_shared_ids(SHARED_IDS_PATH)
    device_doc_ids = build_device_to_doc_ids(doc_device_map)
    allowed_devices_map = build_allowed_devices_map(doc_device_map)
    print(f"  doc_scope: {len(doc_device_map)} entries")
    print(f"  shared: {len(shared_ids)} docs")
    print(f"  devices: {len(device_doc_ids)} unique")

    # Load cached results for baselines + hypothesis generation
    print(f"\nLoading cached results from {MASKED_HYBRID_PATH} ...")
    with MASKED_HYBRID_PATH.open(encoding="utf-8") as f:
        masked_hybrid: list[dict[str, Any]] = json.load(f)
    print(f"  Loaded {len(masked_hybrid)} query entries.")

    # Load P7+ results if available
    p7plus_map: dict[str, dict[str, Any]] = {}
    if P6P7_RESULTS_PATH.exists():
        print(f"Loading P7+ results from {P6P7_RESULTS_PATH} ...")
        with P6P7_RESULTS_PATH.open(encoding="utf-8") as f:
            p7plus_data: list[dict[str, Any]] = json.load(f)
        for row in p7plus_data:
            q_id = str(row.get("q_id") or "")
            if q_id:
                p7plus_map[q_id] = row

    # Load eval set for question_masked
    print(f"\nLoading eval set from {EVAL_PATH} ...")
    eval_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(EVAL_PATH):
        assert isinstance(row, dict)
        q_id = str(row.get("q_id") or "")
        if q_id:
            eval_rows[q_id] = row
    print(f"  Loaded {len(eval_rows)} eval queries.")

    # Connect to ES
    es = _build_es()
    print(f"  ES host: {search_settings.es_host}")

    # Pre-warm models
    print("\nPre-warming models (embedding + reranker)...")
    _ = embed_query("warm up query")
    get_reranker()
    print("  Models ready.")

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

        # Get masked question from eval set
        eval_row = eval_rows.get(q_id, {})
        question_masked = str(eval_row.get("question_masked") or "")
        if not question_masked:
            logger.warning("No masked question for q_id=%s, skipping", q_id)
            continue

        if qi % 50 == 0:
            elapsed = time.time() - start_time
            qps = (qi + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{qi}/{total}] q_id={q_id} device={target_device} ({qps:.1f} q/s)"
            )

        probe_hits: list[dict[str, Any]] = []
        try:
            probe_hits = retrieve_hybrid_rerank(
                es=es,
                query=question_masked,
                top_k=PROBE_TOP_K,
                bm25_filter=None,
                dense_device_filter=None,
            )
        except Exception as exc:
            logger.warning("Probe retrieval failed for q_id=%s: %s", q_id, exc)

        # Grab B3 cached results for hypothesis generation
        conditions_cached = q.get("conditions") or {}
        b3_entry = conditions_cached.get("B3_masked") or {}
        b3_doc_ids: list[str] = [str(d) for d in (b3_entry.get("top_doc_ids") or [])]

        # Build baseline results
        q_result: dict[str, Any] = {
            "q_id": q_id,
            "target_device": target_device,
            "scope_observability": scope_obs,
            "gold_ids_loose": gold_ids_loose,
            "gold_ids_strict": gold_ids_strict,
            "conditions": {},
        }

        # Copy baselines from cached data
        for cond_name in ["B3_masked", "B4_masked", "B4.5_masked"]:
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

        # P8-02 fix: Add B3_live from probe results for fair same-run comparison
        if probe_hits:
            probe_doc_ids = [str(h["doc_id"]) for h in probe_hits[:TOP_K]]
            q_result["conditions"]["B3_live"] = {
                "cont@10": compute_contamination_hits(
                    probe_hits[:TOP_K], target_device, doc_device_map, shared_ids, TOP_K
                ),
                "gold_hit_strict": compute_gold_hit(
                    probe_doc_ids, gold_ids_strict, TOP_K
                ),
                "gold_hit_loose": compute_gold_hit(
                    probe_doc_ids, gold_ids_loose, TOP_K
                ),
                "mrr": compute_mrr(probe_doc_ids, gold_ids_strict, TOP_K),
                "top_doc_ids": probe_doc_ids,
            }

        # Copy P7+ from cached data
        p7plus_row = p7plus_map.get(q_id, {})
        p7plus_cond = (p7plus_row.get("conditions") or {}).get("P7plus_masked", {})
        if p7plus_cond and "error" not in p7plus_cond:
            p7plus_doc_ids = [str(d) for d in (p7plus_cond.get("top_doc_ids") or [])]
            q_result["conditions"]["P7plus_masked"] = {
                "cont@10": float(p7plus_cond.get("cont@10") or 0.0),
                "gold_hit_strict": bool(p7plus_cond.get("gold_hit_strict") or False),
                "gold_hit_loose": bool(p7plus_cond.get("gold_hit_loose") or False),
                "mrr": compute_mrr(p7plus_doc_ids, gold_ids_strict, TOP_K),
                "top_doc_ids": p7plus_doc_ids[:TOP_K],
            }

        # Run P8 conditions
        for cond_name, m_val, sc_val in P8_CONDITIONS:
            p8_result = run_p8_for_query(
                es=es,
                query=question_masked,
                b3_doc_ids=b3_doc_ids,
                target_device=target_device,
                doc_device_map=doc_device_map,
                device_doc_ids=device_doc_ids,
                shared_ids=shared_ids,
                allowed_devices_map=allowed_devices_map,
                m=m_val,
                shared_cap=sc_val,
                probe_hits=probe_hits,
            )

            p8_doc_ids = p8_result["top_doc_ids"]
            p8_hits = cast(list[dict[str, Any]], p8_result.get("hits") or [])
            q_result["conditions"][cond_name] = {
                "cont@10": compute_contamination_hits(
                    p8_hits, target_device, doc_device_map, shared_ids, TOP_K
                ),
                "gold_hit_strict": compute_gold_hit(p8_doc_ids, gold_ids_strict, TOP_K),
                "gold_hit_loose": compute_gold_hit(p8_doc_ids, gold_ids_loose, TOP_K),
                "mrr": compute_mrr(p8_doc_ids, gold_ids_strict, TOP_K),
                "top_doc_ids": p8_doc_ids[:TOP_K],
                "selected_device": p8_result["selected_device"],
                "scope_correct": p8_result["scope_correct"],
                "hypotheses": p8_result["hypotheses"],
                "evidence_scores": p8_result["evidence_scores"],
                "target_hypothesis_rank": p8_result["target_hypothesis_rank"],
                "fallback": p8_result.get("fallback", False),
                "hypothesis_source": p8_result.get("hypothesis_source", "cached"),
            }

        per_query.append(q_result)

    elapsed_total = time.time() - start_time
    print(f"\nCompleted {len(per_query)} queries in {elapsed_total:.1f}s")
    print(f"  ({len(per_query) / elapsed_total:.1f} queries/sec)")

    # Save results
    print(f"\nSaving results to {OUT_PATH} ...")
    serializable: list[JsonValue] = [cast(JsonValue, row) for row in per_query]
    write_json(OUT_PATH, serializable)
    print("Saved.")

    # Print summary
    _print_summary(per_query)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ALL_CONDITIONS_ORDER = [
    "B3_masked",
    "B3_live",
    "B4_masked",
    "B4.5_masked",
    "P7plus_masked",
    "P8_masked",
    "P8_sc1_masked",
    "P8_sc2_masked",
    "P8_m5_masked",
]


def _print_summary(per_query: list[dict[str, Any]]) -> None:
    n_total = len(per_query)

    # --- Overall summary ---
    _print_summary_for_scope(per_query, "ALL", n_total)

    # --- By scope_observability ---
    scope_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in per_query:
        scope_groups[q.get("scope_observability", "unknown")].append(q)

    for scope_name in ["explicit_device", "explicit_equip"]:
        group = scope_groups.get(scope_name, [])
        if group:
            _print_summary_for_scope(group, scope_name, len(group))

    # --- P8 scope accuracy ---
    print(f"\n{'=' * 72}")
    print("P8 Scope Selection Accuracy")
    print("=" * 72)
    for cond_name, _, _ in P8_CONDITIONS:
        scope_correct_all = []
        target_in_hyp = []
        for q in per_query:
            cdata = q.get("conditions", {}).get(cond_name, {})
            if cdata and "scope_correct" in cdata:
                scope_correct_all.append(1.0 if cdata["scope_correct"] else 0.0)
                target_in_hyp.append(
                    1.0 if cdata.get("target_hypothesis_rank") is not None else 0.0
                )
        if scope_correct_all:
            acc = sum(scope_correct_all) / len(scope_correct_all)
            in_hyp = sum(target_in_hyp) / len(target_in_hyp)
            print(
                f"  {cond_name:<20}  scope_acc={acc:.3f}  "
                f"target_in_hypotheses={in_hyp:.3f}  (n={len(scope_correct_all)})"
            )

    # --- By scope, scope accuracy ---
    for scope_name in ["explicit_device", "explicit_equip"]:
        group = scope_groups.get(scope_name, [])
        if not group:
            continue
        print(f"\n  --- {scope_name} (n={len(group)}) ---")
        for cond_name, _, _ in P8_CONDITIONS:
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
    run()
