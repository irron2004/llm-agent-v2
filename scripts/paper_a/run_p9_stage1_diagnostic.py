"""P9 Phase 1: Stage 1 hypothesis recall + Stage 3 oracle diagnostic.

Phase 1-A: Compare different device proposal methods
  - B3 cached top-10 device distribution
  - B3_live (probe top-10) device distribution
  - Probe top-40 device distribution (live ES query)
  - P7+ top-10 device distribution
  - Probe top-40 with rank-decay weighting

Phase 1-B: Stage 3 oracle analysis
  - Given target device IS in hypothesis set, how often does score_sum pick it?
  - Compare score_sum vs max_score vs mean_top3

Output: data/paper_a/p9_stage1_diagnostic.json + console summary
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

LOG_FILE = ROOT / "data/paper_a/p9_stage1_diagnostic.log"
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
PROBE_TOP_K = 40

EVAL_PATH = ROOT / "data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl"
DOC_SCOPE_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/doc_scope.jsonl"
SHARED_IDS_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/shared_doc_ids.txt"
MASKED_HYBRID_PATH = ROOT / "data/paper_a/masked_hybrid_results.json"
P6P7_RESULTS_PATH = ROOT / "data/paper_a/masked_p6p7_results.json"
P8_RESULTS_PATH = ROOT / "data/paper_a/p8_results.json"
OUT_PATH = ROOT / "data/paper_a/p9_stage1_diagnostic.json"


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


# ---------------------------------------------------------------------------
# Embedding + Reranker (for probe retrieval)
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
# Retrieval (same as P8 script)
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
        "_source": False,
        "fields": ["doc_id"],
    }
    resp = es.search(index=EMBED_INDEX, body=body)
    hits = []
    for h in resp["hits"]["hits"]:
        doc_id = ""
        if "fields" in h and "doc_id" in h["fields"]:
            doc_id = h["fields"]["doc_id"][0] if isinstance(h["fields"]["doc_id"], list) else h["fields"]["doc_id"]
        hits.append({"doc_id": doc_id, "score": float(h["_score"])})
    return hits


def rrf_fuse(
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for rank, h in enumerate(bm25_hits):
        scores[h["doc_id"]] += 1.0 / (k + rank + 1)
    for rank, h in enumerate(dense_hits):
        scores[h["doc_id"]] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]


def fetch_contents(
    es: Elasticsearch, doc_ids: list[str]
) -> dict[str, str]:
    if not doc_ids:
        return {}
    body: dict[str, Any] = {
        "query": {"terms": {"doc_id": doc_ids}},
        "size": len(doc_ids),
        "_source": ["doc_id", "content"],
    }
    resp = es.search(index=CONTENT_INDEX, body=body)
    result: dict[str, str] = {}
    for h in resp["hits"]["hits"]:
        src = h["_source"]
        result[src.get("doc_id", h["_id"])] = src.get("content", "")
    return result


def retrieve_hybrid_rerank(
    *,
    es: Elasticsearch,
    query: str,
    top_k: int,
    bm25_filter: dict[str, Any] | None = None,
    dense_device_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    bm25_hits = bm25_search(es, query, FETCH_N, extra_filter=bm25_filter)
    qvec = embed_query(query)
    dense_hits = dense_search(es, qvec, FETCH_N, device_filter=dense_device_filter)
    fused_ids = rrf_fuse(bm25_hits, dense_hits)[:FETCH_N]
    contents = fetch_contents(es, fused_ids)
    reranker = get_reranker()
    pairs = [(query, contents.get(did, "")) for did in fused_ids if did in contents]
    valid_ids = [did for did in fused_ids if did in contents]
    if not pairs:
        return []
    from typing import cast as _cast
    reranker_impl = _cast(Any, reranker)
    scores = reranker_impl.model.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return [{"doc_id": did, "score": float(s)} for did, s in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Hypothesis generation methods
# ---------------------------------------------------------------------------
def devices_from_doc_ids(
    doc_ids: list[str],
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> list[tuple[str, int]]:
    """Extract (normalized_device, count) from doc_ids, excluding shared."""
    counter: Counter[str] = Counter()
    for did in doc_ids:
        if did in shared_ids:
            continue
        device = doc_device_map.get(did, "")
        norm = normalize_device_name(device)
        if norm:
            counter[norm] += 1
    return counter.most_common()


def devices_from_hits_ranked(
    hits: list[dict[str, Any]],
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> list[tuple[str, float]]:
    """Extract (normalized_device, weighted_score) with rank-decay weighting."""
    scores: dict[str, float] = defaultdict(float)
    for rank, h in enumerate(hits):
        did = h["doc_id"]
        if did in shared_ids:
            continue
        device = doc_device_map.get(did, "")
        norm = normalize_device_name(device)
        if norm:
            w = 1.0 / np.log2(rank + 2)
            scores[norm] += w
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def target_in_topM(
    device_ranking: list[tuple[str, Any]],
    target_device: str,
    m: int,
) -> bool:
    """Check if target_device (normalized) is in top-M of device_ranking."""
    tgt = normalize_device_name(target_device)
    top_devices = [d for d, _ in device_ranking[:m]]
    return tgt in top_devices


def target_rank(
    device_ranking: list[tuple[str, Any]],
    target_device: str,
) -> int | None:
    """Return 1-based rank of target_device, or None if not found."""
    tgt = normalize_device_name(target_device)
    for i, (d, _) in enumerate(device_ranking):
        if d == tgt:
            return i + 1
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    logger.info("P9 Stage 1 Diagnostic starting (pid=%d)", os.getpid())

    # Load data
    logger.info("Loading policy data...")
    doc_device_map = load_doc_scope(DOC_SCOPE_PATH)
    shared_ids = load_shared_ids(SHARED_IDS_PATH)
    logger.info("  doc_scope=%d, shared=%d", len(doc_device_map), len(shared_ids))

    logger.info("Loading cached results...")
    with MASKED_HYBRID_PATH.open(encoding="utf-8") as f:
        masked_hybrid: list[dict[str, Any]] = json.load(f)

    p7plus_map: dict[str, dict[str, Any]] = {}
    if P6P7_RESULTS_PATH.exists():
        with P6P7_RESULTS_PATH.open(encoding="utf-8") as f:
            p7data: list[dict[str, Any]] = json.load(f)
        for row in p7data:
            q_id = str(row.get("q_id") or "")
            if q_id:
                p7plus_map[q_id] = row

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
    logger.info("Pre-warming models...")
    _ = embed_query("warm up query")
    get_reranker()
    logger.info("Models ready.")

    # Run diagnostic
    results: list[dict[str, Any]] = []
    total = len(masked_hybrid)
    start_time = time.time()

    for qi, q in enumerate(masked_hybrid):
        q_id = str(q.get("q_id") or qi)
        target_device = str(q.get("target_device") or "")
        scope_obs = str(q.get("scope_observability") or "")

        if not target_device:
            continue

        eval_row = eval_rows.get(q_id, {})
        question_masked = str(eval_row.get("question_masked") or "")
        if not question_masked:
            continue

        elapsed = time.time() - start_time
        qps = (qi + 1) / elapsed if elapsed > 0 else 0
        logger.info("[%d/%d] q_id=%s device=%s obs=%s (%.1f q/s)",
                    qi, total, q_id, target_device, scope_obs, qps)

        conditions_cached = q.get("conditions") or {}
        b3_entry = conditions_cached.get("B3_masked") or {}
        b3_doc_ids: list[str] = [str(d) for d in (b3_entry.get("top_doc_ids") or [])]

        # P7+ doc_ids
        p7plus_row = p7plus_map.get(q_id, {})
        p7plus_cond = (p7plus_row.get("conditions") or {}).get("P7plus_masked", {})
        p7plus_doc_ids: list[str] = [str(d) for d in (p7plus_cond.get("top_doc_ids") or [])]

        # Method A: B3 cached top-10 device distribution (count-based)
        method_a = devices_from_doc_ids(b3_doc_ids, doc_device_map, shared_ids)

        # Method B: P7+ top-10 device distribution (count-based)
        method_b = devices_from_doc_ids(p7plus_doc_ids, doc_device_map, shared_ids)

        # Method C: Probe top-40 device distribution (count-based, live ES)
        probe_hits: list[dict[str, Any]] = []
        try:
            probe_hits = retrieve_hybrid_rerank(
                es=es,
                query=question_masked,
                top_k=PROBE_TOP_K,
                bm25_filter=None,
                dense_device_filter=None,
            )
        except Exception:
            logger.exception("Probe failed for q_id=%s", q_id)

        probe_doc_ids = [h["doc_id"] for h in probe_hits]
        method_c = devices_from_doc_ids(probe_doc_ids, doc_device_map, shared_ids)

        # Method D: Probe top-40 with rank-decay weighting
        method_d = devices_from_hits_ranked(probe_hits, doc_device_map, shared_ids)

        # Method E: P7+ with rank-decay weighting (simulated from doc_ids order)
        p7plus_as_hits = [{"doc_id": did} for did in p7plus_doc_ids]
        method_e = devices_from_hits_ranked(p7plus_as_hits, doc_device_map, shared_ids)

        # Collect recall metrics
        q_result: dict[str, Any] = {
            "q_id": q_id,
            "target_device": target_device,
            "target_device_norm": normalize_device_name(target_device),
            "scope_observability": scope_obs,
            "methods": {},
        }

        methods = {
            "B3_cached_count": method_a,
            "P7plus_count": method_b,
            "probe40_count": method_c,
            "probe40_rankdecay": method_d,
            "P7plus_rankdecay": method_e,
        }

        for method_name, device_ranking in methods.items():
            q_result["methods"][method_name] = {
                "top_devices": [(d, float(s)) for d, s in device_ranking[:10]],
                "target_rank": target_rank(device_ranking, target_device),
                "in_top1": target_in_topM(device_ranking, target_device, 1),
                "in_top3": target_in_topM(device_ranking, target_device, 3),
                "in_top5": target_in_topM(device_ranking, target_device, 5),
            }

        # Phase 1-B: Stage 3 oracle analysis
        # If target is in probe top-40 hypotheses, test different score selectors
        tgt_norm = normalize_device_name(target_device)
        probe_devices_in_top5 = [d for d, _ in method_c[:5]]

        if tgt_norm in probe_devices_in_top5 and probe_hits:
            # Build device_doc_ids for this query from probe hits
            dev_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for h in probe_hits:
                did = h["doc_id"]
                if did in shared_ids:
                    continue
                dev = normalize_device_name(doc_device_map.get(did, ""))
                if dev:
                    dev_groups[dev].append(h)

            # Compute different evidence scores for each hypothesis device
            stage3_scores: dict[str, dict[str, float]] = {}
            for dev in probe_devices_in_top5:
                if dev not in dev_groups:
                    continue
                hits = dev_groups[dev]
                scores_list = [h["score"] for h in hits]
                stage3_scores[dev] = {
                    "score_sum": sum(scores_list),
                    "score_max": max(scores_list) if scores_list else -999.0,
                    "mean_top3": (sum(sorted(scores_list, reverse=True)[:3]) / min(3, len(scores_list)))
                                if scores_list else -999.0,
                    "hit_count": float(len(scores_list)),
                }

            # Which method picks the correct device?
            q_result["stage3_oracle"] = {
                "target_in_hypotheses": True,
                "hypothesis_devices": probe_devices_in_top5,
                "device_scores": stage3_scores,
            }
            for metric in ["score_sum", "score_max", "mean_top3", "hit_count"]:
                selected = max(
                    stage3_scores.keys(),
                    key=lambda d: stage3_scores[d].get(metric, -999.0),
                )
                q_result["stage3_oracle"][f"selected_by_{metric}"] = selected
                q_result["stage3_oracle"][f"correct_by_{metric}"] = (selected == tgt_norm)
        else:
            q_result["stage3_oracle"] = {
                "target_in_hypotheses": False,
            }

        results.append(q_result)

    elapsed_total = time.time() - start_time
    logger.info("Completed %d queries in %.1fs", len(results), elapsed_total)

    # Save
    logger.info("Saving to %s ...", OUT_PATH)
    write_json(OUT_PATH, results)  # type: ignore[arg-type]
    logger.info("Saved.")

    # Print summary
    _print_summary(results)


def _print_summary(results: list[dict[str, Any]]) -> None:
    n = len(results)
    print(f"\n{'=' * 80}")
    print(f"P9 Stage 1 Diagnostic — {n} queries")
    print("=" * 80)

    # Phase 1-A: Hypothesis recall
    method_names = [
        "B3_cached_count",
        "P7plus_count",
        "probe40_count",
        "probe40_rankdecay",
        "P7plus_rankdecay",
    ]

    for scope_label, filter_fn in [
        ("ALL", lambda _q: True),
        ("explicit_device", lambda _q: _q["scope_observability"] == "explicit_device"),
        ("explicit_equip", lambda _q: _q["scope_observability"] == "explicit_equip"),
    ]:
        subset = [q for q in results if filter_fn(q)]
        ns = len(subset)
        if ns == 0:
            continue

        print(f"\n--- {scope_label} (n={ns}) ---")
        print(f"  {'method':<22}  {'@1':>6}  {'@3':>6}  {'@5':>6}  {'miss':>6}")
        print("  " + "-" * 60)

        for method in method_names:
            at1 = sum(1 for q in subset if q["methods"][method]["in_top1"])
            at3 = sum(1 for q in subset if q["methods"][method]["in_top3"])
            at5 = sum(1 for q in subset if q["methods"][method]["in_top5"])
            miss = ns - at5
            print(
                f"  {method:<22}  {at1/ns:.3f}  {at3/ns:.3f}  {at5/ns:.3f}  "
                f"{miss:>3}/{ns}"
            )

    # Phase 1-B: Stage 3 selector accuracy (given target in hypotheses)
    print(f"\n{'=' * 80}")
    print("Phase 1-B: Stage 3 selector accuracy (target in probe40 top-5 hypotheses)")
    print("=" * 80)

    for scope_label, filter_fn in [
        ("ALL", lambda _q: True),
        ("explicit_device", lambda _q: _q["scope_observability"] == "explicit_device"),
        ("explicit_equip", lambda _q: _q["scope_observability"] == "explicit_equip"),
    ]:
        subset = [
            q for q in results
            if filter_fn(q) and q["stage3_oracle"].get("target_in_hypotheses", False)
        ]
        ns = len(subset)
        if ns == 0:
            continue

        print(f"\n--- {scope_label} (n={ns}, target in hypotheses) ---")
        metrics = ["score_sum", "score_max", "mean_top3", "hit_count"]
        print(f"  {'metric':<16}  {'correct':>8}  {'accuracy':>8}")
        print("  " + "-" * 40)

        for metric in metrics:
            correct = sum(
                1 for q in subset
                if q["stage3_oracle"].get(f"correct_by_{metric}", False)
            )
            print(f"  {metric:<16}  {correct:>5}/{ns}  {correct/ns:.3f}")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logger.exception("FATAL: run() crashed")
        sys.exit(1)
