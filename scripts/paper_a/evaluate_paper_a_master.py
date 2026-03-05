"""Master-schema evaluator for Paper A.

Consumes query_gold_master_v0_5.jsonl and supports deterministic row
selection, multi-allowed devices/equips, and full system matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import rag_settings, search_settings
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
from backend.llm_infrastructure.retrieval.filters.scope_filter import (
    apply_scope_filter,
    build_scope_filter_by_doc_ids,
    build_scope_filter_by_fields,
)
from backend.llm_infrastructure.reranking.adapters.cross_encoder import (
    CrossEncoderReranker,
)
from backend.services.embedding_service import EmbeddingService
from elasticsearch import Elasticsearch, NotFoundError

from scripts.paper_a._io import read_jsonl, write_json
from scripts.paper_a.canonicalize import compact_key

K_VALUES = (1, 3, 5, 10)
REQUIRED_SYSTEMS = {"B0", "B1", "B2", "B3", "B4", "P1"}
OPTIONAL_SYSTEMS = {"P2", "P3", "P4", "P6", "P7"}
CONTROL_SYSTEMS = {"C1_random_scope", "C2_global_postfilter", "C3_per_scope_merge", "C4_dedupe_only"}
ALL_SYSTEMS = REQUIRED_SYSTEMS | OPTIONAL_SYSTEMS | CONTROL_SYSTEMS
DEFAULT_SYSTEMS = "B0,B1,B2,B3,B4,P1"
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class MasterEvalQuery:
    q_id: str
    split: str
    question: str
    scope_observability: str
    intent_primary: str
    target_scope_level: str
    allowed_devices: list[str]
    allowed_equips: list[str]
    shared_allowed: bool
    family_allowed: bool
    gold_doc_ids: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper A master-schema evaluator (v0.5)"
    )
    _ = parser.add_argument("--eval-set", required=True, help="Path to query_gold_master_v0_5.jsonl")
    _ = parser.add_argument("--systems", default=DEFAULT_SYSTEMS, help="Comma-separated system IDs")
    _ = parser.add_argument("--corpus-filter", required=True, help="Path to corpus_doc_ids.txt")
    _ = parser.add_argument("--doc-scope", required=True, help="Path to doc_scope.jsonl")
    _ = parser.add_argument("--family-map", required=True, help="Path to family_map.json")
    _ = parser.add_argument("--out-dir", required=True, help="Output directory")
    _ = parser.add_argument("--shared-doc-ids", default=None, help="Optional shared_doc_ids.txt")
    _ = parser.add_argument("--index", default=None, help="ES alias/index override")
    _ = parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    _ = parser.add_argument("--seed", type=int, default=20260305)
    _ = parser.add_argument("--bootstrap-samples", type=int, default=2000)
    _ = parser.add_argument("--use-es-scope-fields", action="store_true")
    _ = parser.add_argument("--reranker-model", default=None)
    _ = parser.add_argument(
        "--split", default="all", choices=["dev", "test", "all"],
        help="Filter by split"
    )
    _ = parser.add_argument(
        "--scope-observability", default="all",
        help="Filter by scope_observability (explicit_device,explicit_equip,implicit,ambiguous,all)"
    )
    _ = parser.add_argument("--limit", type=int, default=0, help="Limit rows after filtering (0=no limit)")
    _ = parser.add_argument("--repeats", type=int, default=1, help="Repeat runs for stability analysis")
    return parser.parse_args()


def _load_master_eval_set(path: Path) -> list[MasterEvalQuery]:
    rows: list[MasterEvalQuery] = []
    for raw in read_jsonl(path):
        if not isinstance(raw, dict):
            raise RuntimeError("eval row must be an object")
        q_id = str(raw.get("q_id") or "").strip()
        split = str(raw.get("split") or "").strip()
        question = str(raw.get("question") or "").strip()
        scope_obs = str(raw.get("scope_observability") or "").strip()
        intent = str(raw.get("intent_primary") or "").strip()
        target_scope = str(raw.get("target_scope_level") or "").strip()

        allowed_devices_raw = raw.get("allowed_devices")
        if isinstance(allowed_devices_raw, list):
            allowed_devices = [str(d).strip() for d in allowed_devices_raw if str(d).strip()]
        else:
            allowed_devices = []

        allowed_equips_raw = raw.get("allowed_equips")
        if isinstance(allowed_equips_raw, list):
            allowed_equips = [str(e).strip() for e in allowed_equips_raw if str(e).strip()]
        else:
            allowed_equips = []

        shared_allowed = bool(raw.get("shared_allowed", False))
        family_allowed = bool(raw.get("family_allowed", False))

        gold_raw = raw.get("gold_doc_ids")
        gold_doc_ids = []
        if isinstance(gold_raw, list):
            gold_doc_ids = [str(g).strip() for g in gold_raw if str(g).strip()]

        if not q_id or not split or not question:
            raise RuntimeError(f"eval row missing q_id/split/question: {raw}")

        rows.append(MasterEvalQuery(
            q_id=q_id, split=split, question=question,
            scope_observability=scope_obs, intent_primary=intent,
            target_scope_level=target_scope,
            allowed_devices=allowed_devices, allowed_equips=allowed_equips,
            shared_allowed=shared_allowed, family_allowed=family_allowed,
            gold_doc_ids=gold_doc_ids,
        ))
    if not rows:
        raise RuntimeError(f"No eval rows in {path}")
    return rows


def _filter_queries(
    queries: list[MasterEvalQuery],
    split: str,
    scope_obs: str,
    limit: int,
) -> list[MasterEvalQuery]:
    filtered = queries
    if split != "all":
        filtered = [q for q in filtered if q.split == split]
    if scope_obs != "all":
        obs_values = {s.strip() for s in scope_obs.split(",") if s.strip()}
        filtered = [q for q in filtered if q.scope_observability in obs_values]
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def _load_doc_scope(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(path):
        if not isinstance(raw, dict):
            raise RuntimeError("doc_scope row must be object")
        rows.append(cast(dict[str, Any], raw))
    if not rows:
        raise RuntimeError(f"No rows in doc_scope: {path}")
    return rows


def _load_family_map(path: Path) -> dict[str, Any]:
    loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict):
        raise RuntimeError("family_map.json must be an object")
    return cast(dict[str, Any], loaded)


def _load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _resolve_alias_or_index(manager: EsIndexManager, requested: str) -> tuple[str, str]:
    default_alias = manager.get_alias_name()
    if requested == default_alias:
        target = manager.get_alias_target()
        if not target:
            raise RuntimeError(f"Alias '{requested}' does not point to any index")
        return requested, target
    try:
        alias_response = manager.es.indices.get_alias(name=requested)
        alias_targets = list(cast(dict[str, Any], cast(object, alias_response.body)).keys())
        if not alias_targets:
            raise RuntimeError(f"Alias '{requested}' exists but has no targets")
        return requested, alias_targets[0]
    except NotFoundError:
        if manager.es.indices.exists(index=requested):
            return requested, requested
        raise RuntimeError(f"Alias/index '{requested}' was not found") from None


def _build_es_client() -> Elasticsearch:
    kwargs: dict[str, Any] = {"hosts": [search_settings.es_host], "verify_certs": True}
    if search_settings.es_user and search_settings.es_password:
        kwargs["basic_auth"] = (search_settings.es_user, search_settings.es_password)
    return Elasticsearch(**kwargs)


def _merge_filters(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    if extra is None:
        return base
    combined = apply_scope_filter(base, extra)
    if combined is None:
        raise RuntimeError("unexpected empty combined filter")
    return cast(dict[str, Any], combined)


def _dedupe_top_doc_ids(results: list[RetrievalResult], top_k: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for result in results:
        doc_id = str(result.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        rows.append({
            "doc_id": doc_id,
            "score": float(result.score),
            "metadata": dict(result.metadata or {}),
        })
        if len(rows) >= top_k:
            break
    return rows


class RetrievalRunner:
    def __init__(self, *, index_name: str, corpus_doc_ids: list[str], reranker_model: str | None) -> None:
        self.es_engine = EsSearchEngine(
            es_client=_build_es_client(),
            index_name=index_name,
            text_fields=["search_text^1.0", "chunk_summary^0.7", "chunk_keywords^0.8"],
        )
        corpus_filter = self.es_engine.build_filter(doc_ids=corpus_doc_ids)
        if corpus_filter is None:
            raise RuntimeError("Failed to build corpus whitelist filter")
        self.base_filter = cast(dict[str, Any], corpus_filter)
        self.embed_svc = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
            device=rag_settings.embedding_device,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )
        self._embed_cache: dict[str, list[float]] = {}
        self._reranker_model = reranker_model
        self._reranker: CrossEncoderReranker | None = None

    def _embed_query(self, query: str) -> list[float]:
        cached = self._embed_cache.get(query)
        if cached is not None:
            return cached
        vector = np.asarray(self.embed_svc.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        out = list(vector)
        self._embed_cache[query] = out
        return out

    def _to_result(self, hit: Any) -> RetrievalResult:
        return RetrievalResult(
            doc_id=hit.doc_id, content=hit.content,
            score=hit.score, metadata=hit.metadata, raw_text=hit.raw_text,
        )

    def _maybe_rerank(self, *, query: str, results: list[RetrievalResult], top_k: int, rerank: bool) -> list[RetrievalResult]:
        if not rerank:
            return results
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self._reranker_model, device=rag_settings.embedding_device,
            )
        return self._reranker.rerank(query=query, results=results, top_k=top_k)

    def run_bm25(self, *, query: str, top_k: int, extra_filter: dict[str, Any] | None, rerank: bool) -> list[dict[str, Any]]:
        final = _merge_filters(self.base_filter, extra_filter)
        hits = self.es_engine.sparse_search(query_text=query, top_k=top_k, filters=final)
        return _dedupe_top_doc_ids(self._maybe_rerank(query=query, results=[self._to_result(h) for h in hits], top_k=top_k, rerank=rerank), top_k)

    def run_dense(self, *, query: str, top_k: int, extra_filter: dict[str, Any] | None, rerank: bool) -> list[dict[str, Any]]:
        final = _merge_filters(self.base_filter, extra_filter)
        hits = self.es_engine.dense_search(query_vector=self._embed_query(query), top_k=top_k, filters=final)
        return _dedupe_top_doc_ids(self._maybe_rerank(query=query, results=[self._to_result(h) for h in hits], top_k=top_k, rerank=rerank), top_k)

    def run_hybrid(self, *, query: str, top_k: int, extra_filter: dict[str, Any] | None, rerank: bool) -> list[dict[str, Any]]:
        final = _merge_filters(self.base_filter, extra_filter)
        hits = self.es_engine.hybrid_search(
            query_vector=self._embed_query(query), query_text=query,
            top_k=top_k, dense_weight=0.7, sparse_weight=0.3,
            filters=final, use_rrf=True, rrf_k=60,
        )
        return _dedupe_top_doc_ids(self._maybe_rerank(query=query, results=[self._to_result(h) for h in hits], top_k=top_k, rerank=rerank), top_k)


def _extract_hard_devices(query: str, device_candidates: list[str]) -> list[str]:
    query_compact = compact_key(query)
    if not query_compact:
        return []

    def is_valid(name: str) -> bool:
        cleaned = str(name).strip()
        if not cleaned:
            return False
        token = compact_key(cleaned)
        if token in {"all", "etc"}:
            return False
        raw = "".join(ch for ch in cleaned if ch.isalnum())
        return not (raw.isalpha() and len(raw) <= 4)

    matches: list[str] = []
    for candidate in device_candidates:
        if not is_valid(candidate):
            continue
        cand_compact = compact_key(candidate)
        if cand_compact and cand_compact in query_compact:
            matches.append(candidate)
    seen: set[str] = set()
    deduped: list[str] = []
    for item in matches:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped[:1]


def _extract_hard_equip_ids(query: str, equip_candidates: set[str]) -> list[str]:
    query_upper = query.upper()
    return [eid for eid in sorted(equip_candidates) if eid and eid in query_upper][:3]


def _expand_family_devices(
    hard_devices: list[str],
    device_to_family: dict[str, str],
    families: dict[str, list[str]],
) -> list[str]:
    expanded: set[str] = set()
    for device in hard_devices:
        family_id = device_to_family.get(device)
        if not family_id:
            expanded.add(device)
            continue
        for member in families.get(family_id, []):
            if member:
                expanded.add(member)
    return sorted(expanded)


def _compute_metrics(
    *,
    doc_rows: list[dict[str, Any]],
    gold_doc_ids: list[str],
    allowed_devices: set[str],
    shared_doc_ids: set[str],
) -> dict[str, float]:
    gold_set = set(gold_doc_ids)
    metrics: dict[str, float] = {}

    mrr = 0.0
    for idx, row in enumerate(doc_rows, start=1):
        if str(row.get("doc_id") or "") in gold_set:
            mrr = 1.0 / float(idx)
            break
    metrics["mrr"] = mrr

    for k in K_VALUES:
        top_rows = doc_rows[:k]
        if not top_rows:
            for pfx in ["raw_cont", "adj_cont", "shared", "ce", "hit"]:
                metrics[f"{pfx}@{k}"] = 0.0
            continue

        raw_oos = adj_oos = shared_count = ce = 0
        top_ids = [str(r.get("doc_id") or "") for r in top_rows if str(r.get("doc_id") or "")]

        for row in top_rows:
            doc_id = str(row.get("doc_id") or "")
            meta = cast(dict[str, Any], row.get("metadata") or {})
            doc_device = str(meta.get("device_name") or "").strip()
            in_scope = bool(doc_device and doc_device in allowed_devices)
            is_shared = doc_id in shared_doc_ids
            if is_shared:
                shared_count += 1
            if not in_scope:
                raw_oos += 1
            if not in_scope and not is_shared:
                adj_oos += 1
                ce = 1

        metrics[f"raw_cont@{k}"] = float(raw_oos) / float(k)
        metrics[f"adj_cont@{k}"] = float(adj_oos) / float(k)
        metrics[f"shared@{k}"] = float(shared_count) / float(k)
        metrics[f"ce@{k}"] = float(ce)
        metrics[f"hit@{k}"] = 1.0 if any(d in gold_set for d in top_ids) else 0.0

    return metrics


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_delta(
    *,
    per_query_rows: list[dict[str, Any]],
    system_a: str, system_b: str,
    metric: str, seed: int, samples: int,
) -> dict[str, Any]:
    metric_by_qid: dict[str, dict[str, float]] = defaultdict(dict)
    for row in per_query_rows:
        if str(row.get("status") or "ok") != "ok":
            continue
        qid = str(row.get("q_id") or "")
        sid = str(row.get("system_id") or "")
        val = row.get(metric)
        if qid and sid and isinstance(val, (int, float)):
            metric_by_qid[qid][sid] = float(val)

    paired = [(v[system_a], v[system_b]) for v in metric_by_qid.values() if system_a in v and system_b in v]
    if not paired:
        return {"system_a": system_a, "system_b": system_b, "metric": metric,
                "n_queries": 0, "delta_mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = random.Random(seed)
    n = len(paired)
    deltas = sorted([_mean([b - a for a, b in [paired[rng.randrange(n)] for _ in range(n)]]) for _ in range(samples)])
    point = _mean([b - a for a, b in paired])
    lo = max(0, int(0.025 * len(deltas)) - 1)
    hi = min(len(deltas) - 1, int(0.975 * len(deltas)) - 1)
    return {"system_a": system_a, "system_b": system_b, "metric": metric,
            "n_queries": n, "delta_mean": point, "ci_lower": deltas[lo], "ci_upper": deltas[hi]}


def _mcnemar_test(
    *,
    per_query_rows: list[dict[str, Any]],
    system_a: str, system_b: str,
    metric: str,
) -> dict[str, Any]:
    by_qid: dict[str, dict[str, float]] = defaultdict(dict)
    for row in per_query_rows:
        if str(row.get("status") or "ok") != "ok":
            continue
        qid = str(row.get("q_id") or "")
        sid = str(row.get("system_id") or "")
        val = row.get(metric)
        if qid and sid and isinstance(val, (int, float)):
            by_qid[qid][sid] = float(val)

    # Count discordant pairs
    b_count = c_count = 0  # b: A=1,B=0; c: A=0,B=1
    n_paired = 0
    for vals in by_qid.values():
        if system_a in vals and system_b in vals:
            n_paired += 1
            a_val = int(vals[system_a] > 0)
            b_val = int(vals[system_b] > 0)
            if a_val == 1 and b_val == 0:
                b_count += 1
            elif a_val == 0 and b_val == 1:
                c_count += 1

    # McNemar chi-squared (with continuity correction)
    denom = b_count + c_count
    if denom == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = float((abs(b_count - c_count) - 1) ** 2) / float(denom)
        # Approximate p-value from chi2(1) using survival function
        # For simplicity, use a lookup for common thresholds
        if chi2 >= 10.828:
            p_value = 0.001
        elif chi2 >= 6.635:
            p_value = 0.01
        elif chi2 >= 3.841:
            p_value = 0.05
        elif chi2 >= 2.706:
            p_value = 0.10
        else:
            p_value = 0.5  # rough approximation

    return {
        "system_a": system_a, "system_b": system_b, "metric": metric,
        "n_paired": n_paired, "b_count": b_count, "c_count": c_count,
        "chi2": chi2, "p_value_approx": p_value,
    }


def _holm_bonferroni(p_values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni correction to a list of test results."""
    indexed = [(i, d.get("p_value_approx", d.get("p_value", 1.0))) for i, d in enumerate(p_values)]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    corrected = list(p_values)
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (m - rank)
        corrected[orig_idx] = dict(p_values[orig_idx])
        corrected[orig_idx]["p_corrected"] = min(adjusted, 1.0)
        corrected[orig_idx]["significant_corrected"] = adjusted < 0.05
    return corrected


def _summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(k) or "") for k in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key_tuple, items in sorted(grouped.items()):
        agg: dict[str, Any] = {k: v for k, v in zip(group_keys, key_tuple, strict=True)}
        ok_items = [i for i in items if str(i.get("status") or "ok") == "ok"]
        agg["n_queries"] = len(items)
        agg["n_ok"] = len(ok_items)
        agg["n_skipped"] = len(items) - len(ok_items)
        for metric in ["raw_cont@5", "adj_cont@5", "shared@5", "ce@5", "hit@5", "mrr"]:
            vals = [float(i[metric]) for i in ok_items if isinstance(i.get(metric), (int, float))]
            agg[f"mean_{metric}"] = _mean(vals)
        out.append(agg)
    return out


METRIC_KEYS: list[str] = []
for _k in K_VALUES:
    for _pfx in ["raw_cont", "adj_cont", "shared", "ce", "hit"]:
        METRIC_KEYS.append(f"{_pfx}@{_k}")
METRIC_KEYS.append("mrr")

SKIP_METRICS = {k: "" for k in METRIC_KEYS}

PER_QUERY_FIELDS = [
    "q_id", "split", "scope_observability", "system_id", "status", "skip_reason",
    "intent_primary", "target_scope_level",
    "allowed_devices", "allowed_equips", "shared_allowed", "family_allowed",
    "parsed_hard_devices", "parsed_hard_equip_ids", "family_devices",
    "allowed_scope_size", "latency_ms",
] + METRIC_KEYS + ["top_doc_ids", "in_corpus_top_docs"]


def run() -> int:
    args = _parse_args()
    eval_set_path = Path(cast(str, args.eval_set))
    corpus_filter_path = Path(cast(str, args.corpus_filter))
    doc_scope_path = Path(cast(str, args.doc_scope))
    family_map_path = Path(cast(str, args.family_map))
    out_dir = Path(cast(str, args.out_dir))

    for p, name in [(eval_set_path, "eval-set"), (corpus_filter_path, "corpus-filter"),
                     (doc_scope_path, "doc-scope"), (family_map_path, "family-map")]:
        if not p.exists() or not p.is_file():
            print(f"{name} file not found: {p}", file=sys.stderr)
            return 1

    systems = [s.strip() for s in str(args.systems).split(",") if s.strip()]
    if not systems:
        print("No systems selected", file=sys.stderr)
        return 1

    try:
        all_queries = _load_master_eval_set(eval_set_path)
        queries = _filter_queries(all_queries, str(args.split), str(args.scope_observability), int(args.limit))
        if not queries:
            print("No queries after filtering", file=sys.stderr)
            return 1
        print(f"Selected {len(queries)} queries (from {len(all_queries)} total)")

        doc_scope_rows = _load_doc_scope(doc_scope_path)
        family_map = _load_family_map(family_map_path)
        device_to_family = {
            str(k): str(v) for k, v in cast(dict[str, Any], family_map.get("device_to_family") or {}).items()
            if str(k).strip() and str(v).strip()
        }
        families = {
            str(fid): [str(m) for m in members if str(m).strip()]
            for fid, members in cast(dict[str, Any], family_map.get("families") or {}).items()
            if isinstance(members, list)
        }

        corpus_doc_ids = _load_lines(corpus_filter_path)
        if not corpus_doc_ids:
            raise RuntimeError("corpus filter is empty")
        corpus_doc_set = set(corpus_doc_ids)

        shared_doc_ids: set[str] = {
            str(row.get("es_doc_id") or "").strip()
            for row in doc_scope_rows
            if bool(row.get("is_shared")) and str(row.get("es_doc_id") or "").strip()
        }
        shared_path = Path(cast(str, args.shared_doc_ids)) if args.shared_doc_ids else doc_scope_path.parent / "shared_doc_ids.txt"
        if shared_path.exists() and shared_path.is_file():
            shared_doc_ids.update(_load_lines(shared_path))

        device_candidates = sorted({
            str(row.get("es_device_name") or "").strip()
            for row in doc_scope_rows if str(row.get("es_device_name") or "").strip()
        })
        equip_candidates = {
            str(row.get("es_equip_id") or "").strip().upper()
            for row in doc_scope_rows if str(row.get("es_equip_id") or "").strip()
        }

        device_doc_types = sorted({
            str(row.get("es_doc_type") or "").strip()
            for row in doc_scope_rows
            if str(row.get("scope_level") or "") == "device" and str(row.get("es_doc_type") or "").strip()
        }) or ["setup", "sop", "ts"]

        equip_doc_types = sorted({
            str(row.get("es_doc_type") or "").strip()
            for row in doc_scope_rows
            if str(row.get("scope_level") or "") == "equip" and str(row.get("es_doc_type") or "").strip()
        }) or ["gcb", "myservice"]

        # Build device-by-doc-id lookup
        doc_scope_device_by_id = {
            str(row.get("es_doc_id") or "").strip(): str(row.get("es_device_name") or "").strip()
            for row in doc_scope_rows if str(row.get("es_doc_id") or "").strip()
        }

        # ES setup
        manager = EsIndexManager(
            es_host=search_settings.es_host, env=search_settings.es_env,
            index_prefix=search_settings.es_index_prefix,
            es_user=search_settings.es_user or None,
            es_password=search_settings.es_password or None, verify_certs=True,
        )
        alias_or_index = cast(str | None, args.index) or manager.get_alias_name()
        alias_name, resolved_index = _resolve_alias_or_index(manager, alias_or_index)
        print(f"Resolved index: {resolved_index}")

        retrieval = RetrievalRunner(
            index_name=resolved_index,
            corpus_doc_ids=corpus_doc_ids,
            reranker_model=cast(str | None, args.reranker_model),
        )

        per_query_rows: list[dict[str, Any]] = []

        for query in queries:
            # Parse hard devices/equips from query text
            hard_devices = _extract_hard_devices(query.question, device_candidates)
            hard_equip_ids = _extract_hard_equip_ids(query.question, equip_candidates)

            # Determine allowed scope for metric computation (from gold labels)
            allowed_scope_devices: set[str] = set(query.allowed_devices)
            if query.family_allowed and hard_devices:
                family_expanded = _expand_family_devices(hard_devices, device_to_family, families)
                allowed_scope_devices.update(family_expanded)

            for system_id in systems:
                row_base: dict[str, Any] = {
                    "q_id": query.q_id,
                    "split": query.split,
                    "scope_observability": query.scope_observability,
                    "system_id": system_id,
                    "status": "ok",
                    "skip_reason": "",
                    "intent_primary": query.intent_primary,
                    "target_scope_level": query.target_scope_level,
                    "allowed_devices": "|".join(query.allowed_devices),
                    "allowed_equips": "|".join(query.allowed_equips),
                    "shared_allowed": query.shared_allowed,
                    "family_allowed": query.family_allowed,
                    "parsed_hard_devices": "|".join(hard_devices),
                    "parsed_hard_equip_ids": "|".join(hard_equip_ids),
                    "family_devices": "|".join(
                        _expand_family_devices(hard_devices, device_to_family, families)
                    ),
                    "allowed_scope_size": len(allowed_scope_devices),
                    "latency_ms": 0.0,
                }

                # Skip if no gold docs and we need them for metrics
                if not query.gold_doc_ids:
                    skip_row = dict(row_base)
                    skip_row["status"] = "skipped"
                    skip_row["skip_reason"] = "no_gold_doc_ids"
                    skip_row.update(SKIP_METRICS)
                    per_query_rows.append(skip_row)
                    continue

                # Skip if no allowed devices for contamination evaluation
                if not allowed_scope_devices:
                    skip_row = dict(row_base)
                    skip_row["status"] = "skipped"
                    skip_row["skip_reason"] = "no_allowed_devices"
                    skip_row.update(SKIP_METRICS)
                    per_query_rows.append(skip_row)
                    continue

                # Router-dependent systems
                if system_id in {"P2", "P3", "P4", "P6", "P7"}:
                    skip_row = dict(row_base)
                    skip_row["status"] = "skipped"
                    skip_row["skip_reason"] = "router_not_implemented"
                    skip_row.update(SKIP_METRICS)
                    per_query_rows.append(skip_row)
                    continue

                # Control baselines
                if system_id in CONTROL_SYSTEMS:
                    skip_row = dict(row_base)
                    skip_row["status"] = "skipped"
                    skip_row["skip_reason"] = "control_not_implemented"
                    skip_row.update(SKIP_METRICS)
                    per_query_rows.append(skip_row)
                    continue

                extra_filter: dict[str, Any] | None = None
                t0 = time.monotonic()

                if system_id == "B0":
                    doc_rows = retrieval.run_bm25(query=query.question, top_k=int(args.top_k), extra_filter=None, rerank=False)
                elif system_id == "B1":
                    doc_rows = retrieval.run_dense(query=query.question, top_k=int(args.top_k), extra_filter=None, rerank=False)
                elif system_id == "B2":
                    doc_rows = retrieval.run_hybrid(query=query.question, top_k=int(args.top_k), extra_filter=None, rerank=False)
                elif system_id == "B3":
                    doc_rows = retrieval.run_hybrid(query=query.question, top_k=int(args.top_k), extra_filter=None, rerank=True)
                elif system_id == "B4":
                    if hard_devices:
                        extra_filter = retrieval.es_engine.build_filter(device_names=hard_devices)
                    doc_rows = retrieval.run_hybrid(query=query.question, top_k=int(args.top_k), extra_filter=cast(dict[str, Any] | None, extra_filter), rerank=True)
                elif system_id == "P1":
                    if bool(args.use_es_scope_fields):
                        extra_filter = build_scope_filter_by_fields(
                            allowed_devices=hard_devices, allowed_equip_ids=hard_equip_ids,
                        )
                    else:
                        extra_filter = build_scope_filter_by_doc_ids(
                            allowed_devices=hard_devices, allowed_equip_ids=hard_equip_ids,
                            shared_doc_ids=sorted(shared_doc_ids),
                            device_doc_types=device_doc_types, equip_doc_types=equip_doc_types,
                        )
                    doc_rows = retrieval.run_hybrid(query=query.question, top_k=int(args.top_k), extra_filter=cast(dict[str, Any] | None, extra_filter), rerank=True)
                else:
                    skip_row = dict(row_base)
                    skip_row["status"] = "skipped"
                    skip_row["skip_reason"] = "unknown_system"
                    skip_row.update(SKIP_METRICS)
                    per_query_rows.append(skip_row)
                    continue

                latency_ms = (time.monotonic() - t0) * 1000.0

                # Compute metrics using gold labels
                metrics = _compute_metrics(
                    doc_rows=doc_rows,
                    gold_doc_ids=query.gold_doc_ids,
                    allowed_devices=allowed_scope_devices,
                    shared_doc_ids=shared_doc_ids,
                )

                out_row = dict(row_base)
                out_row.update(metrics)
                out_row["latency_ms"] = round(latency_ms, 2)
                out_row["top_doc_ids"] = "|".join(str(r.get("doc_id") or "") for r in doc_rows)
                out_row["in_corpus_top_docs"] = all(
                    str(r.get("doc_id") or "") in corpus_doc_set for r in doc_rows
                )
                per_query_rows.append(out_row)

        # Write outputs
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "per_query.csv", per_query_rows, PER_QUERY_FIELDS)

        summary_all = _summarize(per_query_rows, ["system_id"])
        _write_csv(out_dir / "summary_all.csv", summary_all,
                    ["system_id", "n_queries", "n_ok", "n_skipped",
                     "mean_raw_cont@5", "mean_adj_cont@5", "mean_shared@5",
                     "mean_ce@5", "mean_hit@5", "mean_mrr"])

        summary_by_obs = _summarize(per_query_rows, ["scope_observability", "system_id"])
        _write_csv(out_dir / "summary_by_observability.csv", summary_by_obs,
                    ["scope_observability", "system_id", "n_queries", "n_ok", "n_skipped",
                     "mean_raw_cont@5", "mean_adj_cont@5", "mean_shared@5",
                     "mean_ce@5", "mean_hit@5", "mean_mrr"])

        # Bootstrap CIs for preregistered comparisons
        preregistered_pairs = [("B3", "B4"), ("B4", "P1")]
        active_pairs = [(a, b) for a, b in preregistered_pairs if a in systems and b in systems]
        bootstrap_comparisons: list[dict[str, Any]] = []
        for sys_a, sys_b in active_pairs:
            for metric in ["adj_cont@5", "hit@5", "mrr"]:
                bootstrap_comparisons.append(_bootstrap_delta(
                    per_query_rows=per_query_rows,
                    system_a=sys_a, system_b=sys_b, metric=metric,
                    seed=int(args.seed), samples=int(args.bootstrap_samples),
                ))

        bootstrap_payload: dict[str, Any] = {
            "seed": int(args.seed), "samples": int(args.bootstrap_samples),
            "comparisons": bootstrap_comparisons,
        }
        write_json(out_dir / "bootstrap_ci.json", bootstrap_payload)

        # McNemar tests for CE@5
        mcnemar_results: list[dict[str, Any]] = []
        for sys_a, sys_b in active_pairs:
            mcnemar_results.append(_mcnemar_test(
                per_query_rows=per_query_rows,
                system_a=sys_a, system_b=sys_b, metric="ce@5",
            ))
        corrected = _holm_bonferroni(mcnemar_results) if mcnemar_results else []
        write_json(out_dir / "mcnemar.json", {"tests": corrected})

        # Manifest with hashes
        git_sha = ""
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
        except Exception:
            pass

        manifest: dict[str, Any] = {
            "git_sha": git_sha,
            "alias": alias_name,
            "resolved_index": resolved_index,
            "corpus_doc_id_count": len(corpus_doc_ids),
            "systems": systems,
            "seed": int(args.seed),
            "split": str(args.split),
            "scope_observability": str(args.scope_observability),
            "limit": int(args.limit),
            "n_queries_selected": len(queries),
            "n_queries_total": len(all_queries),
            "hashes": {
                "eval_set": _sha256_file(eval_set_path),
                "corpus_filter": _sha256_file(corpus_filter_path),
                "doc_scope": _sha256_file(doc_scope_path),
                "family_map": _sha256_file(family_map_path),
            },
            "config": {
                "eval_set": str(args.eval_set),
                "corpus_filter": str(args.corpus_filter),
                "doc_scope": str(args.doc_scope),
                "family_map": str(args.family_map),
                "top_k": int(args.top_k),
                "bootstrap_samples": int(args.bootstrap_samples),
                "use_es_scope_fields": bool(args.use_es_scope_fields),
                "reranker_model": cast(str | None, args.reranker_model),
                "device_doc_types": device_doc_types,
                "equip_doc_types": equip_doc_types,
            },
        }
        write_json(out_dir / "run_manifest.json", manifest)

        print(f"Wrote outputs to {out_dir}")
        print(f"Queries: {len(queries)}; systems: {','.join(systems)}")
        return 0
    except Exception as exc:
        print(f"evaluate_paper_a_master failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
