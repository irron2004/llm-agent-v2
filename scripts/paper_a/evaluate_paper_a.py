from __future__ import annotations

# pyright: basic

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
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
DEFAULT_SYSTEMS = "B0,B1,B2,B3,B4,P1,P2,P3,P4"
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class EvalQuery:
    qid: str
    split: str
    query: str
    target_device: str
    gold_doc_ids: list[str]


@dataclass(frozen=True)
class ScopeContext:
    hard_devices: list[str]
    hard_equip_ids: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Paper A retrieval systems")
    _ = parser.add_argument("--eval-set", required=True, help="Path to eval set JSONL")
    _ = parser.add_argument(
        "--systems", default=DEFAULT_SYSTEMS, help="Comma-separated systems"
    )
    _ = parser.add_argument(
        "--corpus-filter",
        required=True,
        help="Path to corpus_doc_ids.txt (mandatory whitelist)",
    )
    _ = parser.add_argument(
        "--doc-scope", required=True, help="Path to doc_scope.jsonl"
    )
    _ = parser.add_argument(
        "--family-map", required=True, help="Path to family_map.json"
    )
    _ = parser.add_argument("--out-dir", required=True, help="Output directory")
    _ = parser.add_argument(
        "--shared-doc-ids",
        default=None,
        help="Optional shared_doc_ids.txt path; defaults to sibling of doc_scope",
    )
    _ = parser.add_argument(
        "--index",
        default=None,
        help="Alias/index name override (default: rag_chunks_{env}_current)",
    )
    _ = parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    _ = parser.add_argument("--seed", type=int, default=20260304)
    _ = parser.add_argument("--bootstrap-samples", type=int, default=2000)
    _ = parser.add_argument("--use-es-scope-fields", action="store_true")
    _ = parser.add_argument(
        "--reranker-model",
        default=None,
        help="Optional CrossEncoder model override for rerank systems",
    )
    return parser.parse_args()


def _load_eval_set(path: Path) -> list[EvalQuery]:
    rows: list[EvalQuery] = []
    for raw in read_jsonl(path):
        if not isinstance(raw, dict):
            raise RuntimeError("eval row must be an object")
        qid = str(raw.get("qid") or "").strip()
        split = str(raw.get("split") or "").strip()
        query = str(raw.get("query") or "").strip()
        target_device = str(raw.get("target_device") or "").strip()
        gold_raw = raw.get("gold_doc_ids")
        if not isinstance(gold_raw, list):
            raise RuntimeError(f"Invalid gold_doc_ids for qid={qid}")
        gold_doc_ids = [str(item).strip() for item in gold_raw if str(item).strip()]
        if not qid or not split or not query:
            raise RuntimeError("eval row missing qid/split/query")
        if not gold_doc_ids:
            raise RuntimeError(f"eval row missing gold_doc_ids: {qid}")
        rows.append(
            EvalQuery(
                qid=qid,
                split=split,
                query=query,
                target_device=target_device,
                gold_doc_ids=gold_doc_ids,
            )
        )
    if not rows:
        raise RuntimeError(f"No eval rows in {path}")
    return rows


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
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_alias_or_index(manager: EsIndexManager, requested: str) -> tuple[str, str]:
    default_alias = manager.get_alias_name()
    if requested == default_alias:
        target = manager.get_alias_target()
        if not target:
            raise RuntimeError(f"Alias '{requested}' does not point to any index")
        return requested, target

    try:
        alias_response = manager.es.indices.get_alias(name=requested)
        alias_targets = list(
            cast(dict[str, Any], cast(object, alias_response.body)).keys()
        )
        if not alias_targets:
            raise RuntimeError(f"Alias '{requested}' exists but has no targets")
        return requested, alias_targets[0]
    except NotFoundError:
        if manager.es.indices.exists(index=requested):
            return requested, requested
        raise RuntimeError(f"Alias/index '{requested}' was not found") from None


def _build_es_client() -> Elasticsearch:
    kwargs: dict[str, Any] = {
        "hosts": [search_settings.es_host],
        "verify_certs": True,
    }
    if search_settings.es_user and search_settings.es_password:
        kwargs["basic_auth"] = (search_settings.es_user, search_settings.es_password)
    return Elasticsearch(**kwargs)


def _merge_filters(
    base_filter: dict[str, Any], extra_filter: dict[str, Any] | None
) -> dict[str, Any]:
    if extra_filter is None:
        return base_filter
    combined = apply_scope_filter(base_filter, extra_filter)
    if combined is None:
        raise RuntimeError("unexpected empty combined filter")
    return cast(dict[str, Any], combined)


def _dedupe_top_doc_ids(
    results: list[RetrievalResult], top_k: int
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for result in results:
        doc_id = str(result.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        rows.append(
            {
                "doc_id": doc_id,
                "score": float(result.score),
                "metadata": dict(result.metadata or {}),
            }
        )
        if len(rows) >= top_k:
            break
    return rows


class RetrievalRunner:
    def __init__(
        self,
        *,
        index_name: str,
        corpus_doc_ids: list[str],
        reranker_model: str | None,
    ) -> None:
        self.es_engine = EsSearchEngine(
            es_client=_build_es_client(),
            index_name=index_name,
            text_fields=[
                "search_text^1.0",
                "chunk_summary^0.7",
                "chunk_keywords^0.8",
            ],
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
            doc_id=hit.doc_id,
            content=hit.content,
            score=hit.score,
            metadata=hit.metadata,
            raw_text=hit.raw_text,
        )

    def _maybe_rerank(
        self, *, query: str, results: list[RetrievalResult], top_k: int, rerank: bool
    ) -> list[RetrievalResult]:
        if not rerank:
            return results
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self._reranker_model,
                device=rag_settings.embedding_device,
            )
        return self._reranker.rerank(query=query, results=results, top_k=top_k)

    def run_bm25(
        self,
        *,
        query: str,
        top_k: int,
        extra_filter: dict[str, Any] | None,
        rerank: bool,
    ) -> list[dict[str, Any]]:
        final_filter = _merge_filters(self.base_filter, extra_filter)
        hits = self.es_engine.sparse_search(
            query_text=query, top_k=top_k, filters=final_filter
        )
        results = [self._to_result(hit) for hit in hits]
        return _dedupe_top_doc_ids(
            self._maybe_rerank(
                query=query, results=results, top_k=top_k, rerank=rerank
            ),
            top_k,
        )

    def run_dense(
        self,
        *,
        query: str,
        top_k: int,
        extra_filter: dict[str, Any] | None,
        rerank: bool,
    ) -> list[dict[str, Any]]:
        final_filter = _merge_filters(self.base_filter, extra_filter)
        hits = self.es_engine.dense_search(
            query_vector=self._embed_query(query), top_k=top_k, filters=final_filter
        )
        results = [self._to_result(hit) for hit in hits]
        return _dedupe_top_doc_ids(
            self._maybe_rerank(
                query=query, results=results, top_k=top_k, rerank=rerank
            ),
            top_k,
        )

    def run_hybrid(
        self,
        *,
        query: str,
        top_k: int,
        extra_filter: dict[str, Any] | None,
        rerank: bool,
    ) -> list[dict[str, Any]]:
        final_filter = _merge_filters(self.base_filter, extra_filter)
        hits = self.es_engine.hybrid_search(
            query_vector=self._embed_query(query),
            query_text=query,
            top_k=top_k,
            dense_weight=0.7,
            sparse_weight=0.3,
            filters=final_filter,
            use_rrf=True,
            rrf_k=60,
        )
        results = [self._to_result(hit) for hit in hits]
        return _dedupe_top_doc_ids(
            self._maybe_rerank(
                query=query, results=results, top_k=top_k, rerank=rerank
            ),
            top_k,
        )


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
    deduped: list[str] = []
    seen: set[str] = set()
    for item in matches:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped[:1]


def _extract_hard_equip_ids(query: str, equip_candidates: set[str]) -> list[str]:
    query_upper = query.upper()
    matched = [eid for eid in sorted(equip_candidates) if eid and eid in query_upper]
    return matched[:3]


def _build_scope_context(
    *,
    query: EvalQuery,
    device_candidates: list[str],
    equip_candidates: set[str],
) -> ScopeContext:
    hard_devices = _extract_hard_devices(query.query, device_candidates)
    hard_equip_ids = _extract_hard_equip_ids(query.query, equip_candidates)
    return ScopeContext(hard_devices=hard_devices, hard_equip_ids=hard_equip_ids)


def canonicalize_target_device(target_device: str, candidates: list[str]) -> str | None:
    target_compact = compact_key(target_device)
    if not target_compact:
        return None
    for candidate in candidates:
        if compact_key(candidate) == target_compact:
            return candidate
    return None


def _resolve_canonical_target_device(
    *,
    gold_doc_ids: list[str],
    target_device: str,
    doc_scope_by_doc_id: dict[str, str],
    candidates: list[str],
) -> str | None:
    if gold_doc_ids:
        gold_device = doc_scope_by_doc_id.get(gold_doc_ids[0], "").strip()
        if gold_device:
            return gold_device
    return canonicalize_target_device(target_device, candidates)


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


def _ranked_doc_ids(doc_rows: list[dict[str, Any]], k: int) -> list[str]:
    return [
        str(row.get("doc_id") or "").strip()
        for row in doc_rows[:k]
        if str(row.get("doc_id") or "").strip()
    ]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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
        doc_id = str(row.get("doc_id") or "")
        if doc_id in gold_set:
            mrr = 1.0 / float(idx)
            break
    metrics["mrr"] = mrr

    for k in K_VALUES:
        top_rows = doc_rows[:k]
        if not top_rows:
            metrics[f"raw_cont@{k}"] = 0.0
            metrics[f"adj_cont@{k}"] = 0.0
            metrics[f"shared@{k}"] = 0.0
            metrics[f"ce@{k}"] = 0.0
            metrics[f"hit@{k}"] = 0.0
            continue

        raw_oos = 0
        adj_oos = 0
        shared_count = 0
        ce = 0

        top_doc_ids = _ranked_doc_ids(top_rows, k)
        for row in top_rows:
            doc_id = str(row.get("doc_id") or "")
            metadata = cast(dict[str, Any], row.get("metadata") or {})
            doc_device = str(metadata.get("device_name") or "").strip()
            in_scope = bool(doc_device and doc_device in allowed_devices)
            is_shared = doc_id in shared_doc_ids
            if is_shared:
                shared_count += 1
            if not in_scope:
                raw_oos += 1
            if (not in_scope) and (not is_shared):
                adj_oos += 1
                ce = 1

        metrics[f"raw_cont@{k}"] = float(raw_oos) / float(k)
        metrics[f"adj_cont@{k}"] = float(adj_oos) / float(k)
        metrics[f"shared@{k}"] = float(shared_count) / float(k)
        metrics[f"ce@{k}"] = float(ce)
        metrics[f"hit@{k}"] = (
            1.0 if any(doc_id in gold_set for doc_id in top_doc_ids) else 0.0
        )

    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_delta(
    *,
    per_query_rows: list[dict[str, Any]],
    system_a: str,
    system_b: str,
    metric: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    metric_by_qid: dict[str, dict[str, float]] = defaultdict(dict)
    for row in per_query_rows:
        if str(row.get("status") or "ok") != "ok":
            continue
        qid = str(row.get("qid") or "")
        system_id = str(row.get("system_id") or "")
        raw_value = row.get(metric)
        if not qid or not system_id or not isinstance(raw_value, (int, float)):
            continue
        metric_by_qid[qid][system_id] = float(raw_value)

    paired = [
        (values[system_a], values[system_b])
        for values in metric_by_qid.values()
        if system_a in values and system_b in values
    ]
    if not paired:
        return {
            "system_a": system_a,
            "system_b": system_b,
            "metric": metric,
            "n_queries": 0,
            "delta_mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    rng = random.Random(seed)
    n = len(paired)
    deltas: list[float] = []
    for _ in range(samples):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        deltas.append(_mean([right - left for left, right in sample]))
    deltas.sort()

    point = _mean([right - left for left, right in paired])
    lower_idx = max(0, int(0.025 * len(deltas)) - 1)
    upper_idx = min(len(deltas) - 1, int(0.975 * len(deltas)) - 1)
    return {
        "system_a": system_a,
        "system_b": system_b,
        "metric": metric,
        "n_queries": n,
        "delta_mean": point,
        "ci_lower": deltas[lower_idx],
        "ci_upper": deltas[upper_idx],
    }


def _summarize(
    rows: list[dict[str, Any]], group_keys: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key) or "") for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key_tuple, items in sorted(grouped.items()):
        agg: dict[str, Any] = {k: v for k, v in zip(group_keys, key_tuple, strict=True)}
        ok_items = [item for item in items if str(item.get("status") or "ok") == "ok"]
        agg["n_queries"] = len(items)
        agg["n_ok"] = len(ok_items)
        agg["n_skipped"] = len(items) - len(ok_items)
        for metric in [
            "raw_cont@5",
            "adj_cont@5",
            "shared@5",
            "ce@5",
            "hit@5",
            "mrr",
        ]:
            values = [
                float(item[metric])
                for item in ok_items
                if isinstance(item.get(metric), (int, float))
            ]
            agg[f"mean_{metric}"] = _mean(values)
        out.append(agg)
    return out


def _build_manifest(
    *,
    args: argparse.Namespace,
    alias_name: str,
    resolved_index: str,
    corpus_doc_count: int,
    systems: list[str],
    device_doc_types: list[str],
    equip_doc_types: list[str],
) -> dict[str, Any]:
    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip()
    except Exception:
        git_sha = ""

    return {
        "git_sha": git_sha,
        "alias": alias_name,
        "resolved_index": resolved_index,
        "corpus_doc_id_count": corpus_doc_count,
        "systems": systems,
        "seed": int(args.seed),
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


def run() -> int:
    args = _parse_args()
    eval_set_path = Path(cast(str, args.eval_set))
    corpus_filter_path = Path(cast(str, args.corpus_filter))
    doc_scope_path = Path(cast(str, args.doc_scope))
    family_map_path = Path(cast(str, args.family_map))
    out_dir = Path(cast(str, args.out_dir))

    if not eval_set_path.exists() or not eval_set_path.is_file():
        print(f"eval-set file not found: {eval_set_path}", file=sys.stderr)
        return 1
    if not corpus_filter_path.exists() or not corpus_filter_path.is_file():
        print(f"corpus-filter file not found: {corpus_filter_path}", file=sys.stderr)
        return 1
    if not doc_scope_path.exists() or not doc_scope_path.is_file():
        print(f"doc-scope file not found: {doc_scope_path}", file=sys.stderr)
        return 1
    if not family_map_path.exists() or not family_map_path.is_file():
        print(f"family-map file not found: {family_map_path}", file=sys.stderr)
        return 1

    systems = [s.strip() for s in str(args.systems).split(",") if s.strip()]
    if not systems:
        print("No systems selected", file=sys.stderr)
        return 1

    try:
        eval_queries = _load_eval_set(eval_set_path)
        doc_scope_rows = _load_doc_scope(doc_scope_path)
        family_map = _load_family_map(family_map_path)
        device_to_family = {
            str(k): str(v)
            for k, v in cast(
                dict[str, Any], family_map.get("device_to_family") or {}
            ).items()
            if str(k).strip() and str(v).strip()
        }
        families = {
            str(fid): [str(member) for member in members if str(member).strip()]
            for fid, members in cast(
                dict[str, Any], family_map.get("families") or {}
            ).items()
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
        shared_doc_ids_path = (
            Path(cast(str, args.shared_doc_ids))
            if args.shared_doc_ids
            else doc_scope_path.parent / "shared_doc_ids.txt"
        )
        if shared_doc_ids_path.exists() and shared_doc_ids_path.is_file():
            shared_doc_ids.update(_load_lines(shared_doc_ids_path))

        device_candidates = sorted(
            {
                str(row.get("es_device_name") or "").strip()
                for row in doc_scope_rows
                if str(row.get("es_device_name") or "").strip()
            }
        )
        equip_candidates = {
            str(row.get("es_equip_id") or "").strip().upper()
            for row in doc_scope_rows
            if str(row.get("es_equip_id") or "").strip()
        }

        device_doc_types = sorted(
            {
                str(row.get("es_doc_type") or "").strip()
                for row in doc_scope_rows
                if str(row.get("scope_level") or "") == "device"
                and str(row.get("es_doc_type") or "").strip()
            }
        )
        equip_doc_types = sorted(
            {
                str(row.get("es_doc_type") or "").strip()
                for row in doc_scope_rows
                if str(row.get("scope_level") or "") == "equip"
                and str(row.get("es_doc_type") or "").strip()
            }
        )
        if not device_doc_types:
            device_doc_types = ["setup", "sop", "ts"]
        if not equip_doc_types:
            equip_doc_types = ["gcb", "myservice"]

        doc_type_by_doc_id = {
            str(row.get("es_doc_id") or "").strip(): str(
                row.get("es_doc_type") or ""
            ).strip()
            for row in doc_scope_rows
            if str(row.get("es_doc_id") or "").strip()
        }
        doc_scope_device_by_doc_id = {
            str(row.get("es_doc_id") or "").strip(): str(
                row.get("es_device_name") or ""
            ).strip()
            for row in doc_scope_rows
            if str(row.get("es_doc_id") or "").strip()
        }

        manager = EsIndexManager(
            es_host=search_settings.es_host,
            env=search_settings.es_env,
            index_prefix=search_settings.es_index_prefix,
            es_user=search_settings.es_user or None,
            es_password=search_settings.es_password or None,
            verify_certs=True,
        )
        alias_or_index = cast(str | None, args.index) or manager.get_alias_name()
        alias_name, resolved_index = _resolve_alias_or_index(manager, alias_or_index)

        retrieval = RetrievalRunner(
            index_name=resolved_index,
            corpus_doc_ids=corpus_doc_ids,
            reranker_model=cast(str | None, args.reranker_model),
        )

        per_query_rows: list[dict[str, Any]] = []

        for query in eval_queries:
            scope_ctx = _build_scope_context(
                query=query,
                device_candidates=device_candidates,
                equip_candidates=equip_candidates,
            )
            canonical_target_device = _resolve_canonical_target_device(
                gold_doc_ids=query.gold_doc_ids,
                target_device=query.target_device,
                doc_scope_by_doc_id=doc_scope_device_by_doc_id,
                candidates=device_candidates,
            )
            allowed_scope_devices = (
                {canonical_target_device} if canonical_target_device else set()
            )
            query_doc_type = doc_type_by_doc_id.get(query.gold_doc_ids[0], "")

            for system_id in systems:
                row_base: dict[str, Any] = {
                    "qid": query.qid,
                    "split": query.split,
                    "system_id": system_id,
                    "status": "ok",
                    "skip_reason": "",
                    "query_doc_type": query_doc_type,
                    "canonical_target_device": canonical_target_device or "",
                    "parsed_hard_devices": "|".join(scope_ctx.hard_devices),
                    "parsed_hard_equip_ids": "|".join(scope_ctx.hard_equip_ids),
                    "family_devices": "|".join(
                        _expand_family_devices(
                            scope_ctx.hard_devices,
                            device_to_family=device_to_family,
                            families=families,
                        )
                    ),
                }

                if not canonical_target_device:
                    skip_row = dict(row_base)
                    skip_row["status"] = "skipped"
                    skip_row["skip_reason"] = "target_device_unresolvable"
                    for metric_key in [
                        "raw_cont@1",
                        "raw_cont@3",
                        "raw_cont@5",
                        "raw_cont@10",
                        "adj_cont@1",
                        "adj_cont@3",
                        "adj_cont@5",
                        "adj_cont@10",
                        "shared@1",
                        "shared@3",
                        "shared@5",
                        "shared@10",
                        "ce@1",
                        "ce@3",
                        "ce@5",
                        "ce@10",
                        "hit@1",
                        "hit@3",
                        "hit@5",
                        "hit@10",
                        "mrr",
                    ]:
                        skip_row[metric_key] = ""
                    per_query_rows.append(skip_row)
                    continue

                if system_id in {"P2", "P3", "P4"}:
                    skip_row = dict(row_base)
                    skip_row["status"] = "skipped"
                    skip_row["skip_reason"] = "router_not_implemented"
                    for metric_key in [
                        "raw_cont@1",
                        "raw_cont@3",
                        "raw_cont@5",
                        "raw_cont@10",
                        "adj_cont@1",
                        "adj_cont@3",
                        "adj_cont@5",
                        "adj_cont@10",
                        "shared@1",
                        "shared@3",
                        "shared@5",
                        "shared@10",
                        "ce@1",
                        "ce@3",
                        "ce@5",
                        "ce@10",
                        "hit@1",
                        "hit@3",
                        "hit@5",
                        "hit@10",
                        "mrr",
                    ]:
                        skip_row[metric_key] = ""
                    per_query_rows.append(skip_row)
                    continue

                extra_filter: dict[str, Any] | None = None
                if system_id == "B0":
                    doc_rows = retrieval.run_bm25(
                        query=query.query,
                        top_k=int(args.top_k),
                        extra_filter=None,
                        rerank=False,
                    )
                elif system_id == "B1":
                    doc_rows = retrieval.run_dense(
                        query=query.query,
                        top_k=int(args.top_k),
                        extra_filter=None,
                        rerank=False,
                    )
                elif system_id == "B2":
                    doc_rows = retrieval.run_hybrid(
                        query=query.query,
                        top_k=int(args.top_k),
                        extra_filter=None,
                        rerank=False,
                    )
                elif system_id == "B3":
                    doc_rows = retrieval.run_hybrid(
                        query=query.query,
                        top_k=int(args.top_k),
                        extra_filter=None,
                        rerank=True,
                    )
                elif system_id == "B4":
                    if scope_ctx.hard_devices:
                        extra_filter = retrieval.es_engine.build_filter(
                            device_names=scope_ctx.hard_devices
                        )
                    doc_rows = retrieval.run_hybrid(
                        query=query.query,
                        top_k=int(args.top_k),
                        extra_filter=cast(dict[str, Any] | None, extra_filter),
                        rerank=True,
                    )
                elif system_id == "P1":
                    if bool(args.use_es_scope_fields):
                        extra_filter = build_scope_filter_by_fields(
                            allowed_devices=scope_ctx.hard_devices,
                            allowed_equip_ids=scope_ctx.hard_equip_ids,
                        )
                    else:
                        extra_filter = build_scope_filter_by_doc_ids(
                            allowed_devices=scope_ctx.hard_devices,
                            allowed_equip_ids=scope_ctx.hard_equip_ids,
                            shared_doc_ids=sorted(shared_doc_ids),
                            device_doc_types=device_doc_types,
                            equip_doc_types=equip_doc_types,
                        )
                    doc_rows = retrieval.run_hybrid(
                        query=query.query,
                        top_k=int(args.top_k),
                        extra_filter=cast(dict[str, Any] | None, extra_filter),
                        rerank=True,
                    )
                else:
                    skip_row = dict(row_base)
                    skip_row["status"] = "skipped"
                    skip_row["skip_reason"] = "unknown_system"
                    for metric_key in [
                        "raw_cont@1",
                        "raw_cont@3",
                        "raw_cont@5",
                        "raw_cont@10",
                        "adj_cont@1",
                        "adj_cont@3",
                        "adj_cont@5",
                        "adj_cont@10",
                        "shared@1",
                        "shared@3",
                        "shared@5",
                        "shared@10",
                        "ce@1",
                        "ce@3",
                        "ce@5",
                        "ce@10",
                        "hit@1",
                        "hit@3",
                        "hit@5",
                        "hit@10",
                        "mrr",
                    ]:
                        skip_row[metric_key] = ""
                    per_query_rows.append(skip_row)
                    continue

                metrics = _compute_metrics(
                    doc_rows=doc_rows,
                    gold_doc_ids=query.gold_doc_ids,
                    allowed_devices=allowed_scope_devices,
                    shared_doc_ids=shared_doc_ids,
                )
                out_row = dict(row_base)
                out_row.update(metrics)
                out_row["top_doc_ids"] = "|".join(
                    [str(row.get("doc_id") or "") for row in doc_rows]
                )
                out_row["in_corpus_top_docs"] = all(
                    str(row.get("doc_id") or "") in corpus_doc_set for row in doc_rows
                )
                per_query_rows.append(out_row)

        out_dir.mkdir(parents=True, exist_ok=True)

        per_query_fieldnames = [
            "qid",
            "split",
            "system_id",
            "status",
            "skip_reason",
            "query_doc_type",
            "canonical_target_device",
            "parsed_hard_devices",
            "parsed_hard_equip_ids",
            "family_devices",
            "raw_cont@1",
            "raw_cont@3",
            "raw_cont@5",
            "raw_cont@10",
            "adj_cont@1",
            "adj_cont@3",
            "adj_cont@5",
            "adj_cont@10",
            "shared@1",
            "shared@3",
            "shared@5",
            "shared@10",
            "ce@1",
            "ce@3",
            "ce@5",
            "ce@10",
            "hit@1",
            "hit@3",
            "hit@5",
            "hit@10",
            "mrr",
            "top_doc_ids",
            "in_corpus_top_docs",
        ]
        _write_csv(out_dir / "per_query.csv", per_query_rows, per_query_fieldnames)

        summary_all = _summarize(per_query_rows, ["system_id"])
        _write_csv(
            out_dir / "summary_all.csv",
            summary_all,
            [
                "system_id",
                "n_queries",
                "n_ok",
                "n_skipped",
                "mean_raw_cont@5",
                "mean_adj_cont@5",
                "mean_shared@5",
                "mean_ce@5",
                "mean_hit@5",
                "mean_mrr",
            ],
        )

        summary_by_split = _summarize(per_query_rows, ["split", "system_id"])
        _write_csv(
            out_dir / "summary_by_split.csv",
            summary_by_split,
            [
                "split",
                "system_id",
                "n_queries",
                "n_ok",
                "n_skipped",
                "mean_raw_cont@5",
                "mean_adj_cont@5",
                "mean_shared@5",
                "mean_ce@5",
                "mean_hit@5",
                "mean_mrr",
            ],
        )

        summary_by_doc_type = _summarize(
            per_query_rows, ["query_doc_type", "system_id"]
        )
        _write_csv(
            out_dir / "summary_by_doc_type.csv",
            summary_by_doc_type,
            [
                "query_doc_type",
                "system_id",
                "n_queries",
                "n_ok",
                "n_skipped",
                "mean_raw_cont@5",
                "mean_adj_cont@5",
                "mean_shared@5",
                "mean_ce@5",
                "mean_hit@5",
                "mean_mrr",
            ],
        )

        bootstrap_payload = {
            "seed": int(args.seed),
            "samples": int(args.bootstrap_samples),
            "comparisons": [
                _bootstrap_delta(
                    per_query_rows=per_query_rows,
                    system_a="B0",
                    system_b="P1",
                    metric="adj_cont@5",
                    seed=int(args.seed),
                    samples=int(args.bootstrap_samples),
                ),
                _bootstrap_delta(
                    per_query_rows=per_query_rows,
                    system_a="B0",
                    system_b="P1",
                    metric="hit@5",
                    seed=int(args.seed) + 1,
                    samples=int(args.bootstrap_samples),
                ),
            ],
        }
        write_json(
            out_dir / "bootstrap_ci.json", cast(dict[str, Any], bootstrap_payload)
        )

        manifest = _build_manifest(
            args=args,
            alias_name=alias_name,
            resolved_index=resolved_index,
            corpus_doc_count=len(corpus_doc_ids),
            systems=systems,
            device_doc_types=device_doc_types,
            equip_doc_types=equip_doc_types,
        )
        write_json(out_dir / "run_manifest.json", cast(dict[str, Any], manifest))

        print(f"Wrote outputs to {out_dir}")
        print(f"Resolved index: {resolved_index}")
        print(f"Queries: {len(eval_queries)}; systems: {','.join(systems)}")
        return 0
    except Exception as exc:
        print(f"evaluate_paper_a failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
