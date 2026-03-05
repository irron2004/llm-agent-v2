from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from elasticsearch import Elasticsearch

from backend.config.settings import rag_settings, search_settings
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.retrieval.engines.es_search import (
    EsSearchEngine,
    EsSearchHit,
)
from backend.llm_infrastructure.reranking.adapters.cross_encoder import (
    CrossEncoderReranker,
)
from backend.services.embedding_service import EmbeddingService

ROOT = Path(__file__).resolve().parents[2]
SMOKE_EVIDENCE_PATH = ROOT / ".sisyphus/evidence/task-09-retrieval-smoke.txt"
SMOKE_ERROR_EVIDENCE_PATH = (
    ROOT / ".sisyphus/evidence/task-09-retrieval-smoke-error.txt"
)


def _write_error_evidence(message: str) -> None:
    SMOKE_ERROR_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ = SMOKE_ERROR_EVIDENCE_PATH.write_text(message.strip() + "\n", encoding="utf-8")


def _load_corpus_doc_ids(corpus_doc_ids_path: str) -> list[str]:
    if not corpus_doc_ids_path:
        raise ValueError(
            "corpus_doc_ids_path is required. Retrieval runner always requires a corpus whitelist filter."
        )

    path = Path(corpus_doc_ids_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing corpus whitelist file: {path}. Provide corpus_doc_ids_path to enforce corpus filtering."
        )

    doc_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not doc_ids:
        raise ValueError(f"Corpus whitelist file is empty: {path}")
    return doc_ids


def _build_es_engine() -> EsSearchEngine:
    if search_settings.es_user and search_settings.es_password:
        es_client = Elasticsearch(
            hosts=[search_settings.es_host],
            verify_certs=True,
            basic_auth=(search_settings.es_user, search_settings.es_password),
        )
    else:
        es_client = Elasticsearch(
            hosts=[search_settings.es_host],
            verify_certs=True,
        )
    index_name = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"
    return EsSearchEngine(
        es_client=es_client,
        index_name=index_name,
        text_fields=[
            "search_text^1.0",
            "chunk_summary^0.7",
            "chunk_keywords^0.8",
        ],
    )


def _build_corpus_filter(
    engine: EsSearchEngine, doc_ids: list[str]
) -> dict[str, object]:
    corpus_filter = engine.build_filter(doc_ids=doc_ids)
    if corpus_filter is None:
        raise RuntimeError("Failed to build corpus whitelist filter")
    return corpus_filter


def _as_retrieval_result(hit: EsSearchHit) -> RetrievalResult:
    return RetrievalResult(
        doc_id=hit.doc_id,
        content=hit.content,
        score=hit.score,
        metadata=hit.metadata,
        raw_text=hit.raw_text,
    )


def _format_rows(results: list[RetrievalResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, result in enumerate(results, start=1):
        metadata = result.metadata or {}
        rows.append(
            {
                "rank": rank,
                "doc_id": result.doc_id,
                "score": float(result.score),
                "metadata": {
                    "device_name": metadata.get("device_name"),
                    "equip_id": metadata.get("equip_id"),
                    "doc_type": metadata.get("doc_type"),
                    "chunk_id": metadata.get("chunk_id"),
                    "page": metadata.get("page"),
                },
            }
        )
    return rows


def _maybe_rerank(
    *,
    query: str,
    results: list[RetrievalResult],
    top_k: int,
    rerank: bool,
    reranker_model_name: str | None,
) -> list[RetrievalResult]:
    if not rerank:
        return results

    reranker = CrossEncoderReranker(
        model_name=reranker_model_name,
        device=rag_settings.embedding_device,
    )
    return reranker.rerank(query=query, results=results, top_k=top_k)


def run_bm25(
    *,
    query: str,
    corpus_doc_ids_path: str,
    top_k: int = 10,
    rerank: bool = False,
    reranker_model_name: str | None = None,
) -> list[dict[str, object]]:
    engine = _build_es_engine()
    doc_ids = _load_corpus_doc_ids(corpus_doc_ids_path)
    filters = _build_corpus_filter(engine, doc_ids)
    hits = engine.sparse_search(query_text=query, top_k=top_k, filters=filters)
    results = [_as_retrieval_result(hit) for hit in hits]
    reranked = _maybe_rerank(
        query=query,
        results=results,
        top_k=top_k,
        rerank=rerank,
        reranker_model_name=reranker_model_name,
    )
    return _format_rows(reranked)


def run_dense(
    *,
    query: str,
    corpus_doc_ids_path: str,
    top_k: int = 10,
    rerank: bool = False,
    reranker_model_name: str | None = None,
) -> list[dict[str, object]]:
    engine = _build_es_engine()
    doc_ids = _load_corpus_doc_ids(corpus_doc_ids_path)
    filters = _build_corpus_filter(engine, doc_ids)

    embed_svc = EmbeddingService(
        method=rag_settings.embedding_method,
        version=rag_settings.embedding_version,
        device=rag_settings.embedding_device,
        use_cache=rag_settings.embedding_use_cache,
        cache_dir=rag_settings.embedding_cache_dir,
    )
    query_vector = np.asarray(embed_svc.embed_query(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm

    dense_vector = list(query_vector)
    hits = engine.dense_search(query_vector=dense_vector, top_k=top_k, filters=filters)
    results = [_as_retrieval_result(hit) for hit in hits]
    reranked = _maybe_rerank(
        query=query,
        results=results,
        top_k=top_k,
        rerank=rerank,
        reranker_model_name=reranker_model_name,
    )
    return _format_rows(reranked)


def run_hybrid(
    *,
    query: str,
    corpus_doc_ids_path: str,
    top_k: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    use_rrf: bool = True,
    rrf_k: int = 60,
    rerank: bool = False,
    reranker_model_name: str | None = None,
) -> list[dict[str, object]]:
    engine = _build_es_engine()
    doc_ids = _load_corpus_doc_ids(corpus_doc_ids_path)
    filters = _build_corpus_filter(engine, doc_ids)

    embed_svc = EmbeddingService(
        method=rag_settings.embedding_method,
        version=rag_settings.embedding_version,
        device=rag_settings.embedding_device,
        use_cache=rag_settings.embedding_use_cache,
        cache_dir=rag_settings.embedding_cache_dir,
    )
    query_vector = np.asarray(embed_svc.embed_query(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm

    hybrid_vector = list(query_vector)
    hits = engine.hybrid_search(
        query_vector=hybrid_vector,
        query_text=query,
        top_k=top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        filters=filters,
        use_rrf=use_rrf,
        rrf_k=rrf_k,
    )
    results = [_as_retrieval_result(hit) for hit in hits]
    reranked = _maybe_rerank(
        query=query,
        results=results,
        top_k=top_k,
        rerank=rerank,
        reranker_model_name=reranker_model_name,
    )
    return _format_rows(reranked)


def smoke_test(*, corpus_doc_ids_path: str) -> None:
    try:
        query = "semiconductor equipment alarm troubleshooting"
        hits = run_hybrid(
            query=query,
            corpus_doc_ids_path=corpus_doc_ids_path,
            top_k=5,
            use_rrf=True,
            rerank=False,
        )
        if not hits:
            raise RuntimeError("Smoke test returned zero hits")

        payload = {
            "query": query,
            "hit_count": len(hits),
            "hits": hits,
        }
        SMOKE_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ = SMOKE_EVIDENCE_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        _write_error_evidence(f"Retrieval smoke test failed: {exc}")
        raise


__all__ = [
    "run_bm25",
    "run_dense",
    "run_hybrid",
    "smoke_test",
]
