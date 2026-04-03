from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from backend.config.settings import rag_settings, search_settings
from backend.domain.doc_type_mapping import DOC_TYPE_GROUPS, normalize_doc_type
from backend.llm_infrastructure.preprocessing.registry import get_preprocessor
from backend.llm_infrastructure.reranking import get_reranker
from backend.llm_infrastructure.reranking.base import BaseReranker
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
from backend.llm_infrastructure.retrieval.postprocessors.relation_expander import RelationExpander
from backend.llm_infrastructure.retrieval.rrf import merge_retrieval_result_lists_rrf
from backend.llm_infrastructure.text_quality import is_noisy_chunk, strip_noisy_lines
from backend.services.embedding_service import EmbeddingService
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


def _normalize_doc_types_to_v3_groups(doc_types: list[str] | None) -> list[str] | None:
    """Convert variant doc_type values to v3 canonical group names.

    v3 indices store doc_type as group name (sop, setup, myservice, ...),
    but route/filter nodes may send variant names (SOP/Manual, Global SOP, ...).
    """
    if not doc_types:
        return doc_types
    # Build reverse map: normalized variant -> group name (lowercase)
    variant_to_group: dict[str, str] = {}
    for group_name, variants in DOC_TYPE_GROUPS.items():
        group_lower = group_name.lower()
        for v in variants:
            variant_to_group[normalize_doc_type(v)] = group_lower
    result: list[str] = []
    seen: set[str] = set()
    for dt in doc_types:
        normalized = normalize_doc_type(dt)
        group = variant_to_group.get(normalized, normalized)
        if group and group not in seen:
            seen.add(group)
            result.append(group)
    return result or None


def _normalize_device_names_to_v3(device_names: list[str] | None) -> list[str] | None:
    """Normalize device names to v3 format.

    v3 indices store device_name in multiple formats:
    - 'SUPRA_VPLUS' (underscore, from manifest)
    - 'SUPRA VPLUS' (space, from filename parser)
    - 'SUPRA Vplus' (mixed case, from runtime)
    Include all forms so the filter matches any format.
    """
    if not device_names:
        return device_names
    result: list[str] = []
    seen: set[str] = set()
    for name in device_names:
        name = str(name).strip()
        if not name:
            continue
        # original form
        if name not in seen:
            seen.add(name)
            result.append(name)
        # v3 normalized form: spaces → underscores, uppercase
        v3_name = name.replace(" ", "_").upper()
        if v3_name not in seen:
            seen.add(v3_name)
            result.append(v3_name)
        # uppercase with spaces (filename parser format)
        upper_space = name.upper()
        if upper_space not in seen:
            seen.add(upper_space)
            result.append(upper_space)
    return result or None


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        norm = eps
    return np.asarray(vec / norm, dtype=np.float32)


def _safe_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _result_key(result: RetrievalResult) -> tuple[str] | tuple[str, str]:
    metadata = result.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id not in (None, ""):
        return (result.doc_id, str(chunk_id))
    page = metadata.get("page")
    if page not in (None, ""):
        return (result.doc_id, str(page))
    return (result.doc_id,)


def _filter_noisy_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    if not results:
        return results
    filtered: list[RetrievalResult] = []
    dropped = 0
    for result in results:
        text = result.raw_text or result.content or ""
        cleaned = strip_noisy_lines(text)
        if not cleaned or is_noisy_chunk(cleaned):
            dropped += 1
            continue
        filtered.append(result)
    if dropped:
        logger.debug("Filtered %d noisy chunk_v3 results", dropped)
    return filtered


@dataclass
class _DenseCandidate:
    chunk_id: str
    score: float
    rank: int


class EsChunkV3SearchService:
    def __init__(
        self,
        *,
        es_client: Elasticsearch,
        content_index: str,
        embed_index: str,
        embedder: Any,
        preprocessor: Any = None,
        normalize_vectors: bool = True,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        reranker: BaseReranker | None = None,
        raptor_enabled: bool = False,
        relation_expander: RelationExpander | None = None,
    ) -> None:
        self.es = es_client
        self.content_index = content_index
        self.embed_index = embed_index
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.normalize_vectors = normalize_vectors
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.raptor_enabled = raptor_enabled
        self.relation_expander = relation_expander
        self._last_cross_type_suggestions: dict[str, list[RetrievalResult]] = {}

        self.es_engine = EsSearchEngine(
            es_client=es_client,
            index_name=content_index,
            text_fields=[
                "search_text^1.0",
                "chunk_summary^0.7",
                "chunk_keywords^0.8",
            ],
        )
        if self.raptor_enabled:
            logger.info("RAPTOR supplementary retrieval enabled")

    @classmethod
    def from_settings(
        cls,
        *,
        content_index: str | None = None,
        embed_index: str | None = None,
        es_client: Elasticsearch | None = None,
    ) -> EsChunkV3SearchService:
        if es_client is None:
            client_kwargs: dict[str, Any] = {
                "hosts": [search_settings.es_host],
                "verify_certs": True,
            }
            if search_settings.es_user and search_settings.es_password:
                client_kwargs["basic_auth"] = (
                    search_settings.es_user,
                    search_settings.es_password,
                )
            es_client = Elasticsearch(**client_kwargs)

        resolved_content_index = (content_index or search_settings.v3_content_index or "").strip()
        if not resolved_content_index:
            raise RuntimeError("SEARCH_V3_CONTENT_INDEX is required for chunk_v3 runtime")

        resolved_embed_index = (embed_index or search_settings.v3_embed_index or "").strip()
        if not resolved_embed_index:
            model_key = (search_settings.v3_embed_model_key or "").strip()
            if model_key:
                resolved_embed_index = f"chunk_v3_embed_{model_key}_v1"
        if not resolved_embed_index:
            raise RuntimeError(
                "SEARCH_V3_EMBED_INDEX (or SEARCH_V3_EMBED_MODEL_KEY) "
                "is required for chunk_v3 runtime"
            )

        embed_svc = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
            device=rag_settings.embedding_device,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )
        embedder = cast(Any, embed_svc.get_raw_embedder())
        embedder_dims = int(embedder.get_dimension())
        config_dims = int(search_settings.es_embedding_dims)
        if embedder_dims != config_dims:
            raise RuntimeError(
                "Embedding dimension mismatch for chunk_v3 runtime: "
                f"embedder={embedder_dims}, SEARCH_ES_EMBEDDING_DIMS={config_dims}, "
                f"RAG_EMBEDDING_METHOD={rag_settings.embedding_method}, "
                f"RAG_EMBEDDING_VERSION={rag_settings.embedding_version}"
            )

        preprocessor = get_preprocessor(
            rag_settings.preprocess_method,
            version=rag_settings.preprocess_version,
            level=rag_settings.preprocess_level,
        )

        cls._validate_v3_indices(
            es_client=es_client,
            content_index=resolved_content_index,
            embed_index=resolved_embed_index,
            embedder_dims=embedder_dims,
        )

        reranker: BaseReranker | None = None
        if rag_settings.rerank_enabled:
            reranker = get_reranker(
                rag_settings.rerank_method,
                version="v1",
                model_name=rag_settings.rerank_model,
                device=rag_settings.embedding_device,
            )

        relation_expander = RelationExpander.from_settings(rag_settings)

        return cls(
            es_client=es_client,
            content_index=resolved_content_index,
            embed_index=resolved_embed_index,
            embedder=embedder,
            preprocessor=preprocessor,
            normalize_vectors=rag_settings.vector_normalize,
            top_k=rag_settings.retrieval_top_k,
            dense_weight=rag_settings.hybrid_dense_weight,
            sparse_weight=rag_settings.hybrid_sparse_weight,
            rrf_k=rag_settings.hybrid_rrf_k,
            reranker=reranker,
            raptor_enabled=rag_settings.raptor_enabled,
            relation_expander=relation_expander,
        )

    @staticmethod
    def _validate_v3_indices(
        *,
        es_client: Elasticsearch,
        content_index: str,
        embed_index: str,
        embedder_dims: int,
    ) -> None:
        if not es_client.indices.exists(index=content_index):
            raise RuntimeError(f"chunk_v3 content index not found: {content_index}")
        if not es_client.indices.exists(index=embed_index):
            raise RuntimeError(f"chunk_v3 embed index not found: {embed_index}")

        mapping_response = es_client.indices.get_mapping(index=embed_index)
        index_mapping = mapping_response.get(embed_index)
        if index_mapping is None and mapping_response:
            index_mapping = next(iter(mapping_response.values()))
        if index_mapping is None:
            raise RuntimeError(f"Could not read mapping for embed index: {embed_index}")

        mappings = index_mapping.get("mappings", {})
        meta = mappings.get("_meta", {}) if isinstance(mappings, dict) else {}
        properties = mappings.get("properties", {}) if isinstance(mappings, dict) else {}
        embedding_field = properties.get("embedding", {}) if isinstance(properties, dict) else {}

        dims_from_meta = meta.get("dims") if isinstance(meta, dict) else None
        dims_from_field = embedding_field.get("dims") if isinstance(embedding_field, dict) else None

        index_dims = None
        if dims_from_meta is not None:
            index_dims = int(dims_from_meta)
        elif dims_from_field is not None:
            index_dims = int(dims_from_field)

        if index_dims is None:
            raise RuntimeError(
                f"Embed index '{embed_index}' does not expose embedding dims in mapping/_meta"
            )

        if index_dims != embedder_dims:
            embedding_model = meta.get("embedding_model") if isinstance(meta, dict) else ""
            raise RuntimeError(
                "chunk_v3 embed index dimension mismatch: "
                f"embed_index={embed_index}, index_dims={index_dims}, "
                f"embedder_dims={embedder_dims}, "
                f"RAG_EMBEDDING_METHOD={rag_settings.embedding_method}, "
                f"RAG_EMBEDDING_VERSION={rag_settings.embedding_version}, "
                f"index_embedding_model={embedding_model}"
            )

    def _preprocess_query(self, query: str) -> str:
        if self.preprocessor is None:
            return query
        try:
            processed = list(self.preprocessor.preprocess([query]))
            if processed:
                return str(processed[0])
        except Exception as exc:
            logger.warning("chunk_v3 query preprocessing failed: %s", exc)
        return query

    def _embed_query(self, query: str) -> list[float]:
        if hasattr(self.embedder, "embed_batch"):
            vec = self.embedder.embed_batch([query])[0]
        elif hasattr(self.embedder, "embed_texts"):
            vec = self.embedder.embed_texts([query])[0]
        elif hasattr(self.embedder, "embed"):
            vec = self.embedder.embed(query)
        else:
            raise TypeError("embedder must implement embed(), embed_batch(), or embed_texts()")

        arr = np.asarray(vec, dtype=np.float32)
        if self.normalize_vectors:
            arr = _l2_normalize(arr)
        return [float(value) for value in arr.tolist()]

    def _dense_search_candidates(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[_DenseCandidate]:
        knn_query: dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 2,
        }
        if filters:
            knn_query["filter"] = filters

        body: dict[str, Any] = {
            "knn": knn_query,
            "size": top_k,
            "_source": [
                "chunk_id",
                "doc_id",
                "doc_type",
                "device_name",
                "equip_id",
                "lang",
                "chapter",
                "content_hash",
            ],
            "track_total_hits": False,
        }

        response = self.es.search(index=self.embed_index, body=body)
        hits = response.get("hits", {}).get("hits", [])

        candidates: list[_DenseCandidate] = []
        for rank, hit in enumerate(hits, start=1):
            source = hit.get("_source", {})
            chunk_id = str(source.get("chunk_id") or hit.get("_id") or "").strip()
            if not chunk_id:
                continue
            score = _safe_float(hit.get("_score"))
            candidates.append(_DenseCandidate(chunk_id=chunk_id, score=score, rank=rank))
        return candidates

    def _sparse_search_results(
        self,
        *,
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
        text_fields: list[str] | None,
        device_boost: str | None,
        device_boost_weight: float,
    ) -> list[RetrievalResult]:
        engine = copy.copy(self.es_engine)
        if text_fields is not None:
            engine.text_fields = list(text_fields)

        text_query = engine._build_text_query(
            query_text,
            device_boost=device_boost,
            device_boost_weight=device_boost_weight,
        )
        query: dict[str, Any] = {"bool": {"must": text_query}}
        if filters:
            query["bool"]["filter"] = filters

        body: dict[str, Any] = {
            "query": query,
            "size": top_k,
            "_source": engine._source_fields(),
            "track_total_hits": False,
        }
        response = self.es.search(index=self.content_index, body=body)
        hits = engine._parse_hits(response)

        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            result = hit.to_retrieval_result()
            metadata = dict(result.metadata or {})
            metadata.setdefault("chunk_id", hit.chunk_id)
            metadata["sparse_score"] = hit.score
            metadata["sparse_rank"] = rank
            result.metadata = metadata
            results.append(result)
        return results

    def _mget_content_docs(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        response = self.es.mget(
            index=self.content_index,
            body={"ids": chunk_ids},
            _source=self.es_engine._source_fields(),
        )
        docs = response.get("docs", [])
        output: dict[str, dict[str, Any]] = {}
        for doc in docs:
            if doc.get("found"):
                output[str(doc.get("_id", ""))] = doc
        return output

    def _content_doc_to_result(self, doc: dict[str, Any], *, score: float) -> RetrievalResult:
        source = doc.get("_source", {})
        chunk_id = str(source.get("chunk_id") or doc.get("_id") or "")
        doc_id = str(source.get("doc_id") or chunk_id)
        content = str(source.get("search_text") or source.get("content") or "")
        raw_text = str(source.get("content") or content)

        metadata = {
            key: value
            for key, value in source.items()
            if key not in {"embedding", "content", "search_text"}
        }
        metadata["chunk_id"] = chunk_id

        return RetrievalResult(
            doc_id=doc_id,
            content=content,
            score=score,
            metadata=metadata,
            raw_text=raw_text,
        )

    def _join_dense_candidates(self, candidates: list[_DenseCandidate]) -> list[RetrievalResult]:
        if not candidates:
            return []

        ordered_chunk_ids: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            ordered_chunk_ids.append(candidate.chunk_id)

        docs_by_chunk_id = self._mget_content_docs(ordered_chunk_ids)
        results: list[RetrievalResult] = []
        missing = 0

        for candidate in candidates:
            doc = docs_by_chunk_id.get(candidate.chunk_id)
            if doc is None:
                missing += 1
                continue
            result = self._content_doc_to_result(doc, score=candidate.score)
            metadata = dict(result.metadata or {})
            metadata["dense_score"] = candidate.score
            metadata["dense_rank"] = candidate.rank
            result.metadata = metadata
            results.append(result)

        if missing:
            logger.warning(
                "chunk_v3 join dropped %d dense candidates missing in content index", missing
            )

        return results

    def _weighted_merge(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        *,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[RetrievalResult]:
        dense_rank: dict[tuple[str] | tuple[str, str], int] = {}
        sparse_rank: dict[tuple[str] | tuple[str, str], int] = {}
        dense_repr: dict[tuple[str] | tuple[str, str], RetrievalResult] = {}
        sparse_repr: dict[tuple[str] | tuple[str, str], RetrievalResult] = {}

        for rank, result in enumerate(dense_results, start=1):
            key = _result_key(result)
            if key not in dense_rank:
                dense_rank[key] = rank
                dense_repr[key] = result

        for rank, result in enumerate(sparse_results, start=1):
            key = _result_key(result)
            if key not in sparse_rank:
                sparse_rank[key] = rank
                sparse_repr[key] = result

        merged: list[tuple[RetrievalResult, float, int]] = []
        all_keys = list(dict.fromkeys([*dense_rank.keys(), *sparse_rank.keys()]))

        for key in all_keys:
            d_rank = dense_rank.get(key)
            s_rank = sparse_rank.get(key)
            d_score = 1.0 / (d_rank + 1) if d_rank is not None else 0.0
            s_score = 1.0 / (s_rank + 1) if s_rank is not None else 0.0
            combined_score = dense_weight * d_score + sparse_weight * s_score

            representative = dense_repr.get(key) or sparse_repr.get(key)
            if representative is None:
                continue

            metadata = dict(representative.metadata or {})
            metadata["weighted_dense_weight"] = dense_weight
            metadata["weighted_sparse_weight"] = sparse_weight
            metadata["weighted_dense_rank"] = d_rank
            metadata["weighted_sparse_rank"] = s_rank

            merged_result = RetrievalResult(
                doc_id=representative.doc_id,
                content=representative.content,
                score=combined_score,
                metadata=metadata,
                raw_text=representative.raw_text,
            )
            best_rank = min(r for r in [d_rank, s_rank] if r is not None)
            merged.append((merged_result, combined_score, best_rank))

        merged.sort(key=lambda item: (-item[1], item[2], item[0].doc_id))
        return [item[0] for item in merged]

    def _raptor_summary_search(
        self,
        query_text: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search RAPTOR summary nodes via BM25 and expand to leaf children.

        1. BM25 search on content index filtered to is_summary_node=true
        2. Collect children IDs from matched summaries
        3. Fetch children docs via mget → return as rank-scored list for RRF
        """
        summary_filter: list[dict[str, Any]] = [{"term": {"is_summary_node": True}}]
        if filters:
            if isinstance(filters, list):
                summary_filter.extend(filters)
            elif isinstance(filters, dict):
                summary_filter.append(filters)

        combined_filter = summary_filter if len(summary_filter) > 1 else summary_filter[0]

        # BM25 search for summaries
        text_query = self.es_engine._build_text_query(query_text)
        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": text_query,
                    "filter": combined_filter,
                }
            },
            "size": min(top_k, 20),
            "_source": ["chunk_id", "raptor_children_ids", "partition_key", "raptor_level", "content"],
            "track_total_hits": False,
        }
        try:
            response = self.es.search(index=self.content_index, body=body)
        except Exception as exc:
            logger.warning("raptor_supplementary: summary BM25 search failed: %s", exc)
            return []

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            logger.debug("raptor_supplementary: no summary nodes matched query")
            return []

        # Collect children IDs from matched summaries
        children_ids: list[str] = []
        seen: set[str] = set()
        summary_count = len(hits)
        for hit in hits:
            source = hit.get("_source", {})
            cids = source.get("raptor_children_ids") or []
            if isinstance(cids, str):
                cids = [cids]
            for cid in cids:
                cid_str = str(cid).strip()
                if cid_str and cid_str not in seen:
                    seen.add(cid_str)
                    children_ids.append(cid_str)

        if not children_ids:
            logger.debug("raptor_supplementary: %d summaries matched but no children IDs", summary_count)
            return []

        # Fetch children docs (limit to avoid excessive fetches)
        max_children = top_k * 3
        fetch_ids = children_ids[:max_children]
        children_docs = self._mget_content_docs(fetch_ids)

        results: list[RetrievalResult] = []
        for rank, cid in enumerate(fetch_ids, start=1):
            doc = children_docs.get(cid)
            if doc is None:
                continue
            score = 1.0 / rank  # rank-based score for RRF compatibility
            result = self._content_doc_to_result(doc, score=score)
            metadata = dict(result.metadata or {})
            metadata["raptor_supplementary"] = True
            metadata["raptor_summary_count"] = summary_count
            result.metadata = metadata
            results.append(result)

        logger.info(
            "raptor_supplementary: %d summaries → %d children IDs → %d leaf results",
            summary_count,
            len(children_ids),
            len(results),
        )
        return results

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        doc_type: str | None = None,
        doc_types: list[str] | None = None,
        doc_ids: list[str] | None = None,
        equip_ids: list[str] | None = None,
        lang: str | None = None,
        text_fields: list[str] | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        use_rrf: bool | None = None,
        rrf_k: int | None = None,
        device_name: str | None = None,
        device_names: list[str] | None = None,
        device_boost_weight: float = 2.0,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        _ = kwargs
        k = int(top_k or self.top_k)
        candidate_n = min(max(k * 3, 30), 200)

        processed_query = self._preprocess_query(query)
        query_vector = self._embed_query(processed_query)

        v3_doc_types = _normalize_doc_types_to_v3_groups(doc_types)
        v3_doc_type = _normalize_doc_types_to_v3_groups([doc_type])[0] if doc_type else None
        v3_device_names = _normalize_device_names_to_v3(device_names)

        filters = self.es_engine.build_filter(
            tenant_id=tenant_id,
            project_id=project_id,
            doc_type=v3_doc_type,
            doc_types=v3_doc_types,
            doc_ids=doc_ids,
            equip_ids=equip_ids,
            lang=lang,
            device_names=v3_device_names,
        )

        # Preserve base filters for RAPTOR summary search (which adds its own
        # is_summary_node=True filter). The main search gets an extra exclusion.
        base_filters = filters

        # Exclude RAPTOR summary nodes from main dense/sparse search.
        # Summary nodes have broad aggregated content that can dominate results;
        # they are handled separately by _raptor_summary_search → children expansion.
        if self.raptor_enabled:
            exclude_summary = {"term": {"is_summary_node": False}}
            if filters is None:
                filters = exclude_summary
            elif isinstance(filters, dict) and "bool" in filters and "must" in filters["bool"]:
                filters = copy.deepcopy(filters)
                filters["bool"]["must"].append(exclude_summary)
            else:
                filters = {"bool": {"must": [filters, exclude_summary]}}

        dense_candidates = self._dense_search_candidates(
            query_vector,
            top_k=candidate_n,
            filters=filters,
        )
        dense_results = self._join_dense_candidates(dense_candidates)

        sparse_results = self._sparse_search_results(
            query_text=processed_query,
            top_k=candidate_n,
            filters=filters,
            text_fields=text_fields,
            device_boost=device_name,
            device_boost_weight=device_boost_weight,
        )

        resolved_use_rrf = True if use_rrf is None else use_rrf
        resolved_rrf_k = int(rrf_k if rrf_k is not None else self.rrf_k)
        resolved_dense_weight = float(
            dense_weight if dense_weight is not None else self.dense_weight
        )
        resolved_sparse_weight = float(
            sparse_weight if sparse_weight is not None else self.sparse_weight
        )

        # RAPTOR supplementary: search summary nodes → expand children
        raptor_results: list[RetrievalResult] = []
        if self.raptor_enabled:
            raptor_results = self._raptor_summary_search(
                processed_query,
                query_vector,
                top_k=k,
                filters=base_filters,
            )

        if resolved_use_rrf:
            result_lists = [dense_results, sparse_results]
            if raptor_results:
                result_lists.append(raptor_results)
            merged = merge_retrieval_result_lists_rrf(
                result_lists,
                k=resolved_rrf_k,
            )
            for result in merged:
                metadata = dict(result.metadata or {})
                metadata.setdefault("rrf_k", resolved_rrf_k)
                result.metadata = metadata
        else:
            if raptor_results:
                logger.warning(
                    "raptor_supplementary: %d results discarded (use_rrf=False, weighted merge only supports 2 lists)",
                    len(raptor_results),
                )
            merged = self._weighted_merge(
                dense_results,
                sparse_results,
                dense_weight=resolved_dense_weight,
                sparse_weight=resolved_sparse_weight,
            )

        filtered = _filter_noisy_results(merged[:k])

        # Relation expansion: enrich results with related chunks
        if self.relation_expander and self.relation_expander.enabled:
            expand_result = self.relation_expander.expand(
                filtered, self.es, self.content_index
            )
            self._last_cross_type_suggestions = (
                expand_result.cross_type_suggestions()
            )
            if self._last_cross_type_suggestions:
                hint_types = {
                    dt: len(rs)
                    for dt, rs in self._last_cross_type_suggestions.items()
                }
                logger.info(
                    "[RelationExpander] cross_type suggestions: %s", hint_types
                )
            return expand_result.all_results()

        self._last_cross_type_suggestions = {}
        return filtered

    def fetch_doc_pages(
        self,
        doc_id: str,
        pages: list[int],
        *,
        max_docs: int | None = None,
    ) -> list[RetrievalResult]:
        page_values = sorted({p for p in pages if isinstance(p, int) and p > 0})
        if not doc_id or not page_values:
            return []

        size = max_docs or len(page_values)

        doc_filter = {
            "bool": {
                "should": [
                    {"term": {"doc_id": doc_id}},
                    {"term": {"doc_id.keyword": doc_id}},
                ],
                "minimum_should_match": 1,
            }
        }

        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "filter": [
                        doc_filter,
                        {"terms": {"page": page_values}},
                    ]
                }
            },
            "size": size,
            "sort": [{"page": "asc"}],
            "_source": self.es_engine._source_fields(),
            "track_total_hits": False,
        }

        try:
            response = self.es.search(index=self.content_index, body=body)
            hits = self.es_engine._parse_hits(response)
            return [hit.to_retrieval_result() for hit in hits]
        except Exception as exc:
            logger.warning(
                "chunk_v3 fetch_doc_pages failed: doc_id=%s pages=%s index=%s err=%s",
                doc_id,
                page_values,
                self.content_index,
                exc,
            )
            return []

    def fetch_doc_chunks(
        self,
        doc_id: str,
        *,
        max_chunks: int = 50,
    ) -> list[RetrievalResult]:
        if not doc_id:
            return []

        doc_filter = {
            "bool": {
                "should": [
                    {"term": {"doc_id": doc_id}},
                    {"term": {"doc_id.keyword": doc_id}},
                ],
                "minimum_should_match": 1,
            }
        }

        body: dict[str, Any] = {
            "query": {"bool": {"filter": [doc_filter]}},
            "size": max_chunks,
            "sort": [{"chunk_id": "asc"}],
            "_source": self.es_engine._source_fields(),
            "track_total_hits": False,
        }

        try:
            response = self.es.search(index=self.content_index, body=body)
            hits = self.es_engine._parse_hits(response)
            return [hit.to_retrieval_result() for hit in hits]
        except Exception as exc:
            logger.warning(
                "chunk_v3 fetch_doc_chunks failed: doc_id=%s index=%s err=%s",
                doc_id,
                self.content_index,
                exc,
            )
            return []

    def health_check(self) -> bool:
        try:
            return self.es.ping()
        except Exception as exc:
            logger.warning("chunk_v3 ES health check failed: %s", exc)
            return False


__all__ = ["EsChunkV3SearchService"]
