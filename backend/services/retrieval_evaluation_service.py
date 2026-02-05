"""Elasticsearch-backed retrieval evaluation service (query-unit storage).

Provides storage and retrieval of query-unit relevance evaluations for:
- Retrieval test set creation
- Search parameter tuning
- Retrieval quality analysis

Storage structure (query-unit):
    {
        "query_id": "sess1:turn1",  # PK (chat: session:turn, search: search:timestamp)
        "query": "원본 쿼리",
        "relevant_docs": ["doc_001", "doc_003"],      # Auto-generated (score >= 3)
        "irrelevant_docs": ["doc_002", "doc_004"],    # Auto-generated (score < 3)
        "doc_details": [{ ... }]                      # Required, individual doc scores
    }

Usage:
    # Create from settings
    svc = RetrievalEvaluationService.from_settings()

    # Save query-unit evaluation
    svc.save_query_evaluation(query_id, data)

    # Get evaluation by query_id
    eval = svc.get_query_evaluation(query_id)

    # Export for retrieval testing
    data = svc.export_for_retrieval_test(min_relevance=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from elasticsearch import Elasticsearch, NotFoundError

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.mappings import (
    get_retrieval_evaluation_mapping,
    get_index_settings,
)

logger = logging.getLogger(__name__)

# Index naming
RETRIEVAL_EVALUATION_INDEX_PREFIX = "retrieval_evaluations"
RETRIEVAL_EVALUATION_SCHEMA_VERSION = "v2"  # Updated for query-unit structure

# Relevance threshold
DEFAULT_RELEVANCE_THRESHOLD = 3


@dataclass
class DocDetail:
    """Individual document detail in evaluation."""

    doc_id: str
    doc_rank: int  # 1-based
    doc_title: str = ""
    relevance_score: int = 0  # 1-5
    retrieval_score: Optional[float] = None
    doc_snippet: str = ""
    chunk_id: Optional[str] = None
    page: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_rank": self.doc_rank,
            "doc_title": self.doc_title,
            "relevance_score": self.relevance_score,
            "retrieval_score": self.retrieval_score,
            "doc_snippet": self.doc_snippet,
            "chunk_id": self.chunk_id,
            "page": self.page,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocDetail":
        return cls(
            doc_id=data.get("doc_id", ""),
            doc_rank=data.get("doc_rank", 0),
            doc_title=data.get("doc_title", ""),
            relevance_score=data.get("relevance_score", 0),
            retrieval_score=data.get("retrieval_score"),
            doc_snippet=data.get("doc_snippet", ""),
            chunk_id=data.get("chunk_id"),
            page=data.get("page"),
        )


@dataclass
class QueryEvaluation:
    """Query-unit evaluation containing multiple document relevance scores."""

    query_id: str  # PK: chat="{session_id}:{turn_id}", search="search:{timestamp}"
    source: Literal["chat", "search"]
    query: str
    doc_details: list[DocDetail] = field(default_factory=list)
    # Chat context (optional for search)
    session_id: Optional[str] = None
    turn_id: Optional[int] = None
    # Filter context
    filter_devices: Optional[list[str]] = None
    filter_doc_types: Optional[list[str]] = None
    search_queries: Optional[list[str]] = None
    # Search params (search only)
    search_params: Optional[dict[str, Any]] = None
    # Reviewer info
    reviewer_name: Optional[str] = None
    # Auto-generated fields
    relevant_docs: list[str] = field(default_factory=list)
    irrelevant_docs: list[str] = field(default_factory=list)
    # Timestamps
    ts: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Calculate derived fields (relevant_docs, irrelevant_docs)."""
        self._calculate_relevance_lists()

    def _calculate_relevance_lists(self, threshold: int = DEFAULT_RELEVANCE_THRESHOLD):
        """Calculate relevant_docs and irrelevant_docs from doc_details."""
        self.relevant_docs = []
        self.irrelevant_docs = []
        for detail in self.doc_details:
            if detail.relevance_score >= threshold:
                self.relevant_docs.append(detail.doc_id)
            else:
                self.irrelevant_docs.append(detail.doc_id)

    def to_dict(self) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        self._calculate_relevance_lists()
        return {
            "query_id": self.query_id,
            "source": self.source,
            "query": self.query,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "relevant_docs": self.relevant_docs,
            "irrelevant_docs": self.irrelevant_docs,
            "doc_details": [d.to_dict() for d in self.doc_details],
            "filter_devices": self.filter_devices,
            "filter_doc_types": self.filter_doc_types,
            "search_queries": self.search_queries,
            "search_params": self.search_params,
            "reviewer_name": self.reviewer_name,
            "ts": (self.ts or datetime.utcnow()).isoformat(),
            "created_at": self.created_at.isoformat() if self.created_at else now,
            "updated_at": now,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryEvaluation":
        doc_details = [
            DocDetail.from_dict(d) for d in data.get("doc_details", [])
        ]
        return cls(
            query_id=data.get("query_id", ""),
            source=data.get("source", "chat"),
            query=data.get("query", ""),
            session_id=data.get("session_id"),
            turn_id=data.get("turn_id"),
            doc_details=doc_details,
            filter_devices=data.get("filter_devices"),
            filter_doc_types=data.get("filter_doc_types"),
            search_queries=data.get("search_queries"),
            search_params=data.get("search_params"),
            reviewer_name=data.get("reviewer_name"),
            relevant_docs=data.get("relevant_docs", []),
            irrelevant_docs=data.get("irrelevant_docs", []),
            ts=datetime.fromisoformat(data["ts"]) if data.get("ts") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


# Legacy dataclass for backwards compatibility
@dataclass
class RetrievalEvaluation:
    """Document relevance evaluation for a retrieved document.

    @deprecated: Use QueryEvaluation for query-unit storage.
    """

    session_id: str
    turn_id: int
    doc_id: str
    relevance_score: int  # 1-5
    query: str
    doc_rank: int  # 1-based rank
    doc_title: str = ""
    doc_snippet: str = ""
    message_id: Optional[str] = None
    chunk_id: Optional[str] = None
    retrieval_score: Optional[float] = None
    is_relevant: bool = False
    reviewer_name: Optional[str] = None
    filter_devices: Optional[list[str]] = None
    filter_doc_types: Optional[list[str]] = None
    search_queries: Optional[list[str]] = None
    ts: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        self.is_relevant = self.relevance_score >= DEFAULT_RELEVANCE_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "message_id": self.message_id,
            "query": self.query,
            "query_id": f"{self.session_id}:{self.turn_id}",
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "doc_title": self.doc_title,
            "doc_snippet": self.doc_snippet,
            "doc_rank": self.doc_rank,
            "retrieval_score": self.retrieval_score,
            "relevance_score": self.relevance_score,
            "is_relevant": self.is_relevant,
            "reviewer_name": self.reviewer_name,
            "filter_devices": self.filter_devices,
            "filter_doc_types": self.filter_doc_types,
            "search_queries": self.search_queries,
            "ts": (self.ts or datetime.utcnow()).isoformat(),
            "created_at": self.created_at.isoformat() if self.created_at else now,
            "updated_at": now,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalEvaluation":
        return cls(
            session_id=data.get("session_id", ""),
            turn_id=data.get("turn_id", 0),
            message_id=data.get("message_id"),
            query=data.get("query", ""),
            doc_id=data.get("doc_id", ""),
            chunk_id=data.get("chunk_id"),
            doc_title=data.get("doc_title", ""),
            doc_snippet=data.get("doc_snippet", ""),
            doc_rank=data.get("doc_rank", 0),
            retrieval_score=data.get("retrieval_score"),
            relevance_score=data.get("relevance_score", 0),
            is_relevant=data.get("is_relevant", False),
            reviewer_name=data.get("reviewer_name"),
            filter_devices=data.get("filter_devices"),
            filter_doc_types=data.get("filter_doc_types"),
            search_queries=data.get("search_queries"),
            ts=datetime.fromisoformat(data["ts"]) if data.get("ts") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


class RetrievalEvaluationService:
    """Elasticsearch-backed retrieval evaluation storage and retrieval."""

    def __init__(
        self,
        *,
        es_client: Elasticsearch,
        index: str,
    ) -> None:
        """Initialize retrieval evaluation service.

        Args:
            es_client: Elasticsearch client instance.
            index: Index name or alias to use.
        """
        self.es = es_client
        self.index = index

    @classmethod
    def from_settings(
        cls,
        *,
        es_client: Elasticsearch | None = None,
        env: str | None = None,
    ) -> "RetrievalEvaluationService":
        """Create RetrievalEvaluationService from global settings.

        Args:
            es_client: Pre-configured ES client. Created from settings if None.
            env: Environment name. Defaults to search_settings.es_env.

        Returns:
            Configured RetrievalEvaluationService instance.
        """
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

        env = env or search_settings.es_env
        index = f"{RETRIEVAL_EVALUATION_INDEX_PREFIX}_{env}_current"

        return cls(es_client=es_client, index=index)

    def ensure_index(self) -> bool:
        """Ensure the retrieval evaluation index exists, create if not.

        Returns:
            True if index exists or was created successfully.
        """
        # Check if alias exists
        if self.es.indices.exists_alias(name=self.index):
            return True

        # Create versioned index
        env = search_settings.es_env
        versioned_index = f"{RETRIEVAL_EVALUATION_INDEX_PREFIX}_{env}_v2"

        if not self.es.indices.exists(index=versioned_index):
            self.es.indices.create(
                index=versioned_index,
                body={
                    "settings": get_index_settings(),
                    "mappings": get_retrieval_evaluation_mapping(),
                },
            )
            logger.info(f"Created retrieval evaluation index: {versioned_index}")

        # Create alias pointing to the versioned index
        if not self.es.indices.exists_alias(name=self.index):
            self.es.indices.put_alias(index=versioned_index, name=self.index)
            logger.info(f"Created alias {self.index} -> {versioned_index}")

        return True

    # =========================================================================
    # Query-unit API (new)
    # =========================================================================

    def save_query_evaluation(self, query_id: str, evaluation: QueryEvaluation) -> str:
        """Save query-unit evaluation to ES (upsert).

        Args:
            query_id: Query ID (PK).
            evaluation: QueryEvaluation to save.

        Returns:
            Document ID (query_id).
        """
        evaluation.query_id = query_id
        self.es.index(
            index=self.index,
            id=query_id,
            body=evaluation.to_dict(),
            refresh=True,
        )
        logger.debug(f"Saved query evaluation {query_id}")
        return query_id

    def get_query_evaluation(self, query_id: str) -> QueryEvaluation | None:
        """Get evaluation by query_id.

        Args:
            query_id: Query ID.

        Returns:
            QueryEvaluation if found, None otherwise.
        """
        try:
            result = self.es.get(index=self.index, id=query_id)
            return QueryEvaluation.from_dict(result["_source"])
        except NotFoundError:
            return None

    def list_query_evaluations(
        self,
        limit: int = 100,
        offset: int = 0,
        source: Optional[Literal["chat", "search"]] = None,
    ) -> tuple[list[QueryEvaluation], int]:
        """List query evaluations.

        Args:
            limit: Maximum number of results.
            offset: Offset for pagination.
            source: Filter by source ("chat" or "search").

        Returns:
            Tuple of (list of QueryEvaluation, total count).
        """
        query_body: dict[str, Any] = {
            "size": limit,
            "from": offset,
            "sort": [{"ts": "desc"}],
        }

        if source:
            query_body["query"] = {"term": {"source": source}}
        else:
            query_body["query"] = {"match_all": {}}

        try:
            result = self.es.search(index=self.index, body=query_body)
            hits = result.get("hits", {})
            total = hits.get("total", {}).get("value", 0)
            items = [QueryEvaluation.from_dict(hit["_source"]) for hit in hits.get("hits", [])]
            return items, total
        except Exception as e:
            logger.error(f"Failed to list query evaluations: {e}")
            return [], 0

    def delete_query_evaluation(self, query_id: str) -> bool:
        """Delete query evaluation by query_id.

        Args:
            query_id: Query ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            self.es.delete(index=self.index, id=query_id, refresh=True)
            return True
        except NotFoundError:
            return False

    # =========================================================================
    # Export API
    # =========================================================================

    def export_for_retrieval_test(
        self,
        min_relevance: int = DEFAULT_RELEVANCE_THRESHOLD,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Export evaluation data for retrieval testing.

        Exports query-unit data with relevant/irrelevant document lists.
        Suitable for evaluating retrieval pipelines with metrics like NDCG, MAP.

        Args:
            min_relevance: Minimum relevance score to include as "relevant".
            limit: Maximum number of records to export.

        Returns:
            List of dicts with query, relevant_docs, irrelevant_docs.
        """
        query = {
            "size": limit,
            "query": {"match_all": {}},
            "sort": [{"ts": "desc"}],
            "_source": [
                "query_id",
                "source",
                "query",
                "relevant_docs",
                "irrelevant_docs",
                "doc_details",
                "filter_devices",
                "filter_doc_types",
                "search_queries",
                "ts",
            ],
        }

        try:
            result = self.es.search(index=self.index, body=query)
            hits = result.get("hits", {}).get("hits", [])
            export_data = []
            for hit in hits:
                src = hit["_source"]
                # Recalculate relevance based on min_relevance if different from default
                if min_relevance != DEFAULT_RELEVANCE_THRESHOLD:
                    doc_details = src.get("doc_details", [])
                    relevant = [d["doc_id"] for d in doc_details if d.get("relevance_score", 0) >= min_relevance]
                    irrelevant = [d["doc_id"] for d in doc_details if d.get("relevance_score", 0) < min_relevance]
                    src["relevant_docs"] = relevant
                    src["irrelevant_docs"] = irrelevant
                export_data.append(src)
            return export_data
        except Exception as e:
            logger.error(f"Failed to export retrieval evaluations: {e}")
            return []

    def get_statistics(self) -> dict[str, Any]:
        """Get retrieval evaluation statistics.

        Returns:
            Dict with total count, average relevance, and relevance distribution.
        """
        query = {
            "size": 0,
            "aggs": {
                "total_queries": {"value_count": {"field": "query_id"}},
                "by_source": {"terms": {"field": "source"}},
                "avg_docs_per_query": {
                    "nested": {"path": "doc_details"},
                    "aggs": {
                        "doc_count": {"value_count": {"field": "doc_details.doc_id"}},
                        "avg_relevance": {"avg": {"field": "doc_details.relevance_score"}},
                        "relevance_distribution": {"terms": {"field": "doc_details.relevance_score"}},
                    },
                },
            },
        }

        try:
            result = self.es.search(index=self.index, body=query)
            aggs = result.get("aggregations", {})
            nested = aggs.get("avg_docs_per_query", {})
            total_queries = int(aggs.get("total_queries", {}).get("value", 0))
            total_docs = int(nested.get("doc_count", {}).get("value", 0))
            return {
                "total_queries": total_queries,
                "total_doc_evaluations": total_docs,
                "avg_docs_per_query": total_docs / total_queries if total_queries > 0 else 0,
                "avg_relevance": nested.get("avg_relevance", {}).get("value"),
                "source_distribution": {
                    bucket["key"]: bucket["doc_count"]
                    for bucket in aggs.get("by_source", {}).get("buckets", [])
                },
                "relevance_distribution": {
                    str(bucket["key"]): bucket["doc_count"]
                    for bucket in nested.get("relevance_distribution", {}).get("buckets", [])
                },
            }
        except Exception as e:
            logger.error(f"Failed to get retrieval evaluation statistics: {e}")
            return {}

    # =========================================================================
    # Legacy API (deprecated, for backwards compatibility)
    # =========================================================================

    def save_evaluation(self, evaluation: RetrievalEvaluation) -> str:
        """Save evaluation to ES.

        @deprecated: Use save_query_evaluation for query-unit storage.

        Args:
            evaluation: Evaluation to save.

        Returns:
            Document ID of the saved evaluation.
        """
        doc_id = f"{evaluation.session_id}:{evaluation.turn_id}:{evaluation.doc_id}"
        self.es.index(
            index=self.index,
            id=doc_id,
            body=evaluation.to_dict(),
            refresh=True,
        )
        logger.debug(f"Saved retrieval evaluation {doc_id}")
        return doc_id

    def get_evaluation(
        self, session_id: str, turn_id: int, doc_id: str
    ) -> RetrievalEvaluation | None:
        """Get evaluation for a specific document in a turn.

        @deprecated: Use get_query_evaluation for query-unit storage.

        Args:
            session_id: Session ID.
            turn_id: Turn number within session.
            doc_id: Document ID.

        Returns:
            RetrievalEvaluation if found, None otherwise.
        """
        es_doc_id = f"{session_id}:{turn_id}:{doc_id}"
        try:
            result = self.es.get(
                index=self.index,
                id=es_doc_id,
            )
            return RetrievalEvaluation.from_dict(result["_source"])
        except NotFoundError:
            return None

    def list_evaluations_for_turn(
        self, session_id: str, turn_id: int
    ) -> list[RetrievalEvaluation]:
        """List all evaluations for a specific turn.

        @deprecated: Use get_query_evaluation with query_id="{session_id}:{turn_id}".

        Args:
            session_id: Session ID.
            turn_id: Turn number within session.

        Returns:
            List of evaluations for the turn, sorted by doc_rank.
        """
        query = {
            "size": 100,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"session_id": session_id}},
                        {"term": {"turn_id": turn_id}},
                    ]
                }
            },
            "sort": [{"doc_rank": "asc"}],
        }

        try:
            result = self.es.search(index=self.index, body=query)
            hits = result.get("hits", {}).get("hits", [])
            return [RetrievalEvaluation.from_dict(hit["_source"]) for hit in hits]
        except Exception as e:
            logger.error(f"Failed to list evaluations for turn: {e}")
            return []
