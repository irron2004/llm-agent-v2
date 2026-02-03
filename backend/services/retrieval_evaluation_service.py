"""Elasticsearch-backed retrieval evaluation service.

Provides storage and retrieval of document relevance evaluations for:
- Retrieval test set creation
- Search parameter tuning
- Retrieval quality analysis

Usage:
    # Create from settings
    svc = RetrievalEvaluationService.from_settings()

    # Save evaluation
    evaluation = RetrievalEvaluation(...)
    svc.save_evaluation(evaluation)

    # Get evaluation for a document
    eval = svc.get_evaluation(session_id, turn_id, doc_id)

    # Export for retrieval testing
    data = svc.export_for_retrieval_test(min_relevance=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from elasticsearch import Elasticsearch, NotFoundError

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.mappings import (
    get_retrieval_evaluation_mapping,
    get_index_settings,
)

logger = logging.getLogger(__name__)

# Index naming
RETRIEVAL_EVALUATION_INDEX_PREFIX = "retrieval_evaluations"
RETRIEVAL_EVALUATION_SCHEMA_VERSION = "v1"


@dataclass
class RetrievalEvaluation:
    """Document relevance evaluation for a retrieved document."""

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
    search_queries: Optional[list[str]] = None  # Multi-query expansion results
    ts: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Calculate derived fields."""
        self.is_relevant = self.relevance_score >= 3

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
        versioned_index = f"{RETRIEVAL_EVALUATION_INDEX_PREFIX}_{env}_v1"

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

    def save_evaluation(self, evaluation: RetrievalEvaluation) -> str:
        """Save evaluation to ES.

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

    def export_for_retrieval_test(
        self,
        min_relevance: int = 3,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Export evaluation data for retrieval testing.

        Exports query-document pairs with relevance labels.
        Suitable for evaluating retrieval pipelines with metrics like NDCG, MAP.

        Args:
            min_relevance: Minimum relevance score to include as "relevant".
            limit: Maximum number of records to export.

        Returns:
            List of dicts with query, doc_id, relevance_score, and context.
        """
        query = {
            "size": limit,
            "query": {"match_all": {}},
            "sort": [{"query_id": "asc"}, {"doc_rank": "asc"}],
            "_source": [
                "query",
                "query_id",
                "doc_id",
                "chunk_id",
                "doc_title",
                "doc_rank",
                "retrieval_score",
                "relevance_score",
                "is_relevant",
                "filter_devices",
                "filter_doc_types",
                "ts",
            ],
        }

        try:
            result = self.es.search(index=self.index, body=query)
            hits = result.get("hits", {}).get("hits", [])
            return [hit["_source"] for hit in hits]
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
                "total_count": {"value_count": {"field": "doc_id"}},
                "avg_relevance": {"avg": {"field": "relevance_score"}},
                "relevant_count": {
                    "filter": {"term": {"is_relevant": True}},
                },
                "relevance_distribution": {
                    "terms": {"field": "relevance_score"},
                },
                "unique_queries": {
                    "cardinality": {"field": "query_id"},
                },
            },
        }

        try:
            result = self.es.search(index=self.index, body=query)
            aggs = result.get("aggregations", {})
            total = int(aggs.get("total_count", {}).get("value", 0))
            relevant = int(aggs.get("relevant_count", {}).get("doc_count", 0))
            return {
                "total_count": total,
                "avg_relevance": aggs.get("avg_relevance", {}).get("value"),
                "relevant_count": relevant,
                "relevant_ratio": relevant / total if total > 0 else 0,
                "unique_queries": int(aggs.get("unique_queries", {}).get("value", 0)),
                "relevance_distribution": {
                    str(bucket["key"]): bucket["doc_count"]
                    for bucket in aggs.get("relevance_distribution", {}).get("buckets", [])
                },
            }
        except Exception as e:
            logger.error(f"Failed to get retrieval evaluation statistics: {e}")
            return {}
