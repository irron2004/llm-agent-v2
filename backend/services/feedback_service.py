"""Elasticsearch-backed feedback service.

Provides storage and retrieval of detailed feedback for LLM fine-tuning.
Stores feedback separately from chat turns for efficient analysis and export.

Usage:
    # Create from settings
    svc = FeedbackService.from_settings()

    # Save feedback
    feedback = Feedback(...)
    svc.save_feedback(feedback)

    # List feedback
    items, total = svc.list_feedback(limit=50)

    # Export for fine-tuning
    data = svc.export_for_finetuning(min_score=4.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from elasticsearch import Elasticsearch, NotFoundError

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.mappings import (
    get_feedback_mapping,
    get_index_settings,
)

logger = logging.getLogger(__name__)

# Index naming
FEEDBACK_INDEX_PREFIX = "feedback"
FEEDBACK_SCHEMA_VERSION = "v1"


@dataclass
class Feedback:
    """Detailed feedback for a conversation turn."""

    session_id: str
    turn_id: int
    user_text: str
    assistant_text: str
    accuracy: int  # 1-5
    completeness: int  # 1-5
    relevance: int  # 1-5
    avg_score: float = 0.0
    rating: str = ""  # "up" | "down"
    comment: Optional[str] = None
    reviewer_name: Optional[str] = None  # 피드백 제출자 이름 (선택)
    logs: Optional[list[str]] = None
    ts: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Calculate derived fields."""
        if self.avg_score == 0.0:
            self.avg_score = (self.accuracy + self.completeness + self.relevance) / 3.0
        if not self.rating:
            self.rating = "up" if self.avg_score >= 3.0 else "down"

    def to_dict(self) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "avg_score": self.avg_score,
            "rating": self.rating,
            "comment": self.comment,
            "reviewer_name": self.reviewer_name,
            "logs": self.logs,
            "ts": (self.ts or datetime.utcnow()).isoformat(),
            "created_at": self.created_at.isoformat() if self.created_at else now,
            "updated_at": now,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Feedback":
        return cls(
            session_id=data.get("session_id", ""),
            turn_id=data.get("turn_id", 0),
            user_text=data.get("user_text", ""),
            assistant_text=data.get("assistant_text", ""),
            accuracy=data.get("accuracy", 0),
            completeness=data.get("completeness", 0),
            relevance=data.get("relevance", 0),
            avg_score=data.get("avg_score", 0.0),
            rating=data.get("rating", ""),
            comment=data.get("comment"),
            reviewer_name=data.get("reviewer_name"),
            logs=data.get("logs"),
            ts=datetime.fromisoformat(data["ts"]) if data.get("ts") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


class FeedbackService:
    """Elasticsearch-backed feedback storage and retrieval."""

    def __init__(
        self,
        *,
        es_client: Elasticsearch,
        index: str,
    ) -> None:
        """Initialize feedback service.

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
    ) -> "FeedbackService":
        """Create FeedbackService from global settings.

        Args:
            es_client: Pre-configured ES client. Created from settings if None.
            env: Environment name. Defaults to search_settings.es_env.

        Returns:
            Configured FeedbackService instance.
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
        index = f"{FEEDBACK_INDEX_PREFIX}_{env}_current"

        return cls(es_client=es_client, index=index)

    def ensure_index(self) -> bool:
        """Ensure the feedback index exists, create if not.

        Returns:
            True if index exists or was created successfully.
        """
        # Check if alias exists
        if self.es.indices.exists_alias(name=self.index):
            return True

        # Create versioned index
        env = search_settings.es_env
        versioned_index = f"{FEEDBACK_INDEX_PREFIX}_{env}_v1"

        if not self.es.indices.exists(index=versioned_index):
            self.es.indices.create(
                index=versioned_index,
                body={
                    "settings": get_index_settings(),
                    "mappings": get_feedback_mapping(),
                },
            )
            logger.info(f"Created feedback index: {versioned_index}")

        # Create alias pointing to the versioned index
        if not self.es.indices.exists_alias(name=self.index):
            self.es.indices.put_alias(index=versioned_index, name=self.index)
            logger.info(f"Created alias {self.index} -> {versioned_index}")

        return True

    def save_feedback(self, feedback: Feedback) -> str:
        """Save feedback to ES.

        Args:
            feedback: Feedback to save.

        Returns:
            Document ID of the saved feedback.
        """
        doc_id = f"{feedback.session_id}:{feedback.turn_id}"
        self.es.index(
            index=self.index,
            id=doc_id,
            body=feedback.to_dict(),
            refresh=True,
        )
        logger.debug(f"Saved feedback {doc_id}")
        return doc_id

    def get_feedback(self, session_id: str, turn_id: int) -> Feedback | None:
        """Get feedback for a specific turn.

        Args:
            session_id: Session ID.
            turn_id: Turn number within session.

        Returns:
            Feedback if found, None otherwise.
        """
        doc_id = f"{session_id}:{turn_id}"
        try:
            result = self.es.get(
                index=self.index,
                id=doc_id,
            )
            return Feedback.from_dict(result["_source"])
        except NotFoundError:
            return None

    def list_feedback(
        self,
        limit: int = 50,
        offset: int = 0,
        rating: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> tuple[list[Feedback], int]:
        """List feedback with filtering.

        Args:
            limit: Maximum number of feedback to return.
            offset: Number of feedback to skip.
            rating: Filter by rating ("up" or "down").
            min_score: Filter by minimum average score.
            max_score: Filter by maximum average score.

        Returns:
            Tuple of (list of Feedback, total count).
        """
        must_clauses = []

        if rating:
            must_clauses.append({"term": {"rating": rating}})

        if min_score is not None or max_score is not None:
            range_clause: dict[str, Any] = {}
            if min_score is not None:
                range_clause["gte"] = min_score
            if max_score is not None:
                range_clause["lte"] = max_score
            must_clauses.append({"range": {"avg_score": range_clause}})

        query: dict[str, Any] = {
            "size": limit,
            "from": offset,
            "sort": [{"ts": "desc"}],
        }

        if must_clauses:
            query["query"] = {"bool": {"must": must_clauses}}
        else:
            query["query"] = {"match_all": {}}

        try:
            result = self.es.search(index=self.index, body=query)
            hits = result.get("hits", {})
            total = hits.get("total", {}).get("value", 0)
            items = [Feedback.from_dict(hit["_source"]) for hit in hits.get("hits", [])]
            return items, total
        except Exception as e:
            logger.error(f"Failed to list feedback: {e}")
            return [], 0

    def export_for_finetuning(
        self,
        min_score: float = 3.0,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Export feedback data for LLM fine-tuning.

        Args:
            min_score: Minimum average score to include.
            limit: Maximum number of records to export.

        Returns:
            List of dicts with user_text, assistant_text, and scores.
        """
        query = {
            "size": limit,
            "query": {
                "range": {"avg_score": {"gte": min_score}}
            },
            "sort": [{"avg_score": "desc"}, {"ts": "desc"}],
            "_source": [
                "user_text",
                "assistant_text",
                "accuracy",
                "completeness",
                "relevance",
                "avg_score",
                "rating",
                "comment",
                "ts",
            ],
        }

        try:
            result = self.es.search(index=self.index, body=query)
            hits = result.get("hits", {}).get("hits", [])
            return [hit["_source"] for hit in hits]
        except Exception as e:
            logger.error(f"Failed to export feedback: {e}")
            return []

    def get_statistics(self) -> dict[str, Any]:
        """Get feedback statistics.

        Returns:
            Dict with total count, average scores, and rating distribution.
        """
        query = {
            "size": 0,
            "aggs": {
                "total_count": {"value_count": {"field": "session_id"}},
                "avg_accuracy": {"avg": {"field": "accuracy"}},
                "avg_completeness": {"avg": {"field": "completeness"}},
                "avg_relevance": {"avg": {"field": "relevance"}},
                "avg_score": {"avg": {"field": "avg_score"}},
                "rating_distribution": {
                    "terms": {"field": "rating"}
                },
            },
        }

        try:
            result = self.es.search(index=self.index, body=query)
            aggs = result.get("aggregations", {})
            return {
                "total_count": int(aggs.get("total_count", {}).get("value", 0)),
                "avg_accuracy": aggs.get("avg_accuracy", {}).get("value"),
                "avg_completeness": aggs.get("avg_completeness", {}).get("value"),
                "avg_relevance": aggs.get("avg_relevance", {}).get("value"),
                "avg_score": aggs.get("avg_score", {}).get("value"),
                "rating_distribution": {
                    bucket["key"]: bucket["doc_count"]
                    for bucket in aggs.get("rating_distribution", {}).get("buckets", [])
                },
            }
        except Exception as e:
            logger.error(f"Failed to get feedback statistics: {e}")
            return {}
