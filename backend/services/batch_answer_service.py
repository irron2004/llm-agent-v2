"""Elasticsearch-backed batch answer service.

Provides storage and retrieval of batch answer generation runs and results.
Used for evaluating retrieval quality with answer generation.

Index naming convention:
    - Runs: batch_answer_runs_{env}_v{version}
    - Results: batch_answer_results_{env}_v{version}

Usage:
    # Create from settings
    svc = BatchAnswerService.from_settings()

    # Create a new run
    run = svc.create_run(name="Test Run", source_run_id="retrieval_run_123", source_config={...})

    # Save result
    result = BatchAnswerResult(run_id=run.run_id, question_id="q1", ...)
    svc.save_result(result)

    # Update rating
    svc.save_rating(result_id="result_123", rating=4, comment="Good answer")
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from elasticsearch import Elasticsearch, NotFoundError

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch.mappings import (
    get_batch_answer_runs_mapping,
    get_batch_answer_results_mapping,
    get_index_settings,
)

logger = logging.getLogger(__name__)

# Index naming
BATCH_ANSWER_RUNS_INDEX_PREFIX = "batch_answer_runs"
BATCH_ANSWER_RESULTS_INDEX_PREFIX = "batch_answer_results"
BATCH_ANSWER_SCHEMA_VERSION = "v1"

# Status types
RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
ResultStatus = Literal["pending", "completed", "failed"]


@dataclass
class SearchResultItem:
    """Search result item in batch answer result."""

    rank: int
    doc_id: str
    score: float
    title: str = ""
    snippet: str = ""
    content: str = ""  # Full content for LLM context
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    device_name: Optional[str] = None
    doc_type: Optional[str] = None
    expanded_pages: Optional[list[int]] = None
    expanded_page_urls: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "score": self.score,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "device_name": self.device_name,
            "doc_type": self.doc_type,
            "expanded_pages": self.expanded_pages,
            "expanded_page_urls": self.expanded_page_urls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResultItem":
        return cls(
            rank=data.get("rank", 0),
            doc_id=data.get("doc_id", ""),
            score=data.get("score", 0.0),
            title=data.get("title", ""),
            snippet=data.get("snippet", ""),
            content=data.get("content", ""),
            chunk_id=data.get("chunk_id"),
            page=data.get("page"),
            device_name=data.get("device_name"),
            doc_type=data.get("doc_type"),
            expanded_pages=data.get("expanded_pages"),
            expanded_page_urls=data.get("expanded_page_urls"),
        )


@dataclass
class RetrievalMetrics:
    """Retrieval metrics for a single question."""

    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    reciprocal_rank: Optional[float] = None
    first_relevant_rank: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "reciprocal_rank": self.reciprocal_rank,
            "first_relevant_rank": self.first_relevant_rank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalMetrics":
        return cls(
            hit_at_1=data.get("hit_at_1", False),
            hit_at_3=data.get("hit_at_3", False),
            hit_at_5=data.get("hit_at_5", False),
            hit_at_10=data.get("hit_at_10", False),
            reciprocal_rank=data.get("reciprocal_rank"),
            first_relevant_rank=data.get("first_relevant_rank"),
        )


@dataclass
class RunProgress:
    """Progress of a batch answer run."""

    total: int = 0
    completed: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunProgress":
        return cls(
            total=data.get("total", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
        )


@dataclass
class RunMetrics:
    """Aggregated metrics for a batch answer run."""

    avg_rating: Optional[float] = None
    rating_count: int = 0
    avg_latency_ms: Optional[float] = None
    total_tokens: int = 0
    # Hit@K ratios
    hit_at_1_ratio: Optional[float] = None
    hit_at_3_ratio: Optional[float] = None
    hit_at_5_ratio: Optional[float] = None
    mrr: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_rating": self.avg_rating,
            "rating_count": self.rating_count,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "hit_at_1_ratio": self.hit_at_1_ratio,
            "hit_at_3_ratio": self.hit_at_3_ratio,
            "hit_at_5_ratio": self.hit_at_5_ratio,
            "mrr": self.mrr,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetrics":
        return cls(
            avg_rating=data.get("avg_rating"),
            rating_count=data.get("rating_count", 0),
            avg_latency_ms=data.get("avg_latency_ms"),
            total_tokens=data.get("total_tokens", 0),
            hit_at_1_ratio=data.get("hit_at_1_ratio"),
            hit_at_3_ratio=data.get("hit_at_3_ratio"),
            hit_at_5_ratio=data.get("hit_at_5_ratio"),
            mrr=data.get("mrr"),
        )


@dataclass
class BatchAnswerRun:
    """Batch answer generation run metadata."""

    run_id: str
    status: RunStatus = "pending"
    name: Optional[str] = None
    description: Optional[str] = None
    # Source
    source_type: str = "retrieval_test"
    source_run_id: Optional[str] = None
    source_config: Optional[dict[str, Any]] = None  # Search config snapshot
    # LLM config
    llm_config: Optional[dict[str, Any]] = None
    # Progress and metrics
    progress: RunProgress = field(default_factory=RunProgress)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    error_message: Optional[str] = None
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        return {
            "run_id": self.run_id,
            "status": self.status,
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type,
            "source_run_id": self.source_run_id,
            "source_config": self.source_config,
            "llm_config": self.llm_config,
            "progress": self.progress.to_dict(),
            "metrics": self.metrics.to_dict(),
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else now,
            "updated_at": now,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchAnswerRun":
        return cls(
            run_id=data.get("run_id", ""),
            status=data.get("status", "pending"),
            name=data.get("name"),
            description=data.get("description"),
            source_type=data.get("source_type", "retrieval_test"),
            source_run_id=data.get("source_run_id"),
            source_config=data.get("source_config"),
            llm_config=data.get("llm_config"),
            progress=RunProgress.from_dict(data.get("progress", {})),
            metrics=RunMetrics.from_dict(data.get("metrics", {})),
            error_message=data.get("error_message"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


@dataclass
class BatchAnswerResult:
    """Individual answer result within a batch run."""

    result_id: str
    run_id: str
    question_id: str
    question: str
    status: ResultStatus = "pending"
    # Answer
    answer: str = ""
    reasoning: Optional[str] = None
    # Search results
    search_results: list[SearchResultItem] = field(default_factory=list)
    search_result_count: int = 0
    # Ground truth
    ground_truth_doc_ids: list[str] = field(default_factory=list)
    category: Optional[str] = None
    # Metrics
    retrieval_metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    latency_ms: Optional[int] = None
    token_count: Optional[dict[str, int]] = None  # {"input": N, "output": M}
    # Human evaluation
    rating: Optional[int] = None  # 1-5
    rating_comment: Optional[str] = None
    rated_by: Optional[str] = None
    rated_at: Optional[datetime] = None
    # Error
    error_message: Optional[str] = None
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        return {
            "result_id": self.result_id,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "question": self.question,
            "status": self.status,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "search_results": [r.to_dict() for r in self.search_results],
            "search_result_count": len(self.search_results),
            "ground_truth_doc_ids": self.ground_truth_doc_ids,
            "category": self.category,
            "retrieval_metrics": self.retrieval_metrics.to_dict(),
            "latency_ms": self.latency_ms,
            "token_count": self.token_count,
            "rating": self.rating,
            "rating_comment": self.rating_comment,
            "rated_by": self.rated_by,
            "rated_at": self.rated_at.isoformat() if self.rated_at else None,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else now,
            "updated_at": now,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchAnswerResult":
        search_results = [
            SearchResultItem.from_dict(r) for r in data.get("search_results", [])
        ]
        return cls(
            result_id=data.get("result_id", ""),
            run_id=data.get("run_id", ""),
            question_id=data.get("question_id", ""),
            question=data.get("question", ""),
            status=data.get("status", "pending"),
            answer=data.get("answer", ""),
            reasoning=data.get("reasoning"),
            search_results=search_results,
            search_result_count=data.get("search_result_count", len(search_results)),
            ground_truth_doc_ids=data.get("ground_truth_doc_ids", []),
            category=data.get("category"),
            retrieval_metrics=RetrievalMetrics.from_dict(data.get("retrieval_metrics", {})),
            latency_ms=data.get("latency_ms"),
            token_count=data.get("token_count"),
            rating=data.get("rating"),
            rating_comment=data.get("rating_comment"),
            rated_by=data.get("rated_by"),
            rated_at=datetime.fromisoformat(data["rated_at"]) if data.get("rated_at") else None,
            error_message=data.get("error_message"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


class BatchAnswerService:
    """Elasticsearch-backed batch answer storage and retrieval."""

    def __init__(
        self,
        *,
        es_client: Elasticsearch,
        runs_index: str,
        results_index: str,
    ) -> None:
        """Initialize batch answer service.

        Args:
            es_client: Elasticsearch client instance.
            runs_index: Index name/alias for batch answer runs.
            results_index: Index name/alias for batch answer results.
        """
        self.es = es_client
        self.runs_index = runs_index
        self.results_index = results_index

    @classmethod
    def from_settings(
        cls,
        *,
        es_client: Elasticsearch | None = None,
        env: str | None = None,
    ) -> "BatchAnswerService":
        """Create BatchAnswerService from global settings.

        Args:
            es_client: Pre-configured ES client. Created from settings if None.
            env: Environment name. Defaults to search_settings.es_env.

        Returns:
            Configured BatchAnswerService instance.
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
        runs_index = f"{BATCH_ANSWER_RUNS_INDEX_PREFIX}_{env}_current"
        results_index = f"{BATCH_ANSWER_RESULTS_INDEX_PREFIX}_{env}_current"

        return cls(es_client=es_client, runs_index=runs_index, results_index=results_index)

    def ensure_indices(self) -> bool:
        """Ensure both indices exist, create if not.

        Returns:
            True if indices exist or were created successfully.
        """
        env = search_settings.es_env

        # Runs index
        if not self.es.indices.exists_alias(name=self.runs_index):
            versioned_runs_index = f"{BATCH_ANSWER_RUNS_INDEX_PREFIX}_{env}_v1"
            if not self.es.indices.exists(index=versioned_runs_index):
                self.es.indices.create(
                    index=versioned_runs_index,
                    body={
                        "settings": get_index_settings(),
                        "mappings": get_batch_answer_runs_mapping(),
                    },
                )
                logger.info(f"Created batch answer runs index: {versioned_runs_index}")

            if not self.es.indices.exists_alias(name=self.runs_index):
                self.es.indices.put_alias(index=versioned_runs_index, name=self.runs_index)
                logger.info(f"Created alias {self.runs_index} -> {versioned_runs_index}")

        # Results index
        if not self.es.indices.exists_alias(name=self.results_index):
            versioned_results_index = f"{BATCH_ANSWER_RESULTS_INDEX_PREFIX}_{env}_v1"
            if not self.es.indices.exists(index=versioned_results_index):
                self.es.indices.create(
                    index=versioned_results_index,
                    body={
                        "settings": get_index_settings(),
                        "mappings": get_batch_answer_results_mapping(),
                    },
                )
                logger.info(f"Created batch answer results index: {versioned_results_index}")

            if not self.es.indices.exists_alias(name=self.results_index):
                self.es.indices.put_alias(index=versioned_results_index, name=self.results_index)
                logger.info(f"Created alias {self.results_index} -> {versioned_results_index}")

        return True

    # =========================================================================
    # Run CRUD
    # =========================================================================

    def create_run(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        source_run_id: str | None = None,
        source_config: dict[str, Any] | None = None,
        llm_config: dict[str, Any] | None = None,
        question_count: int = 0,
    ) -> BatchAnswerRun:
        """Create a new batch answer run.

        Args:
            name: User-defined name for the run.
            description: Optional description.
            source_run_id: Reference to source retrieval test run.
            source_config: Search config snapshot from source run.
            llm_config: LLM configuration.
            question_count: Total number of questions to process.

        Returns:
            Created BatchAnswerRun.
        """
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()

        run = BatchAnswerRun(
            run_id=run_id,
            status="pending",
            name=name,
            description=description,
            source_run_id=source_run_id,
            source_config=source_config,
            llm_config=llm_config,
            progress=RunProgress(total=question_count, completed=0, failed=0),
            created_at=now,
            updated_at=now,
        )

        self.es.index(
            index=self.runs_index,
            id=run_id,
            body=run.to_dict(),
            refresh=True,
        )
        logger.info(f"Created batch answer run: {run_id}")
        return run

    def get_run(self, run_id: str) -> BatchAnswerRun | None:
        """Get a batch answer run by ID.

        Args:
            run_id: Run ID.

        Returns:
            BatchAnswerRun if found, None otherwise.
        """
        try:
            result = self.es.get(index=self.runs_index, id=run_id)
            return BatchAnswerRun.from_dict(result["_source"])
        except NotFoundError:
            return None

    def list_runs(
        self,
        limit: int = 50,
        offset: int = 0,
        status: RunStatus | None = None,
    ) -> tuple[list[BatchAnswerRun], int]:
        """List batch answer runs.

        Args:
            limit: Maximum number of runs to return.
            offset: Number of runs to skip.
            status: Filter by status.

        Returns:
            Tuple of (list of BatchAnswerRun, total count).
        """
        query_body: dict[str, Any] = {
            "size": limit,
            "from": offset,
            "sort": [{"created_at": "desc"}],
        }

        if status:
            query_body["query"] = {"term": {"status": status}}
        else:
            query_body["query"] = {"match_all": {}}

        try:
            result = self.es.search(index=self.runs_index, body=query_body)
            hits = result.get("hits", {})
            total = hits.get("total", {}).get("value", 0)
            items = [BatchAnswerRun.from_dict(hit["_source"]) for hit in hits.get("hits", [])]
            return items, total
        except Exception as e:
            logger.error(f"Failed to list batch answer runs: {e}")
            return [], 0

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        progress: RunProgress | None = None,
        metrics: RunMetrics | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> bool:
        """Update a batch answer run.

        Args:
            run_id: Run ID.
            status: New status.
            progress: New progress.
            metrics: New metrics.
            error_message: Error message (for failed status).
            started_at: Start timestamp.
            completed_at: Completion timestamp.

        Returns:
            True if updated, False if not found.
        """
        doc: dict[str, Any] = {"updated_at": datetime.utcnow().isoformat()}

        if status is not None:
            doc["status"] = status
        if progress is not None:
            doc["progress"] = progress.to_dict()
        if metrics is not None:
            doc["metrics"] = metrics.to_dict()
        if error_message is not None:
            doc["error_message"] = error_message
        if started_at is not None:
            doc["started_at"] = started_at.isoformat()
        if completed_at is not None:
            doc["completed_at"] = completed_at.isoformat()

        try:
            self.es.update(
                index=self.runs_index,
                id=run_id,
                body={"doc": doc},
                refresh=True,
            )
            return True
        except NotFoundError:
            return False

    def delete_run(self, run_id: str) -> bool:
        """Delete a batch answer run and all its results.

        Args:
            run_id: Run ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        # Delete all results for this run
        try:
            self.es.delete_by_query(
                index=self.results_index,
                body={"query": {"term": {"run_id": run_id}}},
                refresh=True,
            )
        except Exception as e:
            logger.warning(f"Failed to delete results for run {run_id}: {e}")

        # Delete the run
        try:
            self.es.delete(index=self.runs_index, id=run_id, refresh=True)
            logger.info(f"Deleted batch answer run: {run_id}")
            return True
        except NotFoundError:
            return False

    # =========================================================================
    # Result CRUD
    # =========================================================================

    def save_result(self, result: BatchAnswerResult) -> str:
        """Save a batch answer result.

        Args:
            result: Result to save.

        Returns:
            Result ID.
        """
        self.es.index(
            index=self.results_index,
            id=result.result_id,
            body=result.to_dict(),
            refresh=True,
        )
        logger.debug(f"Saved batch answer result: {result.result_id}")
        return result.result_id

    def get_result(self, result_id: str) -> BatchAnswerResult | None:
        """Get a batch answer result by ID.

        Args:
            result_id: Result ID.

        Returns:
            BatchAnswerResult if found, None otherwise.
        """
        try:
            result = self.es.get(index=self.results_index, id=result_id)
            return BatchAnswerResult.from_dict(result["_source"])
        except NotFoundError:
            return None

    def list_results(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
        status: ResultStatus | None = None,
    ) -> tuple[list[BatchAnswerResult], int]:
        """List results for a run.

        Args:
            run_id: Run ID.
            limit: Maximum number of results.
            offset: Offset for pagination.
            status: Filter by status.

        Returns:
            Tuple of (list of BatchAnswerResult, total count).
        """
        must_clauses = [{"term": {"run_id": run_id}}]
        if status:
            must_clauses.append({"term": {"status": status}})

        query_body: dict[str, Any] = {
            "size": limit,
            "from": offset,
            "query": {"bool": {"must": must_clauses}},
            "sort": [{"created_at": "asc"}],
        }

        try:
            result = self.es.search(index=self.results_index, body=query_body)
            hits = result.get("hits", {})
            total = hits.get("total", {}).get("value", 0)
            items = [BatchAnswerResult.from_dict(hit["_source"]) for hit in hits.get("hits", [])]
            return items, total
        except Exception as e:
            logger.error(f"Failed to list batch answer results: {e}")
            return [], 0

    def get_next_pending_result(self, run_id: str) -> BatchAnswerResult | None:
        """Get the next pending result for a run.

        Args:
            run_id: Run ID.

        Returns:
            Next pending BatchAnswerResult, or None if all completed.
        """
        query_body: dict[str, Any] = {
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"run_id": run_id}},
                        {"term": {"status": "pending"}},
                    ]
                }
            },
            "sort": [{"created_at": "asc"}],
        }

        try:
            result = self.es.search(index=self.results_index, body=query_body)
            hits = result.get("hits", {}).get("hits", [])
            if hits:
                return BatchAnswerResult.from_dict(hits[0]["_source"])
            return None
        except Exception as e:
            logger.error(f"Failed to get next pending result: {e}")
            return None

    def save_rating(
        self,
        result_id: str,
        rating: int,
        comment: str | None = None,
        rated_by: str | None = None,
    ) -> bool:
        """Save rating for a result.

        Args:
            result_id: Result ID.
            rating: Rating value (1-5).
            comment: Optional comment.
            rated_by: Reviewer name.

        Returns:
            True if updated, False if not found.
        """
        doc = {
            "rating": rating,
            "rating_comment": comment,
            "rated_by": rated_by,
            "rated_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        try:
            self.es.update(
                index=self.results_index,
                id=result_id,
                body={"doc": doc},
                refresh=True,
            )
            return True
        except NotFoundError:
            return False

    # =========================================================================
    # Metrics Aggregation
    # =========================================================================

    def calculate_run_metrics(self, run_id: str) -> RunMetrics:
        """Calculate aggregated metrics for a run.

        Args:
            run_id: Run ID.

        Returns:
            Aggregated RunMetrics.
        """
        query_body = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"run_id": run_id}},
                        {"term": {"status": "completed"}},
                    ]
                }
            },
            "aggs": {
                "avg_rating": {"avg": {"field": "rating"}},
                "rating_count": {"value_count": {"field": "rating"}},
                "avg_latency": {"avg": {"field": "latency_ms"}},
                "total_tokens_input": {"sum": {"field": "token_count.input"}},
                "total_tokens_output": {"sum": {"field": "token_count.output"}},
                "hit_at_1_count": {
                    "filter": {"term": {"retrieval_metrics.hit_at_1": True}}
                },
                "hit_at_3_count": {
                    "filter": {"term": {"retrieval_metrics.hit_at_3": True}}
                },
                "hit_at_5_count": {
                    "filter": {"term": {"retrieval_metrics.hit_at_5": True}}
                },
                "avg_rr": {"avg": {"field": "retrieval_metrics.reciprocal_rank"}},
                "total_completed": {"value_count": {"field": "result_id"}},
            },
        }

        try:
            result = self.es.search(index=self.results_index, body=query_body)
            aggs = result.get("aggregations", {})
            total_completed = int(aggs.get("total_completed", {}).get("value", 0))

            metrics = RunMetrics(
                avg_rating=aggs.get("avg_rating", {}).get("value"),
                rating_count=int(aggs.get("rating_count", {}).get("value", 0)),
                avg_latency_ms=aggs.get("avg_latency", {}).get("value"),
                total_tokens=int(
                    (aggs.get("total_tokens_input", {}).get("value", 0) or 0) +
                    (aggs.get("total_tokens_output", {}).get("value", 0) or 0)
                ),
            )

            if total_completed > 0:
                metrics.hit_at_1_ratio = aggs.get("hit_at_1_count", {}).get("doc_count", 0) / total_completed
                metrics.hit_at_3_ratio = aggs.get("hit_at_3_count", {}).get("doc_count", 0) / total_completed
                metrics.hit_at_5_ratio = aggs.get("hit_at_5_count", {}).get("doc_count", 0) / total_completed
                metrics.mrr = aggs.get("avg_rr", {}).get("value")

            return metrics
        except Exception as e:
            logger.error(f"Failed to calculate run metrics: {e}")
            return RunMetrics()

    def update_run_metrics(self, run_id: str) -> bool:
        """Recalculate and update metrics for a run.

        Args:
            run_id: Run ID.

        Returns:
            True if updated successfully.
        """
        metrics = self.calculate_run_metrics(run_id)
        return self.update_run(run_id, metrics=metrics)
