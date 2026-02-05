"""Batch Answer Generation API.

Provides endpoints for batch answer generation from retrieval test results.
Answer-only mode: uses saved search results without performing new search.
"""

import logging
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from backend.api.dependencies import get_rag_agent
from backend.services.batch_answer_service import (
    BatchAnswerResult,
    BatchAnswerRun,
    BatchAnswerService,
    RetrievalMetrics,
    RunMetrics,
    RunProgress,
    SearchResultItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch-answer", tags=["Batch Answer"])


# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================


class SearchResultItemResponse(BaseModel):
    """Search result item in response."""

    model_config = ConfigDict(from_attributes=True)

    rank: int
    doc_id: str
    score: float
    title: str = ""
    snippet: str = ""
    content: str = ""
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    device_name: Optional[str] = None
    doc_type: Optional[str] = None
    expanded_pages: List[int] = Field(default_factory=list)
    expanded_page_urls: List[str] = Field(default_factory=list)


class RetrievalMetricsResponse(BaseModel):
    """Retrieval metrics in response."""

    model_config = ConfigDict(from_attributes=True)

    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    reciprocal_rank: Optional[float] = None
    first_relevant_rank: Optional[int] = None


class RunProgressResponse(BaseModel):
    """Run progress in response."""

    model_config = ConfigDict(from_attributes=True)

    total: int = 0
    completed: int = 0
    failed: int = 0


class RunMetricsResponse(BaseModel):
    """Run metrics in response."""

    model_config = ConfigDict(from_attributes=True)

    avg_rating: Optional[float] = None
    rating_count: int = 0
    avg_latency_ms: Optional[float] = None
    total_tokens: int = 0
    hit_at_1_ratio: Optional[float] = None
    hit_at_3_ratio: Optional[float] = None
    hit_at_5_ratio: Optional[float] = None
    mrr: Optional[float] = None


class SourceConfigResponse(BaseModel):
    """Source search config in response."""

    dense_weight: Optional[float] = None
    sparse_weight: Optional[float] = None
    use_rrf: Optional[bool] = None
    rrf_k: Optional[int] = None
    rerank: Optional[bool] = None
    rerank_top_k: Optional[int] = None
    top_k: Optional[int] = None


# ============================================================================
# Run API Models
# ============================================================================


class CreateRunRequest(BaseModel):
    """Request to create a new batch answer run."""

    name: Optional[str] = Field(None, description="User-defined name for the run")
    description: Optional[str] = Field(None, description="Optional description")
    source_run_id: Optional[str] = Field(
        None, description="Reference to retrieval test run (currently optional, use questions directly)"
    )
    source_config: Optional[Dict[str, Any]] = Field(
        None, description="Search config snapshot (copied from retrieval test)"
    )
    questions: List[Dict[str, Any]] = Field(
        ..., description="Questions to process (id, question, ground_truth_doc_ids, search_results, category)"
    )


class RunResponse(BaseModel):
    """Batch answer run response."""

    model_config = ConfigDict(from_attributes=True)

    run_id: str
    status: str
    name: Optional[str] = None
    description: Optional[str] = None
    source_type: str = "retrieval_test"
    source_run_id: Optional[str] = None
    source_config: Optional[Dict[str, Any]] = None
    progress: RunProgressResponse
    metrics: RunMetricsResponse
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RunListResponse(BaseModel):
    """List of batch answer runs."""

    items: List[RunResponse]
    total: int


# ============================================================================
# Result API Models
# ============================================================================


class ResultResponse(BaseModel):
    """Batch answer result response."""

    model_config = ConfigDict(from_attributes=True)

    result_id: str
    run_id: str
    question_id: str
    question: str
    status: str
    answer: str = ""
    reasoning: Optional[str] = None
    search_results: List[SearchResultItemResponse] = Field(default_factory=list)
    search_result_count: int = 0
    ground_truth_doc_ids: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    retrieval_metrics: RetrievalMetricsResponse
    latency_ms: Optional[int] = None
    token_count: Optional[Dict[str, int]] = None
    rating: Optional[int] = None
    rating_comment: Optional[str] = None
    rated_by: Optional[str] = None
    rated_at: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ResultListResponse(BaseModel):
    """List of batch answer results."""

    items: List[ResultResponse]
    total: int


class ExecuteNextResponse(BaseModel):
    """Response from execute-next endpoint."""

    result_id: str
    question_id: str
    question: str
    answer: str
    reasoning: Optional[str] = None
    search_results: List[SearchResultItemResponse] = Field(default_factory=list)
    metrics: RetrievalMetricsResponse
    progress: RunProgressResponse
    status: str = "completed"
    error_message: Optional[str] = None


class RatingRequest(BaseModel):
    """Request to save rating for a result."""

    rating: int = Field(..., ge=1, le=5, description="Rating value (1-5)")
    comment: Optional[str] = Field(None, description="Optional comment")
    rated_by: Optional[str] = Field(None, description="Reviewer name")


# ============================================================================
# Service Dependency
# ============================================================================


def get_batch_answer_service() -> BatchAnswerService:
    """Get batch answer service instance."""
    svc = BatchAnswerService.from_settings()
    svc.ensure_indices()
    return svc


# ============================================================================
# Helper Functions
# ============================================================================


def _run_to_response(run: BatchAnswerRun) -> RunResponse:
    """Convert BatchAnswerRun to RunResponse."""
    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        name=run.name,
        description=run.description,
        source_type=run.source_type,
        source_run_id=run.source_run_id,
        source_config=run.source_config,
        progress=RunProgressResponse(
            total=run.progress.total,
            completed=run.progress.completed,
            failed=run.progress.failed,
        ),
        metrics=RunMetricsResponse(
            avg_rating=run.metrics.avg_rating,
            rating_count=run.metrics.rating_count,
            avg_latency_ms=run.metrics.avg_latency_ms,
            total_tokens=run.metrics.total_tokens,
            hit_at_1_ratio=run.metrics.hit_at_1_ratio,
            hit_at_3_ratio=run.metrics.hit_at_3_ratio,
            hit_at_5_ratio=run.metrics.hit_at_5_ratio,
            mrr=run.metrics.mrr,
        ),
        error_message=run.error_message,
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        created_at=run.created_at.isoformat() if run.created_at else None,
        updated_at=run.updated_at.isoformat() if run.updated_at else None,
    )


def _result_to_response(result: BatchAnswerResult) -> ResultResponse:
    """Convert BatchAnswerResult to ResultResponse."""
    return ResultResponse(
        result_id=result.result_id,
        run_id=result.run_id,
        question_id=result.question_id,
        question=result.question,
        status=result.status,
        answer=result.answer,
        reasoning=result.reasoning,
        search_results=[
            SearchResultItemResponse(
                rank=r.rank,
                doc_id=r.doc_id,
                score=r.score,
                title=r.title,
                snippet=r.snippet,
                content=r.content,
                chunk_id=r.chunk_id,
                page=r.page,
                device_name=r.device_name,
                doc_type=r.doc_type,
                expanded_pages=r.expanded_pages or [],
                expanded_page_urls=r.expanded_page_urls or [],
            )
            for r in result.search_results
        ],
        search_result_count=result.search_result_count,
        ground_truth_doc_ids=result.ground_truth_doc_ids,
        category=result.category,
        retrieval_metrics=RetrievalMetricsResponse(
            hit_at_1=result.retrieval_metrics.hit_at_1,
            hit_at_3=result.retrieval_metrics.hit_at_3,
            hit_at_5=result.retrieval_metrics.hit_at_5,
            hit_at_10=result.retrieval_metrics.hit_at_10,
            reciprocal_rank=result.retrieval_metrics.reciprocal_rank,
            first_relevant_rank=result.retrieval_metrics.first_relevant_rank,
        ),
        latency_ms=result.latency_ms,
        token_count=result.token_count,
        rating=result.rating,
        rating_comment=result.rating_comment,
        rated_by=result.rated_by,
        rated_at=result.rated_at.isoformat() if result.rated_at else None,
        error_message=result.error_message,
        created_at=result.created_at.isoformat() if result.created_at else None,
        updated_at=result.updated_at.isoformat() if result.updated_at else None,
    )


def _calculate_retrieval_metrics(
    search_results: List[Dict[str, Any]],
    ground_truth_doc_ids: List[str],
) -> RetrievalMetrics:
    """Calculate retrieval metrics from search results and ground truth."""
    if not ground_truth_doc_ids:
        return RetrievalMetrics()

    retrieved_doc_ids = [r.get("doc_id", r.get("id", "")) for r in search_results]
    ground_truth_set = set(ground_truth_doc_ids)

    first_relevant_rank = None
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in ground_truth_set:
            first_relevant_rank = i + 1  # 1-based
            break

    return RetrievalMetrics(
        hit_at_1=first_relevant_rank is not None and first_relevant_rank <= 1,
        hit_at_3=first_relevant_rank is not None and first_relevant_rank <= 3,
        hit_at_5=first_relevant_rank is not None and first_relevant_rank <= 5,
        hit_at_10=first_relevant_rank is not None and first_relevant_rank <= 10,
        reciprocal_rank=1.0 / first_relevant_rank if first_relevant_rank else None,
        first_relevant_rank=first_relevant_rank,
    )


# ============================================================================
# Run Endpoints
# ============================================================================


@router.post("/runs", response_model=RunResponse)
async def create_run(
    request: CreateRunRequest,
    svc: BatchAnswerService = Depends(get_batch_answer_service),
):
    """Create a new batch answer run.

    Creates a run with pending results for each question.
    Use execute-next to process questions one by one.
    """
    try:
        # Create the run
        run = svc.create_run(
            name=request.name,
            description=request.description,
            source_run_id=request.source_run_id,
            source_config=request.source_config,
            question_count=len(request.questions),
        )

        # Create pending results for each question
        for q in request.questions:
            search_results_raw = q.get("search_results", [])
            search_results = [
                SearchResultItem(
                    rank=r.get("rank", i + 1),
                    doc_id=r.get("doc_id", r.get("id", "")),
                    score=r.get("score", 0.0),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    content=r.get("content", ""),
                    chunk_id=r.get("chunk_id"),
                    page=r.get("page"),
                    device_name=r.get("device_name"),
                    doc_type=r.get("doc_type"),
                    expanded_pages=r.get("expanded_pages"),
                    expanded_page_urls=r.get("expanded_page_urls"),
                )
                for i, r in enumerate(search_results_raw)
            ]

            ground_truth = q.get("ground_truth_doc_ids", q.get("groundTruthDocIds", []))
            metrics = _calculate_retrieval_metrics(search_results_raw, ground_truth)

            result = BatchAnswerResult(
                result_id=str(uuid.uuid4()),
                run_id=run.run_id,
                question_id=q.get("id", str(uuid.uuid4())),
                question=q.get("question", ""),
                status="pending",
                search_results=search_results,
                ground_truth_doc_ids=ground_truth,
                category=q.get("category"),
                retrieval_metrics=metrics,
            )
            svc.save_result(result)

        return _run_to_response(run)

    except Exception as e:
        logger.exception("Failed to create batch answer run")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    svc: BatchAnswerService = Depends(get_batch_answer_service),
):
    """List batch answer runs."""
    try:
        runs, total = svc.list_runs(limit=limit, offset=offset, status=status)
        return RunListResponse(
            items=[_run_to_response(r) for r in runs],
            total=total,
        )
    except Exception as e:
        logger.exception("Failed to list batch answer runs")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    svc: BatchAnswerService = Depends(get_batch_answer_service),
):
    """Get a batch answer run by ID."""
    run = svc.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return _run_to_response(run)


@router.delete("/runs/{run_id}")
async def delete_run(
    run_id: str,
    svc: BatchAnswerService = Depends(get_batch_answer_service),
):
    """Delete a batch answer run and all its results."""
    if not svc.delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return {"status": "deleted", "run_id": run_id}


# ============================================================================
# Execute Endpoint
# ============================================================================


@router.post("/runs/{run_id}/execute-next", response_model=ExecuteNextResponse)
async def execute_next(
    run_id: str,
    svc: BatchAnswerService = Depends(get_batch_answer_service),
    rag_agent=Depends(get_rag_agent),
):
    """Execute answer generation for the next pending question.

    Answer-only mode: uses saved search results from the pending result,
    does not perform new search.
    """
    # Get run
    run = svc.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    # Update run status to running if pending
    if run.status == "pending":
        svc.update_run(run_id, status="running", started_at=datetime.utcnow())

    # Get next pending result
    pending_result = svc.get_next_pending_result(run_id)
    if not pending_result:
        # No more pending - mark as completed
        svc.update_run(run_id, status="completed", completed_at=datetime.utcnow())
        svc.update_run_metrics(run_id)
        raise HTTPException(status_code=404, detail="No pending questions")

    # Generate answer using LLM
    start_time = time.time()
    try:
        # Build docs context from search results
        docs_for_llm = []
        for sr in pending_result.search_results:
            doc = {
                "id": sr.doc_id,
                "title": sr.title,
                "content": sr.content or sr.snippet,
                "snippet": sr.snippet,
                "page": sr.page,
                "score": sr.score,
                "device_name": sr.device_name,
                "doc_type": sr.doc_type,
            }
            docs_for_llm.append(doc)

        # Call answer_only on the agent
        answer_result = rag_agent.answer_only(
            query=pending_result.question,
            docs=docs_for_llm,
        )

        answer = answer_result.get("answer", "")
        reasoning = answer_result.get("reasoning")
        token_count = answer_result.get("token_count")

        latency_ms = int((time.time() - start_time) * 1000)

        # Update result
        pending_result.status = "completed"
        pending_result.answer = answer
        pending_result.reasoning = reasoning
        pending_result.latency_ms = latency_ms
        pending_result.token_count = token_count
        svc.save_result(pending_result)

        # Update run progress
        run = svc.get_run(run_id)
        new_progress = RunProgress(
            total=run.progress.total,
            completed=run.progress.completed + 1,
            failed=run.progress.failed,
        )
        svc.update_run(run_id, progress=new_progress)

        # Check if all done
        if new_progress.completed + new_progress.failed >= new_progress.total:
            svc.update_run(run_id, status="completed", completed_at=datetime.utcnow())
            svc.update_run_metrics(run_id)

        return ExecuteNextResponse(
            result_id=pending_result.result_id,
            question_id=pending_result.question_id,
            question=pending_result.question,
            answer=answer,
            reasoning=reasoning,
            search_results=[
                SearchResultItemResponse(
                    rank=r.rank,
                    doc_id=r.doc_id,
                    score=r.score,
                    title=r.title,
                    snippet=r.snippet,
                    content=r.content,
                    chunk_id=r.chunk_id,
                    page=r.page,
                    device_name=r.device_name,
                    doc_type=r.doc_type,
                    expanded_pages=r.expanded_pages or [],
                    expanded_page_urls=r.expanded_page_urls or [],
                )
                for r in pending_result.search_results
            ],
            metrics=RetrievalMetricsResponse(
                hit_at_1=pending_result.retrieval_metrics.hit_at_1,
                hit_at_3=pending_result.retrieval_metrics.hit_at_3,
                hit_at_5=pending_result.retrieval_metrics.hit_at_5,
                hit_at_10=pending_result.retrieval_metrics.hit_at_10,
                reciprocal_rank=pending_result.retrieval_metrics.reciprocal_rank,
                first_relevant_rank=pending_result.retrieval_metrics.first_relevant_rank,
            ),
            progress=RunProgressResponse(
                total=new_progress.total,
                completed=new_progress.completed,
                failed=new_progress.failed,
            ),
            status="completed",
        )

    except Exception as e:
        logger.exception("Failed to generate answer for question %s", pending_result.question_id)

        latency_ms = int((time.time() - start_time) * 1000)

        # Mark result as failed
        pending_result.status = "failed"
        pending_result.error_message = str(e)
        pending_result.latency_ms = latency_ms
        svc.save_result(pending_result)

        # Update run progress
        run = svc.get_run(run_id)
        new_progress = RunProgress(
            total=run.progress.total,
            completed=run.progress.completed,
            failed=run.progress.failed + 1,
        )
        svc.update_run(run_id, progress=new_progress)

        # Check if all done
        if new_progress.completed + new_progress.failed >= new_progress.total:
            svc.update_run(run_id, status="completed", completed_at=datetime.utcnow())
            svc.update_run_metrics(run_id)

        return ExecuteNextResponse(
            result_id=pending_result.result_id,
            question_id=pending_result.question_id,
            question=pending_result.question,
            answer="",
            reasoning=None,
            search_results=[
                SearchResultItemResponse(
                    rank=r.rank,
                    doc_id=r.doc_id,
                    score=r.score,
                    title=r.title,
                    snippet=r.snippet,
                    content=r.content,
                    chunk_id=r.chunk_id,
                    page=r.page,
                    device_name=r.device_name,
                    doc_type=r.doc_type,
                    expanded_pages=r.expanded_pages or [],
                    expanded_page_urls=r.expanded_page_urls or [],
                )
                for r in pending_result.search_results
            ],
            metrics=RetrievalMetricsResponse(
                hit_at_1=pending_result.retrieval_metrics.hit_at_1,
                hit_at_3=pending_result.retrieval_metrics.hit_at_3,
                hit_at_5=pending_result.retrieval_metrics.hit_at_5,
                hit_at_10=pending_result.retrieval_metrics.hit_at_10,
                reciprocal_rank=pending_result.retrieval_metrics.reciprocal_rank,
                first_relevant_rank=pending_result.retrieval_metrics.first_relevant_rank,
            ),
            progress=RunProgressResponse(
                total=new_progress.total,
                completed=new_progress.completed,
                failed=new_progress.failed,
            ),
            status="failed",
            error_message=str(e),
        )


# ============================================================================
# Result Endpoints
# ============================================================================


@router.get("/runs/{run_id}/results", response_model=ResultListResponse)
async def list_results(
    run_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None),
    svc: BatchAnswerService = Depends(get_batch_answer_service),
):
    """List results for a run."""
    try:
        results, total = svc.list_results(run_id, limit=limit, offset=offset, status=status)
        return ResultListResponse(
            items=[_result_to_response(r) for r in results],
            total=total,
        )
    except Exception as e:
        logger.exception("Failed to list batch answer results")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/results/{result_id}", response_model=ResultResponse)
async def get_result(
    result_id: str,
    svc: BatchAnswerService = Depends(get_batch_answer_service),
):
    """Get a batch answer result by ID."""
    result = svc.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")
    return _result_to_response(result)


@router.put("/results/{result_id}/rating")
async def save_rating(
    result_id: str,
    request: RatingRequest,
    svc: BatchAnswerService = Depends(get_batch_answer_service),
):
    """Save rating for a result."""
    if not svc.save_rating(
        result_id=result_id,
        rating=request.rating,
        comment=request.comment,
        rated_by=request.rated_by,
    ):
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")

    # Get the result to find run_id for metrics update
    result = svc.get_result(result_id)
    if result:
        svc.update_run_metrics(result.run_id)

    return {"status": "saved", "result_id": result_id, "rating": request.rating}
