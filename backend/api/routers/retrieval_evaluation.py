"""Retrieval Evaluation API - Document relevance scoring for retrieval testing.

Provides endpoints for:
- Saving document relevance evaluations (1-5 stars)
- Getting evaluations for a specific turn
- Exporting evaluation data for retrieval testing
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import csv
import io
import json

from backend.services.retrieval_evaluation_service import (
    RetrievalEvaluationService,
    RetrievalEvaluation,
)

router = APIRouter(prefix="/retrieval-evaluation", tags=["Retrieval Evaluation"])

# Global service instance (initialized on startup)
_retrieval_evaluation_service: RetrievalEvaluationService | None = None


def get_retrieval_evaluation_service() -> RetrievalEvaluationService:
    """Get the retrieval evaluation service instance."""
    global _retrieval_evaluation_service
    if _retrieval_evaluation_service is None:
        _retrieval_evaluation_service = RetrievalEvaluationService.from_settings()
        _retrieval_evaluation_service.ensure_index()
    return _retrieval_evaluation_service


# --- Request/Response Models ---


class RetrievalEvaluationRequest(BaseModel):
    """Request to save a document relevance evaluation."""

    relevance_score: int = Field(..., ge=1, le=5, description="관련성 점수 (1-5)")
    reviewer_name: Optional[str] = Field(None, description="평가자 이름 (선택)")
    query: str = Field(..., description="원본 검색 쿼리")
    doc_rank: int = Field(..., ge=1, description="검색 결과 순위 (1-based)")
    doc_title: str = Field("", description="문서 제목")
    doc_snippet: str = Field("", description="문서 스니펫")
    message_id: Optional[str] = Field(None, description="프론트엔드 메시지 ID")
    chunk_id: Optional[str] = Field(None, description="청크 ID")
    retrieval_score: Optional[float] = Field(None, description="원본 검색 점수")
    filter_devices: Optional[List[str]] = Field(None, description="검색 필터 (장비)")
    filter_doc_types: Optional[List[str]] = Field(None, description="검색 필터 (문서 종류)")
    search_queries: Optional[List[str]] = Field(None, description="Multi-query expansion 결과")


class RetrievalEvaluationResponse(BaseModel):
    """Response for a document relevance evaluation."""

    session_id: str
    turn_id: int
    doc_id: str
    relevance_score: int
    is_relevant: bool
    query: str
    doc_rank: int
    doc_title: str = ""
    doc_snippet: str = ""
    message_id: Optional[str] = None
    chunk_id: Optional[str] = None
    retrieval_score: Optional[float] = None
    reviewer_name: Optional[str] = None
    filter_devices: Optional[List[str]] = None
    filter_doc_types: Optional[List[str]] = None
    search_queries: Optional[List[str]] = None
    ts: str


class RetrievalEvaluationListResponse(BaseModel):
    """Response for a list of evaluations."""

    items: List[RetrievalEvaluationResponse]
    total: int


class RetrievalEvaluationStatisticsResponse(BaseModel):
    """Response for evaluation statistics."""

    total_count: int
    avg_relevance: Optional[float] = None
    relevant_count: int
    relevant_ratio: float
    unique_queries: int
    relevance_distribution: dict


def _evaluation_to_response(evaluation: RetrievalEvaluation) -> RetrievalEvaluationResponse:
    """Convert RetrievalEvaluation dataclass to response model."""
    return RetrievalEvaluationResponse(
        session_id=evaluation.session_id,
        turn_id=evaluation.turn_id,
        doc_id=evaluation.doc_id,
        relevance_score=evaluation.relevance_score,
        is_relevant=evaluation.is_relevant,
        query=evaluation.query,
        doc_rank=evaluation.doc_rank,
        doc_title=evaluation.doc_title,
        doc_snippet=evaluation.doc_snippet,
        message_id=evaluation.message_id,
        chunk_id=evaluation.chunk_id,
        retrieval_score=evaluation.retrieval_score,
        reviewer_name=evaluation.reviewer_name,
        filter_devices=evaluation.filter_devices,
        filter_doc_types=evaluation.filter_doc_types,
        search_queries=evaluation.search_queries,
        ts=evaluation.ts.isoformat() if evaluation.ts else "",
    )


# --- Endpoints ---
# NOTE: Static routes MUST come BEFORE dynamic routes


@router.get("/statistics", response_model=RetrievalEvaluationStatisticsResponse)
async def get_statistics(
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Get retrieval evaluation statistics."""
    stats = service.get_statistics()
    return RetrievalEvaluationStatisticsResponse(
        total_count=stats.get("total_count", 0),
        avg_relevance=stats.get("avg_relevance"),
        relevant_count=stats.get("relevant_count", 0),
        relevant_ratio=stats.get("relevant_ratio", 0.0),
        unique_queries=stats.get("unique_queries", 0),
        relevance_distribution=stats.get("relevance_distribution", {}),
    )


@router.get("/export/json")
async def export_retrieval_test_json(
    min_relevance: int = 3,
    limit: int = 10000,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Export evaluation data as JSON for retrieval testing.

    Returns all evaluations, marking is_relevant based on relevance_score >= min_relevance.
    """
    data = service.export_for_retrieval_test(min_relevance=min_relevance, limit=limit)

    content = json.dumps(data, ensure_ascii=False, indent=2)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={
            "Content-Disposition": "attachment; filename=retrieval_evaluations.json"
        },
    )


@router.get("/export/csv")
async def export_retrieval_test_csv(
    min_relevance: int = 3,
    limit: int = 10000,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Export evaluation data as CSV for retrieval testing.

    Returns all evaluations in CSV format for spreadsheet analysis.
    """
    data = service.export_for_retrieval_test(min_relevance=min_relevance, limit=limit)

    # Create CSV content
    output = io.StringIO()
    if data:
        # Flatten filter arrays for CSV
        for row in data:
            if row.get("filter_devices"):
                row["filter_devices"] = ",".join(row["filter_devices"])
            if row.get("filter_doc_types"):
                row["filter_doc_types"] = ",".join(row["filter_doc_types"])
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    content = output.getvalue()
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=retrieval_evaluations.csv"
        },
    )


# Dynamic routes with path parameters - MUST come AFTER static routes


@router.post("/{session_id}/{turn_id}/{doc_id}", response_model=RetrievalEvaluationResponse)
async def save_evaluation(
    session_id: str,
    turn_id: int,
    doc_id: str,
    req: RetrievalEvaluationRequest,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Save a document relevance evaluation.

    Stores relevance score (1-5) for a specific document in a search result.
    """
    # Create evaluation object
    evaluation = RetrievalEvaluation(
        session_id=session_id,
        turn_id=turn_id,
        doc_id=doc_id,
        relevance_score=req.relevance_score,
        query=req.query,
        doc_rank=req.doc_rank,
        doc_title=req.doc_title,
        doc_snippet=req.doc_snippet,
        message_id=req.message_id,
        chunk_id=req.chunk_id,
        retrieval_score=req.retrieval_score,
        reviewer_name=req.reviewer_name,
        filter_devices=req.filter_devices,
        filter_doc_types=req.filter_doc_types,
        search_queries=req.search_queries,
    )

    service.save_evaluation(evaluation)

    # Retrieve and return saved evaluation
    saved = service.get_evaluation(session_id, turn_id, doc_id)
    if saved is None:
        raise HTTPException(status_code=500, detail="Failed to save evaluation")

    return _evaluation_to_response(saved)


@router.get("/{session_id}/{turn_id}/{doc_id}", response_model=RetrievalEvaluationResponse)
async def get_evaluation(
    session_id: str,
    turn_id: int,
    doc_id: str,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Get evaluation for a specific document."""
    evaluation = service.get_evaluation(session_id, turn_id, doc_id)
    if evaluation is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return _evaluation_to_response(evaluation)


@router.get("/{session_id}/{turn_id}", response_model=RetrievalEvaluationListResponse)
async def list_evaluations_for_turn(
    session_id: str,
    turn_id: int,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """List all evaluations for a specific turn."""
    items = service.list_evaluations_for_turn(session_id, turn_id)

    return RetrievalEvaluationListResponse(
        items=[_evaluation_to_response(e) for e in items],
        total=len(items),
    )
