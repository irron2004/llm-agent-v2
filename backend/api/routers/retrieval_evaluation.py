"""Retrieval Evaluation API - Query-unit relevance scoring for retrieval testing.

Provides endpoints for:
- Saving query-unit evaluations (batch save with multiple documents)
- Getting evaluations by query_id
- Listing all evaluations with pagination
- Exporting evaluation data for retrieval testing

Storage structure (query-unit):
    {
        "query_id": "sess1:turn1",  # PK (chat: session:turn, search: search:timestamp)
        "query": "원본 쿼리",
        "relevant_docs": ["doc_001", "doc_003"],      # Auto-generated (score >= 3)
        "irrelevant_docs": ["doc_002", "doc_004"],    # Auto-generated (score < 3)
        "doc_details": [{ ... }]                      # Required, individual doc scores
    }
"""

from typing import Any, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import csv
import io
import json

from backend.services.retrieval_evaluation_service import (
    RetrievalEvaluationService,
    QueryEvaluation,
    DocDetail,
    RetrievalEvaluation,  # Legacy
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


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models (Query-unit)
# ─────────────────────────────────────────────────────────────────────────────


class DocDetailRequest(BaseModel):
    """Individual document detail in evaluation request."""

    doc_id: str = Field(..., description="문서 ID")
    doc_rank: int = Field(..., ge=1, description="검색 결과 순위 (1-based)")
    doc_title: str = Field("", description="문서 제목")
    relevance_score: int = Field(..., ge=1, le=5, description="관련성 점수 (1-5)")
    retrieval_score: Optional[float] = Field(None, description="원본 검색 점수")
    doc_snippet: str = Field("", description="문서 스니펫")
    chunk_id: Optional[str] = Field(None, description="청크 ID")
    page: Optional[int] = Field(None, description="페이지 번호")


class QueryEvaluationRequest(BaseModel):
    """Request to save a query-unit evaluation (batch save)."""

    source: Literal["chat", "search"] = Field(..., description="출처 (chat 또는 search)")
    query: str = Field(..., description="원본 검색 쿼리")
    doc_details: List[DocDetailRequest] = Field(..., min_length=1, description="문서별 평가 상세 (필수)")
    # Chat context (optional for search)
    session_id: Optional[str] = Field(None, description="세션 ID (chat만 해당)")
    turn_id: Optional[int] = Field(None, description="턴 ID (chat만 해당)")
    # Filter context
    filter_devices: Optional[List[str]] = Field(None, description="검색 필터 (장비)")
    filter_doc_types: Optional[List[str]] = Field(None, description="검색 필터 (문서 종류)")
    search_queries: Optional[List[str]] = Field(None, description="Multi-query expansion 결과")
    # Search params (search only)
    search_params: Optional[dict[str, Any]] = Field(None, description="검색 파라미터 (search만 해당)")
    # Reviewer info
    reviewer_name: Optional[str] = Field(None, description="평가자 이름 (선택)")


class DocDetailResponse(BaseModel):
    """Individual document detail in evaluation response."""

    doc_id: str
    doc_rank: int
    doc_title: str = ""
    relevance_score: int
    retrieval_score: Optional[float] = None
    doc_snippet: str = ""
    chunk_id: Optional[str] = None
    page: Optional[int] = None


class QueryEvaluationResponse(BaseModel):
    """Response for a query-unit evaluation."""

    query_id: str
    source: str
    query: str
    relevant_docs: List[str]
    irrelevant_docs: List[str]
    doc_details: List[DocDetailResponse]
    session_id: Optional[str] = None
    turn_id: Optional[int] = None
    filter_devices: Optional[List[str]] = None
    filter_doc_types: Optional[List[str]] = None
    search_queries: Optional[List[str]] = None
    search_params: Optional[dict[str, Any]] = None
    reviewer_name: Optional[str] = None
    ts: str


class QueryEvaluationListResponse(BaseModel):
    """Response for a list of query evaluations."""

    items: List[QueryEvaluationResponse]
    total: int


class EvaluationStatisticsResponse(BaseModel):
    """Response for evaluation statistics."""

    total_queries: int
    total_doc_evaluations: int
    avg_docs_per_query: float
    avg_relevance: Optional[float] = None
    source_distribution: dict
    relevance_distribution: dict


def _query_evaluation_to_response(evaluation: QueryEvaluation) -> QueryEvaluationResponse:
    """Convert QueryEvaluation dataclass to response model."""
    return QueryEvaluationResponse(
        query_id=evaluation.query_id,
        source=evaluation.source,
        query=evaluation.query,
        relevant_docs=evaluation.relevant_docs,
        irrelevant_docs=evaluation.irrelevant_docs,
        doc_details=[
            DocDetailResponse(
                doc_id=d.doc_id,
                doc_rank=d.doc_rank,
                doc_title=d.doc_title,
                relevance_score=d.relevance_score,
                retrieval_score=d.retrieval_score,
                doc_snippet=d.doc_snippet,
                chunk_id=d.chunk_id,
                page=d.page,
            )
            for d in evaluation.doc_details
        ],
        session_id=evaluation.session_id,
        turn_id=evaluation.turn_id,
        filter_devices=evaluation.filter_devices,
        filter_doc_types=evaluation.filter_doc_types,
        search_queries=evaluation.search_queries,
        search_params=evaluation.search_params,
        reviewer_name=evaluation.reviewer_name,
        ts=evaluation.ts.isoformat() if evaluation.ts else "",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Query-unit Endpoints
# NOTE: Static routes MUST come BEFORE dynamic routes
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/statistics", response_model=EvaluationStatisticsResponse)
async def get_statistics(
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Get retrieval evaluation statistics."""
    stats = service.get_statistics()
    return EvaluationStatisticsResponse(
        total_queries=stats.get("total_queries", 0),
        total_doc_evaluations=stats.get("total_doc_evaluations", 0),
        avg_docs_per_query=stats.get("avg_docs_per_query", 0.0),
        avg_relevance=stats.get("avg_relevance"),
        source_distribution=stats.get("source_distribution", {}),
        relevance_distribution=stats.get("relevance_distribution", {}),
    )


@router.get("/export/json")
async def export_retrieval_test_json(
    min_relevance: int = Query(3, ge=1, le=5, description="Minimum relevance score for 'relevant'"),
    limit: int = Query(10000, ge=1, le=100000, description="Maximum number of records"),
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Export evaluation data as JSON for retrieval testing.

    Returns query-unit data with relevant/irrelevant document lists.
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
    min_relevance: int = Query(3, ge=1, le=5, description="Minimum relevance score for 'relevant'"),
    limit: int = Query(10000, ge=1, le=100000, description="Maximum number of records"),
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Export evaluation data as CSV for retrieval testing.

    Returns query-unit data in CSV format for spreadsheet analysis.
    """
    data = service.export_for_retrieval_test(min_relevance=min_relevance, limit=limit)

    # Create CSV content
    output = io.StringIO()
    if data:
        # Flatten arrays for CSV
        csv_data = []
        for row in data:
            csv_row = {
                "query_id": row.get("query_id", ""),
                "source": row.get("source", ""),
                "query": row.get("query", ""),
                "relevant_docs": ",".join(row.get("relevant_docs", [])),
                "irrelevant_docs": ",".join(row.get("irrelevant_docs", [])),
                "filter_devices": ",".join(row.get("filter_devices", []) or []),
                "filter_doc_types": ",".join(row.get("filter_doc_types", []) or []),
                "ts": row.get("ts", ""),
            }
            csv_data.append(csv_row)

        writer = csv.DictWriter(output, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

    content = output.getvalue()
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=retrieval_evaluations.csv"
        },
    )


@router.get("/list", response_model=QueryEvaluationListResponse)
async def list_evaluations(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    source: Optional[Literal["chat", "search"]] = Query(None, description="Filter by source"),
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """List all query evaluations with pagination."""
    items, total = service.list_query_evaluations(limit=limit, offset=offset, source=source)
    return QueryEvaluationListResponse(
        items=[_query_evaluation_to_response(e) for e in items],
        total=total,
    )


# Dynamic routes with path parameters - MUST come AFTER static routes


@router.post("/query/{query_id}", response_model=QueryEvaluationResponse)
async def save_query_evaluation(
    query_id: str,
    req: QueryEvaluationRequest,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Save a query-unit evaluation (batch save).

    Stores relevance scores for all documents in a single query.
    relevant_docs and irrelevant_docs are auto-generated based on relevance_score >= 3.
    """
    # Convert request to QueryEvaluation
    doc_details = [
        DocDetail(
            doc_id=d.doc_id,
            doc_rank=d.doc_rank,
            doc_title=d.doc_title,
            relevance_score=d.relevance_score,
            retrieval_score=d.retrieval_score,
            doc_snippet=d.doc_snippet,
            chunk_id=d.chunk_id,
            page=d.page,
        )
        for d in req.doc_details
    ]

    evaluation = QueryEvaluation(
        query_id=query_id,
        source=req.source,
        query=req.query,
        doc_details=doc_details,
        session_id=req.session_id,
        turn_id=req.turn_id,
        filter_devices=req.filter_devices,
        filter_doc_types=req.filter_doc_types,
        search_queries=req.search_queries,
        search_params=req.search_params,
        reviewer_name=req.reviewer_name,
    )

    service.save_query_evaluation(query_id, evaluation)

    # Retrieve and return saved evaluation
    saved = service.get_query_evaluation(query_id)
    if saved is None:
        raise HTTPException(status_code=500, detail="Failed to save evaluation")

    return _query_evaluation_to_response(saved)


@router.get("/query/{query_id}", response_model=QueryEvaluationResponse)
async def get_query_evaluation(
    query_id: str,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Get evaluation by query_id."""
    evaluation = service.get_query_evaluation(query_id)
    if evaluation is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return _query_evaluation_to_response(evaluation)


@router.delete("/query/{query_id}")
async def delete_query_evaluation(
    query_id: str,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Delete evaluation by query_id."""
    deleted = service.delete_query_evaluation(query_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return {"message": "Evaluation deleted", "query_id": query_id}


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Endpoints (deprecated, for backwards compatibility)
# ─────────────────────────────────────────────────────────────────────────────


class LegacyEvaluationRequest(BaseModel):
    """Request to save a document relevance evaluation (legacy, per-document)."""

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


class LegacyEvaluationResponse(BaseModel):
    """Response for a document relevance evaluation (legacy)."""

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


class LegacyEvaluationListResponse(BaseModel):
    """Response for a list of evaluations (legacy)."""

    items: List[LegacyEvaluationResponse]
    total: int


def _legacy_evaluation_to_response(evaluation: RetrievalEvaluation) -> LegacyEvaluationResponse:
    """Convert RetrievalEvaluation dataclass to response model (legacy)."""
    return LegacyEvaluationResponse(
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


@router.post("/{session_id}/{turn_id}/{doc_id}", response_model=LegacyEvaluationResponse, deprecated=True)
async def save_evaluation_legacy(
    session_id: str,
    turn_id: int,
    doc_id: str,
    req: LegacyEvaluationRequest,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Save a document relevance evaluation (legacy, per-document).

    @deprecated: Use POST /query/{query_id} for query-unit storage.
    """
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

    saved = service.get_evaluation(session_id, turn_id, doc_id)
    if saved is None:
        raise HTTPException(status_code=500, detail="Failed to save evaluation")

    return _legacy_evaluation_to_response(saved)


@router.get("/{session_id}/{turn_id}/{doc_id}", response_model=LegacyEvaluationResponse, deprecated=True)
async def get_evaluation_legacy(
    session_id: str,
    turn_id: int,
    doc_id: str,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """Get evaluation for a specific document (legacy).

    @deprecated: Use GET /query/{query_id} for query-unit storage.
    """
    evaluation = service.get_evaluation(session_id, turn_id, doc_id)
    if evaluation is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return _legacy_evaluation_to_response(evaluation)


@router.get("/{session_id}/{turn_id}", response_model=LegacyEvaluationListResponse, deprecated=True)
async def list_evaluations_for_turn_legacy(
    session_id: str,
    turn_id: int,
    service: RetrievalEvaluationService = Depends(get_retrieval_evaluation_service),
):
    """List all evaluations for a specific turn (legacy).

    @deprecated: Use GET /query/{query_id} with query_id="{session_id}:{turn_id}".
    """
    items = service.list_evaluations_for_turn(session_id, turn_id)

    return LegacyEvaluationListResponse(
        items=[_legacy_evaluation_to_response(e) for e in items],
        total=len(items),
    )
