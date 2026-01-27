"""Feedback API - Detailed feedback storage for LLM fine-tuning.

Provides endpoints for:
- Saving detailed feedback (accuracy, completeness, relevance scores)
- Getting feedback for a specific turn
- Listing feedback with filters
- Exporting feedback data for fine-tuning
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import csv
import io
import json

from backend.services.feedback_service import (
    FeedbackService,
    Feedback,
)

router = APIRouter(prefix="/feedback", tags=["Feedback"])

# Global service instance (initialized on startup)
_feedback_service: FeedbackService | None = None


def get_feedback_service() -> FeedbackService:
    """Get the feedback service instance."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService.from_settings()
        _feedback_service.ensure_index()
    return _feedback_service


# --- Request/Response Models ---


class FeedbackRequest(BaseModel):
    """Request to save detailed feedback."""

    accuracy: int = Field(..., ge=1, le=5, description="정확성 점수 (1-5)")
    completeness: int = Field(..., ge=1, le=5, description="완성도 점수 (1-5)")
    relevance: int = Field(..., ge=1, le=5, description="관련성 점수 (1-5)")
    comment: Optional[str] = Field(None, description="자유 의견 (선택사항)")
    reviewer_name: Optional[str] = Field(None, description="피드백 제출자 이름 (선택)")
    logs: Optional[List[str]] = Field(None, description="피드백 시점 실행 로그")
    user_text: Optional[str] = Field(None, description="사용자 질문 (자동 저장용)")
    assistant_text: Optional[str] = Field(None, description="AI 답변 (자동 저장용)")


class FeedbackResponse(BaseModel):
    """Response for feedback data."""

    session_id: str
    turn_id: int
    user_text: str
    assistant_text: str
    accuracy: int
    completeness: int
    relevance: int
    avg_score: float
    rating: str
    comment: Optional[str] = None
    reviewer_name: Optional[str] = None
    logs: Optional[List[str]] = None
    ts: str


class FeedbackListResponse(BaseModel):
    """Response for feedback list."""

    items: List[FeedbackResponse]
    total: int


class FeedbackStatisticsResponse(BaseModel):
    """Response for feedback statistics."""

    total_count: int
    avg_accuracy: Optional[float] = None
    avg_completeness: Optional[float] = None
    avg_relevance: Optional[float] = None
    avg_score: Optional[float] = None
    rating_distribution: dict


def _feedback_to_response(feedback: Feedback) -> FeedbackResponse:
    """Convert Feedback dataclass to response model."""
    return FeedbackResponse(
        session_id=feedback.session_id,
        turn_id=feedback.turn_id,
        user_text=feedback.user_text,
        assistant_text=feedback.assistant_text,
        accuracy=feedback.accuracy,
        completeness=feedback.completeness,
        relevance=feedback.relevance,
        avg_score=feedback.avg_score,
        rating=feedback.rating,
        comment=feedback.comment,
        reviewer_name=feedback.reviewer_name,
        logs=feedback.logs,
        ts=feedback.ts.isoformat() if feedback.ts else "",
    )


# --- Endpoints ---
# NOTE: Static routes MUST come BEFORE dynamic routes like /{session_id}/{turn_id}


@router.get("/statistics", response_model=FeedbackStatisticsResponse)
async def get_statistics(
    service: FeedbackService = Depends(get_feedback_service),
):
    """Get feedback statistics."""
    stats = service.get_statistics()
    return FeedbackStatisticsResponse(
        total_count=stats.get("total_count", 0),
        avg_accuracy=stats.get("avg_accuracy"),
        avg_completeness=stats.get("avg_completeness"),
        avg_relevance=stats.get("avg_relevance"),
        avg_score=stats.get("avg_score"),
        rating_distribution=stats.get("rating_distribution", {}),
    )


@router.get("/export/json")
async def export_finetuning_json(
    min_score: float = 3.0,
    limit: int = 10000,
    service: FeedbackService = Depends(get_feedback_service),
):
    """Export feedback data as JSON for fine-tuning.

    Returns feedback with avg_score >= min_score.
    """
    data = service.export_for_finetuning(min_score=min_score, limit=limit)

    content = json.dumps(data, ensure_ascii=False, indent=2)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={
            "Content-Disposition": "attachment; filename=feedback_export.json"
        },
    )


@router.get("/export/csv")
async def export_finetuning_csv(
    min_score: float = 3.0,
    limit: int = 10000,
    service: FeedbackService = Depends(get_feedback_service),
):
    """Export feedback data as CSV for fine-tuning.

    Returns feedback with avg_score >= min_score.
    """
    data = service.export_for_finetuning(min_score=min_score, limit=limit)

    # Create CSV content
    output = io.StringIO()
    if data:
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    content = output.getvalue()
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=feedback_export.csv"
        },
    )


@router.get("", response_model=FeedbackListResponse)
async def list_feedback(
    limit: int = 50,
    offset: int = 0,
    rating: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    service: FeedbackService = Depends(get_feedback_service),
):
    """List feedback with optional filters.

    Args:
        limit: Maximum number of results (default 50).
        offset: Number of results to skip.
        rating: Filter by rating ("up" or "down").
        min_score: Filter by minimum average score.
        max_score: Filter by maximum average score.
    """
    if rating and rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    items, total = service.list_feedback(
        limit=limit,
        offset=offset,
        rating=rating,
        min_score=min_score,
        max_score=max_score,
    )

    return FeedbackListResponse(
        items=[_feedback_to_response(f) for f in items],
        total=total,
    )


# Dynamic routes with path parameters - MUST come AFTER static routes
@router.post("/{session_id}/{turn_id}", response_model=FeedbackResponse)
async def save_feedback(
    session_id: str,
    turn_id: int,
    req: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service),
):
    """Save detailed feedback for a specific turn.

    Stores feedback with accuracy, completeness, relevance scores (1-5)
    and optional comment and execution logs.
    """
    # Create feedback object
    feedback = Feedback(
        session_id=session_id,
        turn_id=turn_id,
        user_text=req.user_text or "",
        assistant_text=req.assistant_text or "",
        accuracy=req.accuracy,
        completeness=req.completeness,
        relevance=req.relevance,
        comment=req.comment,
        reviewer_name=req.reviewer_name,
        logs=req.logs,
    )

    service.save_feedback(feedback)

    # Retrieve and return saved feedback
    saved = service.get_feedback(session_id, turn_id)
    if saved is None:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    return _feedback_to_response(saved)


@router.get("/{session_id}/{turn_id}", response_model=FeedbackResponse)
async def get_feedback(
    session_id: str,
    turn_id: int,
    service: FeedbackService = Depends(get_feedback_service),
):
    """Get feedback for a specific turn."""
    feedback = service.get_feedback(session_id, turn_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found")

    return _feedback_to_response(feedback)
