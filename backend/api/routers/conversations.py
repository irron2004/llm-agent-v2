"""Conversations API - Chat history storage and retrieval.

Provides endpoints for:
- Listing recent chat sessions
- Getting a specific session with all turns
- Saving turns (user + assistant messages)
- Deleting sessions
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.services.chat_history_service import (
    ChatHistoryService,
    ChatTurn,
    DocRef,
    SessionSummary,
)

router = APIRouter(prefix="/conversations", tags=["Conversations"])

# Global service instance (initialized on startup)
_chat_history_service: ChatHistoryService | None = None


def get_chat_history_service() -> ChatHistoryService:
    """Get the chat history service instance."""
    global _chat_history_service
    if _chat_history_service is None:
        _chat_history_service = ChatHistoryService.from_settings()
        _chat_history_service.ensure_index()
    return _chat_history_service


# ─── Request/Response Models ───


class DocRefModel(BaseModel):
    """Document reference in assistant response."""
    slot: int = Field(..., description="User-visible document number (1, 2, 3...)")
    doc_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    snippet: str = Field(..., description="Content snippet")
    page: Optional[int] = Field(None, description="Page number")
    pages: Optional[List[int]] = Field(None, description="Expanded pages")
    score: Optional[float] = Field(None, description="Relevance score")


class TurnRequest(BaseModel):
    """Request to save a conversation turn."""
    user_text: str = Field(..., description="User's question/message")
    assistant_text: str = Field(..., description="Assistant's response")
    doc_refs: List[DocRefModel] = Field(default_factory=list, description="Referenced documents")
    title: Optional[str] = Field(None, description="Session title (set on first turn)")


class TurnResponse(BaseModel):
    """Response for a saved turn."""
    session_id: str
    turn_id: int
    user_text: str
    assistant_text: str
    doc_refs: List[DocRefModel]
    title: Optional[str]
    ts: str
    feedback_rating: Optional[str] = None
    feedback_reason: Optional[str] = None
    feedback_ts: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request to save satisfaction feedback."""
    rating: str = Field(..., description="Satisfaction rating: up/down")
    reason: Optional[str] = Field(None, description="Reason for dissatisfaction")


class SessionListItem(BaseModel):
    """Session summary for listing."""
    id: str = Field(..., alias="session_id")
    title: str
    preview: str
    turn_count: int = Field(..., alias="turnCount")
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")

    class Config:
        populate_by_name = True


class SessionDetail(BaseModel):
    """Full session detail with turns."""
    session_id: str
    title: str
    turns: List[TurnResponse]
    turn_count: int


class SessionListResponse(BaseModel):
    """Response for session list."""
    sessions: List[SessionListItem]
    total: int


# ─── Endpoints ───


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    service: ChatHistoryService = Depends(get_chat_history_service),
):
    """List recent chat sessions.

    Returns sessions sorted by most recent activity.
    """
    sessions = service.list_sessions(limit=limit, offset=offset)
    return SessionListResponse(
        sessions=[
            SessionListItem(
                session_id=s.session_id,
                title=s.title,
                preview=s.preview,
                turnCount=s.turn_count,
                createdAt=s.created_at.isoformat(),
                updatedAt=s.updated_at.isoformat(),
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    service: ChatHistoryService = Depends(get_chat_history_service),
):
    """Get a session with all its turns.

    Returns the full conversation history for a session.
    """
    turns = service.get_session(session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="Session not found")

    title = turns[0].title or turns[0].user_text[:50] if turns else ""

    return SessionDetail(
        session_id=session_id,
        title=title,
        turns=[
            TurnResponse(
                session_id=t.session_id,
                turn_id=t.turn_id,
                user_text=t.user_text,
                assistant_text=t.assistant_text,
                doc_refs=[
                    DocRefModel(
                        slot=ref.slot,
                        doc_id=ref.doc_id,
                        title=ref.title,
                        snippet=ref.snippet,
                        page=ref.page,
                        pages=ref.pages,
                        score=ref.score,
                    )
                    for ref in t.doc_refs
                ],
                title=t.title,
                ts=t.ts.isoformat() if t.ts else "",
                feedback_rating=t.feedback_rating,
                feedback_reason=t.feedback_reason,
                feedback_ts=t.feedback_ts.isoformat() if t.feedback_ts else None,
            )
            for t in turns
        ],
        turn_count=len(turns),
    )


@router.post("/{session_id}/turns", response_model=TurnResponse)
async def save_turn(
    session_id: str,
    req: TurnRequest,
    service: ChatHistoryService = Depends(get_chat_history_service),
):
    """Save a conversation turn.

    Creates or appends a turn to the specified session.
    """
    # Get next turn ID
    turn_id = service.get_next_turn_id(session_id)

    # Convert doc refs
    doc_refs = [
        DocRef(
            slot=ref.slot,
            doc_id=ref.doc_id,
            title=ref.title,
            snippet=ref.snippet,
            page=ref.page,
            pages=ref.pages,
            score=ref.score,
        )
        for ref in req.doc_refs
    ]

    # Create and save turn
    turn = ChatTurn(
        session_id=session_id,
        turn_id=turn_id,
        user_text=req.user_text,
        assistant_text=req.assistant_text,
        doc_refs=doc_refs,
        title=req.title if turn_id == 1 else None,  # Only set title on first turn
    )
    service.save_turn(turn)

    return TurnResponse(
        session_id=session_id,
        turn_id=turn_id,
        user_text=req.user_text,
        assistant_text=req.assistant_text,
        doc_refs=req.doc_refs,
        title=req.title,
        ts=turn.ts.isoformat() if turn.ts else "",
        feedback_rating=turn.feedback_rating,
        feedback_reason=turn.feedback_reason,
        feedback_ts=turn.feedback_ts.isoformat() if turn.feedback_ts else None,
    )


@router.post("/{session_id}/turns/{turn_id}/feedback", response_model=TurnResponse)
async def save_feedback(
    session_id: str,
    turn_id: int,
    req: FeedbackRequest,
    service: ChatHistoryService = Depends(get_chat_history_service),
):
    """Save satisfaction feedback for a specific turn."""
    rating = req.rating.strip().lower()
    if rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    reason = req.reason.strip() if isinstance(req.reason, str) else None
    if rating == "down" and not reason:
        raise HTTPException(status_code=400, detail="reason is required for down rating")
    if rating == "up":
        reason = None

    updated = service.update_turn_feedback(
        session_id,
        turn_id,
        rating=rating,
        reason=reason,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Session or turn not found")

    return TurnResponse(
        session_id=updated.session_id,
        turn_id=updated.turn_id,
        user_text=updated.user_text,
        assistant_text=updated.assistant_text,
        doc_refs=[
            DocRefModel(
                slot=ref.slot,
                doc_id=ref.doc_id,
                title=ref.title,
                snippet=ref.snippet,
                page=ref.page,
                pages=ref.pages,
                score=ref.score,
            )
            for ref in updated.doc_refs
        ],
        title=updated.title,
        ts=updated.ts.isoformat() if updated.ts else "",
        feedback_rating=updated.feedback_rating,
        feedback_reason=updated.feedback_reason,
        feedback_ts=updated.feedback_ts.isoformat() if updated.feedback_ts else None,
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    service: ChatHistoryService = Depends(get_chat_history_service),
):
    """Delete a session and all its turns (hard delete)."""
    deleted = service.delete_session(session_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": deleted, "session_id": session_id}


@router.post("/{session_id}/hide")
async def hide_session(
    session_id: str,
    service: ChatHistoryService = Depends(get_chat_history_service),
):
    """Hide a session (soft delete).

    The session is hidden from the UI but remains in the database.
    Can be restored later using the unhide endpoint.
    """
    updated = service.hide_session(session_id)
    if updated == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"hidden": updated, "session_id": session_id}


@router.post("/{session_id}/unhide")
async def unhide_session(
    session_id: str,
    service: ChatHistoryService = Depends(get_chat_history_service),
):
    """Unhide a session (restore from soft delete).

    Makes a previously hidden session visible again.
    """
    updated = service.unhide_session(session_id)
    if updated == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"unhidden": updated, "session_id": session_id}
