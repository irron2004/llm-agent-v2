"""Compatibility endpoints for legacy retrieval-evaluation UI calls."""

from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config.settings import api_settings

router = APIRouter(prefix="/retrieval-evaluation", tags=["Retrieval Evaluation"])


class RetrievalEvaluationListResponse(BaseModel):
    """Legacy-compatible list payload."""

    items: List[dict[str, Any]] = Field(default_factory=list)
    evaluations: List[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    limit: int = 100
    offset: int = 0


@router.get("/list", response_model=RetrievalEvaluationListResponse)
async def list_retrieval_evaluations(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """Return empty list instead of 404 for old frontend bundles.

    The current codebase stores retrieval-test results client-side, so this endpoint
    intentionally returns an empty list for backward compatibility.
    """
    if not api_settings.enable_legacy_compat_routes:
        raise HTTPException(status_code=404, detail="Not Found")

    return RetrievalEvaluationListResponse(
        items=[],
        evaluations=[],
        total=0,
        limit=limit,
        offset=offset,
    )


__all__ = ["router", "RetrievalEvaluationListResponse"]
