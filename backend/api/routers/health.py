"""Health check endpoint."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("", response_model=HealthResponse)
async def health_check(request: Request):
    """Simple liveness probe with version info."""
    return HealthResponse(status="ok", version=request.app.version)
