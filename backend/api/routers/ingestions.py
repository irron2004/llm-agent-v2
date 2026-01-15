"""Ingestions API router - VLM parsing results browser."""

import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/ingestions", tags=["Ingestions"])

# 기본 ingestions 폴더 경로 (환경변수로 오버라이드 가능)
# Docker: /data/ingestions, Local: /home/hskim/work/llm-agent-v2/data/ingestions
INGESTIONS_BASE = os.environ.get("INGESTIONS_BASE", "/data/ingestions")


class RunFolder(BaseModel):
    name: str
    path: str


class RunFoldersResponse(BaseModel):
    folders: List[RunFolder]
    base_path: str


@router.get("/runs", response_model=RunFoldersResponse)
async def list_run_folders():
    """List available ingestion run folders (date folders)."""
    base_path = Path(INGESTIONS_BASE)

    if not base_path.exists():
        raise HTTPException(status_code=404, detail=f"Ingestions base path not found: {INGESTIONS_BASE}")

    folders = []
    for item in sorted(base_path.iterdir(), reverse=True):  # 최신순 정렬
        if item.is_dir() and not item.name.startswith("."):
            folders.append(RunFolder(
                name=item.name,
                path=str(item.relative_to(base_path)),
            ))

    return RunFoldersResponse(folders=folders, base_path=str(base_path))


__all__ = ["router"]
