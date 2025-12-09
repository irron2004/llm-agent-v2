"""Search Service API (검색 결과 + 페이지네이션)."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from backend.api.dependencies import get_search_service
from backend.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search Service"])


class SearchResultItem(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rank": 1,
                "id": "doc-001",
                "title": "PM 점검 가이드",
                "snippet": "PM 예방 점검은 장비의 안정적인 운영을 위해...",
                "score": 0.92,
                "score_display": "92%",
                "highlight_terms": ["PM", "점검"]
            }
        }
    )

    rank: int
    id: str
    title: str
    snippet: str
    score: float
    score_display: str
    highlight_terms: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    clean_query: str
    items: List[SearchResultItem]
    total: int
    page: int
    size: int
    has_next: bool
    reranked: bool = Field(default=False, description="Whether results were reranked")


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="검색어", min_length=1),
    page: int = Query(default=1, ge=1, description="페이지 번호"),
    size: int = Query(default=10, ge=1, le=100, description="페이지 크기"),
    rerank: Optional[bool] = Query(
        default=None, description="Enable reranking (None = use service default)"
    ),
    rerank_top_k: Optional[int] = Query(
        default=None, ge=1, le=100, description="Number of results after reranking"
    ),
    search_service: SearchService = Depends(get_search_service),
):
    """문서 검색 API.

    Args:
        q: 검색어
        page: 페이지 번호 (1부터 시작)
        size: 페이지당 결과 수
        rerank: 리랭킹 활성화 여부 (None이면 서비스 기본값 사용)
        rerank_top_k: 리랭킹 후 반환할 결과 수
    """
    try:
        top_k = page * size + size  # 여유분 포함
        # If rerank_top_k not specified, use top_k to avoid pagination mismatch
        effective_rerank_top_k = rerank_top_k if rerank_top_k is not None else top_k
        results = search_service.search(
            q,
            top_k=top_k,
            rerank=rerank,
            rerank_top_k=effective_rerank_top_k,
        )

        start_idx = (page - 1) * size
        end_idx = start_idx + size
        page_results = results[start_idx:end_idx]

        items = _to_search_items(page_results, start_idx, q)
        total = len(results)
        has_next = end_idx < total

        # Determine if reranking was actually applied
        was_reranked = (
            rerank if rerank is not None else search_service.rerank_enabled
        ) and search_service.reranker is not None

        return SearchResponse(
            query=q,
            clean_query=q,
            items=items,
            total=total,
            page=page,
            size=size,
            has_next=has_next,
            reranked=was_reranked,
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _to_search_items(results, start_idx: int, query: str) -> list[SearchResultItem]:
    items: list[SearchResultItem] = []
    for idx, result in enumerate(results):
        title = ""
        if getattr(result, "metadata", None):
            title = result.metadata.get("title", "")
        if not title:
            title = (result.raw_text or result.content or "").split("\n")[0][:50]

        snippet_source = result.raw_text or result.content or ""
        snippet = snippet_source[:150] + ("..." if len(snippet_source) > 150 else "")

        score = getattr(result, "score", 0.0) or 0.0
        score_display = f"{int(score * 100)}%" if score <= 1 else f"{score:.2f}"

        items.append(
            SearchResultItem(
                rank=start_idx + idx + 1,
                id=getattr(result, "doc_id", ""),
                title=title,
                snippet=snippet,
                score=score,
                score_display=score_display,
                highlight_terms=query.split()[:3],
            )
        )
    return items


__all__ = [
    "router",
    "SearchResponse",
    "SearchResultItem",
]
