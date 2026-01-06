"""Search Service API (검색 결과 + 페이지네이션)."""

import inspect
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
                "highlight_terms": ["PM", "점검"],
                "chunk_summary": "PM 점검 절차 요약",
                "chunk_keywords": ["PM", "점검", "장비"],
                "chapter": "3. 예방 점검",
                "page": 15,
                "doc_type": "SOP",
                "device_name": "SUPRAN",
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
    chunk_summary: Optional[str] = None
    chunk_keywords: List[str] = Field(default_factory=list)
    chapter: Optional[str] = None
    page: Optional[int] = None
    doc_type: Optional[str] = None
    device_name: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    clean_query: str
    items: List[SearchResultItem]
    total: int
    page: int
    size: int
    has_next: bool
    multi_query_used: bool = Field(default=False, description="Whether multi-query expansion was used")
    reranked: bool = Field(default=False, description="Whether results were reranked")


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="검색어", min_length=1),
    page: int = Query(default=1, ge=1, description="페이지 번호"),
    size: int = Query(default=10, ge=1, le=100, description="페이지 크기"),
    multi_query: Optional[bool] = Query(
        default=None, description="Enable multi-query expansion (None = use service default)"
    ),
    multi_query_n: Optional[int] = Query(
        default=None, ge=1, le=10, description="Number of expanded queries"
    ),
    rerank: Optional[bool] = Query(
        default=None, description="Enable reranking (None = use service default)"
    ),
    rerank_top_k: Optional[int] = Query(
        default=None, ge=1, le=100, description="Number of results after reranking"
    ),
    field_weights: Optional[str] = Query(
        default=None,
        description="Field weights in format: field1^weight1,field2^weight2 (e.g. search_text^1.0,chunk_summary^0.7)"
    ),
    dense_weight: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Dense (vector) weight for hybrid search (0.0 = BM25 only)"
    ),
    sparse_weight: Optional[float] = Query(
        default=None, ge=0.0, le=1.0, description="Sparse (BM25) weight for hybrid search (1.0 = BM25 only)"
    ),
    search_service: SearchService = Depends(get_search_service),
):
    """문서 검색 API.

    Args:
        q: 검색어
        page: 페이지 번호 (1부터 시작)
        size: 페이지당 결과 수
        multi_query: 멀티쿼리 확장 활성화 여부 (None이면 서비스 기본값 사용)
        multi_query_n: 확장 쿼리 개수
        rerank: 리랭킹 활성화 여부 (None이면 서비스 기본값 사용)
        rerank_top_k: 리랭킹 후 반환할 결과 수
        field_weights: 필드별 가중치 (e.g. search_text^1.0,chunk_summary^0.7)
    """
    try:
        top_k = page * size + size  # 여유분 포함
        # If rerank_top_k not specified, use top_k to avoid pagination mismatch
        effective_rerank_top_k = rerank_top_k if rerank_top_k is not None else top_k

        # Parse field weights if provided
        text_fields = None
        if field_weights:
            text_fields = [f.strip() for f in field_weights.split(",") if f.strip()]

        search_kwargs = {
            "top_k": top_k,
            "multi_query": multi_query,
            "multi_query_n": multi_query_n,
            "rerank": rerank,
            "rerank_top_k": effective_rerank_top_k,
        }

        if text_fields is not None:
            signature = inspect.signature(search_service.search)
            if "text_fields" in signature.parameters:
                search_kwargs["text_fields"] = text_fields

        # Add hybrid search weights if provided (for EsSearchService)
        if dense_weight is not None:
            search_kwargs["dense_weight"] = dense_weight
        if sparse_weight is not None:
            search_kwargs["sparse_weight"] = sparse_weight

        results = search_service.search(q, **search_kwargs)

        start_idx = (page - 1) * size
        end_idx = start_idx + size
        page_results = results[start_idx:end_idx]

        items = _to_search_items(page_results, start_idx, q)
        total = len(results)
        has_next = end_idx < total

        # Determine if multi-query was actually applied
        service_multi_query_enabled = bool(getattr(search_service, "multi_query_enabled", False))
        was_multi_query = (
            multi_query if multi_query is not None else service_multi_query_enabled
        ) and getattr(search_service, "query_expander", None) is not None

        # Determine if reranking was actually applied
        service_rerank_enabled = bool(getattr(search_service, "rerank_enabled", False))
        was_reranked = (
            rerank if rerank is not None else service_rerank_enabled
        ) and getattr(search_service, "reranker", None) is not None

        return SearchResponse(
            query=q,
            clean_query=q,
            items=items,
            total=total,
            page=page,
            size=size,
            has_next=has_next,
            multi_query_used=was_multi_query,
            reranked=was_reranked,
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _to_search_items(results, start_idx: int, query: str) -> list[SearchResultItem]:
    items: list[SearchResultItem] = []
    for idx, result in enumerate(results):
        metadata = getattr(result, "metadata", {}) or {}

        title = metadata.get("title", "")
        if not title:
            title = (result.raw_text or result.content or "").split("\n")[0][:50]

        snippet_source = result.raw_text or result.content or ""
        snippet = snippet_source[:150] + ("..." if len(snippet_source) > 150 else "")

        score = getattr(result, "score", 0.0) or 0.0
        score_display = f"{int(score * 100)}%" if score <= 1 else f"{score:.2f}"

        # Extract new fields from metadata
        chunk_summary = metadata.get("chunk_summary")
        chunk_keywords = metadata.get("chunk_keywords", [])
        if isinstance(chunk_keywords, str):
            chunk_keywords = [chunk_keywords]
        elif not isinstance(chunk_keywords, list):
            chunk_keywords = []

        chapter = metadata.get("chapter")
        page = metadata.get("page")
        doc_type = metadata.get("doc_type")
        device_name = metadata.get("device_name")

        items.append(
            SearchResultItem(
                rank=start_idx + idx + 1,
                id=getattr(result, "doc_id", ""),
                title=title,
                snippet=snippet,
                score=score,
                score_display=score_display,
                highlight_terms=query.split()[:3],
                chunk_summary=chunk_summary,
                chunk_keywords=chunk_keywords,
                chapter=chapter,
                page=page,
                doc_type=doc_type,
                device_name=device_name,
            )
        )
    return items


__all__ = [
    "router",
    "SearchResponse",
    "SearchResultItem",
]
