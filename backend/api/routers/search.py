"""Search Service API (검색 결과 + 페이지네이션)."""

import inspect
import threading
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from backend.api.dependencies import get_search_service
from backend.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search Service"])


# =============================================================================
# Active Request Counter (for monitoring concurrent searches)
# =============================================================================
class ActiveRequestCounter:
    """Thread-safe counter for active search requests."""

    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._count += 1
            return self._count

    def decrement(self) -> int:
        with self._lock:
            self._count = max(0, self._count - 1)
            return self._count

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


_active_requests = ActiveRequestCounter()


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
                "expanded_pages": [14, 15, 16],
                "expanded_page_urls": [
                    "/api/assets/docs/doc-001/pages/14",
                    "/api/assets/docs/doc-001/pages/15",
                    "/api/assets/docs/doc-001/pages/16",
                ],
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
    expanded_pages: List[int] = Field(default_factory=list)
    expanded_page_urls: List[str] = Field(default_factory=list)


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
    use_rrf: Optional[bool] = Query(
        default=None, description="Use RRF (Reciprocal Rank Fusion) for score combination. When True, weights are ignored."
    ),
    rrf_k: Optional[int] = Query(
        default=None, ge=1, le=100, description="RRF rank constant (only used if use_rrf=True)"
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

        # Add RRF parameters if provided
        if use_rrf is not None:
            search_kwargs["use_rrf"] = use_rrf
        if rrf_k is not None:
            search_kwargs["rrf_k"] = rrf_k

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


# Doc types that should fetch all chunks from same doc_id (myservice: status, action, reason)
DOC_TYPES_SAME_DOC = {"sop", "ts", "setup"}

# Page window for expanding adjacent pages
PAGE_WINDOW = 1  # -1, +1 from current page


def _normalize_doc_type(doc_type: Optional[str]) -> Optional[str]:
    """Normalize doc_type to lowercase for comparison."""
    if not doc_type or not isinstance(doc_type, str):
        return None
    return doc_type.strip().lower()


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
        # chunk_summary가 list인 경우 문자열로 변환
        if isinstance(chunk_summary, list):
            chunk_summary = " ".join(str(s) for s in chunk_summary) if chunk_summary else None
        elif chunk_summary is not None and not isinstance(chunk_summary, str):
            chunk_summary = str(chunk_summary)

        chunk_keywords = metadata.get("chunk_keywords", [])
        if isinstance(chunk_keywords, str):
            chunk_keywords = [chunk_keywords]
        elif not isinstance(chunk_keywords, list):
            chunk_keywords = []

        chapter = metadata.get("chapter")
        page = metadata.get("page")
        doc_type = metadata.get("doc_type")
        device_name = metadata.get("device_name")

        # Expand pages based on doc type
        doc_id = getattr(result, "doc_id", "")
        expanded_pages: List[int] = []
        expanded_page_urls: List[str] = []

        normalized_doc_type = _normalize_doc_type(doc_type)
        if normalized_doc_type in DOC_TYPES_SAME_DOC:
            # For myservice docs (sop, ts, setup), use page 0 (entire doc)
            if page is not None and isinstance(page, int):
                expanded_pages = [page]
            else:
                expanded_pages = [0]  # Default to page 0 for myservice docs
        elif page is not None and isinstance(page, int):
            # For regular docs, expand to adjacent pages
            page_min = max(1, page - PAGE_WINDOW)
            page_max = page + PAGE_WINDOW
            expanded_pages = list(range(page_min, page_max + 1))

        # Generate URLs for expanded pages
        if expanded_pages and doc_id:
            expanded_page_urls = [
                f"/api/assets/docs/{doc_id}/pages/{p}" for p in expanded_pages
            ]

        items.append(
            SearchResultItem(
                rank=start_idx + idx + 1,
                id=doc_id,
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
                expanded_pages=expanded_pages,
                expanded_page_urls=expanded_page_urls,
            )
        )
    return items


# =============================================================================
# Chat Pipeline Search (Retrieval Test용)
# =============================================================================
from backend.api.dependencies import get_default_llm, get_prompt_spec_cached
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent

# Singleton agent for chat pipeline search
_chat_pipeline_agent: Optional[LangGraphRAGAgent] = None


def _get_chat_pipeline_agent(search_service: SearchService) -> LangGraphRAGAgent:
    """Get or create chat pipeline agent for retrieval test."""
    global _chat_pipeline_agent
    if _chat_pipeline_agent is None:
        llm = get_default_llm()
        spec = get_prompt_spec_cached()
        _chat_pipeline_agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=spec,
            top_k=20,
            retrieval_top_k=100,
            mode="base",  # No retry/judge
            auto_parse_enabled=True,  # Enable translate + auto_parse
        )
    return _chat_pipeline_agent


class ChatPipelineSearchRequest(BaseModel):
    """Chat 파이프라인 검색 요청."""
    query: str = Field(..., description="검색 쿼리")
    search_override: Optional[dict] = Field(
        default=None,
        description="검색 파라미터 override (dense_weight, sparse_weight, rerank 등)"
    )
    selected_devices: Optional[List[str]] = Field(default=None, description="선택된 장비 필터")
    selected_doc_types: Optional[List[str]] = Field(default=None, description="선택된 문서 타입 필터")


class ChatPipelineSearchResult(BaseModel):
    """Chat 파이프라인 검색 결과 아이템."""
    rank: int
    id: str
    title: str
    snippet: str
    content: Optional[str] = None  # 본문 (LLM 컨텍스트용)
    score: float
    score_display: str
    chapter: Optional[str] = None
    page: Optional[int] = None
    doc_type: Optional[str] = None
    device_name: Optional[str] = None
    expanded_pages: List[int] = Field(default_factory=list)
    expanded_page_urls: List[str] = Field(default_factory=list)


class ChatPipelineSearchResponse(BaseModel):
    """Chat 파이프라인 검색 응답."""
    query: str
    query_en: Optional[str] = None
    query_ko: Optional[str] = None
    route: Optional[str] = None
    search_queries: List[str] = Field(default_factory=list)
    auto_parsed_device: Optional[str] = None
    auto_parsed_doc_type: Optional[str] = None
    items: List[ChatPipelineSearchResult]
    total: int


class ActiveRequestsResponse(BaseModel):
    """현재 실행 중인 검색 요청 수."""
    active_count: int


@router.get("/active-requests", response_model=ActiveRequestsResponse)
async def get_active_requests():
    """현재 실행 중인 chat-pipeline 검색 요청 수를 반환합니다."""
    return ActiveRequestsResponse(active_count=_active_requests.count)


@router.post("/chat-pipeline", response_model=ChatPipelineSearchResponse)
async def search_with_chat_pipeline(
    request: ChatPipelineSearchRequest,
    search_service: SearchService = Depends(get_search_service),
):
    """Chat과 동일한 파이프라인으로 검색 (Retrieval Test용).

    translate → auto_parse → route → mq → retrieve → expand 파이프라인을 실행.
    답변 생성은 하지 않음.

    search_override로 검색 파라미터 조절 가능:
    - dense_weight, sparse_weight
    - use_rrf, rrf_k
    - rerank, rerank_top_k, top_k
    """
    _active_requests.increment()
    try:
        agent = _get_chat_pipeline_agent(search_service)

        result = agent.retrieve_only(
            query=request.query,
            search_override=request.search_override,
            selected_devices=request.selected_devices,
            selected_doc_types=request.selected_doc_types,
        )

        # Convert docs to response items
        docs = result.get("docs", [])
        answer_ref_json = result.get("answer_ref_json", [])

        # Build doc_id to ref_json mapping for expanded pages
        ref_map = {ref.get("doc_id"): ref for ref in answer_ref_json}

        items = []
        for idx, doc in enumerate(docs):
            metadata = getattr(doc, "metadata", {}) or {}
            doc_id = getattr(doc, "doc_id", "")

            title = metadata.get("title", "")
            if not title:
                title = (doc.raw_text or doc.content or "").split("\n")[0][:50]

            snippet_source = doc.raw_text or doc.content or ""
            snippet = snippet_source[:150] + ("..." if len(snippet_source) > 150 else "")

            # Full content for LLM context
            content = doc.raw_text or doc.content or ""
            if len(content) > 10000:
                content = content[:10000]  # Truncate

            score = getattr(doc, "score", 0.0) or 0.0
            score_display = f"{int(score * 100)}%" if score <= 1 else f"{score:.2f}"

            # Get expanded pages from answer_ref_json
            ref = ref_map.get(doc_id, {})
            expanded_pages = ref.get("expanded_pages", [])
            expanded_page_urls = ref.get("expanded_page_urls", [])

            # Fallback to simple page expansion if not in ref_json
            if not expanded_pages:
                page = metadata.get("page")
                if page is not None:
                    doc_type = metadata.get("doc_type", "")
                    if doc_type and doc_type.lower() in {"sop", "ts", "setup"}:
                        expanded_pages = [page]
                    else:
                        expanded_pages = list(range(max(1, page - 1), page + 2))
                    expanded_page_urls = [
                        f"/api/assets/docs/{doc_id}/pages/{p}" for p in expanded_pages
                    ]

            items.append(ChatPipelineSearchResult(
                rank=idx + 1,
                id=doc_id,
                title=title,
                snippet=snippet,
                content=content,
                score=score,
                score_display=score_display,
                chapter=metadata.get("chapter"),
                page=metadata.get("page"),
                doc_type=metadata.get("doc_type"),
                device_name=metadata.get("device_name"),
                expanded_pages=expanded_pages,
                expanded_page_urls=expanded_page_urls,
            ))

        return ChatPipelineSearchResponse(
            query=request.query,
            query_en=result.get("query_en"),
            query_ko=result.get("query_ko"),
            route=result.get("route"),
            search_queries=result.get("search_queries", []),
            auto_parsed_device=result.get("auto_parsed_device"),
            auto_parsed_doc_type=result.get("auto_parsed_doc_type"),
            items=items,
            total=len(items),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _active_requests.decrement()


__all__ = [
    "router",
    "SearchResponse",
    "SearchResultItem",
    "ChatPipelineSearchRequest",
    "ChatPipelineSearchResponse",
]
