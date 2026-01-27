"""Chat Service API (LLM 단독 / RAG 기반 두 엔드포인트 제공)."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.api.dependencies import (
    get_chat_service,
    get_default_llm,
    get_prompt_spec_cached,
    get_search_service,
    get_simple_chat_prompt,
)
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent
from backend.services.chat_service import ChatService
from backend.services.search_service import SearchService

router = APIRouter(prefix="/chat", tags=["Chat Service"])


# ─── Request/Response Models ───


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "PM 예방 점검 주기는?",
                "history": [],
                "top_k": 3
            }
        }
    )

    message: str = Field(..., description="사용자 질문")
    history: List[HistoryMessage] = Field(default_factory=list, description="대화 히스토리")
    top_k: int = Field(default=10, ge=1, description="검색할 문서 수")


class RetrievedDoc(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    score_percent: int
    page: int | None = None
    page_image_url: str | None = None
    expanded_pages: List[int] | None = None
    expanded_page_urls: List[str] | None = None


class ChatResponse(BaseModel):
    query: str
    clean_query: str
    answer: str
    retrieved_docs: List[RetrievedDoc]
    follow_ups: List[str]
    metadata: dict = Field(default_factory=dict)


# ─── Endpoints ───


@router.post("/simple", response_model=ChatResponse)
async def simple_chat(
    req: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    system_prompt: str | None = Depends(get_simple_chat_prompt),
):
    """LLM 단독 질문 응답 API (검색/컨텍스트 없이 답변만 제공)."""
    try:
        history_payload = [msg.model_dump() for msg in req.history]
        llm_response = chat_service.chat(
            req.message,
            history=history_payload,
            system_prompt=system_prompt,
        )

        retrieved_docs: list[RetrievedDoc] = []
        follow_ups = _generate_follow_ups(req.message)

        return ChatResponse(
            query=req.message,
            clean_query=req.message,
            answer=llm_response.text,
            retrieved_docs=retrieved_docs,
            follow_ups=follow_ups,
            metadata={
                "num_results": len(retrieved_docs),
                "top_k": req.top_k,
                "history_len": len(history_payload),
                "system_prompt": bool(system_prompt),
            },
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/retrieval", response_model=ChatResponse)
async def retrieval_chat(
    req: ChatRequest,
    search_service: SearchService = Depends(get_search_service),
    llm=Depends(get_default_llm),
    prompt_spec=Depends(get_prompt_spec_cached),
):
    """LangGraph 기반 질문 응답 API (검색 + 답변, HIL 없음)."""
    if not hasattr(search_service, "search"):
        raise HTTPException(status_code=503, detail="Search service not configured")

    try:
        agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=prompt_spec,
            top_k=req.top_k,
            mode="base",
            ask_user_after_retrieve=False,
            ask_device_selection=False,
            checkpointer=None,
        )
        result = agent.run(req.message, attempts=0, max_attempts=0, thread_id=None)

        display_docs = result.get("display_docs") or result.get("docs") or []
        retrieved_docs = _to_retrieved_docs(display_docs)
        follow_ups = _generate_follow_ups(req.message)

        return ChatResponse(
            query=req.message,
            clean_query=req.message,
            answer=result.get("answer", ""),
            retrieved_docs=retrieved_docs,
            follow_ups=follow_ups,
            metadata={
                "num_results": len(retrieved_docs),
                "top_k": req.top_k,
                "route": result.get("route"),
                "st_gate": result.get("st_gate"),
                "search_queries": result.get("search_queries", []),
            },
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ─── Helpers ───


def _to_retrieved_docs(results):
    docs: list[RetrievedDoc] = []
    for result in results or []:
        metadata = getattr(result, "metadata", None)
        title = ""
        if metadata:
            # Try title first, then doc_description (used by myservice/gcb)
            title = metadata.get("title", "") or metadata.get("doc_description", "")
        if not title:
            title = (result.raw_text or result.content or "").split("\n")[0][:80]

        # Use full expanded content (raw_text) or original content without truncation
        snippet = result.raw_text or result.content or ""

        score = getattr(result, "score", 0.0) or 0.0
        score_percent = int(score * 100) if score <= 1 else int(min(score, 100))

        # Extract page from metadata for image URL
        doc_id = getattr(result, "doc_id", "")
        page = None
        page_image_url = None
        expanded_pages: List[int] | None = None
        expanded_page_urls: List[str] | None = None
        if metadata:
            page = metadata.get("page_start") or metadata.get("page")
            if isinstance(page, int) and doc_id:
                page_image_url = f"/api/assets/docs/{doc_id}/pages/{page}"
            exp_pages = metadata.get("expanded_pages")
            if exp_pages and isinstance(exp_pages, list):
                collected: List[int] = []
                for p in exp_pages:
                    try:
                        page_num = int(p)
                    except (TypeError, ValueError):
                        continue
                    if page_num < 0:
                        continue
                    collected.append(page_num)
                if collected:
                    expanded_pages = sorted(set(collected))

        if expanded_pages is None and isinstance(page, int):
            expanded_pages = [page]

        if expanded_pages and doc_id:
            expanded_page_urls = [
                f"/api/assets/docs/{doc_id}/pages/{p}" for p in expanded_pages
            ]
            if page is None:
                page = expanded_pages[0]
            if not page_image_url:
                page_image_url = expanded_page_urls[0]

        docs.append(
            RetrievedDoc(
                id=doc_id,
                title=title,
                snippet=snippet,
                score=score,
                score_percent=score_percent,
                page=page,
                page_image_url=page_image_url,
                expanded_pages=expanded_pages,
                expanded_page_urls=expanded_page_urls,
            )
        )
    return docs


def _generate_follow_ups(query: str) -> List[str]:
    """간단한 후속 질문 생성 (추후 LLM 대체 가능)."""
    return [
        f"{query}에 대한 예시를 더 보여줘",
        f"{query}를 실제 업무에 적용하는 방법은?",
        f"{query} 관련 주의사항이 있어?",
    ]


__all__ = [
    "router",
    "ChatRequest",
    "ChatResponse",
    "HistoryMessage",
    "RetrievedDoc",
]
