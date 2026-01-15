"""Chat Service API (LLM 단독 / RAG 기반 두 엔드포인트 제공)."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.api.dependencies import (
    get_chat_service,
    get_rag_service,
    get_simple_chat_prompt,
)
from backend.services.chat_service import ChatService
from backend.services.rag_service import RAGService

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
    top_k: int = Field(default=3, ge=1, description="검색할 문서 수")


class RetrievedDoc(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    score_percent: int
    page: int | None = None
    page_image_url: str | None = None


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
    rag_service: RAGService = Depends(get_rag_service),
):
    """RAG 기반 질문 응답 API (검색 + 답변)."""
    try:
        history_payload = [msg.model_dump() for msg in req.history]
        rag_response = rag_service.query(
            req.message,
            top_k=req.top_k,
            history=history_payload,
        )

        retrieved_docs = _to_retrieved_docs(rag_response.context)
        follow_ups = _generate_follow_ups(req.message)

        clean_query = (
            rag_response.metadata.get("preprocessed_query", req.message)
            if rag_response.metadata
            else req.message
        )

        return ChatResponse(
            query=req.message,
            clean_query=clean_query,
            answer=rag_response.answer,
            retrieved_docs=retrieved_docs,
            follow_ups=follow_ups,
            metadata={
                "num_results": len(retrieved_docs),
                "top_k": req.top_k,
                "history_len": len(history_payload),
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
            title = metadata.get("title", "")
        if not title:
            title = (result.raw_text or result.content or "").split("\n")[0][:50]

        snippet_source = result.raw_text or result.content or ""
        snippet = snippet_source[:200] + ("..." if len(snippet_source) > 200 else "")

        score = getattr(result, "score", 0.0) or 0.0
        score_percent = int(score * 100) if score <= 1 else int(min(score, 100))

        # Extract page from metadata for image URL
        doc_id = getattr(result, "doc_id", "")
        page = None
        page_image_url = None
        if metadata:
            page = metadata.get("page_start") or metadata.get("page")
            if isinstance(page, int) and doc_id:
                page_image_url = f"/api/assets/docs/{doc_id}/pages/{page}"

        docs.append(
            RetrievedDoc(
                id=doc_id,
                title=title,
                snippet=snippet,
                score=score,
                score_percent=score_percent,
                page=page,
                page_image_url=page_image_url,
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
