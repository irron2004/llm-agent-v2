"""LangGraph RAG Agent API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import (
    get_default_llm,
    get_prompt_spec_cached,
    get_search_service,
)
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent
from backend.services.search_service import SearchService


router = APIRouter(prefix="/agent", tags=["LangGraph Agent"])


class AgentRequest(BaseModel):
    message: str = Field(..., description="사용자 질문")
    top_k: int = Field(3, ge=1, le=50, description="검색 상위 문서 수 (기본 3)")
    max_attempts: int = Field(1, ge=0, le=3, description="judge 실패 시 재시도 횟수")
    mode: str = Field("verified", description="base 또는 verified")
    thread_id: Optional[str] = Field(None, description="LangGraph thread_id (checkpoint)")


class RetrievedDoc(BaseModel):
    id: str
    title: str
    snippet: str
    score: float | None = None
    score_percent: int | None = None
    metadata: Dict[str, Any] | None = None


class AgentResponse(BaseModel):
    query: str
    answer: str
    judge: Dict[str, Any]
    retrieved_docs: List[RetrievedDoc]
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _to_retrieved_docs(results: List[RetrievalResult]) -> List[RetrievedDoc]:
    docs: List[RetrievedDoc] = []
    for r in results or []:
        title = ""
        if getattr(r, "metadata", None):
            title = r.metadata.get("title", "")
        if not title:
            title = (r.raw_text or r.content or "").split("\n")[0][:80]

        snippet_source = r.raw_text or r.content or ""
        snippet = snippet_source[:400] + ("..." if len(snippet_source) > 400 else "")

        score = getattr(r, "score", None)
        score_percent = None
        if score is not None:
            score_percent = int(score * 100) if score <= 1 else int(min(score, 100))

        docs.append(
            RetrievedDoc(
                id=getattr(r, "doc_id", ""),
                title=title,
                snippet=snippet,
                score=score,
                score_percent=score_percent,
                metadata=getattr(r, "metadata", None),
            )
        )
    return docs


@router.post("/run", response_model=AgentResponse)
async def run_agent(
    req: AgentRequest,
    search_service: SearchService = Depends(get_search_service),
    llm=Depends(get_default_llm),
    prompt_spec=Depends(get_prompt_spec_cached),
):
    """LangGraph RAG 에이전트 실행."""
    if not hasattr(search_service, "search"):
        raise HTTPException(status_code=503, detail="Search service not configured")

    try:
        agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=prompt_spec,
            top_k=req.top_k,
            mode=req.mode,
        )

        result = agent.run(
            req.message,
            attempts=0,
            max_attempts=req.max_attempts,
            thread_id=req.thread_id,
        )

        retrieved_docs = _to_retrieved_docs(result.get("docs", []))

        return AgentResponse(
            query=req.message,
            answer=result.get("answer", ""),
            judge=result.get("judge", {}),
            retrieved_docs=retrieved_docs,
            metadata={
                "route": result.get("route"),
                "st_gate": result.get("st_gate"),
                "search_queries": result.get("search_queries", []),
                "attempts": result.get("attempts"),
                "max_attempts": req.max_attempts,
            },
        )
    except Exception as exc:
        # FastAPI가 스택을 로깅하도록 그대로 re-raise
        raise


__all__ = ["router", "AgentRequest", "AgentResponse"]
