"""LangGraph RAG Agent API."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from backend.api.dependencies import (
    get_default_llm,
    get_prompt_spec_cached,
    get_search_service,
)
from backend.domain.doc_type_mapping import group_doc_type_buckets
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent
from backend.services.search_service import SearchService


router = APIRouter(prefix="/agent", tags=["LangGraph Agent"])
logger = logging.getLogger(__name__)

# =============================================================================
# 전역 상태: HIL(Human-in-the-Loop) 지원 및 Auto-Parse 모드
# =============================================================================
# 핵심: interrupt/resume가 동작하려면 동일한 graph 인스턴스와 checkpointer 필요
_checkpointer: MemorySaver = MemorySaver()
_hil_agent: Optional[LangGraphRAGAgent] = None
_auto_parse_agent: Optional[LangGraphRAGAgent] = None


def _create_device_fetcher(search_service):
    """Create a device/doc_type fetcher function using ES aggregation."""
    def _fetch_devices() -> Dict[str, Any] | list[Dict[str, Any]]:
        # Check if ES engine is available
        if not hasattr(search_service, 'es_engine') or search_service.es_engine is None:
            logger.warning("ES engine not available for device fetching")
            return []

        es = search_service.es_engine.es
        index = search_service.es_engine.index_name

        agg_query = {
            "size": 0,
            "aggs": {
                "devices": {
                    "terms": {
                        "field": "device_name",
                        "size": 200,
                        "order": {"_count": "desc"},
                    },
                    "aggs": {
                        "unique_docs": {
                            "cardinality": {
                                "script": {
                                    "lang": "painless",
                                    "source": (
                                        "def v = doc.containsKey(params.f) && !doc[params.f].empty ? doc[params.f].value : null;"
                                        "if (v == null) return null;"
                                        "int idx = v.indexOf('#');"
                                        "if (idx == -1) idx = v.indexOf(':');"
                                        "return idx > 0 ? v.substring(0, idx) : v;"
                                    ),
                                    "params": {"f": "doc_id"},
                                }
                            }
                        }
                    }
                },
                "doc_types": {
                    "terms": {
                        "field": "doc_type",
                        "size": 12,
                        "order": {"_count": "desc"},
                    },
                    "aggs": {
                        "unique_docs": {
                            "cardinality": {
                                "script": {
                                    "lang": "painless",
                                    "source": (
                                        "def v = doc.containsKey(params.f) && !doc[params.f].empty ? doc[params.f].value : null;"
                                        "if (v == null) return null;"
                                        "int idx = v.indexOf('#');"
                                        "if (idx == -1) idx = v.indexOf(':');"
                                        "return idx > 0 ? v.substring(0, idx) : v;"
                                    ),
                                    "params": {"f": "doc_id"},
                                }
                            }
                        }
                    }
                },
            }
        }

        try:
            result = es.search(index=index, body=agg_query)
            device_buckets = result.get("aggregations", {}).get("devices", {}).get("buckets", [])
            doc_type_buckets = result.get("aggregations", {}).get("doc_types", {}).get("buckets", [])
            devices = [
                {"name": bucket["key"], "doc_count": bucket.get("unique_docs", {}).get("value", bucket["doc_count"])}
                for bucket in device_buckets
                if bucket.get("key")
            ]
            doc_types = group_doc_type_buckets(doc_type_buckets, use_unique_docs=True)
            return {"devices": devices, "doc_types": doc_types}
        except Exception as e:
            logger.error(f"Failed to fetch device list: {e}")
            return []

    return _fetch_devices


def _build_state_overrides(req: "AgentRequest") -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    if req.filter_devices is not None:
        devices = [str(d).strip() for d in req.filter_devices if str(d).strip()]
        if devices:
            overrides["selected_devices"] = devices

    if req.filter_doc_types is not None:
        doc_types = [str(d).strip() for d in req.filter_doc_types if str(d).strip()]
        if doc_types:
            overrides["selected_doc_types"] = doc_types

    if req.search_queries is not None:
        queries = [str(q).strip() for q in req.search_queries if str(q).strip()]
        if queries:
            overrides["search_queries"] = queries
            overrides["skip_mq"] = True

    if req.selected_doc_ids is not None:
        doc_ids = [str(d).strip() for d in req.selected_doc_ids if str(d).strip()]
        if doc_ids:
            overrides["selected_doc_ids"] = doc_ids

    return overrides

def _get_hil_agent(llm, search_service, prompt_spec) -> LangGraphRAGAgent:
    """HIL용 싱글톤 에이전트. 동일한 graph 인스턴스로 interrupt/resume 보장."""
    global _hil_agent
    if _hil_agent is None:
        logger.info("Creating HIL agent singleton with top_k=20 retrieval_top_k=50")
        _hil_agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=prompt_spec,
            top_k=20,
            retrieval_top_k=50,
            mode="verified",
            ask_user_after_retrieve=True,
            ask_device_selection=True,
            device_fetcher=_create_device_fetcher(search_service),
            checkpointer=_checkpointer,
        )
    return _hil_agent


def _get_auto_parse_agent(llm, search_service, prompt_spec) -> LangGraphRAGAgent:
    """Auto-parse용 싱글톤 에이전트. 장비/문서종류를 자동으로 파싱."""
    global _auto_parse_agent
    if _auto_parse_agent is None:
        logger.info("Creating Auto-parse agent singleton with top_k=20 retrieval_top_k=50")
        _auto_parse_agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=prompt_spec,
            top_k=20,
            retrieval_top_k=50,
            mode="verified",
            ask_user_after_retrieve=False,  # 문서 선택 UI 비활성화
            ask_device_selection=False,      # 기기 선택 UI 비활성화
            auto_parse_enabled=True,         # 자동 파싱 활성화
            checkpointer=_checkpointer,
        )
    return _auto_parse_agent


# =============================================================================
# Request/Response Models
# =============================================================================
class AgentRequest(BaseModel):
    message: str = Field(..., description="사용자 질문")
    top_k: int = Field(10, ge=1, le=50, description="검색 상위 문서 수")
    max_attempts: int = Field(3, ge=0, le=3, description="judge 실패 시 재시도 횟수")
    mode: str = Field("verified", description="base 또는 verified")
    thread_id: Optional[str] = Field(None, description="LangGraph thread_id")
    ask_user_after_retrieve: bool = Field(False, description="검색 후 사용자 확인")
    resume_decision: Optional[Any] = Field(None, description="interrupt 재개 응답")
    auto_parse: bool = Field(True, description="자동 장비/문서종류 파싱 (기본값: True)")
    # 재생성 시 사용할 필터 오버라이드
    filter_devices: Optional[List[str]] = Field(None, description="장비 필터 오버라이드")
    filter_doc_types: Optional[List[str]] = Field(None, description="문서종류 필터 오버라이드")
    search_queries: Optional[List[str]] = Field(None, description="검색 쿼리 오버라이드 (MQ 수정)")
    selected_doc_ids: Optional[List[str]] = Field(None, description="사용할 문서 ID 선택 (재생성)")


class RetrievedDoc(BaseModel):
    id: str
    title: str
    snippet: str
    score: float | None = None
    score_percent: int | None = None
    metadata: Dict[str, Any] | None = None
    page: int | None = None
    page_image_url: str | None = None
    expanded_pages: List[int] | None = None
    expanded_page_urls: List[str] | None = None


class ExpandedDoc(BaseModel):
    """확장된 문서 정보 (답변 생성에 사용된 컨텍스트)"""
    rank: int
    doc_id: str
    content: str  # 확장된 전체 내용
    content_length: int  # 내용 길이


class AutoParseResult(BaseModel):
    """자동 파싱 결과"""
    device: Optional[str] = Field(None, description="파싱된 장비명")
    doc_type: Optional[str] = Field(None, description="파싱된 문서종류")
    devices: Optional[List[str]] = Field(None, description="파싱된 장비명 목록")
    doc_types: Optional[List[str]] = Field(None, description="파싱된 문서종류 목록")
    language: Optional[str] = Field(None, description="감지된 언어 (ko, en, ja)")
    message: Optional[str] = Field(None, description="사용자에게 표시할 메시지")


class AgentResponse(BaseModel):
    query: str
    answer: str
    reasoning: Optional[str] = Field(None, description="LLM reasoning 과정 (reasoning 모델인 경우)")
    judge: Dict[str, Any]
    retrieved_docs: List[RetrievedDoc]
    all_retrieved_docs: Optional[List[RetrievedDoc]] = Field(None, description="전체 검색 문서 (재생성용, 20개)")
    expanded_docs: Optional[List[ExpandedDoc]] = Field(None, description="확장된 문서 (답변 생성용)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    interrupted: bool = Field(False)
    interrupt_payload: Optional[Dict[str, Any]] = Field(None)
    thread_id: Optional[str] = Field(None)
    # Auto-parse results
    auto_parse: Optional[AutoParseResult] = Field(None, description="자동 파싱 결과")
    # Filter info for regeneration
    selected_devices: Optional[List[str]] = Field(None, description="사용된 장비 필터")
    selected_doc_types: Optional[List[str]] = Field(None, description="사용된 문서종류 필터")
    search_queries: Optional[List[str]] = Field(None, description="사용된 검색 쿼리 (MQ)")
    # Language detection
    detected_language: Optional[str] = Field(None, description="감지된 언어 (ko, en, ja)")


def _to_expanded_docs(answer_ref_json: List[Dict[str, Any]] | None) -> List[ExpandedDoc] | None:
    """answer_ref_json을 ExpandedDoc 리스트로 변환."""
    if not answer_ref_json:
        return None
    docs: List[ExpandedDoc] = []
    for ref in answer_ref_json:
        content = ref.get("content", "")
        docs.append(ExpandedDoc(
            rank=ref.get("rank", 0),
            doc_id=ref.get("doc_id", ""),
            content=content,
            content_length=len(content),
        ))
    return docs if docs else None


def _select_display_docs(result: Dict[str, Any]) -> List[RetrievalResult]:
    """Prefer UI display docs from expansion, fallback to full docs list."""
    if "display_docs" in result:
        return result.get("display_docs") or []
    return result.get("docs", []) or []


def _to_retrieved_docs(results: List[RetrievalResult]) -> List[RetrievedDoc]:
    docs: List[RetrievedDoc] = []
    for r in results or []:
        title = ""
        metadata = getattr(r, "metadata", None)
        if metadata:
            # Try title first, then doc_description (used by myservice/gcb)
            title = metadata.get("title", "") or metadata.get("doc_description", "")
        if not title:
            title = (r.raw_text or r.content or "").split("\n")[0][:80]

        # Use full expanded content (raw_text) or original content without truncation
        snippet = r.raw_text or r.content or ""

        score = getattr(r, "score", None)
        score_percent = int(score * 100) if score and score <= 1 else None

        # Extract page from metadata for image URL
        doc_id = getattr(r, "doc_id", "")
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

        docs.append(RetrievedDoc(
            id=doc_id,
            title=title,
            snippet=snippet,
            score=score,
            score_percent=score_percent,
            metadata=metadata,
            page=page,
            page_image_url=page_image_url,
            expanded_pages=expanded_pages,
            expanded_page_urls=expanded_page_urls,
        ))
    return docs


# =============================================================================
# API Endpoint
# =============================================================================
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

    tid = req.thread_id or str(uuid.uuid4())
    is_resume = req.resume_decision is not None and req.thread_id is not None

    state_overrides = _build_state_overrides(req)
    has_overrides = bool(state_overrides)

    try:
        # Auto-parse 모드 (기본값: True), skip when overrides are provided
        if req.auto_parse and not is_resume and not req.ask_user_after_retrieve and not has_overrides:
            agent = _get_auto_parse_agent(llm, search_service, prompt_spec)
            result = agent.run(req.message, attempts=0, max_attempts=req.max_attempts, thread_id=tid)
        elif has_overrides and not is_resume:
            # Regeneration with filter/query/doc overrides
            agent = LangGraphRAGAgent(
                llm=llm,
                search_service=search_service,
                prompt_spec=prompt_spec,
                top_k=req.top_k,
                mode=req.mode,
                ask_user_after_retrieve=False,
                checkpointer=None,
            )
            result = agent.run(
                req.message,
                attempts=0,
                max_attempts=req.max_attempts,
                thread_id=tid,
                state_overrides=state_overrides or None,
            )
        # HIL 모드 (ask_user 또는 resume)
        elif req.ask_user_after_retrieve or is_resume:
            agent = _get_hil_agent(llm, search_service, prompt_spec)
            config = {"configurable": {"thread_id": tid}}

            if is_resume:
                # 체크포인트 확인
                state = agent._graph.get_state(config)
                logger.info(f"[resume] tid={tid}, state exists={state is not None}")

                if state is None or not state.values:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No checkpoint for thread_id={tid}. Server may have restarted."
                    )

                logger.info(f"[resume] next={state.next}, keys={list(state.values.keys())}")
                result = agent._graph.invoke(Command(resume=req.resume_decision), config)
            else:
                result = agent.run(
                    req.message,
                    attempts=0,
                    max_attempts=req.max_attempts,
                    thread_id=tid,
                    state_overrides=state_overrides or None,
                )
        else:
            # 일반 모드: 체크포인터 없이 새 에이전트
            agent = LangGraphRAGAgent(
                llm=llm,
                search_service=search_service,
                prompt_spec=prompt_spec,
                top_k=req.top_k,
                mode=req.mode,
                ask_user_after_retrieve=False,
                checkpointer=None,
            )
            result = agent.run(
                req.message,
                attempts=0,
                max_attempts=req.max_attempts,
                thread_id=tid,
                state_overrides=state_overrides or None,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Extract auto_parse results
    auto_parse_result = None
    if result.get("auto_parsed_device") or result.get("auto_parsed_doc_type") or result.get("auto_parsed_devices") or result.get("auto_parsed_doc_types") or result.get("auto_parse_message") or result.get("detected_language"):
        auto_parse_result = AutoParseResult(
            device=result.get("auto_parsed_device"),
            doc_type=result.get("auto_parsed_doc_type"),
            devices=result.get("auto_parsed_devices"),
            doc_types=result.get("auto_parsed_doc_types"),
            language=result.get("detected_language"),
            message=result.get("auto_parse_message"),
        )

    # Interrupt 체크
    interrupt_info = result.get("__interrupt__")
    if interrupt_info:
        payload = None
        if len(interrupt_info) > 0:
            payload = interrupt_info[0].value if hasattr(interrupt_info[0], "value") else None

        return AgentResponse(
            query=req.message,
            answer=result.get("answer", "") or "",
            judge=result.get("judge", {}) or {},
            retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
            all_retrieved_docs=_to_retrieved_docs(result.get("docs", [])),
            expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
            metadata={
                "route": result.get("route"),
                "st_gate": result.get("st_gate"),
                "search_queries": result.get("search_queries", []),
                "selected_device": result.get("selected_device"),
            },
            interrupted=True,
            interrupt_payload=payload,
            thread_id=tid,
            auto_parse=auto_parse_result,
            selected_devices=result.get("selected_devices"),
            selected_doc_types=result.get("selected_doc_types"),
            search_queries=result.get("search_queries"),
            detected_language=result.get("detected_language"),
        )

    # 정상 완료
    return AgentResponse(
        query=req.message,
        answer=result.get("answer", ""),
        judge=result.get("judge", {}),
        retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
        all_retrieved_docs=_to_retrieved_docs(result.get("docs", [])),
        expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
        metadata={
            "route": result.get("route"),
            "st_gate": result.get("st_gate"),
            "search_queries": result.get("search_queries", []),
            "selected_device": result.get("selected_device"),
        },
        interrupted=False,
        thread_id=tid,
        auto_parse=auto_parse_result,
        selected_devices=result.get("selected_devices"),
        selected_doc_types=result.get("selected_doc_types"),
        search_queries=result.get("search_queries"),
        detected_language=result.get("detected_language"),
    )


@router.post("/run/stream")
async def run_agent_stream(
    req: AgentRequest,
    request: Request,
    search_service: SearchService = Depends(get_search_service),
    llm=Depends(get_default_llm),
    prompt_spec=Depends(get_prompt_spec_cached),
):
    """LangGraph RAG 에이전트 실행 (SSE: 노드 실행 로그 스트리밍)."""
    if not hasattr(search_service, "search"):
        raise HTTPException(status_code=503, detail="Search service not configured")

    tid = req.thread_id or str(uuid.uuid4())
    is_resume = req.resume_decision is not None and req.thread_id is not None

    state_overrides = _build_state_overrides(req)
    has_overrides = bool(state_overrides)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

    def _enqueue(event: Dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False)

        def _put() -> None:
            # Best-effort: do not block the graph thread if the client is slow.
            if not queue.full():
                queue.put_nowait(payload)

        loop.call_soon_threadsafe(_put)

    def _close() -> None:
        loop.call_soon_threadsafe(queue.put_nowait, None)

    def _worker() -> None:
        try:
            # Auto-parse 모드 (기본값: True), skip when overrides are provided
            if req.auto_parse and not is_resume and not req.ask_user_after_retrieve and not has_overrides:
                agent = _get_auto_parse_agent(llm, search_service, prompt_spec)
                agent._event_sink = _enqueue  # type: ignore[attr-defined]
                result = agent.run(req.message, attempts=0, max_attempts=req.max_attempts, thread_id=tid)
            elif has_overrides and not is_resume:
                # Regeneration with filter/query/doc overrides
                agent = LangGraphRAGAgent(
                    llm=llm,
                    search_service=search_service,
                    prompt_spec=prompt_spec,
                    top_k=req.top_k,
                    mode=req.mode,
                    ask_user_after_retrieve=False,
                    checkpointer=None,
                    event_sink=_enqueue,
                )
                result = agent.run(
                    req.message,
                    attempts=0,
                    max_attempts=req.max_attempts,
                    thread_id=tid,
                    state_overrides=state_overrides or None,
                )
            elif req.ask_user_after_retrieve or is_resume:
                # NOTE: HIL resume requires shared graph/checkpointer; reuse singleton.
                agent = _get_hil_agent(llm, search_service, prompt_spec)
                # Attach per-request sink (best-effort; concurrent streams may interleave).
                agent._event_sink = _enqueue  # type: ignore[attr-defined]
                config = {"configurable": {"thread_id": tid}}

                if is_resume:
                    state = agent._graph.get_state(config)
                    if state is None or not state.values:
                        _enqueue({
                            "type": "error",
                            "status": 400,
                            "detail": f"No checkpoint for thread_id={tid}. Server may have restarted.",
                        })
                        return
                    result = agent._graph.invoke(Command(resume=req.resume_decision), config)
                else:
                    result = agent.run(
                        req.message,
                        attempts=0,
                        max_attempts=req.max_attempts,
                        thread_id=tid,
                        state_overrides=state_overrides or None,
                    )
            else:
                agent = LangGraphRAGAgent(
                    llm=llm,
                    search_service=search_service,
                    prompt_spec=prompt_spec,
                    top_k=req.top_k,
                    mode=req.mode,
                    ask_user_after_retrieve=False,
                    checkpointer=None,
                    event_sink=_enqueue,
                )
                result = agent.run(
                    req.message,
                    attempts=0,
                    max_attempts=req.max_attempts,
                    thread_id=tid,
                    state_overrides=state_overrides or None,
                )

            # Extract auto_parse results
            auto_parse_result = None
            if result.get("auto_parsed_device") or result.get("auto_parsed_doc_type") or result.get("auto_parsed_devices") or result.get("auto_parsed_doc_types") or result.get("auto_parse_message") or result.get("detected_language"):
                auto_parse_result = AutoParseResult(
                    device=result.get("auto_parsed_device"),
                    doc_type=result.get("auto_parsed_doc_type"),
                    devices=result.get("auto_parsed_devices"),
                    doc_types=result.get("auto_parsed_doc_types"),
                    language=result.get("detected_language"),
                    message=result.get("auto_parse_message"),
                )

            interrupt_info = result.get("__interrupt__")
            logger.info(f"[agent stream] result keys: {list(result.keys())}")
            logger.info(f"[agent stream] interrupt_info: {interrupt_info}")
            if interrupt_info:
                payload = None
                if len(interrupt_info) > 0:
                    payload = interrupt_info[0].value if hasattr(interrupt_info[0], "value") else None

                resp = AgentResponse(
                    query=req.message,
                    answer=result.get("answer", "") or "",
                    reasoning=result.get("reasoning"),
                    judge=result.get("judge", {}) or {},
                    retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
                    all_retrieved_docs=_to_retrieved_docs(result.get("docs", [])),
                    expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
                    metadata={
                        "route": result.get("route"),
                        "st_gate": result.get("st_gate"),
                        "search_queries": result.get("search_queries", []),
                        "selected_device": result.get("selected_device"),
                    },
                    interrupted=True,
                    interrupt_payload=payload,
                    thread_id=tid,
                    auto_parse=auto_parse_result,
                    selected_devices=result.get("selected_devices"),
                    selected_doc_types=result.get("selected_doc_types"),
                    search_queries=result.get("search_queries"),
                    detected_language=result.get("detected_language"),
                )
            else:
                resp = AgentResponse(
                    query=req.message,
                    answer=result.get("answer", ""),
                    reasoning=result.get("reasoning"),
                    judge=result.get("judge", {}),
                    retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
                    all_retrieved_docs=_to_retrieved_docs(result.get("docs", [])),
                    expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
                    metadata={
                        "route": result.get("route"),
                        "st_gate": result.get("st_gate"),
                        "search_queries": result.get("search_queries", []),
                        "selected_device": result.get("selected_device"),
                    },
                    interrupted=False,
                    thread_id=tid,
                    auto_parse=auto_parse_result,
                    selected_devices=result.get("selected_devices"),
                    selected_doc_types=result.get("selected_doc_types"),
                    search_queries=result.get("search_queries"),
                    detected_language=result.get("detected_language"),
                )

            _enqueue({"type": "final", "result": resp.model_dump()})

        except RuntimeError as exc:
            _enqueue({"type": "error", "status": 503, "detail": str(exc)})
        except Exception as exc:
            _enqueue({"type": "error", "status": 500, "detail": str(exc)})
        finally:
            _close()

    asyncio.create_task(asyncio.to_thread(_worker))

    async def _gen():
        # Initial handshake event
        yield f"data: {json.dumps({'type': 'open', 'thread_id': tid}, ensure_ascii=False)}\n\n"

        while True:
            if await request.is_disconnected():
                break
            item = await queue.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Hint for nginx reverse proxies to disable response buffering for SSE.
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router", "AgentRequest", "AgentResponse"]
