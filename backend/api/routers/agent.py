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
from backend.llm_infrastructure.llm.langgraph_agent import ChatHistoryEntry
from backend.services.agents.langgraph_rag_agent import (
    LangGraphRAGAgent,
    reset_event_sink_context,
    set_event_sink_context,
)
from backend.services.chat_history_service import ChatHistoryService
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
_device_selection_agent: Optional[LangGraphRAGAgent] = None


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


def _summarize_answer(answer: str, max_length: int = 150) -> str:
    """답변을 요약 (truncate 방식)."""
    if not answer:
        return ""
    if len(answer) <= max_length:
        return answer
    # 단어 중간 잘림 방지 (공백 기준)
    truncated = answer[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    return truncated + "..."


def _extract_refs_from_result(result: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """결과에서 참조 문서 title과 doc_id 추출 (rank 순)."""
    refs: List[str] = []
    doc_ids: List[str] = []
    seen_doc_ids: set = set()

    # answer_ref_json에서 추출 (확장된 문서, rank 순)
    answer_ref_json = result.get("answer_ref_json", [])
    for ref in answer_ref_json:
        doc_id = ref.get("doc_id", "")
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        doc_ids.append(doc_id)
        # title 추출: metadata에서 title 또는 doc_description
        title = ""
        metadata = ref.get("metadata", {})
        if metadata:
            title = metadata.get("title", "") or metadata.get("doc_description", "")
        if not title:
            # content 첫 줄에서 추출
            content = ref.get("content", "")
            if content:
                title = content.split("\n")[0][:80]
        refs.append(title or doc_id)

    # fallback: docs에서 추출
    if not doc_ids:
        docs = result.get("docs", [])
        for doc in docs[:5]:  # 상위 5개만
            doc_id = getattr(doc, "doc_id", "")
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            doc_ids.append(doc_id)
            # title 추출
            title = ""
            metadata = getattr(doc, "metadata", None)
            if metadata:
                title = metadata.get("title", "") or metadata.get("doc_description", "")
            if not title:
                content = getattr(doc, "content", "") or getattr(doc, "raw_text", "")
                if content:
                    title = content.split("\n")[0][:80]
            refs.append(title or doc_id)

    return refs, doc_ids


def _load_chat_history_from_session(session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """session_id로 DB에서 chat_history 로드 (ChatTurn → ChatHistoryEntry 변환).

    Args:
        session_id: 대화 세션 ID
        limit: 최근 N턴만 로드 (기본 5턴)

    Returns:
        ChatHistoryEntry 형식의 리스트
    """
    try:
        svc = ChatHistoryService.from_settings()
        turns = svc.get_session(session_id)
        if not turns:
            logger.info("[_load_chat_history_from_session] No turns found for session_id=%s", session_id)
            return []

        # 최근 N턴만 사용 (오래된 턴은 제외)
        recent_turns = turns[-limit:] if len(turns) > limit else turns

        chat_history: List[Dict[str, Any]] = []
        for turn in recent_turns:
            # User entry
            chat_history.append({
                "role": "user",
                "content": turn.user_text,
            })
            # Assistant entry
            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "summary": _summarize_answer(turn.assistant_text) if turn.assistant_text else "",
            }
            # doc_refs에서 doc_ids 추출
            if turn.doc_refs:
                doc_ids = [ref.doc_id for ref in turn.doc_refs if ref.doc_id]
                refs = [ref.title for ref in turn.doc_refs if ref.title]
                if doc_ids:
                    assistant_entry["doc_ids"] = doc_ids
                if refs:
                    assistant_entry["refs"] = refs
            chat_history.append(assistant_entry)

        logger.info(
            "[_load_chat_history_from_session] Loaded %d entries from session_id=%s",
            len(chat_history), session_id
        )
        return chat_history

    except Exception as e:
        logger.warning("[_load_chat_history_from_session] Failed to load history: %s", e)
        return []


def _build_state_overrides(req: "AgentRequest") -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    if req.filter_devices is not None:
        devices = [str(d).strip() for d in req.filter_devices if str(d).strip()]
        if devices:
            overrides["selected_devices"] = devices
            # filter_devices가 명시적으로 제공되면 auto_parse 건너뛰기
            overrides["skip_auto_parse"] = True

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

    # Chat history: session_id 우선, 없으면 클라이언트 전달값 사용
    if req.session_id:
        # BE에서 DB 조회하여 history 로드
        chat_history = _load_chat_history_from_session(req.session_id)
        if chat_history:
            overrides["chat_history"] = chat_history
    elif req.chat_history is not None:
        # Fallback: 클라이언트에서 전달한 history 사용
        chat_history = [entry.model_dump(exclude_none=True) for entry in req.chat_history]
        if chat_history:
            overrides["chat_history"] = chat_history

    return overrides


def _save_turn_to_history(
    session_id: str,
    user_text: str,
    assistant_text: str,
    ref_doc_ids: List[str] | None = None,
    refs: List[str] | None = None,
) -> None:
    """Agent 응답 후 ChatHistoryService에 턴 저장.

    Args:
        session_id: 대화 세션 ID
        user_text: 사용자 질문
        assistant_text: 어시스턴트 응답
        ref_doc_ids: 참조 문서 ID 목록
        refs: 참조 문서 제목 목록
    """
    if not session_id:
        return

    try:
        from backend.services.chat_history_service import ChatTurn, DocRef

        svc = ChatHistoryService.from_settings()

        # DocRef 구성
        doc_refs: List[DocRef] = []
        if ref_doc_ids:
            for i, doc_id in enumerate(ref_doc_ids):
                title = refs[i] if refs and i < len(refs) else ""
                doc_refs.append(DocRef(
                    slot=i + 1,
                    doc_id=doc_id,
                    title=title,
                    snippet="",
                ))

        # 턴 저장
        turn_id = svc.get_next_turn_id(session_id)
        turn = ChatTurn(
            session_id=session_id,
            turn_id=turn_id,
            user_text=user_text,
            assistant_text=assistant_text,
            doc_refs=doc_refs,
        )
        svc.save_turn(turn)
        logger.info("[_save_turn_to_history] Saved turn %d for session %s", turn_id, session_id)

    except Exception as e:
        logger.warning("[_save_turn_to_history] Failed to save turn: %s", e)


def _get_hil_agent(llm, search_service, prompt_spec) -> LangGraphRAGAgent:
    """HIL용 싱글톤 에이전트. 동일한 graph 인스턴스로 interrupt/resume 보장."""
    global _hil_agent
    if _hil_agent is None:
        logger.info("Creating HIL agent singleton with top_k=20 retrieval_top_k=100")
        _hil_agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=prompt_spec,
            top_k=20,
            retrieval_top_k=100,
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
        logger.info("Creating Auto-parse agent singleton with top_k=20 retrieval_top_k=100")
        _auto_parse_agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=prompt_spec,
            top_k=20,
            retrieval_top_k=100,
            mode="verified",
            ask_user_after_retrieve=False,  # 문서 선택 UI 비활성화
            ask_device_selection=False,      # 기기 선택 UI 비활성화
            auto_parse_enabled=True,         # 자동 파싱 활성화
            checkpointer=_checkpointer,
        )
    return _auto_parse_agent


def _get_device_selection_agent(llm, search_service, prompt_spec) -> LangGraphRAGAgent:
    """Device selection용 싱글톤 에이전트. 기기/문서종류 선택만 수행."""
    global _device_selection_agent
    if _device_selection_agent is None:
        logger.info("Creating device-selection agent singleton with top_k=20 retrieval_top_k=100")
        _device_selection_agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=prompt_spec,
            top_k=20,
            retrieval_top_k=100,
            mode="verified",
            ask_user_after_retrieve=False,
            ask_device_selection=True,
            device_fetcher=_create_device_fetcher(search_service),
            checkpointer=_checkpointer,
        )
    return _device_selection_agent


# =============================================================================
# Request/Response Models
# =============================================================================
class ChatHistoryEntryModel(BaseModel):
    """대화 히스토리 항목 (API 스키마)."""

    role: str = Field(..., description="역할: user 또는 assistant")
    # user용 필드
    content: Optional[str] = Field(None, description="user 원본 질문")
    # assistant용 필드
    summary: Optional[str] = Field(None, description="assistant 답변 요약 (150자)")
    refs: Optional[List[str]] = Field(None, description="참조 문서 title")
    doc_ids: Optional[List[str]] = Field(None, description="참조 문서 ID (rank 순)")


class AgentRequest(BaseModel):
    message: str = Field(..., description="사용자 질문")
    top_k: int = Field(20, ge=1, le=50, description="검색 상위 문서 수")
    max_attempts: int = Field(2, ge=0, le=3, description="judge 실패 시 재시도 횟수")
    mode: str = Field("verified", description="base 또는 verified")
    thread_id: Optional[str] = Field(None, description="LangGraph thread_id")
    session_id: Optional[str] = Field(None, description="대화 세션 ID (BE에서 history 자동 로드)")
    ask_user_after_retrieve: bool = Field(False, description="검색 후 사용자 확인")
    ask_device_selection: bool = Field(False, description="검색 전 기기/문서 종류 선택")
    resume_decision: Optional[Any] = Field(None, description="interrupt 재개 응답")
    auto_parse: bool = Field(True, description="자동 장비/문서종류 파싱 (기본값: True)")
    # 대화 히스토리 (클라이언트 전달 방식 - session_id 없을 때 fallback)
    chat_history: Optional[List[ChatHistoryEntryModel]] = Field(
        None, description="대화 히스토리 (최근 5턴, session_id가 없을 때 사용)"
    )
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


class SuggestedDevice(BaseModel):
    """추천 장비 (검색 결과 기반)"""
    name: str = Field(..., description="장비명")
    count: int = Field(..., description="검색된 문서 수")


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
    # Chat history support (다음 턴 history 구성용)
    summary: Optional[str] = Field(None, description="답변 요약 (다음 턴 history용, 150자)")
    refs: Optional[List[str]] = Field(None, description="참조 문서 title (rank 순)")
    ref_doc_ids: Optional[List[str]] = Field(None, description="참조 문서 ID (rank 순)")
    # Device suggestion (장비 미지정 시 추천)
    suggested_devices: Optional[List[SuggestedDevice]] = Field(
        None, description="추천 장비 목록 (장비 미지정 시, count 내림차순)"
    )


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


def _to_suggested_devices(devices: List[Dict[str, Any]] | None) -> List[SuggestedDevice] | None:
    """suggested_devices dict 리스트를 SuggestedDevice 리스트로 변환."""
    if not devices:
        return None
    return [
        SuggestedDevice(name=d.get("name", ""), count=d.get("count", 0))
        for d in devices
        if d.get("name")
    ] or None


def _collect_suggested_devices_from_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    """검색 결과에서 device_name 집계 (count 내림차순)."""
    from collections import Counter

    EXCLUDE_NAMES = {"", "ALL", "etc", "ETC", "all", "N/A", "Unknown"}

    device_counts: Counter = Counter()
    for doc in docs or []:
        # RetrievalResult 또는 dict 모두 지원
        if hasattr(doc, "metadata"):
            metadata = doc.metadata
        elif isinstance(doc, dict):
            metadata = doc.get("metadata", {})
        else:
            continue

        device_name = metadata.get("device_name", "") if metadata else ""
        if device_name and device_name.strip() not in EXCLUDE_NAMES:
            device_counts[device_name.strip()] += 1

    return [
        {"name": name, "count": count}
        for name, count in device_counts.most_common()
    ]


def _resolve_suggested_devices(result: Dict[str, Any]) -> List[SuggestedDevice] | None:
    """suggested_devices가 비어있을 때, docs에서 fallback 계산."""
    existing = _to_suggested_devices(result.get("suggested_devices"))
    if existing:
        return existing

    has_device_filter = bool(
        result.get("auto_parsed_device")
        or result.get("auto_parsed_devices")
        or result.get("selected_devices")
        or result.get("selected_device")
    )
    if has_device_filter:
        return None

    docs = result.get("all_docs") or result.get("docs") or result.get("display_docs") or []
    if not docs:
        return None

    return _to_suggested_devices(_collect_suggested_devices_from_docs(docs))


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
        # Auto-parse 모드 (기본값: True), skip when overrides or device selection are provided
        if req.auto_parse and not is_resume and not req.ask_user_after_retrieve and not req.ask_device_selection and not has_overrides:
            agent = _get_auto_parse_agent(llm, search_service, prompt_spec)
            result = agent.run(req.message, attempts=0, max_attempts=req.max_attempts, thread_id=tid)
        elif req.ask_device_selection and not is_resume:
            agent = _get_device_selection_agent(llm, search_service, prompt_spec)
            result = agent.run(
                req.message,
                attempts=0,
                max_attempts=req.max_attempts,
                thread_id=tid,
                state_overrides=state_overrides or None,
            )
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
        elif req.ask_user_after_retrieve or req.ask_device_selection or is_resume:
            if req.ask_device_selection and not req.ask_user_after_retrieve:
                agent = _get_device_selection_agent(llm, search_service, prompt_spec)
            else:
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

    # Extract refs for chat history
    answer = result.get("answer", "")
    summary = _summarize_answer(answer) if answer else None
    refs, ref_doc_ids = _extract_refs_from_result(result)

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
            all_retrieved_docs=_to_retrieved_docs(result.get("all_docs") or result.get("docs", [])),
            expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
            metadata={
                "route": result.get("route"),
                "st_gate": result.get("st_gate"),
                "search_queries": result.get("search_queries", []),
                "selected_device": result.get("selected_device"),
                "is_chat_query": result.get("is_chat_query", False),
                "chat_type": result.get("chat_type"),
            },
            interrupted=True,
            interrupt_payload=payload,
            thread_id=tid,
            auto_parse=auto_parse_result,
            selected_devices=result.get("selected_devices"),
            selected_doc_types=result.get("selected_doc_types"),
            search_queries=result.get("search_queries"),
            detected_language=result.get("detected_language"),
            summary=summary,
            refs=refs if refs else None,
            ref_doc_ids=ref_doc_ids if ref_doc_ids else None,
        )

    # 정상 완료 - 턴 저장
    if req.session_id and answer:
        _save_turn_to_history(
            session_id=req.session_id,
            user_text=req.message,
            assistant_text=answer,
            ref_doc_ids=ref_doc_ids,
            refs=refs,
        )

    return AgentResponse(
        query=req.message,
        answer=result.get("answer", ""),
        judge=result.get("judge", {}),
        retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
        all_retrieved_docs=_to_retrieved_docs(result.get("all_docs") or result.get("docs", [])),
        expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
        metadata={
            "route": result.get("route"),
            "st_gate": result.get("st_gate"),
            "search_queries": result.get("search_queries", []),
            "selected_device": result.get("selected_device"),
            "is_chat_query": result.get("is_chat_query", False),
            "chat_type": result.get("chat_type"),
        },
        interrupted=False,
        thread_id=tid,
        auto_parse=auto_parse_result,
        selected_devices=result.get("selected_devices"),
        selected_doc_types=result.get("selected_doc_types"),
        search_queries=result.get("search_queries"),
        detected_language=result.get("detected_language"),
        summary=summary,
        refs=refs if refs else None,
        ref_doc_ids=ref_doc_ids if ref_doc_ids else None,
        suggested_devices=_resolve_suggested_devices(result),
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
        token = set_event_sink_context(_enqueue)
        try:
            # Auto-parse 모드 (기본값: True), skip when overrides or device selection are provided
            if req.auto_parse and not is_resume and not req.ask_user_after_retrieve and not req.ask_device_selection and not has_overrides:
                agent = _get_auto_parse_agent(llm, search_service, prompt_spec)
                result = agent.run(req.message, attempts=0, max_attempts=req.max_attempts, thread_id=tid)
            elif req.ask_device_selection and not is_resume:
                agent = _get_device_selection_agent(llm, search_service, prompt_spec)
                result = agent.run(
                    req.message,
                    attempts=0,
                    max_attempts=req.max_attempts,
                    thread_id=tid,
                    state_overrides=state_overrides or None,
                )
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
            elif req.ask_user_after_retrieve or req.ask_device_selection or is_resume:
                # NOTE: HIL resume requires shared graph/checkpointer; reuse singleton.
                if req.ask_device_selection and not req.ask_user_after_retrieve:
                    agent = _get_device_selection_agent(llm, search_service, prompt_spec)
                else:
                    agent = _get_hil_agent(llm, search_service, prompt_spec)
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

            # Extract refs for chat history
            stream_answer = result.get("answer", "")
            stream_summary = _summarize_answer(stream_answer) if stream_answer else None
            stream_refs, stream_ref_doc_ids = _extract_refs_from_result(result)

            interrupt_info = result.get("__interrupt__")
            logger.info(f"[agent stream] result keys: {list(result.keys())}")
            logger.info(f"[agent stream] search_queries: {result.get('search_queries')}")
            logger.info(f"[agent stream] all_docs count: {len(result.get('all_docs', []))}")
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
                    all_retrieved_docs=_to_retrieved_docs(result.get("all_docs") or result.get("docs", [])),
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
                    summary=stream_summary,
                    refs=stream_refs if stream_refs else None,
                    ref_doc_ids=stream_ref_doc_ids if stream_ref_doc_ids else None,
                    suggested_devices=_resolve_suggested_devices(result),
                )
            else:
                # 정상 완료 - 턴 저장
                if req.session_id and stream_answer:
                    _save_turn_to_history(
                        session_id=req.session_id,
                        user_text=req.message,
                        assistant_text=stream_answer,
                        ref_doc_ids=stream_ref_doc_ids,
                        refs=stream_refs,
                    )

                resp = AgentResponse(
                    query=req.message,
                    answer=result.get("answer", ""),
                    reasoning=result.get("reasoning"),
                    judge=result.get("judge", {}),
                    retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
                    all_retrieved_docs=_to_retrieved_docs(result.get("all_docs") or result.get("docs", [])),
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
                    summary=stream_summary,
                    refs=stream_refs if stream_refs else None,
                    ref_doc_ids=stream_ref_doc_ids if stream_ref_doc_ids else None,
                    suggested_devices=_resolve_suggested_devices(result),
                )

            _enqueue({"type": "final", "result": resp.model_dump()})

        except RuntimeError as exc:
            _enqueue({"type": "error", "status": 503, "detail": str(exc)})
        except Exception as exc:
            _enqueue({"type": "error", "status": 500, "detail": str(exc)})
        finally:
            reset_event_sink_context(token)
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
