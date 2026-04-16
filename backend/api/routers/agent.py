"""LangGraph RAG Agent API."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, ClassVar, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.api.dependencies import (
    get_default_llm,
    get_prompt_spec_cached,
    get_search_service,
)
from backend.config.settings import agent_settings
from backend.domain.doc_type_mapping import (
    DOC_TYPE_GROUPS,
    expand_doc_type_selection,
    group_doc_type_buckets,
    normalize_doc_type,
)
from backend.llm_infrastructure.llm.langgraph_agent import (
    ParsedQuery,
    _detect_language_rule_based,
    _extract_devices_from_query,
)
from backend.services.device_cache import get_device_cache, ensure_device_cache_initialized
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent
from backend.llm_infrastructure.llm.react_agent import ReactRAGAgent
from backend.services.search_service import SearchService


router = APIRouter(prefix="/agent", tags=["LangGraph Agent"])
logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_TOP_K = 100
DEFAULT_FINAL_TOP_K = 50
MAX_SEARCH_QUERY_METADATA_ITEMS = 5
MAX_SEARCH_QUERY_METADATA_CHARS = 120
MAX_INDEX_METADATA_CHARS = 120

# =============================================================================
# 전역 상태: HIL(Human-in-the-Loop) 지원 및 Auto-Parse 모드
# =============================================================================
# 핵심: interrupt/resume가 동작하려면 동일한 checkpointer(thread_id 기준 상태 공유) 필요
_checkpointer: MemorySaver = MemorySaver()


def _create_device_fetcher(search_service):
    """Create a device/doc_type fetcher function using ES aggregation."""

    def _fetch_devices() -> Dict[str, Any] | list[Dict[str, Any]]:
        # Check if ES engine is available
        if not hasattr(search_service, "es_engine") or search_service.es_engine is None:
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
                    },
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
                    },
                },
            },
        }

        try:
            result = es.search(index=index, body=agg_query)
            device_buckets = result.get("aggregations", {}).get("devices", {}).get("buckets", [])
            doc_type_buckets = (
                result.get("aggregations", {}).get("doc_types", {}).get("buckets", [])
            )
            devices = [
                {
                    "name": bucket["key"],
                    "doc_count": bucket.get("unique_docs", {}).get("value", bucket["doc_count"]),
                }
                for bucket in device_buckets
                if bucket.get("key")
            ]
            doc_types = group_doc_type_buckets(doc_type_buckets, use_unique_docs=True)
            return {"devices": devices, "doc_types": doc_types}
        except Exception as e:
            logger.error(f"Failed to fetch device list: {e}")
            return []

    return _fetch_devices


def _build_mode_variants(*group_names: str) -> set[str]:
    variants: set[str] = set()
    for group_name in group_names:
        variants.add(normalize_doc_type(group_name))
        for raw in DOC_TYPE_GROUPS.get(group_name, []):
            normalized = normalize_doc_type(raw)
            if normalized:
                variants.add(normalized)
    return variants


SOP_DOC_TYPES = _build_mode_variants("SOP", "setup")
ISSUE_DOC_TYPES = _build_mode_variants("myservice", "gcb", "ts") | {
    normalize_doc_type("pems"),
    normalize_doc_type("trouble_shooting_guide"),
    normalize_doc_type("trouble shooting guide"),
}


def _infer_task_mode(doc_types: list[str] | None) -> str | None:
    """Infer task_mode from filter_doc_types selection."""
    if not doc_types:
        return None
    types_set = {normalize_doc_type(dt) for dt in doc_types if normalize_doc_type(dt)}
    if not types_set:
        return None
    if types_set <= SOP_DOC_TYPES:
        return "sop"
    if types_set <= ISSUE_DOC_TYPES:
        return "issue"
    return None


def _build_state_overrides(req: "AgentRequest") -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    pq_fields: Dict[str, Any] = {}

    if req.filter_devices is not None:
        devices = [str(d).strip() for d in req.filter_devices if str(d).strip()]
        if devices:
            overrides["selected_devices"] = devices
            pq_fields["selected_devices"] = devices

    if req.filter_doc_types is not None:
        doc_types = [str(d).strip() for d in req.filter_doc_types if str(d).strip()]
        if doc_types:
            # FE는 그룹명("sop","ts" 등)을 보내므로 ES 실제값으로 확장 후 strict 적용
            expanded = expand_doc_type_selection(doc_types)
            overrides["selected_doc_types"] = expanded
            overrides["selected_doc_types_strict"] = True
            pq_fields["selected_doc_types"] = expanded
            pq_fields["doc_types_strict"] = True

            inferred_task_mode = _infer_task_mode(doc_types) or _infer_task_mode(expanded)
            if inferred_task_mode:
                overrides["task_mode"] = inferred_task_mode
                pq_fields["task_mode"] = inferred_task_mode

    if req.filter_equip_ids is not None:
        equip_ids = [str(e).strip().upper() for e in req.filter_equip_ids if str(e).strip()]
        if equip_ids:
            deduped = list(dict.fromkeys(equip_ids))
            overrides["selected_equip_ids"] = deduped
            pq_fields["selected_equip_ids"] = deduped

    if req.search_queries is not None:
        queries = [str(q).strip() for q in req.search_queries if str(q).strip()]
        if queries:
            overrides["search_queries"] = queries
            overrides["skip_mq"] = True
            pq_fields["search_queries"] = queries

    if req.selected_doc_ids is not None:
        doc_ids = [str(d).strip() for d in req.selected_doc_ids if str(d).strip()]
        if doc_ids:
            overrides["selected_doc_ids"] = doc_ids

    # Override/regeneration 경로는 auto_parse 노드를 건너뛰므로
    # 답변 언어 템플릿 선택을 위해 최소 언어 감지를 보정한다.
    # 장비 필터가 없으면 질문에서 장비도 파싱한다.
    if overrides:
        detected_lang = _detect_language_rule_based(req.message or "")
        overrides["detected_language"] = detected_lang
        overrides["target_language"] = detected_lang
        pq_fields["detected_language"] = detected_lang

        # 장비 필터가 명시되지 않은 경우, 질문에서 장비를 파싱
        if "selected_devices" not in overrides and req.message:
            cache = get_device_cache()
            if cache.is_initialized:
                detected_devices = _extract_devices_from_query(cache.device_names, req.message)
                if detected_devices:
                    overrides["selected_devices"] = detected_devices[:1]
                    pq_fields["selected_devices"] = detected_devices[:1]

        overrides["parsed_query"] = pq_fields

    return overrides


def _new_hil_agent(
    llm,
    search_service,
    prompt_spec,
    *,
    use_canonical_retrieval: bool = False,
    event_sink=None,
) -> LangGraphRAGAgent:
    """HIL용 요청 단위 에이전트. checkpointer를 공유해 interrupt/resume를 유지한다."""
    return LangGraphRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=DEFAULT_FINAL_TOP_K,
        retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
        mode="verified",
        ask_user_after_retrieve=True,
        ask_device_selection=True,
        use_canonical_retrieval=use_canonical_retrieval,
        device_fetcher=_create_device_fetcher(search_service),
        checkpointer=_checkpointer,
        event_sink=event_sink,
    )


def _new_auto_parse_agent(
    llm,
    search_service,
    prompt_spec,
    *,
    top_k: int,
    use_canonical_retrieval: bool = False,
    event_sink=None,
) -> LangGraphRAGAgent:
    """Auto-parse용 요청 단위 에이전트."""
    return LangGraphRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=top_k,
        retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
        mode="verified",
        ask_user_after_retrieve=False,  # 문서 선택 UI 비활성화
        ask_device_selection=False,  # 기기 선택 UI 비활성화
        auto_parse_enabled=True,  # 자동 파싱 활성화
        use_canonical_retrieval=use_canonical_retrieval,
        checkpointer=_checkpointer,
        event_sink=event_sink,
    )


def _new_react_agent(
    llm,
    search_service,
    prompt_spec,
    *,
    top_k: int,
    event_sink=None,
) -> ReactRAGAgent:
    """[실험적] ReAct planner loop 에이전트.
    non-resume / non-retrieval_only / non-guided 경로에만 적용.
    """
    cache = ensure_device_cache_initialized(search_service)
    return ReactRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=top_k,
        retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
        checkpointer=_checkpointer,
        device_names=cache.device_names,
        doc_type_names=cache.doc_type_names,
        equip_id_set=cache.equip_id_set,
        event_sink=event_sink,
    )


def _new_guided_confirm_agent(
    llm,
    search_service,
    prompt_spec,
    *,
    top_k: int,
    use_canonical_retrieval: bool = False,
    event_sink=None,
) -> LangGraphRAGAgent:
    """Auto-parse guided-confirm용 요청 단위 에이전트."""
    return LangGraphRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=top_k,
        retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
        mode="verified",
        ask_user_after_retrieve=False,
        ask_device_selection=False,
        auto_parse_enabled=True,
        use_canonical_retrieval=use_canonical_retrieval,
        device_fetcher=_create_device_fetcher(search_service),
        checkpointer=_checkpointer,
        event_sink=event_sink,
    )


# =============================================================================
# Request/Response Models
# =============================================================================
class ChatHistoryTurn(BaseModel):
    """이전 대화 턴 (후속 질문 지원용)."""

    user_text: str = Field(..., description="이전 사용자 질문")
    assistant_text: str = Field(..., description="이전 답변")
    doc_ids: List[str] = Field(default_factory=list, description="이전 참조 문서 ID")


class AgentRequest(BaseModel):
    message: str = Field(..., description="사용자 질문")
    top_k: int = Field(DEFAULT_FINAL_TOP_K, ge=1, le=50, description="검색 상위 문서 수")
    max_attempts: int = Field(3, ge=0, le=3, description="judge 실패 시 재시도 횟수")
    mode: str = Field("verified", description="base 또는 verified")
    thread_id: Optional[str] = Field(None, description="LangGraph thread_id")
    ask_user_after_retrieve: bool = Field(False, description="검색 후 사용자 확인")
    resume_decision: Optional[Any] = Field(None, description="interrupt 재개 응답")
    auto_parse: bool = Field(True, description="자동 장비/문서종류 파싱 (기본값: True)")
    guided_confirm: bool = Field(False, description="auto-parse 가이드 확인 플로우 활성화 여부")
    # 대화 이력 (후속 질문 지원)
    chat_history: Optional[List[ChatHistoryTurn]] = Field(
        None, description="이전 대화 이력 (최근 1턴)"
    )
    # 재생성 시 사용할 필터 오버라이드
    filter_devices: Optional[List[str]] = Field(None, description="장비 필터 오버라이드")
    filter_doc_types: Optional[List[str]] = Field(None, description="문서종류 필터 오버라이드")
    filter_equip_ids: Optional[List[str]] = Field(None, description="equip_id 필터 오버라이드")
    search_queries: Optional[List[str]] = Field(None, description="검색 쿼리 오버라이드 (MQ 수정)")
    selected_doc_ids: Optional[List[str]] = Field(None, description="사용할 문서 ID 선택 (재생성)")
    mq_mode: Optional[Literal["off", "fallback", "on"]] = Field(
        None,
        description="MQ retrieval mode override (off, fallback, on)",
    )
    use_canonical_retrieval: bool = Field(
        False,
        description="검색 단계에서 canonical retrieval pipeline 사용 여부 (기본값: False)",
    )
    retrieval_only: bool = Field(
        False,
        description="답변 생성 전 문서 검색 단계까지만 수행 (전체 문서 대상)",
    )
    use_react_agent: bool = Field(
        False,
        description="[실험적] ReAct planner loop 기반 에이전트 사용 (non-resume, non-retrieval_only 경로만 적용)",
    )
    context_chunk_ids: Optional[List[str]] = Field(
        None,
        description="관련 문서 제안 버튼 클릭 시 사용할 chunk_id 목록. "
        "제공 시 검색을 스킵하고 해당 chunk로 직접 답변 생성.",
    )


class AutoParseConfirmDecision(BaseModel):
    type: Literal["auto_parse_confirm"] = Field(..., description="resume decision type")
    target_language: Literal["ko", "en", "zh", "ja"] = Field("ko", description="답변 언어")
    selected_device: Optional[str] = Field(None, description="선택된 장비명")
    selected_equip_id: Optional[str] = Field(None, description="선택된 설비 ID")
    task_mode: Literal["sop", "issue", "all"] = Field("all", description="작업 모드")

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class IssueConfirmDecision(BaseModel):
    type: Literal["issue_confirm"] = Field(..., description="resume decision type")
    nonce: str = Field(..., description="interrupt nonce")
    stage: Literal["post_summary", "post_detail"] = Field(..., description="issue flow stage")
    confirm: bool = Field(..., description="confirm decision")

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class IssueCaseSelectionDecision(BaseModel):
    type: Literal["issue_case_selection"] = Field(..., description="resume decision type")
    nonce: str = Field(..., description="interrupt nonce")
    selected_doc_id: str = Field(..., description="selected issue document id")

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class IssueSopConfirmDecision(BaseModel):
    type: Literal["issue_sop_confirm"] = Field(..., description="resume decision type")
    nonce: str = Field(..., description="interrupt nonce")
    confirm: bool = Field(..., description="confirm decision")

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


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
    start_page: int | None = None  # 확장 섹션의 시작 페이지 (스크롤 위치용)


class AutoParseResult(BaseModel):
    """자동 파싱 결과"""

    device: Optional[str] = Field(None, description="파싱된 장비명")
    doc_type: Optional[str] = Field(None, description="파싱된 문서종류")
    devices: Optional[List[str]] = Field(None, description="파싱된 장비명 목록")
    doc_types: Optional[List[str]] = Field(None, description="파싱된 문서종류 목록")
    equip_id: Optional[str] = Field(None, description="파싱된 equip_id")
    equip_ids: Optional[List[str]] = Field(None, description="파싱된 equip_id 목록")
    language: Optional[str] = Field(None, description="감지된 언어 (ko, en, ja, zh)")
    message: Optional[str] = Field(None, description="사용자에게 표시할 메시지")


class AgentResponse(BaseModel):
    query: str
    answer: str
    reasoning: Optional[str] = Field(None, description="LLM reasoning 과정 (reasoning 모델인 경우)")
    judge: Dict[str, Any]
    retrieved_docs: List[RetrievedDoc]
    all_retrieved_docs: Optional[List[RetrievedDoc]] = Field(
        None, description="전체 검색 문서 (재생성용, 최대 retrieval_top_k개)"
    )
    expanded_docs: Optional[List[ExpandedDoc]] = Field(
        None, description="확장된 문서 (답변 생성용)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    interrupted: bool = Field(False)
    interrupt_payload: Optional[Dict[str, Any]] = Field(None)
    thread_id: Optional[str] = Field(None)
    # Auto-parse results
    auto_parse: Optional[AutoParseResult] = Field(None, description="자동 파싱 결과")
    # Filter info for regeneration
    selected_devices: Optional[List[str]] = Field(None, description="사용된 장비 필터")
    selected_doc_types: Optional[List[str]] = Field(None, description="사용된 문서종류 필터")
    selected_equip_ids: Optional[List[str]] = Field(None, description="사용된 equip_id 필터")
    search_queries: Optional[List[str]] = Field(None, description="사용된 검색 쿼리 (MQ)")
    # Language detection
    detected_language: Optional[str] = Field(None, description="감지된 언어 (ko, en, ja, zh)")
    # Auto-parse 후속 UX
    suggest_additional_device_search: bool = Field(
        False,
        description="auto_parse에서 device를 찾지 못해 추가 장비 검색을 제안해야 하는지 여부",
    )
    related_doc_types: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="다른 doc_type에서 발견된 관련 문서 제안 [{doc_type, count, message}]",
    )


class TraceContext(BaseModel):
    trace_id: str
    traceparent: Optional[str] = None
    tracestate: Optional[str] = None


def _is_auto_parse_confirm_resume(req: AgentRequest) -> bool:
    if req.resume_decision is None or req.thread_id is None:
        return False
    if not isinstance(req.resume_decision, dict):
        return False
    return req.resume_decision.get("type") == "auto_parse_confirm"


def _is_guided_resume(req: AgentRequest) -> bool:
    if req.resume_decision is None or req.thread_id is None:
        return False
    if not isinstance(req.resume_decision, dict):
        return False
    decision_type = req.resume_decision.get("type")
    return decision_type in {
        "auto_parse_confirm",
        "issue_confirm",
        "issue_case_selection",
        "issue_sop_confirm",
    }


def _validated_auto_parse_confirm_decision(decision: Any) -> Dict[str, Any]:
    try:
        return AutoParseConfirmDecision.model_validate(decision).model_dump()
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid auto_parse_confirm resume_decision: {exc.errors()}",
        ) from exc


def _validated_guided_resume_decision(decision: Any) -> Dict[str, Any]:
    if not isinstance(decision, dict):
        raise HTTPException(
            status_code=400, detail="Invalid guided resume_decision: expected object"
        )
    decision_type = decision.get("type")

    # abbreviation_resolve: 별도 Pydantic 모델 없이 직접 통과
    if decision_type == "abbreviation_resolve":
        if not isinstance(decision.get("selections"), dict):
            raise HTTPException(
                status_code=400, detail="Invalid abbreviation_resolve: selections must be dict"
            )
        return decision

    model_map: Dict[str, type[BaseModel]] = {
        "auto_parse_confirm": AutoParseConfirmDecision,
        "issue_confirm": IssueConfirmDecision,
        "issue_case_selection": IssueCaseSelectionDecision,
        "issue_sop_confirm": IssueSopConfirmDecision,
    }
    model_cls = model_map.get(str(decision_type))
    if model_cls is None:
        raise HTTPException(
            status_code=400, detail="Invalid guided resume_decision: unsupported type"
        )

    try:
        return model_cls.model_validate(decision).model_dump()
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {decision_type} resume_decision: {exc.errors()}",
        ) from exc


_TRACEPARENT_PATTERN = re.compile(r"^[\da-f]{2}-[\da-f]{32}-[\da-f]{16}-[\da-f]{2}$", re.IGNORECASE)


def _resolve_trace_context(traceparent: Optional[str], tracestate: Optional[str]) -> TraceContext:
    cleaned_traceparent = (traceparent or "").strip()
    if not _TRACEPARENT_PATTERN.match(cleaned_traceparent):
        return TraceContext(trace_id=uuid.uuid4().hex)

    segments = cleaned_traceparent.split("-")
    trace_id_segment, parent_id_segment = segments[1], segments[2]
    if int(trace_id_segment, 16) == 0 or int(parent_id_segment, 16) == 0:
        return TraceContext(trace_id=uuid.uuid4().hex)

    cleaned_tracestate = (tracestate or "").strip() or None
    return TraceContext(
        trace_id=uuid.uuid4().hex,
        traceparent=cleaned_traceparent,
        tracestate=cleaned_tracestate,
    )


def _to_expanded_docs(answer_ref_json: List[Dict[str, Any]] | None) -> List[ExpandedDoc] | None:
    """answer_ref_json을 ExpandedDoc 리스트로 변환.

    같은 doc_id의 ref들 중 최소 page를 start_page로 설정하여
    프론트엔드 스크롤 위치를 섹션 시작 페이지로 안내한다.
    """
    if not answer_ref_json:
        return None

    # doc_id별 최소 페이지 계산
    doc_min_page: Dict[str, int] = {}
    for ref in answer_ref_json:
        doc_id = ref.get("doc_id", "")
        page = ref.get("page")
        if doc_id and page is not None:
            if doc_id not in doc_min_page or page < doc_min_page[doc_id]:
                doc_min_page[doc_id] = page

    docs: List[ExpandedDoc] = []
    for ref in answer_ref_json:
        content = ref.get("content", "")
        doc_id = ref.get("doc_id", "")
        docs.append(
            ExpandedDoc(
                rank=ref.get("rank", 0),
                doc_id=doc_id,
                content=content,
                content_length=len(content),
                start_page=doc_min_page.get(doc_id),
            )
        )
    return docs if docs else None


def _select_display_docs(result: Dict[str, Any]) -> List[RetrievalResult]:
    """Prefer UI display docs from expansion, fallback to full docs list."""
    if "display_docs" in result:
        return result.get("display_docs") or []
    return result.get("docs", []) or []


def _to_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    return [s] if s else []


def _is_effective_parsed_device(value: str) -> bool:
    token = re.sub(r"[\s\-_./]+", "", str(value or "").strip().upper())
    if not token:
        return False
    # Short alphabetic tokens are often component labels rather than equipment models.
    if token.isalpha() and len(token) <= 4:
        return False
    return True


def _extract_auto_parse_result(result: Dict[str, Any]) -> Optional[AutoParseResult]:
    """result dict에서 AutoParseResult를 추출한다. ParsedQuery 우선, 기존 필드 fallback."""
    pq_raw = result.get("parsed_query")
    if pq_raw:
        pq = ParsedQuery(**pq_raw)
        has_auto_parse_fields = bool(
            pq.device_names
            or pq.doc_types
            or pq.equip_ids
            or pq.detected_language is not None
            or pq.message
        )
        if has_auto_parse_fields:
            return AutoParseResult(
                device=pq.device_names[0] if pq.device_names else None,
                doc_type=pq.doc_types[0] if pq.doc_types else None,
                devices=pq.device_names or None,
                doc_types=pq.doc_types or None,
                equip_id=pq.equip_ids[0] if pq.equip_ids else None,
                equip_ids=pq.equip_ids or None,
                language=pq.detected_language,
                message=pq.message,
            )

    # 기존 fallback
    if (
        result.get("auto_parsed_device")
        or result.get("auto_parsed_doc_type")
        or result.get("auto_parsed_devices")
        or result.get("auto_parsed_doc_types")
        or result.get("auto_parsed_equip_id")
        or result.get("auto_parsed_equip_ids")
        or result.get("auto_parse_message")
        or result.get("detected_language")
    ):
        return AutoParseResult(
            device=result.get("auto_parsed_device"),
            doc_type=result.get("auto_parsed_doc_type"),
            devices=result.get("auto_parsed_devices"),
            doc_types=result.get("auto_parsed_doc_types"),
            equip_id=result.get("auto_parsed_equip_id"),
            equip_ids=result.get("auto_parsed_equip_ids"),
            language=result.get("detected_language"),
            message=result.get("auto_parse_message"),
        )

    return None


def _should_suggest_additional_device_search(
    result: Dict[str, Any],
    auto_parse_result: Optional[AutoParseResult],
) -> bool:
    """Suggest additional device search when auto-parse ran but parsed no device."""
    # ParsedQuery 우선 참조
    pq_raw = result.get("parsed_query")
    if pq_raw:
        pq = ParsedQuery(**pq_raw)
        # Regeneration/override path may store selected_* filters in parsed_query even when
        # auto-parse did not run; do not emit additional auto-parse suggestion in that case.
        if pq.selected_devices or pq.selected_doc_types or pq.selected_equip_ids:
            return False
        effective = [d for d in pq.device_names if _is_effective_parsed_device(d)]
        return not effective and pq.detected_language is not None

    # 기존 fallback 로직
    auto_parse_ran = auto_parse_result is not None or any(
        key in result
        for key in (
            "auto_parsed_device",
            "auto_parsed_devices",
            "auto_parsed_doc_type",
            "auto_parsed_doc_types",
            "auto_parsed_equip_id",
            "auto_parsed_equip_ids",
            "auto_parse_message",
            "detected_language",
            "device_selection_skipped",
        )
    )
    if not auto_parse_ran:
        return False

    parsed_devices = (
        _to_str_list(result.get("auto_parsed_devices"))
        or _to_str_list(auto_parse_result.devices if auto_parse_result else None)
        or _to_str_list(result.get("auto_parsed_device"))
        or _to_str_list(auto_parse_result.device if auto_parse_result else None)
    )
    effective_devices = [d for d in parsed_devices if _is_effective_parsed_device(d)]
    return not effective_devices


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
            expanded_page_urls = [f"/api/assets/docs/{doc_id}/pages/{p}" for p in expanded_pages]
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
                metadata=metadata,
                page=page,
                page_image_url=page_image_url,
                expanded_pages=expanded_pages,
                expanded_page_urls=expanded_page_urls,
            )
        )
    return docs


def _sanitize_search_queries_raw(result: Dict[str, Any]) -> Optional[List[str]]:
    raw_keys = [
        "search_queries_raw",
        "raw_search_queries",
        "general_mq_list",
        "setup_mq_list",
        "ts_mq_list",
        "general_mq_ko_list",
        "setup_mq_ko_list",
        "ts_mq_ko_list",
    ]
    merged: List[str] = []
    for key in raw_keys:
        merged.extend(_to_str_list(result.get(key)))

    if not merged:
        return None

    cleaned: List[str] = []
    seen: set[str] = set()
    for query in merged:
        clipped = query.strip()[:MAX_SEARCH_QUERY_METADATA_CHARS]
        if not clipped:
            continue
        dedupe_key = re.sub(r"\s+", " ", clipped).strip().casefold()
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(clipped)
        if len(cleaned) >= MAX_SEARCH_QUERY_METADATA_ITEMS:
            break

    return cleaned or None


_DOC_TYPE_LABELS: dict[str, str] = {
    "ts": "Trouble Shooting",
    "setup": "Setup/설치",
    "sop": "SOP",
    "myservice": "MyService 이력",
    "gcb": "GCB",
}


def _build_related_doc_type_suggestions(
    search_service: SearchService,
) -> Optional[List[Dict[str, Any]]]:
    """search_service의 cross_type_suggestions를 API 응답 형식으로 변환."""
    suggestions_map = getattr(search_service, "_last_cross_type_suggestions", {})
    if not suggestions_map:
        return None
    result: list[dict[str, Any]] = []
    for doc_type, chunks in suggestions_map.items():
        label = _DOC_TYPE_LABELS.get(doc_type, doc_type.upper())
        chunk_ids = [(c.metadata or {}).get("chunk_id", c.doc_id) for c in chunks]
        result.append(
            {
                "doc_type": doc_type,
                "count": len(chunks),
                "chunk_ids": chunk_ids,
                "message": f"{label} 문서에도 관련 내용이 {len(chunks)}건 있습니다. 확인하시겠습니까?",
            }
        )
    return result


def _resolve_index_name(search_service: SearchService) -> Optional[str]:
    es_engine = getattr(search_service, "es_engine", None)
    if es_engine is None:
        return None
    index_name = getattr(es_engine, "index_name", None)
    if index_name is None:
        return None
    safe = re.sub(r"[^\w.:-]+", "_", str(index_name).strip())
    if not safe:
        return None
    return safe[:MAX_INDEX_METADATA_CHARS]


def _build_response_metadata(
    result: Dict[str, Any],
    mq_mode: str,
    max_attempts: int,
    final_search_queries: List[str],
    index_name: Optional[str],
) -> Dict[str, Any]:
    valid_languages = {"ko", "en", "zh", "ja"}

    target_language_raw = result.get("target_language")
    target_language = ""
    if isinstance(target_language_raw, str):
        target_language = target_language_raw.strip().lower()
    has_valid_user_target = bool(target_language) and target_language in valid_languages

    detected_language_raw = result.get("detected_language")
    detected_language = ""
    if isinstance(detected_language_raw, str):
        detected_language = detected_language_raw.strip().lower()

    answer_language = target_language if has_valid_user_target else (detected_language or "ko")

    if has_valid_user_target:
        selected_language_source = "user"
    elif detected_language_raw is not None:
        selected_language_source = "auto_parse"
    else:
        selected_language_source = "default"

    selected_task_mode_raw = result.get("task_mode")
    selected_task_mode: Optional[str]
    if selected_task_mode_raw is None:
        selected_task_mode = None
    else:
        normalized_task_mode = str(selected_task_mode_raw).strip().lower()
        selected_task_mode = normalized_task_mode or None

    applied_doc_type_scope_raw = result.get("selected_doc_types") or []
    applied_doc_type_scope: List[str]
    if isinstance(applied_doc_type_scope_raw, list):
        applied_doc_type_scope = [
            str(item).strip() for item in applied_doc_type_scope_raw if str(item).strip()
        ]
    elif isinstance(applied_doc_type_scope_raw, str):
        scoped_value = applied_doc_type_scope_raw.strip()
        applied_doc_type_scope = [scoped_value] if scoped_value else []
    else:
        applied_doc_type_scope = []

    retrieval_stage2_raw = result.get("retrieval_stage2")
    retrieval_stage2_doc_ids: List[str] = []
    retrieval_stage2_enabled = False
    if isinstance(retrieval_stage2_raw, dict):
        retrieval_stage2_enabled = bool(retrieval_stage2_raw.get("enabled", False))
        doc_ids = retrieval_stage2_raw.get("doc_ids")
        if isinstance(doc_ids, list):
            retrieval_stage2_doc_ids = [str(x).strip() for x in doc_ids if str(x).strip()]

    metadata: Dict[str, Any] = {
        "route": result.get("route"),
        "st_gate": result.get("st_gate"),
        "search_queries": final_search_queries,
        "search_queries_final": final_search_queries,
        "target_language": answer_language,
        "selected_language_source": selected_language_source,
        "selected_device": result.get("selected_device"),
        "selected_task_mode": selected_task_mode,
        "applied_doc_type_scope": applied_doc_type_scope,
        "scope_by_task_override": bool(selected_task_mode and selected_task_mode != "all"),
        "retrieval_stage2": {
            "enabled": retrieval_stage2_enabled,
            "doc_ids": retrieval_stage2_doc_ids,
        },
        "mq_mode": mq_mode,
        "mq_used": bool(result.get("mq_used", False)),
        "mq_reason": result.get("mq_reason"),
        "attempts": int(result.get("attempts", 0) or 0),
        "max_attempts": int(max_attempts),
        "retry_strategy": result.get("retry_strategy"),
        "guardrail_dropped_numeric": int(result.get("guardrail_dropped_numeric", 0) or 0),
        "guardrail_dropped_anchor": int(result.get("guardrail_dropped_anchor", 0) or 0),
        "guardrail_final_count": int(
            result.get("guardrail_final_count", len(final_search_queries)) or 0
        ),
    }

    issue_policy_tier = result.get("issue_policy_tier")
    issue_policy_tier_shadow = result.get("issue_policy_tier_shadow")
    issue_fallback_reason = result.get("issue_fallback_reason")
    issue_detail_ref_source = result.get("issue_detail_ref_source")
    issue_policy_rollout_phase = result.get("issue_policy_rollout_phase")
    issue_routing_signals = result.get("issue_routing_signals")
    issue_case_refs = result.get("issue_case_refs")
    issue_answer_refs = result.get("answer_ref_json")
    issue_case_refs_shadow = result.get("issue_case_refs_shadow")
    issue_answer_refs_shadow = result.get("issue_answer_refs_shadow")
    issue_mode = bool(
        str(result.get("task_mode") or "").strip().lower() == "issue"
        or isinstance(issue_policy_tier, str)
        or isinstance(issue_policy_tier_shadow, str)
        or isinstance(issue_case_refs, list)
    )

    if issue_mode and isinstance(issue_policy_tier, str) and issue_policy_tier:
        metadata["issue_policy_tier"] = issue_policy_tier
    if issue_mode and isinstance(issue_policy_tier_shadow, str) and issue_policy_tier_shadow:
        metadata["issue_policy_tier_shadow"] = issue_policy_tier_shadow
    if issue_mode and isinstance(issue_fallback_reason, str) and issue_fallback_reason:
        metadata["issue_fallback_reason"] = issue_fallback_reason
    if issue_mode and isinstance(issue_detail_ref_source, str) and issue_detail_ref_source:
        metadata["issue_detail_ref_source"] = issue_detail_ref_source
    if issue_mode and issue_policy_rollout_phase is not None:
        try:
            metadata["issue_policy_rollout_phase"] = int(issue_policy_rollout_phase)
        except (TypeError, ValueError):
            pass

    if issue_mode and isinstance(issue_case_refs, list):
        metadata["issue_case_count"] = len(issue_case_refs)
    if issue_mode and isinstance(issue_answer_refs, list):
        metadata["issue_answer_ref_count"] = len(issue_answer_refs)
    if issue_mode and isinstance(issue_case_refs_shadow, list):
        metadata["issue_case_count_shadow"] = len(issue_case_refs_shadow)
    if issue_mode and isinstance(issue_answer_refs_shadow, list):
        metadata["issue_answer_ref_count_shadow"] = len(issue_answer_refs_shadow)

    if issue_mode and isinstance(issue_routing_signals, dict):
        key_map = {
            "score_gap_12": "issue_signal_score_gap_12",
            "myservice_share_50": "issue_signal_myservice_share_50",
            "gcb_count_50": "issue_signal_gcb_count_50",
            "ts_count_50": "issue_signal_ts_count_50",
            "doc_type_entropy_20": "issue_signal_doc_type_entropy_20",
            "non_myservice_presence_50": "issue_signal_non_myservice_presence_50",
            "gcb_chapter_coverage_10": "issue_signal_gcb_chapter_coverage_10",
            "issue_signal_k_effective": "issue_signal_k_effective",
        }
        for src_key, dst_key in key_map.items():
            value = issue_routing_signals.get(src_key)
            if value is None:
                continue
            metadata[dst_key] = value

    answer_format = result.get("answer_format")
    if isinstance(answer_format, dict):
        metadata["answer_format"] = answer_format
    retries_raw = result.get("answer_format_retries")
    if retries_raw is not None:
        try:
            metadata["answer_format_retries"] = int(retries_raw or 0)
        except (TypeError, ValueError):
            metadata["answer_format_retries"] = 0
    search_queries_raw = _sanitize_search_queries_raw(result)
    if search_queries_raw:
        metadata["search_queries_raw"] = search_queries_raw
    if index_name:
        metadata["index_name"] = index_name
    human_action = result.get("human_action")
    run_id = result.get("canonical_run_id")
    config_hash = result.get("canonical_effective_config_hash")
    if isinstance(human_action, dict):
        run_id = run_id or human_action.get("canonical_run_id")
        config_hash = config_hash or human_action.get("canonical_effective_config_hash")
    if run_id:
        metadata["run_id"] = run_id
    if config_hash:
        metadata["effective_config_hash"] = config_hash
    return metadata


# =============================================================================
# API Endpoint
# =============================================================================
@router.post("/run", response_model=AgentResponse)
async def run_agent(
    req: AgentRequest,
    traceparent: Optional[str] = Header(default=None),
    tracestate: Optional[str] = Header(default=None),
    search_service: SearchService = Depends(get_search_service),
    llm=Depends(get_default_llm),
    prompt_spec=Depends(get_prompt_spec_cached),
):
    """LangGraph RAG 에이전트 실행."""
    if not hasattr(search_service, "search"):
        raise HTTPException(status_code=503, detail="Search service not configured")
    if req.resume_decision is not None and req.thread_id is None:
        raise HTTPException(status_code=400, detail="thread_id required for resume")

    tid = req.thread_id or str(uuid.uuid4())
    is_resume = req.resume_decision is not None and req.thread_id is not None
    is_guided_resume = _is_guided_resume(req)
    validated_guided_resume_decision = (
        _validated_guided_resume_decision(req.resume_decision) if is_guided_resume else None
    )
    trace_context = _resolve_trace_context(traceparent, tracestate)
    effective_mq_mode = req.mq_mode or agent_settings.mq_mode_default

    state_overrides = _build_state_overrides(req)
    has_overrides = bool(state_overrides)
    state_overrides["mq_mode"] = effective_mq_mode

    # context_chunk_ids: 관련 문서 제안 버튼 클릭 → 해당 chunk로 직접 답변 생성
    if req.context_chunk_ids:
        fetcher = getattr(search_service, "fetch_chunks_by_ids", None)
        if fetcher:
            context_docs = fetcher(req.context_chunk_ids)
            if context_docs:
                state_overrides["context_docs"] = context_docs
                has_overrides = True
                logger.info(
                    "[run] context_chunk_ids: %d requested, %d fetched",
                    len(req.context_chunk_ids),
                    len(context_docs),
                )

    # Build chat_history state (separate from overrides to avoid triggering regeneration path)
    chat_state: Dict[str, Any] = {}
    chat_state["mq_mode"] = effective_mq_mode
    chat_state["guided_confirm"] = req.guided_confirm
    if req.chat_history:
        chat_state["chat_history"] = [h.model_dump() for h in req.chat_history]

    try:
        if req.retrieval_only and not is_resume:
            retrieval_only_agent = LangGraphRAGAgent(
                llm=llm,
                search_service=search_service,
                prompt_spec=prompt_spec,
                top_k=req.top_k,
                retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
                mode="base",
                ask_user_after_retrieve=True,
                ask_device_selection=False,
                auto_parse_enabled=False,
                use_canonical_retrieval=req.use_canonical_retrieval,
                checkpointer=None,
            )
            result = retrieval_only_agent.run(
                req.message,
                attempts=0,
                max_attempts=req.max_attempts,
                thread_id=tid,
                state_overrides={"mq_mode": effective_mq_mode},
            )
        # ReAct agent: use_react_agent가 명시적으로 요청된 경우 (overrides 유무 무관)
        elif req.use_react_agent and not is_resume:
            agent = _new_react_agent(
                llm,
                search_service,
                prompt_spec,
                top_k=req.top_k,
            )
            result = agent.run(
                req.message,
                attempts=0,
                max_attempts=req.max_attempts,
                thread_id=tid,
                state_overrides={
                    **(chat_state or {}),
                    "chat_history": [h.model_dump() for h in req.chat_history] if req.chat_history else [],
                },
            )
        # Auto-parse 모드 (기본값: True), skip when overrides are provided
        elif (
            req.auto_parse
            and not is_resume
            and not req.ask_user_after_retrieve
            and not has_overrides
        ):
            if req.guided_confirm:
                chat_state["auto_parse_confirmed"] = False
                agent = _new_guided_confirm_agent(
                    llm,
                    search_service,
                    prompt_spec,
                    top_k=req.top_k,
                    use_canonical_retrieval=req.use_canonical_retrieval,
                )
            elif req.use_react_agent:
                # [실험적] ReAct planner loop — non-resume/non-guided 경로
                # retrieval_only=True 시 _search_node에서 interrupt 처리 (C-API-003)
                agent = _new_react_agent(
                    llm,
                    search_service,
                    prompt_spec,
                    top_k=req.top_k,
                )
            else:
                agent = _new_auto_parse_agent(
                    llm,
                    search_service,
                    prompt_spec,
                    top_k=req.top_k,
                    use_canonical_retrieval=req.use_canonical_retrieval,
                )
            result = agent.run(
                req.message,
                attempts=0,
                max_attempts=req.max_attempts,
                thread_id=tid,
                state_overrides=chat_state or None,
            )
        elif has_overrides and not is_resume:
            # Regeneration with filter/query/doc overrides
            agent = LangGraphRAGAgent(
                llm=llm,
                search_service=search_service,
                prompt_spec=prompt_spec,
                top_k=req.top_k,
                retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
                mode=req.mode,
                ask_user_after_retrieve=False,
                use_canonical_retrieval=req.use_canonical_retrieval,
                checkpointer=_checkpointer,
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
            if is_resume and is_guided_resume:
                agent = _new_guided_confirm_agent(
                    llm,
                    search_service,
                    prompt_spec,
                    top_k=req.top_k,
                    use_canonical_retrieval=req.use_canonical_retrieval,
                )
            else:
                agent = _new_hil_agent(
                    llm,
                    search_service,
                    prompt_spec,
                    use_canonical_retrieval=req.use_canonical_retrieval,
                )
            config = {"configurable": {"thread_id": tid}}

            if is_resume:
                # 체크포인트 확인
                state = agent._graph.get_state(config)
                logger.info(f"[resume] tid={tid}, state exists={state is not None}")

                if state is None or not state.values:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No checkpoint for thread_id={tid}. Server may have restarted.",
                    )

                logger.info(f"[resume] next={state.next}, keys={list(state.values.keys())}")
                resume_payload = (
                    validated_guided_resume_decision if is_guided_resume else req.resume_decision
                )
                result = agent._graph.invoke(Command(resume=resume_payload), config)
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
                retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
                mode=req.mode,
                ask_user_after_retrieve=False,
                use_canonical_retrieval=req.use_canonical_retrieval,
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
    auto_parse_result = _extract_auto_parse_result(result)
    suggest_additional_device_search = _should_suggest_additional_device_search(
        result, auto_parse_result
    )
    final_search_queries = _to_str_list(result.get("search_queries"))
    index_name = _resolve_index_name(search_service)

    # Interrupt 체크
    interrupt_info = result.get("__interrupt__")
    if interrupt_info:
        payload = None
        if len(interrupt_info) > 0:
            payload = interrupt_info[0].value if hasattr(interrupt_info[0], "value") else None

        return AgentResponse(
            query=req.message,
            answer=result.get("answer", "") or "",
            reasoning=result.get("reasoning"),
            judge=result.get("judge", {}) or {},
            retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
            all_retrieved_docs=_to_retrieved_docs(result.get("all_docs") or result.get("docs", [])),
            expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
            metadata={
                **_build_response_metadata(
                    result,
                    effective_mq_mode,
                    req.max_attempts,
                    final_search_queries,
                    index_name,
                ),
                "trace": trace_context.model_dump(exclude_none=True),
                "response_mode": "retrieval_only" if req.retrieval_only else "full",
            },
            interrupted=True,
            interrupt_payload=payload,
            thread_id=tid,
            auto_parse=auto_parse_result,
            selected_devices=result.get("selected_devices"),
            selected_doc_types=result.get("selected_doc_types"),
            selected_equip_ids=result.get("selected_equip_ids"),
            search_queries=final_search_queries,
            detected_language=result.get("detected_language"),
            suggest_additional_device_search=suggest_additional_device_search,
            related_doc_types=_build_related_doc_type_suggestions(search_service),
        )

    # 정상 완료
    return AgentResponse(
        query=req.message,
        answer=result.get("answer", ""),
        reasoning=result.get("reasoning"),
        judge=result.get("judge", {}),
        retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
        all_retrieved_docs=_to_retrieved_docs(result.get("all_docs") or result.get("docs", [])),
        expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
        metadata={
            **_build_response_metadata(
                result,
                effective_mq_mode,
                req.max_attempts,
                final_search_queries,
                index_name,
            ),
            "trace": trace_context.model_dump(exclude_none=True),
            "response_mode": "retrieval_only" if req.retrieval_only else "full",
        },
        interrupted=False,
        interrupt_payload=None,
        thread_id=tid,
        auto_parse=auto_parse_result,
        selected_devices=result.get("selected_devices"),
        selected_doc_types=result.get("selected_doc_types"),
        selected_equip_ids=result.get("selected_equip_ids"),
        search_queries=final_search_queries,
        detected_language=result.get("detected_language"),
        suggest_additional_device_search=suggest_additional_device_search,
        related_doc_types=_build_related_doc_type_suggestions(search_service),
    )


@router.post("/run/stream")
async def run_agent_stream(
    req: AgentRequest,
    request: Request,
    traceparent: Optional[str] = Header(default=None),
    tracestate: Optional[str] = Header(default=None),
    search_service: SearchService = Depends(get_search_service),
    llm=Depends(get_default_llm),
    prompt_spec=Depends(get_prompt_spec_cached),
):
    """LangGraph RAG 에이전트 실행 (SSE: 노드 실행 로그 스트리밍)."""
    if not hasattr(search_service, "search"):
        raise HTTPException(status_code=503, detail="Search service not configured")
    if req.resume_decision is not None and req.thread_id is None:
        raise HTTPException(status_code=400, detail="thread_id required for resume")

    tid = req.thread_id or str(uuid.uuid4())
    is_resume = req.resume_decision is not None and req.thread_id is not None
    is_guided_resume = _is_guided_resume(req)
    validated_guided_resume_decision = (
        _validated_guided_resume_decision(req.resume_decision) if is_guided_resume else None
    )
    trace_context = _resolve_trace_context(traceparent, tracestate)
    trace_payload = trace_context.model_dump(exclude_none=True)
    effective_mq_mode = req.mq_mode or agent_settings.mq_mode_default
    index_name = _resolve_index_name(search_service)

    state_overrides = _build_state_overrides(req)
    has_overrides = bool(state_overrides)
    state_overrides["mq_mode"] = effective_mq_mode

    # context_chunk_ids: 관련 문서 제안 버튼 클릭 → 해당 chunk로 직접 답변 생성
    if req.context_chunk_ids:
        fetcher = getattr(search_service, "fetch_chunks_by_ids", None)
        if fetcher:
            context_docs = fetcher(req.context_chunk_ids)
            if context_docs:
                state_overrides["context_docs"] = context_docs
                has_overrides = True
                logger.info(
                    "[run/stream] context_chunk_ids: %d requested, %d fetched",
                    len(req.context_chunk_ids),
                    len(context_docs),
                )

    logger.info(
        "[run/stream] auto_parse=%s, is_resume=%s, ask_user=%s, has_overrides=%s, guided_confirm=%s, overrides_keys=%s",
        req.auto_parse,
        is_resume,
        req.ask_user_after_retrieve,
        has_overrides,
        req.guided_confirm,
        list(state_overrides.keys()),
    )

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

    # Build chat_history state (separate from overrides to avoid triggering regeneration path)
    chat_state_stream: Dict[str, Any] = {}
    chat_state_stream["mq_mode"] = effective_mq_mode
    chat_state_stream["guided_confirm"] = req.guided_confirm
    if req.chat_history:
        chat_state_stream["chat_history"] = [h.model_dump() for h in req.chat_history]

    def _worker() -> None:
        try:
            if req.retrieval_only and not is_resume:
                retrieval_only_agent = LangGraphRAGAgent(
                    llm=llm,
                    search_service=search_service,
                    prompt_spec=prompt_spec,
                    top_k=req.top_k,
                    retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
                    mode="base",
                    ask_user_after_retrieve=True,
                    ask_device_selection=False,
                    auto_parse_enabled=False,
                    use_canonical_retrieval=req.use_canonical_retrieval,
                    checkpointer=None,
                    event_sink=_enqueue,
                )
                result = retrieval_only_agent.run(
                    req.message,
                    attempts=0,
                    max_attempts=req.max_attempts,
                    thread_id=tid,
                    state_overrides={"mq_mode": effective_mq_mode},
                )
            # ReAct agent: use_react_agent가 명시적으로 요청된 경우 (overrides 유무 무관)
            elif req.use_react_agent and not is_resume:
                agent = _new_react_agent(
                    llm,
                    search_service,
                    prompt_spec,
                    top_k=req.top_k,
                    event_sink=_enqueue,
                )
                result = agent.run(
                    req.message,
                    attempts=0,
                    max_attempts=req.max_attempts,
                    thread_id=tid,
                    state_overrides={
                        **(chat_state_stream or {}),
                        "chat_history": [h.model_dump() for h in req.chat_history] if req.chat_history else [],
                    },
                )
            # Auto-parse 모드 (기본값: True), skip when overrides are provided
            elif (
                req.auto_parse
                and not is_resume
                and not req.ask_user_after_retrieve
                and not has_overrides
            ):
                if req.guided_confirm:
                    chat_state_stream["auto_parse_confirmed"] = False
                    agent = _new_guided_confirm_agent(
                        llm,
                        search_service,
                        prompt_spec,
                        top_k=req.top_k,
                        use_canonical_retrieval=req.use_canonical_retrieval,
                        event_sink=_enqueue,
                    )
                else:
                    agent = _new_auto_parse_agent(
                        llm,
                        search_service,
                        prompt_spec,
                        top_k=req.top_k,
                        use_canonical_retrieval=req.use_canonical_retrieval,
                        event_sink=_enqueue,
                    )
                result = agent.run(
                    req.message,
                    attempts=0,
                    max_attempts=req.max_attempts,
                    thread_id=tid,
                    state_overrides=chat_state_stream or None,
                )
            elif has_overrides and not is_resume:
                # Regeneration with filter/query/doc overrides
                agent = LangGraphRAGAgent(
                    llm=llm,
                    search_service=search_service,
                    prompt_spec=prompt_spec,
                    top_k=req.top_k,
                    retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
                    mode=req.mode,
                    ask_user_after_retrieve=False,
                    use_canonical_retrieval=req.use_canonical_retrieval,
                    checkpointer=_checkpointer,
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
                if is_resume and is_guided_resume:
                    agent = _new_guided_confirm_agent(
                        llm,
                        search_service,
                        prompt_spec,
                        top_k=req.top_k,
                        use_canonical_retrieval=req.use_canonical_retrieval,
                        event_sink=_enqueue,
                    )
                else:
                    agent = _new_hil_agent(
                        llm,
                        search_service,
                        prompt_spec,
                        use_canonical_retrieval=req.use_canonical_retrieval,
                        event_sink=_enqueue,
                    )
                config = {"configurable": {"thread_id": tid}}

                if is_resume:
                    state = agent._graph.get_state(config)
                    if state is None or not state.values:
                        _enqueue(
                            {
                                "type": "error",
                                "status": 400,
                                "detail": f"No checkpoint for thread_id={tid}. Server may have restarted.",
                            }
                        )
                        return
                    resume_payload = (
                        validated_guided_resume_decision
                        if is_guided_resume
                        else req.resume_decision
                    )
                    result = agent._graph.invoke(Command(resume=resume_payload), config)
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
                    retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
                    mode=req.mode,
                    ask_user_after_retrieve=False,
                    use_canonical_retrieval=req.use_canonical_retrieval,
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
            auto_parse_result = _extract_auto_parse_result(result)
            suggest_additional_device_search = _should_suggest_additional_device_search(
                result, auto_parse_result
            )
            final_search_queries = _to_str_list(result.get("search_queries"))
            interrupt_info = result.get("__interrupt__")
            if interrupt_info:
                payload = None
                if len(interrupt_info) > 0:
                    payload = (
                        interrupt_info[0].value if hasattr(interrupt_info[0], "value") else None
                    )

                resp = AgentResponse(
                    query=req.message,
                    answer=result.get("answer", "") or "",
                    reasoning=result.get("reasoning"),
                    judge=result.get("judge", {}) or {},
                    retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
                    all_retrieved_docs=_to_retrieved_docs(
                        result.get("all_docs") or result.get("docs", [])
                    ),
                    expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
                    metadata={
                        **_build_response_metadata(
                            result,
                            effective_mq_mode,
                            req.max_attempts,
                            final_search_queries,
                            index_name,
                        ),
                        "trace": trace_payload,
                        "response_mode": "retrieval_only" if req.retrieval_only else "full",
                    },
                    interrupted=True,
                    interrupt_payload=payload,
                    thread_id=tid,
                    auto_parse=auto_parse_result,
                    selected_devices=result.get("selected_devices"),
                    selected_doc_types=result.get("selected_doc_types"),
                    selected_equip_ids=result.get("selected_equip_ids"),
                    search_queries=final_search_queries,
                    detected_language=result.get("detected_language"),
                    suggest_additional_device_search=suggest_additional_device_search,
                    related_doc_types=_build_related_doc_type_suggestions(search_service),
                )
            else:
                resp = AgentResponse(
                    query=req.message,
                    answer=result.get("answer", ""),
                    reasoning=result.get("reasoning"),
                    judge=result.get("judge", {}),
                    retrieved_docs=_to_retrieved_docs(_select_display_docs(result)),
                    all_retrieved_docs=_to_retrieved_docs(
                        result.get("all_docs") or result.get("docs", [])
                    ),
                    expanded_docs=_to_expanded_docs(result.get("answer_ref_json")),
                    metadata={
                        **_build_response_metadata(
                            result,
                            effective_mq_mode,
                            req.max_attempts,
                            final_search_queries,
                            index_name,
                        ),
                        "trace": trace_payload,
                        "response_mode": "retrieval_only" if req.retrieval_only else "full",
                    },
                    interrupted=False,
                    interrupt_payload=None,
                    thread_id=tid,
                    auto_parse=auto_parse_result,
                    selected_devices=result.get("selected_devices"),
                    selected_doc_types=result.get("selected_doc_types"),
                    selected_equip_ids=result.get("selected_equip_ids"),
                    search_queries=final_search_queries,
                    detected_language=result.get("detected_language"),
                    suggest_additional_device_search=suggest_additional_device_search,
                    related_doc_types=_build_related_doc_type_suggestions(search_service),
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
        yield (
            f"data: {json.dumps({'type': 'open', 'thread_id': tid, 'trace': trace_payload}, ensure_ascii=False)}\n\n"
        )

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
