"""ReAct (Reasoning + Acting) RAG Agent.

기존 20+ 노드 DAG를 3-노드 루프로 대체한다.

  preprocess → plan → search → plan (loop, max MAX_SEARCH_ITERATIONS)
                     → answer → judge → END | plan (retry, max MAX_JUDGE_RETRIES)

중앙 plan_node LLM이 매 스텝 결정:
  - search        : 새 쿼리로 문서 검색
  - search_solution : 문제 원인 문서 발견 후 해결 절차 추가 검색
  - answer        : 수집된 문서로 최종 답변 생성

이점:
  - 대화 히스토리가 항상 LLM context에 포함 → 후속 질문 컨텍스트 유지
  - LLM이 런타임에 검색 전략 결정 → 사전 정의 안 된 질문 처리
  - 다중 검색 루프 → problem doc → solution doc 멀티 홉 지원

.run() 인터페이스는 LangGraphRAGAgent와 동일 → C-API-001 호환.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from pydantic import BaseModel as PydanticBaseModel

from backend.config.settings import agent_settings, rag_settings
from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.llm.langgraph_agent import (
    PromptSpec,
    SearchServiceRetriever,
    _merge_display_docs,
    answer_node,
    auto_parse_node,
    expand_related_docs_node,
    judge_node,
    load_prompt_spec,
    results_to_ref_json,
    retrieve_node,
    translate_node,
)
from backend.services.search_service import SearchService

logger = logging.getLogger(__name__)


def _infer_route(state: "ReactAgentState") -> str:  # type: ignore[name-defined]
    """parsed_query 또는 action_trace에서 route를 추정한다."""
    pq = state.get("parsed_query") or {}
    route = pq.get("route") or state.get("route")
    if route in ("setup", "ts", "general"):
        return route
    # doc_types 기반 추정
    doc_types = pq.get("selected_doc_types") or pq.get("doc_types") or []
    if any("sop" in str(d).lower() for d in doc_types):
        return "setup"
    if any("tsg" in str(d).lower() or "ts" in str(d).lower() for d in doc_types):
        return "ts"
    return "general"

# ── 상수 ────────────────────────────────────────────────────────────────────

MAX_SEARCH_ITERATIONS = 3   # 최대 검색 횟수 (초과 시 강제 answer)
MAX_JUDGE_RETRIES = 2       # judge unfaithful 시 최대 재시도 횟수
TEMP_PLAN = 0.0             # planner 온도 (결정론적)


# ── NextAction 모델 (BaseLLM.generate(response_model=NextAction) 용) ──────────

class NextAction(PydanticBaseModel):
    """planner LLM이 결정하는 다음 행동."""
    action: Literal["search", "search_solution", "answer"]
    reason: str
    query: Optional[str] = None
    device_names: List[str] = []
    doc_types: List[str] = []

# ── State ────────────────────────────────────────────────────────────────────


class ReactAgentState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────
    query: str
    chat_history: List[Dict[str, Any]]
    max_attempts: int
    thread_id: str
    retrieval_only: bool

    # ── Preprocessing (auto_parse + translate) ───────────────────
    detected_language: Optional[str]
    target_language: Optional[str]
    query_en: Optional[str]
    query_ko: Optional[str]
    parsed_query: Dict[str, Any]   # device_names, doc_types, equip_ids, route …
    original_query: str

    # ── ReAct 루프 ────────────────────────────────────────────────
    plan: Optional[Dict[str, Any]]    # planner 결정 JSON
    iterations: int                   # 실행된 검색 횟수
    action_trace: List[Dict[str, Any]]  # 검색 이력 (디버깅/프롬프트용)
    collected_docs: List[Any]           # 누적 RetrievalResult
    search_queries_used: List[str]      # 실제 사용된 쿼리 목록

    # ── C-API-001 메타데이터 ──────────────────────────────────────
    route: Optional[str]
    st_gate: Optional[str]
    mq_used: bool
    mq_reason: Optional[str]
    mq_mode: str
    attempts: int
    retry_strategy: Optional[str]
    search_queries: List[str]
    guardrail_dropped_numeric: int
    guardrail_dropped_anchor: int
    guardrail_final_count: int

    # ── 출력 ──────────────────────────────────────────────────────
    answer: str
    display_docs: List[Any]
    answer_ref_json: List[Dict[str, Any]]
    docs: List[Any]
    all_docs: List[Any]
    ref_json: List[Dict[str, Any]]
    judge: Dict[str, Any]

    # ── 내부 ──────────────────────────────────────────────────────
    _skip_human_review: bool


# ── Planner 프롬프트 ──────────────────────────────────────────────────────────

_PLAN_SYSTEM = """\
당신은 반도체 PE(Process Engineering) 트러블슈팅 전문 에이전트입니다.

사용자 질문, 대화 맥락, 수집된 문서를 분석해 다음 행동을 결정하세요.

행동 종류:
- search         : 문서를 검색 (처음이거나 현재 문서가 부적절할 때)
- search_solution: 문제 원인 문서는 있지만 해결 절차 문서가 없을 때 추가 검색
- answer         : 수집된 문서로 답변 생성 가능할 때

판단 기준:
- 수집된 문서가 질문의 **구체적 행위**(교체/청소/조정/캘리브레이션 등)에 직접 관련되는지 확인하세요.
- 문서가 같은 부품이라도 다른 행위(예: 교체를 물어봤는데 청소 문서만 있음)에 대한 것이면 search로 재검색하세요.
- doc_id에 행위 유형이 포함됩니다: rep=교체(replacement), cln=청소(cleaning), adj=조정(adjustment), cal=캘리브레이션. 질문의 행위와 doc_id의 행위가 일치하는지 확인하세요.
- reranker score가 낮은 문서(score < 0.1)가 대부분이면, 검색어를 더 구체적으로 변경하여 재검색하세요.
"""

_PLAN_USER_TMPL = """\
## 사용자 질문
{query}

## 대화 맥락
{history}

## 장치/필터 컨텍스트
장치: {devices}
문서 타입: {doc_types}

## 검색 이력 ({iterations}/{max_iter} 완료, 남은 횟수: {remaining})
{action_trace}

## 수집된 문서 ({doc_count}개 청크)
{doc_summary}
{judge_feedback}
다음 행동을 JSON으로 결정하세요.\
"""


def _format_history(chat_history: List[Dict[str, Any]]) -> str:
    """ChatHistoryTurn(user_text/assistant_text) 또는 일반 role/content 형식을 모두 처리."""
    if not chat_history:
        return "(없음)"
    lines = []
    for turn in chat_history[-6:]:   # 최근 3턴 (6개 메시지)
        # ChatHistoryTurn 형식 (agent.py가 model_dump()로 넘기는 형식)
        if "user_text" in turn:
            user = str(turn.get("user_text", ""))[:200]
            asst = str(turn.get("assistant_text", ""))[:200]
            if user:
                lines.append(f"사용자: {user}")
            if asst:
                lines.append(f"에이전트: {asst}")
        else:
            # 일반 role/content 형식
            role = turn.get("role", "user")
            content = str(turn.get("content", ""))[:200]
            prefix = "사용자" if role == "user" else "에이전트"
            lines.append(f"{prefix}: {content}")
    return "\n".join(lines) if lines else "(없음)"


def _format_action_trace(trace: List[Dict[str, Any]]) -> str:
    if not trace:
        return "(검색 없음)"
    lines = []
    for i, act in enumerate(trace, 1):
        action = act.get("action", "search")
        query = act.get("query", "")
        found = act.get("found", 0)
        lines.append(f"{i}. [{action}] 쿼리='{query}' → {found}개 청크 수집")
    return "\n".join(lines)


def _format_doc_summary(docs: List[Any]) -> str:
    """수집된 문서를 간략히 요약 (planner 프롬프트용). reranker score 포함."""
    if not docs:
        return "(없음)"
    seen: Dict[str, str] = {}
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        doc_id = meta.get("doc_id") or meta.get("id", "")
        title = (
            meta.get("title")
            or meta.get("file_name")
            or meta.get("doc_type", "")
        )
        device = meta.get("device_name", "")
        score = getattr(doc, "score", None) or meta.get("score")
        if doc_id and doc_id not in seen:
            label = f"[{device}] {doc_id}"
            if title:
                label += f" | {title}"
            if score is not None:
                try:
                    label += f" (score={float(score):.3f})"
                except (ValueError, TypeError):
                    pass
            seen[doc_id] = label

    summaries = [f"- {v}" for v in list(seen.values())[:10]]
    if len(seen) > 10:
        summaries.append(f"  ... (총 {len(seen)}개 문서)")
    return "\n".join(summaries) if summaries else "(없음)"


def _plan_fallback(has_docs: bool = False) -> Dict[str, Any]:
    """planner 실패 시 기본 plan. 문서가 없으면 search, 있으면 answer."""
    if has_docs:
        return {"action": "answer", "reason": "plan generation failed; using collected docs"}
    return {"action": "search", "reason": "plan generation failed; fallback to search"}


# ── 장비명 토큰 기반 확장 ─────────────────────────────────────────────────────

def _expand_device_names_by_tokens(
    selected: List[str],
    all_devices: List[str],
) -> List[str]:
    """선택된 장비명의 토큰이 모두 포함된 다른 장비명 변형을 자동 확장한다.

    예: selected=["GENEVA XP"] → "GENEVA STP300 xp", "GENEVA_STP300_XP" 등 추가.
    ES terms 필터가 exact match이므로, 변형을 미리 추가해야 검색됨.
    """
    expanded: List[str] = list(selected)
    expanded_lower = {d.lower() for d in expanded}

    for sel in selected:
        # 선택 장비명을 토큰으로 분리 (공백, _, -, / 등으로)
        sel_norm = sel.strip().lower().replace("_", " ").replace("-", " ")
        sel_tokens = set(sel_norm.split())
        if not sel_tokens or len(sel_tokens) < 1:
            continue

        for candidate in all_devices:
            cand_lower = candidate.strip().lower()
            if cand_lower in expanded_lower:
                continue
            # 후보를 토큰으로 분리
            cand_norm = cand_lower.replace("_", " ").replace("-", " ")
            cand_tokens = set(cand_norm.split())
            # 선택 장비의 모든 토큰이 후보에 포함되면 같은 계열
            if sel_tokens <= cand_tokens:
                expanded.append(candidate)
                expanded_lower.add(cand_lower)

    if len(expanded) > len(selected):
        logger.info(
            "device token expansion: %s → %s (%d variants added)",
            selected, expanded[:5], len(expanded) - len(selected),
        )
    return expanded


# ── 행위 유형 필터링 ──────────────────────────────────────────────────────────

# 질문에서 행위 의도를 감지하는 키워드 → doc_id action code 매핑
_ACTION_KEYWORDS: Dict[str, List[str]] = {
    "rep": ["교체", "replacement", "replace", "교환", "바꾸", "바꿔"],
    "cln": ["청소", "clean", "cleaning", "세정", "세척"],
    "adj": ["조정", "adjust", "adjustment", "조절", "정렬", "align"],
    "cal": ["캘리브레이션", "calibration", "calibrate", "교정", "보정"],
}

# doc_id에서 action code를 추출하는 패턴 (e.g. ..._rep_..., ..._cln_...)
_DOC_ID_ACTION_RE = re.compile(r"_(rep|cln|adj|cal)_")


def _detect_action_intent(query: str) -> Optional[str]:
    """질문에서 행위 의도를 감지하여 action code를 반환한다."""
    q_lower = query.lower()
    for action_code, keywords in _ACTION_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return action_code
    return None


def _filter_docs_by_action_type(
    docs: List[Any],
    query: str,
) -> List[Any]:
    """질문의 행위 의도와 doc_id의 action code가 불일치하는 문서를 후순위로 밀어낸다.

    완전 제거하지 않고, 불일치 문서를 리스트 뒤로 이동시킨다.
    (같은 부품의 다른 행위 문서도 참고 가치가 있을 수 있으므로)
    """
    intent = _detect_action_intent(query)
    if not intent:
        return docs  # 행위 의도가 불명확하면 필터링 안 함

    matched: List[Any] = []
    unmatched: List[Any] = []
    neutral: List[Any] = []  # doc_id에 action code가 없는 문서

    for doc in docs:
        doc_id = (
            getattr(doc, "doc_id", None)
            or (getattr(doc, "metadata", {}) or {}).get("doc_id", "")
        )
        m = _DOC_ID_ACTION_RE.search(str(doc_id))
        if not m:
            neutral.append(doc)
        elif m.group(1) == intent:
            matched.append(doc)
        else:
            unmatched.append(doc)

    if matched or unmatched:
        logger.info(
            "action_type_filter: intent=%s matched=%d unmatched=%d neutral=%d",
            intent, len(matched), len(unmatched), len(neutral),
        )

    # 매칭 문서 우선, 중립 문서 다음, 불일치 문서 마지막
    return matched + neutral + unmatched


# ── ReactRAGAgent ─────────────────────────────────────────────────────────────


class ReactRAGAgent:
    """ReAct 루프 기반 RAG 에이전트.

    LangGraphRAGAgent와 동일한 .run() 인터페이스를 제공한다.
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        search_service: SearchService,
        prompt_spec: Optional[PromptSpec] = None,
        top_k: int = 20,
        retrieval_top_k: int = 50,
        checkpointer: Optional[MemorySaver] = None,
        device_names: Optional[List[str]] = None,
        doc_type_names: Optional[List[str]] = None,
        equip_id_set: Optional[set] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.llm = llm
        self.search_service = search_service
        self.retriever = SearchServiceRetriever(search_service, top_k=retrieval_top_k)
        self.spec = prompt_spec or load_prompt_spec(version=rag_settings.prompt_spec_version)
        self.top_k = top_k
        self.retrieval_top_k = retrieval_top_k
        self.reranker = getattr(search_service, "reranker", None)
        self.page_fetcher = getattr(search_service, "fetch_doc_pages", None)
        self.doc_fetcher = getattr(search_service, "fetch_doc_chunks", None)
        _es_engine = getattr(search_service, "es_engine", None)
        self.section_fetcher = getattr(_es_engine, "fetch_section_chunks", None)
        self.checkpointer = checkpointer or MemorySaver()
        self._device_names: List[str] = device_names or []
        self._doc_type_names: List[str] = doc_type_names or []
        self._equip_id_set: set = equip_id_set or set()
        self._event_sink = event_sink

        self._graph = self._build_graph()

    # ── 이벤트 ────────────────────────────────────────────────────────────────

    def _emit(self, event: Dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(event)
        except Exception:
            logger.exception("react_agent: event_sink failed")

    def _emit_node(self, name: str, elapsed_ms: float, details: str = "") -> None:
        elapsed_str = f"{elapsed_ms / 1000:.1f}s" if elapsed_ms >= 1000 else f"{elapsed_ms:.0f}ms"
        self._emit({
            "type": "log",
            "level": "info",
            "node": name,
            "ts": time.time(),
            "elapsed_ms": round(elapsed_ms, 1),
            "details": details,
            "message": f"{name} ({elapsed_str})" + (f" - {details}" if details else ""),
        })

    # ── 노드 구현 ─────────────────────────────────────────────────────────────

    def _preprocess_node(self, state: ReactAgentState) -> Dict[str, Any]:
        """auto_parse + translate 재사용. 기존 AgentState 호환 래퍼."""
        t0 = time.perf_counter()
        # auto_parse_node는 AgentState를 기대하므로 호환 state 구성
        compat: Dict[str, Any] = {
            "query": state["query"],
            "parsed_query": state.get("parsed_query") or {},
        }
        ap_result = auto_parse_node(
            cast(Any, compat),
            llm=self.llm,
            spec=self.spec,
            device_names=self._device_names,
            doc_type_names=self._doc_type_names,
            equip_id_set=self._equip_id_set,
        )
        update: Dict[str, Any] = {}
        update.update(ap_result)

        # translate
        compat2: Dict[str, Any] = {**compat, **update}
        tr_result = translate_node(cast(Any, compat2), llm=self.llm, spec=self.spec)
        update.update(tr_result)

        # route 분류 (rule-based): preprocess 단계에서 명시적으로 결정
        pq = update.get("parsed_query") or {}
        if not pq.get("route"):
            doc_types = pq.get("selected_doc_types") or pq.get("doc_types") or []
            dt_lower = [str(d).lower() for d in doc_types]
            if any("sop" in d or "setup" in d for d in dt_lower):
                route = "setup"
            elif any("tsg" in d or "ts" == d for d in dt_lower):
                route = "ts"
            else:
                route = "general"
            pq["route"] = route
            update["parsed_query"] = pq
            update["route"] = route

        # 트러블슈팅 질문 시 doc_types를 myservice/gcb/ts로 확장
        # (이상 원인, 점검/조치 이력 등 문제 해결 질문은 세 doc_type 모두 검색 필요)
        raw_query = state["query"]
        _TS_KEYWORDS = ("트러블슈팅", "troubleshoot", "이상", "abnormal", "점검", "조치", "이력", "알람", "에러", "error", "alarm")
        if any(kw in raw_query.lower() for kw in _TS_KEYWORDS):
            current_dt = pq.get("selected_doc_types") or pq.get("doc_types") or []
            ts_types = {"myservice", "gcb", "ts"}
            expanded = list(ts_types | {str(d).lower() for d in current_dt})
            pq["selected_doc_types"] = expanded
            pq["doc_types"] = expanded
            if not pq.get("route") or pq.get("route") == "general":
                pq["route"] = "ts"
            update["parsed_query"] = pq
            update["route"] = pq["route"]

        elapsed = (time.perf_counter() - t0) * 1000
        devices = (pq.get("device_names") or [])[:2]
        lang = update.get("detected_language", "")
        route_label = pq.get("route", "?")
        self._emit_node("preprocess", elapsed, f"lang={lang} devices={devices} route={route_label}")
        return update

    def _plan_node(self, state: ReactAgentState) -> Dict[str, Any]:
        """중앙 LLM이 다음 행동을 결정한다."""
        t0 = time.perf_counter()
        query = state.get("query_en") or state.get("query", "")
        iterations = int(state.get("iterations", 0) or 0)
        max_iter = MAX_SEARCH_ITERATIONS

        pq = state.get("parsed_query") or {}
        devices = pq.get("selected_devices") or pq.get("device_names") or []
        doc_types = pq.get("selected_doc_types") or pq.get("doc_types") or []

        history_text = _format_history(state.get("chat_history") or [])
        action_trace_text = _format_action_trace(state.get("action_trace") or [])
        doc_summary = _format_doc_summary(state.get("collected_docs") or [])
        doc_count = len(state.get("collected_docs") or [])
        remaining = max(0, max_iter - iterations)

        # judge retry 시 피드백 구성
        judge_hint = state.get("_judge_retry_hint") or ""
        judge_issues = state.get("_judge_retry_issues") or []
        if judge_hint or judge_issues:
            feedback_lines = ["\n## ⚠️ 이전 답변 품질 문제 (재시도 필요)"]
            if judge_hint:
                feedback_lines.append(f"판정 힌트: {judge_hint}")
            if judge_issues:
                feedback_lines.append("문제점:")
                for issue in judge_issues[:5]:
                    feedback_lines.append(f"  - {issue}")
            feedback_lines.append("→ 다른 검색어나 다른 문서 타입으로 재검색하세요.")
            judge_feedback = "\n".join(feedback_lines)
        else:
            judge_feedback = ""

        user = _PLAN_USER_TMPL.format(
            query=query,
            history=history_text,
            devices=", ".join(devices) if devices else "(없음)",
            doc_types=", ".join(doc_types) if doc_types else "(없음)",
            iterations=iterations,
            max_iter=max_iter,
            remaining=remaining,
            action_trace=action_trace_text,
            doc_count=doc_count,
            doc_summary=doc_summary,
            judge_feedback=judge_feedback,
        )

        has_docs = bool(state.get("collected_docs"))

        # 최대 반복 횟수 초과 시 강제 answer
        if iterations >= max_iter:
            plan: Dict[str, Any] = {
                "action": "answer",
                "reason": f"max iterations ({max_iter}) reached",
            }
        else:
            try:
                next_action: NextAction = self.llm.generate(
                    messages=[
                        {"role": "system", "content": _PLAN_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    response_model=NextAction,
                    temperature=TEMP_PLAN,
                )
                plan = next_action.model_dump()
            except Exception:
                logger.warning("react_agent: plan generation failed, trying raw JSON parse", exc_info=True)
                # structured output 실패 시 raw JSON 파싱 시도
                plan = self._try_raw_plan_parse(user) or _plan_fallback(has_docs)

        elapsed = (time.perf_counter() - t0) * 1000
        self._emit_node("plan", elapsed, f"→ {plan.get('action')} | {plan.get('reason', '')[:60]}")
        return {"plan": plan}

    def _try_raw_plan_parse(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """structured output 실패 시 raw text → JSON 파싱 시도."""
        try:
            raw_resp = self.llm.generate(
                messages=[
                    {"role": "system", "content": _PLAN_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMP_PLAN,
            )
            text = raw_resp.text if hasattr(raw_resp, "text") else str(raw_resp)
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
            if not json_match:
                return None
            parsed = json.loads(json_match.group(0))
            if not isinstance(parsed, dict) or "action" not in parsed:
                return None
            action = parsed["action"]
            valid_actions = ("search", "search_solution", "answer")
            if action not in valid_actions:
                action = "search"
            return {
                "action": action,
                "reason": parsed.get("reason", "raw parse fallback"),
                "query": parsed.get("query"),
                "device_names": parsed.get("device_names", []),
                "doc_types": parsed.get("doc_types", []),
            }
        except Exception:
            logger.debug("react_agent: raw plan parse also failed", exc_info=True)
            return None

    def _search_node(self, state: ReactAgentState) -> Dict[str, Any]:
        """plan에 따라 문서를 검색하고 collected_docs에 누적한다."""
        t0 = time.perf_counter()
        plan = state.get("plan") or {}
        search_query = plan.get("query") or state.get("query_en") or state.get("query", "")

        # AbbreviationExpander: 약어를 풀네임으로 확장 (0ms, rule-based)
        if agent_settings.abbreviation_expand_enabled:
            try:
                from backend.llm_infrastructure.query_expansion.abbreviation_expander import (
                    get_abbreviation_expander,
                )
                _expander = get_abbreviation_expander(agent_settings.abbreviation_dict_path)
                expand_result = _expander.expand_query(search_query)
                if expand_result.expanded_query != search_query:
                    logger.info(
                        "react_agent: abbreviation expanded '%s' → '%s'",
                        search_query[:60], expand_result.expanded_query[:60],
                    )
                    search_query = expand_result.expanded_query
            except Exception:
                logger.debug("react_agent: abbreviation expansion failed", exc_info=True)

        # 필터: plan 우선, parsed_query 폴백
        pq = state.get("parsed_query") or {}
        device_names: List[str] = plan.get("device_names") or pq.get("selected_devices") or pq.get("device_names") or []
        doc_types: List[str] = plan.get("doc_types") or pq.get("selected_doc_types") or pq.get("doc_types") or []
        equip_ids: List[str] = pq.get("selected_equip_ids") or pq.get("equip_ids") or []

        # 장비명 토큰 기반 확장: "GENEVA XP" → "GENEVA STP300 xp", "GENEVA_STP300_XP" 등 포함
        # ES terms 필터가 exact match이므로, known device list에서 토큰이 모두 포함된 변형을 추가
        if device_names and self._device_names:
            device_names = _expand_device_names_by_tokens(device_names, self._device_names)

        # AgentState 호환 state 구성 후 retrieve_node 재사용
        compat: Dict[str, Any] = {
            "query": search_query,
            "query_en": state.get("query_en") or search_query,
            "query_ko": state.get("query_ko") or state.get("query", ""),
            "search_queries": [search_query],
            "selected_devices": device_names,
            "selected_doc_types": doc_types,
            "selected_equip_ids": equip_ids,
            "parsed_query": {
                **pq,
                "device_names": device_names,
                "doc_types": doc_types,
                "equip_ids": equip_ids,
                "selected_devices": device_names,
                "selected_doc_types": doc_types,
                "selected_equip_ids": equip_ids,
            },
        }

        retrieve_result = retrieve_node(
            cast(Any, compat),
            retriever=self.retriever,
            reranker=self.reranker,
            retrieval_top_k=self.retrieval_top_k,
            final_top_k=self.top_k,
        )

        new_docs: List[Any] = retrieve_result.get("docs") or []

        # 행위 유형 필터링: 질문의 행위(교체/청소/조정 등)와 doc_id의 action code 매칭
        # doc_id 패턴: ..._rep_... (교체), ..._cln_... (청소), ..._adj_... (조정), ..._cal_... (캘리브레이션)
        new_docs = _filter_docs_by_action_type(
            new_docs,
            state.get("query") or state.get("query_ko") or "",
        )

        # 기존 collected_docs에 중복 없이 누적
        existing = state.get("collected_docs") or []
        existing_ids = {
            getattr(d, "doc_id", None) or (getattr(d, "metadata", {}) or {}).get("doc_id")
            for d in existing
        }
        added = [
            d for d in new_docs
            if (
                getattr(d, "doc_id", None)
                or (getattr(d, "metadata", {}) or {}).get("doc_id")
            ) not in existing_ids
        ]
        combined = existing + added

        # 이력 기록
        trace = list(state.get("action_trace") or [])
        trace.append({
            "action": plan.get("action", "search"),
            "query": search_query,
            "found": len(new_docs),
            "added": len(added),
        })

        queries_used = list(state.get("search_queries_used") or [])
        queries_used.append(search_query)

        update: Dict[str, Any] = {
            "collected_docs": combined,
            "action_trace": trace,
            "iterations": int(state.get("iterations", 0) or 0) + 1,
            "search_queries_used": queries_used,
            # guardrail 메타데이터 갱신
            "guardrail_dropped_numeric": int(
                retrieve_result.get("guardrail_dropped_numeric", 0) or 0
            ),
            "guardrail_dropped_anchor": int(
                retrieve_result.get("guardrail_dropped_anchor", 0) or 0
            ),
            "guardrail_final_count": len(combined),
        }

        elapsed = (time.perf_counter() - t0) * 1000
        self._emit_node(
            "search",
            elapsed,
            f"q='{search_query[:40]}' found={len(new_docs)} added={len(added)}",
        )

        # C-API-003: retrieval_only=True 시 answer 생성 전 interrupt
        if state.get("retrieval_only"):
            display = _merge_display_docs(combined)
            update["display_docs"] = display
            update["docs"] = combined
            interrupt({
                "type": "retrieval_review",
                "docs": display,
                "response_mode": "retrieval_only",
            })

        return update

    def _answer_node(self, state: ReactAgentState) -> Dict[str, Any]:
        """수집된 문서로 최종 답변을 생성한다. 기존 answer_node 재사용."""
        t0 = time.perf_counter()
        docs = state.get("collected_docs") or []

        # retrieved docs를 ref_json으로 변환
        ref_json = results_to_ref_json(docs)

        # expand_related_docs_node 재사용: 챕터/섹션 확장
        expand_compat: Dict[str, Any] = {
            "query": state.get("query_en") or state.get("query", ""),
            "docs": docs,
            "all_docs": docs,
            "ref_json": ref_json,
            "route": _infer_route(state),
            "detected_language": state.get("detected_language") or "ko",
            "target_language": state.get("target_language"),
            "query_en": state.get("query_en"),
            "query_ko": state.get("query_ko"),
            "parsed_query": state.get("parsed_query") or {},
            "selected_doc_types": (state.get("parsed_query") or {}).get("selected_doc_types") or [],
        }
        try:
            expand_result = expand_related_docs_node(
                cast(Any, expand_compat),
                page_fetcher=self.page_fetcher,
                doc_fetcher=self.doc_fetcher,
                section_fetcher=self.section_fetcher,
            )
            ref_json = expand_result.get("answer_ref_json") or ref_json
        except Exception:
            logger.debug("react_agent: expand_related_docs failed", exc_info=True)

        # answer_node 재사용
        answer_compat: Dict[str, Any] = {
            **expand_compat,
            "answer_ref_json": ref_json,
            "ref_json": ref_json,
            "docs": docs,
            "all_docs": docs,
            "task_mode": "",
            "attempts": state.get("attempts", 0),
            "max_attempts": state.get("max_attempts", 1),
        }
        ans_result = answer_node(cast(Any, answer_compat), llm=self.llm, spec=self.spec)

        display = _merge_display_docs(docs)
        elapsed = (time.perf_counter() - t0) * 1000
        answer_len = len(ans_result.get("answer") or "")
        self._emit_node("answer", elapsed, f"{answer_len}자")
        return {
            "answer": ans_result.get("answer", ""),
            "answer_ref_json": ref_json,
            "ref_json": ref_json,
            "docs": docs,
            "all_docs": docs,
            "display_docs": display,
        }

    def _judge_node(self, state: ReactAgentState) -> Dict[str, Any]:
        """기존 judge_node 재사용. unfaithful 시 retry 힌트를 plan에 전달."""
        t0 = time.perf_counter()
        route = _infer_route(state)
        attempts = int(state.get("attempts", 0) or 0)
        max_attempts = int(state.get("max_attempts", MAX_JUDGE_RETRIES) or MAX_JUDGE_RETRIES)
        judge_compat: Dict[str, Any] = {
            "query": state.get("query", ""),
            "query_en": state.get("query_en"),
            "route": route,
            "answer": state.get("answer", ""),
            "answer_ref_json": state.get("answer_ref_json") or [],
            "ref_json": state.get("ref_json") or [],
            "attempts": attempts,
            "max_attempts": max_attempts,
        }
        result = judge_node(cast(Any, judge_compat), llm=self.llm, spec=self.spec)
        judge_result = result.get("judge") or {}
        faithful = bool(judge_result.get("faithful", False))

        # attempts 증가
        result["attempts"] = attempts + 1

        # unfaithful 시 judge hint를 plan에 전달하여 재검색/재답변 유도
        if not faithful and (attempts + 1) < max_attempts:
            hint = judge_result.get("hint", "")
            issues = judge_result.get("issues", [])
            logger.info(
                "react_agent: judge unfaithful (attempt %d/%d), will retry. hint=%s issues=%s",
                attempts + 1, max_attempts, hint[:100], issues[:3],
            )
            result["_judge_retry_hint"] = hint
            result["_judge_retry_issues"] = issues

        elapsed = (time.perf_counter() - t0) * 1000
        self._emit_node("judge", elapsed, "✓ 충실" if faithful else f"✗ 불충실 (attempt {attempts + 1}/{max_attempts})")
        return result

    @staticmethod
    def _route_after_judge(state: ReactAgentState) -> str:
        """Judge 결과에 따라 retry 또는 종료를 결정한다."""
        judge = state.get("judge") or {}
        faithful = bool(judge.get("faithful", False))
        attempts = int(state.get("attempts", 0) or 0)
        max_attempts = int(state.get("max_attempts", MAX_JUDGE_RETRIES) or MAX_JUDGE_RETRIES)

        if faithful or attempts >= max_attempts:
            return "done"
        return "retry"

    # ── 라우팅 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_plan(state: ReactAgentState) -> str:
        plan = state.get("plan") or {}
        action = plan.get("action", "answer")
        if action in ("search", "search_solution"):
            return "search"
        return "answer"

    # ── 그래프 조립 ───────────────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(ReactAgentState)  # type: ignore[type-var]

        builder.add_node("preprocess", self._preprocess_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("search", self._search_node)
        builder.add_node("answer", self._answer_node)
        builder.add_node("judge", self._judge_node)

        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "plan")
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"search": "search", "answer": "answer"},
        )
        builder.add_edge("search", "plan")   # 루프: search → plan → search | answer
        builder.add_edge("answer", "judge")
        builder.add_conditional_edges(
            "judge",
            self._route_after_judge,
            {"done": END, "retry": "plan"},
        )

        return builder.compile(checkpointer=self.checkpointer)

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        *,
        attempts: int = 0,
        max_attempts: int = MAX_JUDGE_RETRIES,
        thread_id: Optional[str] = None,
        state_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """그래프 실행. LangGraphRAGAgent.run()과 동일한 인터페이스."""
        tid = thread_id or str(uuid.uuid4())

        state: ReactAgentState = {  # type: ignore[assignment]
            "query": query,
            "original_query": query,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "thread_id": tid,
            "iterations": 0,
            "action_trace": [],
            "collected_docs": [],
            "search_queries_used": [],
            "mq_used": False,
            "mq_reason": None,
            "mq_mode": "react",
            "route": None,
            "st_gate": None,
            "retry_strategy": None,
            "search_queries": [],
            "guardrail_dropped_numeric": 0,
            "guardrail_dropped_anchor": 0,
            "guardrail_final_count": 0,
            "_skip_human_review": True,
        }

        # state_overrides: chat_history, parsed_query, selected_* 등
        if state_overrides:
            state.update(cast(Any, state_overrides))  # type: ignore[arg-type]

        config: Dict[str, Any] = kwargs.pop("config", {})
        config = {
            **config,
            "configurable": {**config.get("configurable", {}), "thread_id": tid},
            "recursion_limit": 50,
        }

        result: Dict[str, Any] = self._graph.invoke(state, config=cast(Any, config))

        # ── C-API-001 메타데이터 정규화 ────────────────────────────────────
        # search_queries: 실제 사용된 쿼리 목록
        queries_used: List[str] = result.get("search_queries_used") or []
        result["search_queries"] = queries_used

        # mq_used: 검색을 2회 이상 했으면 True
        iterations = int(result.get("iterations", 0) or 0)
        result["mq_used"] = iterations > 1
        result["mq_reason"] = (
            f"react loop: {iterations} search(es)" if iterations > 1 else None
        )
        # mq_mode: state_overrides로 주입된 값 우선, 없으면 "react"
        if not result.get("mq_mode"):
            result["mq_mode"] = "react"

        # search_queries_raw semantics:
        # _sanitize_search_queries_raw()는 general_mq_list를 fallback으로 읽는다.
        # planner 쿼리 = 최종 쿼리이므로 general_mq_list = search_queries_used 로 동등 처리.
        result["general_mq_list"] = queries_used

        # selected_doc_types: parsed_query에서 읽어서 보조 metadata 채우기
        pq = result.get("parsed_query") or {}
        if not result.get("selected_doc_types"):
            result["selected_doc_types"] = (
                pq.get("selected_doc_types") or pq.get("doc_types") or []
            )

        # route: parsed_query에서 읽거나 "general" 기본값
        if not result.get("route"):
            pq = result.get("parsed_query") or {}
            result["route"] = pq.get("route") or "general"

        # st_gate 기본값
        if not result.get("st_gate"):
            result["st_gate"] = "no_st"

        # attempts
        result["attempts"] = result.get("attempts", attempts)
        result["retry_strategy"] = result.get("retry_strategy")

        return result
