"""LangGraph node helpers + prompt spec.

이 파일은 노드/프롬프트 스펙/헬퍼를 제공하고, 실제 그래프 조립은
service 계층에서 담당한다.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict

from langgraph.types import Command, interrupt

from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate, load_prompt_template
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.search_service import SearchService

logger = logging.getLogger(__name__)


# -----------------------------
# 1) State schema
# -----------------------------
Route = Literal["setup", "ts", "general"]
Gate = Literal["need_st", "no_st"]


class AgentState(TypedDict, total=False):
    query: str
    route: Route

    # Multi-query outputs (raw strings)
    setup_mq: str
    ts_mq: str
    general_mq: str

    st_gate: Gate
    search_queries: List[str]

    # Retrieval outputs
    docs: List[RetrievalResult]
    ref_json: List[Dict[str, Any]]

    # Answer + judge
    answer: str
    judge: Dict[str, Any]

    # Retry / HIL
    attempts: int
    max_attempts: int
    human_action: Optional[Dict[str, Any]]


# -----------------------------
# 2) Prompt loading
# -----------------------------
@dataclass
class PromptSpec:
    router: PromptTemplate
    setup_mq: PromptTemplate
    ts_mq: PromptTemplate
    general_mq: PromptTemplate
    st_gate: PromptTemplate
    st_mq: PromptTemplate
    setup_ans: PromptTemplate
    ts_ans: PromptTemplate
    general_ans: PromptTemplate
    judge_setup_sys: str
    judge_ts_sys: str
    judge_general_sys: str


DEFAULT_JUDGE_SETUP = """
# 역할
설치/세팅 답변이 질문과 검색 증거에 충실한지 판정한다.

# 입력
- 질문, 답변, REF_JSON (검색 결과)

# 출력
JSON 한 줄: {"faithful": bool, "issues": ["..."], "hint": "..."}
- faithful: 근거 기반이면 true, 아니면 false
- issues: 부족한 근거/누락된 단계/잘못된 조치 등
- hint: 재검색/보강에 도움이 되는 한 줄
""".strip()

DEFAULT_JUDGE_TS = """
# 역할
TS 답변이 질문과 검색 증거에 근거했는지 판정한다.
출력: {"faithful": bool, "issues": [..], "hint": "..."}
faithful이 false면 조치/근거 누락, 불일치 등을 issues에 적는다.
""".strip()

DEFAULT_JUDGE_GENERAL = """
# 역할
일반 답변이 질문 의도와 검색 증거를 충실히 반영했는지 판정한다.
출력: {"faithful": bool, "issues": [..], "hint": "..."}
""".strip()


def load_prompt_spec(version: str = "v1") -> PromptSpec:
    """Load required prompts from YAML (router/MQ/gate/answer)."""

    router = load_prompt_template("router", version)
    setup_mq = load_prompt_template("setup_mq", version)
    ts_mq = load_prompt_template("ts_mq", version)
    general_mq = load_prompt_template("general_mq", version)
    st_gate = load_prompt_template("st_gate", version)
    st_mq = load_prompt_template("st_mq", version)
    setup_ans = load_prompt_template("setup_ans", version)
    ts_ans = load_prompt_template("ts_ans", version)
    general_ans = load_prompt_template("general_ans", version)

    return PromptSpec(
        router=router,
        setup_mq=setup_mq,
        ts_mq=ts_mq,
        general_mq=general_mq,
        st_gate=st_gate,
        st_mq=st_mq,
        setup_ans=setup_ans,
        ts_ans=ts_ans,
        general_ans=general_ans,
        judge_setup_sys=DEFAULT_JUDGE_SETUP,
        judge_ts_sys=DEFAULT_JUDGE_TS,
        judge_general_sys=DEFAULT_JUDGE_GENERAL,
    )


# -----------------------------
# 3) Interfaces
# -----------------------------
class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int = 8) -> List[RetrievalResult]:
        ...


class SearchServiceRetriever:
    """Adapter to reuse existing SearchService inside the graph."""

    def __init__(self, search_service: SearchService, *, top_k: int = 8) -> None:
        self.search_service = search_service
        self.top_k = top_k

    def retrieve(self, query: str, *, top_k: int | None = None) -> List[RetrievalResult]:
        k = top_k or self.top_k
        # 그래프 레벨에서 MQ/재시도를 수행하므로 내부 MQ/rerank는 끈다.
        return self.search_service.search(
            query,
            top_k=k,
            multi_query=False,
            rerank=False,
        )


# -----------------------------
# 4) LLM helpers
# -----------------------------
def _invoke_llm(llm: BaseLLM, system: str, user: str, **kwargs: Any) -> str:
    messages: List[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    out = llm.generate(messages, **kwargs)
    return out.text.strip()


def _format_prompt(template: str, mapping: Dict[str, str]) -> str:
    """Lightweight placeholder replacement without raising on missing keys."""
    rendered = template
    for key, val in mapping.items():
        rendered = rendered.replace(f"{{{key}}}", val)
    return rendered


def _parse_route(text: str) -> Route:
    t = text.strip().lower()
    if t in ("setup", "ts", "general"):
        return t  # type: ignore[return-value]
    m = re.search(r"\b(setup|ts|general)\b", t)
    if m:
        return m.group(1)  # type: ignore[return-value]
    return "general"


def _parse_gate(text: str) -> Gate:
    t = text.strip().lower()
    if t in ("need_st", "no_st"):
        return t  # type: ignore[return-value]
    if "need" in t:
        return "need_st"
    return "no_st"


def _parse_queries(text: str) -> List[str]:
    """Robust parser: JSON object/list or one-per-line strings."""
    t = text.strip()

    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
            qs = [str(x).strip() for x in obj["queries"] if str(x).strip()]
            return qs[:5]
        if isinstance(obj, list):
            qs = [str(x).strip() for x in obj if str(x).strip()]
            return qs[:5]
    except Exception:
        pass

    if t.startswith("[") and t.endswith("]"):
        items = re.findall(r'"([^"]+)"', t)
        qs = [i.strip() for i in items if i.strip()]
        if qs:
            return qs[:5]

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[:5]


# -----------------------------
# 5) Retrieval helpers
# -----------------------------
def results_to_ref_json(docs: List[RetrievalResult]) -> List[Dict[str, Any]]:
    ref: List[Dict[str, Any]] = []
    max_chars = 200  # ultra-compact to avoid context overflow
    for i, d in enumerate(docs, start=1):
        content = d.metadata['search_text']
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        # Keep metadata minimal to reduce prompt size
        metadata: Dict[str, Any] = {}
        if truncated:
            metadata["truncated"] = True

        ref.append(
            {
                "rank": i,
                "doc_id": d.doc_id,
                "content": content,
                "metadata": metadata,
                "score": getattr(d, "score", None),
            }
        )
    return ref


# -----------------------------
# 6) Graph node helpers
# -----------------------------
def route_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    user = _format_prompt(spec.router.user, {"sys.query": state["query"]})
    route = _parse_route(_invoke_llm(llm, spec.router.system, user))
    return {"route": route}


def mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    q = state["query"]

    setup_mq = ts_mq = general_mq = ""

    if route == "setup":
        user = _format_prompt(spec.setup_mq.user, {"sys.query": q})
        setup_mq = _invoke_llm(llm, spec.setup_mq.system, user)
    elif route == "ts":
        user = _format_prompt(spec.ts_mq.user, {"sys.query": q})
        ts_mq = _invoke_llm(llm, spec.ts_mq.system, user)
    else:
        user = _format_prompt(spec.general_mq.user, {"sys.query": q})
        general_mq = _invoke_llm(llm, spec.general_mq.system, user)

    return {"setup_mq": setup_mq, "ts_mq": ts_mq, "general_mq": general_mq}


def st_gate_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    mapping = {
        "sys.query": state["query"],
        "setup_mq": state.get("setup_mq", ""),
        "ts_mq": state.get("ts_mq", ""),
        "general_mq": state.get("general_mq", ""),
    }
    user = _format_prompt(spec.st_gate.user, mapping)
    gate = _parse_gate(_invoke_llm(llm, spec.st_gate.system, user))
    return {"st_gate": gate}


def st_mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    mapping = {
        "sys.query": state["query"],
        "setup_mq": state.get("setup_mq", ""),
        "ts_mq": state.get("ts_mq", ""),
        "general_mq": state.get("general_mq", ""),
        "st_gate": state.get("st_gate", "no_st"),
    }
    user = _format_prompt(spec.st_mq.user, mapping)
    raw = _invoke_llm(llm, spec.st_mq.system, user)
    queries = _parse_queries(raw)

    q0 = state["query"].strip()
    merged = [q0] + [q for q in queries if q and q != q0]
    return {"search_queries": merged[:5]}


def retrieve_node(state: AgentState, *, retriever: Retriever, top_k: int = 8) -> Dict[str, Any]:
    queries = state.get("search_queries", [state["query"]])
    all_docs: List[RetrievalResult] = []
    seen = set()

    for q in queries:
        for d in retriever.retrieve(q, top_k=top_k):
            key = (d.doc_id, hash(d.raw_text or d.content))
            if key in seen:
                continue
            seen.add(key)
            all_docs.append(d)

    docs = all_docs[:top_k]
    logger.info("retrieve_node: keeping %d docs (top_k=%d)", len(docs), top_k)
    return {"docs": docs, "ref_json": results_to_ref_json(docs)}


def answer_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    ref_json = json.dumps(state.get("ref_json", []), ensure_ascii=False)
    logger.info(
        "answer_node: route=%s, ref_json_chars=%d, docs=%d",
        route,
        len(ref_json),
        len(state.get("ref_json", [])),
    )
    mapping = {"sys.query": state["query"], "ref_json": ref_json}

    if route == "setup":
        tmpl = spec.setup_ans
    elif route == "ts":
        tmpl = spec.ts_ans
    else:
        tmpl = spec.general_ans

    user = _format_prompt(tmpl.user, mapping)
    answer = _invoke_llm(llm, tmpl.system, user)
    return {"answer": answer}


def judge_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    ref_json = json.dumps(state.get("ref_json", []), ensure_ascii=False)

    if route == "setup":
        sys = spec.judge_setup_sys
    elif route == "ts":
        sys = spec.judge_ts_sys
    else:
        sys = spec.judge_general_sys

    user = (
        f"질문: {state['query']}\n"
        f"답안: {state.get('answer', '')}\n"
        f"증거(JSON): {ref_json}\n"
        "JSON 한 줄로 반환: {\"faithful\": bool, \"issues\": [...], \"hint\": \"...\"}"
    )

    raw = _invoke_llm(llm, sys, user)
    try:
        judge = json.loads(raw)
        if not isinstance(judge, dict):
            raise ValueError("judge not dict")
    except Exception:
        judge = {"faithful": False, "issues": ["parse_error"], "hint": "judge JSON parse failed"}
    return {"judge": judge}


def should_retry(state: AgentState) -> Literal["done", "retry", "human"]:
    judge = state.get("judge", {})
    faithful = bool(judge.get("faithful", False))
    if faithful:
        return "done"
    if state.get("attempts", 0) < state.get("max_attempts", 0):
        return "retry"
    return "human"


def retry_bump_node(state: AgentState) -> Dict[str, Any]:
    return {"attempts": int(state.get("attempts", 0)) + 1}


def refine_queries_node(state: AgentState, *, llm: BaseLLM) -> Dict[str, Any]:
    hint = state.get("judge", {}).get("hint", "")
    prev = state.get("search_queries", [])
    sys = (
        "역할: 검색 질의 리파이너\n"
        "입력(원 질문, 기존 질의, judge hint)을 보고 더 좋은 검색 질의 3~5개를 JSON으로 만든다.\n"
        "출력: {\"queries\":[...]} 한 줄만."
    )
    user = (
        f"원 질문: {state['query']}\n"
        f"기존 질의: {json.dumps(prev, ensure_ascii=False)}\n"
        f"judge hint: {hint}\n"
    )
    raw = _invoke_llm(llm, sys, user)
    queries = _parse_queries(raw)
    merged = [state["query"]] + [q for q in queries if q and q != state["query"]]
    return {"search_queries": merged[:5]}


def human_review_node(state: AgentState) -> Command[Literal["done", "retry"]]:
    payload = {
        "question": state["query"],
        "route": state.get("route"),
        "judge": state.get("judge", {}),
        "answer": state.get("answer", ""),
        "retrieved_refs_preview": state.get("ref_json", [])[:3],
        "instruction": (
            "승인(true) / 거절(false) / 문자열로 답변을 덮어쓰기."
        ),
    }
    decision = interrupt(payload)

    if isinstance(decision, str):
        return Command(goto="done", update={"answer": decision, "human_action": {"edited": True}})

    if bool(decision):
        return Command(goto="done", update={"human_action": {"approved": True}})
    else:
        # retry_bump는 서비스 그래프에서 재시도 경로의 첫 노드 이름이다.
        return Command(goto="retry_bump", update={"human_action": {"approved": False}})


__all__ = [
    "AgentState",
    "Route",
    "Gate",
    "PromptSpec",
    "SearchServiceRetriever",
    "Retriever",
    "load_prompt_spec",
    # Node helpers
    "route_node",
    "mq_node",
    "st_gate_node",
    "st_mq_node",
    "retrieve_node",
    "answer_node",
    "judge_node",
    "should_retry",
    "retry_bump_node",
    "refine_queries_node",
    "human_review_node",
]
