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

    # Multi-query outputs (parsed lists)
    setup_mq_list: List[str]
    ts_mq_list: List[str]
    general_mq_list: List[str]

    st_gate: Gate
    search_queries: List[str]

    # Device selection (HIL)
    available_devices: List[Dict[str, Any]]
    selected_devices: List[str]  # Multiple devices can be selected
    device_selection_skipped: bool

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

    # User feedback after retrieval (for ask_user node)
    user_feedback: Optional[str]
    retrieval_confirmed: bool
    thread_id: Optional[str]


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
- 질문, 답변, REFS (검색 결과)

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

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        device_name: str | None = None,
        device_names: List[str] | None = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        k = top_k or self.top_k
        # 그래프 레벨에서 MQ/재시도를 수행하므로 내부 MQ/rerank는 끈다.
        search_kwargs: Dict[str, Any] = {
            "top_k": k,
            "multi_query": False,
            "rerank": False,
        }
        # Pass device_names for filtering (OR logic)
        if device_names:
            search_kwargs["device_names"] = device_names
        # Legacy: device_name for boosting if provided
        elif device_name:
            search_kwargs["device_name"] = device_name
        return self.search_service.search(query, **search_kwargs)


# -----------------------------
# 4) LLM helpers
# -----------------------------
# 노드 타입별 max_tokens 설정
MAX_TOKENS_CLASSIFICATION = 256   # 라우팅/분류용 (짧은 응답)
MAX_TOKENS_ANSWER = 4096         # 답변 생성용


def _invoke_llm(llm: BaseLLM, system: str, user: str, **kwargs: Any) -> str:
    messages: List[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    # max_tokens 기본값: 분류용 (answer_node에서는 명시적으로 더 큰 값 전달)
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = MAX_TOKENS_CLASSIFICATION
    logger.debug("_invoke_llm: max_tokens=%s, system_len=%d, user_len=%d", kwargs.get("max_tokens"), len(system), len(user))
    out = llm.generate(messages, **kwargs)
    result = out.text.strip()
    logger.debug("_invoke_llm: output_len=%d", len(result))
    return result


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

    if '"queries"' in t:
        items = re.findall(r'"([^"]+)"', t)
        qs = [i.strip() for i in items if i.strip() and i.strip().lower() != "queries"]
        if qs:
            return qs[:5]

    # Filter out meta-explanation lines (English prompt patterns)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    filtered = []
    for line in lines:
        lower = line.lower()
        # Skip lines that look like prompt/explanation patterns
        if any(pattern in lower for pattern in [
            "given original", "they want", "from mq:", "we need", "could be:",
            "example output", "example input", "output only", "no explanations",
            "# ", "q1:", "q2:", "q3:", "query generation", "output format"
        ]):
            continue
        # Skip numbered lines like "1.", "2.", "3."
        if re.match(r'^\d+[\.\)]\s', line):
            continue
        filtered.append(line)

    return filtered[:5]


# -----------------------------
# 5) Retrieval helpers
# -----------------------------
def results_to_ref_json(docs: List[RetrievalResult]) -> List[Dict[str, Any]]:
    ref: List[Dict[str, Any]] = []
    max_chars = 200  # ultra-compact to avoid context overflow
    for i, d in enumerate(docs, start=1):
        content = ""
        if isinstance(d.metadata, dict):
            content = str(d.metadata.get("search_text") or "").strip()
        if not content:
            content = str(d.raw_text or d.content or "").strip()
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


def ref_json_to_text(ref_json: List[Dict[str, Any]]) -> str:
    """Format retrieval evidence as plain text for LLM prompts.

    vLLM + 일부 모델에서 JSON(중괄호/대괄호/따옴표)이 포함된 프롬프트가
    비정상 토큰('!') 반복/빈 응답으로 붕괴하는 사례가 있어 텍스트로 전달한다.
    """
    if not ref_json:
        return "EMPTY"

    lines: List[str] = []
    for r in ref_json:
        rank = r.get("rank", "?")
        doc_id = str(r.get("doc_id", "")).strip()
        content = str(r.get("content", "")).strip()
        content = " ".join(content.split())
        lines.append(f"[{rank}] {doc_id}: {content}")
    return "\n".join(lines)


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

    setup_mq_list: List[str] = []
    ts_mq_list: List[str] = []
    general_mq_list: List[str] = []

    # MQ generation needs more tokens than classification (3 queries ~300 tokens)
    # Reasoning models need extra tokens for thinking process (~1000) + output (~200)
    mq_kwargs = {"max_tokens": 4096}

    if route == "setup":
        user = _format_prompt(spec.setup_mq.user, {"sys.query": q})
        raw = _invoke_llm(llm, spec.setup_mq.system, user, **mq_kwargs)
        setup_mq_list = _parse_queries(raw)
        logger.info("mq_node(setup): generated %d queries: %s", len(setup_mq_list), setup_mq_list)
    elif route == "ts":
        user = _format_prompt(spec.ts_mq.user, {"sys.query": q})
        raw = _invoke_llm(llm, spec.ts_mq.system, user, **mq_kwargs)
        ts_mq_list = _parse_queries(raw)
        logger.info("mq_node(ts): generated %d queries: %s", len(ts_mq_list), ts_mq_list)
    else:
        user = _format_prompt(spec.general_mq.user, {"sys.query": q})
        raw = _invoke_llm(llm, spec.general_mq.system, user, **mq_kwargs)
        general_mq_list = _parse_queries(raw)
        logger.info("mq_node(general): generated %d queries: %s", len(general_mq_list), general_mq_list)

    return {"setup_mq_list": setup_mq_list, "ts_mq_list": ts_mq_list, "general_mq_list": general_mq_list}


def st_gate_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    # Convert lists back to text for the prompt
    setup_mq_list = state.get("setup_mq_list", [])
    ts_mq_list = state.get("ts_mq_list", [])
    general_mq_list = state.get("general_mq_list", [])

    mapping = {
        "sys.query": state["query"],
        "setup_mq": "\n".join(setup_mq_list),
        "ts_mq": "\n".join(ts_mq_list),
        "general_mq": "\n".join(general_mq_list),
    }
    user = _format_prompt(spec.st_gate.user, mapping)
    gate = _parse_gate(_invoke_llm(llm, spec.st_gate.system, user))
    return {"st_gate": gate}


def st_mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    # Get MQ lists from previous node
    setup_mq_list = state.get("setup_mq_list", [])
    ts_mq_list = state.get("ts_mq_list", [])
    general_mq_list = state.get("general_mq_list", [])

    mapping = {
        "sys.query": state["query"],
        "setup_mq": "\n".join(setup_mq_list),
        "ts_mq": "\n".join(ts_mq_list),
        "general_mq": "\n".join(general_mq_list),
        "st_gate": state.get("st_gate", "no_st"),
    }
    user = _format_prompt(spec.st_mq.user, mapping)
    raw = _invoke_llm(llm, spec.st_mq.system, user)
    queries = _parse_queries(raw)

    q0 = state["query"].strip()
    merged = [q0] + [q for q in queries if q and q != q0]
    logger.info("st_mq_node: final search_queries=%s", merged[:5])
    return {"search_queries": merged[:5]}


def retrieve_node(
    state: AgentState,
    *,
    retriever: Retriever,
    reranker: Any = None,
    retrieval_top_k: int = 30,
    final_top_k: int = 10,
) -> Dict[str, Any]:
    """Retrieve documents with dual search strategy and rerank.

    If devices are selected:
      - Search 1: 30 docs filtered by selected devices (OR filter)
      - Search 2: 30 docs without filter (general search)
      - Combine and rerank to get final 10 docs

    If no devices selected:
      - Search 60 docs without filter
      - Rerank to get final 10 docs
    """
    queries = state.get("search_queries", [state["query"]])
    selected_devices = state.get("selected_devices", [])
    original_query = state["query"]

    all_docs: List[RetrievalResult] = []
    seen: set = set()

    def _add_docs(docs_to_add: List[RetrievalResult]) -> None:
        """Add docs to all_docs, avoiding duplicates."""
        for d in docs_to_add:
            key = (d.doc_id, hash(d.raw_text or d.content))
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    if selected_devices:
        # Dual search strategy: device-filtered + general
        logger.info("retrieve_node: dual search with devices=%s", selected_devices)

        # Search 1: Device-filtered search (30 docs) with device_names as OR filter
        for q in queries:
            device_docs = retriever.retrieve(q, top_k=retrieval_top_k, device_names=selected_devices)
            _add_docs(device_docs)

        device_filtered_count = len(all_docs)
        logger.info("retrieve_node: device-filtered search found %d docs", device_filtered_count)

        # Search 2: General search without device filter (30 docs)
        for q in queries:
            general_docs = retriever.retrieve(q, top_k=retrieval_top_k)
            _add_docs(general_docs)

        logger.info("retrieve_node: after general search, total %d docs", len(all_docs))

    else:
        # No device selection: search 60 docs without filter
        logger.info("retrieve_node: general search (no device filter)")
        for q in queries:
            docs = retriever.retrieve(q, top_k=retrieval_top_k * 2)
            _add_docs(docs)

    logger.info("retrieve_node: collected %d unique docs before rerank", len(all_docs))

    # Rerank if reranker is available
    if reranker is not None and all_docs:
        logger.info("retrieve_node: reranking %d docs to top %d", len(all_docs), final_top_k)
        docs = reranker.rerank(original_query, all_docs, top_k=final_top_k)
    else:
        # No reranker: just take top final_top_k by score
        docs = sorted(all_docs, key=lambda d: d.score, reverse=True)[:final_top_k]

    logger.info("retrieve_node: returning %d docs", len(docs))
    return {"docs": docs, "ref_json": results_to_ref_json(docs)}


def ask_user_after_retrieve_node(state: AgentState) -> Command[Literal["answer", "refine_and_retrieve"]]:
    """Retrieval 후 사용자에게 검색 결과를 보여주고 피드백을 받는 노드.

    사용자 응답:
    - True 또는 빈 문자열: 검색 결과 승인 → answer로 진행
    - 문자열(키워드/피드백): 재검색 쿼리에 반영 → refine_and_retrieve로 이동
    - False: 검색 결과 부적절 → refine_and_retrieve로 이동 (기존 쿼리 유지)
    """
    ref_json = state.get("ref_json", [])
    search_queries = state.get("search_queries", [])

    payload = {
        "type": "retrieval_review",
        "question": state["query"],
        "route": state.get("route"),
        "search_queries": search_queries,
        "retrieved_docs": ref_json,
        "doc_count": len(ref_json),
        "instruction": (
            "검색 결과를 확인하세요.\n"
            "- 승인(true 또는 빈 문자열): 답변 생성으로 진행\n"
            "- 추가 키워드/피드백 입력: 해당 내용으로 재검색\n"
            "- 거절(false): 재검색 시도"
        ),
    }

    decision = interrupt(payload)

    # 사용자가 검색어를 수정한 경우: 수정된 검색어로 재검색
    if isinstance(decision, dict):
        decision_type = decision.get("type")

        # Handle search query modification
        if decision_type == "modify_search_queries":
            modified_queries = decision.get("search_queries", [])

            if isinstance(modified_queries, list) and len(modified_queries) > 0:
                valid_queries = [
                    str(q).strip()
                    for q in modified_queries
                    if str(q).strip()
                ]

                if valid_queries:
                    return Command(
                        goto="refine_and_retrieve",
                        update={
                            "retrieval_confirmed": False,
                            "user_feedback": f"Modified queries: {', '.join(valid_queries)}",
                            "search_queries": valid_queries[:5],
                        }
                    )
                else:
                    # Empty queries - fall back to refine_and_retrieve
                    return Command(
                        goto="refine_and_retrieve",
                        update={
                            "retrieval_confirmed": False,
                            "user_feedback": "Empty queries provided",
                        }
                    )

        # 사용자가 특정 문서를 선택한 경우: 선택 문서만으로 answer 진행
        selected_ids = decision.get("selected_doc_ids") or decision.get("selected_docs")
        selected_ranks = decision.get("selected_ranks")

        if isinstance(selected_ids, str):
            selected_ids = [selected_ids]
        if isinstance(selected_ranks, (int, str)):
            selected_ranks = [selected_ranks]

        selected_id_list: list[str] = []
        if isinstance(selected_ids, list):
            selected_id_list = [str(x).strip() for x in selected_ids if str(x).strip()]

        selected_rank_list: list[int] = []
        if isinstance(selected_ranks, list):
            for r in selected_ranks:
                try:
                    selected_rank_list.append(int(r))
                except Exception:
                    continue

        if selected_id_list or selected_rank_list:
            docs = state.get("docs", [])
            selected_docs = []

            if selected_id_list:
                selected_docs.extend([d for d in docs if str(d.doc_id) in selected_id_list])

            if selected_rank_list:
                rank_set = set(selected_rank_list)
                selected_docs.extend([
                    d for idx, d in enumerate(docs, start=1) if idx in rank_set
                ])

            # Deduplicate while preserving order
            seen_keys = set()
            deduped_docs = []
            for d in selected_docs:
                key = (d.doc_id, id(d))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped_docs.append(d)

            if deduped_docs:
                return Command(
                    goto="answer",
                    update={
                        "docs": deduped_docs,
                        "ref_json": results_to_ref_json(deduped_docs),
                        "selected_doc_ids": selected_id_list,
                        "selected_ranks": selected_rank_list,
                        "retrieval_confirmed": True,
                        "user_feedback": None,
                    },
                )

    # 승인: True 또는 빈 문자열
    if decision is True or decision == "":
        return Command(
            goto="answer",
            update={"retrieval_confirmed": True, "user_feedback": None}
        )

    # 피드백 제공: 문자열로 키워드/피드백 입력
    if isinstance(decision, str) and decision.strip():
        feedback = decision.strip()
        # 기존 쿼리에 피드백 키워드 추가
        new_queries = [state["query"], feedback] + [
            q for q in search_queries if q != state["query"]
        ]
        return Command(
            goto="refine_and_retrieve",
            update={
                "retrieval_confirmed": False,
                "user_feedback": feedback,
                "search_queries": new_queries[:5],
            }
        )

    # 거절: False
    return Command(
        goto="refine_and_retrieve",
        update={"retrieval_confirmed": False, "user_feedback": None}
    )


def answer_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    ref_items = state.get("ref_json", [])
    ref_text = ref_json_to_text(ref_items)
    logger.info(
        "answer_node: route=%s, refs_chars=%d, docs=%d",
        route,
        len(ref_text),
        len(ref_items),
    )
    mapping = {"sys.query": state["query"], "ref_text": ref_text}

    if route == "setup":
        tmpl = spec.setup_ans
    elif route == "ts":
        tmpl = spec.ts_ans
    else:
        tmpl = spec.general_ans

    user = _format_prompt(tmpl.user, mapping)
    logger.info("answer_node: user_prompt_chars=%d, system_prompt_chars=%d", len(user), len(tmpl.system))
    answer = _invoke_llm(llm, tmpl.system, user, max_tokens=MAX_TOKENS_ANSWER)
    logger.info("answer_node: answer_chars=%d, answer_preview=%s", len(answer), answer[:500] if answer else "(empty)")
    return {"answer": answer}


def judge_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    ref_text = ref_json_to_text(state.get("ref_json", []))

    if route == "setup":
        sys = spec.judge_setup_sys
    elif route == "ts":
        sys = spec.judge_ts_sys
    else:
        sys = spec.judge_general_sys

    user = (
        f"질문: {state['query']}\n"
        f"답안: {state.get('answer', '')}\n"
        f"증거(REFS): {ref_text}\n"
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
    if state.get("retrieval_confirmed"):
        return "done"
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
        "type": "human_review",
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


def device_selection_node(
    state: AgentState,
    *,
    device_fetcher: Optional[Any] = None,
) -> Command[Literal["mq"]]:
    """Device selection node - interrupts to let user select devices (multiple).

    Args:
        state: Agent state.
        device_fetcher: Callable that returns list of devices.
            Each device should have 'name' and 'doc_count'.

    Returns:
        Command to proceed to 'mq' node with selected devices.
    """
    # Fetch available devices if fetcher provided
    available_devices: List[Dict[str, Any]] = []
    if device_fetcher is not None:
        try:
            available_devices = device_fetcher()
        except Exception as e:
            logger.warning(f"Failed to fetch devices: {e}")
            available_devices = []

    if not available_devices:
        # No devices available, skip selection
        logger.info("device_selection_node: no devices available, skipping")
        return Command(
            goto="mq",
            update={
                "available_devices": [],
                "selected_devices": [],
                "device_selection_skipped": True,
            }
        )

    payload = {
        "type": "device_selection",
        "question": state["query"],
        "route": state.get("route"),
        "devices": available_devices,
        "device_count": len(available_devices),
        "instruction": (
            "검색할 기기를 선택하세요 (다중 선택 가능).\n"
            "- 기기 선택: 선택 기기 문서 10개 + 전체 문서 10개 검색\n"
            "- 건너뛰기: 전체 문서 20개 검색"
        ),
    }

    decision = interrupt(payload)
    logger.info(f"device_selection_node: user decision={decision}")

    # User skipped device selection
    if decision is None or decision == "" or decision == "skip":
        return Command(
            goto="mq",
            update={
                "available_devices": available_devices,
                "selected_devices": [],
                "device_selection_skipped": True,
            }
        )

    # Parse selected devices (can be single string, list, or dict with selected_devices)
    selected_devices: List[str] = []

    if isinstance(decision, dict):
        # New format: {"type": "device_selection", "selected_devices": [...]}
        devices_from_dict = decision.get("selected_devices", [])
        if isinstance(devices_from_dict, list):
            selected_devices = [str(d).strip() for d in devices_from_dict if d]
        elif isinstance(devices_from_dict, str):
            selected_devices = [devices_from_dict.strip()]
        # Also check legacy format
        legacy_device = decision.get("device") or decision.get("selected_device")
        if legacy_device and isinstance(legacy_device, str):
            if legacy_device.strip() not in selected_devices:
                selected_devices.append(legacy_device.strip())
    elif isinstance(decision, list):
        selected_devices = [str(d).strip() for d in decision if d]
    elif isinstance(decision, str):
        selected_devices = [decision.strip()]

    # Validate selections against available devices
    valid_names = {d.get("name") for d in available_devices if d.get("name")}
    selected_devices = [d for d in selected_devices if d in valid_names]

    if not selected_devices:
        logger.warning("No valid device selections after validation")

    logger.info(f"device_selection_node: validated selected_devices={selected_devices}")

    return Command(
        goto="mq",
        update={
            "available_devices": available_devices,
            "selected_devices": selected_devices,
            "device_selection_skipped": len(selected_devices) == 0,
        }
    )


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
    "ask_user_after_retrieve_node",
    "answer_node",
    "judge_node",
    "should_retry",
    "retry_bump_node",
    "refine_queries_node",
    "human_review_node",
    "device_selection_node",
]
