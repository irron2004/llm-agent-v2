"""LangGraph node helpers + prompt spec.

이 파일은 노드/프롬프트 스펙/헬퍼를 제공하고, 실제 그래프 조립은
service 계층에서 담당한다.
"""

from __future__ import annotations

import json
import re
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict

from langgraph.types import Command, interrupt

from backend.domain.doc_type_mapping import DOC_TYPE_GROUPS, expand_doc_type_selection, normalize_doc_type
from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate, load_prompt_template
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.search_service import SearchService

logger = logging.getLogger(__name__)
# RAG 파이프라인 전용 트레이스 로거 (별도 파일/핸들러 설정 가능)
trace_logger = logging.getLogger("rag_trace")


# -----------------------------
# RAG Pipeline Trace Logging
# -----------------------------
def _trace_log(node_name: str, state: "AgentState", event: str, data: Optional[Dict[str, Any]] = None) -> None:
    """RAG 파이프라인 트레이스 로그.

    Args:
        node_name: 노드 이름 (예: "translate_node", "mq_node")
        state: 현재 AgentState
        event: 이벤트 타입 (예: "ENTER", "EXIT", "LLM_CALL", "RESULT")
        data: 추가 데이터
    """
    trace_id = state.get("thread_id") or "no-thread"
    query = state.get("query", "")[:80]
    timestamp = datetime.now().isoformat()

    log_data = {
        "ts": timestamp,
        "trace_id": trace_id,
        "node": node_name,
        "event": event,
        "query": query,
    }
    if data:
        log_data.update(data)

    # JSON 형식으로 로그 (파싱 용이)
    trace_logger.info(json.dumps(log_data, ensure_ascii=False, default=str))


# -----------------------------
# 1) State schema
# -----------------------------
Route = Literal["general", "doc_lookup", "history_answer", "retrieval"]
Gate = Literal["need_st", "no_st"]


class ChatHistoryEntry(TypedDict, total=False):
    """대화 히스토리 항목.

    - user: content 필드 사용 (원본 질문)
    - assistant: summary 필드 사용 (답변 요약 truncate 150자)
    """

    role: str  # "user" | "assistant"
    # user용 필드
    content: str  # user만 사용: 원본 질문 (그대로 유지)
    # assistant용 필드
    summary: str  # assistant만 사용: 답변 요약 (truncate 150자)
    refs: List[str]  # 참조 문서 title (rank 순)
    doc_ids: List[str]  # 참조 문서 ID (rank 순 - 첫 번째가 가장 중요)


class AgentState(TypedDict, total=False):
    query: str
    route: Route
    is_chat_query: bool
    chat_type: str

    # Chat history for multi-turn conversation
    chat_history: List[ChatHistoryEntry]  # 대화 히스토리 (클라이언트 전달)
    lookup_doc_ids: List[str]  # doc_lookup용 doc_id 목록
    lookup_source: str  # "query" | "history" - doc_id 추출 출처

    # History intent detection (history_intent_node 결과)
    history_intent: bool  # 이전 대화 참조 의도 여부
    history_confidence: float  # 신뢰도 (0~1)
    history_source: str  # "rule" | "keyword_overlap" | "none"

    # Multi-query outputs (parsed lists)
    # Unified retrieval MQ (new structure)
    retrieval_mq_list: List[str]  # English queries
    retrieval_mq_ko_list: List[str]  # Korean queries
    # Legacy MQ lists (kept for compatibility during transition)
    setup_mq_list: List[str]
    ts_mq_list: List[str]
    general_mq_list: List[str]
    setup_mq_ko_list: List[str]
    ts_mq_ko_list: List[str]
    general_mq_ko_list: List[str]

    st_gate: Gate
    search_queries: List[str]
    skip_mq: bool

    # Retry strategy configuration
    expand_top_k: int  # Number of docs to expand (default: 5, retry: 10)
    retry_strategy: str  # "expand_more" | "refine_queries" | "regenerate_mq"

    # Auto-parsed filters (from LLM)
    auto_parsed_device: Optional[str]  # First parsed device from query
    auto_parsed_doc_type: Optional[str]  # First parsed doc type from query
    auto_parsed_devices: List[str]
    auto_parsed_doc_types: List[str]
    auto_parse_message: Optional[str]  # Message to display (e.g., "SUPRA N장비로 검색합니다")

    # Language detection and translation
    detected_language: Optional[str]  # "ko", "en", "ja" - detected from query
    query_en: Optional[str]  # English version of query (for internal processing)
    query_ko: Optional[str]  # Korean version of query (for retrieval)

    # Device selection (HIL)
    available_devices: List[Dict[str, Any]]
    selected_devices: List[str]  # Multiple devices can be selected
    device_selection_skipped: bool
    available_doc_types: List[Dict[str, Any]]
    selected_doc_types: List[str]
    doc_type_selection_skipped: bool
    selected_doc_ids: List[str]

    # Retrieval outputs
    docs: List[RetrievalResult]
    all_docs: List[RetrievalResult]  # 재생성용 전체 문서 (rerank 전, 최대 20개)
    display_docs: List[RetrievalResult]
    ref_json: List[Dict[str, Any]]
    answer_ref_json: List[Dict[str, Any]]

    # Answer + judge
    answer: str
    reasoning: Optional[str]  # Reasoning process from reasoning models
    judge: Dict[str, Any]

    # Retry / HIL
    attempts: int
    max_attempts: int
    human_action: Optional[Dict[str, Any]]

    # User feedback after retrieval (for ask_user node)
    user_feedback: Optional[str]
    retrieval_confirmed: bool
    thread_id: Optional[str]

    # Internal flags
    _skip_human_review: bool  # Auto-parse 모드에서 human_review 건너뛰기

    # Search parameter override (for Retrieval Test)
    search_override: Optional[Dict[str, Any]]  # dense_weight, sparse_weight, rerank, etc.


# -----------------------------
# 2) Prompt loading
# -----------------------------
@dataclass
class PromptSpec:
    router: PromptTemplate
    # Unified retrieval MQ (replaces setup_mq, ts_mq, general_mq)
    retrieval_mq: PromptTemplate
    st_gate: PromptTemplate
    st_mq: PromptTemplate
    # Unified retrieval answer (replaces setup_ans, ts_ans, general_ans)
    retrieval_ans: PromptTemplate
    judge_setup_sys: str
    judge_ts_sys: str
    judge_general_sys: str
    auto_parse: Optional[PromptTemplate] = None  # Auto-parse device/doc_type
    auto_parse_device: Optional[PromptTemplate] = None  # Auto-parse device only (LLM fallback)
    translate: Optional[PromptTemplate] = None  # Translate query to en/ko
    # Unified retrieval answer - language-specific
    retrieval_ans_en: Optional[PromptTemplate] = None
    retrieval_ans_ja: Optional[PromptTemplate] = None


DEFAULT_JUDGE_SETUP = """
# 역할
설치/세팅 답변이 질문과 검색 증거에 충실한지 판정한다.

# 입력
- 질문, 답변, REFS (검색 결과)

# 판정 기준 (관대하게 적용)
- faithful=true: 답변의 **핵심 절차**가 REFS에서 유추 가능하면 충분
- faithful=true: REFS 내용을 요약/재구성/순서 변경해도 OK
- faithful=true: 일반 상식 수준의 연결/추론은 허용
- faithful=false: REFS에 전혀 없는 수치/토크값/압력을 단정적으로 언급한 경우만

# 출력
JSON 한 줄: {"faithful": bool, "issues": ["..."], "hint": "..."}
""".strip()

DEFAULT_JUDGE_TS = """
# 역할
TS(트러블슈팅) 답변이 질문과 검색 증거에 근거했는지 판정한다.

# 판정 기준 (관대하게 적용)
- faithful=true: 답변의 **원인/조치**가 REFS에서 유추 가능하면 충분
- faithful=true: REFS 내용을 요약/재구성해도 OK
- faithful=true: 일반 상식 수준의 진단 연결은 허용
- faithful=false: REFS와 **명백히 모순**되는 조치를 제시한 경우만

출력: {"faithful": bool, "issues": [..], "hint": "..."}
""".strip()

DEFAULT_JUDGE_GENERAL = """
# 역할
일반 답변이 질문 의도와 검색 증거를 충실히 반영했는지 판정한다.

# 예외 (항상 faithful=true)
다음 유형의 질문은 REFS 없이 답변해도 faithful로 판정한다:
- 에이전트 정체성 질문 ("너는 누구니?", "뭘 할 수 있어?", "Who are you?")
- 인사/일상 대화 ("안녕", "고마워", "잘가")
- 에이전트 사용법 질문

# 일반 규칙
위 예외에 해당하지 않는 경우, 답변이 REFS 증거에 **대체로** 기반했는지 판정한다.

# 판정 기준 (관대하게 적용)
- faithful=true: 답변의 **핵심 내용**이 REFS에서 유추 가능하면 충분
- faithful=true: REFS 내용을 요약/재구성/번역해도 OK
- faithful=true: 일반 상식 수준의 연결/추론은 허용
- faithful=false: 답변이 REFS와 **명백히 모순**되거나, REFS에 전혀 없는 수치/명칭을 단정적으로 언급한 경우만

출력: {"faithful": bool, "issues": [..], "hint": "..."}
""".strip()


def _try_load_prompt(name: str, version: str) -> Optional[PromptTemplate]:
    """Try to load a prompt template, return None if not found."""
    try:
        return load_prompt_template(name, version)
    except FileNotFoundError:
        return None


def load_prompt_spec(version: str = "v1") -> PromptSpec:
    """Load required prompts from YAML (router/MQ/gate/answer)."""

    router = load_prompt_template("router", version)
    # Unified retrieval MQ
    retrieval_mq = load_prompt_template("retrieval_mq", version)
    st_gate = load_prompt_template("st_gate", version)
    st_mq = load_prompt_template("st_mq", version)
    # Unified retrieval answer
    retrieval_ans = load_prompt_template("retrieval_ans", version)

    # Try to load optional prompts
    auto_parse = _try_load_prompt("auto_parse", version)
    auto_parse_device = _try_load_prompt("auto_parse_device", version)
    translate = _try_load_prompt("translate", version)

    # Unified retrieval answer - language-specific
    retrieval_ans_en = _try_load_prompt("retrieval_ans_en", version)
    retrieval_ans_ja = _try_load_prompt("retrieval_ans_ja", version)

    return PromptSpec(
        router=router,
        retrieval_mq=retrieval_mq,
        st_gate=st_gate,
        st_mq=st_mq,
        retrieval_ans=retrieval_ans,
        judge_setup_sys=DEFAULT_JUDGE_SETUP,
        judge_ts_sys=DEFAULT_JUDGE_TS,
        judge_general_sys=DEFAULT_JUDGE_GENERAL,
        auto_parse=auto_parse,
        auto_parse_device=auto_parse_device,
        translate=translate,
        retrieval_ans_en=retrieval_ans_en,
        retrieval_ans_ja=retrieval_ans_ja,
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
        doc_types: List[str] | None = None,
        search_override: Dict[str, Any] | None = None,
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
        if doc_types:
            search_kwargs["doc_types"] = doc_types

        # Apply search parameter overrides (for Retrieval Test)
        if search_override:
            # Override supported parameters: dense_weight, sparse_weight, use_rrf, rrf_k, rerank, rerank_top_k
            for key in ["dense_weight", "sparse_weight", "use_rrf", "rrf_k", "rerank", "rerank_top_k"]:
                if key in search_override:
                    search_kwargs[key] = search_override[key]
            # top_k override
            if "top_k" in search_override:
                search_kwargs["top_k"] = search_override["top_k"]

        return self.search_service.search(query, **search_kwargs)


# -----------------------------
# 4) LLM helpers
# -----------------------------
# 노드 타입별 max_tokens 설정
MAX_TOKENS_CLASSIFICATION = 256   # 라우팅/분류용 (짧은 응답)
MAX_TOKENS_JUDGE = 1024          # Judge용 (reasoning + JSON)
MAX_TOKENS_ANSWER = 4096         # 답변 생성용
MAX_REF_CHARS_REVIEW = 200       # 검색 결과 리뷰용
MAX_REF_CHARS_ANSWER = 1200      # 답변 생성용
RELATED_PAGE_WINDOW = 2          # 인접 페이지 범위 (±N)
DOC_TYPES_SAME_DOC = {"gcb", "myservice"}
EXPAND_TOP_K = 20                # 확장 대상 최대 개수 (rerank 상위)


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


def _invoke_llm_with_reasoning(llm: BaseLLM, system: str, user: str, **kwargs: Any) -> tuple[str, str | None]:
    """Invoke LLM and return both text and reasoning content."""
    messages: List[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = MAX_TOKENS_CLASSIFICATION
    out = llm.generate(messages, **kwargs)
    text = out.text.strip()
    reasoning = out.reasoning
    logger.debug("_invoke_llm_with_reasoning: output_len=%d, reasoning_len=%d", len(text), len(reasoning) if reasoning else 0)
    return text, reasoning


def _format_prompt(template: str, mapping: Dict[str, str]) -> str:
    """Lightweight placeholder replacement without raising on missing keys."""
    rendered = template
    for key, val in mapping.items():
        rendered = rendered.replace(f"{{{key}}}", val)
    return rendered


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text that may contain reasoning/explanations.

    Handles cases where LLM outputs reasoning text before/after JSON.
    Tries multiple strategies:
    1. Direct parse (if text is pure JSON)
    2. Find JSON object pattern {...}
    3. Find JSON in code blocks ```json ... ```
    """
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON object pattern
    # Look for outermost { ... } that forms valid JSON
    import re
    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    matches = json_pattern.findall(text)
    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON in code blocks
    code_block_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
    code_matches = code_block_pattern.findall(text)
    for match in code_matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


def _parse_route(text: str) -> Route:
    """Parse route from LLM response: general | doc_lookup | history_answer | retrieval."""
    t = text.strip().lower()
    valid_routes = ("general", "doc_lookup", "history_answer", "retrieval")
    if t in valid_routes:
        return t  # type: ignore[return-value]
    m = re.search(r"\b(general|doc_lookup|history_answer|retrieval)\b", t)
    if m:
        return m.group(1)  # type: ignore[return-value]
    # fallback to retrieval (safer - will search)
    return "retrieval"


def _parse_gate(text: str) -> Gate:
    t = text.strip().lower()
    if t in ("need_st", "no_st"):
        return t  # type: ignore[return-value]
    if "need" in t:
        return "need_st"
    return "no_st"


def _is_general_chat_query(query: str) -> bool:
    """Heuristic guardrail for agent-identity/small-talk queries."""
    q = (query or "").strip().lower()
    if not q:
        return False

    identity_patterns = [
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bwhat can you do\b",
        r"\bwhat is your name\b",
        r"\bwhat'?s your name\b",
        r"너는?\s*누구",
        r"넌\s*누구",
        r"당신은?\s*누구",
        r"너는?\s*뭘\s*할\s*수",
    ]
    if any(re.search(pat, q) for pat in identity_patterns):
        return True

    # Guardrail against technical/domain-like tokens (APC, E-001, 3.2Nm, etc.).
    # This prevents domain questions with greetings from being misrouted to chat.
    if re.search(r"[A-Z]{2,}|\d", query or ""):
        return False

    patterns = [
        # English - greeting
        r"\bhello\b",
        r"\bhi\b",
        r"\bhey\b",
        # English - thanks
        r"\bthanks?\b",
        r"\bthank you\b",
        # English - farewell
        r"\bbye\b",
        r"\bgoodbye\b",
        r"\bsee you\b",
        # Korean - greeting
        r"^안녕",
        r"^ㅎㅇ",
        r"^하이",
        r"^헬로",
        # Korean - thanks
        r"고마워",
        r"고맙습니다",
        r"감사합니다",
        r"감사해",
        r"땡큐",
        r"ㄱㅅ",
        # Korean - farewell
        r"잘\s*가",
        r"안녕히",
        r"바이",
        r"다음에\s*(봐|봬)",
    ]
    return any(re.search(pat, q) for pat in patterns)


def _parse_queries(text: str) -> List[str]:
    """Robust parser: JSON object/list or one-per-line strings."""
    t = text.strip()
    if not t:
        return []

    # Strip common code-fence wrappers.
    t = re.sub(r"```[a-zA-Z]*", "", t).strip()

    def _extract_queries(obj: Any) -> List[str]:
        if isinstance(obj, dict):
            for key in ("queries", "search_queries"):
                if isinstance(obj.get(key), list):
                    return [str(x).strip() for x in obj[key] if str(x).strip()]
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        return []

    # Try to parse JSON directly or from an embedded snippet.
    candidates = [t]
    obj_match = re.search(r"\{.*\}", t, flags=re.S)
    list_match = re.search(r"\[.*\]", t, flags=re.S)
    if obj_match:
        candidates.append(obj_match.group(0))
    if list_match:
        candidates.append(list_match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        qs = _extract_queries(obj)
        if qs:
            return _dedupe_queries(qs)

    if t.startswith("[") and t.endswith("]"):
        items = re.findall(r'"([^"]+)"', t)
        qs = [i.strip() for i in items if i.strip()]
        if qs:
            return _dedupe_queries(qs)

    if '"queries"' in t or '"search_queries"' in t:
        # Known garbage labels that may leak from prompts
        garbage_labels = {"queries", "search_queries", "setup_mq", "ts_mq", "general_mq", "gate"}
        items = re.findall(r'"([^"]+)"', t)
        qs = [
            i.strip()
            for i in items
            if i.strip() and i.strip().lower() not in garbage_labels
        ]
        if qs:
            return _dedupe_queries(qs)

    def _is_meta_line(line: str) -> bool:
        lower = line.lower()
        if any(pattern in lower for pattern in [
            "given original", "they want", "from mq:", "we need", "could be:",
            "example output", "example input", "output only", "no explanations",
            "query generation", "output format",
        ]):
            return True
        if lower.startswith(("output:", "example:", "format:", "input:")):
            return True
        # Filter out prompt template labels that may leak into output
        if any(label in lower for label in [
            "setup_mq:", "ts_mq:", "general_mq:", "gate:", "질문:",
        ]):
            return True
        return False

    def _strip_prefix(line: str) -> str:
        cleaned = line.strip()
        cleaned = re.sub(r"^[-*•]\s+", "", cleaned)
        cleaned = re.sub(r"^(?:q\d+|\d+)\s*[:\.\)\-]\s*", "", cleaned, flags=re.I)
        cleaned = cleaned.strip().strip("\"'`").strip()
        cleaned = re.sub(r"[;,，]\s*$", "", cleaned).strip()
        return cleaned

    # Filter out meta-explanation lines and clean numbered/bulleted outputs.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    filtered: List[str] = []
    for line in lines:
        if _is_meta_line(line):
            continue
        if line.startswith(("{", "[")):
            continue
        cleaned = _strip_prefix(line)
        if cleaned:
            filtered.append(cleaned)

    if len(filtered) == 1:
        single = filtered[0]
        for delim in (" / ", " | ", ";", "；"):
            if delim in single:
                split_items = [part.strip() for part in single.split(delim) if part.strip()]
                if len(split_items) > 1:
                    filtered = split_items
                break

    return _dedupe_queries(filtered)


def _dedupe_queries(queries: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for q in queries:
        normalized = q.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:5]


# -----------------------------
# 5) Retrieval helpers
# -----------------------------
def results_to_ref_json(
    docs: List[RetrievalResult],
    *,
    max_chars: int | None = MAX_REF_CHARS_REVIEW,
    prefer_raw_text: bool = False,
) -> List[Dict[str, Any]]:
    ref: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, start=1):
        content = ""
        if prefer_raw_text:
            content = str(d.raw_text or d.content or "").strip()
        if not content and isinstance(d.metadata, dict):
            content = str(d.metadata.get("search_text") or "").strip()
        if not content:
            content = str(d.raw_text or d.content or "").strip()
        truncated = False
        if max_chars is not None and max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        # Keep metadata minimal to reduce prompt size
        metadata: Dict[str, Any] = {}
        if truncated:
            metadata["truncated"] = True

        item = {
            "rank": i,
            "doc_id": d.doc_id,
            "content": content,
            "metadata": metadata,
        }
        score = getattr(d, "score", None)
        if score:  # score가 0이거나 None이면 제외
            item["score"] = score
        ref.append(item)
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


def _normalize_doc_type(doc_type: str | None) -> str:
    return normalize_doc_type(doc_type or "")


def _normalize_device_name(device_name: str | None) -> str:
    if not device_name:
        return ""
    return str(device_name).strip().lower()


def _extract_page_value(metadata: Dict[str, Any] | None) -> int | None:
    if not isinstance(metadata, dict):
        return None
    page = metadata.get("page")
    if page is None:
        page = metadata.get("page_start") or metadata.get("page_end")
    try:
        page_num = int(page)
    except (TypeError, ValueError):
        return None
    # Allow page 0 (valid for myservice docs)
    return page_num if page_num >= 0 else None


def _extract_expanded_pages(metadata: Dict[str, Any] | None) -> list[int]:
    if not isinstance(metadata, dict):
        return []
    raw_pages = metadata.get("expanded_pages")
    if not isinstance(raw_pages, list):
        return []
    pages: list[int] = []
    for page in raw_pages:
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            continue
        if page_num < 0:
            continue
        pages.append(page_num)
    return sorted(set(pages))


def _merge_display_docs(docs: List[RetrievalResult]) -> List[RetrievalResult]:
    merged: List[RetrievalResult] = []
    index_by_doc_id: Dict[str, int] = {}

    for doc in docs:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        doc_type = _normalize_doc_type(meta.get("doc_type"))
        if doc_type not in DOC_TYPES_SAME_DOC:
            merged.append(doc)
            continue

        doc_id = doc.doc_id
        if doc_id not in index_by_doc_id:
            index_by_doc_id[doc_id] = len(merged)
            merged.append(doc)
            continue

        idx = index_by_doc_id[doc_id]
        base = merged[idx]
        base_meta = base.metadata if isinstance(base.metadata, dict) else {}
        doc_meta = meta if isinstance(meta, dict) else {}
        pages = _extract_expanded_pages(base_meta) + _extract_expanded_pages(doc_meta)
        merged_pages = sorted(set(pages)) if pages else []

        merged_meta = dict(base_meta)
        for key, value in doc_meta.items():
            if key not in merged_meta:
                merged_meta[key] = value
        if merged_pages:
            merged_meta["expanded_pages"] = merged_pages

        base_raw = base.raw_text or base.content or ""
        doc_raw = doc.raw_text or doc.content or ""
        raw_text = doc_raw if len(doc_raw) > len(base_raw) else base_raw

        base_content = base.content or ""
        doc_content = doc.content or ""
        content = doc_content if len(doc_content) > len(base_content) else base_content

        score = max(base.score, doc.score)

        merged[idx] = RetrievalResult(
            doc_id=base.doc_id,
            content=content,
            score=score,
            metadata=merged_meta,
            raw_text=raw_text,
        )

    return merged


def _combine_related_text(docs: List[RetrievalResult]) -> str:
    seen: set = set()
    parts: List[str] = []

    def _sort_key(item: RetrievalResult) -> tuple[int, int, str]:
        meta = item.metadata if isinstance(item.metadata, dict) else {}
        page = _extract_page_value(meta)
        chunk_id = str(meta.get("chunk_id", ""))
        if page is not None:
            return (0, page, chunk_id)
        return (1, 0, chunk_id)

    for d in sorted(docs, key=_sort_key):
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        page = _extract_page_value(meta)
        section = meta.get("section_type") or meta.get("chapter")
        text = (d.raw_text or d.content or "").strip()
        if not text:
            continue
        chunk_id = meta.get("chunk_id")
        key = chunk_id or (d.doc_id, page, section, text[:80])
        if key in seen:
            continue
        seen.add(key)
        if page is not None:
            parts.append(f"p{page}: {text}")
        elif section:
            parts.append(f"{section}: {text}")
        else:
            parts.append(text)

    return "\n\n".join(parts)


# -----------------------------
# 6) doc_lookup helpers
# -----------------------------

# doc_id 추출 패턴 (myservice 29392, gcb 12345, sop-001 등)
DOC_ID_PATTERNS = [
    # myservice 패턴 (오타 허용: myserv로 시작하면 매칭)
    r"(myserv[a-z]*)\s+(\d+)",  # myservice 29392, myservvice 29392
    r"(myserv[a-z]*)[-_](\d+)",  # myservice-29392, myservice_29392
    r"(myserv[a-z]*)(\d{4,})",  # myservice29392 (4자리 이상 숫자만)
    # gcb 패턴
    r"(gcb)\s+(\d+)",  # gcb 12345
    r"(gcb)[-_](\d+)",  # gcb-12345
    r"(gcb)(\d{4,})",  # gcb12345
    # sop 패턴
    r"(sop)\s*[-_]?\s*(\d+)",  # sop-001, sop 001
]


def _extract_doc_id_from_query(query: str) -> Optional[tuple[str, str]]:
    """쿼리에서 doc_type과 doc_id 추출 (Rule 기반).

    Returns:
        (doc_type, doc_number) 튜플 또는 None
        예: ("myservice", "29392")
    """
    query_lower = query.lower()
    for pattern in DOC_ID_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            doc_type, doc_number = match.groups()
            # doc_type 정규화 (오타 허용)
            if doc_type.startswith("myserv"):
                doc_type = "myservice"
            return (doc_type, doc_number)
    return None


def _format_history_for_prompt(chat_history: List[ChatHistoryEntry]) -> str:
    """chat_history를 프롬프트용 텍스트로 변환.

    주의: user는 content, assistant는 summary 필드 사용
    """
    lines = []
    for entry in chat_history[-5:]:  # 최근 5턴만
        role = entry.get("role", "")
        if role == "user":
            lines.append(f"User: {entry.get('content', '')}")  # user는 content
        elif role == "assistant":
            summary = entry.get("summary", "")  # assistant는 summary
            refs = entry.get("refs", [])
            lines.append(f"Assistant: {summary}")
            if refs:
                lines.append(f"  참조 문서: {', '.join(refs[:3])}")  # 상위 3개만
    return "\n".join(lines)


HISTORY_REFERENCE_PATTERNS = [
    r"이전\s*(대화|문서|내용)",  # 이전 대화, 이전 문서
    r"(위|앞)\s*(에서|의|문서|내용)",  # 위에서, 앞의 문서
    r"(그|저)\s*(문서|내용)",  # 그 문서, 저 내용
    r"아까\s*\w*\s*(문서|내용|결과)",  # 아까 찾은 내용, 아까 검색한 결과
    r"(방금|조금\s*전)\s*\w*\s*(말한|검색|찾은|내용|문서)",  # 방금 말한, 조금 전 검색한
    r"더\s*자세히",  # 더 자세히
    r"(검색|찾은)\s*(결과|문서).*(요약|정리|설명)",  # 검색 결과 요약
]


def _is_history_reference(query: str) -> bool:
    """쿼리가 이전 대화/문서를 참조하는지 Rule 기반 체크."""
    query_lower = query.lower()
    for pattern in HISTORY_REFERENCE_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


def _get_last_assistant_summary(chat_history: List[ChatHistoryEntry]) -> str:
    """history에서 가장 최근 assistant의 summary 추출."""
    for entry in reversed(chat_history):
        if entry.get("role") == "assistant":
            return entry.get("summary", "") or ""
    return ""


def _extract_keywords(text: str) -> set:
    """텍스트에서 키워드 추출 (한글 명사 + 영문 단어)."""
    if not text:
        return set()
    # 한글 단어 (2글자 이상)
    korean_words = set(re.findall(r"[가-힣]{2,}", text))
    # 영문 단어 (3글자 이상)
    english_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text))
    return korean_words | english_words


def _keyword_overlap_score(query: str, summary: str) -> float:
    """질문과 summary 간 키워드 overlap 점수 (0~1)."""
    if not query or not summary:
        return 0.0
    query_keywords = _extract_keywords(query)
    summary_keywords = _extract_keywords(summary)
    if not query_keywords or not summary_keywords:
        return 0.0
    overlap = query_keywords & summary_keywords
    # Jaccard 유사도 변형: overlap / query_keywords (질문 기준)
    return len(overlap) / len(query_keywords) if query_keywords else 0.0


def _needs_detail(query: str) -> bool:
    """질문이 자세한 내용/근거/수치를 요구하는지 체크."""
    detail_patterns = [
        r"더\s*자세히",
        r"(근거|출처|레퍼런스)",
        r"(정확한|구체적인)\s*(절차|방법|수치|값)",
        r"(몇|얼마|어느\s*정도)",  # 수치 관련
        r"(토크|압력|온도|시간)\s*(값|설정|스펙)",
    ]
    for pattern in detail_patterns:
        if re.search(pattern, query.lower()):
            return True
    return False


def history_intent_node(state: AgentState) -> Dict[str, Any]:
    """이전 대화 참조 의도 판단 (route_node 이전 실행).

    판단 순서:
    1. Rule 체크: 명시적 history 참조 패턴
    2. Keyword overlap: 질문 vs 직전 assistant summary

    Returns:
        {
            "history_intent": bool,
            "history_confidence": float (0~1),
            "history_source": str ("rule" | "keyword_overlap" | "none")
        }
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    history_doc_ids = _get_doc_ids_from_history(chat_history) if chat_history else []

    # history 없으면 바로 종료
    if not chat_history:
        logger.info("[history_intent_node] No chat_history, skipping")
        return {
            "history_intent": False,
            "history_confidence": 0.0,
            "history_source": "none",
            "_events": [
                "history_intent: history_count=0 (skip)"
            ],
        }

    # [1] Rule 체크: 명시적 패턴
    if _is_history_reference(query):
        logger.info("[history_intent_node] DETECTED: rule match, query='%s'", query[:50])
        return {
            "history_intent": True,
            "history_confidence": 1.0,
            "history_source": "rule",
            "_events": [
                f"history_intent: source=rule, confidence=1.00, history_count={len(chat_history)}, doc_ids={history_doc_ids}"
            ],
        }

    # [2] Keyword overlap: 질문 vs 직전 assistant summary
    last_summary = _get_last_assistant_summary(chat_history)
    if last_summary:
        overlap_score = _keyword_overlap_score(query, last_summary)
        logger.info(
            "[history_intent_node] Keyword overlap: score=%.2f, query='%s', summary='%s'",
            overlap_score, query[:30], last_summary[:50]
        )
        if overlap_score >= 0.3:  # threshold
            return {
                "history_intent": True,
                "history_confidence": overlap_score,
                "history_source": "keyword_overlap",
                "_events": [
                    f"history_intent: source=keyword_overlap, score={overlap_score:.2f}, history_count={len(chat_history)}, doc_ids={history_doc_ids}"
                ],
            }

    logger.info("[history_intent_node] No history intent detected")
    return {
        "history_intent": False,
        "history_confidence": 0.0,
        "history_source": "none",
        "_events": [
            f"history_intent: source=none, history_count={len(chat_history)}, doc_ids={history_doc_ids}"
        ],
    }


def _detect_doc_lookup_intent(
    llm: BaseLLM,
    query: str,
    chat_history: List[ChatHistoryEntry],
) -> dict:
    """LLM으로 doc_lookup 의도 판단 (_invoke_llm 사용)."""

    # history가 없으면 doc_lookup 불가
    if not chat_history:
        return {"is_doc_lookup": False}

    # 최근 assistant 응답에 doc_ids가 없으면 불가
    has_doc_refs = False
    for entry in reversed(chat_history):
        if entry.get("role") == "assistant" and entry.get("doc_ids"):
            has_doc_refs = True
            break
    if not has_doc_refs:
        return {"is_doc_lookup": False}

    # history 포맷팅
    history_text = _format_history_for_prompt(chat_history)

    system = "사용자의 질문이 이전 대화에서 언급된 문서를 참조하는지 판단하세요."
    user = f"""이전 대화:
{history_text}

현재 질문: {query}

판단 기준:
- "그 문서", "아까", "위에서 말한", "더 자세히", "방금 말한" 등의 표현
- 이전 답변의 특정 부분을 깊이 묻는 경우
- 새로운 장비나 주제가 아닌, 이전 맥락을 이어가는 경우

이 질문이 이전 문서를 참조하면 "yes", 새로운 검색이 필요하면 "no"로만 답하세요."""

    response = _invoke_llm(llm, system, user)
    is_doc_lookup = "yes" in response.lower()

    logger.info("_detect_doc_lookup_intent: query=%s, result=%s", query[:50], is_doc_lookup)
    return {"is_doc_lookup": is_doc_lookup}


def _get_doc_ids_from_history(chat_history: List[ChatHistoryEntry]) -> List[str]:
    """history에서 가장 최근 assistant의 doc_ids 추출."""
    for entry in reversed(chat_history):
        if entry.get("role") == "assistant":
            doc_ids = entry.get("doc_ids", [])
            if doc_ids:
                return doc_ids[:3]  # 최대 3개
    return []


def doc_lookup_node(
    state: AgentState,
    *,
    doc_fetcher,  # Callable[[str], List[RetrievalResult]]
) -> Dict[str, Any]:
    """doc_id로 직접 문서 조회 (MQ 생략, retrieve 우회).

    Args:
        state: AgentState (lookup_doc_ids 필드 사용)
        doc_fetcher: fetch_doc_chunks 함수 (doc_id -> chunks)

    Returns:
        - 성공: {"docs": [...], "lookup_doc_ids": [...]}
        - 실패: {"route": "general"} (fallback)
    """
    doc_ids = state.get("lookup_doc_ids", [])
    lookup_source = state.get("lookup_source", "unknown")
    events: List[str] = []

    if not doc_ids:
        msg = "doc_lookup_node: no doc_ids, fallback to MQ"
        logger.info(msg)
        return {"route": "general", "_events": [msg]}

    # fetch_doc_chunks로 직접 조회
    all_docs: List[RetrievalResult] = []
    valid_doc_ids: List[str] = []

    for doc_id in doc_ids[:3]:  # 최대 3개 문서
        if doc_fetcher is None:
            msg = "doc_lookup_node: doc_fetcher is None"
            logger.warning(msg)
            events.append(msg)
            break
        chunks = doc_fetcher(doc_id)
        if chunks:
            all_docs.extend(chunks)
            valid_doc_ids.append(doc_id)
            logger.info("doc_lookup_node: fetched %d chunks for doc_id=%s", len(chunks), doc_id)
        else:
            logger.warning("doc_lookup_node: doc_id=%s not found in ES", doc_id)

    if not all_docs:
        # 모든 doc_id 검증 실패 → fallback
        msg = "doc_lookup_node: all doc_ids invalid, fallback to MQ"
        logger.info(msg)
        return {"route": "general", "_events": events + [msg]}

    # docs 필드에 직접 설정
    logger.info(
        "doc_lookup_node: success, source=%s, doc_ids=%s, total_chunks=%d",
        lookup_source,
        valid_doc_ids,
        len(all_docs),
    )
    events.append(
        f"doc_lookup_node: success source={lookup_source} doc_ids={valid_doc_ids} chunks={len(all_docs)}"
    )
    return {
        "docs": all_docs,
        "all_docs": all_docs,  # 재생성용
        "lookup_doc_ids": valid_doc_ids,
        "_events": events,
    }


# -----------------------------
# 7) Graph node helpers
# -----------------------------
def route_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    """라우팅 노드: general | doc_lookup | retrieval 분기.

    Router가 역할/행동 기반으로 분류:
    - general: 일반 대화 (RAG 우회)
    - doc_lookup: 이전 문서/대화 기반 답변 (MQ 생략)
    - retrieval: 검색 필요 (RAG 파이프라인)

    실행 순서:
    1. Rule 기반 doc_id 체크 (LLM 호출 없음) - "myservice 12345"
    2. LLM 라우터 호출 (general | doc_lookup | retrieval)
    3. doc_lookup인 경우 doc_ids 유무에 따라 분기
    """
    original_query = state.get("query") or ""
    query = state.get("query_en") or original_query
    chat_history = state.get("chat_history", [])

    _trace_log("route_node", state, "ENTER", {
        "query_full": original_query,
        "chat_history_count": len(chat_history),
    })

    events: List[str] = []

    # === 디버깅 로그: 입력 상태 ===
    history_doc_ids = _get_doc_ids_from_history(chat_history) if chat_history else []
    events.append(f"router_input: history_count={len(chat_history)}, doc_ids={history_doc_ids}")
    logger.info(
        "[route_node] INPUT: query='%s', history_count=%d, history_doc_ids=%s",
        original_query[:60] if original_query else "",
        len(chat_history),
        history_doc_ids,
    )

    def _return(payload: Dict[str, Any], note: str | None = None) -> Dict[str, Any]:
        if note:
            events.append(note)
        payload["_events"] = list(events)
        _trace_log("route_node", state, "EXIT", {
            "route": payload.get("route"),
            "lookup_doc_ids": payload.get("lookup_doc_ids"),
        })
        return payload

    # [0] 강제 retrieval: selected_devices가 명시적으로 전달된 경우 (기기 재검색)
    selected_devices = state.get("selected_devices", [])
    if selected_devices:
        logger.info("[route_node] FORCED: retrieval (selected_devices=%s)", selected_devices)
        return _return(
            {"route": "retrieval"},
            f"router_forced: retrieval (selected_devices={selected_devices})",
        )

    # [1] Rule 기반: 쿼리에서 doc_id 패턴 체크 (LLM 호출 전에 빠르게 처리)
    doc_info = _extract_doc_id_from_query(original_query)
    if doc_info:
        doc_type, doc_number = doc_info
        logger.info("[route_node] DECISION: doc_lookup (rule: doc_id pattern), doc_type=%s, doc_id=%s", doc_type, doc_number)
        return _return(
            {
                "route": "doc_lookup",
                "lookup_doc_ids": [doc_number],
                "lookup_source": "query",
            },
            f"router_decision: doc_lookup (doc_id pattern), doc_id={doc_number}",
        )

    # [2] LLM Router 호출 (general | doc_lookup | retrieval)
    history_text = _format_history_for_prompt(chat_history) if chat_history else "없음"
    logger.info("[route_node] LLM INPUT: history_text='%s'", history_text[:200] if history_text else "없음")

    user = _format_prompt(spec.router.user, {
        "sys.query": query,
        "sys.history": history_text,
    })
    route = _parse_route(_invoke_llm(llm, spec.router.system, user))

    logger.info("[route_node] LLM DECISION: route=%s", route)
    events.append(f"router_llm: route={route}, history_count={len(chat_history)}, doc_ids={history_doc_ids}")

    # [3] doc_lookup인 경우 history에서 doc_ids 추출
    if route == "doc_lookup":
        if history_doc_ids:
            logger.info("[route_node] FINAL: doc_lookup, doc_ids=%s", history_doc_ids)
            return _return(
                {
                    "route": route,
                    "lookup_doc_ids": history_doc_ids,
                    "lookup_source": "history",
                },
                f"router_final: doc_lookup (llm) doc_ids={history_doc_ids}",
            )

        # doc_ids를 찾을 수 없으면 history_answer로 fallback (summary 기반)
        logger.info("[route_node] FALLBACK: doc_lookup but no doc_ids -> history_answer")
        return _return(
            {"route": "history_answer"},
            "router_fallback: doc_lookup→history_answer (no doc_ids in history)",
        )

    logger.info("[route_node] FINAL: route=%s", route)
    return _return({"route": route}, f"router_final: {route}")


def mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    """Generate multi-queries for retrieval using unified retrieval_mq prompt."""
    # Generate MQ in both English and Korean for bilingual retrieval
    query_en = state.get("query_en") or state["query"]
    query_ko = state.get("query_ko") or state["query"]

    _trace_log("mq_node", state, "ENTER", {
        "query_en": query_en,
        "query_ko": query_ko,
    })

    logger.info("mq_node: bilingual - EN=%s..., KO=%s...",
                query_en[:40] if query_en else None,
                query_ko[:40] if query_ko else None)

    if state.get("skip_mq") and state.get("search_queries"):
        logger.info("mq_node: search_queries override provided, skipping MQ generation")
        _trace_log("mq_node", state, "SKIP", {"reason": "search_queries override"})
        return {}

    # MQ generation needs more tokens than classification
    mq_kwargs = {"max_tokens": 4096}

    def _generate_mq_bilingual(spec_template) -> tuple[List[str], List[str]]:
        """Generate MQ in both English and Korean."""
        # English MQ - add explicit language instruction
        system_en = spec_template.system + "\n\n**IMPORTANT: Generate all queries in English.**"
        user_en = _format_prompt(spec_template.user, {"sys.query": query_en})
        raw_en = _invoke_llm(llm, system_en, user_en, **mq_kwargs)
        logger.debug("mq_node(retrieval/en) raw LLM output:\n%s", raw_en)
        mq_en = _parse_queries(raw_en)
        logger.info("mq_node(retrieval/en): %d queries generated", len(mq_en))
        for i, q in enumerate(mq_en, 1):
            logger.info("  [EN-%d] %s", i, q)
        if len(mq_en) == 0:
            logger.warning("mq_node(retrieval/en): parsing failed! raw output:\n%s", raw_en)

        # Korean MQ - add explicit Korean language instruction
        system_ko = spec_template.system + "\n\n**중요: 모든 검색어를 반드시 한국어로 생성하세요. Generate all queries in Korean.**"
        user_ko = _format_prompt(spec_template.user, {"sys.query": query_ko})
        raw_ko = _invoke_llm(llm, system_ko, user_ko, **mq_kwargs)
        logger.debug("mq_node(retrieval/ko) raw LLM output:\n%s", raw_ko)
        mq_ko = _parse_queries(raw_ko)
        logger.info("mq_node(retrieval/ko): %d queries generated", len(mq_ko))
        for i, q in enumerate(mq_ko, 1):
            logger.info("  [KO-%d] %s", i, q)
        if len(mq_ko) == 0:
            logger.warning("mq_node(retrieval/ko): parsing failed! raw output:\n%s", raw_ko)

        return mq_en, mq_ko

    # Use unified retrieval_mq prompt
    retrieval_mq_list, retrieval_mq_ko_list = _generate_mq_bilingual(spec.retrieval_mq)

    logger.info(
        "mq_node: total - retrieval(en=%d, ko=%d)",
        len(retrieval_mq_list), len(retrieval_mq_ko_list),
    )

    _trace_log("mq_node", state, "EXIT", {
        "mq_en_count": len(retrieval_mq_list),
        "mq_ko_count": len(retrieval_mq_ko_list),
        "mq_en": retrieval_mq_list,
        "mq_ko": retrieval_mq_ko_list,
    })

    # Return both new unified fields and legacy fields for compatibility
    return {
        "retrieval_mq_list": retrieval_mq_list,
        "retrieval_mq_ko_list": retrieval_mq_ko_list,
        # Legacy fields (for compatibility with st_gate_node, st_mq_node)
        "general_mq_list": retrieval_mq_list,
        "general_mq_ko_list": retrieval_mq_ko_list,
        "setup_mq_list": [],
        "ts_mq_list": [],
        "setup_mq_ko_list": [],
        "ts_mq_ko_list": [],
    }


def st_gate_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    """Determine if additional query refinement (sequential thinking) is needed."""
    # Get retrieval MQ list (unified structure)
    retrieval_mq_list = state.get("retrieval_mq_list", [])

    # Use English query for processing
    q = state.get("query_en") or state["query"]
    mapping = {
        "sys.query": q,
        "retrieval_mq": "\n".join(retrieval_mq_list),
    }
    user = _format_prompt(spec.st_gate.user, mapping)
    gate = _parse_gate(_invoke_llm(llm, spec.st_gate.system, user))
    return {"st_gate": gate}


def _is_garbage_query(q: str) -> bool:
    """Check if a query is garbage (prompt label leak or too short)."""
    lower = q.lower().strip()
    # Filter out prompt label leaks
    garbage_patterns = [
        "setup_mq:", "ts_mq:", "general_mq:", "gate:",
        "질문:", "queries:", "search_queries:",
    ]
    for pat in garbage_patterns:
        if pat in lower:
            return True
    # Common placeholder outputs
    if re.fullmatch(r"(q|query)\s*[-_]*\s*\d+", lower):
        return True
    # Ellipsis or punctuation-only
    if re.fullmatch(r"[.\-–—_…·\s]+", q.strip()):
        return True
    # Likely truncated outputs (dangling stopwords/particles/verb stems)
    trimmed_lower = re.sub(r"[\"'`.,;:!?]+$", "", lower).strip()
    last_token = re.split(r"\s+", trimmed_lower)[-1] if trimmed_lower else ""
    if last_token in {
        "a", "an", "the", "of", "to", "for", "with", "and", "or", "in", "on",
        "at", "from", "by", "as", "is", "are", "was", "were",
    }:
        return True
    if last_token in {
        "의", "을", "를", "에", "에서", "에게", "으로", "로", "와", "과", "및",
        "또는", "부터", "까지", "처럼",
    }:
        return True
    if last_token in {"있", "하", "되", "보", "알", "가", "오", "주", "말"}:
        return True
    # Meta/explanatory lines that sometimes leak into outputs
    if "let's" in lower or "let us" in lower:
        if "query" in lower or "queries" in lower or "produce" in lower or "example" in lower:
            return True
    if "example:" in lower and ("query" in lower or "queries" in lower or "output" in lower):
        return True
    if "output format" in lower or "output only" in lower:
        return True
    # Filter out queries that are just labels without actual content
    if lower in {"setup_mq", "ts_mq", "general_mq", "gate", "no_st", "need_st"}:
        return True
    # Filter out very short queries (less than 3 chars)
    if len(q.strip()) < 3:
        return True
    return False


def _contains_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


def _fill_to_n(base: List[str], candidates: List[str], n: int) -> List[str]:
    for q in candidates:
        q = str(q).strip()
        if q and q not in base:
            base.append(q)
        if len(base) >= n:
            break
    return base


def st_mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    """Refine search queries when need_st is triggered."""
    # Get unified retrieval MQ lists
    mq_en_list = state.get("retrieval_mq_list", [])
    mq_ko_list = state.get("retrieval_mq_ko_list", [])

    # Use English query for processing
    q_en = state.get("query_en") or state["query"]
    # Get Korean query for bilingual search
    q_ko = state.get("query_ko")

    if state.get("skip_mq") and state.get("search_queries"):
        provided = [str(q).strip() for q in state.get("search_queries", []) if str(q).strip()]
        if provided:
            return {"search_queries": _dedupe_queries(provided)}
        return {"search_queries": [q_en] if q_en else []}

    gate = state.get("st_gate", "no_st")
    queries: List[str] = []

    # When no structured transform is needed, skip the extra LLM call but
    # keep the downstream bilingual merge/translation behavior.
    if gate == "no_st":
        logger.info("st_mq_node: st_gate=no_st, skipping st_mq LLM call")
    else:
        mapping = {
            "sys.query": q_en,
            "retrieval_mq": "\n".join(mq_en_list),
            "st_gate": gate,
        }
        user = _format_prompt(spec.st_mq.user, mapping)
        raw = _invoke_llm(llm, spec.st_mq.system, user, max_tokens=1024)
        logger.info("st_mq_node: raw output=%s", raw)
        queries = _parse_queries(raw)
        # Filter out garbage queries (prompt label leaks, too short, etc.)
        queries = [q for q in queries if not _is_garbage_query(q)]

    # Also filter MQ lists for garbage labels/leaks before merging.
    mq_en_list = [q for q in mq_en_list if not _is_garbage_query(q)]
    mq_ko_list = [q for q in mq_ko_list if not _is_garbage_query(q)]

    # Build English queries (prefer st_mq output, then q_en, then MQ list)
    english_queries: List[str] = []
    _fill_to_n(english_queries, queries, 3)
    if q_en:
        _fill_to_n(english_queries, [q_en], 3)
    _fill_to_n(english_queries, mq_en_list, 3)

    # Build Korean queries (prefer KO MQ list, then q_ko, then translate EN if needed)
    ko_candidates = [q for q in mq_ko_list if _contains_korean(q)]
    if q_ko and _contains_korean(q_ko):
        ko_candidates = [q_ko] + ko_candidates
    korean_queries = _dedupe_queries(ko_candidates)

    if len(korean_queries) < 3 and spec.translate is not None:
        def _translate(text: str, target_lang: str) -> str:
            target_name = {"en": "English", "ko": "Korean"}.get(target_lang, target_lang)
            user = _format_prompt(spec.translate.user, {
                "query": text,
                "target_language": target_name,
            })
            result = _invoke_llm(llm, spec.translate.system, user)
            result = result.strip().strip('"').strip("'").strip()
            return result if result else text

        for q in english_queries:
            if len(korean_queries) >= 3:
                break
            if _contains_korean(q):
                continue
            translated = _translate(q, "ko")
            if translated and translated not in korean_queries:
                korean_queries.append(translated)

    korean_queries = korean_queries[:3]
    english_queries = english_queries[:3]

    merged = english_queries + korean_queries
    logger.info("st_mq_node: final search_queries (bilingual)=%s", merged)
    return {"search_queries": merged}


def retrieve_node(
    state: AgentState,
    *,
    retriever: Retriever,
    reranker: Any = None,
    retrieval_top_k: int = 20,
    final_top_k: int = 10,
) -> Dict[str, Any]:
    """Retrieve documents with dual search strategy and rerank.

    If devices are selected:
      - Search 1: retrieval_top_k docs filtered by selected devices (OR filter)
      - Search 2: retrieval_top_k docs without filter (general search)
      - Combine and rerank to get final_top_k docs

    If no devices selected:
      - Search retrieval_top_k docs without filter
      - Rerank to get final_top_k docs
    """
    queries = state.get("search_queries", [state["query"]])
    selected_devices = state.get("selected_devices", [])
    selected_doc_types = state.get("selected_doc_types", [])
    selected_doc_ids = [str(x).strip() for x in state.get("selected_doc_ids", []) if str(x).strip()]
    selected_doc_type_filters = expand_doc_type_selection(selected_doc_types)
    route = state.get("route", "general")
    original_query = state["query"]

    _trace_log("retrieve_node", state, "ENTER", {
        "search_queries": queries,
        "selected_devices": selected_devices,
        "selected_doc_types": selected_doc_types,
        "route": route,
    })

    # Search parameter override (for Retrieval Test)
    search_override = state.get("search_override")

    # search_queries from st_mq_node already contains EN+KO queries
    # No need to add bilingual queries here
    query_en = state.get("query_en")

    # Route-based doc_type bias (boost-like behavior): when the user did not
    # explicitly select doc types, run an additional route-biased search and
    # merge with a general search to avoid hard filter recall loss.
    route_doc_type_groups: Dict[str, List[str]] = {
        "setup": ["setup", "SOP"],
        "ts": ["ts"],
        "general": [],
    }
    route_doc_type_filters: List[str] = []
    route_bias_enabled = False
    if not selected_doc_types:
        route_doc_type_filters = expand_doc_type_selection(route_doc_type_groups.get(route, []))
        route_bias_enabled = bool(route_doc_type_filters)
        if route_bias_enabled:
            logger.info(
                "retrieve_node: route bias enabled route=%s doc_types=%s",
                route,
                route_doc_type_filters,
            )

    selected_device_set = {
        _normalize_device_name(d) for d in selected_devices if _normalize_device_name(d)
    }
    # Only user-selected doc types act as a hard filter.
    selected_doc_type_set = {
        _normalize_doc_type(dt) for dt in selected_doc_type_filters if _normalize_doc_type(dt)
    }

    candidate_k = max(retrieval_top_k, final_top_k * 2, 20)

    all_docs: List[RetrievalResult] = []
    seen: set = set()

    def _matches_doc_type(doc: RetrievalResult) -> bool:
        if not selected_doc_type_set:
            return True
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        doc_type = _normalize_doc_type(meta.get("doc_type"))
        return bool(doc_type) and doc_type in selected_doc_type_set

    def _matches_device(doc: RetrievalResult) -> bool:
        if not selected_device_set:
            return True
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        device_name = _normalize_device_name(meta.get("device_name"))
        return bool(device_name) and device_name in selected_device_set

    def _add_docs(docs_to_add: List[RetrievalResult], *, filter_devices: bool) -> None:
        """Add docs to all_docs, avoiding duplicates and honoring filters."""
        for d in docs_to_add:
            if not _matches_doc_type(d):
                continue
            if filter_devices and not _matches_device(d):
                continue
            key = (d.doc_id, hash(d.raw_text or d.content))
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    # Use search_queries directly (already contains EN+KO from st_mq_node)
    all_queries = queries

    if selected_devices:
        # Dual search strategy: device-filtered + general
        logger.info("retrieve_node: dual search with devices=%s, queries=%d", selected_devices, len(all_queries))

        # Search 1: Device-filtered search with device_names as OR filter
        device_doc_types = selected_doc_type_filters or (route_doc_type_filters if route_bias_enabled else None)
        for q in all_queries:
            device_docs = retriever.retrieve(
                q,
                top_k=candidate_k,
                device_names=selected_devices,
                doc_types=device_doc_types,
                search_override=search_override,
            )
            _add_docs(device_docs, filter_devices=True)

        device_filtered_count = len(all_docs)
        logger.info("retrieve_node: device-filtered search found %d docs", device_filtered_count)

        # Search 2: General search without device filter
        general_doc_types = selected_doc_type_filters or None
        for q in all_queries:
            general_docs = retriever.retrieve(
                q,
                top_k=candidate_k,
                doc_types=general_doc_types,
                search_override=search_override,
            )
            _add_docs(general_docs, filter_devices=False)

        logger.info("retrieve_node: after general search, total %d docs", len(all_docs))

    else:
        # No device selection: optionally run route-biased search + general search.
        logger.info("retrieve_node: general search (no device filter), queries=%d", len(all_queries))
        if selected_doc_type_filters:
            # User-selected doc types: honor as hard filter.
            for q in all_queries:
                docs = retriever.retrieve(
                    q,
                    top_k=candidate_k,
                    doc_types=selected_doc_type_filters,
                    search_override=search_override,
                )
                _add_docs(docs, filter_devices=False)
        elif route_bias_enabled:
            # Route-biased search (doc_type filter) + general search (no filter).
            for q in all_queries:
                biased_docs = retriever.retrieve(
                    q,
                    top_k=candidate_k,
                    doc_types=route_doc_type_filters,
                    search_override=search_override,
                )
                _add_docs(biased_docs, filter_devices=False)
            for q in all_queries:
                general_docs = retriever.retrieve(
                    q,
                    top_k=candidate_k,
                    doc_types=None,
                    search_override=search_override,
                )
                _add_docs(general_docs, filter_devices=False)
        else:
            for q in all_queries:
                docs = retriever.retrieve(
                    q,
                    top_k=candidate_k,
                    doc_types=None,
                    search_override=search_override,
                )
                _add_docs(docs, filter_devices=False)

    logger.info("retrieve_node: collected %d unique docs before rerank", len(all_docs))
    if selected_doc_ids:
        before = len(all_docs)
        all_docs = [d for d in all_docs if str(d.doc_id) in set(selected_doc_ids)]
        logger.info("retrieve_node: filtered by selected_doc_ids %d -> %d", before, len(all_docs))
    if len(all_docs) > candidate_k:
        all_docs = sorted(all_docs, key=lambda d: d.score, reverse=True)[:candidate_k]

    # Store all_docs before reranking for regeneration (up to retrieval_top_k)
    # Sort by score and take top retrieval_top_k for regeneration options
    all_docs_for_regen = sorted(all_docs, key=lambda d: d.score, reverse=True)[:retrieval_top_k]

    # Rerank if reranker is available
    # Use English query for reranking - cross-encoder models often work better with English
    rerank_query = query_en if query_en else original_query
    if reranker is not None and all_docs:
        logger.info("retrieve_node: reranking %d docs to top %d (using query_en)", len(all_docs), final_top_k)
        docs = reranker.rerank(rerank_query, all_docs, top_k=final_top_k)
    else:
        # No reranker: just take top final_top_k by score
        docs = sorted(all_docs, key=lambda d: d.score, reverse=True)[:final_top_k]

    logger.info("retrieve_node: returning %d docs (all_docs_for_regen: %d)", len(docs), len(all_docs_for_regen))

    # 검색 결과 상세 트레이스
    doc_summaries = []
    for i, d in enumerate(docs[:5]):  # 상위 5개만 기록
        meta = d.metadata if hasattr(d, "metadata") and d.metadata else {}
        doc_summaries.append({
            "rank": i + 1,
            "doc_id": d.doc_id,
            "score": round(d.score, 4) if d.score else None,
            "device": meta.get("device_name"),
            "doc_type": meta.get("doc_type"),
            "title": (meta.get("doc_description") or "")[:50],
        })

    _trace_log("retrieve_node", state, "EXIT", {
        "doc_count": len(docs),
        "all_docs_count": len(all_docs_for_regen),
        "top_docs": doc_summaries,
    })

    return {
        "docs": docs,
        "ref_json": results_to_ref_json(docs),
        "all_docs": all_docs_for_regen,  # 재생성용 전체 문서 (최대 retrieval_top_k개)
    }


def expand_related_docs_node(
    state: AgentState,
    *,
    page_fetcher: Any = None,
    doc_fetcher: Any = None,
    page_window: int = RELATED_PAGE_WINDOW,
    max_ref_chars: int | None = None,
) -> Dict[str, Any]:
    """Expand answer references by doc_type rules."""
    if page_fetcher is None and doc_fetcher is None:
        msg = "expand_related: skipped (no fetcher available)"
        logger.info(msg)
        return {"_events": [msg]}

    docs = state.get("docs", [])
    if not docs:
        msg = "expand_related: skipped (no docs)"
        logger.info(msg)
        return {"_events": [msg]}

    expanded_docs: List[RetrievalResult] = []
    # Use expand_top_k from state if set, otherwise use default EXPAND_TOP_K
    expand_top_k = state.get("expand_top_k") or EXPAND_TOP_K
    max_expand = max(0, int(expand_top_k))
    total_docs = len(docs)
    same_doc_targets = 0
    page_targets = 0
    skipped_targets = 0
    fetched_related_total = 0
    expanded_count = 0

    for idx, doc in enumerate(docs):
        if idx >= max_expand:
            expanded_docs.append(doc)
            continue

        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        doc_type = _normalize_doc_type(meta.get("doc_type"))
        related_docs: List[RetrievalResult] = []

        expanded_pages: List[int] = []
        if doc_type in DOC_TYPES_SAME_DOC and doc_fetcher is not None:
            same_doc_targets += 1
            related_docs = doc_fetcher(doc.doc_id)
            # Extract all pages from fetched chunks
            for rd in related_docs:
                rd_meta = rd.metadata if isinstance(rd.metadata, dict) else {}
                rd_page = _extract_page_value(rd_meta)
                if rd_page is not None and rd_page not in expanded_pages:
                    expanded_pages.append(rd_page)
            expanded_pages.sort()
        elif page_fetcher is not None:
            page = _extract_page_value(meta)
            if page is not None:
                page_targets += 1
                page_min = max(1, page - page_window)
                page_max = page + page_window
                pages = list(range(page_min, page_max + 1))
                expanded_pages = pages
                related_docs = page_fetcher(doc.doc_id, pages)
            else:
                skipped_targets += 1
        else:
            skipped_targets += 1

        if related_docs:
            fetched_related_total += len(related_docs)
            combined = _combine_related_text(related_docs)
            if combined:
                expanded_count += 1
                # Update metadata with expanded pages info
                updated_meta = dict(meta) if meta else {}
                if expanded_pages:
                    updated_meta["expanded_pages"] = expanded_pages
                expanded_docs.append(
                    RetrievalResult(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        score=doc.score,
                        metadata=updated_meta,
                        raw_text=combined,
                    )
                )
                continue

        if expanded_pages:
            updated_meta = dict(meta) if meta else {}
            updated_meta["expanded_pages"] = expanded_pages
            expanded_docs.append(
                RetrievalResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=doc.score,
                    metadata=updated_meta,
                    raw_text=doc.raw_text,
                )
            )
        else:
            expanded_docs.append(doc)

    summary = (
        "expand_related: total_docs=%d expand_top_k=%d same_doc_targets=%d "
        "page_targets=%d skipped_targets=%d fetched_related=%d expanded=%d"
        % (
            total_docs,
            min(total_docs, max_expand),
            same_doc_targets,
            page_targets,
            skipped_targets,
            fetched_related_total,
            expanded_count,
        )
    )
    logger.info(summary)

    display_docs = _merge_display_docs(expanded_docs[:min(total_docs, max_expand)])
    return {
        "docs": expanded_docs,
        "display_docs": display_docs,
        "answer_ref_json": results_to_ref_json(
            display_docs,
            max_chars=max_ref_chars,
            prefer_raw_text=True,
        ),
        "_events": [summary],
    }


def ask_user_after_retrieve_node(state: AgentState) -> Command[Literal["expand_related", "refine_and_retrieve"]]:
    """Retrieval 후 사용자에게 검색 결과를 보여주고 피드백을 받는 노드.

    사용자 응답:
    - True 또는 빈 문자열: 검색 결과 승인 → expand_related로 진행
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
                    goto="expand_related",
                    update={
                        "docs": deduped_docs,
                        "ref_json": results_to_ref_json(deduped_docs),
                        "retrieval_confirmed": True,
                        "user_feedback": None,
                    },
                )

    # 승인: True 또는 빈 문자열
    if decision is True or decision == "":
        return Command(
            goto="expand_related",
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


def _get_answer_template(
    spec: PromptSpec,
    route: Route,
    language: Optional[str],
) -> PromptTemplate:
    """Get the unified retrieval answer template based on language."""
    if language == "en" and spec.retrieval_ans_en is not None:
        return spec.retrieval_ans_en
    if language == "ja" and spec.retrieval_ans_ja is not None:
        return spec.retrieval_ans_ja
    return spec.retrieval_ans


def _collect_suggested_devices(docs: List[Any]) -> List[Dict[str, Any]]:
    """검색 결과에서 device_name 집계 (문서 수 내림차순).

    Args:
        docs: RetrievalResult 또는 dict 리스트

    Returns:
        [{"name": "SUPRA XP", "count": 18}, ...] 형태의 리스트
        count는 청크 수가 아닌 고유 문서(doc_id) 수
    """
    from collections import defaultdict

    EXCLUDE_NAMES = {"", "ALL", "etc", "ETC", "all", "N/A", "Unknown"}

    # device_name -> set of unique doc_ids
    device_doc_ids: dict[str, set] = defaultdict(set)

    for doc in docs:
        # RetrievalResult 또는 dict 모두 지원
        if hasattr(doc, "metadata"):
            metadata = doc.metadata
            doc_id = getattr(doc, "doc_id", None)
        elif isinstance(doc, dict):
            metadata = doc.get("metadata", {})
            doc_id = doc.get("doc_id")
        else:
            continue

        device_name = metadata.get("device_name", "") if metadata else ""
        if device_name and device_name.strip() not in EXCLUDE_NAMES and doc_id:
            device_doc_ids[device_name.strip()].add(doc_id)

    # count 내림차순 정렬
    sorted_devices = sorted(
        device_doc_ids.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    return [
        {"name": name, "count": len(doc_ids)}
        for name, doc_ids in sorted_devices
    ]


def answer_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    """Generate answer using unified retrieval_ans prompt."""
    route = state["route"]
    detected_language = state.get("detected_language", "ko")
    ref_items = state.get("answer_ref_json")
    if not ref_items:
        docs = state.get("display_docs") or state.get("docs") or []
        if docs and hasattr(docs[0], "doc_id"):
            ref_items = results_to_ref_json(docs, max_chars=None, prefer_raw_text=True)
        else:
            ref_items = state.get("ref_json", [])
    ref_text = ref_json_to_text(ref_items)

    _trace_log("answer_node", state, "ENTER", {
        "route": route,
        "detected_language": detected_language,
        "ref_count": len(ref_items),
        "ref_chars": len(ref_text),
    })

    logger.info(
        "answer_node: route=%s, language=%s, refs_chars=%d, docs=%d",
        route,
        detected_language,
        len(ref_text),
        len(ref_items),
    )

    # Use English query for processing if available
    query_for_prompt = state.get("query_en") or state["query"]
    mapping = {"sys.query": query_for_prompt, "ref_text": ref_text}

    # Select language-specific unified template
    # Routes: retrieval, doc_lookup → use retrieval_ans
    if detected_language == "en":
        tmpl = spec.retrieval_ans_en or spec.retrieval_ans
    elif detected_language == "ja":
        tmpl = spec.retrieval_ans_ja or spec.retrieval_ans
    else:  # ko or default
        tmpl = spec.retrieval_ans
    logger.info("answer_node: using retrieval_ans template (lang=%s) for route=%s", detected_language, route)

    user = _format_prompt(tmpl.user, mapping)
    logger.info("answer_node: user_prompt_chars=%d, system_prompt_chars=%d", len(user), len(tmpl.system))
    answer, reasoning = _invoke_llm_with_reasoning(llm, tmpl.system, user, max_tokens=MAX_TOKENS_ANSWER)
    logger.info("answer_node: answer_chars=%d, reasoning_chars=%d, answer_preview=%s", len(answer), len(reasoning) if reasoning else 0, answer[:500] if answer else "(empty)")

    # suggested_devices 집계 (장비 미지정 시에만)
    events: List[str] = []
    suggested_devices = None
    auto_parsed_device = state.get("auto_parsed_device")
    auto_parsed_devices = state.get("auto_parsed_devices")
    selected_devices = state.get("selected_devices")
    selected_device = state.get("selected_device")
    has_device_filter = bool(auto_parsed_device or auto_parsed_devices or selected_devices or selected_device)

    logger.info(
        "answer_node: auto_parsed_device=%s, auto_parsed_devices=%s, selected_devices=%s, selected_device=%s, has_device_filter=%s",
        auto_parsed_device, auto_parsed_devices, selected_devices, selected_device, has_device_filter
    )

    if not has_device_filter:
        # all_docs (retrieve 결과) 또는 docs 사용
        docs_for_suggestion = state.get("all_docs") or state.get("docs") or state.get("display_docs") or []
        logger.info(
            "answer_node: docs_for_suggestion count=%d (from all_docs=%s, docs=%s, display_docs=%s)",
            len(docs_for_suggestion) if docs_for_suggestion else 0,
            len(state.get("all_docs", [])) if state.get("all_docs") else "None",
            len(state.get("docs", [])) if state.get("docs") else "None",
            len(state.get("display_docs", [])) if state.get("display_docs") else "None",
        )
        if docs_for_suggestion:
            # 디버깅: 처음 3개 문서의 device_name 확인
            for i, doc in enumerate(docs_for_suggestion[:3]):
                meta = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {}) if isinstance(doc, dict) else {}
                device_name = meta.get("device_name", "(없음)") if meta else "(메타없음)"
                logger.info("answer_node: doc[%d] device_name=%s", i, device_name)

            suggested_devices = _collect_suggested_devices(docs_for_suggestion)
            logger.info(
                "answer_node: suggested_devices=%d devices (top3: %s)",
                len(suggested_devices),
                [d["name"] for d in suggested_devices[:3]] if suggested_devices else []
            )
            events.append(
                f"answer_node: suggested_devices={len(suggested_devices)} (top3: {[d['name'] for d in suggested_devices[:3]] if suggested_devices else []})"
            )
        else:
            logger.info("answer_node: no docs for suggestion (all_docs and docs are empty)")
            events.append("answer_node: no docs for suggestion")
    else:
        logger.info("answer_node: skipping device suggestion (device filter active)")
        events.append("answer_node: skip suggested_devices (device filter active)")

    _trace_log("answer_node", state, "EXIT", {
        "answer_chars": len(answer) if answer else 0,
        "answer_preview": (answer[:200] + "...") if answer and len(answer) > 200 else answer,
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "suggested_devices_count": len(suggested_devices) if suggested_devices else 0,
    })

    return {
        "answer": answer,
        "reasoning": reasoning,
        "suggested_devices": suggested_devices,
        "_events": events,
    }


def _detect_chat_type(query: str) -> str:
    """Detect the type of chat query for appropriate response."""
    q = (query or "").strip().lower()

    # Identity patterns
    identity_patterns = [
        r"\bwho are you\b", r"\bwhat are you\b", r"\bwhat can you do\b",
        r"\bwhat is your name\b", r"\bwhat'?s your name\b",
        r"너는?\s*누구", r"넌\s*누구", r"당신은?\s*누구",
        r"너는?\s*뭘\s*할\s*수",
    ]
    if any(re.search(pat, q) for pat in identity_patterns):
        return "identity"

    # Greeting patterns
    greeting_patterns = [
        r"\bhello\b", r"\bhi\b", r"\bhey\b",
        r"^안녕", r"^ㅎㅇ", r"^하이", r"^헬로",
    ]
    if any(re.search(pat, q) for pat in greeting_patterns):
        return "greeting"

    # Thanks patterns
    thanks_patterns = [
        r"\bthanks?\b", r"\bthank you\b",
        r"고마워", r"고맙습니다", r"감사합니다", r"감사해", r"땡큐", r"ㄱㅅ",
    ]
    if any(re.search(pat, q) for pat in thanks_patterns):
        return "thanks"

    # Farewell patterns
    farewell_patterns = [
        r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b",
        r"잘\s*가", r"안녕히", r"바이", r"다음에\s*(봐|봬)",
    ]
    if any(re.search(pat, q) for pat in farewell_patterns):
        return "farewell"

    return "general"


def chat_answer_node(state: AgentState) -> Dict[str, Any]:
    """Answer small-talk / agent identity questions without retrieval."""
    lang = state.get("detected_language", "ko")
    query = state.get("query", "")
    chat_type = _detect_chat_type(query)

    # Language-specific responses by chat type
    responses = {
        "identity": {
            "ko": (
                "저는 반도체 장비 문서 기반 RAG 어시스턴트입니다. "
                "알람/에러 원인, 트러블슈팅 절차, 설치/셋업 방법, 장비별 문서 내용을 물어보시면 "
                "가능한 경우 근거와 함께 답변하겠습니다."
            ),
            "en": (
                "I am a document-grounded RAG assistant for semiconductor equipment. "
                "Ask me about alarms, troubleshooting steps, setup/installation procedures, "
                "or device-specific documentation, and I will cite sources when available."
            ),
            "ja": (
                "私は半導体装置向けの文書根拠型RAGアシスタントです。"
                "アラーム、トラブルシューティング手順、セットアップ/設置手順、"
                "装置別ドキュメントについて質問してください。根拠があれば出典も示します。"
            ),
        },
        "greeting": {
            "ko": "안녕하세요! 무엇을 도와드릴까요?",
            "en": "Hello! How can I help you today?",
            "ja": "こんにちは！何かお手伝いできることはありますか？",
        },
        "thanks": {
            "ko": "도움이 되어 기쁩니다. 다른 질문이 있으시면 말씀해주세요!",
            "en": "You're welcome! Let me know if you have any other questions.",
            "ja": "お役に立てて嬉しいです。他にご質問があればお知らせください！",
        },
        "farewell": {
            "ko": "다음에 또 질문해주세요. 좋은 하루 되세요!",
            "en": "Feel free to ask anytime. Have a great day!",
            "ja": "またいつでもご質問ください。良い一日を！",
        },
        "general": {
            "ko": "네, 무엇을 도와드릴까요?",
            "en": "Yes, how can I help you?",
            "ja": "はい、何かお手伝いできますか？",
        },
    }

    answer = responses.get(chat_type, responses["general"]).get(lang, responses[chat_type]["ko"])

    # Provide a faithful judge result to avoid retry loops.
    judge = {"faithful": True, "issues": [], "hint": f"chat_query:{chat_type}"}
    return {
        "answer": answer,
        "chat_type": chat_type,
        "judge": judge,
        "docs": [],
        "display_docs": [],
        "ref_json": [],
        "answer_ref_json": [],
    }


def history_answer_node(state: AgentState, *, llm: BaseLLM) -> Dict[str, Any]:
    """이전 대화 summary 기반 답변 (간단한 후속 질문용).

    route_node에서 history_intent=True이고 doc_ids가 없거나 summary로 충분할 때 호출.
    직전 assistant 응답의 summary를 기반으로 후속 질문에 답변.

    Returns:
        - answer: 생성된 답변
        - judge: {faithful: True} (history 기반이므로 검증 불필요)
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    lang = state.get("detected_language", "ko")

    # 직전 assistant summary 가져오기
    last_summary = _get_last_assistant_summary(chat_history)

    if not last_summary:
        # summary 없으면 fallback 응답
        fallback = {
            "ko": "이전 대화 내용을 찾을 수 없습니다. 새로운 질문을 해주세요.",
            "en": "I couldn't find the previous conversation. Please ask a new question.",
            "ja": "以前の会話が見つかりませんでした。新しい質問をしてください。",
        }
        return {
            "answer": fallback.get(lang, fallback["ko"]),
            "judge": {"faithful": True, "issues": [], "hint": "history_answer:no_summary"},
            "docs": [],
            "display_docs": [],
            "ref_json": [],
            "answer_ref_json": [],
        }

    # LLM으로 history 기반 답변 생성
    system_prompt = """당신은 이전 대화 내용을 기반으로 후속 질문에 답변하는 어시스턴트입니다.

규칙:
1. 이전 답변에서 제공된 정보만 사용하여 답변하세요
2. 이전 답변에 없는 내용은 "이전 대화에서 해당 내용을 찾을 수 없습니다"라고 말하세요
3. 답변은 간결하고 명확하게 작성하세요
4. 사용자가 요청한 언어로 답변하세요"""

    user_prompt = f"""이전 답변 내용:
{last_summary}

사용자 후속 질문: {query}

위 이전 답변 내용을 바탕으로 사용자의 질문에 답변해주세요."""

    try:
        answer = _invoke_llm(llm, system_prompt, user_prompt)
        logger.info(
            "[history_answer_node] Generated answer from summary, query='%s', answer_len=%d",
            query[:50], len(answer)
        )
    except Exception as e:
        logger.error("[history_answer_node] LLM error: %s", e)
        answer = "답변 생성 중 오류가 발생했습니다."

    return {
        "answer": answer,
        "judge": {"faithful": True, "issues": [], "hint": "history_answer"},
        "docs": [],
        "display_docs": [],
        "ref_json": [],
        "answer_ref_json": [],
    }


MAX_JUDGE_REF_CHARS = 50000  # Truncate refs to avoid LLM context overflow


def judge_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    if state.get("is_chat_query"):
        _trace_log("judge_node", state, "SKIP", {"reason": "chat_query"})
        return {"judge": {"faithful": True, "issues": [], "hint": "chat_query"}}

    route = state["route"]
    ref_items = state.get("answer_ref_json") or state.get("ref_json", [])
    ref_text = ref_json_to_text(ref_items)
    answer = state.get('answer', '')
    query_for_judge = state.get("query_en") or state["query"]
    attempts = state.get("attempts", 0)

    _trace_log("judge_node", state, "ENTER", {
        "route": route,
        "attempt": attempts,
        "answer_chars": len(answer),
        "ref_chars": len(ref_text),
        "ref_count": len(ref_items),
    })

    # Truncate refs to avoid LLM context overflow
    truncated = False
    if len(ref_text) > MAX_JUDGE_REF_CHARS:
        ref_text = ref_text[:MAX_JUDGE_REF_CHARS] + "\n... (truncated)"
        truncated = True
        logger.info("[judge_node] ref_text truncated to %d chars", MAX_JUDGE_REF_CHARS)

    if route == "setup":
        sys = spec.judge_setup_sys
    elif route == "ts":
        sys = spec.judge_ts_sys
    else:
        sys = spec.judge_general_sys

    user = (
        f"질문: {query_for_judge}\n"
        f"답안: {answer}\n"
        f"증거(REFS): {ref_text}\n"
        "JSON 한 줄로 반환: {\"faithful\": bool, \"issues\": [...], \"hint\": \"...\"}"
    )
    logger.info("[judge_node] input lengths: system=%d, user=%d, answer=%d, refs=%d",
                len(sys), len(user), len(answer), len(ref_text))

    # Use _invoke_llm_with_reasoning to handle reasoning models properly
    # max_tokens increased to allow reasoning + JSON output
    raw, reasoning = _invoke_llm_with_reasoning(llm, sys, user, max_tokens=MAX_TOKENS_JUDGE)
    if not raw:
        logger.warning("[judge_node] LLM returned empty response! input_user_len=%d", len(user))
    logger.debug("[judge_node] raw LLM output: %s, reasoning: %s", raw, reasoning[:200] if reasoning else None)

    # Try to extract JSON from the output (handles reasoning text mixed with JSON)
    judge = _extract_json_from_text(raw)
    if judge is None:
        logger.warning("[judge_node] JSON extraction failed from raw output:\n%s", raw[:500])
        judge = {"faithful": False, "issues": ["parse_error"], "hint": "judge JSON parse failed"}

    # 불충실 판정 시 상세 로그
    faithful = judge.get("faithful", False)
    issues = judge.get("issues", [])
    hint = judge.get("hint", "")

    if not faithful:
        logger.warning(
            "[judge_node] UNFAITHFUL: issues=%s, hint='%s', query='%s'",
            issues,
            hint,
            (query_for_judge[:80] + "...") if len(query_for_judge) > 80 else query_for_judge,
        )

        # 상세 트레이스 로그: 불충실 판정 이유 분석
        _trace_log("judge_node", state, "UNFAITHFUL", {
            "faithful": False,
            "issues": issues,
            "hint": hint,
            "attempt": attempts,
            "route": route,
            "query_en": query_for_judge,
            "answer_preview": (answer[:300] + "...") if len(answer) > 300 else answer,
            "ref_truncated": truncated,
            "ref_chars": len(ref_text),
            "raw_llm_output": raw[:500] if raw else "(empty)",
        })
    else:
        logger.info("[judge_node] FAITHFUL: query='%s'", query_for_judge[:50])
        _trace_log("judge_node", state, "FAITHFUL", {
            "faithful": True,
            "hint": hint,
            "attempt": attempts,
        })

    return {"judge": judge}


def should_retry(state: AgentState) -> Literal["done", "retry", "retry_expand", "retry_mq", "human"]:
    """Determine retry strategy based on attempt count.

    Retry strategies:
    - 1st unfaithful (attempt 0→1): retry_expand - use more docs (5→10)
    - 2nd unfaithful (attempt 1→2): retry - refine queries
    - 3rd unfaithful (attempt 2→3): retry_mq - regenerate multi-query from scratch
    """
    judge = state.get("judge", {})
    faithful = bool(judge.get("faithful", False))
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 0)
    route = state.get("route")

    if state.get("retrieval_confirmed"):
        decision = "done"
        reason = "retrieval_confirmed"
    elif faithful:
        decision = "done"
        reason = "faithful"
    elif route == "doc_lookup":
        # doc_lookup은 재검색 없이 같은 문서로 재답변만 허용
        if attempts < max_attempts:
            decision = "retry_expand"
            reason = "doc_lookup_retry"
        else:
            decision = "done"
            reason = "doc_lookup_max_attempts"
    elif attempts < max_attempts:
        if attempts == 0:
            # 1st retry: expand more docs (5→10)
            decision = "retry_expand"
            reason = "1st_retry_expand_docs"
        elif attempts == 1:
            # 2nd retry: refine queries
            decision = "retry"
            reason = "2nd_retry_refine_queries"
        else:
            # 3rd+ retry: regenerate MQ from scratch
            decision = "retry_mq"
            reason = "3rd+_retry_regenerate_mq"
    elif state.get("_skip_human_review"):
        # HIL 비활성화 모드 (auto_parse 등)에서는 human_review 건너뛰기
        decision = "done"
        reason = "skip_human_review"
    else:
        decision = "human"
        reason = "need_human_review"

    _trace_log("should_retry", state, "DECISION", {
        "decision": decision,
        "reason": reason,
        "faithful": faithful,
        "attempts": attempts,
        "max_attempts": max_attempts,
        "route": route,
        "judge_issues": judge.get("issues", []),
        "judge_hint": judge.get("hint", ""),
    })

    return decision


def retry_bump_node(state: AgentState) -> Dict[str, Any]:
    """Increment attempt counter."""
    return {"attempts": int(state.get("attempts", 0)) + 1}


def retry_expand_node(state: AgentState) -> Dict[str, Any]:
    """1st retry strategy: increase expand_top_k and re-retrieve.

    Re-retrieves documents with expanded top_k for better coverage.
    """
    attempts = int(state.get("attempts", 0)) + 1
    logger.info("retry_expand_node: re-retrieving with expand_top_k=20 (attempt %d)", attempts)

    _trace_log("retry_expand_node", state, "RETRY", {
        "strategy": "expand_more",
        "attempt": attempts,
        "expand_top_k": 20,
        "prev_judge_issues": state.get("judge", {}).get("issues", []),
        "prev_judge_hint": state.get("judge", {}).get("hint", ""),
    })

    return {
        "attempts": attempts,
        "expand_top_k": 20,
        "retry_strategy": "expand_more",
    }


def retry_mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    """3rd+ retry strategy: regenerate multi-query from scratch.

    Clears previous MQ lists and triggers fresh MQ generation.
    """
    attempts = int(state.get("attempts", 0)) + 1
    logger.info("retry_mq_node: regenerating MQ from scratch (attempt %d)", attempts)

    _trace_log("retry_mq_node", state, "RETRY", {
        "strategy": "regenerate_mq",
        "attempt": attempts,
        "prev_mq_en": state.get("retrieval_mq_list", []),
        "prev_mq_ko": state.get("retrieval_mq_ko_list", []),
        "prev_judge_issues": state.get("judge", {}).get("issues", []),
        "prev_judge_hint": state.get("judge", {}).get("hint", ""),
    })

    # Clear previous MQ lists to force regeneration
    return {
        "attempts": attempts,
        "retry_strategy": "regenerate_mq",
        "setup_mq_list": [],
        "ts_mq_list": [],
        "general_mq_list": [],
        "setup_mq_ko_list": [],
        "ts_mq_ko_list": [],
        "general_mq_ko_list": [],
        "search_queries": [],
        "expand_top_k": 10,  # Keep expanded doc count
    }


def refine_queries_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    hint = state.get("judge", {}).get("hint", "")
    prev = state.get("search_queries", [])
    # Use English query for refining (for consistent LLM processing)
    query_en = state.get("query_en") or state["query"]
    query_ko = state.get("query_ko") or state["query"]
    route = state.get("route", "general")

    # Get previous MQ lists for Korean queries
    if route == "setup":
        mq_ko_list = state.get("setup_mq_ko_list", [])
    elif route == "ts":
        mq_ko_list = state.get("ts_mq_ko_list", [])
    else:
        mq_ko_list = state.get("general_mq_ko_list", [])

    # Generate refined English queries
    sys_en = (
        "Role: Search Query Refiner\n"
        "Given the original question, existing queries, and judge hint, generate 3 improved search queries.\n"
        "**IMPORTANT: Generate all queries in English.**\n"
        "Output: {\"queries\":[...]} in one line only."
    )
    user_en = (
        f"Original question: {query_en}\n"
        f"Previous queries: {json.dumps(prev, ensure_ascii=False)}\n"
        f"Judge hint: {hint}\n"
    )
    raw_en = _invoke_llm(llm, sys_en, user_en)
    queries_en = _parse_queries(raw_en)

    # Build English queries (3)
    english_queries: List[str] = []
    _fill_to_n(english_queries, queries_en, 3)
    if query_en:
        _fill_to_n(english_queries, [query_en], 3)
    english_queries = english_queries[:3]

    # Build Korean queries (3) - prefer existing KO MQ list
    ko_candidates = [q for q in mq_ko_list if _contains_korean(q)]
    if query_ko and _contains_korean(query_ko):
        ko_candidates = [query_ko] + ko_candidates
    korean_queries = _dedupe_queries(ko_candidates)[:3]

    # If not enough Korean queries, translate from English
    if len(korean_queries) < 3 and spec.translate is not None:
        def _translate(text: str) -> str:
            user = _format_prompt(spec.translate.user, {
                "query": text,
                "target_language": "Korean",
            })
            result = _invoke_llm(llm, spec.translate.system, user)
            return result.strip().strip('"').strip("'").strip() or text

        for q in english_queries:
            if len(korean_queries) >= 3:
                break
            if _contains_korean(q):
                continue
            translated = _translate(q)
            if translated and translated not in korean_queries:
                korean_queries.append(translated)
        korean_queries = korean_queries[:3]

    merged = english_queries + korean_queries
    logger.info("refine_queries_node: bilingual queries EN=%d, KO=%d", len(english_queries), len(korean_queries))
    return {"search_queries": merged}


def _parse_auto_parse_result(text: str) -> Dict[str, Any]:
    """Parse LLM output for auto-parsed devices, doc_types, and language."""
    result: Dict[str, Any] = {"devices": [], "doc_types": [], "language": None}
    t = text.strip()
    if not t:
        return result

    # Strip code fences
    t = re.sub(r"```[a-zA-Z]*", "", t).strip()

    # Try to parse JSON
    try:
        obj_match = re.search(r"\{.*\}", t, flags=re.S)
        if obj_match:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict):
                devices = obj.get("devices")
                doc_types = obj.get("doc_types")
                language = obj.get("language")

                # Backward compat: allow single fields
                if devices is None and "device" in obj:
                    devices = [obj.get("device")]
                if doc_types is None and "doc_type" in obj:
                    doc_types = [obj.get("doc_type")]

                parsed_devices: List[str] = []
                if isinstance(devices, list):
                    for d in devices:
                        s = str(d).strip() if d is not None else ""
                        if s and s.lower() != "null":
                            parsed_devices.append(s)
                elif devices:
                    s = str(devices).strip()
                    if s and s.lower() != "null":
                        parsed_devices.append(s)

                parsed_doc_types: List[str] = []
                if isinstance(doc_types, list):
                    for d in doc_types:
                        s = str(d).strip() if d is not None else ""
                        if s and s.lower() != "null":
                            parsed_doc_types.append(s)
                elif doc_types:
                    s = str(doc_types).strip()
                    if s and s.lower() != "null":
                        parsed_doc_types.append(s)

                # Parse language
                parsed_language: Optional[str] = None
                if language:
                    lang_str = str(language).strip().lower()
                    if lang_str in ("ko", "en", "ja"):
                        parsed_language = lang_str

                result["devices"] = _dedupe_queries(parsed_devices)[:2]
                result["doc_types"] = _dedupe_queries(parsed_doc_types)[:2]
                result["language"] = parsed_language
    except Exception:
        pass

    return result


def _compact_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", str(value).lower())


def _strip_parenthetical(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"[\(\[\{].*?[\)\]\}]", "", str(value))


def _device_aliases_from_name(device_name: str) -> list[str]:
    """Generate normalized aliases for device name matching.

    Examples:
    - 'SUPRA XP' -> ['supraxp']
    - 'SUPRA-XP' -> ['supraxp']
    - 'SUPRA XP SEM' -> ['supraxpsem', 'supraxp'] (prefix alias)
    """
    cleaned = _strip_parenthetical(device_name or "")
    tokens = [t for t in re.split(r"[\s\-_./]+", cleaned) if t]
    aliases: set[str] = set()

    def _add_alias(text: str) -> None:
        normalized = _compact_text(text)
        if len(normalized) >= 4:
            aliases.add(normalized)

    _add_alias(cleaned)
    if len(tokens) >= 2:
        _add_alias("".join(tokens[:2]))
    if len(tokens) >= 3:
        _add_alias("".join(tokens[:3]))

    return sorted(aliases)


def _filter_devices_by_query(
    devices: List[str],
    device_names: List[str],
    query: str,
) -> List[str]:
    if not devices or not device_names:
        return []
    device_map = {str(name).strip().lower(): str(name).strip() for name in device_names if str(name).strip()}
    query_compact = _compact_text(query)
    filtered: List[str] = []
    for d in devices:
        key = str(d).strip().lower()
        canonical = device_map.get(key)
        if not canonical:
            continue
        aliases = _device_aliases_from_name(canonical)
        if any(alias in query_compact for alias in aliases):
            filtered.append(canonical)
    return _dedupe_queries(filtered)[:2]


def _extract_devices_from_query(device_names: List[str], query: str) -> List[str]:
    if not device_names or not query:
        return []
    query_compact = _compact_text(query)
    matches: List[str] = []
    for name in device_names:
        cleaned = str(name).strip()
        if not cleaned:
            continue
        aliases = _device_aliases_from_name(cleaned)
        if any(alias in query_compact for alias in aliases):
            matches.append(cleaned)
    return _dedupe_queries(matches)[:2]


def _extract_device_with_llm(
    llm: BaseLLM,
    device_names: List[str],
    query: str,
    *,
    prompt_template: Optional[PromptTemplate] = None,
    top_k: int = 20,
) -> Optional[str]:
    """LLM을 사용하여 쿼리에서 장비명을 추출합니다.

    사용자가 다양한 형태로 장비명을 입력할 수 있습니다:
    - "supra XP", "SUPRA XP", "supraxp", "수프라XP" 등

    Args:
        llm: LLM 인스턴스
        device_names: 유효한 장비명 목록
        query: 사용자 쿼리
        top_k: 프롬프트에 포함할 장비 수 (문서 수 기준 상위)

    Returns:
        매칭된 장비명 또는 None
    """
    if not device_names or not query:
        return None

    template = prompt_template if isinstance(prompt_template, PromptTemplate) else None
    if template is None:
        logger.warning("auto_parse_device prompt missing; skipping LLM device extraction")
        return None

    # 상위 장비만 선택 (프롬프트 길이 제한)
    device_list = device_names[:top_k]
    device_list_str = "\n".join(f"- {name}" for name in device_list)

    user = _format_prompt(template.user, {
        "sys.query": query,
        "sys.devices": device_list_str,
    })

    try:
        result = _invoke_llm(llm, template.system, user)

        # 결과 정리
        result = result.strip('"\'').strip()

        # None 체크
        if result.lower() == "none" or not result:
            return None

        # 유효한 장비명인지 확인
        for name in device_names:
            if result.lower() == name.lower():
                return name

        # 부분 매칭 시도
        for name in device_names:
            if result.lower() in name.lower() or name.lower() in result.lower():
                return name

        return None
    except Exception as e:
        logger.warning("LLM device extraction failed: %s", e)
        return None


def _extract_doc_types_from_query(query: str) -> List[str]:
    """Extract doc type keys based on predefined keyword variants."""
    if not query:
        return []
    normalized_query = normalize_doc_type(query)
    matched: List[str] = []
    for group_name, variants in DOC_TYPE_GROUPS.items():
        for variant in variants:
            normalized_variant = normalize_doc_type(variant)
            if not normalized_variant:
                continue
            # For short ASCII tokens like 'ts', enforce word boundary
            if normalized_variant.isalnum() and len(normalized_variant) <= 3:
                pattern = r"\b" + re.escape(normalized_variant) + r"\b"
                if re.search(pattern, normalized_query, flags=re.IGNORECASE):
                    matched.append(group_name)
                    break
            else:
                if normalized_variant in normalized_query:
                    matched.append(group_name)
                    break
    return _dedupe_queries(matched)[:2]


def _detect_language_rule_based(text: str) -> str:
    """Rule-based language detection.

    Detects language based on character scripts:
    - Korean (Hangul): ko
    - Japanese (Hiragana/Katakana): ja
    - Otherwise: en

    Note: Domain terms (device names, technical terms) may be in English
    even in Korean/Japanese sentences. This function prioritizes
    Korean/Japanese characters over English.

    Returns:
        "ko", "en", or "ja"
    """
    if not text:
        return "ko"  # Default to Korean

    # Count Korean characters (Hangul)
    korean_chars = len(re.findall(r'[가-힣]', text))

    # Count Japanese characters (Hiragana + Katakana)
    # Hiragana: \u3040-\u309f, Katakana: \u30a0-\u30ff
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))

    # Prioritize Korean if any Korean characters exist
    if korean_chars > 0:
        return "ko"

    # Then Japanese if any Japanese characters exist
    if japanese_chars > 0:
        return "ja"

    # Default to English
    return "en"


def auto_parse_node(
    state: AgentState,
    *,
    llm: BaseLLM,
    spec: PromptSpec,
    device_names: List[str],
    doc_type_names: List[str],
) -> Dict[str, Any]:
    """Auto-parse device, doc_type from user query using LLM.

    translate_node 이후에 실행되므로 query_en을 사용하여 장비 파싱.
    언어 정보는 translate_node에서 이미 감지됨.

    Args:
        state: Agent state with query, query_en (from translate_node).
        llm: LLM instance for generation.
        spec: Prompt specification (must have auto_parse template).
        device_names: List of available device names.
        doc_type_names: List of available doc type names.

    Returns:
        State update with auto_parsed_device, auto_parsed_doc_type, and auto_parse_message.
    """
    query = state["query"]
    # translate_node에서 번역된 영어 쿼리 사용 (장비명 파싱에 유리)
    query_en = state.get("query_en", query)
    # 언어는 translate_node에서 이미 감지됨
    detected_language = state.get("detected_language") or _detect_language_rule_based(query)

    # skip_auto_parse가 True이면 건너뛰기 (filter_devices 재검색 시)
    if state.get("skip_auto_parse"):
        selected_devices = state.get("selected_devices", [])
        logger.info("auto_parse_node: SKIPPED (skip_auto_parse=True), using selected_devices=%s", selected_devices)
        return {
            "auto_parsed_device": selected_devices[0] if selected_devices else None,
            "auto_parsed_devices": selected_devices if selected_devices else None,
            "auto_parse_message": f"필터 적용: {', '.join(selected_devices)}" if selected_devices else "",
        }

    # 1. 규칙 기반 파싱 - query_en 사용 (장비명은 영어)
    devices = _extract_devices_from_query(device_names, query_en)
    # doc_types는 원본 쿼리에서도 추출 (한국어 문서 종류명)
    doc_types = _extract_doc_types_from_query(query) or _extract_doc_types_from_query(query_en)

    # 2. 규칙 기반으로 장비를 찾지 못하면 LLM 사용 (영어 쿼리로)
    if not devices and device_names:
        llm_device = _extract_device_with_llm(
            llm,
            device_names,
            query_en,
            prompt_template=spec.auto_parse_device,
        )
        if llm_device:
            devices = [llm_device]
            logger.info("auto_parse_node: LLM detected device=%s from query_en", llm_device)

    logger.info("auto_parse_node: devices=%s, doc_types=%s, language=%s", devices, doc_types, detected_language)

    # Build display message based on detected language
    # Language display labels
    lang_labels = {"ko": "kor", "en": "eng", "ja": "jap"}
    lang_label = lang_labels.get(detected_language, "kor")

    message_parts: List[str] = []
    if detected_language == "en":
        if devices:
            message_parts.append(f"Device: {', '.join(devices)}")
        if doc_types:
            message_parts.append(f"Doc type: {', '.join(doc_types)}")
        message_parts.append(f"lang: {lang_label}")
        auto_parse_message = f"Parsed - {', '.join(message_parts)}"
    elif detected_language == "ja":
        if devices:
            message_parts.append(f"機器: {', '.join(devices)}")
        if doc_types:
            message_parts.append(f"文書: {', '.join(doc_types)}")
        message_parts.append(f"lang: {lang_label}")
        auto_parse_message = f"パース結果 - {', '.join(message_parts)}"
    else:  # ko (default)
        if devices:
            message_parts.append(f"장비: {', '.join(devices)}")
        if doc_types:
            message_parts.append(f"문서: {', '.join(doc_types)}")
        message_parts.append(f"lang: {lang_label}")
        auto_parse_message = f"파싱 결과 - {', '.join(message_parts)}"

    # Set selected_devices and selected_doc_types for downstream nodes
    selected_devices = devices[:2]
    selected_doc_types = doc_types[:2]

    # 언어는 항상 반환 (다른 파싱 결과가 없어도)
    result: Dict[str, Any] = {
        "detected_language": detected_language,
    }

    if devices or doc_types:
        # 파싱 결과가 있으면 추가
        result.update({
            "auto_parsed_device": devices[0] if devices else None,
            "auto_parsed_doc_type": doc_types[0] if doc_types else None,
            "auto_parsed_devices": devices,
            "auto_parsed_doc_types": doc_types,
            "auto_parse_message": auto_parse_message,
            "selected_devices": selected_devices,
            "selected_doc_types": selected_doc_types,
            "device_selection_skipped": not bool(devices),
            "doc_type_selection_skipped": not bool(doc_types),
            "_events": [
                {
                    "type": "auto_parse",
                    "device": devices[0] if devices else None,
                    "doc_type": doc_types[0] if doc_types else None,
                    "devices": devices,
                    "doc_types": doc_types,
                    "language": detected_language,
                    "message": auto_parse_message,
                }
            ] if auto_parse_message else [],
        })
    else:
        # 이전 턴의 파싱 결과가 남지 않도록 명시적으로 초기화
        result.update({
            "auto_parsed_device": None,
            "auto_parsed_doc_type": None,
            "auto_parsed_devices": None,
            "auto_parsed_doc_types": None,
            "auto_parse_message": None,
            "selected_devices": [],
            "selected_doc_types": [],
            "device_selection_skipped": True,
            "doc_type_selection_skipped": True,
            "_events": [
                {
                    "type": "auto_parse",
                    "device": None,
                    "doc_type": None,
                    "devices": [],
                    "doc_types": [],
                    "language": detected_language,
                    "message": "auto_parse: no device/doc_type detected",
                }
            ],
        })

    return result


def translate_node(
    state: AgentState,
    *,
    llm: BaseLLM,
    spec: PromptSpec,
) -> Dict[str, Any]:
    """Translate query to English and Korean for better retrieval coverage.

    - If detected_language is 'en': query_en = query, translate to Korean
    - If detected_language is 'ko': query_ko = query, translate to English
    - Otherwise (ja, etc.): translate to both

    Also detects language if not already set.
    """
    query = state["query"]
    _trace_log("translate_node", state, "ENTER", {"query_full": query})

    # 언어가 아직 감지되지 않았으면 여기서 감지
    detected_language = state.get("detected_language")
    if not detected_language:
        detected_language = _detect_language_rule_based(query)
        logger.info("translate_node: detected language = %s", detected_language)

    # Check if translate prompt is available
    if spec.translate is None:
        logger.warning("translate_node: no translate prompt, using original query only")
        return {
            "query_en": query,
            "query_ko": query,
        }

    def _translate(text: str, target_lang: str) -> str:
        """Translate text to target language using LLM."""
        target_name = {"en": "English", "ko": "Korean"}.get(target_lang, target_lang)
        system = spec.translate.system
        user = _format_prompt(spec.translate.user, {
            "query": text,
            "target_language": target_name,
        })
        result = _invoke_llm(llm, system, user)
        logger.debug("translate_node: raw LLM output for %s: %s", target_lang, result)

        # Extract content after "Translation:" if present
        if "translation:" in result.lower():
            parts = re.split(r"translation\s*:", result, flags=re.IGNORECASE)
            if len(parts) > 1:
                result = parts[-1]

        # Clean up result (remove quotes, extra whitespace, newlines)
        result = result.strip().strip('"').strip("'").strip()
        result = result.split('\n')[0].strip()  # Take first line only

        # Detect meta-explanations (LLM outputting instructions instead of translation)
        meta_patterns = [
            "we need to", "let me", "i will", "i'll", "here is", "here's",
            "the translation", "translated", "translating", "sure", "of course",
            "certainly", "below is", "following is",
        ]
        result_lower = result.lower()
        if any(result_lower.startswith(p) for p in meta_patterns):
            logger.warning("translate_node: LLM returned meta-explanation: %s", result[:100])
            return text  # Fall back to original

        # If result is too long (likely explanation), fallback
        if len(result) > len(text) * 3:
            logger.warning("translate_node: result too long, likely explanation: %d vs %d", len(result), len(text))
            return text

        return result if result else text

    query_en = query
    query_ko = query

    if detected_language == "en":
        # Already English, translate to Korean
        query_ko = _translate(query, "ko")
        logger.info("translate_node: en->ko: %s", query_ko)
    elif detected_language == "ko":
        # Already Korean, translate to English
        query_en = _translate(query, "en")
        logger.info("translate_node: ko->en: %s", query_en)
    else:
        # Japanese or other - translate to both
        query_en = _translate(query, "en")
        query_ko = _translate(query, "ko")
        logger.info("translate_node: %s->en: %s", detected_language, query_en)
        logger.info("translate_node: %s->ko: %s", detected_language, query_ko)

    _trace_log("translate_node", state, "EXIT", {
        "detected_language": detected_language,
        "query_en": query_en,
        "query_ko": query_ko,
    })

    return {
        "query_en": query_en,
        "query_ko": query_ko,
        "detected_language": detected_language,  # 언어 정보 전달
        "_events": [
            {
                "type": "translate",
                "original": query,
                "query_en": query_en,
                "query_ko": query_ko,
                "detected_language": detected_language,
            }
        ],
    }


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
        device_fetcher: Callable that returns devices (and optionally doc types).
            Devices/doc types should have 'name' and 'doc_count'.

    Returns:
        Command to proceed to 'mq' node with selected devices.
    """
    # Fetch available devices if fetcher provided
    available_devices: List[Dict[str, Any]] = []
    available_doc_types: List[Dict[str, Any]] = []
    if device_fetcher is not None:
        try:
            fetched = device_fetcher()
            if isinstance(fetched, dict):
                available_devices = fetched.get("devices", []) or []
                available_doc_types = fetched.get("doc_types", []) or []
            elif isinstance(fetched, list):
                available_devices = fetched
            else:
                available_devices = []
        except Exception as e:
            logger.warning(f"Failed to fetch devices: {e}")
            available_devices = []

    if not available_devices and not available_doc_types:
        # No selection options available, skip selection
        logger.info("device_selection_node: no devices/doc_types available, skipping")
        return Command(
            goto="mq",
            update={
                "available_devices": [],
                "selected_devices": [],
                "device_selection_skipped": True,
                "available_doc_types": [],
                "selected_doc_types": [],
                "doc_type_selection_skipped": True,
            }
        )

    payload = {
        "type": "device_selection",
        "question": state["query"],
        "route": state.get("route"),
        "devices": available_devices,
        "device_count": len(available_devices),
        "doc_types": available_doc_types,
        "doc_type_count": len(available_doc_types),
        "instruction": (
            "검색할 기기/문서 종류를 선택하세요 (다중 선택 가능).\n"
            "- 기기/문서 각각 최소 1개 선택 필요\n"
            "- 전체 기기: 모든 기기 선택\n"
            "- 전체 문서: 모든 문서 종류 선택"
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
                "available_doc_types": available_doc_types,
                "selected_doc_types": [],
                "doc_type_selection_skipped": True,
            }
        )

    # Parse selected devices (can be single string, list, or dict with selected_devices)
    selected_devices: List[str] = []
    selected_doc_types: List[str] = []

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

        doc_types_from_dict = decision.get("selected_doc_types") or decision.get("doc_types") or []
        if isinstance(doc_types_from_dict, list):
            selected_doc_types = [str(d).strip() for d in doc_types_from_dict if d]
        elif isinstance(doc_types_from_dict, str):
            selected_doc_types = [doc_types_from_dict.strip()]
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

    valid_doc_types = {d.get("name") for d in available_doc_types if d.get("name")}
    selected_doc_types = [d for d in selected_doc_types if d in valid_doc_types]

    if selected_doc_types:
        logger.info(f"device_selection_node: validated selected_doc_types={selected_doc_types}")

    return Command(
        goto="mq",
        update={
            "available_devices": available_devices,
            "selected_devices": selected_devices,
            "device_selection_skipped": len(selected_devices) == 0,
            "available_doc_types": available_doc_types,
            "selected_doc_types": selected_doc_types,
            "doc_type_selection_skipped": len(selected_doc_types) == 0,
        }
    )


__all__ = [
    "AgentState",
    "ChatHistoryEntry",
    "Route",
    "Gate",
    "PromptSpec",
    "SearchServiceRetriever",
    "Retriever",
    "load_prompt_spec",
    # Node helpers
    "route_node",
    "doc_lookup_node",
    "mq_node",
    "st_gate_node",
    "st_mq_node",
    "retrieve_node",
    "expand_related_docs_node",
    "ask_user_after_retrieve_node",
    "answer_node",
    "chat_answer_node",
    "judge_node",
    "should_retry",
    "retry_bump_node",
    "retry_expand_node",
    "retry_mq_node",
    "refine_queries_node",
    "human_review_node",
    "device_selection_node",
    "auto_parse_node",
]
