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

from backend.domain.doc_type_mapping import DOC_TYPE_GROUPS, expand_doc_type_selection, normalize_doc_type
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
    auto_parse: Optional[PromptTemplate] = None  # Auto-parse device/doc_type
    translate: Optional[PromptTemplate] = None  # Translate query to en/ko
    # Language-specific answer prompts
    setup_ans_en: Optional[PromptTemplate] = None
    setup_ans_ja: Optional[PromptTemplate] = None
    ts_ans_en: Optional[PromptTemplate] = None
    ts_ans_ja: Optional[PromptTemplate] = None
    general_ans_en: Optional[PromptTemplate] = None
    general_ans_ja: Optional[PromptTemplate] = None


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


def _try_load_prompt(name: str, version: str) -> Optional[PromptTemplate]:
    """Try to load a prompt template, return None if not found."""
    try:
        return load_prompt_template(name, version)
    except FileNotFoundError:
        return None


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

    # Try to load optional prompts
    auto_parse = _try_load_prompt("auto_parse", version)
    translate = _try_load_prompt("translate", version)

    # Load language-specific answer prompts (optional)
    setup_ans_en = _try_load_prompt("setup_ans_en", version)
    setup_ans_ja = _try_load_prompt("setup_ans_ja", version)
    ts_ans_en = _try_load_prompt("ts_ans_en", version)
    ts_ans_ja = _try_load_prompt("ts_ans_ja", version)
    general_ans_en = _try_load_prompt("general_ans_en", version)
    general_ans_ja = _try_load_prompt("general_ans_ja", version)

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
        auto_parse=auto_parse,
        translate=translate,
        setup_ans_en=setup_ans_en,
        setup_ans_ja=setup_ans_ja,
        ts_ans_en=ts_ans_en,
        ts_ans_ja=ts_ans_ja,
        general_ans_en=general_ans_en,
        general_ans_ja=general_ans_ja,
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
        return self.search_service.search(query, **search_kwargs)


# -----------------------------
# 4) LLM helpers
# -----------------------------
# 노드 타입별 max_tokens 설정
MAX_TOKENS_CLASSIFICATION = 256   # 라우팅/분류용 (짧은 응답)
MAX_TOKENS_ANSWER = 4096         # 답변 생성용
MAX_REF_CHARS_REVIEW = 200       # 검색 결과 리뷰용
MAX_REF_CHARS_ANSWER = 1200      # 답변 생성용
RELATED_PAGE_WINDOW = 2          # 인접 페이지 범위 (±N)
DOC_TYPES_SAME_DOC = {"gcb", "myservice"}
EXPAND_TOP_K = 5                 # 확장 대상 최대 개수 (rerank 상위)


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
    max_chars: int = MAX_REF_CHARS_REVIEW,
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
# 6) Graph node helpers
# -----------------------------
def route_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    # Use English query for routing (after translation)
    query = state.get("query_en") or state["query"]
    user = _format_prompt(spec.router.user, {"sys.query": query})
    route = _parse_route(_invoke_llm(llm, spec.router.system, user))
    logger.info("route_node: query=%s..., route=%s", query[:50] if query else None, route)
    return {"route": route}


def mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    # Generate MQ in both English and Korean for bilingual retrieval
    query_en = state.get("query_en") or state["query"]
    query_ko = state.get("query_ko") or state["query"]
    logger.info("mq_node: bilingual - EN=%s..., KO=%s...",
                query_en[:40] if query_en else None,
                query_ko[:40] if query_ko else None)

    if state.get("skip_mq") and state.get("search_queries"):
        logger.info("mq_node: search_queries override provided, skipping MQ generation")
        return {}

    setup_mq_list: List[str] = []
    ts_mq_list: List[str] = []
    general_mq_list: List[str] = []

    # MQ generation needs more tokens than classification
    mq_kwargs = {"max_tokens": 4096}

    def _generate_mq_bilingual(spec_template) -> tuple[List[str], List[str]]:
        """Generate MQ in both English and Korean."""
        # English MQ - add explicit language instruction
        system_en = spec_template.system + "\n\n**IMPORTANT: Generate all queries in English.**"
        user_en = _format_prompt(spec_template.user, {"sys.query": query_en})
        raw_en = _invoke_llm(llm, system_en, user_en, **mq_kwargs)
        mq_en = _parse_queries(raw_en)
        logger.info("mq_node(%s/en): %d queries: %s", route, len(mq_en), mq_en)

        # Korean MQ - add explicit Korean language instruction
        system_ko = spec_template.system + "\n\n**중요: 모든 검색어를 반드시 한국어로 생성하세요. Generate all queries in Korean.**"
        user_ko = _format_prompt(spec_template.user, {"sys.query": query_ko})
        raw_ko = _invoke_llm(llm, system_ko, user_ko, **mq_kwargs)
        mq_ko = _parse_queries(raw_ko)
        logger.info("mq_node(%s/ko): %d queries: %s", route, len(mq_ko), mq_ko)

        return mq_en, mq_ko

    setup_mq_ko_list: List[str] = []
    ts_mq_ko_list: List[str] = []
    general_mq_ko_list: List[str] = []

    if route == "setup":
        setup_mq_list, setup_mq_ko_list = _generate_mq_bilingual(spec.setup_mq)
    elif route == "ts":
        ts_mq_list, ts_mq_ko_list = _generate_mq_bilingual(spec.ts_mq)
    else:
        general_mq_list, general_mq_ko_list = _generate_mq_bilingual(spec.general_mq)

    logger.info(
        "mq_node: total - setup(en=%d,ko=%d), ts(en=%d,ko=%d), general(en=%d,ko=%d)",
        len(setup_mq_list), len(setup_mq_ko_list),
        len(ts_mq_list), len(ts_mq_ko_list),
        len(general_mq_list), len(general_mq_ko_list),
    )

    return {
        "setup_mq_list": setup_mq_list,
        "ts_mq_list": ts_mq_list,
        "general_mq_list": general_mq_list,
        "setup_mq_ko_list": setup_mq_ko_list,
        "ts_mq_ko_list": ts_mq_ko_list,
        "general_mq_ko_list": general_mq_ko_list,
    }


def st_gate_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    # Convert lists back to text for the prompt
    setup_mq_list = state.get("setup_mq_list", [])
    ts_mq_list = state.get("ts_mq_list", [])
    general_mq_list = state.get("general_mq_list", [])

    # Use English query for processing
    q = state.get("query_en") or state["query"]
    mapping = {
        "sys.query": q,
        "setup_mq": "\n".join(setup_mq_list),
        "ts_mq": "\n".join(ts_mq_list),
        "general_mq": "\n".join(general_mq_list),
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
    # Get MQ lists from previous node
    setup_mq_list = state.get("setup_mq_list", [])
    ts_mq_list = state.get("ts_mq_list", [])
    general_mq_list = state.get("general_mq_list", [])

    # Use English query for processing
    q_en = state.get("query_en") or state["query"]
    # Get Korean query for bilingual search
    q_ko = state.get("query_ko")
    route = state.get("route", "general")
    if route == "setup":
        mq_en_list = setup_mq_list
        mq_ko_list = state.get("setup_mq_ko_list", [])
    elif route == "ts":
        mq_en_list = ts_mq_list
        mq_ko_list = state.get("ts_mq_ko_list", [])
    else:
        mq_en_list = general_mq_list
        mq_ko_list = state.get("general_mq_ko_list", [])

    if state.get("skip_mq") and state.get("search_queries"):
        provided = [str(q).strip() for q in state.get("search_queries", []) if str(q).strip()]
        if provided:
            return {"search_queries": _dedupe_queries(provided)}
        return {"search_queries": [q_en] if q_en else []}

    mapping = {
        "sys.query": q_en,
        "setup_mq": "\n".join(setup_mq_list),
        "ts_mq": "\n".join(ts_mq_list),
        "general_mq": "\n".join(general_mq_list),
        "st_gate": state.get("st_gate", "no_st"),
    }
    user = _format_prompt(spec.st_mq.user, mapping)
    raw = _invoke_llm(llm, spec.st_mq.system, user)
    logger.info("st_mq_node: raw output=%s", raw)
    queries = _parse_queries(raw)

    # Filter out garbage queries (prompt label leaks, too short, etc.)
    queries = [q for q in queries if not _is_garbage_query(q)]

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
    original_query = state["query"]

    # search_queries from st_mq_node already contains EN+KO queries
    # No need to add bilingual queries here
    query_en = state.get("query_en")

    selected_device_set = {
        _normalize_device_name(d) for d in selected_devices if _normalize_device_name(d)
    }
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
        for q in all_queries:
            device_docs = retriever.retrieve(
                q,
                top_k=candidate_k,
                device_names=selected_devices,
                doc_types=selected_doc_type_filters,
            )
            _add_docs(device_docs, filter_devices=True)

        device_filtered_count = len(all_docs)
        logger.info("retrieve_node: device-filtered search found %d docs", device_filtered_count)

        # Search 2: General search without device filter
        for q in all_queries:
            general_docs = retriever.retrieve(
                q,
                top_k=candidate_k,
                doc_types=selected_doc_type_filters,
            )
            _add_docs(general_docs, filter_devices=False)

        logger.info("retrieve_node: after general search, total %d docs", len(all_docs))

    else:
        # No device selection: search without filter
        logger.info("retrieve_node: general search (no device filter), queries=%d", len(all_queries))
        for q in all_queries:
            docs = retriever.retrieve(
                q,
                top_k=candidate_k,
                doc_types=selected_doc_type_filters,
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
    max_ref_chars: int = MAX_REF_CHARS_ANSWER,
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
    """Get the appropriate answer template based on route and language."""
    # Default templates (Korean or language-agnostic)
    default_templates = {
        "setup": spec.setup_ans,
        "ts": spec.ts_ans,
        "general": spec.general_ans,
    }

    # Language-specific templates
    lang_templates = {
        "en": {
            "setup": spec.setup_ans_en,
            "ts": spec.ts_ans_en,
            "general": spec.general_ans_en,
        },
        "ja": {
            "setup": spec.setup_ans_ja,
            "ts": spec.ts_ans_ja,
            "general": spec.general_ans_ja,
        },
    }

    # Try to get language-specific template
    if language and language in lang_templates:
        lang_tmpl = lang_templates[language].get(route)
        if lang_tmpl is not None:
            return lang_tmpl

    # Fallback to default template
    return default_templates.get(route, spec.general_ans)


def answer_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    detected_language = state.get("detected_language", "ko")
    ref_items = state.get("answer_ref_json") or state.get("ref_json", [])
    ref_text = ref_json_to_text(ref_items)
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

    # Select language-specific template
    # Korean (ko): use default template
    # English (en): use *_en template if available
    # Japanese (ja): use *_ja template if available
    if detected_language == "en":
        templates = {
            "setup": spec.setup_ans_en or spec.setup_ans,
            "ts": spec.ts_ans_en or spec.ts_ans,
            "general": spec.general_ans_en or spec.general_ans,
        }
    elif detected_language == "ja":
        templates = {
            "setup": spec.setup_ans_ja or spec.setup_ans,
            "ts": spec.ts_ans_ja or spec.ts_ans,
            "general": spec.general_ans_ja or spec.general_ans,
        }
    else:  # ko or default
        templates = {
            "setup": spec.setup_ans,
            "ts": spec.ts_ans,
            "general": spec.general_ans,
        }

    tmpl = templates.get(route, spec.general_ans)
    logger.info("answer_node: using %s template for route=%s", detected_language, route)

    user = _format_prompt(tmpl.user, mapping)
    logger.info("answer_node: user_prompt_chars=%d, system_prompt_chars=%d", len(user), len(tmpl.system))
    answer, reasoning = _invoke_llm_with_reasoning(llm, tmpl.system, user, max_tokens=MAX_TOKENS_ANSWER)
    logger.info("answer_node: answer_chars=%d, reasoning_chars=%d, answer_preview=%s", len(answer), len(reasoning) if reasoning else 0, answer[:500] if answer else "(empty)")
    return {"answer": answer, "reasoning": reasoning}


def judge_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    ref_items = state.get("answer_ref_json") or state.get("ref_json", [])
    ref_text = ref_json_to_text(ref_items)
    # Use English query for consistent judge evaluation
    query_for_judge = state.get("query_en") or state["query"]

    if route == "setup":
        sys = spec.judge_setup_sys
    elif route == "ts":
        sys = spec.judge_ts_sys
    else:
        sys = spec.judge_general_sys

    user = (
        f"질문: {query_for_judge}\n"
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


def should_retry(state: AgentState) -> Literal["done", "retry", "retry_expand", "retry_mq", "human"]:
    """Determine retry strategy based on attempt count.

    Retry strategies:
    - 1st unfaithful (attempt 0→1): retry_expand - use more docs (5→10)
    - 2nd unfaithful (attempt 1→2): retry - refine queries
    - 3rd unfaithful (attempt 2→3): retry_mq - regenerate multi-query from scratch
    """
    if state.get("retrieval_confirmed"):
        return "done"
    judge = state.get("judge", {})
    faithful = bool(judge.get("faithful", False))
    if faithful:
        return "done"

    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 0)

    if attempts < max_attempts:
        if attempts == 0:
            # 1st retry: expand more docs (5→10)
            return "retry_expand"
        elif attempts == 1:
            # 2nd retry: refine queries
            return "retry"
        else:
            # 3rd+ retry: regenerate MQ from scratch
            return "retry_mq"

    # HIL 비활성화 모드 (auto_parse 등)에서는 human_review 건너뛰기
    if state.get("_skip_human_review"):
        return "done"
    return "human"


def retry_bump_node(state: AgentState) -> Dict[str, Any]:
    """Increment attempt counter."""
    return {"attempts": int(state.get("attempts", 0)) + 1}


def retry_expand_node(state: AgentState) -> Dict[str, Any]:
    """1st retry strategy: increase expand_top_k from 5 to 10.

    This doesn't re-retrieve docs, just uses more of the already retrieved docs
    for answer generation.
    """
    attempts = int(state.get("attempts", 0)) + 1
    logger.info("retry_expand_node: increasing expand_top_k to 10 (attempt %d)", attempts)
    return {
        "attempts": attempts,
        "expand_top_k": 10,
        "retry_strategy": "expand_more",
    }


def retry_mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    """3rd+ retry strategy: regenerate multi-query from scratch.

    Clears previous MQ lists and triggers fresh MQ generation.
    """
    attempts = int(state.get("attempts", 0)) + 1
    logger.info("retry_mq_node: regenerating MQ from scratch (attempt %d)", attempts)

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
    return re.sub(r"[\s\-_./]+", "", str(value).lower())


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
        # Only accept if the full device name appears in the query (strict match)
        if _compact_text(canonical) and _compact_text(canonical) in query_compact:
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
        if _compact_text(cleaned) and _compact_text(cleaned) in query_compact:
            matches.append(cleaned)
    return _dedupe_queries(matches)[:2]


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
    """Auto-parse device, doc_type, and language from user query using LLM.

    Args:
        state: Agent state with query.
        llm: LLM instance for generation.
        spec: Prompt specification (must have auto_parse template).
        device_names: List of available device names.
        doc_type_names: List of available doc type names.

    Returns:
        State update with auto_parsed_device, auto_parsed_doc_type, detected_language, and auto_parse_message.
    """
    query = state["query"]

    # 규칙 기반만 사용 (LLM 호출 없음 - 속도 최적화)
    detected_language = _detect_language_rule_based(query)
    devices = _extract_devices_from_query(device_names, query)
    doc_types = _extract_doc_types_from_query(query)

    logger.info("auto_parse_node: devices=%s, doc_types=%s, language=%s (rule-based only)", devices, doc_types, detected_language)

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

    # 파싱 결과가 있으면 추가
    if devices or doc_types:
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
    """
    query = state["query"]
    detected_language = state.get("detected_language", "ko")

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
        # Clean up result (remove quotes, extra whitespace)
        result = result.strip().strip('"').strip("'").strip()
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

    return {
        "query_en": query_en,
        "query_ko": query_ko,
        "_events": [
            {
                "type": "translate",
                "original": query,
                "query_en": query_en,
                "query_ko": query_ko,
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
    "expand_related_docs_node",
    "ask_user_after_retrieve_node",
    "answer_node",
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
