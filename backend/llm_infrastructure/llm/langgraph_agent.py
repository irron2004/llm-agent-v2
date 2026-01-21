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

from backend.domain.doc_type_mapping import expand_doc_type_selection, normalize_doc_type
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
    available_doc_types: List[Dict[str, Any]]
    selected_doc_types: List[str]
    doc_type_selection_skipped: bool

    # Retrieval outputs
    docs: List[RetrievalResult]
    display_docs: List[RetrievalResult]
    ref_json: List[Dict[str, Any]]
    answer_ref_json: List[Dict[str, Any]]

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
        items = re.findall(r'"([^"]+)"', t)
        qs = [
            i.strip()
            for i in items
            if i.strip() and i.strip().lower() not in {"queries", "search_queries"}
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
    selected_doc_types = state.get("selected_doc_types", [])
    selected_doc_type_filters = expand_doc_type_selection(selected_doc_types)
    original_query = state["query"]

    selected_device_set = {
        _normalize_device_name(d) for d in selected_devices if _normalize_device_name(d)
    }
    selected_doc_type_set = {
        _normalize_doc_type(dt) for dt in selected_doc_type_filters if _normalize_doc_type(dt)
    }

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

    if selected_devices:
        # Dual search strategy: device-filtered + general
        logger.info("retrieve_node: dual search with devices=%s", selected_devices)

        # Search 1: Device-filtered search (30 docs) with device_names as OR filter
        for q in queries:
            device_docs = retriever.retrieve(
                q,
                top_k=retrieval_top_k,
                device_names=selected_devices,
                doc_types=selected_doc_type_filters,
            )
            _add_docs(device_docs, filter_devices=True)

        device_filtered_count = len(all_docs)
        logger.info("retrieve_node: device-filtered search found %d docs", device_filtered_count)

        # Search 2: General search without device filter (30 docs)
        for q in queries:
            general_docs = retriever.retrieve(
                q,
                top_k=retrieval_top_k,
                doc_types=selected_doc_type_filters,
            )
            _add_docs(general_docs, filter_devices=False)

        logger.info("retrieve_node: after general search, total %d docs", len(all_docs))

    else:
        # No device selection: search 60 docs without filter
        logger.info("retrieve_node: general search (no device filter)")
        for q in queries:
            docs = retriever.retrieve(
                q,
                top_k=retrieval_top_k * 2,
                doc_types=selected_doc_type_filters,
            )
            _add_docs(docs, filter_devices=False)

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
    max_expand = max(0, int(EXPAND_TOP_K))
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


def answer_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    ref_items = state.get("answer_ref_json") or state.get("ref_json", [])
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
    ref_items = state.get("answer_ref_json") or state.get("ref_json", [])
    ref_text = ref_json_to_text(ref_items)

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
            "- 기기 선택: 선택 기기 문서 10개 + 전체 문서 10개 검색\n"
            "- 문서 종류 선택: 선택한 문서 종류로 검색 범위 제한\n"
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
    "refine_queries_node",
    "human_review_node",
    "device_selection_node",
]
