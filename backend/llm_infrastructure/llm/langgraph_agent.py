"""LangGraph node helpers + prompt spec.

이 파일은 노드/프롬프트 스펙/헬퍼를 제공하고, 실제 그래프 조립은
service 계층에서 담당한다.
"""

from __future__ import annotations

import json
import re
import logging
import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, Set, Tuple, TypedDict

from langgraph.types import Command, interrupt
from pydantic import BaseModel

from backend.config.settings import agent_settings, rag_settings
from backend.domain.doc_type_mapping import (
    DOC_TYPE_GROUPS,
    expand_doc_type_selection,
    normalize_doc_type,
)
from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate, load_prompt_template
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.retrieval.rrf import (
    merge_retrieval_result_lists_rrf,
    merge_retrieval_results_rrf,
)
from backend.services.search_service import SearchService

logger = logging.getLogger(__name__)


# -----------------------------
# 1) State schema
# -----------------------------
Route = Literal["setup", "ts", "general"]
Gate = Literal["need_st", "no_st"]


class ParsedQuery(BaseModel):
    """사용자 쿼리에서 파싱된 구조화된 검색 조건.

    파이프라인 노드들이 점진적으로 필드를 채운다.
    """

    # 필터 (auto_parse_node에서 채움)
    device_names: list[str] = []  # ES: device_name
    equip_ids: list[str] = []  # ES: equip_id
    doc_types: list[str] = []  # ES: doc_type
    detected_language: str | None = None  # "ko", "en", "ja", "zh"

    # 라우팅 (route_node에서 채움)
    route: str | None = None  # "setup", "ts", "general"

    # 검색 쿼리 (st_mq_node에서 채움)
    search_queries: list[str] = []

    # 선택 필터 (auto_parse 또는 사용자 선택)
    selected_devices: list[str] = []
    selected_equip_ids: list[str] = []
    selected_doc_types: list[str] = []
    doc_types_strict: bool = False

    # 메타 (UI 표시용)
    message: str | None = None
    device_selection_skipped: bool = False
    doc_type_selection_skipped: bool = False


class AgentState(TypedDict, total=False):
    query: str
    route: Route
    parsed_query: dict  # ParsedQuery.model_dump() — TypedDict은 Pydantic 객체 불가

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
    mq_mode: str
    mq_used: bool
    mq_reason: Optional[str]
    guardrail_dropped_numeric: int
    guardrail_dropped_anchor: int
    guardrail_final_count: int

    # Retry strategy configuration
    expand_top_k: int  # Number of docs to expand (default: 20, retry: 40)
    retry_strategy: str  # "expand_more" | "refine_queries" | "regenerate_mq"

    # Auto-parsed filters (from LLM)
    auto_parsed_device: Optional[str]  # First parsed device from query
    auto_parsed_doc_type: Optional[str]  # First parsed doc type from query
    auto_parsed_devices: List[str]
    auto_parsed_doc_types: List[str]
    auto_parsed_equip_id: Optional[str]  # First parsed equip_id from query
    auto_parsed_equip_ids: List[str]
    auto_parse_message: Optional[str]  # Message to display (e.g., "SUPRA N장비로 검색합니다")

    # Language detection and translation
    detected_language: Optional[str]  # "ko", "en", "ja", "zh" - detected from query
    target_language: Optional[str]  # Answer language override
    query_en: Optional[str]  # English version of query (for internal processing)
    query_ko: Optional[str]  # Korean version of query (for retrieval)
    task_mode: Optional[str]  # Guided confirm selection ("sop"|"issue"|"all")
    issue_stage: Optional[str]  # "post_summary" | "post_detail"
    issue_top10_cases: List[Dict[str, Any]]
    issue_case_refs: List[Dict[str, Any]]
    issue_case_ref_map: Dict[str, List[Dict[str, Any]]]
    issue_routing_signals: Dict[str, Any]
    issue_policy_tier: Optional[str]
    issue_policy_tier_shadow: Optional[str]
    issue_case_refs_shadow: List[Dict[str, Any]]
    issue_answer_refs_shadow: List[Dict[str, Any]]
    issue_case_ref_map_shadow: Dict[str, List[Dict[str, Any]]]
    issue_policy_rollout_phase: Optional[int]
    issue_fallback_reason: Optional[str]
    issue_detail_ref_source: Optional[str]
    issue_confirm_nonce: Optional[str]
    issue_case_selection_nonce: Optional[str]
    issue_sop_confirm_nonce: Optional[str]
    issue_selected_doc_id: Optional[str]

    # Guided auto-parse confirmation
    guided_confirm: bool
    auto_parse_confirmed: bool

    # Device selection (HIL)
    available_devices: List[Dict[str, Any]]
    selected_devices: List[str]  # Multiple devices can be selected
    selected_equip_ids: List[str]
    device_selection_skipped: bool
    available_doc_types: List[Dict[str, Any]]
    selected_doc_types: List[str]
    selected_doc_types_strict: bool
    doc_type_selection_skipped: bool
    selected_doc_ids: List[str]

    # Chat history (follow-up query support)
    chat_history: List[Dict[str, Any]]  # [{user_text, assistant_text, doc_ids}]
    needs_history: bool
    prev_doc_ids: List[str]  # 이전 턴 참조 문서 ID

    # Pre-fetched context docs (관련 문서 제안 버튼 클릭 시)
    context_docs: List[RetrievalResult]

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

    # Abbreviation disambiguation
    abbreviation_resolved: bool  # 약어 모호성 해소 완료 여부
    abbreviation_selections: Dict[str, str]  # {약어: 선택된 풀네임}
    original_query: Optional[str]  # 약어 확장 전 원본 쿼리 (동의어 variant 생성용)

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
    issue_mq: Optional[PromptTemplate] = None
    # Language-specific answer prompts
    setup_ans_en: Optional[PromptTemplate] = None
    setup_ans_zh: Optional[PromptTemplate] = None
    setup_ans_ja: Optional[PromptTemplate] = None
    ts_ans_en: Optional[PromptTemplate] = None
    ts_ans_zh: Optional[PromptTemplate] = None
    ts_ans_ja: Optional[PromptTemplate] = None
    general_ans_en: Optional[PromptTemplate] = None
    general_ans_zh: Optional[PromptTemplate] = None
    general_ans_ja: Optional[PromptTemplate] = None
    issue_ans: Optional[PromptTemplate] = None
    issue_detail_ans: Optional[PromptTemplate] = None
    issue_ans_en: Optional[PromptTemplate] = None
    issue_ans_zh: Optional[PromptTemplate] = None
    issue_ans_ja: Optional[PromptTemplate] = None
    issue_detail_ans_en: Optional[PromptTemplate] = None
    issue_detail_ans_zh: Optional[PromptTemplate] = None
    issue_detail_ans_ja: Optional[PromptTemplate] = None


DEFAULT_JUDGE_SETUP = """
# 역할
설치/세팅 답변이 질문과 검색 증거(REFS)에 충실한지 판정한다.

# 평가 기준
- 답변의 내용이 REFS 근거에 기반하는지 확인한다.
- 답변의 형식(한국어, 마크다운 등)은 평가 대상이 아니다.
- issues에는 내용상 문제(누락된 단계, 잘못된 수치, 근거 없는 주장 등)만 기록한다.

# 너의 출력 형식 (반드시 JSON 한 줄로)
{"faithful": bool, "issues": ["..."], "hint": "..."}
- faithful: 답변이 REFS 근거에 기반하면 true, 아니면 false
- issues: 근거 누락/부정확한 내용/잘못된 조치 등 (내용 문제만)
- hint: 재검색/보강에 도움이 되는 한 줄 제안
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


def _load_setup_answer_prompt(version: str) -> PromptTemplate:
    """Load setup answer prompt with v3 override for v2 runtime."""
    if version == "v2":
        setup_ans_v3 = _try_load_prompt("setup_ans", "v3")
        if setup_ans_v3 is not None:
            logger.info("Using setup_ans_v3 prompt while running prompt spec v2")
            return setup_ans_v3
    return load_prompt_template("setup_ans", version)


def load_prompt_spec(version: str = "v1") -> PromptSpec:
    """Load required prompts from YAML (router/MQ/gate/answer)."""

    router = load_prompt_template("router", version)
    setup_mq = load_prompt_template("setup_mq", version)
    ts_mq = load_prompt_template("ts_mq", version)
    general_mq = load_prompt_template("general_mq", version)
    st_gate = load_prompt_template("st_gate", version)
    st_mq = load_prompt_template("st_mq", version)
    setup_ans = _load_setup_answer_prompt(version)
    ts_ans = load_prompt_template("ts_ans", version)
    general_ans = load_prompt_template("general_ans", version)

    # Try to load optional prompts
    auto_parse = _try_load_prompt("auto_parse", version)
    translate = _try_load_prompt("translate", version)
    issue_mq = _try_load_prompt("issue_mq", version)

    # Load language-specific answer prompts (optional, prefer v3 over v2)
    def _load_lang_prompt(name: str, ver: str) -> PromptTemplate | None:
        if ver == "v2":
            v3 = _try_load_prompt(name, "v3")
            if v3 is not None:
                return v3
        return _try_load_prompt(name, ver)

    setup_ans_en = _load_lang_prompt("setup_ans_en", version)
    setup_ans_zh = _load_lang_prompt("setup_ans_zh", version)
    setup_ans_ja = _load_lang_prompt("setup_ans_ja", version)
    ts_ans_en = _try_load_prompt("ts_ans_en", version)
    ts_ans_zh = _try_load_prompt("ts_ans_zh", version)
    ts_ans_ja = _try_load_prompt("ts_ans_ja", version)
    general_ans_en = _try_load_prompt("general_ans_en", version)
    general_ans_zh = _try_load_prompt("general_ans_zh", version)
    general_ans_ja = _try_load_prompt("general_ans_ja", version)
    issue_ans = _try_load_prompt("issue_ans", version)
    issue_detail_ans = _try_load_prompt("issue_detail_ans", version)
    issue_ans_en = _try_load_prompt("issue_ans_en", version)
    issue_ans_zh = _try_load_prompt("issue_ans_zh", version)
    issue_ans_ja = _try_load_prompt("issue_ans_ja", version)
    issue_detail_ans_en = _try_load_prompt("issue_detail_ans_en", version)
    issue_detail_ans_zh = _try_load_prompt("issue_detail_ans_zh", version)
    issue_detail_ans_ja = _try_load_prompt("issue_detail_ans_ja", version)

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
        issue_mq=issue_mq,
        setup_ans_en=setup_ans_en,
        setup_ans_zh=setup_ans_zh,
        setup_ans_ja=setup_ans_ja,
        ts_ans_en=ts_ans_en,
        ts_ans_zh=ts_ans_zh,
        ts_ans_ja=ts_ans_ja,
        general_ans_en=general_ans_en,
        general_ans_zh=general_ans_zh,
        general_ans_ja=general_ans_ja,
        issue_ans=issue_ans,
        issue_detail_ans=issue_detail_ans,
        issue_ans_en=issue_ans_en,
        issue_ans_zh=issue_ans_zh,
        issue_ans_ja=issue_ans_ja,
        issue_detail_ans_en=issue_detail_ans_en,
        issue_detail_ans_zh=issue_detail_ans_zh,
        issue_detail_ans_ja=issue_detail_ans_ja,
    )


# -----------------------------
# 3) Interfaces
# -----------------------------
class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int = 8) -> List[RetrievalResult]: ...


class SearchServiceRetriever:
    """Adapter to reuse existing SearchService inside the graph."""

    def __init__(self, search_service: SearchService, *, top_k: int = 8) -> None:
        self.search_service = search_service
        self.top_k = top_k
        es_engine = getattr(search_service, "es_engine", None)
        if es_engine is not None:
            self.es_engine = es_engine

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        device_name: str | None = None,
        device_names: List[str] | None = None,
        equip_ids: List[str] | None = None,
        doc_types: List[str] | None = None,
        doc_ids: List[str] | None = None,
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
        if equip_ids:
            search_kwargs["equip_ids"] = equip_ids
        if doc_types:
            search_kwargs["doc_types"] = doc_types
        if doc_ids:
            search_kwargs["doc_ids"] = doc_ids
        return self.search_service.search(query, **search_kwargs)


# -----------------------------
# 4) LLM helpers
# -----------------------------
# 노드 타입별 max_tokens 설정
MAX_TOKENS_CLASSIFICATION = 256  # 라우팅/분류용 (짧은 응답)
MAX_TOKENS_JUDGE = 1024  # judge용 (reasoning 토큰 포함 여유분)
MAX_TOKENS_ANSWER = 16384  # 답변 생성용 (oss-120b 32K context 기준, 입력 ~16K 여유)
MAX_REF_CHARS_REVIEW = 200  # 검색 결과 리뷰용
MAX_REF_CHARS_ANSWER = 16000  # 답변 생성용 (SOP 다페이지 절차 원문 번역용)
RELATED_PAGE_WINDOW = 2  # 인접 페이지 범위 (±N)
DOC_TYPES_SAME_DOC = {"gcb", "myservice", "pems"}
EXPAND_TOP_K = 10  # 확장 대상 최대 개수 (rerank 상위)
REPETITION_MIN_BLOCK_LEN = 40  # 반복 감지 최소 블록 길이 (문자)
REPETITION_MAX_REPEATS = 2  # 같은 블록 최대 허용 반복 횟수


def _truncate_repetition(
    text: str,
    min_block_len: int = REPETITION_MIN_BLOCK_LEN,
    max_repeats: int = REPETITION_MAX_REPEATS,
) -> str:
    """텍스트에서 반복되는 블록을 감지하고 절삭한다.

    긴 블록(min_block_len 이상)이 max_repeats회를 초과하여 연속 반복되면,
    max_repeats회까지만 유지하고 나머지를 제거한다.
    """
    if not text or len(text) < min_block_len * (max_repeats + 1):
        return text

    # 각 위치에서 최소 반복 단위를 찾아 반복 횟수 확인
    i = 0
    while i < len(text) - min_block_len * 2:
        for block_len in range(min_block_len, min(800, (len(text) - i) // 2) + 1):
            # text[i:i+L] == text[i+L:i+2L] 인지 확인
            if text[i : i + block_len] != text[i + block_len : i + block_len * 2]:
                continue

            # 반복 발견 — 횟수 카운트
            block = text[i : i + block_len]
            count = 2
            j = i + block_len * 2
            while j + block_len <= len(text) and text[j : j + block_len] == block:
                count += 1
                j += block_len

            if count > max_repeats:
                before = text[:i]
                kept = block * max_repeats
                after = text[i + block_len * count :]
                truncated = before + kept + after
                logger.info(
                    "_truncate_repetition: %d → %d chars (block=%d, repeats=%d→%d)",
                    len(text),
                    len(truncated),
                    block_len,
                    count,
                    max_repeats,
                )
                # 재귀: 추가 반복이 있을 수 있음
                return _truncate_repetition(truncated, min_block_len, max_repeats)

            # 최소 반복 단위를 찾았으면 더 큰 블록 시도 불필요
            break

        i += 1

    return text


# --- Per-node temperature settings ---
# 분류/판단 노드: 결정적 (동일 입력 → 동일 출력)
TEMP_CLASSIFICATION = 0.0  # route, st_gate, judge
TEMP_TRANSLATION = 0.0  # translate
# 생성 노드: 약간의 다양성 허용
TEMP_QUERY_GEN = 0.3  # mq, st_mq, refine_queries
TEMP_ANSWER = 0.5  # answer
TEMP_ANSWER_SETUP = 0.2  # answer (setup)

ISSUE_CASE_EMPTY_MESSAGE = "관련 이슈 사례를 찾지 못했습니다."

_ISSUE_SCOPE_GROUPS = ("myservice", "gcb", "ts")
_SOP_SCOPE_GROUPS = ("SOP", "setup")


def _build_scope_variants(*group_names: str) -> set[str]:
    variants: set[str] = set()
    for group_name in group_names:
        variants.add(normalize_doc_type(group_name))
        for raw in DOC_TYPE_GROUPS.get(group_name, []):
            normalized = normalize_doc_type(raw)
            if normalized:
                variants.add(normalized)
    return variants


_ISSUE_SCOPE_VARIANTS = _build_scope_variants(*_ISSUE_SCOPE_GROUPS) | {
    normalize_doc_type("pems"),
    normalize_doc_type("trouble_shooting_guide"),
    normalize_doc_type("trouble shooting guide"),
}
_TS_SCOPE_VARIANTS = _build_scope_variants("ts") | {
    normalize_doc_type("pems"),
    normalize_doc_type("trouble_shooting_guide"),
    normalize_doc_type("trouble shooting guide"),
}
_SOP_SCOPE_VARIANTS = _build_scope_variants(*_SOP_SCOPE_GROUPS)


def _infer_task_mode_from_doc_types(doc_types: Any) -> Optional[str]:
    if not isinstance(doc_types, list):
        return None
    normalized = {
        normalize_doc_type(str(value)) for value in doc_types if normalize_doc_type(str(value))
    }
    if not normalized:
        return None
    if normalized <= _TS_SCOPE_VARIANTS:
        return "ts"
    if normalized <= _ISSUE_SCOPE_VARIANTS:
        return "issue"
    if normalized <= _SOP_SCOPE_VARIANTS:
        return "sop"
    return None


MAX_ANSWER_FORMAT_RETRIES = 2

# --- Intent keyword sets for routing & retrieval gating ---
# Procedure intent: 절차/교체/설치 등 작업 수행 의도
_PROCEDURE_KEYWORDS_SET = frozenset({
    "교체", "절차", "작업", "방법", "replacement", "procedure",
    "how to", "install", "설치", "수리", "repair",
})
# Inquiry intent: 문서 내 특정 섹션 조회/열람 의도
_INQUIRY_KEYWORDS = frozenset({
    "조회", "보여줘", "알려줘", "목록", "리스트",
    "worksheet", "work sheet", "tool list", "check sheet",
    "scope", "목차", "part 위치", "개요", "overview",
    "show me", "list of",
})
# 복합 패턴: procedure 키워드를 포함하지만 실제로는 조회 의도인 구문
# "작업 check sheet", "작업 체크시트" 등 — procedure wins를 무효화
_INQUIRY_COMPOUND_PATTERNS = (
    "작업 check", "작업 체크", "작업check", "작업체크",
    "work check", "work sheet",
)

# Tokens to exclude from doc_id boost (too generic or structural segments in doc_ids)
_DOC_ID_BOOST_STOP = frozenset({
    "the", "how", "to", "of", "in", "for", "and", "is", "are", "what",
    "sop", "pems", "manual", "tsg", "global", "eng", "kor", "en",
})


def _extract_doc_id_boost_tokens(
    query: str,
    device_names: list,
) -> list:
    """Extract meaningful ASCII tokens from query for doc_id wildcard boosting.

    Keeps only tokens that contain ASCII letters (doc_ids are English),
    excludes device name tokens and procedural stopwords.
    """
    tokens = re.split(r"[\s,.\-_]+", query.lower())
    device_tokens: set = set()
    for d in device_names:
        for t in re.split(r"[\s,.\-_]+", d.lower()):
            if t:
                device_tokens.add(t)

    result: list = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if not any(c.isascii() and c.isalpha() for c in tok):
            continue
        if tok in _DOC_ID_BOOST_STOP or tok in _PROCEDURE_KEYWORDS_SET:
            continue
        if tok in device_tokens:
            continue
        result.append(tok)
    return result


def _normalize_device_in_query(query: str, canonical_device: str) -> str:
    """Replace fuzzy device name variant in query with canonical form.

    Uses compact text comparison + rapidfuzz WRatio to find the variant
    span in the query and replace it with the canonical device name.
    Reuses the same matching strategy as _extract_devices_from_query.
    """
    canonical_compact = _compact_text(canonical_device)
    if not canonical_compact or len(canonical_compact) < 3:
        return query

    # Split query into word tokens (Korean separators included)
    raw_tokens = re.split(r"([\s가-힣,;:()\"'?!。，；：（）의]+)", query)
    word_tokens = [t for t in raw_tokens if t.strip() and not re.fullmatch(r"[\s가-힣,;:()\"'?!。，；：（）의]+", t)]

    if not word_tokens:
        return query

    # Try 1-, 2-, 3-token spans (matching _extract_devices_from_query strategy)
    best_span: str | None = None
    best_score: float = 0

    for width in range(1, min(4, len(word_tokens) + 1)):
        for i in range(len(word_tokens) - width + 1):
            span_text = " ".join(word_tokens[i : i + width])
            span_compact = _compact_text(span_text)
            if not span_compact or len(span_compact) < 3:
                continue
            # Exact compact match
            if span_compact == canonical_compact:
                return query.replace(span_text, canonical_device, 1)
            # Fuzzy match (same threshold as device detection)
            try:
                from rapidfuzz import fuzz

                score = fuzz.WRatio(span_compact, canonical_compact)
                if score >= 82 and score > best_score:
                    best_score = score
                    best_span = span_text
            except ImportError:
                pass

    if best_span is not None:
        return query.replace(best_span, canonical_device, 1)
    return query


_CITATION_RE = re.compile(r"\[[0-9]+\]")
_EMOJI_NUMERAL_RE = re.compile(r"[0-9]️⃣")
_MARKDOWN_TABLE_LINE_RE = re.compile(r"(?m)^\|.*\|\s*$")

_REQUIRED_SECTIONS_KO = [
    "## 작업 절차",
]


def _answer_language_ok(answer: str, target_language: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False
    lang = (target_language or "").strip().lower()
    if lang == "ko":
        return bool(re.search(r"[가-힣]", text))
    if lang == "en":
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return (non_ascii / max(len(text), 1)) < 0.2
    return True


def _validate_answer_format(
    answer: str,
    *,
    target_language: str,
    has_refs: bool,
) -> Dict[str, Any]:
    text = answer or ""
    stripped = text.lstrip()
    title_ok = stripped.startswith("# ")
    missing_sections = [s for s in _REQUIRED_SECTIONS_KO if s not in text]
    has_emoji_numbering = bool(_EMOJI_NUMERAL_RE.search(text))
    has_markdown_table = "|---|" in text or bool(_MARKDOWN_TABLE_LINE_RE.search(text))
    # 번호 목록(N. )이 있으면 OK — 원문 Step 번호가 1이 아닐 수 있으므로 임의 숫자 허용
    numbering_ok = bool(re.search(r"(?m)^\s*\d+\.\s+", text))
    citations_ok = True
    references_ok = True
    if has_refs:
        citations_ok = bool(_CITATION_RE.search(text))
        # 참고문헌 섹션은 선택 — 없어도 통과
        references_ok = True
    language_ok = _answer_language_ok(text, target_language)

    ok = (
        title_ok
        and not missing_sections
        and numbering_ok
        and not has_emoji_numbering
        and not has_markdown_table
        and citations_ok
        and language_ok
    )
    return {
        "ok": ok,
        "title_ok": title_ok,
        "missing_sections": missing_sections,
        "numbering_ok": numbering_ok,
        "citations_ok": citations_ok,
        "references_ok": references_ok,
        "language_ok": language_ok,
        "has_emoji_numbering": has_emoji_numbering,
        "has_markdown_table": has_markdown_table,
    }


def resolve_querygen_temperature(state: AgentState, *, mq_invoked: bool = False) -> float:
    mq_mode = str(state.get("mq_mode") or "on").strip().lower()
    attempts = int(state.get("attempts", 0) or 0)

    if mq_mode == "on":
        return TEMP_QUERY_GEN
    if mq_mode == "fallback" and mq_invoked:
        return TEMP_CLASSIFICATION
    if mq_mode in {"off", "fallback"} and attempts < 2:
        return TEMP_CLASSIFICATION
    return TEMP_QUERY_GEN


def resolve_answer_temperature(route: Route) -> float:
    if route == "setup":
        return TEMP_ANSWER_SETUP
    return TEMP_ANSWER


def _invoke_llm(llm: BaseLLM, system: str, user: str, **kwargs: Any) -> str:
    messages: List[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    # max_tokens 기본값: 분류용 (answer_node에서는 명시적으로 더 큰 값 전달)
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = MAX_TOKENS_CLASSIFICATION
    logger.debug(
        "_invoke_llm: max_tokens=%s, system_len=%d, user_len=%d",
        kwargs.get("max_tokens"),
        len(system),
        len(user),
    )
    out = llm.generate(messages, **kwargs)
    result = out.text.strip()
    logger.debug("_invoke_llm: output_len=%d", len(result))
    return result


def _invoke_llm_with_reasoning(
    llm: BaseLLM, system: str, user: str, **kwargs: Any
) -> tuple[str, str | None]:
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
    logger.debug(
        "_invoke_llm_with_reasoning: output_len=%d, reasoning_len=%d",
        len(text),
        len(reasoning) if reasoning else 0,
    )
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


def _strip_regeneration_prefixes(text: str) -> str:
    normalized = str(text).strip()
    if not normalized:
        return ""
    normalized = re.sub(
        r"^(?:\[\s*regenerate with[^\]]*\]\s*)+", "", normalized, flags=re.I
    ).strip()
    normalized = re.sub(
        r"^(?:regenerate with\b[^:\n]*[:\-]?\s*)+", "", normalized, flags=re.I
    ).strip()
    normalized = re.sub(
        r"^(?:재검색\s*(?:조건|필터)?\s*[:\-]?\s*)+", "", normalized, flags=re.I
    ).strip()
    return normalized


def _normalize_query_text(text: Any) -> str:
    normalized = _strip_regeneration_prefixes(str(text))
    normalized = normalized.strip().strip("\"'`").strip()
    return normalized


def _looks_like_placeholder_query(text: str) -> bool:
    original = str(text).strip()
    if not original:
        return False
    value = _normalize_query_text(original).lower()
    if not value:
        return True

    if re.fullmatch(r"[.\s…·•\-_=~]+", value):
        return True

    if value in {"query", "queries", "search query", "search queries", "검색어", "질문"}:
        return True

    compact = re.sub(r"[\s\-_]+", "", value)
    if compact in {"query", "queries", "searchquery", "searchqueries", "검색어", "질문"}:
        return True

    if re.fullmatch(r"(?:q|query|searchquery|검색어|질문)\d{0,3}", compact):
        return True

    return False


def _parse_queries(text: str) -> List[str]:
    """Robust parser: JSON object/list or one-per-line strings."""
    t = text.strip()
    if not t:
        return []

    # Strip common code-fence wrappers.
    t = re.sub(r"```[a-zA-Z]*", "", t).strip()

    def _extract_queries(obj: Any) -> List[str]:
        def _collect(items: List[Any]) -> List[str]:
            collected: List[str] = []
            for item in items:
                cleaned = _normalize_query_text(item)
                if cleaned and not _looks_like_placeholder_query(cleaned):
                    collected.append(cleaned)
            return collected

        if isinstance(obj, dict):
            for key in ("queries", "search_queries"):
                if isinstance(obj.get(key), list):
                    return _collect(obj[key])
        if isinstance(obj, list):
            return _collect(obj)
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
        qs = [
            cleaned
            for i in items
            if (cleaned := _normalize_query_text(i)) and not _looks_like_placeholder_query(cleaned)
        ]
        if qs:
            return _dedupe_queries(qs)

    if '"queries"' in t or '"search_queries"' in t:
        # Known garbage labels that may leak from prompts
        garbage_labels = {"queries", "search_queries", "setup_mq", "ts_mq", "general_mq", "gate"}
        items = re.findall(r'"([^"]+)"', t)
        qs = [
            cleaned
            for i in items
            if (cleaned := _normalize_query_text(i))
            and cleaned.lower() not in garbage_labels
            and not _looks_like_placeholder_query(cleaned)
        ]
        if qs:
            return _dedupe_queries(qs)

    def _is_meta_line(line: str) -> bool:
        lower = line.lower()
        if any(
            pattern in lower
            for pattern in [
                "given original",
                "they want",
                "from mq:",
                "we need",
                "could be:",
                "example output",
                "example input",
                "output only",
                "no explanations",
                "query generation",
                "output format",
            ]
        ):
            return True
        if lower.startswith(("output:", "example:", "format:", "input:")):
            return True
        # Filter out prompt template labels that may leak into output
        if any(
            label in lower
            for label in [
                "setup_mq:",
                "ts_mq:",
                "general_mq:",
                "gate:",
                "질문:",
            ]
        ):
            return True
        return False

    def _strip_prefix(line: str) -> str:
        cleaned = line.strip()
        cleaned = re.sub(r"^[-*•]\s+", "", cleaned)
        cleaned = re.sub(r"^(?:q(?:uery)?\d+|\d+)\s*[:\.\)\-]\s*", "", cleaned, flags=re.I)
        cleaned = cleaned.strip().strip("\"'`").strip()
        cleaned = re.sub(r"[;,，]\s*$", "", cleaned).strip()
        return cleaned

    # Filter out meta-explanation lines and clean numbered/bulleted outputs.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    filtered: List[str] = []
    for line in lines:
        if _is_meta_line(line):
            continue
        if line.startswith("{"):
            continue
        if line.startswith("[") and line.endswith("]") and "regenerate with" not in line.lower():
            continue
        cleaned = _normalize_query_text(_strip_prefix(line))
        if cleaned and not _looks_like_placeholder_query(cleaned):
            filtered.append(cleaned)

    if len(filtered) == 1:
        single = filtered[0]
        for delim in (" / ", " | ", ";", "；"):
            if delim in single:
                split_items = [part.strip() for part in single.split(delim) if part.strip()]
                if len(split_items) > 1:
                    filtered = [
                        cleaned
                        for part in split_items
                        if (cleaned := _normalize_query_text(part))
                        and not _looks_like_placeholder_query(cleaned)
                    ]
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


_UNIT_LIKE_PATTERN = re.compile(
    r"(?i)(?:\b(?:psi|bar|pa|kpa|mpa|nm|mm|cm|um|v|kv|a|ma|w|kw|rpm|degc)\b|%|°c)"
)


def _query_dedupe_key(query: str) -> str:
    return " ".join(str(query).split()).strip().lower()


def _anchor_tokens(text: str) -> List[str]:
    tokens = re.split(r"[\s\W_]+", text or "", flags=re.UNICODE)
    anchors: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        tok = token.strip()
        if len(tok) < 2:
            continue
        if not re.search(r"[A-Za-z가-힣]", tok):
            continue
        lowered = tok.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        anchors.append(tok)
    return anchors


def validate_search_queries(state: AgentState, candidates: List[str]) -> Dict[str, Any]:
    original_query = str(state.get("query") or "").strip()
    stable_query = _normalize_query_text(state.get("query_en") or state.get("query") or "")

    normalized_candidates: List[str] = []
    for candidate in candidates:
        cleaned = _normalize_query_text(candidate)
        if not cleaned or _is_garbage_query(cleaned):
            continue
        normalized_candidates.append(cleaned)

    with_original = ([original_query] if original_query else []) + normalized_candidates

    deduped: List[str] = []
    seen_keys: set[str] = set()
    for query in with_original:
        key = _query_dedupe_key(query)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(query)

    original_key = _query_dedupe_key(original_query)
    has_user_digits = bool(re.search(r"\d", original_query))
    user_digit_substrings = set(re.findall(r"\d+", original_query)) if has_user_digits else set()
    anchors = _anchor_tokens(original_query)

    dropped_numeric = 0
    dropped_anchor = 0
    filtered: List[str] = []

    for query in deduped:
        key = _query_dedupe_key(query)
        if original_key and key == original_key:
            filtered.append(query)
            continue

        if has_user_digits:
            candidate_digits = re.findall(r"\d+", query)
            if any(digit not in user_digit_substrings for digit in candidate_digits):
                dropped_numeric += 1
                continue
        else:
            if re.search(r"\d", query) or _UNIT_LIKE_PATTERN.search(query):
                dropped_numeric += 1
                continue

        matched = sum(1 for anchor in anchors if anchor.lower() in query.lower())
        recall = matched / (len(anchors) or 1)
        if recall < 0.6:
            dropped_anchor += 1
            continue

        filtered.append(query)

    if original_query:
        if not filtered or _query_dedupe_key(filtered[0]) != original_key:
            filtered = [original_query] + [
                q for q in filtered if _query_dedupe_key(q) != original_key
            ]

    filtered = filtered[:5]

    if not filtered:
        fallback = stable_query or original_query
        filtered = [fallback] if fallback else []

    return {
        "search_queries": filtered,
        "guardrail_dropped_numeric": int(dropped_numeric),
        "guardrail_dropped_anchor": int(dropped_anchor),
        "guardrail_final_count": int(len(filtered)),
    }


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

        # Keep metadata compact but preserve issue-route disambiguation signals.
        metadata: Dict[str, Any] = {}
        if truncated:
            metadata["truncated"] = True
        if isinstance(d.metadata, dict):
            for key in ("device_name", "equip_id", "doc_type"):
                val = d.metadata.get(key)
                if val and str(val).strip():
                    metadata[key] = str(val).strip()
            section = (
                d.metadata.get("section_type")
                or d.metadata.get("section_chapter")
                or d.metadata.get("chapter")
            )
            if section and str(section).strip():
                metadata["section"] = str(section).strip()

        # Extract page number for scroll positioning
        page_val = None
        if isinstance(d.metadata, dict):
            raw_page = d.metadata.get("page")
            if raw_page is not None:
                try:
                    page_val = int(raw_page)
                except (ValueError, TypeError):
                    pass

        ref.append(
            {
                "rank": i,
                "doc_id": d.doc_id,
                "content": content,
                "metadata": metadata,
                "score": getattr(d, "score", None),
                "page": page_val,
            }
        )
    return ref


def _strip_latex_noise(text: str) -> str:
    """Remove LaTeX markup, empty table cells, image refs, and page headers."""
    # 이미지 참조 제거
    text = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]*\}", "", text)
    # LaTeX 테이블 환경 태그 제거
    text = re.sub(r"\\begin\{tabular[^}]*\}(\[[^\]]*\])?", "", text)
    text = re.sub(r"\\end\{tabular\}", "", text)
    # LaTeX 명령어 제거 (textbf, hline, cline, multirow, parbox 등)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textcolor\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:hline|cline\{[^}]*\})", "", text)
    text = re.sub(r"\\(?:multirow|multicolumn)\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\parbox\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:newline|\\)", " ", text)
    text = re.sub(r"\\(?:rightarrow|neq)", "→", text)
    text = re.sub(r"\$[^$]*\$", "", text)  # inline math
    # 빈 테이블 셀 반복 제거 (& & \\ 패턴)
    text = re.sub(r"(?:\s*&\s*&\s*\\\\)+", " ", text)
    text = re.sub(r"(?:\s*&\s*\\\\)+", " ", text)
    # 코드 펜스 제거
    text = re.sub(r"```(?:markdown)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    # 페이지 헤더/푸터 반복 제거
    text = re.sub(r"PS-320-R0\s*(?:PSK\s*)?A4\([^)]*\)", "", text)
    # HTML 주석 (테이블 좌표 등) 제거
    text = re.sub(r"<!--[^>]*-->", "", text)
    # 페이지 구분 마커 정리 (p11:, p12: 등 → 줄바꿈)
    text = re.sub(r"\bp\d+:\s*", "\n", text)
    # 연속 공백/줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


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
        content = _strip_latex_noise(content)
        content = " ".join(content.split())

        # 장비 + 문서유형 태그 추가
        meta = r.get("metadata") or {}
        device = meta.get("device_name", "")
        equip = meta.get("equip_id", "")
        doc_type = meta.get("doc_type", "")
        section = meta.get("section", "")

        tags: List[str] = []
        if device and equip:
            tags.append(f"{device} / {equip}")
        elif device:
            tags.append(str(device))
        elif equip:
            tags.append(str(equip))
        if doc_type:
            tags.append(f"type={doc_type}")
        if section:
            tags.append(f"section={section}")

        tag = f" ({' | '.join(tags)})" if tags else ""

        lines.append(f"[{rank}] {doc_id}{tag}: {content}")
    return "\n".join(lines)


PROCEDURE_FIRST_KEYWORDS: tuple[str, ...] = (
    "work procedure",
    "workflow",
    "procedure",
    "작업 절차",
    "절차",
    "교체",
    "replacement",
    "설치",
    "install",
    "installation",
    "조정",
    "adjust",
    "adjustment",
    "setting",
    "셋업",
    "setup",
)

CAUTION_KEYWORDS: tuple[str, ...] = (
    "warning",
    "caution",
    "note",
    "주의",
    "경고",
)

OVERVIEW_KEYWORDS: tuple[str, ...] = (
    "scope",
    "overview",
    "purpose",
    "background",
    "개요",
    "목적",
    "배경",
)

STEP_EMOJI_TO_NUMBER: dict[str, str] = {
    "0️⃣": "1.",
    "1️⃣": "1.",
    "2️⃣": "2.",
    "3️⃣": "3.",
    "4️⃣": "4.",
    "5️⃣": "5.",
    "6️⃣": "6.",
    "7️⃣": "7.",
    "8️⃣": "8.",
    "9️⃣": "9.",
}


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _score_setup_ref_priority(text: str) -> int:
    if _contains_any_keyword(text, PROCEDURE_FIRST_KEYWORDS):
        return 0
    if _contains_any_keyword(text, CAUTION_KEYWORDS):
        return 1
    if _contains_any_keyword(text, OVERVIEW_KEYWORDS):
        return 3
    return 2


def _prioritize_setup_answer_refs(ref_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ref_items:
        return ref_items

    scored_items: list[tuple[int, int, int, Dict[str, Any]]] = []
    for idx, item in enumerate(ref_items):
        content = str(item.get("content") or "")
        metadata_raw = item.get("metadata")
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        extra = " ".join(
            str(metadata.get(key) or "") for key in ("section_type", "chapter", "title", "source")
        )
        priority = _score_setup_ref_priority(f"{content}\n{extra}")

        rank_raw = item.get("rank")
        try:
            original_rank = int(str(rank_raw))
        except (TypeError, ValueError):
            original_rank = idx + 1

        scored_items.append((priority, original_rank, idx, item))

    sorted_items = [
        item
        for _, _, _, item in sorted(
            scored_items,
            key=lambda row: (row[0], row[1], row[2]),
        )
    ]

    reordered: List[Dict[str, Any]] = []
    for new_rank, item in enumerate(sorted_items, start=1):
        updated = dict(item)
        updated["rank"] = new_rank
        reordered.append(updated)
    return reordered


MAX_SETUP_DOC_TRIES = 5  # 적합성 판정 최대 문서 수
MIN_REFS_FOR_ACCEPT = 3  # 그룹 즉시 채택 최소 refs 수
CONSECUTIVE_EMPTY_LIMIT = 2  # 연속 빈 응답 시 early-exit 임계값
MAX_ANSWER_REFS = 10  # 답변 생성 시 최대 REFS 수 (다페이지 SOP 원문 번역 대응)
MAX_ISSUE_REFS = 10  # issue 사례 노출 최대 REFS 수
MAX_ISSUE_CASE_MAP_DOCS = 10
MAX_ISSUE_CASE_MAP_REFS_PER_DOC = 12
MAX_ISSUE_CASE_MAP_CONTENT_CHARS = 800
ISSUE_POLICY_ROLLOUT_PHASE_DEFAULT = 3
ISSUE_POLICY_ROLLOUT_PHASE_MIN = 1
ISSUE_POLICY_ROLLOUT_PHASE_MAX = 3


def _group_refs_by_doc_id(
    ref_items: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """ref_items를 doc_id별로 그룹핑. 첫 등장 순서 유지."""
    from collections import OrderedDict

    groups: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for item in ref_items:
        doc_id = str(item.get("doc_id", "unknown"))
        groups.setdefault(doc_id, []).append(item)
    return list(groups.items())


def _group_refs_by_doc_section_chapter(
    ref_items: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """ref_items를 (doc_id, section_chapter)별로 그룹핑. 첫 등장 순서 유지.

    같은 doc_id라도 section_chapter가 다르면 별도 그룹으로 분리한다.
    이를 통해 하나의 물리 문서에 여러 SOP가 포함된 경우
    (e.g. REPLACEMENT vs CALIBRATION) 각각 독립적으로 적합성 판정 가능.

    ref_items의 metadata["section"]은 원본 section_chapter에서 유래한 값이다.
    """
    from collections import OrderedDict

    groups: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    for item in ref_items:
        doc_id = str(item.get("doc_id", "unknown"))
        meta = item.get("metadata") or {}
        section_chapter = str(meta.get("section", "") or "").strip()
        # section_chapter가 있으면 doc_id + section_chapter로 구분, 없으면 doc_id만
        group_key = f"{doc_id}::{section_chapter}" if section_chapter else doc_id
        groups.setdefault(group_key, []).append(item)
    return list(groups.items())


def _check_doc_relevance(
    query: str,
    doc_ref_text: str,
    *,
    llm: "BaseLLM",
) -> Optional[bool]:
    """문서가 질문에 답할 수 있는 절차/작업 정보를 포함하는지 가볍게 판정.

    Returns:
        True: 관련 있음 ("yes" 포함)
        False: 관련 없음 ("yes" 미포함)
        None: 빈 응답으로 판정 불가 — 호출자가 처리
    """
    system = (
        "You are a relevance checker. Determine if the document contains "
        "procedure/work steps that can answer the user's question.\n"
        "Compare the SPECIFIC subject in the question (model number, part name, "
        "procedure type) against what the document actually describes.\n"
        "Reply ONLY 'yes' or 'no'."
    )
    user = f"Question: {query}\nDocument:\n{doc_ref_text[:6000]}"

    raw = _invoke_llm(llm, system, user, max_tokens=10, temperature=TEMP_CLASSIFICATION)
    raw_stripped = (raw or "").strip().lower()
    if not raw_stripped:
        logger.info("_check_doc_relevance: raw empty → None (undecidable)")
        return None
    result = "yes" in raw_stripped
    logger.info("_check_doc_relevance: raw=%r → %s", raw_stripped[:50], result)
    return result


def _postprocess_setup_answer_text(answer: str) -> str:
    normalized = answer
    for emoji, number_token in STEP_EMOJI_TO_NUMBER.items():
        normalized = normalized.replace(emoji, number_token)

    normalized = re.sub(r"【\s*\[?(\d+)\]?[^】]*】", r"[\1]", normalized)
    normalized = re.sub(r"\[\s*\[(\d+)\]\s*\]", r"[\1]", normalized)
    normalized = re.sub(r"\[\s*(?:…|\.\.\.)\s*\]", "", normalized)
    normalized = normalized.replace("REFS", "").replace("TBD", "")

    lines: list[str] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
                continue
            cells = [cell for cell in cells if cell]
            if cells:
                lines.append(" - ".join(cells))
            continue
        lines.append(line)
    normalized = "\n".join(lines)

    if "### 작업 절차" not in normalized and re.search(r"(?m)^\s*1\.\s+", normalized):
        normalized = "### 작업 절차\n" + normalized

    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    normalized = re.sub(r"\bEFIM\b", "EFEM", normalized)
    return normalized


def _normalize_doc_type(doc_type: str | None) -> str:
    return normalize_doc_type(doc_type or "")


def _normalize_device_name(device_name: str | None) -> str:
    if not device_name:
        return ""
    # v3 stores underscored names (SUPRA_XP), v2 uses spaces (SUPRA XP)
    # Normalize underscores and hyphens to spaces for consistent matching
    return str(device_name).strip().lower().replace("_", " ").replace("-", " ")


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
    task_mode = str(state.get("task_mode") or "").strip().lower()
    if not task_mode:
        parsed_query = (
            state.get("parsed_query") if isinstance(state.get("parsed_query"), dict) else {}
        )
        inferred_mode = _infer_task_mode_from_doc_types(state.get("selected_doc_types"))
        if inferred_mode is None:
            inferred_mode = _infer_task_mode_from_doc_types(parsed_query.get("selected_doc_types"))
        if inferred_mode is not None:
            task_mode = inferred_mode

    if task_mode == "issue":
        pq_dict = dict(state.get("parsed_query") or {})
        pq_dict["route"] = "general"
        pq_dict["task_mode"] = "issue"
        return {"route": "general", "task_mode": "issue", "parsed_query": pq_dict}

    if task_mode == "ts":
        pq_dict = dict(state.get("parsed_query") or {})
        pq_dict["route"] = "ts"
        pq_dict["task_mode"] = "ts"
        return {"route": "ts", "task_mode": "ts", "parsed_query": pq_dict}

    # Use English query for routing (after translation)
    route_query = state.get("query_en") or state["query"]
    query = _normalize_query_text(route_query) or str(route_query).strip()
    user = _format_prompt(spec.router.user, {"sys.query": query})
    route = _parse_route(
        _invoke_llm(llm, spec.router.system, user, temperature=TEMP_CLASSIFICATION)
    )
    logger.info("route_node: query=%s..., route=%s", query[:50] if query else None, route)

    # --- Inquiry safe override (B + 일부 A) ---
    # SOP/Setup 선택 상태에서 라우터가 setup을 반환했지만, 실제 질문이 정보 조회
    # (worksheet, tool list, scope 등)인 경우 general로 오버라이드한다.
    # 단, procedure 키워드가 있으면 절차 의도 우선(procedure wins).
    if route == "setup" and task_mode in ("sop", ""):
        user_query_lower = (state["query"] or "").lower()
        _has_inquiry = any(kw in user_query_lower for kw in _INQUIRY_KEYWORDS)
        _has_procedure = any(kw in user_query_lower for kw in _PROCEDURE_KEYWORDS_SET)
        _has_compound_inquiry = any(cp in user_query_lower for cp in _INQUIRY_COMPOUND_PATTERNS)
        if _has_inquiry and (not _has_procedure or _has_compound_inquiry):
            logger.info(
                "route_node: inquiry override setup->general (query=%s)",
                user_query_lower[:50],
            )
            route = "general"

    # parsed_query 업데이트
    pq_dict = dict(state.get("parsed_query") or {})
    pq_dict["route"] = route

    return {"route": route, "parsed_query": pq_dict}


def mq_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    task_mode = str(state.get("task_mode") or "").strip().lower()
    # Generate MQ in both English and Korean for bilingual retrieval
    query_en = state.get("query_en") or state["query"]
    query_ko = state.get("query_ko") or state["query"]
    logger.info(
        "mq_node: bilingual - EN=%s..., KO=%s...",
        query_en[:40] if query_en else None,
        query_ko[:40] if query_ko else None,
    )

    if state.get("skip_mq") and state.get("search_queries"):
        logger.info("mq_node: search_queries override provided, skipping MQ generation")
        return {}

    setup_mq_list: List[str] = []
    ts_mq_list: List[str] = []
    general_mq_list: List[str] = []

    # MQ generation needs more tokens than classification
    mq_kwargs = {
        "max_tokens": 4096,
        "temperature": resolve_querygen_temperature(state, mq_invoked=True),
    }

    def _generate_mq_bilingual(spec_template) -> tuple[List[str], List[str]]:
        """Generate MQ in both English and Korean."""
        # English MQ - add explicit language instruction
        system_en = spec_template.system + "\n\n**IMPORTANT: Generate all queries in English.**"
        user_en = _format_prompt(spec_template.user, {"sys.query": query_en})
        raw_en = _invoke_llm(llm, system_en, user_en, **mq_kwargs)
        mq_en = _parse_queries(raw_en)
        logger.info("mq_node(%s/en): %d queries: %s", route, len(mq_en), mq_en)

        # Korean MQ - add explicit Korean language instruction
        system_ko = (
            spec_template.system
            + "\n\n**중요: 모든 검색어를 반드시 한국어로 생성하세요. Generate all queries in Korean.**"
        )
        user_ko = _format_prompt(spec_template.user, {"sys.query": query_ko})
        raw_ko = _invoke_llm(llm, system_ko, user_ko, **mq_kwargs)
        mq_ko = _parse_queries(raw_ko)
        logger.info("mq_node(%s/ko): %d queries: %s", route, len(mq_ko), mq_ko)

        return mq_en, mq_ko

    setup_mq_ko_list: List[str] = []
    ts_mq_ko_list: List[str] = []
    general_mq_ko_list: List[str] = []

    if task_mode == "issue":
        issue_mq_template = spec.issue_mq or spec.general_mq
        general_mq_list, general_mq_ko_list = _generate_mq_bilingual(issue_mq_template)
    elif route == "setup":
        setup_mq_list, setup_mq_ko_list = _generate_mq_bilingual(spec.setup_mq)
    elif route == "ts":
        ts_mq_list, ts_mq_ko_list = _generate_mq_bilingual(spec.ts_mq)
    else:
        general_mq_list, general_mq_ko_list = _generate_mq_bilingual(spec.general_mq)

    logger.info(
        "mq_node: total - setup(en=%d,ko=%d), ts(en=%d,ko=%d), general(en=%d,ko=%d)",
        len(setup_mq_list),
        len(setup_mq_ko_list),
        len(ts_mq_list),
        len(ts_mq_ko_list),
        len(general_mq_list),
        len(general_mq_ko_list),
    )

    return {
        "mq_used": True,
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
    gate = _parse_gate(_invoke_llm(llm, spec.st_gate.system, user, temperature=TEMP_CLASSIFICATION))
    return {"st_gate": gate}


def _is_garbage_query(q: str) -> bool:
    """Check if a query is garbage (prompt label leak or too short)."""
    normalized = _normalize_query_text(q)
    lower = normalized.lower().strip()
    if not lower:
        return True
    if _looks_like_placeholder_query(normalized):
        return True
    # Filter out prompt label leaks
    garbage_patterns = [
        "setup_mq:",
        "ts_mq:",
        "general_mq:",
        "gate:",
        "질문:",
        "queries:",
        "search_queries:",
        "[regenerate with",
    ]
    for pat in garbage_patterns:
        if pat in lower:
            return True
    # Filter out queries that are just labels without actual content
    if lower in {"setup_mq", "ts_mq", "general_mq", "gate", "no_st", "need_st"}:
        return True
    if re.fullmatch(r"[.\s…·•\-_=~]+", normalized):
        return True
    # Filter out very short queries (less than 3 chars)
    if len(normalized) < 3:
        return True
    return False


def _contains_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


def _fill_to_n(base: List[str], candidates: List[str], n: int) -> List[str]:
    for q in candidates:
        q = _normalize_query_text(q)
        if not q or _is_garbage_query(q):
            continue
        if q not in base:
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
    base_query_en = state.get("query_en") or state["query"]
    q_en = _normalize_query_text(base_query_en) or str(base_query_en).strip()
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

    def _translate(text: str, target_lang: str) -> str:
        if spec.translate is None:
            return text
        target_name = {"en": "English", "ko": "Korean"}.get(target_lang, target_lang)
        user = _format_prompt(
            spec.translate.user,
            {
                "query": text,
                "target_language": target_name,
            },
        )
        result = _invoke_llm(llm, spec.translate.system, user, temperature=TEMP_TRANSLATION)
        result = _normalize_query_text(result.strip().strip('"').strip("'").strip())
        return result if result else _normalize_query_text(text)

    used_existing_queries = bool(state.get("skip_mq") and state.get("search_queries"))
    if used_existing_queries:
        queries = [
            cleaned
            for q in state.get("search_queries", [])
            if (cleaned := _normalize_query_text(q)) and not _is_garbage_query(cleaned)
        ]
        logger.info("st_mq_node: using provided search_queries (skip_mq=True): %s", queries)
    else:
        mapping = {
            "sys.query": q_en,
            "setup_mq": "\n".join(setup_mq_list),
            "ts_mq": "\n".join(ts_mq_list),
            "general_mq": "\n".join(general_mq_list),
            "st_gate": state.get("st_gate", "no_st"),
        }
        user = _format_prompt(spec.st_mq.user, mapping)
        raw = _invoke_llm(
            llm,
            spec.st_mq.system,
            user,
            temperature=resolve_querygen_temperature(state, mq_invoked=True),
        )
        logger.info("st_mq_node: raw output=%s", raw)
        parsed = _parse_queries(raw)
        queries = [q for q in parsed if not _is_garbage_query(q)]

    english_queries: List[str] = []
    english_seed = [q for q in queries if not _contains_korean(q)]
    _fill_to_n(english_queries, english_seed, 3)
    if q_en and not _contains_korean(q_en):
        _fill_to_n(english_queries, [q_en], 3)
    _fill_to_n(english_queries, [q for q in mq_en_list if not _contains_korean(q)], 3)

    if len(english_queries) < 3:
        ko_seed_for_en = [q for q in queries if _contains_korean(q)]
        if q_ko and _contains_korean(q_ko):
            ko_seed_for_en = [q_ko] + ko_seed_for_en
        for q in _dedupe_queries(ko_seed_for_en):
            if len(english_queries) >= 3:
                break
            translated = _translate(q, "en")
            if (
                translated
                and not _contains_korean(translated)
                and not _is_garbage_query(translated)
            ):
                _fill_to_n(english_queries, [translated], 3)

    korean_queries: List[str] = []
    ko_seed = [q for q in queries if _contains_korean(q)]
    if q_ko and _contains_korean(q_ko):
        ko_seed = [q_ko] + ko_seed
    ko_seed.extend([q for q in mq_ko_list if _contains_korean(q)])
    _fill_to_n(korean_queries, ko_seed, 3)

    if len(korean_queries) < 3:
        for q in english_queries:
            if len(korean_queries) >= 3:
                break
            if _contains_korean(q):
                continue
            translated = _translate(q, "ko")
            if translated and _contains_korean(translated) and not _is_garbage_query(translated):
                _fill_to_n(korean_queries, [translated], 3)

    english_queries = english_queries[:3]
    korean_queries = korean_queries[:3]

    merged = english_queries + korean_queries

    # ── 동의어/약어 변형 쿼리 추가 (결정적 variant, max +2) ──
    if agent_settings.abbreviation_expand_enabled and not used_existing_queries:
        try:
            from backend.llm_infrastructure.query_expansion.abbreviation_expander import (
                get_abbreviation_expander,
            )

            _syn_expander = get_abbreviation_expander(agent_settings.abbreviation_dict_path)
            abbr_sels: Dict[str, str] = state.get("abbreviation_selections") or {}
            original_query = str(state.get("original_query") or state.get("query") or "")
            syn_variants = _syn_expander.get_synonym_variants(
                original_query,
                max_variants=2,
                abbr_selections=abbr_sels,
            )
            if syn_variants:
                logger.info(
                    "st_mq_node: synonym variants for '%s': %s",
                    original_query,
                    syn_variants,
                )
                merged.extend(syn_variants)
        except Exception:
            logger.debug("st_mq_node: synonym variant generation failed", exc_info=True)

    guardrail_result = (
        {
            "search_queries": merged,
            "guardrail_dropped_numeric": 0,
            "guardrail_dropped_anchor": 0,
            "guardrail_final_count": len(merged),
        }
        if used_existing_queries
        else validate_search_queries(state, merged)
    )
    final_queries = guardrail_result["search_queries"]
    logger.info("st_mq_node: final search_queries (bilingual/guardrailed)=%s", final_queries)

    # parsed_query 업데이트
    pq_dict = dict(state.get("parsed_query") or {})
    pq_dict["search_queries"] = final_queries

    return {
        "search_queries": final_queries,
        "parsed_query": pq_dict,
        "guardrail_dropped_numeric": int(guardrail_result["guardrail_dropped_numeric"]),
        "guardrail_dropped_anchor": int(guardrail_result["guardrail_dropped_anchor"]),
        "guardrail_final_count": int(guardrail_result["guardrail_final_count"]),
    }


def _apply_section_expansion(
    docs: List[RetrievalResult],
    es_engine: Any,
    query: str = "",
) -> List[RetrievalResult]:
    """Apply section expansion to retrieval results.

    For top groups with reliable section_chapter, fetch all section chunks
    and insert them into the result list at the trigger hit position.

    query is used for query-aware candidate scoring: candidates whose doc_id
    contains query terms are prioritised when selecting top_groups.
    """
    from backend.llm_infrastructure.retrieval.postprocessors.section_expander import (
        SectionExpander,
    )

    expander = SectionExpander.from_settings(rag_settings)
    if not expander.enabled:
        return docs

    # --- Collect ALL valid candidates with page-aware multi-SOP dedup ---
    # In multi-SOP documents, the same section_chapter (e.g. "3. Part 위치")
    # repeats at different page ranges for each SOP.  Pages >20 apart are
    # treated as distinct groups so each SOP gets its own expansion slot.
    _MULTI_SOP_PAGE_GAP = 20
    seen_group_pages: dict[tuple[str, str], list[int]] = {}
    all_candidates: list[tuple[int, RetrievalResult]] = []

    for idx, doc in enumerate(docs):
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        section_chapter = str(meta.get("section_chapter", "") or "")
        chapter_source = str(meta.get("chapter_source", "") or "")
        chapter_ok = bool(meta.get("chapter_ok", False))

        if not section_chapter or not chapter_ok or chapter_source not in expander.allowed_sources:
            continue

        page = int(meta.get("page") or 0)
        base_key = (str(doc.doc_id), section_chapter)

        # Page-aware dedup: same (doc_id, section_chapter) within PAGE_GAP = same SOP
        is_dup = False
        if base_key in seen_group_pages:
            for rep_page in seen_group_pages[base_key]:
                if abs(page - rep_page) < _MULTI_SOP_PAGE_GAP:
                    is_dup = True
                    break
        if is_dup:
            continue

        seen_group_pages.setdefault(base_key, []).append(page)
        all_candidates.append((idx, doc))

    # --- Query-aware scoring: prefer candidates whose doc_id matches query terms ---
    query_tokens = (
        [t for t in re.split(r"[\s,.\-_]+", query.lower()) if len(t) >= 2] if query else []
    )

    def _candidate_score(idx: int, doc: RetrievalResult) -> tuple[int, int]:
        """(negative query_match_score, retrieval_rank) — lower = better."""
        doc_id_lower = str(doc.doc_id).lower()
        q_score = sum(2 for tok in query_tokens if tok in doc_id_lower)
        return (-q_score, idx)

    all_candidates.sort(key=lambda pair: _candidate_score(pair[0], pair[1]))
    candidates = all_candidates[: expander.top_groups]

    if candidates:
        logger.info(
            "section_expansion: %d valid candidates, selected %d (query_tokens=%s, keys=%s)",
            len(all_candidates),
            len(candidates),
            query_tokens[:6],
            [
                (str(d.doc_id)[-30:], (d.metadata or {}).get("section_chapter", ""))
                for _, d in candidates
            ],
        )

    if not candidates:
        return docs

    # Cross-section rules: when a trigger section is hit, also fetch target sections
    # from the same doc_id (within page gap). Covers common SOP patterns where
    # non-procedural sections (Part 위치, Flow Chart) score high but lack procedure text.
    _CROSS_SECTION_TRIGGERS: dict[str, list[str]] = {
        "flow chart": ["Work Procedure"],
        "workflow": ["Work Procedure"],
        "part 위치": ["Work Procedure"],
        "location of parts": ["Work Procedure"],
    }

    # Fetch expanded chunks for each candidate group
    expanded_map: dict[int, list[RetrievalResult]] = {}
    for idx, doc in candidates:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        section_chapter = str(meta.get("section_chapter", "") or "")

        # Page-scoped expansion: in multi-SOP documents, the same section_chapter
        # name repeats at different page ranges.  Scope the fetch to the trigger
        # hit's page neighbourhood so we only get chunks from the correct SOP.
        trigger_page = int(meta.get("page") or 0)
        if trigger_page > 0:
            scope_min = max(0, trigger_page - _MULTI_SOP_PAGE_GAP)
            scope_max = trigger_page + _MULTI_SOP_PAGE_GAP
        else:
            scope_min = None
            scope_max = None

        section_hits = es_engine.fetch_section_chunks(
            doc_id=str(doc.doc_id),
            section_chapter=section_chapter,
            max_pages=expander.max_pages,
            content_index=None,
            min_page=scope_min,
            max_page=scope_max,
        )
        if section_hits:
            logger.info(
                "section_expansion: doc=%s section=%s trigger_page=%d scope=[%s,%s] fetched=%d chunks (pages=%s)",
                str(doc.doc_id)[-40:],
                section_chapter[:30],
                trigger_page,
                scope_min,
                scope_max,
                len(section_hits),
                [h.page for h in section_hits[:5]],
            )

        # Cross-section expansion: fetch nearby Work Procedure when a
        # non-procedural section (Flow Chart, Part 위치 등) is the trigger hit.
        # min_page filter ensures we only grab the Work Procedure belonging to
        # the SAME SOP within a multi-SOP physical document.
        _CROSS_SECTION_PAGE_GAP = 3
        section_lower = section_chapter.lower()
        for trigger, target_keywords in _CROSS_SECTION_TRIGGERS.items():
            if trigger in section_lower:
                trigger_max_page = max(
                    (h.page for h in (section_hits or []) if h.page is not None),
                    default=(doc.metadata or {}).get("page") or 0,
                )
                min_page = max(0, trigger_max_page - _CROSS_SECTION_PAGE_GAP)
                for kw in target_keywords:
                    cross_hits = es_engine.fetch_section_chunks_by_keyword(
                        doc_id=str(doc.doc_id),
                        keyword=kw,
                        max_pages=expander.max_pages,
                        content_index=None,
                        min_page=min_page,
                    )
                    if cross_hits:
                        first_page = min(
                            (h.page for h in cross_hits if h.page is not None),
                            default=None,
                        )
                        if (
                            first_page is not None
                            and first_page <= trigger_max_page + _CROSS_SECTION_PAGE_GAP
                        ):
                            section_hits = (section_hits or []) + cross_hits
                            logger.info(
                                "cross_section_expansion: %s (max_page=%d) -> %s (first_page=%d, %d chunks, min_page=%d) for doc %s",
                                section_chapter,
                                trigger_max_page,
                                kw,
                                first_page,
                                len(cross_hits),
                                min_page,
                                doc.doc_id,
                            )
                        else:
                            logger.debug(
                                "cross_section_expansion: skipped %s (first_page=%s, gap > %d) for doc %s",
                                kw,
                                first_page,
                                _CROSS_SECTION_PAGE_GAP,
                                doc.doc_id,
                            )
                break  # one trigger match is enough

        if section_hits:
            expanded_map[idx] = [hit.to_retrieval_result() for hit in section_hits]

    if not expanded_map:
        return docs

    # Build final result list: at each trigger position, insert expanded chunks
    result: list[RetrievalResult] = []
    seen_doc_ids_pages: set[tuple[str, Any]] = set()

    def _dedup_key(d: RetrievalResult) -> tuple[str, Any]:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        return (str(d.doc_id), meta.get("chunk_id") or meta.get("page"))

    for idx, doc in enumerate(docs):
        key = _dedup_key(doc)
        if key in seen_doc_ids_pages:
            continue

        if idx in expanded_map:
            # Insert expanded section chunks (includes trigger hit by page order)
            for expanded_doc in expanded_map[idx]:
                ekey = _dedup_key(expanded_doc)
                if ekey not in seen_doc_ids_pages:
                    seen_doc_ids_pages.add(ekey)
                    result.append(expanded_doc)
            # Ensure trigger hit is included
            if key not in seen_doc_ids_pages:
                seen_doc_ids_pages.add(key)
                result.append(doc)
        else:
            seen_doc_ids_pages.add(key)
            result.append(doc)

    logger.info(
        "section_expansion: expanded %d groups, %d -> %d docs",
        len(expanded_map),
        len(docs),
        len(result),
    )
    return result


def retrieve_node(
    state: AgentState,
    *,
    retriever: Retriever,
    reranker: Any = None,
    retrieval_top_k: int = 20,
    final_top_k: int = 20,
) -> Dict[str, Any]:
    """Retrieve documents and rerank.

    If devices are selected:
      - Search only with selected device filter (strict)

    If no devices selected:
      - Search without device filter
    """
    queries = [
        cleaned
        for q in state.get("search_queries", [state["query"]])
        if (cleaned := _normalize_query_text(q)) and not _is_garbage_query(cleaned)
    ]
    if not queries:
        fallback_query = _normalize_query_text(state.get("query_en") or state["query"])
        if fallback_query and not _is_garbage_query(fallback_query):
            queries = [fallback_query]

    # ── 약어 확장 (domain dictionary) ──
    # 1:1은 자동 치환, 1:N은 사용자가 선택한 결과(abbreviation_selections)를 적용
    if agent_settings.abbreviation_expand_enabled:
        try:
            from backend.llm_infrastructure.query_expansion.abbreviation_expander import (
                get_abbreviation_expander,
            )

            _abbr_expander = get_abbreviation_expander(agent_settings.abbreviation_dict_path)
            abbr_selections: Dict[str, str] = state.get("abbreviation_selections") or {}
            expanded_queries = []
            for q in queries:
                result = _abbr_expander.expand_query(q)
                expanded = result.expanded_query
                # 사용자가 선택한 모호 약어도 치환
                if abbr_selections and result.ambiguous:
                    import re as _re

                    for match in result.matches:
                        if not match.ambiguous:
                            continue
                        selected_eng = abbr_selections.get(match.abbr_key)
                        if not selected_eng:
                            continue
                        if selected_eng.lower() in expanded.lower():
                            continue
                        pattern = _re.compile(
                            rf"\b{_re.escape(match.token)}\b",
                            _re.IGNORECASE,
                        )
                        replacement = f"{match.token} ({selected_eng})"
                        new_expanded = pattern.sub(replacement, expanded, count=1)
                        if new_expanded != expanded:
                            expanded = new_expanded
                            logger.info(
                                "Abbreviation resolved (user): %s → %s | query: %s → %s",
                                match.token,
                                selected_eng,
                                q,
                                expanded,
                            )
                if result.auto_expanded:
                    logger.info(
                        "Abbreviation expanded: %s → %s (tokens: %s)",
                        q,
                        result.expanded_query,
                        result.auto_expanded,
                    )
                expanded_queries.append(expanded)
            queries = expanded_queries
        except Exception:
            logger.debug("Abbreviation expander not available", exc_info=True)

    # ParsedQuery 우선, 없으면 기존 필드 fallback
    pq_raw = state.get("parsed_query")
    if pq_raw:
        pq = ParsedQuery(**pq_raw)
        selected_devices = pq.selected_devices
        selected_equip_ids = pq.selected_equip_ids
        selected_doc_types = pq.selected_doc_types
        selected_doc_types_strict = pq.doc_types_strict
    else:
        selected_devices = state.get("selected_devices", [])
        selected_equip_ids = state.get("selected_equip_ids", [])
        selected_doc_types = state.get("selected_doc_types", [])
        selected_doc_types_strict = bool(state.get("selected_doc_types_strict"))
    selected_doc_ids = [str(x).strip() for x in state.get("selected_doc_ids", []) if str(x).strip()]
    if selected_doc_types_strict:
        selected_doc_type_filters = _dedupe_queries(
            [str(dt).strip() for dt in selected_doc_types if _normalize_doc_type(str(dt))]
        )
    else:
        selected_doc_type_filters = expand_doc_type_selection(selected_doc_types)
    original_query = _normalize_query_text(state["query"]) or state["query"]

    # search_queries from st_mq_node already contains EN+KO queries
    # No need to add bilingual queries here
    query_en = _normalize_query_text(state.get("query_en") or "")

    # Expand device aliases (e.g., SUPRA XP ↔ ZEDIUS XP, SUPRA V+ ↔ SUPRA Vplus)
    _device_aliases = agent_settings.device_aliases
    _alias_lower = {k.lower(): v for k, v in _device_aliases.items()}
    expanded_devices: List[str] = []
    for d in selected_devices:
        expanded_devices.append(d)
        aliases = _alias_lower.get(d.lower(), [])
        for alias in aliases:
            if alias not in expanded_devices:
                expanded_devices.append(alias)
    if len(expanded_devices) > len(selected_devices):
        logger.info(
            "retrieve_node: device alias expansion %s -> %s", selected_devices, expanded_devices
        )
        selected_devices = expanded_devices

    selected_device_set = {
        _normalize_device_name(d) for d in selected_devices if _normalize_device_name(d)
    }
    selected_equip_id_set = {
        _normalize_equip_id(eid)
        for eid in selected_equip_ids
        if _is_valid_equip_id(_normalize_equip_id(eid))
    }
    selected_doc_type_set = {
        _normalize_doc_type(dt) for dt in selected_doc_type_filters if _normalize_doc_type(dt)
    }
    # v3 stores canonical group names (sop, setup, ts, gcb, myservice).
    # Add group names so post-retrieval filtering accepts v3 doc_type values.
    for group_name, variants in DOC_TYPE_GROUPS.items():
        variant_set = {normalize_doc_type(v) for v in variants}
        if selected_doc_type_set & variant_set:
            selected_doc_type_set.add(normalize_doc_type(group_name))

    candidate_k = max(retrieval_top_k, final_top_k * 2, 20)

    all_docs: List[RetrievalResult] = []
    seen: set = set()

    def _normalize_meta_value(value: Any) -> str:
        return str(value).strip()

    def _stable_dedupe_key(doc: RetrievalResult) -> tuple[Any, ...]:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        doc_id = _normalize_meta_value(doc.doc_id)

        chunk_id = _normalize_meta_value(meta.get("chunk_id"))
        if chunk_id:
            return (doc_id, chunk_id)

        page = _normalize_meta_value(meta.get("page"))
        if page:
            return (doc_id, page)

        page_start = _normalize_meta_value(meta.get("page_start"))
        if page_start:
            return (doc_id, page_start)

        text = (doc.raw_text or doc.content or "").strip()
        if text:
            text_digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            return (doc_id, text_digest)

        return (doc_id,)

    def _stable_tie_break_key(doc: RetrievalResult) -> tuple[float, str, int, int, int, str]:
        def _parse_int_like(value: Any) -> int | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.isdigit():
                    return int(stripped)
            return None

        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        doc_id = _normalize_meta_value(doc.doc_id)

        chunk_id = _normalize_meta_value(meta.get("chunk_id"))
        if chunk_id:
            secondary_rank = 0
            secondary_value = chunk_id
            secondary_numeric_rank = 1
            secondary_numeric_value = 0
        else:
            page = _normalize_meta_value(meta.get("page"))
            if page:
                secondary_rank = 1
                secondary_value = page
                parsed_page = _parse_int_like(meta.get("page"))
                if parsed_page is None:
                    secondary_numeric_rank = 1
                    secondary_numeric_value = 0
                else:
                    secondary_numeric_rank = 0
                    secondary_numeric_value = parsed_page
            else:
                page_start = _normalize_meta_value(meta.get("page_start"))
                if page_start:
                    secondary_rank = 2
                    secondary_value = page_start
                    parsed_page_start = _parse_int_like(meta.get("page_start"))
                    if parsed_page_start is None:
                        secondary_numeric_rank = 1
                        secondary_numeric_value = 0
                    else:
                        secondary_numeric_rank = 0
                        secondary_numeric_value = parsed_page_start
                else:
                    secondary_rank = 3
                    secondary_value = ""
                    secondary_numeric_rank = 1
                    secondary_numeric_value = 0

        return (
            -float(doc.score),
            doc_id,
            secondary_rank,
            secondary_numeric_rank,
            secondary_numeric_value,
            secondary_value,
        )

    def _matches_doc_type(doc: RetrievalResult) -> bool:
        if not selected_doc_type_set:
            return True
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        doc_type = _normalize_doc_type(meta.get("doc_type"))
        return bool(doc_type) and doc_type in selected_doc_type_set

    # Token containment 매칭용: 선택된 장비명의 토큰 집합을 미리 계산
    # e.g. "geneva xp" → {"geneva", "xp"}
    _selected_device_token_sets: List[set[str]] = []
    for d in selected_device_set:
        tokens = set(d.split())
        if tokens:
            _selected_device_token_sets.append(tokens)

    def _matches_device(doc: RetrievalResult) -> bool:
        if not selected_device_set:
            return True
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        device_name = _normalize_device_name(meta.get("device_name"))
        if not device_name:
            return False
        # 1차: exact match (기존 방식)
        if device_name in selected_device_set:
            return True
        # 2차: token containment — 선택 장비 토큰이 모두 문서 장비명에 포함되면 매칭
        # e.g. selected="geneva xp" tokens={"geneva","xp"} ⊆ doc="geneva stp300 xp" → match
        doc_tokens = set(device_name.split())
        for sel_tokens in _selected_device_token_sets:
            if sel_tokens <= doc_tokens:  # subset check
                return True
        return False

    def _matches_equip_id(doc: RetrievalResult) -> bool:
        if not selected_equip_id_set:
            return True
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        equip_id = _normalize_equip_id(meta.get("equip_id"))
        return bool(equip_id) and equip_id in selected_equip_id_set

    def _add_docs(docs_to_add: List[RetrievalResult], *, filter_devices: bool) -> None:
        """Add docs to all_docs, avoiding duplicates and honoring filters."""
        for d in docs_to_add:
            if not _matches_doc_type(d):
                continue
            if filter_devices and not _matches_device(d):
                continue
            if not _matches_equip_id(d):
                continue
            key = _stable_dedupe_key(d)
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    # Use search_queries directly (already contains EN+KO from st_mq_node)
    all_queries = queries

    # --- Modification F: doc_id keyword boost for setup route ---
    # Extract meaningful tokens from the query to boost doc_ids containing
    # component keywords (e.g., "apc", "pdb") via ES wildcard should-clause.
    _route_for_boost = state.get("route", "general")
    doc_id_boost_tokens: list | None = None
    if _route_for_boost == "setup":
        doc_id_boost_tokens = _extract_doc_id_boost_tokens(original_query, selected_devices)
        if doc_id_boost_tokens:
            logger.info("retrieve_node: doc_id_boost_tokens=%s", doc_id_boost_tokens)

    if selected_devices:
        # Strict device filter: selected devices only
        logger.info(
            "retrieve_node: strict device-filtered search with devices=%s equip_ids=%s queries=%d",
            selected_devices,
            selected_equip_ids,
            len(all_queries),
        )
        for q in all_queries:
            device_docs = retriever.retrieve(
                q,
                top_k=candidate_k,
                device_names=selected_devices,
                equip_ids=selected_equip_ids,
                doc_types=selected_doc_type_filters,
                doc_id_boost_tokens=doc_id_boost_tokens,
            )
            _add_docs(device_docs, filter_devices=True)
        logger.info("retrieve_node: strict device-filtered search found %d docs", len(all_docs))

    else:
        # No device selection: search without filter
        logger.info(
            "retrieve_node: general search (no device filter), equip_ids=%s, queries=%d",
            selected_equip_ids,
            len(all_queries),
        )
        for q in all_queries:
            docs = retriever.retrieve(
                q,
                top_k=candidate_k,
                equip_ids=selected_equip_ids,
                doc_types=selected_doc_type_filters,
                doc_id_boost_tokens=doc_id_boost_tokens,
            )
            _add_docs(docs, filter_devices=False)

    logger.info("retrieve_node: collected %d unique docs before rerank", len(all_docs))

    sop_variants = {normalize_doc_type(v) for v in DOC_TYPE_GROUPS.get("SOP", [])}
    selected_doc_types_normalized = {
        normalize_doc_type(dt) for dt in selected_doc_types if normalize_doc_type(dt)
    }
    route = state.get("route", "general")
    sop_only_predicate = bool(
        state.get("sop_intent") is True
        or route == "setup"
        or bool(selected_doc_types_normalized.intersection(sop_variants))
    )

    # --- Intent-based gating for SOP adjustments ---
    # SOP 문서 선택 시에도, 비절차(조회) 질문이면 절차 편향 조정을 건너뛴다.
    query_lower_for_intent = (original_query or "").lower()
    has_procedure_intent_early = any(
        kw in query_lower_for_intent for kw in _PROCEDURE_KEYWORDS_SET
    )
    has_inquiry_intent = any(
        kw in query_lower_for_intent for kw in _INQUIRY_KEYWORDS
    )
    # Procedure wins: 절차 키워드가 있으면 inquiry 무시
    # 단, 복합 조회 패턴("작업 check sheet" 등)은 procedure wins를 무효화
    has_compound_inquiry = any(
        cp in query_lower_for_intent for cp in _INQUIRY_COMPOUND_PATTERNS
    )
    is_procedural_context = (
        route == "setup" or has_procedure_intent_early
    ) and not (has_inquiry_intent and (not has_procedure_intent_early or has_compound_inquiry))

    logger.info(
        "retrieve_node: intent gating — sop_pred=%s, procedural=%s, inquiry=%s, route=%s",
        sop_only_predicate, is_procedural_context, has_inquiry_intent, route,
    )

    def _apply_early_page_penalty(docs: List[RetrievalResult]) -> List[RetrievalResult]:
        penalty_max_page = int(agent_settings.early_page_penalty_max_page)
        penalty_factor = float(agent_settings.early_page_penalty_factor)
        penalized_docs: List[RetrievalResult] = []
        for doc in docs:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            page = _extract_page_value(metadata)
            if page is not None and page <= penalty_max_page:
                penalized_docs.append(
                    RetrievalResult(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        score=float(doc.score) * penalty_factor,
                        metadata=doc.metadata,
                        raw_text=doc.raw_text,
                    )
                )
            else:
                penalized_docs.append(doc)
        return sorted(penalized_docs, key=_stable_tie_break_key)

    if agent_settings.early_page_penalty_enabled and sop_only_predicate and is_procedural_context and all_docs:
        all_docs = _apply_early_page_penalty(all_docs)

    if selected_doc_ids:
        before = len(all_docs)
        selected_doc_id_set = set(selected_doc_ids)
        all_docs = [d for d in all_docs if str(d.doc_id) in selected_doc_id_set]
        logger.info("retrieve_node: filtered by selected_doc_ids %d -> %d", before, len(all_docs))

    if sop_only_predicate and is_procedural_context and all_docs:
        soft_boost_factor = float(agent_settings.sop_soft_boost_factor)
        boosted_docs: List[RetrievalResult] = []
        for doc in all_docs:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            if _normalize_doc_type(metadata.get("doc_type")) == "sop":
                boosted_docs.append(
                    RetrievalResult(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        score=float(doc.score) * soft_boost_factor,
                        metadata=doc.metadata,
                        raw_text=doc.raw_text,
                    )
                )
            else:
                boosted_docs.append(doc)
        all_docs = boosted_docs

    # --- Procedure boost & Scope penalty ---
    _PROCEDURE_KEYWORDS = {
        "교체",
        "절차",
        "작업",
        "방법",
        "replacement",
        "procedure",
        "how to",
        "install",
        "설치",
        "수리",
        "repair",
    }
    _PROCEDURE_CHAPTERS = {"work procedure", "flow chart", "work 절차"}
    _SCOPE_MARKERS = {"scope", "contents", "목차", "table of contents", "revision history"}

    query_lower = original_query.lower()
    has_procedure_intent = any(kw in query_lower for kw in _PROCEDURE_KEYWORDS)

    if sop_only_predicate and all_docs:
        proc_boost = (
            float(agent_settings.procedure_boost_factor)
            if agent_settings.procedure_boost_enabled
            else 1.0
        )
        scope_pen = (
            float(agent_settings.scope_penalty_factor)
            if agent_settings.scope_penalty_enabled
            else 1.0
        )
        # Scope penalty는 inquiry 의도가 아닐 때만 적용 (조회 질문에서 scope/목차 패널티 방지)
        effective_scope_pen = scope_pen if not has_inquiry_intent else 1.0
        should_apply = (has_procedure_intent and proc_boost != 1.0) or effective_scope_pen != 1.0

        if should_apply:
            adjusted_docs: List[RetrievalResult] = []
            proc_boosted = 0
            scope_penalized = 0
            for doc in all_docs:
                meta = doc.metadata if isinstance(doc.metadata, dict) else {}
                chapter = str(meta.get("section_chapter", "") or "").lower()
                content_preview = (doc.content or "")[:300].lower()
                score = float(doc.score)

                # Boost Work Procedure / Flow Chart pages when query has procedure intent
                if (
                    has_procedure_intent
                    and proc_boost != 1.0
                    and any(pc in chapter for pc in _PROCEDURE_CHAPTERS)
                ):
                    score *= proc_boost
                    proc_boosted += 1
                # Penalize Scope/Contents/TOC pages (low-value for procedure questions)
                # inquiry 의도일 때는 패널티 건너뜀
                elif agent_settings.scope_penalty_enabled and effective_scope_pen != 1.0:
                    is_scope = any(sm in content_preview for sm in _SCOPE_MARKERS)
                    page = _extract_page_value(meta)
                    # Also penalize chapter_ok=false non-procedure pages
                    if is_scope or (page is not None and page <= 1):
                        score *= effective_scope_pen
                        scope_penalized += 1

                adjusted_docs.append(
                    RetrievalResult(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        score=score,
                        metadata=doc.metadata,
                        raw_text=doc.raw_text,
                    )
                )
            all_docs = adjusted_docs
            logger.info(
                "retrieve_node: procedure_boost=%d scope_penalty=%d (query_procedure_intent=%s)",
                proc_boosted,
                scope_penalized,
                has_procedure_intent,
            )

    stage1_ranked_docs = sorted(all_docs, key=_stable_tie_break_key)[:candidate_k]

    stage2_enabled = bool(agent_settings.second_stage_doc_retrieve_enabled)
    stage2_doc_ids: List[str] = []
    if stage2_enabled:
        if selected_doc_ids:
            stage2_doc_ids = list(selected_doc_ids)
        else:
            seen_doc_ids: set[str] = set()
            for doc in stage1_ranked_docs:
                doc_id = str(doc.doc_id).strip()
                if not doc_id or doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                stage2_doc_ids.append(doc_id)
        max_doc_ids = max(0, int(agent_settings.second_stage_max_doc_ids))
        if max_doc_ids:
            stage2_doc_ids = stage2_doc_ids[:max_doc_ids]
        else:
            stage2_doc_ids = []

    stage2_docs: List[RetrievalResult] = []
    if stage2_enabled and stage2_doc_ids and all_queries:
        stage2_top_k = max(1, int(agent_settings.second_stage_top_k))
        stage2_result_lists: List[List[RetrievalResult]] = []
        for doc_id in stage2_doc_ids:
            for q in all_queries:
                stage2_kwargs: Dict[str, Any] = {
                    "top_k": stage2_top_k,
                    "equip_ids": selected_equip_ids,
                    "doc_types": selected_doc_type_filters,
                    "doc_ids": [doc_id],
                }
                if selected_devices:
                    stage2_kwargs["device_names"] = selected_devices
                per_call_results = retriever.retrieve(q, **stage2_kwargs)
                if (
                    agent_settings.early_page_penalty_enabled
                    and sop_only_predicate
                    and is_procedural_context
                    and per_call_results
                ):
                    per_call_results = _apply_early_page_penalty(per_call_results)
                stage2_result_lists.append(per_call_results)
        stage2_docs = merge_retrieval_result_lists_rrf(stage2_result_lists, k=60)

    all_docs = (
        merge_retrieval_results_rrf(stage1_ranked_docs, stage2_docs, k=60)
        if stage2_docs
        else stage1_ranked_docs
    )

    # Store all_docs before reranking for regeneration (up to retrieval_top_k)
    # Sort by score and take top retrieval_top_k for regeneration options
    all_docs_for_regen = sorted(all_docs, key=_stable_tie_break_key)[:retrieval_top_k]

    # Rerank if reranker is available
    # Use English query for reranking - cross-encoder models often work better with English
    rerank_query = query_en if query_en else original_query
    if reranker is not None and all_docs:
        logger.info(
            "retrieve_node: reranking %d docs to top %d (using query_en)",
            len(all_docs),
            final_top_k,
        )
        docs = reranker.rerank(rerank_query, all_docs, top_k=final_top_k)
    else:
        # No reranker: just take top final_top_k by score
        docs = sorted(all_docs, key=_stable_tie_break_key)[:final_top_k]

    # Score threshold: filter out docs below minimum score
    score_threshold = float(agent_settings.score_threshold)
    if score_threshold > 0.0 and docs:
        before_count = len(docs)
        docs = [d for d in docs if float(d.score) >= score_threshold]
        filtered_count = before_count - len(docs)
        if filtered_count > 0:
            logger.info(
                "retrieve_node: score_threshold=%.3f filtered %d/%d docs",
                score_threshold,
                filtered_count,
                before_count,
            )

    # Deduplicate by base doc_id (keep highest-scoring page per document)
    if agent_settings.dedupe_by_doc_id and docs:

        def _base_doc_id(doc_id: str) -> str:
            """Extract base doc_id by removing page/chunk suffix (#0010, :chunk_3, etc.)."""
            s = str(doc_id).strip()
            for sep in ("#", ":chunk_", ":"):
                idx = s.find(sep)
                if idx > 0:
                    return s[:idx]
            return s

        before_count = len(docs)
        seen_base: dict[str, int] = {}  # base_doc_id -> index in deduped list
        deduped: List[RetrievalResult] = []
        for doc in docs:
            base = _base_doc_id(doc.doc_id)
            if base not in seen_base:
                seen_base[base] = len(deduped)
                deduped.append(doc)
            # else: skip duplicate (first occurrence has highest score after sort)
        if len(deduped) < before_count:
            logger.info(
                "retrieve_node: dedupe_by_doc_id removed %d duplicates (%d -> %d)",
                before_count - len(deduped),
                before_count,
                len(deduped),
            )
        docs = deduped

    # --- Doc-type diversity quota (SOP + Setup 동시 선택 시) ---
    setup_variants = {normalize_doc_type(v) for v in DOC_TYPE_GROUPS.get("setup", [])}
    both_selected = bool(
        selected_doc_types_normalized.intersection(sop_variants)
        and selected_doc_types_normalized.intersection(setup_variants)
    )
    if agent_settings.doc_type_diversity_enabled and both_selected and len(docs) > 1:
        min_setup = agent_settings.doc_type_diversity_min_setup
        min_sop = agent_settings.doc_type_diversity_min_sop

        sop_docs: List[RetrievalResult] = []
        setup_docs: List[RetrievalResult] = []
        other_docs: List[RetrievalResult] = []
        for doc in docs:
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            dt_norm = normalize_doc_type(str(meta.get("doc_type", "")))
            if dt_norm in setup_variants:
                setup_docs.append(doc)
            elif dt_norm in sop_variants:
                sop_docs.append(doc)
            else:
                other_docs.append(doc)

        need_rebalance = (
            (len(setup_docs) < min_setup and setup_docs)
            or (len(sop_docs) < min_sop and sop_docs)
        )
        if need_rebalance:
            # 각 그룹에서 최소 쿼터만큼 확보, 나머지는 원래 score 순서로 채움
            guaranteed: List[RetrievalResult] = []
            guaranteed.extend(sop_docs[:min_sop])
            guaranteed.extend(setup_docs[:min_setup])
            guaranteed_ids = {id(d) for d in guaranteed}
            remaining = [d for d in docs if id(d) not in guaranteed_ids]
            max_fill = max(0, len(docs) - len(guaranteed))
            docs = guaranteed + remaining[:max_fill]
            logger.info(
                "retrieve_node: doc_type diversity rebalanced — sop=%d setup=%d other=%d (total=%d)",
                min(len(sop_docs), min_sop),
                min(len(setup_docs), min_setup),
                len(other_docs),
                len(docs),
            )

    # Section expansion: expand top groups by fetching full section chunks
    if rag_settings.section_expand_enabled and docs and hasattr(retriever, "es_engine"):
        docs = _apply_section_expansion(docs, retriever.es_engine, query=state.get("query", ""))

    logger.info(
        "retrieve_node: returning %d docs (all_docs_for_regen: %d)",
        len(docs),
        len(all_docs_for_regen),
    )
    mq_mode = str(state.get("mq_mode") or "")
    empty_retrieval_fallback = (
        mq_mode == "fallback" and not bool(state.get("mq_used", False)) and len(docs) == 0
    )

    # --- Debug events for workflow trace ---
    _retrieve_events: List[str] = []
    _retrieve_events.append(
        f"[retrieve] sop_pred={sop_only_predicate} procedural={is_procedural_context} "
        f"inquiry={has_inquiry_intent} route={route}"
    )
    if sop_only_predicate:
        _penalty_parts = []
        if agent_settings.early_page_penalty_enabled:
            _penalty_parts.append(f"early_page(factor={agent_settings.early_page_penalty_factor})")
        if agent_settings.scope_penalty_enabled:
            _penalty_parts.append(f"scope(factor={agent_settings.scope_penalty_factor})")
        if agent_settings.procedure_boost_enabled:
            _penalty_parts.append(f"proc_boost(factor={agent_settings.procedure_boost_factor})")
        _retrieve_events.append(f"[retrieve] penalties: {', '.join(_penalty_parts) or 'none'}")
    else:
        _retrieve_events.append("[retrieve] penalties: SKIPPED (sop_pred=False)")
    # Top docs summary
    _top_summary = []
    for _di, _d in enumerate(docs[:5]):
        _dm = _d.metadata if isinstance(_d.metadata, dict) else {}
        _top_summary.append(
            f"#{_di+1} p={_dm.get('page','?')} ch='{_dm.get('section_chapter','')}' "
            f"s={float(_d.score):.4f} {str(_d.doc_id)[:50]}"
        )
    if _top_summary:
        _retrieve_events.append("[retrieve] top docs: " + " | ".join(_top_summary))

    return {
        "docs": docs,
        "ref_json": results_to_ref_json(docs),
        "all_docs": all_docs_for_regen,  # 재생성용 전체 문서 (최대 retrieval_top_k개)
        "mq_reason": "empty_retrieval" if empty_retrieval_fallback else state.get("mq_reason"),
        "retrieval_stage2": {
            "enabled": stage2_enabled,
            "doc_ids": stage2_doc_ids,
        },
        "_events": _retrieve_events,
    }


def expand_related_docs_node(
    state: AgentState,
    *,
    page_fetcher: Any = None,
    doc_fetcher: Any = None,
    section_fetcher: Any = None,
    chapter_resolver: Any = None,
    page_window: int = RELATED_PAGE_WINDOW,
    max_ref_chars: int = MAX_REF_CHARS_ANSWER,
) -> Dict[str, Any]:
    """Expand answer references by doc_type rules."""
    if page_fetcher is None and doc_fetcher is None and section_fetcher is None:
        msg = "expand_related: skipped (no fetcher available)"
        logger.info(msg)
        return {"_events": [msg]}

    docs = state.get("docs", [])
    if not docs:
        msg = "expand_related: skipped (no docs)"
        logger.info(msg)
        return {"_events": [msg]}

    expanded_docs: List[RetrievalResult] = []
    # Use expand_top_k from state when explicitly set (including 0), otherwise default.
    expand_top_k_raw = state.get("expand_top_k")
    expand_top_k = EXPAND_TOP_K if expand_top_k_raw is None else expand_top_k_raw
    max_expand = max(0, int(expand_top_k))
    total_docs = len(docs)
    same_doc_targets = 0
    page_targets = 0
    section_targets = 0
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
        elif (
            section_fetcher is not None
            and str(meta.get("section_chapter", "") or "")
            and str(meta.get("chapter_source", "") or "") in {"title", "rule", "toc_match", "carry"}
        ):
            # Chapter-based expansion: fetch all pages in the same section
            section_targets += 1
            section_chapter = str(meta.get("section_chapter", ""))
            _ch_src = str(meta.get("chapter_source", ""))
            if _ch_src == "carry":
                logger.warning(
                    "[expand_related] CARRY-TRIGGERED section expansion: "
                    "doc_id=%s, page=%s, chapter='%s', chapter_source=%s",
                    doc.doc_id, _extract_page_value(meta), section_chapter, _ch_src,
                )
            section_hits = section_fetcher(
                doc_id=str(doc.doc_id),
                section_chapter=section_chapter,
                max_pages=20,
            )
            if section_hits:
                all_section_docs = [
                    hit.to_retrieval_result() if hasattr(hit, "to_retrieval_result") else hit
                    for hit in section_hits
                ]
                # 연속 페이지 그룹만 유지: 같은 chapter라도 비연속이면 분리
                hit_page = _extract_page_value(meta)
                all_section_pages = sorted(
                    set(
                        p
                        for rd in all_section_docs
                        for p in [
                            _extract_page_value(
                                rd.metadata if isinstance(rd.metadata, dict) else {}
                            )
                        ]
                        if p is not None
                    )
                )
                # hit_page를 포함하는 연속 구간만 선택
                contiguous_group: List[int] = []
                if hit_page is not None and all_section_pages:
                    current_group: List[int] = [all_section_pages[0]]
                    for i in range(1, len(all_section_pages)):
                        if all_section_pages[i] - all_section_pages[i - 1] <= 3:
                            current_group.append(all_section_pages[i])
                        else:
                            if hit_page in current_group:
                                break
                            current_group = [all_section_pages[i]]
                    if hit_page in current_group:
                        contiguous_group = current_group
                allowed_pages = (
                    set(contiguous_group) if contiguous_group else set(all_section_pages)
                )
                related_docs = [
                    rd
                    for rd in all_section_docs
                    if _extract_page_value(rd.metadata if isinstance(rd.metadata, dict) else {})
                    in allowed_pages
                ]
                expanded_pages = sorted(allowed_pages)
            elif page_fetcher is not None:
                # Fallback to page window if section fetch fails
                page = _extract_page_value(meta)
                if page is not None:
                    page_targets += 1
                    section_targets -= 1
                    page_min = max(1, page - page_window)
                    page_max = page + page_window
                    pages = list(range(page_min, page_max + 1))
                    expanded_pages = pages
                    related_docs = page_fetcher(doc.doc_id, pages)
        elif chapter_resolver is not None:
            # Fallback: section_chapter is empty — resolve from neighbor pages
            page = _extract_page_value(meta)
            if page is not None:
                section_targets += 1
                resolved_hits = chapter_resolver(
                    doc_id=str(doc.doc_id),
                    hit_page=page,
                    max_pages=20,
                )
                if resolved_hits:
                    related_docs = [
                        hit.to_retrieval_result() if hasattr(hit, "to_retrieval_result") else hit
                        for hit in resolved_hits
                    ]
                    expanded_pages = sorted(
                        set(
                            p
                            for rd in related_docs
                            for p in [
                                _extract_page_value(
                                    rd.metadata if isinstance(rd.metadata, dict) else {}
                                )
                            ]
                            if p is not None
                        )
                    )
                    logger.info(
                        "[expand_related] chapter_resolver fallback: doc_id=%s, "
                        "hit_page=%d, resolved %d pages",
                        doc.doc_id, page, len(expanded_pages),
                    )
                elif page_fetcher is not None:
                    # chapter_resolver returned nothing, fall back to page window
                    page_targets += 1
                    section_targets -= 1
                    page_min = max(1, page - page_window)
                    page_max = page + page_window
                    pages = list(range(page_min, page_max + 1))
                    expanded_pages = pages
                    related_docs = page_fetcher(doc.doc_id, pages)
            else:
                skipped_targets += 1
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
        "section_targets=%d page_targets=%d skipped_targets=%d fetched_related=%d expanded=%d"
        % (
            total_docs,
            min(total_docs, max_expand),
            same_doc_targets,
            section_targets,
            page_targets,
            skipped_targets,
            fetched_related_total,
            expanded_count,
        )
    )
    logger.info(summary)

    display_docs = _merge_display_docs(expanded_docs[: min(total_docs, max_expand)])
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


def ask_user_after_retrieve_node(
    state: AgentState,
) -> Command[Literal["expand_related", "refine_and_retrieve"]]:
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
                valid_queries = [str(q).strip() for q in modified_queries if str(q).strip()]

                if valid_queries:
                    return Command(
                        goto="refine_and_retrieve",
                        update={
                            "retrieval_confirmed": False,
                            "user_feedback": f"Modified queries: {', '.join(valid_queries)}",
                            "search_queries": valid_queries[:5],
                        },
                    )
                else:
                    # Empty queries - fall back to refine_and_retrieve
                    return Command(
                        goto="refine_and_retrieve",
                        update={
                            "retrieval_confirmed": False,
                            "user_feedback": "Empty queries provided",
                        },
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
                selected_docs.extend([d for idx, d in enumerate(docs, start=1) if idx in rank_set])

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
            goto="expand_related", update={"retrieval_confirmed": True, "user_feedback": None}
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
            },
        )

    # 거절: False
    return Command(
        goto="refine_and_retrieve", update={"retrieval_confirmed": False, "user_feedback": None}
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
        "zh": {
            "setup": spec.setup_ans_zh,
            "ts": spec.ts_ans_zh,
            "general": spec.general_ans_zh,
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


def _get_issue_answer_template(
    spec: PromptSpec,
    language: Optional[str],
    *,
    detail: bool,
) -> PromptTemplate:
    issue_ans = getattr(spec, "issue_ans", None)
    issue_detail_ans = getattr(spec, "issue_detail_ans", None)
    issue_ans_en = getattr(spec, "issue_ans_en", None)
    issue_ans_zh = getattr(spec, "issue_ans_zh", None)
    issue_ans_ja = getattr(spec, "issue_ans_ja", None)
    issue_detail_ans_en = getattr(spec, "issue_detail_ans_en", None)
    issue_detail_ans_zh = getattr(spec, "issue_detail_ans_zh", None)
    issue_detail_ans_ja = getattr(spec, "issue_detail_ans_ja", None)

    if detail:
        default_template = issue_detail_ans or spec.general_ans
        lang_templates = {
            "en": issue_detail_ans_en,
            "zh": issue_detail_ans_zh,
            "ja": issue_detail_ans_ja,
        }
    else:
        default_template = issue_ans or spec.general_ans
        lang_templates = {
            "en": issue_ans_en,
            "zh": issue_ans_zh,
            "ja": issue_ans_ja,
        }

    if language and language in lang_templates and lang_templates[language] is not None:
        return lang_templates[language]  # type: ignore[return-value]
    return default_template


def _extract_doc_type_from_metadata(metadata: Dict[str, Any]) -> str:
    return _normalize_doc_type(metadata.get("doc_type"))


def _extract_section_from_metadata(metadata: Dict[str, Any]) -> str:
    section = (
        metadata.get("section")
        or metadata.get("section_type")
        or metadata.get("section_chapter")
        or metadata.get("chapter")
    )
    text = str(section or "").strip().lower()
    return text if text else "unknown"


def _compute_issue_routing_signals(docs: List[RetrievalResult]) -> Dict[str, Any]:
    if not docs:
        return {
            "issue_signal_k_effective": 0,
            "score_gap_12": 0.0,
            "myservice_share_50": 0.0,
            "gcb_count_50": 0,
            "ts_count_50": 0,
            "non_myservice_presence_50": 0,
            "doc_type_entropy_20": 0.0,
            "gcb_chapter_coverage_10": 0.0,
            "recentness_ratio_180d": None,
        }

    top50 = docs[: min(50, len(docs))]
    top20 = docs[: min(20, len(docs))]
    top10 = docs[: min(10, len(docs))]

    score_1 = float(top50[0].score) if top50 else 0.0
    score_2 = float(top50[1].score) if len(top50) >= 2 else score_1
    score_gap_12 = (score_1 - score_2) / max(abs(score_1), 1e-9)

    count_myservice = 0
    count_gcb = 0
    count_ts = 0
    for doc in top50:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        doc_type = _extract_doc_type_from_metadata(meta)
        if doc_type == "myservice":
            count_myservice += 1
        elif doc_type == "gcb":
            count_gcb += 1
        elif doc_type == "ts":
            count_ts += 1

    k_effective = len(top50)
    myservice_share_50 = count_myservice / k_effective if k_effective else 0.0
    non_myservice_presence_50 = count_gcb + count_ts

    # Normalized entropy over doc_type distribution in top-20.
    counts_20: Dict[str, int] = {}
    for doc in top20:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        doc_type = _extract_doc_type_from_metadata(meta) or "unknown"
        counts_20[doc_type] = counts_20.get(doc_type, 0) + 1
    entropy_raw = 0.0
    total_20 = len(top20)
    if total_20 > 0:
        for count in counts_20.values():
            p = count / total_20
            if p > 0:
                entropy_raw -= p * math.log(p)
    entropy_denom = math.log(len(counts_20)) if len(counts_20) > 1 else 0.0
    doc_type_entropy_20 = entropy_raw / entropy_denom if entropy_denom > 0 else 0.0

    # Distinct section coverage ratio for gcb in top-10.
    gcb_sections: List[str] = []
    for doc in top10:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        if _extract_doc_type_from_metadata(meta) == "gcb":
            gcb_sections.append(_extract_section_from_metadata(meta))
    gcb_chapter_coverage_10 = len(set(gcb_sections)) / len(gcb_sections) if gcb_sections else 0.0

    return {
        "issue_signal_k_effective": k_effective,
        "score_gap_12": score_gap_12,
        "myservice_share_50": myservice_share_50,
        "gcb_count_50": count_gcb,
        "ts_count_50": count_ts,
        "non_myservice_presence_50": non_myservice_presence_50,
        "doc_type_entropy_20": doc_type_entropy_20,
        "gcb_chapter_coverage_10": gcb_chapter_coverage_10,
        "recentness_ratio_180d": None,
    }


def _resolve_issue_policy_tier(signals: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    k_effective = int(signals.get("issue_signal_k_effective") or 0)
    score_gap_12 = float(signals.get("score_gap_12") or 0.0)
    myservice_share_50 = float(signals.get("myservice_share_50") or 0.0)
    gcb_count_50 = int(signals.get("gcb_count_50") or 0)
    ts_count_50 = int(signals.get("ts_count_50") or 0)
    non_myservice_presence_50 = int(signals.get("non_myservice_presence_50") or 0)

    if k_effective < 20:
        if k_effective == 0:
            return "tier3", "empty_retrieval"
        return "tier2", "insufficient_signal_k"

    if (
        score_gap_12 >= 0.12
        and myservice_share_50 <= 0.70
        and (gcb_count_50 >= 3 or ts_count_50 >= 2)
    ):
        return "tier1", None

    if myservice_share_50 >= 0.85 and non_myservice_presence_50 == 0:
        return "tier3", "myservice_dominant_sparse_alt"

    return "tier2", None


def _doc_type_from_ref_item(ref: Dict[str, Any]) -> str:
    raw_meta = ref.get("metadata")
    metadata = raw_meta if isinstance(raw_meta, dict) else {}
    return _extract_doc_type_from_metadata(metadata)


def _select_issue_doc_ids_by_tier(
    ref_items: List[Dict[str, Any]],
    *,
    tier: str,
    max_docs: int,
) -> List[str]:
    ordered_doc_types: Dict[str, str] = {}
    for item in ref_items:
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id or doc_id in ordered_doc_types:
            continue
        ordered_doc_types[doc_id] = _doc_type_from_ref_item(item)

    ordered_doc_ids = list(ordered_doc_types.keys())
    if not ordered_doc_ids:
        return []

    if tier == "tier3":
        return ordered_doc_ids[:max_docs]

    selected: List[str] = []
    myservice_cap = 2 if tier == "tier1" else 5
    myservice_count = 0
    has_non_myservice = False

    if tier == "tier1":
        for doc_id in ordered_doc_ids:
            if ordered_doc_types[doc_id] != "myservice":
                selected.append(doc_id)
                has_non_myservice = True
                if len(selected) >= max_docs:
                    return selected

        for doc_id in ordered_doc_ids:
            if ordered_doc_types[doc_id] == "myservice" and myservice_count < myservice_cap:
                selected.append(doc_id)
                myservice_count += 1
                if len(selected) >= max_docs:
                    return selected

        return selected[:max_docs] if selected else ordered_doc_ids[:max_docs]

    # tier2
    for doc_id in ordered_doc_ids:
        doc_type = ordered_doc_types[doc_id]
        if doc_type == "myservice":
            if myservice_count >= myservice_cap:
                continue
            myservice_count += 1
        else:
            has_non_myservice = True
        selected.append(doc_id)
        if len(selected) >= max_docs:
            break

    if not has_non_myservice:
        for doc_id in ordered_doc_ids:
            if ordered_doc_types[doc_id] != "myservice" and doc_id not in selected:
                if len(selected) < max_docs:
                    selected.append(doc_id)
                else:
                    for idx in range(len(selected) - 1, -1, -1):
                        if ordered_doc_types.get(selected[idx]) == "myservice":
                            selected[idx] = doc_id
                            break
                break

    return selected[:max_docs] if selected else ordered_doc_ids[:max_docs]


def _build_refs_for_selected_doc_ids(
    ref_items: List[Dict[str, Any]],
    selected_doc_ids: List[str],
    *,
    max_refs: int,
) -> List[Dict[str, Any]]:
    if not ref_items or not selected_doc_ids:
        return []
    selected_set = set(selected_doc_ids)
    out: List[Dict[str, Any]] = []
    for item in ref_items:
        doc_id = str(item.get("doc_id") or "").strip()
        if doc_id in selected_set:
            out.append(item)
            if len(out) >= max_refs:
                break
    return out


def _select_issue_answer_refs(
    case_refs: List[Dict[str, Any]],
    *,
    tier: str,
    max_refs: int,
) -> List[Dict[str, Any]]:
    if not case_refs:
        return []
    if tier == "tier3":
        return case_refs[:max_refs]

    myservice_cap = 1 if tier == "tier1" else 2
    myservice_count = 0
    selected: List[Dict[str, Any]] = []
    selected_ids: set[int] = set()

    # First pass: one representative ref per doc to maximize diversity.
    seen_doc_ids: set[str] = set()
    representative_refs: List[Tuple[int, Dict[str, Any]]] = []
    for idx, ref in enumerate(case_refs):
        doc_id = str(ref.get("doc_id") or "").strip()
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        representative_refs.append((idx, ref))

    for idx, ref in representative_refs:
        doc_type = _doc_type_from_ref_item(ref)
        if doc_type == "myservice":
            if myservice_count >= myservice_cap:
                continue
            myservice_count += 1
        selected.append(ref)
        selected_ids.add(idx)
        if len(selected) >= max_refs:
            break

    # Fill remaining slots from the original list while respecting myservice caps.
    if len(selected) < max_refs:
        for idx, ref in enumerate(case_refs):
            if idx in selected_ids:
                continue
            doc_type = _doc_type_from_ref_item(ref)
            if doc_type == "myservice" and myservice_count >= myservice_cap:
                continue
            if doc_type == "myservice":
                myservice_count += 1
            selected.append(ref)
            if len(selected) >= max_refs:
                break

    if not selected:
        return case_refs[:max_refs]
    return selected[:max_refs]


def _resolve_issue_rollout_phase(state: AgentState) -> int:
    raw = state.get("issue_policy_rollout_phase")
    if raw is None:
        raw = os.getenv("ISSUE_POLICY_ROLLOUT_PHASE", str(ISSUE_POLICY_ROLLOUT_PHASE_DEFAULT))
    try:
        phase = int(raw)
    except (TypeError, ValueError):
        phase = ISSUE_POLICY_ROLLOUT_PHASE_DEFAULT
    return max(ISSUE_POLICY_ROLLOUT_PHASE_MIN, min(ISSUE_POLICY_ROLLOUT_PHASE_MAX, phase))


def _stable_issue_nonce(state: AgentState, *, kind: str, extra: str = "") -> str:
    parts = [
        kind,
        str(state.get("thread_id") or ""),
        str(state.get("query") or ""),
        str(state.get("issue_stage") or ""),
        str(state.get("issue_selected_doc_id") or ""),
        extra,
    ]
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _build_issue_cases(
    ref_items: List[Dict[str, Any]], *, max_cases: int = 10
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for idx, ref in enumerate(ref_items, start=1):
        raw_doc_id = str(ref.get("doc_id") or "").strip()
        if not raw_doc_id or raw_doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(raw_doc_id)
        title = str(ref.get("title") or raw_doc_id).strip() or raw_doc_id
        content = str(ref.get("content") or "").strip()
        summary = content[:240] if content else ""
        cases.append(
            {
                "index": len(cases) + 1,
                "doc_id": raw_doc_id,
                "title": title,
                "summary": summary,
            }
        )
        if len(cases) >= max_cases:
            break
    return cases


def _build_issue_case_ref_map(
    ref_items: List[Dict[str, Any]],
    *,
    max_docs: int = MAX_ISSUE_CASE_MAP_DOCS,
    max_refs_per_doc: int = MAX_ISSUE_CASE_MAP_REFS_PER_DOC,
    max_content_chars: int = MAX_ISSUE_CASE_MAP_CONTENT_CHARS,
) -> Dict[str, List[Dict[str, Any]]]:
    case_ref_map: Dict[str, List[Dict[str, Any]]] = {}
    for ref in ref_items:
        raw_doc_id = str(ref.get("doc_id") or "").strip()
        if not raw_doc_id:
            continue
        if raw_doc_id not in case_ref_map and len(case_ref_map) >= max_docs:
            continue

        bucket = case_ref_map.setdefault(raw_doc_id, [])
        if len(bucket) >= max_refs_per_doc:
            continue

        copied = dict(ref)
        # Always deep-copy metadata to prevent shared-reference mutations
        raw_meta = copied.get("metadata")
        meta_copy: Dict[str, Any] = (
            {str(k): v for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}
        )
        content = str(copied.get("content") or "")
        if len(content) > max_content_chars:
            copied["content"] = content[:max_content_chars]
            meta_copy["truncated"] = True
        copied["metadata"] = meta_copy

        bucket.append(copied)
    return case_ref_map


def _ensure_issue_detail_sections(answer: str, language: str) -> str:
    if str(language).strip().lower() != "ko":
        return answer

    text = (answer or "").strip()
    has_issue = "## 이슈 내용" in text
    has_solution = "## 해결 방안" in text
    if has_issue and has_solution:
        return text

    body = text or "(내용 없음)"
    if not has_issue and not has_solution:
        return f"## 이슈 내용\n{body}\n\n## 해결 방안\n(REFS 기반 해결 방안을 확인하세요.)"
    if not has_issue and has_solution:
        return f"## 이슈 내용\n(원문을 참고하세요.)\n\n{text}"
    return f"{text}\n\n## 해결 방안\n(원문을 참고하세요.)"


def answer_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    answer_language = state.get("target_language") or state.get("detected_language") or "ko"
    task_mode = str(state.get("task_mode") or "").strip().lower()
    ref_items = state.get("answer_ref_json") or state.get("ref_json", [])

    if task_mode == "issue":
        issue_case_refs_raw = (
            state.get("issue_case_refs")
            or state.get("answer_ref_json")
            or state.get("ref_json", [])
        )
        if len(issue_case_refs_raw) > MAX_ISSUE_REFS:
            issue_case_refs_raw = issue_case_refs_raw[:MAX_ISSUE_REFS]

        baseline_answer_refs = state.get("answer_ref_json") or state.get("ref_json", [])
        if len(baseline_answer_refs) > MAX_ANSWER_REFS:
            baseline_answer_refs = baseline_answer_refs[:MAX_ANSWER_REFS]
        if not baseline_answer_refs and issue_case_refs_raw:
            baseline_answer_refs = issue_case_refs_raw[:MAX_ANSWER_REFS]

        signal_docs_raw = state.get("all_docs") or state.get("docs") or []
        signal_docs = [doc for doc in signal_docs_raw if isinstance(doc, RetrievalResult)]
        issue_routing_signals = _compute_issue_routing_signals(signal_docs)
        issue_policy_tier_shadow, issue_fallback_reason = _resolve_issue_policy_tier(
            issue_routing_signals
        )

        shadow_doc_ids = _select_issue_doc_ids_by_tier(
            issue_case_refs_raw,
            tier=issue_policy_tier_shadow,
            max_docs=MAX_ISSUE_REFS,
        )
        issue_case_refs_shadow = _build_refs_for_selected_doc_ids(
            issue_case_refs_raw,
            shadow_doc_ids,
            max_refs=MAX_ISSUE_REFS,
        )
        if not issue_case_refs_shadow:
            issue_case_refs_shadow = issue_case_refs_raw[:MAX_ISSUE_REFS]

        issue_answer_refs_shadow = _select_issue_answer_refs(
            issue_case_refs_shadow,
            tier=issue_policy_tier_shadow,
            max_refs=MAX_ANSWER_REFS,
        )
        if not issue_answer_refs_shadow and issue_case_refs_shadow:
            issue_answer_refs_shadow = issue_case_refs_shadow[:MAX_ANSWER_REFS]

        rollout_phase = _resolve_issue_rollout_phase(state)
        live_policy_enabled = rollout_phase >= ISSUE_POLICY_ROLLOUT_PHASE_MAX

        issue_case_refs = issue_case_refs_shadow if live_policy_enabled else issue_case_refs_raw
        answer_refs = issue_answer_refs_shadow if live_policy_enabled else baseline_answer_refs
        issue_policy_tier = issue_policy_tier_shadow if live_policy_enabled else "baseline"

        if not answer_refs and issue_case_refs:
            answer_refs = issue_case_refs[:MAX_ANSWER_REFS]

        if not answer_refs:
            return {
                "answer": ISSUE_CASE_EMPTY_MESSAGE,
                "issue_top10_cases": [],
                "issue_stage": None,
                "issue_routing_signals": issue_routing_signals,
                "issue_policy_tier": issue_policy_tier,
                "issue_policy_tier_shadow": issue_policy_tier_shadow,
                "issue_case_refs_shadow": issue_case_refs_shadow,
                "issue_answer_refs_shadow": issue_answer_refs_shadow,
                "issue_policy_rollout_phase": rollout_phase,
                "issue_fallback_reason": issue_fallback_reason,
            }

        query_for_prompt = state.get("query_en") if answer_language == "en" else state.get("query")
        if not query_for_prompt:
            query_for_prompt = state.get("query", "")

        ref_text = ref_json_to_text(answer_refs)
        mapping = {"sys.query": query_for_prompt, "ref_text": ref_text}
        tmpl = _get_issue_answer_template(spec, answer_language, detail=False)
        user = _format_prompt(tmpl.user, mapping)
        answer_temperature = resolve_answer_temperature(route)
        answer, reasoning = _invoke_llm_with_reasoning(
            llm,
            tmpl.system,
            user,
            max_tokens=MAX_TOKENS_ANSWER,
            temperature=answer_temperature,
        )
        answer = _truncate_repetition(answer)

        issue_case_ref_map = _build_issue_case_ref_map(issue_case_refs)
        issue_case_ref_map_shadow = _build_issue_case_ref_map(issue_case_refs_shadow)
        return {
            "answer": answer,
            "reasoning": reasoning,
            "answer_ref_json": answer_refs,
            "issue_case_refs": issue_case_refs,
            "issue_case_ref_map": issue_case_ref_map,
            "issue_routing_signals": issue_routing_signals,
            "issue_policy_tier": issue_policy_tier,
            "issue_policy_tier_shadow": issue_policy_tier_shadow,
            "issue_case_refs_shadow": issue_case_refs_shadow,
            "issue_answer_refs_shadow": issue_answer_refs_shadow,
            "issue_case_ref_map_shadow": issue_case_ref_map_shadow,
            "issue_policy_rollout_phase": rollout_phase,
            "issue_fallback_reason": issue_fallback_reason,
            "issue_top10_cases": _build_issue_cases(issue_case_refs),
            "issue_stage": "post_summary",
        }

    if route == "setup":
        ref_items = _prioritize_setup_answer_refs(ref_items)

    # Use original (pre-abbreviation-expansion) query for the prompt so the
    # LLM sees natural phrasing (e.g. "APC valve") instead of the expanded
    # form ("APC (Automated Process Control) valve") which can confuse it
    # into thinking REFS don't match.
    if answer_language == "en":
        query_for_prompt = state.get("query_en") or state["query"]
    else:
        query_for_prompt = state.get("original_query") or state["query"]

    # Modification D (approach 2): normalize device name variant in query
    # using the canonical name already detected by fuzzy matching in auto_parse.
    # e.g. "SUPRAvvplus APC 교체 방법" → "SUPRA Vplus APC 교체 방법"
    if route == "setup":
        _pq_raw = state.get("parsed_query")
        _canonical_devices = (_pq_raw.get("selected_devices") or []) if isinstance(_pq_raw, dict) else []
        if _canonical_devices:
            _original_qfp = query_for_prompt
            query_for_prompt = _normalize_device_in_query(query_for_prompt, _canonical_devices[0])
            if query_for_prompt != _original_qfp:
                logger.info(
                    "answer_node: device name normalized in query: %r → %r",
                    _original_qfp, query_for_prompt,
                )

    # Setup route: (doc_id, section)별 그룹핑 → 적합성 판정 → 선택 → REFS 제한
    # MAX_ANSWER_REFS를 그룹 선택 이후에 적용하여, 적합한 section이
    # 순위가 낮더라도 relevance check에 참여할 수 있도록 함.
    doc_groups: list = []
    is_fallback_selection = False
    if route == "setup" and ref_items:
        doc_groups = _group_refs_by_doc_section_chapter(ref_items)

        # Query-aware group ordering: prefer groups whose doc_id contains
        # query tokens so that relevance checks are tried in priority order.
        _query_lower = query_for_prompt.lower()
        _q_tokens = [t for t in re.split(r"[\s,.\-_]+", _query_lower) if len(t) >= 2]

        def _group_query_score(item: Tuple[str, List]) -> Tuple[int, int]:
            gkey, grefs = item
            gkey_lower = gkey.lower()
            score = sum(2 for tok in _q_tokens if tok in gkey_lower)
            # Modification B: also sample content of top refs for keyword match
            content_sample = " ".join(
                str(r.get("content", ""))[:500].lower() for r in grefs[:3]
            )
            score += sum(1 for tok in _q_tokens if tok in content_sample)
            return (-score, 0)  # higher score first

        doc_groups.sort(key=_group_query_score)

        logger.info(
            "answer_node: setup doc_section_groups=%d keys=%s",
            len(doc_groups),
            [k[:60] for k, _ in doc_groups[:8]],
        )
        if len(doc_groups) > 1:
            import time as _time

            selected_refs = None
            fallback_refs = None
            fallback_group_key = None
            consecutive_empty = 0
            checks_done = 0
            for i, (group_key, group_refs) in enumerate(doc_groups[:MAX_SETUP_DOC_TRIES]):
                t0 = _time.time()
                group_text = ref_json_to_text(group_refs)
                relevant = _check_doc_relevance(query_for_prompt, group_text, llm=llm)
                elapsed = _time.time() - t0
                checks_done = i + 1
                logger.info(
                    "answer_node: setup relevance check %d/%d group=%s refs=%d relevant=%s (%.1fs)",
                    i + 1,
                    min(len(doc_groups), MAX_SETUP_DOC_TRIES),
                    group_key[:60],
                    len(group_refs),
                    relevant,
                    elapsed,
                )
                if relevant is None:
                    consecutive_empty += 1
                    if consecutive_empty >= CONSECUTIVE_EMPTY_LIMIT:
                        logger.warning(
                            "answer_node: %d consecutive empty responses, early-exit",
                            consecutive_empty,
                        )
                        break
                else:
                    consecutive_empty = 0
                    if relevant:
                        if len(group_refs) >= MIN_REFS_FOR_ACCEPT:
                            selected_refs = group_refs
                            break
                        elif fallback_refs is None:
                            fallback_refs = group_refs
                            fallback_group_key = group_key

            # 3단계 우선순위 선택
            if selected_refs is not None:
                ref_items = selected_refs
                is_fallback_selection = False
                logger.info(
                    "answer_node: setup selected group=%s (%d refs) after %d checks (direct)",
                    selected_refs[0].get("doc_id", "?")[:50],
                    len(selected_refs),
                    checks_done,
                )
            elif fallback_refs is not None:
                ref_items = fallback_refs
                is_fallback_selection = True
                logger.info(
                    "answer_node: setup fallback group=%s (%d refs) after %d checks",
                    fallback_group_key[:60] if fallback_group_key else "?",
                    len(fallback_refs),
                    checks_done,
                )
            else:
                ref_items = doc_groups[0][1]
                is_fallback_selection = True
                logger.info(
                    "answer_node: no relevant group found, fallback to first group=%s (%d refs)",
                    doc_groups[0][0][:60],
                    len(doc_groups[0][1]),
                )
        else:
            # 그룹이 1개면 그대로 사용
            ref_items = doc_groups[0][1] if doc_groups else ref_items

    # 답변 생성 시 REFS 수 제한 (setup route는 그룹 선택 후 적용)
    if len(ref_items) > MAX_ANSWER_REFS:
        ref_items = ref_items[:MAX_ANSWER_REFS]

    # --- Debug events for workflow trace ---
    _answer_events: List[str] = []
    if route == "setup" and doc_groups:
        _group_info = [(k[:60], len(refs)) for k, refs in doc_groups[:5]]
        _answer_events.append(f"[answer] groups({len(doc_groups)}): {_group_info}")
        _sel_doc = ref_items[0].get("doc_id", "?")[:50] if ref_items else "none"
        _sel_pages = [str(r.get("page", "?")) for r in ref_items[:5]]
        _sel_mode = "direct" if not is_fallback_selection else "fallback"
        _answer_events.append(
            f"[answer] selected: {_sel_doc} pages=[{','.join(_sel_pages)}] refs={len(ref_items)} ({_sel_mode})"
        )
        if route == "setup" and not is_fallback_selection:
            _answer_events.append(f"[answer] answer_ref_json: set (refs={len(ref_items)})")
        elif route == "setup" and is_fallback_selection:
            _answer_events.append("[answer] answer_ref_json: unset (fallback)")
    _answer_events.append(
        f"[answer] route={route} lang={answer_language} refs={len(ref_items)}"
    )

    ref_text = ref_json_to_text(ref_items)
    logger.info(
        "answer_node: route=%s, answer_language=%s, refs_chars=%d, docs=%d",
        route,
        answer_language,
        len(ref_text),
        len(ref_items),
    )

    mapping = {"sys.query": query_for_prompt, "ref_text": ref_text}

    # Select language-specific template
    # Korean (ko): use default template
    # English (en): use *_en template if available
    # Japanese (ja): use *_ja template if available
    # Chinese (zh): use *_zh template if available
    if answer_language == "en":
        templates = {
            "setup": spec.setup_ans_en or spec.setup_ans,
            "ts": spec.ts_ans_en or spec.ts_ans,
            "general": spec.general_ans_en or spec.general_ans,
        }
    elif answer_language == "zh":
        templates = {
            "setup": spec.setup_ans_zh or spec.setup_ans,
            "ts": spec.ts_ans_zh or spec.ts_ans,
            "general": spec.general_ans_zh or spec.general_ans,
        }
    elif answer_language == "ja":
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
    logger.info("answer_node: using %s template for route=%s", answer_language, route)

    user = _format_prompt(tmpl.user, mapping)

    # Prepend previous conversation context for follow-up queries
    if state.get("needs_history") and state.get("chat_history"):
        last = state["chat_history"][-1]
        prev_context = (
            f"[Previous Q&A]\n"
            f"Q: {last.get('user_text', '')}\n"
            f"A: {last.get('assistant_text') or ''}\n\n"
        )
        user = prev_context + user

    logger.info(
        "answer_node: user_prompt_chars=%d, system_prompt_chars=%d", len(user), len(tmpl.system)
    )
    answer_temperature = resolve_answer_temperature(route)
    answer, reasoning = _invoke_llm_with_reasoning(
        llm,
        tmpl.system,
        user,
        max_tokens=MAX_TOKENS_ANSWER,
        temperature=answer_temperature,
    )
    # 반복 블록 감지 및 절삭 (안전망)
    original_len = len(answer)
    answer = _truncate_repetition(answer)
    if len(answer) < original_len:
        logger.warning(
            "answer_node: [REPETITION_FALLBACK] 반복 절삭 발생! %d → %d chars (%.0f%% 제거). "
            "repeat_penalty/repeat_last_n 설정 점검 필요.",
            original_len,
            len(answer),
            (1 - len(answer) / original_len) * 100,
        )

    if route == "setup":
        post_processed = _postprocess_setup_answer_text(answer)
        if post_processed != answer:
            logger.info(
                "answer_node: setup post-process applied (%d -> %d chars)",
                len(answer),
                len(post_processed),
            )
        answer = post_processed
    logger.info(
        "answer_node: answer_chars=%d, reasoning_chars=%d, answer_preview=%s",
        len(answer),
        len(reasoning) if reasoning else 0,
        answer[:500] if answer else "(empty)",
    )

    _answer_events.append(f"[answer] ref_chars={len(ref_text)} answer_chars={len(answer)}")

    enforce_format = str(answer_language).strip().lower() == "ko"
    if not enforce_format:
        _result: Dict[str, Any] = {
            "answer": answer,
            "reasoning": reasoning,
            "answer_format": {"ok": True, "skipped": True, "target_language": answer_language},
            "answer_format_retries": 0,
            "_events": _answer_events,
        }
        if route == "setup" and not is_fallback_selection:
            _result["answer_ref_json"] = ref_items
        return _result

    has_refs = bool(ref_items)
    format_result = _validate_answer_format(answer, target_language="ko", has_refs=has_refs)
    retries = 0
    while not bool(format_result.get("ok")) and retries < MAX_ANSWER_FORMAT_RETRIES:
        retries += 1
        violations: List[str] = []
        if not bool(format_result.get("title_ok")):
            violations.append("- 첫 줄에 '# {제목}' 타이틀이 필요합니다.")
        missing = format_result.get("missing_sections")
        if isinstance(missing, list) and missing:
            violations.append("- 누락 섹션: " + ", ".join(str(x) for x in missing))
        if not bool(format_result.get("numbering_ok")):
            violations.append("- '## 작업 절차' 아래에 '1.' 번호 목록이 필요합니다.")
        if bool(format_result.get("has_emoji_numbering")):
            violations.append("- 이모지 번호(1️⃣ 등)는 금지입니다.")
        if bool(format_result.get("has_markdown_table")):
            violations.append("- 마크다운 테이블은 금지입니다.")
        if has_refs and not bool(format_result.get("citations_ok")):
            violations.append("- REFS가 있으면 본문에 [1] 같은 인용이 최소 1개 필요합니다.")
        if has_refs and not bool(format_result.get("references_ok")):
            violations.append("- REFS가 있으면 '## 참고문헌' 섹션이 필요합니다.")
        if not bool(format_result.get("language_ok")):
            violations.append("- target_language=ko에 맞는 언어(한국어)로 작성해야 합니다.")

        fix_system = (
            "\n\n[FORMAT FIX]\n"
            "아래 템플릿을 정확히 준수하여 답변 전체를 다시 작성하세요. "
            "금지 사항(이모지 번호, 마크다운 테이블, 혼합 언어 제목)을 절대 사용하지 마세요.\n"
            "위반 사항:\n" + ("\n".join(violations) if violations else "- 템플릿 미준수") + "\n"
        )

        answer_retry, reasoning_retry = _invoke_llm_with_reasoning(
            llm,
            tmpl.system + fix_system,
            user,
            max_tokens=MAX_TOKENS_ANSWER,
            temperature=answer_temperature,
        )
        answer_retry = _truncate_repetition(answer_retry)
        if route == "setup":
            answer_retry = _postprocess_setup_answer_text(answer_retry)
        answer = answer_retry
        reasoning = reasoning_retry
        format_result = _validate_answer_format(answer, target_language="ko", has_refs=has_refs)
        logger.warning(
            "answer_node: format retry %d/%d ok=%s",
            retries,
            MAX_ANSWER_FORMAT_RETRIES,
            bool(format_result.get("ok")),
        )

    _result_ko: Dict[str, Any] = {
        "answer": answer,
        "reasoning": reasoning,
        "answer_format": format_result,
        "answer_format_retries": retries,
        "_events": _answer_events,
    }
    if route == "setup" and not is_fallback_selection:
        _result_ko["answer_ref_json"] = ref_items
    return _result_ko


def issue_confirm_node(
    state: AgentState,
) -> Command[Literal["issue_case_selection", "done"]]:
    stage = str(state.get("issue_stage") or "post_summary")
    if stage not in {"post_summary", "post_detail"}:
        stage = "post_summary"
    nonce = _stable_issue_nonce(state, kind="issue_confirm", extra=stage)
    prompt = (
        "상세히 보고싶은 문서가 있습니까?"
        if stage == "post_summary"
        else "다른 이슈 사례도 상세히 보시겠습니까?"
    )
    payload = {
        "type": "issue_confirm",
        "nonce": nonce,
        "stage": stage,
        "question": state.get("query", ""),
        "instruction": "summary confirm" if stage == "post_summary" else "other confirm",
        "prompt": prompt,
    }
    decision = interrupt(payload)
    if isinstance(decision, dict):
        decision_nonce = str(decision.get("nonce") or "").strip()
        if decision_nonce != nonce:
            return Command(goto="done", update={"issue_confirm_nonce": nonce})
        if bool(decision.get("confirm", False)):
            return Command(goto="issue_case_selection", update={"issue_confirm_nonce": nonce})
    return Command(goto="done", update={"issue_confirm_nonce": nonce})


def issue_case_selection_node(
    state: AgentState,
) -> Command[Literal["issue_detail_answer", "done"]]:
    cases = state.get("issue_top10_cases") or []
    if not cases:
        return Command(goto="done")

    case_ids = ",".join(str(case.get("doc_id") or "") for case in cases)
    nonce = _stable_issue_nonce(state, kind="issue_case_selection", extra=case_ids)
    payload = {
        "type": "issue_case_selection",
        "nonce": nonce,
        "question": state.get("query", ""),
        "instruction": "case pick",
        "cases": cases,
    }
    decision = interrupt(payload)
    selected_doc_id = ""
    if isinstance(decision, dict):
        decision_nonce = str(decision.get("nonce") or "").strip()
        if decision_nonce != nonce:
            return Command(goto="done", update={"issue_case_selection_nonce": nonce})
        selected_doc_id = str(decision.get("selected_doc_id") or "").strip()
    if not selected_doc_id:
        return Command(goto="done", update={"issue_case_selection_nonce": nonce})

    return Command(
        goto="issue_detail_answer",
        update={
            "issue_selected_doc_id": selected_doc_id,
            "issue_case_selection_nonce": nonce,
        },
    )


def issue_detail_answer_node(
    state: AgentState,
    *,
    llm: BaseLLM,
    spec: PromptSpec,
) -> Dict[str, Any]:
    answer_language = state.get("target_language") or state.get("detected_language") or "ko"
    selected_doc_id = str(state.get("issue_selected_doc_id") or "").strip()
    all_refs = state.get("answer_ref_json") or state.get("ref_json") or []
    selected_refs: List[Dict[str, Any]] = []
    detail_ref_source = "fallback"

    issue_case_ref_map = state.get("issue_case_ref_map")
    if isinstance(issue_case_ref_map, dict) and selected_doc_id:
        mapped_refs = issue_case_ref_map.get(selected_doc_id)
        if isinstance(mapped_refs, list):
            selected_refs = [ref for ref in mapped_refs if isinstance(ref, dict)]
            if selected_refs:
                detail_ref_source = "case_ref_map"

    if not selected_refs:
        selected_refs = [
            ref for ref in all_refs if str(ref.get("doc_id") or "").strip() == selected_doc_id
        ]
        if selected_refs:
            detail_ref_source = "answer_ref_json"

    if not selected_refs:
        issue_case_refs = state.get("issue_case_refs") or []
        selected_refs = [
            ref
            for ref in issue_case_refs
            if isinstance(ref, dict) and str(ref.get("doc_id") or "").strip() == selected_doc_id
        ]
        if selected_refs:
            detail_ref_source = "issue_case_refs"

    if not selected_refs:
        logger.warning(
            "issue_detail: no refs found for selected_doc_id=%s, returning empty",
            selected_doc_id,
        )
        return {
            "answer": ISSUE_CASE_EMPTY_MESSAGE,
            "reasoning": None,
            "answer_ref_json": [],
            "issue_detail_ref_source": "not_found",
            "issue_stage": "post_detail",
        }

    query_for_prompt = state.get("query_en") if answer_language == "en" else state.get("query")
    if not query_for_prompt:
        query_for_prompt = state.get("query", "")

    ref_text = ref_json_to_text(selected_refs)
    mapping = {"sys.query": query_for_prompt, "ref_text": ref_text}
    tmpl = _get_issue_answer_template(spec, answer_language, detail=True)
    user = _format_prompt(tmpl.user, mapping)
    answer_temperature = resolve_answer_temperature(state.get("route", "general"))
    answer, reasoning = _invoke_llm_with_reasoning(
        llm,
        tmpl.system,
        user,
        max_tokens=MAX_TOKENS_ANSWER,
        temperature=answer_temperature,
    )
    answer = _truncate_repetition(answer)
    answer = _ensure_issue_detail_sections(answer, answer_language)
    return {
        "answer": answer,
        "reasoning": reasoning,
        "answer_ref_json": selected_refs,
        "issue_detail_ref_source": detail_ref_source,
        "issue_stage": "post_detail",
    }


def issue_sop_confirm_node(
    state: AgentState,
) -> Command[Literal["issue_confirm", "done", "mq"]]:
    nonce = _stable_issue_nonce(state, kind="issue_sop_confirm")
    payload = {
        "type": "issue_sop_confirm",
        "nonce": nonce,
        "question": state.get("query", ""),
        "instruction": "sop confirm",
        "prompt": "관련 SOP 확인으로 이어가시겠습니까?",
        "has_sop_ref": False,
        "sop_hint": None,
    }
    decision = interrupt(payload)
    if isinstance(decision, dict):
        decision_nonce = str(decision.get("nonce") or "").strip()
        if decision_nonce != nonce:
            return Command(
                goto="issue_confirm",
                update={"issue_stage": "post_detail", "issue_sop_confirm_nonce": nonce},
            )
        if bool(decision.get("confirm", False)):
            # 관련 SOP 조회: route를 setup으로 전환하여 SOP 검색 파이프라인 실행
            # parsed_query의 doc_types도 갱신해야 retrieve_node가 올바른 필터 사용
            pq_raw = state.get("parsed_query")
            updated_pq = dict(pq_raw) if isinstance(pq_raw, dict) else {}
            updated_pq["selected_doc_types"] = ["sop", "setup"]
            updated_pq["doc_types_strict"] = True
            updated_pq["route"] = "setup"
            return Command(
                goto="mq",
                update={
                    "route": "setup",
                    "task_mode": "sop",
                    "selected_doc_types": ["sop", "setup"],
                    "selected_doc_types_strict": True,
                    "parsed_query": updated_pq,
                    "issue_sop_confirm_nonce": nonce,
                },
            )
    return Command(
        goto="issue_confirm",
        update={"issue_stage": "post_detail", "issue_sop_confirm_nonce": nonce},
    )


_SUPPLEMENT_SETUP_SYSTEM = """
당신은 설치/셋업 SOP 답변 보완 전문가입니다.

## 역할
기존 답변과 REFS 근거를 비교하여, 누락된 내용을 통합한 **하나의 완성된 답변**을 다시 작성합니다.

## 규칙
- 기존 답변의 구조와 형식을 유지하세요.
- REFS에 있지만 답변에서 누락된 단계, 수치, 조건, 주의사항을 해당 위치에 자연스럽게 삽입하세요.
- 내용을 중복하지 마세요. 이미 있는 내용은 그대로 두고 누락된 부분만 추가하세요.
- REFS에 없는 내용을 추가하지 마세요.
- 인용 번호 [N]을 유지하세요.
- 보완할 내용이 없으면 정확히 "NONE"만 출력하세요.
- 보완할 내용이 있으면 통합된 **전체 답변**을 출력하세요 (보완 부분만이 아님).
- 한국어로 작성하세요.
""".strip()


def _supplement_setup_answer(
    answer: str,
    query: str,
    ref_text: str,
    *,
    llm: "BaseLLM",
) -> tuple[str, dict]:
    """Setup 답변을 REFS와 비교하여 누락된 내용을 통합한 완성된 답변을 생성한다.

    Returns:
        (통합된 답변, judge dict)
    """
    user = (
        f"질문: {query}\n\n"
        f"기존 답변:\n{answer}\n\n"
        f"REFS:\n{ref_text}\n\n"
        "위 기존 답변에서 REFS 대비 누락된 단계, 수치, 조건, 주의사항을 찾아 "
        "해당 위치에 통합한 완성된 답변을 작성하세요. "
        "보완할 내용이 없으면 NONE만 출력하세요."
    )
    raw = _invoke_llm(
        llm,
        _SUPPLEMENT_SETUP_SYSTEM,
        user,
        max_tokens=MAX_TOKENS_ANSWER,
        temperature=0.2,
    )
    raw_stripped = (raw or "").strip()

    if not raw_stripped or raw_stripped.upper() == "NONE":
        logger.info("judge_node(supplement): no supplement needed")
        return answer, {
            "faithful": True,
            "issues": [],
            "hint": "supplement: no additions needed",
            "supplement_applied": False,
        }

    # LLM이 통합된 전체 답변을 반환
    merged_answer = raw_stripped
    logger.info(
        "judge_node(supplement): merged answer (%d -> %d chars)",
        len(answer),
        len(merged_answer),
    )
    return merged_answer, {
        "faithful": True,
        "issues": [],
        "hint": "supplement: merged into answer",
        "supplement_applied": True,
        "original_length": len(answer),
        "merged_length": len(merged_answer),
    }


def judge_node(state: AgentState, *, llm: BaseLLM, spec: PromptSpec) -> Dict[str, Any]:
    route = state["route"]
    ref_items = state.get("answer_ref_json") or state.get("ref_json", [])
    ref_text = ref_json_to_text(ref_items)
    query_for_judge = state.get("query_en") or state["query"]
    answer = state.get("answer", "")

    # Setup route: 보완형 — 답변의 누락 내용을 보완하고 재시도 없이 종료
    if route == "setup" and answer and "찾지 못했습니다" not in answer:
        supplemented_answer, judge = _supplement_setup_answer(
            answer,
            query_for_judge,
            ref_text,
            llm=llm,
        )
        result: Dict[str, Any] = {"judge": judge}
        if supplemented_answer != answer:
            result["answer"] = supplemented_answer
        return result

    # Other routes (ts, general) + setup fallback: 기존 faithful 판정 유지
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
        "\n위 답안의 내용이 REFS 근거에 충실한지 판정하세요. "
        "답안의 형식은 평가하지 마세요. 내용만 평가하세요.\n"
        '출력: {"faithful": bool, "issues": [...], "hint": "..."}'
    )

    raw = _invoke_llm(llm, sys, user, max_tokens=MAX_TOKENS_JUDGE, temperature=TEMP_CLASSIFICATION)
    judge: Dict[str, Any] | None = None
    # 1차: non-greedy nested JSON 파싱
    try:
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict) and "faithful" in parsed:
                judge = parsed
    except Exception:
        pass
    # 2차: greedy fallback (단일 JSON 객체)
    if judge is None:
        try:
            json_match = re.search(r"\{.*\}", raw, flags=re.S)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    judge = parsed
        except Exception:
            pass
    # 3차: raw text에서 faithful 키워드 추출 (parse 완전 실패 시)
    if judge is None:
        raw_lower = (raw or "").lower()
        if '"faithful": true' in raw_lower or '"faithful":true' in raw_lower:
            judge = {"faithful": True, "issues": [], "hint": "parsed from raw text (faithful=true)"}
        elif '"faithful": false' in raw_lower or '"faithful":false' in raw_lower:
            # hint/issues 추출 시도
            hint_match = re.search(r'"hint"\s*:\s*"([^"]*)"', raw or "")
            hint = hint_match.group(1) if hint_match else ""
            judge = {"faithful": False, "issues": [], "hint": hint or "parsed from raw text"}
        else:
            logger.warning(
                "judge_node: failed to parse LLM output: %s", raw[:200] if raw else "(empty)"
            )
            judge = {"faithful": False, "issues": ["parse_error"], "hint": "judge JSON parse failed"}
    attempts = int(state.get("attempts", 0) or 0)
    max_attempts = int(state.get("max_attempts", 0) or 0)
    faithful = bool(judge.get("faithful", False))
    if not faithful and attempts >= max_attempts:
        issues = judge.get("issues")
        if not isinstance(issues, list):
            issues = []
        if "max_attempts_reached" not in issues:
            issues.append("max_attempts_reached")
        judge = {
            "faithful": False,
            "issues": issues,
            "hint": "Reached max_attempts without a faithful answer; stopping retries.",
        }
    return {"judge": judge}


def should_retry(
    state: AgentState,
) -> Literal["done", "retry", "retry_expand", "retry_mq", "human"]:
    """Determine retry strategy based on attempt count.

    Retry strategies:
    - 1st unfaithful (attempt 0→1): retry_expand - use more docs (8→20)
    - 2nd unfaithful (attempt 1→2): retry - refine queries
    - 3rd unfaithful (attempt 2→3): retry_mq - regenerate multi-query from scratch
    """
    if state.get("retrieval_confirmed"):
        return "done"
    judge = state.get("judge", {})
    faithful = bool(judge.get("faithful", False))
    if faithful:
        return "done"

    attempts = int(state.get("attempts", 0) or 0)
    max_attempts = int(state.get("max_attempts", 0) or 0)
    mq_mode = str(state.get("mq_mode") or "fallback")
    mq_reason = state.get("mq_reason")

    # Hard ceiling: always terminate gracefully once max attempts is reached.
    if attempts >= max_attempts:
        return "done"

    if mq_mode == "off":
        if attempts == 0:
            return "retry_expand"
        return "retry"

    if mq_mode == "fallback":
        if mq_reason == "empty_retrieval" or attempts >= 2:
            return "retry_mq"
        if attempts == 0:
            return "retry_expand"
        return "retry"

    # mq_mode == "on"
    if attempts == 0:
        return "retry_expand"
    if attempts == 1:
        return "retry"
    return "retry_mq"


def retry_bump_node(state: AgentState) -> Dict[str, Any]:
    """Increment attempt counter."""
    return {
        "attempts": int(state.get("attempts", 0)) + 1,
        "retry_strategy": "refine_queries",
    }


def retry_expand_node(state: AgentState) -> Dict[str, Any]:
    """1st retry strategy: increase expand_top_k from 8 to 20.

    This doesn't re-retrieve docs, just uses more of the already retrieved docs
    for answer generation.
    """
    attempts = int(state.get("attempts", 0)) + 1
    logger.info("retry_expand_node: increasing expand_top_k to 20 (attempt %d)", attempts)
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

    # Clear previous MQ lists to force regeneration
    return {
        "attempts": attempts,
        "mq_used": True,
        "mq_reason": (
            "empty_retrieval"
            if state.get("mq_reason") == "empty_retrieval"
            else "unfaithful_after_deterministic_retries"
        ),
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
        'Output: {"queries":[...]} in one line only.'
    )
    user_en = (
        f"Original question: {query_en}\n"
        f"Previous queries: {json.dumps(prev, ensure_ascii=False)}\n"
        f"Judge hint: {hint}\n"
    )
    raw_en = _invoke_llm(
        llm,
        sys_en,
        user_en,
        temperature=resolve_querygen_temperature(state, mq_invoked=False),
    )
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
            user = _format_prompt(
                spec.translate.user,
                {
                    "query": text,
                    "target_language": "Korean",
                },
            )
            result = _invoke_llm(llm, spec.translate.system, user, temperature=TEMP_TRANSLATION)
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
    guardrail_result = validate_search_queries(state, merged)
    final_queries = guardrail_result["search_queries"]
    logger.info(
        "refine_queries_node: bilingual queries EN=%d, KO=%d",
        len(english_queries),
        len(korean_queries),
    )
    return {
        "search_queries": final_queries,
        "guardrail_dropped_numeric": int(guardrail_result["guardrail_dropped_numeric"]),
        "guardrail_dropped_anchor": int(guardrail_result["guardrail_dropped_anchor"]),
        "guardrail_final_count": int(guardrail_result["guardrail_final_count"]),
    }


def _parse_auto_parse_result(text: str) -> Dict[str, Any]:
    """Parse LLM output for auto-parsed devices, doc_types, equip_ids, and language."""
    result: Dict[str, Any] = {"devices": [], "doc_types": [], "equip_ids": [], "language": None}
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
                equip_ids = obj.get("equip_ids")
                language = obj.get("language")

                # Backward compat: allow single fields
                if devices is None and "device" in obj:
                    devices = [obj.get("device")]
                if doc_types is None and "doc_type" in obj:
                    doc_types = [obj.get("doc_type")]
                if equip_ids is None and "equip_id" in obj:
                    equip_ids = [obj.get("equip_id")]

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

                parsed_equip_ids: List[str] = []
                if isinstance(equip_ids, list):
                    for eid in equip_ids:
                        normalized = _normalize_equip_id(eid)
                        if _is_valid_equip_id(normalized):
                            parsed_equip_ids.append(normalized)
                elif equip_ids:
                    normalized = _normalize_equip_id(equip_ids)
                    if _is_valid_equip_id(normalized):
                        parsed_equip_ids.append(normalized)

                # Parse language
                parsed_language: Optional[str] = None
                if language:
                    lang_str = str(language).strip().lower()
                    if lang_str in ("ko", "en", "ja", "zh"):
                        parsed_language = lang_str

                result["devices"] = _dedupe_queries(parsed_devices)[:2]
                result["doc_types"] = _dedupe_queries(parsed_doc_types)[:2]
                result["equip_ids"] = _dedupe_queries(parsed_equip_ids)[:2]
                result["language"] = parsed_language
    except Exception:
        pass

    return result


GARBAGE_EQUIP_IDS = {"", "-", ".", "/", "1", "NA", "N/A"}
_EQUIP_ID_EXPLICIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)\b(?:equip(?:ment)?[\s_-]*id|eq[\s_-]*id|장비[\s_-]*(?:id|번호)|설비[\s_-]*(?:id|번호))\s*(?:is|=|:)?\s*([A-Z0-9][A-Z0-9_-]{2,})"
    ),
]


def _normalize_equip_id(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    normalized = normalized.strip("\"'`[](){}<>")
    normalized = normalized.rstrip(".,;:!?")
    normalized = normalized.strip()
    return normalized.upper()


def _is_valid_equip_id(value: str) -> bool:
    if not value:
        return False
    normalized = value.strip().upper()
    if not normalized or normalized in GARBAGE_EQUIP_IDS:
        return False
    if len(normalized) < 4 or len(normalized) > 24:
        return False
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]*", normalized):
        return False
    if not re.search(r"\d", normalized):
        return False
    # Exclude common short error code patterns (e.g., E001, E-1234)
    if re.fullmatch(r"E-?\d{2,5}", normalized):
        return False
    return True


def _extract_equip_ids_by_lookup(query: str, known_equip_ids: Set[str]) -> List[str]:
    """Extract equip_ids from query using lookup against known equip_id set.

    Tokenizes the query and checks each token (and adjacent token combinations)
    against the known equip_id set for exact matching.
    """
    if not query or not known_equip_ids:
        return []

    # Tokenize: split on whitespace, then strip common punctuation
    raw_tokens = re.split(r"[\s,;:!?()]+", query.strip())
    tokens = [t.strip("\"'`[](){}<>.,;:!?").upper() for t in raw_tokens if t.strip()]

    matched: List[str] = []

    # 1) Single-token lookup
    for token in tokens:
        if token in known_equip_ids:
            matched.append(token)

    # 2) Adjacent token combination (e.g., "DES 02" → "DES02", "5 EBP 0701" → "5EBP0701")
    if len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            combined = tokens[i] + tokens[i + 1]
            if combined in known_equip_ids and combined not in matched:
                matched.append(combined)
        # 3-token combination for cases like "5 EBP 0701"
        for i in range(len(tokens) - 2):
            combined = tokens[i] + tokens[i + 1] + tokens[i + 2]
            if combined in known_equip_ids and combined not in matched:
                matched.append(combined)

    return _dedupe_queries(matched)[:2]


def _extract_equip_ids_from_query(
    query: str,
    known_equip_ids: Set[str] | None = None,
) -> List[str]:
    if not query:
        return []

    # Lookup-based extraction when known equip_ids are available
    if known_equip_ids:
        lookup_results = _extract_equip_ids_by_lookup(query, known_equip_ids)
        if lookup_results:
            return lookup_results

    # Regex fallback: explicit cue patterns (equip_id, 장비번호, ...)
    extracted: List[str] = []

    for pattern in _EQUIP_ID_EXPLICIT_PATTERNS:
        for match in pattern.findall(query):
            normalized = _normalize_equip_id(match)
            if _is_valid_equip_id(normalized):
                extracted.append(normalized)

    if extracted:
        return _dedupe_queries(extracted)[:2]

    # Fallback: standalone token query like "EPAG50" or "equip id EPAG50"
    compact_query = query.strip()
    standalone_match = re.match(
        r"(?i)^\s*(?:equip(?:ment)?[\s_-]*id|eq[\s_-]*id|장비[\s_-]*(?:id|번호)|설비[\s_-]*(?:id|번호))?\s*([A-Z0-9][A-Z0-9_-]{2,})\s*$",
        compact_query,
    )
    if standalone_match:
        normalized = _normalize_equip_id(standalone_match.group(1))
        if _is_valid_equip_id(normalized):
            return [normalized]

    # Conservative fallback: if query has exactly one strong candidate token.
    token_candidates = [
        _normalize_equip_id(token)
        for token in re.findall(r"\b[A-Z0-9][A-Z0-9_-]{2,}\b", compact_query.upper())
    ]
    valid_candidates = [token for token in token_candidates if _is_valid_equip_id(token)]
    deduped = _dedupe_queries(valid_candidates)
    if len(deduped) == 1:
        return deduped[:1]

    return []


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
    device_map = {
        str(name).strip().lower(): str(name).strip() for name in device_names if str(name).strip()
    }
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


def _is_valid_device_candidate(name: str) -> bool:
    cleaned = str(name).strip()
    if not cleaned:
        return False
    compact = _compact_text(cleaned)
    if compact in {"all", "etc"}:
        return False
    token = re.sub(r"[\s\-_./]+", "", cleaned)
    # Short alphabetic tokens (e.g., APC, ALL) are typically component/noise labels,
    # not equipment models. Ignore them for auto-parse to reduce false positives.
    if token.isalpha() and len(token) <= 4:
        return False
    return True


# 한국어 음차 → canonical 장비명 매핑 (exact match 전 정규화용)
_DEVICE_KO_ALIASES: Dict[str, str] = {
    "수프라": "SUPRA",
    "제니바": "GENEVA",
    "프레시아": "PRECIA",
    "인테저": "INTEGER",
    "옴니스": "OMNIS",
    "테라": "TERA",
    "지비스": "ZIVIS",
    "티그마": "TIGMA",
    "제디우스": "ZEDIUS",
}


def _extract_devices_from_query(device_names: List[str], query: str) -> List[str]:
    if not device_names or not query:
        return []

    query_compact = _compact_text(query)

    # Phase 0: 한국어 음차를 영문으로 치환하여 query_compact 보강
    query_for_match = query_compact
    for ko, en in _DEVICE_KO_ALIASES.items():
        if ko in query:
            query_for_match = query_for_match.replace(_compact_text(ko), _compact_text(en))

    # Phase 1: exact substring match (기존 로직)
    # 단어 경계 인식을 위해 원본 query 토큰도 함께 사용
    query_tokens_compact = [_compact_text(t) for t in re.split(r"[\s가-힣,;:()\"'?!。，；：（）의]+", query) if t.strip()]
    # 연속 토큰 조합도 생성 (multi-word device: "supra vplus" → "supravplus")
    query_token_combos: set[str] = set(query_tokens_compact)
    for i in range(len(query_tokens_compact) - 1):
        query_token_combos.add(query_tokens_compact[i] + query_tokens_compact[i + 1])
    if len(query_tokens_compact) >= 3:
        for i in range(len(query_tokens_compact) - 2):
            query_token_combos.add(query_tokens_compact[i] + query_tokens_compact[i + 1] + query_tokens_compact[i + 2])

    token_matches: List[str] = []
    substr_matches: List[str] = []
    for name in device_names:
        cleaned = str(name).strip()
        if not cleaned or not _is_valid_device_candidate(cleaned):
            continue
        name_compact = _compact_text(cleaned)
        if not name_compact:
            continue
        # 토큰 조합 기반 매칭 (word boundary 보존, 우선)
        if name_compact in query_token_combos:
            token_matches.append(cleaned)
        # fallback: 전체 compact query에서 substring 매칭
        elif name_compact in query_for_match:
            substr_matches.append(cleaned)
    # 토큰 조합 매칭 우선, 없으면 substring fallback
    matches = token_matches if token_matches else substr_matches
    if matches:
        # 긴 이름 우선 (SUPRA Vplus > SUPRA V)
        matches.sort(key=lambda n: len(_compact_text(n)), reverse=True)
        best = matches[0]
        best_compact = _compact_text(best)
        # prefix 모호성 체크: 매칭된 이름이 더 긴 device name의 prefix인 경우
        # Phase 2 fuzzy로 더 specific한 매칭 시도 (오타 대응: "vvplus" → "vplus")
        has_longer_candidate = any(
            _compact_text(n).startswith(best_compact)
            and len(_compact_text(n)) > len(best_compact)
            and _is_valid_device_candidate(n)
            for n in device_names
        )
        if not has_longer_candidate:
            return _dedupe_queries(matches)[:1]
        # prefix 모호성이 있으면 Phase 2로 fall through
        logger.debug(
            "device Phase 1 match '%s' is prefix of longer candidate, trying fuzzy",
            best,
        )

    # Phase 2: token-level fuzzy (rapidfuzz)
    # exact match 실패 또는 prefix 모호성 시 실행. 오타 내성 향상.
    try:
        from rapidfuzz import fuzz, process

        candidates: Dict[str, str] = {}  # compact → original
        for name in device_names:
            cleaned = str(name).strip()
            if not cleaned or not _is_valid_device_candidate(cleaned):
                continue
            candidates[_compact_text(cleaned)] = cleaned

        if not candidates:
            return []

        # 토큰 추출: 한글/공백/구두점으로 분리하여 영문+숫자 토큰만 추출
        raw_tokens = re.split(r"[\s가-힣,;:()\"'?!。，；：（）]+", query)
        alpha_tokens = [
            _compact_text(t) for t in raw_tokens
            if len(t) > 3 and not t.isdigit()
        ]
        # 연속 토큰 2-3개 결합 (multi-word 장비명 대응: "supra vplus", "integer plus")
        combined_tokens: List[str] = list(alpha_tokens)
        for i in range(len(raw_tokens) - 1):
            pair = _compact_text(raw_tokens[i] + raw_tokens[i + 1])
            if len(pair) > 4:
                combined_tokens.append(pair)
        if len(raw_tokens) >= 3:
            for i in range(len(raw_tokens) - 2):
                triple = _compact_text(raw_tokens[i] + raw_tokens[i + 1] + raw_tokens[i + 2])
                if len(triple) > 4:
                    combined_tokens.append(triple)

        # 각 토큰을 장비명 후보와 WRatio로 매칭 (partial_ratio보다 오타에 강건)
        best_match: tuple | None = None  # (score, compact, original)
        candidate_keys = list(candidates.keys())
        for token in combined_tokens:
            if not token or len(token) < 4:
                continue
            result = process.extractOne(
                token,
                candidate_keys,
                scorer=fuzz.WRatio,
                score_cutoff=82,
            )
            if result:
                matched_compact, score, _idx = result
                if best_match is None:
                    best_match = (score, matched_compact, candidates[matched_compact])
                elif score > best_match[0] + 5:
                    # 확실히 높은 score → 교체
                    best_match = (score, matched_compact, candidates[matched_compact])
                elif score >= best_match[0] - 5 and len(matched_compact) > len(best_match[1]):
                    # score 비슷하지만 더 specific(긴 이름) → 우선
                    best_match = (score, matched_compact, candidates[matched_compact])

        # fallback: 전체 쿼리로도 시도 (토큰 분리가 실패한 경우 대비)
        if best_match is None:
            result = process.extractOne(
                query_for_match,
                candidate_keys,
                scorer=fuzz.WRatio,
                score_cutoff=82,
            )
            if result:
                matched_compact, score, _idx = result
                best_match = (score, matched_compact, candidates[matched_compact])

        if best_match:
            score, matched_compact, original = best_match
            # Phase 1 prefix 모호성으로 fall-through한 경우:
            # fuzzy 결과가 Phase 1보다 더 specific(긴 이름)이면 fuzzy 우선,
            # 아니면 Phase 1 결과 유지
            if matches:
                phase1_best = matches[0]
                phase1_len = len(_compact_text(phase1_best))
                fuzzy_len = len(matched_compact)
                if fuzzy_len >= phase1_len:
                    logger.info(
                        "device fuzzy preferred over Phase 1: '%s'→'%s' (score=%d, Phase1='%s')",
                        query[:40], original, score, phase1_best,
                    )
                    return [original]
                else:
                    logger.info(
                        "device Phase 1 kept over fuzzy: '%s' (Phase1='%s', fuzzy='%s' score=%d)",
                        query[:40], phase1_best, original, score,
                    )
                    return [phase1_best]
            logger.info(
                "device fuzzy match: query='%s' → '%s' (score=%d)",
                query[:40], original, score,
            )
            return [original]
    except Exception:
        logger.debug("device fuzzy match failed", exc_info=True)

    # Phase 2 실패 시 Phase 1 결과가 있으면 fallback
    if matches:
        logger.info("device fuzzy failed, falling back to Phase 1: '%s'", matches[0])
        return _dedupe_queries(matches)[:1]

    return []


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
    - Chinese (CJK ideographs without kana): zh
    - Otherwise: en

    Note: Domain terms (device names, technical terms) may be in English
    even in Korean/Japanese/Chinese sentences. This function prioritizes
    CJK scripts over English.

    Returns:
        "ko", "en", "ja", or "zh"
    """
    if not text:
        return "ko"  # Default to Korean

    # Count Korean characters (Hangul)
    korean_chars = len(re.findall(r"[가-힣]", text))

    # Count Japanese characters (Hiragana + Katakana)
    # Hiragana: \u3040-\u309f, Katakana: \u30a0-\u30ff
    japanese_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text))
    # Count CJK ideographs (Han characters used by Chinese/Japanese)
    cjk_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))

    # Prioritize Korean if any Korean characters exist
    if korean_chars > 0:
        return "ko"

    # Then Japanese if any Japanese characters exist
    if japanese_chars > 0:
        return "ja"

    # Chinese: CJK ideographs without Japanese kana are treated as Chinese.
    # This handles common zh queries like "设备校准步骤".
    if cjk_chars > 0:
        return "zh"

    # Default to English
    return "en"


# -----------------------------
# 7) Chat history / follow-up detection
# -----------------------------
# Rule-based patterns for detecting follow-up queries
_FOLLOWUP_PATTERNS_KO = re.compile(
    r"더\s*자세히|아까|방금|위에서|그\s*문서|해당|같은\s*장비|다시|구체적으로"
    r"|앞서|이전|그거|그것|위\s*내용|추가로|보충|마저",
    re.IGNORECASE,
)
_FOLLOWUP_PATTERNS_EN = re.compile(
    r"more\s+detail|previous|above|that\s+document|the\s+same|tell\s+me\s+more"
    r"|elaborate|earlier|you\s+(?:just\s+)?(?:said|mentioned)|could\s+you\s+explain",
    re.IGNORECASE,
)
_SHORT_PRONOUN_KO = re.compile(r"그[거것걸게]|뭐|왜|어떻게")
_SHORT_PRONOUN_EN = re.compile(r"\b(?:it|that|this|those|these|what|why|how)\b", re.IGNORECASE)

# Short imperative requests without a subject → likely follow-up
# e.g., "교체 절차를 알려줘", "자세히 설명해줘", "다른 방법 보여줘"
_IMPERATIVE_ENDINGS_KO = re.compile(
    r"(?:알려|설명해|보여|가르쳐|말해|안내해|정리해|요약해)\s*(?:줘|줘요|주세요|주십시오|봐|봐요|보세요)|"
    r"뭐야\??|뭔가요\??|인가요\??$",
)

# Minimum content-word length for overlap check
_CONTENT_WORD_RE = re.compile(r"[가-힣]{2,}|[a-zA-Z]{3,}")
# Korean stop words to exclude from overlap check
_STOP_WORDS_KO = {
    "하는",
    "이런",
    "저런",
    "어떤",
    "있는",
    "없는",
    "하고",
    "그리고",
    "또는",
    "대해",
    "대한",
    "위한",
    "통해",
    "에서",
    "으로",
    "부터",
    "까지",
    "에게",
}


def _has_topic_overlap(query: str, prev_user: str, prev_assistant: str) -> bool:
    """Check if current query shares topic words with the previous turn."""
    current_words = set(_CONTENT_WORD_RE.findall(query.lower())) - _STOP_WORDS_KO
    if not current_words:
        return False
    # Only check first 300 chars of assistant text to avoid noise
    prev_text = f"{prev_user} {prev_assistant[:300]}".lower()
    prev_words = set(_CONTENT_WORD_RE.findall(prev_text)) - _STOP_WORDS_KO
    if not prev_words:
        return False
    overlap = current_words & prev_words
    # If half or more of the current query's content words appear in previous turn
    return len(overlap) / len(current_words) >= 0.5


def _history_check_rule_based(state: AgentState) -> bool:
    """Fallback rule-based check for follow-up detection."""
    chat_history = state.get("chat_history") or []
    if not chat_history:
        return False

    query = (state.get("query") or "").strip()
    if not query:
        return False

    # Pattern matching for Korean / English follow-up cues
    if _FOLLOWUP_PATTERNS_KO.search(query):
        return True
    if _FOLLOWUP_PATTERNS_EN.search(query):
        return True

    # Short query with pronoun → likely follow-up
    if len(query) <= 15:
        if _SHORT_PRONOUN_KO.search(query) or _SHORT_PRONOUN_EN.search(query):
            return True

    last = chat_history[-1]
    prev_user = last.get("user_text", "")
    prev_assistant = last.get("assistant_text", "")

    # Short imperative request without explicit subject → likely follow-up
    # e.g., "교체 절차를 알려줘" (what to replace? → depends on previous context)
    if len(query) <= 30 and _IMPERATIVE_ENDINGS_KO.search(query):
        if _has_topic_overlap(query, prev_user, prev_assistant):
            return True

    # Vocabulary overlap: short query reusing previous turn's topic words
    if len(query) <= 25 and _has_topic_overlap(query, prev_user, prev_assistant):
        return True

    return False


def _parse_needs_history_from_text(text: str) -> bool | None:
    raw = (text or "").strip()
    if not raw:
        return None

    # Try JSON first
    try:
        obj_match = re.search(r"\{.*\}", raw, flags=re.S)
        if obj_match:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("needs_history"), bool):
                return bool(obj["needs_history"])
    except Exception:
        pass

    lowered = raw.lower()
    # Common deterministic forms
    if lowered in {"true", "yes", "need_history", "needs_history"}:
        return True
    if lowered in {"false", "no", "no_history", "independent"}:
        return False
    if re.search(r"\bneeds_history\s*[:=]\s*true\b", lowered):
        return True
    if re.search(r"\bneeds_history\s*[:=]\s*false\b", lowered):
        return False

    return None


def history_check_node(state: AgentState, *, llm: BaseLLM) -> Dict[str, Any]:
    """LLM-based check: does the current query need previous conversation context?

    Falls back to rule-based logic if parsing/model call fails.
    """
    chat_history = state.get("chat_history") or []
    if not chat_history:
        return {"needs_history": False}

    query = (state.get("query") or "").strip()
    if not query:
        return {"needs_history": False}

    last = chat_history[-1]
    prev_user = str(last.get("user_text", "") or "")
    prev_assistant = str(last.get("assistant_text", "") or "")

    system = (
        "You classify whether the current user query requires previous conversation context.\n"
        'Return ONE JSON line only: {"needs_history": true|false}.\n'
        "Use true when the query depends on prior Q/A (pronouns, omitted subject, "
        "references like 'that/above/previous', requests to elaborate/continue).\n"
        "Use false when the query is self-contained.\n"
        "If uncertain, return false."
    )
    user = (
        f"[Previous User Query]\n{prev_user}\n\n"
        f"[Previous Assistant Answer]\n{prev_assistant}\n\n"
        f"[Current Query]\n{query}\n\n"
        "Output JSON:"
    )

    try:
        raw = _invoke_llm(
            llm,
            system,
            user,
            max_tokens=MAX_TOKENS_CLASSIFICATION,
            temperature=TEMP_CLASSIFICATION,
        )
        parsed = _parse_needs_history_from_text(raw)
        if parsed is not None:
            logger.info(
                "history_check_node: llm decision needs_history=%s query=%s",
                parsed,
                query[:60],
            )
            return {"needs_history": parsed}
        logger.warning(
            "history_check_node: could not parse LLM output, fallback to rules: %s", raw[:120]
        )
    except Exception as exc:
        logger.warning("history_check_node: LLM check failed, fallback to rules: %s", exc)

    fallback = _history_check_rule_based(state)
    logger.info(
        "history_check_node: fallback decision needs_history=%s query=%s", fallback, query[:60]
    )
    return {"needs_history": fallback}


def query_rewrite_node(state: AgentState, *, llm: BaseLLM) -> Dict[str, Any]:
    """Rewrite a follow-up query into a self-contained question using the previous turn.

    Single LLM call with temperature 0.0 for deterministic output.
    """
    chat_history = state.get("chat_history") or []
    if not chat_history:
        return {}

    last = chat_history[-1]
    prev_user = last.get("user_text", "")
    prev_assistant = last.get("assistant_text") or ""
    prev_doc_ids = last.get("doc_ids") or []
    current_query = state.get("query", "")

    system = (
        "You are a query rewriter. Given the previous Q&A context and the current follow-up question, "
        "rewrite the current question into a single self-contained question that can be understood without "
        "any prior context. Keep the same language as the current question. "
        "Output ONLY the rewritten question, nothing else."
    )
    user = (
        f"[Previous Q&A]\n"
        f"Q: {prev_user}\n"
        f"A: {prev_assistant}\n\n"
        f"[Current follow-up question]\n"
        f"{current_query}\n\n"
        f"Rewritten self-contained question:"
    )

    rewritten = _invoke_llm(llm, system, user, temperature=TEMP_CLASSIFICATION)
    rewritten = rewritten.strip().strip('"').strip("'").strip()
    if not rewritten:
        rewritten = current_query

    logger.info("query_rewrite_node: '%s' → '%s'", current_query[:60], rewritten[:60])
    return {
        "query": rewritten,
        "prev_doc_ids": prev_doc_ids,
    }


def auto_parse_node(
    state: AgentState,
    *,
    llm: BaseLLM,
    spec: PromptSpec,
    device_names: List[str],
    doc_type_names: List[str],
    equip_id_set: Set[str] | None = None,
) -> Dict[str, Any]:
    """Auto-parse device, doc_type, and language from user query using LLM.

    Args:
        state: Agent state with query.
        llm: LLM instance for generation.
        spec: Prompt specification (must have auto_parse template).
        device_names: List of available device names.
        doc_type_names: List of available doc type names.
        equip_id_set: Optional set of known equip_ids for lookup-based extraction.

    Returns:
        State update with auto_parsed_device, auto_parsed_doc_type, detected_language, and auto_parse_message.
    """
    query = _normalize_query_text(state["query"]) or state["query"]

    # 규칙 기반만 사용 (LLM 호출 없음 - 속도 최적화)
    detected_language = _detect_language_rule_based(query)
    detected_devices = _extract_devices_from_query(device_names, query)
    detected_doc_types = _extract_doc_types_from_query(query)
    # equip_id 추출: lookup 기반만 활성화 (known set 대조, 안전)
    # regex fallback은 모델명(SR8241, 3000QC 등) 오인 문제로 비활성화 유지 (2026-03-12)
    if equip_id_set:
        detected_equip_ids = _extract_equip_ids_by_lookup(query, equip_id_set)
    else:
        detected_equip_ids = []

    chat_history = state.get("chat_history") or []
    needs_history = bool(state.get("needs_history"))
    if chat_history:
        needs_history = _history_check_rule_based(state)

    prev_devices = list(state.get("selected_devices") or [])
    prev_doc_types = list(state.get("selected_doc_types") or [])
    prev_equip_ids = list(state.get("selected_equip_ids") or [])

    devices = detected_devices if detected_devices else prev_devices
    parsed_query = state.get("parsed_query")
    selected_doc_types_strict = bool(state.get("selected_doc_types_strict"))
    if (not selected_doc_types_strict) and isinstance(parsed_query, dict):
        selected_doc_types_strict = bool(parsed_query.get("doc_types_strict"))

    if selected_doc_types_strict and prev_doc_types and not detected_doc_types:
        doc_types = prev_doc_types
    elif detected_doc_types:
        doc_types = detected_doc_types
    else:
        doc_types = prev_doc_types
    equip_ids = detected_equip_ids if detected_equip_ids else prev_equip_ids

    logger.info(
        "auto_parse_node: needs_history=%s, detected_devices=%s, detected_doc_types=%s, detected_equip_ids=%s, effective_devices=%s, effective_doc_types=%s, effective_equip_ids=%s, language=%s (rule-based only)",
        needs_history,
        detected_devices,
        detected_doc_types,
        detected_equip_ids,
        devices,
        doc_types,
        equip_ids,
        detected_language,
    )

    # Build display message based on detected language
    # Language display labels
    lang_labels = {"ko": "kor", "en": "eng", "ja": "jap", "zh": "zho"}
    lang_label = lang_labels.get(detected_language, "kor")

    message_parts: List[str] = []
    if detected_language == "en":
        if devices:
            message_parts.append(f"Device: {', '.join(devices)}")
        if doc_types:
            message_parts.append(f"Doc type: {', '.join(doc_types)}")
        if equip_ids:
            message_parts.append(f"Equip ID: {', '.join(equip_ids)}")
        message_parts.append(f"lang: {lang_label}")
        auto_parse_message = f"Parsed - {', '.join(message_parts)}"
    elif detected_language == "ja":
        if devices:
            message_parts.append(f"機器: {', '.join(devices)}")
        if doc_types:
            message_parts.append(f"文書: {', '.join(doc_types)}")
        if equip_ids:
            message_parts.append(f"装置ID: {', '.join(equip_ids)}")
        message_parts.append(f"lang: {lang_label}")
        auto_parse_message = f"パース結果 - {', '.join(message_parts)}"
    elif detected_language == "zh":
        if devices:
            message_parts.append(f"设备: {', '.join(devices)}")
        if doc_types:
            message_parts.append(f"文档: {', '.join(doc_types)}")
        if equip_ids:
            message_parts.append(f"设备ID: {', '.join(equip_ids)}")
        message_parts.append(f"lang: {lang_label}")
        auto_parse_message = f"解析结果 - {', '.join(message_parts)}"
    else:  # ko (default)
        if devices:
            message_parts.append(f"장비: {', '.join(devices)}")
        if doc_types:
            message_parts.append(f"문서: {', '.join(doc_types)}")
        if equip_ids:
            message_parts.append(f"장비ID: {', '.join(equip_ids)}")
        message_parts.append(f"lang: {lang_label}")
        auto_parse_message = f"파싱 결과 - {', '.join(message_parts)}"

    # Set selected_devices and selected_doc_types for downstream nodes
    # STRICT: Only one device allowed
    selected_devices = devices[:1]
    if selected_doc_types_strict and prev_doc_types and not detected_doc_types:
        selected_doc_types = prev_doc_types
    else:
        selected_doc_types = doc_types[:2]
    # equip_id auto-parse 비활성화 (모델명 오인 방지)
    # selected_equip_ids = equip_ids[:1]
    selected_equip_ids: list[str] = []

    # ParsedQuery 생성
    pq = ParsedQuery(
        device_names=devices,
        # equip_ids=equip_ids,  # 비활성화
        equip_ids=[],
        doc_types=doc_types,
        detected_language=detected_language,
        selected_devices=selected_devices,
        selected_equip_ids=selected_equip_ids,
        selected_doc_types=selected_doc_types,
        doc_types_strict=selected_doc_types_strict,
        message=auto_parse_message,
        device_selection_skipped=len(selected_devices) == 0,
        doc_type_selection_skipped=len(selected_doc_types) == 0,
    )

    # 언어는 항상 반환하고, 파싱 이벤트도 항상 발행한다.
    # (장비/문서 미감지 케이스에서도 상단 배너 표시를 위해 필요)
    return {
        "parsed_query": pq.model_dump(),
        # 하위 호환 필드 (기존 코드가 참조하므로 유지)
        "detected_language": detected_language,
        "auto_parsed_device": devices[0] if devices else None,
        "auto_parsed_doc_type": doc_types[0] if doc_types else None,
        "auto_parsed_devices": devices,
        "auto_parsed_doc_types": doc_types,
        "auto_parsed_equip_id": equip_ids[0] if equip_ids else None,
        "auto_parsed_equip_ids": equip_ids,
        "auto_parse_message": auto_parse_message,
        "selected_devices": selected_devices,
        "selected_doc_types": selected_doc_types,
        "selected_equip_ids": selected_equip_ids,
        "device_selection_skipped": len(selected_devices) == 0,
        "doc_type_selection_skipped": len(selected_doc_types) == 0,
        "_events": [
            {
                "type": "auto_parse",
                "device": devices[0] if devices else None,
                "doc_type": doc_types[0] if doc_types else None,
                "devices": devices,
                "doc_types": doc_types,
                "equip_id": equip_ids[0] if equip_ids else None,
                "equip_ids": equip_ids,
                "language": detected_language,
                "message": auto_parse_message,
            }
        ],
    }


def auto_parse_confirm_node(
    state: AgentState,
    *,
    device_fetcher: Any | None = None,
) -> Command[Literal["history_check"]]:
    guided_confirm = state.get("guided_confirm") is True
    already_confirmed = state.get("auto_parse_confirmed") is True
    abbreviation_resolved = state.get("abbreviation_resolved") is True
    if (not guided_confirm) or already_confirmed or abbreviation_resolved:
        return Command(goto="history_check", update={})

    query = str(state.get("query") or "")
    pq_dict = dict(state.get("parsed_query") or {})

    parsed_devices = _dedupe_queries(
        [
            str(v).strip()
            for v in (
                pq_dict.get("device_names")
                or state.get("auto_parsed_devices")
                or ([state.get("auto_parsed_device")] if state.get("auto_parsed_device") else [])
            )
            if str(v).strip()
        ]
    )
    detected_language_raw = str(
        pq_dict.get("detected_language") or state.get("detected_language") or "ko"
    ).lower()
    detected_language = (
        detected_language_raw if detected_language_raw in {"ko", "en", "zh", "ja"} else "ko"
    )

    device_options: List[Dict[str, Any]] = []
    seen_devices: Set[str] = set()

    for idx, device_name in enumerate(parsed_devices):
        if device_name in seen_devices:
            continue
        seen_devices.add(device_name)
        device_options.append(
            {
                "value": device_name,
                "label": device_name,
                "recommended": idx == 0,
            }
        )

    if device_fetcher is not None:
        try:
            fetched = device_fetcher()
            fetched_devices: List[Dict[str, Any]] = []
            if isinstance(fetched, dict):
                fetched_devices = fetched.get("devices", []) or []
            elif isinstance(fetched, list):
                fetched_devices = fetched

            added = 0
            for item in fetched_devices:
                if added >= 8:
                    break
                if not isinstance(item, dict):
                    continue
                device_name = str(item.get("name") or "").strip()
                if (not device_name) or (device_name in seen_devices):
                    continue
                seen_devices.add(device_name)
                option: Dict[str, Any] = {
                    "value": device_name,
                    "label": device_name,
                    "recommended": False,
                }
                doc_count = item.get("doc_count")
                if isinstance(doc_count, int):
                    option["doc_count"] = doc_count
                device_options.append(option)
                added += 1
        except Exception as exc:
            logger.warning("auto_parse_confirm_node: failed to fetch devices: %s", exc)

    device_options.append(
        {"value": "__skip__", "label": "건너뛰기", "recommended": len(parsed_devices) == 0}
    )

    task_options = [
        {"value": "sop", "label": "작업절차검색", "recommended": False},
        {"value": "issue", "label": "이슈조회", "recommended": True},
        {"value": "all", "label": "전체조회", "recommended": False},
    ]

    defaults = {
        "target_language": detected_language,
        "device": parsed_devices[0] if parsed_devices else None,
        "equip_id": None,
        "task_mode": "issue",
    }

    payload = {
        "type": "auto_parse_confirm",
        "question": query,
        "instruction": "",
        "steps": ["device", "task"],
        "options": {
            "device": device_options,
            "task": task_options,
        },
        "defaults": defaults,
    }

    decision = interrupt(payload)

    update: Dict[str, Any] = {"auto_parse_confirmed": True}
    merged_pq = dict(pq_dict)

    if isinstance(decision, dict) and decision.get("type") == "auto_parse_confirm":
        target_language = str(decision.get("target_language") or "").lower().strip()
        if target_language in {"ko", "en", "zh", "ja"}:
            update["target_language"] = target_language

        selected_device = decision.get("selected_device")
        selected_devices: List[str] = []
        if isinstance(selected_device, str):
            selected_device = selected_device.strip()
            if selected_device and selected_device != "__skip__":
                selected_devices = [selected_device]

        selected_equip_id = decision.get("selected_equip_id")
        selected_equip_ids: List[str] = []
        if isinstance(selected_equip_id, str):
            normalized_eid = _normalize_equip_id(selected_equip_id)
            if selected_equip_id.strip() in {"__skip__", "__manual__"}:
                normalized_eid = ""
            if _is_valid_equip_id(normalized_eid):
                selected_equip_ids = [normalized_eid]

        task_mode = str(decision.get("task_mode") or "").lower().strip()
        if task_mode in {"sop", "issue", "all"}:
            update["task_mode"] = task_mode

            if task_mode == "sop":
                selected_doc_types = expand_doc_type_selection(["sop"])
                selected_doc_types_strict = True
            elif task_mode == "issue":
                selected_doc_types = expand_doc_type_selection(["myservice", "gcb", "ts"])
                selected_doc_types_strict = True
            else:
                selected_doc_types = []
                selected_doc_types_strict = False

            update["selected_doc_types"] = selected_doc_types
            update["selected_doc_types_strict"] = selected_doc_types_strict
            update["doc_type_selection_skipped"] = len(selected_doc_types) == 0

            merged_pq["selected_doc_types"] = selected_doc_types
            merged_pq["doc_types_strict"] = selected_doc_types_strict
            merged_pq["doc_type_selection_skipped"] = len(selected_doc_types) == 0

        update["selected_devices"] = selected_devices
        update["selected_equip_ids"] = selected_equip_ids

        merged_pq["selected_devices"] = selected_devices
        merged_pq["selected_equip_ids"] = selected_equip_ids

    update["parsed_query"] = merged_pq
    return Command(goto="history_check", update=update)


def abbreviation_resolve_node(
    state: AgentState,
) -> Dict[str, Any]:
    """약어/동의어를 확장하고, 모호한 약어(1:N)는 사용자에게 선택을 요청.

    - 1:1 약어 + 동의어: 자동 치환하여 state["query"] 업데이트
    - 1:N 모호 약어: interrupt로 사용자 선택 후 치환
    - 확장된 쿼리가 이후 모든 노드(translate, mq, retrieve, answer)에 전파됨
    """
    already_resolved = state.get("abbreviation_resolved") is True
    if already_resolved:
        return {}

    if not agent_settings.abbreviation_expand_enabled:
        return {}

    query = str(state.get("query") or "")
    if not query:
        return {}

    try:
        from backend.llm_infrastructure.query_expansion.abbreviation_expander import (
            get_abbreviation_expander,
        )

        expander = get_abbreviation_expander(agent_settings.abbreviation_dict_path)
        result = expander.expand_query(query)
    except Exception:
        logger.debug("abbreviation_resolve_node: expander not available", exc_info=True)
        return {}

    # 매칭된 약어/동의어가 없으면 통과
    if not result.matches:
        return {}

    # ── 1:N 모호 약어 처리: 사용자에게 선택 요청 ──
    selections: Dict[str, str] = {}
    if result.ambiguous:
        abbreviation_items: List[Dict[str, Any]] = []
        concept_eng_by_id: Dict[str, str] = {}
        seen_tokens: Set[str] = set()

        for match in result.matches:
            if not match.ambiguous:
                continue
            if match.abbr_key in seen_tokens:
                continue

            candidates = [m for m in result.matches if m.abbr_key == match.abbr_key and m.ambiguous]
            options = []
            for c in candidates:
                concept_value = str(c.concept_id)
                selected_eng = str(c.primary_eng or "").strip()
                if selected_eng:
                    concept_eng_by_id[concept_value] = selected_eng
                options.append(
                    {
                        "value": concept_value,
                        "label": f"{c.primary_eng} ({c.primary_kr})"
                        if c.primary_kr
                        else c.primary_eng,
                        "eng": c.primary_eng,
                        "kr": c.primary_kr,
                    }
                )
            abbreviation_items.append(
                {
                    "token": match.token,
                    "abbr_key": match.abbr_key,
                    "options": options,
                }
            )
            seen_tokens.add(match.abbr_key)

        if abbreviation_items:
            logger.info(
                "[abbreviation_resolve] ambiguous abbreviations detected: %s",
                [item["abbr_key"] for item in abbreviation_items],
            )

            payload_base = {
                "type": "abbreviation_resolve",
                "question": query,
                "instruction": "다음 약어의 의미를 선택해주세요.",
                "abbreviations": abbreviation_items,
            }

            def _resolve_selections(candidate: Any) -> Dict[str, str] | None:
                if not isinstance(candidate, dict):
                    return None
                if candidate.get("type") != "abbreviation_resolve":
                    return None
                user_selections = candidate.get("selections")
                if not isinstance(user_selections, dict):
                    return None

                resolved: Dict[str, str] = {}
                for item in abbreviation_items:
                    abbr_key = str(item.get("abbr_key") or "").strip()
                    if not abbr_key:
                        return None
                    concept_id_raw = user_selections.get(abbr_key)
                    if concept_id_raw is None:
                        return None
                    concept_id_value = str(concept_id_raw).strip()
                    if not concept_id_value:
                        return None

                    selected_eng = concept_eng_by_id.get(concept_id_value, "")
                    if not selected_eng:
                        return None
                    resolved[abbr_key] = selected_eng

                return resolved if resolved else None

            decision = interrupt(payload_base)
            resolved = _resolve_selections(decision)
            while resolved is None:
                logger.warning("[abbreviation_resolve] invalid decision, re-prompting")
                decision = interrupt(
                    {
                        **payload_base,
                        "instruction": "선택값이 올바르지 않습니다. 모든 약어의 의미를 다시 선택해주세요.",
                    }
                )
                resolved = _resolve_selections(decision)

            selections = resolved
            for abbr_key, selected_eng in selections.items():
                logger.info(
                    "[abbreviation_resolve] user selected: '%s' → '%s'",
                    abbr_key,
                    selected_eng,
                )

    # ── 쿼리 확장 적용 (1:1 자동 + 1:N 사용자 선택) ──
    expanded = str(getattr(result, "expanded_query", query) or query)  # 1:1 + 동의어는 이미 치환됨

    # 사용자 선택한 모호 약어도 치환
    if selections:
        import re as _re

        for match in result.matches:
            if not match.ambiguous:
                continue
            selected_eng = selections.get(match.abbr_key)
            if not selected_eng:
                continue
            if selected_eng.lower() in expanded.lower():
                continue
            is_synonym = match.abbr_key.startswith("SYN:")
            if is_synonym:
                idx = expanded.lower().find(match.token.lower())
                if idx == -1:
                    continue
                original_token = expanded[idx : idx + len(match.token)]
                replacement = f"{original_token} ({selected_eng})"
                expanded = expanded[:idx] + replacement + expanded[idx + len(match.token) :]
            else:
                pattern = _re.compile(
                    rf"\b{_re.escape(match.token)}\b",
                    _re.IGNORECASE,
                )
                replacement = f"{match.token} ({selected_eng})"
                new_expanded = pattern.sub(replacement, expanded, count=1)
                if new_expanded != expanded:
                    expanded = new_expanded

    # 쿼리가 변경되었으면 state["query"] 업데이트
    update: Dict[str, Any] = {
        "abbreviation_resolved": True,
        "abbreviation_selections": selections,
        "original_query": query,  # 확장 전 원본 쿼리 (동의어 variant 생성용)
    }
    if expanded != query:
        update["query"] = expanded
        logger.info(
            "[abbreviation_resolve] query expanded: '%s' → '%s'",
            query,
            expanded,
        )

    return update


def translate_node(
    state: AgentState,
    *,
    llm: BaseLLM,
    spec: PromptSpec,
) -> Dict[str, Any]:
    """Translate query to English and Korean for better retrieval coverage.

    - If detected_language is 'en': query_en = query, translate to Korean
    - If detected_language is 'ko': query_ko = query, translate to English
    - Otherwise (ja, zh, etc.): translate to both
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
        user = _format_prompt(
            spec.translate.user,
            {
                "query": text,
                "target_language": target_name,
            },
        )
        result = _invoke_llm(llm, system, user, temperature=TEMP_TRANSLATION)
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
        # Japanese/Chinese/other - translate to both
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
        "instruction": ("승인(true) / 거절(false) / 문자열로 답변을 덮어쓰기."),
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
) -> Command[Literal["mq", "prepare_retrieve"]]:
    """Device selection node - interrupts to let user select devices (multiple).

    Args:
        state: Agent state.
        device_fetcher: Callable that returns devices (and optionally doc types).
            Devices/doc types should have 'name' and 'doc_count'.

    Returns:
        Command to proceed to 'mq' or 'prepare_retrieve' with selected devices.
    """

    def _next_goto() -> Literal["mq", "prepare_retrieve"]:
        mq_mode = state.get("mq_mode")
        attempts = state.get("attempts", 0)
        if mq_mode == "off":
            return "prepare_retrieve"
        if mq_mode == "fallback" and attempts == 0:
            return "prepare_retrieve"
        return "mq"

    goto_target = _next_goto()

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
        pq_dict = dict(state.get("parsed_query") or {})
        pq_dict.update(
            selected_devices=[],
            device_selection_skipped=True,
            selected_doc_types=[],
            doc_types_strict=False,
            doc_type_selection_skipped=True,
        )
        return Command(
            goto=goto_target,
            update={
                "available_devices": [],
                "selected_devices": [],
                "device_selection_skipped": True,
                "available_doc_types": [],
                "selected_doc_types": [],
                "selected_doc_types_strict": False,
                "doc_type_selection_skipped": True,
                "parsed_query": pq_dict,
            },
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
        pq_dict = dict(state.get("parsed_query") or {})
        pq_dict.update(
            selected_devices=[],
            device_selection_skipped=True,
            selected_doc_types=[],
            doc_types_strict=False,
            doc_type_selection_skipped=True,
        )
        return Command(
            goto=goto_target,
            update={
                "available_devices": available_devices,
                "selected_devices": [],
                "device_selection_skipped": True,
                "available_doc_types": available_doc_types,
                "selected_doc_types": [],
                "selected_doc_types_strict": False,
                "doc_type_selection_skipped": True,
                "parsed_query": pq_dict,
            },
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

    pq_dict = dict(state.get("parsed_query") or {})
    pq_dict.update(
        selected_devices=selected_devices,
        device_selection_skipped=len(selected_devices) == 0,
        selected_doc_types=selected_doc_types,
        doc_types_strict=len(selected_doc_types) > 0,
        doc_type_selection_skipped=len(selected_doc_types) == 0,
    )

    inferred_task_mode = _infer_task_mode_from_doc_types(selected_doc_types)
    if inferred_task_mode:
        pq_dict["task_mode"] = inferred_task_mode
    if inferred_task_mode == "issue":
        pq_dict["route"] = "general"
    elif inferred_task_mode == "ts":
        pq_dict["route"] = "ts"

    update_payload: Dict[str, Any] = {
        "available_devices": available_devices,
        "selected_devices": selected_devices,
        "device_selection_skipped": len(selected_devices) == 0,
        "available_doc_types": available_doc_types,
        "selected_doc_types": selected_doc_types,
        "selected_doc_types_strict": len(selected_doc_types) > 0,
        "doc_type_selection_skipped": len(selected_doc_types) == 0,
        "parsed_query": pq_dict,
    }
    if inferred_task_mode:
        update_payload["task_mode"] = inferred_task_mode
    if inferred_task_mode == "issue":
        update_payload["route"] = "general"
    elif inferred_task_mode == "ts":
        update_payload["route"] = "ts"

    return Command(
        goto=goto_target,
        update=update_payload,
    )


__all__ = [
    "AgentState",
    "ParsedQuery",
    "Route",
    "Gate",
    "PromptSpec",
    "ISSUE_CASE_EMPTY_MESSAGE",
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
    "issue_confirm_node",
    "issue_case_selection_node",
    "issue_detail_answer_node",
    "issue_sop_confirm_node",
    "judge_node",
    "should_retry",
    "retry_bump_node",
    "retry_expand_node",
    "retry_mq_node",
    "refine_queries_node",
    "human_review_node",
    "device_selection_node",
    "auto_parse_node",
    "auto_parse_confirm_node",
    "history_check_node",
    "query_rewrite_node",
]
