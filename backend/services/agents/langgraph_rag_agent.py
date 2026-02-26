"""LangGraph RAG 에이전트 (서비스 계층).

- infra의 노드/프롬프트 헬퍼를 사용해 그래프를 조립한다.
- SearchService/LLM 등을 주입받아 실험적으로 교체하기 쉽게 설계.
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.llm.langgraph_agent import (
    AgentState,
    PromptSpec,
    Retriever,
    SearchServiceRetriever,
    answer_node,
    ask_user_after_retrieve_node,
    auto_parse_node,
    device_selection_node,
    expand_related_docs_node,
    history_check_node,
    human_review_node,
    judge_node,
    load_prompt_spec,
    mq_node,
    query_rewrite_node,
    refine_queries_node,
    retrieve_node,
    retry_bump_node,
    retry_expand_node,
    retry_mq_node,
    route_node,
    should_retry,
    st_gate_node,
    st_mq_node,
    translate_node,
)
from backend.services.retrieval_effective_config import (
    effective_config_hash,
    resolve_effective_config,
)
from backend.services.retrieval_pipeline import run_retrieval_pipeline
from backend.services.search_service import SearchService
from backend.services.device_cache import ensure_device_cache_initialized


logger = logging.getLogger(__name__)
# Ensure LangGraph node logs are emitted even if root logger is higher level
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class LangGraphRAGAgent:
    def __init__(
        self,
        *,
        llm: BaseLLM,
        search_service: SearchService,
        prompt_spec: Optional[PromptSpec] = None,
        top_k: int = 20,
        retrieval_top_k: int = 50,
        mode: str = "verified",  # base | verified
        checkpointer: Optional[MemorySaver] = None,
        ask_user_after_retrieve: bool = False,
        ask_device_selection: bool = False,
        auto_parse_enabled: bool = False,  # Auto-parse device/doc_type from query
        use_canonical_retrieval: bool = False,
        device_fetcher: Callable[[], Dict[str, Any] | list[Dict[str, Any]]] | None = None,
        event_sink: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        self.llm = llm
        self.search_service = search_service
        # Use the provided retrieval_top_k for initial retrieval
        self.retriever: Retriever = SearchServiceRetriever(search_service, top_k=retrieval_top_k)
        self.spec = prompt_spec or load_prompt_spec()
        self.top_k = top_k  # Final top_k after rerank
        self.retrieval_top_k = retrieval_top_k  # Initial retrieval top_k
        self.reranker = search_service.reranker  # Use reranker from search_service
        self.page_fetcher = getattr(search_service, "fetch_doc_pages", None)
        self.doc_fetcher = getattr(search_service, "fetch_doc_chunks", None)
        self.mode = mode
        self.ask_user_after_retrieve = ask_user_after_retrieve
        self.ask_device_selection = ask_device_selection
        self.auto_parse_enabled = auto_parse_enabled
        self.use_canonical_retrieval = use_canonical_retrieval
        self.device_fetcher = device_fetcher
        self.checkpointer = checkpointer or MemorySaver()
        self._event_sink = event_sink

        # Initialize device cache for auto_parse mode
        # Use full device_names list for better LLM parsing accuracy
        self._device_names: list[str] = []
        self._doc_type_names: list[str] = []
        self._equip_id_set: set[str] = set()
        if auto_parse_enabled:
            cache = ensure_device_cache_initialized(search_service)
            self._device_names = cache.device_names  # 전체 장비 목록 사용 (LLM 파싱용)
            self._doc_type_names = cache.doc_type_names
            self._equip_id_set = cache.equip_id_set
            logger.info(
                "Auto-parse enabled with %d devices, %d doc types, %d equip_ids",
                len(self._device_names),
                len(self._doc_type_names),
                len(self._equip_id_set),
            )

        self._graph = self._build_graph(mode)

    def _emit_event(self, event: Dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(event)
        except Exception:
            # Never allow observability hooks to break the agent execution.
            logger.exception("event_sink failed")

    def _wrap_node(self, name: str, fn):
        """Wrap a node to emit a single log event with timing and result info."""

        def _wrapped(state: AgentState, *args: Any, **kwargs: Any):
            # Log input state (key fields only)
            input_info = {
                "query": state.get("query", "")[:50] if state.get("query") else None,
                "query_en": state.get("query_en", "")[:50] if state.get("query_en") else None,
                "query_ko": state.get("query_ko", "")[:50] if state.get("query_ko") else None,
                "detected_language": state.get("detected_language"),
                "route": state.get("route"),
            }
            logger.info("[langgraph] >>> %s INPUT: %s", name, input_info)

            t0 = time.perf_counter()
            result = fn(state, *args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000

            extra_events = None
            if isinstance(result, dict):
                extra_events = result.pop("_events", None)

            # Log output result
            if result:
                output_info = {}
                for k, v in result.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, str) and len(v) > 100:
                        output_info[k] = v[:100] + "..."
                    elif isinstance(v, list) and len(v) > 0:
                        output_info[k] = f"[{len(v)} items] {str(v)[:100]}..."
                    else:
                        output_info[k] = v
                logger.info("[langgraph] <<< %s OUTPUT: %s", name, output_info)

            # Build node-specific details for display
            details = self._build_node_details(name, state, result)

            # Format elapsed time
            if elapsed >= 1000:
                elapsed_str = f"{elapsed / 1000:.1f}s"
            else:
                elapsed_str = f"{elapsed:.0f}ms"

            logger.info("[langgraph] %s (%s) %s", name, elapsed_str, details)

            self._emit_event(
                {
                    "type": "log",
                    "level": "info",
                    "node": name,
                    "ts": time.time(),
                    "elapsed_ms": round(elapsed, 1),
                    "details": details,
                    "message": f"{name} ({elapsed_str})" + (f" - {details}" if details else ""),
                }
            )

            if extra_events:
                for evt in extra_events:
                    payload: Dict[str, Any]
                    if isinstance(evt, str):
                        payload = {
                            "type": "log",
                            "level": "info",
                            "node": name,
                            "message": evt,
                            "ts": time.time(),
                        }
                    elif isinstance(evt, dict):
                        payload = dict(evt)
                        payload.setdefault("type", "log")
                        payload.setdefault("level", "info")
                        payload.setdefault("node", name)
                        payload.setdefault("ts", time.time())
                    else:
                        continue
                    self._emit_event(payload)
            return result

        return _wrapped

    def _build_node_details(
        self, name: str, state: AgentState, result: Dict[str, Any] | None
    ) -> str:
        """Build human-readable details for each node type."""
        details_parts = []

        if name == "route":
            route = result.get("route") if result else state.get("route")
            if route:
                details_parts.append(f"→ {route}")

        elif name == "auto_parse":
            if result:
                pq = result.get("parsed_query", {})
                device = (pq.get("device_names") or [None])[0]
                doc_type = (pq.get("doc_types") or [None])[0]
                equip_id = (pq.get("equip_ids") or [None])[0]
                if device:
                    details_parts.append(f"장비: {device}")
                if doc_type:
                    details_parts.append(f"문서: {doc_type}")
                if equip_id:
                    details_parts.append(f"장비ID: {equip_id}")
                if not device and not doc_type and not equip_id:
                    details_parts.append("파싱 결과 없음")

        elif name == "translate":
            if result:
                lang = state.get("detected_language") or result.get("detected_language")
                query_en = result.get("query_en")
                query_ko = result.get("query_ko")
                if lang:
                    details_parts.append(f"언어: {lang}")
                if query_en:
                    preview = query_en[:50] + "..." if len(query_en) > 50 else query_en
                    details_parts.append(f"EN: {preview}")
                if query_ko:
                    preview = query_ko[:50] + "..." if len(query_ko) > 50 else query_ko
                    details_parts.append(f"KO: {preview}")

        elif name == "mq":
            if result:
                route = state.get("route", "")
                key = f"{route}_mq_list"
                mq_list = result.get(key, [])
                if mq_list:
                    details_parts.append(f"{len(mq_list)}개 쿼리 생성")

        elif name == "st_gate":
            gate = result.get("st_gate") if result else None
            if gate:
                details_parts.append(f"→ {gate}")

        elif name == "st_mq":
            if result:
                queries = result.get("search_queries", [])
                details_parts.append(f"{len(queries)}개 검색 쿼리")

        elif name == "retrieve" or name == "retrieve_retry":
            if result:
                docs = result.get("docs", [])
                details_parts.append(f"{len(docs)}개 문서 검색")

        elif name == "expand_related":
            if result:
                ref_json = result.get("answer_ref_json", [])
                details_parts.append(f"{len(ref_json)}개 문서 확장")

        elif name == "answer":
            if result:
                answer = result.get("answer", "")
                reasoning = result.get("reasoning")
                details_parts.append(f"{len(answer)}자")
                if reasoning:
                    details_parts.append(f"reasoning: {len(reasoning)}자")

        elif name == "judge":
            if result:
                judge = result.get("judge", {})
                faithful = judge.get("faithful")
                if faithful is not None:
                    details_parts.append("✓ 충실" if faithful else "✗ 불충실")

        elif name == "retry_expand":
            attempts = result.get("attempts") if result else state.get("attempts")
            expand_k = result.get("expand_top_k") if result else state.get("expand_top_k")
            details_parts.append(f"문서 확장 증가 →{expand_k or 40}개")
            if attempts:
                details_parts.append(f"attempt {attempts}")

        elif name == "retry_mq":
            attempts = result.get("attempts") if result else state.get("attempts")
            details_parts.append("MQ 재생성")
            if attempts:
                details_parts.append(f"attempt {attempts}")

        elif name == "history_check":
            if result:
                needs = result.get("needs_history", False)
                details_parts.append("후속 질문" if needs else "독립 질문")

        elif name == "query_rewrite":
            if result:
                rewritten = result.get("query", "")
                if rewritten:
                    preview = rewritten[:60] + "..." if len(rewritten) > 60 else rewritten
                    details_parts.append(f"→ {preview}")

        return " | ".join(details_parts) if details_parts else ""

    def _collect_canonical_state_overrides(self, state: AgentState) -> Dict[str, Any] | None:
        keys = (
            "parsed_query",
            "selected_devices",
            "selected_doc_types",
            "selected_doc_types_strict",
            "selected_equip_ids",
            "selected_doc_ids",
            "search_queries",
            "skip_mq",
            "query_en",
            "query_ko",
            "detected_language",
        )
        overrides: Dict[str, Any] = {}
        for key in keys:
            if key in state and state.get(key) is not None:
                overrides[key] = state.get(key)
        return overrides or None

    def _canonical_retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        query = str(state.get("query") or "")
        reranker_available = self.reranker is not None and hasattr(self.reranker, "rerank")

        effective_config = resolve_effective_config(
            query,
            ["retrieve"],
            False,
            True,
            final_top_k=self.top_k,
            retrieval_top_k=self.retrieval_top_k,
            auto_parse=self.auto_parse_enabled,
            skip_mq=True,
            reranker_available=reranker_available,
        )
        policies = cast(dict[str, Any], effective_config.get("policies", {}))
        defaults = cast(dict[str, Any], effective_config.get("defaults", {}))

        pipeline_result = run_retrieval_pipeline(
            query=query,
            llm=self.llm,
            spec=self.spec,
            retriever=self.retriever,
            reranker=self.reranker,
            rerank_enabled=bool(policies.get("rerank_enabled", True)),
            retrieval_top_k=int(defaults.get("retrieval_top_k", self.retrieval_top_k)),
            final_top_k=int(defaults.get("final_top_k", self.top_k)),
            steps=["retrieve"],
            deterministic=True,
            auto_parse_enabled=False,
            state_overrides=self._collect_canonical_state_overrides(state),
            effective_config=effective_config,
        )

        canonical_state = cast(dict[str, Any], pipeline_result.get("state", {}))
        executed_steps = cast(list[str], pipeline_result.get("executed_steps", []))
        effective_config_payload = dict(effective_config)
        effective_config_payload["executed_steps"] = executed_steps

        canonical_run_id = uuid.uuid4().hex
        canonical_hash = effective_config_hash(effective_config_payload)

        update: Dict[str, Any] = {
            "docs": list(canonical_state.get("docs") or []),
            "all_docs": list(canonical_state.get("all_docs") or []),
            "ref_json": list(canonical_state.get("ref_json") or []),
            "search_queries": list(canonical_state.get("search_queries") or []),
            "canonical_run_id": canonical_run_id,
            "canonical_effective_config_hash": canonical_hash,
        }
        human_action = state.get("human_action")
        merged_human_action: Dict[str, Any] = (
            dict(human_action) if isinstance(human_action, dict) else {}
        )
        merged_human_action["canonical_run_id"] = canonical_run_id
        merged_human_action["canonical_effective_config_hash"] = canonical_hash
        update["human_action"] = merged_human_action
        return update

    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        if self.use_canonical_retrieval:
            return self._canonical_retrieve_node(state)
        return retrieve_node(
            state,
            retriever=self.retriever,
            reranker=self.reranker,
            retrieval_top_k=self.retrieval_top_k,
            final_top_k=self.top_k,
        )

    def _build_graph(self, mode: str):
        builder = StateGraph(AgentState)

        # Common nodes
        builder.add_node(
            "route",
            self._wrap_node("route", functools.partial(route_node, llm=self.llm, spec=self.spec)),
        )
        builder.add_node(
            "mq", self._wrap_node("mq", functools.partial(mq_node, llm=self.llm, spec=self.spec))
        )
        builder.add_node(
            "st_gate",
            self._wrap_node(
                "st_gate", functools.partial(st_gate_node, llm=self.llm, spec=self.spec)
            ),
        )
        builder.add_node(
            "st_mq",
            self._wrap_node("st_mq", functools.partial(st_mq_node, llm=self.llm, spec=self.spec)),
        )
        retrieve_fn = self._retrieve_node
        builder.add_node("retrieve", self._wrap_node("retrieve", retrieve_fn))
        # retry 경로용 retrieve: ask_user 없이 바로 answer로 연결
        builder.add_node("retrieve_retry", self._wrap_node("retrieve_retry", retrieve_fn))
        builder.add_node(
            "expand_related",
            self._wrap_node(
                "expand_related",
                functools.partial(
                    expand_related_docs_node,
                    page_fetcher=self.page_fetcher,
                    doc_fetcher=self.doc_fetcher,
                ),
            ),
        )
        builder.add_node(
            "answer",
            self._wrap_node("answer", functools.partial(answer_node, llm=self.llm, spec=self.spec)),
        )
        builder.add_node(
            "judge",
            self._wrap_node("judge", functools.partial(judge_node, llm=self.llm, spec=self.spec)),
        )

        # Auto-parse mode: auto_parse → translate → route → mq
        # This ensures route receives translated query (query_en)
        if self.auto_parse_enabled:
            builder.add_node(
                "auto_parse",
                self._wrap_node(
                    "auto_parse",
                    functools.partial(
                        auto_parse_node,
                        llm=self.llm,
                        spec=self.spec,
                        device_names=self._device_names,
                        doc_type_names=self._doc_type_names,
                        equip_id_set=self._equip_id_set,
                    ),
                ),
            )
            # Translate node: translate query to EN and KO for better retrieval
            builder.add_node(
                "translate",
                self._wrap_node(
                    "translate",
                    functools.partial(
                        translate_node,
                        llm=self.llm,
                        spec=self.spec,
                    ),
                ),
            )
            # Chat history nodes: history_check → (conditional) → query_rewrite
            builder.add_node(
                "history_check",
                self._wrap_node(
                    "history_check",
                    functools.partial(history_check_node, llm=self.llm),
                ),
            )
            builder.add_node(
                "query_rewrite",
                self._wrap_node(
                    "query_rewrite",
                    functools.partial(query_rewrite_node, llm=self.llm),
                ),
            )
            # Flow: START → auto_parse → history_check → [query_rewrite] → translate → route → mq
            builder.add_edge(START, "auto_parse")
            builder.add_edge("auto_parse", "history_check")
            builder.add_conditional_edges(
                "history_check",
                lambda s: "query_rewrite" if s.get("needs_history") else "translate",
                {"query_rewrite": "query_rewrite", "translate": "translate"},
            )
            builder.add_edge("query_rewrite", "translate")
            builder.add_edge("translate", "route")
            builder.add_edge("route", "mq")
        # Device selection node (optional HIL) - legacy mode
        elif self.ask_device_selection:
            builder.add_node(
                "device_selection",
                self._wrap_node(
                    "device_selection",
                    functools.partial(device_selection_node, device_fetcher=self.device_fetcher),
                ),
            )
            builder.add_edge(START, "route")
            builder.add_edge("route", "device_selection")
            # device_selection_node returns Command(goto="mq"), so no explicit edge needed
        else:
            # Default flow: START → route → mq
            builder.add_edge(START, "route")
            builder.add_edge("route", "mq")

        builder.add_edge("mq", "st_gate")
        builder.add_edge("st_gate", "st_mq")
        builder.add_edge("st_mq", "retrieve")

        # ask_user_after_retrieve 옵션: retrieve 후 사용자 확인
        if self.ask_user_after_retrieve:
            builder.add_node("ask_user", self._wrap_node("ask_user", ask_user_after_retrieve_node))
            # refine_and_retrieve: 사용자 피드백 후 다시 retrieve로
            builder.add_node(
                "refine_and_retrieve", self._wrap_node("refine_and_retrieve", lambda s: {})
            )

            builder.add_edge("retrieve", "ask_user")
            # ask_user_after_retrieve_node는 Command를 반환하므로 conditional edge 불필요
            # Command의 goto="expand_related" 또는 goto="refine_and_retrieve"가 라우팅 담당
            builder.add_edge("refine_and_retrieve", "retrieve")
            # retry 경로의 retrieve_retry는 ask_user 없이 바로 expand_related로
            builder.add_edge("retrieve_retry", "expand_related")
        else:
            builder.add_edge("retrieve", "expand_related")
            builder.add_edge("retrieve_retry", "expand_related")

        builder.add_edge("expand_related", "answer")

        builder.add_edge("answer", "judge")

        if mode == "base":
            builder.add_edge("judge", END)
            return builder.compile(
                checkpointer=self.checkpointer if self.ask_user_after_retrieve else None
            )

        # verified: add retry/human with different strategies
        # retry_expand: 1st retry - use more docs (20→40)
        builder.add_node("retry_expand", self._wrap_node("retry_expand", retry_expand_node))
        # retry_bump + refine_queries: 2nd retry - refine queries and re-retrieve
        builder.add_node("retry_bump", self._wrap_node("retry_bump", retry_bump_node))
        builder.add_node(
            "refine_queries",
            self._wrap_node(
                "refine_queries",
                functools.partial(refine_queries_node, llm=self.llm, spec=self.spec),
            ),
        )
        # retry_mq: 3rd+ retry - regenerate MQ from scratch
        builder.add_node(
            "retry_mq",
            self._wrap_node(
                "retry_mq", functools.partial(retry_mq_node, llm=self.llm, spec=self.spec)
            ),
        )
        builder.add_node("human_review", self._wrap_node("human_review", human_review_node))
        # 호환성을 위해 'retry' 별칭을 두고 바로 retry_bump로 연결
        builder.add_node("retry", self._wrap_node("retry", lambda s: {}))
        builder.add_node("done", self._wrap_node("done", lambda s: {}))

        # Conditional edges based on retry strategy
        # - retry_expand: 1st unfaithful → expand more docs (no re-retrieval)
        # - retry: 2nd unfaithful → refine queries and re-retrieve
        # - retry_mq: 3rd+ unfaithful → regenerate MQ from scratch
        builder.add_conditional_edges(
            "judge",
            should_retry,
            {
                "done": "done",
                "retry_expand": "retry_expand",
                "retry": "retry_bump",
                "retry_mq": "retry_mq",
                "human": "human_review",
            },
        )

        # retry_expand: just increase expand_top_k and go back to expand_related
        builder.add_edge("retry_expand", "expand_related")

        # retry (refine_queries): refine queries and re-retrieve
        builder.add_edge("retry_bump", "refine_queries")
        builder.add_edge("refine_queries", "retrieve_retry")

        # retry_mq: regenerate MQ from scratch → mq → st_gate → st_mq → retrieve
        builder.add_edge("retry_mq", "mq")

        builder.add_edge("retry", "retry_bump")
        builder.add_edge("human_review", "retry_bump")
        builder.add_edge("human_review", "done")
        builder.add_edge("done", END)

        return builder.compile(checkpointer=self.checkpointer)

    def run(
        self,
        query: str,
        *,
        attempts: int = 0,
        max_attempts: int = 1,
        thread_id: str | None = None,
        state_overrides: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """그래프 실행. kwargs는 LangGraph invoke config로 전달 가능."""
        tid = thread_id or str(uuid.uuid4())
        # HIL 비활성화 모드에서는 human_review 건너뛰기
        skip_human = not self.ask_user_after_retrieve and not self.ask_device_selection
        state: AgentState = {
            "query": query,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "thread_id": tid,
            "_skip_human_review": skip_human,
        }
        if state_overrides:
            state.update(state_overrides)
        config = kwargs.pop("config", {})
        # thread_id is required for checkpointer
        # recursion_limit: max_attempts=3일 때 재시도 루프를 위해 충분한 값 설정
        config = {
            **config,
            "configurable": {**config.get("configurable", {}), "thread_id": tid},
            "recursion_limit": 100,
        }

        return self._graph.invoke(state, config=cast(Any, config))


__all__ = ["LangGraphRAGAgent"]
