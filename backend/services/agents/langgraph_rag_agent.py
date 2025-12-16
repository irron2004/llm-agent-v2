"""LangGraph RAG 에이전트 (서비스 계층).

- infra의 노드/프롬프트 헬퍼를 사용해 그래프를 조립한다.
- SearchService/LLM 등을 주입받아 실험적으로 교체하기 쉽게 설계.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.llm.langgraph_agent import (
    AgentState,
    PromptSpec,
    Retriever,
    SearchServiceRetriever,
    answer_node,
    human_review_node,
    judge_node,
    load_prompt_spec,
    mq_node,
    refine_queries_node,
    retrieve_node,
    retry_bump_node,
    route_node,
    should_retry,
    st_gate_node,
    st_mq_node,
)
from backend.services.search_service import SearchService


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
    MAX_TOP_K = 3  # tighter cap to avoid overlong prompts

    def __init__(
        self,
        *,
        llm: BaseLLM,
        search_service: SearchService,
        prompt_spec: Optional[PromptSpec] = None,
        top_k: int = 8,
        mode: str = "verified",  # base | verified
        checkpointer: Optional[MemorySaver] = None,
    ) -> None:
        self.llm = llm
        capped_top_k = min(top_k, self.MAX_TOP_K)
        self.retriever: Retriever = SearchServiceRetriever(search_service, top_k=capped_top_k)
        self.spec = prompt_spec or load_prompt_spec()
        self.top_k = capped_top_k
        self.mode = mode
        self.checkpointer = checkpointer or MemorySaver()
        self._graph = self._build_graph(mode)

    def _wrap_node(self, name: str, fn):
        """Wrap a node to log start/end without leaking user content."""

        def _wrapped(state: AgentState, *args: Any, **kwargs: Any):
            import time

            meta = {
                "route": state.get("route"),
                "st_gate": state.get("st_gate"),
                "attempts": state.get("attempts"),
                "max_attempts": state.get("max_attempts"),
                "thread_id": state.get("thread_id"),
            }
            logger.info("[langgraph] node start: %s %s", name, meta)
            t0 = time.perf_counter()
            result = fn(state, *args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("[langgraph] node done: %s (%.1f ms)", name, elapsed)
            return result

        return _wrapped

    def _build_graph(self, mode: str):
        builder = StateGraph(AgentState)

        # Common nodes
        builder.add_node("route", self._wrap_node("route", functools.partial(route_node, llm=self.llm, spec=self.spec)))
        builder.add_node("mq", self._wrap_node("mq", functools.partial(mq_node, llm=self.llm, spec=self.spec)))
        builder.add_node("st_gate", self._wrap_node("st_gate", functools.partial(st_gate_node, llm=self.llm, spec=self.spec)))
        builder.add_node("st_mq", self._wrap_node("st_mq", functools.partial(st_mq_node, llm=self.llm, spec=self.spec)))
        builder.add_node(
            "retrieve",
            self._wrap_node("retrieve", functools.partial(retrieve_node, retriever=self.retriever, top_k=self.top_k)),
        )
        builder.add_node("answer", self._wrap_node("answer", functools.partial(answer_node, llm=self.llm, spec=self.spec)))
        builder.add_node("judge", self._wrap_node("judge", functools.partial(judge_node, llm=self.llm, spec=self.spec)))

        builder.add_edge(START, "route")
        builder.add_edge("route", "mq")
        builder.add_edge("mq", "st_gate")
        builder.add_edge("st_gate", "st_mq")
        builder.add_edge("st_mq", "retrieve")
        builder.add_edge("retrieve", "answer")
        builder.add_edge("answer", "judge")

        if mode == "base":
            builder.add_edge("judge", END)
            return builder.compile()

        # verified: add retry/human
        builder.add_node("retry_bump", self._wrap_node("retry_bump", retry_bump_node))
        builder.add_node("refine_queries", self._wrap_node("refine_queries", functools.partial(refine_queries_node, llm=self.llm)))
        builder.add_node("human_review", self._wrap_node("human_review", human_review_node))
        # 호환성을 위해 'retry' 별칭을 두고 바로 retry_bump로 연결
        builder.add_node("retry", self._wrap_node("retry", lambda s: {}))
        builder.add_node("done", self._wrap_node("done", lambda s: {}))

        builder.add_conditional_edges(
            "judge",
            should_retry,
            {"done": "done", "retry": "retry_bump", "human": "human_review"},
        )

        builder.add_edge("retry_bump", "refine_queries")
        builder.add_edge("refine_queries", "retrieve")
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """그래프 실행. kwargs는 LangGraph invoke config로 전달 가능."""

        import uuid

        tid = thread_id or str(uuid.uuid4())
        state: AgentState = {
            "query": query,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "thread_id": tid,
        }
        config = kwargs.pop("config", {})
        # thread_id is required for checkpointer
        config = {**config, "configurable": {**config.get("configurable", {}), "thread_id": tid}}

        return self._graph.invoke(state, config=config)


__all__ = ["LangGraphRAGAgent"]
