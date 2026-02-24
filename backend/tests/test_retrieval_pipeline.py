# pyright: reportMissingImports=false
from __future__ import annotations

from collections.abc import Iterable
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import PromptSpec
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.retrieval_pipeline import run_retrieval_pipeline


def _prompt(name: str, user: str = "") -> PromptTemplate:
    return PromptTemplate(name=name, version="v1", system="", user=user, raw={})


def _make_spec() -> PromptSpec:
    empty = _prompt("empty", user="")
    return PromptSpec(
        router=_prompt("router", user="ROUTE:{sys.query}"),
        setup_mq=_prompt("setup_mq", user="MQ_SETUP:{sys.query}"),
        ts_mq=_prompt("ts_mq", user="MQ_TS:{sys.query}"),
        general_mq=_prompt("general_mq", user="MQ_GENERAL:{sys.query}"),
        st_gate=_prompt("st_gate", user="ST_GATE:{sys.query}|{setup_mq}|{ts_mq}|{general_mq}"),
        st_mq=_prompt("st_mq", user="ST_MQ:{sys.query}|{st_gate}"),
        setup_ans=empty,
        ts_ans=empty,
        general_ans=empty,
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
        translate=_prompt("translate", user="TRANSLATE:{query}|{target_language}"),
    )


class SpyLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.user_messages: list[str] = []

    def generate(self, messages: Iterable[dict[str, str]], *, response_model=None, **kwargs):
        if response_model is not None:
            raise NotImplementedError
        message_list = list(messages)
        user_text = message_list[-1]["content"] if message_list else ""
        self.user_messages.append(user_text)

        if user_text.startswith("ROUTE:"):
            return LLMResponse(text='{"route":"general"}')
        if user_text.startswith("TRANSLATE:"):
            payload = user_text[len("TRANSLATE:") :]
            source, target = payload.split("|", 1)
            if target.strip() == "English":
                return LLMResponse(text=f"translated-en::{source.strip()}")
            if target.strip() == "Korean":
                return LLMResponse(text=f"translated-ko::{source.strip()}")
            return LLMResponse(text=source.strip())
        if user_text.startswith("ST_GATE:"):
            return LLMResponse(text="no_st")
        if user_text.startswith("ST_MQ:"):
            return LLMResponse(text='{"queries":["q1","q2","q3"]}')
        if user_text.startswith("MQ_"):
            return LLMResponse(text='{"queries":["mq1","mq2","mq3"]}')

        return LLMResponse(text="general")


class SpyRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query: str, top_k: int = 8, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, **kwargs})
        return [
            RetrievalResult(
                doc_id=f"doc::{query}",
                content="snippet",
                score=1.0,
                metadata={"chunk_id": "chunk-1", "doc_type": "setup"},
                raw_text="snippet",
            )
        ]


class SpyReranker:
    def __init__(self) -> None:
        self.called = False

    def rerank(self, query: str, docs: list[RetrievalResult], top_k: int = 20):
        self.called = True
        return docs[:top_k]


def test_retrieval_pipeline_stops_at_route_without_calling_retriever() -> None:
    llm = SpyLLM()
    retriever = SpyRetriever()

    result = run_retrieval_pipeline(
        query="route-only question",
        llm=llm,
        spec=_make_spec(),
        retriever=retriever,
        steps=["route"],
    )

    assert result["executed_steps"] == ["route"]
    assert retriever.calls == []
    assert result["steps"]["route"]["artifacts"]["route"] == "general"


def test_retrieval_pipeline_deterministic_bypasses_mq_and_stable_queries() -> None:
    llm = SpyLLM()
    retriever = SpyRetriever()

    first = run_retrieval_pipeline(
        query="pump pressure alarm",
        llm=llm,
        spec=_make_spec(),
        retriever=retriever,
        steps=["retrieve"],
        deterministic=True,
        rerank_enabled=False,
    )
    second = run_retrieval_pipeline(
        query="pump pressure alarm",
        llm=llm,
        spec=_make_spec(),
        retriever=retriever,
        steps=["retrieve"],
        deterministic=True,
        rerank_enabled=False,
    )

    assert first["executed_steps"] == ["translate", "retrieve"]
    assert second["executed_steps"] == ["translate", "retrieve"]
    assert first["state"]["search_queries"] == second["state"]["search_queries"]
    assert first["state"]["search_queries"] == ["translated-en::pump pressure alarm"]
    assert all(not msg.startswith("MQ_") for msg in llm.user_messages)
    assert all(not msg.startswith("ST_GATE:") for msg in llm.user_messages)
    assert all(not msg.startswith("ST_MQ:") for msg in llm.user_messages)


def test_retrieval_pipeline_propagates_effective_config_and_disables_rerank() -> None:
    llm = SpyLLM()
    retriever = SpyRetriever()
    reranker = SpyReranker()
    effective_config = {
        "policies": {
            "rerank_enabled": False,
        }
    }

    result = run_retrieval_pipeline(
        query="config propagation check",
        llm=llm,
        spec=_make_spec(),
        retriever=retriever,
        reranker=reranker,
        effective_config=effective_config,
        steps=["retrieve"],
        deterministic=True,
    )

    assert reranker.called is False
    assert result["effective_config"]["policies"]["rerank_enabled"] is False
    assert result["steps"]["retrieve"]["artifacts"]["rerank_enabled"] is False


def test_retrieval_pipeline_execute_until_rerank_records_step_artifact() -> None:
    llm = SpyLLM()
    retriever = SpyRetriever()
    reranker = SpyReranker()

    result = run_retrieval_pipeline(
        query="rerank execution",
        llm=llm,
        spec=_make_spec(),
        retriever=retriever,
        reranker=reranker,
        rerank_enabled=True,
        final_top_k=3,
        steps=["rerank"],
        deterministic=True,
    )

    assert result["executed_steps"] == ["translate", "retrieve", "rerank"]
    assert reranker.called is True
    rerank_artifacts = result["steps"]["rerank"]["artifacts"]
    assert rerank_artifacts["rerank_applied"] is True
    assert rerank_artifacts["top_k"] == 3
    assert rerank_artifacts["doc_ids"]


def test_retrieval_pipeline_auto_parse_override_controls_step_execution() -> None:
    llm = SpyLLM()
    retriever = SpyRetriever()

    include_auto_parse = run_retrieval_pipeline(
        query="include auto parse",
        llm=llm,
        spec=_make_spec(),
        retriever=retriever,
        steps=["route"],
        auto_parse_enabled=True,
        deterministic=True,
    )
    assert include_auto_parse["executed_steps"] == ["auto_parse", "route"]

    skip_auto_parse = run_retrieval_pipeline(
        query="skip auto parse",
        llm=llm,
        spec=_make_spec(),
        retriever=retriever,
        steps=["auto_parse", "route"],
        auto_parse_enabled=False,
        deterministic=True,
    )
    assert skip_auto_parse["executed_steps"] == ["route"]
