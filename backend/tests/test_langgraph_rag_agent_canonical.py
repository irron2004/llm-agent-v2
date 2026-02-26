from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent


class StubLLM(BaseLLM):
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model=None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(text="")


def test_canonical_retrieve_disables_pipeline_autoparse(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_resolve_effective_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "policies": {
                "rerank_enabled": True,
                "auto_parse_enabled": True,
            },
            "defaults": {
                "retrieval_top_k": 7,
                "final_top_k": 3,
            },
        }

    def fake_run_retrieval_pipeline(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "state": {
                "docs": [],
                "all_docs": [],
                "ref_json": [],
                "search_queries": [],
            },
            "executed_steps": ["retrieve"],
        }

    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.resolve_effective_config",
        fake_resolve_effective_config,
    )
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.run_retrieval_pipeline",
        fake_run_retrieval_pipeline,
    )
    monkeypatch.setattr(
        "backend.services.agents.langgraph_rag_agent.ensure_device_cache_initialized",
        lambda _: SimpleNamespace(
            device_names=["SUPRA N"], doc_type_names=["ts"], equip_id_set=set()
        ),
    )

    search_service = SimpleNamespace(reranker=None)
    agent = LangGraphRAGAgent(
        llm=StubLLM(),
        search_service=search_service,
        prompt_spec=object(),
        use_canonical_retrieval=True,
        auto_parse_enabled=True,
    )

    agent._canonical_retrieve_node({"query": "hello", "selected_devices": ["SUPRA N"]})

    assert captured["auto_parse_enabled"] is False
    assert captured["state_overrides"] == {
        "selected_devices": ["SUPRA N"],
    }
