from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langgraph.types import Command

from backend.llm_infrastructure.llm import langgraph_agent as langgraph_agent_module


class _AmbiguousExpander:
    def __init__(self) -> None:
        self._concepts = {
            1: {"primary_eng": "Aspect Ratio", "primary_kr": "종횡비"},
            2: {"primary_eng": "As Received", "primary_kr": "입고 상태"},
        }

    def expand_query(self, _query: str) -> Any:
        return SimpleNamespace(
            ambiguous=True,
            matches=[
                SimpleNamespace(
                    ambiguous=True,
                    abbr_key="AR",
                    token="AR",
                    concept_id=1,
                    primary_eng="Aspect Ratio",
                    primary_kr="종횡비",
                ),
                SimpleNamespace(
                    ambiguous=True,
                    abbr_key="AR",
                    token="AR",
                    concept_id=2,
                    primary_eng="As Received",
                    primary_kr="입고 상태",
                ),
            ],
        )


class _NonAmbiguousExpander:
    def expand_query(self, _query: str) -> Any:
        return SimpleNamespace(ambiguous=False, matches=[])


def test_abbreviation_resolve_reprompts_until_valid_selection(monkeypatch: Any) -> None:
    decisions = iter(
        [
            {"type": "abbreviation_resolve", "selections": {"AR": "999"}},
            {"type": "abbreviation_resolve", "selections": {"AR": "1"}},
        ]
    )
    payloads: list[dict[str, Any]] = []

    def _fake_interrupt(payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        return next(decisions)

    monkeypatch.setattr(langgraph_agent_module, "interrupt", _fake_interrupt)
    monkeypatch.setattr(
        "backend.llm_infrastructure.query_expansion.abbreviation_expander.get_abbreviation_expander",
        lambda _path: _AmbiguousExpander(),
    )
    monkeypatch.setattr(
        langgraph_agent_module.agent_settings,
        "abbreviation_expand_enabled",
        True,
        raising=False,
    )

    command = langgraph_agent_module.abbreviation_resolve_node({"query": "AR alarm"})

    assert isinstance(command, Command)
    assert command.goto == "history_check"
    assert command.update == {
        "abbreviation_resolved": True,
        "abbreviation_selections": {"AR": "Aspect Ratio"},
        "query": "Aspect Ratio (AR) alarm",
    }
    assert len(payloads) == 2
    assert payloads[0]["instruction"] == "다음 약어의 의미를 선택해주세요."
    assert "선택값이 올바르지 않습니다" in payloads[1]["instruction"]


def test_abbreviation_resolve_skips_interrupt_when_not_ambiguous(monkeypatch: Any) -> None:
    def _should_not_interrupt(_payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("interrupt should not be called for non-ambiguous query")

    monkeypatch.setattr(langgraph_agent_module, "interrupt", _should_not_interrupt)
    monkeypatch.setattr(
        "backend.llm_infrastructure.query_expansion.abbreviation_expander.get_abbreviation_expander",
        lambda _path: _NonAmbiguousExpander(),
    )
    monkeypatch.setattr(
        langgraph_agent_module.agent_settings,
        "abbreviation_expand_enabled",
        True,
        raising=False,
    )

    command = langgraph_agent_module.abbreviation_resolve_node({"query": "normal query"})

    assert isinstance(command, Command)
    assert command.goto == "history_check"
    assert command.update == {}
