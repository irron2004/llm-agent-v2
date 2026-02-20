from __future__ import annotations

from typing import Iterable

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm.langgraph_agent import history_check_node


class StaticLLM(BaseLLM):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
        self.calls = 0

    def generate(self, messages: Iterable[dict[str, str]], *, response_model=None, **kwargs):
        if response_model is not None:
            raise NotImplementedError
        self.calls += 1
        return LLMResponse(text=self.text)


class FailingLLM(BaseLLM):
    def generate(self, messages: Iterable[dict[str, str]], *, response_model=None, **kwargs):
        raise RuntimeError("llm failure")


def test_history_check_node_uses_llm_true_decision() -> None:
    llm = StaticLLM('{"needs_history": true}')
    state = {
        "query": "그럼 이거는 어떻게 해?",
        "chat_history": [
            {
                "user_text": "SUPRA N power calibration 절차 알려줘",
                "assistant_text": "먼저 interlock 상태를 확인하세요.",
                "doc_ids": ["doc1"],
            }
        ],
    }

    result = history_check_node(state, llm=llm)

    assert result["needs_history"] is True
    assert llm.calls == 1


def test_history_check_node_uses_llm_false_decision() -> None:
    llm = StaticLLM('{"needs_history": false}')
    state = {
        "query": "SUPRA N power calibration 절차 알려줘",
        "chat_history": [
            {
                "user_text": "이전 질문",
                "assistant_text": "이전 답변",
                "doc_ids": [],
            }
        ],
    }

    result = history_check_node(state, llm=llm)

    assert result["needs_history"] is False
    assert llm.calls == 1


def test_history_check_node_falls_back_to_rules_when_llm_output_invalid() -> None:
    llm = StaticLLM("not-json")
    state = {
        "query": "아까 말한 내용 더 자세히 알려줘",
        "chat_history": [
            {
                "user_text": "SUPRA N 에러 해결 방법",
                "assistant_text": "원인은 pressure drift입니다.",
                "doc_ids": [],
            }
        ],
    }

    result = history_check_node(state, llm=llm)

    assert result["needs_history"] is True
    assert llm.calls == 1


def test_history_check_node_returns_false_without_chat_history() -> None:
    llm = StaticLLM('{"needs_history": true}')
    state = {"query": "SUPRA N setup 절차"}

    result = history_check_node(state, llm=llm)

    assert result["needs_history"] is False
    assert llm.calls == 0


def test_history_check_node_falls_back_to_rules_when_llm_raises() -> None:
    llm = FailingLLM()
    state = {
        "query": "그거 다시 설명해줘",
        "chat_history": [
            {
                "user_text": "RF match 문제 원인 분석",
                "assistant_text": "첫 번째로 pressure를 확인하세요.",
                "doc_ids": [],
            }
        ],
    }

    result = history_check_node(state, llm=llm)

    assert result["needs_history"] is True
