from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm import get_llm  # noqa: E402
from backend.llm_infrastructure.llm.base import LLMResponse  # noqa: E402
from backend.services.chat_service import ChatService  # noqa: E402


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def json(self):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": self._content}}
            ]
        }

    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.called = False

    def post(self, url, json):
        self.called = True
        return _FakeResponse(self.content)


def test_vllm_adapter_uses_client(monkeypatch):
    fake_client = _FakeClient("hello vllm")
    # Swap httpx.Client constructor to return our fake client
    monkeypatch.setattr(
        "backend.llm_infrastructure.llm.engines.vllm.httpx.Client",
        lambda timeout=None: fake_client,
    )

    llm = get_llm("vllm", version="v1")
    resp = llm.generate([{"role": "user", "content": "hi"}])

    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello vllm"
    assert fake_client.called


def test_chat_service_can_use_stubbed_llm(monkeypatch):
    class _StubLLM:
        def __init__(self, **_: object) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            return LLMResponse(text=f"echo:{messages[-1]['content']}", raw={"messages": list(messages)})

    monkeypatch.setattr(
        "backend.services.chat_service.get_llm",
        lambda name, version="v1", **kwargs: _StubLLM(),
    )

    svc = ChatService(llm_method="vllm", llm_version="v1")
    resp = svc.chat("ping", history=[{"role": "user", "content": "hi"}])
    assert resp.text.startswith("echo:ping")
