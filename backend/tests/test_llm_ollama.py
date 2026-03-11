from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm import get_llm  # noqa: E402
from backend.llm_infrastructure.llm.base import LLMResponse  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.called = False
        self.last_url = ""
        self.last_json: dict[str, object] | None = None

    def post(self, url, json):
        self.called = True
        self.last_url = url
        self.last_json = json
        return _FakeResponse(self.payload)


class _StructuredOutput(BaseModel):
    answer: str


def test_ollama_adapter_uses_native_api(monkeypatch):
    fake_client = _FakeClient(
        {
            "message": {
                "role": "assistant",
                "content": "hello ollama",
            }
        }
    )
    monkeypatch.setattr(
        "backend.llm_infrastructure.llm.engines.ollama.httpx.Client",
        lambda timeout=None: fake_client,
    )

    llm = get_llm("ollama", version="v1", base_url="http://localhost:11434", model="qwen2.5:14b")
    resp = llm.generate([{"role": "user", "content": "hi"}])

    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello ollama"
    assert fake_client.called
    assert fake_client.last_url.endswith("/api/chat")
    assert isinstance(fake_client.last_json, dict)
    assert fake_client.last_json.get("model") == "qwen2.5:14b"


def test_ollama_adapter_supports_openai_compatible_base(monkeypatch):
    fake_client = _FakeClient(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello compat",
                    }
                }
            ]
        }
    )
    monkeypatch.setattr(
        "backend.llm_infrastructure.llm.engines.ollama.httpx.Client",
        lambda timeout=None: fake_client,
    )

    llm = get_llm("ollama", version="v1", base_url="http://localhost:11434/v1")
    resp = llm.generate([{"role": "user", "content": "hi"}])

    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello compat"
    assert fake_client.last_url.endswith("/v1/chat/completions")


def test_ollama_adapter_parses_response_model(monkeypatch):
    fake_client = _FakeClient(
        {
            "message": {
                "role": "assistant",
                "content": '{"answer":"ok"}',
            }
        }
    )
    monkeypatch.setattr(
        "backend.llm_infrastructure.llm.engines.ollama.httpx.Client",
        lambda timeout=None: fake_client,
    )

    llm = get_llm("ollama", version="v1", base_url="http://localhost:11434")
    parsed = llm.generate(
        [{"role": "user", "content": "hi"}],
        response_model=_StructuredOutput,
    )

    assert isinstance(parsed, _StructuredOutput)
    assert parsed.answer == "ok"
    assert isinstance(fake_client.last_json, dict)
    assert fake_client.last_json.get("format") == "json"


def test_ollama_adapter_normalizes_known_model_alias(monkeypatch):
    fake_client = _FakeClient(
        {
            "message": {
                "role": "assistant",
                "content": "alias ok",
            }
        }
    )
    monkeypatch.setattr(
        "backend.llm_infrastructure.llm.engines.ollama.httpx.Client",
        lambda timeout=None: fake_client,
    )

    llm = get_llm(
        "ollama",
        version="v1",
        base_url="http://localhost:11434",
        model="openai/gpt-oss-20b",
    )
    resp = llm.generate([{"role": "user", "content": "hi"}])

    assert isinstance(resp, LLMResponse)
    assert resp.text == "alias ok"
    assert isinstance(fake_client.last_json, dict)
    assert fake_client.last_json.get("model") == "gpt-oss:120b"
