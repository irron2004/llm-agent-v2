from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterator, cast

from fastapi.testclient import TestClient

from backend.api.routers import conversations as conversations_router
from backend.services.chat_history_service import ChatTurn


class _InMemoryChatHistoryService:
    def __init__(self) -> None:
        self._turns: dict[str, list[ChatTurn]] = {}

    def get_next_turn_id(self, session_id: str) -> int:
        return len(self._turns.get(session_id, [])) + 1

    def save_turn(self, turn: ChatTurn) -> str:
        if turn.ts is None:
            turn.ts = datetime.utcnow()
        self._turns.setdefault(turn.session_id, []).append(turn)
        return f"{turn.session_id}:{turn.turn_id}"

    def get_session(self, session_id: str) -> list[ChatTurn]:
        return self._turns.get(session_id, [])


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _assert_search_queries_limits(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().startswith("search_queries") and isinstance(item, list):
                assert len(item) <= 5
                assert all(isinstance(q, str) and len(q) <= 120 for q in item)
            _assert_search_queries_limits(item)
    elif isinstance(value, list):
        for item in value:
            _assert_search_queries_limits(item)


def test_conversations_retrieval_meta_roundtrip(client_minimal: TestClient) -> None:
    service = _InMemoryChatHistoryService()
    client_any = cast(Any, client_minimal)
    app = cast(Any, client_any.app)
    app.dependency_overrides[conversations_router.get_chat_history_service] = (
        lambda: service
    )

    oversized_map = {f"blob_{i}": "z" * 300 for i in range(60)}
    payload = {
        "user_text": "Question",
        "assistant_text": "Answer",
        "doc_refs": [],
        "title": "Session title",
        "retrieval_meta": {
            "query": "x" * 500 + " reach me at qa-user@example.com",
            "notes": "Call +1 (415) 555-1212 for updates",
            "search_queries_raw": [
                "alpha qa-user@example.com +1 (415) 555-1212 " + ("y" * 200),
                "beta " + ("y" * 200),
                "gamma " + ("y" * 200),
                "delta " + ("y" * 200),
                "epsilon " + ("y" * 200),
                "zeta " + ("y" * 200),
            ],
            "oversized": oversized_map,
        },
    }

    post_resp = client_minimal.post("/api/conversations/session-1/turns", json=payload)
    assert post_resp.status_code == 200
    post_body = post_resp.json()
    assert "retrieval_meta" in post_body
    retrieval_meta = post_body["retrieval_meta"]

    assert retrieval_meta.get("truncated") is True
    serialized = json.dumps(
        retrieval_meta, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    assert len(serialized.encode("utf-8")) <= 8 * 1024

    if "safe_subset" in retrieval_meta:
        _assert_search_queries_limits(retrieval_meta["safe_subset"])

    all_strings = list(_iter_strings(retrieval_meta))
    assert all(len(value) <= 256 for value in all_strings)
    assert all("qa-user@example.com" not in value for value in all_strings)
    assert all("555-1212" not in value for value in all_strings)
    if "safe_subset" in retrieval_meta:
        subset_strings = list(_iter_strings(retrieval_meta["safe_subset"]))
        assert any("[REDACTED]" in value for value in subset_strings)

    get_resp = client_minimal.get("/api/conversations/session-1")
    assert get_resp.status_code == 200
    get_body = get_resp.json()
    turns = get_body["turns"]
    assert len(turns) == 1
    assert turns[0]["retrieval_meta"] == retrieval_meta
