from types import SimpleNamespace
from typing import Any, cast

from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.routers import retrieval as retrieval_router
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.retrieval_run_store import RetrievalRunSnapshotStore


class _FakeLLMOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeLLM:
    def generate(self, messages: list[dict[str, str]], **_: object) -> _FakeLLMOutput:
        del messages
        return _FakeLLMOutput("setup")


class _DriftLLM:
    def __init__(self) -> None:
        self._st_mq_counter = 0

    def generate(self, messages: list[dict[str, str]], **_: object) -> _FakeLLMOutput:
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""

        if "router-system" in system:
            return _FakeLLMOutput("general")
        if "st-gate-system" in system:
            return _FakeLLMOutput("no_st")
        if "st-mq-system" in system:
            self._st_mq_counter += 1
            return _FakeLLMOutput(f'["mq-drift-{self._st_mq_counter}"]')
        if "translate-system" in system:
            return _FakeLLMOutput(user)
        return _FakeLLMOutput("[]")


class _DriftSearchService:
    def search(self, query: str, top_k: int = 10, **_: object) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                doc_id=f"{query}-doc-{idx}",
                content=f"content-{query}-{idx}",
                score=1.0 - (idx * 0.01),
                metadata={
                    "title": f"title-{query}",
                    "doc_type": "manual",
                    "device_name": "supra",
                    "equip_id": "eq-1",
                    "page": idx + 1,
                },
                raw_text=f"raw-{query}-{idx}",
            )
            for idx in range(top_k)
        ]


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _contains_key_recursive(payload: object, target_key: str) -> bool:
    if isinstance(payload, dict):
        if target_key in payload:
            return True
        return any(
            _contains_key_recursive(value, target_key) for value in payload.values()
        )
    if isinstance(payload, list):
        return any(_contains_key_recursive(item, target_key) for item in payload)
    return False


def _install_retrieval_overrides(client: TestClient) -> None:
    prompt_spec = SimpleNamespace(
        router=SimpleNamespace(system="route", user="{sys.query}"),
        translate=None,
    )
    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _FakeLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: prompt_spec
    app.dependency_overrides[dependencies.get_reranker] = lambda: None


def _install_replay_overrides(
    client: TestClient, run_store: RetrievalRunSnapshotStore
) -> None:
    drift_llm = _DriftLLM()
    prompt_spec = SimpleNamespace(
        router=SimpleNamespace(system="router-system", user="{sys.query}"),
        setup_mq=SimpleNamespace(system="setup-mq-system", user="{sys.query}"),
        ts_mq=SimpleNamespace(system="ts-mq-system", user="{sys.query}"),
        general_mq=SimpleNamespace(system="general-mq-system", user="{sys.query}"),
        st_gate=SimpleNamespace(
            system="st-gate-system",
            user="{sys.query}|{setup_mq}|{ts_mq}|{general_mq}",
        ),
        st_mq=SimpleNamespace(
            system="st-mq-system",
            user="{sys.query}|{setup_mq}|{ts_mq}|{general_mq}|{st_gate}",
        ),
        translate=SimpleNamespace(
            system="translate-system",
            user="{query}|{target_language}",
        ),
    )
    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: drift_llm
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: prompt_spec
    app.dependency_overrides[dependencies.get_reranker] = lambda: None
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _DriftSearchService()
    )
    app.dependency_overrides[retrieval_router.get_retrieval_run_store] = (
        lambda: run_store
    )


def test_retrieval_run_contract_and_hash_stability(client: TestClient) -> None:
    _install_retrieval_overrides(client)

    payload = {
        "query": "PM 점검 절차 알려줘",
        "steps": ["retrieve"],
        "debug": False,
        "deterministic": True,
    }
    resp = client.post("/api/retrieval/run", json=payload)

    assert resp.status_code == 200
    data = cast(dict[str, Any], resp.json())

    assert {
        "run_id",
        "effective_config",
        "effective_config_hash",
        "warnings",
        "steps",
        "docs",
    }.issubset(data.keys())
    assert isinstance(data["run_id"], str)
    assert len(cast(str, data["effective_config_hash"])) == 64

    effective_config = cast(dict[str, Any], data["effective_config"])
    assert effective_config["deterministic"] is True
    assert effective_config["requested_steps"] == ["retrieve"]
    assert effective_config["executed_steps"] == ["translate", "retrieve"]
    assert cast(dict[str, Any], effective_config["request_vs_applied"])[
        "final_top_k"
    ] == {
        "requested": None,
        "applied": 5,
        "source": "env_default",
    }
    assert cast(dict[str, Any], effective_config["request_vs_applied"])[
        "rerank_enabled"
    ] == {
        "requested": None,
        "applied": False,
        "source": "env_default",
    }

    steps = cast(dict[str, dict[str, Any]], data["steps"])
    assert "translate" in steps
    assert "retrieve" in steps
    assert steps["retrieve"]["status"] == "completed"

    docs = cast(list[dict[str, Any]], data["docs"])
    assert docs
    first_doc = docs[0]
    assert {"doc_id", "title", "snippet", "metadata"}.issubset(first_doc.keys())
    assert "raw_text" not in first_doc
    assert "raw_text" not in str(first_doc)

    resp_repeat = client.post("/api/retrieval/run", json=payload)
    assert resp_repeat.status_code == 200
    data_repeat = cast(dict[str, Any], resp_repeat.json())
    assert data_repeat["effective_config_hash"] == data["effective_config_hash"]


def test_retrieval_run_accepts_new_contract_fields_and_records_rerank_step(
    client: TestClient,
) -> None:
    _install_retrieval_overrides(client)

    payload = {
        "query": "contract alignment",
        "steps": ["rerank"],
        "deterministic": True,
        "final_top_k": 2,
        "rerank_enabled": True,
        "auto_parse": True,
        "skip_mq": True,
    }
    resp = client.post("/api/retrieval/run", json=payload)

    assert resp.status_code == 200
    data = cast(dict[str, Any], resp.json())

    warnings = cast(list[str], data["warnings"])
    assert any("Reranker unavailable" in warning for warning in warnings)

    effective_config = cast(dict[str, Any], data["effective_config"])
    request_vs_applied = cast(dict[str, Any], effective_config["request_vs_applied"])
    assert request_vs_applied["final_top_k"] == {
        "requested": 2,
        "applied": 2,
        "source": "request",
    }
    assert request_vs_applied["rerank_enabled"] == {
        "requested": True,
        "applied": False,
        "source": "policy",
    }

    policies = cast(dict[str, Any], effective_config["policies"])
    assert policies["skip_mq"] is True
    assert policies["auto_parse_enabled"] is True

    steps = cast(dict[str, dict[str, Any]], data["steps"])
    assert "auto_parse" in steps
    assert "rerank" in steps
    assert cast(dict[str, Any], steps["rerank"]["artifacts"])["rerank_applied"] is False

    docs = cast(list[dict[str, Any]], data["docs"])
    assert len(docs) == 2


def test_retrieval_run_stops_at_route_and_omits_docs(client: TestClient) -> None:
    _install_retrieval_overrides(client)

    payload = {
        "query": "route only please",
        "steps": ["route"],
        "deterministic": True,
    }
    resp = client.post("/api/retrieval/run", json=payload)

    assert resp.status_code == 200
    data = cast(dict[str, Any], resp.json())
    steps = cast(dict[str, dict[str, Any]], data["steps"])

    assert "route" in steps
    assert "retrieve" not in steps
    assert data["docs"] == []


def test_retrieval_run_store_snapshot_no_raw_text(client: TestClient) -> None:
    store = RetrievalRunSnapshotStore(ttl_seconds=300)
    _install_replay_overrides(client, store)

    resp = client.post(
        "/api/retrieval/run",
        json={
            "query": "prevent mq drift",
            "steps": ["retrieve"],
            "deterministic": False,
        },
    )
    assert resp.status_code == 200
    run_id = cast(str, resp.json()["run_id"])

    get_resp = client.get(f"/api/retrieval/runs/{run_id}")
    assert get_resp.status_code == 200
    snapshot = cast(dict[str, Any], get_resp.json())
    assert snapshot["run_id"] == run_id
    assert _contains_key_recursive(snapshot, "raw_text") is False


def test_retrieval_run_store_replay_reuses_search_queries(client: TestClient) -> None:
    store = RetrievalRunSnapshotStore(ttl_seconds=300)
    _install_replay_overrides(client, store)

    payload = {
        "query": "prevent mq drift",
        "steps": ["retrieve"],
        "deterministic": False,
    }

    first = client.post("/api/retrieval/run", json=payload)
    assert first.status_code == 200
    first_data = cast(dict[str, Any], first.json())
    first_run_id = cast(str, first_data["run_id"])
    first_queries = cast(
        list[str],
        cast(dict[str, Any], cast(dict[str, Any], first_data["steps"])["st_mq"])[
            "artifacts"
        ]["search_queries"],
    )
    first_doc_ids = [
        cast(str, item["doc_id"])
        for item in cast(list[dict[str, Any]], first_data["docs"])
    ]

    # guardrail이 원본 쿼리를 보존하므로 drift 검증 대신 replay 동일성만 검증
    replay = client.post(
        "/api/retrieval/run",
        json={
            **payload,
            "replay_run_id": first_run_id,
        },
    )
    assert replay.status_code == 200
    replay_data = cast(dict[str, Any], replay.json())
    replay_queries = cast(
        list[str],
        cast(dict[str, Any], cast(dict[str, Any], replay_data["steps"])["st_mq"])[
            "artifacts"
        ]["search_queries"],
    )
    replay_doc_ids = [
        cast(str, item["doc_id"])
        for item in cast(list[dict[str, Any]], replay_data["docs"])
    ]

    assert replay_queries == first_queries
    assert replay_doc_ids == first_doc_ids


def test_retrieval_run_store_ttl_eviction_with_fake_clock(client: TestClient) -> None:
    clock = _FakeClock(start=50.0)
    store = RetrievalRunSnapshotStore(ttl_seconds=5.0, clock=clock)
    _install_replay_overrides(client, store)

    resp = client.post(
        "/api/retrieval/run",
        json={
            "query": "ttl check",
            "steps": ["retrieve"],
        },
    )
    assert resp.status_code == 200
    run_id = cast(str, resp.json()["run_id"])

    before_expiry = client.get(f"/api/retrieval/runs/{run_id}")
    assert before_expiry.status_code == 200

    clock.now += 6.0
    after_expiry = client.get(f"/api/retrieval/runs/{run_id}")
    assert after_expiry.status_code == 404




class _RawTextLeakSearchService:
    """Search service that returns doc with empty content but raw_text with secret."""

    def search(self, query: str, top_k: int = 10, **_: object) -> list[RetrievalResult]:
        # Return a doc with empty content but raw_text containing secret
        return [
            RetrievalResult(
                doc_id="secret-doc-1",
                content="",  # Empty content
                score=1.0,
                metadata={
                    "doc_type": "manual",
                },
                raw_text="SECRET_RAW_TEXT_SHOULD_NOT_LEAK",
            )
        ]


def _install_raw_text_leak_overrides(
    client: TestClient, run_store: RetrievalRunSnapshotStore
) -> None:
    """Install overrides for testing raw_text leakage."""
    prompt_spec = SimpleNamespace(
        router=SimpleNamespace(system="route", user="{sys.query}"),
        translate=None,
    )
    app = cast(Any, client.app)
    app.dependency_overrides[dependencies.get_default_llm] = lambda: _FakeLLM()
    app.dependency_overrides[dependencies.get_prompt_spec_cached] = lambda: prompt_spec
    app.dependency_overrides[dependencies.get_reranker] = lambda: None
    app.dependency_overrides[dependencies.get_search_service] = (
        lambda: _RawTextLeakSearchService()
    )
    app.dependency_overrides[retrieval_router.get_retrieval_run_store] = (
        lambda: run_store
    )


def test_retrieval_run_does_not_leak_raw_text_when_content_empty(
    client: TestClient,
) -> None:
    """Verify that raw_text is never leaked into response when content is empty.

    This test would have failed before the fix because _to_doc_item used
    raw_text as a fallback for title when content was empty.
    """
    store = RetrievalRunSnapshotStore(ttl_seconds=300)
    _install_raw_text_leak_overrides(client, store)

    payload = {
        "query": "secret query",
        "steps": ["retrieve"],
        "deterministic": True,
    }
    resp = client.post("/api/retrieval/run", json=payload)

    assert resp.status_code == 200
    data = cast(dict[str, Any], resp.json())

    # Verify the response JSON does NOT contain the secret raw_text
    response_str = str(data)
    assert "SECRET_RAW_TEXT_SHOULD_NOT_LEAK" not in response_str, (
        "raw_text leaked into response!"
    )

    # Also verify snapshot does not contain raw_text
    run_id = cast(str, data["run_id"])
    get_resp = client.get(f"/api/retrieval/runs/{run_id}")
    assert get_resp.status_code == 200
    snapshot = cast(dict[str, Any], get_resp.json())
    snapshot_str = str(snapshot)
    assert "SECRET_RAW_TEXT_SHOULD_NOT_LEAK" not in snapshot_str, (
        "raw_text leaked into snapshot!"
    )


def test_retrieval_run_handles_non_list_search_queries(
    client: TestClient,
) -> None:
    """Verify that non-list search_queries values don't cause errors.

    This tests the defensive type-check added for search_queries extraction.
    The pipeline should handle cases where state['search_queries'] is not a list.
    """
    _install_retrieval_overrides(client)

    # Save original function and patch it to return a non-list search_queries
    original_func = retrieval_router.run_retrieval_pipeline

    def _patched_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_func(*args, **kwargs)
        # Simulate a case where search_queries is a string instead of a list
        result["state"]["search_queries"] = "not_a_list_queries"
        return result

    retrieval_router.run_retrieval_pipeline = _patched_pipeline

    try:
        payload = {
            "query": "test query",
            "steps": ["retrieve"],
            "deterministic": True,
        }
        resp = client.post("/api/retrieval/run", json=payload)

        # Should succeed without errors (the defensive check prevents iteration errors)
        assert resp.status_code == 200
        data = cast(dict[str, Any], resp.json())

        # Verify search_queries is empty/absent, not character-split
        steps = cast(dict[str, Any], data["steps"])
        retrieve_step = steps.get("retrieve", {})
        artifacts = retrieve_step.get("artifacts", {})
        stored_queries = artifacts.get("search_queries", [])

        # The defensive check should result in empty list, NOT character-split string
        assert stored_queries == [] or stored_queries is None, (
            f"Expected empty list or None, got {stored_queries!r}"
        )

        # Also verify the stored snapshot has empty search_queries
        run_id = cast(str, data["run_id"])
        get_resp = client.get(f"/api/retrieval/runs/{run_id}")
        assert get_resp.status_code == 200
        snapshot = cast(dict[str, Any], get_resp.json())
        snapshot_queries = snapshot.get("search_queries", [])
        assert snapshot_queries == [] or snapshot_queries is None, (
            f"Snapshot should have empty search_queries, got {snapshot_queries!r}"
        )
    finally:
        # Restore original function
        retrieval_router.run_retrieval_pipeline = original_func
