from __future__ import annotations

from copy import deepcopy
from typing import cast

from backend.services.retrieval_effective_config import (
    effective_config_hash,
    resolve_effective_config,
)


def _request_vs_applied(config: dict[str, object]) -> dict[str, dict[str, object]]:
    payload = config["request_vs_applied"]
    assert isinstance(payload, dict)
    return cast(dict[str, dict[str, object]], payload)


def test_request_vs_applied_entries_have_requested_applied_and_source() -> None:
    config = resolve_effective_config(
        "source contract",
        ["retrieve"],
        False,
        True,
    )

    tracked = _request_vs_applied(config)
    assert set(tracked.keys()) == {
        "retrieval_top_k",
        "final_top_k",
        "rerank_enabled",
        "auto_parse",
        "skip_mq",
    }

    for entry in tracked.values():
        assert {"requested", "applied", "source"}.issubset(entry.keys())
        assert entry["source"] in {"request", "env_default", "policy"}


def test_rerank_enabled_source_is_policy_when_server_forces_false() -> None:
    config = resolve_effective_config(
        "policy rerank",
        ["retrieve"],
        False,
        True,
        rerank_enabled=True,
        reranker_available=False,
    )

    rerank_entry = _request_vs_applied(config)["rerank_enabled"]
    assert rerank_entry == {
        "requested": True,
        "applied": False,
        "source": "policy",
    }


def test_omitted_fields_are_marked_env_default() -> None:
    config = resolve_effective_config(
        "defaults",
        ["retrieve"],
        False,
        True,
        final_top_k=None,
        retrieval_top_k=None,
        rerank_enabled=None,
        auto_parse=None,
        skip_mq=None,
    )

    tracked = _request_vs_applied(config)
    for key in (
        "retrieval_top_k",
        "final_top_k",
        "rerank_enabled",
        "auto_parse",
        "skip_mq",
    ):
        assert tracked[key]["source"] == "env_default"


def test_effective_config_hash_changes_when_source_changes() -> None:
    config = resolve_effective_config("hash", ["retrieve"], False, True)
    baseline_hash = effective_config_hash(config)

    mutated = deepcopy(config)
    tracked = _request_vs_applied(mutated)
    tracked["final_top_k"]["source"] = "request"

    assert effective_config_hash(mutated) != baseline_hash


def test_trace_fields_are_not_in_effective_config_payload() -> None:
    config = resolve_effective_config("trace exclusion", ["retrieve"], False, True)

    assert "trace" not in config
    assert "trace_id" not in config
    assert "traceparent" not in config
    assert "tracestate" not in config
