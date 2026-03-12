from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _set_es_env(monkeypatch: pytest.MonkeyPatch, main_module, *, chunk_version: str) -> None:
    monkeypatch.setattr(main_module.search_settings, "backend", "es", raising=False)
    monkeypatch.setattr(main_module.search_settings, "chunk_version", chunk_version, raising=False)


def test_configure_search_service_v3_sets_chunk_v3_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = importlib.import_module("backend.api.main")

    _set_es_env(monkeypatch, main_module, chunk_version="v3")
    monkeypatch.setattr(
        main_module.search_settings,
        "v3_content_index",
        "chunk_v3_content",
        raising=False,
    )
    monkeypatch.setattr(
        main_module.search_settings,
        "v3_embed_index",
        "chunk_v3_embed_bge_m3_v1",
        raising=False,
    )
    monkeypatch.setattr(main_module.search_settings, "v3_embed_model_key", "", raising=False)

    sentinel_service = object()
    captured: dict[str, object] = {}

    class _FakeChunkService:
        @classmethod
        def from_settings(cls, **kwargs):
            captured["kwargs"] = kwargs
            return sentinel_service

    monkeypatch.setattr(main_module, "EsChunkV3SearchService", _FakeChunkService)
    monkeypatch.setattr(
        main_module, "set_search_service", lambda service: captured.setdefault("service", service)
    )

    main_module._configure_search_service()

    assert captured.get("service") is sentinel_service
    assert captured.get("kwargs") == {
        "content_index": "chunk_v3_content",
        "embed_index": "chunk_v3_embed_bge_m3_v1",
    }


def test_configure_search_service_v3_requires_embed_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = importlib.import_module("backend.api.main")

    _set_es_env(monkeypatch, main_module, chunk_version="v3")
    monkeypatch.setattr(
        main_module.search_settings,
        "v3_content_index",
        "chunk_v3_content",
        raising=False,
    )
    monkeypatch.setattr(main_module.search_settings, "v3_embed_index", "", raising=False)
    monkeypatch.setattr(main_module.search_settings, "v3_embed_model_key", "", raising=False)

    with pytest.raises(RuntimeError):
        main_module._configure_search_service()


def test_startup_reraises_when_v3_configuration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = importlib.import_module("backend.api.main")

    _set_es_env(monkeypatch, main_module, chunk_version="v3")
    monkeypatch.setattr(
        main_module,
        "_configure_search_service",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    app = main_module.create_app()
    startup_handler = app.router.on_startup[-1]

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(startup_handler())


def test_startup_keeps_previous_non_failfast_behavior_for_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = importlib.import_module("backend.api.main")

    _set_es_env(monkeypatch, main_module, chunk_version="v2")
    monkeypatch.setattr(
        main_module,
        "_configure_search_service",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    app = main_module.create_app()
    startup_handler = app.router.on_startup[-1]

    asyncio.run(startup_handler())
