from __future__ import annotations

from ..dependencies import (
    get_default_llm,
    get_prompt_spec_cached,
    get_reranker,
    get_search_service,
)
from ...services.retrieval_effective_config import (
    CANONICAL_STEPS,
    resolve_effective_config,
    effective_config_hash,
)
from ...services.retrieval_pipeline import run_retrieval_pipeline
from ...services.retrieval_run_store import (
    RetrievalRunStore,
    get_default_retrieval_run_store,
)
from ...services.search_service import SearchService

from typing import Any, ClassVar, cast
import re
import uuid
from collections.abc import Mapping
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


class RetrievalRunRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "PM 점검 절차 알려줘",
                "steps": ["translate", "retrieve", "rerank"],
                "debug": True,
                "deterministic": False,
            }
        }
    )

    query: str
    steps: list[str] | None = None
    debug: bool = False
    deterministic: bool = False
    replay_run_id: str | None = None
    final_top_k: int | None = Field(default=None, ge=1)
    rerank_enabled: bool | None = None
    auto_parse: bool | None = None
    skip_mq: bool | None = None
    device_names: list[str] | None = None
    doc_types: list[str] | None = None
    doc_types_strict: bool | None = None
    equip_ids: list[str] | None = None


class DocItem(BaseModel):
    doc_id: str
    title: str | None = None
    snippet: str | None = None
    score: float | None = None
    metadata: dict[str, object] | None = None
    page: int | None = None
    page_image_url: str | None = None
    expanded_pages: list[int] | None = None
    expanded_page_urls: list[str] | None = None


class RetrievalRunResponse(BaseModel):
    run_id: str
    effective_config: Mapping[str, object]
    effective_config_hash: str
    warnings: list[str] = Field(default_factory=list)
    steps: dict[str, dict[str, object]] = Field(default_factory=dict)
    docs: list[DocItem] = Field(default_factory=list)
    trace: TraceContext | None = None


class TraceContext(BaseModel):
    trace_id: str
    traceparent: str | None = None
    tracestate: str | None = None


_TRACEPARENT_PATTERN = re.compile(r"^[\da-f]{2}-[\da-f]{32}-[\da-f]{16}-[\da-f]{2}$", re.IGNORECASE)


def _resolve_trace_context(
    traceparent: str | None,
    tracestate: str | None,
) -> TraceContext:
    cleaned_traceparent = (traceparent or "").strip()
    if not _TRACEPARENT_PATTERN.match(cleaned_traceparent):
        return TraceContext(trace_id=uuid.uuid4().hex)

    segments = cleaned_traceparent.split("-")
    trace_id_segment, parent_id_segment = segments[1], segments[2]
    if int(trace_id_segment, 16) == 0 or int(parent_id_segment, 16) == 0:
        return TraceContext(trace_id=uuid.uuid4().hex)

    cleaned_tracestate = (tracestate or "").strip() or None
    return TraceContext(
        trace_id=uuid.uuid4().hex,
        traceparent=cleaned_traceparent,
        tracestate=cleaned_tracestate,
    )


class _SearchServiceRetriever:
    def __init__(self, search_service: SearchService):
        self._search_service = search_service

    def retrieve(self, query: str, *, top_k: int = 8, **kwargs: object) -> list[object]:
        if not hasattr(self._search_service, "search"):
            raise RuntimeError("Search service not configured")

        search_kwargs: dict[str, Any] = {"top_k": top_k}
        for key in ("device_name", "device_names", "equip_ids", "doc_types"):
            if kwargs.get(key) is not None:
                search_kwargs[key] = kwargs[key]

        try:
            return list(cast(Any, self._search_service).search(query, **search_kwargs))
        except TypeError:
            return list(cast(Any, self._search_service).search(query, top_k=top_k))


def _resolve_retriever(search_service: SearchService) -> Any:
    retriever = getattr(search_service, "retriever", None)
    if retriever is not None and hasattr(retriever, "retrieve"):
        return retriever
    return _SearchServiceRetriever(search_service)


def _extract_minimal_metadata(metadata: Mapping[str, object] | None) -> dict[str, object] | None:
    if metadata is None:
        return None

    allowed_keys = (
        "doc_type",
        "device_name",
        "equip_id",
        "chapter",
        "chunk_id",
        "page",
        "page_start",
        "source",
    )
    payload = {
        key: metadata[key] for key in allowed_keys if key in metadata and metadata[key] is not None
    }
    return payload or None


def _extract_page_value(metadata: Mapping[str, object] | None) -> int | None:
    if metadata is None:
        return None
    page_raw = metadata.get("page_start") or metadata.get("page")
    if isinstance(page_raw, int):
        return page_raw
    try:
        page = int(str(page_raw))
    except (TypeError, ValueError):
        return None
    return page if page >= 0 else None


def _extract_expanded_pages(
    metadata: Mapping[str, object] | None, page: int | None
) -> list[int] | None:
    if metadata is not None and isinstance(metadata.get("expanded_pages"), list):
        collected: list[int] = []
        for item in cast(list[object], metadata["expanded_pages"]):
            try:
                num = int(str(item))
            except (TypeError, ValueError):
                continue
            if num >= 0:
                collected.append(num)
        if collected:
            return sorted(set(collected))

    if page is not None:
        return [page]
    return None


def _to_doc_item(doc: object) -> DocItem:
    doc_id = str(getattr(doc, "doc_id", ""))
    metadata_raw = getattr(doc, "metadata", None)
    metadata = (
        cast(Mapping[str, object], metadata_raw) if isinstance(metadata_raw, Mapping) else None
    )

    title = ""
    if metadata:
        title = str(metadata.get("title") or metadata.get("doc_description") or "").strip()
    if not title and doc_id:
        # Use doc_id as fallback instead of raw_text to avoid leaking full document body
        title = f"[Document: {doc_id}]"

    snippet_source = str(getattr(doc, "content", "") or "")
    snippet = snippet_source[:300] + ("..." if len(snippet_source) > 300 else "")

    score_raw = getattr(doc, "score", None)
    score = float(score_raw) if isinstance(score_raw, (int, float)) else None

    page = _extract_page_value(metadata)
    page_image_url = (
        f"/api/assets/docs/{doc_id}/pages/{page}" if page is not None and doc_id else None
    )
    expanded_pages = _extract_expanded_pages(metadata, page)
    expanded_page_urls = (
        [f"/api/assets/docs/{doc_id}/pages/{p}" for p in expanded_pages]
        if expanded_pages and doc_id
        else None
    )

    return DocItem(
        doc_id=doc_id,
        title=title or None,
        snippet=snippet or None,
        score=score,
        metadata=_extract_minimal_metadata(metadata),
        page=page,
        page_image_url=page_image_url,
        expanded_pages=expanded_pages,
        expanded_page_urls=expanded_page_urls,
    )


def _safe_int(value: object, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def get_retrieval_run_store() -> RetrievalRunStore:
    return get_default_retrieval_run_store()


def _sanitize_request(req: RetrievalRunRequest) -> dict[str, object]:
    return {
        "query": req.query,
        "steps": list(req.steps or []),
        "debug": bool(req.debug),
        "deterministic": bool(req.deterministic),
        "replay_run_id": req.replay_run_id,
        "final_top_k": req.final_top_k,
        "rerank_enabled": req.rerank_enabled,
        "auto_parse": req.auto_parse,
        "skip_mq": req.skip_mq,
        "device_names": list(req.device_names or []),
        "doc_types": list(req.doc_types or []),
        "doc_types_strict": req.doc_types_strict,
        "equip_ids": list(req.equip_ids or []),
    }


def _sanitize_ranked_docs(docs: list[DocItem]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for doc in docs:
        item: dict[str, object] = {"doc_id": doc.doc_id}
        if doc.title is not None:
            item["title"] = doc.title
        if doc.score is not None:
            item["score"] = doc.score
        if doc.metadata is not None:
            item["metadata"] = dict(doc.metadata)
        if doc.page is not None:
            item["page"] = doc.page
        payload.append(item)
    return payload


def _build_run_snapshot(
    *,
    run_id: str,
    req: RetrievalRunRequest,
    effective_config: Mapping[str, object],
    effective_config_hash_value: str,
    executed_steps: list[str],
    steps_payload: dict[str, dict[str, object]],
    docs: list[DocItem],
    search_queries: list[str],
) -> dict[str, object]:
    doc_ids = [doc.doc_id for doc in docs]
    return {
        "run_id": run_id,
        "request": _sanitize_request(req),
        "effective_config": dict(effective_config),
        "effective_config_hash": effective_config_hash_value,
        "executed_steps": list(executed_steps),
        "search_queries": list(search_queries),
        "selected_doc_ids": doc_ids,
        "doc_ids": doc_ids,
        "ranked_docs": _sanitize_ranked_docs(docs),
        "steps": steps_payload,
    }


@router.post("/run", response_model=RetrievalRunResponse)
async def run_retrieval(
    req: RetrievalRunRequest,
    traceparent: str | None = Header(default=None),
    tracestate: str | None = Header(default=None),
    search_service: SearchService = Depends(get_search_service),
    llm=Depends(get_default_llm),
    prompt_spec=Depends(get_prompt_spec_cached),
    reranker=Depends(get_reranker),
    run_store: RetrievalRunStore = Depends(get_retrieval_run_store),
) -> RetrievalRunResponse:
    trace_context = _resolve_trace_context(traceparent, tracestate)
    ignored_steps = [s for s in (req.steps or []) if s not in CANONICAL_STEPS]

    warnings: list[str] = []
    if ignored_steps:
        warnings.append(f"Ignored unknown steps: {', '.join(ignored_steps)}")

    reranker_available = reranker is not None and hasattr(reranker, "rerank")

    effective_config = resolve_effective_config(
        req.query,
        req.steps,
        req.debug,
        req.deterministic,
        final_top_k=req.final_top_k,
        rerank_enabled=req.rerank_enabled,
        auto_parse=req.auto_parse,
        skip_mq=req.skip_mq,
        reranker_available=reranker_available,
    )

    policies = cast(dict[str, object], effective_config.get("policies", {}))
    defaults = cast(dict[str, object], effective_config.get("defaults", {}))

    if req.rerank_enabled is True and not reranker_available:
        warnings.append("Reranker unavailable; rerank disabled")

    state_overrides: dict[str, object] = {}
    selected_devices = [str(item).strip() for item in (req.device_names or []) if str(item).strip()]
    selected_doc_types = [str(item).strip() for item in (req.doc_types or []) if str(item).strip()]
    selected_equip_ids = [str(item).strip() for item in (req.equip_ids or []) if str(item).strip()]

    if selected_devices:
        state_overrides["selected_devices"] = selected_devices
    if selected_doc_types:
        state_overrides["selected_doc_types"] = selected_doc_types
        state_overrides["selected_doc_types_strict"] = bool(req.doc_types_strict)
    if selected_equip_ids:
        state_overrides["selected_equip_ids"] = selected_equip_ids

    if bool(policies.get("skip_mq", False)):
        state_overrides["skip_mq"] = True
        if "search_queries" not in state_overrides:
            cleaned_query = req.query.strip()
            if cleaned_query:
                state_overrides["search_queries"] = [cleaned_query]

    if req.replay_run_id:
        replay_snapshot = run_store.get(req.replay_run_id)
        if replay_snapshot is None:
            raise HTTPException(
                status_code=404,
                detail=f"Retrieval run not found: {req.replay_run_id}",
            )

        state_overrides["skip_mq"] = True

        replay_queries = replay_snapshot.get("search_queries")
        if isinstance(replay_queries, list):
            cleaned_queries = [
                str(item).strip()
                for item in replay_queries
                if isinstance(item, str) and str(item).strip()
            ]
            if cleaned_queries:
                state_overrides["search_queries"] = cleaned_queries

        replay_doc_ids = replay_snapshot.get("selected_doc_ids")
        if not isinstance(replay_doc_ids, list):
            replay_doc_ids = replay_snapshot.get("doc_ids")
        if isinstance(replay_doc_ids, list):
            cleaned_doc_ids = [
                str(item).strip()
                for item in replay_doc_ids
                if isinstance(item, str) and str(item).strip()
            ]
            if cleaned_doc_ids:
                state_overrides["selected_doc_ids"] = cleaned_doc_ids

    try:
        pipeline_result = run_retrieval_pipeline(
            query=req.query,
            llm=llm,
            spec=prompt_spec,
            retriever=_resolve_retriever(search_service),
            reranker=reranker,
            rerank_enabled=bool(policies.get("rerank_enabled", True)),
            retrieval_top_k=_safe_int(defaults.get("retrieval_top_k"), 20),
            final_top_k=_safe_int(defaults.get("final_top_k"), 20),
            steps=req.steps,
            deterministic=req.deterministic,
            auto_parse_enabled=cast(bool | None, req.auto_parse),
            state_overrides=state_overrides or None,
            effective_config=effective_config,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    executed_steps = cast(list[str], pipeline_result.get("executed_steps", []))
    effective_config_payload = dict(effective_config)
    effective_config_payload["executed_steps"] = executed_steps

    steps_payload = cast(dict[str, dict[str, object]], pipeline_result.get("steps", {}))
    state = cast(Mapping[str, object], pipeline_result.get("state", {}))
    docs_raw = state.get("docs", []) if isinstance(state, Mapping) else []

    docs: list[DocItem] = []
    if any(step in executed_steps for step in ("retrieve", "rerank")) and isinstance(
        docs_raw, list
    ):
        docs = [_to_doc_item(doc) for doc in docs_raw]

    run_id = uuid.uuid4().hex
    # Defensive: ensure search_queries is a list before iterating
    sq_raw = state.get("search_queries")
    search_queries: list[str] = []
    if isinstance(sq_raw, list):
        search_queries = [
            str(item).strip() for item in sq_raw if isinstance(item, str) and str(item).strip()
        ]
    config_hash = effective_config_hash(effective_config_payload)
    snapshot = _build_run_snapshot(
        run_id=run_id,
        req=req,
        effective_config=effective_config_payload,
        effective_config_hash_value=config_hash,
        executed_steps=executed_steps,
        steps_payload=steps_payload,
        docs=docs,
        search_queries=search_queries,
    )
    run_store.put(run_id, snapshot)

    return RetrievalRunResponse(
        run_id=run_id,
        effective_config=effective_config_payload,
        effective_config_hash=config_hash,
        warnings=warnings,
        steps=steps_payload,
        docs=docs,
        trace=trace_context,
    )


@router.get("/runs/{run_id}")
async def get_retrieval_run(
    run_id: str,
    run_store: RetrievalRunStore = Depends(get_retrieval_run_store),
) -> dict[str, object]:
    snapshot = run_store.get(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Retrieval run not found: {run_id}")
    return snapshot


__all__ = [
    "router",
    "RetrievalRunRequest",
    "RetrievalRunResponse",
    "DocItem",
    "get_retrieval_run_store",
]
