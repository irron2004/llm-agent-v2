from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ..llm_infrastructure.llm.base import BaseLLM
from ..llm_infrastructure.llm.langgraph_agent import (
    AgentState,
    PromptSpec,
    Retriever,
    auto_parse_node,
    mq_node,
    retrieve_node,
    route_node,
    st_gate_node,
    st_mq_node,
    translate_node,
)

CANONICAL_RETRIEVAL_STEPS: tuple[str, ...] = (
    "auto_parse",
    "translate",
    "route",
    "mq",
    "st_gate",
    "st_mq",
    "retrieve",
    "rerank",
)

_MQ_STEPS = {"mq", "st_gate", "st_mq"}


def _clean_query(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _normalize_steps(steps: Sequence[str] | None) -> list[str]:
    if steps is None:
        return []
    return [step for step in steps if step in CANONICAL_RETRIEVAL_STEPS]


def _collect_prerequisites(
    step: str,
    *,
    deterministic: bool,
    collector: set[str],
) -> None:
    prereq_map: dict[str, tuple[str, ...]] = {
        "mq": ("route",),
        "st_gate": ("mq",),
        "st_mq": ("st_gate",),
        "retrieve": ("translate",) if deterministic else ("st_mq",),
        "rerank": ("retrieve",),
    }
    for dep in prereq_map.get(step, ()):
        if deterministic and dep in _MQ_STEPS:
            continue
        if dep in collector:
            continue
        collector.add(dep)
        _collect_prerequisites(dep, deterministic=deterministic, collector=collector)


def _resolve_execution_steps(
    steps: Sequence[str] | None,
    *,
    deterministic: bool,
    auto_parse_enabled: bool | None,
) -> list[str]:
    requested = _normalize_steps(steps)
    if requested:
        target_step = max(requested, key=CANONICAL_RETRIEVAL_STEPS.index)
    else:
        target_step = "retrieve"

    needed: set[str] = {target_step, *requested}
    _collect_prerequisites(target_step, deterministic=deterministic, collector=needed)

    if auto_parse_enabled is True:
        needed.add("auto_parse")
    elif auto_parse_enabled is False:
        needed.discard("auto_parse")

    if deterministic:
        needed = {step for step in needed if step not in _MQ_STEPS}

    return [step for step in CANONICAL_RETRIEVAL_STEPS if step in needed]


def _build_artifact(
    step: str,
    state: AgentState,
    *,
    rerank_enabled: bool,
    reranker_available: bool,
    rerank_applied: bool,
    final_top_k: int,
) -> dict[str, Any]:
    if step == "auto_parse":
        parsed_query = state.get("parsed_query") or {}
        return {
            "detected_language": state.get("detected_language"),
            "selected_devices": (parsed_query.get("selected_devices") or [])[:3],
            "selected_doc_types": (parsed_query.get("selected_doc_types") or [])[:3],
            "selected_equip_ids": (parsed_query.get("selected_equip_ids") or [])[:3],
        }
    if step == "translate":
        return {
            "query_en": state.get("query_en"),
            "query_ko": state.get("query_ko"),
        }
    if step == "route":
        return {"route": state.get("route")}
    if step == "mq":
        route = state.get("route", "general")
        key = f"{route}_mq_list"
        return {
            "route": route,
            "mq_count": len(state.get(key, [])),
        }
    if step == "st_gate":
        return {"st_gate": state.get("st_gate")}
    if step == "st_mq":
        return {"search_queries": list(state.get("search_queries", []))}
    if step == "retrieve":
        docs = list(state.get("docs", []))
        return {
            "doc_ids": [str(doc.doc_id) for doc in docs[:10]],
            "doc_count": len(docs),
            "rerank_enabled": rerank_enabled,
            "reranker_available": reranker_available,
            "query_count": len(state.get("search_queries", [])),
        }
    if step == "rerank":
        docs = list(state.get("docs", []))
        return {
            "rerank_applied": rerank_applied,
            "rerank_enabled": rerank_enabled,
            "reranker_available": reranker_available,
            "top_k": final_top_k,
            "doc_ids": [str(doc.doc_id) for doc in docs[:10]],
            "doc_count": len(docs),
        }
    return {}


def run_retrieval_pipeline(
    *,
    query: str,
    llm: BaseLLM,
    spec: PromptSpec,
    retriever: Retriever,
    reranker: Any = None,
    rerank_enabled: bool = True,
    retrieval_top_k: int = 20,
    final_top_k: int = 20,
    steps: Sequence[str] | None = None,
    deterministic: bool = False,
    auto_parse_enabled: bool | None = None,
    device_names: Sequence[str] | None = None,
    doc_type_names: Sequence[str] | None = None,
    equip_id_set: set[str] | None = None,
    state_overrides: Mapping[str, Any] | None = None,
    effective_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state: AgentState = {"query": query}
    mutable_state = cast(dict[str, Any], cast(object, state))
    if state_overrides:
        mutable_state.update(dict(state_overrides))

    if effective_config:
        policies = effective_config.get("policies")
        if isinstance(policies, Mapping) and "rerank_enabled" in policies:
            rerank_enabled = bool(policies["rerank_enabled"])

    execution_steps = _resolve_execution_steps(
        steps,
        deterministic=deterministic,
        auto_parse_enabled=auto_parse_enabled,
    )
    step_artifacts: dict[str, dict[str, Any]] = {}
    executed_steps: list[str] = []
    reranker_available = reranker is not None and hasattr(reranker, "rerank")
    rerank_applied = False

    for step in execution_steps:
        if deterministic and step in _MQ_STEPS:
            continue

        update: dict[str, Any]
        if step == "auto_parse":
            update = auto_parse_node(
                state,
                llm=llm,
                spec=spec,
                device_names=list(device_names or []),
                doc_type_names=list(doc_type_names or []),
                equip_id_set=equip_id_set,
            )
        elif step == "translate":
            update = translate_node(state, llm=llm, spec=spec)
        elif step == "route":
            update = route_node(state, llm=llm, spec=spec)
        elif step == "mq":
            update = mq_node(state, llm=llm, spec=spec)
        elif step == "st_gate":
            update = st_gate_node(state, llm=llm, spec=spec)
        elif step == "st_mq":
            update = st_mq_node(state, llm=llm, spec=spec)
        elif step == "retrieve":
            if deterministic:
                stable_query = _clean_query(state.get("query_en") or state.get("query"))
                mutable_state["search_queries"] = [stable_query] if stable_query else []

            update = retrieve_node(
                state,
                retriever=retriever,
                reranker=reranker if rerank_enabled else None,
                retrieval_top_k=retrieval_top_k,
                final_top_k=final_top_k,
            )
            docs = cast(list[Any], update.get("docs", []))
            rerank_applied = bool(rerank_enabled and reranker_available and docs)
        elif step == "rerank":
            update = {}
        else:
            continue

        mutable_state.update(update)
        executed_steps.append(step)
        step_artifacts[step] = {
            "status": "completed",
            "artifacts": _build_artifact(
                step,
                state,
                rerank_enabled=rerank_enabled,
                reranker_available=reranker_available,
                rerank_applied=rerank_applied,
                final_top_k=final_top_k,
            ),
        }

    return {
        "state": state,
        "steps": step_artifacts,
        "executed_steps": executed_steps,
        "deterministic": deterministic,
        "effective_config": dict(effective_config or {}),
    }


__all__ = [
    "CANONICAL_RETRIEVAL_STEPS",
    "run_retrieval_pipeline",
]
