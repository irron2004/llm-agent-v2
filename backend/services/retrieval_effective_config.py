from __future__ import annotations

import hashlib
import json

from ..config.settings import rag_settings

PIPELINE_VERSION = "retrieval-run-v1"
CANONICAL_STEPS = (
    "auto_parse",
    "translate",
    "route",
    "mq",
    "st_gate",
    "st_mq",
    "retrieve",
    "rerank",
)


def resolve_effective_config(
    query: str,
    requested_steps: list[str] | None,
    debug: bool,
    deterministic: bool,
    *,
    final_top_k: int | None = None,
    retrieval_top_k: int | None = None,
    rerank_enabled: bool | None = None,
    auto_parse: bool | None = None,
    skip_mq: bool | None = None,
    reranker_available: bool = True,
) -> dict[str, object]:
    requested = [step for step in (requested_steps or []) if step in CANONICAL_STEPS]

    executed_steps: list[str] = []
    if debug:
        executed_steps = list(CANONICAL_STEPS)
    elif requested_steps is not None:
        executed_steps = requested

    requested_final_top_k = (
        int(final_top_k) if isinstance(final_top_k, int) and final_top_k > 0 else None
    )
    applied_final_top_k = requested_final_top_k or int(rag_settings.rerank_top_k)

    requested_retrieval_top_k = (
        int(retrieval_top_k) if isinstance(retrieval_top_k, int) and retrieval_top_k > 0 else None
    )
    applied_retrieval_top_k = requested_retrieval_top_k or int(rag_settings.retrieval_top_k)

    requested_rerank_enabled = rerank_enabled
    applied_rerank_enabled = (
        bool(rag_settings.rerank_enabled)
        if requested_rerank_enabled is None
        else bool(requested_rerank_enabled)
    )
    if applied_rerank_enabled and not reranker_available:
        applied_rerank_enabled = False
    rerank_policy_forced = requested_rerank_enabled is True and not reranker_available

    requested_auto_parse = auto_parse
    applied_auto_parse = bool(requested_auto_parse) if requested_auto_parse is not None else False

    requested_skip_mq = skip_mq
    applied_skip_mq = bool(requested_skip_mq) if requested_skip_mq is not None else False

    def _resolve_source(
        requested_value: object,
        *,
        policy_forced: bool = False,
    ) -> str:
        if policy_forced:
            return "policy"
        if requested_value is not None:
            return "request"
        return "env_default"

    return {
        "version": PIPELINE_VERSION,
        "query": query,
        "debug": debug,
        "deterministic": deterministic,
        "requested_steps": requested,
        "executed_steps": executed_steps,
        "policies": {
            "mq_strategy": "bypass" if deterministic or applied_skip_mq else "llm",
            "rerank_enabled": applied_rerank_enabled,
            "multi_query_enabled": rag_settings.multi_query_enabled,
            "auto_parse_enabled": applied_auto_parse,
            "skip_mq": applied_skip_mq,
        },
        "defaults": {
            "retrieval_top_k": applied_retrieval_top_k,
            "rerank_top_k": rag_settings.rerank_top_k,
            "final_top_k": applied_final_top_k,
            "hybrid_dense_weight": rag_settings.hybrid_dense_weight,
            "hybrid_sparse_weight": rag_settings.hybrid_sparse_weight,
            "hybrid_rrf_k": rag_settings.hybrid_rrf_k,
        },
        "request_vs_applied": {
            "retrieval_top_k": {
                "requested": requested_retrieval_top_k,
                "applied": applied_retrieval_top_k,
                "source": _resolve_source(requested_retrieval_top_k),
            },
            "final_top_k": {
                "requested": requested_final_top_k,
                "applied": applied_final_top_k,
                "source": _resolve_source(requested_final_top_k),
            },
            "rerank_enabled": {
                "requested": requested_rerank_enabled,
                "applied": applied_rerank_enabled,
                "source": _resolve_source(
                    requested_rerank_enabled,
                    policy_forced=rerank_policy_forced,
                ),
            },
            "auto_parse": {
                "requested": requested_auto_parse,
                "applied": applied_auto_parse,
                "source": _resolve_source(requested_auto_parse),
            },
            "skip_mq": {
                "requested": requested_skip_mq,
                "applied": applied_skip_mq,
                "source": _resolve_source(requested_skip_mq),
            },
        },
    }


def effective_config_hash(effective_config: dict[str, object]) -> str:
    canonical = json.dumps(
        effective_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "PIPELINE_VERSION",
    "CANONICAL_STEPS",
    "resolve_effective_config",
    "effective_config_hash",
]
