# Learnings


## Findings (auto-generated)
- Located TurnRequest/TurnResponse models in backend/api/routers/conversations.py
- ChatTurn dataclass is defined in backend/services/chat_history_service.py
- Elasticsearch mappings for chat turns defined in backend/llm_infrastructure/elasticsearch/mappings.py (CHAT_TURNS_MAPPING)
- Frontend chat types defined in frontend/src/features/chat/types.ts (SaveTurnRequest, TurnResponse)
- Mapping change: add retrieval_meta field stored with enabled:false in chat turns mapping
- Migration note: if alias exists, create a new versioned index, reindex, and switch alias to safely add new field

Proposed changes (high level):
- Extends TurnRequest/TurnResponse to include retrieval_meta: Optional[Dict[str, Any]]
- Extend ChatTurn to include retrieval_meta; update to_dict/from_dict to serialize/deserialize
- Update CHAT_TURNS_MAPPING to include retrieval_meta with enabled=false
- Extend frontend SaveTurnRequest with retrieval_meta?: Record<string, unknown>

- mq_mode effective value is computed in `run_agent`/`run_agent_stream` as `req.mq_mode or agent_settings.mq_mode_default`, injected into both `state_overrides` and auto-parse `chat_state` (`chat_state_stream`) so every run path carries `state["mq_mode"]`, and echoed in API response via `_build_response_metadata(..., mq_mode=effective_mq_mode)`.
- Added `prepare_retrieve` graph node in `backend/services/agents/langgraph_rag_agent.py` to build a stable single query (`query_en` fallback to `query`), set `skip_mq=True`, initialize `mq_used=False` and `mq_reason=None`, and branch `route` to `prepare_retrieve` on attempt 0 when `mq_mode` is `off`/`fallback` (otherwise route to `mq`).
- Task 3: Added reason-coded MQ fallback logic in `backend/llm_infrastructure/llm/langgraph_agent.py`: `retrieve_node` now tags `mq_reason="empty_retrieval"` when fallback mode gets 0 docs on stable (non-MQ) retrieval; `retry_mq_node` now sets `mq_reason` to either `empty_retrieval` or `unfaithful_after_deterministic_retries`, and marks `mq_used=True`.
- `should_retry` now enforces mode-aware retry routing: `mq_mode="off"` never returns `retry_mq`; `mq_mode="fallback"` allows MQ only for `attempts>=2` or `mq_reason=="empty_retrieval"`; and hard ceiling (`attempts>=max_attempts`) returns `done` to terminate loops.
- `judge_node` now overwrites unfaithful terminal outcomes at ceiling with a clear hint (`Reached max_attempts...`) and `issues += ["max_attempts_reached"]`, which gives graceful bounded termination metadata.
- Updated MQ-entry behavior to guarantee off-mode never enters MQ start branch: `backend/services/agents/langgraph_rag_agent.py` route branch and `device_selection_node` now send `mq_mode="off"` to `prepare_retrieve` always.
- `backend/api/routers/agent.py` metadata now includes `mq_used`, `mq_reason`, `attempts`, and `retry_strategy` so fallback diagnostics are surfaced in both run and stream final payloads.
- Added `backend/tests/test_agent_mq_fallback_reasons.py` with fake LLM/search + monkeypatched nodes to verify: empty retrieval fallback reason, `mq_mode=off` MQ suppression + hard ceiling, and unfaithful-after-retries reason at attempt>=2.
- Updated `backend/tests/test_agent_graph_mq_bypass.py` fallback bypass expectation: attempt-0 still bypasses MQ nodes, but `mq_reason` is now `"empty_retrieval"` when stable retrieval returns zero docs (Task-3 reason-coded fallback trigger).
- Task 4 guardrails: added `validate_search_queries()` in `backend/llm_infrastructure/llm/langgraph_agent.py` with programmatic G0-G5 enforcement (original-first insertion, case/whitespace dedup, numeric/unit injection filtering, anchor recall threshold at 0.6, max-5 cap, stable fallback via `query_en`/`query`).
- Guardrail diagnostics (`guardrail_dropped_numeric`, `guardrail_dropped_anchor`, `guardrail_final_count`) are now written by `st_mq_node` and `refine_queries_node` so metadata can surface deterministic drop reasons.
- `st_mq_node` applies guardrails to generated MQ/refine outputs while preserving skip-MQ passthrough behavior to avoid changing stable-path precomputed query lists.
- Added `backend/tests/test_search_queries_guardrails.py` coverage for numeric/unit injection rejection and anchor-drift filtering scenarios.
- Task 5: query-generation temperature control is centralized in `resolve_querygen_temperature()` (`backend/llm_infrastructure/llm/langgraph_agent.py`) and applied only at MQ/refine generation call sites (`mq_node`, `st_mq_node`, `refine_queries_node`), leaving route/judge/translation temperatures unchanged.
- Task 6: `backend/api/routers/agent.py` now normalizes final `search_queries` once per run and reuses that list for both top-level `AgentResponse.search_queries` and `metadata.search_queries_final` (plus legacy `metadata.search_queries`) to keep run/stream payloads contract-stable.
- Task 6: metadata builder now emits stable observability keys with defaults (`mq_*`, `route`, `st_gate`, `attempts`, `max_attempts`, `retry_strategy`, guardrail counters) and optional safe index hint (`index_name`) sourced from `search_service.es_engine.index_name`.
- Task 6: optional `metadata.search_queries_raw` is populated only when raw MQ candidates exist, with minimal redaction/truncation applied in-router (drop empties, dedupe, max 5 entries, max 120 chars each).
- Added `tests/api/test_agent_response_metadata_contract.py` to validate required metadata keys/types and parity between run endpoint and stream final SSE payload.

- Task 7: Added per-turn `retrieval_meta` roundtrip support in conversations API and chat storage (`TurnRequest`/`TurnResponse` + `ChatTurn`).
- Task 7: Implemented deterministic `sanitize_retrieval_meta` with PII redaction (email/phone), string truncation (256), `search_queries*` cap (5 items, 120 chars each), and 8KB payload cap fallback to `{\"truncated\": true}` + safe subset.
- Task 7: ES chat-turn mapping now stores `retrieval_meta` as `object` with `enabled: false`; `ensure_index()` now applies `indices.put_mapping` on existing alias targets when the field is missing (no reindex).


- Task 8: Frontend now attaches `retrieval_meta` to each saved turn in `useChatSession` hook (`handleAgentResponse` callback). The hook extracts `mq_mode`, `mq_used`, `mq_reason`, `route`, `st_gate`, `attempts`, `retry_strategy` from the agent response's `metadata` field, plus `effectiveSearchQueries` (the final deduplicated search query list) for `search_queries`. Payload is optional—if metadata is missing, the field is omitted entirely to avoid blocking chat flow.

- Task 9: Added `tests/api/test_agent_retrieval_stability_default.py` with deterministic dependency overrides (`get_search_service`, `get_default_llm`, `get_prompt_spec_cached`) to run `/api/agent/run` without real ES/LLM.
- Task 9: Stability gate test posts the same payload 10 times with fixed `thread_id` and omitted `mq_mode`, asserting `metadata.mq_mode == "fallback"`, identical `search_queries`, and ordered `retrieved_docs[*].id` parity across runs.
- Task 9: Forced-empty regression test uses scripted retrieval (`"pump alarm reset" -> []`, `"pump alarm reset procedure" -> docs`) to verify fallback MQ invocation with `metadata.mq_used is True` and `metadata.mq_reason == "empty_retrieval"`.

- Task 10: Updated diagnosis report (docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md) with new Section 13 documenting mq_mode contract, stability guarantees, and response metadata keys.

## F2 Review Learnings (2026-02-28 18:05 UTC)
- Keep FE mocked metadata values contract-valid: using out-of-contract enums in tests (for example `mq_mode: "semantic"`) weakens run/stream parity guarantees and can hide schema drift.
- For this plan, highest-value quality checks remain metadata contract parity and sanitized persistence behavior; current backend implementation appears aligned, and the only concrete mismatch observed was test-fixture enum validity.

- Verified `chat-request-payload.test.tsx` already has valid `mq_mode: "fallback"` (not "semantic"); test passes. No changes needed - previous fix already applied.
- Section 13 (After Fix) keys corrected: removed `guardrail_blocked_count`, added `search_queries` legacy row, updated retrieval_meta description to match FE contract.