# Agent Retrieval Stability Hardening

## TL;DR
> **Summary**: Make `/api/agent/run` stable-by-default by bypassing MQ on the first pass, adding MQ as a controlled fallback, enforcing programmatic `search_queries` guardrails, and persisting minimal per-turn debug metadata for fast diagnosis.
> **Deliverables**: (1) `mq_mode` (off|fallback|on) policy + stable graph branching, (2) query guardrails (numeric/unit injection + anchor retention), (3) agent response + conversation persistence of retrieval metadata, (4) regression tests + stability gates.
> **Effort**: Medium
> **Parallel**: YES - 3 waves
> **Critical Path**: Agent policy contract → Graph rewiring → Guardrails → Observability+persistence → Tests

## Context
### Original Request
- Read `docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md` and create a development plan to fix the issues described.

### Interview Summary
- Default `/api/agent/run` policy: `Stable + MQ fallback`.
- `/api/conversations` persistence: store a **minimal, non-indexed debug blob** per turn.

### Research Findings (repo-grounded)
- Integrated diagnosis: `docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md`
- Retrieval pipeline determinism exists and is MQ-bypassable via `deterministic=true`:
  - `backend/api/routers/retrieval.py`
  - `backend/services/retrieval_effective_config.py`
  - `backend/services/retrieval_pipeline.py`
- Agent graph currently hard-wires MQ before retrieve:
  - `backend/api/routers/agent.py`
  - `backend/services/agents/langgraph_rag_agent.py`
- MQ/refine query generation uses non-zero temperature:
  - `backend/llm_infrastructure/llm/langgraph_agent.py` (`TEMP_QUERY_GEN = 0.3`)
- Conversations persistence currently stores only `user_text`, `assistant_text`, `doc_refs`:
  - `backend/api/routers/conversations.py`
  - `backend/services/chat_history_service.py`
  - `backend/llm_infrastructure/elasticsearch/mappings.py#get_chat_turns_mapping()`
- Frontend saves turns via conversations API and can be extended to send retrieval metadata:
  - `frontend/src/features/chat/hooks/use-chat-session.ts`
  - `frontend/src/features/chat/types.ts`
  - `frontend/src/features/chat/api.ts`

### Metis Review (gaps addressed)
- Define executable rules for “MQ fallback” and attach reason codes.
- Keep determinism scope focused on retrieval/search_queries/doc_refs (answer-text determinism is out-of-scope by default).
- Persist debug metadata safely: redaction/truncation, size cap, ES mapping that avoids field explosion (`enabled:false`).

## Work Objectives
### Core Objective
- For common queries, identical requests to `/api/agent/run` produce stable `search_queries` and stable top-k `doc_refs`, while preventing MQ from injecting unsupported numeric/unit parameters.

### Deliverables
- D1. Agent policy contract: `mq_mode` (`off|fallback|on`) + defaults controlled by server settings.
- D2. LangGraph topology change: conditional branch `route → (stable_prepare → retrieve)` vs `route → mq → st_gate → st_mq → retrieve`.
- D3. Programmatic guardrails for `search_queries` (applied after MQ/refine).
- D4. Observability:
  - AgentResponse.metadata includes stable debug keys (route, st_gate, mq_mode, mq_reason, attempts, retry strategy, search_queries).
  - Conversations turn persistence includes a non-indexed `retrieval_meta` blob.
- D5. Verification:
  - New/updated pytest covering stable default, fallback triggers, guardrails, and conversation round-trip.
  - A small API-level stability check (repeat runs) with evidence artifacts.

### Definition of Done (verifiable)
- [x] Stable default (MQ fallback mode) produces identical `search_queries` and identical top-k `doc_refs` across 10 repeats.
  - Command: `pytest tests/api/test_agent_retrieval_stability_default.py -q`
  - Evidence: `.sisyphus/evidence/agent-stability/task-?.txt`
- [x] MQ is not invoked on attempt 0 under default policy; MQ is invoked only when fallback triggers.
  - Command: `pytest backend/tests/test_agent_graph_mq_bypass.py -q`
- [x] Guardrails reject numeric/unit injection when absent from user query and fall back to stable queries.
  - Command: `pytest backend/tests/test_search_queries_guardrails.py -q`
- [x] Conversations persistence round-trips `retrieval_meta` (redacted/truncated) without ES mapping explosion.
  - Command: `pytest tests/api/test_conversations_retrieval_meta_roundtrip.py -q`

### Must Have
- Stable-by-default agent retrieval: MQ is not part of the first-pass execution.
- MQ fallback is reason-coded and bounded (no infinite loops; strict attempt ceilings).
- Guardrails are programmatic (not prompt-only).
- Debug metadata is persisted in a safe, non-indexed form with size caps.

### Must NOT Have
- No new retrieval algorithms/baselines (no new hybrid tuning, no web search fallback).
- No claim of global determinism (only defined stability: search_queries + doc_refs under stable default).
- No persistence of full retrieved document bodies in conversation history.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (pytest) + a small API repeatability check.
- Evidence policy: write artifacts under `.sisyphus/evidence/agent-stability/`.

## Execution Strategy
### Parallel Execution Waves
Wave 1 (Policy + graph wiring)
Wave 2 (Guardrails + observability + persistence)
Wave 3 (Tests + QA gates + docs)

### Dependency Matrix (high level)
- Wave 1 blocks Wave 2/3.
- Wave 2 blocks Wave 3.

### Agent Dispatch Summary
- Wave 1: deep (graph + policy)
- Wave 2: unspecified-high (guardrails + persistence)
- Wave 3: deep/unspecified-high (tests + QA gates)

## TODOs
> Implementation + Test = ONE task.
> EVERY task includes QA scenarios.

- [x] 1. Add agent retrieval policy contract (`mq_mode`) + server default

  **What to do**:
  - Add a request-level field `mq_mode` to the agent request model:
    - Type: literal enum string: `"off" | "fallback" | "on"`
    - Default: `fallback` **resolved on server-side** (do not require FE changes to activate)
  - Add a server-side setting for the default policy:
    - New settings class: `AgentSettings` with `env_prefix="AGENT_"`
    - Field: `mq_mode_default: str = "fallback"` (validate it is one of off/fallback/on)
    - Export instance `agent_settings = AgentSettings()` from `backend/config/settings.py`
  - In `backend/api/routers/agent.py`, when building `LangGraphRAGAgent` state overrides, set:
    - `state_overrides["mq_mode"]` to request value if present else `agent_settings.mq_mode_default`
    - `state_overrides["mq_mode_default"]` for debugging (optional)
  - In `backend/api/routers/agent.py`, include `mq_mode` in `AgentResponse.metadata` so FE can log and persist it.

  **Must NOT do**:
  - Do not change the existing behavior for explicit opt-in `mq_mode="on"`.
  - Do not remove existing request fields (`search_queries`, `use_canonical_retrieval`, filters).

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: API contract + settings + agent state propagation
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2-9 | Blocked By: none

  **References**:
  - Agent request model: `backend/api/routers/agent.py` (`class AgentRequest`)
  - Settings patterns: `backend/config/settings.py` (`RAGSettings`, `SearchSettings`)
  - FE payload entrypoint: `frontend/src/features/chat/api.ts` (`sendChatMessage`)

  **Acceptance Criteria**:
  - [ ] `POST /api/agent/run` accepts `mq_mode` and echoes it back in `AgentResponse.metadata`.
  - [ ] When `mq_mode` is omitted, `AgentSettings.mq_mode_default` is applied.

  **QA Scenarios**:
  ```
  Scenario: mq_mode defaulting works
    Tool: Bash
    Steps: pytest tests/api/test_agent_mq_mode_defaulting.py -q
    Expected: tests pass; response metadata contains mq_mode=fallback by default
    Evidence: .sisyphus/evidence/agent-stability/task-1-mq-mode-defaulting.txt

  Scenario: Explicit mq_mode=on is preserved
    Tool: Bash
    Steps: pytest tests/api/test_agent_mq_mode_defaulting.py -q
    Expected: tests pass; mq_mode=on is echoed when requested
    Evidence: .sisyphus/evidence/agent-stability/task-1-mq-mode-on.txt
  ```

- [x] 2. Rewire agent LangGraph topology to bypass MQ on first pass

  **What to do**:
  - In `backend/services/agents/langgraph_rag_agent.py`, add a new node `prepare_retrieve` that:
    - Computes a stable query string: `stable_query = " ".join((state.get("query_en") or state.get("query") or "").split()).strip()`
    - Sets `state["search_queries"] = [stable_query]` if non-empty
    - Sets `state["skip_mq"] = True`
    - Sets `state["mq_used"] = False` and `state["mq_reason"] = None` (initialize)
  - Replace the unconditional edge `route -> mq` with a conditional branch:
    - If `state["mq_mode"] == "on"`: go to `mq`
    - Else (`off|fallback`) and `state.get("attempts", 0) == 0`: go to `prepare_retrieve`
    - Else (`fallback` with attempts>=2): go to `mq`
  - Wire `prepare_retrieve -> retrieve`.
  - Ensure this applies consistently in all three start flows:
    - Auto-parse flow: `... translate -> route -> (branch)`
    - Device-selection flow: `route -> device_selection -> (Command goto branch target)`
      - Decision-complete: update `device_selection_node` so its returned `Command(goto=...)` chooses:
        - `goto="prepare_retrieve"` when `mq_mode in {off,fallback}` and `attempts==0`
        - `goto="mq"` when `mq_mode==on` or (`mq_mode==fallback` and `attempts>=2`)
    - Default flow: `START -> route -> (branch)`
  - Update any `device_selection_node` goto targets if needed so it can land on `prepare_retrieve` when stable-by-default.

  **Must NOT do**:
  - Do not remove `mq`, `st_gate`, `st_mq` nodes; they remain for `mq_mode=on` and fallback.
  - Do not change the retrieval request schema of `/api/retrieval/run`.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: LangGraph topology + state semantics + regression risk
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 3-9 | Blocked By: 1

  **References**:
  - Current hard-wired edges: `backend/services/agents/langgraph_rag_agent.py` (`builder.add_edge("route", "mq")`)
  - Agent state keys: `backend/llm_infrastructure/llm/langgraph_agent.py` (`AgentState` usage)

  **Acceptance Criteria**:
  - [ ] With default `mq_mode=fallback`, the executed path from `route` goes to `prepare_retrieve` (not `mq`) on attempt 0.
  - [ ] With explicit `mq_mode=on`, the executed path remains `route -> mq -> st_gate -> st_mq -> retrieve`.

  **QA Scenarios**:
  ```
  Scenario: Stable-by-default bypasses MQ nodes
    Tool: Bash
    Steps: pytest backend/tests/test_agent_graph_mq_bypass.py -q
    Expected: tests pass; for mq_mode=fallback attempt=0, mq/st_gate/st_mq are not invoked
    Evidence: .sisyphus/evidence/agent-stability/task-2-mq-bypass.txt

  Scenario: mq_mode=on keeps MQ path
    Tool: Bash
    Steps: pytest backend/tests/test_agent_graph_mq_bypass.py -q
    Expected: tests pass; for mq_mode=on, mq/st_gate/st_mq are invoked
    Evidence: .sisyphus/evidence/agent-stability/task-2-mq-on-path.txt
  ```

- [x] 3. Add MQ fallback triggers + reason codes (no thrash)

  **What to do**:
  - Implement reason-coded fallback in stable mode (`mq_mode=fallback`):
    - Trigger `mq_reason="empty_retrieval"` when retrieve step returns 0 docs.
    - Trigger `mq_reason="unfaithful_after_deterministic_retries"` when judge remains unfaithful at/after attempt 2.
  - Update retry selection to respect `mq_mode`:
    - If `mq_mode="off"`: never transition to `retry_mq`.
    - If `mq_mode="fallback"`: allow `retry_mq` only when `attempts >= 2` OR when `mq_reason=="empty_retrieval"`.
  - Enforce a hard ceiling:
    - If `attempts >= max_attempts`: stop and return a graceful answer with `judge.faithful=false` and a clear hint.
  - Persist the following state fields into `AgentResponse.metadata`:
    - `mq_used` (bool), `mq_reason` (string|None), `attempts` (int), `retry_strategy` (string)

  **Must NOT do**:
  - Do not introduce new retry tiers beyond the existing 3-level structure.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: correctness in retry branching + termination guarantees
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 4-9 | Blocked By: 2

  **References**:
  - Retry selection function: `backend/llm_infrastructure/llm/langgraph_agent.py` (`should_retry`)
  - Retry edges: `backend/services/agents/langgraph_rag_agent.py` (edges from `judge`)
  - Diagnosis report retry tiers: `docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md`

  **Acceptance Criteria**:
  - [ ] Empty retrieval under `mq_mode=fallback` transitions to MQ path with `mq_reason=empty_retrieval`.
  - [ ] Under `mq_mode=off`, MQ path is never taken.
  - [ ] No infinite loops: agent run terminates within configured attempts.

  **QA Scenarios**:
  ```
  Scenario: Empty retrieval triggers MQ fallback with reason
    Tool: Bash
    Steps: pytest backend/tests/test_agent_mq_fallback_reasons.py -q
    Expected: tests pass; mq_used=true and mq_reason==empty_retrieval when docs are empty
    Evidence: .sisyphus/evidence/agent-stability/task-3-empty-retrieval-fallback.txt

  Scenario: mq_mode=off never uses MQ
    Tool: Bash
    Steps: pytest backend/tests/test_agent_mq_fallback_reasons.py -q
    Expected: tests pass; mq_used=false for all attempts
    Evidence: .sisyphus/evidence/agent-stability/task-3-mq-off-never.txt
  ```

- [x] 4. Implement `search_queries` guardrails (numeric/unit injection + anchor retention)

  **What to do**:
  - Add helper `validate_search_queries()` and apply it after MQ/refine generation:
    - Apply in `st_mq_node` and `refine_queries_node` in `backend/llm_infrastructure/llm/langgraph_agent.py`.
  - Guardrail rules (decision-complete):
    - **G0 First query**: output list must start with original query exactly (`state["query"]` trimmed). If missing, insert it at index 0.
    - **G1 Dedup**: de-duplicate preserving order (case-insensitive, whitespace-normalized).
    - **G2 Numeric/unit injection ban**: if user query contains no digit (`[0-9]`), then drop any generated query containing:
      - any digit `[0-9]`, OR
      - unit-like patterns: `(?i)\b(psi|bar|pa|kpa|mpa|nm|mm|cm|um|v|kv|a|ma|w|kw|rpm|%|degc|°c)\b`.
      If user query contains digits, allow only digit substrings that already appear in the user query.
    - **G3 Anchor retention**:
      - Build anchor tokens from user query: split on whitespace/punctuation; keep tokens that contain at least one of `[A-Za-z]` or Hangul, length>=2.
      - For each candidate query, compute recall = (#anchor tokens appearing as substrings, case-insensitive) / (len(anchor_tokens) or 1).
      - Drop candidate if recall < 0.6.
    - **G4 Bounds**: keep at most 5 queries total.
    - **G5 Fallback**: if after filtering there are 0 queries, set `search_queries=[stable_query]` where `stable_query` is `state.get("query_en") or state.get("query")` (trimmed).
  - Record guardrail diagnostics into state + metadata:
    - `guardrail_dropped_numeric`, `guardrail_dropped_anchor`, `guardrail_final_count`.

  **Must NOT do**:
  - Do not rely on prompt text alone to enforce these constraints.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: careful text processing + correctness tests
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 6-9 | Blocked By: 3

  **References**:
  - Temperature + MQ outputs: `backend/llm_infrastructure/llm/langgraph_agent.py` (`TEMP_QUERY_GEN`, `st_mq_node`)
  - Report examples of numeric hallucination: `docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md`

  **Acceptance Criteria**:
  - [ ] If user query contains no digits, any generated MQ query that contains digits/units is dropped.
  - [ ] Final `search_queries` always starts with the original query.
  - [ ] Guardrail counters are present in `AgentResponse.metadata`.

  **QA Scenarios**:
  ```
  Scenario: Numeric/unit injection is rejected
    Tool: Bash
    Steps: pytest backend/tests/test_search_queries_guardrails.py -q
    Expected: tests pass; guardrail drops injected numeric queries and falls back to stable query list
    Evidence: .sisyphus/evidence/agent-stability/task-4-guardrails-numeric.txt

  Scenario: Anchor retention filter rejects drift
    Tool: Bash
    Steps: pytest backend/tests/test_search_queries_guardrails.py -q
    Expected: tests pass; low-anchor queries are dropped
    Evidence: .sisyphus/evidence/agent-stability/task-4-guardrails-anchor.txt
  ```

- [x] 5. Make query-generation temperature deterministic in fallback mode

  **What to do**:
  - Introduce a single helper to resolve query-generation temperature from state:
    - Input: `mq_mode` and `attempts`
    - Output: `temperature` used by MQ/refine LLM calls
  - Decision-complete temperature policy:
    - If `mq_mode in {"off", "fallback"}` and `attempts < 2`: use `temperature = 0.0` for any refine/MQ generation calls.
    - If `mq_mode == "on"`: keep `temperature = TEMP_QUERY_GEN` (0.3).
    - If `mq_mode == "fallback"` and MQ is actually invoked (due to fallback): use `temperature = 0.0` (stability-first).
  - Apply this helper in:
    - `mq_node`, `st_mq_node`, and `refine_queries_node` invocation sites in `backend/llm_infrastructure/llm/langgraph_agent.py`.

  **Must NOT do**:
  - Do not change `TEMP_CLASSIFICATION` or judge/route temperatures.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: cross-cutting change affecting multiple nodes
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 6-9 | Blocked By: 3

  **References**:
  - Existing temperature constants: `backend/llm_infrastructure/llm/langgraph_agent.py` (TEMP_* constants)

  **Acceptance Criteria**:
  - [ ] In `mq_mode=fallback`, any MQ/refine LLM calls use temperature 0.0.

  **QA Scenarios**:
  ```
  Scenario: Temperature policy applied in fallback mode
    Tool: Bash
    Steps: pytest backend/tests/test_agent_querygen_temperature_policy.py -q
    Expected: tests pass; captured calls show temperature=0.0 for mq/refine when mq_mode=fallback
    Evidence: .sisyphus/evidence/agent-stability/task-5-temperature-policy.txt
  ```

- [x] 6. Add stable observability fields to AgentResponse + stream payloads

  **What to do**:
  - In `backend/api/routers/agent.py`, ensure the response always includes:
    - `AgentResponse.search_queries` (top-level) = final `state.search_queries`
    - `AgentResponse.metadata` keys (decision-complete list):
      - `mq_mode`, `mq_used`, `mq_reason`
      - `route`, `st_gate`
      - `attempts`, `max_attempts`, `retry_strategy`
      - `guardrail_dropped_numeric`, `guardrail_dropped_anchor`, `guardrail_final_count`
      - `search_queries_raw` (optional; redacted/truncated) and `search_queries_final`
      - `index_name` or alias info when available (safe string)
  - For `/api/agent/run/stream` (SSE), include the same metadata in the final event payload, and do not break existing client parsing.

  **Must NOT do**:
  - Do not include full document bodies in metadata.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: API response contract stability + SSE compatibility
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7-9 | Blocked By: 4-5

  **References**:
  - Response models: `backend/api/routers/agent.py` (`class AgentResponse`)
  - FE reads metadata.search_queries: `frontend/src/features/chat/hooks/use-chat-session.ts`

  **Acceptance Criteria**:
  - [ ] Both run and stream responses contain `mq_mode` and final `search_queries`.

  **QA Scenarios**:
  ```
  Scenario: AgentResponse contains required metadata keys
    Tool: Bash
    Steps: pytest tests/api/test_agent_response_metadata_contract.py -q
    Expected: tests pass; keys exist and are correct types
    Evidence: .sisyphus/evidence/agent-stability/task-6-response-metadata.txt
  ```

- [x] 7. Persist minimal retrieval debug blob in conversations (non-indexed, safe)

  **What to do**:
  - Extend conversations API to accept an optional `retrieval_meta` payload:
    - FE request type: extend `SaveTurnRequest` in `frontend/src/features/chat/types.ts`.
    - BE request model: extend `TurnRequest` in `backend/api/routers/conversations.py`.
    - Storage: add `retrieval_meta: dict[str, object] | None` to `backend/services/chat_history_service.py` (`ChatTurn`).
  - Return `retrieval_meta` back to clients:
    - Extend `TurnResponse` in `backend/api/routers/conversations.py` to include `retrieval_meta` (optional).
    - Extend `TurnResponse` in `frontend/src/features/chat/types.ts` to include `retrieval_meta?` (optional).
  - Redaction + size cap (decision-complete):
    - Truncate any string field to 256 chars.
    - For `search_queries*`: store at most 5 entries; each entry max 120 chars.
    - Drop/replace obvious PII patterns (email, phone) with `[REDACTED]`.
    - Enforce serialized JSON size <= 8 KB; if over, store `{ "truncated": true }` plus a small subset of keys.
  - Elasticsearch mapping update:
    - Add a top-level property `retrieval_meta` with type `object` and `enabled: false` in `backend/llm_infrastructure/elasticsearch/mappings.py#get_chat_turns_mapping()`.
    - Update `ChatHistoryService.ensure_index()` so that even when alias exists, it calls `indices.put_mapping` to add `retrieval_meta` if missing.
      - Decision-complete: do not reindex; only add a new field.

  **Must NOT do**:
  - Do not index individual debug keys (avoid mapping explosion).
  - Do not persist raw retrieved document contents.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ES mapping + backward compatible persistence
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 8-9 | Blocked By: 6

  **References**:
  - Conversations router: `backend/api/routers/conversations.py`
  - Chat turn dataclass: `backend/services/chat_history_service.py`
  - ES mapping: `backend/llm_infrastructure/elasticsearch/mappings.py` (`get_chat_turns_mapping`)
  - FE saveTurn call: `frontend/src/features/chat/hooks/use-chat-session.ts` (saveTurn payload)

  **Acceptance Criteria**:
  - [ ] `POST /api/conversations/{session_id}/turns` accepts `retrieval_meta`.
  - [ ] `GET /api/conversations/{session_id}` returns the stored `retrieval_meta` per turn.
  - [ ] ES mapping does not explode (field is non-indexed).

  **QA Scenarios**:
  ```
  Scenario: retrieval_meta round-trips through conversations
    Tool: Bash
    Steps: pytest tests/api/test_conversations_retrieval_meta_roundtrip.py -q
    Expected: tests pass; stored retrieval_meta matches (after redaction/truncation)
    Evidence: .sisyphus/evidence/agent-stability/task-7-conversations-roundtrip.txt
  ```

- [x] 8. Extend frontend to persist retrieval_meta with each saved turn

  **What to do**:
  - In `frontend/src/features/chat/hooks/use-chat-session.ts`, when calling `saveTurn(...)`, include `retrieval_meta` derived from the latest AgentResponse:
    - `mq_mode`, `mq_used`, `mq_reason`, `route`, `st_gate`, `attempts`, `retry_strategy`, `search_queries` (effectiveSearchQueries)
  - Update FE types:
    - `frontend/src/features/chat/types.ts` (`SaveTurnRequest`) add optional `retrieval_meta?: Record<string, unknown>`.
  - Add/adjust FE unit test(s) ensuring payload includes retrieval_meta when available:
    - `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx`

  **Must NOT do**:
  - Do not block chat flow if saving retrieval_meta fails; saving turn must still succeed without it.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: localized FE payload + type change
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 9 | Blocked By: 7

  **References**:
  - FE saveTurn: `frontend/src/features/chat/hooks/use-chat-session.ts`
  - SaveTurn types: `frontend/src/features/chat/types.ts`
  - Conversations API: `frontend/src/features/chat/api.ts`

  **Acceptance Criteria**:
  - [ ] FE includes retrieval_meta in turn save payload when agent response contains metadata.

  **QA Scenarios**:
  ```
  Scenario: FE saveTurn payload includes retrieval_meta
    Tool: Bash
    Steps: pnpm -C frontend test
    Expected: test passes; request payload includes retrieval_meta keys
    Evidence: .sisyphus/evidence/agent-stability/task-8-fe-payload.txt
  ```

- [x] 9. Add regression tests + a stability gate for `/api/agent/run`

  **What to do**:
  - Add a new API test for stable default repeatability:
    - Run the same `/api/agent/run` request 10 times (same session/thread) with `mq_mode` omitted.
    - Assert:
      - `response.metadata.mq_mode == "fallback"`
      - `response.search_queries` identical across runs
      - `response.retrieved_docs[*].id` identical across runs (ordered list)
  - Add a forced-empty retrieval test:
    - Stub retriever/search_service to return empty docs on attempt 0.
    - Assert MQ is invoked in fallback mode and `mq_reason == "empty_retrieval"`.
  - Ensure tests do not require real ES/LLM:
    - Use dependency overrides + monkeypatch for nodes or a fake LLM.

  **Must NOT do**:
  - Do not rely on network services (real ES, real vLLM) for unit tests.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: integration-level tests with careful stubbing
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: none | Blocked By: 2-8

  **References**:
  - Existing API tests patterns: `tests/api/test_agent_top_k_autoparse.py`, `tests/api/test_retrieval_run_stability_and_golden.py`
  - Agent endpoint: `backend/api/routers/agent.py`

  **Acceptance Criteria**:
  - [ ] New tests pass locally and fail on known regressions (MQ invoked on attempt 0, numeric injection not filtered, instability across repeats).

  **QA Scenarios**:
  ```
  Scenario: Stable default is repeatable across 10 runs
    Tool: Bash
    Steps: pytest tests/api/test_agent_retrieval_stability_default.py -q
    Expected: tests pass; doc_ids/search_queries identical
    Evidence: .sisyphus/evidence/agent-stability/task-9-stable-default.txt

  Scenario: Empty retrieval triggers MQ fallback
    Tool: Bash
    Steps: pytest tests/api/test_agent_retrieval_stability_default.py -q
    Expected: tests pass; mq_used=true with reason
    Evidence: .sisyphus/evidence/agent-stability/task-9-empty-fallback.txt
  ```

- [x] 10. Update diagnosis + API documentation to reflect new stability policy

  **What to do**:
  - Update `docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md`:
    - Add a short “After Fix” section documenting:
      - `mq_mode` contract and default (`fallback`)
      - What is guaranteed stable (search_queries + doc_refs) and what is not (answer text)
      - How to opt-in MQ (`mq_mode=on`) and how to force stable (`mq_mode=off`)
  - Update any developer-facing API docs if present (or add a short section to the same doc):
    - `/api/agent/run` request/response fields for `mq_mode`, `metadata` keys, and `retrieval_meta` persistence.

  **Must NOT do**:
  - Do not rewrite the entire report; only add the minimal delta needed to operate the system.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: concise doc update with correct contracts
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 1-9

  **References**:
  - Report: `docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md`
  - Agent request model: `backend/api/routers/agent.py` (`AgentRequest`)
  - Conversations turn model: `backend/api/routers/conversations.py` (`TurnRequest`)

  **Acceptance Criteria**:
  - [ ] Documentation accurately reflects new defaults and how to reproduce stable vs MQ behavior.

  **QA Scenarios**:
  ```
  Scenario: Docs mention mq_mode + guarantees
    Tool: Bash
    Steps: grep -n "mq_mode" -n docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md
    Expected: contains contract + default + guarantees section
    Evidence: .sisyphus/evidence/agent-stability/task-10-docs-grep.txt
  ```

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Reproducibility Run-Through — deep
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: Agent request policy + graph branching
- Commit 2: Guardrails + metadata + conversations persistence
- Commit 3: Tests + docs

## Success Criteria
- Production default agent path is stable-by-default and diagnosable: repeated queries return stable doc_refs unless MQ fallback triggers, and guardrails prevent ungrounded numeric/unit injection.
