# Issues

- Verification conflict: `lsp_diagnostics` on changed backend files reports many pre-existing `reportMissingImports` and TypedDict/type-check warnings across the module baseline (not introduced by this task). Pytest target for Task 3 passes, but workspace LSP is not clean due environment/type-check baseline noise.
- Task 4 verification note: `lsp_diagnostics` still reports baseline unresolved-import/type-noise on `backend/llm_infrastructure/llm/langgraph_agent.py` and test module import resolution in this environment; functional verification was completed via `pytest backend/tests/test_search_queries_guardrails.py -q` (passed).
- Task 5 blocker note: no new functional blockers; only the same baseline LSP noise persists, so verification used targeted pytest (`backend/tests/test_agent_querygen_temperature_policy.py`).
- Task 6 verification note: `lsp_diagnostics` remains noisy on changed files (missing-import/environment + strict basedpyright Any/deprecation warnings), so contract verification used targeted API pytest (`pytest tests/api/test_agent_response_metadata_contract.py -q`) which passed.

- Task 7 verification note: `lsp_diagnostics` still reports baseline environment import-resolution errors (`reportMissingImports`) in backend modules (not introduced by this task); functional verification completed via `pytest tests/api/test_conversations_retrieval_meta_roundtrip.py -q` (passed).


- Task 8 verification note: No blockers. All frontend tests pass (`npm test -- --run` in frontend directory - 53 tests including 2 new retrieval_meta payload tests). LSP diagnostics clean on modified files (`use-chat-session.ts`, `chat-request-payload.test.tsx`).
#KT|
#KT|- Scope creep removed: Task 8 subagent accidentally created paper/strategy docs (`docs/paper/ie_phd_thesis_strategy_roadmap.md`, `docs/paper/pe_agent_논문가능성.md`) which are out of scope for agent-retrieval stability plan. Deleted both files.

- Task 9 note: No functional blockers. Targeted verification passed with `pytest tests/api/test_agent_retrieval_stability_default.py -q` (2 passed); only unrelated FastAPI/Pydantic deprecation warnings surfaced.


## F1 Plan Compliance Audit (2026-02-28 17:27 UTC)
- PASS (deliverables + evidence present), with mismatches to fix before merge.
- Mismatch: DoD item for MQ-bypass references `pytest backend/tests/test_agent_mq_bypass_and_fallback.py -q` in `.sisyphus/plans/agent-retrieval-stability-hardening.md`, but repo uses `backend/tests/test_agent_graph_mq_bypass.py` (evidence: `.sisyphus/evidence/agent-stability/task-2-mq-bypass.txt`).
- Scope noise: unrelated diffs currently present in `.sisyphus/plans/paper-b-stability.md` and `.sisyphus/boulder.json`; exclude/revert or commit separately so PR aligns with the agent-retrieval plan.
- Evidence artifacts exist under `.sisyphus/evidence/agent-stability/` for tasks 1-10 + DoD (e.g. `task-9-stable-default.txt`, `task-6-response-metadata.txt`, `task-7-conversations-roundtrip.txt`).

## F4 Scope Fidelity Check (2026-02-28 17:44 UTC)
- Core implementation changes are in-scope for plan tasks D1-D5: policy/defaulting (`mq_mode`), graph bypass/fallback wiring, guardrails, response metadata + conversations `retrieval_meta`, and regression/API tests.
- Must-NOT check passed on inspected diffs: no hunk persists full retrieved document bodies into conversation history; frontend sends only compact retrieval metadata keys and search query list.
- Out-of-scope modified files detected: `.sisyphus/boulder.json` (active plan/session metadata) and `.sisyphus/plans/paper-b-stability.md` (unrelated DoD checkbox updates). Keep these out of the delivery commit.
- Incidental non-deliverable churn detected: typing/format-only hunks (for example `meta: dict[str, Any]` in `backend/llm_infrastructure/elasticsearch/mappings.py`) that are not required for D1-D5 behavior.

- F3 deep run-through (2026-02-28): executed the required 8 pytest commands in strict order and all passed (2, 2, 3, 2, 8, 1, 1, 2 tests). No flakiness, no ordering issues, and no hidden real ES/vLLM dependency observed; only known FastAPI/Pydantic deprecation warnings appeared.

## F2 Code Quality Review (2026-02-28 18:05 UTC)
- MEDIUM: `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx` uses `mq_mode: "semantic"` in mocked metadata payload and assertion, but backend contract is strict `Literal["off", "fallback", "on"]` (`backend/api/routers/agent.py`, `backend/config/settings.py`). This test codifies an invalid enum value and can mask FE/BE metadata contract drift.
- Minimal fix: replace mocked/expected `mq_mode` in that test with a valid contract value (recommend `"fallback"` for plan-default parity) and keep other retrieval_meta assertions unchanged.
