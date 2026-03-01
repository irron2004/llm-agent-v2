# F3. Reproducibility Run-Through (Deep)

## Environment precheck
1. `git status --short`
   - Expected: existing workspace deltas may exist; no requirement for a clean tree for this gate.
2. `git diff --stat`
   - Expected: summary of current local modifications; used for context only.

## Clean linear test sequence (authoritative order)
Run each command exactly in this order:

1. `pytest tests/api/test_agent_mq_mode_defaulting.py -q`
   - Expected high-level output: `2 passed` (warnings allowed).
2. `pytest backend/tests/test_agent_graph_mq_bypass.py -q`
   - Expected high-level output: `2 passed`.
3. `pytest backend/tests/test_agent_mq_fallback_reasons.py -q`
   - Expected high-level output: `3 passed`.
4. `pytest backend/tests/test_search_queries_guardrails.py -q`
   - Expected high-level output: `2 passed`.
5. `pytest backend/tests/test_agent_querygen_temperature_policy.py -q`
   - Expected high-level output: `8 passed`.
6. `pytest tests/api/test_agent_response_metadata_contract.py -q`
   - Expected high-level output: `1 passed` (warnings allowed).
7. `pytest tests/api/test_conversations_retrieval_meta_roundtrip.py -q`
   - Expected high-level output: `1 passed` (warnings allowed).
8. `pytest tests/api/test_agent_retrieval_stability_default.py -q`
   - Expected high-level output: `2 passed` (includes `test_agent_run_default_fallback_is_stable_across_10_repeats`).

## Stability claim criteria
- Deterministic claim is accepted when step 8 passes and no intermittent failures appear across the sequence.
- No real external ES/vLLM calls should occur during the run.

## Observed run result in this environment
- Sequence status: PASS (all 8 commands passed in order).
- Flakiness: none observed.
- Ordering issues: none observed.
- Hidden network dependency: none observed in this run.
- Known non-blocking noise: FastAPI/Pydantic deprecation warnings.
