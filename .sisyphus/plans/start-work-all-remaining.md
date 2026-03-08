# Unified Start-Work Plan (Manual Reconciliation)

## Intent
- This file is the only execution plan for /start-work.
- `.sisyphus/plans/.legacy/` is source spec storage, not a direct execution target.
- Every task below is reconciled against the current repository layout before merge.

## Reconciliation Method (One-by-One)
1. Read each legacy plan in order.
2. Compare pending TODOs with current files/tests/scripts.
3. Replace stale references with current equivalents.
4. Exclude completed scopes from active execution queue.
5. Run only the integrated backlog in this file.

## Process Safety Rules (Zombie Prevention)
- Forbidden in plan execution: &, nohup, disown, orphaned background loops.
- Long-running commands must use bounded execution (timeout with explicit seconds).
- If any worker process is spawned, cleanup is mandatory before moving to next task.
- Evidence paths are explicit per phase to avoid ambiguous reruns.

## Legacy-by-Legacy Consistency Review

### 1) `.sisyphus/plans/.legacy/ui-autoparse-confirm-task-mode-2026-03-05.md`
- Status: completed.
- Decision: keep as historical evidence only; no active backlog.

### 2) `.sisyphus/plans/.legacy/ui-chat-improvements-v2.md`
- Status: partially complete.
- Remaining scope: REQ-6 (tabbed guided-selection UI) + final audit gates.
- File consistency: `frontend/src/features/chat/components/guided-selection-panel.tsx`, `frontend/src/features/chat/hooks/use-chat-session.ts`, `frontend/src/features/chat/pages/chat-page.tsx`, `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`.

### 3) `.sisyphus/plans/.legacy/chunk_v3_embed_ingest_plan.md`
- Status: large active backlog.
- Scope alignment: keep chunking/embedding/ingest/eval tracks, remove stale symbol-style references.
- File consistency: `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/run_embedding.py`, `scripts/chunk_v3/run_ingest.py`, `scripts/chunk_v3/validate_vlm.py`, `backend/llm_infrastructure/elasticsearch/mappings.py`, `backend/domain/doc_type_mapping.py`, `scripts/evaluation/evaluate_sop_agent_page_hit.py`.

### 4) `.sisyphus/plans/.legacy/chapter-grouping-retrieval.md`
- Status: phase-2 style follow-up remains.
- Scope alignment: keep reindex + retrieval integration tasks only.
- File consistency: `scripts/chunk_v3/section_extractor.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/services/search_service.py`.

### 5) `.sisyphus/plans/.legacy/agent-retrieval-followups-2026-03-04.md`
- Status: active backlog.
- Mismatch corrected:
  - Legacy ref tests/api/test_agent_rrf_and_sticky_gates.py does not exist.
  - Current verification set: `tests/api/test_agent_stage2_retrieval.py`, `tests/api/test_agent_sticky_policy_followup_only.py`, `tests/api/test_agent_interrupt_resume_regression.py`, `backend/tests/test_sop_intent_heuristic.py`, `backend/tests/test_retrieve_node_sop_soft_boost.py`.

### 6) `.sisyphus/plans/.legacy/before-after-regression-compare.md`
- Status: active backlog.
- Scope alignment: keep before/after reproducible run + report generation.
- File consistency: `scripts/evaluation/regression_compare_manifest.py`, `scripts/evaluation/evaluate_sop_agent_page_hit.py`, `scripts/evaluation/agent_eval_report.py`, `tests/evaluation/test_regression_compare_manifest.py`.

### 7) `.sisyphus/plans/.legacy/paper-a-scope-implementation.md`
- Status: active backlog.
- Scope alignment: policy/evaluator/evidence pipeline remains valid.
- File consistency: `scripts/paper_a/build_corpus_meta.py`, `scripts/paper_a/build_shared_and_scope.py`, `scripts/paper_a/build_family_map.py`, `scripts/paper_a/build_eval_sets.py`, `scripts/paper_a/evaluate_paper_a.py`, `scripts/paper_a/retrieval_runner.py`, `backend/llm_infrastructure/retrieval/filters/scope_filter.py`, `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`.

### 8) `.sisyphus/plans/.legacy/paper-a-supervisor-review-plan.md`
- Status: active backlog, overlaps #7.
- Merge decision: execute as one paper stream; #7 is implementation track, #8 is verification/writing track.
- File consistency: `docs/papers/20_paper_a_scope/evidence_mapping.md`, `docs/papers/20_paper_a_scope/review/`, `scripts/paper_a/`.

### 9) `.sisyphus/plans/.legacy/agent-retrieval-stability-hardening.md`
- Status: completed.
- Decision: no active backlog.

### 10) `.sisyphus/plans/.legacy/paper-b-stability.md`
- Status: completed.
- Decision: no active backlog.

## Integrated Execution Backlog (Current-Code Synced)

## Phase A - Baseline Consistency Freeze (blocking)
- [x] A1. Create consolidated mismatch ledger at `.sisyphus/evidence/start-work/phase-a/consistency-ledger.md`.
- [x] A2. Confirm all execution references in this plan resolve to current files (or explicit "historical only").
- [x] A3. Freeze this plan as the single source and stop direct execution from `.sisyphus/plans/.legacy/`.

## Phase B - UI + Agent Retrieval Follow-ups
- [x] B1. Finish REQ-6 tabbed guided-selection UI and keep existing guided-confirm behavior stable.
  - Target files: `frontend/src/features/chat/components/guided-selection-panel.tsx`, `frontend/src/features/chat/pages/chat-page.tsx`, `frontend/src/features/chat/hooks/use-chat-session.ts`, `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`.
- [x] B2. Complete retrieval follow-ups (stage2 + strict override + sticky policy + SOP soft boost).
  - Target files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/config/settings.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`, `scripts/evaluation/evaluate_sop_agent_page_hit.py`.
- [x] B3. Replace stale test references with current test suite and pass regressions.
  - Test files: `tests/api/test_agent_stage2_retrieval.py`, `tests/api/test_agent_sticky_policy_followup_only.py`, `tests/api/test_agent_interrupt_resume_regression.py`, `backend/tests/test_sop_intent_heuristic.py`, `backend/tests/test_retrieve_node_sop_soft_boost.py`.
- [ ] B4. Execute HTTP-mode quality gate + before/after report pipeline.
  - Evidence: `.sisyphus/evidence/start-work/phase-b/http_eval/`, `.sisyphus/evidence/start-work/phase-b/before_after/`.

## Phase C - Chunk v3 + Chapter Grouping Integration
- [ ] C1. Finalize canonical doc_type + manifest + VLM coverage gates.
  - Target files: `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/validate_vlm.py`, `normalize.py`.
- [ ] C2. Harden embedding/ingest/verify contracts and mapping safety.
  - Target files: `scripts/chunk_v3/run_embedding.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/elasticsearch/mappings.py`.
- [ ] C3. Complete section grouping rollout path (JSONL regen + reindex + retrieval integration).
  - Target files: `scripts/chunk_v3/section_extractor.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/services/search_service.py`.
- [ ] C4. Run smoke/formal eval and update runbook alignment.
  - Target files: `scripts/chunk_v3/smoke_eval.py`, `scripts/chunk_v3/eval_sop_questionlist.py`, `.omc/plans/chunk_v3_embed_ingest_plan.md`.

## Phase D - Paper A Unified Stream (Implementation + Supervisor Verification)
- [ ] D1. Lock eval schema/policy artifacts and resolve doc-data-code mismatches first.
  - Target files: `scripts/paper_a/rebuild_query_gold_master_splits.py`, `scripts/paper_a/validate_master_eval_jsonl.py`, `scripts/paper_a/build_shared_and_scope.py`, `docs/papers/20_paper_a_scope/evidence_mapping.md`.
- [ ] D2. Complete evaluator parity and statistics pipeline.
  - Target files: `scripts/paper_a/evaluate_paper_a.py`, `scripts/paper_a/retrieval_runner.py`, `backend/llm_infrastructure/retrieval/filters/scope_filter.py`.
- [ ] D3. Execute reproducible runs and produce reviewer-grade audit outputs.
  - Evidence: `.sisyphus/evidence/start-work/phase-d/runs/`, `.sisyphus/evidence/start-work/phase-d/review/`.
- [ ] D4. Update manuscript-aligned docs using only evidence-backed claims.
  - Target files: `docs/papers/20_paper_a_scope/paper_a_scope.md`, `docs/papers/20_paper_a_scope/review/consistency_audit.md`, `docs/papers/20_paper_a_scope/review/`.

## Definition of Done
- All pending tasks in Phases A-D are complete.
- No stale reference remains unresolved in this plan.
- Every execution phase has evidence under `.sisyphus/evidence/start-work/`.
- No zombie process remains after each phase boundary.
