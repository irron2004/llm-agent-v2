# Unified Start-Work Plan (Manual Reconciliation)

## Intent
- This file is the only execution plan for /start-work.
- `.sisyphus/plans/.legacy/` is source spec storage, not a direct execution target.
- Active writable plan files in `.sisyphus/plans/` are limited to `start-work-all-remaining.md` and `start-work-remaining.md`.
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
- Do not recreate legacy task files in `.sisyphus/plans/`; keep legacy specs only in `.sisyphus/plans/.legacy/`.
- Do not run wildcard restore/create flows against `.sisyphus/plans/*.md`.

## Legacy-by-Legacy Consistency Review

### 1) `.sisyphus/plans/.legacy/ui-autoparse-confirm-task-mode-2026-03-05.md`
- Status: completed.
- Decision: keep as historical evidence only; no active backlog.

### 2) `.sisyphus/plans/.legacy/ui-chat-improvements-v2.md`
- Status: partially complete.
- Decision: keep only REQ-2 robustness hardening in active backlog; all other REQ items remain closed.
- File consistency: `frontend/src/features/chat/components/guided-selection-panel.tsx`, `frontend/src/features/chat/hooks/use-chat-session.ts`, `frontend/src/features/chat/pages/chat-page.tsx`, `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`.

### 3) `.sisyphus/plans/.legacy/chunk_v3_embed_ingest_plan.md`
- Status: mostly complete (C1/C2/C4 done), one operational rollout task remains.
- Scope alignment: keep section-rollout execution (C3) and remove stale symbol-style references.
- File consistency: `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/run_embedding.py`, `scripts/chunk_v3/run_ingest.py`, `scripts/chunk_v3/validate_vlm.py`, `backend/llm_infrastructure/elasticsearch/mappings.py`, `backend/domain/doc_type_mapping.py`, `scripts/evaluation/evaluate_sop_agent_page_hit.py`.

### 4) `.sisyphus/plans/.legacy/chapter-grouping-retrieval.md`
- Status: partially complete, operational reindex follow-up remains.
- Scope alignment: keep reindex + retrieval integration tasks only.
- File consistency: `scripts/chunk_v3/section_extractor.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/llm_infrastructure/llm/langgraph_agent.py`.

### 5) `.sisyphus/plans/.legacy/agent-retrieval-followups-2026-03-04.md`
- Status: partially complete.
- Scope alignment: keep required Stage2/MQ follow-ups, but remove B4-coupled before/after gate tasks from active execution.
- Mismatch corrected:
  - Legacy ref tests/api/test_agent_rrf_and_sticky_gates.py does not exist.
  - Current verification set: `tests/api/test_agent_stage2_retrieval.py`, `tests/api/test_agent_sticky_policy_followup_only.py`, `tests/api/test_agent_interrupt_resume_regression.py`, `backend/tests/test_sop_intent_heuristic.py`, `backend/tests/test_retrieve_node_sop_soft_boost.py`.

### 6) `.sisyphus/plans/.legacy/before-after-regression-compare.md`
- Status: deferred by user decision (2026-03-09).
- Scope alignment: remove from active backlog for now; execute only on later explicit request.
- File consistency: `scripts/evaluation/regression_compare_manifest.py`, `scripts/evaluation/evaluate_sop_agent_page_hit.py`, `scripts/evaluation/agent_eval_report.py`, `tests/evaluation/test_regression_compare_manifest.py`.

### 7) `.sisyphus/plans/.legacy/paper-a-scope-implementation.md`
- Status: deferred (lowest priority, separate future request).
- Scope alignment: exclude from current active execution queue.
- File consistency: `scripts/paper_a/build_corpus_meta.py`, `scripts/paper_a/build_shared_and_scope.py`, `scripts/paper_a/build_family_map.py`, `scripts/paper_a/build_eval_sets.py`, `scripts/paper_a/evaluate_paper_a.py`, `scripts/paper_a/retrieval_runner.py`, `backend/llm_infrastructure/retrieval/filters/scope_filter.py`, `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`.

### 8) `.sisyphus/plans/.legacy/paper-a-supervisor-review-plan.md`
- Status: deferred (lowest priority, separate future request).
- Merge decision: keep as historical paper stream; do not execute in this cycle.
- File consistency: `docs/papers/20_paper_a_scope/evidence_mapping.md`, `docs/papers/20_paper_a_scope/review/`, `scripts/paper_a/`.

### 9) `.sisyphus/plans/.legacy/agent-retrieval-stability-hardening.md`
- Status: completed.
- Decision: no active backlog.

### 10) `.sisyphus/plans/.legacy/paper-b-stability.md`
- Status: deferred (lowest priority, separate future request).
- Decision: exclude from current active execution queue.

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
- [x] B4. De-scope HTTP-mode before/after gate from this cycle (user-deferred).
  - Decision: keep historical evidence only; do not execute B4 unless explicitly re-requested.
  - Evidence (historical): `.sisyphus/evidence/start-work/phase-b/http_eval/`, `.sisyphus/evidence/start-work/phase-b/before_after/`.
- [x] B5. Top priority: finish REQ-2 model-format robustness hardening.
  - Scope: shared parse fallback util, judge JSON mode, retry cap, translate/query_rewrite normalization, prompt hardening.
  - Primary files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/llm_infrastructure/llm/prompts/auto_parse_v1.yaml`, `backend/llm_infrastructure/llm/prompts/router_v1.yaml`, `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`.

## Phase C - Chunk v3 + Chapter Grouping Integration
- [x] C1. Finalize canonical doc_type + manifest + VLM coverage gates.
  - Target files: `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/validate_vlm.py`, `normalize.py`.
- [x] C2. Harden embedding/ingest/verify contracts and mapping safety.
  - Target files: `scripts/chunk_v3/run_embedding.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/elasticsearch/mappings.py`.
- [x] C3. Re-verify section grouping rollout already in progress/executed.
  - Scope: validate section_* fields in regenerated JSONL, verify `chunk_v3_content` sync for 3 models, and confirm retrieval section-expansion behavior on current indices.
  - Target files/scripts: `scripts/chunk_v3/section_extractor.py`, `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/llm_infrastructure/llm/langgraph_agent.py`.
- [x] C4. Run smoke/formal eval and update runbook alignment.
  - Target files: `scripts/chunk_v3/smoke_eval.py`, `scripts/chunk_v3/eval_sop_questionlist.py`, `.sisyphus/plans/.legacy/chunk_v3_embed_ingest_plan.md`, `docs/2026-03-07-3Model-Embedding-Eval-Report.md`.

## Phase D - Paper A Unified Stream (Implementation + Supervisor Verification)
- [x] D1. Lock eval schema/policy artifacts and resolve doc-data-code mismatches first.
  - Target files: `scripts/paper_a/rebuild_query_gold_master_splits.py`, `scripts/paper_a/validate_master_eval_jsonl.py`, `scripts/paper_a/build_shared_and_scope.py`, `docs/papers/20_paper_a_scope/evidence_mapping.md`.
- [x] D2. Complete evaluator parity and statistics pipeline.
  - Target files: `scripts/paper_a/evaluate_paper_a.py`, `scripts/paper_a/retrieval_runner.py`, `backend/llm_infrastructure/retrieval/filters/scope_filter.py`.
- [x] D3. Execute reproducible runs and produce reviewer-grade audit outputs.
  - Evidence: `.sisyphus/evidence/paper-a/runs/`, `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md`, `docs/papers/20_paper_a_scope/review/reviewer_report.md`.
- [x] D4. Update manuscript-aligned docs using only evidence-backed claims.
  - Target files: `docs/papers/20_paper_a_scope/paper_a_scope.md`, `docs/papers/20_paper_a_scope/review/consistency_audit.md`, `docs/papers/20_paper_a_scope/review/`.

## Phase E - Agent Retrieval Follow-up Replan (C3-aware)
- [x] E1. Rebaseline Stage2/MQ follow-up plan against current C3 outputs.
  - Scope: align Stage2/MQ tasks to canonical doc_type, section grouping metadata availability, and current index state.
  - Reference plans: `.sisyphus/plans/.legacy/agent-retrieval-followups-2026-03-04.md`, `.sisyphus/plans/.legacy/chapter-grouping-retrieval.md`.
- [x] E2. Execute required retrieval follow-up subset (excluding B4/before-after gates).
  - Include: stage2 retrieve_node integration, integration tests, sticky policy freeze, eval JSONL normalization, docs sync.
  - Exclude: HTTP before/after quality gate and final before/after report tasks.

## Definition of Done
- All pending tasks in Phases A-E are complete.
- Deferred (not in active queue): B4 before/after gate, Paper A stream, Paper B stream, and cross-plan F* audit bundles.
- No stale reference remains unresolved in this plan.
- Every execution phase has evidence under `.sisyphus/evidence/start-work/` or canonical phase-specific evidence paths referenced above.
- No zombie process remains after each phase boundary.
