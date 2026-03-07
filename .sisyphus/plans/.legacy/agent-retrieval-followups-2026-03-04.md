# Agent Retrieval Follow-ups (Stage2 + MQ Sweep + SOP Boost)

## TL;DR
> **Summary**: Implement optional stage2 (doc-local) retrieval + early-page penalty to address SOP TOC bias, add MQ mode sweep automation to the `/api/agent/run` evaluator, and tune SOP soft-boost defaults for post-RRF scoring.
> **Deliverables**: (1) Stage2 retrieval behind env flags with deterministic RRF merge + tests, (2) Doc-id filtering plumbing down to ES filter builder, (3) MQ mode sweep (`off|fallback|on`) in evaluator + ablation report, (4) SOP soft boost default update + tests, (5) strict doc_type override bug fix + regression test, (6) sticky inheritance scope decision + tests, (7) real HTTP-mode quality gate + 79-question integrated report, (8) Docs updated to reflect stage2 and evaluation modes.
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: Doc-id filter plumbing → stage2 retrieve_node integration → `/api/agent/run` integration tests → MQ sweep + schema normalization → HTTP-mode quality gate → final 79-question before/after report

## Context
### Original Request
- Read and incorporate review feedback from `docs/papers/00_thesis_strategy/2026-03-04_agent_retrieval_todo-review.md` (review of prior retrieval improvements and next work).

### Interview Summary
- No additional user preferences provided; apply safe defaults.

### Repo-Grounded Findings
- Stage2/early-page-penalty code and env vars referenced by the review are not present in repo (confirmed via grep); plan treats stage2 as a new feature to implement.
- Current orchestration point for retrieval is `backend/llm_infrastructure/llm/langgraph_agent.py:1243` (`retrieve_node`).
- ES filter builder is `backend/llm_infrastructure/retrieval/engines/es_search.py:517` (`EsSearchEngine.build_filter`) and currently supports tenant/project/doc_type(s)/equip_ids/lang/device_names but not doc_ids.
- SOP soft boost is applied in `backend/llm_infrastructure/llm/langgraph_agent.py:1471` by multiplying `doc.score`; default is `1.05` in `backend/config/settings.py:663`.
- `/api/agent/run` request supports `mq_mode` override (see `backend/api/routers/agent.py:248`).

### Oracle Review (architecture)
- Implement stage2 as an extension inside `retrieve_node` (avoid changing graph shape / API contracts).
- Add optional `doc_ids` filter plumbing down to ES filter builder; default `None` everywhere.
- Merge stage1+stage2 deterministically using rank-based RRF (not raw-score comparison).
- Add strict caps (doc fan-out, stage2 top_k) and default-off env flags.

### Metis Review (gaps addressed)
- Stage2 defaults clarified: SOP-only; constrained to stage1 top doc_ids; hard caps to avoid fan-out.
- Determinism acceptance criteria: repeated runs must produce identical ordering.
- MQ sweep must set request `mq_mode` per run (not rely on env default) and emit per-mode aggregates.

## Work Objectives
### Core Objective
- Restore the stage2 + early-page penalty capability (as optional feature) to mitigate SOP TOC/cover bias, while keeping `/api/agent/run` stable and verifiable without a live ES.

### Scope Boundaries
- INCLUDE: stage2 doc-local re-search, early-page penalty, doc_id filter plumbing to ES, deterministic stage merge, MQ sweep evaluator automation, SOP boost default tuning, strict override bug fix, sticky scope policy fix, real HTTP-mode verification, tests/docs.
- EXCLUDE: new retrieval algorithms beyond stage2 + rank-based RRF merge; unrelated refactors; KPI dashboards (P2).

### Definition of Done (agent-executable)
- [ ] Stage2 feature can be enabled/disabled via env flags; disabled by default.
- [ ] When enabled (in tests with fakes), stage2 performs doc-local retrieval using a real doc_id filter (asserted in tests), and final ordering is deterministic across N repeats.
- [ ] Evaluator can run `mq_mode=off|fallback|on` sweeps and emits a reproducible ablation report; all produced JSONL validates.
- [ ] SOP soft boost default is updated and covered by a unit/integration test.
- [ ] `selected_doc_types_strict=True`일 때 `selected_doc_types`가 auto-parse에서 보존된다 (회귀 테스트 포함).
- [ ] Sticky 상속 범위 정책이 문서/코드/테스트에서 동일하게 고정된다.
- [ ] 최종 품질 판정은 TestClient가 아닌 실제 HTTP 모드 `/api/agent/run` 결과로 확인한다.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Primary quality verification: `/api/agent/run` HTTP mode against live server (`--api-base-url`, no `--use-testclient`).
- Fast feedback verification: FastAPI TestClient + dependency overrides (schema/tooling/integration guard tests).
- Unit tests: filter builder + deterministic RRF merge + SOP boost behavior.
- Script verification: evaluator supports both modes; TestClient for tooling checks, HTTP mode for final quality claims.
- Real quality gate: evaluator must run once in HTTP mode (`--api-base-url ...`, without `--use-testclient`) for final acceptance.
- Evidence: write artifacts under `.sisyphus/evidence/agent-retrieval-followups/`.

## Execution Strategy
### Parallel Execution Waves
Wave 1 (Foundations + Independent fixes)
- T1 Discovery evidence + invariants
- T2 Add agent settings/env flags for stage2 + penalty
- T3 Doc-id filter plumbing down to ES filter builder + unit tests
- T4 Deterministic RRF merge utility for RetrievalResult lists + unit tests
- T10 Fix strict_doc_type_override bug + regression test
- T12 Normalize eval JSONL semantics + validator sync

Wave 2 (Stage2 feature + sticky policy)
- T5 Implement stage2 + early-page penalty in `retrieve_node`
- T6 `/api/agent/run` integration tests for stage2 (doc_ids asserted) + determinism gate
- T11 Freeze sticky inheritance scope policy + align tests (after T10)

Wave 3 (Evaluation + tuning + docs)
- T7 MQ mode sweep automation + `mq_ablation_report.md`
- T8 SOP soft boost default update + tests
- T9 Docs updates (stage2 + flags + evaluator modes)

Wave 4 (Quality gate)
- T13 Real HTTP-mode quality gate
- T14 Final integrated 79-question before/after gate

### Dependency Matrix
- T5 depends on T2–T4.
- T6 depends on T5.
- T7 can run after T2 (to include new env keys), but functionally independent of stage2.
- T8 independent; safe to do in parallel with T7.
- T9 depends on T2/T5/T7/T8.
- T10 depends on current SOP intent path (independent of stage2).
- T11 depends on T10 (sticky policy should be validated after strict override fix).
- T12 can run in parallel with T7/T8.
- T13 depends on T5/T7/T8/T10/T11/T12.
- T14 depends on T13.

## TODOs
> Implementation + Test = ONE task. Every task includes QA scenarios with concrete evidence outputs.

- [x] 1. Capture Current State + Evidence (stage2 absent)

  **What to do**:
  - Confirm (via grep) that stage2/early-penalty symbols are absent in repo and record evidence.
  - Record current insertion points for retrieval + filters:
    - `backend/llm_infrastructure/llm/langgraph_agent.py:1243` (`retrieve_node`)
    - `backend/llm_infrastructure/retrieval/engines/es_search.py:517` (`build_filter`)

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: evidence capture only
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2-6 | Blocked By: none

  **Acceptance Criteria**:
  - [ ] Evidence files exist documenting grep results and insertion points.

  **QA Scenarios**:
  ```
  Scenario: Stage2 symbols absent
    Tool: Bash
    Steps: grep -R "AGENT_EARLY_PAGE_PENALTY_ENABLED\|AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED\|_apply_early_page_penalty\|stage2" -n backend || true
    Expected: no matches (or only docs references)
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-1-stage2-grep.txt
  ```

- [x] 2. Add Stage2 + Early-Page Penalty Settings (env flags + caps)

  **What to do**:
  - Extend `backend/config/settings.py:644` (`AgentSettings`) with additive fields (defaults are decision-complete):
    - `second_stage_doc_retrieve_enabled: bool = False`  (env: `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED`)
    - `early_page_penalty_enabled: bool = False`         (env: `AGENT_EARLY_PAGE_PENALTY_ENABLED`)
    - `early_page_penalty_max_page: int = 2`             (env: `AGENT_EARLY_PAGE_PENALTY_MAX_PAGE`)
    - `early_page_penalty_factor: float = 0.3`           (env: `AGENT_EARLY_PAGE_PENALTY_FACTOR`)
    - `second_stage_max_doc_ids: int = 1`                (env: `AGENT_SECOND_STAGE_MAX_DOC_IDS`)
    - `second_stage_top_k: int = 50`                     (env: `AGENT_SECOND_STAGE_TOP_K`)
  - Update evaluator key-env capture list to include these env vars:
    - `scripts/evaluation/evaluate_sop_agent_page_hit.py` (`KEY_ENV_VARS`)

  **Must NOT do**:
  - Do not enable stage2 by default.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: config contract + backward compat
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5-6,9 | Blocked By: 1

  **References**:
  - Settings: `backend/config/settings.py:644`
  - Evaluator env manifest: `scripts/evaluation/evaluate_sop_agent_page_hit.py:36`

  **Acceptance Criteria**:
  - [ ] `python -c "from backend.config.settings import agent_settings; print(agent_settings.second_stage_doc_retrieve_enabled)"` prints `False` by default.

  **QA Scenarios**:
  ```
  Scenario: Defaults are disabled
    Tool: Bash
    Steps: python -c "from backend.config.settings import agent_settings; print(agent_settings.second_stage_doc_retrieve_enabled, agent_settings.early_page_penalty_enabled)"
    Expected: prints "False False"
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-2-defaults.txt
  ```

- [x] 3. Implement Doc-ID Filtering Plumbing Down To ES (doc-local search primitive)

  **What to do**:
  - Add an optional `doc_ids: list[str] | None` plumbing path:
    - `backend/services/search_service.py:231` (`SearchService.search`) accepts `doc_ids` and forwards to retriever.
    - `backend/llm_infrastructure/retrieval/adapters/es_hybrid.py:115` (`EsHybridRetriever.retrieve`) accepts `doc_ids` and passes to `es_engine.build_filter`.
    - `backend/llm_infrastructure/retrieval/adapters/es_hybrid.py:241` (`EsDenseRetriever.retrieve`) accepts `doc_ids` and passes to `es_engine.build_filter`.
    - `backend/llm_infrastructure/retrieval/engines/es_search.py:517` (`EsSearchEngine.build_filter`) accepts `doc_ids` and adds an OR-terms filter for `doc_id`/`doc_id.keyword`.
    - `backend/llm_infrastructure/llm/langgraph_agent.py:284` (`SearchServiceRetriever.retrieve`) forwards `doc_ids` to `search_service.search`.
  - Implement doc-id filter using the same robustness strategy as existing filters (match both field and `.keyword`).
  - Add unit test(s) for `build_filter` verifying doc_ids are included and combined with other filters.

  **Must NOT do**:
  - Do not rely on post-filtering `selected_doc_ids` in `retrieve_node` as a substitute for doc-local search.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: multi-layer plumbing + backward compat
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5-6 | Blocked By: 1

  **References**:
  - ES filter builder: `backend/llm_infrastructure/retrieval/engines/es_search.py:517`
  - ES hybrid retriever: `backend/llm_infrastructure/retrieval/adapters/es_hybrid.py:115`
  - Graph adapter retriever: `backend/llm_infrastructure/llm/langgraph_agent.py:284`

  **Acceptance Criteria**:
  - [ ] New unit tests pass (executor to create under `backend/tests/`), proving doc_ids filter is present.

  **QA Scenarios**:
  ```
  Scenario: build_filter includes doc_ids
    Tool: Bash
    Steps: pytest -q backend/tests/test_es_build_filter_doc_ids.py
    Expected: pass
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-3-build-filter-doc-ids.txt
  ```

- [x] 4. Add Deterministic RRF Merge For RetrievalResult Lists (stage merge)

  **What to do**:
  - Add a pure utility (recommended location: `backend/llm_infrastructure/retrieval/rrf.py`) to merge two ranked `list[RetrievalResult]` via rank-based RRF:
    - Score: `sum(1/(k + rank))`, 1-based ranks
    - Dedupe key: prefer `(doc_id, chunk_id)` then `(doc_id, page)` then `(doc_id,)` using metadata.
    - Tie-break: `(rrf_score desc, best_rank asc, doc_id asc, page asc, chunk_id asc)`
  - Add unit tests for determinism and tie-break.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: correctness + determinism
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5 | Blocked By: 1

  **References**:
  - Existing hit-level RRF: `backend/llm_infrastructure/retrieval/rrf.py`
  - Existing deterministic tie-break pattern: `backend/llm_infrastructure/llm/langgraph_agent.py:1337`

  **Acceptance Criteria**:
  - [ ] `pytest -q backend/tests/test_rrf_merge_results.py` passes.

  **QA Scenarios**:
  ```
  Scenario: RRF merge is deterministic
    Tool: Bash
    Steps: pytest -q backend/tests/test_rrf_merge_results.py
    Expected: pass
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-4-rrf-merge-results.txt
  ```

- [ ] 5. Implement Stage2 Retrieval + Early-Page Penalty In `retrieve_node`

  **What to do**:
  - Extend `backend/llm_infrastructure/llm/langgraph_agent.py:1243` (`retrieve_node`) with an optional stage2 flow:
    1) Stage1: existing retrieval across `state.search_queries`.
    2) Early-page penalty (if enabled): for SOP-only, multiply `doc.score *= factor` when `page` is int and `page <= max_page`, then **re-sort** the affected list using the existing stable tie-break.
       - SOP-only predicate (decision-complete): apply when `state.get("sop_intent") is True` OR normalized selected_doc_types contains an SOP variant.
    3) Select stage2 doc_ids (if enabled): take the first `agent_settings.second_stage_max_doc_ids` unique `doc_id` values from stage1 docs ordered by current deterministic ordering.
       - Precedence rule (decision-complete): if request/state already provides `selected_doc_ids`, use those doc_ids (deduped + capped) instead of deriving from stage1.
    4) Stage2 doc-local retrieval: for each selected doc_id and for each query in `state.search_queries`, call `retriever.retrieve(..., doc_ids=[doc_id], top_k=agent_settings.second_stage_top_k, ...)`.
    5) Apply early-page penalty to each stage2 per-call list (same rule) and re-sort per-call list using stable tie-break.
    6) Merge **stage2** across (doc_id, query) lists using rank-based RRF in a deterministic fold order:
       - List order is doc_id selection order (from stage1) then query order (from `state.search_queries`).
    7) Merge stage1 + merged-stage2 via Task 4 RRF merge utility (k=60 unless already centralized).
    8) Continue with existing rerank + final selection logic.
  - Add minimal debug metadata to support evaluation (decision-complete): attach to the **top-level response metadata** (not per-doc):
    - `response.metadata["retrieval_stage2"] = {"enabled": bool, "doc_ids": [...]} `

  **Must NOT do**:
  - Do not change graph structure or `/api/agent/run` schema.
  - Do not run stage2 unless the feature flag is enabled.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: core retrieval behavior change + regression risk
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6,9 | Blocked By: 2-4

  **References**:
  - retrieve_node: `backend/llm_infrastructure/llm/langgraph_agent.py:1243`
  - SOP boost (context): `backend/llm_infrastructure/llm/langgraph_agent.py:1471`

  **Acceptance Criteria**:
  - [ ] New unit-level tests for stage2 selection/merge (executor to add) pass.

  **QA Scenarios**:
  ```
  Scenario: Stage2 is a no-op when disabled
    Tool: Bash
    Steps: pytest -q tests/api/test_agent_stage2_disabled.py
    Expected: pass; asserts only stage1 retrieval called
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-5-stage2-disabled.txt
  ```

- [ ] 6. Add `/api/agent/run` Integration Tests For Stage2 + Determinism

  **What to do**:
  - Add tests under `tests/api/` using FastAPI TestClient + dependency overrides (pattern from existing API tests) that:
    - Use a fake SearchService which records calls.
    - Assert stage1 calls have `doc_ids is None` and stage2 calls have `doc_ids == [expected_doc_id]`.
    - Return deterministic docs with pages 1-2 vs deeper pages to prove early-page penalty changes ordering.
    - Run the same request N times and assert identical `retrieved_docs[*].id/page` ordering.

  **Recommended Agent Profile**:
  - Category: `deep`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 9 | Blocked By: 5

  **References**:
  - Agent route: `backend/api/routers/agent.py:248`
  - Existing patterns: `tests/api/test_agent_rrf_and_sticky_gates.py`

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/api/test_agent_stage2_retrieval.py` passes.

  **QA Scenarios**:
  ```
  Scenario: Stage2 passes doc_ids filter to search
    Tool: Bash
    Steps: pytest -q tests/api/test_agent_stage2_retrieval.py
    Expected: pass; test asserts doc_ids observed
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-6-stage2-doc-ids.txt

  Scenario: Deterministic ordering across repeats
    Tool: Bash
    Steps: pytest -q tests/api/test_agent_stage2_retrieval.py
    Expected: pass; order identical across repeats
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-6-stage2-determinism.txt
  ```

- [ ] 7. Add MQ Mode Sweep Automation To Evaluator + Ablation Report

  **What to do**:
  - Extend `scripts/evaluation/evaluate_sop_agent_page_hit.py` with a sweep option (decision-complete):
    - New CLI: `--mq-modes off,fallback,on` (default empty → current behavior).
    - For each mode, set request payload `mq_mode=mode` (use `/api/agent/run` contract).
    - Write per-mode outputs under `out_dir/mq_{mode}/agent_eval.jsonl` and `out_dir/mq_{mode}/report.md`.
    - Write combined `out_dir/mq_ablation_report.md` summarizing doc/page hit and Jaccard@k per mode.
  - Ensure all per-mode JSONL passes `scripts/evaluation/validate_agent_eval_jsonl.py`.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: tooling + report correctness
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 9 | Blocked By: 2

  **References**:
  - Evaluator: `scripts/evaluation/evaluate_sop_agent_page_hit.py`
  - Report module: `scripts/evaluation/agent_eval_report.py`
  - Validator: `scripts/evaluation/validate_agent_eval_jsonl.py`

  **Acceptance Criteria**:
  - [ ] Running with sweep produces 3 folders + ablation report in TestClient mode.

  **QA Scenarios**:
  ```
  Scenario: MQ sweep produces valid JSONL per mode
    Tool: Bash
    Steps: python scripts/evaluation/evaluate_sop_agent_page_hit.py --out-dir /tmp/agent_eval_mq --use-testclient --limit 2 --mq-modes off,fallback,on
    Expected: creates /tmp/agent_eval_mq/mq_off/agent_eval.jsonl etc; validator passes for each
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-7-mq-sweep.txt
  ```

- [ ] 8. Tune SOP Soft Boost Default (1.05 → 1.30) + Test

  **What to do**:
  - Update `backend/config/settings.py:663` default `sop_soft_boost_factor` from `1.05` to `1.30`.
  - Add a unit test for `retrieve_node` proving SOP docs are boosted when `sop_intent=true` and mode=soft.

  **Must NOT do**:
  - Do not change SOP intent detection logic in this task.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: small config change + small test
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 9 | Blocked By: none

  **References**:
  - AgentSettings: `backend/config/settings.py:644`
  - SOP boost application: `backend/llm_infrastructure/llm/langgraph_agent.py:1471`

  **Acceptance Criteria**:
  - [ ] `pytest -q backend/tests/test_retrieve_node_sop_soft_boost.py` passes.

  **QA Scenarios**:
  ```
  Scenario: SOP soft boost applies factor
    Tool: Bash
    Steps: pytest -q backend/tests/test_retrieve_node_sop_soft_boost.py
    Expected: pass
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-8-sop-boost.txt
  ```

- [ ] 9. Update Docs To Reflect Stage2 + MQ Sweep

  **What to do**:
  - Update `docs/papers/00_thesis_strategy/2026-03-02_agent_retrieval_todo.md`:
    - Replace “stage2 N/A” notes with: stage2 is optional feature controlled by env flags; document semantics and defaults.
    - Document new env vars for stage2 and early-page penalty.
    - Document evaluator `--mq-modes` sweep output and how to interpret `mq_ablation_report.md`.

  **Recommended Agent Profile**:
  - Category: `writing`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 2,5,7,8

  **Acceptance Criteria**:
  - [ ] `grep -n "AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED" -n docs/papers/00_thesis_strategy/2026-03-02_agent_retrieval_todo.md` finds the new section.

  **QA Scenarios**:
  ```
  Scenario: Docs mention stage2 flags and mq sweep
    Tool: Bash
    Steps: grep -n "AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED\|--mq-modes\|mq_ablation_report" -n docs/papers/00_thesis_strategy/2026-03-02_agent_retrieval_todo.md
    Expected: contains references
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-9-docs-grep.txt
  ```

- [x] 10. Fix `strict_doc_type_override` Preservation Bug + Regression Test

  **What to do**:
  - Patch `auto_parse_node` in `backend/llm_infrastructure/llm/langgraph_agent.py` so that when `selected_doc_types_strict=True`, existing `selected_doc_types` are preserved.
  - Ensure SOP intent auto-parse does not overwrite strict user selection.
  - Make failing regression test pass:
    - `backend/tests/test_sop_intent_heuristic.py::test_hard_sets_sop_doc_type_but_does_not_override_strict_filter_doc_types`

  **Must NOT do**:
  - Do not relax strict mode behavior.
  - Do not change retrieval ranking logic in this task.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: localized bug fix + regression test
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 11,13 | Blocked By: none

  **References**:
  - Review basis: `docs/papers/00_thesis_strategy/2026-03-04_agent_retrieval_todo-review.md` (section 2-C)
  - Code: `backend/llm_infrastructure/llm/langgraph_agent.py`
  - Existing test: `backend/tests/test_sop_intent_heuristic.py`

  **Acceptance Criteria**:
  - [ ] strict override regression test passes.

  **QA Scenarios**:
  ```
  Scenario: strict selected_doc_types is preserved
    Tool: Bash
    Steps: pytest -q backend/tests/test_sop_intent_heuristic.py -k strict_filter_doc_types
    Expected: pass
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-10-strict-override.txt
  ```

- [ ] 11. Freeze Sticky Inheritance Scope Policy + Align Tests

  **What to do**:
  - Declare policy as decision-complete in code + docs:
    - Follow-up(`needs_history=True`): inherit `doc_type` only.
    - New session / non-follow-up: inherit nothing.
    - `devices/equip_ids` are not inherited unless explicitly provided in current request.
  - Implement in `auto_parse_node` (decision-complete):
    - Keep: `prev_doc_types` fallback when `needs_history=True`.
    - Remove: `prev_devices` and `prev_equip_ids` fallback even when `needs_history=True`.
    - Concretely: set `prev_devices=[]` and `prev_equip_ids=[]` regardless of `needs_history`.
    - Ensure `devices = detected_devices` (no fallback) and `equip_ids = detected_equip_ids` (no fallback).
  - Align implementation and tests under `tests/api/test_agent_sticky_policy_followup_only.py`.
  - Document policy in strategy/todo docs.

  **Must NOT do**:
  - Do not silently keep mixed behavior across fields.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: policy-level behavior lock + test alignment
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 13 | Blocked By: 10

  **References**:
  - Sticky tests: `tests/api/test_agent_sticky_policy_followup_only.py`
  - Auto-parse inheritance: `backend/llm_infrastructure/llm/langgraph_agent.py:2786`
  - Review basis: `docs/papers/00_thesis_strategy/2026-03-04_agent_retrieval_todo-review.md` (section 3-7)

  **Acceptance Criteria**:
  - [ ] sticky behavior tests reflect and enforce one consistent policy.

  **QA Scenarios**:
  ```
  Scenario: follow-up inherits only doc_type
    Tool: Bash
    Steps: pytest -q tests/api/test_agent_sticky_policy_followup_only.py
    Expected: pass (doc_type inherit yes, devices/equip_ids inherit no)
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-11-sticky-scope.txt
  ```

- [ ] 12. Normalize Eval JSONL Semantics (`answer`, `trace.retry_count`) + Validator Sync

  **What to do**:
  - Confirm evaluator always writes `answer` and `trace.retry_count` in JSONL row.
  - Normalize fallback mapping: `metadata.attempts` -> `trace.retry_count` (single canonical field).
  - Ensure validator checks canonical schema only.
  - Add/refresh fixture tests for good/bad JSONL examples.

  **Must NOT do**:
  - Do not introduce duplicate retry fields in output schema.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 13 | Blocked By: none

  **References**:
  - Evaluator: `scripts/evaluation/evaluate_sop_agent_page_hit.py`
  - Validator: `scripts/evaluation/validate_agent_eval_jsonl.py`
  - Reporter: `scripts/evaluation/agent_eval_report.py`

  **Acceptance Criteria**:
  - [ ] validator passes for generated JSONL and enforces canonical keys.

  **QA Scenarios**:
  ```
  Scenario: generated JSONL includes answer + trace.retry_count
    Tool: Bash
    Steps: python scripts/evaluation/evaluate_sop_agent_page_hit.py --out-dir /tmp/agent_eval_schema --use-testclient --limit 2 && python scripts/evaluation/validate_agent_eval_jsonl.py --jsonl /tmp/agent_eval_schema/agent_eval.jsonl
    Expected: pass
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-12-jsonl-schema.txt
  ```

- [ ] 13. Add Real HTTP-Mode Quality Gate (Non-TestClient)

  **What to do**:
  - Run evaluator against live `/api/agent/run` endpoint (without `--use-testclient`) using fixed env manifest.
  - Produce artifacts:
    - `http_eval/agent_eval.jsonl`
    - `http_eval/report.md`
    - `http_eval/mq_ablation_report.md` (if sweep enabled)
  - Document explicit separation:
    - TestClient = schema/tooling checks
    - HTTP mode = quality/retrieval behavior checks

  **Must NOT do**:
  - Do not claim quality improvements based only on TestClient runs.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: end-to-end quality gate
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: 14 | Blocked By: 5,7,8,10,11,12

  **References**:
  - Evaluator CLI: `scripts/evaluation/evaluate_sop_agent_page_hit.py`
  - Review basis: `docs/papers/00_thesis_strategy/2026-03-04_agent_retrieval_todo-review.md` (section 2-E)

  **Acceptance Criteria**:
  - [ ] HTTP-mode report generated successfully and included in evidence bundle.

  **QA Scenarios**:
  ```
  Scenario: HTTP mode evaluation run
    Tool: Bash
    Steps: python scripts/evaluation/evaluate_sop_agent_page_hit.py --api-base-url http://127.0.0.1:18021 --out-dir .sisyphus/evidence/agent-retrieval-followups/http_eval --limit 79 --mq-modes off,fallback,on
    Expected: JSONL/report files generated; validator passes
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-13-http-mode.txt
  ```

- [ ] 14. Final Integrated 79-Question Before/After Gate

  **What to do**:
  - Produce BEFORE/AFTER artifacts via HTTP-mode evaluator:
    - BEFORE precondition (decision-complete): API server is started with:
      - `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED=false`
      - `AGENT_EARLY_PAGE_PENALTY_ENABLED=false`
    - AFTER precondition (decision-complete): restart API server with:
      - `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED=true`
      - `AGENT_EARLY_PAGE_PENALTY_ENABLED=true`
    - BEFORE (stage2 disabled): `.sisyphus/evidence/agent-retrieval-followups/final_79/before/agent_eval.jsonl`
    - AFTER (stage2 + early penalty enabled): `.sisyphus/evidence/agent-retrieval-followups/final_79/after/agent_eval.jsonl`
  - Extend `scripts/evaluation/agent_eval_report.py` so the report includes a KPI table computed from JSONL `docs` + `expected_pages`:
    - doc-hit@10
    - page-hit@1/@3/@5
    - first_page=1 count (docs[0].page == 1)
    - Jaccard@k (when available)
  - Generate final before/after report to:
    - `.sisyphus/evidence/agent-retrieval-followups/final_79/report.md`

  **Must NOT do**:
  - Do not close follow-up plan without this integrated gate.

  **Recommended Agent Profile**:
  - Category: `deep`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: none | Blocked By: 13

  **References**:
  - Evaluator: `scripts/evaluation/evaluate_sop_agent_page_hit.py`
  - Report generator: `scripts/evaluation/agent_eval_report.py`

  **Acceptance Criteria**:
  - [ ] BEFORE/AFTER JSONL exist under `.sisyphus/evidence/agent-retrieval-followups/final_79/`.
  - [ ] Final integrated report exists with before/after KPI table and regression flags.
  - [ ] Final report includes KPI rows/columns for doc-hit@10, page-hit@1/@3/@5, and first_page=1 count.

  **QA Scenarios**:
  ```
  Scenario: generate BEFORE 79-question JSONL (HTTP mode)
    Tool: Bash
    Steps: python scripts/evaluation/evaluate_sop_agent_page_hit.py --api-base-url http://127.0.0.1:18021 --out-dir .sisyphus/evidence/agent-retrieval-followups/final_79/before --limit 79
    Expected: creates .sisyphus/evidence/agent-retrieval-followups/final_79/before/agent_eval.jsonl and report.md
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-14-before-run.txt

  Scenario: generate AFTER 79-question JSONL (HTTP mode)
    Tool: Bash
    Steps: python scripts/evaluation/evaluate_sop_agent_page_hit.py --api-base-url http://127.0.0.1:18021 --out-dir .sisyphus/evidence/agent-retrieval-followups/final_79/after --limit 79
    Expected: creates .sisyphus/evidence/agent-retrieval-followups/final_79/after/agent_eval.jsonl and report.md
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-14-after-run.txt

  Scenario: final before/after report generation
    Tool: Bash
    Steps: python scripts/evaluation/agent_eval_report.py --before-jsonl .sisyphus/evidence/agent-retrieval-followups/final_79/before/agent_eval.jsonl --after-jsonl .sisyphus/evidence/agent-retrieval-followups/final_79/after/agent_eval.jsonl --out .sisyphus/evidence/agent-retrieval-followups/final_79/report.md
    Expected: report generated with KPI deltas
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-14-final-gate.txt

  Scenario: report contains KPI table fields
    Tool: Bash
    Steps: grep -n "doc-hit@10\|page-hit@1\|page-hit@3\|page-hit@5\|first_page=1" -n .sisyphus/evidence/agent-retrieval-followups/final_79/report.md
    Expected: matches found for all KPI labels
    Evidence: .sisyphus/evidence/agent-retrieval-followups/task-14-report-grep.txt
  ```

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real QA Simulation — unspecified-high (HTTP mode + evaluator sweep; TestClient is supplemental)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: `feat(retrieval): add stage2 settings and doc_id filter plumbing`
- Commit 2: `feat(agent): implement stage2 retrieval + early-page penalty with deterministic merge`
- Commit 3: `fix(agent): preserve strict doc_type override + freeze sticky inheritance scope`
- Commit 4: `feat(eval): add mq_mode sweep + normalize JSONL schema + tune sop soft boost + docs`
- Commit 5: `test(eval): HTTP-mode quality gate + 79-question before/after report`

## Success Criteria
- Stage2 can be enabled safely (default-off) and is verifiably doc-local (doc_ids filter asserted in tests).
- Deterministic ordering gates pass across repeats in `/api/agent/run` integration tests.
- MQ sweep automation produces reproducible, schema-valid outputs and a clear ablation report.
- strict `selected_doc_types` user intent is preserved (regression test locked).
- Sticky inheritance scope is explicitly defined and test-enforced.
- Final acceptance is based on integrated HTTP-mode 79-question before/after report.
