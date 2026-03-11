# 2026-03-11 RRF Stage1 Path Fix (App-level RRF)

## TL;DR
> **Summary**: Replace ES native `rank.rrf` stage1 hybrid path with app-level RRF fusion (dense + sparse + `rrf.py`) and remove silent script_score fallback; add RRF metadata + regression tests + chat-flow eval harness.
> **Deliverables**: ES stage1 app-level RRF, no native RRF request, no silent fallback, rrf_* metadata, null-_score parsing fix, regression tests, minimal eval runner.
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: Implement app-level RRF in ES engine → add RRF metadata + no-fallback policy → tests → eval harness → docker-compose verification

## Context
### Original Request
- `docs/2026-03-11-rrf-문제해결.md` 내용을 기준으로 현재 런타임의 stage1 hybrid가 ES native RRF + fallback으로 동작하는 문제를 해결한다.

### Interview Summary
- User requested full, decision-complete plan.

### Metis Review (gaps addressed)
- Default policy selected: `use_rrf=True` uses app-level RRF with no silent fallback; script_score is explicit only.
- Weights ignored in RRF mode (matches `backend/api/routers/search.py`).
- RRF metadata added outside `rrf.py` to avoid back-compat break with `backend/tests/test_rrf_merge_results.py`.
- Add explicit tests for: no `rank.rrf` request, no fallback log, deterministic merge, `_score=None` parsing.

### Oracle Review (pitfalls incorporated)
- App-level RRF requires careful candidate window sizing (`top_k*2`) and consistent dedupe keys.
- Ensure `chunk_id` is always present in metadata to avoid over-deduping.
- Fix `_score=None` parsing when ES requests use `sort`.

## Work Objectives
### Core Objective
- Make stage1 hybrid retrieval use app-level RRF fusion consistently and observably.

### Deliverables
- `backend/llm_infrastructure/retrieval/engines/es_search.py`: app-level RRF implementation for `use_rrf=True`.
- No ES native RRF request building (`sub_searches` + `rank.rrf`) in runtime path.
- No silent `script_score` fallback when `use_rrf=True`.
- Stage1 fused results include `rrf_dense_rank`, `rrf_sparse_rank`, `rrf_score`, `rrf_k` metadata.
- Fix ES hit parsing for `_score=None` to prevent page/chunk fetch errors.
- Regression tests covering these invariants.
- Minimal chat-flow retrieval eval runner that captures top-k docs + debug metadata without requiring LLM answer generation.

### Definition of Done (verifiable)
- `cd backend && uv run pytest tests/test_rrf_merge_results.py -q`
- `cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q`
- `cd backend && uv run pytest tests/test_es_parse_hits_null_score.py -q`
- `cd backend && uv run pytest tests/test_chat_flow_rrf_eval_runner.py -q`
- `make dev-up` then run eval runner against `http://localhost:8011` producing an output JSONL artifact.

### Must Have
- Stage1 hybrid with `use_rrf=True` performs 2 ES searches (dense + sparse) and merges via app-level RRF.
- No logging string `RRF search failed, falling back to script_score` emitted in normal RRF mode.
- `_parse_hits` tolerant of `_score=None`.

### Must NOT Have
- No reintroduction of ES native RRF path (`rank.rrf`, `sub_searches`).
- No “silent fallback” from RRF to script_score; any fallback must be explicit by configuration/flag.
- No changes to core RRF utility behavior (`backend/llm_infrastructure/retrieval/rrf.py`) that break `backend/tests/test_rrf_merge_results.py`.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (pytest)
- QA policy: every task includes runnable assertions (unit tests + docker-compose smoke)
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves

Wave 1 (Foundation - Retrieval Path + Bugfix)
- ES engine app-level RRF implementation + no-fallback policy
- `_score=None` parse fix
- RRF metadata injection strategy

Wave 2 (Safety Nets + Evaluation)
- Regression tests + log assertions
- Chat-flow eval runner + sample dataset + CI-safe mocked test
- Docker compose verification steps + doc sync notes

### Dependency Matrix
- Wave 1 blocks Wave 2 tests (tests require final API/engine behavior)

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Fix ES Hit Parsing for `_score=None` (page/chunk fetch stability)

  **What to do**:
  - Update `backend/llm_infrastructure/retrieval/engines/es_search.py` `EsSearchEngine._parse_hits` (see `backend/llm_infrastructure/retrieval/engines/es_search.py:461`) so `_score` being `None` (or non-numeric) does NOT crash; default to `0.0`.
  - Ensure `chunk_id` is always present in `EsSearchHit.metadata` (if `_source.chunk_id` missing, inject the computed `chunk_id` used by `EsSearchHit.chunk_id`). This prevents over-deduping when stage1 RRF uses `rrf.py` dedupe key `chunk_id -> page -> doc_id`.

  **Must NOT do**:
  - Do not change response shapes in API routers.

  **Recommended Agent Profile**:
  - Category: `quick` — small, localized bugfix + tests.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 3, 4 | Blocked By: none

  **References**:
  - Code: `backend/llm_infrastructure/retrieval/engines/es_search.py:461` — `_parse_hits` uses `float(hit.get('_score', 0.0))`.
  - Symptom spec: `docs/2026-03-11-rrf-문제해결.md:266` — page fetch error mention.

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_es_parse_hits_null_score.py -q`

  **QA Scenarios**:
  ```
  Scenario: _parse_hits tolerates null _score
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_parse_hits_null_score.py -q
    Expected: test passes; no TypeError from float(None)
    Evidence: .sisyphus/evidence/task-1-null-score.txt
  ```

  **Commit**: YES | Message: `fix(retrieval): tolerate null _score in es hit parsing` | Files: `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/tests/test_es_parse_hits_null_score.py`

- [x] 2. Re-implement Stage1 Hybrid RRF as App-level RRF (remove ES native `rank.rrf`)

  **What to do**:
  - Rework `backend/llm_infrastructure/retrieval/engines/es_search.py`:
    - `EsSearchEngine.hybrid_search(... use_rrf=True ...)` MUST route to an app-level RRF implementation (keep method signature).
    - Replace `_hybrid_search_rrf` (see `backend/llm_infrastructure/retrieval/engines/es_search.py:223`) so it does:
      1) Run dense search (kNN) with `top_n = top_k * 2` and same `filters`.
      2) Run sparse search (BM25) with `top_n = top_k * 2`, using `_build_text_query(query_text, device_boost, device_boost_weight)` and same `filters`.
      3) Convert both hit lists to `RetrievalResult` lists (retain metadata).
      4) Merge via `backend/llm_infrastructure/retrieval/rrf.py:merge_retrieval_result_lists_rrf` with `k=rrf_k`.
      5) Convert fused `RetrievalResult` back into `EsSearchHit` (preserve content/page/raw_text/metadata).
      6) Return top `top_k` fused hits.
  - Remove ES native RRF request building using `sub_searches` + `rank.rrf` (currently at `backend/llm_infrastructure/retrieval/engines/es_search.py:252`).

  **Must NOT do**:
  - Do not keep any silent fallback from RRF mode to script_score.
  - Do not modify `backend/llm_infrastructure/retrieval/rrf.py` behavior.

  **Recommended Agent Profile**:
  - Category: `deep` — core retrieval behavior change.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 3, 4, 5 | Blocked By: 1

  **References**:
  - Current native path: `backend/llm_infrastructure/retrieval/engines/es_search.py:223` and `backend/llm_infrastructure/retrieval/engines/es_search.py:252`.
  - Fallback log: `backend/llm_infrastructure/retrieval/engines/es_search.py:272`.
  - App-level RRF util: `backend/llm_infrastructure/retrieval/rrf.py`.
  - Stage2 already uses RRF util: `backend/llm_infrastructure/llm/langgraph_agent.py:1855`, `backend/llm_infrastructure/llm/langgraph_agent.py:1858`.
  - ES adapter uses `use_rrf=True` by default: `backend/llm_infrastructure/retrieval/adapters/es_hybrid.py:83`.

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q`
  - [ ] Grep invariant: `rank.rrf` is not present in ES request bodies built for stage1 RRF path.

  **QA Scenarios**:
  ```
  Scenario: stage1 use_rrf performs app-level merge
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q
    Expected: test passes; engine issues dense + sparse searches and no rank.rrf body
    Evidence: .sisyphus/evidence/task-2-stage1-app-rrf.txt
  ```

  **Commit**: YES | Message: `fix(retrieval): replace native es rrf with app-level fusion` | Files: `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/tests/test_es_stage1_app_rrf.py`

- [x] 3. Enforce No Silent Fallback (script_score only explicit)

  **What to do**:
  - Remove the `try/except` in the RRF path that logs `RRF search failed, falling back to script_score` and calls `_hybrid_search_script_score` (currently `backend/llm_infrastructure/retrieval/engines/es_search.py:267`).
  - Update docstrings in `backend/llm_infrastructure/retrieval/engines/es_search.py:181` to reflect:
    - `use_rrf=True` = app-level RRF.
    - `use_rrf=False` = script_score weighted.

  **Must NOT do**:
  - Do not introduce a hidden config flag that re-enables fallback by default.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5 | Blocked By: 2

  **References**:
  - Spec requirement: `docs/2026-03-11-rrf-문제해결.md:192`.
  - Current fallback log: `backend/llm_infrastructure/retrieval/engines/es_search.py:272`.

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q` (contains assertion that fallback log is absent)

  **QA Scenarios**:
  ```
  Scenario: no fallback log exists in RRF path
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q
    Expected: test passes; no 'falling back to script_score' is logged
    Evidence: .sisyphus/evidence/task-3-no-fallback.txt
  ```

  **Commit**: YES | Message: `fix(retrieval): remove silent rrf->script_score fallback` | Files: `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/tests/test_es_stage1_app_rrf.py`

- [x] 4. Add Stage1 RRF Metadata (rrf_dense_rank/rrf_sparse_rank/rrf_score/rrf_k)

  **What to do**:
  - In the app-level RRF fusion code (Task 2), compute per-source ranks and attach metadata:
    - `rrf_dense_rank`: 1-based rank in dense list for the fused dedupe key (or `None`).
    - `rrf_sparse_rank`: 1-based rank in sparse list for the fused dedupe key (or `None`).
    - `rrf_score`: the fused RRF score (should match returned hit score).
    - `rrf_k`: the RRF constant used.
  - Dedupe key MUST match `backend/llm_infrastructure/retrieval/rrf.py` behavior (chunk_id -> page -> doc_id). Implement the same key in the ES fusion code (do not import private helpers).

  **Must NOT do**:
  - Do not modify `backend/llm_infrastructure/retrieval/rrf.py` to emit these fields.

  **Recommended Agent Profile**:
  - Category: `deep`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 5 | Blocked By: 2

  **References**:
  - Metadata spec: `docs/2026-03-11-rrf-문제해결.md:226`.
  - Existing RRF util must remain compatible with: `backend/tests/test_rrf_merge_results.py`.

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q` (asserts rrf_* metadata exists)

  **QA Scenarios**:
  ```
  Scenario: fused hits carry rrf_* metadata
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q
    Expected: test passes; metadata contains rrf_dense_rank/rrf_sparse_rank/rrf_score/rrf_k
    Evidence: .sisyphus/evidence/task-4-rrf-metadata.txt
  ```

  **Commit**: YES | Message: `feat(retrieval): attach rrf rank metadata in stage1 fusion` | Files: `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/tests/test_es_stage1_app_rrf.py`

- [x] 5. Add Regression Tests Preventing Native ES RRF Reintroduction

  **What to do**:
  - Add `backend/tests/test_es_stage1_app_rrf.py` (or expand if already created) with these assertions:
    - When `EsSearchEngine.hybrid_search(use_rrf=True)` runs, the ES client never receives a request body containing `rank` with `rrf`, nor `sub_searches`.
    - It issues exactly two searches (dense + sparse), each with `size=top_k*2`.
    - It does not emit the fallback log string.
    - Returned hits are deterministic (stable ordering) on fixed dense/sparse result lists.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — tests with mocks + log assertions.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7, 8 | Blocked By: 2, 3, 4

  **References**:
  - Existing RRF tests: `backend/tests/test_rrf_merge_results.py`.
  - Native RRF request body: `backend/llm_infrastructure/retrieval/engines/es_search.py:252`.

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q`

  **QA Scenarios**:
  ```
  Scenario: unit tests prevent rank.rrf regression
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_stage1_app_rrf.py -q
    Expected: test passes; assertions fail if sub_searches/rank.rrf reappears
    Evidence: .sisyphus/evidence/task-5-regression-tests.txt
  ```

  **Commit**: YES | Message: `test(retrieval): prevent native rrf + assert rrf metadata` | Files: `backend/tests/test_es_stage1_app_rrf.py`

- [x] 6. Implement Chat-flow Retrieval Eval Runner (no answer generation)

  **What to do**:
  - Add new script: `scripts/evaluation/run_chat_flow_retrieval_rrf_eval.py`.
  - Input: JSONL with at least `{ "qid": str, "query": str }`.
  - For each query, call `POST {base}/api/agent/run` with:
    - `message`: query
    - `ask_user_after_retrieve: true` (forces interrupt after retrieval)
    - `auto_parse: false`
    - `guided_confirm: false`
    - `use_canonical_retrieval: false` (must hit real chat flow)
    - `mq_mode: off` (reduce LLM usage)
    - `max_attempts: 0`
    - `mode: base` (skip judge/retry)
    - `thread_id`: unique per query (e.g., `{qid}`)
  - Output: JSONL rows with:
    - `qid`, `query`, `thread_id`
    - `search_queries` (from response)
    - `retrieved_docs` (doc_id, page, metadata subset including rrf_* if present)
    - `metadata.retrieval_debug` if present
    - `interrupted` + `interrupt_payload.type` (should be `retrieval_review`)
  - Write artifacts into an output path controlled by CLI flag.

  **Must NOT do**:
  - Do not require browser/UI.
  - Do not require an LLM answer generation (the runner stops at retrieval interrupt).

  **Recommended Agent Profile**:
  - Category: `writing` — CLI runner + docs.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8 | Blocked By: 2, 3, 4

  **References**:
  - Existing runner patterns: `scripts/evaluation/run_agent_regression.py`.
  - Agent endpoint: `backend/api/routers/agent.py:953` (`POST /agent/run`).

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run python -m scripts.evaluation.run_chat_flow_retrieval_rrf_eval --help`

  **QA Scenarios**:
  ```
  Scenario: runner produces JSONL output
    Tool: Bash
    Steps:
      1) make dev-up
      2) cd backend && uv run python -m scripts.evaluation.run_chat_flow_retrieval_rrf_eval --api-base-url http://localhost:8011 --queries scripts/evaluation/fixtures/rrf_smoke.jsonl --out /tmp/rrf_eval.jsonl
    Expected: /tmp/rrf_eval.jsonl exists; each row contains retrieved_docs and interrupt_payload.type == retrieval_review
    Evidence: .sisyphus/evidence/task-6-eval-runner.txt
  ```

  **Commit**: YES | Message: `chore(eval): add chat-flow retrieval rrf eval runner` | Files: `scripts/evaluation/run_chat_flow_retrieval_rrf_eval.py`, `scripts/evaluation/fixtures/rrf_smoke.jsonl`

- [x] 7. Add CI-safe Unit Test for Eval Runner (mocked HTTP)

  **What to do**:
  - Add `backend/tests/test_chat_flow_rrf_eval_runner.py` (or under `scripts/evaluation` if test infra expects) that:
    - Mocks `urllib.request.urlopen` (or the runner’s HTTP function) to return a canned AgentResponse with `interrupted=true` and `interrupt_payload.type=retrieval_review`.
    - Asserts the runner writes expected fields and preserves `rrf_*` metadata when present.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 9 | Blocked By: 6

  **References**:
  - Runner implementation: `scripts/evaluation/run_chat_flow_retrieval_rrf_eval.py`.

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_chat_flow_rrf_eval_runner.py -q`

  **QA Scenarios**:
  ```
  Scenario: eval runner test passes without docker
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_chat_flow_rrf_eval_runner.py -q
    Expected: test passes; no network access needed
    Evidence: .sisyphus/evidence/task-7-eval-runner-test.txt
  ```

  **Commit**: YES | Message: `test(eval): add mocked test for chat-flow rrf runner` | Files: `backend/tests/test_chat_flow_rrf_eval_runner.py`

- [x] 8. Docker Compose Verification: ES Backend Uses App-level RRF

  **What to do**:
  - Add a short how-to section in the plan’s evidence (executor writes it) verifying via docker compose:
    - `make dev-up`
    - Run eval runner against `http://localhost:8011`
    - Confirm no `rank.rrf` requests and no fallback log lines in API logs.
  - Provide a log-grep command for `make logs-api` output.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: 2, 3, 6

  **References**:
  - Compose: `docker-compose.yml` (api-dev uses ES service).
  - Make targets: `Makefile:76` (`dev-up`), `Makefile:101` (`logs-api`).

  **Acceptance Criteria**:
  - [ ] `make dev-up` and runner command from Task 6 completes successfully.

  **QA Scenarios**:
  ```
  Scenario: docker-compose smoke for RRF
    Tool: Bash
    Steps:
      1) make dev-up
      2) make logs-api (in parallel terminal) and run Task-6 runner
      3) grep logs for 'rank.rrf' and 'falling back to script_score'
    Expected: both strings absent; retrieval_review interrupts returned
    Evidence: .sisyphus/evidence/task-8-docker-smoke.txt
  ```

  **Commit**: NO | Message: n/a | Files: n/a

- [x] 9. Doc Sync (make “truth baseline” explicit)

  **What to do**:
  - Update `docs/2026-03-11-rrf-문제해결.md` (or add an addendum section) with the post-fix “현재 코드 기준 실제 상태” table showing stage1 is now app-level RRF.
  - Add pointers to the new test + runner paths.

  **Recommended Agent Profile**:
  - Category: `writing`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: 2, 3, 6

  **References**:
  - Source doc: `docs/2026-03-11-rrf-문제해결.md`.

  **Acceptance Criteria**:
  - [ ] The doc includes: fixed path summary, test file references, runner usage command.

  **QA Scenarios**:
  ```
  Scenario: documentation includes verification commands
    Tool: Bash
    Steps: (manual) open doc and verify commands exist
    Expected: doc contains exact commands for tests + runner
    Evidence: .sisyphus/evidence/task-9-doc-sync.txt
  ```

  **Commit**: YES | Message: `docs(retrieval): document stage1 app-level rrf fix` | Files: `docs/2026-03-11-rrf-문제해결.md`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Retrieval Behavior QA (docker compose) — unspecified-high
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Prefer atomic commits:
  - `fix(retrieval): stage1 app-level rrf fusion`
  - `fix(retrieval): tolerate null _score in es hit parsing`
  - `test(retrieval): prevent native rrf + add rrf metadata assertions`
  - `chore(eval): add chat-flow retrieval eval runner`

## Success Criteria
- Stage1 hybrid retrieval in ES backend never issues `rank.rrf` requests.
- No silent fallback logs; script_score only when explicitly selected.
- RRF metadata present on stage1 results; enables debugging of rank fusion.
- Chat-flow eval runner can compare before/after top-k stability in bilingual queries.
