# Before/After Regression Compare (Same Questions)

## TL;DR
> **Summary**: Re-run the same fixed query set against two git SHAs (before/after) under a synthetic ES namespace, collect retrieval + agent artifacts, compute deltas, and emit a single reproducible Markdown report with machine-verifiable evidence.
> **Deliverables**: retrieval metrics (before/after), agent flow/quality captures (before/after), delta tables + representative diffs, reproducible run manifest.
> **Effort**: Medium
> **Parallel**: NO (default; safer on shared hardware)
> **Critical Path**: Synthetic ES isolation → Ingest synthetic corpus → Run retrieval eval (before, after) → Run agent eval (before, after) → Generate report

## Context
### Original Request
- Re-ask previously used questions unchanged.
- Compare before vs after across: logs/traceability, consistent flow, answer quality.
- Write a report with quantitative deltas + qualitative examples.
- Papers doc set should include A/B/C writing approach + baseline construction process.

### Interview Summary
- Use repo-provided “previous questions” as the canonical query set:
  - Primary (fast): `.sisyphus/evidence/paper-b/task-10/queries_subset.jsonl` (48)
  - Secondary (full): `data/synth_benchmarks/stability_bench_v1/queries.jsonl` (240)
- Before/after SHAs are fixed:
  - Before: `73ca832`
  - After: `c04fa25`
- Reproducibility and safety must follow Paper B runbook isolation: `docs/papers/20_paper_b_stability/paper_b_synth_runbook.md`

### Metis Review (gaps addressed)
- Guardrails added: never touch non-`rag_synth_*` aliases; never use moving refs like `HEAD~1`; avoid evidence overwrite (per `run_paper_b_eval.py` fixed output filenames).
- Added explicit acceptance criteria and hard preflight checks.
- Added decision defaults (no blocking user questions) and captured confounders (index mapping drift, LLM nondeterminism).

## Work Objectives
### Core Objective
Produce a reproducible before/after regression comparison on identical queries with evidence artifacts + a Markdown report that can be re-run and audited.

### Deliverables
- `.sisyphus/evidence/regression_compare/<run_id>/manifest.json` (run metadata, SHAs, env, endpoints)
- Retrieval evidence (both SHAs, both query sets):
  - `.../{before,after}/retrieval/{subset_48,full_240}/{det_true,det_false}/results.jsonl`
  - `.../{before,after}/retrieval/{subset_48,full_240}/{det_true,det_false}/metrics.json`
- Agent evidence (both SHAs, subset_48 only by default):
  - `.../{before,after}/agent/subset_48/run.jsonl` (per-query AgentResponse summary)
  - `.../{before,after}/agent/subset_48/stream_events.ndjson` (SSE events)
- `.sisyphus/evidence/regression_compare/<run_id>/report.md` (tables + deltas + examples)

### Definition of Done (agent-verifiable)
- Evidence directory exists and contains before/after retrieval metrics for subset_48:
  - `test -f ".sisyphus/evidence/regression_compare/<run_id>/before/retrieval/subset_48/det_true/metrics.json"`
  - `test -f ".sisyphus/evidence/regression_compare/<run_id>/after/retrieval/subset_48/det_true/metrics.json"`
- Evidence directory contains agent run captures for subset_48:
  - `test -s ".sisyphus/evidence/regression_compare/<run_id>/before/agent/subset_48/run.jsonl"`
  - `test -s ".sisyphus/evidence/regression_compare/<run_id>/after/agent/subset_48/run.jsonl"`
- Report references exact SHAs and exact commands:
  - `python -c "import pathlib; t=pathlib.Path('.sisyphus/evidence/regression_compare/<run_id>/report.md').read_text(); assert '73ca832' in t and 'c04fa25' in t and 'run_paper_b_eval.py' in t"`

### Must Have
- Synthetic ES isolation enforced (prefix `rag_synth`, env `synth`) per `docs/papers/20_paper_b_stability/paper_b_synth_runbook.md`.
- Identical query set(s) used for both SHAs (no rewriting, no reordering).
- Evidence includes `effective_config_hash`, `run_id`, and trace IDs where available.

### Must NOT Have (guardrails)
- MUST NOT touch any `*_current` alias that is not `rag_synth_*_current`.
- MUST NOT use moving git refs (`HEAD`, `HEAD~1`) in commands, filenames, or report.
- MUST NOT overwrite evidence directories (unique `<run_id>` required).
- MUST NOT introduce new bespoke metrics unless explicitly requested; use existing keys from `scripts/paper_b/run_paper_b_eval.py`.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (Pytest for new utilities; reuse existing integration scripts)
- Primary truth artifacts: `metrics.json` and `results.jsonl` emitted by `scripts/paper_b/run_paper_b_eval.py`
- Agent flow truth artifacts: `stream_events.ndjson` derived from `/api/agent/run/stream` SSE
- Evidence convention reference: `.sisyphus/evidence/paper-b/*` and runbook `docs/papers/20_paper_b_stability/paper_b_synth_runbook.md`

## Execution Strategy
### Defaults Applied (no user input needed)
- Query set: run subset_48 for both retrieval+agent; additionally run full_240 for retrieval only.
- Agent request mode: `auto_parse=true`, `mode=verified`, `max_attempts=3`, `use_canonical_retrieval=true` (explicitly set by runner for traceability).
- LLM determinism: set `VLLM_TEMPERATURE=0.0` for both SHAs during agent evaluation.
- Run order: sequential (one API at a time) to avoid resource contention.

### Parallel Execution Waves
Wave 1: Safety + dataset + synthetic ES alias (foundation)
Wave 2: Before SHA runs (retrieval + agent)
Wave 3: After SHA runs (retrieval + agent)
Wave 4: Diff + report generation

### Dependency Matrix (high level)
- ES alias creation + ingestion blocks all eval runs.
- Worktrees must exist before starting each SHA’s API.
- Retrieval/agent eval artifacts block report generation.

## TODOs
> Implementation + Test = ONE task. Never separate.
> Every task produces evidence under `.sisyphus/evidence/regression_compare/<run_id>/...`.

- [x] 1. Create a new regression run workspace (run_id + manifest)

  **What to do**:
  - Implement `scripts/evaluation/regression_compare_manifest.py` (stdlib-only) with CLI shown in the command block below.
  - Define `<run_id>` as `YYYYMMDD_HHMMSS` and create evidence root at `.sisyphus/evidence/regression_compare/<run_id>/`.
  - Write `manifest.json` capturing: timestamps, `before_sha=73ca832`, `after_sha=c04fa25`, query file paths, API base URLs, ES host/prefix/env, and any non-secret env vars used (whitelist: `SEARCH_`, `ES_`, `RAG_`, `VLLM_`; redact `*_KEY`, `*_PASSWORD`, `*_TOKEN`).
  - Record SHA256 hashes + line counts for each query file to prove “same questions”.

  **Commands (decision-complete)**:
  ```bash
  # Run from repo root: /home/hskim/work/llm-agent-v2
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export RUN_ID="$(date +%Y%m%d_%H%M%S)"  # <-- this is <run_id>; reuse the same value for all tasks
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"
  export QUERY_SUBSET="$REPO_ROOT/.sisyphus/evidence/paper-b/task-10/queries_subset.jsonl"
  export QUERY_FULL="$REPO_ROOT/data/synth_benchmarks/stability_bench_v1/queries.jsonl"

  mkdir -p "$RUN_ROOT"

  python "$REPO_ROOT/scripts/evaluation/regression_compare_manifest.py" \
    --run-id "$RUN_ID" \
    --before-sha 73ca832 \
    --after-sha c04fa25 \
    --queries-subset "$QUERY_SUBSET" \
    --queries-full "$QUERY_FULL" \
    --es-host "http://localhost:8002" \
    --es-env synth \
    --es-index-prefix rag_synth \
    --out "$RUN_ROOT/manifest.json"
  ```
  **Must NOT do**: Do not dump full environment or secrets into manifest.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: small utility + file I/O
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [2,3,4,5,6,7,8,9] | Blocked By: []

  **References**:
  - Evidence patterns: `.sisyphus/evidence/paper-b/` — existing naming + structure

  **Acceptance Criteria**:
  - [ ] `test -f ".sisyphus/evidence/regression_compare/<run_id>/manifest.json"`

  **QA Scenarios**:
  ```
  Scenario: Manifest is created and redacted
    Tool: Bash
    Steps: run the manifest creation command/script
    Expected: manifest.json exists; contains before/after SHAs; does not contain API keys/tokens/passwords
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/manifest.json

  Scenario: Missing run_id
    Tool: Bash
    Steps: invoke without run_id (or empty)
    Expected: runner refuses to proceed with a clear error
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/logs/manifest_error.txt
  ```

  **Commit**: YES | Message: `chore(eval): add regression compare run manifest helper` | Files: [new helper script(s) if created]

- [x] 2. Enforce synthetic Elasticsearch isolation (preflight checks)

  **What to do**: Implement a preflight command sequence that verifies:
  1) ES host is reachable (default per `.env`: `SEARCH_ES_HOST=http://localhost:8002`).
  2) If the synthetic alias already exists, it is isolated: `rag_synth_synth_current` points only to `rag_synth_synth_v*`.
  3) Global scan of `*_current` does not show accidental crossover.

  **Commands (decision-complete)**:
  ```bash
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export RUN_ID="<run_id>"  # Use the RUN_ID produced in task 1
  export ES_HOST="http://localhost:8002"
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"
  mkdir -p "$RUN_ROOT/preflight"

  # Ensure ES is running (start only Elasticsearch; avoid api-dev)
  docker compose --env-file "$REPO_ROOT/.env" --env-file "$REPO_ROOT/.env.dev" up -d elasticsearch

  # Wait for ES to respond
  for i in $(seq 1 60); do
    if curl -sf "$ES_HOST/_cluster/health" > "$RUN_ROOT/preflight/es_health.json"; then break; fi
    sleep 1
  done

  # Evidence capture
  curl -s "$ES_HOST/_alias/rag_synth_synth_current" > "$RUN_ROOT/preflight/alias_rag_synth_synth_current.json" || true
  curl -s "$ES_HOST/_alias/*_current" > "$RUN_ROOT/preflight/alias_all_current.json"

  # Human-readable CLI list (also evidence)
  ES_ENV=synth ES_INDEX_PREFIX=rag_synth ES_HOST="$ES_HOST" \
    python -m backend.llm_infrastructure.elasticsearch.cli list \
    > "$RUN_ROOT/preflight/es_cli_list.txt"

  # Machine check: if alias exists, its keys must be rag_synth_synth_v*
  python -c "import json,pathlib,re; p=pathlib.Path('$RUN_ROOT/preflight/alias_rag_synth_synth_current.json'); obj=json.loads(p.read_text() or '{}'); bad=[k for k in obj.keys() if not re.match(r'^rag_synth_synth_v\\d+$', k)]; assert not bad, f'non-synth alias targets: {bad}'"
  ```

  **Must NOT do**: Must not touch or switch any alias not starting with `rag_synth_`.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: shell + safety assertions
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [4,5,6,7,8,9] | Blocked By: [1]

  **References**:
  - Runbook: `docs/papers/20_paper_b_stability/paper_b_synth_runbook.md`
  - Docker compose ports: `docker-compose.yml` (ES on `${ES_PORT}:9200`)

  **Acceptance Criteria**:
  - [ ] `test -f ".sisyphus/evidence/regression_compare/<run_id>/preflight/es_health.json"`
  - [ ] If `rag_synth_synth_current` exists, it targets only `rag_synth_synth_v*` (enforced by the Python check in the command block)

  **QA Scenarios**:
  ```
  Scenario: Synthetic alias is isolated
    Tool: Bash
    Steps: run preflight checks exactly as in runbook
    Expected: checks pass; evidence file written with raw responses
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/preflight/es_alias_check.json

  Scenario: Missing SEARCH_ES_ENV
    Tool: Bash
    Steps: unset SEARCH_ES_ENV/SEARCH_ES_INDEX_PREFIX and re-run preflight
    Expected: preflight fails fast with a clear message explaining required vars
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/preflight/es_alias_check_error.txt
  ```

  **Commit**: NO

- [x] 3. Create/switch synthetic index and ingest synthetic corpus (one-time)

  **What to do**:
  - Start docker ES (`elasticsearch` service) if not running.
  - Create and switch synthetic index `rag_synth_synth_current` → `rag_synth_synth_v1` using the CLI.
  - Ingest `data/synth_benchmarks/stability_bench_v1/corpus.jsonl` into the synthetic alias using `scripts/paper_b/ingest_synth_corpus.py`.
  - Save ingestion report into the regression evidence root (do not reuse Paper-B task reports).

  **Commands (decision-complete)**:
  ```bash
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export ES_HOST="http://localhost:8002"
  export RUN_ID="<run_id>"  # Use the RUN_ID produced in task 1
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"
  mkdir -p "$RUN_ROOT/ingest"

  # Start only Elasticsearch (avoid starting api-dev to prevent port conflicts)
  docker compose --env-file "$REPO_ROOT/.env" --env-file "$REPO_ROOT/.env.dev" up -d elasticsearch

  # Create/switch synthetic index
  DIMS="$(python -c 'from backend.config.settings import search_settings; print(search_settings.es_embedding_dims)')"
  ES_ENV=synth ES_INDEX_PREFIX=rag_synth ES_HOST="$ES_HOST" \
    python -m backend.llm_infrastructure.elasticsearch.cli create \
      --version 1 --dims "$DIMS" --switch-alias \
    > "$RUN_ROOT/ingest/es_create.log" 2>&1 \
    || ES_ENV=synth ES_INDEX_PREFIX=rag_synth ES_HOST="$ES_HOST" \
      python -m backend.llm_infrastructure.elasticsearch.cli switch --version 1 \
      > "$RUN_ROOT/ingest/es_switch.log" 2>&1

  # Verify synthetic alias now exists and is isolated
  curl -s "$ES_HOST/_alias/rag_synth_synth_current" > "$RUN_ROOT/ingest/alias_rag_synth_synth_current.json"
  python -c "import json,pathlib,re; p=pathlib.Path('$RUN_ROOT/ingest/alias_rag_synth_synth_current.json'); obj=json.loads(p.read_text() or '{}'); assert obj, 'missing rag_synth_synth_current'; bad=[k for k in obj.keys() if not re.match(r'^rag_synth_synth_v\\d+$', k)]; assert not bad, f'non-synth alias targets: {bad}'"

  # Ingest corpus into synthetic alias
  SEARCH_ES_HOST="$ES_HOST" SEARCH_ES_ENV=synth SEARCH_ES_INDEX_PREFIX=rag_synth \
    python "$REPO_ROOT/scripts/paper_b/ingest_synth_corpus.py" \
      --corpus "$REPO_ROOT/data/synth_benchmarks/stability_bench_v1/corpus.jsonl" \
      --report "$RUN_ROOT/ingest/ingest_summary.json" \
    > "$RUN_ROOT/ingest/ingest.log" 2>&1
  ```

  **Must NOT do**: Must not ingest into `rag_chunks_*_current`.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` — Reason: environment orchestration, safe scripting
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [4,5,6,7,8,9] | Blocked By: [2]

  **References**:
  - Dataset: `data/synth_benchmarks/stability_bench_v1/README.md`
  - Ingest script: `scripts/paper_b/ingest_synth_corpus.py`
  - ES CLI: `backend/llm_infrastructure/elasticsearch/cli.py` (invoked via `python -m ...` in runbook)

  **Acceptance Criteria**:
  - [ ] `test -f ".sisyphus/evidence/regression_compare/<run_id>/ingest/ingest_summary.json"`
  - [ ] Ingest report JSON includes `index` starting with `rag_synth_synth_` and `failures_count == 0`

  **QA Scenarios**:
  ```
  Scenario: Fresh ingest into synthetic alias
    Tool: Bash
    Steps: run CLI create/switch, then run ingest_synth_corpus.py with SEARCH_ES_* set to synth namespace
    Expected: report shows docs_processed=120 (or expected count), failures_count=0
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/ingest/ingest_summary.json

  Scenario: Accidental prod prefix
    Tool: Bash
    Steps: set SEARCH_ES_INDEX_PREFIX=rag_chunks and re-run ingest command
    Expected: task aborts before ingest with a safety error (explicit check)
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/ingest/safety_abort.txt
  ```

  **Commit**: NO

- [x] 4. Create two git worktrees pinned to before/after SHAs

  **What to do**: Use `git worktree` to create isolated working directories under `.sisyphus/worktrees/`:
  - `.sisyphus/worktrees/before-73ca832` at `73ca832`
  - `.sisyphus/worktrees/after-c04fa25` at `c04fa25`
  This avoids mutating the current branch or mixing dependencies.

  **Commands (decision-complete)**:
  ```bash
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export RUN_ID="<run_id>"  # Use the RUN_ID produced in task 1
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"
  mkdir -p "$REPO_ROOT/.sisyphus/worktrees" "$RUN_ROOT/git"

  git worktree add "$REPO_ROOT/.sisyphus/worktrees/before-73ca832" 73ca832
  git worktree add "$REPO_ROOT/.sisyphus/worktrees/after-c04fa25" c04fa25
  git worktree list > "$RUN_ROOT/git/worktree_list.txt"
  ```

  **Must NOT do**: Must not change the current working tree branch or discard local changes.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: bounded git operations
  - Skills: [`git-master`] — ensure safe worktree usage

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [5,6,7,8,9] | Blocked By: [1]

  **References**:
  - SHAs: `git log -5 --oneline` (shows `73ca832`, `c04fa25`)

  **Acceptance Criteria**:
  - [ ] `.sisyphus/worktrees/before-73ca832/backend/api/main.py` exists
  - [ ] `.sisyphus/worktrees/after-c04fa25/backend/api/main.py` exists

  **QA Scenarios**:
  ```
  Scenario: Worktrees created
    Tool: Bash
    Steps: git worktree add ... 73ca832; git worktree add ... c04fa25
    Expected: git worktree list shows both entries at correct SHAs
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/git/worktree_list.txt
  ```

  **Commit**: NO

- [x] 5. Run retrieval evaluation on the before SHA (subset_48 + full_240)

  **What to do**:
  - Start a local API from the before worktree on a dedicated port (e.g., `18111`) with env:
    - `SEARCH_ES_HOST=http://localhost:8002`, `SEARCH_ES_ENV=synth`, `SEARCH_ES_INDEX_PREFIX=rag_synth`
  - Run `scripts/paper_b/run_paper_b_eval.py` against that API for:
    - subset_48: `.sisyphus/evidence/paper-b/task-10/queries_subset.jsonl`
    - full_240: `data/synth_benchmarks/stability_bench_v1/queries.jsonl`
  - Run 2 modes for each query set:
    - `det_true` (`--deterministic true`)
    - `det_false` (`--deterministic false`)
  - Write outputs to the regression evidence root (unique out dirs per mode).

  **Commands (decision-complete)**:
  ```bash
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export RUN_ID="<run_id>"  # Use the RUN_ID produced in task 1
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"
  export BEFORE_ROOT="$REPO_ROOT/.sisyphus/worktrees/before-73ca832"
  export BEFORE_PORT=18111
  export ES_HOST="http://localhost:8002"
  export QUERY_SUBSET="$REPO_ROOT/.sisyphus/evidence/paper-b/task-10/queries_subset.jsonl"
  export QUERY_FULL="$REPO_ROOT/data/synth_benchmarks/stability_bench_v1/queries.jsonl"
  mkdir -p "$RUN_ROOT/before/api" "$RUN_ROOT/before/retrieval"

  # Start API (before)
  ( \
    cd "$BEFORE_ROOT" \
    && SEARCH_ES_HOST="$ES_HOST" SEARCH_ES_ENV=synth SEARCH_ES_INDEX_PREFIX=rag_synth \
       VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.10.100.45:8003}" VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-openai/gpt-oss-20b}" VLLM_TEMPERATURE=0.0 \
       RAG_PROMPT_SPEC_VERSION=v1 \
       python -m uvicorn backend.api.main:app --host 0.0.0.0 --port "$BEFORE_PORT" \
  ) > "$RUN_ROOT/before/api/uvicorn.log" 2>&1 & echo $! > "$RUN_ROOT/before/api/pid"

  # Wait for API health
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$BEFORE_PORT/health" > "$RUN_ROOT/before/api/health.json"; then break; fi
    sleep 1
  done

  # subset_48 det_true
  (cd "$BEFORE_ROOT" && python scripts/paper_b/run_paper_b_eval.py \
    --api-base-url "http://localhost:$BEFORE_PORT" \
    --queries "$QUERY_SUBSET" \
    --out-dir "$RUN_ROOT/before/retrieval/subset_48/det_true" \
    --repeats 10 --mode-name subset_48_det_true --deterministic true \
  ) > "$RUN_ROOT/before/retrieval/subset_48_det_true.log" 2>&1

  # subset_48 det_false
  (cd "$BEFORE_ROOT" && python scripts/paper_b/run_paper_b_eval.py \
    --api-base-url "http://localhost:$BEFORE_PORT" \
    --queries "$QUERY_SUBSET" \
    --out-dir "$RUN_ROOT/before/retrieval/subset_48/det_false" \
    --repeats 10 --mode-name subset_48_det_false --deterministic false \
  ) > "$RUN_ROOT/before/retrieval/subset_48_det_false.log" 2>&1

  # full_240 det_true
  (cd "$BEFORE_ROOT" && python scripts/paper_b/run_paper_b_eval.py \
    --api-base-url "http://localhost:$BEFORE_PORT" \
    --queries "$QUERY_FULL" \
    --out-dir "$RUN_ROOT/before/retrieval/full_240/det_true" \
    --repeats 10 --mode-name full_240_det_true --deterministic true \
  ) > "$RUN_ROOT/before/retrieval/full_240_det_true.log" 2>&1

  # full_240 det_false
  (cd "$BEFORE_ROOT" && python scripts/paper_b/run_paper_b_eval.py \
    --api-base-url "http://localhost:$BEFORE_PORT" \
    --queries "$QUERY_FULL" \
    --out-dir "$RUN_ROOT/before/retrieval/full_240/det_false" \
    --repeats 10 --mode-name full_240_det_false --deterministic false \
  ) > "$RUN_ROOT/before/retrieval/full_240_det_false.log" 2>&1

  # Stop API (before)
  kill "$(cat "$RUN_ROOT/before/api/pid")" || true
  sleep 2
  ```

  **Must NOT do**: Must not reuse `.sisyphus/evidence/paper-b/...` output dirs (avoid overwrite).

  **Recommended Agent Profile**:
  - Category: `unspecified-low` — Reason: process orchestration + long-running calls
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [8,9] | Blocked By: [2,3,4]

  **References**:
  - Eval harness: `scripts/paper_b/run_paper_b_eval.py`
  - Query sets: `.sisyphus/evidence/paper-b/task-10/queries_subset.jsonl`, `data/synth_benchmarks/stability_bench_v1/queries.jsonl`
  - API base default: `DEFAULT_API_BASE_URL = "http://localhost:8011"` (override with `--api-base-url`)

  **Acceptance Criteria**:
  - [ ] `test -f ".sisyphus/evidence/regression_compare/<run_id>/before/retrieval/subset_48/det_true/metrics.json"`
  - [ ] `test -f ".sisyphus/evidence/regression_compare/<run_id>/before/retrieval/full_240/det_true/metrics.json"`

  **QA Scenarios**:
  ```
  Scenario: subset_48 retrieval metrics generated (before)
    Tool: Bash
    Steps: start API on 18111; run run_paper_b_eval.py with --queries queries_subset.jsonl --out-dir .../subset_48/det_true
    Expected: metrics.json contains keys hit@10, MRR, RepeatJaccard@10, p95_latency_ms
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/before/retrieval/subset_48/det_true/metrics.json

  Scenario: API not running
    Tool: Bash
    Steps: run eval without starting API
    Expected: script fails with clear connection error; error captured
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/before/retrieval/subset_48/det_true/error.txt
  ```

  **Commit**: NO

- [x] 6. Run agent evaluation on the before SHA (subset_48)

  **What to do**:
  - Implement a new runner script (recommended location): `scripts/evaluation/run_agent_regression.py`.
  - Runner inputs:
    - `--api-base-url` (e.g., `http://localhost:18111`)
    - `--queries` (JSONL; support the same `prefix|{json}` parsing used in `run_paper_b_eval.py`)
    - `--out-dir` (write `run.jsonl` and `stream_events.ndjson`)
    - `--use-canonical-retrieval true`
    - `--max-attempts 3`, `--auto-parse true`, `--mode verified`
  - Runner outputs (fixed filenames inside `--out-dir`):
    - `run.jsonl` (1 row per query; include qid/group_id/query + judge + metadata + retrieved doc ids)
    - `stream_events.ndjson` (all SSE `data:` payloads as NDJSON; include `open`, `log`, `final`)
    - `errors.jsonl` (optional; 1 row per failed query/stream)
  - For each query:
    - Call `POST /api/agent/run` and store a compact per-row JSON (qid, group_id, query, judge summary, metadata, retrieved doc ids, trace info).
    - Call `POST /api/agent/run/stream` and store all `data:` payloads as NDJSON events, including `open`, `log`, and `final`.
  - Set `VLLM_TEMPERATURE=0.0` for the API process.

  **Commands (decision-complete)**:
  ```bash
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export RUN_ID="<run_id>"  # Use the RUN_ID produced in task 1
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"
  export BEFORE_ROOT="$REPO_ROOT/.sisyphus/worktrees/before-73ca832"
  export BEFORE_PORT=18111
  export ES_HOST="http://localhost:8002"
  export QUERY_SUBSET="$REPO_ROOT/.sisyphus/evidence/paper-b/task-10/queries_subset.jsonl"
  mkdir -p "$RUN_ROOT/before/agent/subset_48"

  # Start API (before) if not already running
  ( \
    cd "$BEFORE_ROOT" \
    && SEARCH_ES_HOST="$ES_HOST" SEARCH_ES_ENV=synth SEARCH_ES_INDEX_PREFIX=rag_synth \
       VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.10.100.45:8003}" VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-openai/gpt-oss-20b}" VLLM_TEMPERATURE=0.0 \
       RAG_PROMPT_SPEC_VERSION=v1 \
       python -m uvicorn backend.api.main:app --host 0.0.0.0 --port "$BEFORE_PORT" \
  ) > "$RUN_ROOT/before/api/uvicorn_agent.log" 2>&1 & echo $! > "$RUN_ROOT/before/api/pid_agent"
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$BEFORE_PORT/health" > "$RUN_ROOT/before/api/health_agent.json"; then break; fi
    sleep 1
  done

  # Run agent runner
  python "$REPO_ROOT/scripts/evaluation/run_agent_regression.py" \
    --api-base-url "http://localhost:$BEFORE_PORT" \
    --queries "$QUERY_SUBSET" \
    --out-dir "$RUN_ROOT/before/agent/subset_48" \
    --use-canonical-retrieval true \
    --auto-parse true \
    --mode verified \
    --max-attempts 3 \
    > "$RUN_ROOT/before/agent/subset_48/run.log" 2>&1

  kill "$(cat "$RUN_ROOT/before/api/pid_agent")" || true
  sleep 2
  ```

  **Must NOT do**: Must not store full expanded doc content in evidence by default (size + potential leakage); store doc IDs/snippets only.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: new runner + SSE parsing + robust I/O
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [8,9] | Blocked By: [5]

  **References**:
  - Request/response contract: `backend/api/routers/agent.py` (AgentRequest, SSE event shapes)
  - SSE parsing example: `tests/api/test_agent_response_metadata_contract.py` (`_parse_sse_final_result`)
  - Evidence naming convention: `.sisyphus/evidence/agent-stability/*`

  **Acceptance Criteria**:
  - [ ] `test -s ".sisyphus/evidence/regression_compare/<run_id>/before/agent/subset_48/run.jsonl"`
  - [ ] `python -c "import json, pathlib; p=pathlib.Path('.sisyphus/evidence/regression_compare/<run_id>/before/agent/subset_48/stream_events.ndjson'); rows=[json.loads(l) for l in p.open()]; assert rows[0]['type']=='open'; assert any(r.get('type')=='final' for r in rows)"`

  **QA Scenarios**:
  ```
  Scenario: Agent run and stream captured (before)
    Tool: Bash
    Steps: run run_agent_regression.py against /api/agent/run and /api/agent/run/stream
    Expected: run.jsonl has 48 rows; stream_events.ndjson includes open/log/final events
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/before/agent/subset_48/run.jsonl

  Scenario: SSE final missing
    Tool: Bash
    Steps: simulate truncated stream (drop connection)
    Expected: runner records an error row with qid and exception, continues remaining queries
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/before/agent/subset_48/errors.jsonl
  ```

  **Commit**: YES | Message: `feat(eval): add agent regression runner with SSE capture` | Files: [`scripts/evaluation/run_agent_regression.py`]

- [ ] 7. Run retrieval + agent evaluation on the after SHA (mirror tasks 5-6)

  **What to do**: Repeat task 5 and task 6 against the after worktree API (dedicated port, e.g., `18112`) and write outputs under `.../after/...`.

  **Commands (decision-complete)**:
  ```bash
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export RUN_ID="<run_id>"  # Use the RUN_ID produced in task 1
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"
  export AFTER_ROOT="$REPO_ROOT/.sisyphus/worktrees/after-c04fa25"
  export AFTER_PORT=18112
  export ES_HOST="http://localhost:8002"
  export QUERY_SUBSET="$REPO_ROOT/.sisyphus/evidence/paper-b/task-10/queries_subset.jsonl"
  export QUERY_FULL="$REPO_ROOT/data/synth_benchmarks/stability_bench_v1/queries.jsonl"
  mkdir -p "$RUN_ROOT/after/api" "$RUN_ROOT/after/retrieval" "$RUN_ROOT/after/agent/subset_48"

  # Start API (after)
  ( \
    cd "$AFTER_ROOT" \
    && SEARCH_ES_HOST="$ES_HOST" SEARCH_ES_ENV=synth SEARCH_ES_INDEX_PREFIX=rag_synth \
       VLLM_BASE_URL="${VLLM_BASE_URL:-http://10.10.100.45:8003}" VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-openai/gpt-oss-20b}" VLLM_TEMPERATURE=0.0 \
       RAG_PROMPT_SPEC_VERSION=v1 \
       python -m uvicorn backend.api.main:app --host 0.0.0.0 --port "$AFTER_PORT" \
  ) > "$RUN_ROOT/after/api/uvicorn.log" 2>&1 & echo $! > "$RUN_ROOT/after/api/pid"
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$AFTER_PORT/health" > "$RUN_ROOT/after/api/health.json"; then break; fi
    sleep 1
  done

  # Retrieval evals (after)
  (cd "$AFTER_ROOT" && python scripts/paper_b/run_paper_b_eval.py \
    --api-base-url "http://localhost:$AFTER_PORT" \
    --queries "$QUERY_SUBSET" \
    --out-dir "$RUN_ROOT/after/retrieval/subset_48/det_true" \
    --repeats 10 --mode-name subset_48_det_true --deterministic true \
  ) > "$RUN_ROOT/after/retrieval/subset_48_det_true.log" 2>&1

  (cd "$AFTER_ROOT" && python scripts/paper_b/run_paper_b_eval.py \
    --api-base-url "http://localhost:$AFTER_PORT" \
    --queries "$QUERY_SUBSET" \
    --out-dir "$RUN_ROOT/after/retrieval/subset_48/det_false" \
    --repeats 10 --mode-name subset_48_det_false --deterministic false \
  ) > "$RUN_ROOT/after/retrieval/subset_48_det_false.log" 2>&1

  (cd "$AFTER_ROOT" && python scripts/paper_b/run_paper_b_eval.py \
    --api-base-url "http://localhost:$AFTER_PORT" \
    --queries "$QUERY_FULL" \
    --out-dir "$RUN_ROOT/after/retrieval/full_240/det_true" \
    --repeats 10 --mode-name full_240_det_true --deterministic true \
  ) > "$RUN_ROOT/after/retrieval/full_240_det_true.log" 2>&1

  (cd "$AFTER_ROOT" && python scripts/paper_b/run_paper_b_eval.py \
    --api-base-url "http://localhost:$AFTER_PORT" \
    --queries "$QUERY_FULL" \
    --out-dir "$RUN_ROOT/after/retrieval/full_240/det_false" \
    --repeats 10 --mode-name full_240_det_false --deterministic false \
  ) > "$RUN_ROOT/after/retrieval/full_240_det_false.log" 2>&1

  # Agent eval (after)
  python "$REPO_ROOT/scripts/evaluation/run_agent_regression.py" \
    --api-base-url "http://localhost:$AFTER_PORT" \
    --queries "$QUERY_SUBSET" \
    --out-dir "$RUN_ROOT/after/agent/subset_48" \
    --use-canonical-retrieval true \
    --auto-parse true \
    --mode verified \
    --max-attempts 3 \
    > "$RUN_ROOT/after/agent/subset_48/run.log" 2>&1

  kill "$(cat "$RUN_ROOT/after/api/pid")" || true
  sleep 2
  ```

  **Must NOT do**: Must not reuse before outputs; must not change the synthetic ES alias between before and after runs.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` — Reason: repeatable run orchestration
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [8,9] | Blocked By: [2,3,4]

  **References**:
  - Same as tasks 5-6

  **Acceptance Criteria**:
  - [ ] `test -f ".sisyphus/evidence/regression_compare/<run_id>/after/retrieval/subset_48/det_true/metrics.json"`
  - [ ] `test -s ".sisyphus/evidence/regression_compare/<run_id>/after/agent/subset_48/run.jsonl"`

  **QA Scenarios**:
  ```
  Scenario: After evidence present
    Tool: Bash
    Steps: run evaluations for after SHA
    Expected: all after artifacts exist and are non-empty
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/after/retrieval/subset_48/det_true/metrics.json
  ```

  **Commit**: NO

- [ ] 8. Compute before/after deltas and regression gates (retrieval + agent)

  **What to do**:
  - Implement `scripts/evaluation/compare_before_after.py` that:
    - Loads `metrics.json` pairs and prints/writes a delta table.
    - Applies default regression gates (record pass/fail + reasons):
      - `hit@10` drop > 0.02 => FAIL
      - `MRR` drop > 0.02 => FAIL
      - `RepeatJaccard@10` drop > 0.02 => FAIL
      - `ParaphraseJaccard@10` drop > 0.02 => FAIL
    - For agent runs: computes distributions of `metadata.route`, `metadata.mq_used`, `metadata.attempts`, and `judge.faithful` (if present), plus doc-id Jaccard@k between before/after per qid.
  - Write outputs:
    - `.sisyphus/evidence/regression_compare/<run_id>/deltas/retrieval_subset_48.json`
    - `.sisyphus/evidence/regression_compare/<run_id>/deltas/agent_subset_48.json`

  **Commands (decision-complete)**:
  ```bash
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export RUN_ID="<run_id>"  # Use the RUN_ID produced in task 1
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"
  mkdir -p "$RUN_ROOT/deltas"

  python "$REPO_ROOT/scripts/evaluation/compare_before_after.py" \
    --run-root "$RUN_ROOT" \
    --out-dir "$RUN_ROOT/deltas" \
    > "$RUN_ROOT/deltas/compare.log" 2>&1
  ```

  **Must NOT do**: Must not require external services; pure file-based.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: careful metric alignment + robust comparisons
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: [9] | Blocked By: [5,6,7]

  **References**:
  - Retrieval metrics keys: `scripts/paper_b/run_paper_b_eval.py`
  - Agent metadata contract: `tests/api/test_agent_response_metadata_contract.py`

  **Acceptance Criteria**:
  - [ ] `test -f ".sisyphus/evidence/regression_compare/<run_id>/deltas/retrieval_subset_48.json"`
  - [ ] delta output includes both before and after values for `hit@10` and `RepeatJaccard@10`

  **QA Scenarios**:
  ```
  Scenario: Deltas computed
    Tool: Bash
    Steps: run compare_before_after.py pointing at evidence root
    Expected: delta JSON files exist; include pass/fail + thresholds
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/deltas/retrieval_subset_48.json

  Scenario: Missing metrics.json
    Tool: Bash
    Steps: remove/rename one metrics.json and rerun
    Expected: comparator errors clearly with missing path and stops
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/deltas/error.txt
  ```

  **Commit**: YES | Message: `feat(eval): add before/after comparator and regression gates` | Files: [`scripts/evaluation/compare_before_after.py`]

- [ ] 9. Generate the final Markdown report (single file)

  **What to do**:
  - Implement `scripts/evaluation/generate_regression_report.py` that consumes:
    - manifest.json
    - delta JSON outputs
    - selected example queries (top N largest changes for stability + a few representative tags)
  - Report sections (fixed):
    - Setup (endpoints, ES alias, SHAs, query set hashes)
    - Retrieval metrics tables (subset_48 and full_240; det_true and det_false)
    - Agent flow metrics (route/mq_used/attempts distributions)
    - Quality signals (judge faithful rate + issue summaries)
    - Representative diffs (qid, query, top-k doc ids before/after, answer excerpts)
    - Confounders & notes (index mapping drift, LLM nondeterminism)
  - Write: `.sisyphus/evidence/regression_compare/<run_id>/report.md`

  **Commands (decision-complete)**:
  ```bash
  export REPO_ROOT="/home/hskim/work/llm-agent-v2"
  export RUN_ID="<run_id>"  # Use the RUN_ID produced in task 1
  export RUN_ROOT="$REPO_ROOT/.sisyphus/evidence/regression_compare/$RUN_ID"

  python "$REPO_ROOT/scripts/evaluation/generate_regression_report.py" \
    --run-root "$RUN_ROOT" \
    --out "$RUN_ROOT/report.md" \
    > "$RUN_ROOT/report_gen.log" 2>&1
  ```

  **Must NOT do**: Must not embed large full documents; keep report small and link to evidence files.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: report composition + deterministic formatting
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: [] | Blocked By: [8]

  **References**:
  - Paper B assets style: `docs/papers/20_paper_b_stability/paper_b_assets/table_1_metrics.md`

  **Acceptance Criteria**:
  - [ ] `test -f ".sisyphus/evidence/regression_compare/<run_id>/report.md"`
  - [ ] report contains both SHAs and at least one table row for `hit@10` and `RepeatJaccard@10`

  **QA Scenarios**:
  ```
  Scenario: Report generated
    Tool: Bash
    Steps: run generate_regression_report.py
    Expected: report.md exists and includes setup + metrics + representative diffs
    Evidence: .sisyphus/evidence/regression_compare/<run_id>/report.md
  ```

  **Commit**: YES | Message: `feat(eval): generate before/after regression report markdown` | Files: [`scripts/evaluation/generate_regression_report.py`]

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Regression Report Sanity Review — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Prefer 1-3 atomic commits:
  - `feat(eval): add agent regression runner with SSE capture`
  - `feat(eval): add before/after comparator and regression gates`
  - `feat(eval): generate before/after regression report markdown`
- Do not commit generated evidence under `.sisyphus/evidence/`.

## Success Criteria
- Running the plan twice with the same SHAs and same synthetic ES alias yields identical retrieval `metrics.json` for `det_true` mode (within float serialization tolerance).
- The report is reproducible and contains direct links/paths to the underlying evidence files.
- Any regressions are clearly flagged with thresholds and concrete examples (qid-level diffs).
