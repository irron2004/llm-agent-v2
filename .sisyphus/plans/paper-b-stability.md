# Paper B: Stability-Aware Retrieval for Reliable Industrial RAG (with Synthetic Benchmark)

## TL;DR
> **Summary**: Produce a 6–8 page Paper B that formalizes Top-k stability as an operational reliability objective for industrial RAG, demonstrates stability controls (deterministic protocol + consensus retrieval), and ships a fully reproducible synthetic benchmark + runbook.
> **Deliverables**: (1) synthetic benchmark generator + dataset spec, (2) ingest + evaluation harness producing stability/effectiveness tradeoffs, (3) paper draft + figures/tables ready for submission.
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: Benchmark spec → generator+ingest → evaluation harness → experiments+figures → paper draft

## Context
### Original Request
- Review `docs/paper/` and propose a paper-writing strategy for the current project.

### Interview Summary
- Primary deliverable chosen: **Paper B (stability-aware retrieval)**.
- Reproducibility approach chosen: **Synthetic Benchmark** (publicly releasable).

### Repo Grounding (key references)
- Strategy spine / paper decomposition: `docs/paper/research_toc.md`
- PhD context scan: `docs/paper/Phd_paper_trend.md`
- Existing golden set seeds + MQ principles: `data/golden_set/retrieval_golden_set_v2.md`, `data/golden_set/queries_v2.jsonl`
- Existing evaluation harnesses:
  - `scripts/evaluation/retrieval_stability_audit.py`
  - `scripts/evaluation/search_sweep.py`
- Determinism & tie-breaking already exist in retrieval stack:
  - API: `backend/api/routers/retrieval.py`
  - Deterministic step resolution + MQ bypass: `backend/services/retrieval_pipeline.py`
  - Determinism/mq_strategy policy surface: `backend/services/retrieval_effective_config.py`
  - Pipeline orchestration: `backend/services/retrieval_pipeline.py`
  - Stable tie-break + multi-query merge: `backend/llm_infrastructure/llm/langgraph_agent.py` (`retrieve_node`)
  - Stable ES shard routing: `backend/llm_infrastructure/retrieval/engines/es_search.py` (`preference=rag-<md5(query)>`)
  - Prompt spec versioning via YAML filenames: `backend/llm_infrastructure/llm/prompt_loader.py` (loads `{name}_{version}.yaml`)

### Metis Review (gaps addressed)
- Define stability target + perturbation tiers explicitly (repeat query vs paraphrase vs restart vs reindex).
- Base evidence on *observed* retrieval behavior (avoid relying on harness-simulated instability).
- Add acceptance criteria for restart-tier and reindex-tier stability (even if reindex breaks determinism—report as driver).
- Add leakage controls for synthetic paraphrases (avoid copying unique doc tokens/IDs).

## Work Objectives
### Core Objective
- Deliver Paper B as a reliability-first retrieval paper with crisp novelty: **metrics + protocol + control levers + driver analysis + tradeoffs**, validated on a **fully reproducible synthetic benchmark**.

### Deliverables
- D1. Synthetic benchmark v1
  - Generator script + deterministic seed + manifest (corpus hash, query hash, generation params)
  - Corpus (small) + query sets with paraphrase groups + relevance labels
  - Leakage checks (paraphrase vs document overlap constraints)
- D2. Evaluation harness for Paper B
  - Run-to-run stability metrics (repeat runs)
  - Paraphrase stability metrics (equivalent queries)
  - Effectiveness metrics (hit@k, MRR)
  - Latency metrics (p95)
  - Reindex-tier experiment + report
- D3. Paper artifacts
  - 6–8 page draft (venue-agnostic short paper format)
  - Figures/tables (tradeoff curves + driver analysis)
  - Reproducibility checklist + runbook

### Definition of Done (verifiable)
- [ ] Synthetic benchmark generation is deterministic given a seed and produces identical hashes:
  - Command: `python scripts/paper_b/generate_synth_benchmark.py --seed 123 --out data/synth_benchmarks/stability_bench_v1`
  - Evidence: `data/synth_benchmarks/stability_bench_v1/manifest.json` contains corpus/query hashes.
- [ ] Synthetic corpus ingests into a dedicated ES index prefix without touching production indices.
- [ ] Paper-B evaluation script runs end-to-end and writes:
  - `hit@k`, `MRR`, `Stability@k (repeat)`, `Stability@k (paraphrase)`, `p95_latency_ms`
  - Evidence: `.sisyphus/evidence/paper-b/metrics.json` + `.sisyphus/evidence/paper-b/results.jsonl`
- [ ] Deterministic ordering guarantees are validated by existing tests:
  - Command: `pytest backend/tests/test_retrieve_node_deterministic.py`
  - Command: `pytest backend/tests/test_retrieval_pipeline.py`
- [ ] Reindex-tier experiment is executed and results are captured:
  - Evidence: `.sisyphus/evidence/paper-b/reindex_stability_report.md`
- [ ] Paper draft exists with filled Results + Discussion + Limitations sections and references the released benchmark.

  Notes (executor-facing):
  - Router-local endpoints are defined as `/retrieval/run` and `/retrieval/runs/{run_id}` in `backend/api/routers/retrieval.py`.
  - The FastAPI app mounts this router under `/api` in `backend/api/main.py`, so the deployed endpoints are:
    - `POST /api/retrieval/run`
    - `GET /api/retrieval/runs/{run_id}`
  - The run returns `run_id`; a full snapshot (including `search_queries`, `selected_doc_ids`) is available via `GET /api/retrieval/runs/{run_id}`.

### Must Have
- Stability tiers are explicit and tested: repeat, paraphrase, restart, reindex.
- Deterministic protocol is documented as a *guarantee tier* (not vague).
- Synthetic benchmark demonstrates the same failure modes as the real domain (abbreviations, ko/en mixing, code-like tokens, near-duplicate docs).

### Must NOT Have (guardrails)
- No “multi-agent novelty” claims.
- No absolute determinism claims beyond the defined tiers.
- No reliance on private corpora for the key quantitative claims.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (reuse existing scripts; add targeted pytest where it strengthens determinism guarantees)
- Evidence policy: every experiment writes artifacts under `.sisyphus/evidence/paper-b/`

## Execution Strategy
### Parallel Execution Waves
Wave 1 (Spec + scaffolding)
- Benchmark schema + leakage rules + runbook skeleton
- Paper outline + contribution statements

Wave 2 (Synthetic benchmark + ingest)
- Generator + corpus/query export
- ES index isolation + ingest script

Wave 3 (Evaluation harness + stability tiers)
- Repeat + paraphrase stability metrics
- Restart-tier + reindex-tier experiments
- Baselines/ablations wired to existing retrieval API knobs

Wave 4 (Results + manuscript)
- Produce tables/plots and write Results/Discussion/Limitations

### Dependency Matrix (high level)
- Wave 1 blocks Wave 2/3 (schema defines what to generate + measure)
- Wave 2 blocks Wave 3 (must have indexed corpus)
- Wave 3 blocks Wave 4 (must have numbers + figures)

## TODOs

- [x] 1. Freeze Paper B scope, claims, and stability tiers

  **What to do**:
  - Define EXACT stability targets (FIXED FOR THIS PAPER; no further discussion):
    - `RepeatJaccard@10`: average pairwise Jaccard between Top-10 doc_id *sets* across N repeats of the same query.
    - `RepeatExactMatch@10`: fraction of repeat-pairs whose ordered Top-10 doc_id lists match exactly.
    - `ParaphraseJaccard@10`: average pairwise Jaccard between Top-10 doc_id sets across paraphrases within a group.
    - `ParaphraseExactMatch@10`: fraction of paraphrase-pairs whose ordered Top-10 doc_id lists match exactly.
    - Effectiveness: `hit@5`, `hit@10`, `MRR` against `expected_doc_ids`.
    - Latency: `p95_latency_ms` from per-request timings.
  - Fix evaluation constants:
    - `k=10`, `hit_k={5,10}`, `N_repeats=10` for repeat stability, `paraphrases_per_group=4`.
  - Define perturbation tiers to be measured:
    - T1 Repeat (same query, same service, repeated runs)
    - T2 Paraphrase (equivalent queries)
    - T3 Restart (service restart between runs)
    - T4 Reindex (fresh index build from same corpus snapshot)
  - FIXED determinism guarantee statements (use verbatim in spec + paper):
    - T1 Repeat: With `deterministic=true`, and with a fixed ES alias target (no reindex/switch), the ordered Top-10 `doc_id` list MUST be identical across repeats.
    - T2 Paraphrase: No determinism guarantee; stability is a measured property across semantically equivalent queries.
    - T3 Restart: With `deterministic=true` and a fixed ES alias target, the ordered Top-10 `doc_id` list MUST be identical across a backend restart.
    - T4 Reindex: No determinism guarantee; report observed delta and treat as instability driver.

  **Must NOT do**:
  - Do not introduce new tasks (numeric faithfulness, hierarchy constraints) into Paper B.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: locks narrative + definitions
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2-12 | Blocked By: none

  **References**:
  - Strategy framing: `docs/paper/research_toc.md`
  - Existing stability harness: `scripts/evaluation/retrieval_stability_audit.py`

  **Acceptance Criteria**:
  - [ ] A single-page spec exists at `docs/paper/paper_b_stability_spec.md` defining: metrics, tiers, and guarantee text.

  **QA Scenarios**:
  ```
  Scenario: Stability tier checklist is unambiguous
    Tool: Bash
    Steps: open and grep for T1..T4 definitions; verify each tier has (metric, procedure, guarantee)
    Expected: all tiers have complete fields; no "TBD" remains
    Evidence: .sisyphus/evidence/paper-b/task-1-spec-check.txt

  Scenario: Scope creep guardrails are explicit
    Tool: Bash
    Steps: grep paper_b_stability_spec.md for "Must NOT Have" section
    Expected: contains at least 3 explicit exclusions
    Evidence: .sisyphus/evidence/paper-b/task-1-guardrails-check.txt
  ```

- [x] 2. Design synthetic benchmark v1 schema (corpus + queries + labels)

  **What to do**:
  - Create a benchmark spec that is *publicly releasable* and small.
  - Corpus schema (JSONL): `doc_id`, `doc_type`, `device_name`, `equip_id` (optional), `chapter` (optional), `content`, `tags`.
  - Query schema (JSONL): `qid`, `group_id`, `canonical_query`, `query`, `paraphrase_level`, `expected_doc_ids[]`, `tags[]`.
  - Require paraphrase groups (3–5 queries per group).
  - Define leakage checks:
    - forbid doc_ids inside queries
    - cap character-level overlap with the gold passage
    - cap n-gram overlap between queries and any document content
  - Fix dataset sizes (decision-complete):
    - 60 groups × 4 queries/group = 240 queries
    - 120 documents, including 30 near-duplicate pairs designed to create tie/near-tie regimes

  **Must NOT do**:
  - Do not copy any real internal identifiers, tool names, or doc snippets.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: dataset/benchmark design needs rigor
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 3-6 | Blocked By: 1

  **References**:
  - Failure modes to preserve: `data/golden_set/retrieval_golden_set_v2.md`
  - ES stability routing: `backend/llm_infrastructure/retrieval/engines/es_search.py`

  **Acceptance Criteria**:
  - [ ] Spec file exists at `docs/paper/synth_benchmark_stability_v1.md` with fixed schema + sizes + leakage rules.

  **QA Scenarios**:
  ```
  Scenario: Benchmark schema is mechanically checkable
    Tool: Bash
    Steps: grep for field lists and required constraints in synth_benchmark_stability_v1.md
    Expected: schema includes corpus JSONL and query JSONL definitions; sizes fixed
    Evidence: .sisyphus/evidence/paper-b/task-2-schema-check.txt

  Scenario: Leakage rules are explicit
    Tool: Bash
    Steps: grep for "Leakage" section and thresholds
    Expected: contains at least 3 rules with numeric thresholds (use exactly these thresholds):
      - Rule L1: queries MUST NOT contain any `doc_id` tokens (pattern: `DOC_` or `SYNTH_`)
      - Rule L2: longest common substring(query, gold_doc_content) <= 40 characters
      - Rule L3: 5-gram Jaccard(query, gold_doc_content) <= 0.35
    Evidence: .sisyphus/evidence/paper-b/task-2-leakage-check.txt
  ```

- [x] 3. Implement synthetic benchmark generator (deterministic)

  **What to do**:
  - Add `scripts/paper_b/generate_synth_benchmark.py` that:
    - takes `--seed`, `--out`
    - generates corpus docs + queries + paraphrase groups
    - writes `manifest.json` with hashes (sha256) of each output file
    - runs leakage checks and fails fast if violated
    - supports a `--selftest` mode that intentionally injects a leakage violation and asserts the generator fails (for CI-style verification)
  - Encode industrial-like stressors:
    - abbreviation tokens (PCW/MFC/APC/ESC/OES/EPD/TM/LL/etc.) without expanding them
    - mixed ko/en tokens
    - error-code-like strings (e.g., 7Ab, 252)
    - near-duplicate documents differing by one critical token

  **Must NOT do**:
  - Do not pull in any real documents from `data/ingestions/` or production ES.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: careful generator + leakage checks
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 4-6 | Blocked By: 2

  **References**:
  - Hashing approach: use `hashlib.sha256` (avoid Python `hash()` randomness)
  - Leakage/tie drivers rationale: `docs/paper/research_toc.md`

  **Acceptance Criteria**:
  - [ ] Running the generator twice with the same seed yields identical `manifest.json` hashes.

  **QA Scenarios**:
  ```
  Scenario: Generator determinism
    Tool: Bash
    Steps: run generator twice with same seed into two dirs; compare manifest.json
    Expected: manifests identical; corpus/query files identical
    Evidence: .sisyphus/evidence/paper-b/task-3-generator-determinism.txt

  Scenario: Leakage checks catch violations
    Tool: Bash
    Steps: run `python scripts/paper_b/generate_synth_benchmark.py --selftest`
    Expected: generator exits non-zero and reports leakage rule violated
    Evidence: .sisyphus/evidence/paper-b/task-3-leakage-fail.txt
  ```

- [x] 4. Isolate ES indices for synthetic benchmark (index prefix + runbook)

  **What to do**:
  - Define a dedicated index prefix + env for synthetic benchmark and ALWAYS set both backend + CLI env vars:
    - Backend (SearchSettings): `SEARCH_ES_ENV=synth`, `SEARCH_ES_INDEX_PREFIX=rag_synth`
    - ES CLI: `ES_ENV=synth`, `ES_INDEX_PREFIX=rag_synth`
  - Use `backend/llm_infrastructure/elasticsearch/cli.py` to create/switch the synthetic alias:
    - Determine dims from repo settings (decision-complete):
      - `python -c "from backend.config.settings import search_settings; print(search_settings.es_embedding_dims)"`
    - Create and switch:
      - `ES_ENV=synth ES_INDEX_PREFIX=rag_synth python -m backend.llm_infrastructure.elasticsearch.cli create --version 1 --dims $(python -c "from backend.config.settings import search_settings; print(search_settings.es_embedding_dims)") --switch-alias`
  - Document exact env vars and commands in `docs/paper/paper_b_synth_runbook.md`.

  **Must NOT do**:
  - Do not reuse or overwrite `rag_chunks_*_current` used by real data.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: reproducibility + isolation correctness
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 5-6 | Blocked By: 3

  **References**:
  - Index naming: `backend/llm_infrastructure/elasticsearch/manager.py`
  - Settings prefix: `backend/config/settings.py` (SearchSettings env_prefix=`SEARCH_`)

  **Acceptance Criteria**:
  - [ ] Runbook demonstrates creating and switching alias for synthetic index without affecting default index.

  **QA Scenarios**:
  ```
  Scenario: Index isolation
    Tool: Bash
    Steps: list indices and alias targets before/after; confirm synthetic alias uses rag_synth_* only
    Expected: production alias unchanged
    Evidence: .sisyphus/evidence/paper-b/task-4-index-isolation.txt
  ```

- [x] 5. Ingest synthetic corpus into ES using existing ingest service

  **What to do**:
  - Add `scripts/paper_b/ingest_synth_corpus.py` that:
    - loads corpus JSONL
    - builds `Section` objects and calls `EsIngestService.ingest_sections(...)`
    - writes a summary report (docs ingested, chunks, failures)
  - Keep ingestion deterministic (ordering stable; refresh index at end).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: correct ingestion and metadata
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6-10 | Blocked By: 4

  **References**:
  - Ingest API: `backend/services/es_ingest_service.py` (`ingest_sections`)
  - ES mapping meta: `backend/llm_infrastructure/elasticsearch/mappings.py`

  **Acceptance Criteria**:
  - [ ] After ingestion, a smoke query returns hits via `/api/search` and `/api/retrieval/run`.

  **QA Scenarios**:
  ```
  Scenario: Ingest + smoke retrieval
    Tool: Bash
    Steps: ingest synthetic corpus; call /api/retrieval/run with a known query
    Expected: returns non-empty docs; metadata includes doc_type/device_name
    Evidence: .sisyphus/evidence/paper-b/task-5-ingest-smoke.json
  ```

- [x] 6. Build Paper B evaluation runner (repeat + paraphrase stability)

  **What to do**:
  - Implement `scripts/paper_b/run_paper_b_eval.py` that:
    - reads synthetic query JSONL
    - calls `POST /api/retrieval/run` with controlled knobs (pin exact request schema below)
    - computes:
      - hit@k + MRR against `expected_doc_ids`
      - repeat stability (T1): run-to-run Jaccard@10 + ExactMatch@10 over `N_repeats=10`
      - paraphrase stability (T2): group-wise Jaccard@k across paraphrases
      - latency p95
    - writes `results.jsonl` + `metrics.json`
  - Ensure evidence is based on observed API outputs.

  Pin these exact request fields (schema from `backend/api/routers/retrieval.py`):
  - `query`: string (required)
  - `steps`: list[string] | null
  - `debug`: bool (default false)
  - `deterministic`: bool (default false)
  - `final_top_k`: int | null
  - `rerank_enabled`: bool | null
  - `auto_parse`: bool | null
  - `skip_mq`: bool | null
  - `device_names`: list[string] | null
  - `doc_types`: list[string] | null
  - `doc_types_strict`: bool | null
  - `equip_ids`: list[string] | null

  Decision-complete payloads for this paper:
  - Deterministic protocol run (T1/T3 measurement baseline):
    - `{"query": q, "steps": ["retrieve"], "debug": false, "deterministic": true, "auto_parse": false, "rerank_enabled": false}`
  - Non-deterministic baseline run (Paper-B baseline; still rerank OFF):
    - `{"query": q, "steps": ["retrieve"], "debug": false, "deterministic": false, "auto_parse": false, "rerank_enabled": false}`
  - Stability-aware MQ run (v2 prompt spec; rerank OFF):
    - `{"query": q, "steps": ["retrieve"], "debug": false, "deterministic": false, "auto_parse": false, "rerank_enabled": false}` plus env `RAG_PROMPT_SPEC_VERSION=v2`

  Required result capture per call:
  - From `POST /api/retrieval/run` response:
    - `run_id`, `effective_config_hash`, `docs[].doc_id`, `trace.trace_id`, `warnings[]`
  - (Optional but recommended for debugging) also fetch snapshot:
    - `GET /api/retrieval/runs/{run_id}` and record `search_queries`, `executed_steps`, `selected_doc_ids`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: evaluation rigor + metrics
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 7-10 | Blocked By: 5

  **References**:
  - Retrieval endpoint: `backend/api/routers/retrieval.py` (`/api/retrieval/run`)
  - Stable tie-break and multi-query merge: `backend/llm_infrastructure/llm/langgraph_agent.py` (`retrieve_node`)

  **Acceptance Criteria**:
  - [ ] Running the eval produces `.sisyphus/evidence/paper-b/metrics.json` with non-empty metrics.

  **QA Scenarios**:
  ```
  Scenario: End-to-end metrics generation
    Tool: Bash
    Steps: run eval against synthetic benchmark
    Expected: metrics.json includes hit@5, mrr, repeat_stability@10, paraphrase_stability@10, p95_latency_ms
    Evidence: .sisyphus/evidence/paper-b/task-6-metrics.json
  ```

- [x] 7. Implement stability controls for Paper B methods (deterministic protocol + stability-aware MQ)

  **What to do**:
  - Deterministic protocol:
    - Use existing `deterministic=true` behavior (in `backend/services/retrieval_pipeline.py`) which:
      - removes MQ steps (`mq`, `st_gate`, `st_mq`) from execution
      - forces a stable single-item `search_queries` list at `retrieve` step
    - Document guarantees using tier language from `docs/paper/paper_b_stability_spec.md`.
  - Stability-aware MQ (new):
    - Create prompt spec version `v2` YAMLs under `backend/llm_infrastructure/llm/prompts/`.
    - REQUIRED new files (decision-complete names; loader expects `{name}_{version}.yaml`):
      - `router_v2.yaml`
      - `setup_mq_v2.yaml`
      - `ts_mq_v2.yaml`
      - `general_mq_v2.yaml`
      - `st_gate_v2.yaml`
      - `st_mq_v2.yaml`
      - `setup_ans_v2.yaml`
      - `ts_ans_v2.yaml`
      - `general_ans_v2.yaml`
      - optional but recommended for parity: `translate_v2.yaml`, `auto_parse_v2.yaml`
    - v2 MQ requirements (MUST be enforced by prompt text):
      - Abbreviation/error-code tokens MUST be preserved verbatim (no expansions).
      - The original user query MUST appear as the first element of the MQ list.
      - Queries MUST NOT contain prompt labels ("setup_mq:", "queries:", etc.)—align with `_is_garbage_query` rules in `backend/llm_infrastructure/llm/langgraph_agent.py`.
      - Align with MQ principles in `data/golden_set/retrieval_golden_set_v2.md`.
  - Wire prompt spec version selection (experiment-only, default stays v1):
    - Add `RAG_PROMPT_SPEC_VERSION` setting (default `v1`) to `backend/config/settings.py` (RAGSettings).
    - Update `backend/api/dependencies.py` `get_prompt_spec_cached()` to call `load_prompt_spec(version=rag_settings.prompt_spec_version)`.
    - Update `backend/services/agents/langgraph_rag_agent.py` default `load_prompt_spec()` call to pass the same version when `prompt_spec` is not provided.

  **Must NOT do**:
  - Do not change production defaults unless explicitly required; keep changes gated behind prompt version.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: careful interaction of prompts, retrieval pipeline, and determinism
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 8-10 | Blocked By: 6

  **References**:
  - Current MQ prompt encourages abbreviation expansion (to be fixed): `backend/llm_infrastructure/llm/prompts/general_mq_v1.yaml`
  - Translation rule already preserves terms: `backend/llm_infrastructure/llm/prompts/translate_v1.yaml`
  - Prompt file resolver contract: `backend/llm_infrastructure/llm/prompt_loader.py` (`{name}_{version}.yaml`)
  - PromptSpec load function: `backend/llm_infrastructure/llm/langgraph_agent.py` (`load_prompt_spec(version=...)`)

  **Acceptance Criteria**:
  - [ ] Under v2 prompts, abbreviation tokens (PCW/MFC/APC/ESC/OES/EPD/TM/LL, error codes) are preserved verbatim in `search_queries`.
  - [ ] With `RAG_PROMPT_SPEC_VERSION` unset, app loads v1 prompt spec (no behavior change).
  - [ ] With `RAG_PROMPT_SPEC_VERSION=v2`, app loads v2 prompt spec and runs successfully.

  **QA Scenarios**:
  ```
  Scenario: Abbreviation preservation
    Tool: Bash
    Steps: run /api/retrieval/run debug=true on a synthetic abbreviation-heavy query; inspect steps.st_mq.search_queries
    Expected: contains original abbreviations unchanged; no expansions like "Process Cooling Water" unless explicitly present in input
    Evidence: .sisyphus/evidence/paper-b/task-7-abbrev-preservation.json

  Scenario: Prompt spec gating works (v1 default, v2 opt-in)
    Tool: Bash
    Steps:
      1) Start backend with no `RAG_PROMPT_SPEC_VERSION` and call /api/retrieval/run once; record effective_config
      2) Start backend with `RAG_PROMPT_SPEC_VERSION=v2` and call /api/retrieval/run once; record effective_config
    Expected: both runs succeed; prompt spec version change only occurs when env is set; v1 remains default
    Evidence: .sisyphus/evidence/paper-b/task-7-prompt-version-gating.txt
  ```

- [x] 8. Add restart-tier (T3) stability experiment

  **What to do**:
  - Execute Paper B eval in two phases with a service restart in between.
  - Compare deterministic-mode outputs (hash of top-k doc_id list per query).

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: operational reproducibility test
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 10-12 | Blocked By: 6

  **Acceptance Criteria**:
  - [ ] Deterministic mode has identical top-k hashes across restart for T1 repeat runs.

  **QA Scenarios**:
  ```
  Scenario: Restart-tier determinism
    Tool: Bash
    Steps:
      1) run eval phase A
      2) restart backend (default dev stack): `docker compose --profile dev restart api-dev`
      3) run eval phase B
      4) diff top-k hash files
    Expected: no diffs in deterministic mode; nondeterministic mode diffs allowed and reported
    Evidence: .sisyphus/evidence/paper-b/task-8-restart-diff.txt
  ```

- [x] 9. Add reindex-tier (T4) stability experiment

  **What to do**:
  - Create two indices from the same synthetic corpus snapshot (v1 and v2) using the ES CLI.
  - Run deterministic evaluation on both and quantify stability delta.
  - If not perfectly stable, treat as a driver analysis result and document mitigation options.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: metis-flagged publication-critical evidence
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 10-12 | Blocked By: 5

  **Acceptance Criteria**:
  - [ ] `reindex_stability_report.md` exists with: delta metrics + interpretation + mitigation.

  **QA Scenarios**:
  ```
  Scenario: Reindex-tier measurement exists
    Tool: Bash
    Steps: build index v1 and v2; run eval; generate report
    Expected: report contains numeric deltas for hit@k and stability@k
    Evidence: .sisyphus/evidence/paper-b/task-9-reindex-report.md
  ```

- [x] 10. Run full ablation matrix and generate paper-ready figures/tables

  **What to do**:
  - Run evaluation for these configurations (fixed list):
    - Baseline: v1 prompts, deterministic=false
    - Deterministic protocol: deterministic=true
    - Stability-aware MQ: v2 prompts, deterministic=false
    - Rerank is OFF for Paper B (keep confounds out of scope): `rerank_enabled=false`
  - Produce:
    - `Table 1`: hit@k, MRR, stability@10 (repeat & paraphrase), p95 latency
    - `Figure 1`: stability vs recall tradeoff
    - `Figure 2`: driver breakdown (abbrev/codes/near-dup)

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: experiment discipline + artifact generation
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: 11-12 | Blocked By: 7-9

  **Acceptance Criteria**:
  - [ ] Artifacts exist under `docs/paper/paper_b_assets/` and are referenced in the draft.

- [x] 11. Write Paper B draft (6–8 pages) with full Results/Limitations

  **What to do**:
  - Create `docs/paper/paper_b_stability.md` with:
    - Abstract, Intro, Metrics/Protocol, Methods, Experimental Setup, Results, Driver Analysis, Limitations, Reproducibility
  - Ensure Limitations explicitly cover what determinism does NOT guarantee (tiers).

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: coherent manuscript writing
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 12 | Blocked By: 10

  **Acceptance Criteria**:
  - [ ] Draft is complete enough that a reviewer could reproduce the benchmark and rerun the metrics.

- [x] 12. Package synthetic benchmark for release

  **What to do**:
  - Add `LICENSE` and README under `data/synth_benchmarks/stability_bench_v1/`.
  - Provide a single-command run path in the README:
    - generate → create index → ingest → run eval → produce figures

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: release-quality documentation
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: none | Blocked By: 10-11

  **Acceptance Criteria**:
  - [ ] A clean runbook exists and references only synthetic assets.

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Reproducibility Run-Through — deep
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: Synthetic benchmark spec + generator scaffolding
- Commit 2: ES isolation + ingest
- Commit 3: Evaluation harness + stability tier experiments
- Commit 4: Prompt v2 stability-aware MQ + ablation wiring
- Commit 5: Paper draft + assets + runbook

## Success Criteria
- Paper B draft + synthetic benchmark enable an external reader to rerun the reported stability/effectiveness results on the released synthetic corpus.
