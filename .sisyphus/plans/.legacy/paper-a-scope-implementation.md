# Paper A: Hierarchy-aware Scope Routing (Shared/Family/ScopeLevel) + Retrieval-only Evaluation

## TL;DR
> **Summary**: Implement Paper A's scope routing policy artifacts (Shared + Family + ScopeLevel) and a retrieval-only evaluation harness that produces reproducible contamination/recall results from the current ES index alias, using a versioned corpus manifest derived from `data/chunk_v3_normalize_table.md` with snapshot-aware drift checks for continuous operations.
> **Deliverables**:
> - `scripts/paper_a/` CLIs: corpus join + shared/family builders + eval-set builders + evaluator + ES backfill helpers
> - `backend/llm_infrastructure/retrieval/filters/scope_filter.py` (scope-level-aware ES bool filter builder)
> - Unit tests for normalization + filter DSL + family determinism
> - Reproducible run artifacts in `.sisyphus/evidence/paper-a/*` (per-query + summaries + bootstrap CIs + run manifest)
> - Drift artifacts for continuous runs (`corpus_snapshot.json`, `policy_snapshot.json`, optional `drift_report.json`)
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: Preflight (ES + alias) -> corpus doc_meta+whitelist -> shared+doc_scope -> family_map -> eval_sets -> evaluator+bootstrap -> (optional) ES backfill + router

## Context
### Original Request
- Plan Paper A work based on `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`.
- Incorporate/improve the user's first plan: `.omc/plans/paper-a-scope-implementation.md`.

### Interview Summary
- Primary experiment path is retrieval-only (no generation), per `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` Section 12.
- Canonical corpus stats source-of-truth is `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md` (baseline snapshot reference only; operational acceptance must not hardcode historical counts).
- Shared policy is topic-based with fixed threshold `T(shared)=3` and doc-type restriction (SOP + TS only; exclude setup manual).
- Family construction is deterministic (weighted-Jaccard >= tau=0.2 + connected components; no Louvain).
- Scope-level-aware filter must be an OR over branches (shared OR device-in-scope OR equip-in-scope with fallback), because current ES filtering ANDs `device_names` and `equip_ids`.
- Router/Matryoshka is OPTIONAL (Wave 4) with a required deterministic fallback (no "TODO: heuristic").

### Metis Review (gaps addressed)
- Locked family clustering to deterministic connected components (no optional Louvain).
- Added explicit canonicalization contracts for `device_name`, `doc_type`, and `doc_id` (and required failure reports when joins fail).
- Defined evaluation granularity: metrics computed on top-k unique `doc_id` (doc-level), not raw chunks.
- Made scope filter DSL decision-complete (exact ES bool structure) and added guardrails to prevent accidentally reusing the too-strict AND behavior.
- Tightened reproducibility artifacts to match `docs/papers/10_common_protocol/paper_common_protocol.md`.
- Moved outcome targets (e.g., “Adj_Cont@5 drops 15%”) out of acceptance criteria; acceptance is correctness + artifact completeness.

## Work Objectives
### Core Objective
- Implement the Paper A policy artifacts (Shared/Family/ScopeLevel) and a reproducible retrieval-only evaluator that can run B0-B4 and P1-P4 on Explicit/Masked/Ambiguous subsets.

### Deliverables
- Data artifacts (generated, not manually edited):
  - Corpus join/meta: resolved ES `doc_id` list for the current manifest-derived corpus + `doc_meta` JSONL (topic, device, doc_type)
  - `shared_topics.json` + `doc_scope.jsonl` (`is_shared`, `scope_level`)
  - `family_map.json` (device -> family_id + families)
  - Eval sets: `explicit.jsonl`, `masked.jsonl`, `ambiguous.jsonl` (Paper A schema)
- Evaluation harness outputs (per run):
  - `per_query.csv`, `summary_all.csv`, `summary_by_split.csv`, `summary_by_doc_type.csv`
  - `bootstrap_ci.json` (+ optional `mcnemar.json`)
  - `run_manifest.json` capturing config, seeds, alias->index resolution, git sha
- Code changes:
  - New scope filter builder `backend/llm_infrastructure/retrieval/filters/scope_filter.py`
  - New Paper A scripts under `scripts/paper_a/`
  - Unit tests under `backend/tests/`

### Definition of Done (agent-executable)
- [ ] Preflight script can resolve ES alias and corpus doc_ids (no missing docs):
- `python scripts/paper_a/build_corpus_meta.py --index rag_chunks_dev_current --normalize-table data/chunk_v3_normalize_table.md --out-dir .sisyphus/evidence/paper-a/corpus`
- [ ] Eval sets are built and schema-validated:
  - `python scripts/paper_a/build_eval_sets.py --sop-csv "data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv" --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --out-dir .sisyphus/evidence/paper-a/eval_sets`
- [ ] Evaluator runs B0-B4 + P1 on Explicit and produces all required outputs:
  - `python scripts/paper_a/evaluate_paper_a.py --eval-set .sisyphus/evidence/paper-a/eval_sets/explicit.jsonl --systems B0,B1,B2,B3,B4,P1 --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --out-dir .sisyphus/evidence/paper-a/runs/run_explicit_001`
- [ ] Unit tests for scope filter DSL + normalization + family determinism pass:
  - `python -m pytest backend/tests -q`

### Must Have
- Retrieval-only evaluation (no LLM calls) and reproducible run manifests.
- Doc-level contamination metrics: Raw/Adjusted/Shared Cont@k + CE@k per `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` Section 4.
- Scope-level-aware filtering semantics implemented as ES bool.should branches.
- Deterministic family graph clustering and deterministic bootstrap (seeded).
- Continuous-operation safety: artifact counts validated by cross-file consistency (manifest rows vs resolved docs), with drift surfaced as reports rather than hidden assumptions.

### Must NOT Have (guardrails)
- No manuscript writing or camera-ready formatting.
- No non-deterministic clustering (no Louvain unless later explicitly requested and seeded).
- No “hidden” LLM dependencies in evaluator (translation/router generation must be deterministic; embedder is OK).
- No production rollout defaults; any online integration must be feature-flagged and OFF by default.
- No evaluation that silently proceeds with missing corpus joins (must fail fast + write an unmatched report).

## Verification Strategy
> ZERO HUMAN INTERVENTION: all checks are script- or test-driven.
- Test decision: tests-after (pytest) + script-based end-to-end checks.
- QA policy: every task includes 2 scenarios (happy + failure) with stored evidence.
- Evidence locations:
  - `.sisyphus/evidence/task-{N}-{slug}.*` for task-level evidence
  - `.sisyphus/evidence/paper-a/*` for run artifacts
- Continuous plan policy:
  - Treat `2026-03-04` stats as baseline reference, not hard acceptance constants.
  - For every run, persist snapshot metadata (`manifest_path`, `manifest_sha256`, `manifest_row_count`, `resolved_doc_count`, `built_at`).
  - Prefer deterministic consistency checks (`doc_meta == corpus_doc_ids == manifest_row_count`) over absolute fixed counts.
  - If previous snapshot is available, emit a drift report with deltas for docs/devices/topics/shared topics; fail only on data integrity violations (e.g., unresolved joins), not on legitimate corpus growth.

## Execution Strategy
### Parallel Execution Waves
Wave 1 (Foundation + Corpus Join)
- Task 1 (preflight ES/index + capture env)
- Task 2 (scripts/paper_a scaffolding)
- Task 3 (canonicalization contract + unit tests)
- Task 4 (corpus doc_meta + whitelist builder)

Wave 2 (Policy Artifacts + Eval Sets)
- Task 5 (shared_topics + doc_scope)
- Task 6 (family_map)
- Task 7 (explicit eval set)
- Task 8 (masked + ambiguous eval sets)

Wave 3 (Evaluator + Metrics)
- Task 9 (retrieval runner building blocks)
- Task 10 (scope filter DSL + tests)
- Task 11 (metrics + aggregation + bootstrap + run manifest)

Wave 4 (Optional: ES Backfill + Router)
- Task 12 (ES mapping update + backfill + coverage verification)
- Task 13 (router prototypes + Matryoshka router + deterministic fallback)
- Task 14 (run P2-P4 ablations and write paper-ready tables)

### Dependency Matrix (all tasks)
- 1 blocks 4, 9, 12 (needs ES connectivity + index name)
- 3 blocks 4, 7, 8 (join correctness depends on canonicalization)
- 4 blocks 5, 6, 7, 8 (policy + eval sets depend on corpus meta)
- 5 blocks 10, 11 (scope filter uses doc_scope)
- 6 blocks 11, 13 (family expansion + router optional)
- 7 blocks 11 (explicit eval set needed for first full run)
- 8 blocks 11, 14 (masked/ambiguous for router runs)
- 9 blocks 11 (evaluator needs retrievers)
- 10 blocks 11 (scope-aware systems use the new filter)
- 12 blocks 14 if using ES-native scope fields (optional)
- 13 blocks 14 (router runs)

### Agent Dispatch Summary
- Wave 1: 4 tasks -> categories: unspecified-high (ES/scripts), quick (scaffolding), deep (canonicalization)
- Wave 2: 4 tasks -> categories: deep (policy math), unspecified-high (datasets)
- Wave 3: 3 tasks -> categories: deep (metrics/CI), unspecified-high (filters)
- Wave 4: 3 tasks -> categories: unspecified-high (ES backfill), deep (router), writing (tables/docs)

## TODOs
> Implementation + verification live in the same task.

- [x] 1. Preflight: ES connectivity + alias resolution + corpus sanity snapshot

  **What to do**:
  - Implement `scripts/paper_a/preflight_es.py` to:
    - Ping ES using `backend/config/settings.py:SearchSettings` (`SEARCH_ES_HOST`, auth if set).
    - Resolve alias -> concrete index name using `backend/llm_infrastructure/elasticsearch/manager.py:EsIndexManager.get_alias_target()`.
    - Write `.sisyphus/evidence/task-01-preflight-es.json` with: host, env, alias, resolved_index, timestamp.
  - Include a CLI flag `--index` to override alias (default: `rag_chunks_{SEARCH_ES_ENV}_current`).

  **Must NOT do**:
  - Do not create/delete/switch ES indices or aliases.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ES + settings wiring needs care
  - Skills: []
  - Omitted: [`playwright`] — No browser work

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4,9,12 | Blocked By: none

  **References**:
  - API/Type: `backend/config/settings.py` (`SearchSettings`)
  - ES alias logic: `backend/llm_infrastructure/elasticsearch/manager.py`
  - Existing ES inspection CLI: `scripts/es_query.py`

  **Acceptance Criteria**:
  - [ ] `python scripts/paper_a/preflight_es.py --out .sisyphus/evidence/task-01-preflight-es.json` exits 0
  - [ ] Output JSON contains non-empty `resolved_index`

  **QA Scenarios**:
  ```
  Scenario: ES reachable and alias resolves
    Tool: Bash
    Steps: python scripts/paper_a/preflight_es.py --out .sisyphus/evidence/task-01-preflight-es.json
    Expected: exit code 0; JSON has keys host, alias, resolved_index
    Evidence: .sisyphus/evidence/task-01-preflight-es.json

  Scenario: ES unreachable
    Tool: Bash
    Steps: SEARCH_ES_HOST=http://127.0.0.1:1 python scripts/paper_a/preflight_es.py --out .sisyphus/evidence/task-01-preflight-es.json
    Expected: exit code !=0 with clear error; no partial success message
    Evidence: .sisyphus/evidence/task-01-preflight-es-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add ES preflight helper` | Files: [`scripts/paper_a/preflight_es.py`]

- [x] 2. Scaffold `scripts/paper_a/` CLIs with consistent runbook patterns

  **What to do**:
  - Create `scripts/paper_a/` directory and baseline CLI style mirroring `scripts/paper_b/run_paper_b_eval.py`:
    - argparse + dataclass for args
    - stable out-dir defaults under `.sisyphus/evidence/paper-a/`
    - JSON/JSONL writers with UTF-8 and trailing newline
  - Add `scripts/paper_a/_io.py` (write_json, write_jsonl, read_jsonl) and `scripts/paper_a/_paths.py` (resolve project root).

  **Must NOT do**:
  - Do not copy Paper B code verbatim; keep only shared patterns.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: scaffolding and small utilities
  - Skills: []
  - Omitted: [`git-master`] — no complex git ops

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 3,4,5,6,7,8,9,11 | Blocked By: none

  **References**:
  - Pattern: `scripts/paper_b/run_paper_b_eval.py`

  **Acceptance Criteria**:
  - [ ] `python -c "import scripts.paper_a._io"` succeeds

  **QA Scenarios**:
  ```
  Scenario: Module import works
    Tool: Bash
    Steps: python -c "import scripts.paper_a._io; import scripts.paper_a._paths"
    Expected: exit code 0
    Evidence: .sisyphus/evidence/task-02-import-ok.txt

  Scenario: Missing module path
    Tool: Bash
    Steps: python -c "import scripts.paper_a.does_not_exist"
    Expected: exit code !=0
    Evidence: .sisyphus/evidence/task-02-import-error.txt
  ```

  **Commit**: YES | Message: `chore(paper-a): scaffold scripts package` | Files: [`scripts/paper_a/`]

- [x] 3. Canonicalization contract for device/doc/doc_type + unit tests

  **What to do**:
  - Implement `scripts/paper_a/canonicalize.py`:
    - `compact_key(text)`: lower + remove whitespace/[_\-./] (same spirit as `backend/llm_infrastructure/llm/langgraph_agent.py:_compact_text`).
    - `doc_id_variant_vlm(name)`: Python normalization matching `scripts/vlm_es_ingest.py:generate_doc_id`.
    - `doc_id_variant_batch_sop(name)`: Bash normalization matching `scripts/batch_ingest_sop.sh` (ASCII-only, Korean collapsed).
    - `normalize_doc_type_es(value)`: map corpus doc_type (`sop_pdf/sop_pptx/ts/setup_manual`) and golden set doc types (`SOP`, etc.) into ES `doc_type` keys (`sop`, `ts`, `setup`, `myservice`, `gcb`). Reuse `backend/domain/doc_type_mapping.py:normalize_doc_type` logic by importing, but return canonical ES key.
    - `canonicalize_device_name(name, candidates)`: return exact candidate string when `compact_key` matches.
  - Add tests:
    - `backend/tests/test_paper_a_canonicalize.py` for doc_id variants and device canonicalization.

  **Must NOT do**:
  - Do not hardcode environment-specific device lists; the function must take candidates list.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: this contract prevents silent metric corruption
  - Skills: []
  - Omitted: [`playwright`]

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4,7,8 | Blocked By: 2

  **References**:
  - Ingest normalization: `scripts/vlm_es_ingest.py:91` (`generate_doc_id`)
  - Batch SOP normalization: `scripts/batch_ingest_sop.sh:26`
  - Doc type mapping: `backend/domain/doc_type_mapping.py`
  - Similar compacting: `backend/llm_infrastructure/llm/langgraph_agent.py:2445`

  **Acceptance Criteria**:
  - [ ] `python -m pytest backend/tests/test_paper_a_canonicalize.py -q` passes

  **QA Scenarios**:
  ```
  Scenario: doc_id variants differ as expected
    Tool: Bash
    Steps: python -m pytest backend/tests/test_paper_a_canonicalize.py -q
    Expected: tests pass
    Evidence: .sisyphus/evidence/task-03-tests.txt

  Scenario: device canonicalization fails fast
    Tool: Bash
    Steps: python -c "from scripts.paper_a.canonicalize import canonicalize_device_name; print(canonicalize_device_name('NOPE', ['SUPRA XP']))"
    Expected: prints empty/None (per chosen API) and caller can detect
    Evidence: .sisyphus/evidence/task-03-device-miss.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add canonicalization contract` | Files: [`scripts/paper_a/canonicalize.py`, `backend/tests/test_paper_a_canonicalize.py`]

- [x] 4. Build corpus doc_meta (manifest -> ES join) + corpus doc_id whitelist

  **What to do**:
  - Implement `scripts/paper_a/build_corpus_meta.py`:
    - Input: `data/chunk_v3_normalize_table.md` (source-of-truth manifest; includes original file name, doc_type section, topic).
    - For each row:
      - Compute candidate `doc_id` values from `file_name` stem using BOTH doc_id variants (Task 3).
      - Resolve which candidate exists in ES by querying `doc_id` terms in the resolved index.
      - Fetch 1 representative chunk for the resolved `doc_id` to capture ES-side `device_name`, `doc_type`, `equip_id`.
    - Output files (under `--out-dir`):
      - `doc_meta.jsonl` (one row per manifest doc): `{source_file, topic, manifest_doc_type, es_doc_id, es_doc_type, es_device_name, es_equip_id}`
      - `corpus_doc_ids.txt` (one ES doc_id per line; length must equal `manifest_row_count`)
      - `unresolved_docs.jsonl` for any manifest rows that cannot be resolved (and FAIL the command if non-empty)
      - `corpus_snapshot.json` with counts + metadata (`manifest_path`, `manifest_sha256`, `manifest_row_count`, `resolved_doc_count`, docs/devices/topics/doc_types)
  - IMPORTANT: device counts must use ES `device_name` strings (exact), since filters operate on ES terms.

  **Must NOT do**:
  - Do not proceed with partial joins; if any doc is unresolved, exit non-zero.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ES join logic + failure modes
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 5,6,7,8,9,11,12 | Blocked By: 1,3

  **References**:
  - Corpus normalize table: `data/chunk_v3_normalize_table.md`
  - ES hit fields: `backend/llm_infrastructure/retrieval/engines/es_search.py:341` (`_source_fields` includes `device_name`, `doc_type`, `equip_id`)
  - ES doc_id vs chunk_id: `backend/llm_infrastructure/elasticsearch/document.py:177`

  **Acceptance Criteria**:
  - [ ] `python scripts/paper_a/build_corpus_meta.py --normalize-table data/chunk_v3_normalize_table.md --out-dir .sisyphus/evidence/paper-a/corpus` exits 0
  - [ ] `.sisyphus/evidence/paper-a/corpus/doc_meta.jsonl` line count equals `corpus_snapshot.json.manifest_row_count`
  - [ ] `.sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt` line count equals `corpus_snapshot.json.manifest_row_count`
  - [ ] `.sisyphus/evidence/paper-a/corpus/unresolved_docs.jsonl` does not exist OR is empty

  **QA Scenarios**:
  ```
  Scenario: Corpus meta builds cleanly
    Tool: Bash
    Steps: python scripts/paper_a/build_corpus_meta.py --normalize-table data/chunk_v3_normalize_table.md --out-dir .sisyphus/evidence/paper-a/corpus
    Expected: exit 0; doc_meta.jsonl and corpus_doc_ids.txt each match corpus_snapshot.manifest_row_count
    Evidence: .sisyphus/evidence/task-04-corpus-meta.txt

  Scenario: Manifest path missing
    Tool: Bash
    Steps: python scripts/paper_a/build_corpus_meta.py --normalize-table data/does_not_exist.md --out-dir .sisyphus/evidence/paper-a/corpus
    Expected: exit !=0 with clear message
    Evidence: .sisyphus/evidence/task-04-corpus-meta-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): build corpus meta joiner` | Files: [`scripts/paper_a/build_corpus_meta.py`]

- [x] 5. Build Shared topics + doc_scope (is_shared + scope_level)

  **What to do**:
  - Implement `scripts/paper_a/build_shared_and_scope.py`:
    - Input: corpus `doc_meta.jsonl` from Task 4.
    - Compute topic -> set(device_name_es) and `deg(topic)`.
    - Shared topic rule (fixed): `deg(topic) >= 3` AND `manifest_doc_type in {sop_pdf, sop_pptx, ts}`.
      - Explicitly exclude `setup_manual` from being shared.
    - Output under `--out-dir`:
      - `shared_topics.json` with `{topic: {deg:int, devices:[...], is_shared:bool}}`
      - `doc_scope.jsonl` with `{es_doc_id, es_device_name, es_doc_type, topic, is_shared, scope_level}`
      - `shared_doc_ids.txt` (one `es_doc_id` per line where `is_shared=true`; used by evaluator without ES backfill)
    - Scope level assignment (fixed):
      - if `is_shared`: `scope_level="shared"`
      - else if `es_doc_type in {"myservice","gcb"}`: `scope_level="equip"`
      - else: `scope_level="device"`
    - Write `policy_snapshot.json` with counts: shared_topic_count, shared_doc_count.
  - Validate deterministic output for identical input; compare `shared_topic_count` against `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md` only as baseline reference (non-blocking).
  - If previous snapshot is provided, emit drift deltas (`shared_topic_count_delta`, `shared_doc_count_delta`) into `policy_snapshot.json` or `drift_report.json`.

  **Must NOT do**:
  - Do not introduce any probabilistic/shared-by-embedding policy; this is rule-based and deterministic.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: correctness affects all paper claims
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 10,11,12 | Blocked By: 4

  **References**:
  - Spec shared rule: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (Section 2.1)
  - Corpus stats: `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md`

  **Acceptance Criteria**:
  - [ ] Running the script writes `shared_topics.json` and `doc_scope.jsonl`
  - [ ] `policy_snapshot.json` reports numeric `shared_topic_count` and `shared_doc_count`
  - [ ] `doc_scope.jsonl` contains only `scope_level` in `{shared,device,equip}`

  **QA Scenarios**:
  ```
  Scenario: Shared topics computed deterministically
    Tool: Bash
    Steps: python scripts/paper_a/build_shared_and_scope.py --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --out-dir .sisyphus/evidence/paper-a/policy
    Expected: exit 0; policy_snapshot.json contains shared counts and values are deterministic for the same input
    Evidence: .sisyphus/evidence/task-05-shared-scope.json

  Scenario: Corrupt corpus meta
    Tool: Bash
    Steps: python scripts/paper_a/build_shared_and_scope.py --corpus-meta .sisyphus/evidence/paper-a/corpus/does_not_exist.jsonl --out-dir .sisyphus/evidence/paper-a/policy
    Expected: exit !=0
    Evidence: .sisyphus/evidence/task-05-shared-scope-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): build shared topics and scope levels` | Files: [`scripts/paper_a/build_shared_and_scope.py`]

- [x] 6. Build Family(device) via weighted Jaccard + connected components (deterministic)

  **What to do**:
  - Implement `scripts/paper_a/build_family_map.py`:
    - Input: corpus `doc_meta.jsonl` and `shared_topics.json`.
    - Build device -> set(topics).
    - Compute weighted-Jaccard for all pairs:
      - `w(topic) = 1 / log(1 + deg(topic))` where deg is topic device_count.
      - `wj(a,b) = sum(w(t) for t in I) / sum(w(t) for t in U)`.
    - Edge rule: include undirected edge if `wj(a,b) >= 0.2` (fixed tau).
    - Clustering: connected components over this graph.
    - Deterministic family_id assignment:
      - Sort devices lexicographically within each component.
      - Sort components by their first device name.
      - Assign family ids `F00`, `F01`, ... in that order.
    - Output `family_map.json`: `{device_to_family: {...}, families: {"F00": [...], ...}, params: {...}}`.
  - Add unit test `backend/tests/test_paper_a_family_determinism.py` that runs the builder on a tiny fixed fixture and asserts stable family ids.

  **Must NOT do**:
  - Do not use Louvain or any non-deterministic community detection.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: graph math + determinism constraints
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 11,13,14 | Blocked By: 4,5

  **References**:
  - Spec family graph: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (Section 2.2)
  - Determinism guardrail: Metis review

  **Acceptance Criteria**:
  - [ ] `python scripts/paper_a/build_family_map.py ...` writes `family_map.json`
  - [ ] `family_map.json` contains `params.tau == 0.2`
  - [ ] `python -m pytest backend/tests/test_paper_a_family_determinism.py -q` passes

  **QA Scenarios**:
  ```
  Scenario: Family map builds and tests pass
    Tool: Bash
    Steps: python scripts/paper_a/build_family_map.py --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --shared-topics .sisyphus/evidence/paper-a/policy/shared_topics.json --out .sisyphus/evidence/paper-a/policy/family_map.json && python -m pytest backend/tests/test_paper_a_family_determinism.py -q
    Expected: exit 0; family_map.json present
    Evidence: .sisyphus/evidence/task-06-family-map.json

  Scenario: deg(topic)=0 handling
    Tool: Bash
    Steps: python -c "import math; print(1.0/max(math.log(1+1), 1e-9))"
    Expected: prints finite value (guard against div-by-zero)
    Evidence: .sisyphus/evidence/task-06-deg-guard.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add deterministic family map builder` | Files: [`scripts/paper_a/build_family_map.py`, `backend/tests/test_paper_a_family_determinism.py`]

- [x] 7. Build Explicit eval set from SOP CSV (SOP79-style) + resolve gold doc_ids

  **What to do**:
  - Implement `scripts/paper_a/build_eval_sets.py` supporting:
    - `--sop-csv "data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv"`
    - `--corpus-meta` (Task 4 output) for doc_id resolution
  - Parse CSV columns (header row in Korean):
    - query: `질문내용` (fallback to English lines if present)
    - target_device: `장비`
    - gold source file: `정답문서`
    - expected pages: `정답 페이지`
  - For each row, resolve `gold_doc_ids`:
    - Prefer mapping `정답문서` -> `es_doc_id` by matching `source_file` in `doc_meta.jsonl`.
    - If no match, attempt doc_id variants on the filename stem and check membership in corpus doc_id whitelist.
    - If still unresolved: write to `unmatched_gold.jsonl` and FAIL the build.
  - Write `.jsonl` in Paper A schema (spec Section 12.1) to:
    - `explicit.jsonl` with `split="explicit"` and `qid` as `A-E-0001` ...
  - Also write a schema validator `scripts/paper_a/validate_eval_jsonl.py` (or reuse `scripts/evaluation/validate_agent_eval_jsonl.py` style) to ensure required keys.

  **Must NOT do**:
  - Do not silently drop rows; if a row cannot be resolved to a gold doc_id, fail.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: dataset join correctness is critical
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 11 | Blocked By: 4

  **References**:
  - SOP CSV: `data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv`
  - Corpus meta join: Task 4 output
  - Spec schema: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (Section 12.1)

  **Acceptance Criteria**:
  - [ ] `explicit.jsonl` exists and has >= 60 rows
  - [ ] `unmatched_gold.jsonl` is empty/non-existent
  - [ ] `python scripts/paper_a/validate_eval_jsonl.py --path explicit.jsonl` exits 0

  **QA Scenarios**:
  ```
  Scenario: Explicit set builds and validates
    Tool: Bash
    Steps: python scripts/paper_a/build_eval_sets.py --sop-csv "data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv" --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --out-dir .sisyphus/evidence/paper-a/eval_sets && python scripts/paper_a/validate_eval_jsonl.py --path .sisyphus/evidence/paper-a/eval_sets/explicit.jsonl
    Expected: exit 0; explicit.jsonl row count >=60
    Evidence: .sisyphus/evidence/task-07-explicit-set.txt

  Scenario: Broken CSV schema
    Tool: Bash
    Steps: python scripts/paper_a/build_eval_sets.py --sop-csv data/sample_maintenance_report.txt --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --out-dir .sisyphus/evidence/paper-a/eval_sets
    Expected: exit !=0 and error mentions missing required columns
    Evidence: .sisyphus/evidence/task-07-explicit-set-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): build explicit eval set from SOP CSV` | Files: [`scripts/paper_a/build_eval_sets.py`, `scripts/paper_a/validate_eval_jsonl.py`]

- [x] 8. Build Masked + Ambiguous eval sets (deterministic masking rules)

  **What to do**:
  - Extend `scripts/paper_a/build_eval_sets.py` to additionally produce:
    - `masked.jsonl`: derived from explicit; remove/mask device tokens.
    - `ambiguous.jsonl`: subset of explicit where topic is shared across >=2 devices (use `doc_meta.jsonl` topic + device stats).
  - Masking rules (fixed, deterministic):
    - For each row, build device alias list from `target_device`:
      - Raw string
      - Lower/upper variants
      - Compact variants removing whitespace/underscores
    - Replace occurrences in the query with `[DEVICE]` (case-insensitive).
    - Also remove patterns like "{device} 설비".
    - Do NOT delete component words (controller/ffu/robot etc).
  - Ambiguous set selection rule (fixed):
    - If the row's `topic` (from corpus meta join by `source_file`) has `deg(topic) >= 2`, include in ambiguous.
    - Always mask device name in ambiguous queries.

  **Must NOT do**:
  - Do not use LLM rewriting.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: dataset determinism + join logic
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 14 | Blocked By: 7

  **References**:
  - Corpus stats shared topics list: `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md` (Section 5)

  **Acceptance Criteria**:
  - [ ] `masked.jsonl` has the same row count as `explicit.jsonl`
  - [ ] For every row in `masked.jsonl`, `query` does not contain the literal target device (compact-key check)
  - [ ] `ambiguous.jsonl` row count is `>= 0` and `<= explicit.jsonl`, and every row satisfies `deg(topic) >= 2`

  **QA Scenarios**:
  ```
  Scenario: Masked and ambiguous sets build
    Tool: Bash
    Steps: python scripts/paper_a/build_eval_sets.py --sop-csv "data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv" --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --out-dir .sisyphus/evidence/paper-a/eval_sets
    Expected: exit 0; masked.jsonl count==explicit.jsonl count; ambiguous.jsonl count is within [0, explicit]
    Evidence: .sisyphus/evidence/task-08-masked-ambiguous.txt

  Scenario: Device masking misses a token
    Tool: Bash
    Steps: python -c "from scripts.paper_a.canonicalize import compact_key; print(compact_key('SUPRA XP') in compact_key('SUPRA XP 설비의 ...'))"
    Expected: True (demonstrates why compact checks are required)
    Evidence: .sisyphus/evidence/task-08-masking-check.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add masked and ambiguous eval sets` | Files: [`scripts/paper_a/build_eval_sets.py`]

- [x] 9. Retrieval runner building blocks (BM25/Dense/Hybrid/Rerank) with corpus whitelist filter

  **What to do**:
  - Implement `scripts/paper_a/retrieval_runner.py`:
    - Create ES client and `EsSearchEngine` with the same weighted text fields as `backend/services/es_search_service.py:124`.
    - Build query embeddings via `backend/services/embedding_service.py:EmbeddingService` (no LLM).
    - Implement methods:
      - BM25: `EsSearchEngine.sparse_search()`
      - Dense: `EsSearchEngine.dense_search()`
      - Hybrid: `EsSearchEngine.hybrid_search(use_rrf=True)`
    - Support a mandatory `corpus_doc_ids` filter applied to ALL systems (manifest-scoped corpus).
    - Support optional rerank via `backend/llm_infrastructure/reranking/adapters/cross_encoder.py:CrossEncoderReranker`.
    - Output a unified hit format: `{rank, doc_id, score, metadata{device_name,equip_id,doc_type,chunk_id,page}}`.

  **Must NOT do**:
  - Do not call `backend/services/retrieval_pipeline.py:run_retrieval_pipeline` (would pull in LLM steps).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: careful alignment with production retrieval primitives
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 11 | Blocked By: 4

  **References**:
  - Production weighting: `backend/services/es_search_service.py:124`
  - ES engine: `backend/llm_infrastructure/retrieval/engines/es_search.py`
  - Hybrid retriever reference: `backend/llm_infrastructure/retrieval/adapters/es_hybrid.py`
  - Reranker: `backend/llm_infrastructure/reranking/adapters/cross_encoder.py`

  **Acceptance Criteria**:
  - [ ] A small smoke run returns non-empty hits for a known query when corpus filter is provided
  - [ ] Returned hit rows include `doc_id` and `metadata.device_name`

  **QA Scenarios**:
  ```
  Scenario: Retrieval runner smoke test
    Tool: Bash
    Steps: python -c "from scripts.paper_a.retrieval_runner import smoke_test; smoke_test(corpus_doc_ids_path='.sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt')"
    Expected: prints/returns non-empty hits
    Evidence: .sisyphus/evidence/task-09-retrieval-smoke.txt

  Scenario: Missing corpus filter
    Tool: Bash
    Steps: python -c "from scripts.paper_a.retrieval_runner import smoke_test; smoke_test(corpus_doc_ids_path='')"
    Expected: raises with message that corpus filter is mandatory
    Evidence: .sisyphus/evidence/task-09-retrieval-smoke-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add retrieval runner primitives` | Files: [`scripts/paper_a/retrieval_runner.py`]

- [x] 10. Implement scope-level-aware ES filter DSL + unit tests

  **What to do**:
  - Add `backend/llm_infrastructure/retrieval/filters/scope_filter.py` implementing TWO equivalent builders:
    - Doc-id mode (DEFAULT for Paper A evaluator; requires NO ES writes):
      - `build_scope_filter_by_doc_ids(allowed_devices, allowed_equip_ids, *, shared_doc_ids, device_doc_types, equip_doc_types)` returning an ES bool filter fragment that matches:
        - Shared branch: `doc_id in shared_doc_ids`
        - Device branch: `doc_type in device_doc_types` AND `device_name in allowed_devices` AND `doc_id not in shared_doc_ids`
        - Equip branch: `doc_type in equip_doc_types` AND (`equip_id in allowed_equip_ids` if provided else `device_name in allowed_devices`)
        - Combine with `bool.should` + `minimum_should_match=1`
    - Field mode (OPTIONAL; used only if Task 12 backfilled ES fields):
      - `build_scope_filter_by_fields(allowed_devices, allowed_equip_ids)` using `scope_level` / `is_shared` fields:
        - Always allow `scope_level==shared`
        - Allow `scope_level==device` AND `device_name in allowed_devices`
        - Allow `scope_level==equip` AND (`equip_id in allowed_equip_ids` else device fallback)
        - Combine with `bool.should` + `minimum_should_match=1`
    - `apply_scope_filter(base_filter, scope_filter)` to combine with existing filters using `bool.filter`.
  - Add tests `backend/tests/test_scope_filter_dsl.py` with golden expected dicts for:
    - devices only
    - equip only
    - devices+equip
    - empty scope (must produce None or global behavior per fixed spec)
  - Guardrail: Document in code and tests that this replaces the incorrect AND behavior of `EsSearchEngine.build_filter(device_names=..., equip_ids=...)` for scope routing.

  **Must NOT do**:
  - Do not change existing `EsSearchEngine.build_filter()` behavior yet (keep backward compatibility).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: correctness-critical query DSL
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 11 | Blocked By: 5

  **References**:
  - Existing filter builder (too strict for scope): `backend/llm_infrastructure/retrieval/engines/es_search.py:450`
  - Oracle guidance: scope filter as should-branches

  **Acceptance Criteria**:
  - [ ] `python -m pytest backend/tests/test_scope_filter_dsl.py -q` passes

  **QA Scenarios**:
  ```
  Scenario: Scope filter DSL tests pass
    Tool: Bash
    Steps: python -m pytest backend/tests/test_scope_filter_dsl.py -q
    Expected: all tests pass
    Evidence: .sisyphus/evidence/task-10-scope-filter-tests.txt

  Scenario: Empty allowed_devices for device-level docs
    Tool: Bash
    Steps: python -c "from backend.llm_infrastructure.retrieval.filters.scope_filter import build_scope_filter_by_doc_ids; print(build_scope_filter_by_doc_ids([], None, shared_doc_ids=[], device_doc_types=['sop','ts','setup'], equip_doc_types=['myservice','gcb']))"
    Expected: returns None or a filter that does not accidentally exclude everything (as per decided API)
    Evidence: .sisyphus/evidence/task-10-empty-scope.txt
  ```

  **Commit**: YES | Message: `feat(retrieval): add scope-level filter builder` | Files: [`backend/llm_infrastructure/retrieval/filters/scope_filter.py`, `backend/tests/test_scope_filter_dsl.py`]

- [ ] 11. Paper A evaluator: systems B0-B4/P1-P4, metrics, bootstrap, run manifests

  **What to do**:
  - Implement `scripts/paper_a/evaluate_paper_a.py` that:
    - Loads eval set JSONL (explicit/masked/ambiguous).
    - Loads `doc_scope.jsonl` and `family_map.json`.
    - Runs each system id with fixed mapping:
      - B0: BM25
      - B1: Dense
      - B2: Hybrid (RRF)
      - B3: Hybrid + rerank
      - B4: Hard-device filter if a device is parsable from query; else global
      - P1: Hard + Shared (scope filter)
      - P2: Router Top-M (optional; if router disabled, skip with clear status)
      - P3: Router + Family
      - P4: Router + Family + Shared
    - Parsing (deterministic, no LLM):
      - Use device candidate list from corpus meta (unique `es_device_name`).
      - Detect device by compact-key substring match (same policy as `backend/llm_infrastructure/llm/langgraph_agent.py:_extract_devices_from_query`).
      - Equip_id parsing is optional for this corpus; if present, uppercase.
    - Retrieval:
      - Always apply corpus whitelist filter.
      - For scoped systems, apply the scope filter DSL (Task 10) in doc-id mode by default:
        - load `shared_doc_ids.txt` derived from `doc_scope.jsonl`
        - infer device/equip doc_types from ES `doc_type` (and the fixed mapping)
      - If `--use-es-scope-fields` is passed, switch to field-mode scope filter (requires Task 12).
    - Metrics (doc-level):
      - Deduplicate hits to top-k unique `doc_id` before computing all @k metrics.
      - Contamination metrics per `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`:
        - Raw_Cont@k (shared counted as contamination)
        - Adj_Cont@k (exclude shared)
        - Shared@k
        - CE@k (binary any adjusted OOS)
      - Quality:
        - Hit@k: `gold_doc_ids` intersects top-k
        - MRR: reciprocal rank of first gold doc
    - Outputs:
      - `per_query.csv`, `summary_all.csv`, `summary_by_split.csv`, `summary_by_doc_type.csv`
      - `bootstrap_ci.json` for requested comparisons
      - `run_manifest.json` containing:
        - git sha (read from `git rev-parse HEAD`)
        - alias + resolved index
        - corpus doc_id count
        - config dict + seed
  - Add unit tests for metric calculations on a tiny fixture (optional but recommended).

  **Must NOT do**:
  - Do not silently skip systems without writing a skip reason.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: metric correctness + reproducibility requirements
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 14 | Blocked By: 7,9,10

  **References**:
  - Paper A metrics: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (Section 4)
  - Common protocol run metadata: `docs/papers/10_common_protocol/paper_common_protocol.md` (Section 6)
  - ES hit shape: `backend/llm_infrastructure/retrieval/engines/es_search.py:425`

  **Acceptance Criteria**:
  - [ ] Running evaluator on explicit set writes all expected output files
  - [ ] `per_query.csv` contains required columns: `qid,split,system_id,raw_cont@5,adj_cont@5,shared@5,ce@5,hit@5,mrr`
  - [ ] `run_manifest.json` includes non-empty `git_sha` and `resolved_index`
  - [ ] Bootstrap json contains numeric `ci_lower`, `ci_upper`, `delta_mean` for each comparison

  **QA Scenarios**:
  ```
  Scenario: Explicit evaluator run completes
    Tool: Bash
    Steps: python scripts/paper_a/evaluate_paper_a.py --eval-set .sisyphus/evidence/paper-a/eval_sets/explicit.jsonl --systems B0,B1,B2,B3,B4,P1 --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --out-dir .sisyphus/evidence/paper-a/runs/run_explicit_001
    Expected: exit 0; output dir contains per_query.csv, summary_all.csv, bootstrap_ci.json, run_manifest.json
    Evidence: .sisyphus/evidence/task-11-eval-explicit.txt

  Scenario: Missing doc_scope join
    Tool: Bash
    Steps: python scripts/paper_a/evaluate_paper_a.py --eval-set .sisyphus/evidence/paper-a/eval_sets/explicit.jsonl --systems P1 --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/paper-a/policy/does_not_exist.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --out-dir .sisyphus/evidence/paper-a/runs/run_error
    Expected: exit !=0 with clear error about missing doc_scope file
    Evidence: .sisyphus/evidence/task-11-eval-explicit-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add retrieval-only evaluator with contamination metrics` | Files: [`scripts/paper_a/evaluate_paper_a.py`]

- [ ] 11-fix. P1 scope filter Hit@5 급락 원인 분석 및 수정

  **What to do**:
  - **문제**: run_explicit_003에서 P1(Hard+Shared)의 Hit@5가 0.165로 B4(0.899) 대비 급락.
    Shared@5=0.954로 shared 문서가 top-k를 거의 독점하여 정답 문서가 밀려남.
  - **원인 조사**:
    1. `build_scope_filter_by_doc_ids()`에서 P1 시스템의 scope filter가 Hard device 문서를 정상적으로 포함하는지 확인.
       - Shared branch만 열리고 device branch가 닫혀 있을 가능성 (bool.should 조건 확인).
    2. `evaluate_paper_a.py`에서 P1 시스템의 `allowed_devices` 전달 확인.
       - auto_parse가 device를 잡았는데 scope filter에 전달 안 되는 경우?
    3. per_query.csv에서 P1의 hit=1인 질의 vs hit=0인 질의 비교.
  - **예상 원인**: P1은 "Hard + Shared"인데 scope filter가 `device_name in S AND is_shared` 대신
    shared-only로 동작하고 있을 수 있음. 또는 shared_doc_ids에 해당 장비 문서가 다수 포함되어
    BM25/Dense 점수에서 shared 문서가 device 문서를 이기는 현상.
  - **수정 후 검증**: P1 재실행하여 Hit@5 >= 0.85 (B4 대비 2%p 이내) 확인.

  **추가 발견사항 (run_explicit_003 리뷰)**:
  1. **equip_doc_types에 `gcb` 누락**: `run_manifest.json`에 `equip_doc_types: ["myservice"]`만 기록됨.
     spec §1.2와 T5(line 340)에서 `gcb`도 equip scope로 정의했으므로,
     `evaluate_paper_a.py` 호출 시 `--equip-doc-types myservice,gcb`로 수정 필요.
     → scope filter의 equip branch에서 gcb 문서가 누락되어 P1 Hit@5 하락 원인 중 하나일 수 있음.
  2. **corpus device count 불일치 (29 ES vs 21 normalize_table)**:
     `corpus_snapshot.json`에 device 29개 기록, `data/chunk_v3_normalize_table.md` 기준 21개.
     ES에서 variant/alias가 별도 device로 카운트되는 것으로 추정.
     → `build_corpus_meta.py`의 canonicalization이 device_name을 정규화하지 않고 ES raw 값을 그대로 쓰는지 확인.
     → family_map과 doc_scope에서 device 기준이 일관적인지 cross-check 필요.
  3. **bootstrap_samples 2000 → 10000 권장**:
     `run_manifest.json`에 `bootstrap_samples: 2000`. 논문 제출용 최종 실험에서는
     CI 안정성을 위해 10000 권장 (spec §4 참조). 디버깅 단계에서는 2000 유지 가능.

  **Must NOT do**:
  - Do not change B0-B4 baseline 로직.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 14 | Blocked By: 11

  **Acceptance Criteria**:
  - [ ] P1 scope filter가 Hard device 문서 + Shared 문서를 모두 포함하는 것을 단위 테스트로 확인
  - [ ] equip_doc_types에 `gcb` 포함된 상태로 재실행 (`--equip-doc-types myservice,gcb`)
  - [ ] corpus device count 불일치 원인 확인 및 문서화 (29 ES vs 21 normalize_table)
  - [ ] P1 재실행 시 Hit@5 >= 0.85 (B4 대비 절대 하락 5%p 이내)
  - [ ] Adj_Cont@5 여전히 B4 대비 감소

  **Commit**: YES | Message: `fix(paper-a): fix P1 scope filter to include hard device docs` | Files: TBD after root cause

- [ ] 12. (Optional) ES mapping update + backfill `is_shared`/`scope_level` for corpus docs (online/perf only)

  **What to do**:
  - Add fields to mapping in `backend/llm_infrastructure/elasticsearch/mappings.py:get_rag_chunks_mapping`:
    - `"scope_level": {"type": "keyword"}`
    - `"is_shared": {"type": "boolean"}`
  - Implement `scripts/paper_a/es_update_mapping_scope_fields.py`:
    - `PUT /{index}/_mapping` add the two fields (non-destructive).
  - Implement `scripts/paper_a/es_backfill_scope_fields.py`:
    - Input: `doc_scope.jsonl`.
    - For each `es_doc_id`, update all chunks with that `doc_id` setting `scope_level` and `is_shared`.
    - Provide `--dry-run` (prints counts only).
  - Implement `scripts/paper_a/es_verify_scope_coverage.py`:
    - Aggregates docs missing `scope_level` within the corpus whitelist and fails if coverage < 0.99.

  **Must NOT do**:
  - Do not run against prod env by default; require explicit `--index` and print it.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ES writes + safety
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: none | Blocked By: 5

  **References**:
  - Mapping: `backend/llm_infrastructure/elasticsearch/mappings.py:16`
  - ES migration pattern: `scripts/es_migrate_v2.py`

  **Acceptance Criteria**:
  - [ ] Mapping update script exits 0 (idempotent)
  - [ ] Backfill script updates >0 docs and exits 0
  - [ ] Coverage script reports >=99% docs in corpus have `scope_level`
  - [ ] Paper A evaluator still runs with doc-id mode even if this task is skipped

  **QA Scenarios**:
  ```
  Scenario: Dry-run then backfill then coverage
    Tool: Bash
    Steps: python scripts/paper_a/es_update_mapping_scope_fields.py --index rag_chunks_dev_current && python scripts/paper_a/es_backfill_scope_fields.py --index rag_chunks_dev_current --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl --dry-run && python scripts/paper_a/es_backfill_scope_fields.py --index rag_chunks_dev_current --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl && python scripts/paper_a/es_verify_scope_coverage.py --index rag_chunks_dev_current --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt
    Expected: exit 0; coverage >=0.99
    Evidence: .sisyphus/evidence/task-12-es-backfill.txt

  Scenario: Wrong index name
    Tool: Bash
    Steps: python scripts/paper_a/es_update_mapping_scope_fields.py --index does_not_exist
    Expected: exit !=0 and message mentions index not found
    Evidence: .sisyphus/evidence/task-12-es-backfill-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add optional ES scope field backfill` | Files: [`backend/llm_infrastructure/elasticsearch/mappings.py`, `scripts/paper_a/es_update_mapping_scope_fields.py`, `scripts/paper_a/es_backfill_scope_fields.py`, `scripts/paper_a/es_verify_scope_coverage.py`]

- [ ] 13. (Optional) Router prototypes + Matryoshka router + deterministic fallback

  **What to do**:
  - Implement `scripts/paper_a/build_device_prototypes.py`:
    - For each device in corpus meta, build a prototype text consisting of top-N topics and representative keywords.
    - Output `device_prototypes.jsonl`: `{device_name, text}`.
  - Implement `scripts/paper_a/router.py`:
    - Primary: Matryoshka-capable embed model (default `nomic-embed-text-v1.5`) for prototypes and queries.
    - Support `--dim {64,128,256,768}` by slicing embedding vectors (only meaningful for MRL-trained models).
    - Return Top-M devices (default M=3).
    - Fallback (required): if model load fails, use deterministic BM25/keyword overlap over prototype texts.
  - Wire evaluator systems P2/P3/P4 to call router when auto-parse finds no device.

  **Must NOT do**:
  - Do not leave router unimplemented; fallback must exist and be exercised in tests.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: model wiring + ablation controls
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 14 | Blocked By: 6,8,11

  **References**:
  - Router requirement: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (Section 2.3 + 3)

  **Acceptance Criteria**:
  - [ ] `python scripts/paper_a/build_device_prototypes.py ...` writes `device_prototypes.jsonl`
  - [ ] `python scripts/paper_a/router.py --self-test ...` demonstrates Top-M output
  - [ ] Evaluator can run `--systems P2` on masked set without crashing (even if fallback router used)

  **QA Scenarios**:
  ```
  Scenario: Router fallback works without model
    Tool: Bash
    Steps: python scripts/paper_a/router.py --prototypes .sisyphus/evidence/paper-a/policy/device_prototypes.jsonl --query "controller 교체 방법" --top-m 3 --force-fallback
    Expected: exit 0; prints 3 devices
    Evidence: .sisyphus/evidence/task-13-router-fallback.txt

  Scenario: Router model load fails gracefully
    Tool: Bash
    Steps: python scripts/paper_a/router.py --prototypes .sisyphus/evidence/paper-a/policy/device_prototypes.jsonl --query "controller" --model does_not_exist --top-m 3
    Expected: exit 0 using fallback; warns about model load
    Evidence: .sisyphus/evidence/task-13-router-model-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add device router with matryoshka+fallback` | Files: [`scripts/paper_a/build_device_prototypes.py`, `scripts/paper_a/router.py`]

- [ ] 14. (Optional) Run P2-P4 ablations and write paper-ready tables into docs evidence

  **What to do**:
  - Run evaluator on:
    - Masked set: systems P2,P3,P4 with router dim=64/128/256 and topM=1/3/5.
    - Ambiguous set: P3,P4 focus.
  - Save outputs under `.sisyphus/evidence/paper-a/runs/` with run ids.
  - Generate markdown tables (Tab-A1/Tab-A2 equivalents) and write into:
    - `docs/papers/20_paper_a_scope/evidence/2026-03-XX_paper_a_main_results.md`
    - `docs/papers/20_paper_a_scope/evidence/2026-03-XX_paper_a_matryoshka_ablation.md`
  - Update `docs/papers/20_paper_a_scope/evidence_mapping.md` rows with file pointers to these evidences.

  **Must NOT do**:
  - Do not commit `.sisyphus/evidence/*` artifacts.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: produce paper-facing tables and evidence pointers
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: none | Blocked By: 11,13

  **References**:
  - Evidence mapping template: `docs/papers/20_paper_a_scope/evidence_mapping.md`
  - Expected tables: `docs/papers/20_paper_a_scope/evidence_mapping.md` (Tab-A1/Tab-A2)

  **Acceptance Criteria**:
  - [ ] New evidence markdown files exist under `docs/papers/20_paper_a_scope/evidence/`
  - [ ] `docs/papers/20_paper_a_scope/evidence_mapping.md` updated with links/paths

  **QA Scenarios**:
  ```
  Scenario: Main results evidence generated
    Tool: Bash
    Steps: ls docs/papers/20_paper_a_scope/evidence/
    Expected: contains new paper_a_*_results.md files
    Evidence: .sisyphus/evidence/task-14-evidence-files.txt

  Scenario: Evidence mapping references missing file
    Tool: Bash
    Steps: python -c "from pathlib import Path; import re; p=Path('docs/papers/20_paper_a_scope/evidence_mapping.md').read_text(); missing=[m for m in re.findall(r'`([^`]+\.md)`', p) if not Path(m).exists()]; print(missing)"
    Expected: prints []
    Evidence: .sisyphus/evidence/task-14-evidence-missing.txt
  ```

  **Commit**: YES | Message: `docs(paper-a): add experiment evidence tables` | Files: [`docs/papers/20_paper_a_scope/evidence/*`, `docs/papers/20_paper_a_scope/evidence_mapping.md`]

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. End-to-End Eval Smoke (Explicit + Masked) — unspecified-high
- [ ] F4. Scope Fidelity Check (no manuscript/prod rollout creep) — deep

## Commit Strategy
- Prefer one commit per task group:
  - Wave 1: scaffolding + canonicalization + corpus meta
  - Wave 2: shared/scope + family + eval sets
  - Wave 3: scope filter + evaluator
  - Wave 4 (optional): ES backfill + router + docs evidence
- Commit messages use `feat(paper-a): ...` / `feat(retrieval): ...` / `docs(paper-a): ...`.

## Success Criteria
- The evaluator can reproduce Paper A’s required metrics and artifacts for Explicit/Masked/Ambiguous sets with a manifest-scoped corpus and deterministic policies.
- All runs are attributable (git sha + resolved index + config hash + seed) and rerunnable.
- Continuous reruns can detect and report corpus/policy drift without breaking on legitimate dataset growth.
