# chunk_v3 Embed + Ingest + Model Eval (Doc-Type Aware)

## TL;DR
> **Summary**: Fix `chunk_v3` pipeline blockers, implement doc-type-aware chunking (SOP/TS/Setup/MyService/GCB), generate embeddings for 3 models, ingest into Elasticsearch with safe mappings/guardrails, then run both smoke checks and formal SOP evaluation.
> **Deliverables**:
> - Executable CLI pipeline: VLM validate → chunk → embed → ingest → verify
> - Doc-type-aware chunking aligned to `docs/2026-03-04_doc_type_chunking_plan.md`
> - ES indices: `chunk_v3_content` + `chunk_v3_embed_{model}_v1` (3 models)
> - Smoke evidence (counts, sync, spot queries) + formal SOP metrics report
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: DocType Canonicalization → Chunkers → Embedding Engine Fix/Model Contract → ES Mapping+Ingest → Eval

## Context
### Original Request
- Review existing runbook `.omc/plans/chunk_v3_embed_ingest_plan.md` and generate an updated, executable plan using `docs/2026-03-04_doc_type_chunking_plan.md`.

### Interview Summary
- Scope: implement full doc-type-aware chunking now.
- Model comparison: include both smoke checks and formal metrics.

### Metis Review (gaps addressed)
- Metis consult was started but tool timed out; plan includes an explicit self-review guardrail set instead.
- Added guardrails:
  - Canonical `doc_type` normalization with alias handling to avoid taxonomy drift.
  - Hard gates at each stage (VLM coverage → chunk invariants → embedding invariants → ES sync).
  - ES mapping safety: `extra_meta` stored as disabled object + `dynamic: false` to prevent field explosion.
  - Model contract validation (dims/NaN/norm) before index creation to prevent silent vector incompatibility.

## Work Objectives
### Core Objective
- Produce an end-to-end, repeatable `chunk_v3` data build: VLM coverage gate → doc-type-aware chunking → multi-model embeddings → ES ingest (content + embed indices) → verification + evaluation.

### Deliverables
- Updated scripts under `scripts/chunk_v3/` to run reliably and deterministically.
- Safe ES mappings to prevent dynamic field explosion while preserving required metadata.
- Evaluation artifacts (CSV/JSON/MD) saved under `.sisyphus/evidence/`.
- Updated operational runbook `.omc/plans/chunk_v3_embed_ingest_plan.md` reflecting the fixed commands and new gates.

### Definition of Done (agent-executable)
- `python scripts/chunk_v3/validate_vlm.py --parsed-dir data/vlm_parsed --source-dir /home/llm-share/datasets/pe_agent_data/pe_preprocess_data --fail-on-mismatch` exits 0.
- `python normalize.py --data-root /home/llm-share/datasets/pe_agent_data/pe_preprocess_data --output data/chunk_v3_manifest.json` produces a JSON list file.
- `python scripts/chunk_v3/run_chunking.py --vlm-dir data/vlm_parsed --manifest data/chunk_v3_manifest.json --output data/chunks_v3/all_chunks.jsonl` produces JSONL with canonical `doc_type` and non-empty `content`.
- For each model in `{qwen3_emb_4b,bge_m3,jina_v5}`: `python scripts/chunk_v3/run_embedding.py ...` produces `embeddings_{model}.npy` and `chunk_ids_{model}.jsonl` with matching row counts.
- `python scripts/chunk_v3/run_ingest.py content --chunks data/chunks_v3/all_chunks.jsonl` completes without ES mapping errors.
- For each model: `python scripts/chunk_v3/run_ingest.py embed --model {model} --embeddings ... --chunk-ids ...` completes without ES mapping errors.
- For each model: `python scripts/chunk_v3/run_ingest.py verify --model {model}` exits 0.
- Smoke evaluation artifacts exist for all models (see tasks below).
- Formal SOP evaluation artifacts exist for all models using `docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv`.

### Must Have
- Canonical `doc_type` normalization (accept aliases as input; store canonical in outputs).
- No silent page-drop regressions: coverage gate enforced.
- Embedding dimensionality validated before index creation.
- ES ingest is idempotent (`_id=chunk_id`) and rerunnable.

### Embedding Model Contracts (locked)
- `qwen3_emb_4b`
  - HF: `Qwen/Qwen3-Embedding-4B`
  - License: Apache-2.0
  - Native dims: 2560 (MRL supports 32..2560)
  - Query encoding: SentenceTransformers `prompt_name="query"` (recommended by model card)
  - Document encoding: default encode (no query prompt)
  - Requirements: transformers>=4.51.0 (per model card)
- `bge_m3`
  - HF: `BAAI/bge-m3`
  - License: MIT
  - Dims: 1024
  - Query/document prefixes: none required (FAQ: no instruction needed for queries)
- `jina_v5`
  - HF: `jinaai/jina-embeddings-v5-text-small`
  - License: CC BY-NC 4.0 (non-commercial)
  - Dims: 1024 (Matryoshka 32..1024)
  - Pooling: last-token pooling (per model card)
  - Query/document encoding: `task="retrieval"` + `prompt_name="query"|"document"` (per model card)
  - Requirements: transformers>=4.57.0, peft>=0.15.2, trust_remote_code=True (per model card)

### Must NOT Have
- No ES dynamic field explosion from unbounded metadata keys.
- No “count must match” verification that is invalidated by planned skip policies (e.g., empty chunks) — rules must be explicit and consistent.
- No manual-only verification steps.
- No accidental production use of CC BY-NC model outputs (`jina_v5` is non-commercial; keep it evaluation-only unless business explicitly approves licensing).

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (pytest) + script-level smoke runs.
- QA policy: Every task includes at least 1 happy path + 1 failure/edge scenario.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
Wave 1: Canonicalization + gates + ES mapping guardrails + streaming/stats
Wave 2: Doc-type-aware chunkers (SOP/TS/Setup/MyService/GCB)
Wave 3: Embedding pipeline fixes + model contract validation + ES ingest updates
Wave 4: Evaluation (smoke + formal metrics) + runbook updates

### Dependency Matrix (full, all tasks)
- W1 tasks block all downstream.
- W2 depends on W1 canonical doc_type helpers + manifest contract.
- W3 depends on W1 ES mapping decisions + W2 chunk schema invariants.
- W4 depends on W3 indices existing + query embedding path validated.

### Agent Dispatch Summary
- Wave 1: 6 tasks (quick/unspecified-high)
- Wave 2: 5 tasks (unspecified-high)
- Wave 3: 5 tasks (unspecified-high)
- Wave 4: 4 tasks (writing + unspecified-high)

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Canonical DocType + Input Alias Normalization (Folders + Stored doc_type)

  **What to do**:
  - Define canonical `doc_type` enum for chunk_v3 stored outputs (ES + JSONL): `sop`, `ts`, `setup`, `myservice`, `gcb`.
  - Accept input aliases (folder names and meta labels) and normalize to canonical:
    - TS inputs: `trouble_shooting`, `trouble_shooting_guide`, `troubleshooting`, `t/s` → canonical `ts`.
    - Setup inputs: `setup_manual`, `set_up_manual`, `installation manual` → canonical `setup`.
  - Preserve original/source doc_type label for traceability:
    - Store `extra_meta.doc_type_raw` (or `extra_meta.source_doc_type`) before normalization.
  - Fix Setup Manual folder mismatch by making chunking read both `data/vlm_parsed/setup_manual/` and `data/vlm_parsed/set_up_manual/` (priority: canonical first, fallback to alias).
  - Ensure `scripts/chunk_v3/chunkers.py` stores canonical `doc_type` (remove `split('_')[0]` behavior).
  - Ensure all downstream (chunk_id format, ES filters, eval scripts) use canonical.

  **Must NOT do**:
  - Do not change existing dataset root paths under `/home/llm-share/...`.
  - Do not introduce a 2nd parallel doc_type taxonomy (one canonical only).

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: localized fixes + mechanical normalization.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2-19 | Blocked By: none

  **References**:
  - Runner doc_types: `scripts/chunk_v3/run_chunking.py` (VLM doc_type loop)
  - Buggy doc_type set: `scripts/chunk_v3/chunkers.py` (VLM chunker sets `doc_type`)
  - Existing grouping norms: `backend/domain/doc_type_mapping.py` (canonical bucket includes `setup` and variant `set_up_manual`)
  - Runbook mismatch: `.omc/plans/chunk_v3_embed_ingest_plan.md`

  **Acceptance Criteria**:
  - [ ] Running `python -c "from scripts.chunk_v3.chunkers import chunk_vlm_parsed; import json, tempfile;"`-style unit smoke (or pytest) shows Setup Manual chunks have `doc_type=='setup'`.
  - [ ] `python scripts/chunk_v3/run_chunking.py --vlm-dir data/vlm_parsed --skip-myservice --skip-gcb --output /tmp/all_chunks.jsonl` processes Setup Manual whether folder is `setup_manual` or `set_up_manual`.

  **QA Scenarios**:
  ```
  Scenario: Setup manual alias folder ingested
    Tool: Bash
    Steps:
      1) Create a minimal fixture VLM JSON under data/vlm_parsed/set_up_manual/x.json
      2) Run scripts/chunk_v3/run_chunking.py with --skip-myservice --skip-gcb
      3) Parse output JSONL and assert doc_type is "setup"
    Expected: chunk count > 0 and all setup chunks doc_type==setup
    Evidence: .sisyphus/evidence/task-1-doctype-alias.txt

  Scenario: Reject unknown doc_type
    Tool: Bash
    Steps:
      1) Provide a VLM JSON in an unexpected folder data/vlm_parsed/unknown/
      2) Run chunking
    Expected: unknown doc_type is skipped with explicit WARNING log (not silent)
    Evidence: .sisyphus/evidence/task-1-doctype-unknown.txt
  ```

  **Commit**: YES | Message: `fix(chunk_v3): canonicalize doc_type and accept setup_manual aliases` | Files: `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/chunkers.py`, (tests)

- [ ] 2. Enforce VLM Coverage Gate (No Page Drop Regressions)

  **What to do**:
  - Make `scripts/chunk_v3/validate_vlm.py` the required pre-step for chunking/ingest.
  - Add a small wrapper command or a `--validate-vlm` option to `scripts/chunk_v3/run_chunking.py` to run validation and fail fast when:
    - any document has `page_match==False`, OR
    - `coverage_ratio < 0.98`.
  - Ensure the validation report JSON is written to a stable location (`data/vlm_parsed/validation_report.json`).

  **Must NOT do**:
  - Do not block on “empty pages” alone; treat them as warnings (as long as parsed pages exist) unless the doc-type plan explicitly requires hard-fail.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: wire existing validator into runners.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6-19 | Blocked By: 1

  **References**:
  - Validator: `scripts/chunk_v3/validate_vlm.py`
  - Doc requirement: `docs/2026-03-04_doc_type_chunking_plan.md` (coverage_ratio guardrail)

  **Acceptance Criteria**:
  - [ ] If a fixture VLM JSON has missing page numbers, the gate exits non-zero.
  - [ ] Gate produces `data/vlm_parsed/validation_report.json`.

  **QA Scenarios**:
  ```
  Scenario: Missing pages blocks pipeline
    Tool: Bash
    Steps:
      1) Create a VLM JSON with total_pages=3 but only pages [1,3]
      2) Run validate_vlm with --fail-on-mismatch
    Expected: exit code != 0 and report contains MISSING_PAGES
    Evidence: .sisyphus/evidence/task-2-vlm-gate-missing.txt

  Scenario: Empty pages only warns
    Tool: Bash
    Steps:
      1) Create a VLM JSON with all pages present but page 2 text empty
      2) Run validate_vlm
    Expected: exit 0 (unless policy changed), report flags EMPTY_PAGES
    Evidence: .sisyphus/evidence/task-2-vlm-gate-empty.txt
  ```

  **Commit**: YES | Message: `feat(chunk_v3): gate chunking on VLM coverage validation` | Files: `scripts/chunk_v3/validate_vlm.py`, `scripts/chunk_v3/run_chunking.py`

- [ ] 3. Fix Manifest Contract + Update References (normalize.py path)

  **What to do**:
  - Standardize manifest generation command in runbook and scripts to repo-root `normalize.py` (not `scripts/chunk_v3/normalize.py`).
  - Ensure `scripts/chunk_v3/chunkers.py` manifest lookup uses `file_name` matching with `source_file` (already does via `_find_manifest_meta`).
  - Add a chunking-time warning if manifest is missing or a file has no manifest meta (meta_source should become `filename_parser`).

  **Must NOT do**:
  - Do not require manifest for MyService/GCB inputs (manifest is SOP/TS/Setup-oriented).

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: docs + small guardrails.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6-8,17-18 | Blocked By: 1

  **References**:
  - Manifest generator: `normalize.py` (outputs `data/chunk_v3_manifest.json`)
  - Manifest consumer: `scripts/chunk_v3/chunkers.py`
  - Operational runbook path (must stay in sync): `.omc/plans/chunk_v3_embed_ingest_plan.md`

  **Acceptance Criteria**:
  - [ ] Running `python normalize.py --stats-only` works and prints doc_type stats.
  - [ ] Chunk output includes `extra_meta.meta_source` = `manifest` or `filename_parser`.

  **QA Scenarios**:
  ```
  Scenario: Missing manifest falls back to filename parser
    Tool: Bash
    Steps:
      1) Run chunking with --manifest pointing to non-existent path
    Expected: chunking completes; meta_source=filename_parser in output
    Evidence: .sisyphus/evidence/task-3-manifest-fallback.txt

  Scenario: Present manifest sets topic/module
    Tool: Bash
    Steps:
      1) Generate manifest via normalize.py
      2) Chunk SOP/TS VLM JSON
    Expected: extra_meta contains module/topic for manifest-covered files
    Evidence: .sisyphus/evidence/task-3-manifest-applied.txt
  ```

  **Commit**: YES | Message: `docs(chunk_v3): correct manifest generation path and add fallback logging` | Files: `.omc/plans/chunk_v3_embed_ingest_plan.md`, `scripts/chunk_v3/chunkers.py`

- [ ] 4. ES Mapping Guardrails for chunk_v3 (Prevent Dynamic Field Explosion)

  **What to do**:
  - Update `backend/llm_infrastructure/elasticsearch/mappings.py` chunk_v3 mappings to explicitly store `extra_meta` as a bounded field:
    - `extra_meta`: `{ "type": "object", "enabled": false }` (stored, not indexed).
  - Ensure `scripts/chunk_v3/run_ingest.py` does NOT flatten `extra_meta` to top-level fields.
  - Decide and enforce top-level `dynamic` policy for chunk_v3 indices:
    - Set `dynamic: false` at root to prevent accidental new fields.
  - Add an ES safety limit in settings when creating chunk_v3 indices:
    - `index.mapping.total_fields.limit` set to a conservative value (e.g., 2000) to surface mapping blowups early.
  - Optimize bulk ingest performance:
    - Temporarily set index `refresh_interval` to `-1` during bulk ingest and restore to `1s` at end.
  - Add an ingest-time guard: if a chunk contains unexpected top-level keys, fail with an explicit error before bulk.

  **Must NOT do**:
  - Do not remove required filter fields already in mapping (`doc_type`, `device_name`, `equip_id`, `chapter`, `content_hash`).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ES mapping + ingest contract.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 13-15,16-18 | Blocked By: 1

  **References**:
  - Current content mapping: `backend/llm_infrastructure/elasticsearch/mappings.py` (chunk_v3 content mapping)
  - Ingest flattening: `scripts/chunk_v3/run_ingest.py` (`ingest_content` flattens extra_meta)
  - ES dense_vector docs: `https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector`

  **Acceptance Criteria**:
  - [ ] Creating `chunk_v3_content` index with updated mapping succeeds.
  - [ ] Ingesting content chunks with large/variable `extra_meta` keys does not create new ES fields.

  **QA Scenarios**:
  ```
  Scenario: extra_meta does not expand mapping
    Tool: Bash
    Steps:
      1) Create a chunk with extra_meta containing 50 unique keys
      2) Ingest content
      3) Fetch mapping and confirm those keys are not present as top-level fields
    Expected: mapping has only "extra_meta" field; no dynamic keys
    Evidence: .sisyphus/evidence/task-4-es-mapping-guardrail.json

  Scenario: Unexpected top-level key fails fast
    Tool: Bash
    Steps:
      1) Inject a chunk JSONL line with a top-level field not in ChunkV3Document
      2) Run ingest content
    Expected: script exits non-zero with clear error message
    Evidence: .sisyphus/evidence/task-4-es-mapping-unexpected.txt
  ```

  **Commit**: YES | Message: `fix(es): store chunk_v3 extra_meta safely and disable dynamic mapping` | Files: `backend/llm_infrastructure/elasticsearch/mappings.py`, `scripts/chunk_v3/run_ingest.py`

- [ ] 5. Fix run_ingest CLI Contracts + Idempotent Index Creation

  **What to do**:
  - Align CLI with runbook and make it explicit:
    - `verify` must require `--model` (keep) and runbook must include it.
    - Add `--content-index` and `--embed-index` overrides (default to existing names).
    - Add `--recreate` flag to delete and recreate target index (safe for dev).
  - Replace “scan content index for meta join” during embed ingest with local join:
    - Embed ingest takes `--chunks data/chunks_v3/all_chunks.jsonl` and builds a dict `chunk_id -> {doc_type, device_name, chapter, content_hash}`.
  - Ensure ingest creates indices using `backend/llm_infrastructure/elasticsearch/mappings.py` (or `EsIndexManager`).

  **Must NOT do**:
  - Do not require content index to exist before embedding ingest if `--chunks` is provided.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: operational correctness + performance.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 13-19 | Blocked By: 4

  **References**:
  - Current ingest + verify: `scripts/chunk_v3/run_ingest.py`
  - Index manager utilities: `backend/llm_infrastructure/elasticsearch/manager.py`

  **Acceptance Criteria**:
  - [ ] `python scripts/chunk_v3/run_ingest.py verify --model bge_m3` works and is documented in runbook.
  - [ ] Embed ingest no longer does `scan()` over ES for content metadata.

  **QA Scenarios**:
  ```
  Scenario: verify requires --model
    Tool: Bash
    Steps:
      1) Run: python scripts/chunk_v3/run_ingest.py verify
    Expected: argparse error (exit!=0) stating --model is required
    Evidence: .sisyphus/evidence/task-5-verify-requires-model.txt

  Scenario: embed ingest succeeds without content index
    Tool: Bash
    Steps:
      1) Ensure content index is deleted
      2) Run embed ingest with --chunks path to all_chunks.jsonl
    Expected: embed index created and documents indexed
    Evidence: .sisyphus/evidence/task-5-embed-ingest-no-content.txt
  ```

  **Commit**: YES | Message: `feat(chunk_v3): make ingest CLI explicit and remove ES scan joins` | Files: `scripts/chunk_v3/run_ingest.py`, `.omc/plans/chunk_v3_embed_ingest_plan.md`

- [ ] 6. SOP Chunker v3 (Heading/Step Aware + Token Window Fallback)

  **What to do**:
  - Implement SOP chunking aligned to `docs/2026-03-04_doc_type_chunking_plan.md` using a deterministic heuristic pipeline:
    - Input unit: VLM page text (`pages[i].text`) (do not merge across pages in v1).
    - Pre-clean: normalize Windows newlines → `\n`, strip trailing spaces, preserve blank lines.
    - Block split: split page text into paragraph blocks separated by 1+ blank lines.
    - Block classification:
      - `heading` if block is a single line and matches any:
        - `^STEP\s*\d+` (case-insensitive)
        - `^\d+(?:\.\d+)*\s+` (numbered heading)
        - Korean section headers: `^(목적|범위|준비물|절차|주의|주의사항|경고|참고|점검|교체|조정)\b`
      - `table_like` if block has >=2 lines and either:
        - at least 2 lines with `|` count >= 2, OR
        - at least 2 lines with 2+ occurrences of 2+ spaces between tokens (column-ish).
      - `toc_like` if block has >=8 short lines and many dot leaders (e.g., `....`) OR high ratio of lines ending with numbers.
    - Section building:
      - Start a new section when a `heading` block appears.
      - Append subsequent blocks until next heading.
      - Never split inside a `table_like` block.
    - Chunk packing within a page:
      - Token estimate default: `len(text.split())`.
      - Target window: 350-700 estimated tokens.
      - Overlap: 80 estimated tokens (carry trailing content into next chunk).
      - If a single section exceeds max: sub-split with a fixed window while preserving overlap.
    - Caption/warning merge policy (page-local, to preserve page-hit determinism):
      - If a block is very short (<=120 chars) and matches `^(주의|경고|WARNING|CAUTION|NOTE)\b` OR looks like an image caption (`^Figure\b|^Fig\.|^그림\b`), append it to the immediately previous non-TOC chunk within the SAME page.
      - If such block appears at start of a page (no previous chunk in page), keep it as its own chunk with `extra_meta.caption_or_warning=true`.
      - Do NOT merge across pages (keeps `page` field single-valued for evaluation).
    - Search text policy:
      - For `toc_like` blocks, exclude them from `search_text` (keep in `content` but set `extra_meta.is_toc=true`).
      - For non-TOC, `search_text == content`.
  - Preserve metadata: page_no/slide_no, section_title, language, work_type/module/topic when available.
  - Ensure deterministic chunk_id sequencing per doc_id.

  **Must NOT do**:
  - Do not emit empty content chunks.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: chunking heuristics + regressions risk.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 11-19 | Blocked By: 1-3

  **References**:
  - Current VLM chunker: `scripts/chunk_v3/chunkers.py` (`chunk_vlm_parsed`)
  - Chunking strategy: `docs/2026-03-04_doc_type_chunking_plan.md` (SOP section)

  **Acceptance Criteria**:
  - [ ] SOP chunks include `extra_meta.section_title` when detectable.
  - [ ] SOP chunk sizes respect target window by token estimate (exceptions only for table_like blocks or very small pages).

  **QA Scenarios**:
  ```
  Scenario: SOP step boundaries preserved
    Tool: Bash
    Steps:
      1) Create a synthetic SOP VLM JSON page with STEP headings and a table block
      2) Run chunk_vlm_parsed for sop
    Expected: chunks do not split inside the table block; STEP headings start chunks
    Evidence: .sisyphus/evidence/task-6-sop-step-chunking.json

  Scenario: No headings -> fallback windowing
    Tool: Bash
    Steps:
      1) Provide long plain text with no headings
      2) Chunk
    Expected: multiple chunks with overlap; deterministic boundaries
    Evidence: .sisyphus/evidence/task-6-sop-fallback.json
  ```

  **Commit**: YES | Message: `feat(chunk_v3): SOP heading/step-aware chunking with safe fallback` | Files: `scripts/chunk_v3/chunkers.py`, (tests)

- [ ] 7. TS Chunker v3 (Symptom/Cause/Action/Result + Log Excerpt subtype)

  **What to do**:
  - Implement TS chunking:
    - Input unit: VLM page text (page-local chunking only).
    - Section detection (case-insensitive, Korean/English):
      - symptom: `^(증상|symptom)`
      - cause: `^(원인|cause)`
      - action: `^(조치|action|solution|countermeasure)`
      - result: `^(결과|result)`
    - Alarm code detection: capture first match of patterns like `alarm\s*[(:]?\s*(\d{3,8})\b` and `error\s*code\s*[:#]?\s*(\w+)`.
    - Log excerpt detection:
      - A block is `log_like` if it has >=5 lines and >=3 lines match timestamp-ish patterns (e.g., `\d{4}-\d{2}-\d{2}` or `\d{2}:\d{2}:\d{2}`) OR has high ratio of `=`/`-` separators.
      - Emit `extra_meta.chunk_subtype=log_excerpt` for those chunks.
    - Packing:
      - Target 300-650 estimated tokens; overlap 60.
      - Ensure any detected alarm code token appears at the start of each section chunk (prepend `ALARM {code}` line).

  **Must NOT do**:
  - Do not drop alarm codes from the visible chunk text; keep at chunk start.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 11-19 | Blocked By: 1-3

  **References**:
  - Strategy: `docs/2026-03-04_doc_type_chunking_plan.md` (TS section)
  - Current VLM chunker: `scripts/chunk_v3/chunkers.py`

  **Acceptance Criteria**:
  - [ ] Chunks with log dumps have `extra_meta.chunk_subtype=="log_excerpt"`.
  - [ ] Alarm code extraction populates `extra_meta.alarm_code` when present.

  **QA Scenarios**:
  ```
  Scenario: TS log excerpt isolated
    Tool: Bash
    Steps:
      1) Provide TS text containing a long log block
      2) Chunk
    Expected: at least one chunk_subtype=log_excerpt
    Evidence: .sisyphus/evidence/task-7-ts-log.json

  Scenario: Alarm code preserved
    Tool: Bash
    Steps:
      1) Provide TS text with alarm(123456)
      2) Chunk
    Expected: alarm_code=123456 and chunk content starts with alarm token
    Evidence: .sisyphus/evidence/task-7-ts-alarm.json
  ```

  **Commit**: YES | Message: `feat(chunk_v3): TS section-aware chunking and log excerpt subtype` | Files: `scripts/chunk_v3/chunkers.py`, (tests)

- [ ] 8. Setup Manual Chunker v3 (Heading-aware + Sequence Preservation)

  **What to do**:
  - Implement Setup Manual chunking:
    - Input unit: VLM page text.
    - Heading markers (case-insensitive):
      - `^(chapter|unit|section)\s+\d+` and Korean: `^(장|챕터|유닛|절)\s*\d+`
      - procedure step markers: `^STEP\s*\d+` or `^\d+\)`.
    - Segment into sections at headings; then pack to 350-750 estimated tokens with overlap 80.
    - Attach `extra_meta.sequence_no` as the 1-based order of chunks within the document (stable across reruns).
  - Ensure canonical `doc_type` stored as `setup`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 11-19 | Blocked By: 1-3

  **References**:
  - Strategy: `docs/2026-03-04_doc_type_chunking_plan.md` (Setup Manual section)
  - Current VLM chunker: `scripts/chunk_v3/chunkers.py`

  **Acceptance Criteria**:
  - [ ] setup chunks include increasing `extra_meta.sequence_no` starting from 1.

  **QA Scenarios**:
  ```
  Scenario: sequence_no preserved
    Tool: Bash
    Steps:
      1) Provide a multi-section setup manual text
      2) Chunk
    Expected: sequence_no monotonically increases; no duplicates per doc_id
    Evidence: .sisyphus/evidence/task-8-setup-sequence.json
  ```

  **Commit**: YES | Message: `feat(chunk_v3): setup manual heading-aware chunking with sequence meta` | Files: `scripts/chunk_v3/chunkers.py`, (tests)

- [ ] 9. MyService Chunker v3 (Section-Specific Multi-Chunk + Meta)

  **What to do**:
  - Replace single-chunk MyService output with section-specific chunks:
    - Split into `section=status|action|cause|result` first.
    - Then sub-split long sections to meet recommended sizes:
      - status: 200-450 tokens
      - action: 300-700 tokens
      - cause/result: 80-250 tokens (prefer single chunk)
  - Ensure `chunk_id` increments per produced chunk.
  - Keep `search_text` weighting (title + cause + status first).
  - Emit `extra_meta.sections_present.*` and `extra_meta.completeness`.

  **Must NOT do**:
  - Do not merge different sections into a single chunk.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 11-19 | Blocked By: 1

  **References**:
  - Current parser: `scripts/chunk_v3/chunkers.py` (`_parse_myservice_txt`, `chunk_myservice`)
  - Strategy: `docs/2026-03-04_doc_type_chunking_plan.md` (MyService section)

  **Acceptance Criteria**:
  - [ ] For a fixture MyService TXT with all sections, output includes >=4 chunks with correct `extra_meta.section`.
  - [ ] completeness==empty still results in 0 output chunks.

  **QA Scenarios**:
  ```
  Scenario: Multi-chunk per section
    Tool: Bash
    Steps:
      1) Create a MyService TXT fixture with long action section
      2) Run chunk_myservice
    Expected: action split into multiple chunks; cause/result remain single if small
    Evidence: .sisyphus/evidence/task-9-myservice-multichunk.json

  Scenario: completeness empty skipped
    Tool: Bash
    Steps:
      1) Create meta completeness=empty
      2) Run chunk_myservice
    Expected: zero chunks
    Evidence: .sisyphus/evidence/task-9-myservice-empty.json
  ```

  **Commit**: YES | Message: `feat(chunk_v3): myservice section-aware multi-chunking` | Files: `scripts/chunk_v3/chunkers.py`, (tests)

- [ ] 10. GCB Chunker v3 (Record Summary + Section-aware Detail Chunks)

  **What to do**:
  - Implement 2-tier chunking per record:
    - Summary chunk: `Title + Request_Item2 + Status + Model Name`.
    - Detail chunks:
      - First split `Content` by section keywords (case-insensitive):
        - `\bDescription\b`, `\bCause\b`, `\bResult\b`, `\bBackground\b`, `\bRequest\b`, plus Korean fallbacks `설명|원인|결과|요청|배경`.
      - Then split long timelines by date boundaries matching `\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b`.
      - Finally window to 300-700 estimated tokens with overlap 60.
  - Preserve GCB meta: gcb_number, status, request_type, equip_id, model_name.
  - Keep deterministic chunk ordering and `chunk_of` counts.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 11-19 | Blocked By: 1

  **References**:
  - Current fixed-size splitter: `scripts/chunk_v3/chunkers.py` (`chunk_gcb`)
  - Strategy: `docs/2026-03-04_doc_type_chunking_plan.md` (GCB section)

  **Acceptance Criteria**:
  - [ ] For a fixture GCB entry, output includes exactly 1 summary chunk and >=1 detail chunks.
  - [ ] Summary chunk is identifiable via `extra_meta.chunk_tier=="summary"` (or equivalent decided field).

  **QA Scenarios**:
  ```
  Scenario: Summary + detail chunks
    Tool: Bash
    Steps:
      1) Create a small GCB JSON array fixture with Content containing "Cause:" and "Result:"
      2) Run chunk_gcb
    Expected: first chunk tier=summary; subsequent tier=detail and preserve sections
    Evidence: .sisyphus/evidence/task-10-gcb-tiered.json
  ```

  **Commit**: YES | Message: `feat(chunk_v3): gcb tiered and section-aware chunking` | Files: `scripts/chunk_v3/chunkers.py`, (tests)

- [ ] 11. Fix run_embedding Batch API + Add Model Contract Validation

  **What to do**:
  - Fix `scripts/chunk_v3/run_embedding.py` `MODEL_CONFIGS` to match model card contracts (notably `qwen3_emb_4b` dims=2560; `jina_v5` requires task/prompt and trust_remote_code).
  - Fix the batch embedding call:
    - Use `SentenceTransformerEmbedder.embed_batch()` (or `encode()`), not `.embed()`.
  - Add a “model contract validation” step executed before full embedding:
    - Load model
    - Compute embedding for 2 strings
    - Assert dims match the locked model contract:
      - qwen3_emb_4b: 2560 (unless explicitly using MRL truncation)
      - bge_m3: 1024
      - jina_v5: 1024
    - Assert no NaN/Inf
    - Assert norms are ~1.0 when normalize==l2
  - Implement per-model loader strategy:
    - Extend `backend/llm_infrastructure/embedding/engines/sentence/embedder.py` to optionally pass `trust_remote_code=True` into SentenceTransformer/transformers stack when configured.
      - Add a constructor flag `trust_remote_code: bool = False` and plumb it into SentenceTransformer initialization (or underlying AutoModel loading if ST path doesn’t support).
    - First try SentenceTransformer path.
    - If ST path fails or model is not ST-compatible, fallback to Transformers mean-pooling over last hidden state with attention mask, then L2 normalize.
  - Ensure evaluation uses the same embedding code path as indexing:
    - Expose a helper in `scripts/chunk_v3/run_embedding.py` to embed queries with `query_prefix` and same normalization.
    - Alternatively, wire `backend/services/embedding_service.py` (registry-based) for both indexing and evaluation, but only if it can be configured per model_key deterministically.

  **Must NOT do**:
  - Do not silently proceed when dims mismatch; fail with actionable error.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: model APIs vary; failure modes common.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 12-19 | Blocked By: 6-10

  **References**:
  - Embedding script: `scripts/chunk_v3/run_embedding.py`
  - Engine signature: `backend/llm_infrastructure/embedding/engines/sentence/embedder.py`
  - Model configs: `scripts/chunk_v3/run_embedding.py` (`MODEL_CONFIGS`)
  - Model cards:
    - `https://huggingface.co/Qwen/Qwen3-Embedding-4B`
    - `https://huggingface.co/BAAI/bge-m3`
    - `https://huggingface.co/jinaai/jina-embeddings-v5-text-small`

  **Acceptance Criteria**:
  - [ ] Running `python scripts/chunk_v3/run_embedding.py --chunks ... --models bge_m3 --device cpu --batch-size 2` completes.
  - [ ] For each model, validation runs and logs dims + norm range; dims match the model card contract.

  **QA Scenarios**:
  ```
  Scenario: Batch embedding works
    Tool: Bash
    Steps:
      1) Create a tiny chunks JSONL (3 lines)
      2) Run run_embedding.py with batch-size=2
    Expected: embeddings saved with shape (3,dims)
    Evidence: .sisyphus/evidence/task-11-embed-batch.txt

  Scenario: Dims mismatch fails fast
    Tool: Bash
    Steps:
      1) Temporarily set MODEL_CONFIGS dims to wrong value in a test
      2) Run validation
    Expected: exits non-zero with message including expected vs actual dims
    Evidence: .sisyphus/evidence/task-11-embed-dims-mismatch.txt
  ```

  **Commit**: YES | Message: `fix(chunk_v3): batch embedding API and validate model contracts` | Files: `scripts/chunk_v3/run_embedding.py`, (tests)

- [ ] 12. Embedding Artifact Invariants Checker (Offline)

  **What to do**:
  - Add a `scripts/chunk_v3/check_embeddings.py` (or subcommand) that verifies:
    - `len(chunk_ids)==N` equals embeddings rows
    - dims match config
    - no NaN/Inf
    - norm distribution reasonable (min/max on sample)

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 16-19 | Blocked By: 11

  **References**:
  - Output artifacts: `data/chunks_v3/embeddings_{model}.npy`, `data/chunks_v3/chunk_ids_{model}.jsonl`

  **Acceptance Criteria**:
  - [ ] Checker exits 0 on valid artifacts and non-zero on corrupted artifacts.

  **QA Scenarios**:
  ```
  Scenario: Detect NaN
    Tool: Bash
    Steps:
      1) Create a small npy containing NaN
      2) Run checker
    Expected: non-zero exit and explicit NaN error
    Evidence: .sisyphus/evidence/task-12-embed-nan.txt
  ```

  **Commit**: YES | Message: `feat(chunk_v3): add offline embedding invariants checker` | Files: `scripts/chunk_v3/check_embeddings.py`

- [ ] 13. Content Ingest Hardening (Mapping, recreate, strict schema)

  **What to do**:
  - Update `scripts/chunk_v3/run_ingest.py content` to:
    - Create index with updated chunk_v3 content mapping (Task 4)
    - Optionally recreate
    - Index `extra_meta` as a single stored field
    - Use bulk with retry/backoff for transient errors
  - Ensure `created_at` is populated (mapping expects it).

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 14-19 | Blocked By: 4-5,6-10

  **References**:
  - Mapping: `backend/llm_infrastructure/elasticsearch/mappings.py` (`get_chunk_v3_content_mapping`)
  - Ingest script: `scripts/chunk_v3/run_ingest.py` (`ingest_content`)

  **Acceptance Criteria**:
  - [ ] Content ingest indexes exactly `wc -l all_chunks.jsonl` documents (minus any explicitly skipped policy if defined).
  - [ ] No ES mapping errors in bulk response.

  **QA Scenarios**:
  ```
  Scenario: Recreate index
    Tool: Bash
    Steps:
      1) Run ingest content with --recreate twice
    Expected: second run succeeds and doc count stable
    Evidence: .sisyphus/evidence/task-13-content-recreate.txt
  ```

  **Commit**: YES | Message: `feat(chunk_v3): harden content ingest with safe meta storage` | Files: `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/elasticsearch/mappings.py`

- [ ] 14. Embed Ingest Hardening (Local meta join + dims safety)

  **What to do**:
  - Update `scripts/chunk_v3/run_ingest.py embed` to:
    - Create embed index with dims from validated `MODEL_CONFIGS`
    - Join required metadata from `--chunks` local file, not ES scan
    - Stream bulk actions without materializing full `vec.tolist()` in memory where possible (batch chunks)
  - Ensure embed docs include: chunk_id, embedding, content_hash, doc_type, device_name, chapter.
  - ES constraints:
    - `dense_vector.dims` must be <= 4096 (Qwen3-Embedding-4B at 2560 is OK).
    - With `similarity="cosine"`, ES normalizes indexed vectors; still keep L2 normalization to reduce risk of inconsistent scoring.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 15-19 | Blocked By: 11-13

  **References**:
  - Embed mapping: `backend/llm_infrastructure/elasticsearch/mappings.py` (`get_chunk_v3_embed_mapping`)
  - Ingest script: `scripts/chunk_v3/run_ingest.py` (`ingest_embeddings`)

  **Acceptance Criteria**:
  - [ ] For each model, embed index doc count equals embedding rows.
  - [ ] No ES `dense_vector` dims errors.

  **QA Scenarios**:
  ```
  Scenario: Dims mismatch fails
    Tool: Bash
    Steps:
      1) Create embed index with dims=768 but attempt ingest 1024 vectors
    Expected: script exits non-zero and prints ES error
    Evidence: .sisyphus/evidence/task-14-embed-dims-error.txt
  ```

  **Commit**: YES | Message: `feat(chunk_v3): harden embed ingest and metadata join` | Files: `scripts/chunk_v3/run_ingest.py`

- [ ] 15. Sync Verification Policy (Counts + ID Set + Sample content_hash match)

  **What to do**:
  - Update `verify_sync()` to:
    - Compare counts content vs embed
    - Compare ID sets (full for <=200k, else sample)
    - Sample 200 chunk_ids and ensure `content_hash` matches between indices
  - Ensure verify output is machine-readable JSON in addition to logs.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 16-19 | Blocked By: 13-14

  **References**:
  - Current verify: `scripts/chunk_v3/run_ingest.py` (`verify_sync`)

  **Acceptance Criteria**:
  - [ ] `python scripts/chunk_v3/run_ingest.py verify --model bge_m3` exits 0 when synced and exits 1 when mismatched.

  **QA Scenarios**:
  ```
  Scenario: Missing ids detected
    Tool: Bash
    Steps:
      1) Ingest content fully
      2) Ingest only first N embeddings
      3) Run verify
    Expected: FAIL with missing ids count
    Evidence: .sisyphus/evidence/task-15-verify-missing.txt
  ```

  **Commit**: YES | Message: `feat(chunk_v3): strengthen sync verification with hash sampling` | Files: `scripts/chunk_v3/run_ingest.py`

- [ ] 16. Smoke Model Comparison (kNN spot queries + artifact export)

  **What to do**:
  - Implement a script `scripts/chunk_v3/smoke_eval.py` that for each model:
    - Selects 20 queries (first 20 from SOP CSV `question` column) + 5 GCB/MyService hand-picked queries
    - Computes query embedding using the SAME embedder config as indexing (Task 11)
    - Executes ES kNN search on `chunk_v3_embed_{model}_v1` (use `EsSearchEngine.dense_search` or `EsVectorDB.search` patterns)
    - Fetches matching content docs from `chunk_v3_content` and exports top-k with doc_id/page/doc_type
  - Write evidence outputs per model.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ES query correctness + embedding parity.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 17-18 | Blocked By: 14-15

  **References**:
  - SOP query source: `docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv`
  - ES host config: `backend/config/settings.py` (`SearchSettings.es_host`)

  **Acceptance Criteria**:
  - [ ] Script produces per-model JSONL/CSV of top-k results.

  **QA Scenarios**:
  ```
  Scenario: Smoke eval runs for all models
    Tool: Bash
    Steps:
      1) Run smoke_eval.py with --models all
    Expected: 3 output files exist and contain >=20*top_k rows
    Evidence: .sisyphus/evidence/task-16-smoke-eval-files.txt
  ```

  **Commit**: YES | Message: `feat(chunk_v3): add smoke model evaluation script` | Files: `scripts/chunk_v3/smoke_eval.py`

- [ ] 17. Formal SOP Evaluation (page-hit@k + hit@k per model)

  **What to do**:
  - Implement `scripts/chunk_v3/eval_sop_questionlist.py`:
    - Input: `docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv`
    - For each model: query embedding (same as indexing) -> knn -> compute metrics:
      - hit@{1,3,5,10}
      - page-hit@{1,3,5,10} using expected_pages ranges
      - rank distribution
    - Export summary markdown + detailed per-row CSV

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 18 | Blocked By: 16

  **References**:
  - Evidence CSV: `docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv`
  - Existing page-hit parsing patterns: `scripts/evaluation/evaluate_sop_agent_page_hit.py` (`_normalize_doc_id`, `_parse_page_range`, `_in_range`)

  **Acceptance Criteria**:
  - [ ] Produces `.sisyphus/evidence/task-17-sop-eval-summary.md` and `.sisyphus/evidence/task-17-sop-eval-rows.csv`.

  **QA Scenarios**:
  ```
  Scenario: Metrics computed deterministically
    Tool: Bash
    Steps:
      1) Run eval twice with deterministic=true
    Expected: summary metrics identical between runs
    Evidence: .sisyphus/evidence/task-17-sop-eval-determinism.txt
  ```

  **Commit**: YES | Message: `feat(chunk_v3): add formal SOP evaluation against questionlist CSV` | Files: `scripts/chunk_v3/eval_sop_questionlist.py`

- [ ] 18. Update Operational Runbook (.omc) to Match Fixed Pipeline

  **What to do**:
  - Rewrite `.omc/plans/chunk_v3_embed_ingest_plan.md` to match the corrected commands and new gates:
    - correct Setup folder naming/alias behavior
    - include `verify --model`
    - include embedding model validation + checker
    - include smoke + formal evaluation commands
  - Ensure all shell snippets are runnable (valid JSON bodies, no Python-only syntax in curl).

  **Recommended Agent Profile**:
  - Category: `writing`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: none | Blocked By: 15-17

  **References**:
  - Existing runbook: `.omc/plans/chunk_v3_embed_ingest_plan.md`
  - Source doc: `docs/2026-03-04_doc_type_chunking_plan.md`
  - Scripts: `scripts/chunk_v3/*`

  **Acceptance Criteria**:
  - [ ] Running each code block in order works in a clean environment (documented prerequisites included).

  **QA Scenarios**:
  ```
  Scenario: Runbook command audit
    Tool: Bash
    Steps:
      1) Execute each runbook command up to the first safe dry-run stage (validate/generate manifest)
    Expected: no syntax errors; commands match actual argparse
    Evidence: .sisyphus/evidence/task-18-runbook-audit.txt
  ```

  **Commit**: YES | Message: `docs(chunk_v3): update embed+ingest runbook to match fixed pipeline` | Files: `.omc/plans/chunk_v3_embed_ingest_plan.md`

- [ ] 19. Add Pytest Coverage for chunk_v3 Critical Contracts

  **What to do**:
  - Add tests under `backend/tests/` that validate:
    - doc_type normalization produces canonical values
    - setup aliases (`setup_manual`, `set_up_manual`) normalize to canonical `setup` consistently (assert canonical contract)
    - MyService multi-chunk behavior
    - GCB tiered chunking behavior
    - run_embedding uses batch API (no `.embed(list)`)
    - ES mapping dict contains `extra_meta` as disabled object and `dynamic: false`

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: Final verification | Blocked By: 1-15

  **References**:
  - Existing chunking tests pattern: `backend/tests/test_chunking.py`
  - Doc type mapping: `backend/domain/doc_type_mapping.py`

  **Acceptance Criteria**:
  - [ ] `pytest -q` passes.

  **QA Scenarios**:
  ```
  Scenario: Regression test for setup doc_type
    Tool: Bash
    Steps:
      1) Run pytest -k setup
    Expected: passes and asserts canonical doc_type behavior
    Evidence: .sisyphus/evidence/task-19-pytest.txt
  ```

  **Commit**: YES | Message: `test(chunk_v3): add regression tests for chunking/embedding/ES mapping contracts` | Files: `backend/tests/test_chunk_v3_*.py`

- [ ] 20. Streaming Chunk Output + Stage Accounting (raw→parsed→chunk→embed→index)

  **What to do**:
  - Refactor `scripts/chunk_v3/run_chunking.py` to stream-write JSONL (do not accumulate `all_chunks` in RAM):
    - Write output incrementally as each doc/file is processed.
    - Also write a stats JSON: `data/chunks_v3/chunking_stats.json` containing:
      - per doc_type: input_docs, parsed_units(pages), output_chunks, skipped_empty_chunks
      - totals + runtime seconds
      - explicit skip reasons counts (empty_text, completeness_empty, etc.)
  - Add a `--stats-only` mode to compute counts without writing full JSONL (fast sanity).
  - Ensure chunk IDs are deterministic across streaming order:
    - Sort inputs deterministically (already done for VLM JSON glob and myservice glob).

  **Must NOT do**:
  - Do not change chunk_id format (`{doc_type}_{doc_id}#{index:04d}` from `scripts/chunk_v3/common.py`).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: correctness + performance + determinism.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 11-18 | Blocked By: 1-3

  **References**:
  - Current chunking orchestrator (RAM-accumulating): `scripts/chunk_v3/run_chunking.py`
  - Chunk ID generator: `scripts/chunk_v3/common.py` (`generate_chunk_id`)

  **Acceptance Criteria**:
  - [ ] `python scripts/chunk_v3/run_chunking.py ...` completes for MyService without OOM on typical dataset size.
  - [ ] `data/chunks_v3/chunking_stats.json` exists and totals match `wc -l data/chunks_v3/all_chunks.jsonl`.

  **QA Scenarios**:
  ```
  Scenario: Deterministic stats + counts
    Tool: Bash
    Steps:
      1) Run chunking twice with same inputs
      2) Compare chunking_stats totals and first 100 chunk_ids
    Expected: identical totals; identical first 100 chunk_ids
    Evidence: .sisyphus/evidence/task-20-streaming-determinism.txt
  ```

  **Commit**: YES | Message: `perf(chunk_v3): stream chunk output and emit stage accounting stats` | Files: `scripts/chunk_v3/run_chunking.py`

- [ ] 21. Formal Eval Ground-Truth Mapping (Use matched_doc_id + expected_pages)

  **What to do**:
  - In `scripts/chunk_v3/eval_sop_questionlist.py`, define ground truth mapping rules explicitly:
    - `expected_doc_id` = CSV column `matched_doc_id` (preferred) when present/non-empty.
    - Fallback: derive from `expected_doc` file name using the same normalization rules as VLM doc_id generation.
    - `expected_pages` parse:
      - Accept `"a-b"` (inclusive range) and comma-separated ranges.
  - Document this in the report header to avoid ambiguity.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: none | Blocked By: 17

  **References**:
  - Evidence CSV columns: `docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv`
  - Page-range parsing reference: `scripts/evaluation/evaluate_sop_agent_page_hit.py` (`_parse_page_range`)

  **Acceptance Criteria**:
  - [ ] 100% of rows resolve an `expected_doc_id` and parse expected_pages without crashing.

  **QA Scenarios**:
  ```
  Scenario: expected_doc_id resolved
    Tool: Bash
    Steps:
      1) Run eval in a "dry" mode that only parses CSV
    Expected: reports unresolved rows == 0
    Evidence: .sisyphus/evidence/task-21-eval-gt-resolution.txt
  ```

  **Commit**: YES | Message: `fix(eval): define SOP ground-truth mapping rules for chunk_v3 evaluation` | Files: `scripts/chunk_v3/eval_sop_questionlist.py`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Use small, reviewable commits by concern: doc_type/gates → chunkers → embedding → ingest/mapping → evaluation → docs.

## Success Criteria
- End-to-end run produces ES indices with verified sync for all 3 models.
- Chunking aligns with `docs/2026-03-04_doc_type_chunking_plan.md` intent (section/heading-aware + deterministic fallback).
- Evaluation reports show model-by-model performance and are reproducible.
