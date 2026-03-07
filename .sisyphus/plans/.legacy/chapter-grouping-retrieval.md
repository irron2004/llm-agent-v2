# Chapter-Aware Retrieval (Section/Chapter Grouping)

## AGENT NOTE — 구현 완료 상태 (2026-03-07)

> **이 플랜의 Phase 1, 3이 구현 완료됨. 아래 코드를 검토하고 Phase 2 (JSONL 재생성 + ES 재인덱싱)를 진행해주세요.**

### 구현된 파일 목록

| File | Status | Description |
|------|--------|-------------|
| `scripts/chunk_v3/section_extractor.py` | **NEW** | 섹션 경계 추출 모듈 (SOP/SETUP: TOC+헤더, TS: alpha TOC+X-N., PEMS: no-op) |
| `scripts/chunk_v3/common.py` | **MODIFIED** | `ChunkV3Document`에 `section_chapter`, `section_number`, `chapter_source`, `chapter_ok` 필드 추가 |
| `scripts/chunk_v3/chunkers.py` | **MODIFIED** | `chunk_vlm_parsed()`에서 `extract_sections()` 호출 + 청크 생성시 섹션 필드 할당 |
| `backend/llm_infrastructure/elasticsearch/mappings.py` | **MODIFIED** | `get_chunk_v3_content_mapping()`에 section 필드 4개 추가 |
| `scripts/chunk_v3/run_ingest.py` | **MODIFIED** | `allowed_fields`에 section 필드 4개 추가 |
| `backend/config/settings.py` | **MODIFIED** | `RAGSettings`에 `section_expand_*` 설정 4개 추가 |
| `backend/llm_infrastructure/retrieval/engines/es_search.py` | **MODIFIED** | `fetch_section_chunks()` 메서드 + `_source_fields()`에 section 필드 추가 |
| `backend/llm_infrastructure/retrieval/postprocessors/section_expander.py` | **NEW** | `SectionExpander` 후처리기 (expand + ordering + dedup) |
| `backend/llm_infrastructure/retrieval/postprocessors/__init__.py` | **NEW** | postprocessors 패키지 init |
| `backend/tests/test_section_expansion.py` | **NEW** | 16 tests (SOP 4, TS 2, PEMS 2, Expander 8) — all passing |

### 검증된 실제 데이터 결과

- **SOP fill_valve (17p)**: 16/17 pages 섹션 할당 성공 (page 1 = TOC 스킵)
- **SOP safety_valve, vent_valve, ctc**: 정상 동작 확인
- **TS ffu_abnormal (6p)**: LaTeX 테이블 형식이라 X-N. 패턴 미매칭 (known limitation)

### 남은 작업 (Phase 2 — 이 플랜의 TODO 5, 10에 해당)

1. **JSONL 재생성**: `run_chunking.py`로 전체 VLM parsed JSON → JSONL 재생성 (section 필드 포함)
2. **ES 재인덱싱**: `run_ingest.py content`로 ~390k docs 재인덱싱
3. **Retrieval pipeline 통합**: `SectionExpander`를 실제 retrieval pipeline에 연결 (LangGraph RAG agent 또는 search service)
4. **TS LaTeX 테이블 대응**: 테이블 셀 내 `A-1.` 패턴 파싱 (optional enhancement)

### 검토 요청사항

- [ ] `section_extractor.py`의 TOC 파싱 로직 검토 (edge case 없는지)
- [ ] `section_expander.py`의 `all_results_ordered()` 정렬 로직 검토
- [ ] `fetch_section_chunks()` ES 쿼리 효율성 검토
- [ ] 실제 retrieval pipeline에 SectionExpander 통합 위치 결정

---

## TL;DR
> **Summary**: Add TOC/section-aware metadata at ingest time and expand retrieval results by `doc_id + section_chapter` so multi-page procedures are returned as a coherent unit.
> **Deliverables**: section metadata fields on chunk_v3 docs + ES mapping + deterministic post-rerank expansion + fallbacks/caps + tests.
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: Extend chunk schema + ES mapping → backfill/reindex strategy → retrieval post-processor expansion + token/page caps → tests/evidence

## Context
### Original Request
- Review `docs/2026-03-07-Chapter-Grouping-Retrieval.md` and create an implementation plan.

### Interview Summary
- No user preferences provided; plan follows the document's “방법 A” (index-time section metadata + query-time expansion).

### Metis Review (gaps addressed)
- Metis call timed out in this environment; plan compensates by adding explicit guardrails, acceptance criteria, and failure-mode tests based on repo exploration + Oracle guidance.

## Work Objectives
### Core Objective
- When a chunk from a multi-page procedure section is retrieved (e.g., SOP “10. Work Procedure”), fetch the rest of that section’s pages/chunks and deliver them ordered by `page` without breaking existing behavior for non-TOC documents.

### Deliverables
- New chunk metadata fields persisted in `chunk_v3_content` docs:
  - `section_chapter` (keyword)
  - `section_number` (integer)
  - `chapter_source` (keyword) and `chapter_ok` (boolean) to gate expansion
- ES mapping updated in `backend/llm_infrastructure/elasticsearch/mappings.py`.
- Ingest-time extraction/mapping in `backend/services/ingest/metadata_extractor.py` (or adjacent ingest pipeline module) that assigns section metadata per page/chunk by doc_type.
- Retrieval-time deterministic expansion post-processor in `backend/services/search_service.py` (or `backend/services/retrieval_pipeline.py`) executed after reranking.
- Tests proving: mapping correctness, expansion behavior, fallback behavior, budget adherence, and no regression.

### Definition of Done (verifiable conditions with commands)
- `uv run pytest -q` (or repo-standard pytest command) passes.
- New/updated tests cover expansion + gating + caps.
- A manual/agent-executed QA run shows that SOP Work Procedure queries include all pages (up to cap) in correct order.

### Must Have
- Expansion triggers only on high-confidence sections (`chapter_ok=true` and `chapter_source` in allowlist).
- Expanded chunks do NOT reorder/displace the original top hits; they attach as supporting context.
- Enforce both `max_chapter_pages` and `max_expansion_tokens` (or equivalent token budget) with deterministic trimming.

### Must NOT Have (guardrails, scope boundaries)
- No LLM-based query rewriting for chapter expansion (avoid drift).
- No expansion for free-form doc types unless explicitly implemented with reliable signals.
- No “best effort” mapping that produces wrong `section_chapter`; prefer `UNKNOWN` + fallback.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (existing pytest suite in `backend/tests/`).
- QA policy: every task includes runnable scenarios; evidence captured under `.sisyphus/evidence/`.

## Execution Strategy
### Parallel Execution Waves
Wave 1: schema + extraction groundwork (ingest + mapping + settings)
Wave 2: retrieval expansion post-processor + packaging/budgeting
Wave 3: tests + regression + perf/observability

### Dependency Matrix (full, all tasks)
- Tasks 1-3 unblock 4-6; tests (7-10) depend on implementation.

### Agent Dispatch Summary
- Wave 1: 3 tasks (unspecified-high, deep)
- Wave 2: 3 tasks (unspecified-high)
- Wave 3: 4 tasks (quick, deep)

## TODOs

- [x] 1. Confirm current chunk_v3 schema + ingest write path and decide exact fields placement

  **What to do**:
  - Locate the code that creates/writes `chunk_v3_content` documents (likely `backend/services/es_ingest_service.py` and/or `backend/llm_infrastructure/elasticsearch/document.py`).
  - Identify existing metadata fields used at retrieval time (e.g., `doc_id`, `page`, `doc_type`, `chapter`, `device_name`, `equip_id`, `extra_meta`).
  - Decide definitive field names and types (use plan defaults):
    - `section_chapter: str` (keyword)
    - `section_number: int` (integer)
    - `chapter_source: str` (keyword; values: `title|rule|toc_match|carry_forward|unknown|none`)
    - `chapter_ok: bool` (boolean)
  - Decide where to store doc_type-specific extras:
    - Keep the above as top-level fields (NOT in `extra_meta`) so ES can filter efficiently.

  **Must NOT do**:
  - Do not overload existing `chapter` meaning (keep it as topic/title-level as the doc states).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: requires cross-module tracing across ingest + ES mapping + retrieval.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2,3,4,5,6,7] | Blocked By: []

  **References**:
  - ES mapping: `backend/llm_infrastructure/elasticsearch/mappings.py` (`get_chunk_v3_content_mapping`).
  - Existing metadata extraction: `backend/services/ingest/metadata_extractor.py` (already writes `chapter_source`).
  - Guardrail test: `backend/tests/test_chunk_v3_contracts.py`.

  **Acceptance Criteria**:
  - [ ] Identified the authoritative writer for `chunk_v3_content` and documented it in code comments or plan evidence.

  **QA Scenarios**:
  ```
  Scenario: Locate ingest writer and fields
    Tool: Grep
    Steps: Search for writes to index 'chunk_v3_content' and construction of docs with chunk_id/doc_id/page.
    Expected: One clear codepath identified; fields list captured.
    Evidence: .sisyphus/evidence/task-1-writer-locations.txt

  Scenario: No accidental schema collision
    Tool: Grep
    Steps: Search for existing 'section_chapter' usage.
    Expected: No conflicting semantics; field name safe.
    Evidence: .sisyphus/evidence/task-1-field-collision.txt
  ```

  **Commit**: YES | Message: `docs(plan): lock chunk_v3 section field contracts` | Files: [internal notes/evidence only if repo tracks it; otherwise code changes later]

- [x] 2. Extend `chunk_v3_content` ES mapping with section fields

  **What to do**:
  - Update `backend/llm_infrastructure/elasticsearch/mappings.py` `get_chunk_v3_content_mapping()` properties to include:
    - `section_chapter` as keyword
    - `section_number` as integer
    - `chapter_source` as keyword
    - `chapter_ok` as boolean
  - Update/extend mapping guardrail tests in `backend/tests/test_chunk_v3_contracts.py` to assert:
    - `dynamic` remains `False`
    - `extra_meta` remains disabled
    - New fields exist with correct types

  **Must NOT do**:
  - Do not set mapping `dynamic=true`.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: localized mapping + test update.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [6,7,8] | Blocked By: [1]

  **References**:
  - Mapping: `backend/llm_infrastructure/elasticsearch/mappings.py`.
  - Test pattern: `backend/tests/test_chunk_v3_contracts.py`.

  **Acceptance Criteria**:
  - [ ] `pytest -q backend/tests/test_chunk_v3_contracts.py` passes.

  **QA Scenarios**:
  ```
  Scenario: Mapping has new fields
    Tool: Bash
    Steps: Run pytest for mapping contract test.
    Expected: Green.
    Evidence: .sisyphus/evidence/task-2-mapping-test.txt

  Scenario: Mapping remains strict
    Tool: Bash
    Steps: Ensure dynamic false assertion still passes.
    Expected: Green.
    Evidence: .sisyphus/evidence/task-2-dynamic-false.txt
  ```

  **Commit**: YES | Message: `feat(es): add section fields to chunk_v3_content mapping` | Files: `backend/llm_infrastructure/elasticsearch/mappings.py`, `backend/tests/test_chunk_v3_contracts.py`

- [x] 3. Implement section boundary extraction for SOP/SETUP (TOC + numbered headers)

  **What to do**:
  - Implement a deterministic mapper that produces per-page `section_chapter`, `section_number`, `chapter_source`, `chapter_ok`.
  - Use the document’s fallback order:
    1) TOC parse from first 5 pages: detect Contents/목차 section and extract items like `10. Work Procedure`.
    2) Header direct extraction from top-of-page text: match `## 10. Work Procedure`, `10. Work Procedure`, `10) Work Procedure`.
    3) TOC keyword match against page text to pick best TOC item.
    4) Carry-forward previous section.
    5) Number-jump safety: if detected section numbers jump, set `section_chapter="UNKNOWN"`, `chapter_ok=false`, `chapter_source="unknown"` for affected pages.
  - Place implementation adjacent to existing patterns in `backend/services/ingest/metadata_extractor.py` (preferred) or a new ingest module under `backend/services/ingest/`.

  **Must NOT do**:
  - Do not mark `chapter_ok=true` when derived from carry-forward only; only for reliable sources.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: non-trivial parsing with edge cases.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [5,6,7,8] | Blocked By: [1,2]

  **References**:
  - Existing heading regexes: `backend/services/ingest/metadata_extractor.py` (`HEADING_PATTERNS`).
  - Doc requirements: `docs/2026-03-07-Chapter-Grouping-Retrieval.md`.

  **Acceptance Criteria**:
  - [ ] Unit tests cover TOC parse and header detection on synthetic page texts with noise/format variations.

  **QA Scenarios**:
  ```
  Scenario: SOP TOC + header mapping
    Tool: Bash
    Steps: Run new unit tests for SOP mapping.
    Expected: Correct section_number/title assignment and carry-forward behavior.
    Evidence: .sisyphus/evidence/task-3-sop-mapping-tests.txt

  Scenario: Number jump safety
    Tool: Bash
    Steps: Feed pages where '1' then '3' appears; check UNKNOWN for gap.
    Expected: chapter_ok=false for ambiguous pages; expansion later gated off.
    Evidence: .sisyphus/evidence/task-3-number-jump.txt
  ```

  **Commit**: YES | Message: `feat(ingest): extract SOP/SETUP section chapters for chunk_v3` | Files: `backend/services/ingest/metadata_extractor.py`, new/updated tests

- [x] 4. Implement TS doc_type section strategy (alpha TOC + X-N subsection heuristic)

  **What to do**:
  - Add TS-specific section mapping:
    - If alpha TOC present, map pages by detecting first `X-N.` pattern on the page and selecting TOC item `X. ...`.
    - If no TOC, set `chapter_ok=false` and leave `section_chapter=""` or `UNKNOWN` and rely on existing adjacent-page window expansion.
  - Ensure `chapter_source` distinguishes TS cases (e.g., `rule` vs `none`).

  **Must NOT do**:
  - Do not attempt fuzzy mapping when alpha TOC absent (prefer fallback).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: heuristic mapping requires careful false-positive control.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [6,7,9] | Blocked By: [1,2]

  **References**:
  - TS heading regex: `backend/services/ingest/metadata_extractor.py` (`HEADING_PATTERNS["ts"]`).
  - Doc TS strategy: `docs/2026-03-07-Chapter-Grouping-Retrieval.md`.

  **Acceptance Criteria**:
  - [ ] Tests cover TS with TOC and TS without TOC; only TOC-present docs yield `chapter_ok=true`.

  **QA Scenarios**:
  ```
  Scenario: TS page has A-1 and maps to A.
    Tool: Bash
    Steps: Run TS mapping unit tests.
    Expected: section_chapter equals the correct A. TOC entry.
    Evidence: .sisyphus/evidence/task-4-ts-mapping.txt

  Scenario: TS no TOC disables expansion
    Tool: Bash
    Steps: Run TS mapping tests with missing TOC.
    Expected: chapter_ok=false and section_chapter empty/UNKNOWN.
    Evidence: .sisyphus/evidence/task-4-ts-no-toc.txt
  ```

  **Commit**: YES | Message: `feat(ingest): add TS section mapping via X-N heuristic` | Files: ingest mapper + tests

- [ ] 5. Wire section metadata into chunk creation/writing (chunk JSONL + ES ingest)

  **What to do**:
  - Ensure the ingest pipeline attaches section fields to each chunk document before indexing into `chunk_v3_content`.
  - Ensure `page` ordering is preserved and `section_number` is integer for sorting/filtering.
  - Ensure unknown/unreliable sections yield `chapter_ok=false`.

  **Must NOT do**:
  - Do not store section fields only inside `extra_meta` (cannot filter/sort efficiently).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: touches ingestion + ES indexing pipeline.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [6,7,8,9] | Blocked By: [3,4]

  **References**:
  - Ingest services: `backend/services/es_ingest_service.py` (likely writer).
  - ES index creation: `backend/llm_infrastructure/elasticsearch/manager.py` (`create_chunk_v3_content_index`).

  **Acceptance Criteria**:
  - [ ] A local ingest run (small fixture) indexes docs containing new fields and ES accepts them with mapping.

  **QA Scenarios**:
  ```
  Scenario: Index small fixture with section fields
    Tool: Bash
    Steps: Run an ingest/test helper that indexes 1 SOP fixture into a test index.
    Expected: Documents have section_chapter/section_number/chapter_ok fields in ES.
    Evidence: .sisyphus/evidence/task-5-index-fixture.txt

  Scenario: Unknown sections do not set chapter_ok
    Tool: Bash
    Steps: Ingest fixture with missing TOC.
    Expected: chapter_ok=false.
    Evidence: .sisyphus/evidence/task-5-unknown.txt
  ```

  **Commit**: YES | Message: `feat(ingest): persist section metadata into chunk_v3_content docs` | Files: ingest writer + tests/fixtures

- [x] 6. Implement retrieval post-processor: expand by `doc_id + section_chapter` after rerank

  **What to do**:
  - Add a post-processing step in the retrieval orchestration (preferred location):
    - `backend/services/search_service.py` after reranking OR
    - `backend/services/retrieval_pipeline.py` right before assembling context.
  - Algorithm (decision-complete):
    1) Run existing retrieval + rerank to obtain base ranked chunks.
    2) Choose up to `expand_top_groups=2` groups from the base list where `chapter_ok=true` and `section_chapter` non-empty.
       - Group key = (`doc_id`, `section_chapter`).
       - Deduplicate group keys.
    3) For each chosen group key, fetch all chunks matching the key from ES, sorted by `page` asc.
    4) Apply caps:
       - `max_chapter_pages` hard limit (default 8).
       - `max_expansion_tokens` (estimate tokens from content length; deterministic).
       - If over budget, keep a contiguous window centered on triggering hit page; preserve order.
    5) Attach expanded chunks to the triggering hit (do not merge into global ranking).
  - Ensure expansion is disabled when `chapter_source` not in allowlist (e.g., allow: `rule`, `toc_match`, `title`; deny: `carry_forward`, `unknown`, `none`).

  **Must NOT do**:
  - Do not re-run retrieval with modified queries.
  - Do not allow expanded chunks to displace other top hits.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: touches core retrieval behavior and packaging.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [7,8,9,10] | Blocked By: [2,5]

  **References**:
  - Retrieval orchestrator: `backend/services/search_service.py`.
  - Existing graph expansion concept: `backend/services/agents/langgraph_rag_agent.py` (expand nodes).
  - Doc gating rules: `docs/2026-03-07-Chapter-Grouping-Retrieval.md`.

  **Acceptance Criteria**:
  - [ ] For a mocked ES response, expansion fetch executes only for allowed sources and respects caps.

  **QA Scenarios**:
  ```
  Scenario: Expand SOP Work Procedure section
    Tool: Pytest
    Steps: Mock initial retrieval returns a hit with doc_id=X, section_chapter='10. Work Procedure', chapter_ok=true; mock ES group fetch returns pages 9-14.
    Expected: Context includes pages 9-14 ordered; base ranking of other docs unchanged.
    Evidence: .sisyphus/evidence/task-6-expand-happy.txt

  Scenario: chapter_source carry_forward blocks expansion
    Tool: Pytest
    Steps: Same as above but chapter_source='carry_forward'.
    Expected: No expansion; fallback behavior remains.
    Evidence: .sisyphus/evidence/task-6-expand-gated.txt
  ```

  **Commit**: YES | Message: `feat(retrieval): expand context by doc section after rerank` | Files: retrieval orchestrator + tests

- [x] 7. Add configuration knobs (settings + preset defaults) for expansion and budgets

  **What to do**:
  - Add settings (pydantic) with defaults:
    - `RAG_SECTION_EXPAND_ENABLED=true`
    - `RAG_SECTION_EXPAND_TOP_GROUPS=2`
    - `RAG_SECTION_EXPAND_MAX_PAGES=8`
    - `RAG_SECTION_EXPAND_MAX_TOKENS=<reasonable default>`
    - `RAG_SECTION_EXPAND_ALLOWED_SOURCES=rule,toc_match,title`
  - Ensure these can be overridden by retrieval preset YAML if presets system supports it.

  **Must NOT do**:
  - Do not hardcode constants inside retrieval logic; wire through settings.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: settings/preset wiring.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [9,10] | Blocked By: [6]

  **References**:
  - Settings: `backend/config/settings.py`.
  - Retrieval presets: `backend/config/presets/retrieval_full_pipeline.yaml` and loader in `backend/config/preset_loader.py`.

  **Acceptance Criteria**:
  - [ ] Running service loads defaults with no config errors; tests cover settings parsing.

  **QA Scenarios**:
  ```
  Scenario: Disable expansion via env
    Tool: Pytest
    Steps: Set env RAG_SECTION_EXPAND_ENABLED=false and run retrieval test.
    Expected: No expansion performed.
    Evidence: .sisyphus/evidence/task-7-disable.txt

  Scenario: Tight max pages
    Tool: Pytest
    Steps: Set RAG_SECTION_EXPAND_MAX_PAGES=2.
    Expected: Only 2 pages attached, ordered.
    Evidence: .sisyphus/evidence/task-7-max-pages.txt
  ```

  **Commit**: YES | Message: `feat(config): add section expansion settings and defaults` | Files: settings + preset(s) + tests

- [x] 8. Add/extend ES query helper for group fetch (doc_id + section_chapter) with page sorting

  **What to do**:
  - Implement an ES query path in `backend/services/es_search_service.py` (or existing ES helper) to fetch by filters:
    - `term doc_id` AND `term section_chapter` AND `term chapter_ok=true`
  - Sort by `page` asc; return chunks in that order.
  - Ensure this uses `chunk_v3_content` (BM25/meta) index and not embedding index.

  **Must NOT do**:
  - Do not use full-text match; only metadata filters.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ES query correctness + integration.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [6,9] | Blocked By: [2]

  **References**:
  - ES service: `backend/services/es_search_service.py`.
  - Index mapping: `backend/llm_infrastructure/elasticsearch/mappings.py`.

  **Acceptance Criteria**:
  - [ ] Unit test verifies query DSL contains only term filters and sort by page.

  **QA Scenarios**:
  ```
  Scenario: ES group fetch query DSL
    Tool: Pytest
    Steps: Call helper and inspect generated DSL.
    Expected: bool/filter term doc_id + term section_chapter + term chapter_ok; sort page asc.
    Evidence: .sisyphus/evidence/task-8-dsl.txt

  Scenario: Empty section_chapter yields no fetch
    Tool: Pytest
    Steps: Call helper with empty section_chapter.
    Expected: Short-circuit; no ES call.
    Evidence: .sisyphus/evidence/task-8-empty.txt
  ```

  **Commit**: YES | Message: `feat(es): add section group fetch by doc_id+section_chapter` | Files: es search service + tests

- [x] 9. End-to-end retrieval tests for expansion + fallback + ranking fairness

  **What to do**:
  - Add tests that simulate:
    - Base retrieval returns multiple docs; only top group expands.
    - Expansion does not reorder top-level hits.
    - Token/page caps trim deterministically.
    - Docs without TOC remain unchanged.
  - Prefer existing test patterns under `backend/tests/` (there are already retrieval pipeline tests).

  **Must NOT do**:
  - Do not rely on live ES; use mocks/fixtures.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: integration-style behavioral guarantees.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [10] | Blocked By: [6,7,8]

  **References**:
  - Retrieval tests: `backend/tests/test_retrieval_pipeline.py`.
  - Determinism tests: `backend/tests/test_retrieve_node_deterministic.py`.

  **Acceptance Criteria**:
  - [ ] New tests pass and fail meaningfully when expansion behavior is broken.

  **QA Scenarios**:
  ```
  Scenario: Ranking fairness preserved
    Tool: Pytest
    Steps: Assert original top hits order unchanged after expansion attaches pages.
    Expected: Same top-level ordering; expansions nested/flagged.
    Evidence: .sisyphus/evidence/task-9-fairness.txt

  Scenario: Over-budget chapter trims window around hit
    Tool: Pytest
    Steps: Provide 20-page section; set max pages 8.
    Expected: 8 pages centered on hit page; ordered.
    Evidence: .sisyphus/evidence/task-9-trim.txt
  ```

  **Commit**: YES | Message: `test(retrieval): cover section expansion behavior and fallbacks` | Files: new/updated tests

- [ ] 10. Operational plan for backfill/reindex and rollout gates

  **What to do**:
  - Implement an ES migration/runbook:
    - Create new index with updated mapping.
    - Reindex/backfill strategy for ~390k docs.
    - Rollout toggles: keep expansion disabled until backfill completes.
  - Add logging/metrics fields:
    - expansion applied yes/no
    - groups expanded count
    - pages attached
    - budget trimmed yes/no
    - fallback reason

  **Must NOT do**:
  - Do not enable expansion by default in production before backfill.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: operational correctness and rollback planning.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [] | Blocked By: [2,5,6]

  **References**:
  - ES manager: `backend/llm_infrastructure/elasticsearch/manager.py`.
  - Existing migration scripts: `scripts/es_migrate_v2.py`.

  **Acceptance Criteria**:
  - [ ] A documented sequence exists to migrate indices with zero downtime (alias swap) and a rollback step.

  **QA Scenarios**:
  ```
  Scenario: Rollout gate
    Tool: Bash
    Steps: Start service with expansion disabled; run a retrieval call.
    Expected: No expansion applied.
    Evidence: .sisyphus/evidence/task-10-gate.txt

  Scenario: Backfill dry run
    Tool: Bash
    Steps: Run migration script in dry-run mode (or against test index).
    Expected: Mapping created; reindex plan validated.
    Evidence: .sisyphus/evidence/task-10-backfill.txt
  ```

  **Commit**: YES | Message: `docs(ops): add reindex and rollout plan for section expansion` | Files: scripts/docs as appropriate

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Prefer atomic commits per TODO (mapping, ingest mapper, retrieval expansion, settings, tests).
- Suggested message style used in TODOs.

## Success Criteria
- SOP/SETUP: queries that hit procedure sections return full section pages (capped) in order.
- TS: only TOC-present TS expands by alpha section; no-TOC TS uses existing behavior.
- Free-form docs: no regressions; expansion off.
- Budget adherence: never exceed caps; deterministic trimming.
