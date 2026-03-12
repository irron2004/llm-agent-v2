# Chunk Version Runtime Wiring (v2 vs chunk_v3)

## TL;DR
> **Summary**: Add an explicit runtime switch (`SEARCH_CHUNK_VERSION=v2|v3`) so the backend can use either the existing v2 single-index alias path or a new chunk_v3 split-index retrieval path (dense on embed index + BM25 on content index + `chunk_id` join + app-level RRF), without breaking current production behavior.
> **Deliverables**:
> - New v3 search service wired at startup via `backend/api/main.py`
> - New env settings for v2/v3 selection and v3 index names
> - v3 mappings/ingest updates to support filter parity and BM25 fields
> - Section expansion compatibility (content index) + unit tests
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: Settings/env wiring -> v3 service join+RRF -> mappings/ingest parity -> tests -> startup integration

## Context

### Original Request
- Read `docs/2026-03-12-chunk-version-반영.md` and write an implementation plan.

### Spec Snapshot (inlined; executor should not need external doc)
- Problem: runtime retrieval currently uses `rag_chunks_{env}_current` alias which points to a v2 index that lacks `section_chapter/chapter_source/chapter_ok` fields; chunk_v3 content index has these fields but is split across content+embed indices.
- Requirements:
  - v2 path: keep existing alias-based single-index hybrid.
  - v3 path: dense search on `chunk_v3_embed_*`, sparse search + metadata on `chunk_v3_content`, join by `chunk_id`, then app-level RRF.
  - Section/group expansion must use v3 content metadata (`section_*`, `chapter_*`).
  - Switching must be controllable by env and safe in operation; do not overload the legacy index-version env.
  - Embedding version reflection: the v3 embed index MUST correspond to the currently configured runtime embedder (`RAG_EMBEDDING_METHOD` + `RAG_EMBEDDING_VERSION`). Startup must fail fast if embed index dims/model metadata mismatch the embedder.
- Recommended env (decision-locked in this plan):
  - `SEARCH_CHUNK_VERSION=v2|v3`
  - `SEARCH_V2_ALIAS` (optional override)
  - `SEARCH_V3_CONTENT_INDEX`
  - `SEARCH_V3_EMBED_INDEX`
  - Optional convenience env:
    - `SEARCH_V3_EMBED_MODEL_KEY` (build index name `chunk_v3_embed_{key}_v1`)

### Interview Summary
- No code changes requested yet; produce a decision-complete plan only.
- Must keep `v2` behavior intact and allow switching to `v3` via env.
- User also hinted at "embedding version reflection"; plan includes explicit embed-index selection that can be aligned to embedding model/version.

### Oracle Review (pitfalls incorporated)
- Ensure v3 is a retriever swap: section expansion and stage2 filtering rely on `retriever.es_engine` pointing to content index.
- Ensure embed mapping carries all filter fields; otherwise knn `filter` can yield 0 hits silently.
- Join dense candidates to content docs before RRF so `doc_id` is correct and RRF dedupe key `(doc_id, chunk_id)` works.
- Fail fast at startup on missing indices or dimension mismatches.

### Metis Review (gaps addressed)
- v3 content mapping currently lacks `chunk_summary` / `chunk_keywords` but v2 search defaults include those fields; plan adds mapping parity to avoid ES query errors.
- Section expansion currently won’t run in the agent path unless `SearchServiceRetriever` exposes `.es_engine`; plan includes explicit wiring.
- Plan includes tests for: (a) request bodies/filters for dense+sparse, (b) join/mget behavior, (c) section expansion uses content index.

## Work Objectives

### Core Objective
- Make runtime retrieval support `chunk_v3` (split index) with safe env-controlled switching.

### Deliverables
- Env-controlled switch:
  - `SEARCH_CHUNK_VERSION=v2|v3`
  - v2 uses `rag_chunks_{env}_current` alias (current behavior).
  - v3 uses explicit indices: `chunk_v3_content` + `chunk_v3_embed_{model}_v1`.
- New v3 service that:
  - Runs dense kNN on embed index.
  - Runs sparse BM25 on content index.
  - Joins dense hits to content docs by `chunk_id` using ES `mget`.
  - Fuses dense+sparse via app-level RRF (`backend/llm_infrastructure/retrieval/rrf.py`).
  - Returns `RetrievalResult` where:
    - `doc_id` is the real document id from content index.
    - `content` is `search_text` (preprocessed), `raw_text` is `content` (original).
    - `metadata` contains `chunk_id`, `page`, `section_*`, plus any debug info.
- Mapping + ingest parity to prevent hard failures:
  - v3 content mapping supports fields referenced by BM25 search (`chunk_summary`, `chunk_keywords`).
  - v3 embed mapping supports fields used in filters (`doc_id`, `equip_id`, `lang`, `tenant_id`, `project_id`, etc.).
  - v3 ingest embeds those fields into embed docs.
- Section expansion compatibility:
  - Agent’s section expansion queries content index (not embed).
  - Works when `RAG_SECTION_EXPAND_ENABLED=true`.

### Definition of Done (verifiable)
- Backend unit tests + static checks:
  - `cd backend && uv run pytest tests/test_chunk_v3_contracts.py -v`
  - `cd backend && uv run pytest tests/test_es_chunk_v3_search_service.py -v` (new)
  - `cd backend && uv run pytest tests/test_langgraph_section_expansion_wiring.py -v` (new)
  - `cd backend && uv run ruff check .`
  - `cd backend && uv run mypy .`

### Must Have
- `SEARCH_CHUNK_VERSION=v2` path is unchanged.
- `SEARCH_CHUNK_VERSION=v3` uses both indices and joins by `chunk_id`.
- Dense and sparse stages accept the same filter set and do not silently drop to “empty retrieval” due to missing mapping fields.
- Section expansion uses content index and sees `section_chapter` / `chapter_source` / `chapter_ok` from content docs.
- When embedding model/version changes, v3 startup validation prevents querying a mismatched embed index.

### Must NOT Have (guardrails)
- Do not repurpose `SEARCH_ES_INDEX_VERSION` for v3 (the v3 layout is not a version bump).
- Do not “fallback to v2” silently if v3 env is misconfigured; fail fast with actionable error.
- Do not return results where `doc_id == chunk_id` (a sign that embed hits were not joined to content).
- Do not introduce network-dependent tests as mandatory gates (unit tests must mock ES).

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (pytest + ruff + mypy)
- Evidence artifacts: `.sisyphus/evidence/task-{N}-{slug}.txt`

## Execution Strategy

### Parallel Execution Waves

Wave 1 (Settings + wiring + contracts)
- Define env + config resolution rules; wire startup branching.
- Define v3 service interface (method signatures + return invariants).
- Add mapping guardrails tests.

Wave 2 (v3 retrieval implementation)
- Implement v3 dense+sparse+join+RRF.
- Ensure filter parity.

Wave 3 (integration + section expansion + regression tests)
- Ensure `.es_engine` availability for device fetch + section expansion.
- Add tests for section expansion and agent path.

### Dependency Matrix (high level)
- Wave 1 blocks Wave 2/3.
- Mapping/ingest parity blocks “filter parity” in Wave 2.

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Define new env settings for chunk version switching (v2 vs v3)

  **What to do**:
  - Extend `SearchSettings` (env prefix `SEARCH_`) to include:
    - `chunk_version: Literal["v2","v3"]` via env `SEARCH_CHUNK_VERSION` (default `v2`)
    - `v2_alias: str` via env `SEARCH_V2_ALIAS` (default: `${es_index_prefix}_${es_env}_current`)
    - `v3_content_index: str` via env `SEARCH_V3_CONTENT_INDEX` (default: `chunk_v3_content`)
    - `v3_embed_index: str` via env `SEARCH_V3_EMBED_INDEX` (no safe default; required if v3)
    - `v3_embed_model_key: str` via env `SEARCH_V3_EMBED_MODEL_KEY` (optional convenience; if provided and `v3_embed_index` is empty, build it)
  - Add explicit config validation in `backend/api/main.py:_configure_search_service()`:
    - If `SEARCH_CHUNK_VERSION=v3` and any required v3 env is missing -> raise `NotImplementedError` or `RuntimeError` with a clear message.
    - If `SEARCH_CHUNK_VERSION=v3`, also assert the indices exist:
      - `es.indices.exists(index=v3_content_index)` and `es.indices.exists(index=v3_embed_index)` must both be true.
  - Update the startup handler in `backend/api/main.py` so v3 misconfiguration truly fails fast:
    - If `SEARCH_CHUNK_VERSION=v3`, do NOT swallow exceptions in `startup_search_service()`; re-raise after logging.

  **Must NOT do**:
  - Do not overload `SEARCH_ES_INDEX_VERSION`.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: settings + wiring, localized.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2-6 | Blocked By: none

  **References**:
  - Spec snapshot: this plan section "Spec Snapshot (inlined)"
  - Existing settings: `backend/config/settings.py:503` (SearchSettings)
  - Startup wiring: `backend/api/main.py:128` (`_configure_search_service`)

  **Acceptance Criteria**:
  - [ ] With no new env set, backend startup still configures v2 ES alias.
  - [ ] With `SEARCH_CHUNK_VERSION=v3` and missing `SEARCH_V3_EMBED_INDEX`, startup fails with a clear error.

  **QA Scenarios**:
  ```
  Scenario: v3 missing embed index fails fast
    Tool: Bash
    Steps: run backend unit test that calls _configure_search_service() with env missing SEARCH_V3_EMBED_INDEX
    Expected: raises NotImplementedError/RuntimeError with actionable message
    Evidence: .sisyphus/evidence/task-1-v3-missing-env.txt

  Scenario: default still selects v2
    Tool: Bash
    Steps: run backend unit test that calls _configure_search_service() with default env
    Expected: EsSearchService.from_settings called with alias `${prefix}_${env}_current`
    Evidence: .sisyphus/evidence/task-1-v2-default.txt
  ```

  **Commit**: YES | Message: `feat(search): add SEARCH_CHUNK_VERSION v2/v3 wiring settings` | Files: `backend/config/settings.py`, `backend/api/main.py`, tests

- [ ] 2. Implement v3 split-index search service (dense+BM25+join+RRF)

  **What to do**:
  - Create `backend/services/es_chunk_v3_search_service.py` with `class EsChunkV3SearchService`.
  - Provide `from_settings()` that:
    - Creates an ES client using existing `search_settings.es_host` (+ auth).
    - Creates embedder instance via `EmbeddingService` like `EsSearchService.from_settings()`.
    - Creates an `EsSearchEngine` bound to **content index** with BM25 fields:
      - `text_fields=["search_text^1.0","chunk_summary^0.7","chunk_keywords^0.8"]`.
    - Stores `self.es_engine` (must exist for device aggregation in `backend/api/routers/agent.py:_create_device_fetcher`).
    - Defines `self.reranker` attribute (set to `None` unless explicitly supported later) because `LangGraphRAGAgent` reads `search_service.reranker`.
  - Implement `search()` signature compatible with `EsSearchService.search()` (accept **kwargs).
  - Implement `fetch_doc_pages()` / `fetch_doc_chunks()` to always query v3 content index.

  **Must NOT do**:
  - Do not reuse `EsHybridRetriever` (it assumes single index for dense+sparse).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: core retrieval correctness.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 3-6 | Blocked By: 1

  **References**:
  - v2 ES service for interface: `backend/services/es_search_service.py`
  - ES engine: `backend/llm_infrastructure/retrieval/engines/es_search.py`
  - RRF merge: `backend/llm_infrastructure/retrieval/rrf.py`
  - join key contract: `scripts/chunk_v3/common.py:22` (`chunk_id`)

  **Acceptance Criteria**:
  - [ ] Unit test (new) proves v3 search issues:
    - kNN query to `SEARCH_V3_EMBED_INDEX`
    - BM25 query to `SEARCH_V3_CONTENT_INDEX`
    - `mget` to `SEARCH_V3_CONTENT_INDEX`
    - RRF merge is applied

  **QA Scenarios**:
  ```
  Scenario: v3 service issues correct ES calls
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_chunk_v3_search_service.py -v
    Expected: asserts call order + indices + query bodies and output invariants
    Evidence: .sisyphus/evidence/task-2-v3-service.txt

  Scenario: join missing content docs handled deterministically
    Tool: Bash
    Steps: run a test variant where mget returns missing for some chunk_ids
    Expected: those candidates are dropped (or scored 0) consistently; warning logged; no crash
    Evidence: .sisyphus/evidence/task-2-v3-service-mget-miss.txt
  ```

  **Commit**: YES | Message: `feat(search): add chunk_v3 split-index search service` | Files: `backend/services/es_chunk_v3_search_service.py`, tests

- [ ] 3. Wire `SEARCH_CHUNK_VERSION=v3` into backend startup

  **What to do**:
  - Update `backend/api/main.py:_configure_search_service()`:
    - If `SEARCH_CHUNK_VERSION=v2` -> current behavior (alias + `EsSearchService.from_settings(index_alias)`).
    - If `SEARCH_CHUNK_VERSION=v3` -> instantiate `EsChunkV3SearchService.from_settings(...)` using v3 index envs; call `set_search_service(...)`.
  - Ensure log line includes chosen chunk_version and the content/embed index names.

  **Must NOT do**:
  - No silent fallback to v2 when v3 is requested.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6 | Blocked By: 1,2

  **References**:
  - Existing wiring: `backend/api/main.py:128`
  - Dependency injection: `backend/api/dependencies.py:set_search_service`

  **Acceptance Criteria**:
  - [ ] Unit test verifies `SEARCH_CHUNK_VERSION=v3` results in v3 service instance.
  - [ ] Unit test verifies that when `SEARCH_CHUNK_VERSION=v3` and required indices are missing, app startup raises (no swallow).

  **QA Scenarios**:
  ```
  Scenario: v3 wiring sets search service
    Tool: Bash
    Steps: run a unit test that sets env to v3 and calls _configure_search_service()
    Expected: get_search_service() returns instance of EsChunkV3SearchService
    Evidence: .sisyphus/evidence/task-3-wiring.txt
  ```

  **Commit**: YES | Message: `feat(api): wire v3 chunk search service at startup` | Files: `backend/api/main.py`, tests

- [ ] 4. Fix v3 mappings for BM25 field parity and filter parity

  **What to do**:
  - Update `backend/llm_infrastructure/elasticsearch/mappings.py`:
    - In `get_chunk_v3_content_mapping()` add:
      - `chunk_summary` (text, searchable)
      - `chunk_keywords` (keyword with a `.text` subfield like v2)
    - In `get_chunk_v3_embed_mapping()` add keyword fields required for dense filter:
      - `doc_id`, `equip_id`, `lang`
  - Update `backend/tests/test_chunk_v3_contracts.py` to assert these fields exist.

  **Must NOT do**:
  - Keep `dynamic: false` and `extra_meta.enabled=false` guardrails.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: mapping edits + unit tests.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5 | Blocked By: none

  **References**:
  - v3 mapping functions: `backend/llm_infrastructure/elasticsearch/mappings.py:555` and `:586`
  - v2 mapping baseline: `backend/llm_infrastructure/elasticsearch/mappings.py:120` (`chunk_summary`, `chunk_keywords`)
  - Source fields list: `backend/llm_infrastructure/retrieval/engines/es_search.py:380`

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_chunk_v3_contracts.py -v` passes.

  **QA Scenarios**:
  ```
  Scenario: mapping guardrails enforce required fields
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_chunk_v3_contracts.py -v
    Expected: test asserts content mapping contains chunk_summary/chunk_keywords and embed mapping contains equip_id/lang/doc_id
    Evidence: .sisyphus/evidence/task-4-mapping-guardrails.txt
  ```

  **Commit**: YES | Message: `feat(es): align chunk_v3 mappings with runtime filter/BM25 fields` | Files: `backend/llm_infrastructure/elasticsearch/mappings.py`, `backend/tests/test_chunk_v3_contracts.py`

- [ ] 5. Update chunk_v3 embed ingest to include dense-filter fields

  **What to do**:
  - Update `scripts/chunk_v3/run_ingest.py:ingest_embeddings()` to include in embed docs `_source`:
    - `doc_id`, `equip_id`, `lang` (empty string ok if unknown)
  - Keep `_id` = `chunk_id`.
  - Ensure `verify_sync` still passes.

  **Must NOT do**:
  - Do not change join key (`chunk_id`) format.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: none | Blocked By: 4

  **References**:
  - Current embed ingest source: `scripts/chunk_v3/run_ingest.py:281` (content_meta)
  - Mapping requirement: `backend/llm_infrastructure/elasticsearch/mappings.py:get_chunk_v3_embed_mapping`
  - Join spec: this plan section "Spec Snapshot (inlined)" (join by chunk_id)

  **Acceptance Criteria**:
  - [ ] New unit test (or extended existing) asserts embed ingest emits these fields when building actions.

  **QA Scenarios**:
  ```
  Scenario: embed ingest action includes filter fields
    Tool: Bash
    Steps: run a unit test that calls ingest_embeddings with small fixture and inspects generated actions
    Expected: each action._source contains doc_id/equip_id/lang
    Evidence: .sisyphus/evidence/task-5-embed-ingest-fields.txt
  ```

  **Commit**: YES | Message: `fix(chunk_v3): include filter fields in embed index ingestion` | Files: `scripts/chunk_v3/run_ingest.py`, tests

- [ ] 6. Ensure section expansion is actually enabled in agent runtime (expose retriever.es_engine)

  **What to do**:
  - In `backend/services/agents/langgraph_rag_agent.py` after creating `SearchServiceRetriever`, add:
    - if `hasattr(search_service, "es_engine")` and it is not None, set `self.retriever.es_engine = search_service.es_engine`.
  - This enables the existing check in `backend/llm_infrastructure/llm/langgraph_agent.py`:
    - `if rag_settings.section_expand_enabled and docs and hasattr(retriever, "es_engine"):`
  - Ensure the engine points to the v3 **content index** (by making v3 service expose `es_engine` bound to content index).

  **Must NOT do**:
  - Do not pass `content_index=None` while pointing engine to embed index.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 2,3

  **References**:
  - Current agent setup: `backend/services/agents/langgraph_rag_agent.py:90`
  - Section expansion trigger: `backend/llm_infrastructure/llm/langgraph_agent.py` (search for `hasattr(retriever, "es_engine")`)
  - fetch_section_chunks uses `content_index or self.index_name`: `backend/llm_infrastructure/retrieval/engines/es_search.py:428`

  **Acceptance Criteria**:
  - [ ] New unit test proves that after this change, `retrieve_node` performs section expansion when `chapter_ok` is true.

  **QA Scenarios**:
  ```
  Scenario: section expansion runs in agent path
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_langgraph_section_expansion_wiring.py -v
    Expected: test asserts fetch_section_chunks called on content index engine
    Evidence: .sisyphus/evidence/task-6-section-expansion.txt
  ```

  **Commit**: YES | Message: `fix(agent): expose es_engine to enable section expansion` | Files: `backend/services/agents/langgraph_rag_agent.py`, tests

- [ ] 7. Implement v3 query bodies + join semantics (decision-locked)

  **What to do**:
  - In `EsChunkV3SearchService.search()` implement the exact algorithm below:
    1) **Candidate window**: `candidate_n = min(max(top_k * 3, 30), 200)`.
    2) **Build filter once** using the content engine helper:
       - `filters = self.es_engine.build_filter(tenant_id=None, project_id=None, doc_type=..., doc_types=..., doc_ids=..., equip_ids=..., lang=..., device_names=...)`.
       - Apply the same `filters` to both dense and sparse queries.
    3) **Dense kNN** (embed index):
       - `knn.field = "embedding"`
       - `knn.query_vector = query_embedding`
       - `knn.k = candidate_n`
       - `knn.num_candidates = candidate_n * 2`
       - If `filters` is not None: `knn.filter = filters`
       - `_source` MUST include at least: `chunk_id`, plus any filter fields required for debugging.
       - Index = `SEARCH_V3_EMBED_INDEX`.
    4) **Sparse BM25** (content index): call `self.es_engine.sparse_search(query_text, top_k=candidate_n, filters=filters)`.
    5) **Join dense hits to content**:
       - Extract `chunk_id`s from dense hits (fall back to `_id` if missing).
       - `mget` into `SEARCH_V3_CONTENT_INDEX` with `ids=[chunk_id...]`.
       - Drop missing docs; keep deterministic order (preserve original dense rank order).
       - Build `RetrievalResult` for joined dense candidates:
         - `doc_id = _source.doc_id`
         - `content = _source.search_text or _source.content`
         - `raw_text = _source.content`
         - `metadata` must include: `chunk_id`, `page`, `doc_type`, `device_name`, `equip_id`, `lang`, `section_chapter`, `section_number`, `chapter_source`, `chapter_ok` (if present)
         - Also store `dense_score` and `dense_rank`.
       - Build sparse `RetrievalResult` from sparse hits similarly, also storing `sparse_score` and `sparse_rank`.
    6) **Fuse** using app-level RRF:
       - `fused = merge_retrieval_result_lists_rrf([dense_results, sparse_results], k=rrf_k)`.
       - Set each fused result’s `score` to the fused rrf score (already done by util).
       - Optional but recommended: attach `rrf_dense_rank`, `rrf_sparse_rank`, `rrf_k` to metadata (mirror v2 stage1 implementation).
    7) Return `fused[:top_k]`.

  **Must NOT do**:
  - Do not run RRF over embed-only docs without joining to content (breaks `doc_id` and section expansion).
  - Do not use ES native `rank.rrf`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 8-9 | Blocked By: 2,4,5

  **References**:
  - Spec algorithm: this plan section "Spec Snapshot (inlined)" (dense+content join+RRF)
  - RRF util + dedupe key: `backend/llm_infrastructure/retrieval/rrf.py:28`
  - v2 app-level RRF example: `backend/llm_infrastructure/retrieval/engines/es_search.py:227`

  **Acceptance Criteria**:
  - [ ] Unit tests prove: same filter applied to both dense+sparse; joined dense results have real `doc_id`; fused results contain `metadata.chunk_id`.

  **QA Scenarios**:
  ```
  Scenario: Filter parity across dense+sparse
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_chunk_v3_search_service.py -v -k "filter_parity"
    Expected: test inspects captured ES bodies and asserts identical filter clauses
    Evidence: .sisyphus/evidence/task-7-filter-parity.txt

  Scenario: Dense join enforces real doc_id
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_chunk_v3_search_service.py -v -k "join"
    Expected: output results have doc_id == content doc_id (not chunk_id)
    Evidence: .sisyphus/evidence/task-7-join-docid.txt
  ```

  **Commit**: YES | Message: `feat(search): implement v3 dense+sparse join and RRF fusion` | Files: `backend/services/es_chunk_v3_search_service.py`, tests

- [ ] 8. Add focused unit tests for v3 service (mocked ES)

  **What to do**:
  - Create `backend/tests/test_es_chunk_v3_search_service.py` with a fake ES client that captures:
    - `search(index=..., body=...)` calls (embed + content)
    - `mget(index=..., body=...)` call
  - Include 4 tests (names fixed):
    - `test_v3_search_uses_embed_and_content_indices`
    - `test_v3_search_applies_same_filters_to_dense_and_sparse`
    - `test_v3_search_joins_dense_hits_by_chunk_id`
    - `test_v3_search_rrf_dedupes_by_doc_id_and_chunk_id`
  - Ensure tests do not require a running ES.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: correctness + request-body assertions.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 7

  **References**:
  - Existing ES request-body tests style: `backend/tests/test_es_stage1_app_rrf.py`
  - RRF dedupe behavior: `backend/llm_infrastructure/retrieval/rrf.py`

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_es_chunk_v3_search_service.py -v` passes.

  **QA Scenarios**:
  ```
  Scenario: v3 unit test suite
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_es_chunk_v3_search_service.py -v
    Expected: 4 tests pass; no network calls
    Evidence: .sisyphus/evidence/task-8-v3-tests.txt
  ```

  **Commit**: YES | Message: `test(search): add mocked tests for v3 split-index retrieval` | Files: `backend/tests/test_es_chunk_v3_search_service.py`

- [ ] 9. Add an integration-level unit test proving section expansion runs in agent path

  **What to do**:
  - Create `backend/tests/test_langgraph_section_expansion_wiring.py` that:
    - Builds a LangGraphRAGAgent with a search_service stub exposing `es_engine` and `fetch_section_chunks`.
    - Uses docs that include `section_chapter`, `chapter_source`, `chapter_ok=True`.
    - Asserts that `fetch_section_chunks` is called and expanded docs are inserted.
  - Do NOT require ES; mock `fetch_section_chunks` to return deterministic hits.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 6

  **References**:
  - Section expander behavior tests: `backend/tests/test_section_expansion.py`
  - Expansion callsite: `backend/llm_infrastructure/llm/langgraph_agent.py` (function `_apply_section_expansion`)

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_langgraph_section_expansion_wiring.py -v` passes.

  **QA Scenarios**:
  ```
  Scenario: Agent path triggers section expansion
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_langgraph_section_expansion_wiring.py -v
    Expected: mocked fetch_section_chunks called; expanded docs inserted
    Evidence: .sisyphus/evidence/task-9-section-expansion-agent.txt
  ```

  **Commit**: YES | Message: `test(agent): prove section expansion executes in langgraph flow` | Files: `backend/tests/test_langgraph_section_expansion_wiring.py`

- [ ] 10. Optional: align ingest/summarization services with v3 content index (non-blocking)

  **What to do**:
  - If the project expects ingest/summarization to follow the same chunk version:
    - Update `backend/services/es_ingest_service.py` and `backend/services/es_summarization_service.py` to accept an explicit `index` parameter from settings when `SEARCH_CHUNK_VERSION=v3`, pointing to `SEARCH_V3_CONTENT_INDEX`.
  - Keep default unchanged for v2.

  **Must NOT do**:
  - Do not change existing default index for v2.

  **Recommended Agent Profile**:
  - Category: `unspecified-low`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 1

  **References**:
  - Ingest default index: `backend/services/es_ingest_service.py:166`
  - Summarization default index: `backend/services/es_summarization_service.py:91`

  **Acceptance Criteria**:
  - [ ] Unit tests (new or extended) confirm v2 default remains and v3 selects content index when configured.

  **QA Scenarios**:
  ```
  Scenario: v3 selects content index for ingest/summarization
    Tool: Bash
    Steps: run a small unit test that constructs services with v3 env and asserts index==SEARCH_V3_CONTENT_INDEX
    Expected: pass
    Evidence: .sisyphus/evidence/task-10-ingest-summarization-v3.txt
  ```

  **Commit**: YES/NO (optional) | Message: `chore(es): allow ingest/summarization to target chunk_v3 content index` | Files: `backend/services/es_ingest_service.py`, `backend/services/es_summarization_service.py`, tests

- [ ] 11. Embedding version reflection: validate v3 embed index matches runtime embedder (fail-fast)

  **What to do**:
  - In v3 startup path (or v3 service `from_settings()`), add a validation step:
    - Read embed index mapping via `es.indices.get_mapping(index=v3_embed_index)`.
    - Determine expected dims:
      - Primary: mapping `_meta.dims` if present
      - Fallback: mapping `properties.embedding.dims`
    - Determine runtime dims: `EmbeddingService(...).get_raw_embedder().get_dimension()`.
    - If mismatch: raise `RuntimeError` with:
      - `RAG_EMBEDDING_METHOD`, `RAG_EMBEDDING_VERSION`
      - embed index name
      - mapping dims vs embedder dims
    - If mapping `_meta.embedding_model` exists, also log it and include in mismatch error.

  **Must NOT do**:
  - Do not allow v3 to start when embedder dims mismatch the embed index.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2-3 | Blocked By: 1

  **References**:
  - Runtime embedder config: `backend/config/settings.py:41` (`RAG_EMBEDDING_METHOD`, `RAG_EMBEDDING_VERSION`)
  - Embed index meta writer: `scripts/chunk_v3/run_ingest.py:301` (passes `model_meta` into mapping)
  - Embed mapping meta: `backend/llm_infrastructure/elasticsearch/mappings.py:621`

  **Acceptance Criteria**:
  - [ ] Unit test (mocked ES mapping + fake embedder dims) proves mismatch raises with actionable message.

  **QA Scenarios**:
  ```
  Scenario: v3 embed dims mismatch blocks startup
    Tool: Bash
    Steps: run a unit test that stubs ES get_mapping to return dims=1024 while fake embedder returns 768
    Expected: RuntimeError mentioning embed index + RAG_EMBEDDING_METHOD/VERSION and dims mismatch
    Evidence: .sisyphus/evidence/task-11-embed-dims-mismatch.txt
  ```

  **Commit**: YES | Message: `feat(search): validate v3 embed index matches runtime embedding config` | Files: `backend/api/main.py` (or v3 service), tests

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Test/QA Run — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- 4-7 small commits aligned to tasks (settings, v3 service, wiring, mappings, ingest, section expansion, tests).

## Success Criteria
- v2 runtime unchanged.
- v3 runtime returns retrieval results joined by `chunk_id` with content metadata present.
- Filters work for dense+sparse (no silent empty results due to mapping mismatch).
- Section expansion uses content index and activates in agent runtime.
