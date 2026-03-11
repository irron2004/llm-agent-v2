# Issues

- 2026-03-11: ES kNN + RRF Error Analysis
  - `[knn] unknown field [k]` root cause: Using deprecated `sub_searches` format OR client library bug generating invalid request shape.
    - Evidence: GitHub elasticsearch-net #8124 - client generated `k: 0` at wrong level causing parse failure.
    - Stack Overflow cases show users confused about knn placement (query vs top-level).
  - **Modern RRF does NOT use `sub_searches`** - docs explicitly say it's deprecated.
  - **Constraints for native RRF**:
    - Requires `retrievers` array with minimum 2 child retrievers
    - Each child can be `standard` (BM25 query) or `knn` retriever
    - Parameters: `rank_constant` (default 60), `rank_window_size` (default = size)
    - If `k` > `rank_window_size`, results truncated to `rank_window_size`
  - **Implication for stage1 plan**: Removing native RRF path is correct - avoid complex ES version-specific syntax. Use simple top-level knn + query hybrid instead (supported in ES 8.x standard).

(End of file)

---

## RRF Implementation Issues & Gotchas - 2026-03-11

### Critical Implementation Issues Found

1. **Score Normalization Variance**
   - Issue: Different retrievers return scores in wildly different ranges
   - Resolution: RRF uses ranks, not scores - this is intentional
   - Watch: Some implementations normalize scores post-RRF (Haystack does this)

2. **ID Format Mismatch Between Retrievers**
   - Issue: Dense and sparse retrievers may use different ID formats
   - Example: Elasticsearch uses `_id`, vector DB uses internal IDs
   - Resolution: Normalize IDs before fusion - map to common `chunk_id` format

3. **Rank Indexing**
   - Issue: 0-indexed Python lists vs 1-indexed paper
   - Resolution: Use k=61 (60+1) or adjust formula to `1/(k+rank+1)`

4. **Weight Configuration**
   - Issue: Unequal retriever quality not accounted for
   - Resolution: Tune weights based on offline evaluation
   - Common: 0.5/0.5 for balanced, 0.3/0.7 if sparse is stronger

5. **Top-K Before vs After Fusion**
   - Issue: Fetch too many from each retriever before fusion
   - Resolution: Fetch top_k * 3 from each, then fuse and re-limit

### Elasticsearch-Specific Considerations

1. **BM25 scoring**: Probabilistic, unbounded range
2. **Dense scoring**: Typically 0-1 (cosine) or unbounded (dot product)
3. **Hybrid pipelines**: ES 8.x supports RRF natively via `rank_feature` or manual fusion
4. **Metadata propagation**: Ensure `_source` includes all needed fields

### Metadata Requirements for Stage1

Based on findings, stage1 results should include:
- `chunk_id`: Unique identifier (REQUIRED for dedupe)
- `doc_id`: Parent document reference
- `source_retriever`: "dense" | "sparse" | "fused"
- `original_rank`: Rank in individual retriever result
- `original_score`: Score before fusion
- `rrf_score`: Computed fusion score
- `rrf_rank`: Final rank after fusion

### Testing Considerations

1. **Determinism test**: Run same query multiple times, verify identical ordering
2. **Dedupe test**: Insert same chunk in both retrievers, verify single result
3. **Weight sensitivity**: Test that weights actually affect ranking
4. **Edge cases**: Empty results from one retriever, all overlap, no overlap


#XV|
#QT|### Scope Creep Prevention - 2026-03-11
#QM|
#YQ|Caught massive scope creep during task execution where ~30+ files were modified across backend/frontend/tests/api.
#QK|Caused by: Previous agent session had attempted multiple tasks but left worktree dirty.
#RR|Prevention: Used `git restore` to revert all files except the two explicitly allowed:
#JR|  - `.sisyphus/boulder.json` (allowed)
#RX|  - `backend/llm_infrastructure/retrieval/engines/es_search.py` (allowed)
#RR|Restored deleted file: `.sisyphus/plans/start-work-all-remaining.md`
#BQ|Untracked new file kept: `backend/tests/test_es_parse_hits_null_score.py`
#YJ|Learned: Always check `git status --porcelain` before completing to ensure only intended changes remain.

- 2026-03-11: Test import path gotcha while adding stage1 app-level RRF test.
  - Importing `llm_infrastructure...` directly during collection triggered duplicate retriever registration (`es_hybrid v1`).
  - Resolved by dynamically importing `backend.llm_infrastructure...` first, with fallback import only if needed.

- 2026-03-11: Scope creep cause and constraint action (Task 2 cleanup).
  - Cause: pre-existing dirty tracked edits in unrelated API/frontend files from prior multitask work.
  - Constraint: used `git restore` only on those tracked paths and re-verified with `git diff --name-only` so allowed tracked scope remained isolated.
