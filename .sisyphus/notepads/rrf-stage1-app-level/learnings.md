# Learnings

- 2026-03-11: Momus review OKAY; plan references and cited code locations match repo.

- 2026-03-11: ES kNN + RRF Research
  - Error `[knn] unknown field [k]` occurs when using OLD RRF `sub_searches` syntax (pre-8.9 era) or when client incorrectly serializes knn params. The error means the parser doesn't recognize `k` at that location in the request body.
  - **Modern (8.9+) RRF syntax**: Uses `retriever` parameter with nested `rrf` object containing `retrievers` array. Each retriever can be `standard` or `knn`.
  - **Top-level knn option** (preferred): `{"knn": {"field": "...", "query_vector": [...], "k": 10, "num_candidates": 100}}` at search request root level.
  - **Query DSL knn** (expert): `{"query": {"knn": {...}}}` inside bool/must.
  - **RRF retriever**: `{"retriever": {"rrf": {"retrievers": [...], "rank_constant": 60, "rank_window_size": 100}}}` - requires minimum 2 child retrievers.
  - **sub_searches DEPRECATED**: Official docs state "RRF using sub searches is no longer supported. Use the retriever API instead."
  - Doc refs:
    - kNN search: https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html
    - kNN query: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-knn-query.html
    - RRF: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion
  - GitHub issue #8124 confirms: elasticsearch-net client was generating invalid `k: 0` in wrong context causing this error.

(End of file)

---

## RRF Implementation Research - 2026-03-11

### High-Signal Repos/Examples Found

#### 1. Haystack DocumentJoiner (deepset-ai/haystack)
**URL**: https://github.com/deepset-ai/haystack/blob/main/haystack/components/joiners/document_joiner.py

Key patterns:
- Join modes: concatenate, merge, reciprocal_rank_fusion, distribution_based_rank_fusion
- RRF k constant: 61 (paper suggested 60 + 1 for 0-index adjustment)
- Dedupe by `doc.id` using defaultdict
- Supports weights for weighted RRF
- Score normalization: `scores_map[_id] /= len(document_lists) / k`

```python
def _reciprocal_rank_fusion(self, document_lists):
    k = 61
    scores_map = defaultdict(int)
    documents_map = {}
    weights = self.weights or [1 / len(document_lists)] * len(document_lists)
    for documents, weight in zip(document_lists, weights):
        for rank, doc in enumerate(documents):
            scores_map[doc.id] += (weight * len(document_lists)) / (k + rank)
            documents_map[doc.id] = doc
    # Normalize scores
    for _id in scores_map:
        scores_map[_id] /= len(document_lists) / k
```

#### 2. LangChain EnsembleRetriever (langchain-ai/langchain)
**URL**: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/ensemble.py

Key patterns:
- `id_key` parameter for custom dedupe key (defaults to page_content)
- Supports `c=60` constant (configurable)
- Weighted RRF via weights parameter
- Dedupe via unique_by_key() using content or metadata key

```python
def weighted_reciprocal_rank(self, doc_lists):
    rrf_score = defaultdict(float)
    for doc_list, weight in zip(doc_lists, self.weights):
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content if self.id_key is None else doc.metadata[self.id_key]
            rrf_score[key] += weight / (rank + self.c)
```

#### 3. Qdrant Client Fusion (qdrant/qdrant-client)
**URL**: https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/hybrid/fusion.py

Key patterns:
- DEFAULT_RANKING_CONSTANT_K = 2 (different from 60!)
- Uses point.id for dedupe
- Custom score computation: `1 / ((pos + 1.0) / score_weight + ranking_constant - 1.0)`
- Also implements distribution_based_score_fusion

#### 4. LlamaIndex RAG Fusion (run-llama/llama-hub)
**URL**: https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/query/rag_fusion_pipeline/base.py

Key patterns:
- Multi-query generation + RRF
- Uses k=60 (standard)
- Returns NodeWithScore objects

#### 5. RAGFlow (infiniflow/ragflow)
**URL**: https://github.com/infiniflow/ragflow

Key patterns for dedupe:
- Uses `doc_id`, `_id`, `id` fallbacks for document identification
- Metadata includes: doc_id, document_id, doc_hash, dataset_id
- chunk_id tracking via chunk_index

#### 6. Dify (langgenius/dify)
**URL**: https://github.com/langgenius/dify/blob/main/api/core/rag/docstore/dataset_docstore.py

Key patterns:
- `index_node_id` as doc_id
- Metadata: doc_id, doc_hash, document_id, dataset_id

### Dedupe Key Strategies

| System | Primary Key | Fallback | Notes |
|--------|-------------|----------|-------|
| Haystack | doc.id | - | Built-in Document.id |
| LangChain | id_key param | page_content | Configurable metadata key |
| Qdrant | point.id | - | Vector DB point ID |
| RAGFlow | doc_id | _id, id | Multiple fallbacks |
| Dify | index_node_id | - | Segment ID |

### Metadata Fields for Debug/Tracing

Common patterns:
- `doc_id` / `chunk_id`: Unique chunk identifier
- `document_id`: Parent document ID
- `doc_hash`: Content hash for change detection
- `dataset_id`: Collection/KB identifier
- `source`: Source file or URL
- `chunk_index`: Position in original document
- `rrf_score`: Computed fusion score (post-fusion)
- `rrf_rank`: Position after fusion
- `source_retriever`: Which retriever found the chunk (dense/sparse)
- `original_score`: Score before fusion

### RRF Constant (k) Values

| System | k value | Notes |
|--------|---------|-------|
| Original Paper | 60 | Cormack et al. 2009 |
| Haystack | 61 | 60 + 1 for 0-index |
| LangChain | 60 (default) | Configurable |
| Qdrant | 2 | Mitigates outlier impact |
| Pyserini | 60 | Standard |
| RAGFusion | 60 | Standard |

### Deterministic Ordering Notes

1. **Sort by RRF score descending**: Primary sort
2. **Secondary sort by original score**: For tie-breaking
3. **Tertiary sort by doc_id**: For complete determinism
4. **Reproducibility**: Use fixed k, weights, and sort stable

### Pitfalls/Anti-Patterns to Avoid

1. **Score range inconsistency**: Dense (cosine) vs sparse (BM25) scores have different ranges - RRF handles this by using ranks, not raw scores
2. **Missing dedupe key**: Always specify explicit id_key; don't rely on content equality
3. **k too small**: Makes top results too dominant; k=60 is well-tested
4. **k too large**: Overly flattens ranking differences
5. **Ignoring weight normalization**: Weights should sum to 1.0
6. **Forgetting rank offset**: Python is 0-indexed; add 1 or use k+1
7. **Inconsistent IDs across retrievers**: Ensure chunk_id format is identical between dense/sparse results
8. **Memory blow-up**: Limit top_k before fusion to avoid processing thousands of docs

- 2026-03-11: ES Hit Parsing Fix
  - Fixed `EsSearchEngine._parse_hits()` to handle `_score` being `None`, missing, or non-numeric (defaults to 0.0)
  - Fixed `EsSearchHit.metadata` to always contain `chunk_id` (computed from `_id` if `_source.chunk_id` missing)
  - Test file created: `tests/test_es_parse_hits_null_score.py` with 6 test cases covering edge cases
  - Issue: `float(hit.get("_score", 0.0))` crashes when `_score` key exists but value is `None` - `hit.get()` returns `None`, not the default

- 2026-03-11: Stage1 app-level RRF path now runs two independent searches and fuses in Python.
  - `hybrid_search(... use_rrf=True ...)` calls dense and sparse retrieval separately with identical filters.
  - Both searches use `top_n = top_k * 2` candidate windows.
  - Fusion uses `merge_retrieval_result_lists_rrf(..., k=rrf_k)` on `EsSearchHit.to_retrieval_result()` outputs.
  - Result conversion back to `EsSearchHit` preserves content/raw_text/metadata/doc_id and keeps chunk/page propagation.
  - Stage1 request bodies no longer include native ES RRF fields (`rank.rrf`, `sub_searches`).
- Stage1 app-level RRF metadata should compute rank maps from dense/sparse EsSearchHit lists using first-seen dedupe key rank with key priority chunk_id -> page -> doc_id-only, matching retrieval/rrf.py behavior.

QB|- 2026-03-11: Added scripts.evaluation.run_chat_flow_retrieval_rrf_eval for retrieval-only chat-flow evaluation. Usage: python -m scripts.evaluation.run_chat_flow_retrieval_rrf_eval --input scripts/evaluation/fixtures/rrf_smoke.jsonl --out /tmp/rrf_eval.jsonl. Output row schema: qid, query, thread_id, interrupted, interrupt_payload_type, search_queries, retrieved_docs (doc_id/page/metadata with rrf_* keys when present), and metadata.retrieval_debug.

XY|- 2026-03-11: Import/path resolution via sitecustomize.py approach
  - Created `backend/sitecustomize.py` - runs automatically when Python imports from backend/
  - Inserts repo root at sys.path[0] so repo-root `scripts/` wins over any backend/scripts
  - Relative paths in runner resolved via `Path(__file__).resolve().parents[2]` (repo root)
  - Output schema now includes nested `interrupt_payload` object (with `type` key) instead of flat field
- 2026-03-11: Backend module execution compatibility
  - `cd backend && uv run python -m scripts...` failed when backend had no `scripts` path in `sys.path`.
  - `usercustomize.py` did not load because `site.ENABLE_USER_SITE` is false in this environment.
  - Added `backend/scripts -> ../scripts` symlink so module resolution and fixture path both work from backend cwd.

- 2026-03-11 correction: earlier `sitecustomize.py` note is historical only; final applied approach is `backend/scripts` symlink, and `backend/sitecustomize.py` is not used in final state.
