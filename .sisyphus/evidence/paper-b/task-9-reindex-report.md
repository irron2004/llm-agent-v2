# Task 9 Reindex-Tier (T4) Stability Report

## Scope
- Synthetic-only namespace used throughout: `rag_synth_*` (`ES_ENV=synth`, `ES_INDEX_PREFIX=rag_synth`, `SEARCH_ES_HOST=http://localhost:8002`).
- Alias before run: `rag_synth_synth_current -> rag_synth_synth_v1` (see `task-9-alias-before.json`).
- Alias after run: `rag_synth_synth_current -> rag_synth_synth_v2` (see `task-9-alias-after.json`).
- Deterministic evals executed with identical params for both index versions:
  - `--k 10 --repeats 1 --limit 20`
  - v1 out: `.sisyphus/evidence/paper-b/reindex_t4_v1/`
  - v2 out: `.sisyphus/evidence/paper-b/reindex_t4_v2/`

## Comparison Method
- Unit of comparison: per-`qid`, deterministic protocol rows only (`mode == deterministic_protocol`, `repeat_index == 0`).
- Stability delta across reindex versions:
  - Mean Jaccard@10 of `top_k_doc_ids` set between v1 and v2.
  - Exact-match-rate@10 of ordered `top_k_doc_ids` list between v1 and v2.
- Effectiveness delta from each run's `metrics.json`:
  - `hit@5` and `MRR` for v1 and v2, plus `v2 - v1` deltas.

## Results (n=20 qids)

| Metric | v1 | v2 | Delta (v2 - v1) |
|---|---:|---:|---:|
| Mean Jaccard@10 (cross-index, deterministic repeat_index=0) | N/A | N/A | 1.000000 |
| Exact-match-rate@10 (cross-index, deterministic repeat_index=0) | N/A | N/A | 0.950000 |
| hit@5 | 0.200000 | 0.200000 | 0.000000 |
| MRR | 0.276409 | 0.275516 | -0.000893 |

Notes:
- Jaccard@10 is reported as a cross-index average (v1 vs v2) and is perfect (`1.0`) for all compared qids.
- Exact-match-rate@10 is `0.95`, indicating one or more rank-order flips inside the same top-10 sets.

## Interpretation
- Reindex-tier behavior is highly stable at the set level (no top-10 membership drift on this sample).
- Minor ordering instability remains after reindex (5% of qids changed order within top-10).
- Retrieval effectiveness impact is negligible in this run:
  - `hit@5` unchanged.
  - `MRR` changed by `-0.000893` (very small absolute shift).

## Mitigation Options
1. Keep deterministic tie-break policy strict for equal/near-equal score cases (stable secondary sort key on `doc_id`).
2. Pin ANN/search parameters and shard routing in reindex workflows to reduce rank-order sensitivity.
3. Add a reindex acceptance gate in CI/runbook: fail if exact-match-rate@10 drops below target threshold.
4. Preserve and compare per-qid rank fingerprints between index versions before alias promotion.
