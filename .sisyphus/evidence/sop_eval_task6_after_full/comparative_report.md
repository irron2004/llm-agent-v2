# SOP Filter Eval Comparative Report

Baseline source:
- `.sisyphus/evidence/2026-03-11_sop_filter_eval/sop_only_results.jsonl`

After source:
- `.sisyphus/evidence/sop_eval_task6_after_full/sop_only_results.jsonl`

## Retrieval Metrics (doc/page hits)

Baseline (79 rows):
- errors: 0
- doc_hit: 75/79
- page_hit: 75/79

After (79 rows):
- errors: 79
- doc_hit: 0/0
- page_hit: 0/0

## Notes

- The "after" run executed via `--use-testclient` but every row errored with `_NotConfiguredSearchService`, so no retrieval results (top_docs) or answers were produced.
- Because answers were not generated, answer-format compliance (sections/citations/language) cannot be meaningfully compared in this environment.
- If you want a meaningful after-run comparison, run the eval in an environment where the backend search service and LLM provider are configured (same setup as the baseline run).
