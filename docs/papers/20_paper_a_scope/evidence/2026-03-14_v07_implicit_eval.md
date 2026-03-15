# v0.7 Implicit Eval Attempt (2026-03-14)

Date: 2026-03-14
Status: blocked by index/query embedding dimension mismatch

## Goal

- Run `B0/B3/B4/B4.5` on the restored mixed-scope set for the `implicit` slice.

## Command

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_7_mixed.jsonl \
  --systems B0,B3,B4,B4.5 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --shared-doc-ids .sisyphus/evidence/paper-a/policy/shared_doc_ids.txt \
  --out-dir .sisyphus/evidence/paper-a/runs/2026-03-14_v07_implicit_eval \
  --split all \
  --scope-observability implicit
```

## Result

- The run did not complete.
- `evaluate_paper_a_master.py` routes `B3/B4/B4.5` through `run_hybrid`, which hit the same dense-search blocker already documented in Paper A evidence.
- Error:

```text
BadRequestError(400, 'search_phase_execution_exception', 'failed to create query: the query vector has a different dimension [1024] than the index vectors [768]')
```

## Grounded Interpretation

- This is an infrastructure blocker, not an eval-schema blocker.
- `query_gold_master_v0_7_mixed.jsonl` itself is valid and leak-safe after split rebuild.
- The missing piece is a compatible hybrid/dense index path for `evaluate_paper_a_master.py`.

## What is already unblocked

- Mixed-scope eval master exists: `data/paper_a/eval/query_gold_master_v0_7_mixed.jsonl`
- Split report exists with leak overlap `0`: `data/paper_a/eval/query_gold_master_v0_7_mixed_split_report.json`
- Ambiguous/implicit slice restoration is complete at the dataset level.

## Next Required Fix

- Either point `evaluate_paper_a_master.py` at a 1024-dim compatible embed index,
- or add a BM25-only eval path for `B0/B4/B4.5` that bypasses dense/hybrid retrieval until index alignment is fixed.
