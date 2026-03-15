# Oracle vs Parser Gap (2026-03-14)

Date: 2026-03-14
Status: generated from `scripts/paper_a/measure_parser_accuracy.py`

## Inputs

- Eval set: `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl`
- Doc scope: `.sisyphus/evidence/paper-a/policy/doc_scope.jsonl`
- Shared doc ids: `.sisyphus/evidence/paper-a/policy/shared_doc_ids.txt`
- Corpus filter: `.sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt`

## Command

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/measure_parser_accuracy.py
```

## Parser Accuracy

- Total queries: 578
- Exact match: 380/578 (65.7%)
- No detection: 150/578 (25.9%)
- Wrong detection: 48/578 (8.3%)

## Oracle vs Real Parser Retrieval (BM25, top-10)

- Oracle gold_hit@10: 92.7%
- Real parser gold_hit@10: 91.9%
- Delta: -0.9%p
- Oracle adj_cont@10: 0.0%
- Real parser adj_cont@10: 30.6%
- Parsed filters: 428
- No-filter fallback: 150

## By scope_observability

- explicit_device (n=429): gold_hit 96.7% -> 90.7%, adj_cont 0.0% -> 11.4%
- explicit_equip (n=149): gold_hit 81.2% -> 95.3%, adj_cont 0.0% -> 85.9%

## Equip-aware realistic mode

- Scope-aware parser gold_hit@10: 92.6%
- Scope-aware parser adj_cont@10: 8.5%
- Parsed device filters: 428
- Parsed equip filters: 149
- No-filter fallback: 1

## Output Artifacts

- JSON report: `data/paper_a/parser_accuracy_report.json`
- Per-query diff CSV: `data/paper_a/parser_accuracy_per_query_diff.csv`

## Interpretation Notes

- Oracle numbers are upper bounds and must not be reported as realistic production performance.
- Device-only parsing fails mainly on explicit_equip rows; the equip-aware comparison is the more realistic upper baseline for those queries.
- Use the per-query CSV to inspect cases where parser failure preserves hit but sharply increases contamination.
