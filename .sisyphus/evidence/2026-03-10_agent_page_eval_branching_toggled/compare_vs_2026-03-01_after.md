# Agent Eval Before/After Report

- before_jsonl: `docs/evidence/2026-03-01_agent_page_eval_after/raw.jsonl`
- after_jsonl: `.sisyphus/evidence/2026-03-10_agent_page_eval_branching_toggled/agent_eval.jsonl`
- total(before/after): 79/79

| KPI | BEFORE | AFTER | DELTA |
|---|---:|---:|---:|
| doc-hit@10 | 56/79 (70.89%) | 57/79 (72.15%) | +1 (+1.27pp) |
| page-hit@1 | 39/79 (49.37%) | 11/79 (13.92%) | -28 (-35.44pp) |
| page-hit@3 | 46/79 (58.23%) | 37/79 (46.84%) | -9 (-11.39pp) |
| page-hit@5 | 50/79 (63.29%) | 41/79 (51.90%) | -9 (-11.39pp) |
| first_page=1 | 1/79 (1.27%) | 25/79 (31.65%) | +24 (+30.38pp) |
| failures | 0/79 (0.00%) | 0/79 (0.00%) | +0 (+0.00pp) |
| mean-jaccard@10 | - | 0.3917 | paired by idx |

- mean-jaccard@10 pairs: 79

Metrics footer:
- doc-id normalization: lower-case, strip .pdf/.docx/.doc/.txt, non-alnum->underscore, collapse underscores.
- page range parsing: accepts empty, single int (`7`), or range (`6-14`); reversed range is normalized.
- doc-hit@10: expected doc id appears in top 10 retrieved docs.
- page-hit@1/@3/@5: within top K, expected doc appears and retrieved page falls in expected page range.
- first_page=1: first retrieved doc page equals 1 (int or digit string).
- failures: rows where `error` is non-empty.
