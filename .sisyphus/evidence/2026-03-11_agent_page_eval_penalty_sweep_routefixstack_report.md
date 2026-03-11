# Penalty Sweep Report (Routefix Stack)

## Runtime

- `SEARCH_BACKEND=es`
- `SEARCH_ES_HOST=http://localhost:8002`
- `OLLAMA_BASE_URL=http://10.10.100.45:11435`
- `OLLAMA_MODEL_NAME=gpt-oss:120b`
- `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED=false`
- `AGENT_EARLY_PAGE_PENALTY_ENABLED=true`
- `AGENT_EARLY_PAGE_PENALTY_MAX_PAGE=2`

## Full-79 Results

| factor | failures | doc@1 | doc@3 | doc@5 | doc@10 | page@1 | page@3 | page@5 | page@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 1 | 0.6456 | 0.7089 | 0.7468 | 0.7722 | 0.3924 | 0.6456 | 0.6962 | 0.7595 |
| 0.25 | 0 | 0.6456 | 0.7215 | 0.7595 | 0.7848 | 0.3797 | 0.6582 | 0.7089 | 0.7722 |
| 0.30 | 31 | 0.4430 | 0.4810 | 0.4937 | 0.5190 | 0.2532 | 0.4304 | 0.4684 | 0.5063 |

Source summaries:

- `.sisyphus/evidence/2026-03-11_agent_page_eval_full79_tc_penalty_02_routefixstack_rerun/summary.json`
- `.sisyphus/evidence/2026-03-11_agent_page_eval_full79_tc_penalty_025_routefixstack_rerun/summary.json`
- `.sisyphus/evidence/2026-03-11_agent_page_eval_full79_tc_penalty_03_routefixstack_rerun/summary.json`

Latest stability confirmation for `0.25`:

- `.sisyphus/evidence/2026-03-11_agent_page_eval_full79_tc_penalty_025_routefixstack_rerun5/summary.json`
- Metrics: `failures=0`, `doc@1=0.6582`, `doc@10=0.7848`, `page@1=0.4304`, `page@10=0.7722`

## Comparison vs Previous Routefix Baseline (`factor=0.3`)

Baseline:

- `.sisyphus/evidence/2026-03-10_agent_page_eval_full79_tc_after_routefix/summary.json`

Baseline metrics:

- doc: `@1 0.4304`, `@3 0.5696`, `@5 0.6582`, `@10 0.7089`
- page: `@1 0.3418`, `@3 0.4684`, `@5 0.5443`, `@10 0.6582`

Observed in this rerun:

- `factor=0.20` and `factor=0.25` both outperform baseline across all listed doc/page hit metrics.
- `factor=0.30` run is not reliable due to high failure count (`31/79`).

## Recommendation

- Finalize `AGENT_EARLY_PAGE_PENALTY_FACTOR=0.25` on this routefix stack.
  - Clean full-79 reruns (`rerun` and `rerun5`) both finished with `0` failures.
  - `0.25` retains the best top-end quality (`doc@10=0.7848`, `page@10=0.7722`) among stable successful runs.
  - Latest clean rerun (`rerun5`) also improved early precision (`doc@1=0.6582`, `page@1=0.4304`) over prior `0.25` and `0.20` runs.
- Keep `0.20` as fallback for conservative behavior (`failures=1`, lower hit ratios).
- Treat `0.30` as non-viable under this stack (`failures=31`).

## Reproducibility Follow-up (same stack)

- Additional full-79 rerun (`factor=0.25`):
  - `.sisyphus/evidence/2026-03-11_agent_page_eval_full79_tc_penalty_025_routefixstack_rerun2/summary.json`
  - Result: `failures=20`, doc/page metrics dropped to baseline-like levels.
  - Runtime signal: repeated Elasticsearch connectivity outages (`Connection refused`) and `503 Service Unavailable` responses during execution.
- Additional confirmatory attempt (`factor=0.25`):
  - `.sisyphus/evidence/2026-03-11_agent_page_eval_full79_tc_penalty_025_routefixstack_rerun3/`
  - Result: incomplete run due command timeout (`rows.csv` currently `49` lines including header).
  - No `summary.json` generated yet.
- Clean stability rerun with extended timeout (`factor=0.25`):
  - `.sisyphus/evidence/2026-03-11_agent_page_eval_full79_tc_penalty_025_routefixstack_rerun5/summary.json`
  - Result: `failures=0` with stable quality metrics (`doc@1=0.6582`, `doc@10=0.7848`, `page@1=0.4304`, `page@10=0.7722`).
  - This reproduces the strong `0.25` outcome and resolves the prior infra-noise concern from rerun2.

Interpretation:

- Retrieval-stack instability explains the degraded rerun2 outlier.
- With rerun5, the `0.25` setting now has multiple clean full-79 confirmations and remains the best-performing stable choice.
