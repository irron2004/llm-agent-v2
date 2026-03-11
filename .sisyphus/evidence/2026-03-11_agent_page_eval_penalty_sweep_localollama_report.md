# Penalty Sweep Report (testclient, local ollama)

- Dataset: `.sisyphus/evidence/2026-03-10_quality_eval_questions.csv` (79 rows)
- Shared config: `stage2=false`, `early_page_penalty_enabled=true`, `max_page=2`
- Runtime note: `OLLAMA_MODEL_NAME=qwen2.5:0.5b`, `SEARCH_ES_HOST=http://localhost:8002`

| factor | failures | doc@1 | doc@3 | doc@5 | doc@10 | page@1 | page@3 | page@5 | page@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0 | 0.4557 | 0.6329 | 0.6456 | 0.6962 | 0.3291 | 0.4937 | 0.5063 | 0.5823 |
| 0.25 | 0 | 0.4557 | 0.6329 | 0.6456 | 0.6962 | 0.3291 | 0.4937 | 0.5063 | 0.5823 |
| 0.30 | 0 | 0.4557 | 0.6329 | 0.6456 | 0.6962 | 0.3291 | 0.4937 | 0.5063 | 0.5823 |

## Interpretation

- All three factors produced identical metrics at this evaluation granularity.
- Under this local-ollama testclient setup, changing `AGENT_EARLY_PAGE_PENALTY_FACTOR` did not produce a measurable ranking delta.
- This run is not directly comparable to the earlier `...full79_tc_after_routefix` evidence because the LLM backend/model differs.
