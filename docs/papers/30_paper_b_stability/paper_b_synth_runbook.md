# Paper B Synthetic Benchmark ES Isolation Runbook

This runbook creates and verifies a dedicated Elasticsearch namespace for synthetic benchmark data only.

Synthetic benchmark dataset path: `data/synth_benchmarks/stability_bench_v1/`

## Safety rules (read first)

- Use a dedicated namespace for synthetic runs: env=`synth`, prefix=`rag_synth`.
- DO NOT reuse or overwrite `rag_chunks_*_current`.
- DO NOT touch any production `*_current` alias.
- If an alias/index name does not start with `rag_synth_`, treat it as out of scope for this runbook.

## Required environment split

Set both backend and ES CLI variables. They are intentionally separate surfaces.

- Backend (SearchSettings, `SEARCH_` prefix):

```bash
export SEARCH_ES_ENV=synth
export SEARCH_ES_INDEX_PREFIX=rag_synth
```

- ES CLI (`ES_` variables read by `backend.llm_infrastructure.elasticsearch.cli`):

```bash
export ES_ENV=synth
export ES_INDEX_PREFIX=rag_synth
```

Optional connection/auth variables if needed by your environment:

```bash
export ES_HOST=http://localhost:9200
# export ES_USER=...
# export ES_PASSWORD=...
```

## Create and switch the synthetic alias

Use the configured embedding dimensions from repo settings:

```bash
python -c "from backend.config.settings import search_settings; print(search_settings.es_embedding_dims)"
```

Create synthetic index v1 and atomically switch `rag_synth_synth_current` to it:

```bash
ES_ENV=synth ES_INDEX_PREFIX=rag_synth python -m backend.llm_infrastructure.elasticsearch.cli create --version 1 --dims $(python -c "from backend.config.settings import search_settings; print(search_settings.es_embedding_dims)") --switch-alias
```

Expected naming from manager convention (`{prefix}_{env}_v{version}`, `{prefix}_{env}_current`):

- Index: `rag_synth_synth_v1`
- Alias: `rag_synth_synth_current`

## Index isolation checklist

Run all checks before ingestion/evaluation.

- [ ] **CLI list confirms synthetic alias target only**

```bash
ES_ENV=synth ES_INDEX_PREFIX=rag_synth python -m backend.llm_infrastructure.elasticsearch.cli list
```

Expected:
- `Alias: rag_synth_synth_current -> rag_synth_synth_v1` (or another `rag_synth_synth_v*` target)
- Listed indices are only `rag_synth_synth_v*`

- [ ] **REST alias inspection confirms target pattern**

```bash
curl -s "$ES_HOST/_alias/rag_synth_synth_current"
```

Expected:
- Response keys contain only `rag_synth_synth_v*`
- No alias/index names with `rag_chunks_` or any production prefix

- [ ] **Global alias scan shows no accidental synthetic/prod crossover**

```bash
curl -s "$ES_HOST/_alias/*_current"
```

Expected:
- `rag_synth_synth_current` points only to `rag_synth_synth_v*`
- Existing non-synthetic aliases remain unchanged

## Troubleshooting

- If create fails with dimension mismatch, re-run the first dimension command and use that exact value for `--dims`.
- If alias switch fails because target index is missing, create the version first (`create --version <n>`) and then switch.
- If you accidentally omit `ES_ENV`/`ES_INDEX_PREFIX`, stop and re-run with explicit inline env vars to avoid touching default namespaces.
