# Learnings

- Agent settings in `backend/config/settings.py` rely on `env_prefix="AGENT_"`, so snake_case fields map directly to upper-case env vars (e.g., `second_stage_top_k` -> `AGENT_SECOND_STAGE_TOP_K`).
- Evaluation scripts can capture experiment toggles by defining a `KEY_ENV_VARS` tuple and persisting a `key_env_vars` object in run summaries for reproducibility.
- Doc-local retrieval plumbing works cleanly by forwarding optional `doc_ids` through `SearchService`/retriever adapters and reusing `EsSearchEngine.build_filter`'s `field` + `field.keyword` OR helper for index-mapping robustness.
