# Decisions

- Stage1 `use_rrf=True` must use app-level RRF fusion (dense + sparse + `backend/llm_infrastructure/retrieval/rrf.py`).
- Remove ES native RRF request (`rank.rrf` / `sub_searches`) from stage1 path.
- No silent fallback to script_score when `use_rrf=True`; script_score only when explicitly selected.
- Candidate window sizing default: `top_n = top_k * 2` for each source list.
