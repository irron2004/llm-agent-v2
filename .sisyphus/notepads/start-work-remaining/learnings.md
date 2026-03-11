# Learnings

## 2026-03-09
- For bge_m3/jina_v5 87-missing case, existing `chunk_ids_*.jsonl` artifacts had zero overlap with missing IDs, so ingest-only replay was impossible; minimal subset re-embedding (87 rows) + `run_ingest.py embed` backfill restored sync.
