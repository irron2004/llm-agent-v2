# Issues

## 2026-03-09
- C3 sync check currently fails: `chunk_v3_content` (390472) vs embed indices (390385) for bge_m3/jina_v5/qwen3_emb_4b (diff=87).
- bge_m3/jina_v5 mismatch is resolved via 87-row backfill; qwen3_emb_4b was explicitly excluded by user decision in this run.
