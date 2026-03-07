# chunk_v3 Embed + Ingest Runbook (Current CLI)

> Cleanup note (2026-03-06): This file is the execution runbook.
> Canonical planning/spec file is `.sisyphus/plans/chunk_v3_embed_ingest_plan.md`.

This runbook is aligned to the current scripts in `scripts/chunk_v3/`.

## 0) Prerequisites

- Elasticsearch is reachable via backend settings (`backend/config/settings.py`).
- Input paths exist:
  - VLM parsed dir: `data/vlm_parsed`
  - source dataset root: `/home/llm-share/datasets/pe_agent_data/pe_preprocess_data`
- Python environment is activated in this repository.

## 1) Build/Refresh Manifest

```bash
python normalize.py \
  --data-root /home/llm-share/datasets/pe_agent_data/pe_preprocess_data \
  --output data/chunk_v3_manifest.json
```

## 2) Run Chunking (with VLM validation gate)

```bash
python scripts/chunk_v3/run_chunking.py \
  --vlm-dir data/vlm_parsed \
  --manifest data/chunk_v3_manifest.json \
  --output data/chunks_v3/all_chunks.jsonl \
  --stats-path data/chunks_v3/chunking_stats.json \
  --validate-vlm \
  --source-dir /home/llm-share/datasets/pe_agent_data/pe_preprocess_data \
  --validation-output data/vlm_parsed/validation_report.json
```

Notes:
- Setup aliases are supported (`setup_manual`, `set_up_manual`) and normalized to `setup`.
- `--stats-only` is available for dry accounting without writing JSONL.

## 3) Generate Embeddings (3 models)

```bash
python scripts/chunk_v3/run_embedding.py \
  --chunks data/chunks_v3/all_chunks.jsonl \
  --models qwen3_emb_4b bge_m3 jina_v5 \
  --output-dir data/chunks_v3 \
  --batch-size 64 \
  --device cuda
```

Each model run performs contract validation (dims, NaN/Inf, L2 norm range) before full embedding.

## 4) Check Embedding Artifacts

```bash
python scripts/chunk_v3/check_embeddings.py \
  --embeddings data/chunks_v3/embeddings_qwen3_emb_4b.npy \
  --chunk-ids data/chunks_v3/chunk_ids_qwen3_emb_4b.jsonl \
  --expected-dim 2560 \
  --check-norm

python scripts/chunk_v3/check_embeddings.py \
  --embeddings data/chunks_v3/embeddings_bge_m3.npy \
  --chunk-ids data/chunks_v3/chunk_ids_bge_m3.jsonl \
  --expected-dim 1024 \
  --check-norm

python scripts/chunk_v3/check_embeddings.py \
  --embeddings data/chunks_v3/embeddings_jina_v5.npy \
  --chunk-ids data/chunks_v3/chunk_ids_jina_v5.jsonl \
  --expected-dim 1024 \
  --check-norm
```

## 5) Ingest Content Index

```bash
python scripts/chunk_v3/run_ingest.py content \
  --chunks data/chunks_v3/all_chunks.jsonl \
  --content-index chunk_v3_content \
  --batch-size 500
```

인덱스를 강제로 다시 만들려면 `--recreate`를 추가한다.

## 6) Ingest Embedding Indices

```bash
python scripts/chunk_v3/run_ingest.py embed \
  --model qwen3_emb_4b \
  --embeddings data/chunks_v3/embeddings_qwen3_emb_4b.npy \
  --chunk-ids data/chunks_v3/chunk_ids_qwen3_emb_4b.jsonl \
  --chunks data/chunks_v3/all_chunks.jsonl \
  --embed-index chunk_v3_embed_qwen3_emb_4b_v1 \
  --batch-size 500

python scripts/chunk_v3/run_ingest.py embed \
  --model bge_m3 \
  --embeddings data/chunks_v3/embeddings_bge_m3.npy \
  --chunk-ids data/chunks_v3/chunk_ids_bge_m3.jsonl \
  --chunks data/chunks_v3/all_chunks.jsonl \
  --embed-index chunk_v3_embed_bge_m3_v1 \
  --batch-size 500

python scripts/chunk_v3/run_ingest.py embed \
  --model jina_v5 \
  --embeddings data/chunks_v3/embeddings_jina_v5.npy \
  --chunk-ids data/chunks_v3/chunk_ids_jina_v5.jsonl \
  --chunks data/chunks_v3/all_chunks.jsonl \
  --embed-index chunk_v3_embed_jina_v5_v1 \
  --batch-size 500
```

임베딩 적재는 `--chunks` 로컬 파일에서 메타를 조인하며, content 인덱스 스캔을 사용하지 않는다.

## 7) Verify Content/Embed Sync

```bash
python scripts/chunk_v3/run_ingest.py verify \
  --model qwen3_emb_4b \
  --content-index chunk_v3_content \
  --embed-index chunk_v3_embed_qwen3_emb_4b_v1 \
  --output-json data/chunks_v3/verify_qwen3_emb_4b.json

python scripts/chunk_v3/run_ingest.py verify \
  --model bge_m3 \
  --content-index chunk_v3_content \
  --embed-index chunk_v3_embed_bge_m3_v1 \
  --output-json data/chunks_v3/verify_bge_m3.json

python scripts/chunk_v3/run_ingest.py verify \
  --model jina_v5 \
  --content-index chunk_v3_content \
  --embed-index chunk_v3_embed_jina_v5_v1 \
  --output-json data/chunks_v3/verify_jina_v5.json
```

`verify` checks:
- content/embed counts
- chunk_id set equality
- sampled `content_hash` consistency

## 8) Smoke Evaluation

```bash
python scripts/chunk_v3/smoke_eval.py \
  --models qwen3_emb_4b bge_m3 jina_v5 \
  --queries-csv docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv \
  --limit 20 \
  --top-k 10 \
  --num-candidates 100 \
  --output-dir data/chunks_v3/eval_smoke \
  --device cuda
```

## 9) Formal SOP Evaluation

```bash
python scripts/chunk_v3/eval_sop_questionlist.py \
  --models qwen3_emb_4b bge_m3 jina_v5 \
  --input-csv docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv \
  --top-k 10 \
  --num-candidates 100 \
  --output-dir data/chunks_v3/eval_formal \
  --device cuda
```

Outputs:
- `data/chunks_v3/eval_formal/sop_eval_summary.md`
- `data/chunks_v3/eval_formal/sop_eval_summary.json`
- `data/chunks_v3/eval_formal/sop_eval_rows.csv`
