# chunk_v3 Embedding + ES 적재 계획

## 상태 요약 (2026-03-05)

| Phase | 스크립트 | 상태 |
|-------|---------|------|
| 1. VLM Parse | `scripts/chunk_v3/vlm_parse.py` | 완료 (sop:384, ts:79, set_up_manual:15, pems:미실행) |
| 2. Chunking | `scripts/chunk_v3/run_chunking.py` | 미실행 |
| 3. Embedding | `scripts/chunk_v3/run_embedding.py` | 미실행 |
| 4. ES Ingest | `scripts/chunk_v3/run_ingest.py` | 미실행 |

## 요구사항

- 5종 문서(SOP, TS, SetupManual, MyService, GCB) 전체 chunking
- 3개 임베딩 모델 비교 평가: `qwen3_emb_4b`, `bge_m3`, `jina_v5`
- ES 2-index 구조: `chunk_v3_content` (원문+BM25) + `chunk_v3_embed_{model}_v1` (벡터)
- 모델별 독립 인덱스로 추가/삭제 용이

## 수락 기준

1. `data/chunks_v3/all_chunks.jsonl` 생성, 총 chunk 수 > 100,000
2. 3개 모델 각각 `embeddings_{model}.npy` + `chunk_ids_{model}.jsonl` 생성
3. ES `chunk_v3_content` 인덱스 문서 수 == JSONL chunk 수
4. ES `chunk_v3_embed_{model}_v1` 3개 인덱스 각각 문서 수 == chunk 수
5. `run_ingest.py verify` 통과 (content ↔ embed 동기화)

## 실행 단계

### Step 0: 사전 확인
- [ ] VLM 서버 종료 (GPU 메모리 확보 필요 — 임베딩 모델 로드용)
- [ ] ES 서버 가동 확인: `curl -s http://localhost:9200/_cluster/health`
- [ ] set_up_manual VLM JSON 존재 확인: `ls data/vlm_parsed/set_up_manual/*.json | wc -l` (15개 예상)

### Step 1: Phase 2 — Chunking
```bash
python scripts/chunk_v3/run_chunking.py \
    --vlm-dir data/vlm_parsed \
    --manifest data/chunk_v3_manifest.json \
    --output data/chunks_v3/all_chunks.jsonl
```

예상 출력:
| doc_type | 입력 | 예상 chunks |
|----------|------|------------|
| sop | 384 JSON | ~14,000 (page-based) |
| ts | 79 JSON | ~700 |
| setup_manual | 15 JSON | ~2,800 |
| myservice | ~99,000 TXT | ~80,000+ (empty 제외) |
| gcb | 1 JSON (16,354건) | ~25,000+ (split) |
| **합계** | | **~120,000+** |

검증:
```bash
wc -l data/chunks_v3/all_chunks.jsonl
python -c "
import json
from collections import Counter
c = Counter()
for line in open('data/chunks_v3/all_chunks.jsonl'):
    c[json.loads(line)['doc_type']] += 1
for k,v in c.most_common(): print(f'  {k}: {v}')
print(f'  TOTAL: {sum(c.values())}')
"
```

### Step 2: Phase 3 — Embedding (모델별 순차)

GPU 메모리: 2x A6000 (각 48GB) — VLM 서버 종료 후 사용 가능

```bash
# 모델별 순차 실행 (GPU 메모리 한 모델씩)
python scripts/chunk_v3/run_embedding.py \
    --chunks data/chunks_v3/all_chunks.jsonl \
    --models qwen3_emb_4b \
    --batch-size 128 --device cuda:0

python scripts/chunk_v3/run_embedding.py \
    --chunks data/chunks_v3/all_chunks.jsonl \
    --models bge_m3 \
    --batch-size 128 --device cuda:0

python scripts/chunk_v3/run_embedding.py \
    --chunks data/chunks_v3/all_chunks.jsonl \
    --models jina_v5 \
    --batch-size 128 --device cuda:0
```

예상 시간: 모델당 ~10-30분 (120K chunks × 1024dim)
산출물:
- `data/chunks_v3/embeddings_{model}.npy` (N × 1024 float32)
- `data/chunks_v3/chunk_ids_{model}.jsonl`

검증:
```bash
python -c "
import numpy as np
for m in ['qwen3_emb_4b', 'bge_m3', 'jina_v5']:
    v = np.load(f'data/chunks_v3/embeddings_{m}.npy')
    print(f'{m}: shape={v.shape}, dtype={v.dtype}')
    # L2 norm check
    norms = np.linalg.norm(v[:100], axis=1)
    print(f'  norm range: [{norms.min():.4f}, {norms.max():.4f}]')
"
```

### Step 3: Phase 4 — ES 적재

```bash
# 3-1. Content 인덱스 적재
python scripts/chunk_v3/run_ingest.py content \
    --chunks data/chunks_v3/all_chunks.jsonl

# 3-2. Embed 인덱스 적재 (모델별)
python scripts/chunk_v3/run_ingest.py embed \
    --model bge_m3 \
    --embeddings data/chunks_v3/embeddings_bge_m3.npy \
    --chunk-ids data/chunks_v3/chunk_ids_bge_m3.jsonl

python scripts/chunk_v3/run_ingest.py embed \
    --model qwen3_emb_4b \
    --embeddings data/chunks_v3/embeddings_qwen3_emb_4b.npy \
    --chunk-ids data/chunks_v3/chunk_ids_qwen3_emb_4b.jsonl

python scripts/chunk_v3/run_ingest.py embed \
    --model jina_v5 \
    --embeddings data/chunks_v3/embeddings_jina_v5.npy \
    --chunk-ids data/chunks_v3/chunk_ids_jina_v5.jsonl

# 3-3. 동기화 검증
python scripts/chunk_v3/run_ingest.py verify
```

### Step 4: 최종 검증

```bash
# ES 인덱스별 문서 수 확인
curl -s 'http://localhost:9200/chunk_v3_content/_count' | python -m json.tool
curl -s 'http://localhost:9200/chunk_v3_embed_bge_m3_v1/_count' | python -m json.tool
curl -s 'http://localhost:9200/chunk_v3_embed_qwen3_emb_4b_v1/_count' | python -m json.tool
curl -s 'http://localhost:9200/chunk_v3_embed_jina_v5_v1/_count' | python -m json.tool

# 샘플 kNN 검색 테스트
curl -s -X POST 'http://localhost:9200/chunk_v3_embed_bge_m3_v1/_search' \
  -H 'Content-Type: application/json' \
  -d '{"size":3,"knn":{"field":"embedding","query_vector":[0.1]*1024,"k":3,"num_candidates":50}}'
```

## 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| GPU 메모리 부족 (VLM 서버 미종료) | 임베딩 모델 로드 실패 | VLM 서버 종료 후 실행, 또는 CPU fallback |
| Qwen3-Embedding-4B 모델 다운로드 실패 | 첫 실행 시 HF 다운로드 필요 | 사전 다운로드: `huggingface-cli download Qwen/Qwen3-Embedding-4B` |
| MyService 99K 파일 chunking 느림 | Phase 2 장시간 | `--skip-vlm` / `--skip-gcb`로 분할 실행 가능 |
| ES bulk 적재 시 메모리 초과 | 적재 중단 | batch-size 500 → 200으로 줄임 |
| jina-v5 라이선스 (CC-BY-NC) | 상용 배포 불가 | 평가 비교용으로만 사용, 운영은 qwen3/bge_m3 |

## 코드 수정 필요 사항

1. **`run_embedding.py`**: Qwen3-Embedding-4B는 SentenceTransformer가 아닐 수 있음 — HF transformers 직접 로드 필요 여부 확인
2. **`run_embedding.py`**: jina-v5도 trust_remote_code=True 필요할 수 있음
3. **`run_chunking.py`**: set_up_manual 폴더명이 vlm_parsed 내 실제 이름과 일치하는지 확인
4. **`run_ingest.py`**: embed 서브커맨드의 CLI 인자가 .npy/.jsonl 경로를 올바르게 받는지 확인

## 파일 참조

| 파일 | 역할 |
|------|------|
| `scripts/chunk_v3/common.py` | ChunkV3Document 스키마, JSONL I/O |
| `scripts/chunk_v3/chunkers.py:102-227` | VLM chunker + manifest lookup |
| `scripts/chunk_v3/run_chunking.py` | Phase 2 오케스트레이션 |
| `scripts/chunk_v3/run_embedding.py:31-53` | MODEL_CONFIGS 정의 |
| `scripts/chunk_v3/run_ingest.py` | Phase 4: content/embed/verify |
| `backend/llm_infrastructure/elasticsearch/mappings.py:555-622` | ES 인덱스 매핑 |
| `backend/llm_infrastructure/embedding/engines/sentence/embedder.py` | SentenceTransformerEmbedder |
