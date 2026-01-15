# Golden Set Pooling 스크립트 가이드

**작성일**: 2026-01-08
**목적**: 검색 품질 평가용 Golden Set 구축을 위한 Pooling 스크립트 사용법

---

## 1. 개요

### 1.1 목적

15개 테스트 질문에 대해 다양한 검색 방식(BM25, Dense, Hybrid, Stratified)으로 후보 문서를 수집하여 Annotation 준비를 위한 Pool을 생성합니다.

### 1.2 Pooling이란?

338K+ chunks 전체를 사람이 라벨링할 수 없으므로, 여러 검색 방식의 상위 결과만 모아서 평가 대상을 축소하는 방법입니다.

```
전체 문서 338K → Pooling → 질문당 ~150개 → 사람이 라벨링
```

---

## 2. 파일 구조

```
scripts/golden_set/
├── __init__.py
├── config.py                    # 데이터 모델 정의
├── convert_questions.py         # TypeScript → JSONL 변환
├── create_pools.py              # 메인 CLI 스크립트
├── deduplication.py             # Near-duplicate 제거
└── pooling/
    ├── __init__.py
    ├── base.py                  # PoolingStrategy 추상 클래스
    ├── bm25_pooler.py           # BM25 (Sparse) 검색
    ├── dense_pooler.py          # Dense (Vector) 검색
    ├── hybrid_pooler.py         # Hybrid (RRF) 검색
    └── stratified_pooler.py     # doc_type별 다양성 보장

data/golden_set/
├── queries.jsonl                # 입력: 15개 질문
├── pools/
│   ├── q001.jsonl               # 출력: 질문별 Pool
│   ├── q002.jsonl
│   └── ...
└── pool_stats.json              # 통계 요약
```

---

## 3. 사용법

### 3.1 Step 1: 질문 변환

Retrieval Test Lab의 TypeScript 질문을 JSONL로 변환합니다.

```bash
python scripts/golden_set/convert_questions.py
```

**입력**: `frontend/src/features/retrieval-test/data/test-questions.ts`
**출력**: `data/golden_set/queries.jsonl`

### 3.2 Step 2: Pool 생성

```bash
# 기본 실행
python scripts/golden_set/create_pools.py

# 옵션 지정
python scripts/golden_set/create_pools.py \
    --queries data/golden_set/queries.jsonl \
    --output data/golden_set/pools/ \
    --pool-size 150 \
    --top-k 50 \
    --min-per-type 10

# ES 호스트 지정 (로컬 개발 시)
python scripts/golden_set/create_pools.py --es-host http://localhost:9200
```

### 3.3 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--queries` | `data/golden_set/queries.jsonl` | 입력 질문 파일 |
| `--output` | `data/golden_set/pools/` | Pool 출력 디렉토리 |
| `--es-host` | settings에서 로드 | Elasticsearch 호스트 |
| `--pool-size` | 150 | 질문당 최대 Pool 크기 |
| `--top-k` | 50 | 검색 방식당 top-k |
| `--min-per-type` | 10 | doc_type당 최소 문서 수 |
| `--similarity-threshold` | 0.85 | Near-duplicate 임계값 |
| `--embedder` | settings에서 로드 | Embedder 이름 (koe5 등) |

---

## 4. Pooling 방식

### 4.1 4가지 검색 방식

| 방식 | 설명 | 구현 |
|------|------|------|
| **BM25** | 키워드 기반 희소 검색 | `EsSearchEngine.sparse_search()` |
| **Dense** | 벡터 유사도 기반 밀집 검색 | `EsSearchEngine.dense_search()` |
| **Hybrid** | Dense + BM25 결합 (RRF) | `EsSearchEngine.hybrid_search()` |
| **Stratified** | Hybrid + doc_type별 최소 보장 | Hybrid 후 부족한 type 추가 |

### 4.2 Stratified Pooling 동작

```
1. Hybrid 검색으로 top_k 문서 획득
2. doc_type별 개수 확인
3. min_per_type 미달 type에 대해 필터 검색으로 추가
```

### 4.3 Near-Duplicate 제거

- TF-IDF 기반 cosine similarity 계산
- doc_type별 임계값 적용:
  - `maintenance_log`: 0.90 (더 엄격)
  - `sop`, `ts_guide`: 0.75
  - `setup`: 0.70
- 각 클러스터에서 score 높은 문서만 유지

---

## 5. 출력 포맷

### 5.1 Pool 파일 (JSONL)

`data/golden_set/pools/q001.jsonl`:

```jsonl
{"_meta": true, "query_id": "q001", "query_text": "ECOLITE3000...", "category": "troubleshooting", "difficulty": "hard", "pool_stats": {"total_before_dedup": 180, "total_after_dedup": 147, "by_doc_type": {"sop": 45, "ts_guide": 38}}, "created_at": "2026-01-08T10:30:00"}
{"chunk_id": "sop_ecolite3000#0012", "doc_id": "sop_ecolite3000", "content": "PM Chamber...", "score": 0.95, "source_method": "hybrid", "doc_type": "sop", "metadata": {...}}
{"chunk_id": "ts_guide_arcing#0005", "doc_id": "ts_guide_arcing", "content": "Arcing 발생...", "score": 0.92, "source_method": "bm25", "doc_type": "ts_guide", "metadata": {...}}
```

### 5.2 통계 파일

`data/golden_set/pool_stats.json`:

```json
{
  "created_at": "2026-01-08T10:30:00",
  "config": {
    "top_k_per_method": 50,
    "total_pool_size": 150,
    "min_per_doc_type": 10,
    "similarity_threshold": 0.85,
    "embedder": "koe5",
    "index": "rag_chunks_dev_current"
  },
  "queries": [
    {
      "query_id": "q001",
      "query_text": "ECOLITE3000 설비에서...",
      "total_before_dedup": 180,
      "total_after_dedup": 147,
      "by_doc_type": {"sop": 45, "ts_guide": 38, "maintenance_log": 42, "setup": 22}
    }
  ]
}
```

---

## 6. 테스트 질문 목록

| ID | 카테고리 | 난이도 | 질문 요약 |
|----|---------|--------|----------|
| q001 | troubleshooting | hard | ECOLITE3000 PM Chamber Local Plasma/Arcing 원인 |
| q002 | information | easy | GENEVA XP APC 관련 GCB 번호 |
| q003 | information | medium | SEC SRD EPA404 LL 점검 이력 |
| q004 | information | medium | mySERVICE EPA404 LL 점검 이력 |
| q005 | information | medium | SEC SRD EPA404 LL MYSERVICE 점검 이력 |
| q006 | setup | easy | SUPRA N Baffle Screw 토크 스펙 |
| q007 | setup | easy | SUPRA N TM ROBOT ENDEFFECTOR Screw 토크 스펙 |
| q008 | troubleshooting | medium | SUPRA III APC Pressure Hunting 점검 포인트 |
| q009 | troubleshooting | hard | SUPRA N APC Position 이상 점검 포인트 |
| q010 | information | medium | SUPRA Np Issue GCB 기반 정리 |
| q011 | information | easy | SUPRA V→Np 개조 설비 호기명 |
| q012 | information | medium | EPAGQ03 Source Unready Alarm 이력 |
| q013 | troubleshooting | medium | INTEGER Main Rack Door Open Interlock 해결 |
| q014 | information | easy | SUPRA Vplus APC Sensor Part Number |
| q015 | setup | hard | INTEGER Plus PM PIN Motor Pin 높이/S/W 설정 |

**분포**:
- 카테고리: information(8) / troubleshooting(4) / setup(3)
- 난이도: easy(5) / medium(7) / hard(3)

---

## 7. 다음 단계

### 7.1 Annotation

Pool 생성 후, 각 문서에 Graded Relevance (0~3) 부여:

| Grade | 의미 | 예시 |
|-------|------|------|
| 3 | Must-have (필수) | 직접 답/절차 포함 |
| 2 | Should-have (필요) | 사전 작업/배경지식 |
| 1 | Nice-to-have (참고) | 과거 사례/유사 장비 |
| 0 | Not relevant | 키워드만 겹침 |

### 7.2 groundTruthDocIds 업데이트

Annotation 완료 후, Grade 2~3 문서를 `test-questions.ts`의 `groundTruthDocIds`에 반영합니다.

### 7.3 평가 실행

Golden Set 완성 후, 검색 품질 지표 계산:
- Recall@k
- nDCG@k
- MRR

---

## 8. 의존성

### 8.1 Python 패키지

```python
# 필수 (이미 설치됨)
elasticsearch          # ES 클라이언트
numpy                  # 벡터 연산
scikit-learn          # TF-IDF, 유사도 계산
tqdm                  # 진행률 표시
```

### 8.2 프로젝트 내부 모듈

```python
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
from backend.llm_infrastructure.embedding import get_embedder
from backend.config.settings import search_settings, rag_settings
```

---

## 9. 트러블슈팅

### ES 연결 실패

```bash
# ES 호스트 확인
curl http://localhost:9200

# 인덱스 확인
curl http://localhost:9200/_cat/indices?v
```

### Embedder 로드 실패

```bash
# 사용 가능한 embedder 확인
python -c "from backend.llm_infrastructure.embedding import EmbedderRegistry; print(EmbedderRegistry.list_methods())"
```

### Pool 크기가 너무 작음

- `--top-k` 값 증가 (기본 50 → 100)
- `--min-per-type` 값 확인

---

## 참고 문서

- [Golden Set 구축 전략](./2026-01-07_retrieval_golden_set_strategy.md)
- [Retrieval 아키텍처 리뷰](./2026-01-02_retrieval%20review.md)
