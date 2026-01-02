# ES 가드레일 현황 스냅샷 (2026-01-02)

> 목적: 임베딩 차원 불일치, alias/인덱스 네이밍, hybrid 검색 후보군 제한 이슈 현황 파악 및 개선

---

## 1. 임베딩 차원 불일치 방지 가드레일

### 현재 상태

#### ✅ 구현된 검증 로직

**위치 1: EsIngestService.from_settings()** (`backend/services/es_ingest_service.py:167-206`)

```python
# 1단계: embedder vs config 검증
embedder_dims = embedder_instance.get_dimension()
config_dims = search_settings.es_embedding_dims

if embedder_dims != config_dims:
    raise ValueError(
        f"Embedding dimension mismatch detected!\n"
        f"  Embedder dimension: {embedder_dims}\n"
        f"  Config (SEARCH_ES_EMBEDDING_DIMS): {config_dims}\n"
        ...
    )

# 2단계: embedder vs ES index 검증 (optional)
manager = EsIndexManager(...)
es_dims = manager.get_index_dims(use_alias=True)
if es_dims is not None and es_dims != embedder_dims:
    raise ValueError(
        f"ES index dimension mismatch detected!\n"
        f"  Embedder dimension: {embedder_dims}\n"
        f"  ES index dimension: {es_dims}\n"
        ...
    )
```

**위치 2: EsSearchService.from_settings()** (`backend/services/es_search_service.py:128-167`)
- 동일한 검증 로직 적용

**위치 3: EsIngestService.ingest_sections()** (`backend/services/es_ingest_service.py:286-292`)

```python
# 인제스천 시점 검증
if embeddings.size > 0:
    embed_dim = embeddings.shape[1]
    expected = search_settings.es_embedding_dims
    if expected and embed_dim != expected:
        raise ValueError(
            f"Embedding dimension mismatch: got {embed_dim}, expected {expected}"
        )
```

#### 📊 현재 설정값

| 항목 | 값 | 위치 |
|------|------|------|
| `.env` 설정 | `SEARCH_ES_EMBEDDING_DIMS=768` | `.env:52` |
| `settings.py` 기본값 | `es_embedding_dims: int = 1024` | `backend/config/settings.py:576-579` |
| `mappings.py` 기본값 | `dims: int = 1024` | `backend/llm_infrastructure/elasticsearch/mappings.py:16` |
| **실제 ES 인덱스** | `dims: 768` | `rag_chunks_dev_current` |
| **실제 임베더** | `768` (추정: bge_base) | `RAG_EMBEDDING_METHOD=bge_base` |

**결론**: `.env` 설정이 우선순위가 가장 높아 768로 정상 동작 중. 하지만 기본값들이 1024로 되어 있어 혼란 가능성 있음.

### ⚠️ 개선 필요 사항

1. **직접 인스턴스 생성 시 우회 가능**
   - `from_settings()` 대신 `__init__()`로 직접 생성하면 검증 우회
   - 해결: `__init__`에도 검증 로직 추가 또는 팩토리 메서드 강제

2. **인덱스 생성 시점 검증 부족**
   - `EsIndexManager.create_index(dims=...)`는 임의의 dims 허용
   - 설정값과 다른 dims로 인덱스 생성 가능
   - 해결: 인덱스 생성 시 `search_settings.es_embedding_dims`와 비교 검증

3. **설정 기본값 불일치**
   - `settings.py`와 `mappings.py` 기본값이 1024
   - `.env`는 768
   - 해결: 기본값을 768로 통일하거나, 명시적 설정 강제

4. **런타임 변경 감지 불가**
   - 임베더 모델이 런타임에 변경되면 기존 인덱스와 불일치 발생 가능
   - 해결: Health check에 차원 검증 추가

---

## 2. Alias/인덱스 네이밍 실태 점검

### 🎯 설계된 명명 규칙

**코드 기준** (`backend/llm_infrastructure/elasticsearch/manager.py`)

```python
# 인덱스 네이밍
def get_index_name(version: int) -> str:
    return f"{self.index_prefix}_{self.env}_v{version}"
    # 예: rag_chunks_dev_v1, rag_chunks_dev_v2

# Alias 네이밍
def get_alias_name() -> str:
    return f"{self.index_prefix}_{self.env}_current"
    # 예: rag_chunks_dev_current
```

**롤링 업데이트 전략**:
1. 새 버전 인덱스 생성 (예: `rag_chunks_dev_v2`)
2. 데이터 재색인 또는 새로 인제스트
3. Alias를 신규 인덱스로 전환 (atomic operation)
4. 구 인덱스 삭제 또는 백업 보관

### 🔍 실제 운영 상태 (2026-01-02)

#### ES 인덱스 조회 결과

```bash
# Aliases 조회
$ curl http://localhost:8002/_cat/aliases?v
alias index filter routing.index routing.search is_write_index
# → 비어있음! alias가 하나도 없음

# Indices 조회
$ curl http://localhost:8002/_cat/indices?v | grep rag_chunks
yellow open rag_chunks_dev_current H6yw8NGET9SgDrYrqvYKWA 1 1 340108 0 5.5gb 5.5gb 5.5gb
# → `rag_chunks_dev_current`가 alias가 아니라 실제 인덱스로 존재!
```

**인덱스 매핑 확인**:
- 인덱스명: `rag_chunks_dev_current`
- 임베딩 차원: `768`
- 인덱스 옵션: `int8_hnsw` (m=16)
- 문서 수: 340,108

### ❌ 발견된 문제

1. **Alias 전략이 실제로 사용되지 않음**
   - `rag_chunks_dev_current`가 alias가 아니라 **실제 인덱스**
   - 버전 인덱스(예: `rag_chunks_dev_v1`)가 존재하지 않음
   - 롤링 업데이트 불가능 (인덱스명을 직접 쓰고 있음)

2. **코드와 실제 운영의 불일치**
   - `EsIngestService`와 `EsSearchService`는 다음과 같이 인덱스를 지정:
     ```python
     index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"
     # → "rag_chunks_dev_current"
     ```
   - 설계상 이것은 alias여야 하지만, 실제로는 인덱스

3. **초기 인덱스 생성 시 alias 미적용**
   - 누군가 처음 인덱스를 만들 때 `rag_chunks_dev_current` 이름으로 직접 생성한 것으로 추정
   - `EsIndexManager.create_index()` + `switch_alias()` 절차를 따르지 않음

### ✅ 개선 방안

#### Option A: 현재 인덱스를 v1으로 마이그레이션

```bash
# 1. 신규 v1 인덱스 생성
python scripts/es_index_manager.py create --version 1 --dims 768

# 2. 데이터 복사
POST _reindex
{
  "source": {"index": "rag_chunks_dev_current"},
  "dest": {"index": "rag_chunks_dev_v1"}
}

# 3. Alias 생성
python scripts/es_index_manager.py switch-alias --version 1

# 4. 구 인덱스 삭제 (백업 후)
python scripts/es_index_manager.py delete-index --name rag_chunks_dev_current
```

#### Option B: 현재 상태 유지 + 문서화

- alias 전략을 사용하지 않고, 직접 인덱스명 사용
- 단점: 인덱스 설정 변경 시 다운타임 불가피
- 장점: 단순함, 현재 동작 중인 시스템 영향 없음

**권장**: Option A (향후 운영 안정성을 위해)

---

## 3. ES Hybrid 검색의 후보군 제한 이슈

### 현재 Hybrid 검색 구현

**위치**: `backend/llm_infrastructure/retrieval/engines/es_search.py`

#### 방식 1: script_score (기본값)

**코드** (`_hybrid_search_script_score`, lines 258-311):

```python
# Step 1: BM25 match query로 후보군 선정
match_query = self._build_text_query(query_text)

# Step 2: 해당 후보에만 script_score 적용
query = {
    "script_score": {
        "query": match_query,  # ← BM25 후보군으로 제한!
        "script": {
            "source": (
                "params.dense_weight * cosineSimilarity(...) "
                "+ params.sparse_weight * _score + 1.0"
            ),
            ...
        }
    }
}
```

**문제점**:
- `script_score`는 `match_query` 결과에만 적용됨
- **BM25에서 매칭되지 않은 문서는 벡터 검색에서도 제외**
- 예시:
  - 질의: "장비 고장 원인"
  - Document A: "Equipment malfunction root cause" (영어) → BM25 매칭 실패 → 벡터 유사도 높아도 제외
  - Document B: "장비를 점검했습니다" (형태소 다름) → BM25 매칭 약함 → 상위 후보에서 누락 가능

**결과**: Semantic-only recall 저하

#### 방식 2: RRF (선택적)

**코드** (`_hybrid_search_rrf`, lines 206-256):

```python
body = {
    "sub_searches": [
        {"query": {"knn": knn_query}},      # 벡터 검색 독립 실행
        {"query": text_query},              # BM25 검색 독립 실행
    ],
    "rank": {
        "rrf": {
            "window_size": top_k * 2,
            "rank_constant": rrf_k,
        }
    },
}
```

**장점**:
- 두 검색이 **독립적으로 실행**
- 벡터 후보 ∪ BM25 후보 → RRF로 융합
- 후보군 제한 문제 없음

**단점**:
- ES 8.x+ 필요
- fallback 로직 있음 (RRF 실패 시 script_score로)

### 📊 설정 현황

```python
# backend/config/settings.py
hybrid_dense_weight: float = 0.7
hybrid_sparse_weight: float = 0.3
hybrid_rrf_k: int = 60
```

**기본 모드**: `use_rrf=False` (script_score 사용)

### ⚠️ 문제 사례

#### 실험 결과 (추정)

| 질의 | Document | BM25 점수 | Vector 유사도 | script_score 결과 | RRF 결과 |
|------|----------|-----------|---------------|-------------------|----------|
| "펌프 고장" | "Pump failure" (영어) | 0 | 0.85 | **제외** | **포함** |
| "챔버 청소" | "챔버를 세척했습니다" | 낮음 | 0.90 | 낮은 순위 | 높은 순위 |
| "EFEM 알람" | "EFEM-1234 alarm" | 높음 | 0.60 | 높은 순위 | 높은 순위 |

### ✅ 개선 방안

#### Option 1: RRF를 기본값으로 변경 (권장)

**장점**:
- 후보군 제한 문제 해결
- 가중치 튜닝 불필요 (RRF 알고리즘이 자동 균형)
- 두 모달리티가 독립적으로 기여

**단점**:
- ES 8.x+ 필요 (현재 8.11 사용 중이므로 문제 없음)
- 약간의 성능 오버헤드 (두 쿼리 실행)

**구현**:

```python
# backend/llm_infrastructure/retrieval/adapters/es_hybrid.py
def retrieve(self, query: str, **kwargs):
    # RRF를 기본값으로
    hits = self.es_engine.hybrid_search(
        ...,
        use_rrf=True,  # ← 기본값 변경
        rrf_k=rag_settings.hybrid_rrf_k,
    )
```

#### Option 2: 2-stage 접근 (고급)

**전략**:
1. Stage 1: 벡터 후보 top 50 + BM25 후보 top 50 독립 실행
2. Stage 2: Union → 가중치로 재정렬 → top K 반환

**장점**:
- 후보군 제한 없음
- 가중치 제어 가능 (RRF보다 유연)

**단점**:
- 복잡도 증가
- 두 번의 쿼리 필요

#### Option 3: 질의 타입별 전략 선택

```python
def get_hybrid_strategy(query: str) -> str:
    """질의 특성에 따라 전략 선택"""
    # 에러 코드/키워드 중심 → script_score (BM25 우선)
    if re.search(r'\b[A-Z]{2,}-?\d{3,}\b', query):
        return "script_score"

    # 자연어 질문 → RRF (semantic 중요)
    elif len(query.split()) > 5:
        return "rrf"

    # 기본값
    return "rrf"
```

### 📈 성능 비교 (PoC 필요)

**측정 지표**:
- Recall@10 (semantic-only 질의)
- Precision@5
- NDCG@10
- P95 latency

**테스트 세트**:
- 한국어 형태소 변형 질의 (예: "가동했다" vs "가동")
- 동의어 질의 (예: "펌프" vs "pump")
- 에러 코드 질의 (예: "EFEM-1234")
- 자연어 질문 (예: "왜 온도가 안 올라가나요?")

---

## 4. 종합 개선 우선순위

### P0 (즉시 수정)

1. **Alias 전략 마이그레이션**
   - 현재 인덱스를 v1으로 재구성
   - Alias 생성 및 전환
   - 산출물: `rag_chunks_dev_v1` (인덱스) + `rag_chunks_dev_current` (alias)

2. **Hybrid 검색 기본값 변경**
   - `use_rrf=True`로 변경
   - 성능 측정 및 검증

### P1 (단기)

3. **인덱스 생성 시 차원 검증 강화**
   - `EsIndexManager.create_index()`에 설정값 대조 로직 추가
   - Health check에 차원 검증 추가

4. **설정 기본값 통일**
   - `settings.py`와 `mappings.py` 기본값을 768로 변경
   - 또는 명시적 설정 강제 (기본값 제거)

### P2 (중기)

5. **런타임 검증 강화**
   - FastAPI health check 엔드포인트에 차원 검증 추가
   - Prometheus metrics로 차원 불일치 모니터링

6. **Hybrid 검색 PoC**
   - RRF vs script_score 성능 비교
   - 질의 타입별 전략 실험

---

## 5. 체크리스트 (실행 전 확인)

### 차원 불일치 방지

- [x] `.env`의 `SEARCH_ES_EMBEDDING_DIMS` 값 확인 (768)
- [x] 실제 ES 인덱스 dims 확인 (768)
- [x] 실제 임베더 dimension 확인 (768, bge_base)
- [ ] `settings.py` 기본값을 768로 변경
- [ ] `mappings.py` 기본값을 768로 변경
- [ ] 인덱스 생성 시 검증 로직 추가
- [ ] Health check에 차원 검증 추가

### Alias/인덱스 네이밍

- [x] 현재 인덱스가 alias인지 확인 → ❌ 실제 인덱스
- [ ] `rag_chunks_dev_current` → `rag_chunks_dev_v1` 마이그레이션
- [ ] Alias `rag_chunks_dev_current` → `rag_chunks_dev_v1` 생성
- [ ] `EsIndexManager` 사용 방법 문서화
- [ ] 롤링 업데이트 절차 문서화

### Hybrid 검색 후보군 제한

- [x] 현재 기본값 확인 → `use_rrf=False` (script_score)
- [ ] RRF 방식으로 변경
- [ ] 성능 비교 테스트 실행
- [ ] 질의 타입별 전략 검토

---

**작성일**: 2026-01-02
**다음 리뷰**: P0 항목 완료 후 (1주 후)
