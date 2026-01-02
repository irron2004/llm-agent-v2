# Embedding Dimension Configuration Snapshot (2026-01-02)

## 현재 상태 요약

### 1. Elasticsearch Index (실제)
```json
{
  "embedding": {
    "type": "dense_vector",
    "dims": 768,
    "index": true,
    "similarity": "cosine"
  }
}
```
**실제 차원: 768**

### 2. Environment Variables (.env)
```bash
SEARCH_ES_EMBEDDING_DIMS=768
```
**설정 차원: 768**

### 3. Code Defaults (backend/config/settings.py:576-579)
```python
es_embedding_dims: int = Field(
    default=1024,
    description="Embedding vector dimensions (1024 for KoE5/multilingual-e5)",
)
```
**코드 기본값: 1024**
**실제 사용값: 768** (환경변수가 우선)

### 4. Embedder Registry (backend/llm_infrastructure/embedding/adapters/sentence.py)

등록된 모델과 차원:

| Alias | Model | Dimension |
|-------|-------|-----------|
| `bge_base` (기본) | BAAI/bge-base-en-v1.5 | **768** |
| `koe5` | nlpai-lab/KoE5 | **1024** |
| `multilingual_e5` | intfloat/multilingual-e5-large | **1024** |

### 5. RAG Settings (backend/config/settings.py:43-46)
```python
embedding_method: str = Field(
    default="bge_base",
    description="Embedding method name"
)
```
**기본 embedder: bge_base (768차원)**

## 불일치 위험 시나리오

### ❌ 시나리오 1: Embedder 변경 시 ES dims 불일치
```bash
# ES index는 768차원
# 사용자가 KoE5로 변경
RAG_EMBEDDING_METHOD=koe5  # 1024차원

# 결과: 차원 불일치 → 에러 또는 잘못된 결과
```

### ❌ 시나리오 2: 코드 기본값과 실제 환경 불일치
```python
# settings.py 기본값: 1024
# .env 파일 누락 시
SEARCH_ES_EMBEDDING_DIMS가 없으면 → 1024 사용
# 하지만 실제 ES index는 768
```

### ❌ 시나리오 3: Reindex 시 설정 불일치
```python
# 새 인덱스 생성 시
# SEARCH_ES_EMBEDDING_DIMS가 768인데
# RAG_EMBEDDING_METHOD=multilingual_e5 (1024차원)
# 인덱스는 768로 생성되지만 임베딩은 1024차원으로 들어감
```

## 현재 검증 상태

### ✅ 있는 검증:
- embedder.get_dimension() 메서드 존재 (runtime에서 차원 확인 가능)

### ❌ 없는 검증:
- [ ] 인덱스 생성 시 embedder dimension과 mapping dimension 비교
- [ ] 인제스천 시작 전 차원 일치 검증
- [ ] 서빙(검색) 시작 전 차원 일치 검증
- [ ] 설정 변경 시 경고 메시지

## 권장 가드레일

### 1. 인덱스 생성 시 자동 검증
```python
# backend/llm_infrastructure/elasticsearch/manager.py
def create_index_with_validation(embedder: BaseEmbedder):
    embedder_dims = embedder.get_dimension()
    config_dims = search_settings.es_embedding_dims

    if embedder_dims != config_dims:
        raise ValueError(
            f"Dimension mismatch: "
            f"embedder={embedder_dims}, config={config_dims}"
        )
```

### 2. 서비스 시작 시 검증
```python
# backend/services/es_ingest_service.py
def validate_dimensions():
    # ES index 차원
    es_dims = get_index_dims()
    # Embedder 차원
    embedder_dims = self.embedder.get_dimension()
    # 설정 차원
    config_dims = search_settings.es_embedding_dims

    assert es_dims == embedder_dims == config_dims
```

### 3. 설정 파일 검증 스크립트
```bash
# scripts/validate_embedding_config.py
# 모든 설정 값을 출력하고 불일치 감지
```

## 다음 단계

1. [ ] ES index에서 현재 dimension 읽어오는 헬퍼 함수 작성
2. [ ] EsIndexManager에 dimension 검증 로직 추가
3. [ ] EsIngestService 초기화 시 검증 추가
4. [ ] EsSearchService 초기화 시 검증 추가
5. [ ] 설정 검증 스크립트 작성 (scripts/validate_embedding_config.py)
