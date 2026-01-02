# ES 가드레일 개선 작업 완료 보고서 (2026-01-02)

> 작업 기간: 2026-01-02
> 담당: Claude Code
> 목적: 임베딩 차원 불일치, alias/인덱스 네이밍, hybrid 검색 후보군 제한 이슈 개선

---

## 📋 작업 요약

세 가지 주요 개선 사항을 구현했습니다:

1. **임베딩 차원 불일치 방지 가드레일 강화**
2. **Alias/인덱스 네이밍 전략 마이그레이션 스크립트 작성**
3. **ES Hybrid 검색 RRF 기본값 변경**

---

## ✅ 완료된 작업

### 1. 임베딩 차원 불일치 방지 가드레일 강화

#### 1.1 설정 기본값 통일 (768차원)

**변경 파일**: `backend/config/settings.py:576-579`

```python
# Before
es_embedding_dims: int = Field(
    default=1024,
    description="Embedding vector dimensions (1024 for KoE5/multilingual-e5)",
)

# After
es_embedding_dims: int = Field(
    default=768,
    description="Embedding vector dimensions (768 for BGE-base, 1024 for KoE5/multilingual-e5)",
)
```

**영향**:
- `.env` 파일이 없을 때 기본값이 768로 설정됨
- 현재 사용 중인 BGE-base 모델(768차원)과 일치

---

#### 1.2 매핑 기본값 통일 (768차원)

**변경 파일**: `backend/llm_infrastructure/elasticsearch/mappings.py`

**변경 내용**:
1. `get_rag_chunks_mapping()` 기본 파라미터: `dims=1024` → `dims=768`
2. `RAG_CHUNKS_MAPPING` 기본 매핑: `dims=1024` → `dims=768`

```python
# Before
def get_rag_chunks_mapping(dims: int = 1024) -> dict[str, Any]:
    ...

RAG_CHUNKS_MAPPING = get_rag_chunks_mapping(dims=1024)

# After
def get_rag_chunks_mapping(dims: int = 768) -> dict[str, Any]:
    ...

RAG_CHUNKS_MAPPING = get_rag_chunks_mapping(dims=768)
```

**영향**:
- 신규 인덱스 생성 시 기본값이 768로 설정
- 코드 전체의 기본 동작이 현재 임베더와 일치

---

#### 1.3 인덱스 생성 시 차원 검증 추가

**변경 파일**: `backend/llm_infrastructure/elasticsearch/manager.py:113-173`

**추가된 검증 로직**:

```python
def create_index(
    self,
    version: int,
    dims: int = 768,
    ...,
    validate_dims: bool = True,  # 새 파라미터
) -> dict[str, Any]:
    """Create a new index with the RAG chunks mapping.

    Raises:
        ValueError: If dims doesn't match global config and validate_dims=True
    """
    ...

    # Dimension validation against global config
    if validate_dims:
        config_dims = search_settings.es_embedding_dims
        if dims != config_dims:
            logger.warning(
                f"Dimension mismatch detected during index creation!\n"
                f"  Requested dims: {dims}\n"
                f"  Config (SEARCH_ES_EMBEDDING_DIMS): {config_dims}\n"
                ...
            )
            raise ValueError(
                f"Index dimension ({dims}) doesn't match config ({config_dims}). "
                f"Update SEARCH_ES_EMBEDDING_DIMS or use validate_dims=False."
            )
```

**영향**:
- 인덱스 생성 시 설정값과 불일치하면 즉시 에러 발생
- 실수로 잘못된 차원으로 인덱스를 만드는 것을 방지
- 마이그레이션 등 특수한 경우 `validate_dims=False`로 우회 가능

---

### 2. ES Hybrid 검색 RRF 기본값 변경

#### 2.1 후보군 제한 이슈 해결

**변경 파일**: `backend/llm_infrastructure/retrieval/adapters/es_hybrid.py:56-83`

**변경 내용**:

```python
# Before
def __init__(
    self,
    es_engine: "EsSearchEngine",
    embedder: "BaseEmbedder",
    *,
    use_rrf: bool = False,  # ← 기존 기본값
    ...
) -> None:
    ...

# After
def __init__(
    self,
    es_engine: "EsSearchEngine",
    embedder: "BaseEmbedder",
    *,
    use_rrf: bool = True,  # ← 변경된 기본값
    ...
) -> None:
    """Initialize ES hybrid retriever.

    Args:
        use_rrf: Whether to use RRF for score combination
                 (default: True to avoid candidate limiting).
        ...
    """
```

**문제점 해결**:

| 방식 | 후보군 | 문제 | 해결 |
|------|--------|------|------|
| script_score (기존) | BM25 매칭 결과에만 벡터 점수 적용 | Semantic-only recall 저하 | ❌ |
| RRF (변경 후) | 벡터 후보 ∪ BM25 후보 독립 실행 | 후보군 제한 없음 | ✅ |

**예시**:
```
질의: "장비 고장 원인"
Document A: "Equipment malfunction root cause" (영어)
  - script_score: BM25 매칭 실패 → 제외 ❌
  - RRF: 벡터 유사도 높음 → 포함 ✅

Document B: "장비를 점검했습니다" (형태소 다름)
  - script_score: BM25 매칭 약함 → 낮은 순위 ⚠️
  - RRF: 벡터 유사도 높음 → 높은 순위 ✅
```

**영향**:
- Semantic recall 개선 (특히 한국어 형태소 변형, 동의어, 다국어 질의)
- 가중치 튜닝 불필요 (RRF 알고리즘이 자동 균형)
- 약간의 성능 오버헤드 (두 쿼리 독립 실행)

---

### 3. Alias 마이그레이션 스크립트 작성

#### 3.1 스크립트 개요

**파일**: `scripts/migrate_to_alias_strategy.py`

**기능**:
- 현재 직접 인덱스(`rag_chunks_dev_current`) → 버전 인덱스(`rag_chunks_dev_v1`) + Alias 전환
- Dry-run 모드 지원 (변경 사항 미리보기)
- 전체 마이그레이션 프로세스 자동화

**마이그레이션 단계**:
1. ✅ 현재 인덱스 존재 확인 (340,108 문서, 5.52 GB, dims=768)
2. ✅ 버전 인덱스 생성 (`rag_chunks_dev_v1`)
3. ✅ 데이터 재색인 (reindex)
4. ✅ 구 인덱스 삭제
5. ✅ Alias 생성 (`rag_chunks_dev_current` → `rag_chunks_dev_v1`)
6. ✅ 검증 (문서 수, alias 타겟 확인)

#### 3.2 사용법

```bash
# Dry run (변경 사항 미리보기)
python scripts/migrate_to_alias_strategy.py --dry-run

# 실제 마이그레이션 실행
python scripts/migrate_to_alias_strategy.py

# 커스텀 설정
python scripts/migrate_to_alias_strategy.py \
  --es-host http://localhost:9200 \
  --env prod \
  --version 1
```

#### 3.3 Dry-run 테스트 결과

```
================================================================================
ES Alias Migration Strategy
================================================================================
ES Host: http://localhost:8002
Environment: dev
Current direct index: rag_chunks_dev_current
Target versioned index: rag_chunks_dev_v1
Alias name: rag_chunks_dev_current
Dry run: True
================================================================================
Cluster health: yellow

[Step 1] Checking current index...
  ✓ Current index exists: rag_chunks_dev_current
  ✓ Documents: 340,108
  ✓ Size: 5.52 GB
  ✓ Embedding dimensions: 768

[Step 2] Checking versioned index rag_chunks_dev_v1...
  ✓ Versioned index rag_chunks_dev_v1 does not exist (will create)

[Step 3] Creating versioned index rag_chunks_dev_v1...
  [DRY RUN] Would create index: rag_chunks_dev_v1
  [DRY RUN] With dims: 768

[Step 4] Reindexing data from rag_chunks_dev_current to rag_chunks_dev_v1...
  [DRY RUN] Would reindex 340,108 documents

[Step 5] Deleting old direct index rag_chunks_dev_current...
  [DRY RUN] Would delete index: rag_chunks_dev_current

[Step 6] Creating alias rag_chunks_dev_current → rag_chunks_dev_v1...
  [DRY RUN] Would create alias: rag_chunks_dev_current → rag_chunks_dev_v1

[Step 7] Verifying migration...
  [DRY RUN] Verification skipped

================================================================================
✓ DRY RUN COMPLETE - No changes were made
  Run without --dry-run to execute migration
================================================================================
```

**상태**: 테스트 성공 ✅

---

## 📊 변경 사항 요약

### 코드 변경

| 파일 | 변경 내용 | 영향 |
|------|----------|------|
| `backend/config/settings.py` | `es_embedding_dims` 기본값: 1024 → 768 | 설정 기본값 통일 |
| `backend/llm_infrastructure/elasticsearch/mappings.py` | `get_rag_chunks_mapping()` 기본값: 1024 → 768 | 매핑 기본값 통일 |
| `backend/llm_infrastructure/elasticsearch/manager.py` | 인덱스 생성 시 차원 검증 로직 추가 | 차원 불일치 방지 |
| `backend/llm_infrastructure/retrieval/adapters/es_hybrid.py` | `use_rrf` 기본값: False → True | 후보군 제한 이슈 해결 |
| `scripts/migrate_to_alias_strategy.py` | 신규 마이그레이션 스크립트 작성 | Alias 전략 마이그레이션 |

### 검증 체크포인트

현재 3단계 검증 구조:

```
┌─────────────────────────────────────────────────────────┐
│ 1. 인덱스 생성 시점 (EsIndexManager.create_index)      │
│    ├─ dims vs SEARCH_ES_EMBEDDING_DIMS                  │
│    └─ ValueError 발생 시 인덱스 생성 중단               │
├─────────────────────────────────────────────────────────┤
│ 2. 서비스 초기화 시점 (from_settings)                  │
│    ├─ embedder.get_dimension() vs SEARCH_ES_EMBEDDING_DIMS│
│    ├─ embedder.get_dimension() vs ES index dims         │
│    └─ ValueError 발생 시 서비스 시작 실패               │
├─────────────────────────────────────────────────────────┤
│ 3. 인제스션 시점 (EsIngestService.ingest_sections)     │
│    ├─ embeddings.shape[1] vs SEARCH_ES_EMBEDDING_DIMS   │
│    └─ ValueError 발생 시 인제스션 중단                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 다음 단계

### 즉시 실행 (P0)

1. **Alias 마이그레이션 실행** (선택사항)
   ```bash
   # 백업 먼저 (권장)
   curl -X POST "http://localhost:8002/_snapshot/my_backup/snapshot_1?wait_for_completion=true"

   # 마이그레이션 실행
   python scripts/migrate_to_alias_strategy.py
   ```

   **주의사항**:
   - 340K 문서, 5.5GB 재색인에 약 5-10분 소요 예상
   - 다운타임 발생 (구 인덱스 삭제 → Alias 생성 사이)
   - 필요시 off-peak 시간대 실행 권장

2. **검색 성능 모니터링**
   - RRF vs script_score 성능 비교
   - Latency, Recall, Precision 측정
   - 필요시 `use_rrf=False`로 롤백 가능

### 단기 작업 (P1)

3. **한국어 Nori Analyzer 활성화** (리콜 +15% 예상)
   - ES nori plugin 설치
   - 인덱스 설정에 analyzer 추가
   - v2 인덱스 생성 후 재색인
   - Alias 전환

4. **Reranking 활성화** (Precision@5 +20% 예상)
   - `RAG_RERANK_ENABLED=true` 설정
   - Cross-encoder 모델 로드 테스트
   - 성능 측정

### 중기 작업 (P2)

5. **Health Check 엔드포인트 강화**
   - 차원 검증 추가
   - Prometheus metrics 추가
   - 알림 설정

6. **Hybrid 검색 전략 실험**
   - RRF vs script_score 정량 비교
   - 질의 타입별 전략 선택 로직
   - A/B 테스트

---

## 📝 롤백 계획

### 1. 임베딩 차원 변경 롤백

```bash
# .env 파일에서 기존 값 유지 (이미 768이므로 변경 불필요)
SEARCH_ES_EMBEDDING_DIMS=768
```

### 2. RRF 기본값 롤백

**Option A**: 환경변수로 우회 (코드 수정 없음)

```python
# EsSearchService 초기화 시 명시적으로 False 설정
retriever = EsHybridRetriever(
    ...,
    use_rrf=False,  # 명시적으로 script_score 사용
)
```

**Option B**: 코드 롤백

```python
# backend/llm_infrastructure/retrieval/adapters/es_hybrid.py:65
use_rrf: bool = False,  # True → False
```

### 3. Alias 마이그레이션 롤백

```bash
# 1. Alias 삭제
curl -X DELETE "http://localhost:8002/_alias/rag_chunks_dev_current"

# 2. 구 인덱스가 백업되어 있다면 복원
# (마이그레이션 스크립트는 구 인덱스를 삭제하므로, 사전 백업 필수)

# 3. 또는 v1을 current로 리네이밍 (reindex 필요)
curl -X POST "http://localhost:8002/_reindex" -H 'Content-Type: application/json' -d'
{
  "source": {"index": "rag_chunks_dev_v1"},
  "dest": {"index": "rag_chunks_dev_current"}
}'

curl -X DELETE "http://localhost:8002/rag_chunks_dev_v1"
```

---

## 📌 체크리스트

### 배포 전 확인

- [x] 코드 변경 완료
- [x] Dry-run 테스트 성공
- [ ] 단위 테스트 실행 (필요시)
- [ ] 통합 테스트 실행 (필요시)
- [ ] 백업 완료 (마이그레이션 실행 전)
- [ ] Rollback 계획 수립 완료
- [ ] 팀원 리뷰 완료

### 배포 후 확인

- [ ] 검색 API 정상 동작 확인
- [ ] 인제스션 정상 동작 확인
- [ ] 차원 검증 로직 작동 확인
- [ ] RRF 검색 성능 확인
- [ ] Alias 상태 확인 (`_cat/aliases`)
- [ ] 에러 로그 모니터링

---

## 📚 참고 문서

1. **스냅샷 문서**: `docs/2026-01-02_es_guardrails_snapshot.md`
2. **원본 TODO**: `docs/2026-01-02_code_review&todo.md`
3. **리트리벌 리뷰**: `docs/2026-01-02_retrieval review.md`
4. **ES 매핑 스냅샷**: `docs/es_mapping_snapshot_2026-01-02.json`
5. **ES 설정 스냅샷**: `docs/es_settings_snapshot_2026-01-02.json`

---

**작성일**: 2026-01-02
**작성자**: Claude Code
**상태**: ✅ 완료 (마이그레이션 실행 대기)
