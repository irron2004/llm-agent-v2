# Hybrid Search: Client-side RRF 구현 계획

> **문서 상태**: 구현 계획서 (미구현)
> **마지막 업데이트**: 2026-02-05

## 현재 상태 요약

| 항목 | 현재 값 | 비고 |
|------|---------|------|
| `use_rrf` | `True` (ES backend 기본값) | 현재 ES RRF 시도 → 403 시 script_score fallback |
| 검색 방식 | `script_score` | 현재는 fallback으로 동작 |
| `_rrf_merge` 키 | `res.doc_id` | ⚠️ RRF 전 `chunk_id`로 remap 필요 |

---

## 1. 배경

### 1.1 현재 상황
- Elasticsearch Basic 라이선스 사용 중
- 현재 클러스터에서 ES 네이티브 RRF 호출 시 403 에러 발생 (라이선스 제한으로 추정)
- 현재는 `script_score` 방식으로 hybrid search 구현

### 1.2 현재 구현 (script_score)
```python
# backend/llm_infrastructure/retrieval/engines/es_search.py
script_source = (
    "params.dense_weight * cosineSimilarity(params.query_vector, 'embedding') "
    "+ params.sparse_weight * _score + 1.0"
)
```

**특징:**
- 1회 ES 호출로 dense + sparse 결합
- 가중치 기반: `0.7 * cosine + 0.3 * BM25 + 1.0`
- `+1.0` 오프셋으로 음수 방지

**문제점:**
- 점수 스케일이 다름 (cosine: -1~1, BM25: 0~수십)
- 가중치 튜닝이 어려움
- 스케일 정규화 필요

### 1.3 기존 RRF 구현 (재사용 대상)

**이미 존재하는 코드:**
```python
# backend/llm_infrastructure/retrieval/adapters/hybrid.py

def _rrf_merge(
    result_lists: Iterable[list[RetrievalResult]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion with optional list-wise weights."""
    score_map: dict[str, float] = defaultdict(float)
    doc_payload: dict[str, RetrievalResult] = {}

    for list_idx, results in enumerate(result_lists):
        weight = 1.0 if weights is None else weights[list_idx]
        for rank, res in enumerate(results):
            score_map[res.doc_id] += weight * (1.0 / (k + rank + 1))
            if res.doc_id not in doc_payload:
                doc_payload[res.doc_id] = res
    ...
```

**결정:** 새로 구현하지 않고 `_rrf_merge`를 재사용

**⚠️ 주의 - RRF 키 이슈:**
현재 `_rrf_merge`는 `res.doc_id`를 키로 사용함. 그러나:
- `RetrievalResult.doc_id`는 실제로 **원본 문서 ID**를 담고 있음
- 같은 문서의 여러 chunk가 동일한 doc_id를 가질 수 있음
- RRF 병합 시 같은 doc_id의 chunk들이 하나로 합쳐지는 문제 발생 가능

**해결 방안 (구현 시 선택):**
1. `_rrf_merge` 수정: `res.metadata.get("chunk_id", res.doc_id)`를 키로 사용
2. 또는: 입력 시 `RetrievalResult.doc_id`에 chunk_id를 매핑

---

## 2. RRF (Reciprocal Rank Fusion) 개요

### 2.1 RRF란?
여러 랭킹 리스트를 **순위(rank) 기반**으로 합치는 방법. 점수 스케일 정규화가 필요 없음.

### 2.2 공식
```
score(d) = Σ weight_i * (1 / (k + rank_i(d) + 1))
```

- `k`: rank_constant (기본값 60)
- `rank_i(d)`: i번째 리스트에서 문서 d의 순위 (0부터 시작, 기존 코드 기준)
- `weight_i`: i번째 리스트의 가중치

### 2.3 RRF vs script_score 비교

| 항목 | script_score (현재) | RRF (목표) |
|------|---------------------|------------|
| ES 호출 | 1회 | 2회 (_msearch) |
| 점수 기준 | 점수 가중합 | 순위 기반 |
| 정규화 | 필요 (+1.0 오프셋) | 불필요 |
| 스케일 민감도 | 높음 | 없음 |
| 튜닝 용이성 | 어려움 | 쉬움 |

---

## 3. 설계 결정사항

### 3.1 RRF 대상 키: `doc_id` vs `chunk_id` vs `_id`

**현재 ES 매핑 분석:**
```python
# es_search.py _parse_hits()
doc_id=source.get("doc_id", hit.get("_id", "")),
chunk_id=source.get("chunk_id", hit.get("_id", "")),
```

| 키 | 의미 | 중복 가능성 |
|----|------|-------------|
| `_id` | ES 문서 고유 ID | 없음 |
| `doc_id` | 원본 문서 ID | 여러 chunk가 같은 doc_id 공유 |
| `chunk_id` | 청크 ID | 없음 (unique) |

**결정: `chunk_id` 사용**
- 이유: RRF는 개별 검색 결과(chunk)를 합치는 것이므로 chunk 단위가 적합

**구현 방식:**
- `_rrf_merge`는 그대로 두고, RRF 직전에 `RetrievalResult.doc_id`를 `chunk_id`로 치환
- 원본 `doc_id`는 `metadata["doc_id"]`로 보존하여 최종 응답에서 복구

### 3.2 설정/동작 정리 (use_rrf 재정의)

- 기존 `use_rrf=True`는 ES 네이티브 RRF 시도였으나, 현재 클러스터에서는 403으로 불가
- 새 구현 이후에는 **`use_rrf=True`를 클라이언트 RRF로 사용**
- `use_rrf=False`는 기존 `script_score` 방식 유지

**추가할 설정(필요 시):**
```python
# backend/config/settings.py - RAGSettings 확장
class RAGSettings:
    # RRF parameters
    hybrid_rrf_k: int = Field(default=60)  # 기존 유지
    hybrid_rrf_window_size: int = Field(default=100)
    hybrid_rrf_sparse_weight: float = Field(default=1.0)
    hybrid_rrf_dense_weight: float = Field(default=1.0)
    hybrid_rrf_num_candidates_multiplier: int = Field(default=2)
```

> ES 네이티브 RRF를 다시 쓰고 싶으면 `use_es_rrf` 같은 **별도 플래그**로 분리하는 것을 권장.

---

## 4. 구현 설계

### 4.1 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    EsSearchEngine                        │
├─────────────────────────────────────────────────────────┤
│  hybrid_search(use_rrf=True)                            │
│       │                                                  │
│       ├─► _hybrid_search_rrf_client()  ◄── 새로 구현    │
│       │       │                                          │
│       │       ├─► _msearch (BM25 + kNN 동시 실행)       │
│       │       ├─► _rrf_merge() 재사용 (hybrid.py)       │
│       │       └─► top_k 반환                            │
│       │                                                  │
│       └─► _hybrid_search_script_score()  ◄── use_rrf=False일 때 │
└─────────────────────────────────────────────────────────┘
```

### 4.2 파일 변경 계획

```
backend/
├── config/
│   └── settings.py                    # hybrid_rrf_* 파라미터 추가 (선택)
├── llm_infrastructure/retrieval/
│   ├── adapters/
│   │   ├── hybrid.py                  # _rrf_merge 공용으로 export
│   │   └── es_hybrid.py               # use_rrf 분기 추가 (client RRF)
│   └── engines/
│       └── es_search.py               # _hybrid_search_rrf_client 추가
└── services/
    └── es_search_service.py           # rrf 파라미터 전달
```

### 4.3 핵심 구현

#### 4.3.1 `_rrf_merge` 공용화 (hybrid.py)

```python
# backend/llm_infrastructure/retrieval/adapters/hybrid.py

# 기존 함수를 모듈 레벨로 export
__all__ = ["HybridRetriever", "rrf_merge"]

def rrf_merge(
    result_lists: Iterable[list[RetrievalResult]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion with optional list-wise weights.

    기존 _rrf_merge를 public으로 변경.
    """
    # 기존 구현 그대로
    ...
```

#### 4.3.2 ES Search Engine 확장 (es_search.py)

```python
def _hybrid_search_rrf_client(
    self,
    query_vector: list[float],
    query_text: str,
    top_k: int,
    filters: dict[str, Any] | None,
    rrf_k: int = 60,
    rank_window_size: int = 100,
    weights: tuple[float, float] = (1.0, 1.0),  # (sparse, dense)
    device_boost: str | None = None,
    device_boost_weight: float = 2.0,
    num_candidates_multiplier: int = 2,  # kNN num_candidates = window_size * multiplier
) -> list[EsSearchHit]:
    """Client-side RRF를 사용한 hybrid search.

    Args:
        query_vector: 쿼리 임베딩 벡터
        query_text: 검색 쿼리 텍스트
        top_k: 최종 반환할 문서 수
        filters: ES 필터 조건
        rrf_k: RRF k 파라미터 (기본값 60)
        rank_window_size: 각 검색에서 가져올 후보 수 (>= top_k)
        weights: (sparse_weight, dense_weight) 튜플
        device_boost: 부스팅할 device_name
        device_boost_weight: device 부스팅 가중치
        num_candidates_multiplier: kNN num_candidates 배수

    Returns:
        RRF로 정렬된 검색 결과

    Raises:
        ValueError: rank_window_size < top_k인 경우
    """
    # Validation
    if rank_window_size < top_k:
        raise ValueError(f"rank_window_size ({rank_window_size}) must be >= top_k ({top_k})")

    # 1. BM25 쿼리 구성
    bm25_query = self._build_text_query(query_text, device_boost, device_boost_weight)
    if filters:
        bm25_query = {"bool": {"must": bm25_query, "filter": filters}}

    bm25_body = {
        "query": bm25_query,
        "size": rank_window_size,
        "_source": self._source_fields(),
        "track_total_hits": False,
    }

    # 2. kNN 쿼리 구성
    num_candidates = rank_window_size * num_candidates_multiplier
    knn_query: dict[str, Any] = {
        "field": self.vector_field,
        "query_vector": query_vector,
        "k": rank_window_size,
        "num_candidates": num_candidates,
    }
    if filters:
        knn_query["filter"] = filters

    knn_body = {
        "knn": knn_query,
        "size": rank_window_size,
        "_source": self._source_fields(),
        "track_total_hits": False,
    }

    # 3. _msearch로 동시 실행
    responses = self.es.msearch(
        index=self.index_name,
        body=[
            {},  # BM25 header
            bm25_body,
            {},  # kNN header
            knn_body,
        ],
    )

    # 4. 결과 파싱
    bm25_response = responses["responses"][0]
    knn_response = responses["responses"][1]

    # 에러 체크
    for resp in [bm25_response, knn_response]:
        if "error" in resp:
            raise RuntimeError(f"ES search error: {resp['error']}")

    bm25_hits = self._parse_hits(bm25_response)
    knn_hits = self._parse_hits(knn_response)

    # 5. RetrievalResult로 변환
    bm25_results = [hit.to_retrieval_result() for hit in bm25_hits]
    knn_results = [hit.to_retrieval_result() for hit in knn_hits]

    # 6. chunk_id 기준으로 RRF 키 구성 (doc_id는 metadata로 보존)
    bm25_results = _remap_doc_id_to_chunk_id(bm25_results)
    knn_results = _remap_doc_id_to_chunk_id(knn_results)

    # 7. RRF fusion (기존 함수 재사용)
    from backend.llm_infrastructure.retrieval.adapters.hybrid import rrf_merge

    fused = rrf_merge(
        [bm25_results, knn_results],
        k=rrf_k,
        weights=list(weights),
    )

    # 8. top_k 적용 및 EsSearchHit으로 변환
    top_results = fused[:top_k]

    # RetrievalResult → EsSearchHit 변환
    return [
        EsSearchHit(
            doc_id=(r.metadata.get("doc_id") if r.metadata else r.doc_id),
            chunk_id=(r.metadata.get("chunk_id") if r.metadata else r.doc_id),
            content=r.content,
            score=r.score,
            page=r.metadata.get("page") if r.metadata else None,
            metadata=r.metadata or {},
            raw_text=r.raw_text,
        )
        for r in top_results
    ]
```

**헬퍼 스케치 (doc_id → chunk_id remap):**
```python
def _remap_doc_id_to_chunk_id(results: list[RetrievalResult]) -> list[RetrievalResult]:
    remapped: list[RetrievalResult] = []
    for r in results:
        meta = dict(r.metadata or {})
        meta.setdefault("doc_id", r.doc_id)
        chunk_id = meta.get("chunk_id") or r.doc_id
        remapped.append(RetrievalResult(
            doc_id=chunk_id,
            content=r.content,
            score=r.score,
            metadata=meta,
            raw_text=r.raw_text,
        ))
    return remapped
```

### 4.4 설정 변경

**현재 동작 기준:**
- `use_rrf`는 EsHybridRetriever / API 파라미터에서 제어
- 기본값은 True (ES backend)
- 새 구현 이후 `use_rrf=True` → 클라이언트 RRF, `use_rrf=False` → script_score

**추가할 설정(필요 시):**
```python
# backend/config/settings.py (추가)
class RAGSettings(BaseSettings):
    # ... 기존 설정 ...

    # RRF parameters
    hybrid_rrf_k: int = Field(default=60)
    hybrid_rrf_window_size: int = Field(
        default=100,
        description="Number of candidates from each retriever for RRF"
    )
    hybrid_rrf_sparse_weight: float = Field(default=1.0)
    hybrid_rrf_dense_weight: float = Field(default=1.0)
    hybrid_rrf_num_candidates_multiplier: int = Field(
        default=2,
        description="kNN num_candidates multiplier"
    )
```

---

## 5. 파라미터 가이드

### 5.1 rrf_k (rank_constant)

| 값 | 효과 | 사용 케이스 |
|----|------|-------------|
| 10~30 | 상위권 집중 | 한쪽 retriever 결과가 더 신뢰할 만할 때 |
| 60 (기본) | 균형 | 일반적인 경우 |
| 100+ | 완만한 합성 | 하위권 문서도 고려하고 싶을 때 |

**권장:** 60부터 시작, 검색 품질 보면서 조정

### 5.2 rank_window_size

| 값 | 장점 | 단점 |
|----|------|------|
| 50 | 빠름 | recall 낮을 수 있음 |
| 100 (권장) | 균형 | - |
| 200+ | recall 높음 | 느림, 메모리 사용 증가 |

**제약조건:** `rank_window_size >= top_k` 필수
**권장:** `final_top_k * 5` (예: top_k=20이면 window=100)

### 5.3 num_candidates_multiplier

kNN 검색의 `num_candidates = rank_window_size * multiplier`

| 값 | 효과 |
|----|------|
| 1 | 최소 (빠르지만 recall 낮음) |
| 2 (기본) | 균형 |
| 3~5 | recall 향상 (더 느림) |

**참고:** ES 공식 문서에서 num_candidates를 충분히 크게 설정할 것을 권장

### 5.4 weights (가중치)

| sparse_weight | dense_weight | 사용 케이스 |
|---------------|--------------|-------------|
| 1.0 | 1.0 | 기본 (동등 가중치) |
| 1.5 | 1.0 | 키워드 매칭 중시 (짧은 쿼리) |
| 1.0 | 1.5 | 의미 유사도 중시 (긴 자연어 쿼리) |

**권장:** 1:1로 시작, A/B 테스트로 조정

---

## 6. 마이그레이션 가이드

### 6.1 현재 상태

```bash
# 현재 .env 설정 (예시)
# RAG_HYBRID_RRF_K=60
```

### 6.2 구현 후 설정

```bash
# 클라이언트 RRF 파라미터
RAG_HYBRID_RRF_K=60
RAG_HYBRID_RRF_WINDOW_SIZE=100
RAG_HYBRID_RRF_SPARSE_WEIGHT=1.0
RAG_HYBRID_RRF_DENSE_WEIGHT=1.0
RAG_HYBRID_RRF_NUM_CANDIDATES_MULTIPLIER=2

# 기존 script_score 유지 시: API 요청에 use_rrf=false 사용
# 예: /api/search?use_rrf=false
```

### 6.3 하위 호환성

| 시나리오 | 동작 |
|----------|------|
| `use_rrf` 파라미터 없음 | 기본값에 따라 동작 (ES backend는 True → client RRF) |
| `use_rrf=false` | script_score 방식 |
| `use_rrf=true` | 클라이언트 RRF 방식 |

---

## 7. 체크리스트

### 7.1 구현 전
- [ ] 현재 script_score 검색 품질 baseline 측정
- [ ] 테스트 쿼리셋 준비
- [x] ES `_id`와 `doc_id`/`chunk_id` 관계 확인 → `chunk_id` 사용 결정

### 7.2 구현 중
- [ ] RRF 직전에 `_remap_doc_id_to_chunk_id` 적용 (chunk 기준 fuse)
- [ ] `hybrid.py`의 `_rrf_merge`를 `rrf_merge`로 export
- [ ] `es_search.py`에 `_hybrid_search_rrf_client` 추가
- [ ] `settings.py`에 `hybrid_rrf_*` 파라미터 추가 (선택)
- [ ] `EsHybridRetriever`에서 `use_rrf` 분기 추가 (client RRF ↔ script_score)

### 7.3 구현 후
- [ ] 단위 테스트 작성 (rrf_merge, _hybrid_search_rrf_client)
- [ ] 통합 테스트: script_score vs client RRF (use_rrf=false/true) 결과 비교
- [ ] 성능 측정 (latency 증가량)
- [ ] 파라미터 튜닝 (rrf_k, window_size, weights)

---

## 8. 예상 효과

### 8.1 장점
1. **점수 스케일 문제 해결**: 순위 기반이라 정규화 불필요
2. **튜닝 용이**: rrf_k, weights로 직관적 조정
3. **기존 코드 재사용**: `_rrf_merge` 활용
4. **디버깅 용이**: 각 retriever의 순위를 명확히 추적 가능

### 8.2 단점
1. **네트워크 오버헤드**: 2회 검색 (단, _msearch로 1 RTT)
2. **메모리 사용**: rank_window_size만큼 후보 보관
3. **코드 복잡도**: 설정 분기 추가

### 8.3 성능 예상
- Latency: +10~30ms (msearch 오버헤드)
- 검색 품질: 동등 또는 향상 (스케일 문제 해결로)

---

## 9. 참고 자료

- [Elastic RRF Retriever 문서](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever)
- [Elastic Hybrid Search 가이드](https://www.elastic.co/search-labs/blog/hybrid-search-elasticsearch)
- [RRF 공식 설명](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [kNN Search 문서](https://www.elastic.co/docs/solutions/search/vector/knn)
- 기존 구현: `backend/llm_infrastructure/retrieval/adapters/hybrid.py`
