# ES RRF 검색 쿼리 오류 수정

## 날짜
2026-02-03

## 증상
- 모든 검색 쿼리에서 "관련 문서를 찾지 못했습니다" 에러 발생
- `retrieved_docs: []` - 검색 결과가 항상 비어있음
- Judge에서 `parse_error` 발생 (문서가 없어서 답변 생성 실패)
- 재생성 버튼이 표시되지 않음

## 원인 분석

### 1차 원인: ES 쿼리 구조 변경 오류

`backend/llm_infrastructure/retrieval/engines/es_search.py`의 `_hybrid_search_rrf` 메서드에서 ES 8.x RRF 쿼리 구조가 잘못 변경됨.

**변경 전 (정상 작동):**
```python
body = {
    "sub_searches": [
        {"query": {"knn": knn_query}},
        {"query": text_query},
    ],
    "rank": { "rrf": {...} },
}
```

**잘못된 변경:**
```python
body = {
    "knn": knn_query,
    "query": text_query,
    "rank": { "rrf": {...} },
}
```

→ ES 8.x RRF는 `sub_searches` 형식이 필요. 단순히 `knn`과 `query`를 top-level에 두면 RRF가 작동하지 않음.

### 2차 원인: `k` 파라미터 호환성 문제

`sub_searches` 형식으로 복원했지만, knn 쿼리 안에 `k` 파라미터를 포함시킴:

```python
knn_query = {
    "field": ...,
    "query_vector": ...,
    "k": top_k,           # ← 이게 문제!
    "num_candidates": ...,
}
```

**ES 에러 로그:**
```
BadRequestError(400, 'x_content_parse_exception', '[1:16281] [knn] unknown field [k]')
WARNING: RRF search failed, falling back to script_score
```

→ ES 8.x에서 `sub_searches` 안의 knn은 `k` 파라미터를 **지원하지 않음**

### 왜 fallback도 실패했나?

`script_score` fallback이 있었지만, 결과가 0개로 나온 이유:
- fallback은 작동했지만 (`status:200`)
- 검색 쿼리 자체의 문제가 아닌 다른 이슈가 있었을 가능성
- 또는 fallback 로직의 검색 결과가 필터링되어 0개가 됨

## 해결 방법

### 최종 수정된 코드

```python
knn_query: dict[str, Any] = {
    "field": self.vector_field,
    "query_vector": query_vector,
    "num_candidates": max(top_k * 2, 100),  # k 제거! num_candidates만 사용
}

body: dict[str, Any] = {
    "sub_searches": [
        {"query": {"knn": knn_query}},
        {"query": text_query},
    ],
    "rank": {
        "rrf": {
            "window_size": max(top_k * 2, 100),
            "rank_constant": rrf_k,
        }
    },
    "size": top_k,  # 최종 결과 개수는 size로 제어
    "_source": self._source_fields(),
}
```

### 핵심 변경사항
1. `sub_searches` 배열 형식 사용 (ES 8.x RRF 필수)
2. knn 쿼리에서 `k` 파라미터 **제거**
3. `num_candidates`만 사용 (충분히 크게: `max(top_k * 2, 100)`)
4. 최종 결과 개수는 `size` 파라미터로 제어

## 영향 범위
- 모든 문서 검색 기능
- 채팅 답변 생성
- Retrieval Test
- Batch Answer

## 관련 파일
- `backend/llm_infrastructure/retrieval/engines/es_search.py`

## ES 8.x RRF 쿼리 참고사항

ES 8.x에서 RRF (Reciprocal Rank Fusion)를 사용한 하이브리드 검색 시:

| 항목 | 설명 |
|------|------|
| `sub_searches` | 필수. 각 검색 쿼리를 배열로 분리 |
| knn 쿼리 형식 | `{"query": {"knn": {...}}}` 형태로 감싸야 함 |
| `k` 파라미터 | **sub_searches 안에서는 지원 안 됨** (에러 발생) |
| `num_candidates` | knn 후보 개수 (충분히 크게 설정) |
| `size` | 최종 결과 개수 제어 |
| `window_size` | RRF 윈도우 크기 (num_candidates와 비슷하게) |

## 재발 방지
- ES 쿼리 구조 변경 시 반드시 Docker 환경에서 테스트 후 배포
- ES 버전별 API 차이 확인 필요
- 검색 결과가 비어있는 경우 로그 확인 (fallback 여부, 에러 메시지)

## 디버깅 팁

검색이 안 될 때 확인할 로그:
```bash
docker logs rag-api --tail=100 2>&1 | grep -i "rrf\|error\|warning\|retrieve"
```

정상 작동 시 로그:
```
retrieve_node: collected 20 unique docs before rerank
retrieve_node: returning 20 docs
```

오류 시 로그:
```
WARNING: RRF search failed, falling back to script_score: BadRequestError(400, ...)
retrieve_node: collected 0 unique docs before rerank
```
