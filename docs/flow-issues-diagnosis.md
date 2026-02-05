# RAG Agent Flow 문제 진단 보고서

> 작성일: 2026-02-03

## 개요

RAG Agent 파이프라인에서 발견된 문제점들과 그로 인한 flow 이상을 정리합니다.

---

## 1. MQ (Multi-Query) 파싱 로그 부족

### 현상
- MQ 생성 시 로그에 쿼리 개수만 표시되고 실제 내용이 보이지 않음
- 파싱 실패 시 원인 파악이 어려움

### 영향
- 디버깅 어려움
- MQ 품질 모니터링 불가

### 해결 (완료)
`langgraph_agent.py`의 `mq_node` 로그 개선:

```python
# 변경 전
logger.info("mq_node(retrieval/en): %d queries: %s", len(mq_en), mq_en)

# 변경 후
logger.info("mq_node(retrieval/en): %d queries generated", len(mq_en))
for i, q in enumerate(mq_en, 1):
    logger.info("  [EN-%d] %s", i, q)
if len(mq_en) == 0:
    logger.warning("mq_node(retrieval/en): parsing failed! raw output:\n%s", raw_en)
```

### 개선된 로그 출력 예시
```
mq_node(retrieval/en): 3 queries generated
  [EN-1] SUPRA N TM robot end effector mounting screw specs
  [EN-2] SUPRA N robot end effector installation torque
  [EN-3] SUPRA N TM end effector screw specification
mq_node(retrieval/ko): 3 queries generated
  [KO-1] SUPRA N robot end effector 장착 시 screw 체결 토크 스펙
  [KO-2] SUPRA N 로봇 end effector 조립 시 screw 고정 토크 스펙
  [KO-3] SUPRA N robot end effector 장착 시 screw 체결 토크 파라미터
```

---

## 2. Judge Node JSON 파싱 실패

### 현상
```
judge OUTPUT: {'judge': {'faithful': False, 'issues': ['parse_error'], 'hint': 'judge JSON parse failed'}}
```

LLM이 JSON 형식으로 응답하지 않아 `judge_node`에서 파싱 실패 발생.

### 원인 코드
`langgraph_agent.py:2217-2222`:
```python
try:
    judge = json.loads(raw)
    if not isinstance(judge, dict):
        raise ValueError("judge not dict")
except Exception:
    judge = {"faithful": False, "issues": ["parse_error"], "hint": "judge JSON parse failed"}
```

### Flow 영향

```
judge_node (파싱 실패)
    ↓
faithful=False (강제 설정)
    ↓
should_retry() 호출
    ↓
┌─────────────────────────────────────────────────────┐
│ attempts=0 → "retry_expand"  (문서 확장 5→10)       │
│ attempts=1 → "retry"         (쿼리 개선)            │
│ attempts=2 → "retry_mq"      (MQ 재생성)            │
│ attempts≥max → "human" 또는 "done"                  │
└─────────────────────────────────────────────────────┘
```

### 문제점

| 상황 | 결과 |
|------|------|
| 답변이 실제로 좋았는데 judge JSON 파싱 실패 | **불필요한 재시도 최대 3회 발생** |
| 각 재시도마다 LLM 호출 추가 | 응답 시간 2~4배 증가 |
| 비용 증가 | 불필요한 LLM 토큰 소비 |

### 근본 원인 조사

로그에서 발견된 `query_en` 이상:
```
query_en: 'We need to translate the Korean query to English.'
```

이것은 **`translate_node`에서 LLM이 번역 대신 메타 설명을 출력**한 것이 원인.
잘못된 `query_en`이 `judge_node`로 전달되어 파싱/판정 오류 발생.

### 해결 (완료)

1. **Judge 로그 개선** - 파싱 실패 시 raw LLM output 출력:
```python
except Exception as e:
    logger.warning("[judge_node] JSON parse failed: %s\nraw output:\n%s", e, raw[:500])
    judge = {"faithful": False, "issues": ["parse_error"], "hint": "judge JSON parse failed"}
```

2. **Translate 메타 설명 필터링** - LLM이 번역 대신 설명을 출력하면 원본 사용:
```python
meta_patterns = ["we need to", "let me", "i will", "here is", "the translation", ...]
if any(result_lower.startswith(p) for p in meta_patterns):
    logger.warning("translate_node: LLM returned meta-explanation: %s", result[:100])
    return text  # Fall back to original
```

### 상태
- [x] 로그 개선 적용
- [x] 메타 설명 필터링 적용
- [ ] 근본 원인 (LLM 프롬프트 품질) 추가 조사 필요

---

## 3. Elasticsearch RRF 검색 실패

### 현상
```
RRF search failed, falling back to script_score: BadRequestError(400, 'x_content_parse_exception', '[1:16288] [knn] unknown field [k]')
```

매 검색 요청마다 RRF 시도 → 실패 → script_score 폴백 발생.

### 원인
- ES 버전: 8.14.0 (RRF 지원)
- 문제: `sub_searches` 내 knn 쿼리에서 `k` 필드 구문 오류

`es_search.py:223-228`:
```python
knn_query = {
    "field": self.vector_field,
    "query_vector": query_vector,
    "k": top_k,              # ← sub_searches 컨텍스트에서 지원 안 됨
    "num_candidates": top_k * 2,
}
```

### Flow 영향

```
hybrid_search(use_rrf=True)
    ↓
_hybrid_search_rrf() 시도
    ↓
ES knn 쿼리 구문 오류 → 400 BadRequest
    ↓
⚠️ Warning 로그 출력
    ↓
_hybrid_search_script_score()로 폴백
    ↓
검색 결과 반환 (정상)
```

### 문제점

| 항목 | 영향 |
|------|------|
| Flow 중단 | ❌ 없음 (폴백 성공) |
| 검색 품질 | ⚠️ RRF 대비 약간 저하 가능 |
| 성능 | ⚠️ 매 요청마다 실패/재시도 오버헤드 |
| 로그 | ⚠️ 매 검색마다 warning 노이즈 |

### 해결 (완료)

`es_search.py`의 `_hybrid_search_rrf` 메서드에서 knn 쿼리 구문 수정:

```python
# 변경 전 (오류 발생)
knn_query = {
    "field": self.vector_field,
    "query_vector": query_vector,
    "k": top_k,              # ← ES 8.x rank API에서 지원 안 됨
    "num_candidates": top_k * 2,
}

# 변경 후 (정상 동작)
knn_query = {
    "field": self.vector_field,
    "query_vector": query_vector,
    "num_candidates": max(top_k * 2, 100),  # k 제거, num_candidates만 사용
}
# 결과 개수는 body의 "size" 파라미터로 제어
```

**핵심 수정 사항:**
- ES 8.x rank API와 함께 사용 시, knn 절에서 `k` 필드 제거
- `num_candidates`만 사용하고, 최종 결과 개수는 `size`로 제어
- `window_size`도 충분히 크게 설정 (최소 100)

### 상태
- [x] 완전 해결 (knn 쿼리 구문 수정 완료)

---

## 4. Translate Node 메타 설명 출력

### 현상
```
query_en: 'We need to translate the Korean query to English. Keep technical terms unchanged...'
```

LLM이 번역 결과 대신 **자신의 생각/지시문을 출력**.

### 원인
- 일부 LLM 모델에서 지시를 따르지 않고 메타 설명을 출력하는 경향
- `openai/gpt-oss-20b` 모델의 instruction following 품질 문제 가능성

### Flow 영향

```
translate_node
    ↓
query_en = "We need to translate..." (잘못된 값)
    ↓
mq_node: 잘못된 query_en으로 MQ 생성
    ↓
judge_node: 잘못된 query_en으로 판정
    ↓
파싱 실패 또는 오판정 → 불필요한 재시도
```

### 완화 조치 (부분 해결)

메타 설명 패턴 감지 시 원본 쿼리로 폴백:

```python
meta_patterns = [
    "we need to", "let me", "i will", "i'll", "here is", "here's",
    "the translation", "translated", "translating",
]
if any(result_lower.startswith(p) for p in meta_patterns):
    logger.warning("translate_node: LLM returned meta-explanation: %s", result[:100])
    return text  # Fall back to original query
```

### 추가 개선 방안
1. translate 프롬프트 강화 (출력 형식 더 명확히)
2. 더 나은 instruction following 모델 사용
3. 번역 결과 검증 로직 추가 (언어 감지)

### 상태
- [x] 메타 설명 필터링 적용 (완화 조치)
- [ ] 프롬프트 개선 (근본 해결 필요)
- [ ] 언어 감지 기반 검증 로직 추가

> **한계**: 현재 필터링은 특정 패턴만 감지하므로 새로운 형태의 메타 설명은 통과할 수 있습니다.
> LLM 프롬프트 품질 개선이 근본적인 해결책입니다.

---

## 요약

| 문제 | 심각도 | Flow 영향 | 상태 |
|------|--------|----------|------|
| MQ 파싱 로그 부족 | 낮음 | 없음 (디버깅 불편) | ✅ 해결 |
| Judge JSON 파싱 실패 | **높음** | 불필요한 재시도 3회 | ✅ 로그 개선 |
| ES RRF 검색 실패 | 중간 | 폴백으로 동작 | ✅ 해결 (knn 구문 수정) |
| Translate 메타 설명 | **높음** | 잘못된 query_en 전파 | ⚠️ 완화 (패턴 필터링만 적용) |

---

## 변경 파일 목록

1. `backend/llm_infrastructure/llm/langgraph_agent.py`
   - MQ 로그 개선: 각 쿼리 개별 출력 + 파싱 실패 시 raw output 출력
   - Judge 로그 개선: 파싱 실패 시 raw LLM output 출력
   - Translate 필터링: 메타 설명 패턴 감지 및 원본 폴백

2. `backend/llm_infrastructure/retrieval/engines/es_search.py`
   - `_hybrid_search_rrf`: ES 8.x rank API 호환 knn 구문으로 수정 (k 필드 제거)
