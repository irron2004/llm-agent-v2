# Agent Bilingual Query 누락 버그 보고서

> 작성일: 2026-03-01
> 영향 범위: `/api/agent/run` 경로 전체 (실제 chat)
> 심각도: **Critical** — 문서 정확도 70.9% (정상 100%)
> 수정: `_prepare_retrieve_node`에 `query_ko` 추가

---

## 1. 현상

`/api/agent/run` (실제 chat 경로)으로 79건 SOP 질문 평가 시:

| 지표 | `/api/search` (검색 전용) | `/api/agent/run` (실제 chat) |
|---|---|---|
| 문서 hit@10 | 73/73 (100%) | **56/79 (70.9%)** |
| DOC_MISS | 0건 | **23건** |

**동일한 질문, 동일한 ES 인덱스인데 정확도가 30% 차이**.

---

## 2. 원인 추적

### 2.1 Query 흐름 비교

```
/api/search:
  사용자 입력 → "ZEDIUS XP 설비의 PENDULUM VALVE 교체 작업" (한국어 그대로)
  → ES BM25 on search_text (한국어) → ✓ 정답 문서 찾음

/api/agent/run:
  사용자 입력 → translate_node → query_en + query_ko 생성
  → prepare_retrieve → search_queries = [query_en]  ← 한국어 누락!
  → retrieve_node → ES BM25 (영문만) → ✗ 한국어 SOP 내용과 미스매치
```

### 2.2 버그 위치

`backend/services/agents/langgraph_rag_agent.py:415-424`

```python
def _prepare_retrieve_node(self, state: AgentState) -> Dict[str, Any]:
    # query_en만 사용, query_ko를 버림
    stable_query = state.get("query_en") or state.get("query")
    update["search_queries"] = [stable_query]  # ← 영문 1개만!
```

### 2.3 왜 이런 코드가 되었나

`_prepare_retrieve_node`는 `mq_mode=off` 또는 `fallback` 1차 시도에서 MQ(Multi-Query) 생성을 건너뛰고 직접 검색하는 노드. 원래 의도는 "단일 안정 쿼리로 deterministic 검색"이었으나, **bilingual을 고려하지 않고 `query_en`만 사용**.

반면 MQ 경로(`st_mq_node`)는 EN + KO 쿼리를 합쳐서 `search_queries`에 넣기 때문에 정상 동작:

```python
# st_mq_node (정상 동작)
english_queries = setup_mq_list + ts_mq_list + general_mq_list
korean_queries = setup_mq_ko_list + ts_mq_ko_list + general_mq_ko_list
search_queries = english_queries + korean_queries  # ← EN + KO 모두 포함
```

### 2.4 영향받는 경로

| mq_mode | 경로 | bilingual |
|---|---|---|
| `off` | `prepare_retrieve` → `retrieve` | **EN만 (버그)** |
| `fallback` (1차) | `prepare_retrieve` → `retrieve` | **EN만 (버그)** |
| `fallback` (2차+) | `mq` → `st_mq` → `retrieve` | EN+KO (정상) |
| `on` | `mq` → `st_mq` → `retrieve` | EN+KO (정상) |

**기본값 `mq_mode=fallback`의 1차 시도가 영향받으므로 대부분의 요청이 영문 전용 검색.**

---

## 3. 수정

### 3.1 코드 변경

```python
# 수정 후
def _prepare_retrieve_node(self, state: AgentState) -> Dict[str, Any]:
    stable_query_en = " ".join((state.get("query_en") or state.get("query") or "").split()).strip()
    stable_query_ko = " ".join((state.get("query_ko") or state.get("query") or "").split()).strip()
    update: Dict[str, Any] = {
        "skip_mq": True,
        "mq_used": False,
        "mq_reason": None,
    }
    # Bilingual: include both EN and KO for BM25 coverage
    queries = []
    if stable_query_en:
        queries.append(stable_query_en)
    if stable_query_ko and stable_query_ko != stable_query_en:
        queries.append(stable_query_ko)
    if queries:
        update["search_queries"] = queries
    return update
```

### 3.2 검증

수정 후 `search_queries` 확인:

```json
// Before (영문만)
"search_queries": ["PENDULUM VALVE replacement work for the ZEDIUS XP equipment"]

// After (bilingual)
"search_queries": [
  "PENDULUM VALVE replacement work for the ZEDIUS XP equipment",
  "ZEDIUS XP 설비의 PENDULUM VALVE 교체 작업"
]
```

### 3.3 단위 테스트

15/15 PASS (기존 테스트 전부 통과).

---

## 4. 평가 결과

### 4.1 수정 전 (영문 전용)

| 지표 | 값 |
|---|---|
| 문서 hit@10 | 56/79 (70.9%) |
| DOC_MISS | 23건 |
| page-hit@1 (문서 hit 중) | 52/56 (92.9%) |
| first_page=1 | 0건 |

### 4.2 수정 후 (bilingual)

| 지표 | 수정 전 (EN only) | 수정 후 (EN+KO) | Δ |
|---|---|---|---|
| 문서 hit@10 | 56/79 (70.9%) | 60/79 (75.9%) | +4 |
| DOC_MISS | 23건 | 19건 | -4 |
| page-hit@1 (전체) | 51/79 (64.6%) | 38/79 (48.1%) | -13 |
| page-hit@1 (문서 hit 중) | 52/56 (92.9%) | 38/60 (63.3%) | -18 |
| first_page=1 | 0건 | 0건 | 0 |

### 4.3 bilingual 부작용 분석

문서 hit은 개선(+4)되었으나, **page-hit@1이 하락**. 원인:
- EN/KO 2개 쿼리가 각각 다른 chunk를 반환 → 병합 순서에서 잘못된 페이지가 1위로
- 예: Q5 — EN 쿼리가 page 19(REP 섹션), KO 쿼리가 page 6(ADJ 섹션) 반환 → page 19가 1위
- 예: Q28 — expected 6-27이지만 page 28이 1위 (1페이지 차이)

**근본 원인**: 2단계 재검색에서 EN/KO 각각 top-5를 가져와 점수순 병합하는데,
EN 쿼리 결과와 KO 쿼리 결과의 점수 스케일이 달라 순서가 흔들림.

**개선 방향**: 2단계에서는 **KO query만 사용** (SOP 콘텐츠가 한국어이므로),
또는 EN/KO 결과를 RRF(Reciprocal Rank Fusion)로 병합.

---

## 5. 추가 발견: doc_type 필터 부재

bilingual 수정과 별개로, agent `auto_parse`가 **`doc_types: null`**을 반환하여 SOP 질문에서도 GCB 문서가 함께 검색됨.

```json
// auto_parse 결과 (Q67: TM DC Power SUPPLY 교체 작업)
{
  "device": "ZEDIUS XP",
  "doc_type": null,      // ← SOP로 감지해야 하는데 null
  "doc_types": null
}
```

이로 인해 GCB 문서(40050558 등)가 SOP 문서보다 높은 점수를 받아 정답 문서를 밀어내는 경우 발생.

**추후 개선**: auto_parse에서 "교체/조정/calibration" 패턴 → `doc_type=SOP` 자동 설정.

---

## 6. 타임라인

| 시점 | 사건 |
|---|---|
| 최초 구현 | `prepare_retrieve` 노드 작성 시 `query_en`만 사용 |
| 2026-02-27 | `mq_mode=fallback` 기본값 설정 → 대부분 요청이 영문 전용 경로 |
| 2026-03-01 | 79건 SOP 평가에서 DOC_MISS 23건 발견 |
| 2026-03-01 | `_prepare_retrieve_node`에 `query_ko` 추가하여 수정 |
