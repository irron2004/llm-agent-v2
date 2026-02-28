# Agent-Retrieval 통합 진단 보고서 (재검증 반영)

> 작성일: 2026-02-28  
> 통합 기준 문서:
> - `docs/2026-02-24_agent_architecture_and_debugging_report.md`
> - `docs/2026-02-26_retrieval_quality_analysis_report.md`
>  
> 이번 업데이트 추가 근거:
> - `/api/conversations` 실데이터(최근 100개 세션)
> - `/api/retrieval/run` 실시간 재현
> - `/api/agent/run` 실시간 재현

---

## 1) 핵심 결론

사용자가 제기한 문제는 **아직 재현된다**.

1. 같은 질문인데 다른 문서가 검색되는 문제: **재현됨**
2. 답변 품질 저하(질문-문서 미스매치): **재현됨**
3. MQ(멀티쿼리) 변동성 문제: **재현됨**
4. fallback/경로 일관성 문제: **재현됨(높은 가능성)**
5. agent 구조 복잡성으로 디버깅이 느린 문제: **여전히 존재**

---

## 2) 이번 재검증 범위 (2026-02-28)

### 2.1 데이터 소스

- 대화 이력: `GET /api/conversations?limit=100`
- 세션 상세: `GET /api/conversations/{session_id}`
- 검색 파이프라인 재현: `POST /api/retrieval/run`
- 실제 채팅 경로 재현: `POST /api/agent/run`

### 2.2 수집 규모

- 세션: 100
- 턴: 211
- `doc_refs` 포함 턴: 190
- `doc_refs` 미포함 턴: 21
- 동일 정규화 질문(첫 턴 기준) 반복 그룹: 20

---

## 3) 이슈별 반영 여부

| 사용자 문제 | 현재 상태 | 실증 근거 |
|---|---|---|
| 같은 질문인데 다른 문서 검색 | 발생 중 | 반복 질문 그룹에서 문서 집합 Jaccard 낮음(예: 0.046, 0.080 등) |
| 답변 품질 저하 | 발생 중 | 문서 미첨부 턴 21/211, 질문-문서 토큰 겹침 낮은 턴 다수 |
| MQ 문제 | 발생 중 | `deterministic=false`에서 `search_queries`/top-k 모두 run마다 변경 |
| fallback으로 검색 경로 비일관 | 발생 중(높은 가능성) | 동일 질문에서 `st_gate`(`no_st`/`need_st`) 분기 + fallback 쿼리 코드 존재 |
| agent 구조 비효율 | 미해결 | 노드/상태 결합도 높고 경로 실험 비용 큼 |

---

## 4) 대화 이력 기반 품질 진단 결과

### 4.1 반복 질문군의 검색 일관성

아래는 동일 질문군(첫 턴 정규화 기준)의 세션 간 문서 집합 일관성이다.

| 질의 그룹 | 세션 수 | 평균 Jaccard(첫 턴 doc_refs) | 고유 문서 패턴 수 |
|---|---:|---:|---:|
| `how can i increase the ashing rate ...` | 10 | 0.411 | 6 |
| `for a supra vplus system using dsm ...` | 6 | 0.172 | 5 |
| `tell me how to check and resolve cooling stage ...` | 6 | 0.407 | 4 |
| `apc position을 교체하는 방법` | 6 | 0.080 | 6 |
| `apc position 교체 방법` | 5 | 0.046 | 5 |
| `如何更换 apc 位置？` | 5 | 0.039 | 5 |

해석:
- 일부 그룹은 세션마다 거의 다른 문서 조합이 반환됨.
- 특히 `apc position` 계열은 일관성이 매우 낮다.

### 4.2 doc_id 체계 혼합 (경로 불안정 신호)

반복 질문군 20개 중 10개 그룹에서 첫 턴 문서가 아래처럼 혼합됐다.

- `numeric`(예: `40114268`)
- `GCB_*`
- `global_sop_*`/기타 prefixed

이는 동일 질문이더라도 검색 경로/쿼리 집합이 달라질 가능성을 시사한다.

### 4.3 검색 누락 신호

반복 그룹 중 1개는 모든 세션에서 첫 턴 `doc_refs`가 비어 있었다.

- 질의: `supra vplus 설비의 setup manual ... utility on 단계 중 pcw turn on ...`
- 세션 수: 2
- 결과: 2/2 세션 모두 문서 미첨부

### 4.4 질문-문서 적합도(휴리스틱)

질문 토큰과 `doc_refs(title+snippet)`의 겹침 비율(휴리스틱) 결과:

- best overlap 평균: 0.482
- average overlap 평균: 0.254
- best overlap <= 0.2 인 턴: 44/211 (20.9%)

주의:
- 이 지표는 빠른 1차 스크리닝용이다.
- 다국어/약어/숫자 중심 문서는 실제 적합도를 과소평가할 수 있다.

### 4.5 다국어 패러프레이즈(T2) 안정성 신호

동일 의도(`APC position 교체`)의 다국어/표현 변형이 서로 다른 반복 그룹으로 분리되었고, 각 그룹 내부 일관성도 낮다.

| 질의 | 평균 Jaccard |
|---|---:|
| `apc position을 교체하는 방법` | 0.080 |
| `apc position 교체 방법` | 0.046 |
| `如何更换 apc 位置？` | 0.039 |

해석:
- 반복 안정성(T1)뿐 아니라 패러프레이즈 안정성(T2)도 동시에 약한 상태다.
- 이 항목은 Paper B의 Paraphrase 계층 실증 근거로 직접 사용 가능하다.

---

## 5) 실시간 재현 결과

### 5.1 `/api/retrieval/run` 재현 (질문 2개 x 각 3회)

조건: `steps=["retrieve"]`, `auto_parse=false`, `rerank_enabled=false`

- `deterministic=false`
  - Query A/B 모두 `unique_top5=3`, `unique_search_queries=3`
  - `mq_strategy=llm`
- `deterministic=true`
  - Query A/B 모두 `unique_top5=1`, `unique_search_queries=1`
  - `mq_strategy=bypass`

해석:
- MQ를 쓰는 경로(`llm`)에서 검색어와 결과가 동시에 흔들린다.
- MQ bypass 시 재현 안정성은 확보된다.

### 5.2 `/api/agent/run` 재현 (동일 질문 3회)

질문: `APC position 교체 방법`

- Run1: `route=setup`, `st_gate=no_st`, top5는 숫자/sop 혼합
- Run2: `route=setup`, `st_gate=need_st`, top5는 GCB 중심
- Run3: `route=setup`, `st_gate=need_st`, top5가 Run2와 다시 다름

동시에 `search_queries`에는 run마다 다른 파라미터가 생성됨:
- `torque 15 Nm pressure 2 bar`
- `torque 50Nm`
- `5 Nm / 10 psi` 등

해석:
- 실제 채팅 경로에서도 동일 질문의 검색 결과가 안정적이지 않다.
- MQ 생성 쿼리에 근거 불명 수치가 섞이면서 recall/precision을 동시에 훼손할 가능성이 높다.
- 또한 judge 이후 retry 경로(`retry_expand`, `refine_queries`, `retry_mq`) 진입 여부가 run마다 다르면 변동성이 더 커질 수 있다.

---

## 6) 코드 관찰 요약 (현재 기준)

1. 기본 그래프는 `route -> mq -> st_gate -> st_mq -> retrieve` 경로를 사용한다.  
   (`backend/services/agents/langgraph_rag_agent.py`)
2. `run_retrieval_pipeline(... deterministic=True)`는 MQ step(`mq`, `st_gate`, `st_mq`)을 스킵한다.  
   (`backend/services/retrieval_pipeline.py`)
3. `retrieve_node`는 `search_queries`가 비면 `query_en`/원문으로 fallback한다.  
   (`backend/llm_infrastructure/llm/langgraph_agent.py`)
4. `use_canonical_retrieval`는 dead 파라미터가 아니라 **실제 분기 플래그**다.  
   (`backend/services/agents/langgraph_rag_agent.py`: `if self.use_canonical_retrieval: return self._canonical_retrieve_node(state)`)
5. `use_canonical_retrieval=True`일 때 `_canonical_retrieve_node()`가 `run_retrieval_pipeline(... steps=["retrieve"], deterministic=True, skip_mq=True)`로 실행된다.
6. 현재 FE 기본 요청은 `/api/agent/run(또는 /stream)` + `auto_parse=true`이며 `use_canonical_retrieval`를 기본으로 전달하지 않는다.  
   (`frontend/src/features/chat/hooks/use-chat-session.ts`)

---

## 7) Retry 3단계 메커니즘 (누락 보완)

`judge`가 unfaithful를 반환하면 아래 escalation이 적용된다.

| 단계 | attempt 전이 | 노드/전략 | 동작 |
|---|---|---|---|
| Tier 1 | 0 → 1 | `retry_expand` | `expand_top_k` 확장(20→40), 재검색 없이 확장 문맥 증가 |
| Tier 2 | 1 → 2 | `retry_bump -> refine_queries -> retrieve_retry` | LLM 쿼리 정제 후 재검색 |
| Tier 3 | 2 → 3 | `retry_mq -> mq -> st_gate -> st_mq -> retrieve` | MQ 완전 재생성 후 재검색 |

보강 포인트:
- Tier 2/3은 추가 LLM 호출을 포함한다.
- `TEMP_QUERY_GEN=0.3`이 `mq`, `st_mq`, `refine_queries`에 공통 적용된다.
- 따라서 MQ 변동성 + retry 변동성이 합산될 수 있다.
- 현재 응답/저장 데이터에는 `attempt`, `retry_strategy`, `search_queries(before/after)`가 충분히 남지 않아 영향 분리가 어렵다.

---

## 8) 이전(2/24) 개선 이력과 현재 코드 상태

이전 세션에서 완료로 기록된 항목과 현재 코드 상태를 분리해 정리한다.

| 항목 | 현재 코드 기준 상태 | 비고 |
|---|---|---|
| FE 채팅 경로를 `agent`로 통합 | 유지 | `chat` 라우터 대신 `/api/agent/run` 사용 |
| FE 미사용 라우터 주석 처리 | 유지 | `backend/api/main.py`에서 7개 라우터 주석 |
| `withCanonicalRetrievalDefault()` 제거 | 확인 | 현재 코드 검색 시 미검출 |
| `_build_canonical_graph` 제거 | 확인 | 함수 미검출 |
| `use_canonical_retrieval` 파라미터 잔존 | 유지(활성) | `agent.py`/`langgraph_rag_agent.py`에서 실제 전파·분기 |
| `_canonical_retrieve_node` 잔존 | 유지(활성) | `deterministic=True`, `skip_mq=True` 경로로 실행 가능 |

정리:
- "canonical graph 빌더 제거"와 "canonical retrieval 분기 비활성화"는 다른 문제다.
- 현재는 전자는 제거된 반면, 후자는 옵션 경로로 살아있다.

---

## 9) 개선 상태 판단

### 9.1 개선된 부분

- 테스트/실험용 retrieval API에서 `deterministic` 비교가 가능해 원인 분리가 쉬워졌다.
- 대화 이력 API로 실데이터 품질 진단이 가능해졌다.

### 9.2 아직 남은 핵심 문제

1. 운영 기본 경로에서 MQ 비결정성으로 결과가 흔들림
2. `st_gate`/MQ 분기 + fallback + retry가 겹치며 검색 경로 일관성이 낮음
3. 질문-문서 적합도 하위 케이스를 자동 차단하는 guardrail 부족
4. retry Tier별 효과(개선/악화)를 정량 측정하는 로깅 부재

---

## 10) 즉시 실행 권장안 (P0)

1. 운영 기본값을 `deterministic=true`로 두고, MQ는 명시 opt-in으로 전환
2. `search_queries` 생성 후 규칙 검증 추가
   - 근거 없는 수치/단위 삽입 금지
   - 원문 핵심 키워드 보존율 기준 미달 시 원문 fallback
3. 응답 저장 시 `doc_refs` 외에 아래 필드를 함께 저장
   - `route`, `st_gate`, `search_queries`, `mq_strategy`, `attempt`, `retry_strategy`
4. retry 효과 측정 지표 추가
   - Tier별 성공률
   - Tier 진입 전후 top-k Jaccard 변화
   - Tier 진입 전후 질문-문서 overlap 변화
5. 배포 전 품질 게이트 추가
   - 반복 질의 Stability@k
   - 패러프레이즈 Stability@k(T2)
   - Chat vs Retrieval 경로 parity

---

## 11) Paper B 연결 (복구)

| 본 보고서 항목 | Paper B 연결 |
|---|---|
| 섹션 4.1 반복 질문군 Jaccard | T1 Repeat 안정성 근거 |
| 섹션 4.5 다국어 패러프레이즈 저안정성 | T2 Paraphrase 안정성 근거 |
| 섹션 5.1 deterministic on/off 비교 | Stability-aware MQ 정책 효과 근거 |
| 섹션 5.2 agent 경로 재현 | 운영 경로 실효성 검증 근거 |
| 섹션 7 Retry 3단계 | 안정성 악화 요인/통제 변수 정의 |
| 섹션 10 P0 권장안 | Paper B 실험 프로토콜(가드레일+게이트) |

---

## 12) 결론

이번 재검증으로, 과거 보고서의 핵심 문제는 “과거 이슈”가 아니라 **현재 진행형**임이 확인됐다.  
다음 단계는 문서 보강이 아니라, **운영 기본 경로의 결정성 확보 + MQ/Retry guardrail + 계측 로그 강화**를 즉시 적용하는 것이다.
