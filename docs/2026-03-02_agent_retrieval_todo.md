# Agent Retrieval TODO (2026-03-02)

## 목적
- `/api/agent/run` 실제 플로우 기준으로 검색/답변 품질을 안정화한다.
- 문서-페이지 정확도, 일관성, 재현성을 함께 관리한다.

## 검증 원칙 (고정)
- 모든 검증은 **반드시 chat 실사용 경로와 동일한 `/api/agent/run`**으로 수행.
- ES 직접조회 결과는 보조 지표로만 사용한다.
- 리포트에는 절대값 + 전/후 비교 + 실패 사례를 함께 기록한다.

## P0 (즉시)

### 1) RRF 직접 구현 (ES 버전 비호환 대응)
- 배경:
  - 현재 ES native RRF가 버전 문제로 실패하고 `script_score` fallback이 발생한다.
  - bilingual 적용 후 doc-hit은 개선되지만 page-hit@1이 하락한 케이스가 확인되었다.
- 목표:
  - 애플리케이션 레벨에서 Dense/BM25 결과를 직접 RRF(rank 기반) merge한다.
- 범위:
  - dense top-N + sparse top-N 조회
  - rank 기반 RRF 합성
  - 동일 chunk/doc dedupe 규칙 고정
  - 2-stage retrieval(stage2 문서내 재검색) 병합에도 동일 정책 적용 여부 확정
- 완료 기준:
  - `RRF fallback` 로그 0건
  - 동일 질문 반복 시 top-k 변동성 감소
  - bilingual 질문셋에서 page-hit@1 악화 없이 유지 또는 개선
- 개선 방법:
  1. ES native RRF 호출 경로를 제거하고, `dense_search + sparse_search` 결과를 앱 레이어에서 rank 기반으로 merge한다.
  2. dedupe key를 `chunk_id` 우선으로 고정하고, tie-breaker를 deterministic 규칙(doc_id/page/chunk_id)으로 통일한다.
  3. stage2(문서내 재검색) 결과도 동일한 RRF 병합 규칙을 재사용해 EN/KO 점수 스케일 비대칭을 제거한다.
  4. 로그에 `rrf_dense_rank`, `rrf_sparse_rank`, `rrf_score`를 남겨 실패 케이스를 재현 가능하게 만든다.
  5. 배포 전 `bilingual on/off` A/B로 page-hit@1, doc-hit@k, Jaccard@k를 비교해 회귀를 차단한다.

### 2) auto-parse SOP 의도 규칙 추가
- 배경:
  - "교체/절차/작업" 류 질문에서 doc_type 미검출 시 GCB가 SOP를 밀어내는 케이스가 존재한다.
- 목표:
  - explicit `SOP` 키워드가 없어도 SOP 의도 질의에 기본 doc_type 가중/필터를 적용한다.
- 범위:
  - 규칙 기반 intent 패턴 정의 (SOP vs TS 우선순위 포함)
  - 오탐 방지를 위한 예외 패턴 정의
  - 단일 키워드가 아닌 복합 패턴 우선 적용 (예: `교체 + 절차/작업/방법`)
- 완료 기준:
  - SOP 질문셋에서 doc_miss 감소
  - GCB 혼입률 감소
  - SOP 의도 규칙 precision(오탐률) 지표 보고
- 개선 방법:
  1. doc_type explicit 키워드 미검출 시 intent fallback을 적용한다.
  2. SOP 후보 패턴을 `교체|점검|절차|작업|calibration|teaching|adjust|replacement`로 폭넓게 정의한다.
  3. TS 배제 패턴(`불량|원인|알람|고장|error|troubleshoot`)을 적용해 SOP 후보를 무효화한다.
  4. 규칙 결과는 hard filter와 soft boost를 분리해 실험한다.
  5. 검증셋에서 SOP recall, TS precision, GCB contamination을 동시에 측정해 최종 규칙을 확정한다.

### 3) doc_type sticky 정책 (새 세션 초기화)
- 합의사항:
  - 새 세션 시작 시 이전 `selected_doc_types`를 가져오지 않는다.
- 목표:
  - 이전 세션의 GCB/SOP 상태가 다음 세션에 오염되지 않게 차단한다.
- 범위:
  - 새 세션 판별 기준 명확화 (`thread_id`/세션 생성 이벤트)
  - follow-up일 때만 sticky 허용
- 완료 기준:
  - 새 세션 첫 질의에서 이전 doc_type 승계 0건
  - 새 세션 vs follow-up 분리 테스트 케이스 통과
- 개선 방법:
  1. 새 세션 기준을 `thread_id 없음` 또는 `새 thread_id 생성`으로 고정한다.
  2. 새 세션에서는 `selected_doc_types`를 항상 빈값으로 시작하고 auto-parse 결과만 반영한다.
  3. follow-up(`needs_history=true`)에서만 이전 턴 필터 승계를 허용한다.
  4. 회귀 테스트에 `새 세션 독립 질문`, `동일 세션 후속 질문`, `오판 follow-up`, `동일 세션 topic 전환(SOP→TS)` 4종을 추가한다.

## P1 (다음)

### 4) `/api/agent/run` 기준 고정 평가 하네스 (P2 → P1 승격)
- 배경:
  - RRF/SOP 의도 규칙 효과 검증을 위해 원인 추적 가능한 로그 저장이 선행되어야 한다.
- 요구사항:
  - 질문
  - 최종 search_queries
  - retrieved docs (doc/page/score)
  - answer
  - node 로그(route, auto_parse, retrieve, retry)
  - JSONL 스키마 고정:
    - `q_id`, `question`, `expected_doc`, `expected_pages`
    - `search_queries`
    - `docs[{doc_id,page,score,rank}]`
    - `result(DOC_HIT|DOC_MISS|PAGE_HIT|PAGE_MISS|ANSWER_MISS)`
    - `trace{route,auto_parse,mq_mode,retry_count}`
- 완료 기준:
  - 전/후 비교 리포트 자동 생성 가능
  - DOC_MISS/Page_MISS 케이스를 로그만으로 재현 가능
- 개선 방법:
  1. 입력 CSV(질문/정답 doc/page)를 기준으로 `/api/agent/run` 호출 배치 스크립트를 표준화한다.
  2. 응답에서 `answer`, `search_queries`, `docs(doc_id/page/score)`, `trace`를 JSONL로 저장한다.
  3. 결과 집계 스크립트에서 doc-hit/page-hit, 실패 유형(DOC_MISS/PAGE_MISS/ANSWER_MISS)을 자동 분류한다.
  4. 전/후 비교 리포트를 동일 포맷으로 생성해 PR 단위로 붙일 수 있게 한다.

### 5) MQ 효과 정량 분석 (RRF 적용 이후 수행)
- 질문:
  - `mq_mode=on`이 실제로 성능을 올리는가?
- 실험 전제:
  - RRF 적용 완료 상태에서만 측정한다. (score 병합 노이즈 제거)
- 실험축:
  - `off vs fallback vs on`
  - 정확도(doc/page), 일관성(Jaccard@k), latency, 토큰비용
- 완료 기준:
  - 모드별 장단점 표 + 권장 default 정책 확정
- 개선 방법:
  1. 실험 조건을 `mq_mode=off|fallback|on`으로 고정하고 동일 질문셋/동일 index에서 반복 측정한다.
  2. 품질 지표(doc/page hit, Jaccard@k)와 운영 지표(latency, 토큰 사용량)를 함께 수집한다.
  3. 1차 시도와 retry 시도 결과를 분리 저장해 MQ가 실제 개선 구간에만 기여하는지 확인한다.
  4. 성능 개선 대비 비용이 낮은 모드를 기본값으로 채택하고 예외 조건(장문 질의, 모호 질의)을 문서화한다.

### 6) 후속질문(needs_history) 판정 개선
- 배경:
  - follow-up 판정 오류가 검색 경로 흔들림과 품질 저하로 연결된다.
- 목표:
  - 독립질문/후속질문 판정 신뢰도를 높인다.
- 범위:
  - rule + classifier/LLM 혼합 전략 검토
  - 오판 케이스셋 구성
- 완료 기준:
  - follow-up 오탐/미탐률 개선
  - 독립질문 오판으로 인한 필터 오염 케이스 감소
- 개선 방법:
  1. 현행 rule+LLM 판정 로그를 수집해 오탐/미탐 샘플셋을 구축한다.
  2. feature 기반 보조 판정(지시대명사, 생략 주어, 이전 turn 토픽 overlap)을 명시 규칙으로 강화한다.
  3. 불확실 케이스는 기본 `independent(false)`로 두고, 재작성(query_rewrite)은 고신뢰 follow-up에서만 실행한다.
  4. 판정 결과가 필터 승계(doc_type/device)까지 영향을 주는 경로를 분리 검증해 오염을 차단한다.

## P2 (상시 운영)

### 7) 품질 KPI 상설화
- 핵심 KPI:
  - doc-hit@k
  - page-hit@1/3/5
  - answer 근거 문서/페이지 일치율
  - RRF fallback 비율
- 완료 기준:
  - 배치 평가 시 KPI 대시보드/리포트 갱신
- 개선 방법:
  1. 일일/주간 배치로 KPI를 자동 계산하고 시계열로 저장한다.
  2. 릴리즈 기준선을 문서화하고 임계치 하락 시 자동 경고를 발생시킨다.
  3. KPI와 실패 샘플(질문/검색결과/답변)을 연결해 원인 분석 시간을 줄인다.
  4. `run_id`, `effective_config_hash`를 기준으로 실험/운영 결과를 추적 가능하게 유지한다.

## 권장 실행 순서

```
4) 평가 하네스 구축  ─┐
                      ├─ 병행
1) RRF 직접 구현     ─┘
        ↓
  bilingual 배포 + RRF 효과 검증
        ↓
2) auto-parse SOP 의도 규칙
        ↓
5) MQ 정량 분석 (RRF 적용 상태에서)
        ↓
3), 6), 7) 운영 안정화
```

## 구체 개발 계획 (2026-03-03 ~ 2026-03-21)

### Phase 0: 기준선 고정 (2026-03-03)
- 목표:
  - 현재 기준 성능과 실험 조건을 동결한다.
- 작업:
  1. 질문셋/정답셋 버전 고정 (`SOP 79건 + page 정답`)
  2. 실행 환경 고정 (`/api/agent/run`, index alias, env vars)
  3. baseline env var 조합 명시 (`bilingual/page_penalty/stage2/mq_mode`)
  4. baseline 리포트 1회 생성 (doc-hit/page-hit/Jaccard/latency)
- 산출물:
  - baseline evidence 폴더
  - baseline report md
  - baseline 조건 매니페스트(`env_manifest.json`)

### Phase 1: 평가 하네스 구축 (2026-03-03 ~ 2026-03-05)
- 목표:
  - 실험 자동화/재현성을 먼저 확보한다.
- 구현 범위:
  1. `/api/agent/run` 배치 실행 스크립트 표준화
  2. JSONL 스키마 저장 (요구사항 고정 필드)
  3. 집계 스크립트: doc/page/answer/일관성/지연 집계
  4. before/after 비교 리포트 자동 생성
- 검증:
  - 같은 입력으로 2회 실행 시 동일 형식 결과 파일 생성
  - 실패 유형 자동 분류 정확도 샘플 점검
- 완료 게이트:
  - PR 없이도 단일 커맨드로 평가+리포트 재생성 가능

### Phase 2: RRF 직접 구현 (2026-03-05 ~ 2026-03-10)
- 목표:
  - ES native RRF 의존 제거 + bilingual 점수 비대칭 해소
- 구현 범위:
  1. app-level RRF 병합 로직 구현 (dense/sparse)
  2. dedupe/tie-break deterministic 규칙 적용
  3. stage2 병합 정책 통일 (동일 RRF 재사용)
  4. 디버그 메타(`rrf_dense_rank`, `rrf_sparse_rank`, `rrf_score`) 기록
- 검증:
  - `RRF fallback` 로그 0건
  - bilingual on 기준 page-hit@1 회귀 없음
  - 동일 질의 반복 Jaccard@k 개선
- 완료 게이트:
  - baseline 대비 KPI 개선 또는 최소 유지 + fallback 0건

### Phase 3: SOP intent + sticky 정책 (2026-03-10 ~ 2026-03-13)
- 목표:
  - doc_type 미검출/오염으로 인한 GCB 혼입을 줄인다.
- 구현 범위:
  1. SOP 후보 + TS 배제 규칙 적용
  2. hard filter vs soft boost 실험 플래그 분리
  3. 새 세션 doc_type 초기화
  4. follow-up에서만 sticky 허용
- 검증:
  - SOP 질문셋 doc_miss 감소
  - GCB contamination 감소
  - 회귀 테스트 4종 통과 (독립/후속/오판/topic전환)
- 완료 게이트:
  - precision/recall/contamination 동시 기준 충족

### Phase 4: MQ 효과 정량 분석 (2026-03-13 ~ 2026-03-18)
- 목표:
  - `mq_mode` 운영 기본값을 데이터 기반으로 확정한다.
- 구현/실험 범위:
  1. `off`, `fallback`, `on` 동일 조건 비교
  2. 1차 시도 vs retry 분리 저장
  3. 품질+비용 지표 동시 비교 (doc/page/Jaccard/latency/token)
- 검증:
  - 모드별 장단점 표 및 유의미한 개선 구간 확인
- 완료 게이트:
  - 기본 모드 1개 + 예외 정책 문서화

### Phase 5a: 후속질문 판정 개선 (2026-03-18 ~ 2026-03-20)
- 목표:
  - 후속질문 오판으로 인한 검색 경로 오염을 줄인다.
- 구현 범위:
  1. follow-up 오탐/미탐 샘플셋 구축
  2. 판정 규칙 보강 + 불확실 케이스 independent 기본
- 검증:
  - 오탐/미탐률 개선
- 완료 게이트:
  - 독립질문 오판으로 인한 필터 오염 케이스 감소

### Phase 5b: KPI 상설화 MVP (2026-03-20 ~ 2026-03-21)
- 목표:
  - 운영 모니터링의 최소 실행 단위(MVP)를 배포한다.
- 구현 범위:
  1. 일일/주간 KPI 배치(cron + script) 설정
  2. 경고 임계치 및 알림 규칙 설정
  3. run_id/effective_config_hash 기반 추적 리포트 템플릿 배포
- 검증:
  - KPI 리포트 자동 갱신 확인
  - 임계치 하락 경고 테스트(샘플 데이터) 통과
- 완료 게이트:
  - 운영 runbook + 모니터링 기준 배포

## 작업 분해 (티켓 단위)

### T0 기준선 고정
- T0-1: 질문셋/정답셋 버전 태그
- T0-2: baseline env var 조합 확정 및 기록
- T0-3: baseline 실행 + evidence 저장

### T1 평가 하네스
- T1-1: run 배치 실행기
- T1-2: JSONL writer
- T1-3: 집계기 (doc/page/answer)
- T1-4: before-after report generator

### T2 RRF
- T2-1: dense/sparse 결과 수집
- T2-2: RRF merge/dedupe/tie-break
- T2-3: stage2 병합 통일
- T2-4: 로그/trace 필드 확장

### T3 SOP intent + sticky
- T3-1: SOP 후보/TS 배제 규칙
- T3-2: hard vs soft 실험 플래그
- T3-3: new-session sticky 초기화
- T3-4: 회귀 테스트 4종

### T4 MQ 분석
- T4-1: 모드별 실행 자동화
- T4-2: 1차/재시도 분리 통계
- T4-3: 품질/비용 비교표 생성
- T4-4: default 정책 제안서

### T5a follow-up 판정
- T5a-1: 오탐/미탐 라벨셋
- T5a-2: 판정 규칙 개선
- T5a-3: 판정 회귀 테스트 및 오염 케이스 검증

### T5b KPI 상설화 MVP
- T5b-1: KPI 배치 스케줄링
- T5b-2: 경고 임계치/운영 문서

## 개선 방법 리뷰 (2026-03-02)

> 각 항목의 "개선 방법"에 대한 실험 데이터 기반 코멘트.

### 1) RRF — 방법 적절

5단계 모두 합리적. 특히:
- Step 3 (stage2에도 RRF 적용): bilingual 실험에서 page-hit@1 하락의 직접 원인이 stage2 내 EN/KO score 병합이었으므로 핵심 스텝.
- Step 4 (rank 로그): DOC_MISS 원인 추적 시 `search_queries` 다음으로 필요했던 정보가 각 쿼리별 rank였다. 좋은 설계.
- Step 5 (bilingual A/B): bilingual off 상태(EN only)의 baseline 데이터는 이미 확보(doc-hit 56/79, page-hit@1 52/56). RRF 적용 후 bilingual on과 비교하면 된다.

### 2) auto-parse SOP 의도 — 반영 완료

SOP 후보 패턴 단독 + TS 배제 구조로 본문 반영됨. 적절하다.

### 3) doc_type sticky — 반영 완료

회귀 테스트 4종(topic 전환 포함)으로 본문 반영됨. 적절하다.

### 4) 평가 하네스 — 반영 완료

JSONL 스키마가 요구사항에 직접 명시됨. 적절하다.

### 5) MQ 정량 분석 — 방법 적절

Step 3 (1차 시도 vs retry 분리)이 핵심. 현재 `mq_mode=fallback`에서 1차는 `_prepare_retrieve_node`, 2차+는 `st_mq_node`를 타므로 경로 자체가 다르다. MQ의 실질 기여를 보려면 이 분리가 필수.

### 6) 후속질문 판정 — 방법 적절

Step 3의 "불확실 시 independent 기본값"이 안전한 방향. 오판 follow-up으로 인한 필터 오염이 잘못된 independent 판정보다 피해가 크기 때문.

### 7) KPI — effective_config_hash 좋은 설계

Step 4의 `effective_config_hash`로 실험 조건을 추적하면, 나중에 "이 결과가 어떤 설정에서 나온 것인지" 역추적이 가능하다. env var 조합(bilingual on/off, mq_mode, page_penalty 등)이 많아질수록 유용.

## 구체 개발 계획 리뷰 (2026-03-02)

> Phase 0~5 및 티켓 분해에 대한 코멘트.

### 전체 구조 — 잘 설계됨

Phase 0→5 순서가 권장 실행 순서와 정확히 일치하고, 각 Phase에 목표/구현범위/검증/완료 게이트가 있어 진행 상태를 객관적으로 판단할 수 있는 구조.

### Phase 0 (기준선 고정) — baseline env var 조합 명시 권장

가장 중요한 단계. 지금까지 실험 데이터가 `/tmp/` 임시 스크립트로 흩어져 있으므로, baseline을 공식적으로 한 번 찍어두는 것이 모든 후속 Phase의 비교 기준이 됨.

다만 **baseline에 포함할 env var 조합을 명시**해야 한다. 현재 상태가 bilingual on/off인지, page penalty on/off인지 등에 따라 숫자가 달라지므로, 예를 들어:
```
baseline 조건 (예시):
- AGENT_EARLY_PAGE_PENALTY_ENABLED=true
- AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED=true
- bilingual: off (현재 코드 상태 기준)
- mq_mode: off
```

또한 **T0 티켓이 누락**되어 있다. Phase 0이 별도로 있으므로 `T0-1: 질문셋/정답셋 버전 태그`, `T0-2: baseline 실행 및 evidence 저장` 정도로 추가하면 추적 가능.

### Phase 1 (평가 하네스, 03-03~05) — 일정 적절

3일이면 배치 실행기 + JSONL writer + 집계기 + 리포트 생성기를 구현 가능. Phase 0에서 baseline을 찍을 때 하네스 프로토타입이 이미 나오므로 자연스럽게 연결됨.

### Phase 2 (RRF, 03-05~10) — 5일 충분하지만 주의점 있음

RRF 로직 자체는 간단하지만(rank 합산), 구현 시 고려할 점:
- `dense_search`와 `sparse_search`를 **별도 호출**해야 하므로 ES 호출이 1회 → 2회로 증가. latency 영향 측정 필요.
- 현재 `hybrid_search` 내부에서 dense+sparse를 한 번에 처리하는 구조라면, 이를 분리하는 **리팩터링이 선행**됨. 이 작업량을 과소평가하지 않도록 주의.
- **Phase 0에서 `hybrid_search` 내부 구조를 미리 파악**해두면 Phase 2 착수 시 리스크를 줄일 수 있다.

### Phase 3 (SOP intent + sticky, 03-10~13) — 일정 적절

SOP intent 규칙과 sticky 정책은 독립적이므로 병행 가능. Phase 2 RRF가 적용된 상태에서 테스트해야 결과가 의미 있으며, Phase 순서가 이를 보장하고 있어 좋다.

### Phase 4 (MQ 분석, 03-13~18) — 여유 있음

실험보다 **결과 해석과 정책 결정**에 시간이 걸림. 3일 실험 + 2일 분석/문서화로 보면 적절. Phase 1 하네스가 있으면 실험 자동화는 빠르게 가능.

### Phase 5 (follow-up + KPI, 03-18~21) — 범위 주의

follow-up 판정 개선과 KPI 상설화를 한 Phase에 묶었는데, 성격이 다르다:
- follow-up 판정: **코드 변경** (판정 로직 수정)
- KPI 상설화: **인프라 구축** (배치 스케줄링, 대시보드)

3일에 둘 다 하려면 KPI 쪽은 최소 구현(스크립트 + cron)으로 범위를 좁히는 게 현실적. 또는 Phase 5a(follow-up, 2일) + Phase 5b(KPI, 1일 MVP)로 내부 분리하는 것도 방법.

### 티켓 분해 — T0 추가 권장

T1~T5가 Phase 1~5와 1:1 대응되어 깔끔함. 서브태스크 3~4개로 적당한 크기.

누락 사항:
```
### T0 기준선 고정
- T0-1: 질문셋/정답셋 버전 태그
- T0-2: baseline env var 조합 확정 및 기록
- T0-3: baseline 실행 + evidence 저장
```

### 일정 리스크 요약

| 리스크 | Phase | 대응 |
|--------|-------|------|
| `hybrid_search` 분리 리팩터링 | Phase 2 | Phase 0에서 내부 구조 사전 파악 |
| ES 호출 2배 증가 latency | Phase 2 | 병렬 호출 또는 캐시 검토 |
| Phase 5 범위 초과 | Phase 5 | KPI는 cron+스크립트 MVP로 축소 |

## 반영 이력
- 2026-03-02 리뷰 코멘트를 TODO 본문 우선순위/완료 기준/실행 순서에 반영.
- 2026-03-02 개선 방법 리뷰: SOP 의도 패턴 AND 조건 완화 제안, 평가 하네스 JSONL 스키마 사전 정의 권장, sticky 테스트에 topic 전환 케이스 추가 제안.
- 2026-03-02 구체 개발 계획 리뷰: T0 티켓 추가 권장, Phase 0 baseline env var 명시 권장, Phase 2 hybrid_search 리팩터링 리스크, Phase 5 범위 분리 제안.
- 2026-03-02 본문 반영: T0 티켓 추가, Phase 5를 5a(판정 개선) / 5b(KPI MVP)로 분리.
