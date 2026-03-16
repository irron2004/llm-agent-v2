# Retrieval 개선을 위한 doc_type별 문서 특징 브리프

날짜: 2026-03-16
대상: 외부/내부 retrieval 전문가, RAG agent 개선 검토자
목적: 현재 RAG agent의 retrieval 개선 논의를 위해, `doc_type`별 문서 형태와 검색상 함의를 한 문서로 정리한다.

## 1. 왜 이 문서가 필요한가

현재 agent는 상위 수준에서 크게 두 흐름으로 나뉜다.

- `이슈조회`: `myservice + gcb + ts`
- `절차조회`: `setup + SOP`

이 구분은 제품 UX 관점에서는 단순하고 합리적이다.
하지만 retrieval 관점에서는 같은 그룹 안의 문서 구조가 상당히 다르다.
특히 `이슈조회` 내부의 `myservice`, `gcb`, `ts`는 형태가 크게 달라서,
같은 검색 전략과 같은 ref 조립 방식으로 처리하면 recall과 answer quality가 동시에 손실될 수 있다.

이 문서는 전문가가 아래 질문에 답할 수 있도록 작성했다.

1. 어떤 `doc_type`에 어떤 retrieval 전략이 맞는가
2. 어떤 `doc_type`은 문서 단위 fetch가 필요하고, 어떤 `doc_type`은 page/section 단위 확장이 맞는가
3. 어떤 `doc_type`은 analyzer, field, reranking, diversity policy를 별도로 가져가야 하는가

## 2. 현재 제품이 사용하는 canonical doc_type 그룹

현재 코드 기준 canonical group은 아래 5개다.
근거: `backend/domain/doc_type_mapping.py`

| canonical group | 대표 raw 값 | 현재 제품 의미 |
| --- | --- | --- |
| `myservice` | `myservice` | 유지보수/서비스 이력 |
| `gcb` | `gcb`, `maintenance` | 이슈 케이스/분석/커뮤니케이션 이력 |
| `ts` | `Trouble Shooting Guide`, `trouble shooting`, `Guide`, `ts` | 트러블슈팅 가이드 |
| `setup` | `Installation Manual`, `setup`, `Manual`, `set_up_manual` | 설치/셋업 매뉴얼 |
| `SOP` | `SOP`, `Global SOP`, `SOP/Manual`, `generic`, `에스오피` | 표준 작업 절차 문서 |

현재 route/UX 관점의 큰 구분은 아래와 같다.

- `task_mode=issue`: `myservice`, `gcb`, `ts`
- `task_mode=sop`: `setup`, `SOP`

즉, 현재 제품은 doc_type별 route를 따로 두기보다, `issue` vs `sop` 2축으로 먼저 나누고 있다.
전문가 검토 포인트는 “이 상위 구분은 유지하되, retrieval 내부 전략은 doc_type-aware로 세분화할 필요가 있는가”이다.

## 3. 공통 전제: 현재 인덱스/검색 구조

현재 문서들은 chunk 단위로 Elasticsearch에 적재되어 있고, 주요 특성은 아래와 같다.

- chunking: fixed-size 기반
- 주요 검색 필드: `content`, `search_text`, `doc_type`, `device_name`, `equip_id`, `page`, `chapter`
- analyzer: `nori`
- 현재 이슈 경로는 hybrid retrieval 이후 chunk 기반으로 refs를 조립한다

retrieval 관점에서 중요한 공통 사실은 아래와 같다.

1. `doc_type`마다 “한 문서의 정보가 chunk에 분산되는 방식”이 다르다.
2. `device_name`, `equip_id`는 공통 신호로 쓸 수 있지만, 내용 구조는 매우 다르다.
3. 현재 확장 로직도 문서 종류에 따라 다르게 동작한다.
   - `myservice`, `gcb`, `pems`: 같은 `doc_id` 전체를 다시 fetch하는 쪽이 맞다고 가정
   - 일반 manual 계열: page window 확장이 기본

즉, 이미 코드에도 “문서 종류에 따라 retrieval 후처리 전략이 달라야 한다”는 전제가 일부 들어가 있다.

## 4. doc_type별 문서 특징

### 4.1 `myservice`

#### 문서 성격

- 유지보수/서비스 이력
- 사건 단위 로그에 가깝다
- 전체 corpus에서 비중이 가장 높다
- 같은 유형의 짧은 이력 문서가 대량으로 존재한다

#### 실제 구조

하나의 이슈가 보통 아래 4개 섹션으로 분리 저장된다.

- `status`: 증상/현상
- `cause`: 원인
- `action`: 조치
- `result`: 결과

중요한 특징:

- `status`, `cause`는 매우 짧다
- `action`이 가장 정보량이 많은 경우가 많다
- 한 이슈의 정보가 4개 chunk로 나뉘므로, 1개 chunk만 hit되면 맥락이 깨진다

#### retrieval 관점의 시사점

- 높은 빈도 때문에 BM25/dense 모두 `myservice`로 과도하게 쏠리기 쉽다
- 같은 증상 표현이 반복되어 top-k diversity가 쉽게 무너진다
- `cause` 단독 hit는 설명력이 낮고, `action` 단독 hit는 증상이 없는 채로 나올 수 있다

#### retrieval 전략 제안

- chunk-level retrieval 후 `doc_id` 기준 section merge가 사실상 필수
- top-k에서 `myservice` cap 또는 diversity quota가 필요할 수 있음
- symptom/alarm code/device/equip exact match는 강하게 활용 가능
- recency가 의미가 있다면 `updated_at` 기반 가중치를 검토할 가치가 있음

#### 전문가에게 묻고 싶은 포인트

- `myservice` 대량 편중을 줄이면서도 recall을 잃지 않는 quota/diversity 전략
- `status/cause/action/result`를 late fusion할지, index 단계에서 doc-level companion field를 만들지
- 짧은 `cause` chunk를 reranker가 과대평가/과소평가하지 않게 하는 방법

### 4.2 `gcb` / `maintenance`

#### 문서 성격

- 이슈 케이스 기록
- 엔지니어 간 커뮤니케이션 이력
- 분석 결과와 타임라인이 들어간 장문 보고서 성격
- `myservice`보다 문서 1건의 정보 밀도가 훨씬 높다

#### 실제 구조

실제 적재 데이터는 대체로 아래 조합이다.

- `summary`
- `detail`
- `timeline`
- 일부 문서에 `background`, `cause` 등 추가 섹션

중요한 특징:

- `summary`는 짧고 식별용 성격이 강하다
- `detail`, `timeline`은 길고 정보 밀도가 높다
- 한 문서가 2~3개 이상의 chunk로 분리된다
- 영문 비중이 높다

#### retrieval 관점의 시사점

- `summary`만 hit되면 상세 원인/조치가 빠진다
- `detail`만 hit되면 사건의 식별자나 배경이 약할 수 있다
- 영문 중심 데이터인데 nori analyzer 기반 검색이라 BM25 정밀도 손실 가능성이 있다
- `myservice`보다 문서 수는 적지만, 실제 root cause 정보는 더 풍부한 경우가 많다

#### retrieval 전략 제안

- `doc_id` 기준 doc-level fetch 또는 section bundle이 매우 중요
- query expansion에서 영어/한글 이중 검색어를 더 의식해야 함
- `summary`는 retrieval anchor, `detail/timeline`은 answer evidence로 쓰는 2단계 전략이 유효할 수 있음
- 영문용 sub-field 또는 analyzer 보강을 별도 검토할 가치가 있음

#### 전문가에게 묻고 싶은 포인트

- 영문 case log에 대해 현재 hybrid retrieval의 BM25 비중을 어떻게 조정할지
- `summary/detail/timeline` 섹션별 가중치를 retrieval과 reranking에서 다르게 둘지
- `timeline` 장문 chunk를 passage로 유지할지, 더 잘게 자를지

### 4.3 `ts`

#### 문서 성격

- Trouble Shooting Guide
- 과거 사례 모음보다 “표준 진단 절차서”에 가깝다
- 이슈 질의에 매우 유용하지만 corpus 비중은 작다

#### 실제 구조

대체로 아래 형태를 가진다.

- failure symptom
- possible cause
- check point
- key point
- recommended action

즉, 구조화된 진단 표와 체크리스트형 문서다.

#### retrieval 관점의 시사점

- 문서 수가 매우 적어 raw top-k 경쟁에서 `myservice`에 쉽게 밀린다
- 그러나 hit만 되면 답변 가치가 높다
- 케이스 기반 매칭보다 symptom-to-procedure 매칭이 중요하다
- device가 `ALL`인 경우가 많아 device filtering만으로는 잘 안 걸러진다

#### retrieval 전략 제안

- issue query라도 `ts`는 절차 가이드형 evidence로 우대하는 보정이 필요할 수 있음
- symptom, alarm, check, reset, mitigation 계열 query expansion이 특히 중요
- `ts` 전용 field boost 또는 reranker prior를 별도로 둘지 검토할 가치가 있음
- `ts only` 선택 시에는 case search보다 procedure-like retrieval/answering이 더 적합할 수 있음

#### 전문가에게 묻고 싶은 포인트

- low-volume/high-value doc_type을 top-k에서 살리는 안정적인 mixing 전략
- `ts`를 issue corpus 안의 한 종류로 둘지, 별도 retrieval profile로 둘지
- symptom table형 문서를 위한 chunking/reranking 최적화 포인트

### 4.4 `setup`

#### 문서 성격

- 설치/셋업 매뉴얼
- 절차와 페이지 연속성이 강한 manual 문서
- 특정 단계 전후 문맥이 중요하다

#### 실제 구조

- 설치 순서
- 장비 구성
- 점검 항목
- page/section 연속성이 있는 문서

#### retrieval 관점의 시사점

- 개별 문장보다 페이지 주변 맥락이 중요하다
- 같은 `doc_id` 전체보다 현재 page 주변 window를 확장하는 방식이 더 자연스럽다
- 절차성 헤더, step 번호, warning/caution이 중요한 신호다

#### retrieval 전략 제안

- page-window expansion 유지가 타당
- section/chapter continuity를 우선시하는 것이 좋음
- query가 절차형이면 문서 내 step-heavy page를 우선시키는 reranking이 유효할 수 있음

#### 전문가에게 묻고 싶은 포인트

- page-level retrieval과 doc-level retrieval의 혼합 기준
- setup/manual에서 section-aware chunking이 현 구조보다 유리한지

### 4.5 `SOP`

#### 문서 성격

- 표준 작업 절차 문서
- setup보다 더 canonical하고 규범적인 절차 문서
- 작업 순서, 안전, 체크리스트가 핵심이다

#### 실제 구조

- 목적/범위
- 준비/안전
- 작업 절차
- 확인/복구/체크리스트

#### retrieval 관점의 시사점

- 절차 질문에서는 높은 precision이 중요하다
- 유사 문서가 여러 버전으로 존재할 수 있어 title/section quality가 중요하다
- page local context와 section 제목 신호가 특히 중요하다

#### retrieval 전략 제안

- heading-aware retrieval이 중요
- `work procedure`, `workflow`, `checklist`, `precaution` 같은 section cue를 살릴 필요가 있음
- exact step을 찾는 query와 개괄 절차를 묻는 query를 구분하면 더 좋아질 수 있음

#### 전문가에게 묻고 싶은 포인트

- SOP류 문서에서 chunk_summary, chapter, title을 어떻게 조합하면 precision이 좋아지는지
- 절차형 질의에서 semantic retrieval보다 lexical/heading prior를 더 강하게 둘지

## 5. retrieval 설계상 핵심 차이

전문가가 바로 볼 수 있게 차이를 표로 요약하면 아래와 같다.

| doc_type | 문서 단위 성격 | 정보 분산 방식 | retrieval 핵심 리스크 | 우선 검토 전략 |
| --- | --- | --- | --- | --- |
| `myservice` | 짧은 서비스 로그 | 같은 이슈가 4섹션으로 분리 | 대량 편중, 짧은 chunk, 중복 사례 | doc merge, diversity cap, exact match 강화 |
| `gcb` | 장문 케이스/분석 보고서 | summary/detail/timeline 분리 | summary-only hit, 영문 검색 손실 | doc fetch, bilingual query, section weighting |
| `ts` | 구조화된 진단 가이드 | symptom/check/action 구조 | 저빈도라 top-k에서 밀림 | low-volume boost, issue-specific MQ, ts-aware rerank |
| `setup` | 설치 매뉴얼 | page/section 연속성 | 주변 페이지 맥락 손실 | page-window, section continuity |
| `SOP` | 표준 절차 문서 | heading/step 중심 | precision 저하, 버전 혼선 | heading-aware retrieval, section cue boost |

## 6. 현재 agent 관점에서 특히 중요한 판단

### 6.1 지금 구조가 완전히 틀린 것은 아니다

현재의 큰 구분인

- `이슈조회 = myservice/gcb/ts`
- `절차조회 = setup/SOP`

은 제품 관점에서 유지할 수 있다.

문제는 이 상위 구분 안에서 retrieval 내부 전략이 아직 너무 균일하다는 점이다.
즉, 전문가에게 요청할 방향은 “route를 무조건 더 쪼개라”보다 아래에 가깝다.

- 상위 `task_mode`는 유지 가능
- 하지만 retrieval query generation, field weighting, chunk merge, reranking, diversity policy는 doc_type-aware로 조정 필요

### 6.2 특히 `issue` 내부는 세 종류를 같은 데이터처럼 다루면 안 된다

- `myservice`: 대량, 짧음, 섹션 분산
- `gcb`: 장문, 영문, 케이스 분석
- `ts`: 소량, 구조화, 진단 절차

이 셋은 역할이 다르다.
따라서 retrieval 전문가에게는 “issue 검색 품질 개선”을 요청하되, 실제 검토 단위는 `issue` 하나가 아니라 `myservice/gcb/ts` 세 갈래로 쪼개서 봐 달라고 전달하는 것이 맞다.

## 7. 전문가에게 요청할 우선 질문

아래 질문에 답을 받는 것이 실무적으로 가장 가치가 크다.

1. `myservice` 편중을 줄이면서 `gcb`와 `ts`를 안정적으로 top-k에 포함시키는 retrieval/diversity 전략은 무엇인가
2. `myservice`와 `gcb`는 chunk retrieval 이후 `doc_id` 단위 section merge를 어느 단계에서 하는 것이 가장 좋은가
3. `gcb` 영문 데이터에 대해 analyzer 또는 sub-field를 어떻게 보강하는 것이 가장 효과적인가
4. `ts`처럼 low-volume/high-value corpus를 현재 hybrid retrieval에서 살리기 위한 prior 또는 reranker 설계는 무엇이 좋은가
5. `setup`과 `SOP`는 page/heading 중심 retrieval을 어떻게 최적화할 것인가
6. doc_type별로 query expansion 템플릿을 분리하는 것이 실제 성능 차이를 만드는가

## 8. 당장 공유해야 할 핵심 요약

- 현재 제품의 상위 구분은 `issue(myservice/gcb/ts)` vs `sop(setup/SOP)`다.
- 그러나 retrieval 관점에서 `myservice`, `gcb`, `ts`는 데이터 구조가 매우 다르다.
- `myservice`는 짧고 많으며 section merge와 diversity 제어가 중요하다.
- `gcb`는 영문 장문 케이스라 doc-level fetch와 analyzer 보강이 중요하다.
- `ts`는 소량이지만 가장 구조화되어 있어 low-volume boost가 필요하다.
- `setup`과 `SOP`는 page/section continuity가 중요해 manual형 retrieval 전략이 더 맞다.

즉, 전문가에게는 “route를 새로 나눌지”보다 먼저, “doc_type별 retrieval profile을 어떻게 다르게 가져갈지”를 검토해 달라고 요청하는 것이 적절하다.

## 9. 참고 근거

- `backend/domain/doc_type_mapping.py`
- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/tests/test_expand_related_docs_node.py`
- `docs/es-schema.md`
- `docs/2026-03-14-이슈조회-성능-개선-연구.md`
- `docs/2026-03-14-issue-route-data-aware-design.md`
