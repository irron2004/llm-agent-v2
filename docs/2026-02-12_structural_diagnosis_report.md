# 2026-02-12 구조적 진단 보고서 (RAG / Regenerate)

**작성일**: 2026-02-12  
**작성자**: AI Assistant  
**범위**: FE Regenerate 패널 ↔ BE Agent ↔ MQ/검색/재랭킹/확장 파이프라인

---

## 1) 진단 요약

현재 기능 저하는 단일 버그가 아니라, 아래 4개 축이 겹쳐 발생하고 있습니다.

1. **쿼리 품질 축**: MQ 생성/파싱 불안정(placeholder/메타문구 유입, 재시도 시 쿼리 드리프트)
2. **필터 정합성 축**: FE 선택값과 BE 실제 필터 적용/표시 방식의 불일치
3. **검색 파이프라인 축**: `retrieve → rerank → expand` 결과 표시 기준이 사용자 기대와 다름
4. **구조 축**: API/흐름이 분산·중복되어 원인 추적과 회귀 테스트가 어려움

---

## 2) 관찰된 대표 증상

- `SOP`만 선택했는데 비의도 문서가 보이거나, 반대로 REFS가 비어버림
- Regenerate 후 `search_queries`가 화면의 MQ 인식과 다르게 보임
- MQ에 `query1, query2, query3` 또는 의미 없는 메타 문자열이 포함됨
- `selected_device`가 `null`로 남아 장비 필터가 기대대로 적용되지 않음
- 검색은 많이 했다고 인지되지만, Reference Documents에는 일부만 노출됨

---

## 3) 핵심 원인 진단 (우선순위 순)

| ID | 문제 | 근본 원인 | 영향도 | 우선순위 |
|---|---|---|---|---|
| P1 | Regenerate 질의 오염 | `[Regenerate with ...]` 접두/중복이 질의에 섞이며 검색어 품질 저하 | 매우 큼 | 최우선 |
| P2 | MQ 출력 불안정 | 프롬프트 지시-실행 로직 충돌 + 파서 휴리스틱 의존 | 매우 큼 | 최우선 |
| P3 | 문서 타입 필터 혼선 | FE는 그룹명(`sop/ts/setup`)을 쓰고 BE는 variant 확장값으로 처리/반환 | 큼 | 높음 |
| P4 | 장비 필터 미적용 케이스 | `selected_device(s)` 미설정 시 일반 검색 분기까지 포함되어 타 장비 문서 유입 | 큼 | 높음 |
| P5 | 결과 표시 기준 불투명 | `retrieval_top_k` 전체가 아니라 `final_top_k/rerank/expand` 결과 중심으로 표시 | 중간~큼 | 높음 |
| P6 | 재시도 시 문서 수 불일치 | `retry_mq_node`가 `expand_top_k`를 10으로 리셋하는 경로 존재 | 중간 | 중간 |
| P7 | 문서 ID 교차필터 과제약 | Regenerate에서 `selected_doc_ids` + `doc_type` 동시 사용 시 결과 0건 가능 | 중간 | 중간 |
| P8 | API/흐름 중복 | `chat/*`와 `agent/*`가 기능적으로 겹쳐 테스트/운영 표면 증가 | 중간 | 중간 |

---

## 4) 상세 분석

### 4.1 MQ/검색어 품질 문제

- **확인된 현상**
  - `query1/query2/query3`, 메타성 문장(예: 번역 시작 문구), 과도한 설명형 질의가 유입됨
  - 동일 질문에서도 재시도 경로마다 MQ가 크게 달라짐
- **구조적 원인**
  - MQ 생성 이후 `st_mq`, `refine_queries`, `retry_mq`에서 질의가 재구성됨
  - 출력 파싱이 규칙 기반(휴리스틱)이라 모델 출력 스타일 변화에 취약
- **현재 상태**
  - placeholder/접두 제거는 보강되었지만, 근본적으로는 프롬프트-파서 계약(스키마 강제)이 아직 약함

### 4.2 문서 타입 필터 문제 (`sop` 선택 시 기대와 다른 결과)

- **확인된 현상**
  - 화면에서는 `sop` 선택인데, raw response의 `selected_doc_types`는 긴 variant 리스트로 보임
  - 케이스에 따라 REFS 0건 또는 비의도 결과 체감
- **구조적 원인**
  - FE의 canonical 그룹 키와 BE의 실제 ES 필터 키(variant) 간 표현 차이
  - 필터 “적용값”과 “응답 표시값”이 동일 레벨로 노출되지 않아 사용자가 상태를 오해하기 쉬움
  - `SOP` 그룹에 `generic`이 포함되어 있어 “일반 문서 제외” 요구와 정책 충돌 가능성 존재

### 4.3 장비 필터 문제 (`SUPRA V` 선택했는데 `OMNI` 검색)

- **확인된 현상**
  - raw response에서 `selected_device: null`이 나타난 케이스 존재
- **구조적 원인**
  - 장비 선택이 state에 반영되지 않으면 검색이 사실상 일반 검색으로 동작
  - 현재 retrieve는 장비 선택 시에도 “장비 필터 검색 + 일반 검색” 병행 전략이 있어, 필터 state가 비어 있으면 타 장비 문서 유입 가능

### 4.4 Reference Documents 개수 인식 문제

- **확인된 현상**
  - “50개 검색/10개 rerank/expand” 설정 대비 화면 표시 수가 적어 보임
- **구조적 원인**
  - 파이프라인별 산출물이 다름: `all_retrieved_docs`(후보) ≠ `retrieved_docs`(표시 기본) ≠ `expanded_docs`(답변 근거)
  - 사용자에게 “현재 숫자가 어느 단계 숫자인지” 명확히 표시되지 않음

### 4.5 재시도 전략 일관성 문제

- **확인된 현상**
  - 설정상 expand를 크게 잡아도 특정 재시도 후 문서 수가 줄어든 체감
- **구조적 원인**
  - `retry_mq` 경로에서 `expand_top_k=10`으로 리셋되는 코드 경로 존재
  - 운영자가 기대하는 전역 파라미터(예: expand=20)와 런타임 상태가 불일치

### 4.6 API 및 테스트 구조 문제

- **확인된 현상**
  - 유사 기능 API가 병존해 운영/디버깅 경로가 분산됨
  - 현재 체인은 `질문 → MQ → retrieval → answer`의 큰 단위로만 검증되는 경향
- **구조적 원인**
  - 모듈 단위 계약 테스트보다 통합 경로 중심으로 구성되어 결함 위치 특정이 늦음

---

## 5) FE → BE → LLM 전달 정합성 점검 항목 (필수)

아래 6개는 회귀 테스트의 최소 체크리스트입니다.

1. FE Regenerate 선택값(`selectedDevices`, `selectedDocTypes`, `searchQueries`)이 요청 payload에 그대로 담기는지
2. BE가 override state(`selected_*`, `search_queries`, `skip_mq`)를 정확히 반영하는지
3. 실제 retrieval 단계에서 적용된 필터와 검색어가 응답 metadata와 일치하는지
4. `selected_doc_ids` 적용 전/후 문서 수 감소가 로그로 추적 가능한지
5. `retrieved_docs / all_retrieved_docs / expanded_docs`의 의미를 UI에 분리 표기하는지
6. 재시도(`retry/refine/retry_mq`) 후 최종 `search_queries`가 화면 MQ와 동기화되는지

---

## 6) 구조 개선 권고 (작업 전 설계 관점)

### 6.1 API 정리

- `chat/*`와 `agent/*`의 역할을 분리하거나 하나로 수렴
- “실험용 API”와 “운영 API”를 명시적으로 분리

### 6.2 파이프라인 모듈화

- 현재 큰 체인을 계약 기반 단계로 분해:
  - `query_normalize`
  - `mq_generate`
  - `mq_validate`
  - `retrieve`
  - `rerank`
  - `expand`
  - `answer`
- 각 단계별 입력/출력 스키마를 고정하고 단위 테스트를 독립 실행 가능하게 구성

### 6.3 성공 지표(우선순위 기준)

- 1차 지표: **검색 성공률** (`retrieved_docs >= 1`, `REFS 비어있지 않은 응답 비율`)
- 2차 지표: 필터 정합성 (선택한 `doc_type/device` 외 문서 유입률)
- 3차 지표: MQ 품질 (placeholder/메타문구 유입률, 중복률)

---

## 7) 결론

현재 이슈의 본질은 “모델 품질” 단독 문제가 아니라, **필터/쿼리/상태 전이/표시 기준의 계약 불일치**입니다.  
따라서 해결 우선순위는 다음 순서가 맞습니다.

1. FE-BE 상태 전달 계약 고정  
2. MQ 출력 스키마 강제 + 파싱 안정화  
3. 검색 단계별 수치/표시 기준 분리 노출  
4. API 및 파이프라인 모듈 경계 재정의

