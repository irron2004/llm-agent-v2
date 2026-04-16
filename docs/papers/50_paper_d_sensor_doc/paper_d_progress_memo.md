# Paper D — Progress Memo

> 작성일: 2026-04-16  
> 목적: 지금까지의 Paper D 논의를 한 번에 정리하고, 현재 상태와 다음 단계를 명확히 하기 위한 진행 메모

---

## 1. 현재 한 줄 정의

Paper D는 다음 문제를 푸는 연구로 정리되고 있다.

> **반도체 장비의 센서 상태를 event/state로 구조화하고, 이를 과거 maintenance case 및 SOP/manual evidence와 연결하여 grounded diagnosis를 수행하는 retrieval 중심 프레임워크**

즉, 일반적인 anomaly detection 전체를 새로 푸는 것이 아니라,
**sensor event ↔ maintenance case ↔ document** 연결을 retrieval 관점에서 정리하는 방향이다.

---

## 2. 지금까지 확인된 핵심 사실

### 2.1 문헌 측면
- 시계열-언어 정렬, maintenance log mining, SOP/manual grounding, semiconductor diagnosis 관련 문헌 축은 이미 정리됨
- 실재가 불분명한 후보(S2S-FDD, RAIDR 등)는 정리했고, 더 안전한 대체 문헌(FD-LLM, KEO 등)로 보강함

### 2.2 데이터 측면
- `SUPRA Vplus` + `myservice/gcb` 기준 ES 조회에서 **raw sensor name만으로는 대부분 문서와 직접 연결되지 않음**
- 직접 연결이 확인된 센서군:
  - `APC_Position`
  - `APC_Pressure`
  - `Temp1`
  - `Temp2`

### 2.3 전체 스캔 결과
- 전체 센서-문서 매칭 스캔 결과, **Temp/Heater**, **Gas/MFC**, **EPD**, **APC** 순으로 관련 문서가 많이 존재함
- raw sensor name 기준 hit만 보면 적지만,
  확장 검색과 document term 기준으로 보면 **sensor-document semantic gap**이 실제로 존재함이 보임

### 2.4 relevance 검토 결과
- `Position` single-token은 실제로 false positive 사례를 만듦
  - `40044000` → robot teaching position 문서 → `irrelevant`
- 반대로 좋은 seed/gold case 후보도 보임
  - `40036448` (`APC_Pressure`): pressure hunting + SOP 참조
  - `40042585` (`Temp1`): FDC out of spec → chuck 교체 → 정상화

---

## 3. 현재 Paper D의 가장 좋은 framing

### 하지 않는 것
- 일반적인 multivariate anomaly detection 전체를 푸는 논문
- LLM이 수치를 직접 계산하고 anomaly를 end-to-end로 판정하는 구조

### 하는 것
- sensor raw stream을 maintenance-relevant event/state로 요약
- temporal uncertainty-aware retrieval로 maintenance case를 찾음
- 관련 SOP/manual/FMEA를 grounding evidence로 연결
- failure mode ranking과 grounded diagnosis를 수행

### graph 아이디어의 위치
graph는 Paper D의 핵심 전체라기보다,

> **Eventizer / state representation을 더 잘 만들기 위한 강화층**

으로 두는 것이 가장 안전하다.

즉,
- graph-based anomaly detection 논문이 아니라
- **graph-structured state retrieval + grounded diagnosis** 쪽이 가장 적절하다.

---

## 4. 지금 단계에서 하고 있는 일

현재는 retriever를 곧바로 학습하는 단계가 아니라,
다음 작업을 하고 있다.

1. 센서명으로 문서 검색
2. exact / spaced / token combo / single token으로 후보 확장
3. 조회된 문서를 사람이 보고 `relevant / partial / irrelevant / unreviewed` 로 판정
4. relevant 문서에서 실제 현장 표현을 추출
5. validated / provisional vocabulary를 만들 준비

즉, 지금은

> **센서명과 문서 표현 사이의 간극을 실제 데이터에서 확인하고, retrieval에 쓸 vocabulary와 seed cases를 구축하는 단계**

이다.

---

## 5. 다음 단계

### 5.1 가장 먼저 할 일
1. `paper_d_keyword_query_log.md`를 계속 채우기
2. `unreviewed` 문서를 사람이 검토하기
3. `relevant` / `partial` 문서에서 반복 표현 추출하기

### 5.2 그다음 할 일
1. validated vocabulary 문서 만들기
2. lexical expansion 규칙을 보정하기
3. seed/gold case 후보를 별도 표로 정리하기

### 5.3 pilot 설계
우선순위는 다음이 적절하다.

1. **APC 계열**
2. **Temp 계열**
3. **Gas/MFC 계열**
4. **EPD 계열**

즉, 초기 pilot은 APC/Temp로 시작하는 것이 가장 안전하다.

---

## 6. 지금까지 만든 핵심 문서

- `paper_d_algorithm_design.md`
- `paper_d_map_and_graph_framing.md`
- `paper_d_es_query_results.md`
- `paper_d_keyword_query_log.md`
- `evidence/paper_d_surveyed_references.md`
- `evidence/paper_d_bibtex_priority.md`
- `evidence/paper_d_paper_comparison_table.md`

이 문서들은 각각 다음 역할을 한다.

- 알고리즘 정의
- 개념 framing
- 데이터 조회 결과 요약
- relevance 판정 작업장
- 문헌 reference / BibTeX / 비교표

---

## 7. 현재 상태 한 줄 요약

> Paper D는 지금 “아이디어 정리 단계”를 지나, **실제 센서-문서 연결 가능성을 검증하고 retrieval vocabulary와 gold case를 수집하는 단계**로 들어와 있다.
