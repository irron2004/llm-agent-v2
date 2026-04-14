# Paper D — 논문 전략 및 프레이밍

> 작성일: 2026-04-14

---

## 1. 핵심 프레이밍

### 강한 버전 (추천)
> 실제 반도체 센서와 실제 정비로그를 이용한
> - event-centric alignment benchmark
> - temporal-uncertainty-aware retrieval
> - grounded failure diagnosis

### 약한 버전 (피해야 함)
- "반도체 데이터에 anomaly detection 적용"
- "로그도 같이 참고"
- "LLM agent 데모"
- "실제 데이터를 써봤다" (단독 기여로 불충분)

---

## 2. 기여 포인트 3가지

### 2.1 데이터 기여
- 실제 반도체 설비 센서 trace + 실제 정비로그를 연결한 event-centric 연구 세트
- 공개 데이터는 work order와 센서가 함께 있는 경우가 극히 드묾
- 비공개 데이터여도, **문제 정의와 정렬 프로토콜**만 잘 쓰면 기여 성립

### 2.2 방법 기여
- 센서 이상 window와 정비로그를 **시간 불확실성 하에서** 연결하는 retrieval/alignment
- 정비로그 시각 ≠ 실제 고장 시작시각 → temporal compatibility 모델링
- 단순 nearest timestamp가 아닌 구조화된 정렬

### 2.3 실무 기여
- 현재 episode에 대해 failure mode 후보 + 관련 과거 정비사례를 함께 제시
- 센서 신호 → 자연어 요약 → historical maintenance documents 활용

---

## 3. 논문 분할 전략 (3편)

### 1편: 데이터셋/문제정의
**제목 방향**: *Linking Semiconductor Tool Sensor Episodes and Maintenance Logs under Temporal Uncertainty*

**내용**:
- 실제 반도체 센서 + 실제 정비로그 소개
- Event-centric 샘플링 방식
- Gold / silver / weak link 정의
- Baseline 정렬 결과

**특징**: 빠르게 낼 수 있음. 공개 데이터 scarcity 자체가 배경.

### 2편: Retrieval (박사논문 중심 챕터)
**제목 방향**: *Temporal-Uncertainty-Aware Retrieval of Maintenance Cases from Semiconductor Sensor Windows*

**내용**:
- Sensor encoder + Log encoder
- Temporal compatibility score
- Top-k maintenance case retrieval

### 3편: Grounded Diagnosis
**제목 방향**: *Grounded Failure Diagnosis in Semiconductor Equipment using Retrieved Maintenance Cases and SOP Evidence*

**내용**:
- Retrieved log + SOP/manual
- Failure mode ranking
- 근거 기반 설명

---

## 4. 초록 중심 메시지

```
Semiconductor equipment maintenance relies on heterogeneous evidence
scattered across multivariate sensor traces and unstructured maintenance
logs. However, these sources are rarely aligned at the episode level,
and maintenance timestamps often do not correspond exactly to fault onset.
We propose an event-centric framework that links semiconductor sensor
anomalies to maintenance cases under temporal uncertainty, enabling
failure-mode ranking and retrieval of relevant historical maintenance logs.
Using real-world semiconductor sensor traces and actual maintenance records,
we show that modeling temporal compatibility and component-level constraints
improves retrieval quality over rule-based and text-only baselines.
```

---

## 5. 논문에서 조심해야 할 표현

### 피할 것
- "세계 최초"
- "완전한 root cause 자동화"
- "정확한 정답 원인 예측"

### 이유
- 실제 정비로그는 noisy하고 비정형
- root cause를 절대적 진실로 맞히는 문제보다, **failure mode와 관련 사례를 근거 있게 찾는 문제**로 두는 게 더 강함

### 논문 가능성 판정 기준 (7개 핵심 숫자)

| 항목 | 기준 |
|------|------|
| APC/pressure 계열 로그 수 | 최소 수백 건 |
| 관련 센서 수 | 10개 이상 |
| setpoint/actual 쌍 존재 여부 | 있어야 함 |
| recipe/step 정보 존재 여부 | 있어야 함 |
| 로그 component 명시 비율 | 높을수록 좋음 |
| ±3일 내 episode-로그 연결 비율 | 높을수록 좋음 |
| Gold link 수 | 50건 이상 → 파일럿 가능, 100~300건 → retrieval 논문 가능, 300건+ → 박사 주제 안정 |

---

## 6. 가장 빠른 실험 순서

### 1주차
- failure family 하나 선택
- 로그 100건 수동 읽기
- 센서 20개 내외 선정
- event 정의 5개: tracking failure, saturation, drift, oscillation, stuck

### 2주차
- episode 50~100개 추출
- 로그 후보 자동 매칭
- 사람이 gold/silver/weak 라벨링

### 3주차
- baseline 2개: 규칙 기반 매칭 + BM25 로그 검색

### 4~5주차
- sensor summary → dense retrieval
- top-k case retrieval 평가
- 첫 실험 결과 그림 완성

---

## 7. 관련 연구 키워드

| 영역 | 핵심 논문/시스템 |
|------|-----------------|
| 시계열-언어 정렬 | CLaSP, LaSTR, SensorLM |
| 산업 RAG | SOPRAG, GraphRAG, ManuRAG, KEO |
| 이상탐지+LLM | RAAD-LLM, AAD-LLM |
| 정비로그 분석 | Condition Insight, MindRAG |
| 산업 에이전트 | TAMO, S2S-FDD |
| 비지도 이상탐지 | Anomaly Transformer |
| 센서-고장 관계 | FailureSensorIQ |
