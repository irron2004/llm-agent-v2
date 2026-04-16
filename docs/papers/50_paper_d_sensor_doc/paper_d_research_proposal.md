# Paper D — 박사과정 연구계획서

> 작성일: 2026-04-14
> 상태: 초안 완성, 지도교수 제출용

---

## 1. 연구제목

**불완전한 시간 정렬을 갖는 산업 센서데이터와 정비로그의 연계를 위한 근거기반 고장진단 및 정비사례 검색 방법**

A Grounded Diagnosis and Maintenance-Case Retrieval Framework for Linking Industrial Sensor Data and Maintenance Logs under Temporal Uncertainty

---

## 2. 연구배경 및 필요성

산업 현장의 유지보수 데이터는 세 부류로 나뉜다:
1. **시계열 형태의 센서데이터**
2. **비정형 정비로그** (작업지시, 정비이력, 정비기사 메모)
3. **구조화된 엔지니어링 지식** (SOP, 매뉴얼, FMEA, 자산 메타데이터)

이 정보들이 서로 강하게 연결되어 있음에도 실제 분석은 개별 시스템 안에서 따로 수행된다.

**핵심 gap**: 센서 이상 episode와 실제 정비로그를 시간 불확실성까지 고려해 직접 연결하고, 원인 후보와 관련 정비사례를 함께 제시하는 프레임워크가 충분히 정립되지 않음.

**시간 불확실성 문제**: 정비로그 시간은 실제 이상 시작 시점이 아니라 기록 시점, 승인 시점, 종료 시점일 수 있어 단순 최근접 매칭으로는 오류가 큼.

---

## 3. 연구목표

산업 장비에서 수집되는 다변량 센서데이터와 비정형 정비로그를 연결하여, 센서데이터에서 이상 징후 발생 시:

1. **failure mode 추론**: 해당 이상 episode의 가능성 높은 failure mode를 추론
2. **정비사례 검색**: 의미적으로 유사한 과거 정비로그 및 작업사례를 top-K 검색
3. **근거기반 설명**: SOP·FMEA·부품 계층 지식으로 근거가 연결된 설명과 점검 우선순위 제시

→ 단순한 anomaly detection이 아닌, **근거 기반 진단 및 사례 검색 통합 프레임워크**.

---

## 4. 연구문제 (Research Questions)

### 핵심 RQ

| ID | 연구문제 |
|----|----------|
| RQ1 | 불완전한 시간 정렬을 가진 센서데이터와 정비로그를 어떤 방식으로 연결해야 실제 정비 episode를 가장 잘 복원할 수 있는가? |
| RQ2 | 센서 window만을 입력으로 했을 때, failure mode와 관련 정비로그를 top-K 안에 얼마나 정확하게 검색할 수 있는가? |
| RQ3 | FMEA, SOP, 자산-부품 계층 같은 구조화된 지식을 결합하면 원인 추론과 설명의 근거성은 얼마나 향상되는가? |

### 보조 RQ

| ID | 연구문제 |
|----|----------|
| RQ4 | 근거 없는 조치 권고를 줄이기 위한 검증 루프를 넣었을 때, 설명의 신뢰성과 현장 활용성은 얼마나 개선되는가? |

---

## 5. 연구가설

| ID | 가설 |
|----|------|
| H1 | 센서데이터를 원시 시계열 그대로 사용하는 것보다, 이벤트 중심의 sensor window로 재구성한 뒤 정비로그와 연결할 때 더 안정적인 정렬이 가능하다. |
| H2 | 센서 window와 정비로그를 공통 임베딩 공간에서 정렬하되, 시간 호환성·부품 일치도·자산 계층 제약을 함께 반영하면 텍스트 검색 또는 센서 분류 단독 방식보다 더 높은 retrieval 성능을 얻을 수 있다. |
| H3 | SOP, FMEA, 부품 계층 지식을 결합한 neuro-symbolic grounding은 failure mode 추천의 설명 가능성과 근거 충실도를 높인다. |
| H4 | 검증 루프를 통해 근거 없는 문장을 억제하면, 전문가가 체감하는 신뢰성과 실제 활용 가능성이 향상된다. |

---

## 6. 연구범위

초기 범위 (과도하게 넓히지 않음):
- 장비군 **1종** 또는 라인 **1개**
- 주요 failure family **2~4개**
- 핵심 센서 **10~50개**
- 정비로그 **1~3년치**
- SOP, setup manual, FMEA, 알람 로그, 부품 교체 이력 포함

**초기 타겟 예시**:
> APC/pressure control 계열 이상에 대해, 센서 window로부터 failure mode와 관련 정비로그를 검색하고, SOP 근거를 덧붙여 설명하는 프레임워크

---

## 7. 연구내용 및 방법

### 7.1 데이터 구축 및 Ontology 설계

**Ontology 엔터티**: asset, component, sensor, alarm, work order, symptom, failure mode, maintenance action, outcome

**정비로그 정규화 필드**:
- component
- observed symptom
- diagnosed cause
- action taken
- replaced part
- resolved 여부
- confidence
- free text 원문

**Timestamp 분리**: 생성시각, 승인시각, 작업시각, 완료시각

**LLM 활용**: 정규화 보조도구로 사용, gold label은 전문가 검토로 확정.

### 7.2 Event-Centric 데이터셋 생성

정비 episode를 anchor로 하는 event-centric window 구성:
- 각 work order에 대해 이전 2시간, 12시간, 48시간 등의 pre-failure window 추출

**Link 라벨 3단계**:
| 등급 | 정의 |
|------|------|
| **Gold** | 원인, 부품, 시각이 로그에 비교적 명시된 경우 |
| **Silver** | 동일 component + 인접 시각은 맞지만 확정 근거가 약한 경우 |
| **Weak** | 동일 asset의 maintenance episode 수준에서만 연결 가능한 경우 |

**Split 기준**: 시간 순서 분리 + 같은 failure episode는 같은 split + unseen asset 별도 test

### 7.3 Baseline 모델 (4종)

| Baseline | 설명 |
|----------|------|
| 규칙 기반 eventizer + BM25 | drift, oscillation, saturation 등을 규칙으로 추출 → 키워드 로그 검색 |
| 센서 only 분류기 | XGBoost, LightGBM, 1D CNN, Transformer → failure mode 분류 |
| 로그 only 검색기 | BM25, dense, hybrid → 로그 유사도 비교 |
| Event summary 기반 retrieval | sensor window → 구조화된 텍스트 → 검색 |

### 7.4 제안모델: Temporal Uncertainty-Aware Sensor–Log Retrieval

**4개 모듈**:

1. **Sensor Encoder**: 다변량 window + 자산 metadata → sensor embedding + symptom logits
2. **Maintenance Log Encoder**: 정규화된 로그 + 원문 → log embedding + cause/action representation
3. **Temporal Compatibility Module**: 의미 유사도 + 시간 거리 + component 일치 + asset hierarchy + 알람 동시발생 → pair score
4. **Knowledge-Grounded Reasoner**: top-K 로그 + SOP section + FMEA → failure mode + 점검 우선순위

### 7.5 학습 전략 (3단계)

| 단계 | 내용 |
|------|------|
| 1단계 | Self-supervised / weakly supervised pretraining (pseudo-caption 기반 정렬) |
| 2단계 | Cross-modal contrastive learning (sensor window ↔ maintenance log) |
| 3단계 | Multi-task fine-tuning |

**Loss 구성**:
```
L_total = L_retrieval + L_cause + L_time + L_grounding
```
- L_retrieval: sensor-window ↔ maintenance-log contrastive loss
- L_cause: failure mode classification loss
- L_time: temporal compatibility calibration loss
- L_grounding: 근거 없는 결론 억제 regularization

### 7.6 구조화된 지식 결합

3가지 지식원:
1. **Component hierarchy**
2. **FMEA / failure ontology**
3. **SOP / maintenance action library**

혼합 구조: KG 기반 전역 추론 + SOP text retrieval 기반 국소 절차 검색

### 7.7 사례기반 추론과 검증 루프

- 최종 진단을 단일 classifier 출력으로 고정하지 않고, **유사 사례 검색(case retrieval)**을 핵심 축으로 사용
- 최종 설명에 **deterministic verification step** → 근거 없는 문장/조치 억제
- 연결되지 않으면 "추가 확인 필요"로 낮춰 출력

---

## 8. 평가방법 (4층위)

### 8.1 분류 성능
- Macro-F1, PR-AUC, Top-k accuracy

### 8.2 검색 성능
- Recall@K, MRR, nDCG

### 8.3 시간 정렬 성능
- Gold/Silver/Weak 별 hit rate
- ±1일, ±3일, ±7일 시간 허용범위별 성능

### 8.4 실무 유용성 (전문가 평가)
- top-3 추천 사례 유용성
- 잘못된 조치 추천 비율
- unsupported claim 비율
- 원인 조사 시간 단축 정도
- 전문가 신뢰도

---

## 9. 기대효과 및 학문적 기여

1. **데이터 기여**: 센서-로그-조치 간 연결을 다루는 event-centric benchmark
2. **방법 기여**: 시간 불확실성을 고려한 cross-modal retrieval
3. **진단 기여**: 구조화된 엔지니어링 지식 결합 grounded diagnosis framework
4. **평가 기여**: 근거성, 신뢰성, 진단 시간 단축 효과를 함께 평가하는 실무형 평가 프레임워크

---

## 10. 연차별 연구계획

### 1차 연도
- 연구범위 확정, 데이터 인벤토리 정리
- Ontology 설계, 정비로그 정규화 기준 수립
- 전문가 태깅 가이드 작성
- Baseline 1차 구현
- **목표**: 데이터셋/벤치마크 논문 1편

### 2차 연도
- Event-centric dataset 구축
- Sensor-window ↔ maintenance-log retrieval 모델 개발
- 시간 호환성 모듈 + hard negative mining
- Contrastive retrieval 실험
- **목표**: retrieval 중심 논문 1편

### 3차 연도
- FMEA, SOP, component hierarchy 결합 grounded diagnosis 모듈
- Case retrieval + KG reasoning + procedure retrieval 결합
- 전문가 기반 평가
- **목표**: grounded reasoning 논문 1편

### 4차 연도
- 전체 시스템 통합, ablation study
- 현장형 사용자 평가
- 박사학위논문 집필
- **목표**: 학위논문 완성 + 학회/저널 제출

---

## 11. 예상 위험요인 및 대응전략

| 위험 | 대응 |
|------|------|
| 정비로그 품질 불균일 | symptom–failure mode–action 3층 구조로, root cause 없어도 중간 수준 학습 |
| 정확한 고장 시각 부재 | gold/silver/weak link 체계 도입 |
| 라벨 부족 | pseudo-caption + weak supervision + active learning + expert review |
| LLM 설명 과잉 일반화 | retrieval evidence 연결 필수 + unsupported claim suppression |

---

## 12. 박사학위논문 구성안

| 장 | 제목 | 내용 |
|----|------|------|
| 1 | 서론 | 연구배경, 문제정의, 연구목표, 기여 |
| 2 | 관련연구 | anomaly detection, predictive maintenance, work-order mining, cross-modal retrieval, RAG/KG/CBR |
| 3 | 문제정의 및 데이터셋 | ontology, event-centric benchmark, temporal uncertainty labeling |
| 4 | Sensor-Log Retrieval 모델 | sensor encoder, log encoder, temporal compatibility, 학습전략 |
| 5 | Grounded Diagnosis | FMEA/SOP/KG 결합, case retrieval, 검증 루프 |
| 6 | 실험 및 평가 | 정량 평가, ablation, 전문가 평가, 오류분석 |
| 7 | 결론 | 요약, 한계, 향후 연구 |

---

## 13. 제안서용 요약문

본 연구는 산업 장비에서 수집되는 다변량 센서데이터와 비정형 정비로그를 연계하여, 이상 징후 발생 시 failure mode와 관련 정비사례를 근거 기반으로 검색·설명하는 방법을 제안한다. 기존 연구는 센서 기반 이상탐지, 정비로그 텍스트 분석, SOP 기반 질의응답 등을 주로 분리하여 다루어 왔으나, 실제 유지보수 의사결정은 이들 정보원의 통합적 해석을 필요로 한다. 이에 본 연구는 정비 episode를 중심으로 센서 window를 구성하고, 센서데이터와 정비로그를 공통 임베딩 공간에서 정렬하되 시간 불확실성, 부품 일치도, 자산 계층 제약을 함께 반영하는 retrieval 프레임워크를 제안한다. 더 나아가 FMEA, SOP, 자산 구조지식을 결합한 grounded reasoning 모듈을 통해 failure mode 후보, 관련 정비사례, 점검 우선순위를 함께 제시한다. 본 연구의 기대 기여는 event-centric benchmark 구축, temporal uncertainty-aware sensor–log retrieval 방법 제안, 구조화된 지식을 활용한 근거 기반 진단 프레임워크 개발, 그리고 실제 전문가 평가를 포함하는 실무형 성능평가 체계 수립에 있다.
