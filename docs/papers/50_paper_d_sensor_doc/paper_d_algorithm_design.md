# Paper D — 알고리즘 설계

> 작성일: 2026-04-14
> 목적: 조사한 문헌과 기존 Paper D 설계를 바탕으로, 논문용 방법(Method) 중심의 알고리즘 구상을 정리한다.

---

## 1. 알고리즘 목표

Paper D의 목표는 단순 anomaly detection이 아니다.

핵심 목표는 다음과 같다.

1. 반도체 장비의 **sensor episode**를 구조화된 symptom/event로 변환한다.
2. 해당 episode를 **temporal uncertainty** 하에서 과거 **maintenance case**와 연결한다.
3. 관련 **SOP / setup manual / FMEA / troubleshooting guide**를 함께 검색한다.
4. 검색된 evidence를 바탕으로 **failure mode ranking**과 **grounded diagnosis**를 생성한다.

즉, 알고리즘의 중심 문제는 다음과 같이 정의된다.

> **불완전한 시간 정렬 아래에서 반도체 센서 episode를 정비사례와 절차 문서에 연결하는 grounded retrieval and diagnosis framework**

---

## 2. 핵심 아이디어

조사한 문헌을 종합하면, strongest thesis framing은 다음 세 축을 결합하는 것이다.

- **시계열-텍스트 정렬**: CLaSP, LaSTR, TRACE, SensorLM
- **정비로그 구조화**: maintenance work order mining, causal extraction, failure-time extraction
- **문서 grounding**: SOPRAG, Document GraphRAG, KG-RAG FMEA, KEO

따라서 알고리즘은 “센서와 로그를 그냥 같이 넣는 모델”이 아니라,
다음과 같은 **모듈형 구조**가 되어야 한다.

1. **Eventizer**: raw sensor → maintenance-relevant symptom
2. **Temporal Alignment**: episode ↔ maintenance log 시간 불확실성 모델링
3. **Case Retriever**: 현재 episode와 유사한 과거 정비사례 검색
4. **Document Retriever**: SOP/manual/FMEA evidence 검색
5. **Knowledge-Grounded Reasoner**: failure mode ranking + 점검 순서 생성
6. **Evidence-backed Diagnosis Generator**: 근거 기반 최종 설명 생성

---

## 3. 전체 파이프라인

```text
Multivariate Sensor Stream
    → Eventizer
    → Temporal Uncertainty Alignment
    → Maintenance Case Retriever
    → Document Retriever
    → Knowledge-Grounded Reasoner
    → Evidence-backed Diagnosis
```

### 입력
- multivariate sensor window
- equipment/chamber metadata
- recipe / step / mode
- calibration / PM history
- alarms / interlock
- maintenance logs
- SOP / manual / FMEA / troubleshooting documents

### 출력
- failure mode top-k
- 관련 maintenance case top-k
- 관련 SOP/manual sections
- recommended check sequence
- safety notes
- confidence + evidence references

---

## 4. 모듈별 설계

### 4.1 Eventizer

#### 역할
원시 센서 시계열을 정비 가능한 symptom/event 표현으로 변환한다.

#### 입력 예시
- `apc_position_actual`
- `apc_position_setpoint`
- `pressure`
- `actuator current`
- `valve state`
- `recipe_step`
- `mode`
- `recent calibration age`

#### 파생 feature
- tracking error
- absolute error duration
- saturation ratio
- stuck score
- response lag
- oscillation score
- pressure coupling
- actuator inconsistency

#### 출력 예시
```json
{
  "component": "APC",
  "symptoms": [
    {"name": "tracking_failure", "score": 0.93},
    {"name": "high_saturation", "score": 0.81},
    {"name": "pressure_oscillation", "score": 0.72}
  ],
  "features": {
    "tracking_error": 31.2,
    "error_duration_sec": 42,
    "stuck_score": 0.88
  }
}
```

#### 구현 전략
- 초기: 규칙 기반 eventizer
- 중기: XGBoost / LightGBM / 1D CNN
- 후기: Transformer encoder + multitask classifier

#### 설계 이유
문헌과 기존 Paper D 문서가 공통으로 강조하듯,
문서는 anomaly가 아니라 **symptom / failure mode / action** 수준에서 연결된다.

---

### 4.2 Temporal Uncertainty Alignment

#### 역할
센서 episode와 maintenance log 사이의 시간 불확실성을 명시적으로 모델링한다.

#### 문제 정의
maintenance log timestamp는 실제 failure onset이 아닐 수 있다.

예:
- 기록 시각
- 작업 시작 시각
- 작업 종료 시각
- 승인 시각

따라서 exact timestamp matching 대신 **soft alignment**가 필요하다.

#### 제안 score
\[
AlignmentScore =
\alpha \cdot semantic\_similarity +
\beta \cdot temporal\_compatibility +
\gamma \cdot component\_match +
\delta \cdot hierarchy\_match +
\epsilon \cdot alarm\_consistency
\]

#### temporal compatibility 요소
- ±1일 / ±3일 / ±7일 decay
- recipe step overlap
- chamber/equipment 동일성
- action-before/after consistency
- alarm co-occurrence

#### 설계 이유
이 모듈이 Paper D에서 가장 강한 novelty 포인트다.
기존 연구는 retrieval, grounding, diagnosis를 다루더라도
이 문제를 전처리 수준으로만 다루는 경우가 많다.

---

### 4.3 Maintenance Case Retriever

#### 역할
현재 sensor episode와 가장 관련된 과거 maintenance case를 top-k로 검색한다.

#### 구조
- `E_s(sensor_window)`
- `E_m(maintenance_log)`

#### positive / negative 구성
- positive: 실제 연결된 sensor window ↔ maintenance log
- hard negative 1: 같은 component, 다른 symptom
- hard negative 2: 같은 symptom, 다른 장비/recipe
- hard negative 3: 같은 action, 다른 failure mode

#### 출력
- related maintenance logs top-k
- candidate failure modes
- candidate actions
- retrieval confidence

#### loss
\[
L_{total} = L_{retrieval} + \lambda_1 L_{fault\_type} + \lambda_2 L_{time}
\]

- `L_retrieval`: contrastive / InfoNCE
- `L_fault_type`: failure mode classification
- `L_time`: temporal compatibility calibration

#### 문헌 기반 근거
- CLaSP
- LaSTR
- TRACE
- SensorLM

---

### 4.4 Document Retriever

#### 역할
SOP / setup manual / FMEA / troubleshooting guide에서 공식 procedural evidence를 검색한다.

#### 왜 maintenance case retrieval과 분리해야 하는가
- maintenance case는 **과거 실제 행동 근거**
- SOP/manual은 **공식 절차 근거**

즉, 둘은 성격이 다르므로 분리된 retrieval이 필요하다.

#### 추천 retrieval score
\[
FinalDocScore =
0.35 \cdot dense +
0.25 \cdot BM25 +
0.20 \cdot metadata +
0.20 \cdot graph
\]

#### 문서 저장 단위
문서는 flat chunking이 아니라 아래 구조를 유지해야 한다.

- component
- symptom
- failure mode
- action
- safety
- applicable recipe step
- related parts

#### 문헌 기반 근거
- SOPRAG
- Document GraphRAG
- KG-RAG FMEA
- KEO

---

### 4.5 Knowledge-Grounded Reasoner

#### 역할
retrieved maintenance cases + SOP/manual + ontology/FMEA를 결합하여 failure mode를 ranking한다.

#### 입력
- eventizer output
- maintenance case top-k
- SOP/manual sections top-k
- ontology
- FMEA
- component hierarchy

#### 출력
- failure mode ranking
- recommended check sequence
- evidence map

#### 설계 원칙
LLM은 “지식 생성기”가 아니라 **근거 정리기**로 사용한다.

즉:
- 근거 없는 조치를 만들지 않는다.
- 검색된 문서/사례를 재구성해 설명한다.
- evidence gap이 있으면 “추가 확인 필요”로 출력한다.

---

### 4.6 Evidence-backed Diagnosis Generator

#### 역할
최종 사용자에게 readable한 grounded diagnosis를 생성한다.

#### 출력 형식
1. 가능한 failure mode 1~3개
2. 관련 maintenance case
3. 관련 SOP/manual section
4. 점검 순서
5. 추가 확인 센서
6. safety notes
7. confidence

#### 필수 제약
- SOP에 없는 shutdown/reset 제안 금지
- retrieval되지 않은 원인 단정 금지
- evidence가 약하면 “추가 점검 필요”로 downgrade

#### 핵심 원칙
```text
If no evidence → do not conclude
If weak evidence → request additional checks
If safety-critical → require explicit SOP support
```

---

## 5. 학습 전략

### Stage A — Rule-based MVP
- 규칙 기반 Eventizer
- BM25 / metadata 기반 case retrieval
- SOP retrieval
- evidence-based template answer

**목표**: 데이터 정합성과 문제 정의 검증

### Stage B — Cross-modal Retrieval 학습
- sensor encoder
- maintenance log encoder
- contrastive retrieval
- temporal compatibility scoring

**목표**: Paper D의 중심 retrieval 모델 구축

### Stage C — Grounded Diagnosis 확장
- retrieved cases + SOP/manual + FMEA 결합
- failure mode ranking
- evidence verification loop

**목표**: grounded diagnosis 논문 단계

### Stage D — Answer Style Fine-tuning (선택)
- evidence-attached response style 학습
- provenance/citation 표현 방식 정제

**주의**: retrieval을 빼고 LLM만 fine-tuning 하는 방향은 피한다.

---

## 6. 단계별 구현 계획

### 1단계: 가장 빠른 파일럿
1. APC/pressure control failure family 하나 선택
2. sensor episode 50~100개 추출
3. maintenance logs에 gold/silver/weak link 부착
4. baseline 2개 비교
   - 규칙 기반 매칭
   - BM25 / metadata search

### 2단계: retrieval 모델 실험
1. sensor-window encoder 구현
2. maintenance-log encoder 구현
3. contrastive loss 기반 학습
4. top-k retrieval 평가

### 3단계: grounded diagnosis
1. SOP/manual structure-aware indexing
2. ontology/FMEA 결합
3. evidence-backed reasoner 구현
4. grounded answer 평가

---

## 7. 평가 계획

### Retrieval 평가
- Recall@K
- MRR
- nDCG

### Temporal Alignment 평가
- ±1일 / ±3일 / ±7일 hit rate
- gold / silver / weak 별 성능

### Diagnosis 평가
- Macro-F1
- Top-k accuracy
- failure mode ranking accuracy

### Grounding 평가
- evidence correctness
- citation faithfulness
- unsupported claim ratio
- expert usefulness

---

## 8. 피해야 할 안 좋은 설계

### 8.1 Raw sensor + raw text naive concat
- temporal uncertainty를 무시함
- semantic gap이 큼
- thesis novelty가 약해짐

### 8.2 LLM 하나로 diagnosis까지 end-to-end
- grounding 약화
- hallucination 증가
- industrial reviewer 설득력 저하

### 8.3 Retrieval 없는 classifier-only 구조
- maintenance case retrieval 기여 상실
- 정비로그 활용 장점이 약해짐

### 8.4 Temporal uncertainty를 전처리 문제로만 취급
- thesis 핵심 novelty를 놓침
- 실제 산업 데이터의 난제를 회피하게 됨

---

## 9. 논문용 알고리즘 한 문장

> 본 연구는 반도체 장비의 multivariate sensor window와 unstructured maintenance logs를 공통 의미 공간에서 정렬하되, maintenance timestamp의 불확실성을 temporal compatibility로 명시적으로 모델링하고, retrieved maintenance cases와 SOP/FMEA evidence를 결합하여 grounded diagnosis를 생성하는 프레임워크를 제안한다.

---

## 10. 추천 MVP

가장 현실적인 시작점은 다음 조합이다.

```text
Rule-based Eventizer
+ Temporal Compatibility Scorer
+ Maintenance Case Retriever
+ SOP Retriever
+ Evidence-backed Answer Template
```

이후 다음 단계로 확장한다.

```text
Sensor Encoder
+ Log Encoder
+ Contrastive Retrieval
+ Knowledge-Grounded Reasoner
```

---

## Related Documents

- `paper_d_paper_strategy.md`
- `paper_d_architecture.md`
- `paper_d_research_proposal.md`
- `paper_d_training_roadmap.md`
- `evidence/2026-04-14_related_literature_survey.md`
