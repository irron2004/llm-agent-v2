# Paper D — 교수님 보고용 1페이지 요약

> 작성일: 2026-04-14  
> 목적: Paper D의 연구 문제, 핵심 기여, 방법, 현재 상태를 지도교수에게 빠르게 설명하기 위한 1페이지 요약

---

## 1. 연구 주제

**반도체 장비의 센서 데이터와 정비로그, SOP/manual을 연결하여 근거 기반 진단을 수행하는 retrieval 중심 프레임워크**

한 줄로 말하면,

> 장비에서 이상 신호가 들어왔을 때, 과거 정비사례와 관련 문서를 자동으로 찾아서 원인 후보와 점검 순서를 근거와 함께 제시하는 방법을 연구하는 것이다.

---

## 2. 왜 이 연구가 필요한가

현장에서는 센서 이상이 보이면 보통 사람이 다음을 수동으로 수행한다.

1. 현재 센서 패턴 해석
2. 과거 유사 정비 사례 검색
3. SOP / setup manual / troubleshooting guide 확인
4. 원인 후보와 점검 순서 결정

문제는 이 정보들이 분절되어 있다는 점이다.

- 센서 데이터는 현재 상태를 보여주지만 원인과 조치를 직접 말해주지 못함
- 정비로그는 원인/조치 정보가 있지만 현재 상황과 정확히 연결하기 어려움
- SOP/manual은 공식 절차를 제공하지만 언제 꺼내야 하는지가 자동화되어 있지 않음

따라서 **sensor episode ↔ maintenance case ↔ procedural evidence**를 하나의 프레임에서 연결하는 방법이 필요하다.

---

## 3. 핵심 연구 질문

### RQ1
센서 시계열을 어떻게 **정비 가능한 event/symptom** 으로 바꿀 수 있는가?

### RQ2
불완전한 시간 정렬 아래에서 현재 sensor episode와 과거 maintenance case를 어떻게 연결할 수 있는가?

### RQ3
retrieved case와 SOP/manual evidence를 이용해 어떻게 **grounded diagnosis** 를 만들 수 있는가?

---

## 4. 제안 방법 요약

```text
Sensor Stream
  → Eventizer
  → Temporal Uncertainty Alignment
  → Maintenance Case Retriever
  → Document Retriever
  → Knowledge-Grounded Reasoner
  → Evidence-backed Diagnosis
```

### Eventizer
- tracking failure
- saturation
- oscillation
- feedback mismatch
같은 **정비 가능한 symptom/event** 추출

### Temporal Uncertainty Alignment
- maintenance log 시각이 실제 고장 발생 시각과 다를 수 있으므로 soft alignment 사용

### Maintenance Case Retriever
- 현재 sensor episode와 유사한 과거 정비사례 top-k 검색

### Document Retriever
- SOP / setup manual / troubleshooting guide / FMEA 검색

### Grounded Diagnosis
- 가능한 원인 1~3개
- 근거 사례와 문서 section
- 점검 순서
- confidence

---

## 5. 핵심 contribution 후보

### Contribution 1 — 데이터/문제 정의
센서 episode와 maintenance log를 **temporal uncertainty** 하에서 연결하는 문제 정의 및 dataset protocol

### Contribution 2 — 방법론
sensor event ↔ maintenance case retrieval + document grounding을 결합한 framework

### Contribution 3 — 실무성
실제 반도체 센서 데이터 + 실제 정비로그 + 실제 SOP/manual을 함께 사용하는 grounded diagnosis 구조

---

## 6. scope 원칙

### 하지 않을 것
- 일반적인 multivariate anomaly detection 전체를 새로 푸는 것
- LLM 하나로 end-to-end diagnosis를 다 해결하려는 것
- retrieval 없이 classifier만 만드는 것

### 할 것
- event/symptom 정의
- maintenance case retrieval
- temporal uncertainty-aware alignment
- document grounding
- evidence-based diagnosis

### graph 아이디어의 위치
graph-based anomaly/state 아이디어는 핵심 전체라기보다,

> **Eventizer / state representation을 강화하는 확장 방향**

으로 두는 것이 가장 안전하다.

---

## 7. 현재 상태와 다음 단계

### 현재까지 완료
- 아이디어 브레인스토밍
- 알고리즘 설계 문서 작성
- graph framing 정리
- literature survey / reference note 정리
- BibTeX 우선 수집 목록 및 논문 비교표 정리

### 다음 단계
1. 8대 장비에서 sensor family 후보 선정
2. 관련 maintenance logs / SOP / gcb/myservice 키워드 수집
3. sensor episode 50~100개 구축
4. gold / silver / weak link 부착
5. baseline retrieval 실험

---

## 8. 한 문장 결론

> Paper D는 “반도체 장비 센서 이상을 정비 사례와 문서 근거에 연결하는 retrieval 중심 grounded diagnosis 연구”로 정리하는 것이 가장 적절하다.
