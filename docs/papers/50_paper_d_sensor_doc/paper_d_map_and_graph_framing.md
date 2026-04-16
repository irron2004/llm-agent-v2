# Paper D — 지도 만들기와 graph-based anomaly framing

> 작성일: 2026-04-14  
> 목적: Paper D의 핵심을 더 쉬운 말로 정리하고, 사용자가 제안한 graph 기반 이상 정의 아이디어가 어디에 놓여야 하는지 분명히 한다.

---

## 1. Paper D를 쉬운 말로 다시 정리

Paper D는 한마디로 말하면 다음 문제를 푸는 연구다.

> 장비 센서에 이상 신호가 보였을 때, 과거 정비 사례와 SOP/manual을 같이 찾아서 왜 이런 문제가 났는지, 무엇을 먼저 점검해야 하는지를 근거 있게 알려주는 시스템.

즉, 단순히 anomaly를 감지하는 것이 아니라,

1. 센서 데이터를 **의미 있는 이벤트(symptom/event)** 로 바꾸고
2. 그 이벤트를 **과거 maintenance case** 와 연결하고
3. 관련 **SOP/manual/FMEA** 를 검색하고
4. 그 근거로 **failure mode ranking + grounded diagnosis** 를 생성하는 구조다.

---

## 2. "지도 만들기"와 "길 찾기"의 차이

Paper D는 사실 두 가지 일을 한다.

### 2.1 지도 만들기
- 센서 데이터를 어떤 event node로 볼지 정의
- 정비 이력을 어떤 maintenance case node로 볼지 정의
- sensor event ↔ maintenance case ↔ document 사이의 relation 정의
- ontology 구성
  - `sensor → component → symptom → failure mode → action → document`

즉, **무엇을 node로 보고 무엇을 edge로 볼지 정하는 일**이다.

### 2.2 길 찾기
- 새로운 이상이 들어왔을 때
- 위에서 만든 지도 위에서
- 관련 case와 문서를 자동으로 retrieval하고
- 근거 기반 diagnosis를 생성하는 일

즉, **만들어진 지도를 이용해 현재 문제와 가장 관련 있는 evidence를 찾는 일**이다.

### 핵심 관계

사용자가 말한 것 = **지도를 더 잘 만드는 일**  
Paper D의 retrieval/diagnosis = **그 지도로 길을 찾는 일**

둘은 경쟁 관계가 아니라 **앞뒤 관계**다.

---

## 3. 사용자가 말한 내용의 위치

사용자가 제안한 핵심은 다음과 같이 정리할 수 있다.

1. 센서데이터의 지도를 잘 만들려면 먼저 **이상(event/symptom)** 을 잘 정의해야 한다.
2. 그 이상이 결국 node가 된다.
3. 단일 센서 이상보다 센서 간 관계를 함께 보면 더 좋은 node를 만들 수 있다.
4. 앞에서 만든 지도를 이용해서 이상을 다시 더 잘 찾는 구조로 갈 수 있다.

이 생각은 Paper D 안에서 매우 중요하다.

왜냐하면 Paper D도 raw sensor sample 자체를 문서와 연결하지 않고,
이미 **event/symptom 수준으로 올려서** maintenance case 및 document와 연결하도록 설계되어 있기 때문이다.

따라서 사용자의 아이디어는 Paper D와 다르지 않고,
오히려 **Eventizer / representation layer를 더 강하게 만드는 방향**으로 해석하는 것이 맞다.

---

## 4. graph 기반 이상 정의 아이디어에 대한 판단

사용자가 제안한 아이디어는 대략 다음과 같다.

- 과거 정상/기준 패턴과의 DTW distance
- 센서별 통계량
- attention score
- 센서 간 correlation
- 과거 빈도 / Bayesian relation

같은 값들을 node/edge에 표현하고,
현재 이상이 들어오면 graph 탐색으로 anomaly class를 정의하고,
그 뒤 문서를 검색하자는 구상이다.

### 4.1 좋은 점

이 아이디어는 **node를 더 의미 있게 만들 수 있다**는 점에서 좋다.

예를 들어 단순히 `APC 이상`보다,

- `APC tracking failure`
- `pressure oscillation 동반`
- `actuator inconsistency 존재`

같은 관계 기반 event node가 훨씬 failure mode와 잘 연결될 수 있다.

즉, graph는 **복합 이벤트를 더 잘 표현하는 도구**가 될 수 있다.

### 4.2 위험한 점

하지만 이를 바로

> “Paper D의 핵심은 graph-based anomaly detection이다”

라고 바꾸면 위험하다.

이유는 다음과 같다.

1. Paper D의 현재 핵심 novelty는 **temporal uncertainty-aware retrieval** 이다.
2. anomaly 자체를 graph 탐색으로 정의하려고 하면 scope이 급격히 커진다.
3. node/edge 설계 자유도가 너무 커져서, retrieval/diagnosis보다 graph construction 자체가 논문의 중심이 될 수 있다.
4. graph 탐색이 anomaly를 잘 정의하는지, 아니면 graph를 어떻게 설계했는지에 따라 결과가 바뀌는지 분리 평가가 어려워진다.

### 4.3 가장 안전한 해석

가장 안전한 framing은 다음과 같다.

> **이상 자체를 그래프로 새로 정의하는 것이 아니라, 탐지된 sensor event를 graph 위에 올려서 더 풍부한 맥락 검색과 grounded diagnosis를 가능하게 한다.**

즉,

- graph는 anomaly detector 그 자체라기보다
- **Eventizer를 강화하고 retrieval을 돕는 중간 표현층**으로 두는 것이 좋다.

---

## 5. 가장 안전한 Paper D framing

### 좋지 않은 framing
- Paper D = graph-based anomaly detection
- anomaly 자체를 graph traversal로 정의
- retrieval보다 graph anomaly modeling이 핵심 기여

이렇게 쓰면 thesis scope가 커지고,
현재 Paper D의 retrieval/grounding 중심 프레이밍이 흐려진다.

### 좋은 framing

Paper D는 여전히 다음이 중심이다.

1. 센서 event ↔ maintenance case ↔ document 연결
2. temporal uncertainty-aware alignment
3. grounded retrieval and diagnosis

여기서 graph-based anomaly 아이디어는 다음 위치가 적절하다.

> **Eventizer 강화 모듈**  
> 또는  
> **지도 품질을 높이기 위한 representation enhancement**

즉,

- 현재 핵심 = retrieval and grounding
- graph 아이디어 = 더 좋은 event node를 만들기 위한 방법론적 확장

---

## 6. 가장 좋은 통합 스토리

Paper D를 가장 자연스럽게 설명하면 다음과 같다.

1. 센서 이상을 event/symptom으로 정의한다.
2. 필요하면 이 event를 graph-based relational context로 보강한다.
3. 그렇게 만든 event node를 maintenance case와 시간 불확실성 하에서 연결한다.
4. 관련 SOP/manual/FMEA를 검색한다.
5. 검색된 evidence를 바탕으로 grounded diagnosis를 생성한다.

즉,

> **graph는 anomaly의 본질 정의라기보다, anomaly/event를 더 풍부하게 표현해서 retrieval과 grounded diagnosis를 더 잘 하게 만드는 층**으로 쓰는 것이 가장 강하다.

---

## 7. 쉬운 결론

### 질문 1
"내가 말한 것들은 지도를 더 잘 만들기 위한 내용이었던 거잖아?"

→ **맞다. 정확히 그렇다.**

### 질문 2
"이상을 잘 정의해야 node가 되니까, graph 기반으로 이상을 생각하는 건 어떠냐?"

→ **좋은 생각이다.** 다만 그것을 Paper D의 핵심 전체로 올리기보다,
**Eventizer를 더 잘 만드는 방법**으로 두는 것이 가장 안전하고 강하다.

---

## 8. 한 문장 요약

> Paper D의 핵심은 센서 event, maintenance case, 문서를 연결한 지도를 만들고 그 위에서 retrieval과 grounded diagnosis를 수행하는 것이다. graph-based anomaly 아이디어는 이 지도에서 **event node를 더 잘 만드는 방법**으로 넣을 때 가장 설득력이 높다.

---

## Related Documents

- `paper_d_algorithm_design.md`
- `paper_d_architecture.md`
- `paper_d_research_proposal.md`
- `paper_d_paper_strategy.md`
