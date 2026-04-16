# Paper D — Paper Comparison Table

> 작성일: 2026-04-14  
> 목적: survey한 논문들을 비교 관점으로 한눈에 볼 수 있게 정리한 표 문서

---

## 1. 이 문서의 목적

이 문서는 Paper D와 관련된 논문들을 다음 기준으로 비교하기 위해 만든다.

- 데이터 modality
- retrieval target
- maintenance log 사용 여부
- SOP/manual grounding 여부
- temporal uncertainty 처리 여부
- graph 사용 위치
- Paper D와의 차이

즉, 이 문서는 "무슨 논문이 있었나"보다
**"각 논문이 무엇을 하고, Paper D와 무엇이 다른가"**를 빠르게 보는 용도다.

---

## 2. 핵심 비교표

| 논문 | 입력 데이터 | 핵심 문제 | Retrieval 대상 | Maintenance log | SOP/manual | Temporal uncertainty | Graph 사용 | Paper D와의 차이 |
|---|---|---|---|---|---|---|---|---|
| CLaSP | time series + text | contrastive alignment | text / time-series | 없음 | 없음 | 없음 | 없음 | retrieval backbone만 제공 |
| LaSTR | time series segment + language query | segment retrieval | segment | 없음 | 없음 | 없음 | 없음 | event-centric retrieval 참고 |
| TRACE | time series + text | multimodal retrieval | aligned context | 없음 | 없음 | 없음 | 없음 | hard negative / multimodal retrieval 참고 |
| SensorLM | sensor + text | sensor-language model | text-aligned sensor state | 없음 | 없음 | 없음 | 없음 | caption/pseudo-text 전략 참고 |
| FD-LLM | sensor signal + language | fault diagnosis with LLM | language-grounded diagnosis | 간접적 | 약함 | 약함 | 없음 | signal-to-language explanation 층의 대체 근거 |
| Technical language processing for maintenance work orders | maintenance logs | work order mining | 없음(분석 중심) | 있음 | 없음 | 없음 | 없음 | maintenance log가 핵심 데이터라는 근거 |
| Causal knowledge extraction from maintenance documents | maintenance text | causal triple extraction | causal graph entries | 있음 | 간접적 | 없음 | 문서/원인 graph | evidence graph 구축 근거 |
| Extracting failure time data from industrial maintenance records | maintenance records | failure time reconstruction | 없음 | 있음 | 없음 | 강함 | 없음 | temporal uncertainty 직접 지지 |
| SOPRAG | SOP/manual | structured SOP retrieval | SOP sections | 없음 | 있음 | 없음 | SOP graph | 문서 grounding 구조 핵심 근거 |
| Document GraphRAG | manufacturing documents | graph-based document QA | document chunks/graph nodes | 없음 | 있음 | 없음 | document graph | SOP/manual grounding 보강 |
| Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry | FDC sensors + contextual info | semiconductor diagnosis | 없음 | 간접적 | 없음 | 없음 | 없음 | sensor-only 한계의 도메인 근거 |
| KEO | O&M text + knowledge graph | KG-augmented retrieval for industrial QA | graph/text evidence | 간접적 | 있음 | 없음 | knowledge graph | graph-grounded evidence retrieval 참고 |
| RAGLog | logs + KB | retrieval-based anomaly reasoning | similar logs / KB | 로그 중심 | 없음 | 없음 | 없음 | anomaly를 retrieval로 보는 framing 참고 |
| RAINDROP | multivariate time series | graph-guided representation | 없음 | 없음 | 없음 | 없음 | sensor graph | graph-based state representation 배경 |
| FC-STGNN | multivariate time series | spatial-temporal sensor graph modeling | 없음 | 없음 | 없음 | 없음 | sensor graph | multi-sensor relation graph 배경 |
| THGNN / HTGNN 계열 | heterogeneous time-series system | temporal heterogeneous graph learning | 없음 | 없음 | 없음 | 없음 | heterogeneous graph | 개념은 유용하지만 final citation 전 재검증 필요 |
| The ROAD from Sensor Data to Process Instances | sensor streams | process instance reconstruction | event/process instance | 없음 | 없음 | 간접적 | process/event relation | sensor→event abstraction 근거 |
| Inferring Missing Event Log Data from IoT Sensor Data | IoT sensor streams | event log reconstruction | missing event log | 간접적 | 없음 | 간접적 | process/event relation | sensor stream ↔ event log alignment 강화 |

---

## 3. 이 비교표로 바로 볼 수 있는 것

### 3.1 기존 논문들이 잘하는 것
- time-series ↔ text retrieval
- maintenance text mining
- SOP/manual structured retrieval
- semiconductor sensor-only diagnosis
- graph-based multivariate representation

### 3.2 기존 논문들이 놓치는 것
- **sensor episode + maintenance log + document** 를 동시에 한 프레임에 올리는 것
- **temporal uncertainty** 를 retrieval 중심 문제로 다루는 것
- **retrieved evidence의 citation correctness / grounding quality** 평가
- **real semiconductor sensor + real maintenance log + real SOP/manual** 조합
- **검증된 graph-based state retrieval 직접 선행연구가 아직 제한적**

### 3.3 Paper D의 상대적 위치

Paper D는 위 논문들 사이에서 다음처럼 위치한다.

> **센서 상태를 event/state로 구조화하고, 그것을 maintenance case와 시간 불확실성 하에서 연결하며, 관련 SOP/manual을 함께 retrieval하여 grounded diagnosis를 수행하는 프레임워크**

즉,

- CLaSP/LaSTR/TRACE 쪽의 retrieval backbone,
- maintenance log mining,
- SOPRAG/Document GraphRAG 쪽의 document grounding,
- semiconductor diagnosis motivation,

을 한 프레임으로 통합하는 위치에 있다.

---

## 4. 추천 활용법

이 표는 아래 상황에서 특히 유용하다.

1. **Related Work 문단 쓸 때**
   - 어떤 논문을 어느 축에 넣을지 빠르게 결정
2. **Method novelty 설명할 때**
   - "우리는 무엇을 새로 하는가"를 기존 논문과 대비
3. **지도교수와 방향 논의할 때**
   - Paper D가 anomaly detection인지 retrieval인지 grounding인지 설명
4. **scope control 할 때**
   - 무엇을 넣고 무엇을 빼야 하는지 판단

---

## 5. 관련 문서

- `paper_d_surveyed_references.md`
- `paper_d_bibtex_priority.md`
- `2026-04-14_related_literature_survey.md`
