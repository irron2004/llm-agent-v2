# Paper D — Surveyed References

> 작성일: 2026-04-14  
> 목적: Paper D 관련해서 지금까지 조사한 논문들을 나중에 빠르게 다시 참고할 수 있도록 정리한 reference note

---

## 1. 이 문서의 목적

이 문서는 Paper D 관련 논문들을 다음 기준으로 빠르게 다시 보기 위한 참고 문서다.

- 어떤 논문이 있었는가
- 각 논문이 무엇을 해결하는가
- Paper D에서 왜 중요한가
- 어디에 인용하면 좋은가
- 지금 강하게 써도 되는지, 아니면 재검증이 필요한지

즉, 자세한 survey 원문을 대체하는 문서가 아니라,
**나중에 다시 꺼내 보기 쉬운 요약형 레퍼런스 문서**다.

---

## 2. Quick Lookup

| # | 논문 | 분야 | 핵심 기여 | Paper D에서의 역할 |
|---|------|------|-----------|---------------------|
| 1 | CLaSP | 시계열-언어 정렬 | time-series ↔ text contrastive alignment | sensor-window ↔ maintenance-log retrieval backbone |
| 2 | LaSTR | 시계열 retrieval | segment-level time series retrieval | event-centric window retrieval 근거 |
| 3 | TRACE | multimodal retrieval | time-series grounding + hard negative mining | retrieval 설계 정교화 참고 |
| 4 | SensorLM | sensor-language model | sensor-text alignment + captioning | 데이터 부족 시 pseudo-text / caption 전략 |
| 5 | S2S-FDD | explainable fault diagnosis | time series → semantic summary → document-supported diagnosis | eventizer + explanation layer 근거 |
| 6 | Technical language processing for maintenance work orders | maintenance NLP | work order 기반 corrective action / topic extraction | maintenance log mining 축 핵심 |
| 7 | Causal knowledge extraction from maintenance documents | maintenance KG | maintenance text에서 causal triple 추출 | evidence graph / cause-action relation 근거 |
| 8 | Extracting failure time data from industrial maintenance records using text mining | temporal alignment | work order 시각으로 failure time 복원 | temporal uncertainty 핵심 근거 |
| 9 | SOPRAG | SOP retrieval | SOP를 graph 구조로 retrieval | SOP/manual 구조화 근거 |
| 10 | Document GraphRAG | manufacturing RAG | manufacturing 문서 QA에서 graph retrieval | document grounding 참고 |
| 11 | Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry | semiconductor diagnosis | sensor-only 진단의 한계 + contextual info 필요 | 반도체 도메인 motivation |
| 12 | RAINDROP | sensor graph | graph-guided multivariate time series modeling | graph-based state representation background |
| 13 | FC-STGNN | sensor graph | spatial-temporal graph for multi-sensor data | multi-sensor relation modeling |
| 14 | THGNN | heterogeneous sensor graph | temporal + heterogeneous graph learning | sensor/case/document graph 확장 논리 |
| 15 | RAGLog | retrieval anomaly/log reasoning | retrieval-augmented anomaly handling | retrieval-style anomaly/state matching 근거 |
| 16 | RAIDR | retrieval incident diagnosis | sensor relation + historical incident + docs | graph-structured state retrieval에 가장 가까운 축 |
| 17 | The ROAD from Sensor Data to Process Instances | sensor→event alignment | sensor data에서 process instance 복원 | event/process abstraction 근거 |
| 18 | Inferring Missing Event Log Data from IoT Sensor Data | event log reconstruction | sensor로부터 missing event log 추정 | sensor stream → event alignment 근거 |

---

## 3. 주제별 정리

### 3.1 시계열-언어 정렬 / Cross-modal Retrieval

#### CLaSP
- **핵심**: 시계열과 자연어를 같은 임베딩 공간에 맞춘다.
- **Paper D 의미**: sensor-window ↔ maintenance-log retrieval을 연구 문제로 잡는 데 가장 직접적인 출발점이다.
- **언제 참고?** retrieval backbone 설명할 때.

#### LaSTR
- **핵심**: 전체 시계열이 아니라 segment-level retrieval을 정식화한다.
- **Paper D 의미**: 긴 센서 스트림 전체가 아니라 pre-failure window를 다루는 설정과 잘 맞는다.
- **언제 참고?** event-centric window retrieval 설명할 때.

#### TRACE
- **핵심**: time series를 context-aware multimodal retrieval 대상으로 다루고 hard negative를 적극 사용한다.
- **Paper D 의미**: retrieval 학습과 hard negative 설계를 정교화할 때 도움이 된다.
- **언제 참고?** retriever 학습 파트와 hard negative 설계.

#### SensorLM
- **핵심**: sensor-text alignment, captioning, pseudo text generation.
- **Paper D 의미**: paired sensor-text가 부족할 때 event summary/caption을 만드는 전략의 참고 문헌이다.
- **언제 참고?** 데이터 부족 대응 전략.

#### S2S-FDD
- **핵심**: 시계열을 semantic summary로 바꾸고 문서 근거를 붙여 fault diagnosis를 시도한다.
- **Paper D 의미**: eventizer + explanation layer + document-supported diagnosis 구조가 Paper D와 매우 유사하다.
- **언제 참고?** eventizer 서술, explanation layer, related work.

---

### 3.2 Maintenance Log / Work Order Mining

#### Technical language processing for maintenance work orders
- **핵심**: maintenance work order 자체가 중요한 predictive/prescriptive signal이라는 점을 보여준다.
- **Paper D 의미**: maintenance log를 부가 정보가 아니라 핵심 retrieval 대상이라고 정당화할 수 있다.
- **언제 참고?** maintenance log mining 섹션.

#### Causal knowledge extraction from maintenance documents
- **핵심**: maintenance text에서 cause-effect triple을 뽑아 causal graph를 만든다.
- **Paper D 의미**: 문서에서 같이 나오는 센서/부품/원인 표현을 weak relation 또는 evidence graph로 올리는 발상과 잘 맞는다.
- **언제 참고?** evidence graph, document-derived relation, maintenance KG.

#### Extracting failure time data from industrial maintenance records using text mining
- **핵심**: maintenance record만으로 failure time을 더 정확히 복원하는 문제를 다룬다.
- **Paper D 의미**: temporal uncertainty가 왜 핵심 난제인지 직접적으로 뒷받침한다.
- **언제 참고?** temporal alignment / problem definition.

#### Toward Semi-Autonomous Information Extraction for Unstructured Maintenance Data in RCA
- **핵심**: maintenance text를 구조화된 RCA 입력으로 바꾸는 단계적 파이프라인.
- **Paper D 의미**: log normalization과 ontology 구성의 필요성을 보여준다.
- **언제 참고?** 데이터 준비 / preprocessing.

---

### 3.3 SOP / Manual / Document Grounding

#### SOPRAG
- **핵심**: SOP는 일반 문서처럼 flat chunking 하면 안 되고, 절차 구조를 반영해야 한다.
- **Paper D 의미**: SOP/manual을 structured knowledge unit으로 저장해야 한다는 강한 근거다.
- **언제 참고?** document retriever / procedural evidence.

#### Document GraphRAG
- **핵심**: manufacturing 문서 QA에서 graph retrieval이 naive RAG보다 유리할 수 있음을 보인다.
- **Paper D 의미**: 문서 grounding에서 graph-based retrieval을 왜 쓰는지 설명할 수 있다.
- **언제 참고?** document grounding / GraphRAG 관련 work.

---

### 3.4 Semiconductor / Industrial Diagnosis Motivation

#### Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry
- **핵심**: FDC signal만으로는 오진 가능성이 있고, contextual/statistical information이 함께 필요하다.
- **Paper D 의미**: 반도체 진단에서 sensor-only 접근이 부족하다는 domain-specific motivation을 제공한다.
- **언제 참고?** introduction, domain motivation.

#### Machine Learning-Based Integrated Database System for PECVD Fault Diagnosis
- **핵심**: 반도체 장비와 센서 데이터를 fault diagnosis DB와 연결하는 실무형 구조.
- **Paper D 의미**: 반도체 장비 데이터와 diagnosis DB 결합의 실무성을 뒷받침한다.
- **언제 참고?** application context / semiconductor systems.

---

### 3.5 Graph-based State Representation / Retrieval-style Matching

#### RAINDROP
- **핵심**: irregularly sampled multivariate time series를 graph-guided 구조로 표현한다.
- **Paper D 의미**: 여러 센서를 graph-like state로 보는 첫 번째 방법론적 근거가 된다.
- **언제 참고?** graph-based state representation.

#### FC-STGNN
- **핵심**: multi-sensor 시계열을 spatial-temporal graph로 모델링한다.
- **Paper D 의미**: 센서 간 관계를 graph로 보려는 현재 아이디어와 매우 잘 맞는다.
- **언제 참고?** sensor relation graph 배경.

#### THGNN
- **핵심**: temporal + heterogeneous graph neural network.
- **Paper D 의미**: sensor / maintenance case / document를 heterogeneous graph로 확장하는 논리의 bridge 역할을 한다.
- **언제 참고?** heterogeneous graph 확장 아이디어.

#### RAGLog
- **핵심**: anomaly/log reasoning을 retrieval-augmented 방식으로 다룬다.
- **Paper D 의미**: anomaly를 pure classifier로 다루지 않고 retrieval로 접근하는 사고방식을 지지한다.
- **언제 참고?** retrieval-style anomaly/state matching framing.

#### RAIDR
- **핵심**: historical incident retrieval + document/report grounding.
- **Paper D 의미**: graph-structured state retrieval + linked documents grounding 쪽에서 가장 가까운 구조다.
- **언제 참고?** related work에서 가장 직접 비교 가능한 후보 중 하나.

---

### 3.6 Process Mining / Event Alignment from Sensor Streams

#### The ROAD from Sensor Data to Process Instances
- **핵심**: sensor data로부터 process instance를 복원한다.
- **Paper D 의미**: raw ts를 바로 anomaly detection 하지 않고 event/state로 추상화한다는 발상과 잘 맞는다.
- **언제 참고?** event abstraction / process alignment.

#### Inferring Missing Event Log Data from IoT Sensor Data
- **핵심**: sensor data로부터 event log를 추정/보완한다.
- **Paper D 의미**: sensor stream ↔ event log 연결과 temporal uncertainty 논의를 강화한다.
- **언제 참고?** event-centric alignment / process mining 연결.

#### Process mining on sensor data: a review
- **핵심**: sensor data 기반 process mining을 정리한 survey.
- **Paper D 의미**: sensor→event abstraction이 독립된 연구 축이라는 점을 보여준다.
- **언제 참고?** gap analysis / related work 서론.

#### SensorStream
- **핵심**: event log에 sensor stream 정보를 enrich하는 형식적 프레임워크.
- **Paper D 의미**: sensor와 event log를 함께 표현하는 데이터 모델 참고용.
- **언제 참고?** dataset/representation discussion.

---

## 4. 우선순위

### Tier 1 — 바로 다시 봐야 하는 논문
1. CLaSP
2. LaSTR
3. S2S-FDD
4. Technical language processing for maintenance work orders
5. Causal knowledge extraction from maintenance documents
6. Extracting failure time data from industrial maintenance records using text mining
7. SOPRAG
8. Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry

### Tier 2 — 지금 대화 이후 새로 중요해진 논문
1. RAIDR
2. RAGLog
3. RAINDROP
4. FC-STGNN
5. THGNN
6. Inferring Missing Event Log Data from IoT Sensor Data
7. The ROAD from Sensor Data to Process Instances

### Tier 3 — 방향은 좋지만 사용 전 다시 확인할 논문
1. HTGNN 계열
2. LogSentry
3. EnrichLog
4. Graph-augmented fault diagnosis (2026)
5. Condition Insight Agent
6. MaintAGT
7. DiagnosticIQ

---

## 5. 재검증 필요 후보

아래는 아이디어상 매우 유망하지만, 최종 논문에 강하게 넣기 전에는 메타데이터/venue/실험 설정을 다시 확인해야 한다.

- Condition Insight Agent
- DML-LLM Hybrid fault diagnosis
- MaintAGT
- DiagnosticIQ
- HTGNN 일부 버전
- LogSentry
- EnrichLog
- Graph-augmented fault diagnosis (2026)

원칙:

> 이 논문들은 현재는 “방향 참고용”으로 두고,
> final related work에 들어가기 전에는 다시 검증한다.

---

## 6. Paper D에서의 사용 위치

### Introduction / Problem Motivation
- Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry
- Technical language processing for maintenance work orders
- Extracting failure time data from industrial maintenance records using text mining

### Method — Retrieval Backbone
- CLaSP
- LaSTR
- TRACE
- SensorLM

### Method — Eventizer / Explanation Layer
- S2S-FDD
- RAAD-LLM

### Method — Document Grounding
- SOPRAG
- Document GraphRAG

### Method — Graph-based State Representation (if used)
- RAINDROP
- FC-STGNN
- THGNN
- RAGLog
- RAIDR

### Method — Event / Process Alignment
- The ROAD from Sensor Data to Process Instances
- Inferring Missing Event Log Data from IoT Sensor Data
- Process mining on sensor data: a review

### Data / Preprocessing / Ontology
- Causal knowledge extraction from maintenance documents
- Toward Semi-Autonomous Information Extraction for Unstructured Maintenance Data in RCA

---

## 7. 이 문서 사용법

이 문서는 아래 순서로 쓰는 것이 좋다.

1. **논문 초안 쓸 때**: Quick Lookup에서 어떤 논문을 어디에 쓸지 빠르게 확인
2. **Related Work 정리할 때**: 주제별 정리를 보고 축별로 문단 구성
3. **Method 쓰기 전**: 어떤 논문이 retrieval backbone이고 어떤 논문이 document grounding 근거인지 확인
4. **최종 인용 전**: Tier 3 및 재검증 필요 후보는 다시 확인

---

## Related Documents

- `2026-04-14_literature_survey.md`
- `2026-04-14_related_literature_survey.md`
- `2026-04-14_raad_llm_review.md`
- `../paper_d_algorithm_design.md`
- `../paper_d_map_and_graph_framing.md`
