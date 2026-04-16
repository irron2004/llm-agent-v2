# Paper D — BibTeX Priority List

> 작성일: 2026-04-14  
> 목적: Paper D 집필 전에 어떤 논문부터 BibTeX와 정확한 메타데이터를 확보해야 하는지 우선순위로 정리한 참고 문서

---

## 1. 이 문서의 목적

이 문서는 survey한 논문들 중에서,

- 어떤 논문부터 BibTeX를 확보해야 하는지
- 어떤 논문은 final citation으로 바로 써도 되는지
- 어떤 논문은 아직 재검증이 필요한지

를 빠르게 확인하기 위한 체크리스트다.

즉, 실제 논문 작성 전에 "citation 작업 순서"를 정리해 둔 문서다.

---

## 2. 최우선 확보 목록

이 논문들은 Paper D의 중심 논리를 받쳐주는 핵심 문헌들이다. 
가장 먼저 BibTeX와 정확한 메타데이터를 확보해야 한다.

| 우선순위 | 논문 | 이유 | Paper D에서 쓰는 위치 |
|---|---|---|---|
| 1 | CLaSP | sensor-window ↔ text/log retrieval backbone | Method / Related Work |
| 2 | LaSTR | event-centric window retrieval 정당화 | Method / Related Work |
| 3 | S2S-FDD | eventizer + document-supported diagnosis 구조 참고 | Eventizer / Related Work |
| 4 | Technical language processing for maintenance work orders | maintenance log mining 핵심 | Introduction / Related Work |
| 5 | Causal knowledge extraction from maintenance documents | maintenance text → causal graph 근거 | Data / Evidence Graph |
| 6 | Extracting failure time data from industrial maintenance records using text mining | temporal uncertainty 핵심 근거 | Problem Definition |
| 7 | SOPRAG | SOP/manual structured retrieval 근거 | Document Retriever |
| 8 | Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry | semiconductor motivation | Introduction |
| 9 | RAIDR | graph-structured state retrieval + linked docs에 가장 가까운 구조 | Related Work / Discussion |
| 10 | RAGLog | retrieval-style anomaly/state matching framing | Related Work |

---

## 3. 2순위 확보 목록

이 논문들은 핵심 spine은 아니지만,
방법론을 정교하게 설명하거나 확장 방향을 설명할 때 유용하다.

| 우선순위 | 논문 | 이유 | Paper D에서 쓰는 위치 |
|---|---|---|---|
| 11 | TRACE | hard negative / multimodal retrieval 정교화 | Retriever 학습 |
| 12 | SensorLM | pseudo-caption / sensor-text 데이터 부족 대응 | Data 부족 대응 |
| 13 | Document GraphRAG | graph-based document grounding 근거 | Document Grounding |
| 14 | RAINDROP | graph-guided multivariate representation | Graph-based state representation |
| 15 | FC-STGNN | multi-sensor relation graph background | Graph-based state representation |
| 16 | THGNN | heterogeneous graph 확장 논리 | Heterogeneous graph discussion |
| 17 | The ROAD from Sensor Data to Process Instances | sensor → process/event abstraction | Process/Event alignment |
| 18 | Inferring Missing Event Log Data from IoT Sensor Data | sensor stream ↔ event log 연결 | Event-centric alignment |
| 19 | Process mining on sensor data: a review | process mining literature bridge | Related Work 서론 |
| 20 | SensorStream | sensor + event log representation 형식 | Dataset / Representation |

---

## 4. 재검증 후 확보 목록

아래 논문들은 방향상 중요하지만,
최종 논문에 직접 강하게 쓰기 전에 metadata/venue/실험 설정을 다시 확인해야 한다.

| 논문 | 현재 상태 | 메모 |
|---|---|---|
| Condition Insight Agent | 재검증 필요 | 아이디어는 좋지만 서지 정보 추가 확인 필요 |
| DML-LLM Hybrid fault diagnosis | 재검증 필요 | semiconductor diagnosis 연결 가능성 있음 |
| MaintAGT | 재검증 필요 | maintenance multimodal LLM 계열 |
| DiagnosticIQ | 재검증 필요 | benchmark/evaluation 축 가능성 |
| HTGNN 일부 버전 | 재검증 필요 | 버전/venue에 따라 쓰임이 달라질 수 있음 |
| LogSentry | 재검증 필요 | retrieval anomaly 처리 쪽 참고용 |
| EnrichLog | 재검증 필요 | knowledge-enriched anomaly reasoning |
| Graph-augmented fault diagnosis (2026) | 재검증 필요 | 구조는 좋으나 final citation 전 다시 확인 |

---

## 5. 실제 수집 순서 추천

### Step 1 — 핵심 spine 먼저 확보
먼저 아래 8개는 무조건 확보:

1. CLaSP
2. LaSTR
3. S2S-FDD
4. Technical language processing for maintenance work orders
5. Causal knowledge extraction from maintenance documents
6. Extracting failure time data from industrial maintenance records using text mining
7. SOPRAG
8. Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry

### Step 2 — graph/retrieval 확장 축 확보

9. RAIDR
10. RAGLog
11. RAINDROP
12. FC-STGNN
13. THGNN

### Step 3 — event alignment / process mining 축 확보

14. The ROAD from Sensor Data to Process Instances
15. Inferring Missing Event Log Data from IoT Sensor Data
16. Process mining on sensor data: a review

---

## 6. 집필 단계별 사용법

### Proposal / Introduction 쓰기 전
- 최우선 확보 목록 1~8부터 확인

### Method 쓰기 전
- 9~16번까지 같이 정리

### Related Work 마무리 전
- 재검증 후 확보 목록을 다시 확인하고,
  final version에 넣을지 말지 결정

---

## 7. 관련 문서

- `paper_d_surveyed_references.md`
- `paper_d_paper_comparison_table.md`
- `2026-04-14_related_literature_survey.md`
