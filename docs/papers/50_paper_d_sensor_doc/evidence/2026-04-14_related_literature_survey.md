# Paper D — Related Literature Survey

> 작성일: 2026-04-14
> 목적: 반도체 센서 데이터와 정비로그를 연결하는 박사논문 주제를 위한 관련 문헌 지형도 정리
> 상태: 1차 서베이 완료 (핵심 논문/갭/포지셔닝 포함)

---

## 1. 한 줄 요약

이 주제에서 가장 강한 연구 포지셔닝은 **"센서 시계열 + 정비로그 + 문서"를 같이 쓰는 것 자체가 아니라, 불완전한 시간 정렬(temporal uncertainty) 아래에서 센서 episode를 정비 사례와 근거 문서에 안정적으로 연결하는 grounded retrieval and diagnosis** 이다.

---

## 2. 문헌 지형도 (큰 흐름)

이 주제와 직접적으로 맞닿는 문헌은 대략 4개 축으로 나뉜다.

### A. 시계열-언어 정렬 / Cross-modal Retrieval
- 센서/시계열과 자연어를 같은 임베딩 공간에 맞추는 연구
- 논문 질문과 가장 직접적으로 연결되는 축
- 핵심 키워드: contrastive alignment, cross-modal retrieval, sensor-text retrieval, time-series language model

### B. 정비로그 / Maintenance Work Order Mining
- 실제 현장의 정비로그, work order, WRN(Work Request Notification)에서 원인/조치/증상 정보를 추출하는 연구
- thesis에서 maintenance log 측 근거성과 weak label의 기반이 되는 축

### C. Manufacturing/Industrial RAG 및 SOP/Document Grounding
- SOP, manual, troubleshooting guide 같은 절차 문서를 RAG 또는 GraphRAG로 검색하는 연구
- thesis에서 "관련 문서를 인용하며 답하는 진단기" 부분과 연결됨

### D. Semiconductor Fault Diagnosis / Predictive Maintenance
- 반도체 장비의 센서 기반 FDC/PdM/diagnosis 연구
- 실제 적용 도메인 설득력과 실데이터 기여를 강화하는 축

---

## 3. 우선 읽어야 할 핵심 논문 (Verified Core Papers)

아래 논문들은 이번 서베이에서 비교적 확인이 잘 된 핵심 문헌들이다.

### 3.1 시계열-언어 정렬 / Retrieval

#### 1) CLaSP: Learning Concepts for Time-Series Signals from Natural Language Supervision
- **출처**: arXiv 2411.08397
- **핵심**: 자연어 질의로 시계열 패턴을 검색하기 위해 contrastive learning으로 시계열과 텍스트를 공통 공간에 정렬
- **왜 중요한가**: thesis의 sensor-window ↔ maintenance-log retrieval 아이디어와 가장 직접적으로 닿는 출발점
- **한계**: maintenance log나 industrial diagnosis 자체를 다루지 않음
- **Paper D relevance**: 매우 높음 — retrieval backbone 관련 work의 핵심 anchor

#### 2) LaSTR: Language-Driven Time-Series Segment Retrieval
- **출처**: 2026 arXiv PDF 확인
- **핵심**: 전체 시계열이 아니라 **segment-level retrieval**을 정식화
- **왜 중요한가**: thesis에서 긴 센서 스트림 전체가 아니라 정비 episode 직전의 pre-failure window를 찾는 설정과 잘 맞음
- **한계**: maintenance log와의 연결은 직접 다루지 않음
- **Paper D relevance**: 매우 높음 — event-centric window retrieval과 직접 연결 가능

#### 3) TRACE: Grounding Time Series in Context for Multimodal Embedding and Retrieval
- **출처**: arXiv 2506.09114
- **핵심**: time series와 text를 aligned context로 묶고 hard negative mining을 사용하는 multimodal retriever
- **왜 중요한가**: 단순한 sample-level matching이 아니라, 더 정교한 retrieval 설계를 참고할 수 있음
- **한계**: semiconductor maintenance domain은 아님
- **Paper D relevance**: 높음 — dual encoder / hard negative 설계 참고용

#### 4) SensorLM
- **출처**: NeurIPS 2025 방향의 sensor-language foundation model 결과 검색됨
- **핵심**: 대규모 센서-텍스트 정렬, contrastive + captioning 조합
- **왜 중요한가**: paired sensor-text가 적은 상황에서 pseudo-caption이나 구조적 text 생성으로 학습 데이터를 늘리는 방향의 근거
- **한계**: wearable 중심, 산업 maintenance와 거리 있음
- **Paper D relevance**: 중간 이상 — 데이터 부족 대응 전략에서 중요

#### 5) FD-LLM: Large Language Model for Fault Diagnosis of Machines
- **출처**: arXiv 2412.01218
- **핵심**: 센서 신호를 언어 표현과 연결해 fault diagnosis를 수행하는 LLM 기반 접근
- **왜 중요한가**: signal-to-language 형태의 설명층과 diagnosis 연결을 지지
- **한계**: retrieval benchmark나 time-uncertainty-aware alignment까지는 약함
- **Paper D relevance**: 높음 — explanation layer / eventizer의 더 안전한 대체 근거

### 3.2 Maintenance Log / Work Order Mining

#### 6) Technical language processing for PHM: applying text similarity and topic modeling to maintenance work orders
- **출처**: Journal of Intelligent Manufacturing (2024/2025)
- **핵심**: maintenance work order를 technical language processing 관점에서 다루며, corrective action prediction과 topic extraction을 수행
- **왜 중요한가**: maintenance log가 단순 부가 정보가 아니라 predictive/prescriptive maintenance의 핵심 데이터 소스라는 강한 근거
- **한계**: 센서-로그 alignment는 직접 다루지 않음
- **Paper D relevance**: 매우 높음 — maintenance text 축의 대표 related work

#### 7) Causal knowledge extraction from long text maintenance documents
- **출처**: Computers in Industry 161 (2024), DOI 10.1016/j.compind.2024.104110
- **핵심**: 98,000개 real industrial WRN에서 cause-effect triple을 추출하고 causal graph DB를 구성
- **왜 중요한가**: long maintenance text에서 원인-결과 관계를 뽑아 graph화하는 방향이 thesis의 failure mode / action / evidence graph와 잘 맞음
- **한계**: 센서 데이터와 직접 결합하지 않음
- **Paper D relevance**: 매우 높음 — maintenance log를 구조화된 causal evidence로 바꾸는 근거

#### 8) Extracting failure time data from industrial maintenance records using text mining
- **출처**: Advanced Engineering Informatics 33 (2017), DOI 10.1016/j.aei.2016.11.004
- **핵심**: work orders와 downtime records에서 failure time을 더 정확히 복원
- **왜 중요한가**: thesis의 핵심 난제인 **temporal uncertainty**와 직접 연결됨
- **한계**: 현대적 multimodal retrieval 구조는 아님
- **Paper D relevance**: 매우 높음 — 시간 정렬 문제정의의 classic support

#### 9) Toward Semi-Autonomous Information Extraction for Unstructured Maintenance Data in RCA
- **출처**: NIST 공개 문서
- **핵심**: maintenance ticket를 구조화된 RCA 입력으로 바꾸는 PoC
- **왜 중요한가**: free-form maintenance logs를 tag/cluster/normalize해야 한다는 practical argument 제공
- **한계**: retrieval model보다는 preparation pipeline 성격
- **Paper D relevance**: 높음 — 데이터 정규화 파트에서 유용

### 3.3 Document Grounding / Industrial RAG

#### 10) SOPRAG: Multi-view Graph Experts Retrieval for Industrial Standard Operating Procedures
- **출처**: arXiv 2602.01858
- **핵심**: SOP retrieval은 일반 문서 RAG와 다르게 절차 구조, hierarchy, dependency를 반영해야 한다는 문제의식
- **왜 중요한가**: setup manual / SOP / troubleshooting guide를 flat chunking 하지 말아야 한다는 strong support
- **한계**: sensor-side retrieval과는 분리됨
- **Paper D relevance**: 매우 높음 — 문서 chunking 및 retrieval 설계 핵심 근거

#### 11) Document GraphRAG: Knowledge Graph Enhanced Retrieval Augmented Generation for Document QA within the Manufacturing Domain
- **출처**: Electronics 2025, DOI 10.3390/electronics14112102
- **핵심**: manufacturing dataset에서 naive RAG 대비 document graph 기반 retrieval 개선
- **왜 중요한가**: multi-hop 또는 문서 구조 기반 질문에 graph retrieval이 왜 필요한지 설득력 제공
- **한계**: maintenance-case retrieval 자체는 아님
- **Paper D relevance**: 높음 — SOP/manual grounding 계층 설계 근거

### 3.4 Industrial / Semiconductor Diagnosis

#### 12) Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry
- **출처**: PHM Society paper
- **핵심**: FDC sensor signals만으로는 오진 가능성이 있고, contextual/statistical information을 함께 써야 한다고 주장
- **왜 중요한가**: thesis에서 sensor-only diagnosis를 넘어서 maintenance/context/log를 함께 써야 한다는 도메인-specific 근거
- **한계**: modern multimodal retrieval 구조는 아님
- **Paper D relevance**: 매우 높음 — semiconductor-specific motivation 문장에 직접 사용 가능

#### 13) Machine Learning-Based Integrated Database System for PECVD Fault Diagnosis
- **출처**: Applied Science and Convergence Technology 2024
- **핵심**: 반도체 장비와 센서 데이터를 동기화 수집하고 fault diagnosis DB를 구성
- **왜 중요한가**: semiconductor equipment 데이터베이스와 fault diagnosis 연결의 실무성 강조
- **한계**: maintenance log retrieval까지는 가지 않음
- **Paper D relevance**: 중간 이상 — 실제 반도체 도메인 적용 측면에서 유용

---

## 4. 후보 문헌 (Promising but needs stricter verification)

아래는 방향은 매우 좋지만, 이번 턴에서 메타데이터/실험 설정을 더 엄밀히 확인해야 하는 문헌들이다.

- **Condition Insight Agent** — maintenance language + behavioral abstraction + failure semantics 쪽으로 매우 유망하나, 이번 턴에서 정확한 서지정보 추가 검증 필요
- **DML-LLM Hybrid fault diagnosis** — semiconductor + diagnosis 관점에서 흥미롭지만, 논문 메타데이터 확인 필요
- **MaintAGT** — maintenance multimodal LLM 계열로 유망하지만 thesis에서 직접 인용하려면 검증 필요
- **DiagnosticIQ** — evaluation benchmark 관점에서 좋아 보이나, 실제 공개성/정확한 위치 검증 필요

> 원칙: 이들은 서술에는 참고하되, 최종 논문 Related Work에 넣기 전에는 재검증이 필요하다.

---

## 5. 현재 문헌의 공백 (Gap Analysis)

### Gap 1. 센서-로그-문서를 한 프레임에서 다루는 연구가 드묾
- 시계열-언어 alignment 연구는 retrieval에 강하지만 maintenance log와 SOP grounding이 약함
- maintenance text mining 연구는 log structure extraction에 강하지만 sensor alignment가 약함
- manufacturing RAG 연구는 문서 grounding에 강하지만 sensor episode와의 직접 연결이 약함

### Gap 2. Temporal uncertainty가 주연으로 다뤄지지 않음
- maintenance log 시각은 실제 failure onset이 아니라 기록/승인/종료 시각일 수 있음
- 그러나 많은 파이프라인이 이 문제를 전처리 세부사항으로만 취급하거나, 아예 동기화된 것으로 가정함
- **Paper D가 가장 강하게 가져갈 수 있는 novelty 포인트는 여기**

### Gap 3. Grounded diagnosis의 evidence 평가가 약함
- 많은 논문이 "설명 가능" 또는 "grounded"를 말하지만,
  실제로는 특정 sensor segment, log span, SOP section을 명시적으로 연결해 평가하지 않음
- thesis에서는 retrieval accuracy 외에 **citation correctness / evidence faithfulness** 평가를 넣는 것이 차별점이 될 수 있음

### Gap 4. Semiconductor 실제 데이터 기반 공개 근거가 약함
- 반도체 장비 진단 연구는 많지만,
  **실제 maintenance log + 실제 sensor stream + 실제 diagnosis support**를 같이 쓰는 공개 연구는 매우 제한적
- 따라서 실데이터 기여는 방법론과 결합될 때 매우 강함

---

## 6. Thesis 포지셔닝 권고

### 가장 강한 주장

> 본 연구는 **반도체 장비의 실제 센서 데이터와 실제 정비로그를 이용하여**, 불완전한 시간 정렬 하에서 센서 episode와 maintenance case를 연결하고, 관련 문서 근거를 함께 제시하는 **uncertainty-aware grounded retrieval and diagnosis framework**를 제안한다.

### 피해야 할 약한 주장
- "최초의 multimodal retrieval"
- "최초의 LLM maintenance assistant"
- "센서와 로그를 같이 썼다"
- "RAG를 manufacturing에 적용했다"

→ 이런 문장은 약하거나 쉽게 반례가 나온다.

### 추천 논문 분할 포인트

#### Paper 1 — Dataset / Problem Definition
- 핵심: sensor episodes와 maintenance logs를 temporal uncertainty 하에서 연결하는 benchmark
- contribution: gold/silver/weak linking protocol, event-centric dataset, baseline

#### Paper 2 — Retrieval Model
- 핵심: uncertainty-aware sensor-window ↔ maintenance-log retrieval
- contribution: temporal compatibility score, contrastive alignment, hard negatives

#### Paper 3 — Grounded Diagnosis
- 핵심: retrieved cases + SOP/manual + ontology/FMEA를 사용한 근거 기반 diagnosis
- contribution: evidence-cited answer, verification loop, safety-aware response

---

## 7. 읽기 우선순위

### Tier 1 — 바로 읽기
1. CLaSP
2. LaSTR
3. FD-LLM
4. Technical language processing for maintenance work orders
5. Causal knowledge extraction from long text maintenance documents
6. Extracting failure time data from industrial maintenance records using text mining
7. SOPRAG
8. Effective Maintenance by Reducing Failure-Cause Misdiagnosis in Semiconductor Industry

### Tier 2 — 방법 확장용
1. TRACE
2. SensorLM
3. Document GraphRAG
4. KG-based in-context learning for fault diagnosis

### Tier 3 — 후보/검증 후 사용
1. Condition Insight Agent
2. DML-LLM hybrid diagnosis
3. MaintAGT
4. DiagnosticIQ

---

## 8. 지금 문헌 서베이가 Paper D에 주는 직접 시사점

### 이미 맞게 잡은 부분
- 센서 raw window를 바로 문서에 붙이지 않고 event/symptom으로 올리는 방향
- maintenance log를 weak label과 causal evidence의 원천으로 보는 방향
- SOP/manual을 flat chunk가 아닌 구조화된 절차 문서로 보는 방향
- grounded answer를 위해 citation/evidence를 강조한 방향

### 더 날카롭게 바꿔야 할 부분
- novelty 중심을 "agent"가 아니라 **temporal uncertainty-aware grounding**으로 이동
- evaluation 중심을 "accuracy"가 아니라 **retrieval + grounding quality + robustness to timestamp noise**로 이동
- maintenance log retrieval을 단순 검색이 아니라 **case retrieval under alignment uncertainty**로 명확히 문제정의

---

## 9. 다음 액션

1. 위 Tier 1 논문 8편의 정식 서지정보(BibTeX 수준) 재확인
2. `paper_d_research_proposal.md`의 Related Work/RQ 문장을 이번 서베이 기준으로 sharpen
3. `paper_d_paper_strategy.md`에 "temporal uncertainty-aware grounded retrieval" 문구를 명시적으로 반영
4. 논문별 비교표 작성
   - data modality
   - sensor-text alignment
   - maintenance log use
   - SOP/manual use
   - temporal uncertainty handling
   - evidence/citation output
5. 실데이터 agent에게 다음 숫자 확인 요청
   - sensor episode 수
   - maintenance log 수
   - gold/silver/weak link 가능 비율
   - SOP/manual alignment 가능 여부

---

## 10. 관련 내부 문서

- [[../paper_d_research_proposal.md]]
- [[../paper_d_architecture.md]]
- [[../paper_d_paper_strategy.md]]
- [[2026-04-14_raad_llm_review.md]]
