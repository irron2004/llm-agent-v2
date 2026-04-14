# Paper D — 관련 문헌 조사

> 조사일: 2026-04-14
> 목적: 센서-정비로그 연결 기반 근거형 진단 프레임워크 관련 연구 현황 파악

---

## 조사 영역 개요

Paper D의 핵심 주제는 6개 연구 영역에 걸쳐 있다:

1. **시계열-언어 정렬** (Time Series–Language Alignment)
2. **산업 문서 RAG** (Industrial Document RAG)
3. **이상탐지 + LLM** (Anomaly Detection with LLMs)
4. **정비로그 NLP/텍스트 마이닝** (Maintenance Log NLP)
5. **Knowledge Graph 기반 고장진단** (KG-Based Fault Diagnosis)
6. **센서 고장진단 + 크로스모달** (Cross-Modal Fault Diagnosis)

---

## 영역 1: 시계열-언어 정렬

Paper D의 핵심 학습 방법인 "sensor window ↔ document chunk contrastive retrieval"의 이론적 기반.

### CLaSP (2024)
- **제목**: CLaSP: Learning Concepts for Time-Series Signals from Natural Language Supervision
- **링크**: [arXiv 2411.08397](https://arxiv.org/abs/2411.08397)
- **핵심**: 자연어 질의로 시계열 신호를 검색하는 contrastive learning 모델. synonym dictionary 없이 LLM의 contextual knowledge를 활용.
- **데이터**: TRUCE, SUSHI 데이터셋 (시계열-자연어 쌍)
- **Paper D 관련성**: ★★★★★ — "자연어 → 시계열 검색"을 뒤집어 "시계열 → 문서 검색"으로 응용하는 핵심 참조

### LaSTR (2026)
- **제목**: LaSTR: Language-Driven Time-Series Segment Retrieval
- **링크**: [arXiv 2603.00725](https://arxiv.org/abs/2603.00725)
- **핵심**: 자연어 질의로 시계열 segment를 검색. TV2 기반 segmentation + GPT-5.2로 segment caption 생성 → Conformer 기반 contrastive retriever 학습. 공유 text–time-series 임베딩 공간.
- **Paper D 관련성**: ★★★★★ — segment 단위 검색이 Paper D의 sensor window 단위 검색과 직결

### SensorLM (2025, NeurIPS 2025 Poster)
- **제목**: SensorLM: Learning the Language of Wearable Sensors
- **링크**: [arXiv 2506.09108](https://arxiv.org/abs/2506.09108), [Google Research Blog](https://research.google/blog/sensorlm-learning-the-language-of-wearable-sensors/)
- **핵심**: 센서-언어 foundation model. **Hierarchical caption generation** (statistical → structural → semantic 3단계)으로 59.7M 시간, 103K명 규모의 센서-언어 데이터셋 구축. CLIP/CoCa 확장 아키텍처.
- **성과**: zero-shot, few-shot, cross-modal retrieval 모두 SOTA
- **Paper D 관련성**: ★★★★☆ — wearable 도메인이지만, hierarchical caption 전략과 paired data 생성 방법론이 Paper D의 pseudo-caption 학습에 직접 참고 가능

### SensorLLM (2025, EMNLP 2025)
- **제목**: SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition
- **링크**: [arXiv 2410.10624](https://arxiv.org/abs/2410.10624)
- **핵심**: 모션 센서와 LLM 정렬. HAR(Human Activity Recognition) 특화.
- **Paper D 관련성**: ★★★☆☆ — 센서-LLM 정렬의 방법론적 참고

---

## 영역 2: 산업 문서 RAG

Paper D의 문서 검색 및 구조화된 지식 활용 계층.

### SOPRAG (2026)
- **제목**: SOPRAG: Multi-view Graph Experts Retrieval for Industrial Standard Operating Procedures
- **링크**: [arXiv 2602.01858](https://arxiv.org/abs/2602.01858)
- **핵심**: SOP의 절차적 "형태"를 명시적으로 모델링하는 structure-aware RAG. 조건 의존성(condition-dependency)과 실행 가능성(actionable execution)을 위한 specialized graph experts 도입.
- **성과**: MRR 0.76, Acc@5 0.93 (Airline Services), GraphRAG 대비 5배 빠른 검색 (2.89s vs 14s+)
- **Paper D 관련성**: ★★★★★ — 반도체 SOP 검색의 직접 참조. flat chunking 대신 graph 구조 사용의 근거

### Document GraphRAG (2025)
- **제목**: Document GraphRAG: Knowledge Graph Enhanced RAG for Document QA Within the Manufacturing Domain
- **링크**: [MDPI Electronics](https://www.mdpi.com/2079-9292/14/11/2102)
- **핵심**: 문서 내부 구조로부터 KG를 구축하여 RAG 파이프라인에 통합. SQuAD, HotpotQA, 자체 제조 데이터셋에서 naive RAG baseline 대비 일관된 성능 향상.
- **Paper D 관련성**: ★★★★☆ — 문서 내부 구조 KG의 제조 도메인 적용 사례

### ManuRAG (2026)
- **제목**: ManuRAG: Multi-modal RAG for Manufacturing Question Answering
- **링크**: [Journal of Intelligent Manufacturing](https://link.springer.com/article/10.1007/s10845-026-02800-y)
- **핵심**: 제조 QA를 위한 멀티모달 RAG. 텍스트 외에 표, 수식, 이미지까지 포함한 문서 처리. 답변 정확도, 신뢰성, 해석가능성 개선.
- **Paper D 관련성**: ★★★★☆ — 반도체 매뉴얼의 멀티모달 특성(도면, 표, 다이어그램) 처리 참고

### KG-RAG for FMEA (2024-2025)
- **제목**: Knowledge Graph Enhanced Retrieval-Augmented Generation for Failure Mode and Effects Analysis
- **링크**: [arXiv 2406.18114](https://arxiv.org/abs/2406.18114), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2452414X25000317)
- **핵심**: FMEA 데이터를 KG로 구조화 → RAG에 통합. FMEA 데이터의 set-theoretic 표준화, FMEA-KG에서 vector embedding 생성 알고리즘. 전기차 배터리 셀 고전압 시스템 FMEA 데이터로 검증.
- **Paper D 관련성**: ★★★★★ — FMEA+KG+RAG 결합의 직접적 선행 연구. Paper D의 Stage 3 (구조화된 지식 결합)에 핵심 참고

### KEO (2025)
- **제목**: KEO: Knowledge Extraction on OMIn via Knowledge Graphs and RAG
- **링크**: [arXiv 2510.05524](https://arxiv.org/abs/2510.05524)
- **핵심**: 항공 정비 기록에서 KG 구축 → KG-based RAG로 safety-critical QA. semantic node identification, importance-aware graph expansion, structured context reconstruction. **KG가 global sensemaking에 강하고, text-chunk RAG가 local procedure 질의에 강함을 실험적으로 확인.**
- **Paper D 관련성**: ★★★★★ — KG vs text-chunk RAG의 역할 분리가 Paper D의 hybrid 검색 설계에 직접 참고

### Graph RAG Survey (2025)
- **제목**: Graph Retrieval-Augmented Generation: A Survey
- **링크**: [ACM TOIS](https://dl.acm.org/doi/10.1145/3777378)
- **핵심**: Graph RAG의 체계적 서베이. entity-relationship 구조가 reasoning 능력을 크게 개선하나, KG 추출 비용이 baseline RAG의 3~5배이고, 평균 2.3배 latency 증가.
- **Paper D 관련성**: ★★★☆☆ — 배경 서베이

---

## 영역 3: 이상탐지 + LLM

Paper D의 앞단 이벤트화(Eventizer) 계층.

### RAAD-LLM (2025)
- **제목**: RAAD-LLM: Adaptive Anomaly Detection Using LLMs and RAG Integration
- **링크**: [arXiv 2503.02800](https://arxiv.org/abs/2503.02800)
- **핵심**: frozen Llama 3.1 8B + SPC 기반 adaptive baseline + z-score CSV RAG + binarization rule. AAD-LLM 확장판.
- **성과**: use-case accuracy 0.71→0.89, F1 0.77→0.92; SKAB accuracy 0.58→0.72
- **한계**: RAG가 z-score lookup 중심 (문서 검색 아님), 변수 독립 처리, FAR 42%
- **Paper D 관련성**: ★★★★☆ — 앞단 이벤트화의 아이디어 참고용 (상세 리뷰: `2026-04-14_raad_llm_review.md`)

### RAG4CTS (2026)
- **제목**: RAG4CTS: Regime-Aware Native RAG Framework for Covariate Time-Series
- **링크**: [arXiv 2603.04951](https://arxiv.org/abs/2603.04951)
- **핵심**: 산업 시계열(predictive maintenance 포함)을 위한 regime-aware RAG. hierarchical knowledge base + two-stage bi-weighted retrieval + agentic augmentation strategy.
- **Paper D 관련성**: ★★★☆☆ — 시계열 전용 RAG 아키텍처 참고

### Agentic RAG for Industrial Anomaly Detection (2025)
- **링크**: [ResearchGate](https://www.researchgate.net/publication/390398885_Agentic_Retrieval-Augmented_Generation_for_Industrial_Anomaly_Detection)
- **핵심**: vector-based embeddings + vector DB로 실시간 파이프라인 센서 데이터 이상탐지.
- **Paper D 관련성**: ★★★☆☆ — agent 기반 산업 이상탐지의 최신 사례

---

## 영역 4: 정비로그 NLP / 텍스트 마이닝

Paper D의 정비로그 정규화 및 structured knowledge 구축 기반.

### Cleaning Maintenance Logs with LLM Agents (2025)
- **제목**: Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance
- **링크**: [arXiv 2511.05311](https://arxiv.org/abs/2511.05311), [PHM Society](http://papers.phmsociety.org/index.php/phmap/article/view/4486)
- **핵심**: LLM 에이전트로 정비로그 클리닝 파이프라인 구축. 6가지 noise 유형(typos, 결측, 중복, 잘못된 날짜 등) 처리. LLM이 generic cleaning에 효과적이나 도메인 특화 정확도는 한계.
- **Paper D 관련성**: ★★★★★ — Paper D의 로그 정규화 단계에 직접 적용 가능. 자동화 가능 범위와 한계를 명확히 보여줌

### Technical Language Processing for PHM (2024)
- **제목**: Technical Language Processing for Prognostics and Health Management
- **링크**: [Journal of Intelligent Manufacturing](https://link.springer.com/article/10.1007/s10845-024-02323-4)
- **핵심**: 항공기 MWO(Maintenance Work Orders)에 text similarity + topic modeling 적용. 기술 용어, 약어, 비표준 표현 처리.
- **Paper D 관련성**: ★★★★☆ — 산업 MWO의 NLP 처리 방법론 참고

### Automated Analysis of Maintenance Work Orders (2024)
- **제목**: Automated analysis and assignment of maintenance work orders using NLP
- **링크**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0926580524002371)
- **핵심**: NLP로 MWO 자동 분석 및 작업자 배정. 병원 MWO 데이터에서 accuracy 0.83.
- **Paper D 관련성**: ★★★☆☆ — MWO 자동 분류의 방법론 참고

### NLP in Industrial Maintenance: Systematic Review (2024)
- **제목**: Natural Language Processing Approaches in Industrial Maintenance: A Systematic Literature Review
- **링크**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050924002060)
- **핵심**: 산업 정비에서의 NLP 접근법 체계적 리뷰.
- **Paper D 관련성**: ★★★★☆ — 관련 연구 섹션의 핵심 서베이 참고

### Predictive Maintenance with Linguistic Text Mining (2024)
- **제목**: Predictive Maintenance with Linguistic Text Mining
- **링크**: [MDPI Mathematics](https://www.mdpi.com/2227-7390/12/7/1089)
- **핵심**: 텍스트 마이닝과 사이버물리 인프라를 결합한 예측 정비.
- **Paper D 관련성**: ★★★☆☆ — 배경 참고

---

## 영역 5: Knowledge Graph 기반 고장진단

Paper D의 구조화된 지식 결합(FMEA/SOP/Component Hierarchy) 계층.

### KG-Driven Equipment Fault Diagnosis (2024)
- **제목**: Knowledge Graph-Driven Equipment Fault Diagnosis Method for Intelligent Manufacturing
- **링크**: [Int J Adv Manuf Tech](https://link.springer.com/article/10.1007/s00170-024-12998-x)
- **핵심**: 다수준 KG 구축 → 다중 소스 데이터 기반 장비 고장진단. 각 레벨이 시스템 상태에 영향을 주는 요인 분석.
- **Paper D 관련성**: ★★★★☆ — component hierarchy + KG의 산업 적용

### KG for Operational Decision-Making in Maintenance: Systematic Review (2025)
- **제목**: Knowledge Graphs for Operational Decision-Making in Industrial Maintenance: A Systematic Review
- **링크**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S095741742503787X)
- **핵심**: 산업 정비에서의 KG 체계적 리뷰. 운영 의사결정 지원에서의 KG 역할.
- **Paper D 관련성**: ★★★★☆ — 관련 연구 섹션의 핵심 서베이

### KG-Enhanced Fault Diagnosis Bibliometric Review (2025)
- **제목**: Knowledge Graph-Enhanced Fault Diagnosis: A Bibliometric Review of AI Applications in Sensor Management (1998-2024)
- **링크**: [Springer Discover AI](https://link.springer.com/article/10.1007/s44163-025-00688-w)
- **핵심**: 센서 관리에서의 KG+AI 고장진단 bibliometric 분석.
- **Paper D 관련성**: ★★★☆☆ — 배경 서베이

### LLM-Based Fault Diagnosis: From Traditional ML to LLM Fusion (2025)
- **제목**: A Review of Fault Diagnosis Methods: From Traditional ML to Large Language Model Fusion Paradigm
- **링크**: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12846000/)
- **핵심**: 전통 ML → LLM 융합까지의 고장진단 방법론 리뷰. digital twin, KG, LLM 통합의 새로운 지능형 융합 특성.
- **Paper D 관련성**: ★★★★☆ — 포지셔닝용 핵심 서베이

---

## 영역 6: 크로스모달 / 멀티모달 고장진단

Paper D의 센서-텍스트 크로스모달 학습 및 멀티모달 진단.

### FailureSensorIQ (2025)
- **제목**: FailureSensorIQ: A Multi-Choice QA Dataset for Understanding Sensor Relationships and Failure Modes
- **링크**: [arXiv 2506.03278](https://arxiv.org/abs/2506.03278)
- **핵심**: 센서와 failure mode 간 관계 이해를 위한 MCQA 벤치마크. 10개 산업 자산, 8,296문항. ISO 문서 기반. FM2Sensor (failure mode → 센서) + Sensor2FM (센서 → failure mode) 양방향 질의. GPT-4, Llama, Mistral 등 12+ LLM 평가.
- **핵심 발견**: closed-source 모델도 perturbation/distraction에 취약, 도메인 지식 gap 존재.
- **Paper D 관련성**: ★★★★★ — "어떤 센서 이상이 어떤 failure mode와 연결되는가"가 Paper D의 핵심 질문과 정확히 일치

### MIT Multimodal Chatbot for Root Cause Diagnosis (2025)
- **제목**: Multimodal Generative AI Chatbot for Root Cause Diagnosis in Predictive Maintenance
- **링크**: [MIT DSpace](https://dspace.mit.edu/handle/1721.1/163337)
- **핵심**: 멀티모달 RAG 기반 root cause 진단 챗봇. 시계열 분석 모듈 + RAG 엔진 (기술 매뉴얼 + 정비 이력) + 이미지/오디오/비디오 지원. **진단 시간 약 30% 감소**.
- **Paper D 관련성**: ★★★★★ — Paper D의 최종 시스템과 가장 유사한 선행 연구. 차별점은 Paper D가 temporal uncertainty-aware alignment을 명시적으로 다루는 점

### TAMO (2025)
- **제목**: TAMO: Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data
- **링크**: [arXiv 2504.20462](https://arxiv.org/abs/2504.20462)
- **핵심**: multi-modal 관측 데이터를 time-aligned representation으로 통합 → 특화 도구로 RCA 수행하는 LLM agent.
- **Paper D 관련성**: ★★★★☆ — 클라우드 시스템 대상이지만, multi-modal time-aligned RCA의 아키텍처 참고

### Evidence-Driven Reasoning for Industrial Maintenance (2026)
- **제목**: Evidence-Driven Reasoning for Industrial Maintenance Using Heterogeneous Data
- **링크**: [arXiv 2603.08171](https://arxiv.org/abs/2603.08171)
- **핵심**: 이질적 데이터(센서, work order, FMEA, 운영 이력)를 근거 기반으로 통합한 산업 정비 추론. unsupported conclusion 억제를 위한 검증 루프. 주장과 권고를 명시적 근거에 연결.
- **Paper D 관련성**: ★★★★★ — Paper D의 "grounded diagnosis + verification loop" 개념의 가장 직접적인 선행 연구

### Weak Supervision for Predictive Maintenance: Survey (2025)
- **제목**: Weak Supervision: A Survey on Predictive Maintenance
- **링크**: [WIREs Data Mining](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.70022)
- **핵심**: labeled failure가 부족한 환경에서의 약한 감독 학습 전략 리뷰. historical work order와 SME knowledge 활용.
- **Paper D 관련성**: ★★★★☆ — Paper D의 weak label 생성 전략 참고

### Semiconductor Predictive Maintenance (2025)
- **링크**: [RSIS International](https://rsisinternational.org/journals/ijrsi/uploads/vol12-iss10-pg170-176-202510_pdf.pdf)
- **핵심**: 반도체 제조 환경에서의 예측 정비. class imbalance (AUC 0.95, Precision 0.66, Recall 0.96).
- **Paper D 관련성**: ★★★☆☆ — 반도체 도메인 배경

---

## 문헌 지도: Paper D 기여의 포지셔닝

```
                    시계열-언어 정렬
                   (CLaSP, LaSTR, SensorLM)
                          |
                          | contrastive retrieval
                          |
    정비로그 NLP ----→  [Paper D] ←---- 산업 문서 RAG
    (Log Cleaning,       핵심:          (SOPRAG, GraphRAG,
     TLP for PHM)    temporal-uncertainty  ManuRAG, KG-RAG FMEA)
                     aware sensor-log
                     retrieval + grounded
                     diagnosis
                          |
                          | 구조화된 지식
                          |
    이상탐지+LLM ----→ eventizer ←---- KG 고장진단
    (RAAD-LLM)        (앞단)         (KG-Driven FD,
                                      FailureSensorIQ)
                          |
                          | 멀티모달 진단
                          |
                   (MIT Chatbot, TAMO,
                    Evidence-Driven Reasoning)
```

### Paper D의 차별점

기존 연구와 비교했을 때 Paper D가 차별화되는 지점:

| 기존 연구 | 한계 | Paper D 차별점 |
|-----------|------|----------------|
| CLaSP, LaSTR | 일반 시계열-언어 검색 (산업 정비 미적용) | 산업 센서 → 정비로그 검색으로 특화 |
| RAAD-LLM | 이상탐지만 (문서 연결 없음) | 이상탐지 후 문서 연결 + 근거 기반 답변 |
| SOPRAG, ManuRAG | 문서 검색만 (센서 입력 없음) | 센서 이벤트를 검색 키로 사용 |
| MIT Chatbot | 범용 멀티모달 (시간 불확실성 미고려) | temporal uncertainty-aware alignment 명시적 모델링 |
| KG-RAG FMEA | FMEA 구조만 (센서 연결 없음) | 센서 + FMEA + SOP + 정비이력 통합 |
| FailureSensorIQ | 벤치마크만 (검색/진단 시스템 아님) | 검색 + 진단 + 설명 통합 프레임워크 |
| Evidence-Driven | 근거 추론만 (cross-modal 학습 없음) | sensor-log contrastive 학습 + grounded reasoning |

---

## 핵심 참고 논문 우선순위

### Tier 1: 반드시 정독 (Paper D의 직접 기반)
1. CLaSP — 시계열-언어 contrastive retrieval 원형
2. LaSTR — segment 단위 검색, caption 생성
3. SensorLM — hierarchical caption, 대규모 센서-언어 데이터셋
4. SOPRAG — SOP 구조 기반 RAG
5. KG-RAG FMEA — FMEA+KG+RAG 결합
6. KEO — KG vs text-chunk RAG 역할 분리
7. FailureSensorIQ — 센서-failure mode 관계 벤치마크
8. Evidence-Driven Reasoning — 근거 기반 정비 추론
9. Cleaning Maintenance Logs with LLM — 로그 클리닝 자동화
10. MIT Multimodal Chatbot — 가장 유사한 시스템

### Tier 2: 방법론적 참고
11. RAAD-LLM — eventizer 아이디어
12. Document GraphRAG — 제조 문서 KG-RAG
13. ManuRAG — 멀티모달 제조 RAG
14. TAMO — multi-modal time-aligned RCA
15. Technical Language Processing for PHM
16. Weak Supervision for PdM Survey

### Tier 3: 배경 서베이
17. NLP in Industrial Maintenance Review
18. KG for Maintenance Decision-Making Review
19. FDD in Industry 4.0 Review
20. LLM Fault Diagnosis Review (Traditional ML → LLM Fusion)
21. Graph RAG Survey

---

## 추가 조사 결과 (2차 검색)

### Condition Insight = Evidence-Driven Reasoning (2026)
- **제목**: Evidence-Driven Reasoning for Industrial Maintenance Using Heterogeneous Data
- **링크**: [arXiv 2603.08171](https://arxiv.org/abs/2603.08171)
- **핵심**: "Condition Insight"는 이 논문에서 제안된 deployed reasoning 프레임워크의 이름. **deterministic evidence construction과 constrained LLM reasoning을 분리**하는 구조.
  - LLM은 raw telemetry가 아닌 **structured summary 위에서 reasoning**
  - FMEA-derived component/mechanism constraints가 admissible explanation space를 제한
  - IoT/SCADA 플랫폼의 inconsistent naming, 비정형 정비 이력, 통합되지 않은 FMEA를 통합
  - 실무자들은 raw alert보다 "why"와 "what next" 중심의 structured reasoning을 선호
- **Paper D 관련성**: ★★★★★ — 이미 영역 6에 포함. Paper D의 "grounded diagnosis + verification loop" 설계에 가장 직접적인 참고. 특히 **evidence construction과 LLM reasoning 분리** 원칙은 Paper D 아키텍처의 핵심 설계 원칙으로 채택 가능.

### Anomaly Transformer (ICLR 2022 Spotlight)
- **제목**: Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy
- **링크**: [arXiv 2110.02642](https://arxiv.org/abs/2110.02642), [OpenReview](https://openreview.net/forum?id=LzQQ89U1qm_), [GitHub](https://github.com/thuml/Anomaly-Transformer)
- **저자**: Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long (THU)
- **핵심**: 비지도 시계열 이상탐지. **Association Discrepancy** — 이상 지점은 전체 시계열과 non-trivial association을 형성하기 어려워 인접 시점에 집중하는 경향을 이용. Anomaly-Attention 메커니즘 + minimax strategy로 정상-이상 구분력 증폭.
- **성과**: 6개 벤치마크 SOTA (service monitoring, space/earth exploration, water treatment)
- **Paper D 관련성**: ★★★★☆ — Paper D Stage 1 (이벤트화)에서 비지도 이상탐지 baseline으로 사용 가능. 규칙 기반 eventizer 이후의 고도화 옵션.

### FD-LLM: LLM for Fault Diagnosis (2024-2025)
- **제목**: FD-LLM: Large Language Model for Fault Diagnosis of Machines
- **링크**: [arXiv 2412.01218](https://arxiv.org/abs/2412.01218)
- **핵심**: 진동 신호를 **textual representation으로 encoding** → LLM과 정렬. classification-oriented 접근 + context-aware spectrum language modeling으로 explainable fault analysis.
- **방법**: 시계열 → 시간-주파수 도메인 feature 추출 → 텍스트화 → LoRA/Q-LoRA fine-tuning
- **성과**: zero-shot cross-condition 진단, cross-dataset few-shot 일반화
- **Paper D 관련성**: ★★★★☆ — 센서 신호 → 텍스트 변환 방법론의 직접 참고. Paper D의 eventizer에서 rule-based 이후 LLM 기반 고도화 옵션.
- **참고**: "S2S-FDD"라는 명칭의 단일 논문은 발견되지 않았으나, FD-LLM 및 관련 연구가 동일한 "sensor signal → sentence → fault diagnosis" 패러다임을 다룸.

### LLMs for Explainable Fault Diagnosis (2025)
- **제목**: Large Language Models for Explainable Fault Diagnosis of Machines
- **링크**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625031628)
- **핵심**: LLM이 센서 데이터에서 직접 고장을 감지/분류하면서 자연어 reasoning으로 설명 가능한 출력 생성.
- **Paper D 관련성**: ★★★☆☆ — explainable 고장 진단의 방법론 참고

### Fine-Tuned Thoughts (EMNLP 2025 Findings)
- **제목**: Fine-Tuned Thoughts: Leveraging Chain-of-Thought Reasoning for Industrial Asset Health Monitoring
- **링크**: [arXiv 2510.18817](https://arxiv.org/abs/2510.18817), [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1126.pdf)
- **핵심**: LLM → SLM으로 **CoT reasoning distillation**. IoT 센서(온도, 전력, 압력)로부터 산업 자산 건강 모니터링. MCQA 프롬프트로 reasoning 능력 전이.
- **성과**: fine-tuned SLM이 base model 대비 큰 폭으로 개선, LLM과의 격차 축소
- **Paper D 관련성**: ★★★☆☆ — Stage D(Answer SFT)에서 cost-effective한 SLM 활용 전략 참고

### Multimodal Large Model for Machine Fault Diagnosis (2026)
- **제목**: Multimodal Data-Enabled Large Model for Machine Fault Diagnosis Towards Intelligent O&M
- **링크**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2452414X26000026)
- **핵심**: 시계열 진동 신호와 텍스트 고장 지식을 **KG + GNN으로 cross-modal 정렬** → 텍스트 진단 리포트 자동 생성.
- **Paper D 관련성**: ★★★★☆ — sensor-text cross-modal alignment에 KG/GNN을 사용한 점이 Paper D의 ontology + contrastive retriever 설계와 직접 연관

### MindRAG 관련
- 검색 결과에서 "MindRAG"라는 제목의 특정 논문은 발견되지 않음
- 이전 대화에서 언급된 "annotation 내용과 기록의 전후 시차를 함께 해석" 개념은 **Evidence-Driven Reasoning (Condition Insight)**과 **Cleaning Maintenance Logs with LLM Agents**에서 더 명확하게 다루어짐
- 상태: 별도 논문으로 존재하지 않을 가능성 높음 → Condition Insight로 대체

---

## 최종 논문 목록 (업데이트)

### Tier 1: 반드시 정독 (12편)
1. CLaSP — 시계열-언어 contrastive retrieval 원형
2. LaSTR — segment 단위 검색, caption 생성
3. SensorLM — hierarchical caption, 대규모 센서-언어 데이터셋
4. SOPRAG — SOP 구조 기반 RAG
5. KG-RAG FMEA — FMEA+KG+RAG 결합
6. KEO — KG vs text-chunk RAG 역할 분리
7. FailureSensorIQ — 센서-failure mode 관계 벤치마크
8. Evidence-Driven Reasoning (Condition Insight) — 근거 기반 정비 추론, evidence/reasoning 분리
9. Cleaning Maintenance Logs with LLM — 로그 클리닝 자동화
10. MIT Multimodal Chatbot — 가장 유사한 시스템
11. FD-LLM — 센서 신호 → 텍스트 변환 + LLM 고장진단
12. Anomaly Transformer — 비지도 시계열 이상탐지 baseline

### Tier 2: 방법론적 참고 (8편)
13. RAAD-LLM — eventizer 아이디어
14. Document GraphRAG — 제조 문서 KG-RAG
15. ManuRAG — 멀티모달 제조 RAG
16. TAMO — multi-modal time-aligned RCA
17. Technical Language Processing for PHM
18. Weak Supervision for PdM Survey
19. Multimodal Large Model for Machine FD (2026) — KG+GNN cross-modal 정렬
20. Fine-Tuned Thoughts — CoT distillation for industrial SLM

### Tier 3: 배경 서베이 (6편)
21. NLP in Industrial Maintenance Review
22. KG for Maintenance Decision-Making Review
23. FDD in Industry 4.0 Review
24. LLM Fault Diagnosis Review (Traditional ML → LLM Fusion)
25. Graph RAG Survey
26. LLMs for Explainable Fault Diagnosis

---

## Sources

- [CLaSP - arXiv](https://arxiv.org/abs/2411.08397)
- [LaSTR - arXiv](https://arxiv.org/abs/2603.00725)
- [SensorLM - arXiv](https://arxiv.org/abs/2506.09108)
- [SensorLM - Google Research Blog](https://research.google/blog/sensorlm-learning-the-language-of-wearable-sensors/)
- [SensorLLM - arXiv](https://arxiv.org/abs/2410.10624)
- [RAAD-LLM - arXiv](https://arxiv.org/abs/2503.02800)
- [SOPRAG - arXiv](https://arxiv.org/abs/2602.01858)
- [Document GraphRAG - MDPI](https://www.mdpi.com/2079-9292/14/11/2102)
- [ManuRAG - Springer](https://link.springer.com/article/10.1007/s10845-026-02800-y)
- [KG-RAG FMEA - arXiv](https://arxiv.org/abs/2406.18114)
- [KG-RAG FMEA - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2452414X25000317)
- [KEO - arXiv](https://arxiv.org/abs/2510.05524)
- [Graph RAG Survey - ACM](https://dl.acm.org/doi/10.1145/3777378)
- [RAG4CTS - arXiv](https://arxiv.org/abs/2603.04951)
- [Agentic RAG for Industrial AD - ResearchGate](https://www.researchgate.net/publication/390398885_Agentic_Retrieval-Augmented_Generation_for_Industrial_Anomaly_Detection)
- [Cleaning Maintenance Logs - arXiv](https://arxiv.org/abs/2511.05311)
- [TLP for PHM - Springer](https://link.springer.com/article/10.1007/s10845-024-02323-4)
- [Automated MWO Analysis - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0926580524002371)
- [NLP in Industrial Maintenance Review - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877050924002060)
- [Predictive Maintenance with Text Mining - MDPI](https://www.mdpi.com/2227-7390/12/7/1089)
- [FailureSensorIQ - arXiv](https://arxiv.org/abs/2506.03278)
- [MIT Multimodal Chatbot - MIT DSpace](https://dspace.mit.edu/handle/1721.1/163337)
- [TAMO - arXiv](https://arxiv.org/abs/2504.20462)
- [Evidence-Driven Reasoning - arXiv](https://arxiv.org/abs/2603.08171)
- [Weak Supervision PdM Survey - WIREs](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.70022)
- [KG-Driven FD - Springer](https://link.springer.com/article/10.1007/s00170-024-12998-x)
- [KG for Maintenance Review - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S095741742503787X)
- [KG-Enhanced FD Bibliometric - Springer](https://link.springer.com/article/10.1007/s44163-025-00688-w)
- [FDD in Industry 4.0 Review - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723332/)
- [LLM Fault Diagnosis Review - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12846000/)
- [Condition Monitoring NLP-Assisted Review - MDPI](https://www.mdpi.com/2076-3417/15/10/5465)
- [Anomaly Transformer - arXiv](https://arxiv.org/abs/2110.02642)
- [Anomaly Transformer - GitHub](https://github.com/thuml/Anomaly-Transformer)
- [FD-LLM - arXiv](https://arxiv.org/abs/2412.01218)
- [LLMs for Explainable FD - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625031628)
- [Fine-Tuned Thoughts - arXiv](https://arxiv.org/abs/2510.18817)
- [Multimodal Large Model for Machine FD - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2452414X26000026)
- [LLM-based PHM Paper List - GitHub](https://github.com/CHAOZHAO-1/LLM-based-PHM)
- [Fault Cause ID across Manufacturing Lines - arXiv](https://arxiv.org/abs/2510.15428)
- [FMEA Builder - arXiv](https://arxiv.org/abs/2411.05054)
