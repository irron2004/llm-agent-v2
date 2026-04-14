# Papers TOC

이 디렉터리는 박사/논문 문서를 **주제별**로 정리한 canonical 위치입니다.
중복 성격 문서는 `90_legacy/`로 이동했습니다.

## 1. 박사 전략
- [박사 실행 로드맵](./00_strategy/ie_phd_rag_reliability_roadmap.md)
- [논문 가능성/리스크 평가](./00_strategy/pe_agent_논문가능성.md)
- [박사 챕터/논문화 설계](./00_strategy/research_toc.md)

## 2. 공통 규격 (A/B/C 공통)
- [공통 프로토콜 (Golden Set/Baseline/평가)](./10_common_protocol/paper_common_protocol.md)
- [합성 벤치마크 스펙](./10_common_protocol/synth_benchmark_stability_v1.md)

## 3. Paper A — Scope Safety
- 준비 중
- 예정 주제: Hierarchy-Constrained Retrieval, Contamination@k

## 4. Paper B — Stability
- [Paper B 본문 초고](./20_paper_b_stability/paper_b_stability.md)
- [Paper B 스펙/의사결정](./20_paper_b_stability/paper_b_stability_spec.md)
- [합성 벤치마크 실행 가이드](./20_paper_b_stability/paper_b_synth_runbook.md)
- Paper B 결과 자산:
  - [Table metrics (md)](./20_paper_b_stability/paper_b_assets/table_1_metrics.md)
  - [Table metrics (csv)](./20_paper_b_stability/paper_b_assets/table_1_metrics.csv)
  - [Figure 1 PNG](./20_paper_b_stability/paper_b_assets/figure_1_stability_vs_recall.png)
  - [Figure 2 PNG](./20_paper_b_stability/paper_b_assets/figure_2_driver_breakdown.png)

## 5. Paper C — Lifecycle Control
- 준비 중
- 예정 주제: Drift detection, rollback policy, risk-cost optimization

## 6. Paper D — Sensor-Document Linking Agent
- [README / 진행 현황](./50_paper_d_sensor_doc/README.md)
- [아이디어 정리 + 4단계 파이프라인](./50_paper_d_sensor_doc/paper_d_ideation.md)
- [학습 로드맵 (A~D 단계)](./50_paper_d_sensor_doc/paper_d_training_roadmap.md)
- [APC 에이전트 상세 아키텍처](./50_paper_d_sensor_doc/paper_d_architecture.md)
- [박사과정 연구계획서 초안](./50_paper_d_sensor_doc/paper_d_research_proposal.md)
- [논문 전략 및 프레이밍](./50_paper_d_sensor_doc/paper_d_paper_strategy.md)
- [데이터 검증 Agent 지시문](./50_paper_d_sensor_doc/paper_d_data_verification.md)
- [3가지 원칙 구현 예시](./50_paper_d_sensor_doc/paper_d_implementation_examples.md)
- Evidence:
  - [RAAD-LLM 논문 상세 리뷰](./50_paper_d_sensor_doc/evidence/2026-04-14_raad_llm_review.md)
  - [6개 영역 문헌 조사 (26편)](./50_paper_d_sensor_doc/evidence/2026-04-14_literature_survey.md)
  - [Paper D 포지셔닝 중심 문헌조사](./50_paper_d_sensor_doc/evidence/2026-04-14_related_literature_survey.md)
- 주제: 센서 이상 → 이벤트화 → 문서 검색 → 근거 기반 트러블슈팅 답변
- 핵심 키워드: Contrastive Retriever, Hybrid RAG, Sensor Eventization, Temporal Uncertainty, Ontology

## 7. Legacy
- [Paper B 작업용 초안(중복 성격, 보관)](./90_legacy/paper_b_stability_aware_retrieval.md)

---

## 정리 원칙
- `00_strategy`: 박사 레벨 전략/챕터 설계/실행 계획
- `10_common_protocol`: 논문 공통 데이터/평가 규격
- `20~`: 논문 주제별 문서와 산출물
- `90_legacy`: 중복/초안성 문서(참고용)
