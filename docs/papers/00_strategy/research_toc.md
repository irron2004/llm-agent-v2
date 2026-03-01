# IE 박사 논문 연구 설계 및 로드맵

> 작성일: 2026-02-28
> 전제: 산업공학 박사, Q1 저널 1편 졸업요건
> 기반: `pe_agent_논문가능성.md` 주제 후보 → 본 문서에서 구체화
> [Role: 논문 구조 설계 / 연구 로드맵 / Paper A/B/C 아웃라인]

---

## 섹션 네비게이션

| 섹션 | 내용 |
|---|---|
| 0) Thesis Statement | 박사논문 한 줄 정의 |
| 1) 기여 4종 세트 | IE 박사 핵심 기여 구조 |
| 2) 챕터 구성 (Ch1-Ch9) | 목차 + 챕터별 기여/가설/모형/실험 |
| 3) 논문화 로드맵 | Paper A/B/C + Journal J 매핑 |
| 4) 연구 로드맵 (Stage 0-6) | 단계별 Task + 산출물 + 제출 계획 |
| 5) Paper A 상세 아웃라인 | 6-8p, Hierarchy-Constrained Retrieval |
| 6) Paper B 상세 아웃라인 | 6-8p, Stability-Aware Retrieval |
| 7) Paper C 상세 아웃라인 | 6-8p, Lifecycle Reliability Control |
| 8) Q1 저널 전략 (Journal J) | 12-18p, 통합 프레임워크 |
| 9) 운영 전략 | 제출 순서, 킬러 플롯, 저자 구조 |

> Paper B 상세 구현 문서:
> - Working draft → `docs/papers/20_paper_b_stability/paper_b_stability.md`
> - Decision spec → `docs/papers/20_paper_b_stability/paper_b_stability_spec.md`
> - Synth runbook → `docs/papers/20_paper_b_stability/paper_b_synth_runbook.md`
>
> 공통 실험 규격 → `docs/papers/10_common_protocol/paper_common_protocol.md`

---

## 0) Thesis Statement

> "Fab 폐쇄망 유지보수 RAG에서 정확도(accuracy)만이 아니라, 스코프 오류(타 장비 혼입), 검색/결론 변동성(instability), 수치/조건 정합성 위반(unfaithfulness)을 '운영 리스크'로 정식화하고, 이를 최소화하는 제약/정규화/검증/업데이트 통제 정책을 설계/최적화하여 현장 KPI와 함께 검증한다."

### 추천 논문 제목 (안)

1. 산업용 RAG의 운영 신뢰성 최적화: 장비 계층 제약, Top-k 안정성, 정합성 검증 및 폐쇄망 라이프사이클 통제
2. 폐쇄망 반도체 유지보수 지식서비스에서 Retrieval-Generation 파이프라인의 신뢰성 지표 체계와 최적화 방법론
3. (영문) Reliability-Aware Retrieval-Augmented Generation for On-Prem Semiconductor Maintenance: Scope Constraints, Stability Optimization, and Lifecycle Control

---

## 1) IE 박사 핵심 기여 4종

| 기여 | 설명 |
|---|---|
| **기여 1: 운영 신뢰성 지표/리스크 정식화** | Top-k 안정성, 타 장비군 혼입률, 수치/조건 불일치율을 현장 리스크(재작업, 재진입, 조치 실패)와 연결되는 품질지표로 정식화 |
| **기여 2: 스코프 제약 검색/재랭킹** | "검색 결과에 타 장비 문서가 섞이지 않게"를 제약조건으로 넣고, 정확도/재현율 손실을 최소화하는 방식 제안 |
| **기여 3: Top-k/결론 변동성 안정성 최적화** | "같은 질문인데 답이 흔들린다"를 분산/변동성 최소화 문제로 만들고, paraphrase set 기반 안정성 정규화/강건최적화 제안 |
| **기여 4: 지식 업데이트/롤백/드리프트 통제 정책** | 문서 버전 변화 → 임베딩 인덱스 변화 → 성능 회귀/불안정을 통계적 공정관리(SPC)/최적 정책으로 다룸 |

---

## 2) 챕터 구성 (Ch1-Ch9)

### Ch1. Introduction (서론)

- 산업 RAG 운영 환경의 특수성 (폐쇄망, 장비 계층, 보안 제약)
- 기존 retrieval 연구의 한계: 평균 정확도 중심 → 운영 리스크 미고려
- 연구 목적: 5대 품질특성 기반 통합 신뢰성 프레임워크

### Ch2. Literature Review (문헌 연구)

- RAG 시스템 발전사 (Dense Retrieval → Hybrid → Rerank → Agentic)
- 산업 도메인 IR/QA 연구 동향
- 검색 안정성/재현성 관련 연구 (limited)
- IE 품질공학/SPC/신뢰성 이론과의 접점
- 연구 갭: 운영 품질을 체계적으로 다룬 RAG 연구 부재

### Ch3. Research Framework (연구 프레임워크)

- **품질특성 5대 체계**: 정확도, 안전성(스코프), 반복성(안정성), 충실성(정합성), 지속가능성(라이프사이클)
- **운영 리스크 정식화**: `Risk = c₁·C@k + c₂·(1-Stab@k) + c₃·E[V]`
- **교란 계층 (Perturbation Tiers)**: T1 Repeat, T2 Paraphrase, T3 Restart, T4 Reindex
- **Golden Set 설계**: → 상세: `docs/papers/10_common_protocol/paper_common_protocol.md`
- **Baseline 정의**: BM25 / Dense / Hybrid / Hybrid+Rerank 4종

### Ch4. Hierarchy-Constrained Retrieval (Paper A)

> → 상세 구현 시 `paper_a_*.md` 문서 생성 예정

- **가설 H-A**: 장비 계층 메타데이터를 검색 단계에 제약으로 주입하면, Contamination@k를 α% 이상 감소시키면서 Recall@k 손실을 β% 이내로 유지할 수 있다
- **모형**: 계층 필터링 (pre-retrieval) + 제약 리랭킹 (post-retrieval)
- **실험**:
  - Contamination Challenge Set에서 제약 유무 비교
  - 계층 수준별 (equipment > module > sub-module) 정밀도-재현율 트레이드오프
  - Ablation: 필터 vs 리랭킹 vs 결합

### Ch5. Stability-Aware Retrieval (Paper B)

> → 상세: `docs/papers/20_paper_b_stability/paper_b_stability.md`, `docs/papers/20_paper_b_stability/paper_b_stability_spec.md`

- **가설 H-B1**: stability-aware 설정은 Stability@k(Jaccard)와 Consistency를 유의하게 개선
- **가설 H-B2**: 안정성 개선은 Recall@k 손실 없이 또는 제한된 손실로 달성 가능
- **가설 H-B3**: instability는 ANN 비결정성/문서 중복/질의 모호성/업데이트 요인으로 분해 가능
- **모형**:
  - Deterministic protocol (seed, shard routing, tie-break)
  - Stability-aware MQ (consensus retrieval)
  - 강건 최적화: `max E[R@k] s.t. Stab@k >= τ`
- **실험**:
  - T1-T4 계층별 안정성 측정
  - Stability-Recall 파레토 곡선
  - Instability 원인 분해 (driver analysis)
- **현재 상태**: 합성 벤치마크 완료, deterministic protocol 구현 완료, 논문 초고 작성 중

### Ch6. Lifecycle Reliability Control (Paper C)

> → 상세 구현 시 `paper_c_*.md` 문서 생성 예정

- **가설 H-C**: 문서 업데이트 후 인덱스 변경이 검색 성능 회귀를 유발하며, SPC 기반 모니터링과 최적 업데이트 정책으로 통제 가능
- **모형**:
  - 변경 영향 전파 모형: `ΔDoc → ΔEmb → ΔIndex → ΔPerf`
  - SPC 관리도 기반 드리프트 감지 (X-bar, CUSUM)
  - 최적 업데이트 정책: batch vs incremental, rollback trigger
- **실험**:
  - 문서 10%/20%/50% 변경 시 성능 회귀 측정
  - 관리도 감도 분석 (ARL₀, ARL₁)
  - Version-Stability Set에서 정책 비교

### Ch7. Integrated Reliability Framework (Journal J)

> → 상세: §8 Q1 저널 전략 참조

- Paper A/B/C 결과를 통합하는 다목적 최적화 프레임워크
- 파레토 최적해 집합 도출
- 현장 가중치 학습 또는 전문가 설정
- end-to-end 파이프라인 최적화

### Ch8. Empirical Validation (실증 검증)

- Golden set 기반 종합 평가
- 합성 벤치마크 + 실운영 데이터 교차 검증
- 현장 KPI 연결 (재작업률 감소, 대응시간 단축)
- 공개 도메인 2차 검증 (일반화 입증)

### Ch9. Conclusion (결론)

- 기여 요약, 한계점, 향후 연구
- 산업 RAG 운영 품질 연구의 로드맵 제시

---

## 3) 논문화 로드맵: Paper → Chapter 매핑

| Paper | 주제 | 목표 학회/저널 | Thesis Ch |
|---|---|---|---|
| **Paper B** | Stability-Aware Retrieval | EMNLP Industry / ECIR | Ch5 |
| **Paper A** | Hierarchy-Constrained Retrieval | CIKM / SIGIR Industry | Ch4 |
| **Paper C** | Lifecycle Reliability Control | IISE Trans. / C&IE | Ch6 |
| **Paper D** (opt.) | Numeric Faithfulness Verification | — (Ch7에 통합 가능) | — |
| **Journal J** | Integrated Reliability Framework | IISE Trans. / JQT / C&IE (Q1) | Ch7 |

- Paper A/B/C: 각 6-8페이지, 컨퍼런스 레벨
- Journal J: 12-18페이지, Paper A/B/C 결과 통합 + 프레임워크
- 졸업 요건: Journal J (Q1) 1편 = 충분

---

## 4) 연구 로드맵 (Stage 0-6)

> 실행 체크리스트: `docs/papers/00_strategy/ie_phd_rag_reliability_roadmap.md` 참조

### Stage 0: 기반 구축 (현재 ~ +1개월) ← **진행중**

- [x] 시스템 안정화 (mq_mode, guardrails, deterministic protocol)
- [x] Paper B 합성 벤치마크 구축
- [x] 안정성 평가 인프라 (eval runner, stability audit)
- [ ] Golden set v0 설계 및 pilot 라벨링 (50건)
- [ ] Baseline 4종 성능 측정

### Stage 1: Paper B 완성 (+1 ~ +3개월)

- [ ] Golden set v0 확장 → Paraphrase Stability Set (100건)
- [ ] Deterministic vs Consensus 실험 (n=20 반복)
- [ ] Instability driver 분석
- [ ] Paper B 원고 완성 및 투고

### Stage 2: Paper A 실험 (+3 ~ +6개월)

- [ ] Contamination Challenge Set 구축 (60건)
- [ ] 계층 필터/리랭킹 구현 및 실험
- [ ] Paper A 원고 작성 및 투고

### Stage 3: Paper C 실험 (+6 ~ +10개월)

- [ ] 문서 버전 관리 인프라 구축
- [ ] 변경 영향 전파 실험 + SPC 관리도
- [ ] Paper C 원고 작성 및 투고

### Stage 4: 통합 프레임워크 (+10 ~ +14개월)

- [ ] Paper A/B/C 결과 통합 + 다목적 최적화
- [ ] Journal J 원고 작성 및 투고

### Stage 5: 현장 검증 (+14 ~ +18개월)

- [ ] 실운영 A/B 테스트 + 현장 KPI 수집
- [ ] 공개 도메인 2차 검증

### Stage 6: 학위논문 통합 (+18 ~ +24개월)

- [ ] Ch1-Ch9 통합 집필 + 최종 방어

---

## 5) Paper A 상세 아웃라인

### 제목 (안)
**Hierarchy-Constrained Retrieval for Scope-Safe Industrial RAG: Reducing Cross-Equipment Contamination in Semiconductor Maintenance Knowledge Systems**

### 구성 (6-8p)

| § | 내용 | 분량 |
|---|---|---|
| 1 | Introduction: 산업 RAG에서 scope 오류의 위험성 | 0.8p |
| 2 | Related Work: 계층 검색, 필터링, constrained retrieval | 0.7p |
| 3 | Problem: Contamination@k 정의, 장비 계층 모델 | 0.8p |
| 4 | Method: Pre-retrieval 계층 필터 + Post-retrieval 제약 리랭킹 | 1.2p |
| 5 | Experiments: Contamination Challenge Set, 계층 수준별 비교 | 1.5p |
| 6 | Results & Analysis: Contamination 감소 + Recall 트레이드오프 | 1.0p |
| 7 | Discussion & Conclusion | 0.5p |

### 핵심 그림/표
- **Fig A1**: 장비 계층 구조 + 검색 범위 오류 예시
- **Table A1**: Contamination@k (계층 수준별, 방법별)
- **Table A2**: Recall@k vs Contamination@k 트레이드오프
- **Fig A2**: Precision-Recall 곡선 (제약 유무)

### RQ & Hypotheses
- RQ-A: 장비 계층 정보를 검색에 주입하면 scope 오류를 유의하게 줄일 수 있는가?
- H-A1: Pre-retrieval 필터는 Contamination@k를 50%+ 감소
- H-A2: Post-retrieval 리랭킹은 Recall@k 손실 5% 이내로 Contamination 감소

---

## 6) Paper B 상세 아웃라인

> → 전체 구현: `docs/papers/20_paper_b_stability/paper_b_stability.md` (논문 초고)
> → Decision spec: `docs/papers/20_paper_b_stability/paper_b_stability_spec.md`

### 제목
**Stability-Aware Retrieval for Reliable RAG Operations: Optimizing Top-k Consistency under Repeated and Equivalent Queries**

### 구성 (6-8p)

| § | 내용 | 분량 |
|---|---|---|
| 1 | Introduction: RAG 검색 비결정성과 운영 영향 | 0.8p |
| 2 | Metrics and Protocol: T1-T4 교란 계층, Jaccard/ExactMatch 지표 | 1.0p |
| 3 | Methods: Deterministic protocol + Stability-aware MQ | 1.2p |
| 4 | Synthetic Benchmark: 60 groups × 4 paraphrases = 240 queries | 0.8p |
| 5 | Experiments: T1-T4 결과 + driver analysis | 1.5p |
| 6 | Discussion: 한계점, reindex 불안정성 | 0.5p |
| 7 | Conclusion | 0.3p |

### 핵심 그림/표
- **Fig B1**: Stability-aware retrieval 아키텍처
- **Table B1**: RepeatJaccard@10, ParaphraseJaccard@10, hit@10, p95 latency
- **Fig B2**: Stability-Recall 파레토 곡선
- **Table B2**: Instability 원인 분해 (ANN / 중복 / 모호성 / 업데이트)
- **Table B3**: Paraphrase 강도별 성능

### 현재 진행 상태
- [x] 합성 벤치마크 생성기 (`scripts/evaluation/synth_bench_generator.py`)
- [x] Eval runner (`scripts/evaluation/retrieval_stability_audit.py`)
- [x] Deterministic protocol 구현 및 검증
- [x] T1/T3 안정성 = 1.0 달성
- [x] 논문 초고 작성 (`docs/papers/20_paper_b_stability/paper_b_stability.md`)
- [ ] T2 Paraphrase 실험 확장 (n=20)
- [ ] Driver analysis 정량화
- [ ] 최종 원고 polish + 투고

---

## 7) Paper C 상세 아웃라인

### 제목 (안)
**Lifecycle Reliability Control for Industrial RAG: Detecting and Mitigating Retrieval Performance Drift under Document Updates**

### 구성 (6-8p)

| § | 내용 | 분량 |
|---|---|---|
| 1 | Introduction: 문서 업데이트가 RAG 성능에 미치는 영향 | 0.8p |
| 2 | Related Work: 개념 드리프트, 모델 모니터링, SPC | 0.7p |
| 3 | Problem: 변경 전파 모형 ΔDoc → ΔEmb → ΔIndex → ΔPerf | 0.8p |
| 4 | Method: SPC 관리도 (X-bar, CUSUM) + 최적 업데이트 정책 | 1.2p |
| 5 | Experiments: 변경 비율별 회귀 측정, 관리도 감도 분석 | 1.5p |
| 6 | Results & Discussion | 1.0p |
| 7 | Conclusion | 0.3p |

### 핵심 그림/표
- **Fig C1**: 변경 전파 다이어그램 (Doc → Emb → Index → Perf)
- **Table C1**: 변경 비율별 (10%/20%/50%) 성능 회귀
- **Fig C2**: SPC 관리도 (Recall@k 시계열 + 관리 한계)
- **Table C2**: 업데이트 정책 비교 (batch vs incremental vs hybrid)
- **Fig C3**: ARL₀ vs ARL₁ 감도 곡선

### RQ & Hypotheses
- RQ-C: 문서 업데이트 후 검색 성능 회귀를 SPC로 감지하고 통제할 수 있는가?
- H-C1: 문서 20% 이상 변경 시 Recall@k 통계적 유의 감소 발생
- H-C2: CUSUM 관리도는 5% 회귀를 ARL₁ < 10 내에 감지
- H-C3: 최적 업데이트 정책은 naive batch 대비 드리프트 기간 50%+ 감소

---

## 8) Q1 저널 전략 (Journal J)

### 제목 (안)
**Reliability-Aware Retrieval-Augmented Generation for Industrial Knowledge Services: An Integrated Framework for Scope Safety, Stability Optimization, and Lifecycle Control**

### 목표
- Q1 저널 1편 (졸업 요건 충족)
- 타겟: IISE Transactions / Journal of Quality Technology / Computers & Industrial Engineering

### 구성 (12-18p)

| § | 내용 | 분량 |
|---|---|---|
| 1 | Introduction | 1.5p |
| 2 | Literature Review & Research Gap | 2.0p |
| 3 | Framework: 5대 품질특성 + 운영 리스크 공식화 | 2.0p |
| 4 | Method 1: Scope Constraint (Paper A 핵심) | 1.5p |
| 5 | Method 2: Stability Optimization (Paper B 핵심) | 1.5p |
| 6 | Method 3: Lifecycle Control (Paper C 핵심) | 1.5p |
| 7 | Integrated Optimization: 다목적 최적화 + 파레토 | 2.0p |
| 8 | Experiments: Golden set 종합 평가 + 현장 KPI | 2.5p |
| 9 | Discussion | 1.0p |
| 10 | Conclusion | 0.5p |

### IE 저널 적합성

| 저널 | IF | 적합 근거 |
|---|---|---|
| IISE Transactions | ~2.6 | IE 시스템 최적화, 품질관리 방법론 |
| JQT (Journal of Quality Technology) | ~2.5 | SPC 기반 통제, 품질특성 정식화 |
| Computers & IE | ~7.9 | 산업 AI 시스템, 데이터 기반 의사결정 |
| Expert Systems with Applications | ~8.5 | 산업 응용, AI 시스템 (backup) |

---

## 9) 운영 전략

### 제출 순서 (추천)

```
Paper B (안정성) → Paper A (스코프) → Paper C (라이프사이클) → Journal J (통합)
```

**근거:**
1. Paper B가 가장 진행도 높음 (합성 벤치마크 + 코드 완성)
2. Paper A는 B 다음으로 데이터 확보 용이
3. Paper C는 버전 관리 인프라 구축 필요 → 가장 나중
4. Journal J는 A/B/C 결과 확보 후 통합

### 킬러 플롯 전략

- 매 Paper마다 1개 "한눈에 보이는 그림" 필수
- Paper B: Stability-Recall Pareto curve (기존 연구에 없는 trade-off 시각화)
- Paper A: Contamination heatmap (장비 계층별 오류 분포)
- Paper C: SPC 관리도 + 드리프트 타임라인
- Journal J: 3D Pareto surface 또는 Quality Radar Chart

### 저자 구조

| 순서 | 역할 |
|---|---|
| 1저자 | 본인 (연구 설계, 구현, 실험, 집필) |
| 교신저자 | 지도교수 |
| 공저자 (선택) | 도메인 전문가 / 공동 연구자 |

### Golden Set 공통 규격

→ `docs/papers/10_common_protocol/paper_common_protocol.md` 참조 (baseline 4종, 지표 6종, golden set 스키마, 실행 로그)

---

## Appendix: 교차참조 테이블

| 문서 | 역할 |
|---|---|
| `pe_agent_논문가능성.md` | 가능성 진단, 리스크 분석, 주제 후보 6개 |
| `research_toc.md` (본 문서) | 챕터 구조, Paper 아웃라인, Stage 로드맵 |
| `docs/papers/10_common_protocol/paper_common_protocol.md` | Golden Set 스키마, Baseline, 평가 하네스 |
| `ie_phd_rag_reliability_roadmap.md` | 실행 체크리스트, 우선순위 Top 10 |
| `docs/papers/20_paper_b_stability/paper_b_stability.md` | Paper B 논문 초고 |
| `docs/papers/20_paper_b_stability/paper_b_stability_spec.md` | Paper B decision-complete 스펙 |
| `docs/papers/20_paper_b_stability/paper_b_synth_runbook.md` | 합성 벤치마크 생성 runbook |
