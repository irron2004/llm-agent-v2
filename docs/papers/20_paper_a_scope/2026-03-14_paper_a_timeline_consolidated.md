# Paper A 통합 타임라인

작성일: 2026-03-14  
상태: Paper A 문서 묶음 전체를 위한 타임라인 및 소스 오브 트루스 가이드

## 읽기 가이드

이 문서는 Paper A 문서 세트를 타임라인 중심으로 통합 정리한 가이드다.  
원본 문서를 대체하지 않으며, 다음을 설명한다.

- Paper A 서사가 어떻게 바뀌었는지
- 어떤 문서가 기반 문서인지, 운영 문서인지, 오래된 문서인지
- 주요 전환점이 어디였는지
- 현재 방어 가능한 상태가 무엇인지
- 무엇이 아직 막혀 있는지

개별 Paper A 문서를 읽기 전에, 이 파일을 정식 인덱스이자 읽기 가이드로 사용한다.

## 이 문서가 하지 않는 일

- 기존 Paper A 원본 문서를 대체하거나 삭제하지 않는다.
- 과거 초안을 파일 안에서 직접 다시 쓰지 않는다.
- 뒤에 작성된 문서가 자동으로 더 낫다고 가정하지 않는다. 대신 어떤 목적에 어떤 문서가 더 강한지 설명한다.
- 실험 근거 추적 가능성을 위해, 의도적으로 링크 중심으로 구성한다.

## 한 줄 요약

Paper A는 처음에 hierarchy-aware scope-routing 논문으로 시작했고, v0.5 시기 평가에서 하드 필터가 재현율을 망친다는 부정적 결론을 거쳤다. 이후 2026-03-12에 편향을 줄인 masked-query 평가 프로토콜로 전환하면서 중심 주장이 바뀌었다. 즉, oracle device filtering은 masked 평가에서 contamination을 거의 제거하고 재현율까지 개선할 수 있다. 다만 masked 평가는 어디까지나 대리 설정(proxy)이며, 실제 배포 성능은 parser 품질, gold audit 한계, mixed-scope 평가 제약에 여전히 크게 의존한다.

## 문서 계열

### 1) 방향성과 범위를 정의하는 문서

이 문서들은 Paper A가 무엇을 목표로 하는지 정의한다.

| 경로 | 역할 | 현재 읽기 권장 |
|---|---|---|
| `docs/papers/20_paper_a_scope/README.md` | 상위 개요 및 초기 포지셔닝 | 좋은 입문 요약이지만, 최신 증거 전환 이전 내용 |
| `docs/papers/20_paper_a_scope/paper_a_series_map.md` | Paper A vs A-1 vs A-2 정의 | 여전히 유용 |
| `docs/papers/20_paper_a_scope/paper_a_series_blueprint.md` | 시리즈 전체 연구 청사진 | 범위 경계 확인에 여전히 유용 |
| `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` | 정식 실험 설계/spec v0.6 | 의도된 설계 확인에 중요하지만, 최신 결과 서사와는 차이 있음 |
| `docs/papers/20_paper_a_scope/related_work.md` | 관련연구 뼈대 | 보조 자료 |
| `docs/papers/20_paper_a_scope/references.bib` | 참고문헌 | 참조용 |

### 2) 메인 원고 상태 문서

이 문서들은 논문 작성 시점의 상태를 나타내며, 최신 검증 증거 상태와 완전히 같지는 않을 수 있다.

| 경로 | 역할 | 현재 읽기 권장 |
|---|---|---|
| `docs/papers/20_paper_a_scope/paper_a_scope.md` | v0.5/v0.5 유사 서사 중심의 구 원고 | 일부 내용은 구버전이지만 방법론 프레이밍에는 유용 |
| `docs/papers/20_paper_a_scope/paper_a_draft_v2.md` | masked-query 돌파 결과 중심의 신버전 초안 | 현재 논지에 가장 가깝지만, 표현 가드레일은 추가 필요 |

### 3) 증거 및 실험 문서

실제로 무엇이 있었는지 복원할 때 가장 중요한 문서들이다.

| 경로 | 역할 |
|---|---|
| `docs/papers/20_paper_a_scope/evidence/2026-01-08_meta_guided_hierarchical_rag.md` | 매우 초기 상위 설계 조상 문서. 현재 Paper A보다 범위가 넓음 |
| `docs/papers/20_paper_a_scope/evidence/2026-02-12_gcb_equip_id_matching_report.md` | GCB 문서에서 Equip-ID 매칭 가능성 검토 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md` | 코퍼스/device/shared-topic 통계 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md` | 초기 평가 라운드 실행 인덱스 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md` | 초기 오류 분석 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_main_results.md` | v0.5 시기 평가 기반 초기 메인 결과 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-09_gold_rejudging_analysis.md` | 과거 결과에 대한 재판정/alias 보정 시기 분석 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-12_cross_device_topic_feasibility.md` | shared/cross-device topic 가능성 분석 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-12_dataset_protocol_redesign.md` | 평가 편향 진단 후 프로토콜 재설계 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-12_slot_valve_hard_filter_recall_loss.md` | 구 서사 하에서 특정 실패 사례 분석 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md` | masked-query BM25 실험 돌파 요약 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_full_experiment_results.md` | 구/신 결과를 합친 마스터 스냅샷 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md` | oracle 대비 현실 parser 격차(장비+설비 모드 포함) |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md` | v0.6 gold 신뢰도 샘플 감사 요약 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md` | mixed-scope 평가 복원 및 split 재구성 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_implicit_eval.md` | 인덱스 차원 불일치로 인한 implicit 평가 차단 기록 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md` | B4.5가 B4보다 나쁜 이유 진단 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md` | Hybrid/rerank masked 실험 복구 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md` | 새 설정에서 soft scoring 재실험 음성 결과 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_remaining_tasks.md` | 돌파 이후 남은 백로그 |

### 4) 리뷰 및 감사 문서

이 문서들은 내부 비판 검토용이다. 과거 주장 중 일부를 그대로 믿으면 안 되는 이유를 설명한다.

| 경로 | 역할 |
|---|---|
| `docs/papers/20_paper_a_scope/review/preregistration.md` | 사전등록 가설 프레이밍 |
| `docs/papers/20_paper_a_scope/review/hypotheses_experiments.md` | 가설별 실험 계획 |
| `docs/papers/20_paper_a_scope/review/reviewer_report.md` | 초기 원고 상태에 대한 내부 리뷰 비판 |
| `docs/papers/20_paper_a_scope/review/consistency_audit.md` | 문서/데이터 불일치 및 오래된 주장 감사 |

### 5) 매핑 및 운영 문서

주장과 증거를 연결하거나, 다음 실행 항목을 정의하는 문서들이다.

| 경로 | 역할 | 현재 읽기 권장 |
|---|---|---|
| `docs/papers/20_paper_a_scope/evidence_mapping.md` | 주장-증거 매핑 매트릭스 | 전환 이전 주장 맵으로는 유용하지만 masked-query 서사와 재동기화 필요 |
| `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md` | 현재 실행 우선순위 문서 | 현재 운영 소스 오브 트루스 |

### 6) Paper A 폴더 밖의 작업/실행 로그 문서

원고 일부는 아니지만, 저장소에서 실제로 무엇을 실행했는지 확인할 때 중요하다.

| 경로 | 역할 | 현재 읽기 권장 |
|---|---|---|
| `docs/tasks/TASK-20260314-paper-a-execution-phase0.md` | 2026-03-14 Paper A 구현/증거 실행 로그 | `2026-03-14_execution_tasks.md`의 운영 보조 문서로 유용 |
| `docs/tasks/TASK-20260314-paper-a-doc-consolidation.md` | 이 통합 타임라인 문서 생성 작업 로그 | 메타 용도 |

## 타임라인

### 2026-01-08 — 넓은 아키텍처 조상 단계

- `docs/papers/20_paper_a_scope/evidence/2026-01-08_meta_guided_hierarchical_rag.md`
- 이 시점은 아직 현재의 Paper A가 아니다. contamination 제어 서사로 좁혀지기 전의 더 넓은 meta-guided hierarchical RAG 개념이다.
- 역사적 가치: 라우팅 중심 관점의 출발점을 보여주고, 이후 Paper A에 scope/hierarchy/routing 언어가 남은 이유를 설명한다.

### 2026-02-12 — Equip-ID 가능성이 구체화됨

- `docs/papers/20_paper_a_scope/evidence/2026-02-12_gcb_equip_id_matching_report.md`
- 핵심 결과: `GCB_number + Title` 조합으로 많은 GCB 아티팩트에서 유효한 `Equip_ID`를 복원할 수 있음.
- 역사적 가치: equip-level scope 처리가 개념 단계를 넘어 실무 가능해 보이기 시작한 지점.

### 2026-03-04 — 코퍼스 형태 정량화

- `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md`
- 핵심 결과:
  - 문서 578개
  - SUPRA 계열 비중이 높음
  - shared topic이 존재하되 보편적이지는 않음
  - topic 기반 family/shared 로직이 가능해 보임
- 역사적 가치: contamination 및 shared 정책 분석이 왜 필요한지 정당화한다.

### 2026-03-05 — 첫 본격 평가 라운드(구 프로토콜 기준)

핵심 문서:

- `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_main_results.md`
- `docs/papers/20_paper_a_scope/review/reviewer_report.md`
- `docs/papers/20_paper_a_scope/review/preregistration.md`
- `docs/papers/20_paper_a_scope/review/hypotheses_experiments.md`
- `docs/papers/20_paper_a_scope/review/consistency_audit.md`
- `docs/papers/20_paper_a_scope/evidence_mapping.md`

이 단계에서의 인식:

- 하드 필터는 contamination을 줄이지만 재현율에는 자주 불리해 보였다.
- shared/scope 정책은 불안정하거나 동어반복처럼 보였다.
- implicit/equip 슬라이스 표본은 작고 취약했다.

이 단계가 중요한 이유:

- 첫 번째 본격적인 음성 증거(negative evidence)를 만들었다.
- 동시에 가장 강한 자기비판도 생산했고, 이것이 이후 프로토콜 전환을 가능하게 했다.

왜 현재의 최종 진실은 아닌가:

- 평가셋이 작았다.
- parser alias 이슈와 shared 메트릭 동어반복이 해석을 흐렸다.
- 문서 중심 평가 프로토콜의 편향이 아직 진단되지 않았다.

### 2026-03-09 — 재판정 및 alias 보정 이후 정제

- `docs/papers/20_paper_a_scope/evidence/2026-03-09_gold_rejudging_analysis.md`
- 재판정 및 alias 정규화를 반영해 결과를 다시 점검한 단계.
- 분석 품질은 좋아졌지만, 여전히 구 평가 관점 안에서 이루어졌다.
- 역사적 가치: 과도기 문서이며 최종 상태는 아님.

### 2026-03-12 — 프로토콜 전환점

핵심 문서:

- `docs/papers/20_paper_a_scope/evidence/2026-03-12_dataset_protocol_redesign.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-12_cross_device_topic_feasibility.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-12_slot_valve_hard_filter_recall_loss.md`

실질적인 전환점이다.

변경된 내용:

- 팀이 순환형 gold bias와 lexical leakage를 진단했다.
- 장비명이 포함된 질의에서 BM25가 지름길로 정답 문서를 찾는 문제를 끊기 위해 `question_masked`를 도입했다.
- 질문이 “필터링이 재현율을 깨는가?”에서 “구 프로토콜이 필터링 효과를 제대로 측정할 수 있었는가?”로 바뀌었다.

이 날짜를 기준으로 구 Paper A 서사에서 현재 masked-query 서사로 전환된다.

### 2026-03-13 — masked-query BM25로 돌파

- `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`

돌파 내용을 가장 짧고 명확하게 요약한 문서다.

이 시점의 핵심 주장:

- contamination은 심각하다.
- oracle device filtering은 contamination을 거의 0으로 줄일 수 있다.
- masked 평가에서는 필터링이 재현율을 망치기보다 오히려 개선할 수 있다.
- 과거 음성 결과 상당수는 편향된 평가의 산물이었다.

즉, 논문 핵심이 “필터링은 해롭다”에서 “필터링을 잘못된 프로토콜로 측정하고 있었다”로 바뀐 시점이다.

### 2026-03-14 — 통합, 현실성 점검, 운영화

핵심 문서:

- `docs/papers/20_paper_a_scope/evidence/2026-03-14_full_experiment_results.md`
- `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_remaining_tasks.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_implicit_eval.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`
- `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`
- `docs/tasks/TASK-20260314-paper-a-execution-phase0.md`

2026-03-14가 달성한 것:

- masked-query 돌파를 전체 증거 패키지로 확장
- oracle 대비 현실 parser 격차 정량화
- 샘플 기반 감사 형태로 gold 신뢰도 패키징
- mixed-scope 데이터셋 커버리지 복원
- `B4.5 < B4` 역설 원인 설명
- 새 셋업에서 과거 P6/P7 soft-scoring 낙관이 성립하지 않음을 확인

이 라운드 이후에도 남은 차단 요인:

- 인덱스 임베딩 차원 불일치로 full implicit mixed-scope hybrid 평가가 막힘
- 원고 문구에 가드레일이 필요함(oracle 상한선을 배포 성능처럼 오해하지 않게)
- ambiguous 행은 empty-gold 커버리지 문제로 pooled hit 기반 주장에 아직 부적합

### 2026-03-15 — 지표 동기화 및 주장 가드레일

- `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
- masked BM25 hit 표를 per-query 아티팩트
  `data/paper_a/trap_masked_results.json`
  기준으로 동기화함.
- 동기화 값 (ALL, loose):
  - `B0_masked`: `287/578 (50%)`
  - `B4_masked`: `530/578 (92%)`
  - `B4.5_masked`: `430/578 (74%)`
- 방향성 논지는 바뀌지 않았다(필터링이 masked BM25에서 contamination과 hit를 모두 크게 개선). 다만 이 날짜는 서사 층의 수치 불일치를 닫았다는 점에서 중요하다.

## 의사결정 로그 (채택 / 기각 / 보류)

| 날짜 | 의사결정 | 상태 | 근거 문서 |
|---|---|---|---|
| 2026-03-12 | 구 문서 시드 평가 대신 편향 완화 masked-query 프로토콜을 주 평가 경로로 채택 | **채택** | `2026-03-12_dataset_protocol_redesign.md` |
| 2026-03-14 | Oracle B4는 배포 성능이 아니라 **상한선(upper bound)** 으로 보고 | **채택** | `2026-03-14_oracle_vs_parser_gap.md` |
| 2026-03-14 | 단순 shared 포함(`B4.5`)은 현재 패키지의 기본 정책으로 사용하지 않음 | **기각(현시점)** | `2026-03-14_b45_failure_decomposition.md` |
| 2026-03-14 | P6/P7 soft scoring은 현재 셋업의 주 방법으로 승격하지 않음 | **기각(현시점)** | `2026-03-14_masked_p6p7_reexperiment.md` |
| 2026-03-14 | Full BSP/Bayesian routing은 Paper A core 주장에 포함하지 않음 | **보류** | `paper_a_scope_spec.md` + 실행/증거 패키지 |

## 주요 서사 전환점

### 전환 1 — 넓은 라우팅 아키텍처에서 contamination 중심 논문으로

- 초기 설계의 에너지는 더 넓고 아키텍처 지향적이었다.
- 이후 Paper A는 contamination, scope, 평가 프로토콜 중심의 retrieval-safety 서사로 좁혀졌다.

### 전환 2 — “하드 필터는 재현율을 깨뜨린다”에서 “구 평가가 필터 가치를 숨겼다”로

- 구 서사: `2026-03-05_paper_a_main_results.md`, `paper_a_scope.md` 일부, 각종 리뷰 문서
- 신 서사: `2026-03-12_dataset_protocol_redesign.md` + `2026-03-13_paper_a_progress_summary.md` + `paper_a_draft_v2.md`

전체 문서 세트에서 가장 중요한 전환이다.

### 전환 3 — oracle-only 낙관에서 현실 제약 포함으로

- `2026-03-14_oracle_vs_parser_gap.md`는 과대주장을 막는 핵심 문서다.
- device-only parser는 contamination 격차가 크고, equip-aware 현실 모드는 그 격차를 상당히 줄임을 보여준다.
- 따라서 Paper A는 “B4가 된다”에서 멈추면 안 되고, “oracle B4는 상한선이며 현실 라우팅 품질은 parsing에 좌우된다”를 함께 말해야 한다.

### 전환 4 — “shared는 도움이 된다”에서 “shared는 순서/정책 설계가 중요하다”로

- 과거 문서들은 shared/family 완화를 재현율 회복 수단으로 다뤘다.
- `2026-03-14_b45_failure_decomposition.md`와 `2026-03-14_hybrid_rerank_recovery.md`는 단순 shared 포함이 재현율을 해칠 수 있음을 보여준다.
- 현재 패키지에서 shared 정책은 “검증된 승리 카드”가 아니라 “진단 및 수정이 필요한 영역”으로 남아 있다.

### 전환 5 — “soft scoring이 하드 필터를 이길 수 있다”에서 음성 결과로

- 과거 자료에는 P6/P7 낙관 여지가 있었다.
- `2026-03-14_masked_p6p7_reexperiment.md`가 현재 셋업에서는 그 가능성을 닫는다.

## 지금 시점에서 가장 신뢰할 문서

### 현재 운영/증거 소스 오브 트루스

현재 상태를 빠르게 파악하려면 아래를 먼저 읽는다.

1. `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
2. `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md`
3. `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md`
4. `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md`
5. `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md`
6. `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md`
7. `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md`
8. `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`

### 현재 원고 시작점으로 가장 좋은 문서

- `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`

### 설계/spec 참고에 가장 좋은 문서

- `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`

### 자기비판/감사에 가장 좋은 문서

- `docs/papers/20_paper_a_scope/review/consistency_audit.md`
- `docs/papers/20_paper_a_scope/review/reviewer_report.md`

## 주의해서 읽어야 할 문서

### `paper_a_scope.md`

- 방법론 프레이밍과 구 평가 프로토콜 설명에는 가치가 있다.
- 다만 small-split / planned-router / pre-breakthrough 서사가 여러 곳에 남아 있어, 단독 최신 상태 문서로 쓰기에는 위험하다.

### `evidence_mapping.md`

- 의도한 주장 구조를 이해하는 데는 유용하다.
- 하지만 일부 항목이 산출물이 이미 존재하는데도 과거 기준으로 미완료로 표기되어 있어, 상태 대시보드로는 신뢰하기 어렵다.

### `2026-03-14_full_experiment_results.md`

- 마스터 스냅샷으로 매우 유용하다.
- 다만 완료된 증거, 해석, 미래 작업 placeholder가 섞여 있다. 최종 원고 텍스트가 아니라 합성 노트(synthesis notebook)로 읽어야 한다.

### `2026-03-14_remaining_tasks.md`

- 당시 백로그 스냅샷으로는 좋다.
- 실행 순서는 `2026-03-14_execution_tasks.md`를 우선한다.

## 문서 간 수치 드리프트(불일치) 현황

반드시 오류라고 단정할 수는 없지만, 문서 문맥을 함께 읽어야 한다.

### 코퍼스 크기 드리프트: `578` vs `508`

- `578`은 `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md`, `docs/papers/20_paper_a_scope/README.md` 등 코퍼스/증거 중심 문서에 등장한다.
- `508`은 `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`에 등장한다.
- 원고 전체 동기화 전까지는, 하나가 단순 오기라고 단정하지 말고 문서 문맥별 카운트로 해석한다.

### shared 문서 수 드리프트: `124` vs `60`

- `124`는 `docs/papers/20_paper_a_scope/paper_a_scope.md`의 구 원고 프레이밍에 등장한다.
- `60`은 `docs/papers/20_paper_a_scope/evidence/2026-03-14_full_experiment_results.md`, `docs/papers/20_paper_a_scope/paper_a_draft_v2.md` 등 2026-03-14 증거 패키지 및 최신 원고에 등장한다.
- 구 서사 문서와 최신 증거가 아직 완전히 동기화되지 않았음을 보여주는 강한 신호로 본다.

### masked loose-hit 드리프트: `530/578` vs `532/578` (및 짝지어진 관련 값)

- `530/578 (92%)`는
  `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
  의 2026-03-15 동기화 상태에 등장하며,
  근거는 `data/paper_a/trap_masked_results.json`이다.
- `532/578 (92%)`는
  `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md` 및 관련 문서의 2026-03-14 hybrid 패키지 문맥에서 등장한다.
- 원고 본문용 단일 canonical scoreboard가 명시되기 전까지는 **실행 계열(run-family) 문맥 차이**로 해석한다.

## 현재 상태

2026-03-14 증거 패키지 기준으로, 가장 방어 가능한 Paper A 상태는 다음과 같다.

- 강하게 지지됨:
  - cross-equipment contamination은 심각하다.
  - oracle device filtering은 contamination을 거의 제거할 수 있다.
  - masked-query 평가는 기존 문서 시드 평가가 필터링 가치를 과소평가했음을 보여준다.
  - P6/P7 soft scoring은 현재 하드 필터보다 우수하지 않다.
- 단서와 함께 지지됨:
  - parser 품질을 명시적으로 다룰 때에만 현실 배포 성능이 oracle hit에 근접할 수 있다. contamination을 함께 보고하지 않으면 device-only parser의 hit는 오해를 부른다.
- 복원되었지만 아직 충분히 활용되지 않음:
  - mixed-scope 데이터셋 커버리지(`explicit_device`, `explicit_equip`, `implicit`, `ambiguous`)는 데이터셋 수준에서 복원됨.
- 아직 차단됨:
  - full mixed implicit hybrid 평가
  - 신뢰구간을 갖춘 원시 주석 기반의 더 엄격한 gold audit
  - 구/신 주장을 아우르는 원고 전역 동기화

## 기준 지표 (2026-03-15 동기화 기준)

현재 서사 레이어의 헤드라인 지표로 아래 값을 권장한다.

| 지표 | 값 | 출처 |
|---|---:|---|
| BM25 masked `B0` contamination@10 (ALL) | `0.518` | `2026-03-13_paper_a_progress_summary.md` |
| BM25 masked `B4` contamination@10 (ALL) | `0.000` | `2026-03-13_paper_a_progress_summary.md` |
| BM25 masked `B0` gold hit@10 loose (ALL) | `287/578 (50%)` | `2026-03-13_paper_a_progress_summary.md` |
| BM25 masked `B4` gold hit@10 loose (ALL) | `530/578 (92%)` | `2026-03-13_paper_a_progress_summary.md` |
| 현실 parser adj_cont@10 (device-only parser) | `30.6%` | `2026-03-14_oracle_vs_parser_gap.md` |
| scope-aware 현실 parser adj_cont@10 | `8.5%` | `2026-03-14_oracle_vs_parser_gap.md` |
| v0.6 strict precision (샘플 감사) | `97.2% (172/177)` | `2026-03-14_v06_gold_audit.md` |

## 인덱스 제약 스냅샷 (왜 일부 평가는 계속 막히는가)

현재 로컬 ES 셋업에서 관측된 상태:

| 인덱스 | lexical 평가용 text/doc_id | 임베딩 | `equip_id` 사용성 | 실무적 의미 |
|---|---|---|---|---|
| `chunk_v3_content` | 가능 | 없음 | 가능 | BM25/scope 메타데이터 분석에 강함, native dense retrieval은 없음 |
| `chunk_v3_embed_bge_m3_v1` | 메타데이터 뷰 제한적 | 있음 (1024) | 불가 | dense retrieval은 가능하나 equip-aware 메타데이터 평가는 제약 큼 |
| `rag_chunks_dev_v2` | 가능 | 있음 (legacy dims) | 가능 | 통합 fallback으로 유용하지만 v3 임베딩 셋업과 완전 정렬되지는 않음 |

이 표는 2026-03-14 증거 문서에서 반복되는 “index mismatch” 언급의 배경을 설명한다.

## 산출물 무결성 맵 (주장 -> 증거 -> 기계 아티팩트)

| 주장 계열 | 핵심 증거 문서 | 기계가 읽을 수 있는 아티팩트 |
|---|---|---|
| Masked BM25 contamination/hit 돌파 | `2026-03-13_paper_a_progress_summary.md` | `data/paper_a/trap_masked_results.json` |
| Oracle vs 현실 parser 격차 | `2026-03-14_oracle_vs_parser_gap.md` | `data/paper_a/parser_accuracy_report.json`, `data/paper_a/parser_accuracy_per_query_diff.csv` |
| v0.6 gold 신뢰도 | `2026-03-14_v06_gold_audit.md` | `data/paper_a/gold_verification_report.json` |
| `B4.5 < B4` 역설 진단 | `2026-03-14_b45_failure_decomposition.md` | `data/paper_a/masked_hybrid_results.json` |
| masked 셋업에서 P6/P7 음성 결과 | `2026-03-14_masked_p6p7_reexperiment.md` | `data/paper_a/masked_p6p7_results.json` |

## 권장 정본 읽기 순서

Paper A를 빠르고 정확하게 파악하려면 아래 순서를 권장한다.

1. `docs/papers/20_paper_a_scope/README.md`
2. `docs/papers/20_paper_a_scope/paper_a_series_map.md`
3. `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`
4. `docs/papers/20_paper_a_scope/review/consistency_audit.md`
5. `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
6. `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md`
7. `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md`
8. `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md`
9. `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md`
10. `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md`
11. `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md`
12. `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`
13. `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`

## 권장 사용 시나리오

### 논문 본문을 쓰는 경우

- `paper_a_draft_v2.md`를 작성 베이스로 사용한다.
- 2026-03-14 증거 문서로 주장 문구를 단단하게 조인다.
- `paper_a_scope.md`는 교차검증 없이 최종 진실 소스로 쓰지 않는다.

### 다음 구현 작업을 정할 때

- `2026-03-14_execution_tasks.md`를 먼저 본다.
- 과제를 미완료로 판단하기 전에 관련 증거 문서가 이미 있는지 확인한다.

### 주장이 오래되었는지 감사할 때

- `paper_a_scope.md` 및 2026-03-05 / 2026-03-09 구 증거와 아래 문서를 비교한다.
  - `2026-03-12_dataset_protocol_redesign.md`
  - `2026-03-13_paper_a_progress_summary.md`
  - 2026-03-14 증거 패키지

## 결론

Paper A 문서 세트는 단순 메모 모음이 아니다. 실제 서사 역전 과정을 기록한다.

- 초기 증거는 scope filtering이 재현율을 해친다고 보였다.
- 이후 프로토콜 분석으로 그 결론 상당 부분이 인공물(artifact)임이 드러났다.
- 현재 증거 패키지는 더 강한 contamination-control 논지를 지지한다.
- 단, oracle 대 현실 구분, gold audit 한계, mixed-scope 커버리지 제약을 명시적으로 함께 적을 때에만 그렇다.

이 타임라인 문서가 보존하려는 핵심은 바로 이 점이다.
