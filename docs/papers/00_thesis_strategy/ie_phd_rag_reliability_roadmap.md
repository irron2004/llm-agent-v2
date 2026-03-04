# 산업공학 박사 연구 로드맵 (RAG 신뢰성 중심)

> [Role: 실행 로드맵 / 체크리스트]
> 관련 문서:
> - 가능성 진단 + 리스크: `docs/papers/00_strategy/pe_agent_논문가능성.md`
> - 챕터 구조 + Paper 아웃라인: `docs/papers/00_strategy/research_toc.md`
> - Golden Set + Baseline + 평가 규격: `docs/papers/10_common_protocol/paper_common_protocol.md`

## 1) 목적과 전제
- 목적: 반도체 Fab 폐쇄망 환경의 RAG 서비스를 박사 수준 연구로 승격하고, 학회 논문(A/B/C) + Q1 저널 1편으로 연결한다.
- 전제: 제안서 자체를 논문화하는 것이 아니라, 제안서 내 연구 주제를 **측정 가능 지표 + 방법론 + 실험 프로토콜**로 재구성한다.
- 핵심 방향: 정확도 단일 지표가 아니라, 신뢰성(reliability) 관점(스코프 안전성, 안정성, 정합성, 운영통제)을 최적화한다.

## 2) 박사 스파인(큰 줄기)
다음 4축을 하나의 통합 프레임으로 본다.

1. 스코프 안전성: 타 장비군 문서 혼입(Contamination) 최소화
2. 검색 안정성: 동일/동의 질의에서 Top-k와 결론 변동 최소화
3. 정합성: 수치/조건 정보의 근거 일치성 보장
4. 운영 통제: 업데이트/드리프트/롤백 정책 최적화

권장 박사 한 줄 정의:

"Fab 폐쇄망 유지보수 RAG에서 스코프 오류, 검색 변동성, 정합성 위반을 운영 리스크로 정식화하고 이를 최소화하는 제약/정규화/검증/업데이트 통제 정책을 제안한다."

## 3) 논문 포트폴리오 설계

### Paper A (검색 스코프 안전성)
- 주제: Hierarchy-Constrained Retrieval
- 질문: 장비 hierarchy 제약으로 혼입을 줄이면서 Recall을 유지할 수 있는가?
- 핵심 지표: `Recall@k`, `Contamination@k`, `p95 latency`

### Paper B (검색 안정성)
- 주제: Stability-Aware Retrieval
- 질문: 동의 질의/반복 질의에서 Top-k/결론 흔들림을 줄일 수 있는가?
- 핵심 지표: `Jaccard@k`, `Top-k exact match`, `결론 일치율`, `Recall@k`

### Paper C (운영 라이프사이클 통제)
- 주제: Lifecycle Reliability Control
- 질문: 업데이트/드리프트/롤백 정책이 운영 리스크를 줄이는가?
- 핵심 지표: 정책별 리스크-비용, 회귀 탐지율, 롤백 성능, 안정성 변화

### Option Paper D (정합성)
- 주제: Numeric/Condition Faithfulness
- 질문: 수치/조건 검증기가 위반률을 유의미하게 줄이는가?

### Q1 저널 통합 전략
- 졸업 요건(Q1 1편) 대응을 위해 Paper A/B 결과를 컴포넌트 기여로 사용하고,
  Paper C의 운영 통제(리스크-비용 최적화)를 메인 기여로 확장해 통합 저널 원고를 구성한다.

## 4) 선행 필수 자산(논문 작성 전 완료)

- [ ] `Corpus v1` (문서/패시지/메타/버전)
- [ ] `Scope taxonomy` (장비군-설비군-모듈-부품 매핑)
- [ ] `Golden set v1` (질의/근거/스코프/동의질의/수치조건)
- [ ] `Baseline suite` (BM25, Dense, Hybrid, Hybrid+Rerank)
- [x] `Evaluation harness` (지표 자동 산출 + 실험 로그) — `scripts/evaluation/retrieval_stability_audit.py` 구현 완료
- [ ] `Versioned regression suite` (v1/v2/v3 비교 가능한 고정 질의군)

## 5) Golden Set 구체 설계

### 5.1 데이터 소스(S1~S5)
1. S1: 매뉴얼/SOP/PM 체크리스트
2. S2: 정비 이력/트러블 티켓
3. S3: 연구소-Fab Q&A 스레드
4. S4: 엔지니어 인터뷰(고통점/고위험 시나리오)
5. S5: 운영 로그(재질의/실패 패턴)

### 5.2 권장 스키마(JSONL)

> 정식 스키마 정의: `docs/papers/10_common_protocol/paper_common_protocol.md` §1

```json
{
  "qid": "GS-001",
  "category": "troubleshooting | specification | procedure | safety",
  "canonical_query": "RF 매칭 불안정 시 점검 순서",
  "paraphrases": ["...", "..."],
  "allowed_scope": {
    "equipment_group": "ETCH_A",
    "module": ["RF_GEN", "MATCHING"],
    "doc_type": ["SOP", "Troubleshooting Guide"]
  },
  "gold_evidence": [
    {"doc_id": "MNL_001", "version": "v7", "passage_id": "MNL_001#p034", "relevance": "primary"}
  ],
  "gold_answer": {
    "summary": "기대 정답 요약",
    "numeric_constraints": [
      {"field": "pressure", "value": 30, "unit": "mTorr", "tolerance": 5}
    ]
  },
  "tags": ["confusable", "high_risk", "update_sensitive"],
  "paper_subset": ["A:contamination", "B:paraphrase"],
  "difficulty": "medium"
}
```

### 5.3 버전 전략
- `v0` (빠른 파일럿): 150~250 canonical queries
- `v1` (학회 제출급): 400~800 canonical queries
- `v2/v3` (운영통제/Q1 저널급): 업데이트 이벤트 반영

### 5.4 논문별 서브세트 분리
- A용: `Contamination Challenge Set` (혼입 유발 질의)
- B용: `Paraphrase Stability Set` (동의질의 그룹)
- C용: `Regression + Update Set` (버전 변화 추적)

### 5.5 라벨링/QA 원칙
- 근거 라벨 단위는 문서보다 `passage_id` 우선
- 전체의 15~20% 이중 라벨링 + adjudication
- paraphrase는 그룹 단위 분할로 누수 방지
- baseline 사전 실행으로 데이터 결함(문서 누락/패시지 분할 오류/과도한 모호성) 제거

### 5.6 Split/Freeze 거버넌스 (필수)
- 분할 원칙:
  - `train/dev/test`는 `paraphrase_group_id` 단위로 분할한다(그룹 분산 금지).
  - 동일 `canonical_query`에서 파생된 변형 질의는 반드시 동일 split에 배치한다.
- 동결 원칙:
  - 논문별 실험 시작 전 `golden_set_vX`를 동결하고, 결과 표/그림에는 버전 해시를 명시한다.
  - 동결 후 라벨 수정이 필요하면 `vX+1`로만 반영하고, 기존 버전은 재기록하지 않는다.
- 업데이트 실험(C용) 원칙:
  - `test` 질의군은 고정하고, 비교 대상은 코퍼스/인덱스 버전(`corpus_v1 -> v2 -> v3`)만 변경한다.
  - 운영 회귀용 질의군(`regression suite`)은 별도 고정 파일로 관리한다.

## 6) Baseline + 평가 하네스 준비 Task

### 6.1 Baseline 고정
- BM25
- Dense Retriever
- Hybrid
- Hybrid + Rerank

### 6.2 공통 지표 자동화
- 정확도: `Recall@5`, `Recall@10`, `MRR`
- 안전성: `Contamination@k`
- 안정성: `Jaccard@k`, `Top-k exact match`, 결론 일치율
- 정합성: `Numeric/Condition violation rate`
- 운영성: `p95 latency`

### 6.3 재현성 로그
- 데이터/인덱스 식별:
  - `dataset_version`, `dataset_hash`
  - `corpus_version`, `corpus_hash`
  - `index_alias`, `index_build_id`, `index_mapping_hash`
- 모델/프롬프트 식별:
  - `llm_model_id`, `embedding_model_id`, `reranker_model_id`
  - `prompt_spec_version`
- 실험 설정:
  - `retriever/reranker config`, `top_k`, `rerank_enabled`
  - `seed`, ANN 관련 파라미터, tie-break 정책
- 결과 아티팩트:
  - top-k 문서 ID/점수, scope 판단 결과
  - 질의별 실행시간 + `p95` 집계
- 실행 재현 정보:
  - `git_commit_sha`, `runtime_env`(OS/Python/라이브러리 버전)
  - `hardware_profile`(CPU/GPU/RAM)

### 6.4 통계 검증 프로토콜 (유의성 기준 명시)
- 기본 보고 형식:
  - 모든 핵심 지표는 `95% CI`와 함께 보고한다.
  - 질의 단위 paired 비교를 기본으로 한다(동일 질의군에서 baseline vs proposed).
- 권장 검정:
  - 비모수: `paired bootstrap`(10,000 resamples) 또는 `Wilcoxon signed-rank`
  - 비율 지표(예: contamination rate): `McNemar` 또는 bootstrap CI
- 판정 기준(기본):
  - `p < 0.05` 또는 `95% CI`가 0(차이 없음)을 교차하지 않을 때 유의 개선으로 판정
  - 다중 비교 시 Holm-Bonferroni 보정 적용
- 최소 표본 기준:
  - Paper A/B 핵심 비교는 `canonical query >= 200` 권장
  - Paper C 정책 비교는 `regression queries >= 50` + 업데이트 시나리오 3종 이상 권장

## 7) 단계별 실행 로드맵

> 상세 Stage 설계: `docs/papers/00_strategy/research_toc.md` §4

### Stage 0: 연구 프레임 고정 ← **진행중**
- [x] RQ/Hypothesis/지표/리스크 함수 확정
- [x] 시스템 안정화 (mq_mode, guardrails, deterministic protocol)
- [ ] Golden set v0 pilot 라벨링 (50건)

### Stage 1: 공통 자산 구축
- corpus + taxonomy + golden v0 + baseline + eval harness

### Stage 2: Paper B (검색 안정성) ← **가장 진행도 높음**
- [x] 합성 벤치마크 구축 (60 groups × 4 = 240 queries)
- [x] Deterministic protocol 구현 + T1/T3 = 1.0 달성
- [x] Eval runner 구현 (`retrieval_stability_audit.py`)
- [x] 논문 초고 작성 (`paper_b_stability.md`)
- [ ] T2 Paraphrase 실험 확장 (n=20 반복)
- [ ] Driver analysis 정량화 + 최종 투고

### Stage 3: Paper A (스코프 안전성)
- hierarchy 제약(hard/soft) + contamination challenge 실험

### Stage 4: Paper C (운영 라이프사이클 통제)
- update/drift/rollback 정책 실험 + 비용-리스크 분석

### Stage 5: Option Paper D (정합성)
- 수치/조건 검증기 및 품질-지연 트레이드오프

### Stage 6: Q1 저널 통합
- A/B/C 통합 프레임 + 현장 KPI 연결 + 운영 가이드라인 제시

## 8) 즉시 실행할 우선순위 Top 10

1. [ ] Golden set 스키마 v0 동결
2. [ ] 후보 질의 풀 대량 추출(S1~S3)
3. [ ] scope taxonomy 최종 확정
4. [ ] v0(약 200문항) 샘플링
5. [ ] v0 라벨링(근거/스코프/기본 paraphrase)
6. [ ] baseline 4종 1차 실행
7. [ ] A용 contamination challenge 30문항 파일럿
8. [x] B용 paraphrase 60그룹 합성 벤치마크 구축 완료 (240 queries)
9. [ ] C용 업데이트 시나리오 1종(v1->v2) 파일럿
10. [ ] QA 회의 후 가이드 v1 업데이트

## 9) 완료(DoD) 기준

### Paper A 착수 DoD
- contamination challenge에서 baseline 혼입이 관측되고,
  제약 방법이 혼입 감소를 유의하게 보여야 함(`p<0.05` 또는 95% CI 기준).

### Paper B 착수 DoD
- paraphrase/repeat에서 baseline instability가 관측되고,
  안정성 개선 방법이 Jaccard/일치율을 유의하게 개선해야 함(`p<0.05` 또는 95% CI 기준).

### Paper C 착수 DoD
- v1->v2 업데이트 전후 지표 변화가 관측되고,
  정책 비교(Always/Scheduled/Drift-triggered)가 가능해야 함.
- 정책별 리스크-비용 비교에서 최소 1개 정책이 기준 정책 대비 유의한 개선을 보여야 함.

### Q1 저널 제출 DoD
- A/B/C 결과를 하나의 리스크-비용 프레임으로 통합
- 운영 KPI(탐색 시간/리드타임/재작업 등)와 연구 지표 연결
- 재현 가능한 평가 프로토콜/버전 로그 제시
- 통계 검증 결과(검정 방법, CI, p값, 다중비교 보정)를 부록 또는 본문에 포함

## 10) 레포 내 재사용 자산 (바로 활용)

### 연구 전략 문서
- 가능성 진단 + 리스크: `docs/papers/00_strategy/pe_agent_논문가능성.md`
- 챕터 구조 + Paper 아웃라인: `docs/papers/00_strategy/research_toc.md`
- Golden Set + Baseline + 평가 규격: `docs/papers/10_common_protocol/paper_common_protocol.md`

### 골든셋/평가
- 골든셋 전략 문서: `docs/2026-01-07_retrieval_golden_set_strategy.md`
- 골든셋 스크립트 가이드: `docs/2026-01-08_golden_set_pooling_script.md`
- 골든셋 스크립트: `scripts/golden_set/create_pools.py`
- 변환/중복제거: `scripts/golden_set/convert_questions.py`, `scripts/golden_set/deduplication.py`
- 평가/ablation 패턴: `scripts/evaluation/ablation_configs.py`, `scripts/evaluation/retrieval_parity_check.py`
- 안정성 평가: `scripts/evaluation/retrieval_stability_audit.py`

### Paper B
- 스펙 (decision-complete): `docs/papers/20_paper_b_stability/paper_b_stability_spec.md`
- 논문 초고: `docs/papers/20_paper_b_stability/paper_b_stability.md`
- 합성 벤치마크: `data/synth_benchmarks/stability_bench_v1/`
- 생성 runbook: `docs/papers/20_paper_b_stability/paper_b_synth_runbook.md`

## 11) A/B/C 원고 작성 실행 템플릿 (바로 집필용)

각 논문은 아래 8개 섹션을 동일 템플릿으로 작성하고, 섹션별 완료 기준을 체크한다.

1. **Problem & Gap**
   - 이 논문이 줄이려는 실패 모드 1개를 명시
   - 실패 비용(운영 리스크) 2-3줄로 연결
2. **Research Question / Hypothesis**
   - RQ 1개 + H 2개를 수식/판정 기준과 함께 명시
3. **Method**
   - 알고리즘 도식 1개 + 핵심 수식 1개 + pseudo-code 1개
4. **Benchmark & Split**
   - 사용 세트(vX), split 규칙, freeze hash 기재
5. **Baselines & Ablations**
   - B0-B3 + 제안법 + ablation(최소 2개)
6. **Results (Main Table + Trade-off Figure)**
   - 메인 표 1개, 보조 표 1개, trade-off 그림 1개
7. **Failure Analysis**
   - 실패 케이스 3유형 이상 + 원인 분해 표
8. **Threats to Validity / Reproducibility**
   - 내부/외부 타당성, 재현 조건, 한계 명시

### A/B/C 섹션별 핵심 차별화
- **Paper A**: Contamination@k 중심. 스코프 오류 감소와 Recall 손실 제한을 동시에 증명.
- **Paper B**: Stability 중심. Repeat/Paraphrase 안정성과 Recall trade-off를 증명.
- **Paper C**: Lifecycle 중심. 업데이트 정책의 리스크-비용 우위를 증명.

### 원고 착수 체크리스트
- [ ] 제목/초록/기여 3줄이 실패 모드 하나로 일관됨
- [ ] 메인 지표 1개 + 보조 지표 2개가 명확함
- [ ] baseline/ablation 표 스켈레톤이 먼저 채워져 있음
- [ ] 재현성 블록(버전/해시/실행환경) 문단 포함

## 12) Baseline 구축 프로세스 (실행 순서 + Gate)

자세한 규격은 `docs/papers/10_common_protocol/paper_common_protocol.md`를 따르고,
아래 순서로 실행하면 A/B/C 공통 baseline이 즉시 재사용 가능하다.

### Step 0. 실험 스냅샷 고정
- [ ] `golden_set_version`, `golden_set_hash`, `index_alias`, `git_commit_sha` 기록
- [ ] 실험 설정 템플릿(상수: k, repeats, deterministic, rerank) 고정

### Step 1. Baseline B0-B3 실행
- [ ] B0: BM25
- [ ] B1: Dense
- [ ] B2: Hybrid
- [ ] B3: Hybrid + Rerank

### Step 2. 공통 지표 계산
- [ ] Recall@5/10, hit@k, MRR
- [ ] Paper별 지표(A: Contamination, B: Stability, C: Drift/ARL)
- [ ] p95 latency

### Step 3. 통계 검증
- [ ] paired bootstrap(10,000) 또는 Wilcoxon
- [ ] 95% CI + p-value + 다중비교 보정(Holm)

### Step 4. Baseline Self-check (데이터 결함 탐지)
- [ ] easy/medium 질의에서 hit@10 = 0인 항목 분류
- [ ] 원인 태깅: 라벨 문제 / 문서 누락 / 분할 문제 / 검색 실패
- [ ] 데이터 결함 수정은 `golden_set_vX+1`로만 반영

### Step 5. 아티팩트 저장
- [ ] `results.jsonl`(질의 단위)
- [ ] `metrics.json`(집계)
- [ ] `main_table.csv`(원고 표 바로 사용)

### Baseline 완료 Gate (공통)
- [ ] B0-B3 전체가 동일 설정/동일 세트에서 재실행 가능
- [ ] 결과 파일 3종(results/metrics/table)이 모두 생성됨
- [ ] 메인 지표에 95% CI가 포함됨
- [ ] paper_subset(A/B/C)별 비교 표가 분리됨

---

이 문서는 실행 체크리스트 역할로, 상세 설계는 위 관련 문서를 참조한다.
제출 순서: **Paper B → Paper A → Paper C → Q1 저널 J** (진행도 기준 현실 반영).
