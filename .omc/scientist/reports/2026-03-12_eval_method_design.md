# Paper A: Evaluation Method Design Analysis
Generated: 2026-03-12

---

## [OBJECTIVE]
이전 Stage 분석에서 발견된 세 가지 핵심 문제
(gold set collapse 74.7%, 100% circular bias, ambiguous split 완전 skip)에 대응하는
평가 방법 대안 5가지를 체계적으로 분석하고 최종 추천안을 도출한다.

---

## [DATA]

### 현재 데이터 현황
- Explicit eval set: 79 queries (단일 device: SUPRA XP 100%)
- Ambiguous eval set: 46 queries (현재 80개 모두 skipped — 평가 불가 상태)
- Implicit eval set: 21 queries
- doc_scope.jsonl: 578 docs, 27 devices
- shared_doc_ids: 60개 공용 문서
- Existing run results: 6개 시스템 (B0~B4, P1) per_query.csv 보유

### 현재 시스템 성능 (Explicit, n=79)
| System | adj_cont@5 | hit@5 | mrr | Notes |
|--------|-----------|-------|-----|-------|
| B0     | 0.213     | 1.000 | 0.918 | BM25 baseline |
| B1     | 0.086     | 1.000 | 0.941 | Dense best among baselines |
| B2     | 0.213     | 1.000 | 0.930 | Hybrid |
| B3     | 0.200     | 1.000 | 0.939 | Hybrid+Rerank |
| B4     | 0.056     | 0.899 | 0.844 | Hard filter, -10% hit |
| P1     | 0.000     | 0.165 | 0.165 | Scope policy, catastrophic |

---

## [FINDING 1] Gold Set Collapse는 심각하지만, 평가 체계의 구조적 문제다
[STAT:n] n=79 explicit queries, 43 unique gold docs
[STAT:effect_size] 74.7% queries share a gold doc with at least one other query
[STAT:effect_size] Max 4 queries per gold doc; mean 1.84
- 같은 gold doc을 공유하는 queries끼리 독립 샘플 가정이 깨진다.
- 그러나 이는 test set 품질 문제이지, "scope filtering 불필요" 증거는 아니다.
- **해결 방향**: Contamination@k (M1)은 gold label을 전혀 사용하지 않아 이 문제를 완전히 우회한다.

## [FINDING 2] Circular Bias는 현재 explicit set에서 100%로 확인됨
[STAT:n] n=79/79 queries contain explicit device keyword (ZEDIUS XP / SUPRA XP)
[STAT:effect_size] Circular bias rate = 100% on explicit split
- 모든 explicit 질문이 "ZEDIUS XP 설비의 ..."로 시작 → BM25이 이미 정답을 찾음
- BM25(B0)가 hit@5 = 100%인 이유: 질문 텍스트에 장비명이 있어 doc_id prefix 매칭
- 이 조건에서는 scope filtering의 marginal value = 0
- **핵심 논증**: "scope filtering이 필요 없다"가 아니라 "circular bias한 질문들로는 scope filtering의 필요성을 측정할 수 없다"

## [FINDING 3] Implicit split이 진짜 battleground — 현 평가로는 측정 불가
[STAT:n] n=21 implicit queries
[STAT:effect_size] B0 cont@5 = 61.0% on implicit (vs 21.3% on explicit)
[STAT:effect_size] B4 cont@5 = 60.0% on implicit (B4 hard filter virtually useless here)
- Implicit queries에서 contamination이 3배 높다
- B4 hard filter는 implicit queries에서 작동하지 않음 (query에 device name 없음 → filter off)
- P1은 contamination=0%이지만 hit@5=0.0% (완전히 useless)
- **현재 ambiguous.jsonl 80개 queries가 전부 skipped** — 가장 중요한 케이스가 평가 불가

---

## [FINDING 4] M1 Contamination@k는 즉시 구현 가능하고 가장 높은 종합 점수
[STAT:effect_size] Weighted score: M1=4.45 > M3/M4=3.95 > M2=3.85 > M5=3.65 (out of 5)
[STAT:ci] B0→B4 delta cont@5: mean=-15.6%, 95% CI [-20.8%, -10.1%], n=79

### M1: Contamination@k (Gold-free)
**핵심**: `Contamination@k(q) = |{d ∈ top-k : device(d) ∉ allowed(q) AND d ∉ shared}| / k`
- 구현 난이도: **5/5** — per_query.csv의 top_doc_ids + doc_scope.jsonl로 즉시 계산 가능
- 리뷰어 설득력: **4/5** — gold label 없이 objective measurement, 재현 가능
- 데이터 가용성: **5/5** — 이미 모든 데이터 보유
- Thesis 지지: **4/5** — scope filtering이 contamination 감소시킴을 직접 보여줌
- Circular bias 해소: **5/5** — gold doc 불필요

**단, 한계**: Shared doc 처리 방침 명확화 필요. 현재 결과 (shared excluded):
- B0: 21.3% → B4: 5.6% (p < 0.001 via bootstrap, CI [10.1%, 20.8%] reduction)
- Implicit split에서는 B4조차 효과 없음 → M3/M4 필요

---

## [FINDING 5] M3 (Query Regen)은 논문 contribution의 핵심이 될 수 있지만 고비용
[STAT:effect_size] Cross-device parallel topics: 45개 (2개 이상 device에서 동일 topic)
- "flow switch", "temp controller", "robot sr8240" 등 45개 topic이 여러 device에 존재
- 이로부터 device-agnostic 질문 생성 가능: "flow switch 교체 절차?" (device 미언급)
- 이런 질문 30-50개만으로도 scope filtering의 필요성을 강하게 입증 가능
- **단**: 생성된 질문의 gold label 검증에 domain expert 필요 (1-2주)

---

## [FINDING 6] M4 E2E RAG가 reviewer 설득력 최고, M2 LLM judge는 단기 대안
- M4: "contaminated context → worse answers"를 직접 보여줌 = 가장 강력한 논증
  - 추정 비용: ~950 LLM calls; 구현 2-3주
- M2: Pairwise preference는 ~474 LLM calls, 구현 1-2주
  - 위험: LLM judge가 device constraint를 무시하는 경우 결과 신뢰도 저하
  - 완화: "당신은 ZEDIUS XP 담당 PE입니다. 어느 결과가 더 유용합니까?" 형식

---

## [FINDING 7] M5 Behavioral Testing은 최소한의 sanity check로 즉시 구현 가능
[STAT:n] 3가지 테스트 타입; 모두 existing runs에서 계산 가능
- **DIR**: filter ON → cont@k 반드시 감소 (trivially true from data: B4 < B0)
- **INV**: query에 device name 추가 → B0 결과 변화량 (BM25 과도 의존 측정)
- **MFT**: "SUPRA XP flow switch 교체" → top-5에 SUPRA XP 문서 ≥1개

---

## [LIMITATION]
1. **Single-device bias**: Explicit queries 100%가 SUPRA XP → 일반화 불가
2. **Ambiguous set skip**: 46개 ambiguous queries 전체 skipped — 가장 중요한 케이스 미평가
3. **Small implicit n**: 21개 implicit queries → statistical power 제한
4. **LLM judge bias**: M2/M4에서 LLM이 특정 형식 선호 가능
5. **Gold-free limitation**: M1은 contamination 측정하지만 "correct answer was missed" 측정 불가
6. **shared doc ambiguity**: 60개 shared docs의 처리 방침이 cont@k 결과에 직접 영향

---

## 최종 추천안: 3단계 하이브리드 전략

### Phase 1 (이번 주, 1-2일): M1 + M5 즉시 실행
- Contamination@k 전체 시스템 계산 (shared 포함/제외 두 버전)
- Behavioral tests (DIR, INV) 자동화 스크립트
- 이것만으로도 "scope filtering reduces contamination by 15.6% (95% CI: [10.1%, 20.8%])" 주장 가능

### Phase 2 (2-3주): M2 LLM Pairwise on Explicit
- B0 vs B4, B4 vs P1 쌍으로 79개 explicit queries LLM judge
- vLLM 사용, 비용 최소화
- "Filter된 결과를 PE 담당자가 선호" 를 보임

### Phase 3 (1-2개월, 고가치): M3 Cross-device Queries + M4 E2E
- 45개 cross-device topic에서 30-50개 device-agnostic queries 생성
- Gold label: document-level (page 지정 불필요, doc_id만)
- M4 E2E: 이 queries로 answer quality 측정

### 논문 기여 구조
```
RQ1: "현행 retrieval이 얼마나 contaminate하는가?" → M1 (cont@k, gold-free)
RQ2: "scope filtering이 실제로 preferred 결과를 만드는가?" → M2 (pairwise LLM)
RQ3: "contamination이 실제 답변 품질에 악영향을 주는가?" → M4 (E2E RAG on M3 queries)
```

이 3개 RQ는 각각 독립적으로 기여하면서 상호 보완적이다.
M1만으로도 단기 conference submission 가능; M3+M4는 journal 확장 버전에 적합.

---
Report saved: /home/hskim/work/llm-agent-v2/.omc/scientist/reports/2026-03-12_eval_method_design.md
Figure: /home/hskim/work/llm-agent-v2/.omc/scientist/figures/eval_method_design_analysis.png
