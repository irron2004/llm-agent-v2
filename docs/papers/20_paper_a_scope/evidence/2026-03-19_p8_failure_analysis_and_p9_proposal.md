# P8 실패 분석 및 P9 제안

**작성일**: 2026-03-19
**상태**: P9-min 구현 준비
**근거**: P8 실험 결과 (`data/paper_a/p8_results.json`)

---

## 1. P8 실험 결과 요약

### 전체 (n=578)

| Condition | cont@10 | gold_strict | MRR |
|-----------|---------|-------------|-----|
| B3_masked | 0.584 | 351/578 (60.7%) | 0.335 |
| B3_live | 0.711 | 274/578 (47.4%) | 0.217 |
| B4_masked | 0.001 | 527/578 (91.2%) | 0.562 |
| B4.5_masked | 0.001 | 406/578 (70.2%) | 0.421 |
| P7plus_masked | 0.515 | 506/578 (87.5%) | 0.521 |
| **P8_masked** | **0.643** | **217/578 (37.5%)** | **0.234** |
| P8_sc1_masked | 0.574 | 230/578 (39.8%) | 0.236 |
| P8_sc2_masked | 0.605 | 241/578 (41.7%) | 0.237 |
| P8_m5_masked | 0.649 | 216/578 (37.4%) | 0.229 |

### P8 Scope Selection Accuracy

| Condition | scope_acc | target_in_hypotheses |
|-----------|-----------|---------------------|
| P8_masked (M=3) | 0.396 | 0.690 |
| P8_m5_masked (M=5) | 0.389 | 0.749 |

### scope_observability별

| Condition | explicit_device (429) scope_acc | explicit_equip (149) scope_acc |
|-----------|-------------------------------|-------------------------------|
| P8_masked | 0.471 | 0.181 |
| P8_m5_masked | 0.476 | 0.141 |

---

## 2. P8 실패의 구조적 원인

### P8 실패는 "튜닝 실패"가 아니라 "구조 실패"

핵심 수치:
- `target_in_hypotheses = 69%`
- `B4_masked gold_strict = 91.2%`

이론적 상한: `0.69 × 0.912 ≈ 0.629`
→ Stage 3가 완벽해도 strict hit는 62.9% 정도가 상한
→ 이미 B3(60.7%)를 거의 못 넘는 구조

실제 P8은 37.5%이므로:
1. **Stage 1이 크게 부족** — 정답 device가 hypothesis에 31% 누락
2. **Stage 3도 bias** — score_sum이 문서 많은 device에 유리

### 하지 말 것

- 광범위 파라미터 스윕
- P8 score-sum selector 위에 미세 tuning
- global soft boost/soft penalty 계열로 회귀

### 할 것

- P8의 Stage 1 후보 생성기 자체를 바꾸기
- P7+를 문서 랭커가 아니라 device proposal teacher로 재활용
- TF-IDF / log-odds / term profile은 Stage 1 전용으로 도입

---

## 3. P7+의 올바른 역할

P7+ 결과:
- contamination: 0.515 (높음 → 최종 랭킹으로 부적합)
- gold_strict: 87.5% (높음 → 정답 device 쪽 단서 끌어올리는 능력 강함)

**P7+의 올바른 역할: 최종 랭킹이 아니라 Stage 1 device hypothesis generator**

1. P7+로 top-N 문서를 가져온다
2. 문서 점수를 device별로 집계한다
3. 상위 device 3~5개를 hypothesis로 만든다
4. 각 device마다 hard filter retrieval을 돌린다
5. 최종은 단일 device scope에서만 뽑는다

→ P7+의 높은 recall은 살리고, contamination은 final hard scope로 막을 수 있다

**단, P7+는 B3/B4/B4.5 cached candidates에 의존하므로 end-to-end 주장이 약해짐.**
**→ P9-min에서는 probe top-40 device mass를 먼저 시도하고, P7+ mass는 ablation 비교군으로.**

---

## 4. P9: Proposal-Verified Hard Scope Retrieval

### 4.1 핵심 아이디어

P8의 실패를 두 단계에서 교정:
- **Stage 1**: device 후보를 더 잘 만드는 데 집중
  → hard evidence + probe device mass (+ TF-IDF/log-odds device profile)
- **Stage 2**: 각 후보 device마다 hard-filter retrieval
- **Stage 3**: score_sum 대신 query coverage + top-rank evidence + doc_type consistency
- **Stage 4**: 선택된 단일 device scope에서만 최종 결과 반환

> soft reweighting이 아니라, **stronger proposal + hard-scope verification** 알고리즘

### 4.2 Stage 1: Device Proposal

쿼리 q에 대해 후보 device 집합 H(q)를 생성.

```
S_prop(d|q) = λ_h · S_hard(d|q) + λ_p · S_probe(d|q) + λ_l · S_lex(d|q)
```

**Hard evidence** (S_hard):
- explicit device mention
- alias normalization (ZEDIUS XP → SUPRA_XP, SUPRA V → SUPRA_VPLUS)
- explicit equip_id → device mapping
- sticky context inheritance
- 존재 시: `M=1` 또는 `M=2`

**Probe-based device mass** (S_probe):
```
S_probe(d|q) = Σ_{x ∈ TopN_probe(q), dev(x)=d} w(rank(x))
w(r) = 1 / log2(r + 2)
```

**Lexical device prior** (S_lex) — P9+에서 추가:
```
S_lex(d|q) = Σ_{t∈q} tf(t,q) · log((P(t|d,c_q) + ε) / (P(t|bg,c_q) + ε))
```
- c_q: document-type group (procedure / history)
- device-specific vocabulary 강조, globally frequent terms 자동 감쇠

### 4.3 Stage 2: Per-Hypothesis Hard Retrieval

```
R_L(q; d) = Retrieve(q, D_{device=d}, L)
```

각 후보 device마다 독립적으로 hard-filtered hybrid+rerank 수행.
→ candidate ceiling 문제 없음 (P6/P7과의 핵심 차별점)

### 4.4 Stage 3: Evidence-Based Scope Verification

```
E(d;q) = α · MaxScore(R_L(q;d))
       + β · MeanTopK(R_L(q;d))
       + γ · Coverage_idf(q, R_L(q;d))
       + δ · DocTypeMatch(q, R_L(q;d))
       - ξ · SharedDominance(R_L(q;d))
```

- MaxScore: 해당 device 안에서 정말 강한 hit가 있는가
- MeanTopK: 상위 몇 개가 안정적으로 강한가
- Coverage_idf: 질문의 중요한 단어를 잘 덮는가
- DocTypeMatch: 질문 유형과 문서 유형이 맞는가
- SharedDominance: shared만 많이 뜨는 device는 패널티

**주의**: α,β,γ,δ,ξ 5개 파라미터. P9-min에서는 2-3개만 사용하여 과적합 방지.

최종 device 선택:
```
d* = argmax_{d ∈ H(q)} E(d;q)
```

### 4.5 Stage 4: Final Retrieval

```
R_k^final(q) = R_k(q; d*)
```

선택된 단일 device scope에서만 최종 결과 반환.
→ cross-equipment contamination은 construction에 의해 제어됨.

### 4.6 Selective Shared Gate

B4.5 실패 원인: shared_overload 85.7%.
P9은 unconditional shared 대신 selective shared gate 도입.

```
g_shared(q) = 1[Σ_{t∈q} idf(t) ≥ τ_shared]
```

활성화 시:
```
R_k^final(q) = TopK(R_k(q; d*) ∪ TopB(R^shared(q)))
```
B ≪ k, 예: B=1 또는 2.

---

## 5. 구현 관점: P9-min vs P9+

### P9-min (먼저 구현)

| Stage | 구성 |
|-------|------|
| Stage 1 | hard evidence (parser/alias/sticky) + probe top-40 device mass |
| Stage 2 | per-device hard retrieval |
| Stage 3 | max_score + mean_top3 (파라미터 2개만) |
| Stage 4 | single-scope final |

- TF-IDF profile 불필요
- 추가 사전 구축 없음
- P8과 직접 비교 가능

### P9+ (보강 버전)

| Stage | 추가 구성 |
|-------|----------|
| Stage 1 | + TF-IDF / BM25 device profile 또는 log-odds lexical prior |
| Stage 3 | + Coverage_idf + doc_type_match + shared_penalty |
| Stage 4 | + selective shared gate |

- implicit / cold-start / explicit_equip 보강

---

## 6. TF-IDF / 중요어휘 기반 device profile

### 정당성

- P8 실험 전: "과하다" → 맞았음
- P8 실험 후: **Stage 1 miss = 31%라는 실증 근거** → 이제 정당화됨

### 올바른 사용

- **좋은 사용**: device 후보 생성기, query → device posterior, hypothesis recall 향상
- **나쁜 사용**: global document ranking soft bonus, P6/P7류 post-hoc 문서 점수 교정

### 추천 profile 설계

device별 두 개의 pseudo-document:

**Procedure profile**: SOP title, manual title, TS title, chapter heading, topic, component/part name, alias

**History profile**: myservice/gcb frequent terms, alarm/action terms, entity names

### 구현 순서

1. **TF-IDF pseudo-document**: device별 profile text 생성 → query와 cosine/BM25
2. **Log-odds / NB**: background 대비 특징어 강조 → common term 자동 감쇠

---

## 7. 실험 체크리스트: P8 → P9 전환

### Phase 1. 진단 분해 (가장 먼저)

**A. Stage 1 hypothesis recall 측정**

비교 대상:
- parser/context only
- probe top-40 device mass only
- P7+ aggregated device mass
- TF-IDF device profile
- probe + TF-IDF 결합

측정: `target_in_hypotheses@1`, `@3`, `@5`
목표: `@3 ≥ 90%`, `@5 ≥ 95%`

**B. Stage 3 selector oracle 분해**

가정: 정답 device가 hypothesis 안에 있다고 가정
비교: score_sum vs max_score vs mean_top3 vs coverage_idf vs 조합
측정: `scope_correct_given_target_in_H`
목표: 기존 39.6%보다 크게 개선

### Phase 2. P9-min

구성: hard evidence + probe device mass + max/mean verification
비교: B3, B4, P7+, P8, P9-min

### Phase 3. P9+

Stage 1 강화: TF-IDF device profile 추가
비교: P9-min vs P9+TFIDF vs P9+log-odds
집중: explicit_equip / implicit 개선폭

### Phase 4. Shared gate

비교: no shared vs naive shared vs selective shared gate
측정: contamination, strict gold, shared hit ratio, device-specific doc displacement

---

## 8. 검토 의견

### Stage 3 파라미터 수 주의

E(d;q)에 α,β,γ,δ,ξ 5개 가중치. HVSR의 14개보다 적지만 578쿼리 기준으로 과적합 위험.
→ P9-min에서는 feature 2-3개만 사용. 효과 확인된 feature만 P9+에서 추가.

### P7+ device mass의 의존성

P7+는 B3/B4/B4.5 cached candidates 사용 → P9도 "B3-assisted" P8-03 이슈 재발.
→ P9-min에서 **probe top-40 device mass** 먼저 시도. P7+ mass는 ablation 비교군.

### 실질 이득 구간

explicit_device 429건은 parser/alias로 M=1 가능. P9의 추가 이득 거의 없음.
**실질 이득 구간은 explicit_equip 149건** → 실험 보고 시 서브그룹 결과가 핵심 지표.

### Phase 1이 P9 구현보다 먼저

hypothesis recall이 90% 이상 나오는 proposal 방식을 찾은 후에 P9 전체를 조립하는 순서가 맞음.

---

## 9. 논문 메시지 전환

### 기존 초안
- hard filter 좋음
- soft scoring 안 됨

### 바뀐 메시지
1. cross-equipment contamination은 **candidate generation-level failure**
2. post-hoc soft scoring은 구조적으로 한계가 있음 (P6/P7 negative result)
3. naive multi-hypothesis retrieval도 scope selection 정확도 부족으로 실패 (P8 negative result)
4. 이를 해결하기 위해 **proposal-verified hard scope retrieval (P9)** 제안
5. device proposal 품질이 성능의 핵심 병목이며, lexical/device-profile priors가 이를 보완
6. naive shared allowance는 성능을 해치며, **selective shared gate**가 필요

---

## 10. 우선순위

| 순서 | 작업 | 목적 |
|------|------|------|
| 1 | Phase 1-A: hypothesis recall 측정 | 어떤 proposal이 @3≥90% 달성하는지 확인 |
| 2 | Phase 1-B: Stage 3 oracle 분해 | Stage 1 vs Stage 3 기여 분리 |
| 3 | P9-min 구현 (probe mass + hard evidence) | 첫 end-to-end 결과 |
| 4 | TF-IDF profile (필요 시) | explicit_equip 보강 |
