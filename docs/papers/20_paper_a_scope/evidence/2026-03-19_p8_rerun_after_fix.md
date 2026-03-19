# P8 재실행 결과 (chunk-level RRF 버그 수정 후)

**작성일**: 2026-03-19
**수정 내용**: P9a에서 발견한 두 가지 버그를 P8에도 적용
**목적**: P8의 낮은 성능이 버그 때문인지, 구조적 문제인지 확인

---

## 1. 수정 사항

1. **`build_allowed_devices_map`**: ES aggregation으로 device_name variant 수집 (대소문자 불일치 수정)
2. **`retrieve_hybrid_rerank`**: reranking 후 doc_id dedup 추가 (chunk당 1개 → doc당 최고 chunk 유지)

참고: chunk_id level RRF는 이전 코드에서 이미 적용되어 있었음.

---

## 2. Before vs After 비교

### 전체 (n=578)

| Condition | 수정 전 cont | 수정 후 cont | Δcont | 수정 전 strict | 수정 후 strict | Δstrict | 수정 후 MRR |
|-----------|-------------|-------------|-------|---------------|---------------|---------|------------|
| P8_masked | 0.643 | **0.461** | -0.182 | 217 (37.5%) | **230 (39.8%)** | +13 | 0.288 |
| P8_sc1_masked | 0.574 | **0.450** | -0.124 | 230 (39.8%) | **243 (42.0%)** | +13 | 0.290 |
| P8_sc2_masked | 0.605 | **0.501** | -0.104 | 241 (41.7%) | **279 (48.3%)** | +38 | 0.305 |
| P8_m5_masked | 0.649 | **0.471** | -0.178 | 216 (37.4%) | **224 (38.8%)** | +8 | 0.279 |

### Scope Selection Accuracy

| Condition | 수정 전 | 수정 후 | target_in_hyp |
|-----------|---------|---------|---------------|
| P8_masked | 0.396 | **0.408** | 0.702 |
| P8_m5_masked | 0.389 | **0.391** | 0.744 |

---

## 3. 수정 후에도 P8은 B3보다 나쁨

| Condition | cont@10 | gold_strict | MRR | scope_acc |
|-----------|---------|-------------|-----|-----------|
| B3_masked | 0.584 | 351/578 (**60.7%**) | 0.335 | N/A |
| **P8_masked (fixed)** | **0.461** | 230/578 (39.8%) | 0.288 | 0.408 |
| P9a_masked | 0.048 | 492/578 (85.1%) | 0.618 | 0.934 |

- P8 gold_strict(39.8%) < B3(60.7%) — **수정 후에도 B3에 크게 못 미침**
- contamination만 0.643 → 0.461로 개선 (B3의 0.584보다 낮아짐)
- scope_acc 0.408: target_in_hyp=0.702인데 scope_acc=0.408 → **Stage 3 selector가 30%에서 오판**

---

## 4. explicit_equip 구간

| Condition | cont@10 | gold_strict | MRR |
|-----------|---------|-------------|-----|
| B3_masked | 0.881 | 10/149 (6.7%) | 0.027 |
| B4_masked | 0.000 | 111/149 (74.5%) | 0.432 |
| **P8_masked (fixed)** | **0.676** | **20/149 (13.4%)** | 0.102 |
| P9a_masked | 0.000 | 112/149 (75.2%) | 0.448 |

P8은 explicit_equip에서도 B3(6.7%)보다 조금 낫지만(13.4%), P9a(75.2%)와는 비교 불가.

---

## 5. 결론: 버그가 아니라 구조적 실패

### 수정 전 해석 vs 수정 후 해석

| 항목 | 수정 전 | 수정 후 | 변화 |
|------|---------|---------|------|
| gold_strict | 217 (37.5%) | 230 (39.8%) | +13 (+2.3%p) |
| scope_acc | 0.396 | 0.408 | +0.012 |
| contamination | 0.643 | 0.461 | -0.182 |

**버그 수정은 contamination 개선에만 기여하고, 핵심 문제(scope_acc, gold_strict)에는 미미한 영향.**

### P8 실패의 근본 원인 (변경 없음)

1. **Stage 1 hypothesis recall 부족**: target_in_hyp = 70.2% → 상한 ceiling이 낮음
2. **Stage 3 score_sum selector bias**: 문서 많은 장비에 유리 → scope_acc = 0.408
3. 이론적 상한: 0.702 × 0.912 ≈ 0.640 → 이미 B3(60.7%)를 겨우 넘는 수준

### P9a가 P8보다 우월한 이유

| 비교 항목 | P8 | P9a |
|-----------|-----|-----|
| Proposal source | B3 cached top-10 | P7+ top-10 |
| Hypothesis recall | 70.2% (M=3) | 93.4% (top-1) |
| Scope selection | Stage 3 score_sum (40.8%) | No Stage 3, trust top-1 (93.4%) |
| Verification overhead | Stage 3가 오판 유발 | 없음 |

→ **P7+ device mass top-1을 그대로 신뢰하는 것이 Stage 3 검증보다 훨씬 정확함.**

---

## 6. 논문 메시지

1. **P8 retrieval 버그를 수정해도 결론은 동일**: P8은 구조적으로 실패한 방법
2. chunk-level RRF + device_name variant 수정으로 contamination은 개선되었으나, scope selection accuracy의 근본 한계는 변하지 않음
3. **P8의 공정한 평가 완료**: 버그를 변명으로 남기지 않고 수정 후 재확인한 결과, 여전히 B3 이하
4. 이는 P9a(proposal-only hard scope)의 우월성을 더욱 강화하는 근거
