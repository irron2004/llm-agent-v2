# P9a 실험 결과 요약

**작성일**: 2026-03-19
**실험**: P9a — Proposal-Only Hard Scope Retrieval
**알고리즘**: P7+ device mass top-1 → hard filter hybrid+rerank retrieval
**데이터**: `data/paper_a/p9a_results.json`

---

## 1. 알고리즘 요약

```
Stage 1: Device proposal
  - P7+ top-10 결과에서 rank-decay 가중 device mass 계산
  - w(rank) = 1 / log2(rank + 2)
  - top-1 device 선택 (Stage 3 검증 없음)

Stage 2: Hard filter retrieval
  - 선택된 device의 문서만 대상으로 hybrid+rerank 수행
  - BM25 filter: {"terms": {"doc_id": device_doc_ids}}
  - Dense filter: {"terms": {"device_name": device_name_variants}}
  - RRF fusion (chunk_id level) → cross-encoder reranking → doc dedup
```

파라미터: 없음 (P7+ top-1을 그대로 사용)

---

## 2. 전체 결과 (n=578)

| Condition | cont@10 | gold_strict | gold_loose | MRR |
|-----------|---------|-------------|------------|-----|
| B3_masked | 0.584 | 351/578 (60.7%) | 380/578 (65.7%) | 0.335 |
| B4_masked (oracle) | 0.001 | 527/578 (91.2%) | 532/578 (92.0%) | 0.562 |
| B4.5_masked | 0.001 | 406/578 (70.2%) | 439/578 (75.9%) | 0.421 |
| P7plus_masked | 0.515 | 506/578 (87.5%) | 518/578 (89.6%) | 0.521 |
| P8_masked | 0.641 | 218/578 (37.7%) | 351/578 (60.7%) | 0.234 |
| **P9a_masked** | **0.048** | **492/578 (85.1%)** | **509/578 (88.1%)** | **0.618** |
| P9a_sc1_masked | 0.048 | 496/578 (85.8%) | 519/578 (89.8%) | 0.580 |
| P9a_sc2_masked | 0.071 | 496/578 (85.8%) | 517/578 (89.4%) | 0.542 |

### Delta vs B3_masked

| Condition | Δcont@10 | Δstrict |
|-----------|----------|---------|
| B4_masked | -0.584 | +0.304 |
| P7plus_masked | -0.069 | +0.268 |
| **P9a_masked** | **-0.536** | **+0.265** |
| P9a_sc1_masked | -0.536 | +0.272 |

---

## 3. explicit_device 결과 (n=429)

| Condition | cont@10 | gold_strict | MRR |
|-----------|---------|-------------|-----|
| B3_masked | 0.481 | 341/429 (79.5%) | 0.442 |
| B4_masked | 0.001 | 416/429 (97.0%) | 0.608 |
| P7plus_masked | 0.490 | 402/429 (93.7%) | 0.566 |
| **P9a_masked** | **0.065** | **380/429 (88.6%)** | **0.677** |

- gold_strict: B4의 97.0%에는 못 미치나, P7+(93.7%)에 근접
- MRR: 0.677로 B4(0.608)보다 높음 — 맞출 때 더 높은 순위에 배치
- contamination: 0.481 → 0.065로 급감

---

## 4. explicit_equip 결과 (n=149) — 핵심 구간

| Condition | cont@10 | gold_strict | gold_loose | MRR |
|-----------|---------|-------------|------------|-----|
| B3_masked | 0.881 | 10/149 (6.7%) | 23/149 (15.4%) | 0.027 |
| B4_masked | 0.000 | 111/149 (74.5%) | 116/149 (77.9%) | 0.432 |
| B4.5_masked | 0.000 | 84/149 (56.4%) | 98/149 (65.8%) | 0.275 |
| P7plus_masked | 0.589 | 104/149 (69.8%) | 116/149 (77.9%) | 0.392 |
| P8_masked | 0.819 | 20/149 (13.4%) | 78/149 (52.3%) | 0.103 |
| **P9a_masked** | **0.000** | **112/149 (75.2%)** | **116/149 (77.9%)** | **0.448** |

**P9a가 B4(oracle)를 초과:**
- gold_strict: 112 vs 111 (+1)
- MRR: 0.448 vs 0.432 (+0.016)
- contamination: 둘 다 0.000

이 구간에서 P9a ≥ B4는 P7+ proposal이 parser보다 equip→device 매핑을 더 잘 수행함을 의미.

---

## 5. Scope Selection Accuracy

| Condition | ALL | explicit_device | explicit_equip |
|-----------|-----|-----------------|----------------|
| P9a_masked | 93.4% | 92.1% | 97.3% |

- 전체 scope_acc = 93.4% (540/578 정확)
- explicit_equip에서 97.3% (145/149) — parser(0%)보다 압도적
- scope miss 38건이 P9a와 B4의 gap(35건) 대부분을 설명

---

## 6. shared_cap 효과

| shared_cap | gold_strict | cont@10 | MRR |
|------------|-------------|---------|-----|
| 0 (P9a) | 492 (85.1%) | 0.048 | 0.618 |
| 1 (P9a_sc1) | 496 (85.8%) | 0.048 | 0.580 |
| 2 (P9a_sc2) | 496 (85.8%) | 0.071 | 0.542 |

- shared_cap=1: gold_strict +4건, MRR 하락 (-0.038)
- shared_cap=2: contamination 증가 (0.048→0.071), MRR 추가 하락
- **shared_cap=0이 가장 균형 잡힌 결과** — shared 문서 추가는 순이득이 아님

---

## 7. P9a의 위치 (방법론 간 비교)

### contamination vs gold_strict 2D 맵

```
                     contamination
              0.0    0.2    0.4    0.6    0.8
gold_strict   |      |      |      |      |
  90% --------B4-----+------+------+------+--
              |      |      |      |      |
  85% --------+--P9a-+------+------P7+----+--
              |      |      |      |      |
  80% --------+------+------+------+------+--
              |      |      |      |      |
  60% --------+------+------+---B3-+------+--
              |      |      |      |      |
  40% --------+------+------+------P8-----+--
```

- B4: 최고 gold_strict (91.2%), 최저 contamination (0.001) — oracle upper bound
- **P9a: B4에 근접 (85.1%), contamination 거의 제거 (0.048)**
- P7+: gold_strict 높으나 (87.5%) contamination도 높음 (0.515)
- B3: baseline
- P8: 구조 실패 (retrieval 버그 + scope selection 실패)

---

## 8. 논문 메시지

### 핵심 기여

1. **Cross-equipment contamination은 candidate generation 문제** — post-hoc soft scoring(P6/P7)으로는 해결 불가
2. **Naive multi-hypothesis verification(P8)도 실패** — scope selector bias + retrieval 구조 문제
3. **Candidate-assisted top-1 scope proposal + hard retrieval(P9a)이 실용적 해법**
   - B4(oracle)의 93%에 도달
   - contamination 91.8% 감소 (0.584 → 0.048)
   - explicit_equip 구간에서 B4 초과
4. **Shared document 추가는 순이득이 아님** — B4.5 실패와 일관된 결론

### 한계 (논문에 명시)

- P9a의 scope proposal은 P7+ cached results에 의존 (end-to-end 독립 아님)
- scope_acc = 93.4% → 6.6% miss가 성능 ceiling
- independent proposal (TF-IDF device profile 등)은 future work

---

## 9. 후속 실험

| 우선순위 | 실험 | 목적 |
|---------|------|------|
| 1 | P8 retrieval 버그 수정 후 재실행 | P8의 공정한 평가 (chunk-level RRF) |
| 2 | P9b (margin-gated verifier) | scope miss 38건 중 일부 복구 가능성 |
| 3 | TF-IDF device profile proposal | P7+ 의존 없는 독립 proposal line |
| 4 | Per-query error analysis | scope miss 38건의 패턴 분석 |

---

## 10. 버그 수정 기록

P9a 초기 결과(gold_strict=26%)는 retrieval 구현 버그에 의한 것.
수정 내용은 `2026-03-19_p9a_retrieval_bug_report.md` 참조.

- **doc_id 레벨 RRF → chunk_id 레벨 RRF** (핵심)
- **dense search device_name 대소문자 불일치 수정** (보조)
- 수정 후: gold_strict 150→492, MRR 0.226→0.618
