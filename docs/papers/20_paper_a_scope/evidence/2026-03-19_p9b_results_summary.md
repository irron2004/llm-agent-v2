# P9b Margin-Gated Verifier 실험 결과

**작성일**: 2026-03-19
**알고리즘**: P7+ top-1 device + margin-gated verification
**데이터**: `data/paper_a/p9b_results.json`

---

## 1. 알고리즘

```
1. P7+ device mass ranking 계산 (rank-decay weighted)
2. margin = top-1 score - top-2 score
3. if margin > threshold:
     → top-1 직접 사용 (P9a와 동일)
   else:
     → top-M 후보에 대해 hard retrieval 수행
     → evidence score (mean_top3) 기준으로 최종 device 선택
4. 선택된 device의 결과 반환
```

---

## 2. 전체 결과 (n=578)

| Condition | cont@10 | gold_strict | MRR | scope_acc | verify_rate |
|-----------|---------|-------------|-----|-----------|-------------|
| B3_masked | 0.584 | 351 (60.7%) | 0.335 | — | — |
| B4_masked | 0.001 | 527 (91.2%) | 0.562 | — | — |
| P7plus_masked | 0.515 | 506 (87.5%) | 0.521 | — | — |
| **P9a_masked** | **0.048** | **492 (85.1%)** | **0.618** | **0.934** | **0%** |
| P9b_m03 | 0.056 | 488 (84.4%) | 0.612 | 0.950 | 5.1% |
| P9b_m05 | 0.077 | 476 (82.4%) | 0.595 | 0.929 | 11.2% |
| P9b_m07 | 0.098 | 466 (80.6%) | 0.581 | 0.908 | 16.0% |
| P9b_m03_v3 | 0.061 | 485 (83.9%) | 0.607 | 0.945 | 5.1% |
| P9b_m05_v3 | 0.087 | 470 (81.3%) | 0.585 | 0.918 | 11.2% |

### Delta from P9a

| Condition | Δgold_strict | Δcont@10 | Δscope_acc |
|-----------|-------------|----------|-----------|
| P9b_m03 | **-4** | +0.008 | +0.016 |
| P9b_m05 | **-16** | +0.029 | -0.005 |
| P9b_m07 | **-26** | +0.050 | -0.026 |
| P9b_m03_v3 | **-7** | +0.013 | +0.011 |
| P9b_m05_v3 | **-22** | +0.039 | -0.016 |

**모든 P9b 조건이 P9a보다 나쁨.** verification이 더 많이 trigger될수록 성능이 더 떨어짐.

---

## 3. explicit_equip (n=149)

| Condition | gold_strict | MRR | scope_acc |
|-----------|-------------|-----|-----------|
| P9a_masked | 112 (75.2%) | 0.448 | — |
| P9b_m03 | 112 (75.2%) | 0.444 | 1.000 |
| P9b_m05 | 110 (73.8%) | 0.436 | 0.986 |

explicit_equip에서는 verification이 거의 trigger 안 됨 (0.7~2.8%).
→ P7+ device mass의 margin이 equip 쿼리에서 충분히 큼.

---

## 4. 실패 원인 분석

### Stage 3 역설 재확인

P9a의 scope_acc = 93.4%. P9b verification의 역할:
- **올바른 top-1을 뒤집을 확률** vs **잘못된 top-1을 교정할 확률**

verification이 trigger되는 29건(m03 기준) 중:
- P9a에서 이미 맞는 것을 P9b가 뒤집는 경우가 더 많음
- evidence score(mean_top3)가 문서 볼륨이 큰 장비에 유리 → 같은 bias

### 왜 verification이 해로운가

1. **margin이 작다 ≠ top-1이 틀리다**: margin이 작아도 top-1이 맞을 수 있음
2. **evidence score bias**: 문서가 많은 장비의 retrieval 결과가 양적으로 풍부 → evidence score가 높음
3. **P9a의 93.4%는 이미 높은 정확도**: 뒤집을 여지보다 뒤집혀서 손해볼 여지가 더 큼

### 수치적 확인

P9a: 540/578 correct, 38 miss
P9b_m03: 29건 verification → 그 중 net effect = -4

→ verification 29건 중 약 12~13건은 올바르게 교정, 16~17건은 잘못 뒤집음
→ **net negative: 교정 < 오판**

---

## 5. 결론

### P9b는 negative result

1. **Margin-gated verification은 P9a 대비 성능을 떨어뜨림**
2. 가장 보수적인 threshold(0.3)에서도 -4건, 가장 공격적인(0.7)에서는 -26건
3. evidence score 기반 Stage 3 selector는 문서 볼륨 bias를 극복하지 못함
4. **P9a(top-1 무조건 신뢰)가 현재 데이터에서 최적 전략**

### 논문 메시지

- P8의 Stage 3 실패가 P9b에서도 재현됨
- **"strong proposal + no verification"이 "moderate proposal + verification"보다 우월**
- 이는 proposal accuracy가 충분히 높을 때(>90%) 일반적으로 성립하는 패턴
- verification이 유효하려면 evidence score가 scope-agnostic해야 함 (현재는 그렇지 않음)

### 후속 방향

P9b 실패로 인해 scope miss 복구 전략 재검토:
1. ~~margin-gated verifier~~ → 실패 확인
2. **TF-IDF device profile**: P7+ device mass에 lexical prior 추가 (Stage 1 강화)
   - verification(Stage 3)이 아니라 proposal(Stage 1) 자체를 개선
   - P7+ top-5에도 없는 22건 해결 가능성
3. **Hybrid proposal**: P7+ mass + TF-IDF mass 결합
