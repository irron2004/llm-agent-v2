# P9a Scope Miss 분석

**작성일**: 2026-03-19
**대상**: B4 vs P9a 교차 분석 + P9a scope miss 상세
**데이터**: `data/paper_a/p9c_results.json`, `data/paper_a/p9a_results.json`, `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl`

---

## 1. B4 vs P9a 교차 분석 (n=564)

| 구분 | 건수 | 비율 | 의미 |
|------|------|------|------|
| Both hit | 485 | 86.0% | 두 방법 모두 정답 |
| **B4 only (→ §2~6)** | **32** | **5.7%** | B4(oracle filter)가 잡지만 P9a는 miss |
| **P9a only (→ §7)** | **7** | **1.2%** | B4 못잡는데 P9a는 잡는 것 |
| **Both miss (→ §8)** | **40** | **7.1%** | 둘 다 못잡는 것 |

→ P9a는 B4 대비 32건 손해, 7건 이득. **net gap = -25건.**

---

## 2. 요약 — B4 hit & P9a miss (32건)

이 32건은 두 가지 유형으로 분류됨:

| 유형 | 건수 | 설명 |
|------|------|------|
| **진짜 scope miss** | 약 23건 | 잘못된 장비를 선택 |
| **retrieval miss** (올바른 장비, gold 누락) | 약 9건 | 올바른 장비를 선택했지만 gold 문서가 top-10에서 누락 |

→ **순수 scope selection 오류는 ~23건** (전체의 4.1%). 나머지 9건은 올바른 장비를 선택했으나 gold 문서가 top-10에서 누락된 retrieval quality 문제.

---

## 2. scope_observability 분포

| scope_observability | 건수 |
|---------------------|------|
| explicit_device | **42건 (100%)** |
| explicit_equip | 0건 |

**모든 miss가 explicit_device 구간**에서 발생. explicit_equip에서는 miss 없음.
→ `[DEVICE]`로 마스킹된 후 범용 키워드만 남아 장비 식별이 어려운 구조적 문제.

---

## 3. 장비별 분포

| 장비 | 건수 | 비율 |
|------|------|------|
| PRECIA | 16 | 38.1% |
| ZEDIUS XP | 8 | 19.0% |
| INTEGER plus | 7 | 16.7% |
| SUPRA N | 4 | 9.5% |
| SUPRA N series | 4 | 9.5% |
| SUPRA Vplus | 3 | 7.1% |

**PRECIA가 42건 중 16건(38%)으로 최다.** ZEDIUS XP, INTEGER plus가 그 다음.

공통점: 이 장비들은 문서 볼륨이 상대적으로 작은 장비. P7+ device mass가 문서 볼륨이 큰 장비(GENEVA XP 등)에 쏠리면서 miss 발생.

---

## 4. 키워드 패턴 — 범용 부품명이 핵심 원인

| 키워드 | 건수 | 관련 장비 |
|--------|------|-----------|
| device net board | 5 | PRECIA, ZEDIUS XP |
| slot valve | 3 | INTEGER plus, PRECIA |
| solenoid valve | 3 | INTEGER plus, PRECIA |
| ctc | 3 | PRECIA, SUPRA N, ZEDIUS XP |
| ffu | 3 | PRECIA, SUPRA N series, ZEDIUS XP |
| controller | 3 | SUPRA N, SUPRA Vplus, ZEDIUS XP |
| fluorescent lamp | 2 | PRECIA, SUPRA N series |
| sensor board | 2 | PRECIA |
| manometer | 2 | PRECIA, ZEDIUS XP |
| heater chuck | 2 | SUPRA N series, ZEDIUS XP |

**모든 키워드가 여러 장비에 공통으로 등장하는 범용 부품명/컴포넌트명.**

마스킹 후 쿼리가 `[DEVICE] 설비에서 device net board 관련 절차와 주의사항은 무엇인가?` 형태가 되면, 장비를 구별할 단서가 전혀 없음. P7+ device mass는 문서 볼륨이 큰 장비 쪽으로 기울어짐.

---

## 5. 구조적 원인 분석

### 5.1 마스킹이 유일한 장비 단서를 제거

```
원문: "PRECIA 설비에서 device net board 관련 절차와 주의사항은 무엇인가?"
마스킹: "[DEVICE] 설비에서 device net board 관련 절차와 주의사항은 무엇인가?"
```

마스킹 후 남는 키워드 "device net board"는 GENEVA XP, SUPRA Vplus 등 여러 장비에 공통으로 존재.
→ P7+ device mass가 문서가 많은 GENEVAXP 쪽으로 기울어짐.

### 5.2 소형 장비의 구조적 불리함

| 장비 | 대략적 문서 수 | miss 건수 |
|------|---------------|-----------|
| PRECIA | 소 | 16 |
| ZEDIUS XP | 소 | 8 |
| INTEGER plus | 중 | 7 |
| GENEVA XP | 대 | 0 |
| TIGMA Vplus | 대 | 0 |

문서 볼륨이 큰 장비는 P7+ device mass에서 항상 높은 점수를 받음.
범용 키워드로만 검색하면, 문서가 많은 장비가 상위에 올 확률이 높음.

### 5.3 retrieval miss 9건 상세

올바른 장비를 선택했지만 gold 문서가 top-10에서 누락된 9건:

| q_id | 장비 | P9a 반환 문서 수 | B4 gold rank |
|------|------|------------------|-------------|
| A-gen0098 | INTEGER plus | 10 | 8 |
| A-gen0202 | SUPRA N series | 9 | 1 |
| A-gen0203 | SUPRA N series | 9 | 1 |
| A-gen0214 | SUPRA N series | 9 | 1 |
| A-gen0236 | SUPRA N | 9 | 7 |
| A-gen0300 | ZEDIUS XP | 8 | 9 |
| A-gen0329 | SUPRA Vplus | 8 | 2 |
| A-gen0335 | ZEDIUS XP | 7 | 10 |
| A-gen0414 | SUPRA N series | 9 | 1 |

특징:
- SUPRA N series 4건: B4에서 gold_rank=1이지만 P9a에서 누락 → P9a와 B4의 dense filter 차이
- ZEDIUS XP, SUPRA N: B4에서 gold_rank이 7~10 경계선 → 검색 품질 차이로 순위 밀려남
- P9a 반환 문서 수가 7~9개인 경우: dense filter에서 일부 chunk가 누락되어 후보 풀이 작음

---

## 6. P9b 복구 가능성 시뮬레이션

### P7+ device mass에서 target 장비의 위치

| target 위치 | 건수 | 비율 |
|------------|------|------|
| P7+ top-1 (retrieval miss) | 9 | 21.4% |
| P7+ top-2 | 10 | 23.8% |
| P7+ top-3 | 1 | 2.4% |
| P7+ top-5 이내 | 0 | 0.0% |
| **P7+ top-5 밖 (복구 불가)** | **22** | **52.4%** |

- 42건 중 20건(47.6%)은 P7+ top-3에 target이 존재 → P9b verifier로 복구 가능
- 22건(52.4%)은 P7+ top-5에도 없음 → **P7+ 기반으로는 복구 불가** (TF-IDF profile 필요)

### margin 분석

top-1과 target의 device mass 차이(margin)가 작을수록 verifier로 뒤집기 쉬움:

| 조건 | 건수 |
|------|------|
| target in top-3 AND margin < 0.5 | 18 |
| target in top-3 AND margin ≥ 0.5 | 2 |

→ **18건이 small-margin으로 P9b verifier 최적 대상**

### P9b 기대 효과

최선: scope miss 42건 중 20건 복구 → gold_strict +20 (492→512, 88.6%)
현실적: 18건 복구 (small-margin) → gold_strict +18 (492→510, 88.2%)
최악: verifier 오판으로 기존 정답도 뒤집힘 → 순이득 감소

---

## 7. 대응 방안

### 7.1 P9b margin-gated verifier (우선순위 높음)

scope miss 42건 중 20건을 복구할 수 있는 가능성:
- top-1 device의 evidence score가 너무 낮으면 top-2, top-3으로 fallback
- margin threshold를 통해 "확신이 없을 때만" 검증 수행
- 기대: margin < 0.5인 18건 복구

### 7.2 TF-IDF device profile (우선순위 중간)

P7+ top-5에도 없는 22건 해결:
- 장비별 고유 용어 프로필 구축
- "device net board"가 PRECIA에서 특히 자주/특이하게 등장하면 PRECIA 선호
- background frequency 대비 device-specific frequency로 가중
- PRECIA 16건 중 대부분이 여기에 해당 — TF-IDF가 PRECIA 특화 용어를 잡아야 함

### 7.3 retrieval miss 9건 개선 (우선순위 낮음)

- dense filter의 device_name variant 매칭 개선
- chunk-level reranking에서 gold 문서의 관련 chunk를 더 잘 포착
- SUPRA N series 4건은 B4에서 gold_rank=1인데 P9a에서 누락 — filter 차이 조사 필요

---

## 7. P9a hit & B4 miss (7건) — P9a가 B4보다 나은 케이스

B4(oracle hard filter)가 정답을 못 찾지만 P9a는 찾는 7건:

| q_id | 장비 | P9a MRR | B4 MRR | P9a가 찾은 문서 |
|------|------|---------|--------|----------------|
| A-gen0520 | PRECIA | 1.00 | 0.00 | TSG (trouble_shooting_guide) |
| A-gen0540 | PRECIA | 1.00 | 0.00 | TSG (trouble_shooting_guide) |
| A-gen0549 | SUPRA N | 1.00 | 0.00 | TSG + set_up_manual |
| A-gen0553 | SUPRA XP | 0.50 | 0.00 | TSG (device_net_abnormal 등) |
| A-gen0481 | SUPRA N | 0.12 | 0.00 | 트러블 로그 + SW operation |
| A-gen0005 | GENEVA XP | 0.10 | 0.00 | SOP (device_net, disc_amplifier) |
| A-gen0345 | SUPRA Vplus | 0.10 | 0.00 | 트러블 로그 |

### 패턴: P9a의 hybrid retrieval이 B4보다 다양한 문서 타입을 커버

- **4건(MRR≥0.5)**: P9a가 TSG(Trouble Shooting Guide) 문서를 top에 배치
  - B4는 SOP 문서 위주로 반환 → gold가 TSG에 있어서 miss
- P9a의 hybrid retrieval(BM25+dense+rerank)이 문서 타입에 관계없이 relevance 기반 정렬
- B4는 hard filter 후 검색하므로 후보 풀이 제한적 → TSG 같은 비SOP 문서를 놓칠 수 있음

→ **P9a의 retrieval 품질이 B4보다 우수한 경우가 존재** (7건, 1.2%)

---

## 8. Both miss (40건) — 두 방법 모두 못 찾는 케이스

### scope 분포

| scope_observability | 건수 | 비율 |
|---------------------|------|------|
| explicit_equip | **33** | 82.5% |
| explicit_device | 7 | 17.5% |

### 장비 분포

| 장비 | 건수 |
|------|------|
| SUPRA Vplus | **27** (67.5%) |
| SUPRA N | 4 |
| ZEDIUS XP | 3 |
| SUPRA XP | 2 |
| 기타 (SUPRA Vm, SUPRA Nm, TIGMA Vplus, INTEGER plus) | 4 |

### 분석

- **33/40건이 explicit_equip**: equip_id 기반 쿼리에서 B4(oracle)도 못 찾음
- **SUPRA Vplus가 27건 독점**: 특정 장비의 문서 커버리지 자체가 부족
- B3(unscoped)도 40건 중 1건만 hit → **gold 문서가 ES에 인덱싱되지 않았거나, gold 자체가 현재 코퍼스로는 해결 불가능한 쿼리**
- 이 40건은 scope selection 문제가 아니라 **코퍼스 커버리지 문제**

---

## 9. 논문 메시지

### 교차 분석 종합

| 구분 | 건수 | 원인 | 해결 가능성 |
|------|------|------|------------|
| B4 only (32건) | scope miss + retrieval miss | 마스킹 + 범용 키워드 | P9b/P9c 실패 → 구조적 한계 |
| P9a only (7건) | P9a retrieval 우위 | TSG 등 다양한 문서 타입 커버 | P9a의 장점 |
| Both miss (40건) | 코퍼스 커버리지 부족 | SUPRA Vplus 문서 부재 | 코퍼스 확장 필요 |

### 핵심 메시지

1. P9a의 scope miss는 **마스킹 + 범용 키워드** 조합에서 집중적으로 발생
2. **B4 only miss 32건은 전부 explicit_device** 구간 — explicit_equip에서는 0건
3. 문서 볼륨이 작은 장비(PRECIA, ZEDIUS XP)에 편중
4. **P9a가 B4보다 나은 7건도 존재** — hybrid retrieval이 TSG 등 다양한 문서 타입을 포착
5. **Both miss 40건은 코퍼스 한계** — scope selection과 무관
6. P9b(verification), P9c(TF-IDF) 모두 실패 → **scope miss ~5.7%는 마스킹 평가의 구조적 상한**
