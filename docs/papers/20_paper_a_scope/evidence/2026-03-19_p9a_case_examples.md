# P9a 알고리즘 효과: 사례 기반 분석

**작성일**: 2026-03-19
**목적**: Cross-equipment contamination 문제와 P9a 해법의 효과를 구체적 쿼리 사례로 설명

---

## 배경: 왜 기존 검색이 문제인가?

PE 트러블슈팅 시스템에서 사용자가 특정 장비에 대한 질문을 하면, 검색 시스템은 해당 장비의 문서를 찾아야 한다. 그러나 기존 검색(B3 baseline)은 **다른 장비의 문서가 결과에 섞이는 cross-equipment contamination** 문제를 가지고 있다.

- 전체 578개 평가 쿼리에서 B3의 contamination rate = **58.4%**
- 즉, 검색 결과 10개 중 평균 5.8개가 잘못된 장비의 문서

**P9a 알고리즘**은 이 문제를 해결하기 위해:
1. P7+ 검색 결과에서 가장 관련성 높은 장비를 1개 선정 (device proposal)
2. 해당 장비의 문서만을 대상으로 hybrid+rerank 검색 수행 (hard scope)

→ contamination을 0.584에서 **0.048로 91.8% 감소**시키면서, 정답 문서 검색률은 유지.

---

## 사례 1: 장비 ID가 있는데도 오염되는 경우

### 쿼리

> **A-gen0021**
> `[EQUIP] 장비에서 HOOK LIFTER LM GUIDE 관련 이상 원인과 점검/조치 이력은 무엇인가?`
> (원문: "WPSKK22U00 장비에서 HOOK LIFTER LM GUIDE 관련...")
> 정답 장비: **GENEVA XP** | scope_observability: `explicit_device`

### B3 (기존 검색) 결과

| 순위 | 문서 | 장비 | 정답? |
|------|------|------|-------|
| 1 | INTEGER plus 문서 | INTEGER plus | ✗ |
| 2 | INTEGER plus 문서 | INTEGER plus | ✗ |
| 3 | INTEGER plus 문서 | INTEGER plus | ✗ |
| ... | ... | ... | ... |
| 10 | INTEGER plus 문서 | INTEGER plus | ✗ |

- **contamination: 1.00** (10개 전부 잘못된 장비)
- **gold hit: 없음** — GENEVA XP 문서가 결과에 단 1개도 없음

쿼리에 장비 ID "WPSKK22U00"이 있지만 문서에는 장비명(GENEVA XP)으로 저장되어 있어 매칭 실패. BM25+Dense 검색이 "HOOK LIFTER LM GUIDE"라는 일반적 키워드에 끌려 다른 장비 문서를 우선 반환.

### P9a 결과

| 순위 | 문서 | 장비 | 정답? |
|------|------|------|-------|
| 1 | GENEVA XP 문서 | GENEVA XP | ✗ |
| 2 | GENEVA XP 문서 | GENEVA XP | ✗ |
| 3 | GENEVA XP gold 문서 | GENEVA XP | **✓** |
| ... | ... | ... | ... |
| 10 | GENEVA XP 문서 | GENEVA XP | ✗ |

- **contamination: 0.00**
- **gold hit: 있음** (rank 3, MRR=0.333)
- P9a가 GENEVA XP를 정확히 선택 → 해당 장비 문서만 검색 → 정답 문서 발견

### 시사점

장비명이 쿼리에 명시되어 있어도, 키워드 기반 검색은 문서 볼륨이 큰 다른 장비에 의해 오염될 수 있다. P9a는 장비를 먼저 특정하고 해당 범위에서만 검색하므로 이 문제를 원천 차단한다.

---

## 사례 2: 장비 ID(equip_id)만 있는 경우 — B3가 완전히 실패하는 구간

### 쿼리

> **A-gen0341**
> `[EQUIP] 장비에서 FFU CONTROLLER 관련 이상 원인과 점검/조치 이력은 무엇인가?`
> (원문: "WPSKK44H00 장비에서 FFU CONTROLLER 관련...")
> 정답 장비: **TIGMA Vplus** | scope_observability: `explicit_equip`

이 유형의 쿼리는 장비명 대신 장비 ID(예: "WPSKK44H00")만 포함하고 있어, 텍스트 매칭으로는 장비를 특정하기 어렵다.

### B3 결과

- **contamination: 0.90** (10개 중 9개가 다른 장비)
- **gold hit: 없음**
- 장비 ID → 장비명 매핑 없이는 올바른 문서를 찾을 수 없음

### P9a 결과

- **contamination: 0.00**
- **gold hit: 있음** (MRR=0.333)
- P7+의 device mass 분석이 TIGMA Vplus를 정확히 제안 → hard scope 검색 성공

### 이 구간의 전체 통계

| 조건 | gold_strict | contamination |
|------|-------------|---------------|
| B3 | 10/149 (**6.7%**) | 0.881 |
| B4 (oracle) | 111/149 (74.5%) | 0.000 |
| **P9a** | **112/149 (75.2%)** | **0.000** |

- B3는 explicit_equip 149건 중 **단 10건**만 정답을 찾음 (6.7%)
- P9a는 **112건** — oracle(정답 장비를 미리 알고 있는 B4)보다 1건 더 많음

---

## 사례 3: P9a가 Oracle(B4)을 이기는 경우

### 쿼리

> **A-gen0005**
> `[EQUIP] 장비에서 LOAD PORT CERTIFICATION 관련 이상 원인과 점검/조치 이력은 무엇인가?`
> (원문: "EPAG04 장비에서 LOAD PORT CERTIFICATION 관련...")
> 정답 장비: **GENEVA XP** | scope_observability: `explicit_device`

### B4 (Oracle) 결과

B4는 정답 장비를 미리 알고 있으므로 해당 장비 문서만 검색한다. 그런데도:

- **gold hit: 없음**
- 이유: B4의 parser가 추출한 device_name이 ES 인덱스의 변형과 미세하게 불일치하거나, 해당 장비 내에서 정답 문서의 relevance score가 낮음

### P9a 결과

- **gold hit: 있음**
- P7+가 제안한 device_name 변형이 ES 인덱스와 더 잘 매칭됨
- 또는 P9a의 chunk-level RRF가 더 다양한 후보를 생성하여 정답 문서를 포착

### 시사점

P9a가 oracle보다 나은 경우가 존재하는 이유:
1. P7+의 device proposal이 parser보다 유연한 장비명 매핑을 수행
2. explicit_equip 쿼리에서 parser는 장비명을 전혀 추출하지 못하지만(0%), P7+는 97.3% 정확도로 장비를 제안

이것이 explicit_equip 구간에서 P9a(75.2%) ≥ B4(74.5%)인 핵심 이유다.

---

## 사례 4: P9a가 실패하는 경우 — scope miss

### 쿼리

> **A-gen0079**
> `[DEVICE] 설비에서 APC Abnormal 트러블슈팅 시 핵심 점검 포인트는 무엇인가?`
> (원문: "supra_n 설비에서 APC Abnormal 트러블슈팅 시...")
> 정답 장비: **INTEGER plus** | scope_observability: `explicit_device`

### P9a 결과

- **scope 선택: 오답** (다른 장비 선택)
- **정답 장비: INTEGER plus**
- **contamination: 1.00** — 결과 전부가 잘못된 장비
- gold hit: 없음

"APC Abnormal"이라는 키워드가 여러 장비 문서에 공통으로 등장하고, 마스킹된 쿼리에서 장비 단서가 `[DEVICE]`로 가려져 있어 P7+ device mass가 다른 장비 쪽으로 기울어짐. (B4 oracle은 gold hit 성공)

### 분석

P9a의 전체 scope accuracy = 93.4% (540/578). 나머지 **38건의 scope miss**가 P9a의 성능 ceiling을 결정한다.

scope miss 패턴:
- 여러 장비에 공통으로 등장하는 일반 용어 (예: "net board", "pump", "valve")
- 문서 볼륨이 큰 장비에 device mass가 쏠리는 경향

→ 이 38건을 줄이는 것이 후속 연구(P9b margin-gated verifier, TF-IDF device profile)의 목표.

---

## 사례 5: P7+의 contamination을 P9a가 정화하는 경우

### 쿼리

> **A-gen0031**
> `[DEVICE] 설비에서 load port 관련 절차와 주의사항은 무엇인가?`
> (원문: "GENEVA XP 설비에서 load port 관련 절차와 주의사항은...")
> 정답 장비: **INTEGER plus** | scope_observability: `explicit_device`

### P7+ 결과

- **contamination: 0.60** (10개 중 6개가 다른 장비 문서)
- **gold hit: 있음** (MRR=0.250) — 정답은 찾았으나, 다른 장비 문서 6개가 섞여 있음

P7+는 정답 문서를 찾는 능력은 높지만(gold_strict=87.5%), 동시에 다른 장비 문서도 함께 반환하므로 실제 사용 시 혼란을 야기한다.

### P9a 결과

- **contamination: 0.00**
- **gold hit: 있음** (MRR=1.000, rank 1!) — 동일한 정답 문서를 반환하면서, 오염 문서는 모두 제거되고 순위도 1위로 상승

### 시사점

P7+가 "정답을 포함하지만 오염된" 결과를 생성할 때, P9a는 P7+의 device proposal 능력만 활용하고 최종 검색은 해당 장비 범위에서만 수행하므로, **정답은 유지하면서 오염만 제거**할 수 있다.

이것이 P9a의 핵심 설계 원리다:
- P7+를 **최종 검색기가 아니라 장비 제안기**로 사용
- 최종 검색은 제안된 장비의 hard scope에서 수행

---

## 전체 요약

| | B3 (기존) | P7+ (소프트 스코어링) | B4 (Oracle) | **P9a (제안)** |
|---|---|---|---|---|
| contamination | 0.584 | 0.515 | 0.001 | **0.048** |
| gold_strict | 60.7% | 87.5% | 91.2% | **85.1%** |
| MRR | 0.335 | 0.521 | 0.562 | **0.618** |

### 핵심 메시지

1. **Cross-equipment contamination은 candidate generation 단계의 문제**다. 검색 결과에 점수를 다시 매기는 방식(soft scoring)으로는 해결할 수 없다.

2. **P9a는 "장비를 먼저 특정하고, 그 범위에서만 검색"하는 hard scope 전략**으로, contamination을 91.8% 감소시키면서 정답 검색률은 B4 oracle의 93% 수준을 달성한다.

3. **특히 장비 ID만 있는 쿼리(explicit_equip)**에서 P9a는 B4 oracle을 초과하는 성능을 보인다 — parser가 전혀 대응하지 못하는 구간에서 P7+ device proposal이 97.3% 정확도로 장비를 특정하기 때문이다.

4. **한계**: scope miss 38건(6.6%)이 성능 ceiling을 결정한다. 이를 개선하기 위한 후속 연구(margin-gated verifier, TF-IDF device profile)가 필요하다.
