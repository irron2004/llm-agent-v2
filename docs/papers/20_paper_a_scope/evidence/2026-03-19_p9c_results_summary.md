# P9c TF-IDF Device Profile 실험 결과

**작성일**: 2026-03-19
**알고리즘**: TF-IDF device profile proposal (단독 + P7+ hybrid)
**데이터**: `data/paper_a/p9c_results.json`

---

## 1. 알고리즘

```
1. ES chapter 필드 aggregation으로 장비별 pseudo-document 구축
   - 각 장비의 chapter title + doc_id 패턴을 하나의 텍스트로 결합
2. TfidfVectorizer(sublinear_tf=True)로 벡터화
3. 쿼리와 각 장비 프로필 간 cosine similarity 계산
4. Hybrid: combined = p7_w * p7_norm + tfidf_w * tfidf_norm
5. Top-1 device로 hard filter retrieval 수행
```

---

## 2. 전체 결과 (n=564)

| Condition | cont@10 | gold_strict | MRR | scope_acc |
|-----------|---------|-------------|-----|-----------|
| B3_masked | 0.590 | 341 (60.5%) | 0.339 | — |
| B4_masked | 0.001 | 517 (91.7%) | 0.564 | — |
| P7plus_masked | 0.516 | 497 (88.1%) | 0.529 | — |
| **P9a_masked** | **0.048** | **492 (87.2%)** | **0.618** | **0.957** |
| P9c_tfidf | 0.704 | 179 (31.7%) | 0.258 | 0.293 |
| P9c_hybrid_03 | 0.474 | 296 (52.5%) | 0.415 | 0.527 |
| P9c_hybrid_05 | 0.181 | 422 (74.8%) | 0.553 | 0.824 |
| P9c_hybrid_07 | 0.048 | 492 (87.2%) | 0.618 | 0.957 |
| P9c_hybrid_09 | 0.048 | 492 (87.2%) | 0.618 | 0.957 |

### Delta from P9a

| Condition | Δgold_strict | Δcont@10 | ΔMRR |
|-----------|-------------|----------|------|
| P9c_tfidf | **-313** | +0.655 | -0.361 |
| P9c_hybrid_03 | **-196** | +0.426 | -0.203 |
| P9c_hybrid_05 | **-70** | +0.133 | -0.066 |
| P9c_hybrid_07 | **0** | 0.000 | 0.000 |
| P9c_hybrid_09 | **0** | 0.000 | 0.000 |

---

## 3. explicit_equip (n=145)

| Condition | gold_strict | MRR |
|-----------|-------------|-----|
| P9a_masked | 112 (77.2%) | 0.448 |
| P9c_tfidf | 0 (0.0%) | 0.000 |
| P9c_hybrid_05 | 49 (33.8%) | 0.215 |
| P9c_hybrid_07 | 112 (77.2%) | 0.444 |

TF-IDF는 equip_id 기반 식별을 전혀 못함 → explicit_equip에서 0%.

---

## 4. explicit_device (n=419)

| Condition | gold_strict | MRR |
|-----------|-------------|-----|
| P9a_masked | 380 (90.7%) | 0.677 |
| P9c_tfidf | 179 (42.7%) | 0.347 |
| P9c_hybrid_05 | 373 (89.0%) | 0.670 |
| P9c_hybrid_07 | 380 (90.7%) | 0.679 |

explicit_device에서도 TF-IDF 단독은 42.7%로 B3(79%)에도 못 미침.

---

## 5. P9a scope miss 복구 분석

### 핵심 결과: TF-IDF top-1이 miss 32건 중 0건 정답

| Condition | 복구 건수 | scope_correct |
|-----------|----------|---------------|
| P9c_tfidf | 7/32 | 1/32 |
| P9c_hybrid_03 | 5/32 | 1/32 |
| P9c_hybrid_05 | 0/32 | 10/32 |
| P9c_hybrid_07 | 0/32 | 10/32 |

- **TF-IDF top-1 correct for miss cases: 0/32** — 모든 miss 케이스에서 TF-IDF가 오답
- TF-IDF가 복구한 7건도 scope가 맞아서가 아님 (scope_correct=1) → 우연히 다른 장비 문서가 hit
- hybrid_05: P9a 정답 492건 중 70건을 파괴하면서 miss 32건 중 0건 복구 → net -70

### TF-IDF의 편향: 거의 항상 SUPRAXP 선택

```
A-gen0079: target=INTEGER plus, tfidf_top1=SUPRAXP ✗
A-gen0166: target=PRECIA,       tfidf_top1=SUPRAXP ✗
A-gen0172: target=PRECIA,       tfidf_top1=SUPRAXP ✗
A-gen0201: target=SUPRA N,      tfidf_top1=SUPRAXP ✗
A-gen0202: target=SUPRA N series, tfidf_top1=ETC ✗
```

SUPRAXP가 chapter 수가 가장 많아 TF-IDF pseudo-document가 가장 풍부 → cosine similarity가 SUPRAXP에 편향.

---

## 6. 실패 원인 분석

### 6.1 문서 볼륨 bias 재현

P7+ device mass의 근본 문제가 TF-IDF에서도 동일하게 발생:

| 장비 | chapter 수 (상대) | TF-IDF 선호도 |
|------|-------------------|---------------|
| SUPRA XP (GENEVA XP) | 대 | 최고 |
| SUPRA N | 중 | 중간 |
| PRECIA | 소 | 최저 |

→ chapter title을 pseudo-document로 쓰면 문서가 많은 장비의 프로필이 더 풍부 → 범용 키워드와의 유사도가 항상 높음.

### 6.2 TF-IDF의 구조적 한계

1. **chapter title은 장비 고유성을 반영하지 않음**: "heater chuck", "device net board" 등이 모든 장비에 공통으로 존재
2. **IDF가 작동하려면 장비 간 차별화된 용어가 필요**: 하지만 PE 장비의 부품명은 대부분 공통
3. **pseudo-document 길이 불균형**: 문서 많은 장비 = 긴 프로필 = cosine similarity 유리

### 6.3 Hybrid에서 weight 딜레마

- **TF-IDF weight ≥ 0.5**: TF-IDF의 SUPRAXP 편향이 P7+ 정보를 덮어씀 → 성능 급락
- **TF-IDF weight ≤ 0.3**: P7+ 정보가 지배적 → P9a와 동일 → TF-IDF 무의미
- **sweet spot 부재**: 어떤 비율에서도 P9a를 이길 수 없음

### 6.4 부품명 빈도 분포와 P9a 정확도의 상관관계

ES content에서 실제 부품명별 장비 분포를 확인한 결과, **장비 집중도와 P9a 정확도가 강하게 상관**:

| 부품명 | ES 장비 수 | top1 장비 (점유율) | P9a 정답률 |
|--------|-----------|-------------------|-----------|
| pressure relief valve | 5 | PRECIA (51%) | 3/3 (100%) |
| vacuum line | 31 | INTEGERPLUS (40%) | 4/4 (100%) |
| gas spring | 25 | SUPRAN (47%) | 3/3 (100%) |
| robot | 34 | SUPRAN (36%) | 6/6 (100%) |
| **device net board** | **30** | **SUPRAN (29%)** | **2/7 (29%)** |
| **solenoid valve** | **31** | **SUPRAN (24%)** | **2/6 (33%)** |
| **controller** | **33** | **SUPRAN (23%)** | **2/5 (40%)** |
| **sensor board** | **26** | **SUPRAN (27%)** | **2/4 (50%)** |

→ **집중도가 높은 부품(top1 > 40%)은 P7+가 이미 정답** → TF-IDF 불필요
→ **miss가 발생하는 부품(top1 < 30%)은 25~33개 장비에 분산** → TF-IDF로도 구분 불가

### 6.5 마스킹 쿼리의 정보량 한계

miss 케이스 32건의 마스킹된 쿼리를 확인한 결과, **전부 동일한 템플릿**:

```
[DEVICE] 설비에서 {부품명} 관련 절차와 주의사항은 무엇인가?
```

- 쿼리에 포함된 정보가 **부품명 하나뿐** (예: "device net board", "solenoid valve")
- 장비별 고유 동반 용어(PRECIA→"fuse"/"cicem", INTEGERPLUS→"sbb"/"hook")는 **문서 내에만 존재**, 쿼리에는 없음
- TF-IDF가 쿼리를 프로필에 매칭하려 해도, 부품명 하나로 30개 장비를 구분하는 것은 정보이론적으로 불가능

### 6.6 구현 개선으로 결과가 바뀔 수 있는가?

현재 구현은 chapter title 기반이지만, 개선하더라도 근본 한계는 동일:

| 개선 방안 | 기대 효과 | 한계 |
|-----------|----------|------|
| 실제 document content 기반 프로필 | 빈도 정보 풍부 | 쿼리에 부품명 하나뿐이라 매칭할 정보 부족 |
| n-gram(bigram/trigram) 사용 | 복합 표현 포착 | "device net board"가 이미 30개 장비에 공통 |
| doc_id에서 장비명 제거 | noise 감소 | 근본 문제(쿼리 정보량)는 불변 |
| BM25 length normalization | 볼륨 bias 감소 | top1 점유율 < 30%인 부품은 어떤 normalization으로도 구분 불가 |

---

## 7. 결론

### P9c는 negative result

1. **TF-IDF device profile은 scope selection에 전혀 도움이 안 됨**
2. 단독 사용 시 scope_acc=29.3% (P9a의 95.7% 대비 참담)
3. Hybrid에서 의미 있는 비중을 주면 성능 급락, 적은 비중은 효과 없음
4. **P9a의 42건 scope miss 중 0건을 TF-IDF로 복구 불가**

### P9b + P9c 실패의 종합 메시지

| 실험 | 전략 | 결과 |
|------|------|------|
| P9b | Stage 3 verification 추가 | negative (net -4 ~ -26) |
| P9c | Stage 1에 TF-IDF proposal 추가 | negative (0건 복구) |

→ **P9a(P7+ top-1 hard scope)가 현 데이터에서 최적이며, scope miss 42건은 마스킹 조건에서 구조적으로 해결 불가.**

### 논문 메시지

1. P7+ device mass의 document volume bias는 TF-IDF에서도 재현 → 이는 "문서가 많은 장비가 유리한" 구조적 편향
2. 범용 부품명(device net board, slot valve 등)으로만 남은 마스킹 쿼리에서 장비를 식별하는 것은 lexical/statistical 방법으로는 불가능
3. **마스킹 환경에서의 scope selection 상한은 ~93%이며, 이를 넘으려면 장비 고유 용어 사전(ontology) 또는 device-specific embedding이 필요**
4. 하지만 이는 마스킹의 본래 목적(장비 독립적 평가)과 상충 → 마스킹 평가의 구조적 한계로 보고하는 것이 적절
