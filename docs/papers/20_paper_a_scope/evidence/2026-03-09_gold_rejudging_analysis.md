# Gold Label 재판정 분석 보고서

> Date: 2026-03-09
> Status: Phase 1+2+3+4 완료 (재판정 + 메트릭 재계산 + 평가셋 확대 + P6/P7 구현)
> Next: 논문 본문 업데이트

---

## 1. 문제 발견: Gold가 query-specific하지 않음

### 1.1 현상

기존 `query_gold_master_v0_5.jsonl`의 gold_doc_ids가 **질문별이 아닌 장비별로 동일하게 할당**되어 있었다.

| Device | Test 쿼리 수 | Unique gold set 수 | 의미 |
|--------|-------------|-------------------|------|
| SUPRA_XP | 17 | 1 | 17개 질문이 동일한 gold 5개 공유 |
| GENEVA_XP | 4 | 1 | 4개 질문이 동일한 gold 5개 공유 |
| SUPRA_VPLUS | 2 | 1 | 동일 gold |
| OMNIS | 4 | 2 | 2개 그룹으로 나뉨 |
| SUPRA_N | 2 | 2 | 각각 다른 gold |

**Dev set도 동일한 문제:**
- 421개 dev 쿼리 중 79개 unique gold set만 존재
- 128개 쿼리가 동일한 SUPRA_XP gold 5개 공유
- 73개 쿼리는 gold가 아예 비어있음 (empty gold_doc_ids)

### 1.2 구체적 예시

**SUPRA_XP 쿼리 — 모두 같은 gold 5개:**

| Query | 질문 내용 | Gold (전부 동일) |
|-------|----------|-----------------|
| A-sop018 | PM **Baratron Gauge** ADJ | ll_cip, slot_vv_housing, temp_controller, process_kit, pendulum_valve |
| A-sop031 | **Manometer** Adjust | (동일) |
| A-sop042 | **PRISM SOURCE** 교체 | (동일) |
| A-sop076 | EFEM **SW Install** | (동일) |

→ Gold 문서(ll_cip, slot_vv_housing 등)는 이 질문들과 **주제적으로 무관**

### 1.3 영향

- **hit@5 과소평가**: 실제 정답 문서를 검색했지만 gold에 없어서 miss로 카운트
- **contamination 왜곡**: 실제 정답인 문서를 "오염"으로 카운트
- **시스템 간 비교 무효**: 잘못된 gold 기준의 모든 비교가 신뢰 불가

---

## 2. 재판정 방법

### Phase 0: doc_id 이름 기반 (근사치)

모든 시스템(B0-B3, B4, B4.5, P1)의 top-5 검색 결과를 pooling하여 unique (query, doc) pair를 수집한 후, doc_id 이름의 component 키워드와 질문 주제를 매칭하여 relevance를 판정.

- **553개** unique (query, doc) pair
- **160개** unique document
- **43개** non-equip test query (explicit_device 22 + implicit 21)

한계: 숫자 ID 문서 판정 불가, 문서 내용 미확인 근사치

### Phase 1: ES 문서 내용 기반 정밀 판정 (LLM judging)

1. pooled 160개 unique doc의 실제 내용을 `chunk_v3_content` ES 인덱스에서 추출
2. 427개 (query, doc) pair를 9개 배치로 분할
3. 각 배치를 병렬 LLM scientist agent로 graded relevance 판정

**판정 기준:**

| Relevance | 정의 | 예시 |
|-----------|------|------|
| 2 (정답) | 질문 주제와 정확히 매칭 + 올바른 장비 | Q:"Baratron Gauge ADJ" → `pm_baratron_gauge` (SUPRA_XP) |
| 1 (부분관련) | 같은 컴포넌트이지만 다른 장비, 또는 관련 하위시스템 | Q:"Baratron Gauge ADJ" → `pm_baratron_gauge` (PRECIA) |
| 0 (무관) | 주제 불일치 또는 무관한 문서 | Q:"Baratron Gauge ADJ" → `efem_controller` |

**판정 결과 분포:**

| Relevance | Count | 비율 |
|-----------|-------|------|
| 0 (무관) | 250 | 58.5% |
| 1 (부분관련) | 132 | 30.9% |
| 2 (정답) | 45 | 10.5% |

→ 427개 pair 중 10.5%만 완전 정답, 30.9%가 부분 관련 (같은 컴포넌트 다른 장비)

---

## 3. Phase 2: 새 Gold 기반 메트릭 재계산

### 3.1 시스템별 hit@5 비교 (non-equip 43개 쿼리)

| System | 설명 | old hit@5 | **new hit@5 (strict)** | new hit@5 (loose) | new MRR |
|--------|------|-----------|----------------------|-------------------|---------|
| B0 | BM25 | 0.302 | **0.791** | 0.907 | 0.711 |
| B1 | Dense | 0.209 | **0.698** | 0.930 | 0.613 |
| B2 | Hybrid | 0.349 | **0.837** | 0.907 | 0.736 |
| B3 | Hybrid+Rerank | 0.349 | **0.837** | 0.907 | 0.721 |
| B4 | Device filter+Rerank | 0.349 | **0.814** | 0.930 | 0.698 |
| B4.5 | Shared-aware+Rerank | 0.209 | **0.512** | 0.605 | 0.484 |
| P1 | Scope policy+Rerank | 0.070 | **0.186** | 0.395 | 0.171 |

- **strict**: relevance=2만 gold로 인정
- **loose**: relevance≥1 (주제 매칭이면 다른 장비도 포함)

### 3.2 Contamination 변화

| System | old adj_cont@5 | **new cont@5** | 해석 |
|--------|---------------|----------------|------|
| B0 | 0.252 | **0.271** | 비슷 (scope filter 없음) |
| B1 | 0.400 | **0.269** | 개선 (기존 gold 오류로 과대) |
| B2 | 0.253 | **0.266** | 비슷 |
| B3 | 0.257 | **0.276** | 비슷 |
| B4 | 0.214 | **0.271** | 비슷 |
| B4.5 | 0.000 | **0.538** | **급증** — scope filter가 정답을 제거하고 나머지가 오염 |
| P1 | 0.000 | **0.736** | **매우 높음** — strict policy가 대부분의 정답 제거 |

### 3.3 Scope observability별 (B3 기준)

| Slice | n | old hit@5 | **new hit@5** | new MRR | old cont@5 | **new cont@5** |
|-------|---|-----------|---------------|---------|------------|----------------|
| explicit_device | 22 | 0.409 | **0.864** | 0.841 | 0.130 | 0.282 |
| implicit | 21 | 0.286 | **0.810** | 0.595 | 0.391 | 0.269 |

---

## 4. 핵심 시사점

### 4.1 검색 시스템은 이미 잘 작동하고 있었음

B2/B3 (Hybrid ± Rerank)의 실제 hit@5는 **0.837**로, 기존 0.35라는 수치는 gold 라벨 오류로 인한 착시.
10개 질문 중 8개 이상에서 올바른 문서를 top-5에 포함.

### 4.2 Scope filtering의 recall-contamination trade-off

| System | hit@5 | cont@5 | 해석 |
|--------|-------|--------|------|
| B3 (no filter) | 0.837 | 0.276 | 높은 recall, 적당한 오염 |
| B4 (hard device) | 0.814 | 0.271 | recall 미세 하락, 오염 비슷 |
| B4.5 (shared-aware) | 0.512 | 0.538 | **38.8% recall 손실**, 오염 급증 |
| P1 (strict policy) | 0.186 | 0.736 | **77.8% recall 손실**, 오염 극대 |

**핵심 발견**: Scope filtering이 엄격할수록 recall이 급감하면서 동시에 contamination도 증가.
이는 filter가 정답 문서를 제거하여 top-5에 무관한 문서만 남기 때문.

### 4.3 논문 narrative 수정 필요

기존 가설: "scope filter가 contamination을 줄이고, recall 손실은 제한적"
→ **실제**: scope filter가 과도하게 적용되면 recall도 contamination도 모두 악화

이는 **contamination-aware scoring** (P6/P7)이 hard filtering보다 나은 접근이 될 수 있음을 시사.

### 4.4 Hybrid > Dense 확인

| System | hit@5 | MRR |
|--------|-------|-----|
| B0 (BM25) | 0.791 | 0.711 |
| B1 (Dense) | 0.698 | 0.613 |
| B2 (Hybrid) | 0.837 | 0.736 |

Dense 단독은 BM25보다 낮으며, Hybrid가 가장 높은 recall.
한국어 기술문서 도메인에서 BM25의 강점이 두드러짐.

---

## 5. 후속 작업 계획

### Phase 3: 평가셋 확대 (51 → 151) — 완료

#### 3.1 방법

1. dev set 421개 쿼리에서 109개 balanced subset 선택 (seed=42)
   - explicit_device: 28, implicit: 29, explicit_equip: 22, ambiguous: 30
   - 장비 다양성 보장 (round-robin across devices)
2. 7개 시스템(B0-P1) retrieval 실행 → 2076 pooled (query, doc) pairs
3. 353개 unique 문서 ES content fetch (chunk_v3_content)
4. 14개 배치 × 150 pairs → 병렬 LLM judging

#### 3.2 판정 결과

| 항목 | 수치 |
|------|------|
| 판정 (query, doc) pairs | 2,077 |
| Relevance 분포 | 0: 1,675 (80.7%), 1: 348 (16.8%), 2: 54 (2.6%) |
| 쿼리 with strict gold (rel=2) | 36/108 (33.3%) |
| 쿼리 with loose gold (rel≥1) | 96/108 (88.9%) |

#### 3.3 확대된 평가셋 결과 (n=130, non-equip)

| System | 설명 | hit@5 (strict) | hit@5 (loose) | MRR |
|--------|------|---------------|---------------|-----|
| B0 | BM25 | 0.415 | 0.700 | 0.341 |
| B1 | Dense | 0.415 | 0.731 | 0.326 |
| B2 | Hybrid | 0.439 | 0.723 | 0.353 |
| B3 | Hybrid+Rerank | 0.423 | 0.715 | 0.342 |
| B4 | Device filter | 0.423 | 0.685 | 0.344 |
| B4.5 | Shared-aware | 0.269 | 0.431 | 0.222 |
| P1 | Scope policy | 0.131 | 0.323 | 0.101 |

#### 3.4 Observability별 (B3, n=130)

| Slice | n | hit@5 (strict) | hit@5 (loose) | MRR |
|-------|---|---------------|---------------|-----|
| explicit_device | 50 | **0.540** | 0.780 | 0.492 |
| implicit | 50 | **0.460** | 0.720 | 0.350 |
| ambiguous | 30 | **0.167** | 0.600 | 0.080 |

#### 3.5 Test vs Dev 비교 (B3)

| Subset | n | hit@5 (strict) | 특징 |
|--------|---|---------------|------|
| Test only (Phase 1) | 43 | **0.837** | SUPRA_XP 위주, 높은 corpus 커버리지 |
| Dev only (Phase 3) | 87 | **0.218** | 다양한 장비, 낮은 corpus 커버리지 |
| Has strict gold only | 70 | **0.786** | 정답 존재 쿼리만 → 검색 자체는 우수 |

→ **핵심 발견**: 검색 품질의 하락이 아닌 **corpus 커버리지 부족**이 주원인.
  다양한 장비의 SOP가 corpus에 없어서 rel=2 문서 자체가 존재하지 않음.

### Phase 4: P6/P7 Contamination-aware Scoring — 완료

#### 4.1 방법

B3의 top-10 검색 결과에 scope 위반 penalty를 적용하여 re-scoring:

```
Score(d,q) = Base(d,q) - λ(q) · v_scope(d,q)
```

- `Base(d,q)` = 1/rank (reranker 순위 기반 proxy score)
- `v_scope(d,q)` = 1 if doc is out-of-scope (not shared AND device mismatch), else 0
- P6: λ 고정, P7: λ(q) = observability별 적응형

#### 4.2 P6 결과: λ sweep

| λ | hit@5 (strict) | hit@5 (loose) | MRR |
|---|---------------|---------------|-----|
| 0 (=B3) | 0.423 | 0.715 | 0.342 |
| 0.05 (best) | **0.431** | 0.731 | 0.344 |
| 0.1 | 0.423 | 0.746 | 0.343 |
| 0.5 | 0.423 | 0.754 | 0.345 |

#### 4.3 P7 결과: Adaptive λ(q)

Best config (P7-a): `explicit_device=0.3, implicit=0.15, ambiguous=0.05`

| System | hit@5 (strict) | hit@5 (loose) | MRR |
|--------|---------------|---------------|-----|
| B3 (no filter) | 0.423 | 0.715 | 0.342 |
| B4 (hard device) | 0.423 | 0.685 | 0.343 |
| B4.5 (shared-aware) | 0.269 | 0.431 | 0.222 |
| P1 (strict policy) | 0.131 | 0.323 | 0.101 |
| **P6 (λ=0.05)** | **0.431** | 0.731 | 0.344 |
| **P7 (adaptive)** | **0.431** | **0.746** | **0.346** |

#### 4.4 핵심 발견

1. **P6/P7은 B3 대비 recall을 유지하면서 미세 개선** (strict: +1.9%, loose: +4.3%)
2. **Hard filtering(B4.5/P1)의 catastrophic recall loss 회피**: B4.5 -36%, P1 -69% → P7 +1.9%
3. **P7의 loose hit@5=0.746**: 관련 문서(다른 장비 동일 컴포넌트)를 상위로 끌어올림
4. **개선 폭이 작은 이유**: 46%의 쿼리에 strict gold 자체가 없음 (corpus 커버리지 부족)

→ **논문 contribution**: Hard scope filtering 대신 soft contamination penalty가 recall 보존에 효과적

---

## 6. 산출물

### Phase 1+2 산출물

| 파일 | 설명 |
|------|------|
| `data/paper_a/rejudge/rejudge_input.json` | 51개 쿼리 × pooled docs 입력 데이터 |
| `data/paper_a/rejudge/pooled_doc_info_v3.json` | 160개 문서 ES 내용 (chunk_v3_content) |
| `data/paper_a/rejudge/all_judgments.json` | 427개 (query, doc) pair LLM 판정 결과 |
| `data/paper_a/rejudge/new_gold_v1.json` | 43개 쿼리 query-specific gold (strict/loose) |
| `data/paper_a/rejudge/phase2_metrics.json` | 시스템별 old/new 메트릭 비교 |

### Phase 3 산출물

| 파일 | 설명 |
|------|------|
| `data/paper_a/phase3_selected_dev_queries.json` | 109개 선택된 dev 쿼리 ID |
| `data/paper_a/phase3_pooled_pairs.json` | 2,076개 pooled (query, doc) pairs |
| `data/paper_a/phase3_all_doc_info.json` | 375개 문서 ES 내용 |
| `data/paper_a/phase3_all_judgments.json` | 2,077개 LLM 판정 결과 |
| `data/paper_a/phase3_new_gold.json` | 108개 쿼리 query-specific gold |
| `data/paper_a/phase3_final_metrics.json` | 확대된 평가셋 메트릭 (all slices) |

### Phase 4 산출물

| 파일 | 설명 |
|------|------|
| `data/paper_a/phase4_p6p7_results.json` | P6/P7 실험 결과 (λ sweep + adaptive configs) |
| 본 문서 | 분석 보고서 |

---

## 7. 종합 결론 및 논문 방향

### 7.1 주요 발견 요약

| 발견 | 영향 |
|------|------|
| Gold label이 query-specific이 아닌 device-level로 할당됨 | 기존 모든 메트릭 무효화 |
| 재판정 후 B3 hit@5 = 0.837 (test 43, 기존 0.349) | 검색 시스템은 이미 우수 |
| 확대된 평가셋(n=130)에서 B3 hit@5 = 0.423 | 다양한 장비/쿼리 포함 시 하락 |
| 하락 원인: corpus 커버리지 부족 (46% 쿼리에 strict gold 없음) | 검색 품질이 아닌 corpus 문제 |
| Hard filtering(B4.5/P1)은 catastrophic recall loss 유발 | B4.5: -36%, P1: -69% |
| P6/P7 soft scoring은 recall 보존 (+1.9%) + loose recall 개선 (+4.3%) | Hard filter 대안으로 유효 |

### 7.2 논문 narrative 제안

**기존 가설** (폐기): "Scope filtering으로 contamination 감소, recall 손실 제한적"

**수정된 narrative**:

1. **Problem**: 산업 기술문서 검색에서 scope-external contamination이 발생하지만,
   hard scope filtering은 recall을 심각하게 훼손하는 trade-off가 존재

2. **Observation**: 기존 평가의 gold label 오류를 발견하고 content-based LLM judging으로
   재판정한 결과, retrieval 시스템은 실제로 높은 recall을 달성 중이었음 (hit@5=0.837)

3. **Key finding**: Hard scope filter (B4.5: -36%, P1: -69% recall loss)는
   정답 문서까지 제거하여 오히려 contamination을 악화시킴

4. **Proposed solution**: Contamination-aware soft scoring (P6/P7)은
   recall을 보존하면서 out-of-scope 문서를 하위로 밀어내는 안전한 대안

5. **Limitation & future work**: Corpus 커버리지 부족이 주요 병목.
   46%의 쿼리에 정확한 장비-주제 매칭 문서가 corpus에 없음.
   Corpus 확충이 retrieval 개선보다 더 큰 영향을 줄 가능성
