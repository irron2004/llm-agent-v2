# Paper A 진행 경과 및 결론 정리

Date: 2026-03-13
Status: Breakthrough — masked query 실험으로 thesis 입증 가능성 확인

---

## 1. Paper A 목표

> 반도체 유지보수 RAG에서 cross-equipment contamination(타 장비 문서 혼입)을 줄이되,
> 공용 SOP/유사 장비로 인한 recall 손실을 최소화하는 스코프 정책을 설계·검증한다.

**핵심 주장**: Device-aware scope filtering이 contamination을 제거하면서 recall을 유지(또는 향상)시킨다.

---

## 2. 진행 경과

### Phase 0 (기존 eval set 구축)
- `query_gold_master_v0_5.jsonl`: 472 queries (421 dev + 51 test)
- 문서를 보고 질문 생성 → gold = source doc

### Phase 1 — Content-based LLM Judging (2026-03-09)
- TREC pooling: B0~P1 시스템의 top-k 결과를 pool
- 2,077 (query, doc) 쌍을 LLM judge로 판정 (relevance 0/1/2)
- 결과: grade 0 = 80.6%, grade 1 = 16.8%, grade 2 = 2.6%

### Phase 2 — Metrics Recalculation
- 새 gold label 생성: strict (grade=2) + loose (grade≥1)
- 108 queries에 대한 system별 메트릭 재계산
- **문제 발견**: strict gold 없는 query가 67% (corpus coverage 부족)

### Phase 3 — Eval Set Expansion
- 109 dev queries 추가 (20+ devices, balanced distribution)
- Phase 3 결과에서 scope filtering 효과 미미:
  - B3 (no filter) hit@5 = 0.423
  - B4.5 (hard filter) hit@5 = 0.269 (**-36%**)
  - P1 (scope policy) hit@5 = 0.131 (**-69%**)
- **결론**: scope filtering이 recall을 파괴하는 것으로 보임

### Phase 4 — Soft Scoring (P6/P7)
- Contamination-aware scoring: `Score(d,q) = Base(d,q) - λ·v_scope(d,q)`
- P6 (λ=0.05): hit@5 = 0.431 (+1.9%)
- P7 (adaptive λ): hit@5 = 0.431 (+1.9%)
- **결론**: soft scoring도 개선 미미

### 근본 원인 진단 (2026-03-12)

Phase 1-4의 부정적 결과가 나온 이유를 분석:

1. **Gold label circular bias**: 문서→질문 방향으로 생성 → gold이 source doc에 편향
2. **Doc_id에 device명 인코딩**: `global_sop_supra_xp_all_*` → BM25가 lexical match로 이미 찾음
3. **Gold set collapse**: 98.5%의 query가 동일 device 내 gold를 공유 (SUPRA_XP 34% 집중)
4. **Scope filter marginal value ≈ 0**: 71.2%의 query에서 scope filter 불필요

**핵심 모순**: Contamination은 실제로 심각한데(B3 cont@5 = 62.6%), 현재 gold로는 측정 불가

### Dataset Protocol Redesign (2026-03-12)

`docs/papers/20_paper_a_scope/evidence/2026-03-12_dataset_protocol_redesign.md`:
- 3-channel 후보 생성 (pool bias 해소)
- `--gold-mode strict|loose|both` 옵션 추가
- `strict_eligible` 분리 (coverage confounding 해소)
- Multi-axis annotation schema (topical_relevance × scope_correctness)
- `question_masked` 필드 도입 (device명 → `[DEVICE]`/`[EQUIP]` 치환)

### v0.6 Generated Eval Set (2026-03-12)

- `query_gold_master_v0_6_generated_full.jsonl`: **578 queries**
  - 27 devices에 골고루 분포
  - unique gold set: 482/578 (83%) — collapse 문제 해소
  - `question_masked` 포함: device명이 `[DEVICE]`/`[EQUIP]`로 마스킹
  - `gold_doc_ids` (loose) + `gold_doc_ids_strict` 분리

---

## 3. Breakthrough: Masked Query 실험 (2026-03-13)

### 실험 설계
- 578 queries × 4 conditions:
  - **B0_orig**: BM25 no filter, 원본 질문 (device명 포함)
  - **B0_masked**: BM25 no filter, 마스킹 질문 (`[DEVICE]`/`[EQUIP]`)
  - **B4_masked**: BM25 + hard device filter, 마스킹 질문
  - **B4.5_masked**: BM25 + device + shared, 마스킹 질문

### 결과

#### Contamination@10

| Condition | explicit_device | explicit_equip | ALL |
|-----------|:-:|:-:|:-:|
| B0 orig (baseline) | 0.381 | 0.957 | **0.529** |
| B0 masked (no filter) | 0.352 | 0.996 | **0.518** |
| B4 masked (device filter) | 0.000 | 0.000 | **0.000** |
| B4.5 masked (device+shared) | 0.000 | 0.000 | **0.000** |

→ Device filter가 contamination을 **52% → 0%**로 완전 제거

#### Gold Hit Rate (loose)

| Condition | explicit_device | explicit_equip | ALL |
|-----------|:-:|:-:|:-:|
| B0 orig | 366/429 (85%) | 53/149 (36%) | 419/578 (72%) |
| B0 masked | 326/429 (76%) | 17/149 (11%) | 343/578 (59%) |
| **B4 masked** | **417/429 (97%)** | **118/149 (79%)** | **535/578 (93%)** |
| B4.5 masked | 370/429 (86%) | 97/149 (65%) | 467/578 (81%) |

→ **Device filter가 recall을 파괴하지 않고 오히려 +34%p 향상** (59% → 93%)

#### Device별 B4 masked gold hit (loose)

| Device | Hit Rate |
|--------|----------|
| TIGMA Vplus | 50/50 (100%) |
| SUPRA N series | 20/20 (100%) |
| OMNIS plus | 12/12 (100%) |
| INTEGER plus | 90/91 (99%) |
| GENEVA XP | 68/69 (99%) |
| SUPRA N | 80/82 (98%) |
| PRECIA | 65/68 (96%) |
| ZEDIUS XP | 38/41 (93%) |
| SUPRA XP | 24/26 (92%) |
| SUPRA Vplus | 58/88 (66%) |

### 해석

이전 Phase 1-4에서 "scope filtering이 recall을 파괴한다"고 나왔던 이유:
- 질문에 이미 장비명이 있어서 BM25가 lexical match로 correct doc을 찾고 있었음
- Scope filter를 걸면 이미 맞는 결과에서 일부가 빠져서 recall이 떨어지는 것처럼 보임

**장비명을 마스킹하면 실제 운영 환경에 더 가까운 시나리오**:
- PE 엔지니어가 "Controller 교체 절차 알려줘"라고 물으면 → 어떤 장비인지 모름
- B0는 여러 장비의 Controller SOP를 섞어서 반환 (contamination 52%)
- B4는 context에서 추론한 장비의 문서만 반환 → gold hit 93%

---

## 4. 현재 한계

### 4.1 실험 범위 한계
- **BM25-only**: Dense/Hybrid+Rerank 실험은 embedding 차원 불일치로 미실시
  - `chunk_v3_content`: text만 있고 embedding 없음
  - `chunk_v3_embed_bge_m3_v1`: 1024-dim embedding만 있고 text/doc_id 없음
  - `rag_chunks_dev_current`: 768-dim (이전 모델) + text 있음
  - → cross-index 조합 또는 새 통합 인덱스 필요

### 4.2 Question Masking의 한계
- `[DEVICE]`/`[EQUIP]` 토큰이 BM25에서 매칭되지 않는 것은 당연
- 실제 운영에서는 장비명이 없는 것이 아니라 **context에서 추론해야 함**
- 마스킹은 "scope filtering이 필요한 시나리오"의 upper bound를 보여줌

### 4.3 Device Filter의 Oracle 가정
- 현재 B4는 **정답 device를 알고 있는 상태**에서 필터링 (oracle filter)
- 실제 시스템에서는 query에서 device를 파싱/추론해야 함
- Parser accuracy가 전체 파이프라인 성능을 결정

### 4.4 Gold Label 신뢰도
- v0.6 gold는 자동 생성 → LLM judge 검증 미완료
- strict gold의 정확도/coverage 미확인

### 4.5 Scope Observability 분포 편향
- v0.6에 implicit/ambiguous 없음 (explicit_device 429 + explicit_equip 149만)
- implicit/ambiguous가 실제 운영에서 중요한 케이스인데 eval에 없음

---

## 5. 논문 방향 제안

### 강한 claim (데이터가 뒷받침)
1. **Cross-equipment contamination은 심각하다** — BM25 기준 52.9% (explicit_equip에서 95.7%)
2. **Device-aware hard filter는 contamination을 완전 제거한다** — 0%
3. **Hard filter가 recall을 향상시킨다** — masked query에서 59% → 93% (+34%p)
4. **Masking이 evaluation bias를 드러낸다** — 기존 eval에서 scope filter가 무효해 보인 건 gold bias 때문

### 주의가 필요한 claim
- "Soft scoring이 hard filter보다 우수하다" → Phase 4에서 +1.9%만 나옴, 재실험 필요
- "Shared document policy가 recall을 보존한다" → B4.5가 B4보다 낮음 (93% → 81%), 역설적
- "모든 query에서 효과적이다" → implicit/ambiguous 미측정

### 논문 RQ 제안
- **RQ1**: 산업 RAG에서 cross-equipment contamination은 얼마나 심각한가? → 매우 심각 (52-96%)
- **RQ2**: Device-aware scope filtering이 contamination과 recall에 미치는 영향은? → contamination 제거 + recall 향상
- **RQ3**: 기존 "문서→질문" 평가 방식이 scope filtering 효과를 어떻게 왜곡하는가? → gold bias로 marginal value ≈ 0으로 과소추정

---

## 6. 즉시 진행 가능한 다음 작업

1. **Hybrid+Rerank 실험** — cross-index 또는 통합 인덱스로 B1~B3+filter 실험
2. **implicit/ambiguous query 추가** — v0.6에 없는 scope_observability 보완
3. **Parser accuracy 측정** — oracle filter vs real parser의 갭 정량화
4. **LLM judge로 gold 검증** — v0.6 strict gold의 정확도 확인
5. **P6/P7 재실험** — masked query + new gold로 soft scoring 재평가

---

## 7. 파일 위치

| 파일 | 설명 |
|------|------|
| `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl` | v0.6 eval set (578q, masked 포함) |
| `data/paper_a/eval/query_gold_master_v0_6_generated_full_strict.jsonl` | v0.6 strict gold 버전 |
| `data/paper_a/trap_masked_results.json` | Masked query 실험 결과 (578q × 4 conditions) |
| `data/paper_a/eval/trap_pilot_v1.jsonl` | Counterfactual trap pilot (20q) |
| `docs/papers/20_paper_a_scope/evidence/2026-03-12_dataset_protocol_redesign.md` | 평가 프로토콜 재설계 문서 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-12_cross_device_topic_feasibility.md` | Cross-device topic 분석 |
| `.sisyphus/evidence/paper-a/corpus/cross_device_trap_candidates.json` | Trap 후보 68 topics |
| `docs/papers/20_paper_a_scope/evidence/2026-03-09_gold_rejudging_analysis.md` | Phase 1-4 분석 리포트 |
| `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` | Paper A 실험 정의서 v0.6 |
