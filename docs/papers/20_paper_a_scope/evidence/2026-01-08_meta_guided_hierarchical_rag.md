# Meta-guided Hierarchical RAG with Robust Routing

> **목적**: 제조/정비 도메인에서 메타데이터를 계층 인덱싱의 상위 prior로 사용하고,
> 메타 결측/오분류에 강인한 확률적 soft routing 기반 robust retrieval 시스템

**작성일**: 2026-01-08
**상태**: 설계 완료, 구현 대기

---

## 1. 핵심 기여점 (저널급)

### Main Contributions
1. **Meta-prior Routing with Probabilistic Soft Membership**
   - 메타데이터를 latent group의 prior로 사용
   - 메타 결측을 missing observation으로 모델링 → posterior p(z|x, m_obs) 근사
   - DP mixture 관점에서 novelty detection 정당화

2. **Local RAPTOR with Robust Indexing**
   - 각 meta-group 내부에서 RAPTOR 트리 구축
   - 요약 노드 신뢰성 검증 (NLI 기반 evidence linking)

### Extended Contributions (Appendix)
- Online/Offline 운영 전략
- Cross-group similarity links
- Adaptive threshold tuning

---

## 2. 용어 정의

| 용어 | 정의 | 예시 |
|------|------|------|
| **meta-group** | 메타데이터로 나뉜 최상위 버킷 | "SUPRA_XP_sop", "EFEM_ts" |
| **raptor-cluster** | meta-group 내부에서 GMM이 생성한 클러스터 | 요약 부모 노드 단위 |
| **None-pool** | 메타 결측/신뢰도 낮은 샘플의 대기 버킷 | soft routing 전 임시 저장 |
| **group-edge** | leaf와 meta-group 간 소속 관계 | (leaf_id, group_id, weight, type) |

---

## 3. 아키텍처

```
                    ┌─────────────────────────────────────────┐
                    │            Query Router                  │
                    │   p(g|q) = softmax(β·sim + α·meta_match) │
                    └─────────────────────────────────────────┘
                              │ (mixture-of-experts)
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Meta Group:   │    │ Meta Group:   │    │  None Pool    │
│ SUPRA_XP_sop  │    │ EFEM_ts       │    │  (soft links) │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
   Local RAPTOR          Local RAPTOR         Pending →
   Tree (L0-L3)          Tree (L0-L3)         Offline RAPTOR
        │                     │
        └──────── cross-group similarity links ────────┘

Leaf Storage: (leaf_id, group_id, weight) edges (no duplication)
```

### 핵심 개선점

| 기존 설계 | 개선된 설계 |
|----------|------------|
| None → 새 군집 무조건 생성 | Novelty detection (τ 임계값) 기반 조건부 생성 |
| 메타 있으면 hard 고정 | Soft escape hatch (semantic mismatch 시 보조 군집 허용) |
| None에만 RAPTOR | 모든 메타 군집에 Local RAPTOR + None Pool에 Offline RAPTOR |
| 데이터 복제로 다중 소속 | Edge 기반 (leaf_id, group_id, weight) - 저장 비용 절감 |

---

## 4. 핵심 수식

### 4.1 생성모형 (Generative Formulation)

```
z: latent group (meta-group)
x: text embedding (observed)
m: metadata (partially observed - can be missing)

Generative process:
1. z ~ p(z)                    # group prior
2. x | z ~ N(μ_z, Σ_z)         # embedding given group
3. m | z ~ Cat(θ_z)            # metadata given group

Posterior inference:
p(z | x, m_obs) ∝ p(x | z) · p(m_obs | z) · p(z)

When m is missing:
p(z | x) ∝ p(x | z) · p(z)     # naturally marginalizes out m
```

**Log-linear 근사**:
```
score(g | leaf_i) = β · sim(e_i, c_g) + Σ_j α_j · match(meta_j) + b_g
```

### 4.2 Novelty Detection (DP Mixture)

```
Chinese Restaurant Process 해석:
- 기존 group g에 배정될 확률 ∝ n_g (그룹 크기)
- 새 group 생성 확률 ∝ α (concentration parameter)

적응형 임계값:
τ_g = quantile(similarities_g, p=0.05)  # per-group 5th percentile

novelty_score(x_i) = max_g { (sim(x_i, c_g) - τ_g) / σ_g }
if novelty_score < 0: create_new_group()
```

### 4.3 Soft Escape Hatch (메타 오분류 대응)

```
if meta_exists(leaf_i):
    primary_group = meta_group(leaf_i)
    escape_score = (sim(e_i, c_primary) - μ_primary) / σ_primary

    if escape_score < -k:  # z-score 기반 탈출 조건
        secondary_groups = top_k_by_posterior(leaf_i, exclude=primary)
        add_secondary_links(leaf_i, secondary_groups)
```

### 4.4 구조 Prior (절차형 문서 특화)

```
score(g | leaf_i) = β · sim(e_i, c_g)
                  + Σ_j α_j · match(meta_j(i), meta_j(g))
                  + γ · adjacency_prior(i, g)      # 연속 청크 유도
                  + δ · doc_scope_prior(i, g)      # 문서 범위 제약
                  + κ · step_coherence(i, g)       # 절차 패턴 점수
                  + b_g
```

### 4.5 복잡도 분석

```
Naive softmax over all G groups: O(G · d)

Optimized (2-stage):
1. 메타 있음 → primary group + escape top-k: O(k · d)
2. 메타 없음 → ANN으로 centroid top-M 후보: O(log G + M · d)

Space: O(N + E) where N = leaves, E = edges (no duplication)
```

---

## 5. 구조적 리스크 및 대응

### RAPTOR 구조적 리스크

| 리스크 | 설명 | 원인 |
|--------|------|------|
| **Cascading Error** | 초기 군집 오류 → 요약 오류 → 상위 레벨 오류 누적 | 트리 구축의 확정적 결정 |
| **절차 분절** | 중간 단계만 있는 청크가 다른 장비 절차와 혼동 | 페이지/표 분할 + 의미 임베딩 한계 |
| **메타 오분류** | 잘못된 메타로 hard partition → recall 붕괴 | 메타 품질 문제 |

### 군집 품질 진단 지표

| 지표 | 정의 | 의심 조건 |
|------|------|----------|
| `cluster_cohesion` | mean(sim(leaf, centroid)) | < threshold |
| `outlier_rate` | ratio of sim < outlier_τ | > 20% |
| `meta_conflict` | 장비군/고장유형 혼재 비율 | 단일이어야 하는데 섞임 |
| `step_discontinuity` | 절차 번호 불연속/역행 | 연속 step이 분절됨 |
| `summary_support` | 요약 문장의 entailment 비율 | < 70% |
| `self_retrieval` | 요약으로 자식 검색 시 재현율 | 자식이 안 나옴 |

### 수정 프로세스 (Repair Loop)

```
1. Outlier 이동 (경량)
   - 의심 군집에서 outlier leaf 추출
   - top-k 대체 군집으로 재라우팅
   - 해당 군집 요약만 재생성

2. Split/Merge (구조적)
   - 의심 군집 내부 재클러스터링 → split
   - 유사한 두 군집 → merge
   - 변경된 부분만 요약 재생성 + 상위 전파

3. 제약 기반 재클러스터링 (절차 특화)
   - must-link: 같은 절차명/문서/연속 step
   - cannot-link: 메타 확정된 다른 장비군
```

---

## 6. 구현 파일 구조

```
backend/llm_infrastructure/raptor/
├── __init__.py
├── schemas.py              # RaptorNode, Partition, GroupEdge 등
├── partition.py            # 메타데이터 기반 파티셔닝
├── clustering.py           # GMM 클러스터링 (UMAP 차원축소)
├── tree_builder.py         # Local RAPTOR 트리 구축
├── soft_router.py          # Soft membership + novelty detection (핵심)
├── query_router.py         # 쿼리 라우팅 (mixture-of-experts)
├── summary_validator.py    # 요약 품질 검증 (entailment)

backend/llm_infrastructure/elasticsearch/
├── raptor_mappings.py      # RAPTOR 전용 ES 매핑

backend/llm_infrastructure/retrieval/adapters/
├── raptor_retriever.py     # RaptorHierarchicalRetriever

backend/services/
├── raptor_ingest_service.py    # Online 라우팅 서비스
├── raptor_rebuild_service.py   # Offline RAPTOR 재구성 서비스

scripts/evaluation/
├── raptor_evaluation.py    # 평가 + degradation 곡선
├── ablation_configs.py     # Ablation study 설정
```

---

## 7. 구현 순서

| 단계 | 기간 | 파일 | 내용 |
|------|------|------|------|
| 1 | Day 1-2 | schemas.py, raptor_mappings.py | 데이터 클래스, ES 매핑 |
| 2 | Day 3-5 | partition.py, clustering.py, soft_router.py, summary_validator.py | 코어 로직 |
| 3 | Day 6-7 | tree_builder.py | Local RAPTOR + cross-group 링크 |
| 4 | Day 8-9 | query_router.py, raptor_retriever.py | MoE 검색 |
| 5 | Day 10-11 | raptor_ingest_service.py, raptor_rebuild_service.py | Online/Offline 서비스 |
| 6 | Day 12-14 | raptor_evaluation.py, ablation_configs.py | 평가 |

---

## 8. 평가 설계

### 비교군 (Baselines)

| 비교군 | 설명 | 목적 |
|--------|------|------|
| **Flat Hybrid** | 메타 없이 dense+BM25 hybrid | 기본 베이스라인 |
| **Hard Meta Filter** | 메타로 필터링 후 flat retrieval | hard filter의 한계 노출 |
| **Global RAPTOR** | 메타 없이 전체에 RAPTOR 적용 | RAPTOR 자체 효과 분리 |
| **Meta + Flat** | 메타 파티션 + flat retrieval | 파티셔닝만의 효과 |
| **Meta + RAPTOR (hard)** | 메타 파티션 + local RAPTOR (soft 없음) | soft routing 효과 분리 |
| **Full System** | 메타 + local RAPTOR + soft + novelty | 제안 시스템 |

### Ablation 설정

| Config Name | Partition | RAPTOR | Soft | Novelty | Missing | Noise |
|-------------|-----------|--------|------|---------|---------|-------|
| baseline_hybrid | X | X | X | X | 0% | 0% |
| partition_only | O | X | X | X | 0% | 0% |
| partition_raptor | O | O | X | X | 0% | 0% |
| soft_no_novelty | O | O | O | X | 0% | 0% |
| **full_system** | O | O | O | O | 0% | 0% |
| missing_10pct | O | O | O | O | 10% | 0% |
| missing_30pct | O | O | O | O | 30% | 0% |
| missing_50pct | O | O | O | O | 50% | 0% |
| noise_10pct | O | O | O | O | 0% | 10% |
| noise_30pct | O | O | O | O | 0% | 30% |

### 핵심 그래프 (논문 Figure)

1. **Metadata Missing Rate vs Recall@10**
   - X축: 0%, 10%, 30%, 50%
   - 라인: hard_filter (급락), soft_membership (완만), full_system (안정)

2. **Metadata Noise Rate vs Recall@10**
   - 오분류가 있어도 soft escape로 회복

3. **Ablation Bar Chart**
   - 각 컴포넌트 제거 시 성능 하락 시각화

---

## 9. 도메인 특성

| 항목 | 답변 | 설계 반영 |
|------|------|----------|
| **절차 ID 추출** | 가능 (헤더/표 제목) | `step_coherence`, `doc_scope_prior` 적극 활용 |
| **메타 오분류율** | 5-15% | Soft escape hatch 필수, 적응형 임계값 필요 |
| **평가 중점** | **Precision 중심** | Reranking 강화, 오탐 최소화 전략 |

### Precision 중심 설계 조정

```
1. Reranking 강화
   - Cross-encoder reranker 필수 적용
   - meta_match + structure_prior 기반 2차 필터링

2. 오탐 최소화
   - soft routing 시 top-k를 작게 (k=2-3)
   - escape threshold를 보수적으로 (낮게)

3. 평가 지표 우선순위
   - Precision@5 > Recall@10 > MRR
   - 오탐률(false positive rate) 별도 측정
```

---

## 10. 하이퍼파라미터

| 파라미터 | 범위 | 영향 |
|----------|------|------|
| β (의미 유사도 가중치) | 0.5-2.0 | 메타 vs 의미 균형 |
| α (메타 필드 가중치) | 0.1-1.0 | 필드별 중요도 |
| τ (novelty threshold) | 0.2-0.5 | 새 군집 생성 빈도 |
| escape_threshold | 0.4-0.7 | soft escape 민감도 |
| top_k (soft routing) | 2-3 | 다중 소속 개수 (Precision 중심) |
| max_levels (RAPTOR) | 2-4 | 트리 깊이 |
| global_fallback_weight | 0.1-0.3 | recall 안전망 강도 |

---

## 11. Related Work

### 계층 요약 인덱싱 / 트리 기반 RAG
- **RAPTOR** [arXiv:2401.18059]: soft clustering + 요약 반복 트리
- **HiRAG** [EMNLP 2025]: 계층적 지식을 인덱싱/리트리벌에 반영

### 메타데이터 활용 RAG
- **Multi-Meta-RAG** [arXiv:2406.13213]: LLM으로 메타 추출 → DB 필터링
- **Metadata-Driven RAG** [arXiv:2510.24402]: 메타를 contextual chunk에 주입

### 라우팅 / Mixture-of-Experts
- **RouterRetriever** [arXiv:2409.02685]: 도메인별 임베딩 expert 라우팅

### 평가/벤치마킹
- **RAGAS** [arXiv:2309.15217]: reference-free RAG 평가
- **HiChunk** [arXiv:2509.11552]: hierarchical chunking 벤치마크

### 요약 신뢰성
- **SummaC** [TACL 2022]: NLI 기반 factual consistency 검출

### 제조/정비 도메인
- **NASA ASRS** [data.gov]: 항공 정비 보고서 공개 데이터
- **LogQA** [arXiv:2303.11715]: 비정형 로그 QA 벤치마크

---

## 12. 공개 데이터 후보

| 데이터셋 | 설명 | 용도 |
|----------|------|------|
| **NASA ASRS** | 항공 정비 보고서 | 정비 도메인 프록시 |
| **Aircraft Maintenance** [Kaggle] | 항공기 정비 이력 | 메타(failure code) 포함 |
| **LogQA** | 로그 QA 벤치마크 | 평가 프레임 참고 |
| **내부 데이터** | SOP/정비로그 | 실제 성능 검증 (비공개 OK) |
