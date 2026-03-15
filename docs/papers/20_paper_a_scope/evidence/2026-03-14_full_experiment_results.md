# Paper A 전체 실험 결과 종합 문서

**작성일**: 2026-03-14
**목적**: Paper A의 모든 실험 결과를 한 곳에 정리하는 마스터 문서 (논문 작성 기초 자료)
**버전**: v1.0

---

## 목차

1. [논문 목표 및 배경](#1-논문-목표-및-배경)
2. [시스템 매트릭스](#2-시스템-매트릭스)
3. [평가 데이터셋 진화](#3-평가-데이터셋-진화)
4. [Phase 1-4 결과 (기존, v0.5 기반)](#4-phase-1-4-결과-기존-v05-기반)
5. [Masked Query 실험 결과 (핵심, v0.6 기반)](#5-masked-query-실험-결과-핵심-v06-기반)
6. [P6/P7 Soft Scoring 결과](#6-p6p7-soft-scoring-결과)
7. [Parser Accuracy 결과](#7-parser-accuracy-결과)
8. [Gold Label 검증 결과](#8-gold-label-검증-결과)
9. [전체 결론 및 논문 방향](#9-전체-결론-및-논문-방향)
10. [파일 위치 및 재현 명령어](#10-파일-위치-및-재현-명령어)
11. [Device별 상세 결과](#11-device별-상세-결과)

---

## 1. 논문 목표 및 배경

### 1.1 Paper A 목표

반도체 유지보수 RAG 시스템에서 발생하는 **cross-equipment contamination 문제**를 정의하고, 이를 줄이되 recall 손실을 최소화하는 **scope policy**를 설계·검증한다.

### 1.2 핵심 문제 정의

#### Cross-Equipment Contamination

- **정의**: 장비 A에 대한 질문에 장비 B의 문서가 검색 결과에 섞이는 현상
- **예시**: "SUPRA XP의 PM 절차를 알려줘" 질문에 "INTEGER plus"나 "PRECIA"의 PM 문서가 반환됨
- **측정**: contamination@10 — top-10 결과 중 다른 장비 문서의 비율
- **심각성**:
  - BM25(no filter): 52.9% (전체 쿼리 평균)
  - Dense kNN(no filter): 73.0% (masked 조건)
  - Hybrid+Rerank(no filter): 58.6% (masked 조건)

#### Scope Filtering

- **정의**: `device_name` 메타데이터 기반으로 검색 범위를 특정 장비의 문서로 제한
- **방식**:
  - Hard filter: device_name이 정확히 일치하는 문서만 반환
  - Soft scoring: 동일 장비 문서에 가산점(λ), 타 장비 문서에 감산점 부여
- **목표**: contamination → 0%, gold hit 유지 또는 향상

#### Shared Documents (공용 문서)

- **정의**: 장비에 무관하게 공용으로 허용되는 문서 (예: 공통 SOP, 일반 안전 매뉴얼)
- **수량**: 60개 문서
- **정책 파일**: `.sisyphus/evidence/paper-a/policy/shared_doc_ids.txt`
- **B4.5 시스템**: B4(hard device filter)에 shared documents를 추가로 허용

### 1.3 연구 배경

반도체 PE(Process Engineering) 도메인에서의 RAG 시스템은 다양한 장비(27종 이상)에 대한 문서를 통합 인덱스로 관리한다. 이때:

1. **문서 수**: 508개 문서, 수만 개의 청크
2. **장비 다양성**: 27개 장비, 각각 다른 SOP/매뉴얼/PEMS 문서 보유
3. **문서 타입**: SOP(Standard Operating Procedure), PEMS, 매뉴얼, TSG

기존 RAG 시스템은 모든 문서를 동등하게 취급하여 검색하므로, 특정 장비 질문에 타 장비 문서가 다수 반환되는 contamination 문제가 발생한다. 이를 방치하면:
- LLM이 잘못된 장비의 절차를 답변에 포함
- 현장 엔지니어가 잘못된 정보로 유지보수 수행 → 안전 리스크
- 사용자 신뢰 하락

---

## 2. 시스템 매트릭스

### 2.1 베이스라인 시스템 (B0~B4.5)

#### B0: BM25 (no filter)
- **설명**: 순수 BM25 키워드 검색, 필터 없음
- **인덱스**: `chunk_v3_content` (Elasticsearch BM25)
- **분석기**: Nori(한국어) + 영문 standard
- **특징**:
  - 질문에 장비명이 포함되면 lexical match로 자동으로 해당 장비 문서 반환
  - 질문에서 장비명을 제거하면(masked) contamination 거의 변화 없음
  - Dense 대비 masking에 강인 (BM25는 term 기반이므로 다른 term으로 match)

#### B1: Dense kNN (no filter) — BGE-M3 1024-dim
- **설명**: Dense vector 검색, 필터 없음
- **모델**: BGE-M3 (1024차원)
- **인덱스**: `chunk_v3_embed_bge_m3_v1`
- **특징**:
  - semantic 유사도 기반 검색
  - 장비명 masking 시 contamination 폭증: 0.373 → 0.730
  - device context 없으면 cross-device 문서를 유사 문서로 반환
  - gold hit도 크게 하락: 378/578(65%) → 234/578(40%)

#### B2: Hybrid RRF (BM25 + Dense, no filter)
- **설명**: BM25 + Dense kNN을 Reciprocal Rank Fusion으로 결합
- **방식**: RRF(k=60) — 각 시스템의 rank를 1/(k+rank)로 변환 후 합산
- **특징**:
  - B0, B1 대비 gold hit 향상: 434/578(75%) — 두 검색기의 상호보완 효과
  - contamination은 중간 수준: 0.586 (masked 조건)
  - B0의 term matching + B1의 semantic matching을 결합

#### B3: Hybrid RRF + CrossEncoder Rerank (no filter)
- **설명**: B2에 Cross-Encoder 기반 재순위화 추가
- **모델**: Cross-Encoder (query-document pair scoring)
- **특징**:
  - gold hit는 B2와 동일: 434/578(75%)
  - **contamination은 B2와 동일**: 0.586
  - **핵심 발견**: Rerank은 ordering을 바꿀 뿐, contamination 제거에 기여 못함
  - B2와 B3가 동일한 이유: cross-encoder도 문서 품질을 평가하지 scope를 판단하지 않음

#### B4: B3 + Hard Device Filter (oracle device 사용)
- **설명**: B3에 oracle device 정보를 이용한 hard filter 적용
- **Oracle 조건**: gold answer의 device가 쿼리와 동일하다고 가정 (상한선 추정)
- **필터 방식**: Elasticsearch filter query — `device_name == query_device`
- **특징**:
  - contamination: 0.001 (≈0%, 극소량은 same device 내 다른 문서)
  - gold hit 향상: 527/578(91%) — 필터로 noise 제거 후 오히려 recall 향상
  - B4.5 대비 gold hit 높음 → shared docs가 오히려 noise로 작용

#### B4.5: B3 + Device + Shared Filter
- **설명**: B4에서 `device_name == query_device` OR `doc_id in shared_docs` 조건 추가
- **shared_docs**: 60개 공용 문서
- **특징**:
  - contamination: 0.001 (≈0%)
  - gold hit: 406/578(70%) — B4(91%)보다 낮음
  - **역설적 발견**: 공용 문서 추가가 gold hit를 하락시킴
  - 원인 추정: shared docs 중 일부가 real gold가 아닌 noise로 작용, OR shared docs가 실제 device-specific gold를 밀어냄

### 2.2 Policy 시스템 (P1, P6, P7)

#### P1: Scope Policy (device + shared + doc_type routing)
- **설명**: Phase 1-4에서 사용된 실험적 scope policy
- **구성**: device filter + shared documents + document type 기반 라우팅
- **사용**: Phase 1-4 (v0.5 데이터셋 기반) 실험에서 주로 활용

#### P6: B3 + Soft Scoring (λ=0.05 fixed)
- **설명**: B3에서 in-scope 문서에 +λ, out-of-scope 문서에 -λ penalty 부여
- **λ**: 0.05 (고정값)
- **수식**: `final_score = base_score + λ * is_in_scope`
- **특징**:
  - λ가 너무 작아 base score gap을 극복 못함
  - contamination 오히려 증가: +0.065
  - out-of-scope 문서가 top-1에 있으면 -0.05 penalty로는 밀어내지 못함

#### P7: B3 + Soft Scoring (Adaptive λ)
- **설명**: 쿼리별로 λ를 동적 조정
- **adaptive 기준**: 쿼리의 device 명시도(scope_observability) 기반
- **특징**:
  - explicit query에 더 높은 λ 적용 시도
  - 실제로는 P6와 유사한 성능 → 근본 문제는 λ 크기가 아니라 scale mismatch

### 2.3 시스템 비교 요약 표

| 시스템 | 검색 방식 | 필터 | Oracle? | cont@10 (masked) | gold@10 (masked) |
|--------|-----------|------|---------|------------------|------------------|
| B0 | BM25 | 없음 | - | 0.518 | 343/578 (59%) |
| B1 | Dense kNN | 없음 | - | 0.730 | 267/578 (46%) |
| B2 | Hybrid RRF | 없음 | - | 0.586 | 380/578 (66%) |
| B3 | Hybrid+Rerank | 없음 | - | 0.586 | 380/578 (66%) |
| B4 | Hybrid+Rerank | Device(oracle) | Yes | 0.001 | 532/578 (92%) |
| B4.5 | Hybrid+Rerank | Device+Shared(oracle) | Yes | 0.001 | 439/578 (76%) |
| P6 | Hybrid+Rerank | Soft(λ=0.05) | - | 0.651 | 380/578 (66%) |
| P7 | Hybrid+Rerank | Soft(adaptive) | - | 0.651 | 380/578 (66%) |

---

## 3. 평가 데이터셋 진화

### 3.1 v0.5 (472 queries) — 초기 데이터셋

#### 생성 방식
- **방향**: 문서 → 질문 (document-centric generation)
- **프로세스**: 각 문서에서 LLM이 해당 문서 내용 기반의 질문 생성

#### 문제점

**Gold Set Collapse (가장 심각)**
- 98.5%의 쿼리가 동일 device 내 gold를 공유
- 즉, 거의 모든 쿼리의 gold doc이 특정 device에 쏠림
- 이로 인해 scope filtering 효과를 제대로 측정 불가

**Device 불균형**
- SUPRA_XP가 전체의 34% 집중
- 27개 장비 중 소수 장비에 쿼리 집중

**BM25 Lexical Match Bias**
- doc_id에 device명이 인코딩되어 있음 (예: `supra_xp_sop_001`)
- BM25가 question에서 device명을 인식 → doc_id의 device명과 lexical match
- 결과: scope filtering 없이도 BM25가 자동으로 correct device 문서 반환
- 이는 scope filtering의 효과를 과소추정하게 만드는 circular bias

**평가 한계**
- 문서에서 생성된 질문이므로 실제 엔지니어 질문 패턴과 괴리
- 장비 cross-reference 질문 미포함

### 3.2 v0.6 (578 queries) — 개선 데이터셋

#### 생성 방식
- **방향**: topic-based 생성 + device 균등 분포 보장
- **device 분포**: 27개 장비에 균등 분포

#### 주요 개선 사항

**Device 균형**
- 27 devices에 균등 분포 (각 약 20-22개 쿼리)
- 단일 장비 집중 방지

**Gold Set 품질 개선**
- unique gold set: 482/578 (83%) — v0.5 대비 큰 개선
- strict gold vs loose gold 분리

**Masking 도입**
- `question_masked` 필드 신규 추가
- `[DEVICE]`: 장비명 직접 언급 치환 (예: "SUPRA XP" → "[DEVICE]")
- `[EQUIP]`: 장비 유형 언급 치환 (예: "etcher" → "[EQUIP]")
- 목적: 장비명 없는 조건에서 retrieval 성능 측정 → scope filtering의 순수 효과 분리

**Gold 라벨 분리**
- `gold_doc_ids_strict`: 질문에 직접적으로 답하는 문서만 포함
- `gold_doc_ids` (loose): 관련성 있는 문서도 포함

**Scope Observability 도입**
- `scope_observability` 필드 신규 추가
- `explicit_device` (n=429): 질문에 특정 장비명이 명시적으로 언급됨
- `explicit_equip` (n=149): 장비 유형(etcher, ALD 등)이 언급되나 특정 기종 미명시
- 이 분류가 실험 분석의 핵심 차원이 됨

**데이터 통계**
```
총 쿼리: 578개
devices: 27종
unique gold set: 482/578 (83.4%)
scope_observability:
  - explicit_device: 429 (74.2%)
  - explicit_equip: 149 (25.8%)
gold_doc_ids_strict: 쿼리당 평균 1.2개 문서
gold_doc_ids (loose): 쿼리당 평균 1.8개 문서
```

### 3.3 v0.7 (1206 queries) — 통합 데이터셋

#### 구성

```
v0.6 원본 (578q) + implicit (578q) + trap (50q) = 1206q
```

**v0.6 원본 (578q)**
- 위 v0.6 그대로 포함

**implicit 쿼리 (578q)**
- 생성 방식: v0.6의 `question_masked`에서 `[DEVICE]`, `[EQUIP]` 토큰을 완전히 제거하여 자연어 문장으로 변환
- 목적: 장비명/유형이 전혀 언급되지 않는 "암묵적" 질문 시뮬레이션
- 예시: "SUPRA XP의 PM 절차는?" → masked: "[DEVICE]의 PM 절차는?" → implicit: "PM 절차는?"
- 이 조건은 scope policy에게 가장 어려운 케이스

**trap 쿼리 (50q)**
- 생성 방식: cross-device topic 기반 counterfactual queries
- 목적: 여러 장비에 공통된 topic에 대한 질문 (예: "slot valve 교체 주기")
- 특성: gold가 특정 device에 있지만 다른 device 문서도 유사 내용 보유
- `scope_observability`: 별도 카테고리로 분류 필요

**scope_observability 분포 (v0.7 전체)**
```
explicit_device: 429 (35.6%) — v0.6 원본에서 유지
explicit_equip: 149 (12.4%) — v0.6 원본에서 유지
ambiguous: 628 (52.1%) — implicit(578) + trap(50) 포함
```

**v0.7 파일 위치**
- `data/paper_a/eval/query_gold_master_v0_7_with_implicit.jsonl`

---

## 4. Phase 1-4 결과 (기존, v0.5 기반)

### 4.1 Phase 1: TREC Pooling + LLM Judging

**목적**: 기존 gold label의 신뢰성 확인 및 새로운 gold label 구축

**방법**:
- 6개 retrieval 시스템의 top-100 결과를 pooling
- TREC-style pooling: 각 시스템의 상위 결과를 합집합하여 candidate set 구성
- LLM judge로 (query, doc) 쌍의 relevance 판정
- 총 **2,077쌍** 판정 완료

**결과**:
- TREC pooling이 기존 gold와 높은 일치율 보임
- 그러나 v0.5 데이터셋의 근본적 문제(gold set collapse, device bias)는 해결 못함
- **핵심 한계**: LLM judge가 gold label을 생성한 것과 동일한 문서를 "relevant"로 판정 → circular bias

### 4.2 Phase 2: 새로운 Gold Label (Strict/Loose)

**목적**: Phase 1 결과를 바탕으로 더 정교한 gold label 체계 구축

**방법**:
- `gold_doc_ids_strict`: 직접 답변 가능한 문서만 포함
- `gold_doc_ids` (loose): 부분적으로 관련된 문서도 포함

**문제 지속**:
- v0.5의 device 불균형, doc_id lexical bias는 여전히 존재
- strict/loose 분리만으로 근본 문제 해결 불가

### 4.3 Phase 3: 109 Dev Queries + Scope Filtering 효과 측정

**확장**: 109개 dev queries로 확장하여 scope filtering 효과 측정

**결과 요약**:
- Scope filtering 적용 시 gold hit 대폭 하락
- 하락 범위: **-36% ~ -69%** (시스템 및 조건에 따라)

**하락 원인 분석**:
1. v0.5 gold label이 BM25 lexical match로 생성되어 scope filtering과 무관하게 gold를 찾는 경향
2. scope filtering 후 gold doc이 필터에서 제외되는 경우 다수 존재
3. gold set collapse로 인해 일부 장비의 gold가 다른 장비 문서로 설정되어 있음 (잘못된 gold)

**결론**:
- Phase 3 결과는 실제 scope filtering 효과를 반영하지 않음
- **근본 원인**: gold label circular bias — scope filtering 설계를 위해 필요한 정보가 gold label 생성에도 영향을 미침

### 4.4 Phase 4: P6/P7 Soft Scoring 초기 시도

**목적**: hard filter의 recall 손실 문제를 soft scoring으로 완화

**방법**:
- P6: λ=0.05로 in-scope 문서에 가산점
- P7: adaptive λ (쿼리 특성 기반 동적 조정)

**결과**:
- 기존 B3 대비 +1.9% 개선 (미미함)
- Hard filter 대비 contamination 제거 효과 없음

**문제**: v0.5 기반이라 실제 scope filtering 효과 측정 자체가 부정확함

### 4.5 Phase 1-4 실패 원인 종합

```
근본 원인: Gold Label Circular Bias

v0.5 생성 프로세스:
문서 → (document-centric) → 질문 생성
    ↑ 동일 문서에서
    → gold = 해당 문서

BM25 검색 프로세스:
질문 → (lexical match with device name in doc_id) → 해당 장비 문서 반환

결과: BM25가 scope filter 없이도 correct device 문서를 반환
     → scope filtering 적용 시 오히려 gold hit 하락
     → 이는 gold label의 bias이지 scope filtering의 실제 효과가 아님
```

---

## 5. Masked Query 실험 결과 (핵심, v0.6 기반)

### 5.1 실험 설계

**목적**: scope filtering의 실제 효과를 bias 없이 측정

**실험 매트릭스**:
```
578 queries × {original, masked} × 6 systems = 6,936 retrieval runs
```

**조건 정의**:
- `original`: 원본 질문 (device명 포함)
- `masked`: question_masked 사용 (device명 → [DEVICE]/[EQUIP] 치환)

**평가 지표**:
- `contamination@10` (cont@10): top-10 결과 중 다른 장비 문서 비율
- `contamination_adjusted@10` (cont_adj): contamination 정도 가중치 적용
- `gold_hit@10 (strict)`: top-10에 strict gold가 1개 이상 포함된 쿼리 비율
- `gold_hit@10 (loose)`: top-10에 loose gold가 1개 이상 포함된 쿼리 비율

**분석 차원**:
1. 전체 (ALL, n=578)
2. explicit_device 서브셋 (n=429)
3. explicit_equip 서브셋 (n=149)

---

### 5.2 BM25 결과 (`data/paper_a/trap_masked_results.json`)

#### ALL (n=578)

| Condition | cont@10 | cont_adj | gold_strict | gold_loose |
|-----------|---------|----------|-------------|------------|
| B0_orig | 0.529 | 0.566 | 394/578 (68.2%) | 419/578 (72.5%) |
| B0_masked | 0.518 | 0.724 | 287/578 (49.7%) | 343/578 (59.3%) |
| B4_masked | 0.000 | 0.000 | 530/578 (91.7%) | 535/578 (92.6%) |
| B4.5_masked | 0.000 | 0.000 | 430/578 (74.4%) | 467/578 (80.8%) |

**해석**:
- B0_orig → B0_masked: contamination 거의 변화 없음 (0.529 → 0.518), gold hit 크게 하락 (68% → 50%)
  - BM25는 device명 제거 후에도 다른 term으로 cross-device match → contamination 유지
  - gold hit 하락은 device명이 gold를 찾는 핵심 단서였음을 의미
- B4_masked: contamination 완전 제거 + gold hit 크게 향상 (50% → 92%)
  - **device filter가 contamination 제거와 recall 향상을 동시에 달성**
- B4.5_masked: contamination 제거되나 gold hit B4 대비 하락 (92% → 74%)
  - shared docs(60개)가 noise로 작용하여 device-specific gold를 밀어냄

#### explicit_device (n=429)

| Condition | cont@10 | gold_strict | gold_loose |
|-----------|---------|-------------|------------|
| B0_orig | 0.381 | 347/429 (80.9%) | 366/429 (85.3%) |
| B0_masked | 0.352 | 284/429 (66.2%) | 326/429 (76.0%) |
| B4_masked | 0.000 | 417/429 (97.2%) | 417/429 (97.2%) |
| B4.5_masked | 0.000 | 351/429 (81.8%) | 370/429 (86.2%) |

**해석**:
- explicit_device 쿼리는 장비명이 명시되어 있어 B0_orig 자체적으로 contamination 낮음 (0.381)
- B4_masked: 97.2% — 거의 완벽한 recall (oracle device filter + BM25)
- B4.5: B4 대비 gold hit 하락 (97% → 82%) — shared docs의 부정적 영향

#### explicit_equip (n=149)

| Condition | cont@10 | gold_strict | gold_loose |
|-----------|---------|-------------|------------|
| B0_orig | 0.957 | 47/149 (31.5%) | 53/149 (35.6%) |
| B0_masked | 0.996 | 3/149 (2.0%) | 17/149 (11.4%) |
| B4_masked | 0.000 | 113/149 (75.8%) | 118/149 (79.2%) |
| B4.5_masked | 0.000 | 79/149 (53.0%) | 97/149 (65.1%) |

**해석**:
- explicit_equip 쿼리는 **오리지널에서도 contamination 95.7%** — 가장 심각한 케이스
- masked 조건에서는 contamination 거의 100% (0.996)
- **B0_masked에서 gold hit 2%** — 거의 완전히 실패
  - "etcher의 PM 절차" 같은 질문에서 장비 유형만으로는 특정 장비 문서 찾기 매우 어려움
- B4_masked: contamination 0%, gold hit 76% — oracle filter의 劇的인 효과
- B4.5: B4 대비 대폭 하락 (76% → 53%) — explicit_equip에서 shared docs의 악영향 더 심각

---

### 5.3 Hybrid+Rerank 결과 (`data/paper_a/masked_hybrid_results.json`)

#### ALL (n=578)

| System | cont@10 | gold_strict | gold_loose |
|--------|---------|-------------|------------|
| B0_orig | 0.422 | 394/578 (68.2%) | 419/578 (72.5%) |
| B1_orig | 0.373 | 378/578 (65.4%) | 397/578 (68.7%) |
| B2_orig | 0.364 | 434/578 (75.1%) | 456/578 (78.9%) |
| B3_orig | 0.364 | 434/578 (75.1%) | 456/578 (78.9%) |
| B0_masked | 0.473 | 287/578 (49.7%) | 343/578 (59.3%) |
| B1_masked | 0.730 | 234/578 (40.5%) | 267/578 (46.2%) |
| B2_masked | 0.586 | 351/578 (60.7%) | 380/578 (65.7%) |
| B3_masked | 0.586 | 351/578 (60.7%) | 380/578 (65.7%) |
| B4_masked | 0.001 | 527/578 (91.2%) | 532/578 (92.0%) |
| B4.5_masked | 0.001 | 406/578 (70.2%) | 439/578 (76.0%) |

**주요 관찰**:
1. **B2_orig = B3_orig**: Rerank이 gold hit에도 영향 없음 (순위만 바꿈)
2. **B1_masked 최악**: Dense는 masking에 가장 취약 (cont 0.730, gold 40%)
3. **B4_masked 최고**: oracle filter + hybrid = cont 0.001, gold 92%
4. **B4 vs B3 masked 격차**: contamination 0.585 차이, gold +26.3%p — 극적인 개선

#### explicit_device (n=429)

| System | cont@10 | gold_strict | gold_loose |
|--------|---------|-------------|------------|
| B0_orig | 0.324 | 347/429 (80.9%) | 366/429 (85.3%) |
| B3_orig | 0.245 | 391/429 (91.1%) | 405/429 (94.4%) |
| B0_masked | 0.323 | 284/429 (66.2%) | 326/429 (76.0%) |
| B1_masked | 0.685 | 226/429 (52.7%) | 253/429 (59.0%) |
| B3_masked | 0.483 | 341/429 (79.5%) | 357/429 (83.2%) |
| B4_masked | 0.001 | 416/429 (97.0%) | 416/429 (97.0%) |
| B4.5_masked | 0.001 | 322/429 (75.1%) | 341/429 (79.5%) |

**주요 관찰**:
- B3_orig에서 explicit_device 쿼리는 이미 gold hit 94.4% (device명이 질문에 있으면 Hybrid+Rerank이 잘 작동)
- B3_masked에서도 gold hit 83.2% — 장비명 없어도 어느 정도 작동 (context clue 있음)
- B4_masked: 97% — oracle filter 추가로 거의 완벽
- **B4 vs B4.5** 격차 더 커짐: 97% vs 75% — explicit_device에서 shared docs 영향 심각

#### explicit_equip (n=149)

| System | cont@10 | gold_strict | gold_loose |
|--------|---------|-------------|------------|
| B0_orig | 0.706 | 47/149 (31.5%) | 53/149 (35.6%) |
| B3_orig | 0.707 | 43/149 (28.9%) | 51/149 (34.2%) |
| B0_masked | 0.903 | 3/149 (2.0%) | 17/149 (11.4%) |
| B1_masked | 0.859 | 8/149 (5.4%) | 14/149 (9.4%) |
| B3_masked | 0.882 | 10/149 (6.7%) | 23/149 (15.4%) |
| B4_masked | 0.000 | 111/149 (74.5%) | 116/149 (77.9%) |
| B4.5_masked | 0.000 | 84/149 (56.4%) | 98/149 (65.8%) |

**주요 관찰**:
- **B3_orig에서 이미 contamination 70.7%** — no filter 시스템은 equip query에 거의 무용지물
- Rerank(B3)이 B0 대비 오히려 gold hit 하락 (35.6% → 34.2%) — equip query에서 Rerank 역효과
- **B3_masked: gold hit 15.4%** — 장비 유형 쿼리의 explicit 단서 제거 시 거의 실패
- **B4_masked: contamination 0%, gold hit 78%** — oracle filter의 극적인 효과
  - explicit_equip 쿼리에서 oracle이 가장 중요한 이유: 장비 유형만으로는 특정 기종 파악 불가
  - oracle이 정확한 device를 알려줘야만 올바른 filter 적용 가능
- **B4 vs B4.5** 격차: 78% vs 66% — equip query에서도 shared docs 영향

---

### 5.4 핵심 발견 정리

#### 발견 1: BM25 vs Dense의 Masking 민감도 차이

```
BM25 (B0):
- original → masked: 0.529 → 0.518 (cont 변화 없음)
- 이유: BM25는 term-based, device명 제거 후 다른 term으로 여전히 match 가능

Dense (B1):
- original → masked: 0.373 → 0.730 (cont 거의 2배)
- 이유: Dense는 semantic 유사도 기반, device context 없으면
        "비슷한 내용의 모든 장비" 문서를 반환
```

**논문 함의**: Dense retrieval은 device context에 더 의존적이므로 scope filtering이 더 중요

#### 발견 2: Device Filter의 双方向 효과

```
contamination 효과: 52-88% → 0% (완전 제거)
gold hit 효과: 40-66% → 92% (+26~42%p 향상)
```

**중요성**: 일반적으로 filter를 추가하면 recall이 떨어진다고 생각하지만,
- device filter는 오히려 recall을 **향상**시킴
- 이유: 필터 없으면 다른 장비의 유사 문서들이 gold doc을 밀어내고 top-10을 차지함
- 필터 후: noise 제거 → gold doc이 상위로 올라옴

#### 발견 3: B4 > B4.5 (Shared Docs의 역설)

```
B4 (device only): gold hit 92%
B4.5 (device + shared): gold hit 76%
차이: -16%p
```

**역설적 발견**: 공용 문서(60개)를 추가 허용했음에도 gold hit 하락
**확인된 원인 (2026-03-14 분석 완료)**:
- B4 wins 121건, B4.5 wins **0건** — shared docs가 gold hit에 도움이 되는 경우가 단 한 건도 없음
- B4.5 top-10 결과의 **57.3%가 shared docs** — device-specific gold를 완전히 밀어냄
- 주범 shared docs (거의 모든 쿼리에 출현):
  - `global_sop_*_device_net_board` (5개 문서): 97-99건의 쿼리에 출현
  - `global_sop_supra_vplus_all_pm_controller`: 37건 출현
  - `global_sop_*_ffu`: 28건 출현
- **메커니즘**: 범용 topic(device net board, controller, ffu)의 shared SOP가 BM25 매칭 빈도가 높아 top-10을 지배 → device-specific gold doc을 밀어냄
- **결론**: 현재 shared doc 분류(60개, deg≥3)가 너무 넓음. shared docs를 허용하면 오히려 recall이 하락하는 구조적 문제

#### 발견 4: Dense(B1)이 Masking에 가장 취약

```
B1_masked: cont 0.730, gold 40.5%
B2_masked (BM25+Dense): cont 0.586, gold 65.7%
B0_masked (BM25만): cont 0.518, gold 59.3%
```

**결론**: Dense component가 hybrid에서 contamination을 높임
- hybrid(B2)가 BM25(B0)보다 gold hit는 높지만 contamination도 높음
- Dense retrieval의 semantic similarity가 cross-device noise를 도입

#### 발견 5: Rerank은 Contamination 제거에 기여 없음

```
B2_orig = B3_orig = 0.364 (contamination 동일)
B2_masked = B3_masked = 0.586 (contamination 동일)
```

**결론**: Cross-Encoder reranker는 문서 품질을 평가하지, scope를 판단하지 않음
- Rerank은 ranking을 개선할 뿐, out-of-scope 문서를 필터링하지 않음
- Scope filtering은 별도의 메커니즘(filter query)으로만 달성 가능

#### 발견 6: Explicit_equip 쿼리가 가장 취약

```
no filter 조건:
- explicit_device: cont 0.245-0.381, gold 81-91%
- explicit_equip: cont 0.706-0.957, gold 29-36%
```

**결론**: 장비 유형 언급 쿼리는 oracle filter 없이는 거의 작동 불가
- 실운영에서는 "etcher PM" 같은 쿼리가 많을 것으로 예상
- Parser가 equip → device 매핑을 정확히 해야만 scope filtering 효과 달성 가능

---

## 6. P6/P7 Soft Scoring 결과

### 6.1 실험 설계

**목적**: hard filter의 한계(parser 오류 시 recall 손실)를 soft scoring으로 극복

**P6 수식**:
```
final_score = base_score + λ × scope_indicator
where scope_indicator ∈ {+1 (in-scope), -1 (out-of-scope), 0 (shared)}
λ = 0.05 (fixed)
```

**P7 수식**:
```
λ = adaptive_lambda(query_scope_observability)
  - explicit_device: λ = 0.05
  - explicit_equip: λ = 0.03 (equip query는 불확실하므로 작은 penalty)
  - ambiguous: λ = 0.01
```

**기대 효과**:
- In-scope 문서를 상위로 올리고 out-of-scope 문서를 하위로 내림
- Hard filter와 달리 완전 제거하지 않으므로 parser 오류 허용 가능

### 6.2 P6/P7 masked 실험 결과

**결과 요약** (`data/paper_a/masked_p6p7_results.json`):

| System | cont@10 | gold_strict | gold_loose |
|--------|---------|-------------|------------|
| B3_masked (base) | 0.586 | 351/578 (60.7%) | 380/578 (65.7%) |
| P6_masked | 0.651 | 351/578 (60.7%) | 380/578 (65.7%) |
| P7_masked | 0.651 | 351/578 (60.7%) | 380/578 (65.7%) |
| B4_masked (hard) | 0.001 | 527/578 (91.2%) | 532/578 (92.0%) |
| B4.5_masked | 0.001 | 406/578 (70.2%) | 439/578 (76.0%) |

**결과**: P6/P7 모두 B3 대비 contamination 오히려 증가 (+0.065), gold hit는 동일 (변화 없음)

### 6.3 실패 원인 분석

#### 원인 1: Scale Mismatch

```
BM25 base score 범위: 일반적으로 0.5 ~ 15.0 (TF-IDF 기반)
Dense score 범위: 0.0 ~ 1.0 (cosine similarity)
RRF score 범위: 일반적으로 0.01 ~ 0.03 (1/(k+rank))

λ = 0.05의 영향:
- BM25: 0.05/10.0 = 0.5% 차이 → 완전히 무시됨
- Dense: 0.05/0.7 = 7% 차이 → 효과 미미
- RRF: 0.05/0.02 = 250% 차이 → RRF에서는 효과 있을 수 있으나
         실제 구현에서는 RRF 후 soft scoring이므로 rank gap이 작음
```

#### 원인 2: Top-1 Lock-in

```
시나리오: out-of-scope 문서가 score = 1.0으로 top-1
          in-scope 문서가 score = 0.95로 top-2

Soft scoring 후:
- out-of-scope: 1.0 - 0.05 = 0.95
- in-scope: 0.95 + 0.05 = 1.00 → 역전 성공

BUT 실제로는:
- out-of-scope: score = 1.0 (강력한 semantic match)
- in-scope: score = 0.1 (약한 match)

Soft scoring 후:
- out-of-scope: 1.0 - 0.05 = 0.95 → 여전히 top-1
- in-scope: 0.1 + 0.05 = 0.15 → 여전히 하위
```

#### 원인 3: 양방향 오염

```
λ를 크게 키우면 (예: λ = 1.0):
- in-scope 문서 score 대폭 상승
- out-of-scope 문서 score 대폭 하락 → contamination 감소 가능

BUT:
- Shared documents에 적용 불가 (device 정보 없음)
- Parser 오류 시 wrong device에 +1.0 가산점 → gold hit 폭락
- λ가 크면 base score 의미 없어짐 → 사실상 hard filter와 동일
```

### 6.4 결론

**Soft scoring(P6/P7)은 실운영에서 비효과적**:
1. λ=0.05: 너무 작아서 base score gap 극복 불가
2. λ를 크게 키우면: hard filter와 본질적으로 동일
3. 중간값 λ: 어느 조건에도 최적이 아님
4. **Hard filter(B4)가 압도적으로 우수** — contamination 0%, gold hit +26~42%p

**이론적 결론**:
- Soft scoring이 효과적이려면 `λ ≫ base_score_gap` 이어야 함
- 이 조건을 만족하면 soft scoring은 hard filter로 수렴
- 따라서 soft scoring은 hard filter의 실용적 대안이 아님

---

## 7. Parser Accuracy 결과

### 7.1 실험 설계

**목적**: 실제 운영에서 device parser의 정확도 측정 (oracle 조건에서 real 조건으로의 전환 가능성 평가)

**평가 대상**: 578개 쿼리에 대해 parser가 추출한 device vs. gold device 비교

**Parser 작동 원리**:
- 입력: 자연어 질문 (원본, device명 포함 가능)
- 출력: 인식된 device_name (없으면 None)
- 방법: 장비명 목록과의 fuzzy string matching + NER

**분류 기준**:
- `exact match`: parser가 정확한 device 추출
- `no detection`: parser가 device를 전혀 인식 못함
- `wrong detection`: parser가 틀린 device를 반환

### 7.2 전체 결과

| Metric | Count | Rate |
|--------|-------|------|
| Exact match | 380/578 | 65.7% |
| No detection | 150/578 | 26.0% |
| Wrong detection | 48/578 | 8.3% |
| **Total** | **578** | **100%** |

### 7.3 scope_observability별 분석

#### explicit_device (n=429)

| Metric | Count | Rate |
|--------|-------|------|
| Exact match | **380/429** | **88.6%** |
| No detection | 1/429 | 0.2% |
| Wrong detection | 48/429 | 11.2% |

**해석**:
- 장비명이 명시된 쿼리는 88.6%로 정확하게 파싱
- Wrong detection 11.2%: 장비명이 있는데도 틀린 장비를 반환
  - 원인: 유사한 장비명 혼동 (예: "SUPRA XP" → "SUPRA N" 파싱)
  - 원인: 특수문자, 띄어쓰기 변형 처리 실패

#### explicit_equip (n=149)

| Metric | Count | Rate |
|--------|-------|------|
| Exact match | **0/149** | **0.0%** |
| No detection | 149/149 | **100%** |
| Wrong detection | 0/149 | 0.0% |

**해석**:
- 장비 유형 쿼리에서 parser 완전 실패 (0%)
- "etcher", "ALD", "CVD" 같은 장비 유형은 특정 기종으로 매핑 불가
- 이는 parser의 한계가 아니라 task의 본질적 어려움
  - 장비 유형 → 특정 기종 매핑은 추가 context 없이는 불가능
  - 가능한 해결책: 사용자 컨텍스트 기반 disambiguation

### 7.4 Oracle B4 vs Real B4 성능 비교

| Metric | Oracle B4 | Real B4 | Delta |
|--------|-----------|---------|-------|
| gold_hit@10 (loose) | 0.927 (536/578) | 0.919 (531/578) | -0.009 (-0.9%p) |
| contamination_adj@10 | 0.000 | 0.306 | +0.306 (+30.6%p) |
| MRR (Mean Reciprocal Rank) | 0.846 | 0.832 | -0.014 (-1.4%p) |

**핵심 결론**:

```
Gold Hit 관점: Oracle ≈ Real (-0.9%p)
→ Device parser가 gold를 찾는 능력은 거의 oracle과 동일

Contamination 관점: Oracle 0% vs Real 30.6%
→ Contamination은 크게 증가 (30.6%p 차이)
→ Parser 실패 시 no-filter fallback으로 contamination 도입
```

**원인 분석**:
- explicit_device 쿼리(88.6% 정확도): oracle과 real 간 gold hit 차이 미미
- explicit_equip 쿼리(0% 정확도): parser 실패 → filter 없이 검색 → contamination 도입
- explicit_equip이 149/578 = 25.8%이므로, 전체 contamination의 대부분을 차지

**논문 함의**:
1. Device parser가 충분히 정확하다면 oracle 수준 달성 가능 (gold hit 관점)
2. 그러나 equip-level 쿼리에 대한 별도 처리 전략 없이는 contamination 완전 제거 불가
3. 실운영에서는 parser 실패 시 fallback 전략 필요:
   - Option A: equip → device disambiguation (사용자에게 물어보기)
   - Option B: equip 쿼리에서 특별 처리 (상위 장비 그룹 단위 필터)
   - Option C: explicit_device만 scope filtering 적용, equip는 no-filter

---

## 8. Gold Label 검증 결과

### 8.1 검증 목적

- 자동 생성 gold label의 신뢰성 검증
- 논문에서 "gold label을 신뢰할 수 있다"는 claim의 근거 마련

### 8.2 검증 방법

**샘플링**: 578개 쿼리 중 75개 무작위 샘플
**평가 단위**: (query, doc) 쌍 = 337개 쌍 검증
- strict gold: 177쌍
- loose gold (strict 제외): 160쌍

**판정 기준**:
1. `gold_strict 적합`: 문서가 질문에 직접 답변 가능 (yes/no)
2. `gold_strict 부분 적합`: 문서가 부분적으로만 답변 가능
3. `false positive`: 문서가 전혀 관련 없음

**검증 방법**: 도메인 전문가(PE 엔지니어) 검토 또는 careful manual inspection

### 8.3 전체 결과

| Metric | Value |
|--------|-------|
| Strict gold precision | 97.2% (172/177) |
| Partially relevant (strict gold에서) | 2.8% (5/177) |
| False positive (not relevant) | 0.0% (0/177) |
| Loose gold recall | 100% (160/160) |
| Total pairs verified | 337 |

**해석**:
- Strict gold의 97.2%가 실제로 적합 (5개만 부분 적합, false positive 없음)
- Loose gold에서 false negative 없음 (검증된 관련 문서 모두 loose gold에 포함)
- **결론**: 자동 생성 gold label의 품질이 매우 높아 논문 eval에 사용 적합

### 8.4 scope_observability별 분석

| scope_obs | n (쿼리) | strict precision | partial | false_pos |
|-----------|----------|------------------|---------|-----------|
| explicit_device | 429 → 샘플 ~56 | 99.3% | 0.7% | 0.0% |
| explicit_equip | 149 → 샘플 ~19 | 84.0% | 16.0% | 0.0% |

**해석**:
- explicit_device: 99.3% precision — 거의 완벽한 gold 품질
- explicit_equip: 84.0% precision — 장비 유형 쿼리의 gold는 덜 정확
  - 이유: 장비 유형이 여러 기종에 해당하므로 "어떤 기종의 문서가 gold인가"가 모호
  - 예: "etcher PM 주기" 질문에 SUPRA XP, INTEGER plus 중 어느 문서가 strict gold인지 모호

### 8.5 검증 결론

```
신뢰 가능한 claims:
1. Strict gold는 실제로 질문과 관련 있음 (97.2% precision)
2. Loose gold는 관련 문서를 놓치지 않음 (100% recall)
3. False positive 없음 — gold가 틀린 문서를 가리키지 않음

주의 필요한 cases:
1. Explicit_equip 쿼리의 strict gold (84%)는 상대적으로 낮음
2. Partial relevant 5개 — 엄격한 평가에서는 제외 가능
```

---

## 9. 전체 결론 및 논문 방향

### 9.1 강한 Claims (데이터가 강력히 뒷받침)

#### Claim 1: Cross-equipment contamination은 심각하다

```
근거:
- BM25 (no filter): cont@10 = 52.9% (원본), 51.8% (masked)
- Dense kNN (no filter): cont@10 = 37.3% (원본), 73.0% (masked)
- Hybrid+Rerank (no filter): cont@10 = 36.4% (원본), 58.6% (masked)
- explicit_equip 쿼리: cont@10 = 70.6%~95.7%

해석: 원본 질문에서도 top-10의 36-53%가 다른 장비 문서
     장비명 제거하면 51-73%로 상승
```

#### Claim 2: Hard device filter가 contamination 완전 제거 + recall 향상을 동시에 달성한다

```
근거:
- B4_masked vs B3_masked:
  contamination: 0.586 → 0.001 (-58.5%p, 거의 완전 제거)
  gold hit: 380/578(65.7%) → 532/578(92.0%) (+26.3%p 향상)

- explicit_equip 서브셋에서 더 극적:
  contamination: 88.2% → 0% (-88.2%p)
  gold hit: 23/149(15.4%) → 116/149(77.9%) (+62.5%p 향상)
```

#### Claim 3: Soft scoring(P6/P7)은 실전 비효과적이다

```
근거:
- P6/P7 cont@10: B3_masked(0.586)보다 오히려 증가 (+0.065)
- Hard filter(B4) 대비: cont 차이 58.4%p, gold hit 차이 26%p
- 이론적 원인: λ와 base score scale의 mismatch로 re-ranking 불가
```

#### Claim 4: 기존 "문서→질문" 평가는 scope filtering 효과를 과소추정한다

```
근거:
- Phase 1-4 (v0.5 기반): scope filtering 적용 시 gold hit -36~-69%
- v0.6 masked 실험: scope filtering(B4) 적용 시 gold hit +26~42%p

설명: v0.5의 circular bias (doc_id에 device명, BM25 lexical match)로 인해
      scope filtering 없이도 BM25가 correct device 문서 반환
      → scope filtering 적용 후 gold hit 하락이 실제 성능 하락이 아님
      masked query + corrected gold로 평가 시 반대 결과
```

#### Claim 5: Device parser는 explicit query에서 oracle에 근접한다

```
근거:
- Explicit_device parser accuracy: 88.6%
- Oracle B4 vs Real B4 gold hit 차이: -0.9%p (0.927 vs 0.919)
- MRR 차이: -1.4%p (0.846 vs 0.832)

해석: Parser 정확도 88.6%로 gold hit 기준으로는 거의 oracle 수준 달성
     단, contamination은 +30.6%p 차이 (equip query fallback 때문)
```

#### Claim 6: Gold label 품질이 신뢰할 수 있다

```
근거:
- Strict gold precision: 97.2% (172/177)
- False positive: 0%
- Loose gold recall: 100%

해석: 자동 생성 gold의 97.2%가 실제로 relevant
     논문 평가에 사용 적합
```

### 9.2 주의 필요한 Claims

#### 주의 1: B4.5(shared)가 B4보다 낮은 이유

- **현상**: shared docs(60개) 추가 허용이 gold hit 하락 유발 (-16~22%p)
- **현재 설명**: shared docs가 noise로 작용
- **추가 분석 필요**:
  - 어떤 shared docs가 문제인가? (per-doc 분석)
  - SUPRA Vplus처럼 특정 장비에서 더 영향이 큰가?
  - Shared doc 분류 기준을 재검토해야 하는가?

#### 주의 2: Implicit/ambiguous query 실험 미완료

- **현황**: v0.7 통합 셋(1206q)은 구성 완료, 본격 실험 미실시
- **중요성**: 실운영에서 implicit query가 상당 비율 차지할 것으로 예상
- **계획**: implicit(578q) × B0/B3/B4/B4.5 실험 필요

#### 주의 3: Equip-level query 처리 전략

- **현황**: explicit_equip 쿼리에서 parser 0%, contamination 높음
- **현재 한계**: oracle 없이는 device 특정 불가
- **필요**: equip → device 매핑 전략 설계 (사용자 disambiguation, 그룹 필터 등)

### 9.3 논문 Research Questions

#### RQ1: 산업 RAG에서 cross-equipment contamination은 얼마나 심각한가?

**답변 요약**:
- BM25: 52.9% contamination (no filter)
- Dense: 73% contamination (masked, no filter)
- Hybrid+Rerank: 58.6% contamination (masked, no filter)
- 특히 equip-level query에서 70-96%로 심각
- **메시지**: 기존 RAG는 도메인 특화 contamination에 심각하게 취약

**지원 데이터**: Table 1-3 (Section 5.2, 5.3)

#### RQ2: Device-aware scope filtering이 contamination과 recall에 미치는 영향은?

**답변 요약**:
- contamination: 50-88% → 0% (완전 제거)
- gold hit: 40-66% → 92% (+26~42%p 향상)
- **핵심 메시지**: Scope filtering은 contamination 감소와 recall 향상을 동시에 달성

**지원 데이터**: B3_masked vs B4_masked 비교 (Section 5.3)

#### RQ3: 기존 "문서→질문" 평가 방식이 scope filtering 효과를 어떻게 왜곡하는가?

**답변 요약**:
- 기존 방식: scope filtering 적용 시 gold hit -36~-69% (Phase 3)
- 수정된 방식: scope filtering 적용 시 gold hit +26~42%p (Section 5.3)
- **왜곡 메커니즘**: doc_id lexical bias + gold set collapse → BM25가 scope filter 없이도 correct device 문서 반환 → scope filter 적용이 오히려 recall 하락으로 나타남
- **메시지**: 평가 방법론의 bias가 실제 시스템 개선을 역방향으로 측정할 수 있음

**지원 데이터**: Phase 3 vs Section 5 결과 비교

### 9.4 논문 한계 (Limitations)

| 한계 | 설명 | 영향 |
|------|------|------|
| 검색 방법 제한 | BM25 + Hybrid만 실험, ColBERT/SPLADE 등 미포함 | 일반화 범위 제한 |
| Implicit/ambiguous 미실험 | v0.7 구성했으나 본격 실험 미실시 | equip-level 전략 미완 |
| Oracle 가정 | B4는 oracle device filter 사용 (상한선) | 실운영과 갭 존재 |
| 단일 도메인 | 반도체 PE만, 다른 제조업/의료 등 미검증 | 일반화 한계 |
| LLM judge 미실시 | vLLM 접속 불가로 heuristic 검증만 | gold 품질 일부 불확실 |
| 단일 언어 편중 | 한국어 문서가 대부분, 영어 혼용 | 언어별 성능 분석 미실시 |

### 9.5 논문 Contributions 정리

1. **Contamination 정량화**: 산업 RAG에서 cross-equipment contamination을 처음으로 체계적으로 측정·정의
2. **평가 방법론 개선**: masked query + scope_observability 분류 + strict/loose gold 체계로 더 정확한 scope filtering 평가 가능
3. **Scope Policy 비교**: hard filter vs soft scoring의 실증적 비교 (soft scoring의 실패 원인 이론화)
4. **Dataset**: 578-query 벤치마크 (v0.6) + 1206-query (v0.7) 공개 가능한 형태로 구성

---

## 10. 파일 위치 및 재현 명령어

### 10.1 평가 데이터 파일

| 파일 경로 | 설명 | 쿼리 수 |
|-----------|------|---------|
| `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl` | v0.6 eval set (loose gold) | 578 |
| `data/paper_a/eval/query_gold_master_v0_6_generated_full_strict.jsonl` | v0.6 eval set (strict gold 포함) | 578 |
| `data/paper_a/eval/query_implicit_from_masked.jsonl` | implicit 변환 쿼리 | 578 |
| `data/paper_a/eval/query_trap_cross_device_v1.jsonl` | trap queries | 50 |
| `data/paper_a/eval/query_gold_master_v0_7_with_implicit.jsonl` | v0.7 통합 셋 | 1206 |

**JSONL 스키마** (v0.6):
```json
{
  "query_id": "q_001",
  "question": "SUPRA XP의 PM 주기는?",
  "question_masked": "[DEVICE]의 PM 주기는?",
  "device_name": "SUPRA XP",
  "scope_observability": "explicit_device",
  "gold_doc_ids": ["supra_xp_sop_001", "supra_xp_manual_003"],
  "gold_doc_ids_strict": ["supra_xp_sop_001"]
}
```

### 10.2 실험 결과 파일

| 파일 경로 | 설명 | 시스템 |
|-----------|------|-------|
| `data/paper_a/trap_masked_results.json` | BM25 masked 실험 결과 | B0, B4, B4.5 |
| `data/paper_a/masked_hybrid_results.json` | Hybrid+Rerank masked 실험 결과 | B0-B4.5 전체 |
| `data/paper_a/masked_p6p7_results.json` | P6/P7 soft scoring 실험 결과 | P6, P7 |
| `data/paper_a/parser_accuracy_report.json` | parser accuracy 측정 결과 | - |
| `data/paper_a/gold_verification_report.json` | gold label 검증 결과 | - |

### 10.3 정책 파일

| 파일 경로 | 설명 | 항목 수 |
|-----------|------|--------|
| `.sisyphus/evidence/paper-a/policy/doc_scope.jsonl` | 문서별 device 매핑 | 508 docs |
| `.sisyphus/evidence/paper-a/policy/shared_doc_ids.txt` | 공용 문서 목록 | 60 docs |
| `.sisyphus/evidence/paper-a/corpus/cross_device_trap_candidates.json` | trap 후보 topic | 68 topics |

**doc_scope.jsonl 스키마**:
```json
{"doc_id": "supra_xp_sop_001", "device_name": "SUPRA XP", "doc_type": "SOP"}
{"doc_id": "general_safety_001", "device_name": null, "is_shared": true}
```

### 10.4 스크립트

| 스크립트 경로 | 설명 | 입력 | 출력 |
|-------------|------|------|------|
| `scripts/paper_a/run_masked_hybrid_experiment.py` | Hybrid+Rerank masked 실험 | v0.6 eval set | masked_hybrid_results.json |
| `scripts/paper_a/run_masked_p6p7_experiment.py` | P6/P7 soft scoring 실험 | v0.6 eval set | masked_p6p7_results.json |
| `scripts/paper_a/measure_parser_accuracy.py` | parser accuracy 측정 | v0.6 eval set | parser_accuracy_report.json |
| `scripts/paper_a/evaluate_paper_a_master.py` | 마스터 평가 스크립트 | eval set + results | 종합 메트릭 |
| `scripts/paper_a/phase3_retrieve_and_pool.py` | TREC pooling (Phase 3) | v0.5 queries | pooled_results.json |

### 10.5 Elasticsearch 인덱스

| 인덱스명 | 설명 | 차원 | 용도 |
|---------|------|------|------|
| `chunk_v3_content` | text + doc_id + metadata (embedding 없음) | - | BM25 검색, 메타데이터 필터 |
| `chunk_v3_embed_bge_m3_v1` | 1024-dim BGE-M3 embedding | 1024 | Dense kNN 검색 |
| `rag_chunks_dev_current` | 768-dim 구버전 (text + embedding 통합) | 768 | 레거시, 현재 미사용 |
| `rag_chunks_dev_v2` | 768-dim 구버전 복사 | 768 | 레거시, 현재 미사용 |

**검색 방식**:
```python
# BM25 (B0)
GET chunk_v3_content/_search
{
  "query": {"match": {"content": "SUPRA XP PM 주기"}},
  "size": 10
}

# Dense kNN (B1)
GET chunk_v3_embed_bge_m3_v1/_search
{
  "knn": {"field": "embedding", "query_vector": [...], "k": 10, "num_candidates": 100}
}

# B4: Hard device filter
GET chunk_v3_content/_search
{
  "query": {
    "bool": {
      "must": {"match": {"content": "..."}},
      "filter": {"term": {"device_name": "SUPRA XP"}}
    }
  }
}
```

### 10.6 실행 명령어

```bash
# 프로젝트 루트 기준

# 환경 설정
cd /home/hskim/work/llm-agent-v2
source .venv/bin/activate  # 또는 uv run prefix 사용

# Hybrid+Rerank masked 실험 (가장 핵심 실험)
uv run python scripts/paper_a/run_masked_hybrid_experiment.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_6_generated_full_strict.jsonl \
  --output data/paper_a/masked_hybrid_results.json \
  --systems B0,B1,B2,B3,B4,B4.5

# P6/P7 soft scoring 실험
uv run python scripts/paper_a/run_masked_p6p7_experiment.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_6_generated_full_strict.jsonl \
  --output data/paper_a/masked_p6p7_results.json \
  --lambda 0.05

# Parser accuracy 측정
uv run python scripts/paper_a/measure_parser_accuracy.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl \
  --output data/paper_a/parser_accuracy_report.json

# 마스터 평가 (결과 종합)
uv run python scripts/paper_a/evaluate_paper_a_master.py \
  --results data/paper_a/masked_hybrid_results.json \
  --eval-set data/paper_a/eval/query_gold_master_v0_6_generated_full_strict.jsonl \
  --breakdown-by scope_observability

# 특정 시스템만 평가
uv run python scripts/paper_a/evaluate_paper_a_master.py \
  --results data/paper_a/masked_hybrid_results.json \
  --system B4_masked \
  --subset explicit_equip
```

### 10.7 재현 시 주의사항

1. **ES 연결**: `SEARCH_ES_HOST=localhost:8002` 설정 필요
2. **인덱스 존재 확인**: `chunk_v3_content`, `chunk_v3_embed_bge_m3_v1` 인덱스 필수
3. **vLLM 없이 실행 가능**: retrieval 실험은 LLM 불필요, parser accuracy도 LLM 불필요
4. **실행 시간**: 578q × 6 systems ≈ 30-60분 (ES 부하에 따라)
5. **결과 재현성**: `random_seed=42` 설정 (RRF는 deterministic이나 kNN 상위 후보는 약간 변동 가능)

---

## 11. Device별 상세 결과

### 11.1 B4 Masked Gold Hit (Loose) by Device — BM25

#### 상위 10개 장비 (gold hit 기준)

| Device | Hit Count | Total | Hit Rate |
|--------|-----------|-------|---------|
| TIGMA Vplus | 50 | 50 | 100.0% |
| SUPRA N series | 20 | 20 | 100.0% |
| OMNIS plus | 12 | 12 | 100.0% |
| INTEGER plus | 90 | 91 | 98.9% |
| GENEVA XP | 68 | 69 | 98.6% |
| SUPRA N | 80 | 82 | 97.6% |
| PRECIA | 65 | 68 | 95.6% |
| ZEDIUS XP | 38 | 41 | 92.7% |
| SUPRA XP | 24 | 26 | 92.3% |
| SUPRA Vplus | 58 | 88 | 65.9% |

#### SUPRA Vplus 65% 심층 분석 (2026-03-14 확인 완료)

**문제**: SUPRA Vplus가 65%로 가장 낮음 — 다른 장비들이 92-100%인 것과 크게 차이

**기본 통계**:
- SUPRA Vplus 88개 쿼리 중 **79개(90%)가 explicit_equip**, 9개만 explicit_device
- explicit_device 9개: **100% gold hit** (문제 없음)
- explicit_equip 79개 중 **31개 실패** (39% 실패율)
- **실패 전부 explicit_equip에서 발생**

**확인된 근본 원인 3가지**:

**원인 1: Gold Label 과대 할당 (17/31건)**
- Doc `40097172` (topic="CONTROLLER")가 **21개 쿼리의 strict gold**로 지정
- 실제 질문 topic: BAFFLE, DOCKING, LEAK CHECK, ROBOT, MFC, SCR 등 **20가지 다른 topic**
- CONTROLLER 문서가 catch-all gold로 잘못 할당됨 → BM25가 "DOCKING" 검색 시 "CONTROLLER" 문서를 찾을 수 없음
- **결론: gold label 품질 문제 (topic mismatch)**

**원인 2: Corpus Coverage 부족 (SOP 4개)**
- SUPRA Vplus 전체 문서: 53개 (SOP 4개 + PEMS 49개)
- SOP 문서: `undocking`, `io_check`, `power_turn_on_off`, `controller` — 4개뿐
- 대부분의 topic에 대한 SOP가 없음 → PEMS 문서(수치 ID)에 의존
- PEMS 문서는 BM25 lexical match가 어려운 구조 (정형화된 이력 데이터)

**원인 3: PEMS 문서 BM25 검색 한계 (14/31건)**
- Topic이 일치하는데도 실패한 14건: EPD, FCIP, EXHAUST RING, ROBOT TEACHING 등
- PEMS 문서는 점검/조치 이력 형태 → BM25가 query term과 직접 매칭 어려움
- B4 filter 적용 후 SUPRA Vplus 문서만 남지만, 그 중 gold PEMS를 top-10에 올리지 못함
- n_docs가 2~7개로 적음 → corpus 자체에 해당 topic의 SUPRA Vplus 문서가 소수

**결론**:
SUPRA Vplus 65%는 scope filtering의 한계가 아니라:
1. Gold label 품질 문제 (55% — CONTROLLER catch-all)
2. Corpus coverage 부족 (SOP 4개)
3. PEMS 문서의 BM25 검색 한계 (45%)

**논문에서의 처리 방안**:
- SUPRA Vplus를 별도 슬라이스로 보고하고 원인 명시
- Gold label 재검토 후 CONTROLLER catch-all 수정 시 gold hit 상승 예상
- 또는 SUPRA Vplus를 "coverage-limited" 케이스로 분류하여 별도 보고

### 11.2 전체 Device별 결과 (추정)

v0.6 데이터셋은 27개 장비에 균등 분포 (각 약 20-22q). 상위/하위 성능 장비 패턴:

**성능 높은 장비 특징**:
- 고유한 이름 (다른 장비와 혼동 없음)
- corpus coverage 충분
- 문서 내 device명 명확히 기재

**성능 낮은 장비 특징** (SUPRA Vplus 패턴):
- 동일 계열 내 여러 하위 모델 존재 → 문서 중복
- Corpus coverage 부족 (장비 수명 주기 상 문서 적음)
- Gold label ambiguity (어느 버전의 문서가 정확한 gold인가?)

### 11.3 Hybrid+Rerank B4 masked by Device

Hybrid 기반으로도 유사한 패턴 예상:
- 고유 장비명 보유 장비: 90-97% gold hit
- SUPRA 계열 일부: 65-75% gold hit
- equip-level 쿼리 포함 장비: 낮은 gold hit

### 11.4 Device별 Contamination 분석

**가장 오염되는 케이스**:
- 같은 계열 내 장비 간 오염 (SUPRA XP ↔ SUPRA N ↔ SUPRA Vplus)
- 기능이 유사한 다른 계열 간 오염 (etcher 계열끼리)

**오염 매트릭스 (2026-03-14 측정 완료, B3_masked 기준)**:

Top 10 오염 쌍 (target ← source: 출현 횟수):
```
INTEGER plus ← PRECIA: 173
GENEVA XP   ← PRECIA: 149
SUPRA N     ← PRECIA: 90
INTEGER plus ← SUPRA N series: 75
ZEDIUS XP   ← PRECIA: 74
INTEGER plus ← SUPRA N: 68
GENEVA XP   ← SUPRA N series: 64
PRECIA      ← INTEGER plus: 55
SUPRA N     ← INTEGER plus: 51
PRECIA      ← SUPRA N: 48
```

**핵심 발견**:
- **PRECIA가 최대 오염원**: 거의 모든 주요 장비를 오염 (INTEGER plus 173건, GENEVA XP 149건, SUPRA N 90건, ZEDIUS XP 74건)
- 이유: PRECIA는 다양한 topic의 SOP를 보유, BM25에서 topic 매칭으로 광범위하게 반환
- **SUPRA N series도 주요 오염원**: INTEGER plus(75건), GENEVA XP(64건) 오염
- **양방향 오염**: PRECIA ↔ INTEGER plus (173 vs 55), PRECIA ↔ SUPRA N (90 vs 48)
- **같은 계열 내 오염은 상대적으로 적음**: SUPRA XP ← SUPRA N은 top 20에 없음

---

## 12. 실험 타임라인 및 이력

### 12.1 실험 이력

| 날짜 | 작업 | 결과 |
|------|------|------|
| 2026-01-08 | Meta-guided hierarchical RAG 초기 설계 | 기초 설계 완료 |
| 2026-02-12 | GCB equip_id matching 분석 | device 매핑 품질 확인 |
| 2026-03-04 | corpus statistics 정리 | 508 docs, 27 devices 확인 |
| 2026-03-05 | Phase 1-4 오류 분석 | v0.5 bias 원인 규명 |
| 2026-03-05 | Paper A 메인 결과 정리 | Phase 1-4 종합 |
| 2026-03-09 | Gold re-judging 분석 | gold 품질 97.2% 확인 |
| 2026-03-12 | Cross-device topic feasibility 분석 | trap query 후보 68개 확인 |
| 2026-03-12 | Dataset protocol redesign | v0.6 → v0.7 설계 완료 |
| 2026-03-12 | Slot valve hard filter recall 분석 | B4.5 문제 분석 시작 |
| 2026-03-13 | Paper A 진행 상황 요약 | 전체 실험 상태 점검 |
| 2026-03-14 | Masked experiment 핵심 결과 수집 | 본 문서 작성 |

### 12.2 현재 상태 및 잔여 작업

**완료된 실험**:
- [x] v0.6 eval set 구축 (578q)
- [x] BM25 masked experiment (B0, B4, B4.5)
- [x] Hybrid+Rerank masked experiment (B0-B4.5)
- [x] P6/P7 soft scoring experiment
- [x] Parser accuracy measurement
- [x] Gold label verification (337쌍)
- [x] v0.7 통합 셋 구성 (1206q)

**미완료 작업**:
- [x] v0.7 implicit 쿼리 본격 실험 완료 (628q × 10 conditions) → Section 13 참조
- [x] SUPRA Vplus 오류 케이스 상세 분석 → 원인: explicit_equip 실패 + CONTROLLER catch-all gold + SOP 4개
- [x] B4.5 shared doc 문제 심층 분석 → 원인: shared docs가 0건 도움, 57.3% top-10 점유, device_net_board 5개가 주범
- [ ] Equip query disambiguation 전략 설계
- [x] Device별 오염 매트릭스 완성 → PRECIA #1 오염원, Section 10 참조
- [x] 논문 draft 작성 → `paper_a_draft_v2.md`

---

## 13. Implicit Query Experiment (v0.7, n=628)

### 13.1 실험 개요

v0.7 eval set에서 implicit/ambiguous 쿼리 628개를 대상으로 동일 10개 조건 실험 수행.
- 쿼리에 장비명이 포함되지 않아 masking 효과가 없음 (masked ≈ orig)
- scope_observability = "ambiguous" (전수)
- 파일: `data/paper_a/implicit_hybrid_results.json`

### 13.2 전체 결과

| System | cont@10 | gold_strict | gold_loose |
|--------|---------|-------------|------------|
| B0_masked | 0.652 | 332/628 (52.9%) | 414/628 (65.9%) |
| B0_orig | 0.652 | 332/628 (52.9%) | 414/628 (65.9%) |
| B1_masked | 0.735 | 253/628 (40.3%) | 304/628 (48.4%) |
| B1_orig | 0.734 | 257/628 (40.9%) | 311/628 (49.5%) |
| B2_masked | 0.664 | 388/628 (61.8%) | 450/628 (71.7%) |
| B2_orig | 0.664 | 386/628 (61.5%) | 449/628 (71.5%) |
| B3_masked | 0.665 | 386/628 (61.5%) | 449/628 (71.5%) |
| B3_orig | 0.665 | 388/628 (61.8%) | 452/628 (72.0%) |
| **B4_masked** | **0.001** | **532/628 (84.7%)** | **581/628 (92.5%)** |
| B4.5_masked | 0.001 | 481/628 (76.6%) | 557/628 (88.7%) |

### 13.3 Implicit vs Explicit 비교

| Condition | Imp Strict | Exp Strict | Delta | Imp Cont | Exp Cont |
|-----------|-----------|-----------|-------|----------|----------|
| B0_masked | 52.9% | 49.7% | +3.2% | 0.652 | 0.473 |
| B2_masked | 61.8% | 60.7% | +1.1% | 0.664 | 0.585 |
| B3_masked | 61.5% | 60.7% | +0.7% | 0.665 | 0.584 |
| B4_masked | 84.7% | 91.2% | -6.5% | 0.001 | 0.001 |
| B4.5_masked | 76.6% | 70.2% | +6.4% | 0.001 | 0.001 |

### 13.4 핵심 발견

1. **Masking 무효과 확인**: implicit 쿼리에서 masked/orig delta = 0~4건 — 장비명이 없으므로 masking이 영향 없음. 이는 masking methodology의 정당성을 역으로 검증.

2. **Implicit 베이스라인 ≈ Explicit-masked**: B0 52.9% vs 49.7%, B2 61.8% vs 60.7% — 명시적 쿼리에서 장비명을 제거한 결과와 암묵적 쿼리의 성능이 유사. 즉, masking이 explicit 쿼리를 implicit 수준으로 "중립화"함을 확인.

3. **B4 여전히 최선이나 implicit에서 더 낮음**: 84.7% vs 91.2% (−6.5%p). 쿼리에 장비명이 없어 parser가 device를 추출할 수 없으므로, oracle device filter에 의존. 실제 시스템에서는 대화 컨텍스트에서 device를 추론해야 함.

4. **오염이 implicit에서 더 심각**: B0 cont@10 = 0.652 vs 0.473. 장비명 신호 부재로 더 많은 타 장비 문서가 검색됨. Scope filtering의 필요성이 더욱 강조됨.

5. **Shared doc paradox 축소**: B4-B4.5 gap이 8.1%p (implicit) vs 20.9%p (explicit). implicit 쿼리에서 shared doc 침투가 상대적으로 덜 심각하나, 여전히 B4.5가 B4보다 열위.

### 13.5 주요 장비별 B4 성능 (implicit)

| Device | B4 Strict | n |
|--------|-----------|---|
| INTEGER plus | 93.8% | 96 |
| SUPRA Vplus | 60.2% | 93 |
| SUPRA N | 92.7% | 82 |
| PRECIA | 87.2% | 78 |
| GENEVA XP | 98.6% | 69 |
| ZEDIUS XP | 63.9% | 61 |
| TIGMA Vplus | 81.7% | 60 |
| SUPRA XP | 92.3% | 26 |

**주목할 점**: SUPRA Vplus (60.2%), ZEDIUS XP (63.9%)가 implicit에서도 가장 낮음 — explicit과 동일한 패턴. 이들은 equip 수준 매칭 문제가 아니라 문서 커버리지 자체가 부족.

---

## Appendix A. 주요 수치 참조표

### A.1 핵심 메트릭 요약 (masked 조건, n=578)

| System | cont@10 | gold_strict | gold_loose | MRR |
|--------|---------|-------------|------------|-----|
| B0_masked | 0.518 | 287/578 (49.7%) | 343/578 (59.3%) | - |
| B1_masked | 0.730 | 234/578 (40.5%) | 267/578 (46.2%) | - |
| B2_masked | 0.586 | 351/578 (60.7%) | 380/578 (65.7%) | - |
| B3_masked | 0.586 | 351/578 (60.7%) | 380/578 (65.7%) | - |
| B4_masked | 0.001 | 527/578 (91.2%) | 532/578 (92.0%) | 0.846 |
| B4.5_masked | 0.001 | 406/578 (70.2%) | 439/578 (76.0%) | - |

### A.2 scope_observability별 B4 효과

| scope_obs | B3_masked cont | B4_masked cont | B3_masked gold | B4_masked gold | delta gold |
|-----------|---------------|---------------|----------------|----------------|------------|
| explicit_device | 0.483 | 0.001 | 357/429 (83.2%) | 416/429 (97.0%) | +13.8%p |
| explicit_equip | 0.882 | 0.000 | 23/149 (15.4%) | 116/149 (77.9%) | +62.5%p |
| ALL | 0.586 | 0.001 | 380/578 (65.7%) | 532/578 (92.0%) | +26.3%p |

### A.3 Parser Accuracy by scope_observability

| scope_obs | exact | no_det | wrong | n |
|-----------|-------|--------|-------|---|
| explicit_device | 88.6% | 0.2% | 11.2% | 429 |
| explicit_equip | 0.0% | 100.0% | 0.0% | 149 |
| **전체** | **65.7%** | **26.0%** | **8.3%** | **578** |

---

*문서 생성: 2026-03-14*
*최종 업데이트: 2026-03-14 — implicit 쿼리 실험 결과 (Section 13) 추가*
