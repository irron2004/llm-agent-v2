# Paper A 실험 정의서 v0.6

## Hierarchy-aware Scope Routing (G) + Shared Doc Policy + Contamination-aware Scoring + Matryoshka Router

> **v0.6 변경사항 (PE 1차 피드백 반영)**:
> - Paper A v1 범위 고정: **core = device + doc_type + shared**, family = exploratory/ablation
> - doc_type canonical 정규화 통일: `ts_guide→ts`, `setup_manual→manual`
> - `device_family_gold.csv` → `device_family_seed.csv` (gold 미승격, seed/prior로 사용)
> - family는 본 실험 main table이 아닌 ablation/appendix로 포지셔닝

## 0) 목표

반도체 유지보수 RAG에서 **cross-equipment contamination(타 장비 문서 혼입)**을 줄이되, **공용 SOP/유사 장비로 인한 recall 손실을 최소화**하는 스코프 정책을 설계/검증한다.

### 0.1 Paper A v1 범위 (PE 1차 피드백 기준 확정)

**Core (main table):**
1. **device-aware scope restriction** — contamination 감소의 주 메커니즘
2. **doc_type-aware routing/weighting** — intent에 따른 문서 유형 선택
3. **shared document policy** — hard filter의 recall 손실 완화

**Exploratory (ablation/appendix):**
4. family-aware retrieval — `device_family_seed.csv` 기반 예비실험만
5. equip-level contamination — partial experiment (field_record scope label 부족)
6. acronym/term-sense disambiguation (ASD) — 약어 다의어 보정 (예: `AR`)

**Future extension (Paper A-1/A-2):**
7. engineer-validated family graph → main experiment 반영
8. field_record(myservice/gcb) scope table 보강 → equip 실험 본격화

> **논문 서술 권장 (PE)**:
> - "We use shared-document labels as a verified relaxation mechanism."
> - "We further explore family-level scope expansion using a provisional similarity graph."
> - "Family-based retrieval is treated as an exploratory extension rather than a core assumption."

---

## 1) 핵심 개념 정의

### 1.1 Allowed Scope(허용 스코프)

질의 q에 대해 허용되는 장비 범위를 S(q)로 정의:

* **Hard scope**: S_hard(q) = 파서가 확실히 잡은 device/equip (예: `device_name=SUPRA XP`, `equip_id=EPAG50`)
* **Family scope**: S_fam(q) = S_hard(q) ∪ Family(device)
* **Shared docs**: 장비와 무관하게 공용으로 허용되는 문서 집합 D_shared

최종 허용 스코프:

* 문서 d가 허용되는 조건:

```
d ∈ D_shared  OR  device(d) ∈ S(q)
```

> 해석: "공용 문서는 항상 허용", 그 외 문서는 스코프(device/equip) 안에 있어야 허용

### 1.2 Scope Level (문서별 스코프 적용 수준)

장비 계층은 `device_name → equip_id` 2-level. 모든 문서에 동일한 필터 강도를 적용하면
recall 손실이 크므로, **문서별 `scope_level`을 부여**하여 필터 강도를 분리:

| scope_level | 의미 | 필터 적용 | 대상 doc_type |
|-------------|------|-----------|---------------|
| `shared` | 공용 문서 (모든 장비에서 허용) | 필터 없음 | D_shared로 판정된 SOP/TS |
| `device` | device_name까지만 유의미 | `device_name ∈ S(q)` | SOP, setup_manual, TS (대부분) |
| `equip` | equip_id까지 유의미 | `equip_id = parsed.equip_id` | 정비이력, 로그, 설비별 기록 |

**doc_type → scope_level 기본 매핑**:

* `sop`, `manual`, `ts` → `device`
* `myservice`, `gcb` (로그/이력) → `equip`
* D_shared 판정 문서 → `shared` (doc_type 무관하게 override)

> **설계 원칙**: equip_id는 의미(semantics)가 아닌 식별자(identifier)이므로
> 임베딩 모델로 "추정"하지 않고, 파서가 잡으면 사용 / 못 잡으면 device level로 fallback.

### 1.3 질의 스코프 결정 규칙

| 파싱 결과 | 질의 성격 | 스코프 결정 |
|-----------|-----------|------------|
| equip_id 있음 | — | Hard(equip): `scope_level=equip` 문서는 equip_id 필터, 나머지는 device 필터 |
| device_name만 있음 | — | Hard(device): device_name 필터 |
| 둘 다 없음 + SOP/절차 성격 | 절차 질의 | Router → Family 확장 (device level) |
| 둘 다 없음 + 이력/로그 성격 | 인스턴스 질의 | equip_id 요구(되묻기) 또는 세션 컨텍스트에서 추출 |

> **Family 확장은 SOP/TS(절차 문서)에만 적용**. 로그/이력에 family 확장 시 노이즈 증가.

### 1.4 Contamination-aware Scoring Function (v0.4 신규)

기존 설계는 scope를 **binary filter**로만 적용(허용/차단). v0.4에서는 filter 이후 재랭킹 단계에서 **scope 위반을 penalty로 반영하는 점수함수**를 도입하여, contamination을 목적함수 수준에서 제어한다.

#### 점수함수 정의

```
Score(d, q) = Base(d, q) - λ(q) · v_scope(d, q)
```

| 항 | 정의 | 설명 |
|---|---|---|
| `Base(d, q)` | cross-encoder reranker 점수 (또는 Hybrid+RRF 점수) | 기본 relevance |
| `v_scope(d, q)` | scope 위반 indicator: `1[d ∉ D_shared ∧ device(d) ∉ S(q)]` | out-of-scope이면 1, 아니면 0 |
| `λ(q)` | 질의별 penalty 강도 | 라우팅 확신도에 따라 적응적 조절 |

#### λ(q) 적응형 조절 (C4 통합)

라우팅 확신도가 높으면 λ→큰값 (hard filter에 수렴), 낮으면 λ→작은값 (soft penalty):

```
confidence(q) = score_top1 - score_top2   # router Top-1 vs Top-2 gap
λ(q) = λ_max · σ(α · confidence(q) - β)  # sigmoid 스케일링
```

| 케이스 | confidence | λ(q) | 동작 |
|---|---|---|---|
| 파서가 장비명 확정 | ∞ (명시적) | λ_max | ≈ hard filter |
| 라우터 Top-1 우세 | 높음 | λ_max에 근접 | 강한 penalty |
| 라우터 Top-1/2 비슷 | 낮음 | 작은 값 | soft penalty (recall 보존) |

> **설계 원칙**: 파서가 device_name을 확정한 경우(§1.3 Hard 모드)는 기존처럼 binary filter 유지.
> λ(q) 적응형 조절은 **Router 모드(장비명 미기재/모호 질의)**에서만 적용.

#### 확장 (Appendix 후보): doc_type penalty

```
Score(d, q) = Base(d, q) - λ₁(q) · v_scope(d, q) - λ₂(q) · v_type(d, q)
```

- `v_type(d, q)`: intent와 doc_type 부적합도 (`1 - w(intent(q), doc_type(d))`)
- `λ₂(q)`: intent 분류 confidence에 따라 조절
- 본문에서는 `λ₁·v_scope`만 사용, `λ₂·v_type`은 데이터 충분성 확인 후 Appendix로 분리

### 1.5 Acronym/Term-Sense Disambiguation (ASD, v0.7 추가)

현장 질의에는 `AR`처럼 의미가 여러 개인 약어가 자주 등장한다.
예: `AR 수치 저하`는 의도상 `ashing rate 저하`인데, 검색에서는 `atomic ratio` 관련 문서가 상위로 섞일 수 있다.

이 문제는 Paper A의 core(장비 스코프)와 충돌하지 않는다. 위치는 다음과 같다.

* **Step 0 전처리 모듈**: 파서/라우터 이전에 약어 의미를 정규화
* **역할**: query lexical ambiguity 완화 (device scope 정책 대체 아님)

v1 정책(규칙 기반, 설명 가능성 우선):

1. 약어 사전 후보 생성: `AR -> {ashing_rate, atomic_ratio, ...}`
2. 문맥 키워드 점수: 질의 주변 단어(`저하`, `증가`, `알람`, `막힘`, `조성`, `비율` 등)와 sense별 cue 매칭
3. doc_type prior 점수: intent/doc_type와 sense의 적합도 반영
4. 최종 결정:
   - confidence >= `τ_asd` 이면 canonical term으로 치환/확장
   - confidence < `τ_asd` 이면 원문 유지 + `needs_clarification` 플래그

간단 점수식(규칙 기반):

```
score(s | q) = w_ctx * match_ctx(s, q) + w_type * prior_doc_type(s, q) + w_dev * prior_device(s, q)
sense*(q) = argmax_s score(s | q)
```

출력 필드(로그/분석용):

* `asd_term_raw` (예: `AR`)
* `asd_sense_pred` (예: `ashing_rate`)
* `asd_confidence`
* `asd_mode` (`rewrite` / `expand` / `abstain`)

> 운영 원칙: ASD는 **query normalization 보조기능**이며, Paper A 메인 주장(스코프 필터링)의 대체 기법으로 기술하지 않는다.

---

## 2) 오프라인 준비(필수 3종)

### 2.1 Shared 문서 판정(공용 SOP/TS 처리)

문서 단위 키(가능하면 `doc_id`, 없으면 안정적인 대체키)로 **몇 개 device_name에 매핑되는지**를 집계:

* deg(doc) = |{device_name}|
* **Shared rule** (예시):
  * `device_name == "ALL"` 또는
  * deg(doc) >= T (예: T=3) 또는
  * 특정 doc_type(SOP/TS)에서만 shared 허용

→ 이 문서들을 D_shared로 마킹(메타데이터 `is_shared=true` 권장)

**코퍼스 현실 참고**: 현재 데이터에서 같은 topic은 장비별로 **별도 doc_id**로 존재.
따라서 "doc_id 공유" 방식보다 **"topic 공유" 기반 유사도**로 shared 판정이 더 적합할 수 있음.

### 2.2 Family(device) 구축(유사 장비/공유 문서 기반) — **exploratory, seed 단계**

> **PE 피드백 (v0.6)**: family는 Paper A에서 hard filter의 recall 회복 장치이므로 라벨 품질이 핵심.
> 현재 `device_family_seed.csv`는 키워드 기반 자동 생성본(431 pairs)으로, **gold가 아닌 seed/prior**.
> 본 실험 main table에서는 family 미사용 또는 보조 실험으로만 포함.
> 엔지니어 검증 후 gold로 승격하여 본실험에 반영.

가장 설명 가능한 방식(추천): **공유 문서(topic) 기반 그래프 + 군집화**

* device 그래프: 노드=device, 엣지 가중치=공유 topic 유사도

```
w(a,b) = |Topics(a) ∩ Topics(b)| / |Topics(a) ∪ Topics(b)|   (Jaccard)
```

* **핵심 주의**: shared 문서가 family 그래프를 망치지 않도록 처리
  * 권장: `w(topic) = 1/log(1 + device_count(topic))` — 다수 장비 걸친 topic은 가중치 낮춤
  * 대안: shared 판정된 topic은 그래프 구성에서 완전 제외

* 군집화: Louvain / connected components (임계치 τ=0.2) 등
* 결과: `Family(device)` = 같은 클러스터(장비 family)

> 코퍼스 현실: 5개 주요 family(SUPRA/INTEGER/PRECIA/GENEVA/OMNIS)가 자연스럽게 형성될 것으로 예상.
> SUPRA family 내 9개 변형 간 intra-family 유사도가 높을 것.

**Family 확장 적용 범위 제한**:
* Family 확장은 **절차 문서(`scope_level=device`)에만 적용**.
* 로그/이력(`scope_level=equip`)에 family 확장 시 타 설비 데이터 혼입 → 노이즈 증가.
* 즉: 절차 문서는 family로 recall 보존, 로그/이력은 equip_id로 정밀 필터.

### 2.3 Matryoshka Router 준비(장비 후보 Top-M 선택용)

목표: **질의에 장비명이 없거나 모호할 때도 device 후보를 싸게 뽑기**

* device prototype 문서(권장 구성):
  * `device_name`별로 대표 텍스트(제목/챕터 헤더/자주 쓰는 트러블 키워드/요약) 생성
  * 이를 임베딩하여 `device_proto_index`에 저장
* 라우팅은 **저차원(Matryoshka 128d 기본, ablation: 64/128/256)**로 Top-M device를 검색
* **필수 확인**: 현재 embedding 모델이 MRL 지원하는지 (지원 안 하면 효과 보장 없음)

### 2.4 ASD 사전/규칙셋 구축 (v0.7 추가, exploratory)

파일(권장): `data/paper_a/metadata/acronym_sense_lexicon.csv`

| 컬럼 | 설명 |
|---|---|
| `term_raw` | 원형 약어 (예: `AR`) |
| `sense_id` | 의미 ID (예: `ashing_rate`, `atomic_ratio`) |
| `canonical_term` | 치환/확장에 쓸 정규형 |
| `positive_cues` | 해당 의미를 지지하는 문맥 cue (`|` 구분) |
| `negative_cues` | 해당 의미와 충돌하는 cue |
| `preferred_doc_types` | 의미와 잘 맞는 doc_type |
| `notes` | 비고 |

최초 시드 구축은 상위 빈도 약어(AR, PR, RF, OES 등)부터 시작하고, 오탐 사례를 누적 반영한다.

---

## 3) 온라인 파이프라인(의사코드)

```pseudo
function ANSWER(q):
  # 0) Acronym/term-sense normalization (optional exploratory module)
  q_norm, asd = NORMALIZE_QUERY_TERMS(q)

  # 1) Parse (이미 auto-parse 존재)
  parsed = AUTO_PARSE(q_norm)  # device_name?, equip_id?, intent?

  # 2) Decide scope S(q)
  if parsed.equip_id:
      S_device = {parsed.device_name}
      S_equip  = {parsed.equip_id}
      mode = "HARD_EQUIP"
  elif parsed.device_name:
      S_device = {parsed.device_name}
      S_equip  = None  # equip 필터 없음
      mode = "HARD_DEVICE"
  else:
      # 2-1) Matryoshka router: get Top-M device candidates
      C = ROUTE_BY_MATRYOSHKA(q_norm, dim=128, topM=3)
      # 2-2) Family expansion (절차 문서에만 적용)
      S_device = C ∪ UNION(Family(c) for c in C)
      S_equip  = None
      mode = "ROUTED_FAMILY"

  # 3) Retrieve under scope with scope_level-aware filtering
  candidates = HYBRID_RRF_RETRIEVE(
      q_norm,
      filter = BUILD_SCOPE_FILTER(S_device, S_equip),
      topN = 60
  )
  # BUILD_SCOPE_FILTER 로직:
  #   scope_level=shared  → 항상 허용
  #   scope_level=device  → device_name ∈ S_device
  #   scope_level=equip   → S_equip이면 equip_id ∈ S_equip,
  #                          없으면 device_name ∈ S_device (fallback)

  # 4) Rerank (optional, baseline/ablation)
  ranked = CROSS_ENCODER_RERANK(q_norm, candidates)  # topK output

  # 5) Contamination-aware Scoring (v0.4 신규)
  if mode == "ROUTED_FAMILY":
      conf = ROUTER_CONFIDENCE(q_norm)          # top1-top2 score gap
      lam = LAM_MAX * sigmoid(ALPHA * conf - BETA)
  else:
      lam = LAM_MAX                        # Hard 모드: 사실상 binary filter

  for d in ranked:
      v_scope = 0 if (d.is_shared or d.device_name in S_device) else 1
      d.final_score = d.rerank_score - lam * v_scope

  ranked = SORT_BY(ranked, key=final_score, desc=True)

  # 6) Generate (RAG)
  answer = LLM_GENERATE_WITH_CITATIONS(q_norm, top_docs=ranked[1..K])
  return answer, ranked, mode, S_device, S_equip, asd
```

### 코드베이스 매핑

| 의사코드 함수 | 현재 구현 | 위치 | 상태 |
|---|---|---|---|
| `NORMALIZE_QUERY_TERMS(q)` | — | — | **새로 구현 필요** (ASD 전처리) |
| `AUTO_PARSE(q)` | `auto_parse_node` | `langgraph_agent.py:2764` | 있음 (룰 기반) |
| `ROUTE_BY_MATRYOSHKA(q)` | — | — | **새로 구현 필요** |
| `Family(device)` | — | — | **새로 구축 필요** (오프라인) |
| `is_shared` 필터 | — | ES mapping에 필드 없음 | **필드 추가 필요** |
| `scope_level` 필터 | — | ES mapping에 필드 없음 | **필드 추가 필요** |
| `BUILD_SCOPE_FILTER` | — | — | **새로 구현 필요** (scope_level 기반 multi-level filter) |
| `HYBRID_RRF_RETRIEVE` | `EsHybridRetriever.retrieve()` | `es_hybrid.py:115` | 있음 |
| `device_name in S` 필터 | `build_filter(device_names=)` | `es_search.py:517` | 있음 |
| `CROSS_ENCODER_RERANK` | reranker in `retrieve_node` | `langgraph_agent.py:1243` | 있음 |
| `ROUTER_CONFIDENCE(q)` | — | — | **새로 구현 필요** (router top1-top2 gap) |
| `CONT_AWARE_SCORE` | — | — | **새로 구현 필요** (v0.4 점수함수) |

### 오프라인 준비물 → 구현 매핑

| 준비물 | 데이터 소스 | 구현 방법 |
|---|---|---|
| ASD lexicon/rules | 약어 오탐 로그 + 도메인 용어집 | `acronym_sense_lexicon.csv` 구축 + 룰 엔진 |
| D_shared 판정 | topic × device_name 분포 분석 | 스크립트로 집계 → `is_shared` 필드 업데이트 |
| scope_level 부여 | doc_type + D_shared 판정 결과 | 스크립트로 `scope_level` 필드 산출 → ES 매핑 추가 |
| Family graph | device 쌍별 공유 topic Jaccard | 스크립트로 그래프 구축 → JSON/config 저장 |
| Device prototype index | device별 대표 텍스트 생성 → Matryoshka 임베딩 | 별도 ES 인덱스 또는 in-memory FAISS |

---

## 4) 메트릭 정의(논문 핵심)

### 4.1 Retrieval contamination (주 메트릭) — 3종 동시 보고

| 메트릭 | 정의 | 역할 |
|--------|------|------|
| **Raw Cont@k** | shared도 타 장비면 오염 처리 (엄격) | 투명성/기준선 |
| **Adjusted Cont@k** | shared 제외한 실질 오염 | **주장 메트릭** |
| **Shared@k** | top-k 중 shared 문서 비율 | 도메인 특성 증명 |

```
Raw_Cont@k = (1/k) * Σ 1[device(d_i) ∉ S(q)]
Adj_Cont@k = (1/k) * Σ 1[d_i ∉ D_shared ∧ device(d_i) ∉ S(q)]
Shared@k   = (1/k) * Σ 1[d_i ∈ D_shared]
```

* **ContamExist@k**: 하나라도 out-of-scope이면 1

```
CE@k = 1[∃ i <= k : d_i is out-of-scope and not shared]
```

* Shared 임계치 T 민감도 분석 → Appendix

**Appendix 후보: Equip-level Contamination** (데이터 충분성 확인 후 결정)

| 메트릭 | 정의 | 적용 조건 |
|--------|------|-----------|
| **Equip-Cont@k** | top-k 중 equip_id가 허용 밖인 비율 | 질의가 equip-target일 때만 |

> Device-Cont@k(본문)와 Equip-Cont@k(Appendix)를 분리 보고하면 계층형 정책의 효과를 더 명확하게 증명.
> 단, 현재 코퍼스에서 equip-level 문서(myservice/gcb)가 충분한지 먼저 확인 필요.

### 4.2 Retrieval quality (리콜 손실 방지)

* **Hit@k / MRR**: expected_doc (또는 expected_pages) 기준
* **ScopeAccuracy@M**: 정답 device가 router Top-M에 포함되는 비율
  (장비명 마스킹 셋에서 특히 중요)

### 4.3 Generation contamination (보조 메트릭)

* **Citation Contamination**

```
CiteCont = #(out-of-scope citations) / #(all citations)
```

### 4.4 Doc-type drift (보조 분석, Appendix 후보) (v0.4 신규)

* **TypeDrift@k**: 의도와 무관한 doc_type이 top-k에 과도하게 등장하는 비율

```
TypeDrift@k = (1/k) · Σ 1[doc_type(d_i) ∉ expected_types(intent(q))]
```

* 메인 메트릭이 아닌 **분석/Discussion 절**에서 contamination의 또 다른 차원으로 제시
* 특히 myservice 편중 코퍼스에서 SOP 질의 시 로그가 상위에 등장하는 패턴 분석에 유용

### 4.5 Acronym Sense Metrics (보조 분석, v0.7 추가)

약어 다의어(예: AR) 관련 보조 지표:

| 메트릭 | 정의 | 목적 |
|---|---|---|
| `ASD-Acc` | `#(정답 sense 예측) / #(약어 다의어 질의)` | 전처리 품질 확인 |
| `ASD-Abstain` | `#(abstain) / #(약어 다의어 질의)` | 모호 질의 보수성 측정 |
| `Sense-Cont@k` | top-k 중 의도 sense와 충돌하는 문서 비율 | 의미 혼입 정량화 |

```
SenseCont@k = (1/k) * Σ 1[sense(d_i) ∉ allowed_senses(q)]
```

> 주의: 본 지표는 Paper A 메인 성과지표가 아니라, AR류 오탐을 줄이기 위한 보조 분석으로 보고한다.

---

## 5) Baseline / Ablation Matrix (실험 표)

| ID | Scope 결정 | Retrieval | Rerank | 목적 |
|----|-----------|-----------|--------|------|
| B0 | 없음(글로벌) | BM25 | X | 최저선 |
| B1 | 없음(글로벌) | Dense | X | dense-only |
| B2 | 없음(글로벌) | Hybrid+RRF | X | 현 표준 |
| B3 | 없음(글로벌) | Hybrid+RRF | O | strong baseline |
| B4 | auto-parse Hard | Hybrid+RRF (filter) | O | "필터만" 효과 (≈현 프로덕션) |
| A1 | B4 + ASD 전처리 | Hybrid+RRF (filter) | O | 약어 다의어 보정 효과 (예: AR) |
| P1 | Hard + Shared | Hybrid+RRF | O | 공용문서 정책 효과 |
| P2 | Matryoshka Router Top-M | Hybrid+RRF (filter) | O | **G(라우팅) 핵심** |
| P3 | Router + Family | Hybrid+RRF | O | 유사장비/공유 SOP 대응 |
| P4 | Router + Family + Shared | Hybrid+RRF | O | 최종 제안(기존 권장) |
| P5 | (선택) F 구현 | per-device index | O | 효율/레이턴시 비교(보너스) |
| P6 | P4 + Cont-aware Scoring (λ 고정) | Hybrid+RRF | O | **C5 핵심**: 고정 λ로 scope penalty |
| P7 | P4 + Cont-aware Scoring (λ(q) 적응형) | Hybrid+RRF | O | **C4+C5 통합**: confidence 기반 λ 조절 — **최종 제안** |

**Matryoshka ablation(권장)**

* dim: {64, 128, 256, 768} — 메인 ablation
* M: {1, 3, 5} — 메인 ablation
* Family 확장 크기 제한 L: {0, 3, 10} — Appendix

모든 실험은 **Explicit / Masked / Ambiguous 서브셋별로 분리** 보고.
ASD 실험은 별도 `Acronym-Ambiguous` subset에서 `B4 vs A1` 중심으로 비교.

---

## 6) 데이터 구성

* **Original SOP79 (Explicit)**: 장비명 100% 포함(현재 셋) → hard scope 성능 확인용
* **Mask set**: SOP79에서 device/equip 토큰 제거/치환 → **라우팅 필요성 검증용**
* **Ambiguous challenge set**: 공유 topic(controller, FFU, robot 등)이 많은 장비/문서만 골라 구성
* **Real-Implicit set** (가능 시): 운영 로그에서 auto_parse가 device를 못 잡은 질의 (최소 200-300개)
* **Acronym-Ambiguous set** (신규): 약어 다의어 질의(예: `AR 수치 저하`) 최소 80개

---

## 7) 통계 검정

* 질의 단위로 **paired bootstrap**으로 Cont@k, MRR 차이 CI 제시
* CE@k 같은 이진은 **McNemar** 또는 bootstrap on proportions
* p<0.05, 효과크기 (delta Cont@k, delta MRR) 함께 보고

---

## 8) 논문 기여 문장 템플릿

* "우리는 반도체 유지보수 RAG에서 **Hierarchy-aware scope routing**과 **shared/family 허용 정책**을 통해 contamination을 줄이면서도 recall을 유지하는 방법을 제안한다."
* "장비 타입(`device_name`)과 설비 인스턴스(`equip_id`) 2-level 계층에 따라 **문서별 scope level(shared/device/equip)**을 분리 적용하여, 절차 문서는 family 확장으로 recall을 보존하고 로그/이력은 equip_id로 정밀 필터링한다."
* "특히 **Matryoshka 저차원 라우터**를 통해 스코프 후보를 효율적으로 생성하여 대규모 로그 코퍼스에서도 실용적 latency로 동작함을 보였다."

---

## 9) 문서 기반 현황 평가 (2026-03-04)

> 근거 문서:
> - `docs/2026-03-01_page_accuracy_improvement_report.md`
> - `docs/2026-03-02_agent_retrieval_todo.md`
> - `docs/2026-03-04_agent_retrieval_todo-review.md`
> - `docs/papers/10_common_protocol/paper_common_protocol.md`
> - `docs/papers/00_strategy/ie_phd_rag_reliability_roadmap.md`

### 9.1 현재 강점

1. 문제정의가 명확함
   - 단순 정확도 개선이 아니라 `contamination 감소 + recall 보존 + 운영 가능성(latency)`으로 목표가 정리됨.
2. 실서비스 근거가 존재함
   - `/api/agent/run` 기준 페이지 정확도 이슈와 개선 이력이 축적되어 있어 실증 스토리를 만들 수 있음.
3. 공통 실험 프로토콜이 이미 있음
   - Golden set 스키마, baseline 4종, 통계 검정 기준(bootstrap/McNemar)이 문서화되어 있어 Paper A 실험을 빠르게 시작 가능.

### 9.2 현재 약점/리스크

1. Paper A 전용 데이터셋이 아직 약함
   - SOP79는 장비명 명시 비율이 높아 라우팅 기여를 강하게 증명하기 어려움.
2. Shared 정책의 착시 위험
   - shared 허용 정책이 contamination 수치를 과도하게 낮춰 보이게 만들 수 있음.
   - → Raw + Adjusted 동시 보고로 대응 (D2에서 확정)
3. 코드 baseline 고정 리스크
   - stage2/strict override/sticky 동작 관련 보완 이슈가 존재해, 실험 기준선이 흔들릴 수 있음.
4. Matryoshka 포지셔닝 리스크
   - Matryoshka를 주기여로 두면 "효율화 엔지니어링"으로 보일 가능성이 높음.
   - → scope 정책(G)의 보조기여로 고정 (D6에서 확정)

### 9.3 논문화 관점 판정

- **권장 메인 기여**: G (Hierarchy-aware scope routing policy)
- **권장 보조 기여**:
  - F (single-index filter vs per-device index 선택 비교: 효율 실험)
  - Matryoshka (라우팅 단계 비용 절감)
- **최소 성립 조건**:
  1. `Cont@k(raw)` + `Cont@k(adjusted)` + `Shared@k` 동시 보고
  2. `DocHit@k`와 `PageHit@k` 동시 보고
  3. Explicit/Masked/Ambiguous 최소 3개 서브셋으로 결과 분리

---

## 10) 의사결정 로그 (확정)

### D1. 투고 트랙 — **CIKM Applied/Industry 1차, SIGIR Industry 2차**
- 산업 적용 + 실증(오염 감소/효율)을 논문 가치로 정면 평가하는 venue
- NLP venue(ACL/EMNLP)는 현재 무게중심(retrieval 설계)과 맞지 않음

### D2. Contamination 보고 — **Raw + Adjusted + Shared@k 3종 동시 보고**
- Raw: shared도 타 장비면 오염 (엄격 기준선)
- Adjusted: shared 제외한 실질 오염 (주장 메트릭)
- Shared@k: top-k 중 shared 비율 (도메인 특성 증명)
- 임계치 T 민감도 분석은 Appendix

### D3. Family 구성 — **공유 topic 그래프 기반 1순위**
- 메인: topic 공유 기반 Jaccard 그래프 (코퍼스 현실: 같은 topic이 장비별 별도 doc_id)
- 핵심: `w(topic) = 1/log(1 + device_count(topic))` 가중치 감쇠
- 보조: 임베딩 centroid family

### D4. Real-Implicit 질의 — **운영 로그에서 확보 시도 + Mask set 안전장치**
- Real-Implicit: 최소 200-300개, single-turn 추론 가능한 것만
- Mask set: SOP79에서 device 토큰 제거 → 통제된 비교

### D5. 실험 경로 — **retrieval-only 주 실험, /api/agent/run 보조**
- B4(auto-parse Hard) ≈ 현 프로덕션 → Explicit vs Implicit 분리 보고 필수

### D6. Matryoshka 전략 — **2단계 접근**
- 1단계: 라우터(Top-M device)에만 적용
- 2단계(여력 시): 문서 chunk 임베딩도 교체
- 필수: 현재 embedding 모델 MRL 지원 확인

### D7. Scope Level 계층화 — **문서별 scope_level 3단 분류**
- `shared` / `device` / `equip` 3단 분류를 doc_type 기반으로 부여
- SOP/Manual/TS → `device`, 로그/이력(myservice/gcb) → `equip`, D_shared → `shared`
- equip_id는 "추정" 대상이 아닌 "확인" 대상 (파서가 잡으면 사용, 못 잡으면 device fallback)
- Family 확장은 절차 문서에만 적용, 로그/이력에는 미적용
- Equip-Cont@k는 데이터 충분성 확인 후 Appendix 후보로 결정

### D8. Contamination-aware Scoring (C5) — **점수함수에 scope 위반 penalty 반영** (v0.4)
- `Score(d,q) = Base(d,q) - λ(q) · v_scope(d,q)` 형태의 목적함수 도입
- 기존 binary filter를 대체하는 것이 아니라 **filter 이후 reranking 단계에서 추가 적용**
- v_scope = 1[d ∉ D_shared ∧ device(d) ∉ S(q)] (binary indicator)
- 이 점수함수가 논문의 **알고리즘적 기여**를 구성하는 핵심 (저널급 차별화)
- 근거: PE 1차 피드백 — "메트릭을 직접 최적화하는 정책"으로 기여 격상

### D9. Hard↔Soft 적응형 전환 (C4) — **C5의 λ(q) 적응형 조절로 통합** (v0.4)
- C4를 별도 기여로 세우지 않고, C5 점수함수의 λ(q) 조절 메커니즘으로 자연스럽게 통합
- `λ(q) = λ_max · σ(α · confidence(q) - β)` — router confidence 기반 sigmoid 스케일링
- confidence = router Top-1 vs Top-2 score gap
- 파서가 장비명을 확정한 Hard 모드에서는 λ=λ_max (binary filter에 수렴)
- Router 모드에서만 적응형 λ 적용
- ablation: P6(λ 고정) vs P7(λ 적응형) 비교로 C4 효과 실증
- doc_type penalty (λ₂·v_type)는 Appendix 후보로 분리 — 본문 scope 관리

### D10. 약어 다의어(ASD) 포지셔닝 — **보조 모듈로 제한** (v0.7)
- AR류 다의어는 실제 오염 원인이지만, Paper A core 주장(스코프 정책)과 분리해 **exploratory ablation**으로만 보고
- 메인 테이블은 기존 B0~P7 유지, ASD는 `A1` 보조 표/부록으로 보고
- 결과 해석 시 `ASD-Acc`, `Sense-Cont@k`를 contamination 보조 증거로 사용

---

## 10-추가) 초기 실험 파라미터 고정값 (v0.2)

- `T(shared) = 3`: deg(doc) >= 3이면 shared 후보 (doc_type 조건과 함께)
- `tau(family) = 0.2`: Jaccard 그래프 family 연결 임계치
- `M(top device) = 3`: 라우터 device 후보 개수
- `router_dim = 128`: Matryoshka 라우터 기본 차원 (ablation: 64/128/256)

운영 원칙:
- 위 값은 **dev set에서만 튜닝**하고 test set에서는 고정.
- 본문에는 고정값 결과를 보고하고, 민감도 분석은 Appendix로 분리.

---

## 10-추가) 코퍼스 통계 요약 (2026-03-04 확인)

> 상세: `evidence/2026-03-04_corpus_statistics.md`

| 항목 | 값 | 시사점 |
|------|-----|--------|
| 총 문서 | 578건 | |
| 고유 device | 21종 (실질 6종 주요) | router Top-M의 선택적 가치 제한적 |
| 고유 topic | 418종 | |
| SOP 비율 | 87.7% (pdf+pptx) | SOP 중심 코퍼스 |
| 2개+ 장비 공유 topic | **77개 (21%)** | family/shared 정책 필요성 확인 |
| 3개+ 장비 공유 topic | **29개 (7.9%)** | shared 후보 존재 확인 |
| 주요 family | SUPRA(329), INTEGER(89), PRECIA(71), GENEVA(69), OMNIS(15) | 5개 family로 99% 커버 |

**핵심 발견**: 같은 topic이지만 장비별로 별도 문서 → **topic 공유 기반 유사도**가 family 구축에 더 적합.

## 10-추가) 미확정 잔여 질문

1. 현재 embedding 모델의 MRL 지원 여부 확인
2. ES 인덱스 실제 chunk 수 vs 파싱 문서 578건 차이
3. ~~doc_id 공유 vs topic 공유: family 구축 기준 최종 확정~~ → D3에서 topic 기반 확정
4. equip_id null 비율 (doc_type별) — equip-level 실험 실현 가능성 판단
5. 현재 코퍼스에 myservice/gcb(로그/이력) 문서가 몇 건인지 확인

---

## 11) 즉시 실행 체크리스트 (Paper A 착수용)

- [x] (v0.4) `device_catalog.csv` 생성 — 21개 device, 5개 family (`data/paper_a/metadata/`)
- [x] (v0.4) `doc_type_map.csv` 생성 — 6개 raw → 5개 norm + doc_group (`data/paper_a/metadata/`)
- [x] (v0.4) `document_scope_table.csv` 생성 — 585건 auto-labeled (`data/paper_a/corpus_labels/`)
- [x] (v0.4) `shared_doc_gold.csv` 생성 — 151행, 54 topics, 13 shared topics (`data/paper_a/corpus_labels/`)
- [x] (v0.4) `device_family_gold.csv` 생성 — 210 pairs, Jaccard 기반 (`data/paper_a/corpus_labels/`)
- [x] (v0.4) `query_gold_master.jsonl` 생성 — **380건** PE 최소 요건 충족 (`data/paper_a/eval/`)
  - Explicit: 150건 ✓ (v1 15 + v2 20 + SOP79 79 + cross-device 36)
  - Implicit/Masked: 150건 ✓ (auto-masked 131 + synthetic_implicit 19)
  - Ambiguous: 80건 ✓ (shared topic 기반 multi-device 질의)
- [x] (v0.4) `equip_catalog.csv` 생성 — 2,718개 equip_id, GCB 기반 (`data/paper_a/metadata/`)
- [x] (v0.4) Masked set 자동 생성 — device 토큰 → [DEVICE] 치환
- [x] (v0.4) Ambiguous challenge set — shared topic + component/alarm/process 기반
- [x] (v0.4) Cross-device balance — GENEVA/INTEGER/PRECIA/OMNIS 보강
- [ ] scope_level 부여 규칙 고정 (doc_type → scope_level 매핑 + D_shared override)
- [ ] equip_id null 비율 집계 (doc_type별) — equip-level 실험 가능성 판단
- [ ] Family(device) 그래프 1차 구축 (Jaccard 임계치 실험 포함)
- [ ] Router baseline 확정 (auto-parse only vs Matryoshka top-M)
- [ ] 평가 리포트 템플릿 확정 (`Cont raw/adjusted/shared`, `DocHit`, `PageHit`, latency)
- [ ] `acronym_sense_lexicon.csv` 구축 (AR, PR, RF, OES 우선)
- [ ] ASD 전처리 구현 (`NORMALIZE_QUERY_TERMS`) + `asd_confidence` 로깅
- [ ] Acronym-Ambiguous set 구축 (최소 80질의, sense gold 포함)
- [ ] ASD ablation 실행 (`B4` vs `A1`) + `ASD-Acc`, `Sense-Cont@k` 산출
- [x] query_gold_master 확장: Explicit 150 + Implicit 150 + Ambiguous 80 = 380건 확보
- [ ] (조건부) Equip-Cont@k 포함 여부 결정 — equip-level 데이터 충분성 확인 후
- [ ] (v0.4) C5 Contamination-aware scoring 구현: `Score = Base - λ·v_scope`
- [ ] (v0.4) λ(q) 적응형 조절 구현: router confidence → sigmoid 스케일링
- [ ] (v0.4) P6(λ 고정) vs P7(λ 적응형) ablation 실험 설계
- [ ] (v0.4, Appendix) doc_type drift 보조 분석 — TypeDrift@k 산출

---

## 12) 평가 코드 구현 청사진 (retrieval-only 주 실험)

### 12.1 입력 스키마 (`query_gold_master.jsonl`)

> v0.4에서 PE 스키마로 확장. 상세 컬럼 정의는 §13.3 참조.

```json
{
  "q_id": "A-gs101",
  "split": "dev",
  "source": "golden_set_v2",
  "question": "PCW Flow Limit Alarm 발생 시 Flow Switch 교체 기준은?",
  "question_masked": "",
  "scope_observability": "explicit_device",
  "intent_primary": "troubleshooting",
  "target_scope_level": "device",
  "canonical_device_name": "SUPRA_N",
  "canonical_equip_id": "",
  "allowed_devices": ["SUPRA_N"],
  "allowed_equips": [],
  "preferred_doc_types": ["myservice"],
  "acceptable_doc_types": ["ts", "sop"],
  "gold_doc_ids": ["40052288", "40033437"],
  "shared_allowed": true,
  "family_allowed": true
}
```

필수 필드: `q_id`, `question`, `split`, `scope_observability`, `allowed_devices`, `shared_allowed`
선택 필드: 나머지 전부 (상세 §13.3)

파일 위치: `data/paper_a/eval/query_gold_master.jsonl`

### 12.2 코퍼스 메타 전처리 출력

> v0.4에서 PE 스키마로 확장. 상세 컬럼 정의는 §13.2 참조.

평가 전에 아래 파일을 로드:

- `data/paper_a/corpus_labels/document_scope_table.csv` → `doc_meta[doc_id]`
- `data/paper_a/corpus_labels/device_family_gold.csv` → `family_map[device]`
- `data/paper_a/corpus_labels/shared_doc_gold.csv` → `shared_docs`
- `data/paper_a/metadata/device_catalog.csv` → device 정규화
- `shared_threshold_T` (예: 3)

### 12.3 평가 스크립트 함수 구조

핵심 함수 시그니처(권장):

- `build_allowed_scope(query_row, parse_result, family_map, policy_cfg)`
- `is_out_of_scope(hit, allowed_scope, doc_meta)`
- `compute_metrics_per_query(hits, query_row, allowed_scope, doc_meta, k_list)`
- `run_system(system_id, query_rows, policy_cfg, backend_cfg)`
- `aggregate_metrics(per_query_rows, by=["all","split","doc_type"])`
- `run_paired_bootstrap(per_query_rows, metric_name, a_id, b_id, n_boot=10000)`

### 12.4 산출물 파일(고정)

평가 1회 실행마다 아래 파일을 생성:

- `results/per_query.csv`
- `results/summary_all.csv`
- `results/summary_by_split.csv`
- `results/summary_by_doc_type.csv`
- `results/bootstrap_ci.json`

### 12.5 지표 계산 규칙(고정)

- contamination: `Raw Cont@k`, `Adjusted Cont@k`, `Shared@k`
- 품질: `Hit@k`, `MRR@k`
- 라우팅: `ScopeAccuracy@M` (라우터 사용 실험에서 필수)
- 보조(옵션): `Equip-Cont@k` (equip-target subset에서만)

### 12.6 실험군 고정표 (논문 표 1:1 매핑)

- Baseline: `B0`, `B1`, `B2`, `B3`, `B4`
- ASD Ablation: `A1` (`B4 + ASD`)
- Proposed: `P1`, `P2`, `P3`, `P4`, `P6`, `P7`
- Optional: `P5` (F 구현, 효율 비교)
- Matryoshka ablation: `dim={64,128,256,768}`, `M={1,3,5}`
- Scoring ablation: `P6`(λ 고정) vs `P7`(λ(q) 적응형) — C4+C5 효과 실증

### 12.7 고정 파라미터 (v0.2)

- `T(shared)=3`
- `tau(family)=0.2`
- `M(top device)=3`
- `router_dim=128`

원칙: dev set에서만 튜닝, test set에서는 고정.

---

## 13) 실험 데이터셋 정의 (v0.5 신규)

### 13.0 데이터셋 총괄 (6종)

| # | 데이터셋 | 우선순위 | 용도 |
|---|---------|---------|------|
| 1 | 정규화 메타데이터 (device/equip/doc_type catalog) | 1순위 | 모든 실험의 바닥 |
| 2 | 문서 적용 범위 (document_scope_table) | 1순위 | contamination 계산 핵심 |
| 3 | 질의 평가셋 (query_gold_master) | 1순위 | B0-P7 실험 실행 |
| 4 | Shared/Family 검증셋 | 2순위 | 정책 정의 검증 |
| 5 | Intent-DocType 라벨셋 | 2순위 | C5/doc_type drift 검증 |
| 6 | 학습용 pair 셋 (reranker/router) | 3순위 | 저널 고도화용 |

### 13.1 정규화 메타데이터 테이블 (1순위)

> **원칙**: device_name/equip_id/doc_type이 흔들리면 contamination 계산 전부 틀어짐.

#### `device_catalog.csv`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `device_name_raw` | str | ES 원본 값 |
| `device_name_norm` | str | 정규화된 이름 |
| `device_aliases` | str (`\|` 구분) | 알려진 별칭들 |
| `family_seed` | str | 초기 family 힌트 (있으면) |
| `notes` | str | 비고 |

#### `equip_catalog.csv`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `equip_id_raw` | str | ES 원본 값 |
| `equip_id_norm` | str | 정규화된 ID |
| `device_name_norm` | str | 소속 device |
| `equip_aliases` | str | 별칭 |
| `fab_or_line` | str | 소속 FAB/라인 (있으면) |
| `notes` | str | 비고 |

#### `doc_type_map.csv`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `doc_type_raw` | str | ES/코퍼스 원본 값 |
| `doc_type_norm` | str | 논문용 5종: `sop`, `ts`, `manual`, `gcb`, `myservice` |
| `doc_group` | str | `procedure` (sop/manual/ts) 또는 `log_history` (gcb/myservice) |

### 13.2 문서 적용 범위 테이블 (1순위)

> **이 데이터가 논문에서 contamination 정의를 지키는 핵심 데이터셋.**

파일: `document_scope_table.csv` (또는 `.parquet`)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `doc_id` | str | ES doc_id |
| `doc_type_norm` | str | 정규화 doc_type |
| `device_names` | list[str] | 소속 device 목록 |
| `equip_ids` | list[str] | 소속 equip 목록 |
| `scope_level_gold` | str | `shared` / `device` / `equip` / `unknown` |
| `shared_scope_type` | str | `none` / `global` / `family` / `multi_device` |
| `family_id` | str | 소속 family (있으면) |
| `is_shared_gold` | bool | shared 여부 |
| `source_confidence` | str | `auto` / `human_verified` / `mixed` |
| `notes` | str | 비고 |

**shared_scope_type 세분화**:
* `global`: 거의 모든 장비 공용
* `family`: 특정 유사 장비군끼리 공용
* `multi_device`: 일부 여러 장비에 걸침

**1차 생성 방식** (query-centric):
* 평가 질의의 `gold_doc_id`에 해당하는 문서
* baseline 검색 top-k에서 자주 섞이는 문서
* shared 후보 문서
→ 전체 코퍼스가 아니라 **실험에 등장하는 문서 우선**.

### 13.3 질의 평가셋 — Master Query Gold Set (1순위)

파일: `query_gold_master.jsonl`

| 컬럼 | 필수 | 타입 | 설명 |
|------|------|------|------|
| `q_id` | O | str | 질의 ID |
| `split` | O | str | `explicit` / `implicit` / `ambiguous` / `equip_centric` |
| `source` | O | str | 출처 (SOP79, masked, log 등) |
| `question` | O | str | 원본 질의 |
| `question_masked` | - | str | device/equip 토큰 제거 버전 |
| `scope_observability` | O | str | `explicit_device` / `explicit_equip` / `implicit` / `ambiguous` / `context_dependent` |
| `intent_primary` | O | str | 주 의도 (procedure / troubleshooting / history_lookup 등) |
| `intent_secondary` | - | str | 부 의도 |
| `target_scope_level` | O | str | 정답이 요구하는 scope: `shared` / `device` / `equip` |
| `canonical_device_name` | O | str | 정규화 device |
| `canonical_equip_id` | - | str | 정규화 equip_id |
| `allowed_devices` | O | list[str] | contamination 계산용 허용 device 목록 |
| `allowed_equips` | - | list[str] | contamination 계산용 허용 equip 목록 |
| `preferred_doc_types` | - | list[str] | 최적 doc_type (C5 검증용) |
| `acceptable_doc_types` | - | list[str] | 허용 doc_type (TypeDrift 검증용) |
| `gold_doc_ids` | O | list[str] | 정답 문서 |
| `gold_pages` | - | list[str] | 정답 페이지 |
| `gold_passages` | - | list[str] | 정답 패시지 |
| `shared_allowed` | O | bool | shared 문서 허용 여부 |
| `family_allowed` | O | bool | family 확장 허용 여부 |
| `notes` | - | str | 비고 |

#### 평가셋 하위 구성 (4개 subset)

| Subset | 목적 | 최소 | 권장 | 구성 비율 |
|--------|------|------|------|----------|
| Explicit | baseline 회귀 확인 | 150 | 300 | procedure 40%, TS 30%, log/history 30% |
| Implicit/Masked | 라우팅(G)/Router 검증 | 150 | 300 | explicit에서 device/equip 토큰 제거 |
| Ambiguous/Shared/Family challenge | trade-off 해결 실증 | 80 | 150 | shared 정답, family 문서 정답, hard filter 리콜 깨지는 케이스 |
| Equip-centric log/history | equip_id 가치 실증 | 100 | 200 | 알람 이력, 반복 에러, 교체 이력, 정비 기반 추정 |

> **주의**: explicit과 masked는 같은 원문이면 같은 split에 묶기 (data leakage 방지).

### 13.4 Shared/Family 검증셋 (2순위)

#### `shared_doc_gold.csv` (100~200 문서)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `doc_id` | str | 문서 ID |
| `doc_type_norm` | str | 정규화 doc_type |
| `candidate_devices` | list[str] | 매핑된 device 목록 |
| `candidate_equips` | list[str] | 매핑된 equip 목록 |
| `is_shared_gold` | bool | shared 여부 gold label |
| `shared_scope_type_gold` | str | `global` / `family` / `multi_device` / `none` |
| `notes` | str | 비고 |

#### `device_family_gold.csv` (50~100 pair)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `device_a` | str | device 1 |
| `device_b` | str | device 2 |
| `same_family_seed` | bool | 같은 family 여부 |
| `evidence_type` | str | `shared_doc` / `engineer_judgment` / `same_module_structure` / `not_family` |
| `notes` | str | 비고 |

### 13.5 Intent-DocType 라벨셋 (2순위)

> "장비 인식 후 task에 따라 doc_type을 정하는 구조" 검증용. C5 doc_type penalty + TypeDrift@k 분석에 필요.

파일: `intent_doctype_labels.jsonl` (200~400개)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `q_id` | str | 질의 ID |
| `question` | str | 질의 텍스트 |
| `intent_primary` | str | `procedure` / `troubleshooting` / `history_lookup` / `spec_check` / `alarm_analysis` |
| `intent_secondary` | str | 부 의도 |
| `preferred_doc_types` | list[str] | 최적 doc_type |
| `acceptable_doc_types` | list[str] | 허용 doc_type |
| `requires_equip_level` | bool | equip_id 필요 여부 |
| `notes` | str | 비고 |

**intent → doc_type 가중치 예시**:

| intent | sop | manual | ts | myservice | gcb |
|--------|-----|--------|-----|-----------|-----|
| procedure | 1.0 | 0.7 | 0.4 | 0.1 | 0.1 |
| troubleshooting | 0.4 | 0.3 | 1.0 | 0.4 | 0.3 |
| history_lookup | 0.2 | 0.1 | 0.4 | 1.0 | 0.8 |
| alarm_analysis | 0.2 | 0.1 | 0.6 | 1.0 | 0.8 |
| spec_check | 0.3 | 1.0 | 0.3 | 0.1 | 0.1 |

### 13.6 학습용 Pair 셋 (3순위, 저널 고도화용)

#### `reranker_pairs.jsonl` (1,000~5,000+)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `q_id` | str | 질의 ID |
| `question` | str | 질의 텍스트 |
| `doc_id` | str | 문서 ID |
| `label_relevance` | int | 0/1/2 |
| `label_scope_violation` | int | 0/1/2 |
| `negative_type` | str | negative 유형 |
| `doc_type_norm` | str | 정규화 doc_type |
| `device_name_norm` | str | 정규화 device |
| `equip_id_norm` | str | 정규화 equip_id |

**negative_type 분류**:
* `same_device_wrong_doc`: 같은 장비, 다른 문서
* `same_family_wrong_device`: 같은 family, 다른 장비
* `other_device_semantic_confusion`: 의미적으로 유사하지만 타 장비
* `wrong_equip_same_device`: 같은 장비, 다른 설비 (equip-level 오염)
* `wrong_doc_type`: 의도와 무관한 doc_type

#### `router_train.jsonl`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `q_id` | str | 질의 ID |
| `question` | str | 질의 텍스트 |
| `target_device` | str | 정답 device |
| `target_equip` | str | 정답 equip_id |
| `candidate_devices` | list[str] | 후보 device 목록 |
| `intent_primary` | str | 주 의도 |

### 13.7 데이터 생성 원칙

1. **myservice 편중 균형화**: doc_type / device / intent 기준으로 균형 샘플링. 그대로 뽑으면 로그 질의만 과다.
2. **Leakage 방지**: explicit과 masked는 같은 원문이면 같은 split에 묶기.
3. **allowed scope 명시 필수**: 정답 문서만으로는 부족. `allowed_devices` / `allowed_equips` / `shared_allowed`가 있어야 contamination 계산 가능.
4. **shared 세분화**: `global` / `family` / `multi_device` 구분이 있어야 hard filter vs family policy 차이가 살아남.
5. **equip_id canonicalization 최우선**: 임베딩보다 정규화가 먼저.

### 13.8 최소 실행 버전 (리소스 제약 시)

1. device/equip/doc_type 정규화 테이블
2. `query_gold_master.jsonl` (explicit + masked 최소 150건씩)
3. query에 걸리는 `gold_doc_ids`와 혼입 후보 문서에 대한 `document_scope_table`

→ **논문에 등장하는 질의/문서 주변만 먼저 라벨링**해도 시작 가능.

### 13.9 파일 구조 (권장)

```
data/paper_a/
├── metadata/
│   ├── device_catalog.csv
│   ├── equip_catalog.csv
│   └── doc_type_map.csv
├── corpus_labels/
│   ├── document_scope_table.csv
│   ├── shared_doc_gold.csv
│   └── device_family_gold.csv
├── eval/
│   └── query_gold_master.jsonl
└── train_optional/
    ├── intent_doctype_labels.jsonl
    ├── reranker_pairs.jsonl
    └── router_train.jsonl
```
