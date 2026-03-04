# Paper A 실험 정의서 v0.3

## Hierarchy-aware Scope Routing (G) + Family Scope + Shared Doc Policy + Matryoshka Router

## 0) 목표

반도체 유지보수 RAG에서 **cross-equipment contamination(타 장비 문서 혼입)**을 줄이되, **공용 SOP/유사 장비로 인한 recall 손실을 최소화**하는 스코프 정책을 설계/검증한다.

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

* `sop_pdf`, `sop_pptx`, `setup_manual`, `ts` → `device`
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

### 2.2 Family(device) 구축(유사 장비/공유 문서 기반)

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

---

## 3) 온라인 파이프라인(의사코드)

```pseudo
function ANSWER(q):
  # 1) Parse (이미 auto-parse 존재)
  parsed = AUTO_PARSE(q)  # device_name?, equip_id?, intent?

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
      C = ROUTE_BY_MATRYOSHKA(q, dim=128, topM=3)
      # 2-2) Family expansion (절차 문서에만 적용)
      S_device = C ∪ UNION(Family(c) for c in C)
      S_equip  = None
      mode = "ROUTED_FAMILY"

  # 3) Retrieve under scope with scope_level-aware filtering
  candidates = HYBRID_RRF_RETRIEVE(
      q,
      filter = BUILD_SCOPE_FILTER(S_device, S_equip),
      topN = 60
  )
  # BUILD_SCOPE_FILTER 로직:
  #   scope_level=shared  → 항상 허용
  #   scope_level=device  → device_name ∈ S_device
  #   scope_level=equip   → S_equip이면 equip_id ∈ S_equip,
  #                          없으면 device_name ∈ S_device (fallback)

  # 4) Rerank (optional, baseline/ablation)
  ranked = CROSS_ENCODER_RERANK(q, candidates)  # topK output

  # 5) Generate (RAG)
  answer = LLM_GENERATE_WITH_CITATIONS(q, top_docs=ranked[1..K])
  return answer, ranked, mode, S_device, S_equip
```

### 코드베이스 매핑

| 의사코드 함수 | 현재 구현 | 위치 | 상태 |
|---|---|---|---|
| `AUTO_PARSE(q)` | `auto_parse_node` | `langgraph_agent.py:2764` | 있음 (룰 기반) |
| `ROUTE_BY_MATRYOSHKA(q)` | — | — | **새로 구현 필요** |
| `Family(device)` | — | — | **새로 구축 필요** (오프라인) |
| `is_shared` 필터 | — | ES mapping에 필드 없음 | **필드 추가 필요** |
| `scope_level` 필터 | — | ES mapping에 필드 없음 | **필드 추가 필요** |
| `BUILD_SCOPE_FILTER` | — | — | **새로 구현 필요** (scope_level 기반 multi-level filter) |
| `HYBRID_RRF_RETRIEVE` | `EsHybridRetriever.retrieve()` | `es_hybrid.py:115` | 있음 |
| `device_name in S` 필터 | `build_filter(device_names=)` | `es_search.py:517` | 있음 |
| `CROSS_ENCODER_RERANK` | reranker in `retrieve_node` | `langgraph_agent.py:1243` | 있음 |

### 오프라인 준비물 → 구현 매핑

| 준비물 | 데이터 소스 | 구현 방법 |
|---|---|---|
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

---

## 5) Baseline / Ablation Matrix (실험 표)

| ID | Scope 결정 | Retrieval | Rerank | 목적 |
|----|-----------|-----------|--------|------|
| B0 | 없음(글로벌) | BM25 | X | 최저선 |
| B1 | 없음(글로벌) | Dense | X | dense-only |
| B2 | 없음(글로벌) | Hybrid+RRF | X | 현 표준 |
| B3 | 없음(글로벌) | Hybrid+RRF | O | strong baseline |
| B4 | auto-parse Hard | Hybrid+RRF (filter) | O | "필터만" 효과 (≈현 프로덕션) |
| P1 | Hard + Shared | Hybrid+RRF | O | 공용문서 정책 효과 |
| P2 | Matryoshka Router Top-M | Hybrid+RRF (filter) | O | **G(라우팅) 핵심** |
| P3 | Router + Family | Hybrid+RRF | O | 유사장비/공유 SOP 대응 |
| P4 | Router + Family + Shared | Hybrid+RRF | O | 최종 제안(권장) |
| P5 | (선택) F 구현 | per-device index | O | 효율/레이턴시 비교(보너스) |

**Matryoshka ablation(권장)**

* dim: {64, 128, 256, 768} — 메인 ablation
* M: {1, 3, 5} — 메인 ablation
* Family 확장 크기 제한 L: {0, 3, 10} — Appendix

모든 실험은 **Explicit / Masked / Ambiguous 서브셋별로 분리** 보고.

---

## 6) 데이터 구성

* **Original SOP79 (Explicit)**: 장비명 100% 포함(현재 셋) → hard scope 성능 확인용
* **Mask set**: SOP79에서 device/equip 토큰 제거/치환 → **라우팅 필요성 검증용**
* **Ambiguous challenge set**: 공유 topic(controller, FFU, robot 등)이 많은 장비/문서만 골라 구성
* **Real-Implicit set** (가능 시): 운영 로그에서 auto_parse가 device를 못 잡은 질의 (최소 200-300개)

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

- [ ] A-Set v0 확정: Explicit / Masked / Ambiguous 분할 파일 생성
- [ ] Shared 판정 스크립트 작성 및 `is_shared` 메타 산출
- [ ] scope_level 부여 규칙 고정 (doc_type → scope_level 매핑 + D_shared override)
- [ ] equip_id null 비율 집계 (doc_type별) — equip-level 실험 가능성 판단
- [ ] Family(device) 그래프 1차 구축 (Jaccard 임계치 실험 포함)
- [ ] Router baseline 확정 (auto-parse only vs Matryoshka top-M)
- [ ] 평가 리포트 템플릿 확정 (`Cont raw/adjusted/shared`, `DocHit`, `PageHit`, latency)
- [ ] (조건부) Equip-Cont@k 포함 여부 결정 — equip-level 데이터 충분성 확인 후

---

## 12) 평가 코드 구현 청사진 (retrieval-only 주 실험)

### 12.1 입력 스키마 (`queries.jsonl`)

각 질의는 아래 필드를 기본으로 사용:

```json
{
  "qid": "A-0001",
  "query": "apc position 교체 방법 알려줘",
  "split": "explicit|implicit|ambiguous",
  "target_device": "SUPRA XP",
  "target_equip": "EPAG50",
  "gold_doc_ids": ["DOC-123"],
  "gold_passages": ["DOC-123#p14"]
}
```

필수 필드: `qid`, `query`, `split`  
선택 필드: `target_device`, `target_equip`, `gold_doc_ids`, `gold_passages`

### 12.2 코퍼스 메타 전처리 출력

평가 전에 아래 구조를 생성:

- `doc_meta[doc_id] = {device_name, equip_id, doc_type, is_shared, scope_level}`
- `family_map[device_name] = {device_name...}`
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
- Proposed: `P1`, `P2`, `P3`, `P4`
- Optional: `P5` (F 구현, 효율 비교)
- Matryoshka ablation: `dim={64,128,256,768}`, `M={1,3,5}`

### 12.7 고정 파라미터 (v0.2)

- `T(shared)=3`
- `tau(family)=0.2`
- `M(top device)=3`
- `router_dim=128`

원칙: dev set에서만 튜닝, test set에서는 고정.
