# 산업용 RAG에서 교차 장비 오염 제어를 위한 장비 인식 스코프 필터링

**초안 v2** — 2026-03-14, 마스킹 질의 실험 기반

---

## 초록

반도체 제조 환경에 배포되는 Retrieval-Augmented Generation(RAG) 시스템은 수십 종의 장비를 포괄하는 공유 코퍼스에서 장비 특화 문서를 검색해야 한다. 우리는 **교차 장비 오염(cross-equipment contamination)**, 즉 무관한 장비의 문서가 검색되는 현상을 핵심 실패 모드로 정의하며, 이 현상이 검색 방식에 따라 top-10 결과의 47–73%를 차지함을 보인다. 또한 단순한 장비 인식 하드 필터만으로 오염을 사실상 제거하면서 재현율을 최대 42%p까지 개선할 수 있음을 확인한다. 더 나아가 기존의 문서 시드(document-seeded) 평가가 gold 라벨의 어휘 편향 때문에 스코프 필터링의 가치를 체계적으로 과소평가함을 보이고, 이를 완화하는 프로토콜로 **마스킹 질의 평가(masked-query evaluation)**를 제안한다. 27개 장비 타입, 1,206개 질의(명시형 578 + 암시형 628), 4개 검색 구조(BM25, Dense, Hybrid, Hybrid+Rerank) 실험 결과는 산업용 RAG 안전성 확보에 스코프 인식 검색이 필수임을 강하게 뒷받침한다.

---

## 1. 서론

반도체 팹 환경의 대규모 RAG 시스템은 일반적으로 여러 장비 타입의 문서를 하나의 통합 코퍼스에 색인한다. 유지보수 엔지니어는 이 공유 코퍼스에 장애 조치 절차, 예방 정비 가이드, 운영 파라미터를 질의한다. 이때 검색 시스템이 잘못된 장비의 문서를 반환하면, 후단 LLM이 잘못된 절차에 근거한 답변을 생성할 수 있고, 이는 고위험 제조 현장에서 직접적인 안전 리스크가 된다.

우리는 이러한 실패 모드를 **교차 장비 오염(cross-equipment contamination)**으로 정의한다. 즉, top-k 검색 결과에 스코프 밖 장비 문서가 포함되는 현상이다. 이는 일반적인 관련도 실패와 다르다. 교차 장비 오염은 주제는 유사하지만 장비가 틀린 문서가 정답 문서를 밀어내는 체계적 편향이다. 예를 들어 SUPRA XP의 heater chuck 교체 절차와 GENEVA XP의 절차는 어휘·의미적으로 매우 유사할 수 있지만, 서로 잘못 적용하면 장비 손상으로 이어질 수 있다.

기존 도메인 특화 RAG 연구는 하이브리드 검색, 재랭킹, 질의 확장 등을 통해 전체 검색 품질을 개선하는 데 초점을 맞춰왔다. 그러나 검색 문서가 올바른 장비 맥락에 속하는지를 보는 **스코프 정합성(scope correctness)** 관점은 상대적으로 덜 다뤄졌다. 우리는 다중 장비 산업 코퍼스에서 스코프 필터링이 단순 최적화가 아니라 안전 요구사항이라고 본다.

### 기여

본 논문은 네 가지 기여를 제시한다.

기여를 나열하기 전에 방법 범위를 명확히 한다. Paper A의 **핵심 알고리즘 기여**는 contamination 중심 평가 프로토콜 하에서 장비 스코프를 제어하는 검색 정책(장비 인식 필터 + shared 문서 처리)이다. contamination-aware soft scoring 항은 주된 검색 메커니즘이 아니라 **보조 재랭킹 확장**으로 평가한다.

1. **오염 정량화**: gold 라벨 없이 계산 가능한 Contamination@k를 정의하고, 4개 검색 구조에서 측정하여 마스킹 질의 기준 47–73% 오염률을 보고한다(4절).

2. **장비 필터 효과 검증**: oracle 장비 필터링이 오염을 거의 0으로 줄이면서 strict gold hit를 50–61%에서 91–92%로 향상시키는, 즉 필터링이 오히려 재현율을 높이는 반직관적 결과를 보인다(5.1절).

3. **평가 편향 발견**: 기존 문서 시드 평가(“문서에서 질문 생성”)가 순환 편향을 만들어 스코프 필터링 가치를 가리는 문제를 밝히고, 이를 완화하는 마스킹 질의 평가 프로토콜을 제안한다(5.3절).

4. **Soft scoring 음성 결과 보고**: penalty 기반 contamination-aware soft scoring이 본 도메인에서는 하드 필터보다 비효율적임을 보인다. 원인은 penalty 항과 검색 점수 스케일의 근본적인 불일치다(5.2절).

---

## 2. 관련 연구

### 2.1 Retrieval-Augmented Generation

RAG 시스템은 검색과 생성 모델을 결합해 외부 지식에 기반한 응답을 생성한다 [Lewis et al., 2020; Guu et al., 2020]. 최근에는 dense retrieval [Karpukhin et al., 2020], sparse+dense 결합 하이브리드 접근 [Ma et al., 2021], cross-encoder 재랭킹 [Nogueira and Cho, 2019]이 검색 성능 향상을 위해 폭넓게 연구되었다.

### 2.2 도메인 특화 RAG

산업 도메인 RAG는 도메인 용어, 다국어 문서, 안전 요구 등 고유의 제약을 가진다. Peng et al. (2024)은 제조·의료 같은 응용에서 일반 QA보다 더 강한 grounding 보장이 필요하다고 지적한다. Barnett et al. (2024)은 RAG의 7개 실패 지점을 제시하며, 그중 “wrong granularity(부적절한 스코프에서의 일치)”는 본 연구의 교차 장비 오염 문제와 밀접하다. 제조 RAG 선행연구는 주로 문서 전처리와 용어 정규화에 집중했으며, 스코프 인식 검색은 상대적으로 미흡했다.

### 2.3 검색에서의 메타데이터 필터링

메타데이터 기반 사전 필터링은 상용 벡터 DB(Pinecone, Weaviate, Milvus)의 표준 기능이며, 다중 테넌트 RAG 모범사례로도 권장된다(LlamaIndex, 2024). 그러나 필터링이 오염과 재현율에 미치는 영향을 함께 체계적으로 검증한 연구는 부족하다. 기존 문서는 지연시간·확장성 중심이며, 검색 품질 관점 검증은 약하다. 본 연구는 다중 장비 산업 코퍼스에서 장비 수준 필터링을 엄밀히 평가해 필터링-재현율 상호작용이 단순하지 않음을 보여준다.

### 2.4 IR 평가 방법론

IR 커뮤니티는 pooling bias [Buckley et al., 2007], topic bias [Carterette et al., 2006] 등 평가 편향 문제를 오래전부터 다뤄왔다. 본 연구에서 발견한 문서 시드 평가 편향, 즉 문서에서 생성된 질문이 baseline 성능을 부풀리는 어휘 신호를 물려받는 현상은 특히 도메인 특화 평가셋에서 중요한 새로운 편향 축을 제시한다.

---

## 3. 문제 설정 및 방법

### 3.1 코퍼스와 장비 계층

코퍼스 $\mathcal{D}$는 반도체 제조 시설의 27개 장비 타입(device)에 걸친 508개 문서로 구성된다. 문서 유형은 SOP, 장비 매뉴얼, 트러블슈팅 가이드, 정비 로그를 포함한다. 각 문서 $d$는 `device_name(d)`, `doc_type(d)`, `topic(d)` 메타데이터를 갖는다.

장비 네임스페이스는 2단 계층 구조를 따른다.
- **device_name**: 장비 모델 (예: "SUPRA XP", "INTEGER plus")
- **equip_id**: 물리 인스턴스 식별자 (예: "EPAG50", "WPSKAU8X00")

### 3.2 교차 장비 오염

**정의.** 목표 장비 $\text{dev}(q)$를 갖는 질의 $q$와 크기 $k$의 검색 결과 집합 $R_k(q)$에 대해 오염률은 다음과 같다.

$$\text{Cont@k}(q) = \frac{|\{d \in R_k(q) : \text{dev}(d) \neq \text{dev}(q) \land d \notin \mathcal{D}_{\text{shared}}\}|}{k}$$

여기서 $\mathcal{D}_{\text{shared}}$는 교차 장비 shared 문서 집합이며(본 코퍼스 60개), $\geq 3$개 장비 간 topic 중복으로 식별한다.

**핵심 성질**: Cont@k는 gold 관련도 라벨 없이도 계산 가능하다. 장비 메타데이터만 있으면 되므로 관련도 판정과 독립적인 안전 지표로 적합하다.

### 3.3 장비 인식 하드 필터

파싱된 장비 $\text{dev}(q)$가 주어진 질의 $q$에 대해, 하드 필터는 검색 대상을 다음으로 제한한다.

$$\mathcal{D}_{\text{allowed}}(q) = \{d \in \mathcal{D} : \text{dev}(d) = \text{dev}(q)\}$$

구현은 Elasticsearch `device_name.keyword`에 대한 `terms` 필터를 BM25/kNN 점수 계산 전에 pre-filter로 적용한다.

**Oracle vs. Real**: 메인 실험에서는 gold 장비 정보를 사용해 상한(upper bound)을 측정하고, regex/dictionary 파서 적용 시 격차는 별도로 측정한다(5.4절).

### 3.4 검색 시스템

장비 필터 유무를 포함해 4개 검색 구조를 평가한다.

| System | Retrieval | Rerank | Index |
|--------|-----------|--------|-------|
| B0 | BM25 | — | chunk_v3_content |
| B1 | Dense kNN (BGE-M3, 1024d) | — | chunk_v3_embed_bge_m3_v1 |
| B2 | Hybrid (BM25 + Dense, RRF) | — | Cross-index |
| B3 | Hybrid + CrossEncoder | Yes | Cross-index |

**Cross-index 구조**: BM25 점수는 `chunk_v3_content`(텍스트 인덱스), dense 점수는 `chunk_v3_embed_bge_m3_v1`(벡터 인덱스)에서 얻고, `chunk_id`로 조인 후 RRF($k=60$)로 결합한다.

필터 적용 시스템은 다음을 추가한다.
- **B4**: B3 + 하드 장비 필터(oracle 장비)
- **B4.5**: B3 + 장비 필터 + shared 문서 허용

### 3.5 방법 범위 명확화

본 논문의 주된 방법은 penalty 기반 재랭킹이 아니라 **검색 전(pre-retrieval) 스코프 제어**(B4/B4.5)다. 즉, 먼저 장비 스코프로 후보를 제한하고, 이후 마스킹 질의 디바이어싱 평가에서 contamination과 재현율을 측정한다. 이 선택은 안전 목표를 반영한다. 스코프 밖 문서를 검색 후 감점하는 것보다, 애초에 top-k 후보로 들어오지 않게 하는 방식이 더 직접적이기 때문이다.

contamination-aware soft scoring(P6/P7)은 이러한 하드 스코프 제어를 대체할 수 있는지 검증하기 위한 비교 확장으로 포함한다.

### 3.6 Soft Scoring (P6/P7)

하드 필터의 대안으로 contamination-aware soft scoring을 평가한다.

$$\text{Score}(d, q) = \text{Base}(d, q) - \lambda \cdot v_{\text{scope}}(d, q)$$

여기서 $\text{Base}(d,q)$는 B3 재랭크 점수, $v_{\text{scope}} \in \{0,1\}$은 스코프 위반 여부, $\lambda$는 penalty 강도다. P6은 고정 $\lambda=0.05$, P7은 질의 스코프 관측성에 따라 적응형 $\lambda$를 사용한다.

### 3.7 마스킹 질의 평가 프로토콜

**동기**: 기존 평가는 문서에서 질문을 생성하기 때문에 질문에 장비명이 포함되고, BM25는 이를 문서 ID(장비명이 포함됨)와 직접 매칭해 비필터 baseline에 인위적 이점을 준다.

**프로토콜**: 장비 언급이 있는 각 질의 $q$에 대해, 장비명은 `[DEVICE]`, 장비 타입은 `[EQUIP]`으로 치환한 마스킹 질의 $q_m$을 만든다.

- 원문: "GENEVA XP 설비에서 Heater Chuck leveling 절차는?"
- 마스킹: "[DEVICE] 설비에서 Heater Chuck leveling 절차는?"

마스킹 질의는 엔지니어가 장비명을 직접 말하지 않고 절차를 묻는 현실 시나리오(세션 문맥으로 장비를 식별)와 유사한 조건을 모사한다.

**평가 매트릭스**: 모든 시스템을 원문 질의와 마스킹 질의 양쪽에서 평가해 평가 편향을 직접 비교한다.

---

## 4. 실험 설정

### 4.1 평가 데이터셋 (v0.6)

27개 장비에 걸친 578개 질의로 구성했다.
- **장비 분포 균형화**: 장비별 비례 표본
- **scope 관측성 라벨**: `explicit_device` (n=429, 장비명 명시), `explicit_equip` (n=149, 장비 타입 명시)
- **이중 gold 라벨**: `gold_strict`(직접 정답 문서), `gold_loose`(주제 관련 문서)
- **마스킹 변형**: `question_masked` 필드 제공
- **고유 gold 집합 비율**: 482/578 (83%)

### 4.2 암시형 평가 데이터셋 (v0.7)

기계적 마스킹을 넘어 일반화 가능성을 검증하기 위해, 장비명·장비 타입이 전혀 등장하지 않는 628개 암시형 질의셋을 구축했다.
- **578개 implicit 질의** (`scope_observability = ambiguous`): v0.6과 동일 source 문서에서 장비/설비 언급 없이 재구성
- **50개 trap 질의**: 교차 장비 topic 경계(예: shared 정비 절차)를 검증하는 질의
- gold 라벨은 source 문서의 장비 할당을 계승

이로써 “명시형-마스킹(v0.6)”과 “자연 암시형(v0.7)”을 직접 비교해, 기계적 마스킹이 실제 장비 비명시 검색 조건을 얼마나 충실히 근사하는지 점검한다.

### 4.3 Gold 라벨 품질 (v0.6)

층화 표본 75개 질의(337 query-document pair)로 품질 검증:
- strict gold precision: **97.2%** (172/177 관련 문서 확인)
- false positive rate: **0.0%**
- loose gold recall: **100%**

### 4.3 지표

- **Contamination@10** (Cont@10): top-10 중 오장비 문서 비율(shared 제외)
- **Gold Hit@10 (strict)**: top-10 내 strict gold 존재 질의 비율
- **Gold Hit@10 (loose)**: top-10 내 loose gold 존재 질의 비율
- **MRR**: 첫 gold hit의 평균 역순위

### 4.4 Elasticsearch 인프라

- 텍스트 인덱스: `chunk_v3_content` (한글 Nori, 영문 standard)
- 벡터 인덱스: `chunk_v3_embed_bge_m3_v1` (BGE-M3, 1024차원)
- 하이브리드 결합: `chunk_id` cross-index join

---

## 5. 결과

### 5.1 주요 결과: 장비 필터는 오염을 제거하고 재현율을 높인다

**표 1. 전체 결과 (n=578, k=10, 마스킹 질의)**

| System | Cont@10 | Gold Strict | Gold Loose | MRR |
|--------|:-------:|:-----------:|:----------:|:---:|
| B0 (BM25) | 0.473 | 49.7% | 59.3% | 0.355 |
| B1 (Dense) | 0.730 | 39.4% | 45.2% | 0.245 |
| B2 (Hybrid) | 0.585 | 60.7% | 65.7% | 0.413 |
| B3 (Hybrid+Rerank) | 0.584 | 60.7% | 65.7% | 0.335 |
| **B4 (Hard filter)** | **0.001** | **91.2%** | **92.0%** | **0.562** |
| B4.5 (Filter+Shared) | 0.001 | 70.2% | 76.0% | 0.421 |

gold hit의 모든 pairwise 차이는(B2 vs B3 제외) 통계적으로 유의했다(McNemar, p < 0.001; B2 vs B3, p = 0.48). strict gold hit 기준 bootstrap 95% CI:
- B3 → B4: Δ = +30.4%p [+26.6, +34.4], p < 10⁻³⁰ (discordant: 179 개선, 3 악화)
- B4 → B4.5: Δ = −20.9%p [−24.2, −17.6], p < 10⁻²⁵ (discordant: 0 개선, 121 악화)
- B0 → B2: Δ = +11.1%p [+8.0, +14.4], p < 10⁻⁹

**발견 1**: 장비 인식 하드 필터는 모든 검색 구조에서 오염을 47–73%에서 거의 0으로 낮춘다.

**발견 2**: 필터링은 재현율도 동시에 개선한다. strict gold hit가 49.7–60.7%에서 91.2%로 상승(+30.5~+41.5%p). 이는 오염 문서가 top-k를 점유해 정답 문서를 밀어내던 것을 제거했기 때문이다.

**발견 3**: shared 문서를 추가한 B4.5는 B4보다 성능이 낮다. strict gold hit가 91.2%에서 70.2%로 하락(McNemar p < 10⁻²⁵). 121개 discordant 케이스에서 B4는 성공하고 B4.5는 실패했으며, shared 추가로 이득 본 질의는 0개였다.

**표 2. scope 관측성별 결과 (마스킹 질의)**

| Scope | System | Cont@10 | Gold Strict |
|-------|--------|:-------:|:-----------:|
| explicit_device (n=429) | B3 | 0.481 | 79.5% |
| | **B4** | **0.001** | **97.0%** |
| explicit_equip (n=149) | B3 | 0.881 | 6.7% |
| | **B4** | **0.000** | **74.5%** |

**발견 4**: `explicit_equip` 질의는 오염(88.1%)이 가장 심하고, 필터링 이득(+67.8%p)이 가장 크다.

### 5.2 Soft Scoring은 효과적이지 않다

**표 3. Soft scoring vs. hard filter (마스킹 질의, n=578)**

| System | Cont@10 | Gold Strict | MRR |
|--------|:-------:|:-----------:|:---:|
| B3 (baseline) | 0.584 | 60.7% | 0.335 |
| P6 (λ=0.05) | 0.649 | 60.7% | 0.338 |
| P7 (adaptive λ) | 0.649 | 60.7% | 0.337 |
| **B4 (hard filter)** | **0.001** | **91.2%** | — |

$\lambda=0.05$의 soft scoring은 오염을 줄이지 못했고, 오히려 6.5%p 악화시켰으며 gold hit는 동일했다. 핵심 원인은 스케일 불일치다. 문서 간 검색 점수 차이는 보통 0.1–1.0인데 penalty 항 $\lambda \cdot v_{\text{scope}} = 0.05$는 순위를 바꾸기에 너무 작다. $\lambda$를 키우면 결국 하드 필터에 수렴해 실용적 이점이 사라진다.

### 5.3 평가 편향: 왜 기존 연구가 스코프 필터 가치를 놓쳤는가

**표 4. 원문 질의 vs. 마스킹 질의 비교 (n=578)**

| System | Query | Cont@10 | Gold Strict |
|--------|-------|:-------:|:-----------:|
| B0 | original | 0.422 | 68.2% |
| B0 | masked | 0.473 | 49.7% |
| B3 | original | 0.365 | 75.1% |
| B3 | masked | 0.584 | 60.7% |
| B4 | masked | 0.001 | 91.2% |

원문 질의에서는 B3가 필터 없이도 75.1% gold hit를 보여 스코프 필터 효과가 작아 보인다. 그러나 이는 초기 Phase 1–4 결과와 동일한 **평가 편향 산물**이다.

1. **어휘 누설(lexical leakage)**: 문서에서 생성된 질문이 장비명을 포함하고, BM25가 이를 장비명 포함 문서 ID와 직접 매칭
2. **부풀려진 baseline**: 이 인위적 매칭 때문에 필터가 없어도 정답 문서를 쉽게 찾음
3. **마스킹 디바이어싱**: `[DEVICE]` 치환으로 지름길을 끊으면 실제 오염률과 필터 가치가 드러남

**편향은 결론을 뒤집는다**: 원문 평가는 필터링이 재현율을 깎는 것처럼 보였고(Phase 1–4: −36~−69%), 마스킹 평가는 필터링이 재현율을 크게 올린다(+30~+42%p). 이 효과는 모든 검색 방식에서 유의했다(McNemar p < 10⁻¹³; 최대 하락은 B1: Δ = −26.1%p, B0: −18.5%p, B2/B3: −14.5%p).

### 5.4 검색 방식별 오염 취약성

**표 5. 검색 방식별 마스킹 민감도**

| System | Cont@10 (orig) | Cont@10 (masked) | Delta |
|--------|:-:|:-:|:-:|
| B0 (BM25) | 0.422 | 0.473 | +0.051 |
| B1 (Dense) | 0.373 | 0.730 | **+0.357** |
| B2 (Hybrid) | 0.365 | 0.585 | +0.220 |
| B3 (Hybrid+Rerank) | 0.365 | 0.584 | +0.219 |

Dense(B1)는 마스킹에 가장 취약하며 오염이 거의 2배(37.3%→73.0%) 증가한다. 의미 임베딩은 장비 맥락이 제거되면 장비 간 주제 유사 문서를 높은 관련도로 올리기 쉽다. BM25는 장비명 외 다양한 토큰 일치에 의존해 상대적으로 견고(+5.1%)하다.

또한 cross-encoder 재랭킹(B3 vs B2)은 오염을 줄이지 못했다. 재랭커는 문서 품질/관련도를 평가하지만 스코프 정합성은 직접 제어하지 못한다.

### 5.5 파서 정확도와 Oracle 격차

**표 6. 장비 파서 정확도**

| Scope | Exact Match | No Detection | Wrong Detection |
|-------|:-----------:|:------------:|:---------------:|
| explicit_device (n=429) | 88.6% | 0.2% | 11.2% |
| explicit_equip (n=149) | 0.0% | 100% | 0.0% |
| Overall (n=578) | 65.7% | 26.0% | 8.3% |

**표 7. Oracle vs 실제 파서 검색 (BM25, loose gold, adjusted Cont@10)**

| | Gold Hit@10 (loose) | Cont_adj@10 | MRR |
|--|:-:|:-:|:-:|
| Oracle B4 | 92.7% | 0.0% | 0.846 |
| Real B4 | 91.9% | 30.6% | 0.832 |
| Delta | −0.9%p | +30.6%p | −0.014 |

*참고: 이 비교는 파서 정확도 효과를 분리하기 위해 BM25 + loose gold로 평가했다. Adjusted contamination(Cont_adj)은 no-filter fallback 문서를 오염으로 계산한다.*

regex 기반 파서는 explicit_device 질의에서 88.6% 정확도로 oracle 대비 gold hit 손실이 −0.9%p에 불과했다. 그러나 explicit_equip 질의에서 완전 실패(0%)하면서 150개 미인식 질의가 비필터 fallback으로 흘러 contamination이 30.6%까지 증가했다.

### 5.6 암시형 질의 검증

기계적 마스킹을 넘어 결과의 일반성을 검증하기 위해 장비/설비명이 전혀 없는 자연 암시형 질의 628개로 평가했다.

**표 8. 암시형 질의 결과 (n=628) vs 명시형-마스킹 (n=578)**

| System | Implicit Strict | Explicit-Masked Strict | Implicit Cont@10 | Explicit-Masked Cont@10 |
|--------|:-:|:-:|:-:|:-:|
| B0 | 52.9% | 49.7% | 0.652 | 0.473 |
| B1 | 40.3% | 39.4% | 0.735 | 0.730 |
| B2 | 61.8% | 60.7% | 0.664 | 0.585 |
| B3 | 61.5% | 60.7% | 0.665 | 0.584 |
| B4 | **84.7%** | **91.2%** | 0.001 | 0.001 |
| B4.5 | 76.6% | 70.2% | 0.001 | 0.001 |

아래 3가지가 결과의 강건성을 확인해준다.

1. **마스킹 중립성**: 암시형 질의에서 마스킹/원문 조건 결과가 거의 동일(strict delta 0–4건, loose delta 0–7건)하며, 마스킹이 장비명 지름길만 제거하고 다른 인공효과를 만들지 않음을 시사한다.

2. **Baseline 수렴**: 암시형 baseline(B0: 52.9%, B2: 61.8%)이 명시형-마스킹(49.7%, 60.7%)과 통계적으로 유의한 차이가 없다(two-proportion z-test: B0 p = 0.27, B2 p = 0.71). 즉, 마스킹은 장비 비명시 검색 시나리오를 잘 근사한다.

3. **오염 증폭**: 장비 신호가 없으면 오염이 더 증가(B0: 0.652 vs 0.473)해 스코프 필터의 필요성이 더 커진다. 하드 필터(B4)는 여전히 가장 큰 개선을 보이며, 비필터 최선 대비 +31.8%p 향상한다.

암시형 B4(84.7%)와 명시형-마스킹 B4(91.2%) 간 격차는 암시형 질의의 본질적 난도를 보여준다. 같은 장비 내부에서도 어휘 단서가 적어 BM25 구분이 어려워진다. shared 역설(B4 > B4.5)은 유지되지만 격차는 명시형(20.9%p)보다 줄어(8.1%p), shared 문서 치환 효과 일부가 장비명 매칭 영향과 결합됨을 시사한다.

---

## 6. 논의

### 6.1 왜 하드 필터가 재현율을 높이는가

필터링이 재현율을 높인다는 반직관적 결과(B4가 비필터 최선 B3 대비 +30.5%p)는 구조적으로 설명된다. 다중 장비 코퍼스에서 오염 문서는 무작위 노이즈가 아니라 “주제는 비슷하지만 장비가 다른 문서”다. 이러한 근접 중복 문서가 top-k를 점유해 정답 장비 문서를 밀어낸다. 이를 제거하면 정답 문서가 상위로 올라온다.

이 효과는 `explicit_equip` 질의에서 가장 강하다(오염 88%, gold hit +67.8%p). 이런 질의는 “etcher PM procedure”처럼 장비 범주 단어만 포함해 동일 범주 여러 장비 문서와 쉽게 매칭되기 때문이다.

### 6.2 Shared 문서 역설

B4.5(장비 필터 + shared)는 B4(장비 필터 단독)보다 strict gold hit가 21%p 낮다. 분석 결과 메커니즘은 명확하다. B4 성공/B4.5 실패 121개 질의 중 shared 포함으로 이득을 본 질의는 **0개**였다. 해당 질의의 B4.5 top-10 중 57.3%가 shared 문서였고, 특히 5개 `device_net_board` SOP가 각 97–99개 질의에 반복 등장해 장비 특화 gold 문서를 밀어냈다.

이는 실무적으로 shared 임계치($\geq 3$ devices)가 지나치게 완화적일 수 있음을 뜻한다. controller, FFU, device net board 같은 광범위 topic은 유용한 교차 장비 자산보다 검색 노이즈로 작동할 수 있다. 향후에는 topic 특이성 가중 임계치나 질의 조건부 shared gating이 필요하다.

### 6.3 RAG 평가에 대한 시사점

문서 시드 평가 편향 발견은 일반적 함의를 가진다.

1. **문서 기반 생성 평가셋은 검색 지름길을 내장**해 baseline을 부풀릴 수 있다.
2. **표준 지표만으로는 지름길이 드러나지 않는다** — 오염률이 중간 수준처럼 보여도 실제로는 어휘 매칭이 지배할 수 있다.
3. **마스킹은 단순하면서도 실용적인 디바이어싱 프로토콜**로, 엔티티 메타데이터가 있는 평가셋에 광범위하게 적용 가능하다.

도메인 특화 RAG 평가는 엔티티 이름 매칭과 독립적인 검색 품질을 확인하기 위해 마스킹 변형을 포함하는 것을 권장한다.

### 6.4 실제 배포 고려사항

실서비스에서는 oracle 가정을 파서/라우터 컴포넌트로 대체해야 한다. 5.5절 분석 결과:
- 장비명이 명시된 질의: regex 파서가 oracle에 근접(−0.9%p)
- 장비 타입만 있는 질의: 별도 장치가 필요

#### 장비 타입 질의의 식별 전략

149개 `explicit_equip` 질의(파서 정확도 0%)는 배포 단계의 핵심 공백이다. 현실적인 3가지 전략은 다음과 같다.

1. **대화 문맥 기반 라우팅**: 다중 턴 세션에서는 이전 턴에서 목표 장비가 이미 정해진 경우가 많다. 세션 단위 장비 추적기를 두면 이후 implicit/equip 질의로 장비 문맥을 전달할 수 있다.

2. **장비군 기반 필터링**: 특정 장비가 아니라 장비 타입만 알려진 경우(예: “etcher”), 단일 장비 대신 장비군으로 제한한다. CVD 등 이종 장비 오염을 줄이면서 동종 장비 문서를 유지할 수 있다. 다만 장비군 내부 오염 증가는 감수해야 한다.

3. **LLM 기반 장비 추론**: 질의 문맥, 사용자 프로필(담당 장비), 대화 이력을 활용해 LLM이 목표 장비를 추론하도록 한다. 유연성은 높지만 지연과 추론 오류가 추가된다.

implicit 질의(장비/설비 언급 없음)에서는 1번이 필수다. 암시형 실험에서 oracle 장비를 준 B4가 84.7%를 보인다는 점은 장비만 올바르게 식별되면 검색 품질이 충분히 높다는 뜻이다. 배포의 병목은 검색 자체보다 장비 식별 단계에 있다.

### 6.5 한계

1. **단일 도메인**: 결과는 하나의 반도체 코퍼스 기반이며, 다른 산업 도메인으로의 일반화는 추가 검증이 필요하다.
2. **Oracle 필터 의존**: 메인 결과는 gold 장비 라벨 기반이며, 실제 파서는 equip-type 질의에서 30.6% contamination을 유발한다.
3. **BM25/Hybrid 중심**: SPLADE, ColBERT 같은 고급 검색기는 본 연구 범위에서 제외했다.
4. **마스킹은 유효하지만 완전 동일하지 않음**: 5.6절에서 baseline은 통계적으로 동등했지만 B4는 유의한 차이(암시형 84.7% vs 명시형-마스킹 91.2%, p < 0.001)를 보여, 암시형 질의가 본질적으로 더 어렵다는 점을 시사한다.

---

## 7. 결론

본 연구는 산업용 RAG에서 교차 장비 오염이 47–73% 수준으로 심각하며 기존에 과소평가되어 왔음을 보였다. 단순한 장비 인식 하드 필터는 오염을 사실상 제거하면서 재현율을 30–42%p 향상시킨다. 이는 문서 시드 벤치마크의 평가 편향 때문에 가려져 있던 결과다.

또한 결과는 기계적 마스킹 명시형 질의(n=578)와 자연 암시형 질의(n=628) 모두에서 일관되게 관찰되었다. 두 조건에서 baseline 성능은 수렴하고 하드 필터가 가장 큰 개선을 보였다. 이는 마스킹 프로토콜이 단순 토큰 치환 인공물(artifact)이 아니라 유효한 디바이어싱 접근임을 뒷받침한다.

아울러 soft scoring은 점수 스케일 불일치로 하드 필터를 대체하지 못했고, shared 문서 포함은 장비 특화 gold 문서를 밀어내어 성능 저하를 유발할 수 있음을 확인했다.

이 결과는 산업용 RAG에서 스코프 인식 검색을 1급 구성요소로 채택해야 함을 강하게 시사하며, 질의와 문서 모두에 도메인 엔티티가 포함된 환경에서는 평가 방법론 자체를 재검토해야 함을 보여준다.

---

## 참고문헌

- Buckley, C., et al. (2007). Bias and the limits of pooling for large collections. *Information Retrieval*.
- Carterette, B., et al. (2006). Minimal test collections for retrieval evaluation. *SIGIR*.
- Guu, K., et al. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML*.
- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Ma, X., et al. (2021). A Replication Study of Dense Passage Retriever. *arXiv*.
- Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv*.
- Peng, B., et al. (2024). Domain-Specific Retrieval-Augmented Generation: A Survey. *arXiv*.
- Barnett, S., et al. (2024). Seven Failure Points When Engineering a Retrieval Augmented Generation System. *arXiv*.
- LlamaIndex (2024). Building Production RAG over Complex Documents. *LlamaIndex Documentation*.

---

## Appendix A: 장비별 결과

### B4 masked Gold Hit (Strict) — Hybrid+Rerank

| Device | Queries | Hit Rate |
|--------|:-------:|:--------:|
| SUPRA N series | 20 | 100% |
| OMNIS plus | 12 | 100% |
| INTEGER plus | 91 | 99% |
| GENEVA XP | 69 | 99% |
| TIGMA Vplus | 50 | 98% |
| PRECIA | 68 | 97% |
| SUPRA N | 82 | 93% |
| ZEDIUS XP | 41 | 93% |
| SUPRA XP | 26 | 88% |
| SUPRA Vplus | 88 | 64% |

**SUPRA Vplus (64%)**: 88개 질의 중 79개가 explicit_equip 유형이다. 32개 실패가 모두 explicit_equip에서 발생했으며, 원인은 (a) 단일 CONTROLLER 문서에 21개 비유사 질의를 과할당한 gold 라벨 문제, (b) 해당 장비 SOP 커버리지 부족(4개)이다.

### B4 실패 분석

전체 578개 질의 중 B4 실패는 51개(8.8%)이며, 이 중 38개(75%)가 `explicit_equip`, 13개가 `explicit_device`다. 실패는 특정 장비에 집중된다.

| Device | Failures | Total | Failure Rate | Primary Cause |
|--------|:--------:|:-----:|:------------:|---------------|
| SUPRA Vplus | 32 | 88 | 36% | SOP 부족 + equip 중심 gold |
| SUPRA N | 6 | 82 | 7% | SUPRA family 내 topic 중첩 |
| SUPRA XP | 3 | 26 | 12% | shared SOP 기반 cross-reference gold |
| ZEDIUS XP | 3 | 41 | 7% | SUPRA XP 기원 SOP로 gold 할당 |
| Other | 7 | 341 | 2% | 개별 사례 |

패턴은 명확하다. B4의 잔여 오류는 주로 **검색 실패보다 equip-type 질의의 gold 라벨 품질**에서 비롯된다. gold가 목표 장비에 올바르게 할당된 경우 B4는 거의 완전한 검색 성능을 보인다.

## Appendix C: 교차 장비 오염 매트릭스

B3 마스킹 조건(비-shared 문서)에서 주요 오염 원천:

| Source Device | Contaminating Docs | Target Devices Affected |
|---------------|:------------------:|:-----------------------:|
| SUPRA N | 205 | 15 |
| INTEGER plus | 179 | 12 |
| PRECIA | 107 | 11 |
| GENEVA XP | 100 | 11 |
| ZEDIUS XP | 72 | 10 |
| SUPRA N series | 72 | 10 |

오염은 무작위가 아니다. 문서 수가 큰 장비(SUPRA N: 82개, INTEGER plus: 91개)는 topic 커버리지가 넓어 교차 장비 오염을 더 많이 유발한다. 최대 오염 쌍은 SUPRA N → INTEGER plus(57건)이며, 동일 제조 영역 장비군 간 topic 중첩을 반영한다.

## Appendix B: 평가셋 통계

| Statistic | Explicit (v0.6) | Implicit (v0.7) | Total |
|-----------|:---:|:---:|:---:|
| Queries | 578 | 628 | 1,206 |
| Devices | 27 | 27 | 27 |
| scope: explicit_device | 429 | 0 | 429 |
| scope: explicit_equip | 149 | 0 | 149 |
| scope: ambiguous | 0 | 628 | 628 |
| Documents in corpus | 508 | 508 | 508 |
| Shared documents | 60 | 60 | 60 |
| Strict gold precision (verified) | 97.2% | — | — |
| Top-k | 10 | 10 | 10 |
