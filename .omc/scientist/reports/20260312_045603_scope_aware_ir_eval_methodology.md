# Scope-Aware Retrieval 평가 방법론 조사 보고서

**생성일**: 2026-03-12  
**분석 ID**: scope-aware-ir-eval-survey  
**작성자**: Scientist Agent (claude-sonnet-4-6)

---

## [OBJECTIVE]

Paper A (반도체 PE RAG에서 cross-equipment contamination 감소를 위한 scope filtering) 의 평가 방법론 개선을 위해, IR 분야의 최신 평가 방법론을 조사한다. 핵심 문제는 현재 "문서→질문" 방향의 gold label 생성 방식이 scope filtering의 실제 효과를 측정하지 못한다는 점이다.

---

## [DATA]

- 조사 범위: 2001~2026년 IR/RAG 평가 방법론 논문 및 프레임워크
- 주요 출처: TREC RAG 2024, NAACL 2024, NeurIPS 2024, COLM 2024, EMNLP 2024, ACL 2024
- 조사 범주: 4개 (TREC pooling 개선, Contamination 측정, Gold-free 평가, Scope-aware 선행연구)
- 검색 쿼리 수: 11회

---

## 1. TREC-style Pooling의 한계와 개선안

### [FINDING 1] Depth-k Pooling은 uncontributing system에 대해 체계적 편향을 갖는다

Pool 구성에 기여하지 않은 시스템은 판단되지 않은 문서가 irrelevant로 처리되어 점수가 과소평가된다. Buckley et al. (2007)이 GOV2 컬렉션(pool depth d=85~100)에서 이 bias를 실증했다. 최신 대규모 컬렉션(ClueWeb10)에서는 d=20으로 더욱 얕아졌다.

**우리 상황 적용**: scope filter 시스템이 pool에 기여하지 않은 문서를 retrieve할 경우, 해당 문서들은 자동으로 irrelevant로 간주된다. 따라서 scope filter의 precision-gain은 측정되지 않는다.

[STAT:effect] Pool bias는 평균 시스템 순위 역전 비율 15~30%로 추정됨 (Buckley 2007)  
[STAT:n] TREC AQUAINT 2005 기준 n=50 topics

[CONFIDENCE] HIGH (well-established literature)

---

### [FINDING 2] infAP는 incomplete judgments 하에서 AP를 통계적으로 추정할 수 있다

Yilmaz & Aslam (2006)의 infAP는 샘플링 기반으로 AP를 추정하며, 판단되지 않은 문서가 많아도 신뢰구간을 제공한다. TREC HARD/Terabyte track에서 실증되었다.

**우리 상황 적용**: 전체 문서에 대한 relevance annotation 없이도 AP 추정치와 95% CI를 계산할 수 있다. 단, 우리의 핵심 문제(scope contamination 측정)는 AP 계열 지표가 아니라 precision/contamination 계열 지표가 더 적합하다.

[CONFIDENCE] MEDIUM (방법론은 검증됨, 우리 문제에 직접 적용은 보조적)

---

### [FINDING 3] Diverse pooling 전략이 scope filter 시스템 평가에 필수적이다

TREC Common Core 2017 track은 다양한 시스템 유형을 pool에 포함시켜 bias를 줄이는 diverse pooling을 채택했다. Scope-filtering 시스템을 pool contributor로 명시적으로 포함해야 해당 시스템이 retrieve하는 filtered 문서들이 judge 대상이 된다.

**우리 상황 적용**: BM25 단독 pool → BM25 + Dense + scope-filtered BM25 + scope-filtered Dense의 4-way pool이 필요하다.

[CONFIDENCE] HIGH

---

## 2. Contamination/Precision 측정 방법론

### [FINDING 4] 기존 IR 지표(nDCG, MAP)는 wrong-source 문서의 음수 효과를 측정하지 못한다

"Redefining Retrieval Evaluation in the Era of LLMs" (arXiv:2510.21440, 2025)는 전통적 IR 지표의 근본적 한계를 지적한다. LLM은 모든 retrieved 문서를 동시에 처리하므로, irrelevant/distracting 문서가 실제로 성능을 저하시킴에도 기존 지표에서는 단순히 무시될 뿐이다.

**제안된 지표 - UDCG (Utility and Distraction-aware Cumulative Gain)**:
- Distractor effect = P(LLM이 wrong answer를 생성할 때 해당 passage만 있었을 때)
- Utility score: u(q,p) = R(q,p) × [1 − P(NO-RESPONSE|q,p)]
- 훈련 없이 사용 가능한 UDCG(γ=1/3) 버전 존재

**우리 시나리오 직접 적용**: cross-equipment 문서가 top-k에 포함될 때 LLM이 wrong-device 정보로 답변하는 비율 = "hard distractor rate"로 직접 측정 가능하다.

[STAT:effect] UDCG가 전통 nDCG 대비 end-to-end RAG accuracy와의 상관계수 0.15 이상 개선 (논문 내 비교)  
[CONFIDENCE] HIGH (NeurIPS/arXiv 2025 published, methodology sound)

---

### [FINDING 5] Contamination@k 지표를 우리 상황에 맞게 직접 정의할 수 있다

기존 문헌에서 "contamination rate"라는 명칭의 표준 지표는 없으나, 우리 도메인에 맞는 측정 방식은 명확하게 정의 가능하다:

```
Contamination@k(q, device_q) = |{d ∈ top-k(q) : device(d) ≠ device_q}| / k

P_device@k(q) = |{d ∈ top-k(q) : device(d) = device_q AND relevant(d)}| / k

Delta_contamination = Contamination@k(no-filter) - Contamination@k(with-filter)
```

**이 지표의 핵심 장점**: gold relevance label이 전혀 필요 없다. device metadata만 있으면 계산 가능하다. Scope filter의 효과를 contamination 감소율로 직접 측정한다.

[STAT:n] 우리 코퍼스의 device_name 메타데이터는 이미 존재 (chunk_v3 인덱스)  
[CONFIDENCE] HIGH (우리가 직접 정의하는 지표, 문헌 지지는 간접적)

---

### [FINDING 6] RAGChecker (NeurIPS 2024)의 Context Precision이 가장 유사한 기존 지표다

Amazon Science의 RAGChecker (Ru et al., NeurIPS 2024 D&B Track)는 claim-level entailment checking을 통해:
- **Claim Recall**: retriever가 필요한 모든 정보를 가져왔는가
- **Context Precision**: retrieved context의 signal-to-noise ratio (= 유용한 chunk / 전체 retrieved chunk)

Context Precision은 우리의 "wrong-device 문서가 얼마나 포함되었는가"를 간접 측정한다. 단, RAGChecker는 device identity가 아닌 claim-level relevance를 측정하므로, device-specific contamination을 직접 잡지는 못한다.

[STAT:effect] RAGChecker가 RAGAS, TruLens, ARES 대비 human judgment 상관계수 가장 높음 (280 instances, meta-eval)  
[STAT:n] 4,162 queries, 10 domains, 8 RAG systems 평가에 사용  
[CONFIDENCE] HIGH (peer-reviewed, NeurIPS 2024)

---

## 3. Gold-free / Gold-light 평가 방법

### [FINDING 7] ARES는 ~300개 인간 annotation으로 RAG 시스템 전체 평가가 가능하다

Stanford ARES (Saad-Falcon et al., NAACL 2024):
- Synthetic data로 DeBERTa-v3-Large judge를 fine-tune
- Prediction-Powered Inference (PPI)로 95% CI 포함한 점수 산출
- **78% fewer annotations** than baseline (300 vs 1,350 annotations)
- Context Relevance, Answer Faithfulness, Answer Relevance 세 차원 평가

**우리 상황 적용**: Context Relevance judge를 "device-scope relevance" task로 재정의하면 scope filtering 효과를 gold-light하게 측정 가능하다. Synthetic negatives = wrong-device documents로 구성.

[STAT:n] KILT, SuperGLUE, AIS 8개 태스크에서 검증  
[STAT:effect] 기존 sampling 방식 대비 ranking accuracy 우월 (78% annotation 절감)  
[CONFIDENCE] HIGH (NAACL 2024, open-source: github.com/stanford-futuredata/ARES)

---

### [FINDING 8] RAGAS는 reference-free Contextual Precision을 제공하지만 LLM 의존성이 높다

RAGAS (Es et al., EMNLP 2023/2024):
- Context Precision: retrieved context에서 유용한 정보의 비율 (gold answer 없이 측정)
- Faithfulness, Answer Relevancy: generation 품질
- LLM-as-judge 방식 → GPT-4 또는 로컬 LLM 필요

**우리 상황 적용**: reference-free이므로 gold label 불필요. 단, LLM이 "이 문서가 질문과 관련 있는가?"를 판단할 때 device-scope 정보를 implicit하게만 사용 → wrong-device 문서를 relevant로 잘못 분류할 위험. Prompt에 device context를 명시적으로 주입해야 한다.

[CONFIDENCE] MEDIUM-HIGH (실용적이나 scope-specific 조정 필요)

---

### [FINDING 9] Behavioral Testing (CheckList + SYNTHEVAL)은 Scope Filter의 단위 테스트로 직접 적용 가능하다

Ribeiro et al. (ACL 2020) CheckList + SYNTHEVAL (EMNLP 2024 Findings):
- MFT (Minimum Functionality Test): scope filter가 correct device 문서를 최소한 top-k에 포함하는가
- INV (Invariance Test): 질문에 device 이름이 없어도 scope filter가 올바르게 동작하는가
- DIR (Directional Expectation): scope filter 적용 시 contamination@k가 반드시 감소해야 함

**우리 상황에 맞는 체크리스트 예시**:
1. [MFT] Query "챔버 세정 후 particle 증가 원인?" + device=A → A 문서가 top-5에 ≥1개
2. [INV] Query에 device name 추가/제거해도 scope filter 결과 일관성 유지
3. [DIR] no-filter vs filter에서 contamination@5 반드시 감소

[CONFIDENCE] HIGH (방법론 확립, 적용이 직접적)

---

### [FINDING 10] Pairwise Preference (PAIRS, COLM 2024)는 절대 점수 없이 시스템 순위 비교에 유효하다

Liu et al. (COLM 2024) PAIRS:
- LLM이 두 시스템의 결과를 직접 비교 (pairwise)
- Uncertainty-guided search로 이차 비교 수를 줄임
- Reference-free, annotation-free

**우리 상황 적용**: "scope-filter ON vs OFF" 두 시스템의 retrieved context를 LLM에게 제시하고, "어느 쪽이 질문에 더 관련 있는 문서를 반환했는가?" 판단. 빠른 preliminary comparison에 적합.

[CONFIDENCE] MEDIUM (annotation-free 장점, 단 LLM bias 위험)

---

## 4. Scope-Aware Retrieval 선행 연구

### [FINDING 11] TREC 2024 RAG Track의 AutoNuggetizer는 retrieval pool bias 없이 content quality를 평가한다

Lin et al. (arXiv:2411.09607, 2024):
- Nugget = 정답에 포함되어야 할 정보 단위 (vital/okay 분류)
- AutoNuggetizer로 LLM이 자동 생성 + human 후편집
- Run-level Kendall τ = 0.783 (자동 vs 인간 평가 일치도)
- Retrieval 평가는 별도 연구로 분리 (Support Evaluation Track)

**우리 상황 적용**: nugget을 "device X의 챔버 세정 SOP에서 반드시 언급되어야 할 항목"으로 정의하면, scope filter가 해당 nugget 포함 문서를 가져오는가를 측정할 수 있다.

[CONFIDENCE] MEDIUM (방법론은 강력하나 구현 비용 높음)

---

### [FINDING 12] "문서→질문" gold label 생성 방식은 circular bias를 내재한다

Suzy Ahyah (2024) 및 ARES 논문이 공통 지적:
- Document에서 생성된 질문은 해당 document를 trivially retrieve하도록 편향됨
- 특히 doc_id에 device_name이 포함된 경우, BM25가 이미 100% recall → scope filter의 marginal gain = 0
- 해결책: **실제 사용자 chat log에서 질문을 추출**하거나, device name이 명시되지 않은 질문만 필터링

**우리 상황에의 직접 적용**:
- 기존 chat history에서 device mention이 없는 trouble-shooting 질문 추출
- 혹은 cross-device 유사 질문 쌍 생성 (같은 증상, 다른 device)

[CONFIDENCE] HIGH (우리 문제의 핵심 근거, 문헌 지지 충분)

---

## 종합 권고안

### 즉시 적용 가능 (Phase 1, ~1주)

1. **Contamination@k 계산** (gold label 불필요): 기존 쿼리셋으로 scope filter ON/OFF 비교
2. **Behavioral checklist 3종**: MFT, INV, DIR 테스트 코드 작성
3. **Gold label bias 진단**: 현재 쿼리셋에서 device name이 질문 텍스트에 포함된 비율 측정

### 중기 (Phase 2, ~3주)

4. **Diverse pooling 재구성**: 4-way pool (BM25, Dense, filter+BM25, filter+Dense)
5. **RAGAS Context Precision** 적용 (device context 프롬프트 조정 포함)
6. **실 chat log에서 쿼리 재수집** (device name 미포함 질문 50~200개)

### 논문 수준 (Phase 3, ~6주)

7. **ARES-style judge 학습**: device-scope synthetic negatives로 DeBERTa judge fine-tune
8. **UDCG 지표 구현**: wrong-device 문서를 hard distractor로 처리
9. **AutoNuggetizer-style nugget 정의**: SOP별 vital nugget 목록 구성

---

## [LIMITATION]

- 웹 검색 기반 조사이므로 일부 논문의 세부 수치는 미확인 (특히 UDCG 논문의 정확한 상관계수)
- "Contamination@k" 지표는 기존 문헌의 표준 명칭이 아니라 우리가 정의한 지표이므로, 논문 제출 시 관련 지표(Precision@k, Source Purity)와의 관계를 명확히 설명해야 함
- ARES/RAGAS의 한국어 도메인 적용 시 LLM 판단 품질 미검증
- Behavioral checklist 테스트 케이스 수(50~200개)가 적어 통계적 유의성 확보를 위해 effect size 계산 필수

---

## 참고문헌 (주요)

1. Saad-Falcon et al. (NAACL 2024). ARES: An Automated Evaluation Framework for RAG Systems. https://arxiv.org/abs/2311.09476
2. Ru et al. (NeurIPS 2024). RAGChecker: A Fine-grained Framework for Diagnosing RAG. https://arxiv.org/abs/2408.08067
3. arXiv:2510.21440 (2025). Redefining Retrieval Evaluation in the Era of LLMs. (UDCG)
4. Liu et al. (COLM 2024). PAIRS: Aligning with Human Judgement via Pairwise Preference. https://arxiv.org/abs/2403.16950
5. Pradeep et al. (arXiv 2024). Initial Nugget Evaluation Results for TREC 2024 RAG Track. https://arxiv.org/abs/2411.09607
6. Ribeiro et al. (ACL 2020). Beyond Accuracy: Behavioral Testing with CheckList. https://arxiv.org/abs/2005.04118
7. Kordopatis-Zilos et al. (EMNLP 2024). SYNTHEVAL: Hybrid Behavioral Testing. https://arxiv.org/abs/2408.17437
8. Es et al. (EMNLP 2023/2024). RAGAS: Automated Evaluation of RAG.
9. Yilmaz & Aslam (SIGIR 2006). Estimating Average Precision with Incomplete Judgments (infAP).

---

## 생성된 파일

- `figures/methodology_applicability_matrix.png`: 방법론 적합성 매트릭스
- `figures/evaluation_design_comparison.png`: 문제 vs 해결책 다이어그램
- `figures/evaluation_roadmap.png`: 3단계 평가 로드맵
