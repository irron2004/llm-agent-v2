# Paper A Related Work — Scope Safety / Contamination Control

## Purpose

Paper A의 "Hierarchy-aware scope routing for cross-equipment contamination control" 주제에 대한
문헌 조사 scaffold. 기여 3축(G: 라우팅 정책, Family/Shared 정책, Matryoshka 효율화)에 맞춰 구성.

## Paper A Positioning

- **Problem**: RAG에서 타 장비 문서가 혼입(cross-equipment contamination)되어 잘못된 근거로 답변 생성
- **Gap**: 기존 RAG/ODQA는 retrieval effectiveness만 최적화, contamination을 first-class metric으로 다루지 않음
- **Proposed**:
  - G: Hierarchy-aware scope routing (Hard/Family/Shared 3단 정책)
  - Matryoshka 저차원 라우터로 scope 후보 선정 효율화
  - Contamination@k를 safety metric으로 정의 및 평가

---

## Cluster 1: RAG / Knowledge-Intensive QA Background

- **Why needed**: retrieval-grounded generation 파이프라인의 기반 정당화
- Core refs: `lewis2020rag`, `guu2020realm`, `petroni2021kilt`

## Cluster 2: Retriever Baselines (Sparse / Dense / Hybrid)

- **Why needed**: baseline families(BM25, DPR, Hybrid+RRF) 정의
- Core refs: `karpukhin2020dpr`, `khattab2020colbert`, `xiong2021ance`, `ni2022gtr`, `cormack2009rrf`

## Cluster 3: Structured / Fielded / Faceted / Hierarchical Retrieval

- **Why needed**: equipment hierarchy를 metadata/facet으로 활용하는 기존 접근과의 관계 정립
- Paper A의 **novelty anchor** — 기존 faceted search는 contamination 제어를 목표로 하지 않음
- Core refs: `robertson2004bm25f`, `hearst2006faceted`, `stoica2007faceted`, `ontology2007review`

## Cluster 4: Query Routing / Collection Selection (G 관련)

- **Why needed**: "어떤 장비/스코프를 선택할지"를 라우터가 결정하는 기존 접근
- 주요 방향:
  - Collection selection in distributed IR (전통)
  - RAG에서의 query routing (semantic router, intent classification)
  - Hierarchy-based query classification
- Core refs: `amazon2022hqc`, `callan2000distributed` (추가 필요)
- **Paper A와의 차별점**: 기존은 주로 topic routing, 우리는 equipment scope routing + contamination metric

## Cluster 5: Matryoshka / Efficient Embeddings (효율 축)

- **Why needed**: 저차원 라우터의 이론적 기반 + 실용적 정당화
- 주요 방향:
  - Matryoshka Representation Learning: 중첩 표현으로 차원별 탄력적 사용
  - 2D-Matryoshka: 레이어 + 차원 동시 축소
  - Truncation 후 재정규화 프로토콜
- Core refs: `kusupati2022matryoshka`, `li2024twodmatryoshka`
- **Paper A에서의 역할**: scope routing의 비용을 줄이는 도구 (주기여가 아님)

## Cluster 6: Industrial Maintenance QA / Domain RAG

- **Why needed**: 반도체 Fab 유지보수 도메인의 applied significance 정당화
- Core refs: `gavrilov2023manufacturingqa`
- 추가 후보: semiconductor equipment maintenance, MES/CMMS 관련 QA 논문

## Cluster 7: Safety Metrics in IR/RAG (contamination 정의 관련)

- **Why needed**: contamination을 "safety metric"으로 정의하는 근거
- 주요 방향:
  - Precision at risk / false positive in high-stakes retrieval
  - Faithfulness/grounding metrics in RAG (RAGAS 등)
  - Out-of-scope detection in QA
- Core refs: 추가 조사 필요
- **Paper A와의 연결**: Contamination@k는 "wrong scope"에 대한 precision-like safety metric

---

## Gap Statement for Paper A

기존 RAG/ODQA 연구는 retrieval effectiveness(Recall, MRR, NDCG)에 집중하며,
cross-equipment contamination(타 장비 문서 혼입)을 별도 메트릭으로 다루지 않는다.

Structured/faceted retrieval은 metadata 기반 필터링을 지원하지만,
"공용 문서(shared SOP)" 및 "유사 장비(equipment family)"가 존재하는 산업 환경에서의
scope 정책 설계와 contamination-recall trade-off 분석은 다루지 않는다.

**Paper A의 gap**:
1. **Metric novelty**: Contamination@k를 first-class safety metric으로 정의
2. **Method novelty**: Hard/Family/Shared 3단 scope routing policy
3. **Efficiency novelty**: Matryoshka 저차원 라우터로 scope 후보 선정 비용 절감
4. **Evaluation novelty**: Explicit/Masked/Ambiguous 3종 평가셋으로 robustness 검증

---

## Related Work Writing Plan (for manuscript)

- **RW-1**: RAG and evidence-grounded QA background (Cluster 1)
- **RW-2**: Retriever baselines and hybrid fusion (Cluster 2)
- **RW-3**: Structured/fielded/faceted retrieval for scope control (Cluster 3)
- **RW-4**: Query routing and collection selection (Cluster 4) — G와의 차별점 명시
- **RW-5**: Efficient embeddings and Matryoshka (Cluster 5) — 효율 축 근거
- **RW-6**: Industrial maintenance QA context and remaining gap (Cluster 6 + 7)

---

## Immediate To-Read Priority (first pass)

1. `kusupati2022matryoshka` — Matryoshka 원논문
2. `lewis2020rag` — RAG 기반
3. `cormack2009rrf` — RRF fusion (현재 시스템 핵심)
4. `robertson2004bm25f` — structured retrieval 기반
5. `amazon2022hqc` — hierarchy-based query classification
6. Collection selection in distributed IR (survey 논문 탐색 필요)

Use citation details from `references.bib`.
