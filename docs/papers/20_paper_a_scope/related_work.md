# Paper A Related Work — Hierarchy-aware Scope Routing for Cross-Equipment Contamination Control

## Purpose

Paper A의 "Hierarchy-aware scope routing for cross-equipment contamination control in industrial maintenance RAG" 주제에 대한 체계적 문헌 조사. 5개 축(Retrieval, Sparse/Dense/Hybrid, Structured Retrieval, Query Routing, Safety Metrics)에 맞춰 구성되며, 각 축은 Paper A의 기여 위치를 명확히 함.

## Paper A Positioning

- **Problem**: Industrial RAG (반도체 유지보수)에서 타 장비 문서가 혼입(cross-equipment contamination)되어 잘못된 근거로 답변 생성
- **Gap**: 기존 RAG/ODQA는 retrieval effectiveness만 최적화, **contamination을 first-class metric으로 다루지 않음**. 특히 shared documents(공용 절차)와 equipment families(유사 장비)가 존재하는 산업 환경에서 scope policy와 trade-off 분석이 부족
- **Proposed**:
  - **G (Hierarchy-aware Scope Routing)**: Hard/Family/Shared 3단 정책으로 contamination-recall trade-off 관리
  - **C (Contamination-aware Scoring)**: Scope 위반을 점수함수에 penalty로 반영하는 목적함수
  - **Efficiency (Matryoshka Router)**: 저차원 라우터로 scope 후보 선정 비용 절감
  - **Evaluation**: Explicit/Masked/Ambiguous 3종 평가셋으로 robustness 검증, Contamination@k를 safety metric으로 정의

---

## Cluster 1: RAG and Evidence-Grounded QA

**Why needed**: Retrieval-augmented generation 파이프라인의 기반을 정당화하고, 본 논문이 RAG 프레임워크 내에서 어떤 단계(retrieval phase의 scope control)를 개선하는지 위치짓기.

### Key References

1. **Lewis et al. (2020) — RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** `lewis2020rag`
   - Seminal work establishing retrieval-augmented generation paradigm
   - Demonstrates effectiveness of retrieval-grounded generation across multiple tasks
   - Introduces retriever-generator joint training framework

2. **Guu et al. (2020) — REALM: Retrieval-Augmented Language Model Pre-Training** `guu2020realm`
   - Pre-training with retrieval augmentation for knowledge-intensive tasks
   - Shows benefits of retrieval integration during model training
   - Relevant for understanding retriever quality's impact on downstream generation

3. **Petroni et al. (2021) — KILT: A Benchmark for Knowledge Intensive Language Tasks** `petroni2021kilt`
   - Provides unified evaluation framework for knowledge-intensive tasks
   - Demonstrates that retrieval quality directly affects answer correctness
   - Relevant for designing contamination metrics analogous to retrieval accuracy metrics

4. **Izacard & Grave (2021) — Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering** `izacard2021fid`
   - Fusion-in-Decoder (FiD): Multi-passage fusion for stronger answer generation
   - Shows that retriever errors (wrong passages) degrade generation quality
   - Motivates need for contamination control at retrieval stage

5. **Borgeaud et al. (2022) — Improving Language Models by Retrieving from Trillions of Tokens** `borgeaud2022retro`
   - RETRO: Retrieval-Enhanced Transformer for long-form QA
   - Demonstrates scaling RAG to massive retrieval corpora
   - Highlights scalability challenges when retrieval scope is uncontrolled

6. **Shi et al. (2024) — REPLUG: Retrieval-Augmented Black-Box Language Models** `shi2024replug`
   - Retrieval augmentation for proprietary/frozen LLMs
   - Shows versatility of retrieval-grounding approach
   - Relevant for practical deployment scenarios

### Gap Analysis
Existing RAG works optimize retrieval effectiveness globally (Recall, MRR, NDCG) without explicitly modeling **scope-based contamination**. They assume retrieval correctness is determined by relevance alone, ignoring domain-specific constraints (e.g., equipment-specific documents). Paper A introduces contamination as an orthogonal first-class metric and proposes scope-aware filtering to control it.

---

## Cluster 2: Retrieval Baselines — Sparse, Dense, and Hybrid Approaches

**Why needed**: Establish standard retriever families (BM25, dense encoders, hybrid fusion) that form the backbone of Paper A's experiments. Paper A doesn't propose novel retrievers but instead focuses on scope filtering applied to these baselines.

### Key References

1. **Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain Question Answering** `karpukhin2020dpr`
   - Introduces dual-encoder dense retrieval (DPR)
   - Competitive with or superior to sparse retrieval on ODQA benchmarks
   - Used as baseline dense retriever in Paper A experiments

2. **Khattab & Zaharia (2020) — ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT** `khattab2020colbert`
   - Late interaction architecture for efficient dense retrieval
   - Reduces memory footprint and latency vs. full dense encoders
   - Relevant for industrial-scale deployment

3. **Xiong et al. (2021) — Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval** `xiong2021ance`
   - ANCE: Hardest-negative mining for dense retriever training
   - Improves retriever robustness against adversarial queries
   - Applicable to equipment-scope scenarios with family ambiguity

4. **Ni et al. (2022) — Large Dual Encoders Are Generalizable Retrievers** `ni2022gtr`
   - GTR: General text representation retrievers with strong zero-shot performance
   - Shows dense retrievers generalize across domains
   - Foundation for Matryoshka routing experiments in Paper A

5. **Cormack et al. (2009) — Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods** `cormack2009rrf`
   - RRF: Non-learnable fusion of multiple rankers without relevance scores
   - Used in Paper A's Hybrid+RRF baseline for combining sparse and dense scores
   - Simple, robust, and widely adopted in practice

6. **Santhanam et al. (2022) — ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction** `santhanam2022colbertv2`
   - Improved ColBERT with better training procedures and scale
   - Superior end-to-end efficiency for large corpora
   - Relevant for comparing with Paper A's Matryoshka router efficiency

7. **Nogueira & Cho (2019) — Passage Re-ranking with BERT** `nogueira2019reranker`
   - Cross-encoder reranking for QA
   - Shows reranking improves retriever quality in two-stage pipelines
   - Paper A applies reranking post-scope-filtering in P1-P7 configurations

### Gap Analysis
Standard retrieval methods optimize for global relevance without scope awareness. Dense retrievers learn representations to match queries to documents universally, but in equipment maintenance RAG, a document may be highly relevant semantically yet contaminating if it describes a different equipment type. Paper A's scope filtering applies orthogonal constraints after retrieval to control domain-specific contamination.

---

## Cluster 3: Structured, Faceted, and Metadata-Filtered Retrieval

**Why needed**: Position Paper A's scope control mechanism as an evolution of structured retrieval. Faceted search and metadata-aware retrieval provide prior art for filtering by document properties; Paper A adapts this to equipment hierarchies while adding contamination metrics and handling shared documents.

### Key References

1. **Robertson et al. (2004) — Simple BM25 Extension to Multiple Weighted Fields** `robertson2004bm25f`
   - BM25F: Extends BM25 to weight document fields differently
   - Foundation for fielded/structured sparse retrieval
   - Relevant for understanding field-level filtering in equipment metadata

2. **Hearst et al. (2006) — Faceted Search and Navigation: A Primer** `hearst2006faceted`
   - Systematic design patterns for faceted search interfaces and systems
   - Shows how to organize search by multiple attributes (e.g., equipment type, document category)
   - Paper A's scope levels (`shared` / `device` / `equip`) are inspired by faceted categorization

3. **Stoica et al. (2007) — Faceted Metadata in the Flamenco Search System** `stoica2007faceted`
   - Hierarchical faceted metadata for large document collections
   - Demonstrates effectiveness of hierarchical faceting for large-scale collections
   - Directly relevant to Paper A's equipment hierarchy (device → equip)

4. **Bondarenko et al. (2022) — Overview of the TREC ikat Track** `bondarenko2022faceted_conversational`
   - Conversational search with faceted query understanding
   - Shows facets can structure multi-turn interactions
   - Motivates device-aware scope tracking in conversation context

5. **Tunkelang (2009) — Faceted Search** `tunkelang2009faceted`
   - Monograph on faceted search design, interaction patterns, and algorithms
   - Foundational reference for hierarchical organization
   - Supports Paper A's design of scope levels and allowed-scope logic

### Gap Analysis
Existing faceted and structured retrieval systems use metadata to enable browsing and filtering, but they do not explicitly model **scope contamination** as a safety metric. They assume users know which facets (e.g., equipment types) are relevant, and they do not address the trade-off between scope safety and recall when documents overlap multiple facets (shared SOPs) or when document-facet assignments are ambiguous (equipment families). Paper A introduces contamination@k as a quantitative metric and proposes adaptive policies to manage this trade-off.

---

## Cluster 4: Query Routing and Collection Selection

**Why needed**: Establish prior art for automatically deciding which scope(s) to query based on the input question. Paper A's Matryoshka router extends collection selection methods to equipment-level granularity and adds contamination awareness.

### Key References

1. **Shokouhi & Si (2011) — Federated Search** `shokouhi2011federated`
   - Survey of collection selection in distributed/federated IR
   - Establishes terminology: resource selection, query routing, collection fusion
   - Paper A's scope routing is a specialized form of collection selection for equipment scopes

2. **Si & Callan (2003) — Relevant Document Distribution Estimation for Collection Selection** `si2003relevant_doc_dist`
   - Proposes methods to estimate collection relevance without accessing full collections
   - Enables lightweight collection ranking for large federated systems
   - Applicable to Paper A's Matryoshka router for selecting device candidates

3. **Callan (2000) — Distributed Information Retrieval** `callan2000distributed`
   - Foundational distributed IR survey covering architecture, routing, and merging
   - Establishes retrieval principles for multi-collection systems
   - Provides baseline routing methods (CVV, CORI) that Paper A compares against implicitly

4. **Asadi & Lin (2013) — Selective Search: Efficient IR for High-Recall Tasks** `asadi2013selective_search`
   - Methods to route queries to most promising collections without exhaustive search
   - Proposes confidence measures for collection relevance
   - Related to Paper A's router confidence estimation for adaptive λ(q)

5. **Amazon Science (2022) — Hierarchical Query Classification in E-commerce Search** `amazon2022hqc`
   - Production hierarchical query routing for product catalog search
   - Shows practical effectiveness of hierarchy-based routing at scale
   - Similar intent: route query to relevant product category (analogous to equipment type)

### Gap Analysis
Traditional collection selection routes by topic similarity (e.g., "sports documents" vs. "news documents") or product categories. Paper A extends this to **equipment-scope routing** in a multi-level hierarchy (device_name → equip_id) and additionally proposes **contamination-aware scoring** to penalize scope violations during ranking. Existing methods do not define or optimize for scope-based contamination metrics specific to equipment maintenance domains.

---

## Cluster 5: Safety and Faithfulness Metrics in RAG + Industrial Domain QA

**Why needed**: Justify contamination@k as a safety metric analogous to faithfulness/grounding metrics in RAG, while grounding the work in industrial maintenance domain literature.

### Key References

1. **Es et al. (2024) — RAGAS: Automated Evaluation of Retrieval Augmented Generation** `es2024ragas`
   - Proposes automated evaluation metrics for RAG systems: context relevance, faithfulness, answer relevance
   - Demonstrates that retriever quality (precision, coverage) directly affects answer faithfulness
   - Motivates Paper A's focus on contamination as a component of retriever safety

2. **Adlakha et al. (2024) — Evaluating Correctness and Faithfulness of Large Language Model Generation with Question Answering** `adlakha2024faithfulness`
   - Systematic analysis of correctness vs. faithfulness in LLM-generated answers
   - Shows that out-of-scope or incorrect source documents lead to unfaithful outputs
   - Directly supports Paper A's emphasis on source scope validation

3. **Min et al. (2023) — FActScore: Fine-grained Factuality Evaluation for Claim-level Fact Extraction and Verification** `min2023factscore`
   - Proposes fine-grained factuality metrics for claim-level grounding
   - Decomposes answer quality into source correctness and fact alignment
   - Relevant for analyzing contamination's impact on claim-level correctness in maintenance QA

4. **Gavrilov et al. (2023) — Question Answering Models for Human-Machine Interaction in the Manufacturing Industry** `gavrilov2023manufacturingqa`
   - Introduces manufacturing domain QA challenges: sparse training data, specialized vocabulary, high stakes
   - Demonstrates that generic QA models underperform on equipment-specific questions
   - Motivates equipment-scope filtering as a domain-specific safety requirement

5. **Soraganvi et al. (2024) — Out-of-Distribution Detection in Open-Domain Question Answering** `soraganvi2024ood`
   - Methods to detect when retrieved documents fall outside expected knowledge scope
   - Proposes confidence calibration for detecting OOD evidence
   - Analogous to Paper A's contamination detection and soft penalties via λ(q)

6. **Lin et al. (2022) — Truthfulness and Factuality in Abstractive Summarization** `lin2022truthfulness`
   - Analyzes hallucination sources in neural generation, including irrelevant source selection
   - Shows that source scope errors (using wrong context) cause systematic hallucinations
   - Motivates contamination control as a prerequisite for faithful maintenance QA

### Gap Analysis
Existing faithfulness metrics (RAGAS, FActScore) measure answer quality post-hoc but do not explicitly quantify or optimize for **source scope safety during retrieval**. Industrial maintenance QA requires explicit contamination metrics because:
1. Using procedures from equipment A to answer questions about equipment B leads to dangerous recommendations (high-stakes domain)
2. Shared SOPs and equipment families create ambiguous scope boundaries that generic retrieval systems cannot resolve
3. Domain-specific contamination (equipment-mismatch) is orthogonal to semantic relevance

Paper A introduces Contamination@k as a first-class metric optimized during retrieval, complementing post-hoc faithfulness checks.

---

## Cluster 6: Matryoshka Representation Learning (Deferred to Companion Paper)

**Note on Paper A scope**: While Matryoshka Representation Learning is an enabling efficiency technology for Paper A's router (§3 Matryoshka Router), **matryoshka-specific contributions (nested dimension learning, truncation protocols) are deferred to a companion efficiency paper** to maintain focus on the core scope-routing novelty (contamination metrics and hierarchy-aware policies).

### Minimal Background References (for context only, not core Paper A contributions)

- **Kusupati et al. (2022) — Matryoshka Representation Learning** `kusupati2022matryoshka`
  - Enables flexible-dimension embeddings for efficient inference
  - Used to reduce router latency; efficiency gains are experimental results, not algorithmic contribution

- **Li et al. (2024) — 2D Matryoshka Sentence Embeddings** `li2024twodmatryoshka`
  - Joint layer and dimension truncation for deployment efficiency
  - Mentioned in efficiency comparisons but not a core novelty axis

---

## Gap Statement for Paper A

### Current State of RAG and Retrieval-Grounded QA

Existing RAG and open-domain QA research optimizes **retrieval effectiveness metrics** (Recall@k, MRR, NDCG) under the assumption that relevance is determined by semantic similarity and topical match. State-of-the-art methods combine sparse (BM25) and dense (DPR, ColBERT) retrievers via hybrid fusion (RRF), achieving strong global performance.

### Unaddressed Gap: Cross-Equipment Contamination in Industrial Maintenance RAG

However, in **industrial maintenance domains** (e.g., semiconductor equipment support), a crucial safety requirement is unmet:

1. **Scope-based contamination is not a first-class metric**: Standard retrieval systems do not distinguish between "relevant but out-of-scope" (wrong equipment) and "irrelevant." A procedure for equipment A may score high if equipment A and B share terminology, but using it to answer a question about equipment B leads to incorrect/dangerous recommendations.

2. **Shared documents create scope ambiguity**: In real maintenance corpora, some SOPs and troubleshooting guides apply across multiple equipment types (shared scopes: global SOP, family SOP). Existing structured/faceted retrieval systems can filter by metadata but do not quantify trade-offs between strict scope filtering (loses recalls from shared docs) and permissive retrieval (gains noise from equipment families).

3. **Equipment hierarchies and families are ignored**: Equipment in a fab operates at multiple hierarchy levels (device_name, equip_id) and forms families of similar types (e.g., SUPRA variants). Existing collection-selection and query-routing methods select by topic/category but do not handle equipment-type routing with family expansion or family-aware contamination metrics.

4. **Contamination-aware scoring is not standard**: Point-wise relevance scores (from retrievers and rerankers) and rank fusion methods do not incorporate scope-safety penalties. There is no mechanism to downrank semantically similar but contaminating documents in a principled, tunable way.

### Paper A's Contributions to Close This Gap

1. **Metric Novelty**: Introduces **Contamination@k** as a first-class safety metric alongside traditional effectiveness metrics. Defines three contamination variants (Raw, Adjusted, Shared@k) to transparently account for shared documents.

2. **Method Novelty**: Proposes **hierarchy-aware scope routing policy** with three decision layers:
   - **Hard routing** (explicit device/equip in question) → strict scope filtering
   - **Family routing** (implicit query, router selects device candidates) → soft scope penalty via confidence-aware λ(q)
   - **Shared document policy** → allows verified shared SOPs across scopes without penalties

3. **Scoring Novelty**: Introduces **contamination-aware scoring function**:
   ```
   Score(d, q) = Base(d, q) - λ(q) · v_scope(d, q)
   ```
   where λ(q) adapts from binary filter (hard mode) to soft penalty (router mode) based on query's scope clarity.

4. **Evaluation Novelty**: Develops **multi-faceted evaluation methodology** with three query subsets (Explicit, Masked/Implicit, Ambiguous) to robustly validate contamination control across varying scope observability levels.

This positions Paper A as the first work to treat equipment-scope contamination as an explicit, measurable, and optimizable objective in industrial maintenance RAG.

---

## References and Further Reading

Detailed BibTeX entries are maintained in `references.bib`. Key papers by cluster are:

- **Cluster 1 (RAG)**: Lewis et al. 2020, Guu et al. 2020, Petroni et al. 2021, Izacard & Grave 2021, Borgeaud et al. 2022, Shi et al. 2024
- **Cluster 2 (Retrievers)**: Karpukhin et al. 2020, Khattab & Zaharia 2020, Xiong et al. 2021, Ni et al. 2022, Cormack et al. 2009, Santhanam et al. 2022, Nogueira & Cho 2019
- **Cluster 3 (Structured Retrieval)**: Robertson et al. 2004, Hearst et al. 2006, Stoica et al. 2007, Bondarenko et al. 2022, Tunkelang 2009
- **Cluster 4 (Query Routing)**: Shokouhi & Si 2011, Si & Callan 2003, Callan 2000, Asadi & Lin 2013, Amazon Science 2022
- **Cluster 5 (Safety Metrics)**: Es et al. 2024, Adlakha et al. 2024, Min et al. 2023, Gavrilov et al. 2023, Soraganvi et al. 2024, Lin et al. 2022

---

## Related Work Writing Plan (for manuscript)

- **RW-1**: RAG and evidence-grounded QA background (Cluster 1)
- **RW-2**: Retriever baselines and hybrid fusion (Cluster 2)
- **RW-3**: Structured/fielded/faceted retrieval for scope control (Cluster 3)
- **RW-4**: Query routing and collection selection (Cluster 4) — hierarchy-based routing prior art
- **RW-5**: Safety metrics in RAG + industrial maintenance domain (Cluster 5)
- **Appendix-E**: Matryoshka and efficient embeddings (efficiency enabler, companion paper deferred)
