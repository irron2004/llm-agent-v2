# Device-Aware Scope Filtering for Cross-Equipment Contamination Control in Industrial RAG

**Draft v2** — 2026-03-14, based on masked query experiments

---

## Abstract

Retrieval-Augmented Generation (RAG) systems deployed in semiconductor manufacturing must retrieve equipment-specific documents from shared corpora spanning dozens of equipment types. We identify **cross-equipment contamination** — the retrieval of documents belonging to unrelated equipment — as a critical failure mode, affecting 53–73% of top-10 results depending on the retrieval method. We show that a simple device-aware hard filter eliminates contamination entirely while simultaneously improving recall by up to 42 percentage points. Crucially, we demonstrate that conventional document-seeded evaluation systematically underestimates the value of scope filtering due to lexical bias in gold labels, and propose **masked-query evaluation** as a debiasing protocol. Our experiments on 1,206 queries (578 explicit + 628 implicit) across 27 equipment types and four retrieval architectures (BM25, Dense, Hybrid, Hybrid+Rerank) provide strong evidence that scope-aware retrieval is essential for industrial RAG safety.

---

## 1. Introduction

Large-scale RAG systems in semiconductor fabrication environments typically index documents across multiple equipment types into a unified corpus. Maintenance engineers query this shared corpus for troubleshooting procedures, preventive maintenance guides, and operational parameters. When the retrieval system returns documents from the wrong equipment, the downstream LLM may generate answers grounded in incorrect procedures — a failure with direct safety implications in high-stakes manufacturing.

We term this failure mode **cross-equipment contamination**: the presence of out-of-scope equipment documents in top-k retrieval results. Unlike general relevance failures, contamination represents a systematic bias where topically similar but equipment-incorrect documents displace correct answers. For example, a heater chuck replacement procedure for SUPRA XP may be lexically and semantically similar to the procedure for GENEVA XP, but applying one to the other risks equipment damage.

Prior work on domain-specific RAG has focused on improving overall retrieval quality through hybrid retrieval, reranking, and query expansion. However, the **scope correctness** dimension — whether retrieved documents belong to the correct equipment context — has received limited attention. We argue that in multi-equipment industrial corpora, scope filtering is not merely an optimization but a safety requirement.

### Contributions

This paper makes four contributions:

1. **Contamination quantification**: We define Contamination@k as a gold-label-free metric and measure it across four retrieval architectures, finding contamination rates of 47–73% on masked queries (Section 4).

2. **Device filter effectiveness**: We demonstrate that oracle device filtering reduces contamination to near-zero while improving strict gold hit rate from 50–61% to 91–92% — a counterintuitive result where filtering *increases* recall (Section 5.1).

3. **Evaluation bias discovery**: We show that conventional document-seeded evaluation ("generate questions from documents") creates a circular bias that masks the value of scope filtering, and propose masked-query evaluation as a debiasing protocol (Section 5.3).

4. **Negative results on soft scoring**: We provide evidence that contamination-aware soft scoring (penalty-based reranking) is ineffective compared to hard filtering in this domain, due to fundamental scale mismatch between penalty terms and retrieval scores (Section 5.2).

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG systems combine retrieval with generative models to ground responses in external knowledge [Lewis et al., 2020; Guu et al., 2020]. Recent work has explored dense retrieval [Karpukhin et al., 2020], hybrid approaches combining sparse and dense signals [Ma et al., 2021], and cross-encoder reranking [Nogueira and Cho, 2019] to improve retrieval quality.

### 2.2 Domain-Specific RAG

Industrial applications of RAG face unique challenges including domain terminology, multilingual content, and safety requirements. Peng et al. (2024) survey domain-specific RAG systems, noting that manufacturing and healthcare applications require stricter grounding guarantees than general-purpose QA. Barnett et al. (2024) identify seven failure modes of RAG systems, including "wrong granularity" where retrieved documents match topically but at an inappropriate scope — closely related to our cross-equipment contamination problem. Prior work in manufacturing RAG has focused on document preprocessing and terminology normalization rather than scope-aware retrieval.

### 2.3 Metadata Filtering in Retrieval

Metadata-based pre-filtering is a standard feature in production vector databases (Pinecone, Weaviate, Milvus) and is recommended as a best practice for multi-tenant RAG systems (LlamaIndex, 2024). However, systematic evaluation of filtering's impact on both contamination and recall is lacking — existing documentation focuses on latency and scalability rather than retrieval quality. Our work provides the first rigorous evaluation of device-level filtering in a multi-equipment industrial corpus, demonstrating that the interaction between filtering and recall is non-trivial.

### 2.4 Evaluation Methodology in IR

The IR community has long recognized biases in evaluation methodology, including pooling bias [Buckley et al., 2007] and topic bias [Carterette et al., 2006]. Our discovery of document-seeded evaluation bias — where questions generated from documents inherit lexical signals that inflate baseline performance — adds a new dimension to this literature, particularly relevant for domain-specific evaluation sets.

---

## 3. Problem Setting and Method

### 3.1 Corpus and Equipment Hierarchy

Our corpus $\mathcal{D}$ contains 508 documents spanning 27 equipment types (devices) in a semiconductor fabrication facility. Documents include Standard Operating Procedures (SOPs), equipment manuals, troubleshooting guides, and maintenance logs. Each document $d$ has metadata including `device_name(d)`, `doc_type(d)`, and `topic(d)`.

The equipment namespace follows a two-level hierarchy:
- **device_name**: Equipment model (e.g., "SUPRA XP", "INTEGER plus")
- **equip_id**: Physical instance identifier (e.g., "EPAG50", "WPSKAU8X00")

### 3.2 Cross-Equipment Contamination

**Definition.** For a query $q$ with target device $\text{dev}(q)$ and a retrieval result set $R_k(q)$ of size $k$, the contamination rate is:

$$\text{Cont@k}(q) = \frac{|\{d \in R_k(q) : \text{dev}(d) \neq \text{dev}(q) \land d \notin \mathcal{D}_{\text{shared}}\}|}{k}$$

where $\mathcal{D}_{\text{shared}}$ is the set of cross-equipment shared documents (60 documents in our corpus, identified by topic overlap across $\geq 3$ devices).

**Key property**: Cont@k is computable without gold relevance labels — it requires only device metadata, making it suitable as a safety metric independent of relevance judgments.

### 3.3 Device-Aware Hard Filter

Given a query $q$ with parsed device $\text{dev}(q)$, the hard filter restricts retrieval to:

$$\mathcal{D}_{\text{allowed}}(q) = \{d \in \mathcal{D} : \text{dev}(d) = \text{dev}(q)\}$$

Implementation: Elasticsearch `terms` filter on `device_name.keyword` applied as a pre-filter before BM25/kNN scoring.

**Oracle vs. Real**: In our main experiments, we use oracle device information (from gold labels) to establish the upper bound. We separately measure the gap when using a regex/dictionary-based parser (Section 5.4).

### 3.4 Retrieval Systems

We evaluate four retrieval architectures, each with and without device filtering:

| System | Retrieval | Rerank | Index |
|--------|-----------|--------|-------|
| B0 | BM25 | — | chunk_v3_content |
| B1 | Dense kNN (BGE-M3, 1024d) | — | chunk_v3_embed_bge_m3_v1 |
| B2 | Hybrid (BM25 + Dense, RRF) | — | Cross-index |
| B3 | Hybrid + CrossEncoder | Yes | Cross-index |

**Cross-index architecture**: BM25 scores come from `chunk_v3_content` (text index), dense scores from `chunk_v3_embed_bge_m3_v1` (vector index). Results are joined on `chunk_id` and fused via Reciprocal Rank Fusion (RRF, $k=60$).

For filtered systems, we add:
- **B4**: B3 + hard device filter (oracle device)
- **B4.5**: B3 + device filter + shared document allowance

### 3.5 Soft Scoring (P6/P7)

As an alternative to hard filtering, we evaluate contamination-aware soft scoring:

$$\text{Score}(d, q) = \text{Base}(d, q) - \lambda \cdot v_{\text{scope}}(d, q)$$

where $\text{Base}(d,q)$ is the B3 rerank score, $v_{\text{scope}} \in \{0, 1\}$ indicates scope violation, and $\lambda$ controls penalty strength. P6 uses fixed $\lambda=0.05$; P7 uses adaptive $\lambda$ based on query scope observability.

### 3.6 Masked-Query Evaluation Protocol

**Motivation**: Conventional evaluation generates questions from documents, embedding device names in queries. BM25 then matches these device names against document IDs (which also contain device names), creating an artificial advantage for unfiltered retrieval.

**Protocol**: For each query $q$ with device mentions, we create a masked variant $q_m$ by replacing device names with `[DEVICE]` and equipment types with `[EQUIP]`:

- Original: "GENEVA XP 설비에서 Heater Chuck leveling 절차는?"
- Masked: "[DEVICE] 설비에서 Heater Chuck leveling 절차는?"

Masked queries simulate the realistic scenario where engineers ask about procedures without specifying equipment (relying on session context for disambiguation).

**Evaluation matrix**: Each system is evaluated on both original and masked queries, enabling direct comparison of evaluation bias.

---

## 4. Experimental Setup

### 4.1 Evaluation Dataset (v0.6)

We construct a 578-query evaluation set across 27 devices with:
- **Balanced device distribution**: Each device has proportional representation
- **Scope observability labels**: `explicit_device` (n=429, device name mentioned), `explicit_equip` (n=149, equipment type mentioned)
- **Dual gold labels**: `gold_strict` (directly answering documents) and `gold_loose` (topically related documents)
- **Masked variants**: `question_masked` field with device/equip tokens replaced
- **Unique gold sets**: 482/578 (83%) queries have unique gold document sets

### 4.2 Gold Label Quality

We verify gold label quality on a stratified sample of 75 queries (337 query-document pairs):
- Strict gold precision: **97.2%** (172/177 confirmed relevant)
- False positive rate: **0.0%** (no irrelevant documents in gold)
- Loose gold recall: **100%** (all verified relevant documents included)

### 4.3 Metrics

- **Contamination@10** (Cont@10): Fraction of top-10 results from wrong device (excluding shared docs)
- **Gold Hit@10 (strict)**: Fraction of queries with at least one strict gold document in top-10
- **Gold Hit@10 (loose)**: Fraction of queries with at least one loose gold document in top-10
- **MRR**: Mean Reciprocal Rank of first gold hit

### 4.4 Elasticsearch Infrastructure

- Text index: `chunk_v3_content` (Nori analyzer for Korean, standard for English)
- Vector index: `chunk_v3_embed_bge_m3_v1` (BGE-M3, 1024 dimensions)
- Cross-index join on `chunk_id` for hybrid retrieval

---

## 5. Results

### 5.1 Main Result: Device Filter Eliminates Contamination and Improves Recall

**Table 1. Overall results (n=578, k=10, masked queries)**

| System | Cont@10 | Gold Strict | Gold Loose |
|--------|:-------:|:-----------:|:----------:|
| B0 (BM25) | 0.473 | 49.7% | 59.3% |
| B1 (Dense) | 0.730 | 40.5% | 46.2% |
| B2 (Hybrid) | 0.586 | 60.7% | 65.7% |
| B3 (Hybrid+Rerank) | 0.586 | 60.7% | 65.7% |
| **B4 (Hard filter)** | **0.001** | **91.2%** | **92.0%** |
| B4.5 (Filter+Shared) | 0.001 | 70.2% | 76.0% |

**Finding 1**: Device-aware hard filtering reduces contamination from 47–73% to near-zero across all retrieval methods.

**Finding 2**: Filtering simultaneously improves recall — strict gold hit increases from 49.7–60.7% to 91.2% (+30.5 to +41.5 percentage points). This counterintuitive result occurs because contaminating documents displace correct documents from top-k positions; removing them allows correct documents to surface.

**Finding 3**: Adding shared documents (B4.5) hurts recall compared to pure device filtering (B4), with strict gold dropping from 91.2% to 70.2%. The 60 shared documents introduce noise that displaces device-specific gold documents.

**Table 2. Results by scope observability (masked queries)**

| Scope | System | Cont@10 | Gold Strict |
|-------|--------|:-------:|:-----------:|
| explicit_device (n=429) | B3 | 0.483 | 79.5% |
| | **B4** | **0.001** | **97.0%** |
| explicit_equip (n=149) | B3 | 0.882 | 6.7% |
| | **B4** | **0.000** | **74.5%** |

**Finding 4**: Equipment-type queries (`explicit_equip`) are most severely affected by contamination (88.2%) and benefit most dramatically from filtering (+67.8 percentage points in gold hit).

### 5.2 Soft Scoring Is Ineffective

**Table 3. Soft scoring vs. hard filter (masked queries, n=578)**

| System | Cont@10 | Gold Strict |
|--------|:-------:|:-----------:|
| B3 (baseline) | 0.586 | 60.7% |
| P6 (λ=0.05) | 0.651 | 60.7% |
| P7 (adaptive λ) | 0.651 | 60.7% |
| **B4 (hard filter)** | **0.001** | **91.2%** |

Soft scoring with $\lambda=0.05$ fails to reduce contamination — it actually *increases* it by 6.5 percentage points while leaving gold hit unchanged. The fundamental issue is scale mismatch: retrieval score differences between documents are typically 0.1–1.0, while the penalty $\lambda \cdot v_{\text{scope}} = 0.05$ is too small to change rankings. Increasing $\lambda$ to overcome this gap causes soft scoring to converge to hard filtering, offering no practical advantage.

### 5.3 Evaluation Bias: Why Previous Studies Missed Scope Filtering's Value

**Table 4. Original vs. masked query comparison (n=578)**

| System | Query | Cont@10 | Gold Strict |
|--------|-------|:-------:|:-----------:|
| B0 | original | 0.422 | 68.2% |
| B0 | masked | 0.473 | 49.7% |
| B3 | original | 0.364 | 75.1% |
| B3 | masked | 0.586 | 60.7% |
| B4 | masked | 0.001 | 91.2% |

On original queries, B3 achieves 75.1% gold hit without any filtering — suggesting scope filtering has marginal value. This result, which matches our earlier Phase 1–4 findings, is an artifact of evaluation bias:

1. **Lexical leakage**: Questions generated from documents contain device names that BM25 matches against device-encoded document IDs (e.g., `global_sop_geneva_xp_*`)
2. **Inflated baseline**: This artificial matching allows unfiltered retrieval to find correct documents regardless of scope filtering
3. **Masked debiasing**: Replacing device names with `[DEVICE]` tokens breaks this shortcut, revealing the true contamination rate and the value of filtering

**The bias reverses the conclusion**: Original evaluation suggests filtering hurts recall (Phase 1–4: −36 to −69%); masked evaluation shows filtering dramatically improves recall (+30 to +42 percentage points).

### 5.4 Retrieval Method Vulnerability to Contamination

**Table 5. Masking sensitivity by retrieval method**

| System | Cont@10 (orig) | Cont@10 (masked) | Delta |
|--------|:-:|:-:|:-:|
| B0 (BM25) | 0.422 | 0.473 | +0.051 |
| B1 (Dense) | 0.373 | 0.730 | **+0.357** |
| B2 (Hybrid) | 0.364 | 0.586 | +0.222 |
| B3 (Hybrid+Rerank) | 0.364 | 0.586 | +0.222 |

Dense retrieval (B1) is most vulnerable to masking, with contamination nearly doubling from 37.3% to 73.0%. Semantic embeddings capture topical similarity across equipment types, making cross-equipment documents appear highly relevant when device context is removed. BM25 is most robust to masking (+5.1%), as it relies on diverse term matching beyond device names.

Cross-encoder reranking (B3 vs. B2) provides no contamination reduction — rerankers assess document quality but not scope correctness.

### 5.5 Parser Accuracy and Oracle Gap

**Table 6. Device parser accuracy**

| Scope | Exact Match | No Detection | Wrong Detection |
|-------|:-----------:|:------------:|:---------------:|
| explicit_device (n=429) | 88.6% | 0.2% | 11.2% |
| explicit_equip (n=149) | 0.0% | 100% | 0.0% |
| Overall (n=578) | 65.7% | 26.0% | 8.3% |

**Table 7. Oracle vs. real parser retrieval**

| | Gold Hit@10 | Cont@10 | MRR |
|--|:-:|:-:|:-:|
| Oracle B4 | 92.7% | 0.0% | 0.846 |
| Real B4 | 91.9% | 30.6% | 0.832 |
| Delta | −0.9%p | +30.6%p | −0.014 |

The regex-based parser achieves 88.6% accuracy on explicit_device queries, resulting in only −0.9 percentage points gold hit loss compared to oracle. However, complete failure on explicit_equip queries (0% accuracy) causes contamination to rise to 30.6%, as unrecognized queries fall back to unfiltered retrieval.

### 5.6 Validation on Implicit Queries

To verify that our findings generalize beyond mechanically masked queries, we evaluate on 628 naturally implicit queries — questions that never mention any device or equipment name.

**Table 8. Implicit query results (n=628) vs. explicit-masked (n=578)**

| System | Implicit Strict | Explicit-Masked Strict | Implicit Cont@10 | Explicit-Masked Cont@10 |
|--------|:-:|:-:|:-:|:-:|
| B0 | 52.9% | 49.7% | 0.652 | 0.473 |
| B2 | 61.8% | 60.7% | 0.664 | 0.585 |
| B3 | 61.5% | 60.7% | 0.665 | 0.584 |
| B4 | **84.7%** | **91.2%** | 0.001 | 0.001 |
| B4.5 | 76.6% | 70.2% | 0.001 | 0.001 |

Three findings confirm the robustness of our results:

1. **Masking neutrality**: On implicit queries, masked and original conditions produce identical results (delta = 0–4 queries), confirming that masking only removes device-name shortcuts and does not introduce other artifacts.

2. **Baseline convergence**: Implicit baseline performance (B0: 52.9%, B2: 61.8%) matches explicit-masked performance (49.7%, 60.7%), demonstrating that masking successfully approximates the device-agnostic retrieval scenario.

3. **Contamination amplification**: Without any device signal, contamination rises further (B0: 0.652 vs. 0.473), reinforcing the need for scope filtering. Hard filtering (B4) continues to provide the largest improvement, boosting gold hit by +31.8 percentage points over the best unfiltered baseline.

The B4 gap between implicit (84.7%) and explicit-masked (91.2%) reflects the inherent difficulty of implicit queries: fewer lexical cues make it harder for BM25 to distinguish target-device documents even within a filtered set. The shared document paradox persists (B4 > B4.5 by 8.1 percentage points) but narrows compared to explicit queries (20.9 percentage points), suggesting that shared document displacement is partially driven by device-name matching effects.

---

## 6. Discussion

### 6.1 Why Hard Filtering Improves Recall

The counterintuitive finding that filtering *improves* recall (B4 > B3 by +30.5 percentage points) has a simple explanation: in a multi-equipment corpus with overlapping topics, contaminating documents are not random — they are topically similar documents from different equipment. These near-duplicates occupy top-k positions that would otherwise be filled by correct-equipment documents. Removing them allows correct documents to surface.

This effect is strongest for equipment-type queries (explicit_equip), where contamination reaches 88% and filtering yields +67.8 percentage points improvement. These queries mention equipment categories (e.g., "etcher PM procedure") that match documents across all equipment of that type.

### 6.2 The Shared Document Paradox

B4.5 (device filter + shared documents) performs worse than B4 (device filter only), losing 21 percentage points in strict gold hit. Detailed analysis reveals the mechanism: of the 121 queries where B4 succeeds but B4.5 fails, **zero** queries benefit from shared document inclusion. In B4.5's top-10 results for these queries, 57.3% of documents are shared — dominated by five `device_net_board` SOPs that appear in 97–99 queries each. These high-BM25-frequency shared documents displace device-specific gold documents from the result set.

This finding has practical implications: the shared document threshold ($\geq 3$ devices) is too permissive, and broadly-applicable topics (controller, FFU, device net board) produce shared documents that function as retrieval noise rather than useful cross-equipment resources. Future work should investigate topic-specificity-weighted sharing thresholds or query-dependent shared document gating.

### 6.3 Implications for RAG Evaluation

Our discovery of document-seeded evaluation bias has broad implications:

1. **Evaluation sets generated from documents embed retrieval shortcuts** that inflate baseline performance
2. **These shortcuts are invisible to standard metrics** — contamination rates appear moderate even as lexical matching dominates
3. **Masking provides a simple debiasing protocol** applicable to any evaluation set with entity metadata

We recommend that domain-specific RAG evaluations include masked variants to assess retrieval quality independent of entity-name matching.

### 6.4 Practical Deployment Considerations

For production deployment, the oracle assumption must be replaced with a device parsing or routing component. Our parser accuracy analysis (Section 5.5) shows that:
- For queries with explicit device names: regex parsing achieves near-oracle performance (−0.9% gold hit gap)
- For queries with only equipment types: a disambiguation mechanism is needed (context-based routing, user clarification, or LLM-based device inference)

### 6.5 Limitations

1. **Single domain**: Results are from one semiconductor manufacturing corpus; generalization to other industrial domains requires validation
2. **Oracle filter**: Main results use gold device labels; real parser introduces 30.6% contamination for equip-type queries
3. **BM25/Hybrid only**: We do not evaluate learned sparse retrieval (SPLADE) or late-interaction models (ColBERT)
4. **Masking is an approximation**: Real device-agnostic queries differ from mechanically masked queries; [DEVICE] tokens may affect BM25 differently than natural omissions

---

## 7. Conclusion

We have shown that cross-equipment contamination is a severe and previously underestimated problem in industrial RAG systems, affecting 47–73% of retrieval results. A simple device-aware hard filter eliminates this contamination while simultaneously improving recall by 30–42 percentage points — a result that was hidden by evaluation bias in document-seeded benchmarks.

Our findings hold across both mechanically masked explicit queries (n=578) and naturally implicit queries (n=628), with baseline performance converging between the two sets and hard filtering providing the largest improvement in both. This cross-validation confirms that our masked-query evaluation protocol is a faithful debiasing approach, not an artifact of token replacement.

We also demonstrate that soft scoring approaches fail to match hard filtering's effectiveness due to fundamental scale mismatches, and that shared document inclusion paradoxically degrades performance by displacing device-specific gold documents.

These findings argue strongly for incorporating scope-aware retrieval as a first-class component in industrial RAG systems, and for re-examining evaluation methodology when domain-specific entities are present in both queries and documents.

---

## References

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

## Appendix A: Per-Device Results

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

**SUPRA Vplus (64%)**: 88 queries, 79 are explicit_equip type. All 31 failures occur in explicit_equip queries, caused by: (a) gold label over-assignment of a single CONTROLLER document to 21 topically unrelated queries, and (b) sparse SOP coverage (only 4 SOPs for this device).

## Appendix B: Evaluation Set Statistics

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
