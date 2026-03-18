# Device-Aware Scope Filtering for Cross-Equipment Contamination Control in Industrial RAG

**Draft v2** — 2026-03-14, based on masked query experiments

---

## Abstract

Retrieval-Augmented Generation (RAG) systems deployed in semiconductor manufacturing must retrieve equipment-specific documents from shared corpora spanning dozens of equipment types. We identify **cross-equipment contamination** — the retrieval of documents belonging to unrelated equipment — as a critical failure mode, affecting 47–73% of top-10 results depending on the retrieval method. We show that a simple device-aware hard filter eliminates contamination entirely while simultaneously improving recall by up to 42 percentage points. Crucially, we demonstrate that conventional document-seeded evaluation systematically underestimates the value of scope filtering due to lexical bias in gold labels, and propose **masked-query evaluation** as a debiasing protocol. Our experiments on 1,206 queries (578 explicit + 628 implicit) across 27 equipment types and four retrieval architectures (BM25, Dense, Hybrid, Hybrid+Rerank) provide strong evidence that scope-aware retrieval is essential for industrial RAG safety.

---

## 1. Introduction

Large-scale RAG systems in semiconductor fabrication environments typically index documents across multiple equipment types into a unified corpus. Maintenance engineers query this shared corpus for troubleshooting procedures, preventive maintenance guides, and operational parameters. When the retrieval system returns documents from the wrong equipment, the downstream LLM may generate answers grounded in incorrect procedures — a failure with direct safety implications in high-stakes manufacturing.

We term this failure mode **cross-equipment contamination**: the presence of out-of-scope equipment documents in top-k retrieval results. Unlike general relevance failures, contamination represents a systematic bias where topically similar but equipment-incorrect documents displace correct answers. For example, a heater chuck replacement procedure for SUPRA XP may be lexically and semantically similar to the procedure for GENEVA XP, but applying one to the other risks equipment damage.

Prior work on domain-specific RAG has focused on improving overall retrieval quality through hybrid retrieval, reranking, and query expansion. However, the **scope correctness** dimension — whether retrieved documents belong to the correct equipment context — has received limited attention. We argue that in multi-equipment industrial corpora, scope filtering is not merely an optimization but a safety requirement.

### Contributions

This paper makes five contributions:

1. **Context-Aware Scope Routing**: We propose a three-stage pipeline — query-level device parsing, sticky context inheritance across turns, and pre-retrieval hard filtering — that automatically determines equipment scope from conversational context and eliminates cross-equipment contamination (Section 3.3).

2. **Contamination quantification**: We define Contamination@k as a gold-label-free safety metric and measure it across four retrieval architectures, finding contamination rates of 47–73% on masked queries (Section 5.1).

3. **Counterintuitive recall improvement**: We demonstrate that scope filtering not only eliminates contamination but simultaneously improves recall by 30–42 percentage points — because contaminating documents displace correct ones from top-k positions (Section 5.1).

4. **Evaluation bias discovery**: We show that conventional document-seeded evaluation creates a circular bias that masks the value of scope filtering, and propose masked-query evaluation as a debiasing protocol. Without masking, filtering appears to *hurt* recall (−36 to −69%); with masking, filtering dramatically *improves* recall (+30 to +42pp) (Section 5.3).

5. **Soft scoring failure and recovery direction**: We provide negative evidence that contamination-aware soft scoring (P6/P7) is ineffective compared to hard filtering, then show that in an oracle-assisted cached-candidate simulation, a confidence-gated hybrid variant (P7+) improves strict hit@10 from 60.7% (B3) to 87.5% while reducing contamination@10 from 0.584 to 0.515, yet still trails oracle hard filtering (91.2% strict hit@10; contamination@10=0.001) (Section 5.2).

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

### 3.3 Context-Aware Scope Routing

The core algorithm is a **multi-turn scope routing pipeline** that determines the target device from conversational context and applies it as a pre-retrieval filter. The pipeline operates in three stages:

**Stage 1: Query-Level Device Parsing.** For each user query $q_t$ at turn $t$, a rule-based parser (regex + dictionary matching against 27 known device names) extracts device mentions:

$$\text{parse}(q_t) \rightarrow D_{\text{detected}} \subseteq \mathcal{V}_{\text{devices}}$$

The parser handles normalized forms, abbreviations, and partial matches (e.g., "SUPRA XP", "supra-xp", "SXP").

**Stage 2: Sticky Context Inheritance.** When the parser detects no device in the current query ($D_{\text{detected}} = \emptyset$), the system inherits the device scope from the previous turn:

$$\text{dev}(q_t) = \begin{cases} D_{\text{detected}}[0] & \text{if } D_{\text{detected}} \neq \emptyset \\ \text{dev}(q_{t-1}) & \text{if } D_{\text{detected}} = \emptyset \text{ and } t > 1 \\ \emptyset & \text{otherwise (no filter)} \end{cases}$$

This **sticky device policy** reflects the natural structure of troubleshooting sessions: an engineer opens a session about a specific equipment and asks multiple follow-up questions without repeating the device name. Example:

| Turn | Query | Detected | Effective Device |
|:----:|-------|----------|:----------------:|
| 1 | "SUPRA XP의 PM 절차는?" | SUPRA XP | SUPRA XP |
| 2 | "heater chuck 교체 방법은?" | ∅ | SUPRA XP (inherited) |
| 3 | "INTEGER plus에서 같은 절차는?" | INTEGER plus | INTEGER plus |
| 4 | "다음 단계는?" | ∅ | INTEGER plus (inherited) |

**Stage 3: Pre-Retrieval Hard Filter.** When $\text{dev}(q_t) \neq \emptyset$, an Elasticsearch `terms` filter restricts retrieval to matching documents:

$$\mathcal{D}_{\text{allowed}}(q_t) = \{d \in \mathcal{D} : \text{dev}(d) = \text{dev}(q_t)\}$$

When $\text{dev}(q_t) = \emptyset$ (no device detected and no history), the system falls back to unfiltered retrieval over the full corpus.

**Oracle vs. Real**: In our main experiments, we use oracle device information (from gold labels) to establish the upper bound of Stage 3 in isolation. We separately measure the end-to-end gap including Stage 1 parser errors (Section 5.5). The sticky policy (Stage 2) is evaluated implicitly through the implicit query experiment (Section 5.6), where queries never mention devices — simulating the case where all device context must come from prior turns.

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

### 3.5 Method Overview

The proposed system, **Context-Aware Scope Routing**, is a three-stage pipeline that automatically determines the target equipment from multi-turn conversational context and applies it as a pre-retrieval constraint (Section 3.3). The key insight is that in troubleshooting sessions, the equipment context is typically established early and persists across subsequent queries — a pattern we exploit through the sticky device policy.

This is fundamentally a **pre-retrieval scope control** approach: we constrain retrieval candidates using device scope before any scoring or ranking occurs. This design reflects the safety objective — preventing out-of-scope candidates from entering top-k is more direct and reliable than trying to demote them after retrieval.

We include contamination-aware soft scoring (P6/P7, Section 3.6) as a comparative alternative to test whether post-hoc penalties can substitute for hard scope filtering.

### 3.6 Soft Scoring (P6/P7)

As an alternative to hard filtering, we evaluate contamination-aware soft scoring that penalizes out-of-scope documents without removing them:

$$\text{Score}(d, q) = \text{Base}(d, q) - \lambda \cdot v_{\text{scope}}(d, q)$$

where $\text{Base}(d,q) = 1/(r+1)$ is a rank-based score derived from the B3 reranking order ($r$ = 0-indexed rank), and the scope violation indicator is:

$$v_{\text{scope}}(d, q) = \begin{cases} 0 & \text{if } \text{dev}(d) = \text{dev}(q) \text{ (in-scope)} \\ 0 & \text{if } d \in \mathcal{D}_{\text{shared}} \text{ (shared document)} \\ 1 & \text{otherwise (out-of-scope)} \end{cases}$$

We evaluate two penalty strategies:

- **P6 (fixed λ)**: $\lambda = 0.05$, applied uniformly to all queries.
- **P7 (adaptive λ)**: $\lambda_q = 0.05 \times \frac{|\{d \in R_k(q) : v_{\text{scope}}(d,q)=1\}|}{|R_k(q)|}$, scaling the penalty proportionally to the contamination level in the initial result set. Queries with higher initial contamination receive stronger penalties.

The motivation for soft scoring is robustness to parser errors: unlike hard filtering, soft scoring does not completely exclude documents when the parsed device is wrong, preserving a fallback path through the original ranking.

We also evaluate an extension, **P7+**, that combines hard/soft signals under a confidence-gated policy. P7+ blends cached candidates from B3/B4/B4.5, applies shared-document exposure caps in early ranks, and uses scope-dependent penalties. This variant is reported as an **offline ablation** (oracle-assisted candidate pool + confidence proxy), not as an end-to-end production retrieval result.

### 3.7 Masked-Query Evaluation Protocol

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

### 4.2 Implicit Evaluation Dataset (v0.7)

To validate findings beyond mechanically masked queries, we construct a 628-query implicit set where queries never mention any device or equipment name:
- **578 implicit queries** (`scope_observability = ambiguous`): generated from the same source documents as v0.6, but phrased without device/equipment references
- **50 trap queries**: cross-device topic queries designed to test contamination boundary cases (e.g., shared maintenance procedures)
- Gold labels are inherited from the source documents' device assignments

This set enables direct comparison: explicit-masked queries (v0.6 with masking) vs. naturally implicit queries (v0.7), testing whether mechanical masking faithfully approximates real device-agnostic retrieval.

### 4.3 Gold Label Quality (v0.6)

We verify gold label quality on a stratified sample of 75 queries (337 query-document pairs):
- Strict gold precision: **97.2%** (172/177 confirmed relevant)
- False positive rate: **0.0%** (no irrelevant documents in gold)
- Loose gold recall: **100%** (all verified relevant documents included)

### 4.4 Metrics

- **Contamination@10** (Cont@10): Fraction of top-10 results from wrong device (excluding shared docs)
- **Gold Hit@10 (strict)**: Fraction of queries with at least one strict gold document in top-10
- **Gold Hit@10 (loose)**: Fraction of queries with at least one loose gold document in top-10
- **MRR**: Mean Reciprocal Rank of first gold hit

### 4.5 Elasticsearch Infrastructure

- Text index: `chunk_v3_content` (Nori analyzer for Korean, standard for English)
- Vector index: `chunk_v3_embed_bge_m3_v1` (BGE-M3, 1024 dimensions)
- Cross-index join on `chunk_id` for hybrid retrieval

---

## 5. Results

### 5.1 Main Result: Device Filter Eliminates Contamination and Improves Recall

**Table 1. Overall results (n=578, k=10, masked queries)**

| System | Cont@10 | Gold Strict | Gold Loose | MRR |
|--------|:-------:|:-----------:|:----------:|:---:|
| B0 (BM25) | 0.473 | 49.7% | 59.3% | 0.356 |
| B1 (Dense) | 0.730 | 39.4% | 45.2% | 0.250 |
| B2 (Hybrid) | 0.585 | 60.7% | 65.7% | 0.420 |
| B3 (Hybrid+Rerank) | 0.584 | 60.7% | 65.7% | 0.340 |
| **B4 (Hard filter)** | **0.001** | **91.2%** | **92.0%** | **0.568** |
| B4.5 (Filter+Shared) | 0.001 | 70.2% | 76.0% | 0.425 |

All pairwise differences in gold hit are statistically significant (McNemar test, p < 0.001) except B2 vs. B3 (p = 0.48). Key comparisons with bootstrap 95% CIs on strict gold hit rate:
- B3 → B4: Δ = +30.4pp [+26.6, +34.4], p < 10⁻³⁰ (discordant: 179 improved, 3 degraded)
- B4 → B4.5: Δ = −20.9pp [−24.2, −17.6], p < 10⁻²⁵ (discordant: 0 improved, 121 degraded)
- B0 → B2: Δ = +11.1pp [+8.0, +14.4], p < 10⁻⁹

**Finding 1**: Device-aware hard filtering reduces contamination from 47–73% to near-zero across all retrieval methods.

**Finding 2**: Filtering simultaneously improves recall — strict gold hit increases from 49.7–60.7% to 91.2% (+30.5 to +41.5 percentage points). This counterintuitive result occurs because contaminating documents displace correct documents from top-k positions; removing them allows correct documents to surface.

**Finding 3**: Adding shared documents (B4.5) hurts recall compared to pure device filtering (B4), with strict gold dropping from 91.2% to 70.2% (McNemar p < 10⁻²⁵). In all 121 discordant pairs, B4 succeeds where B4.5 fails — zero queries benefit from shared document inclusion.

**Table 2. Results by scope observability (masked queries)**

| Scope | System | Cont@10 | Gold Strict |
|-------|--------|:-------:|:-----------:|
| explicit_device (n=429) | B3 | 0.481 | 79.5% |
| | **B4** | **0.001** | **97.0%** |
| explicit_equip (n=149) | B3 | 0.881 | 6.7% |
| | **B4** | **0.000** | **74.5%** |

**Finding 4**: Equipment-type queries (`explicit_equip`) are most severely affected by contamination (88.1%) and benefit most dramatically from filtering (+67.8 percentage points in gold hit).

**Table 2b. Recall@k and ranking quality (masked queries, n=578)**

| System | R@1 | R@3 | R@5 | R@10 | NDCG@10 | MRR |
|--------|:---:|:---:|:---:|:----:|:-------:|:---:|
| B0 (BM25) | 28.5% | 41.0% | 43.8% | 49.7% | 0.282 | 0.356 |
| B1 (Dense) | 18.2% | 29.8% | 34.9% | 39.4% | 0.192 | 0.250 |
| B2 (Hybrid) | 32.7% | 48.3% | 55.5% | 60.7% | 0.325 | 0.420 |
| B3 (Hybrid+Rerank) | 22.3% | 40.1% | 50.0% | 60.7% | 0.290 | 0.340 |
| **B4 (Hard filter)** | **41.3%** | **65.9%** | **82.5%** | **91.2%** | **0.588** | **0.568** |
| B4.5 (Filter+Shared) | 31.7% | 47.4% | 58.8% | 70.2% | 0.390 | 0.425 |

**Finding 5**: B4's advantage is consistent across all cutoff values, growing from +19.0pp at k=1 to +32.5pp at k=5 before slightly narrowing at k=10. This monotonic growth indicates that device filtering not only retrieves more gold documents but also ranks them higher (NDCG@10: 0.588 vs. 0.290 for B3, a 2× improvement). Notably, cross-encoder reranking (B3) *hurts* early precision compared to Hybrid without reranking (B2): R@1 drops from 32.7% to 22.3%, suggesting that the reranker promotes contaminating documents that are topically relevant but scope-incorrect.

### 5.2 Soft Scoring Fails, but P7+ Shows Recovery in Offline Ablation

**Table 3. Soft scoring vs. hard filter (masked queries, n=578)**

| System | Gold Strict | Gold Loose | MRR | NDCG@10 |
|--------|:-----------:|:----------:|:---:|:-------:|
| B3 (baseline) | 60.7% | 65.7% | 0.340 | 0.290 |
| P6 (λ=0.05) | 60.7% | 65.7% | 0.343 | 0.292 |
| P7 (adaptive λ) | 60.7% | 65.7% | 0.342 | 0.291 |
| **B4 (hard filter)** | **91.2%** | **92.0%** | **0.568** | **0.588** |

**Table 3b. Recall@k for soft scoring vs. hard filter (masked queries, n=578)**

| System | R@1 | R@3 | R@5 | R@10 |
|--------|:---:|:---:|:---:|:----:|
| B3 (baseline) | 22.3% | 40.1% | 50.0% | 60.7% |
| P6 (λ=0.05) | 22.3% | 40.1% | 51.7% | 60.7% |
| P7 (adaptive λ) | 22.3% | 40.1% | 51.4% | 60.7% |
| **B4 (hard filter)** | **41.3%** | **65.9%** | **82.5%** | **91.2%** |

Soft scoring leaves all key metrics unchanged: P6 and P7 produce identical gold hit, near-identical MRR, and virtually the same Recall@k curve as B3 at all cutoff values. Per-query analysis confirms that across 578 queries, **zero** queries change gold hit status between B3 and P6/P7 — the penalty is too small to move any document across a metric-relevant threshold. B4 dominates across all k, with NDCG@10 2× that of any soft scoring variant.

The failure has three structural causes:

**Cause 1: Scale mismatch.** The rank-based base score $1/(r+1)$ creates gaps between adjacent ranks that far exceed the penalty $\lambda = 0.05$:

| Rank pair | Score gap | λ=0.05 sufficient? |
|:---------:|:---------:|:-------------------:|
| 0 → 1 | 0.500 | No (10× too small) |
| 1 → 2 | 0.167 | No (3× too small) |
| 2 → 3 | 0.083 | No |
| 3 → 4 | 0.050 | Marginal (tie) |
| 4 → 5 | 0.033 | Yes |

To displace a rank-0 out-of-scope document below a rank-1 in-scope document requires $\lambda > 0.5$ — ten times the deployed value. More generally, correcting $k$ out-of-scope documents at the top requires $\lambda > k/(k+1)$, approaching 1.0 asymptotically. At $\lambda \geq 0.9$, soft scoring becomes functionally equivalent to hard filtering over the top-10 candidate set — there is no effective intermediate regime.

**Cause 2: Candidate set ceiling.** P6/P7 re-rank B3's top-10 results without introducing new documents. When B3 fails to retrieve the gold document in its top-10 (39.3% of queries), P6/P7 cannot produce a gold hit regardless of $\lambda$. This ceiling (B3 recall@10 = 60.7%) explains the entire 30.5 percentage point gap with B4 (91.2%): the gap is a candidate generation problem, not a re-ranking problem.

**Cause 3: λ convergence paradox.** Increasing $\lambda$ to overcome scale mismatch causes soft scoring to converge to hard filtering. At $\lambda = 0.9$, every in-scope document outscores every out-of-scope document regardless of rank — the soft penalty becomes a de facto binary filter. This means soft scoring offers no practical advantage over hard filtering: it is either too weak to change rankings or strong enough to replicate hard filtering exactly.

### 5.2.1 P7+ Hybrid Policy (Offline Ablation)

To test whether a non-trivial soft/hard hybrid can recover value beyond P6/P7, we run P7+ with confidence-gated blending over cached B3/B4/B4.5 candidates.

**Table 3c. P7+ offline ablation (masked queries, n=578)**

| System | Cont@10 | Gold Strict | Gold Loose | Note |
|--------|:-------:|:-----------:|:----------:|------|
| B3 (baseline) | 0.584 | 60.7% | 65.7% | Unfiltered hybrid+rerank |
| P7 (adaptive λ) | 0.649 | 60.7% | 65.7% | Soft penalty only |
| **P7+ (gated hybrid)** | **0.515** | **87.5%** | **89.6%** | Cached-candidate policy simulation |
| B4 (hard filter) | 0.001 | 91.2% | 92.0% | Oracle upper bound |

P7+ substantially outperforms P6/P7 and B3 in strict hit (+26.8pp vs B3) while reducing contamination (-0.069 absolute vs B3). The largest gain appears on `explicit_equip` queries: strict hit rises from 10/149 to 104/149 and contamination drops from 0.881 to 0.589. However, P7+ still trails B4 in both contamination and strict hit, confirming that hard scope control remains the strongest condition when reliable scope is available.

**Reporting caveat.** P7+ is an oracle-assisted, cached-candidate simulation: it uses B4/B4.5 candidate pools and an annotation-derived confidence proxy. Therefore, we treat P7+ as a promising algorithmic direction and ablation result, not as a final end-to-end deployment claim.

### 5.3 Evaluation Bias: Why Previous Studies Missed Scope Filtering's Value

**Table 4. Original vs. masked query comparison (n=578)**

| System | Query | Cont@10 | Gold Strict |
|--------|-------|:-------:|:-----------:|
| B0 | original | 0.422 | 68.2% |
| B0 | masked | 0.473 | 49.7% |
| B3 | original | 0.365 | 75.1% |
| B3 | masked | 0.584 | 60.7% |
| B4 | masked | 0.001 | 91.2% |

On original queries, B3 achieves 75.1% gold hit without any filtering — suggesting scope filtering has marginal value. This result, which matches our earlier Phase 1–4 findings, is an artifact of evaluation bias:

1. **Lexical leakage**: Questions generated from documents contain device names that BM25 matches against device-encoded document IDs (e.g., `global_sop_geneva_xp_*`)
2. **Inflated baseline**: This artificial matching allows unfiltered retrieval to find correct documents regardless of scope filtering
3. **Masked debiasing**: Replacing device names with `[DEVICE]` tokens breaks this shortcut, revealing the true contamination rate and the value of filtering

**The bias reverses the conclusion**: Original evaluation suggests filtering hurts recall (Phase 1–4: −36 to −69%); masked evaluation shows filtering dramatically improves recall (+30 to +42 percentage points). The masking effect is statistically significant for all retrieval methods (McNemar p < 10⁻¹³; B1 shows the largest drop: Δ = −26.1pp, B0: −18.5pp, B2/B3: −14.5pp).

### 5.4 Retrieval Method Vulnerability to Contamination

**Table 5. Masking sensitivity by retrieval method**

| System | Cont@10 (orig) | Cont@10 (masked) | Delta |
|--------|:-:|:-:|:-:|
| B0 (BM25) | 0.422 | 0.473 | +0.051 |
| B1 (Dense) | 0.373 | 0.730 | **+0.357** |
| B2 (Hybrid) | 0.365 | 0.585 | +0.220 |
| B3 (Hybrid+Rerank) | 0.365 | 0.584 | +0.219 |

Dense retrieval (B1) is most vulnerable to masking, with contamination nearly doubling from 37.3% to 73.0%. Semantic embeddings capture topical similarity across equipment types, making cross-equipment documents appear highly relevant when device context is removed. BM25 is most robust to masking (+5.1%), as it relies on diverse term matching beyond device names.

Cross-encoder reranking (B3 vs. B2) provides no contamination reduction — rerankers assess document quality but not scope correctness.

### 5.5 Parser Accuracy and Oracle Gap

**Table 6. Device parser accuracy**

| Scope | Exact Match | No Detection | Wrong Detection |
|-------|:-----------:|:------------:|:---------------:|
| explicit_device (n=429) | 88.6% | 0.2% | 11.2% |
| explicit_equip (n=149) | 0.0% | 100% | 0.0% |
| Overall (n=578) | 65.7% | 26.0% | 8.3% |

**Table 7. Oracle vs. real parser retrieval (BM25, loose gold, adjusted Cont@10)**

| | Gold Hit@10 (loose) | Cont_adj@10 | MRR |
|--|:-:|:-:|:-:|
| Oracle B4 | 92.7% | 0.0% | 0.846 |
| Real B4 | 91.9% | 30.6% | 0.832 |
| Delta | −0.9%p | +30.6%p | −0.014 |

*Note: This comparison uses BM25 retrieval and loose gold labels to isolate the parser accuracy effect. Adjusted contamination (Cont_adj) counts no-filter fallback documents as contaminating.*

The regex-based parser achieves 88.6% accuracy on explicit_device queries, resulting in only −0.9 percentage points gold hit loss compared to oracle. However, complete failure on explicit_equip queries (0% accuracy) causes contamination to rise to 30.6%, as 150 unrecognized queries fall back to unfiltered retrieval.

### 5.6 Validation on Implicit Queries

To verify that our findings generalize beyond mechanically masked queries, we evaluate on 628 naturally implicit queries — questions that never mention any device or equipment name.

**Table 8. Implicit query results (n=628) vs. explicit-masked (n=578)**

| System | Implicit Strict | Explicit-Masked Strict | Implicit Cont@10 | Explicit-Masked Cont@10 |
|--------|:-:|:-:|:-:|:-:|
| B0 | 52.9% | 49.7% | 0.652 | 0.473 |
| B1 | 40.3% | 39.4% | 0.735 | 0.730 |
| B2 | 61.8% | 60.7% | 0.664 | 0.585 |
| B3 | 61.5% | 60.7% | 0.665 | 0.584 |
| B4 | **84.7%** | **91.2%** | 0.001 | 0.001 |
| B4.5 | 76.6% | 70.2% | 0.001 | 0.001 |

**Table 8b. Recall@k and ranking quality on implicit queries (n=628, masked)**

| System | R@1 | R@3 | R@5 | R@10 | NDCG@10 | MRR |
|--------|:---:|:---:|:---:|:----:|:-------:|:---:|
| B0 | 31.8% | 44.9% | 49.4% | 52.9% | 0.344 | 0.394 |
| B1 | 19.6% | 32.6% | 35.8% | 40.3% | 0.209 | 0.268 |
| B2 | 33.3% | 49.4% | 58.0% | 61.8% | 0.357 | 0.430 |
| B3 | 30.4% | 49.8% | 56.1% | 61.5% | 0.349 | 0.412 |
| **B4** | **48.9%** | **71.8%** | **81.4%** | **84.7%** | **0.614** | **0.612** |
| B4.5 | 41.9% | 59.1% | 70.7% | 76.6% | 0.496 | 0.527 |

Implicit queries show even higher B4 gains at early ranks: R@1 = 48.9% vs. 30.4% (B3), a +18.5pp advantage. Notably, B4's MRR on implicit queries (0.612) exceeds its explicit-masked MRR (0.568), indicating that once the correct device scope is established, implicit queries — which lack device-name lexical noise — allow cleaner topic-based ranking within the filtered set. B3's MRR also improves (0.412 vs. 0.340), suggesting that cross-encoder reranking benefits from the absence of device-name distractors in implicit queries.

Three findings confirm the robustness of our results:

1. **Masking neutrality**: On implicit queries, masked and original conditions produce near-identical results (strict delta = 0–4 queries, loose delta = 0–7 queries), confirming that masking only removes device-name shortcuts and does not introduce other artifacts.

2. **Baseline convergence**: Implicit baseline performance (B0: 52.9%, B2: 61.8%) matches explicit-masked performance (49.7%, 60.7%) with no statistically significant difference (two-proportion z-test: B0 p = 0.27, B2 p = 0.71), demonstrating that masking successfully approximates the device-agnostic retrieval scenario.

3. **Contamination amplification**: Without any device signal, contamination rises further (B0: 0.652 vs. 0.473), reinforcing the need for scope filtering. Hard filtering (B4) continues to provide the largest improvement, boosting gold hit by +31.8 percentage points over the best unfiltered baseline.

The B4 gap between implicit (84.7%) and explicit-masked (91.2%) reflects the inherent difficulty of implicit queries: fewer lexical cues make it harder for BM25 to distinguish target-device documents even within a filtered set. The shared document paradox persists (B4 > B4.5 by 8.1 percentage points) but narrows compared to explicit queries (20.9 percentage points), suggesting that shared document displacement is partially driven by device-name matching effects.

---

## 6. Discussion

### 6.1 Why Hard Filtering Improves Recall

The counterintuitive finding that filtering *improves* recall (B4 > B3, the best unfiltered baseline, by +30.5 percentage points) has a simple explanation: in a multi-equipment corpus with overlapping topics, contaminating documents are not random — they are topically similar documents from different equipment. These near-duplicates occupy top-k positions that would otherwise be filled by correct-equipment documents. Removing them allows correct documents to surface.

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
- For queries with only equipment types: a disambiguation mechanism is needed

#### Equipment-Type Query Disambiguation

The 149 `explicit_equip` queries (0% parser accuracy) represent the primary deployment gap. We identify three viable disambiguation strategies:

1. **Conversational context routing**: In multi-turn sessions, the target device is often established in earlier turns. A session-level device tracker can propagate the most recently mentioned device to subsequent implicit or equip-type queries. This covers the common case where an engineer opens a session about a specific equipment and asks follow-up questions without repeating the device name.

2. **Equipment group filtering**: When the equipment type is known (e.g., "etcher") but the specific device is not, filter by the equipment family rather than a single device. This reduces contamination from cross-type sources (e.g., CVD documents in etcher queries) while preserving within-type results. The trade-off is higher within-family contamination, which our data shows is less severe than cross-type contamination.

3. **LLM-based device inference**: Use the LLM to infer the target device from query context, user profile (assigned equipment), and conversation history. This is the most flexible approach but introduces latency and potential inference errors.

For implicit queries (no device or equipment mention), strategy 1 is essential — our implicit experiment shows B4 achieves 84.7% even with oracle device, confirming that the retrieval quality is high once the correct device is known. The deployment challenge is device identification, not retrieval quality.

**Latency**: Device filtering via Elasticsearch `terms` filter adds no measurable latency overhead (p50: 2.2ms vs. 2.3ms unfiltered on our 508-document corpus). At larger corpus scales, pre-filtering should *reduce* latency by narrowing the candidate set before scoring.

### 6.5 Limitations

1. **Single domain**: Results are from one semiconductor manufacturing corpus; generalization to other industrial domains requires validation
2. **Oracle filter**: Main results use gold device labels; real parser introduces 30.6% contamination for equip-type queries
3. **BM25/Hybrid only**: We do not evaluate learned sparse retrieval (SPLADE) or late-interaction models (ColBERT)
4. **Masking approximation validated but not identical**: Our implicit query experiment (Section 5.6) confirms that masked and naturally implicit queries produce statistically equivalent baseline performance, but B4 shows a significant gap (84.7% implicit vs. 91.2% explicit-masked, p < 0.001), suggesting that implicit queries are inherently harder for within-device retrieval
5. **P7+ is offline ablation, not end-to-end**: The P7+ result uses cached candidate pools (including B4/B4.5 outputs) and an annotation-derived confidence proxy. It demonstrates policy potential, but does not yet establish production-ready end-to-end gains.

---

## 7. Conclusion

We have shown that cross-equipment contamination is a severe and previously underestimated problem in industrial RAG systems, affecting 47–73% of retrieval results. A simple device-aware hard filter eliminates this contamination while simultaneously improving recall by 30–42 percentage points — a result that was hidden by evaluation bias in document-seeded benchmarks.

Our findings hold across both mechanically masked explicit queries (n=578) and naturally implicit queries (n=628), with baseline performance converging between the two sets and hard filtering providing the largest improvement in both. This cross-validation confirms that our masked-query evaluation protocol is a faithful debiasing approach, not an artifact of token replacement.

We also show that naive soft scoring (P6/P7) fails to match hard filtering's effectiveness due to scale mismatches, while a confidence-gated hybrid extension (P7+) recovers most of the strict-hit gap in offline ablation but remains below hard-filter performance. Shared document inclusion still paradoxically degrades performance when not tightly controlled.

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

**SUPRA Vplus (64%)**: 88 queries, 79 are explicit_equip type. All 32 failures occur in explicit_equip queries, caused by: (a) gold label over-assignment of a single CONTROLLER document to 21 topically unrelated queries, and (b) sparse SOP coverage (only 4 SOPs for this device).

### B4 Failure Analysis

Of 51 B4 failures (8.8% of queries), 38 (75%) are `explicit_equip` type and 13 are `explicit_device`. Failures are concentrated in a few devices:

| Device | Failures | Total | Failure Rate | Primary Cause |
|--------|:--------:|:-----:|:------------:|---------------|
| SUPRA Vplus | 32 | 88 | 36% | Sparse SOP coverage + equip-level gold |
| SUPRA N | 6 | 82 | 7% | Topic overlap with SUPRA family |
| SUPRA XP | 3 | 26 | 12% | Cross-reference gold from shared SOPs |
| ZEDIUS XP | 3 | 41 | 7% | Gold assigned to SUPRA XP-origin SOPs |
| Other | 7 | 341 | 2% | Isolated cases |

The failure pattern is clear: B4's residual errors are driven by **equipment-type query gold label quality**, not retrieval failures. When the gold document is correctly assigned to the target device, B4 achieves near-perfect retrieval.

## Appendix B: Cross-Equipment Contamination Matrix

### B.1 Top Contamination Sources (B3 masked, non-shared documents)

| Source Device | Contaminating Docs | Target Devices Affected |
|---------------|:------------------:|:-----------------------:|
| SUPRA N | 205 | 15 |
| INTEGER plus | 179 | 12 |
| PRECIA | 107 | 11 |
| GENEVA XP | 100 | 11 |
| ZEDIUS XP | 72 | 10 |
| SUPRA N series | 72 | 10 |

Contamination is not random — devices with larger document counts (SUPRA N: 82 docs, INTEGER plus: 91 docs) produce more cross-device contamination due to broader topic coverage.

### B.2 Top 10 Contamination Pairs (Target ← Source)

| Target | Source | Occurrences |
|--------|--------|:-----------:|
| INTEGER plus | PRECIA | 173 |
| GENEVA XP | PRECIA | 149 |
| SUPRA N | PRECIA | 90 |
| INTEGER plus | SUPRA N series | 75 |
| ZEDIUS XP | PRECIA | 74 |
| INTEGER plus | SUPRA N | 68 |
| GENEVA XP | SUPRA N series | 64 |
| PRECIA | INTEGER plus | 55 |
| SUPRA N | INTEGER plus | 51 |
| PRECIA | SUPRA N | 48 |

**PRECIA is the dominant contamination source**, appearing in 4 of the top 5 pairs (INTEGER plus: 173, GENEVA XP: 149, SUPRA N: 90, ZEDIUS XP: 74). This occurs because PRECIA has diverse SOP coverage across many topics, producing strong BM25 matches to queries about other equipment. Contamination is **bidirectional but asymmetric**: PRECIA contaminates INTEGER plus 173 times vs. 55 in the reverse direction (3.1× ratio). Notably, same-family contamination (e.g., SUPRA XP ← SUPRA N) does not appear in the top 10, suggesting that cross-family topical overlap is a larger problem than within-family model confusion.

## Appendix C: BM25-Only Experiment Results

To isolate the effect of device filtering independent of dense retrieval and reranking, we evaluate BM25 (B0) with oracle device filter on both explicit and implicit queries.

**Table C1. BM25-only results by scope observability (masked, n=578)**

| Scope | System | Cont@10 | Gold Strict | Gold Loose |
|-------|--------|:-------:|:-----------:|:----------:|
| explicit_device (n=429) | B0 | 0.323 | 66.2% | 76.0% |
| | B4 (BM25+filter) | 0.000 | 97.2% | 97.2% |
| explicit_equip (n=149) | B0 | 0.903 | 2.0% | 11.4% |
| | B4 (BM25+filter) | 0.000 | 75.8% | 79.2% |
| ALL (n=578) | B0 | 0.473 | 49.7% | 59.3% |
| | B4 (BM25+filter) | 0.000 | 91.7% | 92.6% |

BM25 with device filter achieves 91.7% strict gold hit — comparable to B4 on Hybrid+Rerank (91.2%). This suggests that the device filter, not the retrieval architecture, is the primary driver of recall improvement. For `explicit_equip` queries, BM25 without filtering achieves only 2.0% strict gold hit (vs. 6.7% for B3), demonstrating that equipment-type queries are catastrophic for all unfiltered systems.

## Appendix C.2: Implicit Per-Device B4 Performance

**Table C2. B4 strict gold hit by device on implicit queries (n=628)**

| Device | B4 Strict | Queries | Note |
|--------|:---------:|:-------:|------|
| GENEVA XP | 98.6% | 69 | Near-perfect |
| INTEGER plus | 93.8% | 96 | |
| SUPRA N | 92.7% | 82 | |
| SUPRA XP | 92.3% | 26 | |
| PRECIA | 87.2% | 78 | |
| TIGMA Vplus | 81.7% | 60 | Lower than explicit (98%) |
| B4.5 average | 76.6% | 628 | Shared doc paradox persists |
| ZEDIUS XP | 63.9% | 61 | Coverage-limited |
| SUPRA Vplus | 60.2% | 93 | Coverage-limited |

The same devices that underperform on explicit queries (SUPRA Vplus, ZEDIUS XP) also underperform on implicit queries, confirming that the performance gap is driven by corpus coverage limitations and gold label quality rather than query type. TIGMA Vplus shows a notable drop from 98% (explicit) to 81.7% (implicit), suggesting that its explicit queries benefited from strong lexical cues that are absent in the implicit set.

## Appendix D: Evaluation Set Statistics

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
