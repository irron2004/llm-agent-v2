# Hierarchy-aware Scope Routing for Cross-Equipment Contamination Control in Industrial Maintenance RAG

---

# 3. Methodology

## 3.1 Problem Setting

We consider a Retrieval-Augmented Generation (RAG) system deployed in a semiconductor fabrication facility, where maintenance engineers query a shared document corpus spanning multiple equipment types. The corpus $\mathcal{D}$ contains documents of varying types (SOPs, setup manuals, troubleshooting guides, maintenance logs) across $|\mathcal{E}|$ distinct equipment devices.

**Cross-equipment contamination** occurs when documents from unrelated equipment appear in the top-$k$ retrieval results, potentially causing the generation module to produce answers grounded in incorrect procedures.

**Goal**: Design a scope routing policy that reduces contamination while preserving recall for relevant documents, including shared (cross-equipment) procedures.

## 3.2 Equipment Hierarchy

The equipment namespace follows a two-level hierarchy:

$$\text{device\_name} \rightarrow \text{equip\_id}$$

- **device_name**: Equipment model/type (e.g., "SUPRA XP", "INTEGER 45")
- **equip_id**: Physical instance identifier (e.g., "EPAG50")

Documents are authored at different granularity levels, motivating the scope_level classification (§3.4).

## 3.3 Allowed Scope

For a query $q$, we define the **allowed scope** $S(q)$ as the set of equipment entities whose documents are considered in-scope:

$$S(q) = S_{\text{hard}}(q) \cup S_{\text{family}}(q)$$

where:
- $S_{\text{hard}}(q)$: devices/equips explicitly parsed from $q$ (e.g., device name mentioned in query text)
- $S_{\text{family}}(q) = \bigcup_{d \in S_{\text{hard}}(q)} \text{Family}(d)$: family expansion for procedure documents only

A document $d$ is **in-scope** if and only if:

$$d \in D_{\text{shared}} \quad \text{OR} \quad \text{device}(d) \in S(q)$$

## 3.4 Document Scope Level

Each document is assigned a **scope_level** determining the filter granularity applied:

| scope_level | Semantics | Filter Applied | Target doc_types |
|------------|-----------|---------------|-----------------|
| `shared` | Cross-equipment document (in $D_{\text{shared}}$) | No filter (always allowed) | SOPs/TS shared across $\geq T$ devices |
| `device` | Device-level procedure | $\text{device\_name} \in S(q)$ | SOP, setup_manual, TS |
| `equip` | Instance-level record | $\text{equip\_id} \in S_{\text{equip}}(q)$ | maintenance logs (myservice, gcb) |

**Assignment rule**: scope_level is determined by doc_type with a shared override:
1. If the document's topic appears in $D_{\text{shared}}$: scope_level = `shared`
2. Else if doc_type $\in$ {myservice, gcb}: scope_level = `equip`
3. Else: scope_level = `device`

## 3.5 Shared Document Classification ($D_{\text{shared}}$)

A topic is classified as **shared** if it appears across $\geq T$ distinct devices (default $T=3$):

$$D_{\text{shared}} = \{d \mid \text{topic}(d) \in \mathcal{T}_{\text{shared}}\}$$
$$\mathcal{T}_{\text{shared}} = \{t \mid |\{e \in \mathcal{E} : t \in \text{Topics}(e)\}| \geq T\}$$

Only procedure document types (SOP, TS) are eligible for shared classification.

## 3.6 Equipment Family Construction

We construct equipment families using a **topic-sharing graph**:
- Nodes: devices $e \in \mathcal{E}$
- Edge weight: weighted Jaccard similarity

$$w(a, b) = \frac{\sum_{t \in T(a) \cap T(b)} \omega(t)}{\sum_{t \in T(a) \cup T(b)} \omega(t)}$$

where $\omega(t) = 1 / \log(1 + |\text{devices}(t)|)$ downweights topics shared across many devices.

Families are formed by connected components at threshold $\tau$ (default $\tau = 0.2$).

**Family expansion applies only to procedure documents** (scope_level = `device`). Logs and instance records (scope_level = `equip`) are never family-expanded to prevent cross-instance data leakage.

## 3.7 Query Scope Determination

| Parser Result | Query Type | Scope Decision |
|--------------|-----------|---------------|
| equip_id extracted | Instance query | Hard(equip): equip-level filter for logs, device filter for procedures |
| device_name only | Device query | Hard(device): device_name filter |
| Neither extracted | Implicit/ambiguous | Router mode (planned) or global fallback |

## 3.8 Scope-Level-Aware Filter Construction

The retrieval filter uses boolean OR (`should`) branches to apply different filter strengths per scope_level:

```
filter = OR(
  shared_docs,                           # scope_level=shared: always included
  device_docs AND device_name ∈ S(q),    # scope_level=device: device filter
  equip_docs AND equip_scope_filter      # scope_level=equip: equip/device filter
)
```

This is implemented in `scope_filter.py:build_scope_filter_by_doc_ids`.

## 3.9 Contamination-Aware Scoring Function (planned extension)

Beyond binary filtering, we define a scoring function that penalizes scope violations in the reranking stage:

$$\text{Score}(d, q) = \text{Base}(d, q) - \lambda(q) \cdot v_{\text{scope}}(d, q)$$

| Term | Definition |
|------|-----------|
| $\text{Base}(d, q)$ | Cross-encoder reranker score (normalized to [0, 1]) |
| $v_{\text{scope}}(d, q)$ | $\mathbb{1}[d \notin D_{\text{shared}} \wedge \text{device}(d) \notin S(q)]$ |
| $\lambda(q)$ | Penalty strength, adapted by router confidence |

**Adaptive $\lambda(q)$** (for router-mode queries):

$$\lambda(q) = \lambda_{\max} \cdot \sigma(\alpha \cdot \text{confidence}(q) - \beta)$$

where $\text{confidence}(q) = \text{score}_{\text{top1}} - \text{score}_{\text{top2}}$ from the router.

For parser-confirmed queries (Hard mode), $\lambda(q) = \lambda_{\max}$ (equivalent to binary filter).

> **Current status**: The contamination-aware scoring function (systems P6, P7) is **planned-not-reported** pending router implementation. The current evaluation covers systems B0–B4 and P1.

## 3.10 Matryoshka Router (planned extension)

For queries where the parser cannot extract device information, a low-dimensional router is planned:
- Device prototype embeddings at reduced dimensionality (64/128/256d)
- Top-$M$ device candidates selected by cosine similarity
- Family expansion applied to router output

> **Current status**: The Matryoshka router (systems P2–P4) is **planned-not-reported**. The current embedding stack (SentenceTransformer-based) supports `truncate_dim` but models have not been verified for Matryoshka Representation Learning (MRL) training (Kusupati et al., NeurIPS 2022). Fair evaluation requires MRL-trained models with full-dimension baselines.

## 3.11 System Matrix

| ID | Scope Decision | Retrieval | Rerank | Status |
|----|---------------|-----------|--------|--------|
| B0 | Global | BM25 | No | Reported |
| B1 | Global | Dense | No | Reported |
| B2 | Global | Hybrid+RRF | No | Reported |
| B3 | Global | Hybrid+RRF | Yes | Reported |
| B4 | Auto-parse Hard | Hybrid+RRF (filtered) | Yes | Reported |
| P1 | Hard + Shared + scope_level | Hybrid+RRF (filtered) | Yes | Reported |
| P2 | Matryoshka Router Top-M | Hybrid+RRF (filtered) | Yes | Planned |
| P3 | Router + Family | Hybrid+RRF | Yes | Planned |
| P4 | Router + Family + Shared | Hybrid+RRF | Yes | Planned |
| P6 | P4 + scoring (fixed $\lambda$) | Hybrid+RRF | Yes | Planned |
| P7 | P4 + scoring (adaptive $\lambda(q)$) | Hybrid+RRF | Yes | Planned |

---

# 4. Experimental Setup

## 4.1 Corpus

- 578 documents across multiple equipment types
- Document types: SOP, setup manual, troubleshooting guide, maintenance logs (myservice, gcb)
- ES index: `rag_chunks_dev_v2`

## 4.2 Evaluation Set

- 472 total queries in `query_gold_master_v0_5.jsonl`
- Split: dev/test by stratified assignment
- Scope observability labels: `explicit_device`, `explicit_equip`, `implicit`, `ambiguous`
- Test slices: explicit_device (22), implicit (21), explicit_equip (8)
- Dev slice: ambiguous (80, no gold docs — descriptive only)

## 4.3 Systems Evaluated

- B0–B4: Baseline progression (BM25 → Dense → Hybrid → Reranker → Hard filter)
- P1: Proposed scope-level-aware filter (Hard + Shared + scope_level routing)
- P2–P7: Planned extensions (not reported)

## 4.4 Statistical Protocol

- Bootstrap CI: 2000 samples, seed 20260305
- McNemar test with continuity correction for binary CE@k
- Holm-Bonferroni correction for multiple comparisons
- All tuning on dev split only; test split results reported once

---

# Algorithm 1: Hierarchy-Aware Scope Routing

```
Input: query q, corpus D, policy artifacts (D_shared, family_map, doc_scope)
Output: ranked documents R, scope metadata

1. parsed ← AUTO_PARSE(q)                    // Extract device_name, equip_id
2. if parsed.equip_id:
3.     S_device ← {parsed.device_name}
4.     S_equip ← {parsed.equip_id}
5.     mode ← HARD_EQUIP
6. elif parsed.device_name:
7.     S_device ← {parsed.device_name}
8.     S_equip ← ∅
9.     mode ← HARD_DEVICE
10. else:
11.    S_device ← ∅; S_equip ← ∅
12.    mode ← GLOBAL_FALLBACK              // Router planned (P2-P4)

13. filter ← BUILD_SCOPE_FILTER(S_device, S_equip, D_shared, doc_scope)
14.     // OR(shared_docs, device_match, equip_match)

15. candidates ← HYBRID_RRF_RETRIEVE(q, filter, top_n=60)
16. R ← CROSS_ENCODER_RERANK(q, candidates, top_k=K)

17. return R, mode, S_device, S_equip
```

---

# Symbol Table

| Symbol | Definition |
|--------|-----------|
| $q$ | Input query |
| $d$ | Document |
| $\mathcal{D}$ | Full document corpus |
| $\mathcal{E}$ | Set of equipment devices |
| $S(q)$ | Allowed scope for query $q$ |
| $S_{\text{hard}}(q)$ | Parser-extracted scope (device/equip) |
| $S_{\text{family}}(q)$ | Family-expanded scope |
| $D_{\text{shared}}$ | Set of shared (cross-equipment) documents |
| $\mathcal{T}_{\text{shared}}$ | Set of shared topics |
| $T$ | Shared topic threshold (default: 3) |
| $\tau$ | Family graph edge threshold (default: 0.2) |
| $M$ | Top-M device candidates from router |
| $K$ | Top-K retrieval cutoff |
| $\text{Family}(d)$ | Equipment family containing device $d$ |
| $v_{\text{scope}}(d, q)$ | Scope violation indicator (binary) |
| $\lambda(q)$ | Query-adaptive penalty strength |
| $\omega(t)$ | Topic weight: $1/\log(1 + |\text{devices}(t)|)$ |

---

# Metric Table

| Metric | Formula | Role |
|--------|---------|------|
| Raw Cont@k | $(1/k) \sum_{i=1}^{k} \mathbb{1}[\text{device}(d_i) \notin S(q)]$ | Strict contamination (includes shared) |
| Adj Cont@k | $(1/k) \sum_{i=1}^{k} \mathbb{1}[d_i \notin D_{\text{shared}} \wedge \text{device}(d_i) \notin S(q)]$ | **Primary claim metric** |
| Shared@k | $(1/k) \sum_{i=1}^{k} \mathbb{1}[d_i \in D_{\text{shared}}]$ | Shared doc proportion |
| CE@k | $\mathbb{1}[\exists i \leq k : d_i \text{ is OOS and not shared}]$ | Binary contamination existence |
| Hit@k | $\mathbb{1}[\exists i \leq k : d_i \in \text{Gold}(q)]$ | Recall |
| MRR | $1 / \text{rank}(\text{first gold doc})$ | Ranking quality |
| ScopeAccuracy@M | $\mathbb{1}[\text{gold device} \in \text{Router Top-M}]$ | Router quality (P2-P4 only) |
