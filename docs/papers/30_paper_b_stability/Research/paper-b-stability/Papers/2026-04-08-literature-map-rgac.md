---
type: paper-note
title: Literature Map for Stability-Aware Retrieval and RGAC
project: paper-b-stability
tags:
  - papers/literature-map
  - paper-b
  - stability
status: active
date: 2026-04-08
---

# Literature Map for Paper B (RGAC Direction)

## Scope

Target question: how to position a decision-oriented stability method for retrieval (risk-gated adaptive consensus) against prior work in RAG stability, retrieval fusion, and risk-aware IR control.

## Search Axes

1. RAG stability and reproducibility
2. Rank fusion and consensus retrieval
3. Query performance prediction and risk-sensitive selection
4. Query perturbation/paraphrase robustness

## Core Papers (high relevance)

### A. RAG Stability / Reproducibility

1. **On The Reproducibility Limitations of RAG Systems** (Wang et al., 2025)
   - URL: https://arxiv.org/abs/2509.18869
   - Relevance: introduces ReproRAG and metrics for retrieval reproducibility (Exact Match, Jaccard, Kendall-style consistency).

2. **RAGGED: Towards Informed Design of Scalable and Stable RAG Systems** (Hsia et al., ICML 2025)
   - URL: https://openreview.net/forum?id=4ufjBV6S4I
   - Relevance: shows stability depends strongly on reader noise robustness and retrieval-depth behavior.

3. **Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation** (Zhang et al., 2026)
   - URL: https://arxiv.org/abs/2601.02993
   - Relevance: directly relevant to order/permutation sensitivity; complementary to Top-k stability claims.

### B. Fusion / Consensus Foundations

4. **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods** (Cormack et al., SIGIR 2009)
   - DBLP: https://dblp.org/rec/conf/sigir/CormackCB09
   - Relevance: foundational RRF reference for consensus-style retrieval.

5. **An Analysis of Fusion Functions for Hybrid Retrieval** (Bruch et al., TOIS 2024)
   - DOI: 10.1145/3596512
   - Relevance: careful empirical analysis of convex fusion vs RRF, parameter sensitivity, and sample efficiency.

### C. Risk-Aware Selection / QPP

6. **Query Performance Prediction for Neural IR: Are We There Yet?** (Faggioli & Formal, 2023)
   - URL: https://arxiv.org/abs/2302.09947
   - Relevance: survey-style anchor for neural QPP capabilities and limits.

7. **Selective Query Processing: A Risk-Sensitive Selection of Search Configurations** (Mothe & Ullah, TOIS 2024)
   - DOI: 10.1145/3608474
   - Relevance: strong conceptual precedent for per-query risk-sensitive system configuration.

8. **Two-stage Risk Control with Application to Ranked Retrieval** (Xu et al., IJCAI 2025)
   - URL: https://www.ijcai.org/proceedings/2025/1012.pdf
   - Relevance: modern risk-control framing directly aligned with gate-style retrieval policies.

### D. Perturbation / Paraphrase Robustness (adjacent but useful)

9. **Large Scale Question Paraphrase Retrieval with Smoothed Deep Metric Learning** (Bonadiman et al., 2019)
   - URL: https://arxiv.org/abs/1905.12786
   - Relevance: task-level paraphrase retrieval and equivalence mapping perspective.

10. **Ontology-Guided Query Expansion for Biomedical Document Retrieval using LLMs** (Al Nazi et al., 2025)
   - URL: https://arxiv.org/abs/2508.11784
   - Relevance: explicitly reports robustness under query perturbation settings.

## Positioning Implications for Paper B

1. Existing RAG stability work mostly diagnoses instability; fewer works provide explicit per-query decision policies under latency budget.
2. Fusion literature is strong, but usually not coupled with query-level instability risk estimation.
3. Risk-sensitive IR selection exists, but direct integration with modern RAG-style stability metrics (Repeat/Paraphrase Jaccard) is underexplored.

## Candidate Claim Framing

- **Claim A**: deterministic protocol addresses procedural instability (T1/T3), but not semantic instability (T2).
- **Claim B**: adaptive risk-gated consensus improves stability-cost tradeoff compared with fixed-budget consensus.
- **Claim C**: margin-only signals are insufficient; hybrid risk features better guide stabilization actions.

## Verification Notes

- Metadata above was collected from arXiv/OpenReview/DBLP/Crossref endpoints.
- Some additional candidates from rate-limited sources should be re-checked before camera-ready bibliography export.
