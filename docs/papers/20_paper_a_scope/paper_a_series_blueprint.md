# Paper A / A-1 / A-2 Research Blueprint

## Overview

| Track | Core Nature | Main Contribution | Learning Need |
|---|---|---|---|
| Paper A | Core retrieval paper | Hierarchy + doc_type + shared/family contamination-aware policy | Low |
| Paper A-1 | Learning extension | CHNM-based contamination-aware reranker/retriever | High |
| Paper A-2 | RAG safety extension | Evidence Consistency Gate for safe response control | Medium |

## Paper A

### Research Questions (RQ)

- RQ-A1: Does scope-constrained retrieval using `device_name`/`equip_id`/`doc_type` reduce cross-equipment contamination vs global retrieval?
- RQ-A2: Do shared/family-aware relaxed policies preserve recall better than hard filtering while keeping contamination controlled?
- RQ-A3: Is doc-type-aware scope (`procedure -> device`, `log/history -> equip`) better than a single-scope policy?
- RQ-A4: Can Matryoshka router candidate selection reduce retrieval cost with minimal quality loss?

### Hypotheses (H)

- H-A1: Hierarchy-aware scope policy significantly lowers `Contamination@k` vs global retrieval.
- H-A2: Shared/family-aware policy keeps or improves `Hit@k`/`MRR` vs hard filter with limited contamination increase.
- H-A3: Doc-type-aware scope has better contamination-recall trade-off than device-only scope.
- H-A4: Matryoshka router reduces latency/candidate space with minimal quality degradation.

### Core Experiment Table

| ID | Method | Description |
|---|---|---|
| B0 | BM25 | Global retrieval baseline |
| B1 | Dense | Global retrieval baseline |
| B2 | Hybrid+RRF | Global retrieval baseline |
| B3 | Hybrid+RRF+Rerank | Strong baseline |
| B4 | Hard Device Filter | Auto-parse device-only scope |
| B5 | Hard Device+Equip Filter | Stronger filter when equip exists |
| P1 | Doc-type-aware Scope | Split policy by procedure vs log/history |
| P2 | Scope + Shared | Shared allowance |
| P3 | Scope + Family | Family expansion |
| P4 | Scope + Shared + Family | Recommended core policy |
| P5 | P4 + Matryoshka Router | Efficiency extension |

### Dataset Requirement

- Required:
  - `query_gold_master.jsonl`
  - `document_scope_table.csv` (or `.parquet`)
  - `device_catalog.csv`
  - `equip_catalog.csv`
  - `doc_type_map.csv`
- Strongly recommended:
  - `shared_doc_gold.csv`
  - `device_family_gold.csv`
  - `intent_doctype_labels.jsonl`
- Optional:
  - device prototype text/index for router
  - Matryoshka router candidate labels

### Completion Criteria

- Contamination metric definition finalized (`Raw/Adjusted/Shared` + `CE@k`).
- Shared/family policy finalized and reproducible.
- Explicit / Implicit / Ambiguous / Equip-centric evaluations completed.
- Clear contamination reduction with recall retention vs baselines.

## Paper A-1 (CHNM Extension)

### Research Questions (RQ)

- RQ-A1-1: Can CHNM hard negatives reduce contamination more than policy-only Paper A best model?
- RQ-A1-2: Can metadata-based weak supervision provide effective training signal without heavy manual labeling?
- RQ-A1-3: Do negative types (cross-device, wrong-equip, wrong-doc-type, family-confusable) yield different gains?

### Hypotheses (H)

- H-A1-1: CHNM-trained reranker reduces cross-device contamination more than generic reranker.
- H-A1-2: Adding wrong-equip negatives improves equip-centric performance.
- H-A1-3: Multi-source negative mix is more stable than single-type negative mining.

### Core Experiment Table

| ID | Method | Description |
|---|---|---|
| B0 | Paper A best model | Best policy-only baseline |
| B1 | Generic cross-encoder reranker | Relevance-only trained baseline |
| P1 | CHNM reranker (cross-device) | Cross-device hard negatives only |
| P2 | CHNM reranker (+ wrong-equip) | Equip negatives added |
| P3 | CHNM reranker (+ wrong-doc-type) | Doc-type negatives added |
| P4 | CHNM reranker (all negatives) | Full negative mix |
| P5 | CHNM dense retriever FT | Retriever fine-tuning variant |

### Dataset Requirement

- Required:
  - All Paper A datasets
  - `reranker_pairs.jsonl`
- Recommended:
  - `router_train.jsonl`
  - retrieval top-k logs
  - negative-type annotations (auto + partial human verified)
- `reranker_pairs` key fields:
  - `q_id`, `question`, `doc_id`
  - `label_relevance`, `label_scope_violation`
  - `negative_type`
  - `doc_type_norm`, `device_name_norm`, `equip_id_norm`

### Completion Criteria

- Automated hard-negative pipeline built and reproducible.
- Contamination-aware reranker/retriever training completed.
- Additional contamination reduction vs Paper A best model demonstrated.
- Per-negative-type ablation analysis reported.

## Paper A-2 (Evidence Consistency Gate Extension)

### Research Questions (RQ)

- RQ-A2-1: Can evidence consistency signals predict citation contamination risk?
- RQ-A2-2: Can clarification/re-retrieval/abstention reduce contamination with acceptable coverage cost?
- RQ-A2-3: Does adding gate policy on top of Paper A improve end-to-end safety?

### Hypotheses (H)

- H-A2-1: Top-k metadata inconsistency is positively correlated with citation contamination.
- H-A2-2: Selective response policy lowers unsafe answer rate on high-risk queries.
- H-A2-3: Paper A + gate policy yields safer grounded answers than retrieval-only improvements.

### Core Experiment Table

| ID | Method | Description |
|---|---|---|
| B0 | Vanilla RAG | Global retrieval + generation |
| B1 | Paper A retrieval + RAG | Retrieval improved only |
| P1 | B1 + Clarification Gate | Ask device/equip clarification when risky |
| P2 | B1 + Re-retrieval Gate | Retry with stricter/alternative scope |
| P3 | B1 + Abstention Gate | Defer unsafe answers |
| P4 | B1 + Hybrid Gate | Clarification + re-retrieval + abstention |

### Dataset Requirement

- Required:
  - Paper A retrieval result logs
  - final answer + citation logs
  - `query_gold_master.jsonl`
- Recommended:
  - human answer audit set
  - risk-labeled subset
  - clarification resolution logs
- Additional labels:
  - `citation_is_in_scope`
  - `answer_supported_by_gold`
  - `needs_clarification`
  - `safe_to_answer`

### Completion Criteria

- Evidence consistency score/policy definition fixed.
- High-risk detection (rule/classifier) implemented.
- Safety vs coverage trade-off reported.
- Citation contamination and unsafe answer rate reduced.

## Recommended Execution Order

1. Paper A (core retrieval story)
2. Paper A-1 (learning-strengthened algorithmic extension)
3. Paper A-2 (safe generation extension)
