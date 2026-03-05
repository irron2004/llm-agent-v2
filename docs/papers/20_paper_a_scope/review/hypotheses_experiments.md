# Paper A: Falsifiable Hypotheses and Experiment Matrix

> Version: 1.0
> Date: 2026-03-05

---

## H1: Hard device filter reduces contamination vs global retrieval

- **Dataset slice**: `split=test`, `scope_observability=explicit_device`
- **Systems compared**: B3 vs B4
- **Primary metric**: adj_cont@5
- **Secondary metrics**: hit@5, mrr, ce@5
- **Expected outcome**: B4 adj_cont@5 < B3 adj_cont@5 (filter removes OOS docs)
- **Failure implies**: Auto-parse device extraction is unreliable or corpus has minimal cross-device overlap
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B3,B4 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability explicit_device \
  --out-dir .sisyphus/evidence/paper-a/runs/H1_explicit_device
```

---

## H2: Hard filter causes recall loss on explicit queries

- **Dataset slice**: `split=test`, `scope_observability=explicit_device`
- **Systems compared**: B3 vs B4
- **Primary metric**: hit@5
- **Secondary metrics**: mrr
- **Expected outcome**: B4 hit@5 <= B3 hit@5 (filter may exclude relevant docs)
- **Failure implies**: Parser is accurate enough that filter doesn't harm recall — strengthens B4 as production baseline
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B3,B4 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability explicit_device \
  --out-dir .sisyphus/evidence/paper-a/runs/H2_recall_loss
```

---

## H3: Shared doc policy recovers recall lost by hard filter

- **Dataset slice**: `split=test`, `scope_observability=explicit_device`
- **Systems compared**: B4 vs P1
- **Primary metric**: hit@5
- **Secondary metrics**: adj_cont@5, shared@5
- **Expected outcome**: P1 hit@5 >= B4 hit@5 (shared docs provide additional relevant results)
- **Failure implies**: Shared docs are not relevant to explicit-device queries — shared policy contributes only to contamination accounting, not recall
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B4,P1 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability explicit_device \
  --out-dir .sisyphus/evidence/paper-a/runs/H3_shared_recall
```

---

## H4: Shared policy reduces adjusted contamination vs hard filter alone

- **Dataset slice**: `split=test`, `scope_observability=explicit_device`
- **Systems compared**: B4 vs P1
- **Primary metric**: adj_cont@5
- **Secondary metrics**: raw_cont@5, shared@5
- **Expected outcome**: P1 adj_cont@5 <= B4 adj_cont@5 (shared docs reclassified as in-scope)
- **Failure implies**: Shared classification threshold T is too permissive, or shared docs genuinely belong to single devices
- **Command**: Same as H3 (shared run)

---

## H5: Global retrieval has higher contamination on implicit queries than explicit queries

- **Dataset slice**: `split=test`, `scope_observability=explicit_device` AND `scope_observability=implicit`
- **Systems compared**: B3 (cross-slice comparison)
- **Primary metric**: adj_cont@5
- **Expected outcome**: B3 adj_cont@5 (implicit) > B3 adj_cont@5 (explicit_device)
- **Failure implies**: Implicit queries are not harder for global retrieval — weakens the motivation for router
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B3 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability explicit_device \
  --out-dir .sisyphus/evidence/paper-a/runs/H5_explicit

python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B3 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability implicit \
  --out-dir .sisyphus/evidence/paper-a/runs/H5_implicit
```

---

## H6: Hard filter degrades on implicit queries (parser cannot extract device)

- **Dataset slice**: `split=test`, `scope_observability=implicit`
- **Systems compared**: B3 vs B4
- **Primary metric**: hit@5
- **Secondary metrics**: adj_cont@5
- **Expected outcome**: B4 hit@5 << B3 hit@5 on implicit (parser fails → no filter → fallback to global)
- **Failure implies**: Parser still finds device hints in "implicit" queries — reevaluate scope_observability labeling
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B3,B4 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability implicit \
  --out-dir .sisyphus/evidence/paper-a/runs/H6_implicit_degradation
```

---

## H7: Scope-level-aware filter (P1) outperforms flat device filter (B4) on equip queries

- **Dataset slice**: `split=test`, `scope_observability=explicit_equip`
- **Systems compared**: B4 vs P1
- **Primary metric**: adj_cont@5
- **Secondary metrics**: hit@5
- **Expected outcome**: P1 adj_cont@5 < B4 adj_cont@5 (equip-level filtering removes more OOS docs)
- **Failure implies**: Equip-level docs (myservice/gcb) have too little contamination to detect, or equip metadata is sparse
- **Condition**: Only report if explicit_equip test slice has >= 5 queries with non-empty gold
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B4,P1 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability explicit_equip \
  --out-dir .sisyphus/evidence/paper-a/runs/H7_equip_scope
```

---

## H8: Shared doc proportion varies across device families

- **Dataset slice**: `split=dev`, `scope_observability=all` (corpus-level analysis)
- **Systems compared**: N/A (descriptive analysis)
- **Primary metric**: shared@5 by device family
- **Expected outcome**: SUPRA family (largest, 56.9% of corpus) has lower shared@5 than smaller families
- **Failure implies**: Shared doc distribution is uniform — challenges the "family structure matters" narrative
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B3 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split dev --scope-observability all \
  --out-dir .sisyphus/evidence/paper-a/runs/H8_shared_distribution
```

---

## H9: Contamination increases with corpus imbalance (SUPRA dominance)

- **Dataset slice**: `split=test`, `scope_observability=explicit_device`
- **Systems compared**: B3 (stratified by target device family)
- **Primary metric**: adj_cont@5
- **Expected outcome**: Queries targeting small families (OMNIS, GENEVA) have higher adj_cont@5 than SUPRA queries
- **Failure implies**: Contamination is not driven by corpus imbalance — reconsider the problem motivation
- **Command**: Same as H1 B3 run, then post-process per_query.csv to stratify by target device family

---

## H10: Dense retrieval (B1) has higher contamination than BM25 (B0)

- **Dataset slice**: `split=test`, `scope_observability=explicit_device`
- **Systems compared**: B0 vs B1
- **Primary metric**: adj_cont@5
- **Expected outcome**: B1 adj_cont@5 > B0 adj_cont@5 (semantic similarity crosses device boundaries)
- **Failure implies**: BM25 keyword matching also suffers from cross-device contamination (shared terminology)
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B0,B1 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability explicit_device \
  --out-dir .sisyphus/evidence/paper-a/runs/H10_bm25_vs_dense
```

---

## H11: Reranking (B3 vs B2) does not reduce contamination

- **Dataset slice**: `split=test`, `scope_observability=explicit_device`
- **Systems compared**: B2 vs B3
- **Primary metric**: adj_cont@5
- **Expected outcome**: B3 adj_cont@5 ≈ B2 adj_cont@5 (reranker optimizes relevance, not scope)
- **Failure implies**: Cross-encoder implicitly learns scope signals — weakens the case for explicit scope policy
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B2,B3 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability explicit_device \
  --out-dir .sisyphus/evidence/paper-a/runs/H11_rerank_contamination
```

---

## H12: Scope policy benefit is stronger for procedure queries than troubleshooting

- **Dataset slice**: `split=test`, `scope_observability=explicit_device`, stratified by `intent_primary`
- **Systems compared**: B3 vs P1
- **Primary metric**: adj_cont@5
- **Expected outcome**: delta(adj_cont@5) is larger for `intent_primary=procedure` than `troubleshooting`
- **Failure implies**: Contamination is equally distributed across intent types — scope policy benefit is uniform
- **Command**:
```bash
python scripts/paper_a/evaluate_paper_a_master.py \
  --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
  --systems B3,P1 \
  --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt \
  --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl \
  --family-map .sisyphus/evidence/paper-a/policy/family_map.json \
  --split test --scope-observability explicit_device \
  --out-dir .sisyphus/evidence/paper-a/runs/H12_intent_stratified
```

---

## Summary: Hypothesis → Evidence File Mapping

| Hypothesis | Run ID | Key Outputs |
|-----------|--------|-------------|
| H1, H2 | H1_explicit_device | per_query.csv, bootstrap_ci.json |
| H3, H4 | H3_shared_recall | per_query.csv, bootstrap_ci.json |
| H5 | H5_explicit, H5_implicit | per_query.csv (cross-slice comparison) |
| H6 | H6_implicit_degradation | per_query.csv |
| H7 | H7_equip_scope | per_query.csv (conditional on slice size) |
| H8 | H8_shared_distribution | per_query.csv (post-process by family) |
| H9 | H1_explicit_device (B3) | per_query.csv (stratify by target family) |
| H10 | H10_bm25_vs_dense | per_query.csv, bootstrap_ci.json |
| H11 | H11_rerank_contamination | per_query.csv |
| H12 | H12_intent_stratified | per_query.csv (stratify by intent_primary) |
