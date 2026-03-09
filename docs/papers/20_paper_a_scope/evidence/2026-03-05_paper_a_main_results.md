# Paper A: Main Experimental Results

> Date: 2026-03-05
> Status: Primary evaluation — test split only

---

## Main Results Tables

### Table 1: Main Results — test_explicit_device (n=22)

| System | Scope | Retrieval | Rerank | adj_cont@5 | raw_cont@5 | shared@5 | ce@5 | hit@5 | mrr |
|--------|-------|-----------|--------|-----------|-----------|---------|------|-------|-----|
| B0 | Global | BM25 | No | 0.282 | 0.309 | 0.118 | 0.727 | 0.364 | 0.274 |
| B1 | Global | Dense | No | 0.209 | 0.218 | 0.055 | 0.500 | 0.318 | 0.295 |
| B2 | Global | Hybrid+RRF | No | 0.273 | 0.300 | 0.118 | 0.682 | 0.409 | 0.278 |
| B3 | Global | Hybrid+RRF | Yes | 0.255 | 0.282 | 0.127 | 0.682 | 0.409 | 0.322 |
| B4 | Hard(device) | Hybrid+RRF | Yes | 0.027 | 0.045 | 0.155 | 0.045 | 0.091 | 0.091 |
| P1 | Scope-level | Hybrid+RRF | Yes | 0.000 | 0.264 | 0.882 | 0.000 | 0.136 | 0.136 |

**Notes:**
- adj_cont@5 excludes shared documents from contamination count (primary metric).
- raw_cont@5 counts all out-of-scope docs including shared.
- P1 raw_cont@5 is high because shared docs (scope_level=shared) are included in top-5 but not counted as contamination.
- B4 and P1 ce@5 near zero indicates near-complete elimination of hard contamination events.

---

### Table 2: Cross-Slice Comparison (adj_cont@5 / hit@5)

| System | explicit_device (n=22) | implicit (n=21) | explicit_equip (n=8) |
|--------|----------------------|----------------|---------------------|
| B3 | 0.255 / 0.409 | 0.600 / 0.286 | 0.900 / 0.000 |
| B4 | 0.027 / 0.091 | 0.600 / 0.286 | 0.900 / 0.000 |
| P1 | 0.000 / 0.136 | 0.000 / 0.000 | 0.000 / 0.000 |

**Notes:**
- B4 = B3 on implicit slice: parser fallback to global retrieval, no filter applied.
- P1 on explicit_equip: adj_cont@5 = 0.000 but hit@5 = 0.000 — no gold docs found in top-5 after scope filter.
- explicit_equip slice (n=8) does not meet the >= 5 queries with non-empty gold threshold; interpret with caution.

---

### Table 3: Statistical Tests — test_explicit_device

| Comparison | Metric | Delta | 95% CI | McNemar p | Significant |
|-----------|--------|-------|--------|-----------|-------------|
| B3→B4 | adj_cont@5 | -0.227 | [-0.345, -0.127] | — | Yes (CI) |
| B3→B4 | hit@5 | -0.318 | [-0.545, -0.091] | — | Yes (CI) |
| B3→B4 | mrr | -0.231 | [-0.451, -0.034] | — | Yes (CI) |
| B3→B4 | ce@5 | — | — | 0.001 | Yes (Holm) |
| B4→P1 | adj_cont@5 | -0.027 | [-0.082, 0.000] | — | Marginal |
| B4→P1 | hit@5 | +0.045 | [-0.136, 0.227] | — | No |
| B4→P1 | ce@5 | — | — | 0.500 | No |

**Protocol**: Bootstrap CI with 2000 samples, seed 20260305. McNemar test with continuity correction for binary CE@k. Holm-Bonferroni correction applied across all pairwise comparisons. All significance decisions use alpha=0.05.

---

### Table 4: Hypothesis Verdict Summary (H1–H12)

| Hypothesis | Verdict | Evidence |
|-----------|---------|---------|
| H1: Hard device filter reduces adj_cont@5 vs global retrieval | **Supported** | B4 adj_cont@5 = 0.027 vs B3 = 0.255; delta = -0.227, 95% CI [-0.345, -0.127] excludes zero |
| H2: Hard filter causes recall loss on explicit queries | **Supported** | B4 hit@5 = 0.091 vs B3 = 0.409; delta = -0.318, 95% CI [-0.545, -0.091] excludes zero |
| H3: Shared doc policy recovers recall lost by hard filter | **Inconclusive** | P1 hit@5 = 0.136 > B4 = 0.091 (delta +0.045), but 95% CI [-0.136, 0.227] crosses zero; not significant |
| H4: Shared policy reduces adj_cont@5 vs hard filter alone | **Supported** | P1 adj_cont@5 = 0.000 vs B4 = 0.027; note partially tautological — shared docs reclassified by design |
| H5: Global retrieval has higher adj_cont@5 on implicit than explicit queries | **Supported** | B3 adj_cont@5: implicit = 0.600 > explicit_device = 0.255; large positive difference |
| H6: Hard filter degrades on implicit queries (parser falls back to global) | **Not Supported** | B4 = B3 on implicit slice (0.600 / 0.286); parser fallback means no filter applied — no degradation but no improvement |
| H7: P1 outperforms B4 on explicit_equip queries | **Inconclusive** | All systems hit@5 = 0.000 on explicit_equip (n=8); no gold docs retrieved; slice does not meet reporting threshold |
| H8: Shared doc proportion varies across device families | **Inconclusive** | Ambiguous dev slice has 0 evaluable queries with gold docs; descriptive analysis only, no test-split evidence |
| H9: Contamination increases with corpus imbalance (SUPRA dominance) | **Cannot evaluate** | Requires per-family stratification of B3 per_query results; not yet computed |
| H10: Dense retrieval (B1) has higher adj_cont@5 than BM25 (B0) | **Not Supported** | B1 adj_cont@5 = 0.209 < B0 = 0.282; opposite of expected; BM25 suffers from shared terminology, semantic retrieval not worse |
| H11: Reranking (B3 vs B2) does not reduce adj_cont@5 | **Supported** | B3 adj_cont@5 = 0.255 vs B2 = 0.273; difference -0.018 is negligible; reranker does not learn scope signals |
| H12: Scope policy benefit is stronger for procedure vs troubleshooting queries | **Cannot evaluate** | Requires per-intent stratification of B3 vs P1 per_query results; not yet computed |

---

## Key Findings

- Hard device scope filtering (B4) is highly effective at eliminating cross-equipment contamination (adj_cont@5: 0.255 → 0.027, -91%), but at severe recall cost (hit@5: 0.409 → 0.091, -78%). The contamination-recall trade-off is sharp and statistically robust.

- The scope-level-aware shared doc policy (P1) achieves zero adjusted contamination on the explicit_device slice and provides a small, non-significant recall improvement over hard filtering alone. Shared document reclassification accounts for the full contamination reduction.

- Implicit queries exhibit substantially higher contamination than explicit queries under global retrieval (adj_cont@5: 0.600 vs 0.255), confirming that scope observability is the key determinant of contamination risk. Hard filtering cannot help implicit queries due to parser fallback.

- Reranking (B3 vs B2) provides no contamination reduction. Cross-encoder rerankers optimize relevance within the retrieved candidate set but do not learn to respect equipment boundaries. Explicit scope policy is required.

- Dense retrieval (B1) does not increase contamination relative to BM25 (B0). Both retrieval modes exhibit similar contamination levels (~0.21–0.28), suggesting that cross-device vocabulary overlap is pervasive enough to affect keyword and semantic retrieval equally.

- The explicit_equip slice (n=8) shows zero recall for all systems after scope filtering, indicating that equip-level gold documents are not indexed or not retrieved effectively. This slice is below the minimum reporting threshold and requires corpus investigation before conclusions can be drawn.

- H9 and H12 (corpus imbalance and intent stratification effects) remain unevaluated pending per-query stratification analysis. These represent high-value follow-on experiments.

---

## Limitations

- The test_explicit_device slice (n=22) is small. Bootstrap CIs are wide for hit@5 and mrr, and effect size estimates may shift with additional queries.

- The explicit_equip slice (n=8) does not meet the pre-specified reporting threshold of >= 5 queries with non-empty gold docs, making H7 unevaluable from current data.

- P1's adj_cont@5 = 0.000 result is partially tautological: shared documents are reclassified as in-scope by the metric definition, so perfect adj_cont@5 is achievable by design if all top-5 positions are filled with shared docs. The raw_cont@5 = 0.264 and shared@5 = 0.882 values reveal this structure.

- H6 "Not Supported" does not mean B4 is robust on implicit queries — it means the parser falls back to global retrieval and the filter is never applied, so B4 and B3 are identical. The absence of degradation is a no-op, not a positive result.

- Per-family and per-intent stratification (H9, H12) are not yet computed. These analyses are necessary to evaluate whether corpus imbalance and query intent drive contamination heterogeneity.

---

## 2026-03-09 Results (Post-Alias Fix)

> Date: 2026-03-09
> Status: Updated evaluation with canonical alias normalization and B4.5 system
> Key additions: adj_den@5 and shared_rel@5 metrics for R2 mitigation validation

### Table 5: Baseline Systems — Full Test Set (n=51)

| System | Scope | Retrieval | Rerank | n_ok | mean_raw_cont@5 | mean_adj_cont@5 | mean_adj_den@5 | mean_shared@5 | mean_shared_rel@5 | mean_ce@5 | mean_hit@5 | mean_mrr |
|--------|-------|-----------|--------|------|-----------------|-----------------|----------------|----------------|-------------------|-----------|-----------|---------|
| B0 | Global | BM25 | No | 51 | 0.3961 | 0.3592 | 4.1765 | 0.1647 | 0.0000 | 0.4706 | 0.2549 | 0.1918 |
| B1 | Global | Dense | No | 51 | 0.4667 | 0.4748 | 4.6078 | 0.0784 | 0.0196 | 0.6863 | 0.1765 | 0.1667 |
| B2 | Global | Hybrid+RRF | No | 51 | 0.3961 | 0.3601 | 4.1765 | 0.1647 | 0.0196 | 0.4706 | 0.2941 | 0.1941 |
| B3 | Global | Hybrid+RRF | Yes | 51 | 0.3961 | 0.3592 | 4.1373 | 0.1725 | 0.0196 | 0.4706 | 0.2941 | 0.2239 |

**Notes:**
- Full test set (n=51) spans all scope observability slices: explicit_device (n=22), explicit_equip (n=8), implicit (n=21).
- Baselines show stable contamination across different retrieval modes: BM25 (B0) 0.3592 ≈ Hybrid (B2/B3) 0.3592–0.3601.
- adj_den@5 (adjusted density) ≈ 4.1–4.6, indicating baseline systems retrieve ~4 in-scope docs per query in top-5.

### Table 6: Scoped Systems — Full Test Set (n=51)

| System | Scope | Retrieval | Rerank | n_ok | mean_raw_cont@5 | mean_adj_cont@5 | mean_adj_den@5 | mean_shared@5 | mean_shared_rel@5 | mean_ce@5 | mean_hit@5 | mean_mrr |
|--------|-------|-----------|--------|------|-----------------|-----------------|----------------|----------------|-------------------|-----------|-----------|---------|
| B4 | Hard(device) | Hybrid+RRF | Yes | 51 | 0.3608 | 0.3229 | 4.1373 | 0.1725 | 0.0000 | 0.4118 | 0.2941 | 0.2239 |
| B4.5 | Shared-aware | Dense | Yes | 51 | 0.4863 | 0.0000 | 1.8627 | 0.6275 | 0.0065 | 0.0000 | 0.1765 | 0.1417 |
| P1 | Shared-aware | Hybrid+RRF | Yes | 51 | 0.5451 | 0.0000 | 0.4510 | 0.9098 | 0.0163 | 0.0000 | 0.0588 | 0.0588 |

**Key findings:**
- **B4 (hard device filter)**: Reduces adj_cont@5 from 0.3592 (B3) to 0.3229 (−10.1%) while preserving recall (hit@5 stays 0.2941). Density remains high (adj_den@5=4.1373).
- **B4.5 (shared-aware dense)**: Achieves adj_cont@5 = 0.0000 with moderate recall (hit@5=0.1765), outperforming P1 on recall while using fewer shared docs (shared@5=0.6275).
- **P1 (shared-aware hybrid)**: Also achieves adj_cont@5 = 0.0000, but with the strongest shared dominance (shared@5=0.9098) and the lowest recall (hit@5=0.0588).
- **R2 mitigation interpretation**: `adj_den@5` and `shared_rel@5` now clearly separate “contamination removal” from “result density/utility” trade-offs across B4/B4.5/P1.

### Table 7: Explicit Device Slice — Scoped Systems Only (n=22)

| System | Scope | Retrieval | Rerank | n_ok | mean_raw_cont@5 | mean_adj_cont@5 | mean_adj_den@5 | mean_shared@5 | mean_shared_rel@5 | mean_ce@5 | mean_hit@5 | mean_mrr |
|--------|-------|-----------|--------|------|-----------------|-----------------|----------------|----------------|-------------------|-----------|-----------|---------|
| B4 | Hard(device) | Hybrid+RRF | Yes | 22 | 0.0455 | 0.0455 | 4.3636 | 0.1273 | 0.0000 | 0.0455 | 0.4091 | 0.3220 |
| B4.5 | Shared-aware | Dense | Yes | 22 | 0.1000 | 0.0000 | 3.8636 | 0.2273 | 0.0152 | 0.0000 | 0.4091 | 0.3285 |
| P1 | Shared-aware | Hybrid+RRF | Yes | 22 | 0.2364 | 0.0000 | 0.5909 | 0.8818 | 0.0379 | 0.0000 | 0.1364 | 0.1364 |

**R2 mitigation validation (new metrics):**
- **adj_den@5 (R2 metric)**: B4 and B4.5 both keep high in-scope density (4.36 and 3.86 respectively), while P1 remains sparse (0.59) due to shared-heavy retrieval.
- **shared_rel@5 (R2 metric)**: B4.5 (0.0152) is less shared-dependent than P1 (0.0379), indicating a lower shared-doc reliance at comparable adjusted contamination.
- **B4 vs B4.5**: B4.5 eliminates adjusted contamination (0.0) while matching B4 recall (hit@5 both 0.4091) and improving MRR slightly (0.3285 vs 0.3220).
- **P1 vs B4.5**: Both have adj_cont@5=0.0, but B4.5 is substantially better on utility (hit@5 0.4091 vs 0.1364; mrr 0.3285 vs 0.1364).

### Key Differences from 2026-03-05 Evaluation

1. **Canonical alias normalization**: 2026-03-09 run includes corrected alias handling in scope matching. Previously unreported systems B4.5 and refined P1 now included.

2. **New metrics** (adj_den@5, shared_rel@5): Designed to evaluate R2 mitigation trade-offs between contamination reduction and result sparsity.

3. **Expanded slice analysis**: 2026-03-05 reported only explicit_device (n=22). 2026-03-09 adds full test set (n=51) baselines and scoped systems, enabling cross-slice comparisons.

4. **B4.5 system**: New dense-retrieval shared-aware variant. It reaches zero adjusted contamination while preserving more recall than P1 (hit@5 0.1765 vs 0.0588 on full test set).
