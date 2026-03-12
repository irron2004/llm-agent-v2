# Cross-Equipment Contamination Analysis Report
Generated: 2026-03-12 05:04:25

## [OBJECTIVE]
Quantify cross-equipment contamination in the Phase 3 dev corpus retrieval results:
1. Measure Contamination@k for B3 (no filter) vs B4.5/P1 (scope filter) by scope observability
2. Compute delta (contamination reduction) and test statistical significance
3. Determine whether contaminated (cross-device) documents are actually irrelevant

---

## [DATA]
- Source: `.sisyphus/evidence/paper-a/runs/phase3_dev_109/per_query.csv`
- 109 dev queries, 7 systems (B0-B4.5, P1), 763 rows total (602 non-skipped)
- B3 (no filter): 86 non-skipped queries  
- Systems analyzed: B3 (dense, no scope filter), B4.5 (scope filter), P1 (scope filter + rerank)
- Scope observability categories: explicit_device (n=27), explicit_equip (n=15), implicit (n=27), ambiguous (n=17)
- Judgments: `data/paper_a/phase3_all_judgments.json` — 2,077 (query, doc) pairs with relevance 0/1/2
- Doc-device map: `.sisyphus/evidence/paper-a/policy/doc_scope.jsonl` — 508 documents, 27 devices

---

## [FINDING 1] B3 (no filter) has high baseline contamination rate
Across all 86 non-skipped queries, B3 retrieves 62.6% cross-device documents at top-5.

[STAT:mean] B3 raw_cont@5 = 0.626 (62.6% of top-5 docs are from wrong device)
[STAT:ci] 95% CI: [0.524, 0.728]
[STAT:n] n=86 queries

Breakdown by k:
- raw_cont@1 = 0.628  (first retrieved doc is cross-device 63% of the time)
- raw_cont@3 = 0.620
- raw_cont@5 = 0.626
- raw_cont@10 = 0.505

---

## [FINDING 2] Scope filter reduces contamination only for explicit_device and ambiguous queries
B4.5 vs B3 (k=5):

| Observability   | n  | B3    | B4.5  | Delta  | p-value    | Cohen's d |
|-----------------|-----|-------|-------|--------|------------|-----------|
| ALL             | 86  | 0.626 | 0.540 | +0.086 | p=0.019*   | d=+0.258 (small) |
| explicit_device | 27  | 0.519 | 0.356 | +0.163 | p=0.016*   | d=+0.496 (small) |
| explicit_equip  | 15  | 0.893 | 0.920 | -0.027 | p=0.685    | d=-0.107 (negligible) |
| implicit        | 27  | 0.719 | 0.770 | -0.052 | p=0.364    | d=-0.178 (negligible) |
| ambiguous       | 17  | 0.412 | 0.129 | +0.282 | p=0.005**  | d=+0.782 (medium) |

[STAT:effect_size] Largest reduction: ambiguous queries, Cohen's d=0.782 (medium effect)
[STAT:p_value] Significant reductions: ALL (p=0.019), explicit_device (p=0.016), ambiguous (p=0.005)
[STAT:n] Paired t-tests on 86/27/15/27/17 matched query pairs

P1 vs B3: Smaller overall reduction (delta=+0.047, p=0.211, not significant for ALL queries).
P1 shows same reduction as B4.5 only for ambiguous queries (identical because same filter applied).

---

## [FINDING 3] Scope filter INCREASES contamination for explicit_equip and implicit queries
B4.5 shows HIGHER contamination than B3 for:
- explicit_equip: B3=0.893, B4.5=0.920 (+0.027 worse, not significant p=0.685)
- implicit: B3=0.719, B4.5=0.770 (+0.051 worse, not significant p=0.364)

[STAT:p_value] Neither increase is statistically significant (p>0.05)
[STAT:n] explicit_equip: n=15; implicit: n=27

---

## [FINDING 4] Cross-device documents are 3.3x more likely to be irrelevant than same-device documents
From 2,077 judged (query, doc) pairs:
- Cross-device docs: 85.3% irrelevant (rel=0)  vs  14.7% relevant (rel≥1)
- Same-device docs:  63.6% irrelevant (rel=0)  vs  36.4% relevant (rel≥1)

[STAT:effect_size] Odds Ratio = 3.31 (cross-device irrelevance vs same-device)
[STAT:ci] 95% CI: [2.61, 4.20]
[STAT:p_value] Chi-squared test: chi2=103.44, p<0.001
[STAT:n] n_cross=1,635, n_same=442

Relative Risk: RR=1.34 — cross-device docs are 34% more likely to be irrelevant.
Absolute risk difference: +21.7 percentage points.

Cross-device irrelevance rate by observability:
- explicit_device: 86.4% irrelevant (n=369)
- explicit_equip:  86.5% irrelevant (n=452)
- implicit:        86.1% irrelevant (n=469)
- ambiguous:       81.2% irrelevant (n=345) — slightly lower, suggesting shared knowledge

---

## [FINDING 5] A meaningful minority of cross-device docs (14.7%) are partially relevant
Cross-device docs are not purely noise:
- rel=0 (irrelevant):  85.3%
- rel=1 (partial):     13.8%
- rel=2 (highly rel):   1.0%

Mean relevance: 0.157 (cross-device) vs 0.450 (same-device)

[STAT:n] 1,635 cross-device judged pairs
This explains why scope filter does not fully eliminate contamination signal and why
some contamination persists even in B4.5/P1 — similar equipment shares some procedures.

---

## [LIMITATION]
1. Sample size: Only 86 non-skipped queries in B3; subgroup analyses (n=15-27) have low power.
2. Judgment coverage: Judgments cover 353 unique documents (69% of 508 in corpus). Unjudged docs treated as unknown.
3. Family device matching: implicit/ambiguous queries have family_devices=nan in CSV, so per-row cross-device classification was not possible for those groups; analysis uses pre-computed raw_cont@k from harness.
4. No-gold queries: 23 queries were skipped (no gold document found) — these are excluded and may differ systematically.
5. Judgment quality: LLM-generated relevance judgments (0/1/2 scale); inter-annotator agreement not reported.
6. Causality: We show correlation between contamination and irrelevance; not that contamination *causes* quality degradation (no downstream generation QA metric measured here).

---

## Figures
- `.omc/scientist/figures/contamination_analysis.png` — 3-panel: contamination by system/obs, delta, relevance stacked bar
- `.omc/scientist/figures/contamination_by_k_and_obs.png` — contamination@k curves and observability breakdown
