# Paper B Evidence Mapping

## Purpose
- Map Paper B claims to literature anchors and to reproducible experiment outputs.
- Keep aligned with:
  - `paper_b_stability_spec.md`
  - `related_work.md`
  - `references.bib`

## Claim-to-Evidence Map

| Claim ID | Claim (Paper B) | Literature Support | Experiment Evidence |
|---|---|---|---|
| B-C1 | Top-k overlap should be measured with both set overlap and rank-aware metrics. | `webber2010rbo`, `kendall1938tau` | `RepeatJaccard@10`, optional RBO/Kendall in appendix |
| B-C2 | Stability is a query-level reliability property, not only mean accuracy. | `liu2008querylevelstability` | Query-level variance and group-level stability reports |
| B-C3 | Semantically equivalent variations can cause ranking instability. | `hagen2024queryvariation`, `zhuang2022typorobustness`, `topic2023adversarialrank` | T2 paraphrase stability experiments |
| B-C4 | Paraphrase-group construction requires explicit semantic-equivalence protocol. | `dolan2005mrpc`, `reimers2019sbert`, `cer2017semeval` | Paraphrase set generation/validation logs |
| B-C5 | Multi-query expansion + late fusion/consensus can reduce volatility. | `cormack2009rrf`, `mao2021gar`, `wang2023query2doc`, `gao2023hyde`, `kuo2025mmlf`, `liu2025exp4fuse` | Consensus retrieval ablation vs deterministic baseline |
| B-C6 | ANN backend behavior can contribute to ranking instability. | `ram2009rankann`, `malkov2020hnsw` | T4 reindex/rebuild stability deltas |
| B-C7 | Robustness claims should be checked across heterogeneous retrieval settings. | `thakur2021beir` | Cross-config and cross-subset robustness tables |
| B-C8 | If answer consistency is included, consensus selection methods are justified. | `wang2022selfconsistency` | Conclusion consistency metric (optional) |
| B-C9 | Companion-paper references must distinguish executed experiments from design-only follow-up work. | Internal review note (A-2 status audit) | `evidence/2026-03-25_a2_phase2_bayesian_review.md` |
| B-C10 | Bayesian LR with decision-theoretic threshold predicts query-level instability and provides a principled flagging policy. | `gelman2013bda`, `vehtari2017loo`, EPV < 10 justification | Bayesian stability gate experiments (§7.6), PyMC posterior + LOO-CV |
| B-C11 | The stability gate and A-2's evidence gate share a unified decision-theoretic framework composable into a single risk score. | A-2 Bayesian framework design (cited as design-only) | `evidence/2026-03-26_bayesian_framework_review.md`, §8.3 |

## Executed Experiments (2026-03-26)

### S1_direct (Full-scale, 339 groups, 1,356 queries, 3 repeats)
- **Evidence**: `.sisyphus/evidence/paper-b/S1_direct/metrics.json`, `results.jsonl`
- **Script**: `scripts/paper_b/run_paper_b_eval_direct.py`
- RepeatJaccard@10 = 0.9666 [0.963, 0.970]
- ParaphraseJaccard@10 = 0.3943 [0.383, 0.405]
- 91.2% groups unstable (Jaccard < 0.7)
- BoundaryMargin@10 mean=0.000461, median=0.000233

### Bayesian Analysis (Full-scale, N=339)
- **Evidence**: `.sisyphus/evidence/paper-b/bayesian_full/bayesian_analysis.json`
- **Script**: `scripts/paper_b/bayesian_stability_analysis.py`
- **Figures**: `fig4_margin_vs_jaccard.png`, `fig6_bayesian_posterior.png`, `fig7_flag_rate_vs_recall.png`
- β_margin = −0.779 [−1.054, −0.515] (σ=1.0 prior)
- Spearman ρ = 0.174 (p = 0.001), Pearson r = 0.407 (p < 10⁻¹⁴)
- LOO-ELPD = −85.4 ± 11.0
- R̂ = 1.000, ESS > 4,900, no divergences

### Preliminary S0 vs S1 (10 groups, API-based)
- **Evidence**: `.sisyphus/evidence/paper-b/preliminary_comparison.md`
- S0 RepeatJaccard=0.7614, S1 RepeatJaccard=0.9788
- Preliminary only — confirms direction, insufficient for correlation analysis

## Paper B Core Outputs
- Figure B1: stability-aware retrieval architecture
- Figure B2: Stability vs Recall trade-off
- Figure 4: BoundaryMargin vs ParaphraseJaccard scatter (executed)
- Figure 6: Bayesian posterior distributions (executed)
- Figure 7: Flag rate vs recall curve (executed)
- Table 1: S1_direct metrics (executed)
- Table 2: Binned margin → Jaccard (executed)
- Table B3: method ablation (deterministic, consensus, optional regularization)

## Notes
- Maintain Paper B scope boundaries:
  - include: retrieval stability and reproducibility
  - exclude: hierarchy constraints (Paper A), faithfulness validator (Paper D)
- When citing A-2 from Paper B, use `evidence/2026-03-25_a2_phase2_bayesian_review.md` as the status guardrail:
  - Phase 2 = documented simulation
  - Bayesian = design only / not yet executed
