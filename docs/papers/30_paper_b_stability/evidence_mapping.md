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

## Paper B Core Outputs (expected)
- Figure B1: stability-aware retrieval architecture
- Figure B2: Stability vs Recall trade-off
- Table B1: T1/T2/T3/T4 stability metrics
- Table B2: instability driver decomposition
- Table B3: method ablation (deterministic, consensus, optional regularization)

## Notes
- Maintain Paper B scope boundaries:
  - include: retrieval stability and reproducibility
  - exclude: hierarchy constraints (Paper A), faithfulness validator (Paper D)
