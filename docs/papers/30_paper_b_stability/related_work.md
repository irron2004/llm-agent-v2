# Paper B Related Work (Stability / Repeatability)

## Purpose
- Organize literature for Paper B: retrieval stability under repeated and semantically equivalent queries.
- Target argument:
  - define stability as first-class metric
  - identify instability drivers
  - reduce volatility via deterministic control and consensus/fusion methods

## Cluster 1: Ranking Stability Metrics
- Why needed:
  - justify `Jaccard@k` and add rank-sensitive complements
- Core refs:
  - `webber2010rbo` (RBO for top-weighted overlap)
  - `kendall1938tau` (rank correlation classic)

## Cluster 2: Query-Level Stability in Learning to Rank
- Why needed:
  - connect operational stability to ranking learning theory
- Core refs:
  - `liu2008querylevelstability`

## Cluster 3: Query Variation Robustness (Paraphrase / Typo / Perturbation)
- Why needed:
  - motivate why identical intent can still yield unstable top-k
- Core refs:
  - `hagen2024queryvariation`
  - `zhuang2022typorobustness`
  - `typo2024multipositive`
  - `pan2023robustranker`
  - `topic2023adversarialrank`

## Cluster 4: Semantic Equivalence Construction (Paraphrase / STS)
- Why needed:
  - justify paraphrase-group construction and validation protocol
- Core refs:
  - `dolan2005mrpc`
  - `reimers2019sbert`
  - `cer2017semeval`

## Cluster 5: Stability Improvement Strategies (Multi-query / Fusion / Expansion)
- Why needed:
  - support consensus retrieval pipeline in Paper B
- Core refs:
  - `cormack2009rrf`
  - `mao2021gar`
  - `wang2023query2doc`
  - `gao2023hyde`
  - `kuo2025mmlf`
  - `liu2025exp4fuse`
  - `li2019neuralprf`
  - `zukerman2002queryparaphrase`

## Cluster 6: ANN Backend as Instability Driver
- Why needed:
  - retrieval infrastructure can add ranking variance
- Core refs:
  - `ram2009rankann`
  - `malkov2020hnsw`

## Cluster 7: Robust Retrieval Benchmarks / Generalization
- Why needed:
  - avoid single-domain-only framing
- Core refs:
  - `thakur2021beir`

## Optional Cluster: Answer Consistency (if included in B scope)
- Core refs:
  - `wang2022selfconsistency`

## Gap Statement for Paper B
- Prior work studies robustness and ranking quality, but operational stability is often not optimized as a primary objective.
- Paper B gap:
  - metric-level contribution: repeat/paraphrase stability as formal KPI
  - method-level contribution: deterministic controls + consensus retrieval under fixed budget
  - analysis-level contribution: instability driver decomposition (query variation, ANN, corpus changes)

## Related Work Writing Plan (for manuscript)
- RW-1: ranking stability metrics and query-level stability theory
- RW-2: query variation robustness of neural retrieval/ranking
- RW-3: paraphrase equivalence resources and labeling practices
- RW-4: multi-query fusion/expansion methods for stabilization
- RW-5: system-level volatility from ANN and domain shift benchmarks

## Immediate Reading Order (first pass)
1. `webber2010rbo`
2. `cormack2009rrf`
3. `liu2008querylevelstability`
4. `hagen2024queryvariation`
5. `zhuang2022typorobustness`
6. `wang2023query2doc`
7. `kuo2025mmlf`
8. `malkov2020hnsw`
9. `thakur2021beir`
10. `dolan2005mrpc`
