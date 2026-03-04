# Paper C Related Work (Lifecycle Reliability Control)

## Purpose
- Organize literature for Paper C: detect lifecycle regression (drift) and control rollout with testing/rollback policies.
- Target argument:
  - updates can improve average quality while degrading stability/safety
  - drift monitoring + regression tests + policy control reduces operational risk

## Cluster 1: Dataset Shift / Concept Drift Foundations
- Why needed:
  - formalize "performance regression under changing environment"
- Core refs:
  - `gama2014conceptdrift`
  - `quinonero2009datasetshift`

## Cluster 2: Drift Detection Algorithms
- Why needed:
  - justify online detectors and alarming logic
- Core refs:
  - `bifet2007adwin`
  - `bifet2006adwinpreprint`
  - `lu2014driftdetectors`

## Cluster 3: SPC and Quality Engineering View (IE Anchor)
- Why needed:
  - connect drift monitoring to quality/process-control language
- Core refs:
  - `spc2024oodmonitoring`
  - `jin2025processdrift`

## Cluster 4: MLOps / Technical Debt / Production Governance
- Why needed:
  - explain why lifecycle control is a research problem, not only engineering hygiene
- Core refs:
  - `sculley2015technicaldebt`
  - `breck2017mltestscore`

## Cluster 5: ML Regression Testing and CI
- Why needed:
  - formalize update-gate criteria before deployment
- Core refs:
  - `renggli2019easemlci`

## Cluster 6: Testing ML Systems (Oracle / Metamorphic)
- Why needed:
  - RAG regression tests suffer from incomplete oracle coverage
- Core refs:
  - `zhang2020mltestingmapping`
  - `dwarakanath2021metamorphic`

## Cluster 7: Continuous Evaluation for RAG/LLM Apps
- Why needed:
  - define monitoring loop for retrieval+generation quality after updates
- Core refs:
  - `es2023ragas`
  - `trulensragtriad`

## Gap Statement for Paper C
- Prior studies provide drift detectors, CI/testing frameworks, and RAG metrics separately.
- Paper C gap:
  - integrate them into one deployment-control policy for on-prem RAG lifecycle
  - optimize policy by risk-cost objective (update benefit vs regression risk vs rollback cost)
  - provide reproducible versioned evaluation with regression suites

## Related Work Writing Plan (for manuscript)
- RW-1: shift/drift theory and operational implications
- RW-2: detector families and alarm trade-offs
- RW-3: SPC framing for IE-quality perspective
- RW-4: MLOps debt, CI, and ML testing principles
- RW-5: RAG continuous evaluation and lifecycle control gap

## Immediate Reading Order (first pass)
1. `sculley2015technicaldebt`
2. `breck2017mltestscore`
3. `renggli2019easemlci`
4. `gama2014conceptdrift`
5. `bifet2007adwin`
6. `lu2014driftdetectors`
7. `es2023ragas`
8. `trulensragtriad`
