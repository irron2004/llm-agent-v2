# Paper C Evidence Mapping

## Purpose
- Connect Paper C claims to literature and to deploy-time validation artifacts.
- Keep aligned with:
  - `related_work.md`
  - `references.bib`
  - lifecycle experiment protocol and policy reports

## Claim-to-Evidence Map

| Claim ID | Claim (Paper C) | Literature Support | Experiment Evidence |
|---|---|---|---|
| C-C1 | Operational performance regression under distribution change is expected and must be monitored. | `gama2014conceptdrift`, `quinonero2009datasetshift` | Versioned evaluation (`v1->v2->v3`) with metric deltas |
| C-C2 | Online drift detection can trigger lifecycle control actions. | `bifet2007adwin`, `bifet2006adwinpreprint`, `lu2014driftdetectors` | Drift alarm precision/recall, false alarm/miss analysis |
| C-C3 | SPC framing is appropriate for model/data monitoring in high-stakes operations. | `spc2024oodmonitoring`, `jin2025processdrift` | SPC chart or threshold policy comparison results |
| C-C4 | Lifecycle control is a technical debt and governance problem, not just one-time tuning. | `sculley2015technicaldebt`, `breck2017mltestscore` | Governance checklist and deployment gate outcomes |
| C-C5 | Update approval must include statistical regression testing with cost constraints. | `renggli2019easemlci` | Regression suite acceptance tests + labeling budget tracking |
| C-C6 | ML system testing needs explicit oracle strategy; pure end-metric checks are insufficient. | `zhang2020mltestingmapping`, `dwarakanath2021metamorphic` | Rule-based + golden-set + metamorphic checks ablation |
| C-C7 | RAG-specific continuous evaluation should include retrieval and generation quality dimensions. | `es2023ragas`, `trulensragtriad` | Continuous evaluation loop with groundedness/context/answer tracking |

## Policy Evaluation Targets
- Policy P1: Always update
- Policy P2: Scheduled update
- Policy P3: Drift-triggered update/rollback

Compare by:
- expected risk
- operating cost
- regression detection rate
- rollback success rate
- latency impact

## Expected Paper C Outputs
- Figure C1: lifecycle control loop (monitor -> detect -> test -> deploy/rollback)
- Figure C2: version-wise drift trend and alarm timeline
- Table C1: policy comparison (risk/cost/performance)
- Table C2: detector comparison (false alarm, miss rate, delay)

## Notes
- Before camera-ready, fill exact author metadata for placeholder entries in `references.bib`.
