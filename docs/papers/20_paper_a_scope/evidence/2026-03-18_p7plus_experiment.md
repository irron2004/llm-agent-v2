# P7+ Experiment (2026-03-18)

Date: 2026-03-18
Status: generated from `scripts/paper_a/run_masked_p6p7_experiment.py`

## Inputs

- Base per-query conditions: `data/paper_a/masked_hybrid_results.json`
- Doc scope: `.sisyphus/evidence/paper-a/policy/doc_scope.jsonl`
- Shared doc ids: `.sisyphus/evidence/paper-a/policy/shared_doc_ids.txt`

## Command

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/run_masked_p6p7_experiment.py
```

## Overall Results (n=578, top-10)

- `B3_masked`: cont@10 `0.584`, gold_strict `351/578`, gold_loose `380/578`
- `P6_masked`: cont@10 `0.649`, gold_strict `351/578`, gold_loose `380/578`
- `P7_masked`: cont@10 `0.649`, gold_strict `351/578`, gold_loose `380/578`
- `P7plus_masked`: cont@10 `0.515`, gold_strict `506/578`, gold_loose `518/578`
- `B4_masked`: cont@10 `0.001`, gold_strict `527/578`, gold_loose `532/578`

## Delta vs B3

- `P6_masked`: Δcont `+0.065`, Δgold_strict `+0.000`
- `P7_masked`: Δcont `+0.065`, Δgold_strict `+0.000`
- `P7plus_masked`: Δcont `-0.069`, Δgold_strict `+0.268`

## By Scope

- `explicit_device` (n=429)
  - `B3_masked`: cont@10 `0.481`, gold_strict `341/429`
  - `P7plus_masked`: cont@10 `0.490`, gold_strict `402/429`
  - `B4_masked`: cont@10 `0.001`, gold_strict `416/429`
- `explicit_equip` (n=149)
  - `B3_masked`: cont@10 `0.881`, gold_strict `10/149`
  - `P7plus_masked`: cont@10 `0.589`, gold_strict `104/149`
  - `B4_masked`: cont@10 `0.000`, gold_strict `111/149`

## P7+ Diagnostics

- P7+ params (mean): `lambda=0.05287`, `mu=0.04142`, `eta=0.08284`
- Confidence proxy range: `0.60 ~ 0.85`
- Identity check (`P7plus` top-10):
  - same as `B4`: `6/578`
  - same as `B3`: `0/578`
  - same as `P7`: `0/578`

## Interpretation

- The current P7+ policy is a meaningful upgrade over P6/P7 and B3 in strict hit@10.
- P7+ still does not reach B4-level contamination/hit performance.
- The gain mainly comes from confidence-gated blending with device-filtered candidates and shared-cap control, which is consistent with the shared-overload diagnosis.

## Important Caveat (Upper-Bound Risk)

- This experiment is a policy simulation on cached condition outputs (`B3/B4/B4.5` candidate pools), not a full end-to-end online retrieval run.
- Therefore, treat P7+ as a promising algorithmic direction and ablation result, not as final production performance.
- Oracle upper-bound (`B4`) and realistic parser-based results must be reported separately.

## Output Artifact

- `data/paper_a/masked_p6p7_results.json`
