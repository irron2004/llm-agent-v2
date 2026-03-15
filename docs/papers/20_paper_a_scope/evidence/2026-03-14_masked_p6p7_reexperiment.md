# Masked P6/P7 Re-experiment (2026-03-14)

Date: 2026-03-14
Status: generated from `scripts/paper_a/run_masked_p6p7_experiment.py`

## Inputs

- Base result: `data/paper_a/masked_hybrid_results.json`
- Doc scope: `.sisyphus/evidence/paper-a/policy/doc_scope.jsonl`
- Shared doc ids: `.sisyphus/evidence/paper-a/policy/shared_doc_ids.txt`

## Command

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/run_masked_p6p7_experiment.py
```

## All Queries (n=578)

- `B3_masked`: cont@10 `0.584`, gold_strict `351/578`, gold_loose `380/578`
- `B4_masked`: cont@10 `0.001`, gold_strict `527/578`, gold_loose `532/578`
- `B4.5_masked`: cont@10 `0.001`, gold_strict `406/578`, gold_loose `439/578`
- `P6_masked`: cont@10 `0.649`, gold_strict `351/578`, gold_loose `380/578`
- `P7_masked`: cont@10 `0.649`, gold_strict `351/578`, gold_loose `380/578`

## Delta vs `B3_masked`

- `B4_masked`: cont@10 `-0.584`, gold_strict `+0.304`, gold_loose `+0.263`
- `B4.5_masked`: cont@10 `-0.584`, gold_strict `+0.095`, gold_loose `+0.102`
- `P6_masked`: cont@10 `+0.065`, gold_strict `+0.000`, gold_loose `+0.000`
- `P7_masked`: cont@10 `+0.065`, gold_strict `+0.000`, gold_loose `+0.000`

## By Scope

- `explicit_device` (n=429)
  - `B3_masked`: cont@10 `0.481`, gold_loose `357/429`
  - `B4_masked`: cont@10 `0.001`, gold_loose `416/429`
  - `P6_masked`: cont@10 `0.530`, gold_loose `357/429`
  - `P7_masked`: cont@10 `0.530`, gold_loose `357/429`
- `explicit_equip` (n=149)
  - `B3_masked`: cont@10 `0.881`, gold_loose `23/149`
  - `B4_masked`: cont@10 `0.000`, gold_loose `116/149`
  - `P6_masked`: cont@10 `0.991`, gold_loose `23/149`
  - `P7_masked`: cont@10 `0.991`, gold_loose `23/149`

## Adaptive Lambda Stats

- Mean `lambda_p7`: `0.03246`
- Max `lambda_p7`: `0.05000`
- Non-zero lambda queries: `559/578`

## Key Takeaways

- On the masked v0.6+ setting, current P6/P7 do not improve over `B3_masked`; they preserve neither recall nor contamination.
- `B4_masked` remains clearly better than both soft-scoring variants, so the earlier `+1.9%` narrative does not carry over to this masked experiment setup.
- The current adaptive lambda rule is too weak to reorder useful documents, and in practice it slightly worsens contamination without recovering additional gold hits.

## Output Artifact

- Full per-query result: `data/paper_a/masked_p6p7_results.json`
