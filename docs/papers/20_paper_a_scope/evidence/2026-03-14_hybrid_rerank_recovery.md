# Hybrid/Rerank Recovery (2026-03-14)

Date: 2026-03-14
Status: generated from `scripts/paper_a/run_masked_hybrid_experiment.py`

## Inputs

- Eval set: `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl`
- Doc scope: `.sisyphus/evidence/paper-a/policy/doc_scope.jsonl`
- Shared doc ids: `.sisyphus/evidence/paper-a/policy/shared_doc_ids.txt`

## Command

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/run_masked_hybrid_experiment.py
```

## All Queries (n=578)

- `B0_orig`: cont@10 `0.422`, gold_strict `394/578`, gold_loose `419/578`
- `B1_orig`: cont@10 `0.373`, gold_strict `379/578`, gold_loose `397/578`
- `B2_orig`: cont@10 `0.365`, gold_strict `435/578`, gold_loose `457/578`
- `B3_orig`: cont@10 `0.365`, gold_strict `434/578`, gold_loose `456/578`
- `B0_masked`: cont@10 `0.473`, gold_strict `287/578`, gold_loose `343/578`
- `B1_masked`: cont@10 `0.730`, gold_strict `228/578`, gold_loose `261/578`
- `B2_masked`: cont@10 `0.585`, gold_strict `351/578`, gold_loose `380/578`
- `B3_masked`: cont@10 `0.584`, gold_strict `351/578`, gold_loose `380/578`
- `B4_masked`: cont@10 `0.001`, gold_strict `527/578`, gold_loose `532/578`
- `B4.5_masked`: cont@10 `0.001`, gold_strict `406/578`, gold_loose `439/578`

## By Scope

- `explicit_device` (n=429)
  - `B3_masked`: cont@10 `0.481`, gold_loose `357/429`
  - `B4_masked`: cont@10 `0.001`, gold_loose `416/429`
  - `B4.5_masked`: cont@10 `0.001`, gold_loose `341/429`
- `explicit_equip` (n=149)
  - `B3_masked`: cont@10 `0.881`, gold_loose `23/149`
  - `B4_masked`: cont@10 `0.000`, gold_loose `116/149`
  - `B4.5_masked`: cont@10 `0.000`, gold_loose `98/149`

## Key Takeaways

- Hybrid/Rerank recovery does not weaken the masked-filter story. `B4_masked` remains the strongest condition on both contamination and gold hit.
- Dense-only masked retrieval (`B1_masked`) performs worst, suggesting that the shared-topic contamination problem is not solved by embedding retrieval alone.
- `B4.5_masked` still underperforms `B4_masked`, so the shared-policy paradox persists even after adding dense/hybrid/rerank conditions.

## Output Artifact

- Full per-query result: `data/paper_a/masked_hybrid_results.json`
