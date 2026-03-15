# B4.5 Failure Decomposition (2026-03-14)

Date: 2026-03-14
Status: generated from `scripts/paper_a/analyze_b45_failure_decomposition.py`

## Inputs

- Results: `data/paper_a/masked_hybrid_results.json`
- Doc scope: `.sisyphus/evidence/paper-a/policy/doc_scope.jsonl`
- Shared doc ids: `.sisyphus/evidence/paper-a/policy/shared_doc_ids.txt`

## Command

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/analyze_b45_failure_decomposition.py
```

## Summary

- Total masked queries analyzed: 578
- B4 loose hit@10: 532/578 (92.0%)
- B4.5 loose hit@10: 439/578 (76.0%)
- Paradox cases (B4 hit, B4.5 miss): 98/578 (17.0%)

## Category Breakdown

- shared_overload: 84/98 (85.7%)
- ranking_dilution: 14/98 (14.3%)

## Representative Cases

### shared_overload

- `A-gen0003` (explicit_device, GENEVA XP): shared=10, target=0, oos=0, top5 B4.5=['global_sop_integer_plus_all_am_heater_chuck', 'global_sop_precia_all_efem_device_net_board', 'global_sop_precia_all_pm_device_net_board', 'global_sop_precia_all_tm_device_net_board', 'global_sop_precia_all_efem_device_net_board']
- `A-gen0020` (explicit_device, GENEVA XP): shared=10, target=0, oos=0, top5 B4.5=['global_sop_precia_all_efem_device_net_board', 'global_sop_supra_n_series_all_pm_device_net_board', 'global_sop_precia_all_pm_device_net_board', 'global_sop_precia_all_efem_device_net_board', 'global_sop_precia_all_tm_device_net_board']
- `A-gen0032` (explicit_device, SUPRA Vplus): shared=5, target=0, oos=0, top5 B4.5=['40139507', '40153236', '40081665', '40096119', '40148205']

### ranking_dilution

- `A-gen0403` (explicit_equip, SUPRA Vplus): shared=4, target=1, oos=0, top5 B4.5=['40043406', 'global_sop_supra_n_series_all_tm_robot', 'global_sop_supra_xp_all_efem_ffu', 'global_sop_geneva_xp_rep_efem_ffu', '40151321']
- `A-gen0433` (explicit_device, SUPRA N): shared=4, target=1, oos=0, top5 B4.5=['40115819', '40115923', 'global_sop_integer_plus_all_ll_sensor_board', '40052429', 'global_sop_precia_all_tm_device_net_board']
- `A-gen0454` (explicit_equip, TIGMA Vplus): shared=4, target=3, oos=0, top5 B4.5=['global_sop_supra_n_series_all_tm_robot', 'global_sop_supra_xp_all_efem_ffu', 'global_sop_geneva_xp_rep_efem_ffu', '40044494', '40057774']

## Proposed Policy Fix

- Recommendation: when a target device is already known, rank target-device docs before shared docs and cap shared-doc exposure in the early top-k window.
- If the dominant category (`shared_overload`) were fully recovered to B4-level behavior, the expected loose hit@10 gain would be about +14.5%p on the full masked set.

## Interpretation Notes

- This decomposition uses the masked hybrid result set because it provides per-query top_doc_ids for B4 and B4.5.
- The same recall inversion pattern (B4.5 < B4) also appears in the broader 2026-03-14 execution narrative, so this analysis is intended as a diagnosis aid, not a final claim by itself.
