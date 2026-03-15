# v0.7 Mixed Eval Restoration (2026-03-14)

Date: 2026-03-14
Status: generated from `scripts/paper_a/build_v07_mixed_eval_set.py`

## Inputs

- Explicit base: `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl`
- Restored slices: `data/paper_a/eval/query_gold_master_v0_5.jsonl` (`implicit`, `ambiguous` only)

## Command

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/build_v07_mixed_eval_set.py
```

## Summary

- Total rows: 808
- Restored rows from v0.5: 230
- Scope counts: {'ambiguous': 80, 'explicit_device': 429, 'explicit_equip': 149, 'implicit': 150}
- Split counts: {'dev': 626, 'test': 182}
- Dev/Test leak-key overlap: 0

## Scope by split

- ambiguous: {'test': 9, 'dev': 71}
- explicit_device: {'dev': 324, 'test': 105}
- explicit_equip: {'dev': 114, 'test': 35}
- implicit: {'dev': 117, 'test': 33}

## Empty gold counts

- ambiguous: 37
- explicit_device: 0
- explicit_equip: 0
- implicit: 10

## Interpretation

- v0.7 mixed restores the missing implicit/ambiguous slices while preserving the stronger v0.6 explicit set as the base.
- ambiguous rows still carry empty-gold limitations from v0.5 and should be reported separately from gold-bearing slices.
- The merged file is intended to unblock mixed-scope reporting and future experiment runs, not to hide the coverage limits of ambiguous rows.
