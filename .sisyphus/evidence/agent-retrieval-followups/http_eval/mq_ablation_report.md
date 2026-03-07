# MQ Ablation Report

| mode | failures | doc@1 | doc@3 | doc@5 | doc@10 | page@1 | page@3 | page@5 | page@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 31 | 0.3924 | 0.4177 | 0.4177 | 0.5063 | 0.1266 | 0.3671 | 0.3671 | 0.4430 |
| fallback | 79 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| on | 79 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- Jaccard@k is omitted because this evaluator run does not include a paired baseline needed for overlap computation.
