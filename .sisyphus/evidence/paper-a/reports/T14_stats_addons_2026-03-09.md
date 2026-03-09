# Paper A — Stats Add-ons Report

Generated: 2026-03-09

## H5: Cross-Slice CI — B3 Implicit vs Explicit-Device (adj_cont@5)

| Slice | N | Mean adj_cont@5 |
|---|---|---|
| implicit | 21 | 0.3913 |
| explicit_device | 22 | 0.1295 |

- **Delta** (implicit − explicit_device): 0.2617
- **95% Bootstrap CI**: [0.0411, 0.4670]

**Interpretation**: The 95% CI [0.0411, 0.4670] lies entirely above zero, suggesting that B3 performs meaningfully better on implicit queries than on explicit-device queries (delta = 0.2617).

## H10: Paired Bootstrap CI — B1 vs B0, Explicit-Device Slice (adj_cont@5)

- **N paired queries**: 22
- **Delta mean** (B1 − B0): 0.0530
- **95% Bootstrap CI**: [-0.0182, 0.1379]

**Interpretation**: The paired 95% CI [-0.0182, 0.1379] crosses zero (delta = 0.0530), so there is insufficient evidence of a systematic advantage for B1 over B0 on the explicit-device slice.

## H11: Equivalence Test — B3 vs B2, Explicit-Device Slice (adj_cont@5)

- **N paired queries**: 22
- **Delta mean** (B3 − B2): 0.0091
- **90% Bootstrap CI**: [0.0000, 0.0273]
- **Equivalence margin**: ±0.02
- **Equivalent**: NO

**Interpretation**: The 90% CI [0.0000, 0.0273] extends outside the equivalence band [−0.02, +0.02] (delta = 0.0091). Equivalence between B3 and B2 cannot be claimed at the 0.02 margin.
