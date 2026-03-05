# Paper A: Evidence Run Index

> Date: 2026-03-05
> Git SHA: 3c3f7d505dc26c2fa9a10fb45b865ea23e641ff6
> Evaluator: `scripts/paper_a/evaluate_paper_a_master.py`

---

## Artifact Hashes (shared across all runs)

| Artifact | SHA-256 |
|----------|---------|
| eval_set | `e76fba2522dc961fe2f2d83bce80d1296a5e3b6523f37a6b67a4ee9c30d9a815` |
| corpus_filter | `55786f5e363d0c10c33d16e6c3080839581cd075f08069e18ac0f13094c72759` |
| doc_scope | `13dc72a40542eebb7d0f022ffa224bfec36394d86dea6a4b9d9a1a2b03b11d60` |
| family_map | `ed2c1d05aa528d138be4ed9ffdd9e1a29a42dd1405290eeb2a961c78d2a2abcb` |

Corpus: 578 doc_ids, ES index: `rag_chunks_dev_v2` (alias `rag_chunks_dev_current`)

---

## Run 1: test_explicit_device

- **Path**: `.sisyphus/evidence/paper-a/runs/test_explicit_device/`
- **Split**: test | **Observability**: explicit_device
- **Queries**: 22 (all OK, 0 skipped)
- **Systems**: B0, B1, B2, B3, B4, P1
- **Hypotheses served**: H1, H2, H3, H4, H9, H10, H11, H12

| System | adj_cont@5 | raw_cont@5 | shared@5 | ce@5 | hit@5 | mrr |
|--------|-----------|-----------|---------|------|-------|-----|
| B0 | 0.282 | 0.309 | 0.118 | 0.727 | 0.364 | 0.274 |
| B1 | 0.209 | 0.218 | 0.055 | 0.500 | 0.318 | 0.295 |
| B2 | 0.273 | 0.300 | 0.118 | 0.682 | 0.409 | 0.278 |
| B3 | 0.255 | 0.282 | 0.127 | 0.682 | 0.409 | 0.322 |
| B4 | **0.027** | 0.045 | 0.155 | **0.045** | 0.091 | 0.091 |
| P1 | **0.000** | 0.264 | 0.882 | **0.000** | 0.136 | 0.136 |

---

## Run 2: test_implicit

- **Path**: `.sisyphus/evidence/paper-a/runs/test_implicit/`
- **Split**: test | **Observability**: implicit
- **Queries**: 21 (all OK, 0 skipped)
- **Systems**: B0, B1, B2, B3, B4, P1
- **Hypotheses served**: H5, H6

| System | adj_cont@5 | raw_cont@5 | shared@5 | ce@5 | hit@5 | mrr |
|--------|-----------|-----------|---------|------|-------|-----|
| B0 | 0.610 | 0.848 | 0.238 | 0.905 | 0.238 | 0.179 |
| B1 | 0.743 | 0.876 | 0.133 | 1.000 | 0.095 | 0.095 |
| B2 | 0.610 | 0.848 | 0.238 | 0.905 | 0.286 | 0.180 |
| B3 | 0.600 | 0.848 | 0.248 | 0.905 | 0.286 | 0.206 |
| B4 | 0.600 | 0.848 | 0.248 | 0.905 | 0.286 | 0.206 |
| P1 | **0.000** | 0.905 | 0.905 | **0.000** | 0.000 | 0.000 |

---

## Run 3: test_explicit_equip

- **Path**: `.sisyphus/evidence/paper-a/runs/test_explicit_equip/`
- **Split**: test | **Observability**: explicit_equip
- **Queries**: 8 (all OK, 0 skipped)
- **Systems**: B0, B1, B2, B3, B4, P1
- **Hypotheses served**: H7

| System | adj_cont@5 | raw_cont@5 | shared@5 | ce@5 | hit@5 | mrr |
|--------|-----------|-----------|---------|------|-------|-----|
| B0 | 0.900 | 1.000 | 0.100 | 1.000 | 0.000 | 0.000 |
| B1 | 0.975 | 0.975 | 0.000 | 1.000 | 0.000 | 0.000 |
| B2 | 0.900 | 1.000 | 0.100 | 1.000 | 0.000 | 0.000 |
| B3 | 0.900 | 1.000 | 0.100 | 1.000 | 0.000 | 0.000 |
| B4 | 0.900 | 1.000 | 0.100 | 1.000 | 0.000 | 0.000 |
| P1 | **0.000** | 0.950 | 1.000 | **0.000** | 0.000 | 0.000 |

---

## Run 4: dev_ambiguous

- **Path**: `.sisyphus/evidence/paper-a/runs/dev_ambiguous/`
- **Split**: dev | **Observability**: ambiguous
- **Queries**: 80 (0 OK, 80 skipped — all queries have empty gold_doc_ids)
- **Systems**: B0, B1, B2, B3, B4, P1
- **Hypotheses served**: H8 (descriptive analysis only)
- **Note**: All 80 ambiguous queries were skipped due to empty gold_doc_ids. These queries are forced to dev split by the split assignment rule. This run confirms the ambiguous slice cannot support recall/contamination metrics — only retrieval result inspection is possible.

---

## Key Observations

1. **H1 supported**: B4 adj_cont@5 = 0.027 vs B3 = 0.255 on explicit_device (−89%)
2. **H2 concern**: B4 hit@5 = 0.091 vs B3 = 0.409 — hard filter causes significant recall loss
3. **P1 eliminates adjusted contamination**: adj_cont@5 = 0.000 across all slices by reclassifying OOS docs as shared
4. **P1 recall trade-off**: hit@5 drops from B4's 0.091 to 0.136 (explicit_device), but drops to 0.000 on implicit — shared-only retrieval is insufficient for implicit queries
5. **H5 supported**: B3 adj_cont@5 implicit (0.600) > explicit_device (0.255) — implicit queries are harder
6. **H6 partial**: B4 = B3 on implicit (both 0.600 adj_cont@5, 0.286 hit@5) — parser fallback to global means no filter applied
7. **Equip slice**: 8 queries meets H7 minimum threshold (>=5). P1 eliminates contamination but all systems show hit@5 = 0.000 — gold docs may not be in corpus

---

## Files per Run

Each run directory contains:
- `run_manifest.json` — config, hashes, query counts
- `per_query.csv` — per-query per-system metrics
- `summary_all.csv` — aggregated metrics
- `summary_by_observability.csv` — metrics by scope_observability
- `bootstrap_ci.json` — bootstrap confidence intervals (2000 samples)
- `mcnemar.json` — McNemar test results for CE@k
