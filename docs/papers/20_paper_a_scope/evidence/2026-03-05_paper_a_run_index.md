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

## 2026-03-09 Reruns (Alias Fix + B4.5 + Shared Metrics)

> Git SHA: 90cd9c24e572403d6d1e902ce757a9da021122bb
> Date: 2026-03-09
> Key changes: canonical alias normalization, B4.5 system added, adj_den/shared_rel metrics, rerank proof fields
> Status: rerun completed with norm->raw ES device expansion applied

### Artifact Hashes (2026-03-09)

| Artifact | SHA-256 |
|----------|---------|
| eval_set | `e76fba2522dc961fe2f2d83bce80d1296a5e3b6523f37a6b67a4ee9c30d9a815` |
| corpus_filter | `55786f5e363d0c10c33d16e6c3080839581cd075f08069e18ac0f13094c72759` |
| doc_scope | `13dc72a40542eebb7d0f022ffa224bfec36394d86dea6a4b9d9a1a2b03b11d60` |
| family_map | `ed2c1d05aa528d138be4ed9ffdd9e1a29a42dd1405290eeb2a961c78d2a2abcb` |
| device_catalog | `dab77950a9018472ddae2d6c0729fe864e1161c3eadf03d3f488b70c386787ed` |
| test_baselines/run_manifest.json | `983959a17441e8b02adca9cdde804cc32c955fa0c037e86927e8b921604b8d70` |
| test_baselines/per_query.csv | `8a15aaaaf45d3b86b305c298dceb722bf1d9699807f2f0f7f0ad3a5caffc7174` |
| test_core/run_manifest.json | `2c6060f06714febe243ace7c1f9408e62608d6b0c9727d9b01a52f24fb3b9a7d` |
| test_core/per_query.csv | `35e681d825da82a49c18e3c462a70f0089c1c34d0958a14811ae76a65c2a2e7e` |
| test_explicit_device_core/run_manifest.json | `e84fcb4dcd56fa1d517e30d68df12c50e050e4e095e9737bbb879ba91e91776c` |
| test_explicit_device_core/per_query.csv | `8f08c485f30156698d87ae401310a26161c4a001cb7fc971d55d0b099061be1f` |

### Run 1: test_baselines (n=51)

- **Path**: `.sisyphus/evidence/paper-a/runs/2026-03-09_test_baselines/`
- **Split**: test | **Systems**: B0, B1, B2, B3
- **Queries**: 51 (all OK, 0 skipped)
- **Note**: Baseline systems across full test set (explicit_device + explicit_equip + implicit)

| System | n_ok | mean_raw_cont@5 | mean_adj_cont@5 | mean_adj_den@5 | mean_shared@5 | mean_shared_rel@5 | mean_ce@5 | mean_hit@5 | mean_mrr |
|--------|------|-----------------|-----------------|----------------|----------------|-------------------|-----------|-----------|---------|
| B0 | 51 | 0.3961 | 0.3592 | 4.1765 | 0.1647 | 0.0000 | 0.4706 | 0.2549 | 0.1918 |
| B1 | 51 | 0.4667 | 0.4748 | 4.6078 | 0.0784 | 0.0196 | 0.6863 | 0.1765 | 0.1667 |
| B2 | 51 | 0.3961 | 0.3601 | 4.1765 | 0.1647 | 0.0196 | 0.4706 | 0.2941 | 0.1941 |
| B3 | 51 | 0.3961 | 0.3592 | 4.1373 | 0.1725 | 0.0196 | 0.4706 | 0.2941 | 0.2239 |

### Run 2: test_core (n=51)

- **Path**: `.sisyphus/evidence/paper-a/runs/2026-03-09_test_core/`
- **Split**: test | **Systems**: B4, B4.5, P1
- **Queries**: 51 (all OK, 0 skipped)
- **Note**: Scoped systems (hard filter B4, new B4.5 dense variant, shared-aware P1) across full test set

| System | n_ok | mean_raw_cont@5 | mean_adj_cont@5 | mean_adj_den@5 | mean_shared@5 | mean_shared_rel@5 | mean_ce@5 | mean_hit@5 | mean_mrr |
|--------|------|-----------------|-----------------|----------------|----------------|-------------------|-----------|-----------|---------|
| B4 | 51 | 0.3608 | 0.3229 | 4.1373 | 0.1725 | 0.0000 | 0.4118 | 0.2941 | 0.2239 |
| B4.5 | 51 | 0.4863 | 0.0000 | 1.8627 | 0.6275 | 0.0065 | 0.0000 | 0.1765 | 0.1417 |
| P1 | 51 | 0.5451 | 0.0000 | 0.4510 | 0.9098 | 0.0163 | 0.0000 | 0.0588 | 0.0588 |

**New in B4.5**: Dense vector variant of scoped retrieval. It keeps zero adjusted contamination while recovering substantial recall vs P1 (hit@5 0.1765 vs 0.0588), but still trails B4/B3.

### Run 3: test_explicit_device_core (n=22)

- **Path**: `.sisyphus/evidence/paper-a/runs/2026-03-09_test_explicit_device_core/`
- **Split**: test | **Observability**: explicit_device
- **Queries**: 22 (all OK, 0 skipped)
- **Systems**: B4, B4.5, P1
- **Note**: Scoped systems on device-observability queries only (R2 mitigation validation)

| System | n_ok | mean_raw_cont@5 | mean_adj_cont@5 | mean_adj_den@5 | mean_shared@5 | mean_shared_rel@5 | mean_ce@5 | mean_hit@5 | mean_mrr |
|--------|------|-----------------|-----------------|----------------|----------------|-------------------|-----------|-----------|---------|
| B4 | 22 | 0.0455 | 0.0455 | 4.3636 | 0.1273 | 0.0000 | 0.0455 | 0.4091 | 0.3220 |
| B4.5 | 22 | 0.1000 | 0.0000 | 3.8636 | 0.2273 | 0.0152 | 0.0000 | 0.4091 | 0.3285 |
| P1 | 22 | 0.2364 | 0.0000 | 0.5909 | 0.8818 | 0.0379 | 0.0000 | 0.1364 | 0.1364 |

**New metrics (R2 mitigation validation)**:
- **adj_den@5**: adjusted density — mean number of in-scope documents per query in top-5 after filtering. B4=4.36 and B4.5=3.86 maintain strong density, while P1=0.59 remains sparse due to shared-heavy routing.
- **shared_rel@5**: shared document relative ratio — B4.5 (0.0152) uses shared docs modestly, whereas P1 (0.0379) remains more shared-dominant.

---

## Files per Run

Each run directory contains:
- `run_manifest.json` — config, hashes, query counts
- `per_query.csv` — per-query per-system metrics
- `summary_all.csv` — aggregated metrics
- `summary_by_observability.csv` — metrics by scope_observability
- `bootstrap_ci.json` — bootstrap confidence intervals (2000 samples)
- `mcnemar.json` — McNemar test results for CE@k
