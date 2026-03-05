# Paper A: Preregistered Comparisons, Splits, and Tuning Rules

> Version: 1.0
> Date: 2026-03-05
> Status: LOCKED (do not modify without version bump)

---

## 1. Evaluation Slices (reported as "main")

| Slice | `scope_observability` | Reported? | Condition |
|-------|----------------------|-----------|-----------|
| Explicit Device | `explicit_device` | YES (always) | — |
| Implicit | `implicit` | YES (always) | — |
| Ambiguous | `ambiguous` | YES (always) | — |
| Explicit Equip | `explicit_equip` | CONDITIONAL | Only if Task 3 passes AND slice has non-empty gold for reported split |

---

## 2. Split Policy

| Split | Usage | Rules |
|-------|-------|-------|
| `dev` | Tune/iterate all parameters | All hyperparameter decisions (T, tau, M, router_dim, lambda) use dev only |
| `test` | Final reported numbers | Lock before first test evaluation; no re-tuning after test results are seen |

- **Authoritative eval set**: `data/paper_a/eval/query_gold_master_v0_5.jsonl`
- **Frozen backup**: `data/paper_a/eval/query_gold_master_v0_4_frozen.jsonl`
- **Split assignment**: Stable hash per `leak_key` group (prevents explicit<->masked leakage)

---

## 3. Systems Reported in Main Table

### Required (always reported)

| System | Description |
|--------|-------------|
| B0 | BM25, no scope, no rerank |
| B1 | Dense, no scope, no rerank |
| B2 | Hybrid+RRF, no scope, no rerank |
| B3 | Hybrid+RRF, no scope, rerank (strong baseline) |
| B4 | Hybrid+RRF, auto-parse hard filter, rerank (current production proxy) |
| P1 | Hard + Shared + scope_level-aware filter, rerank |

### Optional (only if runnable + QA passes)

| System | Description | Condition |
|--------|-------------|-----------|
| P2 | Matryoshka Router Top-M, filter, rerank | Router artifacts exist |
| P3 | Router + Family expansion, rerank | Router + family_map artifacts exist |
| P4 | Router + Family + Shared, rerank | All above |
| P6 | P4 + Cont-aware scoring (fixed lambda) | Scoring function implemented |
| P7 | P4 + Cont-aware scoring (adaptive lambda(q)) | Scoring + router confidence implemented |

---

## 4. Comparison Set (fixed, runnable-minimum)

These comparisons MUST appear in the main results table with paired statistical tests:

| Pair | Systems | Primary Metric | Purpose |
|------|---------|---------------|---------|
| CP-1 | B3 vs B4 | adj_cont@5, hit@5 | Effect of hard device filter |
| CP-2 | B4 vs P1 | adj_cont@5, hit@5 | Effect of shared doc policy |

### Optional Extension Comparisons

Only reported if corresponding systems are implemented AND QA passes:

| Pair | Systems | Primary Metric | Purpose | Condition |
|------|---------|---------------|---------|-----------|
| CP-3 | P1 vs P2 | adj_cont@5, hit@5, ScopeAccuracy@M | Router effect on implicit/ambiguous | P2 runnable |
| CP-4 | P4 vs P6 | adj_cont@5, hit@5 | Fixed-lambda scoring effect | P6 runnable |
| CP-5 | P6 vs P7 | adj_cont@5, hit@5 | Adaptive vs fixed lambda | P7 runnable |

---

## 5. Metrics

### Primary (reported in main table)

| Metric | Definition | Role |
|--------|-----------|------|
| Raw Cont@k | `(1/k) * sum 1[device(d_i) not in S(q)]` | Transparency baseline |
| Adj Cont@k | `(1/k) * sum 1[d_i not in D_shared AND device(d_i) not in S(q)]` | **Claim metric** |
| Shared@k | `(1/k) * sum 1[d_i in D_shared]` | Domain characteristic |
| CE@k | `1[exists i<=k : d_i is OOS and not shared]` | Binary contamination |
| Hit@k | `1[exists gold doc in top-k]` | Recall |
| MRR | `1/rank(first gold doc)` | Ranking quality |

### Secondary (reported if applicable)

| Metric | Condition |
|--------|-----------|
| ScopeAccuracy@M | Only for P2/P3/P4 (router systems) |
| p95_latency_ms | All systems |
| Equip-Cont@k | Only if explicit_equip slice is reported |

### k values

- Primary: k=5
- Supplementary: k=1, k=3, k=10

---

## 6. Tuning Rules

| Parameter | Tuned on | Reported on | Constraint |
|-----------|----------|-------------|------------|
| T(shared) | dev-only | test | Default: 3 |
| tau(family) | dev-only | test | Default: 0.2 |
| M(top device) | dev-only | test | Default: 3 |
| router_dim | dev-only | test | Default: 128 |
| lambda_max | dev-only | test | Default: TBD on dev |
| alpha, beta (sigmoid) | dev-only | test | Default: TBD on dev |

- **Hard rule**: No parameter may be tuned after seeing test results.
- **Record**: All tuned values must be recorded in `run_manifest.json`.
- **Sensitivity**: T(shared) sensitivity analysis in Appendix.

---

## 7. Leakage Rules

1. `v_scope(d, q)` MUST be derived from inference-time metadata only (device_name, equip_id from ES fields or parsed from query). No test-label usage.
2. `allowed_devices` / `allowed_equips` in the gold master are used ONLY for metric computation, never for scope construction during retrieval.
3. The `leak_key` grouping in split assignment ensures no question variant appears in both dev and test.
4. Shared doc classification (`D_shared`) is computed from corpus metadata only, not from eval queries.

---

## 8. Statistical Tests

| Test | Applied to | Correction |
|------|-----------|------------|
| Paired bootstrap CI (95%) | adj_cont@5, hit@5, mrr for all CP-* pairs | Holm-Bonferroni across the full comparison set |
| McNemar test | CE@5 for all CP-* pairs | Holm-Bonferroni |
| Effect size | Delta values reported alongside p-values | — |

- Bootstrap samples: 2000 (default)
- Significance threshold: p < 0.05 (after correction)

---

## 9. Adding New Metrics or Comparisons

- New metrics or comparisons MUST be added by updating this preregistration with a version bump (e.g., v1.0 -> v1.1).
- The version bump MUST be committed before any runs using the new metrics.
- "Significance fishing" (adding comparisons after seeing results) is prohibited.

---

## 10. Run Reproducibility Requirements

Every reported run MUST include a `run_manifest.json` with:

- `git_sha`: commit hash at run time
- `resolved_index`: actual ES index used (not alias)
- `hashes.eval_set`: SHA-256 of the eval JSONL consumed
- `hashes.corpus_filter`: SHA-256 of corpus_doc_ids.txt
- `hashes.doc_scope`: SHA-256 of doc_scope.jsonl
- `hashes.family_map`: SHA-256 of family_map.json
- `seed`: RNG seed
- `systems`: list of system IDs executed
- All parameter values from §6
