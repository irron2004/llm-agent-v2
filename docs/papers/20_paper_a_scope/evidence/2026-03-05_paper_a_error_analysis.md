# Paper A: Error Analysis and Failure Taxonomy

> Date: 2026-03-05
> Source: per_query.csv from test_explicit_device (22q), test_implicit (21q), test_explicit_equip (8q)

---

## 1. Failure Taxonomy

| Code | Name | Description |
|------|------|-------------|
| F1 | Parser Mismatch | Auto-parse extracts wrong device name (e.g., "ZEDIUS XP" instead of "SUPRA_XP"), causing B4 to filter on wrong scope |
| F2 | Parser Miss | Auto-parse fails to extract any device from query; B4 falls back to global (no filter applied) |
| F3 | Shared Over-inclusion | P1 fills top-k with shared docs (shared@5 > 0.5), pushing device-specific relevant docs below cutoff |
| F4 | Gold Not In Corpus | Gold doc_ids not retrievable by any system; hit@5=0 across all systems regardless of filtering |
| F5 | Cross-Device Semantic Overlap | Global retrieval pulls semantically similar docs from wrong device into top-k |
| F6 | Recall-Contamination Trade-off | Correct filter eliminates contamination but also excludes the gold doc (gold doc indexed under different device variant) |
| F7 | Equip-Level Corpus Gap | Equip-specific queries have no matching gold docs in indexed corpus; all systems fail |

---

## 2. Quantitative Failure Counts

### test_explicit_device (n=22)

| System | hit@5=0 (recall fail) | adj_cont@5>0 (contaminated) | ce@5=1 (binary contam.) | shared@5>0.5 |
|--------|----------------------|----------------------------|------------------------|--------------|
| B0 | 14 | 16 | 16 | 0 |
| B1 | 15 | 11 | 11 | 0 |
| B2 | 13 | 15 | 15 | 0 |
| B3 | 13 | 15 | 15 | 0 |
| B4 | 20 | 1 | 1 | 2 |
| P1 | 19 | 0 | 0 | 20 |

### test_implicit (n=21)

| System | hit@5=0 | adj_cont@5>0 | ce@5=1 |
|--------|---------|-------------|--------|
| B0 | 16 | 19 | 19 |
| B1 | 19 | 21 | 21 |
| B2 | 15 | 19 | 19 |
| B3 | 15 | 19 | 19 |
| B4 | 15 | 19 | 19 |
| P1 | 21 | 0 | 0 |

### test_explicit_equip (n=8)

All 8 queries: hit@5=0 across all systems (F7). P1 eliminates contamination (adj_cont@5=0) but retrieves no gold docs.

---

## 3. Parser Analysis: ZEDIUS XP vs SUPRA_XP

A critical finding: 16 of 22 explicit_device queries have `allowed_devices=SUPRA_XP` but `parsed_hard_devices=ZEDIUS XP`. The parser extracts "ZEDIUS XP" from the query text, but the gold label expects "SUPRA_XP". This is an **alias mismatch** — ZEDIUS XP is a product name variant for SUPRA XP equipment.

Impact:
- B4 filters on "ZEDIUS XP" scope, which contains different documents than "SUPRA_XP" scope
- 8 queries where B3 hits but B4 misses are all caused by this alias mismatch (F1)
- B4's adj_cont@5=0.027 is artificially low because the wrong-scope filter happens to exclude contamination too

---

## 4. Representative Error Cases (10)

| # | q_id | System | Failure | Description |
|---|------|--------|---------|-------------|
| 1 | A-gs110 | B4 | F2 | Parser extracts no device; B4 falls back to global. allowed=SUPRA_XP but parsed=empty. B4=B3 (adj_cont=0.60) |
| 2 | A-sop002 | B4 | F1 | Parser extracts "ZEDIUS XP", gold expects "SUPRA_XP". B3 hits (h=1.0), B4 misses (h=0.0) |
| 3 | A-sop044 | B4 | F1 | Same ZEDIUS/SUPRA mismatch. B3 hits with contamination (c=0.40), B4 eliminates both contamination and recall |
| 4 | A-sop072 | B4 | F1 | ZEDIUS/SUPRA mismatch. B3 h=1.0 c=0.60, B4 h=0.0 c=0.00 — worst recall-contamination trade-off |
| 5 | A-xdev016 | B4 | F1 | Parser extracts "SUPRA V", gold expects "SUPRA_VPLUS". Partial name match fails. B3 h=1.0, B4 h=0.0, P1 h=1.0 |
| 6 | A-sop011 | P1 | F3 | Parser correct (SUPRA XP). B3 h=1.0, B4 h=1.0, but P1 h=0.0 — shared docs flood top-5, pushing gold doc out |
| 7 | A-xdev002 | P1 | — | Success case: P1 h=1.0 where B3 h=0.0. Shared policy recovers gold doc for GENEVA_XP query |
| 8 | A-xdev021 | B3 | F5 | B3 adj_cont=1.00 — all top-5 docs from wrong device. B4 correctly filters (h=1.0). Best case for hard filter |
| 9 | A-q006 | All | F4 | hit@5=0 across all systems. Gold doc not in retrievable corpus. SUPRA_N query, parser correct |
| 10 | A-sop048 | B3 | F5+F4 | B3 has contamination (c=0.20) and no recall (h=0.0). Neither filtering nor global retrieval helps |

---

## 5. Cross-System Error Correlation

### Consistently hard queries (all systems hit@5=0)

- **test_explicit_device**: 9/22 queries (41%) — A-q006, A-gs110, A-sop018, A-sop038, A-sop042, A-sop048, A-sop051, A-sop076, A-sop077
- **test_implicit**: 15/21 queries (71%)
- **test_explicit_equip**: 8/8 queries (100%)

These represent F4 (gold not in corpus) or fundamental retrieval failures unrelated to scope policy.

### Queries that only fail under filtering (B3 hits, B4/P1 misses)

- B3 hits but B4 misses: 8 queries — all due to ZEDIUS XP / SUPRA_XP alias mismatch (F1)
- B3 hits but P1 misses: 8 queries — 7 overlap with B4 misses (F1), plus A-sop011 (F3, shared over-inclusion)

### Queries that only succeed under filtering

- B4 hits but B3 misses: 1 query (A-xdev021) — B4 correctly filters OMNIS scope, removing SUPRA contamination
- P1 hits but B3 misses: 2 queries (A-xdev002, A-xdev004) — shared policy surfaces GENEVA_XP gold docs

---

## 6. Implications for Paper Claims

### F1 (Parser Mismatch) — Dominant failure mode
- Affects 16/22 explicit_device queries (73%)
- **Actionable**: Add device alias mapping (ZEDIUS XP → SUPRA_XP) to parser
- If fixed, B4 hit@5 would likely improve dramatically; current results understate hard filter recall
- **Critical for paper**: Current B4 results are pessimistic due to this systematic parser bug

### F2 (Parser Miss) — Affects 1/22 explicit_device queries
- Addressable by planned Matryoshka router (P2-P4) which doesn't rely on text parsing

### F3 (Shared Over-inclusion) — Affects P1 recall
- 20/22 queries have shared@5 > 0.5 under P1
- Shared docs flood results, suppressing device-specific docs
- **Addressable by**: Contamination-aware scoring (P6-P7) which can downweight shared docs when device-specific ones available

### F4 (Gold Not In Corpus) — Irreducible
- 9/22 explicit_device queries have no retrievable gold docs
- Not addressable by any scope policy
- **Paper should report**: Conditional metrics on queries with gold docs in corpus

### F5 (Cross-Device Semantic Overlap) — Core motivation
- Validates the paper's central claim that global retrieval causes contamination
- Hard filter (B4) and scope policy (P1) both address this

### F7 (Equip-Level Corpus Gap) — Blocks H7
- All 8 equip queries fail across all systems
- Equip-level gold docs not in indexed corpus
- H7 cannot be evaluated from current data

---

## 7. Recommended Actions

1. **Fix ZEDIUS XP alias** in auto-parser and re-run B4 evidence to get unbiased hard filter results
2. **Report conditional hit@5** excluding F4 queries (gold not retrievable) for cleaner comparison
3. **Add shared-doc cap** to P1 to prevent F3 (e.g., max 3 shared docs in top-5, fill rest with device docs)
4. **Defer H7** until equip-level gold docs are verified in corpus
5. **Stratify F1 impact** in paper: show B4 results with and without alias fix
