# Synthetic Benchmark v1: Stability Bench (Public Release Spec)

## Purpose
Provide a fully reproducible, publicly releasable benchmark that stresses the same failure modes observed in industrial RAG retrieval stability:
- abbreviation preservation (do NOT expand domain acronyms)
- mixed ko/en tokens
- code-like identifiers and error-code strings
- near-duplicate documents that create tie / near-tie rankings

This benchmark is the primary evidence source for Paper B quantitative results.

## Output Location (fixed)
Generator MUST write to:
- `data/synth_benchmarks/stability_bench_v1/`

Files:
- `corpus.jsonl`
- `queries.jsonl`
- `manifest.json` (hashes + generation params)

## Corpus Schema (JSONL)
One JSON object per line.

Required fields:
- `doc_id` (string): MUST start with `DOC_`.
- `doc_type` (string): one of `{manual, troubleshooting, setup}`.
- `device_name` (string): synthetic device label (e.g., `SUPRA_N_SYNTH`).
- `content` (string): document body.
- `tags` (array[string]): may include `abbr`, `mixed_lang`, `error_code`, `near_dup`, `high_risk`.

Optional fields (allowed but not required):
- `equip_id` (string)
- `chapter` (string)

## Query Schema (JSONL)
One JSON object per line.

Required fields:
- `qid` (string): MUST start with `Q_`.
- `group_id` (string): paraphrase group identifier, MUST start with `G_`.
- `canonical_query` (string): the canonical intent expression.
- `query` (string): the actual query text for this record.
- `paraphrase_level` (string): one of `{low, mid, high}`.
- `expected_doc_ids` (array[string]): list of relevant `doc_id` values.
- `tags` (array[string]): may include `abbr`, `mixed_lang`, `error_code`, `near_dup`, `ambiguous`.

Notes:
- A paraphrase group contains exactly 4 query records (see sizes).
- `expected_doc_ids` MUST refer to doc_ids that exist in `corpus.jsonl`.

## Fixed Dataset Sizes
Corpus:
- Total documents: 120
- Near-duplicate design: 30 near-duplicate *pairs* (60 docs) where each pair differs by exactly one critical token.
  - Example token classes: abbreviation vs expanded form, error code digit, module code, unit.
  - Purpose: induce tie/near-tie rankings and stability failures.

Queries:
- Paraphrase groups: 60
- Queries per group: 4
- Total query records: 240

Distribution constraints (hard requirements):
- At least 20 groups tagged `abbr`.
- At least 20 groups tagged `mixed_lang`.
- At least 10 groups tagged `error_code`.
- At least 15 groups whose relevant docs include near-duplicate pairs (`near_dup`).

## Leakage Rules (MUST enforce)
These are hard constraints; generator MUST fail fast if violated.

Rule L1 (doc_id token leakage)
- A query MUST NOT contain any doc_id token patterns.
- Forbidden patterns: `DOC_` or `SYNTH_` anywhere in the `query` string.

Rule L2 (string overlap)
- Let `gold_doc_content` be the concatenation of the contents of the docs referenced by `expected_doc_ids`.
- Longest common substring length between `query` and `gold_doc_content` MUST be <= 40 characters.

Rule L3 (5-gram overlap)
- Compute 5-gram Jaccard overlap between `query` and `gold_doc_content`.
- 5-gram Jaccard MUST be <= 0.35.

## Determinism + Manifest
Generator MUST be deterministic given a seed.

`manifest.json` MUST include:
- `seed` (int)
- `generator_version` (string)
- `files`:
  - `corpus.jsonl`: sha256
  - `queries.jsonl`: sha256
- `counts` (docs, groups, queries)
- `leakage_rules` (L1-L3 thresholds)

## Evaluation Contract (for Paper B)
Evaluation MUST compute (k fixed at 10):
- hit@5, hit@10, MRR
- RepeatJaccard@10, RepeatExactMatch@10 (N_repeats=10)
- ParaphraseJaccard@10, ParaphraseExactMatch@10 (within each group)
- p95_latency_ms
