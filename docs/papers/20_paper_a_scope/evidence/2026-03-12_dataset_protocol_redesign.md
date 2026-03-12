# Paper A Dataset Protocol Redesign (Bias-Resilient)

Date: 2026-03-12
Status: Proposed protocol for next evaluation cycle

## 1) Why redesign is required

Current evidence and code indicate that the present gold construction and metric path blur the paper's core claim.

- Gold construction is retrieval-pool dependent (`scripts/paper_a/phase3_retrieve_and_pool.py`): judged candidates only come from systems already in the pool.
- Gold expansion is source-seeded (`scripts/paper_a/generate_question_gold_from_corpus.py`): source doc is always included, then expanded by heuristic topical overlap.
- Evaluation currently consumes loose gold only (`scripts/paper_a/evaluate_paper_a_master.py`): `gold_doc_ids_strict` exists but is not used in scoring.
- Evidence confirms strict-gold sparsity and large no-strict regions (`docs/papers/20_paper_a_scope/evidence/2026-03-09_gold_rejudging_analysis.md`), making strict recall sensitive to corpus coverage rather than retrieval quality.

Result: a large part of measured variance can come from pool bias, loose-vs-strict label choice, and coverage gaps, not from scope policy itself.

## 2) Observed methodological biases

### B1. Pooling bias (candidate truncation)
- If a relevant document is never surfaced by pooled systems, it cannot become gold.
- This creates circularity: systems define the candidate universe used to evaluate those same systems.

### B2. Source-anchoring bias
- Gold starts with source doc and expands from retrieved neighbors.
- This can over-represent source-local phrasing and under-represent alternate but valid formulations.

### B3. Label-policy mismatch in evaluation
- Strict labels are generated but not used for primary scoring.
- The current primary endpoint therefore reflects looser topical relevance more than exact scope correctness.

### B4. Coverage confounding
- Many queries have no strict gold because corpus does not contain exact in-scope evidence.
- In these cases, low strict hit@k mostly measures corpus incompleteness.

### B5. Metric denominator effects
- Adjusted contamination excludes shared docs from denominator.
- This is valid but can hide operational risk unless reported with denominator and shared composition in parallel.

## 3) Redesign principles

1. Separate retrieval quality from corpus coverage.
2. Separate scope correctness from topical relevance.
3. Eliminate self-referential gold construction where possible.
4. Report enough slices so reviewers can attribute gains to the intended mechanism.

## 4) Proposed dataset generation protocol

### Stage A: Query set design (balanced by intent and observability)

Create fixed strata and lock sample counts before retrieval runs.

- Explicit device
- Implicit device
- Ambiguous
- Explicit equip
- Counterfactual scope traps (topic match but wrong device)

For each stratum, include both:
- Field-derived queries (doc-seeded, current style)
- Operator-style paraphrases (non-seeded natural wording)

Acceptance rule:
- No stratum below 15% of total.
- At least 30% non-seeded wording.

### Stage B: Candidate generation (de-circularized)

For each query, build candidates from three independent channels:

1. System pool channel:
   - Current B0/B1/B2/B3/B4/B4.5/P1 top-k union.
2. Lexical diversification channel:
   - BM25 deep retrieval (k=100) with no scope filter.
3. Metadata expansion channel:
   - Device/equip/topic neighbors from doc metadata graph.

Candidate set for judging is union of all channels with per-channel provenance tags.

Acceptance rule:
- At least 25% of judged candidates must come from non-system-pool channels.

### Stage C: Annotation schema (multi-axis, not single scalar)

For each (query, doc):

- topical_relevance: {0,1,2}
- scope_correctness: {in_scope, shared, family, out_of_scope}
- answer_support: {none, partial, sufficient}
- evidence_confidence: {low, medium, high}
- rationale_short: short justification

Derived labels:
- strict_gold: topical_relevance=2 AND scope_correctness in {in_scope, shared}
- loose_gold: topical_relevance>=1

### Stage D: Split and freeze policy

Use three frozen sets:

- `dev_scope`: tune routing/penalties
- `test_scope`: final scope-policy claim
- `test_coverage`: coverage stress subset (reported separately)

Hard rule:
- Any parameter tuning (thresholds, lambda, filter settings) uses `dev_scope` only.
- `test_scope` evaluated once per snapshot.

## 5) Proposed primary/secondary endpoints

Primary endpoints (align to Paper A claim):

1. `StrictInScopeHit@5` on `test_scope`
2. `OutOfScopeRate@5` on `test_scope` (raw and adjusted, both)

Secondary endpoints:

- `LooseHit@5`, `MRR`
- `SharedFraction@5`, `SharedRelevantRate@5`
- `CoverageAwareHit@5` where denominator excludes queries with no strict candidate in judged pool

Always report:
- query count, strict-eligible query count, shared denominator terms.

## 6) Claim-to-metric mapping (paper narrative safety)

- Claim: "Scope control reduces contamination without catastrophic recall loss"
  - Must pass both:
    - OutOfScopeRate improves vs B3
    - StrictInScopeHit does not drop beyond pre-registered margin

- Claim: "Soft contamination-aware scoring is safer than hard filter"
  - Compare P6/P7 vs B4/B4.5/P1 on identical frozen test and strict endpoints.

If strict-eligible coverage is low, result must be tagged "coverage-limited" and not framed as retrieval failure.

## 7) Minimal implementation plan in this repo

1. Add candidate channel provenance
   - Extend pooling output with `source_channel` tags.
   - Target: `scripts/paper_a/phase3_retrieve_and_pool.py`

2. Add multi-axis annotation schema
   - New judged schema artifact with topical/scope/support/confidence fields.
   - Target: `data/paper_a/rejudge/*` generation scripts.

3. Add strict-primary scoring path
   - Evaluator option: `--gold-mode strict|loose|both`.
   - Target: `scripts/paper_a/evaluate_paper_a_master.py`

4. Add coverage-aware reporting
   - Output strict-eligible counts and separate coverage-limited slice metrics.

5. Freeze split manifests
   - Commit split manifest JSON/CSV for reproducibility.

## 8) Decision gate before writing paper results

Proceed to paper table generation only if all are true:

- strict-eligible query ratio on `test_scope` >= 0.6
- non-system-pool judged candidate ratio >= 0.25
- primary endpoints reported with denominator details
- dev/test leakage check passed

If not, report as "protocol iteration" rather than final claim evidence.
