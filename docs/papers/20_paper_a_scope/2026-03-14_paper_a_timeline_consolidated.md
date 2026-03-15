# Paper A Consolidated Timeline

Date: 2026-03-14
Status: timeline and source-of-truth guide for the full Paper A document cluster

## Reading Guide

This document consolidates the Paper A document set into one timeline-oriented guide.
It does not replace the source documents. Instead, it explains:

- how the Paper A narrative evolved,
- which documents are foundational vs operational vs stale,
- where the major pivots happened,
- what the current defensible state is,
- and what remains blocked.

Use this file as the canonical index and reading guide before reading individual Paper A documents.

## What This Document Does Not Do

- It does not replace or delete the underlying Paper A source documents.
- It does not rewrite older drafts in place.
- It does not treat every later document as automatically better; instead, it explains which documents are stronger for which purpose.
- It is intentionally link-first so that experimental provenance stays in the dated evidence files.

## One-Line Summary

Paper A started as a hierarchy-aware scope-routing paper, went through a negative v0.5-era evaluation cycle that suggested hard filtering destroyed recall, then pivoted on 2026-03-12 to a debiased masked-query evaluation protocol, after which the central claim changed: oracle device filtering can eliminate contamination and improve recall under masked evaluation. At the same time, masked evaluation is itself a proxy setting, and realistic deployment still depends on parser quality, gold audit caveats, and mixed-scope evaluation limits.

## Document Families

### 1) Orientation and scope-definition docs

These define what Paper A is supposed to be.

| Path | Role | Current reading |
|---|---|---|
| `docs/papers/20_paper_a_scope/README.md` | High-level overview and original positioning | Good entry summary, but earlier than the latest evidence pivot |
| `docs/papers/20_paper_a_scope/paper_a_series_map.md` | Defines Paper A vs A-1 vs A-2 | Still useful |
| `docs/papers/20_paper_a_scope/paper_a_series_blueprint.md` | Research blueprint for the paper series | Still useful for scope boundaries |
| `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` | Formal experiment design/spec v0.6 | Important for intended design, but not the latest result narrative |
| `docs/papers/20_paper_a_scope/related_work.md` | Literature scaffold | Supporting only |
| `docs/papers/20_paper_a_scope/references.bib` | Bibliography | Reference only |

### 2) Main manuscript-state docs

These represent paper-writing states, not necessarily the latest validated evidence state.

| Path | Role | Current reading |
|---|---|---|
| `docs/papers/20_paper_a_scope/paper_a_scope.md` | Older manuscript centered on v0.5/v0.5-like narrative | Partly stale; still useful for method framing |
| `docs/papers/20_paper_a_scope/paper_a_draft_v2.md` | Newer draft centered on masked-query breakthrough | Closest to current thesis, but still needs wording guardrails |

### 3) Evidence and experiment docs

These are the most important documents for reconstructing what actually happened.

| Path | Role |
|---|---|
| `docs/papers/20_paper_a_scope/evidence/2026-01-08_meta_guided_hierarchical_rag.md` | Very early upstream design ancestor; broader than current Paper A |
| `docs/papers/20_paper_a_scope/evidence/2026-02-12_gcb_equip_id_matching_report.md` | Equip-ID matching feasibility for GCB documents |
| `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md` | Corpus/device/shared-topic statistics |
| `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md` | Run index for early evaluation round |
| `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md` | Early error analysis |
| `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_main_results.md` | Early main results on v0.5-era evaluation |
| `docs/papers/20_paper_a_scope/evidence/2026-03-09_gold_rejudging_analysis.md` | Rejudging/alias-fix era analysis of earlier results |
| `docs/papers/20_paper_a_scope/evidence/2026-03-12_cross_device_topic_feasibility.md` | Shared/cross-device topic feasibility analysis |
| `docs/papers/20_paper_a_scope/evidence/2026-03-12_dataset_protocol_redesign.md` | Protocol redesign after diagnosing evaluation bias |
| `docs/papers/20_paper_a_scope/evidence/2026-03-12_slot_valve_hard_filter_recall_loss.md` | Targeted failure-case analysis under older narrative |
| `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md` | Breakthrough summary after masked-query BM25 experiment |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_full_experiment_results.md` | Master evidence snapshot combining old and new results |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md` | Oracle vs realistic parser gap, including equip-aware mode |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md` | Packaged v0.6 gold reliability audit summary |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md` | Mixed-scope eval restoration and split rebuild |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_implicit_eval.md` | Blocked implicit eval attempt due to index dimension mismatch |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md` | Diagnosis of why B4.5 underperforms B4 |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md` | Hybrid/rerank masked experiment recovery |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md` | Negative result for soft scoring in the new setup |
| `docs/papers/20_paper_a_scope/evidence/2026-03-14_remaining_tasks.md` | Post-breakthrough backlog |

### 4) Review and audit docs

These are internal critical-reading documents. They matter because they explain why some older claims should not be trusted at face value.

| Path | Role |
|---|---|
| `docs/papers/20_paper_a_scope/review/preregistration.md` | Pre-registered hypothesis framing |
| `docs/papers/20_paper_a_scope/review/hypotheses_experiments.md` | Hypothesis-by-hypothesis experiment plan |
| `docs/papers/20_paper_a_scope/review/reviewer_report.md` | Internal reviewer critique of the early paper state |
| `docs/papers/20_paper_a_scope/review/consistency_audit.md` | Audit of doc/data drift and stale claims |

### 5) Mapping and operations docs

These connect claims to evidence or define what to do next.

| Path | Role | Current reading |
|---|---|---|
| `docs/papers/20_paper_a_scope/evidence_mapping.md` | Claim-to-evidence matrix | Useful as a pre-pivot claim map, but needs re-sync with the masked-query narrative |
| `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md` | Current execution priority document | Current operational source of truth |

### 6) Task and execution-log docs outside the Paper A folder

These are not part of the manuscript, but they matter if you want to know what was actually executed in the repo.

| Path | Role | Current reading |
|---|---|---|
| `docs/tasks/TASK-20260314-paper-a-execution-phase0.md` | Execution log for the 2026-03-14 Paper A implementation/evidence run | Useful operational companion to `2026-03-14_execution_tasks.md` |
| `docs/tasks/TASK-20260314-paper-a-doc-consolidation.md` | Task log for creating this consolidated timeline document | Meta only |

## Timeline

### 2026-01-08 — Broad architectural ancestor

- `docs/papers/20_paper_a_scope/evidence/2026-01-08_meta_guided_hierarchical_rag.md`
- This is not yet the present Paper A. It is a broader meta-guided hierarchical RAG concept that predates the narrowed contamination-control story.
- Historical value: it shows the origin of the routing-centric worldview and why later Paper A includes scope, hierarchy, and routing language.

### 2026-02-12 — Equip-ID feasibility becomes concrete

- `docs/papers/20_paper_a_scope/evidence/2026-02-12_gcb_equip_id_matching_report.md`
- Key finding: `GCB_number + Title` can recover valid `Equip_ID` for a large portion of GCB artifacts.
- Historical value: this is where equip-level scope handling starts to look practical rather than purely conceptual.

### 2026-03-04 — Corpus shape is quantified

- `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md`
- Key findings:
  - 578 docs,
  - strong SUPRA dominance,
  - shared topics exist but are not ubiquitous,
  - topic-based family/shared logic is plausible.
- Historical value: this document justifies why contamination and shared-policy analysis matter in the first place.

### 2026-03-05 — First serious evaluation round, but still under old protocol

Core docs:

- `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_main_results.md`
- `docs/papers/20_paper_a_scope/review/reviewer_report.md`
- `docs/papers/20_paper_a_scope/review/preregistration.md`
- `docs/papers/20_paper_a_scope/review/hypotheses_experiments.md`
- `docs/papers/20_paper_a_scope/review/consistency_audit.md`
- `docs/papers/20_paper_a_scope/evidence_mapping.md`

What this phase believed:

- Hard filtering reduced contamination but often looked bad for recall.
- Shared/scope policies looked unstable or tautological.
- Implicit/equip slices were small and fragile.

Why this phase is important:

- It generated the first serious negative evidence.
- It also generated the strongest self-critique, which later made the protocol pivot possible.

Why it is not the current truth:

- The evaluation set was small.
- Parser alias issues and shared-metric tautology complicated interpretation.
- The document-centric evaluation protocol had not yet been diagnosed as biased.

### 2026-03-09 — Rejudging and post-alias-fix refinement

- `docs/papers/20_paper_a_scope/evidence/2026-03-09_gold_rejudging_analysis.md`
- This phase re-examined results under revised judging and alias normalization.
- It improved analysis quality, but it still lived inside the older evaluation worldview.
- Historical value: transitional, not final.

### 2026-03-12 — The protocol pivot

Core docs:

- `docs/papers/20_paper_a_scope/evidence/2026-03-12_dataset_protocol_redesign.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-12_cross_device_topic_feasibility.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-12_slot_valve_hard_filter_recall_loss.md`

This is the real turning point.

What changed:

- The team diagnosed circular gold bias and lexical leakage.
- `question_masked` was introduced to break the shortcut where device-bearing queries let BM25 find the right docs without any genuine scope reasoning.
- The project moved from “does filtering hurt recall?” to “was the old protocol incapable of measuring filtering correctly?”

This date marks the transition from the old Paper A narrative to the current masked-query narrative.

### 2026-03-13 — Breakthrough via masked-query BM25

- `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`

This document is the shortest clean summary of the breakthrough.

Key claims introduced here:

- contamination is severe,
- oracle device filtering can reduce contamination to near-zero,
- under masked evaluation, filtering can improve recall rather than destroy it,
- earlier negative results were largely artifacts of biased evaluation.

This is the moment where the thesis stops being “filtering seems harmful” and becomes “filtering was being measured under the wrong protocol.”

### 2026-03-14 — Consolidation, realism checks, and operationalization

Core docs:

- `docs/papers/20_paper_a_scope/evidence/2026-03-14_full_experiment_results.md`
- `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_remaining_tasks.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_implicit_eval.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`
- `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`
- `docs/tasks/TASK-20260314-paper-a-execution-phase0.md`

What 2026-03-14 accomplishes:

- converts the masked-query breakthrough into a full evidence package,
- quantifies oracle vs realistic parser gap,
- packages gold reliability as a sampled audit,
- restores mixed-scope dataset coverage,
- explains the `B4.5 < B4` paradox,
- and confirms that the earlier P6/P7 soft-scoring optimism does not hold in the new setup.

What is still blocked even after this round:

- full implicit mixed-scope hybrid evaluation is blocked by index embedding dimension mismatch,
- manuscript wording still needs guardrails so oracle upper bounds are not mistaken for deployment performance,
- ambiguous rows still cannot support pooled hit-based claims because of empty-gold coverage.

### 2026-03-15 — Metric synchronization and claim guardrails

- `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
- This update synchronized the masked BM25 hit table to the per-query artifact
  `data/paper_a/trap_masked_results.json`.
- Synchronized values (ALL, loose):
  - `B0_masked`: `287/578 (50%)`
  - `B4_masked`: `530/578 (92%)`
  - `B4.5_masked`: `430/578 (74%)`
- Directional thesis did not change (filtering still strongly improves contamination and hit under masked BM25), but this date matters because it closes a numeric inconsistency in the narrative layer.

## Decision Log (Adopted / Rejected / Deferred)

| Date | Decision | Status | Evidence anchor |
|---|---|---|---|
| 2026-03-12 | Debiased masked-query protocol replaces old document-seeded eval as primary evidence route | **Adopted** | `2026-03-12_dataset_protocol_redesign.md` |
| 2026-03-14 | Oracle B4 is reported as **upper bound**, not deployment performance | **Adopted** | `2026-03-14_oracle_vs_parser_gap.md` |
| 2026-03-14 | Naive shared inclusion (`B4.5`) is not default policy in current package | **Rejected (for now)** | `2026-03-14_b45_failure_decomposition.md` |
| 2026-03-14 | P6/P7 soft scoring is not promoted as main method in current setup | **Rejected (for now)** | `2026-03-14_masked_p6p7_reexperiment.md` |
| 2026-03-14 | Full BSP/Bayesian routing is not part of Paper A core claims | **Deferred** | `paper_a_scope_spec.md` + execution/evidence package |

## Major Narrative Pivots

### Pivot 1 — From broad routing architecture to focused contamination paper

- Early design energy was broader and more architectural.
- Paper A later narrowed into a retrieval-safety story centered on contamination, scope, and evaluation protocol.

### Pivot 2 — From “hard filter hurts recall” to “old evaluation hid filter value”

- Old narrative: `2026-03-05_paper_a_main_results.md`, parts of `paper_a_scope.md`, and review docs.
- New narrative: `2026-03-12_dataset_protocol_redesign.md` + `2026-03-13_paper_a_progress_summary.md` + `paper_a_draft_v2.md`.

This is the single most important transition in the entire document set.

### Pivot 3 — From oracle-only optimism to realism with caveats

- `2026-03-14_oracle_vs_parser_gap.md` matters because it prevents overclaim.
- It shows that device-only parsing creates a large contamination gap, while equip-aware realistic mode narrows that gap substantially.
- This means Paper A can no longer stop at “B4 works”; it has to say “oracle B4 is an upper bound, realistic routing quality depends on parsing.”

### Pivot 4 — From “shared should help” to “shared needs careful ordering”

- Earlier documents positioned shared/family relaxation as a recall recovery mechanism.
- `2026-03-14_b45_failure_decomposition.md` and `2026-03-14_hybrid_rerank_recovery.md` show that naive shared inclusion can hurt recall.
- The shared policy survives as a diagnosis-and-fix area, not as a proven win in the current package.

### Pivot 5 — From “soft scoring might beat hard filtering” to negative result

- Earlier material left room for P6/P7 optimism.
- `2026-03-14_masked_p6p7_reexperiment.md` closes that door for the current setup.

## Which Documents Are Most Trustworthy Right Now

### Current operational and evidence source of truth

Read these first if you want the current state:

1. `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
2. `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md`
3. `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md`
4. `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md`
5. `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md`
6. `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md`
7. `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md`
8. `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`

### Best current manuscript starting point

- `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`

### Best design/spec reference

- `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`

### Best critical self-audit docs

- `docs/papers/20_paper_a_scope/review/consistency_audit.md`
- `docs/papers/20_paper_a_scope/review/reviewer_report.md`

## Documents That Need Caution

### `paper_a_scope.md`

- Valuable for method framing and older evaluation protocol description.
- Not safe as the sole “current state” document because it still reflects the old small-split / planned-router / pre-breakthrough narrative in multiple places.

### `evidence_mapping.md`

- Useful for understanding intended claim structure.
- Not a reliable status dashboard because some items were historically marked missing even when artifacts already existed.

### `2026-03-14_full_experiment_results.md`

- Extremely useful as a master snapshot.
- But it mixes completed evidence, interpreted takeaways, and future-work placeholders. It should be read as a synthesis notebook, not as a clean final paper text.

### `2026-03-14_remaining_tasks.md`

- Good historical backlog snapshot.
- Best read as an earlier same-day backlog snapshot. For execution order, prefer `2026-03-14_execution_tasks.md`.

## Known Numeric Drift Across Documents

These are not necessarily mistakes, but they must be read with document context.

### Corpus size drift: `578` vs `508`

- `578` appears in corpus/evidence-oriented documents such as `docs/papers/20_paper_a_scope/evidence/2026-03-04_corpus_statistics.md` and `docs/papers/20_paper_a_scope/README.md`.
- `508` appears in `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`.
- Until the manuscript is synchronized, treat these as document-context-specific counts rather than assuming one is simply wrong.

### Shared-document count drift: `124` vs `60`

- `124` appears in the older manuscript framing in `docs/papers/20_paper_a_scope/paper_a_scope.md`.
- `60` appears in the 2026-03-14 evidence package and newer manuscript draft, including `docs/papers/20_paper_a_scope/evidence/2026-03-14_full_experiment_results.md` and `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`.
- Treat this as one of the strongest signs that older narrative docs and current evidence are not yet fully synchronized.

### Masked loose-hit drift: `530/578` vs `532/578` (and related paired values)

- `530/578 (92%)` appears in the synchronized 2026-03-15 state of
  `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
  (driven by `data/paper_a/trap_masked_results.json`).
- `532/578 (92%)` appears in the 2026-03-14 hybrid package context
  (`docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md` and related docs).
- Treat this as **run-family context drift** unless and until one unified canonical scoreboard is explicitly declared for the manuscript body.

## Current State

As of the 2026-03-14 evidence package, the most defensible Paper A state is:

- Strongly supported:
  - cross-equipment contamination is severe,
  - oracle device filtering can nearly eliminate contamination,
  - masked-query evaluation reveals that old document-seeded evaluation understated filtering value,
  - P6/P7 soft scoring does not currently beat hard filtering.
- Supported with caveat:
  - realistic deployment can approach oracle hit rates only when parser quality is handled explicitly; device-only parser results are misleading if contamination is not reported alongside hit.
- Restored but not fully exploited:
  - mixed-scope dataset coverage (`explicit_device`, `explicit_equip`, `implicit`, `ambiguous`) has been restored at the dataset level.
- Still blocked:
  - full mixed implicit hybrid evaluation,
  - fully rigorous raw-annotation-backed gold audit with confidence intervals,
  - clean manuscript-wide synchronization of all old and new claims.

## As-Of Canonical Metrics (2026-03-15 sync)

These are the recommended headline numbers for the current narrative layer.

| Metric | Value | Source |
|---|---:|---|
| BM25 masked `B0` contamination@10 (ALL) | `0.518` | `2026-03-13_paper_a_progress_summary.md` |
| BM25 masked `B4` contamination@10 (ALL) | `0.000` | `2026-03-13_paper_a_progress_summary.md` |
| BM25 masked `B0` gold hit@10 loose (ALL) | `287/578 (50%)` | `2026-03-13_paper_a_progress_summary.md` |
| BM25 masked `B4` gold hit@10 loose (ALL) | `530/578 (92%)` | `2026-03-13_paper_a_progress_summary.md` |
| Real parser adj_cont@10 (device-only parser) | `30.6%` | `2026-03-14_oracle_vs_parser_gap.md` |
| Scope-aware realistic parser adj_cont@10 | `8.5%` | `2026-03-14_oracle_vs_parser_gap.md` |
| v0.6 strict precision (sample audit) | `97.2% (172/177)` | `2026-03-14_v06_gold_audit.md` |

## Index Constraint Snapshot (why some evaluations stay blocked)

As observed in the current local ES setup:

| Index | Text/doc_id for lexical eval | Embedding | `equip_id` usable | Practical implication |
|---|---|---|---|---|
| `chunk_v3_content` | Yes | No | Yes | Strong for BM25/scope metadata analysis; no native dense retrieval |
| `chunk_v3_embed_bge_m3_v1` | Limited metadata view | Yes (1024) | No | Dense retrieval available, but equip-aware metadata evaluation is constrained |
| `rag_chunks_dev_v2` | Yes | Yes (legacy dims) | Yes | Useful integrated fallback, but not fully aligned with v3 embedding setup |

This table explains recurring references to "index mismatch" in 2026-03-14 evidence docs.

## Artifact Integrity Map (claim -> evidence -> machine artifact)

| Claim family | Primary evidence doc | Machine-readable artifact(s) |
|---|---|---|
| Masked BM25 contamination/hit breakthrough | `2026-03-13_paper_a_progress_summary.md` | `data/paper_a/trap_masked_results.json` |
| Oracle vs realistic parser gap | `2026-03-14_oracle_vs_parser_gap.md` | `data/paper_a/parser_accuracy_report.json`, `data/paper_a/parser_accuracy_per_query_diff.csv` |
| v0.6 gold reliability | `2026-03-14_v06_gold_audit.md` | `data/paper_a/gold_verification_report.json` |
| `B4.5 < B4` paradox diagnosis | `2026-03-14_b45_failure_decomposition.md` | `data/paper_a/masked_hybrid_results.json` |
| P6/P7 negative result in masked setup | `2026-03-14_masked_p6p7_reexperiment.md` | `data/paper_a/masked_p6p7_results.json` |

## Recommended Canonical Reading Order

If you want to understand Paper A quickly and correctly, use this order:

1. `docs/papers/20_paper_a_scope/README.md`
2. `docs/papers/20_paper_a_scope/paper_a_series_map.md`
3. `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`
4. `docs/papers/20_paper_a_scope/review/consistency_audit.md`
5. `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
6. `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md`
7. `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md`
8. `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md`
9. `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md`
10. `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md`
11. `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md`
12. `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`
13. `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`

## Suggested Use Cases

### If you are writing the paper body

- Use `paper_a_draft_v2.md` as the writing base.
- Use the 2026-03-14 evidence docs to tighten claims.
- Do not use `paper_a_scope.md` as the final truth source without cross-checking.

### If you are deciding what to implement next

- Use `2026-03-14_execution_tasks.md` first.
- Then check whether the relevant evidence doc already exists before treating a task as unfinished.

### If you are auditing whether a claim is stale

- Compare `paper_a_scope.md` and older 2026-03-05 / 2026-03-09 evidence against:
  - `2026-03-12_dataset_protocol_redesign.md`
  - `2026-03-13_paper_a_progress_summary.md`
  - the 2026-03-14 evidence package.

## Bottom Line

The Paper A document set is not just a collection of notes. It records a genuine narrative reversal:

- early evidence seemed to say scope filtering hurt recall,
- later protocol analysis showed that conclusion was largely an artifact,
- the current evidence package supports a stronger contamination-control thesis,
- but only if oracle-vs-realistic distinctions, gold-audit caveats, and mixed-scope coverage limits are stated explicitly.

That is the most important thing this timeline is meant to preserve.
