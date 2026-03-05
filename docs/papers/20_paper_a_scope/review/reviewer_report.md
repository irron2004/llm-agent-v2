# Paper A: Reviewer Report

> Date: 2026-03-05
> Paper: "Hierarchy-aware Scope Routing for Cross-Equipment Contamination Control in Industrial Maintenance RAG"

---

## Fatal Flaws

### R1. [FATAL] ZEDIUS XP / SUPRA_XP alias mismatch invalidates B4 results
- **Issue**: 16/22 explicit_device queries have `parsed_hard_devices=ZEDIUS XP` but `allowed_devices=SUPRA_XP`. The auto-parser extracts the wrong device name, so B4's filter operates on incorrect scope for 73% of test queries.
- **Evidence**: B4 hit@5=0.091 (2/22) but 8 of the 20 misses are caused by alias mismatch, not by the filter being too restrictive. B4's adj_cont@5=0.027 is artificially low because wrong-scope filtering coincidentally excludes contamination.
- **Mitigation**: Add device alias normalization to parser. Re-run B4 with fix. Report both pre-fix and post-fix results for transparency.

### R2. [FATAL] P1 adj_cont@5=0.000 is tautological
- **Issue**: Adjusted contamination excludes shared docs. P1's scope-level filter lets through only shared docs and in-scope device docs. When shared docs fill top-k (shared@5=0.882 on explicit_device), adj_cont@5 mechanically reaches zero regardless of retrieval quality.
- **Evidence**: P1 raw_cont@5=0.264 shows that 26% of P1's top-5 results are actually from out-of-scope devices — they just happen to be classified as "shared." The metric definition, not the retrieval quality, produces the zero.
- **Mitigation**: Report raw_cont@5 alongside adj_cont@5. Add a metric for "shared doc relevance" to verify that included shared docs are actually useful, not just noise.

### R3. [FATAL] 41% of explicit_device test queries have zero recall across ALL systems
- **Issue**: 9/22 queries in the primary test slice have hit@5=0 for B0-P1. These represent gold docs not present in the indexed corpus or fundamental retrieval failures.
- **Evidence**: These queries contribute only to contamination metrics, not recall. System comparisons on hit@5 are based on only 13 evaluable queries, reducing effective sample size further.
- **Mitigation**: Verify gold docs are in corpus (check `in_corpus_top_docs` field). Report conditional metrics excluding F4 queries. Acknowledge reduced effective n.

---

## Major Concerns

### R4. [MAJOR] Sample sizes are critically small
- **Issue**: n=22 (explicit_device), n=21 (implicit), n=8 (explicit_equip). Bootstrap CIs are very wide (e.g., hit@5 delta CI: [-0.545, -0.091]). The equip slice is below minimum reporting threshold.
- **Mitigation**: Expand evaluation set. At minimum, report power analysis showing what effect sizes are detectable at n=22.

### R5. [MAJOR] P1 causes severe recall loss on implicit queries (hit@5=0.000)
- **Issue**: P1 retrieves zero gold docs on all 21 implicit queries. The shared-doc-dominant retrieval strategy fails completely when no device-specific docs are in scope.
- **Evidence**: P1 shared@5=0.905 on implicit — all results are shared docs, none are relevant.
- **Mitigation**: P1 should not be deployed for implicit queries. Acknowledge this limitation. The planned router (P2-P4) is meant to address this but is not evaluated.

### R6. [MAJOR] B4=B3 on implicit slice — hard filter adds nothing
- **Issue**: When the parser cannot extract a device, B4 falls back to global retrieval (identical to B3). This means B4 only helps when the parser succeeds, which is already the "easy" case.
- **Evidence**: B4 adj_cont@5=0.600, hit@5=0.286 on implicit — identical to B3.
- **Mitigation**: Acknowledge this is by design. The paper should clearly state that B4's value proposition is limited to explicit queries and that the router is needed for implicit ones.

### R7. [MAJOR] No sensitivity analysis for shared threshold T=3 and family threshold tau=0.2
- **Issue**: Both T and tau are presented as defaults with no justification or ablation.
- **Mitigation**: Run T={2,3,4,5} and tau={0.1,0.2,0.3,0.4} sweeps on dev split. Report impact on shared doc count, family sizes, and downstream metrics.

### R8. [MAJOR] Missing ablations in the system matrix
- **Issue**: No system tests shared-only without equip-level filter (B4+shared). No family-only system. P1 bundles three changes (shared + device filter + equip filter) without isolating each contribution.
- **Mitigation**: Add C1: B4+shared (no equip distinction), C2: device filter + family expansion (no shared). These isolate the marginal value of each component.

### R9. [MAJOR] Reranker model is null in run manifest
- **Issue**: `run_manifest.json` shows `reranker_model: null`. B3 and B4 are described as using cross-encoder reranking, but the manifest suggests no reranker was applied.
- **Mitigation**: Verify whether the evaluator actually applies reranking. If not, B3=B2+rerank claim is false and the system descriptions need correction.

---

## Minor Issues

### R10. [MINOR] Ambiguous slice is a dead evaluation path
- **Issue**: 80 ambiguous queries, all with empty gold_doc_ids, all skipped. This slice contributes zero quantitative evidence.
- **Mitigation**: Either label gold docs for ambiguous queries or remove from the evaluation set. Report as "descriptive analysis only" if kept.

### R11. [MINOR] Single domain limits external validity
- **Issue**: All data from one semiconductor fab. Equipment hierarchy, shared topic patterns, and contamination rates may not generalize.
- **Mitigation**: Discuss as limitation. Frame contributions as methodology + framework, not universal claims.

### R12. [MINOR] Gold doc labeling process not documented
- **Issue**: How were gold_doc_ids assigned? Single annotator? Inter-annotator agreement? Were annotators shown the query in isolation or with equipment context?
- **Mitigation**: Document annotation protocol. Report IAA if multiple annotators.

### R13. [MINOR] H10 result contradicts expected direction
- **Issue**: B1 (dense) adj_cont@5=0.209 < B0 (BM25) adj_cont@5=0.282. The paper expected dense retrieval to have higher contamination, but BM25 is worse.
- **Mitigation**: Discuss why — likely shared industrial vocabulary causes BM25 keyword matches across devices. This is actually an interesting finding worth highlighting.

### R14. [MINOR] Latency not compared across systems
- **Issue**: per_query.csv contains `latency_ms` but no latency comparison is reported. B1 (dense) shows ~3000ms vs B0 (BM25) ~35ms — a 100x difference.
- **Mitigation**: Add latency table. Especially relevant for P1 which adds filter construction overhead.

### R15. [MINOR] P2-P7 occupy half the system matrix but are "planned-not-reported"
- **Issue**: System matrix lists 7 planned systems alongside 6 evaluated ones. This may appear as padding.
- **Mitigation**: Move planned systems to a "Future Work" table. Keep main system matrix focused on evaluated systems.

### R16. [MINOR] Novelty positioning needs strengthening
- **Issue**: Scope-level-aware filtering is essentially metadata filtering with a shared-doc exception. The novelty over standard faceted search is incremental.
- **Mitigation**: Emphasize the hierarchy-awareness (device vs equip granularity), the topic-sharing graph for family construction, and the shared doc classification as the novel contributions. Compare explicitly to standard metadata filtering baselines.

---

## Strengths

### S1. Pre-registration and reproducibility
- Evaluation protocol, statistical tests, and hypotheses documented before runs. SHA-256 hashes for all artifacts. Bootstrap + McNemar with Holm correction is rigorous.

### S2. Contamination metric decomposition
- The raw_cont/adj_cont/shared@k decomposition is well-designed and provides transparency about what the shared policy actually does.

### S3. Error analysis reveals actionable findings
- The ZEDIUS XP alias discovery and shared over-inclusion pattern are practically valuable. The F1-F7 taxonomy is systematic.

### S4. Honest limitation reporting
- Hypothesis verdicts include "Inconclusive" and "Cannot evaluate" rather than overclaiming. Ambiguous slice limitations acknowledged.

---

## Verdict

**Recommendation: Major Revision**

The core idea (hierarchy-aware scope filtering) is sound and practically valuable, but the current evaluation has three fatal flaws that must be fixed before the results are interpretable:

### Top 3 Actions Before Submission

1. **Fix the ZEDIUS XP alias** in the parser and re-run all B4/P1 evidence. Current B4 results are not a fair test of hard filtering — they're a test of a broken parser. This alone would likely change the hit@5 story dramatically.

2. **Address the P1 tautology**: Either (a) define a stricter adj_cont that doesn't auto-exempt shared docs, or (b) add a shared-doc relevance metric showing that the shared docs P1 retrieves are actually useful, not just filling slots.

3. **Expand the evaluation set** or report conditional metrics: With 9/22 queries having zero recall across all systems, the effective sample for recall comparison is n=13. This needs to be transparently reported, and ideally the evaluation set should be expanded.
