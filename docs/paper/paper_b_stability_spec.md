# Paper B Spec: Stability-Aware Retrieval (Decision-Complete)

## Goal
Make retrieval *operationally reliable* by treating Top-k stability as a first-class objective, not a side effect of optimizing average accuracy.

This spec freezes:
- stability targets + formulas
- perturbation tiers + guarantees
- constants (k, repeats, group size)
- what is IN/OUT of Paper B

This doc is binding for Paper B work. If it conflicts with other Paper B notes, this doc wins.

## Stability Targets (FIXED)
Definitions:
- For a query run, let ordered Top-k doc IDs be `L_k = [d1, d2, ..., dk]`.
- Let set Top-k doc IDs be `S_k = set(L_k)`.
- For a paraphrase group `g`, let the group queries be `{q_1, ..., q_m}`.

Metrics:
- `RepeatJaccard@10`
  - For a fixed query `q`, run retrieval `N_repeats` times producing sets `S_10^{(i)}(q)`.
  - Compute mean pairwise Jaccard:
    - `J(A,B) = |A ∩ B| / |A ∪ B|`
    - `RepeatJaccard@10(q) = mean_{i<j} J(S_10^{(i)}(q), S_10^{(j)}(q))`
  - Report dataset average: mean over queries.

- `RepeatExactMatch@10`
  - For the same repeats, compute:
    - `RepeatExactMatch@10(q) = mean_{i<j} I(L_10^{(i)}(q) == L_10^{(j)}(q))`

- `ParaphraseJaccard@10`
  - For a group `g`, retrieve once per query and compute:
    - `ParaphraseJaccard@10(g) = mean_{i<j} J(S_10(q_i), S_10(q_j))`

- `ParaphraseExactMatch@10`
  - For a group `g`, compute:
    - `ParaphraseExactMatch@10(g) = mean_{i<j} I(L_10(q_i) == L_10(q_j))`

Effectiveness:
- `hit@5`, `hit@10`: relevant if any `expected_doc_ids` is in Top-k.
- `MRR`: reciprocal rank of first relevant doc (0 if none).

Latency:
- `p95_latency_ms`: p95 over per-request wall-clock times.

## Fixed Constants
- k: 10
- hit_k: {5, 10}
- Repeat runs: `N_repeats = 10`
- Paraphrase group size in synthetic benchmark: 4 queries per group

## Perturbation Tiers (T1-T4)
T1 Repeat (same query, repeated runs)
- What: same query `q`, same process/service, repeated calls.
- Measure: RepeatJaccard@10, RepeatExactMatch@10.
- Guarantee (deterministic mode): MUST be identical ordered Top-10 doc_id list if ES alias target unchanged.

T2 Paraphrase (equivalent queries)
- What: paraphrases within same semantic group.
- Measure: ParaphraseJaccard@10, ParaphraseExactMatch@10.
- Guarantee: none (this is the point of Paper B).

T3 Restart (service restart)
- What: run T1 split by backend restart.
- Measure: same as T1.
- Guarantee (deterministic mode): MUST be identical ordered Top-10 doc_id list if ES alias target unchanged.

T4 Reindex (fresh index build)
- What: rebuild ES index from the same corpus snapshot into a new index version.
- Measure: deltas in effectiveness + stability.
- Guarantee: none. Report observed drift and treat as an instability driver.

## Deterministic Protocol (what it DOES and DOES NOT claim)
We define determinism narrowly (Paper B):
- Deterministic mode is a *procedure* that enforces stable query selection and stable tie-breaking.
- It guarantees stability under T1/T3 only when the underlying ES alias target does not change.

It does NOT guarantee:
- paraphrase stability (T2)
- reindex stability (T4)
- global bitwise determinism across different ES cluster states

Repo hooks that support this:
- stable ES shard routing preference by query hash: `backend/llm_infrastructure/retrieval/engines/es_search.py`
- stable tie-break ordering before final top-k: `backend/llm_infrastructure/llm/langgraph_agent.py` (`retrieve_node`)
- deterministic pipeline step resolution: `backend/services/retrieval_pipeline.py`

## Methods (what counts as a “method” for Paper B)
Minimum 2 levers must be implemented and compared:
1) Deterministic retrieval control
- Use API `deterministic=true` (skip MQ steps; use stable single search query).
- Keep rerank OFF (Paper B scope).

2) Consensus retrieval (stability-aware)
- A fixed-budget method that reduces variance under paraphrase perturbation.
- Allowed implementations:
  - consensus over multiple query variants (e.g., bilingual queries) with deterministic fusion
  - consensus over multiple retriever settings (bounded grid)

Optional (only if it does not derail schedule): stability-regularized reranker learning.

## What Paper B Must NOT Do
- No “multi-agent novelty” claims.
- No hierarchy/scope constraints as contributions (that is Paper A).
- No numeric/condition validator contributions (Paper D).
- No reranking as a variable (keep rerank disabled for Paper B main results).

## Evidence Requirements
All reported numbers must be reproducible from the synthetic benchmark release:
- generator seed + file hashes
- ES index version + alias target
- retrieval config hash and API request payloads

## References (internal)
- Working draft: `docs/paper/paper_b_stability_aware_retrieval.md`
- Shared protocol: `docs/paper/paper_common_protocol.md`
- Stability audit harness reference: `scripts/evaluation/retrieval_stability_audit.py`
