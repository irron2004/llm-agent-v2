# Analysis Draft: Paper A Supervisor Review Plan vs Repo Truth

## Confirmed Facts (repo)
- Legacy evaluator exists and is runnable: `scripts/paper_a/evaluate_paper_a.py`.
  - Eval-set schema is **legacy**: requires `qid/split/query/target_device/gold_doc_ids` and hard-fails on empty `gold_doc_ids` (`scripts/paper_a/evaluate_paper_a.py` `_load_eval_set`).
  - Implemented systems: `B0..B4` and `P1`.
  - `P2/P3/P4` are explicitly skipped with `skip_reason=router_not_implemented`.
- Master eval-set exists but is **not consumable by legacy evaluator** as-is:
  - Path: `data/paper_a/eval/query_gold_master.jsonl`.
  - Observed distribution: `split` is `dev` and `test`; `scope_observability` includes `implicit`, `explicit_device`, `explicit_equip`, `ambiguous`.
  - `gold_doc_ids` appears to be fully populated (no empty lists detected by JSONL scan).
  - Critical nuance: in the current file, `split=test` contains only `scope_observability=explicit_equip` rows (61 rows). Other observability types appear only in `split=dev`.
- Corpus whitelist is enforced end-to-end in retrieval runner:
  - `scripts/paper_a/build_corpus_meta.py` produces `.sisyphus/**/corpus_doc_ids.txt` and includes `es_equip_id` in `doc_meta.jsonl` rows.
  - `scripts/paper_a/retrieval_runner.py` always requires `corpus_doc_ids_path`.
- Policy builder has a hard-coded drift breaker:
  - `scripts/paper_a/build_shared_and_scope.py` currently raises if `shared_topic_count != 13`.
  - It reads `es_equip_id` from `doc_meta.jsonl`, but **does not emit it** into `doc_scope.jsonl` row payload (only `es_doc_id/es_device_name/es_doc_type/topic/is_shared/scope_level`).
- Scope filter DSL exists and is tested:
  - `backend/llm_infrastructure/retrieval/filters/scope_filter.py` (`build_scope_filter_by_doc_ids`, `build_scope_filter_by_fields`).
  - Tests: `backend/tests/test_scope_filter_dsl.py`.

## High-Risk Gaps / Blockers
- Schema drift between master eval-set and runnable evaluator blocks reproducible “B0–P7” claims.
- The current master split assignment prevents reporting test results for `explicit_device/implicit/ambiguous` without regenerating the master set splits.
  - Plan resolution: versioned split regeneration to `data/paper_a/eval/query_gold_master_v0_5.jsonl` with frozen backup `query_gold_master_v0_4_frozen.jsonl`.
- Hard-coded `shared_topic_count==13` blocks reruns under corpus drift and is reviewer-hostile.
- Equip-level evaluation cannot be claimed unless `es_equip_id` is propagated into policy artifacts and used consistently.

## Plan Deltas Already Applied
- Updated `.sisyphus/plans/paper-a-supervisor-review-plan.md` to:
  - Remove corpus-coupled hard-fail on shared_topic_count (Task 3 requirement).
  - Add a versioned master-split regeneration protocol (Task 2): freeze v0.4 and generate `query_gold_master_v0_5.jsonl` with balanced dev/test across all `scope_observability`.
  - Align evidence runs to master `scope_observability` slices.
  - Add a consistency-audit requirement to reconcile spec vs actual dataset counts.

## Pending External/Deep Research
- Related work + baseline candidates for metadata-constrained retrieval and contamination metrics.
- Matryoshka Representation Learning: confirm correct claims + fair baselines; verify whether repo embedding stack supports MRL.

## Next: Incorporate Architect + Explore/Librarian Findings
- When bg tasks complete: update this draft into a finalized analysis note and (if needed) patch the plan.
