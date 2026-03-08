# Consistency Ledger: Legacy Plans vs Current Repo

## Summary

**Generated:** 2026-03-08

| Metric | Count |
|--------|-------|
| Total legacy plans reviewed | 10 |
| Known stale references (confirmed) | 1 |
| Completed legacy plan scopes | 3 |
| Highest-risk mismatches | 5 |

### Top 5 Highest-Risk Mismatches

1. **`tests/api/test_agent_rrf_and_sticky_gates.py`** - Referenced in unified plan but does NOT exist. Replaced by:
   - `tests/api/test_agent_stage2_retrieval.py`
   - `tests/api/test_agent_sticky_policy_followup_only.py`
   - `tests/api/test_agent_interrupt_resume_regression.py`

2. **`.omc/plans/chunk_v3_embed_ingest_plan.md`** vs `.sisyphus/plans/.legacy/chunk_v3_embed_ingest_plan.md` - Dual runbook references; unified plan should treat `.omc` as runbook and `.legacy` as source spec.

3. **`.omc/plans/ui-chat-improvements-v2.md`** - Legacy original request reference; plan itself exists in `.omc/plans/`.

4. **`normalize.py`** - Referenced at repo root in legacy plan; exists at root (`normalize.py`).

5. **Doc-type canonicalization paths** - Legacy plan references `scripts/chunk_v3/normalize.py` but current correct path is repo root `normalize.py`.

---

## Per-Legacy-Plan Details

### 1) `.sisyphus/plans/.legacy/ui-autoparse-confirm-task-mode-2026-03-05.md`

**Status:** Completed (per unified plan)

**Pending checkbox items:** None (all completed)

**References found:**
| Reference | Classification | Current Status |
|-----------|---------------|----------------|
| `backend/api/routers/agent.py` | exists | Verified |
| Tests for autoparse | exists | Multiple test files exist |

**Proposed replacements:** N/A

---

### 2) `.sisyphus/plans/.legacy/ui-chat-improvements-v2.md`

**Status:** Partially complete

**Pending checkbox items:**
- [ ] 6. REQ-6 tabbed guided-selection UI (line 164)

**References found:**
| Reference | Classification | Current Status |
|-----------|---------------|----------------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | exists | Verified |
| `frontend/src/features/chat/components/guided-selection-panel.tsx` | exists | Verified |
| `frontend/src/features/chat/pages/chat-page.tsx` | exists | Verified |
| `frontend/src/features/chat/hooks/use-chat-session.ts` | exists | Verified |
| `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx` | exists | Verified |
| `.omc/plans/ui-chat-improvements-v2.md` | non-path | Historical reference |

**Proposed replacements:** N/A (all current paths valid)

---

### 3) `.sisyphus/plans/.legacy/chunk_v3_embed_ingest_plan.md`

**Status:** Large active backlog

**Pending checkbox items:** 19 pending tasks (lines 113-1049)

**References found:**
| Reference | Classification | Current Status |
|-----------|---------------|----------------|
| `scripts/chunk_v3/run_chunking.py` | exists | Verified |
| `scripts/chunk_v3/chunkers.py` | exists | Verified |
| `scripts/chunk_v3/run_embedding.py` | exists | Verified |
| `scripts/chunk_v3/run_ingest.py` | exists | Verified |
| `scripts/chunk_v3/validate_vlm.py` | exists | Verified |
| `backend/llm_infrastructure/elasticsearch/mappings.py` | exists | Verified |
| `normalize.py` (repo root) | exists | Verified |
| `docs/2026-03-04_doc_type_chunking_plan.md` | exists | Verified |
| `.omc/plans/chunk_v3_embed_ingest_plan.md` | non-path | Runbook reference |

**Proposed replacements:**
- Legacy ref `scripts/chunk_v3/normalize.py` → use `normalize.py` at repo root

---

### 4) `.sisyphus/plans/.legacy/chapter-grouping-retrieval.md`

**Status:** Phase-2 style follow-up remains

**Pending checkbox items:**
- [ ] 5. Wire section metadata into chunk creation (line 290)
- [ ] 10. Operational plan for backfill/reindex (line 512)

**References found:**
| Reference | Classification | Current Status |
|-----------|---------------|----------------|
| `scripts/chunk_v3/section_extractor.py` | exists | Verified |
| `scripts/chunk_v3/chunkers.py` | exists | Verified |
| `scripts/chunk_v3/run_ingest.py` | exists | Verified |
| `backend/llm_infrastructure/retrieval/engines/es_search.py` | exists | Verified |
| `backend/services/search_service.py` | exists | Verified |
| `backend/llm_infrastructure/retrieval/postprocessors/section_expander.py` | exists | Verified |
| `docs/2026-03-07-Chapter-Grouping-Retrieval.md` | exists | Verified |

**Proposed replacements:** N/A (paths correct)

---

### 5) `.sisyphus/plans/.legacy/agent-retrieval-followups-2026-03-04.md`

**Status:** Active backlog

**Pending checkbox items:**
- [ ] 5. Implement Stage2 Retrieval (line 233)
- [ ] 6. Integration tests for Stage2 (line 277)
- [ ] 7. MQ Mode sweep (line 314)
- [ ] 8. SOP soft boost default update (line 347)
- [ ] 9. Update docs (line 378)
- [ ] 11. Freeze sticky scope policy (line 439)
- [ ] 12. Normalize eval JSONL (line 480)
- [ ] 13. HTTP-mode quality gate (line 513)
- [ ] 14. Final integrated gate (line 551)

**References found:**
| Reference | Classification | Current Status |
|-----------|---------------|----------------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | exists | Verified |
| `backend/config/settings.py` | exists | Verified |
| `backend/llm_infrastructure/retrieval/engines/es_search.py` | exists | Verified |
| `scripts/evaluation/evaluate_sop_agent_page_hit.py` | exists | Verified |
| `tests/api/test_agent_rrf_and_sticky_gates.py` | missing | **DOES NOT EXIST** |

**Proposed replacements:**
- `tests/api/test_agent_rrf_and_sticky_gates.py` → `tests/api/test_agent_stage2_retrieval.py`, `tests/api/test_agent_sticky_policy_followup_only.py`, `tests/api/test_agent_interrupt_resume_regression.py`

---

### 6) `.sisyphus/plans/.legacy/before-after-regression-compare.md`

**Status:** Active backlog

**Pending checkbox items:** 9 total (tasks 1-9, lines 95-762)

**References found:**
| Reference | Classification | Current Status |
|-----------|---------------|----------------|
| `scripts/evaluation/regression_compare_manifest.py` | exists | Verified |
| `scripts/paper_b/run_paper_b_eval.py` | exists | Verified |
| `scripts/evaluation/run_agent_regression.py` | exists | Verified |
| `scripts/evaluation/compare_before_after.py` | exists | Verified |
| `scripts/evaluation/generate_regression_report.py` | exists | Verified |
| `.sisyphus/evidence/paper-b/task-10/queries_subset.jsonl` | exists | Verified |
| `data/synth_benchmarks/stability_bench_v1/queries.jsonl` | exists | Verified |
| `scripts/paper_b/ingest_synth_corpus.py` | exists | Verified |

**Proposed replacements:** N/A (most paths correct)

---

### 7) `.sisyphus/plans/.legacy/paper-a-scope-implementation.md`

**Status:** Active backlog

**Pending checkbox items:** Multiple (tasks 11-14, lines 648-1049)

**References found:**
| Reference | Classification | Current Status |
|-----------|---------------|----------------|
| `scripts/paper_a/build_corpus_meta.py` | exists | Verified |
| `scripts/paper_a/build_shared_and_scope.py` | exists | Verified |
| `scripts/paper_a/build_family_map.py` | exists | Verified |
| `scripts/paper_a/build_eval_sets.py` | exists | Verified |
| `scripts/paper_a/evaluate_paper_a.py` | exists | Verified |
| `scripts/paper_a/retrieval_runner.py` | exists | Verified |
| `backend/llm_infrastructure/retrieval/filters/scope_filter.py` | exists | Verified |
| `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` | exists | Verified |
| `data/chunk_v3_normalize_table.md` | exists | Verified |

**Proposed replacements:** N/A

---

### 8) `.sisyphus/plans/.legacy/paper-a-supervisor-review-plan.md`

**Status:** Active backlog (overlaps #7)

**Pending checkbox items:** Multiple (tasks 0-12)

**References found:**
| Reference | Classification | Current Status |
|-----------|---------------|----------------|
| `scripts/paper_a/preflight_es.py` | exists | Verified |
| `scripts/paper_a/validate_master_eval_jsonl.py` | exists | Verified |
| `scripts/paper_a/rebuild_query_gold_master_splits.py` | exists | Verified |
| `scripts/paper_a/evaluate_paper_a_master.py` | exists | Verified |
| `docs/papers/20_paper_a_scope/evidence_mapping.md` | exists | Verified |
| `data/paper_a/eval/query_gold_master.jsonl` | exists | Verified |

**Proposed replacements:** N/A

---

### 9) `.sisyphus/plans/.legacy/agent-retrieval-stability-hardening.md`

**Status:** Completed (per unified plan)

**Pending checkbox items:** None

**References:** N/A

---

### 10) `.sisyphus/plans/.legacy/paper-b-stability.md`

**Status:** Completed (per unified plan)

**Pending checkbox items:** None

**References:** N/A

---

## Zombie Risk Scan

> Quote any legacy instructions that use backgrounding (e.g., `&`, nohup) or unbounded loops.

**Findings:** No zombie risks detected in legacy plans.

- No `&` backgrounding patterns found
- No `nohup` usage
- No `disown` usage
- No unbounded sleep loops

Note: Before-after-regression-compare plan contains shell loops with `for i in $(seq 1 60); do ...; sleep 1; done` - these are bounded (max 60 iterations) and used for health checks, not zombie risks.

---

## Verification Commands

```bash
# Verify ledger exists
test -f .sisyphus/evidence/start-work/phase-a/consistency-ledger.md && echo "PASS" || echo "FAIL"

# Verify key references
ls -la normalize.py
ls -la scripts/chunk_v3/run_chunking.py
ls -la scripts/paper_a/evaluate_paper_a.py
ls -la frontend/src/features/chat/components/guided-selection-panel.tsx
```

---

## Notes

- The unified plan `.sisyphus/plans/start-work-all-remaining.md` already contains corrected file references in many cases
- Primary action needed: Update any remaining references to `tests/api/test_agent_rrf_and_sticky_gates.py` to use the current test files
- All core script infrastructure exists; only some documentation/data files need verification
