# Elasticsearch Index Alias & Isolation Patterns - Research Findings

## References

1. **Elasticsearch Official Aliases Documentation**
   - URL: https://www.elastic.co/guide/en/elasticsearch/reference/master/aliases.html
   - Key content: Alias types, add/remove operations, atomic swaps, write index routing

2. **Elasticsearch Aliases API Documentation**
   - URL: https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-indices-update-aliases
   - Key content: POST /_aliases with atomic add/remove actions

3. **Elastic Data Stream Naming Scheme**
   - URL: https://elastic.co/blog/an-introduction-to-the-elastic-data-stream-naming-scheme
   - Key content: `{type}-{dataset}-{namespace}` pattern for environment isolation

4. **Stack Overflow - Index Naming Conventions**
   - URL: https://stackoverflow.com/questions/39907523/are-there-conventions-for-naming-organizing-elasticsearch-indexes-which-store-log-data
   - Key content: Industry-standard patterns for index organization by environment

---

## Isolation Checklist

### Alias Safety (for benchmark isolation)
- [ ] **Never reuse production aliases** - Create dedicated aliases for synthetic benchmarks (e.g., `rag_synth_current` instead of `rag_current`)
- [ ] **Use atomic alias swap** - Use `POST /_aliases` with both `remove` and `add` actions in single request to prevent downtime
- [ ] **Verify alias before queries** - Check `GET /_alias/your-alias` before running benchmarks
- [ ] **Separate write index** - Use `is_write_index: true` to direct writes to specific index

### Index Prefix Isolation
- [ ] **Dedicated prefix required** - Use `SEARCH_ES_INDEX_PREFIX=rag_synth` for synthetic indices
- [ ] **Environment variable enforcement** - Use `SEARCH_ES_ENV=synth` to ensure CLI uses correct prefix
- [ ] **No wildcard overlap** - Ensure `rag_synth_*` never matches `rag_prod_*` or `rag_*`
- [ ] **Document prefix pattern** - Record in runbook: `{env}-{purpose}-{version}` (e.g., `synth-rag-v1`)

### Operational Safety
- [ ] **List existing aliases first** - `GET /_alias` to see current state before modifications
- [ ] **Test alias resolution** - Verify `GET /_alias/rag_synth_current` returns expected index
- [ ] **Rollback plan** - Keep previous index until new alias confirmed working
- [ ] **Audit trail** - Log all alias changes with timestamps

---

## Repo Mapping: Applying to Repo CLI Usage

### Environment Variables (from plan)
| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `SEARCH_ES_ENV` | Environment identifier | `synth` |
| `SEARCH_ES_INDEX_PREFIX` | Index name prefix | `rag_synth` |

### CLI Command Pattern
The repo CLI should create/switch synthetic alias using:

```bash
# Step 1: Create new synthetic index with prefixed name
# Index name: rag_synth_20260227 (using prefix + date)

# Step 2: Atomic alias swap via POST /_aliases
POST /_aliases
{
  "actions": [
    { "remove": { "index": "rag_synth_*", "alias": "rag_synth_current" } },
    { "add":    { "index": "rag_synth_20260227", "alias": "rag_synth_current", "is_write_index": true } }
  ]
}
```

### Key Points for Repo CLI Implementation
1. **Construct index name**: `{SEARCH_ES_INDEX_PREFIX}_{timestamp}` or `{SEARCH_ES_INDEX_PREFIX}_{version}`
2. **Query existing indices**: `GET /{SEARCH_ES_INDEX_PREFIX}_*` to find current synthetic indices
3. **Build atomic swap**: Use `POST /_aliases` with both remove (old) and add (new) actions
4. **Verify after swap**: Confirm `rag_synth_current` points to correct index
5. **Do NOT touch production**: Never reference `rag_current` or other prod aliases

### What NOT to Do
- ❌ Don't use `rag_current` for synthetic benchmarks
- ❌ Don't modify production aliases during benchmark runs
- ❌ Don't rely on environment alone—prefix provides additional isolation layer
- ❌ Don't skip atomic operations (could leave alias in inconsistent state)

---

*Generated: 2026-02-27*

---

## Synthetic Benchmark Generator (Task 3) Notes

- Added `scripts/paper_b/generate_synth_benchmark.py` with stdlib-only deterministic generation via `random.Random(seed)` and stable ordered writes.
- Implemented fixed schema/counts exactly: 120 docs, 30 near-duplicate pairs (60 docs), 60 groups, 4 queries/group, 240 total queries.
- Enforced group distribution hard constraints in-generator: `abbr>=20`, `mixed_lang>=20`, `error_code>=10`, `near_dup>=15`.
- Leakage guardrails are hard-fail and fail-fast: L1 (`DOC_`/`SYNTH_` ban), L2 (longest common substring <= 40), L3 (token 5-gram Jaccard <= 0.35).
- `--selftest` injects deliberate L1 leakage (`DOC_9999`) and returns non-zero with explicit rule/qid in output.
- Manifest reproducibility uses `hashlib.sha256` over written `corpus.jsonl` and `queries.jsonl`, and deterministic JSON serialization for stable hash/manifest outputs.

---

# Deterministic Synthetic Dataset Generation Patterns

## References

1. **Moses Tokenizer `--selftest` pattern**  
   https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/mosestokenizer/tokenizer.py  
   - CLI uses `docopt` with `--selftest, -t` flag
   - Runs internal tests and exits non-zero on failure

2. **manifestly** - Directory manifest generator  
   https://github.com/gdoermann/manifestly  
   - Generates JSON manifest with SHA256 hashes
   - Pure Python stdlib

3. **seedbank** - Multi-RNG seed management  
   https://github.com/lenskit/seedbank  
   - Centralized seed config for reproducibility

4. **RandomDataset** - Tabular random dataset generator  
   https://github.com/KCL-BMEIS/RandomDataset  
   - Deterministic random tabular data with seeds

5. **df-fingerprint** - DataFrame canonicalization  
   https://github.com/hjk612/df-fingerprint  
   - Stable fingerprints via hashing

---

## Recommended Minimal Manifest Schema

```json
{
  "version": "1.0",
  "generator": {
    "name": "generate_synth_benchmark",
    "version": "0.1.0"
  },
  "seed": 42,
  "created_at": "2026-02-27T17:00:00Z",
  "files": [
    {
      "path": "data/train.jsonl",
      "sha256": "a1b2c3d4e5f6...",
      "size_bytes": 12345
    }
  ],
  "metadata": {
    "num_samples": 1000,
    "columns": ["input", "output", "metadata"]
  }
}
```

**Key fields:**
- `seed` - integer for random.seed() reproducibility
- `version` - generator version for replay
- `files[].sha256` - use hashlib.sha256 (NOT Python's hash())
- `created_at` - ISO 8601 timestamp

---

## Selftest Implementation Pattern

```python
#!/usr/bin/env python3
"""CLI with self-test mode - intentional failure pattern."""
import argparse
import sys

def run_tests():
    """Run internal tests that intentionally FAIL."""
    import random
    random.seed(42)
    result = random.random()
    # Intentionally assert FALSE to trigger failure
    assert result != result, "Self-test: intentional failure"
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true', 
                       help='Run self-tests and exit')
    args = parser.parse_args()
    
    if args.selftest:
        print("Running self-tests...", file=sys.stderr)
        try:
            run_tests()
            print("All self-tests passed.", file=sys.stderr)
            sys.exit(0)
        except AssertionError as e:
            print(f"SELFTEST FAILED: {e}", file=sys.stderr)
            sys.exit(1)  # Non-zero exit!
    
    # Normal operation here...

if __name__ == "__main__":
    main()
```

**Steps:**
1. Parse `--selftest` flag early, exit immediately
2. Call test function that intentionally asserts FALSE
3. Catch AssertionError and call sys.exit(1)
4. Exit 0 on success

---

## Pitfalls & Best Practices

**Pitfalls:**
- Use `hashlib.sha256()`, NOT Python's built-in `hash()` - plan explicitly forbids hash() due to randomness across runs
- Seed BOTH `random` and `numpy.random` if using both libraries
- Include generator version - API changes break reproducibility
- Compute file hashes AFTER generation completes

**Best practices:**
- Store seed in manifest for replay capability
- Use ISO 8601 timestamps (`datetime.utcnow().isoformat() + 'Z'`)
- Keep manifest alongside dataset in same directory
- Add `--selftest` as early exit before heavy computation

---

## Task 4 (ES isolation runbook) conventions and gotchas

- `SearchSettings` uses `env_prefix="SEARCH_"`, but the standalone ES index CLI reads `ES_ENV` and `ES_INDEX_PREFIX` directly. Set both families when running paper-b synthetic flows.
- `EsIndexManager` naming is `{index_prefix}_{env}_v{version}` and alias `{index_prefix}_{env}_current`; with `ES_INDEX_PREFIX=rag_synth` and `ES_ENV=synth`, the resulting names are `rag_synth_synth_v1` and `rag_synth_synth_current`.
- Safe, decision-complete create/switch command for synthetic namespace is:
  `ES_ENV=synth ES_INDEX_PREFIX=rag_synth python -m backend.llm_infrastructure.elasticsearch.cli create --version 1 --dims $(python -c "from backend.config.settings import search_settings; print(search_settings.es_embedding_dims)") --switch-alias`
- Isolation verification should explicitly check alias targets (`GET /_alias/rag_synth_synth_current` and `GET /_alias/*_current`) and confirm only `rag_synth_*` indices are touched.

## Task 5 (synthetic corpus ingestion) conventions and gotchas

- Reuse `EsIngestService.from_settings()` + `ingest_sections(...)` per document, with exactly one `Section` per JSONL row to preserve deterministic Section=Chunk behavior.
- Keep corpus traversal deterministic by iterating JSONL in file order and avoiding any sort/shuffle stages; use `refresh=False` per ingest call and perform one final `svc.es.indices.refresh(index=svc.index)`.
- Pass corpus `tags` directly into `ingest_sections(..., tags=doc_tags)` so both `tags` and `search_text` reflect benchmark labels, and map row fields into section metadata keys expected by `EsChunkDocument.from_section` (`device_name`, `equip_id`, `chapter`).

## Task 6 (Paper B evaluation runner) implementation notes

- Added `scripts/paper_b/run_paper_b_eval.py` with stdlib-only HTTP (`urllib.request`) to call `POST /api/retrieval/run` using pinned request fields (`steps=["retrieve"]`, `debug=false`, `deterministic` toggle, `rerank_enabled=false`, `auto_parse=false`, and nullable optional fields).
- `results.jsonl` now records one row per execution with required traceability fields: `run_id`, `effective_config_hash`, `top_k_doc_ids`, `trace_id`, `warnings`, plus `qid/group_id/query/mode/repeat_index/latency_ms`.
- Metrics policy is fixed and documented in code docstring: quality (`hit@k`, `MRR`) uses first deterministic run/query; repeat stability compares each repeat to repeat-0 (`Jaccard@10`, `ExactMatch@10`); paraphrase stability uses mean pairwise `Jaccard@10` within each `group_id` from deterministic repeat-0 runs.
- Artifacts are written only under `.sisyphus/evidence/paper-b/` (`results.jsonl`, `metrics.json`), and smoke verification succeeded with `--limit 2 --repeats 2`.

## Task 7 (prompt spec v2 stability controls) implementation notes

- Added env-driven prompt spec selector `RAG_PROMPT_SPEC_VERSION` via `rag_settings.prompt_spec_version` defaulting to `v1`; this keeps existing behavior unchanged when unset.
- Wired prompt-spec loading through settings in both DI and agent constructor paths: `backend/api/dependencies.py#get_prompt_spec_cached` and `backend/services/agents/langgraph_rag_agent.py` now call `load_prompt_spec(version=rag_settings.prompt_spec_version)`.
- Added full v2 prompt set using existing loader contract `{name}_{version}.yaml`, including optional `translate_v2` and `auto_parse_v2`, so v2 activation is file-resolver compatible with no loader change.
- Enforced MQ stability constraints directly in v2 MQ prompts (`setup_mq_v2`, `ts_mq_v2`, `general_mq_v2`, `st_mq_v2`): preserve abbreviations/code tokens verbatim, no label-style outputs, and first query line must be original user query.
- Debug snapshot verification path: `/api/retrieval/run` + `/api/retrieval/runs/{run_id}` with v2 spec showed `search_queries` retaining abbreviation/error tokens (`TM`, `PCW`, `E-0101`, `ALM-01`) and first query equal to the original query string.

## Task 10 (full ablation + paper assets) conventions and gotchas

- Extended `scripts/paper_b/run_paper_b_eval.py` in a backward-compatible way: default behavior still runs deterministic repeats + one nondeterministic baseline, while new flags (`--primary-deterministic`, `--include-baseline`, `--primary-mode-name`) allow nondeterministic repeat-mode evaluations for baseline/v2 ablations.
- For prompt-spec ablations, `RAG_PROMPT_SPEC_VERSION` remains process-level at API startup, so v1/v2 comparisons require separate server runs; per-request payload alone cannot switch prompt spec versions.
- Runtime discipline for nondeterministic MQ runs: use bounded samples (`--limit`) and small repeat counts when producing task-10 assets, and report those sample-size constraints directly in Table 1 (`query_count`, `repeats`).
- Driver-breakdown figure generation should derive per-bucket repeat stability from query `tags` (`abbr`, `mixed_lang`, `error_code`, `near_dup`) keyed by `qid`, then aggregate per-configuration from primary-mode rows only.

## Task 12 (synthetic benchmark release packaging) notes

- Created `data/synth_benchmarks/stability_bench_v1/LICENSE` with MIT license (self-contained, no placeholders).
- Created `data/synth_benchmarks/stability_bench_v1/README.md` documenting the complete single-command run path: generate → create index → ingest → run eval → generate assets.
- README includes prerequisites (dev stack ES :8002, API :8011), exact commands with all required flags, and output locations matching repo structure.
- Environment variable split noted: `SEARCH_ES_*` for backend, `ES_*` for CLI; both required for synthetic flows.
- Index naming convention confirmed: with `ES_ENV=synth` + `ES_INDEX_PREFIX=rag_synth`, results in `rag_synth_synth_v1` index and `rag_synth_synth_current` alias.
- Verified manifest.json seed (123) and SHA256 hashes match what's documented in README reproducibility section.

---

## Task 12 fix (README CLI flags correction)

- Fixed `README.md` CLI flags to match actual script argparse:
  - Step 3 (ingest): changed positional arg to `--corpus` flag
  - Step 4 (eval): changed `--output` to `--out-dir`, added `--api-base-url http://localhost:8011`
  - Step 5 (assets): changed `--evidence-root` from `.sisyphus/evidence/paper-b` to `.sisyphus/evidence/paper-b/task-10`, fixed `--queries` path to `task-10/queries_subset.jsonl`

---

## Task 6 rerun (evaluation runner spec alignment)

- Reworked `scripts/paper_b/run_paper_b_eval.py` to pin request payload keys exactly to the retrieval request schema subset required by Task 6: `query`, `steps`, `debug`, `deterministic`, `final_top_k`, `rerank_enabled`, `auto_parse`, `skip_mq`, `device_names`, `doc_types`, `doc_types_strict`, `equip_ids`.
- Deterministic protocol and nondeterministic baseline now both use decision-complete payloads from plan (`steps=["retrieve"]`, `debug=false`, `auto_parse=false`, `rerank_enabled=false`, deterministic toggle only).
- Repeat stability metric implementation is now strict pairwise mean per query across all `N_repeats` deterministic runs, then averaged across queries for `RepeatJaccard@10` and `RepeatExactMatch@10`.
- Paraphrase stability metric implementation now computes pairwise means within each `group_id` from deterministic repeat-0 results, then averages across groups for `ParaphraseJaccard@10` and `ParaphraseExactMatch@10`.
- `metrics.json` now emits only spec metrics with exact names: `hit@5`, `hit@10`, `MRR`, `RepeatJaccard@10`, `RepeatExactMatch@10`, `ParaphraseJaccard@10`, `ParaphraseExactMatch@10`, `p95_latency_ms`.

## Schema compatibility fix (eval -> assets)

- Updated `scripts/paper_b/run_paper_b_eval.py` to emit `results.jsonl` rows with `top_k_doc_ids` as the primary field expected by assets generation, while also keeping `top10_doc_ids` as a compatibility alias.
- Updated `metrics.json` emission to include assets-compatible keys (`hit_at_5`, `hit_at_10`, `mrr`, `repeat_stability_jaccard_at_10`, `paraphrase_stability_jaccard_at_10`, `p95_latency_ms`, `query_count`, `deterministic_repeats`, `primary_mode_name`) and retained legacy metric keys (`hit@5`, `hit@10`, `MRR`, etc.) for backward compatibility.

## Task 7 stability controls (prompt spec v2 + gating)

- Added `RAG_PROMPT_SPEC_VERSION` to `RAGSettings` with default `v1`; this keeps production/default behavior unchanged unless explicitly overridden.
- Wired versioned prompt loading through the same setting in both call sites: `backend/api/dependencies.py#get_prompt_spec_cached` and `backend/services/agents/langgraph_rag_agent.py` fallback constructor path.
- Added full v2 prompt set using `{name}_{version}.yaml` contract: required files (`router`, `setup_mq`, `ts_mq`, `general_mq`, `st_gate`, `st_mq`, `setup_ans`, `ts_ans`, `general_ans`) plus parity (`translate`, `auto_parse`).
- Encoded stability-aware MQ protocol in v2 prompts: first query line must be exact original user query, abbreviation/error-code tokens must remain verbatim, and outputs must be plain query strings without labels/JSON wrappers.
- Smoke verified both default and opt-in paths locally: `load_prompt_spec(version="v1")` and `load_prompt_spec(version="v2")` load successfully, and `get_prompt_spec_cached().router.version` resolves to `v1` by default and `v2` with `RAG_PROMPT_SPEC_VERSION=v2`.

## Task 10 CLI examples (per-config evidence directories)

- Baseline v1 nondeterministic (`baseline_v1_nondet`):
  `python scripts/paper_b/run_paper_b_eval.py --queries .sisyphus/evidence/paper-b/task-10/queries_subset.jsonl --out-dir .sisyphus/evidence/paper-b/task-10/baseline_v1_nondet --repeats 10 --primary-mode-name baseline_v1_nondet --primary-deterministic false --include-baseline false`
- Deterministic protocol (`deterministic_protocol`):
  `python scripts/paper_b/run_paper_b_eval.py --queries .sisyphus/evidence/paper-b/task-10/queries_subset.jsonl --out-dir .sisyphus/evidence/paper-b/task-10/deterministic_protocol --repeats 10 --primary-mode-name deterministic_protocol --primary-deterministic true --include-baseline false`
- Stability-aware MQ v2 (`stability_aware_mq_v2`, run API with `RAG_PROMPT_SPEC_VERSION=v2`):
  `python scripts/paper_b/run_paper_b_eval.py --queries .sisyphus/evidence/paper-b/task-10/queries_subset.jsonl --out-dir .sisyphus/evidence/paper-b/task-10/stability_aware_mq_v2 --repeats 10 --primary-mode-name stability_aware_mq_v2 --primary-deterministic false --include-baseline false`


- Updated CLI examples to include `--api-base-url http://localhost:8011` flag for all evaluation runs (leverages dev API on port 8011).
- Added `ES_HOST=http://localhost:8002` to the ES CLI environment variables section in the README (required for create/ingest commands; without it, CLI defaults to localhost:9200).
- Added a new "One-Command Run (All 3 Configurations)" block to the README that provides a single copy/paste command to run the complete Paper B evaluation pipeline including all 3 evaluation configurations and asset generation. The command uses `bash -lc 'set -euo pipefail; ...'` style to ensure proper error handling.


- Fixed README v2 instructions: docker-compose.yml `api-dev` service does NOT pass `RAG_PROMPT_SPEC_VERSION` into the container (it's not in the `environment:` list), so shell env prefix won't work. Updated the prerequisite section to instruct users to start API manually with `RAG_PROMPT_SPEC_VERSION=v2 uvicorn backend.api.main:app --host 0.0.0.0 --port 8011`.
- Fixed one-command block: added `mkdir -p .sisyphus/evidence/paper-b/task-10` before writing `queries_subset.jsonl` to ensure the directory exists.
- Removed useless `RAG_PROMPT_SPEC_VERSION=v2` prefix from the v2 eval call in the one-command block since the env var must be set for the API process, not the eval script.

## Task 11 (paper reproduction commands update) notes

- Updated `docs/paper/paper_b_stability.md` Section 9 (Reproducibility) with corrected CLI commands:
  - Ingestion: changed positional corpus arg to `--corpus` flag
  - Evaluation: changed `--output` to `--out-dir`, added `--api-base-url http://localhost:8011`
  - Assets: already correct (uses `--evidence-root .sisyphus/evidence/paper-b/task-10`)
- Added v2 prompt gating clarification: "The stability-aware MQ (v2) evaluation requires the API to be started with `RAG_PROMPT_SPEC_VERSION=v2`. This cannot be toggled per-request and must be set when the API process starts."
- F1/F3 audit findings: paper draft used wrong flags (`--output` vs `--out-dir`) and wrong ingestion invocation.

- 2026-02-27: Generated Task 1/2/3/6 evidence artifacts under .sisyphus/evidence/paper-b/ (spec & guardrails checks, schema & leakage checks, selftest leakage-fail output with exit code, and task-6 metrics JSON via small eval fallback handling).



## Task 11 (Discussion section addition) notes

- Added new Section 7 "Discussion" to `docs/paper/paper_b_stability.md` after "Results" section.
- Renumbered subsequent sections: Driver Analysis (7→8), Limitations (8→9), Reproducibility (9→10).
- Discussion content covers:
  - Stability-Recall tradeoff analysis (deterministic improves both stability and recall)
  - Latency implications (10x reduction with deterministic mode)
  - Tier guarantees in practice (T1/T3 guaranteed, T2/T4 NOT guaranteed per spec)
  - Operational recommendations for production RAG deployments
- Discussion uses tier-bounded language consistent with spec: "NOT guaranteed" for T2 (paraphrase) and T4 (reindex).
- No table values modified; metrics remain unchanged from table_1_metrics.md.

## Task 12 follow-up (single-config eval invocation)

- Simplified `scripts/paper_b/run_paper_b_eval.py` to run exactly one configuration per process via `--deterministic` and `--mode-name`, so README pipelines can invoke baseline/deterministic/v2 as three separate runs with distinct `--out-dir` values.
- Kept retrieval payload schema pinned and explicit, including `final_top_k: 10` on every request, and retained per-call output row compatibility fields (`mode`, `request_payload`, `top10_doc_ids`, trace metadata).

## Task 7 wiring follow-up (env-gated prompt spec selection)

- Added  wiring via  (default ) so existing behavior stays unchanged when unset.
- Both DI () and  fallback spec load now call  for consistent process-level gating.
- Caveat: prompt spec version is read at process/runtime settings load, so switching v1/v2 requires setting env before starting the backend process.

## Task 7 wiring follow-up (env-gated prompt spec selection)

- Added RAG_PROMPT_SPEC_VERSION wiring via rag_settings.prompt_spec_version (default v1) so existing behavior stays unchanged when unset.
- Both DI (get_prompt_spec_cached) and LangGraphRAGAgent fallback spec load now call load_prompt_spec(version=rag_settings.prompt_spec_version) for consistent process-level gating.
- Caveat: prompt spec version is read at process/runtime settings load, so switching v1/v2 requires setting env before starting the backend process.

- Note: Ignore the immediately previous malformed duplicate heading entry; the complete/correct bullets are in this latest Task 7 wiring follow-up block.


## Minimal settings.py restoration (2026-02-28)

- Restored `backend/config/settings.py` to upstream (removed formatting churn).
- Added `prompt_spec_version` field in Retrieval section with default "v1" and description referencing `RAG_PROMPT_SPEC_VERSION` env var.
- The diff is minimal: only 5 lines added (the new field + blank line).
- Verification: `python -c "from backend.config.settings import rag_settings; print(rag_settings.prompt_spec_version)"` prints "v1".
#NQ|- Verification: `python -c "from backend.config.settings import rag_settings; print(rag_settings.prompt_spec_version)"` prints "v1".
#BN|
#BQ|## README CLI flags correction (2026-02-28)
#NX|
#BQ|- Fixed dataset README commands to match current script argparse:
#QZ|  - `--primary-mode-name` -> `--mode-name`
#BQ|  - `--primary-deterministic` -> `--deterministic`
#HV|  - Removed non-existent `--include-baseline` flag
#QT|  - Changed v2 API to run on port 18012 (separate process) instead of reusing 8011
#HQ|  - Made ES index creation idempotent with `create ... || switch ...` pattern
