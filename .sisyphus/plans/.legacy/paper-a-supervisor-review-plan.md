# Paper A: Supervisor-Grade Verification + New Hypotheses + Experiment Plan (Cross-Equipment Contamination in RAG)

## TL;DR
> **Summary**: Audit Paper A as a critical reviewer, eliminate spec↔code↔data inconsistencies, and run a decision-locked experiment program that produces reproducible contamination/recall/latency evidence plus a tightened Methodology (definitions + Algorithm box + symbol table).
> **Deliverables**:
> - Reviewer-style critique + threat-model + leakage audit report under `docs/papers/20_paper_a_scope/`
> - New falsifiable hypotheses (H1..Hn) with concrete experiments + baselines + ablations
> - A single authoritative evaluation schema + validators + runnable harness producing `per_query.csv` + summaries + stats
> - Updated manuscript artifacts: Methodology rewrite + Algorithm 1 + metric/symbol tables
> **Effort**: XL
> **Parallel**: YES - 4 waves
> **Critical Path**: Schema lock + validators → evaluator parity (B0..P7) → run evidence runs → write methodology+experimental setup with evidence links

## Context
### Original Request
- As advisor/reviewer, rigorously verify Paper A and propose new hypotheses + experiment plans, grounded in `docs/papers/` and the `paperA` folder.

### Repo Grounding (what already exists)
- Paper A spec and positioning:
  - `docs/papers/20_paper_a_scope/README.md`
  - `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`
  - `docs/papers/20_paper_a_scope/evidence_mapping.md`
  - `docs/papers/20_paper_a_scope/related_work.md`
- Common evaluation protocol:
  - `docs/papers/10_common_protocol/paper_common_protocol.md`
- Paper A datasets/labels (already present; docs must be updated to reflect this):
  - `data/paper_a/eval/query_gold_master.jsonl` (v0.4 input; will be frozen and superseded)
  - `data/paper_a/eval/query_gold_master_v0_5.jsonl` (to be generated in Task 2; official)
  - `data/paper_a/corpus_labels/document_scope_table.csv`
  - `data/paper_a/corpus_labels/shared_doc_gold.csv`
  - `data/paper_a/corpus_labels/device_family_gold.csv`
  - `data/paper_a/metadata/device_catalog.csv`
  - `data/paper_a/metadata/doc_type_map.csv`
  - `data/paper_a/metadata/equip_catalog.csv`
- Paper A scripts:
  - `scripts/paper_a/build_corpus_meta.py`
  - `scripts/paper_a/build_shared_and_scope.py`
  - `scripts/paper_a/build_family_map.py`
  - `scripts/paper_a/build_eval_sets.py`
  - `scripts/paper_a/evaluate_paper_a.py`
- Scope filtering DSL (OR branches):
  - `backend/llm_infrastructure/retrieval/filters/scope_filter.py`

### Oracle Review (key reviewer risks to address)
- Definition gaps (scopes, `v_scope`, “Base” score scale), leakage risks, ambiguous-gold evaluation, baseline incompleteness, statistical testing gaps, and unfair latency/Matryoshka comparisons.

### Metis Review (gaps addressed in this plan)
- Schema drift: master gold (`data/paper_a/eval/query_gold_master_v0_5.jsonl`) vs legacy eval schema consumed by `scripts/paper_a/evaluate_paper_a.py`.
- Incomplete system coverage: current evaluator effectively supports `B0..B4` + `P1` only; `P2..P4` and `P6..P7` need explicit implementation.
- Equip-level evaluation risk: `scripts/paper_a/build_shared_and_scope.py` does not currently emit `es_equip_id` in `doc_scope.jsonl` but downstream code expects it.
- Docs inconsistency: `docs/papers/20_paper_a_scope/evidence_mapping.md` marks artifacts “미생성” although they exist under `data/paper_a/`.

## Work Objectives
### Core Objective
- Produce a reviewer-proof Paper A package: (1) internally consistent definitions and claims, (2) experiment evidence that matches the spec, (3) a manuscript-ready Methodology/Experimental Setup with verifiable evidence links.

### Definition of Done (agent-executable)
- [ ] A single authoritative evaluation schema is enforced by a validator (no silent row dropping).
- [ ] Systems in spec are runnable or explicitly marked “planned, not reported” with a hard gate.
- [ ] Reproducible runs exist (uncommitted) with: `per_query.csv`, summaries, and stats artifacts; hashes/index target recorded.
- [ ] Reviewer report contains: (a) novelty positioning, (b) threat model + leakage audit, (c) baseline sufficiency checklist, (d) failure taxonomy.
- [ ] `docs/papers/20_paper_a_scope/paper_a_scope.md` contains finalized Methodology + Experimental Setup + Algorithm 1 + symbol/metric tables.

### Must Have
- Main evaluation uses `data/paper_a/eval/query_gold_master_v0_5.jsonl` as the authoritative source of truth (with `query_gold_master_v0_4_frozen.jsonl` retained for traceability).
- Contamination metrics exactly match `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` and are implemented consistently across systems.
- Statistics: paired bootstrap CIs for key deltas; McNemar for CE@k; multiple-comparison correction (Holm-Bonferroni).
- Strong control baselines to separate “smaller corpus” effects from “correct routing” effects.

### Defaults Applied (to remove ambiguity)
- Evaluation split policy (user decision): regenerate a versioned master eval set with balanced dev/test across all `scope_observability`.
  - Keep the current file as frozen backup (`query_gold_master_v0_4_frozen.jsonl`).
  - Use `query_gold_master_v0_5.jsonl` as the official file for all reported tables.

### Must NOT Have (guardrails)
- Do not tune `T(shared)`, `tau(family)`, `M`, `router_dim`, or `λ` on test/final split.
- Do not claim equip-level improvements unless equip metadata is present end-to-end and the equip-centric subset is explicitly defined.
- Do not redefine “shared” or “family” per system/run.
- Do not commit `.sisyphus/evidence/**` or other run artifacts into git.
- Do not mutate any `data/paper_a/**` files EXCEPT `data/paper_a/eval/` and only via the versioned regeneration protocol in Task 2.

## Verification Strategy
> ZERO HUMAN INTERVENTION: all checks are script/test-driven.
- Test mode: tests-after (pytest) + CLI runbooks.
- Evidence policy:
  - Implementation evidence: `.sisyphus/evidence/task-{N}-{slug}.*`
  - Run evidence: `.sisyphus/evidence/paper-a/runs/{run_id}/...`
- Reproducibility policy:
  - Every run records: resolved ES index, git SHA, hashes of eval set + policy artifacts + corpus whitelist.

## Execution Strategy
### Parallel Execution Waves
Wave 0 (Build/Verify Run Inputs)
- 0) Preflight ES + build corpus whitelist

Wave 1 (Truth Lock + Consistency)
- 1) Docs/data/code consistency audit + update `evidence_mapping.md`
- 2) Schema lock decision + validators for master eval set
- 3) Policy artifact parity: equip_id propagation + doc_scope correctness
- 4) Baseline completeness checklist + experimental preregistration (comparisons, splits, tuning rules)

Wave 2 (Evaluation Harness Parity)
- 5) Master-schema evaluator + system mapping (REQUIRED: B0..B4,P1; OPTIONAL: P2/P3/P4/P6/P7)
- 6) Add mandatory control baselines (random-scope, global+postfilter, per-scope merge, dedupe)
- 7) Stats expansion (bootstrap, McNemar, Holm) + stability/repeatability option

Wave 3 (Evidence Runs + Error Analysis)
- 8) Execute core runs (explicit_device/implicit/ambiguous/explicit_equip) and produce evidence artifacts
- 9) Produce failure taxonomy + qualitative slices (wrong-scope, mixed-scope, shared-false-positive, etc.)

Wave 4 (Manuscript-Ready Writing)
- 10) Reviewer report + threat model/leakage audit + baseline adequacy statement
- 11) Methodology rewrite (3.1–3.11) + Algorithm 1 + symbol/metric tables
- 12) Experimental setup + results table generator + evidence linking

### Dependency Matrix
- 2 blocks 5, 8 (schema lock precedes evaluator + runs)
- 3 blocks 5, 8 (equip_id + doc_scope correctness precede equip-level evaluation)
- 5 blocks 8 (evaluator precedes evidence runs)
- 8 blocks 9, 12 (runs precede error analysis + paper tables)

## TODOs
> Implementation + verification live in the same task.

- [ ] 0. Build/Verify Run Inputs: ES preflight + corpus whitelist (manifest→ES join)

  **What to do**:
  - Run and validate the existing Paper A input builders to produce a *single, run-versioned* set of artifacts under `.sisyphus/evidence/paper-a/`:
    1) ES preflight (alias→index resolution)
       - Command:
         - `python scripts/paper_a/preflight_es.py --out .sisyphus/evidence/paper-a/preflight_es.json`
       - Output: `.sisyphus/evidence/paper-a/preflight_es.json`
    2) Corpus metadata join + whitelist (manifest→ES doc_id join)
       - Command:
         - `python scripts/paper_a/build_corpus_meta.py --normalize-table data/chunk_v3_normalize_table.md --out-dir .sisyphus/evidence/paper-a/corpus`
       - Outputs:
         - `.sisyphus/evidence/paper-a/corpus/doc_meta.jsonl`
         - `.sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt`
         - `.sisyphus/evidence/paper-a/corpus/corpus_snapshot.json`
    3) NOTE: Policy artifacts (shared/doc_scope/family_map) are produced after Task 3 updates `build_shared_and_scope.py` to emit equip_id.

  **Must NOT do**:
  - Do not point any run at a moving ES alias without recording the resolved index (preflight output is mandatory).
  - Do not edit `data/paper_a/**` in this task.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: ES connectivity + run hygiene
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 0 | Blocks: 2,3,5,8 | Blocked By: none

  **References**:
  - Builders: `scripts/paper_a/preflight_es.py`, `scripts/paper_a/build_corpus_meta.py`
  - Corpus manifest: `data/chunk_v3_normalize_table.md`

  **Acceptance Criteria**:
  - [ ] All listed output files exist under `.sisyphus/evidence/paper-a/`
  - [ ] `.sisyphus/evidence/paper-a/preflight_es.json` contains a non-empty resolved index
  - [ ] `.sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt` line-count equals `.sisyphus/evidence/paper-a/corpus/corpus_snapshot.json.total_docs`

  **QA Scenarios**:
  ```
  Scenario: Inputs build end-to-end
    Tool: Bash
    Steps: python scripts/paper_a/preflight_es.py --out .sisyphus/evidence/paper-a/preflight_es.json && python scripts/paper_a/build_corpus_meta.py --normalize-table data/chunk_v3_normalize_table.md --out-dir .sisyphus/evidence/paper-a/corpus
    Expected: exit 0; preflight + corpus artifacts exist
    Evidence: .sisyphus/evidence/task-00-inputs-build.txt

  Scenario: ES unreachable
    Tool: Bash
    Steps: SEARCH_ES_HOST=http://127.0.0.1:1 python scripts/paper_a/preflight_es.py --out .sisyphus/evidence/paper-a/preflight_es.json 2> .sisyphus/evidence/task-00-inputs-build-error.txt || true
    Expected: exit !=0 with clear error
    Evidence: `.sisyphus/evidence/task-01-preflight-es-error.txt` (script-written) and `.sisyphus/evidence/task-00-inputs-build-error.txt` (captured stderr)
  ```

  **Commit**: NO | Message: `n/a` | Files: none (do not commit `.sisyphus/evidence/**`)

- [ ] 1. Consistency Audit: reconcile docs vs existing data/code and freeze “source of truth” list

  **What to do**:
  - Create `docs/papers/20_paper_a_scope/review/consistency_audit.md` containing:
    - Which artifacts exist today (paths under `data/paper_a/`, `scripts/paper_a/`, `backend/.../scope_filter.py`).
    - Which docs are out-of-date (minimum: `docs/papers/20_paper_a_scope/evidence_mapping.md`).
    - Quantitative drift checks that must be reconciled in prose:
      - The spec’s stated eval-set size (e.g., “380”) vs the actual `query_gold_master.jsonl` row counts (and dev/test split counts).
    - Matryoshka feasibility check (must be explicitly stated in the paper):
      - Current embedding stack is SentenceTransformer-based (`backend/llm_infrastructure/embedding/engines/sentence/embedder.py`, `backend/llm_infrastructure/embedding/adapters/sentence.py`) and provides only fixed-dimension embeddings (`get_dimension()`); there is no in-repo MRL/Matryoshka implementation.
      - Therefore, Matryoshka results must be marked `planned-not-reported` unless a new embedder is added and evaluated fairly.
    - A correction list (exact edits to make in docs) to prevent reviewer confusion.
  - Update `docs/papers/20_paper_a_scope/evidence_mapping.md` “Data Dependencies” table to reflect reality:
    - Mark `Mask set`, `Ambiguous challenge set`, `D_shared`, `Family graph` as “존재 (data/paper_a/...)”.
    - Keep items truly not implemented (e.g., Matryoshka router prototypes index) as “미구현”.

  **Must NOT do**:
  - Do not change the underlying datasets in this task; only documentation consistency.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: doc audit + precise corrections
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2,3 | Blocked By: none

  **References**:
  - Spec: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`
  - Evidence map: `docs/papers/20_paper_a_scope/evidence_mapping.md`
  - Data: `data/paper_a/`

  **Acceptance Criteria**:
  - [ ] `docs/papers/20_paper_a_scope/review/consistency_audit.md` exists
  - [ ] `docs/papers/20_paper_a_scope/evidence_mapping.md` no longer states “미생성” for artifacts present in `data/paper_a/`

  **QA Scenarios**:
  ```
  Scenario: Docs reflect existing artifacts
    Tool: Bash
    Steps: test -f docs/papers/20_paper_a_scope/review/consistency_audit.md && rg -n "미생성" docs/papers/20_paper_a_scope/evidence_mapping.md
    Expected: audit file exists; any remaining "미생성" entries correspond only to truly missing artifacts
    Evidence: .sisyphus/evidence/task-01-consistency-audit.txt

  Scenario: No dataset mutation
    Tool: Bash
    Steps: git status --porcelain
    Expected: only docs/ changes are present (no data/ changes; data/paper_a/eval/ mutation is reserved for Task 2)
    Evidence: .sisyphus/evidence/task-01-consistency-audit-status.txt
  ```

  **Commit**: YES | Message: `docs(paper-a): reconcile evidence mapping with existing artifacts` | Files: [`docs/papers/20_paper_a_scope/evidence_mapping.md`, `docs/papers/20_paper_a_scope/review/consistency_audit.md`]

- [ ] 2. Schema Lock: declare master eval schema as authoritative and add strict validators

  **What to do**:
  - Declare `data/paper_a/eval/query_gold_master_v0_5.jsonl` as authoritative in `docs/papers/20_paper_a_scope/README.md` and note that `query_gold_master_v0_4_frozen.jsonl` is preserved.
  - Add a new validator script `scripts/paper_a/validate_master_eval_jsonl.py`:
    - Required keys: `q_id`, `split`, `question`, `scope_observability`, `intent_primary`, `target_scope_level`, `allowed_devices`, `allowed_equips`, `shared_allowed`, `family_allowed`.
    - Enforce that *reported* subsets have `gold_doc_ids` non-empty (flag-driven):
      - `--require-gold` => fail if any row in selected subset has empty `gold_doc_ids`.
      - `--scope-observability` supports selecting by `scope_observability` values observed in the file:
        - `explicit_device`, `explicit_equip`, `implicit`, `ambiguous`, `all`
      - `--split` supports selecting `dev|test|all` based on the master file’s `split` field.

  - Add a master regeneration builder `scripts/paper_a/rebuild_query_gold_master_splits.py` (VERSIONED, no silent overwrite):
    - Input: `data/paper_a/eval/query_gold_master.jsonl` (treat as v0.4)
    - Outputs:
      - `data/paper_a/eval/query_gold_master_v0_4_frozen.jsonl` (byte-identical backup of input)
      - `data/paper_a/eval/query_gold_master_v0_5.jsonl` (new official eval set with balanced split)
      - `data/paper_a/eval/query_gold_master_v0_5_split_report.json` (counts per split/observability)
    - Behavior (decision-complete):
      - Copy input to the frozen backup path BEFORE any processing.
      - Reassign `split` for ALL rows (including explicit_equip) using stable hash per leak-group (prevents explicit↔masked leakage):
        - `leak_key = compact(question_masked if present else question)`
        - `h = sha256((leak_key + '|' + str(seed_try)).encode('utf-8')).hexdigest()`
        - `bucket = int(h[:8], 16) % 100`
        - `split = 'test' if bucket < 20 else 'dev'`
        - Assign the same split to all rows with the same `leak_key`.
      - Deterministic retry loop for balance constraints:
        - Try `seed_try` in `[seed, seed+99]` until constraints pass, else fail with a report.
        - Constraints:
          - For each `scope_observability` value present in the dataset, `test_count >= max(1, floor(total_count * 0.1))`.
          - No `leak_key` appears in both splits (should be guaranteed by group assignment; still validate).
      - Preserve all other fields verbatim.
      - Write `query_gold_master_v0_5_split_report.json` with counts per (`split`, `scope_observability`) and per-split unique leak_key counts.

  - After generating v0.5, update Paper A docs to prevent reviewer confusion:
    - Update `docs/papers/20_paper_a_scope/README.md` to point “official eval set” to `data/paper_a/eval/query_gold_master_v0_5.jsonl`.
    - Update `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` to:
      - Replace hard-coded “380건” dataset-size statements with the actual v0.5 row count and per-slice counts (from `query_gold_master_v0_5_split_report.json`).
      - Update file-path references in §12.1 to `data/paper_a/eval/query_gold_master_v0_5.jsonl`.
      - Add a short note: v0.4→v0.5 split regeneration + leak_key grouping rationale (explicit/masked leakage prevention).

  **Must NOT do**:
  - Do not “fix” master dataset by dropping rows; fail fast and write a report of offending IDs.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: evaluation integrity gate
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5,8 | Blocked By: none

  **References**:
  - Master eval (input): `data/paper_a/eval/query_gold_master.jsonl`
  - Master eval (official): `data/paper_a/eval/query_gold_master_v0_5.jsonl`
  - Leakage-key canonicalization: `scripts/paper_a/canonicalize.py:compact_key`
  - Common protocol governance: `docs/papers/10_common_protocol/paper_common_protocol.md`

  **Acceptance Criteria**:
  - [ ] `python scripts/paper_a/validate_master_eval_jsonl.py --path data/paper_a/eval/query_gold_master.jsonl` exits 0
  - [ ] `python scripts/paper_a/rebuild_query_gold_master_splits.py --in data/paper_a/eval/query_gold_master.jsonl --out data/paper_a/eval/query_gold_master_v0_5.jsonl --seed 20260305` exits 0
  - [ ] `python scripts/paper_a/validate_master_eval_jsonl.py --path data/paper_a/eval/query_gold_master_v0_5.jsonl --require-gold` exits 0
  - [ ] `data/paper_a/eval/query_gold_master_v0_5_split_report.json` shows non-zero `scope_nonempty_gold_test_counts` for `explicit_device`, `implicit`, `explicit_equip` (and reports `ambiguous` as non-evaluable when non-empty gold is unavailable)
  - [ ] `data/paper_a/eval/query_gold_master_v0_4_frozen.jsonl` is byte-identical to `data/paper_a/eval/query_gold_master.jsonl` (sha256 match)

  **QA Scenarios**:
  ```
  Scenario: Validator catches empty gold rows deterministically
    Tool: Bash
    Steps: python scripts/paper_a/validate_master_eval_jsonl.py --path data/paper_a/eval/query_gold_master_v0_5.jsonl --require-gold
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-02-master-validator.txt

  Scenario: Validator rejects malformed JSONL
    Tool: Bash
    Steps: python -c "from pathlib import Path; p=Path('.sisyphus/evidence/task-02-bad-master.jsonl'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{bad json\n', encoding='utf-8')" && python scripts/paper_a/validate_master_eval_jsonl.py --path .sisyphus/evidence/task-02-bad-master.jsonl > .sisyphus/evidence/task-02-master-validator-error.txt 2>&1; test $? -ne 0
    Expected: exit !=0; evidence file contains a JSON parsing error
    Evidence: .sisyphus/evidence/task-02-master-validator-error.txt
  ```

  ```
  Scenario: Regenerated master has balanced test rows across observability
    Tool: Bash
    Steps: python scripts/paper_a/rebuild_query_gold_master_splits.py --in data/paper_a/eval/query_gold_master.jsonl --out data/paper_a/eval/query_gold_master_v0_5.jsonl --seed 20260305 && python -c "import json; p='data/paper_a/eval/query_gold_master_v0_5_split_report.json'; d=json.load(open(p)); print(d)"
    Expected: exit 0; report includes non-zero `scope_nonempty_gold_test_counts` for explicit_device/implicit/explicit_equip and marks ambiguous as non-evaluable if non-empty gold is absent
    Evidence: .sisyphus/evidence/task-02-rebuild-master-split.txt
  ```

  ```
  Scenario: Frozen backup is byte-identical
    Tool: Bash
    Steps: python - <<'PY'
import hashlib
def sha256(p):
  h=hashlib.sha256()
  with open(p,'rb') as f:
    for b in iter(lambda: f.read(1024*1024), b''):
      h.update(b)
  return h.hexdigest()
print(sha256('data/paper_a/eval/query_gold_master.jsonl'))
print(sha256('data/paper_a/eval/query_gold_master_v0_4_frozen.jsonl'))
PY
    Expected: the two printed hashes are identical
    Evidence: .sisyphus/evidence/task-02-frozen-sha256.txt
  ```

  **Commit**: YES | Message: `data(paper-a): freeze eval set v0.5 and add validators` | Files: [`scripts/paper_a/validate_master_eval_jsonl.py`, `scripts/paper_a/rebuild_query_gold_master_splits.py`, `data/paper_a/eval/query_gold_master_v0_4_frozen.jsonl`, `data/paper_a/eval/query_gold_master_v0_5.jsonl`, `data/paper_a/eval/query_gold_master_v0_5_split_report.json`, `docs/papers/20_paper_a_scope/README.md`, `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`]

- [ ] 3. Policy Artifact Parity: fix doc_scope to include equip_id and align with paper definitions

  **What to do**:
  - Modify `scripts/paper_a/build_shared_and_scope.py` to include `es_equip_id` in each `doc_scope.jsonl` row.
  - Remove the hard-coded corpus-coupled assertion in `scripts/paper_a/build_shared_and_scope.py`:
    - Current behavior: raises if `shared_topic_count != 13`.
    - Replace with a drift-friendly invariant:
      - Always write `policy_snapshot.json` with `shared_topic_count` + `shared_doc_count`.
      - If an optional `--expected-shared-topic-count` is provided, compare and WARN (non-fatal) + write `drift_report.json`.
      - Never hard-fail on counts unless input integrity is broken (e.g., malformed JSONL).
  - Ensure the emitted `scope_level` assignment matches `data/paper_a/metadata/doc_type_map.csv.default_scope_level` + shared override.
  - Add a policy artifact validator `scripts/paper_a/validate_policy_artifacts.py`:
    - Checks: every row has `es_doc_id`, `es_device_name`, `es_doc_type`, `es_equip_id` (can be empty), `scope_level` in `{shared,device,equip}`, and boolean `is_shared`.
    - Cross-check: `scope_level==equip` implies `es_doc_type in {gcb,myservice}`.

  **Must NOT do**:
  - Do not backfill ES in this task; keep evaluation in doc-id mode (offline artifacts).

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: missing equip_id breaks equip-centric evaluation
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5,8 | Blocked By: 2

  **References**:
  - Policy builder: `scripts/paper_a/build_shared_and_scope.py`
  - Downstream expectation: `scripts/paper_a/evaluate_paper_a.py` (equip_candidates from `doc_scope.jsonl`)
  - Scope-level definition: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`
  - Doc type mapping: `data/paper_a/metadata/doc_type_map.csv`

  **Acceptance Criteria**:
  - [ ] Running policy build produces `doc_scope.jsonl` with `es_equip_id` present
  - [ ] `python scripts/paper_a/validate_policy_artifacts.py --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl` exits 0
  - [ ] `.sisyphus/evidence/paper-a/policy/family_map.json` exists and contains `device_to_family` and `families`

  **QA Scenarios**:
  ```
  Scenario: doc_scope includes es_equip_id
    Tool: Bash
    Steps: python scripts/paper_a/build_shared_and_scope.py --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --out-dir .sisyphus/evidence/paper-a/policy && python scripts/paper_a/build_family_map.py --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --shared-topics .sisyphus/evidence/paper-a/policy/shared_topics.json --out .sisyphus/evidence/paper-a/policy/family_map.json --tau 0.2 && python scripts/paper_a/validate_policy_artifacts.py --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl
    Expected: exit 0; validator passes
    Evidence: .sisyphus/evidence/task-03-policy-equipid.txt

  Scenario: Reject invalid scope_level
    Tool: Bash
    Steps: python -c "import json; print(json.dumps({'es_doc_id':'x','es_device_name':'y','es_doc_type':'sop','es_equip_id':'','topic':'t','is_shared':False,'scope_level':'BAD'}))" > .sisyphus/evidence/task-03-bad.jsonl && python scripts/paper_a/validate_policy_artifacts.py --doc-scope .sisyphus/evidence/task-03-bad.jsonl
    Expected: exit !=0 and points to invalid scope_level
    Evidence: .sisyphus/evidence/task-03-policy-equipid-error.txt
  ```

  **Commit**: YES | Message: `fix(paper-a): include equip_id in doc_scope artifacts` | Files: [`scripts/paper_a/build_shared_and_scope.py`, `scripts/paper_a/validate_policy_artifacts.py`]

- [ ] 4. Preregister comparisons, splits, tuning rules, and leakage guardrails (reviewer-proof)

  **What to do**:
  - Write `docs/papers/20_paper_a_scope/review/preregistration.md` that freezes:
    - Which evaluation slices are reported as “main” (recommended default; based on master `scope_observability`):
      - `explicit_device`, `implicit`, `ambiguous`
      - `explicit_equip` only if Task 3 passes and the slice has non-empty gold for the reported split
    - Which split is reported:
      - Tune/iterate on `split=dev`; report final numbers on `split=test` only.
    - Which systems are reported in main table (recommended default): `B0,B1,B2,B3,B4,P1` always; `P2,P3,P4,P6,P7` only if runnable.
    - Comparison set (fixed; runnable-minimum):
      - `B3 vs B4` (effect of hard filter)
      - `B4 vs P1` (shared policy)
    - Comparison set (optional extension; ONLY if corresponding systems are implemented + QA passes):
      - Router: `P1 vs P2` on implicit/ambiguous
      - Scoring: `P4 vs P6`, `P6 vs P7`
    - Tuning: dev-only; test locked; record tuned values in manifest.
    - Leakage rules: `v_scope` must be derived from inference-time metadata only; no test-label usage.

  **Must NOT do**:
  - Do not add new metrics mid-run; add only by updating preregistration with a version bump.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: reviewer-facing governance
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 8,11 | Blocked By: 1,2

  **References**:
  - Spec: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (§5, §6, §7)
  - Common protocol: `docs/papers/10_common_protocol/paper_common_protocol.md`

  **Acceptance Criteria**:
  - [ ] `docs/papers/20_paper_a_scope/review/preregistration.md` exists and lists fixed comparison pairs + tuning rules

  **QA Scenarios**:
  ```
  Scenario: Preregistration contains required keys
    Tool: Bash
    Steps: rg -n "Comparison set" docs/papers/20_paper_a_scope/review/preregistration.md && rg -n "dev-only" docs/papers/20_paper_a_scope/review/preregistration.md
    Expected: both phrases present
    Evidence: .sisyphus/evidence/task-04-prereg.txt

  Scenario: Missing preregistration breaks run (policy gate)
    Tool: Bash
    Steps: test -f docs/papers/20_paper_a_scope/review/preregistration.md
    Expected: file exists before Wave-3 runs start
    Evidence: .sisyphus/evidence/task-04-prereg-exists.txt
  ```

  **Commit**: YES | Message: `docs(paper-a): add preregistered comparison and tuning rules` | Files: [`docs/papers/20_paper_a_scope/review/preregistration.md`]

- [ ] 5. Evaluation Harness Parity: implement a master-schema evaluator with runnable system IDs

  **What to do**:
  - Create `scripts/paper_a/evaluate_paper_a_master.py` (do NOT retrofit legacy evaluator in a breaking way) that:
    - Consumes `data/paper_a/eval/query_gold_master_v0_5.jsonl` rows.
    - Supports deterministic row selection flags:
      - `--split {dev,test,all}` filters by the master file’s `split` field.
      - `--scope-observability {explicit_device,explicit_equip,implicit,ambiguous,all}` filters by `scope_observability`.
      - `--limit N` limits rows after filtering (smoke/debug only; must be recorded in `run_manifest.json`).
    - Uses `allowed_devices` / `allowed_equips` (multi-allowed) for contamination scoring.
    - Uses `shared_allowed` and `family_allowed` to determine allowed scope construction:
      - If `family_allowed`: expand allowed devices via `family_map.json`.
      - If `shared_allowed`: treat shared docs as in-scope for adjusted contamination.
    - Implements and reports (REQUIRED, reviewer-minimum):
      - `B0`: BM25
      - `B1`: Dense
      - `B2`: Hybrid (RRF)
      - `B3`: Hybrid + rerank
      - `B4`: Hard device filter (parsed from query); fallback to global
      - `P1`: Hard + Shared + scope_level-aware filter (doc-id mode via `backend/llm_infrastructure/retrieval/filters/scope_filter.py:build_scope_filter_by_doc_ids`)
    - Planned-not-reported (OPTIONAL extension; only run if implemented + QA passes in this branch):
      - `P2`, `P3`, `P4` (router + family/shared)
      - `P6`, `P7` (λ scoring)
    - Writes outputs:
      - `per_query.csv` including per-metric columns + `allowed_scope_size` + `router_top_m` + `router_confidence` (where applicable)
      - summary CSVs
      - `bootstrap_ci.json`
      - `mcnemar.json` (CE@k) for preregistered pairs
      - `run_manifest.json` including: resolved index, git sha, hashes of eval/policy/corpus

  **Must NOT do**:
  - Do not silently skip systems; if a system cannot run (e.g., router artifacts missing), mark `status=skipped` with a reason and exclude from stats.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: metric correctness + system parity + reproducibility
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6,7,8 | Blocked By: 2,3,4

  **References**:
  - Legacy evaluator patterns: `scripts/paper_a/evaluate_paper_a.py`
  - Scope filter DSL: `backend/llm_infrastructure/retrieval/filters/scope_filter.py`
  - Spec system matrix: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (§5)
  - Master eval schema: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (§13.3)
  - Hashing pattern (reuse): `scripts/paper_b/generate_synth_benchmark.py` (sha256 helpers for manifests)

  **Acceptance Criteria**:
  - [ ] `python scripts/paper_a/evaluate_paper_a_master.py --help` works
  - [ ] A smoke run on a small subset writes `per_query.csv`, summaries, `run_manifest.json`
  - [ ] `run_manifest.json` contains non-empty hashes for eval/policy/corpus files

  **QA Scenarios**:
  ```
  Scenario: Master evaluator smoke run
    Tool: Bash
    Steps: python scripts/paper_a/evaluate_paper_a_master.py --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl --systems B0,B2,B4,P1 --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --split test --scope-observability explicit_device --out-dir .sisyphus/evidence/paper-a/runs/smoke_001 --limit 20
    Expected: exit 0; per_query.csv and run_manifest.json exist
    Evidence: .sisyphus/evidence/task-05-master-eval-smoke.txt

  Scenario: Missing policy artifact fails fast
    Tool: Bash
    Steps: python scripts/paper_a/evaluate_paper_a_master.py --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl --systems P1 --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/does_not_exist.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --out-dir .sisyphus/evidence/paper-a/runs/error
    Expected: exit !=0 with clear missing-file error
    Evidence: .sisyphus/evidence/task-05-master-eval-error.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add master-schema evaluator with full system map` | Files: [`scripts/paper_a/evaluate_paper_a_master.py`]

- [ ] 6. Mandatory Baselines: add controls that isolate routing vs corpus-size effects

  **What to do**:
  - Implement in `scripts/paper_a/evaluate_paper_a_master.py` (or a helper module) these additional systems:
    - `C1_random_scope`: choose |S(q)| devices uniformly at random (matching proposed allowed scope size), apply same filter; report expected degradation.
    - `C2_global_postfilter`: retrieve globally (B3), then apply scope post-filter on ranked docs (drop OOS, backfill with next docs) and recompute metrics.
    - `C3_per_scope_merge`: retrieve top-k from each allowed device (or family) and merge (RRf or score-merge) without a router.
    - `C4_dedupe_only`: apply near-duplicate suppression by topic/doc_id prefix before any scope penalty (to separate “dup artifact” vs true scope confusion).

  **Must NOT do**:
  - Do not change the core B0..B4/P1..P7 definitions; add new IDs only.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: careful evaluator integration
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8 | Blocked By: 5

  **References**:
  - Reviewer baseline requirements captured in Oracle/Metis reviews
  - Spec baselines: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (§5)

  **Acceptance Criteria**:
  - [ ] Control systems appear in `--systems` help and produce metrics in `per_query.csv`
  - [ ] `C1_random_scope` uses a fixed RNG seed from run manifest

  **QA Scenarios**:
  ```
  Scenario: Random-scope control runs deterministically
    Tool: Bash
    Steps: python scripts/paper_a/evaluate_paper_a_master.py --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --split test --scope-observability implicit --systems C1_random_scope --seed 123 --limit 50 --out-dir .sisyphus/evidence/paper-a/runs/random_001
    Expected: exit 0; run_manifest.json.seed==123; repeated run yields identical per_query router scopes
    Evidence: .sisyphus/evidence/task-06-random-scope.txt

  Scenario: Postfilter control does not crash on empty in-scope hits
    Tool: Bash
    Steps: python scripts/paper_a/evaluate_paper_a_master.py --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --split test --scope-observability ambiguous --systems C2_global_postfilter --seed 123 --limit 20 --out-dir .sisyphus/evidence/paper-a/runs/postfilter_001
    Expected: exit 0; per_query.csv has status=ok or a clear skip_reason per row
    Evidence: .sisyphus/evidence/task-06-postfilter.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add control baselines for routing vs corpus-size effects` | Files: [`scripts/paper_a/evaluate_paper_a_master.py`]

- [ ] 7. Statistics + Repeatability: expand paired tests and apply multiple-comparison correction

  **What to do**:
  - In `scripts/paper_a/evaluate_paper_a_master.py`:
    - Add paired bootstrap CI for each preregistered comparison and each metric: `adj_cont@5`, `hit@5`, `mrr`, `ce@5`.
    - Add McNemar test for `ce@5` (paired binary) for the same comparisons.
    - Add Holm-Bonferroni correction across the set of hypothesis tests.
  - Add optional repeatability mode:
    - `--repeats N` repeats each system with fixed seed and reports Stability@k (Jaccard) for top-k doc_id sets.
    - Report mean±CI across repeats only for latency, not for relevance metrics (which should be deterministic under fixed seed).

  **Must NOT do**:
  - Do not add “significance fishing” comparisons beyond preregistration; require explicit update.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: statistical correctness + reviewer scrutiny
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8 | Blocked By: 5

  **References**:
  - Common protocol stats: `docs/papers/10_common_protocol/paper_common_protocol.md` (§6.3)
  - Spec stats: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (§7)
  - Stability/Jaccard pattern (reuse): `scripts/paper_b/run_paper_b_eval.py` (repeat stability via Jaccard@k)

  **Acceptance Criteria**:
  - [ ] `bootstrap_ci.json` includes entries for all preregistered comparisons
  - [ ] `mcnemar.json` exists and includes corrected p-values

  **QA Scenarios**:
  ```
  Scenario: Stats artifacts generated
    Tool: Bash
    Steps: python scripts/paper_a/evaluate_paper_a_master.py --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --split test --scope-observability explicit_device --systems B3,B4,P1 --seed 123 --limit 100 --bootstrap-samples 2000 --out-dir .sisyphus/evidence/paper-a/runs/stats_001
    Expected: bootstrap_ci.json + mcnemar.json exist and contain numeric fields
    Evidence: .sisyphus/evidence/task-07-stats.txt

  Scenario: Repeatability mode emits stability metrics
    Tool: Bash
    Steps: python scripts/paper_a/evaluate_paper_a_master.py --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl --corpus-filter .sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt --doc-scope .sisyphus/evidence/paper-a/policy/doc_scope.jsonl --family-map .sisyphus/evidence/paper-a/policy/family_map.json --split test --scope-observability explicit_device --systems B3 --seed 123 --limit 50 --repeats 3 --out-dir .sisyphus/evidence/paper-a/runs/repeat_001
    Expected: output contains stability@k columns or a separate stability report file
    Evidence: .sisyphus/evidence/task-07-repeatability.txt
  ```

  **Commit**: YES | Message: `feat(paper-a): add paired stats and repeatability audit` | Files: [`scripts/paper_a/evaluate_paper_a_master.py`]

- [ ] 8. Evidence Runs: execute preregistered experiments and capture reproducible artifacts

  **What to do**:
  - Execute the evaluator on `data/paper_a/eval/query_gold_master_v0_5.jsonl` for each preregistered evaluation slice (reporting by `scope_observability`):
    - `explicit_device`
    - `implicit`
    - `ambiguous`
    - `explicit_equip`
  - For each run, store outputs under `.sisyphus/evidence/paper-a/runs/{run_id}/`.
  - Generate a run index file `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md` linking:
    - run_id → split → systems → key results CSV paths → manifest hash

  **Must NOT do**:
  - Do not rerun with changing parameters without bumping preregistration version.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: careful run hygiene + artifacts
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 9,12 | Blocked By: 5,6,7

  **References**:
  - Run artifact expectations: `docs/papers/10_common_protocol/paper_common_protocol.md` (§6.1)

  **Acceptance Criteria**:
  - [ ] For each reported slice (`scope_observability` × `split=test`), a run directory exists with `per_query.csv`, summaries, stats, and manifest
  - [ ] `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md` exists and references all run dirs

  **QA Scenarios**:
  ```
  Scenario: Full run set produced
    Tool: Bash
    Steps: ls .sisyphus/evidence/paper-a/runs/ && test -f docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md
    Expected: run dirs exist and index file exists
    Evidence: .sisyphus/evidence/task-08-runs.txt

  Scenario: Manifest contains hashes
    Tool: Bash
    Steps: python -c "import json; p='.sisyphus/evidence/paper-a/runs/smoke_001/run_manifest.json'; d=json.load(open(p)); print(bool(d.get('hashes')) and bool(d.get('resolved_index')))"
    Expected: prints True
    Evidence: .sisyphus/evidence/task-08-manifest-hashes.txt
  ```

  **Commit**: NO | Message: `n/a` | Files: none (do not commit `.sisyphus/evidence/**`)

- [ ] 9. Error Analysis: build a failure taxonomy and quantify which errors decrease

  **What to do**:
  - Create `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md` including:
    - Taxonomy definitions (binary, reproducible): wrong-scope evidence, mixed-scope evidence, shared-false-positive contamination, missing-gold, router-miss.
    - Counts per system (at least B3, B4, P1, P4/P7 if available) by split.
    - 10 representative cases with q_id and top-k doc_ids (no private content), pointing to per_query rows.

  **Must NOT do**:
  - Do not handpick cases without documenting selection rule (e.g., highest adj_cont@5 deltas).

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: reviewer-facing error analysis
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 11,12 | Blocked By: 8

  **References**:
  - Expected error table: `docs/papers/20_paper_a_scope/evidence_mapping.md` (Tab-A3)

  **Acceptance Criteria**:
  - [ ] Error analysis evidence file exists and includes taxonomy + counts + example list

  **QA Scenarios**:
  ```
  Scenario: Error analysis includes required sections
    Tool: Bash
    Steps: rg -n "Taxonomy" docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md && rg -n "Representative cases" docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md
    Expected: both sections present
    Evidence: .sisyphus/evidence/task-09-error-analysis.txt

  Scenario: Examples reference q_id
    Tool: Bash
    Steps: rg -n "A-" docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md
    Expected: contains q_id-like identifiers
    Evidence: .sisyphus/evidence/task-09-error-analysis-qids.txt
  ```

  **Commit**: YES | Message: `docs(paper-a): add error analysis evidence` | Files: [`docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_error_analysis.md`]

- [ ] 10a. New Hypotheses + Experiment Matrix: add falsifiable hypotheses and map each to runnable comparisons

  **What to do**:
  - Write `docs/papers/20_paper_a_scope/review/hypotheses_experiments.md` containing:
    - 8–12 hypotheses (H1..H12). Each hypothesis MUST include:
      - Dataset slice definition (exact filter over master fields: `split`, `scope_observability`, optional `intent_primary`, optional doc_type slice).
      - Systems compared (must be from preregistration), plus any new control baseline IDs.
      - Primary metric + secondary metrics.
      - Expected outcome and what failure implies (diagnostic interpretation).
      - Exact evaluator command template (using `scripts/paper_a/evaluate_paper_a_master.py` with concrete flags).
    - A final table mapping: hypothesis → evidence file(s) that will be generated (run_id + output paths).

  **Must NOT do**:
  - Do not add hypotheses that require labels you do not have (e.g., passage-level gold) unless you also add a concrete label-building plan.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: experimental design + falsifiability framing
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 8,11,12 | Blocked By: 4,5

  **References**:
  - Oracle hypothesis set (internal): captured in planning session
  - Spec ablations: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (§5)
  - Master schema fields: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (§13.3)

  **Acceptance Criteria**:
  - [ ] `docs/papers/20_paper_a_scope/review/hypotheses_experiments.md` exists
  - [ ] Contains at least `H1` and `H8` headings and at least 1 concrete command per hypothesis

  **QA Scenarios**:
  ```
  Scenario: Hypotheses file has required structure
    Tool: Bash
    Steps: rg -n "^## H1" docs/papers/20_paper_a_scope/review/hypotheses_experiments.md && rg -n "^## H8" docs/papers/20_paper_a_scope/review/hypotheses_experiments.md && rg -n "python scripts/paper_a/evaluate_paper_a_master\.py" docs/papers/20_paper_a_scope/review/hypotheses_experiments.md
    Expected: H1 and H8 present; evaluator command appears multiple times
    Evidence: .sisyphus/evidence/task-10a-hypotheses.txt

  Scenario: No non-runnable system IDs
    Tool: Bash
    Steps: rg -n "\bP[0-9]\b" docs/papers/20_paper_a_scope/review/hypotheses_experiments.md
    Expected: any referenced P-system is either implemented or explicitly marked "planned-not-reported" in preregistration
    Evidence: .sisyphus/evidence/task-10a-hypotheses-ids.txt
  ```

  **Commit**: YES | Message: `docs(paper-a): add falsifiable hypotheses and experiment matrix` | Files: [`docs/papers/20_paper_a_scope/review/hypotheses_experiments.md`]

- [ ] 10. Reviewer Report: produce a critical, publishability-focused critique (advisor tone)

  **What to do**:
  - Write `docs/papers/20_paper_a_scope/review/reviewer_report.md` with:
    - 15–25 concrete criticisms (definitions, evaluation validity, baseline fairness, leakage, novelty risk).
    - A “fatal flaws if unaddressed” section (explicitly list) and “mitigations implemented” with links.
    - A novelty positioning paragraph referencing `docs/papers/20_paper_a_scope/related_work.md` clusters.
    - A “stats validity” subsection that explicitly cites prior IR guidance on multiplicity adjustments:
      - `Deciding on an adjustment for multiplicity in IR experiments` (SIGIR 2013)
    - A checklist mapping each criticism → either (a) resolved by evidence file path or (b) explicitly deferred.
    - Include a “Matryoshka reality check” subsection:
      - State current repo embedding implementation is SentenceTransformer-based and not Matryoshka/MRL.
      - List exactly what must be implemented + what fair baselines are required before claiming any Matryoshka contribution.
      - Cite Matryoshka primary reference: Kusupati et al., NeurIPS 2022 (`arXiv:2205.13147`) and its reference implementation (RAIVNLab/MRL).
      - If discussing 2D variants: cite `2D Matryoshka Sentence Embeddings` (`arXiv:2402.14776`).

  **Must NOT do**:
  - Do not write the report as praise; it must read like a skeptical reviewer.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: critical academic writing
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 11 | Blocked By: 1,4,8,9

  **References**:
  - Related work scaffold: `docs/papers/20_paper_a_scope/related_work.md`
  - Spec: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`

  **Acceptance Criteria**:
  - [ ] Reviewer report exists and includes “fatal flaws” + “mitigations” sections

  **QA Scenarios**:
  ```
  Scenario: Reviewer report contains required sections
    Tool: Bash
    Steps: rg -n "Fatal flaws" docs/papers/20_paper_a_scope/review/reviewer_report.md && rg -n "Mitigations implemented" docs/papers/20_paper_a_scope/review/reviewer_report.md
    Expected: both present
    Evidence: .sisyphus/evidence/task-10-reviewer-report.txt

  Scenario: Report references evidence files
    Tool: Bash
    Steps: rg -n "docs/papers/20_paper_a_scope/evidence/" docs/papers/20_paper_a_scope/review/reviewer_report.md
    Expected: at least 3 evidence links
    Evidence: .sisyphus/evidence/task-10-reviewer-report-links.txt
  ```

  **Commit**: YES | Message: `docs(paper-a): add reviewer-style critique report` | Files: [`docs/papers/20_paper_a_scope/review/reviewer_report.md`]

- [ ] 11. Methodology Rewrite: convert the provided Methodology draft into a reviewer-proof section + Algorithm 1

  **What to do**:
  - Create manuscript file `docs/papers/20_paper_a_scope/paper_a_scope.md` (this repo’s Paper A manuscript) containing at minimum:
    - `3. Methodology` with subsections 3.1–3.11 aligned to your draft, but with decision-locked definitions:
      - Formal definitions for: contamination; allowed scope; doc scope_level; shared/family; router uncertainty; `v_scope` (doc-type-specific);
      - Explicit statement of inference-time inputs vs label-only artifacts (leakage guardrail).
      - Clarify “Base(d,q)” scale and normalization when combining with λ.
    - `Algorithm 1` (pseudo-code) consistent with the master evaluator systems and the spec.
    - A symbol table and metric table (Raw/Adjusted/Shared Cont@k, CE@k, Hit@k, MRR, ScopeAccuracy@M, latency).
  - Ensure the text matches what is actually runnable in the harness (if router/Matryoshka is not reported, write it as “optional module” with ablation).
  - If Matryoshka is mentioned:
    - Explicitly separate “current implementation” vs “planned extension” and cite the current embedder stack limitations:
      - `backend/llm_infrastructure/embedding/adapters/sentence.py`
      - `backend/llm_infrastructure/embedding/engines/sentence/embedder.py`

  **Must NOT do**:
  - Do not claim Matryoshka results unless Task 8 produced them and the baseline is fair.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: academic prose + consistency with evidence
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 12 | Blocked By: 4,8

  **References**:
  - Spec: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`
  - Metrics: `docs/papers/20_paper_a_scope/paper_a_scope_spec.md` (§4)
  - Scope filter semantics: `backend/llm_infrastructure/retrieval/filters/scope_filter.py`

  **Acceptance Criteria**:
  - [ ] `docs/papers/20_paper_a_scope/paper_a_scope.md` exists
  - [ ] Contains headings `# 3. Methodology` and `Algorithm 1`
  - [ ] Contains a metric definition table and a symbol table

  **QA Scenarios**:
  ```
  Scenario: Manuscript file contains required sections
    Tool: Bash
    Steps: rg -n "^# 3\\. Methodology" docs/papers/20_paper_a_scope/paper_a_scope.md && rg -n "Algorithm 1" docs/papers/20_paper_a_scope/paper_a_scope.md && rg -n "Symbol" docs/papers/20_paper_a_scope/paper_a_scope.md
    Expected: all present
    Evidence: .sisyphus/evidence/task-11-methodology.txt

  Scenario: No ungrounded claims
    Tool: Bash
    Steps: rg -n "we show" docs/papers/20_paper_a_scope/paper_a_scope.md
    Expected: any such claims are followed by an evidence link or a conditional phrasing
    Evidence: .sisyphus/evidence/task-11-claims-audit.txt
  ```

  **Commit**: YES | Message: `docs(paper-a): add manuscript methodology with algorithm and tables` | Files: [`docs/papers/20_paper_a_scope/paper_a_scope.md`]

- [ ] 12. Experimental Setup + Results Tables: make the paper’s evaluation section executable and evidence-linked

  **What to do**:
  - Add `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_main_results.md` that renders:
    - Tab-A1 equivalent: systems × {Raw/Adj/Shared Cont@5, CE@5, Hit@5, MRR, latency} by split.
    - Optional: Tab-A4 equivalent: P4 vs P6 vs P7 (ONLY if implemented + QA passes).
  - Update `docs/papers/20_paper_a_scope/evidence_mapping.md` to link claim rows to evidence files produced in Tasks 8–9 and result tables here.
  - Add “Experimental Setup” section to `docs/papers/20_paper_a_scope/paper_a_scope.md` describing:
    - Dataset construction (`data/paper_a/...`), split definitions, baseline implementations, metrics, stats tests, and latency measurement conditions.

  **Must NOT do**:
  - Do not paste large CSVs into markdown; render summary tables only and link to run dirs.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: paper-facing results formatting + evidence linking
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: none | Blocked By: 8,9,11

  **References**:
  - Evidence map: `docs/papers/20_paper_a_scope/evidence_mapping.md`
  - Expected tables: `docs/papers/20_paper_a_scope/evidence_mapping.md` (Tab-A1/Tab-A4)

  **Acceptance Criteria**:
  - [ ] Main results evidence file exists
  - [ ] Evidence mapping updated with paths to main results and error analysis
  - [ ] Manuscript includes Experimental Setup section with explicit commands/run IDs

  **QA Scenarios**:
  ```
  Scenario: Results tables exist and link to runs
    Tool: Bash
    Steps: test -f docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_main_results.md && rg -n "\.sisyphus/evidence/paper-a/runs/" docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_main_results.md
    Expected: file exists and contains run-dir references
    Evidence: .sisyphus/evidence/task-12-results-tables.txt

  Scenario: Evidence mapping references the new evidence files
    Tool: Bash
    Steps: rg -n "2026-03-05_paper_a_main_results" docs/papers/20_paper_a_scope/evidence_mapping.md
    Expected: at least 1 match
    Evidence: .sisyphus/evidence/task-12-evidence-mapping.txt
  ```

  **Commit**: YES | Message: `docs(paper-a): add main results tables and evidence links` | Files: [`docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_main_results.md`, `docs/papers/20_paper_a_scope/evidence_mapping.md`, `docs/papers/20_paper_a_scope/paper_a_scope.md`]

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Reproducibility Audit (hashes, manifests, non-committed evidence) — deep
- [ ] F4. Scope Fidelity Check (no ungrounded claims; evidence-linked) — deep

## Commit Strategy
- Commit doc/code changes that enable evaluation reproducibility and manuscript text.
- Never commit `.sisyphus/evidence/**` or other run artifacts.
- Recommended commit granularity:
  - `feat(paper-a): add master eval validator`
  - `fix(paper-a): include equip_id in doc_scope artifacts`
  - `feat(paper-a): add master-schema evaluator`
  - `docs(paper-a): add reviewer report and manuscript sections`

## Success Criteria
- Reviewer report identifies and closes (or explicitly defers) all high-risk validity issues.
- Paper A’s main claims can be traced: claim → evidence file → run manifest → reproducible command.
- New hypotheses are testable with declared comparisons, not narrative-only.
  - After implementing the above, generate the policy artifacts under `.sisyphus/evidence/paper-a/policy/` and build the family map:
    - `python scripts/paper_a/build_shared_and_scope.py --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --out-dir .sisyphus/evidence/paper-a/policy`
    - `python scripts/paper_a/build_family_map.py --corpus-meta .sisyphus/evidence/paper-a/corpus/doc_meta.jsonl --shared-topics .sisyphus/evidence/paper-a/policy/shared_topics.json --out .sisyphus/evidence/paper-a/policy/family_map.json --tau 0.2`
