# SOP-Only Filter Eval: Retrieval/Answer Quality Audit + Output Standardization

## TL;DR
> **Summary**: Make SOP eval results *auditable and correctly-measured*, then standardize answer formatting so outputs are consistent and automatically verifiable.
> **Deliverables**: fixed hit-metric logic, richer JSONL artifacts, schema/validators, strict answer template + format checks.
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: Fix hit computation + logging → add validators/metrics → enforce answer format

## Context
### Original Request
- Review `.sisyphus/evidence/2026-03-11_sop_filter_eval/sop_only_results.jsonl` to judge whether retrieved documents match questions and whether answers are good.
- Check inconsistent answer formatting and propose improvements.

### Repo Facts (grounded)
- SOP filter eval generator: `scripts/evaluation/run_sop_filter_eval.py`.
  - Hit logic: `_normalize_doc_name()` + `_check_hit()` (`scripts/evaluation/run_sop_filter_eval.py:76`, `scripts/evaluation/run_sop_filter_eval.py:98`).
- Output JSONL truncates answer twice: `EvalResult.answer = answer[:500]` (`scripts/evaluation/run_sop_filter_eval.py:216`) and `answer_preview = r.answer[:200]` (`scripts/evaluation/run_sop_filter_eval.py:284`).
- Current answer prompts are intentionally minimal and do not define a strict template:
  - `backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml`
  - `backend/llm_infrastructure/llm/prompts/general_ans_v2.yaml`
  - `backend/llm_infrastructure/llm/prompts/ts_ans_v2.yaml`
- REFS text fed to the LLM is rendered by `ref_json_to_text()` in `backend/llm_infrastructure/llm/langgraph_agent.py:833`.

### Evidence Review (what we can and cannot conclude)
- The JSONL contains only `retrieved_doc_ids` (top 10 doc_ids only, no pages/scores) and `answer_preview` (200 chars). This is enough to compute coarse hit rates but *insufficient* to reliably evaluate:
  - whether the retrieved evidence is actually relevant beyond the gold match,
  - whether the answer is faithful to evidence,
  - whether citations/references exist (they may appear after the 200-char preview).
- Observed issues that directly affect conclusions:
  - `hit_doc/hit_page` can be false-negative when `gold_doc` contains punctuation (e.g., `&`) but retrieved `doc_id` omits it, due to weak normalization (`scripts/evaluation/run_sop_filter_eval.py:76`).
  - Many answer previews show inconsistent formatting (emoji numbering, tables, mixed headings/language), consistent with prompts not specifying a strict output template.

### Metis Review (gaps addressed)
- Fix measurement correctness before tuning: guard against empty `gold_doc` always-hit bug (even if current dataset has none), strengthen normalization, and make page casting safe.
- Add versioned schemas + validators; log enough fields to explain why a row was marked hit/miss.
- Add automated format compliance scoring; do not rely on subjective spot checks.

## Work Objectives
### Core Objective
- Produce evaluation artifacts that allow automatic and trustworthy assessment of (1) retrieval correctness at doc/page level and (2) answer quality/format compliance.

### Deliverables
- D1: Robust `hit_doc`/`hit_page` computation (no punctuation false-negatives; safe casting; empty gold guards).
- D2: Rich, versioned JSONL schema for SOP eval outputs (thin summary + optional raw), including retrieved doc metadata and full answer.
- D3: Validators + small reporting script (format compliance + language + citation checks).
- D4: Standard answer format (strict Markdown template) enforced via prompts + validator/retry.

### Definition of Done
- Running `python scripts/evaluation/run_sop_filter_eval.py --use-testclient` produces JSONL that is machine-validated and contains enough information to audit misses (no external API process required).
- `hit_doc/hit_page` for the four previously-missed SW install rows become correct after normalization fix (or are explained by retrieval truly missing gold).
- Answer formatting variance is reduced by policy: no emoji numbering; consistent headings; consistent references section.

### Must Have
- Schema versioning in every eval JSONL record.
- Full, untruncated answer in evaluation artifacts (or a separate `*_raw.jsonl`).
- Logging includes `route`, `detected_language`, `target_language`, and prompt template/version used.

### Must NOT Have
- Do NOT change retrieval/reranking behavior (dedupe/diversity) until measurement/logging is correct.
- Do NOT claim answer faithfulness based on `answer_preview` alone.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Tests-after (no existing unit tests for these scripts assumed; add minimal checks).
- Evidence outputs:
  - `.sisyphus/evidence/sop_eval/task-01-schema-sample.jsonl`
  - `.sisyphus/evidence/sop_eval/task-03-format-report.json`

## Execution Strategy
### Parallel Execution Waves
Wave 1
- Task 1-3: Fix measurement + schema/logging + validators (independent but share schema decisions).

Wave 2
- Task 4-6: Prompt tightening + answer format validator/retry + format scoring in eval.

### Dependency Matrix
- Task 1 blocks Task 2 (shared normalization helpers) and Task 3 (validator expectations).
- Task 2 blocks Task 5 (must log template/language fields).
- Task 4 blocks Task 5 (validator expectations depend on final template).

## TODOs

- [x] 1. Harden hit computation and fix known false negatives

  **What to do**:
  - Update doc name normalization in `scripts/evaluation/run_sop_filter_eval.py` to match the stricter style used elsewhere (strip extensions; collapse non-alphanum to `_`; collapse repeats; trim underscores).
  - Add explicit guard: if `gold_doc` is blank after normalization, set `hit_doc=False` and `hit_page=False`.
  - Make page parsing safe (wrap `int(page)` in try/except; treat parse failures as no page hit).
  - Add `hit_rank` (first rank where gold doc matches) and `hit@k` (k=1,3,5,10) to the written JSONL.
  - Add `match_debug` fields: `matched_field` in `{doc_id,title,source}` and `matched_value` (normalized), to explain hits.

  **Must NOT do**:
  - Do not change the backend retrieval; only evaluation-time matching.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — cross-file correctness + metrics.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2,3 | Blocked By: -

  **References**:
  - Code: `scripts/evaluation/run_sop_filter_eval.py:76` — current weak normalization.
  - Code: `scripts/evaluation/run_sop_filter_eval.py:98` — current `_check_hit()` logic.
  - Pattern: `scripts/evaluation/evaluate_sop_agent_page_hit.py:32` — stricter `_normalize_doc_id()`.

  **Acceptance Criteria**:
  - [ ] `python scripts/evaluation/run_sop_filter_eval.py` exits 0.
  - [ ] Re-running eval yields `hit_doc=true` for SW-install rows where retrieved `doc_id` clearly matches gold except punctuation (regression test case added).

  **QA Scenarios**:
  ```
  Scenario: Punctuation normalization ('&')
    Tool: Bash
    Steps:
      1) Run python scripts/evaluation/run_sop_filter_eval.py
      2) Parse output JSONL and assert rows with gold_doc containing '&' can still match doc_id without '&'
    Expected: hit_doc and hit_page reflect true match, not normalization artifact
    Evidence: .sisyphus/evidence/sop_eval/task-01-normalization.json

  Scenario: Missing/invalid page values
    Tool: Bash
    Steps:
      1) Inject (in a unit test or synthetic list) a retrieved_doc with page='N/A'
      2) Run _check_hit and ensure it does not crash
    Expected: hit_doc may be true; hit_page remains false; no exceptions
    Evidence: .sisyphus/evidence/sop_eval/task-01-page-parse.json
  ```

  **Commit**: YES | Message: `fix(eval): harden SOP hit computation and add hit@k`

- [x] 2. Expand SOP eval JSONL schema (auditable, versioned)

  **What to do**:
  - Add `schema_version` (e.g., `sop_eval_v1`) to each JSONL record in `scripts/evaluation/run_sop_filter_eval.py`.
  - Log both thin and rich artifacts:
    - Thin: current keys + new hit@k.
    - Rich: `request_payload`, `response_metadata`, `top_docs` (rank, doc_id, title/source, page, score, doc_type, device_name, chunk_id), and full `answer`.
  - Parameterize output directory and preview lengths via CLI flags (default to current path but allow override).
  - Add execution mode flags:
    - `--api-base-url` (default `http://localhost:8001`)
    - `--use-testclient` to run via FastAPI TestClient (pattern: `scripts/evaluation/evaluate_sop_agent_page_hit.py:_post_json_testclient`).
  - Include `route`, `search_queries` (from API response metadata if available), `detected_language`, `target_language`, and `template_version`.

  **Must NOT do**:
  - Do not remove the existing thin JSONL without a conversion/compat plan.

  **Recommended Agent Profile**:
  - Category: `deep` — schema design + backward compatibility.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5 | Blocked By: 1

  **References**:
  - Writer: `scripts/evaluation/run_sop_filter_eval.py:273` — current JSONL write block.
  - API metadata: `backend/api/routers/agent.py:685` — how `target_language`, `route`, `search_queries` appear in response metadata.

  **Acceptance Criteria**:
  - [ ] Output JSONL records parse as JSON and include `schema_version` and `top_docs` (rich file) and `answer` (full).

  **QA Scenarios**:
  ```
  Scenario: Audit a miss without rerunning
    Tool: Bash
    Steps:
      1) Run python scripts/evaluation/run_sop_filter_eval.py --out-dir .sisyphus/evidence/sop_eval
      2) Pick a row with hit_doc=false
      3) Confirm the row includes top_docs with doc_id/title/source/page and match_debug
    Expected: Enough data exists to explain the miss from the JSONL alone
    Evidence: .sisyphus/evidence/sop_eval/task-02-miss-audit.json

  Scenario: Backward compatible thin artifact
    Tool: Bash
    Steps:
      1) Ensure the original `{label}_results.jsonl` still exists or a `{label}_thin.jsonl` equivalent is written
    Expected: Downstream consumers relying on thin schema do not break
    Evidence: .sisyphus/evidence/sop_eval/task-02-thin-schema.json
  ```

  ```
  Scenario: Run without external API (TestClient)
    Tool: Bash
    Steps:
      1) Run python scripts/evaluation/run_sop_filter_eval.py --use-testclient --out-dir .sisyphus/evidence/sop_eval
    Expected: Script completes and produces the same schema without requiring make run-api
    Evidence: .sisyphus/evidence/sop_eval/task-02-testclient.json
  ```

  **Commit**: YES | Message: `feat(eval): emit versioned SOP eval JSONL with rich audit fields`

- [x] 3. Add JSONL validators + format/language/citation checks

  **What to do**:
  - Add a validator script for SOP eval JSONL (`scripts/evaluation/validate_sop_eval_jsonl.py`) that checks:
    - required keys, types, schema_version
    - `answer` non-empty when generation is expected
    - `top_docs` list shape
  - Add lightweight answer checks:
    - language compliance vs `target_language`
    - presence of a `참고문헌`/`References` section
    - citation token coverage (at least one `[N]` appears when REFS not empty)
  - Make `run_sop_filter_eval.py` optionally run the validator at end and write a `report.json` summary.

  **Recommended Agent Profile**:
  - Category: `quick` — small scripts and checks.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6 | Blocked By: 2

  **References**:
  - Existing validator (shape idea): `scripts/evaluation/validate_agent_eval_jsonl.py`.
  - Language selection plumbing: `backend/api/routers/agent.py:685`.

  **Acceptance Criteria**:
  - [ ] `python scripts/evaluation/validate_sop_eval_jsonl.py --jsonl <output>` returns exit 0.
  - [ ] Report includes counts for: `format_ok`, `language_ok`, `citations_ok`.

  **QA Scenarios**:
  ```
  Scenario: Detect language drift
    Tool: Bash
    Steps:
      1) Run eval
      2) Run validator
    Expected: Rows with English answers under target_language=ko are flagged
    Evidence: .sisyphus/evidence/sop_eval/task-03-language-report.json

  Scenario: Detect missing references section
    Tool: Bash
    Steps:
      1) Feed validator a synthetic answer missing '참고문헌'
    Expected: citations_ok=false or references_ok=false
    Evidence: .sisyphus/evidence/sop_eval/task-03-format-report.json
  ```

  **Commit**: YES | Message: `test(eval): validate SOP eval JSONL and answer format`

- [x] 4. Define and enforce a single canonical Markdown answer template for SOP procedures

  **What to do**:
  - Update answer prompt templates to include an explicit, strict template (no emoji numbering, no Markdown tables by default):
    - `backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml`
    - `backend/llm_infrastructure/llm/prompts/general_ans_v2.yaml`
    - `backend/llm_infrastructure/llm/prompts/ts_ans_v2.yaml`
  - Canonical template (Korean, SOP-like):
    - Title line
    - `## 준비/안전`
    - `## 작업 절차` with `1.` numbering only
    - `## 복구/확인`
    - `## 주의사항`
    - `## 참고문헌` listing `[N] doc_id (device_name)`
  - Explicitly forbid: emoji numerals (e.g., `1️⃣`), tables (`|---|`), and mixed-language headings.

  **Recommended Agent Profile**:
  - Category: `writing` — prompt spec precision.
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 5 | Blocked By: 3

  **References**:
  - Current minimal prompts: `backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml`, `backend/llm_infrastructure/llm/prompts/general_ans_v2.yaml`, `backend/llm_infrastructure/llm/prompts/ts_ans_v2.yaml`.
  - REFS format: `backend/llm_infrastructure/llm/langgraph_agent.py:833`.

  **Acceptance Criteria**:
  - [ ] For a representative SOP query, answer contains all required sections and uses plain `1.` numbering.

  **QA Scenarios**:
  ```
  Scenario: Format compliance (no emoji/tables)
    Tool: Bash
    Steps:
      1) Run a single query via /api/agent/run with SOP doc_types
      2) Check output contains '## 참고문헌' and does not contain '1️⃣' nor a markdown table row '|---|'
    Expected: Strict template is followed
    Evidence: .sisyphus/evidence/sop_eval/task-04-single-answer.txt

  Scenario: Multi-device REFS separation
    Tool: Bash
    Steps:
      1) Provide REFS containing two different device_name tags
      2) Generate answer
    Expected: Answer splits by device and does not merge values
    Evidence: .sisyphus/evidence/sop_eval/task-04-multidevice.txt
  ```

  **Commit**: YES | Message: `chore(prompts): standardize SOP answer markdown template`

- [x] 5. Add answer format validator + retry loop (enforce target_language, citations, sections)

  **What to do**:
  - Implement a post-generation check in `backend/llm_infrastructure/llm/langgraph_agent.py` near `answer_node()`:
    - validate required sections present
    - validate numbering style
    - validate citations exist when REFS not EMPTY
    - validate language matches `target_language`
  - If validation fails, re-invoke LLM with a short corrective instruction (same REFS) up to N retries (N=1 or 2).
  - Log validator results into response metadata for evaluation scripts to capture.

  **Must NOT do**:
  - Do not introduce JSON-in-prompt; keep REFS as text (`ref_json_to_text()` rationale).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — correctness + retry safety.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6 | Blocked By: 4

  **References**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py:2085` — `answer_node()`.
  - `backend/llm_infrastructure/llm/langgraph_agent.py:833` — REFS text formatting.

  **Acceptance Criteria**:
  - [ ] Running eval yields `format_ok` rate >= 95% on the dataset.

  **QA Scenarios**:
  ```
  Scenario: Retry corrects emoji numbering
    Tool: Bash
    Steps:
      1) Force a model response that uses '1️⃣' (via seed or by mocking)
      2) Ensure validator triggers retry
    Expected: Final answer uses '1.' numbering
    Evidence: .sisyphus/evidence/sop_eval/task-05-retry.txt

  Scenario: Enforce Korean under target_language=ko
    Tool: Bash
    Steps:
      1) Send an English question with target_language='ko'
      2) Validate final answer is Korean (validator passes)
    Expected: No English-only answer slips through
    Evidence: .sisyphus/evidence/sop_eval/task-05-language.txt
  ```

  **Commit**: YES | Message: `feat(agent): validate SOP answer format and retry on violations`

- [x] 6. Re-run evaluation and produce a comparative report (before/after)

  **What to do**:
  - Re-run `run_sop_filter_eval.py` and produce:
    - hit_doc/hit_page/hit@k
    - format compliance stats (no emoji/tables; references present; language)
    - top failure examples with `match_debug`.
  - Save report JSON + a short Markdown summary in `.sisyphus/evidence/sop_eval/`.

  **Recommended Agent Profile**:
  - Category: `quick` — reporting + scripting.
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: - | Blocked By: 1,2,3,5

  **Acceptance Criteria**:
  - [ ] Report file exists and includes both retrieval and format metrics.

  **QA Scenarios**:
  ```
  Scenario: Before/after regression detection
    Tool: Bash
    Steps:
      1) Run eval on baseline commit
      2) Run eval after changes
      3) Compare hit_doc/hit_page and format_ok deltas
    Expected: No retrieval metric regression; format_ok improves
    Evidence: .sisyphus/evidence/sop_eval/task-06-diff.json
  ```

  **Commit**: NO

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Eval Artifact Audit (schema/validator) — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Keep commits atomic by concern:
  - Eval hit logic + metrics
  - Eval schema/logging
  - Validators
  - Prompt template standardization
  - Agent retry validator

## Success Criteria
- SOP eval hit rates are trustworthy (no known false negatives from punctuation; guarded against empty gold cases).
- Evaluation artifacts allow row-level auditing without reruns.
- Answer outputs follow a single template with high compliance and are consistent across the dataset.
