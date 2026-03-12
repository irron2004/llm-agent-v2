# Learnings (append-only)

- `sop_only_results.jsonl` is a thin artifact: `retrieved_doc_ids` (top10, no pages/scores) + `answer_preview` (200 chars), so it cannot reliably support answer faithfulness auditing.
- `scripts/evaluation/run_sop_filter_eval.py` hit logic likely has false negatives for punctuation (e.g., `&`) due to weak normalization.
- Answer formatting varies because answer prompts are minimal and do not specify a strict template.
- `scripts/evaluation/run_sop_filter_eval.py` now uses stricter doc normalization (`lower -> strip ext -> non-alnum to _ -> collapse _`) to avoid punctuation/whitespace false negatives, including `&`, hyphens, and extra spaces.
- Hit evaluation now returns richer per-row metrics (`hit_rank`, `hit_at_{1,3,5,10}`, `match_debug`) and guards blank normalized `gold_doc` to prevent accidental matches.
- Page parsing for hit checks is now defensive (`_safe_parse_page`), so invalid/non-integer `page` values no longer raise exceptions during evaluation.
KP|- SOP filter eval rows now include `schema_version` (`sop_filter_eval_v1`) and emit two artifacts per label: backward-compatible thin `{label}_results.jsonl` plus audit-rich `{label}_results.rich.jsonl` with request payload, response metadata, top-doc fields, full answer, configurable answer preview, hit metrics, elapsed, and error.
- Created `scripts/evaluation/validate_sop_eval_jsonl.py` as a reusable validator for both thin and rich SOP eval JSONL artifacts. It validates:
  - `schema_version` present and non-empty (forward-compatible: accepts any non-empty string)
  - Required keys: `idx`, `question`, `filter`, `gold_doc`, `hit_doc`, `hit_page`, `elapsed_ms`
  - Rich-specific: `top_docs` (list), `answer` (string), `request_payload` exists
  - Answer heuristics: `references_ok` (looks for 참고문헌/References when top_docs present), `citations_ok` (checks for `[n]` pattern), `language_ok` (checks Hangul for ko or ASCII for en)
  - Returns exit code 0 on pass, 1 on failure; prints JSON summary to stdout
- Added `--validate` flag to `run_sop_filter_eval.py` (default false). When enabled, runs validator on both thin and rich output files and writes `report.json` to out_dir with pass/fail counts and failure details.
- Task 01 hardening note: `_normalize_doc_name` now follows strict normalization (`lower -> strip -> strip extension -> non-alnum to _ -> collapse _ -> trim _`), which removes punctuation-driven false negatives such as `&` and mixed separators.
- Task 01 hardening note: `_check_hit` now returns safe defaults for blank normalized `gold_doc` (no doc/page hit), parses candidate `page` defensively (`_safe_parse_page`), and emits rank/debug metadata (`hit_rank`, `hit_at_{1,3,5,10}`, `match_debug`) for downstream diagnostics.
- Correction: `scripts/evaluation/run_sop_filter_eval.py` currently emits `schema_version: sop_eval_v1` in both thin/rich rows; rich rows include `request_payload.requests` (full posted payload sequence), merged `response_metadata` (raw API `metadata` + fallback keys like `route/search_queries/detected_language/target_language/template_version` when present), normalized `top_docs` (`rank/doc_id/title/source/page/score/doc_type/device_name/chunk_id`), full `answer`, configurable `answer_preview`, and hit/debug/timing/error fields.

- `format_ok` in validator summary is computed as: for rich rows, `format_ok = references_ok AND citations_ok AND language_ok`. This aggregates the three format checks into a single boolean metric.