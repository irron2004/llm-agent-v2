# Learnings

- REQ-1: __skip__ option should be forced `recommended=false` regardless of parsed_devices/equip_ids presence; test asserted via interrupt payload options.
- Note: basedpyright/LSP shows many pre-existing missing-import diagnostics for backend; rely on pytest for verification in this repo.
- REQ-4: task_options label lives in `backend/llm_infrastructure/llm/langgraph_agent.py` (task step options); `sop` label updated to `절차조회`.

- Guided confirm payload must explicitly override `__skip__` option flags: force `recommended=False` for both `device` and `equip_id` option lists so upstream/default recommendation logic cannot mark skip as recommended.
- REQ-5 mapping finalized in guided confirm resume: `sop -> expand(["sop","ts","setup"])`, `issue -> expand(["gcb","myservice"])`, `all -> []` (no filter).
- Strict doc_type filtering now preserves full selected_doc_types order without `_dedupe_queries` 5-item cap, so SOP mode keeps TS/setup scope active.
- REQ-3 guard pattern: before `Command(resume=...)`, validate checkpoint has pending interrupt (`state.values` exists and `state.next` non-empty); return 409 detail prefix `RESUME_NO_PENDING_INTERRUPT` when resume is stale.
- Frontend safe-render rule: guided/device selection panels should require live pending-interrupt shape (`threadId` + expected payload `type`) instead of message-order heuristics.
- REQ-2 parsing hardening: shared helpers (`_strip_code_fences`, `_extract_json_substring`, `_parse_json_object_or_none`) reduce per-node parser drift and keep deterministic safe defaults on malformed model outputs.
- REQ-2 judge guardrail: enable `response_format={"type":"json_object"}` only when LLM adapter advertises support; still keep robust fallback parsing so non-compliant outputs resolve to stable `parse_error` defaults.
- REQ-2 output normalization: translation/query-rewrite should pass through a single-line sanitizer that strips labels/numbering/analysis text before use as executable search query text.
- REQ-4: Verified sop label is already `절차조회` in langgraph_agent.py line 3258 (no change needed).
