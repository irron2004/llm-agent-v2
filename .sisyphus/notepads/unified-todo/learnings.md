# unified-todo learnings

- 2026-03-05 UI-1: In auto-parse branches, pass `guided_confirm` via `chat_state`/`chat_state_stream` state_overrides and reset `auto_parse_confirmed=False` only for non-resume guided-confirm requests so legacy auto-parse behavior stays unchanged when flag is absent/false.
- 2026-03-05 UI-2: Resume requests must stay on the same graph/checkpointer family; route only dict resumes with `{"type":"auto_parse_confirm"}` to `_new_guided_confirm_agent` and keep all other resume payloads on `_new_hil_agent` to preserve legacy retrieval/device/human review behavior.
- 2026-03-05 UI-3: `auto_parse_confirm_node` should read recommendations primarily from `parsed_query` and only fallback to `auto_parsed_*` fields, then persist confirmed device/equip selections back to both top-level state and `parsed_query.*` to avoid retrieval-branch divergence.
- 2026-03-05 UI-4: Enforce task-mode doc-type scope in `auto_parse_confirm_node` as authoritative overwrite (`sop`->expanded SOP strict, `issue`->expanded `myservice/gcb/ts` strict, `all`->empty non-strict) and mirror to both top-level state and `parsed_query` keys for retrieval consistency.
- 2026-03-05 UI-5: `answer_node` should derive a dedicated `answer_language` from `target_language` first, then `detected_language`, and use that same value consistently for both answer template routing (`*_en/*_zh/*_ja` fallback chain) and prompt query selection (`query_en` only for English answers).
- 2026-03-05 UI-5: Response metadata language reporting is safest when split into two fields: normalized final `target_language` used for answer rendering and `selected_language_source` (`user` vs `auto_parse` vs `default`) inferred defensively from raw `result` values.

- 2026-03-05 UI-5: Evidence captured at `.sisyphus/evidence/ui-guided-selection/task-5-target-language.txt` (pytest output).

- 2026-03-05 UI-6: Treat `auto_parse_confirm` as its own pending state (`pendingGuidedSelection`) so existing `pendingInterrupt` resume/HIL logic remains unchanged until UI-8.

- 2026-03-05 UI-7: `GuidedSelectionPanel` parses `payload.steps/options/defaults` defensively from `interrupt_payload` and emits a single `{type:"auto_parse_confirm", ...}` decision on completion.

- 2026-03-05 UI-8: Guided confirm resume uses `resume_decision.type="auto_parse_confirm"` with `thread_id` captured from the interrupt, and does not set legacy HIL `ask_user_after_retrieve=true`.
- 2026-03-05 UI-9: API contract tests for guided confirm can stay hermetic by overriding `get_default_llm`/`get_search_service` and monkeypatching `agent_router` agent constructors + shared `_checkpointer`, then asserting resume doc-type scopes (`sop` vs `myservice/gcb/ts`), metadata `target_language`, and missing-checkpoint 400 behavior end-to-end via `/api/agent/run`.
- 2026-03-05 UI-10: `GuidedSelectionPanel` manual equip flow only advances after selecting `__manual__`, typing into the `equip_id` input, and clicking `확인`, which should then emit the typed ID in final `onComplete` decision.
- 2026-03-05 UI-10: Hook payload tests should assert first send carries `guided_confirm: true` and guided resume send carries both interrupt `thread_id` and `resume_decision.type="auto_parse_confirm"`.
