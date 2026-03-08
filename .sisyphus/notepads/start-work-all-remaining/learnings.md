# Learnings

## 2026-03-07
- (init)

## 2026-03-08
- For chunk_v3/Paper A reconciliation, prefer `data/chunk_v3_normalize_table.md` and `data/paper_a/eval/query_gold_master_v0_5.jsonl` as current on-disk sources.
- Guided selection UI REQ-6 touches `frontend/src/features/chat/components/guided-selection-panel.tsx`, `frontend/src/features/chat/hooks/use-chat-session.ts`, and `frontend/src/features/chat/pages/chat-page.tsx`; preserve numeric-input flow and `resume_decision` payload contract.
- Strict doc-type override must be honored in `auto_parse_node` even without `needs_history`; unit test: `backend/tests/test_sop_intent_heuristic.py`.
