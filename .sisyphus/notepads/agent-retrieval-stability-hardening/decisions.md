# Decisions


## F4 Scope Decision (2026-02-28 17:44 UTC)
- Accepted as in-scope: backend/frontend/test/docs changes implementing retrieval stability hardening contract (D1-D5, plus documented contract section in diagnosis doc).
- Rejected from delivery scope: process-state file edits in `.sisyphus/boulder.json` and unrelated checklist edits in `.sisyphus/plans/paper-b-stability.md`; recommend excluding or reverting these before merge.

## DoD Test File Reference Verification (2026-03-01)
- Verified: DoD command at line 63 already references correct test file `pytest backend/tests/test_agent_graph_mq_bypass.py -q`
- No incorrect reference `test_agent_mq_bypass_and_fallback.py` found in plan
- Test file exists at `backend/tests/test_agent_graph_mq_bypass.py`

