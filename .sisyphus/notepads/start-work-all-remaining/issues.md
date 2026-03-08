# Issues

## 2026-03-07
- (init)

## 2026-03-08
- Legacy plan reference `tests/api/test_agent_rrf_and_sticky_gates.py` is stale (file does not exist); use current tests: `tests/api/test_agent_stage2_retrieval.py`, `tests/api/test_agent_sticky_policy_followup_only.py`, `tests/api/test_agent_interrupt_resume_regression.py`.
- Some legacy plans use backticked items that are not filesystem paths (endpoints like /api/agent/run, commands, path:line). Treat them as non-path references in reconciliation.
