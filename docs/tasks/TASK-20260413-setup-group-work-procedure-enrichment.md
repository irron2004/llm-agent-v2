# Task: setup group work procedure enrichment

Status: Done
Owner: Hephaestus
Branch or worktree: main
Created: 2026-04-13

## Goal

Implement setup/SOP procedural group enrichment so that when a setup answer group is selected,
the group can be augmented with same-`doc_id` `Work Procedure` refs before relevance checking and
answer generation. This should improve procedure recall without widening to unrelated same-device SOPs.

## Why

The current setup flow groups refs by `(doc_id, section)` and can evaluate/select a non-procedural
section such as `사고 사례` without the matching `Work Procedure` content from the same document.
Even when the document is correct, the selected group may lack the actual replacement steps the user
needs.

## Contracts To Preserve

- C-API-001
- C-API-003

## Contracts To Update

- None

## Allowed Files

- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/tests/test_setup_group_work_procedure_enrichment.py`
- `docs/tasks/TASK-20260413-setup-group-work-procedure-enrichment.md`

## Out Of Scope

- No frontend changes
- No changes to interrupt/resume semantics beyond preserving current behavior
- No broad same-device ranking override policy
- No unrelated retrieval refactors

## Risks

- Inflating setup groups too much and degrading answer quality
- Injecting duplicate refs or wrong-section refs into setup groups
- Breaking current setup group selection / retry flow
- Accidentally changing retrieval_only behavior or metadata payloads
- LangGraph state propagation can silently drop newly returned fields unless they are declared in `AgentState`

## Verification Plan

```bash
cd backend && uv run pytest tests/test_setup_group_work_procedure_enrichment.py -v
cd backend && uv run pytest tests/test_setup_answer_quality_controls.py -v
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_retrieval_only.py -v
```

## Verification Results

- command: `cd backend && uv run pytest tests/test_setup_group_work_procedure_enrichment.py -v`
  - result: pass
  - note: verifies same-doc Work Procedure ref-map building, per-doc group enrichment order, and answer-node enrichment before setup relevance checking
- command: `cd backend && uv run pytest tests/test_setup_answer_quality_controls.py -v`
  - result: pass
  - note: verifies existing setup answer quality controls still hold after enrichment wiring
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: verifies API metadata contract remains intact
- command: `uv run pytest tests/api/test_agent_retrieval_only.py -v`
  - result: pass
  - note: verifies retrieval_only behavior still stops before answer generation
- command: `cd backend && uv run ruff check llm_infrastructure/llm/langgraph_agent.py tests/test_setup_group_work_procedure_enrichment.py --select E9,F`
  - result: pass
  - note: fatal lint/syntax checks for modified backend files
- command: `uv run python -m py_compile backend/llm_infrastructure/llm/langgraph_agent.py backend/tests/test_setup_group_work_procedure_enrichment.py`
  - result: pass
  - note: byte-compilation confirms modified Python files parse successfully
- command: `runtime schema review`
  - result: pass
  - note: verified and fixed missing `AgentState` declaration for `setup_work_procedure_ref_map`; without this field LangGraph could drop the map between `retrieve_node` and `answer_node`

## Handoff

- Current status: done
- Last passing verification command and result: `cd backend && uv run ruff check llm_infrastructure/llm/langgraph_agent.py tests/test_setup_group_work_procedure_enrichment.py --select E9,F` → pass
- Remaining TODOs (priority order): run scenario-level manual verification for real SOP/setup procedural queries if desired
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-04-13: task created
- 2026-04-13: implemented selected-group same-doc Work Procedure enrichment and added focused backend tests
- 2026-04-13: fixed missing `AgentState` declaration for `setup_work_procedure_ref_map`, which blocked runtime propagation from retrieve_node to answer_node

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [ ] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
