# Task: issue route phase0 quick-win implementation

Status: completed
Owner: OpenCode
Branch or worktree: `86ewk6385-1차-PE-피드백-v2` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-03-14

## Goal

`docs/2026-03-14-issue-route-data-aware-design.md`의 Phase 0(Q1~Q6) 중
핵심 동작을 실제 코드에 반영한다.
특히 issue MQ 분리, issue refs 분리, ts-only 경로 보정, issue detail refs 보존을 구현한다.

## Why

현재 `task_mode=issue` 흐름은 `general_mq`와 `MAX_ANSWER_REFS=5` 경로에 묶여
증거 다양성과 상세 선택 품질이 저하될 수 있다.
설계안의 quick-win을 먼저 반영해 이후 signal/tier 단계의 기반을 안정화한다.

## Contracts To Preserve

- C-API-001
- C-API-002
- C-API-003

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260314-issue-route-phase0-implementation.md`
- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/llm_infrastructure/llm/prompts/issue_mq_v2.yaml`
- `backend/llm_infrastructure/llm/prompts/issue_ans_v2.yaml`
- `backend/llm_infrastructure/llm/prompts/issue_ans_en_v2.yaml`
- `backend/llm_infrastructure/llm/prompts/issue_ans_zh_v2.yaml`
- `backend/llm_infrastructure/llm/prompts/issue_ans_ja_v2.yaml`
- `backend/tests/test_issue_flow_interrupts.py`
- `backend/tests/test_issue_route_phase0_quickwins.py`

## Out Of Scope

- frontend 변경
- Product contract 문서 변경
- signal/tier live gating 전체 구현
- ES index 재가공/재인덱싱

## Risks

- issue mode 변경이 metadata contract를 깨뜨릴 수 있음
- interrupt/resume 상태에서 issue detail refs가 유실될 수 있음
- ts-only 분기 추가 시 기존 issue/sop 분기 회귀 가능성

## Verification Plan

```bash
cd /home/hskim/work/llm-agent-v2
uv run pytest backend/tests/test_issue_flow_interrupts.py -v
uv run pytest backend/tests/test_issue_route_phase0_quickwins.py -v
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_retrieval_only.py -v
```

## Verification Results

- command: `uv run pytest backend/tests/test_issue_route_phase0_quickwins.py -v`
  - result: pass
  - note: Phase 0 quick-win 신규 동작(이슈 MQ, refs 분리, ts-only route, detail refs map) 검증
- command: `uv run pytest backend/tests/test_issue_flow_interrupts.py -v`
  - result: pass
  - note: 기존 issue interrupt/confirm/select/detail 루프 회귀 없음
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: C-API-001 metadata contract 유지 확인
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass
  - note: C-API-002 interrupt/resume continuity 유지 확인
- command: `uv run pytest tests/api/test_agent_retrieval_only.py -v`
  - result: pass
  - note: C-API-003 retrieval_only semantics 유지 확인
- command: `lsp_diagnostics (python files)`
  - result: skip
  - note: 런타임 LSP가 workspace import 해석을 하지 못해 기존 전역 missing-import 진단 다수 발생 (검증은 pytest 기준으로 수행)
- command: `Oracle final review`
  - result: pass
  - note: critical blocker 없음, 구현 진행 가능 판정
- command: `uv run pytest backend/tests/test_issue_route_phase0_quickwins.py -v`
  - result: pass
  - note: follow-up 이후 issue phase0 핵심 회귀 재검증
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: C-API-001 metadata contract 유지 재확인
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass
  - note: C-API-002 interrupt/resume continuity 유지 재확인
- command: `uv run pytest tests/api/test_agent_retrieval_only.py -v`
  - result: pass
  - note: C-API-003 retrieval_only semantics 유지 재확인
- command: `lsp_diagnostics (issue_ans_en/zh/ja yaml, task md)`
  - result: skip
  - note: `yaml-language-server` 미설치, `.md` LSP 미구성으로 진단 불가

## Handoff

- Current status: done
- Last passing verification command and result:
  - `uv run pytest tests/api/test_agent_retrieval_only.py -v` (pass)
- Remaining TODOs (priority order):
  1. none
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-03-14: task created
- 2026-03-14: Phase 0 quick-win 구현 완료 (`issue_mq`, ts-only route 보정, issue refs 분리/맵 저장, issue prompt 보강)
- 2026-03-14: contract 회귀 테스트(C-API-001/002/003) 전부 통과
- 2026-03-14: follow-up으로 `issue_ans_en/zh/ja_v2.yaml`에 doc_type/section 해석 및 gcb 처리 지시 동등 반영

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
