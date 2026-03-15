# Task: gcb-only 필터 전환 시 myservice 검색 누수 수정

Status: completed
Owner: OpenCode
Branch or worktree: `86ewk6385-1차-PE-피드백-v2` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-03-14

## Goal

같은 chat thread에서 기존 issue 범위(`myservice+gcb+ts`)로 대화한 뒤
의도를 `gcb` 중심으로 좁혔을 때 `myservice`가 계속 검색되는 누수 경로를 제거한다.
필터 축소 의도를 우선 반영하도록 상태 병합/파싱 로직을 보정한다.

## Why

현재 상태 유지(sticky) 로직이 strict doc_type 선택을 과도하게 고정하면,
후속 턴에서 `gcb` 의도가 명시되어도 이전 넓은 범위가 유지되어 retrieval 품질이 저하된다.

## Contracts To Preserve

- C-API-001
- C-API-002
- C-API-003

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260314-gcb-only-filter-leak-fix.md`
- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/tests/test_auto_parse_node_events.py`
- `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`

## Out Of Scope

- frontend UI/UX 변경
- unrelated issue-route policy 리팩터링
- Product contract 문서 내용 변경

## Risks

- sticky follow-up 정책이 지나치게 약화되어 기존 기대가 깨질 수 있음
- interrupt/resume 경로에서 선택 범위 반영 방식 회귀 가능
- metadata contract 간접 회귀 가능

## Verification Plan

```bash
cd /home/hskim/work/llm-agent-v2
uv run pytest backend/tests/test_auto_parse_node_events.py -v
uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_retrieval_only.py -v
```

## Verification Results

- command: `uv run pytest backend/tests/test_auto_parse_node_events.py -v`
  - result: pass
  - note: strict sticky 상태에서 query에 `gcb`가 명시되면 선택 범위를 `gcb`로 축소하는 회귀 케이스 포함 전체 통과
- command: `uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v`
  - result: pass
  - note: thread 유지 상태에서 issue broad -> follow-up gcb-only 축소 시 retrieval doc_types가 gcb 그룹만 사용됨 검증
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: C-API-001 유지 확인
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass
  - note: C-API-002 유지 확인
- command: `uv run pytest tests/api/test_agent_retrieval_only.py -v`
  - result: pass
  - note: C-API-003 유지 확인
- command: `python -m py_compile backend/llm_infrastructure/llm/langgraph_agent.py backend/tests/test_auto_parse_node_events.py tests/api/test_agent_autoparse_confirm_interrupt_resume.py`
  - result: pass
  - note: 수정 파일 문법 확인
- command: `Oracle final review (ses_312560fecffeDXH58vB6mnQXFr)`
  - result: pass-with-fixes
  - note: `ParsedQuery.doc_types_strict` 전파 누락 및 테스트-동작 불일치 반영 완료

## Handoff

- Current status: done
- Last passing verification command and result
  - `uv run pytest tests/api/test_agent_retrieval_only.py -v` (pass)
- Remaining TODOs (priority order)
  1. none
- Whether `Allowed Files` changed and why
  - no
- Whether `Contracts To Update` is expected
  - no

## Change Log

- 2026-03-14: task created
- 2026-03-14: `auto_parse_node` strict sticky 규칙 보정(새 doc_type 감지 시 축소 허용)
- 2026-03-14: `ParsedQuery.doc_types_strict` 전파 추가
- 2026-03-14: `backend/tests/test_auto_parse_node_events.py`에 strict 축소 회귀 테스트 추가 및 현행 equip_id 비활성 정책과 테스트 정합화
- 2026-03-14: `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`에 thread 내 issue->gcb 축소 회귀 테스트 추가

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
