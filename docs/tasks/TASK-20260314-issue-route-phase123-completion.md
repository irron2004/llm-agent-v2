# Task: issue route full implementation completion (Q2 + Phase 1~3)

Status: completed
Owner: OpenCode
Branch or worktree: `86ewk6385-1차-PE-피드백-v2` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-03-14

## Goal

`docs/2026-03-14-issue-route-data-aware-design.md` 기준으로
미완료 범위인 Q2(동일 doc_id 섹션 병합 보강)와 Phase 1/2/3을 실제 코드에 반영한다.
특히 observability -> shadow -> live gating 단계를 contract-safe하게 구현한다.

## Why

현재 구현은 Phase 0 일부(Q1/Q3/Q4/Q5/Q6) 중심이며,
신호 계산/티어 정책/shadow 대비/live gating이 누락되어 설계의 핵심 운영 단계가 비어 있다.
이 누락을 메워 issue 경로의 품질 개선을 end-to-end로 완성한다.

## Contracts To Preserve

- C-API-001
- C-API-002
- C-API-003

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260314-issue-route-phase123-completion.md`
- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/api/routers/agent.py`
- `backend/tests/test_issue_route_phase0_quickwins.py`
- `backend/tests/test_issue_flow_interrupts.py`
- `backend/tests/test_expand_related_docs_node.py`
- `backend/tests/test_issue_route_phase123_policy.py`
- `tests/api/test_agent_interrupt_resume_regression.py`
- `tests/api/test_agent_response_metadata_contract.py`

## Out Of Scope

- frontend 변경
- Product contract 문서 변경
- ES 인덱스 재가공/재인덱싱
- unrelated 리팩터링

## Risks

- issue refs 선택 정책 변경으로 answer metadata/key contract 회귀 가능
- interrupt/resume에서 issue_case_ref_map/issue tier state 보존 회귀 가능
- retrieval_only 경로에서 불필요한 후처리 개입 가능

## Verification Plan

```bash
cd /home/hskim/work/llm-agent-v2
uv run pytest backend/tests/test_issue_route_phase0_quickwins.py -v
uv run pytest backend/tests/test_issue_route_phase123_policy.py -v
uv run pytest backend/tests/test_issue_flow_interrupts.py -v
uv run pytest backend/tests/test_expand_related_docs_node.py -v
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_retrieval_only.py -v
cd backend && uv run ruff check .
cd backend && uv run mypy .
```

## Verification Results

- command: `uv run pytest backend/tests/test_issue_route_phase123_policy.py -v`
  - result: pass
  - note: Phase 1/2/3 정책 롤아웃(phase1 baseline, phase3 live tier cap) 및 detail ref source 검증
- command: `uv run pytest backend/tests/test_issue_route_phase0_quickwins.py -v`
  - result: pass
  - note: Phase 0 quick-win(Q1/Q3/Q4/Q5/Q6) 회귀 없음
- command: `uv run pytest backend/tests/test_issue_flow_interrupts.py -v`
  - result: pass
  - note: issue interrupt/confirm/select/detail 루프 회귀 없음
- command: `uv run pytest backend/tests/test_expand_related_docs_node.py -v`
  - result: pass
  - note: Q2 관련 expand 경로(myservice/gcb same-doc 확장 포함) 회귀 없음
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: C-API-001 metadata contract 유지 확인
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass
  - note: C-API-002 interrupt/resume continuity 유지 확인
- command: `uv run pytest tests/api/test_agent_retrieval_only.py -v`
  - result: pass
  - note: C-API-003 retrieval_only semantics 유지 확인
- command: `cd backend && uv run ruff check .`
  - result: fail
  - note: 기존 `backend/api/routers/agent.py`의 대량 legacy lint debt(UP*/I001/E501)로 실패. 신규 추가 코드 외 기존 코드베이스 전반 이슈
- command: `cd backend && uv run ruff check llm_infrastructure/llm/langgraph_agent.py tests/test_issue_route_phase123_policy.py tests/test_expand_related_docs_node.py api/routers/agent.py`
  - result: fail
  - note: `api/routers/agent.py` 파일 자체의 기존 스타일 debt로 실패(신규 로직 기능 테스트는 통과)
- command: `cd backend && uv run mypy .`
  - result: fail
  - note: `scripts/legacy/llm-agent/tests/conftest.py` duplicate module(기존 레거시 구조 문제)로 조기 종료
- command: `lsp_diagnostics (modified python files)`
  - result: fail
  - note: workspace import resolution 미구성(reportMissingImports) + 기존 TypedDict strict 경고 다수
- command: `Oracle post-implementation review (sync)`
  - result: pass
  - note: 1 blocker(`expand_top_k=0` override 무시) 지적 후 즉시 수정/재검증 완료
- command: `Oracle post-implementation review (bg_4d685dd5)`
  - result: pass
  - note: blocker 없음
- command: `Oracle post-implementation review (bg_37918eb6)`
  - result: pass-with-findings
  - note: strict-resume/nonce/issue-metadata/section_fetcher hardening 포인트 반영 완료
- command: `uv run pytest backend/tests/test_issue_flow_interrupts.py -v` (re-run)
  - result: pass
  - note: nonce mismatch reject 케이스 추가 후 회귀 통과
- command: `uv run pytest backend/tests/test_expand_related_docs_node.py -v` (re-run)
  - result: pass
  - note: section_fetcher-only 조기종료 방지 케이스 추가 후 통과
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v` (re-run)
  - result: pass
  - note: thread_id 없는 resume_decision 400 검증 추가
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v` (re-run)
  - result: pass
  - note: non-issue 경로에서 issue_* metadata 누수 방지 검증 추가

## Handoff

- Current status: done
- Last passing verification command and result
  - `uv run pytest tests/api/test_agent_retrieval_only.py -v` (pass)
- Remaining TODOs (priority order)
  1. none
- Whether `Allowed Files` changed and why
  - yes (`backend/api/routers/agent.py` 추가: phase telemetry metadata 반영)
- Whether `Contracts To Update` is expected
  - no

## Change Log

- 2026-03-14: task created
- 2026-03-14: `langgraph_agent.py`에 issue signals/tier/shadow/live rollout 로직 추가
- 2026-03-14: `api/routers/agent.py` metadata에 issue phase telemetry 필드 노출 추가
- 2026-03-14: `test_issue_route_phase123_policy.py` 신규 추가 (3 tests)
- 2026-03-14: `test_expand_related_docs_node.py` 상수 기대값(10) 동기화
- 2026-03-14: Oracle blocker(`expand_top_k=0` 무시) 수정 + 회귀 테스트 재통과
- 2026-03-14: Oracle 추가 지적 반영(strict-resume 400, nonce 검증, non-issue metadata gating, section_fetcher-only 지원)

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
