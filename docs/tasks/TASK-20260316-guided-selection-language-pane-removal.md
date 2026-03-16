# Task: guided selection 언어/설비 pane 제거 및 안내 문구 축소

Status: completed
Owner: OpenCode
Branch or worktree: `86ewk6385-1차-PE-피드백-v2` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-03-16

## Goal

Doc 미선택 상태에서 노출되는 guided selection UI를 단순화한다.
사용자에게는 `기기(model) / 작업`만 보이게 하고, 기존 `"언어"`, `"설비"` 선택 pane과
`"아래 4단계를 숫자 입력 또는 클릭으로 선택하세요..."` 안내 문구는 제거한다.

추가로, issue confirm(`추가 문서를 검색하겠습니까?` 계열)에서 답변 선택 후
이전 대화가 중복 표시되는 UI 회귀를 재현/원인분석하고 수정한다.

## Why

기존 guided selection은 `언어 -> 기기 -> 설비 ID -> 작업`에서 `기기 -> 설비 -> 작업`까지
축소되었지만, Doc 미선택 기본 플로우에서는 설비 선택도 불필요한 경우가 많다.
이 때문에 Doc 미선택 첫 진입에서는 `기기(model) -> 작업`만 남겨 UX를 더 가볍게 만든다.

## Current Findings

- backend `auto_parse_confirm` payload가 guided `steps/options/defaults`를 직접 생성한다.
  - `backend/llm_infrastructure/llm/langgraph_agent.py`
- frontend panel/hook은 payload 기반 step 렌더링이 가능하며, legacy step도 수용한다.
  - `frontend/src/features/chat/components/guided-selection-panel.tsx`
- `defaults.equip_id`가 남아 있으면 equip step이 없는 플로우에서도 숨은 값이 전달될 수 있다.
  - `frontend/src/features/chat/hooks/use-chat-session.ts`
- resume decision 계약에는 여전히 `target_language`가 포함된다.
  - `backend/api/routers/agent.py`
- issue confirm 재개 흐름에서 동일 answer가 연속 interrupt 응답으로 재전달되면
  이전 assistant content가 다시 렌더링되어 중복 대화처럼 보일 수 있다.
  - `frontend/src/features/chat/hooks/use-chat-session.ts`
- issue confirm 버튼 연타(또는 매우 빠른 연속 호출) 시 동일 nonce resume 요청이
  중복 전송될 수 있다.
  - `frontend/src/features/chat/hooks/use-chat-session.ts`
- issue flow 루프에서 backend가 동일 `issue_sop_confirm` nonce를 재사용하면,
  frontend dedupe set이 기존 key를 소비된 상태로 유지해 SOP confirm 버튼 클릭이 무시될 수 있다.
  - `frontend/src/features/chat/hooks/use-chat-session.ts`

## Contracts To Preserve

- C-API-001
- C-API-002
- C-UI-001
- C-UI-002

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260316-guided-selection-language-pane-removal.md`
- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/tests/test_agent_guided_resume_validation.py`
- `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`
- `tests/api/test_agent_interrupt_resume_regression.py`
- `tests/api/test_agent_response_metadata_contract.py`
- `frontend/src/features/chat/components/guided-selection-panel.tsx`
- `frontend/src/features/chat/hooks/use-chat-session.ts`
- `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`
- `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx`
- `frontend/src/features/chat/__tests__/chat-page-device-panel.test.tsx`
- `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx`

## Proposed Change

1. Backend payload 단순화
- `auto_parse_confirm` payload의 `steps` 기본값을 `["device", "task"]`로 축소한다.
- `instruction`은 제거하거나 빈 값으로 내려보낸다.
- `options.equip_id`는 새 payload에서 제외하고, `defaults.equip_id`는 `None`으로 고정한다.
- `defaults.target_language`는 유지한다.

2. Frontend 표시 단순화
- guided selection panel에서 `언어`, `설비` 탭/요약을 기본 플로우에서 제거한다.
- instruction이 비어 있으면 안내 문구 영역을 렌더링하지 않는다.
- 선택 요약은 기본 플로우에서 `기기 / 작업`만 보여준다.

3. Internal language 유지
- `target_language`는 사용자 선택값이 아니라 `detected_language` 또는 기존 기본값을 사용한다.
- resume decision에는 기존처럼 `target_language`를 계속 포함시켜 API 계약 파손을 피한다.

4. Legacy payload 호환
- frontend는 `steps`에 `language` 또는 `equip_id`가 포함된 예전 payload도 계속 렌더링 가능하게 둔다.
- 새 backend는 더 이상 `language`, `equip_id` step을 보내지 않는다.

5. issue confirm 중복 표시 회귀 방지
- interrupt(issue_confirm/issue_case_selection/issue_sop_confirm)에서
  새 응답 answer가 직전 assistant content와 동일하면 prompt/instruction을 우선 표시해
  동일 대화의 재렌더링을 피한다.
- issue confirm/case/sop resume 전송 시 nonce 기반 dedupe key를 적용해
  동일 decision 중복 전송을 차단한다.

## Acceptance Criteria

- Doc 미선택 상태의 guided selection 화면에 `"언어"` pane이 보이지 않는다.
- Doc 미선택 상태의 guided selection 화면에 `"설비"` pane이 보이지 않는다.
- `"아래 4단계를 숫자 입력 또는 클릭으로 선택하세요..."` 문구가 보이지 않는다.
- 숫자 입력과 클릭 흐름이 `기기 -> 작업` 2단계에서 정상 완료된다.
- resume 요청은 기존처럼 `type=auto_parse_confirm`과 `target_language`를 포함한다.
- interrupt/resume의 `thread_id` continuity는 유지된다.
- 기존 missing-device panel 동작은 영향을 받지 않는다.

## Out Of Scope

- 다국어 답변 기능 제거
- `target_language` 필드 삭제 또는 resume schema 축소
- task 선택 UI를 guided selection 밖으로 이동하는 구조 변경
- issue flow, retrieval_only, missing-device flow 리팩터링

## Risks

- backend와 frontend의 step 정의가 어긋나면 숫자 입력 흐름이 틀어질 수 있음
- instruction 제거 시 panel spacing 또는 empty state 렌더링이 깨질 수 있음
- `target_language`를 UI에서 숨기면서도 내부 default가 누락되면 답변 언어가 흔들릴 수 있음
- 배포 중 버전 차이로 legacy `language`/`equip_id` step payload가 들어올 수 있음

## Verification Plan

```bash
cd /home/hskim/work/llm-agent-v2
uv run pytest backend/tests/test_agent_guided_resume_validation.py -v
uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
cd frontend && npm run test -- src/features/chat/__tests__/guided-selection-panel.test.tsx
cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx
cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx
cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx
```

## Verification Results

- command: `uv run pytest backend/tests/test_agent_guided_resume_validation.py -v`
  - result: pass (`2 passed`)
- command: `uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v`
  - result: pass (`8 passed`)
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass (`4 passed`)
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass (`2 passed`)
- command: `python -m py_compile backend/llm_infrastructure/llm/langgraph_agent.py`
  - result: pass
- command: `cd frontend && npm run test -- src/features/chat/__tests__/guided-selection-panel.test.tsx`
  - result: pass (`3 passed`)
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass (`13 passed`)
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass (`12 passed`)
- command: `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx`
  - result: pass (`3 passed`)
- command: `cd frontend && npm run build`
  - result: pass (production bundle built)
  - note: Vite chunk size warning persists as pre-existing optimization warning; not a blocker for this task.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx`
  - result: pass (`3 passed`)
  - note: re-run after dedupe reset/clear logic update in `use-chat-session.ts`.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass (`13 passed`)
  - note: re-run after final issue-flow regression patch.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass (`12 passed`)
  - note: re-run confirms protected device-panel behavior; React act warnings are pre-existing in this suite.
- command: `cd frontend && npm run build`
  - result: pass (production bundle built)
  - note: re-run after latest frontend patch; Vite chunk size warning remains non-blocking and pre-existing.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx`
  - result: pass (`4 passed`)
  - note: added regression for `"이슈 확인: 아니오"` path where final response could replay previous answer.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass (`13 passed`)
  - note: re-run after final no-branch dedupe patch.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass (`12 passed`)
  - note: re-run confirms C-UI-001/C-UI-002 guard behavior; React act warnings are pre-existing.
- command: `cd frontend && npm run build`
  - result: pass (production bundle built)
  - note: re-run after no-branch dedupe patch; Vite chunk size warning remains non-blocking and pre-existing.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx`
  - result: pass (`4 passed`)
  - note: re-run after narrowing duplicate matcher to latest assistant message.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass (`13 passed`)
  - note: re-run after latest matcher refinement.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass (`12 passed`)
  - note: re-run confirms C-UI-001/C-UI-002 behavior unchanged; React act warnings are pre-existing.
- command: `cd frontend && npm run build`
  - result: pass (production bundle built)
  - note: re-run after final matcher refinement; Vite chunk size warning remains non-blocking and pre-existing.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx`
  - result: pass (`5 passed`)
  - note: added regression for repeated SOP-confirm loop with reused SOP nonce.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass (`13 passed`)
  - note: re-run after SOP confirm dedupe lifecycle fix.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass (`12 passed`)
  - note: re-run confirms protected device-panel behavior unchanged; React act warnings are pre-existing.
- command: `cd frontend && npm run build`
  - result: pass (production bundle built)
  - note: re-run after SOP confirm click fix; Vite chunk size warning remains non-blocking and pre-existing.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx`
  - result: pass (`5 passed`)
  - note: re-run after case-selection empty-input guard order refinement.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass (`13 passed`)
  - note: re-run after final issue-flow dedupe adjustments.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass (`12 passed`)
  - note: re-run confirms C-UI contracts remain intact; React act warnings are pre-existing.
- command: `cd frontend && npm run build`
  - result: pass (production bundle built)
  - note: final re-run after all issue-flow patches; Vite chunk size warning remains non-blocking and pre-existing.

## Handoff

- Current status: completed
- Last passing verification command and result
  - `cd frontend && npm run build` -> pass
- Remaining TODOs (priority order)
  1. none
- Whether `Allowed Files` changed and why
  - no
- Whether `Contracts To Update` is expected
  - no

## Change Log

- 2026-03-16: task created
- 2026-03-16: backend auto_parse_confirm payload simplified to 3-step guided flow and frontend panel/hook/tests aligned.
- 2026-03-16: verification plan executed; task marked completed.
- 2026-03-16: follow-up requirement applied — doc-unselected guided flow simplified to 2-step (`device -> task`) by removing equip pane from default payload/UI while keeping legacy payload compatibility.
- 2026-03-16: verification set re-run after 2-step change (backend/API/frontend/build all pass).
- 2026-03-16: issue confirm 답변 선택 후 이전 대화 중복 표시 회귀 수정 (interrupt content dedupe + nonce 기반 중복 resume 방지).
- 2026-03-16: issue-flow regression tests 추가 및 통과.
- 2026-03-16: latest frontend verification set re-run after dedupe reset/clear hardening patch (all targeted tests/build pass).
- 2026-03-16: `"이슈 확인: 아니오"` 후 final 응답이 이전 답변을 재전달할 때 중복 표시되는 경로를 추가로 보완(최종 응답 dedupe + 회귀 테스트 추가).
- 2026-03-16: duplicate matcher를 latest assistant 기준으로 좁혀 과도한 중복판정 위험을 완화(Oracle review 반영).
- 2026-03-16: 반복 issue 루프에서 재사용 nonce로 SOP confirm 버튼 클릭이 무시되는 회귀 수정(guard 순서 보정 + 신규 interrupt 수신 시 nonce key 해제 + 회귀 테스트 추가).
- 2026-03-16: issue_case_selection 빈 입력에서 dedupe key가 선소비되지 않도록 guard 순서를 추가 보정.

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
