# 2026-03-11 Claim-vs-Code Audit (`.sisyphus` / `.omc`)

## 목적
- `.sisyphus`/`.omc`에서 "완료"로 보이는 항목이 실제 코드에 반영되었는지 점검.
- 여러 에이전트 실행 중 `git restore`/범위 이탈로 코드가 되돌아간 흔적을 식별.

## 결론 요약
1. **RRF stage1 관련 완료 주장**은 실제 코드/커밋에 반영되어 있음.
2. **Issue-flow/UI 관련 일부 완료 주장**은 코드가 완전 연결되지 않았거나, 로컬 staged 상태로만 존재.
3. `.omc`는 작업 완료 개수만 기록하고 있어 파일 단위 진위 검증 근거로는 부족.

## A. 실제 반영 확인 (문제 없음)
- `backend/llm_infrastructure/retrieval/engines/es_search.py`:
  - 최근 커밋 `6cdd725`에서 app-level RRF 전환 반영.
- `scripts/evaluation/run_chat_flow_retrieval_rrf_eval.py`:
  - 최근 커밋 `e49a912`에서 runner 반영.
- `backend/tests/test_es_parse_hits_null_score.py`, `backend/tests/test_es_stage1_app_rrf.py`, `backend/tests/test_chat_flow_rrf_eval_runner.py`:
  - 모두 커밋 반영 상태.

## B. "완료 주장" 대비 반영 불일치 (핵심)

### B1) UI 이슈 패널 파일은 존재하지만 실제 UI 플로우에 미연결
- 존재(로컬 staged):
  - `frontend/src/features/chat/components/issue-case-selection-panel.tsx`
  - `frontend/src/features/chat/components/issue-sop-confirm-panel.tsx`
  - `frontend/src/features/chat/components/issue-confirm-panel.tsx`
- 그러나 현재 연결 누락:
  - `frontend/src/features/chat/components/index.ts`에 해당 패널 export 없음.
  - `frontend/src/features/chat/pages/chat-page.tsx`에서 해당 패널 import/render 없음.
  - `frontend/src/features/chat/hooks/use-chat-session.ts`는 여전히
    `const guidedSteps: Array<"language" | "device" | "equip_id" | "task">` 기반 구 플로우.

### B2) "완료"로 보이는 파일 다수가 커밋 전 로컬 staged 상태
- 예: `backend/llm_infrastructure/llm/prompts/issue*_v{1,2}.yaml`,
  `frontend/src/features/chat/__tests__/issue-*.test.tsx`,
  `tests/api/test_agent_issue_flow_interrupt_resume.py` 등.
- 즉, **작업 흔적은 있으나 아직 리포지토리 기준 완결(커밋/푸시) 상태는 아님**.

## C. 되돌림(rollback) 흔적 및 메타-코드 불일치
- `.sisyphus/notepads/rrf-stage1-app-level/issues.md`에
  - 대규모 scope creep 후 `git restore`로 범위 외 파일 복원한 기록 존재.
- `.sisyphus/notepads/*`에 "stale reference"/"scope noise" 경고 다수 존재.

## D. `.omc` 점검 결과
- `backend/.omc/state/subagent-tracking.json`:
  - agent 완료 횟수/시간 정보만 있고, 파일 단위 "무엇을 반영했는지" 근거는 없음.
- `backend/.omc/state/checkpoints/*.json`:
  - todo summary만 있고 코드 반영 여부 판정 데이터는 없음.

## 정리(실행) 항목
1. Issue-flow UI 관련은 "파일 존재"와 "런타임 연결"을 분리해서 재검증 필요.
2. staged-only 상태인 완료 주장 항목은 커밋/푸시 전까지 "반영 완료"로 간주하지 않음.
3. `.sisyphus` 계획 문서의 완료 체크는 본 감사 문서 링크와 함께 재검토 기준으로 사용.
