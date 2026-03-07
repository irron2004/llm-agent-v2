# Draft: UI Chat Improvements v2

## Requirements (confirmed)
- docs/2026-03-06-Agent_개선.md 기반 6개 개선 항목 구현
  - (1) `건너뛰기` 추천 표시 제거: 어떤 조건에서도 recommended=false
  - (2) 모델 포맷 불이행 대응: 파싱 fallback 통일 + judge JSON 강제/재시도 상한 + translate/query_rewrite 정규화 + 프롬프트 강화
  - (3) 답변 완료 후 장비선택 버튼 오류 수정: pending interrupt 있을 때만 UI 노출 + pending 없는 resume은 서버에서 명확한 4xx
  - (4) UI 라벨: "SOP 조회" → "절차조회" (value=sop 유지)
  - (5) 업무 유형별 문서유형 단일 선택 강제: 절차조회(SOP|TS|SETUP), 이슈조회(GCB|MyService|PEMS) 중 1개 필수 + ES 필터 strict
  - (6) 탭 기반 단계 UI: 단계 수/현재 위치 표시, 이전 단계 수정 가능, 마지막 단계에서만 확인/진행 활성화, 전체조회는 문서종류 생략

## Technical Decisions
- 계획 산출물은 .sisyphus/plans/에 작성 (Prometheus 제약)
- 기존 .omc/plans/ui-chat-improvements-v2.md는 참고/차이점(diff)만 기록
- 테스트: frontend는 Vitest (`frontend/package.json` script `test`), backend는 pytest

## Research Findings
- frontend test runner: `frontend/package.json`에 `vitest run`
- frontend vitest config: `frontend/vitest.config.ts` (jsdom, setupFiles)
- backend test runner: pytest (pyproject)

## Open Questions
- 없음 (요구사항 문서에 해결 방향이 명시됨)

## Scope Boundaries
- INCLUDE: 위 6개 항목 + 해당 변경에 필요한 최소 타입/스키마/테스트/검증
- EXCLUDE: 문서에 없는 UI 전면 개편, 검색 품질 개선(랭킹/쿼리 로직 변경) 등 추가 기능
