# Task: issue route data-aware design document

Status: completed
Owner: OpenCode
Branch or worktree: `86ewk6385-1차-PE-피드백-v2` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-03-14
Updated: 2026-03-14

## Goal

docs 리포트와 live Elasticsearch 상태를 함께 반영해,
`task_mode=issue` 경로의 데이터-기반 개선 설계안을 실행 가능한 문서로 남긴다.
이번 라운드에서는 문서를 구현/검증 관점에서 더 구체화해 후속 코드 작업자가 바로 실행할 수 있게 만든다.

## Why

기존 이슈 경로는 동작은 안정화되었지만,
실제 인덱스 분포(특히 doc_type 편중)와 문서 구조 품질 차이를 정책에 반영하지 못한다.
다음 구현 단계가 즉시 착수할 수 있도록 신호 정의, 정책 tier, 롤아웃/검증 기준을 고정해야 한다.

## Contracts To Preserve

- C-API-001
- C-API-002

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260314-issue-route-data-aware-design-doc.md`
- `docs/2026-03-14-issue-route-data-aware-design.md`

## Out Of Scope

- `backend/`, `frontend/`, `tests/` 코드 변경
- product contract 내용 변경
- ES 인덱스 재적재/재청킹 실행

## Risks

- 문서 리포트와 live ES 상태가 상충할 수 있어 설계 기준이 흔들릴 수 있음
- 신호 threshold를 과도하게 잡으면 recall 저하(하드 필터와 유사한 실패) 위험
- 문서 설계가 구현 훅 위치와 어긋나면 후속 작업자가 재탐색 비용을 부담

## Verification Plan

```bash
uv run python - <<'PY'
from pathlib import Path

paths = [
    "docs/tasks/TASK-20260314-issue-route-data-aware-design-doc.md",
    "docs/2026-03-14-issue-route-data-aware-design.md",
]

for p in paths:
    path = Path(p)
    assert path.exists(), f"missing: {p}"
    text = path.read_text(encoding="utf-8")
    assert len(text.strip()) > 0, f"empty: {p}"

print("doc files exist and are non-empty")
PY

git diff -- docs/tasks/TASK-20260314-issue-route-data-aware-design-doc.md docs/2026-03-14-issue-route-data-aware-design.md
git status --short docs/tasks/TASK-20260314-issue-route-data-aware-design-doc.md docs/2026-03-14-issue-route-data-aware-design.md
uv run python - <<'PY'
from pathlib import Path

doc = Path("docs/2026-03-14-issue-route-data-aware-design.md").read_text(encoding="utf-8")
required = [
    "Implementation checklist",
    "Telemetry schema",
    "Go/No-Go gates",
    "Risk register",
]
missing = [x for x in required if x not in doc]
assert not missing, f"missing markers: {missing}"
print("design doc key markers verified")
PY
```

## Verification Results

- command: `uv run python - <<'PY' ... PY`
  - result: pass
  - note: 설계 문서/태스크 문서 파일 존재 및 non-empty 확인
- command: `git diff -- docs/tasks/TASK-20260314-issue-route-data-aware-design-doc.md docs/2026-03-14-issue-route-data-aware-design.md`
  - result: pass
  - note: 신규 파일(untracked) 상태라 출력이 비어있는 것이 정상
- command: `git status --short docs/tasks/TASK-20260314-issue-route-data-aware-design-doc.md docs/2026-03-14-issue-route-data-aware-design.md`
  - result: pass
  - note: allowed files 2개만 신규 추가(`??`) 상태 확인
- command: `lsp_diagnostics docs/*.md (2 files)`
  - result: skip
  - note: 현재 런타임은 `.md`용 LSP 서버 미구성이라 diagnostics 수행 불가 (환경 제한)
- command: `uv run python - <<'PY' ... (marker verification) ... PY`
  - result: pass
  - note: 최신 설계 보강 항목(`Implementation checklist`, `Telemetry schema`, `Go/No-Go gates`, `Risk register`, `C-API-003`, `K_signal`) 존재 확인
- command: `git diff -- docs/tasks/TASK-20260314-issue-route-data-aware-design-doc.md docs/2026-03-14-issue-route-data-aware-design.md && git status --short ...`
  - result: pass
  - note: allowed files 2개만 수정/추적 대상으로 유지

## Handoff

- Current status: done
- Last passing verification command and result:
  - `uv run python - <<'PY' ... marker verification ... PY` (pass)
- Remaining TODOs (priority order):
  1. none
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-03-14: task created
- 2026-03-14: docs 리포트 + live ES 스냅샷을 반영한 issue route data-aware 설계 문서 작성 완료
- 2026-03-14: 설계 문서를 implementation-ready 수준으로 고도화하는 후속 라운드 시작
- 2026-03-14: Oracle 리뷰 반영 (K_signal, metadata key contract, C-API-003, Phase 0 baseline, checkpoint-safe state 규칙) 완료

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
