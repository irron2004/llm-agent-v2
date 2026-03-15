# Task: Paper A document consolidation

Status: Done
Owner: OpenCode
Branch or worktree: `86ewk6385-1차-PE-피드백-v2` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-03-14

## Goal

`docs/papers/20_paper_a_scope/` 아래의 Paper A 관련 문서를 한 개의 정리 문서로 통합한다.
중복 내용을 복붙하지 않고, timeline 중심으로 문서의 진화, 핵심 전환점, 현재 신뢰 가능한 근거,
남은 blocker를 한 번에 이해할 수 있게 재구성한다.

## Why

Paper A 문서군은 spec, draft, evidence, review, task 문서로 퍼져 있어 현재 상태와 narrative pivot을 한 번에 읽기 어렵다.
특히 오래된 주장과 최신 evidence가 충돌하는 지점이 있어, canonical reading order를 대신할 단일 안내 문서가 필요하다.

## Contracts To Preserve

- C-API-001
- C-API-002
- C-API-003

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260314-paper-a-doc-consolidation.md`
- `docs/papers/20_paper_a_scope/2026-03-14_paper_a_timeline_consolidated.md`

## Out Of Scope

- No Paper A experiment/code changes
- No manuscript claim rewrites in existing source docs
- No backend/frontend edits

## Risks

- Timeline 문서가 오래된 주장과 최신 근거를 혼동할 수 있음
- source-of-truth와 stale docs를 잘못 분류할 수 있음
- 너무 장황해져서 오히려 탐색성이 나빠질 수 있음

## Verification Plan

```bash
python - <<'PY'
from pathlib import Path

p = Path('docs/papers/20_paper_a_scope/2026-03-14_paper_a_timeline_consolidated.md')
text = p.read_text(encoding='utf-8')
required = [
    '# Paper A Consolidated Timeline',
    '## Reading Guide',
    '## Timeline',
    '## Current State',
    '## Recommended Canonical Reading Order',
]
for item in required:
    assert item in text, f'missing section: {item}'
print('paper A consolidation doc verified')
PY
```

## Verification Results

- command: `python - <<'PY' ... paper A consolidation doc verified ... PY`
  - result: pass
  - note: consolidated timeline doc required sections 존재 확인
- command: `python - <<'PY' ... paper A consolidation doc finalized ... PY`
  - result: pass
  - note: Oracle 피드백 반영 후 numeric drift / wording guardrail 포함 최종 구조 확인

## Handoff

- Current status: done
- Last passing verification command and result:
  - `python - <<'PY' ... paper A consolidation doc finalized ... PY` (pass)
- Remaining TODOs (priority order):
  1. none
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-03-14: task created
- 2026-03-14: consolidated timeline document created at `docs/papers/20_paper_a_scope/2026-03-14_paper_a_timeline_consolidated.md`
- 2026-03-14: Oracle review reflected; wording softened for canonical/stale language and numeric drift section added

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
