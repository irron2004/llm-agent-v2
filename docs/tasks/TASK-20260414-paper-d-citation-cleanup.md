# Task: Paper D citation cleanup after verification

Status: Done
Owner: OpenCode
Branch or worktree: `main` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-04-14

## Goal

사용자가 제공한 실재 검증 결과를 바탕으로,
Paper D 문헌 문서에서 미검증/오류 후보를 정정하고
안전한 대체 문헌으로 치환한다.

## Why

S2S-FDD, RAIDR 등 일부 논문은 기존 문서에 포함되어 있었으나
실재 여부가 불분명하거나 직접 비교 대상으로 쓰기 위험했다.
논문 방향은 유지하되 citation risk를 낮추는 정리가 필요했다.

## Contracts To Preserve

- None (docs-only task; protected API/UI contracts are not touched)

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260414-paper-d-citation-cleanup.md`
- `docs/papers/50_paper_d_sensor_doc/evidence/paper_d_surveyed_references.md`
- `docs/papers/50_paper_d_sensor_doc/evidence/paper_d_bibtex_priority.md`
- `docs/papers/50_paper_d_sensor_doc/evidence/paper_d_paper_comparison_table.md`
- `docs/papers/50_paper_d_sensor_doc/evidence/2026-04-14_related_literature_survey.md`

## Out Of Scope

- No backend/frontend code changes
- No contract changes

## Verification Plan

```bash
python3 - <<'PY'
from pathlib import Path

paths = [
    Path('docs/papers/50_paper_d_sensor_doc/evidence/paper_d_surveyed_references.md'),
    Path('docs/papers/50_paper_d_sensor_doc/evidence/paper_d_bibtex_priority.md'),
    Path('docs/papers/50_paper_d_sensor_doc/evidence/paper_d_paper_comparison_table.md'),
    Path('docs/papers/50_paper_d_sensor_doc/evidence/2026-04-14_related_literature_survey.md'),
]

for p in paths:
    text = p.read_text(encoding='utf-8')
    assert 'S2S-FDD' not in text, f'S2S-FDD still present in {p}'
    assert 'RAIDR' not in text, f'RAIDR still present in {p}'

sr = paths[0].read_text(encoding='utf-8')
assert 'FD-LLM' in sr, 'FD-LLM not added to surveyed references'
assert 'KEO' in sr, 'KEO not added to surveyed references'

print('paper D citation cleanup verified')
PY
```

## Verification Results

- command: `python3 - <<'PY' ... print('paper D citation cleanup verified') ... PY`
  - result: pass
  - note: 미검증 후보 제거 및 대체 후보 반영 여부 확인

## Handoff

- Current status: done
- Last passing verification command and result:
  - `python3 - <<'PY' ... print('paper D citation cleanup verified') ... PY` (pass)
- Remaining TODOs (priority order):
  1. 실제 BibTeX 확보 시 FD-LLM, KEO 메타데이터 우선 확인
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-04-14: task created
- 2026-04-14: S2S-FDD → FD-LLM 대체
- 2026-04-14: RAIDR → KEO 대체
- 2026-04-14: THGNN / HTGNN 계열을 재검증 후 사용 후보로 하향 조정

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
