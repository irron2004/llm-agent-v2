# Task: Paper D BibTeX priority and comparison docs

Status: Done
Owner: OpenCode
Branch or worktree: `main` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-04-14

## Goal

Paper D에서 survey한 논문들을 기반으로,
1) BibTeX 우선 수집 목록,
2) 논문 비교표
를 나중에 다시 참고하기 쉬운 문서로 만든다.

## Why

현재는 surveyed references 문서가 있지만,
실제로 논문 작성 단계에서 바로 필요한 것은
"어떤 논문부터 BibTeX를 확보해야 하는가"와
"논문들 간 차이가 무엇인가"를 빠르게 보는 문서다.

## Contracts To Preserve

- None (docs-only task; protected API/UI contracts are not touched)

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260414-paper-d-bibtex-and-comparison-docs.md`
- `docs/papers/50_paper_d_sensor_doc/evidence/paper_d_bibtex_priority.md`
- `docs/papers/50_paper_d_sensor_doc/evidence/paper_d_paper_comparison_table.md`
- `docs/papers/50_paper_d_sensor_doc/README.md`

## Out Of Scope

- No backend/frontend code changes
- No contract changes
- No automatic BibTeX generation in this task

## Risks

- 비교표가 과도하게 자세해져 reference 문서의 사용성이 떨어질 수 있음
- 아직 재검증이 필요한 논문이 확정 citation처럼 보일 위험
- 기존 surveyed reference 문서와 중복 서술이 과해질 수 있음

## Verification Plan

```bash
python3 - <<'PY'
from pathlib import Path

bib = Path('docs/papers/50_paper_d_sensor_doc/evidence/paper_d_bibtex_priority.md')
cmp = Path('docs/papers/50_paper_d_sensor_doc/evidence/paper_d_paper_comparison_table.md')
readme = Path('docs/papers/50_paper_d_sensor_doc/README.md')

bib_text = bib.read_text(encoding='utf-8')
cmp_text = cmp.read_text(encoding='utf-8')
readme_text = readme.read_text(encoding='utf-8')

for item in [
    '# Paper D — BibTeX Priority List',
    '## 1. 이 문서의 목적',
    '## 2. 최우선 확보 목록',
    '## 3. 2순위 확보 목록',
    '## 4. 재검증 후 확보 목록',
]:
    assert item in bib_text, f'missing in bib doc: {item}'

for item in [
    '# Paper D — Paper Comparison Table',
    '## 1. 이 문서의 목적',
    '## 2. 핵심 비교표',
    '## 3. 이 비교표로 바로 볼 수 있는 것',
]:
    assert item in cmp_text, f'missing in comparison doc: {item}'

assert 'paper_d_bibtex_priority.md' in readme_text, 'README missing BibTeX priority doc link'
assert 'paper_d_paper_comparison_table.md' in readme_text, 'README missing comparison table doc link'

print('paper D bibtex/comparison docs verified')
PY
```

## Verification Results

- command: `python3 - <<'PY' ... print('paper D bibtex/comparison docs verified') ... PY`
  - result: pass
  - note: 두 문서의 필수 섹션과 README 링크 반영 여부 확인

## Handoff

- Current status: done
- Last passing verification command and result:
  - `python3 - <<'PY' ... print('paper D bibtex/comparison docs verified') ... PY` (pass)
- Remaining TODOs (priority order):
  1. 필요 시 실제 BibTeX 수집 상태를 표에 업데이트
  2. 실제 논문 집필 단계에서 comparison table을 venue별 related work 표로 축소
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-04-14: task created
- 2026-04-14: `paper_d_bibtex_priority.md` 생성
- 2026-04-14: `paper_d_paper_comparison_table.md` 생성
- 2026-04-14: README에 두 문서 링크 추가
- 2026-04-14: verification command 실행 및 통과

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
