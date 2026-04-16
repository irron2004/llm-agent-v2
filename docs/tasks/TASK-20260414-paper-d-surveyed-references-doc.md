# Task: Paper D surveyed references document

Status: Done
Owner: OpenCode
Branch or worktree: `main` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-04-14

## Goal

기존에 survey한 Paper D 관련 논문들을 나중에 빠르게 다시 참고할 수 있도록,
한 파일에 정리된 reference-style 문서를 만든다.

## Why

현재 Paper D에는 literature survey 문서가 여러 개 있으나,
빠르게 "무슨 논문이 있었고, 왜 중요하고, 언제 다시 봐야 하는지"를 확인하는
실무형 reference note가 없다.

## Contracts To Preserve

- None (docs-only task; protected API/UI contracts are not touched)

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260414-paper-d-surveyed-references-doc.md`
- `docs/papers/50_paper_d_sensor_doc/evidence/paper_d_surveyed_references.md`
- `docs/papers/50_paper_d_sensor_doc/README.md`

## Out Of Scope

- No backend/frontend code changes
- No product contract changes
- No removal or merge of existing survey files in this task

## Risks

- 기존 survey들과 내용 중복 가능성
- 아직 재검증이 필요한 논문이 강한 인용 후보처럼 보일 위험
- 빠른 참고용 문서가 너무 길어져 사용성이 떨어질 위험

## Verification Plan

```bash
python3 - <<'PY'
from pathlib import Path

doc = Path('docs/papers/50_paper_d_sensor_doc/evidence/paper_d_surveyed_references.md')
readme = Path('docs/papers/50_paper_d_sensor_doc/README.md')

doc_text = doc.read_text(encoding='utf-8')
readme_text = readme.read_text(encoding='utf-8')

required_doc_sections = [
    '# Paper D — Surveyed References',
    '## 1. 이 문서의 목적',
    '## 2. Quick Lookup',
    '## 3. 주제별 정리',
    '## 4. 우선순위',
    '## 5. 재검증 필요 후보',
    '## 6. Paper D에서의 사용 위치',
]
for item in required_doc_sections:
    assert item in doc_text, f'missing section: {item}'

assert 'paper_d_surveyed_references.md' in readme_text, 'README missing surveyed references doc link'
print('paper D surveyed references doc verified')
PY
```

## Verification Results

- command: `python3 - <<'PY' ... print('paper D surveyed references doc verified') ... PY`
  - result: pass
  - note: surveyed references 문서 필수 섹션과 README 링크 반영 여부 확인

## Handoff

- Current status: done
- Last passing verification command and result:
  - `python3 - <<'PY' ... print('paper D surveyed references doc verified') ... PY` (pass)
- Remaining TODOs (priority order):
  1. 필요 시 기존 `2026-04-14_literature_survey.md` / `2026-04-14_related_literature_survey.md`와의 중복 정리
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-04-14: task created
- 2026-04-14: `paper_d_surveyed_references.md` 생성
- 2026-04-14: README에 reference 문서 링크 추가
- 2026-04-14: verification command 실행 및 통과

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
