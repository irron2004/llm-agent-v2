# Task: Paper D map and graph framing document

Status: Done
Owner: OpenCode
Branch or worktree: `main` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-04-14

## Goal

Paper D의 핵심을 더 쉽게 설명하는 개념 문서를 만들고,
사용자가 제안한 graph-based anomaly framing이 Paper D에서 어디에 놓여야 하는지 정리한다.

## Why

현재 Paper D 문서들은 알고리즘 구조와 연구 프레이밍은 잘 담고 있지만,
"지도를 만드는 일"과 "그 지도로 retrieval/diagnosis를 하는 일"의 관계를
쉽게 설명하는 문서는 없다. 또한 graph 기반 이상 정의 아이디어를
핵심 기여로 볼지, eventizer 강화로 볼지 개념 경계가 필요하다.

## Contracts To Preserve

- None (docs-only task; protected API/UI contracts are not touched)

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260414-paper-d-map-and-graph-framing-doc.md`
- `docs/papers/50_paper_d_sensor_doc/paper_d_map_and_graph_framing.md`
- `docs/papers/50_paper_d_sensor_doc/README.md`

## Out Of Scope

- No backend/frontend code changes
- No product contract changes
- No changes to Paper D core research proposal beyond README link updates

## Risks

- graph-based anomaly 아이디어를 Paper D의 핵심 기여처럼 과장할 위험
- 기존 algorithm/architecture 문서와 개념 중복 가능성
- retrieval 중심 프레이밍이 graph modeling 논의로 희석될 수 있음

## Verification Plan

```bash
python3 - <<'PY'
from pathlib import Path

doc = Path('docs/papers/50_paper_d_sensor_doc/paper_d_map_and_graph_framing.md')
readme = Path('docs/papers/50_paper_d_sensor_doc/README.md')

doc_text = doc.read_text(encoding='utf-8')
readme_text = readme.read_text(encoding='utf-8')

required_doc_sections = [
    '# Paper D — 지도 만들기와 graph-based anomaly framing',
    '## 1. Paper D를 쉬운 말로 다시 정리',
    '## 2. "지도 만들기"와 "길 찾기"의 차이',
    '## 3. 사용자가 말한 내용의 위치',
    '## 4. graph 기반 이상 정의 아이디어에 대한 판단',
    '## 5. 가장 안전한 Paper D framing',
]
for item in required_doc_sections:
    assert item in doc_text, f'missing section: {item}'

assert 'paper_d_map_and_graph_framing.md' in readme_text, 'README missing map/graph framing doc reference'
print('paper D map and graph framing doc verified')
PY
```

## Verification Results

- command: `python3 - <<'PY' ... print('paper D map and graph framing doc verified') ... PY`
  - result: pass
  - note: 새 개념 문서 필수 섹션과 README 링크 반영 여부 확인

## Handoff

- Current status: done
- Last passing verification command and result:
  - `python3 - <<'PY' ... print('paper D map and graph framing doc verified') ... PY` (pass)
- Remaining TODOs (priority order):
  1. 필요 시 `paper_d_algorithm_design.md`에 본 문서의 핵심 문단 요약 반영
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-04-14: task created
- 2026-04-14: `docs/papers/50_paper_d_sensor_doc/paper_d_map_and_graph_framing.md` 생성
- 2026-04-14: README에 새 문서 링크 추가
- 2026-04-14: verification command 실행 및 통과

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
