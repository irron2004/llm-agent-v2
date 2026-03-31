# Task: Paper A execution phase0 implementation

Status: In Progress
Owner: OpenCode
Branch or worktree: `86ewk6385-1차-PE-피드백-v2` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-03-14

## Goal

`docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md`의 P0를 실제 산출물 기준으로 최대한 닫는다.
현재 범위는 T1(parser/oracle gap), T2(gold audit), T3(mixed scope restoration),
T4(shared-policy paradox decomposition)와 그에 연결된 T5/T6 실행 증빙까지 포함한다.

## Why

Paper A의 최신 masked BM25 주장은 oracle upper bound, gold 품질, explicit-only eval,
shared-policy 역설이라는 네 개의 약한 고리에 걸려 있다.
따라서 P0의 핵심 산출물을 실제 파일로 만들고, 실행 문서가 요구하는 evidence chain을
재현 가능한 스크립트/리포트로 연결해야 short paper narrative를 방어할 수 있다.

## Contracts To Preserve

- C-API-001
- C-API-002
- C-API-003

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260314-paper-a-execution-phase0.md`
- `docs/papers/20_paper_a_scope/2026-03-14_execution_tasks.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_hybrid_rerank_recovery.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v06_gold_audit.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_mixed_eval_restoration.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_v07_implicit_eval.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-18_p7plus_algorithm_proposal.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-18_p7plus_experiment.md`
- `docs/papers/20_paper_a_scope/paper_a_draft_v2.md`
- `scripts/paper_a/measure_parser_accuracy.py`
- `scripts/paper_a/analyze_b45_failure_decomposition.py`
- `scripts/paper_a/generate_v06_gold_audit.py`
- `scripts/paper_a/build_v07_mixed_eval_set.py`
- `scripts/paper_a/run_masked_hybrid_experiment.py`
- `scripts/paper_a/run_masked_p6p7_experiment.py`
- `scripts/paper_a/run_p8_evidence_scope_experiment.py`
- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `data/paper_a/parser_accuracy_report.json`
- `data/paper_a/parser_accuracy_per_query_diff.csv`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_oracle_vs_parser_gap.md`
- `data/paper_a/masked_hybrid_results.json`
- `data/paper_a/masked_p6p7_results.json`
- `data/paper_a/p8_results.json`
- `data/paper_a/gold_verification_report.json`
- `data/paper_a/eval/query_gold_master_v0_7_mixed.jsonl`
- `data/paper_a/eval/query_gold_master_v0_7_mixed_split_report.json`
- `.sisyphus/evidence/paper-a/runs/2026-03-14_v07_implicit_eval/`
- `docs/papers/20_paper_a_scope/evidence/2026-03-18_p8_algorithm_spec.md`
- `docs/papers/20_paper_a_scope/evidence/2026-03-18_p8_implementation_issue_report.md`

## Out Of Scope

- frontend 변경
- API route/payload contract 변경
- unrelated issue-route 작업 파일 수정
- Paper A 전체 원고 리라이트(구조 전면 변경)
- API/Frontend feature work

## Risks

- parser 측정 코드가 production parser와 미세하게 달라서 잘못된 gap을 보고할 수 있음
- ES/index 환경 의존성 때문에 오프라인에서 retrieval comparison이 부분 실행될 수 있음
- 이미 존재하는 user-owned untracked 스크립트/문서를 덮어쓸 위험이 있음
- v0.5/v0.6 병합 시 split/field semantics가 어긋날 수 있음
- heuristic gold audit 결과를 LLM judge 수준으로 과대해석할 위험이 있음

## Verification Plan

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/measure_parser_accuracy.py
uv run python scripts/paper_a/generate_v06_gold_audit.py
uv run python scripts/paper_a/build_v07_mixed_eval_set.py
uv run python scripts/paper_a/run_masked_p6p7_experiment.py
uv run python - <<'PY'
import json
from pathlib import Path

p = Path('data/paper_a/parser_accuracy_report.json')
assert p.exists(), f'missing: {p}'
obj = json.loads(p.read_text(encoding='utf-8'))
for key in ('parser_accuracy', 'retrieval_comparison'):
    assert key in obj, f'missing key: {key}'
assert Path('data/paper_a/gold_verification_report.json').exists()
assert Path('data/paper_a/eval/query_gold_master_v0_7_mixed.jsonl').exists()
assert Path('data/paper_a/eval/query_gold_master_v0_7_mixed_split_report.json').exists()
arr = json.loads(Path('data/paper_a/masked_p6p7_results.json').read_text(encoding='utf-8'))
assert arr and 'P7plus_masked' in arr[0]['conditions']
print('parser report verified')
PY
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_retrieval_only.py -v
```

## Verification Results

- command: `uv run python scripts/paper_a/measure_parser_accuracy.py`
  - result: pass
  - note: T1 parser accuracy/oracle gap 리포트와 per-query CSV, evidence markdown 생성 확인; equip-aware realistic mode까지 반영
- command: `uv run python - <<'PY' ... parser report verified ... PY`
  - result: pass
  - note: `data/paper_a/parser_accuracy_report.json`에 `parser_accuracy`, `retrieval_comparison` 키 존재 확인
- command: `python - <<'PY' ... scope aware parser report verified ... PY`
  - result: pass
  - note: `retrieval_comparison_scope_aware` 존재 및 equip-aware realistic mode의 핵심 수치 검증
- command: `uv run python scripts/paper_a/generate_v06_gold_audit.py`
  - result: pass
  - note: T2 gold audit JSON/MD 산출물 생성 확인
- command: `python - <<'PY' ... gold audit verified ... PY`
  - result: pass
  - note: `data/paper_a/gold_verification_report.json`의 핵심 샘플/precision 수치 검증
- command: `uv run python scripts/paper_a/build_v07_mixed_eval_set.py`
  - result: pass
  - note: T3 mixed eval set, split report, evidence markdown 생성 확인
- command: `uv run python scripts/paper_a/validate_master_eval_jsonl.py --path data/paper_a/eval/query_gold_master_v0_7_mixed.jsonl --report-path data/paper_a/eval/query_gold_master_v0_7_mixed_validation_report.json`
  - result: pass
  - note: T3 mixed eval JSONL schema 검증 통과
- command: `uv run python scripts/paper_a/run_masked_hybrid_experiment.py`
  - result: pass
  - note: T5 masked hybrid 결과 `data/paper_a/masked_hybrid_results.json` 생성 확인
- command: `uv run python - <<'PY' ... masked hybrid result verified ... PY`
  - result: pass
  - note: masked hybrid 결과 기본 필드(`conditions`, `target_device`, `scope_observability`) 검증
- command: `uv run python scripts/paper_a/run_masked_p6p7_experiment.py`
  - result: pass
  - note: T6 재실행 + `P7plus_masked` 조건 추가 결과 `data/paper_a/masked_p6p7_results.json` 생성 확인
- command: `python - <<'PY' ... p7plus result artifact verified ... PY`
  - result: pass
  - note: `data/paper_a/masked_p6p7_results.json`에 `P7plus_masked` 조건 존재 확인
- command: `python - <<'PY' ... p7plus evidence doc verified ... PY`
  - result: pass
  - note: `docs/papers/20_paper_a_scope/evidence/2026-03-18_p7plus_experiment.md` 핵심 섹션 확인
- command: `python - <<'PY' ... p7+ proposal updated and verified ... PY`
  - result: pass
  - note: P7+ 제안서에 oracle upper bound / realistic 분리 문구 반영 확인
- command: `python - <<'PY' ... hybrid and p6p7 evidence docs verified ... PY`
  - result: pass
  - note: T5/T6 evidence 문서 `2026-03-14_hybrid_rerank_recovery.md`, `2026-03-14_masked_p6p7_reexperiment.md` 존재 및 핵심 섹션 확인
- command: `uv run python scripts/paper_a/analyze_b45_failure_decomposition.py`
  - result: pass
  - note: T4 보조 분석 문서 `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md` 생성 확인
- command: `python - <<'PY' ... b45 decomposition verified ... PY`
  - result: pass
  - note: B4.5 분해 문서의 핵심 섹션 존재 확인
- command: `python - <<'PY' ... all evidence docs verified ... PY`
  - result: pass
  - note: 생성한 Paper A evidence 문서 전체 비어 있지 않음 확인
- command: `uv run python scripts/paper_a/evaluate_paper_a_master.py ... --scope-observability implicit`
  - result: blocked
  - note: hybrid evaluator가 dense index 차원 불일치(1024 query vs 768 index)로 실패, blocker 문서 `2026-03-14_v07_implicit_eval.md`에 기록
- command: `lsp_diagnostics scripts/paper_a/measure_parser_accuracy.py`
  - result: pass
  - note: 변경된 T1 스크립트에 에러 진단 없음
- command: `lsp_diagnostics scripts/paper_a/analyze_b45_failure_decomposition.py`
  - result: pass
  - note: 새 T4 분석 스크립트에 에러 진단 없음
- command: `lsp_diagnostics scripts/paper_a/generate_v06_gold_audit.py`
  - result: pass
  - note: 새 T2 스크립트에 에러 진단 없음
- command: `lsp_diagnostics scripts/paper_a/build_v07_mixed_eval_set.py`
  - result: pass
  - note: 새 T3 스크립트에 에러 진단 없음
- command: `lsp_diagnostics scripts/paper_a/run_masked_p6p7_experiment.py`
  - result: pass
  - note: P7+를 반영한 T6 스크립트에 에러 진단 없음
- command: `uv run ruff check scripts/paper_a/run_masked_p6p7_experiment.py`
  - result: blocked
  - note: 현재 환경에서 `ruff` 실행 파일 미설치로 실행 불가(`No such file or directory`)
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: C-API-001 metadata contract 유지 확인
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass
  - note: C-API-002 interrupt/resume contract 유지 확인
- command: `uv run pytest tests/api/test_agent_retrieval_only.py -v`
  - result: pass
  - note: C-API-003 retrieval_only contract 유지 확인
- command: `lsp_diagnostics scripts/paper_a/run_p8_evidence_scope_experiment.py`
  - result: pass
  - note: P8 실험 스크립트 타입/진단 오류 없음 확인
- command: `uv run python -m scripts.paper_a.run_p8_evidence_scope_experiment`
  - result: pass
  - note: P8 full rerun 완료, `data/paper_a/p8_results.json` 재생성 및 baseline/P8 요약 출력 확인
- command: `uv run python - <<'PY' ... (source/scope bucket stratification) ... PY`
  - result: pass
  - note: P8 결과를 `hypothesis_source`/`scope_correct` 버킷으로 분해해 지표 단조성 점검
- command: `uv run python - <<'PY' ... (high-contamination spot checks) ... PY`
  - result: pass
  - note: `GENEVA XP` vs `GENEVA_XP` 정규화 불일치가 contamination 과대계산 원인임을 확인

## Handoff

- Current status: in-progress
- Last passing verification command and result:
  - `uv run pytest tests/api/test_agent_retrieval_only.py -v` (pass)
- Remaining TODOs (priority order):
  1. `P7+`를 cached-candidate simulation이 아닌 end-to-end retrieval 모드로 확장 검증
  2. hybrid/dense index 차원 불일치 해결 후 `v0.7` implicit slice 실험 재실행
  3. reconcile generated evidence with other dirty user-owned Paper A docs before any commit
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-03-14: task created
- 2026-03-14: T1 script 보강 - per-query CSV 및 evidence markdown 자동 생성 추가
- 2026-03-14: T1 realistic mode를 equip-aware parser 비교까지 확장
- 2026-03-14: T5 masked hybrid 실험 실행 및 결과 생성
- 2026-03-14: T6 masked P6/P7 재실험 실행 및 결과 생성
- 2026-03-14: T5/T6 evidence 문서(`hybrid_rerank_recovery`, `masked_p6p7_reexperiment`) 추가
- 2026-03-14: T4 보조 분석용 `analyze_b45_failure_decomposition.py` 추가 및 failure decomposition 문서 생성
- 2026-03-14: T2 gold audit 패키징 스크립트 및 evidence/JSON 생성
- 2026-03-14: T3 mixed eval restoration 스크립트 구현, split rebuild 및 validation 완료
- 2026-03-14: v0.7 implicit eval 실행 시도, hybrid index 차원 불일치 blocker 확인 및 evidence 문서화
- 2026-03-14: API protected contracts(C-API-001/002/003) 회귀 테스트 전부 통과
- 2026-03-18: `run_masked_p6p7_experiment.py`에 P7+ (confidence-gated hard/soft blending + shared cap) 구현
- 2026-03-18: P7+ 결과 문서 `2026-03-18_p7plus_experiment.md` 생성
- 2026-03-18: P7+ 제안서에 oracle upper bound vs realistic 결과 분리 문구 반영
- 2026-03-18: `run_p8_evidence_scope_experiment.py` 로직 점검/보정 작업 범위를 Allowed Files에 반영
- 2026-03-18: P8 shared-cap merge 보정(비공유 우선 후 shared 주입 재정렬), Stage1 cached-first/fallback-probe 적용
- 2026-03-18: P8 device 정규화(`GENEVA XP` vs `GENEVA_XP`) 일관화로 scope/contamination 계산 오류 수정

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
