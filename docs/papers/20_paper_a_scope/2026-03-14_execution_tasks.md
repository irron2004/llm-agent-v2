# Paper A 실행 우선순위 작업 문서 (2026-03-14)

## 0) 목적

Paper A를 "결과는 강하지만 과대주장 없이 방어 가능한 상태"로 만드는 실행 계획.

## 0.1) Evidence provenance snapshot (2026-03-14 sync)

- baseline evidence: `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
- baseline metrics (masked BM25, loose gold):
  - `B0_masked`: `287/578 (50%)`
  - `B4_masked`: `530/578 (92%)`
  - delta: `+42%p`
- safety note: 위 수치는 **oracle device filter 상한선**이며,
  실제 운영 성능은 parser 품질과 함께 별도로 보고해야 함.

현재 기준 핵심 관찰:

- masked BM25에서 `B4`는 contamination `0.000`, gold hit(loose) `530/578 (92%)`
- 같은 설정에서 `B0_masked`는 gold hit `287/578 (50%)`
- 즉, oracle device filter 기준 개선폭은 `+42%p`

근거 파일:

- `docs/papers/20_paper_a_scope/evidence/2026-03-13_paper_a_progress_summary.md`
- `data/paper_a/trap_masked_results.json`
- `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl`

---

## 1) 우선순위 개요

### P0 (논문 블로커, 이번 라운드 필수)

1. Oracle 가정 제거/완화: parser 기반 성능 갭 측정
2. v0.6 gold 신뢰도 검증(표본 감사 + strict/loose 정합성)
3. implicit/ambiguous 평가축 복원
4. shared 정책 역설(B4.5 < B4) 원인 분해

### P1 (강한 실험 완성도)

5. Hybrid/Rerank 조건 실험 복구(BM25-only 탈출)
6. P6/P7 soft scoring 재실험(새 eval protocol 기준)
7. gcb equip 결측 보정 규칙 실험

### P2 (원고 완성)

8. Method/Experiment/Limitations 수치 동기화
9. Figure/Table 최종화 + 재현성 패키지

---

## 2) 상세 작업

## T1 [P0] Oracle vs Parser 갭 측정

- 목표:
  - 현재 `B4 masked`는 oracle filter 가정임.
  - 동일 셋에서 `oracle device` vs `parser/auto-parse device` 성능 차이를 정량화.
- 입력:
  - `query_gold_master_v0_6_generated_full.jsonl`
  - `trap_masked_results.json` (oracle 기준치)
- 출력:
  - `evidence/2026-03-14_oracle_vs_parser_gap.md`
  - per-query diff CSV (`oracle_hit`, `parser_hit`, `parser_device_correct`)
- 완료 조건:
  - 최소 지표 4개 보고: `Hit@k`, `Cont@k`, `parser device accuracy`, `oracle-parser delta`

### 실행 커맨드 (즉시 실행 가능)

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/measure_parser_accuracy.py
```

### 검증 커맨드

```bash
cd /home/hskim/work/llm-agent-v2
uv run python - <<'PY'
import json
from pathlib import Path

p = Path("data/paper_a/parser_accuracy_report.json")
assert p.exists(), f"missing: {p}"
obj = json.loads(p.read_text(encoding="utf-8"))
required = ["parser_accuracy", "retrieval_comparison"]
missing = [k for k in required if k not in obj]
assert not missing, f"missing keys: {missing}"
print("parser report verified")
PY
```

## T2 [P0] v0.6 Gold 신뢰도 감사

- 목표:
  - 자동 생성 gold의 신뢰도 리스크를 정량화.
- 방법:
  - 표본 감사(권장 150~200 query): explicit_device/equip 균형 샘플
  - `gold_doc_ids_strict`와 loose의 precision/coverage 확인
- 출력:
  - `evidence/2026-03-14_v06_gold_audit.md`
  - 감사 결과 JSONL (query, doc, judge label, rationale)
- 완료 조건:
  - strict/loose 각각의 precision 추정치 + 95% CI

## T3 [P0] implicit/ambiguous 평가축 복원

- 목표:
  - 현재 v0.6의 explicit-only 편향 해소.
- 방법:
  - 기존 v0.5 implicit/ambiguous 재활용 + v0.6 포맷 병합
  - 최소 목표: implicit 100+, ambiguous 80+
- 출력:
  - `query_gold_master_v0_7_mixed.jsonl`
  - split report (`dev/test`, leak key, empty-gold 통계)
- 완료 조건:
  - `scope_observability` 4축(explicit_device, explicit_equip, implicit, ambiguous) 모두 포함

## T4 [P0] B4.5 역설 원인 분해

- 목표:
  - 왜 `device+shared(B4.5)`가 `device-only(B4)`보다 hit가 낮은지 규명.
- 방법:
  - B4 hit / B4.5 miss 케이스 집중 분석
  - shared 우선순위, rerank 점수, gold 타입 분해
- 출력:
  - `evidence/2026-03-14_b45_failure_decomposition.md`
  - 카테고리별 손실 비율(예: shared overload, wrong shared, ranking dilution)
- 완료 조건:
  - "정책 수정안 1개 이상" + 예상 개선폭 제시

## T5 [P1] Hybrid/Rerank 실험 복구

- 목표:
  - BM25-only 한계 해소.
- 현재 이슈:
  - 인덱스 간 필드/embedding 불일치.
- 실행안:
  - 단기: `rag_chunks_dev_v2` 단일 인덱스에서 B0~B3 재현
  - 중기: 통합 인덱스 또는 cross-index 조인 파이프라인
- 출력:
  - `evidence/2026-03-14_hybrid_rerank_recovery.md`
  - run manifest + per-query 결과
- 완료 조건:
  - B0/B1/B2/B3 + B4/B4.5 동일 eval 셋에서 비교표 확보

### 실행 커맨드 (즉시 실행 가능)

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/run_masked_hybrid_experiment.py
```

### 검증 커맨드

```bash
cd /home/hskim/work/llm-agent-v2
uv run python - <<'PY'
import json
from pathlib import Path

p = Path("data/paper_a/masked_hybrid_results.json")
assert p.exists(), f"missing: {p}"
arr = json.loads(p.read_text(encoding="utf-8"))
assert isinstance(arr, list) and arr, "result must be non-empty list"
first = arr[0]
for key in ("conditions", "target_device", "scope_observability"):
    assert key in first, f"missing field: {key}"
print("masked hybrid result verified")
PY
```

## T6 [P1] P6/P7 soft scoring 재실험

- 목표:
  - 기존 +1.9% 결과를 새 셋(v0.6+)에서 재평가.
- 출력:
  - λ sweep 결과표 (`0.01~0.5` 등)
  - adaptive λ 효과(없음/약함/유의) 판정
- 완료 조건:
  - hard 대비 통계적으로 유의한 개선 여부 결론

### 실행 커맨드 (T5 결과 필요)

```bash
cd /home/hskim/work/llm-agent-v2
uv run python scripts/paper_a/run_masked_p6p7_experiment.py
```

### 검증 커맨드

```bash
cd /home/hskim/work/llm-agent-v2
uv run python - <<'PY'
import json
from pathlib import Path

p = Path("data/paper_a/masked_p6p7_results.json")
assert p.exists(), f"missing: {p}"
arr = json.loads(p.read_text(encoding="utf-8"))
assert isinstance(arr, list) and arr, "result must be non-empty list"
first = arr[0]
assert "conditions" in first and "lambda_p7" in first, "missing conditions/lambda_p7"
print("p6/p7 result verified")
PY
```

## T7 [P1] gcb equip 결측 보정 실험

- 목표:
  - gcb의 equip 결측(약 33%)으로 인한 device fallback 노이즈 완화.
- 방법:
  - 안전 규칙: equip→device 단일매핑 케이스만 자동 보정
  - 충돌 매핑은 보정 금지
- 출력:
  - `evidence/2026-03-14_gcb_equip_imputation.md`
- 완료 조건:
  - 보정 전/후 contamination 및 hit 변화 보고

## T8 [P2] 원고/표/그림 동기화

- 목표:
  - 최신 수치와 본문 불일치 제거.
- 범위:
  - `paper_a_scope.md`, `related_work.md`, main results/evidence 문서
- 완료 조건:
  - 핵심 표/문장 숫자 100% 일치(검증 스크립트 포함)

---

## 3) 실행 순서 (권장)

1. `T1` → `T2` → `T3` → `T4`  (P0 완결)
2. `T5` → `T6` → `T7`         (실험 확장)
3. `T8`                         (원고 동기화)

---

## 4) 제출(Short Paper) 최소 게이트

아래 4개 충족 시 1차 제출 가능:

1. oracle-parser 갭 수치 공개 (`T1`)
2. gold 신뢰도 감사 결과 포함 (`T2`)
3. explicit-only 편향 완화 (`T3`)
4. shared 정책 역설 원인/대응 제시 (`T4`)

---

## 5) 리스크와 대응

- 리스크: parser 성능이 낮으면 masked 개선폭이 크게 축소될 수 있음
  - 대응: oracle 결과는 upper bound로 명시, parser 보정안 병행 보고
- 리스크: strict gold coverage 부족
  - 대응: strict_eligible 분리 지표와 loose 지표 동시 보고
- 리스크: hybrid/rerank 복구 지연
  - 대응: BM25 core story로 short paper 제출, 확장 실험은 camera-ready/후속 논문으로 분리

## 6) Reporting guardrails (metric revision discipline)

진행 중 수치가 바뀌어도 과대주장/재현성 붕괴를 막기 위해 아래를 고정한다.

1. baseline/변경값 동시 보고
   - 표는 항상 `old vs new`를 같이 둔다. (예: `50% -> 92%`)
2. oracle vs realistic 분리
   - oracle 수치는 upper-bound로만 쓰고, parser 기반 실측을 별도 표로 둔다.
3. slice별 보고 고정
   - 최소 `explicit_device`, `explicit_equip`, `implicit`, `masked`별로 contamination/hit를 분리 보고한다.
4. 실행 재현성
   - 각 실험마다 실행 커맨드, 입력 파일 경로, 출력 산출물 경로를 evidence 문서에 남긴다.
5. 변경 이력
   - 핵심 퍼센트 변경 시 evidence 문서 상단 또는 해당 섹션에 날짜 기반 변경 메모를 추가한다.
