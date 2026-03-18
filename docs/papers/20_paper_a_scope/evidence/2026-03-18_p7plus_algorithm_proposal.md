# P7+ 알고리즘 제안서 (Paper A)

Date: 2026-03-18
Status: proposal (based on current negative result evidence)

## 0) 배경

현재 `P6/P7`은 masked v0.6+ 설정에서 `B3_masked` 대비 성능 개선을 만들지 못했다.

- `P6_masked`, `P7_masked`: cont@10 `0.649`, gold_strict `351/578`
- `B3_masked`: cont@10 `0.584`, gold_strict `351/578`
- `B4_masked`: cont@10 `0.001`, gold_strict `527/578`

근거: `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`

## 1) 현재 P6/P7 실패 원인 (확인)

### 1.1 약한 penalty로 상위 랭크 재정렬이 거의 불가

현 구현은 `base(rank)=1/(rank+1)` 위에서 `lambda<=0.05` penalty를 적용한다.

```text
score = base(rank) - lambda * v_scope
```

`rank 1<->2` 뒤집힘에는 최소 `~0.5`, `rank 2<->3`에는 `~0.1667` 수준이 필요해
현재 lambda 범위(`<=0.05`)로는 top-rank 뒤집힘이 거의 발생하지 않는다.

근거:
- `scripts/paper_a/run_masked_p6p7_experiment.py`
- `docs/papers/20_paper_a_scope/evidence/2026-03-14_masked_p6p7_reexperiment.md`

### 1.2 후보집합을 바꾸지 않고 재정렬만 수행

현 구현은 `B3_masked top_doc_ids`를 그대로 받아 soft re-rank만 수행한다.
즉 문서 후보집합 자체는 바뀌지 않는다.

근거: `scripts/paper_a/run_masked_p6p7_experiment.py`

### 1.3 shared overload 문제를 penalty만으로 제어하지 못함

`B4.5 < B4` 역설의 주원인은 `shared_overload`(85.7%)로 보고되었다.
이는 단순 penalty보다 정책적 cap/ordering 제어가 필요하다는 신호다.

근거: `docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md`

## 2) 제안: P7+ (Confidence-Gated Scope Scoring)

P7+는 "soft scoring 단독"이 아니라 "hard/soft/gating"을 결합한다.

### 2.1 핵심 아이디어

1. parser confidence가 높으면 hard device filter를 우선 사용한다.
2. confidence가 중간이면 family/shared를 제한적으로 열되, shared 노출 cap을 건다.
3. confidence가 낮으면 fallback 모드로 가되 contamination penalty를 강화한다.

### 2.2 수식

쿼리 `q`, 문서 `d`에 대해

```text
Score+(d,q) = Base(d,q)
            - λ(q) * v_scope(d,q)
            - μ(q) * v_shared_over(d,q)
            + η(q) * v_target_device(d,q)
```

- `v_scope(d,q)=1` if out-of-scope and not shared, else `0`
- `v_shared_over(d,q)=1` if shared doc and shared exposure cap exceeded, else `0`
- `v_target_device(d,q)=1` if doc device == target device, else `0`

### 2.3 confidence proxy 기반 파라미터

`conf(q)`를 parser confidence proxy(파싱 mode/근거 강도 기반)로 두고, 구간별로 정책 전환:

- `conf >= t_high`: hard device filter (B4-like upper bound; oracle B4와 parser 기반 realistic 결과를 분리 보고)
- `t_mid <= conf < t_high`: device + limited shared (`shared_cap = c1`)
- `conf < t_mid`: relaxed retrieval + stronger contamination penalty (`λ↑`, `μ↑`)

## 3) 최소 실험 계획 (P7+ 검증)

### 3.1 비교군

- Baseline: `B3_masked`, `B4_masked`, `B4.5_masked`, `P6_masked`, `P7_masked`
- Proposal: `P7+_masked`

### 3.2 필수 메트릭

- `cont@10`
- `gold_hit_strict@10`, `gold_hit_loose@10`
- `MRR`
- by-scope (`explicit_device`, `explicit_equip`) 분리

### 3.3 기대 성공 조건

- `P7+_masked`가 최소 `P7_masked`보다 contamination을 낮출 것
- `gold_hit_strict`를 `P7`보다 유지/개선할 것
- `explicit_equip`에서 contamination 악화를 멈출 것

## 4) 논문 포지셔닝 제안

`P6/P7`을 그대로 주기여로 두기보다, 아래처럼 바꾸는 것이 안전하다.

- 기존 P6/P7: "음성 결과(negative result)"로 유지
- 새 기여: `P7+`를 "hard-soft hybrid contamination control"로 제안

이렇게 하면 현재 evidence와 충돌하지 않으면서도, 알고리즘 기여 축을 복원할 수 있다.

## 5) 즉시 후속 작업

1. `run_masked_p6p7_experiment.py`를 `P7+` 옵션까지 확장
2. `shared_cap`, `t_high`, `t_mid`, `λ/μ/η` 그리드 탐색(작은 범위)
3. `2026-03-14_masked_p6p7_reexperiment.md` 형식으로 `P7+` 결과 문서화
4. Oracle 상한선(`B4`)과 realistic(parser/scope-aware) 결과를 같은 표에서 분리 보고
ㄹ