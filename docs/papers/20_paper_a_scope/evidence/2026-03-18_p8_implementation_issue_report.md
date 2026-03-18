# P8 구현 이슈 리포트

작성일: 2026-03-18  
대상: `scripts/paper_a/run_p8_evidence_scope_experiment.py`

---

## 1. 검토 범위

- 문법/실행 가능성: `python -m py_compile` 기준 통과
- 알고리즘 동작 정합성
- 실험 비교 공정성
- 논문 주장(특히 end-to-end)과 구현 일치성

---

## 2. 핵심 이슈 요약

| ID | 심각도 | 이슈 | 영향 |
|----|--------|------|------|
| P8-01 | HIGH | `shared_cap`이 사실상 비활성화될 가능성 | `P8_sc1/sc2`가 `P8`과 거의 동일해질 수 있음 |
| P8-02 | MEDIUM | P8은 live ES, baseline은 cached | 조건 불일치로 비교 공정성 저하 |
| P8-03 | MEDIUM | Stage 1이 B3 cached top-10 의존 | "완전 end-to-end scope routing" 주장 약화 |

---

## 3. 상세 이슈

### P8-01 (HIGH): shared_cap 동작 결함 가능성

근거 코드:
- `shared_cap > 0` 분기: `run_p8_evidence_scope_experiment.py:454`
- append 조건: `len(top_doc_ids) < TOP_K` (`:468`)

문제:
- Stage 2 결과가 이미 `TOP_K=10`이면 길이 조건 때문에 shared 문서가 추가되지 않음
- 결과적으로 `P8_sc1_masked`, `P8_sc2_masked`가 `P8_masked`와 동일해질 가능성이 큼

권고:
- shared를 "추가"가 아니라 "치환" 전략으로 변경
- 예: early rank에서 out-of-scope/shared dominance 조건 충족 시 교체

---

### P8-02 (MEDIUM): baseline 대비 실험 조건 불일치

근거 코드:
- baseline 로드: `run_p8_evidence_scope_experiment.py:583-689`
- P8 live retrieval: `:691-704`

문제:
- `B3/B4/B4.5/P7+`는 cached 결과
- `P8`만 현재 ES/모델 상태로 실시간 재검색
- 인덱스/모델 drift가 있으면 성능 차이가 알고리즘 효과와 섞임

권고:
- 옵션 A: baseline도 동일 run에서 live로 재생성
- 옵션 B: P8도 동일 시점 cached candidate 기반으로 비교 버전 추가
- 논문 본문에 "mixed-evaluation condition" 명시

---

### P8-03 (MEDIUM): end-to-end 주장 범위 주의

근거 코드:
- Stage 1 설명: `run_p8_evidence_scope_experiment.py:5-6`
- 실제 입력: B3 cached top-docs 사용 `:646-649`
- 스펙 문서도 옵션 A(cached) 명시: `2026-03-18_p8_algorithm_spec.md:150-159`

문제:
- hypothesis generation이 B3 캐시에 의존함
- 따라서 완전 독립 라우터라기보다 "B3-assisted scope selection"에 가까움

권고:
- 논문 문구를 "end-to-end retrieval loop with cached hypothesis source"로 제한
- 후속 실험에서 live probe(`top-40`) 기반 Stage 1 버전 추가

---

## 4. 우선 수정 순서

1. `P8-01` 수정 (`shared_cap` 실제 작동 보장)
2. `P8-02` 비교 조건 통일 (live-live 또는 cached-cached)
3. `P8-03` 문구 보정 + live probe ablation 추가

---

## 5. 검증 체크리스트 (수정 후)

- [ ] `P8_sc1/sc2`와 `P8` top-10 동일 비율 보고 (`identical_top10`)
- [ ] baseline/P8 동일 조건 비교 표(동일 시점) 생성
- [ ] Stage 1 source별 성능 비교 (cached vs live probe)
- [ ] 논문 캡션에 평가 조건 명시 (offline/online, cached/live)

