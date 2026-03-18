# Paper A 타임라인 쉬운 요약 (한글)

작성일: 2026-03-18  
용도: Paper A 진행 흐름을 빠르게 이해하기 위한 쉬운 요약본

## 1) 한 줄 요약

Paper A는 처음에는 "하드 필터가 recall을 깬다"는 결론처럼 보였지만,  
평가 방식의 편향을 고친 뒤에는 "장비 스코프 필터가 contamination을 크게 줄이고 hit도 올릴 수 있다"로 결론이 바뀌었다.

---

## 2) 날짜별 핵심 흐름

## 2026-03-05 (초기 라운드)

- v0.5 기반 실험에서 하드 필터가 성능을 깎는 것처럼 보임
- 하지만 이 시점 결과는 아래 이유로 해석이 불안정했음:
  - alias/파서 이슈
  - shared 메트릭 해석 문제
  - 작은 평가셋

## 2026-03-12 (전환점)

- 평가 프로토콜을 재설계함
- 핵심: `question_masked` 도입(질문에서 장비명 단서를 제거)
- 의미: "필터가 진짜로 필요한 상황"을 더 잘 측정하게 됨

## 2026-03-13 (돌파구)

- masked BM25 실험에서 중요한 결과 확인
  - B0_masked contamination@10: `0.518`
  - B4_masked contamination@10: `0.000`
  - B0_masked loose hit@10: `287/578 (50%)`
  - B4_masked loose hit@10: `530/578 (92%)`
- 즉, 이 설정에서는 하드 필터가 contamination을 줄이면서 hit도 크게 개선

## 2026-03-14 (현실성 점검)

- oracle vs parser 갭을 분리해 확인
  - oracle은 상한선(upper bound)으로만 써야 함
- gold audit, mixed-scope 복원, B4.5 역설 분석, P6/P7 재실험까지 진행
- 결론:
  - naive shared(B4.5)는 현재 설정에서 B4보다 불리
  - soft scoring(P6/P7)은 이번 셋업에서 유의미한 개선 없음

## 2026-03-15 (숫자 동기화)

- progress summary의 hit 표를 per-query 결과 파일 기준으로 정합성 맞춤
- 방향성 결론은 유지, 숫자만 정확히 정리

---

## 3) 지금 기준으로 안전하게 말할 수 있는 결론

1. cross-equipment contamination은 실제로 크다.
2. oracle device filter는 contamination을 거의 0으로 만든다.
3. masked 평가에서는 hard filter가 hit를 높이는 효과가 확인된다.
4. 단, 운영 성능은 parser 품질에 크게 좌우된다.

---

## 4) 아직 조심해서 말해야 하는 부분

1. oracle 결과를 운영 성능처럼 말하면 안 됨
2. implicit/ambiguous 혼합 조건에서의 full hybrid 평가는 아직 막힌 부분이 있음
3. gold audit은 더 큰 표본/더 엄격한 신뢰구간 보고가 필요
4. shared 정책(B4.5)은 현재 형태로는 개선 효과가 불안정

---

## 5) 지금 논문 본문에 우선 넣을 숫자(권장)

- BM25 masked B0 contamination@10 (ALL): `0.518`
- BM25 masked B4 contamination@10 (ALL): `0.000`
- BM25 masked B0 loose hit@10 (ALL): `287/578 (50%)`
- BM25 masked B4 loose hit@10 (ALL): `530/578 (92%)`
- parser(device-only) adj_cont@10: `30.6%`
- scope-aware realistic parser adj_cont@10: `8.5%`
- v0.6 strict precision(샘플 감사): `97.2% (172/177)`

---

## 6) 다음 작업 우선순위 (짧게)

1. oracle vs parser 갭을 본문 표로 고정
2. implicit/ambiguous 축 보강 후 재평가
3. shared 정책(B4.5) 보정안 적용 후 재실험
4. 원고 전체 숫자 동기화(표/문장/부록)

---

## 7) 원문 참고

- 상세 타임라인: `2026-03-14_paper_a_timeline_consolidated.md`
- 실행계획: `2026-03-14_execution_tasks.md`
- 핵심 요약: `evidence/2026-03-13_paper_a_progress_summary.md`
- 현실성 점검: `evidence/2026-03-14_oracle_vs_parser_gap.md`
