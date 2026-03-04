# [Role: Paper Working Doc] Paper B - Stability-Aware Retrieval

## 1) 임시 제목

- 영문: **Stability-Aware Retrieval for Reliable RAG Operations: Optimizing Top-k Consistency under Repeated and Equivalent Queries**
- 국문: **RAG 운영 신뢰성을 위한 안정성 인지 검색: 동의 질의 및 반복 질의에서 Top-k 일관성 최적화**

## 2) One-Line Claim

정확도 중심 retrieval에 안정성 목표를 추가하면, 동의 질의/반복 질의에서 top-k 흔들림을 줄이고 결론 재현성을 높일 수 있다.

## 3) 연구질문과 가설

- RQ: retrieval 비결정성과 표현 변형으로 생기는 결과 변동을 통제할 수 있는가?
- H-B1: stability-aware 설정은 `Stability@k(Jaccard)`와 `Consistency`를 유의하게 개선한다.
- H-B2: 안정성 개선은 `Recall@k` 손실 없이 또는 제한된 손실로 달성 가능하다.
- H-B3: instability는 ANN 비결정성/문서 중복/질의 모호성/업데이트 요인으로 분해 가능하다.

## 4) 문제정의/방법

- 동의 질의 집합: `Q(q)={q, q~1, ...}`
- 강건 목적:
  - `max E[R@k]  s.t. Stab@k >= τ`
- 정규화 목적:
  - `max E[R@k - μ(1-Stab@k)]`
- 구현 축(최소 2개):
  - deterministic control (seed, ANN, tie-break, cache)
  - consensus top-k (다중 샘플/다중 retriever 합의)
  - paraphrase consistency regularization (선택)

## 5) 지표와 실험 설계

핵심 지표:
- `Stability@5(Jaccard)` (primary)
- `Top-5 동일률`
- `Conclusion consistency`
- `Recall@5`
- `Latency p95`

실험 프로토콜:
- paraphrase group 단위 평가셋 고정
- 그룹당 3~5개 문장, 변화 강도(낮/중/높) 태그
- 반복 실행(예: 10~20회)과 paraphrase 변형을 분리 측정

비교군:
- Baseline (Hybrid+Rerank, MQ 설정 원형)
- +Deterministic control
- +Consensus retrieval
- +Regularization(가능 시)

## 6) 그림/표 계획

- Fig B1: stability-aware retrieval 구조(2개 방법 비교)
- Table B1: `Recall@5`, `Jaccard@5`, `Consistency`, `p95`
- Fig B2: Stability-Recall 파레토 곡선
- Table B2: instability 원인 분해(ANN/중복/모호성/업데이트)
- Table B3: paraphrase 강도별 성능

## 7) 제출 Gate

- 평균 정확도보다 분산/재현성 지표 개선이 핵심 claim으로 제시됨
- `Stability@5` 개선이 명확하며 latency 영향이 설명됨
- 불안정 원인 분해 결과 포함

## 8) 데이터/자산 요구사항

- `paraphrase_group_id` 필드
- B 전용 `Paraphrase Stability Set`
- 반복 실행 로그(top-k ids, score vector, seed, ANN 설정)
- 공통 규격: `paper_common_protocol.md`

## 9) 현재 상태와 다음 2주

현재 상태:
- MQ temperature 비결정성 이슈 확인
- retrieval fallback 경로 차이로 run-to-run 편차 가능성 존재

다음 2주:
1. paraphrase set v0 구축
2. deterministic on/off + MQ 설정 sweep
3. 반복 실행 n=20 분산 분석
4. 안정성 개선안 1개를 baseline 대비 검증
