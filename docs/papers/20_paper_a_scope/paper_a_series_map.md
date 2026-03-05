# Paper A Series Map

## Purpose

Paper A 본체와 확장형 두 축(A-1, A-2)의 범위를 고정해서, 실험/집필 우선순위를 일관되게 유지한다.

상세 RQ/H/실험/데이터셋/완성조건은 `paper_a_series_blueprint.md`를 기준으로 관리한다.

## Paper A (Core)

- 제목 개념: Hierarchy- and DocType-Aware Contamination-Controlled Retrieval for Semiconductor Maintenance RAG
- 한 줄 정의: device/equip/doc_type/shared/family 정보를 이용한 contamination-aware retrieval 정책(비학습형 코어)
- 방법 성격: training-free(또는 약한 튜닝), retrieval/reranking 중심
- 핵심 구성:
  - hierarchy-aware scope (device, equip, family, shared)
  - doc-type-aware scope (procedure vs log/history)
  - contamination-aware scoring (`Score = Base - lambda * v_scope`)
- 핵심 평가:
  - Raw/Adjusted/Shared Contamination@k
  - Hit@k / MRR
  - Equip-level contamination (조건부)
  - latency (선택)
- 포지션: 가장 먼저 쓰는 본체 논문

## Paper A-1 (Option 1: CHNM)

- 제목 개념: CHNM: Cross-Hierarchy Negative Mining for Contamination-Aware Reranking in Equipment-Constrained Retrieval
- 한 줄 정의: cross-device/cross-equip 혼동을 hard negative로 수집해 contamination-aware reranker/retriever를 학습하는 확장형
- 방법 성격: 학습형 강화 (weak supervision 기반 가능)
- 핵심 아이디어:
  - Cross-device hard negative
  - Same-device wrong-equip negative
  - Family-confusable negative
  - Wrong-doc-type negative
- 입력 데이터:
  - Paper A gold set
  - retrieval top-k 로그
  - 문서 메타(device/equip/doc_type)
  - scope violation 라벨(자동 생성 가능)
- 핵심 평가:
  - Paper A 대비 contamination 추가 감소
  - recall 유지 여부
  - split별 개선 폭(explicit/implicit/equip-centric)
  - negative type별 효과
- 포지션: Paper A의 학습형 강화판

## Paper A-2 (Option 2: Evidence Consistency Gate)

- 제목 개념: Evidence Consistency Gating for Safe Generation under Cross-Equipment Retrieval Uncertainty
- 한 줄 정의: evidence 혼합 위험 시 재검색/확인질문/보류를 수행해 citation contamination을 낮추는 생성단 안전 확장
- 방법 성격: retrieval+generation 연결부 fail-safe 정책
- 핵심 아이디어:
  - top-k evidence consistency 신호(device/equip entropy, 불일치율 등) 계산
  - risk 수준별 분기:
    - low: 바로 답변
    - medium: stricter rerank 또는 re-retrieval
    - high: clarification 또는 abstention
- 입력 데이터:
  - Paper A retrieval 결과
  - citation logs
  - final answer outputs
  - device/equip/doc_type metadata
- 핵심 평가:
  - Citation Contamination
  - unsafe answer rate
  - clarification rate
  - coverage vs safety trade-off
- 포지션: Paper A의 안전 응답 정책 확장판

## Relationship and Priority

- Paper A: 정책 기반 core retrieval (중심축)
- Paper A-1: CHNM 기반 학습형 강화
- Paper A-2: evidence consistency 기반 생성 안전 확장

권장 우선순위:
1. Paper A
2. Paper A-1
3. Paper A-2
