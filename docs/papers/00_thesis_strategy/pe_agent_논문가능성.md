# [Role: 전략 평가] PE Agent → IE 박사 논문 가능성 평가

> 작성일: 2026-02-28
> 전제: 산업공학(IE) 박사, Fab 폐쇄망 PE Agent 프로젝트 기반
> [Role: 논문 가능성 진단 / 리스크 분석 / 주제 후보 선별]

---

## 1. 현재 시스템 자산 요약

| 자산 | 설명 |
|---|---|
| PE Agent RAG | Fab 폐쇄망 환경의 유지보수 지식 검색+생성 시스템 |
| Hybrid Retrieval | BM25 + Dense + Rerank 파이프라인 (ES 기반) |
| MQ Strategy | 다중 질의 생성 → 확장 검색 → 결과 합의 |
| Deterministic Protocol | seed/shard-routing/tie-break 기반 검색 재현성 보장 |
| Stability Evaluation | T1-T4 안정성 계층, Jaccard@10 지표, 합성 벤치마크 |
| 3-Tier Retry | retry_expand → refine_queries → retry_mq 단계적 복구 |
| mq_mode Policy | off/fallback/on 3단계 비결정성 통제 |
| Search Query Guardrails | G0-G5 프로그래밍 방식 질의 품질 보증 |

---

## 2. 리스크 분석

### Risk 1: "엔지니어링만 했다" 비판

- **위험**: 시스템 구현에 그치면 학술 기여가 불충분
- **대응**: 운영 신뢰성 지표를 정식화하고 실험으로 검증 → 지표/최적화/검증 프레임워크가 학술 기여
- **IE 관점**: 품질공학, 신뢰성공학의 형식 언어로 문제 정의

### Risk 2: "데이터 비공개" 문제

- **위험**: Fab 데이터는 보안상 공개 불가 → 재현성 의심
- **대응**: (1) 합성 벤치마크 공개, (2) 공개 도메인(의료/법률 매뉴얼) 2차 검증, (3) 실험 프로토콜/코드 공개
- **선례**: 삼성/TSMC 관련 논문도 동일 제약하에 학술지 게재 이력 다수

### Risk 3: LLM/RAG 분야 빠른 변화

- **위험**: 2-3년 연구 기간 중 기술 landscape 변화
- **대응**: 방법론이 아닌 문제 정의 + 지표 체계가 핵심 기여 → 기술 변화에 불변하는 프레임워크
- **보험**: 문제 정의/지표가 유효하면 specific method는 교체 가능

### Risk 4: IE 학과 수용성

- **위험**: NLP/ML 중심 연구가 IE 학과에서 논문 방어 통과 가능성
- **대응**: 품질특성(정확도/안전성/반복성/충실성/지속가능성)을 IE 품질공학 체계로 매핑, SPC/신뢰성 분석 도구 활용
- **선례**: IE PhD에서 NLP+품질관리 융합 연구 증가 추세 (KAIST, SNU IE)

### Risk 5: 단일 도메인 일반화 의문

- **위험**: Fab PE 도메인 한정 → "이것만 되는 것 아니냐"
- **대응**: (1) 합성+공개 도메인 이중 검증, (2) 프레임워크 수준 기여는 도메인 독립적, (3) ablation으로 도메인 특성 분리

---

## 3. 주제 후보 6개

| # | 주제 | Paper 매핑 | IE 적합도 |
|---|---|---|---|
| 1 | 장비 계층 제약 기반 검색 정확성 (Scope Safety) | Paper A | ★★★★ |
| 2 | 동의/반복 질의 검색 안정성 최적화 | Paper B | ★★★★★ |
| 3 | 수치/조건 정합성 검증 (Faithfulness) | Paper D | ★★★★ |
| 4 | 지식 업데이트/드리프트 통제 (Lifecycle) | Paper C | ★★★★★ |
| 5 | 통합 운영 리스크 최소화 프레임워크 | Journal J | ★★★★★ |
| 6 | LLM 기반 산업 지식서비스 품질 지표 체계 | Thesis 전체 | ★★★★ |

### 주제별 상세

**주제 1 - Hierarchy-Constrained Retrieval (Paper A)**
- 핵심: "다른 장비 문서가 섞이면 안 된다"를 계층 제약으로 정식화
- IE 연결: 계층적 분류 체계(taxonomy) + 제약 최적화
- 실험 가능성: contamination rate 측정, 제약 전/후 비교

**주제 2 - Stability-Aware Retrieval (Paper B)**
- 핵심: 같은 질문에 같은 답 → Top-k 일관성을 목적함수에 추가
- IE 연결: 변동성 최소화, 강건 최적화, SPC 관점
- 실험 가능성: ★ (이미 T1-T4 프로토콜 + 합성 벤치마크 구축 완료)

**주제 3 - Numeric/Condition Faithfulness (Paper D)**
- 핵심: "온도 150°C인데 200°C라고 답했다" 유형 오류 방지
- IE 연결: 공정 조건 정합성, 수치 검증 자동화
- 실험 가능성: gold label 필요, annotation 비용 중간

**주제 4 - Lifecycle Reliability Control (Paper C)**
- 핵심: 문서 업데이트 → 인덱스 변경 → 성능 회귀 감지/통제
- IE 연결: SPC(통계적 공정관리), 드리프트 감지, 변경관리
- 실험 가능성: 버전별 인덱스 스냅샷 필요

**주제 5 - 통합 운영 리스크 프레임워크 (Journal J)**
- 핵심: `Risk = c₁·C@k + c₂·(1-Stab@k) + c₃·E[V]` 복합 리스크
- IE 연결: 다목적 최적화, 리스크 관리, 종합 품질 지표
- 실험 가능성: Paper A/B/C/D 결과 통합으로 구성

**주제 6 - 산업 지식서비스 품질 지표 체계**
- 핵심: 정확도 + 안전성 + 반복성 + 충실성 + 지속가능성 5대 품질특성
- IE 연결: 품질경영(TQM), 서비스 품질 모델(SERVQUAL 변형)
- 실험 가능성: 프레임워크 제안 + Paper A/B/C/D로 실증

---

## 4. IE 박사 관점 재배치

### 왜 IE인가?

기존 NLP/IR 연구와의 차별점:

| 관점 | NLP/IR 논문 | IE 박사 논문 (본 연구) |
|---|---|---|
| 목적함수 | 평균 정확도 최대화 | 운영 리스크(정확도+안정성+정합성) 최소화 |
| 품질 정의 | Recall, Precision, F1 | 5대 품질특성 (정확도/안전성/반복성/충실성/지속가능성) |
| 안정성 취급 | 부수적 관찰 | 최적화 대상 (제약 또는 목적함수) |
| 변화 관리 | 언급 없음 | SPC 기반 드리프트 감지/통제 |
| 실험 설계 | 벤치마크 리더보드 | 교란 계층(T1-T4) 기반 체계적 실험 |
| 현장 연결 | 없음 | 재작업률/대응시간 등 현장 KPI |

### 추천 "큰 줄기" 3개

1. **줄기 A**: 스코프 안전성 (Paper A) → "틀린 장비 문서가 안 섞이게"
2. **줄기 B**: 검색/결론 안정성 (Paper B) → "같은 질문에 같은 답"
3. **줄기 C**: 라이프사이클 안정성 (Paper C) → "업데이트해도 성능 안 빠지게"

→ 이 3개 줄기가 박사논문의 Ch4/Ch5/Ch6이 되고, 각각 Conference Paper 1편씩 매핑
→ 통합 프레임워크(Ch7)가 Journal J 매핑

---

## 5. 리스크 공식화

### 운영 리스크 정의

```
Risk(q, θ) = c₁ · Contamination@k(q, θ)
           + c₂ · (1 - Stability@k(q, θ))
           + c₃ · E[Violation(q, θ)]
```

- `Contamination@k`: 타 장비군 문서 혼입률 (Paper A)
- `Stability@k`: Top-k Jaccard 일관성 (Paper B)
- `Violation`: 수치/조건 불일치율 (Paper D)
- `θ`: 시스템 설정 (검색 전략, MQ 모드, rerank 여부 등)
- `c₁, c₂, c₃`: 현장 가중치 (재작업 비용, 신뢰도 손실 등)

### 품질특성 5대 체계

| 품질특성 | 지표 | 논문 매핑 |
|---|---|---|
| 정확도 (Accuracy) | Recall@k, MRR, hit@k | 모든 Paper |
| 안전성 (Safety/Scope) | Contamination@k | Paper A |
| 반복성 (Repeatability) | RepeatJaccard@k, ParaphraseJaccard@k | Paper B |
| 충실성 (Faithfulness) | Violation rate, Numeric accuracy | Paper D |
| 지속가능성 (Sustainability) | Drift rate, Regression rate | Paper C |

---

## 6. 최종 평가

### 가능성 판정: **높음 (추천)**

**강점:**
- 실제 운영 시스템 기반 → 실증 데이터 확보 가능
- IE 품질공학 프레임으로 자연스러운 학술 포장
- Paper B 이미 상당 부분 진행 → 빠른 첫 출판 가능
- 5대 품질특성이 박사논문 챕터 구조와 직결

**주의점:**
- 합성 벤치마크 + 공개 도메인 이중 검증 필수
- 지도교수의 IE+NLP 융합 연구 수용성 사전 확인 필요
- Q1 저널 타겟 설정 시 IISE Transactions / JQT / Computers & IE 등 고려

**추천 첫 단계:**
1. Paper B (안정성) 완성 → 컨퍼런스 투고
2. Golden set v0 구축 (50건 pilot)
3. Paper A (스코프 제약) 실험 시작
4. 지도교수 미팅에서 프레임워크 발표

→ 상세 연구 설계: `docs/papers/00_strategy/research_toc.md` 참조
→ 공통 실험 규격: `docs/papers/10_common_protocol/paper_common_protocol.md` 참조
→ 실행 체크리스트: `docs/papers/00_strategy/ie_phd_rag_reliability_roadmap.md` 참조
