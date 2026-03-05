# Paper A — Hierarchy-aware Scope Routing for Cross-Equipment Contamination Control

## Goal

반도체 Fab 유지보수 RAG에서 **cross-equipment contamination(타 장비 문서 혼입)**을 줄이되,
**공용 SOP/유사 장비로 인한 recall 손실을 최소화**하는 스코프 정책을 설계/검증한다.

## Research Question

**RQ-A**: 장비 계층 메타데이터 기반 scope routing 정책(Hard/Family/Shared)을 적용하면,
retrieval contamination을 유의하게 줄이면서 recall 손실을 억제할 수 있는가?

## Core Contributions (4축, v0.4)

1. **(주기여) Hierarchy-aware Scope Routing Policy (G)**
   - 질의에서 장비/설비 스코프를 robust하게 결정 (명시/비명시/모호/공용 문서 포함)
   - AllowedScope(q) = Hard ∪ Family ∪ Shared 3단 정책 설계
   - Contamination@k를 first-class safety metric으로 정의 (Raw/Adjusted/Shared 3종 보고)

2. **(알고리즘 기여) Contamination-aware Scoring + 적응형 λ(q)** ← v0.4 신규
   - `Score(d,q) = Base(d,q) - λ(q)·v_scope(d,q)`: scope 위반을 목적함수 수준에서 penalty
   - λ(q)를 라우팅 확신도(router confidence)에 따라 적응적으로 조절
   - 확실→hard filter 수렴, 모호→soft penalty로 recall 보존
   - C4(Hard↔Soft 전환) + C5(Contamination-aware reranking)를 하나의 점수함수로 통합

3. **(현장 반영 기여) Family Scope + Shared Doc Policy**
   - 공유 문서 기반 장비 유사도 그래프로 equipment family 구축
   - 공용 SOP/TS 문서를 D_shared로 분류하여 contamination 정의에서 예외 처리
   - Hard filter의 recall 손실 문제를 family 확장으로 해결

4. **(효율/실용화 기여) Matryoshka 저차원 라우터**
   - 저차원(64/128/256d) device prototype 매칭으로 scope 후보 Top-M을 빠르게 선정
   - 장비명 미기재/모호 질의에서 auto-parse 한계를 보완
   - 대규모 코퍼스에서 스코프 제약 파이프라인의 latency/memory를 실용적 수준으로 유지

## Positioning

- **G(라우팅 정책)가 메인 기여** — "스코프를 어떻게 결정하는가"가 핵심 연구 질문
- **F(인덱스 분리/선택)는 배포 구현의 변형** — 효율 비교 실험에서 보너스로 포함
- **Matryoshka는 G를 실용화하는 엔진** — "라우팅을 싸게 만드는 도구"로 포지셔닝

## Main Metrics

| 카테고리 | 메트릭 | 설명 |
|---------|--------|------|
| Contamination (주) | `Contamination@k` | top-k 중 out-of-scope 비율 (Raw + Adjusted + Shared@k 3종) |
| Contamination (주) | `ContamExist@k` | 하나라도 out-of-scope이면 1 |
| Quality | `Hit@k`, `MRR` | expected_doc/pages 기준 |
| Router | `ScopeAccuracy@M` | 정답 device가 Top-M에 포함되는 비율 |
| Efficiency | `p95_latency_ms`, `index_size_mb` | 실용성 검증 |
| Generation (보조) | `CiteCont` | 답변 인용 중 out-of-scope 비율 |

## Key Decisions (확정)

| 항목 | 결정 | 근거 |
|------|------|------|
| **Venue** | CIKM Applied/Industry 1차, SIGIR Industry 2차 | 산업 적용 + 실증 스토리가 강점 |
| **Contamination 보고** | Raw + Adjusted + Shared@k 3종 동시 | 심사자 투명성 요구 대응 |
| **Family 구성** | 공유 문서 그래프 1순위, 임베딩 centroid 보조 | 설명 가능성 최우선 |
| **평가셋** | Explicit(SOP79) + Mask set + Ambiguous + Real-Implicit(가능시) | 라우팅 필요성 증명 |
| **실험 경로** | retrieval-only 주 실험, agent/run 보조 | IR 기여 분리/통제 |
| **Matryoshka** | 1단계 라우터만, 2단계 chunk 교체는 여력 시 | 범위 제어 |
| **초기 파라미터** | `T(shared)=3`, `tau(family)=0.2`, `M(top device)=3`, `router_dim=128` | 착수 기준선 (dev에서 고정 후 test 평가) |

## Corpus Summary (578 docs)

| Family | 문서 수 | 비율 |
|--------|---------|------|
| SUPRA | 329 | 56.9% |
| INTEGER | 89 | 15.4% |
| PRECIA | 71 | 12.3% |
| GENEVA | 69 | 11.9% |
| OMNIS | 15 | 2.6% |
| 기타 | 5 | 0.9% |

- 21% of topics shared across 2+ devices → family/shared policy justified
- Same topic = separate doc_ids per device → **topic-based similarity** preferred over doc_id sharing

## Open Questions (잔여)

1. 현재 embedding 모델의 MRL(Matryoshka) 지원 여부
2. ES 인덱스 실제 chunk 수 vs 파싱 문서 578건 차이
3. doc_id 공유 vs topic 공유: family 구축 기준 확정 필요

## Paper Lineup

- Paper A / A-1 / A-2 구분 문서: `paper_a_series_map.md`
- Paper A / A-1 / A-2 상세 설계(RQ/H/실험/데이터셋/완성조건): `paper_a_series_blueprint.md`

## File Structure

```
20_paper_a_scope/
├── README.md                 ← 이 파일 (개요)
├── paper_a_series_map.md     ← Paper A / A-1 / A-2 포지셔닝
├── paper_a_series_blueprint.md ← Paper A / A-1 / A-2 상세 실험 블루프린트
├── paper_a_scope_spec.md     ← 실험 정의서 v0.2
├── related_work.md           ← 문헌 조사 scaffold
├── references.bib            ← BibTeX 참고문헌
├── evidence_mapping.md       ← claim → evidence 매핑
└── evidence/                 ← 실험 근거 문서
    └── 2026-03-04_corpus_statistics.md
```

## Cross-references

- 공통 프로토콜: `../10_common_protocol/paper_common_protocol.md`
- 논문 전략: `../00_thesis_strategy/`
- 실험 정의서: `paper_a_scope_spec.md`
- 시리즈 포지셔닝: `paper_a_series_map.md`
- 시리즈 상세 블루프린트: `paper_a_series_blueprint.md`
