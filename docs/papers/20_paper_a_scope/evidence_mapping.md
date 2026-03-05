# Paper A Evidence Mapping

## Purpose

Paper A의 claim을 (1) 문헌 근거, (2) 실험 evidence에 매핑.
기여 3축(G: 라우팅 정책, Family/Shared 정책, Matryoshka 효율화)에 맞춰 구성.

---

## Claim-to-Evidence Map

| Claim ID | Claim | Literature Support | Experiment Evidence | Ablation Row |
|----------|-------|-------------------|-------------------|-------------|
| A-C1 | RAG에서 retrieval quality가 generation quality를 결정한다 | `lewis2020rag`, `guu2020realm`, `petroni2021kilt` | B0-B3 baseline quality 비교 | B0-B3 |
| A-C2 | 글로벌 검색(scope 없음)은 cross-equipment contamination을 유발한다 | (gap statement) | B2/B3의 Cont@k 측정 → 높은 contamination | B2, B3 |
| A-C3 | Hard filter(auto-parse)는 contamination을 줄이지만 recall도 떨어뜨린다 | `robertson2004bm25f`, faceted search | B4 vs B3: Cont@k↓ + Hit@k↓ trade-off | B4 |
| A-C4 | Shared doc policy는 공용 SOP의 false reject를 방지한다 | (domain: 77 shared topics across 2+ devices) | P1 vs B4: Cont@k 유지 + Hit@k↑ (공용 SOP 복구) | P1 |
| A-C5 | Matryoshka 라우터는 장비명 미기재 질의에서도 scope를 robust하게 잡는다 | `kusupati2022matryoshka`, `amazon2022hqc` | P2: ScopeAccuracy@M on Mask set | P2 |
| A-C6 | Family 확장은 유사 장비 문서로 인한 recall 손실을 회복한다 | (Jaccard topic graph) | P3 vs P2: Hit@k↑ on Ambiguous set | P3 |
| A-C7 | 전체 정책(Router+Family+Shared)이 contamination-recall trade-off를 최적화한다 | (integrated) | P4: Cont@k↓ + Hit@k 유지/향상 (3개 서브셋) | P4 |
| A-C8 | Matryoshka 저차원은 latency/memory를 절감하면서 routing 성능을 유지한다 | `kusupati2022matryoshka`, `li2024twodmatryoshka` | dim ablation: 128d vs 768d router quality + latency 비교 | Matryoshka abl. |
| A-C9 | Contamination-aware scoring은 순수 relevance reranking보다 contamination을 추가 감소시킨다 | (gap: 기존 RAG reranking은 relevance만 최적화) | P6 vs P4: Cont@k 비교 (동일 scope 정책에서 scoring 효과 분리) | P6 |
| A-C10 | 적응형 λ(q)는 고정 λ 대비 모호 질의에서 recall 손실을 억제한다 | (C4+C5 통합 기여) | P7 vs P6: Masked/Ambiguous subset에서 Hit@k 비교 | P7 |

---

## Expected Figures/Tables

| ID | Type | Content | Primary Claims |
|----|------|---------|---------------|
| Fig-A1 | Architecture | Scope routing pipeline (parse → route → filter → retrieve → rerank) | Overview |
| Fig-A2 | Trade-off plot | Contamination@k vs Hit@k (B3/B4/P1/P2/P3/P4 비교) | A-C2~C7 |
| Fig-A3 | Bar chart | ScopeAccuracy@M by query type (Explicit/Masked/Ambiguous) | A-C5 |
| Tab-A1 | Main result | 12-row ablation (B0-P7) × {Raw Cont@5, Adj Cont@5, CE@5, Hit@5, MRR, latency} | All |
| Tab-A2 | Matryoshka abl. | dim × M × {ScopeAccuracy, Cont@5, latency, memory} | A-C8 |
| Tab-A3 | Error analysis | Failure mode breakdown: false reject / scope miss / shared ambiguity | A-C3, A-C4 |
| Fig-A4 | λ sensitivity | λ 값에 따른 Cont@5 vs Hit@5 trade-off curve | A-C9, A-C10 |
| Tab-A4 | Scoring ablation | P4 vs P6(λ fixed) vs P7(λ adaptive) × subset별 결과 | A-C9, A-C10 |

---

## Contamination 보고 체계 (확정)

논문 표/그래프에서 Contamination을 3종으로 분해 보고:

| 메트릭 | 정의 | 역할 |
|--------|------|------|
| **Raw Cont@k** | shared도 타 장비면 오염 처리 (엄격) | 투명성/기준선 |
| **Adjusted Cont@k** | shared 제외한 실질 오염 | **주장 메트릭** |
| **Shared@k** | top-k 중 shared 문서 비율 | 도메인 특성 증명 |

- Shared 임계치 T 민감도 분석 → Appendix
- 모든 실험은 Explicit / Masked / Ambiguous 서브셋별로 분리 보고

---

## Data Dependencies

| Artifact | Status | Blocking Claims |
|----------|--------|----------------|
| SOP79 golden set | 있음 | B0-B4 (Explicit subset) |
| Mask set (device 토큰 제거) | **미생성** | A-C5 (P2 ScopeAccuracy) |
| Ambiguous challenge set | **미생성** | A-C6 (P3 family 효과) |
| D_shared 판정 결과 | **미생성** | A-C4 (P1 shared 효과) |
| Family graph (topic-based Jaccard) | **미생성** | A-C6 (P3 family 효과) |
| Device prototype index | **미생성** | A-C5, A-C8 (Matryoshka router) |
| expected_device 라벨 (golden set) | **미추가** | A-C2~C7 (Cont@k 측정 전제) |
| 코퍼스 통계 | **완료** | evidence/2026-03-04_corpus_statistics.md |
| Cont-aware scoring 구현 (λ·v_scope) | **미구현** | A-C9 (P6 scoring 효과) |
| Router confidence → λ(q) 적응형 | **미구현** | A-C10 (P7 적응형 효과) |
| λ sensitivity sweep 데이터 | **미생성** | Fig-A4 (λ vs Cont/Hit trade-off) |

---

## Notes

- Cont@k 측정을 위해 golden set에 `expected_device` 라벨이 필수
- Family 그래프 구성 시 shared topic의 가중치 낮춤: `w(topic) = 1/log(1+device_count(topic))`
- 주 실험: retrieval-only (IR 메트릭 깔끔 비교), 보조: agent end-to-end
- Camera-ready 전에 `references.bib`의 placeholder 엔트리 정확한 메타데이터로 교체 필요
