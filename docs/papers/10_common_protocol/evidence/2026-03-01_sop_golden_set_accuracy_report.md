# SOP 질문 리스트 정답률 평가 보고서

> 작성일: 2026-03-01
> 데이터: `data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv` (79건)
> 평가 방법: ES 직접 검색 (BM25, `rag_chunks_dev_v2` 인덱스, doc_type=SOP 필터)
> 비교 기준: `정답문서` + `정답 페이지` 매칭
>
> 관련 파일:
> - alias: `2026-03-01_sop_questionlist_accuracy_and_tuning_report.md`
> - 요약본: `2026-03-01_sop_questionlist_accuracy_summary.md`
> - 행별 상세 데이터: `docs/evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv`

---

## 1. 정답률 요약 (페이지 포함 기준)

`정답`은 **정답 문서 + 정답 페이지 매칭(page_hit)**으로 판정한다.

### 1.1 페이지 포함 정답률 (메인 지표)

| 지표 | 결과 | 비율 |
|---|---|---|
| **page-hit@1** | **53/79** | **67.1%** |
| **page-hit@3** | **57/79** | **72.2%** |
| **page-hit@5** | **58/79** | **73.4%** |
| **page-hit@10** | **61/79** | **77.2%** |
| **page-hit@20** | **61/79** | **77.2%** |

### 1.2 문서만 기준 (보조 지표)

| 지표 | 결과 | 비율 |
|---|---|---|
| **hit@5** (top-5에 정답 문서 포함) | **76/79** | **96.2%** |
| **hit@10** | **79/79** | **100.0%** |
| **hit@20** | **79/79** | **100.0%** |
| **rank=1** (정답이 1위) | **71/79** | **89.9%** |
| **정답 페이지 포함** | **61/79** | **77.2%** |

### 매칭 순위 분포

| 순위 | 건수 |
|---|---|
| rank 1 | 71건 (89.9%) |
| rank 2-3 | 4건 |
| rank 4-5 | 1건 |
| rank 6-10 | 3건 |
| rank 11+ | 0건 |

---

## 2. 현재 운영 서버 문제 (최우선 조치)

**현재 서버(18001, 18011)가 합성 인덱스를 사용 중:**

```
현재:   SEARCH_ES_INDEX_PREFIX=rag_synth  SEARCH_ES_ENV=synth   → rag_synth_synth_v2 (120건)
필요:   SEARCH_ES_INDEX_PREFIX=rag_chunks  SEARCH_ES_ENV=dev    → rag_chunks_dev_v2 (345,935건)
```

이 설정만 변경하면 SOP 11,452건 + SUPRA XP 18,657건 데이터에 접근 가능합니다.

---

## 3. Top-5 밖 케이스 분석 (3건)

### [Q21] PRISM 3100 SOURCE 교체 — rank 6

| 항목 | 값 |
|---|---|
| 질문 | `ZEDIUS XP 설비의 "PRISM 3100" SOURCE 교체 관련 작업` |
| 정답 문서 | `global sop_supra xp_all_pm_cip chamber.pdf` (p9-37) |
| Top-5에 나온 문서 | `prism_source_3100qc.pdf` (rank 1-3), `prism_source_3000qc.pdf` (rank 4-5) |

**원인**: PRISM SOURCE 전용 문서(`prism_source_3100qc.pdf`)가 BM25에서 더 높은 점수. 정답은 CIP Chamber PM 문서 내 PRISM 3100 섹션.
**개선안**: PRISM SOURCE 교체 시 CIP Chamber 문서도 함께 제공하도록 관련 문서 그룹핑 또는 reranker 적용.

### [Q41] PM Pirani Gauge Adjust — rank 7

| 항목 | 값 |
|---|---|
| 질문 | `ZEDIUS XP 설비의 PM Pirani Gauge Adjust 작업` |
| 정답 문서 | `global sop_supra xp_all_pm_pressure gauge.pdf` (p18-34) |
| Top-5에 나온 문서 | `pm_pirani_gauge.pdf`, `ll_pressure_gauge.pdf`, `tm_pressure_gauge.pdf` |

**원인**: "Pirani Gauge"라는 키워드가 전용 문서(`pirani_gauge.pdf`)에 더 많이 등장. 정답 문서는 "Pressure Gauge" 통합 문서 내 Pirani 섹션.
**개선안**: 동일 장비 내 유사 주제 문서 교차 참조 제공. 또는 `chapter` 필드 활용 re-ranking.

### [Q42] PRISM SOURCE 교체 (preventive maintenance) — rank 10

| 항목 | 값 |
|---|---|
| 질문 | `ZEDIUS XP 설비의 "PRISM" SOURCE 교체 관련 작업` |
| 정답 문서 | `global sop_supra xp_all_pm_preventive maintenance.pdf` (p10-39) |
| Top-5에 나온 문서 | `prism_source.pdf`, `prism_source_3000qc.pdf`, `prism_source_3100qc.pdf` |

**원인**: PRISM SOURCE 전용 문서 3종이 모두 더 높은 BM25 점수. 정답은 PM(예방정비) 통합 문서 내 PRISM 섹션.
**개선안**: 질문에 "preventive maintenance" 키워드가 없어서 BM25가 구분 불가 → MQ 생성 또는 reranker가 PM 문서를 올려줘야 함.

---

## 4. 정답 페이지 미포함 분석 (18건)

18건 모두 **`matched_page=1` (목차/표지 페이지)**가 최상위 매칭.

### 해당 질문 목록

| Q# | 질문 (축약) | 정답 문서 (축약) | 기대 페이지 | 매칭 페이지 |
|---|---|---|---|---|
| 1 | EFEM PIO SENSOR BOARD 교체 | efem_pio sensor board | 6-14 | 1 |
| 5 | EFEM PRESSURE SWITCH 교체 | efem_pressure switch | 6-18 | 1 |
| 14 | LL Flow Switch Replacement | ll_flow switch | 6-18 | 1 |
| 18 | PM Baratron Gauge ADJ | pm_baratron gauge | 6-18 | 1 |
| 19 | PM Baratron Gauge REP | pm_baratron gauge | 19-33 | 1 |
| 28 | PM HEATER CHUCK Replacement | pm_heater chuck | 6-27 | 1 |
| 38 | PM Pirani Gauge ADJ | pm_pirani gauge | 6-21 | 1 |
| 45 | PRISM SOURCE 교체 (3000QC) | pm_prism source 3000qc | 8-37 | 1 |
| 47 | PRISM 3000QC RF GEN Calibration | pm_prism source 3000qc | 54-70 | 1 |
| 49 | PRISM SOURCE 교체 (3100QC) | pm_prism source 3100qc | 8-39 | 1 |
| 51 | PRISM 3100QC RF GEN Calibration | pm_prism source 3100qc | 56-72 | 1 |
| 52 | PRISM SOURCE 교체 | pm_prism source | 9-39 | 1 |
| 56 | SOURCE BOX BOARD 교체 | pm_source box board | 6-15 | 1 |
| 59 | Water Shut Off Valve Replacement | pm_wafer shut off valve | 7-19 | 1 |
| 61 | GAS BOX BOARD 교체 | sub unit_gas box board | 6-18 | 1 |
| 67 | TM DC Power SUPPLY 교체 | tm_dc power supply | 6-18 | 1 |
| 70 | Baratron Gauge Rep/Adj (TM) | tm_pressure gauge | 7-19 | 1 |
| 71 | Pirani Gauge Rep/Adj (TM) | tm_pressure gauge | 20-31 | 1 |

### 원인

SOP 문서 구조:
```
page 1:  문서 목차 / 표지 (모든 키워드 포함)
page 2-5: 목적, 범위, 관련 문서, 공구 목록
page 6+:  실제 작업 절차 (정답 페이지)
```

page 1 chunk가 문서의 모든 핵심 키워드를 포함하므로 BM25 점수가 가장 높음. 실제 작업 절차(page 6+)는 구체적 단계/수치만 포함하여 키워드 겹침이 낮음.

### 영향

- 문서 자체는 정확히 찾았으므로 (hit@5 = 96.2%) 사용자에게 올바른 문서 제공은 가능
- 그러나 정확한 페이지(작업 절차 시작점)를 안내하지 못함

### 개선 방안

| 방안 | 효과 | 구현 난이도 |
|---|---|---|
| **A. 목차 chunk boosting 감소** | page 1-2의 BM25 점수를 낮춤 | 낮음 (인덱싱 시 가중치 조정) |
| **B. 동일 문서 내 다중 페이지 반환** | 정답 문서의 여러 chunk를 반환 | 낮음 (검색 후 필터) |
| **C. chunk에 작업 유형 메타 부착** | "교체"/"조정"/"Calibration" 등 chunk 메타 | 중간 (재인덱싱) |
| **D. Reranker 적용** | cross-encoder가 질문-절차 매칭 개선 | 중간 (모델 추가) |

**추천: B → A → D 순서**
- B (동일 문서 다중 페이지)는 즉시 적용 가능하고 효과가 큼
- 현재 top-20에 동일 문서의 여러 페이지가 이미 포함되어 있으므로, 정답 문서의 모든 chunk를 모아서 페이지 범위를 표시하면 해결

---

## 5. 전체 개선 로드맵

### 즉시 (P0)

| 항목 | 조치 | 기대 효과 |
|---|---|---|
| **서버 인덱스 전환** | env: `rag_synth` → `rag_chunks`, `synth` → `dev` | 실제 SOP 데이터 접근 |
| **동일 문서 다중 페이지 집계** | 검색 결과에서 동일 doc_id의 모든 chunk를 모아 페이지 범위 표시 | 정답 페이지 포함율 77% → ~95% |

### 단기 (P1)

| 항목 | 조치 | 기대 효과 |
|---|---|---|
| **목차 chunk 가중치 조정** | page=1 chunk에 score penalty 적용 | 실제 작업 절차 chunk가 상위로 |
| **Reranker 활성화** | cross-encoder 기반 re-ranking | Top-5 밖 3건 → Top-5 내로 |
| **mq_mode=fallback 배포** | 개선 코드 배포 + 인덱스 전환 | 반복 안정성 확보 + 정답률 유지 |

### 중기 (P2)

| 항목 | 조치 | 기대 효과 |
|---|---|---|
| **chunk 메타 강화** | 작업 유형(교체/조정/Calibration) 메타 부착 | 유사 문서 간 구분력 향상 |
| **PRISM SOURCE 문서 그룹핑** | 동일 주제 4개 문서 교차 참조 | Q21/Q42 해결 |

---

## 6. 결론

**페이지 포함 기준(page-hit@10)은 77.2%(61/79)이며, page-hit@5는 73.4%(58/79)다.**

문서만 기준 hit@10은 100%(79/79)로 높지만, 실제 운영에서 중요한 것은 페이지 정확도이므로 page-hit 계열을 메인 KPI로 관리해야 한다.

**보조 지표(문서만 기준):**

```
hit@5  = 96.2% (76/79)
hit@10 = 100%  (79/79)
rank 1 = 89.9% (71/79)
```

**핵심 병목은 검색 정확도가 아니라 운영 설정과 페이지 정밀도:**

1. **서버가 합성 인덱스를 사용 중** — 실제 SOP 인덱스로 전환하면 즉시 작동
2. **정답 페이지 미스 18건** — 목차 chunk가 최상위로 올라오는 구조적 문제 → 동일 문서 다중 페이지 집계로 해결 가능
3. **Top-5 밖 3건** — 유사 문서명 경합 (PRISM SOURCE 4종, Pirani/Pressure Gauge) → reranker 또는 문서 그룹핑으로 해결 가능

**검색 엔진 자체는 정상 동작하며, 운영 설정 변경만으로 사용 가능한 상태.**
