# Corpus Statistics for Paper A — Scope Routing

> Source: `data/chunk_v3_normalize_table.md` (파싱 문서 578건)
> Date: 2026-03-04

## 1. 기본 통계

| 항목 | 값 |
|------|-----|
| 총 문서 수 | 578 |
| 고유 장비(device) 수 | 21 |
| 주요 장비(50건 이상) | 6 (SUPRA_VPLUS, INTEGER_PLUS, SUPRA_N, PRECIA, GENEVA_XP, SUPRA_XP) |
| 고유 토픽 수 | 418 |
| 2+ 장비 공유 토픽 | 54 (12.9%) |
| 3+ 장비 공유 토픽 | 13 |

## 2. 장비별 문서 분포

| 장비명 | 문서 수 | 비율 |
|--------|---------|------|
| SUPRA_VPLUS | 176 | 30.4% |
| INTEGER_PLUS | 89 | 15.4% |
| SUPRA_N | 85 | 14.7% |
| PRECIA | 71 | 12.3% |
| GENEVA_XP | 64 | 11.1% |
| SUPRA_XP | 59 | 10.2% |
| OMNIS_PLUS | 13 | 2.2% |
| 기타 (14종) | 21 | 3.6% |

## 3. 장비 Family 분류

| Family | 문서 수 | 비율 | 장비 수 | 구성원 |
|--------|---------|------|---------|--------|
| SUPRA | 329 | 56.9% | 9 | VPLUS(176), N(85), XP(59), 기본(3), NM(2), V(1), NP(1), VM(1), XQ(1) |
| INTEGER | 89 | 15.4% | 1 | PLUS(89) |
| PRECIA | 71 | 12.3% | 1 | PRECIA(71) |
| GENEVA | 69 | 11.9% | 3 | XP(64), 기본(4), STP300_XP(1) |
| OMNIS | 15 | 2.6% | 2 | PLUS(13), 기본(2) |
| ETC | 5 | 0.9% | 5 | ALL(1), ECOLITE_2000(1), ECOLITE_3000(1), ECOLITE_II_400(1), ZEDIUS_XP(1) |

## 4. 문서 유형 분포

| Section | 문서 수 | 비율 |
|---------|---------|------|
| sop_pdf | 335 | 57.9% |
| sop_pptx | 172 | 29.8% |
| ts | 57 | 9.9% |
| setup_manual | 14 | 2.4% |

## 5. 공유 토픽 분석 (Top 10)

| 토픽 | 공유 장비 수 | 장비 목록 |
|------|-------------|-----------|
| ffu | 5 | GENEVA_XP, INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_XP |
| controller | 5 | INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_VPLUS, SUPRA_XP |
| mfc | 4 | GENEVA_XP, INTEGER_PLUS, PRECIA, SUPRA_XP |
| device net board | 4 | INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_XP |
| robot | 4 | INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_XP |
| heater chuck | 3 | INTEGER_PLUS, SUPRA_N, SUPRA_XP |
| gas spring | 3 | INTEGER_PLUS, PRECIA, SUPRA_N |
| slot valve | 3 | INTEGER_PLUS, PRECIA, SUPRA_XP |
| solenoid valve | 3 | INTEGER_PLUS, PRECIA, SUPRA_N |
| sensor board | 3 | INTEGER_PLUS, PRECIA, SUPRA_N |

## 6. Paper A 시사점

### Family 구성 타당성
- SUPRA family 내 9개 장비가 공유 토픽 다수 → family 확장 정책으로 intra-family recall 보존 가능
- 동일 토픽이 장비별 별도 doc_id → **doc_id 공유가 아닌 topic 기반 유사도**로 family 구축 필요

### Shared Document 정책
- 54개 토픽(12.9%)이 2+ 장비에 걸침 → `D_shared` 분류 정당화
- `T(shared) = 3` 기준 시 13개 토픽이 shared로 분류 (controller, ffu, mfc, robot 등)
- 이 토픽들은 공용 컴포넌트(FFU, MFC, 로봇 등)에 해당 → 도메인 합리성 확보

### 라우터 설계 고려
- 21개 장비지만 주요 장비 6개가 전체의 96.4% 차지
- 장비 수가 적어 Top-M 라우터의 선택성이 제한적
- **Family-level routing** (5 families → Top-2) 이 대안으로 고려 가능

### Contamination 위험 프로파일
- SUPRA family가 56.9%로 과반 → 다른 장비 질의 시 SUPRA 문서 혼입 가능성 높음
- Cross-family contamination (예: INTEGER↔PRECIA 공유 토픽 다수)이 주요 위험
- 공유 토픽(ffu, controller 등)에서의 contamination은 "false positive contamination" → Adjusted 메트릭 필요

### 평가셋 설계
- Explicit set: SOP79 기반 (장비명 명시 질의)
- Mask set: 주요 6개 장비 × 주요 토픽에서 장비명 마스킹
- Ambiguous set: 공유 토픽(ffu, controller 등) 질의 → 라우터 성능의 핵심 테스트
