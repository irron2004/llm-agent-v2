# Paper A 논문 타당성 검증 — 종합 리포트

**Session ID**: paper-a-validity-2026-03-07
**Date**: 2026-03-07
**Status**: Complete (5/5 stages + cross-validation)

---

## Executive Summary

Paper A "Hierarchy-aware Scope Routing for Cross-Equipment Contamination Control in Industrial Maintenance RAG"는 **조건부 투고 가능** 수준이다. 핵심 기여(contamination metric decomposition, scope-level-aware filtering)는 건전하나, 3개 FATAL 이슈와 5개 MAJOR 이슈를 해결해야 한다. 현재 상태로는 **CIKM Applied/Industry Short Paper**가 가장 현실적 목표이며, Full Paper는 평가셋 확장 + alias fix + P2-P4 결과가 필요하다.

---

## Stage Results Summary

### Stage 1: 방법론 건전성 — **75% 완성**

| 항목 | 판정 |
|------|------|
| 문제 정의 | **건전** — 명확, 측정 가능, 산업적 가치 충분 |
| 수학적 형식화 | **부분 건전** — S(q)가 Router mode 미포괄, in-scope 조건에 scope_level 제한 누락 |
| 시스템 매트릭스 | **건전** — 공정한 ablation, B4.5 추가 권장 |
| 메트릭 설계 | **건전** — Raw/Adj/Shared@k 수학적 일관성 확인 |
| Algorithm 1 | **부분 건전** — 4개 edge case 미명시 |

필수 수정 4건: S(q) 수식 확장, in-scope 조건 재형식화, Algorithm edge case, T=3 근거

### Stage 2: 실험 설계 타당성 — FATAL 2건

| 심각도 | 항목 |
|--------|------|
| **FATAL** | E-12: ZEDIUS XP alias mismatch → B4 결과 73% 왜곡 |
| **FATAL** | E-11: P1 adj_cont@5=0.000 tautology |
| MAJOR | E-02: 유효 표본 n=13 (9/22 전 시스템 실패) |
| MAJOR | E-08: gold labeling 프로토콜 미문서화 |
| MAJOR | E-14: reranker_model=null 불일치 |

### Stage 3: 주장-증거 정합성 — 과대주장 3건

| 주장 | 문제 |
|------|------|
| "-91% contamination 감소" | alias mismatch로 인과적 타당성 불명확 |
| "P1 adj_cont=0.000" | tautological — shared 재분류 효과 |
| "scope observability is key determinant" | cross-slice CI 없음, confound 미통제 |

정직한 판정: H3, H7, H8, H9, H10, H11, H12는 보수적이고 정합적.

### Stage 4: 참신성 — Short Paper 수준 성립

| 기여 | 참신성 |
|------|--------|
| Contamination metric decomposition | **HIGH** — 가장 명확한 기여 |
| Shared document classification | **HIGH** |
| Hierarchy-aware scope filtering | MEDIUM |
| Equipment family construction | MEDIUM (planned-not-reported) |

타겟: CIKM Applied/Industry 1차, SIGIR Industry 2차

### Stage 5: 데이터 품질 — alias 미해결이 최대 리스크

| 상태 | 항목 |
|------|------|
| **미해결** | device_aliases 40개 전부 비어있음 |
| **미해결** | ZEDIUS XP / SUPRA_XP 별개 엔트리 |
| 양호 | 6개 핵심 메트릭 구현 = spec 100% 일치 |
| 양호 | gold_doc_ids 형식 = ES doc_id 일치 |
| 주의 | device_family_seed "gold" 명명 오해 소지 |
| 주의 | gold_doc_ids 커버리지 71.8% (339/472) |

---

## Cross-Validation Results

### 모순 (Contradictions)
- **없음**. 5개 스테이지 간 상충하는 판단 미발견.

### 강화 연결 (Reinforcing connections)
1. Stage 2 E-12 (alias mismatch) + Stage 5 D-16 (catalog 미수정) + Stage 3 C-12 (인과 해석 불가) → **alias fix가 논문 전체의 blocking issue**
2. Stage 1 M-10 (메트릭 건전성) + Stage 4 N-03 (가장 명확한 기여) → **contamination metric이 Paper A의 핵심 selling point**
3. Stage 2 E-11 (P1 tautology) + Stage 3 C-13 (과대주장) → **P1 결과 보고 방식 전면 재설계 필요**

### 커버리지 갭
1. **Related work / 선행 연구 비교** — 어떤 스테이지도 구체적 선행 논문과의 체계적 비교를 수행하지 않음
2. **코드 실행 검증** — 정적 분석만 수행, 실제 재실행 검증 미수행
3. **Latency 비교** — B0 ~35ms vs B1 ~3000ms 차이가 있으나 평가 미수행

### 증거 품질
- LOW confidence: C-12 ("-91% 감소 인과성") — 이 finding이 결론에 큰 영향. alias fix 후 재실험으로만 해소 가능

---

## 최종 판정

### 투고 가능 조건 (Submission-ready conditions)

**필수 (FATAL 해소)**:
1. ZEDIUS XP alias fix → device_catalog aliases 채우기 → parser 수정 → B4/P1 재실험
2. P1 tautology 해소 → raw_cont@5 주 메트릭 병기 + shared doc relevance 메트릭 추가
3. 유효 n=13 투명 보고 → conditional metrics 추가

**권장 (MAJOR 완화)**:
4. gold labeling 프로토콜 문서화
5. reranker_model 구현 검증
6. Algorithm 1 edge case 명시
7. T=3 채택 근거 한 문장 추가
8. B4.5 (Hard+Shared only) 중간 ablation 추가

### 현실적 로드맵

| 단계 | 목표 | 예상 범위 |
|------|------|----------|
| Phase 1 | alias fix + 재실험 | FATAL 해소 |
| Phase 2 | 메트릭 보강 + 통계 강화 | MAJOR 완화 |
| Phase 3 | Short Paper 초고 작성 | CIKM Applied 투고 |
| Phase 4 | 평가셋 확장 (n≥100/slice) | Full Paper 준비 |
| Phase 5 | P2-P4 구현 + BSP-lite | Full Paper 또는 Paper A-BSP |

---

## BSP 의사결정 (PE 확정)

- **Paper A core**: scope policy + contamination metric 고정
- **BSP**: Paper A에서 제외, Paper A-BSP로 분리
- **Alias**: deterministic preprocessing (§3.2)
- **Family**: heuristic/provisional, 본문 메인 결과 제외
- **BSP 재개 조건**: multi-turn 50-100 session + alias 완료 + 파라미터 5-6개
