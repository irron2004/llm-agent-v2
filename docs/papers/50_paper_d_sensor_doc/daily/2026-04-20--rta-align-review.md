---
date: 2026-04-20
type: daily-log
topics: [rta-align, scope-review, methodology-critique, agent-comparison]
status: completed
---

# 2026-04-20 — RTA-Align 제안 검토 및 scope 조정

> **Note:** 이 문서는 중간 검토 기록이다. 최종 1편 논문 방향과 scope freeze는
> [[./2026-04-20--final-strategy-confirmed|2026-04-20--final-strategy-confirmed]]를 기준으로 본다.
> 본 문서의 일부 판단(특히 1편/2편 분리선)은 이후 최종 결정에서 수정되었다.

## 오늘의 목표
1. [x] 다른 agent(Hephaestus)가 제안한 RTA-Align 프레임워크 검토
2. [x] 기존 Paper D 합의와의 충돌점 파악
3. [x] 가져갈 것 / 버릴 것 정리
4. [x] 현실적 1편 논문 scope 재확인

## 완료한 것

### 1. RTA-Align 제안 검토

다른 agent가 제안한 "RTA-Align (Rule-grounded Time-series to Actionable-text Alignment)" 전체 구조를 검토함.

**제안 내용 요약**:
- 4블록 아키텍처: Sensor Event Extractor → Event Encoder → Text Encoder → Cross-modal Alignment
- 4개 Loss: L_align + L_anom + L_cause + L_device
- Detection + Grounding 동시 수행
- 22개 baseline 비교
- 3편 논문 구조

### 2. 기존 합의와의 충돌 분석

| 항목 | 기존 합의 | RTA-Align 제안 | 판단 |
|------|----------|---------------|------|
| 핵심 문제 | retrieval 중심 | detection + grounding 동시 | ⚠️ scope 확대 |
| Loss | L_retrieval + L_cause (2개) | 4개 동시 | ⚠️ 구현 복잡 |
| Baseline | 5~7개 | 22개 | ⚠️ 과다 |
| 논문 수 | 1편 빨리 | 처음부터 3편 | ⚠️ 느림 |
| Anomaly detection | 안 함 (전제로 둠) | 직접 수행 | ⚠️ scope 확대 |

### 3. 판단: 좋은 부분만 가져가기

**가져갈 것**:
1. **Hierarchy-aware hard negative mining** — 같은 family 다른 cause, shared SOP 등
2. **Contamination 평가 지표** — wrong-device@k, wrong-module@k
3. **Text-grounded anomaly 개념** — retrieval 결과가 anomaly 해석을 제공
4. **문제 정의의 4중 출력** (anomaly score + cause ranking + evidence + action) — 평가 프레임

**버릴 것 / 뒤로 미룰 것**:
1. Full 4-block architecture → 1편에서는 Eventizer + Retriever면 충분
2. Loss 4개 동시 학습 → L_align + L_anom으로 시작
3. 22개 baseline → 5~7개로 축소
4. "Rule"을 논문 제목에 넣기 → rule engineering 논문처럼 보일 위험
5. RTA-Align이라는 이름 → 최종 결정 보류

### 4. 두 agent 간 용어 차이 정리

이전에 "graph 기반이냐?" 논쟁이 있었던 것도 정리함.

| Claude (나) | Hephaestus (다른 agent) | 실제 |
|-------------|----------------------|------|
| "graph가 아니라 structured retrieval" | "graph-based state representation" | **같은 말**, 용어 정밀도 차이 |
| "LLM은 해석기" | "LLM은 해석기" | **동일** |
| "수치 기반 시스템이 retrieval" | "수치 기반 시스템이 retrieval" | **동일** |

핵심 합의: **수치 기반 엔진이 retrieval하고, LLM은 결과를 해석/정리한다**.

## 핵심 발견/아이디어

> **RTA-Align의 가장 강한 기여인 "hierarchy-aware contamination reduction"은 반도체 도메인에서 실제로 중요한 문제이고, 2편 논문의 핵심 기여로 발전시킬 수 있다.**

> **하지만 1편은 이전 합의대로 "temporal uncertainty-aware retrieval + pilot set 검증"으로 빨리 내는 것이 맞다.**

## 장애물/문제

1. **scope creep 위험**: 여러 agent와 대화하면서 방향이 조금씩 넓어지고 있음
   - 해결: 1편 scope를 명시적으로 고정하고, 추가 아이디어는 2편/3편으로 분리

2. **시계열 데이터 미확보**: 여전히 pilot set 구축을 시작하지 못함
   - 해결: 이것이 가장 급한 blocker

## 다음 단계

### 최우선: 시계열 데이터 확보 (blocker)
- [ ] APC_Position, APC_Pressure 시계열 데이터 위치 확인
- [ ] 샘플 데이터 로드 테스트
- [ ] episode 추출 가능 여부 확인

### 1편 논문 scope (고정)
```
Eventizer (규칙 기반)
+ Temporal Uncertainty-Aware Alignment
+ Maintenance Case Retrieval (BM25 + dense)
+ Document Retrieval
+ Gold/Silver/Weak 라벨링
+ Baseline 비교 (5~7개)
= 1편 논문
```

### 2편으로 미룬 것
- Hierarchy-aware loss
- Contamination reduction
- Cross-modal contrastive learning
- Full encoder 학습

## 현재 Paper D 문서 상태 요약

| 구분 | 파일 수 | 총 줄 수 | 상태 |
|------|------:|--------:|------|
| 기획/전략 | 8 | ~2,200 | ✅ 충분 |
| 데이터 검증 | 4 | ~1,000 | ✅ 1차 완료 |
| 문헌 | 6 | ~1,300 | ✅ 충분 |
| 시각화 | 2 | ~900 | ✅ |
| daily log | 3 | ~300 | 🔄 진행 중 |
| **합계** | **~23** | **~5,700** | 기획 충분, 실행 필요 |

## 관련 문서

- `paper_d_algorithm_design.md` — 6모듈 방법론 (이전 합의)
- `paper_d_map_and_graph_framing.md` — graph는 Eventizer 강화층
- `paper_d_interim_report.md` — 중간 보고서
- `paper_d_progress_memo.md` — 진행 상태 메모

---
**Written**: 2026-04-20
**Status**: Review completed. Next blocker: 시계열 데이터 확보
