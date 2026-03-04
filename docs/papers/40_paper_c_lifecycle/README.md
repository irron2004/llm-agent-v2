# Paper C — Lifecycle Reliability Control

## Goal
- 논문 주제: 업데이트/드리프트/롤백 기반 운영 신뢰성 통제
- 핵심 문제: 문서/인덱스 업데이트 이후 성능 회귀와 안정성 붕괴
- 목표: 리스크-비용 최소화 정책(업데이트/롤백)을 설계하고 효과를 정량 검증

## Research Question
- RQ-C: 드리프트 감지 + 회귀 테스트 + 롤백 정책이 운영 리스크를 유의하게 줄이는가?

## Main Metrics
- `Delta_Recall@k` (버전 전후 변화)
- `Delta_Stability@k` (버전 전후 변화)
- `Regression_detection_rate`
- `Rollback_success_rate`
- `Risk-Cost` (정책별 기대 리스크/운영 비용)

## Required Inputs
- Versioned corpus: `v1 -> v2 -> v3` 이상
- Regression suite: 고정 질의군(최소 50~150)
- 정책 후보:
  - Always update
  - Scheduled update
  - Drift-triggered update/rollback

공통 규격 참조:
- `../10_common_protocol/paper_common_protocol.md`

## Expected Outputs
- 정책 비교표:
  - Risk 감소량
  - 운영 비용
  - 회귀 탐지율/롤백 성능
- 드리프트 시각화:
  - 버전별 지표 추세
  - 경보/트리거 타임라인
- 운영 권고안:
  - 어떤 조건에서 어떤 정책을 선택할지

## Immediate Tasks
- [ ] 업데이트 이벤트 시나리오 3종 이상 정의
- [ ] regression suite 고정 및 자동 리포트 연결
- [ ] drift score 정의(Delta Recall, Delta Stability, 위험 질의군 가중)
- [ ] 정책 3종 비교 실험 수행
- [ ] 원고용 메인 테이블/정책 의사결정 그림 확정

## Draft/Assets Convention
- 초안 문서: `paper_c_lifecycle.md`
- 실험 스펙: `paper_c_lifecycle_spec.md`
- 자산 폴더: `assets/`

## Literature Assets
- related work: `related_work.md`
- references: `references.bib`
- evidence mapping: `evidence_mapping.md`
