# Paper D — Sensor-Document Linking Agent

## 주제

반도체 장비 센서 이상 데이터와 정비 문서(SOP, Setup Manual, 정비이력)를
자동으로 연결하여 근거 기반 트러블슈팅 답변을 생성하는 에이전트 구조.

## 핵심 질문

> 장비에서 수집되는 센서데이터에서 얻은 정보와 일반적인 setup-manual, SOP,
> 정비 데이터들이 있는 상황에서, 센서데이터와 문서들을 연결해서 대답하는
> agent를 학습하는 방법은?

## 연구 프레이밍

> 불완전한 시간 정렬을 가진 센서 스트림과 정비로그를 연결해
> 원인과 관련 사례를 검색·설명하는 근거기반 진단 프레임워크

## 디렉토리 구조

```
50_paper_d_sensor_doc/
├── README.md                              # 이 파일
├── paper_d_ideation.md                    # 아이디어 정리 + 4단계 파이프라인 설계
├── paper_d_training_roadmap.md            # 학습 전략 (A~D 단계)
├── paper_d_architecture.md                # APC 에이전트 상세 아키텍처 (RAAD-LLM 기반)
├── paper_d_professor_onepage_summary.md   # 교수님께 빠르게 설명하는 1페이지 요약
├── paper_d_research_proposal.md           # 박사과정 연구계획서 초안
├── paper_d_paper_strategy.md              # 논문 전략, 프레이밍, 빠른 실험 순서
├── paper_d_data_verification.md           # 데이터 검증 Agent 지시문 (3종)
├── paper_d_implementation_examples.md     # 3가지 원칙 구현 예시 (코드 스케치)
├── paper_d_es_query_results.md            # SUPRA Vplus 센서명 기반 ES 조회 결과 정리
├── paper_d_keyword_query_log.md           # 키워드별 조회 문서와 relevance 판정 작업장
└── evidence/
    ├── 2026-04-14_raad_llm_review.md              # RAAD-LLM 논문 상세 리뷰
    ├── 2026-04-14_literature_survey.md            # 6개 영역 26편 문헌 조사
    ├── 2026-04-14_related_literature_survey.md    # Paper D 포지셔닝 중심 문헌조사
    ├── paper_d_surveyed_references.md             # 나중에 빠르게 다시 볼 수 있는 참고용 논문 정리
    ├── paper_d_bibtex_priority.md                 # BibTeX를 먼저 확보해야 할 논문 목록
    └── paper_d_paper_comparison_table.md          # 논문별 차이를 한눈에 보는 비교표
```

## Status

### 완료
- [x] 아이디어 브레인스토밍 (2026-04-14)
- [x] 4단계 파이프라인 설계
- [x] 학습 로드맵 (A~D)
- [x] RAAD-LLM 논문 상세 리뷰
- [x] APC 에이전트 상세 아키텍처 설계
- [x] 교수님 보고용 1페이지 요약 작성
- [x] 박사과정 연구계획서 초안
- [x] 논문 전략 및 프레이밍 정리
- [x] 데이터 검증 Agent 지시문 작성
- [x] SUPRA Vplus 센서명 기반 ES 조회 결과 정리
- [x] 키워드별 조회 문서 및 relevance 판정용 로그 문서 작성
- [x] survey한 논문 reference 문서 작성
- [x] BibTeX 우선 수집 목록 작성
- [x] 논문 비교표 문서 작성

### 다음 단계
- [ ] Agent 지시문으로 데이터 검증 수행 (로그/센서/연결)
- [ ] 논문 가능성 판정 (7개 핵심 숫자 확인)
- [ ] Pilot set 50~100개 구축
- [ ] Failure family 선택 및 범위 확정
- [x] 관련 논문 문헌 조사 완료 — 6개 영역, Tier 1~3 총 21편+ (2026-04-14)
- [ ] Baseline 구현
- [ ] 1편 (데이터셋/문제정의) 논문 집필

## 논문 분할 계획 (3편)

| # | 제목 방향 | 내용 |
|---|----------|------|
| 1편 | Linking Sensor Episodes and Maintenance Logs under Temporal Uncertainty | 데이터셋, 문제정의, 정렬 프로토콜, baseline |
| 2편 | Temporal-Uncertainty-Aware Retrieval of Maintenance Cases | Retrieval 모델 (박사 중심 챕터) |
| 3편 | Grounded Failure Diagnosis using Retrieved Cases and SOP Evidence | Grounded diagnosis |
