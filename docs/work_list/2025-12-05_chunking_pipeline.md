# Chunking 파이프라인 검증 (SU-2504)
- 날짜: 2025-12-05
- 담당자: hskim
- 역할: BE
- 관련 이슈/티켓: SU-2504 (ClickUp)
- 관련 브랜치/PR: https://github.com/teamRTM/llm-agent-v2/tree/feature/SU-2504-chunking
- 영역(Tags): [BE], [RAG], [Chunking]

## 1. 작업 목표(What & Why)
- DB 구축 파이프라인에 Chunking 단계를 명확히 포함하고 파라미터를 검증
- FixedSizeChunker 설정을 점검하고 문서/설정 간 싱크를 맞춤
- 향후 멀티 쿼리/리랭크 흐름 대비하여 chunking → embedding → indexing 경로를 확립

## 2. 최종 결과 요약(Outcome)
- Chunking 인프라(Base/Registry/FixedSize) 작성 및 테스트 커버리지 추가
- DocumentIndexService에 chunk metadata/offset 저장 로직 포함
- ClickUp 정책에 맞춰 브랜치(`feature/SU-2504-chunking`) 생성 및 공유

## 3. 작업 과정(Timeline/Steps)
1. Chunking 베이스/레지스트리 초안 작성
2. FixedSizeChunker 구현(문장/토큰 기준, overlap 지원)
3. DocumentIndexService에 chunk 반영 및 저장 메타데이터 확장
4. 테스트(`backend/tests/test_chunking.py`) 작성
5. ClickUp SU-2504에 브랜치 정보 공유

## 4. 추가/수정한 테스트(Tests)
- `backend/tests/test_chunking.py`

## 5. 설계 및 의사결정(Design & Decisions)
- 초기 단계에서는 FixedSizeChunker 하나만 제공하되 레지스트리 구조로 확장성을 확보
- raw_text 보존을 위해 chunk offset 기반 슬라이싱 사용 (전처리 차이에 따른 편차는 이후 개선 예정)
- 브랜치 네이밍/정책은 ClickUp ID 기준을 유지

## 6. 회고 및 다음 할 일(Retrospective / Next Steps)
- 전처리와 raw_text 길이 차이가 큰 경우 offset 오차가 발생할 수 있어 보정 로직 필요
- Chunking 파라미터 튜닝(길이/겹침)과 Multi-Query/Rerank 통합 흐름 문서화 예정
