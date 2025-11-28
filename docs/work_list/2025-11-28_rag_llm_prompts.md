# RAG/LLM 통합 및 프롬프트 분리, 임베더 일관성 검증
- 날짜: 2025-11-28
- 담당자: hskim
- 역할: BE
- 관련 이슈/티켓: -
- 관련 브랜치/PR: main (로컬)
- 영역(Tags): [BE], [LLM], [RAG], [Retrieval], [Preprocessing], [Prompt]

## 1. 작업 목표(What & Why)
- vLLM 기반 LLM 레이어를 엔진-어댑터-레지스트리 패턴으로 추가하고, 서비스 계층(Chat/RAG)을 통해 end-to-end RAG 파이프라인을 구성.
- 멀티 에이전트 프롬프트를 YAML로 분리해 버전/역할별 관리 용이성 확보.
- 인덱싱/검색 시 임베더·전처리 일관성을 보장하고, 원문 컨텍스트/참조 정보가 LLM에 전달되도록 개선.
- 재로딩 시 임베더 불일치 및 전처리 미스매치를 드러내는 테스트 추가.

## 2. 최종 결과 요약(Outcome)
- LLM 인프라: Base/Registry + vLLM 엔진/어댑터 + ChatService 완료. LLM 호출을 레지스트리 기반으로 통합.
- RAGService: 전처리→검색→LLM 생성까지 단일 서비스로 묶고, 예제(`examples/rag_example.py`) 추가.
- 프롬프트: 라우터/MQ/게이트/답변 프롬프트를 `backend/llm_infrastructure/llm/prompts/` 하위 YAML로 분리.
- 인덱싱/검색 일관성: `IndexedCorpus`에 embedder 저장, `SearchService`에서 재사용. `StoredDocument`에 raw_text 추가, 컨텍스트에 doc_id/원문 포함.
- 검증 테스트: 인덱스 저장→로드 후 embedder/preprocessor 일관성 여부를 확인하는 pytest 추가.

## 3. 작업 과정(Timeline/Steps)
1) vLLM LLM 계층 구현: `llm/base.py`, `llm/registry.py`, `engines/vllm.py`, `adapters/vllm.py`, `ChatService` 추가.  
2) RAGService 작성: 전처리→검색→LLM 생성 파이프라인과 예제 코드(`examples/rag_example.py`) 구성.  
3) 프롬프트 분리: ragflow 멀티에이전트 JSON을 분석해 router/MQ/gate/answer 프롬프트를 YAML 파일로 이동.  
4) 검색 일관성 개선: `IndexedCorpus.embedder` 저장, `SearchService`에서 재사용; `StoredDocument.raw_text`/doc_id를 컨텍스트에 포함.  
5) 테스트 추가: 인덱스 저장 후 재로딩 시 embedder 동일성, 전처리 미스매치 사례를 검증하는 `test_rag_service_reload.py` 작성.  
6) 문서화: work_list 업데이트 및 상세 일지 작성.

## 4. 추가/수정한 테스트(Tests)
- backend/tests/test_rag_service_reload.py (새 테스트)  
  - reloaded corpus가 동일 embedder로 검색될 때 정상 동작하는지 검증  
  - 다른 preprocessor를 주입했을 때 메타데이터가 달라지는지 확인

## 5. 설계 및 의사결정(Design & Decisions)
- LLM 패턴: 다른 인프라(전처리/임베딩/검색)와 동일한 엔진-어댑터-레지스트리 패턴을 유지해 교체/확장이 용이하도록 설계.  
- 프롬프트 관리: 에이전트별 YAML 분리(라우터, MQ, 게이트, 답변)로 버전 관리/실험 분리 용이성을 확보.  
- 일관성 유지: 인덱싱 시 사용한 embedder와 전처리를 corpus 메타에 담고, 검색/RAG에서 재사용해 차원 불일치·매칭 오류를 예방.  
- 컨텍스트 품질: raw_text + doc_id를 LLM 컨텍스트에 포함시켜 가독성과 인용 가능성을 높임.

## 6. 회고 및 다음 할 일(Retrospective / Next Steps)
- 테스트 실행: 로컬에서 pytest 전체 실행 시 uv 캐시 권한 이슈가 간헐적으로 발생하므로 캐시 정리 후 재실행 필요.  
- API 계층: RAGService를 FastAPI 라우터에 연결하고, corpus 로딩을 앱 스타트업 훅에 포함시키는 작업이 남음.  
- RAGFlow 마이그레이션: ragflow retriever/클라이언트를 별도 모듈로 이전해 선택적 사용 가능하게 할 계획.  
