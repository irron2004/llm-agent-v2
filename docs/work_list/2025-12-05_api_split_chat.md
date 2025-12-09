# 2025-12-05: Chat API 분리 및 프롬프트 주입 지원

## 작업 개요
- `/api/chat` 엔드포인트를 LLM-only(`simple`)와 RAG 기반(`retrieval`)으로 분리
- Simple 챗에 시스템 프롬프트 파일 주입 기능 추가 (환경변수로 경로 지정)
- 테스트/DI 오버라이드 갱신, 모든 API 테스트 통과

## 주요 변경 사항
- `backend/api/routers/chat.py`
  - `/api/chat/simple`: LLM 단독 답변, 프롬프트 파일을 system prompt로 사용, 검색 결과는 빈 리스트
  - `/api/chat/retrieval`: RAG 기반 답변 (코퍼스 연결 시 동작), 기존 스키마 유지
- `backend/api/dependencies.py`
  - `get_chat_service` 유지, `get_simple_chat_prompt` 추가 (파일 읽어 system prompt 제공)
- `backend/config/settings.py`
  - `API_SIMPLE_CHAT_PROMPT_FILE` 설정 추가 (`simple_chat_prompt_file` 필드)
- 테스트
  - `tests/api/conftest.py`, `tests/api/test_chat_api.py` 업데이트
  - `.venv/bin/pytest tests/api -q` → 7 passed

## 실행/사용법
- 프롬프트 파일 지정: `.env`에 `API_SIMPLE_CHAT_PROMPT_FILE=path/to/prompt.txt`
- LLM-only 호출: `POST /api/chat/simple` (body: `{"message": "..."}`)
- RAG 호출: `POST /api/chat/retrieval` (코퍼스 미연결 시 503 발생할 수 있음)

## TODO / 다음 단계
- 실제 코퍼스/벡터DB 결정 후 `get_search_service`/`get_rag_service`에 인덱스 로딩 로직 연결
- RAG 응답에 citation/메타데이터 확장 여부 검토
