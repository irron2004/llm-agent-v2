# 2025-12-04 LLM 공용 환경 정리 & VLM 설정 중립화

## 작업 개요
- LLM 모델/데이터 캐시를 `/home/llm-share` 하위에서 공용으로 쓰도록 `.env.llm` 템플릿과 README 가이드를 추가했습니다.
- 기존 `DeepSeekSettings`를 벤더 중립형 `VlmParserSettings`로 바꾸고, `VLM_PARSER_*`/`DEEPSEEK_*` 이중 alias를 지원해 설정을 확장했습니다.
- SentenceTransformer 엔진이 추상 베이스 구현을 충족하도록 `embed`/`embed_batch` 메서드를 보완해 레지스트리 초기화 오류를 제거했습니다.

## 상세 변경
1. **공용 .env 구성**
   - `/home/llm-share/.env.llm`에 `LLM_SHARED_ROOT`, `HF_HOME`, `HF_DATASETS_CACHE`를 정의하고 프로젝트에서 먼저 로드하도록 README에 예시를 추가.
   - Docker Compose 예시에서 공용 .env와 프로젝트 `.env`를 동시에 사용하는 방법 문서화.
2. **VLM 설정 리팩토링**
   - `DeepSeekSettings` → `VlmParserSettings` 이름/설명 변경, `AliasChoices`로 `VLM_PARSER_*`와 기존 `DEEPSEEK_*` 환경 변수를 모두 허용.
   - Ingest 서비스가 새 `vlm_parser_settings`를 사용하게 수정하여 기본값/다운로드 설정을 일관되게 관리.
3. **SentenceTransformer 엔진 보완**
   - `SentenceTransformerEmbedder`에 `embed`/`embed_batch`를 추가해 `BaseEmbedder` 추상 메서드를 구현, registry 초기화 시 `TypeError` 해결.
4. **E2E 검증**
    - `main` 브랜치 기반 E2E 테스트를 병행 실행해 통합 동작을 수동 검증.

## 참고/후속
- `/home/llm-share` 하위 권한(예: 1777)과 캐시 경로(`HF_HOME=/home/llm-share/hf`)를 다른 사용자와 공유하도록 운영 가이드 필요.
- `VlmPdfEngine`이 DeepSeek 외 모델(Qwen-VL 등)을 선택할 수 있도록 `model_id`/`prompt`를 문서화하는 README 업데이트 검토.
