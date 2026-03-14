# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mandatory Startup Protocol

Before any non-trivial edit:
1. Read `docs/2026-03-14-agent-개발-운영.md` section `0) 신규 agent 5분 온보딩 (필독)` and section `5.5 비사소 작업 판정 기준`.
2. Run the kickoff checklist from `docs/agent-skills/task-start-kickoff.md`.
3. If local skills are available, the Claude wrapper is `.claude/skills/task-start-kickoff/SKILL.md`.
4. Read `docs/contracts/product-contract.md` and identify the contract IDs to preserve.
5. If the task is non-trivial, create or update a task doc from `docs/tasks/TASK_TEMPLATE.md` before editing code.

If auto skill loading is unavailable, perform the same startup sequence manually from the canonical checklist.

## Project Overview

PE(Process Engineering) 트러블슈팅을 위한 RAG 기반 에이전트 시스템. 반도체 장비 문서(SOP, PEMS, 매뉴얼 등)를 검색·분석하여 답변을 생성한다.

## Commands

### Backend
```bash
# 로컬 API 서버 (uvicorn, hot-reload)
make run-api          # 백그라운드 시작 (port 8001)
make stop-api         # 정지

# 테스트
cd backend && uv run pytest tests/ -v
uv run pytest backend/tests/test_specific.py -v -k "test_name"

# 린트/포맷
uv run ruff check backend/
uv run ruff format backend/
```

### Frontend
```bash
cd frontend
npm run dev           # Vite dev server (port 9097)
npm run build         # Production build
npm run test          # Vitest
npm run test:watch    # Watch mode
```

### Docker
```bash
make up               # api + elasticsearch (외부 vLLM 사용)
make prod-up          # prod profile (.env + .env.prod)
make dev-up           # dev profile (.env + .env.dev)
make up-vllm          # vLLM 포함
make logs             # 로그 tail
make logs-api         # API 로그만
```

## Architecture

### Engine-Adapter-Registry Pattern

모든 핵심 모듈(preprocessing, embedding, retrieval, llm, reranking)이 동일한 3계층 패턴을 따른다:

- **Engine** (`engines/`): 순수 알고리즘 구현. 외부 의존 최소화.
- **Adapter** (`adapters/`): 엔진을 감싸서 `BaseXxx` 인터페이스를 구현. `@register_xxx("name", version="v1")` 데코레이터로 레지스트리 등록.
- **Registry** (`registry.py`): 이름+버전 → 어댑터 매핑. `get_xxx("name", version="v1")` 으로 조회.

```
backend/llm_infrastructure/
├── preprocessing/     # L0~L5 정규화 엔진, 청킹
├── embedding/         # BGE, KoE5, TEI 어댑터
├── retrieval/         # Dense, BM25, Hybrid+RRF
├── llm/               # vLLM, OpenAI 어댑터 + LangGraph 에이전트
├── reranking/         # Cross-encoder, LLM 기반
├── query_expansion/   # Multi-query 확장
└── elasticsearch/     # ES 인덱스 관리
```

새 구현체 추가: adapter 파일 작성 → `@register_xxx` 데코레이터 → `adapters/__init__.py`에서 import.

### Backend Layers

```
api/routers/     →  services/        →  llm_infrastructure/
(HTTP endpoints)    (비즈니스 로직)      (엔진/어댑터/레지스트리)
```

- **api/main.py**: FastAPI 앱 진입점. 7개 라우터를 `/api` prefix로 등록.
- **api/dependencies.py**: LRU 캐시 기반 의존성 주입 (서비스 인스턴스 싱글톤).
- **api/routers/agent.py**: 메인 RAG 에이전트 엔드포인트 (`POST /api/agent/run`).
- **services/agents/langgraph_rag_agent.py**: LangGraph 기반 에이전트 오케스트레이션.
- **config/settings.py**: Pydantic Settings 12개 클래스. 환경변수(`RAG_`, `VLLM_`, `SEARCH_` 등)로 제어.
- **config/presets/**: 검색 프리셋 YAML 파일들.

### Frontend Structure

React 18 + TypeScript + Vite + Ant Design. Feature-based 구조:

```
frontend/src/
├── app/           # providers.tsx (QueryClient, Theme), router.tsx
├── features/
│   ├── chat/      # 메인 채팅 UI (agent 호출)
│   ├── search/    # 검색 인터페이스
│   ├── retrieval-test/  # 검색 테스트/디버깅 UI
│   ├── feedback/  # 피드백 수집
│   └── parsing/   # 문서 파싱 UI
├── components/    # 공유 컴포넌트 (layout, theme)
└── lib/           # API 클라이언트, 유틸
```

### Data & Storage

- **Elasticsearch**: 메인 검색 인덱스 + 채팅 히스토리. Nori 분석기(한국어). ES 데이터는 `/home/llm-share/es_data`에 저장.
- **chunk_v3 인덱스 구조**: `chunk_v3_content` (텍스트+메타) + `chunk_v3_embed_{model}_v1` (벡터) 분리 저장.
- **MinIO**: 문서 이미지 S3 호환 저장소.
- **HuggingFace 캐시**: 모델은 `data/hf_cache/` 또는 `/home/llm-share/hf/`에 캐시.
- SQL DB 없음 — 전부 ES + 파일 기반.

### Key Environment Variables

`.env`에서 주요 설정 관리. `.env.dev`/`.env.prod`로 오버라이드.

- `VLLM_BASE_URL`, `VLLM_MODEL_NAME`: LLM 서버 연결
- `SEARCH_ES_HOST`: Elasticsearch 주소 (기본 `localhost:8002`)
- `RAG_PREPROCESS_METHOD`, `RAG_EMBEDDING_METHOD`, `RAG_RETRIEVAL_PRESET`: 파이프라인 구성
- `ES_DATA_PATH`: ES 데이터 저장 경로

### Domain Context

- 반도체 PE(Process Engineering) 도메인 특화
- 문서 타입: SOP(Standard Operating Procedure), PEMS, 매뉴얼, TSG
- 장비명(device_name), 챕터, 모듈 등의 메타데이터가 검색 필터로 사용됨
- L0~L5 정규화: PM 주소 마스킹, 과학 표기 변환, 반도체 용어 통일 등
