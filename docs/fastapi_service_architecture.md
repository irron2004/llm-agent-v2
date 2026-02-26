# FastAPI 서비스 아키텍처 설계 문서

> **대상 독자**: 주니어 개발자
> **목적**: llm-agent-v2 백엔드를 FastAPI 서비스로 구조화하기 위한 설계 문서 + 코드 골격 + 테스트 가이드

---

## 표기 규칙

이 문서에서는 **현재 구현 상태**와 **목표 설계 상태**를 명확히 구분하기 위해 다음 라벨을 사용합니다:

| 라벨 | 의미 |
|------|------|
| **[As-Is]** | 현재 코드베이스에 실제로 구현되어 있는 내용 (2026-02-25 기준) |
| **[To-Be]** | 목표 설계이며, 아직 미구현이거나 스켈레톤/예시 수준의 내용 |
| **[Gap]** | As-Is에서 To-Be로 전환하기 위해 필요한 작업 목록 |

---

## As-Is (2026-02-25 기준)

### 1. 현재 라우터 구조

현재 `backend/api/main.py`에 포함된 활성 라우터 8개:

| 라우터 파일 | 엔드포인트_prefix | 용도 |
|------------|------------------|------|
| `health.py` | `/health` | 헬스체크 |
| `agent.py` | `/api/agent` | **FE 메인 채팅** (LangGraph RAG 에이전트) |
| `assets.py` | `/api/assets` | 정적 자산 |
| `search.py` | `/api/search` | 문서 검색 |
| `ingestions.py` | `/api/ingestions` | 문서 인제스트 |
| `conversations.py` | `/api/conversations` | 대화 세션 관리 |
| `feedback.py` | `/api/feedback` | 피드백 관리 |
| `retrieval.py` | `/api/retrieval` | 검색 (저수준 파이프라인) |

주석 처리된 비활성 라우터: `chat.py`, `preprocessing.py`, `rerank.py`, `query_expansion.py`, `summarization.py`, `devices.py`, `retrieval_evaluation.py`

### 2. FE 메인 채팅 경로

**핵심 엔드포인트** (현재 FE가 실제로 호출하는 경로):

- **일반 요청**: `POST /api/agent/run`
- **스트리밍 요청**: `POST /api/agent/run/stream`

**라우터 파일**: `backend/api/routers/agent.py`

**LangGraph 파이프라인**: `backend/services/agents/langgraph_rag_agent.py` (auto_parse → history_check → translate → route → mq → retrieve → answer → judge)

### 3. FE 설정 (채팅 경로)

**설정 파일**: `frontend/src/config/env.ts`

```typescript
// VITE_CHAT_PATH 환경변수 (기본값: /api/chat)
chatPath: normalize(import.meta.env.VITE_CHAT_PATH, "/api/chat"),
```

**WARNING**: FE 기본값 `/api/chat`은 현재 BE에서 라우터가 미마운트 상태라 호출 시 404가 발생합니다. 반드시 `VITE_CHAT_PATH`를 `/api/agent/run` 또는 `/api/agent/run/stream`으로 설정해야 합니다.

**경로 파싱 로직**: `frontend/src/features/chat/api.ts`의 `resolveChatPaths()` 함수

```typescript
export function resolveChatPaths(chatPath: string | undefined): {
  runPath: string;
  streamPath: string;
  canStream: boolean;
} {
  const configuredPath = chatPath?.trim() || "/api/agent/run";

  if (configuredPath.endsWith("/stream")) {
    const runPath = configuredPath.slice(0, -"/stream".length) || "/api/agent/run";
    return { runPath, streamPath: configuredPath, canStream: true };
  }

  if (configuredPath.endsWith("/run")) {
    return {
      runPath: configuredPath,
      streamPath: `${configuredPath}/stream`,
      canStream: true,
    };
  }

  // 그 외의 경우 (예: 미마운트된 /api/chat) → canStream: false
  return {
    runPath: configuredPath,
    streamPath: `${configuredPath}/stream`,
    canStream: false,
  };
}
```

**권장 설정값**:
- `VITE_CHAT_PATH=/api/agent/run` (일반 JSON 응답)
- `VITE_CHAT_PATH=/api/agent/run/stream` (SSE 스트리밍)

### 4. DI (의존성 주입) 패턴

**현재 패턴**: `backend/api/dependencies.py`

현재 코드베이스에서 사용되는 DI 함수들:

| 함수 | 용도 |
|------|------|
| `set_search_service(service)` | 앱 시작 시 SearchService 와이어링 (전역 인스턴스 설정) |
| `get_search_service()` | DI를 통해 SearchService 주입, 미설정 시 `_NotConfiguredSearchService` 반환 (`search()` 호출 시 RuntimeError) |
| `get_prompt_spec_cached()` | LangGraph 프롬프트 스펙 캐시 |
| `get_reranker()` | Reranker 인스턴스 (설정 기반) |
| `get_query_expander()` | Query Expander 인스턴스 (설정 기반) |
| `get_preprocessor_factory()` | 전처리기 팩토리 (level override 지원) |
| `get_default_llm()` | vLLM 기반 LLM |
| `get_default_embedder()` | 임베딩 모델 |
| `get_default_preprocessor()` | 기본 전처리기 |
| `get_default_retriever()` | Placeholder retriever (`_UnconfiguredRetriever` 반환) |
| `get_rag_service()` | RAG 서비스 (미설정 시 `_NotConfiguredRAGService` 반환) |
| `get_chat_service()` | Chat 서비스 (LLM only, retrieval 없음) |
| `get_simple_chat_prompt()` | 시스템 프롬프트 파일 로딩 (설정 기반) |

**미설정 시 동작**: `get_search_service()`는 `_NotConfiguredSearchService`를 반환하며, 실제 `search()` 호출 시 원인을 설명하는 `RuntimeError`를 raise합니다.

### 5. 앱 시작 로직

현재 `backend/api/main.py`는 `@app.on_event("startup")`을 사용합니다:

```python
@app.on_event("startup")
async def startup_search_service():
    """Wire SearchService at startup based on SEARCH_* settings."""
    try:
        _configure_search_service()
    except NotImplementedError as exc:
        logger.warning(str(exc))
```

`lifespan` 컨텍스트 매니저는 아직 사용되지 않습니다.

### 6. 테스트 구조

현재 테스트 디렉토리:

| 경로 | 설명 |
|------|------|
| `backend/tests/` | 단위 테스트 (~30개 파일) |
| `tests/api/` | FastAPI 엔드포인트 테스트 (~16개 파일) |

주요 테스트 파일:
- `backend/tests/test_langgraph_rag_agent_canonical.py`
- `backend/tests/test_retrieval_pipeline.py`
- `tests/api/test_health.py`
- `tests/api/test_agent_*.py` (6개)
- `tests/api/test_search_api.py`

---

## To-Be (설계/미구현)

### 1. 목표: Infra API / Service API 분리

**원칙**: FE는 "화면용 DTO (View Model)" 형태의 응답만 받고, 별도의 파싱/정렬/스니펫 생성이 필요 없어야 합니다.

| 구분 | Prefix | 용도 | FE 사용 |
|------|--------|------|---------|
| **Service API** | `/api/*` | 화면 단위 고수준 API | O |
| **Infra API** | `/internal/*` | 개별 모듈 디버그용 | X |

### 2. 제안 라우터 구조 (스켈레톤/예시)

```bash
backend/
├── api/
│   ├── main.py                      # [To-Be] lifespan 컨텍스트 매니저 권장
│   ├── dependencies.py              # [To-Be] load_corpus/get_corpus 패턴 적용 권장
│   └── routers/
│       ├── health.py
│       ├── agent.py                 # [As-Is] 이미 구현됨
│       ├── assets.py                # [As-Is] 이미 구현됨
│       ├── search.py                # [As-Is] 이미 구현됨
│       ├── ingestions.py            # [As-Is] 이미 구현됨
│       ├── conversations.py         # [As-Is] 이미 구현됨
│       ├── feedback.py              # [As-Is] 이미 구현됨
│       ├── retrieval.py             # [As-Is] 이미 구현됨
│       │
│       │ # ─── [To-Be] Service API (신규 구현 필요) ───
│       ├── chat.py                  # /api/chat/* - RAG 기반 채팅 (미구현)
│       │ # - 기존 chat.py는 주석 처리됨, agent.py로 대체됨
│       │ # - 재구현 시 View Model 응답 설계 필요
│       │
│       │ # ─── [To-Be] Infra API (신규 생성 필요) ───
│       ├── internal/
│       │   ├── __init__.py
│       │   ├── preprocessing.py     # /internal/preprocessing/*
│       │   ├── embedding.py         # /internal/embedding/*
│       │   ├── retrieval.py         # /internal/retrieval/*
│       │   └── llm.py               # /internal/llm/*
│       │
│       │ # ─── [To-Be] 문서 관리 API (신규 구현 필요) ───
│       └── documents.py             # /api/documents/* (미구현)
```

### 3. Service API 설계 원칙

**응답은 화면 기준으로 설계**

- DB 스키마 기준 X, infra 객체 그대로 X
- "이 페이지에서 보여줄 카드/리스트/그래프 그대로"를 모델로 정의

**프론트에서 하는 가공을 모두 백엔드로 옮김**

- 날짜 포맷
- 정렬
- top-k 잘라내기
- 점수 → 퍼센트화
- 하이라이트
- snippet 생성

**일관된 응답 형태**

```python
# 리스트 응답
{
    "items": [...],
    "total": int,
    "page": int,
    "size": int
}

# 에러 응답
{
    "error": {
        "code": str,
        "message": str,
        "details": dict | None
    }
}
```

### 4. DI 패턴 (To-Be 예시)

`load_corpus` / `get_corpus` 패턴 (스켈레톤):

```python
# backend/api/dependencies.py (To-Be 예시)
_corpus_instance: Optional[IndexedCorpus] = None


def get_corpus() -> IndexedCorpus:
    """현재 로드된 Corpus 반환 (없으면 에러)"""
    global _corpus_instance
    if _corpus_instance is None:
        raise RuntimeError("Corpus not loaded. Call load_corpus() first.")
    return _corpus_instance


def load_corpus(corpus: IndexedCorpus) -> None:
    """Corpus 로드 (앱 시작 시 또는 문서 인덱싱 후 호출)"""
    global _corpus_instance
    _corpus_instance = corpus
```

### 5. lifespan 권장 사항

FastAPI 권장 패턴: `lifespan` 컨텍스트 매니저 사용

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _configure_search_service()
    yield
    # Shutdown
    # 정리 로직
```

참고: https://fastapi.tiangolo.com/advanced/testing-using-the-test-client/#testing-startup-and-shutdown-events

### 6. 테스트 구조 (To-Be)

```bash
tests/
├── unit/                            # [To-Be] 순수 파이썬/비즈니스 로직 단위 테스트 (신규 생성 필요)
│   ├── test_preprocessing_*.py
│   ├── test_embedding_*.py
│   ├── test_retrieval_*.py
│   └── test_llm_*.py
├── services/                        # [To-Be] 서비스 레이어 테스트 (신규 생성 필요)
│   ├── test_chat_service.py
│   ├── test_search_service.py
│   └── test_rag_service.py
└── api/                             # [As-Is] FastAPI 엔드포인트 테스트
    ├── conftest.py
    ├── test_health.py
    ├── test_agent_*.py
    └── test_search_api.py
```

---

## Gap (As-Is → To-Be 전환 작업)

### P0: 문서 정합성 확보 (현재 진행 중)

- [x] As-Is section에 현재 활성 라우터 8개 명시
- [x] `/api/agent/run`, `/api/agent/run/stream` 엔드포인트 명시
- [x] FE 채팅 경로 설정 (VITE_CHAT_PATH, resolveChatPaths) 문서화
- [x] DI 패턴 (set_search_service, _NotConfiguredSearchService 등) 명시
- [x] 테스트 경로 (backend/tests/, tests/api/) 명시
- [x] @app.on_event("startup") 사용 사실 명시

### P1: 내부 라우팅 구조 설계

| 작업 | 우선순위 | 상태 |
|------|----------|------|
| `/internal/*` prefix 도입 설계 | P1 | 미진행 |
| 기존 `/api/*` 라우터와의 호환 정책 (redirect/alias) 설계 | P1 | 미진행 |
| Infra API 엔드포인트 명세 작성 | P1 | 미진행 |

### P2: 미구현 API 구현

| 작업 | 우선순위 | 상태 |
|------|----------|------|
| `/api/chat/*` 재구현 (View Model 응답) | P2 | 미구현 |
| `/api/documents/*` 구현 | P2 | 미구현 |
| `/internal/*` 라우터 4개 구현 (preprocessing, embedding, retrieval, llm) | P2 | 미구현 |
| lifespan 컨텍스트 매니저 도입 | P2 | 미구현 |
| `load_corpus/get_corpus` 패턴 적용 | P2 | 미구현 |
| `tests/unit/`, `tests/services/` 디렉토리 생성 | P2 | 미구현 |

---

## 부록: 참고 자료

### FastAPI 공식 문서
- https://fastapi.tiangolo.com/
- https://fastapi.tiangolo.com/tutorial/dependencies/
- https://fastapi.tiangolo.com/tutorial/testing/

### 프로젝트 내부 문서
- `backend/llm_infrastructure/preprocessing/README.md`: 전처리 모듈 가이드
- `backend/services/`: 기존 서비스 구현 참고
- `STRUCTURE.txt`: 프로젝트 전체 구조
