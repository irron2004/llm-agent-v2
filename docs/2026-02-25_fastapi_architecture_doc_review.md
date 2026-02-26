# FastAPI 서비스 아키텍처 설계 문서 — 코드베이스 대조 리뷰

> **리뷰 대상**: `docs/fastapi_service_architecture.md` (As-Is / To-Be / Gap 분리 버전)
> **리뷰 일시**: 2026-02-26
> **리뷰 방법**: 문서의 각 라인을 실제 코드베이스와 1:1 대조

---

## 1. As-Is 섹션 검증 결과

### 1-1. 라우터 구조 (문서 line 22~37) — 정확

`main.py` line 98~113과 대조:

| 문서 | main.py | 일치 |
|------|---------|------|
| `health.py` → `/health` | `app.include_router(health.router)` (prefix 없음) + 라우터 내부 `prefix="/health"` | O |
| `agent.py` → `/api/agent` | `app.include_router(agent.router, prefix="/api")` + 라우터 내부 `prefix="/agent"` | O |
| `assets.py` → `/api/assets` | `app.include_router(assets.router, prefix="/api")` + `prefix="/assets"` | O |
| `search.py` → `/api/search` | 동일 패턴 | O |
| `ingestions.py` → `/api/ingestions` | 동일 패턴 | O |
| `conversations.py` → `/api/conversations` | 동일 패턴 | O |
| `feedback.py` → `/api/feedback` | 동일 패턴 | O |
| `retrieval.py` → `/api/retrieval` | 동일 패턴 | O |
| 비활성 7개 | main.py 주석 처리 7개 | O |

### 1-2. FE 메인 채팅 경로 (문서 line 39~48) — 정확

- `resolveChatPaths()` 코드: `frontend/src/features/chat/api.ts` line 29~57과 **완전 일치**
- `VITE_CHAT_PATH` 기본값: `frontend/src/config/env.ts` line 29 `"/api/chat"` 일치
- `.env` 실제 값: `VITE_CHAT_PATH=/api/agent/run` 일치

### 1-3. DI 패턴 (문서 line 97~115) — 4개 함수 누락

문서에 있는 9개 함수는 모두 정확하다. 단, `dependencies.py`에 실제로 존재하는 **13개 함수 중 4개가 누락**:

| 누락된 함수 | 위치 | 용도 |
|---|---|---|
| `get_default_retriever()` | line 79 | Placeholder retriever (`_UnconfiguredRetriever` 반환) |
| `get_rag_service()` | line 138 | RAG 서비스 (`_NotConfiguredRAGService` 반환) |
| `get_chat_service()` | line 144 | Chat 서비스 (LLM only) |
| `get_simple_chat_prompt()` | line 150 | 시스템 프롬프트 파일 로딩 |

### 1-4. 앱 시작 로직 (문서 line 117~131) — 정확 (간소화)

문서의 startup 코드는 실제 코드의 간소화 버전이다. 실제로는 `except` 절이 2개:

```python
# 실제 main.py line 115-123
except NotImplementedError as exc:  # explicit backend stub
    logger.warning(str(exc))
except Exception as exc:  # pragma: no cover - defensive logging
    logger.warning(f"Search service not configured: {exc}")
```

문서에는 첫 번째 `except`만 기재. 동작 이해에 영향 없으므로 **Low** 수준.

### 1-5. 테스트 구조 (문서 line 133~147) — 정확

- `backend/tests/` ~30개 ✓
- `tests/api/` ~16개 ✓
- 주요 파일 목록 ✓

---

## 2. 발견된 이슈

### Issue 1: 오타 — `검색 ( низレベル)` (line 35)

**심각도**: Low

러시아어(`низ`) + 일본어(`レベル`)가 혼합되어 있다.

**수정안**: `검색 (저수준 파이프라인)` 또는 `검색 (step-by-step 실행)`

---

### Issue 2: DI 함수 4개 누락 (line 103~113)

**심각도**: Medium

`dependencies.py`의 public 함수 13개 중 9개만 기재. 누락된 4개:
- `get_default_retriever()` — Placeholder, 실제 사용되지 않으나 DI 체인에 존재
- `get_rag_service()` — `_NotConfiguredRAGService` fallback
- `get_chat_service()` — LLM-only 채팅 서비스
- `get_simple_chat_prompt()` — 시스템 프롬프트 파일 로딩

주니어가 `dependencies.py`를 수정할 때 문서에 없는 함수를 발견하면 혼란 가능.

---

### Issue 3: To-Be 트리 구조 포맷 깨짐 (line 189~193)

**심각도**: Medium

```
│           ├── retrieval.py         # /internal/retrieval/*
│                         # /internal/llm/*
 └── llm.py│
│   │ # ─── [To-Be] 문서 관리 API (신규 구현 필요) ───
│   └── documents.py                 # /api/documents/* (미구현)
```

`llm.py` 위치와 `documents.py` 들여쓰기가 깨져 있어 트리 구조를 파악하기 어렵다.

**수정안**:
```
│           ├── retrieval.py         # /internal/retrieval/*
│           └── llm.py               # /internal/llm/*
│
│       # ─── [To-Be] 문서 관리 API (신규 구현 필요) ───
│       └── documents.py             # /api/documents/* (미구현)
```

---

### Issue 4: `경로解析` 표기 (line 61)

**심각도**: Low

`경로解析 로직` — `解析`는 일본어 한자 표기. 한국어 문서에서는 `경로 해석 로직` 또는 `경로 분석 로직`이 자연스럽다.

---

## 3. 이전 리뷰(v1) 대비 해결 현황

| 이전 지적 | 상태 | 비고 |
|---|---|---|
| H1. `agent.py` 미언급 | **해결** | As-Is 섹션 1~2에서 FE 메인 채팅으로 명시 |
| H2. `/internal/*` 혼동 | **해결** | To-Be로 분리, As-Is에는 현재 구조만 기재 |
| H3. DI 패턴 불일치 | **해결** | `set_search_service` / `_NotConfigured*` 패턴 정확히 기재 |
| M1. 라우터 목록 불일치 | **해결** | 활성 8개 + 비활성 7개 정확 |
| M4. 테스트 구조 불일치 | **해결** | `backend/tests/`, `tests/api/` 기준으로 정리 |
| L1. startup 로직 미언급 | **해결** | `@app.on_event("startup")` 코드 포함 |

---

## 4. 종합 판단

As-Is 섹션의 **코드 정합성이 높다**. 라우터 8개, FE 경로, DI 9개 함수, startup 로직, 테스트 경로 모두 실제 코드와 일치한다.

남은 수정 사항:

| 이슈 | 심각도 | 작업량 |
|------|--------|--------|
| DI 함수 4개 추가 | Medium | 테이블에 4행 추가 |
| To-Be 트리 포맷 수정 | Medium | 5줄 수정 |
| `低レベル` 오타 | Low | 1단어 |
| `경로解析` 표기 | Low | 1단어 |

Medium 2건 수정 후 주니어 배포 가능 수준.
