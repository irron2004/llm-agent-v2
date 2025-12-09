# FastAPI 서비스 아키텍처 설계 문서

> **대상 독자**: 주니어 개발자
> **목적**: llm-agent-v2 백엔드를 FastAPI 서비스로 구조화하기 위한 설계 문서 + 코드 골격 + 테스트 가이드

---

## 1. 목표 정리

### 1-1. 현재 상태

* FastAPI + `llm_infrastructure` 모듈 구현 완료
* `backend/services/`에 비즈니스 로직 서비스 구현 (ChatService, SearchService, RAGService 등)
* FE에서 응답 데이터를 파싱/정제/가공한 뒤 시각화하는 구조

### 1-2. 지향점

* **FE (Frontend)** 는:
  * "버튼 누른다 → 응답 객체를 그대로 컴포넌트에 바인딩해서 그린다"에 가깝게
  * 별도의 파싱/정렬/슬라이스/스니펫 생성이 **필요 없음**

* **BE (Backend) Service API** 는:
  * parsing, 필터링, 정렬, 집계, 하이라이트, 포맷 변환까지 다 끝낸 뒤
  * **"화면용 DTO (View Model)"** 형태로 응답

* 이를 위해 **Infra API (저수준)** 와 **Service API (화면/기능 단위 고수준)** 를 나눈다.

---

## 2. 계층 구조

기존 구조에 "Service API 레이어"를 추가하는 형태입니다.

```bash
backend/
├── llm_infrastructure/              # [기존] 저수준 인프라 (건드리지 말기)
│   ├── preprocessing/               # 텍스트 전처리 (L0-L5 정규화)
│   ├── embedding/                   # 임베딩 (sentence-transformers, TEI)
│   ├── retrieval/                   # 검색 (dense, BM25, hybrid)
│   └── llm/                         # LLM 추론 (vLLM)
│
├── services/                        # [기존] 도메인/비즈니스 로직 (infra 조합)
│   ├── chat_service.py              # LLM 호출 오케스트레이션
│   ├── search_service.py            # 검색 오케스트레이션
│   ├── rag_service.py               # RAG 파이프라인
│   ├── embedding_service.py         # 임베딩 서비스
│   ├── document_service.py          # 문서 인덱싱
│   └── ingest/                      # 문서 인제스트
│
└── api/                             # [구현 대상] FastAPI 레이어
    ├── __init__.py
    ├── main.py                      # FastAPI 앱 엔트리포인트
    ├── dependencies.py              # 공통 의존성 (DI)
    └── routers/
        ├── __init__.py
        ├── health.py                # 헬스체크
        │
        │ # ─── Infra API (저수준, 내부/디버그용) ───
        ├── internal/
        │   ├── __init__.py
        │   ├── preprocessing.py     # /internal/preprocessing/*
        │   ├── embedding.py         # /internal/embedding/*
        │   ├── retrieval.py         # /internal/retrieval/*
        │   └── llm.py               # /internal/llm/*
        │
        │ # ─── Service API (고수준, FE가 실제 사용) ───
        ├── chat.py                  # /api/chat/* - RAG 기반 채팅
        ├── search.py                # /api/search/* - 문서 검색
        └── documents.py             # /api/documents/* - 문서 관리

tests/
├── unit/                            # 순수 파이썬/비즈니스 로직 단위 테스트
│   ├── test_preprocessing_*.py
│   ├── test_embedding_*.py
│   ├── test_retrieval_*.py
│   └── test_llm_*.py
├── services/                        # 서비스 레이어 테스트
│   ├── test_chat_service.py
│   ├── test_search_service.py
│   └── test_rag_service.py
└── api/                             # FastAPI 엔드포인트 테스트
    ├── conftest.py                  # 공통 client fixture
    ├── test_health.py
    ├── test_chat_api.py
    ├── test_search_api.py
    └── internal/
        └── test_internal_*.py
```

---

## 3. Infra API vs Service API

### 3-1. Infra API (저수준)

| 항목 | 설명 |
|------|------|
| **위치** | `/internal/preprocessing`, `/internal/embedding`, `/internal/retrieval`, `/internal/llm` |
| **용도** | 디버그, 내부 테스트, 개별 모듈 동작 확인 |
| **특징** | `llm_infrastructure` 모듈을 1:1로 노출 |
| **FE 사용** | X (사용하지 않음) |

### 3-2. Service API (고수준)

| 항목 | 설명 |
|------|------|
| **위치** | `/api/chat`, `/api/search`, `/api/documents` |
| **용도** | FE에서 실제로 호출하는 엔드포인트 |
| **특징** | 화면 단위로 묶은 고수준 API, View Model 반환 |
| **FE 사용** | O (메인 사용) |

---

## 4. Service API 설계 원칙 (FE 부담 최소화)

### 4-1. 응답은 화면 기준으로 설계

* DB 스키마 기준 X, infra 객체 그대로 X
* "이 페이지에서 보여줄 카드/리스트/그래프 그대로"를 모델로 정의

### 4-2. 프론트에서 하는 가공을 모두 백엔드로 옮긴다

* 날짜 포맷
* 정렬
* top-k 잘라내기
* 점수 → 퍼센트화
* 하이라이트
* snippet 생성

### 4-3. 일관된 응답 형태

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

### 4-4. FE가 할 일 (최소화)

1. API 호출
2. 응답 필드 → 컴포넌트 바인딩
3. 정말 필요한 최소한의 상태 관리만 수행

---

## 5. FastAPI 앱 골격

### 5-1. `backend/api/main.py`

```python
# backend/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import health
from backend.api.routers import chat, search, documents
from backend.api.routers.internal import preprocessing, embedding, retrieval, llm


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Agent API",
        version="0.1.0",
        description="RAG-based PE troubleshooting agent API",
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 구체적인 origin 지정
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    app.include_router(health.router)

    # ─── Service API (FE용, 고수준) ───
    app.include_router(chat.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")

    # ─── Infra API (내부/디버그용, 저수준) ───
    app.include_router(preprocessing.router, prefix="/internal")
    app.include_router(embedding.router, prefix="/internal")
    app.include_router(retrieval.router, prefix="/internal")
    app.include_router(llm.router, prefix="/internal")

    return app


# uvicorn이 참조할 전역 app
app = create_app()
```

> **주니어에게 전달 포인트**
>
> * 항상 `create_app()` 패턴을 쓰자. 테스트에서 `create_app()`을 불러 TestClient를 만들기 좋다.
> * FE가 사용하는 API는 `/api/*` prefix
> * 내부/디버그용 API는 `/internal/*` prefix

---

### 5-2. `backend/api/dependencies.py` (공통 의존성)

Heavy한 객체(모델 로딩, 인덱스 등)는 여기서 한 번 만들고 FastAPI DI로 주입한다.

```python
# backend/api/dependencies.py
from functools import lru_cache
from typing import Optional

from backend.config.settings import rag_settings, vllm_settings
from backend.llm_infrastructure.preprocessing import get_preprocessor
from backend.llm_infrastructure.embedding import get_embedder
from backend.llm_infrastructure.retrieval import get_retriever
from backend.llm_infrastructure.llm import get_llm

from backend.services.chat_service import ChatService
from backend.services.search_service import SearchService
from backend.services.rag_service import RAGService
from backend.services.embedding_service import EmbeddingService
from backend.services.document_service import DocumentIndexService, IndexedCorpus


# ════════════════════════════════════════════════════════════════
# Infra 레벨 의존성 (저수준)
# ════════════════════════════════════════════════════════════════

@lru_cache
def get_default_preprocessor():
    """기본 전처리기 (L3 정규화)"""
    return get_preprocessor(
        rag_settings.preprocess_method,
        version=rag_settings.preprocess_version,
    )


@lru_cache
def get_default_embedder():
    """기본 임베더 (settings 기반)"""
    return get_embedder(
        rag_settings.embedding_method,
        version=rag_settings.embedding_version,
        device=rag_settings.embedding_device,
    )


@lru_cache
def get_default_retriever():
    """주의: Retriever는 corpus가 필요하므로, 실제 사용시 SearchService 권장"""
    raise NotImplementedError("Use SearchService instead - retriever requires corpus")


@lru_cache
def get_default_llm():
    """기본 LLM (vLLM)"""
    return get_llm(
        "vllm",
        version="v1",
        base_url=vllm_settings.base_url,
        model=vllm_settings.model_name,
        temperature=vllm_settings.temperature,
        max_tokens=vllm_settings.max_tokens,
        timeout=vllm_settings.timeout,
    )


# ════════════════════════════════════════════════════════════════
# Service 레벨 의존성 (고수준)
# ════════════════════════════════════════════════════════════════

@lru_cache
def get_chat_service() -> ChatService:
    """ChatService 싱글톤"""
    return ChatService()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """EmbeddingService 싱글톤"""
    return EmbeddingService(
        method=rag_settings.embedding_method,
        version=rag_settings.embedding_version,
        device=rag_settings.embedding_device,
        use_cache=rag_settings.embedding_use_cache,
        cache_dir=rag_settings.embedding_cache_dir,
    )


# Corpus/SearchService/RAGService는 corpus 인스턴스가 필요하므로
# 앱 시작 시 또는 요청 시 생성 로직 필요
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


def get_search_service() -> SearchService:
    """SearchService (corpus 기반)"""
    corpus = get_corpus()
    return SearchService(corpus)


def get_rag_service() -> RAGService:
    """RAGService (corpus 기반)"""
    corpus = get_corpus()
    return RAGService(corpus)
```

> **포인트**
>
> * `@lru_cache`를 사용해서 인스턴스를 한 번만 만들고 재사용 → 모델/인덱스 로딩 비용 절감
> * Infra 레벨과 Service 레벨 의존성을 명확히 분리
> * Corpus가 필요한 서비스는 별도 로딩 로직 필요

---

## 6. Service API 구현 예시

### 6-1. Chat Service API (`api/routers/chat.py`)

FE가 실제로 사용하는 고수준 Chat API. RAG 기반으로 질문에 답변하고, 관련 문서 목록까지 함께 반환.

```python
# backend/api/routers/chat.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import get_rag_service
from backend.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Chat Service"])


# ─── Request/Response Models ───

class HistoryMessage(BaseModel):
    role: str  # "user" / "assistant" / "system"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="사용자 질문")
    history: List[HistoryMessage] = Field(default=[], description="대화 히스토리")
    top_k: int = Field(default=5, description="검색할 문서 수")


class RetrievedDoc(BaseModel):
    """FE에서 바로 사용할 수 있는 문서 카드 형태"""
    id: str
    title: str
    snippet: str  # 200자로 잘린 미리보기
    score: float
    score_percent: int  # 0-100 정수로 변환된 점수


class ChatResponse(BaseModel):
    """FE에서 바로 바인딩할 수 있는 응답 구조"""
    query: str  # 원본 질문
    clean_query: str  # 전처리된 질문
    answer: str  # LLM 응답
    retrieved_docs: List[RetrievedDoc]  # 관련 문서 카드 리스트
    follow_ups: List[str]  # 추천 후속 질문
    metadata: dict = {}


# ─── Endpoints ───

@router.post("/ask", response_model=ChatResponse)
async def ask(
    req: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    RAG 기반 질문 응답 API.

    FE는 이 응답을 그대로 그려주기만 하면 된다:
    - answer → 챗 버블
    - retrieved_docs → 오른쪽 문서 카드 리스트
    - follow_ups → 아래 "추가 질문" 버튼 리스트
    """
    try:
        # RAG 쿼리 실행
        rag_response = rag_service.query(
            req.message,
            top_k=req.top_k,
        )

        # FE용 문서 카드로 변환 (가공은 BE에서!)
        retrieved_docs = []
        for result in rag_response.context:
            # 제목 추출 (metadata에서 또는 content 첫 줄)
            title = ""
            if hasattr(result, "metadata") and result.metadata:
                title = result.metadata.get("title", "")
            if not title:
                title = result.content.split("\n")[0][:50] + "..."

            # snippet 생성 (200자 제한)
            snippet = result.content[:200]
            if len(result.content) > 200:
                snippet += "..."

            # score를 퍼센트로 변환
            score_percent = int(result.score * 100) if result.score <= 1 else int(min(result.score, 100))

            retrieved_docs.append(RetrievedDoc(
                id=result.doc_id,
                title=title,
                snippet=snippet,
                score=result.score,
                score_percent=score_percent,
            ))

        # 후속 질문 생성
        follow_ups = _generate_follow_ups(req.message)

        return ChatResponse(
            query=req.message,
            clean_query=rag_response.metadata.get("preprocessed_query", req.message),
            answer=rag_response.answer,
            retrieved_docs=retrieved_docs,
            follow_ups=follow_ups,
            metadata={
                "num_results": len(retrieved_docs),
                "top_k": req.top_k,
            },
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _generate_follow_ups(query: str) -> List[str]:
    """후속 질문 생성 (간단 버전, 나중에 LLM으로 대체 가능)"""
    return [
        f"{query}에 대한 예시를 더 보여줘",
        f"{query}를 실제 업무에 적용하는 방법은?",
        f"{query} 관련 주의사항이 있어?",
    ]
```

---

### 6-2. Search Service API (`api/routers/search.py`)

문서 검색 페이지용 API. 페이지네이션, 정렬 등 FE에서 할 일을 BE에서 처리.

```python
# backend/api/routers/search.py
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import get_search_service
from backend.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search Service"])


# ─── Response Models ───

class SearchResultItem(BaseModel):
    """검색 결과 아이템 (FE 테이블/카드에 바로 바인딩)"""
    rank: int  # 순위 (1부터 시작)
    id: str
    title: str
    snippet: str  # 미리보기 텍스트
    score: float
    score_display: str  # "95%" 형태로 포맷팅됨
    highlight_terms: List[str] = []  # 하이라이트할 검색어


class SearchResponse(BaseModel):
    """페이지네이션이 적용된 검색 결과"""
    query: str
    clean_query: str
    items: List[SearchResultItem]
    total: int  # 전체 결과 수
    page: int
    size: int
    has_next: bool  # 다음 페이지 존재 여부


# ─── Endpoints ───

@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="검색어", min_length=1),
    page: int = Query(default=1, ge=1, description="페이지 번호"),
    size: int = Query(default=10, ge=1, le=100, description="페이지 크기"),
    search_service: SearchService = Depends(get_search_service),
):
    """
    문서 검색 API.

    FE는 이 응답을 테이블/카드 리스트로 바로 렌더링:
    - items → 테이블 rows 또는 카드 리스트
    - total, page, size, has_next → 페이지네이션 UI
    """
    try:
        # 검색 실행 (top_k를 넉넉하게 가져와서 페이지네이션)
        top_k = page * size + size  # 여유분 포함
        results = search_service.search(q, top_k=top_k)

        # 페이지네이션 적용
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        page_results = results[start_idx:end_idx]

        # FE용 아이템으로 변환
        items = []
        for idx, result in enumerate(page_results):
            # 제목 추출
            title = ""
            if hasattr(result, "metadata") and result.metadata:
                title = result.metadata.get("title", "")
            if not title:
                title = result.content.split("\n")[0][:50]

            # snippet 생성
            snippet = result.content[:150]
            if len(result.content) > 150:
                snippet += "..."

            # score 포맷팅
            if result.score <= 1:
                score_display = f"{int(result.score * 100)}%"
            else:
                score_display = f"{result.score:.2f}"

            items.append(SearchResultItem(
                rank=start_idx + idx + 1,
                id=result.doc_id,
                title=title,
                snippet=snippet,
                score=result.score,
                score_display=score_display,
                highlight_terms=q.split()[:3],  # 검색어 하이라이트용
            ))

        total = len(results)
        has_next = end_idx < total

        return SearchResponse(
            query=q,
            clean_query=q,  # 실제로는 전처리된 쿼리
            items=items,
            total=total,
            page=page,
            size=size,
            has_next=has_next,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
```

---

### 6-3. Health Check (`api/routers/health.py`)

```python
# backend/api/routers/health.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        version="0.1.0",
    )
```

---

## 7. Infra API 구현 예시 (내부/디버그용)

### 7-1. Internal Preprocessing (`api/routers/internal/preprocessing.py`)

```python
# backend/api/routers/internal/preprocessing.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.api.dependencies import get_default_preprocessor

router = APIRouter(prefix="/preprocessing", tags=["Internal - Preprocessing"])


class PreprocessRequest(BaseModel):
    text: str


class PreprocessResponse(BaseModel):
    original: str
    processed: str


@router.post("/apply", response_model=PreprocessResponse)
async def apply_preprocessing(
    body: PreprocessRequest,
    preprocessor=Depends(get_default_preprocessor),
):
    """
    [내부용] 단일 텍스트에 전처리 적용.
    디버그/테스트 목적으로만 사용.
    """
    processed = list(preprocessor.preprocess([body.text]))
    return PreprocessResponse(
        original=body.text,
        processed=processed[0] if processed else body.text,
    )
```

### 7-2. Internal Embedding (`api/routers/internal/embedding.py`)

```python
# backend/api/routers/internal/embedding.py
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.api.dependencies import get_default_embedder

router = APIRouter(prefix="/embedding", tags=["Internal - Embedding"])


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    text: str
    embedding: List[float]
    dimension: int


@router.post("/embed", response_model=EmbedResponse)
async def embed_text(
    body: EmbedRequest,
    embedder=Depends(get_default_embedder),
):
    """
    [내부용] 단일 텍스트를 벡터로 변환.
    디버그/테스트 목적으로만 사용.
    """
    vector = embedder.embed(body.text)
    embedding_list = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
    return EmbedResponse(
        text=body.text,
        embedding=embedding_list,
        dimension=len(embedding_list),
    )
```

### 7-3. Internal LLM (`api/routers/internal/llm.py`)

```python
# backend/api/routers/internal/llm.py
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.api.dependencies import get_default_llm

router = APIRouter(prefix="/llm", tags=["Internal - LLM"])


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMChatRequest(BaseModel):
    messages: List[ChatMessage]


class LLMChatResponse(BaseModel):
    content: str
    model: str = ""


@router.post("/chat", response_model=LLMChatResponse)
async def chat(
    body: LLMChatRequest,
    llm=Depends(get_default_llm),
):
    """
    [내부용] LLM 직접 호출.
    디버그/테스트 목적으로만 사용.
    """
    response = llm.generate(messages=[m.dict() for m in body.messages])
    return LLMChatResponse(
        content=response.text,
        model=getattr(response, "model", ""),
    )
```

---

## 8. 테스트 전략 & 가이드

### 8-1. 테스트 디렉토리 구조

```bash
tests/
├── unit/              # 순수 파이썬/비즈니스 로직 단위 테스트
│   └── ...            # llm_infrastructure 내부 테스트
├── services/          # 서비스 레이어 테스트
│   ├── test_chat_service.py
│   ├── test_search_service.py
│   └── test_rag_service.py
└── api/               # FastAPI 엔드포인트 테스트
    ├── conftest.py    # 공통 fixture
    ├── test_health.py
    ├── test_chat_api.py
    ├── test_search_api.py
    └── internal/
        └── test_internal_*.py
```

### 8-2. 공통 TestClient fixture (`tests/api/conftest.py`)

```python
# tests/api/conftest.py
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api import dependencies


# ─── Fake 구현체들 ───

class FakePreprocessor:
    def preprocess(self, texts):
        return [t.strip().lower() for t in texts]


class FakeEmbedder:
    def embed(self, text: str):
        return [float(len(text)), 1.0, 2.0]

    def embed_batch(self, texts):
        return [[float(len(t)), 1.0, 2.0] for t in texts]


class FakeLLM:
    def generate(self, messages):
        last_user = [m for m in messages if m["role"] == "user"][-1]
        class Resp:
            text = f"Mock answer for: {last_user['content']}"
            model = "fake-model"
        return Resp()


class FakeRetrievalResult:
    def __init__(self, doc_id, content, score):
        self.doc_id = doc_id
        self.content = content
        self.score = score
        self.metadata = {"title": f"Document {doc_id}"}


class FakeSearchService:
    def search(self, query, top_k=10):
        return [
            FakeRetrievalResult(f"doc_{i}", f"Content for {query} - item {i}", 0.9 - i * 0.1)
            for i in range(min(top_k, 5))
        ]


class FakeRAGService:
    def query(self, question, top_k=5):
        class FakeRAGResponse:
            answer = f"This is a mock answer for: {question}"
            context = [
                FakeRetrievalResult(f"doc_{i}", f"Context for {question}", 0.9 - i * 0.1)
                for i in range(top_k)
            ]
            question = question
            metadata = {"preprocessed_query": question.lower()}
        return FakeRAGResponse()


# ─── Override 함수들 ───

def override_get_default_preprocessor():
    return FakePreprocessor()


def override_get_default_embedder():
    return FakeEmbedder()


def override_get_default_llm():
    return FakeLLM()


def override_get_search_service():
    return FakeSearchService()


def override_get_rag_service():
    return FakeRAGService()


# ─── Fixtures ───

@pytest.fixture
def client():
    """모든 의존성이 Fake로 교체된 TestClient"""
    app = create_app()

    # DI 오버라이드
    app.dependency_overrides[dependencies.get_default_preprocessor] = override_get_default_preprocessor
    app.dependency_overrides[dependencies.get_default_embedder] = override_get_default_embedder
    app.dependency_overrides[dependencies.get_default_llm] = override_get_default_llm
    app.dependency_overrides[dependencies.get_search_service] = override_get_search_service
    app.dependency_overrides[dependencies.get_rag_service] = override_get_rag_service

    yield TestClient(app)


@pytest.fixture
def client_minimal():
    """최소 오버라이드만 적용된 TestClient (health check 등)"""
    app = create_app()
    yield TestClient(app)
```

### 8-3. Service API 테스트 예시

```python
# tests/api/test_chat_api.py
def test_chat_ask_success(client):
    """Chat API 정상 동작 테스트"""
    payload = {
        "message": "PM 예방 점검 주기는?",
        "top_k": 3,
    }
    resp = client.post("/api/chat/ask", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    # 응답 구조 검증
    assert "query" in data
    assert "answer" in data
    assert "retrieved_docs" in data
    assert "follow_ups" in data

    # FE가 바로 쓸 수 있는 형태인지 검증
    assert data["query"] == payload["message"]
    assert len(data["retrieved_docs"]) <= payload["top_k"]
    assert len(data["follow_ups"]) > 0

    # 문서 카드 형태 검증
    for doc in data["retrieved_docs"]:
        assert "id" in doc
        assert "title" in doc
        assert "snippet" in doc
        assert "score" in doc
        assert "score_percent" in doc


def test_chat_ask_with_history(client):
    """대화 히스토리 포함 테스트"""
    payload = {
        "message": "더 자세히 설명해줘",
        "history": [
            {"role": "user", "content": "PM이 뭐야?"},
            {"role": "assistant", "content": "PM은 예방 정비입니다."},
        ],
        "top_k": 3,
    }
    resp = client.post("/api/chat/ask", json=payload)

    assert resp.status_code == 200
```

```python
# tests/api/test_search_api.py
def test_search_basic(client):
    """Search API 기본 동작 테스트"""
    resp = client.get("/api/search", params={"q": "PM 점검"})

    assert resp.status_code == 200
    data = resp.json()

    # 응답 구조 검증
    assert "query" in data
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert "has_next" in data

    # 아이템 형태 검증
    for item in data["items"]:
        assert "rank" in item
        assert "id" in item
        assert "title" in item
        assert "snippet" in item
        assert "score_display" in item


def test_search_pagination(client):
    """Search API 페이지네이션 테스트"""
    resp = client.get("/api/search", params={"q": "test", "page": 1, "size": 2})

    assert resp.status_code == 200
    data = resp.json()

    assert data["page"] == 1
    assert data["size"] == 2
    assert len(data["items"]) <= 2
```

### 8-4. Health Check 테스트

```python
# tests/api/test_health.py
def test_health_ok(client_minimal):
    resp = client_minimal.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
```

---

## 9. 구현 순서 (주니어 개발자용 체크리스트)

### Phase 1: 기본 구조 생성
- [ ] `backend/api/main.py` 작성
- [ ] `backend/api/dependencies.py` 작성
- [ ] `backend/api/routers/__init__.py` 생성
- [ ] `backend/api/routers/internal/__init__.py` 생성

### Phase 2: Health Check 구현 및 테스트
- [ ] `backend/api/routers/health.py` 작성
- [ ] `tests/api/conftest.py` 작성
- [ ] `tests/api/test_health.py` 작성
- [ ] 테스트 실행 확인

### Phase 3: Service API 구현 (FE 사용)
- [ ] `backend/api/routers/chat.py` 작성
- [ ] `backend/api/routers/search.py` 작성
- [ ] `tests/api/test_chat_api.py` 작성
- [ ] `tests/api/test_search_api.py` 작성
- [ ] 테스트 실행 및 검증

### Phase 4: Internal API 구현 (디버그용)
- [ ] `backend/api/routers/internal/preprocessing.py` 작성
- [ ] `backend/api/routers/internal/embedding.py` 작성
- [ ] `backend/api/routers/internal/llm.py` 작성
- [ ] 테스트 작성 및 검증

### Phase 5: 통합 테스트 및 문서화
- [ ] 모든 테스트 통합 실행: `pytest tests/api/ -v`
- [ ] API 문서 확인: `http://localhost:8000/docs`
- [ ] 개발 서버 실행 테스트

---

## 10. 실행 방법

### 개발 서버 실행

```bash
# 기본 실행
uvicorn backend.api.main:app --reload

# 커스텀 포트
uvicorn backend.api.main:app --reload --port 8080

# 외부 접속 허용
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### API 문서 확인

서버 실행 후:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 테스트 실행

```bash
# 전체 테스트
pytest

# API 테스트만
pytest tests/api/

# 특정 파일
pytest tests/api/test_chat_api.py -v

# 커버리지
pytest --cov=backend/api tests/api/
```

---

## 11. 주니어 개발자에게 전달할 핵심 요약

### 11-1. API 계층 구분

| 레이어 | Prefix | 용도 | FE 사용 |
|--------|--------|------|---------|
| Service API | `/api/*` | 화면 단위 고수준 API | O |
| Infra API | `/internal/*` | 개별 모듈 디버그용 | X |

### 11-2. 핵심 원칙

> **"infra 레벨 API는 FE에서 직접 쓰지 말고,
> ChatService / SearchService 같이 화면 단위로 묶은 Service API를 만들어서
> FE는 그 응답을 그대로 그리기만 하게 만들자."**

### 11-3. 응답 설계 체크리스트

- [ ] FE에서 추가 가공이 필요한가? → 필요하면 BE로 옮기기
- [ ] snippet은 BE에서 잘라서 주는가?
- [ ] score는 표시용 포맷 (% 등)으로 변환해서 주는가?
- [ ] 페이지네이션 정보가 포함되어 있는가?
- [ ] 에러 메시지가 사용자 친화적인가?

---

## 12. 트러블슈팅

### 문제 1: Import Error
**증상**: `ModuleNotFoundError: No module named 'backend'`

**해결**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# 또는
pip install -e .
```

### 문제 2: Corpus not loaded
**증상**: `RuntimeError: Corpus not loaded`

**해결**:
- 앱 시작 시 `load_corpus()` 호출 필요
- 또는 테스트에서 `get_rag_service` dependency override 사용

### 문제 3: 테스트에서 모델 로딩 시간 초과
**증상**: 테스트가 너무 오래 걸림

**해결**:
- `conftest.py`에서 Fake 클래스로 dependency override
- 실제 모델 테스트는 `@pytest.mark.slow` 마커로 분리

---

## 부록: 참고 자료

### FastAPI 공식 문서
- https://fastapi.tiangolo.com/

### 의존성 주입 (Dependency Injection)
- https://fastapi.tiangolo.com/tutorial/dependencies/

### 테스트
- https://fastapi.tiangolo.com/tutorial/testing/

### 프로젝트 내부 문서
- `backend/llm_infrastructure/preprocessing/README.md`: 전처리 모듈 가이드
- `backend/services/`: 기존 서비스 구현 참고
- `STRUCTURE.txt`: 프로젝트 전체 구조