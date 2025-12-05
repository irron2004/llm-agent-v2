# FastAPI 서비스 아키텍처 설계 문서

> **대상 독자**: 주니어 개발자
> **목적**: llm-agent-v2 백엔드를 FastAPI 서비스로 구조화하기 위한 설계 문서 + 코드 골격 + 테스트 가이드

---

## 1. 목표

* `backend/llm_infrastructure`에 있는 **preprocessing / embedding / retrieval / llm** 모듈들을 FastAPI 서비스로 감싸서 REST API 형태로 제공한다.
* 이 구조를 기반으로 **주니어 개발자가**:
  * 엔드포인트를 추가/수정하고
  * 모듈을 교체하고
  * 테스트를 작성할 수 있게 한다.

---

## 2. FastAPI 프로젝트 구조 제안

기존 구조를 최대한 존중하면서, FastAPI용 레이어만 추가한 형태입니다.

```bash
backend/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 앱 엔트리포인트 (uvicorn이 여기를 띄움)
│   ├── dependencies.py         # 공통 의존성 (preprocessor, embedder, retriever, llm 인스턴스 제공)
│   └── routers/
│       ├── __init__.py
│       ├── health.py           # 헬스체크용 엔드포인트
│       ├── preprocessing.py    # 전처리 관련 API
│       ├── embedding.py        # 임베딩 관련 API
│       ├── retrieval.py        # 검색 관련 API
│       └── llm.py              # LLM 호출 / 채팅 관련 API
├── llm_infrastructure/
│   ├── preprocessing/
│   ├── embedding/
│   ├── retrieval/
│   └── llm/
└── ...

tests/
├── unit/
│   ├── test_preprocessing_standard.py
│   ├── test_embedding_registry.py
│   ├── test_retrieval_presets.py
│   └── test_llm_vllm_adapter.py
└── api/
    ├── conftest.py             # 공통 client fixture
    ├── test_health.py
    ├── test_preprocessing_api.py
    ├── test_embedding_api.py
    ├── test_retrieval_api.py
    └── test_llm_api.py
```

---

## 3. FastAPI 앱 골격

### 3-1. `backend/api/main.py`

```python
# backend/api/main.py
from fastapi import FastAPI
from backend.api.routers import health, preprocessing, embedding, retrieval, llm

def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Infrastructure API",
        version="0.1.0",
    )

    # 라우터 등록
    app.include_router(health.router)
    app.include_router(preprocessing.router)
    app.include_router(embedding.router)
    app.include_router(retrieval.router)
    app.include_router(llm.router)

    return app


# uvicorn이 참조할 전역 app
app = create_app()
```

> **주니어에게 전달 포인트**
>
> * 항상 `create_app()` 패턴을 쓰자. 테스트에서 `create_app()`을 불러 TestClient를 만들기 좋다.
> * 새로운 라우터를 만들면 여기서 `include_router`로 추가한다.

---

### 3-2. `backend/api/dependencies.py` (공통 의존성)

Heavy한 객체(모델 로딩, 인덱스 등)는 여기서 한 번 만들고 FastAPI DI로 주입한다.

```python
# backend/api/dependencies.py
from functools import lru_cache

from backend.llm_infrastructure.preprocessing import get_preprocessor
from backend.llm_infrastructure.embedding import get_embedder
from backend.llm_infrastructure.retrieval import get_retriever
from backend.llm_infrastructure.llm import get_llm

# 타입은 실제 프로젝트에 있는 Base 클래스 import해서 붙여도 됨
# 예: from backend.llm_infrastructure.embedding.base import BaseEmbedder


# ---- Preprocessor ----
@lru_cache
def get_default_preprocessor():
    # 기본값: "standard" 라는 이름의 전처리기를 사용한다고 가정
    return get_preprocessor("standard", version="v1")


# ---- Embedder ----
@lru_cache
def get_default_embedder():
    # 기본값: "bge_base" 임베딩 사용한다고 가정
    return get_embedder("bge_base", version="v1")


# ---- Retriever ----
@lru_cache
def get_default_retriever():
    # 기본값: "hybrid" v1 프리셋을 사용한다고 가정
    return get_retriever("hybrid", version="v1")


# ---- LLM ----
@lru_cache
def get_default_llm():
    # 기본값: vLLM 백엔드 사용한다고 가정
    return get_llm("vllm", version="v1")
```

> **포인트**
>
> * `@lru_cache`를 사용해서 인스턴스를 한 번만 만들고 재사용 → 모델/인덱스 로딩 비용 절감.
> * 나중에 환경 변수/설정 파일 기반으로 이름/버전 바꾸고 싶으면 이 함수 내부만 수정하면 된다.

---

### 3-3. 간단한 헬스체크 라우터

```python
# backend/api/routers/health.py
from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
async def health_check():
    return {"status": "ok"}
```

---

### 3-4. Preprocessing 라우터 골격

```python
# backend/api/routers/preprocessing.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.api.dependencies import get_default_preprocessor
# from backend.llm_infrastructure.preprocessing.base import BasePreprocessor

router = APIRouter(prefix="/preprocessing", tags=["Preprocessing"])


class PreprocessRequest(BaseModel):
    text: str


class PreprocessResponse(BaseModel):
    processed_text: str


@router.post("/apply", response_model=PreprocessResponse)
async def apply_preprocessing(
    body: PreprocessRequest,
    preprocessor = Depends(get_default_preprocessor),
):
    """
    단일 텍스트에 기본 전처리기를 적용.
    """
    processed = preprocessor.preprocess(body.text)
    return PreprocessResponse(processed_text=processed)
```

---

### 3-5. Embedding 라우터 골격

```python
# backend/api/routers/embedding.py
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.api.dependencies import get_default_embedder
# from backend.llm_infrastructure.embedding.base import BaseEmbedder

router = APIRouter(prefix="/embedding", tags=["Embedding"])


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    method: str
    embedding: List[float]


@router.post("/embed", response_model=EmbedResponse)
async def embed_text(
    body: EmbedRequest,
    embedder = Depends(get_default_embedder),
):
    """
    단일 텍스트를 벡터로 변환.
    """
    vector = embedder.embed(body.text)
    # numpy array라면 .tolist() 필요
    return EmbedResponse(method="default", embedding=list(vector))
```

---

### 3-6. Retrieval 라우터 골격

```python
# backend/api/routers/retrieval.py
from typing import List, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.api.dependencies import get_default_retriever

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievalResultModel(BaseModel):
    doc_id: str
    score: float
    content: str
    metadata: dict = {}


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResultModel]


@router.post("/search", response_model=RetrievalResponse)
async def search(
    body: RetrievalRequest,
    retriever = Depends(get_default_retriever),
):
    """
    쿼리에 대해 top_k 문서를 검색.
    """
    results = retriever.retrieve(body.query, top_k=body.top_k)
    # results가 내부적으로 dataclass라면 dict로 변환 필요
    normalized = [
        RetrievalResultModel(
            doc_id=r.doc_id,
            score=r.score,
            content=r.content,
            metadata=getattr(r, "metadata", {}) or {},
        )
        for r in results
    ]
    return RetrievalResponse(query=body.query, results=normalized)
```

---

### 3-7. LLM 라우터 골격 (단순 Chat)

```python
# backend/api/routers/llm.py
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.api.dependencies import get_default_llm

router = APIRouter(prefix="/llm", tags=["LLM"])


class ChatMessage(BaseModel):
    role: str  # "user" / "system" / "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    content: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    llm = Depends(get_default_llm),
):
    """
    단순 Chat API.
    ChatService가 이미 있다면 여기서 ChatService를 DI로 받아도 됨.
    """
    # LLM 모듈의 generate() 인터페이스에 맞춰 호출
    response = llm.generate(messages=[m.dict() for m in body.messages])
    # response가 LLMResponse라면 response.text 또는 response.content 등 사용
    return ChatResponse(content=response.text)
```

---

## 4. 테스트 전략 & 가이드

"이 구조에서 테스트는 어떻게 할까?"에 대한 실무 가이드입니다.
(주니어가 그대로 따라할 수 있도록 단계별로 작성)

---

### 4-1. 테스트 디렉토리 구조

```bash
tests/
├── unit/          # 순수 파이썬/비즈니스 로직 단위 테스트
│   ├── test_preprocessing_standard.py
│   ├── test_embedding_registry.py
│   ├── test_retrieval_presets.py
│   └── test_llm_vllm_adapter.py
└── api/           # FastAPI 엔드포인트 테스트
    ├── conftest.py
    ├── test_health.py
    ├── test_preprocessing_api.py
    ├── test_embedding_api.py
    ├── test_retrieval_api.py
    └── test_llm_api.py
```

* **unit/**: `llm_infrastructure` 내부 기능들을 직접 호출해 검증
* **api/**: `fastapi.testclient.TestClient`로 HTTP 레벨 테스트

---

### 4-2. 공통 TestClient fixture (`tests/api/conftest.py`)

```python
# tests/api/conftest.py
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)
```

> 필요하면 이 안에서 `dependency_overrides`로 DI를 교체(모킹)할 수도 있음. 아래에서 예시.

---

### 4-3. 헬스체크 API 테스트 예시

```python
# tests/api/test_health.py
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
```

---

### 4-4. Preprocessing API 테스트 예시

```python
# tests/api/test_preprocessing_api.py
def test_preprocessing_apply(client):
    payload = {"text": " Hello   World "}
    resp = client.post("/preprocessing/apply", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert "processed_text" in data
    # 표준 전처리가 공백 정리/trim 한다는 가정
    assert data["processed_text"] == "Hello World"
```

---

### 4-5. Embedding API 테스트 예시 (간단 버전)

실제 모델 로딩은 무거우므로, 간단한 smoke test 정도만 하고,
정말 무거운 모델은 유닛 테스트에서 "가짜 어댑터"로 대체하는 것을 권장.

```python
# tests/api/test_embedding_api.py
def test_embedding_embed(client):
    payload = {"text": "hello world"}
    resp = client.post("/embedding/embed", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["method"] == "default"
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) > 0
```

---

### 4-6. 의존성 오버라이드(Mocking) 예시

실제 SentenceTransformer나 vLLM을 테스트마다 로딩하면 너무 느리므로,
테스트에서는 가짜 클래스로 대체하는 패턴을 추천.

#### 예: Embedding용 FakeEmbedder

```python
# tests/api/conftest.py
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api import dependencies


class FakeEmbedder:
    def embed(self, text: str):
        # 단순하게 길이를 기반으로 벡터를 만드는 가짜 구현
        return [float(len(text)), 1.0, 2.0]


def override_get_default_embedder():
    return FakeEmbedder()


@pytest.fixture
def client():
    app = create_app()

    # DI 오버라이드: 실제 embedder 대신 FakeEmbedder 사용
    app.dependency_overrides[dependencies.get_default_embedder] = override_get_default_embedder

    yield TestClient(app)
```

이렇게 하면:

* 실제 모델은 로딩되지 않고,
* API/라우팅 로직만 빠르게 테스트할 수 있음.

---

### 4-7. 유닛 테스트 예시 (전처리 모듈)

```python
# tests/unit/test_preprocessing_standard.py
from backend.llm_infrastructure.preprocessing import get_preprocessor


def test_standard_preprocessor_basic():
    pre = get_preprocessor("standard", version="v1")

    text = "  Hello   WORLD  "
    out = pre.preprocess(text)

    # 예시 검증: 트림 + 중복 공백 제거 + 소문자화 등
    assert out == "hello world"
```

---

### 4-8. 유닛 테스트 예시 (Retrieval presets)

```python
# tests/unit/test_retrieval_presets.py
from backend.llm_infrastructure.retrieval import get_retriever


def test_hybrid_retriever_returns_results():
    retriever = get_retriever("hybrid", version="v1")

    results = retriever.retrieve("test query", top_k=3)

    assert len(results) <= 3
    # 결과 타입/필드 검증
    for r in results:
        assert hasattr(r, "doc_id")
        assert hasattr(r, "score")
        assert hasattr(r, "content")
```

---

### 4-9. LLM API 테스트 예시

LLM도 마찬가지로 실제 vLLM 대신 가짜 구현을 쓰는 것이 좋다.

```python
# tests/api/conftest.py (일부 추가)
from backend.api import dependencies


class FakeLLM:
    def generate(self, messages):
        # 가장 마지막 user 메시지의 content를 그대로 echo
        last_user = [m for m in messages if m["role"] == "user"][-1]
        class Resp:
            text = f"echo: {last_user['content']}"
        return Resp()


def override_get_default_llm():
    return FakeLLM()


@pytest.fixture
def client():
    app = create_app()

    app.dependency_overrides[dependencies.get_default_embedder] = override_get_default_embedder
    app.dependency_overrides[dependencies.get_default_llm] = override_get_default_llm

    yield TestClient(app)
```

```python
# tests/api/test_llm_api.py
def test_llm_chat(client):
    payload = {
        "messages": [
            {"role": "user", "content": "안녕"},
        ]
    }
    resp = client.post("/llm/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"].startswith("echo:")
```

---

### 4-10. 실행 & CI 힌트

* 로컬에서 테스트 실행:

```bash
pytest -q
```

* 나중에 GitHub Actions 등 CI에 붙일 때는:
  * python 세팅 후 `pip install -r requirements.txt`
  * `pytest` 실행
  * 필요하면 `pytest -m "not slow"` 같은 마커로 heavy 테스트는 분리

---

## 5. 주니어 개발자에게 전달할 핵심 요약

### 5-1. FastAPI 구조

* `backend/api/main.py`에 `create_app()`과 `app` 정의
* `backend/api/routers/` 아래에 기능별 라우터 생성
* `backend/api/dependencies.py`에서 heavy한 객체를 한 번만 만들고 `Depends`로 주입

### 5-2. 모듈 연결

* 전처리/임베딩/리트리벌/LLM 로직은 **절대 FastAPI 라우터 안에서 새로 만들지 말 것**
* 반드시 `llm_infrastructure` 모듈의 `get_xxx()` (registry) 기반으로 가져와서 사용

### 5-3. 테스트

* `tests/unit/`: 인프라 모듈 직접 테스트
* `tests/api/`: FastAPI 엔드포인트에 대해 TestClient로 테스트
* 무거운 의존성(vLLM, SentenceTransformer 등)은 **Fake 클래스 + dependency_overrides**로 모킹

---

## 6. 구현 순서 (주니어 개발자용 체크리스트)

### Phase 1: 기본 구조 생성
- [ ] `backend/api/` 디렉토리 생성
- [ ] `backend/api/__init__.py` 생성
- [ ] `backend/api/main.py` 작성 (기본 FastAPI 앱)
- [ ] `backend/api/dependencies.py` 작성 (빈 파일로 시작)
- [ ] `backend/api/routers/` 디렉토리 생성
- [ ] `backend/api/routers/__init__.py` 생성

### Phase 2: Health Check 구현 및 테스트
- [ ] `backend/api/routers/health.py` 작성
- [ ] `tests/api/` 디렉토리 생성
- [ ] `tests/api/conftest.py` 작성 (TestClient fixture)
- [ ] `tests/api/test_health.py` 작성
- [ ] 테스트 실행 확인: `pytest tests/api/test_health.py -v`

### Phase 3: Preprocessing API 구현
- [ ] `backend/api/dependencies.py`에 `get_default_preprocessor()` 추가
- [ ] `backend/api/routers/preprocessing.py` 작성
- [ ] `tests/api/test_preprocessing_api.py` 작성
- [ ] 테스트 실행 및 검증

### Phase 4: Embedding API 구현
- [ ] `backend/api/dependencies.py`에 `get_default_embedder()` 추가
- [ ] `backend/api/routers/embedding.py` 작성
- [ ] `tests/api/conftest.py`에 FakeEmbedder 추가
- [ ] `tests/api/test_embedding_api.py` 작성
- [ ] 테스트 실행 및 검증

### Phase 5: Retrieval API 구현
- [ ] `backend/api/dependencies.py`에 `get_default_retriever()` 추가
- [ ] `backend/api/routers/retrieval.py` 작성
- [ ] `tests/api/test_retrieval_api.py` 작성
- [ ] 테스트 실행 및 검증

### Phase 6: LLM API 구현
- [ ] `backend/api/dependencies.py`에 `get_default_llm()` 추가
- [ ] `backend/api/routers/llm.py` 작성
- [ ] `tests/api/conftest.py`에 FakeLLM 추가
- [ ] `tests/api/test_llm_api.py` 작성
- [ ] 테스트 실행 및 검증

### Phase 7: 통합 테스트 및 문서화
- [ ] 모든 테스트 통합 실행: `pytest tests/api/ -v`
- [ ] API 문서 확인: `http://localhost:8000/docs` (Swagger UI)
- [ ] 개발 서버 실행 테스트: `uvicorn backend.api.main:app --reload`

---

## 7. 실행 방법

### 개발 서버 실행

```bash
# 기본 실행 (포트 8000)
uvicorn backend.api.main:app --reload

# 커스텀 포트로 실행
uvicorn backend.api.main:app --reload --port 8080

# 외부 접속 허용
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### API 문서 확인

서버 실행 후 브라우저에서:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 테스트 실행

```bash
# 전체 테스트 실행
pytest

# API 테스트만 실행
pytest tests/api/

# 특정 파일 테스트
pytest tests/api/test_health.py -v

# 커버리지와 함께 실행
pytest --cov=backend/api tests/api/
```

---

## 8. 트러블슈팅

### 문제 1: Import Error
**증상**: `ModuleNotFoundError: No module named 'backend'`

**해결**:
```bash
# 프로젝트 루트에서 실행
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 또는 개발 모드로 설치
pip install -e .
```

### 문제 2: 의존성 로딩 실패
**증상**: `get_preprocessor()` 등에서 에러 발생

**해결**:
- `backend/api/dependencies.py`에서 실제 사용 가능한 이름/버전 확인
- 레지스트리에 해당 이름이 등록되어 있는지 확인
- 필요시 실제 구현이 완료될 때까지 Fake 클래스 사용

### 문제 3: 테스트에서 모델 로딩 시간 초과
**증상**: 테스트가 너무 오래 걸림

**해결**:
- `conftest.py`에서 `dependency_overrides` 사용
- 실제 모델 대신 Fake 클래스로 교체
- Slow test에 `@pytest.mark.slow` 마커 추가하고 CI에서 분리

---

## 9. 다음 단계

이 문서의 구조를 구현한 후:

1. **RAG Pipeline API 추가**: 전처리 → 임베딩 → 검색 → LLM 생성을 한 번에 수행하는 엔드포인트
2. **문서 인덱싱 API**: 새로운 문서를 추가/업데이트하는 엔드포인트
3. **인증/권한**: JWT 기반 인증 미들웨어 추가
4. **Rate Limiting**: API 호출 제한 구현
5. **로깅/모니터링**: 구조화된 로깅 및 메트릭 수집
6. **배포 준비**: Docker 이미지 빌드 및 프로덕션 설정

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
- `backend/llm_infrastructure/embedding/README.md`: 임베딩 모듈 가이드 (있다면)
- `STRUCTURE.txt`: 프로젝트 전체 구조
