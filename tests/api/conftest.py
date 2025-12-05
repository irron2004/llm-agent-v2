from pathlib import Path
from typing import Callable, Iterable, List

import sys

import pytest
from fastapi.testclient import TestClient

# 프로젝트 루트를 PYTHONPATH에 추가 (백엔드 모듈 임포트용)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.api import dependencies
from backend.api.main import create_app
from backend.llm_infrastructure.preprocessing.base import BasePreprocessor
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.services.rag_service import RAGResponse


class _FakeLLMResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakePreprocessor(BasePreprocessor):
    def preprocess(self, docs: Iterable[str]):
        for doc in docs:
            text = str(doc).strip()
            if not text:
                continue
            yield " ".join(text.split())


class FakeSearchService:
    def __init__(self) -> None:
        self._results = [
            RetrievalResult(
                doc_id="doc-1",
                content="PM 점검 절차 요약",
                score=0.92,
                metadata={"title": "PM 점검 가이드"},
                raw_text="PM 점검 절차 요약",
            ),
            RetrievalResult(
                doc_id="doc-2",
                content="정비 주기와 체크리스트",
                score=0.85,
                metadata={"title": "정비 체크리스트"},
                raw_text="정비 주기와 체크리스트",
            ),
            RetrievalResult(
                doc_id="doc-3",
                content="안전 수칙",
                score=0.75,
                metadata={"title": "안전 수칙"},
                raw_text="안전 수칙",
            ),
        ]

    def search(self, query: str, top_k: int = 10):
        # 단순히 상위 top_k 반환 (정렬된 상태 가정)
        return self._results[:top_k]


class FakeRAGService:
    def __init__(self) -> None:
        self._context = [
            RetrievalResult(
                doc_id="doc-10",
                content="RAG 문서 컨텍스트",
                score=0.9,
                metadata={"title": "문서 컨텍스트"},
                raw_text="RAG 문서 컨텍스트",
            )
        ]

    def query(self, question: str, top_k: int = 3, history: List[dict] | None = None, **_: object):
        return RAGResponse(
            answer="테스트 응답",
            context=self._context[:top_k],
            question=question,
            metadata={
                "preprocessed_query": f"{question} (clean)",
                "history_len": len(history or []),
            },
        )


class FakeChatService:
    def __init__(self) -> None:
        self.last_system_prompt = None

    def chat(self, user_message: str, *, history: List[dict] | None = None, system_prompt=None, **_: object):
        self.last_system_prompt = system_prompt
        return _FakeLLMResponse(text=f"응답: {user_message}; prompt={system_prompt or 'NONE'}")


def override_get_default_preprocessor():
    return FakePreprocessor()


def override_get_preprocessor_factory() -> Callable[[str | None], FakePreprocessor]:
    def factory(level: str | None = None):
        return FakePreprocessor()

    return factory


def override_get_search_service():
    return FakeSearchService()


def override_get_rag_service():
    return FakeRAGService()


def override_get_chat_service():
    return FakeChatService()


def override_get_simple_chat_prompt():
    return "SIMPLE_PROMPT"


@pytest.fixture
def client():
    app = create_app()
    app.dependency_overrides[dependencies.get_default_preprocessor] = (
        override_get_default_preprocessor
    )
    app.dependency_overrides[dependencies.get_preprocessor_factory] = (
        override_get_preprocessor_factory
    )
    app.dependency_overrides[dependencies.get_search_service] = (
        override_get_search_service
    )
    app.dependency_overrides[dependencies.get_rag_service] = (
        override_get_rag_service
    )
    app.dependency_overrides[dependencies.get_chat_service] = (
        override_get_chat_service
    )
    app.dependency_overrides[dependencies.get_simple_chat_prompt] = (
        override_get_simple_chat_prompt
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_minimal():
    """의존성 오버라이드 없이 사용하는 최소 클라이언트."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
