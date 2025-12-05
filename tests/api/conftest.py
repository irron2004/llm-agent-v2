from typing import Callable, Iterable

import pytest
from fastapi.testclient import TestClient

from backend.api import dependencies
from backend.api.main import create_app
from backend.llm_infrastructure.preprocessing.base import BasePreprocessor


class FakePreprocessor(BasePreprocessor):
    def preprocess(self, docs: Iterable[str]):
        for doc in docs:
            text = str(doc).strip()
            if not text:
                continue
            yield " ".join(text.split())


def override_get_default_preprocessor():
    return FakePreprocessor()


def override_get_preprocessor_factory() -> Callable[[str | None], FakePreprocessor]:
    def factory(level: str | None = None):
        return FakePreprocessor()

    return factory


@pytest.fixture
def client():
    app = create_app()
    app.dependency_overrides[dependencies.get_default_preprocessor] = (
        override_get_default_preprocessor
    )
    app.dependency_overrides[dependencies.get_preprocessor_factory] = (
        override_get_preprocessor_factory
    )
    with TestClient(app) as test_client:
        yield test_client
