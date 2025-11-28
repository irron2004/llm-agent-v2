from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.retrieval.adapters.bm25 import BM25Retriever  # noqa: E402
from backend.llm_infrastructure.retrieval.adapters.dense import DenseRetriever  # noqa: E402
from backend.llm_infrastructure.retrieval.adapters.hybrid import (  # noqa: E402
    HybridRetriever,
)
from backend.services.document_service import (  # noqa: E402
    DocumentIndexService,
    SourceDocument,
)


class _FakeEmbedder:
    """Deterministic tiny embedder for tests."""

    def _encode(self, texts):
        vectors = []
        for text in texts:
            vectors.append(
                np.array(
                    [
                        text.count("apple"),
                        text.count("banana"),
                    ],
                    dtype=np.float32,
                )
            )
        return np.vstack(vectors)

    def embed(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def embed_batch(self, texts):
        return self._encode(texts)


def _build_corpus(tmp_path: Path):
    docs = [
        SourceDocument(doc_id="d1", text="apple banana"),
        SourceDocument(doc_id="d2", text="banana banana"),
        SourceDocument(doc_id="d3", text="apple"),
    ]
    svc = DocumentIndexService(embedder=_FakeEmbedder())
    return svc.index(docs, preprocess=False, persist_dir=tmp_path)


def test_dense_retrieval_prefers_semantic_match(tmp_path):
    corpus = _build_corpus(tmp_path)
    dense = DenseRetriever(
        vector_store=corpus.vector_store,
        embedder=_FakeEmbedder(),
        top_k=2,
    )
    results = dense.retrieve("apple banana")
    assert results[0].doc_id == "d1"
    assert len(results) == 2


def test_hybrid_rrf_fuses_dense_and_sparse(tmp_path):
    corpus = _build_corpus(tmp_path)
    dense = DenseRetriever(
        vector_store=corpus.vector_store,
        embedder=_FakeEmbedder(),
        top_k=2,
    )
    sparse = BM25Retriever(corpus.bm25_index, top_k=2)
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        dense_weight=0.8,
        sparse_weight=0.2,
        rrf_k=10,
        top_k=2,
    )
    fused = hybrid.retrieve("apple banana")
    assert {r.doc_id for r in fused} == {"d1", "d2"}
    assert fused[0].doc_id in {"d1", "d2"}


def test_vector_store_persistence_roundtrip(tmp_path):
    corpus = _build_corpus(tmp_path)
    loaded = DocumentIndexService.load(tmp_path)

    dense = DenseRetriever(
        vector_store=loaded.vector_store,
        embedder=_FakeEmbedder(),
        top_k=1,
    )
    again = dense.retrieve("banana")
    assert loaded.vector_store.size == 3
    assert again[0].doc_id in {"d1", "d2"}


def test_search_service_hybrid(monkeypatch, tmp_path):
    corpus = _build_corpus(tmp_path)

    class _StubEmbeddingService:
        def __init__(self, **_: object) -> None:
            self._e = _FakeEmbedder()

        def get_raw_embedder(self):
            return self._e

    # SearchService 내부에서 EmbeddingService를 스텁으로 대체
    monkeypatch.setattr(
        "backend.services.search_service.EmbeddingService",
        _StubEmbeddingService,
    )

    from backend.services.search_service import SearchService

    svc = SearchService(corpus, method="hybrid", top_k=2)
    results = svc.search("apple banana")
    assert results
    assert {r.doc_id for r in results} == {"d1", "d2"}
    assert results[0].doc_id in {"d1", "d2"}
