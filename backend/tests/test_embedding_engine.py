"""Tests for embedding engines and adapters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Fakes for external deps -------------------------------------------------


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        arr = np.array([[1.0, 2.0]] * len(texts), dtype=np.float32)
        return arr

    def get_sentence_embedding_dimension(self) -> int:
        return 2


class _FakeDiskCache:
    def __init__(self, *_args, **_kwargs):
        self._store = {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHTTPXClient:
    def __init__(self, *_, **__):
        pass

    def post(self, _url, json):
        inputs = json["inputs"]
        if isinstance(inputs, str):
            payload = [0.1, 0.2]
        else:
            payload = [[0.1, 0.2] for _ in inputs]
        return _FakeHTTPResponse(payload)

    def close(self):
        pass


def _install_fakes() -> None:
    """Install lightweight fake modules before importing code under test."""
    # sentence-transformers
    fake_st_mod = type(sys)("sentence_transformers")
    fake_st_mod.SentenceTransformer = _FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_st_mod

    # diskcache
    fake_dc_mod = type(sys)("diskcache")
    fake_dc_mod.Cache = _FakeDiskCache
    sys.modules["diskcache"] = fake_dc_mod

    # httpx
    fake_httpx = type(sys)("httpx")
    fake_httpx.Client = _FakeHTTPXClient
    sys.modules["httpx"] = fake_httpx

    # torch (optional): default to unavailable
    if "torch" not in sys.modules:
        fake_torch = type(sys)("torch")
        sys.modules["torch"] = fake_torch
    torch_mod = sys.modules["torch"]
    fake_cuda = type("cuda", (), {})()
    fake_cuda.is_available = staticmethod(lambda: False)
    fake_cuda.mem_get_info = staticmethod(lambda idx: (0, 0))
    fake_cuda.device_count = staticmethod(lambda: 0)
    torch_mod.cuda = fake_cuda


_install_fakes()


# --- Imports after fakes -----------------------------------------------------
from backend.llm_infrastructure.embedding.engines.sentence import create_embedder  # noqa: E402
from backend.llm_infrastructure.embedding.engines.sentence.utils import pick_device  # noqa: E402
from backend.llm_infrastructure.embedding.registry import get_embedder  # noqa: E402
from backend.llm_infrastructure.embedding.adapters.sentence import SentenceEmbedderAdapter  # noqa: E402
from backend.llm_infrastructure.embedding.adapters.tei import TEIEmbedder  # noqa: E402


def test_pick_device_cpu_when_no_cuda():
    device = pick_device("auto")
    assert device == "cpu"


def test_pick_device_round_robin(monkeypatch):
    # fake torch with 2 cuda devices
    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def mem_get_info(idx):
            return (1024, 0)

    fake_torch = type(sys)("torch")
    fake_torch.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # reload pick_device with fake torch
    from importlib import reload
    from backend.llm_infrastructure.embedding.engines import sentence
    reload(sentence.utils)
    from backend.llm_infrastructure.embedding.engines.sentence.utils import pick_device as pick_device_rr

    d1 = pick_device_rr("round-robin")
    d2 = pick_device_rr("round-robin")
    assert {d1, d2} == {"cuda:0", "cuda:1"}


def test_sentence_engine_encode_and_cache(tmp_path):
    cache_dir = tmp_path / "emb_cache"
    embedder = create_embedder(
        typ="sentence",
        model_name="fake/model",
        device="cpu",
        use_cache=True,
        cache_dir=str(cache_dir),
        normalize_embeddings=True,
    )

    vecs1 = embedder.encode(["hello", "world"])
    vecs2 = embedder.encode(["hello", "world"])  # should hit cache

    assert vecs1.shape == (2, 2)
    assert np.allclose(vecs1, vecs2)
    # L2 normalized
    norms = np.linalg.norm(vecs1, axis=1)
    assert np.allclose(norms, 1.0)


def test_sentence_adapter_via_registry():
    emb = get_embedder("bge_base", version="v1", device="cpu", use_cache=False)
    assert isinstance(emb, SentenceEmbedderAdapter)
    vec = emb.embed("text")
    assert vec.shape == (2,)
    vecs = emb.embed_batch(["a", "b"])
    assert vecs.shape == (2, 2)


def test_tei_adapter_mocked():
    emb = get_embedder("tei", version="v1", endpoint_url="http://fake")
    assert isinstance(emb, TEIEmbedder)
    vec = emb.embed("hi")
    assert vec.shape == (2,)
    vecs = emb.embed_batch(["x", "y"])
    assert vecs.shape == (2, 2)


def test_alias_mapping_and_override():
    # default alias → default model
    emb_default = get_embedder("koe5", version="v1", device="cpu")
    assert emb_default.config["model_name"] == "nlpai-lab/KoE5"

    # override model_name
    emb_custom = get_embedder(
        "bge_base",
        version="v1",
        device="cpu",
        model_name="custom/model",
    )
    assert emb_custom.config["model_name"] == "custom/model"

    # unknown alias → ValueError (설정 오류 조기 발견)
    with pytest.raises(ValueError, match="Unknown embedding method"):
        get_embedder("unknown_alias", version="v1", device="cpu")


def test_encode_convenience():
    emb = get_embedder("bge_base", version="v1", device="cpu")
    vecs = emb.encode(["a", "b"])
    assert vecs.shape == (2, 2)


def test_tei_errors(monkeypatch):
    # missing endpoint_url
    with pytest.raises(ValueError):
        get_embedder("tei", version="v1")

    # simulate HTTP error
    class _ErrResponse:
        def raise_for_status(self):
            raise RuntimeError("http error")

        def json(self):
            return []

    class _ErrClient(_FakeHTTPXClient):
        def post(self, *_args, **_kwargs):
            return _ErrResponse()

    fake_httpx = type(sys)("httpx")
    fake_httpx.Client = _ErrClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    emb = get_embedder("tei", version="v1", endpoint_url="http://fake")
    with pytest.raises(RuntimeError):
        emb.embed("x")
