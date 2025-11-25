# Embedding 코드 마이그레이션 가이드

llm-agent → llm-agent-v2 임베딩 모듈 마이그레이션 가이드입니다.

## 📋 목차

1. [전체 구조 비교](#전체-구조-비교)
2. [마이그레이션 매핑](#마이그레이션-매핑)
3. [권장 구조](#권장-구조)
4. [단계별 마이그레이션](#단계별-마이그레이션)
5. [코드 예시](#코드-예시)

## 🏗️ 전체 구조 비교

### llm-agent (원본)

```
core/embedding/
├── embedders/
│   ├── base.py                 # BaseEmbedder (encode 메소드)
│   ├── sentence.py             # SentenceTransformer (싱글톤, L2 정규화)
│   ├── cache.py                # 디스크 캐싱 래퍼
│   └── create_embedder.py      # 팩토리 함수
├── utils/
│   ├── device.py               # GPU 자동 선택 (auto, round-robin)
│   ├── normalize.py            # L2 정규화
│   └── chunking.py             # 텍스트 청킹
├── config/
│   └── settings.py             # EmbeddingSettings (Pydantic)
├── adapters/
│   └── langchain.py            # LangChain 어댑터
├── indexing.py                 # FaissIndex
└── cli.py                      # CLI 인터페이스
```

**특징**:
- 싱글톤 패턴으로 모델 재사용
- 디스크 캐싱 지원
- GPU 자동 선택 (여유 메모리 기반, round-robin)
- LangChain 통합
- FAISS 인덱싱

### llm-agent-v2 (현재 - 스켈레톤)

```
backend/llm_infrastructure/embedding/
├── embedders/
│   ├── sentence_transformer.py  # BGE, E5 등 (레지스트리 패턴)
│   └── tei_client.py            # TEI 클라이언트
├── base.py                      # BaseEmbedder (embed, embed_batch)
└── registry.py                  # EmbedderRegistry
```

**특징**:
- 레지스트리 패턴 (preprocessing과 동일)
- 여러 모델 지원 (bge_base, bge_large, multilingual_e5)
- TEI 클라이언트 포함
- 유틸리티 함수 없음 (추가 필요)

## 🎯 마이그레이션 매핑

### 핵심 모듈 매핑

| llm-agent | llm-agent-v2 (목표) | 설명 |
|-----------|---------------------|------|
| `embedders/base.py` | `engines/sentence/base.py` | 엔진 내부용 BaseEmbedder |
| `embedders/sentence.py` | `engines/sentence/embedder.py` | SentenceTransformer 엔진 |
| `embedders/cache.py` | `engines/sentence/cache.py` | 디스크 캐싱 |
| `embedders/create_embedder.py` | `engines/sentence/factory.py` | 팩토리 함수 |
| `utils/device.py` | `engines/sentence/utils.py` | GPU 선택 로직 |
| `utils/normalize.py` | `engines/sentence/utils.py` | L2 정규화 |
| `utils/chunking.py` | `engines/sentence/utils.py` | 텍스트 청킹 |
| `config/settings.py` | `backend/config/settings.py` | ✅ 이미 존재 |
| `adapters/langchain.py` | `adapters/langchain.py` | LangChain 어댑터 |
| `indexing.py` | `indexing/faiss_index.py` | FAISS 인덱스 (선택) |
| (없음) | `base.py` | ✅ 레지스트리용 BaseEmbedder |
| (없음) | `adapters/sentence.py` | ✅ 레지스트리 어댑터 |

### 메소드 인터페이스 차이

| llm-agent | llm-agent-v2 | 변경 필요 |
|-----------|--------------|-----------|
| `encode(texts)` → ndarray | `embed(text)` → ndarray | ✅ 단일 텍스트용 |
| `encode(texts)` → ndarray | `embed_batch(texts)` → ndarray | ✅ 배치용 |
| `encode_query(text)` → ndarray | `embed(text)` → ndarray | ✅ 통합됨 |

## 🏛️ 권장 구조

preprocessing과 동일하게 **엔진-어댑터 패턴** 적용:

```
backend/llm_infrastructure/embedding/
├── engines/                  # 🔧 엔진: 실제 알고리즘 구현
│   ├── __init__.py
│   └── sentence/             # SentenceTransformer 엔진
│       ├── __init__.py       #    → create_embedder 등 재export
│       ├── base.py           #    → BaseEmbedder (엔진용)
│       ├── embedder.py       #    → SentenceTransformer 래퍼
│       ├── cache.py          #    → 디스크 캐싱
│       ├── utils.py          #    → device, normalize, chunking
│       └── factory.py        #    → create_embedder 팩토리
│
├── adapters/                 # 🔌 어댑터: 레지스트리 인터페이스
│   ├── __init__.py          #    → 모든 어댑터 재export
│   ├── sentence.py           #    → SentenceTransformer 어댑터
│   ├── tei.py                #    → TEI 어댑터
│   └── langchain.py          #    → LangChain 어댑터 (선택)
│
├── indexing/                 # 📊 인덱싱 (선택사항)
│   ├── __init__.py
│   └── faiss_index.py        #    → FAISS IVF 래퍼
│
├── base.py                   # BaseEmbedder (레지스트리용)
└── registry.py               # EmbedderRegistry
```

**중요**: `engines/sentence/base.py`는 엔진 내부용, `embedding/base.py`는 레지스트리용

### 설계 원칙

1. **엔진 (`*_engine/`)**: 순수 알고리즘 로직
   - SentenceTransformer 래핑
   - GPU 자동 선택
   - 캐싱, 정규화 등 유틸
   - 레지스트리 없이 독립적으로 사용 가능

2. **어댑터 (`adapters/`)**: 레지스트리 연결
   - 엔진을 레지스트리에 등록
   - 설정 주입
   - 메타데이터 처리

3. **인덱싱 (`indexing/`)**: 벡터 인덱스 (선택)
   - FAISS, Qdrant 등
   - 검색에 사용

## 📦 단계별 마이그레이션

### 📌 주의사항 (시작 전 필독)

#### 1. 네임스페이스 정리
- ❌ `core.embedding...` → ✅ `backend.llm_infrastructure.embedding...`
- 모든 import 경로 변경
- 각 패키지에 `__init__.py` 추가 필수

#### 2. GPU 선택 로직 안정성
- `auto` 전략: 메모리 조회 실패 시 CPU fallback 추가
- Docker/컨테이너 환경: `CUDA_VISIBLE_DEVICES` 고려
- 멀티프로세스: round-robin이 프로세스 간 동기화 필요

#### 3. Docker 환경 고려
- 캐시 경로: 쓰기 가능한 볼륨 마운트 필요
- `.dockerignore`에 캐시 디렉토리 추가
- 환경변수로 모든 경로 제어 가능하게

#### 4. 의존성
```bash
# pyproject.toml or requirements.txt
sentence-transformers>=2.2.0
diskcache>=5.4.0
faiss-cpu>=1.7.0  # 또는 faiss-gpu
langchain>=0.1.0  # 선택사항
```

#### 5. 기존 v2 임베딩 코드
- `embedders/sentence_transformer.py` → **무시** (덮어쓰기)
- `embedders/tei_client.py` → `adapters/tei.py`로 이동
- `base.py`, `registry.py` → **유지**

---

### Step 1: 엔진 유틸리티 마이그레이션

**파일**: `backend/llm_infrastructure/embedding/engines/sentence/utils.py`

```python
"""Utility functions for SentenceTransformer engine."""

import torch
import numpy as np
from typing import List

# ==================== Device Selection ====================

_gpu_cycle = -1

def pick_device(strategy: str | None = None) -> str:
    """
    GPU 자동 선택 전략 (Docker/컨테이너 안전).

    Args:
        strategy: None/"auto" (여유 메모리 기준)
                  "round-robin" (순환)
                  "cuda:X" (직접 지정)

    Returns:
        Device string (e.g., "cuda:0", "cpu")
    """
    if strategy is None or strategy == "auto":
        if torch.cuda.is_available():
            try:
                # 가장 여유 있는 GPU 선택
                free_mem = [
                    torch.cuda.mem_get_info(i)[0]
                    for i in range(torch.cuda.device_count())
                ]
                best_gpu = int(max(range(len(free_mem)), key=free_mem.__getitem__))
                return f"cuda:{best_gpu}"
            except Exception:
                # 메모리 조회 실패 시 첫 GPU 사용
                return "cuda:0"
        return "cpu"

    if strategy == "round-robin":
        global _gpu_cycle
        if torch.cuda.is_available():
            _gpu_cycle = (_gpu_cycle + 1) % torch.cuda.device_count()
            return f"cuda:{_gpu_cycle}"
        return "cpu"

    # 직접 지정 (CUDA_VISIBLE_DEVICES 고려)
    return strategy

# ==================== Normalization ====================

def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """L2 정규화."""
    return x / np.linalg.norm(x, axis=axis, keepdims=True)

# ==================== Chunking ====================

def split_by_tokens(
    text: str,
    max_tokens: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    텍스트를 토큰 길이 기준으로 청킹.

    Args:
        text: 입력 텍스트
        max_tokens: 최대 토큰 수
        overlap: 오버랩 토큰 수

    Returns:
        청크 리스트
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    step = max_tokens - overlap
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks
```

### Step 2: 엔진 BaseEmbedder 마이그레이션

**파일**: `backend/llm_infrastructure/embedding/engines/sentence/base.py`

```python
"""Base embedder for SentenceTransformer engine (internal use)."""

from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np

class BaseEmbedder(ABC):
    """
    엔진 내부용 BaseEmbedder (llm-agent 호환).

    주의: 레지스트리용 BaseEmbedder(`embedding/base.py`)와 다름.
    """

    @abstractmethod
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """
        배치 임베딩.

        Args:
            texts: 텍스트 리스트

        Returns:
            임베딩 행렬 (n_texts, dimension)
        """
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """
        단일 쿼리 임베딩.

        Args:
            text: 단일 텍스트

        Returns:
            임베딩 벡터 (dimension,)
        """
        return self.encode([text])[0]
```

### Step 3: 캐싱 유틸리티 마이그레이션

**파일**: `backend/llm_infrastructure/embedding/engines/sentence/cache.py`

```python
"""Disk caching for embeddings."""

import hashlib
import numpy as np
from diskcache import Cache
from typing import List

class CachedEmbedder:
    """
    임베딩 결과를 디스크에 캐싱하는 래퍼.

    사용법:
        embedder = SentenceTransformerEmbedder(...)
        cached = CachedEmbedder(embedder, cache_dir=".embed_cache")
        vecs = cached.encode(texts)
    """

    def __init__(self, inner, cache_dir: str = ".embed_cache"):
        """
        Args:
            inner: BaseEmbedder 인스턴스
            cache_dir: 캐시 저장 디렉토리
        """
        self.inner = inner
        self.cache = Cache(cache_dir)

    def _key(self, text: str) -> str:
        """캐시 키 생성 (모델명 + 텍스트)."""
        model_name = getattr(self.inner, "model_name", "unknown")
        h = hashlib.sha256()
        h.update((model_name + "::" + text).encode())
        return h.hexdigest()

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        배치 임베딩 (캐시 활용).

        Args:
            texts: 텍스트 리스트

        Returns:
            임베딩 행렬 (n_texts, dimension)
        """
        vecs = []
        to_compute = []
        indices = []

        # 1) 캐시 확인
        for i, t in enumerate(texts):
            key = self._key(t)
            if key in self.cache:
                vecs.append(self.cache[key])
            else:
                to_compute.append(t)
                indices.append(i)
                vecs.append(None)

        # 2) 캐시 미스만 계산
        if to_compute:
            new_vecs = self.inner.encode(to_compute)
            for i, v in zip(indices, new_vecs):
                key = self._key(to_compute[indices.index(i)])
                self.cache[key] = v
                vecs[i] = v

        return np.vstack(vecs)

    def encode_query(self, text: str) -> np.ndarray:
        """단일 쿼리 임베딩."""
        return self.encode([text])[0]
```

### Step 4: SentenceTransformer 엔진 마이그레이션

**파일**: `backend/llm_infrastructure/embedding/engines/sentence/embedder.py`

```python
"""SentenceTransformer embedding engine."""

from typing import Iterable
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder
from .utils import pick_device, l2_normalize

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    SentenceTransformer 엔진 (싱글톤 패턴).

    특징:
    - 모델명 + 디바이스 조합으로 싱글톤 관리
    - L2 정규화 자동 적용
    - GPU 자동 선택 지원
    """

    _instance_cache: dict[str, "SentenceTransformerEmbedder"] = {}

    def __new__(cls, model_name: str, device: str | None = None, **kwargs):
        """싱글톤 패턴으로 모델 재사용."""
        key = f"{model_name}@{device}"
        if key not in cls._instance_cache:
            cls._instance_cache[key] = super().__new__(cls)
        return cls._instance_cache[key]

    def __init__(self, model_name: str, device: str | None = None, **kwargs):
        """
        Args:
            model_name: HuggingFace 모델 ID
            device: 디바이스 ("auto", "round-robin", "cuda:X", "cpu")
            **kwargs: SentenceTransformer 추가 인자
        """
        if hasattr(self, "_init_done"):
            return  # 싱글톤 재진입 방지

        self.model_name = model_name
        real_device = pick_device(device)
        self.device = real_device
        self.model = SentenceTransformer(
            model_name,
            device=real_device,
            trust_remote_code=True,
            **kwargs
        )
        self._init_done = True

        # E5 계열 모델 prefix 지원
        self.uses_e5_prefix = "e5" in model_name.lower()

    def encode(
        self,
        texts: Iterable[str],
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        배치 임베딩 (L2 정규화 포함).

        Args:
            texts: 텍스트 리스트
            show_progress_bar: 진행바 표시 여부

        Returns:
            L2 정규화된 임베딩 행렬
        """
        vecs = self.model.encode(
            list(texts),
            show_progress_bar=show_progress_bar
        )
        return l2_normalize(vecs)

    def encode_query(self, text: str) -> np.ndarray:
        """단일 쿼리 임베딩."""
        return self.encode([text])[0]

    def get_dimension(self) -> int:
        """임베딩 차원 반환."""
        return self.model.get_sentence_embedding_dimension()
```

### Step 5: 팩토리 함수 마이그레이션

**파일**: `backend/llm_infrastructure/embedding/engines/sentence/factory.py`

```python
"""Factory function for creating embedders."""

from typing import Literal
from .embedder import SentenceTransformerEmbedder
from .cache import CachedEmbedder

_EmbedderType = Literal["sentence", "openai"]

def create_embedder(
    typ: _EmbedderType = "sentence",
    model_name: str = "nlpai-lab/KoE5",
    device: str | None = None,
    use_cache: bool = False,
    cache_dir: str = ".embed_cache",
    **kwargs,
):
    """
    임베더 팩토리 함수.

    Args:
        typ: 임베더 타입 ("sentence", "openai")
        model_name: 모델명
        device: 디바이스
        use_cache: 디스크 캐싱 사용 여부
        cache_dir: 캐시 디렉토리
        **kwargs: 추가 인자

    Returns:
        임베더 인스턴스
    """
    if typ == "sentence":
        embedder = SentenceTransformerEmbedder(
            model_name,
            device=device,
            **kwargs
        )

        if use_cache:
            embedder = CachedEmbedder(embedder, cache_dir=cache_dir)

        return embedder

    raise ValueError(f"Unknown embedder type: {typ}")
```

### Step 6: 엔진 __init__.py

**파일**: `backend/llm_infrastructure/embedding/engines/sentence/__init__.py`

```python
"""SentenceTransformer embedding engine."""

from .embedder import SentenceTransformerEmbedder
from .cache import CachedEmbedder
from .factory import create_embedder
from .utils import pick_device, l2_normalize, split_by_tokens

__all__ = [
    "SentenceTransformerEmbedder",
    "CachedEmbedder",
    "create_embedder",
    "pick_device",
    "l2_normalize",
    "split_by_tokens",
]
```

### Step 7: engines/__init__.py 추가

**파일**: `backend/llm_infrastructure/embedding/engines/__init__.py`

```python
"""Embedding engines."""

# 필요 시 엔진 재export
from .sentence import create_embedder

__all__ = ["create_embedder"]
```

### Step 8: 어댑터 작성 (레지스트리 연결)

**파일**: `backend/llm_infrastructure/embedding/adapters/sentence.py`

```python
"""SentenceTransformer adapter for registry."""

from typing import Any
import numpy as np
import numpy.typing as npt

from ..base import BaseEmbedder
from ..registry import register_embedder
from ..engines.sentence import create_embedder

@register_embedder("koe5", version="v1")
class KoE5Embedder(BaseEmbedder):
    """
    한국어 E5 임베더 (KoE5).

    Config:
        device: str = "auto" - GPU 선택 전략
        use_cache: bool = False - 디스크 캐싱
        cache_dir: str = ".embed_cache"
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        device = self.config.get("device", "auto")
        use_cache = self.config.get("use_cache", False)
        cache_dir = self.config.get("cache_dir", ".embed_cache")

        # 엔진 생성
        self.engine = create_embedder(
            typ="sentence",
            model_name="nlpai-lab/KoE5",
            device=device,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

        self.dimension = self.engine.get_dimension()

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """단일 텍스트 임베딩."""
        return self.engine.encode_query(text)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        """배치 임베딩."""
        # SentenceTransformer는 내부적으로 배치 처리
        return self.engine.encode(texts, show_progress_bar=len(texts) > 100)


@register_embedder("multilingual_e5", version="v1")
class MultilingualE5Embedder(BaseEmbedder):
    """Multilingual E5 임베더."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        device = self.config.get("device", "auto")
        use_cache = self.config.get("use_cache", False)
        cache_dir = self.config.get("cache_dir", ".embed_cache")

        self.engine = create_embedder(
            typ="sentence",
            model_name="intfloat/multilingual-e5-large",
            device=device,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

        self.dimension = self.engine.get_dimension()

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        return self.engine.encode_query(text)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        return self.engine.encode(texts, show_progress_bar=len(texts) > 100)
```

### Step 9: 어댑터 __init__.py

**파일**: `backend/llm_infrastructure/embedding/adapters/__init__.py`

```python
"""Embedding adapters for registry."""

# 레지스트리 어댑터 자동 로드
from . import sentence  # 자동 등록
try:
    from . import langchain  # 선택사항
except ImportError:
    pass

__all__ = []
```

### Step 10: LangChain 어댑터 (선택사항)

**파일**: `backend/llm_infrastructure/embedding/adapters/langchain.py`

```python
"""LangChain adapter for embedders."""

from langchain.embeddings.base import Embeddings
from ..base import BaseEmbedder

class LangChainEmbedderAdapter(Embeddings):
    """
    BaseEmbedder → LangChain 어댑터.

    사용법:
        embedder = get_embedder("koe5", version="v1")
        lc_embedder = LangChainEmbedderAdapter(embedder)
        docs_vecs = lc_embedder.embed_documents(["text1", "text2"])
        query_vec = lc_embedder.embed_query("query")
    """

    def __init__(self, inner: BaseEmbedder):
        """
        Args:
            inner: BaseEmbedder 인스턴스
        """
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 임베딩 (LangChain 인터페이스)."""
        vecs = self.inner.embed_batch(texts)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """쿼리 임베딩 (LangChain 인터페이스)."""
        vec = self.inner.embed(text)
        return vec.tolist()
```

### Step 11: FAISS 인덱싱 (선택사항)

**파일**: `backend/llm_infrastructure/embedding/indexing/faiss_index.py`

**주의**: retrieval 모듈과 경로 충돌 없도록 명확히 분리

```python
"""FAISS indexing for embeddings."""

import faiss
import numpy as np
from pathlib import Path

class FaissIndex:
    """
    FAISS IVF 인덱스 래퍼.

    사용법:
        # 인덱스 생성
        idx = FaissIndex(dim=768, nlist=100, path="docs.ivf")
        idx.train_add(vecs, ids)
        idx.save()

        # 인덱스 로드 및 검색
        idx = FaissIndex.load("docs.ivf")
        distances, indices = idx.search(query_vec, top_k=5)
    """

    def __init__(self, dim: int, nlist: int = 100, path: str | Path = "docs.ivf"):
        """
        Args:
            dim: 임베딩 차원
            nlist: IVF 클러스터 수
            path: 인덱스 저장 경로
        """
        self.dim = dim
        self.nlist = nlist
        self.path = Path(path)

        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        self.index.nprobe = max(4, int(0.05 * nlist))

    def train_add(self, vecs: np.ndarray, ids: np.ndarray):
        """
        인덱스 학습 및 벡터 추가.

        Args:
            vecs: 임베딩 벡터 (n, dim)
            ids: 문서 ID (n,)
        """
        if not self.index.is_trained:
            self.index.train(vecs)
        self.index.add_with_ids(vecs, ids)

    def save(self):
        """인덱스 저장."""
        faiss.write_index(self.index, str(self.path))

    @classmethod
    def load(cls, path: str | Path):
        """
        저장된 인덱스 로드.

        Args:
            path: 인덱스 파일 경로

        Returns:
            FaissIndex 인스턴스
        """
        idx = faiss.read_index(str(path))
        obj = object.__new__(cls)
        obj.dim = idx.d
        obj.nlist = idx.nlist
        obj.path = Path(path)
        obj.index = idx
        return obj

    def search(self, query: np.ndarray, top_k: int = 5):
        """
        검색 수행.

        Args:
            query: 쿼리 벡터 (dim,) or (1, dim)
            top_k: 상위 K개

        Returns:
            distances: 거리 배열 (top_k,)
            indices: 인덱스 배열 (top_k,)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        D, I = self.index.search(query, top_k)
        return D[0], I[0]
```

## 💡 코드 예시

### 엔진 직접 사용 (프로토타입)

```python
from backend.llm_infrastructure.embedding.engines.sentence import create_embedder

# 1. 기본 사용
embedder = create_embedder(
    typ="sentence",
    model_name="nlpai-lab/KoE5",
    device="auto",  # GPU 자동 선택
)

texts = ["안녕하세요", "임베딩 테스트"]
vecs = embedder.encode(texts)  # (2, 1024)
print(vecs.shape, vecs.dtype)

# 2. 캐싱 사용 (Docker 볼륨 마운트 필요)
cached_embedder = create_embedder(
    typ="sentence",
    model_name="nlpai-lab/KoE5",
    device="auto",
    use_cache=True,
    cache_dir="/app/cache/embeddings",  # 쓰기 가능한 경로
)

# 첫 호출: 실제 계산 + 캐싱
vecs1 = cached_embedder.encode(["안녕하세요"])

# 두 번째 호출: 캐시에서 로드 (빠름)
vecs2 = cached_embedder.encode(["안녕하세요"])
```

### 레지스트리 사용 (프로덕션)

```python
from backend.llm_infrastructure.embedding.registry import get_embedder
from backend.config.settings import rag_settings

# 1. 설정 기반
embedder = get_embedder(
    rag_settings.embedding_method,
    version=rag_settings.embedding_version,
    device="auto",
    use_cache=True,
)

# 2. 직접 지정
embedder = get_embedder(
    "koe5",
    version="v1",
    device="cuda:0",
    use_cache=False,
)

# 3. 사용
vec = embedder.embed("단일 텍스트")  # (dim,)
vecs = embedder.embed_batch(["텍스트1", "텍스트2"])  # (2, dim)
```

### LangChain 통합

```python
from backend.llm_infrastructure.embedding.registry import get_embedder
from backend.llm_infrastructure.embedding.adapters.langchain import LangChainEmbedderAdapter

# 임베더 생성
embedder = get_embedder("koe5", version="v1")

# LangChain 어댑터로 감싸기
lc_embedder = LangChainEmbedderAdapter(embedder)

# LangChain에서 사용
from langchain.vectorstores import FAISS

docs = ["문서1", "문서2", "문서3"]
vectorstore = FAISS.from_texts(docs, lc_embedder)
results = vectorstore.similarity_search("쿼리", k=2)
```

### FAISS 인덱싱

```python
from backend.llm_infrastructure.embedding.engines.sentence import create_embedder
from backend.llm_infrastructure.embedding.indexing.faiss_index import FaissIndex
import numpy as np

# 임베딩 생성
embedder = create_embedder("sentence", "nlpai-lab/KoE5")
docs = ["문서1", "문서2", "문서3"]
vecs = embedder.encode(docs)

# 인덱스 생성 및 저장
idx = FaissIndex(dim=1024, nlist=100, path="/app/data/docs.ivf")
ids = np.arange(len(docs))
idx.train_add(vecs, ids)
idx.save()

# 인덱스 로드 및 검색
idx = FaissIndex.load("/app/data/docs.ivf")
query_vec = embedder.encode_query("쿼리")
distances, indices = idx.search(query_vec, top_k=2)
print(f"Top 2: {indices}, Distances: {distances}")
```

## 🔄 환경변수 설정

**.env 파일**:

```bash
# Embedding 설정
RAG_EMBEDDING_METHOD=koe5
RAG_EMBEDDING_VERSION=v1
EMBEDDING_DEVICE=auto          # auto, round-robin, cuda:0, cpu
EMBEDDING_USE_CACHE=false
EMBEDDING_CACHE_DIR=.embed_cache
```

**backend/config/settings.py** (업데이트 필요):

```python
class RAGSettings(BaseSettings):
    # ... 기존 설정 ...

    # Embedding 추가
    embedding_method: str = Field("koe5", env="RAG_EMBEDDING_METHOD")
    embedding_version: str = Field("v1", env="RAG_EMBEDDING_VERSION")
    embedding_device: str = Field("auto", env="EMBEDDING_DEVICE")
    embedding_use_cache: bool = Field(False, env="EMBEDDING_USE_CACHE")
    embedding_cache_dir: str = Field(".embed_cache", env="EMBEDDING_CACHE_DIR")
```

## 🧪 테스트

### 엔진 테스트 (단위)

**파일**: `tests/embedding/test_sentence_engine.py`

```python
"""SentenceTransformer engine tests."""

import numpy as np
import pytest
from backend.llm_infrastructure.embedding.engines.sentence import create_embedder

def test_create_embedder():
    """팩토리 함수 테스트."""
    embedder = create_embedder(
        typ="sentence",
        model_name="nlpai-lab/KoE5",
        device="cpu",  # CI 환경 고려
    )
    assert embedder is not None
    assert hasattr(embedder, "encode")

def test_encode():
    """배치 임베딩 테스트."""
    embedder = create_embedder("sentence", "nlpai-lab/KoE5", device="cpu")
    texts = ["안녕하세요", "테스트"]
    vecs = embedder.encode(texts)

    assert vecs.shape[0] == 2
    assert vecs.shape[1] > 0  # 차원
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)  # L2 정규화

def test_encode_query():
    """단일 쿼리 임베딩 테스트."""
    embedder = create_embedder("sentence", "nlpai-lab/KoE5", device="cpu")
    vec = embedder.encode_query("쿼리")

    assert vec.ndim == 1
    assert len(vec) > 0

def test_cache(tmp_path):
    """캐싱 테스트."""
    cache_dir = tmp_path / "cache"
    embedder = create_embedder(
        "sentence",
        "nlpai-lab/KoE5",
        device="cpu",
        use_cache=True,
        cache_dir=str(cache_dir),
    )

    text = "캐시 테스트"

    # 첫 호출: 캐시 미스
    import time
    start = time.time()
    vec1 = embedder.encode([text])
    t1 = time.time() - start

    # 두 번째 호출: 캐시 히트 (빠름)
    start = time.time()
    vec2 = embedder.encode([text])
    t2 = time.time() - start

    assert np.allclose(vec1, vec2)
    assert t2 < t1  # 캐시가 더 빠름

def test_gpu_selection():
    """GPU 선택 로직 테스트."""
    from backend.llm_infrastructure.embedding.engines.sentence.utils import pick_device

    # CPU fallback
    device = pick_device("cpu")
    assert device == "cpu"

    # auto (메모리 조회 실패 시 안전)
    device = pick_device("auto")
    assert device in ["cpu", "cuda:0"]
```

### 어댑터 테스트 (통합)

**파일**: `tests/embedding/test_adapters.py`

```python
"""Embedding adapter tests."""

import pytest
from backend.llm_infrastructure.embedding.registry import get_embedder

def test_registry_koe5():
    """KoE5 어댑터 레지스트리 테스트."""
    embedder = get_embedder(
        "koe5",
        version="v1",
        device="cpu",
        use_cache=False,
    )

    vec = embedder.embed("테스트")
    assert vec.ndim == 1

    vecs = embedder.embed_batch(["텍스트1", "텍스트2"])
    assert vecs.shape[0] == 2

def test_registry_multilingual_e5():
    """Multilingual E5 어댑터 테스트."""
    embedder = get_embedder(
        "multilingual_e5",
        version="v1",
        device="cpu",
    )

    vec = embedder.embed("test")
    assert vec.ndim == 1
```

### Docker 환경 테스트

**docker-compose.test.yml**:

```yaml
version: '3.8'

services:
  embedding-test:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - EMBEDDING_DEVICE=cpu
      - EMBEDDING_USE_CACHE=true
      - EMBEDDING_CACHE_DIR=/app/cache/embeddings
    volumes:
      - ./tests:/app/tests
      - embedding_cache:/app/cache/embeddings
    command: pytest tests/embedding/ -v

volumes:
  embedding_cache:
```

실행:
```bash
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 🐳 Docker 설정

### .dockerignore 추가

```
# .dockerignore
.embed_cache/
*.cache
*.ivf
*.index
__pycache__/
.pytest_cache/
```

### Dockerfile 예시

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 의존성 설치
COPY pyproject.toml .
RUN pip install -e .[dev]

# 캐시 디렉토리 권한
RUN mkdir -p /app/cache/embeddings && chmod 777 /app/cache/embeddings

COPY . .

CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0"]
```

### docker-compose.yml 볼륨 설정

```yaml
services:
  backend:
    volumes:
      - embedding_cache:/app/cache/embeddings
      - faiss_indices:/app/data/indices
    environment:
      - EMBEDDING_CACHE_DIR=/app/cache/embeddings
      - CUDA_VISIBLE_DEVICES=0,1  # GPU 제한

volumes:
  embedding_cache:
  faiss_indices:
```

## ✅ 마이그레이션 체크리스트

### 코드 마이그레이션
- [ ] **Step 1**: `engines/sentence/utils.py` 생성 (device, normalize, chunking)
- [ ] **Step 2**: `engines/sentence/base.py` 생성 (BaseEmbedder)
- [ ] **Step 3**: `engines/sentence/cache.py` 생성 (캐싱)
- [ ] **Step 4**: `engines/sentence/embedder.py` 생성 (SentenceTransformer)
- [ ] **Step 5**: `engines/sentence/factory.py` 생성 (팩토리)
- [ ] **Step 6**: `engines/sentence/__init__.py` 생성
- [ ] **Step 7**: `engines/__init__.py` 생성
- [ ] **Step 8**: `adapters/sentence.py` 생성 (레지스트리 어댑터)
- [ ] **Step 9**: `adapters/__init__.py` 생성
- [ ] **Step 10**: `adapters/langchain.py` 생성 (선택)
- [ ] **Step 11**: `indexing/faiss_index.py` 생성 (선택)

### 설정 및 환경
- [ ] **Step 12**: `backend/config/settings.py` 업데이트
- [ ] **Step 13**: `.env` 파일 업데이트
- [ ] **Step 14**: `pyproject.toml` 의존성 추가
- [ ] **Step 15**: `.dockerignore` 업데이트
- [ ] **Step 16**: Docker 볼륨 설정

### 테스트 및 검증
- [ ] **Step 17**: 엔진 단위 테스트 작성
- [ ] **Step 18**: 어댑터 통합 테스트 작성
- [ ] **Step 19**: Docker 환경 테스트
- [ ] **Step 20**: GPU 선택 로직 검증
- [ ] **Step 21**: 캐시 hit/miss 검증

## 📝 마이그레이션 후 파일 구조

```
backend/llm_infrastructure/embedding/
├── engines/                     # ✅ 신규
│   ├── __init__.py              # ✅ 엔진 재export
│   └── sentence/                # ✅ SentenceTransformer 엔진
│       ├── __init__.py
│       ├── base.py              # ✅ llm-agent의 embedders/base.py
│       ├── embedder.py          # ✅ llm-agent의 embedders/sentence.py
│       ├── cache.py             # ✅ llm-agent의 embedders/cache.py
│       ├── utils.py             # ✅ llm-agent의 utils/*
│       └── factory.py           # ✅ llm-agent의 create_embedder.py
├── adapters/                    # ✅ 신규
│   ├── __init__.py              # ✅ 어댑터 자동 로드
│   ├── sentence.py              # ✅ 신규 (레지스트리 어댑터)
│   ├── tei.py                   # ✅ 기존 embedders/tei_client.py 이동
│   └── langchain.py             # ✅ llm-agent의 adapters/langchain.py
├── indexing/                    # ✅ 신규 (선택)
│   ├── __init__.py
│   └── faiss_index.py           # ✅ llm-agent의 indexing.py
├── base.py                      # ✅ 유지 (레지스트리용)
└── registry.py                  # ✅ 유지
```

**핵심 변경**:
1. `sentence_engine/` → `engines/sentence/` (계층 추가)
2. 엔진 내부에 `base.py` 추가 (엔진용 BaseEmbedder)
3. `adapters/__init__.py`에서 자동 import로 레지스트리 등록
4. TEI 클라이언트도 `adapters/tei.py`로 이동

## 🎯 다음 단계

1. ✅ 이 가이드를 기반으로 코드 마이그레이션
2. ✅ 테스트 작성 (엔진, 어댑터 각각)
3. ✅ 문서 업데이트 (`embedding/README.md` 작성)
4. ✅ 실험 러너에 통합
5. ✅ Retrieval 모듈 마이그레이션 (다음 단계)

## 📚 관련 문서

- [Preprocessing Guide](../backend/llm_infrastructure/preprocessing/README.md): 엔진-어댑터 패턴 예시
- [프로젝트 README](../README.md): 전체 아키텍처

## 📞 문의

- 임베딩 엔진 관련: `sentence_engine/` 코드 확인
- 레지스트리 통합: `adapters/`, `registry.py` 확인
- 설정 주입: `backend/config/settings.py` 확인
