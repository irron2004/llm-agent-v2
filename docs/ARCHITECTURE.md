# 프로젝트 아키텍처: 엔진-어댑터 패턴

이 문서는 `llm-agent-v2` 프로젝트의 핵심 아키텍처 패턴을 설명합니다.
preprocessing, embedding, retrieval 모두 동일한 **엔진-어댑터-레지스트리** 패턴을 사용합니다.

---

## 0. 용어 정리

| 용어 | 역할 | 예시 |
|------|------|------|
| **엔진(engine)** | 실제 알고리즘 로직. 순수 함수/클래스 위주. | L0~L5 정규화 규칙, SentenceTransformer 래퍼 |
| **어댑터(adapter)** | 엔진을 감싸서 레지스트리에 등록하는 껍데기. 설정(level, variant_map 등)을 받아 엔진을 호출. | `NormalizationPreprocessor`, `SentenceEmbedderAdapter` |
| **레지스트리(registry)** | 이름 → 어댑터 매핑 테이블. `get_preprocessor("normalize")` 같은 식으로 선택. | `PreprocessorRegistry`, `EmbedderRegistry` |

### 왜 이렇게 나누는가?

- **엔진**: 순수 알고리즘이라 테스트/리팩토링이 쉬움. 로직 변경 시 레지스트리/파이프라인 코드는 안 건드림.
- **어댑터**: 서비스/파이프라인 인터페이스. 실험/스위칭이 쉬움.
- **레지스트리**: 설정 기반으로 구현체를 선택할 수 있음.

---

## 1. 디렉토리 구조

```text
backend/
  llm_infrastructure/
    preprocessing/           # 전처리
      base.py                # BasePreprocessor 추상 클래스
      registry.py            # PreprocessorRegistry (이름 → 어댑터)

      normalize_engine/      # 엔진 패키지 (L0~L5 실제 로직)
        __init__.py          # build_normalizer 재export
        factory.py           # build_normalizer(level, variant_map, ...)
        base.py              # L0, L1, L2 (기본 정규화)
        domain.py            # L3, L4, L5 (도메인 특화)
        rules.py             # 변형어/regex 규칙들
        utils.py             # 유틸리티 함수

      adapters/              # 어댑터 (레지스트리와 연결)
        __init__.py
        normalize.py         # NormalizationPreprocessor (level을 받아 엔진 호출)
        standard.py          # StandardPreprocessor → 내부에서 L0 사용
        domain_specific.py   # DomainSpecificPreprocessor → 내부에서 L3/L4 사용

    embedding/               # 임베딩
      base.py                # BaseEmbedder 추상 클래스
      registry.py            # EmbedderRegistry

      engines/               # 엔진 패키지
        sentence/            # SentenceTransformer 엔진
          __init__.py
          factory.py         # create_embedder()
          embedder.py        # SentenceEmbedder 클래스
          cache.py           # 임베딩 캐시
          utils.py           # 디바이스 선택 등 유틸

      adapters/              # 어댑터
        __init__.py
        sentence.py          # SentenceEmbedderAdapter (koe5, bge_base 등)
        tei.py               # TEIEmbedderAdapter

    retrieval/               # 검색 (동일 패턴)
      base.py
      registry.py
      engines/
      adapters/
```

---

## 2. 전처리(Preprocessing) 정규화 레벨

### L0 ~ L5 정의

| 레벨 | 설명 | 주요 기능 |
|------|------|-----------|
| **L0** | 기본 정규화 | 소문자 변환, 유니코드 정규화, 공백 정리 |
| **L1** | L0 + 변형어 치환 | variant_map을 사용한 용어 통일 |
| **L2** | L1 확장 (예약) | 현재 L1과 동일, 추후 확장용 |
| **L3** | 반도체 도메인 규칙 | PM 주소 마스킹, 과학 표기 변환, 오탈자 수정 |
| **L4** | L3 + 고급 규칙 | 엔티티 추출, 헤더 토큰 생성 (MODULE, ALARM 등) |
| **L5** | L4 + 변형어 강화 | 도메인 특화 변형어 맵 적용, 범위 표현 정규화 |

### 엔진 파일 구조

```text
normalize_engine/
  base.py      # normalize_l0(), normalize_l1(), normalize_l2()
  domain.py    # preprocess_semiconductor_domain() [L3]
               # preprocess_l4_advanced_domain() [L4]
               # preprocess_l5_enhanced_domain() [L5]
  factory.py   # build_normalizer(level, variant_map, keep_newlines)
  rules.py     # VARIANT_MAP, TYPO_MAP, REGEX 패턴들
  utils.py     # 헬퍼 함수들
```

### 사용 예시

```python
# 엔진 직접 사용
from backend.llm_infrastructure.preprocessing.normalize_engine import build_normalizer

norm = build_normalizer("L3")
text = "pm 2-1 helium leak 4.0x10^-9"
print(norm(text))  # "PM helium leak 4.0e-09"

# 레지스트리 통해 사용
from backend.llm_infrastructure.preprocessing.registry import get_preprocessor

proc = get_preprocessor("normalize", level="L3")
results = list(proc.preprocess(["pm 2-1 helium leak"]))
```

---

## 3. 임베딩(Embedding) 구조

### 엔진

```text
engines/
  sentence/
    factory.py   # create_embedder(typ, model_name, device, ...)
    embedder.py  # SentenceEmbedder (SentenceTransformer 래퍼)
    cache.py     # EmbeddingCache (디스크 캐시)
    utils.py     # pick_device() 등
```

### 어댑터

```python
# adapters/sentence.py
@register_embedder("koe5", version="v1")
@register_embedder("multilingual_e5", version="v1")
@register_embedder("bge_base", version="v1")
class SentenceEmbedderAdapter(BaseEmbedder):
    def __init__(self, model_name=None, device=None, method_name=None, ...):
        # method_name으로 기본 모델 결정
        resolved_model = model_name or DEFAULT_MODELS.get(method_name)
        self.engine = create_embedder("sentence", model_name=resolved_model, ...)
```

### 사용 예시

```python
from backend.llm_infrastructure.embedding.registry import get_embedder

# 레지스트리로 임베더 선택
emb = get_embedder("bge_base", version="v1", device="cpu")
vector = emb.embed("hello world")
vectors = emb.embed_batch(["hello", "world"])
```

---

## 4. 레지스트리 패턴

모든 레지스트리는 동일한 패턴을 따릅니다:

```python
# 등록 (데코레이터)
@register_preprocessor("normalize", version="v1")
class NormalizationPreprocessor(BasePreprocessor):
    ...

# 조회
preprocessor = get_preprocessor("normalize", version="v1", level="L3")
```

### 레지스트리 구현 패턴

```python
class Registry:
    _registry: dict[str, dict[str, type]] = {}  # name -> version -> class

    @classmethod
    def register(cls, name: str, impl_cls: type, version: str = "v1"):
        cls._registry.setdefault(name, {})[version] = impl_cls

    @classmethod
    def get(cls, name: str, version: str = "v1", **kwargs):
        impl_cls = cls._registry[name][version]
        return impl_cls(**kwargs)

    @classmethod
    def list_methods(cls) -> dict[str, list[str]]:
        return {name: list(versions.keys()) for name, versions in cls._registry.items()}
```

---

## 5. 설정 기반 선택

`.env` 또는 설정 파일로 구현체를 선택합니다:

```bash
# .env
RAG_PREPROCESS_METHOD=normalize
RAG_PREPROCESS_VERSION=v1
RAG_EMBEDDING_METHOD=bge_base
RAG_EMBEDDING_VERSION=v1
RAG_RETRIEVAL_PRESET=hybrid_rrf_v1
```

```python
# 설정 기반 사용
from backend.config.settings import rag_settings
from backend.llm_infrastructure.preprocessing.registry import get_preprocessor

proc = get_preprocessor(
    rag_settings.preprocess_method,
    version=rag_settings.preprocess_version,
)
```

---

## 6. 새 구현체 추가 방법

### 전처리기 추가

1. **엔진 로직 추가** (선택사항 - 기존 엔진 재사용 가능):
   ```python
   # normalize_engine/domain.py
   def preprocess_my_domain(text: str) -> str:
       # 새로운 도메인 로직
       return processed_text
   ```

2. **어댑터 작성**:
   ```python
   # adapters/my_preprocessor.py
   from ..base import BasePreprocessor
   from ..registry import register_preprocessor
   from ..normalize_engine import build_normalizer

   @register_preprocessor("my_method", version="v1")
   class MyPreprocessor(BasePreprocessor):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self._fn = build_normalizer("L3")  # 또는 커스텀 로직

       def preprocess(self, docs):
           for doc in docs:
               yield self._fn(str(doc))
   ```

3. **adapters/__init__.py에서 import**:
   ```python
   from .my_preprocessor import MyPreprocessor
   ```

### 임베더 추가

1. **엔진 작성** (필요시):
   ```python
   # engines/my_engine/embedder.py
   class MyEmbedder:
       def encode(self, texts): ...
   ```

2. **어댑터 작성**:
   ```python
   # adapters/my_embedder.py
   @register_embedder("my_embedder", version="v1")
   class MyEmbedderAdapter(BaseEmbedder):
       def __init__(self, ...):
           super().__init__(...)
           self.engine = MyEmbedder(...)

       def embed(self, text): return self.engine.encode([text])[0]
       def embed_batch(self, texts, batch_size=32): return self.engine.encode(texts)
   ```

---

## 7. 테스트 구조

```text
backend/tests/
  test_preprocessing_normalize_engine.py  # 전처리 엔진 + 어댑터 테스트
  test_embedding_engine.py                # 임베딩 엔진 + 어댑터 테스트
```

테스트는 다음을 검증합니다:
- 엔진 함수의 정확한 동작
- 어댑터가 엔진을 올바르게 호출하는지
- 레지스트리 등록/조회가 정상 동작하는지

```bash
# 테스트 실행
uv run pytest backend/tests/ -v
```

---

## 8. 확장 가이드

### Retrieval도 동일 패턴

```text
retrieval/
  base.py          # BaseRetriever
  registry.py      # RetrieverRegistry

  engines/
    bm25.py        # BM25 검색 엔진
    dense.py       # Dense 벡터 검색
    hybrid.py      # Hybrid (BM25 + Dense)

  adapters/
    bm25.py        # @register_retriever("bm25")
    dense.py       # @register_retriever("dense")
    hybrid.py      # @register_retriever("hybrid")
```

### LLM도 동일 패턴

```text
llm/
  base.py
  registry.py

  engines/
    vllm.py
    openai.py

  adapters/
    vllm.py
    openai.py
```

---

## 요약

| 계층 | 역할 | 파일 위치 |
|------|------|-----------|
| **엔진** | 실제 알고리즘 구현 | `{module}/engines/` 또는 `{module}/normalize_engine/` |
| **어댑터** | 엔진 래핑 + 레지스트리 등록 | `{module}/adapters/` |
| **레지스트리** | 이름 → 어댑터 매핑 | `{module}/registry.py` |
| **베이스** | 추상 클래스 정의 | `{module}/base.py` |

이 패턴을 따르면:
- 엔진 로직 변경 시 어댑터/서비스 코드는 그대로
- 설정만 바꿔서 구현체 스위칭 가능
- 새 구현체 추가가 간단함 (어댑터만 작성)
- 테스트가 깔끔하게 분리됨