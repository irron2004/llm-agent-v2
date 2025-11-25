# PE Agent Service (리팩터링)

프로세스 엔지니어링 문제 해결을 위한 Retrieval-Augmented Generation(RAG) 시스템입니다. 모듈식 설계와 연구 친화적 실험 환경을 제공합니다.

## 🎯 주요 특징

- **레지스트리 패턴**: 전처리·임베딩·검색 방식을 플러그인처럼 교체
- **설정 기반 파이프라인**: 환경변수 또는 YAML 프리셋으로 손쉽게 스위칭
- **실험 러너**: 신규 접근법을 빠르게 평가할 수 있는 내장 프레임워크
- **계층형 아키텍처**: API, 서비스, 도메인, 인프라를 명확히 분리
- **도커 스택**: vLLM, RAGFlow, TEI를 포함한 배포 구성을 제공

## 📁 프로젝트 구조

```
pe-agent-service/
├── backend/                    # 백엔드 애플리케이션
│   ├── api/                    # FastAPI 라우터 (HTTP 엔드포인트)
│   ├── services/               # 서비스 레이어 (업무 오케스트레이션)
│   ├── domain/                 # 도메인 레이어 (PE 전용 로직)
│   ├── llm_infrastructure/     # LLM 인프라 레이어 (재사용 컴포넌트)
│   │   ├── preprocessing/      # 전처리 레지스트리
│   │   ├── embedding/          # 임베딩 레지스트리
│   │   ├── retrieval/          # 검색 레지스트리 + 프리셋
│   │   └── llm/                # LLM 엔진(vLLM 등)
│   └── config/                 # 설정 및 프리셋
│
├── frontend/                   # 프런트엔드 (React UI)
├── docker/                     # 도커 스택
├── experiments/                # 실험 러너 및 설정
├── data/                       # 데이터 볼륨
└── docs/                       # 문서
```

## 🏗️ 아키텍처

### 계층형 설계

```
API Layer → Service Layer → Domain Layer → Infrastructure Layer
```

- **API Layer**: HTTP 엔드포인트만 담당 (FastAPI)
- **Service Layer**: 비즈니스 로직 오케스트레이션
- **Domain Layer**: PE Agent 도메인 로직
- **Infrastructure Layer**: 전처리·임베딩·검색 등 재사용 LLM 컴포넌트

### 레지스트리 패턴

주요 컴포넌트를 동적으로 선택할 수 있도록 **레지스트리 패턴**을 사용합니다.

```python
# 새 전처리기 등록
from backend.llm_infrastructure.preprocessing import register_preprocessor, BasePreprocessor

@register_preprocessor("my_method", version="v1")
class MyPreprocessor(BasePreprocessor):
    def preprocess(self, docs):
        # 구현
        return processed_docs

# 설정을 통해 사용 (.env: RAG_PREPROCESS_METHOD=my_method)
from backend.config.settings import rag_settings
from backend.llm_infrastructure.preprocessing.registry import get_preprocessor

preprocessor = get_preprocessor(rag_settings.preprocess_method)
```

임베딩: `@register_embedder("name", version="v1")`  
검색: `@register_retriever("name", version="v1")`

## 🚀 빠른 시작

### 1) 환경 준비

```bash
cd docker
cp .env.example .env
# .env에서 설정 값을 채워 넣으세요

# 예시:
# RAG_PREPROCESS_METHOD=standard
# RAG_EMBEDDING_METHOD=bge_base
# RAG_RETRIEVAL_PRESET=hybrid_rrf_v1
```

### 2) 서비스 실행

```bash
# 백엔드 + vLLM + RAGFlow + TEI 전체 스택
make up

# 상태 확인
make status

# 로그 확인
make logs-fastapi
make logs-vllm
```

### 3) 서비스 접속

- **PE Agent UI**: http://localhost:3000
- **FastAPI Docs**: http://localhost:8100/docs
- **RAGFlow**: http://localhost:9380

## 🧪 실험 실행

내장 실험 러너로 RAG 파이프라인을 평가할 수 있습니다.

### 새 전처리 방법 테스트 예시

1. 구현
```python
# backend/llm_infrastructure/preprocessing/methods/my_paper_method.py
from ..base import BasePreprocessor
from ..registry import register_preprocessor

@register_preprocessor("paper_xyz", version="v1")
class PaperXYZPreprocessor(BasePreprocessor):
    def preprocess(self, docs):
        # 논문 방법 구현
        return processed_docs
```

2. 실험 설정
```yaml
# experiments/configs/test_paper_xyz.yaml
name: test_paper_xyz
preprocess_method: paper_xyz
preprocess_version: v1
embedding_method: bge_base
retrieval:
  method: hybrid
  top_k: 50
```

3. 실행
```bash
python -m experiments.run \
    --config experiments/configs/test_paper_xyz.yaml \
    --dataset data/eval/pe_agent_eval.jsonl \
    --output experiments/runs/test_paper_xyz/
```

4. 결과 확인
```bash
cat experiments/runs/test_paper_xyz/metrics.json
```

자세한 내용은 `experiments/README.md` 참고.

## 📋 설정

### 환경변수 (.env)

```bash
# 전처리
RAG_PREPROCESS_METHOD=standard      # 예: pe_domain, custom_method
RAG_PREPROCESS_VERSION=v1

# 임베딩
RAG_EMBEDDING_METHOD=bge_base       # 예: bge_large, multilingual_e5, tei
RAG_EMBEDDING_VERSION=v1

# 검색 프리셋
RAG_RETRIEVAL_PRESET=hybrid_rrf_v1  # config/presets/ 참고

# RAGFlow
RAG_RAGFLOW_ENABLED=true
RAG_RAGFLOW_API_KEY=your-key
RAG_RAGFLOW_AGENT_ID=your-agent-id

# vLLM
VLLM_BASE_URL=http://vllm:8000
VLLM_MODEL_NAME=gpt-oss-20b
```

### 검색 프리셋 예시 (`backend/config/presets/`)

- `dense_only.yaml`: 순수 시맨틱 검색(베이스라인)
- `hybrid_rrf_v1.yaml`: Dense+Sparse 하이브리드(RRF, 추천)
- `hybrid_multi_query.yaml`: 하이브리드 + 다중 쿼리 확장
- `hybrid_rerank.yaml`: 하이브리드 + 크로스 인코더 재랭킹
- `full_pipeline.yaml`: 모든 기능 활성화(최고 품질, 더 높은 지연)

복사 후 수정해 커스텀 프리셋을 만들 수 있습니다.

## 🔧 개발 가이드

### 새 임베딩 추가

```python
# backend/llm_infrastructure/embedding/embedders/my_embedder.py
from ..base import BaseEmbedder
from ..registry import register_embedder

@register_embedder("my_embedder", version="v1")
class MyEmbedder(BaseEmbedder):
    def embed(self, text):
        # 구현
        return embedding_vector

    def embed_batch(self, texts, batch_size=32):
        # 배치 구현
        return embeddings_matrix
```

`.env` 업데이트:
```bash
RAG_EMBEDDING_METHOD=my_embedder
RAG_EMBEDDING_VERSION=v1
```

### 새 검색기 추가

```python
# backend/llm_infrastructure/retrieval/methods/my_retriever.py
from ..base import BaseRetriever, RetrievalResult
from ..registry import register_retriever

@register_retriever("my_retriever", version="v1")
class MyRetriever(BaseRetriever):
    def retrieve(self, query, top_k=10):
        # 구현
        return [
            RetrievalResult(doc_id="...", content="...", score=0.95),
            # ...
        ]
```

`config/presets/retrieval_my_method.yaml`와 같은 프리셋을 추가해 사용하세요.

## 📚 문서

- [Migration Guide](docs/MIGRATION_GUIDE.md): 기존 구조에서 이동 방법
- [Experiment Guide](experiments/README.md): 실험 실행 및 ablation 가이드
- [Preprocessing Guide](backend/llm_infrastructure/preprocessing/README.md): 전처리 아키텍처 및 사용법 (엔진-어댑터 패턴)
- [API Docs](http://localhost:8100/docs): 실행 중 접근 가능한 인터랙티브 문서

## 🎓 연구 친화적 설계

1. RAG 관련 새 논문을 읽고
2. 전처리/임베딩/검색 중 하나를 레지스트리에 추가하고
3. 설정 파일에 방법을 지정한 뒤
4. 실험 러너로 베이스라인과 비교
5. 결과를 분석하며 반복

기존 코드를 크게 변경하지 않고도 새 방법을 빠르게 검증할 수 있습니다.

## 🛠️ 기술 스택

- **Backend**: FastAPI, Pydantic Settings
- **LLM**: vLLM(inference), TEI(embeddings)
- **RAG**: RAGFlow(문서 처리, 벡터 DB)
- **Frontend**: React, Vite
- **Infrastructure**: Docker Compose, Nginx
- **Experiment Tracking**: YAML 설정, JSONL 결과(필요 시 WandB/MLflow 확장 가능)

## 📝 마이그레이션

이 프로젝트는 `/home/hskim/work/llm-agent`를 리팩터링한 버전입니다.

개선 사항:
- ✅ 계층형 아키텍처
- ✅ 전 영역 레지스트리 패턴
- ✅ 설정 기반 파이프라인
- ✅ 내장 실험 프레임워크
- ✅ BE/FE 분리
- ✅ 루트 디렉터리 단순화

자세한 이동 방법은 [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)를 확인하세요.

## 🤝 기여

1. 해당 `methods/` 디렉터리에 새 방법을 추가
2. `@register_*` 데코레이터로 등록
3. 테스트와 문서 추가
4. 예시 설정/프리셋 제공

## 📄 라이선스

[Your License Here]

## 🙋 지원

문의나 이슈가 있으면:
- `docs/` 문서를 먼저 확인
- `experiments/configs/`의 예시 설정 참조
- 마이그레이션 가이드를 검토하세요
