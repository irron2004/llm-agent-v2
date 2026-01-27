# LLM-Agent-V2

프로세스 엔지니어링(PE) 트러블슈팅을 위한 **RAG(Retrieval-Augmented Generation) 기반 에이전트 시스템**입니다.

## 목차
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [빠른 시작](#빠른-시작)
- [설정 가이드](#설정-env-기준)
- [API 엔드포인트](#api-엔드포인트)
- [핵심 아키텍처](#핵심-아키텍처)
- [관련 문서](#문서)

---

## 주요 기능
- **RAG 파이프라인**: 문서 수집 → 전처리 → 임베딩 → 검색 → 재순위 → LLM 생성
- **도메인 특화 정규화**: PE 분야에 최적화된 L0~L5 정규화 엔진
- **하이브리드 검색**: Dense(벡터) + Sparse(BM25) + RRF 융합
- **레지스트리 패턴**: 전처리·임베딩·검색을 이름/버전으로 선택·교체 가능
- **설정 기반**: Pydantic Settings(.env) + YAML 프리셋으로 모든 설정 관리
- **실험 프레임워크**: 파이프라인 실험 및 평가 자동화

---

## 기술 스택

### 백엔드
| 분류 | 기술 |
|------|------|
| 언어/프레임워크 | Python 3.12+, FastAPI 0.110+, Uvicorn |
| 검색 엔진 | Elasticsearch 8.0, Rank-BM25 |
| 임베딩 | Sentence-Transformers, BGE, TEI |
| LLM | vLLM, OpenAI SDK, LangChain, LangGraph |
| 저장소 | MinIO (S3 호환), DiskCache |

### 프론트엔드
| 분류 | 기술 |
|------|------|
| 프레임워크 | React 18+, TypeScript, Vite |
| UI | Ant Design |
| 데이터 페칭 | TanStack React Query |

### DevOps
| 분류 | 기술 |
|------|------|
| 컨테이너 | Docker, Docker Compose |
| 패키지 관리 | UV (Python), pnpm (Node) |
| 코드 품질 | Ruff, MyPy, Pytest |

---

## 프로젝트 구조
```
llm-agent-v2/
├── backend/                          # 백엔드 (Python)
│   ├── api/                          # FastAPI API 레이어
│   │   ├── main.py                   # 앱 진입점
│   │   ├── dependencies.py           # 의존성 주입
│   │   └── routers/                  # API 라우터 (11개)
│   ├── config/                       # 설정 관리
│   │   ├── settings.py               # 메인 설정 (Pydantic)
│   │   ├── preset_loader.py          # YAML 프리셋 로더
│   │   └── presets/                  # 검색 프리셋 YAML
│   ├── llm_infrastructure/           # 핵심 LLM 인프라
│   │   ├── preprocessing/            # 전처리 모듈
│   │   │   ├── normalize_engine/     # L0~L5 정규화 엔진
│   │   │   ├── adapters/             # 전처리 어댑터
│   │   │   ├── parsers/              # 문서 파서 (PDF, 텍스트)
│   │   │   └── chunking/             # 청킹 알고리즘
│   │   ├── embedding/                # 임베딩 (BGE, TEI 어댑터)
│   │   ├── retrieval/                # 검색 (Dense, BM25, Hybrid)
│   │   ├── llm/                      # LLM (vLLM, 프롬프트)
│   │   ├── reranking/                # 재순위
│   │   ├── query_expansion/          # 질의 확장
│   │   ├── summarization/            # 요약
│   │   ├── vlm/                      # 비전 LLM
│   │   └── elasticsearch/            # ES 관리
│   ├── services/                     # 서비스 레이어
│   │   ├── chat_service.py           # 챗 오케스트레이션
│   │   ├── search_service.py         # 검색 서비스
│   │   ├── es_search_service.py      # ES 검색
│   │   ├── embedding_service.py      # 임베딩 서비스
│   │   ├── agents/                   # 에이전트 구현
│   │   └── ingest/                   # 수집 파이프라인
│   ├── domain/pe_core/               # PE 도메인 로직
│   ├── tests/                        # 테스트 코드
│   └── pyproject.toml                # 백엔드 의존성
│
├── frontend/                         # 프론트엔드 (React)
│   ├── src/
│   │   ├── components/               # React 컴포넌트
│   │   ├── features/                 # 기능 모듈
│   │   └── lib/                      # 유틸리티
│   ├── package.json                  # Node 의존성
│   └── vite.config.ts                # Vite 설정
│
├── experiments/                      # 실험 프레임워크
│   ├── run.py                        # 실험 러너
│   ├── configs/                      # 실험 설정 YAML
│   └── runs/                         # 실험 결과
│
├── scripts/                          # 유틸리티 스크립트
│   ├── batch_ingest_*.py             # 배치 수집
│   ├── es_migrate_v2.py              # ES 마이그레이션
│   └── evaluation/                   # 평가 스크립트
│
├── data/                             # 데이터 디렉토리
│   ├── ingestions/                   # 수집된 문서
│   ├── elasticsearch/                # ES 인덱스 데이터
│   ├── golden_set/                   # 평가용 골드셋
│   ├── hf_cache/                     # HuggingFace 캐시
│   └── vector_store/                 # 벡터 스토어
│
├── docs/                             # 문서
│   └── work_list/                    # 작업 목록
│
├── docker-compose.yml                # Docker Compose
├── Makefile                          # 개발 명령어
├── pyproject.toml                    # 프로젝트 메타데이터
└── .env                              # 환경변수
```

### 디렉토리별 빠른 참조
| 찾고 싶은 것 | 위치 |
|-------------|------|
| API 라우터 | `backend/api/routers/` |
| 설정 파일 | `backend/config/settings.py` |
| 전처리 엔진 | `backend/llm_infrastructure/preprocessing/normalize_engine/` |
| 임베딩 어댑터 | `backend/llm_infrastructure/embedding/adapters/` |
| 검색 방식 | `backend/llm_infrastructure/retrieval/methods/` |
| LLM 프롬프트 | `backend/llm_infrastructure/llm/prompts/` |
| 서비스 로직 | `backend/services/` |
| 프론트엔드 컴포넌트 | `frontend/src/components/` |
| 실험 설정 | `experiments/configs/` |
| 배치 스크립트 | `scripts/` |

## 빠른 시작
### 1) 파이썬 환경 설치
```bash
cd backend
pip install -e .[dev]   # 또는 pip install -e .
```

### 2) 전처리 엔진 직접 사용 (L3 예시)
```bash
python - <<'PY'
from backend.llm_infrastructure.preprocessing.normalize_engine import build_normalizer

norm = build_normalizer("L3")
text = "pm 2-1 chamber alarm (1234) helium leak 4.0x10^-9"
print(norm(text))
PY
```

### 3) 레지스트리로 전처리 선택
```python
from backend.config.settings import rag_settings
from backend.llm_infrastructure.preprocessing.registry import get_preprocessor

proc = get_preprocessor(
    rag_settings.preprocess_method,
    version=rag_settings.preprocess_version,
)
print(list(proc.preprocess([" sample text "])))
```

### 4) 실험 러너 실행 (파이프라인 구현 필요)
`experiments/run.py`는 스켈레톤입니다. 실제 파이프라인(전처리→임베딩→검색→생성)을 구현한 뒤 실행하세요.
```bash
python -m experiments.run \
  --config experiments/configs/example_experiment.yaml \
  --dataset path/to/dataset.jsonl \
  --output experiments/runs/example/
```

## 설정 (.env 기준)
- RAG: `RAG_PREPROCESS_METHOD`, `RAG_PREPROCESS_VERSION`, `RAG_EMBEDDING_METHOD`, `RAG_EMBEDDING_VERSION`, `RAG_RETRIEVAL_PRESET`, `RAG_RAGFLOW_ENABLED`, `RAG_RAGFLOW_BASE_URL`, `RAG_RAGFLOW_API_KEY`, `RAG_RAGFLOW_AGENT_ID`
- vLLM: `VLLM_BASE_URL`, `VLLM_MODEL_NAME`, `VLLM_TEMPERATURE`, `VLLM_MAX_TOKENS`, `VLLM_TIMEOUT`
- TEI: `TEI_ENDPOINT_URL`, `TEI_TIMEOUT`
- API: `API_TITLE`, `API_VERSION`, `API_DESCRIPTION`, `API_HOST`, `API_PORT`, `API_RELOAD`, `API_LOG_LEVEL`

### 공용 모델 캐시(.env.llm) 활용
여러 사용자가 공유 모델/데이터 캐시를 쓰려면 `/home/llm-share/.env.llm`를 만들어 프로젝트에서 먼저 로드합니다.

```python
from dotenv import load_dotenv
from pathlib import Path
import os

# 1) 공용 .env 먼저 로드
load_dotenv("/home/llm-share/.env.llm")

# 2) 프로젝트 로컬 .env도 있으면 덮어쓰기(옵션)
project_env = Path(__file__).resolve().parents[2] / ".env"
if project_env.exists():
    load_dotenv(project_env, override=True)

LLM_SHARED_ROOT = os.getenv("LLM_SHARED_ROOT", "/home/llm-share")
HF_HOME = os.getenv("HF_HOME", f"{LLM_SHARED_ROOT}/hf")
```

Docker/docker-compose를 사용할 때는 공용 .env + 프로젝트 .env를 같이 물려줍니다.

```yaml
services:
  backend:
    env_file:
      - /home/llm-share/.env.llm  # 공용
      - ./.env                    # 프로젝트 개별(선택)
```

## 전처리(정규화) 개요
- 엔진: `backend/llm_infrastructure/preprocessing/normalize_engine/`에서 L0~L5 구현.
- 어댑터: `adapters/standard.py`, `adapters/domain_specific.py`가 레지스트리에 등록.
- 새 전처리 추가 예시:
```python
# backend/llm_infrastructure/preprocessing/adapters/my_preprocessor.py
from ..base import BasePreprocessor
from ..registry import register_preprocessor
from ..normalize_engine import build_normalizer

@register_preprocessor("my_method", version="v1")
class MyPreprocessor(BasePreprocessor):
    def preprocess(self, docs):
        norm = build_normalizer("L3")
        for doc in docs:
            yield norm(str(doc))
```

## 임베딩/검색
- 임베딩: `embedding/base.py`, `embedding/registry.py`, 샘플 구현(`embedders/sentence_transformer.py`, `tei_client.py`).
- 검색: `retrieval/base.py`, `retrieval/registry.py`, 프리셋(`retrieval/presets.py` + `config/presets/*.yaml`).

## 프런트엔드
- `frontend/`에 React 소스/빌드 산출물이 포함됩니다. 빌드/런 스크립트는 개별 설정에 맞게 추가하세요.

## 문서
- 전처리 가이드: `backend/llm_infrastructure/preprocessing/README.md`
- 실험 가이드: `experiments/README.md`
- 리팩토링 배경/의미: `docs/REFACTORING_RATIONALE.md`

## 기여/문의
- 레지스트리에 새 컴포넌트를 추가하고 예시 설정을 함께 제공해주세요.
- 질문/이슈는 관련 디렉터리의 README와 코드 주석을 먼저 확인한 후 정리해 주세요.
