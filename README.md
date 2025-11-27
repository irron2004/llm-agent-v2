# llm-agent-v2

프로세스 엔지니어링(PE) 트러블슈팅을 위한 RAG 기반 에이전트 스켈레톤입니다. 전처리/임베딩/검색 레지스트리와 실험 러너를 중심으로, 도메인 특화 정규화(L0~L5)를 지원합니다.

## 현재 상태
- FastAPI API/서비스 레이어는 스켈레톤이며, 실제 엔드포인트·파이프라인 연결은 직접 채워야 합니다.
- Docker/Makefile/Compose는 포함되어 있지 않습니다. 로컬 파이썬 환경 기준입니다.
- 핵심 빌딩 블록: 전처리(normalize_engine), 임베딩/검색 레지스트리, 실험 러너(파이프라인 구현은 TODO).

## 주요 특징
- 레지스트리 패턴: 전처리·임베딩·검색을 이름/버전으로 선택·교체
- 전처리 엔진: L0~L5 정규화(도메인 특화, 변형어/알람/모듈 토큰화 포함)
- 설정 기반: Pydantic Settings(.env)로 RAG/VLLM/TEI/API 설정 주입
- 실험 러너 스켈레톤: config/dataset을 받아 파이프라인을 실행하도록 확장 가능

## 프로젝트 구조
```
llm-agent-v2/
├── backend/
│   ├── api/                          # FastAPI 라우터 스켈레톤
│   ├── config/                       # 설정(BaseSettings) + retrieval 프리셋 YAML
│   ├── domain/pe_core/               # 도메인 로직 스켈레톤
│   ├── llm_infrastructure/
│   │   ├── preprocessing/
│   │   │   ├── normalize_engine/     # L0~L5 정규화 엔진
│   │   │   ├── adapters/             # 레지스트리 어댑터(standard, domain_specific)
│   │   │   ├── base.py, registry.py  # 공통 베이스/레지스트리
│   │   │   └── README.md             # 전처리 아키텍처 가이드
│   │   ├── embedding/                # 임베더 베이스/레지스트리/샘플(BGE, TEI)
│   │   └── retrieval/                # 리트리버 베이스/레지스트리/프리셋
│   ├── services/ingest/              # 인제스트용 노멀라이저 팩토리
│   └── pyproject.toml                # 백엔드 의존성
├── experiments/                      # 실험 러너/설정/가이드
├── frontend/                         # React 소스/빌드 산출물
├── data/                             # 데이터 폴더(샘플 비포함)
└── docs/                             # 현재 비어 있음(선택적 문서용)
```

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
