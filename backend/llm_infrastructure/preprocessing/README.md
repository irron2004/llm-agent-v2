# Text Preprocessing Infrastructure

문서 전처리를 위한 모듈식 인프라입니다. **엔진-어댑터 분리 패턴**으로 설계되어 팀 협업과 실험이 용이합니다.

## 📐 아키텍처 개요

```
preprocessing/
├── normalize_engine/             # 🔧 엔진: 실제 알고리즘 구현
│   ├── __init__.py        #    → build_normalizer, NormLevel 등 재export
│   ├── factory.py         #    → 정규화 함수 빌더 (L0~L5)
│   ├── base.py            #    → 기본 정규화 (L0, L1, L2, variant 유틸)
│   ├── domain.py          #    → 도메인 특화 (L3, L4, L5)
│   ├── rules.py           #    → 규칙/패턴 데이터
│   └── utils.py           #    → 토큰화/로그 덤프 등 유틸
│
├── adapters/              # 🔌 어댑터: 레지스트리 인터페이스
│   ├── __init__.py
│   ├── normalize.py       #    → NormalizationPreprocessor (L0~L5 선택)
│   ├── standard.py        #    → StandardPreprocessor (예시)
│   └── domain_specific.py #    → DomainSpecificPreprocessor (예시)
│
├── parsers/               # 📄 파서: 포맷별 문서 파싱
│   └── ...
│
├── base.py                # BasePreprocessor 추상 클래스
└── registry.py            # PreprocessorRegistry (등록/선택)
```

## 🎯 핵심 설계 원칙

### 1. 엔진-어댑터 분리

**엔진 (Engine)**: 순수 알고리즘 로직
- 위치: `normalize_engine/` (추후 embedding/retrieval도 동일 패턴)
- 역할: 실제 처리 로직만 구현 (L0~L5)
- 장점: 레지스트리 없이 독립적으로 사용 가능

```python
# 엔진 직접 사용 (레지스트리 불필요)
from preprocessing.normalize_engine import build_normalizer

normalizer = build_normalizer(level="L3", variant_map={...})
result = normalizer("pm2-1에서 오류 발생")
```

**어댑터 (Adapter)**: 레지스트리 연결 레이어
- 위치: `adapters/`
- 역할: 엔진을 레지스트리에 등록, 메타데이터 처리, 설정 주입
- 장점: 파이프라인에서 이름으로 선택/교체 가능

```python
# 어댑터를 통한 레지스트리 사용
from preprocessing.registry import get_preprocessor

preprocessor = get_preprocessor("normalize", version="v1", level="L3")
results = list(preprocessor.preprocess(docs))
```

**핵심 차이**: `normalize_engine`이라는 명명으로 "이게 엔진이다"를 명확히 표현

### 2. 왜 분리하는가?

**핵심 이유**: "엔진은 순수 알고리즘, 어댑터는 서비스 인터페이스"

#### ✅ **1. 엔진은 순수 알고리즘이라 테스트/리팩토링이 쉬움**
```python
# L3 로직 개선해도 레지스트리/파이프라인 쪽은 안 건드림
# normalize_engine/domain.py만 수정
def normalize_l3(text: str) -> str:
    # 개선된 PM 패턴 인식 로직
    ...

# 엔진만 테스트 (레지스트리 없이)
def test_l3_pm_masking():
    norm = build_normalizer("L3")
    assert "PM" in norm("pm2-1 오류")
```

#### ✅ **2. 어댑터는 서비스 인터페이스라 실험/스위칭이 쉬움**
```python
# 설정 파일에서 level만 바꾸면 즉시 전환
# .env: RAG_PREPROCESS_LEVEL=L3 → L5
preprocessor = get_preprocessor("normalize", level=settings.level)

# 같은 L3인데 구현 바꿔도 어댑터 코드는 그대로
```

#### ✅ **3. 명확한 책임 분리 → 팀 병렬 작업**
```
팀원 A: normalize_engine/base.py 개선 (L0~L2)
팀원 B: normalize_engine/domain.py 확장 (L3~L5 반도체 규칙)
팀원 C: adapters/normalize.py 통합 (파이프라인 연결)
→ 파일 충돌 없이 병렬 작업
```

#### ✅ **중복 코드 제거**
```python
# Before: 중복된 정규화 로직
class StandardPreprocessor:
    def preprocess(self, text):
        return text.strip().lower()  # 중복 1

class DomainPreprocessor:
    def preprocess(self, text):
        text = text.strip().lower()  # 중복 2
        # + 도메인 로직

# After: 엔진 재사용
class StandardPreprocessor:
    def preprocess(self, text):
        normalizer = build_normalizer("L0")  # 엔진 재사용
        return normalizer(text)

class DomainPreprocessor:
    def preprocess(self, text):
        normalizer = build_normalizer("L3")  # 엔진 재사용
        return normalizer(text)
```

#### ✅ **유연한 사용 패턴**
```python
# 패턴 1: 엔진 직접 (프로토타입, Jupyter 실험)
norm = build_normalizer("L3")
result = norm("텍스트")

# 패턴 2: 어댑터 통해 (프로덕션 파이프라인)
proc = get_preprocessor("normalize", level="L3")
results = proc.preprocess(docs)

# 패턴 3: 설정 기반 (.env 또는 YAML)
# RAG_PREPROCESS_METHOD=normalize
# RAG_PREPROCESS_LEVEL=L3
proc = get_preprocessor(settings.preprocess_method, level=settings.level)
```

#### ✅ **4. embedding, retrieval에도 동일 패턴 복붙 가능**
동일한 구조를 다른 모듈에도 적용:
```
embedding/
├── openai_engine/    # 엔진
│   ├── gpt3.py
│   └── gpt4.py
├── bge_engine/       # 엔진
│   └── base.py
├── adapters/         # 어댑터
│   ├── openai.py
│   └── bge.py
└── registry.py

retrieval/
├── bm25_engine/     # 엔진
│   └── scorer.py
├── dense_engine/    # 엔진
│   └── searcher.py
├── adapters/        # 어댑터
│   ├── bm25.py
│   └── dense.py
└── registry.py
```

**일관된 명명**: `*_engine/` 패턴으로 "이게 실제 알고리즘"임을 명확히

## 📊 정규화 레벨 (Normalization Levels)

### L0: Basic Normalization
**용도**: 최소한의 정규화 (베이스라인)
```python
from preprocessing.normalize_engine import build_normalizer

norm_l0 = build_normalizer("L0")
# - 기호 표준화 (µm→um, °C→celsius)
# - 공백 정리 (다중 공백 → 단일 공백)
# - 영문 소문자화
```

### L1: + Variant Mapping
**용도**: L0 + 동의어/변형어 치환
```python
norm_l1 = build_normalizer("L1", variant_map={
    "loadlock": "LL",
    "process module": "PM",
    "transfer module": "TM"
})
# L0 + 사용자 정의 동의어 맵 적용
```

### L2: + Extended Rules
**용도**: L1 + 추가 규칙 (현재 L1과 동일, 확장 훅)
```python
norm_l2 = build_normalizer("L2", variant_map={...})
# 향후 확장을 위한 레벨 (예: 수치/단위 결합 등)
```

### L3: Semiconductor Domain Specialized
**용도**: 반도체 장비 유지보수 로그 특화 (권장)
```python
norm_l3 = build_normalizer("L3")
# - PM 모듈 마스킹: pm2-1 → PM
# - 도메인 동의어: exhasut→exhaust, fdc→FDC
# - 과학적 표기법 정규화: 8.0×10^-9 → 0.0000000080
# - Unicode/dash 표준화
```

**예시**:
```python
text = "pm2-1에서 exhasut 압력 8.0×10^-9 mbar*l/s"
result = norm_l3(text)
# → "PM에서 exhaust 압력 0.0000000080 mbar*l/s"
```

### L4: + Advanced Entity Extraction
**용도**: L3 + 엔티티 추출 및 헤더 토큰
```python
norm_l4 = build_normalizer("L4")
# L3 +
# - 모듈/챔버 패턴 추출: [MODULE PM2-1]
# - 알람 코드 추출: [ALARM 123456]
# - 수치 값 표준화: [HE_LEAK 8.00e-09]
# - Spec 상태: [SPEC OUT]
# - 액션 태깅: [ACTION REP]
```

**예시**:
```python
text = "pm2-1 slot valve alarm(123456) 발생, he leak 8.0e-9, spec out, 교체 필요"
result = norm_l4(text)
# → "[MODULE PM2-1] [ALARM 123456] [HE_LEAK 8.00e-09] [SPEC OUT] [ACTION REP] ::
#     PM slot valve alarm 발생, he leak 8.0e-9, spec out, 교체 필요"
```

### L5: + Enhanced Variant Mapping
**용도**: L4 + 현장 표기 변형어 사전
```python
norm_l5 = build_normalizer("L5")
# L4 +
# - 200+ 현장 용어 표준화
# - 단위 표기 통일: 100mt → 100 mTorr
# - 범위 정규화: -400~-500 → [RANGE -500..-400]
# - 설비/모듈 약어: efem→EFEM, ll1→LL1
```

**용어 정규화 예시**:
```
slot vv        → SLOT VALVE
b.g            → BARATRON
he leak check  → HELIUM LEAK CHECK
spec in        → SPEC IN
open/close     → OPEN/CLOSE
pc             → PC (Particle)
```

## 🚀 빠른 시작

### 1. 엔진 직접 사용 (프로토타입/Jupyter)

```python
from preprocessing.normalize_engine import build_normalizer

# L3: 반도체 도메인 특화 (권장)
normalizer = build_normalizer(level="L3")

text = "pm2-1에서 slot vv alarm 발생"
result = normalizer(text)
print(result)  # "PM에서 slot vv alarm 발생"
```

**사용 시기**: 빠른 실험, Jupyter 노트북, 단순 스크립트

### 2. 어댑터 통해 사용 (프로덕션)

```python
from preprocessing.registry import get_preprocessor

# 레지스트리에서 선택
preprocessor = get_preprocessor(
    "normalize",      # 전처리 방법
    version="v1",     # 버전
    level="L3",       # 정규화 레벨
    variant_map={}    # 추가 동의어 맵 (선택)
)

# 문서 배치 처리
docs = ["pm2 장비 오류", "ll1 pressure 상승", ...]
results = list(preprocessor.preprocess(docs))
```

### 3. 설정 기반 사용 (.env)

```bash
# .env 파일
RAG_PREPROCESS_METHOD=normalize
RAG_PREPROCESS_VERSION=v1
RAG_PREPROCESS_LEVEL=L3
```

```python
from backend.config.settings import rag_settings
from preprocessing.registry import get_preprocessor

# 설정에서 자동 로드
preprocessor = get_preprocessor(
    rag_settings.preprocess_method,
    version=rag_settings.preprocess_version,
    level=rag_settings.preprocess_level
)
```

## 🔧 새 전처리 방법 추가

### Step 1: 엔진 구현

새 알고리즘을 구현합니다 (레지스트리 불필요).

```python
# preprocessing/my_custom_engine/cleaner.py
import re

def clean_special_chars(text: str) -> str:
    """특수문자 제거 엔진"""
    text = re.sub(r'[^\w\s가-힣]', '', text)
    return text.strip()

def build_my_cleaner():
    """엔진 팩토리"""
    return clean_special_chars
```

**명명 규칙**: `*_engine/` 패턴으로 엔진임을 명시

### Step 2: 어댑터 생성 (레지스트리 등록)

엔진을 레지스트리에 연결합니다.

```python
# preprocessing/adapters/my_method.py
from typing import Iterable
from ..base import BasePreprocessor
from ..registry import register_preprocessor
from ..my_custom_engine.cleaner import build_my_cleaner

@register_preprocessor("my_method", version="v1")
class MyMethodPreprocessor(BasePreprocessor):
    """커스텀 전처리 어댑터

    역할:
    - my_custom_engine의 로직을 레지스트리에 연결
    - 설정 주입 및 메타데이터 처리
    """

    def __init__(self, **config):
        super().__init__(**config)
        # 엔진 로드 (실제 알고리즘은 엔진에 있음)
        self.cleaner = build_my_cleaner()

    def preprocess(self, docs: Iterable[str]) -> Iterable[str]:
        """전처리 수행 (엔진 호출만 담당)"""
        for doc in docs:
            text = str(doc)
            if not text.strip():
                continue

            # 엔진 호출 (어댑터는 그냥 감싸기만 함)
            result = self.cleaner(text)
            yield result
```

### Step 3: 사용

```bash
# .env
RAG_PREPROCESS_METHOD=my_method
RAG_PREPROCESS_VERSION=v1
```

```python
from preprocessing.registry import get_preprocessor

proc = get_preprocessor("my_method", version="v1")
results = list(proc.preprocess(docs))
```

## 📚 실험 가이드

### 정규화 레벨 비교 실험

```python
from preprocessing.normalize_engine import build_normalizer

# 테스트 문서
test_doc = "pm2-1에서 exhasut alarm(123456) 발생, he leak 8.0×10^-9"

# 레벨별 결과 비교
for level in ["L0", "L1", "L2", "L3", "L4", "L5"]:
    normalizer = build_normalizer(level=level)
    result = normalizer(test_doc)
    print(f"\n{level}: {result}")
```

### 실험 설정 예시

```yaml
# experiments/configs/test_normalize_l3.yaml
name: test_normalize_l3
preprocess_method: normalize
preprocess_version: v1
preprocess_config:
  level: L3
  keep_newlines: true
embedding_method: bge_base
retrieval:
  method: hybrid
  top_k: 50
```

실행:
```bash
python -m experiments.run \
    --config experiments/configs/test_normalize_l3.yaml \
    --dataset data/eval/pe_agent_eval.jsonl
```

## 🧪 테스트

### 엔진 테스트 (단위 테스트)

```python
# tests/preprocessing/test_normalize.py
from preprocessing.normalize_engine import build_normalizer

def test_l3_pm_masking():
    """L3: PM 모듈 마스킹 테스트 (엔진만 테스트)"""
    norm = build_normalizer("L3")

    # PM 주소 마스킹
    assert "PM" in norm("pm2-1에서 오류")
    assert "PM" in norm("PM 2에서 오류")

def test_l4_entity_extraction():
    """L4: 엔티티 추출 테스트 (엔진만 테스트)"""
    norm = build_normalizer("L4")
    result = norm("pm2-1 alarm(123456)")

    # 헤더 토큰 확인
    assert "[MODULE" in result
    assert "[ALARM 123456]" in result
```

**포인트**: 엔진 테스트는 레지스트리 없이 독립적으로 수행

### 어댑터 테스트 (통합 테스트)

```python
# tests/preprocessing/test_normalize_adapter.py
from preprocessing.registry import get_preprocessor

def test_registry_integration():
    """레지스트리 통합 테스트"""
    proc = get_preprocessor("normalize", version="v1", level="L3")

    docs = ["pm2 오류", "ll1 압력 상승"]
    results = list(proc.preprocess(docs))

    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
```

## 🎓 연구 워크플로우

1. **논문/아이디어**: 새 정규화 방법 발견
2. **엔진 구현**: `normalize_engine/` 또는 새 폴더에 알고리즘 작성
3. **빠른 검증**: Jupyter에서 엔진 직접 테스트
4. **어댑터 작성**: 레지스트리 통합이 필요하면 `adapters/` 추가
5. **실험 실행**: YAML 설정으로 베이스라인 비교
6. **결과 분석**: 메트릭 확인 후 반복

## 📋 레벨 선택 가이드

| 레벨 | 용도 | 속도 | 품질 | 추천 시나리오 |
|------|------|------|------|--------------|
| **L0** | 베이스라인 | ⚡⚡⚡ | ⭐ | 일반 텍스트, 프로토타입 |
| **L1** | + 동의어 | ⚡⚡ | ⭐⭐ | 커스텀 용어집 필요 시 |
| **L2** | + 규칙 확장 | ⚡⚡ | ⭐⭐ | (현재 L1과 동일) |
| **L3** | 반도체 도메인 | ⚡⚡ | ⭐⭐⭐⭐ | **PE Agent 권장** |
| **L4** | + 엔티티 추출 | ⚡ | ⭐⭐⭐⭐⭐ | 고급 검색, 필터링 |
| **L5** | + 현장 용어집 | ⚡ | ⭐⭐⭐⭐⭐ | 최고 품질, 프로덕션 |

### 권장 설정

```bash
# 개발/실험: L3 (속도와 품질 균형)
RAG_PREPROCESS_LEVEL=L3

# 프로덕션: L5 (최고 품질)
RAG_PREPROCESS_LEVEL=L5

# 베이스라인: L0 (비교 대상)
RAG_PREPROCESS_LEVEL=L0
```

## 🔍 디버깅

### 정규화 결과 덤프

```python
from preprocessing.normalize_engine import dump_normalization_log, build_normalizers_by_level

docs = [
    ("pm2-1에서 오류", "doc_001"),
    ("ll1 압력 상승", "doc_002"),
]

# 여러 레벨 비교
normalizers = build_normalizers_by_level()
dump_normalization_log(
    docs,
    normalizers,
    path="normalized_comparison.json",
    parallel=True
)

# 결과 확인
# normalized_comparison.json에
# {
#   "doc_id": "doc_001",
#   "text": "원문",
#   "norm_L0": "L0 결과",
#   "norm_L1": "L1 결과",
#   ...
# }
```

### 프로파일 메타 확인

```python
from preprocessing.normalize_engine import build_normalizer

norm = build_normalizer("L4")

# 프로파일 메타 확인
print(norm.__safe_profile__)
# {
#   'level': 'L4',
#   'sanitized_variants': 0,
#   'keep_newlines': True,
#   'use_prejoin': False,
#   'fast_replace': True,
#   'semiconductor_domain': True,
#   'advanced_entity_extraction': True
# }
```

## 🤝 팀 협업 가이드

### 역할 분담 예시

```
팀원 A (알고리즘 전문가): normalize_engine/
├─ normalize_engine/base.py: L0~L2 로직 개선
├─ normalize_engine/domain.py: L3~L5 반도체 규칙 확장
├─ 정규식 패턴 최적화
└─ 단위 테스트 작성 (엔진만)

팀원 B (인프라 담당): adapters/, registry.py
├─ 레지스트리 어댑터 작성
├─ 설정 주입 처리
├─ 통합 테스트 작성 (엔진 + 어댑터)
└─ 메타데이터 관리

팀원 C (서비스 개발): ../services/, ../api/
├─ 어댑터를 서비스에 연결
├─ 설정 관리 (.env, YAML)
├─ E2E 테스트
└─ API 엔드포인트 개발
```

**포인트**: 엔진/어댑터/서비스가 명확히 분리되어 각자의 전문성에 집중

### Git 브랜치 전략

```bash
# 엔진 개선 (알고리즘만 수정)
git checkout -b feature/normalize-engine-l3-enhancement
# normalize_engine/domain.py 수정
git commit -m "feat(engine): L3 PM 패턴 인식 개선"

# 어댑터 추가 (레지스트리 연결만)
git checkout -b feature/add-custom-adapter
# adapters/my_method.py 추가
git commit -m "feat(adapter): 커스텀 전처리 어댑터 추가"

# 서비스 통합 (파이프라인 연결만)
git checkout -b feature/integrate-new-preprocessor
# services/... 수정
git commit -m "feat(service): 새 전처리 파이프라인 통합"

# 독립적인 PR → 충돌 최소화
```

**커밋 컨벤션**:
- `feat(engine):` - 알고리즘 로직 변경
- `feat(adapter):` - 레지스트리 어댑터 변경
- `feat(service):` - 서비스 레이어 변경

## 💡 더 단순한 구조 (1인 개발/소규모 팀)

만약 **"엔진/어댑터 분리가 과하다"**고 느껴진다면:

```
preprocessing/
├── normalize_engine/      # 엔진만 패키지로
│   ├── __init__.py
│   ├── base.py (L0~L2)
│   └── domain.py (L3~L5)
├── normalize_adapter.py   # 어댑터는 단일 파일
├── base.py
└── registry.py            # 실험 안 하면 생략도 가능
```

**언제 이 구조?**
- 1인 개발, 실험이 적음
- 레지스트리가 필요 없음 (엔진 직접 호출)
- 나중에 확장하면 adapters/ 폴더로 쪼개면 됨

**언제 전체 구조?**
- 팀 협업 (2명 이상)
- embedding/retrieval까지 확장 예정
- 실험이 많음 (레지스트리로 스위칭)

→ **처음엔 단순하게, 필요할 때 확장**도 좋은 전략!

## 📝 변경 이력

### v2.1.0 (2025-11-25) - 명명 개선 및 설명 강화
- ✅ `normalize_engine/` → `normalize_engine/` 명명 (역할 명확화)
- ✅ "왜 분리하는가" 섹션 대폭 강화
- ✅ `rules/` → `rules.py` 단순화 (선택사항)
- ✅ 커밋 컨벤션 추가 (`feat(engine):`, `feat(adapter):`)
- ✅ 더 단순한 구조 옵션 제공

### v2.0.0 (2025-11-25) - 엔진-어댑터 분리
- ✅ `normalize.py` (1062줄) → `normalize_engine/` 패키지로 분할
  - `base.py`: L0~L2 (~300줄)
  - `domain.py`: L3~L5 (~500줄)
  - `factory.py`: 빌더 함수
- ✅ `methods/` → `adapters/` 리네임
- ✅ 중복 로직 제거 (Standard/Domain이 엔진 재사용)
- ✅ 팀 협업 친화적 구조

### v1.0.0 - 초기 구조
- `normalize.py`: L0~L5 단일 파일
- `methods/`: 레지스트리 어댑터

## 🔗 관련 문서

- [프로젝트 README](../../../README.md): 전체 아키텍처
- [실험 가이드](../../../experiments/README.md): 실험 실행 방법
- [마이그레이션 가이드](../../../docs/MIGRATION_GUIDE.md): v1→v2 이동

## 📞 문의

- 전처리 알고리즘 관련: `normalize_engine/` 코드 및 테스트 확인
- 레지스트리 통합: `adapters/`, `registry.py` 확인
- 파이프라인 설정: `backend/config/settings.py` 확인
