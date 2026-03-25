# 도메인 사전 기반 약어 모호성 해소 (Abbreviation Disambiguation)

날짜: 2026-03-19
최종 수정: 2026-03-23 (Phase 3/4 계획 추가)

## 1. 문제 정의

### 현상

사용자가 반도체 도메인 약어(예: "AR")로 질문할 때, 의도하지 않은 동음이의 문서가 함께 검색됨.

```
[사용자] "AR 측정이 낮아요"
  → 의도: Ashing Rate
  → 실제 검색 결과: Ashing Rate 문서 + Atomic Ratio 문서 모두 hit
```

### 원인

- **BM25**: "AR" 토큰이 포함된 모든 문서가 매칭됨 — 의미 구분 불가
- **Dense(embedding)**: 짧은 약어 "AR"의 임베딩은 문맥 없이 어느 의미인지 구분력이 약함
- 문서 원본은 이미 인덱싱/임베딩 완료 → **재인덱싱 없이** 해결 필요

### 영향 범위

약어 사용이 일상적인 반도체 PE 현장에서, 약어 모호성으로 인한 검색 정밀도 저하는 사용자 신뢰도에 직접적 영향.

## 2. 도메인 사전 현황

### 파일: `data/semicon_word.json`

| 항목 | 수치 |
|------|------|
| 총 용어 | 543개 |
| 줄임말 있는 항목 | 274개 (50.5%) |
| 줄임말 없는 항목 | 269개 (49.5%) |

### 사전 가공 결과 (Union-Find 병합)

원본 543개 엔트리에 동일 개념 중복이 다수 존재 (예: "CD", "CDs", "CRITICAL DIMENSION (CD)" → 모두 Critical Dimension).
약어/영어명/한국어명이 겹치는 엔트리를 Union-Find로 병합:

| 항목 | 수치 |
|------|------|
| 병합 후 고유 개념 | 445개 |
| 고유 약어 토큰 | 231개 |
| **1:1 (단일 의미)** | 231개 (100%) — 자동 치환 대상 |
| **1:N (다중 의미)** | 0개 — 사전 내 실제 모호 약어 없음 |

> 사전에 등록되지 않은 의미(예: AR = Ashing Rate)는 engineer에게 데이터를 받아 `semicon_word.json`에 추가하면 1:N이 발생하여 사용자에게 선택 요청 플로우가 활성화됨.

### 사전 항목 구조

```json
{
  "Anneal": {
    "뜻": "웨이퍼를 고온에서 열처리해 ...",
    "동의어": "열처리",
    "영어": "Anneal",
    "한국어": "어닐 공정",
    "줄임말": "",
    "사용예시": "어닐 공정 후 트랜지스터 전류가 증가했다."
  },
  "ABATEMENT": {
    "뜻": "...",
    "동의어": ["gas abatement", "emissions abatement", ...],
    "영어": "Abatement System",
    "한국어": "배출가스 저감 장치",
    "줄임말": "ABMT",
    "사용예시": "..."
  }
}
```

## 3. 해결 전략

### 3.1 런타임 약어 인덱스 빌드

서버 시작 시 `semicon_word.json`에서 약어 → 개념 매핑을 메모리에 빌드:

1. 엔트리 파싱 (줄임말, 영어명, 한국어명, 동의어)
2. Union-Find로 동일 개념 병합
3. 약어 토큰 → 개념 ID 인덱스 생성
4. 1:1 / 1:N 분류

중간 파일(`semicon_word_index.json`) 불필요 — `semicon_word.json` 수정 → 서버 재시작으로 자동 반영.

### 3.2 쿼리 처리 흐름

```
사용자 질문 입력
    │
    ▼
[약어 감지] ─── 쿼리 토큰을 abbr_index와 매칭
    │
    ├─ 약어 없음 → 기존 플로우 진행
    │
    ├─ 1:1 약어만 → 자동 치환 후 기존 플로우 진행
    │    예: "ESC 온도 이상" → "Electrostatic Chuck (ESC) 온도 이상"
    │
    └─ 모호한 약어 있음 → 사용자에게 선택 요청 (인터럽트)
         예: "AR이 여러 의미로 사용됩니다:
              1) Ashing Rate (애싱 속도)
              2) Atomic Ratio (원자 비율)
              어느 의미인가요?"
         → 사용자 선택 → 치환 → 기존 플로우 진행
```

### 3.3 치환 방식

약어를 풀네임으로 **치환**하되, 원본 약어도 유지:

```
원본:  "CVD 챔버 클리닝 주기"
치환:  "chemical vapor deposition (CVD) 챔버 클리닝 주기"
```

이유:
- 문서에 "CVD"만 있고 "chemical vapor deposition"이 없을 수 있으므로 약어도 유지
- 문서에 풀네임이 있으면 BM25에서 더 높은 점수
- Dense 검색 시 풀네임이 포함된 쿼리 임베딩이 더 정확한 의미 포착

### 3.4 치환 제외 토큰

일반 영어 단어와 충돌 가능한 약어는 자동 치환에서 제외:
`AI`, `IC`, `NA`, `MO`, `PR`, `LED`, `DIE`

필요 시 `_SKIP_TOKENS`에 추가/제거 가능.

## 4. 구현 현황

### Phase 1: 자동 치환 (1:1) — 완료 (2026-03-23)

| 파일 | 역할 | 상태 |
|------|------|------|
| `data/semicon_word.json` | 원본 도메인 사전 (543개 용어) | 기존 |
| `backend/llm_infrastructure/query_expansion/abbreviation_expander.py` | 약어 확장 모듈 (런타임 빌드 + 치환) | **신규** |
| `backend/config/settings.py` | `abbreviation_expand_enabled`, `abbreviation_dict_path` 설정 | **수정** |
| `backend/llm_infrastructure/llm/langgraph_agent.py` | `retrieve_node()`에서 검색 전 약어 확장 적용 | **수정** |

#### 통합 위치

```
retrieve_node() 내부:

queries = [normalize(q) for q in search_queries]  # 기존
    ↓
[약어 확장] ← AbbreviationExpander.expand_query()  # 추가됨
    ↓
retriever.retrieve(q, ...)  # 기존
```

`auto_parse_node` 전이 아닌, `retrieve_node` 내부에서 검색 쿼리에 직접 적용.
이유: multi-query 확장된 쿼리들에도 약어 치환이 적용되어야 하므로.

#### 설정

```bash
# 환경변수로 제어
AGENT_ABBREVIATION_EXPAND_ENABLED=true   # 기본값: true
AGENT_ABBREVIATION_DICT_PATH=data/semicon_word.json  # 기본값
```

#### 로깅

모든 치환이 INFO 레벨로 기록됨:

```
# 서버 시작 시
[AbbreviationExpander] built index from 'data/semicon_word.json': 543 entries → 445 concepts, 231 abbreviation tokens
[AbbreviationExpander] loaded: 445 concepts, 231 abbreviations (1:1=231, 1:N=0)

# 쿼리 치환 시
[AbbreviationExpander] EXPANDED: 'CVD' → 'chemical vapor deposition' (화학 기상 증착) | query: 'CVD 챔버 클리닝' → 'chemical vapor deposition (CVD) 챔버 클리닝'

# 모호 약어 skip 시
[AbbreviationExpander] AMBIGUOUS skip: 'AR' in query '...' — candidates: concept_id=42 (Ashing Rate / 애싱 속도)
```

#### 테스트 결과

| 입력 | 출력 | 상태 |
|------|------|------|
| `AR 측정이 낮아요` | `Aspect Ratio (AR) 측정이 낮아요` | 치환 |
| `CMP 후 파티클 문제` | `chemical-mechanical planarization (CMP) 후 파티클 문제` | 치환 |
| `ALD 증착 두께가 spec out` | `Atomic Layer Deposition (ALD) 증착 두께가 spec out` | 치환 |
| `EFEM 로봇 에러 발생` | `equipment front-end modules (EFEM) 로봇 에러 발생` | 치환 |
| `CVD 챔버 클리닝 주기` | `chemical vapor deposition (CVD) 챔버 클리닝 주기` | 치환 |
| `ESC 온도 제어 이상` | `Electrostatic Chuck (ESC) 온도 제어 이상` | 치환 |
| `STI 식각 프로파일 확인` | `Shallow Trench Isolation (STI) 식각 프로파일 확인` | 치환 |
| `PECVD oxide deposition` | `Plasma-Enhanced Chemical Vapor Deposition (PECVD) oxide deposition` | 치환 |
| `HC 교체 절차를 알려줘` | (변경 없음) | skip — 사전 미등록 |
| `일반 한국어 질문입니다` | (변경 없음) | skip — 약어 없음 |

### Phase 2: 모호 약어 인터럽트 (1:N) — 완료 (2026-03-23)

| 파일 | 역할 | 상태 |
|------|------|------|
| `backend/llm_infrastructure/query_expansion/abbreviation_expander.py` | Union-Find 병합 시 이름 유사성 검사 추가 | **수정** |
| `backend/llm_infrastructure/llm/langgraph_agent.py` | `abbreviation_resolve_node` 추가 + `retrieve_node` 사용자 선택 적용 | **수정** |
| `backend/services/agents/langgraph_rag_agent.py` | 그래프에 `abbreviation_resolve` 노드 연결 | **수정** |
| `data/semicon_word.json` | Ashing Rate (AR) 항목 추가 → 1:N 테스트 | **수정** |

#### Union-Find 병합 개선

기존: 같은 약어를 공유하면 무조건 같은 개념으로 병합 → 다른 개념이 하나로 합쳐지는 문제.
개선: 약어 기반 병합 시 **이름 유사성 검사** 추가:
- 한국어명: 포함 관계 비교 (exact match → containment)
- 영어명: 약어 형태(5자 이하) 또는 설명문(60자 초과)이면 같은 개념으로 간주
- 핵심 단어 겹침: 3글자 이상 영단어의 50% 이상 겹치면 같은 개념

결과: 544개 엔트리 → 446개 개념, 1:1=230, **1:N=1** (AR)

#### 노드 흐름

```
auto_parse → auto_parse_confirm → abbreviation_resolve → history_check → ...
                                       │
                                       ├─ 모호 약어 없음 → history_check (바로 통과)
                                       ├─ 이미 해소됨 → history_check (바로 통과)
                                       └─ 모호 약어 있음 → interrupt (사용자 선택 요청)
                                            → 선택 결과를 state["abbreviation_selections"]에 저장
                                            → retrieve_node에서 선택된 풀네임으로 치환
```

#### Interrupt payload (프론트엔드 연동)

```json
{
  "type": "abbreviation_resolve",
  "question": "AR 측정이 낮아요",
  "instruction": "다음 약어의 의미를 선택해주세요.",
  "abbreviations": [
    {
      "token": "AR",
      "abbr_key": "AR",
      "options": [
        {"value": "90", "label": "Aspect Ratio (종횡비)", "eng": "Aspect Ratio", "kr": "종횡비"},
        {"value": "445", "label": "Ashing Rate (애싱 속도)", "eng": "Ashing Rate", "kr": "애싱 속도"}
      ]
    }
  ]
}
```

사용자 응답:
```json
{
  "type": "abbreviation_resolve",
  "selections": {"AR": "445"}
}
```

#### 프론트엔드 UI — 미구현

`resolveInterruptKind()`에 `"abbreviation_resolve"` 타입 추가 + 선택 UI 컴포넌트 필요.
기존 `GuidedSelectionPanel`의 패턴을 따라 구현 예정.

### Phase 3: 약어 확장의 MQ/프롬프트 반영 — 미착수

> Phase 1~2에서 약어 확장이 `retrieve_node` 검색 쿼리에만 적용되는 한계를 해결.

#### 3.1 문제: 현재 약어 확장 적용 범위의 한계

```
현재 흐름:
state["query"] = "CVD 챔버 클리닝" (원본, 약어 그대로)
    → mq_node: "CVD"를 모른 채 multi-query 생성 ← 문제 A
    → st_mq_node: 최종 검색 쿼리 선별
    → retrieve_node: 여기서만 "CVD" → "chemical vapor deposition (CVD)" 치환
    → answer_node: prompt에 용어 정의 없음 ← 문제 B
```

**문제 A — MQ에 약어 의미 미반영**

| 단계 | 현재 | 개선 후 |
|------|------|---------|
| MQ 입력 | `"CVD 챔버 클리닝"` | `"chemical vapor deposition (CVD) 챔버 클리닝"` |
| MQ 출력 예시 | `"CVD cleaning cycle"` (의미 불명확) | `"CVD chamber cleaning interval"`, `"chemical vapor deposition maintenance"` (의미 정확) |
| translate | `"CVD"` 번역 누락 가능 | 풀네임 포함으로 정확한 번역 |

MQ 노드는 `state["query"]` (또는 `query_en`/`query_ko`)를 읽어 다양한 검색 쿼리를 생성.
약어가 풀네임 없이 입력되면, LLM이 약어의 도메인 의미를 정확히 파악하지 못해 생성되는 multi-query 품질이 저하됨.

**문제 B — 답변 프롬프트에 용어 정의 부재**

현재 `answer_node`의 프롬프트 mapping:
```python
mapping = {"sys.query": query, "ref_text": ref_text}
```

검색된 문서(`ref_text`)만 참조하고, 용어 자체의 정의/의미를 제공하지 않음.
반도체 PE는 도메인 전문 용어가 많아, LLM이 문서에서 해당 용어의 정의를 찾지 못하면 일반적/부정확한 답변 생성 가능.

사전에 등록된 `뜻`, `사용예시` 필드를 프롬프트에 추가하면:
- LLM이 용어 정의를 이미 알고 있으므로 답변 정확도 향상
- 검색된 문서가 부족하더라도 기본적인 도메인 지식 제공

#### 3.2 해결 방안 A: 약어 확장의 MQ 반영

**방법**: `abbreviation_resolve_node`에서 1:1 치환 + 사용자 선택(1:N)을 모두 적용하여 `state["query"]`를 업데이트.
이후 모든 하위 노드(translate, mq, st_mq, retrieve, answer)가 확장된 쿼리로 동작.

```
개선 흐름:
abbreviation_resolve_node:
  "CVD 챔버 클리닝" → "chemical vapor deposition (CVD) 챔버 클리닝"
  state["query"] 업데이트 (+ query_original 보존)
    → translate_node: 풀네임 기반 정확한 EN/KO 번역
    → mq_node: 풀네임 기반 multi-query 생성
    → retrieve_node: 추가 치환 불필요 (이미 확장됨)
    → answer_node: 확장된 쿼리가 sys.query에 포함
```

**변경사항**:
- `abbreviation_resolve_node`: 1:1 약어도 치환하여 `state["query"]` 업데이트
- `AgentState`: `query_original` 필드 추가 (원본 보존, UI 표시용)
- `retrieve_node`: 이미 확장된 쿼리이므로 중복 치환 방지 로직 추가

**리스크**:
- 풀네임이 포함된 긴 쿼리가 MQ 생성에 부정적 영향을 줄 가능성 → MQ 품질 A/B 테스트 필요
- `query_original` vs `query` 혼동으로 인한 UI 표시 오류

#### 3.3 해결 방안 B: 답변 프롬프트에 용어 정의 주입

**방법**: 쿼리에서 감지된 약어의 `뜻`과 `사용예시`를 답변 프롬프트에 컨텍스트로 추가.

```
[도메인 용어 참고]
- Ashing Rate (AR, 애싱 속도): 단위 시간당 애싱 공정에서 제거되는 포토레지스트 또는 유기물의 두께.
  플라즈마 애싱 공정의 효율을 나타내는 핵심 지표.
  사용예시: "AR이 낮아서 레지스트 제거가 완전하지 않다."
```

**변경사항**:
- `AbbreviationExpander.expand_query()`: 매칭된 약어의 `meaning`과 `usage_example`도 반환
- `abbreviation_resolve_node` 또는 `retrieve_node`: 감지된 용어 정의를 `state["term_definitions"]`에 저장
- `answer_node`: prompt mapping에 `{term_context}` 변수 추가
- 답변 프롬프트 템플릿 수정: `{term_context}`를 참조 섹션에 포함

**리스크**:
- 프롬프트 길이 증가로 토큰 비용 상승 (용어당 ~50-100 토큰)
- 사전 정의와 검색 문서 내용이 충돌할 경우 LLM 혼동 가능 → 사전 정의는 "참고" 레벨로 제한

#### 3.4 A/B 우선순위

| 방안 | 효과 | 구현 난이도 | 리스크 | 우선순위 |
|------|------|-------------|--------|----------|
| A. MQ 반영 | MQ 품질 향상 + 검색 정밀도 | 낮음 (노드 수정) | MQ 길이 증가 영향 | **1순위** |
| B. 프롬프트 주입 | 답변 정확도 향상 | 중간 (프롬프트 수정) | 토큰 비용, 충돌 | **2순위** |

A는 기존 코드 수정만으로 가능하고 검색 품질에 직접 영향.
B는 프롬프트 템플릿 변경이 필요하지만 답변 품질에 직접 영향.
둘은 독립적이므로 순차 또는 병렬 진행 가능.

### Phase 4: 동의어 확장 (선택적) — 미착수

- "열처리" → "열처리 anneal" (동의어 추가)
- "연마" → "연마 CMP" (관련 용어 추가)
- BM25 `should` 절로 동의어 쿼리 추가

## 5. 사전 업데이트 절차

### 새 용어 추가

1. `data/semicon_word.json`에 항목 추가
2. `make prod-up` 또는 API 서버 재시작
3. 로그에서 빌드 결과 확인: `[AbbreviationExpander] built index from ...`

### 모호 약어 등록 (1:N)

같은 줄임말에 다른 의미를 추가하면 자동으로 1:N으로 분류됨:

```json
{
  "Ashing Rate": {
    "영어": "Ashing Rate",
    "한국어": "애싱 속도",
    "줄임말": "AR",
    ...
  },
  "Aspect Ratio": {
    "영어": "Aspect Ratio",
    "한국어": "종횡비",
    "줄임말": "AR",
    ...
  }
}
```

→ Phase 2 구현 후, "AR" 입력 시 사용자에게 선택 요청.
→ Phase 2 미구현 상태에서는 두 의미 모두 치환하지 않고 skip (로그에 AMBIGUOUS 기록).

## 6. 리스크 및 고려사항

| 리스크 | 대응 |
|--------|------|
| 약어가 일반 영어 단어와 충돌 (예: "AI", "PR") | `_SKIP_TOKENS`로 제외 |
| 자동 치환이 검색 품질 저하 | `AGENT_ABBREVIATION_EXPAND_ENABLED=false`로 즉시 비활성화 |
| 치환된 풀네임이 너무 길어 검색 품질 왜곡 | `pick_primary_eng()`에서 60자 이내 제한 |
| 약어가 문서 제목/장비명의 일부인 경우 | 단어 경계(word boundary) 매칭으로 부분 매칭 방지 |
| 사전 수정 후 반영 안 됨 | 싱글톤 — 서버 재시작 필요 (로그로 확인 가능) |

## 7. 예상 효과

| 항목 | 효과 |
|------|------|
| BM25 precision | 약어 → 풀네임 치환으로 풀네임이 있는 문서 점수 상승 |
| Dense recall | 풀네임 포함 쿼리 임베딩이 더 정확한 의미 포착 |
| 사용자 경험 | 대부분 자동 처리, 모호한 경우만 한번 물어봄 |
| 유지보수 | `semicon_word.json` 수정 + 서버 재시작으로 반영 |
| 추적성 | 모든 치환이 INFO 로그로 기록되어 예상하지 못한 변경 감지 가능 |
