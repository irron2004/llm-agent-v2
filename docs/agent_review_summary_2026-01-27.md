# Agent 구성 및 프롬프트 리뷰 (2026-01-27)

시니어에게 **Agent 구성, Flow, Prompt 조정 포인트**를 전달하기 위한 문서입니다.

---

## 1. 데이터의 종류

### 1.1 AgentState (핵심 상태 객체)

Agent가 처리하는 모든 데이터를 담는 통합 상태 객체입니다.

| 그룹 | 필드 | 타입 | 설명 |
|------|------|------|------|
| **쿼리** | `query` | str | 원본 사용자 질문 |
| | `query_en`, `query_ko` | str | 번역된 영문/한글 버전 |
| | `detected_language` | str | 자동감지 언어 (ko, en, ja) |
| **라우팅** | `route` | str | setup \| ts \| general |
| | `st_gate` | str | need_st \| no_st (순차적 사고 필요 여부) |
| **검색어 생성** | `setup_mq_list`, `ts_mq_list`, `general_mq_list` | list[str] | 경로별 영문 MQ |
| | `setup_mq_ko_list`, `ts_mq_ko_list`, `general_mq_ko_list` | list[str] | 경로별 한글 MQ |
| | `search_queries` | list[str] | 최종 검색에 사용할 통합 쿼리 |
| **자동 파싱** | `auto_parsed_device`, `auto_parsed_devices` | str/list | 질문에서 추출한 장비명 |
| | `auto_parsed_doc_type`, `auto_parsed_doc_types` | str/list | 문서 종류 추출 |
| | `auto_parse_message` | str | 사용자에게 표시할 파싱 결과 |
| **필터링** | `selected_devices` | list[str] | 사용자가 선택한 장비 (OR 필터) |
| | `selected_doc_types` | list[str] | 선택한 문서 종류 |
| | `selected_doc_ids` | list[str] | 특정 문서 ID 필터 |
| **검색 결과** | `docs` | list[RetrievalResult] | 최종 rerank된 문서 (10개) |
| | `all_docs` | list[RetrievalResult] | 재생성용 전체 문서 (20개) |
| | `display_docs` | list[RetrievalResult] | UI 표시용 병합 문서 |
| | `ref_json`, `answer_ref_json` | str | JSON 형식 검색 결과 |
| **답변** | `answer` | str | 생성된 최종 답변 |
| | `reasoning` | str | 추론 과정 (reasoning model) |
| | `judge` | dict | 충실성 평가 `{"faithful": bool, "issues": [...], "hint": "..."}` |
| **재시도** | `attempts` | int | 현재 재시도 횟수 |
| | `max_attempts` | int | 최대 재시도 횟수 |
| | `expand_top_k` | int | 문서 확장 개수 (기본 5, 1차 재시도시 10) |
| | `retry_strategy` | str | expand_more \| refine_queries \| regenerate_mq |

**위치**: `backend/llm_infrastructure/llm/langgraph_agent.py:68`

### 1.2 RetrievalResult (검색 결과)

| 필드 | 설명 |
|------|------|
| `doc_id` | 문서 고유 ID |
| `content` | 전처리된 텍스트 (검색용) |
| `raw_text` | 원본 텍스트 (LLM 컨텍스트용) |
| `score` | 유사도 점수 |
| `metadata` | doc_type, device_name, page, chunk_id 등 |

**위치**: `backend/llm_infrastructure/retrieval/base.py:8`

### 1.3 기타 데이터

| 데이터 | 위치 | 설명 |
|--------|------|------|
| 문서 청크 스키마 | `backend/llm_infrastructure/elasticsearch/mappings.py` | ES 매핑 정의 |
| 문서 타입 그룹 | `backend/domain/doc_type_mapping.py` | SOP, ts, setup, gcb, myservice |
| 장비 목록 캐시 | `backend/services/device_cache.py` | Auto-parse용 장비 목록 |

---

## 2. Agent 위치 및 파일 구조

```
backend/
├── services/agents/
│   └── langgraph_rag_agent.py          # Agent 서비스 (그래프 조립) ★
│
├── llm_infrastructure/llm/
│   ├── langgraph_agent.py              # 노드 구현 + AgentState 정의 ★
│   ├── base.py                         # BaseLLM 인터페이스
│   ├── prompt_loader.py                # 프롬프트 로드
│   └── prompts/                        # YAML 프롬프트 (17개) ★
│       ├── router_v1.yaml
│       ├── setup_mq_v1.yaml
│       ├── ts_mq_v1.yaml
│       ├── general_mq_v1.yaml
│       ├── st_gate_v1.yaml
│       ├── st_mq_v1.yaml
│       ├── setup_ans_v1.yaml
│       ├── ts_ans_v1.yaml
│       ├── general_ans_v1.yaml
│       ├── setup_ans_en_v1.yaml
│       ├── ts_ans_en_v1.yaml
│       ├── general_ans_en_v1.yaml
│       ├── setup_ans_ja_v1.yaml
│       ├── ts_ans_ja_v1.yaml
│       ├── general_ans_ja_v1.yaml
│       ├── auto_parse_v1.yaml
│       └── translate_v1.yaml
│
├── api/routers/
│   └── agent.py                        # FastAPI 라우터 + HIL 로직
│
└── domain/
    └── doc_type_mapping.py             # 문서 종류 그룹화
```

### 핵심 파일 역할

| 파일 | 역할 | 라인 수 |
|------|------|---------|
| `langgraph_rag_agent.py` | 그래프 조립, 모드 분기, 의존성 주입 | ~450줄 |
| `langgraph_agent.py` | 모든 노드 함수, AgentState, PromptSpec | ~2000줄 |
| `prompts/*.yaml` | 프롬프트 템플릿 | 17개 파일 |

---

## 3. Agent Flow (워크플로우)

### 3.1 전체 Flow 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                              START                                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
    [Auto-Parse]          [Device-Select]           [기본]
         │                       │                       │
         ▼                       ▼                       │
   ┌───────────┐          ┌───────────┐                  │
   │auto_parse │          │  route    │                  │
   └─────┬─────┘          └─────┬─────┘                  │
         │                      │                        │
         ▼                      ▼                        │
   ┌───────────┐          ┌─────────────────┐            │
   │ translate │          │device_selection │            │
   └─────┬─────┘          │   (interrupt)   │            │
         │                └────────┬────────┘            │
         ▼                         │                     │
   ┌───────────┐                   │                     │
   │   route   │◄──────────────────┴─────────────────────┘
   └─────┬─────┘
         │
         ▼
   ┌───────────┐     setup_mq_list (EN 3개)
   │    mq     │──▶  ts_mq_list (EN 3개)
   └─────┬─────┘     general_mq_list (EN 3개)
         │           *_mq_ko_list (KO 3개씩)
         ▼
   ┌───────────┐
   │  st_gate  │──▶  need_st | no_st
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │   st_mq   │──▶  search_queries (3~6개 혼합)
   └─────┬─────┘
         │
         ▼
   ┌───────────┐     ES 검색 + Rerank
   │ retrieve  │──▶  docs (10개), all_docs (20개)
   └─────┬─────┘
         │
         ├───────────────────┐ (ask_user_after_retrieve=True)
         │                   ▼
         │            ┌─────────────┐
         │            │  ask_user   │ (interrupt)
         │            └──────┬──────┘
         │                   │
         │         ┌─────────┴─────────┐
         │         │                   │
         │    [승인]                [거절/수정]
         │         │                   │
         │         │                   ▼
         │         │         ┌───────────────────┐
         │         │         │ refine_and_retrieve│
         │         │         └─────────┬─────────┘
         │         │                   │
         ▼         ▼                   │
   ┌─────────────────┐◄────────────────┘
   │ expand_related  │──▶  display_docs, answer_ref_json
   └────────┬────────┘
            │
            ▼
   ┌───────────┐
   │  answer   │──▶  answer, reasoning
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │   judge   │──▶  {"faithful": bool, "issues": [...], "hint": "..."}
   └─────┬─────┘
         │
         ▼
   ┌──────────────┐
   │ should_retry │ (조건부 라우팅)
   └──────┬───────┘
          │
    ┌─────┼─────┬─────────┬─────────────┐
    │     │     │         │             │
 [faithful]  [1차]     [2차]        [3차+]    [max 초과]
    │     │     │         │             │
    ▼     ▼     ▼         ▼             ▼
  ┌────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
  │done│ │retry_exp │ │retry_bump│ │ retry_mq │ │human_review  │
  └─┬──┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │ (interrupt)  │
    │         │            │            │       └──────┬───────┘
    │         │            ▼            │              │
    │         │     ┌──────────────┐    │              │
    │         │     │refine_queries│    │              │
    │         │     └──────┬───────┘    │              │
    │         │            │            │              │
    │         │            ▼            │              │
    │         │     ┌──────────────┐    │              │
    │         │     │retrieve_retry│    │              │
    │         │     └──────┬───────┘    │              │
    │         │            │            │              │
    │         ▼            ▼            ▼              │
    │    [expand_related로]  [mq로]                    │
    │                                                  │
    ▼                                                  ▼
  ┌─────────────────────────────────────────────────────────┐
  │                          END                             │
  └─────────────────────────────────────────────────────────┘
```

### 3.2 노드별 역할

| 노드 | 입력 | 처리 | 출력 | 위치 |
|------|------|------|------|------|
| `route` | query | 3분류 LLM | route | :633 |
| `auto_parse` | query + device_names | 규칙기반 추출 | auto_parsed_* | :553 |
| `translate` | query + detected_language | 이중언어 번역 | query_en, query_ko | :599 |
| `mq` | query_en/ko + route | 경로별 검색어 생성 | *_mq_list (EN+KO) | :684 |
| `st_gate` | query + all MQs | 명확도 평가 | st_gate | :708 |
| `st_mq` | all MQs + gate | 쿼리 통합/정제 | search_queries | :761 |
| `retrieve` | search_queries | ES 검색 + Rerank | docs, all_docs | :844 |
| `expand_related` | docs | 인접 페이지 확장 | display_docs | :984 |
| `answer` | display_docs + route | 경로별 답변 생성 | answer, reasoning | :1209 |
| `judge` | answer + docs | 충실성 평가 | judge (JSON) | :1344 |

### 3.3 재시도 전략 (Verified Mode)

```
┌─────────────────────────────────────────────────────────────────┐
│                       재시도 전략 (3단계)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  attempt=0 (1차 재시도)                                          │
│  ├─ 전략: expand_more                                            │
│  ├─ 동작: expand_top_k 증가 (5 → 10)                            │
│  └─ 효과: 기존 문서에서 더 많은 컨텍스트 활용                     │
│                                                                  │
│  attempt=1 (2차 재시도)                                          │
│  ├─ 전략: refine_queries                                         │
│  ├─ 동작: judge.hint 기반 쿼리 개선 후 재검색                    │
│  └─ 효과: 누락된 정보를 찾기 위한 새 검색                        │
│                                                                  │
│  attempt=2+ (3차 이상 재시도)                                    │
│  ├─ 전략: regenerate_mq                                          │
│  ├─ 동작: MQ 목록 초기화 후 처음부터 재생성                      │
│  └─ 효과: 완전히 새로운 검색 전략                                │
│                                                                  │
│  max_attempts 초과                                               │
│  └─ Human Review (interrupt) → 사용자 최종 판단                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**분기 로직 위치**: `backend/llm_infrastructure/llm/langgraph_agent.py:1378`

---

## 4. Prompt 내용

### 4.1 프롬프트 목록

| 프롬프트 | 파일 | 목적 | 입력 | 출력 |
|----------|------|------|------|------|
| **router** | `router_v1.yaml` | 질문 분류 | query | setup\|ts\|general |
| **setup_mq** | `setup_mq_v1.yaml` | 설치 검색어 생성 | query + route | 3개 검색어 |
| **ts_mq** | `ts_mq_v1.yaml` | 트러블슈팅 검색어 | query + route | 3개 검색어 |
| **general_mq** | `general_mq_v1.yaml` | 일반 검색어 | query + route | 3개 검색어 |
| **st_gate** | `st_gate_v1.yaml` | 쿼리 명확도 판단 | query + MQs | need_st\|no_st |
| **st_mq** | `st_mq_v1.yaml` | 최종 쿼리 통합 | MQs + gate | JSON (3~6개) |
| **setup_ans** | `setup_ans_v1.yaml` | 설치 답변 생성 | query + refs | 한글 답변 |
| **ts_ans** | `ts_ans_v1.yaml` | 트러블슈팅 답변 | query + refs | 한글 답변 |
| **general_ans** | `general_ans_v1.yaml` | 일반 답변 | query + refs | 한글 답변 |
| **\*_ans_en** | `*_ans_en_v1.yaml` | 영문 답변 | query + refs | 영문 답변 |
| **\*_ans_ja** | `*_ans_ja_v1.yaml` | 일문 답변 | query + refs | 일문 답변 |
| **auto_parse** | `auto_parse_v1.yaml` | 장비/문서/언어 추출 | query | JSON |
| **translate** | `translate_v1.yaml` | 이중언어 번역 | query | 번역 텍스트 |

### 4.2 Judge 프롬프트 (코드 내 정의)

Judge는 YAML이 아닌 코드 내 상수로 정의됩니다.

| 상수 | 역할 | 위치 |
|------|------|------|
| `DEFAULT_JUDGE_SETUP` | 설치 답변 충실성 평가 | :128 |
| `DEFAULT_JUDGE_TS` | 트러블슈팅 답변 평가 | :128 |
| `DEFAULT_JUDGE_GENERAL` | 일반 답변 평가 | :128 |

**출력 형식**:
```json
{"faithful": true/false, "issues": ["부족한 근거", ...], "hint": "재검색 힌트"}
```

### 4.3 주요 프롬프트 내용 요약

#### Router Prompt
```
# Role: 반도체 장비 Q&A를 setup/ts/general로 분류

# Label Definitions
- setup: 설치, 교체, 분해, 조립, 초기화, 캘리브레이션
- ts: 알람/에러/경고, 트러블슈팅, 원인분석
- general: 일반 설명, 정책, 명확하지 않은 것들

# Priority: ts > setup > general
# Output: ONE lowercase word only
```

#### MQ Prompt (예: setup_mq)
```
Generate exactly 3 search queries for setup/installation.
- Q1: Topic + Component + Specific Parameters
- Q2: Abbreviations/Synonyms
- Q3: Validation/Spec Keywords
- Use SAME LANGUAGE as input

Output: 3 lines only (NO numbering)
```

#### Answer Prompt (예: setup_ans)
```
당신은 설치/셋업 RAG 어시스턴트입니다.
- REFS 라인만 증거로 사용
- 단계별 절차(번호 목록)로 답변
- 각 단계별 [1] 형식으로 출처 인용
- 반드시 한국어로 답변
```

---

## 5. 핵심 특징

| 기법 | 설명 |
|------|------|
| **다중쿼리 (MQ)** | 3~6개 다양한 검색어로 재현율 향상 |
| **이중언어 검색** | 영문/한글 병렬 검색 |
| **순차적 사고 (ST)** | Gate로 명확도 판단 후 필요시 쿼리 정제 |
| **Rerank** | 20개 → 10개 cross-encoder 재순위 |
| **Judge** | 답변 충실성 자동 평가 |
| **3단계 재시도** | Expand → Refine → Regenerate |
| **언어 자동 감지** | ko/en/ja 감지 후 언어별 프롬프트 적용 |

---

## 6. 조정 포인트 요약

시니어와 논의할 때 아래 파일들을 중심으로 확인하면 됩니다.

| 조정 대상 | 파일 |
|-----------|------|
| **Flow/재시도 전략** | `backend/services/agents/langgraph_rag_agent.py` |
| **노드 로직/분기** | `backend/llm_infrastructure/llm/langgraph_agent.py` |
| **프롬프트 본문** | `backend/llm_infrastructure/llm/prompts/*.yaml` |
| **Judge 프롬프트** | `langgraph_agent.py` 내 `DEFAULT_JUDGE_*` 상수 |
| **문서타입 그룹** | `backend/domain/doc_type_mapping.py` |

### 프롬프트 수정 시 주의사항

- 프롬프트는 `lru_cache`로 캐싱됨 → **서버 재시작 필요**
- 캐시 위치: `backend/api/dependencies.py:176`
