# RLM 적용: 대화 이력 ES 저장 구조와 문서 참조 슬롯

## 배경/목표
- 대화 히스토리를 컨텍스트에 직접 넣지 않고, 환경(ES + 문서 스토리지)에 두고 필요할 때만 부분 관측하는 RLM 스타일로 확장한다.
- 현재 "히스토리 1개 제한"을 완화하고, "이전 1번 문서" 같은 참조 요청을 안정적으로 처리한다.

## RLM 핵심 개념 정리

### 기존 방식 vs RLM 방식

| 구분 | 기존 RAG | RLM |
|-----|---------|-----|
| 히스토리 위치 | LLM 컨텍스트에 직접 삽입 | 환경(ES/DB)에 저장 |
| 접근 방식 | 전체 히스토리를 토큰으로 소비 | 필요한 부분만 검색/로드 |
| 재검색 | 새 LLM 호출 필요 | 같은 세션에서 반복 가능 |
| 원본 확인 | 불가 (이미 잘림/요약됨) | 가능 (환경에 원본 유지) |

### RLM의 본질
- **"검색 방법"이 핵심이 아님** - 정규식이든 벡터든 상관없음
- **핵심은 "상태 유지 + 반복 상호작용"**
  - 검색 결과를 LLM 컨텍스트에 "넣는" 게 아니라 환경에 "두고"
  - 필요할 때 다시 접근, 재검색, 검증 가능
  - "검토 → 재시도 → 검증" 루프를 환경 안에서 수행

### 비용 구조 변화
```
기존 RAG: 검색 → 결과를 LLM 토큰으로 소비 → 한 번에 추론
RLM:      검색 → 결과를 환경에 저장 → 필요한 조각만 sub-LLM 호출
```

## 주요 시나리오
- "이전 1번 문서를 참고해서 답변해줘"
- "이전 1번 문서의 전체 문서를 보여줘"
- 과거 턴의 질문/답변/근거 문서를 빠르게 찾아 검증/재활용

## 저장 설계(ES)
### 1) conv_turns (대화 턴)
한 턴 = user + assistant 쌍. 문서 참조 슬롯(doc_refs)을 반드시 기록한다.

필드(핵심):
- session_id, turn_id, ts
- user_text, assistant_text
- doc_refs[] (nested)
  - slot: 사용자에게 보여준 번호(1, 2, 3...)
  - doc_id, title, source, uri
  - chunk_ids or snippet (근거로 사용한 부분)
  - retrieval_method, score (선택)
- summary (선별용 요약)
- summary_model, summary_ts (요약 생성 메타)
- schema_version (마이그레이션용)

예시 매핑:
```json
{
  "mappings": {
    "properties": {
      "session_id": { "type": "keyword" },
      "turn_id": { "type": "integer" },
      "ts": { "type": "date" },
      "user_text": { "type": "text" },
      "assistant_text": { "type": "text" },
      "summary": { "type": "text" },
      "summary_model": { "type": "keyword" },
      "summary_ts": { "type": "date" },
      "schema_version": { "type": "keyword" },
      "doc_refs": {
        "type": "nested",
        "properties": {
          "slot": { "type": "integer" },
          "doc_id": { "type": "keyword" },
          "title": { "type": "text", "fields": { "raw": { "type": "keyword" } } },
          "source": { "type": "keyword" },
          "uri": { "type": "keyword" },
          "chunk_ids": { "type": "keyword" },
          "snippet": { "type": "text" }
        }
      }
    }
  }
}
```

### 2) docs (문서 메타)
문서 원문은 별도 스토리지에 두고 ES에는 메타만 저장.

필드(핵심):
- doc_id, title, source, uri, version, checksum
- storage_pointer (S3/파일 경로 등)
- (권장) 문서 스냅샷 키: 버전 또는 checksum을 doc_refs에도 기록

### 3) doc_chunks (문서 청크)
문서 본문을 검색/부분 관측하기 위한 청크 저장소.

필드(핵심):
- doc_id, chunk_id, order, section, text, offset

## "이전 1번 문서" 처리 규칙
- 답변 생성 시 doc_refs에 slot 번호를 반드시 기록한다.
- "이전 1번 문서 전체" 요청:
  - 최근 턴의 doc_refs[slot=1] -> doc_id -> docs.storage_pointer로 전체 문서 반환
- "이전 1번 문서 참고" 요청:
  - doc_id의 chunk를 검색/peek해서 필요한 부분만 LLM에 제공

## 문서 슬롯 운영 규칙(권장)
- 슬롯 번호는 **각 답변마다 1부터 재할당**한다.
- "이전 1번 문서"는 **가장 최근 doc_refs가 있는 assistant 턴**을 기본 스코프로 한다.
- 사용자가 "세션 전체 기준"을 명시한 경우에만 세션 전체 누적 슬롯으로 해석한다.
- 모호할 경우(예: 최근 답변에 문서가 없을 때) 사용자에게 확인을 요청한다.

## 세션 상태 캐시(선택)
- 빠른 해석을 위해 세션별로 `last_doc_turn_id`를 캐시(Redis 또는 ES 별도 인덱스)한다.
- 캐시가 없거나 무효하면 ES에서 최근 doc_refs가 있는 턴을 역순 조회한다.

## 히스토리 선별 전략

### 선택: 가벼운 LLM 기반 선별 (Option C)

| 방식 | 검색 방법 | 장점 | 단점 |
|-----|---------|-----|-----|
| A. 키워드 | 질문에서 키워드 추출 → 히스토리 검색 | 빠름, 단순 | 의미적 연관 놓침 |
| B. 벡터 | 질문 임베딩 → 히스토리 임베딩 유사도 | 의미 기반 | 임베딩 비용 |
| **C. LLM 판단** | LLM이 히스토리 목록 보고 관련 턴 선택 | **정확** | LLM 호출 추가 |

**선택 이유**: 가벼운 LLM(Haiku, GPT-4o-mini 등)을 사용하면 비용 부담 적으면서 의미 기반 판단 가능

### 히스토리 선별 플로우

```
새 질문: "아까 말한 E-1234 관련해서..."
                ↓
┌─────────────────────────────────────────────────────┐
│         Step 1: 히스토리 요약 목록 생성               │
│                                                     │
│  turn 1: "slot valve 교체 절차 질문 → 5단계 답변"     │
│  turn 2: "E-1234 에러 원인 질문 → 센서 불량 답변"     │
│  turn 3: "PM 주기 질문 → 3개월 주기 답변"            │
└─────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────┐
│  (선택) Step 1.5: 최근 N턴/키워드로 1차 필터링       │
│                                                     │
│  N=50 등으로 제한해 LLM 선별 비용을 제어             │
└─────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────┐
│      Step 2: 가벼운 LLM이 관련 턴 선택               │
│                                                     │
│  Input: 히스토리 요약 목록 + 새 질문                  │
│  Output: [2]  ← "E-1234 관련이므로 turn 2 선택"      │
└─────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────┐
│      Step 3: 선택된 턴 전체 내용 로드                 │
│                                                     │
│  turn 2의 전체 내용 (질문 + 답변 + 참조문서)          │
└─────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────┐
│      Step 4: 메인 LLM 답변 생성                      │
│                                                     │
│  컨텍스트: 선택된 히스토리 + 새 검색 결과 + 새 질문    │
└─────────────────────────────────────────────────────┘
```

## RLM 스타일 대화 이력 루프(요약)
1) list_turns/search_turns로 후보 턴 좁힘
2) get_turn으로 해당 턴의 user/assistant 텍스트 일부만 읽음
3) get_turn_docs로 doc_refs 확인
4) 필요 시 doc_chunks에서 부분 관측
5) 의미 판단이 필요한 조각만 subcall
6) 버퍼에 근거/결론 조립 후 응답

## 가드레일(필수)
- subcall 횟수/토큰/시간 예산 제한
- search/peek 단계의 최대치 제한
- 긴 궤적 발생 시 중단 또는 요약 모드로 전환
- doc_refs 누락/문서 미존재 시 즉시 폴백 처리

## 예외 처리/폴백(권장)
- doc_refs가 비어 있으면 "이전 1번 문서" 요청을 거절하고 재질문
- 문서 원문 접근 실패 시: 마지막 정상 스냅샷 또는 캐시된 snippet으로 대체
- 요약 누락 시: 최근 N턴만 사용하거나 요약 생성 후 재시도

## 구현 컴포넌트

### 1) HistoryStore (ES 기반)

```python
class HistoryStore:
    """ES 기반 대화 이력 저장소"""

    def __init__(self, es_client, index_name: str = "conv_turns"):
        self.es = es_client
        self.index = index_name

    def add_turn(self, session_id: str, turn: HistoryTurn) -> None:
        """새 턴 저장 (질문 + 답변 + 참조문서 + 요약)"""
        doc = {
            "session_id": session_id,
            "turn_id": turn.turn_id,
            "ts": turn.timestamp,
            "user_text": turn.user_text,
            "assistant_text": turn.assistant_text,
            "doc_refs": turn.doc_refs,
            "summary": turn.summary,  # 선별용 요약 (저장 시점에 생성)
            "summary_model": turn.summary_model,
            "summary_ts": turn.summary_ts,
            "schema_version": "v1",
        }
        doc_id = f"{session_id}:{turn.turn_id}"
        self.es.index(index=self.index, id=doc_id, body=doc, routing=session_id)

    def get_summaries(self, session_id: str, limit: int = 200) -> list[TurnSummary]:
        """세션의 모든 턴 요약 목록 반환 (선별용)"""
        query = {
            "query": {"term": {"session_id": session_id}},
            "sort": [{"turn_id": "asc"}],
            "size": limit,
            "_source": ["turn_id", "summary"],
        }
        hits = self.es.search(index=self.index, body=query, routing=session_id)["hits"]["hits"]
        return [
            TurnSummary(turn_id=h["_source"]["turn_id"], summary=h["_source"]["summary"])
            for h in hits
        ]

    def get_turns(self, session_id: str, turn_ids: list[int]) -> list[HistoryTurn]:
        """선택된 턴들의 전체 내용 로드"""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"session_id": session_id}},
                        {"terms": {"turn_id": turn_ids}}
                    ]
                }
            }
        }
        hits = self.es.search(index=self.index, body=query, routing=session_id)["hits"]["hits"]
        return [HistoryTurn.from_es(h["_source"]) for h in hits]
```

### 2) HistorySelector (가벼운 LLM)

```python
class HistorySelector:
    """가벼운 LLM으로 관련 히스토리 선별"""

    def __init__(self, light_llm):
        self.llm = light_llm  # e.g., Haiku, GPT-4o-mini

    def select_relevant(
        self,
        summaries: list[TurnSummary],
        new_question: str,
        max_select: int = 3
    ) -> list[int]:
        """새 질문과 관련된 턴 번호 반환"""

        # 요약 목록을 텍스트로 변환
        summary_text = "\n".join([
            f"turn {s.turn_id}: {s.summary}"
            for s in summaries
        ])

        prompt = f"""다음은 이전 대화 요약 목록입니다:

{summary_text}

새 질문: {new_question}

새 질문에 답변하는 데 참고할 만한 이전 대화를 선택하세요.
관련 없으면 빈 배열을 반환하세요.
최대 {max_select}개까지 선택 가능합니다.

응답 형식 (JSON 배열만): [1, 3] 또는 []"""

        response = self.llm.generate([{"role": "user", "content": prompt}])

        # JSON 파싱
        import json
        try:
            return json.loads(response.text.strip())
        except:
            return []
```

### 3) LangGraph 노드 통합

```python
# langgraph_agent.py에 추가할 노드

def retrieve_history_node(state: AgentState) -> AgentState:
    """RLM 스타일 히스토리 검색 노드"""

    session_id = state.get("session_id")
    query = state["query"]

    # Step 1: 요약 목록 가져오기
    summaries = history_store.get_summaries(session_id)

    if not summaries:
        state["history_context"] = []
        return state

    # Step 2: 가벼운 LLM으로 관련 턴 선별
    relevant_ids = history_selector.select_relevant(summaries, query, max_select=3)

    if not relevant_ids:
        state["history_context"] = []
        return state

    # Step 3: 선택된 턴 전체 로드
    relevant_turns = history_store.get_turns(session_id, relevant_ids)

    # Step 4: state에 저장 (메인 LLM이 사용)
    state["history_context"] = relevant_turns

    return state
```

### 4) 그래프 구조 변경

```
기존:
  START → route → mq → retrieve → answer → ...

변경 후:
  START → retrieve_history → route → mq → retrieve → answer → ...
           ↑
       새 노드 추가
```

## 턴 요약 생성 시점

| 시점 | 방식 | 장점 | 단점 |
|-----|-----|-----|-----|
| **저장 시점 (권장)** | 답변 생성 후 즉시 요약 | 선별 시 추가 비용 없음 | 저장 시 약간의 지연 |
| 선별 시점 | 필요할 때 요약 생성 | 저장 빠름 | 선별 시 N번 LLM 호출 |

**권장**: 저장 시점에 요약 생성 (가벼운 LLM으로 1회 호출)

```python
# 답변 저장 시
summary = light_llm.generate(f"다음 대화를 한 줄로 요약: Q:{question} A:{answer[:500]}")
turn.summary = summary.text  # e.g., "E-1234 에러 원인 질문 → 센서 불량으로 답변"
history_store.add_turn(session_id, turn)
```

## 결정 필요 사항
- "이전 1번 문서" 기준: 직전 턴 기준 vs 세션 전체 기준
- 문서 원문 저장소: S3/파일/DB 등
- 문서 평균/최대 크기 (ES에 원문을 둘지 결정에 영향)
- **가벼운 LLM 선택**: Haiku / GPT-4o-mini / 로컬 경량 모델
- **히스토리 저장소**: ES 단독 / ES + Redis 캐시

## TODO (확인 필요)
- "이전 1번 문서" 기본 해석을 **최근 doc_refs 턴 기준**으로 확정할지
- 문서 원문 저장소와 버전/스냅샷 정책 결정 (S3/파일/DB)
- 요약 생성용 경량 모델 확정 (Haiku/4o-mini/로컬)
