# RLM 논문 검토 및 프로젝트 적용 가능성 분석

**날짜**: 2026-01-07
**논문**: "Recursive Language Models" (Alex L. Zhang, Tim Kraska, Omar Khattab; MIT CSAIL; arXiv:2512.24601v1, 2025-12-31)

---

## 1. 프로젝트 컨텍스트

### 현재 시스템 구조
- **프로젝트**: 정비로그 기반 QA Agent
- **플로우**: 정비로그 검색 → 관련 문서(SOP/TS-guide/GCB) 검색 → 해결책 제안
- **기술스택**: LangGraph + Elasticsearch + vLLM

### 주요 문제점
1. **멀티홉 추론 실패**: 정비로그 → SOP → TS-guide → GCB 연결이 잘 안 됨
2. **대화 이력 관리**: 토큰 제한으로 과거 대화 참조 불가
   - 연속 질문 시 컨텍스트 손실 (예: "그럼 교체 비용은?")

### 참조 문서 수
- 평균 10개 미만의 문서 참조
- 멀티-문서 연결이 주요 쿼리 유형

---

## 2. RLM 논문 핵심 요약

### 기본 아이디어
> "긴 프롬프트를 LLM 입력으로 넣지 말고, 외부 환경의 '객체(변수)'로 두고
> LLM이 코드로 들여다보며 필요할 때만 자기 자신(또는 서브 LLM)을 재귀적으로 호출"

### 구성 요소
1. **외부 환경 (E)**: Python REPL에 긴 입력을 변수로 저장
2. **루트 LLM**: 코드를 작성해 context를 부분 조회/검색/분해
3. **서브 LLM 호출**: 작은 조각에 대한 의미 판단 수행

### 핵심 실험 결과

| 벤치마크 | 입력 크기 | Base LLM | RLM | 개선 |
|---------|----------|----------|-----|------|
| BrowseComp+ (1K docs) | 6-11M 토큰 | 0.00% | 91.33% | **압도적** |
| OOLONG | 131K 토큰 | 44.00% | 56.50% | +28.4% |
| OOLONG-Pairs | 32K 토큰 | 0.04% | 58.00% | **1450배** |
| CodeQA | 23K-4.2M 토큰 | 24.00% | 62.00% | +158% |

---

## 3. RLM vs Multi-Agent 비교

### 근본적 차이

| | Multi-Agent | RLM |
|---|-------------|-----|
| **입력 저장** | 각 Agent의 LLM 컨텍스트 안 | 외부 메모리 (변수/파일/DB) |
| **입력 크기 한계** | 각 Agent의 윈도우 | 사실상 무제한 |
| **정보 손실** | 요약 과정에서 큼 | 원본 보존 (필요시 재조회) |
| **처리 방식** | 역할별 분업 | 코드 + 서브콜 |

### 비유

**Multi-Agent**:
```
10,000페이지 책을 10명이 1,000페이지씩 나눠 읽고
각자 요약 발표 → 리더가 합침
❌ 초반의 정확한 숫자/세부사항 손실
```

**RLM**:
```
책은 책상에 두고, 목차/색인으로 필요한 부분만 찾아 읽기
코드로 필터링 → 의미 판단만 서브콜
✅ 원본 보존, 정확한 집계 가능
```

---

## 4. 논문의 기여도 비판적 평가

### 기술적 Novelty: ★☆☆☆☆ (낮음)

**솔직한 평가**:
- "검색/필터링 + 필요한 것만 LLM에 넣기" = 이미 다들 하던 것
- REPL에 데이터 두기 = 프로그래밍 101
- 정규식/파싱 = 기본 중의 기본

**사용자 지적이 정확함**:
> "그냥 목차 만들어놓고 검색하는 거 아냐?"
> → 맞습니다. 기술적으로 새로운 건 아닙니다.

### 실제 기여: ★★★★☆ (높음)

**논문의 진짜 가치**:
1. **체계적 평가**: 4개 벤치마크, 입력 길이별/복잡도별 정량 측정
2. **실패 사례 공개**: 언제 비용 폭증하는가, 어떤 버그가 생기는가
3. **OOLONG-Pairs 벤치마크**: 새로운 "이차 복잡도 과업" 제안
4. **설계 선택지별 ablation**: REPL만 vs REPL+서브콜 비교

**결론**:
> 혁명적 기술이 아니라 "좋은 엔지니어링 패턴의 체계적 평가"

---

## 5. 서브콜 설계 가이드 (논문에서 추출)

### 논문에 명시적 가이드가 있는가?
**❌ 아니오.** "이렇게 하라"는 구체적 가이드 없음

### 대신 궤적 분석과 실패 사례에서 추출 가능한 원칙

#### 원칙 1: 코드 우선, 서브콜은 의미 판단만

```python
# ✅ 좋은 예
filtered = grep("ERROR", logs)  # 코드로 필터링
categorized = llm_query(f"카테고리 분류: {filtered}")

# ❌ 나쁜 예
categorized = llm_query(f"로그 전체에서 에러 찾고 분류: {logs}")
```

#### 원칙 2: 서브콜 크기 - 100~1000 항목 또는 200K chars

```python
batch_size = 100  # 논문 BrowseComp+ 사례
for i in range(0, len(docs), batch_size):
    llm_query(docs[i:i+batch_size])
```

**근거**:
- BrowseComp+ 궤적: 100개 문서씩 배치
- Qwen 프롬프트: "200K chars/콜 목표"

#### 원칙 3: 서브콜 횟수 - 5회 이하 권장

```python
MAX_SUBCALLS = 5
subcall_count = 0

while subcall_count < MAX_SUBCALLS:
    result = llm_query(...)
    subcall_count += 1
    if is_done(result):
        break
```

**근거**: 논문에서 수천 번 호출은 비효율적 안티패턴

#### 원칙 4: 검증은 1-2회, 그 이상은 중단

```python
answer = generate_answer()
verified = llm_query(f"검증: {answer}")
if not verified:
    answer = refine_answer()
    verified = llm_query(f"재검증: {answer}")  # 최대 2회
return answer
```

**근거**: 과도한 검증이 비용 폭증

#### 원칙 5: 버퍼 강제 사용

```python
# 서브콜 결과를 버퍼에 누적
buffer = {}
buffer['info'] = llm_query("정보 추출")

# ✅ 최종 출력은 반드시 버퍼에서
final_answer = f"추출 정보: {buffer['info']}"

# ❌ 버퍼 무시하고 재생성하지 마세요
```

**근거**: 부록 B.3 실패 사례 - Qwen이 버퍼에 정답 만들고도 최종 출력에 반영 안 함

---

## 6. 프로젝트 적용 가능성 분석

### 적용 가치: 매우 높음 ✅✅✅

**이유**:
1. **대화 이력 관리** 문제가 RLM의 정확한 use case
2. **멀티홉 추론** 개선도 가능
3. 기존 LangGraph 구조에 노드 추가만으로 점진적 도입 가능

### 추천 접근: Option 1 (경량 적용)

#### Phase 1: 대화 요약 노드 (1주)

```python
class AgentState(TypedDict, total=False):
    # 기존 필드들...
    conversation_history: List[Dict[str, str]]  # NEW
    conversation_summary: str  # NEW

def conversation_summary_node(state, llm, sub_llm):
    """과거 대화를 서브 LLM으로 요약"""
    history = state.get("conversation_history", [])

    if len(history) < 4:  # 2턴 미만
        return {"conversation_summary": ""}

    recent = history[-10:]  # 최근 5턴
    text = '\n'.join([f"{m['role']}: {m['content']}" for m in recent])

    summary = sub_llm.generate(f"""
과거 대화에서 에러코드, 부품번호, 증상만 추출:
{text}
핵심 정보 (3-5줄):
""", max_tokens=512)

    return {"conversation_summary": summary}

def answer_node(state, llm, spec):
    """답변 생성 (대화 요약 포함)"""
    conversation_context = state.get("conversation_summary", "")

    user_prompt = template.user.format(
        query=state['query'],
        refs=format_refs(state['ref_json']),
        conversation_context=conversation_context  # NEW
    )

    return {"answer": llm.generate(user_prompt)}
```

**그래프 수정**:
```python
graph.add_node("conversation_summary", conversation_summary_node)
graph.add_edge("retrieve", "conversation_summary")
graph.add_edge("conversation_summary", "answer")
```

#### Phase 2: 멀티홉 노드 (1주)

```python
def multi_hop_node(state, sub_llm, retriever):
    """정비로그 → SOP → TS-guide → GCB 연결"""
    buffer = {}
    subcall_count = 0
    MAX_SUBCALLS = 5

    # 서브콜 1: 정비로그 분석
    logs = [d for d in state['docs'] if d.metadata['type'] == 'log']
    if logs and subcall_count < MAX_SUBCALLS:
        buffer['error_info'] = sub_llm.generate(f"""
정비로그에서 에러코드, 증상, 부품번호 추출:
{logs[:100]}
""")
        subcall_count += 1

    # 서브콜 2: 에러코드로 SOP 재검색
    if 'error_code' in buffer.get('error_info', {}):
        error_code = buffer['error_info']['error_code']
        sop_docs = retriever.retrieve(
            query=f"SOP {error_code}",
            filters={"doc_type": "sop"},
            top_k=3
        )
        if sop_docs and subcall_count < MAX_SUBCALLS:
            buffer['sop'] = sub_llm.generate(f"절차 추출: {sop_docs}")
            subcall_count += 1

    # 버퍼 기반 문서 생성
    enriched_docs = [
        {"content": f"핵심정보: {buffer.get('error_info', '')}"},
        {"content": f"SOP절차: {buffer.get('sop', '')}"},
    ]

    return {"docs": state['docs'] + enriched_docs}
```

### 비용 추정

```
기존: 1회 LLM 호출 (답변 생성)
추가:
- 대화 요약: 1회 서브 LLM (저렴한 모델)
- 멀티홉: 2-3회 서브 LLM

예상 비용 증가: 10-20%
```

---

## 7. 주의사항 (논문의 실패 사례 기반)

### 안티패턴 1: 라인당 서브콜
**Qwen 실패 사례**: 수천 라인을 각각 서브콜 → 비용 폭증
**대응**: 최소 100줄 이상 배치

### 안티패턴 2: 과도한 검증
**GPT-5 비효율**: 검증을 5-6회 반복
**대응**: 검증은 1-2회로 제한

### 안티패턴 3: 버퍼 무시
**Qwen 실패**: 버퍼에 정답 만들고도 최종 출력에 반영 안 함
**대응**: answer_node에서 버퍼 강제 참조

### 비용 관리
- 서브콜 최대 횟수 제한 (5회)
- 비용 모니터링 (긴 꼬리 주의)
- 대화 이력 4턴 미만이면 요약 스킵

---

## 8. 최종 결론

### RLM의 본질 (최종 이해)

```
┌─────────────────────────────────────────┐
│  외부 메모리 (LLM 컨텍스트 밖)           │
│  - 긴 prompt (과거 대화 50턴)            │
│  - 또는 대용량 문서/로그                 │
└─────────────────────────────────────────┘
              ↑
              │ 검색/필터링
              │ (dense/BM25/정규식/목차/인덱스)
              ↓
┌─────────────────────────────────────────┐
│  LLM에게 실제로 제공하는 것              │
│  - 검색된 일부분만                       │
└─────────────────────────────────────────┘
```

**핵심**:
> "긴 것을 LLM '안'에 넣지 말고 '밖'에 두고,
> 필요한 부분만 검색해서 보자"

### 실무 적용 권장사항

**✅ DO**:
- 논문의 핵심 패턴만 가져가기 (외부 저장 + 검색 + 서브콜)
- 실험 결과를 참고 자료로 활용
- 실패 사례를 회피 전략으로 사용

**❌ DON'T**:
- "RLM"이라는 이름에 현혹되지 말기
- 논문이라서 도입하지 말고, 패턴이 유용해서 도입
- 완전한 REPL 구현보다는 점진적 적용

### 다음 단계

1. **즉시 적용 가능**: 대화 요약 노드 구현 (1주)
2. **점진적 확장**: 멀티홉 노드 추가 (1주)
3. **A/B 테스트**: 비용/성능 검증 (2주)

---

## 9. 구현 복잡도 재평가 (중요한 수정)

### 초기 평가의 오류

**처음 평가:**
> "REPL + 서브콜 + 재귀는 과도하다"

**이 평가가 잘못된 이유:**

#### 착각 1: "REPL = 복잡한 환경 구축"

**제가 생각한 것 (잘못됨):**
```python
- 진짜 Python REPL 환경 구축
- 샌드박스, 보안 설정
- 코드 실행 인프라
→ 복잡해 보임
```

**실제로는:**
```python
# "REPL" = 그냥 변수
conversation_history = [...]  # 외부 저장
docs = [...]  # 외부 저장

# 이게 전부. 별로 안 복잡함.
```

#### 착각 2: "서브콜 = 복잡한 관리 시스템"

**제가 생각한 것 (잘못됨):**
```python
- 서브콜 관리 시스템
- 큐, 비동기 처리
- 복잡한 오케스트레이션
→ 복잡해 보임
```

**실제로는:**
```python
# 서브콜 = LLM 함수 한 번 더 호출
result = sub_llm.generate(prompt)

# 이게 전부. 1줄 추가.
```

#### 착각 3: "재귀 = 여러 단계 깊이"

**제가 생각한 것 (잘못됨):**
```python
- 복잡한 재귀 구조
- 여러 단계 깊이
→ 복잡해 보임
```

**실제로는:**
```python
# 논문도 깊이 1만 사용
# 사실상 재귀 아님, 그냥 서브콜
```

### 수정된 평가

**실제 구현 복잡도:**

```python
def multi_hop_node(state, retriever, sub_llm):
    # 1차 검색 (기존 코드)
    logs = retriever.retrieve(state['query'])

    # 서브콜 추가 (새로 추가되는 코드 - 단 3줄!)
    if logs:
        extracted = sub_llm.generate(f"에러코드 추출: {logs[:3]}")  # ← 이게 전부

    # 2차 검색 (기존과 유사)
    sop_docs = retriever.retrieve(f"SOP {extracted['error_code']}")

    return {"docs": logs + sop_docs}
```

**추가된 것:**
- `sub_llm.generate()` 호출: **1줄**
- if 조건문: **2줄**

**총 추가 코드**: **3줄**

**복잡도 증가**: **거의 없음**

### 왜 착각했는가

**논문의 용어가 혼란을 줌:**
- "REPL" → 마치 복잡한 환경 구축 같음
- "Recursive" → 마치 복잡한 재귀 구조 같음
- "Language Models" (복수) → 마치 여러 모델 관리 같음

**실제로는:**
- 변수 저장
- 함수 호출
- 단순 반복

### 정규식 vs 서브콜 비교

#### 정규식의 한계

**케이스 1: 명확한 패턴**
```python
# 정규식으로 충분
log = "ERROR E1234 at line 42"
error_code = re.search(r'E\d{4}', log).group()  # "E1234"
```

**케이스 2: 모호한 표현**
```python
# 정규식 실패
log = "센서 이상으로 인한 작동 불량"
error_code = re.search(r'E\d{4}', log)  # None

# 서브콜 필요
info = sub_llm.generate(f"이 로그에서 관련 SOP 찾기 위한 키워드: {log}")
# → {"keyword": "센서", "category": "하드웨어"}
```

#### 서브콜 추가의 비용

```
서브콜 1회 (GPT-4o-mini): $0.01
멀티홉당 2-3회: $0.02-0.03

전체 요청 비용: $0.50 (기존)
→ $0.52-0.53 (4-6% 증가)
```

**무시할 수 있는 수준**

---

## 10. 최종 수정 추천

### 현재 시스템 문제 재진단

**문제 1: 멀티홉 추론 실패**

현재 코드 분석 결과:
```python
# backend/llm_infrastructure/llm/langgraph_agent.py:395
def retrieve_node(state):
    queries = state.get("search_queries", [])
    for q in queries:
        all_docs.append(retriever.retrieve(q))  # 동시 검색
    return top_k
```

**진짜 문제**:
- Multi-query (여러 쿼리 동시 검색) ✅ 구현됨
- Multi-hop (순차 검색) ❌ 구현 안 됨

**예시:**
```
현재: ["정비로그 E1234", "SOP", "TS-guide"] 동시 검색
     → E1234를 아직 모르는 상태에서 SOP 검색 → 실패

필요: 정비로그 검색 → E1234 발견 → "SOP E1234" 재검색
```

### 추천 구현 (우선순위)

#### 1순위: 멀티홉 검색 + 하이브리드 추출 ⭐⭐⭐

**정규식 먼저, 실패시 서브콜**

```python
def multi_hop_retrieve_node(state, retriever, sub_llm):
    """순차적 멀티홉 검색"""

    # 1차: 정비로그 검색
    logs = retriever.retrieve(
        state['query'],
        filters={"doc_type": "maintenance_log"},
        top_k=3
    )

    # 키워드 추출 (정규식 먼저 시도)
    error_codes = extract_by_regex(logs)  # 정규식
    symptoms = extract_symptoms_regex(logs)
    part_numbers = extract_parts_regex(logs)

    # 정규식 실패 시에만 서브콜
    if not any([error_codes, symptoms, part_numbers]) and logs:
        extracted = sub_llm.generate(f"""
정비로그에서 추출:
{'\n'.join([d.content for d in logs[:3]])}

JSON 형식으로 반환:
{{
    "error_codes": ["..."],
    "symptoms": ["..."],
    "part_numbers": ["..."]
}}
""", max_tokens=256)

        error_codes = extracted.get('error_codes', [])
        symptoms = extracted.get('symptoms', [])
        part_numbers = extracted.get('part_numbers', [])

    # 2차: SOP 검색 (에러코드 기반)
    sop_docs = []
    for code in error_codes[:3]:  # 최대 3개
        sop_docs.extend(retriever.retrieve(
            f"SOP {code}",
            filters={"doc_type": "sop"},
            top_k=2
        ))

    # 3차: TS-guide 검색 (증상 기반)
    ts_docs = []
    for symptom in symptoms[:2]:  # 최대 2개
        ts_docs.extend(retriever.retrieve(
            f"{symptom}",
            filters={"doc_type": "ts_guide"},
            top_k=2
        ))

    # 4차: GCB 검색 (부품번호 기반)
    gcb_docs = []
    for part in part_numbers[:2]:
        gcb_docs.extend(retriever.retrieve(
            f"part {part}",
            filters={"doc_type": "gcb"},
            top_k=1
        ))

    # 통합 및 중복 제거
    all_docs = logs + sop_docs + ts_docs + gcb_docs
    return {"docs": deduplicate(all_docs)[:10]}
```

**특징:**
- 정규식으로 충분한 경우: 서브콜 0회
- 정규식 실패 시에만: 서브콜 1회
- 평균 비용 증가: 2-5%

#### 2순위: 대화 이력 관리 (Sliding Window + 선택적 요약) ⭐⭐

```python
def conversation_context_node(state, sub_llm):
    """대화 컨텍스트 생성"""
    history = state.get("conversation_history", [])

    # 짧으면 그냥 전체 사용
    if len(history) <= 4:  # 2턴 이하
        context = '\n'.join([f"{m['role']}: {m['content']}" for m in history])
        return {"conversation_context": context}

    # 길면 최근 5턴 + 키워드 기반 검색
    recent = history[-5:]

    # 현재 질문에서 키워드 추출 (정규식)
    current_query = state['query']
    keywords = extract_keywords_simple(current_query)  # 명사/고유명사

    # 과거 턴에서 키워드 매칭
    relevant = [
        turn for turn in history[:-5]  # 최근 5턴 제외
        if any(kw in turn['content'] for kw in keywords)
    ][-3:]  # 최대 3턴

    # 합치기
    combined = relevant + recent
    context = '\n'.join([f"{m['role']}: {m['content']}" for m in combined])

    # 여전히 길면 서브콜로 요약
    if len(context) > 10000:  # chars
        context = sub_llm.generate(f"""
과거 대화 요약 (에러코드, 부품번호, 핵심 문제만):
{context}

요약 (5줄 이내):
""", max_tokens=512)

    return {"conversation_context": context}
```

**특징:**
- 대부분 경우: 서브콜 0회 (sliding window + 키워드 매칭)
- 매우 긴 경우만: 서브콜 1회 (요약)

### 구현 우선순위

| 순위 | 작업 | 기간 | 복잡도 | 예상 효과 |
|------|------|------|--------|-----------|
| 1 | 멀티홉 검색 (정규식 우선) | 3일 | 낮음 | 높음 (80%+) |
| 2 | 대화 이력 sliding window | 2일 | 낮음 | 중간 (50%+) |
| 3 | 선택적 서브콜 추가 | 1일 | 낮음 | 높음 (edge case) |

**총 소요 기간**: 1주 이내
**총 코드 추가량**: 100줄 미만
**비용 증가**: 5% 미만

---

## 11. 결론 (최종 수정)

### RLM 패턴의 실제 복잡도

**오해:**
> "REPL + 서브콜 + 재귀 = 복잡한 시스템"

**실제:**
> "변수 저장 + 함수 호출 + 반복 = 단순한 코드"

**Fancy한 이름이 혼란을 줌:**
- "Recursive Language Models" → 사실 재귀 거의 안 씀
- "REPL 환경" → 사실 그냥 변수
- "서브 LLM 호출" → 사실 함수 호출 1줄

### 실무 적용 최종 권장

**✅ 적극 권장:**
1. **멀티홉 검색**: 정규식 우선 + 서브콜 보완
2. **대화 이력**: Sliding window + 선택적 요약
3. **점진적 도입**: 간단한 것부터 시작

**❌ 불필요:**
1. 완전한 REPL 환경 구축
2. 복잡한 재귀 구조
3. 과도한 서브콜 관리 시스템

**핵심 교훈:**
> 논문의 "아이디어"는 훌륭하지만,
> 논문의 "용어"에 현혹되지 말 것.
> 실제 구현은 놀랍도록 단순함.

---

## 참고자료

- 논문 원문: arXiv:2512.24601v1
- 프로젝트 플로우 문서: `/docs/agent.md`
- LangGraph 구현: `/backend/services/agents/langgraph_rag_agent.py`
- 현재 retrieve_node: `/backend/llm_infrastructure/llm/langgraph_agent.py:395`
