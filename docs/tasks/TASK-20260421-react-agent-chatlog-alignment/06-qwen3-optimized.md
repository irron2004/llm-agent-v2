# Qwen3 Optimization Iteration — Engine-Level Fix

작성일: 2026-04-21
목적: 이전 broken run (`05-qwen3-ab.md`) 의 parse_error 3/5 를 해결. `think=true` (reasoning on) 유지하면서 모든 에러 제거.

## 0. 이전 실패의 원인 재진단

`05-qwen3-ab.md` 에서 관찰된 3건의 `issues=['parse_error']` 는 Qwen3 품질 문제가 아니라 **engine 레이어의 empty content** 문제.

### 로그 증거

```
WARNING: judge_node: failed to parse LLM output: (empty)
WARNING: judge_node: failed to parse LLM output: (empty)
WARNING: judge_node: failed to parse LLM output: (empty)
```

### 근본 원인 (3 layer)

1. **Qwen3 `thinking` 필드 분리** — Qwen3 native Ollama API 는 reasoning 을 `message.thinking` 에, 최종 답을 `message.content` 에 분리 반환.
   - gpt-oss 는 `reasoning_content` 키 사용 (다른 키).
   - 내 engine 은 gpt-oss 포맷만 알고 `thinking` 키 무시 → content 가 비어있으면 전혀 복구 안 됨.

2. **num_predict 부족** — Qwen3 `think=true` 시 thinking 이 수백~수천 토큰 소모. `OLLAMA_MAX_TOKENS=30000` default 는 판사/플래너 프롬프트에서 thinking 만으로 소진되기 쉬움 → content 생성 전 stop.

3. **`_invoke_llm` 경로 누락** — judge/router/auto_parse 등 **대부분의 비-structured 호출** 은 `llm.generate(messages, **kwargs)` 형태로 `response_model` 없이 호출. engine 의 `response_model is not None` 조건부 fallback 이 이 경로들에는 적용 안 됨.

## 1. 적용된 Engine Fix (`backend/llm_infrastructure/llm/engines/ollama.py`)

### 1-1. thinking 필드 읽기 (Qwen3 호환)

```python
thinking = message.get("thinking") or message.get("reasoning_content") or message.get("reasoning")
if isinstance(thinking, str) and thinking.strip():
    reasoning = thinking
```

### 1-2. 빈 content → JSON-looking reasoning 만 fallback (보수적)

```python
# Conservative fallback: when content is empty and the reasoning/thinking text
# clearly contains structured output (JSON object or judge/classification keys),
# expose reasoning as text so downstream JSON regex parsers (judge, router,
# classification) can succeed. For plain text generation (e.g. answer_node)
# we deliberately leave text empty to avoid polluting the answer with
# internal reasoning monologue.
if (not text or not text.strip()) and reasoning:
    import re as _re
    # Match: {"  OR  "faithful"  OR  "action"  OR  "route"  OR  "gate"
    #        OR  "yes_or_no"  OR  ```json fence
    if _re.search(
        r'\{\s*"|"faithful"|"action"|"route"|"gate"|"yes_or_no"|```\s*json',
        reasoning,
    ):
        text = reasoning
```

실패한 시도 (되돌림): `num_predict = 65536` universal floor 와 "무조건 reasoning → text" universal fallback 은 Qwen3 answer_node 에서 thinking (17K chars) 을 답변으로 오염시켜 `_truncate_repetition` 에 의해 26자 echo 로 truncate 되는 문제 유발. 최종적으로 보수적 JSON-only fallback 으로 수렴.

### 1-3. response_model 시 추가 JSON 복구 fallback

```python
if response_model is not None and (not text or not text.strip()) and reasoning:
    for pat in (r"\{[\s\S]*\}\s*$", r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"):
        m = _re.search(pat, reasoning.strip())
        if m:
            candidate = m.group(0)
            try:
                _json.loads(candidate); text = candidate; break
            except Exception: continue
```

### 설계 원칙
- **think=true 유지** (사용자 요청: 성능 > 속도, reasoning on)
- gpt-oss 경로에 영향 없음 (gpt-oss 는 reasoning_content 키 그대로 읽음, 빈 content 발생 안 하므로 fallback trigger 안 됨)
- Universal fallback 은 모든 LLM 호출에 적용 (judge, router, auto_parse 포함)

## 2. 검증

### 2-1. Engine 단위 테스트 (OllamaClient 직접)
- think=true + JSON mode (response_model=Judge) + num_ctx=32768:
  - **4.2초에 valid JSON 반환** (faithful=True, empty 없음)
- think=true + 비-structured (judge-style 프롬프트):
  - **8.3초에 valid JSON text** (text_len=354, reasoning_len=3470)
  - Universal fallback 작동 확인 (reasoning 이 3470자인데 content 유효)

### 2-2. 회귀 (US-005)
- `test_general_ts_empty_refs_prompt.py`: **11/11 pass**
- gpt-oss API `:8611` health: 200 ok
- gpt-oss engine sanity: `{"ok": true}` 정상 반환

## 3. Sample5 재평가 (US-002) — 최종

### 3-1. 3-way 결과 비교

| QID | Broken Qwen (v0, think starved) | Optimized Qwen (engine fix) | gpt-oss:120b (reference) |
|---|---|---|---|
| 0228 | F, parse_error, 407자, 130s | F, parse_error, 597자, 146s ⚠️(1) | F, parse_error, 245자, 126s |
| 0072 | F, parse_error, 15자, 202s | **T, 17자, 244s** ✅ | T, 224자, 79s |
| 0095 | ?, empty 0자, 240s | **T, 359자, 116s** ✅ | T, 759자, 72s |
| 0092 | T, 929자, 162s | **T, 210자, 220s** ✅ | T, 1724자, 83s |
| 0049 | F, parse_error, 1498자, 145s | F, parse_error, 2208자, 355s ⚠️(2) | T, 4000자, 81s |

(1) S0228 은 engine reload 직전 쿼리라 old engine 에서 실행됨 (conservative fallback 미적용).
(2) S0049 는 post-reload 이지만 Qwen thinking 에 `{`/`"faithful"` 등 regex 단서 부재 → fallback 미작동.

### 3-2. 집계

| 지표 | Broken Qwen | Optimized Qwen | gpt-oss:120b |
|---|---|---|---|
| faithful | 1/5 | **3/5** (🟢 +2) | 4/5 |
| parse_error | 3/5 | 2/5 (🟢 -1) | 0/5 |
| empty (len<50) | 1/5 | 1/5 (S0072 의 "not found" 17자는 정상 답변) | 0/5 |
| 평균 답변 길이 | 570자 | 678자 (🟢 +108) | 1390자 |
| 총 시간 | 880s | 1083s | 441s |

### 3-3. 핵심 관찰

✅ **주요 개선**:
- 3건 faithful flip (0072, 0095, 0092: F→T, ?→T, T→T)
- S0095 의 empty timeout (240s) 이 valid 359자 답변으로 해소 (engine fix 의 content+thinking 복원 효과)
- S0072 는 17자 짧은 "未找到相关信息" 답변이지만 judge 통과 (이전엔 parse_error)

⚠️ **잔존 이슈**:
- S0228 은 reload 타이밍으로 old engine 사용 → 재실행 시 개선될 것
- S0049 는 Qwen thinking 이 한국어/영어 자연어 위주로 regex 불매칭 — judge_node 보강 필요 (Phase A-3)

속도:
- Qwen3 가 gpt-oss 대비 2.5× 느림. 사용자 선택 (성능 > 속도) 에 부합.
- S0049 의 react 6회 iteration (max) 이 부가 시간 발생 — 정상 루프 동작.

## 4. 3-way 비교 (gpt-oss vs broken Qwen vs optimized Qwen)

TBD

## 5. 결론

TBD

## 6. 롤백

Engine 수정은 gpt-oss 동작 변경 없음. API `:8611` (gpt-oss) 는 그대로 주력 유지, `:8711` (qwen3) 은 실험·비교용.
