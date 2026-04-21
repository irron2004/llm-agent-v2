# Model A/B: gpt-oss:120b vs qwen3:30b-a3b

작성일: 2026-04-21
목적: 내 ReactRAGAgent (Phase A 프롬프트) 에서 LLM 모델만 바꿨을 때 품질·속도 비교.
제약: 사용자가 요청한 `Qwen/Qwen3.6-35B-A3B` 는 공개 GGUF 포팅이 없어 가장 근접한 `qwen3:30b-a3b` (Qwen3 MoE 30B total / 3B active) 로 대체.

## 0. 환경

| 항목 | A: gpt-oss:120b | B: qwen3:30b-a3b |
|---|---|---|
| 모델 | gpt-oss:120b (dense 116.8B, MXFP4) | qwen3:30b-a3b (MoE 30.5B total / 3B active, Q4_K_M) |
| 디스크 크기 | 60.9GB | 17.3GB |
| GPU VRAM (inference) | 약 60GB | **55GB** (qwen3 가 더 큼 in VRAM due to context) |
| `OLLAMA_NUM_CTX` | 131072 | 131072 ⚠️ 과도 (Qwen3 native 32K) |
| `OLLAMA_TEMPERATURE` | 0.7 (default) | 0.2 (override) |
| `OLLAMA_REPEAT_PENALTY` | 1.3 (default) | 1.05 (override) |
| API 포트 | :8611 | :8711 (신규, 롤백 보장) |
| Phase A 프롬프트 | 동일 (`general_ans_v2`, `ts_ans_v2`) | 동일 |

## 1. 결과 요약 (5 쿼리)

| QID | query 요약 | gpt-oss:120b | qwen3:30b-a3b | Winner |
|---|---|---|---|---|
| 0228 | SK Hynix FFU 차압 | faithful=F, 245자, 126s | faithful=F, **407자 (더 구체)**, 130s, **parse_error** | ≈ (Qwen3 값은 더 정확 '0.2 이상', 판정 실패) |
| 0072 | APC C6000207 3D 도면 | faithful=**T**, 224자, 79s | faithful=F, **15자** (극단 짧음), 202s, **parse_error** | 🟢 gpt-oss |
| 0095 | Vplus pumping time | faithful=**T**, 759자, **표 O**, 72s | faithful=?, **0자 (empty answer)**, 240s timeout | 🟢 gpt-oss |
| 0092 | EPAP715 분석 | faithful=**T**, 1724자, **표×3 + kw-block**, 83s | faithful=**T**, 929자, kw-block, 162s | 🟡 gpt-oss (분량·구조) |
| 0049 | source power fault | faithful=**T**, 4000자, **표 O**, 81s | faithful=F, 1498자, 145s, **parse_error** | 🟢 gpt-oss |

| 지표 | gpt-oss | qwen3 |
|---|---|---|
| faithful | **4/5** | **1/5** |
| 평균 답변 길이 | 1390자 | 570자 (41%) |
| 표 사용 | 3/5 | **0/5** |
| 총 시간 | 441s | **880s (2배 느림)** |
| parse_error 건 | 0 | **3/5** |

## 2. 근본 원인 분석

### 2-1. `parse_error` 3건 — **judge LLM JSON parsing 실패** (품질 문제 아님)

Judge 노드는 LLM 에게 `{"faithful": bool, "issues": [], "hint": "..."}` 구조를 요구. Qwen3 출력이 다음 중 하나의 이유로 깨졌을 가능성:

- **markdown code fence wrapping**: Qwen3 가 `` ```json ... ``` `` 로 감싸면 pydantic 파싱 실패.
- **`<think>` reasoning tokens 노출**: 앞에 붙으면 JSON 인식 실패.
- **한/중/영 혼용 쿼리** 에 대한 output schema 불안정.

답변 본문 자체는 **모두 합리적**:
- S0228: "0.2 이상 [4]" (gpt-oss '0.13' 보다 정확) — 단, judge 가 parse 실패로 faithful 판정 못 함.
- S0072: "在RAG数据中未找到相关信息" 중문 15자 — 매우 짧지만 정확한 "정보 없음" 답변.
- S0049: 장비별 Cause/Action 영문 bullet, 1498자 — 충분한 품질.

→ **실제 답변 품질 격차는 gpt-oss 보다 못하지만 겉보기만큼 심하지 않음**. Judge 안정화가 우선.

### 2-2. S0095 빈 답변 (empty, 240s timeout)

Qwen3 planner 가 `NextAction` pydantic 응답을 반복 실패 → 최종 answer 까지 가지 못함. `react_agent.py:862` 의 `response_model=NextAction` 호출 시 Qwen3 의 JSON 출력이 한 번도 valid 하지 않았을 가능성. 2건의 LLM 호출 재시도 (`_try_raw_plan_parse`) 도 실패 → empty answer 반환.

### 2-3. 속도 2배 느림

**원인 순위**:
1. **`OLLAMA_NUM_CTX=131072`** — Qwen3 native 32K. 128K 로 확장하면 KV cache + attention 연산 폭증. refs 47K chars (~12K tokens) 입력에도 128K 슬롯 스캔.
2. **GPU split** (2x A6000 각 ~29GB) — tensor parallel split 오버헤드. 단일 GPU 에 올릴 수 있는 크기였다면 (num_ctx 하향 + Q4) 더 빨랐을 것.
3. **Ollama MoE routing** — gpt-oss MXFP4 대비 Qwen3 MoE 3B active dispatch 의 커널 최적화 수준 차이 가능성.

### 2-4. 구조 (표) 사용 0건

Phase A 프롬프트가 표 허용을 명시했지만 Qwen3 가 (temperature 0.2 + repeat_penalty 1.05 조합 하에서) **더 보수적으로 서술형 선택**. gpt-oss 는 동일 프롬프트에서 표를 적극 채택. 모델별 출력 stylistic 차이.

## 3. 결론

### 🔴 현재 설정에서는 Qwen3:30b-a3b 를 **도입 비권장**

품질 4/5 → 1/5, 속도 2배 느림, 표 사용 3/5 → 0/5. **지금 교체하면 품질 회귀 확정**.

### 🟡 단, 다음 튜닝 후 재평가 권장

| 조정 | 기대 효과 | 공수 |
|---|---|---|
| `OLLAMA_NUM_CTX=32768` (128K → 32K) | 속도 2-3배 ↑ | .env 1줄 |
| Judge / planner 의 JSON 파싱 방어 | parse_error 3/5 → 0 기대 | engine code 소량 수정 (markdown fence strip + `<think>` strip) |
| `Qwen3.6-35B-A3B` GGUF 공개 시 재시도 | 최신 세대 품질 향상 | pull 만 |
| reasoning off (`num_thread`, `think: false`) | 불필요한 reasoning 제거 | ollama 옵션 |

### 🟢 즉시 액션: gpt-oss:120b 로 롤백 유지

- API `:8611` (gpt-oss) 는 건재. `.env` 는 원래 값 (`OLLAMA_MODEL_NAME=gpt-oss:120b`) 그대로 유지.
- Qwen3 API `:8711` 은 **언제든 재실험용** 으로 보존 가능, 또는 정지:
  ```bash
  # 정지
  kill -TERM $(cat /tmp/llm-agent-v2-api-8711.pid)
  ```
- gpt-oss 모델은 삭제하지 않음 (롤백 보장).

## 4. Phase B/C 로 이관할 실험

1. Qwen3 judge 출력의 `<think>`/code-fence 스트리퍼 추가 → re-run sample5.
2. `NUM_CTX=32768` 로 Qwen3 재측정 (속도 중심).
3. `Qwen/Qwen3.6-35B-A3B` 공식 GGUF 릴리즈 주기적 확인 (1-2주).
4. 정량 평가 확장: 228 rows 전체 돌리기 전에 judge 안정성 먼저 해결.

## 5. 산출물

- `data/eval_results/mybot_march_react_phase_a_qwen3_20260421_sample5/run_min.jsonl` (Qwen3 5건 응답)
- 참조 비교군: `data/eval_results/mybot_march_react_phase_a_20260421_sample5/run_min.jsonl` (gpt-oss 5건)

## 6. 롤백 확인

Phase A prompt/validator 코드는 모델 독립적이므로 gpt-oss 재사용 시 품질 유지. `:8611` API 에 간단 health check 로 확인 가능.
