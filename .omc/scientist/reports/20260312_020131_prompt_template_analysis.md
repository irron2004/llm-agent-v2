# Prompt Template & Agent Configuration Analysis
**Date:** 2026-03-12
**Scope:** `backend/llm_infrastructure/llm/prompts/` + `langgraph_agent.py` answer_node

---

## [OBJECTIVE]
Analyze the prompt templates used in SOP/setup answer generation and the agent's answer_node implementation, with focus on:
1. System prompt structure (v1 vs v2 differences)
2. ref_json injection mechanism
3. Answer format control (length, structure, citation)
4. Concrete improvement recommendations

[STAT:n] n=56 prompt YAML files, 24 answer prompts analyzed (setup/ts/general × ko/en/ja/zh × v1/v2)

---

## [DATA]

### Prompt Inventory
- **Total YAML files:** 56
- **Answer prompt files:** 24 (setup: 8, ts: 8, general: 8)
- **Routes:** setup, ts, general
- **Languages:** ko (Korean, default), en, ja, zh
- **Versions:** v1, v2

### File Naming Convention
```
{route}_ans[_{lang}]_{version}.yaml
```
Korean (ko) has no language suffix — it uses the base template name.

---

## [FINDING 1] Only setup/en, setup/ja, setup/zh were upgraded in v2; all others are identical to v1

[STAT:n] 9 of 12 prompt pairs are byte-for-byte identical between v1 and v2
[STAT:effect_size] setup/en: +681 chars (+129%), setup/ja: +502 chars (+220%), setup/zh: +443 chars (+295%)

The v2 upgrade applied a structured, multi-section format exclusively to non-Korean setup prompts.
The following remain **unchanged** from v1 to v2:
- `setup_ans_v1 == setup_ans_v2` (Korean setup, 297 chars)
- All 4 `ts_ans_*` prompts (ko/en/ja/zh)
- All 4 `general_ans_*` prompts (ko/en/ja/zh)

This means the Korean-language setup prompt, all troubleshooting prompts, and all general prompts
received **no structural improvement** when the system was upgraded to v2.

---

## [FINDING 2] v2 setup prompts (en/ja/zh) introduce 5 structural features absent from all other prompts

[STAT:n] 3 prompts upgraded; 0 of 12 other prompts have these features
[STAT:effect_size] Feature count: v1 setup/en = 6 rules, v2 setup/en = 18 rules (+200%)

Features present **only** in `setup_ans_{en,ja,zh}_v2`:
| Feature | v2 (en/ja/zh) | All others |
|---|---|---|
| `### Section headers` | Yes | No |
| Answer priority guidance | Yes | No |
| Pre-check section | Yes | No |
| Post-check section | Yes | No |
| Caution/Warning section | Yes | No |
| Numbered procedure (1. 2. 3.) | Yes | No |

Common features present in **all** answer prompts (v1 and v2):
- Grounding rule: use only REFS as evidence
- Empty REFS fallback message
- Language lock (IMPORTANT: answer in X only)
- Reference list at end
- Citation format [N]

---

## [FINDING 3] Korean setup prompt (setup_ans_v2) is structurally weaker than its English counterpart

[STAT:n] setup_ans_v2 = 297 chars; setup_ans_en_v2 = 1210 chars
[STAT:effect_size] 4.1x size ratio; setup_ans_v2 missing all 6 structural features present in setup_ans_en_v2

`setup_ans_v2` (Korean) specifies only:
```
- 단계별 절차(번호 목록)로 답변
- REFS 번호에 맞게 [1] 형식으로 출처 인용
- 마지막에 참고문헌 목록 추가
```

`setup_ans_en_v2` (English) additionally specifies:
```
### Purpose / Pre-check / Procedure / Post-check / Cautions / References
```
With explicit section ordering, priority weighting (Work Procedure > Workflow),
and numbered step format. The Korean prompt lacks all of these.

[LIMITATION] The Korean SOP/setup answer quality is bottlenecked by prompt structure disparity.
Users querying in Korean receive answers shaped by a far simpler prompt than English users.

---

## [FINDING 4] ts_ans and general_ans have no structured format in any language (v1 or v2)

[STAT:n] 8 ts_ans prompts: 0 with section_headers, 0 with priority, 0 with pre/post-check
[STAT:n] 8 general_ans prompts: 0 with section_headers

`ts_ans` only specifies: "원인 / 진단 / 조치 섹션으로 답변" (cause / diagnosis / action)
but does NOT define what goes inside each section, does not specify numbered steps,
and has no guidance on prioritizing specific REFS content types (e.g., TSG procedures).

---

## [FINDING 5] ref_json injection mechanism and its constraints

[STAT:n] MAX_REF_CHARS_ANSWER = 1200 chars per document

The `ref_json_to_text()` function delivers evidence as plain text (not JSON):
```
[{rank}] {doc_id} ({device_name}/{equip_id}): {content}
```
- Content is truncated to **1200 characters per document**
- Plain text format chosen deliberately to avoid vLLM tokenization artifacts
  (JSON braces/brackets caused abnormal `!` token repetitions in some models)
- If no docs: string `"EMPTY"` is passed, triggering the fallback rule
- `answer_ref_json` takes priority over `ref_json` (post-rerank slice is used for answer)

The `{ref_text}` placeholder is injected into the **user** message (not system).
The system prompt only defines rules; the evidence arrives at inference time.

---

## [FINDING 6] answer_node language selection uses a 3-tier priority chain

Priority: `target_language` > `detected_language` > `"ko"` (default)

Language detection is rule-based (`_detect_language_rule_based`), not LLM-based.
When `answer_language == "en"`, the node substitutes `query_en` (translated English query)
to prevent language drift in the prompt. For ja/zh, the original query is preserved.

[STAT:n] Temperature for answer generation: 0.5 (non-deterministic)
[STAT:n] max_tokens for answer: 4096

[LIMITATION] Temperature=0.5 introduces answer variability for identical queries.
This can make evaluation/A-B testing results noisy.

---

## [FINDING 7] Post-processing safety net: repetition truncation

The `_truncate_repetition()` function acts as a fallback for model repetition loops:
- Detects blocks >= 40 chars repeating > 2 times
- Truncates to max 2 repetitions
- Triggered when repeat_penalty in Ollama/vLLM is insufficient

[LIMITATION] This is a downstream fix; root cause is model-level repeat_penalty configuration,
not the prompt itself.

---

## Improvement Recommendations

### Priority 1 (Critical): Upgrade Korean setup prompt to v2 structured format
`setup_ans_v2` must be updated to match `setup_ans_en_v2` structure.
Current state: 297-char flat ruleset. Target: ~1000-char structured with sections.

### Priority 2 (High): Upgrade ts_ans prompts (all languages) with structured TS format
Current ts_ans only lists "원인/진단/조치" as section names with no content guidance.

### Priority 3 (Medium): Standardize ts_ans_en_v2 to match setup_ans_en_v2 structure
`ts_ans_en_v2` is identical to `ts_ans_en_v1` — no structured upgrade.

### Priority 4 (Medium): Add `answer_priority` guidance to all remaining prompts

### Priority 5 (Low): Consider reducing TEMP_ANSWER from 0.5 to 0.1-0.2 for procedural routes

### Priority 6 (Low): Add explicit ref truncation notice to prompts
