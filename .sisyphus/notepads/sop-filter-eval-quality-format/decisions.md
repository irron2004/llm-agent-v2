# SOP Canonical Markdown Answer Template - Decision Record

## Date
2026-03-12

## Task Reference
Task 4: Define and enforce a single canonical Markdown answer template for SOP procedures

## Files Modified
- `backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml`
- `backend/llm_infrastructure/llm/prompts/general_ans_v2.yaml`
- `backend/llm_infrastructure/llm/prompts/ts_ans_v2.yaml`

## Canonical Template Structure

The following Markdown template is now enforced in all three SOP answer prompts:

```markdown
## [문제명 또는 제목]

## 준비/안전
- 안전 확보를 위한 준비 사항

## 작업 절차
1. 첫 번째 단계
2. 두 번째 단계
3. 세 번째 단계
(필요한 만큼 순번을 이어가세요)

## 복구/확인
- 복구 후 확인 방법

## 주의사항
- 실행 시 주의할 점

## 참고문헌
[1] doc_id (device_name)
[2] doc_id (device_name)
(REFS에 있는 문서만큼 번호를 매기세요)
```

## Key Design Decisions

1. **Single-line Title**: Required as first line with `## [문제명 또는 제목]` format
2. **Section Headings**: All headings in Korean:
   - `## 준비/안전` - Preparation/Safety
   - `## 작업 절차` - Work Procedure
   - `## 복구/확인` - Recovery/Verification
   - `## 주의사항` - Cautions
   - `## 참고문헌` - References
3. **Numbering**: Only Arabic numerals (`1.`, `2.`, etc.) - NO emoji numerals
4. **Reference Format**: `[N] doc_id (device_name)` - matches existing RAG reference pattern

## Explicitly Forbidden
- Emoji numbering (e.g., `1️⃣`, `2️⃣`)
- Markdown tables (e.g., `|---|` style)
- Mixed-language headings (e.g., English headings in Korean response)

## Verification
- All three YAML files pass `yaml.safe_load()` parsing
- `prompt_loader.load_prompt_template()` successfully loads all prompts
- No Python code changes required

## Dependencies
- Task 5 (validator/retry) will rely on these exact section headings for validation

# Decisions (append-only)

- (TBD) Canonical SOP answer template: strict Markdown sections + `1.` numbering, forbid emoji numbering/tables.
- (TBD) Eval artifacts: thin+rich JSONL vs single canonical JSONL.

#WR|
#WR|- (2026-03-12) Task 4 corrections applied:
#WR|  - Title line changed from `## [문제명 또는 제목]` to `# {제목}` (single-line title, not ##)
#WR|  - Mixed-language marker `**禁止**` replaced with Korean-only `**금지가` (keeping forbid content)
#WR|  - Template now matches canonical SOP format per plan requirements
