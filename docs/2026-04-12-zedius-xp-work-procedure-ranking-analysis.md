# 2026-04-12 ZEDIUS XP procedure-search ranking analysis

## Purpose

This memo records the investigation into why the query

> `ZEDIUS XP 장비의 플라즈마 소스에 장애가 생겼어. 교체 방법을 알려줘`

failed to bring the expected `Work Procedure` content to the top of the SOP/setup search flow.

The goal of this document is to preserve:
- what was checked,
- what was ruled out,
- what was confirmed as the root cause,
- and what implementation directions were discussed.

---

## 1. High-level conclusion

The problem is **not primarily a parsing failure**.

The parsed SOP contains the relevant replacement procedure text for the PRISM SOURCE,
including the source replacement sequence. The failure occurs later, in the **search/ranking and
cross-section expansion path**:

1. The user query contains strong terms like `장애`, `교체`, `방법`.
2. In this SOP, the early searchable sections that directly match `장애` are `사고 사례`-style
   safety/incident sections.
3. `Work Procedure` chunks are not retrieved early enough.
4. The existing procedure-boost logic only boosts `Work Procedure` if those chunks are **already**
   present in the candidate set.
5. The current cross-section expansion rules can bridge from `flow chart` or `part 위치` to
   `Work Procedure`, but **not** from `사고 사례`.
6. As a result, the answer flow can fail before reaching the actual replacement steps.

So the practical issue is: **ranking/expansion failure, not extraction failure**.

---

## 2. Parsed document quality check

### Source reviewed
- `data/vlm_parsed/review/sop/global_sop_supra_xp_all_pm_prism_source/review.md`

### What the review shows
- The document is parsed with VLM output and includes page-level extracted text plus image references.
- The early pages include:
  - `Scope`
  - `Contents`
  - `사고 사례`
  - safety-related pages
- The procedure pages include actual actionable text for PRISM SOURCE replacement.

### Practical interpretation
- There is some LaTeX/table noise in the parsed output.
- However, the procedure text is still present and readable.
- This means the SOP is **usable enough for retrieval and answer generation** if the correct section
  is surfaced.

### Why we do not classify this as a parsing blocker
- The `Work Procedure` content is not empty.
- The replacement steps are not missing.
- The critical failure is that those chunks are not ranked or expanded into view soon enough.

---

## 3. What the procedure content actually contains

The investigation notes identified the relevant section as:

- `global_sop_supra_xp_all_pm_prism_source — 6. Work Procedure`

Within that section, the meaningful procedure chunks include examples like:

- Tool arrange
- PM disable
- PM maint mode 변경
- Pumping
- Leak rate 체크
- Temp off
- PM open
- Upper/Lower baffle 분리
- Matcher power off
- PCW drain
- Gas line 분리
- GDP 분리
- **Source 결합** / source replacement-related steps
- 후속 재조립 및 leak check

The session summary classified these chunks roughly as:
- some LaTeX artifact chunks,
- but the majority still containing real procedure text.

This is consistent with the parsed review artifact: noisy, but not broken.

---

## 4. Existing ranking logic that was checked

### 4.1 Procedure boost exists

Relevant code:
- `backend/llm_infrastructure/llm/langgraph_agent.py`

Confirmed logic:
- `_PROCEDURE_KEYWORDS` includes terms such as:
  - `교체`
  - `절차`
  - `작업`
  - `방법`
  - `replacement`
  - `procedure`
  - `repair`
- `_PROCEDURE_CHAPTERS` includes:
  - `work procedure`
  - `flow chart`
  - `work 절차`

Boost behavior:
- If the query has procedure intent and a retrieved document’s `section_chapter` matches one of the
  procedure chapters, its score is boosted.

### Important limitation
This boost only applies **after** the chunk is already in `all_docs`.

That means:
- if `Work Procedure` chunks are absent from the initial retrieved set,
- then `procedure_boost` has nothing to operate on.

This matches the observed log pattern from the investigation:

> `query_procedure_intent=True` but `procedure_boost=0`

Interpretation:
- the system recognized a procedure-like query,
- but no candidate chunk eligible for procedure boost was present in the ranked set.

---

## 5. Existing cross-section expansion logic that was checked

Relevant code:
- `backend/llm_infrastructure/llm/langgraph_agent.py`

Confirmed trigger map:

```python
_CROSS_SECTION_TRIGGERS = {
    "flow chart": ["Work Procedure"],
    "workflow": ["Work Procedure"],
    "part 위치": ["Work Procedure"],
    "location of parts": ["Work Procedure"],
}
```

### What this means
If an initially hit section is something like:
- `flow chart`
- `workflow`
- `part 위치`

then the retriever can fetch nearby `Work Procedure` chunks from the same `doc_id`.

### What is missing
There is **no trigger** from:
- `사고 사례`

to:
- `Work Procedure`

So if the safety/incident section is what matches the user’s query first, the current cross-section
logic does not bridge from that section into the actual repair procedure.

---

## 6. Confirmed root cause

### Root cause summary
The main failure is the interaction of three facts:

1. The user query contains terms (`장애`, `교체`, `방법`) that can semantically or lexically favor
   safety/incident sections.
2. The actual `Work Procedure` pages often do **not** contain the same Korean trigger words directly
   in the body.
3. The system’s rescue mechanisms only help if:
   - `Work Procedure` is already in the retrieved set, or
   - the trigger section is one of the currently mapped cross-section triggers.

For this SOP, neither condition is reliably true.

### What was ruled out
- “The SOP parser is too broken to use.” → **ruled out**
- “The procedure text is absent.” → **ruled out**
- “There is no Work Procedure section for this task.” → **ruled out**

### What remains true
- The procedure exists.
- The system can miss it because the wrong section gets surfaced first.

---

## 7. Candidate fix directions discussed

Two fix directions were discussed.

### Option A. Add `사고 사례 → Work Procedure` cross-section trigger

Smallest targeted fix.

Idea:
- extend `_CROSS_SECTION_TRIGGERS` so that when `사고 사례` is initially hit,
  nearby `Work Procedure` chunks from the same SOP are also fetched.

Why this helps:
- it addresses the exact observed failure path,
- with minimal policy change,
- and fits the current expansion architecture.

Tradeoff:
- it is still reactive; it depends on the initial safety/incident section hit.
- it may solve this pattern but not fully enforce procedural recall across all SOP/setup cases.

### Option B. Force `Work Procedure` insertion for SOP/setup procedural queries

This was the stronger product idea raised in the conversation.

Initial idea:
- when the UI is already in `절차검색`, meaning doc types are constrained to `sop` / `setup_manual`,
  and the query is recognized as procedural,
  fetch `Work Procedure` chunks explicitly and inject them into the top-ranked set.

Proposed placement in flow (as discussed):
- after procedure-intent detection / procedure boost,
- before final ranking cutoff for stage-1 docs.

Why this helps:
- it does not wait for incidental retrieval of `Work Procedure`.
- it aligns with product intent: if the user explicitly selected procedure search, then procedural
  content should always be represented near the top.

Tradeoff:
- stronger intervention in ranking policy,
- requires careful dedupe and ordering rules,
- the broadest form of this idea (“same-device top-2 forcing”) is too aggressive because it can pull
  the wrong procedure from another SOP belonging to the same tool.

### Refined recommendation: same-`doc_id` Work Procedure top-3 backfill

After further review, the stronger recommendation is **not** to force any same-device Work Procedure
into the top-2. The safer and more precise policy is:

- if the search is SOP/setup-only,
- and the query is procedural,
- and the top-ranked set contains no `Work Procedure` section,
- then fetch `Work Procedure` chunks from the **same `doc_id`** as the top relevant document(s)
  and ensure that at least one such chunk appears within the top-3 evidence set.

Why this is better:
- it keeps the document context intact,
- it treats the current problem as section recovery rather than device-wide override,
- it is much less likely to inject the wrong replacement/calibration procedure,
- and it fits the current same-`doc_id` section-expansion architecture.

Why top-3 instead of top-2:
- top-2 is still a heavy override even if later judge logic exists,
- top-3 preserves stronger visibility for procedural content while keeping at least part of the
  original ranking context,
- and judge/retry logic still has more balanced evidence to work with.

---

## 8. Recommended implementation direction

### Recommended order

1. **Immediate low-risk fix**
   - add `사고 사례 → Work Procedure` as a cross-section trigger.

2. **Policy-level product fix**
   - for `sop_only_predicate=True` and procedural context,
     explicitly backfill same-`doc_id` `Work Procedure` chunks into the top-ranked set and ensure
     top-3 visibility.

### Why this order makes sense
- The trigger fix is the fastest way to remove the specific ZEDIUS XP failure pattern.
- The refined same-`doc_id` top-3 policy better matches the product semantics of “절차검색” while
  keeping the recovery local to the already-matched SOP document.

### Practical recommendation
Do **not** stop at only the trigger fix if the product intent is strong procedural recall for SOP/setup.
The trigger fix is tactical; the same-`doc_id` top-3 Work Procedure backfill is the stronger
medium-term policy.

---

## 9. Suggested future task scope

If this analysis is turned into implementation work, the next task should likely cover:

1. Add missing cross-section trigger(s) for incident/safety sections.
2. Implement SOP/setup procedural same-`doc_id` `Work Procedure` top-3 backfill.
3. Add regression coverage for queries like:
   - `ZEDIUS XP 장비의 플라즈마 소스에 장애가 생겼어. 교체 방법을 알려줘`
4. Verify that:
   - `Work Procedure` appears in the top-ranked evidence,
   - relevance checking now reaches the procedure group,
   - the fix does not overly harm non-procedural SOP queries.

---

## 10. Short version

- The SOP parse is noisy but usable.
- The replacement procedure is present.
- The current failure is a **ranking / expansion failure**, not a parsing failure.
- Procedure boost exists, but it only helps if `Work Procedure` is already retrieved.
- Cross-section expansion exists, but it does not map `사고 사례` to `Work Procedure`.
- Best next move:
  - add the missing trigger,
  - and then implement same-`doc_id` `Work Procedure` backfill so top-3 evidence always includes
    at least one procedural chunk when the query is clearly procedural.

---

## 11. Policy sketch (same-`doc_id` top-3)

The preferred policy can be expressed as follows.

### Preconditions
- `sop_only_predicate == True`
- `is_procedural_context == True`
- current top-ranked docs contain no `Work Procedure` section
- at least one top-ranked doc has a valid `doc_id`

### Action
1. Pick the highest-ranked relevant `doc_id`.
2. Fetch that document’s `Work Procedure` chunks via section keyword lookup.
3. Insert 1-2 of those chunks into the ranked list.
4. Deduplicate by chunk identity.
5. Ensure at least one `Work Procedure` chunk appears within the top-3.

### Why this policy is preferred
- It is narrower than same-device forcing.
- It reuses the same-SOP context already identified as relevant.
- It complements, rather than replaces, the current boost/expansion logic.
