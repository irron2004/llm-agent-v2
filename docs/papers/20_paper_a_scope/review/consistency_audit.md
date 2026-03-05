# Paper A: Consistency Audit

> Date: 2026-03-05
> Purpose: Reconcile docs vs existing data/code artifacts. Identify drift and corrections needed.

---

## 1. Artifact Inventory

### 1.1 Data Artifacts (`data/paper_a/`)

| Path | Status | Notes |
|------|--------|-------|
| `data/paper_a/eval/query_gold_master.jsonl` | EXISTS | v0.4 original, 472 rows |
| `data/paper_a/eval/query_gold_master_v0_4_frozen.jsonl` | EXISTS | Byte-identical backup of v0.4 |
| `data/paper_a/eval/query_gold_master_v0_5.jsonl` | EXISTS | Official eval set, 472 rows, balanced splits |
| `data/paper_a/eval/query_gold_master_v0_5_split_report.json` | EXISTS | Split/observability counts |
| `data/paper_a/corpus_labels/document_scope_table.csv` | EXISTS | Document scope labels |
| `data/paper_a/corpus_labels/shared_doc_gold.csv` | EXISTS | Shared document gold labels |
| `data/paper_a/corpus_labels/device_family_gold.csv` | EXISTS | Device family gold labels |
| `data/paper_a/metadata/device_catalog.csv` | EXISTS | Device catalog |
| `data/paper_a/metadata/doc_type_map.csv` | EXISTS | Doc type to scope_level mapping |
| `data/paper_a/metadata/equip_catalog.csv` | EXISTS | Equipment catalog |

### 1.2 Script Artifacts (`scripts/paper_a/`)

| Script | Status | Notes |
|--------|--------|-------|
| `scripts/paper_a/build_corpus_meta.py` | EXISTS | Corpus metadata builder |
| `scripts/paper_a/build_eval_sets.py` | EXISTS | Eval set builder |
| `scripts/paper_a/build_family_map.py` | EXISTS | Family graph builder (Jaccard) |
| `scripts/paper_a/build_shared_and_scope.py` | EXISTS | Shared/doc_scope policy builder |
| `scripts/paper_a/canonicalize.py` | EXISTS | Key canonicalization (compact_key) |
| `scripts/paper_a/evaluate_paper_a.py` | EXISTS | Legacy evaluator (B0-B4, P1-P4) |
| `scripts/paper_a/preflight_es.py` | EXISTS | ES alias/index preflight check |
| `scripts/paper_a/rebuild_query_gold_master_splits.py` | EXISTS | v0.4->v0.5 split regenerator |
| `scripts/paper_a/validate_master_eval_jsonl.py` | EXISTS | Master eval JSONL validator |
| `scripts/paper_a/validate_eval_jsonl.py` | EXISTS | Legacy eval validator |
| `scripts/paper_a/retrieval_runner.py` | EXISTS | Retrieval execution helper |
| `scripts/paper_a/_io.py` | EXISTS | JSONL/JSON I/O utilities |
| `scripts/paper_a/_paths.py` | EXISTS | Path constants |

### 1.3 Backend Code

| Path | Status | Notes |
|------|--------|-------|
| `backend/llm_infrastructure/retrieval/filters/scope_filter.py` | EXISTS | OR-branch scope filter DSL |

### 1.4 Policy Artifacts (`.sisyphus/evidence/paper-a/`)

| Path | Status | Notes |
|------|--------|-------|
| `.sisyphus/evidence/paper-a/corpus/doc_meta.jsonl` | EXISTS | 578 docs |
| `.sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt` | EXISTS | 578 doc IDs |
| `.sisyphus/evidence/paper-a/corpus/corpus_snapshot.json` | EXISTS | Corpus summary |
| `.sisyphus/evidence/paper-a/policy/doc_scope.jsonl` | EXISTS | Missing `es_equip_id` (Task 3 fix) |
| `.sisyphus/evidence/paper-a/policy/shared_topics.json` | EXISTS | D_shared topics |
| `.sisyphus/evidence/paper-a/policy/shared_doc_ids.txt` | EXISTS | Shared doc IDs |
| `.sisyphus/evidence/paper-a/policy/family_map.json` | EXISTS | Device family graph |
| `.sisyphus/evidence/paper-a/policy/policy_snapshot.json` | EXISTS | Policy summary |

---

## 2. Quantitative Drift Checks

### 2.1 Eval Set Size: Spec vs Reality

| Source | Stated Size | Actual Size | Delta |
|--------|-------------|-------------|-------|
| `paper_a_scope_spec.md` (implicit "380건") | ~380 | 472 | +92 |
| `query_gold_master.jsonl` | — | 472 rows | — |
| `query_gold_master_v0_5.jsonl` | — | 472 rows | — |

**Action Required**: Update spec to reflect actual 472-row eval set. The v0.5 split report shows:
- explicit_device: 146 (dev=124, test=22)
- explicit_equip: 96 (dev=88, test=8)
- implicit: 150 (dev=129, test=21)
- ambiguous: 80 (dev=80, test=0 — all forced to dev due to empty gold)

### 2.2 Corpus Size

| Source | Docs |
|--------|------|
| README.md | 578 |
| corpus_doc_ids.txt | 578 |
| Consistent | YES |

### 2.3 Split Report Anomaly

- `ambiguous` slice has 0 test rows (all 80 forced to dev due to empty `gold_doc_ids`)
- This means ambiguous queries cannot be evaluated with Hit@k / MRR on test split
- **Implication**: Ambiguous slice should be reported on dev only, or gold labels must be added

---

## 3. Matryoshka Feasibility Check

### Current Embedding Stack

| Component | Path | Implementation |
|-----------|------|---------------|
| Base embedder | `backend/llm_infrastructure/embedding/engines/sentence/embedder.py` | SentenceTransformer |
| Adapter | `backend/llm_infrastructure/embedding/adapters/sentence.py` | Fixed-dimension via `get_dimension()` |
| truncate_dim | `embedder.py:26` | Parameter exists but is SentenceTransformer's native truncation |

### Assessment

- The embedder supports `truncate_dim` parameter which SentenceTransformer passes through to the model
- This is NOT the same as Matryoshka Representation Learning (MRL) training
- MRL requires models **trained** with nested dimension loss (Kusupati et al., NeurIPS 2022)
- Simply truncating a non-MRL model's embeddings will degrade quality unpredictably
- **Current models in use** (Qwen3-4B, bge-m3, jina-v5 per recent commits) — need verification whether any support MRL

### Verdict

- Matryoshka results MUST be marked **`planned-not-reported`** unless:
  1. A model with verified MRL training is adopted AND
  2. Fair baselines (full-dim same model) are compared AND
  3. Quality degradation at target dims (64/128/256) is characterized

---

## 4. Documentation Inconsistencies

### 4.1 `evidence_mapping.md` — Incorrect Status Markers

| Artifact | Current Status | Actual Status | Correction |
|----------|---------------|---------------|------------|
| D_shared 판정 결과 | **미생성** | EXISTS (`policy/shared_topics.json`, `shared_doc_ids.txt`) | Change to "존재" |
| Family graph | **미생성** | EXISTS (`policy/family_map.json`) | Change to "존재" |
| Mask set | **미생성** | EXISTS (via `scope_observability=implicit` in v0.5) | Change to "존재 (implicit subset)" |
| Ambiguous challenge set | **미생성** | EXISTS (via `scope_observability=ambiguous` in v0.5) | Change to "존재 (ambiguous subset)" |
| expected_device 라벨 | **미추가** | EXISTS (via `allowed_devices` in v0.5) | Change to "존재 (allowed_devices)" |

### 4.2 `paper_a_scope_spec.md` — Outdated References

| Issue | Location | Correction |
|-------|----------|------------|
| "380건" eval set size | §6 area | Update to 472건 with per-slice breakdown |
| Missing v0.5 reference | Throughout | Add `query_gold_master_v0_5.jsonl` as official |
| Codebase mapping table | §3 | Many items now exist (scope_filter.py, family_map, shared policy) |

### 4.3 `README.md` — Minor Updates Needed

| Issue | Correction |
|-------|------------|
| `paper_a_scope_spec.md` version reference | Update to v0.5 |
| File structure section | Add `review/` directory |
| Open Questions §1 (MRL support) | Answer: truncate_dim exists but not MRL-trained |

---

## 5. Correction Checklist

- [x] `evidence_mapping.md`: Update "미생성" → "존재" for D_shared, Family graph, Mask set, Ambiguous set, expected_device
- [ ] `paper_a_scope_spec.md`: Update "380건" → "472건" with per-slice counts
- [ ] `paper_a_scope_spec.md`: Add v0.5 eval set reference
- [ ] `paper_a_scope_spec.md`: Update codebase mapping table to reflect implemented items
- [ ] `README.md`: Add review/ directory to file structure
- [ ] `README.md`: Answer MRL open question
