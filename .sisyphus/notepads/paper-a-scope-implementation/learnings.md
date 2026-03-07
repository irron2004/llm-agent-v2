# Learnings (append-only)

- 2026-03-04: For ES preflight scripts, reuse `SearchSettings` (`SEARCH_*` env vars) and `EsIndexManager` so alias naming stays consistent with backend (`rag_chunks_{env}_current`).
- 2026-03-04: `EsIndexManager.get_alias_target()` is the safest read-only path for default alias resolution; for custom `--index`, fallback to `indices.get_alias` then `indices.exists` to support alias-or-concrete index inputs.
- 2026-03-04: Keep failure evidence deterministic by always writing connectivity/alias errors to `.sisyphus/evidence/task-01-preflight-es-error.txt`.

- Task 02: added stdlib-only helpers in `scripts/paper_a/_io.py` (UTF-8, mkdir parents, trailing newline).
- Task 02: added `scripts/paper_a/_paths.py:project_root()` using `Path(__file__).resolve().parents[2]`.

- 2026-03-04: Canonical doc_id variants differ: VLM keeps Hangul (`[^a-zA-Z0-9가-힣]` -> `_`), collapses `_+`, strips edge `_`, lowercases, and falls back to `doc` when empty.
- 2026-03-04: Batch SOP pipeline (`tr -cs 'a-z0-9_-' '_'`) lowercases + spaces->`_` + replaces non `[a-z0-9_-]` with `_` and squeezes repeated `_`, but does not strip leading/trailing underscores.
- 2026-03-04: Device matching uses `_compact_text` semantics (drop whitespace + `- _ . /`) so `SUPRA XP`, `supra_xp`, and `SUPRA.XP` all match.

- 2026-03-04: Task 4 `doc_meta.jsonl` schema (one row per doc): `source_file, topic, manifest_doc_type, es_doc_id, es_doc_type, es_device_name, es_equip_id`; `es_doc_type` is canonical ES key via `normalize_doc_type_es` (e.g., `sop`, `ts`).

- 2026-03-04: Task 5 shared topics are computed from SOP-only docs (`manifest_doc_type in {sop_pdf,sop_pptx}`) to match corpus stats `shared_topic_count==13`; doc-level `is_shared` still applies to TS docs when their topic is shared.
- 2026-03-04: `rag_chunks_dev_current` stores searchable `doc_id` but not `doc_id.keyword`; exact matching should try both and prefix fallback should aggregate on `doc_id`.
- 2026-03-04: Some normalize-table stems differ from indexed IDs (language/version suffix drift, truncation), so deterministic fallback using `doc_id` prefix + metadata-biased retrieval is required to reach full 578-row coverage.
- 2026-03-04: Family(device) graph should stay deterministic by sorting devices inside each connected component, sorting components by first device, then assigning sequential ids ; weighted Jaccard uses  with deg fallback from topic device coverage and a hard guard .
- 2026-03-04: Family(device) graph should stay deterministic by sorting devices inside each connected component, sorting components by first device, then assigning sequential ids F00,F01,...; weighted Jaccard uses w(topic)=1/log(1+deg) with deg fallback from topic device coverage and a hard guard deg>=1.\n- 2026-03-04: In family-map generation, keep deterministic ordering (members sorted, components sorted by first member, then assign F00+ in order) and guard topic degree with max(1, deg_from_shared_or_coverage).
- 2026-03-04: SOP CSV has quoted multi-line 질문내용 cells; use csv.DictReader(utf-8-sig,newline="") and fail-fast on missing required Korean headers instead of dropping rows.
- 2026-03-04: For deterministic masking, replace full device aliases only (raw/lower/upper/compact variants) and handle `{device} 설비` before plain alias replacement so component terms (e.g., controller/ffu/robot) are not stripped.
- 2026-03-04: Ambiguous eval rows are safest when filtered from already-masked explicit derivatives using topic degree (`deg(topic)` from doc_meta topic->unique compact-device set), with topic fallback via `gold_doc_ids` mapping.
- 2026-03-04: Task 09 retrieval runner should enforce corpus whitelist through `EsSearchEngine.build_filter(doc_ids=...)`, which already issues robust `doc_id` OR `doc_id.keyword` term clauses for mapping differences.
- 2026-03-04: In this ES environment, RRF `sub_searches` path can reject `knn.k` (`[knn] unknown field [k]`), so hybrid retrieval should tolerate fallback to script_score while preserving the same corpus filter.
