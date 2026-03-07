# Decisions (append-only)

- 2026-03-04: Keep strict contract: if any row remains unresolved, write `unresolved_docs.jsonl` and exit non-zero; otherwise remove stale unresolved file and emit deterministic corpus outputs.
- 2026-03-04: Preserve normalize-table row order for `doc_meta.jsonl` and `corpus_doc_ids.txt`, and sort snapshot count maps for deterministic diffs.
- 2026-03-04: Scope filter DSL empty-scope behavior is `None` in doc-id mode when no devices/equip/shared-doc-ids are provided; evaluator then uses base filters without additional scope restriction.
