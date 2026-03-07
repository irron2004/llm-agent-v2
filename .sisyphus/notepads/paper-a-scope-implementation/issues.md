# Issues (append-only)

- 2026-03-04: No implementation blockers; both preflight QA scenarios executed successfully.
- 2026-03-04: Initial corpus join missed 167+ rows because many entries were not exact `doc_id` matches; resolved by adding staged fallback (suffix stripping, prefix candidates, metadata-biased lookup) while still keeping failure-loud behavior.
