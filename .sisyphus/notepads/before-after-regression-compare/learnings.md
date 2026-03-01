# Learnings


## Task 1: Create regression run workspace manifest

### Key Implementation Details
- Script location: `scripts/evaluation/regression_compare_manifest.py`
- Test location: `tests/evaluation/test_regression_compare_manifest.py`

### Patterns Used
- Stdlib-only: argparse, json, hashlib, os, pathlib, datetime
- JSON output: indent=2, trailing newline for readability
- SHA256 computation: binary read in chunks (8192 bytes)
- Line counting: sum(1 for _ in file)

### Environment Variable Handling
- Whitelist prefixes: SEARCH_, ES_, RAG_, VLLM_
- Redaction: case-insensitive matching for _KEY, _PASSWORD, _TOKEN suffixes
- Redaction value: "***REDACTED***"

### Fail-Fast Validations
- Empty run-id returns exit code 1 with error message
- Missing query files return exit code 1 with error message
- Parent directories created automatically for --out path

### Test Coverage
- Basic manifest creation with all fields
- Empty run-id failure case
- Missing query file failure cases (subset and full)
- Parent directory creation
- SHA256 format validation
- JSON formatting verification
- Environment variable whitelisting
- Secret redaction (exact match)
- Secret redaction (case-insensitive)

## Task 2: ES synthetic isolation preflight

- `docker compose ... up -d elasticsearch` is idempotent and safe to rerun; it kept scope to only the ES service.
- Keep `_cluster/health` polling with `curl -sf` before alias checks, otherwise alias curls can produce misleading empty outputs when ES is still warming up.
- `curl -s "$ES_HOST/_alias/rag_synth_synth_current" || true` preserves the evidence file even if the alias is not yet created.

## Task 3: Synthetic ingest into existing alias target

- If `rag_synth_synth_current` already exists, skip create/switch and ingest through the current alias target to avoid accidental alias mutation.
- Capture alias snapshots before and after ingest to prove the alias target stayed on `rag_synth_synth_v*`.
- `ingest_summary.json` can report the alias name as `index`, so alias evidence files are required to verify concrete index naming.
