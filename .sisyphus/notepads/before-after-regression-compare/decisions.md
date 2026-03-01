# Decisions


## Task 1: Create regression run workspace manifest

### Design Decisions

1. **Stdlib-only approach**: Chose to avoid external dependencies (e.g., pydantic) for simplicity and faster execution. argparse provides sufficient validation capabilities.

2. **SHA256 computation**: Used binary read with 8KB chunks to handle large query files efficiently without loading entire file into memory.

3. **Environment filtering**: Applied whitelist approach (SEARCH_, ES_, RAG_, VLLM_) instead of blacklist to prevent accidental secret leakage. This is more secure by default.

4. **Secret redaction**: Used case-insensitive suffix matching for _KEY, _PASSWORD, _TOKEN to catch variations like API_KEY, api_key, apiKey, etc.

5. **JSON formatting**: Used json.dump with indent=2 and explicit trailing newline for human-readable output and version control friendliness.

6. **CLI design**: Followed existing patterns from `run_paper_b_eval.py` with required arguments and sensible defaults for SHAs (73ca832, c04fa25).

### Trade-offs
- Did not implement TypeScript-style argument parsing (opted for simple argparse)
- Used `dict[str, Any]` return types which trigger basedpyright warnings but work correctly
- No validation of SHA format (assumes valid git SHAs provided)