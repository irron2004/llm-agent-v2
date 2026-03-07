# Issues

- REQ-4 evidence: plan expects `.sisyphus/evidence/task-02-label.png` from manual UI QA; defer capture to Final Manual QA wave (F3) when running the guided flow in browser.
- Vitest targeted run gotcha: use `src/features/...` path (not `frontend/src/...`) when running from `frontend/`, otherwise Vitest reports "No test files found".
- Playwright install gotcha: stale lock at `/home/hskim/.cache/ms-playwright/__dirlock` can block browser installation; clear lock only after confirming no active installer.
- SSE mock gotcha for browser evidence scripts: ensure final event ends with an extra blank line (`\n\n`) so `connectSse()` flushes the final `data:` block.
- Backend LSP remains noisy in this repo context (missing-import + legacy typing diagnostics), so REQ-2 verification relied on targeted pytest regressions to validate retry-cap and safe-default behavior.
