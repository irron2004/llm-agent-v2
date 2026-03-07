# unified-todo issues

- 2026-03-05 UI-1: `lsp_diagnostics` on `backend/api/routers/agent.py` reports pre-existing workspace import-resolution errors (`reportMissingImports` for `backend.*`) in this environment; frontend types file is clean.
- 2026-03-05 UI-1 cleanup: Reverted unintended scope-creep edits in retrieval logic/tests and a different plan notepad to keep UI-1 limited to backend/frontend contract wiring plus boulder session metadata.
- 2026-03-05 UI-2: `lsp_diagnostics` still reports pre-existing `reportMissingImports` (`backend.*`) in this environment, so changed-file verification relies on targeted pytest plus ensuring no new runtime/type errors were introduced in routing logic.
- 2026-03-05 UI-3: `lsp_diagnostics` on the modified backend files still shows pre-existing workspace import-resolution errors for `backend.*`; compile-based verification is required in this environment to validate syntax/integration changes.
- 2026-03-05 UI-4: Changed-file `lsp_diagnostics` remains non-clean due pre-existing `reportMissingImports` for `backend.*` in this workspace; used required `python -m compileall` on modified files to verify syntax after metadata and task-scope updates.
- 2026-03-05 UI-5: Changed-file `lsp_diagnostics` is still non-clean due pre-existing workspace `reportMissingImports`/type-noise in `backend.*`; compile and targeted runtime tests are required for meaningful verification in this environment.
- 2026-03-05 UI-5: Running `pytest -q backend/tests/test_answer_language_templates.py` from repo root failed collection (`ModuleNotFoundError: backend`); test passes with `PYTHONPATH=.` set, indicating environment path setup dependency rather than logic regression.
- 2026-03-05 UI-9: Real guided-confirm graph execution currently crashes inside `LangGraphRAGAgent._wrap_node` when nodes return `Command`; test fixture patched wrapper-only instrumentation to keep real node/flow assertions runnable.
- 2026-03-05 UI-11: Real `/api/agent/run` guided resume reproduced the wrapper crash (`_wrap_node` calling `.items()` on `Command`); minimal fix is wrapper-only logging/details guards for `Command`, and local API process restart is required when uvicorn runs without `--reload`.

- 2026-03-06 RF-13: Concurrent evaluate_sop_agent_page_hit.py runs into the same http_eval out_dir can corrupt agent_eval.jsonl (invalid UTF-8 / UnicodeDecodeError in validator); enforce single-process runs, validate immediately, and rotate old http_eval to a timestamped backup before rerun.
