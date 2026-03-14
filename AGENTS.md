# AGENTS.md

Practical guidance for autonomous coding agents in `llm-agent-v2`.
All commands and conventions below are derived from current repo files.

## 0) Mandatory Startup Protocol
For every non-trivial task, treat the following startup sequence as required before code edits:
1. Read `docs/2026-03-14-agent-개발-운영.md` section `0) 신규 agent 5분 온보딩 (필독)` and section `5.5 비사소 작업 판정 기준`.
2. Run the kickoff checklist from `docs/agent-skills/task-start-kickoff.md`, or use the runtime wrapper if local skills are available:
   - OMC: `.omc/skills/task-start-kickoff/SKILL.md`
   - OpenCode: `.opencode/skills/task-start-kickoff/SKILL.md`
   - Claude: `.claude/skills/task-start-kickoff/SKILL.md`
   - Codex: `.codex/skills/task-start-kickoff/SKILL.md`
3. Read `docs/contracts/product-contract.md` and identify relevant contract IDs.
4. If the task is non-trivial, create or update a task document from `docs/tasks/TASK_TEMPLATE.md` before editing files.

This startup protocol is mandatory even when local skill auto-loading is unavailable.

## 1) Rule Sources and Priority
Follow instructions in this order:
1. Direct user task instructions.
2. This `AGENTS.md`.
3. Repository docs (`CLAUDE.md`, `README.md`, module docs).
4. Existing code patterns in files you modify.

Cursor/Copilot rule status in this repo:
- No `.cursor/rules/` directory found.
- No `.cursorrules` file found.
- No `.github/copilot-instructions.md` file found.

If any of those files are added later, treat them as authoritative and update this file.

## 2) Repository Shape
- Backend: `backend/` (FastAPI + RAG infrastructure).
- Frontend: `frontend/` (React 18 + TypeScript + Vite + Ant Design).
- Orchestration: root `Makefile`, `docker-compose.yml`, `.env*`.
- Core backend pattern: Engine -> Adapter -> Registry under `backend/llm_infrastructure/*`.
- Backend flow: `api/routers -> services -> llm_infrastructure`.

## 3) Build / Run / Lint / Test Commands
Run from repo root unless `cd ...` is shown.

### 3.1 Backend (Python/FastAPI)
Dev server:
```bash
make run-api
make stop-api
```

Run backend test suite:
```bash
cd backend && uv run pytest tests/ -v
```

Run one backend test file:
```bash
cd backend && uv run pytest tests/test_retrieval_pipeline.py -v
```

Run one backend test by name pattern:
```bash
cd backend && uv run pytest tests/test_retrieval_pipeline.py -v -k "deterministic"
```

Run root-level API contract/regression tests:
```bash
uv run pytest tests/api -v
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
```

Lint/format/type-check:
```bash
cd backend && uv run ruff check .
cd backend && uv run ruff format .
cd backend && uv run mypy .
```

### 3.2 Frontend (React/TypeScript)
Dev/build/test:
```bash
cd frontend && npm run dev
cd frontend && npm run build
cd frontend && npm run test
cd frontend && npm run test:watch
```

Run one frontend test file:
```bash
cd frontend && npm run test -- src/features/search/__tests__/search-page.test.tsx
```

Run one frontend test by test name:
```bash
cd frontend && npm run test -- -t "renders search input and controls"
```

Note: there is currently no dedicated frontend `lint` script in `frontend/package.json`.

### 3.3 Docker / Compose
```bash
make up
make prod-up
make dev-up
make dev-rebuild
make up-vllm
make logs
make logs-api
make logs-es
```

## 4) Code Style Guidelines

### 4.1 Python Backend Style
Tooling constraints (`backend/pyproject.toml`):
- Ruff line length `100`, target `py312`.
- Ruff lint rules: `E,F,I,N,W,UP`.
- MyPy strict mode enabled (`strict = true`).

Conventions:
- Use type annotations consistently; prefer explicit return types.
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants.
- Imports typically grouped as stdlib -> third-party -> first-party with blank lines.
- Keep comments minimal; add only when logic is non-obvious.
- In routers, surface user-facing failures via `HTTPException`.
- In services/infrastructure, log contextual errors and raise explicit exceptions for invalid config/state.
- Avoid silent exception swallowing unless intentionally falling back.

Observed error-handling pattern:
- `except Exception as exc:` with structured logging is common.

### 4.2 TypeScript/React Frontend Style
Tooling constraints (`frontend/tsconfig.json`, `frontend/vitest.config.ts`):
- TypeScript strict mode enabled.
- Path alias `@/* -> ./src/*`.
- Tests run on Vitest + jsdom + Testing Library.

Conventions:
- Prefer explicit types for API payloads, hook state, and async returns.
- Use `import type { ... }` for type-only imports.
- Naming: `PascalCase` components, `camelCase` functions/vars, `useXxx` hooks.
- Function components are standard; default exports are common for pages/layouts.
- Reuse shared API helpers (`apiClient`, env utilities) over ad-hoc request logic.
- Keep UI failure paths visible (message/alert/error state), not silent.

Observed error-handling pattern:
- Throw `Error` for invariant violations.

### 4.3 Testing Conventions
- Backend: `pytest`, `test_*.py`, fixtures in `conftest.py`.
- Frontend: `*.test.ts` / `*.test.tsx` under feature folders and `__tests__`.
- Prefer behavior assertions over fragile snapshots.
- Debug quickly with single-file + `-k`/`-t`, then run broader suites.

## 5) Architecture and Change Discipline
- Preserve Engine/Adapter/Registry pattern for new backend retrieval/embedding/preprocessing pieces.
- Register adapters with the appropriate `@register_xxx(...)` decorator and ensure import wiring exists.
- Keep routers thin (validation/orchestration) and place business logic in services.
- Align with existing LangGraph/RAG flow in `backend/services/agents/langgraph_rag_agent.py` unless task explicitly requires redesign.

## 6) Change Safety Protocol
- For full multi-agent operating flow, read `docs/2026-03-14-agent-개발-운영.md` before substantial parallel work.
- For non-trivial task judgment, use the single source of truth: `docs/2026-03-14-agent-개발-운영.md` section `5.5 비사소 작업 판정 기준`.
- For substantial work, read `docs/contracts/product-contract.md` before editing.
- For multi-step or multi-file work, create or update a task document under `docs/tasks/` from `docs/tasks/TASK_TEMPLATE.md`.
- Task docs must list:
  - protected contract IDs to preserve,
  - allowed files,
  - explicit verification commands,
  - any contract IDs that intentionally change.
- Do not silently change behavior covered by the product contract. Update the contract doc and linked tests in the same task if behavior must change.
- Do not run multiple coding agents in the same git worktree. Use separate branches or `git worktree` instances for parallel work.
- When `git status` shows unrelated dirty files, treat them as user-owned changes and do not overwrite or refactor through them unless the task explicitly requires it.
- If a task grows beyond its allowed files, update the task doc first and then edit code.
- Prefer protecting behavior with regression tests in `tests/api`, `backend/tests`, or frontend `__tests__` rather than relying on prose alone.

## 7) Agent Checklist
Before edits:
- Run `git status --short` and note unrelated dirty files.
- Find and follow nearby code patterns in the same layer.
- Confirm backend/frontend impact and plan matching verification scope.
- Identify the relevant contract IDs from `docs/contracts/product-contract.md`.
- If the task is non-trivial, create or update the active task doc before editing files.

After edits:
- Compare the final diff against the task doc's allowed files.
- Run targeted tests first, then broader tests as needed.
- Run backend Ruff + MyPy for Python changes.
- Run frontend build + tests for TS/TSX changes.
- Run the verification commands listed in the task doc.
- Keep changes scoped; avoid unrelated refactors unless requested.
- If protected behavior changed, update `docs/contracts/product-contract.md` and the linked tests in the same change.

When uncertain, prefer existing repository conventions over generic framework defaults.
