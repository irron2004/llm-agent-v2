# Docker Dev/Prod Isolation + Fast Dev Loop

## TL;DR
> **Summary**: Make `prod` containers immutable (no source bind mounts) and make `dev` instantly responsive (backend `--reload` + frontend Vite HMR) without rerunning `make dev-up` for every change.
> **Deliverables**: `docker-compose.yml` updated (prod mounts removed, dev commands updated, Vite-based `frontend-dev`), `Makefile` dev/prod targets updated (`dev-up` fast, `dev-rebuild` explicit).
> **Effort**: Short
> **Parallel**: YES - 2 waves
> **Critical Path**: Compose changes -> Make targets -> Verification

## Context
### Original Request
- Read `docs/2026-03-09-docker-구조변경.md`, confirm alignment with the desired direction, and produce an execution plan.

### Alignment Check (doc vs recommended direction)
- Doc goals match exactly: prod immutable (image-only, no code bind mounts), dev hot reload + Vite HMR, Elasticsearch shared.
- This plan intentionally does NOT add extra scope such as multi-file compose split or CI image registry deployment.

### Metis Review (gaps addressed)
- Ensure `frontend-dev` can install deps reliably using a `node_modules` volume + idempotent `npm ci` bootstrap.
- QA should not assume `/api/health`; use an existing stable `/api/*` endpoint for proxy validation.
- Guardrail: do not touch ES setup and do not split compose files (explicitly out of scope).

## Work Objectives
### Core Objective
- Eliminate prod/dev coupling through bind mounts and remove rebuild-heavy dev workflow.

### Deliverables
- `docker-compose.yml`
  - `api` (prod) has no bind mounts to `/app/backend` or `/app/scripts`.
  - `api-dev` runs Uvicorn with `--reload`.
  - `frontend-dev` runs Vite dev server (Node runtime, bind mount, HMR).
  - Add named volume for `frontend-dev` dependencies (`/app/node_modules`).
- `Makefile`
  - `dev-up` no longer runs `down` or `--build`.
  - New `dev-rebuild` target to rebuild explicitly.
  - `prod-up` no longer runs `down` but keeps `--build`.

### Definition of Done (verifiable conditions with commands)
- Config-level checks:
  - `docker compose --env-file .env --env-file .env.prod --profile prod config` shows no `/app/backend` or `/app/scripts` mounts for `api`.
  - `docker compose --env-file .env --env-file .env.dev --profile dev config` shows:
    - `api-dev` command includes `uvicorn ... --reload`.
    - `frontend-dev` uses `node:20` (or equivalent) and runs Vite with `--host 0.0.0.0`.
  - `make -n dev-up` prints a compose `up` command with neither `down` nor `--build`.
  - `make -n prod-up` prints a compose `up` command with `--build` and no `down`.
- Runtime smoke:
  - `curl -sf http://localhost:${API_PORT:-8001}/health` succeeds.
  - `curl -sf http://localhost:${API_DEV_PORT:-8011}/health` succeeds.
  - `curl -sf http://localhost:${FRONTEND_DEV_PORT:-9098}/api/ingestions/runs` succeeds (proves Vite proxy -> `api-dev`).

### Must Have
- Prod is immutable w.r.t. host source edits.
- Dev UI/API reflect edits without `docker compose ... --build` loops.

### Must NOT Have (guardrails)
- Do NOT change Elasticsearch container, ports, or `ES_DATA_PATH` behavior.
- Do NOT split compose into multiple files (keep current `docker-compose.yml` + profiles).
- Do NOT change prod API/FE ports, routes, or env semantics except for mount isolation.
- Do NOT reintroduce `down` into default dev workflow.

## Verification Strategy
> ZERO HUMAN INTERVENTION: all checks via Bash/Compose inspection + HTTP probes.
- Test decision: none (infra/config change); rely on `docker compose config`, `docker inspect`, and `curl`.
- Evidence: store command outputs in `.sisyphus/evidence/task-{N}-{slug}.txt`.

## Execution Strategy
### Parallel Execution Waves
Wave 1: Docker Compose changes (prod immutability + dev reload/HMR wiring)
Wave 2: Make targets + end-to-end verification

### Dependency Matrix (full)
- 1 blocks 2
- 2 blocks final verification wave

### Agent Dispatch Summary
- Wave 1: 1 task (unspecified-low)
- Wave 2: 1 task (unspecified-low)

## TODOs

- [ ] 1. Update `docker-compose.yml` for prod immutability and dev hot reload

  **What to do**:
  - Update `services.api.volumes` in `docker-compose.yml` to remove the two source bind mounts:
    - remove `./backend:/app/backend`
    - remove `./scripts:/app/scripts`
    - keep all data/cache/log mounts unchanged (e.g., `/home/llm-share/hf:/data/hf_cache`, `./data/ingestions:/data/ingestions`, `./data/vector_store:/data/vector_store`, log mounts).
  - Update `services.api-dev` to explicitly run Uvicorn with reload:
    - Add `command:` overriding the image CMD to: `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload`.
    - Keep existing dev bind mounts in place.
  - Replace `services.frontend-dev` definition to run Vite dev server in Node:
    - Use `image: node:20` (remove `build:` and nginx envs).
    - Add `working_dir: /app`.
    - Preserve these fields from the existing service unless they must change:
      - `container_name: rag-frontend-dev`
      - `depends_on: api-dev`
      - `networks: rag_net`
      - `restart: unless-stopped`
      - `ports: ["${FRONTEND_DEV_PORT:-9098}:9097"]`
      - `profiles: [dev]`
    - Set volumes:
      - `./frontend:/app`
      - `frontend_node_modules:/app/node_modules`
      - keep `./data/ingestions:/data/ingestions`.
    - Set environment:
      - `API_PROXY_TARGET=http://api-dev:8000`
      - `PORT=9097`
    - Set command to be reliable across restarts:
      - `sh -lc 'if [ ! -d node_modules ] || [ -z "$(ls -A node_modules 2>/dev/null)" ]; then npm ci; fi; npm run dev -- --host 0.0.0.0 --port 9097 --strictPort'`
  - Add the named volume `frontend_node_modules:` at the bottom-level `volumes:` in `docker-compose.yml`.

  **Must NOT do**:
  - Do not modify `frontend` (prod) nginx-based service.
  - Do not change Elasticsearch services, volumes, or ports.
  - Do not remove `api-dev` bind mounts.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: small, localized infra YAML change
  - Skills: none
  - Omitted: `playwright` - no browser automation required for this task

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2 | Blocked By: none

  **References**:
  - Design doc: `docs/2026-03-09-docker-구조변경.md` - required behavior + checkpoints
  - Current prod bind mounts: `docker-compose.yml:28`
  - Current dev bind mounts: `docker-compose.yml:70`
  - Current `frontend-dev` nginx setup: `docker-compose.yml:300`
  - Vite proxy and `/data` middleware: `frontend/vite.config.ts:9`
  - API health endpoint: `backend/api/routers/health.py:6`
  - Stable `/api` endpoint for proxy validation: `backend/api/routers/ingestions.py:27`

  **Acceptance Criteria** (agent-executable only):
  - [ ] `docker compose --env-file .env --env-file .env.prod --profile prod config` output for `api` includes no `/app/backend` and no `/app/scripts` mounts.
  - [ ] `docker compose --env-file .env --env-file .env.dev --profile dev config` output for `api-dev` shows `uvicorn ... --reload` in `command`.
  - [ ] `docker compose --env-file .env --env-file .env.dev --profile dev config` output for `frontend-dev` shows a Node image + Vite command + `frontend_node_modules` volume.

  **QA Scenarios** (MANDATORY):
  ```
  Scenario: Prod api has no source mounts
    Tool: Bash
    Steps:
      1) docker compose --env-file .env --env-file .env.prod --profile prod up -d --build
      2) docker inspect rag-api --format '{{range .Mounts}}{{println .Destination}}{{end}}'
      3) curl -sf "http://localhost:${API_PORT:-8001}/health"
    Expected:
      - Output does NOT contain /app/backend
      - Output does NOT contain /app/scripts
      - Health endpoint returns HTTP 200
    Evidence: .sisyphus/evidence/task-1-compose-prod-mounts.txt

  Scenario: Dev frontend proxies /api to api-dev
    Tool: Bash
    Steps:
      1) docker compose --env-file .env --env-file .env.dev --profile dev up -d
      2) for i in {1..60}; do curl -sf "http://localhost:${FRONTEND_DEV_PORT:-9098}/api/ingestions/runs" && break; sleep 1; done
    Expected:
      - HTTP 200 and JSON body with keys 'folders' and 'base_path'
    Evidence: .sisyphus/evidence/task-1-vite-proxy.txt
  ```

  **Commit**: YES | Message: `chore(docker): isolate prod mounts and enable dev reload` | Files: `docker-compose.yml`

- [ ] 2. Update Makefile targets for fast dev loop and safer prod up

  **What to do**:
  - Update `.PHONY` list to include `dev-rebuild`.
  - Update `help:` text to document `dev-rebuild`.
  - Modify `dev-up` target in `Makefile`:
    - Remove the `docker compose ... down` line.
    - Remove `--build` from the `up` line.
    - Keep `--env-file .env --env-file .env.dev --profile dev`.
  - Add `dev-rebuild` target:
    - Runs `docker compose --env-file .env --env-file .env.dev --profile dev up -d --build`.
  - Modify `prod-up` target:
    - Remove the `docker compose ... down` line.
    - Keep `docker compose --env-file .env --env-file .env.prod --profile prod up -d --build`.
  - Ensure `up-dev`/`up-prod` aliases remain valid.

  **Must NOT do**:
  - Do not change unrelated targets (logs, vllm, etc.).
  - Do not change default ports or env file names.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: simple Makefile refactor + verification
  - Skills: none
  - Omitted: `git-master` - no complex history work needed

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: final verification wave | Blocked By: 1

  **References**:
  - Current targets: `Makefile:72`, `Makefile:76`
  - Desired behavior: `docs/2026-03-09-docker-구조변경.md`

  **Acceptance Criteria**:
  - [ ] `make -n dev-up` output contains neither `down` nor `--build`.
  - [ ] `make -n prod-up` output contains `--build` and does not contain `down`.
  - [ ] `make -n dev-rebuild` output contains `--build`.
  - [ ] Running `make dev-up` twice does not trigger rebuild logs (no `Building` lines in compose output).

  **QA Scenarios**:
  ```
  Scenario: Make targets print correct compose commands
    Tool: Bash
    Steps:
      1) make -n dev-up
      2) make -n dev-rebuild
      3) make -n prod-up
    Expected:
      - dev-up: no 'down', no '--build'
      - dev-rebuild: includes '--build'
      - prod-up: includes '--build' and no 'down'
    Evidence: .sisyphus/evidence/task-2-make-dryrun.txt

  Scenario: Dev-up is fast and non-destructive
    Tool: Bash
    Steps:
      1) make dev-up
      2) make dev-up
    Expected:
      - Second run completes quickly and does not rebuild images
    Evidence: .sisyphus/evidence/task-2-dev-up-second-run.txt
  ```

  **Commit**: YES | Message: `chore(make): speed up dev-up; add dev-rebuild` | Files: `Makefile`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit - oracle
- [ ] F2. Code Quality Review - unspecified-high
- [ ] F3. Runtime Smoke (prod+dev) - unspecified-high
- [ ] F4. Scope Fidelity Check - deep

## Commit Strategy
- Preferred: 2 commits (compose changes, then Makefile changes) to keep rollback easy.
- If repo policy prefers one commit: `chore(docker): dev/prod isolation and faster dev loop`.

## Success Criteria
- Editing files under `./backend` does not change running prod behavior until `make prod-up` is executed.
- Dev containers remain running; editing backend triggers reload; frontend edits HMR without rebuild.
- `make dev-up` is a quick "ensure running" command, not a rebuild loop.
