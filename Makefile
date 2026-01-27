.PHONY: clean clean-pyc clean-test clean-build help run-api stop-api up up-vllm build-up logs logs-api logs-es logs-vllm logs-tei

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(shell if [ -x "$(ROOT_DIR).venv/bin/python" ]; then echo "$(ROOT_DIR).venv/bin/python"; else echo python3; fi)
UVICORN_APP ?= backend.api.main:app
API_HOST ?= 0.0.0.0
API_PORT ?= 8001
API_RELOAD ?= true
API_LOG_LEVEL ?= info
API_PID_FILE ?= $(ROOT_DIR).api.pid
API_LOG_FILE ?= $(ROOT_DIR).api.log
UVICORN_RELOAD = $(if $(filter true,$(API_RELOAD)),--reload,)

DOCKER_COMPOSE ?= docker compose
DOCKER_SERVICES ?= api elasticsearch

help:
	@echo "clean        - remove all build, test, coverage and Python artifacts"
	@echo "clean-build  - remove build artifacts"
	@echo "clean-pyc    - remove Python file artifacts"
	@echo "clean-test   - remove test and coverage artifacts"
	@echo "run-api      - start FastAPI dev server (uvicorn) in background"
	@echo "stop-api     - stop FastAPI dev server started via run-api"
	@echo "up           - docker compose up -d (api + ES, uses external vLLM)"
	@echo "up-vllm      - docker compose up -d with vLLM (--profile with-vllm)"
	@echo "build-up     - docker compose build && up -d (rebuild and start)"
	@echo "logs         - tail docker logs for core services"
	@echo "logs-api     - tail docker logs for api"
	@echo "logs-es      - tail docker logs for elasticsearch"
	@echo "logs-vllm    - tail docker logs for vllm"
	@echo "logs-tei     - tail docker logs for tei"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf build/
	rm -rf dist/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -name '.pytest_cache' -exec rm -rf {} +

up:
	$(DOCKER_COMPOSE) up -d $(DOCKER_SERVICES)
	@echo ""
	@echo "=========================================="
	@echo " Services started: $(DOCKER_SERVICES)"
	@echo "=========================================="
	@echo ""
	@echo "[vLLM] Using external vLLM server"
	@echo "  URL: $${VLLM_BASE_URL:-http://localhost:8000}/v1"
	@echo "  Model: $${VLLM_MODEL_NAME:-openai/gpt-oss-20b}"
	@echo ""
	@echo "To run vLLM with docker compose:"
	@echo "  $(DOCKER_COMPOSE) --profile with-vllm up -d"
	@echo ""

up-vllm:
	$(DOCKER_COMPOSE) --profile with-vllm up -d

build-up:
	$(DOCKER_COMPOSE) build
	$(DOCKER_COMPOSE) down
	$(DOCKER_COMPOSE) up -d
	@echo ""
	@echo "=========================================="
	@echo " Rebuilt and started all services"
	@echo "=========================================="

logs:
	$(DOCKER_COMPOSE) logs -f $(DOCKER_SERVICES)

logs-api:
	$(DOCKER_COMPOSE) logs -f api

logs-es:
	$(DOCKER_COMPOSE) logs -f elasticsearch

logs-vllm:
	$(DOCKER_COMPOSE) logs -f vllm

logs-tei:
	$(DOCKER_COMPOSE) logs -f tei

run-api:
	@mkdir -p $(dir $(API_LOG_FILE))
	@if [ -f "$(API_PID_FILE)" ] && kill -0 $$(cat "$(API_PID_FILE)") > /dev/null 2>&1; then \
		echo "FastAPI already running (pid $$(cat $(API_PID_FILE)))"; \
	else \
		echo "Starting FastAPI in background..."; \
		nohup setsid $(PYTHON) -m uvicorn $(UVICORN_APP) \
			--host $(API_HOST) \
			--port $(API_PORT) \
			--log-level $(API_LOG_LEVEL) \
			$(UVICORN_RELOAD) \
			> "$(API_LOG_FILE)" 2>&1 & echo $$! > "$(API_PID_FILE)"; \
		echo "Started FastAPI (pid $$(cat $(API_PID_FILE))), logs: $(API_LOG_FILE)"; \
	fi

stop-api:
	@if [ -f "$(API_PID_FILE)" ]; then \
		PID=$$(cat "$(API_PID_FILE)"); \
		if kill -0 $$PID > /dev/null 2>&1; then \
			PGID=$$(ps -o pgid= $$PID | tr -d ' '); \
			if [ -n "$$PGID" ]; then \
				kill -- -$$PGID 2>/dev/null || true; \
			else \
				kill $$PID 2>/dev/null || true; \
			fi; \
			rm -f "$(API_PID_FILE)"; \
			echo "Stopped FastAPI (pid $$PID)"; \
		else \
			echo "No running FastAPI found for pid $$PID, removing stale pid file"; \
			rm -f "$(API_PID_FILE)"; \
		fi; \
	else \
		echo "No pid file found at $(API_PID_FILE); nothing to stop."; \
	fi
