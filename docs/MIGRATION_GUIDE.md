# Migration Guide: llm-agent → llm-agent-refactor

## Overview

This guide provides a detailed mapping from the original project structure to the refactored structure, along with step-by-step migration instructions.

**Original Path**: `/home/hskim/work/llm-agent`
**New Path**: `/home/hskim/work/llm-agent-refactor`

## Architecture Changes

### Old Structure (Mixed)
```
llm-agent/
├── core/                    # Mixed infrastructure
├── services/pe_agent/       # Mixed API + domain + infrastructure
├── docker/ragflow-stack/    # Docker + Web UI mixed
└── [many test/experiment files in root]
```

### New Structure (Layered)
```
llm-agent-refactor/
├── backend/
│   ├── api/                # API Layer (HTTP only)
│   ├── services/           # Service Layer (orchestration)
│   ├── domain/             # Domain Layer (PE logic)
│   ├── llm_infrastructure/ # Infrastructure Layer (reusable)
│   └── config/             # Configuration
├── frontend/               # Frontend (separated)
├── docker/                 # Docker Stack (clean)
└── experiments/            # Experiment runner
```

## Detailed Migration Mapping

### 1. LLM Infrastructure Layer

| Original Path | New Path | Notes |
|---------------|----------|-------|
| `core/embedding/` | `backend/llm_infrastructure/embedding/` | Embedding models & TEI client |
| `core/embedding/embedders/` | `backend/llm_infrastructure/embedding/embedders/` | SentenceTransformer, TEI implementations |
| `core/embedding/cache.py` | `backend/llm_infrastructure/embedding/cache.py` | Embedding cache |
| | | |
| `core/llm/` | `backend/llm_infrastructure/llm/` | LLM engines & prompts |
| `core/llm/engines/` | `backend/llm_infrastructure/llm/engines/` | vLLM, Transformers engines |
| `core/llm/prompts.py` | `backend/llm_infrastructure/llm/prompts/` | Prompt templates |
| `core/llm/utils/vllm_client.py` | `backend/llm_infrastructure/llm/client.py` | vLLM client |
| | | |
| `core/retrieval/` | `backend/llm_infrastructure/retrieval/` | RAG pipeline & hybrid search |
| `core/retrieval/hybrid.py` | `backend/llm_infrastructure/retrieval/methods/hybrid.py` | Hybrid retrieval |
| `core/retrieval/pipeline.py` | `backend/llm_infrastructure/retrieval/methods/pipeline.py` | RAG pipeline |
| `core/retrieval/multiquery.py` | `backend/llm_infrastructure/retrieval/multi_query.py` | Multi-query expansion |
| | | |
| `core/rag/` | `backend/llm_infrastructure/retrieval/` | Merge into retrieval |
| | | |
| [scattered preprocessing] | `backend/llm_infrastructure/preprocessing/` | **NEW**: Centralized preprocessing |
| `scripts/*chunking*` | `backend/llm_infrastructure/preprocessing/chunking.py` | Chunking logic |
| `scripts/*parser*` | `backend/llm_infrastructure/preprocessing/parsers/` | Document parsers |

### 2. PE Agent Domain Layer

| Original Path | New Path | Notes |
|---------------|----------|-------|
| `services/pe_agent/pe_core/` | `backend/domain/pe_core/` | PE domain logic |
| `services/pe_agent/pe_core/ingestion.py` | `backend/domain/pe_core/ingestion.py` | Document ingestion |
| `services/pe_agent/pe_core/parsers/` | `backend/domain/pe_core/parsers/` | Domain-specific parsers |
| `services/pe_agent/pe_core/retrieval_builders.py` | `backend/domain/pe_core/retrieval_builders.py` | PE-specific retrieval |
| `services/pe_agent/pe_core/chapter_index.py` | `backend/domain/pe_core/chapter_index.py` | Chapter indexing |
| `services/pe_agent/pe_core/models.py` | `backend/domain/pe_core/models.py` | Domain models |

### 3. API Layer

| Original Path | New Path | Notes |
|---------------|----------|-------|
| `services/pe_agent/api/app/main.py` | `backend/api/main.py` | FastAPI app factory |
| `services/pe_agent/api/app/routers/` | `backend/api/routers/` | All routers |
| `services/pe_agent/api/app/routers/chat.py` | `backend/api/routers/chat.py` | Chat endpoint |
| `services/pe_agent/api/app/routers/search.py` | `backend/api/routers/search.py` | Search endpoint |
| `services/pe_agent/api/app/routers/answer.py` | `backend/api/routers/answer.py` | Answer endpoint |
| `services/pe_agent/api/app/routers/ragflow.py` | `backend/api/routers/ragflow.py` | RAGFlow integration |
| `services/pe_agent/api/app/routers/monitoring.py` | `backend/api/routers/monitoring.py` | Monitoring endpoint |

### 4. RAGFlow Client

| Original Path | New Path | Notes |
|---------------|----------|-------|
| `services/pe_agent/ragflow_agent_client.py` | `backend/llm_infrastructure/retrieval/ragflow_client.py` | RAGFlow client |
| `services/pe_agent/api/app/ragflow_client.py` | `backend/llm_infrastructure/retrieval/ragflow_client.py` | Consolidate clients |

### 5. Configuration

| Original Path | New Path | Notes |
|---------------|----------|-------|
| `services/pe_agent/config/` | `backend/config/` | Settings & presets |
| [env variables scattered] | `backend/config/settings.py` | **NEW**: Pydantic Settings |
| [no preset files] | `backend/config/presets/` | **NEW**: YAML presets |

### 6. Frontend

| Original Path | New Path | Notes |
|---------------|----------|-------|
| `docker/ragflow-stack/web/src/` | `frontend/src/` | React source code |
| `docker/ragflow-stack/web/dist/` | `frontend/dist/` | Build output |
| `docker/ragflow-stack/web/package.json` | `frontend/package.json` | Frontend config |

### 7. Docker Stack

| Original Path | New Path | Notes |
|---------------|----------|-------|
| `docker/ragflow-stack/docker-compose.yaml` | `docker/docker-compose.yaml` | Base compose file |
| `docker/ragflow-stack/docker-compose.with-ragflow.yml` | `docker/docker-compose.with-ragflow.yml` | RAGFlow overlay |
| `docker/ragflow-stack/Dockerfile.fastapi` | `docker/Dockerfile.backend` | Backend container |
| `docker/ragflow-stack/Dockerfile.vllm` | `docker/Dockerfile.vllm` | vLLM container |
| `docker/ragflow-stack/Dockerfile.streamlit` | [removed] | Use React frontend |
| `docker/ragflow-stack/web/Dockerfile` | `docker/Dockerfile.frontend` | Frontend container |
| `docker/ragflow-stack/Makefile` | `docker/Makefile` | Make commands |
| `docker/ragflow-stack/ragflow-stack.env` | `docker/.env.example` | Environment template |

### 8. Data & Documentation

| Original Path | New Path | Notes |
|---------------|----------|-------|
| `data/` | `data/` | Keep as-is (or symlink) |
| `docs/` (relevant) | `docs/` | Copy relevant docs only |

## Files NOT to Migrate

### Root-level Test/Experiment Files
```
# DO NOT MIGRATE these files:
test_*.py
*_api.py (except core APIs)
monitor_*.py
ragflow_api_explorer.py
retrieval_api.py
analyze_*.py
```

### Experiment/Example Directories
```
# DO NOT MIGRATE:
/experiments/*     # Old experiment scripts
/examples/*        # Example scripts
/scripts/*         # One-off scripts
```

### Other Services
```
# DO NOT MIGRATE (other projects):
services/rtl_demo/
services/text2sql/
services/log2nl/
services/pe_team_bot/
services/mcp_server/
services/graph_ingest/
```

## Step-by-Step Migration Process

### Phase 1: Infrastructure Layer ✅ COMPLETED
- [x] Create directory structure
- [x] Implement Registry pattern for preprocessing
- [x] Implement Registry pattern for embedding
- [x] Implement Registry pattern for retrieval
- [x] Create config and preset structure
- [x] Create experiment runner skeleton

### Phase 2: Migrate Core Libraries (Next Steps)
1. **Embedding Module**
   ```bash
   # Copy files
   cp -r core/embedding/* backend/llm_infrastructure/embedding/

   # Update imports
   find backend/llm_infrastructure/embedding -type f -name "*.py" -exec \
     sed -i 's/from core\.embedding/from backend.llm_infrastructure.embedding/g' {} \;

   # Register embedders
   # Edit each embedder to add @register_embedder decorator
   ```

2. **LLM Module**
   ```bash
   cp -r core/llm/* backend/llm_infrastructure/llm/

   # Update imports
   find backend/llm_infrastructure/llm -type f -name "*.py" -exec \
     sed -i 's/from core\.llm/from backend.llm_infrastructure.llm/g' {} \;
   ```

3. **Retrieval Module**
   ```bash
   cp -r core/retrieval/* backend/llm_infrastructure/retrieval/

   # Update imports and register retrievers
   ```

4. **Add Preprocessing**
   ```bash
   # Extract preprocessing logic from scattered locations
   # Implement in backend/llm_infrastructure/preprocessing/
   ```

### Phase 3: Migrate Domain Layer
1. **PE Core**
   ```bash
   cp -r services/pe_agent/pe_core/* backend/domain/pe_core/

   # Update imports to use new infrastructure paths
   find backend/domain/pe_core -type f -name "*.py" -exec \
     sed -i 's/from core\./from backend.llm_infrastructure./g' {} \;
   ```

### Phase 4: Create Service Layer (NEW)
1. **Create orchestration services**
   - Implement `backend/services/chat_service.py`
   - Implement `backend/services/search_service.py`
   - Implement `backend/services/document_service.py`

2. **Extract business logic from routers**
   - Move complex logic from routers to services
   - Routers should only handle HTTP concerns

### Phase 5: Migrate API Layer
1. **Copy routers**
   ```bash
   cp -r services/pe_agent/api/app/routers/* backend/api/routers/
   cp services/pe_agent/api/app/main.py backend/api/main.py
   ```

2. **Update imports**
   ```bash
   # Update all imports to new paths
   # Point to service layer instead of direct infrastructure calls
   ```

3. **Add middleware** (if needed)
   - Authentication
   - Logging
   - Rate limiting

### Phase 6: Migrate Configuration
1. **Create settings.py** ✅ DONE
2. **Create preset files** ✅ DONE
3. **Create .env.example** ✅ DONE
4. **Test configuration loading**

### Phase 7: Migrate Frontend
1. **Copy source files**
   ```bash
   cp -r docker/ragflow-stack/web/src/* frontend/src/
   cp docker/ragflow-stack/web/package.json frontend/
   ```

2. **Update API endpoints**
   - Change API base URL to new backend
   - Update endpoint paths if changed

3. **Test build**
   ```bash
   cd frontend
   npm install
   npm run build
   ```

### Phase 8: Migrate Docker
1. **Copy Docker files**
   ```bash
   cp docker/ragflow-stack/docker-compose.yaml docker/
   cp docker/ragflow-stack/docker-compose.with-ragflow.yml docker/
   cp docker/ragflow-stack/Makefile docker/
   ```

2. **Update Dockerfiles**
   - Update paths to reflect new structure
   - Update WORKDIR, COPY paths

3. **Update docker-compose.yaml**
   - Update volume mounts: `./backend:/opt/app/backend`
   - Update build contexts

4. **Update Makefile**
   - Adjust paths if needed

### Phase 9: Testing & Validation
1. **Backend tests**
   ```bash
   cd backend
   python -m pytest tests/
   ```

2. **Import tests**
   ```python
   # Test that all registries work
   from backend.llm_infrastructure.preprocessing.registry import get_preprocessor
   from backend.llm_infrastructure.embedding.registry import get_embedder
   from backend.llm_infrastructure.retrieval.registry import get_retriever

   preprocessor = get_preprocessor("standard")
   embedder = get_embedder("bge_base")
   # etc.
   ```

3. **Docker tests**
   ```bash
   cd docker
   make up
   make status
   make logs
   ```

4. **API tests**
   ```bash
   curl http://localhost:8100/health
   curl http://localhost:8100/docs
   ```

## Import Path Changes Reference

### Before → After

```python
# Preprocessing
from core.preprocessing import ...
→ from backend.llm_infrastructure.preprocessing import ...

# Embedding
from core.embedding.embedders import create_embedder
→ from backend.llm_infrastructure.embedding.embedders import ...
# OR use registry:
→ from backend.llm_infrastructure.embedding.registry import get_embedder

# LLM
from core.llm.engines.vllm import VLLMEngine
→ from backend.llm_infrastructure.llm.engines.vllm import VLLMEngine

# Retrieval
from core.retrieval.hybrid import HybridRetriever
→ from backend.llm_infrastructure.retrieval.methods.hybrid import HybridRetriever
# OR use registry:
→ from backend.llm_infrastructure.retrieval.registry import get_retriever

# PE Domain
from services.pe_agent.pe_core.ingestion import ingest_documents
→ from backend.domain.pe_core.ingestion import ingest_documents

# Config
from services.pe_agent.config.settings import Settings
→ from backend.config.settings import rag_settings, vllm_settings
```

## Configuration Changes

### Environment Variables

```bash
# OLD: Scattered across multiple files
RAGFLOW_API_KEY=...
VLLM_BASE_URL=...

# NEW: Centralized in .env with prefixes
RAG_RAGFLOW_API_KEY=...
RAG_RETRIEVAL_PRESET=hybrid_rrf_v1
VLLM_BASE_URL=...
```

### Docker Volumes

```yaml
# OLD
volumes:
  - ../../services/pe_agent:/opt/app/services/pe_agent
  - ../../core:/opt/app/core

# NEW
volumes:
  - ./backend:/opt/app/backend
  - ./data:/opt/app/data
```

## Validation Checklist

- [ ] All `__init__.py` files in place
- [ ] No circular imports
- [ ] All registries initialized
- [ ] All tests pass
- [ ] Docker builds succeed
- [ ] Services start correctly
- [ ] API endpoints respond
- [ ] Frontend connects to backend
- [ ] RAGFlow integration works
- [ ] vLLM inference works
- [ ] Embedding service works
- [ ] Preprocessing pipeline works
- [ ] Retrieval pipeline works
- [ ] Experiment runner works

## Rollback Plan

If migration fails:

1. **Original project remains untouched** at `/home/hskim/work/llm-agent`
2. **Simply delete** `/home/hskim/work/llm-agent-refactor`
3. **No data loss** as `data/` is kept separate or symlinked
4. **Docker cleanup** if needed: `docker compose down -v`

## Next Steps After Migration

1. **Update documentation** to reflect new structure
2. **Run experiments** to validate RAG pipeline
3. **Add tests** for new service layer
4. **Set up CI/CD** for new structure
5. **Migrate other services** (if needed) using same pattern

## Questions?

If you encounter issues during migration:
1. Check this guide for the specific file/path mapping
2. Verify import paths are updated correctly
3. Check that all dependencies are installed
4. Ensure Docker volumes are mounted correctly
5. Review logs for specific error messages
