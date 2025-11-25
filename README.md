# PE Agent Service (Refactored)

Retrieval-Augmented Generation (RAG) system for Process Engineering troubleshooting with **modular, research-friendly architecture**.

## 🎯 Key Features

- **Registry Pattern**: Plug-and-play preprocessing, embedding, and retrieval methods
- **Config-Driven Pipeline**: Switch methods via environment variables or YAML presets
- **Experiment Runner**: Built-in evaluation framework for testing new approaches
- **Layered Architecture**: Clean separation of API, service, domain, and infrastructure
- **Docker Stack**: Complete deployment with vLLM, RAGFlow, and TEI

## 📁 Project Structure

```
pe-agent-service/
├── backend/                    # Backend Application
│   ├── api/                   # FastAPI Layer (HTTP endpoints)
│   ├── services/              # Service Layer (business orchestration)
│   ├── domain/                # Domain Layer (PE-specific logic)
│   ├── llm_infrastructure/    # Infrastructure Layer (reusable)
│   │   ├── preprocessing/    # Registry-based preprocessing
│   │   ├── embedding/        # Registry-based embedding
│   │   ├── retrieval/        # Registry-based retrieval + presets
│   │   └── llm/              # LLM engines (vLLM, etc.)
│   └── config/                # Settings & presets
│
├── frontend/                   # Frontend (React UI)
├── docker/                     # Docker Stack
├── experiments/                # Experiment runner & configs
├── data/                       # Data volumes
└── docs/                       # Documentation
```

## 🏗️ Architecture

### Layered Design

```
API Layer → Service Layer → Domain Layer → Infrastructure Layer
```

- **API Layer**: HTTP endpoints only (FastAPI routers)
- **Service Layer**: Business logic orchestration
- **Domain Layer**: PE Agent domain-specific logic
- **Infrastructure Layer**: Reusable LLM components (preprocessing, embedding, retrieval)

### Registry Pattern

All major components use a **registry pattern** for dynamic selection:

```python
# Register a new preprocessing method
from backend.llm_infrastructure.preprocessing import register_preprocessor, BasePreprocessor

@register_preprocessor("my_method", version="v1")
class MyPreprocessor(BasePreprocessor):
    def preprocess(self, docs):
        # Your implementation
        return processed_docs

# Use it via config
# .env: RAG_PREPROCESS_METHOD=my_method
from backend.config.settings import rag_settings
from backend.llm_infrastructure.preprocessing.registry import get_preprocessor

preprocessor = get_preprocessor(rag_settings.preprocess_method)
```

Same pattern for:
- **Embedding**: `@register_embedder("name", version="v1")`
- **Retrieval**: `@register_retriever("name", version="v1")`

## 🚀 Quick Start

### 1. Set Up Environment

```bash
cd docker
cp .env.example .env
# Edit .env with your configuration

# Choose your preset and methods:
# RAG_PREPROCESS_METHOD=standard
# RAG_EMBEDDING_METHOD=bge_base
# RAG_RETRIEVAL_PRESET=hybrid_rrf_v1
```

### 2. Start Services

```bash
# Start full stack (backend + vLLM + RAGFlow + TEI)
make up

# Check status
make status

# View logs
make logs-fastapi
make logs-vllm
```

### 3. Access Services

- **PE Agent UI**: http://localhost:3000
- **FastAPI Docs**: http://localhost:8100/docs
- **RAGFlow**: http://localhost:9380

## 🧪 Running Experiments

The project includes a built-in experiment framework for evaluating RAG pipelines.

### Test a New Preprocessing Method

1. **Implement the method**:
```python
# backend/llm_infrastructure/preprocessing/methods/my_paper_method.py
from ..base import BasePreprocessor
from ..registry import register_preprocessor

@register_preprocessor("paper_xyz", version="v1")
class PaperXYZPreprocessor(BasePreprocessor):
    def preprocess(self, docs):
        # Implement paper's approach
        return processed_docs
```

2. **Create experiment config**:
```yaml
# experiments/configs/test_paper_xyz.yaml
name: test_paper_xyz
preprocess_method: paper_xyz
preprocess_version: v1
embedding_method: bge_base
retrieval:
  method: hybrid
  top_k: 50
```

3. **Run experiment**:
```bash
python -m experiments.run \
    --config experiments/configs/test_paper_xyz.yaml \
    --dataset data/eval/pe_agent_eval.jsonl \
    --output experiments/runs/test_paper_xyz/
```

4. **View results**:
```bash
cat experiments/runs/test_paper_xyz/metrics.json
```

See `experiments/README.md` for more details.

## 📋 Configuration

### Environment Variables

All configuration is centralized in `.env`:

```bash
# Preprocessing
RAG_PREPROCESS_METHOD=standard  # or: pe_domain, custom_method
RAG_PREPROCESS_VERSION=v1

# Embedding
RAG_EMBEDDING_METHOD=bge_base   # or: bge_large, multilingual_e5, tei
RAG_EMBEDDING_VERSION=v1

# Retrieval
RAG_RETRIEVAL_PRESET=hybrid_rrf_v1  # See config/presets/

# RAGFlow
RAG_RAGFLOW_ENABLED=true
RAG_RAGFLOW_API_KEY=your-key
RAG_RAGFLOW_AGENT_ID=your-agent-id

# vLLM
VLLM_BASE_URL=http://vllm:8000
VLLM_MODEL_NAME=gpt-oss-20b
```

### Retrieval Presets

Predefined configurations in `backend/config/presets/`:

- **`dense_only.yaml`**: Pure semantic search (baseline)
- **`hybrid_rrf_v1.yaml`**: Hybrid dense+sparse with RRF fusion (recommended)
- **`hybrid_multi_query.yaml`**: Hybrid + multi-query expansion
- **`hybrid_rerank.yaml`**: Hybrid + cross-encoder reranking
- **`full_pipeline.yaml`**: All features enabled (best quality, higher latency)

Create custom presets by copying and modifying existing ones.

## 🔧 Development

### Add a New Embedding Method

```python
# backend/llm_infrastructure/embedding/embedders/my_embedder.py
from ..base import BaseEmbedder
from ..registry import register_embedder

@register_embedder("my_embedder", version="v1")
class MyEmbedder(BaseEmbedder):
    def embed(self, text):
        # Implementation
        return embedding_vector

    def embed_batch(self, texts, batch_size=32):
        # Batch implementation
        return embeddings_matrix
```

Then update `.env`:
```bash
RAG_EMBEDDING_METHOD=my_embedder
RAG_EMBEDDING_VERSION=v1
```

### Add a New Retrieval Method

```python
# backend/llm_infrastructure/retrieval/methods/my_retriever.py
from ..base import BaseRetriever, RetrievalResult
from ..registry import register_retriever

@register_retriever("my_retriever", version="v1")
class MyRetriever(BaseRetriever):
    def retrieve(self, query, top_k=10):
        # Implementation
        return [
            RetrievalResult(doc_id="...", content="...", score=0.95),
            # ...
        ]
```

Create a preset in `config/presets/retrieval_my_method.yaml`.

## 📚 Documentation

- **[Migration Guide](docs/MIGRATION_GUIDE.md)**: Migrating from original structure
- **[Experiment Guide](experiments/README.md)**: Running experiments and ablation studies
- **[API Documentation](http://localhost:8100/docs)**: Interactive API docs (when running)

## 🎓 Research-Friendly Design

This architecture is designed for **rapid experimentation**:

1. **Read a new paper** with an interesting RAG technique
2. **Implement it** as a new registered method (preprocessing/embedding/retrieval)
3. **Create a config** referencing the new method
4. **Run experiments** comparing it to baselines
5. **Analyze results** and iterate

No need to modify existing code - just add new implementations and switch via config.

## 🛠️ Technology Stack

- **Backend**: FastAPI, Pydantic Settings
- **LLM**: vLLM (inference), TEI (embeddings)
- **RAG**: RAGFlow (document processing, vector DB)
- **Frontend**: React, Vite
- **Infrastructure**: Docker Compose, Nginx
- **Experiment Tracking**: YAML configs, JSONL results (extensible to WandB/MLflow)

## 📝 Migration from Original Project

This is a refactored version of `/home/hskim/work/llm-agent`.

Key improvements:
- ✅ Clean layered architecture
- ✅ Registry pattern for all major components
- ✅ Config-driven pipeline
- ✅ Built-in experiment framework
- ✅ Separated BE/FE
- ✅ No root-level clutter

See **[docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)** for detailed migration instructions.

## 🤝 Contributing

This project follows a modular, registry-based pattern. To contribute:

1. Add new methods in the appropriate `methods/` directory
2. Use the `@register_*` decorator
3. Add tests and documentation
4. Create example configs/presets

## 📄 License

[Your License Here]

## 🙋 Support

For questions or issues:
- Check documentation in `docs/`
- Review example configs in `experiments/configs/`
- See migration guide for path mappings# llm-agent-v2
