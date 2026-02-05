"""FastAPI application entrypoint."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import agent, assets, batch_answer, chat, conversations, devices, feedback, health, ingestions, logs, preprocessing, query_expansion, rerank, retrieval_evaluation, search, summarization
from backend.api.dependencies import set_search_service
from backend.config.settings import api_settings, rag_settings, search_settings
from backend.llm_infrastructure.preprocessing import get_preprocessor
from backend.services.document_service import DocumentIndexService
from backend.services.embedding_service import EmbeddingService
from backend.services.search_service import SearchService
from backend.services.es_search_service import EsSearchService
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager

logging.basicConfig(level=logging.INFO)
APP_VERSION = "0.1.0"
logger = logging.getLogger(__name__)


def _setup_rag_trace_logger() -> None:
    """RAG 파이프라인 트레이스 로거 설정.

    로그 파일: logs/rag_trace.log (최대 10MB, 5개 백업)
    형식: JSON (한 줄씩)
    """
    trace_logger = logging.getLogger("rag_trace")
    trace_logger.setLevel(logging.INFO)

    # 이미 핸들러가 있으면 추가하지 않음
    if trace_logger.handlers:
        return

    # 로그 디렉토리 생성
    log_dir = Path(os.getenv("RAG_TRACE_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "rag_trace.log"

    # 파일 핸들러 (회전, 최대 10MB, 5개 백업)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    # JSON 형식이므로 포맷터는 단순하게
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    trace_logger.addHandler(file_handler)

    # 콘솔에도 출력 (환경변수로 제어)
    if os.getenv("RAG_TRACE_CONSOLE", "false").lower() == "true":
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("[RAG_TRACE] %(message)s"))
        trace_logger.addHandler(console_handler)

    # 상위 로거로 전파하지 않음
    trace_logger.propagate = False

    logger.info("RAG trace logger configured: %s", log_file)


# 앱 시작 시 트레이스 로거 설정
_setup_rag_trace_logger()


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(
        title=api_settings.title or "LLM Infrastructure API",
        version=api_settings.version or APP_VERSION,
        description=api_settings.description or None,
        debug=api_settings.reload,  # align debug flag with reload setting
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(preprocessing.router)
    app.include_router(agent.router, prefix="/api")
    app.include_router(assets.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(rerank.router, prefix="/api")
    app.include_router(query_expansion.router, prefix="/api")
    app.include_router(summarization.router, prefix="/api")
    app.include_router(ingestions.router, prefix="/api")
    app.include_router(devices.router, prefix="/api")
    app.include_router(conversations.router, prefix="/api")
    app.include_router(feedback.router, prefix="/api")
    app.include_router(retrieval_evaluation.router, prefix="/api")
    app.include_router(batch_answer.router, prefix="/api")
    app.include_router(logs.router, prefix="/api")

    @app.on_event("startup")
    async def startup_search_service():
        """Wire SearchService at startup based on SEARCH_* settings."""
        try:
            _configure_search_service()
        except NotImplementedError as exc:  # explicit backend stub
            logger.warning(str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Search service not configured: {exc}")

    return app


def _configure_search_service() -> None:
    """Configure SearchService from environment settings."""
    backend = (search_settings.backend or "").lower()

    if backend == "local":
        if not search_settings.local_index_path:
            logger.info("SEARCH_LOCAL_INDEX_PATH is not set; search service left unconfigured.")
            return

        # Build embedder/preprocessor consistent with indexing
        embed_svc = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
            device=rag_settings.embedding_device,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )
        preprocessor = get_preprocessor(
            rag_settings.preprocess_method,
            version=rag_settings.preprocess_version,
            level=rag_settings.preprocess_level,
        )

        corpus = DocumentIndexService.load(
            search_settings.local_index_path,
            build_sparse=True,
            bm25_tokenizer=None,
            embedder=embed_svc.get_raw_embedder(),
            preprocessor=preprocessor,
        )

        svc = SearchService(
            corpus,
            method=rag_settings.retrieval_method,
            version=rag_settings.retrieval_version,
            top_k=rag_settings.retrieval_top_k,
            dense_weight=rag_settings.hybrid_dense_weight,
            sparse_weight=rag_settings.hybrid_sparse_weight,
            rrf_k=rag_settings.hybrid_rrf_k,
            multi_query_enabled=rag_settings.multi_query_enabled,
            multi_query_method=rag_settings.multi_query_method,
            multi_query_n=rag_settings.multi_query_n,
            multi_query_include_original=rag_settings.multi_query_include_original,
            multi_query_prompt=rag_settings.multi_query_prompt,
            rerank_enabled=rag_settings.rerank_enabled,
            rerank_method=rag_settings.rerank_method,
            rerank_model=rag_settings.rerank_model,
            rerank_top_k=rag_settings.rerank_top_k,
            rerank_device=rag_settings.embedding_device,
        )

        set_search_service(svc)
        logger.info("Search service configured with local index: %s", search_settings.local_index_path)
        return

    if backend == "es":
        # Wire Elasticsearch-backed search service
        index_manager = EsIndexManager(
            es_host=search_settings.es_host,
            env=search_settings.es_env,
            index_prefix=search_settings.es_index_prefix,
            es_user=search_settings.es_user or None,
            es_password=search_settings.es_password or None,
        )

        index_alias = index_manager.get_alias_name()
        es_search = EsSearchService.from_settings(index=index_alias)
        set_search_service(es_search)
        logger.info("Search service configured with Elasticsearch alias: %s", index_alias)
        return

    logger.warning("Unknown SEARCH_BACKEND '%s'; search service not configured.", backend)


# Uvicorn entrypoint
app = create_app()
