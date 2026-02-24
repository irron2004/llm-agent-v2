"""FastAPI application entrypoint."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import agent, assets, chat, conversations, devices, feedback, health, ingestions, preprocessing, query_expansion, rerank, retrieval, retrieval_evaluation, search, summarization
from backend.api.dependencies import set_search_service
from backend.config.settings import api_settings, search_settings
from backend.services.es_search_service import EsSearchService
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager

APP_VERSION = "0.1.0"
logger = logging.getLogger(__name__)


def _resolve_log_level(level: str) -> int:
    raw = (level or "info").strip().upper()
    return getattr(logging, raw, logging.INFO)


def _has_file_handler(target_logger: logging.Logger, file_path: Path) -> bool:
    for handler in target_logger.handlers:
        base_filename = getattr(handler, "baseFilename", None)
        if not base_filename:
            continue
        try:
            if Path(base_filename).resolve() == file_path.resolve():
                return True
        except Exception:
            continue
    return False


def _enable_file_logging() -> None:
    level = _resolve_log_level(api_settings.log_level)
    logging.basicConfig(level=level)

    if not api_settings.log_to_file:
        return

    log_path = Path(api_settings.log_file_path).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Cannot create log directory '%s': %s", log_path.parent, exc)
        return

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    target_logger_names = ("", "uvicorn", "uvicorn.error", "uvicorn.access", "fastapi")
    for logger_name in target_logger_names:
        target = logging.getLogger(logger_name)
        target.setLevel(level)
        if _has_file_handler(target, log_path):
            continue
        try:
            handler = RotatingFileHandler(
                log_path,
                maxBytes=max(1024, int(api_settings.log_max_bytes)),
                backupCount=max(1, int(api_settings.log_backup_count)),
                encoding="utf-8",
            )
            handler.setLevel(level)
            handler.setFormatter(formatter)
            target.addHandler(handler)
        except Exception as exc:
            logger.warning("Failed to attach file logging to '%s': %s", logger_name or "root", exc)

    logger.info("API file logging enabled: %s", log_path)


_enable_file_logging()


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
    app.include_router(retrieval.router, prefix="/api")
    app.include_router(retrieval_evaluation.router, prefix="/api")

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

    if backend == "es":
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
