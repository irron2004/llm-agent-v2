"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import chat, health, preprocessing, search

APP_VERSION = "0.1.0"


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(
        title="LLM Infrastructure API",
        version=APP_VERSION,
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
    app.include_router(chat.router, prefix="/api")
    app.include_router(search.router, prefix="/api")

    return app


# Uvicorn entrypoint
app = create_app()
