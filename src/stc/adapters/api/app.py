"""FastAPI application factory with lifespan model loading (D14/S14)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from stc import __version__
from stc.adapters.api.errors import install_error_handlers
from stc.adapters.api.middleware import RequestContextMiddleware
from stc.adapters.api.provider import ClassifierProvider
from stc.adapters.api.routes import WEB_ROOT, router
from stc.config import Settings

__all__ = ["create_app"]


def create_app(settings: Settings, provider: ClassifierProvider) -> FastAPI:
    """Build the API around a classifier provider and typed settings."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = settings
        application.state.provider = provider
        application.state.default_model = provider.load_default()
        application.state.baseline_model = None
        application.state.baseline_available = provider.baseline_available()
        yield

    application = FastAPI(
        title="Spanish text classifier",
        description="Calibrated Spanish intent classification over ONNX Runtime.",
        version=__version__,
        lifespan=lifespan,
    )
    application.add_middleware(RequestContextMiddleware)
    install_error_handlers(application)
    application.include_router(router)
    static_root = Path(WEB_ROOT)
    if static_root.is_dir():
        application.mount("/static", StaticFiles(directory=static_root), name="static")
    return application
