"""The FastAPI application: lifespan, error mapping and routing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sigil import __version__
from sigil.api.container import Container
from sigil.api.middleware import BodyLimitMiddleware
from sigil.api.routes import system, v1
from sigil.config import Settings
from sigil.errors import (
    CaptureError,
    DecodeError,
    LogError,
    NotFoundError,
    OverloadedError,
    UnknownKeyError,
)

DESCRIPTION = """
Counterfeit detection for labelled goods: an Ed25519-signed QR credential, a secret
copy-detection pattern checked by a CNN, a public RFC 9162 transparency log, and an audited LLM
investigator for the cases the deterministic policy cannot settle.
"""


def create_app(settings: Settings | None = None, *, seed_demo: bool = True) -> FastAPI:
    """Build the app. ``seed_demo`` issues demo batches into an empty data directory on first
    start, so the web demo has labels to scan."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = Container(settings)
        await container.start()
        try:
            if seed_demo:
                await container.seed_demo()
            app.state.container = container
            yield
        finally:
            await container.close()

    app = FastAPI(title="Sigil", version=__version__, description=DESCRIPTION, lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)
    app.include_router(system.router)
    app.include_router(v1)
    _map_errors(app)
    return app


def _map_errors(app: FastAPI) -> None:
    """Translate domain errors to HTTP. Domain code never raises ``HTTPException``."""

    def respond(status: int, detail: str, **extra: object) -> JSONResponse:
        headers = {"Retry-After": "1"} if status in (429, 503) else None
        return JSONResponse({"detail": detail, **extra}, status_code=status, headers=headers)

    @app.exception_handler(NotFoundError)
    @app.exception_handler(UnknownKeyError)
    async def not_found(request: Request, exc: Exception) -> JSONResponse:
        return respond(404, str(exc))

    @app.exception_handler(CaptureError)
    async def capture_failed(request: Request, exc: CaptureError) -> JSONResponse:
        return respond(422, str(exc), reason=exc.reason)

    @app.exception_handler(DecodeError)
    async def undecodable(request: Request, exc: Exception) -> JSONResponse:
        return respond(422, str(exc))

    @app.exception_handler(LogError)
    async def bad_log_request(request: Request, exc: Exception) -> JSONResponse:
        return respond(400, str(exc))

    @app.exception_handler(OverloadedError)
    async def overloaded(request: Request, exc: Exception) -> JSONResponse:
        return respond(429, str(exc))

    @app.exception_handler(TimeoutError)
    async def timed_out(request: Request, exc: Exception) -> JSONResponse:
        return respond(503, "the inference service did not answer in time")
