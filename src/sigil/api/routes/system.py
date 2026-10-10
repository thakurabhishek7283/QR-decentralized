"""Health, metrics and the demo page."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from sigil.api.deps import ContainerDep
from sigil.api.schemas import Health

router = APIRouter()

INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"
# The page is self-contained, so it can run under a strict policy: no remote script, style or
# connection, and images only from this origin or the blobs it creates itself.
PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
}


@router.get("/healthz", response_model=Health, tags=["system"])
async def healthz(c: ContainerDep) -> Health:
    return Health(
        status="ok",
        log_size=c.log.size,
        model_version=c.model_version,
        investigator=c.settings.investigator_model,
    )


@router.get("/metrics", tags=["system"], response_class=Response)
async def metrics(c: ContainerDep) -> Response:
    c.metrics.log_size.set(c.log.size)
    return Response(generate_latest(c.registry), media_type=CONTENT_TYPE_LATEST)


@router.get("/", include_in_schema=False, response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX.read_text(encoding="utf-8"), headers=PAGE_HEADERS)
