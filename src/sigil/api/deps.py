"""FastAPI dependencies: the container and API-key authentication."""

from __future__ import annotations

import hashlib
import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from sigil.api.container import Container


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def require_api_key(
    container: ContainerDep, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    """Gate issuer endpoints on ``X-API-Key``.

    Both sides are hashed first so the comparison is constant-time even across different lengths.
    """
    expected = hashlib.sha256(container.settings.api_key.encode()).digest()
    supplied = hashlib.sha256((x_api_key or "").encode()).digest()
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            401, "missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"}
        )


RequireApiKey = Depends(require_api_key)
