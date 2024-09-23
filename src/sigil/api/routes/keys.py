from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Body, Path

from sigil.api.deps import ContainerDep, RequireApiKey
from sigil.api.schemas import KeyInfo, RevokeRequest

router = APIRouter(tags=["keys"])


@router.get("/keys")
async def list_keys(c: ContainerDep) -> list[KeyInfo]:
    """The public keys a verifier should trust, with revocation status."""
    return [KeyInfo.of(record) for record in c.keys.keyring.records()]


@router.post("/keys/{kid}/revoke", dependencies=[RequireApiKey])
async def revoke_key(
    kid: Annotated[str, Path(pattern=r"^[0-9a-f]{8}$")],
    c: ContainerDep,
    body: Annotated[RevokeRequest, Body(default_factory=RevokeRequest)],
) -> KeyInfo:
    """Revoke a key from ``at`` (default now). Credentials issued before ``at`` become suspect."""
    at = body.at or datetime.now(UTC)
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return KeyInfo.of(c.revoke_key(bytes.fromhex(kid), at))
