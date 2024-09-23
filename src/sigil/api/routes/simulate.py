from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Response

from sigil.api.deps import ContainerDep
from sigil.api.schemas import SerialHex, SimulateRequest

router = APIRouter(tags=["demo"])


@router.post("/simulate/{serial}", response_class=Response)
async def simulate_photo(
    serial: SerialHex,
    c: ContainerDep,
    body: Annotated[SimulateRequest, Body(default_factory=SimulateRequest)],
) -> Response:
    """A simulated phone photo of a label: the genuine print, or a counterfeit of it (demo only)."""
    photo = await c.photograph(serial, body)
    return Response(photo.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
