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
    """A simulated phone photo of a label: the genuine print, or a counterfeit of it (demo only).

    The ``X-Sigil-Item`` header names the photographed print; pass it as ``item`` when scanning so
    that a recapture request is answered with another photo of the same print.
    """
    photo = await c.photograph(serial, body)
    return Response(
        photo.jpeg,
        media_type="image/jpeg",
        headers={"X-Sigil-Item": photo.item_id, "Cache-Control": "no-store"},
    )
