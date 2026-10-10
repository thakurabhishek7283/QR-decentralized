from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from sigil.api.deps import ContainerDep
from sigil.api.schemas import InvestigationRef, ScanDetail, ScanResponse
from sigil.errors import NotFoundError
from sigil.verify import ScanContext, VerificationResult

router = APIRouter(tags=["scans"])


@router.post("/scans")
async def create_scan(
    image: Annotated[UploadFile, File(description="A photo of the label (JPEG or PNG).")],
    c: ContainerDep,
    lat: Annotated[float | None, Form(ge=-90, le=90)] = None,
    lon: Annotated[float | None, Form(ge=-180, le=180)] = None,
    device_id: Annotated[str | None, Form(max_length=128)] = None,
    investigate: Annotated[bool, Query(description="Investigate if the policy is unsure.")] = True,
    item: Annotated[
        str | None, Query(max_length=32, description="Demo: simulated print id.")
    ] = None,
) -> ScanResponse:
    """Verify a photo. An unsure verdict is investigated in the background; poll the returned
    ``investigation_id`` at ``/v1/investigations/{id}``."""
    if (lat is None) != (lon is None):
        raise HTTPException(422, "lat and lon must be given together")
    context = ScanContext(datetime.now(UTC), lat, lon, device_id)
    return await c.scan(await image.read(), context, investigate=investigate, item=item)


@router.get("/scans/{scan_id}")
async def get_scan(scan_id: str, c: ContainerDep) -> ScanDetail:
    event, payload = c.scans.get(scan_id)
    if payload is None:
        raise NotFoundError(f"scan {scan_id} has no stored result")
    row = c.investigations.for_scan(scan_id)
    return ScanDetail(
        result=VerificationResult.model_validate(payload),
        final_verdict=event.verdict,
        scanned_at=event.ts,
        investigation=InvestigationRef(id=row.id, status=row.status, outcome=row.outcome)
        if row
        else None,
    )
