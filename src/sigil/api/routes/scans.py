from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from sigil.api.deps import ContainerDep
from sigil.api.schemas import ScanDetail
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
) -> VerificationResult:
    """Verify a photo against the credential, the copy-detection pattern and the scan history."""
    if (lat is None) != (lon is None):
        raise HTTPException(422, "lat and lon must be given together")
    context = ScanContext(datetime.now(UTC), lat, lon, device_id)
    return await c.scan(await image.read(), context)


@router.get("/scans/{scan_id}")
async def get_scan(scan_id: str, c: ContainerDep) -> ScanDetail:
    event, payload = c.scans.get(scan_id)
    if payload is None:
        raise NotFoundError(f"scan {scan_id} has no stored result")
    return ScanDetail(
        result=VerificationResult.model_validate(payload),
        final_verdict=event.verdict,
        scanned_at=event.ts,
    )
