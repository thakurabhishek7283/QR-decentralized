from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from sigil.api.deps import ContainerDep, RequireApiKey
from sigil.api.schemas import BatchOut, BatchRequest, BatchSummary, SerialHex, UnitOut
from sigil.errors import NotFoundError
from sigil.ledger import leaf_hash

router = APIRouter(tags=["issuance"])


@router.post("/batches", status_code=201, dependencies=[RequireApiKey])
async def create_batch(body: BatchRequest, c: ContainerDep) -> BatchOut:
    """Issue ``count`` signed, logged labels and sample the batch's print quality."""
    info = await c.create_batch(body.sku, body.count, body.print_quality)
    return BatchOut.of(info, c.issuance.units(info.batch_id))


@router.get("/batches")
async def list_batches(
    c: ContainerDep, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[BatchSummary]:
    """Recent batches, newest first."""
    recent = reversed(c.issuance.batches()[-limit:])
    return [
        BatchSummary(
            batch_id=b.batch_id,
            sku=b.sku,
            created_at=b.created_at,
            print_quality=b.print_quality,
            units=len(c.issuance.units(b.batch_id)),
        )
        for b in recent
    ]


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str, c: ContainerDep) -> BatchOut:
    return BatchOut.of(c.issuance.batch(batch_id), c.issuance.units(batch_id))


@router.get("/units/{serial}")
async def get_unit(serial: SerialHex, c: ContainerDep) -> UnitOut:
    """What a label's QR code says, and where its credential sits in the transparency log."""
    credential, batch_id = c.issuance.unit(serial)
    index = c.log.index_of(leaf_hash(credential.leaf_data()))
    if index is None:
        raise NotFoundError(f"serial {serial} is not in the transparency log")
    return UnitOut(
        serial=serial, batch_id=batch_id, payload=credential.to_qr_payload(), leaf_index=index
    )


@router.get("/labels/{serial}.png", response_class=Response)
async def get_label(serial: SerialHex, c: ContainerDep) -> Response:
    """The printable label. (Demo only: a real deployment never serves the CDP.)"""
    png = await c.label_png(serial)
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
