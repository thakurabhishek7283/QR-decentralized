"""HTTP routes, grouped by resource. Handlers translate HTTP to container calls and back."""

from fastapi import APIRouter

from sigil.api.routes import batches, investigations, keys, log, scans, simulate

v1 = APIRouter(prefix="/v1")
for module in (keys, batches, simulate, scans, investigations, log):
    v1.include_router(module.router)

__all__ = ["v1"]
