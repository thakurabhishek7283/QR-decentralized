"""Request and response models for the HTTP API.

Domain models (``VerificationResult``, ``Evidence``) are reused as they are; only shapes that exist
purely at the API boundary are defined here.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from sigil.crypto import KeyRecord
from sigil.issuance import BatchInfo, QcStats
from sigil.verify import VerificationResult

# Credentials carry the SKU as 1-32 printable ASCII characters (LLD 3.3).
Sku = Annotated[str, StringConstraints(pattern=r"^[\x20-\x7E]{1,32}$")]
SerialHex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{24}$")]


class Health(BaseModel):
    status: Literal["ok"]
    log_size: int
    model_version: str | None


class KeyInfo(BaseModel):
    kid: str
    purpose: str
    public_key: str  # base64url, unpadded
    created_at: datetime
    revoked_at: datetime | None

    @classmethod
    def of(cls, record: KeyRecord) -> KeyInfo:
        return cls(
            kid=record.kid_hex(),
            purpose=record.purpose,
            public_key=base64.urlsafe_b64encode(record.public_raw).rstrip(b"=").decode("ascii"),
            created_at=record.created_at,
            revoked_at=record.revoked_at,
        )


class RevokeRequest(BaseModel):
    at: datetime | None = Field(default=None, description="Revocation time (default: now).")


class BatchRequest(BaseModel):
    sku: Sku
    count: int = Field(ge=1, le=200)
    print_quality: Literal["good", "worn"] = "good"


class QcOut(BaseModel):
    n: int
    mean: float
    std: float
    min: float

    @classmethod
    def of(cls, qc: QcStats) -> QcOut:
        return cls(n=qc.n, mean=qc.mean, std=qc.std, min=qc.min)


class BatchOut(BaseModel):
    batch_id: str
    sku: str
    created_at: datetime
    print_quality: str
    qc: QcOut | None
    serials: list[str]

    @classmethod
    def of(cls, info: BatchInfo, serials: list[str]) -> BatchOut:
        return cls(
            batch_id=info.batch_id,
            sku=info.sku,
            created_at=info.created_at,
            print_quality=info.print_quality,
            qc=QcOut.of(info.qc) if info.qc else None,
            serials=serials,
        )


class BatchSummary(BaseModel):
    batch_id: str
    sku: str
    created_at: datetime
    print_quality: str
    units: int


class UnitOut(BaseModel):
    serial: str
    batch_id: str
    payload: str  # the exact text in the label's QR code; its decoded bytes are the log leaf
    leaf_index: int


class SimulateRequest(BaseModel):
    attack: Literal["photocopy", "hq_reprint", "screen"] | None = None
    difficulty: Literal["easy", "normal", "hard"] = "normal"
    seed: int | None = Field(default=None, ge=0, description="Makes the photo reproducible.")


class ScanDetail(BaseModel):
    result: VerificationResult  # as computed when the photo was scanned
    final_verdict: str | None
    scanned_at: datetime
