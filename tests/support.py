"""Shared builders for the verification, investigator and eval tests.

The CDP model is replaced by a stub that returns a fixed score, so tests control exactly what the
verification layer sees while the image pipeline (QR decode, rectification, ECC alignment) is real.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from prometheus_client import CollectorRegistry

from sigil.cdp import Attack, generate_template, simulate_capture
from sigil.crypto import FileKeyStore, KeyMaterial
from sigil.errors import CaptureError
from sigil.inference import BatcherConfig, BatcherMetrics, InferenceService
from sigil.issuance import IssuanceStore, IssuedUnit, Issuer
from sigil.ledger import TransparencyLog
from sigil.scans import ScanStore
from sigil.storage import Database
from sigil.verify import PolicyConfig, VerificationPipeline
from sigil.vision import decode, extract_cdp_patch, render_label


class StubScorer:
    """A ``CdpScorer`` that returns the same logit for everything."""

    def __init__(self, logit: float = 4.0) -> None:
        self.logit = logit

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        return np.full(len(batch), self.logit, dtype=np.float32)


@dataclass
class World:
    """A small deployment: keys, log, stores and one issued batch."""

    keys: KeyMaterial
    db: Database
    log: TransparencyLog
    store: IssuanceStore
    scans: ScanStore
    issuer: Issuer
    batch_id: str
    units: list[IssuedUnit]
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def pipeline(self, inference: InferenceService) -> VerificationPipeline:
        return VerificationPipeline(
            keys=self.keys.keyring,
            cdp_key=self.keys.cdp_key,
            log=self.log,
            issuance=self.store,
            scans=self.scans,
            inference=inference,
            policy=self.policy,
            device_salt=b"test-salt",
        )


def build_world(root: Path, *, units: int = 6, sku: str = "TEST-SKU") -> World:
    keys = FileKeyStore(root / "keys").load_or_create()
    db = Database(":memory:")
    log = TransparencyLog(db, keys.log)
    store = IssuanceStore(db)
    issuer = Issuer(keys, log, store)
    info, issued = issuer.issue_batch(sku, units)
    return World(keys, db, log, store, ScanStore(db), issuer, info.batch_id, issued)


@asynccontextmanager
async def running_inference(logit: float = 4.0) -> AsyncIterator[InferenceService]:
    service = InferenceService(
        StubScorer(logit), BatcherConfig(max_delay_ms=1.0), BatcherMetrics(CollectorRegistry())
    )
    await service.start()
    try:
        yield service
    finally:
        await service.stop()


def photograph(
    world: World,
    serial_hex: str,
    *,
    attack: Attack | None = None,
    difficulty: str = "easy",
    seed: int = 0,
) -> np.ndarray:
    """A phone photo of an issued label that the vision layer can read and align.

    The simulator occasionally produces unreadable shots, as real cameras do, so this retakes
    the photo with the next seed until one works.
    """
    template = generate_template(world.keys.cdp_key, bytes.fromhex(serial_hex))
    label = world.issuer.render_label(serial_hex)
    for attempt in range(seed, seed + 20):
        rng = np.random.default_rng(attempt)
        photo = simulate_capture(label, rng, difficulty=difficulty, attack=attack)  # type: ignore[arg-type]
        try:
            cap = decode(photo)
            if cap.quality.ok:
                extract_cdp_patch(cap, template)
                return photo
        except CaptureError:
            continue
    raise AssertionError("could not produce a readable photo")  # pragma: no cover


def photograph_payload(world: World, payload: str, seed: int = 0) -> np.ndarray:
    """A photo of an arbitrary QR payload printed beside a random CDP (for forged labels)."""
    rng = np.random.default_rng(seed)
    template = (rng.random((106, 106)) < 0.5).astype(np.uint8)
    label = render_label(payload, template)
    for attempt in range(seed, seed + 20):
        photo = simulate_capture(label, np.random.default_rng(attempt), difficulty="easy")
        try:
            if decode(photo).quality.ok:
                return photo
        except CaptureError:
            continue
    raise AssertionError("could not produce a readable photo")  # pragma: no cover
