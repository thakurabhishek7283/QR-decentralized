"""Dependency container: builds every service once and owns their lifecycle.

Routes stay thin because everything they need hangs off the container. The same container backs
the CLI, so ``sigil scan`` and ``POST /v1/scans`` run exactly the same code.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Self

import numpy as np
from prometheus_client import CollectorRegistry

from sigil.api.investigations import InvestigationStore, summarize_outcome
from sigil.api.metrics import ApiMetrics
from sigil.api.schemas import ScanResponse, SimulateRequest
from sigil.api.simulation import Photo, SimulationStudio
from sigil.config import Settings
from sigil.crypto import FileKeyStore, KeyRecord
from sigil.errors import SigilError
from sigil.harness import Model
from sigil.inference import BatcherConfig, BatcherMetrics, InferenceService, OnnxCdpScorer
from sigil.investigator import InvestigationDeps, build_model, investigate
from sigil.issuance import BatchInfo, IssuanceStore, Issuer, PrintQuality, run_qc
from sigil.ledger import TransparencyLog
from sigil.scans import ScanStore
from sigil.storage import Database
from sigil.verify import PolicyConfig, ScanContext, VerificationPipeline
from sigil.vision import label_png, load_image

log = logging.getLogger("sigil.api")

SHUTDOWN_GRACE_S = 10.0  # how long running investigations may finish when the server stops
MAX_CONCURRENT_INVESTIGATIONS = 4  # bounds model spend and load under a burst of escalations
DEMO_BATCHES: tuple[tuple[str, int, PrintQuality], ...] = (
    ("ACME-WIDGET-ECO", 6, "worn"),  # a tired press: its labels score lower across the batch
    ("ACME-WIDGET-PRO", 12, "good"),  # issued last, so the demo page opens on it
)


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)

        self.keystore = FileKeyStore(settings.keys_dir)
        self.keys = self.keystore.load_or_create()
        self.db = Database(settings.db_path)
        self.log = TransparencyLog(self.db, self.keys.log)
        self.issuance = IssuanceStore(self.db)
        self.issuer = Issuer(self.keys, self.log, self.issuance)
        self.scans = ScanStore(self.db)
        self.investigations = InvestigationStore(self.db)

        self.registry = CollectorRegistry()
        self.metrics = ApiMetrics(self.registry)
        scorer = OnnxCdpScorer(threads=settings.inference_threads)
        self.model_version: str | None = scorer.metadata.get("version")
        self.inference = InferenceService(
            scorer,
            BatcherConfig(
                settings.batch_max_size, settings.batch_max_delay_ms, settings.batch_queue_limit
            ),
            BatcherMetrics(self.registry),
        )
        self.pipeline = VerificationPipeline(
            keys=self.keys.keyring,
            cdp_key=self.keys.cdp_key,
            log=self.log,
            issuance=self.issuance,
            scans=self.scans,
            inference=self.inference,
            policy=PolicyConfig.from_settings(settings),
            # Derived, not stored: device ids stay pseudonymous without one more secret on disk.
            device_salt=hashlib.sha256(b"sigil-device-salt" + self.keys.cdp_key).digest(),
        )
        self.investigator: Model = build_model(settings)
        self.studio = SimulationStudio()
        self._tasks: set[asyncio.Task[None]] = set()
        self._slots = asyncio.Semaphore(MAX_CONCURRENT_INVESTIGATIONS)

    # --- lifecycle -------------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: Settings) -> AsyncIterator[Self]:
        container = cls(settings)
        await container.start()
        try:
            yield container
        finally:
            await container.close()

    async def start(self) -> None:
        await self.inference.start()
        if stale := self.investigations.fail_pending("interrupted by a server restart"):
            log.warning("marked %d unfinished investigations as failed", stale)

    async def close(self) -> None:
        """Let running investigations finish briefly, cancel the rest, then stop the services."""
        if self._tasks:
            _, running = await asyncio.wait(self._tasks, timeout=SHUTDOWN_GRACE_S)
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
        await self.inference.stop()
        self.db.close()

    async def seed_demo(self) -> None:
        """On a fresh data directory, issue the demo batches so the web demo has labels to scan."""
        if self.issuance.batches():
            return
        for sku, count, quality in DEMO_BATCHES:
            await self.create_batch(sku, count, quality)

    # --- issuance --------------------------------------------------------------------------

    async def create_batch(self, sku: str, count: int, print_quality: PrintQuality) -> BatchInfo:
        """Issue and log a batch, then measure its print quality at the (simulated) QC camera."""
        info, _ = await asyncio.to_thread(
            self.issuer.issue_batch, sku, count, print_quality=print_quality
        )
        samples = min(self.settings.qc_samples_per_batch, count)
        try:
            await run_qc(
                self.issuer,
                self.inference,
                info.batch_id,
                samples=samples,
                rng=np.random.default_rng(),
            )
        except SigilError as exc:  # the batch is valid and logged; it just has no baseline
            log.warning("QC failed for batch %s: %s", info.batch_id, exc)
        return self.issuance.batch(info.batch_id)

    def revoke_key(self, kid: bytes, at: datetime) -> KeyRecord:
        record = self.keys.keyring.revoke(kid, at)
        self.keystore.save_keyring(self.keys.keyring)
        return record

    async def label_png(self, serial_hex: str) -> bytes:
        return await asyncio.to_thread(lambda: label_png(self.issuer.render_label(serial_hex)))

    # --- the demo camera -------------------------------------------------------------------

    async def photograph(self, serial_hex: str, request: SimulateRequest) -> Photo:
        _, batch_id = self.issuance.unit(serial_hex)  # NotFoundError for an unknown serial
        quality = self.issuance.batch(batch_id).print_quality
        label = await asyncio.to_thread(self.issuer.render_label, serial_hex)
        return await asyncio.to_thread(self.studio.shoot, label, quality, request)

    # --- scanning --------------------------------------------------------------------------

    async def scan(
        self,
        data: bytes,
        context: ScanContext,
        *,
        investigate: bool = True,
        item: str | None = None,
    ) -> ScanResponse:
        """Verify an uploaded photo and, if the policy cannot settle it, start an investigation.

        ``item`` names a simulated print whose further photos may be used for recaptures.
        """
        started = time.perf_counter()
        image = await asyncio.to_thread(load_image, data)
        result = await self.pipeline.verify(image, context)
        self.metrics.scans.labels(result.verdict.value).inc()
        self.metrics.scan_seconds.observe(time.perf_counter() - started)

        investigation_id = None
        if investigate and result.escalate and self.settings.investigation_enabled:
            camera = self.studio.camera_for(item)
            investigation_id = self._start_investigation(
                InvestigationDeps(self.pipeline, result, context, camera)
            )
        return ScanResponse(**dict(result), investigation_id=investigation_id)

    # --- investigations --------------------------------------------------------------------

    def _start_investigation(self, deps: InvestigationDeps) -> str:
        investigation_id = uuid.uuid4().hex
        self.investigations.create(investigation_id, deps.result.scan_id)
        self._spawn(self._investigate(investigation_id, deps))
        return investigation_id

    async def _investigate(self, investigation_id: str, deps: InvestigationDeps) -> None:
        async with self._slots:
            try:
                outcome = await investigate(
                    deps,
                    model=self.investigator,
                    settings=self.settings,
                    trace_dir=self.settings.traces_dir,
                    run_id=investigation_id,
                )
                view = summarize_outcome(outcome, self.settings.investigator_model)
                self.scans.update_verdict(deps.result.scan_id, view.final_verdict)
                self.investigations.finish(investigation_id, view)
                self.metrics.observe_investigation(view)
            except asyncio.CancelledError:
                self.investigations.fail(investigation_id, "cancelled by server shutdown")
                raise
            except Exception as exc:
                log.exception("investigation %s crashed", investigation_id)
                self.investigations.fail(investigation_id, f"{type(exc).__name__}: {exc}")
                self.metrics.investigation_errors.inc()

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)  # a strong reference: the loop only keeps weak ones
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for background investigations to finish (used by the CLI and tests)."""
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
