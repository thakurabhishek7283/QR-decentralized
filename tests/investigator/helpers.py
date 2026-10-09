"""Builders for investigator tests: hand-made verification results and a scripted camera.

Results are assembled from evidence rather than photographed, so a test states exactly the scores
and clone risk the investigator should face. Recapture photos are one-pixel arrays that carry the
score the stub pipeline should report for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from sigil.config import Settings
from sigil.harness import ToolCall, ToolContext, ToolRegistry, ToolResult
from sigil.inference import CdpScore, InferenceService
from sigil.investigator import InvestigationDeps, build_registry
from sigil.issuance import QcStats
from sigil.ledger import leaf_hash
from sigil.scans import ScanEvent
from sigil.verify import (
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    ScanContext,
    VerificationPipeline,
    VerificationResult,
    cdp_evidence,
    check_log,
    decide,
)
from tests.support import World

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
HERE = (19.08, 72.88)  # Mumbai
FAR = (51.51, -0.13)  # London


class ScoredPipeline(VerificationPipeline):
    """A pipeline whose ``measure_cdp`` reads the score from the photo's only pixel."""

    async def measure_cdp(self, image: np.ndarray | bytes, serial: bytes) -> Evidence:
        assert isinstance(image, np.ndarray)
        score = CdpScore(float(image[0, 0]), 0.5, 0.5, 0.9)
        return cdp_evidence(score, self.policy, "recapture", EvidenceKind.RECAPTURE)


class ScriptedCamera:
    """Yields photos with preset scores, then declines."""

    def __init__(self, *scores: float) -> None:
        self._scores = list(scores)
        self.requests = 0

    async def next_capture(self) -> np.ndarray | None:
        self.requests += 1
        return np.array([[self._scores.pop(0)]]) if self._scores else None


@dataclass
class Case:
    """One escalated scan, ready to investigate."""

    pipeline: ScoredPipeline
    result: VerificationResult
    context: ScanContext
    camera_scores: tuple[float, ...] = ()
    cameras: list[ScriptedCamera] = field(default_factory=list)

    def deps(self, *, camera: bool = True) -> InvestigationDeps:
        """Fresh deps (and a fresh camera), as a replay or a second run would get."""
        source = None
        if camera:
            source = ScriptedCamera(*self.camera_scores)
            self.cameras.append(source)
        return InvestigationDeps(self.pipeline, self.result, self.context, source)


def add_recapture(deps: InvestigationDeps, score: float, number: int = 1) -> None:
    """Put a recapture measurement straight into an investigation's working bundle."""
    cdp = CdpScore(score, score, score, 0.9)
    recapture = cdp_evidence(cdp, deps.policy, f"recapture-{number}", EvidenceKind.RECAPTURE)
    deps.bundle.add(recapture)


def make_case(
    world: World,
    inference: InferenceService,
    *,
    cnn: float,
    clone_history: bool = False,
    qc: tuple[float, float] | None = (0.9, 0.05),
    camera: tuple[float, ...] = (),
    unit: int = 0,
    sku: str | None = None,
    revoked: str | None = None,
) -> Case:
    """An escalated scan of ``world.units[unit]`` with the given primary CDP score.

    ``clone_history`` seeds a scan of the same serial in London three hours before this scan in
    Mumbai, which is physically impossible and drives clone risk to 0.8.
    """
    pipeline = ScoredPipeline(
        keys=world.keys.keyring,
        cdp_key=world.keys.cdp_key,
        log=world.log,
        issuance=world.store,
        scans=world.scans,
        inference=inference,
        policy=world.policy,
        device_salt=b"test-salt",
    )
    serial_hex = world.units[unit].serial_hex
    credential, batch_id = world.store.unit(serial_hex)
    if qc is not None:
        world.store.set_qc(batch_id, QcStats(5, qc[0], qc[1], qc[0] - qc[1]))
    context = ScanContext(NOW, *HERE, device_id="phone-1")

    risk = 0.0
    if clone_history:
        seeded = ScanEvent(
            f"prior-{unit}", serial_hex, NOW - timedelta(hours=3), *FAR, "devx", "GENUINE", 0.9
        )
        world.scans.add(seeded)
        risk = 0.8

    key: dict[str, Any] = {"revoked": revoked is not None, "issued_before_revocation": True}
    items = [
        Evidence(id="quality", kind=EvidenceKind.QUALITY, passed=True, summary="ok", data={}),
        Evidence(
            id="decode",
            kind=EvidenceKind.DECODE,
            passed=True,
            summary="decoded",
            data={
                "reason": None,
                "serial": serial_hex,
                "untrusted": {
                    "sku": sku or credential.credential.sku,
                    "batch": credential.credential.batch,
                },
            },
        ),
        Evidence(id="signature", kind=EvidenceKind.SIGNATURE, passed=True, summary="ok", data={}),
        Evidence(
            id="key", kind=EvidenceKind.KEY_STATUS, passed=revoked is None, summary="k", data=key
        ),
        check_log(world.log, world.keys.keyring, leaf_hash(credential.leaf_data())),
        cdp_evidence(CdpScore(cnn, cnn, cnn, 0.9), world.policy),
        Evidence(
            id="clone",
            kind=EvidenceKind.CLONE,
            passed=None if risk else True,
            summary="risk",
            data={"risk": risk, "signals": [], "scans_considered": 1},
        ),
    ]
    bundle = EvidenceBundle(scan_id=f"scan-{unit}", serial_hex=serial_hex, items=items)
    decision = decide(bundle, world.policy)
    assert decision.escalate, f"{decision.rule} would not reach the investigator"
    result = VerificationResult(
        scan_id=bundle.scan_id,
        verdict=decision.verdict,
        confidence=decision.confidence,
        rule=decision.rule,
        reasons=decision.reasons,
        escalate=True,
        serial_hex=serial_hex,
        sku=sku or credential.credential.sku,
        batch_id=batch_id,
        evidence=bundle,
        timings_ms={},
    )
    return Case(pipeline, result, context, camera)


# --- driving tools directly --------------------------------------------------------------------


def tool_context(deps: InvestigationDeps) -> ToolContext:
    return ToolContext(run_id="t", step=1, state={}, deps=deps)


async def call(
    registry: ToolRegistry, ctx: ToolContext, name: str, call_id: str = "c1", **args: Any
) -> ToolResult:
    return await registry.execute(ToolCall(call_id, name, args), ctx)


def registry() -> ToolRegistry:
    return build_registry()


def settings(**overrides: Any) -> Settings:
    values = {"investigator_model": "reference", **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]
