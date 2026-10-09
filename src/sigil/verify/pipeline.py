"""The verification pipeline: photo in, evidence bundle and verdict out.

Checks run in dependency order. Decoding comes first; the signature and key checks need the parsed
credential; the log lookup, CDP scoring and clone analytics are independent of each other and run
concurrently. A failure skips only what depends on it (an unreadable photo has no credential to
check) so the bundle always lists every check, either run or marked not-run.

CPU-bound vision work (QR detection, ECC alignment: tens of milliseconds) runs in worker threads
so the event loop stays responsive for other scans.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel

from sigil.cdp import generate_template
from sigil.crypto import KeyRecord, KeyRing, SignedCredential
from sigil.errors import CaptureError, DecodeError, LogError, SignatureError, UnknownKeyError
from sigil.inference import CdpScore, InferenceService
from sigil.issuance import IssuanceStore
from sigil.ledger import TransparencyLog, leaf_hash, verify_inclusion
from sigil.scans import CloneConfig, ScanEvent, ScanStore, assess_clone_risk, hash_device
from sigil.verify.evidence import Evidence, EvidenceBundle, EvidenceKind
from sigil.verify.policy import PolicyConfig, Verdict, decide
from sigil.vision import DecodedCapture, QualityReport, decode, extract_cdp_patch, load_image

NO_CREDENTIAL = "no readable credential"


@dataclass(frozen=True)
class ScanContext:
    ts: datetime
    lat: float | None = None
    lon: float | None = None
    device_id: str | None = None

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError("ScanContext.ts must be timezone-aware")


class VerificationResult(BaseModel):
    scan_id: str
    verdict: Verdict
    confidence: float
    rule: str
    reasons: list[str]
    escalate: bool
    serial_hex: str | None
    sku: str | None  # untrusted display string read from the label
    batch_id: str | None  # only set once the signature has verified
    evidence: EvidenceBundle
    timings_ms: dict[str, float]  # decode, crypto, log, cdp, clone, total


class VerificationPipeline:
    def __init__(
        self,
        *,
        keys: KeyRing,
        cdp_key: bytes,
        log: TransparencyLog,
        issuance: IssuanceStore,
        scans: ScanStore,
        inference: InferenceService,
        policy: PolicyConfig,
        device_salt: bytes,
        clone_config: CloneConfig | None = None,
    ) -> None:
        self.keys = keys
        self.log = log
        self.issuance = issuance
        self.scans = scans
        self.policy = policy
        self._cdp_key = cdp_key
        self._inference = inference
        self._device_salt = device_salt
        self._clone_config = clone_config or CloneConfig()

    async def verify(
        self, image: np.ndarray | bytes, ctx: ScanContext, *, record: bool = True
    ) -> VerificationResult:
        """Run every check on a photo and apply the policy.

        ``record=False`` leaves no trace in the scan store (used for dry runs).
        """
        started = time.perf_counter()
        scan_id = uuid.uuid4().hex
        timings: dict[str, float] = {}

        cap: DecodedCapture | None = None
        sc: SignedCredential | None = None
        failure: str | None = None
        try:
            with _timed(timings, "decode"):
                cap = await asyncio.to_thread(_read, image)
            sc = SignedCredential.from_qr_payload(cap.payload)
        except CaptureError as exc:
            failure = exc.reason
        except DecodeError:
            failure = "malformed_credential"

        trusted = False
        if cap is None or sc is None:
            items = _unreadable(cap, failure or "unknown")
        else:
            items, trusted = await self._check(cap, sc, ctx, scan_id, timings)

        serial_hex = sc.credential.serial_hex if sc else None
        bundle = EvidenceBundle(scan_id=scan_id, serial_hex=serial_hex, items=items)
        decision = decide(bundle, self.policy)
        timings["total"] = _ms(time.perf_counter() - started)
        result = VerificationResult(
            scan_id=scan_id,
            verdict=decision.verdict,
            confidence=decision.confidence,
            rule=decision.rule,
            reasons=decision.reasons,
            escalate=decision.escalate,
            serial_hex=serial_hex,
            sku=sc.credential.sku if sc else None,
            batch_id=sc.credential.batch if sc and trusted else None,
            evidence=bundle,
            timings_ms={k: round(v, 3) for k, v in timings.items()},
        )
        if record:
            self._record(result, ctx)
        return result

    async def measure_cdp(self, image: np.ndarray | bytes, serial: bytes) -> Evidence:
        """Score a further photo of the item with the given serial (the recapture tool).

        Never raises for a bad photo: the evidence says why the measurement is unusable. A photo
        of a label with a different serial is rejected, so a second item cannot stand in.
        """
        recapture = EvidenceKind.RECAPTURE
        try:
            cap = await asyncio.to_thread(_read, image)
            sc = SignedCredential.from_qr_payload(cap.payload)
        except CaptureError as exc:
            return _cdp_error("recapture", recapture, exc.reason)
        except DecodeError:
            return _cdp_error("recapture", recapture, "malformed_credential")
        if sc.credential.serial != serial:
            return _cdp_error("recapture", recapture, "serial_mismatch")
        return await self._score_cdp(cap, serial, "recapture", recapture)

    # --- checks --------------------------------------------------------------------------------

    async def _check(
        self,
        cap: DecodedCapture,
        sc: SignedCredential,
        ctx: ScanContext,
        scan_id: str,
        timings: dict[str, float],
    ) -> tuple[list[Evidence], bool]:
        with _timed(timings, "crypto"):
            signature, key = self._check_signature(sc)
        trusted = signature.passed is True

        async def log_check() -> Evidence:
            with _timed(timings, "log"):
                return await asyncio.to_thread(
                    check_log, self.log, self.keys, leaf_hash(sc.leaf_data()), "log"
                )

        async def cdp_check() -> Evidence:
            if not trusted:
                return Evidence.not_run("cdp", EvidenceKind.CDP, "the signature did not verify")
            with _timed(timings, "cdp"):
                return await self._score_cdp(cap, sc.credential.serial, "cdp", EvidenceKind.CDP)

        async def clone_check() -> Evidence:
            if not trusted:
                return Evidence.not_run("clone", EvidenceKind.CLONE, "the signature did not verify")
            with _timed(timings, "clone"):
                return self._check_clone(sc.credential.serial_hex, ctx, scan_id)

        log_ev, cdp_ev, clone_ev = await asyncio.gather(log_check(), cdp_check(), clone_check())
        items = [
            _quality_evidence(cap.quality),
            _decode_evidence(sc),
            signature,
            key,
            log_ev,
            cdp_ev,
            clone_ev,
        ]
        return items, trusted

    def _check_signature(self, sc: SignedCredential) -> tuple[Evidence, Evidence]:
        """The signature and key-status evidence; only an *issuer* key may sign credentials."""
        kid = sc.credential.kid.hex()
        failure = ""
        record: KeyRecord | None = None
        try:
            record = sc.verify(self.keys)
            if record.purpose != "issuer":
                failure = "wrong_key_purpose"
        except UnknownKeyError:
            failure = "unknown_key"
        except SignatureError:
            failure = "bad_signature"
        if failure:
            signature = Evidence(
                id="signature",
                kind=EvidenceKind.SIGNATURE,
                passed=False,
                summary=f"signature rejected: {failure.replace('_', ' ')}",
                data={"kid": kid, "reason": failure},
            )
            return signature, Evidence.not_run("key", EvidenceKind.KEY_STATUS, "bad signature")
        assert record is not None
        signature = Evidence(
            id="signature",
            kind=EvidenceKind.SIGNATURE,
            passed=True,
            summary="Ed25519 signature verifies under a known issuer key",
            data={"kid": kid},
        )
        return signature, _key_evidence(record, sc.credential.iat)

    async def _score_cdp(
        self, cap: DecodedCapture, serial: bytes, evidence_id: str, kind: EvidenceKind
    ) -> Evidence:
        template = generate_template(self._cdp_key, serial)
        try:
            patch = await asyncio.to_thread(extract_cdp_patch, cap, template)
        except CaptureError as exc:
            return _cdp_error(evidence_id, kind, exc.reason)
        score = await self._inference.score(patch, template)
        return cdp_evidence(score, self.policy, evidence_id, kind)

    def _check_clone(self, serial_hex: str, ctx: ScanContext, scan_id: str) -> Evidence:
        current = ScanEvent(
            scan_id, serial_hex, ctx.ts, ctx.lat, ctx.lon, self._device_hash(ctx.device_id)
        )
        history = self.scans.history(serial_hex, before=ctx.ts, within=self._clone_config.window)
        assessment = assess_clone_risk(history, current, self._clone_config)
        risk = assessment.risk
        passed: bool | None = None
        if risk >= self.policy.clone_risk_high:
            passed = False
        elif risk < self.policy.clone_risk_low:
            passed = True
        return Evidence(
            id="clone",
            kind=EvidenceKind.CLONE,
            passed=passed,
            summary=(
                f"clone risk {risk:.2f} from {assessment.scans_considered} scans "
                f"({len(assessment.signals)} signals)"
            ),
            data={
                "risk": risk,
                "scans_considered": assessment.scans_considered,
                "signals": [
                    {"name": s.name, "weight": s.weight, "detail": s.detail}
                    for s in assessment.signals
                ],
            },
        )

    def _device_hash(self, device_id: str | None) -> str | None:
        return hash_device(device_id, self._device_salt) if device_id else None

    def _record(self, result: VerificationResult, ctx: ScanContext) -> None:
        cdp = result.evidence.get("cdp")
        cnn = cdp.data.get("cnn") if cdp else None
        self.scans.add(
            ScanEvent(
                id=result.scan_id,
                serial_hex=result.serial_hex,
                ts=ctx.ts,
                lat=ctx.lat,
                lon=ctx.lon,
                device_hash=self._device_hash(ctx.device_id),
                verdict=result.verdict.value,
                cdp_score=cnn,
            ),
            payload=result.model_dump(mode="json"),
        )


# --- evidence builders -------------------------------------------------------------------------


def check_log(
    log: TransparencyLog, keys: KeyRing, leaf: bytes, evidence_id: str = "log"
) -> Evidence:
    """Look a leaf up in the log and verify its inclusion proof against a signed tree head.

    Used by the pipeline and, with a fresh proof, by the investigator's ``verify_log_proof``.
    """
    kind = EvidenceKind.LOG_INCLUSION
    if log.index_of(leaf) is None:
        return Evidence(
            id=evidence_id,
            kind=kind,
            passed=False,
            summary="credential is not in the transparency log",
            data={"in_log": False, "leaf_hash": leaf.hex()},
        )
    try:
        sth = log.latest_sth()
        proof = log.prove_inclusion(leaf, sth.tree_size)
        verified = sth.verify(keys.get(sth.log_kid).public_raw) and verify_inclusion(
            leaf, proof.leaf_index, proof.tree_size, proof.path, sth.root_hash
        )
    except (LogError, UnknownKeyError):
        return Evidence(
            id=evidence_id,
            kind=kind,
            passed=False,
            summary="credential is logged but no valid inclusion proof could be built",
            data={"in_log": True, "proof_verified": False, "leaf_hash": leaf.hex()},
        )
    return Evidence(
        id=evidence_id,
        kind=kind,
        passed=verified,
        summary=(
            f"inclusion proof verifies against the signed tree head of size {proof.tree_size}"
            if verified
            else "inclusion proof does not verify against the signed tree head"
        ),
        data={
            "in_log": True,
            "proof_verified": verified,
            "leaf_index": proof.leaf_index,
            "tree_size": proof.tree_size,
            "proof_length": len(proof.path),
            "root_hash": sth.root_hash.hex(),
            "leaf_hash": leaf.hex(),
        },
    )


def cdp_evidence(
    score: CdpScore, cfg: PolicyConfig, evidence_id: str = "cdp",
    kind: EvidenceKind = EvidenceKind.CDP,
) -> Evidence:  # fmt: skip
    """Turn a score into evidence: passed above the genuine bar, failed below the copy bar."""
    passed: bool | None = None
    if score.cnn >= cfg.cdp_genuine_min:
        passed = True
    elif score.cnn <= cfg.cdp_counterfeit_max:
        passed = False
    return Evidence(
        id=evidence_id,
        kind=kind,
        passed=passed,
        summary=(
            f"CNN authenticity {score.cnn:.2f}, correlation {score.correlation:.2f}, "
            f"alignment {score.alignment:.2f}"
        ),
        data={
            "cnn": score.cnn,
            "correlation": score.correlation,
            "raw_correlation": score.raw_correlation,
            "alignment": score.alignment,
        },
    )


def _cdp_error(evidence_id: str, kind: EvidenceKind, reason: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        kind=kind,
        passed=None,
        summary=f"no CDP measurement: {reason.replace('_', ' ')}",
        data={"error": reason},
    )


def _quality_evidence(report: QualityReport) -> Evidence:
    return Evidence(
        id="quality",
        kind=EvidenceKind.QUALITY,
        passed=report.ok,
        summary="capture quality is acceptable"
        if report.ok
        else f"capture quality is poor: {', '.join(report.reasons)}",
        data={
            "sharpness": report.sharpness,
            "brightness": report.brightness,
            "clipped": report.clipped,
            "reasons": list(report.reasons),
        },
    )


def _decode_evidence(sc: SignedCredential) -> Evidence:
    credential = sc.credential
    return Evidence(
        id="decode",
        kind=EvidenceKind.DECODE,
        passed=True,
        summary="QR decoded and credential parsed",
        data={
            "reason": None,
            "serial": credential.serial_hex,
            "iat": credential.iat,
            "untrusted": {"sku": credential.sku, "batch": credential.batch},
        },
    )


def _key_evidence(record: KeyRecord, iat: int) -> Evidence:
    data: dict[str, Any] = {"kid": record.kid_hex(), "purpose": record.purpose, "revoked": False}
    if record.revoked_at is None:
        return Evidence(
            id="key",
            kind=EvidenceKind.KEY_STATUS,
            passed=True,
            summary="issuer key is active",
            data=data,
        )
    before = iat < record.revoked_at.timestamp()
    data |= {
        "revoked": True,
        "revoked_at": record.revoked_at.astimezone(UTC).isoformat(),
        "issued_before_revocation": before,
    }
    when = "before" if before else "after"
    return Evidence(
        id="key",
        kind=EvidenceKind.KEY_STATUS,
        passed=False,
        summary=f"issuer key was revoked; the credential was issued {when} the revocation",
        data=data,
    )


def _unreadable(cap: DecodedCapture | None, reason: str) -> list[Evidence]:
    """The bundle for a photo that yielded no usable credential."""
    quality = (
        _quality_evidence(cap.quality)
        if cap is not None
        else Evidence.not_run("quality", EvidenceKind.QUALITY, "the QR code was not decoded")
    )
    decode_failed = Evidence(
        id="decode",
        kind=EvidenceKind.DECODE,
        passed=False,
        summary=f"no credential could be read: {reason.replace('_', ' ')}",
        data={"reason": reason},
    )
    skipped = [
        Evidence.not_run(id, kind, NO_CREDENTIAL)
        for id, kind in (
            ("signature", EvidenceKind.SIGNATURE),
            ("key", EvidenceKind.KEY_STATUS),
            ("log", EvidenceKind.LOG_INCLUSION),
            ("cdp", EvidenceKind.CDP),
            ("clone", EvidenceKind.CLONE),
        )
    ]
    return [quality, decode_failed, *skipped]


# --- helpers -----------------------------------------------------------------------------------


def _read(image: np.ndarray | bytes) -> DecodedCapture:
    gray = load_image(image) if isinstance(image, bytes) else image
    return decode(gray)


def _ms(seconds: float) -> float:
    return seconds * 1000.0


@contextmanager
def _timed(timings: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + _ms(time.perf_counter() - start)
