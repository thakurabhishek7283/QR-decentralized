"""The deterministic policy: evidence in, verdict out.

An ordered rule table; the first rule that matches wins. The order encodes precedence: capture
problems come first (nothing else can be trusted from a bad photo), then cryptographic and registry
failures (hard, never overridable), then the CDP verdict, and only then the clone analytics.
``GENUINE`` requires affirmative evidence from every check, so a missing item can never yield it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from sigil.verify.evidence import Evidence, EvidenceBundle

if TYPE_CHECKING:
    from sigil.config import Settings

CAPTURE_REASONS = frozenset({"bad_image", "no_qr", "low_quality"})
HARD_CONFIDENCE = 0.99
UNDECIDED_CONFIDENCE = 0.5


class Verdict(StrEnum):
    GENUINE = "GENUINE"
    COUNTERFEIT = "COUNTERFEIT"
    SUSPICIOUS = "SUSPICIOUS"
    RECAPTURE = "RECAPTURE"


@dataclass(frozen=True)
class PolicyConfig:
    cdp_genuine_min: float = 0.966
    cdp_counterfeit_max: float = 0.20
    clone_risk_high: float = 0.70
    clone_risk_low: float = 0.40

    @classmethod
    def from_settings(cls, settings: Settings) -> PolicyConfig:
        return cls(
            cdp_genuine_min=settings.cdp_genuine_min,
            cdp_counterfeit_max=settings.cdp_counterfeit_max,
            clone_risk_high=settings.clone_risk_high,
            clone_risk_low=settings.clone_risk_low,
        )


@dataclass(frozen=True)
class PolicyDecision:
    verdict: Verdict
    confidence: float
    rule: str  # id of the rule that fired, e.g. "R3_signature"
    reasons: list[str]
    hard_fail: bool  # cryptographic or registry failure: never overridable
    escalate: bool  # worth an investigator's time
    agent_may_conclude: frozenset[str]  # subset of GENUINE / COUNTERFEIT / INCONCLUSIVE


NO_CONCLUSION: frozenset[str] = frozenset()
ANY_CONCLUSION = frozenset({"GENUINE", "COUNTERFEIT", "INCONCLUSIVE"})


@dataclass(frozen=True)
class _Facts:
    """What the rules look at, extracted once from the bundle."""

    bundle: EvidenceBundle
    cfg: PolicyConfig
    cdp_score: float | None  # s
    clone_risk: float  # r, 0 when the clone check did not run

    def failed(self, id: str) -> bool:
        ev = self.bundle.get(id)
        return ev is not None and ev.ran and ev.passed is False

    def passed(self, id: str) -> bool:
        ev = self.bundle.get(id)
        return ev is not None and ev.ran and ev.passed is True

    def data(self, id: str) -> dict[str, object]:
        ev: Evidence | None = self.bundle.get(id)
        return ev.data if ev is not None else {}


def _facts(bundle: EvidenceBundle, cfg: PolicyConfig) -> _Facts:
    cdp = bundle.get("cdp")
    cnn = cdp.data.get("cnn") if cdp is not None else None
    clone = bundle.get("clone")
    risk = clone.data.get("risk") if clone is not None and clone.ran else None
    return _Facts(
        bundle=bundle,
        cfg=cfg,
        cdp_score=float(cnn) if isinstance(cnn, int | float) else None,
        clone_risk=float(risk) if isinstance(risk, int | float) else 0.0,
    )


def _decision(
    verdict: Verdict,
    rule: str,
    reasons: list[str],
    *,
    confidence: float,
    hard: bool = False,
    escalate: bool = False,
    may_conclude: frozenset[str] = NO_CONCLUSION,
) -> PolicyDecision:
    return PolicyDecision(verdict, confidence, rule, reasons, hard, escalate, may_conclude)


type Rule = Callable[[_Facts], PolicyDecision | None]


def _r1_capture(f: _Facts) -> PolicyDecision | None:
    decode_reason = f.data("decode").get("reason")
    if f.failed("quality"):
        why = "poor image quality"
    elif f.failed("decode") and decode_reason in CAPTURE_REASONS:
        why = str(decode_reason)
    else:
        return None
    return _decision(
        Verdict.RECAPTURE, "R1_capture", [f"the photo cannot be read ({why})"], confidence=0.0
    )


def _r2_malformed(f: _Facts) -> PolicyDecision | None:
    if f.failed("decode"):
        return _decision(
            Verdict.COUNTERFEIT,
            "R2_malformed",
            ["the QR code does not hold a valid Sigil credential"],
            confidence=HARD_CONFIDENCE,
            hard=True,
        )
    return None


def _r3_signature(f: _Facts) -> PolicyDecision | None:
    if f.failed("signature"):
        why = f.data("signature").get("reason", "invalid")
        return _decision(
            Verdict.COUNTERFEIT,
            "R3_signature",
            [f"the credential signature does not verify ({why})"],
            confidence=HARD_CONFIDENCE,
            hard=True,
        )
    return None


def _revoked_after_issue(f: _Facts) -> bool | None:
    """None if the key is not revoked, else whether the credential was issued before revocation."""
    key = f.data("key")
    if not key.get("revoked"):
        return None
    return bool(key.get("issued_before_revocation"))


def _r4_revoked_after(f: _Facts) -> PolicyDecision | None:
    if _revoked_after_issue(f) is False:
        return _decision(
            Verdict.COUNTERFEIT,
            "R4_revoked_after",
            ["the credential was issued after its signing key was revoked"],
            confidence=HARD_CONFIDENCE,
            hard=True,
        )
    return None


def _r5_unlogged(f: _Facts) -> PolicyDecision | None:
    if f.failed("log"):
        return _decision(
            Verdict.COUNTERFEIT,
            "R5_unlogged",
            ["the credential is not in the transparency log"],
            confidence=HARD_CONFIDENCE,
            hard=True,
        )
    return None


def _r6_align(f: _Facts) -> PolicyDecision | None:
    if f.data("cdp").get("error") == "align_failed":
        return _decision(
            Verdict.RECAPTURE,
            "R6_align",
            ["the copy-detection pattern could not be aligned"],
            confidence=0.0,
        )
    return None


def _r7_copy(f: _Facts) -> PolicyDecision | None:
    s, ceiling = f.cdp_score, f.cfg.cdp_counterfeit_max
    if s is not None and s <= ceiling:
        return _decision(
            Verdict.COUNTERFEIT,
            "R7_copy",
            [f"CDP score {s:.2f} is at or below the copy threshold {ceiling:.2f}"],
            confidence=0.5 + 0.5 * (ceiling - s) / ceiling,
        )
    return None


def _r8_revoked_before(f: _Facts) -> PolicyDecision | None:
    if _revoked_after_issue(f) is True:
        return _decision(
            Verdict.SUSPICIOUS,
            "R8_revoked_before",
            ["the signing key was later revoked; this credential predates the revocation"],
            confidence=UNDECIDED_CONFIDENCE,
            escalate=True,
            may_conclude=frozenset({"COUNTERFEIT", "INCONCLUSIVE"}),
        )
    return None


def _r9_clone(f: _Facts) -> PolicyDecision | None:
    r, high = f.clone_risk, f.cfg.clone_risk_high
    if r >= high:
        return _decision(
            Verdict.SUSPICIOUS,
            "R9_clone",
            [f"clone risk {r:.2f} is at or above {high:.2f}"],
            confidence=UNDECIDED_CONFIDENCE,
            escalate=True,
            may_conclude=ANY_CONCLUSION,
        )
    return None


def _r10_genuine(f: _Facts) -> PolicyDecision | None:
    s, floor = f.cdp_score, f.cfg.cdp_genuine_min
    affirmed = f.passed("signature") and f.passed("log") and f.bundle.get("clone") is not None
    if affirmed and s is not None and s >= floor and f.clone_risk < f.cfg.clone_risk_low:
        return _decision(
            Verdict.GENUINE,
            "R10_genuine",
            [f"CDP score {s:.2f} meets {floor:.2f} and clone risk {f.clone_risk:.2f} is low"],
            confidence=0.5 + 0.5 * (s - floor) / (1.0 - floor),
        )
    return None


def _r11_ambiguous(f: _Facts) -> PolicyDecision | None:
    s = "unavailable" if f.cdp_score is None else f"{f.cdp_score:.2f}"
    return _decision(
        Verdict.SUSPICIOUS,
        "R11_ambiguous",
        [f"CDP score {s} and clone risk {f.clone_risk:.2f} do not settle the case"],
        confidence=UNDECIDED_CONFIDENCE,
        escalate=True,
        may_conclude=ANY_CONCLUSION,
    )


# Order is precedence.
RULES: tuple[Rule, ...] = (
    _r1_capture,
    _r2_malformed,
    _r3_signature,
    _r4_revoked_after,
    _r5_unlogged,
    _r6_align,
    _r7_copy,
    _r8_revoked_before,
    _r9_clone,
    _r10_genuine,
    _r11_ambiguous,
)


def decide(bundle: EvidenceBundle, cfg: PolicyConfig) -> PolicyDecision:
    """Apply the rule table to a bundle; the first matching rule wins."""
    facts = _facts(bundle, cfg)
    for rule in RULES:
        if (decision := rule(facts)) is not None:
            return decision
    raise AssertionError("R11 matches everything")  # pragma: no cover
