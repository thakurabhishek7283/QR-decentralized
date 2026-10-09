"""Verification: evidence, the deterministic policy, and the pipeline that produces both."""

from sigil.verify.evidence import Evidence, EvidenceBundle, EvidenceKind
from sigil.verify.pipeline import (
    ScanContext,
    VerificationPipeline,
    VerificationResult,
    cdp_evidence,
    check_log,
)
from sigil.verify.policy import PolicyConfig, PolicyDecision, Verdict, decide

__all__ = [
    "Evidence",
    "EvidenceBundle",
    "EvidenceKind",
    "PolicyConfig",
    "PolicyDecision",
    "ScanContext",
    "Verdict",
    "VerificationPipeline",
    "VerificationResult",
    "cdp_evidence",
    "check_log",
    "decide",
]
