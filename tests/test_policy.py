"""The policy rule table: one case per rule, then precedence between rules."""

from __future__ import annotations

from typing import Any

import pytest

from sigil.config import Settings
from sigil.verify import (
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    PolicyConfig,
    Verdict,
    decide,
)

# Explicit thresholds, so the table tests the rules rather than whatever the defaults are.
CFG = PolicyConfig(
    cdp_genuine_min=0.80, cdp_counterfeit_max=0.20, clone_risk_high=0.70, clone_risk_low=0.40
)
ANY = frozenset({"GENUINE", "COUNTERFEIT", "INCONCLUSIVE"})


def _ev(id: str, kind: EvidenceKind, passed: bool | None, **data: Any) -> Evidence:
    return Evidence(id=id, kind=kind, passed=passed, summary=id, data=data)


def make_bundle(
    *,
    quality_ok: bool = True,
    decode_reason: str | None = None,
    signature: str | None = None,  # failure reason, None = valid
    revoked: str | None = None,  # None | "before" | "after"
    in_log: bool = True,
    cdp: float | str | None = 0.95,  # score, "align_failed", or None for missing
    clone: float | None = 0.0,
) -> EvidenceBundle:
    """A bundle describing a clean genuine scan; keyword arguments break one thing at a time."""
    readable = decode_reason is None
    items = [
        _ev("quality", EvidenceKind.QUALITY, quality_ok),
        _ev("decode", EvidenceKind.DECODE, readable, reason=decode_reason),
    ]
    if readable:
        key = {"revoked": revoked is not None, "issued_before_revocation": revoked == "before"}
        items += [
            _ev("signature", EvidenceKind.SIGNATURE, signature is None, reason=signature),
            _ev("key", EvidenceKind.KEY_STATUS, revoked is None, **key),
            _ev("log", EvidenceKind.LOG_INCLUSION, in_log),
        ]
        if cdp is not None:
            data = {"error": cdp} if isinstance(cdp, str) else {"cnn": cdp}
            items.append(_ev("cdp", EvidenceKind.CDP, None, **data))
        if clone is not None:
            items.append(_ev("clone", EvidenceKind.CLONE, None, risk=clone))
    return EvidenceBundle(scan_id="s", serial_hex="ab" * 12, items=items)


G, C, S, R = Verdict.GENUINE, Verdict.COUNTERFEIT, Verdict.SUSPICIOUS, Verdict.RECAPTURE
NONE = frozenset[str]()
CI = frozenset({"COUNTERFEIT", "INCONCLUSIVE"})

# (rule, bundle kwargs, verdict, hard_fail, escalate, agent_may_conclude)
RULE_TABLE: list[tuple[str, dict[str, Any], Verdict, bool, bool, frozenset[str]]] = [
    ("R1_capture", {"quality_ok": False}, R, False, False, NONE),
    ("R1_capture", {"decode_reason": "no_qr"}, R, False, False, NONE),
    ("R1_capture", {"decode_reason": "low_quality"}, R, False, False, NONE),
    ("R1_capture", {"decode_reason": "bad_image"}, R, False, False, NONE),
    ("R2_malformed", {"decode_reason": "malformed_credential"}, C, True, False, NONE),
    ("R3_signature", {"signature": "bad_signature"}, C, True, False, NONE),
    ("R3_signature", {"signature": "unknown_key"}, C, True, False, NONE),
    ("R4_revoked_after", {"revoked": "after"}, C, True, False, NONE),
    ("R5_unlogged", {"in_log": False}, C, True, False, NONE),
    ("R6_align", {"cdp": "align_failed"}, R, False, False, NONE),
    ("R7_copy", {"cdp": 0.20}, C, False, False, NONE),
    ("R7_copy", {"cdp": 0.0}, C, False, False, NONE),
    ("R8_revoked_before", {"revoked": "before"}, S, False, True, CI),
    ("R9_clone", {"clone": 0.70}, S, False, True, ANY),
    ("R10_genuine", {}, G, False, False, NONE),
    ("R10_genuine", {"cdp": 0.80, "clone": 0.39}, G, False, False, NONE),
    ("R11_ambiguous", {"cdp": 0.50}, S, False, True, ANY),
    ("R11_ambiguous", {"cdp": 0.95, "clone": 0.40}, S, False, True, ANY),
    ("R11_ambiguous", {"cdp": 0.79}, S, False, True, ANY),
]


@pytest.mark.parametrize(("rule", "kwargs", "verdict", "hard", "escalate", "may"), RULE_TABLE)
def test_rule_table(
    rule: str,
    kwargs: dict[str, Any],
    verdict: Verdict,
    hard: bool,
    escalate: bool,
    may: frozenset[str],
) -> None:
    decision = decide(make_bundle(**kwargs), CFG)
    assert decision.rule == rule
    assert decision.verdict is verdict
    assert decision.hard_fail is hard
    assert decision.escalate is escalate
    assert decision.agent_may_conclude == may
    assert decision.reasons


@pytest.mark.parametrize(
    ("winner", "kwargs"),
    [
        # A bad photo hides everything else.
        ("R1_capture", {"quality_ok": False, "signature": "bad_signature", "in_log": False}),
        ("R1_capture", {"decode_reason": "no_qr"}),
        # Cryptography before registry before pattern.
        ("R2_malformed", {"decode_reason": "malformed_credential"}),
        ("R3_signature", {"signature": "bad_signature", "revoked": "after", "in_log": False}),
        ("R4_revoked_after", {"revoked": "after", "in_log": False, "cdp": 0.0}),
        ("R5_unlogged", {"in_log": False, "cdp": 0.0, "clone": 0.9}),
        ("R5_unlogged", {"in_log": False, "cdp": "align_failed"}),
        # A copy is a copy even when it also looks cloned or follows a revocation.
        ("R6_align", {"cdp": "align_failed", "clone": 0.9}),
        ("R7_copy", {"cdp": 0.1, "clone": 0.9, "revoked": "before"}),
        ("R8_revoked_before", {"revoked": "before", "clone": 0.9, "cdp": 0.5}),
        ("R9_clone", {"clone": 0.9, "cdp": 0.99}),
    ],
)
def test_precedence(winner: str, kwargs: dict[str, Any]) -> None:
    assert decide(make_bundle(**kwargs), CFG).rule == winner


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signature": "bad_signature"},
        {"in_log": False},
        {"revoked": "after"},
        {"decode_reason": "malformed_credential"},
    ],
)
def test_hard_failures_cannot_be_overridden(kwargs: dict[str, Any]) -> None:
    """Whatever the CDP says, a cryptographic or registry failure leaves the agent no verdict."""
    for cdp in (0.0, 0.5, 1.0):
        decision = decide(make_bundle(cdp=cdp, **kwargs), CFG)
        assert decision.hard_fail
        assert decision.verdict is Verdict.COUNTERFEIT
        assert not decision.escalate
        assert decision.agent_may_conclude == frozenset()
        assert decision.confidence == 0.99


def test_genuine_needs_affirmative_evidence() -> None:
    """A missing check is never read as a pass."""
    assert decide(make_bundle(cdp=None), CFG).rule == "R11_ambiguous"
    assert decide(make_bundle(clone=None), CFG).rule == "R11_ambiguous"
    unlogged_unknown = make_bundle()
    unlogged_unknown.items = [i for i in unlogged_unknown.items if i.id != "log"]
    assert decide(unlogged_unknown, CFG).verdict is Verdict.SUSPICIOUS


def test_not_run_evidence_does_not_trigger_failure_rules() -> None:
    bundle = make_bundle()
    bundle.add(Evidence.not_run("log", EvidenceKind.LOG_INCLUSION, "x"))
    assert decide(bundle, CFG).verdict is Verdict.SUSPICIOUS  # not a COUNTERFEIT, not a GENUINE


@pytest.mark.parametrize(
    ("kwargs", "confidence"),
    [
        ({"cdp": 0.80}, 0.5),  # at the bar
        ({"cdp": 1.00}, 1.0),
        ({"cdp": 0.90}, 0.75),
        ({"cdp": 0.0}, 1.0),  # R7: 0.5 + 0.5 * (max - s) / max
        ({"cdp": 0.10}, 0.75),
        ({"cdp": 0.20}, 0.5),
        ({"cdp": 0.5}, 0.5),  # undecided
        ({"in_log": False}, 0.99),
        ({"quality_ok": False}, 0.0),
    ],
)
def test_confidence(kwargs: dict[str, Any], confidence: float) -> None:
    assert decide(make_bundle(**kwargs), CFG).confidence == pytest.approx(confidence)


def test_thresholds_come_from_config() -> None:
    strict = PolicyConfig(cdp_genuine_min=0.95, cdp_counterfeit_max=0.3, clone_risk_low=0.1)
    assert decide(make_bundle(cdp=0.9), strict).rule == "R11_ambiguous"
    assert decide(make_bundle(cdp=0.25), strict).rule == "R7_copy"
    assert decide(make_bundle(cdp=0.99, clone=0.1), strict).rule == "R11_ambiguous"


def test_config_from_settings() -> None:
    settings = Settings(cdp_genuine_min=0.7, cdp_counterfeit_max=0.1, clone_risk_high=0.9)
    cfg = PolicyConfig.from_settings(settings)
    assert (cfg.cdp_genuine_min, cfg.cdp_counterfeit_max) == (0.7, 0.1)
    assert (cfg.clone_risk_high, cfg.clone_risk_low) == (0.9, settings.clone_risk_low)
