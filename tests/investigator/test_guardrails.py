"""Each guardrail, accepting and rejecting."""

from __future__ import annotations

from typing import Any

from sigil.harness import RunView, ToolCall
from sigil.inference import InferenceService
from sigil.investigator import (
    CitationsExist,
    CloneAcknowledged,
    GenuineRequiresCdp,
    PolicyBounds,
    build_guardrails,
)
from sigil.verify import PolicyDecision, Verdict
from tests.investigator.helpers import add_recapture, make_case
from tests.support import World


def submission(verdict: str, *cited: str) -> ToolCall:
    payload: dict[str, Any] = {
        "verdict": verdict,
        "confidence": 0.8,
        "cited_evidence": list(cited),
        "rationale": "Reasoned from the evidence that was retrieved.",
    }
    return ToolCall("s1", "submit_verdict", payload)


def view(*seen: str) -> RunView:
    return RunView(messages=(), tool_results=(), state={"seen_evidence": set(seen)})


# --- citations_exist ---------------------------------------------------------------------------


def test_citations_must_have_been_retrieved() -> None:
    guard = CitationsExist()
    assert guard.check(submission("COUNTERFEIT", "cdp", "clone"), view("cdp", "clone", "log")) == []
    (violation,) = guard.check(submission("COUNTERFEIT", "cdp", "made-up"), view("cdp"))
    assert "made-up" in violation
    assert "cdp" not in violation.split(":", 1)[1]


def test_citations_fail_when_nothing_was_retrieved() -> None:
    assert CitationsExist().check(submission("COUNTERFEIT", "cdp"), view())  # no state at all
    empty = RunView(messages=(), tool_results=(), state={})
    assert CitationsExist().check(submission("COUNTERFEIT", "cdp"), empty)


# --- policy_bounds -----------------------------------------------------------------------------


async def test_policy_bounds_follow_the_decision(world: World, inference: InferenceService) -> None:
    ambiguous = make_case(world, inference, cnn=0.5).deps().decision
    guard = PolicyBounds(ambiguous)
    for verdict in ("GENUINE", "COUNTERFEIT", "INCONCLUSIVE"):
        assert guard.check(submission(verdict, "cdp"), view("cdp")) == []

    revoked = make_case(world, inference, cnn=0.5, unit=1, revoked="before").deps().decision
    assert revoked.rule == "R8_revoked_before"
    guard = PolicyBounds(revoked)
    (violation,) = guard.check(submission("GENUINE", "cdp"), view("cdp"))
    assert "R8_revoked_before" in violation
    assert guard.check(submission("COUNTERFEIT", "cdp"), view("cdp")) == []
    assert guard.check(submission("INCONCLUSIVE", "cdp"), view("cdp")) == []


async def test_a_hard_failure_leaves_no_permitted_verdict() -> None:

    hard = PolicyDecision(
        Verdict.COUNTERFEIT, 0.99, "R3_signature", ["bad"], True, False, frozenset()
    )
    guard = PolicyBounds(hard)
    for verdict in ("GENUINE", "COUNTERFEIT", "INCONCLUSIVE"):
        assert guard.check(submission(verdict, "signature"), view("signature"))


# --- genuine_requires_cdp ----------------------------------------------------------------------


async def test_genuine_must_cite_a_consistent_cdp_measurement(
    world: World, inference: InferenceService
) -> None:
    deps = make_case(world, inference, cnn=0.5, qc=(0.5, 0.05)).deps()
    floor = 0.5 - 2 * 0.05
    guard = GenuineRequiresCdp(deps.bundle, floor)

    # Nothing about the pattern is cited.
    (no_cdp,) = guard.check(submission("GENUINE", "clone", "log"), view("clone", "log"))
    assert "cdp" in no_cdp

    # The primary score is typical for the batch.
    assert guard.check(submission("GENUINE", "cdp"), view("cdp")) == []

    # A score below the batch floor does not support GENUINE...
    low = GenuineRequiresCdp(deps.bundle, 0.7)
    (too_low,) = low.check(submission("GENUINE", "cdp"), view("cdp"))
    assert "0.50" in too_low
    assert "0.70" in too_low

    # ...unless a cited recapture does.
    add_recapture(deps, 0.9)
    assert low.check(submission("GENUINE", "cdp", "recapture-1"), view("cdp", "recapture-1")) == []


async def test_only_genuine_needs_cdp_support(world: World, inference: InferenceService) -> None:
    deps = make_case(world, inference, cnn=0.5).deps()
    guard = GenuineRequiresCdp(deps.bundle, 0.9)
    assert guard.check(submission("COUNTERFEIT", "log"), view("log")) == []
    assert guard.check(submission("INCONCLUSIVE", "log"), view("log")) == []


# --- clone_acknowledged ------------------------------------------------------------------------


async def test_high_clone_risk_must_be_acknowledged(
    world: World, inference: InferenceService
) -> None:
    deps = make_case(world, inference, cnn=0.9, clone_history=True).deps()
    guard = CloneAcknowledged(deps.bundle, risk_high=0.7)
    (violation,) = guard.check(submission("INCONCLUSIVE", "cdp"), view("cdp"))
    assert "0.80" in violation
    assert guard.check(submission("INCONCLUSIVE", "cdp", "clone"), view("cdp", "clone")) == []
    assert guard.check(submission("INCONCLUSIVE", "cdp", "history"), view("cdp", "history")) == []


async def test_low_clone_risk_needs_no_acknowledgement(
    world: World, inference: InferenceService
) -> None:
    deps = make_case(world, inference, cnn=0.5).deps()
    guard = CloneAcknowledged(deps.bundle, risk_high=0.7)
    assert guard.check(submission("INCONCLUSIVE", "cdp"), view("cdp")) == []


# --- wiring ------------------------------------------------------------------------------------


async def test_build_guardrails_binds_all_four(world: World, inference: InferenceService) -> None:
    deps = make_case(world, inference, cnn=0.5).deps()
    guards = build_guardrails(deps)
    assert [g.name for g in guards] == [
        "citations_exist",
        "policy_bounds",
        "genuine_requires_cdp",
        "clone_acknowledged",
    ]
