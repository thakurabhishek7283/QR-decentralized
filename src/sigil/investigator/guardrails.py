"""Guardrails: semantic checks on the investigator's verdict before it is accepted.

The harness validates a submission's shape; these judge its meaning. They exist because a model
can be wrong, rushed or manipulated, and the cost of a wrong ``GENUINE`` is a counterfeit in a
customer's hands. Each guardrail returns plain-language violations so the model can fix its
submission and try again.
"""

from __future__ import annotations

from collections.abc import Sequence

from sigil.harness import Guardrail, RunView, ToolCall
from sigil.investigator.tools import SEEN, InvestigationDeps, genuine_floor
from sigil.verify import EvidenceBundle, PolicyConfig, PolicyDecision


def _cited(call: ToolCall) -> list[str]:
    return [str(c) for c in call.input.get("cited_evidence", [])]


class CitationsExist:
    """Every cited evidence id was actually revealed by a tool during this run."""

    name = "citations_exist"

    def check(self, call: ToolCall, run: RunView) -> list[str]:
        seen = run.state.get(SEEN, set())
        unknown = [c for c in _cited(call) if c not in seen]
        if not unknown:
            return []
        return [f"cited evidence was never retrieved: {', '.join(unknown)}"]


class PolicyBounds:
    """The verdict is one the policy allows the agent to reach.

    This is what makes a cryptographic failure impossible to talk away: for those cases the
    allowed set is empty, and for revoked-key cases it excludes ``GENUINE``.
    """

    name = "policy_bounds"

    def __init__(self, decision: PolicyDecision) -> None:
        self._allowed = decision.agent_may_conclude
        self._rule = decision.rule

    def check(self, call: ToolCall, run: RunView) -> list[str]:
        verdict = call.input.get("verdict")
        if verdict in self._allowed:
            return []
        allowed = ", ".join(sorted(self._allowed)) or "none"
        return [f"{verdict} is not permitted under policy rule {self._rule} (allowed: {allowed})"]


class GenuineRequiresCdp:
    """``GENUINE`` must rest on a CDP measurement that is consistent with a genuine item."""

    name = "genuine_requires_cdp"

    def __init__(self, bundle: EvidenceBundle, floor: float) -> None:
        self._bundle = bundle
        self._floor = floor

    def check(self, call: ToolCall, run: RunView) -> list[str]:
        if call.input.get("verdict") != "GENUINE":
            return []
        scores = [
            score
            for cited in _cited(call)
            if cited == "cdp" or cited.startswith("recapture-")
            if (ev := self._bundle.get(cited)) is not None
            if isinstance(score := ev.data.get("cnn"), int | float)
        ]
        if not scores:
            return ["a GENUINE verdict must cite the 'cdp' or a 'recapture-N' measurement"]
        if max(scores) < self._floor:
            return [
                f"the best cited CDP score {max(scores):.2f} is below {self._floor:.2f}, "
                "the lowest score consistent with a genuine item from this batch"
            ]
        return []


class CloneAcknowledged:
    """When clone risk is high, the verdict must engage with the scan history."""

    name = "clone_acknowledged"

    def __init__(self, bundle: EvidenceBundle, risk_high: float) -> None:
        self._bundle = bundle
        self._risk_high = risk_high

    def check(self, call: ToolCall, run: RunView) -> list[str]:
        clone = self._bundle.get("clone")
        risk = clone.data.get("risk", 0.0) if clone is not None else 0.0
        if risk < self._risk_high or {"clone", "history"} & set(_cited(call)):
            return []
        return [f"clone risk is {risk:.2f}; cite 'clone' or 'history' to show you weighed it"]


def build_guardrails(deps: InvestigationDeps) -> Sequence[Guardrail]:
    """The four guardrails, bound to one investigation's evidence and policy decision."""
    cfg: PolicyConfig = deps.policy
    return (
        CitationsExist(),
        PolicyBounds(deps.decision),
        GenuineRequiresCdp(deps.bundle, genuine_floor(deps.batch_qc(), cfg)),
        CloneAcknowledged(deps.bundle, cfg.clone_risk_high),
    )
