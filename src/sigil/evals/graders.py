"""Grading: turn a scenario and the system's final verdict into judged results and metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sigil.evals.scenarios import Scenario, ScenarioKind
from sigil.verify import Verdict

# Forged and unlogged credentials are digital attacks: their print is as good as a genuine one, so
# they say nothing about how well the CDP separates copies from originals.
DIGITAL_ATTACKS = frozenset({ScenarioKind.FORGED_SIGNATURE, ScenarioKind.ROGUE_UNLOGGED})


@dataclass
class CaseResult:
    scenario: Scenario
    policy_verdict: Verdict  # what the deterministic policy decided on its own
    final_verdict: Verdict  # after the investigator, if the case was escalated
    escalated: bool
    correct: bool  # final verdict is in the scenario's acceptable set
    ideal: bool  # final verdict is the scenario's best answer
    agent_status: str | None  # RunStatus value, None if no investigation ran
    agent_verdict: str | None  # what the agent submitted, before INCONCLUSIVE became SUSPICIOUS
    steps: int
    cost_usd: float
    tokens: int
    latency_ms: float  # verification plus investigation, end to end
    guardrail_violations: int
    trace_path: str | None  # relative to the report directory
    rule: str
    cdp_cnn: float | None  # primary-capture scores, for the AUC metrics
    cdp_correlation: float | None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["scenario"] = {
            "id": self.scenario.id,
            "kind": self.scenario.kind.value,
            "seed": self.scenario.seed,
            "truth": self.scenario.truth,
        }
        for key in ("policy_verdict", "final_verdict"):
            data[key] = data[key].value
        return data


def grade(scenario: Scenario, final: Verdict) -> tuple[bool, bool]:
    """``(correct, ideal)`` for a final verdict."""
    return final in scenario.acceptable, final is scenario.ideal


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half)."""
    if not positives or not negatives:
        return None
    scores = np.concatenate([positives, negatives])
    order = scores.argsort(kind="stable")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # Average ranks over ties.
    start = 0
    for end in range(1, len(scores) + 1):
        if end == len(scores) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    n_pos, n_neg = len(positives), len(negatives)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def compute_metrics(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """The suite metrics of LLD §10.2. Rates over an empty population are ``None``."""
    genuine = [c for c in cases if c.scenario.truth == "GENUINE"]
    counterfeit = [c for c in cases if c.scenario.truth == "COUNTERFEIT"]
    escalated = [c for c in cases if c.escalated]
    decided = [c for c in cases if c.final_verdict is not Verdict.RECAPTURE]
    latencies = [c.latency_ms for c in cases]

    by_kind: dict[str, dict[str, Any]] = {}
    for kind in dict.fromkeys(c.scenario.kind for c in cases):
        group = [c for c in cases if c.scenario.kind is kind]
        by_kind[kind.value] = {
            "n": len(group),
            "accuracy": sum(c.correct for c in group) / len(group),
            "ideal_rate": sum(c.ideal for c in group) / len(group),
            "escalated": sum(c.escalated for c in group),
            "verdicts": dict(Counter(c.final_verdict.value for c in group)),
        }

    scored = [c for c in cases if c.scenario.kind not in DIGITAL_ATTACKS and c.cdp_cnn is not None]
    real = [c for c in scored if c.scenario.truth == "GENUINE"]
    fake = [c for c in scored if c.scenario.truth == "COUNTERFEIT"]
    cdp_auc: dict[str, Any] = {"n_genuine": len(real), "n_counterfeit": len(fake)}
    for name in ("cnn", "correlation"):
        attr = f"cdp_{name}"
        cdp_auc[name] = auc([getattr(c, attr) for c in real], [getattr(c, attr) for c in fake])

    return {
        "cases": len(cases),
        "accuracy": _rate(sum(c.correct for c in cases), len(cases)),
        "decided_accuracy": _rate(sum(c.correct for c in decided), len(decided)),
        "recapture_rate": _rate(len(cases) - len(decided), len(cases)),
        "false_accept_rate": _rate(
            sum(c.final_verdict is Verdict.GENUINE for c in counterfeit), len(counterfeit)
        ),
        "false_reject_rate": _rate(
            sum(c.final_verdict is Verdict.COUNTERFEIT for c in genuine), len(genuine)
        ),
        "friction_rate": _rate(
            sum(c.final_verdict in (Verdict.SUSPICIOUS, Verdict.RECAPTURE) for c in genuine),
            len(genuine),
        ),
        "escalation_rate": _rate(len(escalated), len(cases)),
        "agent_resolution_rate": _rate(
            sum(c.final_verdict.value == c.scenario.truth for c in escalated), len(escalated)
        ),
        "policy_violations": sum(c.guardrail_violations for c in cases),
        "mean_cost_usd": _mean([c.cost_usd for c in escalated]),
        "mean_steps": _mean([c.steps for c in escalated]),
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "by_kind": by_kind,
        "cdp_auc": cdp_auc,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None
