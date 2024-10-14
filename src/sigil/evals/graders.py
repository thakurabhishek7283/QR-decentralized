"""Grading: turn a scenario and the system's verdict into judged results and metrics."""

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
    verdict: Verdict  # what the deterministic policy decided
    correct: bool  # the verdict is in the scenario's acceptable set
    ideal: bool  # the verdict is the scenario's best answer
    latency_ms: float  # verification, end to end
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
        data["verdict"] = data["verdict"].value
        return data


def grade(scenario: Scenario, verdict: Verdict) -> tuple[bool, bool]:
    """``(correct, ideal)`` for a verdict."""
    return verdict in scenario.acceptable, verdict is scenario.ideal


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
    """The suite metrics of LLD §8.2. Rates over an empty population are ``None``."""
    genuine = [c for c in cases if c.scenario.truth == "GENUINE"]
    counterfeit = [c for c in cases if c.scenario.truth == "COUNTERFEIT"]
    decided = [c for c in cases if c.verdict is not Verdict.RECAPTURE]
    latencies = [c.latency_ms for c in cases]

    by_kind: dict[str, dict[str, Any]] = {}
    for kind in dict.fromkeys(c.scenario.kind for c in cases):
        group = [c for c in cases if c.scenario.kind is kind]
        by_kind[kind.value] = {
            "n": len(group),
            "accuracy": sum(c.correct for c in group) / len(group),
            "ideal_rate": sum(c.ideal for c in group) / len(group),
            "verdicts": dict(Counter(c.verdict.value for c in group)),
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
            sum(c.verdict is Verdict.GENUINE for c in counterfeit), len(counterfeit)
        ),
        "false_reject_rate": _rate(
            sum(c.verdict is Verdict.COUNTERFEIT for c in genuine), len(genuine)
        ),
        "friction_rate": _rate(
            sum(c.verdict in (Verdict.SUSPICIOUS, Verdict.RECAPTURE) for c in genuine),
            len(genuine),
        ),
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "by_kind": by_kind,
        "cdp_auc": cdp_auc,
    }
