"""Evaluation: a labelled scenario generator, graders, a runner, a report and a CI gate.

Run it with ``python -m sigil.evals --seeds 3 --gate``.
"""

from sigil.evals.graders import CaseResult, compute_metrics, grade
from sigil.evals.report import check_gate, load_baseline, render_markdown, write
from sigil.evals.runner import SuiteResult, run_suite
from sigil.evals.scenarios import Scenario, ScenarioKind, build_scenarios
from sigil.evals.world import ScenarioWorld

__all__ = [
    "CaseResult",
    "Scenario",
    "ScenarioKind",
    "ScenarioWorld",
    "SuiteResult",
    "build_scenarios",
    "check_gate",
    "compute_metrics",
    "grade",
    "load_baseline",
    "render_markdown",
    "run_suite",
    "write",
]
