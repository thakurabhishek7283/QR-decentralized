"""Reporting and the CI gate: ``report.md`` and ``results.json``, and the regression baseline."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sigil.evals.graders import CaseResult
from sigil.evals.runner import SuiteResult
from sigil.verify import Verdict

BASELINE_PATH = Path(__file__).with_name("baseline.json")
VERDICTS = [
    v.value for v in (Verdict.GENUINE, Verdict.COUNTERFEIT, Verdict.SUSPICIOUS, Verdict.RECAPTURE)
]

# baseline key -> (metric, direction): a ``_max`` bound caps a metric, a ``_min`` bound floors it.
_GATES = {
    "accuracy_min": "accuracy",
    "decided_accuracy_min": "decided_accuracy",
    "false_accept_rate_max": "false_accept_rate",
    "false_reject_rate_max": "false_reject_rate",
    "friction_rate_max": "friction_rate",
    "agent_resolution_rate_min": "agent_resolution_rate",
    "policy_violations_max": "policy_violations",
}


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, float]:
    return {k: float(v) for k, v in json.loads(path.read_text()).items() if k in _GATES}


def check_gate(metrics: Mapping[str, Any], baseline: Mapping[str, float]) -> list[str]:
    """Every threshold the metrics breach, as readable messages. Empty means the gate passes.

    A bound on a metric that could not be measured (no cases of that population) is skipped.
    """
    breaches = []
    for key, bound in baseline.items():
        value = metrics.get(_GATES[key])
        if value is None:
            continue
        if (key.endswith("_max") and value > bound) or (key.endswith("_min") and value < bound):
            relation = "above" if key.endswith("_max") else "below"
            breaches.append(f"{_GATES[key]} = {value:.4g} is {relation} the limit {bound:.4g}")
    return breaches


def write(
    suite: SuiteResult, out_dir: Path, baseline: Mapping[str, float] | None = None
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``results.json`` into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    breaches = check_gate(suite.metrics, baseline) if baseline is not None else None
    report = out_dir / "report.md"
    results = out_dir / "results.json"
    report.write_text(render_markdown(suite, breaches))
    payload = {
        "model": suite.model,
        "seeds_per_kind": suite.seeds_per_kind,
        "duration_s": round(suite.duration_s, 1),
        "cdp_model": suite.cdp_model,
        "metrics": suite.metrics,
        "gate": None if breaches is None else {"passed": not breaches, "breaches": breaches},
        "cases": [c.to_json() for c in suite.cases],
    }
    results.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return report, results


# --- markdown ----------------------------------------------------------------------------------


def render_markdown(suite: SuiteResult, breaches: list[str] | None = None) -> str:
    m = suite.metrics
    cdp = suite.cdp_model
    lines = [
        "# Sigil evaluation report",
        "",
        f"- Investigator: `{suite.model}`",
        f"- Cases: {m['cases']} ({suite.seeds_per_kind} seeds per kind), run in "
        f"{suite.duration_s:.0f} s",
        f"- CDP model: version {cdp.get('version')}, sha256 `{str(cdp.get('sha256'))[:12]}`",
        "",
        "## Headline",
        "",
        "| Metric | Value | Meaning |",
        "|---|---|---|",
        *(f"| {name} | {_fmt(m[key], fmt)} | {meaning} |" for name, key, fmt, meaning in _HEADLINE),
        "",
    ]
    if breaches is not None:
        lines += ["## Gate", ""]
        lines += ["Passed."] if not breaches else ["**Failed:**", *(f"- {b}" for b in breaches)]
        lines.append("")
    lines += [
        "## Confusion matrix",
        "",
        "Rows are the ground truth, columns the final verdict.",
        "",
        *_confusion(suite.cases),
        "",
        "## By scenario kind",
        "",
        *_by_kind(m["by_kind"]),
        "",
        "## CDP discrimination (primary captures)",
        "",
        f"AUC over {m['cdp_auc']['n_genuine']} genuine and {m['cdp_auc']['n_counterfeit']} "
        f"copied items (digital attacks excluded): CNN {_fmt(m['cdp_auc']['cnn'], '.3f')}, "
        f"classical correlation {_fmt(m['cdp_auc']['correlation'], '.3f')}.",
        "",
        "## Investigator",
        "",
        *_agent_section(suite.cases),
        "",
    ]
    return "\n".join(lines)


_HEADLINE = [
    ("accuracy", "accuracy", ".3f", "final verdict is acceptable for the scenario"),
    ("decided accuracy", "decided_accuracy", ".3f", "accuracy excluding RECAPTURE"),
    ("recapture rate", "recapture_rate", ".3f", "cases where the photo could not be judged"),
    ("false accept rate", "false_accept_rate", ".3f", "counterfeits passed as GENUINE (safety)"),
    ("false reject rate", "false_reject_rate", ".3f", "genuine items called COUNTERFEIT"),
    ("friction rate", "friction_rate", ".3f", "genuine items left SUSPICIOUS or RECAPTURE"),
    ("escalation rate", "escalation_rate", ".3f", "cases sent to the investigator"),
    ("agent resolution", "agent_resolution_rate", ".3f", "escalated cases settled correctly"),
    ("policy violations", "policy_violations", "d", "guardrail rejections, recovered or not"),
    ("mean steps", "mean_steps", ".1f", "model turns per investigation"),
    ("mean cost (USD)", "mean_cost_usd", ".4f", "per investigation"),
    ("p95 latency (ms)", "p95_latency_ms", ".0f", "verify plus investigate, under concurrency"),
]


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _confusion(cases: list[CaseResult]) -> list[str]:
    counts = Counter((c.scenario.truth, c.final_verdict.value) for c in cases)
    rows = [
        "| truth | " + " | ".join(VERDICTS) + " |",
        "|---|" + "---|" * len(VERDICTS),
    ]
    for truth in ("GENUINE", "COUNTERFEIT"):
        rows.append(f"| {truth} | " + " | ".join(str(counts[truth, v]) for v in VERDICTS) + " |")
    return rows


def _by_kind(by_kind: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows = [
        "| kind | n | accuracy | ideal | escalated | " + " | ".join(VERDICTS) + " |",
        "|---|---|---|---|---|" + "---|" * len(VERDICTS),
    ]
    for kind, k in by_kind.items():
        verdicts = " | ".join(str(k["verdicts"].get(v, 0)) for v in VERDICTS)
        rows.append(
            f"| {kind} | {k['n']} | {k['accuracy']:.2f} | {k['ideal_rate']:.2f} | "
            f"{k['escalated']} | {verdicts} |"
        )
    return rows


def _agent_section(cases: list[CaseResult]) -> list[str]:
    investigated = [c for c in cases if c.escalated]
    if not investigated:
        return ["No case was escalated to the investigator."]
    statuses = Counter(c.agent_status for c in investigated)
    lines = [
        f"{len(investigated)} cases were escalated. Run status: "
        + ", ".join(f"{n} {status}" for status, n in sorted(statuses.items(), key=str))
        + ".",
        "",
        "| case | policy rule | agent submitted | final | correct | steps | trace |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in investigated:
        lines.append(
            f"| {c.scenario.id} | {c.rule} | {c.agent_verdict or '-'} | {c.final_verdict.value} | "
            f"{'yes' if c.correct else '**no**'} | {c.steps} | [trace]({c.trace_path}) |"
        )
    return lines
