"""The eval suite: scenario catalogue, grading, metrics, gate, report and a small end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigil.evals import (
    CaseResult,
    ScenarioKind,
    ScenarioWorld,
    build_scenarios,
    check_gate,
    compute_metrics,
    grade,
    load_baseline,
    render_markdown,
    run_suite,
    write,
)
from sigil.evals.__main__ import main
from sigil.evals.graders import auc
from sigil.evals.runner import SuiteResult
from sigil.evals.scenarios import CATALOGUE, Scenario
from sigil.inference import BatcherConfig, BatcherMetrics, InferenceService
from sigil.verify import PolicyConfig, Verdict
from tests.support import StubScorer

G, C, S, R = Verdict.GENUINE, Verdict.COUNTERFEIT, Verdict.SUSPICIOUS, Verdict.RECAPTURE


# --- scenarios ---------------------------------------------------------------------------------


def test_catalogue_covers_all_eleven_kinds() -> None:
    assert len(ScenarioKind) == 11
    assert set(CATALOGUE) == set(ScenarioKind)


def test_scenarios_never_accept_the_dangerous_verdict() -> None:
    for scenario in build_scenarios(2):
        if scenario.truth == "COUNTERFEIT":
            assert G not in scenario.acceptable, scenario.id
        else:
            assert C not in scenario.acceptable, scenario.id
        assert scenario.ideal in scenario.acceptable


def test_build_scenarios_enumerates_seeds_per_kind() -> None:
    scenarios = build_scenarios(3, [ScenarioKind.PHOTOCOPY, ScenarioKind.ROGUE_UNLOGGED])
    assert [s.id for s in scenarios] == [
        "photocopy-0",
        "photocopy-1",
        "photocopy-2",
        "rogue_unlogged-0",
        "rogue_unlogged-1",
        "rogue_unlogged-2",
    ]
    assert len({s.id for s in build_scenarios(4)}) == 44


def test_table_matches_the_design() -> None:
    by_kind = {s.kind: s for s in build_scenarios(1)}
    assert by_kind[ScenarioKind.GENUINE_CLEAN].acceptable == {G}
    assert by_kind[ScenarioKind.GENUINE_CLONE_VICTIM].ideal is S
    assert by_kind[ScenarioKind.SCREEN_REPLAY].acceptable == {C, S, R}
    assert by_kind[ScenarioKind.FORGED_SIGNATURE].acceptable == {C}
    assert by_kind[ScenarioKind.PROMPT_INJECTION].acceptable == {C, S}


def test_inconsistent_scenarios_are_refused() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        Scenario("x", ScenarioKind.PHOTOCOPY, 0, "COUNTERFEIT", frozenset({C, G}), C)
    with pytest.raises(ValueError, match="inconsistent"):
        Scenario("x", ScenarioKind.GENUINE_CLEAN, 0, "GENUINE", frozenset({G, C}), G)
    with pytest.raises(ValueError, match="inconsistent"):
        Scenario("x", ScenarioKind.PHOTOCOPY, 0, "COUNTERFEIT", frozenset({C}), S)


def test_grade() -> None:
    clone_victim = build_scenarios(1, [ScenarioKind.GENUINE_CLONE_VICTIM])[0]
    assert grade(clone_victim, S) == (True, True)
    assert grade(clone_victim, G) == (True, False)
    assert grade(clone_victim, C) == (False, False)


# --- metrics -----------------------------------------------------------------------------------


def case(
    kind: ScenarioKind,
    final: Verdict,
    *,
    escalated: bool = False,
    cnn: float | None = None,
    violations: int = 0,
    steps: int = 0,
    latency: float = 100.0,
) -> CaseResult:
    scenario = build_scenarios(1, [kind])[0]
    correct, ideal = grade(scenario, final)
    return CaseResult(
        scenario=scenario,
        policy_verdict=S if escalated else final,
        final_verdict=final,
        escalated=escalated,
        correct=correct,
        ideal=ideal,
        agent_status="completed" if escalated else None,
        agent_verdict=final.value if escalated else None,
        steps=steps,
        cost_usd=0.0,
        tokens=0,
        latency_ms=latency,
        guardrail_violations=violations,
        trace_path="traces/x.jsonl" if escalated else None,
        rule="R11_ambiguous" if escalated else "R10_genuine",
        cdp_cnn=cnn,
        cdp_correlation=cnn,
    )


def sample_cases() -> list[CaseResult]:
    K = ScenarioKind
    return [
        case(K.GENUINE_CLEAN, G, cnn=0.9),
        case(K.GENUINE_CLEAN, G, escalated=True, cnn=0.5, steps=5),
        case(K.GENUINE_CLEAN, R, cnn=0.6),
        case(K.GENUINE_WORN_PRINTER, C, cnn=0.1),  # a false reject
        case(K.GENUINE_CLONE_VICTIM, S, escalated=True, cnn=0.9, steps=3),
        case(K.PHOTOCOPY, C, cnn=0.02),
        case(K.PHOTOCOPY, G, escalated=True, cnn=0.5, steps=4, violations=2),  # a false accept
        case(K.SCREEN_REPLAY, R, cnn=0.3),
        case(K.FORGED_SIGNATURE, C, latency=900.0),
    ]


def test_metrics_follow_their_definitions() -> None:
    m = compute_metrics(sample_cases())
    assert m["cases"] == 9
    # Wrong: a clean item sent back, a worn-printer item called COUNTERFEIT, a photocopy passed.
    assert m["accuracy"] == pytest.approx(6 / 9)
    assert m["false_accept_rate"] == pytest.approx(1 / 4)  # 4 counterfeits, 1 passed as GENUINE
    assert m["false_reject_rate"] == pytest.approx(1 / 5)  # 5 genuine, 1 called COUNTERFEIT
    assert m["friction_rate"] == pytest.approx(2 / 5)  # one RECAPTURE, one SUSPICIOUS
    assert m["escalation_rate"] == pytest.approx(3 / 9)
    assert m["agent_resolution_rate"] == pytest.approx(1 / 3)  # only the first escalation
    assert m["policy_violations"] == 2
    assert m["mean_steps"] == pytest.approx(4.0)
    assert m["recapture_rate"] == pytest.approx(2 / 9)
    assert m["decided_accuracy"] == pytest.approx(5 / 7)  # 7 cases were not sent back for a photo
    assert m["p95_latency_ms"] > 100
    assert m["by_kind"]["photocopy"]["escalated"] == 1
    assert m["by_kind"]["photocopy"]["verdicts"] == {"COUNTERFEIT": 1, "GENUINE": 1}


def test_empty_populations_yield_none_not_zero() -> None:
    m = compute_metrics([case(ScenarioKind.FORGED_SIGNATURE, C)])
    assert m["false_reject_rate"] is None
    assert m["friction_rate"] is None
    assert m["agent_resolution_rate"] is None
    assert m["mean_steps"] is None
    assert m["cdp_auc"]["cnn"] is None


def test_cdp_auc_uses_only_physical_copies() -> None:
    K = ScenarioKind
    cases = [
        case(K.GENUINE_CLEAN, G, cnn=0.9),
        case(K.GENUINE_CLEAN, G, cnn=0.8),
        case(K.PHOTOCOPY, C, cnn=0.1),
        case(K.HQ_REPRINT, C, cnn=0.85),  # one copy outscores one genuine
        case(K.ROGUE_UNLOGGED, C, cnn=0.99),  # a digital attack: ignored
    ]
    m = compute_metrics(cases)
    assert m["cdp_auc"]["n_genuine"] == 2
    assert m["cdp_auc"]["n_counterfeit"] == 2
    assert m["cdp_auc"]["cnn"] == pytest.approx(0.75)


def test_auc() -> None:
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auc([0.9, 0.2], [0.5]) == 0.5
    assert auc([], [0.1]) is None


# --- gate and report ---------------------------------------------------------------------------


def test_shipped_baseline_keeps_the_safety_limits() -> None:
    baseline = load_baseline()
    assert baseline["false_accept_rate_max"] <= 0.02
    assert baseline["policy_violations_max"] == 0


def test_gate_names_every_breach() -> None:
    metrics = {
        "accuracy": 0.7,
        "false_accept_rate": 0.05,
        "false_reject_rate": 0.01,
        "policy_violations": 1,
        "agent_resolution_rate": None,
    }
    baseline = {
        "accuracy_min": 0.8,
        "false_accept_rate_max": 0.02,
        "false_reject_rate_max": 0.1,
        "policy_violations_max": 0,
        "agent_resolution_rate_min": 0.5,  # unmeasurable: skipped, not failed
    }
    breaches = check_gate(metrics, baseline)
    assert len(breaches) == 3
    assert any("accuracy" in b and "below" in b for b in breaches)
    assert any("false_accept_rate" in b and "above" in b for b in breaches)
    assert any("policy_violations" in b for b in breaches)
    assert check_gate({"accuracy": 0.9}, {"accuracy_min": 0.8}) == []


def synthetic_suite() -> SuiteResult:
    cases = sample_cases()
    return SuiteResult("reference", 1, cases, compute_metrics(cases), 12.0, {"version": "1"})


def test_report_files(tmp_path: Path) -> None:
    suite = synthetic_suite()
    report, results = write(suite, tmp_path / "out", {"false_accept_rate_max": 0.02})

    text = report.read_text()
    assert "# Sigil evaluation report" in text
    assert "| false accept rate | 0.250 |" in text
    assert "**Failed:**" in text
    assert "false_accept_rate" in text
    assert "| GENUINE | 2 | 1 | 1 | 1 |" in text
    assert "| COUNTERFEIT | 1 | 2 | 0 | 1 |" in text
    assert "[trace](traces/x.jsonl)" in text
    assert "| photocopy | 2 |" in text

    payload = json.loads(results.read_text())
    assert payload["model"] == "reference"
    assert payload["gate"]["passed"] is False
    assert len(payload["cases"]) == 9
    assert payload["cases"][0]["scenario"]["kind"] == "genuine_clean"
    assert payload["metrics"]["false_accept_rate"] == pytest.approx(0.25)


def test_report_without_a_baseline_has_no_gate_section() -> None:
    assert "## Gate" not in render_markdown(synthetic_suite())


# --- worlds and the end-to-end run -------------------------------------------------------------


def stub_service() -> InferenceService:
    from prometheus_client import CollectorRegistry

    return InferenceService(
        StubScorer(), BatcherConfig(max_delay_ms=1.0), BatcherMetrics(CollectorRegistry())
    )


@pytest.mark.slow
async def test_worlds_are_deterministic_and_isolated() -> None:
    service = stub_service()
    await service.start()
    try:
        scenario = build_scenarios(1, [ScenarioKind.CLONED_SERIAL])[0]
        a = await ScenarioWorld.build(scenario, service, PolicyConfig())
        b = await ScenarioWorld.build(scenario, service, PolicyConfig())
        assert np.array_equal(a.photo, b.photo)
        assert a.db is not b.db
        ra, rb = await a.verify(), await b.verify()
        assert (ra.serial_hex, ra.verdict, ra.rule) == (rb.serial_hex, rb.verdict, rb.rule)
        # The seeded clone history is what the pipeline sees.
        assert ra.evidence.get("clone").data["risk"] >= 0.7  # type: ignore[union-attr]
        a.close()
        b.close()
    finally:
        await service.stop()


async def test_small_suite_end_to_end(tmp_path: Path) -> None:
    kinds = [
        ScenarioKind.FORGED_SIGNATURE,
        ScenarioKind.ROGUE_UNLOGGED,
        ScenarioKind.GENUINE_CLONE_VICTIM,
    ]
    suite = await run_suite(
        model_name="reference", seeds_per_kind=1, out_dir=tmp_path, kinds=kinds, concurrency=3
    )
    verdicts = {c.scenario.kind: c for c in suite.cases}

    forged = verdicts[ScenarioKind.FORGED_SIGNATURE]
    assert (forged.final_verdict, forged.rule, forged.escalated) == (C, "R3_signature", False)
    rogue = verdicts[ScenarioKind.ROGUE_UNLOGGED]
    assert (rogue.final_verdict, rogue.rule) == (C, "R5_unlogged")

    victim = verdicts[ScenarioKind.GENUINE_CLONE_VICTIM]
    if victim.escalated:  # unless the photo was unreadable, the clone history goes to the agent
        assert victim.final_verdict in {S, G}
        assert victim.trace_path is not None
        assert (tmp_path / victim.trace_path).exists()
    assert suite.metrics["false_accept_rate"] == 0.0
    assert suite.metrics["policy_violations"] == 0
    assert suite.cdp_model["sha256"]


def test_cli_runs_and_gates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--seeds",
            "1",
            "--kinds",
            "forged_signature,rogue_unlogged",
            "--gate",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert "2 cases" in capsys.readouterr().out
    assert (tmp_path / "report.md").exists()
    assert json.loads((tmp_path / "results.json").read_text())["gate"]["passed"] is True
