"""Run the scenario suite end to end and collect graded results."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry

from sigil.config import Settings
from sigil.evals.graders import CaseResult, compute_metrics, grade
from sigil.evals.scenarios import Scenario, ScenarioKind, build_scenarios
from sigil.evals.world import ScenarioWorld
from sigil.harness import Model
from sigil.inference import BatcherConfig, BatcherMetrics, InferenceService, OnnxCdpScorer
from sigil.investigator import build_model, investigate
from sigil.verify import PolicyConfig

TRACE_DIR = "traces"


@dataclass
class SuiteResult:
    model: str
    seeds_per_kind: int
    cases: list[CaseResult]
    metrics: dict[str, Any]
    duration_s: float
    cdp_model: dict[str, Any]  # identifies the CDP model that produced these numbers


async def run_suite(
    *,
    model_name: str,
    seeds_per_kind: int,
    out_dir: Path,
    kinds: Sequence[ScenarioKind] | None = None,
    concurrency: int = 4,
) -> SuiteResult:
    """Run ``seeds_per_kind`` seeded cases of each kind and grade the system's final verdicts.

    Cases run concurrently against one shared inference service, as a busy deployment would. Each
    escalated case writes an investigation trace to ``out_dir/traces``.
    """
    started = time.perf_counter()
    settings = Settings(investigator_model=model_name)
    scorer = OnnxCdpScorer(threads=settings.inference_threads)
    inference = InferenceService(
        scorer,
        BatcherConfig(
            settings.batch_max_size, settings.batch_max_delay_ms, settings.batch_queue_limit
        ),
        BatcherMetrics(CollectorRegistry()),
    )
    model = build_model(settings)
    policy = PolicyConfig.from_settings(settings)
    gate = asyncio.Semaphore(concurrency)

    async def run_one(scenario: Scenario) -> CaseResult:
        async with gate:
            return await _run_case(scenario, inference, model, settings, policy, out_dir)

    await inference.start()
    try:
        cases = await asyncio.gather(*(run_one(s) for s in build_scenarios(seeds_per_kind, kinds)))
    finally:
        await inference.stop()
    return SuiteResult(
        model=model_name,
        seeds_per_kind=seeds_per_kind,
        cases=list(cases),
        metrics=compute_metrics(cases),
        duration_s=time.perf_counter() - started,
        cdp_model={k: scorer.metadata.get(k) for k in ("version", "sha256", "trained_at")},
    )


async def _run_case(
    scenario: Scenario,
    inference: InferenceService,
    model: Model,
    settings: Settings,
    policy: PolicyConfig,
    out_dir: Path,
) -> CaseResult:
    started = time.perf_counter()
    world = await ScenarioWorld.build(scenario, inference, policy)
    try:
        result = await world.verify()
        outcome = None
        if result.escalate and settings.investigation_enabled:
            outcome = await investigate(
                world.investigation_deps(result),
                model=model,
                settings=settings,
                trace_dir=out_dir / TRACE_DIR,
                run_id=scenario.id,
            )
    finally:
        world.close()

    final = outcome.final_verdict if outcome else result.verdict
    run = outcome.run if outcome else None
    cdp = result.evidence.get("cdp")
    correct, ideal = grade(scenario, final)
    return CaseResult(
        scenario=scenario,
        policy_verdict=result.verdict,
        final_verdict=final,
        escalated=outcome is not None,
        correct=correct,
        ideal=ideal,
        agent_status=run.status.value if run else None,
        agent_verdict=outcome.submission.verdict if outcome and outcome.submission else None,
        steps=run.steps if run else 0,
        cost_usd=run.cost_usd if run else 0.0,
        tokens=run.usage.total if run else 0,
        latency_ms=(time.perf_counter() - started) * 1000,
        guardrail_violations=len(run.guardrail_violations) if run else 0,
        trace_path=f"{TRACE_DIR}/{scenario.id}.jsonl" if outcome else None,
        rule=result.rule,
        cdp_cnn=cdp.data.get("cnn") if cdp else None,
        cdp_correlation=cdp.data.get("correlation") if cdp else None,
    )
