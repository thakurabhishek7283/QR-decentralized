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
from sigil.inference import BatcherConfig, BatcherMetrics, InferenceService, OnnxCdpScorer
from sigil.verify import PolicyConfig


@dataclass
class SuiteResult:
    seeds_per_kind: int
    cases: list[CaseResult]
    metrics: dict[str, Any]
    duration_s: float
    cdp_model: dict[str, Any]  # identifies the CDP model that produced these numbers


async def run_suite(
    *,
    seeds_per_kind: int,
    out_dir: Path,
    kinds: Sequence[ScenarioKind] | None = None,
    concurrency: int = 4,
) -> SuiteResult:
    """Run ``seeds_per_kind`` seeded cases of each kind and grade the system's final verdicts.

    Cases run concurrently against one shared inference service, as a busy deployment would.
    """
    started = time.perf_counter()
    settings = Settings()
    scorer = OnnxCdpScorer(threads=settings.inference_threads)
    inference = InferenceService(
        scorer,
        BatcherConfig(
            settings.batch_max_size, settings.batch_max_delay_ms, settings.batch_queue_limit
        ),
        BatcherMetrics(CollectorRegistry()),
    )
    policy = PolicyConfig.from_settings(settings)
    gate = asyncio.Semaphore(concurrency)

    async def run_one(scenario: Scenario) -> CaseResult:
        async with gate:
            return await _run_case(scenario, inference, policy)

    await inference.start()
    try:
        cases = await asyncio.gather(*(run_one(s) for s in build_scenarios(seeds_per_kind, kinds)))
    finally:
        await inference.stop()
    return SuiteResult(
        seeds_per_kind=seeds_per_kind,
        cases=list(cases),
        metrics=compute_metrics(cases),
        duration_s=time.perf_counter() - started,
        cdp_model={k: scorer.metadata.get(k) for k in ("version", "sha256", "trained_at")},
    )


async def _run_case(
    scenario: Scenario, inference: InferenceService, policy: PolicyConfig
) -> CaseResult:
    started = time.perf_counter()
    world = await ScenarioWorld.build(scenario, inference, policy)
    try:
        result = await world.verify()
    finally:
        world.close()

    cdp = result.evidence.get("cdp")
    correct, ideal = grade(scenario, result.verdict)
    return CaseResult(
        scenario=scenario,
        verdict=result.verdict,
        correct=correct,
        ideal=ideal,
        latency_ms=(time.perf_counter() - started) * 1000,
        rule=result.rule,
        cdp_cnn=cdp.data.get("cnn") if cdp else None,
        cdp_correlation=cdp.data.get("correlation") if cdp else None,
    )
