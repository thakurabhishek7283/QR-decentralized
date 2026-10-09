"""Running an investigation: wire the harness, tools, guardrails and a model together."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from sigil.config import Settings
from sigil.harness import (
    AgentRunner,
    AnthropicModel,
    Budget,
    JsonlTraceWriter,
    Model,
    RunConfig,
    RunResult,
    RunStatus,
    ScriptedModel,
    TraceRecorder,
)
from sigil.investigator.guardrails import build_guardrails
from sigil.investigator.prompts import SYSTEM_PROMPT, task_message
from sigil.investigator.reference_policy import make_reference_policy
from sigil.investigator.schemas import VerdictSubmission
from sigil.investigator.tools import InvestigationDeps, build_registry
from sigil.verify import EvidenceBundle, PolicyConfig, Verdict

REFERENCE_MODEL = "reference"


@dataclass(frozen=True)
class InvestigationOutcome:
    final_verdict: Verdict  # GENUINE or COUNTERFEIT from the agent; anything else is SUSPICIOUS
    submission: VerdictSubmission | None
    run: RunResult
    trace_path: Path | None
    evidence: EvidenceBundle  # the scan's evidence plus whatever the investigation gathered


def build_model(settings: Settings) -> Model:
    """The configured investigator: the offline reference policy, or a Claude model by id."""
    if settings.investigator_model == REFERENCE_MODEL:
        policy = make_reference_policy(PolicyConfig.from_settings(settings))
        return ScriptedModel(REFERENCE_MODEL, policy)
    return AnthropicModel(settings.investigator_model, effort=settings.investigator_effort)


async def investigate(
    deps: InvestigationDeps,
    *,
    model: Model,
    settings: Settings,
    trace_dir: Path | None,
    run_id: str | None = None,
) -> InvestigationOutcome:
    """Investigate one escalated scan and return the verdict the system should act on.

    Only a completed run whose submission passed every guardrail can yield a decisive verdict.
    A budget overrun, refusal, error, or an ``INCONCLUSIVE`` submission all resolve to
    ``SUSPICIOUS``: when the agent cannot settle a case, the system must not claim it did.
    """
    run_id = run_id or uuid.uuid4().hex
    trace_path = trace_dir / f"{run_id}.jsonl" if trace_dir is not None else None
    writers = [JsonlTraceWriter(trace_path)] if trace_path is not None else []
    runner = AgentRunner(
        model=model,
        tools=build_registry(),
        system=SYSTEM_PROMPT,
        guardrails=build_guardrails(deps),
        config=RunConfig(
            budget=Budget(
                max_steps=settings.agent_max_steps,
                max_cost_usd=settings.agent_max_cost_usd,
                max_wall_s=settings.agent_max_wall_s,
            )
        ),
        trace=TraceRecorder(run_id, writers),
    )
    run = await runner.run(task_message(deps.result, deps.decision), deps=deps)

    submission = None
    final = Verdict.SUSPICIOUS
    if run.status is RunStatus.COMPLETED and run.output is not None:
        submission = VerdictSubmission.model_validate(run.output)
        if submission.verdict != "INCONCLUSIVE":
            final = Verdict(submission.verdict)
    return InvestigationOutcome(final, submission, run, trace_path, deps.bundle)
