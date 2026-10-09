"""Deterministic replay of recorded runs.

``ReplayModel`` plays back the model responses captured in a trace. Re-running the agent loop
against it, with *current* tool code, guardrails and system prompt, shows whether the recorded
behaviour still holds: a recorded real-LLM investigation becomes a free, offline regression test.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sigil.harness.guardrails import Guardrail
from sigil.harness.runner import AgentRunner, RunConfig
from sigil.harness.tools import ToolRegistry
from sigil.harness.trace import (
    EventType,
    MemoryTraceWriter,
    TraceEvent,
    TraceRecorder,
    digest,
    verify_chain,
)
from sigil.harness.types import ModelRequest, ModelResponse


class ReplayDivergence(Exception):
    """The replayed run cannot continue against the recording."""


def _of_type(events: Sequence[TraceEvent], type: EventType) -> list[TraceEvent]:
    return [e for e in events if e.type is type]


def _started(events: Sequence[TraceEvent]) -> dict[str, Any]:
    started = _of_type(events, EventType.RUN_STARTED)
    if not started:
        raise ValueError("trace has no run_started event")
    return started[0].data


class ReplayModel:
    """Serves recorded responses in order.

    Raises ``ReplayDivergence`` if the tool set differs from the recording or if it is asked for
    more responses than were recorded. A request whose content differs from the recorded one (for
    instance because a tool now returns something else) is *not* fatal: it is noted in
    ``divergences`` and the recorded response is still returned, so later steps stay comparable.
    """

    def __init__(self, events: Sequence[TraceEvent]) -> None:
        started = _started(events)
        self._name: str = started["model"]
        self._tool_names = sorted(spec["name"] for spec in started["tools"])
        self._request_digests = [
            e.data["request_digest"] for e in _of_type(events, EventType.MODEL_REQUEST)
        ]
        self._responses = [
            ModelResponse.from_json(e.data) for e in _of_type(events, EventType.MODEL_RESPONSE)
        ]
        self._calls = 0
        self.divergences: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: ModelRequest) -> ModelResponse:
        step = self._calls + 1
        if self._calls >= len(self._responses):
            raise ReplayDivergence(
                f"step {step}: model called more times than the {len(self._responses)} recorded"
            )
        names = sorted(spec.name for spec in request.tools)
        if names != self._tool_names:
            raise ReplayDivergence(
                f"step {step}: tools {names} differ from recorded {self._tool_names}"
            )
        if digest(request.to_json()) != self._request_digests[self._calls]:
            self.divergences.append(f"step {step} request differs from the recording")
        response = self._responses[self._calls]
        self._calls += 1
        return response


@dataclass(frozen=True)
class ReplayReport:
    identical: bool
    divergences: list[str]
    original_status: str
    replay_status: str


async def replay(
    events: Sequence[TraceEvent],
    *,
    tools: ToolRegistry,
    system: str,
    guardrails: Sequence[Guardrail] = (),
    deps: Any = None,
    config: RunConfig | None = None,
) -> ReplayReport:
    """Re-run a recorded trace against current code and report every difference.

    ``config`` defaults to the one recorded in the trace.
    """
    if not verify_chain(events):
        raise ValueError("refusing to replay a trace that fails hash-chain verification")
    started = _started(events)
    model = ReplayModel(events)
    sink = MemoryTraceWriter()
    runner = AgentRunner(
        model=model,
        tools=tools,
        system=system,
        guardrails=guardrails,
        config=config or RunConfig.from_json(started["config"]),
        trace=TraceRecorder(events[0].run_id, [sink]),
    )
    await runner.run(started["task"], deps=deps)

    divergences = []
    if hashlib.sha256(system.encode()).hexdigest() != started["system_sha256"]:
        divergences.append("system prompt differs from the recording")
    divergences += model.divergences
    divergences += _tool_divergences(events, sink.events)
    divergences += _outcome_divergences(events, sink.events)
    return ReplayReport(
        identical=not divergences,
        divergences=divergences,
        original_status=_finished(events)["status"],
        replay_status=_finished(sink.events)["status"],
    )


def _finished(events: Sequence[TraceEvent]) -> dict[str, Any]:
    finished = _of_type(events, EventType.RUN_FINISHED)
    return finished[-1].data if finished else {"status": "unfinished", "output": None}


def _tool_divergences(original: Sequence[TraceEvent], replayed: Sequence[TraceEvent]) -> list[str]:
    replayed_results = {
        e.data["call_id"]: e.data for e in _of_type(replayed, EventType.TOOL_RESULT)
    }
    found = []
    for event in _of_type(original, EventType.TOOL_RESULT):
        data = event.data
        label = f"step {data['step']} tool {data['name']}"
        other = replayed_results.get(data["call_id"])
        if other is None:
            found.append(f"{label} was not executed in the replay")
        elif (other["content"], other["is_error"]) != (data["content"], data["is_error"]):
            found.append(f"{label} result differs")
    return found


def _outcome_divergences(
    original: Sequence[TraceEvent], replayed: Sequence[TraceEvent]
) -> list[str]:
    before, after = _finished(original), _finished(replayed)
    found = []
    if before["status"] != after["status"]:
        found.append(f"status changed from {before['status']} to {after['status']}")
        if after.get("error"):
            found.append(f"replay error: {after['error']}")
    if before["output"] != after["output"]:
        found.append("final output differs")
    guardrails_before = [e.data["violations"] for e in _of_type(original, EventType.GUARDRAIL)]
    guardrails_after = [e.data["violations"] for e in _of_type(replayed, EventType.GUARDRAIL)]
    if guardrails_before != guardrails_after:
        found.append("guardrail outcomes differ")
    return found
