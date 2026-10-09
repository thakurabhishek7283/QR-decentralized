from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from sigil.harness import (
    AgentRunner,
    EventType,
    JsonlTraceWriter,
    MemoryTraceWriter,
    ModelRequest,
    ModelResponse,
    NoInput,
    ReplayDivergence,
    ReplayModel,
    RunConfig,
    ScriptedModel,
    Tool,
    ToolContext,
    ToolRegistry,
    TraceEvent,
    TraceRecorder,
    TraceWriter,
    load_trace,
    replay,
    tool,
)
from tests.harness.helpers import (
    FakeModel,
    MinValueGuardrail,
    call,
    make_submit_tool,
    reply,
)

SYSTEM = "You are an investigator."


def lookup_tool(answer: str = "alpha") -> Tool[Any]:
    @tool(name="lookup", description="Look something up.")
    async def lookup(args: NoInput, ctx: ToolContext) -> dict[str, str]:
        return {"answer": answer}

    return lookup


def registry(answer: str = "alpha", extra_tools: bool = True) -> ToolRegistry:
    tools: list[Tool[Any]] = [make_submit_tool()]
    if extra_tools:
        tools.append(lookup_tool(answer))
    return ToolRegistry(tools)


async def record(
    *, guardrails: Sequence[MinValueGuardrail] = (), path: Path | None = None
) -> list[TraceEvent]:
    model = FakeModel(
        [
            reply(call("c1", "lookup")),
            reply(call("c2", "submit", value=1)),
            reply(call("c3", "submit", value=20)),
        ],
        name="claude-opus-5-5",
    )
    sink = MemoryTraceWriter()
    writers: list[TraceWriter] = [sink] if path is None else [sink, JsonlTraceWriter(path)]
    runner = AgentRunner(
        model=model,
        tools=registry(),
        system=SYSTEM,
        guardrails=guardrails,
        trace=TraceRecorder("recorded", writers),
        config=RunConfig(),
    )
    await runner.run("investigate scan 7")
    return sink.events


async def test_replaying_an_unchanged_run_is_identical() -> None:
    events = await record(guardrails=[MinValueGuardrail(10)])
    report = await replay(
        events, tools=registry(), system=SYSTEM, guardrails=[MinValueGuardrail(10)]
    )
    assert report.divergences == []
    assert report.identical
    assert report.original_status == report.replay_status == "completed"


async def test_replay_from_a_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    await record(path=path)
    report = await replay(load_trace(path), tools=registry(), system=SYSTEM)
    assert report.identical


async def test_changed_tool_output_is_reported_as_a_result_divergence() -> None:
    events = await record()
    report = await replay(events, tools=registry(answer="beta"), system=SYSTEM)

    assert not report.identical
    assert "step 1 tool lookup result differs" in report.divergences
    # The tool's new output also changes what the model would have seen at the next step.
    assert "step 2 request differs from the recording" in report.divergences
    assert report.replay_status == "completed"  # recorded model actions are replayed regardless


async def test_stricter_guardrail_rejects_the_recorded_submission() -> None:
    events = await record()
    report = await replay(
        events, tools=registry(), system=SYSTEM, guardrails=[MinValueGuardrail(10)]
    )

    assert not report.identical
    assert "guardrail outcomes differ" in report.divergences
    # The recorded model never saw the rejection, so the recording runs out of responses.
    assert "status changed from completed to error" in report.divergences
    assert any("more times than the 2 recorded" in d for d in report.divergences)


async def test_changed_system_prompt_is_reported() -> None:
    report = await replay(await record(), tools=registry(), system="A different prompt.")
    assert "system prompt differs from the recording" in report.divergences
    assert "step 1 request differs from the recording" in report.divergences


async def test_removed_tool_stops_the_replay_and_says_why() -> None:
    report = await replay(await record(), tools=registry(extra_tools=False), system=SYSTEM)
    assert not report.identical
    assert report.replay_status == "error"
    assert any("differ from recorded" in d for d in report.divergences)
    assert report.original_status == "completed"


async def test_tampered_trace_is_refused() -> None:
    events = await record()
    events[3] = events[3].model_copy(update={"data": {"content": "forged"}})
    with pytest.raises(ValueError, match="hash-chain"):
        await replay(events, tools=registry(), system=SYSTEM)


async def test_replay_model_serves_recorded_responses_then_raises() -> None:
    events = await record(guardrails=[MinValueGuardrail(10)])
    model = ReplayModel(events)
    request = ModelRequest(SYSTEM, (), registry().specs(), 100)

    assert model.name == "claude-opus-5-5"
    first = await model.complete(request)
    await model.complete(request)
    await model.complete(request)
    assert first.tool_calls[0].name == "lookup"
    with pytest.raises(ReplayDivergence, match="more times than the 3 recorded"):
        await model.complete(request)


async def test_replay_model_raises_when_tool_names_differ() -> None:
    model = ReplayModel(await record())
    request = ModelRequest(SYSTEM, (), registry(extra_tools=False).specs(), 100)
    with pytest.raises(ReplayDivergence, match="differ from recorded"):
        await model.complete(request)


async def test_replay_model_notes_request_divergence_without_failing() -> None:
    model = ReplayModel(await record())
    await model.complete(ModelRequest(SYSTEM, (), registry().specs(), 100))
    assert model.divergences == ["step 1 request differs from the recording"]


def test_replay_model_needs_a_run_started_event() -> None:
    with pytest.raises(ValueError, match="run_started"):
        ReplayModel([])


async def test_replaying_a_run_that_errored_reproduces_the_error() -> None:
    sink = MemoryTraceWriter()
    runner = AgentRunner(
        model=FakeModel([reply(call("c1", "lookup")), RuntimeError("503")]),
        tools=registry(),
        system=SYSTEM,
        trace=TraceRecorder("recorded", [sink]),
    )
    await runner.run("task")
    report = await replay(sink.events, tools=registry(), system=SYSTEM)
    assert report.original_status == report.replay_status == "error"
    assert report.identical


async def test_replay_works_for_scripted_reference_runs() -> None:
    def policy(request: ModelRequest) -> ModelResponse:
        done = any(m.role == "assistant" for m in request.messages)
        return reply(call("s", "submit", value=3) if done else call("l", "lookup"))

    sink = MemoryTraceWriter()
    runner = AgentRunner(
        model=ScriptedModel("reference", policy),
        tools=registry(),
        system=SYSTEM,
        trace=TraceRecorder("ref", [sink]),
    )
    result = await runner.run("task")
    assert result.cost_usd == 0.0
    assert (await replay(sink.events, tools=registry(), system=SYSTEM)).identical
    assert sink.events[-1].type == EventType.RUN_FINISHED
