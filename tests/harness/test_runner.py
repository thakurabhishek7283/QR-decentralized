from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sigil.harness import (
    AgentRunner,
    Budget,
    EventType,
    MemoryTraceWriter,
    Message,
    NoInput,
    RunConfig,
    RunStatus,
    RunView,
    StopReason,
    TextBlock,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
    TraceRecorder,
    Usage,
    tool,
    verify_chain,
)
from tests.harness.helpers import (
    Echo,
    FakeClock,
    FakeModel,
    MinValueGuardrail,
    call,
    make_echo_tool,
    make_registry,
    reply,
)


def traced_runner(
    model: FakeModel, registry: ToolRegistry | None = None, **kwargs: Any
) -> tuple[AgentRunner, MemoryTraceWriter]:
    sink = MemoryTraceWriter()
    runner = AgentRunner(
        model=model,
        tools=registry or make_registry(),
        system="You are a test agent.",
        trace=TraceRecorder("run-1", [sink]),
        **kwargs,
    )
    return runner, sink


def event_types(sink: MemoryTraceWriter) -> list[EventType]:
    return [e.type for e in sink.events]


# --- happy path ----------------------------------------------------------------------------------


async def test_happy_path_runs_tool_calls_concurrently_and_keeps_order() -> None:
    a_started, b_started = asyncio.Event(), asyncio.Event()

    @tool(name="a", description="waits for b", timeout_s=2)
    async def a(args: NoInput, ctx: ToolContext) -> str:
        a_started.set()
        await b_started.wait()  # only returns if b runs at the same time
        return "a-done"

    @tool(name="b", description="waits for a", timeout_s=2)
    async def b(args: NoInput, ctx: ToolContext) -> str:
        b_started.set()
        await a_started.wait()
        return "b-done"

    model = FakeModel(
        [
            reply(TextBlock("checking"), call("ca", "a"), call("cb", "b")),
            reply(call("cs", "submit", value=42)),
        ]
    )
    runner, sink = traced_runner(model, make_registry(a, b))
    result = await runner.run("do the thing")

    assert result.status is RunStatus.COMPLETED
    assert result.output == {"value": 42}
    assert result.terminal_tool == "submit"
    assert result.steps == 2

    # task, assistant(calls), user(results), assistant(submit), user(ack)
    roles = [m.role for m in result.messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    results_message = result.messages[2]
    assert results_message.blocks == (
        ToolResult("ca", "a-done"),
        ToolResult("cb", "b-done"),
    )
    assert result.messages[4].blocks == (ToolResult("cs", "accepted"),)
    assert event_types(sink)[0] is EventType.RUN_STARTED
    assert event_types(sink)[-1] is EventType.RUN_FINISHED
    assert verify_chain(sink.events)


async def test_history_is_append_only_across_requests() -> None:
    model = FakeModel(
        [
            reply(call("c1", "echo", value=1)),
            reply(call("c2", "echo", value=2)),
            reply(call("c3", "submit", value=3)),
        ]
    )
    runner, _ = traced_runner(model)
    await runner.run("task")
    first, second, third = (r.messages for r in model.requests)
    assert second[: len(first)] == first
    assert third[: len(second)] == second
    assert [len(m) for m in (first, second, third)] == [1, 3, 5]


async def test_sequential_mode_runs_one_tool_at_a_time() -> None:
    log: list[str] = []

    def make(name: str) -> Any:
        @tool(name=name, description="d")
        async def t(args: NoInput, ctx: ToolContext) -> str:
            log.append(f"{name}-start")
            await asyncio.sleep(0.01)
            log.append(f"{name}-end")
            return name

        return t

    model = FakeModel([reply(call("1", "x"), call("2", "y")), reply(call("3", "submit", value=1))])
    runner, _ = traced_runner(
        model, make_registry(make("x"), make("y")), config=RunConfig(parallel_tool_calls=False)
    )
    await runner.run("task")
    assert log == ["x-start", "x-end", "y-start", "y-end"]


async def test_tool_errors_are_fed_back_to_the_model_not_raised() -> None:
    model = FakeModel(
        [
            reply(call("c1", "does_not_exist"), call("c2", "echo", value="not-an-int")),
            reply(call("c3", "submit", value=1)),
        ]
    )
    runner, _ = traced_runner(model)
    result = await runner.run("task")
    blocks = result.messages[2].blocks
    assert [b.is_error for b in blocks if isinstance(b, ToolResult)] == [True, True]
    assert result.status is RunStatus.COMPLETED


async def test_usage_and_cost_accumulate_per_model_price() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    model = FakeModel(
        [
            reply(call("c1", "echo", value=1), usage=usage, model="claude-opus-5-5"),
            reply(call("c2", "submit", value=1), usage=usage, model="claude-opus-5-5"),
        ]
    )
    runner, _ = traced_runner(
        model, config=RunConfig(budget=Budget(max_cost_usd=100, max_total_tokens=10**9))
    )
    result = await runner.run("task")
    assert result.usage == Usage(2_000_000, 200_000)
    assert result.cost_usd == pytest.approx(2 * (4.0 + 2.0))


async def test_tools_share_state_with_guardrails_through_the_run_view() -> None:
    @tool(name="reveal", description="d")
    async def reveal(args: NoInput, ctx: ToolContext) -> str:
        ctx.state.setdefault("seen", []).append("e1")
        return "e1"

    class CitesSeen:
        name = "cites_seen"

        def check(self, call: ToolCall, run: RunView) -> list[str]:
            assert any(c.name == "reveal" for c, _ in run.tool_results)
            return [] if call.input["value"] == 1 and "e1" in run.state["seen"] else ["bad"]

    model = FakeModel([reply(call("c1", "reveal")), reply(call("c2", "submit", value=1))])
    runner, _ = traced_runner(model, make_registry(reveal), guardrails=[CitesSeen()])
    assert (await runner.run("task")).status is RunStatus.COMPLETED


async def test_tool_receives_deps_and_run_context() -> None:
    seen: dict[str, Any] = {}

    @tool(name="peek", description="d")
    async def peek(args: NoInput, ctx: ToolContext) -> str:
        seen.update(deps=ctx.deps, run_id=ctx.run_id, step=ctx.step)
        return "ok"

    model = FakeModel([reply(call("c1", "peek")), reply(call("c2", "submit", value=1))])
    runner, _ = traced_runner(model, make_registry(peek))
    await runner.run("task", deps="DEPS")
    assert seen == {"deps": "DEPS", "run_id": "run-1", "step": 1}


# --- nudges --------------------------------------------------------------------------------------


async def test_model_that_forgets_to_finish_is_nudged_once() -> None:
    model = FakeModel(
        [reply(TextBlock("I think it's fine.")), reply(call("c1", "submit", value=1))]
    )
    runner, sink = traced_runner(model)
    result = await runner.run("task")

    assert result.status is RunStatus.COMPLETED
    nudge = result.messages[2]
    assert nudge.role == "user"
    assert isinstance(nudge.blocks[0], TextBlock)
    assert "terminal tools (submit)" in nudge.blocks[0].text
    assert EventType.NUDGE in event_types(sink)


async def test_exhausted_nudges_end_with_no_terminal() -> None:
    model = FakeModel([reply(TextBlock("a")), reply(TextBlock("b"))])
    runner, _ = traced_runner(model)
    result = await runner.run("task")
    assert result.status is RunStatus.NO_TERMINAL
    assert result.output is None
    assert result.steps == 2


async def test_zero_nudges_fails_immediately() -> None:
    runner, _ = traced_runner(FakeModel([reply(TextBlock("a"))]), config=RunConfig(max_nudges=0))
    result = await runner.run("task")
    assert (result.status, result.steps) == (RunStatus.NO_TERMINAL, 1)


# --- guardrails ----------------------------------------------------------------------------------


async def test_rejected_terminal_call_is_retried_then_accepted() -> None:
    guardrail = MinValueGuardrail(10)
    model = FakeModel([reply(call("c1", "submit", value=1)), reply(call("c2", "submit", value=11))])
    runner, sink = traced_runner(model, guardrails=[guardrail])
    result = await runner.run("task")

    assert result.status is RunStatus.COMPLETED
    assert result.output == {"value": 11}
    assert result.guardrail_violations == ["min_value: value 1 is below 10"]
    rejection = result.messages[2].blocks[0]
    assert isinstance(rejection, ToolResult)
    assert rejection.is_error
    assert rejection.content.startswith("Rejected by guardrails: min_value: value 1 is below 10")
    assert "call submit again" in rejection.content
    verdicts = [e.data["accepted"] for e in sink.events if e.type is EventType.GUARDRAIL]
    assert verdicts == [False, True]


async def test_guardrail_failure_after_max_retries() -> None:
    guardrail = MinValueGuardrail(10)
    model = FakeModel([reply(call(f"c{i}", "submit", value=1)) for i in range(5)])
    runner, _ = traced_runner(
        model, guardrails=[guardrail], config=RunConfig(max_guardrail_retries=2)
    )
    result = await runner.run("task")

    assert result.status is RunStatus.GUARDRAIL_FAILED
    assert result.output is None
    assert result.steps == 3  # the first attempt plus two retries
    assert len(result.guardrail_violations) == 3
    last = result.messages[-1].blocks[0]
    assert isinstance(last, ToolResult)
    assert last.is_error  # history stays well-formed


async def test_a_crashing_guardrail_fails_closed() -> None:
    class Broken:
        name = "broken"

        def check(self, call: ToolCall, run: RunView) -> list[str]:
            raise RuntimeError("oops")

    model = FakeModel([reply(call("c1", "submit", value=1))])
    runner, _ = traced_runner(
        model, guardrails=[Broken()], config=RunConfig(max_guardrail_retries=0)
    )
    result = await runner.run("task")
    assert result.status is RunStatus.GUARDRAIL_FAILED
    assert "check crashed (RuntimeError: oops)" in result.guardrail_violations[0]


async def test_guardrails_run_only_on_terminal_calls() -> None:
    guardrail = MinValueGuardrail(0)
    model = FakeModel([reply(call("c1", "echo", value=-5)), reply(call("c2", "submit", value=1))])
    runner, _ = traced_runner(model, guardrails=[guardrail])
    await runner.run("task")
    assert guardrail.calls == 1


async def test_invalid_terminal_input_is_a_tool_error_not_a_guardrail_pass() -> None:
    guardrail = MinValueGuardrail(0)
    model = FakeModel([reply(call("c1", "submit")), reply(call("c2", "submit", value=1))])
    runner, _ = traced_runner(model, guardrails=[guardrail])
    result = await runner.run("task")
    assert result.status is RunStatus.COMPLETED
    assert guardrail.calls == 1  # the malformed call never reached the guardrails
    first = result.messages[2].blocks[0]
    assert isinstance(first, ToolResult)
    assert first.is_error


async def test_only_the_first_terminal_call_in_a_turn_is_honoured() -> None:
    executed: list[str] = []

    @tool(name="submit_other", description="d", terminal=True)
    async def submit_other(args: Echo, ctx: ToolContext) -> str:
        executed.append("other")
        return "accepted"

    model = FakeModel(
        [
            reply(
                call("c1", "submit", value=1),
                call("c2", "submit_other", value=2),
                call("c3", "submit", value=3),
            )
        ]
    )
    runner, _ = traced_runner(model, make_registry(submit_other))
    result = await runner.run("task")

    assert result.status is RunStatus.COMPLETED
    assert result.output == {"value": 1}
    assert result.terminal_tool == "submit"
    assert executed == []
    results = result.messages[2].blocks
    assert [(r.call_id, r.is_error) for r in results if isinstance(r, ToolResult)] == [
        ("c1", False),
        ("c2", True),
        ("c3", True),
    ]
    assert "Only one terminal call" in results[1].content  # type: ignore[union-attr]


async def test_extra_terminal_call_is_ignored_even_if_the_first_is_rejected() -> None:
    model = FakeModel(
        [
            reply(call("c1", "submit", value=1), call("c2", "submit", value=100)),
            reply(call("c3", "submit", value=100)),
        ]
    )
    runner, _ = traced_runner(model, guardrails=[MinValueGuardrail(10)])
    result = await runner.run("task")
    assert result.status is RunStatus.COMPLETED
    assert result.steps == 2
    assert result.output == {"value": 100}


async def test_non_terminal_tools_in_the_terminal_turn_still_run() -> None:
    model = FakeModel([reply(call("c1", "echo", value=5), call("c2", "submit", value=1))])
    runner, _ = traced_runner(model)
    result = await runner.run("task")
    blocks = result.messages[-1].blocks
    assert [b.call_id for b in blocks if isinstance(b, ToolResult)] == ["c1", "c2"]


# --- budgets -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "usage", "model_name", "reason"),
    [
        (Budget(max_steps=2), Usage(), "fake-model", "steps"),
        (Budget(max_total_tokens=150), Usage(100, 50), "fake-model", "tokens"),
        (Budget(max_cost_usd=1.0), Usage(output_tokens=100_000), "claude-opus-5-5", "cost"),
    ],
)
async def test_budget_exhaustion(
    budget: Budget, usage: Usage, model_name: str, reason: str
) -> None:
    model = FakeModel(
        [reply(call(f"c{i}", "echo", value=i), usage=usage, model=model_name) for i in range(10)]
    )
    runner, sink = traced_runner(model, config=RunConfig(budget=budget))
    result = await runner.run("task")

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.budget_reason == reason
    assert result.output is None
    finished = sink.events[-1].data
    assert finished["status"] == "budget_exhausted"
    assert finished["budget_reason"] == reason
    assert len(model.requests) <= 2  # stopped as soon as the limit was crossed


async def test_wall_clock_budget_with_a_fake_clock() -> None:
    clock = FakeClock()

    @tool(name="slow", description="d")
    async def slow(args: NoInput, ctx: ToolContext) -> str:
        clock.advance(60)
        return "done"

    model = FakeModel([reply(call(f"c{i}", "slow")) for i in range(5)])
    runner, _ = traced_runner(
        model, make_registry(slow), config=RunConfig(budget=Budget(max_wall_s=100)), clock=clock
    )
    result = await runner.run("task")

    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.budget_reason == "wall"
    assert result.steps == 2
    assert result.wall_s == 120


async def test_budget_exhaustion_keeps_partial_history() -> None:
    model = FakeModel([reply(call(f"c{i}", "echo", value=i)) for i in range(10)])
    runner, _ = traced_runner(model, config=RunConfig(budget=Budget(max_steps=1)))
    result = await runner.run("task")
    assert [m.role for m in result.messages] == ["user", "assistant", "user"]


# --- refusal, errors -----------------------------------------------------------------------------


async def test_refusal() -> None:
    model = FakeModel([reply(TextBlock("I can't help with that."), stop=StopReason.REFUSAL)])
    runner, _ = traced_runner(model)
    result = await runner.run("task")
    assert result.status is RunStatus.REFUSED
    assert result.messages[-1].role == "assistant"  # the refusal is kept in the history


async def test_model_exception_becomes_error_status() -> None:
    model = FakeModel([RuntimeError("503 overloaded")])
    runner, sink = traced_runner(model)
    result = await runner.run("task")

    assert result.status is RunStatus.ERROR
    assert result.error == "RuntimeError: 503 overloaded"
    assert EventType.ERROR in event_types(sink)
    assert sink.events[-1].data["error"] == result.error


# --- trace and construction ----------------------------------------------------------------------


async def test_trace_events_have_the_documented_shape() -> None:
    model = FakeModel([reply(call("c1", "echo", value=1)), reply(call("c2", "submit", value=2))])
    runner, sink = traced_runner(model, config=RunConfig(max_tokens_per_turn=123))
    result = await runner.run("the task")

    started = sink.events[0].data
    assert started["task"] == "the task"
    assert started["model"] == "fake-model"
    assert len(started["system_sha256"]) == 64
    assert [t["name"] for t in started["tools"]] == ["echo", "submit"]
    assert started["config"]["max_tokens_per_turn"] == 123

    response = next(e.data for e in sink.events if e.type is EventType.MODEL_RESPONSE)
    assert response["stop_reason"] == "tool_use"
    assert response["blocks"][0]["name"] == "echo"
    assert {"usage", "model", "latency_ms", "cost_usd"} <= response.keys()

    tool_result = next(e.data for e in sink.events if e.type is EventType.TOOL_RESULT)
    assert {"content", "is_error", "latency_ms", "call_id", "name", "step"} <= tool_result.keys()

    finished = sink.events[-1].data
    assert finished["status"] == "completed"
    assert finished["output"] == {"value": 2}
    assert finished["steps"] == result.steps == 2
    assert {"usage", "cost_usd", "wall_s"} <= finished.keys()


async def test_runner_without_a_trace_still_works() -> None:
    model = FakeModel([reply(call("c1", "submit", value=1))])
    runner = AgentRunner(model=model, tools=make_registry(), system="s")
    result = await runner.run("task")
    assert result.status is RunStatus.COMPLETED
    assert len(result.run_id) == 32


def test_registry_must_contain_a_terminal_tool() -> None:
    with pytest.raises(ValueError, match="terminal tool"):
        AgentRunner(model=FakeModel([]), tools=ToolRegistry([make_echo_tool()]), system="s")


def test_run_id_must_match_the_trace() -> None:
    with pytest.raises(ValueError, match="does not match"):
        AgentRunner(
            model=FakeModel([]),
            tools=make_registry(),
            system="s",
            trace=TraceRecorder("a"),
            run_id="b",
        )


def test_run_config_json_round_trip() -> None:
    config = RunConfig(budget=Budget(max_steps=3, max_cost_usd=0.5), max_nudges=0)
    assert RunConfig.from_json(config.to_json()) == config


def test_message_user_text_helper() -> None:
    assert Message.user_text("x") == Message("user", (TextBlock("x"),))
