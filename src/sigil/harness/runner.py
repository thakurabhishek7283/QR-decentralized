"""The agent loop.

One turn is: check budgets, call the model, append its reply, execute the requested tools
(concurrently), append *all* results as a single user message. The history is append-only:
nothing is edited or dropped, which keeps provider prompt caches warm and keeps providers that
bind reasoning blocks to an unedited transcript happy.

A run ends when the model calls a *terminal* tool whose input passes every guardrail, or when a
budget, refusal, error or retry limit stops it. Every outcome is a ``RunResult``; ``run`` does
not raise for anything the model or a tool does.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from sigil.harness.budget import Budget, BudgetTracker
from sigil.harness.guardrails import Guardrail, RunView
from sigil.harness.pricing import cost_usd
from sigil.harness.tools import ToolContext, ToolRegistry
from sigil.harness.trace import EventType, TraceRecorder, TraceSink, digest
from sigil.harness.types import (
    Message,
    Model,
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)


@dataclass(frozen=True)
class RunConfig:
    budget: Budget = field(default_factory=Budget)
    max_tokens_per_turn: int = 8_000
    parallel_tool_calls: bool = True
    max_guardrail_retries: int = 2
    max_nudges: int = 1  # re-prompts when the model ends its turn without a terminal call

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RunConfig:
        return cls(**{**data, "budget": Budget(**data["budget"])})


class RunStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REFUSED = "refused"
    NO_TERMINAL = "no_terminal"
    GUARDRAIL_FAILED = "guardrail_failed"
    ERROR = "error"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    output: dict[str, Any] | None  # the accepted terminal tool's input
    terminal_tool: str | None
    steps: int
    usage: Usage
    cost_usd: float
    wall_s: float
    guardrail_violations: list[str]
    error: str | None
    budget_reason: str | None
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class _Outcome:
    status: RunStatus
    output: dict[str, Any] | None = None
    terminal_tool: str | None = None
    error: str | None = None
    budget_reason: str | None = None


@dataclass
class _RunState:
    messages: list[Message]
    tracker: BudgetTracker
    ctx: ToolContext
    nudges: int = 0
    guardrail_retries: int = 0
    violations: list[str] = field(default_factory=list)
    tool_results: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)

    @property
    def step(self) -> int:
        return self.tracker.steps


class AgentRunner:
    """Runs one agent to completion. Create a new runner per run: it owns the run id and trace."""

    def __init__(
        self,
        *,
        model: Model,
        tools: ToolRegistry,
        system: str,
        guardrails: Sequence[Guardrail] = (),
        config: RunConfig | None = None,
        trace: TraceSink | None = None,
        run_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not tools.terminal_names():
            raise ValueError("the tool registry needs at least one terminal tool")
        if trace is not None and run_id is not None and trace.run_id != run_id:
            raise ValueError(f"run_id {run_id!r} does not match the trace's {trace.run_id!r}")
        self._model = model
        self._tools = tools
        self._system = system
        self._guardrails = tuple(guardrails)
        self._config = config or RunConfig()
        self.run_id = trace.run_id if trace is not None else (run_id or uuid.uuid4().hex)
        self._trace: TraceSink = trace if trace is not None else TraceRecorder(self.run_id)
        self._clock = clock

    async def run(self, task: str, *, deps: Any = None) -> RunResult:
        state = _RunState(
            messages=[Message.user_text(task)],
            tracker=BudgetTracker(self._config.budget, self._clock),
            ctx=ToolContext(run_id=self.run_id, step=0, state={}, deps=deps),
        )
        self._trace.emit(
            EventType.RUN_STARTED,
            {
                "task": task,
                "model": self._model.name,
                "system_sha256": hashlib.sha256(self._system.encode()).hexdigest(),
                "tools": [spec.to_json() for spec in self._tools.specs()],
                "config": self._config.to_json(),
            },
        )
        outcome = await self._loop(state)
        return self._finish(state, outcome)

    # --- loop ----------------------------------------------------------------------------------

    async def _loop(self, state: _RunState) -> _Outcome:
        while True:
            if reason := state.tracker.exceeded():
                return _Outcome(RunStatus.BUDGET_EXHAUSTED, budget_reason=reason)
            state.tracker.next_step()
            state.ctx.step = state.step
            try:
                response = await self._ask_model(state)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._trace.emit(EventType.ERROR, {"step": state.step, "message": message})
                return _Outcome(RunStatus.ERROR, error=message)

            state.messages.append(Message("assistant", response.blocks))
            if response.stop_reason is StopReason.REFUSAL:
                return _Outcome(RunStatus.REFUSED)

            calls = response.tool_calls
            if not calls:
                if state.nudges >= self._config.max_nudges:
                    return _Outcome(RunStatus.NO_TERMINAL)
                self._nudge(state)
                continue

            if outcome := await self._run_tools(state, calls):
                return outcome

    async def _ask_model(self, state: _RunState) -> ModelResponse:
        request = ModelRequest(
            system=self._system,
            messages=tuple(state.messages),
            tools=self._tools.specs(),
            max_tokens=self._config.max_tokens_per_turn,
        )
        self._trace.emit(
            EventType.MODEL_REQUEST,
            {
                "step": state.step,
                "request_digest": digest(request.to_json()),
                "message_count": len(request.messages),
            },
        )
        started = self._clock()
        response = await self._model.complete(request)
        cost = cost_usd(response.model, response.usage)
        state.tracker.record(response.usage, cost)
        self._trace.emit(
            EventType.MODEL_RESPONSE,
            {
                **response.to_json(),
                "step": state.step,
                "latency_ms": _ms(self._clock() - started),
                "cost_usd": cost,
                "budget": state.tracker.snapshot(),
            },
        )
        return response

    def _nudge(self, state: _RunState) -> None:
        state.nudges += 1
        names = ", ".join(self._tools.terminal_names())
        text = (
            f"You must finish by calling one of the terminal tools ({names}). "
            "Respond with a tool call, not plain text."
        )
        state.messages.append(Message.user_text(text))
        self._trace.emit(EventType.NUDGE, {"step": state.step, "count": state.nudges})

    # --- tools ---------------------------------------------------------------------------------

    async def _run_tools(self, state: _RunState, calls: list[ToolCall]) -> _Outcome | None:
        first_terminal = next((i for i, c in enumerate(calls) if self._is_terminal(c)), None)
        for call in calls:
            self._trace.emit(
                EventType.TOOL_CALL,
                {"step": state.step, "call_id": call.id, "name": call.name, "input": call.input},
            )
        timed = await self._execute_all(state, calls, first_terminal)
        results = [result for result, _ in timed]
        for call, (result, latency_ms) in zip(calls, timed, strict=True):
            self._trace.emit(
                EventType.TOOL_RESULT,
                {
                    "step": state.step,
                    "call_id": call.id,
                    "name": call.name,
                    "content": result.content,
                    "is_error": result.is_error,
                    "latency_ms": latency_ms,
                },
            )
            if not self._is_terminal(call):
                state.tool_results.append((call, result))

        outcome = None
        if first_terminal is not None:
            outcome, results[first_terminal] = self._judge_terminal(
                state, calls[first_terminal], results[first_terminal]
            )
        state.messages.append(Message("user", tuple(results)))
        return outcome

    async def _execute_all(
        self, state: _RunState, calls: list[ToolCall], first_terminal: int | None
    ) -> list[tuple[ToolResult, float]]:
        async def run_one(index: int, call: ToolCall) -> tuple[ToolResult, float]:
            if self._is_terminal(call) and index != first_terminal:
                rejection = "Only one terminal call per turn is honoured; this one was ignored."
                return ToolResult(call.id, rejection, is_error=True), 0.0
            started = self._clock()
            result = await self._tools.execute(call, state.ctx)
            return result, _ms(self._clock() - started)

        if self._config.parallel_tool_calls:
            return list(await asyncio.gather(*(run_one(i, c) for i, c in enumerate(calls))))
        return [await run_one(i, c) for i, c in enumerate(calls)]

    def _is_terminal(self, call: ToolCall) -> bool:
        tool = self._tools.get(call.name)
        return tool is not None and tool.terminal

    # --- guardrails ----------------------------------------------------------------------------

    def _judge_terminal(
        self, state: _RunState, call: ToolCall, result: ToolResult
    ) -> tuple[_Outcome | None, ToolResult]:
        """Accept or reject the terminal call. Returns the run outcome (if decided) and the
        result the model should see."""
        if result.is_error:  # failed validation or hit its cap: the model just tries again
            return None, result
        violations = self._violations(state, call)
        self._trace.emit(
            EventType.GUARDRAIL,
            {
                "step": state.step,
                "call_id": call.id,
                "tool": call.name,
                "violations": violations,
                "accepted": not violations,
            },
        )
        if not violations:
            return _Outcome(RunStatus.COMPLETED, call.input, call.name), result

        state.violations.extend(violations)
        state.guardrail_retries += 1
        listed = "; ".join(violations)
        rejection = ToolResult(
            call.id, f"Rejected by guardrails: {listed}. Fix this and call {call.name} again.", True
        )
        if state.guardrail_retries > self._config.max_guardrail_retries:
            return _Outcome(RunStatus.GUARDRAIL_FAILED), rejection
        return None, rejection

    def _violations(self, state: _RunState, call: ToolCall) -> list[str]:
        view = RunView(
            messages=tuple(state.messages),
            tool_results=tuple(state.tool_results),
            state=MappingProxyType(state.ctx.state),
        )
        found: list[str] = []
        for guardrail in self._guardrails:
            try:
                found += [f"{guardrail.name}: {v}" for v in guardrail.check(call, view)]
            except Exception as exc:  # a broken guardrail must fail closed, not open
                found.append(f"{guardrail.name}: check crashed ({type(exc).__name__}: {exc})")
        return found

    # --- finish --------------------------------------------------------------------------------

    def _finish(self, state: _RunState, outcome: _Outcome) -> RunResult:
        tracker = state.tracker
        result = RunResult(
            run_id=self.run_id,
            status=outcome.status,
            output=outcome.output,
            terminal_tool=outcome.terminal_tool,
            steps=tracker.steps,
            usage=tracker.usage,
            cost_usd=tracker.cost_usd,
            wall_s=tracker.elapsed_s,
            guardrail_violations=state.violations,
            error=outcome.error,
            budget_reason=outcome.budget_reason,
            messages=tuple(state.messages),
        )
        self._trace.emit(
            EventType.RUN_FINISHED,
            {
                "status": result.status.value,
                "output": result.output,
                "terminal_tool": result.terminal_tool,
                "usage": result.usage.to_json(),
                "cost_usd": result.cost_usd,
                "steps": result.steps,
                "wall_s": result.wall_s,
                "error": result.error,
                "budget_reason": result.budget_reason,
                "guardrail_violations": result.guardrail_violations,
            },
        )
        return result


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)
