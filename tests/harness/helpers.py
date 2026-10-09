"""Shared test doubles for the harness tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict

from sigil.harness import (
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    Tool,
    ToolCall,
    ToolContext,
    ToolRegistry,
    Usage,
    tool,
)


class Echo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class Submit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def call(call_id: str, name: str, **kwargs: Any) -> ToolCall:
    return ToolCall(call_id, name, kwargs)


def reply(
    *blocks: TextBlock | ToolCall,
    stop: StopReason | None = None,
    usage: Usage | None = None,
    model: str = "fake-model",
) -> ModelResponse:
    if stop is None:
        stop = (
            StopReason.TOOL_USE
            if any(isinstance(b, ToolCall) for b in blocks)
            else StopReason.END_TURN
        )
    return ModelResponse(
        blocks=tuple(blocks), stop_reason=stop, usage=usage or Usage(), model=model
    )


class FakeModel:
    """Plays back queued responses (or raises queued exceptions) and records every request."""

    def __init__(
        self, script: Iterable[ModelResponse | Exception], name: str = "fake-model"
    ) -> None:
        self._script = list(script)
        self._name = name
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) > len(self._script):
            raise AssertionError("FakeModel script exhausted")
        item = self._script[len(self.requests) - 1]
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_echo_tool(name: str = "echo", **options: Any) -> Tool[Any]:
    @tool(name=name, description="Echo the value back.", **options)
    async def echo(args: Echo, ctx: ToolContext) -> dict[str, int]:
        return {"value": args.value}

    return echo


def make_submit_tool(name: str = "submit") -> Tool[Any]:
    @tool(name=name, description="Submit the final answer.", terminal=True)
    async def submit(args: Submit, ctx: ToolContext) -> str:
        return "accepted"

    return submit


def make_registry(*extra: Tool[Any]) -> ToolRegistry:
    return ToolRegistry([make_echo_tool(), make_submit_tool(), *extra])


class MinValueGuardrail:
    """Rejects submissions whose value is below a threshold."""

    name = "min_value"

    def __init__(self, minimum: int) -> None:
        self.minimum = minimum
        self.calls = 0

    def check(self, call: ToolCall, run: Any) -> list[str]:
        self.calls += 1
        value = call.input["value"]
        return [f"value {value} is below {self.minimum}"] if value < self.minimum else []
