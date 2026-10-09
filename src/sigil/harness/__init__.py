"""A small, domain-agnostic agent harness.

Design in one paragraph. ``AgentRunner`` drives a tool-using model in a loop. The conversation
history is **append-only** (nothing is edited, so provider prompt caches and reasoning blocks stay
valid) and all tool results of a turn go back in **one** message, in call order, with the calls
executed concurrently. **Budgets** (steps, tokens, dollars, wall-clock) are checked before every
model call. The run ends only through a **terminal tool**, and that call is vetted by
**guardrails** first: a rejected submission is returned to the model as an error to fix, up to a
retry limit. Every step is written to a hash-chained **trace**; a recorded trace can be
**replayed** against changed tool code with no model and no cost, which turns agent behaviour
into a regression test.

This package imports nothing from the rest of ``sigil``.
"""

from sigil.harness.budget import Budget, BudgetTracker
from sigil.harness.guardrails import Guardrail, RunView
from sigil.harness.pricing import PRICES, ModelPrice, cost_usd
from sigil.harness.providers.anthropic import AnthropicModel
from sigil.harness.providers.scripted import ScriptedModel
from sigil.harness.replay import ReplayDivergence, ReplayModel, ReplayReport, replay
from sigil.harness.runner import AgentRunner, RunConfig, RunResult, RunStatus
from sigil.harness.tools import NoInput, Tool, ToolContext, ToolRegistry, tool
from sigil.harness.trace import (
    EventType,
    JsonlTraceWriter,
    MemoryTraceWriter,
    TraceEvent,
    TraceRecorder,
    TraceSink,
    TraceSummary,
    TraceWriter,
    load_trace,
    summarize,
    verify_chain,
)
from sigil.harness.types import (
    Block,
    Message,
    Model,
    ModelRequest,
    ModelResponse,
    OpaqueBlock,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__all__ = [
    "PRICES",
    "AgentRunner",
    "AnthropicModel",
    "Block",
    "Budget",
    "BudgetTracker",
    "EventType",
    "Guardrail",
    "JsonlTraceWriter",
    "MemoryTraceWriter",
    "Message",
    "Model",
    "ModelPrice",
    "ModelRequest",
    "ModelResponse",
    "NoInput",
    "OpaqueBlock",
    "ReplayDivergence",
    "ReplayModel",
    "ReplayReport",
    "RunConfig",
    "RunResult",
    "RunStatus",
    "RunView",
    "ScriptedModel",
    "StopReason",
    "TextBlock",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "TraceEvent",
    "TraceRecorder",
    "TraceSink",
    "TraceSummary",
    "TraceWriter",
    "Usage",
    "cost_usd",
    "load_trace",
    "replay",
    "summarize",
    "tool",
    "verify_chain",
]
