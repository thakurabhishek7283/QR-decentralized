"""Semantic checks on terminal tool calls.

The tool registry validates the *shape* of a call; guardrails judge its
*meaning* (are the citations real, is the verdict allowed by policy?).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sigil.harness.types import Message, ToolCall, ToolResult


@dataclass(frozen=True)
class RunView:
    """Read-only snapshot of the run handed to guardrails."""

    messages: tuple[Message, ...]
    tool_results: tuple[tuple[ToolCall, ToolResult], ...]
    state: Mapping[str, Any]


class Guardrail(Protocol):
    name: str

    def check(self, call: ToolCall, run: RunView) -> list[str]:
        """Return human-readable violations; an empty list accepts the call."""
        ...
