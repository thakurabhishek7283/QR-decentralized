"""Provider-neutral message and model types.

Everything here is a frozen value object with a lossless ``to_json`` /
``from_json`` pair, because the trace stores serialised responses and replay
feeds them back to the runner unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TextBlock:
        return cls(text=data["text"])


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"type": "tool_call", "id": self.id, "name": self.name, "input": self.input}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ToolCall:
        return cls(id=data["id"], name=data["name"], input=dict(data["input"]))


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "call_id": self.call_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ToolResult:
        return cls(
            call_id=data["call_id"], content=data["content"], is_error=data.get("is_error", False)
        )


@dataclass(frozen=True, slots=True)
class OpaqueBlock:
    """A provider-specific block (e.g. extended thinking) that must be echoed back verbatim."""

    provider: str
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"type": "opaque", "provider": self.provider, "payload": self.payload}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> OpaqueBlock:
        return cls(provider=data["provider"], payload=dict(data["payload"]))


Block = TextBlock | ToolCall | ToolResult | OpaqueBlock


def block_from_json(data: Mapping[str, Any]) -> Block:
    match data["type"]:
        case "text":
            return TextBlock.from_json(data)
        case "tool_call":
            return ToolCall.from_json(data)
        case "tool_result":
            return ToolResult.from_json(data)
        case "opaque":
            return OpaqueBlock.from_json(data)
    raise ValueError(f"unknown block type {data['type']!r}")


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    blocks: tuple[Block, ...]

    def to_json(self) -> dict[str, Any]:
        return {"role": self.role, "blocks": [b.to_json() for b in self.blocks]}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Message:
        return cls(role=data["role"], blocks=tuple(block_from_json(b) for b in data["blocks"]))

    @classmethod
    def user_text(cls, text: str) -> Message:
        return cls(role="user", blocks=(TextBlock(text),))


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts. ``input_tokens`` excludes cached tokens, as in the Anthropic API."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Usage:
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ToolSpec:
        return cls(
            name=data["name"],
            description=data["description"],
            input_schema=dict(data["input_schema"]),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    max_tokens: int

    def to_json(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "messages": [m.to_json() for m in self.messages],
            "tools": [t.to_json() for t in self.tools],
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ModelRequest:
        return cls(
            system=data["system"],
            messages=tuple(Message.from_json(m) for m in data["messages"]),
            tools=tuple(ToolSpec.from_json(t) for t in data["tools"]),
            max_tokens=data["max_tokens"],
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    blocks: tuple[Block, ...]  # TextBlock | ToolCall | OpaqueBlock
    stop_reason: StopReason
    usage: Usage
    model: str

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.blocks if isinstance(b, ToolCall)]

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    def to_json(self) -> dict[str, Any]:
        return {
            "blocks": [b.to_json() for b in self.blocks],
            "stop_reason": self.stop_reason.value,
            "usage": self.usage.to_json(),
            "model": self.model,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ModelResponse:
        return cls(
            blocks=tuple(block_from_json(b) for b in data["blocks"]),
            stop_reason=StopReason(data["stop_reason"]),
            usage=Usage.from_json(data["usage"]),
            model=data["model"],
        )


class Model(Protocol):
    """A chat model. Implementations own transport retries; errors propagate to the runner."""

    @property
    def name(self) -> str: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...
