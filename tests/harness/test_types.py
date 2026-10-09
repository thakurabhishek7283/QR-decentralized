from __future__ import annotations

import json

import pytest

from sigil.harness import (
    Block,
    Message,
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
from sigil.harness.types import block_from_json

THINKING = OpaqueBlock("anthropic", {"type": "thinking", "thinking": "hm", "signature": "s=="})


def test_blocks_round_trip_through_json() -> None:
    blocks: list[Block] = [
        TextBlock("héllo"),
        ToolCall("c1", "echo", {"value": 1, "nested": {"a": [1, 2]}}),
        ToolResult("c1", "ok", is_error=True),
        THINKING,
    ]
    for block in blocks:
        assert block_from_json(json.loads(json.dumps(block.to_json()))) == block


def test_unknown_block_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown block type"):
        block_from_json({"type": "hologram"})


def test_request_and_response_round_trip() -> None:
    request = ModelRequest(
        system="sys",
        messages=(
            Message.user_text("task"),
            Message("assistant", (THINKING, ToolCall("c1", "echo", {"value": 1}))),
            Message("user", (ToolResult("c1", "{}"),)),
        ),
        tools=(ToolSpec("echo", "d", {"type": "object"}),),
        max_tokens=100,
    )
    response = ModelResponse(
        blocks=(THINKING, TextBlock("t"), ToolCall("c2", "echo", {})),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(1, 2, 3, 4),
        model="m",
    )
    assert ModelRequest.from_json(json.loads(json.dumps(request.to_json()))) == request
    assert ModelResponse.from_json(json.loads(json.dumps(response.to_json()))) == response


def test_response_accessors() -> None:
    call = ToolCall("c", "echo", {})
    response = ModelResponse(
        (TextBlock("a"), call, THINKING, TextBlock("b")), StopReason.TOOL_USE, Usage(), "m"
    )
    assert response.tool_calls == [call]
    assert response.text == "ab"


def test_usage_arithmetic() -> None:
    total = Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40)
    assert total == Usage(11, 22, 33, 44)
    assert total.total == 110
