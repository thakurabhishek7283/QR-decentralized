from __future__ import annotations

from typing import Any

import pytest
from anthropic.types.beta import (
    BetaMessage,
    BetaRedactedThinkingBlock,
    BetaTextBlock,
    BetaThinkingBlock,
    BetaToolUseBlock,
    BetaUsage,
)

from sigil.harness import (
    AgentRunner,
    AnthropicModel,
    Message,
    ModelRequest,
    OpaqueBlock,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from tests.harness.helpers import make_registry

THINKING = BetaThinkingBlock(type="thinking", thinking="let me think", signature="c2lnbmF0dXJl")


def api_message(
    *content: Any,
    stop_reason: str = "end_turn",
    usage: BetaUsage | None = None,
    model: str = "claude-opus-5-5",
) -> BetaMessage:
    return BetaMessage(
        id="msg_1",
        type="message",
        role="assistant",
        content=list(content),
        model=model,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        usage=usage or BetaUsage(input_tokens=10, output_tokens=5),
    )


class _Messages:
    def __init__(self, replies: list[BetaMessage | Exception]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, *replies: BetaMessage | Exception) -> None:
        self.messages = _Messages(list(replies))
        self.beta = self

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


SPEC = ToolSpec(
    "lookup", "Look up.", {"type": "object", "properties": {}, "additionalProperties": False}
)


def request(*messages: Message, system: str = "Be careful.") -> ModelRequest:
    return ModelRequest(system, messages or (Message.user_text("hi"),), (SPEC,), 4000)


# --- request shape -------------------------------------------------------------------------------


async def test_request_kwargs_shape() -> None:
    client = FakeClient(api_message(BetaTextBlock(type="text", text="ok")))
    model = AnthropicModel(client=client, effort="high")
    await model.complete(request())

    (kwargs,) = client.calls
    assert kwargs == {
        "model": "claude-opus-5-5",
        "max_tokens": 4000,
        "system": [{"type": "text", "text": "Be careful."}],
        "tools": [
            {
                "name": "lookup",
                "description": "Look up.",
                "input_schema": SPEC.input_schema,
                "strict": True,
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "cache_control": {"type": "ephemeral"},
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }


async def test_forced_tool_choice_is_never_sent() -> None:
    client = FakeClient(api_message(BetaTextBlock(type="text", text="ok")))
    await AnthropicModel(client=client).complete(request())
    assert "tool_choice" not in client.calls[0]


async def test_server_fallbacks_can_be_disabled() -> None:
    client = FakeClient(api_message(BetaTextBlock(type="text", text="ok")))
    await AnthropicModel(client=client, server_fallbacks=False).complete(request())
    assert "betas" not in client.calls[0]
    assert "fallbacks" not in client.calls[0]


async def test_default_model_and_name() -> None:
    assert AnthropicModel(client=FakeClient()).name == "claude-opus-5-5"
    assert AnthropicModel("claude-haiku-5-5", client=FakeClient()).name == "claude-haiku-5-5"


def test_every_kwarg_is_a_typed_parameter_of_the_installed_sdk() -> None:
    """Guards the wiring decision: nothing needs to go through extra_body/extra_headers."""
    import inspect

    from anthropic.resources.beta.messages import AsyncMessages

    accepted = set(inspect.signature(AsyncMessages.create).parameters)
    sent = set(AnthropicModel(client=FakeClient()).build_kwargs(request()))
    assert sent <= accepted


# --- message conversion --------------------------------------------------------------------------


async def test_messages_are_converted_with_tool_results_and_errors() -> None:
    client = FakeClient(api_message(BetaTextBlock(type="text", text="ok")))
    history = (
        Message.user_text("task"),
        Message("assistant", (TextBlock("calling"), ToolCall("toolu_1", "lookup", {"q": "x"}))),
        Message(
            "user",
            (ToolResult("toolu_1", '{"a":1}'), ToolResult("toolu_2", "boom", is_error=True)),
        ),
    )
    await AnthropicModel(client=client).complete(request(*history))

    assert client.calls[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": '{"a":1}',
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": "boom",
                    "is_error": True,
                },
            ],
        },
    ]


async def test_opaque_blocks_from_another_provider_are_rejected() -> None:
    bad = Message("assistant", (OpaqueBlock("openai", {"type": "reasoning"}),))
    with pytest.raises(ValueError, match="openai"):
        await AnthropicModel(client=FakeClient()).complete(request(Message.user_text("x"), bad))


# --- response conversion -------------------------------------------------------------------------


async def test_thinking_blocks_round_trip_verbatim() -> None:
    redacted = BetaRedactedThinkingBlock(type="redacted_thinking", data="ZW5jcnlwdGVk")
    tool_use = BetaToolUseBlock(type="tool_use", id="toolu_9", name="lookup", input={"q": "x"})
    client = FakeClient(
        api_message(
            THINKING,
            redacted,
            BetaTextBlock(type="text", text="hm"),
            tool_use,
            stop_reason="tool_use",
        ),
        api_message(BetaTextBlock(type="text", text="done")),
    )
    model = AnthropicModel(client=client)

    first = await model.complete(request())
    assert first.blocks == (
        OpaqueBlock(
            "anthropic",
            {"type": "thinking", "thinking": "let me think", "signature": "c2lnbmF0dXJl"},
        ),
        OpaqueBlock("anthropic", {"type": "redacted_thinking", "data": "ZW5jcnlwdGVk"}),
        TextBlock("hm"),
        ToolCall("toolu_9", "lookup", {"q": "x"}),
    )
    assert first.stop_reason is StopReason.TOOL_USE

    history = (
        Message.user_text("hi"),
        Message("assistant", first.blocks),
        Message("user", (ToolResult("toolu_9", "{}"),)),
    )
    await model.complete(request(*history))
    echoed = client.calls[1]["messages"][1]["content"]
    assert echoed[0] == THINKING.model_dump(mode="json", exclude_none=True)
    assert echoed[1] == {"type": "redacted_thinking", "data": "ZW5jcnlwdGVk"}
    assert echoed[2:] == [
        {"type": "text", "text": "hm"},
        {"type": "tool_use", "id": "toolu_9", "name": "lookup", "input": {"q": "x"}},
    ]


async def test_unknown_block_types_are_kept_opaque() -> None:
    class Future:
        type = "fallback"

        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"mode": "json", "exclude_none": True}
            return {"type": "fallback", "from": "a", "to": "b"}

    message = api_message()
    message.content.append(Future())  # type: ignore[arg-type]
    response = await AnthropicModel(client=FakeClient(message)).complete(request())
    assert response.blocks == (
        OpaqueBlock("anthropic", {"type": "fallback", "from": "a", "to": "b"}),
    )


@pytest.mark.parametrize(
    ("api_reason", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("tool_use", StopReason.TOOL_USE),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
        ("stop_sequence", StopReason.OTHER),
        ("pause_turn", StopReason.OTHER),
    ],
)
async def test_stop_reason_mapping(api_reason: str, expected: StopReason) -> None:
    client = FakeClient(api_message(BetaTextBlock(type="text", text="x"), stop_reason=api_reason))
    assert (await AnthropicModel(client=client).complete(request())).stop_reason is expected


async def test_usage_mapping_including_cache_tokens() -> None:
    usage = BetaUsage(
        input_tokens=11,
        output_tokens=22,
        cache_read_input_tokens=33,
        cache_creation_input_tokens=44,
    )
    client = FakeClient(api_message(BetaTextBlock(type="text", text="x"), usage=usage))
    response = await AnthropicModel(client=client).complete(request())
    assert response.usage == Usage(11, 22, 33, 44)


async def test_missing_cache_counts_are_zero() -> None:
    client = FakeClient(api_message(BetaTextBlock(type="text", text="x")))
    response = await AnthropicModel(client=client).complete(request())
    assert response.usage == Usage(10, 5, 0, 0)


async def test_response_reports_the_model_that_actually_answered() -> None:
    """With server-side fallbacks the answering model can differ; cost must follow it."""
    client = FakeClient(
        api_message(BetaTextBlock(type="text", text="x"), model="claude-sonnet-5-5")
    )
    assert (await AnthropicModel(client=client).complete(request())).model == "claude-sonnet-5-5"


async def test_api_errors_propagate_for_the_runner_to_handle() -> None:
    client = FakeClient(RuntimeError("overloaded"))
    with pytest.raises(RuntimeError, match="overloaded"):
        await AnthropicModel(client=client).complete(request())


# --- client construction -------------------------------------------------------------------------


def test_client_is_created_lazily_with_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr("anthropic.AsyncAnthropic", StubClient)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    model = AnthropicModel(max_retries=7)
    assert created == []  # constructing needs no key and builds no client
    assert isinstance(model._get_client(), StubClient)
    model._get_client()
    assert created == [{"max_retries": 7}]


def test_construction_without_api_key_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert AnthropicModel().name == "claude-opus-5-5"


# --- end to end with the runner ------------------------------------------------------------------


async def test_runner_drives_the_adapter_end_to_end() -> None:
    client = FakeClient(
        api_message(
            THINKING,
            BetaToolUseBlock(type="tool_use", id="toolu_1", name="submit", input={"value": 3}),
            stop_reason="tool_use",
            usage=BetaUsage(input_tokens=1_000_000, output_tokens=0),
        )
    )
    runner = AgentRunner(model=AnthropicModel(client=client), tools=make_registry(), system="s")
    result = await runner.run("task")

    assert result.output == {"value": 3}
    assert result.cost_usd == pytest.approx(4.0)  # claude-opus-5-5 input price
    sent_tools = client.calls[0]["tools"]
    assert [t["name"] for t in sent_tools] == ["echo", "submit"]
    assert all(
        t["strict"] and t["input_schema"]["additionalProperties"] is False for t in sent_tools
    )
