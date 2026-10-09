"""Claude via the official ``anthropic`` SDK (Messages API, beta surface)."""

from __future__ import annotations

from typing import Any

import anthropic

from sigil.harness.types import (
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

PROVIDER = "anthropic"
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}


class AnthropicModel:
    """``Model`` backed by ``client.beta.messages.create``.

    The client is created lazily, so constructing a model (or importing this module) needs no
    API key. ``max_retries`` configures the SDK's own 429/5xx retry loop and applies only when
    the client is created here, not when one is injected. Errors propagate to the runner.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        effort: str = "medium",
        client: Any | None = None,
        server_fallbacks: bool = True,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._effort = effort
        self._client = client
        self._server_fallbacks = server_fallbacks
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return self._model

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self._get_client().beta.messages.create(**self.build_kwargs(request))
        return _to_model_response(response)

    def build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        """Keyword arguments for ``client.beta.messages.create``.

        Forced ``tool_choice`` is deliberately absent: current models reject it, so the system
        prompt and the runner's nudge steer the model towards the terminal tool instead.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "system": [{"type": "text", "text": request.system}],
            "tools": [_tool_param(spec) for spec in request.tools],
            "messages": [_message_param(m) for m in request.messages],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self._server_fallbacks:
            kwargs["betas"] = [SERVER_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(max_retries=self._max_retries)
        return self._client


# --- harness -> API ----------------------------------------------------------------------------


def _tool_param(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
        "strict": True,
    }


def _message_param(message: Message) -> dict[str, Any]:
    return {"role": message.role, "content": [_block_param(b) for b in message.blocks]}


def _block_param(block: Block) -> dict[str, Any]:
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ToolCall():
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        case ToolResult():
            return {
                "type": "tool_result",
                "tool_use_id": block.call_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        case OpaqueBlock():
            if block.provider != PROVIDER:
                raise ValueError(f"cannot send a {block.provider!r} block to Anthropic")
            return block.payload  # thinking blocks must be echoed back unchanged


# --- API -> harness ----------------------------------------------------------------------------


def _to_model_response(response: Any) -> ModelResponse:
    usage = response.usage
    return ModelResponse(
        blocks=tuple(_from_api_block(b) for b in response.content),
        stop_reason=_STOP_REASONS.get(response.stop_reason, StopReason.OTHER),
        usage=Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        ),
        model=response.model,
    )


def _from_api_block(block: Any) -> Block:
    match block.type:
        case "text":
            return TextBlock(block.text)
        case "tool_use":
            return ToolCall(id=block.id, name=block.name, input=dict(block.input))
        case _:  # thinking, redacted_thinking, fallback, server tool blocks...
            return OpaqueBlock(PROVIDER, block.model_dump(mode="json", exclude_none=True))
