"""Typed tools: pydantic input models, strict JSON schemas, and a registry that never raises."""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from sigil.harness.types import ToolCall, ToolResult, ToolSpec

MAX_RESULT_BYTES = 16 * 1024
MAX_ERROR_CHARS = 500

type ToolFn = Callable[[Any, ToolContext], Awaitable[Any]]


class NoInput(BaseModel):
    # Input model for tools that take no arguments (a comment, not a docstring: pydantic would
    # copy a docstring into the schema the model sees).
    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    """Per-run context shared by all tool calls."""

    run_id: str
    step: int
    state: dict[str, Any]  # scratch space tools use to coordinate (and guardrails read)
    deps: Any  # domain dependencies injected by the caller of ``AgentRunner.run``
    call_counts: Counter[str] = field(default_factory=Counter)


# --- JSON schema -------------------------------------------------------------------------------

# Keywords Anthropic's ``strict: true`` tools accept. Anything else is folded into the description
# so the model still sees the constraint (pydantic enforces it when the call arrives).
_SUPPORTED = frozenset({"type", "enum", "const", "description"})
_IGNORED = frozenset({"title", "default", "examples", "$defs"})


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for ``model`` in the restricted dialect strict tool use requires.

    ``$ref``s are inlined, every object is closed (``additionalProperties: false``) with all of
    its properties required, and unsupported constraints (``minLength``, ``maximum``...) move
    into the description.
    """
    schema = model.model_json_schema()
    return _simplify(schema, schema.get("$defs", {}), ())


def _simplify(
    node: dict[str, Any], defs: dict[str, Any], active: tuple[str, ...]
) -> dict[str, Any]:
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name in active:
            raise ValueError(f"recursive schema {name!r} is not supported by strict tools")
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        return _simplify({**defs[name], **siblings}, defs, (*active, name))

    out: dict[str, Any] = {}
    unsupported: dict[str, Any] = {}
    for key, value in node.items():
        if key in _IGNORED:
            continue
        if key == "properties":
            out[key] = {k: _simplify(v, defs, active) for k, v in value.items()}
        elif key == "items":
            out[key] = _simplify(value, defs, active)
        elif key == "anyOf":
            out[key] = [_simplify(v, defs, active) for v in value]
        elif key in _SUPPORTED or (key == "minItems" and value in (0, 1)):
            out[key] = value
        elif key not in {"required", "additionalProperties"}:
            unsupported[key] = value

    if out.get("type") == "object":
        out["properties"] = out.get("properties", {})
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if unsupported:
        hint = "{" + ", ".join(f"{k}: {v}" for k, v in unsupported.items()) + "}"
        out["description"] = f"{out['description']}\n\n{hint}" if "description" in out else hint
    return out


# --- Tool and decorator ------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool[InT: BaseModel]:
    name: str
    description: str
    input_model: type[InT]
    fn: Callable[[InT, ToolContext], Awaitable[Any]]
    timeout_s: float = 20.0
    terminal: bool = False  # calling it ends the run, subject to guardrails
    max_calls: int | None = None  # per-run cap; excess calls get an error result

    def __post_init__(self) -> None:
        if self.input_model.model_config.get("extra") != "forbid":
            raise ValueError(
                f"tool {self.name!r}: {self.input_model.__name__} must set extra='forbid' "
                "so its schema and its validation agree"
            )

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, strict_json_schema(self.input_model))


def _input_model_of(fn: ToolFn) -> type[BaseModel]:
    params = list(inspect.signature(fn).parameters)
    hint = typing.get_type_hints(fn).get(params[0]) if params else None
    if not (inspect.isclass(hint) and issubclass(hint, BaseModel)):
        raise TypeError(f"{fn.__name__}: first parameter must be annotated with a pydantic model")
    return hint


def tool(
    *,
    name: str,
    description: str,
    timeout_s: float = 20.0,
    terminal: bool = False,
    max_calls: int | None = None,
) -> Callable[[ToolFn], Tool[Any]]:
    """Turn ``async def f(args: SomeModel, ctx: ToolContext) -> Any`` into a ``Tool``."""

    def decorate(fn: ToolFn) -> Tool[Any]:
        return Tool(
            name=name,
            description=description,
            input_model=_input_model_of(fn),
            fn=fn,
            timeout_s=timeout_s,
            terminal=terminal,
            max_calls=max_calls,
        )

    return decorate


# --- Registry ----------------------------------------------------------------------------------


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool[Any]) -> None:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name {t.name!r}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Sorted by name so the tools block is byte-stable across runs (prompt caching)."""
        return tuple(self._tools[name].spec() for name in sorted(self._tools))

    def terminal_names(self) -> list[str]:
        return sorted(name for name, t in self._tools.items() if t.terminal)

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Run ``call``. Every failure mode becomes an ``is_error`` result; this never raises."""
        t = self._tools.get(call.name)
        if t is None:
            available = ", ".join(sorted(self._tools))
            return _error(call, f"unknown tool {call.name!r}; available: {available}")
        try:
            args = t.input_model.model_validate(call.input)
        except ValidationError as exc:
            return _error(call, f"invalid input: {_format_validation(exc)}")
        if t.max_calls is not None and ctx.call_counts[t.name] >= t.max_calls:
            return _error(call, f"tool {t.name!r} is limited to {t.max_calls} call(s) per run")
        ctx.call_counts[t.name] += 1
        return await _invoke(t, args, call, ctx)


async def _invoke(t: Tool[Any], args: BaseModel, call: ToolCall, ctx: ToolContext) -> ToolResult:
    deadline = asyncio.timeout(t.timeout_s)
    try:
        async with deadline:
            value = await t.fn(args, ctx)
        return ToolResult(call.id, _truncate(_render(value)))
    except TimeoutError as exc:
        # Distinguish our deadline from a TimeoutError the tool itself raised.
        message = f"timed out after {t.timeout_s:g}s" if deadline.expired() else _describe(exc)
        return _error(call, message)
    except Exception as exc:  # tools are untrusted; the model gets the message, never a traceback
        return _error(call, _describe(exc))


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call.id, message, is_error=True)


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


def _format_validation(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or '<input>'}: {e['msg']}" for e in errors)


def _json_default(value: object) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, sort_keys=True, default=_json_default)


def _truncate(text: str) -> str:
    raw = text.encode()
    if len(raw) <= MAX_RESULT_BYTES:
        return text
    kept = raw[:MAX_RESULT_BYTES].decode(errors="ignore")
    return f"{kept}\n[truncated: {len(raw) - MAX_RESULT_BYTES} more bytes omitted]"
