from __future__ import annotations

import asyncio
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from sigil.harness import NoInput, Tool, ToolCall, ToolContext, ToolRegistry, tool
from sigil.harness.tools import MAX_RESULT_BYTES, strict_json_schema
from tests.harness.helpers import Echo, call, make_echo_tool, make_submit_tool


def ctx() -> ToolContext:
    return ToolContext(run_id="r", step=1, state={}, deps=None)


# --- schema --------------------------------------------------------------------------------------


class Inner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(ge=0, le=5, description="a small number")


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["GENUINE", "COUNTERFEIT"]
    confidence: float = Field(ge=0, le=1)
    cited: list[str] = Field(min_length=1, max_length=12)
    rationale: str = Field(min_length=20, max_length=1200)
    inner: Inner
    maybe: Inner | None = None
    note: str = "default"


def test_schema_is_closed_and_requires_every_property() -> None:
    schema = strict_json_schema(Verdict)
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(schema["properties"])
    assert "note" in schema["required"]  # fields with defaults are still required in strict mode


def test_schema_inlines_refs_and_closes_nested_objects() -> None:
    schema = strict_json_schema(Verdict)
    assert "$defs" not in str(schema)
    assert "$ref" not in str(schema)
    inner = schema["properties"]["inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["n"]
    assert schema["properties"]["maybe"]["anyOf"][1] == {"type": "null"}


def test_schema_moves_unsupported_constraints_into_the_description() -> None:
    props = strict_json_schema(Verdict)["properties"]
    assert props["confidence"] == {
        "type": "number",
        "description": "{maximum: 1, minimum: 0}",
    }
    assert "maxLength: 1200" in props["rationale"]["description"]
    assert "minLength" in props["rationale"]["description"]
    assert props["cited"]["minItems"] == 1  # minItems 0/1 is supported natively
    assert props["inner"]["properties"]["n"]["description"].startswith("a small number")


def test_schema_drops_titles_and_defaults() -> None:
    schema = strict_json_schema(Verdict)
    assert "title" not in schema
    assert "default" not in schema["properties"]["note"]
    assert "title" not in schema["properties"]["note"]


def test_property_named_like_a_keyword_survives() -> None:
    class Odd(BaseModel):
        model_config = ConfigDict(extra="forbid")
        title: str
        default: int

    schema = strict_json_schema(Odd)
    assert set(schema["properties"]) == {"title", "default"}


def test_recursive_models_are_rejected() -> None:
    class Node(BaseModel):
        model_config = ConfigDict(extra="forbid")
        children: list[Node]

    with pytest.raises(ValueError, match="recursive"):
        strict_json_schema(Node)


def test_no_input_schema() -> None:
    assert strict_json_schema(NoInput) == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_tool_decorator_derives_model_and_spec() -> None:
    echo = make_echo_tool(timeout_s=3, max_calls=2)
    assert isinstance(echo, Tool)
    assert echo.input_model is Echo
    assert (echo.timeout_s, echo.max_calls, echo.terminal) == (3, 2, False)
    spec = echo.spec()
    assert spec.name == "echo"
    assert spec.input_schema["properties"] == {"value": {"type": "integer"}}


def test_decorator_requires_a_model_annotation() -> None:
    with pytest.raises(TypeError, match="pydantic model"):

        @tool(name="bad", description="d")
        async def bad(args: int, ctx: ToolContext) -> None: ...


class Lax(BaseModel):
    x: int


def test_tool_input_model_must_forbid_extras() -> None:
    with pytest.raises(ValueError, match="extra='forbid'"):

        @tool(name="lax", description="d")
        async def lax(args: Lax, ctx: ToolContext) -> None: ...


# --- registry ------------------------------------------------------------------------------------


def test_duplicate_registration_is_an_error() -> None:
    registry = ToolRegistry([make_echo_tool()])
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(make_echo_tool())


def test_specs_are_sorted_by_name() -> None:
    registry = ToolRegistry([make_echo_tool("zeta"), make_echo_tool("alpha"), make_submit_tool()])
    assert [s.name for s in registry.specs()] == ["alpha", "submit", "zeta"]
    assert registry.terminal_names() == ["submit"]


async def test_success_returns_json_with_sorted_keys() -> None:
    @tool(name="multi", description="d")
    async def multi(args: NoInput, ctx: ToolContext) -> dict[str, Any]:
        return {"b": 1, "a": 2}

    result = await ToolRegistry([multi]).execute(call("c1", "multi"), ctx())
    assert (result.call_id, result.content, result.is_error) == ("c1", '{"a": 2, "b": 1}', False)


async def test_pydantic_models_and_strings_are_rendered() -> None:
    @tool(name="model_out", description="d")
    async def model_out(args: NoInput, ctx: ToolContext) -> Echo:
        return Echo(value=7)

    @tool(name="str_out", description="d")
    async def str_out(args: NoInput, ctx: ToolContext) -> str:
        return "plain text"

    @tool(name="nested", description="d")
    async def nested(args: NoInput, ctx: ToolContext) -> list[Echo]:
        return [Echo(value=1)]

    registry = ToolRegistry([model_out, str_out, nested])
    assert (await registry.execute(call("1", "model_out"), ctx())).content == '{"value":7}'
    assert (await registry.execute(call("2", "str_out"), ctx())).content == "plain text"
    assert (await registry.execute(call("3", "nested"), ctx())).content == '[{"value": 1}]'


async def test_unknown_tool_lists_available_tools() -> None:
    registry = ToolRegistry([make_echo_tool(), make_submit_tool()])
    result = await registry.execute(call("c", "nope"), ctx())
    assert result.is_error
    assert "unknown tool 'nope'" in result.content
    assert "available: echo, submit" in result.content


async def test_validation_errors_are_compact() -> None:
    registry = ToolRegistry([make_echo_tool()])
    missing = await registry.execute(call("c", "echo"), ctx())
    wrong_type = await registry.execute(call("c", "echo", value="abc"), ctx())
    extra = await registry.execute(call("c", "echo", value=1, bogus=2), ctx())
    assert all(r.is_error for r in (missing, wrong_type, extra))
    assert "invalid input: value: Field required" in missing.content
    assert "value:" in wrong_type.content
    assert "bogus: Extra inputs are not permitted" in extra.content
    assert "pydantic.dev" not in missing.content  # no doc URLs


async def test_timeout_becomes_an_error_result() -> None:
    @tool(name="slow", description="d", timeout_s=0.01)
    async def slow(args: NoInput, ctx: ToolContext) -> None:
        await asyncio.sleep(5)

    result = await ToolRegistry([slow]).execute(call("c", "slow"), ctx())
    assert result.is_error
    assert result.content == "timed out after 0.01s"


async def test_a_timeout_raised_by_the_tool_is_not_reported_as_our_deadline() -> None:
    @tool(name="inner", description="d", timeout_s=5)
    async def inner(args: NoInput, ctx: ToolContext) -> None:
        raise TimeoutError("upstream took too long")

    result = await ToolRegistry([inner]).execute(call("c", "inner"), ctx())
    assert result.is_error
    assert result.content == "TimeoutError: upstream took too long"


async def test_exceptions_become_short_error_results_without_traceback() -> None:
    @tool(name="boom", description="d")
    async def boom(args: NoInput, ctx: ToolContext) -> None:
        raise RuntimeError("x" * 2000)

    result = await ToolRegistry([boom]).execute(call("c", "boom"), ctx())
    assert result.is_error
    assert result.content.startswith("RuntimeError: xxx")
    assert len(result.content) == 500
    assert "Traceback" not in result.content


async def test_unserialisable_return_value_is_an_error_not_a_crash() -> None:
    @tool(name="bad_out", description="d")
    async def bad_out(args: NoInput, ctx: ToolContext) -> object:
        return {"x": object()}

    result = await ToolRegistry([bad_out]).execute(call("c", "bad_out"), ctx())
    assert result.is_error
    assert result.content.startswith("TypeError")


async def test_max_calls_cap_is_per_run_context() -> None:
    registry = ToolRegistry([make_echo_tool(max_calls=2)])
    first, second = ctx(), ctx()
    results = [await registry.execute(call(str(i), "echo", value=i), first) for i in range(3)]
    assert [r.is_error for r in results] == [False, False, True]
    assert "limited to 2 call(s) per run" in results[2].content
    assert not (await registry.execute(call("x", "echo", value=1), second)).is_error


async def test_invalid_calls_do_not_consume_the_cap() -> None:
    registry = ToolRegistry([make_echo_tool(max_calls=1)])
    context = ctx()
    assert (await registry.execute(call("a", "echo"), context)).is_error
    assert not (await registry.execute(call("b", "echo", value=1), context)).is_error


async def test_large_results_are_truncated_at_16kb() -> None:
    @tool(name="big", description="d")
    async def big(args: NoInput, ctx: ToolContext) -> str:
        return "é" * 20_000  # 2 bytes each

    result = await ToolRegistry([big]).execute(call("c", "big"), ctx())
    body, _, marker = result.content.partition("\n[truncated")
    assert not result.is_error
    assert len(body.encode()) <= MAX_RESULT_BYTES
    assert marker.startswith(": ")


async def test_execute_never_raises_even_for_malformed_calls() -> None:
    registry = ToolRegistry([make_echo_tool()])
    result = await registry.execute(ToolCall("c", "echo", {"value": object()}), ctx())
    assert result.is_error
