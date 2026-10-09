"""A deterministic, offline ``Model``: a pure function from request to response."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from sigil.harness.types import ModelRequest, ModelResponse, Usage

type Policy = Callable[[ModelRequest], ModelResponse]


class ScriptedModel:
    """Wraps a policy as a ``Model``.

    Scripted runs are free by definition, so usage is forced to zero and the response is
    attributed to ``name``; policies only decide the blocks and stop reason.
    """

    def __init__(self, name: str, policy: Policy) -> None:
        self._name = name
        self._policy = policy

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return replace(self._policy(request), usage=Usage(), model=self._name)
