from __future__ import annotations

from sigil.harness import ModelRequest, ModelResponse, ScriptedModel, StopReason, Usage
from tests.harness.helpers import reply


async def test_scripted_model_applies_policy_and_normalises_usage() -> None:
    seen: list[ModelRequest] = []

    def policy(request: ModelRequest) -> ModelResponse:
        seen.append(request)
        return reply(usage=Usage(999, 999), model="whatever")

    model = ScriptedModel("reference", policy)
    request = ModelRequest("sys", (), (), 10)
    response = await model.complete(request)

    assert model.name == "reference"
    assert seen == [request]
    assert response.model == "reference"
    assert response.usage == Usage()
    assert response.stop_reason is StopReason.END_TURN
