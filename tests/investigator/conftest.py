from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from sigil.inference import InferenceService
from tests.support import World, build_world, running_inference


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path, units=6)


@pytest.fixture
async def inference() -> AsyncIterator[InferenceService]:
    async with running_inference() as service:
        yield service
