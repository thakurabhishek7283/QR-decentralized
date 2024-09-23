"""One real app per test package: lifespan running, ONNX model loaded, demo batches seeded.

The apps share an event loop for the whole package because the micro-batcher's worker task is
bound to the loop it started on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sigil.api import create_app
from sigil.api.container import Container
from sigil.config import Settings

API_KEY = "test-key"
AUTH = {"X-API-Key": API_KEY}
LOOP = "package"


@dataclass
class Api:
    """A running app, its HTTP client and its container."""

    app: FastAPI
    client: AsyncClient

    @property
    def container(self) -> Container:
        container: Container = self.app.state.container
        return container

    async def issue(self, count: int = 3, sku: str = "TEST-SKU", **extra: Any) -> list[str]:
        response = await self.client.post(
            "/v1/batches", json={"sku": sku, "count": count, **extra}, headers=AUTH
        )
        assert response.status_code == 201, response.text
        serials: list[str] = response.json()["serials"]
        return serials

    async def photograph(self, serial: str, **body: Any) -> bytes:
        """A simulated photo of an issued label."""
        response = await self.client.post(f"/v1/simulate/{serial}", json=body)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/jpeg"
        return response.content

    async def scan(self, photo: bytes, **params: Any) -> dict[str, Any]:
        response = await self.client.post(
            "/v1/scans", params=params, files={"image": ("photo.jpg", photo, "image/jpeg")}
        )
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body

    async def scan_until(self, serial: str, wanted: set[str], **body: Any) -> dict[str, Any]:
        """Photograph and scan until the verdict is one of ``wanted``.

        Simulated shots are random and some are unreadable, as with a real phone, so a test that
        needs one particular outcome retakes the photo with a fresh seed.
        """
        for seed in range(12):
            photo = await self.photograph(serial, seed=seed, difficulty="easy", **body)
            result = await self.scan(photo)
            if result["verdict"] in wanted:
                return result
        raise AssertionError(f"no shot of {serial} gave one of {wanted}")


async def start(settings: Settings, *, seed_demo: bool) -> AsyncIterator[Api]:
    app = create_app(settings, seed_demo=seed_demo)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://sigil.test") as client,
    ):
        yield Api(app, client)


@pytest_asyncio.fixture(scope="package", loop_scope=LOOP)
async def api(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[Api]:
    """The default deployment, with the demo batches seeded."""
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("api"), api_key=API_KEY, qc_samples_per_batch=3
    )
    async for running in start(settings, seed_demo=True):
        yield running


@pytest_asyncio.fixture(scope="package", loop_scope=LOOP)
async def unsure_api(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[Api]:
    """A deployment whose thresholds make every readable genuine-looking scan ambiguous."""
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("unsure"),
        api_key=API_KEY,
        qc_samples_per_batch=3,
        cdp_genuine_min=1.0,
        cdp_counterfeit_max=0.0,
    )
    async for running in start(settings, seed_demo=False):
        yield running


@pytest_asyncio.fixture(loop_scope=LOOP)
async def fresh_api(tmp_path: Path) -> AsyncIterator[Api]:
    """An empty deployment for tests that change global state (key revocation)."""
    settings = Settings(data_dir=tmp_path, api_key=API_KEY, qc_samples_per_batch=2)
    async for running in start(settings, seed_demo=False):
        yield running
