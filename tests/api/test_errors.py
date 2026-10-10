"""Upload limits, error mapping and the operational endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from sigil.api.middleware import MAX_BODY_BYTES
from sigil.errors import OverloadedError
from tests.api.conftest import Api

pytestmark = pytest.mark.asyncio(loop_scope="package")


async def test_oversized_uploads_are_rejected_before_they_are_read(api: Api) -> None:
    big = b"\0" * (MAX_BODY_BYTES + 1024)
    response = await api.client.post("/v1/scans", files={"image": ("big.jpg", big, "image/jpeg")})
    assert response.status_code == 413


async def test_a_chunked_upload_cannot_dodge_the_limit(api: Api) -> None:
    """No Content-Length, so the limit has to be enforced as the body streams in."""
    header = (
        b'--x\r\nContent-Disposition: form-data; name="image"; filename="a.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
    )

    async def body() -> AsyncIterator[bytes]:
        yield header
        for _ in range(MAX_BODY_BYTES // 65536 + 2):
            yield b"\0" * 65536

    response = await api.client.post(
        "/v1/scans", content=body(), headers={"content-type": "multipart/form-data; boundary=x"}
    )
    assert response.status_code == 413


async def test_a_file_that_is_not_an_image_is_unprocessable(api: Api) -> None:
    response = await api.client.post(
        "/v1/scans", files={"image": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 422
    assert response.json()["reason"] == "bad_image"


async def test_scan_request_validation(api: Api) -> None:
    photo = {"image": ("a.jpg", b"x", "image/jpeg")}
    post = api.client.post
    assert (await post("/v1/scans")).status_code == 422  # no file
    assert (await post("/v1/scans", files=photo, data={"lat": "10"})).status_code == 422
    assert (await post("/v1/scans", files=photo, data={"lat": "91", "lon": "0"})).status_code == 422


async def test_a_photo_without_a_readable_qr_code_asks_for_a_retake(api: Api) -> None:
    import cv2
    import numpy as np

    blank = np.full((480, 640), 128, np.uint8)
    ok, jpeg = cv2.imencode(".jpg", blank)
    assert ok
    result = await api.scan(jpeg.tobytes())
    assert result["verdict"] == "RECAPTURE"
    assert result["rule"] == "R1_capture"


async def test_an_overloaded_inference_queue_is_a_429_with_retry_after(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    serial = (await api.issue(1))[0]
    photo, _ = await api.photograph(serial, difficulty="easy", seed=0)

    async def refuse(*args: object, **kwargs: object) -> None:
        raise OverloadedError("inference queue is full")

    monkeypatch.setattr(api.container.inference, "score", refuse)
    response = await api.client.post(
        "/v1/scans",
        params={"investigate": "false"},
        files={"image": ("photo.jpg", photo, "image/jpeg")},
    )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"


async def test_metrics_expose_scan_inference_and_log_instruments(api: Api) -> None:
    serial = (await api.issue(1))[0]
    photo, _ = await api.photograph(serial, difficulty="easy", seed=0)
    await api.scan(photo, investigate="false")

    response = await api.client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    for name in (
        "sigil_scans_total{verdict=",
        "sigil_scan_seconds_bucket",
        "sigil_inference_batch_size_bucket",
        "sigil_inference_queue_depth",
        "sigil_investigation_cost_usd_total",
        "sigil_guardrail_violations_total",
        "sigil_log_size",
    ):
        assert name in text, name


async def test_the_demo_page_is_served_with_a_strict_content_security_policy(api: Api) -> None:
    response = await api.client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    policy = response.headers["content-security-policy"]
    assert "default-src 'none'" in policy
    assert "https:" not in policy


async def test_the_openapi_schema_lists_the_public_surface(api: Api) -> None:
    paths = (await api.client.get("/openapi.json")).json()["paths"]
    for path in (
        "/v1/scans",
        "/v1/batches",
        "/v1/log/sth",
        "/v1/investigations/{investigation_id}",
    ):
        assert path in paths
