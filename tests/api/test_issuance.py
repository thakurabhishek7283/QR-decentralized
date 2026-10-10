"""Health, keys, issuance, labels and the demo seed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.api.conftest import AUTH, Api

pytestmark = pytest.mark.asyncio(loop_scope="package")


async def test_healthz(api: Api) -> None:
    body = (await api.client.get("/healthz")).json()
    assert body["status"] == "ok"
    assert body["log_size"] >= 18  # the demo batches
    assert body["model_version"]
    assert body["investigator"] == "reference"


async def test_keys_are_listed_with_public_material_only(api: Api) -> None:
    keys = (await api.client.get("/v1/keys")).json()
    assert {k["purpose"] for k in keys} == {"issuer", "log"}
    for key in keys:
        assert len(key["kid"]) == 8
        assert len(key["public_key"]) == 43  # 32 bytes, unpadded base64url
        assert key["revoked_at"] is None
        assert set(key) == {"kid", "purpose", "public_key", "created_at", "revoked_at"}


async def test_demo_batches_are_seeded_with_qc(api: Api) -> None:
    batches = (await api.client.get("/v1/batches")).json()
    assert {b["print_quality"] for b in batches} >= {"good", "worn"}
    detail = (await api.client.get(f"/v1/batches/{batches[0]['batch_id']}")).json()
    assert detail["qc"] is not None
    assert len(detail["serials"]) == batches[0]["units"]


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
async def test_issuing_requires_the_api_key(api: Api, headers: dict[str, str]) -> None:
    response = await api.client.post("/v1/batches", json={"sku": "X", "count": 1}, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"


async def test_issue_a_batch(api: Api) -> None:
    before = (await api.client.get("/healthz")).json()["log_size"]
    response = await api.client.post(
        "/v1/batches", json={"sku": "WIDGET-1", "count": 4, "print_quality": "worn"}, headers=AUTH
    )
    assert response.status_code == 201
    body = response.json()
    assert body["sku"] == "WIDGET-1"
    assert body["print_quality"] == "worn"
    assert len(body["serials"]) == 4
    assert body["qc"]["n"] >= 1
    assert (await api.client.get("/healthz")).json()["log_size"] == before + 4

    fetched = (await api.client.get(f"/v1/batches/{body['batch_id']}")).json()
    assert fetched["serials"] == body["serials"]


@pytest.mark.parametrize(
    "body",
    [
        {"sku": "X", "count": 0},
        {"sku": "X", "count": 201},
        {"sku": "", "count": 1},
        {"sku": "x" * 33, "count": 1},
        {"sku": "naïve", "count": 1},
        {"sku": "X", "count": 1, "print_quality": "mint"},
    ],
)
async def test_invalid_batches_are_rejected(api: Api, body: dict[str, object]) -> None:
    response = await api.client.post("/v1/batches", json=body, headers=AUTH)
    assert response.status_code == 422


async def test_label_png(api: Api) -> None:
    serial = (await api.issue(1))[0]
    response = await api.client.get(f"/v1/labels/{serial}.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert int.from_bytes(response.content[16:20]) == 708  # LABEL_W
    assert int.from_bytes(response.content[20:24]) == 366  # LABEL_H


async def test_unknown_and_malformed_serials(api: Api) -> None:
    assert (await api.client.get(f"/v1/labels/{'0' * 24}.png")).status_code == 404
    assert (await api.client.get("/v1/labels/not-a-serial.png")).status_code == 422
    assert (await api.client.post(f"/v1/simulate/{'0' * 24}", json={})).status_code == 404


async def test_unit_exposes_the_qr_payload(api: Api) -> None:
    serial = (await api.issue(1))[0]
    unit = (await api.client.get(f"/v1/units/{serial}")).json()
    assert unit["payload"].startswith("SG1:")
    assert unit["serial"] == serial


async def test_revoking_a_key_needs_auth_and_marks_it_suspect(fresh_api: Api) -> None:
    serial = (await fresh_api.issue(1))[0]
    issuer = next(
        k for k in (await fresh_api.client.get("/v1/keys")).json() if k["purpose"] == "issuer"
    )
    url = f"/v1/keys/{issuer['kid']}/revoke"

    assert (await fresh_api.client.post(url, json={})).status_code == 401
    assert (
        await fresh_api.client.post("/v1/keys/zzzzzzzz/revoke", headers=AUTH)
    ).status_code == 422
    assert (
        await fresh_api.client.post("/v1/keys/00000000/revoke", headers=AUTH)
    ).status_code == 404

    # Revoked from tomorrow: the credential predates the revocation, so it is suspect, not forged.
    at = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    response = await fresh_api.client.post(url, json={"at": at}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["revoked_at"] is not None

    result = await fresh_api.scan_until(serial, {"SUSPICIOUS"})
    assert result["rule"] == "R8_revoked_before"
