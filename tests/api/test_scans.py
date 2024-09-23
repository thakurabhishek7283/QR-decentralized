"""Simulated photos through the scan endpoint."""

from __future__ import annotations

from typing import Any

import pytest

from tests.api.conftest import Api

pytestmark = pytest.mark.asyncio(loop_scope="package")


async def first_serial(api: Api) -> str:
    """A unit of the seeded batch printed on a good press."""
    batches = (await api.client.get("/v1/batches")).json()
    batch = next(b for b in batches if b["sku"] == "ACME-WIDGET-PRO")
    serials: list[str] = (await api.client.get(f"/v1/batches/{batch['batch_id']}")).json()[
        "serials"
    ]
    return serials[0]


async def test_a_genuine_photo_is_never_called_counterfeit(api: Api) -> None:
    serial = await first_serial(api)
    result = await api.scan_until(serial, {"GENUINE", "SUSPICIOUS"})
    assert result["serial_hex"] == serial
    assert result["sku"] == "ACME-WIDGET-PRO"
    signature = next(e for e in result["evidence"]["items"] if e["id"] == "signature")
    assert signature["passed"] is True
    assert result["timings_ms"]["total"] > 0

    stored = (await api.client.get(f"/v1/scans/{result['scan_id']}")).json()
    assert stored["result"]["scan_id"] == result["scan_id"]
    assert stored["result"]["verdict"] == result["verdict"]
    assert stored["final_verdict"] == result["verdict"]


@pytest.mark.parametrize("attack", ["photocopy", "hq_reprint"])
async def test_a_counterfeit_is_never_accepted(api: Api, attack: str) -> None:
    serial = await first_serial(api)
    photo = await api.photograph(serial, attack=attack, difficulty="easy", seed=3)
    result = await api.scan(photo)
    assert result["verdict"] != "GENUINE"


async def test_scanning_one_serial_in_two_distant_cities_raises_clone_risk(api: Api) -> None:
    serial = (await api.issue(1))[0]
    mumbai = {"lat": "19.08", "lon": "72.88", "device_id": "phone-a"}
    london = {"lat": "51.51", "lon": "-0.13", "device_id": "phone-b"}

    async def scan_at(photo: bytes, place: dict[str, str]) -> dict[str, Any]:
        response = await api.client.post(
            "/v1/scans",
            data=place,
            files={"image": ("photo.jpg", photo, "image/jpeg")},
        )
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        return body

    for seed in range(12):  # retake until the QR code is readable
        photo = await api.photograph(serial, difficulty="easy", seed=seed)
        if (await scan_at(photo, mumbai))["verdict"] != "RECAPTURE":
            break
    second = await scan_at(photo, london)  # minutes later, 7,000 km away

    clone = next(e for e in second["evidence"]["items"] if e["id"] == "clone")
    assert clone["data"]["risk"] >= 0.7
    assert [s["name"] for s in clone["data"]["signals"]] == ["impossible_travel"]
    assert clone["passed"] is False


async def test_an_ambiguous_scan_is_reported_as_suspicious(unsure_api: Api) -> None:
    serial = (await unsure_api.issue(2))[0]
    for seed in range(12):
        photo = await unsure_api.photograph(serial, difficulty="easy", seed=seed)
        scan = await unsure_api.scan(photo)
        if scan["verdict"] != "RECAPTURE":
            break
    else:
        raise AssertionError("no readable shot")
    assert scan["verdict"] == "SUSPICIOUS"
    assert scan["rule"] == "R11_ambiguous"

    stored = (await unsure_api.client.get(f"/v1/scans/{scan['scan_id']}")).json()
    assert stored["final_verdict"] == "SUSPICIOUS"
    assert stored["result"]["verdict"] == "SUSPICIOUS"

    metrics = (await unsure_api.client.get("/metrics")).text
    assert 'sigil_scans_total{verdict="SUSPICIOUS"}' in metrics


async def test_unknown_resources_are_404(api: Api) -> None:
    assert (await api.client.get(f"/v1/scans/{'0' * 32}")).status_code == 404
    assert (await api.client.get("/v1/batches/nope")).status_code == 404
