"""The transparency log over HTTP, verified the way an untrusting client would."""

from __future__ import annotations

import pytest

from sigil.crypto import SignedCredential
from sigil.ledger import SignedTreeHead, leaf_hash, verify_consistency, verify_inclusion
from tests.api.conftest import Api

pytestmark = pytest.mark.asyncio(loop_scope="package")


async def fetch_sth(api: Api, **params: int) -> SignedTreeHead:
    response = await api.client.get("/v1/log/sth", params=params)
    assert response.status_code == 200
    return SignedTreeHead.from_json(response.json())


async def log_public_key(api: Api) -> bytes:
    import base64

    keys = (await api.client.get("/v1/keys")).json()
    key = next(k["public_key"] for k in keys if k["purpose"] == "log")
    return base64.urlsafe_b64decode(key + "=")


async def test_the_tree_head_is_signed_by_the_published_log_key(api: Api) -> None:
    sth = await fetch_sth(api)
    assert sth.tree_size >= 18
    assert sth.verify(await log_public_key(api))
    assert not sth.verify(bytes(32))


async def test_an_inclusion_proof_verifies_against_the_credential_in_the_qr_code(api: Api) -> None:
    serial = (await api.issue(5))[0]
    unit = (await api.client.get(f"/v1/units/{serial}")).json()
    # Recompute the leaf from the QR text alone: nothing the server says about the leaf is trusted.
    leaf = leaf_hash(SignedCredential.from_qr_payload(unit["payload"]).leaf_data())

    response = await api.client.get("/v1/log/inclusion", params={"serial": serial})
    assert response.status_code == 200
    proof = response.json()
    assert proof["leaf_hash"] == leaf.hex()
    assert proof["leaf_index"] == unit["leaf_index"]

    sth = await fetch_sth(api, tree_size=proof["tree_size"])
    assert verify_inclusion(
        leaf, proof["leaf_index"], proof["tree_size"], [bytes.fromhex(p) for p in proof["path"]],
        sth.root_hash,
    )  # fmt: skip

    by_hash = await api.client.get("/v1/log/inclusion", params={"leaf_hash": leaf.hex()})
    assert by_hash.json() == proof


async def test_a_tampered_proof_does_not_verify(api: Api) -> None:
    serial = (await api.issue(3))[0]
    proof = (await api.client.get("/v1/log/inclusion", params={"serial": serial})).json()
    sth = await fetch_sth(api, tree_size=proof["tree_size"])
    leaf = bytes.fromhex(proof["leaf_hash"])
    path = [bytes.fromhex(p) for p in proof["path"]]
    path[0] = bytes(32)
    assert not verify_inclusion(leaf, proof["leaf_index"], proof["tree_size"], path, sth.root_hash)


async def test_consistency_proof_links_an_older_head_to_a_newer_one(api: Api) -> None:
    older = await fetch_sth(api)
    await api.issue(6)
    newer = await fetch_sth(api)
    assert newer.tree_size == older.tree_size + 6

    response = await api.client.get(
        "/v1/log/consistency", params={"first": older.tree_size, "second": newer.tree_size}
    )
    assert response.status_code == 200
    path = [bytes.fromhex(p) for p in response.json()["path"]]
    assert verify_consistency(
        older.tree_size, newer.tree_size, older.root_hash, newer.root_hash, path
    )
    assert not verify_consistency(
        older.tree_size, newer.tree_size, bytes(32), newer.root_hash, path
    )


async def test_bad_log_requests(api: Api) -> None:
    size = (await fetch_sth(api)).tree_size
    get = api.client.get
    assert (
        await get("/v1/log/consistency", params={"first": size + 1, "second": size})
    ).status_code == 400
    assert (await get("/v1/log/consistency", params={"first": 1})).status_code == 422
    assert (await get("/v1/log/inclusion")).status_code == 422  # neither serial nor leaf_hash
    assert (await get("/v1/log/inclusion", params={"leaf_hash": "ab" * 32})).status_code == 404
    assert (await get("/v1/log/sth", params={"tree_size": 10**6})).status_code == 404
