"""The transparency log's public read API. Hashes are hex; the signature is base64."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from sigil.api.deps import ContainerDep
from sigil.api.schemas import SerialHex
from sigil.ledger import leaf_hash as hash_leaf

router = APIRouter(prefix="/log", tags=["transparency log"])


class InclusionOut(BaseModel):
    leaf_hash: str
    leaf_index: int
    tree_size: int
    path: list[str]


class ConsistencyOut(BaseModel):
    first: int
    second: int
    path: list[str]


@router.get("/sth")
async def signed_tree_head(
    c: ContainerDep, tree_size: Annotated[int | None, Query(ge=1)] = None
) -> dict[str, object]:
    """The latest signed tree head, or the one published for ``tree_size``."""
    sth = c.log.latest_sth() if tree_size is None else c.log.sth_at(tree_size)
    return sth.to_json()


@router.get("/inclusion")
async def inclusion_proof(
    c: ContainerDep,
    serial: SerialHex | None = None,
    leaf_hash: Annotated[str | None, Query(pattern=r"^[0-9a-f]{64}$")] = None,
    tree_size: Annotated[int | None, Query(ge=1)] = None,
) -> InclusionOut:
    """Prove a credential is in the log: by ``serial``, or by ``leaf_hash`` (hex). Verify it against
    the signed tree head of the same ``tree_size`` (RFC 9162 section 2.1.3.2)."""
    if (serial is None) == (leaf_hash is None):
        raise HTTPException(422, "give exactly one of serial or leaf_hash")
    if serial is not None:
        credential, _ = c.issuance.unit(serial)
        digest = hash_leaf(credential.leaf_data())
    else:
        digest = bytes.fromhex(leaf_hash or "")
    proof = c.log.prove_inclusion(digest, tree_size)
    return InclusionOut(
        leaf_hash=digest.hex(),
        leaf_index=proof.leaf_index,
        tree_size=proof.tree_size,
        path=[node.hex() for node in proof.path],
    )


@router.get("/consistency")
async def consistency_proof(
    c: ContainerDep,
    first: Annotated[int, Query(ge=1)],
    second: Annotated[int, Query(ge=1)],
) -> ConsistencyOut:
    """Prove the tree of size ``second`` extends the tree of size ``first`` (RFC 9162 2.1.4.2)."""
    proof = c.log.prove_consistency(first, second)
    return ConsistencyOut(
        first=proof.first, second=proof.second, path=[n.hex() for n in proof.path]
    )
