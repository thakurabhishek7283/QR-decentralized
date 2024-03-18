"""The transparency log: an append-only Merkle tree with signed tree heads."""

from __future__ import annotations

import base64
import struct
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Self

from sigil.crypto.keys import Signer, verify_signature
from sigil.errors import LogError, NotFoundError
from sigil.ledger import merkle
from sigil.ledger.store import LogStore, SthRow
from sigil.storage import Database

STH_DOMAIN = b"sigil-sth-v1\x00"


@dataclass(frozen=True, slots=True)
class SignedTreeHead:
    tree_size: int
    root_hash: bytes
    timestamp_ms: int
    signature: bytes
    log_kid: bytes

    def signed_message(self) -> bytes:
        return STH_DOMAIN + struct.pack(">QQ", self.tree_size, self.timestamp_ms) + self.root_hash

    def verify(self, log_public_raw: bytes) -> bool:
        return verify_signature(log_public_raw, self.signed_message(), self.signature)

    def to_json(self) -> dict[str, Any]:
        return {
            "tree_size": self.tree_size,
            "root_hash": self.root_hash.hex(),
            "timestamp_ms": self.timestamp_ms,
            "signature": base64.b64encode(self.signature).decode("ascii"),
            "log_kid": self.log_kid.hex(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        try:
            return cls(
                tree_size=int(data["tree_size"]),
                root_hash=bytes.fromhex(data["root_hash"]),
                timestamp_ms=int(data["timestamp_ms"]),
                signature=base64.b64decode(data["signature"], validate=True),
                log_kid=bytes.fromhex(data["log_kid"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LogError(f"malformed signed tree head: {exc}") from exc


@dataclass(frozen=True, slots=True)
class InclusionProof:
    leaf_index: int
    tree_size: int
    path: list[bytes]


@dataclass(frozen=True, slots=True)
class ConsistencyProof:
    first: int
    second: int
    path: list[bytes]


class TransparencyLog:
    """RFC 9162 log persisted in SQLite.

    Leaf hashes are held in memory and proofs are computed over them in O(n),
    which is fine at demo scale. A production log would persist the tile
    structure used by tlog-tiles and Trillian instead.
    """

    def __init__(
        self,
        db: Database,
        signer: Signer,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = LogStore(db)
        self._db = db
        self._signer = signer
        self._clock = clock
        self._lock = threading.RLock()
        self._leaves = self._store.leaf_hashes()
        self._index = {leaf: i for i, leaf in enumerate(self._leaves)}

    @property
    def size(self) -> int:
        return len(self._leaves)

    def append(self, data: bytes) -> int:
        """Add ``data`` and return its leaf index; a duplicate returns the existing index."""
        return self.append_many([data])[0]

    def append_many(self, items: Sequence[bytes]) -> list[int]:
        """Append all items in one transaction, deduplicating against the log and each other."""
        with self._lock:
            fresh: dict[bytes, bytes] = {}
            for data in items:
                lh = merkle.leaf_hash(data)
                if lh not in self._index:
                    fresh.setdefault(lh, data)
            start = len(self._leaves)
            if fresh:
                with self._db.transaction():
                    self._store.add_leaves(start, list(fresh.items()), self._now_ms())
                for offset, lh in enumerate(fresh):
                    self._index[lh] = start + offset
                self._leaves.extend(fresh)
            return [self._index[merkle.leaf_hash(data)] for data in items]

    def publish(self) -> SignedTreeHead:
        """Sign and persist a tree head for the current size; a no-op if already published."""
        with self._lock:
            size = len(self._leaves)
            existing = self._store.sth(size)
            if existing is not None:
                return self._to_sth(existing)
            head = SignedTreeHead(
                tree_size=size,
                root_hash=merkle.root_hash(self._leaves),
                timestamp_ms=self._now_ms(),
                signature=b"",
                log_kid=self._signer.kid,
            )
            sth = replace(head, signature=self._signer.sign(head.signed_message()))
            self._store.add_sth((sth.tree_size, sth.root_hash, sth.timestamp_ms, sth.signature))
            return sth

    def latest_sth(self) -> SignedTreeHead:
        with self._lock:
            row = self._store.latest_sth()
            return self._to_sth(row) if row is not None else self.publish()

    def sth_at(self, tree_size: int) -> SignedTreeHead:
        row = self._store.sth(tree_size)
        if row is None:
            raise NotFoundError(f"no signed tree head was published for size {tree_size}")
        return self._to_sth(row)

    def index_of(self, leaf_hash: bytes) -> int | None:
        return self._index.get(leaf_hash)

    def prove_inclusion(self, leaf_hash: bytes, tree_size: int | None = None) -> InclusionProof:
        """Prove a leaf is in the tree of ``tree_size`` (default: the latest published head).

        A proof is only useful against a root the log has signed, so the default is
        not the live size.
        """
        with self._lock:
            index = self._index.get(leaf_hash)
            if index is None:
                raise NotFoundError("leaf is not in the log")
            size = self.latest_sth().tree_size if tree_size is None else tree_size
            if not index < size <= len(self._leaves):
                raise LogError(f"leaf {index} is not covered by a tree of size {size}")
            path = merkle.inclusion_proof(index, self._leaves[:size])
            return InclusionProof(index, size, path)

    def prove_consistency(self, first: int, second: int) -> ConsistencyProof:
        with self._lock:
            if not 1 <= first <= second <= len(self._leaves):
                raise LogError(f"cannot prove consistency between sizes {first} and {second}")
            path = merkle.consistency_proof(first, self._leaves[:second])
            return ConsistencyProof(first, second, path)

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _to_sth(self, row: SthRow) -> SignedTreeHead:
        size, root, timestamp_ms, signature = row
        return SignedTreeHead(size, root, timestamp_ms, signature, self._signer.kid)
