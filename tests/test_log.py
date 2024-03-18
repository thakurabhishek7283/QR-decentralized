from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sigil.crypto import Ed25519Signer
from sigil.errors import LogError, NotFoundError
from sigil.ledger import (
    SignedTreeHead,
    TransparencyLog,
    leaf_hash,
    root_hash,
    verify_consistency,
    verify_inclusion,
)
from sigil.ledger.merkle import EMPTY_ROOT
from sigil.storage import Database


@pytest.fixture(scope="module")
def signer() -> Ed25519Signer:
    return Ed25519Signer.generate()


class Clock:
    """A settable clock so tree-head timestamps are deterministic."""

    def __init__(self) -> None:
        self.now = 1_750_000_000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


@pytest.fixture
def log(db: Database, signer: Ed25519Signer) -> TransparencyLog:
    return TransparencyLog(db, signer, clock=Clock())


def entries(n: int, start: int = 0) -> list[bytes]:
    return [f"credential-{i}".encode() for i in range(start, start + n)]


def test_append_assigns_consecutive_indices(log: TransparencyLog) -> None:
    assert [log.append(e) for e in entries(3)] == [0, 1, 2]
    assert log.size == 3
    assert log.index_of(leaf_hash(entries(3)[1])) == 1
    assert log.index_of(leaf_hash(b"never appended")) is None


def test_append_is_idempotent_for_duplicate_data(log: TransparencyLog) -> None:
    first = log.append(b"same")
    log.append(b"other")
    assert log.append(b"same") == first
    assert log.size == 2


def test_append_many_is_ordered_and_deduplicated(log: TransparencyLog) -> None:
    log.append(b"a")
    indices = log.append_many([b"b", b"a", b"c", b"b"])
    assert indices == [1, 0, 2, 1]
    assert log.size == 3
    assert log.append_many([]) == []


def test_append_many_is_atomic(db: Database, signer: Ed25519Signer) -> None:
    log = TransparencyLog(db, signer)
    log.append(b"seed")
    # A row the in-memory cache does not know about makes the batch's second insert collide.
    db.execute(
        "INSERT INTO log_leaves (idx, leaf_hash, data, appended_at) VALUES (2, ?, ?, 0)",
        (leaf_hash(b"b"), b"b"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        log.append_many([b"a", b"b"])
    assert log.size == 1
    assert log.index_of(leaf_hash(b"a")) is None
    count = db.query_one("SELECT COUNT(*) AS n FROM log_leaves")
    assert count is not None
    assert count["n"] == 2


def test_publish_signs_the_current_root(log: TransparencyLog, signer: Ed25519Signer) -> None:
    log.append_many(entries(5))
    sth = log.publish()
    assert sth.tree_size == 5
    assert sth.root_hash == root_hash([leaf_hash(e) for e in entries(5)])
    assert sth.log_kid == signer.kid
    assert sth.verify(signer.public_raw())
    assert log.sth_at(5) == sth


def test_publish_is_a_noop_when_the_size_is_unchanged(log: TransparencyLog) -> None:
    log.append(b"a")
    first = log.publish()
    assert log.publish() == first
    log.append(b"a")
    assert log.publish() == first
    log.append(b"b")
    second = log.publish()
    assert second.tree_size == 2
    assert second.timestamp_ms > first.timestamp_ms


def test_latest_sth_publishes_an_empty_tree_when_nothing_exists(
    log: TransparencyLog, signer: Ed25519Signer
) -> None:
    sth = log.latest_sth()
    assert (sth.tree_size, sth.root_hash) == (0, EMPTY_ROOT)
    assert sth.verify(signer.public_raw())


def test_latest_sth_is_the_newest_published_not_the_live_size(log: TransparencyLog) -> None:
    log.append_many(entries(3))
    log.publish()
    log.append(b"unpublished")
    assert log.latest_sth().tree_size == 3
    assert log.publish().tree_size == 4
    assert log.latest_sth().tree_size == 4


def test_sth_at_unknown_size_is_not_found(log: TransparencyLog) -> None:
    log.append(b"a")
    with pytest.raises(NotFoundError):
        log.sth_at(1)


def test_sth_signature_checks_key_and_every_field(
    log: TransparencyLog, signer: Ed25519Signer
) -> None:
    log.append_many(entries(4))
    sth = log.publish()
    assert not sth.verify(Ed25519Signer.generate().public_raw())
    forged_root = SignedTreeHead(
        sth.tree_size, leaf_hash(b"x"), sth.timestamp_ms, sth.signature, sth.log_kid
    )
    forged_size = SignedTreeHead(
        sth.tree_size + 1, sth.root_hash, sth.timestamp_ms, sth.signature, sth.log_kid
    )
    forged_time = SignedTreeHead(
        sth.tree_size, sth.root_hash, sth.timestamp_ms + 1, sth.signature, sth.log_kid
    )
    for forged in (forged_root, forged_size, forged_time):
        assert not forged.verify(signer.public_raw())


def test_sth_message_layout(log: TransparencyLog) -> None:
    log.append(b"a")
    sth = log.publish()
    message = sth.signed_message()
    assert message.startswith(b"sigil-sth-v1\x00")
    assert int.from_bytes(message[13:21], "big") == 1
    assert int.from_bytes(message[21:29], "big") == sth.timestamp_ms
    assert message[29:] == sth.root_hash


def test_sth_json_round_trip(log: TransparencyLog) -> None:
    log.append(b"a")
    sth = log.publish()
    data = sth.to_json()
    assert data["root_hash"] == sth.root_hash.hex()
    assert SignedTreeHead.from_json(data) == sth


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("root_hash"),
        lambda d: d.update(root_hash="zz"),
        lambda d: d.update(signature="not base64!"),
        lambda d: d.update(tree_size="many"),
        lambda d: d.update(log_kid=None),
    ],
)
def test_sth_from_json_rejects_malformed_input(
    log: TransparencyLog, mutate: Callable[[dict[str, Any]], object]
) -> None:
    log.append(b"a")
    data = log.publish().to_json()
    mutate(data)
    with pytest.raises(LogError):
        SignedTreeHead.from_json(data)


def test_inclusion_proofs_verify_against_the_signed_root(log: TransparencyLog) -> None:
    items = entries(13)
    log.append_many(items)
    sth = log.publish()
    for item in items:
        lh = leaf_hash(item)
        proof = log.prove_inclusion(lh)
        assert proof.tree_size == 13
        assert verify_inclusion(lh, proof.leaf_index, proof.tree_size, proof.path, sth.root_hash)


def test_inclusion_proof_against_an_explicit_older_tree(log: TransparencyLog) -> None:
    log.append_many(entries(5))
    old = log.publish()
    log.append_many(entries(8, start=5))
    new = log.publish()
    lh = leaf_hash(entries(1)[0])
    assert log.prove_inclusion(lh).tree_size == 13
    proof = log.prove_inclusion(lh, tree_size=5)
    assert verify_inclusion(lh, proof.leaf_index, 5, proof.path, old.root_hash)
    assert not verify_inclusion(lh, proof.leaf_index, 5, proof.path, new.root_hash)


def test_inclusion_proof_errors(log: TransparencyLog) -> None:
    log.append_many(entries(4))
    log.publish()
    with pytest.raises(NotFoundError):
        log.prove_inclusion(leaf_hash(b"absent"))
    late = log.append(b"late")
    assert late == 4
    with pytest.raises(LogError, match="not covered"):
        log.prove_inclusion(leaf_hash(b"late"))
    with pytest.raises(LogError, match="not covered"):
        log.prove_inclusion(leaf_hash(entries(1)[0]), tree_size=99)
    with pytest.raises(LogError, match="not covered"):
        log.prove_inclusion(leaf_hash(b"late"), tree_size=4)


def test_consistency_proofs_link_every_pair_of_published_heads(log: TransparencyLog) -> None:
    heads = []
    for start, count in ((0, 1), (1, 2), (3, 4), (7, 6)):
        log.append_many(entries(count, start=start))
        heads.append(log.publish())
    for i, old in enumerate(heads):
        for new in heads[i:]:
            proof = log.prove_consistency(old.tree_size, new.tree_size)
            assert (proof.first, proof.second) == (old.tree_size, new.tree_size)
            assert verify_consistency(
                old.tree_size, new.tree_size, old.root_hash, new.root_hash, proof.path
            )


def test_consistency_proof_argument_errors(log: TransparencyLog) -> None:
    log.append_many(entries(4))
    for first, second in ((0, 4), (3, 2), (2, 5)):
        with pytest.raises(LogError):
            log.prove_consistency(first, second)


def test_reopening_restores_state_and_proofs(db: Database, signer: Ed25519Signer) -> None:
    log = TransparencyLog(db, signer)
    log.append_many(entries(9))
    sth = log.publish()

    reopened = TransparencyLog(db, signer)
    assert reopened.size == 9
    assert reopened.latest_sth() == sth
    assert reopened.publish() == sth
    assert reopened.append(entries(1)[0]) == 0
    lh = leaf_hash(entries(9)[8])
    assert reopened.prove_inclusion(lh) == log.prove_inclusion(lh)
    assert reopened.append(b"tenth") == 9


def test_log_survives_a_real_file_database(tmp_path: Path, signer: Ed25519Signer) -> None:
    path = tmp_path / "sigil.db"
    first = Database(path)
    log = TransparencyLog(first, signer)
    log.append_many(entries(6))
    sth = log.publish()
    first.close()

    second = Database(path)
    assert TransparencyLog(second, signer).latest_sth() == sth
    second.close()


def test_concurrent_appends_keep_the_log_consistent(db: Database, signer: Ed25519Signer) -> None:
    log = TransparencyLog(db, signer)

    def worker(worker_id: int) -> None:
        for i in range(25):
            log.append(f"w{worker_id}-{i}".encode())
            log.append(b"shared")

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert log.size == 6 * 25 + 1
    sth = log.publish()
    stored = db.query("SELECT data FROM log_leaves ORDER BY idx")
    assert sth.root_hash == root_hash([leaf_hash(bytes(row["data"])) for row in stored])
