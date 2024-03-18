from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigil.crypto import Ed25519Signer
from sigil.errors import LogError
from sigil.ledger import (
    Auditor,
    ConsistencyProof,
    SignedTreeHead,
    TransparencyLog,
    leaf_hash,
)
from sigil.storage import Database


@pytest.fixture(scope="module")
def signer() -> Ed25519Signer:
    return Ed25519Signer.generate()


@pytest.fixture
def log(db: Database, signer: Ed25519Signer) -> TransparencyLog:
    return TransparencyLog(db, signer)


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "auditor" / "state.json"


@pytest.fixture
def auditor(signer: Ed25519Signer, state_path: Path) -> Auditor:
    return Auditor(signer.public_raw(), state_path)


def grow(log: TransparencyLog, count: int, tag: str = "c") -> SignedTreeHead:
    start = log.size
    log.append_many([f"{tag}-{i}".encode() for i in range(start, start + count)])
    return log.publish()


def test_first_head_is_trusted_and_persisted(
    auditor: Auditor, log: TransparencyLog, state_path: Path
) -> None:
    sth = grow(log, 5)
    result = auditor.observe(sth, None)
    assert result.ok
    assert auditor.last == sth
    assert json.loads(state_path.read_text()) == sth.to_json()


def test_state_survives_a_restart(
    auditor: Auditor, log: TransparencyLog, signer: Ed25519Signer, state_path: Path
) -> None:
    first = grow(log, 5)
    auditor.observe(first, None)
    restarted = Auditor(signer.public_raw(), state_path)
    assert restarted.last == first
    second = grow(log, 3)
    assert not restarted.observe(second, None).ok
    assert restarted.observe(second, log.prove_consistency(5, 8)).ok


def test_follows_a_growing_log(auditor: Auditor, log: TransparencyLog) -> None:
    previous = grow(log, 1)
    assert auditor.observe(previous, None).ok
    for count in (1, 2, 4, 8, 3):
        head = grow(log, count)
        result = auditor.observe(head, log.prove_consistency(previous.tree_size, head.tree_size))
        assert result.ok, result.reason
        assert auditor.last == head
        previous = head


def test_consistency_proof_may_span_several_heads(auditor: Auditor, log: TransparencyLog) -> None:
    auditor.observe(grow(log, 3), None)
    grow(log, 4)
    latest = grow(log, 5)
    assert auditor.observe(latest, log.prove_consistency(3, 12)).ok


def test_growth_requires_a_proof(auditor: Auditor, log: TransparencyLog) -> None:
    auditor.observe(grow(log, 3), None)
    result = auditor.observe(grow(log, 2), None)
    assert not result.ok
    assert "proof" in result.reason


def test_proof_for_the_wrong_sizes_is_refused(auditor: Auditor, log: TransparencyLog) -> None:
    auditor.observe(grow(log, 3), None)
    head = grow(log, 4)
    wrong = log.prove_consistency(2, 7)
    assert not auditor.observe(head, wrong).ok


def test_garbage_proof_is_refused_and_state_is_kept(auditor: Auditor, log: TransparencyLog) -> None:
    first = grow(log, 3)
    auditor.observe(first, None)
    head = grow(log, 4)
    good = log.prove_consistency(3, 7)
    bad = ConsistencyProof(3, 7, [leaf_hash(b"x"), *good.path[1:]])
    assert not auditor.observe(head, bad).ok
    assert auditor.last == first


def test_signature_from_another_key_is_refused(auditor: Auditor, db: Database) -> None:
    rogue = TransparencyLog(Database(":memory:"), Ed25519Signer.generate())
    result = auditor.observe(grow(rogue, 3), None)
    assert not result.ok
    assert "signature" in result.reason
    assert auditor.last is None


def test_tampered_head_fields_are_refused(auditor: Auditor, log: TransparencyLog) -> None:
    sth = grow(log, 4)
    forged = SignedTreeHead(
        sth.tree_size, leaf_hash(b"evil"), sth.timestamp_ms, sth.signature, sth.log_kid
    )
    assert not auditor.observe(forged, None).ok


def test_shrinking_log_is_refused(auditor: Auditor, log: TransparencyLog) -> None:
    small = grow(log, 3)
    big = grow(log, 4)
    auditor.observe(big, None)
    result = auditor.observe(small, None)
    assert not result.ok
    assert "shrank" in result.reason
    assert auditor.last == big


def test_repeating_the_same_head_is_fine(auditor: Auditor, log: TransparencyLog) -> None:
    sth = grow(log, 3)
    auditor.observe(sth, None)
    assert auditor.observe(sth, None).ok


def test_two_signed_roots_for_one_size_are_a_split_view(
    auditor: Auditor, log: TransparencyLog, signer: Ed25519Signer
) -> None:
    auditor.observe(grow(log, 3), None)
    fork = TransparencyLog(Database(":memory:"), signer)
    fork.append_many([b"other-0", b"other-1", b"other-2"])
    result = auditor.observe(fork.publish(), None)
    assert not result.ok
    assert "different roots" in result.reason


def test_an_empty_first_head_can_be_extended_without_a_proof(
    auditor: Auditor, log: TransparencyLog
) -> None:
    assert auditor.observe(log.latest_sth(), None).ok
    assert auditor.observe(grow(log, 4), None).ok


def test_rewritten_history_is_detected(
    auditor: Auditor, log: TransparencyLog, db: Database, signer: Ed25519Signer
) -> None:
    """An operator who edits a stored leaf and keeps appending cannot satisfy the auditor."""
    before = grow(log, 5)
    assert auditor.observe(before, None).ok

    forged = b"counterfeit credential"
    db.execute(
        "UPDATE log_leaves SET data = ?, leaf_hash = ? WHERE idx = 2", (forged, leaf_hash(forged))
    )
    tampered = TransparencyLog(db, signer)
    assert tampered.size == 5
    after = grow(tampered, 2)  # still signed by the genuine log key

    result = auditor.observe(after, tampered.prove_consistency(5, 7))
    assert not result.ok
    assert "rewritten" in result.reason
    assert auditor.last == before


def test_an_untampered_log_passes_the_same_scenario(
    auditor: Auditor, log: TransparencyLog, db: Database, signer: Ed25519Signer
) -> None:
    before = grow(log, 5)
    assert auditor.observe(before, None).ok
    reopened = TransparencyLog(db, signer)
    after = grow(reopened, 2)
    assert auditor.observe(after, reopened.prove_consistency(5, 7)).ok


def test_corrupt_or_foreign_state_is_an_error_not_a_reset(
    log: TransparencyLog, signer: Ed25519Signer, state_path: Path
) -> None:
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{not json")
    with pytest.raises(LogError, match="corrupt"):
        Auditor(signer.public_raw(), state_path)

    state_path.write_text(json.dumps({"tree_size": 1}))
    with pytest.raises(LogError, match="malformed"):
        Auditor(signer.public_raw(), state_path)

    state_path.write_text(json.dumps(grow(log, 2).to_json()))
    with pytest.raises(LogError, match="not signed"):
        Auditor(Ed25519Signer.generate().public_raw(), state_path)
