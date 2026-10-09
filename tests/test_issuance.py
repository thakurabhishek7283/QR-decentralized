"""Issuing batches, persisting them, and measuring their QC baseline."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from sigil.cdp import PrintProfile
from sigil.crypto import FileKeyStore, KeyMaterial, SignedCredential
from sigil.errors import NotFoundError, SigilError
from sigil.issuance import IssuanceStore, Issuer, QcStats, run_qc
from sigil.ledger import TransparencyLog, leaf_hash, verify_inclusion
from sigil.storage import Database
from sigil.vision import LABEL_H, LABEL_W
from tests.support import running_inference


class Rig(NamedTuple):
    issuer: Issuer
    keys: KeyMaterial
    log: TransparencyLog


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    keys = FileKeyStore(tmp_path).load_or_create()
    db = Database(":memory:")
    log = TransparencyLog(db, keys.log)
    return Rig(Issuer(keys, log, IssuanceStore(db)), keys, log)


@pytest.fixture
def issuer(rig: Rig) -> Issuer:
    return rig.issuer


def test_batch_units_are_signed_logged_and_stored(rig: Rig) -> None:
    issuer = rig.issuer
    info, units = issuer.issue_batch("WIDGET", 5)

    assert info.sku == "WIDGET"
    assert info.print_quality == "good"
    assert info.qc is None
    assert info.batch_id.startswith("B")
    assert len(info.batch_id) == 7
    assert [u.serial_hex for u in units] == issuer.store.units(info.batch_id)
    assert len({u.serial_hex for u in units}) == 5

    for unit in units:
        credential = SignedCredential.from_qr_payload(unit.payload)
        credential.verify(rig.keys.keyring)
        assert credential.credential.batch == info.batch_id
        assert credential.credential.serial_hex == unit.serial_hex
        assert issuer.store.unit(unit.serial_hex) == (credential, info.batch_id)
        assert rig.log.index_of(leaf_hash(credential.leaf_data())) == unit.leaf_index


def test_published_tree_head_covers_the_batch(rig: Rig) -> None:
    _, units = rig.issuer.issue_batch("WIDGET", 4)
    sth = rig.log.latest_sth()
    assert sth.tree_size == 4
    credential, _ = rig.issuer.store.unit(units[2].serial_hex)
    leaf = leaf_hash(credential.leaf_data())
    proof = rig.log.prove_inclusion(leaf)  # defaults to the latest *published* head
    assert verify_inclusion(leaf, proof.leaf_index, proof.tree_size, proof.path, sth.root_hash)


def test_explicit_batch_id_and_print_quality(issuer: Issuer) -> None:
    info, _ = issuer.issue_batch("WIDGET", 1, batch_id="LOT-7", print_quality="worn")
    assert issuer.store.batch("LOT-7") == info
    assert info.print_quality == "worn"
    assert info.created_at.tzinfo is not None


def test_empty_batch_is_rejected(issuer: Issuer) -> None:
    with pytest.raises(ValueError, match="at least one"):
        issuer.issue_batch("WIDGET", 0)


def test_injected_randomness_makes_issuance_reproducible(tmp_path: Path) -> None:
    def issue_once(root: Path) -> list[str]:
        keys = FileKeyStore(root).load_or_create()
        db = Database(":memory:")
        rng = np.random.default_rng(5)
        issuer = Issuer(
            keys,
            TransparencyLog(db, keys.log, clock=lambda: 1_700_000_000.0),
            IssuanceStore(db),
            random_bytes=rng.bytes,
            clock=lambda: 1_700_000_000.0,
        )
        info, units = issuer.issue_batch("WIDGET", 3)
        return [info.batch_id, *(u.serial_hex for u in units)]

    assert issue_once(tmp_path / "a") == issue_once(tmp_path / "a")  # same keys, same bytes


def test_label_is_a_printable_raster(issuer: Issuer) -> None:
    _, units = issuer.issue_batch("WIDGET", 2)
    label = issuer.render_label(units[0].serial_hex)
    assert label.shape == (LABEL_H, LABEL_W)
    assert label.dtype == np.float32
    assert not np.array_equal(label, issuer.render_label(units[1].serial_hex))
    template = issuer.template_for(bytes.fromhex(units[0].serial_hex))
    assert template.shape == (106, 106)


def test_unknown_records_raise_not_found(issuer: Issuer) -> None:
    with pytest.raises(NotFoundError):
        issuer.store.batch("nope")
    with pytest.raises(NotFoundError):
        issuer.store.unit("00" * 12)
    with pytest.raises(NotFoundError):
        issuer.store.set_qc("nope", QcStats(1, 0.5, 0.0, 0.5))


def test_store_lists_batches_oldest_first(issuer: Issuer) -> None:
    issuer.issue_batch("A", 1, batch_id="B1")
    issuer.issue_batch("B", 1, batch_id="B2")
    assert [b.batch_id for b in issuer.store.batches()] == ["B1", "B2"]


async def test_qc_records_the_batch_score_distribution(issuer: Issuer) -> None:
    info, _ = issuer.issue_batch("WIDGET", 4)
    async with running_inference(logit=2.0) as inference:
        stats = await run_qc(
            issuer, inference, info.batch_id, samples=2, rng=np.random.default_rng(1)
        )
    assert stats.n == 2
    assert stats.mean == pytest.approx(1 / (1 + np.exp(-2.0)), abs=1e-5)
    assert stats.std == pytest.approx(0.0, abs=1e-6)
    assert stats.min == pytest.approx(stats.mean, abs=1e-6)
    assert issuer.store.batch(info.batch_id).qc == stats


async def test_qc_uses_one_press_for_the_whole_batch(
    issuer: Issuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch comes off one printer, so QC must not redraw the print profile per sample."""
    drawn: list[str] = []
    real = __import__("sigil.issuance.qc", fromlist=["sample_print_profile"]).sample_print_profile

    def spy(rng: np.random.Generator, domain: str, quality: str) -> PrintProfile:
        drawn.append(quality)
        return real(rng, domain, quality)

    monkeypatch.setattr("sigil.issuance.qc.sample_print_profile", spy)
    info, _ = issuer.issue_batch("WIDGET", 3, print_quality="worn")
    async with running_inference() as inference:
        await run_qc(issuer, inference, info.batch_id, samples=3, rng=np.random.default_rng(2))
    assert drawn == ["worn"]


async def test_qc_fails_loudly_when_nothing_is_readable(
    issuer: Issuer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sigil.issuance.qc._shoot", lambda *args: None)
    info, _ = issuer.issue_batch("WIDGET", 2)
    async with running_inference() as inference:
        with pytest.raises(SigilError, match="could not read"):
            await run_qc(issuer, inference, info.batch_id, samples=2, rng=np.random.default_rng(3))
    assert issuer.store.batch(info.batch_id).qc is None
