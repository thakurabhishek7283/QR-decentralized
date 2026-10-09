from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sigil.crypto import (
    Ed25519Signer,
    FileKeyStore,
    KeyRecord,
    KeyRing,
    kid_for,
    verify_signature,
)
from sigil.errors import UnknownKeyError

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def record_for(signer: Ed25519Signer) -> KeyRecord:
    return KeyRecord(signer.kid, signer.public_raw(), "issuer", NOW)


def test_sign_and_verify() -> None:
    signer = Ed25519Signer.generate()
    sig = signer.sign(b"message")
    assert len(sig) == 64
    assert verify_signature(signer.public_raw(), b"message", sig)


def test_verify_rejects_wrong_message_key_and_malformed_inputs() -> None:
    signer, other = Ed25519Signer.generate(), Ed25519Signer.generate()
    sig = signer.sign(b"message")
    assert not verify_signature(signer.public_raw(), b"massage", sig)
    assert not verify_signature(other.public_raw(), b"message", sig)
    assert not verify_signature(signer.public_raw(), b"message", sig[:-1])
    assert not verify_signature(b"short", b"message", sig)


def test_kid_is_a_short_digest_of_the_public_key() -> None:
    signer = Ed25519Signer.generate()
    assert signer.kid == kid_for(signer.public_raw())
    assert len(signer.kid) == 4
    assert len(signer.public_raw()) == 32


def test_pem_round_trip_keeps_identity() -> None:
    signer = Ed25519Signer.generate()
    clone = Ed25519Signer.from_pem(signer.to_pem())
    assert clone.kid == signer.kid
    assert clone.public_raw() == signer.public_raw()
    assert verify_signature(signer.public_raw(), b"m", clone.sign(b"m"))


def test_from_pem_rejects_other_key_types() -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with pytest.raises(ValueError, match="Ed25519"):
        Ed25519Signer.from_pem(pem)


def test_keyring_add_get_and_unknown() -> None:
    ring = KeyRing()
    signer = Ed25519Signer.generate()
    rec = record_for(signer)
    ring.add(rec)
    assert ring.get(signer.kid) == rec
    assert signer.kid in ring
    assert ring.records() == [rec]
    with pytest.raises(UnknownKeyError):
        ring.get(b"\x00\x00\x00\x00")


def test_keyring_refuses_to_overwrite_a_key() -> None:
    ring = KeyRing()
    rec = record_for(Ed25519Signer.generate())
    ring.add(rec)
    with pytest.raises(ValueError, match="already"):
        ring.add(rec)


def test_revoke_records_the_earliest_time() -> None:
    ring = KeyRing()
    rec = record_for(Ed25519Signer.generate())
    ring.add(rec)
    later = NOW + timedelta(days=2)
    assert ring.revoke(rec.kid, later).revoked_at == later
    assert ring.revoke(rec.kid, NOW + timedelta(days=5)).revoked_at == later
    assert ring.revoke(rec.kid, NOW + timedelta(days=1)).revoked_at == NOW + timedelta(days=1)
    assert ring.get(rec.kid).revoked_at == NOW + timedelta(days=1)
    with pytest.raises(UnknownKeyError):
        ring.revoke(b"\x01\x02\x03\x04", NOW)


class TestFileKeyStore:
    def test_creates_all_files_with_private_permissions(self, tmp_path: Path) -> None:
        root = tmp_path / "keys"
        FileKeyStore(root).load_or_create()
        for name in ("issuer.pem", "log.pem", "cdp.key"):
            assert stat.S_IMODE((root / name).stat().st_mode) == 0o600
        assert (root / "keyring.json").exists()
        assert (root / "issuer.pem").read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----")
        assert (root / "cdp.key").stat().st_size == 32
        assert not list(root.glob(".*.tmp"))

    def test_keyring_lists_issuer_and_log_public_keys(self, tmp_path: Path) -> None:
        material = FileKeyStore(tmp_path).load_or_create()
        by_purpose = {rec.purpose: rec for rec in material.keyring.records()}
        assert by_purpose["issuer"].public_raw == material.issuer.public_raw()
        assert by_purpose["log"].public_raw == material.log.public_raw()
        assert material.issuer.kid != material.log.kid
        data = json.loads((tmp_path / "keyring.json").read_text())
        assert {item["purpose"] for item in data} == {"issuer", "log"}
        assert all(item["revoked_at"] is None for item in data)

    def test_load_or_create_is_idempotent(self, tmp_path: Path) -> None:
        first = FileKeyStore(tmp_path).load_or_create()
        second = FileKeyStore(tmp_path).load_or_create()
        assert second.issuer.public_raw() == first.issuer.public_raw()
        assert second.log.public_raw() == first.log.public_raw()
        assert second.cdp_key == first.cdp_key
        assert second.keyring.records() == first.keyring.records()

    def test_revocations_survive_a_save_and_reload(self, tmp_path: Path) -> None:
        store = FileKeyStore(tmp_path)
        material = store.load_or_create()
        material.keyring.revoke(material.issuer.kid, NOW)
        store.save_keyring(material.keyring)
        reloaded = FileKeyStore(tmp_path).load_or_create()
        assert reloaded.keyring.get(material.issuer.kid).revoked_at == NOW
        assert reloaded.keyring.get(material.log.kid).revoked_at is None

    def test_missing_keyring_is_rebuilt_from_the_private_keys(self, tmp_path: Path) -> None:
        first = FileKeyStore(tmp_path).load_or_create()
        (tmp_path / "keyring.json").unlink()
        second = FileKeyStore(tmp_path).load_or_create()
        assert {r.kid for r in second.keyring.records()} == {first.issuer.kid, first.log.kid}

    def test_rejects_a_truncated_cdp_key(self, tmp_path: Path) -> None:
        FileKeyStore(tmp_path).load_or_create()
        (tmp_path / "cdp.key").write_bytes(b"short")
        with pytest.raises(ValueError, match="32 bytes"):
            FileKeyStore(tmp_path).load_or_create()
