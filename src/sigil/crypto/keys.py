"""Ed25519 signing keys, the public key ring, and a file-backed dev key provider."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from sigil.errors import UnknownKeyError

KID_LEN = 4
CDP_KEY_LEN = 32
Purpose = Literal["issuer", "log"]


def kid_for(public_raw: bytes) -> bytes:
    """Short key id: enough to pick a key from a ring, not meant to be collision-proof."""
    return hashlib.sha256(public_raw).digest()[:KID_LEN]


@dataclass(frozen=True, slots=True)
class KeyRecord:
    kid: bytes
    public_raw: bytes
    purpose: Purpose
    created_at: datetime
    revoked_at: datetime | None = None

    def kid_hex(self) -> str:
        return self.kid.hex()


class Signer(Protocol):
    @property
    def kid(self) -> bytes: ...

    def sign(self, message: bytes) -> bytes: ...

    def public_raw(self) -> bytes: ...


class Ed25519Signer:
    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._key = private_key
        self._public = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.kid = kid_for(self._public)

    @classmethod
    def generate(cls) -> Self:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_pem(cls, pem: bytes) -> Self:
        key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("PEM does not contain an Ed25519 private key")
        return cls(key)

    def to_pem(self) -> bytes:
        return self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)

    def public_raw(self) -> bytes:
        return self._public


def verify_signature(public_raw: bytes, message: bytes, signature: bytes) -> bool:
    """True iff ``signature`` is valid; malformed keys and signatures are simply invalid."""
    try:
        Ed25519PublicKey.from_public_bytes(public_raw).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True


class KeyRing:
    """Public keys known to the verifier. Thread-safe, read-mostly."""

    def __init__(self) -> None:
        self._records: dict[bytes, KeyRecord] = {}
        self._lock = threading.Lock()

    def add(self, rec: KeyRecord) -> None:
        with self._lock:
            if rec.kid in self._records:
                raise ValueError(f"key {rec.kid_hex()} is already in the ring")
            self._records[rec.kid] = rec

    def get(self, kid: bytes) -> KeyRecord:
        with self._lock:
            try:
                return self._records[kid]
            except KeyError:
                raise UnknownKeyError(f"unknown key id {kid.hex()}") from None

    def revoke(self, kid: bytes, at: datetime) -> KeyRecord:
        """Mark a key revoked from ``at``; revoking again keeps the earlier time."""
        with self._lock:
            rec = self._records.get(kid)
            if rec is None:
                raise UnknownKeyError(f"unknown key id {kid.hex()}")
            if rec.revoked_at is None or at < rec.revoked_at:
                rec = replace(rec, revoked_at=at)
                self._records[kid] = rec
            return rec

    def records(self) -> list[KeyRecord]:
        with self._lock:
            return list(self._records.values())

    def __contains__(self, kid: object) -> bool:
        with self._lock:
            return kid in self._records


@dataclass(frozen=True, slots=True)
class KeyMaterial:
    issuer: Ed25519Signer
    log: Ed25519Signer
    cdp_key: bytes
    keyring: KeyRing


class FileKeyStore:
    """Dev key provider. Layout under ``root``:

    ``issuer.pem``, ``log.pem``
        Ed25519 private keys (PKCS8 PEM, unencrypted, mode 0600).
    ``cdp.key``
        32 random bytes: the secret behind every copy-detection template.
    ``keyring.json``
        Public records, including revocations, as ``[{kid, public_b64, purpose,
        created_at, revoked_at}]``.

    A production deployment would put these behind a KMS; this class is the seam.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def load_or_create(self) -> KeyMaterial:
        """Load every key, generating whichever is missing. Safe to call repeatedly."""
        self._root.mkdir(parents=True, exist_ok=True)
        issuer = self._signer("issuer.pem")
        log = self._signer("log.pem")
        cdp_key = self._cdp_key()

        keyring = self._read_keyring()
        added = False
        own: list[tuple[Purpose, Ed25519Signer]] = [("issuer", issuer), ("log", log)]
        for purpose, signer in own:
            if signer.kid not in keyring:
                record = KeyRecord(signer.kid, signer.public_raw(), purpose, datetime.now(UTC))
                keyring.add(record)
                added = True
        if added:
            self.save_keyring(keyring)
        return KeyMaterial(issuer, log, cdp_key, keyring)

    def save_keyring(self, keyring: KeyRing) -> None:
        """Persist the public records, e.g. after a revocation."""
        payload = [
            {
                "kid": rec.kid_hex(),
                "public_b64": base64.b64encode(rec.public_raw).decode("ascii"),
                "purpose": rec.purpose,
                "created_at": rec.created_at.isoformat(),
                "revoked_at": rec.revoked_at.isoformat() if rec.revoked_at else None,
            }
            for rec in keyring.records()
        ]
        _write_atomic(self._root / "keyring.json", json.dumps(payload, indent=2).encode(), 0o644)

    def _signer(self, name: str) -> Ed25519Signer:
        path = self._root / name
        if path.exists():
            return Ed25519Signer.from_pem(path.read_bytes())
        signer = Ed25519Signer.generate()
        _write_atomic(path, signer.to_pem(), 0o600)
        return signer

    def _cdp_key(self) -> bytes:
        path = self._root / "cdp.key"
        if not path.exists():
            _write_atomic(path, secrets.token_bytes(CDP_KEY_LEN), 0o600)
        key = path.read_bytes()
        if len(key) != CDP_KEY_LEN:
            raise ValueError(f"{path} must hold exactly {CDP_KEY_LEN} bytes")
        return key

    def _read_keyring(self) -> KeyRing:
        keyring = KeyRing()
        path = self._root / "keyring.json"
        if not path.exists():
            return keyring
        for item in json.loads(path.read_text()):
            revoked = item["revoked_at"]
            keyring.add(
                KeyRecord(
                    kid=bytes.fromhex(item["kid"]),
                    public_raw=base64.b64decode(item["public_b64"]),
                    purpose=item["purpose"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    revoked_at=datetime.fromisoformat(revoked) if revoked else None,
                )
            )
        return keyring


def _write_atomic(path: Path, data: bytes, mode: int) -> None:
    """Write via a sibling temp file so a crash never leaves a half-written key."""
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
