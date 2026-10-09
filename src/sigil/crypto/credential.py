"""The signed credential carried by a label's QR code.

The binary layout is fixed-width and big-endian, so every credential has exactly
one encoding and a decoder has nothing to be lenient about.
"""

from __future__ import annotations

import secrets
import struct
import time
from dataclasses import dataclass
from typing import Self

from sigil.crypto.base45 import b45decode, b45encode
from sigil.crypto.keys import KID_LEN, KeyRecord, KeyRing, Signer, verify_signature
from sigil.errors import DecodeError, SignatureError

QR_PREFIX = "SG1:"
MAGIC = b"SG"
VERSION = 1
DOMAIN = b"sigil-credential-v1\x00"
SERIAL_LEN = 12
SIGNATURE_LEN = 64
SKU_MAX = 32
BATCH_MAX = 16

_HEADER = struct.Struct(f">2sB{KID_LEN}s{SERIAL_LEN}sI")
_UINT32_MAX = 0xFFFFFFFF


@dataclass(frozen=True, slots=True)
class Credential:
    kid: bytes
    serial: bytes
    iat: int
    sku: str
    batch: str
    version: int = VERSION

    def __post_init__(self) -> None:
        if self.version != VERSION:
            raise ValueError(f"unsupported credential version {self.version}")
        if len(self.kid) != KID_LEN:
            raise ValueError(f"kid must be {KID_LEN} bytes")
        if len(self.serial) != SERIAL_LEN:
            raise ValueError(f"serial must be {SERIAL_LEN} bytes")
        if not 0 <= self.iat <= _UINT32_MAX:
            raise ValueError("iat must fit in an unsigned 32-bit integer")
        _check_text("sku", self.sku, SKU_MAX)
        _check_text("batch", self.batch, BATCH_MAX)

    def body(self) -> bytes:
        """Everything the signature covers (after the domain separator)."""
        sku, batch = self.sku.encode("ascii"), self.batch.encode("ascii")
        header = _HEADER.pack(MAGIC, self.version, self.kid, self.serial, self.iat)
        return b"".join((header, bytes([len(sku)]), sku, bytes([len(batch)]), batch))

    @property
    def serial_hex(self) -> str:
        return self.serial.hex()


@dataclass(frozen=True, slots=True)
class SignedCredential:
    credential: Credential
    signature: bytes

    def __post_init__(self) -> None:
        if len(self.signature) != SIGNATURE_LEN:
            raise ValueError(f"signature must be {SIGNATURE_LEN} bytes")

    def to_bytes(self) -> bytes:
        return self.credential.body() + self.signature

    @classmethod
    def from_bytes(cls, raw: bytes) -> Self:
        """Parse strictly: any deviation, trailing bytes included, is a :class:`DecodeError`."""
        if len(raw) < _HEADER.size:
            raise DecodeError("credential is truncated")
        magic, version, kid, serial, iat = _HEADER.unpack_from(raw)
        if magic != MAGIC:
            raise DecodeError("bad credential magic")
        if version != VERSION:
            raise DecodeError(f"unsupported credential version {version}")
        sku, pos = _read_text(raw, _HEADER.size, SKU_MAX, "sku")
        batch, pos = _read_text(raw, pos, BATCH_MAX, "batch")
        if len(raw) - pos != SIGNATURE_LEN:
            raise DecodeError("credential has a wrong signature length or trailing bytes")
        try:
            credential = Credential(kid=kid, serial=serial, iat=iat, sku=sku, batch=batch)
        except ValueError as exc:
            raise DecodeError(str(exc)) from exc
        return cls(credential, raw[pos:])

    def to_qr_payload(self) -> str:
        return QR_PREFIX + b45encode(self.to_bytes())

    @classmethod
    def from_qr_payload(cls, text: str) -> Self:
        if not text.startswith(QR_PREFIX):
            raise DecodeError("QR payload does not start with the Sigil prefix")
        return cls.from_bytes(b45decode(text.removeprefix(QR_PREFIX)))

    def leaf_data(self) -> bytes:
        """What the transparency log commits to."""
        return self.to_bytes()

    def verify(self, keyring: KeyRing) -> KeyRecord:
        """Check the signature against the key named in the credential.

        Revocation is deliberately not judged here: whether a revoked key matters
        depends on ``iat``, which is a policy decision.
        """
        record = keyring.get(self.credential.kid)
        if not verify_signature(record.public_raw, DOMAIN + self.credential.body(), self.signature):
            raise SignatureError("credential signature does not verify")
        return record


def issue(
    signer: Signer,
    *,
    sku: str,
    batch: str,
    iat: int | None = None,
    serial: bytes | None = None,
) -> SignedCredential:
    """Mint and sign a credential; ``serial`` defaults to 96 fresh random bits."""
    credential = Credential(
        kid=signer.kid,
        serial=serial if serial is not None else secrets.token_bytes(SERIAL_LEN),
        iat=int(time.time()) if iat is None else iat,
        sku=sku,
        batch=batch,
    )
    return SignedCredential(credential, signer.sign(DOMAIN + credential.body()))


def _check_text(name: str, text: str, max_len: int) -> None:
    if not 1 <= len(text) <= max_len:
        raise ValueError(f"{name} must be 1..{max_len} characters")
    if not (text.isascii() and text.isprintable()):
        raise ValueError(f"{name} must be printable ASCII")


def _read_text(raw: bytes, pos: int, max_len: int, name: str) -> tuple[str, int]:
    """Read one length-prefixed ASCII field; semantic checks happen in ``Credential``."""
    if pos >= len(raw):
        raise DecodeError(f"credential is truncated before {name}")
    length = raw[pos]
    if not 1 <= length <= max_len:
        raise DecodeError(f"{name} length {length} is out of range")
    end = pos + 1 + length
    if end > len(raw):
        raise DecodeError(f"credential is truncated inside {name}")
    try:
        return raw[pos + 1 : end].decode("ascii"), end
    except UnicodeDecodeError:
        raise DecodeError(f"{name} is not ASCII") from None
