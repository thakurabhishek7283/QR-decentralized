from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sigil.crypto import (
    QR_PREFIX,
    Credential,
    Ed25519Signer,
    KeyRecord,
    KeyRing,
    SignedCredential,
    issue,
    verify_signature,
)
from sigil.crypto.base45 import ALPHABET
from sigil.crypto.credential import DOMAIN
from sigil.errors import DecodeError, SignatureError, UnknownKeyError

SERIAL = bytes(range(12))
IAT = 1_750_000_000
_SIGNER = Ed25519Signer.generate()
_SAMPLE = issue(_SIGNER, sku="WIDGET-9", batch="B0001", iat=IAT, serial=SERIAL)


@pytest.fixture
def signer() -> Ed25519Signer:
    return _SIGNER


@pytest.fixture
def keyring(signer: Ed25519Signer) -> KeyRing:
    ring = KeyRing()
    ring.add(KeyRecord(signer.kid, signer.public_raw(), "issuer", datetime.now(UTC)))
    return ring


@pytest.fixture
def signed(signer: Ed25519Signer) -> SignedCredential:
    return issue(signer, sku="WIDGET-9", batch="B0001", iat=IAT, serial=SERIAL)


def test_binary_layout(signer: Ed25519Signer, signed: SignedCredential) -> None:
    raw = signed.to_bytes()
    assert raw[:2] == b"SG"
    assert raw[2] == 1
    assert raw[3:7] == signer.kid
    assert raw[7:19] == SERIAL
    assert int.from_bytes(raw[19:23], "big") == IAT
    assert raw[23] == len("WIDGET-9")
    assert raw[24:32] == b"WIDGET-9"
    assert raw[32] == len("B0001")
    assert raw[33:38] == b"B0001"
    assert len(raw) == 38 + 64
    assert signed.leaf_data() == raw


def test_signature_covers_domain_separated_body(
    signer: Ed25519Signer, signed: SignedCredential
) -> None:
    assert verify_signature(
        signer.public_raw(), DOMAIN + signed.credential.body(), signed.signature
    )
    assert not verify_signature(signer.public_raw(), signed.credential.body(), signed.signature)


def test_issue_defaults_to_fresh_random_serials_and_current_time(signer: Ed25519Signer) -> None:
    a = issue(signer, sku="A", batch="B")
    b = issue(signer, sku="A", batch="B")
    assert a.credential.serial != b.credential.serial
    assert len(a.credential.serial) == 12
    assert a.credential.serial_hex == a.credential.serial.hex()
    assert a.credential.iat > 1_700_000_000


def test_qr_round_trip_and_verify(signed: SignedCredential, keyring: KeyRing) -> None:
    payload = signed.to_qr_payload()
    assert payload.startswith(QR_PREFIX)
    parsed = SignedCredential.from_qr_payload(payload)
    assert parsed == signed
    assert parsed.verify(keyring).kid == signed.credential.kid


def test_verify_rejects_unknown_key(signed: SignedCredential) -> None:
    with pytest.raises(UnknownKeyError):
        signed.verify(KeyRing())


def test_verify_rejects_other_issuers_signature(signed: SignedCredential, keyring: KeyRing) -> None:
    forged = SignedCredential(signed.credential, Ed25519Signer.generate().sign(b"x"))
    with pytest.raises(SignatureError):
        forged.verify(keyring)


def test_verify_does_not_judge_revocation(signed: SignedCredential, keyring: KeyRing) -> None:
    keyring.revoke(signed.credential.kid, datetime(2020, 1, 1, tzinfo=UTC))
    assert signed.verify(keyring).revoked_at is not None


def test_flipping_any_byte_is_detected(signed: SignedCredential, keyring: KeyRing) -> None:
    raw = signed.to_bytes()
    for i in range(len(raw)):
        mutated = bytearray(raw)
        mutated[i] ^= 0x01
        with pytest.raises((DecodeError, SignatureError, UnknownKeyError)):
            SignedCredential.from_bytes(bytes(mutated)).verify(keyring)


def test_flipping_any_payload_character_is_detected(
    signed: SignedCredential, keyring: KeyRing
) -> None:
    payload = signed.to_qr_payload()
    for i in range(len(QR_PREFIX), len(payload)):
        other = ALPHABET[(ALPHABET.index(payload[i]) + 1) % len(ALPHABET)]
        mutated = payload[:i] + other + payload[i + 1 :]
        with pytest.raises((DecodeError, SignatureError, UnknownKeyError)):
            SignedCredential.from_qr_payload(mutated).verify(keyring)


def test_trailing_and_truncated_bytes_are_rejected(signed: SignedCredential) -> None:
    raw = signed.to_bytes()
    with pytest.raises(DecodeError):
        SignedCredential.from_bytes(raw + b"\x00")
    for cut in range(len(raw)):
        with pytest.raises(DecodeError):
            SignedCredential.from_bytes(raw[:cut])


def test_bad_header_fields_are_rejected(signed: SignedCredential) -> None:
    raw = signed.to_bytes()
    with pytest.raises(DecodeError, match="magic"):
        SignedCredential.from_bytes(b"XX" + raw[2:])
    with pytest.raises(DecodeError, match="version"):
        SignedCredential.from_bytes(raw[:2] + b"\x02" + raw[3:])


def _with_fields(signed: SignedCredential, sku: bytes, batch: bytes) -> bytes:
    raw = signed.to_bytes()
    return raw[:23] + bytes([len(sku)]) + sku + bytes([len(batch)]) + batch + raw[-64:]


@pytest.mark.parametrize(
    ("sku", "batch"),
    [
        (b"", b"B1"),  # empty sku
        (b"S" * 33, b"B1"),  # sku too long
        (b"S1", b""),  # empty batch
        (b"S1", b"B" * 17),  # batch too long
        ("café".encode(), b"B1"),  # non-ASCII
        (b"S\x00", b"B1"),  # control character
        (b"S\x7f", b"B1"),  # DEL
        (b"S1", b"B\n"),  # newline
    ],
)
def test_invalid_text_fields_are_rejected(
    signed: SignedCredential, sku: bytes, batch: bytes
) -> None:
    with pytest.raises(DecodeError):
        SignedCredential.from_bytes(_with_fields(signed, sku, batch))


def test_boundary_text_fields_are_accepted(signed: SignedCredential) -> None:
    raw = _with_fields(signed, b" " + b"~" * 31, b"~" * 16)
    parsed = SignedCredential.from_bytes(raw)
    assert parsed.credential.sku == " " + "~" * 31
    assert parsed.to_bytes() == raw


def test_qr_payload_requires_prefix_and_base45() -> None:
    for text in ("", "SG2:ABC", "ABC", "SG1:abc", "SG1:A"):
        with pytest.raises(DecodeError):
            SignedCredential.from_qr_payload(text)


def test_maximum_payload_fits_qr_version_9_m(signer: Ed25519Signer) -> None:
    biggest = issue(signer, sku="~" * 32, batch="~" * 16, iat=2**32 - 1)
    assert len(biggest.to_bytes()) == 137
    assert len(biggest.to_qr_payload()) <= 210


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kid": b"abc"}, "kid"),
        ({"serial": b"short"}, "serial"),
        ({"iat": -1}, "iat"),
        ({"iat": 2**32}, "iat"),
        ({"sku": ""}, "sku"),
        ({"sku": "S" * 33}, "sku"),
        ({"batch": ""}, "batch"),
        ({"batch": "B" * 17}, "batch"),
        ({"sku": "naïve"}, "printable ASCII"),
        ({"batch": "tab\there"}, "printable ASCII"),
        ({"version": 2}, "version"),
    ],
)
def test_credential_validates_its_fields(overrides: dict[str, Any], message: str) -> None:
    fields: dict[str, Any] = {
        "kid": b"abcd",
        "serial": SERIAL,
        "iat": IAT,
        "sku": "S",
        "batch": "B",
    }
    with pytest.raises(ValueError, match=message):
        Credential(**{**fields, **overrides})


@given(st.binary(max_size=200))
def test_random_bytes_decode_canonically_or_raise_decode_error(raw: bytes) -> None:
    try:
        parsed = SignedCredential.from_bytes(raw)
    except DecodeError:
        return
    assert parsed.to_bytes() == raw


@given(st.text(max_size=300))
def test_random_text_only_raises_decode_error(text: str) -> None:
    with contextlib.suppress(DecodeError):
        SignedCredential.from_qr_payload(text)


@given(
    st.integers(min_value=0, max_value=200),
    st.binary(max_size=8),
    st.integers(min_value=0, max_value=200),
)
def test_splicing_a_valid_credential_never_escapes_as_another_error(
    cut: int, insert: bytes, where: int
) -> None:
    raw = _SAMPLE.to_bytes()
    spliced = raw[: min(where, len(raw))] + insert + raw[min(where, len(raw)) + cut % 5 :]
    try:
        parsed = SignedCredential.from_bytes(spliced)
    except DecodeError:
        return
    assert parsed.to_bytes() == spliced


@given(
    st.text(
        alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), min_size=1, max_size=32
    ),
    st.text(
        alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), min_size=1, max_size=16
    ),
    st.binary(min_size=12, max_size=12),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_every_valid_credential_round_trips(sku: str, batch: str, serial: bytes, iat: int) -> None:
    signed = issue(_SIGNER, sku=sku, batch=batch, iat=iat, serial=serial)
    assert SignedCredential.from_bytes(signed.to_bytes()) == signed
    assert SignedCredential.from_qr_payload(signed.to_qr_payload()) == signed
    assert len(signed.to_qr_payload()) <= 210
