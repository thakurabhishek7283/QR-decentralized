from __future__ import annotations

import contextlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sigil.crypto import b45decode, b45encode
from sigil.crypto.base45 import ALPHABET
from sigil.errors import DecodeError

RFC_9285_VECTORS = [
    (b"AB", "BB8"),
    (b"Hello!!", "%69 VD92EX0"),
    (b"base-45", "UJCLQE7W581"),
    (b"ietf!", "QED8WEX0"),
    (b"", ""),
]


@pytest.mark.parametrize(("raw", "text"), RFC_9285_VECTORS)
def test_rfc_vectors(raw: bytes, text: str) -> None:
    assert b45encode(raw) == text
    assert b45decode(text) == raw


def test_extremes_of_each_group_size() -> None:
    assert b45encode(b"\xff\xff") == "FGW"
    assert b45decode("FGW") == b"\xff\xff"
    assert b45encode(b"\xff") == "U5"
    assert b45decode("U5") == b"\xff"


@given(st.binary(max_size=400))
def test_round_trip(raw: bytes) -> None:
    text = b45encode(raw)
    assert set(text) <= set(ALPHABET)
    assert b45decode(text) == raw


@given(st.text(alphabet=ALPHABET, max_size=300))
def test_decode_is_canonical_when_it_succeeds(text: str) -> None:
    try:
        raw = b45decode(text)
    except DecodeError:
        return
    assert b45encode(raw) == text


@given(st.text(max_size=100))
def test_arbitrary_text_only_ever_raises_decode_error(text: str) -> None:
    with contextlib.suppress(DecodeError):
        b45decode(text)


@pytest.mark.parametrize(
    "text",
    [
        "A",  # impossible length (1 mod 3)
        "ABCD",  # impossible length (1 mod 3)
        "abc",  # lowercase is outside the alphabet
        "AéC",  # non-ASCII
        "GGW",  # three-char group above 65535
        "::",  # two-char group above 255
        "V6",  # 255 + 1
    ],
)
def test_malformed_text_is_rejected(text: str) -> None:
    with pytest.raises(DecodeError):
        b45decode(text)
