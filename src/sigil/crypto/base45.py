"""Base45 (RFC 9285): the encoding that lets a binary credential use the QR alphanumeric mode."""

from __future__ import annotations

from sigil.errors import DecodeError

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
_VALUES = {char: value for value, char in enumerate(ALPHABET)}
_MAX_PAIR = 0xFFFF


def b45encode(data: bytes) -> str:
    out: list[str] = []
    for i in range(0, len(data) - 1, 2):
        rest, c = divmod(data[i] * 256 + data[i + 1], 45)
        e, d = divmod(rest, 45)
        out.append(ALPHABET[c] + ALPHABET[d] + ALPHABET[e])
    if len(data) % 2:
        d, c = divmod(data[-1], 45)
        out.append(ALPHABET[c] + ALPHABET[d])
    return "".join(out)


def b45decode(text: str) -> bytes:
    """Invert :func:`b45encode`; every malformed input raises :class:`DecodeError`."""
    if len(text) % 3 == 1:
        raise DecodeError("base45 text has an impossible length")
    try:
        values = [_VALUES[char] for char in text]
    except KeyError as exc:
        raise DecodeError(f"character {exc.args[0]!r} is not in the base45 alphabet") from None
    out = bytearray()
    for i in range(0, len(values), 3):
        group = values[i : i + 3]
        n = sum(value * 45**place for place, value in enumerate(group))
        if len(group) == 3:
            if n > _MAX_PAIR:
                raise DecodeError("base45 group overflows two bytes")
            out += n.to_bytes(2, "big")
        else:
            if n > 0xFF:
                raise DecodeError("base45 group overflows one byte")
            out.append(n)
    return bytes(out)
