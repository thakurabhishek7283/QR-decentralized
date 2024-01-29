"""Domain errors. Services raise these; only the API layer maps them to HTTP."""

from __future__ import annotations


class SigilError(Exception):
    """Base class for every error Sigil raises on purpose."""


class DecodeError(SigilError):
    """A QR payload or credential byte string is malformed."""


class SignatureError(SigilError):
    """A signature does not verify."""


class UnknownKeyError(SigilError):
    """A key id is not in the key ring."""


class LogError(SigilError):
    """The transparency log cannot satisfy a request."""


class NotFoundError(SigilError):
    """A requested record does not exist."""


class CaptureError(SigilError):
    """Vision could not extract what it needs from an image.

    ``reason`` is a stable machine-readable code: ``bad_image``, ``no_qr``,
    ``low_quality`` or ``align_failed``.
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason


class OverloadedError(SigilError):
    """The inference queue is full; the caller should retry later."""
