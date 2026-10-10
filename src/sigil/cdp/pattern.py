"""Keyed copy-detection pattern (CDP) template generation."""

from __future__ import annotations

import hashlib
import hmac

import numpy as np

from sigil.vision.geometry import CDP_CELLS


def generate_template(cdp_key: bytes, serial: bytes, cells: int = CDP_CELLS) -> np.ndarray:
    """Derive the secret CDP grid for one serial.

    The bit stream is ``HMAC-SHA256(cdp_key, serial || counter_be32)`` for
    counter = 0, 1, ..., unpacked MSB first and truncated to ``cells**2`` bits
    (an HMAC counter-mode DRBG). Without ``cdp_key`` the grid is
    indistinguishable from random, so an attacker holding only a photo of a
    label cannot regenerate it - they can only copy the print.

    Returns a ``uint8`` array of shape ``(cells, cells)`` where 1 means ink and
    the expected density is 0.5.
    """
    n_bits = cells * cells
    n_blocks = -(-n_bits // 256)
    stream = b"".join(
        hmac.new(cdp_key, serial + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        for counter in range(n_blocks)
    )
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[:n_bits]
    return bits.reshape(cells, cells).copy()
