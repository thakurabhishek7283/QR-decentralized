"""The demo "camera": print a label, optionally counterfeit it, and photograph it.

Nobody can hold a real label up to a browser, so the demo simulates the physical world with the
same channel model the model is trained and evaluated on (``sigil.cdp``).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import cv2
import numpy as np

from sigil.api.schemas import SimulateRequest
from sigil.cdp import (
    Attack,
    capture,
    counterfeit_from,
    print_label,
    sample_capture_profile,
    sample_print_profile,
)
from sigil.cdp.channel import Domain, Quality
from sigil.errors import CaptureError
from sigil.vision import decode

DOMAIN: Domain = "test"  # the held-out channel the evals use
JPEG_QUALITY = 95
MAX_SHOTS = 3  # a user retakes a photo the camera app cannot read


@dataclass(frozen=True, slots=True)
class Photo:
    jpeg: bytes


class SimulationStudio:
    def shoot(self, label: np.ndarray, quality: Quality, request: SimulateRequest) -> Photo:
        """Print ``label`` (copied by an attacker first, if requested) and photograph it.

        CPU-bound (about a tenth of a second); call it from a worker thread. A shot whose QR code
        cannot be found is retaken, except at ``hard`` difficulty, where failure is the point.
        """
        seed = request.seed if request.seed is not None else secrets.randbits(63)
        rng = np.random.default_rng(seed)
        printed = print_label(label, sample_print_profile(rng, DOMAIN, quality), rng)
        if request.attack is not None:
            printed = counterfeit_from(printed, Attack(request.attack), rng, DOMAIN)

        shots = 1 if request.difficulty == "hard" else MAX_SHOTS
        for _ in range(shots):
            photo = capture(printed, sample_capture_profile(rng, DOMAIN, request.difficulty), rng)
            if _readable(photo):
                break
        ok, jpeg = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:  # pragma: no cover - imencode only fails on invalid input
            raise RuntimeError("could not encode the simulated photo")
        return Photo(jpeg.tobytes())


def _readable(photo: np.ndarray) -> bool:
    try:
        decode(photo)
    except CaptureError:
        return False
    return True
