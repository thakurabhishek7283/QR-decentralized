"""The demo "camera": print a label, optionally counterfeit it, and photograph it.

Nobody can hold a real label up to a browser, so the demo simulates the physical world with the
same channel model the model is trained and evaluated on (``sigil.cdp``). The physical item that
was photographed is remembered briefly so the investigator's ``request_recapture`` tool can take
another photo of *the same print*, as a user asked to retake would.
"""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
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
from sigil.evals.world import SimulatedRecapture
from sigil.vision import decode

DOMAIN: Domain = "test"  # the held-out channel the evals use
MAX_ITEMS = 32  # a printed label is ~1 MB as a float raster
JPEG_QUALITY = 95
MAX_SHOTS = 3  # a user retakes a photo the camera app cannot read


@dataclass(frozen=True, slots=True)
class Photo:
    jpeg: bytes
    item_id: str  # handle to the photographed print, for recaptures


class SimulationStudio:
    def __init__(self) -> None:
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()  # shots run in worker threads

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
        return Photo(jpeg.tobytes(), self._remember(printed))

    def camera_for(self, item_id: str | None, seed: int | None = None) -> SimulatedRecapture | None:
        """A source of further photos of a previously photographed print, if it is still known.

        A ``seed`` makes the retakes reproducible, which a trace replay needs."""
        with self._lock:
            printed = self._items.get(item_id) if item_id else None
        return None if printed is None else SimulatedRecapture(printed, np.random.default_rng(seed))

    def _remember(self, printed: np.ndarray) -> str:
        item_id = secrets.token_hex(8)
        with self._lock:
            self._items[item_id] = printed
            while len(self._items) > MAX_ITEMS:
                self._items.popitem(last=False)
        return item_id


def _readable(photo: np.ndarray) -> bool:
    try:
        decode(photo)
    except CaptureError:
        return False
    return True
