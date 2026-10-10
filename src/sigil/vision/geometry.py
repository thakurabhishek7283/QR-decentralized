"""Canonical label geometry (fixed for v1).

Everything downstream works in the *label frame*: the raster that issuance
prints, at roughly 600 dpi. A capture is mapped back into this frame with a
homography estimated from the QR corners, so a CDP cell always lands on the same
pixels regardless of how the phone was held.
"""

from __future__ import annotations

import numpy as np

QR_VERSION = 9  # 53 modules; alphanumeric capacity at EC level M is 262 chars
QR_ERROR = "M"
QR_MODULES = 53
MODULE_PX = 6  # label raster ~ 600 dpi, so one module ~ 0.25 mm
QUIET_MODULES = 4
CDP_CELLS = 106  # two cells per QR module
CELL_PX = 3
SYMBOL_PX = QR_MODULES * MODULE_PX  # 318
CDP_PX = CDP_CELLS * CELL_PX  # 318
QUIET_PX = QUIET_MODULES * MODULE_PX  # 24

# Label canvas, left to right: [quiet][QR symbol][quiet][CDP][quiet].
LABEL_W = QUIET_PX * 3 + SYMBOL_PX + CDP_PX  # 708
LABEL_H = QUIET_PX * 2 + SYMBOL_PX  # 366
QR_ORIGIN = (QUIET_PX, QUIET_PX)  # (x, y) of the symbol's top-left
CDP_ORIGIN = (QUIET_PX * 2 + SYMBOL_PX, QUIET_PX)  # (366, 24)
PATCH_PX = 212  # model input side: two pixels per CDP cell


def _corners(origin: tuple[int, int], side: int) -> np.ndarray:
    x, y = origin
    return np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=np.float32)


def qr_corners_label() -> np.ndarray:
    """Outer corners of the QR symbol in label px (float32, 4x2): TL, TR, BR, BL."""
    return _corners(QR_ORIGIN, SYMBOL_PX)


def cdp_corners_label() -> np.ndarray:
    """Outer corners of the CDP square in label px (float32, 4x2): TL, TR, BR, BL."""
    return _corners(CDP_ORIGIN, CDP_PX)
