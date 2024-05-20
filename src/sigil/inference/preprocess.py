"""Model input construction - the single source of truth shared by training and serving."""

from __future__ import annotations

import numpy as np

from sigil.cdp.features import template_image
from sigil.vision.geometry import CDP_CELLS, PATCH_PX

INPUT_SHAPE = (2, PATCH_PX, PATCH_PX)
_PX_PER_CELL = PATCH_PX // CDP_CELLS
_EPS = 1e-6


def make_input(patch: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Stack the captured patch and the expected template into the CNN input.

    Channel 0 is the patch inverted (ink = high) and standardised per sample, so
    exposure and contrast drop out. Channel 1 is the expected template image
    mapped to {-1, +1} (ink = +1). Returns float32 of shape ``INPUT_SHAPE``.
    """
    if patch.shape != (PATCH_PX, PATCH_PX):
        raise ValueError(f"patch must be {PATCH_PX}x{PATCH_PX}, got {patch.shape}")
    ink = 1.0 - patch.astype(np.float32)
    ink = (ink - ink.mean()) / (ink.std() + _EPS)
    expected = template_image(template, px_per_cell=_PX_PER_CELL) * 2.0 - 1.0
    return np.stack([ink, expected]).astype(np.float32)
