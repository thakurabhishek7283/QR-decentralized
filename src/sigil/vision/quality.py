"""Capture-quality gate: is this photo good enough to trust a CDP score?"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Thresholds are on contrast-normalised label-frame pixels; see ``assess_quality``.
MIN_SHARPNESS = 0.006
MIN_BRIGHTNESS = 0.25
MAX_CLIPPED = 0.60


@dataclass(frozen=True, slots=True)
class QualityReport:
    sharpness: float  # variance of the Laplacian (contrast-normalised)
    brightness: float  # mean intensity, 0..1
    clipped: float  # fraction of pixels at 0 or 255
    ok: bool
    reasons: list[str]  # "blurry", "too_dark", "overexposed"


def assess_quality(gray: np.ndarray, roi: np.ndarray | None = None) -> QualityReport:
    """Measure sharpness, exposure and clipping of ``gray`` (uint8 or float in [0, 1]).

    ``roi`` is an optional polygon, shape (N, 2) in image px; statistics are
    taken inside it. Sharpness is the variance of the Laplacian after stretching
    the 2nd..98th percentile to [0, 1], so it measures edge acuity rather than
    how contrasty the scene happens to be.
    """
    img = gray.astype(np.float32)
    if gray.dtype == np.uint8:
        img /= 255.0
    mask = None
    if roi is not None:
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(roi).astype(np.int32)], 1)
    pixels = img[mask.astype(bool)] if mask is not None else img.ravel()
    if pixels.size == 0:
        return QualityReport(0.0, 0.0, 1.0, False, ["blurry", "too_dark"])

    lo, hi = np.percentile(pixels, [2, 98])
    stretched = np.clip((img - lo) / max(float(hi - lo), 1e-3), 0.0, 1.0).astype(np.float32)
    lap = cv2.Laplacian(stretched, cv2.CV_32F)
    sharpness = float((lap[mask.astype(bool)] if mask is not None else lap).var())
    brightness = float(pixels.mean())
    clipped = float(np.mean((pixels <= 1.5 / 255.0) | (pixels >= 253.5 / 255.0)))

    reasons = []
    if sharpness < MIN_SHARPNESS:
        reasons.append("blurry")
    if brightness < MIN_BRIGHTNESS:
        reasons.append("too_dark")
    if clipped > MAX_CLIPPED:
        reasons.append("overexposed")
    return QualityReport(sharpness, brightness, clipped, not reasons, reasons)
