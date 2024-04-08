"""Classical CDP baseline: normalised correlation between a capture and the template."""

from __future__ import annotations

import cv2
import numpy as np

# (a, b) of the logistic p = sigmoid(a * r + b) that maps Pearson r to [0, 1].
# Fitted on training data by ``training.train_cdp``; these are the shipped defaults.
DEFAULT_CALIBRATION: tuple[float, float] = (12.0, -3.0)


def template_image(template: np.ndarray, px_per_cell: int = 2, blur: float = 0.8) -> np.ndarray:
    """Render the *expected* appearance of a template: float32 in [0, 1], ink = 1.

    Cells are expanded to ``px_per_cell`` pixels and lightly blurred, which is
    roughly what a good capture of an ideal print looks like at that sampling.
    """
    ink = np.kron(template, np.ones((px_per_cell, px_per_cell), np.uint8)).astype(np.float32)
    return cv2.GaussianBlur(ink, (0, 0), blur) if blur > 0 else ink


def raw_correlation(patch: np.ndarray, template: np.ndarray) -> float:
    """Pearson r between the inverted patch (ink high) and the expected template image."""
    expected = template_image(template, px_per_cell=patch.shape[0] // template.shape[0])
    a = (1.0 - patch.astype(np.float64)).ravel()
    b = expected.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 1e-12 else 0.0


def correlation_score(
    patch: np.ndarray,
    template: np.ndarray,
    calibration: tuple[float, float] = DEFAULT_CALIBRATION,
) -> float:
    """Calibrated classical genuineness score in [0, 1]."""
    a, b = calibration
    return float(1.0 / (1.0 + np.exp(-(a * raw_correlation(patch, template) + b))))


__all__ = [
    "DEFAULT_CALIBRATION",
    "correlation_score",
    "raw_correlation",
    "template_image",
]
