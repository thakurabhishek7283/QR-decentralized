"""Rendering of the printable label: a QR symbol next to the CDP square."""

from __future__ import annotations

import cv2
import numpy as np
import segno

from sigil.vision.geometry import (
    CDP_CELLS,
    CDP_ORIGIN,
    CDP_PX,
    CELL_PX,
    LABEL_H,
    LABEL_W,
    MODULE_PX,
    QR_ERROR,
    QR_MODULES,
    QR_ORIGIN,
    QR_VERSION,
    SYMBOL_PX,
)


def render_qr_matrix(payload: str) -> np.ndarray:
    """Return the QR module matrix (bool, 53x53, True = dark).

    The version is pinned so every label has identical geometry; an oversized
    payload raises instead of silently growing the symbol.
    """
    qr = segno.make(
        payload, version=QR_VERSION, error=QR_ERROR.lower(), boost_error=False, mode="alphanumeric"
    )
    matrix = np.array(qr.matrix, dtype=bool)
    if matrix.shape != (QR_MODULES, QR_MODULES):
        raise ValueError(f"unexpected QR size {matrix.shape}")
    return matrix


def render_label(payload: str, template: np.ndarray) -> np.ndarray:
    """Render the label raster: float32 (LABEL_H, LABEL_W), 1 = paper, 0 = ink."""
    if template.shape != (CDP_CELLS, CDP_CELLS):
        raise ValueError(f"template must be {CDP_CELLS}x{CDP_CELLS}, got {template.shape}")
    label = np.ones((LABEL_H, LABEL_W), dtype=np.float32)
    qx, qy = QR_ORIGIN
    modules = np.kron(render_qr_matrix(payload), np.ones((MODULE_PX, MODULE_PX), dtype=bool))
    label[qy : qy + SYMBOL_PX, qx : qx + SYMBOL_PX] = np.where(modules, 0.0, 1.0)
    cx, cy = CDP_ORIGIN
    cells = np.kron(template.astype(bool), np.ones((CELL_PX, CELL_PX), dtype=bool))
    label[cy : cy + CDP_PX, cx : cx + CDP_PX] = np.where(cells, 0.0, 1.0)
    return label


def label_png(label: np.ndarray) -> bytes:
    """Encode a label raster as PNG bytes."""
    ok, buf = cv2.imencode(".png", np.clip(label * 255.0, 0, 255).astype(np.uint8))
    if not ok:  # pragma: no cover - imencode only fails on invalid input
        raise ValueError("PNG encoding failed")
    return buf.tobytes()
