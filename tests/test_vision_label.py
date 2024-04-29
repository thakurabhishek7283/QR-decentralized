from __future__ import annotations

import cv2
import numpy as np
import pytest
import segno

from sigil.cdp import generate_template
from sigil.vision import (
    CDP_CELLS,
    LABEL_H,
    LABEL_W,
    PATCH_PX,
    cdp_corners_label,
    label_png,
    qr_corners_label,
    render_label,
    render_qr_matrix,
)
from sigil.vision import geometry as g

BASE45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


def payload(n: int = 210, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "SG1:" + "".join(BASE45[i] for i in rng.integers(0, len(BASE45), n - 4))


def test_geometry_constants_are_consistent() -> None:
    assert (g.QR_MODULES, g.SYMBOL_PX, g.CDP_PX) == (53, 318, 318)
    assert (LABEL_W, LABEL_H) == (708, 366)
    assert g.CDP_ORIGIN == (366, 24)
    assert PATCH_PX == CDP_CELLS * 2 == 212
    assert qr_corners_label().tolist() == [[24, 24], [342, 24], [342, 342], [24, 342]]
    assert cdp_corners_label()[0].tolist() == [366, 24]


def test_max_payload_fits_version_9_m() -> None:
    qr = segno.make(payload(210), version=9, error="m", boost_error=False, mode="alphanumeric")
    assert qr.version == 9
    assert qr.symbol_size(border=0) == (53, 53)
    assert render_qr_matrix(payload(210)).shape == (53, 53)


def test_oversized_payload_is_rejected() -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - segno raises its own ValueError subclasses
        render_qr_matrix(payload(300))


def test_render_label_layout() -> None:
    template = generate_template(bytes(32), bytes(12))
    label = render_label(payload(), template)
    assert label.shape == (LABEL_H, LABEL_W)
    assert label.dtype == np.float32
    assert set(np.unique(label)) == {0.0, 1.0}
    x, y = g.CDP_ORIGIN
    cells = label[y : y + g.CDP_PX : g.CELL_PX, x : x + g.CDP_PX : g.CELL_PX]
    assert np.array_equal(cells == 0.0, template.astype(bool))  # ink = 1 -> black
    assert label[:, : g.QUIET_PX].min() == 1.0  # quiet zones are paper
    assert label[: g.QUIET_PX].min() == 1.0


def test_rendered_label_decodes() -> None:
    text = payload(seed=3)
    label = render_label(text, generate_template(bytes(32), bytes(12)))
    big = cv2.resize(label, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    gray = cv2.copyMakeBorder(
        (big * 255).astype(np.uint8), 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255
    )
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(gray)
    assert decoded == text


def test_label_png_round_trips() -> None:
    label = render_label(payload(), generate_template(bytes(32), bytes(12)))
    decoded = cv2.imdecode(np.frombuffer(label_png(label), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded.shape == label.shape
    assert np.array_equal(decoded == 255, label == 1.0)


def test_render_label_rejects_wrong_template_size() -> None:
    with pytest.raises(ValueError, match="template must be"):
        render_label(payload(), np.zeros((10, 10), dtype=np.uint8))
