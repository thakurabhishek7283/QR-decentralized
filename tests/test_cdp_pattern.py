from __future__ import annotations

import numpy as np
import pytest

from sigil.cdp import generate_template
from sigil.vision.geometry import CDP_CELLS

KEY = bytes(range(32))
SERIAL = bytes(range(12))


def test_template_shape_dtype_and_values() -> None:
    t = generate_template(KEY, SERIAL)
    assert t.shape == (CDP_CELLS, CDP_CELLS)
    assert t.dtype == np.uint8
    assert set(np.unique(t)) <= {0, 1}


def test_template_is_deterministic() -> None:
    assert np.array_equal(generate_template(KEY, SERIAL), generate_template(KEY, SERIAL))


def test_template_density_is_about_half() -> None:
    assert generate_template(KEY, SERIAL).mean() == pytest.approx(0.5, abs=0.02)


def test_template_is_keyed_and_serial_specific() -> None:
    base = generate_template(KEY, SERIAL)
    other_key = generate_template(bytes(32), SERIAL)
    other_serial = generate_template(KEY, bytes(12))
    for other in (other_key, other_serial):
        # independent random grids disagree on about half of the cells
        assert np.mean(base != other) == pytest.approx(0.5, abs=0.03)


def test_template_supports_other_sizes() -> None:
    small = generate_template(KEY, SERIAL, cells=16)
    assert small.shape == (16, 16)
    # a prefix of the same bit stream
    assert np.array_equal(small.ravel(), generate_template(KEY, SERIAL).ravel()[:256])
