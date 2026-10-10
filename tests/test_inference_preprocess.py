from __future__ import annotations

import numpy as np
import pytest

from sigil.cdp import generate_template, template_image
from sigil.inference import INPUT_SHAPE, make_input
from sigil.vision import PATCH_PX


@pytest.fixture
def template() -> np.ndarray:
    return generate_template(bytes(32), bytes(12))


def test_input_shape_and_dtype(template: np.ndarray) -> None:
    x = make_input(
        np.random.default_rng(0).random((PATCH_PX, PATCH_PX)).astype(np.float32), template
    )
    assert x.shape == INPUT_SHAPE == (2, 212, 212)
    assert x.dtype == np.float32


def test_patch_channel_is_standardised_and_inverted(template: np.ndarray) -> None:
    patch = np.random.default_rng(1).random((PATCH_PX, PATCH_PX)).astype(np.float32)
    ink = make_input(patch, template)[0]
    assert ink.mean() == pytest.approx(0.0, abs=1e-4)
    assert ink.std() == pytest.approx(1.0, abs=1e-3)
    assert np.corrcoef(ink.ravel(), patch.ravel())[0, 1] < -0.999  # ink (dark) maps high


def test_template_channel_is_pm_one_with_ink_positive(template: np.ndarray) -> None:
    expected = make_input(np.zeros((PATCH_PX, PATCH_PX), np.float32), template)[1]
    assert expected.min() >= -1.0
    assert expected.max() <= 1.0
    assert np.allclose(expected, template_image(template) * 2 - 1)
    cell = np.kron(template, np.ones((2, 2)))
    assert expected[cell == 1].mean() > 0 > expected[cell == 0].mean()


def test_patch_channel_ignores_exposure_and_contrast(template: np.ndarray) -> None:
    patch = np.random.default_rng(2).random((PATCH_PX, PATCH_PX)).astype(np.float32)
    a = make_input(patch, template)[0]
    b = make_input(0.5 * patch + 0.2, template)[0]
    assert np.allclose(a, b, atol=1e-4)


def test_flat_patch_does_not_divide_by_zero(template: np.ndarray) -> None:
    x = make_input(np.full((PATCH_PX, PATCH_PX), 0.7, np.float32), template)
    assert np.isfinite(x).all()


def test_wrong_patch_size_is_rejected(template: np.ndarray) -> None:
    with pytest.raises(ValueError, match="patch must be"):
        make_input(np.zeros((100, 100), np.float32), template)
