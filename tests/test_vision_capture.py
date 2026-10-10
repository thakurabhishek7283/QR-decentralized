from __future__ import annotations

import contextlib

import cv2
import numpy as np
import pytest

from sigil.cdp import generate_template, raw_correlation, simulate_capture
from sigil.errors import CaptureError
from sigil.vision import (
    LABEL_H,
    LABEL_W,
    PATCH_PX,
    assess_quality,
    decode,
    extract_cdp_patch,
    load_image,
    render_label,
)

BASE45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


def make_label(seed: int) -> tuple[str, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    payload = "SG1:" + "".join(BASE45[i] for i in rng.integers(0, len(BASE45), 180))
    template = generate_template(rng.bytes(32), rng.bytes(12))
    return payload, template, render_label(payload, template)


@pytest.fixture(scope="module")
def captures() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(payload, template, photo) for a handful of normal-difficulty genuine captures."""
    out = []
    for seed in range(10):
        payload, template, label = make_label(seed)
        rng = np.random.default_rng(1000 + seed)
        out.append(
            (payload, template, simulate_capture(label, rng, domain="train", difficulty="normal"))
        )
    return out


def test_decode_succeeds_on_normal_captures(captures: list) -> None:
    ok = 0
    for payload, _, photo in captures:
        with contextlib.suppress(CaptureError):
            ok += decode(photo).payload == payload
    assert ok >= 9


def test_decode_rectifies_into_the_label_frame(captures: list) -> None:
    cap = decode(captures[0][2])
    assert cap.label_view.shape == (LABEL_H, LABEL_W)
    assert cap.label_view.dtype == np.float32
    assert cap.homography.shape == (3, 3)
    assert cap.qr_corners.shape == (4, 2)
    # the QR corners must map onto the symbol corners of the label frame
    mapped = cv2.perspectiveTransform(cap.qr_corners[None].astype(np.float32), cap.homography)[0]
    assert np.abs(mapped - [[24, 24], [342, 24], [342, 342], [24, 342]]).max() < 6


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_decode_handles_rotated_photos(quarter_turns: int) -> None:
    """Corner order must follow the symbol, not the image, or the CDP crop is misplaced."""
    _, template, label = make_label(5)
    photo = simulate_capture(label, np.random.default_rng(5), domain="train", difficulty="easy")
    rotated = np.ascontiguousarray(np.rot90(photo, quarter_turns))
    patch = extract_cdp_patch(decode(rotated), template)
    assert raw_correlation(patch.patch, template) > 0.8


def test_extract_cdp_patch_recovers_a_shifted_capture(captures: list) -> None:
    _, template, photo = captures[1]
    base = extract_cdp_patch(decode(photo), template)
    shift = np.float32([[1, 0, 9], [0, 1, -7]])
    moved = cv2.warpAffine(photo, shift, photo.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
    shifted = extract_cdp_patch(decode(moved), template)
    assert shifted.patch.shape == (PATCH_PX, PATCH_PX)
    assert shifted.patch.dtype == np.float32
    assert shifted.alignment_score > 0.8
    assert abs(shifted.alignment_score - base.alignment_score) < 0.05


def test_alignment_is_poor_against_the_wrong_template(captures: list) -> None:
    _, _, photo = captures[2]
    wrong = generate_template(bytes(32), bytes(12))
    with pytest.raises(CaptureError) as err:
        extract_cdp_patch(decode(photo), wrong)
    assert err.value.reason == "align_failed"


def test_decode_reports_no_qr_for_a_blank_photo() -> None:
    with pytest.raises(CaptureError) as err:
        decode(np.full((480, 640), 128, dtype=np.uint8))
    assert err.value.reason == "no_qr"


def test_load_image_rejects_garbage_and_decodes_png(captures: list) -> None:
    with pytest.raises(CaptureError) as err:
        load_image(b"not an image")
    assert err.value.reason == "bad_image"
    ok, png = cv2.imencode(".png", captures[0][2])
    assert ok
    assert np.array_equal(load_image(png.tobytes()), captures[0][2])


def test_quality_flags_blur_and_darkness(captures: list) -> None:
    sharp = decode(captures[3][2]).label_view
    assert assess_quality(sharp).ok
    blurred = assess_quality(cv2.GaussianBlur(sharp, (0, 0), 6.0))
    assert not blurred.ok
    assert "blurry" in blurred.reasons
    dark = assess_quality(sharp * 0.1)
    assert "too_dark" in dark.reasons
