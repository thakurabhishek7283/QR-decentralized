from __future__ import annotations

import numpy as np
import pytest

from sigil.cdp import (
    Attack,
    CaptureProfile,
    capture,
    counterfeit_from,
    generate_template,
    print_label,
    raw_correlation,
    sample_capture_profile,
    sample_print_profile,
    simulate_capture,
)
from sigil.errors import CaptureError
from sigil.vision import LABEL_H, LABEL_W, decode, extract_cdp_patch, render_label

BASE45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


def make_label(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    payload = "SG1:" + "".join(BASE45[i] for i in rng.integers(0, len(BASE45), 180))
    template = generate_template(rng.bytes(32), rng.bytes(12))
    return template, render_label(payload, template)


def mean_correlation(
    seeds: range, *, attack: Attack | None = None, quality: str = "good", domain: str = "train"
) -> float:
    values = []
    for seed in seeds:
        template, label = make_label(seed)
        photo = simulate_capture(
            label, np.random.default_rng(seed + 500), domain=domain, difficulty="easy",  # type: ignore[arg-type]
            quality=quality, attack=attack,  # type: ignore[arg-type]
        )  # fmt: skip
        try:
            values.append(
                raw_correlation(extract_cdp_patch(decode(photo), template).patch, template)
            )
        except CaptureError:
            continue
    assert len(values) >= len(seeds) - 2, "too many captures failed to decode"
    return float(np.mean(values))


def test_print_label_is_deterministic_and_bounded() -> None:
    _, label = make_label(0)
    profile = sample_print_profile(np.random.default_rng(1), "train")
    a = print_label(label, profile, np.random.default_rng(2))
    b = print_label(label, profile, np.random.default_rng(2))
    assert np.array_equal(a, b)
    assert a.shape == (LABEL_H, LABEL_W)
    assert a.dtype == np.float32
    assert a.min() >= 0.0
    assert a.max() <= 1.0
    assert a[label == 0].mean() < 0.4 < 0.8 < a[label == 1].mean()  # ink dark, paper bright


def test_capture_output_format() -> None:
    _, label = make_label(1)
    rng = np.random.default_rng(3)
    photo = capture(print_label(label, sample_print_profile(rng, "test"), rng),
                    sample_capture_profile(rng, "test", "normal"), rng)  # fmt: skip
    assert photo.shape == (960, 1280)
    assert photo.dtype == np.uint8
    assert photo.std() > 20  # not a flat frame


def test_capture_profile_default_canvas() -> None:
    assert CaptureProfile(1, 0, 0, 0, 0, 0.01, 1, 1, 90).canvas == (1280, 960)


def test_test_domain_is_wider_than_train() -> None:
    rng = np.random.default_rng(0)
    train = [sample_capture_profile(rng, "train", "normal") for _ in range(300)]
    test = [sample_capture_profile(rng, "test", "normal") for _ in range(300)]
    assert max(c.defocus for c in test) > max(c.defocus for c in train)
    assert min(c.jpeg_quality for c in test) < min(c.jpeg_quality for c in train)
    assert min(c.gamma for c in test) < min(c.gamma for c in train)
    assert max(c.gamma for c in test) > max(c.gamma for c in train)


def test_worn_printers_spread_ink_and_drop_cells() -> None:
    rng = np.random.default_rng(0)
    good = [sample_print_profile(rng, "train", "good") for _ in range(200)]
    worn = [sample_print_profile(rng, "train", "worn") for _ in range(200)]
    assert np.mean([p.dot_gain for p in worn]) > np.mean([p.dot_gain for p in good])
    assert np.mean([p.dropout for p in worn]) > 3 * np.mean([p.dropout for p in good])


@pytest.mark.parametrize("attack", list(Attack))
def test_attacks_return_a_printed_label_without_the_template(attack: Attack) -> None:
    _, label = make_label(2)
    rng = np.random.default_rng(4)
    printed = print_label(label, sample_print_profile(rng, "train"), rng)
    forged = counterfeit_from(printed, attack, rng)
    assert forged.shape == printed.shape
    assert forged.dtype == np.float32
    assert forged.min() >= 0.0
    assert forged.max() <= 1.0
    assert not np.array_equal(forged, printed)


def test_correlation_separates_genuine_from_photocopy() -> None:
    seeds = range(8)
    assert mean_correlation(seeds) > mean_correlation(seeds, attack=Attack.PHOTOCOPY) + 0.05


@pytest.mark.slow
def test_every_attack_scores_below_genuine_on_average() -> None:
    seeds = range(14)
    genuine = mean_correlation(seeds, domain="test")
    for attack in Attack:
        assert mean_correlation(seeds, attack=attack, domain="test") < genuine


@pytest.mark.slow
def test_worn_printer_genuine_labels_correlate_less() -> None:
    seeds = range(14)
    assert mean_correlation(seeds, quality="worn") < mean_correlation(seeds, quality="good")
