"""Print/capture channel simulator and counterfeiting attacks.

A copy-detection pattern works because every print-and-scan cycle is lossy: ink
spreads (dot gain), edges get noisy, optics blur, and a sensor quantises. A
genuine label passes through *one* printer and *one* camera; a counterfeit has to
pass through a scanner, a second printer and then the verifier's camera, so it
carries strictly more degradation. This module models those stages with a small
physically motivated pipeline. It is a **simulation**, not a calibration against
real hardware, and the model card says so.

Stages
------
``print_label``
    Gaussian ink spread (dot gain), a noisy decision threshold, ink density,
    paper grain, and worn print heads that drop or add CDP cells.
``capture``
    Resampling by a scale factor, a random homography onto a cluttered 1280x960
    canvas, defocus and motion blur, uneven lighting, sensor noise, gamma,
    brightness and JPEG compression.
``counterfeit_from``
    Attacks that only ever see the *printed genuine label* (never the secret
    template): photocopy, high-quality reprint and screen re-photography.

Domains
-------
Sampling is split into a ``"train"`` and a ``"test"`` domain with deliberately
different parameter ranges. The test domain is wider (more blur, noisier sensor,
lower JPEG quality, wider gamma, rougher printers and different attacker
equipment) so held-out metrics measure generalisation to an unseen channel
rather than memorisation of the training distribution. The ranges are the
``PRINT_RANGES``, ``CAPTURE_RANGES`` and attack tables below.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import cv2
import numpy as np

from sigil.vision.geometry import CDP_CELLS, CDP_ORIGIN, CDP_PX, CELL_PX, LABEL_H, LABEL_W

Domain = Literal["train", "test"]
Quality = Literal["good", "worn"]
Difficulty = Literal["easy", "normal", "hard"]
Range = tuple[float, float]

PAPER_REFLECTANCE = 0.92
EDGE_SHARPNESS = 12.0  # slope of the ink deposition curve around the threshold
SENSOR_PX = 1.0  # sensor noise is modelled at captured-pixel resolution


@dataclass(frozen=True)
class PrintProfile:
    """One printer plus substrate."""

    dot_gain: float  # gaussian sigma (label px) of ink spread
    ink_noise: float  # std of the per-pixel decision-threshold noise
    ink_density: float  # darkness of solid ink, 0..1
    paper_grain: float  # low-frequency substrate texture amplitude
    dropout: float  # fraction of CDP cells lost/added by worn heads


@dataclass(frozen=True)
class CaptureProfile:
    """One camera shot."""

    scale: float  # captured px per label px
    perspective: float  # max corner jitter as a fraction of label width
    rotation_deg: float
    defocus: float  # gaussian sigma (captured px)
    motion: float  # motion blur length (captured px)
    noise: float  # sensor noise std on a 0..1 scale
    gamma: float
    brightness: float  # multiplicative exposure
    jpeg_quality: int
    shading: float = 0.0  # amplitude of uneven illumination across the frame
    canvas: tuple[int, int] = (1280, 960)  # (width, height)


class Attack(StrEnum):
    PHOTOCOPY = "photocopy"  # office scan (~0.5x) -> laser print with high dot gain
    HQ_REPRINT = "hq_reprint"  # 2x scan + denoise + re-binarise -> good printer
    SCREEN = "screen"  # photographed, shown on a phone screen, photographed again


# --------------------------------------------------------------------------- sampling

PRINT_RANGES: Mapping[Domain, Mapping[str, Range]] = {
    "train": {
        "dot_gain": (0.85, 1.38),
        "ink_noise": (0.07, 0.15),
        "ink_density": (0.78, 0.95),
        "paper_grain": (0.0, 0.05),
        "dropout": (0.0, 0.008),
    },
    "test": {
        "dot_gain": (0.90, 1.45),
        "ink_noise": (0.09, 0.18),
        "ink_density": (0.74, 0.92),
        "paper_grain": (0.01, 0.06),
        "dropout": (0.0, 0.012),
    },
}
# Worn heads: extra spread and noise on top of a good profile, plus many more dead cells.
WORN_RANGES: Mapping[Domain, Mapping[str, Range]] = {
    "train": {"dot_gain": (0.03, 0.10), "ink_noise": (0.01, 0.03), "dropout": (0.015, 0.03)},
    "test": {"dot_gain": (0.05, 0.14), "ink_noise": (0.01, 0.04), "dropout": (0.02, 0.04)},
}

CAPTURE_RANGES: Mapping[tuple[Domain, Difficulty], Mapping[str, Range]] = {
    ("train", "easy"): {
        "scale": (1.05, 1.4),
        "perspective": (0.0, 0.02),
        "rotation_deg": (-8, 8),
        "defocus": (0.0, 0.4),
        "motion": (0.0, 0.8),
        "noise": (0.005, 0.012),
        "gamma": (0.9, 1.1),
        "brightness": (0.9, 1.1),
        "jpeg_quality": (82, 95),
        "shading": (0.0, 0.08),
    },
    ("train", "normal"): {
        "scale": (0.75, 1.18),
        "perspective": (0.01, 0.05),
        "rotation_deg": (-22, 22),
        "defocus": (0.35, 1.1),
        "motion": (0.0, 1.8),
        "noise": (0.008, 0.025),
        "gamma": (0.8, 1.25),
        "brightness": (0.78, 1.15),
        "jpeg_quality": (65, 92),
        "shading": (0.0, 0.18),
    },
    ("train", "hard"): {
        "scale": (0.62, 0.9),
        "perspective": (0.03, 0.06),
        "rotation_deg": (-35, 35),
        "defocus": (0.7, 1.4),
        "motion": (0.8, 2.5),
        "noise": (0.012, 0.025),
        "gamma": (0.8, 1.25),
        "brightness": (0.7, 1.2),
        "jpeg_quality": (60, 85),
        "shading": (0.05, 0.22),
    },
    ("test", "easy"): {
        "scale": (1.0, 1.4),
        "perspective": (0.0, 0.025),
        "rotation_deg": (-10, 10),
        "defocus": (0.0, 0.5),
        "motion": (0.0, 1.0),
        "noise": (0.006, 0.015),
        "gamma": (0.75, 1.3),
        "brightness": (0.85, 1.15),
        "jpeg_quality": (70, 92),
        "shading": (0.0, 0.12),
    },
    ("test", "normal"): {
        "scale": (0.72, 1.25),
        "perspective": (0.01, 0.06),
        "rotation_deg": (-25, 25),
        "defocus": (0.4, 1.25),
        "motion": (0.0, 2.2),
        "noise": (0.01, 0.03),
        "gamma": (0.7, 1.4),
        "brightness": (0.72, 1.2),
        "jpeg_quality": (52, 88),
        "shading": (0.0, 0.25),
    },
    ("test", "hard"): {
        "scale": (0.58, 0.88),
        "perspective": (0.04, 0.08),
        "rotation_deg": (-45, 45),
        "defocus": (0.9, 1.6),
        "motion": (1.0, 3.0),
        "noise": (0.015, 0.03),
        "gamma": (0.7, 1.4),
        "brightness": (0.7, 1.2),
        "jpeg_quality": (55, 80),
        "shading": (0.08, 0.3),
    },
}


def _uniform(rng: np.random.Generator, bounds: Range) -> float:
    return float(rng.uniform(*bounds))


def sample_print_profile(
    rng: np.random.Generator, domain: Domain, quality: Quality = "good"
) -> PrintProfile:
    """Draw a printer. ``"worn"`` heads spread ink more and drop many more cells."""
    r = {name: _uniform(rng, bounds) for name, bounds in PRINT_RANGES[domain].items()}
    if quality == "worn":
        worn = WORN_RANGES[domain]
        r["dot_gain"] += _uniform(rng, worn["dot_gain"])
        r["ink_noise"] += _uniform(rng, worn["ink_noise"])
        r["dropout"] = _uniform(rng, worn["dropout"])
    return PrintProfile(**r)


def sample_capture_profile(
    rng: np.random.Generator, domain: Domain, difficulty: Difficulty = "normal"
) -> CaptureProfile:
    """Draw a camera shot. ``difficulty`` moves scale, blur, noise and tilt together."""
    r = {
        name: _uniform(rng, bounds) for name, bounds in CAPTURE_RANGES[(domain, difficulty)].items()
    }
    jpeg_quality = round(r.pop("jpeg_quality"))
    return CaptureProfile(jpeg_quality=jpeg_quality, **r)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- helpers


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Zero-mean, unit-std gaussian noise low-passed to a correlation length ``sigma``."""
    noise = cv2.GaussianBlur(rng.standard_normal(shape, dtype=np.float32), (0, 0), sigma)
    return noise / (noise.std() + 1e-9)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 1e-3 else img


def _jpeg_roundtrip(gray: np.ndarray, quality: int) -> np.ndarray:
    """Compress a uint8 grayscale image to JPEG and decode it again."""
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else None
    if decoded is None:  # pragma: no cover - imencode only fails on invalid input
        raise ValueError("JPEG encoding failed")
    return np.asarray(decoded, dtype=np.uint8)


def _flip_cells(ink: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Worn print heads: invert a random ``fraction`` of the CDP cells."""
    if fraction <= 0:
        return ink
    flips = rng.random((CDP_CELLS, CDP_CELLS)) < fraction
    mask = np.kron(flips, np.ones((CELL_PX, CELL_PX), dtype=bool))
    x, y = CDP_ORIGIN
    out = ink.copy()
    region = out[y : y + CDP_PX, x : x + CDP_PX]
    region[mask] = 1.0 - region[mask]
    return out


# --------------------------------------------------------------------------- printing


def print_label(label: np.ndarray, p: PrintProfile, rng: np.random.Generator) -> np.ndarray:
    """Print a label raster; returns float32 reflectance in [0, 1] (paper ~0.9, ink dark).

    Ink spreads into a gaussian halo of width ``dot_gain``; the substrate only
    takes ink where that halo exceeds a noisy threshold. Small features (a lone
    3 px CDP cell) therefore shrink, fuse or vanish stochastically, which is the
    information a copier cannot recover.
    """
    ink = _flip_cells(1.0 - label.astype(np.float32), p.dropout, rng)
    coverage = _blur(ink, p.dot_gain)
    threshold = 0.5 + p.ink_noise * _smooth_noise(rng, ink.shape, 0.7)
    deposited = 1.0 / (1.0 + np.exp(-EDGE_SHARPNESS * (coverage - threshold)))
    reflectance = PAPER_REFLECTANCE * (1.0 - p.ink_density * deposited)
    grain = 1.0 + p.paper_grain * _smooth_noise(rng, ink.shape, 5.0)
    return np.clip(reflectance * grain, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- capture


def _background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """A cluttered scene: tinted table, low-frequency shading, random shapes and grain."""
    small = _smooth_noise(rng, (h // 16, w // 16), 3.0)
    bg = rng.uniform(0.12, 0.8) + 0.12 * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(int(rng.integers(4, 10))):
        tone = float(rng.uniform(0.05, 0.95))
        kind = int(rng.integers(0, 3))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        if kind == 0:
            rect = (
                (cx, cy),
                (float(rng.uniform(40, 400)), float(rng.uniform(40, 300))),
                float(rng.uniform(0, 180)),
            )
            cv2.fillConvexPoly(bg, cv2.boxPoints(rect).astype(np.int32), tone)
        elif kind == 1:
            cv2.line(
                bg,
                (cx, cy),
                (int(rng.integers(0, w)), int(rng.integers(0, h))),
                tone,
                int(rng.integers(2, 12)),
            )
        else:
            axes = (int(rng.integers(20, 200)), int(rng.integers(20, 160)))
            cv2.ellipse(bg, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, tone, -1)
    grain = _smooth_noise(rng, (h // 2, w // 2), 0.8)
    bg += 0.03 * cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg.astype(np.float32)


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    size = int(2 * math.ceil(length) + 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    dx, dy = (
        math.cos(math.radians(angle_deg)) * length / 2,
        math.sin(math.radians(angle_deg)) * length / 2,
    )
    cv2.line(kernel, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, 1)
    return kernel / kernel.sum()


def _illumination(rng: np.random.Generator, w: int, h: int, amplitude: float) -> np.ndarray:
    """Multiplicative lighting: a random linear ramp plus a soft vignette."""
    xs = (np.arange(w, dtype=np.float32) / w - 0.5)[None, :]
    ys = (np.arange(h, dtype=np.float32) / h - 0.5)[:, None]
    angle = rng.uniform(0, 2 * math.pi)
    ramp = xs * math.cos(angle) + ys * math.sin(angle)
    vignette = xs**2 + ys**2
    return (1.0 + amplitude * (2.0 * ramp - 1.2 * vignette)).astype(np.float32)


def _place(
    rng: np.random.Generator, size: tuple[int, int], c: CaptureProfile
) -> tuple[np.ndarray, np.ndarray]:
    """Random pose: source corners (in the scaled sheet) and destination quad on the canvas."""
    sw, sh = size
    cw, ch = c.canvas
    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
    theta = math.radians(c.rotation_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    quad = (src - [sw / 2, sh / 2]) @ rot.T
    quad += rng.uniform(-1, 1, quad.shape) * c.perspective * sw
    extent = quad.max(axis=0) - quad.min(axis=0)
    fit = min(1.0, (cw - 24) / extent[0], (ch - 24) / extent[1])
    quad = quad * fit
    lo, hi = quad.min(axis=0), quad.max(axis=0)
    centre = np.array(
        [rng.uniform(12 - lo[0], cw - 12 - hi[0]), rng.uniform(12 - lo[1], ch - 12 - hi[1])]
    )
    return src, (quad + centre).astype(np.float32)


def capture(printed: np.ndarray, c: CaptureProfile, rng: np.random.Generator) -> np.ndarray:
    """Photograph a printed label; returns a uint8 grayscale image of the canvas size."""
    cw, ch = c.canvas
    margin = int(rng.integers(14, 40))  # sticker paper around the printed area
    sheet = cv2.copyMakeBorder(
        printed, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=PAPER_REFLECTANCE
    )
    interp = cv2.INTER_AREA if c.scale < 1 else cv2.INTER_CUBIC
    sheet = cv2.resize(sheet, None, fx=c.scale, fy=c.scale, interpolation=interp)
    src, dst = _place(rng, (sheet.shape[1], sheet.shape[0]), c)
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet, homography, (cw, ch), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(np.ones_like(sheet), homography, (cw, ch), flags=cv2.INTER_LINEAR)
    scene = _background(rng, cw, ch) * (1.0 - alpha) + warped

    scene = _blur(scene, c.defocus)
    if c.motion > 0.8:
        scene = cv2.filter2D(scene, -1, _motion_kernel(c.motion, float(rng.uniform(0, 180))))
    scene *= _illumination(rng, cw, ch, c.shading)
    scene += c.noise * rng.standard_normal(scene.shape, dtype=np.float32) * (0.5 + scene)
    scene = np.clip(scene, 0.0, 1.0) ** c.gamma * c.brightness
    frame = np.clip(scene * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return _jpeg_roundtrip(frame, int(c.jpeg_quality))


# --------------------------------------------------------------------------- attacks

# Attacker equipment, per domain. The test domain uses different scanners, copier
# tones and screens than the ones the model trains against.
PHOTOCOPY_RANGES: Mapping[Domain, Mapping[str, Range]] = {
    "train": {
        "scan_scale": (0.45, 0.55), "scan_blur": (0.15, 0.4), "noise": (0.01, 0.03),
        "steepness": (8, 16), "tone_jitter": (0.0, 0.08),
        "dot_gain": (0.85, 1.2), "ink_noise": (0.07, 0.12),
    },
    "test": {
        "scan_scale": (0.45, 0.60), "scan_blur": (0.15, 0.5), "noise": (0.015, 0.035),
        "steepness": (6, 20), "tone_jitter": (0.0, 0.12),
        "dot_gain": (0.9, 1.25), "ink_noise": (0.07, 0.14),
    },
}  # fmt: skip
HQ_REPRINT_RANGES: Mapping[Domain, Mapping[str, Range]] = {
    "train": {
        "scan_blur": (0.7, 1.2), "noise": (0.012, 0.035), "tone_jitter": (0.0, 0.05),
        "dot_gain": (0.75, 1.15), "ink_noise": (0.06, 0.12),
    },
    "test": {
        "scan_blur": (0.8, 1.3), "noise": (0.015, 0.04), "tone_jitter": (0.0, 0.08),
        "dot_gain": (0.8, 1.25), "ink_noise": (0.07, 0.14),
    },
}  # fmt: skip
SCREEN_RANGES: Mapping[Domain, Mapping[str, Range]] = {
    "train": {
        "photo_scale": (0.7, 1.1), "photo_blur": (0.5, 1.0), "photo_noise": (0.008, 0.02),
        "jpeg": (75, 92), "pitch": (1.3, 2.2), "gap": (0.08, 0.20), "black_lift": (0.04, 0.12),
        "gamma": (0.85, 1.2), "moire": (0.03, 0.08), "glare": (0.0, 0.1),
    },
    "test": {
        "photo_scale": (0.65, 1.1), "photo_blur": (0.55, 1.1), "photo_noise": (0.01, 0.03),
        "jpeg": (65, 90), "pitch": (1.2, 2.4), "gap": (0.06, 0.26), "black_lift": (0.03, 0.15),
        "gamma": (0.8, 1.3), "moire": (0.02, 0.10), "glare": (0.0, 0.15),
    },
}  # fmt: skip


def _draw(rng: np.random.Generator, table: Mapping[str, Range]) -> dict[str, float]:
    return {name: _uniform(rng, bounds) for name, bounds in table.items()}


def _shift(img: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    """Sub-pixel mis-registration between the attacker's scan and the label grid."""
    dx, dy = rng.uniform(-amount, amount, 2)
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)  # fmt: skip


def _scan_tone(
    img: np.ndarray, steepness: float, jitter: float, rng: np.random.Generator
) -> np.ndarray:
    """Copier/scanner thresholding around the mid-grey of the scan, with exposure error."""
    lo, hi = np.percentile(img, [2, 98])
    mid = 0.5 * (lo + hi) + float(rng.uniform(-jitter, jitter)) * (hi - lo)
    return np.asarray(1.0 / (1.0 + np.exp(-steepness * (img - mid) / max(hi - lo, 1e-3))))


def _photocopy(printed: np.ndarray, rng: np.random.Generator, domain: Domain) -> np.ndarray:
    """300-dpi-class office copier: ~0.5x scan, tone clipping, laser reprint with heavy dot gain."""
    a = _draw(rng, PHOTOCOPY_RANGES[domain])
    scan = cv2.resize(
        printed, None, fx=a["scan_scale"], fy=a["scan_scale"], interpolation=cv2.INTER_AREA
    )
    noise = np.asarray(rng.standard_normal(scan.shape), dtype=np.float32)
    scan = _blur(scan, a["scan_blur"]) + a["noise"] * noise
    toned = _scan_tone(scan, a["steepness"], a["tone_jitter"], rng)
    bitmap = cv2.resize(
        toned.astype(np.float32), (LABEL_W, LABEL_H), interpolation=cv2.INTER_LINEAR
    )
    laser = PrintProfile(
        dot_gain=a["dot_gain"], ink_noise=a["ink_noise"], ink_density=_uniform(rng, (0.8, 0.95)),
        paper_grain=_uniform(rng, (0.02, 0.05)), dropout=0.0,
    )  # fmt: skip
    return print_label(bitmap, laser, rng)


def _hq_reprint(printed: np.ndarray, rng: np.random.Generator, domain: Domain) -> np.ndarray:
    """1200-dpi scan (2x), denoise, re-binarise at cell resolution, reprint on a good printer."""
    a = _draw(rng, HQ_REPRINT_RANGES[domain])
    scan = cv2.resize(printed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    scan = _blur(_shift(scan, rng, 1.0), a["scan_blur"])
    scan = scan + a["noise"] * np.asarray(rng.standard_normal(scan.shape), dtype=np.float32)
    scan = cv2.medianBlur(scan.astype(np.float32), 3)
    bitmap = _scan_tone(scan, 40.0, a["tone_jitter"], rng)
    bitmap = cv2.resize(bitmap.astype(np.float32), (LABEL_W, LABEL_H), interpolation=cv2.INTER_AREA)
    printer = PrintProfile(
        dot_gain=a["dot_gain"], ink_noise=a["ink_noise"], ink_density=_uniform(rng, (0.82, 0.95)),
        paper_grain=_uniform(rng, (0.0, 0.03)), dropout=0.0,
    )  # fmt: skip
    return print_label(bitmap, printer, rng)


def _screen(printed: np.ndarray, rng: np.random.Generator, domain: Domain) -> np.ndarray:
    """Photograph the label, show it on a pixel-grid display, which is photographed again.

    The second camera sees a finite pixel grid (aperture gaps), a lifted black
    level, a different tone curve, glare and moire between the grid and its own
    sampling. The bitmap is only as sharp as the display's pixel pitch.
    """
    a = _draw(rng, SCREEN_RANGES[domain])
    photo = cv2.resize(
        printed, None, fx=a["photo_scale"], fy=a["photo_scale"], interpolation=cv2.INTER_AREA
    )
    photo = _blur(_shift(photo, rng, 0.8), a["photo_blur"])
    photo = photo + a["photo_noise"] * np.asarray(
        rng.standard_normal(photo.shape), dtype=np.float32
    )
    compressed = _jpeg_roundtrip(np.clip(photo * 255.0, 0, 255).astype(np.uint8), int(a["jpeg"]))
    photo = compressed.astype(np.float32) / 255.0

    pitch = a["pitch"]
    pixels = cv2.resize(
        photo, (round(LABEL_W / pitch), round(LABEL_H / pitch)), interpolation=cv2.INTER_AREA
    )
    shown = cv2.resize(pixels, (LABEL_W, LABEL_H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.mgrid[0:LABEL_H, 0:LABEL_W].astype(np.float32)

    def edge(t: np.ndarray) -> np.ndarray:
        return np.asarray(((1.0 - np.cos(2 * np.pi * t / pitch)) / 2.0) ** 4)

    aperture = (1.0 - a["gap"] * edge(xs)) * (1.0 - a["gap"] * edge(ys))
    phi = rng.uniform(0, np.pi)
    beat = np.sin(
        2
        * np.pi
        * (rng.uniform(1.5, 5.0) * (xs * np.cos(phi) + ys * np.sin(phi)) / LABEL_W + rng.uniform())
    )
    glare = 1.0 + a["glare"] * (xs / LABEL_W - 0.5) * rng.choice([-1.0, 1.0])
    lit = np.clip(shown, 0.0, 1.0) ** a["gamma"] * aperture * (1.0 + a["moire"] * beat) * glare
    out = np.clip(a["black_lift"] + (1.0 - a["black_lift"]) * lit * PAPER_REFLECTANCE, 0.0, 1.0)
    return np.asarray(out, dtype=np.float32)


_ATTACKS = {
    Attack.PHOTOCOPY: _photocopy,
    Attack.HQ_REPRINT: _hq_reprint,
    Attack.SCREEN: _screen,
}


def counterfeit_from(
    genuine_printed: np.ndarray,
    attack: Attack,
    rng: np.random.Generator,
    domain: Domain = "train",
) -> np.ndarray:
    """Forge a *printed* label from a printed genuine one (float [0, 1], label frame).

    The attacker only ever sees the printed object, never the template, so every
    attack is an image-to-image degradation of ``genuine_printed``. ``domain``
    selects the attacker's equipment ranges.
    """
    return _ATTACKS[attack](genuine_printed, rng, domain)


def simulate_capture(
    label: np.ndarray,
    rng: np.random.Generator,
    *,
    domain: Domain = "test",
    difficulty: Difficulty = "normal",
    quality: Quality = "good",
    attack: Attack | None = None,
) -> np.ndarray:
    """Run a label raster through the whole channel and return the phone photo.

    Print on a printer drawn for ``quality`` (an attacker copies that print when
    ``attack`` is given), then photograph it under ``difficulty`` conditions. This
    is the one entry point that training, evals, QC sampling and tests share.
    """
    printed = print_label(label, sample_print_profile(rng, domain, quality), rng)
    if attack is not None:
        printed = counterfeit_from(printed, attack, rng, domain)
    return capture(printed, sample_capture_profile(rng, domain, difficulty), rng)
