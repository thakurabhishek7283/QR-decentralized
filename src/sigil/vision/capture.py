"""From a phone photo to a rectified label and an aligned CDP patch."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from sigil.cdp.features import template_image
from sigil.errors import CaptureError
from sigil.vision.geometry import (
    CDP_CELLS,
    CDP_ORIGIN,
    CDP_PX,
    LABEL_H,
    LABEL_W,
    MODULE_PX,
    PATCH_PX,
    QR_MODULES,
    QR_ORIGIN,
    SYMBOL_PX,
    qr_corners_label,
)
from sigil.vision.label import render_qr_matrix
from sigil.vision.quality import QualityReport, assess_quality

RETRY_MODULE_PX = 8  # sampling density of the rectified retry crop
REFINE_MARGIN_PX = 12  # quiet-zone context included when refining the QR homography
CROP_MARGIN_PX = 6  # label px kept around the CDP square so ECC can move it
MIN_ALIGNMENT = 0.2  # final ECC correlation below this means the alignment failed
MIN_SYMBOL_MATCH = 0.5  # rectified QR vs the QR re-rendered from its own payload
_PATCH_CROP = (CDP_PX + 2 * CROP_MARGIN_PX) * PATCH_PX // CDP_PX  # crop side at 2 px per cell


@dataclass(frozen=True)
class DecodedCapture:
    payload: str
    qr_corners: np.ndarray  # (4, 2) image px: TL, TR, BR, BL of the symbol
    homography: np.ndarray  # 3x3, image -> label frame
    label_view: np.ndarray  # float32 (LABEL_H, LABEL_W) rectified label, [0, 1]
    quality: QualityReport


@dataclass(frozen=True)
class CdpPatch:
    patch: np.ndarray  # float32 (PATCH_PX, PATCH_PX), [0, 1], ink dark
    alignment_score: float  # final ECC correlation with the expected template


def load_image(data: bytes) -> np.ndarray:
    """Decode an encoded image (JPEG, PNG, ...) to uint8 grayscale."""
    gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise CaptureError("bad_image", "could not decode image bytes")
    return gray


# --------------------------------------------------------------------------- detection

Detector = Callable[[np.ndarray], tuple[str, np.ndarray | None]]


def _legacy(img: np.ndarray) -> tuple[str, np.ndarray | None]:
    text, points, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return text, points


def _aruco(img: np.ndarray) -> tuple[str, np.ndarray | None]:
    text, points, _ = cv2.QRCodeDetectorAruco().detectAndDecode(img)
    return text, points


def _attempts(gray: np.ndarray) -> Iterator[tuple[str, np.ndarray | None, float]]:
    """Yield (payload, corners, scale) from progressively heavier detection strategies."""

    def equalised() -> np.ndarray:
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    def upscaled() -> np.ndarray:
        return cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    plan: tuple[tuple[Callable[[], np.ndarray], float, Detector], ...] = (
        (lambda: gray, 1.0, _legacy),
        (equalised, 1.0, _legacy),
        (upscaled, 1.5, _legacy),
        (lambda: gray, 1.0, _aruco),
        (equalised, 1.0, _aruco),
    )
    for prepare, scale, detector in plan:
        text, points = detector(prepare())
        yield text, points, scale


def _decode_rectified(gray: np.ndarray, corners: np.ndarray) -> str:
    """Re-decode a symbol the detector located but could not read.

    The photo is perspective-corrected to an axis-aligned, upsampled crop (8 px per
    module with a quiet zone) and tried as-is, then locally thresholded, which
    rescues blurry or unevenly lit symbols that defeat the detector's own sampler.
    """
    quiet = 4
    size = (QR_MODULES + 2 * quiet) * RETRY_MODULE_PX
    lo, hi = quiet * RETRY_MODULE_PX, (quiet + QR_MODULES) * RETRY_MODULE_PX
    target = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(_clockwise(corners), target)
    img = gray.astype(np.float32)
    side = float(np.linalg.norm(corners[1] - corners[0])) / (QR_MODULES * RETRY_MODULE_PX)
    if side > 1.2:
        img = np.asarray(cv2.GaussianBlur(img, (0, 0), 0.45 * side), dtype=np.float32)
    crop = cv2.warpPerspective(
        img, homography, (size, size), flags=cv2.INTER_CUBIC, borderValue=255.0
    )
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    adaptive = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 8
    )
    for variant in (crop, adaptive):
        text, _, _ = cv2.QRCodeDetector().detectAndDecode(variant)
        if text:
            return text
    return ""


def _find_qr(gray: np.ndarray) -> tuple[str, np.ndarray]:
    located: np.ndarray | None = None
    for text, points, scale in _attempts(gray):
        if points is None or points.size != 8:
            continue
        corners = points.reshape(4, 2).astype(np.float32) / scale
        if text:
            return text, corners
        located = corners if located is None else located
    if located is not None:
        text = _decode_rectified(gray, located)
        if text:
            return text, located
    raise CaptureError("no_qr", "no decodable QR code found")


# --------------------------------------------------------------------------- geometry


def _clockwise(corners: np.ndarray) -> np.ndarray:
    """Reverse the corner order if the polygon is wound counter-clockwise (image coords)."""
    x, y = corners[:, 0], corners[:, 1]
    area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return corners if area > 0 else corners[::-1].copy()


def _prefilter(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Photo as float32 in [0, 1], low-passed if the label is downsampled to the label frame.

    Without the low-pass a perspective warp that shrinks the image aliases fine
    detail (the CDP) into the rectified view.
    """
    img = gray.astype(np.float32) / 255.0
    px_per_label_px = float(np.linalg.norm(corners[1] - corners[0])) / SYMBOL_PX
    if px_per_label_px > 1.25:
        img = np.asarray(cv2.GaussianBlur(img, (0, 0), 0.45 * px_per_label_px), dtype=np.float32)
    return img


def _warp_to_label(img: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(img, homography, (LABEL_W, LABEL_H), flags=cv2.INTER_LINEAR)


def _symbol_match(sampled: np.ndarray, modules: np.ndarray) -> float:
    """Pearson correlation of a rectified QR (any resolution) with the QR its payload implies."""
    grid = cv2.resize(sampled, (QR_MODULES, QR_MODULES), interpolation=cv2.INTER_AREA)
    expected = np.where(modules, -1.0, 1.0)
    return float(np.corrcoef(grid.ravel(), expected.ravel())[0, 1])


def _orient(img: np.ndarray, corners: np.ndarray, modules: np.ndarray) -> np.ndarray:
    """Pick the cyclic corner order whose rectified symbol best matches ``modules``."""
    side = 2 * QR_MODULES
    square = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
    clockwise = _clockwise(corners)
    best, best_score = clockwise, -2.0
    for shift in range(4):
        candidate = np.roll(clockwise, -shift, axis=0)
        sym = cv2.warpPerspective(img, cv2.getPerspectiveTransform(candidate, square), (side, side))
        score = _symbol_match(sym, modules)
        if score > best_score:
            best, best_score = candidate, score
    return best


def _refine_homography(img: np.ndarray, homography: np.ndarray, modules: np.ndarray) -> np.ndarray:
    """Polish the QR-corner homography against the QR re-rendered from its payload.

    Detector corners can be several pixels off on a degraded symbol, and the CDP
    sits a symbol-width away from them, so errors would be amplified there. The
    exact module matrix is known, so a coarse-to-fine ECC homography fit of the
    rectified symbol onto it removes that error.
    """
    qx, qy = QR_ORIGIN
    m = REFINE_MARGIN_PX
    box = (slice(qy - m, qy + SYMBOL_PX + m), slice(qx - m, qx + SYMBOL_PX + m))
    reference = np.ones((LABEL_H, LABEL_W), dtype=np.float32)
    reference[qy : qy + SYMBOL_PX, qx : qx + SYMBOL_PX] = np.where(
        np.kron(modules, np.ones((MODULE_PX, MODULE_PX), dtype=bool)), 0.0, 1.0
    )
    ref_crop, view_crop = reference[box], _warp_to_label(img, homography)[box]

    offset = np.array([[1, 0, qx - m], [0, 1, qy - m], [0, 0, 1]], dtype=np.float64)
    warp = np.eye(3, dtype=np.float64)  # reference crop px -> view crop px
    for scale, iterations in ((0.25, 40), (0.5, 30), (1.0, 20)):
        s = np.diag([scale, scale, 1.0])
        ref_s = cv2.resize(ref_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        view_s = cv2.resize(view_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-4)
        try:
            _, level = cv2.findTransformECC(
                ref_s, view_s, (s @ warp @ np.linalg.inv(s)).astype(np.float32),
                cv2.MOTION_HOMOGRAPHY, criteria, None, 5,
            )  # type: ignore[call-overload]  # fmt: skip
        except cv2.error:
            return homography  # keep the detector's corners
        warp = np.linalg.inv(s) @ level @ s
    in_label_frame = offset @ warp @ np.linalg.inv(offset)
    return np.asarray(np.linalg.inv(in_label_frame) @ homography)


def decode(gray: np.ndarray) -> DecodedCapture:
    """Find and decode the QR in a grayscale photo and rectify the label.

    Detection tries the OpenCV QR detector, then a CLAHE-equalised copy, a 1.5x
    upscale and the ArUco-based detector, and finally a rectified re-read of any
    symbol that was located but not decoded. The detector's corner order is not
    trusted: the corners are wound clockwise, then the cyclic rotation is chosen
    for which the rectified symbol best matches the QR re-rendered from the
    decoded payload (a wrong rotation would silently put the CDP in the wrong
    place), and the homography is refined against that same rendering.

    Raises ``CaptureError`` with reason ``no_qr`` or ``low_quality`` (decoded, but
    the symbol does not line up with its own payload, so the geometry is unusable).
    Whether the photo is sharp and exposed well enough is reported in
    ``DecodedCapture.quality`` rather than raised.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise CaptureError("bad_image", "expected a uint8 grayscale image")
    payload, corners = _find_qr(gray)
    try:
        modules = render_qr_matrix(payload)
    except ValueError as exc:
        raise CaptureError("no_qr", "decoded a QR that is not a Sigil label") from exc

    img = _prefilter(gray, corners)
    oriented = _orient(img, corners, modules)
    homography = cv2.getPerspectiveTransform(oriented, qr_corners_label())
    view = _warp_to_label(img, homography)
    qx, qy = QR_ORIGIN
    score = _symbol_match(view[qy : qy + SYMBOL_PX, qx : qx + SYMBOL_PX], modules)

    refined = _refine_homography(img, homography, modules)
    refined_view = _warp_to_label(img, refined)
    refined_score = _symbol_match(refined_view[qy : qy + SYMBOL_PX, qx : qx + SYMBOL_PX], modules)
    if refined_score > score:
        score, homography, view = refined_score, refined, refined_view
    if score < MIN_SYMBOL_MATCH:
        raise CaptureError("low_quality", f"rectified symbol does not match payload ({score:.2f})")
    return DecodedCapture(
        payload=payload,
        qr_corners=oriented,
        homography=homography,
        label_view=view,
        quality=assess_quality(view),
    )


# --------------------------------------------------------------------------- CDP patch


def extract_cdp_patch(cap: DecodedCapture, template: np.ndarray) -> CdpPatch:
    """Crop the CDP square from the rectified label and refine its alignment.

    QR corner estimates are only good to about a pixel, and the CDP sits one
    symbol-width away from them, so the crop is aligned to the *expected* template
    image: a coarse integer search, then ``findTransformECC`` with an affine model.
    This is the single function used by training, serving and evals.
    """
    x0, y0 = CDP_ORIGIN
    m = CROP_MARGIN_PX
    crop = cv2.resize(
        cap.label_view[y0 - m : y0 + CDP_PX + m, x0 - m : x0 + CDP_PX + m],
        (_PATCH_CROP, _PATCH_CROP),
        interpolation=cv2.INTER_AREA,
    )
    ink = 1.0 - crop  # ink high, to match the template image's polarity
    ideal = template_image(template, px_per_cell=PATCH_PX // CDP_CELLS, blur=0.8)

    # Coarse: the best integer shift of the ideal image inside the margin.
    response = cv2.matchTemplate(ink, ideal, cv2.TM_CCOEFF_NORMED)
    _, _, _, (dx, dy) = cv2.minMaxLoc(response)
    warp = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        score, warp = cv2.findTransformECC(  # type: ignore[call-overload]
            ideal, ink, warp, cv2.MOTION_AFFINE, criteria, None, 5
        )
    except cv2.error as exc:
        raise CaptureError("align_failed", "ECC did not converge") from exc
    if not math.isfinite(score) or score < MIN_ALIGNMENT:
        raise CaptureError("align_failed", f"alignment correlation {score:.2f} too low")
    patch = cv2.warpAffine(
        crop,
        warp,
        (PATCH_PX, PATCH_PX),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return CdpPatch(patch=patch.astype(np.float32), alignment_score=float(score))
