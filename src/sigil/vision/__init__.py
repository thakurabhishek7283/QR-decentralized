"""Label geometry and rendering, QR decoding, rectification and capture quality."""

from sigil.vision.capture import CdpPatch, DecodedCapture, decode, extract_cdp_patch, load_image
from sigil.vision.geometry import (
    CDP_CELLS,
    LABEL_H,
    LABEL_W,
    PATCH_PX,
    cdp_corners_label,
    qr_corners_label,
)
from sigil.vision.label import label_png, render_label, render_qr_matrix
from sigil.vision.quality import QualityReport, assess_quality

__all__ = [
    "CDP_CELLS",
    "LABEL_H",
    "LABEL_W",
    "PATCH_PX",
    "CdpPatch",
    "DecodedCapture",
    "QualityReport",
    "assess_quality",
    "cdp_corners_label",
    "decode",
    "extract_cdp_patch",
    "label_png",
    "load_image",
    "qr_corners_label",
    "render_label",
    "render_qr_matrix",
]
