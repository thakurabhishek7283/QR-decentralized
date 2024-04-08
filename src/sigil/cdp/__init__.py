"""Copy-detection pattern: keyed template, print/capture simulator, classical features."""

from sigil.cdp.channel import (
    Attack,
    CaptureProfile,
    PrintProfile,
    capture,
    counterfeit_from,
    print_label,
    sample_capture_profile,
    sample_print_profile,
    simulate_capture,
)
from sigil.cdp.features import (
    DEFAULT_CALIBRATION,
    correlation_score,
    raw_correlation,
    template_image,
)
from sigil.cdp.pattern import generate_template

__all__ = [
    "DEFAULT_CALIBRATION",
    "Attack",
    "CaptureProfile",
    "PrintProfile",
    "capture",
    "correlation_score",
    "counterfeit_from",
    "generate_template",
    "print_label",
    "raw_correlation",
    "sample_capture_profile",
    "sample_print_profile",
    "simulate_capture",
    "template_image",
]
