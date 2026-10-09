"""Scan telemetry and clone-risk analytics."""

from sigil.scans.clone import (
    CloneAssessment,
    CloneConfig,
    CloneSignal,
    assess_clone_risk,
    haversine_km,
)
from sigil.scans.store import ScanEvent, ScanStore, hash_device

__all__ = [
    "CloneAssessment",
    "CloneConfig",
    "CloneSignal",
    "ScanEvent",
    "ScanStore",
    "assess_clone_risk",
    "hash_device",
    "haversine_km",
]
