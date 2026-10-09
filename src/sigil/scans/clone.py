"""Clone analytics: does a serial's scan history look like one object or many copies?"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sigil.scans.store import ScanEvent

EARTH_RADIUS_KM = 6371.0088

LatLon = tuple[float, float]


@dataclass(frozen=True, slots=True)
class CloneConfig:
    window: timedelta = timedelta(days=30)
    travel_min_km: float = 50.0
    travel_max_kmh: float = 900.0
    travel_weight: float = 0.80
    cluster_km: float = 300.0
    dispersion_clusters: int = 3
    dispersion_weight: float = 0.50
    volume_max: int = 25
    volume_weight: float = 0.30
    device_min: int = 10
    device_weight: float = 0.30


DEFAULT_CONFIG = CloneConfig()


@dataclass(frozen=True, slots=True)
class CloneSignal:
    name: str
    weight: float
    detail: str


@dataclass(frozen=True, slots=True)
class CloneAssessment:
    risk: float
    signals: list[CloneSignal] = field(default_factory=list)
    scans_considered: int = 0


def haversine_km(a: LatLon, b: LatLon) -> float:
    """Great-circle distance between two ``(lat, lon)`` points in degrees."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + (
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def assess_clone_risk(
    history: Sequence[ScanEvent], current: ScanEvent, cfg: CloneConfig = DEFAULT_CONFIG
) -> CloneAssessment:
    """Combine independent signals as ``1 - prod(1 - w)`` over those that fire.

    Only scans in the config window ending at ``current`` count; ``current`` itself
    is always included, and a stored copy of it in ``history`` is ignored.
    """
    earliest = current.ts - cfg.window
    scans = sorted(
        [ev for ev in history if ev.id != current.id and earliest <= ev.ts <= current.ts]
        + [current],
        key=lambda ev: (ev.ts, ev.id),
    )
    points = [
        (ev.ts, (ev.lat, ev.lon)) for ev in scans if ev.lat is not None and ev.lon is not None
    ]

    checks = (
        _impossible_travel(points, cfg),
        _geo_dispersion([where for _, where in points], cfg),
        _scan_volume(len(scans), cfg),
        _device_diversity(scans, cfg),
    )
    signals = [signal for signal in checks if signal is not None]
    risk = 1.0 - math.prod(1.0 - s.weight for s in signals)
    return CloneAssessment(risk=risk, signals=signals, scans_considered=len(scans))


def _impossible_travel(
    points: Sequence[tuple[datetime, LatLon]], cfg: CloneConfig
) -> CloneSignal | None:
    """Flag the fastest hop between consecutive located scans if it beats ``travel_max_kmh``."""
    worst: tuple[float, float, float] | None = None  # (speed, km, hours)
    for (t0, a), (t1, b) in itertools.pairwise(points):
        km = haversine_km(a, b)
        if km <= cfg.travel_min_km:
            continue
        hours = (t1 - t0).total_seconds() / 3600
        speed = km / hours if hours > 0 else math.inf
        if speed > cfg.travel_max_kmh and (worst is None or speed > worst[0]):
            worst = (speed, km, hours)
    if worst is None:
        return None
    speed, km, hours = worst
    detail = f"{km:.0f} km between consecutive scans in {hours:.2f} h"
    return CloneSignal("impossible_travel", cfg.travel_weight, detail)


def _geo_dispersion(places: Sequence[LatLon], cfg: CloneConfig) -> CloneSignal | None:
    """Greedy leader clustering: a scan joins the first cluster whose leader is near enough."""
    leaders: list[LatLon] = []
    for place in places:
        if all(haversine_km(place, leader) > cfg.cluster_km for leader in leaders):
            leaders.append(place)
    if len(leaders) < cfg.dispersion_clusters:
        return None
    detail = f"scans fall in {len(leaders)} regions more than {cfg.cluster_km:.0f} km apart"
    return CloneSignal("geo_dispersion", cfg.dispersion_weight, detail)


def _scan_volume(count: int, cfg: CloneConfig) -> CloneSignal | None:
    if count <= cfg.volume_max:
        return None
    return CloneSignal("scan_volume", cfg.volume_weight, f"{count} scans in the window")


def _device_diversity(scans: Sequence[ScanEvent], cfg: CloneConfig) -> CloneSignal | None:
    devices = {ev.device_hash for ev in scans if ev.device_hash is not None}
    if len(devices) < cfg.device_min:
        return None
    return CloneSignal("device_diversity", cfg.device_weight, f"{len(devices)} distinct devices")
