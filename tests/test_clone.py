from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sigil.scans import (
    CloneAssessment,
    CloneConfig,
    ScanEvent,
    assess_clone_risk,
    haversine_km,
)

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
PARIS = (48.8566, 2.3522)
LONDON = (51.5074, -0.1278)
NYC = (40.7128, -74.0060)
TOKYO = (35.6762, 139.6503)
SYDNEY = (-33.8688, 151.2093)


def scan(
    i: int,
    where: tuple[float, float] | None = PARIS,
    *,
    at: datetime | None = None,
    device: str | None = "dev0",
) -> ScanEvent:
    return ScanEvent(
        id=f"s{i}",
        serial_hex="ab" * 12,
        ts=at or T0 + timedelta(hours=i),
        lat=where[0] if where else None,
        lon=where[1] if where else None,
        device_hash=device,
    )


def assess(scans: list[ScanEvent], cfg: CloneConfig | None = None) -> CloneAssessment:
    *history, current = scans
    return assess_clone_risk(history, current, cfg or CloneConfig())


def names(scans: list[ScanEvent]) -> list[str]:
    return [s.name for s in assess(scans).signals]


# --- haversine -----------------------------------------------------------------------------------


def test_haversine_known_distances() -> None:
    assert haversine_km(PARIS, PARIS) == 0.0
    assert haversine_km(PARIS, LONDON) == pytest.approx(343.5, abs=1.0)
    assert haversine_km(NYC, LONDON) == pytest.approx(5570, abs=15)
    assert haversine_km((0, 0), (0, 180)) == pytest.approx(math.pi * 6371.0088, rel=1e-6)


@given(
    st.tuples(st.floats(-90, 90), st.floats(-180, 180)),
    st.tuples(st.floats(-90, 90), st.floats(-180, 180)),
)
def test_haversine_is_a_symmetric_bounded_distance(
    a: tuple[float, float], b: tuple[float, float]
) -> None:
    d = haversine_km(a, b)
    assert d == pytest.approx(haversine_km(b, a))
    assert 0 <= d <= math.pi * 6371.0088 + 1e-6


# --- baselines -----------------------------------------------------------------------------------


def test_a_single_scan_has_no_risk() -> None:
    result = assess([scan(0)])
    assert (result.risk, result.signals, result.scans_considered) == (0.0, [], 1)


def test_repeat_scans_in_one_city_have_no_risk() -> None:
    scans = [
        scan(i, (PARIS[0] + i * 0.001, PARIS[1]), at=T0 + timedelta(minutes=i)) for i in range(20)
    ]
    result = assess(scans)
    assert result.risk == 0.0
    assert result.scans_considered == 20


def test_a_plausible_flight_is_not_impossible_travel() -> None:
    assert names([scan(0, PARIS, at=T0), scan(1, NYC, at=T0 + timedelta(hours=9))]) == []


def test_scans_without_location_never_trigger_geo_signals() -> None:
    assert names([scan(0, None), scan(1, None), scan(2, None)]) == []


# --- individual signals --------------------------------------------------------------------------


def test_impossible_travel() -> None:
    result = assess([scan(0, PARIS, at=T0), scan(1, NYC, at=T0 + timedelta(hours=1))])
    assert [s.name for s in result.signals] == ["impossible_travel"]
    assert result.signals[0].weight == 0.80
    assert result.risk == pytest.approx(0.80)
    assert "km" in result.signals[0].detail


def test_impossible_travel_when_scanned_simultaneously_in_two_places() -> None:
    assert names([scan(0, PARIS, at=T0), scan(1, NYC, at=T0)]) == ["impossible_travel"]


def test_short_hops_are_ignored_however_fast() -> None:
    near = (PARIS[0] + 0.2, PARIS[1])  # about 22 km
    assert names([scan(0, PARIS, at=T0), scan(1, near, at=T0 + timedelta(seconds=1))]) == []


def test_travel_is_judged_between_consecutive_located_scans_only() -> None:
    scans = [
        scan(0, PARIS, at=T0),
        scan(1, None, at=T0 + timedelta(minutes=30)),
        scan(2, NYC, at=T0 + timedelta(hours=1)),
    ]
    assert names(scans) == ["impossible_travel"]


def test_geo_dispersion() -> None:
    scans = [
        scan(0, PARIS, at=T0),
        scan(1, NYC, at=T0 + timedelta(days=2)),
        scan(2, TOKYO, at=T0 + timedelta(days=4)),
    ]
    result = assess(scans)
    assert [s.name for s in result.signals] == ["geo_dispersion"]
    assert result.risk == pytest.approx(0.50)


def test_two_regions_are_not_dispersion() -> None:
    assert names([scan(0, PARIS, at=T0), scan(1, NYC, at=T0 + timedelta(days=2))]) == []


def test_nearby_cities_form_one_cluster() -> None:
    scans = [
        scan(0, PARIS, at=T0),
        scan(1, LONDON, at=T0 + timedelta(days=1)),  # 343 km: separate from Paris at 300 km
        scan(2, (50.0, 1.0), at=T0 + timedelta(days=2)),  # joins Paris (about 190 km)
    ]
    assert names(scans) == []
    assert names([*scans, scan(3, NYC, at=T0 + timedelta(days=3))]) == ["geo_dispersion"]


def test_scan_volume_boundary() -> None:
    def minute_apart(n: int) -> list[ScanEvent]:
        return [scan(i, at=T0 + timedelta(minutes=i), device="dev0") for i in range(n)]

    assert names(minute_apart(25)) == []
    result = assess(minute_apart(26))
    assert [s.name for s in result.signals] == ["scan_volume"]
    assert result.risk == pytest.approx(0.30)


def test_device_diversity_boundary() -> None:
    def devices(n: int) -> list[ScanEvent]:
        return [scan(i, device=f"dev{i}") for i in range(n)]

    assert names(devices(9)) == []
    assert names(devices(10)) == ["device_diversity"]
    assert names([scan(i, device=None) for i in range(12)]) == []


def test_scans_outside_the_window_are_ignored() -> None:
    old = scan(0, NYC, at=T0 - timedelta(days=31))
    assert (
        assess([old, scan(1, PARIS, at=T0 - timedelta(minutes=5)), scan(2, PARIS, at=T0)]).risk == 0
    )
    in_window = scan(0, NYC, at=T0 - timedelta(days=29))
    assert "geo_dispersion" not in names([in_window, scan(2, PARIS, at=T0)])
    assert assess([old, scan(2, PARIS, at=T0)]).scans_considered == 1


def test_the_current_scan_is_not_double_counted() -> None:
    current = scan(5)
    result = assess_clone_risk([scan(1), current], current)
    assert result.scans_considered == 2


def test_config_overrides_thresholds() -> None:
    strict = CloneConfig(volume_max=3, volume_weight=0.5)
    scans = [scan(i, at=T0 + timedelta(minutes=i)) for i in range(5)]
    assert [s.name for s in assess(scans, strict).signals] == ["scan_volume"]
    assert assess(scans, strict).risk == pytest.approx(0.5)


# --- combination ---------------------------------------------------------------------------------


def test_risk_combines_signals_as_a_noisy_or() -> None:
    scans = [
        scan(i, PARIS if i % 2 else NYC, at=T0 + timedelta(minutes=i), device=f"d{i}")
        for i in range(12)
    ]
    result = assess(scans)
    assert {s.name for s in result.signals} == {"impossible_travel", "device_diversity"}
    assert result.risk == pytest.approx(1 - (1 - 0.8) * (1 - 0.3))


def test_risk_grows_with_every_additional_signal() -> None:
    base = [
        scan(i, PARIS, at=T0 + timedelta(minutes=i), device="dev0") for i in range(26)
    ]  # volume
    with_devices = [scan(i, PARIS, at=T0 + timedelta(minutes=i), device=f"d{i}") for i in range(26)]
    with_travel = [
        *with_devices[:-2],
        scan(98, PARIS, at=T0 + timedelta(minutes=98), device="d98"),
        scan(99, NYC, at=T0 + timedelta(minutes=99), device="d99"),
    ]
    with_dispersion = [
        scan(0, PARIS, at=T0 - timedelta(days=3)),
        scan(1, TOKYO, at=T0 - timedelta(days=2)),
        scan(2, SYDNEY, at=T0 - timedelta(days=1)),
        *with_travel,
    ]
    risks = [assess(s).risk for s in (base, with_devices, with_travel, with_dispersion)]
    counts = [len(assess(s).signals) for s in (base, with_devices, with_travel, with_dispersion)]
    assert counts == [1, 2, 3, 4]
    assert risks == sorted(risks)
    assert len(set(risks)) == 4
    assert risks[-1] < 1.0


@given(
    st.lists(
        st.tuples(
            st.sampled_from([PARIS, LONDON, NYC, TOKYO, SYDNEY, None]),
            st.integers(0, 60 * 24 * 40),
            st.sampled_from(["a", "b", "c", None]),
        ),
        min_size=1,
        max_size=40,
    ),
    st.randoms(use_true_random=False),
)
def test_assessment_is_deterministic_and_matches_its_own_signals(
    raw: list[tuple[tuple[float, float] | None, int, str | None]], rnd: random.Random
) -> None:
    scans = [scan(i, w, at=T0 + timedelta(minutes=m), device=d) for i, (w, m, d) in enumerate(raw)]
    current = max(scans, key=lambda s: (s.ts, s.id))
    history = [s for s in scans if s is not current]
    shuffled = history[:]
    rnd.shuffle(shuffled)
    result = assess_clone_risk(history, current)
    assert assess_clone_risk(shuffled, current) == result
    assert result.risk == pytest.approx(1 - math.prod(1 - s.weight for s in result.signals))
    assert 0.0 <= result.risk < 1.0
