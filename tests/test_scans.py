from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from sigil.errors import NotFoundError
from sigil.scans import ScanEvent, ScanStore, hash_device
from sigil.storage import Database

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
SERIAL = "ab" * 12


def event(
    i: int, *, at: datetime | None = None, serial: str | None = SERIAL, **kw: Any
) -> ScanEvent:
    fields: dict[str, Any] = {
        "id": f"scan-{i}",
        "serial_hex": serial,
        "ts": at or T0 + timedelta(minutes=i),
        "lat": 48.85,
        "lon": 2.35,
        "device_hash": "dev0",
    }
    return ScanEvent(**{**fields, **kw})


@pytest.fixture
def store(db: Database) -> ScanStore:
    return ScanStore(db)


def test_add_and_get_round_trip(store: ScanStore) -> None:
    ev = event(1, verdict="GENUINE", cdp_score=0.91)
    store.add(ev, {"quality": {"sharpness": 120.5}, "tags": ["a"]})
    got, payload = store.get("scan-1")
    assert got == ev
    assert payload == {"quality": {"sharpness": 120.5}, "tags": ["a"]}


def test_optional_fields_may_be_absent(store: ScanStore) -> None:
    ev = event(1, serial=None, lat=None, lon=None, device_hash=None)
    store.add(ev)
    assert store.get("scan-1") == (ev, None)


def test_timestamps_keep_millisecond_precision(store: ScanStore) -> None:
    ev = event(1, at=T0 + timedelta(milliseconds=1234))
    store.add(ev)
    assert store.get("scan-1")[0].ts == ev.ts


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        event(1, at=datetime(2026, 1, 1))


def test_unknown_scan_is_not_found(store: ScanStore) -> None:
    with pytest.raises(NotFoundError):
        store.get("missing")
    with pytest.raises(NotFoundError):
        store.update_verdict("missing", "GENUINE")


def test_update_verdict(store: ScanStore) -> None:
    store.add(event(1))
    store.update_verdict("scan-1", "SUSPICIOUS")
    assert store.get("scan-1")[0].verdict == "SUSPICIOUS"


def test_history_is_chronological_and_per_serial(store: ScanStore) -> None:
    for i in (3, 1, 2):
        store.add(event(i))
    store.add(event(9, serial="cd" * 12))
    assert [e.id for e in store.history(SERIAL, before=T0 + timedelta(hours=1))] == [
        "scan-1",
        "scan-2",
        "scan-3",
    ]


def test_history_before_is_exclusive(store: ScanStore) -> None:
    for i in range(4):
        store.add(event(i))
    cutoff = T0 + timedelta(minutes=2)
    assert [e.id for e in store.history(SERIAL, before=cutoff)] == ["scan-0", "scan-1"]


def test_history_window_is_measured_back_from_before(store: ScanStore) -> None:
    store.add(event(1, at=T0 - timedelta(days=100)))
    store.add(event(2, at=T0 - timedelta(days=40)))
    store.add(event(3, at=T0 - timedelta(days=5)))
    assert [e.id for e in store.history(SERIAL, before=T0)] == ["scan-2", "scan-3"]
    assert [e.id for e in store.history(SERIAL, before=T0, within=timedelta(days=10))] == ["scan-3"]


def test_history_limit_keeps_the_most_recent_scans(store: ScanStore) -> None:
    for i in range(10):
        store.add(event(i))
    recent = store.history(SERIAL, before=T0 + timedelta(hours=1), limit=3)
    assert [e.id for e in recent] == ["scan-7", "scan-8", "scan-9"]


def test_history_without_before_looks_back_from_now(store: ScanStore) -> None:
    now = datetime.now(UTC)
    store.add(event(1, at=now - timedelta(days=1)))
    store.add(event(2, at=now - timedelta(days=200)))
    assert [e.id for e in store.history(SERIAL)] == ["scan-1"]


def test_hash_device_is_stable_salted_and_short() -> None:
    a = hash_device("iPhone-123", b"salt")
    assert a == hash_device("iPhone-123", b"salt")
    assert a != hash_device("iPhone-124", b"salt")
    assert a != hash_device("iPhone-123", b"pepper")
    assert len(a) == 16
    assert "iPhone" not in a
    int(a, 16)
