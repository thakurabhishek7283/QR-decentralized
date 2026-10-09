"""Scan telemetry: one row per verification, feeding clone analytics."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sigil.errors import NotFoundError
from sigil.storage import Database

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS scans (
        id TEXT PRIMARY KEY,
        serial_hex TEXT,
        ts INTEGER NOT NULL,
        lat REAL,
        lon REAL,
        device_hash TEXT,
        verdict TEXT,
        cdp_score REAL,
        payload_json TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS scans_serial_ts ON scans(serial_hex, ts)",
)
_COLUMNS = "id, serial_hex, ts, lat, lon, device_hash, verdict, cdp_score, payload_json"


@dataclass(frozen=True, slots=True)
class ScanEvent:
    id: str
    serial_hex: str | None
    ts: datetime
    lat: float | None
    lon: float | None
    device_hash: str | None
    verdict: str | None = None
    cdp_score: float | None = None

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError("ScanEvent.ts must be timezone-aware")


class ScanStore:
    def __init__(self, db: Database) -> None:
        self._db = db
        for statement in _SCHEMA:
            db.execute(statement)

    def add(self, ev: ScanEvent, payload: dict[str, Any] | None = None) -> None:
        self._db.execute(
            f"INSERT INTO scans ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ev.id,
                ev.serial_hex,
                _to_ms(ev.ts),
                ev.lat,
                ev.lon,
                ev.device_hash,
                ev.verdict,
                ev.cdp_score,
                None if payload is None else json.dumps(payload),
            ),
        )

    def update_verdict(self, scan_id: str, verdict: str) -> None:
        cur = self._db.execute("UPDATE scans SET verdict = ? WHERE id = ?", (verdict, scan_id))
        if cur.rowcount == 0:
            raise NotFoundError(f"unknown scan {scan_id}")

    def get(self, scan_id: str) -> tuple[ScanEvent, dict[str, Any] | None]:
        row = self._db.query_one(f"SELECT {_COLUMNS} FROM scans WHERE id = ?", (scan_id,))
        if row is None:
            raise NotFoundError(f"unknown scan {scan_id}")
        payload = None if row["payload_json"] is None else json.loads(row["payload_json"])
        return _to_event(row), payload

    def history(
        self,
        serial_hex: str,
        *,
        before: datetime | None = None,
        within: timedelta = timedelta(days=90),
        limit: int = 200,
    ) -> list[ScanEvent]:
        """The most recent ``limit`` scans of a serial in ``[before - within, before)``.

        ``before`` defaults to now, and the window then has no upper bound. Results
        are chronological.
        """
        reference = before or datetime.now(UTC)
        since = _to_ms(reference - within)
        until = _to_ms(before) if before else None
        rows = self._db.query(
            f"SELECT {_COLUMNS} FROM scans WHERE serial_hex = ? AND ts >= ?"
            " AND (? IS NULL OR ts < ?) ORDER BY ts DESC, id DESC LIMIT ?",
            (serial_hex, since, until, until, limit),
        )
        return [_to_event(row) for row in reversed(rows)]


def hash_device(device_id: str, salt: bytes) -> str:
    """Pseudonymise a device id; raw identifiers are never stored."""
    return hmac.new(salt, device_id.encode(), hashlib.sha256).hexdigest()[:16]


def _to_ms(ts: datetime) -> int:
    return round(ts.timestamp() * 1000)


def _to_event(row: sqlite3.Row) -> ScanEvent:
    return ScanEvent(
        id=row["id"],
        serial_hex=row["serial_hex"],
        ts=datetime.fromtimestamp(row["ts"] / 1000, UTC),
        lat=row["lat"],
        lon=row["lon"],
        device_hash=row["device_hash"],
        verdict=row["verdict"],
        cdp_score=row["cdp_score"],
    )
