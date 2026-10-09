"""Persistence for issued batches and units."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sigil.crypto import SignedCredential
from sigil.errors import NotFoundError
from sigil.storage import Database

PrintQuality = Literal["good", "worn"]

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS batches (
        batch_id TEXT PRIMARY KEY,
        sku TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        print_quality TEXT NOT NULL DEFAULT 'good',
        qc_n INTEGER, qc_mean REAL, qc_std REAL, qc_min REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS units (
        serial_hex TEXT PRIMARY KEY,
        batch_id TEXT NOT NULL REFERENCES batches(batch_id),
        credential BLOB NOT NULL,
        leaf_index INTEGER NOT NULL,
        issued_at INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS units_batch ON units(batch_id)",
)


@dataclass(frozen=True, slots=True)
class QcStats:
    """CDP score distribution of a batch, measured by the production-line QC camera."""

    n: int
    mean: float
    std: float
    min: float


@dataclass(frozen=True, slots=True)
class BatchInfo:
    batch_id: str
    sku: str
    created_at: datetime
    print_quality: PrintQuality
    qc: QcStats | None = None


@dataclass(frozen=True, slots=True)
class IssuedUnit:
    serial_hex: str
    payload: str
    leaf_index: int


class IssuanceStore:
    def __init__(self, db: Database) -> None:
        self._db = db
        for statement in _SCHEMA:
            db.execute(statement)

    def add_batch(self, batch: BatchInfo, units: Sequence[tuple[SignedCredential, int]]) -> None:
        """Record a batch with its ``(credential, leaf_index)`` units atomically."""
        issued_at = _to_ms(batch.created_at)
        with self._db.transaction():
            self._db.execute(
                "INSERT INTO batches (batch_id, sku, created_at, print_quality)"
                " VALUES (?, ?, ?, ?)",
                (batch.batch_id, batch.sku, issued_at, batch.print_quality),
            )
            self._db.executemany(
                "INSERT INTO units (serial_hex, batch_id, credential, leaf_index, issued_at)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (sc.credential.serial_hex, batch.batch_id, sc.to_bytes(), index, issued_at)
                    for sc, index in units
                ],
            )

    def set_qc(self, batch_id: str, qc: QcStats) -> None:
        cur = self._db.execute(
            "UPDATE batches SET qc_n = ?, qc_mean = ?, qc_std = ?, qc_min = ? WHERE batch_id = ?",
            (qc.n, qc.mean, qc.std, qc.min, batch_id),
        )
        if cur.rowcount == 0:
            raise NotFoundError(f"unknown batch {batch_id}")

    def batch(self, batch_id: str) -> BatchInfo:
        row = self._db.query_one("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))
        if row is None:
            raise NotFoundError(f"unknown batch {batch_id}")
        return _to_batch(row)

    def batches(self) -> list[BatchInfo]:
        """All batches, oldest first."""
        rows = self._db.query("SELECT * FROM batches ORDER BY created_at, batch_id")
        return [_to_batch(row) for row in rows]

    def unit(self, serial_hex: str) -> tuple[SignedCredential, str]:
        """The credential issued for a serial, and the batch it belongs to."""
        row = self._db.query_one(
            "SELECT credential, batch_id FROM units WHERE serial_hex = ?", (serial_hex,)
        )
        if row is None:
            raise NotFoundError(f"serial {serial_hex} was never issued")
        return SignedCredential.from_bytes(row["credential"]), row["batch_id"]

    def units(self, batch_id: str) -> list[str]:
        """Serials of a batch in issue order."""
        rows = self._db.query(
            "SELECT serial_hex FROM units WHERE batch_id = ? ORDER BY leaf_index", (batch_id,)
        )
        return [row["serial_hex"] for row in rows]


def _to_ms(ts: datetime) -> int:
    return round(ts.timestamp() * 1000)


def _to_batch(row: sqlite3.Row) -> BatchInfo:
    qc = (
        None
        if row["qc_n"] is None
        else QcStats(row["qc_n"], row["qc_mean"], row["qc_std"], row["qc_min"])
    )
    return BatchInfo(
        batch_id=row["batch_id"],
        sku=row["sku"],
        created_at=datetime.fromtimestamp(row["created_at"] / 1000, UTC),
        print_quality=row["print_quality"],
        qc=qc,
    )
