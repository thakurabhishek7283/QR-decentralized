"""SQLite persistence for the transparency log: leaves and signed tree heads."""

from __future__ import annotations

from collections.abc import Sequence

from sigil.storage import Database

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS log_leaves (
        idx INTEGER PRIMARY KEY,
        leaf_hash BLOB NOT NULL UNIQUE,
        data BLOB NOT NULL,
        appended_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS log_sth (
        tree_size INTEGER PRIMARY KEY,
        root_hash BLOB NOT NULL,
        timestamp_ms INTEGER NOT NULL,
        signature BLOB NOT NULL
    )
    """,
)

SthRow = tuple[int, bytes, int, bytes]
"""``(tree_size, root_hash, timestamp_ms, signature)``."""


class LogStore:
    def __init__(self, db: Database) -> None:
        self._db = db
        for statement in _SCHEMA:
            db.execute(statement)

    def leaf_hashes(self) -> list[bytes]:
        rows = self._db.query("SELECT leaf_hash FROM log_leaves ORDER BY idx")
        return [bytes(row["leaf_hash"]) for row in rows]

    def add_leaves(
        self, start: int, items: Sequence[tuple[bytes, bytes]], appended_at: int
    ) -> None:
        """Insert ``(leaf_hash, data)`` pairs at consecutive indices from ``start``."""
        self._db.executemany(
            "INSERT INTO log_leaves (idx, leaf_hash, data, appended_at) VALUES (?, ?, ?, ?)",
            [(start + i, lh, data, appended_at) for i, (lh, data) in enumerate(items)],
        )

    def add_sth(self, row: SthRow) -> None:
        self._db.execute(
            "INSERT INTO log_sth (tree_size, root_hash, timestamp_ms, signature) "
            "VALUES (?, ?, ?, ?)",
            row,
        )

    def sth(self, tree_size: int) -> SthRow | None:
        return self._sth("WHERE tree_size = ?", (tree_size,))

    def latest_sth(self) -> SthRow | None:
        return self._sth("ORDER BY tree_size DESC LIMIT 1", ())

    def _sth(self, clause: str, params: tuple[int, ...]) -> SthRow | None:
        row = self._db.query_one(
            f"SELECT tree_size, root_hash, timestamp_ms, signature FROM log_sth {clause}", params
        )
        if row is None:
            return None
        return (
            row["tree_size"],
            bytes(row["root_hash"]),
            row["timestamp_ms"],
            bytes(row["signature"]),
        )
