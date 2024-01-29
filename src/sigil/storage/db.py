"""A small thread-safe wrapper around one SQLite connection.

Every repository shares one ``Database``. Statements are short, so async code
calls it directly instead of hopping to a thread pool.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        with self._lock:
            if str(path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, seq)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            row: sqlite3.Row | None = self._conn.execute(sql, params).fetchone()
            return row

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """``BEGIN IMMEDIATE`` … ``COMMIT``; nested calls join the outer transaction."""
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
            finally:
                self._depth = 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
