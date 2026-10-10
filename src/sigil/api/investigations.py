"""Persistence for background investigations.

The API answers a scan immediately and investigates in the background, so each investigation needs
a durable status the client can poll. The agent's own record is its JSONL trace; this table only
tracks lifecycle (pending, done, error) and the summarised outcome.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sigil.api.schemas import InvestigationStatus, OutcomeView
from sigil.errors import NotFoundError
from sigil.harness import TraceEvent
from sigil.investigator import InvestigationOutcome
from sigil.storage import Database

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS investigations (
        id TEXT PRIMARY KEY,
        scan_id TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        finished_at INTEGER,
        outcome_json TEXT,
        error TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS investigations_scan ON investigations(scan_id)",
)


@dataclass(frozen=True, slots=True)
class InvestigationRow:
    id: str
    scan_id: str
    status: InvestigationStatus
    created_at: datetime
    finished_at: datetime | None
    outcome: OutcomeView | None
    error: str | None


def summarize_outcome(outcome: InvestigationOutcome, model: str) -> OutcomeView:
    sub, run = outcome.submission, outcome.run
    return OutcomeView(
        final_verdict=outcome.final_verdict.value,
        agent_verdict=sub.verdict if sub else None,
        confidence=sub.confidence if sub else None,
        rationale=sub.rationale if sub else None,
        cited_evidence=sub.cited_evidence if sub else [],
        run_status=run.status.value,
        model=model,
        steps=run.steps,
        tokens=run.usage.total,
        cost_usd=run.cost_usd,
        wall_s=run.wall_s,
        guardrail_violations=run.guardrail_violations,
    )


class InvestigationStore:
    def __init__(self, db: Database) -> None:
        self._db = db
        for statement in _SCHEMA:
            db.execute(statement)

    def create(self, investigation_id: str, scan_id: str) -> None:
        self._db.execute(
            "INSERT INTO investigations (id, scan_id, status, created_at)"
            " VALUES (?, ?, 'pending', ?)",
            (investigation_id, scan_id, _now_ms()),
        )

    def finish(self, investigation_id: str, outcome: OutcomeView) -> None:
        self._db.execute(
            "UPDATE investigations SET status = 'done', finished_at = ?, outcome_json = ?"
            " WHERE id = ?",
            (_now_ms(), outcome.model_dump_json(), investigation_id),
        )

    def fail(self, investigation_id: str, error: str) -> None:
        self._db.execute(
            "UPDATE investigations SET status = 'error', finished_at = ?, error = ? WHERE id = ?",
            (_now_ms(), error, investigation_id),
        )

    def fail_pending(self, error: str) -> int:
        """Mark investigations a previous process never finished; returns how many there were."""
        cur = self._db.execute(
            "UPDATE investigations SET status = 'error', finished_at = ?, error = ?"
            " WHERE status = 'pending'",
            (_now_ms(), error),
        )
        return cur.rowcount

    def get(self, investigation_id: str) -> InvestigationRow:
        row = self._db.query_one("SELECT * FROM investigations WHERE id = ?", (investigation_id,))
        if row is None:
            raise NotFoundError(f"unknown investigation {investigation_id}")
        return _to_row(row)

    def for_scan(self, scan_id: str) -> InvestigationRow | None:
        row = self._db.query_one(
            "SELECT * FROM investigations WHERE scan_id = ? ORDER BY created_at DESC LIMIT 1",
            (scan_id,),
        )
        return _to_row(row) if row else None


def _now_ms() -> int:
    return round(time.time() * 1000)


def _to_row(row: sqlite3.Row) -> InvestigationRow:
    finished = row["finished_at"]
    outcome = row["outcome_json"]
    return InvestigationRow(
        id=row["id"],
        scan_id=row["scan_id"],
        status=row["status"],
        created_at=datetime.fromtimestamp(row["created_at"] / 1000, UTC),
        finished_at=None if finished is None else datetime.fromtimestamp(finished / 1000, UTC),
        outcome=None if outcome is None else OutcomeView.model_validate_json(outcome),
        error=row["error"],
    )


def read_events(path: Path) -> list[TraceEvent]:
    """The events of a trace file, which may still be growing.

    A line cut off by a concurrent write ends the list rather than failing the read.
    """
    events: list[TraceEvent] = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                events.append(TraceEvent.model_validate_json(line))
            except ValueError:
                break
    return events
