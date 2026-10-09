"""Event-sourced, tamper-evident run traces.

Each event commits to its predecessor with ``sha256(canonical_json(...))``, so editing,
reordering or deleting an event in the middle of a trace breaks verification. (Truncating the
tail is not detectable by the chain alone; a complete trace ends with ``RUN_FINISHED``, which
``summarize`` surfaces as ``status``.)
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from sigil.harness.types import Usage

GENESIS_HASH = "0" * 64


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GUARDRAIL = "guardrail"
    NUDGE = "nudge"
    ERROR = "error"
    RUN_FINISHED = "run_finished"


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, non-ASCII kept."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _event_hash(
    run_id: str, seq: int, ts: float, type: EventType, data: dict[str, Any], prev_hash: str
) -> str:
    return digest(
        {
            "run_id": run_id,
            "seq": seq,
            "ts": ts,
            "type": type.value,
            "data": data,
            "prev_hash": prev_hash,
        }
    )


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    seq: int
    ts: float
    type: EventType
    data: dict[str, Any]
    prev_hash: str
    hash: str

    def compute_hash(self) -> str:
        return _event_hash(self.run_id, self.seq, self.ts, self.type, self.data, self.prev_hash)


class TraceWriter(Protocol):
    def write(self, event: TraceEvent) -> None: ...


class TraceSink(Protocol):
    """What the runner needs from a trace: an id and a way to append events."""

    @property
    def run_id(self) -> str: ...

    def emit(self, type: EventType, data: dict[str, Any]) -> TraceEvent: ...


class MemoryTraceWriter:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def write(self, event: TraceEvent) -> None:
        self.events.append(event)


class JsonlTraceWriter:
    """Appends one JSON line per event, closing the file each time so a crash loses nothing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: TraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")


class TraceRecorder:
    """Builds the hash chain and fans events out to its writers."""

    def __init__(
        self,
        run_id: str,
        writers: Sequence[TraceWriter] = (),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._run_id = run_id
        self._writers = tuple(writers)
        self._clock = clock
        self._seq = 0
        self._prev_hash = GENESIS_HASH

    @property
    def run_id(self) -> str:
        return self._run_id

    def emit(self, type: EventType, data: dict[str, Any]) -> TraceEvent:
        # Round-trip through JSON so the in-memory event equals what a reader will load.
        data = json.loads(canonical_json(data))
        ts = self._clock()
        event = TraceEvent(
            run_id=self._run_id,
            seq=self._seq,
            ts=ts,
            type=type,
            data=data,
            prev_hash=self._prev_hash,
            hash=_event_hash(self._run_id, self._seq, ts, type, data, self._prev_hash),
        )
        self._seq += 1
        self._prev_hash = event.hash
        for writer in self._writers:
            writer.write(event)
        return event


def load_trace(path: Path) -> list[TraceEvent]:
    with path.open(encoding="utf-8") as f:
        return [TraceEvent.model_validate_json(line) for line in f if line.strip()]


def verify_chain(events: Sequence[TraceEvent]) -> bool:
    """True iff the events form one unbroken chain from genesis."""
    prev_hash = GENESIS_HASH
    for index, event in enumerate(events):
        in_order = event.seq == index and event.run_id == events[0].run_id
        if not (in_order and event.prev_hash == prev_hash and event.hash == event.compute_hash()):
            return False
        prev_hash = event.hash
    return True


@dataclass(frozen=True)
class TraceSummary:
    run_id: str
    status: str | None  # None when the trace has no RUN_FINISHED (crashed or truncated)
    steps: int
    tools_used: dict[str, int]
    usage: Usage
    cost_usd: float
    wall_s: float
    guardrail_rejections: int


def summarize(events: Sequence[TraceEvent]) -> TraceSummary:
    if not events:
        raise ValueError("cannot summarize an empty trace")
    responses = [e.data for e in events if e.type is EventType.MODEL_RESPONSE]
    finished = next((e.data for e in events if e.type is EventType.RUN_FINISHED), None)
    tools = Counter(e.data["name"] for e in events if e.type is EventType.TOOL_CALL)
    rejections = sum(1 for e in events if e.type is EventType.GUARDRAIL and not e.data["accepted"])
    usage = sum((Usage.from_json(r["usage"]) for r in responses), Usage())
    return TraceSummary(
        run_id=events[0].run_id,
        status=finished["status"] if finished else None,
        steps=len(responses),
        tools_used=dict(tools),
        usage=usage,
        cost_usd=sum(r["cost_usd"] for r in responses),
        wall_s=finished["wall_s"] if finished else events[-1].ts - events[0].ts,
        guardrail_rejections=rejections,
    )
