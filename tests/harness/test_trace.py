from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sigil.harness import (
    AgentRunner,
    Budget,
    EventType,
    JsonlTraceWriter,
    MemoryTraceWriter,
    RunConfig,
    TraceEvent,
    TraceRecorder,
    Usage,
    load_trace,
    summarize,
    verify_chain,
)
from sigil.harness.trace import GENESIS_HASH, canonical_json
from tests.harness.helpers import FakeModel, MinValueGuardrail, call, make_registry, reply


def recorded_events(n: int = 5) -> list[TraceEvent]:
    sink = MemoryTraceWriter()
    recorder = TraceRecorder("run-x", [sink], clock=iter(float(i) for i in range(100)).__next__)
    for i in range(n):
        recorder.emit(EventType.NUDGE, {"i": i, "text": "héllo"})
    return sink.events


def test_canonical_json_is_sorted_compact_and_keeps_unicode() -> None:
    assert canonical_json({"b": [1, 2], "a": "é"}) == '{"a":"é","b":[1,2]}'


def test_chain_links_each_event_to_the_previous_one() -> None:
    events = recorded_events()
    assert events[0].prev_hash == GENESIS_HASH
    assert all(b.prev_hash == a.hash for a, b in pairwise(events))
    assert [e.seq for e in events] == [0, 1, 2, 3, 4]
    assert verify_chain(events)


def test_event_hash_is_sha256_of_canonical_fields() -> None:
    event = recorded_events(1)[0]
    payload = canonical_json(
        {
            "run_id": event.run_id,
            "seq": event.seq,
            "ts": event.ts,
            "type": event.type.value,
            "data": event.data,
            "prev_hash": event.prev_hash,
        }
    )
    assert event.hash == hashlib.sha256(payload.encode()).hexdigest()


def test_empty_chain_verifies() -> None:
    assert verify_chain([])


def test_modifying_a_data_field_is_detected() -> None:
    events = recorded_events()
    events[2] = events[2].model_copy(update={"data": {"i": 999, "text": "héllo"}})
    assert not verify_chain(events)


def test_modifying_and_rehashing_an_event_breaks_the_next_link() -> None:
    events = recorded_events()
    forged = events[2].model_copy(update={"data": {"i": 999}})
    events[2] = forged.model_copy(update={"hash": forged.compute_hash()})
    assert not verify_chain(events)


def test_reordering_is_detected() -> None:
    events = recorded_events()
    events[1], events[2] = events[2], events[1]
    assert not verify_chain(events)


def test_deleting_an_event_is_detected() -> None:
    events = recorded_events()
    del events[2]
    assert not verify_chain(events)


def test_mixing_runs_is_detected() -> None:
    events = recorded_events()
    events[3] = events[3].model_copy(update={"run_id": "other"})
    assert not verify_chain(events)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "run.jsonl"
    memory = MemoryTraceWriter()
    recorder = TraceRecorder("run-j", [JsonlTraceWriter(path), memory])
    for i in range(4):
        recorder.emit(EventType.TOOL_CALL, {"name": "echo", "input": {"i": i, "f": 0.1 + i}})

    loaded = load_trace(path)
    assert loaded == memory.events
    assert verify_chain(loaded)
    assert len(path.read_text().splitlines()) == 4


def test_tampering_with_the_file_is_detected_after_loading(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    recorder = TraceRecorder("run-j", [JsonlTraceWriter(path)])
    for i in range(3):
        recorder.emit(EventType.NUDGE, {"i": i})
    path.write_text(path.read_text().replace('"i":1', '"i":7'))
    assert not verify_chain(load_trace(path))


def test_writers_append_so_a_crash_leaves_a_valid_prefix(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    recorder = TraceRecorder("run-j", [JsonlTraceWriter(path)])
    recorder.emit(EventType.NUDGE, {})
    assert len(load_trace(path)) == 1
    recorder.emit(EventType.NUDGE, {})
    assert verify_chain(load_trace(path))


async def test_summarize_a_real_run(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=5)
    model = FakeModel(
        [
            reply(
                call("c1", "echo", value=1),
                call("c2", "echo", value=2),
                usage=usage,
                model="claude-opus-5-5",
            ),
            reply(call("c3", "submit", value=1), usage=usage, model="claude-opus-5-5"),
            reply(call("c4", "submit", value=50), usage=usage, model="claude-opus-5-5"),
        ]
    )
    runner = AgentRunner(
        model=model,
        tools=make_registry(),
        system="s",
        guardrails=[MinValueGuardrail(10)],
        trace=TraceRecorder("run-s", [JsonlTraceWriter(path)]),
        config=RunConfig(budget=Budget(max_total_tokens=10**9, max_cost_usd=100)),
    )
    result = await runner.run("task")

    summary = summarize(load_trace(path))
    assert summary.run_id == "run-s"
    assert summary.status == "completed"
    assert summary.steps == 3 == result.steps
    assert summary.tools_used == {"echo": 2, "submit": 2}
    assert summary.usage == result.usage == Usage(3_000_000, 300_000, 15)
    assert summary.cost_usd == pytest.approx(result.cost_usd)
    assert summary.guardrail_rejections == 1
    assert summary.wall_s == pytest.approx(result.wall_s)


def test_summary_of_a_truncated_trace_has_no_status() -> None:
    summary = summarize(recorded_events(3))
    assert summary.status is None
    assert summary.wall_s == 2.0  # last ts - first ts from the fake clock


def test_summarize_rejects_empty_traces() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize([])


def test_jsonl_lines_are_valid_json_objects(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    TraceRecorder("r", [JsonlTraceWriter(path)]).emit(EventType.ERROR, {"message": "x"})
    row = json.loads(path.read_text())
    assert set(row) == {"run_id", "seq", "ts", "type", "data", "prev_hash", "hash"}
    assert row["type"] == "error"


@given(length=st.integers(2, 12), data=st.data())
def test_any_single_edit_breaks_the_chain(length: int, data: st.DataObject) -> None:
    events = recorded_events(length)
    index = data.draw(st.integers(0, length - 1))
    edit = data.draw(st.sampled_from(["data", "ts", "type", "delete", "swap"]))
    if edit == "data":
        events[index] = events[index].model_copy(update={"data": {"tampered": True}})
    elif edit == "ts":
        events[index] = events[index].model_copy(update={"ts": events[index].ts + 1})
    elif edit == "type":
        events[index] = events[index].model_copy(update={"type": EventType.ERROR})
    elif edit == "delete" and index < length - 1:  # deleting the tail is a truncation, not an edit
        del events[index]
    elif edit == "swap":
        other = (index + 1) % length
        events[index], events[other] = events[other], events[index]
    else:
        return
    assert not verify_chain(events)
