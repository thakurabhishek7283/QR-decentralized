from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from sigil.errors import OverloadedError
from sigil.inference import BatcherMetrics, MicroBatcher


class Recorder:
    """A model stand-in that doubles its input and remembers every batch it saw."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.batches: list[int] = []
        self.delay_s = delay_s
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.gate.wait(timeout=5)
        time.sleep(self.delay_s)
        self.batches.append(len(x))
        return x * 2.0


Make = Callable[..., MicroBatcher]


@pytest.fixture
async def make() -> AsyncIterator[Make]:
    created: list[MicroBatcher] = []

    def factory(fn: Callable[[np.ndarray], np.ndarray], **kwargs: object) -> MicroBatcher:
        batcher = MicroBatcher(fn, **kwargs)  # type: ignore[arg-type]
        created.append(batcher)
        return batcher

    yield factory
    for batcher in created:
        await batcher.stop()


async def test_results_return_to_the_right_callers(make: Make) -> None:
    batcher = make(Recorder(), max_batch_size=8, max_delay_ms=20)
    await batcher.start()
    results = await asyncio.gather(*(batcher.submit(np.full(3, float(i))) for i in range(20)))
    for i, row in enumerate(results):
        assert np.array_equal(row, np.full(3, 2.0 * i))


async def test_batches_never_exceed_the_maximum_and_do_fill(make: Make) -> None:
    fn = Recorder()
    batcher = make(fn, max_batch_size=4, max_delay_ms=50)
    await batcher.start()
    await asyncio.gather(*(batcher.submit(np.zeros(2)) for i in range(18)))
    assert max(fn.batches) <= 4
    assert sum(fn.batches) == 18
    assert max(fn.batches) == 4  # concurrent arrivals were actually batched
    assert batcher.stats.items == 18


async def test_unbatched_mode_runs_one_item_at_a_time(make: Make) -> None:
    fn = Recorder()
    batcher = make(fn, max_batch_size=1)
    await batcher.start()
    await asyncio.gather(*(batcher.submit(np.zeros(2)) for _ in range(6)))
    assert fn.batches == [1] * 6


async def test_lone_request_completes_after_about_one_delay(make: Make) -> None:
    batcher = make(Recorder(), max_batch_size=32, max_delay_ms=30)
    await batcher.start()
    start = time.perf_counter()
    await batcher.submit(np.zeros(2))
    elapsed = time.perf_counter() - start
    assert 0.025 <= elapsed < 0.25


async def test_no_wait_when_the_caller_reports_nobody_else_in_flight(make: Make) -> None:
    batcher = make(Recorder(), max_batch_size=8, max_delay_ms=500, in_flight=lambda: 1)
    await batcher.start()
    began = time.perf_counter()
    await batcher.submit(np.ones(3))
    assert time.perf_counter() - began < 0.25  # did not idle for the 500 ms delay


async def test_waits_for_requests_the_caller_reports_in_flight(make: Make) -> None:
    fn = Recorder()
    batcher = make(fn, max_batch_size=8, max_delay_ms=100, in_flight=lambda: 3)
    await batcher.start()
    first = asyncio.create_task(batcher.submit(np.ones(3)))
    await asyncio.sleep(0.02)  # two more are "in flight" and arrive within the delay
    rest = [asyncio.create_task(batcher.submit(np.ones(3))) for _ in range(2)]
    await asyncio.gather(first, *rest)
    assert fn.batches == [3]


async def test_full_queue_raises_overloaded(make: Make) -> None:
    fn = Recorder()
    fn.gate.clear()  # the worker blocks inside the model
    batcher = make(fn, max_batch_size=1, max_delay_ms=1, queue_limit=2)
    await batcher.start()
    first = asyncio.create_task(batcher.submit(np.zeros(1)))
    await asyncio.sleep(0.05)  # the worker takes it and blocks in fn
    waiting = [asyncio.create_task(batcher.submit(np.zeros(1))) for _ in range(2)]
    await asyncio.sleep(0.05)
    assert batcher.queue_depth == 2
    with pytest.raises(OverloadedError):
        await batcher.submit(np.zeros(1))
    fn.gate.set()
    await asyncio.gather(first, *waiting)


async def test_model_error_reaches_every_caller_in_the_batch_and_worker_survives(
    make: Make,
) -> None:
    calls = 0

    def flaky(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return x

    batcher = make(flaky, max_batch_size=8, max_delay_ms=30)
    await batcher.start()
    outcomes = await asyncio.gather(
        *(batcher.submit(np.zeros(1)) for _ in range(5)), return_exceptions=True
    )
    assert all(isinstance(o, RuntimeError) and str(o) == "boom" for o in outcomes)
    assert np.array_equal(await batcher.submit(np.ones(1)), np.ones(1))


async def test_cancelled_caller_does_not_break_the_worker(make: Make) -> None:
    fn = Recorder(delay_s=0.05)
    batcher = make(fn, max_batch_size=8, max_delay_ms=30)
    await batcher.start()
    doomed = asyncio.create_task(batcher.submit(np.full(1, 5.0)))
    survivor = asyncio.create_task(batcher.submit(np.full(1, 7.0)))
    await asyncio.sleep(0.005)
    doomed.cancel()
    assert np.array_equal(await survivor, np.full(1, 14.0))
    with pytest.raises(asyncio.CancelledError):
        await doomed
    assert np.array_equal(await batcher.submit(np.full(1, 1.0)), np.full(1, 2.0))


async def test_caller_timeout_raises_and_item_is_skipped(make: Make) -> None:
    fn = Recorder()
    fn.gate.clear()
    batcher = make(fn, max_batch_size=1, max_delay_ms=1)
    await batcher.start()
    blocker = asyncio.create_task(batcher.submit(np.zeros(1)))
    await asyncio.sleep(0.05)
    with pytest.raises(TimeoutError):
        await batcher.submit(np.zeros(1), timeout_s=0.05)
    fn.gate.set()
    await blocker
    assert np.array_equal(await batcher.submit(np.ones(1)), np.full(1, 2.0))
    assert sum(fn.batches) == 2  # the timed-out request never reached the model


async def test_stop_drains_pending_work_and_rejects_new(make: Make) -> None:
    fn = Recorder(delay_s=0.02)
    batcher = make(fn, max_batch_size=4, max_delay_ms=5)
    await batcher.start()
    pending = [asyncio.create_task(batcher.submit(np.full(1, float(i)))) for i in range(10)]
    await asyncio.sleep(0)
    await batcher.stop()
    assert [float((await p)[0]) for p in pending] == [2.0 * i for i in range(10)]
    with pytest.raises(OverloadedError):
        await batcher.submit(np.zeros(1))


async def test_submit_before_start_is_rejected(make: Make) -> None:
    with pytest.raises(OverloadedError):
        await make(Recorder()).submit(np.zeros(1))


async def test_can_restart_after_stop(make: Make) -> None:
    batcher = make(Recorder())
    await batcher.start()
    await batcher.stop()
    await batcher.start()
    assert np.array_equal(await batcher.submit(np.ones(1)), np.full(1, 2.0))


async def test_metrics_record_batches_and_rejections(make: Make) -> None:
    registry = CollectorRegistry()
    batcher = make(Recorder(), max_batch_size=4, max_delay_ms=20, metrics=BatcherMetrics(registry))
    await batcher.start()
    await asyncio.gather(*(batcher.submit(np.zeros(1)) for _ in range(8)))
    assert registry.get_sample_value("sigil_inference_batch_size_sum") == 8
    assert registry.get_sample_value("sigil_inference_batch_size_count") >= 2
    assert registry.get_sample_value("sigil_inference_queue_wait_seconds_count") == 8
    assert registry.get_sample_value("sigil_inference_queue_depth") == 0


def test_metrics_can_share_the_default_registry_across_instances() -> None:
    registry = CollectorRegistry()
    BatcherMetrics(registry)
    BatcherMetrics(registry)  # no duplicate-timeseries error


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MicroBatcher(lambda x: x, max_batch_size=0)
