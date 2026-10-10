"""Dynamic micro-batching for CPU inference.

Single requests underuse a CNN: per-call overhead (Python, graph dispatch, thread
fan-out) is paid for one image. Under concurrent load it is cheaper to wait a few
milliseconds, stack whatever arrived into one tensor and run it once. The
batcher does that with one asyncio worker and one executor thread, so ONNX
Runtime's intra-op threads - not Python threads - provide the parallelism.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

from sigil.errors import OverloadedError

_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
_LATENCY_BUCKETS = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512)


@dataclass(frozen=True, slots=True)
class BatcherConfig:
    max_batch_size: int = 32
    max_delay_ms: float = 4.0
    queue_limit: int = 512


@dataclass(frozen=True, slots=True)
class BatchStats:
    batches: int
    items: int

    @property
    def mean_batch_size(self) -> float:
        return self.items / self.batches if self.batches else 0.0


class BatcherMetrics:
    """Prometheus instruments for a batcher.

    ``registry`` makes tests hermetic. Instruments are created once per registry
    and shared, so building several batchers against the default registry does not
    raise duplicate-timeseries errors.
    """

    _cache: weakref.WeakKeyDictionary[CollectorRegistry, dict[str, object]] = (
        weakref.WeakKeyDictionary()
    )

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        reg = registry if registry is not None else REGISTRY
        shared = self._cache.setdefault(reg, {})
        if not shared:
            shared["batch_size"] = Histogram(
                "sigil_inference_batch_size", "Items per executed batch",
                buckets=_BATCH_BUCKETS, registry=reg,
            )  # fmt: skip
            shared["queue_wait"] = Histogram(
                "sigil_inference_queue_wait_seconds", "Time an item waited before its batch ran",
                buckets=_LATENCY_BUCKETS, registry=reg,
            )  # fmt: skip
            shared["compute"] = Histogram(
                "sigil_inference_compute_seconds", "Model time per batch",
                buckets=_LATENCY_BUCKETS, registry=reg,
            )  # fmt: skip
            shared["queue_depth"] = Gauge(
                "sigil_inference_queue_depth", "Items waiting in the batcher queue", registry=reg
            )
            shared["rejected"] = Counter(
                "sigil_inference_rejected_total", "Requests refused because the queue was full",
                registry=reg,
            )  # fmt: skip
        self.batch_size: Histogram = shared["batch_size"]  # type: ignore[assignment]
        self.queue_wait: Histogram = shared["queue_wait"]  # type: ignore[assignment]
        self.compute: Histogram = shared["compute"]  # type: ignore[assignment]
        self.queue_depth: Gauge = shared["queue_depth"]  # type: ignore[assignment]
        self.rejected: Counter = shared["rejected"]  # type: ignore[assignment]


@dataclass(slots=True)
class _Request:
    item: np.ndarray
    future: asyncio.Future[np.ndarray]
    enqueued: float


class MicroBatcher:
    """Collects concurrent ``submit`` calls into batches for ``fn``.

    ``fn`` maps a stacked ``(N, ...)`` array to ``(N, ...)`` results and runs on a
    dedicated thread. A batch closes when it holds ``max_batch_size`` items or
    ``max_delay_ms`` has passed since its first item arrived, so a lone request
    waits at most that long. ``max_batch_size=1`` disables batching.

    ``in_flight`` is an optional hint: a callable returning how many requests the caller
    is currently handling (before, during or after their ``submit``). When every one of
    them is already in the batch or executing there is nobody to wait for, so the batch
    closes at once instead of idling for ``max_delay_ms``; a lone client then pays no
    batching latency. Without the hint the batcher always waits out the delay.
    """

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        max_batch_size: int = 32,
        max_delay_ms: float = 4.0,
        queue_limit: int = 512,
        metrics: BatcherMetrics | None = None,
        in_flight: Callable[[], int] | None = None,
    ) -> None:
        if max_batch_size < 1 or queue_limit < 1:
            raise ValueError("max_batch_size and queue_limit must be positive")
        self._fn = fn
        self._max_batch = max_batch_size
        self._max_delay = max_delay_ms / 1000.0
        self._queue_limit = queue_limit
        self._metrics = metrics
        self._in_flight = in_flight
        self._executing = 0  # items of the batch currently running on the executor
        self._queue: asyncio.Queue[_Request] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closing = False
        self._batches = 0
        self._items = 0

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize() if self._queue else 0

    @property
    def stats(self) -> BatchStats:
        return BatchStats(self._batches, self._items)

    async def start(self) -> None:
        if self._worker is not None:
            return
        self._queue = asyncio.Queue(maxsize=self._queue_limit)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sigil-infer")
        self._closing = False
        self._worker = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        """Refuse new work, finish everything already queued, then stop the worker."""
        if self._worker is None or self._queue is None:
            return
        self._closing = True
        await self._queue.join()  # every queued item has been resolved
        self._worker.cancel()  # the worker is idle on an empty queue
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._worker = self._queue = self._executor = None

    async def submit(self, item: np.ndarray, timeout_s: float | None = 5.0) -> np.ndarray:
        """Queue ``item`` and wait for its result row.

        Raises ``OverloadedError`` immediately if the queue is full (backpressure)
        and ``TimeoutError`` if no result arrives within ``timeout_s``. A caller that
        times out or is cancelled simply leaves a dead future behind; the worker
        skips it.
        """
        if self._queue is None or self._closing:
            raise OverloadedError("inference batcher is not accepting work")
        future: asyncio.Future[np.ndarray] = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait(_Request(item, future, time.perf_counter()))
        except asyncio.QueueFull:
            if self._metrics:
                self._metrics.rejected.inc()
            raise OverloadedError("inference queue is full") from None
        if self._metrics:
            self._metrics.queue_depth.set(self._queue.qsize())
        return await asyncio.wait_for(future, timeout_s)

    # -- worker -------------------------------------------------------------

    async def _collect(self, queue: asyncio.Queue[_Request]) -> list[_Request]:
        batch = [await queue.get()]
        deadline = asyncio.get_running_loop().time() + self._max_delay
        while len(batch) < self._max_batch:
            try:
                batch.append(queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0 or not self._more_coming(len(batch)):
                break
            try:
                async with asyncio.timeout(remaining):
                    batch.append(await queue.get())
            except TimeoutError:
                break
        return batch

    def _more_coming(self, collected: int) -> bool:
        """Whether a request that has not been queued yet is worth waiting for."""
        if self._in_flight is None:
            return True
        return self._in_flight() - self._executing > collected

    async def _run(self) -> None:
        assert self._queue is not None
        assert self._executor is not None
        queue, executor = self._queue, self._executor
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect(queue)
            live = [r for r in batch if not r.future.done()]  # drop timed-out / cancelled callers
            try:
                if live:
                    await self._execute(loop, executor, live)
            finally:
                for _ in batch:
                    queue.task_done()
                if self._metrics:
                    self._metrics.queue_depth.set(queue.qsize())

    async def _execute(
        self, loop: asyncio.AbstractEventLoop, executor: ThreadPoolExecutor, live: list[_Request]
    ) -> None:
        started = time.perf_counter()
        self._executing = len(live)
        try:
            results = await loop.run_in_executor(
                executor, self._fn, np.stack([r.item for r in live])
            )
            if len(results) != len(live):
                raise ValueError(f"model returned {len(results)} rows for {len(live)} inputs")
        except Exception as exc:
            for r in live:
                if not r.future.done():
                    r.future.set_exception(exc)
            return
        finally:
            self._executing = 0
        finished = time.perf_counter()
        self._batches += 1
        self._items += len(live)
        if self._metrics:
            self._metrics.batch_size.observe(len(live))
            self._metrics.compute.observe(finished - started)
            for r in live:
                self._metrics.queue_wait.observe(started - r.enqueued)
        for r, row in zip(live, results, strict=True):
            if not r.future.done():
                r.future.set_result(row)
