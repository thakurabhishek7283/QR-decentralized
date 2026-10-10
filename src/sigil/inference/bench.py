"""Closed-loop benchmark of the inference service, with and without micro-batching.

``python -m sigil.inference.bench`` prints a markdown table. Inputs are real
patches produced once by the simulator, so the numbers reflect serving cost
(pre-processing, queueing, ONNX Runtime) rather than simulation.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from prometheus_client import CollectorRegistry

from sigil.cdp import generate_template, simulate_capture
from sigil.errors import CaptureError
from sigil.inference.batcher import BatcherConfig, BatcherMetrics
from sigil.inference.scorer import CdpScorer, OnnxCdpScorer
from sigil.inference.service import InferenceService
from sigil.vision import CdpPatch, decode, extract_cdp_patch, render_label

Sample = tuple[CdpPatch, np.ndarray]
BASE45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


@dataclass(frozen=True, slots=True)
class BenchResult:
    requests: int
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_batch_size: float


@dataclass(frozen=True, slots=True)
class BenchRow:
    mode: str  # "unbatched" or "batched"
    concurrency: int
    result: BenchResult


def make_samples(n: int = 24, seed: int = 7) -> list[Sample]:
    """Simulate ``n`` genuine captures and extract their (patch, template) pairs."""
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    while len(samples) < n:
        template = generate_template(rng.bytes(32), rng.bytes(12))
        payload = "SG1:" + "".join(BASE45[i] for i in rng.integers(0, len(BASE45), 180))
        photo = simulate_capture(render_label(payload, template), rng, difficulty="easy")
        try:
            samples.append((extract_cdp_patch(decode(photo), template), template))
        except CaptureError:
            continue
    return samples


async def run_benchmark(
    service_factory: Callable[[], InferenceService],
    *,
    concurrency: int,
    duration_s: float,
    warmup_s: float = 1.0,
    samples: Sequence[Sample] | None = None,
) -> BenchResult:
    """Drive ``concurrency`` closed-loop clients for ``duration_s`` after a warm-up."""
    pool = list(samples) if samples is not None else make_samples()
    service = service_factory()
    await service.start()
    loop = asyncio.get_running_loop()
    completions: list[tuple[float, float]] = []  # (finish time, latency)
    running = True

    async def client(offset: int) -> None:
        i = offset
        while running:
            patch, template = pool[i % len(pool)]
            i += concurrency
            t0 = loop.time()
            await service.score(patch, template)
            completions.append((loop.time(), loop.time() - t0))

    tasks = [asyncio.create_task(client(k)) for k in range(concurrency)]
    await asyncio.sleep(warmup_s)
    window_start, before = loop.time(), service.batch_stats
    await asyncio.sleep(duration_s)
    window_end, after = loop.time(), service.batch_stats
    running = False
    await asyncio.gather(*tasks)
    await service.stop()

    latencies = np.array([lat for done, lat in completions if window_start <= done <= window_end])
    if latencies.size == 0:
        raise RuntimeError("no requests completed inside the measurement window")
    p50, p95, p99 = np.percentile(latencies, [50, 95, 99]) * 1000.0
    batches, items = after.batches - before.batches, after.items - before.items
    return BenchResult(
        requests=int(latencies.size),
        throughput_rps=float(latencies.size / (window_end - window_start)),
        p50_ms=float(p50), p95_ms=float(p95), p99_ms=float(p99),
        mean_batch_size=items / batches if batches else 0.0,
    )  # fmt: skip


def compare_batching(
    concurrency_levels: Sequence[int] = (1, 8, 32),
    duration_s: float = 5.0,
    *,
    scorer: CdpScorer | None = None,
    batched: BatcherConfig | None = None,
    samples: Sequence[Sample] | None = None,
) -> list[BenchRow]:
    """Benchmark ``max_batch_size=1`` against the batched configuration."""
    scorer = scorer or OnnxCdpScorer()
    pool = list(samples) if samples is not None else make_samples()
    configs = {"unbatched": BatcherConfig(max_batch_size=1), "batched": batched or BatcherConfig()}
    rows: list[BenchRow] = []
    for concurrency in concurrency_levels:
        for mode, cfg in configs.items():

            def factory(cfg: BatcherConfig = cfg) -> InferenceService:
                return InferenceService(scorer, cfg, BatcherMetrics(CollectorRegistry()))

            result = asyncio.run(
                run_benchmark(factory, concurrency=concurrency, duration_s=duration_s, samples=pool)
            )
            rows.append(BenchRow(mode, concurrency, result))
    return rows


def format_markdown(rows: Sequence[BenchRow]) -> str:
    """Render benchmark rows as a markdown table, with speed-up over the unbatched baseline."""
    base = {r.concurrency: r.result.throughput_rps for r in rows if r.mode == "unbatched"}
    lines = [
        "| concurrency | mode | req/s | speed-up | p50 ms | p95 ms | p99 ms | mean batch |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        x = r.result
        speedup = f"{x.throughput_rps / base[r.concurrency]:.2f}x" if r.concurrency in base else "-"
        lines.append(
            f"| {r.concurrency} | {r.mode} | {x.throughput_rps:.1f} | {speedup} | {x.p50_ms:.1f} "
            f"| {x.p95_ms:.1f} | {x.p99_ms:.1f} | {x.mean_batch_size:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", default="1,8,32", help="comma-separated client counts")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds per measurement")
    parser.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads")
    parser.add_argument("--out", type=Path, default=None, help="also write the table here")
    args = parser.parse_args(argv)
    levels = [int(c) for c in args.concurrency.split(",")]
    started = time.perf_counter()
    rows = compare_batching(levels, args.duration, scorer=OnnxCdpScorer(threads=args.threads))
    table = format_markdown(rows)
    print(table)
    print(f"\n({time.perf_counter() - started:.0f} s total)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")


if __name__ == "__main__":
    main()
