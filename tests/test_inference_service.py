from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from sigil.errors import OverloadedError
from sigil.inference import (
    INPUT_SHAPE,
    BatcherConfig,
    BatcherMetrics,
    CdpScore,
    InferenceService,
    OnnxCdpScorer,
)
from sigil.inference.bench import compare_batching, format_markdown, make_samples
from sigil.vision import CdpPatch


@pytest.fixture(scope="module")
def samples() -> list[tuple[CdpPatch, np.ndarray]]:
    """A handful of genuine, easy captures (the benchmark's own sample generator)."""
    return make_samples(8, seed=11)


@pytest.fixture(scope="module")
def sample(samples: list[tuple[CdpPatch, np.ndarray]]) -> tuple[CdpPatch, np.ndarray]:
    return samples[0]


@pytest.fixture
async def service() -> AsyncIterator[InferenceService]:
    svc = InferenceService(
        OnnxCdpScorer(),
        BatcherConfig(max_batch_size=8, max_delay_ms=5),
        BatcherMetrics(CollectorRegistry()),
    )
    await svc.start()
    yield svc
    await svc.stop()


async def test_score_end_to_end(
    service: InferenceService, sample: tuple[CdpPatch, np.ndarray]
) -> None:
    patch, template = sample
    score = await service.score(patch, template)
    assert isinstance(score, CdpScore)
    assert 0.0 <= score.cnn <= 1.0
    assert 0.0 <= score.correlation <= 1.0
    assert -1.0 <= score.raw_correlation <= 1.0
    assert score.alignment == patch.alignment_score
    assert score.raw_correlation > 0.5


async def test_genuine_captures_score_as_genuine(
    service: InferenceService, samples: list[tuple[CdpPatch, np.ndarray]]
) -> None:
    """A single capture can land anywhere in the model's genuine score band (a blurry or
    worn-printer one is legitimately low), so assert on the population: most easy genuine
    captures lean genuine and the typical one is clearly above the ambiguous band."""
    scores = await asyncio.gather(*(service.score(patch, tpl) for patch, tpl in samples))
    cnn = np.array([s.cnn for s in scores])
    assert np.median(cnn) > 0.5
    assert np.mean(cnn > 0.5) >= 0.6
    assert all(s.raw_correlation > 0.5 for s in scores)


async def test_concurrent_scores_are_batched_and_consistent(
    service: InferenceService, sample: tuple[CdpPatch, np.ndarray]
) -> None:
    patch, template = sample
    scores = await asyncio.gather(*(service.score(patch, template) for _ in range(16)))
    assert len({round(s.cnn, 5) for s in scores}) == 1
    assert service.batch_stats.mean_batch_size > 1


async def test_overload_surfaces_as_domain_error(sample: tuple[CdpPatch, np.ndarray]) -> None:
    class Slow:
        def predict_logits(self, batch: np.ndarray) -> np.ndarray:
            import time

            time.sleep(0.2)
            return np.zeros(len(batch), np.float32)

    svc = InferenceService(Slow(), BatcherConfig(max_batch_size=1, max_delay_ms=1, queue_limit=1))
    await svc.start()
    patch, template = sample
    tasks = [asyncio.create_task(svc.score(patch, template)) for _ in range(6)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    await svc.stop()
    assert any(isinstance(r, OverloadedError) for r in results)
    assert any(isinstance(r, CdpScore) for r in results)


def test_input_shape_matches_model() -> None:
    assert INPUT_SHAPE == (2, 212, 212)


@pytest.mark.slow
def test_benchmark_runs_and_formats() -> None:
    rows = compare_batching((2,), duration_s=0.5, samples=make_samples(4, seed=3))
    assert [r.mode for r in rows] == ["unbatched", "batched"]
    assert all(r.result.requests > 0 and r.result.throughput_rps > 0 for r in rows)
    table = format_markdown(rows)
    assert table.count("\n") == 3
    assert "p95" in table
