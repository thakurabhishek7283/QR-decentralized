"""The inference seam: score a CDP patch with the CNN and the classical baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigil.cdp.features import DEFAULT_CALIBRATION, raw_correlation
from sigil.inference.batcher import BatcherConfig, BatcherMetrics, BatchStats, MicroBatcher
from sigil.inference.preprocess import make_input
from sigil.inference.scorer import CdpScorer, sigmoid
from sigil.vision.capture import CdpPatch


@dataclass(frozen=True, slots=True)
class CdpScore:
    cnn: float  # sigmoid(logit), in [0, 1]
    correlation: float  # calibrated classical score, in [0, 1]
    raw_correlation: float  # Pearson r
    alignment: float  # ECC correlation from patch extraction


class InferenceService:
    """Scores patches through a micro-batcher.

    Pre-processing (``make_input`` and the correlation, about a millisecond) runs
    inline on the event loop; the CNN call is batched and runs on the batcher's
    worker thread. A remote implementation would expose the same ``score``.
    """

    def __init__(
        self,
        scorer: CdpScorer,
        batcher_cfg: BatcherConfig | None = None,
        metrics: BatcherMetrics | None = None,
        *,
        timeout_s: float | None = 5.0,
    ) -> None:
        cfg = batcher_cfg or BatcherConfig()
        self._batcher = MicroBatcher(
            scorer.predict_logits,
            max_batch_size=cfg.max_batch_size,
            max_delay_ms=cfg.max_delay_ms,
            queue_limit=cfg.queue_limit,
            metrics=metrics,
            in_flight=lambda: self._in_flight,
        )
        self._in_flight = 0
        self._timeout_s = timeout_s
        calibration = getattr(scorer, "metadata", {}).get("corr_calibration")
        self._calibration = (
            (float(calibration["a"]), float(calibration["b"]))
            if calibration
            else DEFAULT_CALIBRATION
        )

    async def start(self) -> None:
        await self._batcher.start()

    async def stop(self) -> None:
        await self._batcher.stop()

    @property
    def queue_depth(self) -> int:
        return self._batcher.queue_depth

    @property
    def batch_stats(self) -> BatchStats:
        return self._batcher.stats

    async def score(self, patch: CdpPatch, template: np.ndarray) -> CdpScore:
        """Score one extracted patch against its serial's template.

        Raises ``OverloadedError`` when the queue is full and ``TimeoutError`` when
        the model does not answer in time.
        """
        self._in_flight += 1
        try:
            r = raw_correlation(patch.patch, template)
            logit = await self._batcher.submit(make_input(patch.patch, template), self._timeout_s)
        finally:
            self._in_flight -= 1
        a, b = self._calibration
        return CdpScore(
            cnn=float(sigmoid(logit)),
            correlation=float(sigmoid(np.asarray(a * r + b))),
            raw_correlation=r,
            alignment=patch.alignment_score,
        )
