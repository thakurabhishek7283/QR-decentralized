"""Hard limits on a run: steps, tokens, dollars and wall-clock time."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from sigil.harness.types import Usage


@dataclass(frozen=True)
class Budget:
    max_steps: int = 12
    max_total_tokens: int = 300_000
    max_cost_usd: float = 1.0
    max_wall_s: float = 120.0


class BudgetTracker:
    """Accumulates consumption and says which limit, if any, has been reached.

    The clock is injected so wall-clock behaviour can be tested without sleeping.
    """

    def __init__(self, budget: Budget, clock: Callable[[], float] = time.monotonic) -> None:
        self._budget = budget
        self._clock = clock
        self._started = clock()
        self._steps = 0
        self._usage = Usage()
        self._cost_usd = 0.0

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def cost_usd(self) -> float:
        return self._cost_usd

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def record(self, usage: Usage, cost_usd: float) -> None:
        self._usage += usage
        self._cost_usd += cost_usd

    def next_step(self) -> None:
        self._steps += 1

    def exceeded(self) -> str | None:
        """Name of the first limit that has been reached: steps, tokens, cost or wall."""
        b = self._budget
        if self._steps >= b.max_steps:
            return "steps"
        if self._usage.total >= b.max_total_tokens:
            return "tokens"
        if self._cost_usd >= b.max_cost_usd:
            return "cost"
        if self.elapsed_s >= b.max_wall_s:
            return "wall"
        return None

    def snapshot(self) -> dict[str, float]:
        return {
            "steps": self._steps,
            "tokens": self._usage.total,
            "cost_usd": self._cost_usd,
            "wall_s": self.elapsed_s,
        }
