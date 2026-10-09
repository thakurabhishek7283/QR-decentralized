"""Model price table and cost arithmetic."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sigil.harness.types import Usage

_log = logging.getLogger(__name__)
_PER_MTOK = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Dollars per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    @classmethod
    def anthropic(cls, input: float, output: float, cache_read: float) -> ModelPrice:
        """Cache writes are billed at 1.25x the input price."""
        return cls(input, output, cache_read, cache_write=input * 1.25)


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice.anthropic(4.0, 20.0, 0.20),
    "claude-sonnet-5-5": ModelPrice.anthropic(2.0, 10.0, 0.20),
    "claude-haiku-5-5": ModelPrice.anthropic(0.10, 0.50, 0.01),
    "claude-fable-5-1": ModelPrice.anthropic(10.0, 50.0, 0.25),
    "reference": ModelPrice(0.0, 0.0, 0.0, 0.0),
}

_warned: set[str] = set()


def _price_for(model: str) -> ModelPrice | None:
    """Exact match first, then the longest known prefix (dated snapshot ids)."""
    if model in PRICES:
        return PRICES[model]
    prefixes = [name for name in PRICES if model.startswith(f"{name}-")]
    return PRICES[max(prefixes, key=len)] if prefixes else None


def cost_usd(model: str, usage: Usage) -> float:
    """Dollar cost of ``usage``. Unknown models cost 0.0 and log one warning."""
    price = _price_for(model)
    if price is None:
        if model not in _warned:
            _warned.add(model)
            _log.warning("no price for model %r; counting its cost as 0.0", model)
        return 0.0
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    ) / _PER_MTOK
