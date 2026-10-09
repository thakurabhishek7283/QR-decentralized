from __future__ import annotations

import logging

import pytest

from sigil.harness import PRICES, Budget, BudgetTracker, Usage, cost_usd, pricing
from tests.harness.helpers import FakeClock


def test_fresh_tracker_is_within_budget() -> None:
    assert BudgetTracker(Budget(), FakeClock()).exceeded() is None


def test_steps_limit() -> None:
    tracker = BudgetTracker(Budget(max_steps=2), FakeClock())
    tracker.next_step()
    assert tracker.exceeded() is None
    tracker.next_step()
    assert tracker.exceeded() == "steps"


def test_token_limit_counts_every_token_kind() -> None:
    tracker = BudgetTracker(Budget(max_total_tokens=100), FakeClock())
    tracker.record(Usage(10, 20, 30, 39), 0.0)
    assert tracker.exceeded() is None
    tracker.record(Usage(output_tokens=1), 0.0)
    assert tracker.exceeded() == "tokens"


def test_cost_limit() -> None:
    tracker = BudgetTracker(Budget(max_cost_usd=0.5), FakeClock())
    tracker.record(Usage(), 0.49)
    assert tracker.exceeded() is None
    tracker.record(Usage(), 0.01)
    assert tracker.exceeded() == "cost"


def test_wall_limit_uses_the_injected_clock() -> None:
    clock = FakeClock()
    tracker = BudgetTracker(Budget(max_wall_s=10), clock)
    clock.advance(9.9)
    assert tracker.exceeded() is None
    clock.advance(0.1)
    assert tracker.exceeded() == "wall"


def test_snapshot() -> None:
    clock = FakeClock()
    tracker = BudgetTracker(Budget(), clock)
    tracker.next_step()
    tracker.record(Usage(1, 2), 0.25)
    clock.advance(3)
    assert tracker.snapshot() == {"steps": 1, "tokens": 3, "cost_usd": 0.25, "wall_s": 3}


def test_first_exceeded_limit_is_reported_in_priority_order() -> None:
    tracker = BudgetTracker(Budget(max_steps=1, max_total_tokens=1), FakeClock())
    tracker.next_step()
    tracker.record(Usage(5), 0.0)
    assert tracker.exceeded() == "steps"


# --- pricing -----------------------------------------------------------------------------------


def test_cost_math_for_every_token_kind() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=400_000,
    )
    # 4.00 input + 2.00 output + 0.40 cache read + 400k * 5.00/MTok cache write
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(4.0 + 2.0 + 0.4 + 2.0)


@pytest.mark.parametrize(
    ("model", "input_", "output", "cache_read", "cache_write"),
    [
        ("claude-opus-5-5", 4, 20, 0.20, 5.0),
        ("claude-sonnet-5-5", 2, 10, 0.20, 2.5),
        ("claude-haiku-5-5", 0.10, 0.50, 0.01, 0.125),
        ("claude-fable-5-1", 10, 50, 0.25, 12.5),
        ("reference", 0, 0, 0, 0),
    ],
)
def test_price_table(
    model: str, input_: float, output: float, cache_read: float, cache_write: float
) -> None:
    price = PRICES[model]
    assert (price.input, price.output, price.cache_read, price.cache_write) == pytest.approx(
        (input_, output, cache_read, cache_write)
    )


def test_reference_model_is_free() -> None:
    assert cost_usd("reference", Usage(10**9, 10**9, 10**9, 10**9)) == 0.0


def test_dated_snapshot_ids_use_the_family_price() -> None:
    usage = Usage(output_tokens=1_000_000)
    assert cost_usd("claude-opus-5-5-20261001", usage) == cost_usd("claude-opus-5-5", usage)


def test_unknown_model_costs_zero_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    pricing._warned.discard("mystery-model")
    with caplog.at_level(logging.WARNING, logger="sigil.harness.pricing"):
        assert cost_usd("mystery-model", Usage(1000, 1000)) == 0.0
        assert cost_usd("mystery-model", Usage(1000, 1000)) == 0.0
    assert [r.getMessage() for r in caplog.records].count(
        "no price for model 'mystery-model'; counting its cost as 0.0"
    ) == 1
