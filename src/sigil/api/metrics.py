"""Prometheus instruments for the API, kept on a registry the container owns.

A private registry (rather than the global one) keeps several apps in one process, such as tests,
from colliding on metric names, and lets the batcher's instruments share the same ``/metrics``.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sigil.api.schemas import OutcomeView

_SCAN_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0)
_STEP_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
_INVESTIGATION_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120)


class ApiMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.scans = Counter(
            "sigil_scans_total", "Scans by the verdict the policy returned",
            ["verdict"], registry=registry,
        )  # fmt: skip
        self.scan_seconds = Histogram(
            "sigil_scan_seconds", "End-to-end scan latency, from uploaded bytes to verdict",
            buckets=_SCAN_BUCKETS, registry=registry,
        )  # fmt: skip
        self.investigations = Counter(
            "sigil_investigations_total", "Finished investigations by the verdict acted on",
            ["verdict"], registry=registry,
        )  # fmt: skip
        self.investigation_errors = Counter(
            "sigil_investigation_errors_total", "Investigations that crashed", registry=registry
        )
        self.investigation_steps = Histogram(
            "sigil_investigation_steps", "Model turns per investigation",
            buckets=_STEP_BUCKETS, registry=registry,
        )  # fmt: skip
        self.investigation_seconds = Histogram(
            "sigil_investigation_seconds", "Wall-clock time per investigation",
            buckets=_INVESTIGATION_BUCKETS, registry=registry,
        )  # fmt: skip
        self.investigation_cost = Counter(
            "sigil_investigation_cost_usd_total", "Model spend on investigations", registry=registry
        )
        self.investigation_tokens = Counter(
            "sigil_investigation_tokens_total", "Tokens used by investigations", registry=registry
        )
        self.guardrail_violations = Counter(
            "sigil_guardrail_violations_total",
            "Verdict submissions a guardrail rejected", registry=registry,
        )  # fmt: skip
        self.log_size = Gauge(
            "sigil_log_size", "Entries in the transparency log", registry=registry
        )

    def observe_investigation(self, outcome: OutcomeView) -> None:
        self.investigations.labels(outcome.final_verdict).inc()
        self.investigation_steps.observe(outcome.steps)
        self.investigation_seconds.observe(outcome.wall_s)
        self.investigation_cost.inc(outcome.cost_usd)
        self.investigation_tokens.inc(outcome.tokens)
        self.guardrail_violations.inc(len(outcome.guardrail_violations))
