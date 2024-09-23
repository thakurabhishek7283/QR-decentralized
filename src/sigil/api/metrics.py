"""Prometheus instruments for the API, kept on a registry the container owns.

A private registry (rather than the global one) keeps several apps in one process, such as tests,
from colliding on metric names, and lets the batcher's instruments share the same ``/metrics``.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_SCAN_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0)


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
        self.log_size = Gauge(
            "sigil_log_size", "Entries in the transparency log", registry=registry
        )
