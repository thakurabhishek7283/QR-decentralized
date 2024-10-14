"""``python -m sigil.evals``: run the scenario suite, write the report, optionally gate on it."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from sigil.evals.report import check_gate, load_baseline, write
from sigil.evals.runner import run_suite
from sigil.evals.scenarios import ScenarioKind


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sigil.evals", description=__doc__)
    parser.add_argument("--seeds", type=int, default=3, help="scenarios per kind")
    parser.add_argument(
        "--kinds",
        type=lambda s: [ScenarioKind(k) for k in s.split(",")],
        help="comma-separated scenario kinds (default: all)",
    )
    parser.add_argument("--gate", action="store_true", help="exit 1 if the baseline is breached")
    parser.add_argument("--out", type=Path, default=Path("reports/eval"))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    suite = asyncio.run(
        run_suite(
            seeds_per_kind=args.seeds,
            out_dir=args.out,
            kinds=args.kinds,
            concurrency=args.concurrency,
        )
    )
    baseline = load_baseline() if args.gate else None
    report, _ = write(suite, args.out, baseline)
    m = suite.metrics
    print(
        f"{m['cases']} cases in {suite.duration_s:.0f} s | accuracy {m['accuracy']:.3f} | "
        f"false accept {m['false_accept_rate']} | false reject {m['false_reject_rate']}"
    )
    print(f"report: {report}")
    if baseline is not None and (breaches := check_gate(m, baseline)):
        print("GATE FAILED:\n  " + "\n  ".join(breaches), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
