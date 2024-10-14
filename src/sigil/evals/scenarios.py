"""The labelled scenario catalogue the verification pipeline is measured against.

Each kind pairs a construction (see ``world``) with a ground truth and the set of final verdicts
that count as correct. Two rules hold for every kind and are enforced here: ``GENUINE`` is never
acceptable for a counterfeit, and ``COUNTERFEIT`` is never acceptable for a genuine item.
Declining to decide (``SUSPICIOUS``) or asking for a better photo (``RECAPTURE``) is acceptable
only where a careful system could reasonably do so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from sigil.verify import Verdict

Truth = Literal["GENUINE", "COUNTERFEIT"]


class ScenarioKind(StrEnum):
    GENUINE_CLEAN = "genuine_clean"
    GENUINE_HARD_CAPTURE = "genuine_hard_capture"
    GENUINE_WORN_PRINTER = "genuine_worn_printer"
    GENUINE_CLONE_VICTIM = "genuine_clone_victim"
    PHOTOCOPY = "photocopy"
    HQ_REPRINT = "hq_reprint"
    SCREEN_REPLAY = "screen_replay"
    CLONED_SERIAL = "cloned_serial"
    FORGED_SIGNATURE = "forged_signature"
    ROGUE_UNLOGGED = "rogue_unlogged"


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: ScenarioKind
    seed: int
    truth: Truth
    acceptable: frozenset[Verdict]  # final verdicts graded as correct
    ideal: Verdict

    def __post_init__(self) -> None:
        forbidden = Verdict.GENUINE if self.truth == "COUNTERFEIT" else Verdict.COUNTERFEIT
        if forbidden in self.acceptable or self.ideal not in self.acceptable:
            raise ValueError(f"inconsistent scenario definition for {self.kind}")


G, C, S, R = Verdict.GENUINE, Verdict.COUNTERFEIT, Verdict.SUSPICIOUS, Verdict.RECAPTURE

# kind -> (truth, acceptable verdicts with the ideal first)
CATALOGUE: dict[ScenarioKind, tuple[Truth, tuple[Verdict, ...]]] = {
    ScenarioKind.GENUINE_CLEAN: ("GENUINE", (G,)),
    ScenarioKind.GENUINE_HARD_CAPTURE: ("GENUINE", (G, R, S)),
    ScenarioKind.GENUINE_WORN_PRINTER: ("GENUINE", (G, S)),
    ScenarioKind.GENUINE_CLONE_VICTIM: ("GENUINE", (S, G)),
    ScenarioKind.PHOTOCOPY: ("COUNTERFEIT", (C, S)),
    ScenarioKind.HQ_REPRINT: ("COUNTERFEIT", (C, S)),
    ScenarioKind.SCREEN_REPLAY: ("COUNTERFEIT", (C, S, R)),
    ScenarioKind.CLONED_SERIAL: ("COUNTERFEIT", (C, S)),
    ScenarioKind.FORGED_SIGNATURE: ("COUNTERFEIT", (C,)),
    ScenarioKind.ROGUE_UNLOGGED: ("COUNTERFEIT", (C,)),
}


def build_scenarios(
    seeds_per_kind: int, kinds: Sequence[ScenarioKind] | None = None
) -> list[Scenario]:
    """``seeds_per_kind`` scenarios (seeds 0..n-1) of each requested kind, in catalogue order."""
    scenarios = []
    for kind in kinds or list(ScenarioKind):
        truth, verdicts = CATALOGUE[kind]
        scenarios += [
            Scenario(f"{kind.value}-{seed}", kind, seed, truth, frozenset(verdicts), verdicts[0])
            for seed in range(seeds_per_kind)
        ]
    return scenarios
