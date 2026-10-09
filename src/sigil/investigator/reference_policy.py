"""The offline reference investigator: an expert procedure written as a ``ScriptedModel`` policy.

It is deterministic, free, and needs no API key, which makes it the floor an LLM must beat in the
evals, the engine behind the CI gate, and a worked example of how a good investigator behaves.

The procedure, in order:

1. Read all evidence. If clone risk is non-zero, read the scan history.
2. If the CDP score is ambiguous, compare it with the batch baseline.
3. If it is still ambiguous, ask for another photo, since a poor first photo is the commonest reason
   a genuine item scores low.
4. Decide conservatively. A wrongly accepted counterfeit is the costly mistake, so ``GENUINE``
   needs two further photos that both agree with the batch (a single lucky photo is not enough),
   and clone signals are never talked away by a good score alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sigil.harness import (
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from sigil.harness.providers.scripted import Policy
from sigil.verify import PolicyConfig

MAX_RECAPTURES = 2
CORROBORATING_RECAPTURES = 2  # photos that must all agree with the batch to call it GENUINE
Json = dict[str, Any]


def make_reference_policy(cfg: PolicyConfig) -> Policy:
    """Build the policy for a given set of verification thresholds."""

    def policy(request: ModelRequest) -> ModelResponse:
        return _Case(cfg, request.messages).respond()

    return policy


@dataclass
class _Case:
    """Everything the expert knows, re-derived from the conversation on every turn.

    The policy keeps no state of its own, so a replayed or resumed run behaves identically.
    """

    cfg: PolicyConfig
    messages: tuple[Message, ...]

    def __post_init__(self) -> None:
        self.calls = 0
        self.results: dict[str, list[Json]] = {}
        names: dict[str, str] = {}
        for message in self.messages:
            for block in message.blocks:
                if isinstance(block, ToolCall):
                    names[block.id] = block.name
                    self.calls += 1
                elif isinstance(block, ToolResult) and not block.is_error:
                    self.results.setdefault(names[block.call_id], []).append(
                        json.loads(block.content)
                    )
        self.task = "".join(b.text for b in self.messages[0].blocks if isinstance(b, TextBlock))

    # --- what the case looks like ------------------------------------------------------------

    @property
    def allowed(self) -> set[str]:
        line = next(ln for ln in self.task.splitlines() if ln.startswith("agent_may_conclude:"))
        return {v.strip() for v in line.split(":", 1)[1].split(",") if v.strip()}

    @property
    def rule(self) -> str:
        line = next(ln for ln in self.task.splitlines() if ln.startswith("policy_rule:"))
        return line.split(":", 1)[1].strip()

    def item(self, evidence_id: str) -> Json:
        shown = self.results.get("get_evidence", [{}])[0].get("evidence", [])
        return next((e for e in shown if e["id"] == evidence_id), {"data": {}})

    @property
    def score(self) -> float | None:
        value = self.item("cdp")["data"].get("cnn")
        return None if value is None else float(value)

    @property
    def clone_risk(self) -> float:
        return float(self.item("clone")["data"].get("risk", 0.0))

    @property
    def has_clone(self) -> bool:
        return self.clone_risk >= self.cfg.clone_risk_low

    @property
    def ambiguous(self) -> bool:
        return self.score is not None and self.score < self.cfg.cdp_genuine_min

    @property
    def floor(self) -> float:
        """Lowest score consistent with a genuine item of this batch."""
        baseline = self.results.get("get_batch_baseline", [{}])[0].get("data", {})
        if baseline.get("available"):
            return float(baseline["genuine_floor"])
        return self.cfg.cdp_genuine_min

    @property
    def recaptures(self) -> list[tuple[str, float]]:
        """``(evidence id, score)`` of every recapture that produced a usable measurement."""
        return [
            (r["id"], r["data"]["cnn"])
            for r in self.results.get("request_recapture", [])
            if "cnn" in r.get("data", {})
        ]

    @property
    def recapture_scores(self) -> list[float]:
        return [score for _, score in self.recaptures]

    @property
    def recaptures_unavailable(self) -> bool:
        return any(r.get("available") is False for r in self.results.get("request_recapture", []))

    @property
    def recaptures_done(self) -> int:
        return len(self.results.get("request_recapture", []))

    # --- what to do next ---------------------------------------------------------------------

    def respond(self) -> ModelResponse:
        if "get_evidence" not in self.results:
            return self._call(("get_evidence", {}))
        gather = []
        if (
            self.clone_risk > 0 or self.rule == "R9_clone"
        ) and "get_scan_history" not in self.results:
            gather.append(("get_scan_history", {"limit": 20}))
        if self.ambiguous and "get_batch_baseline" not in self.results:
            gather.append(("get_batch_baseline", {}))
        if gather:
            return self._call(*gather)
        if self._wants_recapture():
            reason = "The first photo was inconclusive; a closer, steadier photo would settle it."
            return self._call(("request_recapture", {"reason": reason}))
        return self._call(("submit_verdict", self._conclude()))

    def _wants_recapture(self) -> bool:
        if not self.ambiguous or self.recaptures_unavailable:
            return False
        if self.recaptures_done >= MAX_RECAPTURES:
            return False
        scores = self.recapture_scores
        if self.has_clone:
            return not scores and (self.score or 0.0) < self.floor
        if "GENUINE" not in self.allowed:
            return not scores
        # A genuine conclusion needs corroboration, so keep asking while every photo agrees.
        return all(x >= self.floor for x in scores) and len(scores) < CORROBORATING_RECAPTURES

    # --- the verdict -------------------------------------------------------------------------

    def _conclude(self) -> Json:
        verdict, confidence, why = self._judge()
        if verdict not in self.allowed:
            verdict, confidence = "INCONCLUSIVE", 0.5
            why += " The policy does not allow a decisive verdict here."
        cited = ["cdp", *(evidence_id for evidence_id, _ in self.recaptures)]
        if "get_batch_baseline" in self.results:
            cited.append("batch-baseline")
        if self.clone_risk > 0:
            cited.append("clone")
            if "get_scan_history" in self.results:
                cited.append("history")
        return {
            "verdict": verdict,
            "confidence": confidence,
            "cited_evidence": cited,
            "rationale": why,
        }

    def _judge(self) -> tuple[str, float, str]:
        """``(verdict, confidence, rationale)`` before the policy's limits are applied."""
        s, photos = self.score, self.recapture_scores
        shown = f"The photographed pattern scores {s:.2f}" if s is not None else "No score"
        ceiling = self.cfg.cdp_counterfeit_max
        if photos and max(photos) <= ceiling:
            return (
                "COUNTERFEIT",
                0.85,
                f"{shown}, and a closer recapture scores {max(photos):.2f}, at or below the copy "
                f"threshold {ceiling:.2f}. A better photo does not recover the pattern, which "
                "is what a copy looks like.",
            )
        measured = ([s] if s is not None else []) + photos
        if self.has_clone:
            return self._judge_cloned(measured)
        mid = (ceiling + self.cfg.cdp_genuine_min) / 2
        agree = all(x >= self.floor for x in photos)
        corroborated = agree and (
            len(photos) >= CORROBORATING_RECAPTURES
            or (self.recaptures_unavailable and s is not None and s >= max(self.floor, mid))
        )
        if corroborated:
            return (
                "GENUINE",
                0.85,
                f"{shown}, which alone is ambiguous, but it is consistent with the batch "
                f"(lowest genuine score {self.floor:.2f}) and further photos agree. No clone "
                "signals.",
            )
        return (
            "INCONCLUSIVE",
            0.5,
            f"{shown} and the evidence does not separate a genuine item from a copy: no "
            "corroborating measurement is consistent with the batch.",
        )

    def _judge_cloned(self, measured: list[float]) -> tuple[str, float, str]:
        risk = f"Clone risk is {self.clone_risk:.2f}, so copies of this serial exist"
        if measured and all(m < self.floor for m in measured):
            scores = ", ".join(f"{m:.2f}" for m in measured)
            return (
                "COUNTERFEIT",
                0.8,
                f"{risk}, and every CDP measurement ({scores}) is below the {self.floor:.2f} "
                "expected of a genuine item. This unit is likely one of the copies.",
            )
        return (
            "INCONCLUSIVE",
            0.5,
            f"{risk}, but the CDP measurements do not show that this unit is one of them. It "
            "may be the original.",
        )

    def _call(self, *calls: tuple[str, Json]) -> ModelResponse:
        blocks = tuple(
            ToolCall(f"ref-{self.calls + i + 1}", name, args)
            for i, (name, args) in enumerate(calls)
        )
        return ModelResponse(blocks, StopReason.TOOL_USE, Usage(), "reference")
