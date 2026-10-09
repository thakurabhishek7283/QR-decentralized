"""Typed inputs for the investigator's tools, including the terminal verdict."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VerdictSubmission(BaseModel):
    """The investigator's conclusion; the only way an investigation can end."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["GENUINE", "COUNTERFEIT", "INCONCLUSIVE"]
    confidence: float = Field(ge=0, le=1, description="Your probability that the verdict is right.")
    cited_evidence: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Ids of the evidence items the verdict rests on, e.g. 'cdp', 'history'.",
    )
    rationale: str = Field(min_length=20, max_length=1200)


class ScanHistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=20, ge=1, le=50, description="How many recent scans to return.")


class RecaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(
        min_length=5, max_length=200, description="Why another photo helps, shown to the user."
    )
