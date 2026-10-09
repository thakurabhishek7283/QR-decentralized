"""The investigator: an LLM agent that settles cases the deterministic policy cannot.

It works inside hard limits. It sees only tool results, the policy bounds which verdicts it may
reach, guardrails vet every submission, budgets cap its cost, and every run leaves a replayable
trace. The offline reference policy provides the same behaviour with no API key.
"""

from sigil.investigator.agent import (
    REFERENCE_MODEL,
    InvestigationOutcome,
    build_model,
    investigate,
)
from sigil.investigator.guardrails import (
    CitationsExist,
    CloneAcknowledged,
    GenuineRequiresCdp,
    PolicyBounds,
    build_guardrails,
)
from sigil.investigator.prompts import SYSTEM_PROMPT, task_message
from sigil.investigator.reference_policy import make_reference_policy
from sigil.investigator.schemas import RecaptureRequest, ScanHistoryRequest, VerdictSubmission
from sigil.investigator.tools import (
    InvestigationDeps,
    RecaptureSource,
    build_registry,
    genuine_floor,
)

__all__ = [
    "REFERENCE_MODEL",
    "SYSTEM_PROMPT",
    "CitationsExist",
    "CloneAcknowledged",
    "GenuineRequiresCdp",
    "InvestigationDeps",
    "InvestigationOutcome",
    "PolicyBounds",
    "RecaptureRequest",
    "RecaptureSource",
    "ScanHistoryRequest",
    "VerdictSubmission",
    "build_guardrails",
    "build_model",
    "build_registry",
    "genuine_floor",
    "investigate",
    "make_reference_policy",
    "task_message",
]
