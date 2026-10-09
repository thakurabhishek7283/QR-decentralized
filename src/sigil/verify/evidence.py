"""Evidence: the facts a verification produced, each addressable by a stable id.

The policy decides from evidence and the investigator cites it, so every claim in a verdict
traces back to an item here. ``data`` holds numbers and enums only; the few strings an attacker
controls (a label's SKU and batch text) live exclusively under ``data["untrusted"]`` so that
nothing downstream can mistake them for system output.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceKind(StrEnum):
    QUALITY = "quality"
    DECODE = "decode"
    SIGNATURE = "signature"
    KEY_STATUS = "key_status"
    LOG_INCLUSION = "log_inclusion"
    CDP = "cdp"
    CLONE = "clone"
    BATCH_BASELINE = "batch_baseline"
    HISTORY = "history"
    RECAPTURE = "recapture"


class Evidence(BaseModel):
    id: str
    kind: EvidenceKind
    passed: bool | None  # None: informational, ambiguous, or not run
    summary: str = Field(max_length=200)  # one line, never untrusted text
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def not_run(cls, id: str, kind: EvidenceKind, because: str) -> Evidence:
        """Placeholder for a check skipped because something it depends on failed."""
        return cls(
            id=id,
            kind=kind,
            passed=None,
            summary=f"not run: {because}",
            data={"status": "not_run", "because": because},
        )

    @property
    def ran(self) -> bool:
        return self.data.get("status") != "not_run"


class EvidenceBundle(BaseModel):
    scan_id: str
    serial_hex: str | None
    items: list[Evidence] = Field(default_factory=list)

    def get(self, id: str) -> Evidence | None:
        return next((item for item in self.items if item.id == id), None)

    def add(self, ev: Evidence) -> None:
        """Add an item, replacing any existing item with the same id in place."""
        for i, item in enumerate(self.items):
            if item.id == ev.id:
                self.items[i] = ev
                return
        self.items.append(ev)
