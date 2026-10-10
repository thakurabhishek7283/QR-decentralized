from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Path

from sigil.api.deps import ContainerDep
from sigil.api.investigations import read_events
from sigil.api.schemas import InvestigationView
from sigil.harness import summarize, verify_chain

router = APIRouter(tags=["investigations"])


@router.get("/investigations/{investigation_id}")
async def get_investigation(
    investigation_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")], c: ContainerDep
) -> InvestigationView:
    """Status, outcome and the agent's trace. While pending, the trace so far is returned, so a
    client can render the run as it unfolds."""
    row = c.investigations.get(investigation_id)
    events = read_events(c.settings.traces_dir / f"{investigation_id}.jsonl")
    return InvestigationView(
        id=row.id,
        scan_id=row.scan_id,
        status=row.status,
        created_at=row.created_at,
        finished_at=row.finished_at,
        outcome=row.outcome,
        error=row.error,
        chain_verified=verify_chain(events) if events else None,
        trace_summary=asdict(summarize(events)) if events else None,
        events=[event.model_dump(mode="json") for event in events],
    )
