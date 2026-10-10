"""The ``sigil`` command line: run the server, a narrated demo, evals, benchmarks and audits."""

from __future__ import annotations

import asyncio
import base64
import re
import tempfile
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import cv2
import httpx
import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sigil.api.container import Container
from sigil.api.schemas import ScanResponse, SimulateRequest
from sigil.api.simulation import Photo
from sigil.cdp import simulate_capture
from sigil.config import Settings
from sigil.crypto import Ed25519Signer, issue
from sigil.errors import NotFoundError, SigilError
from sigil.evals.scenarios import build_scenarios
from sigil.evals.world import ClaimedKid, ScenarioWorld
from sigil.harness import EventType, TraceEvent, load_trace, replay, summarize, verify_chain
from sigil.investigator import (
    SYSTEM_PROMPT,
    InvestigationDeps,
    InvestigationOutcome,
    build_guardrails,
    build_registry,
    investigate,
)
from sigil.issuance import QcStats
from sigil.ledger import (
    Auditor,
    ConsistencyProof,
    SignedTreeHead,
    leaf_hash,
    verify_consistency,
    verify_inclusion,
)
from sigil.verify import EvidenceBundle, ScanContext, Verdict, VerificationResult
from sigil.vision import CDP_CELLS, render_label

app = typer.Typer(
    name="sigil",
    help="Counterfeit detection: signed QR credentials, copy-detection vision, a transparency "
    "log and an audited AI investigator.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
trace_app = typer.Typer(help="Inspect, verify and replay agent traces.", no_args_is_help=True)
app.add_typer(trace_app, name="trace")

console = Console()
err = Console(stderr=True)

DataDir = Annotated[
    Path | None,
    typer.Option(
        "--data-dir", help="Keys, database and traces. Default: SIGIL_DATA_DIR or ./.sigil"
    ),
]
Attack = Annotated[
    str | None,
    typer.Option("--attack", help="Counterfeit the label first: photocopy, hq_reprint or screen."),
]
Difficulty = Annotated[str, typer.Option(help="Shooting conditions: easy, normal or hard.")]

VERDICT_STYLE = {
    Verdict.GENUINE: "bold green",
    Verdict.COUNTERFEIT: "bold red",
    Verdict.SUSPICIOUS: "bold yellow",
    Verdict.RECAPTURE: "bold cyan",
}
CITIES = {"Mumbai": (19.08, 72.88), "London": (51.51, -0.13), "Tokyo": (35.68, 139.69)}
DEMO_SEED = 11  # makes the demo's simulated photos, and so its narrative, reproducible
MAX_RETAKES = 12


def _settings(data_dir: Path | None) -> Settings:
    return Settings(data_dir=data_dir) if data_dir else Settings()


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion, turning a domain error into a clean exit."""
    try:
        return asyncio.run(coro)
    except SigilError as exc:
        err.print(f"[red]error:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc


def _load_trace(path: Path) -> list[TraceEvent]:
    try:
        events = load_trace(path)
    except ValueError as exc:  # also pydantic's ValidationError
        console.print(f"[red]FAILED[/] {path}: not a valid trace file")
        raise typer.Exit(1) from exc
    if not events:
        console.print(f"[red]FAILED[/] {path}: the trace is empty")
        raise typer.Exit(1)
    return events


def _simulate_request(attack: str | None, difficulty: str, seed: int | None) -> SimulateRequest:
    try:
        return SimulateRequest.model_validate(
            {"attack": attack, "difficulty": difficulty, "seed": seed}
        )
    except ValueError as exc:
        err.print(f"[red]error:[/] invalid simulation options: {escape(str(exc))}")
        raise typer.Exit(2) from exc


# --- server ------------------------------------------------------------------------------------


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    data_dir: DataDir = None,
) -> None:
    """Serve the API and the web demo. The first start issues demo batches."""
    import uvicorn

    from sigil.api import create_app

    uvicorn.run(create_app(_settings(data_dir)), host=host, port=port)


# --- operating on a local data directory --------------------------------------------------------


@app.command("issue")
def issue_labels(
    sku: Annotated[str, typer.Option(help="Product SKU (1-32 printable ASCII characters).")],
    count: Annotated[int, typer.Option(min=1, max=200, help="Units to issue.")] = 10,
    quality: Annotated[str, typer.Option(help="Press quality: good or worn.")] = "good",
    labels: Annotated[
        Path | None, typer.Option(help="Write each label as a PNG into this directory.")
    ] = None,
    data_dir: DataDir = None,
) -> None:
    """Issue a batch of signed, logged labels and measure the batch's print quality."""
    if quality not in ("good", "worn"):
        err.print("[red]error:[/] --quality must be good or worn")
        raise typer.Exit(2)
    print_quality: Literal["good", "worn"] = "worn" if quality == "worn" else "good"

    async def go() -> None:
        async with Container.open(_settings(data_dir)) as c:
            info = await c.create_batch(sku, count, print_quality)
            serials = c.issuance.units(info.batch_id)
            console.print(_batch_panel(info.batch_id, info.sku, info.print_quality, info.qc))
            console.print(f"transparency log now holds {c.log.size} credentials")
            if labels is not None:
                labels.mkdir(parents=True, exist_ok=True)
                for serial in serials:
                    (labels / f"{serial}.png").write_bytes(await c.label_png(serial))
                console.print(f"wrote {len(serials)} labels to {labels}/")
            else:
                console.print("serials: " + ", ".join(serials[:5]) + (" ..." if count > 5 else ""))

    _run(go())


@app.command()
def simulate(
    serial: Annotated[str, typer.Argument(help="Serial of an issued label.")],
    out: Annotated[Path, typer.Option("-o", "--out", help="Where to write the JPEG.")] = Path(
        "photo.jpg"
    ),
    attack: Attack = None,
    difficulty: Difficulty = "normal",
    seed: Annotated[int | None, typer.Option(help="Reproducible photo.")] = None,
    data_dir: DataDir = None,
) -> None:
    """Simulate a phone photo of an issued label, genuine or counterfeited (demo only)."""
    request = _simulate_request(attack, difficulty, seed)

    async def go() -> None:
        async with Container.open(_settings(data_dir)) as c:
            photo = await c.photograph(serial, request)
            out.write_bytes(photo.jpeg)
            what = f"a {request.attack} counterfeit" if request.attack else "the genuine label"
            console.print(f"wrote {out} ({len(photo.jpeg) // 1024} KB): {what}, {difficulty} shot")

    _run(go())


@app.command()
def scan(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A photo of a label.")],
    lat: Annotated[float | None, typer.Option(help="Where the scan happens.")] = None,
    lon: Annotated[float | None, typer.Option()] = None,
    device: Annotated[str | None, typer.Option(help="Scanning device id (stored hashed).")] = None,
    investigate_unsure: Annotated[
        bool, typer.Option("--investigate/--no-investigate", help="Investigate unsure scans.")
    ] = True,
    data_dir: DataDir = None,
) -> None:
    """Verify a photo of a label against this data directory's keys, log and scan history."""
    if (lat is None) != (lon is None):
        err.print("[red]error:[/] --lat and --lon go together")
        raise typer.Exit(2)

    async def go() -> None:
        async with Container.open(_settings(data_dir)) as c:
            context = ScanContext(datetime.now(UTC), lat, lon, device)
            result = await c.scan(image.read_bytes(), context, investigate=investigate_unsure)
            console.print(_result_panel(result))
            console.print(_evidence_table(result.evidence))
            if result.investigation_id:
                console.print("[dim]investigating...[/]")
                await c.drain()
                row = c.investigations.get(result.investigation_id)
                trace = c.settings.traces_dir / f"{row.id}.jsonl"
                if row.outcome is not None and trace.exists():
                    console.print(_timeline(load_trace(trace)))
                    verdict = Verdict(row.outcome.final_verdict)
                    style = VERDICT_STYLE[verdict]
                    console.print(f"after investigation: [{style}]{verdict.value}[/]")
                else:
                    err.print(f"[red]investigation failed:[/] {escape(row.error or 'unknown')}")

    _run(go())


# --- evals and benchmarks -----------------------------------------------------------------------


@app.command("eval")
def run_eval(
    model: Annotated[
        str, typer.Option(help="'reference' (offline) or a Claude model id.")
    ] = "reference",
    seeds: Annotated[int, typer.Option(min=1, help="Scenarios per kind.")] = 3,
    kinds: Annotated[
        str | None, typer.Option(help="Comma-separated scenario kinds (default: all).")
    ] = None,
    gate: Annotated[
        bool, typer.Option(help="Exit 1 if the regression baseline is breached.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("reports/eval"),
    concurrency: Annotated[int, typer.Option(min=1, help="Scenarios in flight at once.")] = 4,
) -> None:
    """Run the labelled scenario suite, write a report and optionally gate on the baseline."""
    from sigil.evals.__main__ import main

    argv = ["--model", model, "--seeds", str(seeds), "--out", str(out)]
    argv += ["--concurrency", str(concurrency)]
    if kinds:
        argv += ["--kinds", kinds]
    if gate:
        argv.append("--gate")
    raise typer.Exit(main(argv))


@app.command()
def bench(
    concurrency: Annotated[str, typer.Option(help="Comma-separated client counts.")] = "1,8,32",
    duration: Annotated[float, typer.Option(help="Seconds per measurement.")] = 5.0,
    threads: Annotated[int, typer.Option(help="ONNX Runtime intra-op threads (0: default).")] = 0,
    out: Annotated[Path | None, typer.Option(help="Also write the table here.")] = None,
) -> None:
    """Benchmark inference throughput and latency, with and without micro-batching."""
    from sigil.inference.bench import main

    argv = ["--concurrency", concurrency, "--duration", str(duration), "--threads", str(threads)]
    if out is not None:
        argv += ["--out", str(out)]
    main(argv)


# --- traces -------------------------------------------------------------------------------------


@trace_app.command("show")
def trace_show(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Render an investigation trace as a timeline."""
    events = _load_trace(path)
    started = events[0].data
    summary = summarize(events)
    console.print(
        Panel(
            f"[bold]{escape(started.get('model', '?'))}[/]  run {summary.run_id[:12]}\n"
            f"{escape(str(started.get('task', '')))}",
            title="investigation",
            border_style="blue",
        )
    )
    console.print(_timeline(events))
    tools = ", ".join(f"{name} x{n}" for name, n in sorted(summary.tools_used.items()))
    console.print(
        f"status [bold]{summary.status or 'unfinished'}[/] | {summary.steps} model turns | "
        f"{summary.usage.total} tokens | ${summary.cost_usd:.4f} | {summary.wall_s:.2f} s\n"
        f"tools: {tools}\n"
        f"hash chain: {'[green]intact[/]' if verify_chain(events) else '[red]BROKEN[/]'}"
    )


@trace_app.command("verify")
def trace_verify(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Check a trace's hash chain. Exit 1 if any event was altered, reordered or removed."""
    events = _load_trace(path)
    if not verify_chain(events):
        console.print(f"[red]FAILED[/] {path}: the hash chain does not verify")
        raise typer.Exit(1)
    finished = events[-1].type is EventType.RUN_FINISHED
    note = "" if finished else " [yellow](no run_finished event: the trace may be truncated)[/]"
    console.print(f"[green]OK[/] {path}: {len(events)} events, hash chain intact{note}")


@trace_app.command("replay")
def trace_replay(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    data_dir: DataDir = None,
) -> None:
    """Re-run a recorded investigation against today's tools, guardrails and prompt.

    The model's recorded answers are played back, so no LLM is called. Any difference in what the
    tools now return, or in whether the verdict is still accepted, is reported. The traces of
    ``sigil eval`` rebuild their scenario world; any other trace needs its scan in the data dir.
    """
    events = _load_trace(path)
    if not verify_chain(events):
        console.print(f"[red]FAILED[/] {path}: refusing to replay a trace whose chain is broken")
        raise typer.Exit(1)

    async def go() -> bool:
        async with Container.open(_settings(data_dir)) as c:
            deps = await _replay_deps(c, events)
            report = await replay(
                events,
                tools=build_registry(),
                system=SYSTEM_PROMPT,
                guardrails=build_guardrails(deps),
                deps=deps,
            )
        console.print(f"recorded: {report.original_status}   replayed: {report.replay_status}")
        for line in report.divergences:
            console.print(f"[yellow]differs:[/] {escape(line)}")
        verdict = "[green]identical[/]" if report.identical else "[red]diverged[/]"
        console.print(f"replay {verdict}")
        return report.identical

    raise typer.Exit(0 if _run(go()) else 1)


async def _replay_deps(c: Container, events: Sequence[TraceEvent]) -> InvestigationDeps:
    """Rebuild the situation a recorded investigation ran in."""
    run_id = events[0].run_id
    match = re.search(r"scan (\w+)", str(events[0].data.get("task", "")))
    if match is None:
        raise SigilError("the trace does not say which scan it investigated")
    scan_id = match.group(1)
    try:
        event, payload = c.scans.get(scan_id)
        if payload is None:
            raise NotFoundError("the scan has no stored result")
    except NotFoundError:
        pass
    else:  # recaptures cannot be reproduced for a real scan: the camera is gone
        context = ScanContext(event.ts, event.lat, event.lon)
        return InvestigationDeps(c.pipeline, VerificationResult.model_validate(payload), context)

    scenarios = {s.id: s for s in build_scenarios(1 + int(run_id.rsplit("-", 1)[-1]))}
    if run_id not in scenarios:
        raise NotFoundError(f"scan {scan_id} is not in {c.settings.data_dir}, and the trace is not")
    world = await ScenarioWorld.build(scenarios[run_id], c.inference, c.pipeline.policy)
    fresh = await world.verify()
    # The world is deterministic except for the scan id, which the recorded tools output echoes.
    bundle = fresh.evidence.model_copy(update={"scan_id": scan_id})
    result = fresh.model_copy(update={"scan_id": scan_id, "evidence": bundle})
    return world.investigation_deps(result)


# --- witness ------------------------------------------------------------------------------------


@app.command()
def audit(
    url: Annotated[
        str, typer.Option(help="Base URL of a running Sigil server.")
    ] = "http://localhost:8000",
    state: Annotated[
        Path | None, typer.Option(help="Where the witness remembers the last tree head.")
    ] = None,
    log_key: Annotated[
        str | None,
        typer.Option(help="Pin the log's public key (base64url) instead of trusting first use."),
    ] = None,
) -> None:
    """Witness a running server's transparency log.

    Fetches the signed tree head, checks its signature and a consistency proof that it extends
    the last head this witness saw, and remembers it. Exits 1 on any violation, 2 if the server
    cannot be read.
    """
    host = re.sub(r"[^A-Za-z0-9.-]+", "_", url.split("://", 1)[-1])
    state_path = state or Path(".sigil") / "witness" / f"{host}.json"
    try:
        with httpx.Client(base_url=url, timeout=10.0) as http:
            keys = http.get("/v1/keys").raise_for_status().json()
            pinned = log_key or next(k["public_key"] for k in keys if k["purpose"] == "log")
            try:
                auditor = Auditor(_b64url(pinned), state_path)
            except SigilError as exc:
                console.print(f"[red]ALERT[/] {escape(str(exc))}")
                raise typer.Exit(1) from exc
            sth = SignedTreeHead.from_json(http.get("/v1/log/sth").raise_for_status().json())
            previous = auditor.last
            proof = None
            if previous is not None and 0 < previous.tree_size < sth.tree_size:
                body = (
                    http.get(
                        "/v1/log/consistency",
                        params={"first": previous.tree_size, "second": sth.tree_size},
                    )
                    .raise_for_status()
                    .json()
                )
                proof = ConsistencyProof(
                    body["first"], body["second"], [bytes.fromhex(h) for h in body["path"]]
                )
    except (httpx.HTTPError, StopIteration, SigilError, ValueError) as exc:
        err.print(f"[red]error:[/] could not read the log at {url}: {escape(str(exc))}")
        raise typer.Exit(2) from exc

    result = auditor.observe(sth, proof)
    if not result.ok:
        console.print(f"[red]ALERT[/] {escape(result.reason)}")
        raise typer.Exit(1)
    first = (
        "first observation, trusted on first use"
        if previous is None
        else (
            f"consistent with the head of size {previous.tree_size}"
            if proof
            else "unchanged since the last observation"
        )
    )
    console.print(
        f"[green]OK[/] tree size {sth.tree_size}, root {sth.root_hash.hex()[:16]}..., "
        f"signature valid, {first}"
    )
    console.print(f"[dim]state: {state_path}[/]")


def _b64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# --- rendering ----------------------------------------------------------------------------------


def _batch_panel(batch_id: str, sku: str, quality: str, qc: QcStats | None) -> Panel:
    lines = [f"batch [bold]{batch_id}[/]  sku {escape(sku)}  press: {quality}"]
    if qc is not None:
        lines.append(
            f"QC camera: {qc.n} labels scored, CDP score mean {qc.mean:.2f} "
            f"(std {qc.std:.2f}, min {qc.min:.2f})"
        )
    return Panel("\n".join(lines), title="issued", border_style="green")


def _result_panel(result: VerificationResult) -> Panel:
    style = VERDICT_STYLE[result.verdict]
    body = Text.assemble(
        (result.verdict.value, style),
        f"   confidence {result.confidence:.2f}   rule {result.rule}\n",
        ("; ".join(result.reasons) or "", "dim"),
    )
    if result.sku:
        body.append(f"\nlabel says: sku {result.sku!r} (untrusted text read off the label)", "dim")
    timings = result.timings_ms
    body.append(f"\n{timings.get('total', 0):.0f} ms total", "dim")
    return Panel(body, title=f"scan {result.scan_id[:8]}", border_style=style.split()[-1])


def _evidence_table(bundle: EvidenceBundle) -> Table:
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    table.add_column("evidence")
    table.add_column("result")
    table.add_column("detail", overflow="fold")
    marks = {True: "[green]pass[/]", False: "[red]FAIL[/]", None: "[dim]-[/]"}
    for item in bundle.items:
        table.add_row(item.id, marks[item.passed], escape(item.summary))
    return table


def _tool_summary(content: str) -> str:
    """One line describing a tool result, whatever its shape."""
    import json

    try:
        data = json.loads(content)
    except ValueError:
        return content[:100]
    if isinstance(data, dict):
        if isinstance(data.get("summary"), str):
            return str(data["summary"])
        if isinstance(data.get("evidence"), list):
            return f"{len(data['evidence'])} evidence items"
        if "available" in data:
            return "no further photo available" if not data["available"] else str(data)[:100]
    return str(data)[:100]


def _timeline(events: Sequence[TraceEvent]) -> Table:
    """The agent's run, one row per thing it said, called, received or was told."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, title="agent timeline")
    table.add_column("step", justify="right", style="dim")
    table.add_column("")
    table.add_column("", overflow="fold")
    for event in events:
        data = event.data
        step = str(data.get("step", ""))
        match event.type:
            case EventType.MODEL_RESPONSE:
                for block in data["blocks"]:
                    if block.get("type") == "text" and block["text"].strip():
                        table.add_row(step, "[magenta]model[/]", escape(block["text"].strip()))
            case EventType.TOOL_CALL:
                args = ", ".join(f"{k}={v!r}" for k, v in data["input"].items())
                if data["name"] == "submit_verdict":
                    table.add_row(
                        step,
                        "[cyan]call[/]",
                        f"submit_verdict({escape(_verdict_line(data['input']))})",
                    )
                else:
                    table.add_row(step, "[cyan]call[/]", f"{data['name']}({escape(args[:90])})")
            case EventType.TOOL_RESULT if data["name"] != "submit_verdict":
                mark = "[red]error[/]" if data["is_error"] else "[blue]result[/]"
                detail = f"{_tool_summary(data['content'])}  [dim]{data['latency_ms']:.0f} ms[/]"
                table.add_row(step, mark, escape(detail) if data["is_error"] else detail)
            case EventType.GUARDRAIL:
                if data["accepted"]:
                    table.add_row(step, "[green]guardrails[/]", "submission accepted")
                else:
                    reasons = escape("; ".join(data["violations"]))
                    table.add_row(step, "[red]guardrails[/]", f"rejected: {reasons}")
            case EventType.NUDGE | EventType.ERROR:
                table.add_row(step, f"[yellow]{event.type.value}[/]", escape(str(data)[:120]))
            case EventType.RUN_FINISHED:
                table.add_row(
                    "", "[bold]finished[/]",
                    f"{data['status']} after {data['steps']} turns, ${data['cost_usd']:.4f}",
                )  # fmt: skip
    return table


def _verdict_line(submission: dict[str, Any]) -> str:
    cites = ", ".join(submission.get("cited_evidence", []))
    return (
        f"{submission.get('verdict')}, confidence {submission.get('confidence')}, cites [{cites}]"
    )


# --- the narrated demo --------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoCase:
    """One situation in the demo: what is photographed, where, and which verdicts are right."""

    title: str
    unit: int | None  # index into the demo batch; None is a forged label
    city: str
    attack: str | None = None
    first_city: str | None = None  # the item was already scanned here, moments earlier
    expect: str = "COUNTERFEIT"
    accept: frozenset[Verdict] = frozenset({Verdict.COUNTERFEIT, Verdict.SUSPICIOUS})


@dataclass(frozen=True)
class CaseOutcome:
    case: DemoCase
    final: Verdict

    @property
    def ok(self) -> bool:
        return self.final in self.case.accept


DEMO_CASES = (
    DemoCase(
        "A real label, scanned in Mumbai", 0, "Mumbai",
        expect="GENUINE", accept=frozenset({Verdict.GENUINE}),
    ),
    DemoCase("A photocopy of a real label", 1, "Mumbai", attack="photocopy"),
    DemoCase(
        "A high-quality reprint of label #1, scanned in London minutes after the original",
        0, "London", attack="hq_reprint",
    ),
    DemoCase(
        "A forged credential: the attacker's own key, the issuer's key id", None, "Tokyo",
        accept=frozenset({Verdict.COUNTERFEIT}),
    ),
    DemoCase(
        "A real label whose serial also turns up on another continent", 2, "Tokyo",
        first_city="Mumbai", expect="SUSPICIOUS",
        accept=frozenset({Verdict.SUSPICIOUS, Verdict.GENUINE}),
    ),
)  # fmt: skip


@app.command()
def demo(
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Keep the demo's data here (default: temporary)."),
    ] = None,
) -> None:
    """A narrated end-to-end run: issue, photograph, counterfeit, verify, investigate, audit."""
    if data_dir is not None:
        _run(_demo(_settings(data_dir)))
        return
    with tempfile.TemporaryDirectory(prefix="sigil-demo-") as tmp:
        _run(_demo(_settings(Path(tmp))))


def _step(number: int, title: str) -> None:
    console.print()
    console.rule(f"[bold]{number}. {title}[/]", align="left")


async def _demo(settings: Settings) -> None:
    async with Container.open(settings) as c:
        console.print(
            Panel(
                "Sigil binds a product's digital identity to its physical print.\n"
                "Every label carries a signed QR credential, a secret copy-detection pattern (CDP) "
                "and an entry in a public transparency log.\n"
                "[dim]The print/scan channel is simulated; everything after the photo is real.[/]",
                title="sigil demo",
                border_style="blue",
            )
        )

        _step(1, "A manufacturer issues a batch")
        info = await c.create_batch("ACME-WIDGET-PRO", 6, "good")
        console.print(_batch_panel(info.batch_id, info.sku, info.print_quality, info.qc))
        serials = c.issuance.units(info.batch_id)
        console.print(
            "Each credential was signed (Ed25519), appended to the transparency log, and the "
            f"log's tree head re-signed: {c.log.size} entries."
        )

        outcomes = [
            await _case(c, number, case, serials) for number, case in enumerate(DEMO_CASES, start=2)
        ]

        await _audit_log(c, serials[0])

        console.print()
        console.rule("[bold]Summary[/]", align="left")
        table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
        for column in ("case", "expected", "final verdict", ""):
            table.add_column(column)
        for o in outcomes:
            mark = "[green]ok[/]" if o.ok else "[red]WRONG[/]"
            verdict = f"[{VERDICT_STYLE[o.final]}]{o.final.value}[/]"
            table.add_row(o.case.title, o.case.expect, verdict, mark)
        console.print(table)
        accepted = sum(
            1 for o in outcomes if o.case.expect != "GENUINE" and o.final is Verdict.GENUINE
        )
        console.print(f"counterfeits accepted as genuine: [bold]{accepted}[/]")
        console.print("[dim]Next: `sigil serve`, then open http://localhost:8000[/]")


async def _case(c: Container, number: int, case: DemoCase, serials: Sequence[str]) -> CaseOutcome:
    """Photograph and scan one situation, investigate if the policy is unsure, and report."""
    _step(number, case.title)
    serial = None if case.unit is None else serials[case.unit]
    if case.first_city is not None:  # the owner scanned it at home first; now it turns up elsewhere
        first, _, _ = await _scan(c, serial, None, case.first_city)
        style = VERDICT_STYLE[first.verdict]
        console.print(f"first scanned in {case.first_city}: [{style}]{first.verdict.value}[/]")
        console.print(f"[bold]now scanned in {case.city}[/]")
    elif case.attack:
        console.print(f"The attacker photographs the label and prints a [bold]{case.attack}[/].")
    result, photo, context = await _scan(c, serial, case.attack, case.city)
    console.print(_result_panel(result))
    console.print(_evidence_table(result.evidence))

    final = result.verdict
    if result.escalate:
        final = (await _investigate(c, result, context, photo)).final_verdict
    return CaseOutcome(case, final)


async def _scan(
    c: Container, serial: str | None, attack: str | None, city: str
) -> tuple[ScanResponse, Photo, ScanContext]:
    """Take photos until the quality gate is satisfied, as a user told to retake would."""
    lat, lon = CITIES[city]
    for seed in range(DEMO_SEED, DEMO_SEED + MAX_RETAKES):
        if serial is None:
            photo = await _forged_photo(c, seed)
        else:
            request = SimulateRequest.model_validate(
                {"attack": attack, "difficulty": "easy", "seed": seed}
            )
            photo = await c.photograph(serial, request)
        context = ScanContext(datetime.now(UTC), lat, lon, f"demo-phone-{city.lower()}")
        result = await c.scan(photo.jpeg, context, investigate=False)
        if result.verdict is not Verdict.RECAPTURE:
            return result, photo, context
        console.print("[cyan]RECAPTURE[/] the quality gate asked for a retake")
    raise SigilError("could not take a readable photo")


async def _forged_photo(c: Container, seed: int) -> Photo:
    """A label an attacker made with their own key but the issuer's key id and a guessed CDP."""
    rng = np.random.default_rng(seed)
    forged = issue(
        ClaimedKid(Ed25519Signer.generate(), c.keys.issuer.kid),
        sku="ACME-WIDGET-PRO",
        batch="BFORGED",
        serial=rng.bytes(12),
    )
    guess = (rng.random((CDP_CELLS, CDP_CELLS)) < 0.5).astype(np.uint8)
    photo = simulate_capture(render_label(forged.to_qr_payload(), guess), rng, difficulty="easy")
    ok, jpeg = cv2.imencode(".jpg", photo)
    assert ok
    return Photo(jpeg.tobytes(), "")


async def _investigate(
    c: Container, result: ScanResponse, context: ScanContext, photo: Photo
) -> InvestigationOutcome:
    """Run the investigator, show its work, then replay the recorded trace as a regression test."""

    def deps() -> InvestigationDeps:
        # A seeded camera, so a replay is offered exactly the photos the original run was.
        return InvestigationDeps(
            c.pipeline, result, context, c.studio.camera_for(photo.item_id, seed=DEMO_SEED)
        )

    console.print(
        f"\n[yellow]The policy cannot settle this ({result.rule}). "
        f"Escalating to the investigator ({c.settings.investigator_model}).[/]"
    )
    outcome = await investigate(
        deps(),
        model=c.investigator,
        settings=c.settings,
        trace_dir=c.settings.traces_dir,
        run_id=result.scan_id,
    )
    c.scans.update_verdict(result.scan_id, outcome.final_verdict.value)
    assert outcome.trace_path is not None
    events = load_trace(outcome.trace_path)
    console.print(_timeline(events))
    sub = outcome.submission
    if sub is not None:
        console.print(f"[dim]rationale:[/] {escape(sub.rationale)}")
    style = VERDICT_STYLE[outcome.final_verdict]
    console.print(f"final verdict: [{style}]{outcome.final_verdict.value}[/]")

    fresh = deps()
    report = await replay(
        events,
        tools=build_registry(),
        system=SYSTEM_PROMPT,
        guardrails=build_guardrails(fresh),
        deps=fresh,
    )
    console.print(
        f"[dim]trace: hash chain {'intact' if verify_chain(events) else 'BROKEN'}; replayed "
        f"against the current tools and guardrails with no model call: "
        f"{'identical' if report.identical else 'diverged'}[/]"
    )
    return outcome


async def _audit_log(c: Container, serial: str) -> None:
    _step(7, "Anyone can audit the log without trusting the operator")
    credential, _ = c.issuance.unit(serial)
    leaf = leaf_hash(credential.leaf_data())
    sth = c.log.latest_sth()
    proof = c.log.prove_inclusion(leaf, sth.tree_size)
    log_key = c.keys.log.public_raw()
    inclusion_ok = verify_inclusion(
        leaf, proof.leaf_index, proof.tree_size, proof.path, sth.root_hash
    )
    console.print(
        f"signed tree head (size {sth.tree_size}) signature: "
        f"{'[green]valid[/]' if sth.verify(log_key) else '[red]INVALID[/]'}"
    )
    console.print(
        f"inclusion proof for label #1 ({len(proof.path)} hashes, recomputed from the QR text): "
        f"{'[green]verifies[/]' if inclusion_ok else '[red]FAILS[/]'}"
    )
    tampered = [bytes(32), *proof.path[1:]]
    rejected = not verify_inclusion(
        leaf, proof.leaf_index, proof.tree_size, tampered, sth.root_hash
    )
    console.print(f"a tampered proof is {'[green]rejected[/]' if rejected else '[red]ACCEPTED[/]'}")

    await asyncio.to_thread(c.issuer.issue_batch, "ACME-WIDGET-ECO", 3)
    newer = c.log.latest_sth()
    chain = c.log.prove_consistency(sth.tree_size, newer.tree_size)
    consistent = verify_consistency(
        sth.tree_size, newer.tree_size, sth.root_hash, newer.root_hash, chain.path
    )
    with tempfile.TemporaryDirectory() as tmp:
        witness = Auditor(log_key, Path(tmp) / "witness.json")
        witness.observe(sth, None)
        verdict = witness.observe(newer, chain)
    console.print(
        f"after issuing 3 more labels the log grew {sth.tree_size} -> {newer.tree_size}; "
        f"consistency proof {'[green]verifies[/]' if consistent else '[red]FAILS[/]'}; "
        f"an independent witness {'[green]accepts[/]' if verdict.ok else '[red]rejects[/]'} "
        "the new head"
    )
