"""The command line: argument plumbing for the wrapped tools, and a few real, light commands."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from sigil.api.container import Container
from sigil.api.schemas import SimulateRequest
from sigil.cli import app
from sigil.config import Settings
from sigil.crypto import Ed25519Signer
from sigil.harness import AgentRunner, JsonlTraceWriter, TraceRecorder
from sigil.ledger import TransparencyLog
from sigil.storage import Database
from sigil.verify import ScanContext, Verdict
from tests.harness.helpers import FakeModel, call, make_registry, reply

runner = CliRunner()
REAL_CLIENT = httpx.Client


def trace_file(path: Path) -> Path:
    """A genuine recorded agent run: one tool call, then a verdict."""
    model = FakeModel([reply(call("c1", "echo", value=3)), reply(call("c2", "submit", value=3))])
    agent = AgentRunner(
        model=model,
        tools=make_registry(),
        system="You are a test agent.",
        trace=TraceRecorder("run-1", [JsonlTraceWriter(path)]),
    )
    asyncio.run(agent.run("Investigate scan abc123."))
    return path


def test_a_recorded_trace_verifies_and_renders(tmp_path: Path) -> None:
    path = trace_file(tmp_path / "run.jsonl")

    verified = runner.invoke(app, ["trace", "verify", str(path)])
    assert verified.exit_code == 0
    assert "hash chain intact" in verified.output

    shown = runner.invoke(app, ["trace", "show", str(path)])
    assert shown.exit_code == 0
    assert "echo(value=3)" in shown.output
    assert "submission accepted" in shown.output
    assert "intact" in shown.output


def test_a_tampered_trace_fails_verification(tmp_path: Path) -> None:
    path = trace_file(tmp_path / "run.jsonl")
    path.write_text(path.read_text().replace('"value":3', '"value":4', 1))
    result = runner.invoke(app, ["trace", "verify", str(path)])
    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_a_file_that_is_not_a_trace_fails_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "junk.jsonl"
    path.write_text("not json\n")
    result = runner.invoke(app, ["trace", "verify", str(path)])
    assert result.exit_code == 1
    assert "not a valid trace" in result.output


def test_replay_refuses_a_trace_whose_chain_is_broken(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(trace_file(tmp_path / "t.jsonl").read_text().replace("scan abc123", "nothing"))
    result = runner.invoke(app, ["trace", "replay", str(path), "--data-dir", str(tmp_path / "d")])
    assert result.exit_code == 1
    assert "chain is broken" in result.output


def test_issue_simulate_and_scan_round_trip(tmp_path: Path) -> None:
    data = ["--data-dir", str(tmp_path / "data")]
    labels = tmp_path / "labels"

    issued = runner.invoke(
        app, ["issue", "--sku", "CLI-TEST", "--count", "2", "--labels", str(labels), *data]
    )
    assert issued.exit_code == 0, issued.output
    assert "CLI-TEST" in issued.output
    serial = sorted(labels.glob("*.png"))[0].stem
    assert len(serial) == 24

    photo = tmp_path / "photo.jpg"
    shot = runner.invoke(
        app, ["simulate", serial, "-o", str(photo), "--difficulty", "easy", "--seed", "1", *data]
    )
    assert shot.exit_code == 0, shot.output
    assert photo.read_bytes()[:2] == b"\xff\xd8"  # JPEG

    scanned = runner.invoke(app, ["scan", str(photo), "--no-investigate", *data])
    assert scanned.exit_code == 0, scanned.output
    assert any(v in scanned.output for v in ("GENUINE", "SUSPICIOUS", "RECAPTURE"))
    assert "COUNTERFEIT" not in scanned.output


def test_simulating_an_unknown_serial_is_a_clean_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["simulate", "0" * 24, "-o", str(tmp_path / "x.jpg"), "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "never issued" in result.output


def test_bad_simulation_options_are_rejected(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["simulate", "0" * 24, "--attack", "voodoo", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 2


def test_eval_forwards_its_options_to_the_suite_and_returns_its_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Sequence[str]] = []

    def fake_main(argv: Sequence[str] | None = None) -> int:
        seen.append(list(argv or []))
        return 3

    monkeypatch.setattr("sigil.evals.__main__.main", fake_main)
    result = runner.invoke(
        app, ["eval", "--model", "reference", "--seeds", "3", "--gate", "--out", "reports/eval"]
    )
    assert result.exit_code == 3
    argv = seen[0]
    for expected in (["--model", "reference"], ["--seeds", "3"], ["--out", "reports/eval"]):
        assert argv[argv.index(expected[0]) + 1] == expected[1]
    assert "--gate" in argv
    assert "--kinds" not in argv


def test_bench_forwards_its_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Sequence[str]] = []
    monkeypatch.setattr(
        "sigil.inference.bench.main", lambda argv=None: seen.append(list(argv or []))
    )
    result = runner.invoke(
        app, ["bench", "--concurrency", "1,8,32", "--duration", "2", "--out", "reports/bench.md"]
    )
    assert result.exit_code == 0
    assert seen[0][:4] == ["--concurrency", "1,8,32", "--duration", "2.0"]
    assert seen[0][-2:] == ["--out", "reports/bench.md"]


# --- audit: a witness against a log served over (mock) HTTP -------------------------------------


def serve_log(
    log: TransparencyLog, signer: Ed25519Signer
) -> Callable[[httpx.Request], httpx.Response]:
    key = base64.urlsafe_b64encode(signer.public_raw()).rstrip(b"=").decode()

    def handle(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        match request.url.path:
            case "/v1/keys":
                return httpx.Response(200, json=[{"purpose": "log", "public_key": key}])
            case "/v1/log/sth":
                return httpx.Response(200, json=log.latest_sth().to_json())
            case "/v1/log/consistency":
                proof = log.prove_consistency(int(params["first"]), int(params["second"]))
                return httpx.Response(
                    200, json={"first": proof.first, "second": proof.second,
                               "path": [h.hex() for h in proof.path]},
                )  # fmt: skip
        return httpx.Response(404)

    return handle


def audit(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response], state: Path
) -> tuple[int, str]:
    monkeypatch.setattr(
        "sigil.cli.httpx.Client",
        lambda **kw: REAL_CLIENT(**kw, transport=httpx.MockTransport(handler)),
    )
    result = runner.invoke(app, ["audit", "--url", "http://log.test", "--state", str(state)])
    return result.exit_code, " ".join(result.output.split())  # undo terminal wrapping


def test_a_witness_follows_a_growing_log_and_catches_a_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signer = Ed25519Signer.generate()
    log = TransparencyLog(Database(":memory:"), signer)
    log.append_many([b"a", b"b", b"c"])
    state = tmp_path / "witness.json"

    code, output = audit(monkeypatch, serve_log(log, signer), state)
    assert code == 0, output
    assert "first observation" in output
    assert json.loads(state.read_text())["tree_size"] == 3

    log.append_many([b"d", b"e"])
    log.publish()
    code, output = audit(monkeypatch, serve_log(log, signer), state)
    assert code == 0, output
    assert "consistent with the head of size 3" in output

    # The operator forks history: same key, a bigger tree that does not contain what was seen.
    forked = TransparencyLog(Database(":memory:"), signer)
    forked.append_many([b"x", b"y", b"z", b"w", b"v", b"u"])
    code, output = audit(monkeypatch, serve_log(forked, signer), state)
    assert code == 1
    assert "ALERT" in output
    assert "rewritten" in output


def test_an_unreachable_log_is_an_error_not_an_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    code, output = audit(monkeypatch, down, tmp_path / "w.json")
    assert code == 2
    assert "could not read the log" in output


def test_an_investigation_from_the_api_replays_identically(tmp_path: Path) -> None:
    """Scan a serial from two continents, let the investigator run, then replay its trace."""
    settings = Settings(data_dir=tmp_path, qc_samples_per_batch=2)

    async def investigate_a_clone() -> Path:
        async with Container.open(settings) as c:
            info = await c.create_batch("REPLAY-TEST", 1, "good")
            serial = c.issuance.units(info.batch_id)[0]
            for seed in range(12):  # retake until the quality gate is satisfied
                photo = await c.photograph(serial, SimulateRequest(difficulty="easy", seed=seed))
                home = ScanContext(datetime.now(UTC), 19.08, 72.88, "phone-a")
                if (
                    await c.scan(photo.jpeg, home, investigate=False)
                ).verdict is not Verdict.RECAPTURE:
                    break
            abroad = ScanContext(datetime.now(UTC), 51.51, -0.13, "phone-b")
            scan = await c.scan(photo.jpeg, abroad)
            assert scan.investigation_id, scan.verdict
            await c.drain()
            return c.settings.traces_dir / f"{scan.investigation_id}.jsonl"

    trace = asyncio.run(investigate_a_clone())
    result = runner.invoke(app, ["trace", "replay", str(trace), "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "replay identical" in result.output
