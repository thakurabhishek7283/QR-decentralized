"""The command line: argument plumbing for the wrapped tools, and a few real, light commands."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner, Result

from sigil import cli
from sigil.cli import app
from sigil.crypto import Ed25519Signer
from sigil.ledger import TransparencyLog
from sigil.storage import Database

runner = CliRunner()
REAL_CLIENT = httpx.Client


@pytest.fixture(autouse=True)
def fixed_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich sizes its console from the terminal at import; pin it so output doesn't vary."""
    monkeypatch.setattr(cli.console, "width", 120)


def text(result: Result) -> str:
    """Command output with rich's terminal wrapping undone, so asserts don't depend on width."""
    return " ".join(result.output.split())


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

    scanned = runner.invoke(app, ["scan", str(photo), *data])
    assert scanned.exit_code == 0, scanned.output
    assert any(v in scanned.output for v in ("GENUINE", "SUSPICIOUS", "RECAPTURE"))
    assert "COUNTERFEIT" not in scanned.output


def test_simulating_an_unknown_serial_is_a_clean_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["simulate", "0" * 24, "-o", str(tmp_path / "x.jpg"), "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "never issued" in text(result)


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
    result = runner.invoke(app, ["eval", "--seeds", "3", "--gate", "--out", "reports/eval"])
    assert result.exit_code == 3
    argv = seen[0]
    for expected in (["--seeds", "3"], ["--out", str(Path("reports/eval"))]):
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
    return result.exit_code, text(result)


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
