"""The demo page verifies the log in the browser, so its JavaScript must agree with the Python."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sigil.api.routes.system import INDEX
from sigil.crypto import b45encode
from sigil.ledger import consistency_proof, inclusion_proof, leaf_hash, root_hash

SCRIPT = Path(__file__).with_name("verify_page.mjs")
TREE_SIZE = 24

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def test_page_verifiers_agree_with_the_python_implementation(tmp_path: Path) -> None:
    leaves = [leaf_hash(bytes([i])) for i in range(TREE_SIZE)]
    inclusion, consistency = [], []
    for n in range(1, TREE_SIZE + 1):
        root = root_hash(leaves[:n])
        for i in range(n):
            proof = [p.hex() for p in inclusion_proof(i, leaves[:n])]
            inclusion.append([i, n, leaves[i].hex(), proof, root.hex()])
        for m in range(1, n + 1):
            proof = [p.hex() for p in consistency_proof(m, leaves[:n])]
            consistency.append([m, n, root_hash(leaves[:m]).hex(), root.hex(), proof])
    samples = [b"AB", b"Hello!!", b"base-45", bytes(range(40))]
    vectors = {
        "inclusion": inclusion,
        "consistency": consistency,
        "base45": [[b45encode(s), s.hex()] for s in samples],
    }
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(vectors))

    run = subprocess.run(
        ["node", str(SCRIPT), str(INDEX), str(path)], capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["checked"] == len(inclusion) + len(consistency)
    assert report["failures"] == []
