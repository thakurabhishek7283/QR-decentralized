from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sigil.cdp import Attack, generate_template, simulate_capture
from sigil.errors import CaptureError
from sigil.inference import INPUT_SHAPE, OnnxCdpScorer, make_input, sigmoid
from sigil.inference.scorer import packaged_model_path
from sigil.vision import decode, extract_cdp_patch, render_label

BASE45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


@pytest.fixture(scope="module")
def scorer() -> OnnxCdpScorer:
    return OnnxCdpScorer()


def test_packaged_model_and_metadata_load(scorer: OnnxCdpScorer) -> None:
    assert packaged_model_path().exists()
    assert scorer.metadata["input_shape"] == list(INPUT_SHAPE)
    assert 0.5 < scorer.metadata["metrics"]["auc_test"] <= 1.0
    assert packaged_model_path().stat().st_size < 1_000_000


@pytest.mark.parametrize("n", [1, 3, 17])
def test_logits_shape_dtype_and_batch_independence(scorer: OnnxCdpScorer, n: int) -> None:
    batch = np.random.default_rng(n).standard_normal((n, *INPUT_SHAPE)).astype(np.float32)
    logits = scorer.predict_logits(batch)
    assert logits.shape == (n,)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()
    single = scorer.predict_logits(batch[:1])
    assert single[0] == pytest.approx(logits[0], abs=1e-4)


def test_rejects_wrong_input_shape(scorer: OnnxCdpScorer) -> None:
    with pytest.raises(ValueError, match="expected"):
        scorer.predict_logits(np.zeros((2, 3, 212, 212), np.float32))


def test_checksum_mismatch_is_detected(tmp_path: Path) -> None:
    model = tmp_path / "cdp_cnn.onnx"
    shutil.copy(packaged_model_path(), model)
    meta = json.loads(packaged_model_path().with_suffix(".json").read_text())
    meta["sha256"] = "0" * 64
    model.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sha256"):
        OnnxCdpScorer(model)


def test_sigmoid_is_stable_and_monotone() -> None:
    x = np.array([-30.0, 0.0, 30.0])
    assert sigmoid(x).tolist() == pytest.approx([0.0, 0.5, 1.0], abs=1e-9)


def test_genuine_scores_exceed_photocopy_scores_on_average(scorer: OnnxCdpScorer) -> None:
    genuine: list[np.ndarray] = []
    forged: list[np.ndarray] = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        template = generate_template(rng.bytes(32), rng.bytes(12))
        label = render_label(
            "SG1:" + "".join(BASE45[i] for i in rng.integers(0, 45, 180)), template
        )
        for bucket, attack in ((genuine, None), (forged, Attack.PHOTOCOPY)):
            photo = simulate_capture(label, rng, domain="test", difficulty="easy", attack=attack)
            try:
                bucket.append(
                    make_input(extract_cdp_patch(decode(photo), template).patch, template)
                )
            except CaptureError:
                continue
    assert len(genuine) >= 5
    assert len(forged) >= 4
    assert (
        scorer.predict_logits(np.stack(genuine)).mean()
        > scorer.predict_logits(np.stack(forged)).mean()
    )
