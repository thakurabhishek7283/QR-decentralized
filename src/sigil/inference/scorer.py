"""ONNX Runtime wrapper for the CDP authenticity CNN."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort

from sigil.inference.preprocess import INPUT_SHAPE

MODEL_FILE = "cdp_cnn.onnx"


class CdpScorer(Protocol):
    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        """Map a ``(N, 2, 212, 212)`` float32 batch to ``(N,)`` logits of P(genuine)."""
        ...


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def packaged_model_path() -> Path:
    """Location of the model shipped inside the ``sigil`` package."""
    return Path(str(resources.files("sigil") / "assets" / MODEL_FILE))


class OnnxCdpScorer:
    """Runs ``cdp_cnn.onnx`` on CPU.

    ``threads`` sets ORT's intra-op thread count (0 lets ORT choose). The session
    is safe to call from several threads; the micro-batcher still funnels calls
    through one so that ORT, not Python threads, supplies the parallelism.
    """

    def __init__(self, model_path: Path | None = None, threads: int = 0) -> None:
        path = Path(model_path) if model_path is not None else packaged_model_path()
        self.metadata: dict[str, Any] = self._load_metadata(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT's intra-op workers busy-wait between calls. The model costs about 0.3 ms per
        # patch, so on a shared 4-core box that spinning starves the event loop and the
        # batcher thread: switching it off made unbatched serving ~4x faster in
        # the serving benchmark in docs/MODEL_CARD.md (about 125 -> 450 req/s).
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def _load_metadata(model: Path) -> dict[str, Any]:
        sidecar = model.with_suffix(".json")
        if not sidecar.exists():
            return {}
        metadata: dict[str, Any] = json.loads(sidecar.read_text())
        expected = metadata.get("sha256")
        if expected and hashlib.sha256(model.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{model.name} does not match the sha256 in {sidecar.name}")
        return metadata

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 4 or batch.shape[1:] != INPUT_SHAPE:
            raise ValueError(f"expected (N, {INPUT_SHAPE}) input, got {batch.shape}")
        x = np.ascontiguousarray(batch, dtype=np.float32)
        (logits,) = self._session.run(["logit"], {"input": x})
        return np.asarray(logits, dtype=np.float32).reshape(-1)
