"""Train, evaluate and export the CDP authenticity CNN.

    python -m training.train_cdp --help
    python -m training.train_cdp --quick     # smoke test in a few minutes, writes nothing
    python -m training.train_cdp             # the shipped model (about an hour on 4 cores)

The data is simulated (``sigil.cdp.channel``) and generated through the same
``decode`` / ``extract_cdp_patch`` / ``make_input`` code that serves traffic.
Outputs: ``src/sigil/assets/cdp_cnn.{onnx,json}`` and ``docs/MODEL_CARD.md``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import queue
import threading
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

from sigil.cdp.features import raw_correlation
from sigil.inference.preprocess import INPUT_SHAPE, make_input
from sigil.vision.geometry import PATCH_PX
from training import model_card
from training.dataset import (
    ATTACKS,
    DIFFICULTIES,
    REASONS,
    CdpDataset,
    generation_seconds,
    load_or_generate,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "src" / "sigil" / "assets"
MODEL_VERSION = "1"
PEAK_LR = 2e-3


# --------------------------------------------------------------------------- model


def _block(c_in: int, c_out: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=2, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


WIDTHS = (16, 32, 64, 64)


class CdpNet(nn.Module):
    """Two-channel (capture, expected template) CNN producing one logit for P(genuine).

    Four stride-2 conv blocks shrink 212 px to 14 px, then global average pooling
    makes the verdict a spatial average of local agreement between capture and template.
    """

    def __init__(self, widths: tuple[int, ...] = WIDTHS) -> None:
        super().__init__()
        blocks, c_in = [], INPUT_SHAPE[0]
        for i, c_out in enumerate(widths):
            blocks.append(_block(c_in, c_out, 5 if i == 0 else 3))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(-1)


# --------------------------------------------------------------------------- data

_CENTRE = (PATCH_PX - 1) / 2.0


def _ramp(rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """A random linear illumination ramp, mean 1, peak-to-peak ``2 * amplitude``."""
    angle = rng.uniform(0.0, 2.0 * np.pi)
    axis = np.linspace(-1.0, 1.0, PATCH_PX, dtype=np.float32)
    ramp = axis[None, :] * np.cos(angle) + axis[:, None] * np.sin(angle)
    return (1.0 + amplitude * ramp).astype(np.float32)


def _line_kernel(length: float, angle_deg: float) -> np.ndarray:
    size = int(2 * np.ceil(length) + 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    dx, dy = np.cos(np.radians(angle_deg)) * length / 2, np.sin(np.radians(angle_deg)) * length / 2
    c = size // 2
    cv2.line(kernel, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, 1)
    return kernel / kernel.sum()


def augment(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Label-preserving perturbations of an aligned capture patch.

    The simulator only produces captures from a limited set of cameras and printers,
    so the training patches are further degraded the way a *worse* phone, a rougher
    alignment or a harsher light would degrade them: sub-pixel shift, a small rotation
    and scale error, defocus or motion blur, sensor noise, a lighting ramp and a gamma
    change. The ranges are physical guesses made before looking at any test result and
    deliberately a little wider than the train-domain capture ranges; they are not
    fitted to the test domain. Class labels do not change: a blurrier genuine capture
    is still genuine, a blurrier copy is still a copy.
    """
    angle = rng.normal(0.0, 0.4)
    scale = 1.0 + rng.normal(0.0, 0.008)
    warp = cv2.getRotationMatrix2D((_CENTRE, _CENTRE), angle, scale)
    warp[:, 2] += rng.uniform(-1.5, 1.5, 2)
    patch = cv2.warpAffine(
        patch, warp, (PATCH_PX, PATCH_PX), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    roll = rng.random()
    if roll < 0.55:
        patch = cv2.GaussianBlur(patch, (0, 0), float(rng.uniform(0.3, 1.2)))
    elif roll < 0.7:
        patch = cv2.filter2D(
            patch, -1, _line_kernel(float(rng.uniform(1.5, 4.0)), float(rng.uniform(0, 180)))
        )
    if rng.random() < 0.5:
        patch = patch * _ramp(rng, float(rng.uniform(0.02, 0.15)))
    patch = np.clip(patch, 0.0, 1.0) ** float(rng.uniform(0.8, 1.3))
    if rng.random() < 0.6:
        patch = patch + rng.normal(0.0, rng.uniform(0.005, 0.05), patch.shape).astype(np.float32)
    return patch.astype(np.float32)


def build_inputs(
    data: CdpDataset, rows: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Model inputs for ``rows`` of the decoded samples; augmented when ``rng`` is given."""
    batch = np.empty((len(rows), *INPUT_SHAPE), dtype=np.float32)
    for k, i in enumerate(rows):
        patch = data.patch[i].astype(np.float32) / 255.0
        batch[k] = make_input(augment(patch, rng) if rng is not None else patch, data.template[i])
    return batch


def predict(model: nn.Module, data: CdpDataset, batch_size: int = 256) -> np.ndarray:
    """Logits for every decoded sample of ``data``."""
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(data.patch), batch_size):
            rows = np.arange(start, min(start + batch_size, len(data.patch)))
            out.append(model(torch.from_numpy(build_inputs(data, rows))).numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def _prefetch(
    train_set: CdpDataset, order: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterator[tuple[np.ndarray, torch.Tensor]]:
    """Build augmented batches on a worker thread so the CPU is busy during back-prop.

    OpenCV and NumPy release the GIL, so the worker overlaps with PyTorch's compute.
    """
    batches: queue.Queue[tuple[np.ndarray, torch.Tensor] | BaseException | None] = queue.Queue(3)

    def work() -> None:
        try:
            for step in range(len(order) // batch_size):
                rows = order[step * batch_size : (step + 1) * batch_size]
                batches.put((rows, torch.from_numpy(build_inputs(train_set, rows, rng))))
            batches.put(None)
        except BaseException as exc:  # surfaced on the consumer side
            batches.put(exc)

    threading.Thread(target=work, daemon=True).start()
    while (item := batches.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


def train(
    train_set: CdpDataset,
    val_set: CdpDataset,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    widths: tuple[int, ...] = WIDTHS,
    checkpoint: Path | None = None,
) -> tuple[CdpNet, list[dict[str, float]]]:
    """Train and return the model with the best validation AUC (not necessarily the last).

    With ``checkpoint`` the state is saved after every epoch and a later call resumes
    from it, so an interrupted run (the full run takes about an hour) loses at most one
    epoch. The augmentation stream of epoch ``e`` depends only on ``(seed, e)``.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = CdpNet(widths)
    optimiser = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=1e-4)
    n = len(train_set.patch)
    steps_per_epoch = n // batch_size
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=PEAK_LR, total_steps=epochs * steps_per_epoch, pct_start=0.2
    )
    labels = torch.from_numpy(train_set.decoded("label").astype(np.float32))
    history: list[dict[str, float]] = []
    best_auc, best_state = -1.0, copy.deepcopy(model.state_dict())
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False)
        if saved["epochs"] == epochs and saved["n"] == n:
            model.load_state_dict(saved["model"])
            optimiser.load_state_dict(saved["optimiser"])
            schedule.load_state_dict(saved["schedule"])
            history, best_auc, best_state = saved["history"], saved["best_auc"], saved["best_state"]
            print(f"resuming after epoch {len(history)}", flush=True)
    for epoch in range(len(history) + 1, epochs + 1):
        model.train()
        rng = np.random.default_rng([seed, epoch])
        order = rng.permutation(n)
        started, total = time.perf_counter(), 0.0
        for rows, x in _prefetch(train_set, order, batch_size, rng):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), labels[rows])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        val_auc = float(roc_auc_score(val_set.decoded("label"), predict(model, val_set)))
        history.append({"epoch": epoch, "loss": total / steps_per_epoch, "val_auc": val_auc})
        if val_auc > best_auc:
            best_auc, best_state = val_auc, copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch:2d}/{epochs}  loss {history[-1]['loss']:.4f}  val AUC {val_auc:.4f}  "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        if checkpoint is not None:
            torch.save(
                {
                    "epochs": epochs, "n": n, "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(), "schedule": schedule.state_dict(),
                    "history": history, "best_auc": best_auc, "best_state": best_state,
                },
                checkpoint,
            )  # fmt: skip
    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda h: h["val_auc"])["epoch"]
    print(f"best validation AUC {best_auc:.4f} at epoch {best_epoch:.0f}", flush=True)
    return model, history


# --------------------------------------------------------------------------- evaluation


def auc_eer(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """ROC AUC and equal error rate (genuine = positive class)."""
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fnr - fpr)))
    return float(roc_auc_score(y, score)), float((fpr[k] + fnr[k]) / 2)


def raw_correlations(data: CdpDataset) -> np.ndarray:
    """Pearson r of every decoded sample (the classical baseline's raw score)."""
    return np.array(
        [
            raw_correlation(data.patch[i].astype(np.float32) / 255.0, data.template[i])
            for i in range(len(data.patch))
        ]
    )


def fit_correlation_calibration(data: CdpDataset) -> tuple[float, float]:
    """Logistic fit of raw Pearson r -> P(genuine) on the training split."""
    fit = LogisticRegression(C=1e4).fit(raw_correlations(data)[:, None], data.decoded("label"))
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def _stats(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"n": 0, **{k: float("nan") for k in ("mean", "p05", "p25", "p50", "p75", "p95")}}
    p05, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n": len(values), "mean": float(values.mean()), "p05": float(p05),
        "p25": float(p25), "p50": float(p50), "p75": float(p75), "p95": float(p95),
    }  # fmt: skip


# Policy operating points are chosen on the validation split only: at most 1 % of genuine
# captures may fall at or below ``cdp_counterfeit_max`` and at most 0.5 % of counterfeits
# may reach ``cdp_genuine_min``. The test split then reports what those thresholds do.
MAX_GENUINE_REJECTED = 0.01
MAX_COUNTERFEIT_ACCEPTED = 0.005
DEFAULT_THRESHOLDS = {"cdp_genuine_min": 0.80, "cdp_counterfeit_max": 0.20}  # LLD 7.2 defaults


def choose_thresholds(cnn: np.ndarray, genuine: np.ndarray) -> dict[str, float]:
    """Pick ``cdp_genuine_min`` / ``cdp_counterfeit_max`` from validation scores.

    Both are rounded outwards to three decimals, so the stated error budget still holds
    after rounding.
    """
    g, c = cnn[genuine].astype(np.float64), cnn[~genuine].astype(np.float64)
    counterfeit_max = np.floor(np.quantile(g, MAX_GENUINE_REJECTED) * 1000) / 1000
    genuine_min = np.ceil(np.quantile(c, 1.0 - MAX_COUNTERFEIT_ACCEPTED) * 1000) / 1000
    return {
        "cdp_genuine_min": float(min(genuine_min, 0.999)),
        "cdp_counterfeit_max": float(max(counterfeit_max, 0.001)),
    }


def policy_rates(cnn: np.ndarray, thresholds: Mapping[str, float]) -> dict[str, float]:
    """Share of ``cnn`` scores accepted (>= genuine_min), rejected (<= counterfeit_max)
    or escalated (in between)."""
    accepted = float(np.mean(cnn >= thresholds["cdp_genuine_min"]))
    rejected = float(np.mean(cnn <= thresholds["cdp_counterfeit_max"]))
    return {
        "accepted": accepted, "escalated": 1.0 - accepted - rejected, "rejected": rejected,
        "n": len(cnn),
    }  # fmt: skip


def policy_table(
    data: CdpDataset, cnn: np.ndarray, thresholds: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """Accepted / escalated / rejected rates per population of a split."""
    genuine, worn = data.decoded("label") == 1, data.decoded("worn")
    attack, difficulty = data.decoded("attack"), data.decoded("difficulty")
    groups = {
        "genuine": genuine,
        "genuine, good printer": genuine & ~worn,
        "genuine, worn printer": genuine & worn,
        **{f"genuine, {name}": genuine & (difficulty == k) for k, name in enumerate(DIFFICULTIES)},
        "counterfeit": ~genuine,
        **{atk.value: attack == k for k, atk in enumerate(ATTACKS)},
    }
    return {name: policy_rates(cnn[mask], thresholds) for name, mask in groups.items()}


def evaluate(
    data: CdpDataset, logits: np.ndarray, corr_raw: np.ndarray, calibration: tuple[float, float]
) -> dict[str, Any]:
    """Held-out metrics: AUC/EER overall, per attack and per difficulty, for both scorers."""
    y = data.decoded("label")
    attack = data.decoded("attack")
    difficulty = data.decoded("difficulty")
    worn = data.decoded("worn")
    quality_ok = data.decoded("quality_ok")
    cnn = 1.0 / (1.0 + np.exp(-logits))
    a, b = calibration
    corr = 1.0 / (1.0 + np.exp(-(a * corr_raw + b)))
    genuine = y == 1

    def pair(mask: np.ndarray) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for name, scores in (("cnn", cnn), ("corr", corr)):
            auc, eer = auc_eer(y[mask], scores[mask])
            row[name] = {"auc": auc, "eer": eer, "n": int(mask.sum())}
        return row

    return {
        "overall": pair(np.ones_like(genuine)),
        "quality_gated": pair(quality_ok),
        "by_attack": {atk.value: pair(genuine | (attack == k)) for k, atk in enumerate(ATTACKS)},
        "by_difficulty": {name: pair(difficulty == k) for k, name in enumerate(DIFFICULTIES)},
        "genuine_scores": {
            "good": {"cnn": _stats(cnn[genuine & ~worn]), "corr": _stats(corr[genuine & ~worn])},
            "worn": {"cnn": _stats(cnn[genuine & worn]), "corr": _stats(corr[genuine & worn])},
        },
        "counterfeit_scores": {
            **{atk.value: {"cnn": _stats(cnn[attack == k])} for k, atk in enumerate(ATTACKS)},
            "all": {"cnn": _stats(cnn[~genuine])},
        },
        "sweep": threshold_sweep(cnn, genuine),
    }


def threshold_sweep(cnn: np.ndarray, genuine: np.ndarray) -> dict[str, Any]:
    """Descriptive trade-off tables for a range of candidate thresholds (not used to choose)."""
    g, c = cnn[genuine], cnn[~genuine]
    return {
        "genuine_min": [
            {"threshold": t, "genuine_pass": float(np.mean(g >= t)),
             "counterfeit_pass": float(np.mean(c >= t))}
            for t in (0.5, 0.8, 0.9, 0.95, 0.99)
        ],
        "counterfeit_max": [
            {"threshold": t, "counterfeit_caught": float(np.mean(c <= t)),
             "genuine_rejected": float(np.mean(g <= t))}
            for t in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
        ],
    }  # fmt: skip


def failure_rates(data: CdpDataset) -> dict[str, Any]:
    """Share of simulated captures that produced no patch, by class and difficulty."""
    result: dict[str, Any] = {}
    for kind, mask in (("genuine", data.label == 1), ("counterfeit", data.label == 0)):
        per: dict[str, Any] = {}
        for k, name in enumerate(DIFFICULTIES):
            sel = mask & (data.difficulty == k)
            reasons = {REASONS[r]: int(np.sum(sel & (data.reason == r))) for r in range(1, 5)}
            per[name] = {
                "rate": float(np.mean(data.reason[sel] != 0)) if sel.any() else 0.0,
                "n": int(sel.sum()),
                "reasons": {r: c for r, c in reasons.items() if c},
            }
        per["all"] = {"rate": float(np.mean(data.reason[mask] != 0)), "n": int(mask.sum())}
        result[kind] = per
    return result


# --------------------------------------------------------------------------- export


def export_onnx(model: CdpNet, path: Path, sample: np.ndarray) -> float:
    """Write the ONNX file (opset 17, dynamic batch); return max |ONNX Runtime - torch|."""
    model.eval()
    torch.onnx.export(
        model, torch.from_numpy(sample[:1]), str(path), input_names=["input"],
        output_names=["logit"], dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17, dynamo=False,
    )  # fmt: skip
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (got,) = session.run(["logit"], {"input": sample})
    with torch.inference_mode():
        want = model(torch.from_numpy(sample)).numpy()
    return float(np.max(np.abs(got - want)))


def write_assets(
    model: CdpNet, test: CdpDataset, run: dict[str, Any], calibration: tuple[float, float]
) -> tuple[dict[str, Any], Path]:
    """Export the model and write ``cdp_cnn.json``; returns (metadata, onnx path)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    onnx_path = ASSETS / "cdp_cnn.onnx"
    max_diff = export_onnx(
        model, onnx_path, build_inputs(test, np.arange(min(16, len(test.patch))))
    )
    print(f"ONNX vs torch max |diff| = {max_diff:.2e}")
    if max_diff > 1e-4:
        raise SystemExit("ONNX output differs from torch by more than 1e-4")
    metrics = run["metrics"]
    metadata = {
        "version": MODEL_VERSION,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "input_shape": list(INPUT_SHAPE),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": {
            "auc_test": metrics["overall"]["cnn"]["auc"],
            "auc_test_by_attack": {k: v["cnn"]["auc"] for k, v in metrics["by_attack"].items()},
            "eer_test": metrics["overall"]["cnn"]["eer"],
            "baseline_corr_auc_test": metrics["overall"]["corr"]["auc"],
        },
        "corr_calibration": {"a": calibration[0], "b": calibration[1]},
        "decode_failure_rate": metrics["decode_failure_rate"]["test"]["genuine"]["all"]["rate"],
        # Chosen on the validation split (<= 1 % of genuine captures at or below
        # cdp_counterfeit_max, <= 0.5 % of counterfeits at or above cdp_genuine_min);
        # the test-split consequences are in docs/MODEL_CARD.md.
        "operating_points": run["thresholds"],
        # CNN score distributions of genuine labels by printer condition (test domain), for
        # the investigator's worn-printer reasoning.
        "genuine_score_stats": {
            kind: metrics["genuine_scores"][kind]["cnn"] for kind in ("good", "worn")
        },
        "architecture": {"widths": run["widths"], "params": run["params"]},
        "data": {"sizes": run["sizes"], "seed": int(run["args"]["seed"])},
    }
    (ASSETS / "cdp_cnn.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata, onnx_path


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--quick", action="store_true", help="tiny sizes, 3 epochs, write nothing")
    p.add_argument("--n-train", type=int, default=18000)
    p.add_argument("--n-val", type=int, default=2500)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=31337)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "training" / "data")
    p.add_argument("--run-dir", type=Path, default=ROOT / "training" / "runs")
    p.add_argument("--widths", default="16,32,64,64", help="channels of the four conv blocks")
    p.add_argument(
        "--bench-md",
        type=Path,
        default=ROOT / "training" / "serving_bench.md",
        help="serving benchmark table to embed in the card (python -m sigil.inference.bench)",
    )
    p.add_argument("--fresh", action="store_true", help="ignore a checkpoint from an earlier run")
    p.add_argument(
        "--card-only", action="store_true", help="re-render the model card from the last run"
    )
    p.add_argument("--write-assets", action="store_true", help="write assets even with --quick")
    args = p.parse_args(argv)
    if args.quick:
        args.n_train, args.n_val, args.n_test, args.epochs = 600, 200, 400, 3
    return args


def render_card(run: dict[str, Any], metadata: dict[str, Any], bench_md: Path | None) -> None:
    bench = bench_md.read_text() if bench_md and bench_md.exists() else None
    card = model_card.render(run, metadata, ASSETS / "cdp_cnn.onnx", bench)
    (ROOT / "docs" / "MODEL_CARD.md").write_text(card)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_file = args.run_dir / "last_run.json"
    if args.card_only:
        metadata = json.loads((ASSETS / "cdp_cnn.json").read_text())
        render_card(json.loads(run_file.read_text()), metadata, args.bench_md)
        print("re-rendered docs/MODEL_CARD.md")
        return

    started = time.perf_counter()
    print("generating data ...", flush=True)
    sets = {
        split: load_or_generate(split, n, args.seed, args.cache_dir, args.workers)
        for split, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test))
    }
    data_seconds = sum(
        generation_seconds(split, len(d.label), args.seed, args.cache_dir)
        or (time.perf_counter() - started) / len(sets)
        for split, d in sets.items()
    )
    calibration = fit_correlation_calibration(sets["train"])

    widths = tuple(int(w) for w in args.widths.split(","))
    params = sum(p.numel() for p in CdpNet(widths).parameters())
    print(f"training CdpNet{widths} ({params:,} parameters) ...", flush=True)
    t0 = time.perf_counter()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        None
        if args.quick
        else args.run_dir / f"ckpt_s{args.seed}_w{'-'.join(map(str, widths))}_e{args.epochs}.pt"
    )
    if checkpoint is not None and args.fresh:
        checkpoint.unlink(missing_ok=True)
    model, history = train(
        sets["train"], sets["val"], epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, widths=widths, checkpoint=checkpoint,
    )  # fmt: skip
    train_seconds = time.perf_counter() - t0

    val, test = sets["val"], sets["test"]
    val_logits, test_logits = predict(model, val), predict(model, test)
    metrics = evaluate(test, test_logits, raw_correlations(test), calibration)
    metrics["decode_failure_rate"] = {s: failure_rates(d) for s, d in sets.items()}
    val_metrics = evaluate(val, val_logits, raw_correlations(val), calibration)
    val_cnn = 1.0 / (1.0 + np.exp(-val_logits))
    test_cnn = 1.0 / (1.0 + np.exp(-test_logits))
    thresholds = choose_thresholds(val_cnn, val.decoded("label") == 1)
    policy = {
        "val": policy_table(val, val_cnn, thresholds),
        "test": policy_table(test, test_cnn, thresholds),
        "test_default": policy_table(test, test_cnn, DEFAULT_THRESHOLDS),
    }
    print(
        f"test AUC  cnn {metrics['overall']['cnn']['auc']:.4f}   "
        f"correlation {metrics['overall']['corr']['auc']:.4f}"
    )
    for name, row in metrics["by_attack"].items():
        print(f"   {name:11s} cnn {row['cnn']['auc']:.4f}  corr {row['corr']['auc']:.4f}")
    print(f"operating points (from validation): {thresholds}")
    for name, row in policy["test"].items():
        print(
            f"   test {name:24s} accept {row['accepted']:.3f}  escalate {row['escalated']:.3f}  "
            f"reject {row['rejected']:.3f}"
        )

    run = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "sizes": {
            s: {"requested": len(d.label), "decoded": int(d.ok.sum())} for s, d in sets.items()
        },
        "history": history,
        "metrics": metrics,
        "metrics_val": val_metrics,
        "thresholds": thresholds,
        "budget": {
            "genuine_rejected": MAX_GENUINE_REJECTED,
            "counterfeit_accepted": MAX_COUNTERFEIT_ACCEPTED,
        },
        "policy": policy,
        "calibration": calibration,
        "seconds": {
            "data": data_seconds,
            "train": train_seconds,
            "total": time.perf_counter() - started,
        },
        "params": params,
        "widths": list(widths),
        "torch": torch.__version__,
    }
    run_file.write_text(json.dumps(run, indent=2))
    torch.save(model.state_dict(), args.run_dir / "cdp_cnn.pt")

    if args.quick and not args.write_assets:
        print("quick mode: assets and model card not written")
        return
    metadata, onnx_path = write_assets(model, test, run, calibration)
    render_card(run, metadata, args.bench_md)
    size_kib = onnx_path.stat().st_size / 1024
    print(f"wrote {onnx_path.name} ({size_kib:.0f} KiB), cdp_cnn.json and MODEL_CARD.md")


if __name__ == "__main__":
    main()
