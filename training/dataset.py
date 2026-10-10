"""Simulated CDP dataset: full print -> (attack) -> capture -> decode -> align pipeline.

Every sample is a pure function of ``(base_seed, split, index)``, so a dataset is
reproducible regardless of the number of worker processes. Samples are produced
by the same ``sigil.vision`` functions that serve production traffic
(``decode`` and ``extract_cdp_patch``); captures that fail to decode are kept in the
bookkeeping (so failure rates can be reported) but carry no patch.

Splits
------
``train`` / ``val``  channel parameters from the *train* domain (disjoint seeds)
``test``             channel parameters from the *test* domain (wider blur, lower
                     JPEG quality, different gamma, rougher printers and different
                     attacker equipment) with its own serials and keys.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sigil.cdp import Attack, generate_template, simulate_capture
from sigil.cdp.channel import Difficulty, Domain, Quality
from sigil.errors import CaptureError
from sigil.vision import decode, extract_cdp_patch, render_label
from sigil.vision.geometry import CDP_CELLS, PATCH_PX

BASE45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
ATTACKS: tuple[Attack, ...] = tuple(Attack)
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "normal", "hard")
DIFFICULTY_WEIGHTS = (0.3, 0.5, 0.2)
WORN_FRACTION = 0.15  # share of printed originals made on a worn printer
REASONS = ("ok", "no_qr", "low_quality", "align_failed", "bad_image")
SPLITS: dict[str, tuple[Domain, int]] = {
    "train": ("train", 0),
    "val": ("train", 1),
    "test": ("test", 2),
}


@dataclass(frozen=True)
class Spec:
    """What to simulate for one sample."""

    split: str
    index: int
    seed: int
    attack: Attack | None  # None = genuine
    quality: Quality
    difficulty: Difficulty

    @property
    def domain(self) -> Domain:
        return SPLITS[self.split][0]


def make_spec(split: str, index: int, seed: int) -> Spec:
    """Deterministic sample plan: classes alternate, negatives cycle through the attacks."""
    rng = np.random.default_rng([seed, SPLITS[split][1], index, 0])
    attack = None if index % 2 == 0 else ATTACKS[(index // 2) % len(ATTACKS)]
    quality: Quality = "worn" if rng.random() < WORN_FRACTION else "good"
    difficulty = DIFFICULTIES[int(rng.choice(len(DIFFICULTIES), p=DIFFICULTY_WEIGHTS))]
    return Spec(split, index, seed, attack, quality, difficulty)


@dataclass(frozen=True)
class Outcome:
    spec: Spec
    reason: str  # one of REASONS
    patch: np.ndarray | None = None  # uint8 (PATCH_PX, PATCH_PX)
    template: np.ndarray | None = None  # uint8 (CDP_CELLS, CDP_CELLS)
    alignment: float = 0.0
    sharpness: float = 0.0
    quality_ok: bool = False


def random_payload(rng: np.random.Generator) -> str:
    """A stand-in for ``SG1:`` + base45 credential of realistic length (150-206 chars)."""
    body = rng.integers(0, len(BASE45), int(rng.integers(146, 203)))
    return "SG1:" + "".join(BASE45[i] for i in body)


def render_sample(spec: Spec) -> Outcome:
    """Run one sample through the complete pipeline."""
    rng = np.random.default_rng([spec.seed, SPLITS[spec.split][1], spec.index, 1])
    template = generate_template(rng.bytes(32), rng.bytes(12))
    label = render_label(random_payload(rng), template)
    photo = simulate_capture(
        label,
        rng,
        domain=spec.domain,
        difficulty=spec.difficulty,
        quality=spec.quality,
        attack=spec.attack,
    )
    try:
        cap = decode(photo)
        found = extract_cdp_patch(cap, template)
    except CaptureError as exc:
        return Outcome(spec, exc.reason)
    patch = np.clip(found.patch * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Outcome(
        spec, "ok", patch, template, found.alignment_score, cap.quality.sharpness, cap.quality.ok
    )


@dataclass(frozen=True)
class CdpDataset:
    """Arrays for one split. Metadata covers every attempted sample; ``patch`` and
    ``template`` only the ones that decoded (``ok``), in order."""

    patch: np.ndarray  # (n_ok, PATCH_PX, PATCH_PX) uint8
    template: np.ndarray  # (n_ok, CDP_CELLS, CDP_CELLS) uint8
    label: np.ndarray  # (n,) 1 = genuine
    attack: np.ndarray  # (n,) index into ATTACKS, -1 = genuine
    worn: np.ndarray  # (n,) bool, printed on a worn printer
    difficulty: np.ndarray  # (n,) index into DIFFICULTIES
    reason: np.ndarray  # (n,) index into REASONS
    alignment: np.ndarray  # (n,) 0 where not ok
    sharpness: np.ndarray  # (n,)
    quality_ok: np.ndarray  # (n,) bool

    @property
    def ok(self) -> np.ndarray:
        return self.reason == 0

    def decoded(self, name: str) -> np.ndarray:
        """Metadata array restricted to the samples that carry a patch."""
        return getattr(self, name)[self.ok]


def _assemble(outcomes: list[Outcome]) -> CdpDataset:
    ok = [o for o in outcomes if o.patch is not None and o.template is not None]
    return CdpDataset(
        patch=np.stack([o.patch for o in ok if o.patch is not None]),
        template=np.stack([o.template for o in ok if o.template is not None]),
        label=np.array([o.spec.attack is None for o in outcomes], dtype=np.uint8),
        attack=np.array(
            [-1 if o.spec.attack is None else ATTACKS.index(o.spec.attack) for o in outcomes],
            dtype=np.int8,
        ),
        worn=np.array([o.spec.quality == "worn" for o in outcomes]),
        difficulty=np.array(
            [DIFFICULTIES.index(o.spec.difficulty) for o in outcomes], dtype=np.int8
        ),
        reason=np.array([REASONS.index(o.reason) for o in outcomes], dtype=np.int8),
        alignment=np.array([o.alignment for o in outcomes], dtype=np.float32),
        sharpness=np.array([o.sharpness for o in outcomes], dtype=np.float32),
        quality_ok=np.array([o.quality_ok for o in outcomes]),
    )


def _init_worker() -> None:
    cv2.setNumThreads(1)  # parallelism comes from processes, not OpenCV's pool


def concat(parts: list[CdpDataset]) -> CdpDataset:
    """Join datasets generated for consecutive index ranges of the same split."""
    return CdpDataset(
        **{
            name: np.concatenate([getattr(part, name) for part in parts])
            for name in CdpDataset.__dataclass_fields__
        }
    )


def generate(
    split: str,
    n: int,
    seed: int,
    workers: int = 4,
    log_every: int = 1000,
    start: int = 0,
) -> CdpDataset:
    """Simulate samples ``start .. start + n - 1`` of ``split`` across ``workers`` processes."""
    specs = [make_spec(split, i, seed) for i in range(start, start + n)]
    began = time.perf_counter()
    outcomes: list[Outcome] = []
    with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
        for done, outcome in enumerate(pool.imap(render_sample, specs, chunksize=8), 1):
            outcomes.append(outcome)
            if log_every and done % log_every == 0:
                rate = done / (time.perf_counter() - began)
                print(f"  [{split}] {start + done}/{start + n}  {rate:.1f} samples/s", flush=True)
    return _assemble(outcomes)


def _save(path: Path, data: CdpDataset) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: getattr(data, k) for k in data.__dataclass_fields__})
    tmp.replace(path)  # atomic, so an interrupted run never leaves a half-written cache


def _load(path: Path) -> CdpDataset:
    with np.load(path) as f:
        return CdpDataset(**{k: f[k] for k in f.files})


def generation_seconds(split: str, n: int, seed: int, cache_dir: Path) -> float | None:
    """Wall-clock time the cached split took to simulate, when it was recorded."""
    meta = cache_dir / f"{split}_n{n}_s{seed}.json"
    return float(json.loads(meta.read_text())["seconds"]) if meta.exists() else None


def load_or_generate(
    split: str, n: int, seed: int, cache_dir: Path, workers: int = 4, chunk: int = 2000
) -> CdpDataset:
    """Return the cached split if present, otherwise simulate and cache it.

    Generation is saved in chunks of ``chunk`` samples, so a run that is killed (the
    full dataset takes about half an hour) resumes from the last finished chunk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}_n{n}_s{seed}.npz"
    if path.exists():
        return _load(path)
    parts: list[CdpDataset] = []
    began, resumed = time.perf_counter(), False
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        part_path = cache_dir / f"{split}_n{n}_s{seed}.part{start // chunk:03d}.npz"
        if part_path.exists():
            parts.append(_load(part_path))
            resumed = True
            continue
        part = generate(split, size, seed, workers, start=start)
        _save(part_path, part)
        parts.append(part)
    data = concat(parts)
    _save(path, data)
    if not resumed:  # a resumed run cannot tell how long the finished chunks took
        path.with_suffix(".json").write_text(json.dumps({"seconds": time.perf_counter() - began}))
    for part_path in cache_dir.glob(f"{split}_n{n}_s{seed}.part*.npz"):
        part_path.unlink()
    return data


__all__ = [
    "ATTACKS",
    "CDP_CELLS",
    "DIFFICULTIES",
    "PATCH_PX",
    "REASONS",
    "CdpDataset",
    "Spec",
    "concat",
    "generate",
    "generation_seconds",
    "load_or_generate",
    "make_spec",
    "render_sample",
]
