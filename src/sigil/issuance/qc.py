"""Production-line quality control: measure how a batch's genuine labels score.

Printers drift, so a batch printed on worn heads scores lower across the board. Recording the
distribution at the factory gives the investigator a baseline to tell "worn printer" from "copy".
"""

from __future__ import annotations

import asyncio

import numpy as np

from sigil.cdp import (
    PrintProfile,
    capture,
    print_label,
    sample_capture_profile,
    sample_print_profile,
)
from sigil.cdp.channel import Domain
from sigil.errors import CaptureError, SigilError
from sigil.inference import InferenceService
from sigil.issuance.issuer import Issuer
from sigil.issuance.store import QcStats
from sigil.vision import CdpPatch, decode, extract_cdp_patch

ATTEMPTS_PER_SAMPLE = 3  # a QC camera simply re-shoots an unreadable label


async def run_qc(
    issuer: Issuer,
    inference: InferenceService,
    batch_id: str,
    *,
    samples: int,
    rng: np.random.Generator,
    domain: Domain = "test",
    print_profile: PrintProfile | None = None,
) -> QcStats:
    """Simulate the QC camera over ``samples`` units of a batch, store and return the stats.

    The whole batch comes off one press, so a single print profile is drawn for the batch's
    ``print_quality`` and shared by every sample; pass ``print_profile`` to reuse the press that
    printed a specific unit. Captures are "easy" (a fixed camera under good light) and each
    sample's randomness comes from its own child generator, so results do not depend on how the
    concurrent samples interleave.
    """
    info = issuer.store.batch(batch_id)
    serials = issuer.store.units(batch_id)
    chosen = [serials[i] for i in rng.permutation(len(serials))[:samples]]
    profile = print_profile or sample_print_profile(rng, domain, info.print_quality)

    async def score_one(serial_hex: str, sample_rng: np.random.Generator) -> float | None:
        label = issuer.render_label(serial_hex)
        template = issuer.template_for(bytes.fromhex(serial_hex))
        for _ in range(ATTEMPTS_PER_SAMPLE):
            patch = await asyncio.to_thread(_shoot, label, template, profile, sample_rng, domain)
            if patch is not None:
                return (await inference.score(patch, template)).cnn
        return None

    results = await asyncio.gather(
        *(score_one(s, r) for s, r in zip(chosen, rng.spawn(len(chosen)), strict=True))
    )
    scores = np.array([s for s in results if s is not None])
    if scores.size == 0:
        raise SigilError(f"QC could not read any label of batch {batch_id}")
    stats = QcStats(
        n=int(scores.size),
        mean=float(scores.mean()),
        std=float(scores.std()),
        min=float(scores.min()),
    )
    issuer.store.set_qc(batch_id, stats)
    return stats


def _shoot(
    label: np.ndarray,
    template: np.ndarray,
    profile: PrintProfile,
    rng: np.random.Generator,
    domain: Domain,
) -> CdpPatch | None:
    """Print, photograph and extract one label; ``None`` if the shot is unreadable."""
    photo = capture(
        print_label(label, profile, rng), sample_capture_profile(rng, domain, "easy"), rng
    )
    try:
        return extract_cdp_patch(decode(photo), template)
    except CaptureError:
        return None
