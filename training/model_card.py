"""Render ``docs/MODEL_CARD.md`` from a training run, so the card cannot drift from the numbers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sigil.cdp import channel
from training.dataset import DIFFICULTIES, DIFFICULTY_WEIGHTS, WORN_FRACTION

NO_BENCH = (
    "_Not recorded. Run `python -m sigil.inference.bench --out training/serving_bench.md` and "
    "re-render the card with `python -m training.train_cdp --card-only`._"
)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    """Markdown table; ``align`` is one character per column (``l`` or ``r``)."""
    align = align or "l" * len(header)
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = [header, rule, *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def _range_table(tables: Mapping[str, Mapping[str, Any]], names: list[str] | None = None) -> str:
    """Train-vs-test parameter ranges for one table of ``channel`` ranges."""
    names = names or list(tables["train"])
    rows = [
        [
            f"`{name}`",
            *(f"{tables[d][name][0]:g} - {tables[d][name][1]:g}" for d in ("train", "test")),
        ]
        for name in names
    ]
    return _table(["parameter", "train", "test"], rows)


def _capture_table() -> str:
    keys = [(d, k) for d in ("train", "test") for k in DIFFICULTIES]
    rows = [
        [
            f"`{name}`",
            *(f"{r[0]:g} - {r[1]:g}" for r in (channel.CAPTURE_RANGES[k][name] for k in keys)),
        ]
        for name in channel.CAPTURE_RANGES[("train", "easy")]
    ]
    return _table(["parameter", *(f"{d} / {k}" for d, k in keys)], rows)


def _metric_table(groups: Mapping[str, Any]) -> str:
    header = ["", "CNN AUC", "CNN EER", "correlation AUC", "correlation EER", "n"]
    rows = [
        [
            name,
            f"{row['cnn']['auc']:.3f}",
            _pct(row["cnn"]["eer"]),
            f"{row['corr']['auc']:.3f}",
            _pct(row["corr"]["eer"]),
            str(row["cnn"]["n"]),
        ]
        for name, row in groups.items()
    ]
    return _table(header, rows, "lrrrrr")


def _failure_table(rates: Mapping[str, Any]) -> str:
    rows = []
    for split, kinds in rates.items():
        for kind, per in kinds.items():
            cells = [f"{_pct(per[d]['rate'])} ({per[d]['n']})" for d in (*DIFFICULTIES, "all")]
            rows.append([split, kind, *cells])
    return _table(["split", "class", *DIFFICULTIES, "all"], rows, "ll" + "r" * 4)


def _score_table(groups: Mapping[str, Mapping[str, Any]], key: str) -> str:
    header = ["group", "n", "mean", "p05", "p25", "median", "p75", "p95"]
    rows = []
    for name, scores in groups.items():
        s = scores[key]
        values = (s["mean"], s["p05"], s["p25"], s["p50"], s["p75"], s["p95"])
        rows.append([name, str(s["n"]), *(f"{v:.3f}" for v in values)])
    return _table(header, rows, "l" + "r" * 7)


def _policy_table(rates: Mapping[str, Mapping[str, float]]) -> str:
    rows = [
        [
            name,
            str(int(r["n"])),
            _pct(r["accepted"]),
            _pct(r["escalated"]),
            _pct(r["rejected"]),
        ]
        for name, r in rates.items()
    ]
    return _table(
        ["population", "n", "accepted (>= genuine_min)", "escalated", "rejected (<= max)"],
        rows,
        "lrrrr",
    )


def render(
    run: dict[str, Any], metadata: dict[str, Any], onnx_path: Path, bench: str | None
) -> str:
    m, sizes, seconds, args = run["metrics"], run["sizes"], run["seconds"], run["args"]
    overall, ops = m["overall"], m["sweep"]
    split_rows = [
        [name, domain, str(sizes[name]["requested"]), str(sizes[name]["decoded"])]
        for name, domain in (
            ("train", "train"),
            ("val", "train (disjoint seeds)"),
            ("test", "**test** (disjoint serials, keys)"),
        )
    ]
    genuine_scores = {
        "genuine, good printer": m["genuine_scores"]["good"],
        "genuine, worn printer": m["genuine_scores"]["worn"],
    }
    counterfeit_scores = {f"counterfeit: {k}": v for k, v in m["counterfeit_scores"].items()}
    sweep = ops
    genuine_min = _table(
        ["threshold", "genuine passing", "counterfeit passing"],
        [
            [f"{r['threshold']:.2f}", _pct(r["genuine_pass"]), _pct(r["counterfeit_pass"])]
            for r in sweep["genuine_min"]
        ],
        "rrr",
    )
    counterfeit_max = _table(
        ["threshold", "counterfeit caught", "genuine wrongly rejected"],
        [
            [f"{r['threshold']:.2f}", _pct(r["counterfeit_caught"]), _pct(r["genuine_rejected"])]
            for r in sweep["counterfeit_max"]
        ],
        "rrr",
    )
    th, policy, budget = run["thresholds"], run["policy"], run["budget"]
    val_scores = run["metrics_val"]["genuine_scores"]
    genuine_val = {
        "genuine, good printer": val_scores["good"],
        "genuine, worn printer": val_scores["worn"],
    }
    g_budget, c_budget = _pct(budget["genuine_rejected"]), _pct(budget["counterfeit_accepted"])
    c_quantile = _pct(1.0 - budget["counterfeit_accepted"])
    threshold_table = _table(
        ["setting", "recommended", "LLD default"],
        [
            ["`cdp_genuine_min`", f"**{th['cdp_genuine_min']:.3f}**", "0.80"],
            ["`cdp_counterfeit_max`", f"**{th['cdp_counterfeit_max']:.3f}**", "0.20"],
        ],
        "lrr",
    )
    best_epoch = int(max(run["history"], key=lambda h: h["val_auc"])["epoch"])
    mix = ", ".join(f"{d} {w:.0%}" for d, w in zip(DIFFICULTIES, DIFFICULTY_WEIGHTS, strict=True))
    size_kib = onnx_path.stat().st_size / 1024
    return f"""# Model card: CDP authenticity CNN

> **The training and test data are simulated.** No real printer, copier or phone
> produced a single pixel. Every number below measures how well the model
> separates genuine labels from counterfeits *inside a parameterised physical
> simulation*, and says nothing certain about field performance. See
> [Limitations](#limitations).

## Summary

{
        _table(
            ["", ""],
            [
                [
                    "Task",
                    "Binary classification: is this captured copy-detection pattern (CDP) the original print of the expected template?",
                ],
                ["Model", f"`CdpNet`, a 4-block strided CNN, {run['params']:,} parameters"],
                [
                    "Input",
                    "`float32 (N, 2, 212, 212)`: captured patch (ink high, standardised) and expected template image, built by `sigil.inference.make_input`",
                ],
                ["Output", "One logit for P(genuine); the service applies a sigmoid"],
                [
                    "Artifact",
                    f"`src/sigil/assets/cdp_cnn.onnx`, {size_kib:.0f} KiB, opset 17, dynamic batch, sha256 `{metadata['sha256'][:16]}...`",
                ],
                ["Trained", f"{metadata['trained_at']} on CPU, PyTorch {run['torch']}"],
                [
                    "Test AUC, CNN",
                    f"**{overall['cnn']['auc']:.3f}** (EER {_pct(overall['cnn']['eer'])}) on a channel the model never saw",
                ],
                [
                    "Test AUC, classical correlation",
                    f"{overall['corr']['auc']:.3f} (EER {_pct(overall['corr']['eer'])})",
                ],
            ],
        )
    }

## Intended use

Score the CDP region of a phone photo of a Sigil label, as *one* signal in the
verification policy (alongside signature, log inclusion and clone analytics). It
is built to separate an original print from a degraded copy of that print. It is
not a stand-alone authenticity oracle, and the verdict policy never lets it
overrule a failed cryptographic check.

Out of scope: any label geometry other than LLD 4.1, colour CDPs, and printers or
phones outside the simulated ranges.

## How the data is generated

Each sample runs the production pipeline end to end:
`render_label` -> `print_label` -> (`counterfeit_from`) -> `capture` -> `decode`
-> `extract_cdp_patch` -> `make_input`. The attacker only ever sees the *printed*
label, never the template.

* **Print** (`print_label`): Gaussian ink spread (dot gain), a noisy decision
  threshold, ink density, paper grain, and worn heads that flip random CDP cells.
* **Capture** (`capture`): resampling by a scale factor, a random homography into
  a cluttered 1280x960 scene, defocus, motion blur, uneven lighting, sensor noise,
  gamma, brightness and JPEG compression through `cv2.imencode`.
* **Photocopy**: ~0.5x scan, tone clipping, laser reprint with heavy dot gain.
* **HQ reprint**: 2x scan, denoise, re-binarise, reprint on a good printer. A good
  scan recovers most cells, so this is the attack that stays closest to the original.
* **Screen**: the label is photographed, shown on a pixel-grid display (aperture
  gaps, lifted blacks, tone curve, glare, moire) and photographed again.

### Train and test domains

The test domain uses deliberately different, mostly wider, parameter ranges and
different attacker equipment, with its own serials and keys. Held-out metrics
therefore measure generalisation to an unseen channel, not memorisation.

Printers (good):

{_range_table(channel.PRINT_RANGES)}

Worn printers add to a drawn good profile (`dot_gain`, `ink_noise`) or replace it (`dropout`):

{_range_table(channel.WORN_RANGES)}

Cameras, by difficulty:

{_capture_table()}

Photocopier:

{_range_table(channel.PHOTOCOPY_RANGES)}

HQ reprint (`scan_blur` is in scan pixels, at 2x the label resolution):

{_range_table(channel.HQ_REPRINT_RANGES)}

Screen re-photography:

{_range_table(channel.SCREEN_RANGES)}

### Splits

{_table(["split", "domain", "requested", "decoded"], split_rows, "llrr")}

Classes alternate 1:1; negatives cycle evenly through the three attacks. Capture
difficulty is drawn as {mix}; {_pct(WORN_FRACTION)} of printed originals come from a
worn printer. Samples whose QR does not decode (or whose CDP does not align) are
dropped, and counted below. Every sample is a pure function of `(seed, split, index)`.

## Training

AdamW (peak lr 2e-3, weight decay 1e-4), one-cycle cosine schedule,
{args["epochs"]} epochs, batch {args["batch_size"]}, binary cross-entropy. The model with the
best **validation** AUC is exported (epoch {best_epoch}).

Augmentation is applied to the aligned capture patch, never to the template, and never
changes the class. Every epoch draws fresh perturbations: a sub-pixel shift (up to 1.5 px),
a small rotation and scale error, defocus blur (55% of samples, sigma up to 1.2 px) or
motion blur (15%), a lighting ramp (50%), a gamma change (0.8-1.3) and sensor noise (60%,
sigma up to 0.05). They are physical guesses made before looking at any test number and are
a little wider than the train-domain capture ranges, so the network sees blurrier, noisier
and more misaligned patches than the simulator's train domain produces. They were **not**
fitted to the test domain.

Data generation took {seconds["data"] / 60:.1f} min on 4 cores, training
{seconds["train"] / 60:.1f} min. Reproduce with `python -m training.train_cdp` (seed
{args["seed"]}); see `training/README.md`.

### How honest is the split

* Train and validation share the *train* domain (disjoint seeds, serials and keys); test
  uses the *test* domain's wider ranges and attacker equipment, with its own serials and
  keys. Model selection (best epoch) and the policy thresholds use validation only.
* An earlier, smaller iteration of this model (about 3k training samples) was compared on
  test-domain samples while the augmentation and architecture were being explored. The
  final design was fixed a priori from that and **the test set used here is a fresh draw**
  (new seed) that nothing was tuned on, but the test *ranges* were known to the authors.
  Treat the test numbers as "out-of-range generalisation inside the same simulator", not as
  an independent evaluation.

## Results (test domain)

AUC and equal error rate, genuine = positive class, over all decoded captures.
"Quality-gated" keeps only captures that pass `assess_quality` (sharp enough, well
exposed), which is what a deployed system would score.

{_metric_table({"all decoded": overall, "quality-gated only": m["quality_gated"]})}

Genuine labels against each attack:

{_metric_table(m["by_attack"])}

By capture difficulty (genuine and counterfeit captured at that difficulty):

{_metric_table(m["by_difficulty"])}

### Captures that produced no patch

Share of simulated captures whose QR did not decode or whose CDP did not align,
with the sample count in brackets. They are *not* in the AUCs above; in production
they become `RECAPTURE` or a no-QR result. Counterfeits fail more often because a
degraded copy also degrades its QR.

{_failure_table(m["decode_failure_rate"])}

## Operating points for the verification policy

The policy (LLD 7.2) acts on the CNN score `s`: `s >= cdp_genuine_min` (and a low clone
risk) may pass as GENUINE, `s <= cdp_counterfeit_max` is a confident COUNTERFEIT, and
anything in between is escalated to the investigator.

**Chosen on the validation split, never on test.** `cdp_counterfeit_max` is the
{g_budget} quantile of genuine validation scores (so at most {g_budget} of genuine captures
fall at or below it) and `cdp_genuine_min` is the {c_quantile} quantile of counterfeit
validation scores (so at most {c_budget} of counterfeits reach it), both rounded outwards:

{threshold_table}

What those thresholds do on the **test** domain (a channel the model never saw, so the
rates differ from the {g_budget} / {c_budget} budget by the size of the domain gap):

{_policy_table(policy["test"])}

For reference, the same populations on the **validation** split (same channel as
training; the budget holds here by construction):

{_policy_table(policy["val"])}

And the test split under the LLD's placeholder defaults (0.80 / 0.20):

{_policy_table(policy["test_default"])}

Reading the tables: an "accepted" counterfeit is a counterfeit whose CDP score alone would
let it pass; in the full policy it still needs a valid signature, a log entry and a low
clone risk. An "escalated" label costs an investigator run, not a wrong verdict. The
thresholds describe the simulated channel and must be re-derived from real captures before
use; the QC sampling at issuance exists for that.

### Worn versus good printers (genuine labels)

The investigator recognises a worn printer from its batch QC baseline. Genuine labels from
worn heads (extra dot gain and noise, 1.5-3% dead cells) score visibly lower than ones from
good printers, which is why a low genuine score is not by itself proof of a copy.

Test domain:

{_score_table(genuine_scores, "cnn")}

Validation split:

{_score_table(genuine_val, "cnn")}

Classical correlation score on the test domain, same populations:

{_score_table(genuine_scores, "corr")}

Counterfeit CNN scores (test domain):

{_score_table(counterfeit_scores, "cnn")}

### Threshold sweep (test domain, descriptive only)

`cdp_genuine_min` (a score at or above it may pass as genuine):

{genuine_min}

`cdp_counterfeit_max` (a score at or below it is a confident copy):

{counterfeit_max}

## Serving performance

{bench or NO_BENCH}

## Limitations

* **The channel is simulated.** The physical model is plausible and the train/test
  split is disjoint, but domain shift between this simulator and real printers,
  copiers and phone cameras is untested. Real optics, halftoning, paper and
  compression artefacts will differ, so the numbers are not field accuracy.
* **The attackers are fixed algorithms.** An adaptive attacker who tunes scanning,
  thresholding and printing against the verifier would do better than any of the
  three modelled attacks.
* **The test domain is wider than the training domain, not independent of it.** It
  was designed by the same author from the same physical model.
* **Hard captures are dominated by decode failures**, which are excluded from the
  AUCs. Reported accuracy is conditional on a readable QR.
* **Good-printer reprints and worn-printer originals overlap**, so a band between
  the two policy thresholds is expected to go to the investigator.
* The classical baseline is fitted on the same simulated training data.
"""  # noqa: E501
