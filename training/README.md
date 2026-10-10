# CDP model training

Trains, evaluates and exports the copy-detection-pattern CNN that `sigil.inference`
serves. Everything here runs on CPU (4 cores is enough). Results and caveats are written
to [`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md).

> The data is **simulated** by `sigil.cdp.channel`. No real printer, copier or phone is
> involved, so the reported numbers measure separation inside a parameterised physical
> model, not field accuracy.

## Setup

The `train` extra provides PyTorch and ONNX (runtime inference only needs `onnxruntime`):

```bash
uv sync --extra train
```

## Reproduce the shipped model

```bash
python -m training.train_cdp          # or: make train
```

| stage | default size | time on 4 cores |
|---|---|---|
| simulate train split | 18 000 samples | about 30 min (9-11 samples/s with 3 workers) |
| simulate validation split | 2 500 samples | about 4 min |
| simulate test split | 4 000 samples | about 7 min |
| train (20 epochs, ~61k parameters) | about 16 700 decoded patches | about 28 min (80-85 s per epoch) |
| evaluate, export ONNX, write assets and model card | | about 1 min |

Simulated splits are cached in `training/data/` (git-ignored), so a second run skips the
simulation. Generation is saved in chunks of 2 000 samples and training checkpoints after
every epoch to `training/runs/`, so an interrupted run resumes where it stopped. Use
`--fresh` to ignore an old checkpoint. Everything is deterministic given `--seed`: sample
`i` of a split is a pure function of `(seed, split, i)`, independent of the worker count.

Outputs:

* `src/sigil/assets/cdp_cnn.onnx` (opset 17, dynamic batch, input `input`, output `logit`)
* `src/sigil/assets/cdp_cnn.json` (version, sha256, test metrics, correlation calibration,
  recommended policy `operating_points`, genuine score statistics by printer condition)
* `docs/MODEL_CARD.md`, rendered from the run so it cannot drift from the numbers

## Quick mode (smoke test)

```bash
python -m training.train_cdp --quick
```

600 / 200 / 400 samples and 3 epochs, a few minutes in total. It exercises the whole
pipeline (simulation, decode, alignment, training, evaluation, threshold selection) and
**writes no assets and no model card**. The resulting numbers are meaningless. Add
`--write-assets` to also export (do not commit the result).

## Other options

```text
--n-train / --n-val / --n-test    split sizes
--epochs, --batch-size, --seed
--widths 16,32,64,64              channels of the four conv blocks
--workers N                       simulation processes
--card-only                       re-render the model card from the last run
--bench-md PATH                   serving benchmark to embed (default: training/serving_bench.md)
```

## Serving benchmark

The "Serving performance" section of the model card embeds `training/serving_bench.md`,
which is hand-maintained so that retraining never wipes it. Refresh it with

```bash
python -m sigil.inference.bench --out training/serving_bench.md
python -m training.train_cdp --card-only
```

and update the commentary in that file if the numbers change materially.

## Train and test domains

`sigil.cdp.channel` defines a *train* and a *test* parameter domain. The test domain has
wider blur, noisier sensors, lower JPEG quality, a wider gamma range, rougher printers and
different attacker equipment. Train and validation draw from the train domain with
disjoint seeds; the test split uses the test domain with its own serials and keys.

* Model selection (best epoch) and the policy thresholds use **validation only**.
* The test split is reported once per run and is not used to tune anything.
* Training-time augmentation (`train_cdp.augment`) is a set of physical guesses that makes
  the network see harder patches than the train domain produces; it is not fitted to the
  test ranges.

## Layout

```text
dataset.py     simulate (print -> attack -> capture -> decode -> align) and cache splits
train_cdp.py   model, augmentation, training, evaluation, thresholds, ONNX export
model_card.py  renders docs/MODEL_CARD.md from a run
serving_bench.md   serving benchmark embedded in the card (hand-maintained)
```
