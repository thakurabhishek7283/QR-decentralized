# Model card: CDP authenticity CNN

> **The training and test data are simulated.** No real printer, copier or phone
> produced a single pixel. Every number below measures how well the model
> separates genuine labels from counterfeits *inside a parameterised physical
> simulation*, and says nothing certain about field performance. See
> [Limitations](#limitations).

## Summary

|  |  |
| --- | --- |
| Task | Binary classification: is this captured copy-detection pattern (CDP) the original print of the expected template? |
| Model | `CdpNet`, a 4-block strided CNN, 61,121 parameters |
| Input | `float32 (N, 2, 212, 212)`: captured patch (ink high, standardised) and expected template image, built by `sigil.inference.make_input` |
| Output | One logit for P(genuine); the service applies a sigmoid |
| Artifact | `src/sigil/assets/cdp_cnn.onnx`, 240 KiB, opset 17, dynamic batch, sha256 `01ac64b86a7c9be0...` |
| Trained | 2026-10-10T00:26:42+00:00 on CPU, PyTorch 2.14.1+cu130 |
| Test AUC, CNN | **0.983** (EER 7.5%) on a channel the model never saw |
| Test AUC, classical correlation | 0.843 (EER 23.0%) |

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

| parameter | train | test |
| --- | --- | --- |
| `dot_gain` | 0.85 - 1.38 | 0.9 - 1.45 |
| `ink_noise` | 0.07 - 0.15 | 0.09 - 0.18 |
| `ink_density` | 0.78 - 0.95 | 0.74 - 0.92 |
| `paper_grain` | 0 - 0.05 | 0.01 - 0.06 |
| `dropout` | 0 - 0.008 | 0 - 0.012 |

Worn printers add to a drawn good profile (`dot_gain`, `ink_noise`) or replace it (`dropout`):

| parameter | train | test |
| --- | --- | --- |
| `dot_gain` | 0.03 - 0.1 | 0.05 - 0.14 |
| `ink_noise` | 0.01 - 0.03 | 0.01 - 0.04 |
| `dropout` | 0.015 - 0.03 | 0.02 - 0.04 |

Cameras, by difficulty:

| parameter | train / easy | train / normal | train / hard | test / easy | test / normal | test / hard |
| --- | --- | --- | --- | --- | --- | --- |
| `scale` | 1.05 - 1.4 | 0.75 - 1.18 | 0.62 - 0.9 | 1 - 1.4 | 0.72 - 1.25 | 0.58 - 0.88 |
| `perspective` | 0 - 0.02 | 0.01 - 0.05 | 0.03 - 0.06 | 0 - 0.025 | 0.01 - 0.06 | 0.04 - 0.08 |
| `rotation_deg` | -8 - 8 | -22 - 22 | -35 - 35 | -10 - 10 | -25 - 25 | -45 - 45 |
| `defocus` | 0 - 0.4 | 0.35 - 1.1 | 0.7 - 1.4 | 0 - 0.5 | 0.4 - 1.25 | 0.9 - 1.6 |
| `motion` | 0 - 0.8 | 0 - 1.8 | 0.8 - 2.5 | 0 - 1 | 0 - 2.2 | 1 - 3 |
| `noise` | 0.005 - 0.012 | 0.008 - 0.025 | 0.012 - 0.025 | 0.006 - 0.015 | 0.01 - 0.03 | 0.015 - 0.03 |
| `gamma` | 0.9 - 1.1 | 0.8 - 1.25 | 0.8 - 1.25 | 0.75 - 1.3 | 0.7 - 1.4 | 0.7 - 1.4 |
| `brightness` | 0.9 - 1.1 | 0.78 - 1.15 | 0.7 - 1.2 | 0.85 - 1.15 | 0.72 - 1.2 | 0.7 - 1.2 |
| `jpeg_quality` | 82 - 95 | 65 - 92 | 60 - 85 | 70 - 92 | 52 - 88 | 55 - 80 |
| `shading` | 0 - 0.08 | 0 - 0.18 | 0.05 - 0.22 | 0 - 0.12 | 0 - 0.25 | 0.08 - 0.3 |

Photocopier:

| parameter | train | test |
| --- | --- | --- |
| `scan_scale` | 0.45 - 0.55 | 0.45 - 0.6 |
| `scan_blur` | 0.15 - 0.4 | 0.15 - 0.5 |
| `noise` | 0.01 - 0.03 | 0.015 - 0.035 |
| `steepness` | 8 - 16 | 6 - 20 |
| `tone_jitter` | 0 - 0.08 | 0 - 0.12 |
| `dot_gain` | 0.85 - 1.2 | 0.9 - 1.25 |
| `ink_noise` | 0.07 - 0.12 | 0.07 - 0.14 |

HQ reprint (`scan_blur` is in scan pixels, at 2x the label resolution):

| parameter | train | test |
| --- | --- | --- |
| `scan_blur` | 0.7 - 1.2 | 0.8 - 1.3 |
| `noise` | 0.012 - 0.035 | 0.015 - 0.04 |
| `tone_jitter` | 0 - 0.05 | 0 - 0.08 |
| `dot_gain` | 0.75 - 1.15 | 0.8 - 1.25 |
| `ink_noise` | 0.06 - 0.12 | 0.07 - 0.14 |

Screen re-photography:

| parameter | train | test |
| --- | --- | --- |
| `photo_scale` | 0.7 - 1.1 | 0.65 - 1.1 |
| `photo_blur` | 0.5 - 1 | 0.55 - 1.1 |
| `photo_noise` | 0.008 - 0.02 | 0.01 - 0.03 |
| `jpeg` | 75 - 92 | 65 - 90 |
| `pitch` | 1.3 - 2.2 | 1.2 - 2.4 |
| `gap` | 0.08 - 0.2 | 0.06 - 0.26 |
| `black_lift` | 0.04 - 0.12 | 0.03 - 0.15 |
| `gamma` | 0.85 - 1.2 | 0.8 - 1.3 |
| `moire` | 0.03 - 0.08 | 0.02 - 0.1 |
| `glare` | 0 - 0.1 | 0 - 0.15 |

### Splits

| split | domain | requested | decoded |
| --- | --- | ---: | ---: |
| train | train | 18000 | 16678 |
| val | train (disjoint seeds) | 2500 | 2309 |
| test | **test** (disjoint serials, keys) | 4000 | 3251 |

Classes alternate 1:1; negatives cycle evenly through the three attacks. Capture
difficulty is drawn as easy 30%, normal 50%, hard 20%; 15.0% of printed originals come from a
worn printer. Samples whose QR does not decode (or whose CDP does not align) are
dropped, and counted below. Every sample is a pure function of `(seed, split, index)`.

## Training

AdamW (peak lr 2e-3, weight decay 1e-4), one-cycle cosine schedule,
20 epochs, batch 128, binary cross-entropy. The model with the
best **validation** AUC is exported (epoch 13).

Augmentation is applied to the aligned capture patch, never to the template, and never
changes the class. Every epoch draws fresh perturbations: a sub-pixel shift (up to 1.5 px),
a small rotation and scale error, defocus blur (55% of samples, sigma up to 1.2 px) or
motion blur (15%), a lighting ramp (50%), a gamma change (0.8-1.3) and sensor noise (60%,
sigma up to 0.05). They are physical guesses made before looking at any test number and are
a little wider than the train-domain capture ranges, so the network sees blurrier, noisier
and more misaligned patches than the simulator's train domain produces. They were **not**
fitted to the test domain.

Data generation took 16.0 min on 4 cores, training
28.4 min. Reproduce with `python -m training.train_cdp` (seed
31337); see `training/README.md`.

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

|  | CNN AUC | CNN EER | correlation AUC | correlation EER | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| all decoded | 0.983 | 7.5% | 0.843 | 23.0% | 3251 |
| quality-gated only | 0.990 | 5.7% | 0.896 | 18.5% | 2775 |

Genuine labels against each attack:

|  | CNN AUC | CNN EER | correlation AUC | correlation EER | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| photocopy | 1.000 | 0.2% | 0.903 | 17.3% | 2258 |
| hq_reprint | 1.000 | 0.4% | 0.803 | 27.3% | 2322 |
| screen | 0.945 | 13.4% | 0.827 | 24.2% | 2239 |

By capture difficulty (genuine and counterfeit captured at that difficulty):

|  | CNN AUC | CNN EER | correlation AUC | correlation EER | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy | 0.997 | 2.0% | 0.920 | 15.1% | 1181 |
| normal | 0.988 | 6.4% | 0.896 | 18.5% | 1805 |
| hard | 0.945 | 9.8% | 0.779 | 29.9% | 265 |

### Captures that produced no patch

Share of simulated captures whose QR did not decode or whose CDP did not align,
with the sample count in brackets. They are *not* in the AUCs above; in production
they become `RECAPTURE` or a no-QR result. Counterfeits fail more often because a
degraded copy also degrades its QR.

| split | class | easy | normal | hard | all |
| --- | --- | ---: | ---: | ---: | ---: |
| train | genuine | 0.3% (2688) | 0.2% (4498) | 15.3% (1814) | 3.3% (9000) |
| train | counterfeit | 1.4% (2714) | 4.7% (4498) | 43.6% (1788) | 11.4% (9000) |
| val | genuine | 0.0% (363) | 0.3% (627) | 17.7% (260) | 3.8% (1250) |
| val | counterfeit | 0.6% (353) | 3.8% (653) | 47.5% (244) | 11.4% (1250) |
| test | genuine | 1.4% (625) | 2.4% (997) | 48.4% (378) | 10.8% (2000) |
| test | counterfeit | 5.5% (598) | 15.9% (989) | 83.1% (413) | 26.7% (2000) |

## Operating points for the verification policy

The policy (LLD 7.2) acts on the CNN score `s`: `s >= cdp_genuine_min` (and a low clone
risk) may pass as GENUINE, `s <= cdp_counterfeit_max` is a confident COUNTERFEIT, and
anything in between is escalated to the investigator.

**Chosen on the validation split, never on test.** `cdp_counterfeit_max` is the
1.0% quantile of genuine validation scores (so at most 1.0% of genuine captures
fall at or below it) and `cdp_genuine_min` is the 99.5% quantile of counterfeit
validation scores (so at most 0.5% of counterfeits reach it), both rounded outwards:

| setting | recommended | LLD default |
| --- | ---: | ---: |
| `cdp_genuine_min` | **0.966** | 0.80 |
| `cdp_counterfeit_max` | **0.422** | 0.20 |

What those thresholds do on the **test** domain (a channel the model never saw, so the
rates differ from the 1.0% / 0.5% budget by the size of the domain gap):

| population | n | accepted (>= genuine_min) | escalated | rejected (<= max) |
| --- | ---: | ---: | ---: | ---: |
| genuine | 1784 | 43.9% | 50.1% | 5.9% |
| genuine, good printer | 1522 | 47.0% | 48.2% | 4.8% |
| genuine, worn printer | 262 | 26.3% | 61.1% | 12.6% |
| genuine, easy | 616 | 79.5% | 20.5% | 0.0% |
| genuine, normal | 973 | 30.1% | 64.3% | 5.5% |
| genuine, hard | 195 | 0.5% | 72.8% | 26.7% |
| counterfeit | 1467 | 0.1% | 8.2% | 91.8% |
| photocopy | 474 | 0.0% | 0.2% | 99.8% |
| hq_reprint | 538 | 0.0% | 0.2% | 99.8% |
| screen | 455 | 0.2% | 25.9% | 73.8% |

For reference, the same populations on the **validation** split (same channel as
training; the budget holds here by construction):

| population | n | accepted (>= genuine_min) | escalated | rejected (<= max) |
| --- | ---: | ---: | ---: | ---: |
| genuine | 1202 | 64.1% | 34.9% | 1.0% |
| genuine, good printer | 1023 | 66.8% | 32.6% | 0.7% |
| genuine, worn printer | 179 | 49.2% | 48.0% | 2.8% |
| genuine, easy | 363 | 94.5% | 5.5% | 0.0% |
| genuine, normal | 625 | 65.3% | 33.9% | 0.8% |
| genuine, hard | 214 | 9.3% | 87.4% | 3.3% |
| counterfeit | 1107 | 0.4% | 16.8% | 82.8% |
| photocopy | 362 | 0.0% | 0.0% | 100.0% |
| hq_reprint | 384 | 0.0% | 2.9% | 97.1% |
| screen | 361 | 1.1% | 48.5% | 50.4% |

And the test split under the LLD's placeholder defaults (0.80 / 0.20):

| population | n | accepted (>= genuine_min) | escalated | rejected (<= max) |
| --- | ---: | ---: | ---: | ---: |
| genuine | 1784 | 74.4% | 24.6% | 1.0% |
| genuine, good printer | 1522 | 76.9% | 22.4% | 0.7% |
| genuine, worn printer | 262 | 59.9% | 37.0% | 3.1% |
| genuine, easy | 616 | 98.2% | 1.8% | 0.0% |
| genuine, normal | 973 | 70.1% | 29.3% | 0.6% |
| genuine, hard | 195 | 21.0% | 72.8% | 6.2% |
| counterfeit | 1467 | 1.2% | 16.0% | 82.8% |
| photocopy | 474 | 0.0% | 0.2% | 99.8% |
| hq_reprint | 538 | 0.0% | 0.2% | 99.8% |
| screen | 455 | 4.0% | 51.0% | 45.1% |

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

| group | n | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| genuine, good printer | 1522 | 0.870 | 0.431 | 0.825 | 0.959 | 0.991 | 0.998 |
| genuine, worn printer | 262 | 0.767 | 0.246 | 0.586 | 0.870 | 0.968 | 0.994 |

Validation split:

| group | n | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| genuine, good printer | 1023 | 0.938 | 0.685 | 0.942 | 0.988 | 0.996 | 0.999 |
| genuine, worn printer | 179 | 0.878 | 0.447 | 0.832 | 0.965 | 0.991 | 0.998 |

Classical correlation score on the test domain, same populations:

| group | n | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| genuine, good printer | 1522 | 0.602 | 0.085 | 0.450 | 0.674 | 0.804 | 0.886 |
| genuine, worn printer | 262 | 0.244 | 0.019 | 0.090 | 0.205 | 0.367 | 0.587 |

Counterfeit CNN scores (test domain):

| group | n | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counterfeit: photocopy | 474 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| counterfeit: hq_reprint | 538 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.009 |
| counterfeit: screen | 455 | 0.303 | 0.044 | 0.117 | 0.234 | 0.443 | 0.774 |
| counterfeit: all | 1467 | 0.096 | 0.000 | 0.000 | 0.000 | 0.102 | 0.570 |

### Threshold sweep (test domain, descriptive only)

`cdp_genuine_min` (a score at or above it may pass as genuine):

| threshold | genuine passing | counterfeit passing |
| ---: | ---: | ---: |
| 0.50 | 91.0% | 6.8% |
| 0.80 | 74.4% | 1.2% |
| 0.90 | 62.9% | 0.5% |
| 0.95 | 50.4% | 0.1% |
| 0.99 | 24.7% | 0.1% |

`cdp_counterfeit_max` (a score at or below it is a confident copy):

| threshold | counterfeit caught | genuine wrongly rejected |
| ---: | ---: | ---: |
| 0.01 | 66.3% | 0.0% |
| 0.05 | 70.3% | 0.1% |
| 0.10 | 74.8% | 0.2% |
| 0.20 | 82.8% | 1.0% |
| 0.30 | 87.8% | 2.7% |
| 0.50 | 93.2% | 9.0% |

## Serving performance

Closed-loop benchmark (`python -m sigil.inference.bench`): N clients each send the next
request as soon as the previous answer arrives, for 5 s after a 1 s warm-up. Each request
is the full `InferenceService.score` path on a real simulated patch (classical
correlation, `make_input`, ONNX Runtime). 4 vCPU container, ORT intra-op threads left at
the default, single process, other light jobs sharing the machine, so treat differences
under ~15% as noise. "unbatched" is `max_batch_size=1`; "batched" is the default
`max_batch_size=32, max_delay_ms=4`.

| concurrency | mode | req/s | speed-up | p50 ms | p95 ms | p99 ms | mean batch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | unbatched | 420.2 | 1.00x | 2.2 | 3.2 | 4.4 | 1.0 |
| 1 | batched | 397.6 | 0.95x | 2.3 | 3.6 | 5.5 | 1.0 |
| 8 | unbatched | 362.6 | 1.00x | 20.1 | 37.6 | 52.3 | 1.0 |
| 8 | batched | 580.4 | 1.60x | 13.4 | 18.2 | 23.0 | 8.0 |
| 32 | unbatched | 432.8 | 1.00x | 71.1 | 92.1 | 190.8 | 1.0 |
| 32 | batched | 559.2 | 1.29x | 55.8 | 68.7 | 79.8 | 32.0 |

How to read it:

* The CNN itself costs about 0.3 ms per patch (0.2 ms per patch in a batch of 8-32), while
  pre-processing on the event loop (Pearson correlation plus `make_input`) costs about
  1 ms. A single Python process therefore tops out near 500-600 req/s whatever the
  batcher does, and batching can only remove the small model share, which is why the
  gain is 1.3-1.6x rather than an order of magnitude. Scaling further means more
  processes, not a bigger batch.
* With one client, batching is neutral (0.95x is within noise). The service tells the
  batcher how many requests are in flight, so a lone request is dispatched immediately
  instead of idling for `max_delay_ms`.
* Two defects were found and fixed while producing these numbers. ONNX Runtime's
  intra-op workers busy-wait between calls; on four shared cores that starved the
  event loop and the batcher thread, and switching it off
  (`session.intra_op.allow_spinning=0`) lifted unbatched throughput from about 125 to
  about 450 req/s with p50 falling from 5.5 ms to 2.2 ms. Before the in-flight hint, a
  lone request in batched mode always waited out the 4 ms delay (0.3-0.6x of unbatched
  at concurrency 1).
* Intra-op thread counts of 1, 2 and 4 gave the same throughput within noise once
  spinning was off, so the default (ORT chooses) is kept.


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
