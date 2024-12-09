# Sigil — High-Level Design

> **Sigil** checks whether a physical product is genuine. Each label carries
> three things: a cryptographic seal, a copy-detection pattern that computer
> vision can check, and an entry in a public transparency log. A fixed,
> ordered policy turns the evidence into a verdict, and a scenario eval suite
> measures it.

| | |
|---|---|
| Status | Approved for implementation |
| Supersedes | `FakeProductDetector.sol` (2022, Next.js + Solidity 0.4 prototype in this repository's history) |
| Companion | [LLD.md](LLD.md) — module contracts, data formats, algorithms |

---

## 1. Problem

Counterfeit and pirated goods are estimated by the OECD/EUIPO at roughly
**2.5 % of world trade**. Brands answer with serialised QR codes. Consumers scan
them, and a server says "genuine".

That approach is broken, and the 2022 prototype in this repository's history
shows exactly how:

| Failure | Why it happens | In the 2022 prototype |
|---|---|---|
| **The QR is copyable** | A QR code is just data. A photocopy scans exactly like the original, so "is this serial registered?" proves nothing about the object in your hand. | Product IDs were only looked up on-chain. A photocopied label passed. |
| **Cloned serials** | One genuine label is copied onto 10,000 fakes. Every copy decodes to a valid serial. | No scan analytics. |
| **Trust in the operator** | The verifier is a server the brand runs, and nobody can audit what it signed or when. A blockchain "fixes" this at enormous cost and complexity. | Public chain, gas per product, and `transferOwnership` had **no authorisation check**: anyone could take ownership of anyone's product. |
| **Binary answers** | Real captures are blurry, glare-ridden and taken on cheap phones. A system that can only say yes or no is either insecure or annoying. | No notion of uncertainty. |

## 2. Goals and non-goals

**Goals**

1. **Bind the digital identity to the physical print.** A photocopy, or a
   high-quality reprint, of a genuine label must be detectable from a phone
   photo.
2. **Unforgeable credentials.** Nobody without the issuer's key can mint a
   valid label.
3. **Auditable issuance without a blockchain.** Anyone can verify that a
   credential was publicly logged, and that the log is append-only. This is the
   Certificate Transparency model (RFC 9162).
4. **Clone detection** from scan telemetry.
5. **Calibrated, explainable outcomes.** The verdicts are `GENUINE`,
   `COUNTERFEIT`, `SUSPICIOUS` and `RECAPTURE`, and each one comes with an
   evidence trail.
6. **Conservative by construction.** Hard failures (signature, key status, log
   inclusion) are decided before the pattern score is looked at, and anything
   the evidence does not settle is `SUSPICIOUS`, never `GENUINE`. A scenario
   eval suite measures the policy and gates CI.
7. **Production-shaped inference.** CV models are served through ONNX Runtime
   behind a dynamic micro-batcher, with backpressure, metrics and a benchmark
   harness.
8. **Runs on a laptop with zero API keys.** It has to be reviewable by anyone.

**Non-goals**

- Calibrating against physical printers and real phone cameras. The print/scan
  channel is simulated with a parameterised physical model, which LLD §4.2
  documents as the main limitation.
- Native mobile apps. The web demo and the API are enough.
- Multi-tenant billing, user accounts, or KMS/HSM integration. The key provider
  is a seam with a file-backed dev implementation.

## 3. System context

```
 Manufacturer ──(issue batch)──►┌───────────────────────────────────────────┐
                                 │                  Sigil                    │
 Consumer / inspector ─(photo)─► │  Issuance · Verification                 │
                                 │  Transparency log · Inference service    │
 Auditor / witness ──(proofs)──► └───────────────────────────────────────────┘
```

| Actor | Needs |
|---|---|
| Manufacturer (issuer) | Mint labels for a batch, revoke a compromised key, see clone alerts |
| Consumer / field inspector | Point a camera at a label and get a trustworthy answer in under a second |
| Auditor / witness | Independently check that the log is append-only and that every credential is in it |
| Operator | Metrics and eval reports |

## 4. Architecture

```
                    ┌──────────────────── API (FastAPI) ────────────────────┐
                    │  /batches    /scans    /log/*    /keys    /metrics  / │
                    └──────┬────────────┬──────────────┬────────────────────┘
                           │            │              │
               ┌───────────▼───┐  ┌─────▼───────────┐  ┌▼───────────────┐
               │   Issuance    │  │  Verification   │  │ Transparency   │
               │ keys, creds,  │  │  pipeline       │  │ log (Merkle,   │
               │ label render  │  │ (deterministic) │  │ STH, proofs)   │
               └──┬──────┬─────┘  └─┬────┬─────┬────┘  └────────▲───────┘
                  │      │          │    │     │                │
                  │      └──────────┼────┼─────┼────────────────┘
                  │                 │    │     │
         ┌────────▼───────┐ ┌───────▼─┐ ┌▼─────▼──────┐
         │ CDP generator  │ │ Vision  │ │ Scan store +│
         │ (keyed PRNG)   │ │ decode, │ │ clone       │
         └────────────────┘ │ rectify,│ │ analytics   │
                            │ quality │ └─────────────┘
                            └───┬─────┘
                                │
                     ┌──────────▼───────────┐
                     │ Inference service     │
                     │ ONNX Runtime +        │
                     │ dynamic micro-batcher │
                     └───────────────────────┘
```

### 4.1 Components

| Component | Responsibility | Key technology |
|---|---|---|
| **Crypto** | Issuer keys (Ed25519) with key IDs and revocation. Compact, canonical credential encoding inside the QR. | `cryptography`, canonical CBOR-free binary codec, base45 for QR alphanumeric mode |
| **Transparency log** | Append-only Merkle tree over credential hashes: signed tree heads (STH), inclusion and consistency proofs, plus a stateless auditor that checks them. | RFC 9162 hashing (`0x00` leaf / `0x01` node prefixes), SQLite |
| **CDP** | A copy-detection pattern: a high-entropy binary grid generated from `HMAC(cdp_key, serial)`. Every print/scan cycle destroys information, so a copy correlates measurably worse with the secret template than an original does. | NumPy, keyed DRBG |
| **Vision** | Find and decode the QR in a phone photo, rectify perspective from the QR corners, crop the CDP region, and gate on capture quality (blur, exposure, resolution). | OpenCV |
| **Inference service** | Score CDP authenticity with a small two-input CNN (the captured patch next to the expected template). ONNX Runtime sits behind an asyncio dynamic micro-batcher with bounded queues, timeouts and metrics. | PyTorch (training only), ONNX, ONNX Runtime |
| **Scan analytics** | Record scan events and compute clone risk: impossible travel, geographic dispersion, scan-count anomalies. | SQLite, haversine |
| **Verification pipeline** | Run every check, assemble an **evidence bundle** in which each fact carries an ID, then apply a deterministic **policy** that maps the evidence to a verdict. | Pure functions plus a policy table |
| **Evals** | A labelled scenario generator, graders and metrics. It writes a markdown report and gates CI on a regression baseline. | pytest-independent runner |
| **API and CLI** | REST API, a single-page demo UI, Prometheus metrics, and a CLI covering demo, issue, scan, eval, bench and audit. | FastAPI, Typer |

### 4.2 Key flows

**Issuance**

1. The manufacturer calls `POST /v1/batches {sku, count}`.
2. For each unit, Sigil generates a random 96-bit serial and builds the
   credential `{v, kid, sku, batch, serial, iat}`, then signs it with Ed25519.
3. The credential is appended to the transparency log, and the tree head is
   re-signed.
4. Sigil renders the label: the QR (payload `SG1:` + base45) next to the CDP,
   with a quiet zone and a fixed geometry relative to the QR.
5. **QC sampling.** A few labels per batch are captured through the channel
   (in production, a QC camera on the line) to record the batch's genuine CDP
   score distribution. Printers drift, and this baseline is what lets the
   verifier tell "worn printer" apart from "copy".

**Verification (fast path, target p95 < 150 ms on CPU, excluding upload)**

1. Image → quality gate → QR decode → credential decode.
2. In parallel:
   - signature check
   - key status check
   - log inclusion check
   - CDP crop → batched CNN score plus classical correlation score
   - clone-risk lookup
3. Evidence bundle → policy → verdict. The scan and its evidence are stored.

**Audit**

A witness fetches successive STHs and checks a consistency proof between each
pair. A consumer app can also ask for an inclusion proof and check it
client-side. A rewritten history is detected without trusting the operator.

## 5. Why these choices

| Decision | Alternatives | Rationale |
|---|---|---|
| Transparency log instead of a blockchain | Public chain (2022), private chain | Gives the property we need (public, append-only, verifiable issuance) at near-zero cost. It is the model Certificate Transparency and Sigstore run on at internet scale. No gas, no wallets, no consensus. |
| Ed25519 + base45 QR | ECDSA P-256, JWT/COSE | Small signatures (64 B) keep the QR at a low version, which is faster and more robust to scan. Base45 uses QR alphanumeric mode (EU DCC precedent). |
| CDP with a secret, keyed template | Holograms, RFID, NFC tags | Costs nothing to print. The verifier knows the template and the attacker does not. Copying is a lossy channel, which is information-theoretically hard to beat. |
| Two-input CNN (capture + template) | Single-input classifier, pure correlation | Conditioning on the template turns the job into comparing the capture against what *should* be there, and that generalises across serials. The classical correlation score is kept as an independent signal and as an eval baseline. |
| ONNX Runtime + micro-batching | Torch in-process, Triton | Inference has no torch dependency and starts fast. Batching amortises per-call overhead under concurrent load. The batcher is about 150 lines of code we can explain end to end. |
| A fixed rule table for the verdict | A learned classifier over all the evidence | Cryptographic and registry failures must never be outvoted by a score, so the order of the rules is the precedence. Every verdict names the rule that produced it and the evidence it rests on. |
| SQLite | Postgres | Zero ops, and enough for one node. Storage sits behind small repository classes. |

## 6. Threat model (abridged)

| Threat | Mitigation | Residual risk |
|---|---|---|
| Photocopy or rephotograph of a label | CDP score (CNN + correlation) | High-end reprint at native resolution: a lower margin, which lands in the ambiguous band and feeds clone analytics |
| Mass cloning of one genuine label | Impossible-travel and dispersion analytics. A per-serial scan budget. | First few clone scans pass. The alert surfaces to the issuer. |
| Forged credential | Ed25519 verification against pinned issuer keys | None without the key |
| Issuer key compromise | Key IDs, revocation with timestamp, and log monitoring for unexpected issuance | Credentials minted before detection |
| Log operator rewrites or forks history | Signed tree heads plus consistency proofs, checked by witnesses | Split view without gossip (documented) |
| Label text (SKU, batch) is attacker-controlled | It is stored only under an `untrusted` field of the evidence, never in a field the policy reads, and the UI shows it as quoted, unverified text. | Display only |
| DoS on inference | Bounded queue, per-request timeout, `429` with `Retry-After` | — |

## 7. Quality attributes

| Attribute | Target | How it is verified |
|---|---|---|
| Fast-path latency | p95 < 150 ms on a 4-core CPU at 16 concurrent scans | `sigil bench` (reported in the README) |
| Inference throughput | ≥ 3× the unbatched baseline at 32 concurrent requests | `sigil bench --compare-batching` |
| CDP discrimination | AUC ≥ 0.97 on a held-out channel distribution (CNN). The classical baseline is reported alongside. | `training/` report + eval |
| False-reject rate on genuine labels | ≤ 1 % at the operating point | Eval suite |
| Counterfeit recall | ≥ 97 % across all attack scenarios | Eval suite, gated in CI |
| Safety | 0 counterfeits passed as `GENUINE` across the scenario suite | Eval suite, gated in CI |
| Test coverage | Property tests for Merkle proofs and codecs. Unit tests per module. An end-to-end API test. | `pytest`, CI |

## 8. Observability

- **Metrics** (Prometheus):
  - scan verdict counts
  - fast-path latency histogram
  - inference queue depth, batch-size histogram and latency
- **Stored scans**: every scan is persisted with its verdict, the rule that
  fired and the full evidence bundle, and the API serves it back.
- **Eval reports**: markdown plus JSON in `reports/`, with a confusion matrix,
  per-scenario accuracy and the latency distribution.

## 9. Deployment

One container (`uvicorn`, which includes the inference service) with a mounted
data volume. `docker compose up` runs the API and seeds a demo issuer. The
inference service is an in-process module behind a stable interface, so it can
be split into its own deployment without touching callers. LLD §5.5 covers this.

## 10. Risks

| Risk | Plan |
|---|---|
| The simulated channel is not representative of real phones and printers | Parameterise the channel. Train on a broad randomised range and test on a *disjoint* range. Document the limitation. The calibration hooks (QC sampling) are exactly what a real deployment would use. |
| QR decoding fails on degraded captures | The quality gate returns `RECAPTURE` with a reason instead of guessing. |
| Friction on genuine items | The policy is conservative, so ambiguous genuine labels are `SUSPICIOUS` or `RECAPTURE`. The eval suite reports the friction rate next to the false accept rate. |

## 11. Roadmap beyond v1

- A witness/gossip network for STHs, and log sharding by issuer.
- Active learning: confirmed counterfeit captures feed retraining.
- Move CDP pre-processing (correlation and input building) off the event loop
  into the batch worker, so batching can approach the ≥ 3× throughput target.
