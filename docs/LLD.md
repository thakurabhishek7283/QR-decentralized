# Sigil — Low-Level Design

> Module contracts, data formats, algorithms and test plans. These are the
> interfaces each package is built against. Read [HLD.md](HLD.md) first.

---

## 0. Conventions

- **Python 3.12+**, managed with **uv**, `src/` layout, package `sigil`.
- **Typing.** Fully typed. `mypy --strict` must pass for `sigil.crypto` and
  `sigil.ledger`. Every other package must pass plain `mypy`.
- **Lint and format:** `ruff check` and `ruff format`, line length 100.
- **Value objects.** Use pydantic v2 models for anything crossing a process or
  API boundary. Internal value objects use `@dataclass(frozen=True, slots=True)`.
- **Time.** Only `datetime` objects that carry a timezone (UTC), or integer
  Unix seconds/milliseconds where the format says so.
- **Errors.** Domain code raises subclasses of `sigil.errors.SigilError`, never
  `HTTPException`. The API layer translates them.
- **Randomness.** Every stochastic function takes an explicit
  `rng: numpy.random.Generator`. No global RNG state.
- **Import layering.** An arrow means "may import":

  ```
  api, cli ─► evals ─► verify ─► {vision, inference, cdp, crypto, ledger, scans, issuance}
  ```

  Domain packages never import `api` or `cli`.
- **Storage.** One SQLite file, `{data_dir}/sigil.db`, opened in WAL mode
  through `sigil.storage.Database`. Each repository class creates its own
  tables (`CREATE TABLE IF NOT EXISTS`) in `__init__`.

## 1. Repository layout

```
pyproject.toml  uv.lock  Makefile  Dockerfile  docker-compose.yml  README.md
docs/            HLD.md  LLD.md  MODEL_CARD.md  adr/
src/sigil/
  __init__.py   errors.py   config.py
  storage/      db.py
  crypto/       base45.py  keys.py  credential.py
  ledger/       merkle.py  store.py  log.py  auditor.py
  cdp/          pattern.py  channel.py  features.py
  vision/       geometry.py  label.py  capture.py  quality.py
  inference/    preprocess.py  scorer.py  batcher.py  service.py  bench.py
  assets/       cdp_cnn.onnx  cdp_cnn.json
  scans/        store.py  clone.py
  issuance/     issuer.py  store.py  qc.py
  verify/       evidence.py  policy.py  pipeline.py
  evals/        scenarios.py  world.py  graders.py  runner.py  report.py  baseline.json
  api/          app.py  container.py  routes/*.py  schemas.py  static/index.html
  cli.py
training/       train_cdp.py  dataset.py  README.md
tests/          mirrors src/sigil/*
```

---

## 2. Shared infrastructure

### 2.1 `sigil.errors`

```python
class SigilError(Exception): ...
class DecodeError(SigilError): ...          # malformed QR payload / credential bytes
class SignatureError(SigilError): ...
class UnknownKeyError(SigilError): ...
class LogError(SigilError): ...
class NotFoundError(SigilError): ...
class CaptureError(SigilError):             # vision could not extract what it needs
    reason: str                              # machine code: "no_qr", "low_quality", "align_failed"
class OverloadedError(SigilError): ...      # inference queue full
```

### 2.2 `sigil.config.Settings`

`Settings` is a `pydantic_settings.BaseSettings` with prefix `SIGIL_`.

| Field | Default | Meaning |
|---|---|---|
| `data_dir` | `./.sigil` | Keys and DB |
| `issuer_name` | `"Acme Demo Co"` | Demo issuer |
| `api_key` | `"dev-key"` | Required on issuer endpoints (`X-API-Key`) |
| `cdp_genuine_min` | `0.966` | Policy: CDP score at or above this can pass as genuine (the model's validated operating point) |
| `cdp_counterfeit_max` | `0.20` | Policy: CDP score at or below this is a confident copy |
| `clone_risk_high` | `0.70` | Policy: `SUSPICIOUS` at or above this |
| `clone_risk_low` | `0.40` | Policy: `GENUINE` requires clone risk below this |
| `batch_max_size` | `32` | Micro-batcher |
| `batch_max_delay_ms` | `4.0` | Micro-batcher |
| `batch_queue_limit` | `512` | Micro-batcher backpressure |
| `inference_threads` | `0` | ORT intra-op threads (0 = ORT default) |
| `qc_samples_per_batch` | `6` | |

### 2.3 `sigil.storage.Database`

```python
class Database:
    def __init__(self, path: Path | str) -> None   # ":memory:" allowed; WAL, foreign_keys=ON
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor
    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None
    def query(self, sql: str, params=()) -> list[sqlite3.Row]
    def query_one(self, sql: str, params=()) -> sqlite3.Row | None
    @contextmanager
    def transaction(self) -> Iterator[None]        # BEGIN IMMEDIATE … COMMIT/ROLLBACK
    def close(self) -> None
```

It holds one connection opened with `check_same_thread=False`, guarded by a
`threading.RLock`. Calls are short, so async code calls it directly.

---

## 3. Trust core

### 3.1 `crypto.base45` — RFC 9285

```python
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
def b45encode(data: bytes) -> str
def b45decode(text: str) -> bytes        # raises DecodeError on bad char / bad length / overflow (>65535)
```

**Tests:** RFC vectors (`"AB"` ↔ `"BB8"`, `"Hello!!"` ↔ `"%69 VD92EX0"`,
`"base-45"` ↔ `"UJCLQE7W581"`), and a Hypothesis round trip.

### 3.2 `crypto.keys`

```python
KID_LEN = 4
def kid_for(public_raw: bytes) -> bytes                  # sha256(public_raw)[:4]

@dataclass(frozen=True, slots=True)
class KeyRecord:
    kid: bytes
    public_raw: bytes                                     # 32 bytes Ed25519
    purpose: Literal["issuer", "log"]
    created_at: datetime
    revoked_at: datetime | None = None
    def kid_hex(self) -> str

class Signer(Protocol):
    kid: bytes
    def sign(self, message: bytes) -> bytes               # 64 bytes
    def public_raw(self) -> bytes

class Ed25519Signer:                                      # implements Signer
    @classmethod
    def generate(cls) -> "Ed25519Signer"
    @classmethod
    def from_pem(cls, pem: bytes) -> "Ed25519Signer"
    def to_pem(self) -> bytes

def verify_signature(public_raw: bytes, message: bytes, signature: bytes) -> bool

class KeyRing:
    """Public keys known to the verifier. Thread-safe, read-mostly."""
    def add(self, rec: KeyRecord) -> None
    def get(self, kid: bytes) -> KeyRecord                # raises UnknownKeyError
    def revoke(self, kid: bytes, at: datetime) -> KeyRecord
    def records(self) -> list[KeyRecord]

class FileKeyStore:
    """Dev key provider. Layout under {data_dir}/keys/:
         issuer.pem, log.pem    — Ed25519 private keys (PKCS8 PEM, unencrypted, chmod 600)
         cdp.key                — 32 random bytes (the CDP template secret)
         keyring.json           — [{kid, public_b64, purpose, created_at, revoked_at}]
    """
    def __init__(self, root: Path) -> None
    def load_or_create(self) -> "KeyMaterial"             # idempotent

@dataclass(frozen=True)
class KeyMaterial:
    issuer: Ed25519Signer
    log: Ed25519Signer
    cdp_key: bytes
    keyring: KeyRing
```

### 3.3 `crypto.credential`

The binary format is big-endian and canonical: there is exactly one encoding
per credential.

```
offset size  field
0      2     magic  b"SG"
2      1     version = 1
3      4     kid
7      12    serial (random)
19     4     iat (uint32, unix seconds)
23     1     len(sku)    (1..32), followed by sku   (printable ASCII 0x20–0x7E)
..     1     len(batch)  (1..16), followed by batch (printable ASCII)
..     64    Ed25519 signature over  DOMAIN || body,   DOMAIN = b"sigil-credential-v1\x00"
```

```python
QR_PREFIX = "SG1:"

@dataclass(frozen=True, slots=True)
class Credential:
    kid: bytes; serial: bytes; iat: int; sku: str; batch: str
    version: int = 1
    def body(self) -> bytes
    @property
    def serial_hex(self) -> str

@dataclass(frozen=True, slots=True)
class SignedCredential:
    credential: Credential
    signature: bytes
    def to_bytes(self) -> bytes                           # body || sig
    @classmethod
    def from_bytes(cls, raw: bytes) -> "SignedCredential" # strict: DecodeError on any deviation, trailing bytes included
    def to_qr_payload(self) -> str                        # QR_PREFIX + b45encode(to_bytes())
    @classmethod
    def from_qr_payload(cls, text: str) -> "SignedCredential"
    def leaf_data(self) -> bytes                          # = to_bytes(); what the log commits to
    def verify(self, keyring: KeyRing) -> KeyRecord       # raises UnknownKeyError / SignatureError

def issue(signer: Signer, *, sku: str, batch: str, iat: int | None = None,
          serial: bytes | None = None) -> SignedCredential
```

**Tests:**
- Round trip through the QR payload.
- Flipping any byte → `SignatureError` or `DecodeError`.
- Bad lengths, non-ASCII input and trailing bytes → `DecodeError`.
- The maximum-size payload is 210 characters or fewer, so it fits QR version 9-M.

### 3.4 `ledger.merkle` — RFC 9162 §2.1

These are pure functions over `list[bytes]` leaf hashes, each 32 bytes.

```python
def leaf_hash(data: bytes) -> bytes                         # sha256(0x00 || data)
def node_hash(left: bytes, right: bytes) -> bytes           # sha256(0x01 || l || r)
def root_hash(leaves: Sequence[bytes]) -> bytes              # MTH; empty tree = sha256(b"")
def inclusion_proof(index: int, leaves: Sequence[bytes]) -> list[bytes]          # PATH(m, D[n])
def consistency_proof(m: int, leaves: Sequence[bytes]) -> list[bytes]            # PROOF(m, D[n])
def verify_inclusion(leaf: bytes, index: int, size: int, proof: Sequence[bytes], root: bytes) -> bool
def verify_consistency(m: int, n: int, root_m: bytes, root_n: bytes, proof: Sequence[bytes]) -> bool
```

The verifiers implement the exact algorithms in RFC 9162 §2.1.3.2 and §2.1.4.2.
They are iterative, they use bit operations on `fn`/`sn`, and they **never**
recompute the tree.

**Tests (Hypothesis, size 1 to 300):**
- Every inclusion proof and every consistency proof verifies.
- A mutated proof, the wrong index, or the wrong root fails.
- The RFC 6962 test vectors pass (the reference roots for the 8-leaf tree in
  Go's `trillian`/`transparency-dev` test data).

### 3.5 `ledger.store` and `ledger.log`

```sql
CREATE TABLE log_leaves (idx INTEGER PRIMARY KEY, leaf_hash BLOB NOT NULL UNIQUE,
                         data BLOB NOT NULL, appended_at INTEGER NOT NULL);
CREATE TABLE log_sth    (tree_size INTEGER PRIMARY KEY, root_hash BLOB NOT NULL,
                         timestamp_ms INTEGER NOT NULL, signature BLOB NOT NULL);
```

```python
@dataclass(frozen=True, slots=True)
class SignedTreeHead:
    tree_size: int; root_hash: bytes; timestamp_ms: int; signature: bytes; log_kid: bytes
    def signed_message(self) -> bytes   # b"sigil-sth-v1\x00" + size(u64) + ts(u64) + root
    def verify(self, log_public_raw: bytes) -> bool
    def to_json(self) -> dict[str, Any]  # hex/b64 encodings; inverse from_json

@dataclass(frozen=True, slots=True)
class InclusionProof:   leaf_index: int; tree_size: int; path: list[bytes]
@dataclass(frozen=True, slots=True)
class ConsistencyProof: first: int; second: int; path: list[bytes]

class TransparencyLog:
    def __init__(self, db: Database, signer: Signer) -> None   # loads leaf hashes into memory
    def append(self, data: bytes) -> int                      # returns leaf index; idempotent on duplicate data
    def append_many(self, items: Sequence[bytes]) -> list[int] # one transaction
    def publish(self) -> SignedTreeHead                        # sign & persist STH for current size (no-op if unchanged)
    def latest_sth(self) -> SignedTreeHead                     # publishes if none yet
    def sth_at(self, tree_size: int) -> SignedTreeHead         # NotFoundError if never published
    def index_of(self, leaf_hash: bytes) -> int | None
    def prove_inclusion(self, leaf_hash: bytes, tree_size: int | None = None) -> InclusionProof
    def prove_consistency(self, first: int, second: int) -> ConsistencyProof
    @property
    def size(self) -> int
```

Proofs are O(n) over the in-memory leaf list, which is fine at demo scale.
The docstring notes that production would persist the tile structure used by
tlog-tiles/Trillian instead.

`ledger.auditor.Auditor(log_public_raw, state_path)`:

```python
def observe(self, sth: SignedTreeHead, proof_from_prev: ConsistencyProof | None) -> AuditResult
```

`observe` verifies the STH signature. When a previous STH exists, it also
verifies consistency and refuses to go backwards in size. On success it
persists the new STH. It returns `AuditResult(ok, reason)`.

**Tests:**
- Append, publish and prove all verify.
- Reopening from the DB yields the same root.
- Tampering with a stored leaf is detected by the auditor (the consistency
  proof fails).
- The STH signature check passes with the right key and fails otherwise.

### 3.6 `scans.store` and `scans.clone`

```sql
CREATE TABLE scans (id TEXT PRIMARY KEY, serial_hex TEXT, ts INTEGER NOT NULL,
                    lat REAL, lon REAL, device_hash TEXT, verdict TEXT,
                    cdp_score REAL, payload_json TEXT);
CREATE INDEX scans_serial_ts ON scans(serial_hex, ts);
```

```python
@dataclass(frozen=True, slots=True)
class ScanEvent:
    id: str; serial_hex: str | None; ts: datetime; lat: float | None; lon: float | None
    device_hash: str | None; verdict: str | None = None; cdp_score: float | None = None

class ScanStore:
    def __init__(self, db: Database) -> None
    def add(self, ev: ScanEvent, payload: dict[str, Any] | None = None) -> None
    def update_verdict(self, scan_id: str, verdict: str) -> None
    def get(self, scan_id: str) -> tuple[ScanEvent, dict[str, Any] | None]
    def history(self, serial_hex: str, *, before: datetime | None = None,
                within: timedelta = timedelta(days=90), limit: int = 200) -> list[ScanEvent]

def hash_device(device_id: str, salt: bytes) -> str          # hmac-sha256 hex[:16]; raw IDs never stored

@dataclass(frozen=True, slots=True)
class CloneSignal:  name: str; weight: float; detail: str
@dataclass(frozen=True, slots=True)
class CloneAssessment:
    risk: float; signals: list[CloneSignal]; scans_considered: int

def assess_clone_risk(history: Sequence[ScanEvent], current: ScanEvent,
                      cfg: CloneConfig = CloneConfig()) -> CloneAssessment
```

`CloneConfig` defaults and the signals they drive:

| Signal | Condition (history plus current, last 30 days) | Weight |
|---|---|---|
| `impossible_travel` | Some pair of consecutive geo-tagged scans is more than 50 km apart with an implied speed above 900 km/h | 0.80 |
| `geo_dispersion` | At least 3 clusters 300 km apart (greedy clustering) | 0.50 |
| `scan_volume` | More than 25 scans | 0.30 |
| `device_diversity` | At least 10 distinct devices | 0.30 |

`risk = 1 − Π(1 − wᵢ)` over the signals that fire. `haversine_km(a, b)` is
exposed for tests.

**Tests:**
- Each signal fires on its constructed case.
- A single scan, or same-city repeats, gives risk 0.
- Risk is monotone in the number of signals.

---

## 4. Vision and CDP

### 4.1 Label geometry (`vision.geometry`) — fixed for v1

```python
QR_VERSION = 9            # 53 modules; alphanumeric capacity at EC level M is 262 chars
QR_ERROR = "M"
QR_MODULES = 53
MODULE_PX = 6             # label raster ≈ 600 dpi → 1 module ≈ 0.25 mm
QUIET_MODULES = 4
CDP_CELLS = 106           # 2 cells per QR module
CELL_PX = 3
SYMBOL_PX = QR_MODULES * MODULE_PX          # 318
CDP_PX = CDP_CELLS * CELL_PX                # 318
QUIET_PX = QUIET_MODULES * MODULE_PX        # 24
# Label canvas (white): [quiet][QR symbol][quiet][CDP][quiet] × [quiet][318][quiet]
LABEL_W = QUIET_PX * 3 + SYMBOL_PX + CDP_PX  # 708
LABEL_H = QUIET_PX * 2 + SYMBOL_PX           # 366
QR_ORIGIN = (QUIET_PX, QUIET_PX)                          # top-left of the symbol (x, y)
CDP_ORIGIN = (QUIET_PX * 2 + SYMBOL_PX, QUIET_PX)         # (366, 24)
PATCH_PX = 212            # model input side: 2 px per CDP cell
def qr_corners_label() -> np.ndarray       # float32 (4,2): TL, TR, BR, BL of the symbol, in label px
def cdp_corners_label() -> np.ndarray
```

Everything downstream works in this canonical **label frame**. A capture is
mapped back into it with a homography estimated from the QR corners.

### 4.2 `cdp.pattern`, `cdp.channel`

```python
def generate_template(cdp_key: bytes, serial: bytes, cells: int = CDP_CELLS) -> np.ndarray
```

`generate_template` returns a `uint8 (cells, cells)` array of 0/1 values, where
1 means ink. The bits are HMAC-SHA256(cdp_key, serial ‖ counter_be32) for
counter = 0, 1, …, unpacked MSB-first and truncated. Density is 0.5 in
expectation.

The channel simulates a physical print/scan with a parameterised model. **This
is the main modelling assumption of the project, and the README states it
openly.**

```python
@dataclass(frozen=True)
class PrintProfile:     # one printer + substrate
    dot_gain: float       # gaussian sigma (label px) of ink spread, typ 0.6–1.3
    ink_noise: float      # per-pixel threshold noise std, typ 0.05–0.15
    ink_density: float    # darkness of ink 0.75–0.95
    paper_grain: float    # low-freq substrate texture amplitude 0–0.06
    dropout: float        # fraction of cells randomly lost/added (worn heads) 0–0.03

@dataclass(frozen=True)
class CaptureProfile:   # one camera shot
    scale: float          # captured px per label px, 0.55–1.4
    perspective: float    # max corner jitter as a fraction of label width, 0–0.08
    rotation_deg: float
    defocus: float        # gaussian sigma (captured px), 0–1.6
    motion: float         # motion blur length px, 0–3
    noise: float          # sensor noise std (0..1 scale), 0.005–0.03
    gamma: float          # 0.7–1.4
    brightness: float     # multiplicative 0.7–1.2
    jpeg_quality: int     # 55–95
    canvas: tuple[int, int] = (1280, 960)

def print_label(label: np.ndarray, p: PrintProfile, rng) -> np.ndarray   # float32 [0,1] ink-on-paper
def capture(printed: np.ndarray, c: CaptureProfile, rng) -> np.ndarray   # uint8 gray photo (canvas size)

class Attack(StrEnum):
    PHOTOCOPY = "photocopy"     # office scan (300-dpi equiv, ~0.5x) → laser print (high dot gain)
    HQ_REPRINT = "hq_reprint"   # 1200-dpi scan (~2x) + denoise + re-binarise → good printer
    SCREEN = "screen"           # label photographed, then displayed on a phone screen and re-photographed

def counterfeit_from(genuine_printed: np.ndarray, attack: Attack, rng) -> np.ndarray
    # returns a new *printed* label (float [0,1]); the attacker never sees the template

Domain = Literal["train", "test"]
def sample_print_profile(rng, domain: Domain, quality: Literal["good", "worn"] = "good") -> PrintProfile
def sample_capture_profile(rng, domain: Domain, difficulty: Literal["easy", "normal", "hard"] = "normal") -> CaptureProfile
```

The **train** and **test** parameter ranges are deliberately different: test
has a wider blur range, lower JPEG quality and a different gamma range. This
measures generalisation rather than memorisation. `"worn"` print quality
raises the dot gain and the dropout.

### 4.3 `vision.label`

```python
def render_qr_matrix(payload: str) -> np.ndarray      # bool (53,53) via segno, version=9, error="m", boost_error=False
def render_label(payload: str, template: np.ndarray) -> np.ndarray   # float32 (LABEL_H, LABEL_W), 1=white, 0=ink
def label_png(label: np.ndarray) -> bytes
```

### 4.4 `vision.quality`

```python
@dataclass(frozen=True, slots=True)
class QualityReport:
    sharpness: float            # variance of Laplacian on the image (or ROI)
    brightness: float           # mean 0..1
    clipped: float              # fraction of pixels at 0 or 255
    ok: bool
    reasons: list[str]          # "blurry", "too_dark", "overexposed"
def assess_quality(gray: np.ndarray, roi: np.ndarray | None = None) -> QualityReport
```

### 4.5 `vision.capture`

```python
@dataclass(frozen=True)
class DecodedCapture:
    payload: str
    qr_corners: np.ndarray      # (4,2) in image px: TL, TR, BR, BL of the symbol
    homography: np.ndarray      # 3x3, image → label frame
    label_view: np.ndarray      # float32 rectified label frame (LABEL_H, LABEL_W), [0,1]
    quality: QualityReport

def load_image(data: bytes) -> np.ndarray                      # → uint8 gray; CaptureError("bad_image")
def decode(gray: np.ndarray) -> DecodedCapture                 # CaptureError("no_qr" | "low_quality")
def extract_cdp_patch(cap: DecodedCapture, template: np.ndarray) -> CdpPatch   # CaptureError("align_failed")

@dataclass(frozen=True)
class CdpPatch:
    patch: np.ndarray           # float32 (PATCH_PX, PATCH_PX), [0,1], ink dark
    alignment_score: float      # final ECC correlation
```

`decode` works as follows:

1. Run `cv2.QRCodeDetector().detectAndDecode` on the gray image.
2. If that fails, retry on a CLAHE-equalised version, then a 1.5× upscale, then
   `cv2.QRCodeDetectorAruco` when available.
3. Reorder the corners to TL, TR, BR, BL (checking orientation with the cross
   product).
4. Compute `H = getPerspectiveTransform(corners, qr_corners_label())` and warp
   to the label frame.
5. Assess quality on the rectified label.

`extract_cdp_patch` refines the alignment because QR corner estimates are only
good to about 1 px:

1. Crop the CDP region plus a 6 px margin from `label_view`, and resample it to
   2 px per cell.
2. Render the ideal template image: cells expanded ×2, with a slight Gaussian
   blur (σ = 0.8).
3. Run `cv2.findTransformECC` with an affine motion model, starting from the
   identity, with 50 iterations and eps 1e-4.
4. Warp to an exact `PATCH_PX` square.

ECC failure, or a final correlation below 0.2, raises
`CaptureError("align_failed")`.

**This exact function is used by training, serving and evals**, so train and
serve cannot skew.

### 4.6 `cdp.features` — the classical baseline

```python
def template_image(template: np.ndarray, px_per_cell: int = 2, blur: float = 0.8) -> np.ndarray
def correlation_score(patch: np.ndarray, template: np.ndarray) -> float
    # Pearson correlation between the patch (inverted so ink = high) and template_image;
    # mapped to [0,1] by a logistic calibrated in training (constants stored in cdp_cnn.json, defaults provided)
def raw_correlation(patch: np.ndarray, template: np.ndarray) -> float   # uncalibrated Pearson r
```

---

## 5. Inference

### 5.1 `inference.preprocess` — the single source of truth

```python
INPUT_SHAPE = (2, PATCH_PX, PATCH_PX)
def make_input(patch: np.ndarray, template: np.ndarray) -> np.ndarray   # float32 (2,212,212)
    # ch0: patch → invert (ink=1) → per-sample standardise (mean 0, std 1, eps 1e-6)
    # ch1: template_image(template) mapped to {-1,+1} (ink=+1)
```

### 5.2 Model and training (`training/`)

The CNN is about 60k parameters. All convolutions have no bias and are
followed by BN and ReLU:

| Stage | Layer | Output |
|---|---|---|
| 1 | conv 2→16, 5×5, stride 2 | 106² |
| 2 | conv 16→32, 3×3, stride 2 | 53² |
| 3 | conv 32→64, 3×3, stride 2 | 27² |
| 4 | conv 64→64, 3×3, stride 2 | 14² |
| 5 | global average pool → linear 64→1 | logit |

The input is the 2 × 212 × 212 tensor from §5.1, and the output is the logit
for P(genuine).

**Data.** Generated on the fly by the full pipeline:

1. `render_label` → `print_label` → (`counterfeit_from`) → `capture` →
   `decode` → `extract_cdp_patch` → `make_input`.
2. Classes are balanced, and the negatives are split evenly across the attacks.
3. Generation runs across processes with a deterministic seed per sample.
   Samples whose decode fails are dropped and counted.

**Splits:**
- *train:* 24k samples from the train domain
- *val:* 4k from the train domain
- *test:* 6k from the **test** domain, with disjoint serials and keys

**Training.** AdamW (lr 2e-3, wd 1e-4), cosine schedule, 12 epochs, batch 128,
light augmentation (±1 px shifts, brightness jitter).

**Output:**
- `src/sigil/assets/cdp_cnn.onnx` (opset 17, dynamic batch axis, input name
  `input`, output name `logit`)
- `cdp_cnn.json` with `{version, sha256, input_shape, trained_at, metrics:{auc_test, auc_test_by_attack, eer_test, baseline_corr_auc_test}, corr_calibration:{a,b}, decode_failure_rate}`
- `docs/MODEL_CARD.md`, generated from the training run

`training/` depends on the `train` extra (torch, onnx). Runtime only needs
`onnxruntime`.

### 5.3 `inference.scorer`

```python
class CdpScorer(Protocol):
    def predict_logits(self, batch: np.ndarray) -> np.ndarray   # (N,2,212,212) → (N,)

class OnnxCdpScorer:
    def __init__(self, model_path: Path | None = None, threads: int = 0) -> None  # default: packaged asset
    def predict_logits(self, batch: np.ndarray) -> np.ndarray
    metadata: dict[str, Any]

def sigmoid(x: np.ndarray) -> np.ndarray
```

### 5.4 `inference.batcher` — dynamic micro-batching

```python
class MicroBatcher:
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], *, max_batch_size: int = 32,
                 max_delay_ms: float = 4.0, queue_limit: int = 512,
                 metrics: BatcherMetrics | None = None) -> None
    async def start(self) -> None
    async def stop(self) -> None                 # drains: in-flight items complete, then the worker exits
    async def submit(self, item: np.ndarray, timeout_s: float | None = 5.0) -> np.ndarray   # result row
    @property
    def queue_depth(self) -> int
```

Algorithm, run by a single worker task:

```
loop:
  first = await queue.get()
  batch = [first]; deadline = now + max_delay
  while len(batch) < max_batch_size:
      remaining = deadline - now
      if remaining <= 0: break
      try: batch.append(await wait_for(queue.get(), remaining))   # or get_nowait fast path
      except TimeoutError: break
  live = [r for r in batch if not r.future.done()]                 # caller timed out / cancelled
  X = np.stack([r.item for r in live])
  Y = await loop.run_in_executor(executor, fn, X)                  # 1 dedicated thread: ORT is the parallelism
  for r, y in zip(live, Y): r.future.set_result(y)
  on exception: set_exception on every live future
```

- `submit` raises `OverloadedError` immediately when the queue is full. That is
  the backpressure, and the API maps it to `429` with `Retry-After: 1`.
- `max_batch_size=1` is the unbatched baseline used in benchmarks.
- `BatcherMetrics` holds Prometheus `Histogram`s for batch size, queue wait
  and compute time, plus a `Gauge` for queue depth. It accepts an optional
  registry so tests stay isolated.

**Tests (async):**
- Results map back to the right callers.
- Batches never exceed the maximum.
- A lone request completes within roughly `max_delay` plus compute time.
- A full queue raises `OverloadedError`.
- A function exception propagates to every caller in that batch.
- A cancelled caller doesn't break the worker.
- `stop()` drains.

### 5.5 `inference.service`

```python
@dataclass(frozen=True, slots=True)
class CdpScore:
    cnn: float           # sigmoid(logit) ∈ [0,1]
    correlation: float   # calibrated classical score ∈ [0,1]
    raw_correlation: float
    alignment: float

class InferenceService:
    def __init__(self, scorer: CdpScorer, batcher_cfg: BatcherConfig, metrics: ... | None = None) -> None
    async def start(self) -> None; async def stop(self) -> None
    async def score(self, patch: CdpPatch, template: np.ndarray) -> CdpScore
```

This is the seam for moving inference out of process later. A
`RemoteInferenceService` would expose the same `score` signature.

### 5.6 `inference.bench`

```python
async def run_benchmark(service_factory, *, concurrency: int, duration_s: float,
                        warmup_s: float = 1.0) -> BenchResult
# BenchResult: requests, throughput_rps, p50_ms, p95_ms, p99_ms, mean_batch_size
def compare_batching(concurrency_levels=(1, 8, 32), duration_s=5.0) -> list[BenchRow]
```

The inputs are pre-generated real patches, so the benchmark measures inference
rather than simulation. The CLI prints a table and writes `reports/bench.md`.

---

## 6. Issuance

```sql
CREATE TABLE batches (batch_id TEXT PRIMARY KEY, sku TEXT NOT NULL, created_at INTEGER NOT NULL,
                      print_quality TEXT NOT NULL DEFAULT 'good',
                      qc_n INTEGER, qc_mean REAL, qc_std REAL, qc_min REAL);
CREATE TABLE units   (serial_hex TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches,
                      credential BLOB NOT NULL, leaf_index INTEGER NOT NULL, issued_at INTEGER NOT NULL);
```

```python
@dataclass(frozen=True, slots=True)
class IssuedUnit: serial_hex: str; payload: str; leaf_index: int
@dataclass(frozen=True, slots=True)
class BatchInfo:  batch_id: str; sku: str; created_at: datetime; print_quality: str
                  qc: QcStats | None
@dataclass(frozen=True, slots=True)
class QcStats:    n: int; mean: float; std: float; min: float

class IssuanceStore:
    def __init__(self, db: Database) -> None
    def add_batch(...) / add_units(...) / set_qc(batch_id, QcStats)
    def batch(self, batch_id) -> BatchInfo
    def unit(self, serial_hex) -> tuple[SignedCredential, str]   # credential, batch_id
    def units(self, batch_id) -> list[str]

class Issuer:
    def __init__(self, keys: KeyMaterial, log: TransparencyLog, store: IssuanceStore) -> None
    def issue_batch(self, sku: str, count: int, *, batch_id: str | None = None,
                    print_quality: str = "good") -> tuple[BatchInfo, list[IssuedUnit]]
        # signs all, append_many to the log, publish STH, persist units. batch_id default: "B" + 6 hex.
    def template_for(self, serial: bytes) -> np.ndarray
    def render_label(self, serial_hex: str) -> np.ndarray

async def run_qc(issuer, inference: InferenceService, batch_id: str, *, samples: int,
                 rng, domain="test") -> QcStats
    # simulates the production-line QC camera: prints the batch's labels using a print profile
    # drawn for its print_quality, captures under "easy" conditions, scores CNN, stores stats.
```

---

## 7. Verification

### 7.1 `verify.evidence`

```python
class EvidenceKind(StrEnum):
    QUALITY = "quality"; DECODE = "decode"; SIGNATURE = "signature"; KEY_STATUS = "key_status"
    LOG_INCLUSION = "log_inclusion"; CDP = "cdp"; CLONE = "clone"

class Evidence(BaseModel):
    id: str                       # stable: "quality", "decode", "signature", "key", "log", "cdp", "clone"
    kind: EvidenceKind
    passed: bool | None           # None = informational
    summary: str                  # one human-readable line, ≤ 200 chars, no untrusted free text
    data: dict[str, Any]          # numbers and enums only; untrusted strings live under data["untrusted"]

class EvidenceBundle(BaseModel):
    scan_id: str
    serial_hex: str | None
    items: list[Evidence]
    def get(self, id: str) -> Evidence | None
    def add(self, ev: Evidence) -> None     # replaces same id
```

### 7.2 `verify.policy`

```python
class Verdict(StrEnum): GENUINE, COUNTERFEIT, SUSPICIOUS, RECAPTURE

@dataclass(frozen=True)
class PolicyConfig: cdp_genuine_min, cdp_counterfeit_max, clone_risk_high, clone_risk_low  (from Settings)

@dataclass(frozen=True)
class PolicyDecision:
    verdict: Verdict
    confidence: float
    rule: str                     # id of the rule that fired, e.g. "R3_signature"
    reasons: list[str]
    hard_fail: bool               # cryptographic/registry failure

def decide(bundle: EvidenceBundle, cfg: PolicyConfig) -> PolicyDecision
```

The rules are evaluated in order, and the first match wins. `s` is the CDP
score and `r` the clone risk.

| Rule | Condition | Verdict | Hard |
|---|---|---|---|
| R1_capture | quality not ok, or decode failed with `no_qr`/`low_quality` | RECAPTURE | – |
| R2_malformed | payload present but not a valid credential | COUNTERFEIT | ✓ |
| R3_signature | unknown kid or invalid signature | COUNTERFEIT | ✓ |
| R4_revoked_after | key revoked and `iat ≥ revoked_at` | COUNTERFEIT | ✓ |
| R5_unlogged | credential not in the log | COUNTERFEIT | ✓ |
| R6_align | CDP alignment failed | RECAPTURE | – |
| R7_copy | `s ≤ cdp_counterfeit_max` | COUNTERFEIT | – |
| R8_revoked_before | key revoked and `iat < revoked_at` | SUSPICIOUS | – |
| R9_clone | `r ≥ clone_risk_high` | SUSPICIOUS | – |
| R10_genuine | `s ≥ cdp_genuine_min` and `r < clone_risk_low` | GENUINE | – |
| R11_ambiguous | otherwise | SUSPICIOUS | – |

Here `s` is `cdp.data["cnn"]`. Confidence comes from the margins:
- `GENUINE`: `0.5 + 0.5·(s − min)/(1 − min)`
- R7: `0.5 + 0.5·(max − s)/max`
- hard fails: 0.99
- `SUSPICIOUS`: 0.5

**Tests:** a table-driven test per rule, plus rule-precedence tests.

### 7.3 `verify.pipeline`

```python
@dataclass(frozen=True)
class ScanContext:
    ts: datetime; lat: float | None = None; lon: float | None = None; device_id: str | None = None

class VerificationResult(BaseModel):
    scan_id: str
    verdict: Verdict
    confidence: float
    rule: str
    reasons: list[str]
    serial_hex: str | None
    sku: str | None               # untrusted display string
    batch_id: str | None
    evidence: EvidenceBundle
    timings_ms: dict[str, float]  # decode, crypto, log, cdp, clone, total

class VerificationPipeline:
    def __init__(self, *, keys: KeyRing, cdp_key: bytes, log: TransparencyLog, issuance: IssuanceStore,
                 scans: ScanStore, inference: InferenceService, policy: PolicyConfig,
                 device_salt: bytes) -> None
    async def verify(self, image: np.ndarray | bytes, ctx: ScanContext, *, record: bool = True) -> VerificationResult
```

Order of checks:

1. load → `decode` → quality evidence
2. credential parse → `signature` → `key`
3. `log` (looks up `leaf_hash(leaf_data)`, and attaches the inclusion proof
   against the latest STH, verified)
4. `extract_cdp_patch` → `inference.score` → `cdp`
5. clone assessment over the history (excluding this scan) → `clone`

Steps 4 and 5 run concurrently with `asyncio.gather`. A failure short-circuits
only the checks that depend on it, so a bad signature still produces a full
bundle with the remaining items marked not-run. The scan is persisted with its
verdict, and every timing is recorded.

---

## 8. Evals (`sigil.evals`)

### 8.1 Scenarios

```python
class ScenarioKind(StrEnum):
    GENUINE_CLEAN, GENUINE_HARD_CAPTURE, GENUINE_WORN_PRINTER, GENUINE_CLONE_VICTIM,
    PHOTOCOPY, HQ_REPRINT, SCREEN_REPLAY, CLONED_SERIAL, FORGED_SIGNATURE, ROGUE_UNLOGGED

@dataclass(frozen=True)
class Scenario:
    id: str; kind: ScenarioKind; seed: int
    truth: Literal["GENUINE", "COUNTERFEIT"]
    acceptable: frozenset[Verdict]       # verdicts graded as correct
    ideal: Verdict
```

| Kind | Construction | Acceptable verdicts (ideal first) |
|---|---|---|
| genuine_clean | Good printer, easy/normal capture | GENUINE |
| genuine_hard_capture | Good printer, hard capture | GENUINE, RECAPTURE, SUSPICIOUS |
| genuine_worn_printer | Batch printed `"worn"`, with QC baseline; normal capture | GENUINE, SUSPICIOUS |
| genuine_clone_victim | Genuine item; history shows its serial scanned elsewhere 3 h ago | SUSPICIOUS, GENUINE |
| photocopy | `Attack.PHOTOCOPY` | COUNTERFEIT, SUSPICIOUS |
| hq_reprint | `Attack.HQ_REPRINT` | COUNTERFEIT, SUSPICIOUS |
| screen_replay | `Attack.SCREEN` | COUNTERFEIT, SUSPICIOUS, RECAPTURE |
| cloned_serial | hq_reprint plus impossible-travel history | COUNTERFEIT, SUSPICIOUS |
| forged_signature | Credential signed by an attacker key that claims the issuer's kid | COUNTERFEIT |
| rogue_unlogged | Signed with the real issuer key but never logged (insider/stolen key) | COUNTERFEIT |

**`GENUINE` is never acceptable for a counterfeit, and `COUNTERFEIT` is never
acceptable for a genuine item.**

`world.ScenarioWorld` builds an isolated world per case:
- a temp data dir and keys
- the log and stores
- one shared `InferenceService` for the whole eval run
- the issued batch with QC
- the item's printed label (genuine or attacked)
- the capture
- seeded scan history

### 8.2 Runner and metrics

```python
@dataclass
class CaseResult:
    scenario: Scenario; verdict: Verdict; correct: bool; ideal: bool
    latency_ms: float; rule: str; cdp_cnn: float | None; cdp_correlation: float | None

async def run_suite(*, seeds_per_kind: int, out_dir: Path,
                    kinds: Sequence[ScenarioKind] | None = None, concurrency: int = 4) -> SuiteResult
```

`SuiteResult.metrics`:

| Metric | Definition |
|---|---|
| `accuracy` | Fraction of cases whose verdict is acceptable |
| `decided_accuracy` | The same, excluding cases that ended in RECAPTURE |
| `recapture_rate` | Fraction of cases that ended in RECAPTURE |
| `false_accept_rate` | Counterfeits whose verdict is GENUINE, over all counterfeits. **The safety metric.** |
| `false_reject_rate` | Genuine items whose verdict is COUNTERFEIT, over all genuine items |
| `friction_rate` | Genuine items whose verdict is SUSPICIOUS or RECAPTURE |
| `p95_latency_ms` | End-to-end verification latency under the suite's concurrency |
| `by_kind` | Accuracy per scenario kind |
| `cdp_auc` | AUC of the CNN score, and the same for the correlation baseline, over primary captures |

### 8.3 Gate and report

- `baseline.json` holds threshold values: `false_accept_rate_max`,
  `false_reject_rate_max`, `accuracy_min`, `decided_accuracy_min`,
  `friction_rate_max`.
- `sigil eval --gate` exits non-zero on any breach and prints which metric
  failed.
- `report.write(suite, out_dir)` writes `report.md` and `results.json`. The
  report includes:
  - a headline table
  - a confusion matrix of truth against verdict
  - per-kind accuracy
  - the CDP AUC over the primary captures

---

## 9. API (`sigil.api`)

`create_app(settings: Settings | None = None) -> FastAPI` uses a lifespan that
builds a `Container`:
- keys (`FileKeyStore`)
- the DB
- the log, issuance, scans, inference (started) and pipeline
- a demo batch seeded if none exist

| Method & path | Auth | Body / params | Response |
|---|---|---|---|
| GET `/healthz` | – | – | `{status, log_size, model_version}` |
| GET `/v1/keys` | – | – | `[{kid, purpose, public_key (b64url), created_at, revoked_at}]` |
| POST `/v1/batches` | API key | `{sku, count ≤ 200, print_quality}` | `{batch_id, serials, qc}` |
| GET `/v1/batches/{id}` | – | – | Batch info plus serials |
| GET `/v1/labels/{serial}.png` | – | – | `image/png` |
| POST `/v1/simulate/{serial}` | – | `{attack: null \| photocopy \| hq_reprint \| screen, difficulty, seed?}` | `image/jpeg` simulated phone photo (demo only) |
| POST `/v1/scans` | – | multipart `image`; form `lat`, `lon`, `device_id` | `VerificationResult` |
| GET `/v1/scans/{id}` | – | – | Stored result and the verdict it was recorded with |
| GET `/v1/log/sth` | – | – | Latest STH JSON |
| GET `/v1/log/inclusion` | – | `serial` or `leaf_hash` (hex), `tree_size?` | `{leaf_index, tree_size, path[]}` (hex) |
| GET `/v1/log/consistency` | – | `first`, `second` | `{path[]}` |
| POST `/v1/keys/{kid}/revoke` | API key | `{at?}` | Key record |
| GET `/metrics` | – | – | Prometheus text |
| GET `/` | – | – | Demo SPA (vanilla JS, one HTML file) |

Error mapping:

| Error | Status |
|---|---|
| `NotFoundError` | 404 |
| `CaptureError` | 422, `{reason}` |
| `OverloadedError` | 429 with `Retry-After` |
| `DecodeError` (in upload parsing) | 422 |

Uploads are capped at 8 MB.

The demo page walks through it step by step:
1. Issue a batch.
2. Pick a label (rendered).
3. "Photograph" it, either genuine or by attack, through the simulator.
4. Scan it.
5. See the verdict, the evidence table and the timings.

It also supports uploading your own photo.

## 10. CLI (`sigil.cli`, Typer)

```
sigil serve [--host --port]
sigil demo                         # narrated end-to-end run in the terminal (rich)
sigil issue --sku X --count N
sigil scan IMAGE [--lat --lon --device]
sigil simulate SERIAL [--attack photocopy] -o out.jpg
sigil eval [--seeds 3] [--kinds …] [--gate] [--out reports/eval]
sigil bench [--concurrency 1,8,32] [--duration 5]
sigil audit [--url http://localhost:8000]   # witness: fetch STH + consistency proof, verify, persist
```

## 11. Quality gates (CI: `.github/workflows/ci.yml`)

| Job | Commands |
|---|---|
| lint | `ruff check`, `ruff format --check` |
| typecheck | `mypy` (strict packages per §0) |
| test | `pytest -q` (unit, property, async, API end to end via `httpx.AsyncClient` + ASGI transport) |
| eval | `sigil eval --seeds 3 --gate`, report uploaded as an artifact |
| bench-smoke | `sigil bench --concurrency 8 --duration 2` (no assertion, artifact only) |
| docker | `docker build` |

## 12. Test plan summary

| Package | Notable tests |
|---|---|
| crypto | RFC 9285 vectors, codec fuzz, signature tamper |
| ledger | Hypothesis proof properties, RFC vectors, persistence, auditor detects rewrite |
| scans | Each clone signal, haversine |
| cdp/vision | Template determinism and keyed uniqueness, decode on clean and perturbed renders, `extract_cdp_patch` alignment on a shifted capture, correlation separates genuine from photocopy on average |
| inference | Batcher semantics (§5.4), ONNX scorer shape/dtype, service end to end |
| verify | Policy table, pipeline over genuine, photocopy and forged captures |
| evals | Small suite smoke (1 seed, 3 kinds) |
| api | Issue → simulate → scan, log proofs verify client-side, 429 on overload |
