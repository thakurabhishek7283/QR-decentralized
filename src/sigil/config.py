"""Runtime configuration, read from ``SIGIL_*`` environment variables or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIGIL_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".sigil")
    issuer_name: str = "Acme Demo Co"
    api_key: str = "dev-key"

    # Verification policy thresholds (LLD §7.2). ``cdp_genuine_min`` is the model's validated
    # operating point (≤ 0.5 % of counterfeits above it; see docs/MODEL_CARD.md). The copy
    # threshold is deliberately stricter than the validated 0.42: calling a customer's genuine
    # item counterfeit from a single photo is the costliest error, so the band in between goes
    # to the investigator, which can ask for a retake.
    cdp_genuine_min: float = 0.966
    cdp_counterfeit_max: float = 0.20
    clone_risk_high: float = 0.70
    clone_risk_low: float = 0.40

    # Inference micro-batcher (LLD §5.4).
    batch_max_size: int = 32
    batch_max_delay_ms: float = 4.0
    batch_queue_limit: int = 512
    inference_threads: int = 0

    # Investigator agent (LLD §9).
    investigation_enabled: bool = True
    investigator_model: str = "reference"
    investigator_effort: str = "medium"
    agent_max_steps: int = 10
    agent_max_cost_usd: float = 0.25
    agent_max_wall_s: float = 90.0

    qc_samples_per_batch: int = 6

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sigil.db"

    @property
    def keys_dir(self) -> Path:
        return self.data_dir / "keys"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
