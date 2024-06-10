"""Minting labels: sign credentials, log them, and render the printable label."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from sigil.cdp import generate_template
from sigil.crypto import KeyMaterial, issue
from sigil.crypto.credential import SERIAL_LEN
from sigil.issuance.store import BatchInfo, IssuanceStore, IssuedUnit, PrintQuality
from sigil.ledger import TransparencyLog
from sigil.vision import render_label

BATCH_ID_BYTES = 3  # "B" + 6 hex characters, as in the LLD; fits the 16-char credential field


class Issuer:
    """Issues batches of unit labels under the issuer key.

    ``random_bytes`` and ``clock`` are seams: production uses the OS CSPRNG and wall time, while
    the eval worlds inject seeded sources so a scenario is reproducible byte for byte.
    """

    def __init__(
        self,
        keys: KeyMaterial,
        log: TransparencyLog,
        store: IssuanceStore,
        *,
        random_bytes: Callable[[int], bytes] = secrets.token_bytes,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._keys = keys
        self._log = log
        self.store = store
        self._random_bytes = random_bytes
        self._clock = clock

    def issue_batch(
        self,
        sku: str,
        count: int,
        *,
        batch_id: str | None = None,
        print_quality: PrintQuality = "good",
    ) -> tuple[BatchInfo, list[IssuedUnit]]:
        """Sign ``count`` credentials, log them, publish a tree head and persist the units.

        Logging happens before persisting, so a unit never exists in the store without being
        verifiable against the log. A crash in between leaves orphan log entries, which is the
        safe direction for a transparency log.
        """
        if count < 1:
            raise ValueError("a batch needs at least one unit")
        batch_id = batch_id or "B" + self._random_bytes(BATCH_ID_BYTES).hex()
        now = self._clock()
        credentials = [
            issue(
                self._keys.issuer,
                sku=sku,
                batch=batch_id,
                iat=int(now),
                serial=self._random_bytes(SERIAL_LEN),
            )
            for _ in range(count)
        ]
        indices = self._log.append_many([sc.leaf_data() for sc in credentials])
        self._log.publish()

        created = BatchInfo(batch_id, sku, datetime.fromtimestamp(now, UTC), print_quality)
        self.store.add_batch(created, list(zip(credentials, indices, strict=True)))
        units = [
            IssuedUnit(sc.credential.serial_hex, sc.to_qr_payload(), index)
            for sc, index in zip(credentials, indices, strict=True)
        ]
        return self.store.batch(batch_id), units  # as stored, i.e. at millisecond precision

    def template_for(self, serial: bytes) -> np.ndarray:
        """The secret CDP grid for a serial."""
        return generate_template(self._keys.cdp_key, serial)

    def render_label(self, serial_hex: str) -> np.ndarray:
        """The printable label raster (QR beside the CDP) for an issued serial."""
        credential, _ = self.store.unit(serial_hex)
        template = self.template_for(credential.credential.serial)
        return render_label(credential.to_qr_payload(), template)
