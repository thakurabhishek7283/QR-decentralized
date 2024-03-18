"""A witness for the log: it trusts only the log's public key and what it saw last."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from sigil.errors import LogError
from sigil.ledger import merkle
from sigil.ledger.log import ConsistencyProof, SignedTreeHead


@dataclass(frozen=True, slots=True)
class AuditResult:
    ok: bool
    reason: str = "ok"


class Auditor:
    """Accepts a tree head only if it is signed, never shrinks, and extends the last one.

    The last accepted head is saved to ``state_path`` so the check survives restarts;
    the first head it ever sees is trusted on first use.
    """

    def __init__(self, log_public_raw: bytes, state_path: Path) -> None:
        self._public = log_public_raw
        self._path = state_path
        self._last = self._load()

    @property
    def last(self) -> SignedTreeHead | None:
        return self._last

    def observe(self, sth: SignedTreeHead, proof_from_prev: ConsistencyProof | None) -> AuditResult:
        if not sth.verify(self._public):
            return AuditResult(False, "tree head signature is invalid")
        prev = self._last
        if prev is None:
            return self._accept(sth)
        if sth.tree_size < prev.tree_size:
            return AuditResult(False, f"log shrank from {prev.tree_size} to {sth.tree_size}")
        if sth.tree_size == prev.tree_size:
            if sth.root_hash != prev.root_hash:
                return AuditResult(False, "two different roots were signed for one tree size")
            return AuditResult(True)
        if prev.tree_size > 0:
            reason = _check_consistency(prev, sth, proof_from_prev)
            if reason is not None:
                return AuditResult(False, reason)
        return self._accept(sth)

    def _accept(self, sth: SignedTreeHead) -> AuditResult:
        self._last = sth
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f".{self._path.name}.tmp")
        tmp.write_text(json.dumps(sth.to_json(), indent=2))
        os.replace(tmp, self._path)
        return AuditResult(True)

    def _load(self) -> SignedTreeHead | None:
        if not self._path.exists():
            return None
        try:
            sth = SignedTreeHead.from_json(json.loads(self._path.read_text()))
        except json.JSONDecodeError as exc:
            raise LogError(f"auditor state {self._path} is corrupt: {exc}") from exc
        if not sth.verify(self._public):
            raise LogError(f"auditor state {self._path} is not signed by this log")
        return sth


def _check_consistency(
    prev: SignedTreeHead, sth: SignedTreeHead, proof: ConsistencyProof | None
) -> str | None:
    """Return why ``sth`` does not extend ``prev``, or ``None`` if it does."""
    if proof is None:
        return "a consistency proof is required to move to a larger tree"
    if (proof.first, proof.second) != (prev.tree_size, sth.tree_size):
        return "consistency proof is for the wrong tree sizes"
    if not merkle.verify_consistency(
        prev.tree_size, sth.tree_size, prev.root_hash, sth.root_hash, proof.path
    ):
        return "consistency proof does not verify: history was rewritten"
    return None
