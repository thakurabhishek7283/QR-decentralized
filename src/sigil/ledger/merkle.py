"""Merkle tree hashing, proofs and verification per RFC 9162 section 2.1.

The generators work on a list of leaf hashes. The verifiers are the iterative
algorithms of sections 2.1.3.2 and 2.1.4.2: they consume only a proof and two
roots, so a client can check a log without ever holding the tree.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"
EMPTY_ROOT = hashlib.sha256(b"").digest()


def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(_LEAF_PREFIX + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


def _split(n: int) -> int:
    """The largest power of two strictly smaller than ``n`` (``n >= 2``)."""
    return 1 << ((n - 1).bit_length() - 1)


def _mth(leaves: Sequence[bytes], lo: int, hi: int) -> bytes:
    """MTH over ``leaves[lo:hi]``, which must be non-empty."""
    if hi - lo == 1:
        return leaves[lo]
    k = lo + _split(hi - lo)
    return node_hash(_mth(leaves, lo, k), _mth(leaves, k, hi))


def root_hash(leaves: Sequence[bytes]) -> bytes:
    """MTH(D[n]); the empty tree hashes the empty string."""
    return _mth(leaves, 0, len(leaves)) if leaves else EMPTY_ROOT


def inclusion_proof(index: int, leaves: Sequence[bytes]) -> list[bytes]:
    """PATH(index, D[n]): the sibling hashes from leaf to root."""
    if not 0 <= index < len(leaves):
        raise ValueError(f"leaf index {index} is outside a tree of {len(leaves)} leaves")
    path: list[bytes] = []
    lo, hi = 0, len(leaves)
    while hi - lo > 1:
        k = lo + _split(hi - lo)
        if index < k:
            path.append(_mth(leaves, k, hi))
            hi = k
        else:
            path.append(_mth(leaves, lo, k))
            lo = k
    path.reverse()
    return path


def consistency_proof(m: int, leaves: Sequence[bytes]) -> list[bytes]:
    """PROOF(m, D[n]): shows the first ``m`` leaves are a prefix of all ``n``."""
    n = len(leaves)
    if not 1 <= m <= n:
        raise ValueError(f"cannot prove consistency of size {m} within a tree of {n} leaves")
    path: list[bytes] = []
    lo, hi = 0, n
    on_left_edge = True  # SUBPROOF's flag: the subtree still starts where the first tree does
    while m != hi:
        k = lo + _split(hi - lo)
        if m <= k:
            path.append(_mth(leaves, k, hi))
            hi = k
        else:
            path.append(_mth(leaves, lo, k))
            lo, on_left_edge = k, False
    if not on_left_edge:
        path.append(_mth(leaves, lo, hi))
    path.reverse()
    return path


def verify_inclusion(
    leaf: bytes, index: int, size: int, proof: Sequence[bytes], root: bytes
) -> bool:
    """RFC 9162 section 2.1.3.2."""
    if index >= size or index < 0:
        return False
    fn, sn = index, size - 1
    r = leaf
    for p in proof:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def verify_consistency(
    m: int, n: int, root_m: bytes, root_n: bytes, proof: Sequence[bytes]
) -> bool:
    """RFC 9162 section 2.1.4.2."""
    if not 1 <= m <= n:
        return False
    if m == n:
        return not proof and root_m == root_n
    path = list(proof)
    if m & (m - 1) == 0:
        path.insert(0, root_m)
    if not path:
        return False
    fn, sn = m - 1, n - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = path[0]
    for c in path[1:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = node_hash(c, fr)
            sr = node_hash(c, sr)
            if not fn & 1:
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    return sn == 0 and fr == root_m and sr == root_n
