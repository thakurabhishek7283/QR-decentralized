from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sigil.ledger import (
    consistency_proof,
    inclusion_proof,
    leaf_hash,
    node_hash,
    root_hash,
    verify_consistency,
    verify_inclusion,
)
from sigil.ledger.merkle import EMPTY_ROOT

# --- an independent, deliberately naive transcription of RFC 9162 section 2.1 -----------------
# It works on raw entries and slices lists, so it shares no code or structure with the module.


def _sha(*parts: bytes) -> bytes:
    return hashlib.sha256(b"".join(parts)).digest()


def _k(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def naive_mth(entries: list[bytes]) -> bytes:
    if not entries:
        return _sha()
    if len(entries) == 1:
        return _sha(b"\x00", entries[0])
    k = _k(len(entries))
    return _sha(b"\x01", naive_mth(entries[:k]), naive_mth(entries[k:]))


def naive_path(m: int, entries: list[bytes]) -> list[bytes]:
    if len(entries) == 1:
        return []
    k = _k(len(entries))
    if m < k:
        return [*naive_path(m, entries[:k]), naive_mth(entries[k:])]
    return [*naive_path(m - k, entries[k:]), naive_mth(entries[:k])]


def naive_subproof(m: int, entries: list[bytes], complete: bool) -> list[bytes]:
    if m == len(entries):
        return [] if complete else [naive_mth(entries)]
    k = _k(len(entries))
    if m <= k:
        return [*naive_subproof(m, entries[:k], complete), naive_mth(entries[k:])]
    return [*naive_subproof(m - k, entries[k:], False), naive_mth(entries[:k])]


# --- test data -----------------------------------------------------------------------------------

RFC_LEAVES = [
    b"",
    b"\x00",
    b"\x10",
    b"\x20\x21",
    b"\x30\x31",
    b"\x40\x41\x42\x43",
    bytes(range(0x50, 0x58)),
    bytes(range(0x60, 0x70)),
]
# Roots of the first n leaves of the RFC 6962 / transparency-dev reference tree.
RFC_ROOTS = {
    1: "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    2: "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    3: "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    4: "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    5: "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
    6: "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
    7: "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
    8: "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
}

MAX_SIZE = 300
ENTRIES = [i.to_bytes(4, "big") for i in range(MAX_SIZE)]
LEAVES = [leaf_hash(e) for e in ENTRIES]
SMALL = 40

sizes = st.integers(min_value=1, max_value=MAX_SIZE)


def flip(h: bytes, bit: int = 0) -> bytes:
    return bytes([h[0] ^ (1 << bit)]) + h[1:]


# --- hashing and the reference vectors ------------------------------------------------------------


def test_empty_tree_root_is_sha256_of_nothing() -> None:
    assert root_hash([]) == EMPTY_ROOT == hashlib.sha256(b"").digest()
    assert EMPTY_ROOT.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_leaf_and_node_hashes_are_domain_separated() -> None:
    a, b = leaf_hash(b"a"), leaf_hash(b"b")
    assert leaf_hash(b"a") == _sha(b"\x00", b"a")
    assert node_hash(a, b) == _sha(b"\x01", a, b)
    assert leaf_hash(a + b) != node_hash(a, b)


@pytest.mark.parametrize(("size", "expected"), RFC_ROOTS.items())
def test_rfc_6962_reference_roots(size: int, expected: str) -> None:
    leaves = [leaf_hash(e) for e in RFC_LEAVES[:size]]
    assert root_hash(leaves).hex() == expected
    assert naive_mth(RFC_LEAVES[:size]).hex() == expected


@pytest.mark.parametrize("size", range(1, SMALL + 1))
def test_root_matches_naive_reference(size: int) -> None:
    assert root_hash(LEAVES[:size]) == naive_mth(ENTRIES[:size])


@pytest.mark.parametrize("size", range(1, SMALL + 1))
def test_proof_generation_matches_naive_reference(size: int) -> None:
    entries, leaves = ENTRIES[:size], LEAVES[:size]
    for index in range(size):
        assert inclusion_proof(index, leaves) == naive_path(index, entries)
    for m in range(1, size + 1):
        assert consistency_proof(m, leaves) == naive_subproof(m, entries, True)


def test_known_proof_shapes() -> None:
    leaves = [leaf_hash(e) for e in RFC_LEAVES]
    assert inclusion_proof(0, leaves[:1]) == []
    assert len(inclusion_proof(5, leaves)) == 3
    assert len(inclusion_proof(6, leaves[:7])) == 2  # the odd leaf hangs off a shallow subtree
    assert inclusion_proof(6, leaves[:7])[0] == root_hash(leaves[4:6])
    assert consistency_proof(8, leaves) == []
    assert consistency_proof(4, leaves) == [root_hash(leaves[4:])]
    assert len(consistency_proof(6, leaves)) == 3


# --- exhaustive checks on small trees ---------------------------------------------------------


@pytest.mark.parametrize("size", range(1, SMALL + 1))
def test_every_inclusion_proof_verifies(size: int) -> None:
    leaves = LEAVES[:size]
    root = root_hash(leaves)
    for index in range(size):
        proof = inclusion_proof(index, leaves)
        assert verify_inclusion(leaves[index], index, size, proof, root)


@pytest.mark.parametrize("n", range(1, SMALL + 1))
def test_every_consistency_proof_verifies(n: int) -> None:
    leaves = LEAVES[:n]
    root_n = root_hash(leaves)
    for m in range(1, n + 1):
        proof = consistency_proof(m, leaves)
        assert verify_consistency(m, n, root_hash(leaves[:m]), root_n, proof)


@pytest.mark.parametrize("size", range(2, SMALL + 1))
def test_inclusion_proofs_reject_every_wrong_index(size: int) -> None:
    leaves = LEAVES[:size]
    root = root_hash(leaves)
    for index in range(size):
        proof = inclusion_proof(index, leaves)
        for other in range(size):
            if other != index:
                assert not verify_inclusion(leaves[index], other, size, proof, root)


@pytest.mark.parametrize("n", range(2, SMALL + 1))
def test_consistency_proofs_reject_every_wrong_first_size(n: int) -> None:
    leaves = LEAVES[:n]
    root_n = root_hash(leaves)
    for m in range(1, n):
        proof = consistency_proof(m, leaves)
        for other in range(1, n + 1):
            if other != m:
                assert not verify_consistency(other, n, root_hash(leaves[:other]), root_n, proof)


# --- hypothesis over sizes 1..300 -------------------------------------------------------------


@settings(deadline=None)
@given(size=sizes, data=st.data())
def test_inclusion_proof_verifies_and_mutations_fail(size: int, data: st.DataObject) -> None:
    leaves = LEAVES[:size]
    index = data.draw(st.integers(0, size - 1))
    root = root_hash(leaves)
    proof = inclusion_proof(index, leaves)
    assert verify_inclusion(leaves[index], index, size, proof, root)

    assert not verify_inclusion(leaf_hash(b"forged"), index, size, proof, root)
    assert not verify_inclusion(leaves[index], index, size, proof, flip(root))
    assert not verify_inclusion(leaves[index], index, size, [*proof, LEAVES[0]], root)
    assert not verify_inclusion(leaves[index], size + index, size, proof, root)
    if proof:
        at = data.draw(st.integers(0, len(proof) - 1))
        bit = data.draw(st.integers(0, 7))
        mutated = [*proof[:at], flip(proof[at], bit), *proof[at + 1 :]]
        assert not verify_inclusion(leaves[index], index, size, mutated, root)
        assert not verify_inclusion(leaves[index], index, size, proof[:-1], root)
        assert not verify_inclusion(leaves[index], index, size, proof[1:], root)


@settings(deadline=None)
@given(n=sizes, data=st.data())
def test_consistency_proof_verifies_and_mutations_fail(n: int, data: st.DataObject) -> None:
    m = data.draw(st.integers(1, n))
    leaves = LEAVES[:n]
    root_m, root_n = root_hash(leaves[:m]), root_hash(leaves)
    proof = consistency_proof(m, leaves)
    assert verify_consistency(m, n, root_m, root_n, proof)

    assert not verify_consistency(m, n, flip(root_m), root_n, proof)
    assert not verify_consistency(m, n, root_m, flip(root_n), proof)
    assert not verify_consistency(m, n, root_m, root_n, [*proof, LEAVES[0]])
    if m != n:
        assert not verify_consistency(m, n, root_n, root_m, proof)
    if proof:
        at = data.draw(st.integers(0, len(proof) - 1))
        bit = data.draw(st.integers(0, 7))
        mutated = [*proof[:at], flip(proof[at], bit), *proof[at + 1 :]]
        assert not verify_consistency(m, n, root_m, root_n, mutated)
        assert not verify_consistency(m, n, root_m, root_n, proof[:-1])


@settings(deadline=None)
@given(n=sizes, data=st.data())
def test_a_forked_history_has_no_valid_consistency_proof(n: int, data: st.DataObject) -> None:
    m = data.draw(st.integers(1, n))
    forked = [*LEAVES[: m - 1], leaf_hash(b"rewritten"), *LEAVES[m:n]]
    proof = consistency_proof(m, forked)
    assert not verify_consistency(m, n, root_hash(LEAVES[:m]), root_hash(forked), proof)


@settings(deadline=None)
@given(size=sizes)
def test_proof_lengths_are_logarithmic(size: int) -> None:
    ceil_log = (size - 1).bit_length()
    assert all(len(inclusion_proof(i, LEAVES[:size])) <= ceil_log for i in (0, size // 2, size - 1))
    assert len(consistency_proof(max(1, size // 3), LEAVES[:size])) <= ceil_log + 1


# --- argument validation ------------------------------------------------------------------------


def test_generators_reject_out_of_range_arguments() -> None:
    with pytest.raises(ValueError, match="outside"):
        inclusion_proof(3, LEAVES[:3])
    with pytest.raises(ValueError, match="outside"):
        inclusion_proof(-1, LEAVES[:3])
    with pytest.raises(ValueError, match="consistency"):
        consistency_proof(0, LEAVES[:3])
    with pytest.raises(ValueError, match="consistency"):
        consistency_proof(4, LEAVES[:3])


def test_verifiers_reject_degenerate_arguments() -> None:
    root = root_hash(LEAVES[:4])
    assert not verify_inclusion(LEAVES[0], 0, 0, [], root)
    assert not verify_inclusion(LEAVES[0], -1, 4, [], root)
    assert not verify_consistency(0, 4, EMPTY_ROOT, root, [])
    assert not verify_consistency(5, 4, root, root, [])
    assert not verify_consistency(2, 4, root_hash(LEAVES[:2]), root, [])
    assert verify_consistency(4, 4, root, root, [])
    assert not verify_consistency(4, 4, root, root, [LEAVES[0]])
    assert not verify_consistency(4, 4, root, flip(root), [])
