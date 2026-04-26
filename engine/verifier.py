"""Deterministic result verification.

Canonicalizes floats to `FLOAT_PRECISION` decimal places and treats NULL
uniformly so that two result sets with the same semantic content hash to
the same digest regardless of row order, floating-point noise, or None vs
SQL NULL representation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

NULL_SENTINEL = "\x00NULL\x00"
FLOAT_PRECISION = 6
_DIGEST_MODULUS = 1 << 256


def _normalize_value(v: Any) -> Any:
    if v is None:
        return NULL_SENTINEL
    if isinstance(v, float):
        # NaN hashes to itself here — a NaN in rows yields a deterministic
        # digest, but two NaNs round-trip to different representations when
        # we `repr`. Guard explicitly.
        if v != v:  # NaN
            return NULL_SENTINEL
        return round(v, FLOAT_PRECISION)
    return v


def _row_digest_int(row: Iterable[Any]) -> int:
    normalized = tuple(_normalize_value(v) for v in row)
    digest = hashlib.sha256(repr(normalized).encode()).digest()
    return int.from_bytes(digest, "big", signed=False)


def canonical_row_hash(rows: Iterable[Iterable[Any]]) -> str:
    """Order-independent hash of a result set.

    This stays order-independent and duplicate-sensitive without
    materializing the full result in memory. Each normalized row is
    hashed once, then folded into three commutative accumulators so the
    final digest is stable across row order and Python processes.
    """
    row_count = 0
    sum_acc = 0
    sumsq_acc = 0
    xor_acc = 0
    for row in rows:
        row_count += 1
        row_digest = _row_digest_int(row)
        sum_acc = (sum_acc + row_digest) % _DIGEST_MODULUS
        sumsq_acc = (sumsq_acc + ((row_digest * row_digest) % _DIGEST_MODULUS)) % _DIGEST_MODULUS
        xor_acc ^= row_digest
    payload = b"".join(
        (
            row_count.to_bytes(32, "big", signed=False),
            sum_acc.to_bytes(32, "big", signed=False),
            sumsq_acc.to_bytes(32, "big", signed=False),
            xor_acc.to_bytes(32, "big", signed=False),
        )
    )
    return hashlib.sha256(payload).hexdigest()


def result_matches(agent_rows: Iterable[Iterable[Any]], gt_hash: str) -> bool:
    """True if `agent_rows` canonicalizes to the ground-truth hash."""
    return canonical_row_hash(agent_rows) == gt_hash


def row_set_jaccard(a: Iterable[Iterable[Any]], b: Iterable[Iterable[Any]]) -> float:
    """Jaccard over normalized row sets (order- and duplicate-insensitive).

    Each input row is normalised with :func:`_normalize_value` and
    collapsed into a :class:`frozenset`-style Python ``set``, so rows
    that repeat within a single result are counted once. This is
    deliberately *not* a multiset Jaccard — multiset semantics would
    punish correct queries that legitimately emit duplicates more
    harshly than intended.

    Not used by the lean reward today, but kept covered by tests so
    we can opt in later without rework.
    """
    norm_a = {tuple(_normalize_value(v) for v in row) for row in a}
    norm_b = {tuple(_normalize_value(v) for v in row) for row in b}
    if not norm_a and not norm_b:
        return 1.0
    union = norm_a | norm_b
    inter = norm_a & norm_b
    return len(inter) / len(union)


__all__ = [
    "FLOAT_PRECISION",
    "NULL_SENTINEL",
    "canonical_row_hash",
    "result_matches",
    "row_set_jaccard",
]
