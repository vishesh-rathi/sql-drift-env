"""P1 — tests for deterministic verifier."""

from __future__ import annotations

import math

from engine.verifier import (
    FLOAT_PRECISION,
    canonical_row_hash,
    result_matches,
    row_set_jaccard,
)


class TestCanonicalRowHash:
    def test_row_order_does_not_matter(self) -> None:
        a = [(1, "x"), (2, "y")]
        b = [(2, "y"), (1, "x")]
        assert canonical_row_hash(a) == canonical_row_hash(b)

    def test_none_and_nan_collapse_to_null_sentinel(self) -> None:
        a = [(1, None)]
        b = [(1, float("nan"))]
        assert canonical_row_hash(a) == canonical_row_hash(b)

    def test_float_rounded_to_precision(self) -> None:
        eps = 10 ** (-(FLOAT_PRECISION + 2))
        assert canonical_row_hash([(1.0,)]) == canonical_row_hash([(1.0 + eps,)])

    def test_float_beyond_precision_differs(self) -> None:
        a = canonical_row_hash([(1.0,)])
        b = canonical_row_hash([(1.01,)])
        assert a != b

    def test_empty_rows_is_stable(self) -> None:
        assert canonical_row_hash([]) == canonical_row_hash([])
        assert len(canonical_row_hash([])) == 64  # sha256 hex digest

    def test_distinct_rows_produce_distinct_hashes(self) -> None:
        assert canonical_row_hash([(1,), (2,)]) != canonical_row_hash([(1,), (3,)])

    def test_result_matches(self) -> None:
        gt = canonical_row_hash([(1, "a"), (2, "b")])
        assert result_matches([(2, "b"), (1, "a")], gt)
        assert not result_matches([(1, "a")], gt)


class TestRowSetJaccard:
    def test_identical_rowsets_jaccard_one(self) -> None:
        assert row_set_jaccard([(1,), (2,)], [(2,), (1,)]) == 1.0

    def test_disjoint_rowsets_jaccard_zero(self) -> None:
        assert row_set_jaccard([(1,)], [(2,)]) == 0.0

    def test_partial_overlap(self) -> None:
        # {1,2} vs {2,3} → |{2}| / |{1,2,3}| = 1/3
        assert math.isclose(row_set_jaccard([(1,), (2,)], [(2,), (3,)]), 1 / 3)

    def test_both_empty_is_one(self) -> None:
        assert row_set_jaccard([], []) == 1.0
