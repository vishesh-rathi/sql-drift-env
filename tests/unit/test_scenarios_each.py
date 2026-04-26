"""P3 — per-scenario golden tests (parameterized over specs × seeds).

Each materialized scenario must satisfy three invariants:

1. The *baseline SQL* and *ground-truth rewrite* produce result sets with
   identical canonical hashes (the rewrite is truly semantics-preserving).
2. Both SQLs run on the materialized DuckDB fixture without error.
3. The schema synopsis is non-empty and mentions at least one table name.

Parameterized over 3 seeds per scenario → covers deterministic seeding plus
cross-seed stability of the rewrite semantics.
"""

from __future__ import annotations

import pytest

import scenarios
from engine.verifier import canonical_row_hash

_STATIC_SCENARIO_IDS = [
    "01_correlated_subquery",
    "02_select_star_join",
    "03_cartesian_join",
    "04_distinct_groupby",
    "05_nested_subquery",
    "06_having_as_where",
]
_SEEDS = (1, 7, 42)


@pytest.mark.parametrize("scenario_id", _STATIC_SCENARIO_IDS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_baseline_matches_ground_truth(scenario_id: str, seed: int) -> None:
    spec = scenarios.get_spec(scenario_id)
    inst = spec.materialize(seed=seed)
    try:
        baseline_rows = inst.conn.execute(inst.baseline_sql).fetchall()
        gt_rows = inst.conn.execute(inst.gt_sql_predrift).fetchall()
        assert canonical_row_hash(baseline_rows) == canonical_row_hash(gt_rows), (
            f"{scenario_id} seed={seed}: baseline vs gt_sql_predrift diverge"
        )
        assert canonical_row_hash(gt_rows) == inst.gt_result_hash_predrift
    finally:
        inst.conn.close()


@pytest.mark.parametrize("scenario_id", _STATIC_SCENARIO_IDS)
def test_schema_synopsis_nonempty(scenario_id: str) -> None:
    spec = scenarios.get_spec(scenario_id)
    inst = spec.materialize(seed=1)
    try:
        assert inst.schema_synopsis
        assert len(inst.schema_synopsis) >= 32  # meaningful description
        # Synopsis should reference at least one identifier we can see in the SQL.
        assert any(
            token in inst.schema_synopsis
            for token in (
                "users",
                "orders",
                "tenants",
                "events",
                "pageviews",
                "authors",
                "articles",
                "comments",
                "products",
                "order_items",
            )
        )
    finally:
        inst.conn.close()


@pytest.mark.parametrize("scenario_id", _STATIC_SCENARIO_IDS)
def test_tags_nonempty_and_family_valid(scenario_id: str) -> None:
    spec = scenarios.get_spec(scenario_id)
    assert spec.tags
    assert spec.family in {"ecommerce", "events", "cms", "saas_logs", "multitenant"}


@pytest.mark.parametrize("scenario_id", _STATIC_SCENARIO_IDS)
def test_drift_config_absent_for_static(scenario_id: str) -> None:
    spec = scenarios.get_spec(scenario_id)
    assert spec.drift_config is None
    assert spec.materialize(seed=1).gt_sql_postdrift is None


@pytest.mark.parametrize("scenario_id", _STATIC_SCENARIO_IDS)
def test_materialize_is_deterministic(scenario_id: str) -> None:
    spec = scenarios.get_spec(scenario_id)
    a = spec.materialize(seed=13)
    b = spec.materialize(seed=13)
    try:
        assert a.gt_result_hash_predrift == b.gt_result_hash_predrift
    finally:
        a.conn.close()
        b.conn.close()
