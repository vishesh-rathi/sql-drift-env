"""Scenario 09 — drift: enum split ('active' → 'ACTIVE' / 'ACTIVE_V2').

Baseline counts tenants whose status is 'active'. After drift, every
previously-'active' row has been relabelled to either 'ACTIVE' or
'ACTIVE_V2' (deterministic round-robin). The agent must filter on the
union of the new values to recover the business-equivalent count.

Note: unlike 07 and 10, the post-drift data changed, so the post-drift
ground-truth hash is computed against the post-drift rows — the agent's
result set now reflects the new status values.
"""

from __future__ import annotations

import duckdb

from ._fixtures import categorical_choices, seeded_rng
from .base import BuilderResult, DriftConfig, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_tenants = max(400, scale)

    statuses = categorical_choices(
        rng,
        ["active", "trial", "suspended", "churned"],
        n_tenants,
        weights=[0.55, 0.2, 0.15, 0.1],
    )
    tiers = categorical_choices(
        rng, ["free", "pro", "business"], n_tenants, weights=[0.6, 0.3, 0.1]
    )

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE tenants( id BIGINT PRIMARY KEY, status VARCHAR, tier VARCHAR);")
    conn.executemany(
        "INSERT INTO tenants VALUES (?, ?, ?)",
        [(i, s, t) for i, (s, t) in enumerate(zip(statuses, tiers, strict=False), 1)],
    )

    # Pre-drift: all 'active' tenants; post-drift: union of the new labels.
    baseline_sql = (
        "SELECT tier, COUNT(*) AS n "
        "FROM tenants WHERE status = 'active' "
        "GROUP BY tier ORDER BY tier"
    )
    gt_sql_predrift = baseline_sql
    gt_sql_postdrift = (
        "SELECT tier, COUNT(*) AS n "
        "FROM tenants WHERE status IN ('ACTIVE', 'ACTIVE_V2') "
        "GROUP BY tier ORDER BY tier"
    )

    synopsis = (
        "tenants(id PK, status, tier). Under drift, status='active' is split "
        "into 'ACTIVE' and 'ACTIVE_V2'; 'trial'/'suspended'/'churned' are unchanged."
    )
    return (
        conn,
        baseline_sql,
        gt_sql_predrift,
        gt_sql_postdrift,
        synopsis,
        frozenset({"ACTIVE", "ACTIVE_V2"}),
        frozenset({"active"}),
    )


SPEC = ScenarioSpec(
    scenario_id="09_drift_enum_rule",
    family="multitenant",
    tags=frozenset({"drift", "enum_rule", "business_rule", "multitenant"}),
    drift_config=DriftConfig(
        kind="enum_rule",
        payload={
            "table": "tenants",
            "col": "status",
            "old_value": "active",
            "new_values": ["ACTIVE", "ACTIVE_V2"],
        },
    ),
    builder=_build,
    base_scale=600,
)
