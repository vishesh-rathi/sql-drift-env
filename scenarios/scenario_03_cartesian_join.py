"""Scenario 03 — accidental cartesian join.

Baseline joins events × tenants without an equijoin clause and relies on a
post-filter `WHERE` to restore tenant scoping. The optimizer cannot push
the filter into the join, so it materializes the full |events|·|tenants|
cross-product first. The rewrite moves the tenant key into an ON clause.

Schema: tenants(id, tier), events(id, tenant_id, kind, severity).
"""

from __future__ import annotations

import duckdb

from ._fixtures import categorical_choices, seeded_rng, zipfian_choices
from .base import BuilderResult, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_tenants = max(20, scale // 40)
    n_events = scale * 4

    tenant_ids = list(range(1, n_tenants + 1))
    tiers = categorical_choices(
        rng, ["free", "pro", "enterprise"], n_tenants, weights=[0.6, 0.3, 0.1]
    )
    event_tenant_ids = zipfian_choices(rng, tenant_ids, n_events)
    kinds = categorical_choices(
        rng,
        ["login", "logout", "action", "error"],
        n_events,
        weights=[0.35, 0.3, 0.3, 0.05],
    )
    severities = categorical_choices(
        rng,
        ["info", "warn", "error", "critical"],
        n_events,
        weights=[0.7, 0.2, 0.08, 0.02],
    )

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE tenants(id BIGINT PRIMARY KEY, tier VARCHAR);")
    conn.execute(
        "CREATE TABLE events("
        " id BIGINT PRIMARY KEY, tenant_id BIGINT, kind VARCHAR, severity VARCHAR);"
    )
    conn.executemany(
        "INSERT INTO tenants VALUES (?, ?)", list(zip(tenant_ids, tiers, strict=False))
    )
    conn.executemany(
        "INSERT INTO events VALUES (?, ?, ?, ?)",
        [
            (i, tid, k, s)
            for i, (tid, k, s) in enumerate(
                zip(event_tenant_ids, kinds, severities, strict=False), start=1
            )
        ],
    )

    baseline_sql = (
        "SELECT t.tier, COUNT(*) AS n "
        "FROM events e, tenants t "
        "WHERE t.id = e.tenant_id + 0 "  # defeat optimizer recognition
        "AND e.severity IN ('error', 'critical') "
        "GROUP BY t.tier ORDER BY t.tier"
    )
    gt_sql_predrift = (
        "SELECT t.tier, COUNT(*) AS n "
        "FROM events e JOIN tenants t ON t.id = e.tenant_id "
        "WHERE e.severity IN ('error', 'critical') "
        "GROUP BY t.tier ORDER BY t.tier"
    )

    synopsis = (
        "tenants(id PK, tier); events(id PK, tenant_id→tenants.id, kind, severity). "
        "Baseline relies on a WHERE-clause equijoin obscured by arithmetic, "
        "forcing a cartesian materialization."
    )
    return conn, baseline_sql, gt_sql_predrift, None, synopsis, frozenset(), frozenset()


SPEC = ScenarioSpec(
    scenario_id="03_cartesian_join",
    family="events",
    tags=frozenset({"cartesian", "missing_join_condition", "events"}),
    drift_config=None,
    builder=_build,
    # Cartesian materialization cost grows with |events|·|tenants| so
    # larger base_scale keeps first-try baseline above the 1 ms floor;
    # the reroll loop doubles from here if needed.
    base_scale=1_200,
)
