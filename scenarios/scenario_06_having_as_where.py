"""Scenario 06 — HAVING used as WHERE.

Baseline filters on a grouping key inside HAVING, which forces the engine
to group first and filter after. The rewrite moves the non-aggregate
predicate into WHERE.

Schema: orders(id, tenant_id, user_id, amount_cents, status, created_at_epoch_s).
"""

from __future__ import annotations

import duckdb

from ._fixtures import categorical_choices, lognormal_amounts, seeded_rng, zipfian_choices
from .base import BuilderResult, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_tenants = max(40, scale // 40)
    n_users = max(200, scale // 4)
    n_orders = scale * 10

    tenant_ids = list(range(1, n_tenants + 1))
    user_ids = list(range(1, n_users + 1))
    user_tenants = rng.choices(tenant_ids, k=n_users)

    order_user = zipfian_choices(rng, user_ids, n_orders)
    order_tenant = [user_tenants[u - 1] for u in order_user]
    order_amount = [int(x * 100) for x in lognormal_amounts(rng, n_orders, mu=3.5, sigma=0.8)]
    statuses = categorical_choices(
        rng,
        ["placed", "fulfilled", "refunded", "cancelled"],
        n_orders,
        weights=[0.55, 0.3, 0.1, 0.05],
    )
    created = [1_700_000_000 + rng.randrange(60 * 86_400) for _ in range(n_orders)]

    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE orders("
        " id BIGINT PRIMARY KEY, tenant_id BIGINT, user_id BIGINT,"
        " amount_cents BIGINT, status VARCHAR, created_at_epoch_s BIGINT);"
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
        [
            (i, t, u, a, s, ts)
            for i, (t, u, a, s, ts) in enumerate(
                zip(order_tenant, order_user, order_amount, statuses, created, strict=False),
                start=1,
            )
        ],
    )

    # Anti-pattern: filter on a row-level column (status) that IS in GROUP BY
    # inside HAVING, forcing the engine to build groups for every status value
    # before discarding most of them. The rewrite moves the filter into WHERE
    # so aggregation only runs over rows we actually care about.
    baseline_sql = (
        "SELECT tenant_id, status, SUM(amount_cents) AS total_cents "
        "FROM orders "
        "GROUP BY tenant_id, status "
        "HAVING status = 'fulfilled' "
        "   AND SUM(amount_cents) >= 100000 "
        "ORDER BY tenant_id"
    )
    gt_sql_predrift = (
        "SELECT tenant_id, status, SUM(amount_cents) AS total_cents "
        "FROM orders "
        "WHERE status = 'fulfilled' "
        "GROUP BY tenant_id, status "
        "HAVING SUM(amount_cents) >= 100000 "
        "ORDER BY tenant_id"
    )

    synopsis = (
        "orders(id PK, tenant_id, user_id, amount_cents, status, created_at_epoch_s). "
        "Baseline filters `status` inside HAVING, forcing aggregation over "
        "every status group before discarding all but 'fulfilled'."
    )
    return conn, baseline_sql, gt_sql_predrift, None, synopsis, frozenset(), frozenset()


SPEC = ScenarioSpec(
    scenario_id="06_having_as_where",
    family="ecommerce",
    tags=frozenset({"having_as_where", "aggregate_filter", "ecommerce"}),
    drift_config=None,
    builder=_build,
    base_scale=1_500,
)
