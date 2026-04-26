"""Scenario 01 — correlated subquery anti-pattern.

Baseline computes a per-user order-count via a correlated subquery in the
projection list. The rewrite replaces it with a LEFT JOIN on a grouped
aggregate so the query runs once instead of once per outer row.

Schema: users(id, signup_month, tier), orders(id, user_id, amount, status).
"""

from __future__ import annotations

import duckdb

from ._fixtures import (
    categorical_choices,
    lognormal_amounts,
    seeded_rng,
    zipfian_choices,
)
from .base import BuilderResult, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_users = scale
    n_orders = scale * 6

    user_ids = list(range(1, n_users + 1))
    tiers = categorical_choices(
        rng, ["free", "pro", "business"], n_users, weights=[0.7, 0.25, 0.05]
    )
    signup_months = rng.choices(list(range(1, 13)), k=n_users)

    order_user_ids = zipfian_choices(rng, user_ids, n_orders)
    amounts = lognormal_amounts(rng, n_orders, mu=3.2, sigma=0.9)
    statuses = categorical_choices(
        rng,
        ["placed", "fulfilled", "refunded", "cancelled"],
        n_orders,
        weights=[0.6, 0.3, 0.05, 0.05],
    )

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE users( id BIGINT PRIMARY KEY, signup_month INTEGER, tier VARCHAR);")
    conn.execute(
        "CREATE TABLE orders("
        " id BIGINT PRIMARY KEY, user_id BIGINT, amount DOUBLE, status VARCHAR);"
    )
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?)",
        list(zip(user_ids, signup_months, tiers, strict=False)),
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        [
            (oid, uid, amt, st)
            for oid, (uid, amt, st) in enumerate(
                zip(order_user_ids, amounts, statuses, strict=False), start=1
            )
        ],
    )

    baseline_sql = (
        "SELECT u.id, u.tier, "
        "(SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id AND o.status = 'fulfilled') "
        "  AS fulfilled_orders "
        "FROM users u "
        "ORDER BY u.id"
    )
    gt_sql_predrift = (
        "SELECT u.id, u.tier, COALESCE(c.n, 0) AS fulfilled_orders "
        "FROM users u LEFT JOIN ("
        "  SELECT user_id, COUNT(*) AS n FROM orders "
        "  WHERE status = 'fulfilled' GROUP BY user_id"
        ") c ON c.user_id = u.id "
        "ORDER BY u.id"
    )

    synopsis = (
        "users(id PK, signup_month, tier); orders(id PK, user_id→users.id, amount, status). "
        "Baseline scans orders once per user via a correlated subquery."
    )
    return conn, baseline_sql, gt_sql_predrift, None, synopsis, frozenset(), frozenset()


SPEC = ScenarioSpec(
    scenario_id="01_correlated_subquery",
    family="ecommerce",
    tags=frozenset({"correlated_subquery", "projection_subquery", "ecommerce"}),
    drift_config=None,
    builder=_build,
    base_scale=800,
)
