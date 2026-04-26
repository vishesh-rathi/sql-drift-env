"""Scenario 07 — drift: column rename (orders.user_id → orders.account_id).

Baseline groups orders by `user_id`. When the drift fires, the column is
renamed; the agent must rewrite its query against `account_id`. Row
semantics are unchanged — the post-drift hash equals the pre-drift hash
because the only thing that changed is the column label.
"""

from __future__ import annotations

import duckdb

from ._fixtures import lognormal_amounts, seeded_rng, zipfian_choices
from .base import BuilderResult, DriftConfig, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_users = max(200, scale // 2)
    n_orders = scale * 4

    user_ids = list(range(1, n_users + 1))
    order_users = zipfian_choices(rng, user_ids, n_orders)
    amounts = lognormal_amounts(rng, n_orders, mu=3.0, sigma=0.8)

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE users(id BIGINT PRIMARY KEY, signup_month INTEGER);")
    conn.execute("CREATE TABLE orders(id BIGINT PRIMARY KEY, user_id BIGINT, amount DOUBLE);")
    conn.executemany(
        "INSERT INTO users VALUES (?, ?)",
        [(uid, (uid % 12) + 1) for uid in user_ids],
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [(i, uid, amt) for i, (uid, amt) in enumerate(zip(order_users, amounts, strict=False), 1)],
    )

    baseline_sql = (
        "SELECT user_id, COUNT(*) AS n_orders, ROUND(SUM(amount), 2) AS total "
        "FROM orders GROUP BY user_id ORDER BY user_id"
    )
    gt_sql_predrift = baseline_sql  # static part — baseline IS correct pre-drift
    gt_sql_postdrift = (
        "SELECT account_id, COUNT(*) AS n_orders, ROUND(SUM(amount), 2) AS total "
        "FROM orders GROUP BY account_id ORDER BY account_id"
    )

    synopsis = (
        "users(id PK, signup_month); orders(id PK, user_id→users.id, amount). "
        "Under drift, orders.user_id is renamed to orders.account_id."
    )
    return (
        conn,
        baseline_sql,
        gt_sql_predrift,
        gt_sql_postdrift,
        synopsis,
        frozenset({"account_id"}),
        frozenset({"user_id"}),
    )


SPEC = ScenarioSpec(
    scenario_id="07_drift_column_rename",
    family="ecommerce",
    tags=frozenset({"drift", "column_rename", "ecommerce"}),
    drift_config=DriftConfig(
        kind="column_rename",
        payload={"table": "orders", "old": "user_id", "new": "account_id"},
    ),
    builder=_build,
    base_scale=2_000,
)
