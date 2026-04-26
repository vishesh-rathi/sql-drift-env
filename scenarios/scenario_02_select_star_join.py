"""Scenario 02 — SELECT * over a join anti-pattern.

Baseline pulls every column from three joined tables when only two columns
are needed. The rewrite projects exactly the requested columns.

Schema: products(id, sku, name, description, price_cents, …), orders(id, user_id, created_at),
        order_items(order_id, product_id, qty, unit_price_cents).
"""

from __future__ import annotations

import duckdb

from ._fixtures import (
    categorical_choices,
    lognormal_amounts,
    seeded_rng,
    unique_names,
    zipfian_choices,
)
from .base import BuilderResult, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_products = max(100, scale // 4)
    n_orders = scale
    n_items = scale * 3
    n_users = max(50, scale // 8)

    product_ids = list(range(1, n_products + 1))
    skus = unique_names(rng, n_products, prefix="sku")
    names = unique_names(rng, n_products, prefix="p")
    descriptions = [f"Long marketing copy for {n}" * 6 for n in names]  # wide col
    prices = [int(x * 100) for x in lognormal_amounts(rng, n_products, mu=3.0, sigma=1.1)]
    categories = categorical_choices(
        rng, ["books", "electronics", "apparel", "grocery", "home"], n_products
    )

    user_ids = list(range(1, n_users + 1))
    order_user_ids = zipfian_choices(rng, user_ids, n_orders)
    order_created = [1_700_000_000 + rng.randrange(60 * 86_400) for _ in range(n_orders)]

    item_order_ids = rng.choices(list(range(1, n_orders + 1)), k=n_items)
    item_product_ids = zipfian_choices(rng, product_ids, n_items)
    item_qty = rng.choices([1, 1, 1, 2, 2, 3, 4, 5], k=n_items)
    item_unit_price = [prices[pid - 1] for pid in item_product_ids]

    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE products("
        " id BIGINT PRIMARY KEY, sku VARCHAR, name VARCHAR, description VARCHAR,"
        " price_cents BIGINT, category VARCHAR);"
    )
    conn.execute(
        "CREATE TABLE orders( id BIGINT PRIMARY KEY, user_id BIGINT, created_at_epoch_s BIGINT);"
    )
    conn.execute(
        "CREATE TABLE order_items("
        " order_id BIGINT, product_id BIGINT, qty INTEGER, unit_price_cents BIGINT);"
    )
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
        list(zip(product_ids, skus, names, descriptions, prices, categories, strict=False)),
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [
            (oid, uid, ts)
            for oid, (uid, ts) in enumerate(zip(order_user_ids, order_created, strict=False), 1)
        ],
    )
    conn.executemany(
        "INSERT INTO order_items VALUES (?, ?, ?, ?)",
        list(zip(item_order_ids, item_product_ids, item_qty, item_unit_price, strict=False)),
    )

    # Anti-pattern: SELECT * inside a subquery that wraps the real join, then
    # the outer query projects only a handful of columns. DuckDB materializes
    # every column of the subquery before the projection can prune it.
    baseline_sql = (
        "SELECT t.order_id, t.name, t.qty "
        "FROM ("
        "  SELECT * FROM order_items oi "
        "  JOIN products p ON p.id = oi.product_id "
        "  JOIN orders o ON o.id = oi.order_id "
        "  WHERE p.category = 'books' AND oi.qty >= 2"
        ") t "
        "ORDER BY t.order_id, t.name"
    )
    gt_sql_predrift = (
        "SELECT oi.order_id, p.name, oi.qty "
        "FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id "
        "JOIN orders o ON o.id = oi.order_id "
        "WHERE p.category = 'books' AND oi.qty >= 2 "
        "ORDER BY oi.order_id, p.name"
    )

    synopsis = (
        "products(id PK, sku, name, description, price_cents, category); "
        "orders(id PK, user_id, created_at_epoch_s); "
        "order_items(order_id, product_id, qty, unit_price_cents). "
        "Baseline wraps a three-way join with SELECT * inside a subquery; "
        "only (order_id, product name, qty) are needed downstream."
    )
    return conn, baseline_sql, gt_sql_predrift, None, synopsis, frozenset(), frozenset()


SPEC = ScenarioSpec(
    scenario_id="02_select_star_join",
    family="ecommerce",
    tags=frozenset({"select_star", "over_projection", "join", "ecommerce"}),
    drift_config=None,
    builder=_build,
    # Three-way join with SELECT * is already well-optimized by DuckDB;
    # a larger base keeps first-try baseline above the 1 ms floor.
    base_scale=1_500,
)
