"""DBA Oracle: per-scenario escalating expert guidance.

Feature-flagged (off by default). When ``enable_dba_oracle=True`` is set
at ``reset(...)`` or ``SQL_DRIFT_ENABLE_DBA_ORACLE=1`` is exported in the
environment, the ``consult_dba`` tool becomes available; three hint
tiers per scenario are shipped here, covering all 10 scenarios (6 static
+ 4 drift).

Hints escalate in specificity: tier 1 names the likely failure mode and
the diagnostic to confirm it, tier 2 gives the rewrite invariant, and
tier 3 is a near-spoiler SQL skeleton. The rubric penalizes each consult
(consultation rubric), so the agent only wins by consulting if the hint net-reduces
downstream steps.
"""

from __future__ import annotations

import os
from typing import Final

# ---------------------------------------------------------------------------
# Per-scenario 3-tier hint tables
# ---------------------------------------------------------------------------


_HINTS: Final[dict[str, tuple[str, str, str]]] = {
    "01_correlated_subquery": (
        "[DBA tier 1] The expensive shape is a projection-time correlated subquery: "
        "a COUNT over `orders` for every `users` row. Confirm by spotting "
        "`SELECT COUNT(*) ... WHERE o.user_id = u.id` in the SELECT list or by "
        "checking EXPLAIN for repeated dependent work. Preserve one output row per user.",
        "[DBA tier 2] Aggregate fulfilled orders once by `user_id`, then LEFT JOIN that "
        "small result to `users`. Keep the join outer and wrap the count with "
        "`COALESCE(..., 0)` so users with no fulfilled orders stay in the result.",
        "[DBA tier 3] Use `SELECT u.id, u.tier, COALESCE(c.n, 0) AS fulfilled_orders "
        "FROM users u LEFT JOIN (SELECT user_id, COUNT(*) AS n FROM orders WHERE "
        "status = 'fulfilled' GROUP BY user_id) c ON c.user_id = u.id ORDER BY u.id`. "
        "Validate the row count equals the number of users.",
    ),
    "02_select_star_join": (
        "[DBA tier 1] The waste is over-projection: the inner three-way join uses "
        "`SELECT *`, including wide product text and order metadata, while the outer "
        "query keeps only `order_id`, product `name`, and `qty`.",
        "[DBA tier 2] Inline the join and project exactly `oi.order_id`, `p.name`, and "
        "`oi.qty`. Keep the products and orders joins plus the filters "
        "`p.category = 'books'` and `oi.qty >= 2`; the wrapper exists only to hide "
        "the star projection.",
        "[DBA tier 3] Rewrite as `SELECT oi.order_id, p.name, oi.qty FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id "
        "WHERE p.category = 'books' AND oi.qty >= 2 ORDER BY oi.order_id, p.name`.",
    ),
    "03_cartesian_join": (
        "[DBA tier 1] This is an accidental cartesian join. `FROM events e, tenants t` "
        "combined with `t.id = e.tenant_id + 0` prevents the optimizer from seeing a "
        "clean tenant-key join early.",
        "[DBA tier 2] Turn the comma join into an explicit equijoin on the tenant key. "
        "Move only `t.id = e.tenant_id` into `ON`; keep the severity filter in `WHERE` "
        "and preserve grouping by tenant tier.",
        "[DBA tier 3] Use `SELECT t.tier, COUNT(*) AS n FROM events e JOIN tenants t "
        "ON t.id = e.tenant_id WHERE e.severity IN ('error', 'critical') GROUP BY "
        "t.tier ORDER BY t.tier`. Avoid arithmetic on the join key.",
    ),
    "04_distinct_groupby": (
        "[DBA tier 1] The duplicate-removal work is redundant. `GROUP BY session_id, path` "
        "already emits one row per `(session_id, path)` pair, so a leading `DISTINCT` "
        "adds a second deduplication pass over grouped rows.",
        "[DBA tier 2] Do not introduce a CTE or change the aggregation grain. Remove only "
        "`DISTINCT`; keep `COUNT(*) AS hits`, the same GROUP BY keys, and the same "
        "ordering so row identity and sort order stay stable.",
        "[DBA tier 3] The target shape is `SELECT session_id, path, COUNT(*) AS hits "
        "FROM pageviews GROUP BY session_id, path ORDER BY session_id, path`. Validate "
        "against the baseline result before comparing runtime.",
    ),
    "05_nested_subquery": (
        "[DBA tier 1] The nested `IN` clauses express a semi-join: authors who wrote "
        "comments on published articles. The important identity is `comments.author_id`, "
        "not `articles.author_id`.",
        "[DBA tier 2] Flatten to `authors -> comments -> articles`, filter "
        "`articles.status = 'published'`, and select distinct author display names. "
        "`DISTINCT` is required here because one author can have many qualifying comments.",
        "[DBA tier 3] Use `SELECT DISTINCT a.display_name FROM authors a JOIN comments c "
        "ON c.author_id = a.id JOIN articles ar ON ar.id = c.article_id WHERE "
        "ar.status = 'published' ORDER BY a.display_name`.",
    ),
    "06_having_as_where": (
        "[DBA tier 1] `status = 'fulfilled'` is a row-level predicate sitting in HAVING, "
        "so the engine groups every status first and discards most groups afterward. "
        "Only `SUM(amount_cents) >= 100000` truly belongs after aggregation.",
        "[DBA tier 2] Move the status filter into `WHERE` before the GROUP BY. Keep "
        "`status` in the projection and grouping to preserve the result shape, then "
        "leave the aggregate threshold in HAVING.",
        "[DBA tier 3] Use `SELECT tenant_id, status, SUM(amount_cents) AS total_cents "
        "FROM orders WHERE status = 'fulfilled' GROUP BY tenant_id, status HAVING "
        "SUM(amount_cents) >= 100000 ORDER BY tenant_id`.",
    ),
    "07_drift_column_rename": (
        "[DBA tier 1] If the old aggregation now fails with an unknown `user_id`, this is "
        "schema drift rather than a performance issue. Read the changelog or describe "
        "`orders`; `users.id` is unchanged, but the order-owner column moved.",
        "[DBA tier 2] Replace every reference to `orders.user_id` with `orders.account_id` "
        "in SELECT, GROUP BY, JOIN, and ORDER BY positions. Do not change the aggregate "
        "logic; the rename preserves row semantics.",
        "[DBA tier 3] Submit `SELECT account_id, COUNT(*) AS n_orders, ROUND(SUM(amount), 2) "
        "AS total FROM orders GROUP BY account_id ORDER BY account_id`. Validate that "
        "counts and totals match the pre-drift business result.",
    ),
    "08_drift_date_format": (
        "[DBA tier 1] The `events.ts` identifier still exists, but its type changed from "
        "ISO text to BIGINT epoch milliseconds. A string date predicate can parse or "
        "compare incorrectly; confirm with `describe_table('events')` and samples.",
        "[DBA tier 2] Keep the same half-open UTC day window, but express both bounds as "
        "epoch-ms integers. For 2026-04-21T00:00:00Z through the next midnight, use "
        "`1776729600000 <= ts < 1776816000000`.",
        "[DBA tier 3] Use `SELECT kind, COUNT(*) AS n FROM events WHERE ts >= "
        "1776729600000 AND ts < 1776816000000 GROUP BY kind ORDER BY kind`. Do not quote "
        "the bounds; they must be numeric comparisons against the BIGINT column.",
    ),
    "09_drift_enum_rule": (
        "[DBA tier 1] A formerly valid equality on `status = 'active'` now silently loses "
        "rows because the business state was split into multiple stored labels. Sample "
        "`tenants.status` before assuming the old lowercase value still exists.",
        '[DBA tier 2] Preserve the business meaning "active tenants" by filtering on the '
        "union of replacement labels. Keep the same grouping by tier and ordering; only "
        "the status predicate changes.",
        "[DBA tier 3] Use `SELECT tier, COUNT(*) AS n FROM tenants WHERE status IN "
        "('ACTIVE', 'ACTIVE_V2') GROUP BY tier ORDER BY tier`. Avoid `LOWER(status) = "
        "'active'`; it misses `ACTIVE_V2`.",
    ),
    "10_drift_field_deprecation": (
        "[DBA tier 1] The inline `posts.author_name` column was normalized away. Describe "
        "`posts` and list tables: you should see `posts.users_id` plus a new `users` "
        "lookup carrying the human-readable name.",
        "[DBA tier 2] Join `posts` to `users` through the new FK, group by `u.full_name`, "
        "and alias it back to `author_name` so the result keeps the old report shape. "
        "The post count still comes from `posts`.",
        "[DBA tier 3] Use `SELECT u.full_name AS author_name, COUNT(*) AS n_posts FROM "
        "posts p JOIN users u ON u.id = p.users_id GROUP BY u.full_name ORDER BY "
        "u.full_name`.",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_enabled(
    reset_flag: bool | None = None, *, env_var: str = "SQL_DRIFT_ENABLE_DBA_ORACLE"
) -> bool:
    """Resolve the feature flag from (reset kwarg, env var, default-off)."""
    if reset_flag is not None:
        return bool(reset_flag)
    raw = os.environ.get(env_var, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def get_hint(scenario_id: str, tier: int) -> str:
    """Return the hint for ``(scenario_id, tier)``; clamps tier to [1, 3].

    Raises :class:`KeyError` on unknown scenario so tests can detect when
    a new scenario was added without a hint table entry.
    """
    if scenario_id not in _HINTS:
        raise KeyError(f"no DBA hints for scenario_id={scenario_id!r}; known: {sorted(_HINTS)}")
    tier = max(1, min(3, int(tier)))
    return _HINTS[scenario_id][tier - 1]


def has_hints(scenario_id: str) -> bool:
    return scenario_id in _HINTS


def known_scenarios() -> frozenset[str]:
    return frozenset(_HINTS)


__all__ = [
    "get_hint",
    "has_hints",
    "is_enabled",
    "known_scenarios",
]
