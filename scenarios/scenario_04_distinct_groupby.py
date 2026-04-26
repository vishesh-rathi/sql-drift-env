"""Scenario 04 — redundant DISTINCT on top of GROUP BY.

Baseline applies DISTINCT after already grouping, forcing a second
deduplication over a result whose GROUP BY keys are already unique.
Rewrite drops the DISTINCT entirely.

Schema: pageviews(id, session_id, path, ts_epoch_s).
"""

from __future__ import annotations

import duckdb

from ._fixtures import categorical_choices, seeded_rng, unique_names, zipfian_choices
from .base import BuilderResult, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_sessions = max(200, scale // 4)
    n_views = scale * 8

    session_ids = unique_names(rng, n_sessions, prefix="sess")
    paths = categorical_choices(
        rng,
        [f"/path/{p}" for p in ["home", "about", "product", "cart", "checkout", "help"]],
        n_views,
    )
    view_sessions = zipfian_choices(rng, list(range(n_sessions)), n_views)
    view_session_ids = [session_ids[i] for i in view_sessions]
    view_ts = [1_700_000_000 + rng.randrange(30 * 86_400) for _ in range(n_views)]

    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE pageviews("
        " id BIGINT PRIMARY KEY, session_id VARCHAR, path VARCHAR, ts_epoch_s BIGINT);"
    )
    conn.executemany(
        "INSERT INTO pageviews VALUES (?, ?, ?, ?)",
        [
            (i, sid, p, ts)
            for i, (sid, p, ts) in enumerate(
                zip(view_session_ids, paths, view_ts, strict=False), start=1
            )
        ],
    )

    baseline_sql = (
        "SELECT DISTINCT session_id, path, COUNT(*) AS hits "
        "FROM pageviews GROUP BY session_id, path "
        "ORDER BY session_id, path"
    )
    gt_sql_predrift = (
        "SELECT session_id, path, COUNT(*) AS hits "
        "FROM pageviews GROUP BY session_id, path "
        "ORDER BY session_id, path"
    )

    synopsis = (
        "pageviews(id PK, session_id, path, ts_epoch_s). "
        "Baseline applies redundant DISTINCT on a GROUP BY whose keys are already unique."
    )
    return conn, baseline_sql, gt_sql_predrift, None, synopsis, frozenset(), frozenset()


SPEC = ScenarioSpec(
    scenario_id="04_distinct_groupby",
    family="saas_logs",
    tags=frozenset({"distinct", "redundant_distinct", "group_by", "saas_logs"}),
    drift_config=None,
    builder=_build,
    base_scale=500,
)
