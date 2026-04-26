"""Scenario 10 — drift: inline string col replaced by FK lookup.

Baseline groups posts by the inline `author_name` string. Under drift, a
`users(id, full_name)` lookup is created, `posts.author_name` is dropped,
and `posts.users_id` is added with a backfilled FK. The correct rewrite
joins through `users` and groups by `full_name`.
"""

from __future__ import annotations

import duckdb

from ._fixtures import seeded_rng, unique_names, zipfian_choices
from .base import BuilderResult, DriftConfig, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_authors = max(40, scale // 10)
    n_posts = scale * 4

    author_names = unique_names(rng, n_authors, prefix="author")
    post_author_idx = zipfian_choices(rng, list(range(n_authors)), n_posts)
    post_author_names = [author_names[i] for i in post_author_idx]

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE posts( id BIGINT PRIMARY KEY, author_name VARCHAR, body VARCHAR);")
    conn.executemany(
        "INSERT INTO posts VALUES (?, ?, ?)",
        [(i, n, f"body {i}") for i, n in enumerate(post_author_names, 1)],
    )

    baseline_sql = (
        "SELECT author_name, COUNT(*) AS n_posts "
        "FROM posts GROUP BY author_name "
        "ORDER BY author_name"
    )
    gt_sql_predrift = baseline_sql
    gt_sql_postdrift = (
        "SELECT u.full_name AS author_name, COUNT(*) AS n_posts "
        "FROM posts p JOIN users u ON u.id = p.users_id "
        "GROUP BY u.full_name ORDER BY u.full_name"
    )

    synopsis = (
        "posts(id PK, author_name, body). Under drift, posts.author_name is "
        "deprecated; a new users(id PK, full_name) table is created and "
        "posts gains a users_id FK. Rewrites must JOIN through users."
    )
    return (
        conn,
        baseline_sql,
        gt_sql_predrift,
        gt_sql_postdrift,
        synopsis,
        frozenset({"users", "users_id", "full_name"}),
        frozenset({"author_name"}),
    )


SPEC = ScenarioSpec(
    scenario_id="10_drift_field_deprecation",
    family="cms",
    tags=frozenset({"drift", "field_deprecation", "fk_backfill", "cms"}),
    drift_config=DriftConfig(
        kind="field_deprecation",
        payload={
            "orig": ("posts", "author_name"),
            "lookup": ("users", "id", "full_name"),
        },
    ),
    builder=_build,
    base_scale=1_500,
)
