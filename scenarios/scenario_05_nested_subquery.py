"""Scenario 05 — deeply nested subquery anti-pattern.

Baseline stacks two layers of `IN (SELECT ...)` where a single join + GROUP
BY expresses the same intent. The rewrite flattens the nesting.

Schema: articles(id, author_id, published_at), comments(id, article_id, author_id),
        authors(id, display_name, is_verified).
"""

from __future__ import annotations

import duckdb

from ._fixtures import categorical_choices, seeded_rng, unique_names, zipfian_choices
from .base import BuilderResult, ScenarioSpec


def _build(spec: ScenarioSpec, seed: int, scale: int) -> BuilderResult:
    rng = seeded_rng(spec.scenario_id, seed, scale)
    n_authors = max(50, scale // 8)
    n_articles = scale
    n_comments = scale * 4

    author_ids = list(range(1, n_authors + 1))
    display_names = unique_names(rng, n_authors, prefix="author")
    verified = rng.choices([True, False], weights=[0.2, 0.8], k=n_authors)

    article_ids = list(range(1, n_articles + 1))
    article_authors = zipfian_choices(rng, author_ids, n_articles)
    article_published = [1_700_000_000 + rng.randrange(120 * 86_400) for _ in range(n_articles)]
    article_statuses = categorical_choices(
        rng, ["draft", "published", "archived"], n_articles, weights=[0.2, 0.7, 0.1]
    )

    comment_ids = list(range(1, n_comments + 1))
    comment_articles = rng.choices(article_ids, k=n_comments)
    comment_authors = zipfian_choices(rng, author_ids, n_comments)

    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE authors( id BIGINT PRIMARY KEY, display_name VARCHAR, is_verified BOOLEAN);"
    )
    conn.execute(
        "CREATE TABLE articles("
        " id BIGINT PRIMARY KEY, author_id BIGINT, published_at_epoch_s BIGINT,"
        " status VARCHAR);"
    )
    conn.execute(
        "CREATE TABLE comments( id BIGINT PRIMARY KEY, article_id BIGINT, author_id BIGINT);"
    )
    conn.executemany(
        "INSERT INTO authors VALUES (?, ?, ?)",
        list(zip(author_ids, display_names, verified, strict=False)),
    )
    conn.executemany(
        "INSERT INTO articles VALUES (?, ?, ?, ?)",
        list(zip(article_ids, article_authors, article_published, article_statuses, strict=False)),
    )
    conn.executemany(
        "INSERT INTO comments VALUES (?, ?, ?)",
        list(zip(comment_ids, comment_articles, comment_authors, strict=False)),
    )

    baseline_sql = (
        "SELECT display_name "
        "FROM authors "
        "WHERE id IN ("
        "  SELECT author_id FROM comments "
        "  WHERE article_id IN ("
        "    SELECT id FROM articles WHERE status = 'published'"
        "  )"
        ") "
        "ORDER BY display_name"
    )
    gt_sql_predrift = (
        "SELECT DISTINCT a.display_name "
        "FROM authors a "
        "JOIN comments c ON c.author_id = a.id "
        "JOIN articles ar ON ar.id = c.article_id "
        "WHERE ar.status = 'published' "
        "ORDER BY a.display_name"
    )

    synopsis = (
        "authors(id PK, display_name, is_verified); "
        "articles(id PK, author_id→authors.id, published_at_epoch_s, status); "
        "comments(id PK, article_id→articles.id, author_id→authors.id). "
        "Baseline chains two IN-subqueries where one JOIN + DISTINCT suffices."
    )
    return conn, baseline_sql, gt_sql_predrift, None, synopsis, frozenset(), frozenset()


SPEC = ScenarioSpec(
    scenario_id="05_nested_subquery",
    family="cms",
    tags=frozenset({"nested_subquery", "in_subquery", "cms"}),
    drift_config=None,
    builder=_build,
    base_scale=1_500,
)
