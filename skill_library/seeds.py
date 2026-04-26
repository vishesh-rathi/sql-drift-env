"""Hand-authored pre-seed library — 8 playbook entries + 4 drift cards.

Tag sets here match the tags on each corresponding scenario
so a Jaccard top-k against the scenario's tag set returns
a relevant entry for every one of the 10 scenarios — with two
generic fallbacks for coverage on novel anti-patterns.
"""

from __future__ import annotations

from skill_library.entries import DriftAdaptationCard, PlaybookEntry

PRESEED_PLAYBOOK: tuple[PlaybookEntry, ...] = (
    # 1. Correlated subquery → LEFT JOIN + GROUP BY
    PlaybookEntry(
        tag_set=frozenset({"correlated_subquery", "projection_subquery"}),
        before_snippet=(
            "SELECT u.*, (SELECT COUNT(*) FROM orders o WHERE o.user_id=u.id) FROM users u"
        ),
        after_snippet=(
            "SELECT u.*, COALESCE(c.n, 0) FROM users u "
            "LEFT JOIN (SELECT user_id, COUNT(*) n FROM orders GROUP BY user_id) c "
            "ON c.user_id = u.id"
        ),
        avg_speedup=6.0,
        scenario_family="ecommerce",
    ),
    # 2. SELECT * + join → project only needed columns
    PlaybookEntry(
        tag_set=frozenset({"select_star", "over_projection", "join"}),
        before_snippet="SELECT * FROM a JOIN b ON a.id=b.a_id",
        after_snippet="SELECT a.id, a.name, b.amount FROM a JOIN b ON a.id=b.a_id",
        avg_speedup=2.5,
        scenario_family="ecommerce",
    ),
    # 3. Cartesian join — add explicit ON clause
    PlaybookEntry(
        tag_set=frozenset({"cartesian", "missing_join_condition"}),
        before_snippet="SELECT * FROM a, b WHERE a.region = 'US'",
        after_snippet="SELECT a.col FROM a JOIN b ON a.id = b.a_id WHERE a.region = 'US'",
        avg_speedup=50.0,
        scenario_family="events",
    ),
    # 4. DISTINCT on GROUP BY — drop one
    PlaybookEntry(
        tag_set=frozenset({"distinct", "redundant_distinct", "group_by"}),
        before_snippet="SELECT DISTINCT tenant_id, count(*) FROM logs GROUP BY tenant_id",
        after_snippet="SELECT tenant_id, count(*) FROM logs GROUP BY tenant_id",
        avg_speedup=1.4,
        scenario_family="saas_logs",
    ),
    # 5. Nested IN-subquery → JOIN
    PlaybookEntry(
        tag_set=frozenset({"nested_subquery", "in_subquery"}),
        before_snippet="WHERE id IN (SELECT x_id FROM x WHERE ... )",
        after_snippet="JOIN x ON x.x_id = table.id WHERE ...",
        avg_speedup=3.0,
        scenario_family="cms",
    ),
    # 6. HAVING filter on groupable column → push to WHERE
    PlaybookEntry(
        tag_set=frozenset({"having_as_where", "aggregate_filter"}),
        before_snippet="GROUP BY x, status HAVING status = 'fulfilled'",
        after_snippet="WHERE status = 'fulfilled' GROUP BY x",
        avg_speedup=2.0,
        scenario_family="ecommerce",
    ),
    # 7. Generic: prefer JOINs over correlated subqueries
    PlaybookEntry(
        tag_set=frozenset({"subquery", "generic"}),
        before_snippet="scalar subquery in SELECT list",
        after_snippet="LEFT JOIN with aggregated CTE",
        avg_speedup=4.0,
        scenario_family="ecommerce",
    ),
    # 8. Generic: project only used columns
    PlaybookEntry(
        tag_set=frozenset({"over_projection", "generic"}),
        before_snippet="SELECT *",
        after_snippet="SELECT <only needed columns>",
        avg_speedup=1.8,
        scenario_family="ecommerce",
    ),
)


PRESEED_DRIFT_CARDS: tuple[DriftAdaptationCard, ...] = (
    DriftAdaptationCard(
        drift_kind="column_rename",
        symptom_regex=r'column ".+" does not exist',
        recovery_template=(
            "Read the changelog, update every identifier referencing the old column, and resubmit."
        ),
        success_rate=0.9,
    ),
    DriftAdaptationCard(
        drift_kind="date_format",
        symptom_regex=r"Could not convert string .+ to TIMESTAMP|BIGINT",
        recovery_template=(
            "Epoch-ms columns are BIGINT; cast your filter bounds with "
            "`EXTRACT(EPOCH FROM TIMESTAMP '...') * 1000` or use numeric literals."
        ),
        success_rate=0.85,
    ),
    DriftAdaptationCard(
        drift_kind="enum_rule",
        symptom_regex=r"(empty|zero) result set on filter `... = 'active'`",
        recovery_template=(
            "A single enum value may have been split into several; use `IN "
            "('ACTIVE', 'ACTIVE_V2')` instead of equality."
        ),
        success_rate=0.8,
    ),
    DriftAdaptationCard(
        drift_kind="field_deprecation",
        symptom_regex=r'column ".+" does not exist|non-existent column',
        recovery_template=(
            "The inline field was replaced by a FK; JOIN the lookup table and "
            "project the human-readable name from there."
        ),
        success_rate=0.75,
    ),
)


__all__ = ["PRESEED_DRIFT_CARDS", "PRESEED_PLAYBOOK"]
