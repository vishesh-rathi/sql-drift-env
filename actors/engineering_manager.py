"""Engineering-manager narrator — authors deterministic drift runbooks.

Consumed by the ``read_changelog`` tool. Output is deterministic per
``DriftConfig`` so tests can assert it character-for-character and the
agent can learn to parse drift kinds from the text.

Entries are concise migration notes: what changed, why the old query
breaks, how to adapt, and what to validate before submission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scenarios.base import DriftConfig


def _sentence(prefix: str, body: str) -> str:
    return f"[changelog] {prefix} {body}".strip()


def author_changelog(drift_config: DriftConfig) -> str:
    kind = drift_config.kind
    p = drift_config.payload
    if kind == "column_rename":
        return _sentence(
            "Schema change -",
            (
                f"`{p['table']}.{p['old']}` was renamed to `{p['table']}.{p['new']}` "
                "as part of a naming normalization. Impact: queries that SELECT, JOIN, "
                "GROUP BY, ORDER BY, or filter on the old identifier now fail at bind "
                "time. Migration: use the new column everywhere the old one appeared; "
                "related table primary keys are unchanged. Validate that counts and "
                "aggregates are unchanged after the rename."
            ),
        )
    if kind == "date_format":
        return _sentence(
            "Format change -",
            (
                f"`{p['table']}.{p['col']}` no longer stores ISO-8601 strings; it now "
                "stores BIGINT milliseconds since the Unix epoch. Impact: quoted "
                "timestamp literals and text comparisons no longer express the right "
                "predicate. Migration: keep the same half-open time window, but compare "
                "against numeric epoch-ms bounds. Validate with describe_table and a "
                "small sample before submitting."
            ),
        )
    if kind == "enum_rule":
        new_values = ", ".join(f"`{v}`" for v in p["new_values"])
        return _sentence(
            "Business-rule change -",
            (
                f"`{p['table']}.{p['col']}` value `{p['old_value']}` has been split "
                f"into {new_values}. Impact: equality predicates on the old value "
                "silently undercount after the deploy. Migration: replace the single "
                "value predicate with an IN predicate over every replacement label. "
                "Validate by sampling the enum distribution and preserving the existing "
                "GROUP BY and projection."
            ),
        )
    if kind == "field_deprecation":
        orig_t, orig_c = p["orig"]
        lt, lid, lname = p["lookup"]
        fk_col = f"{lt}_{lid}"
        return _sentence(
            "Deprecation -",
            (
                f"`{orig_t}.{orig_c}` was deprecated and replaced by `{orig_t}.{fk_col}` "
                f"pointing at `{lt}.{lid}`; the display value now lives on `{lt}.{lname}`. "
                "Impact: projections or groups over the old inline string fail after "
                "drift. Migration: join through the lookup table, project the display "
                "column, and keep the old output alias if callers expect it. Validate "
                "row counts after the join."
            ),
        )
    raise ValueError(f"unknown drift kind={kind!r}")


__all__ = ["author_changelog"]
