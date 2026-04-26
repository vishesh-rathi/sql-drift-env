"""Drift engine: four atomic, idempotent DDL operations.

Each apply_* function mutates ``conn`` in place inside a DuckDB
``BEGIN; ... COMMIT`` pair and returns a machine-readable changelog
string. Humans consume the string via the :class:`read_changelog`
tool; the rubric consults a separate drift-acknowledgement flag on the
runtime state, not the string itself.

Idempotency is enforced via a post-condition schema probe: once the drift
has been applied (the target column / enum value is in the expected
post-state), a second call short-circuits with the same changelog string.
This matters because the environment's drift-trigger check runs every
step and needs to be safe to retry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb


# DuckDB auto-commits DDL and forbids mixing multi-statement transactions
# with schema alterations across commit boundaries. Each drift operation
# therefore executes its statements sequentially on the default
# auto-commit connection; individual DML statements (UPDATEs) are
# internally atomic at the statement level, which is sufficient for the
# fixture mutation the env needs. If a drift operation raises mid-way we
# tear down and re-seed the DuckDB via ScenarioSpec.materialize — there's
# no long-lived on-disk state to roll back.


def _table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows]


def _table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    rows = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    return bool(rows and rows[0])


# =============================================================================
# Column rename
# =============================================================================


def apply_column_rename(conn: duckdb.DuckDBPyConnection, payload: dict[str, Any]) -> str:
    """``{"table": str, "old": str, "new": str}``."""
    table = payload["table"]
    old = payload["old"]
    new = payload["new"]

    cols = _table_columns(conn, table)
    if new in cols and old not in cols:
        return f"rename_already_applied:{table}.{old}->{new}"
    if old not in cols:
        raise ValueError(f"column_rename: {table}.{old} missing (cols={cols})")

    conn.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{old}" TO "{new}"')
    return f"rename:{table}.{old}->{new}"


# =============================================================================
# Date format change (iso_string → epoch_ms)
# =============================================================================


def apply_date_format_change(conn: duckdb.DuckDBPyConnection, payload: dict[str, Any]) -> str:
    """``{"table": str, "col": str, "from": "iso_string", "to": "epoch_ms"}``.

    Only the one direction is supported for now; the payload still carries
    from/to for forward-compatibility and audit.
    """
    table = payload["table"]
    col = payload["col"]
    from_fmt = payload.get("from", "iso_string")
    to_fmt = payload.get("to", "epoch_ms")
    if (from_fmt, to_fmt) != ("iso_string", "epoch_ms"):
        raise NotImplementedError(
            f"date_format_change only supports iso_string→epoch_ms, got {from_fmt}→{to_fmt}"
        )

    cols = _table_columns(conn, table)
    # Idempotent: once column is BIGINT, consider it applied.
    type_row = conn.execute(
        "SELECT data_type FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
        [table, col],
    ).fetchone()
    if type_row is None:
        raise ValueError(f"date_format_change: {table}.{col} missing (cols={cols})")
    if "BIGINT" in type_row[0].upper() or "INT" in type_row[0].upper():
        return f"date_format_already_applied:{table}.{col}"

    tmp = f"{col}_epoch_ms"
    conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{tmp}" BIGINT')
    conn.execute(
        f'UPDATE "{table}" SET "{tmp}" = '
        f'CAST(EXTRACT(EPOCH FROM CAST("{col}" AS TIMESTAMP)) * 1000 AS BIGINT)'
    )
    conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{col}"')
    conn.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{tmp}" TO "{col}"')

    return f"date_format:{table}.{col}:iso_string->epoch_ms"


# =============================================================================
# Enum rule change (split `old_value` into N new values)
# =============================================================================


def apply_enum_rule_change(conn: duckdb.DuckDBPyConnection, payload: dict[str, Any]) -> str:
    """``{"table": str, "col": str, "old_value": str, "new_values": list[str]}``.

    Rows holding ``old_value`` are re-distributed deterministically into
    ``new_values`` (round-robin by rowid) so the split is reproducible.
    """
    table = payload["table"]
    col = payload["col"]
    old_value = payload["old_value"]
    new_values: list[str] = list(payload["new_values"])
    if not new_values:
        raise ValueError("enum_rule_change: new_values must be non-empty")

    count_row = conn.execute(
        f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" = ?', [old_value]
    ).fetchone()
    count_old = count_row[0] if count_row is not None else 0
    # Idempotent: if old_value has already been drained AND any of the
    # new_values is present, treat as applied.
    if count_old == 0:
        has_new_row = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IN ({",".join("?" * len(new_values))})',
            new_values,
        ).fetchone()
        has_new = has_new_row[0] if has_new_row is not None else 0
        if has_new > 0:
            return f"enum_rule_already_applied:{table}.{col}:{old_value}->{new_values}"

    # Deterministic split by rowid mod N.
    case_branches = " ".join(
        f"WHEN mod(rid, {len(new_values)}) = {i} THEN '{v}'" for i, v in enumerate(new_values)
    )
    conn.execute(
        f"CREATE TEMP TABLE _enum_remap AS "
        f"SELECT rowid AS rid, "
        f"CASE {case_branches} END AS new_val "
        f'FROM "{table}" WHERE "{col}" = ?',
        [old_value],
    )
    conn.execute(
        f'UPDATE "{table}" SET "{col}" = _enum_remap.new_val '
        f'FROM _enum_remap WHERE _enum_remap.rid = "{table}".rowid'
    )
    conn.execute("DROP TABLE _enum_remap")

    return f"enum_rule:{table}.{col}:{old_value}->{'+'.join(new_values)}"


# =============================================================================
# Field deprecation (replace inline string col with FK lookup)
# =============================================================================


def apply_field_deprecation(conn: duckdb.DuckDBPyConnection, payload: dict[str, Any]) -> str:
    """``{"orig": (table, col), "lookup": (table, id_col, name_col)}``.

    - Creates the lookup table (if missing) and seeds it with distinct values
      observed on ``orig.col``.
    - Adds ``orig.<lookup_id>`` with a FK-style backfill.
    - Drops ``orig.col``.
    """
    orig_table, orig_col = payload["orig"]
    lookup_table, lookup_id_col, lookup_name_col = payload["lookup"]
    new_fk_col = f"{lookup_table}_{lookup_id_col}"  # e.g. "users_id"

    orig_cols = _table_columns(conn, orig_table)
    if orig_col not in orig_cols and new_fk_col in orig_cols:
        return f"field_deprecation_already_applied:{orig_table}.{orig_col}"
    if orig_col not in orig_cols:
        raise ValueError(f"field_deprecation: {orig_table}.{orig_col} missing (cols={orig_cols})")

    if not _table_exists(conn, lookup_table):
        conn.execute(
            f'CREATE TABLE "{lookup_table}" ('
            f' "{lookup_id_col}" BIGINT PRIMARY KEY,'
            f' "{lookup_name_col}" VARCHAR'
            ");"
        )
    conn.execute(
        f'INSERT INTO "{lookup_table}" ("{lookup_id_col}", "{lookup_name_col}") '
        f"SELECT ROW_NUMBER() OVER (ORDER BY v) + "
        f'COALESCE((SELECT MAX("{lookup_id_col}") FROM "{lookup_table}"), 0), v '
        f'FROM (SELECT DISTINCT "{orig_col}" AS v FROM "{orig_table}") '
        f"WHERE v IS NOT NULL "
        f'  AND v NOT IN (SELECT "{lookup_name_col}" FROM "{lookup_table}");'
    )
    conn.execute(f'ALTER TABLE "{orig_table}" ADD COLUMN "{new_fk_col}" BIGINT')
    conn.execute(
        f'UPDATE "{orig_table}" SET "{new_fk_col}" = lookup."{lookup_id_col}" '
        f'FROM "{lookup_table}" lookup '
        f'WHERE lookup."{lookup_name_col}" = "{orig_table}"."{orig_col}"'
    )
    conn.execute(f'ALTER TABLE "{orig_table}" DROP COLUMN "{orig_col}"')

    return (
        f"field_deprecation:{orig_table}.{orig_col}->"
        f"{orig_table}.{new_fk_col}→{lookup_table}.{lookup_name_col}"
    )


# =============================================================================
# Dispatcher
# =============================================================================


DRIFT_HANDLERS = {
    "column_rename": apply_column_rename,
    "date_format": apply_date_format_change,
    "enum_rule": apply_enum_rule_change,
    "field_deprecation": apply_field_deprecation,
}


def apply_drift(conn: duckdb.DuckDBPyConnection, kind: str, payload: dict[str, Any]) -> str:
    if kind not in DRIFT_HANDLERS:
        raise ValueError(f"unknown drift kind={kind!r}; known: {sorted(DRIFT_HANDLERS)}")
    return DRIFT_HANDLERS[kind](conn, payload)


__all__ = [
    "DRIFT_HANDLERS",
    "apply_column_rename",
    "apply_date_format_change",
    "apply_drift",
    "apply_enum_rule_change",
    "apply_field_deprecation",
]
