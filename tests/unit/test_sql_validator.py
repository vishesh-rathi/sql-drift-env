"""Unit tests for :func:`server.sql_drift_env_environment._validate_read_only_sql`.

These are the gatekeeper checks that separate agent-submitted SQL
from DuckDB's broader surface. They must accept any legitimate read
against the scenario's in-memory tables and reject every known
sandbox-escape vector:

* statements other than ``SELECT`` / ``WITH`` (DDL, DML, ATTACH, COPY),
* table-valued functions that read from the host filesystem
  (``read_csv``, ``read_parquet``, ``read_json_auto``, …),
* DuckDB introspection helpers that leak engine state
  (``duckdb_settings``, ``duckdb_secrets``, ``parquet_metadata``, …),
* ``SELECT * FROM 'path/to/file'`` bare-path FROM forms.
"""

from __future__ import annotations

import pytest

from server.sql_drift_env_environment import _validate_read_only_sql


class TestSafeSql:
    """Legitimate agent-facing queries must round-trip through the validator."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "SELECT COUNT(*) FROM users",
            "SELECT COUNT(*) FROM users WHERE id > 0",
            "SELECT u.id, o.amount FROM users u JOIN orders o ON o.user_id = u.id",
            "WITH t AS (SELECT 1 AS x) SELECT * FROM t",
            "SELECT UPPER(name), LOWER(email) FROM users",
            "SELECT * FROM main.orders",
            "SELECT * FROM generate_series(1, 10)",
            "SELECT tier, COUNT(*) n FROM tenants GROUP BY tier",
        ],
    )
    def test_accepts(self, sql: str) -> None:
        _validate_read_only_sql(sql)


class TestStatementLevelRejection:
    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO users VALUES (1, 'x')",
            "UPDATE users SET name = 'x'",
            "DELETE FROM users",
            "CREATE TABLE evil(x INT)",
            "DROP TABLE users",
            "ALTER TABLE users ADD COLUMN x INT",
            "TRUNCATE users",
            "COPY users TO '/tmp/dump.csv'",
            "ATTACH 'other.duckdb'",
            "DETACH 'other.duckdb'",
            "PRAGMA version",
            "SELECT 1; SELECT 2",
        ],
    )
    def test_rejects(self, sql: str) -> None:
        with pytest.raises(ValueError):
            _validate_read_only_sql(sql)


class TestFileReadingFunctionsRejected:
    """Sandbox-escape function calls inside an otherwise-SELECT shape."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM read_csv('x.csv')",
            "SELECT * FROM read_csv_auto('x.csv')",
            "SELECT * FROM read_parquet('x.parquet')",
            "SELECT * FROM read_json('x.json')",
            "SELECT * FROM read_json_auto('x.json')",
            "SELECT * FROM read_ndjson_auto('x.ndjson')",
            "SELECT read_text('/etc/passwd')",
            "SELECT read_blob('/etc/hosts')",
            "SELECT * FROM glob('/etc/*')",
            "SELECT * FROM parquet_metadata('x.parquet')",
            "SELECT * FROM parquet_schema('x.parquet')",
            "SELECT * FROM sniff_csv('x.csv')",
            "SELECT * FROM duckdb_settings()",
            "SELECT * FROM duckdb_secrets()",
            "SELECT * FROM duckdb_functions()",
            "SELECT * FROM duckdb_extensions()",
            "SELECT load_aws_credentials()",
            "SELECT install_extension('httpfs')",
        ],
    )
    def test_rejects(self, sql: str) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            _validate_read_only_sql(sql)


class TestBarePathFromRejected:
    """``SELECT * FROM '<path>'`` is DuckDB's filesystem-auto-detect sugar."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM 'x.csv'",
            "SELECT * FROM 'x.parquet'",
            "SELECT * FROM '/etc/passwd'",
            "SELECT * FROM '~/secrets.json'",
            "SELECT * FROM 'https://example.com/data.csv'",
            "SELECT * FROM 'data/orders_2024.csv'",
        ],
    )
    def test_rejects(self, sql: str) -> None:
        with pytest.raises(ValueError, match="not a valid unquoted SQL name"):
            _validate_read_only_sql(sql)


class TestMalformedSql:
    def test_parse_error_rejected(self) -> None:
        with pytest.raises(ValueError, match="failed to parse"):
            _validate_read_only_sql("SELECT * FROM WHERE")
