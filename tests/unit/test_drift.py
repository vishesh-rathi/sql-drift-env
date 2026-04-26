"""P4 — unit tests for drift engine + changelog authoring."""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from engine.drift import (
    apply_column_rename,
    apply_date_format_change,
    apply_drift,
    apply_enum_rule_change,
    apply_field_deprecation,
)


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    yield c
    c.close()


# =============================================================================
# Column rename
# =============================================================================


class TestColumnRename:
    def _seed(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("CREATE TABLE orders(id BIGINT, user_id BIGINT, amount DOUBLE);")
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?)", [(1, 10, 9.99), (2, 11, 1.23)])

    def test_renames_column(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        entry = apply_column_rename(
            conn, {"table": "orders", "old": "user_id", "new": "account_id"}
        )
        assert entry.startswith("rename:")
        cols = {r[1] for r in conn.execute("PRAGMA table_info('orders')").fetchall()}
        assert "account_id" in cols and "user_id" not in cols

    def test_idempotent_reapply(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        apply_column_rename(conn, {"table": "orders", "old": "user_id", "new": "account_id"})
        entry2 = apply_column_rename(
            conn, {"table": "orders", "old": "user_id", "new": "account_id"}
        )
        assert entry2.startswith("rename_already_applied:")

    def test_missing_column_errors(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        with pytest.raises(ValueError, match="missing"):
            apply_column_rename(conn, {"table": "orders", "old": "nope", "new": "x"})


# =============================================================================
# Date format change
# =============================================================================


class TestDateFormatChange:
    def _seed(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("CREATE TABLE events(id BIGINT, ts VARCHAR);")
        conn.executemany(
            "INSERT INTO events VALUES (?, ?)",
            [(1, "2026-04-21T12:00:00Z"), (2, "2026-04-21T12:00:01Z")],
        )

    def test_converts_iso_to_epoch_ms(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        entry = apply_date_format_change(
            conn, {"table": "events", "col": "ts", "from": "iso_string", "to": "epoch_ms"}
        )
        assert entry.startswith("date_format:")
        ttype = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'events' AND column_name = 'ts'"
        ).fetchone()[0]
        assert "INT" in ttype.upper()
        vals = [r[0] for r in conn.execute("SELECT ts FROM events ORDER BY id").fetchall()]
        assert vals[1] - vals[0] == 1_000  # one second apart in ms

    def test_idempotent(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        apply_date_format_change(conn, {"table": "events", "col": "ts"})
        entry2 = apply_date_format_change(conn, {"table": "events", "col": "ts"})
        assert entry2.startswith("date_format_already_applied:")


# =============================================================================
# Enum rule change
# =============================================================================


class TestEnumRuleChange:
    def _seed(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("CREATE TABLE tenants(id BIGINT, status VARCHAR);")
        conn.executemany(
            "INSERT INTO tenants VALUES (?, ?)",
            [(i, "active" if i % 2 == 0 else "inactive") for i in range(1, 11)],
        )

    def test_splits_old_into_new(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        entry = apply_enum_rule_change(
            conn,
            {
                "table": "tenants",
                "col": "status",
                "old_value": "active",
                "new_values": ["ACTIVE", "ACTIVE_V2"],
            },
        )
        assert entry.startswith("enum_rule:")
        distinct = {r[0] for r in conn.execute("SELECT DISTINCT status FROM tenants").fetchall()}
        assert "active" not in distinct
        assert distinct & {"ACTIVE", "ACTIVE_V2"}

    def test_idempotent(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        apply_enum_rule_change(
            conn,
            {
                "table": "tenants",
                "col": "status",
                "old_value": "active",
                "new_values": ["ACTIVE", "ACTIVE_V2"],
            },
        )
        entry2 = apply_enum_rule_change(
            conn,
            {
                "table": "tenants",
                "col": "status",
                "old_value": "active",
                "new_values": ["ACTIVE", "ACTIVE_V2"],
            },
        )
        assert entry2.startswith("enum_rule_already_applied:")

    def test_empty_new_values_rejected(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        with pytest.raises(ValueError, match="new_values"):
            apply_enum_rule_change(
                conn,
                {"table": "tenants", "col": "status", "old_value": "active", "new_values": []},
            )


# =============================================================================
# Field deprecation
# =============================================================================


class TestFieldDeprecation:
    def _seed(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("CREATE TABLE posts(id BIGINT, author_name VARCHAR, body VARCHAR);")
        conn.executemany(
            "INSERT INTO posts VALUES (?, ?, ?)",
            [(1, "Alice", "hi"), (2, "Bob", "yo"), (3, "Alice", "ack")],
        )

    def test_creates_fk_and_drops_orig(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        entry = apply_field_deprecation(
            conn,
            {
                "orig": ("posts", "author_name"),
                "lookup": ("users", "id", "full_name"),
            },
        )
        assert entry.startswith("field_deprecation:")
        post_cols = {r[1] for r in conn.execute("PRAGMA table_info('posts')").fetchall()}
        assert "author_name" not in post_cols
        assert "users_id" in post_cols
        users_rows = conn.execute("SELECT full_name FROM users ORDER BY full_name").fetchall()
        assert [r[0] for r in users_rows] == ["Alice", "Bob"]

    def test_idempotent(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._seed(conn)
        apply_field_deprecation(
            conn,
            {
                "orig": ("posts", "author_name"),
                "lookup": ("users", "id", "full_name"),
            },
        )
        entry2 = apply_field_deprecation(
            conn,
            {
                "orig": ("posts", "author_name"),
                "lookup": ("users", "id", "full_name"),
            },
        )
        assert entry2.startswith("field_deprecation_already_applied:")


# =============================================================================
# Dispatcher
# =============================================================================


class TestDispatch:
    def test_unknown_kind_rejected(self, conn: duckdb.DuckDBPyConnection) -> None:
        with pytest.raises(ValueError, match="unknown drift kind"):
            apply_drift(conn, "fake", {})

    def test_dispatch_to_column_rename(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("CREATE TABLE t(a INTEGER);")
        entry = apply_drift(conn, "column_rename", {"table": "t", "old": "a", "new": "b"})
        assert entry.startswith("rename:")


# =============================================================================
# Engineering-manager changelog
# =============================================================================


class TestChangelog:
    def test_column_rename_narrative(self) -> None:
        from actors.engineering_manager import author_changelog
        from scenarios.base import DriftConfig

        cfg = DriftConfig(
            kind="column_rename",
            payload={"table": "orders", "old": "user_id", "new": "account_id"},
        )
        text = author_changelog(cfg)
        assert "orders" in text and "user_id" in text and "account_id" in text

    def test_date_format_narrative(self) -> None:
        from actors.engineering_manager import author_changelog
        from scenarios.base import DriftConfig

        cfg = DriftConfig(
            kind="date_format",
            payload={"table": "events", "col": "ts", "from": "iso_string", "to": "epoch_ms"},
        )
        text = author_changelog(cfg)
        assert "Unix epoch" in text

    def test_enum_narrative(self) -> None:
        from actors.engineering_manager import author_changelog
        from scenarios.base import DriftConfig

        cfg = DriftConfig(
            kind="enum_rule",
            payload={
                "table": "tenants",
                "col": "status",
                "old_value": "active",
                "new_values": ["ACTIVE", "ACTIVE_V2"],
            },
        )
        text = author_changelog(cfg)
        assert "active" in text and "ACTIVE_V2" in text

    def test_field_deprecation_narrative(self) -> None:
        from actors.engineering_manager import author_changelog
        from scenarios.base import DriftConfig

        cfg = DriftConfig(
            kind="field_deprecation",
            payload={
                "orig": ("posts", "author_name"),
                "lookup": ("users", "id", "full_name"),
            },
        )
        text = author_changelog(cfg)
        assert "posts" in text and "users.id" in text and "full_name" in text

    def test_deterministic(self) -> None:
        from actors.engineering_manager import author_changelog
        from scenarios.base import DriftConfig

        cfg = DriftConfig(
            kind="column_rename",
            payload={"table": "t", "old": "a", "new": "b"},
        )
        assert author_changelog(cfg) == author_changelog(cfg)
