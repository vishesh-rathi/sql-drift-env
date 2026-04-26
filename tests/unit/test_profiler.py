"""P1 — tests for the query profiler."""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from engine.profiler import (
    INTERRUPT_GRACE_S,
    QueryWatchdogEscalationError,
    _run_with_watchdog,
    execute_hash_timed,
    execute_once_timed,
    execute_once_with_columns,
    median_of_3_warm_ms,
)
from engine.verifier import canonical_row_hash


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE t(x INTEGER); INSERT INTO t SELECT UNNEST(RANGE(1, 1001));")
    yield c
    c.close()


class TestProfiler:
    def test_median_positive_and_finite(self, conn: duckdb.DuckDBPyConnection) -> None:
        ms = median_of_3_warm_ms(conn, "SELECT COUNT(*) FROM t WHERE x > 500")
        assert ms > 0
        assert ms < 2_000  # within default timeout

    def test_execute_once_returns_rows(self, conn: duckdb.DuckDBPyConnection) -> None:
        rows, ms = execute_once_timed(conn, "SELECT COUNT(*) FROM t")
        assert rows == [(1000,)]
        assert ms >= 0

    def test_invalid_sql_propagates(self, conn: duckdb.DuckDBPyConnection) -> None:
        with pytest.raises(duckdb.Error):
            execute_once_timed(conn, "SELECT * FROM no_such_table")

    def test_timeout_raises(self, conn: duckdb.DuckDBPyConnection) -> None:
        # DuckDB supports recursive CTEs; cap one to a tight timeout so the
        # watchdog fires deterministically without actually waiting seconds.
        sql = (
            "WITH RECURSIVE spin(n) AS ("
            "  SELECT 1 UNION ALL SELECT n + 1 FROM spin WHERE n < 10000000"
            ") SELECT COUNT(*) FROM spin"
        )
        with pytest.raises(TimeoutError):
            execute_once_timed(conn, sql, timeout_s=0.05)

    def test_execute_hash_matches_canonical_hash(self, conn: duckdb.DuckDBPyConnection) -> None:
        expected = canonical_row_hash(conn.execute("SELECT x FROM t WHERE x <= 25").fetchall())
        result_hash, ms = execute_hash_timed(conn, "SELECT x FROM t WHERE x <= 25")
        assert result_hash == expected
        assert ms >= 0

    def test_execute_hash_streams_rows_without_fetchall(self) -> None:
        class _FakeCursor:
            def __init__(self) -> None:
                self.description = [("x",)]
                self._batches = [[(1,), (2,)], [(3,)], []]

            def fetchmany(self, limit: int):
                return self._batches.pop(0)

            def fetchall(self):
                raise AssertionError("execute_hash_timed should not call fetchall()")

        class _FakeConn:
            def execute(self, sql: str) -> _FakeCursor:
                return _FakeCursor()

        result_hash, _ = execute_hash_timed(_FakeConn(), "SELECT x FROM t")
        assert result_hash == canonical_row_hash([(1,), (2,), (3,)])

    def test_watchdog_escalates_when_interrupt_does_not_stop_worker(self, monkeypatch) -> None:
        class _FakeConn:
            def __init__(self) -> None:
                self.interrupted = False

            def interrupt(self) -> None:
                self.interrupted = True

        class _StuckThread:
            def __init__(self, target, daemon: bool) -> None:
                self.join_calls: list[float | None] = []

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                self.join_calls.append(timeout)

            def is_alive(self) -> bool:
                return True

        stuck = _StuckThread(None, daemon=True)
        monkeypatch.setattr(
            "engine.profiler.threading.Thread",
            lambda target, daemon: stuck,
        )
        conn = _FakeConn()

        with pytest.raises(QueryWatchdogEscalationError, match="did not stop after interrupt"):
            _run_with_watchdog(conn, "SELECT 1", timeout_s=0.01, max_rows=None)

        assert conn.interrupted is True
        assert stuck.join_calls == [0.01, INTERRUPT_GRACE_S]


class TestRowCap:
    def test_no_cap_fetches_all(self, conn: duckdb.DuckDBPyConnection) -> None:
        res = execute_once_with_columns(conn, "SELECT x FROM t")
        assert len(res.rows) == 1000
        assert res.truncated is False

    def test_cap_above_result_not_truncated(self, conn: duckdb.DuckDBPyConnection) -> None:
        res = execute_once_with_columns(conn, "SELECT x FROM t", max_rows=5000)
        assert len(res.rows) == 1000
        assert res.truncated is False

    def test_cap_equal_to_result_not_truncated(self, conn: duckdb.DuckDBPyConnection) -> None:
        res = execute_once_with_columns(conn, "SELECT x FROM t", max_rows=1000)
        assert len(res.rows) == 1000
        assert res.truncated is False

    def test_cap_below_result_is_truncated(self, conn: duckdb.DuckDBPyConnection) -> None:
        res = execute_once_with_columns(conn, "SELECT x FROM t", max_rows=10)
        # On overflow we deliberately read exactly cap+1 rows — enough
        # to prove overflow without materialising the full 1,000-row set.
        assert len(res.rows) == 11
        assert res.truncated is True

    def test_cap_does_not_over_materialise(self) -> None:
        """A huge result is *not* fully scanned when a cap is in play."""
        import duckdb as _d

        c = _d.connect(":memory:")
        try:
            # 10M-row source; fetchmany must stop near the cap or this test
            # would take seconds instead of milliseconds.
            c.execute("CREATE TABLE big AS SELECT * FROM range(10000000) big(x)")
            res = execute_once_with_columns(c, "SELECT x FROM big", max_rows=500, timeout_s=5.0)
            assert res.truncated is True
            assert len(res.rows) == 501
        finally:
            c.close()
