"""Query profiling utilities.

A watchdog-wrapped DuckDB execute plus a median-of-3 warm timer.

* :func:`execute_once_timed` runs a statement exactly once, enforcing a
  hard ``timeout_s`` wall-clock budget. It is the single entry point used
  by the env for agent-provided SQL so the documented query timeout
  cannot be bypassed. An optional ``max_rows`` caps result-set
  materialization — the fetch is aborted as soon as more than
  ``max_rows`` rows are observed, so a pathological ``SELECT *`` cannot
  drive the server OOM before the caller's size check runs.
* :func:`execute_hash_timed` executes a statement once and hashes its full
  result incrementally via ``fetchmany`` so correctness checks do not have
  to materialize the full row set in Python memory.
* :func:`median_of_3_warm_ms` performs one untimed warm-up then three
  timed runs and returns the median milliseconds. Used by scenario
  materialization to publish a stable baseline runtime.

Both helpers raise :class:`TimeoutError` when a single run exceeds the
budget; ``duckdb.Error`` propagates unchanged to the caller.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from engine.verifier import canonical_row_hash
from utilities.logger import get_module_logger

if TYPE_CHECKING:
    import duckdb


DEFAULT_TIMEOUT_S: float = 2.0
INTERRUPT_GRACE_S: float = 0.25
# Maximum number of watchdog escalations (leaked threads) tolerated before
# logging at CRITICAL.  Override via SQL_DRIFT_MAX_LEAKED_WATCHDOG_THREADS.
MAX_LEAKED_WATCHDOG_THREADS: int = int(os.environ.get("SQL_DRIFT_MAX_LEAKED_WATCHDOG_THREADS", "3"))

_LOG = get_module_logger(__name__)
_FETCH_CHUNK_ROWS = 1024

# Module-level counter — incremented each time a watchdog thread survives
# interrupt (i.e. a genuine escalation, not a normal timeout).  Thread-safe
# via _watchdog_leak_lock.  Callers can read this via get_watchdog_leak_count().
_watchdog_leak_lock: threading.Lock = threading.Lock()
_watchdog_leaked_count: int = 0


def get_watchdog_leak_count() -> int:
    """Return the cumulative number of watchdog threads that survived interrupt.

    A non-zero value means at least one DuckDB worker thread was not stopped
    cleanly and is still alive in the background.  Production monitoring should
    alert when this exceeds :data:`MAX_LEAKED_WATCHDOG_THREADS`.
    """
    return _watchdog_leaked_count


class QueryWatchdogEscalationError(RuntimeError):
    """DuckDB worker survived interrupt; the connection is no longer safe."""


@dataclass(frozen=True)
class TimedResult:
    """Output of :func:`execute_once_timed`.

    ``columns`` preserves DuckDB's cursor ``description`` order so callers
    can emit a :class:`models.RunQueryResult` without re-executing the
    query just to recover column names.

    ``truncated`` is ``True`` when the caller supplied a ``max_rows`` cap
    and the query produced strictly more rows than that cap; in that
    case ``rows`` contains exactly ``max_rows + 1`` entries (the
    one-over read that proves overflow). Callers that care about size
    limits must branch on ``truncated`` rather than re-checking
    ``len(rows)`` against their cap.
    """

    columns: list[str]
    rows: list[tuple[Any, ...]]
    elapsed_ms: float
    truncated: bool = False


def _fetch_capped(
    cursor: duckdb.DuckDBPyConnection,
    max_rows: int,
) -> tuple[list[tuple[Any, ...]], bool]:
    """Drain at most ``max_rows + 1`` rows from ``cursor`` via fetchmany.

    Returns ``(rows, truncated)``. When ``truncated`` is ``True`` the
    cursor still has unread rows — we stopped on the first row past the
    cap so the caller can signal overflow without materialising the
    rest of a potentially enormous result set.
    """
    # chunk=1024 trades a few extra Python calls for not over-fetching
    # by orders of magnitude when results are modest. The +1 in the
    # final budget is what makes overflow detectable.
    rows: list[tuple[Any, ...]] = []
    budget = max_rows + 1
    while budget > 0:
        batch = cursor.fetchmany(min(_FETCH_CHUNK_ROWS, budget))
        if not batch:
            return rows, False
        rows.extend(batch)
        budget -= len(batch)
    return rows, len(rows) > max_rows


def _iter_cursor_rows(
    cursor: duckdb.DuckDBPyConnection,
) -> Iterator[tuple[Any, ...]]:
    while True:
        batch = cursor.fetchmany(_FETCH_CHUNK_ROWS)
        if not batch:
            return
        yield from batch


def _run_worker_with_watchdog[T](
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    timeout_s: float,
    worker: Callable[[], T],
) -> T:
    result_holder: dict[str, object] = {}

    def runner() -> None:
        try:
            result_holder["result"] = worker()
        except BaseException as exc:  # Must forward all failures from the worker thread.
            result_holder["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        # DuckDB's interrupt API is connection-scoped and thread-safe;
        # we ask the query to unwind and then wait *unconditionally*
        # for the worker to exit before surfacing the timeout to the
        # caller. If we returned while the thread were still alive, it
        # would retain access to ``conn`` and its result could race
        # future queries on the same connection — a previously
        # observed source of flaky post-timeout behaviour. In practice
        # DuckDB's interrupt releases the worker within a handful of
        # milliseconds; if the engine ever fails to honour interrupt
        # the process will hang here, which is the correct failure
        # mode for a connection whose state is no longer safe to
        # reuse.
        with contextlib.suppress(Exception):
            conn.interrupt()
        thread.join(INTERRUPT_GRACE_S)
        if thread.is_alive():
            global _watchdog_leaked_count
            with _watchdog_leak_lock:
                _watchdog_leaked_count += 1
                leak_count = _watchdog_leaked_count
            log_fn = _LOG.critical if leak_count > MAX_LEAKED_WATCHDOG_THREADS else _LOG.error
            log_fn(
                "query watchdog failed to stop worker after %.3fs timeout + %.3fs grace"
                " (cumulative leaked threads: %d)",
                timeout_s,
                INTERRUPT_GRACE_S,
                leak_count,
            )
            raise QueryWatchdogEscalationError(
                f"query exceeded {timeout_s}s and worker did not stop after interrupt: {sql[:120]!r}"
            )
        raise TimeoutError(f"query exceeded {timeout_s}s: {sql[:120]!r}")
    if "error" in result_holder:
        error = result_holder["error"]
        assert isinstance(error, BaseException)
        raise error
    return cast(T, result_holder["result"])


def _run_with_watchdog(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    timeout_s: float,
    max_rows: int | None,
) -> TimedResult:
    def worker() -> TimedResult:
        start = time.perf_counter_ns()
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        if max_rows is None:
            rows = cursor.fetchall()
            truncated = False
        else:
            rows, truncated = _fetch_capped(cursor, max_rows)
        elapsed_ns = time.perf_counter_ns() - start
        return TimedResult(
            columns=columns,
            rows=rows,
            elapsed_ms=elapsed_ns / 1_000_000.0,
            truncated=truncated,
        )

    result = _run_worker_with_watchdog(conn, sql, timeout_s, worker)
    assert isinstance(result, TimedResult)
    return result


def execute_once_timed(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int | None = None,
) -> tuple[list[tuple[Any, ...]], float]:
    """Single timed execution — returns ``(rows, elapsed_ms)``.

    Thin wrapper for callers that don't need column metadata or the
    truncation flag.
    """
    res = _run_with_watchdog(conn, sql, timeout_s, max_rows)
    return res.rows, res.elapsed_ms


def execute_once_with_columns(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int | None = None,
) -> TimedResult:
    """Single timed execution — returns columns + rows + elapsed_ms.

    When ``max_rows`` is supplied, the fetch aborts at the first row
    past the cap and ``TimedResult.truncated`` is set. The elapsed
    milliseconds in that case reflect the partial scan, not the query's
    would-be completion time — a truncated read is a *hard error* in
    agent-facing code paths, not a performance measurement.
    """
    return _run_with_watchdog(conn, sql, timeout_s, max_rows)


def execute_hash_timed(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[str, float]:
    """Single timed execution — returns ``(result_hash, elapsed_ms)``.

    Unlike :func:`execute_once_timed`, this drains the cursor via
    ``fetchmany`` and hashes rows incrementally, so callers can compare a
    large final result to ground truth without materializing the full row
    set in Python memory.
    """

    def worker() -> tuple[str, float]:
        start = time.perf_counter_ns()
        cursor = conn.execute(sql)
        result_hash = canonical_row_hash(_iter_cursor_rows(cursor))
        elapsed_ns = time.perf_counter_ns() - start
        return result_hash, elapsed_ns / 1_000_000.0

    result = _run_worker_with_watchdog(conn, sql, timeout_s, worker)
    result_hash, elapsed_ms = result
    assert isinstance(result_hash, str)
    assert isinstance(elapsed_ms, float)
    return result_hash, elapsed_ms


def median_of_3_warm_ms(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> float:
    """Warm cache, then median-of-3 timed runs. Returns milliseconds."""
    _run_with_watchdog(conn, sql, timeout_s, None)
    timings = [_run_with_watchdog(conn, sql, timeout_s, None).elapsed_ms for _ in range(3)]
    timings.sort()
    return timings[1]


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "INTERRUPT_GRACE_S",
    "MAX_LEAKED_WATCHDOG_THREADS",
    "QueryWatchdogEscalationError",
    "TimedResult",
    "execute_hash_timed",
    "execute_once_timed",
    "execute_once_with_columns",
    "get_watchdog_leak_count",
    "median_of_3_warm_ms",
]
