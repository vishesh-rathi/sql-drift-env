"""Server/runtime settings loaded from the repo-local `.env`."""

from __future__ import annotations

from utilities.env_loader import env_float, env_int, env_str

SERVER_HOST = env_str("SQL_DRIFT_SERVER_HOST", "0.0.0.0")
SERVER_PORT = env_int("SQL_DRIFT_SERVER_PORT", 8000, min_value=1, max_value=65535)
MAX_CONCURRENT_ENVS = env_int("SQL_DRIFT_MAX_CONCURRENT_ENVS", 4, min_value=1)

DEFAULT_STEP_BUDGET = env_int("SQL_DRIFT_DEFAULT_STEP_BUDGET", 25, min_value=1)
MAX_RESULT_ROWS = env_int("SQL_DRIFT_MAX_RESULT_ROWS", 1_000, min_value=1)
QUERY_TIMEOUT_S = env_float("SQL_DRIFT_QUERY_TIMEOUT_S", 2.0, min_value=0.001)

# Session skill-store directories older than this many hours are removed at
# startup and when the owning environment is closed.  Set to 0 to disable
# TTL-based cleanup (directories will still be removed on close when
# cleanup_on_close=True is set for a server-managed environment).
SKILL_STORE_SESSION_TTL_HOURS = env_float(
    "SQL_DRIFT_SKILL_STORE_SESSION_TTL_HOURS", 24.0, min_value=0.0
)

__all__ = [
    "DEFAULT_STEP_BUDGET",
    "MAX_CONCURRENT_ENVS",
    "MAX_RESULT_ROWS",
    "QUERY_TIMEOUT_S",
    "SERVER_HOST",
    "SERVER_PORT",
    "SKILL_STORE_SESSION_TTL_HOURS",
]
