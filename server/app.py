"""FastAPI factory for the SQLDrift environment.

``create_app()`` returns a fully-wired FastAPI app exposing the
stateless HTTP routes (``/health``, ``/schema``, ``/reset``, ``/step``)
and the stateful ``/ws`` WebSocket session. Stateful multi-step
episodes must go through ``/ws``; each HTTP ``/step`` spawns a
fresh env instance that is ``close()``-d in ``finally`` (one env per request).

``main()`` runs the server with Uvicorn — exported as the
``[project.scripts] sql-drift-server`` entry point.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.http_server import create_app as _openenv_create_app

from models import SqlDriftAction, SqlDriftObservation
from skill_library import DEFAULT_STORE_DIR, Store, cleanup_stale_session_dirs

from . import settings
from .sql_drift_env_environment import SqlDriftEnvironment

ENV_NAME = "sql_drift_env"
DEFAULT_MAX_CONCURRENT_ENVS = settings.MAX_CONCURRENT_ENVS
_SESSION_STORE_ROOT = DEFAULT_STORE_DIR / "sessions"

# Purge stale session directories left by previous server runs before
# accepting any traffic.  Failures are non-fatal.
_startup_removed = cleanup_stale_session_dirs(
    _SESSION_STORE_ROOT, settings.SKILL_STORE_SESSION_TTL_HOURS
)
if _startup_removed:
    import logging as _logging

    _logging.getLogger("sql_drift_env.app.server.app").info(
        "startup: removed %d stale session skill-store dirs from %s",
        _startup_removed,
        _SESSION_STORE_ROOT,
    )


def _create_server_environment() -> SqlDriftEnvironment:
    """Build one server-managed env with its own on-disk skill library.

    ``cleanup_on_close=True`` ensures the session directory is deleted when
    the WebSocket session ends, preventing unbounded on-disk session growth.
    """
    session_dir = _SESSION_STORE_ROOT / uuid4().hex
    return SqlDriftEnvironment(
        skill_store=Store(directory=session_dir),
        cleanup_on_close=True,
    )


def create_app(max_concurrent_envs: int | None = None) -> Any:
    """Build the FastAPI app bound to a fresh-env factory per session."""
    if max_concurrent_envs is None:
        max_concurrent_envs = DEFAULT_MAX_CONCURRENT_ENVS
    return _openenv_create_app(
        env=_create_server_environment,
        action_cls=SqlDriftAction,
        observation_cls=SqlDriftObservation,
        env_name=ENV_NAME,
        max_concurrent_envs=max_concurrent_envs,
    )


def main(host: str = settings.SERVER_HOST, port: int = settings.SERVER_PORT) -> None:
    """Uvicorn entry point — matches the [project.scripts] wiring."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)


# Module-level app instance for uvicorn's ``module:attr`` syntax
# (``uvicorn server.app:app``) and the ``openenv.yaml`` ``app:`` field.
# Built at import time; safe because the OpenEnv factory only stores the
# environment factory and instantiates per request / session.
app = create_app()


__all__ = ["ENV_NAME", "app", "create_app", "main"]


if __name__ == "__main__":
    main()
