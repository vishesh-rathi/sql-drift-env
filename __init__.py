"""SQLDrift — OpenEnv gym for SQL repair & optimization under drift.

The repo uses a flat top-level module layout (``models.py``,
``client.py``, ``engine/``, ``scenarios/``, ``skill_library/``,
``actors/``, ``server/``, ``training/``) because it is also run as a
FastAPI server that imports siblings absolutely (``from models import
…``). setuptools republishes ``.`` as the ``sql_drift_env`` package so
both import styles work at runtime, but eagerly re-exporting the flat
submodules from here would shadow the top-level ``import models`` /
``import client`` paths that every flat module relies on, and would
make the import order pytest-collection-sensitive.

The public API for agent code is therefore the flat modules themselves,
imported directly:

    from client import SqlDriftEnv
    from models import SqlDriftAction, SqlDriftObservation
    from server import SqlDriftEnvironment

This mirrors the flat layout both on disk and at import time; the
``sql_drift_env`` namespace exists only so the wheel has a canonical
name and so third parties can depend on a stable version string.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
