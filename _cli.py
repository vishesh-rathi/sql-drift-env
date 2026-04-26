"""Console-script entrypoint that patches sys.path for the flat-import layout.

Problem
-------
The project uses a *flat layout* (``pyproject.toml`` maps ``.`` → the
``sql_drift_env`` package directory).  That means every sibling module
(``models``, ``actors``, ``engine`` …) is imported as a plain top-level
name rather than via the ``sql_drift_env.`` prefix.

When the wheel is installed, those siblings land at
``site-packages/sql_drift_env/models.py`` etc., *not* at the top-level
``site-packages/`` directory.  A naïve console-script that calls
``sql_drift_env.server.app:main`` would fail at ``from models import …``
before reaching any application logic.

Fix
---
Insert the installed package directory (``site-packages/sql_drift_env/``)
onto ``sys.path`` *before* importing anything from the server package.
This mirrors what Docker achieves via ``--app-dir /app/env`` / ``PYTHONPATH``,
but works for any installed-wheel invocation without requiring a wrapper
script or Docker.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    # __file__ resolves to site-packages/sql_drift_env/_cli.py after
    # installation, so its parent IS the directory that contains models.py,
    # server/, actors/, etc.
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)

    # Import lazily so the sys.path fix takes effect before any flat import
    # in server/app.py or its transitive dependencies is attempted.
    from server.app import main as _server_main

    _server_main()


if __name__ == "__main__":
    main()
