"""Shared pytest fixtures for SQLDrift.

Concrete fixtures (seeded env, tmpdir skill-library, fake DuckDB) land as
downstream phases produce the modules they depend on. For P0 we only need
a single sys.path shim so `import engine`, `import scenarios`, etc. resolve
to the in-tree packages without relying on an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
