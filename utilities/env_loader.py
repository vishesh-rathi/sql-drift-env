"""Repo-local `.env` loading and typed env readers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = _REPO_ROOT / ".env"

# Load the repo-local .env once so server and training defaults stay in
# sync when the user edits a single file and restarts the relevant
# process.
load_dotenv(DOTENV_PATH)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def env_int(
    name: str, default: int, *, min_value: int | None = None, max_value: int | None = None
) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return value


def env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a float") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return value


__all__ = ["DOTENV_PATH", "env_float", "env_int", "env_str"]
