"""Sanity checks for ``pyproject.toml``."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_dev_extra_has_core_tooling() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    dev = data["project"]["optional-dependencies"]["dev"]
    # Keep this lightweight: if someone trims dev, tests should fail loudly.
    for needle in ("ruff", "pytest", "mypy", "httpx"):
        assert any(needle in spec for spec in dev), f"dev extra missing {needle}"


def test_train_extra_does_not_pull_plain_unsloth() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    train = data["project"]["optional-dependencies"]["train"]

    assert any(spec == "torch==2.7.0" for spec in train)
    assert not any(spec.startswith("unsloth") for spec in train)
