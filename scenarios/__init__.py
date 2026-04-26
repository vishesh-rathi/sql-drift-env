"""Scenario registry + auto-discovery.

Every sibling module whose name starts with ``scenario_`` is expected to
export a module-level ``SPEC: ScenarioSpec``. This module walks the package
at import time and populates :data:`REGISTRY` so callers (env reset, tests,
CLI eval) can look scenarios up by id without knowing the file layout.
"""

from __future__ import annotations

import importlib
import pkgutil

from .base import (
    DriftConfig,
    DriftKind,
    Family,
    ScenarioInstance,
    ScenarioSpec,
)

REGISTRY: dict[str, ScenarioSpec] = {}


def _discover_specs() -> None:
    """Import every `scenario_*` sibling module and harvest their SPEC."""
    package = __name__  # "scenarios"
    package_path = __path__  # provided by Python's package machinery

    for info in pkgutil.iter_modules(package_path):
        if not info.name.startswith("scenario_"):
            continue
        module = importlib.import_module(f"{package}.{info.name}")
        spec = getattr(module, "SPEC", None)
        if spec is None:
            raise RuntimeError(f"{package}.{info.name} is missing a module-level `SPEC` export")
        if not isinstance(spec, ScenarioSpec):
            raise TypeError(
                f"{package}.{info.name}.SPEC is {type(spec).__name__}; expected ScenarioSpec"
            )
        if spec.scenario_id in REGISTRY:
            raise RuntimeError(
                f"duplicate scenario_id {spec.scenario_id!r} — "
                f"already registered from {REGISTRY[spec.scenario_id]!r}"
            )
        REGISTRY[spec.scenario_id] = spec


def iter_specs() -> list[ScenarioSpec]:
    return sorted(REGISTRY.values(), key=lambda s: s.scenario_id)


def get_spec(scenario_id: str) -> ScenarioSpec:
    try:
        return REGISTRY[scenario_id]
    except KeyError as e:
        raise KeyError(f"unknown scenario_id={scenario_id!r}; known: {sorted(REGISTRY)}") from e


_discover_specs()


__all__ = [
    "DriftConfig",
    "DriftKind",
    "Family",
    "REGISTRY",
    "ScenarioInstance",
    "ScenarioSpec",
    "get_spec",
    "iter_specs",
]
