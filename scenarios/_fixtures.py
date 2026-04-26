"""Deterministic fixture generators (stdlib only).

All distributions are backed by :class:`random.Random(seed)` so a given
``(scenario_id, seed, scale)`` tuple always yields the same table contents.

Exposes a single :func:`seeded_rng` factory plus a handful of domain-specific
generators used by the concrete scenarios. No numpy/pandas runtime deps.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from datetime import UTC


def seeded_rng(*parts: int | str) -> random.Random:
    """Derive a deterministic `random.Random` from mixed scalar parts.

    Uses a stable 64-bit SplitMix-style hash over ``repr(parts)`` — avoids
    Python's per-interpreter salted ``hash()`` for str.
    """
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    for p in parts:
        for byte in repr(p).encode():
            h ^= byte
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return random.Random(h)


def lognormal_amounts(rng: random.Random, n: int, mu: float, sigma: float) -> list[float]:
    """n log-normally distributed positive amounts, rounded to 2dp."""
    return [round(rng.lognormvariate(mu, sigma), 2) for _ in range(n)]


def zipfian_choices(
    rng: random.Random, population: Sequence[int], n: int, *, alpha: float = 1.07
) -> list[int]:
    """n draws from `population` with zipf(alpha) weights.

    `alpha` controls skew: 1.07 is the project default. Uses
    `random.choices` with explicit weights, so the draws are stable across
    Python versions for a given `rng` state.
    """
    weights = [1.0 / ((i + 1) ** alpha) for i in range(len(population))]
    return rng.choices(list(population), weights=weights, k=n)


def date_range_epoch_ms(
    rng: random.Random,
    n: int,
    *,
    start_epoch_ms: int,
    window_days: int,
) -> list[int]:
    """n random timestamps (ms) within `[start, start + window_days)`."""
    span_ms = window_days * 86_400_000
    return [start_epoch_ms + rng.randrange(span_ms) for _ in range(n)]


def iso_strings_from_epoch_ms(epoch_ms: list[int]) -> list[str]:
    """Convert epoch ms to ISO-8601 UTC strings (matching DuckDB's native coerce)."""
    from datetime import datetime

    return [
        datetime.fromtimestamp(t / 1000, tz=UTC).isoformat().replace("+00:00", "Z")
        for t in epoch_ms
    ]


def categorical_choices(
    rng: random.Random,
    categories: Sequence[str],
    n: int,
    *,
    weights: Sequence[float] | None = None,
) -> list[str]:
    return rng.choices(list(categories), weights=list(weights) if weights else None, k=n)


def unique_names(rng: random.Random, n: int, *, prefix: str = "name") -> list[str]:
    """Stable pseudo-unique string IDs of the form `<prefix>_<64-bit-hex>`."""
    return [f"{prefix}_{rng.getrandbits(64):016x}" for _ in range(n)]


def approx_normal(
    rng: random.Random, n: int, *, mu: float, sigma: float, clip_lo: float | None = None
) -> list[float]:
    """n normal draws, optionally clipped below."""
    out: list[float] = []
    for _ in range(n):
        x = rng.gauss(mu, sigma)
        if clip_lo is not None and x < clip_lo:
            x = clip_lo
        out.append(round(x, 4))
    return out


def sanity_nonzero_variance(xs: Sequence[float]) -> bool:
    """Guard: reject obviously degenerate distributions (used in smoke tests)."""
    if not xs:
        return False
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return var > 1e-12 and not math.isnan(var)


__all__ = [
    "approx_normal",
    "categorical_choices",
    "date_range_epoch_ms",
    "iso_strings_from_epoch_ms",
    "lognormal_amounts",
    "sanity_nonzero_variance",
    "seeded_rng",
    "unique_names",
    "zipfian_choices",
]
