"""Scenario spec + instance primitives.

Each concrete scenario file in :mod:`scenarios` exports:

- ``SPEC: ScenarioSpec`` — the immutable metadata (id, family, tags,
  optional drift config) plus a bound ``builder`` callable.

The builder takes ``(spec, seed, scale)`` and returns a ready-to-attach
:class:`ScenarioInstance` whose DuckDB connection has been loaded with
deterministic fixtures, ground-truth hashes pre-computed, and baseline
runtime measured. ``base_scale`` is author-tuned per scenario so the
measured baseline clears :data:`BASELINE_MIN_MS` on a single build —
the old timing-driven reroll loop was removed because it coupled the
fixture RNG seed to the retry count, which destroyed determinism
whenever CI hit a jitter-induced retry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import duckdb

from utilities.logger import get_module_logger

_LOG = get_module_logger(__name__)

Family = Literal["ecommerce", "events", "cms", "saas_logs", "multitenant"]
DriftKind = Literal["column_rename", "date_format", "enum_rule", "field_deprecation"]
DifficultyLevel = Literal["easy", "normal", "hard"]

# Baseline floor — empirically calibrated, not a fixed marketing target
# 50 ms. DuckDB on in-memory fixtures at CI-reasonable scales (a few
# tens of thousands of rows) measures warm baselines of 0.4–2 ms on
# the anti-pattern queries shipped here; reaching 50 ms would require
# multi-minute fixture builds per scenario, which is untenable for
# both CI and RL rollouts (every reset rebuilds).
#
# 0.3 ms is ~3–5× the median-of-3 warm jitter floor on a quiet CPU
# (observed jitter ~60–100 µs). This SNR is tight but workable because
# the rubric gates the speedup reward at 1.2× before any
# credit is issued, so jitter-induced near-1× "speedups" score zero.
# The cap at 64× bounds upside. A 2× rewrite against a 0.3 ms baseline
# lands at 0.15 ms — still distinguishable from jitter under
# median-of-3 smoothing.
#
# The same floor applies in production and CI — no env-var escape
# hatch — so tests exercise the real reward distribution. Per-scenario
# overrides may raise *or* lower this floor when a scenario's query
# shape has a different natural baseline (see the field docstring on
# :class:`ScenarioSpec.baseline_min_ms`).
BASELINE_MIN_MS = 0.3


@dataclass(frozen=True)
class DriftConfig:
    kind: DriftKind
    payload: dict[str, Any]
    min_step: int = 6
    max_step: int = 12
    cooldown_steps: int = 2

    def __post_init__(self) -> None:
        if self.min_step < 1:
            raise ValueError("min_step must be >= 1")
        if self.max_step < self.min_step:
            raise ValueError("max_step must be >= min_step")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")


@dataclass
class ScenarioInstance:
    """Concretized scenario — ready-to-attach DuckDB fixture + ground truths."""

    conn: duckdb.DuckDBPyConnection
    baseline_sql: str
    gt_sql_predrift: str
    gt_sql_postdrift: str | None
    baseline_runtime_ms: float
    baseline_tokens: int
    gt_result_hash_predrift: str
    gt_result_hash_postdrift: str | None
    drift_config: DriftConfig | None
    schema_synopsis: str
    # Drift-distinctive identifier sets consumed by the drift-adapt
    # rubric. ``postdrift_identifiers`` marks identifiers/literals
    # the correct post-drift rewrite MUST introduce; ``predrift_identifiers``
    # marks identifiers/literals a submission that ignored the drift
    # WOULD retain. Together they let the rubric distinguish "adapted"
    # from "did not adapt" for drift kinds where a single identifier
    # (e.g. ``ts`` under date-format drift) is shared by both sides.
    postdrift_identifiers: frozenset[str] = field(default_factory=frozenset)
    predrift_identifiers: frozenset[str] = field(default_factory=frozenset)


# Builder signature: (spec, seed, scale) -> (conn, baseline_sql,
#   gt_sql_predrift, gt_sql_postdrift, schema_synopsis,
#   postdrift_identifiers, predrift_identifiers).
BuilderResult = tuple[
    "duckdb.DuckDBPyConnection",
    str,  # baseline_sql
    str,  # gt_sql_predrift
    str | None,  # gt_sql_postdrift
    str,  # schema_synopsis
    frozenset[str],  # postdrift_identifiers
    frozenset[str],  # predrift_identifiers
]
BuilderFn = Callable[["ScenarioSpec", int, int], BuilderResult]


@dataclass(frozen=True)
class ScenarioSpec:
    """Immutable scenario metadata + bound builder."""

    scenario_id: str
    family: Family
    tags: frozenset[str]
    drift_config: DriftConfig | None
    builder: BuilderFn
    # Row-count scale passed to the builder. Author-tuned so the
    # measured baseline clears ``baseline_min_ms`` on a single build;
    # materialize() emits a warning (but does not retry) if the floor
    # is not met, signalling the author to bump this value.
    base_scale: int = 1_000
    # Per-scenario baseline floor override. Most scenarios inherit the
    # module default. Scenarios whose query shape naturally lands at a
    # very different baseline (e.g. a trivial single-table GROUP BY
    # that can't be meaningfully sped up, or a large join whose raw
    # shape is already expensive) can pin a different floor with a
    # documented rationale at the SPEC site.
    baseline_min_ms: float = BASELINE_MIN_MS

    def materialize(self, seed: int, *, difficulty: DifficultyLevel = "normal") -> ScenarioInstance:
        return materialize(self, seed, difficulty=difficulty)


def count_tokens(sql: str) -> int:
    """Rough whitespace/punctuation token count — good enough for baseline."""
    import re

    return len(re.findall(r"[\w]+|[^\s\w]", sql))


def _scale_for_difficulty(base_scale: int, difficulty: DifficultyLevel) -> int:
    """Map a coarse difficulty level onto the scenario builder's row-count scale."""
    if difficulty == "easy":
        return max(1, base_scale // 2)
    if difficulty == "hard":
        return base_scale * 2
    return base_scale


def materialize(
    spec: ScenarioSpec, seed: int, *, difficulty: DifficultyLevel = "normal"
) -> ScenarioInstance:
    """Build a ScenarioInstance once, measure baseline, and return.

    Single build — deterministic, no retry. If the measured baseline is
    below ``spec.baseline_min_ms`` a warning is logged so scenario
    authors can bump ``base_scale``; the instance is still returned so
    episodes can proceed (the rubric gracefully handles small
    baselines via the 1.2× speedup gate and infinite-speedup cap).
    """
    from engine.profiler import median_of_3_warm_ms
    from engine.verifier import canonical_row_hash

    scale = _scale_for_difficulty(spec.base_scale, difficulty)

    (
        conn,
        baseline_sql,
        gt_pre,
        gt_post,
        synopsis,
        postdrift_ids,
        predrift_ids,
    ) = spec.builder(spec, seed, scale)
    try:
        baseline_ms = median_of_3_warm_ms(conn, baseline_sql)
    except Exception:
        conn.close()
        raise
    if baseline_ms < spec.baseline_min_ms:
        _LOG.warning(
            "%s: baseline %.2fms < %.2fms floor at difficulty=%s scale=%d — bump base_scale",
            spec.scenario_id,
            baseline_ms,
            spec.baseline_min_ms,
            difficulty,
            scale,
        )

    pre_rows = conn.execute(gt_pre).fetchall()
    gt_hash_pre = canonical_row_hash(pre_rows)
    # Post-drift ground-truth hashes are computed AFTER drift is applied
    # at runtime — not here. The env backfills them from gt_post once
    # drift fires.
    return ScenarioInstance(
        conn=conn,
        baseline_sql=baseline_sql,
        gt_sql_predrift=gt_pre,
        gt_sql_postdrift=gt_post,
        baseline_runtime_ms=baseline_ms,
        baseline_tokens=count_tokens(baseline_sql),
        gt_result_hash_predrift=gt_hash_pre,
        gt_result_hash_postdrift=None,
        drift_config=spec.drift_config,
        schema_synopsis=synopsis,
        postdrift_identifiers=postdrift_ids,
        predrift_identifiers=predrift_ids,
    )


__all__ = [
    "BASELINE_MIN_MS",
    "BuilderFn",
    "BuilderResult",
    "DifficultyLevel",
    "DriftConfig",
    "DriftKind",
    "Family",
    "ScenarioInstance",
    "ScenarioSpec",
    "count_tokens",
    "materialize",
]
