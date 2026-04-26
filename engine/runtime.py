"""Private per-episode runtime state (not exposed over the wire).

This module is imported by both :mod:`engine.reward` and
:mod:`server.sql_drift_env_environment` — keeping it out of ``server/``
avoids the import cycle ``engine.reward → server → engine.reward``.

NEVER serialize or expose this over any endpoint. The public state
projection lives in :class:`models.SqlDriftState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from models import EpisodePhase

if TYPE_CHECKING:
    import duckdb


@dataclass
class RuntimeEpisodeState:
    """Private per-episode state — holds secrets (DB handle, ground truth)."""

    episode_id: str
    seed: int
    scenario_id: str

    instance: Any  # scenarios.base.ScenarioInstance — forward ref avoids import cycle
    conn: duckdb.DuckDBPyConnection

    # Ground truth hashes (never exposed)
    gt_result_hash_predrift: str
    gt_result_hash_postdrift: str | None

    # Baseline measurements (used by speedup + baseline-verbatim gates)
    baseline_runtime_ms: float
    baseline_tokens: int
    baseline_sql_canonical: str
    baseline_postdrift_raises: bool

    # Drift timing (scheduled step + cooldown relative to first run_query).
    drift_scheduled_step: int | None
    connection_poisoned: bool = False
    drift_fired_step: int | None = None
    first_run_query_step: int | None = None

    # Episode progression
    step_count: int = 0
    phase: EpisodePhase = EpisodePhase.DIAGNOSE
    budget_steps: int = 25

    # Per-step bookkeeping
    failed_query_hashes: set[str] = field(default_factory=set)
    failed_query_counts: dict[str, int] = field(default_factory=dict)
    changelog_entries: list[str] = field(default_factory=list)
    consultations_used: int = 0
    listed_tables_rewarded: bool = False
    described_tables_rewarded: set[str] = field(default_factory=set)
    sampled_tables_rewarded: set[str] = field(default_factory=set)
    run_query_rewarded: bool = False
    explain_query_rewarded: bool = False
    changelog_rewarded_after_drift: bool = False

    # Phase-machine bookkeeping — counts successful diagnostic tool calls
    # (list_tables, describe_table, sample_rows, run_query, explain_query,
    # read_changelog). The DIAGNOSE → REWRITE transition fires the first
    # time this becomes non-zero; SUBMIT_REWRITE is rejected while this is
    # still zero (ToolErrorCode.SUBMIT_BEFORE_DIAGNOSE).
    diagnostic_actions_taken: int = 0

    # Submission state — populated once SUBMIT_REWRITE is accepted
    submitted: bool = False
    submitted_sql: str | None = None
    submitted_sql_canonical: str | None = None
    submitted_result_hash: str | None = None
    submitted_runtime_ms: float | None = None

    # Last-step signal — consumed by the rubric to compute per-step penalties
    last_step_was_tool_error: bool = False
    last_step_was_repeat_failing_query: bool = False
    last_step_repeat_failing_query_count: int = 0
    last_step_productive_rebate: float = 0.0
    consecutive_tool_errors: int = 0

    # Drift acknowledgement — set True the first time the agent reads the
    # changelog or observes post-drift schema identifiers in a query.
    drift_acknowledged: bool = False

    # DBA Oracle feature flag (read by the ConsultDBA child rubric).
    # Always False unless explicitly enabled at reset (kwarg or env var).
    dba_oracle_enabled: bool = False

    @property
    def drift_fired(self) -> bool:
        return self.drift_fired_step is not None

    @property
    def budget_steps_remaining(self) -> int:
        return max(0, self.budget_steps - self.step_count)
