"""P1 — tests for the private RuntimeEpisodeState dataclass."""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from engine.runtime import RuntimeEpisodeState
from models import EpisodePhase


def _make(**overrides) -> RuntimeEpisodeState:
    defaults: dict = dict(
        episode_id="ep-1",
        seed=42,
        scenario_id="01_correlated_subquery",
        instance=object(),  # opaque — scenarios.base lands in P2
        conn=duckdb.connect(":memory:"),
        gt_result_hash_predrift="pre",
        gt_result_hash_postdrift=None,
        baseline_runtime_ms=60.0,
        baseline_tokens=120,
        baseline_sql_canonical="SELECT 1",
        baseline_postdrift_raises=False,
        drift_scheduled_step=None,
        budget_steps=25,
    )
    defaults.update(overrides)
    return RuntimeEpisodeState(**defaults)


class TestRuntimeEpisodeState:
    def test_defaults(self) -> None:
        rt = _make()
        assert rt.step_count == 0
        assert rt.phase == EpisodePhase.DIAGNOSE
        assert rt.drift_fired is False
        assert rt.budget_steps_remaining == 25
        assert rt.failed_query_hashes == set()

    def test_drift_fired_follows_step(self) -> None:
        rt = _make(drift_scheduled_step=6)
        assert rt.drift_fired is False
        rt.drift_fired_step = 8
        assert rt.drift_fired is True

    def test_budget_remaining_clamped_at_zero(self) -> None:
        rt = _make()
        rt.step_count = 100
        assert rt.budget_steps_remaining == 0
