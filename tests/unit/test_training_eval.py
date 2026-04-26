"""P13 — :mod:`training.eval` unit tests.

These are pure helper tests — the full end-to-end sweep happens in
:mod:`tests.integration.test_training_eval_smoke` which is slow-marked.
"""

from __future__ import annotations

import math

import pytest

import training.eval as eval_mod
from models import (
    REWARD_COMPONENT_KEYS,
    EpisodePhase,
    ListTablesPayload,
    SqlDriftAction,
    SqlDriftObservation,
    SqlDriftState,
    ToolName,
)
from training.eval import (
    EpisodeResult,
    _build_summary,
    _expand_scenarios,
    _format_speedup,
    _run_one_episode,
    render_report,
    run_eval,
)


def _mkres(
    sid: str,
    seed: int,
    terminal_reward: float,
    passed: bool,
    *,
    episode_return: float | None = None,
    steps: int = 10,
    submitted: bool = True,
    drift_fired: bool = False,
    effective_speedup: float | None = None,
) -> EpisodeResult:
    # terminal_reward is the single-step reward that flipped done=True;
    # episode_return is the sum across the whole rollout. They're only
    # equal in the degenerate one-step case, so we default to a
    # different value (terminal − 0.03·steps, approximating the
    # accumulated step tax) to catch call sites that conflate them.
    if episode_return is None:
        episode_return = terminal_reward - 0.03 * max(steps - 1, 0)
    return EpisodeResult(
        scenario_id=sid,
        seed=seed,
        terminal_reward=terminal_reward,
        episode_return=episode_return,
        steps=steps,
        passed=passed,
        submitted=submitted,
        drift_fired=drift_fired,
        wall_ms=100.0,
        reward_components={k: 0.0 for k in REWARD_COMPONENT_KEYS},
        effective_speedup=effective_speedup,
    )


class TestExpandScenarios:
    def test_range(self) -> None:
        out = _expand_scenarios("1-3")
        assert out == [
            "01_correlated_subquery",
            "02_select_star_join",
            "03_cartesian_join",
        ]

    def test_comma_digits(self) -> None:
        out = _expand_scenarios("1,7")
        assert out == ["01_correlated_subquery", "07_drift_column_rename"]

    def test_raw_ids(self) -> None:
        raw = "07_drift_column_rename, 09_drift_enum_rule"
        assert _expand_scenarios(raw) == [
            "07_drift_column_rename",
            "09_drift_enum_rule",
        ]

    def test_full_range_covers_all_ten(self) -> None:
        assert len(_expand_scenarios("1-10")) == 10


class TestSummary:
    def test_per_scenario_aggregation(self) -> None:
        results = [
            _mkres("01_correlated_subquery", 0, 1.0, True, effective_speedup=2.0),
            _mkres("01_correlated_subquery", 1, 0.0, False),
            _mkres("02_select_star_join", 0, 0.5, True, effective_speedup=1.5),
        ]
        summary = _build_summary(
            results,
            checkpoint="base",
            scenarios=["01_correlated_subquery", "02_select_star_join"],
            seeds_per_scenario=2,
        )
        overall = summary["overall"]
        assert overall["n_episodes"] == 3
        assert overall["pass_rate"] == pytest.approx(2 / 3)
        assert overall["submit_rate"] == 1.0

        per = summary["by_scenario"]
        assert per["01_correlated_subquery"]["pass_rate"] == 0.5
        assert per["01_correlated_subquery"]["mean_terminal_reward"] == pytest.approx(0.5)
        # episode_return must differ from terminal_reward — the fixture
        # subtracts an approximated step tax, so the mean must not equal
        # mean_terminal_reward in a multi-step rollout.
        assert per["01_correlated_subquery"]["mean_episode_return"] != pytest.approx(
            per["01_correlated_subquery"]["mean_terminal_reward"]
        )
        assert per["02_select_star_join"]["pass_rate"] == 1.0
        assert per["02_select_star_join"]["mean_effective_speedup"] == pytest.approx(1.5)
        assert overall["mean_effective_speedup"] == pytest.approx((2.0 + 1.5) / 2)
        assert overall["infinite_speedup_count"] == 0

    def test_infinite_speedup_excluded_from_mean_but_counted(self) -> None:
        # Correct-post-drift submission with a baseline that raises on
        # the post-drift schema → effective_speedup = +∞. The finite
        # mean must ignore it and the infinite count must expose it.
        results = [
            _mkres("07_drift_column_rename", 0, 1.5, True, effective_speedup=math.inf),
            _mkres("07_drift_column_rename", 1, 1.0, True, effective_speedup=4.0),
        ]
        summary = _build_summary(
            results,
            checkpoint="base",
            scenarios=["07_drift_column_rename"],
            seeds_per_scenario=2,
        )
        per = summary["by_scenario"]["07_drift_column_rename"]
        assert per["mean_effective_speedup"] == pytest.approx(4.0)
        assert per["infinite_speedup_count"] == 1
        assert summary["overall"]["infinite_speedup_count"] == 1

    def test_all_infinite_yields_none_mean(self) -> None:
        results = [_mkres("07", 0, 1.0, True, effective_speedup=math.inf)]
        summary = _build_summary(
            results,
            checkpoint="base",
            scenarios=["07"],
            seeds_per_scenario=1,
        )
        # All speedups infinite → finite mean is None (no data to
        # average) but the infinite_count surfaces the full set.
        assert summary["overall"]["mean_effective_speedup"] is None
        assert summary["overall"]["infinite_speedup_count"] == 1

    def test_passed_flag_respected_in_pass_rate(self) -> None:
        """Pass rate derives exclusively from the ``passed`` flag.

        A large terminal_reward does not imply ``passed=True`` and vice
        versa — the caller of :func:`_build_summary` already decided
        the pass condition (per :data:`PASS_REWARD_THRESHOLD`) before
        constructing the :class:`EpisodeResult`.
        """
        results = [
            _mkres("01", 0, 0.4, passed=True),  # below threshold but marked passed
            _mkres("01", 1, 0.9, passed=False),  # above threshold but marked failed
        ]
        summary = _build_summary(
            results,
            checkpoint="base",
            scenarios=["01"],
            seeds_per_scenario=2,
        )
        assert summary["overall"]["pass_rate"] == 0.5

    def test_empty_scenario_filtered(self) -> None:
        summary = _build_summary(
            [_mkres("01_correlated_subquery", 0, 1.0, True)],
            checkpoint="base",
            scenarios=["01_correlated_subquery", "99_nonexistent"],
            seeds_per_scenario=1,
        )
        assert "99_nonexistent" not in summary["by_scenario"]


class TestEpisodeRunner:
    def test_run_one_episode_passes_scenario_id_to_agent_reset(self) -> None:
        class _SpyAgent:
            def __init__(self) -> None:
                self.reset_calls: list[tuple[int | None, str | None]] = []

            def reset(self, seed: int | None = None, scenario_id: str | None = None) -> None:
                self.reset_calls.append((seed, scenario_id))

            def act(self, obs: SqlDriftObservation) -> SqlDriftAction:
                return SqlDriftAction(tool=ToolName.LIST_TABLES, payload=ListTablesPayload())

        class _StubEnv:
            def __init__(self) -> None:
                self.reset_calls: list[tuple[int, str]] = []
                self._obs0 = SqlDriftObservation(
                    step=0,
                    phase=EpisodePhase.DIAGNOSE,
                    budget_steps_remaining=25,
                )
                self._obs1 = SqlDriftObservation(
                    step=1,
                    phase=EpisodePhase.REWRITE,
                    budget_steps_remaining=24,
                    reward_components={k: 0.0 for k in REWARD_COMPONENT_KEYS},
                )
                self._obs1.reward = 0.0
                self._obs1.done = True

            def reset(self, *, seed: int, scenario_id: str) -> SqlDriftObservation:
                self.reset_calls.append((seed, scenario_id))
                return self._obs0

            def step(self, action: SqlDriftAction) -> SqlDriftObservation:
                return self._obs1

            @property
            def state(self) -> SqlDriftState:
                return SqlDriftState(
                    scenario_id="07_drift_column_rename",
                    phase=EpisodePhase.FINALIZE,
                    budget_steps_remaining=24,
                    drift_fired=False,
                    consultations_used=0,
                    submitted=False,
                )

            def effective_speedup(self) -> float | None:
                return None

        env = _StubEnv()
        agent = _SpyAgent()
        _run_one_episode(
            env,
            agent,
            scenario_id="07_drift_column_rename",
            seed=11,
            max_steps=2,
        )
        assert env.reset_calls == [(11, "07_drift_column_rename")]
        assert agent.reset_calls == [(11, "07_drift_column_rename")]


class TestRunEvalSeeding:
    def test_run_eval_seeds_before_loading_agent(self, monkeypatch, tmp_path) -> None:
        events: list[tuple[str, int | str]] = []

        class _StubEnv:
            def close(self) -> None:
                events.append(("close", 0))

        monkeypatch.setattr(eval_mod, "SqlDriftEnvironment", _StubEnv)
        monkeypatch.setattr(eval_mod, "set_seed", lambda seed: events.append(("seed", seed)))
        monkeypatch.setattr(
            eval_mod,
            "load_agent",
            lambda checkpoint, **kwargs: events.append(("load_agent", kwargs["seed"])) or object(),
        )
        monkeypatch.setattr(
            eval_mod,
            "_run_one_episode",
            lambda env, agent, *, scenario_id, seed, max_steps: EpisodeResult(
                scenario_id=scenario_id,
                seed=seed,
                terminal_reward=0.0,
                episode_return=0.0,
                steps=1,
                passed=False,
                submitted=False,
                drift_fired=False,
                wall_ms=1.0,
                reward_components={k: 0.0 for k in REWARD_COMPONENT_KEYS},
                effective_speedup=None,
            ),
        )
        monkeypatch.setattr(eval_mod, "_write_per_episode_csv", lambda results, path: None)
        monkeypatch.setattr(eval_mod, "render_report", lambda summary, results: "ok")

        run_eval(
            checkpoint="base",
            scenarios=["01_correlated_subquery"],
            seeds_per_scenario=1,
            out_dir=tmp_path,
            base_seed=13,
        )

        assert events[:2] == [("seed", 13), ("load_agent", 13)]


class TestFormatSpeedup:
    def test_none_renders_as_empty(self) -> None:
        assert _format_speedup(None) == ""

    def test_infinite_renders_as_inf(self) -> None:
        assert _format_speedup(math.inf) == "inf"

    def test_finite_renders_with_three_decimals(self) -> None:
        assert _format_speedup(1.23456) == "1.235"

    def test_episode_result_csv_row_uses_formatter(self) -> None:
        row = _mkres("01", 0, 1.0, True, effective_speedup=math.inf).as_row()
        assert row["effective_speedup"] == "inf"
        row2 = _mkres("01", 0, 1.0, True, effective_speedup=None).as_row()
        assert row2["effective_speedup"] == ""


class TestRenderReport:
    def test_report_has_expected_sections(self) -> None:
        results = [
            _mkres("01_correlated_subquery", 0, 1.0, True),
            _mkres("02_select_star_join", 0, 0.5, True),
        ]
        summary = _build_summary(
            results,
            checkpoint="base",
            scenarios=["01_correlated_subquery", "02_select_star_join"],
            seeds_per_scenario=1,
        )
        text = render_report(summary, results)
        assert "# SQLDrift evaluation report" in text
        assert "## Per-scenario" in text
        assert "## Reward-component bars" in text
        assert "`01_correlated_subquery`" in text
        assert "100%" in text  # pass rate

    def test_report_empty_handles_gracefully(self) -> None:
        summary = _build_summary([], checkpoint="base", scenarios=[], seeds_per_scenario=0)
        text = render_report(summary, [])
        assert "0" in text
