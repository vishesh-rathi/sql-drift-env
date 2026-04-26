"""P8 — end-to-end: learned_hints flow from skill library into reset()."""

from __future__ import annotations

from pathlib import Path

from models import (
    ListTablesPayload,
    ReadChangelogPayload,
    RunQueryPayload,
    SqlDriftAction,
    SubmitRewritePayload,
    ToolName,
)
from server import SqlDriftEnvironment
from skill_library import Store


def _diagnostic(env) -> None:
    """Satisfy the phase machine: one diagnostic before submit is mandatory."""
    env.step(SqlDriftAction(tool=ToolName.LIST_TABLES, payload=ListTablesPayload()))


def _submit(env, sql: str):
    return env.step(
        SqlDriftAction(tool=ToolName.SUBMIT_REWRITE, payload=SubmitRewritePayload(sql=sql))
    )


class TestLearnedHintsInReset:
    def test_initial_obs_carries_non_empty_learned_hints(self) -> None:
        env = SqlDriftEnvironment()
        try:
            obs = env.reset(seed=42, scenario_id="01_correlated_subquery")
            # At least the correlated-subquery preseed should match.
            assert obs.learned_hints.startswith("- ")
            assert any(line.startswith("- ") for line in obs.learned_hints.splitlines())
        finally:
            env.close()

    def test_caller_can_override_learned_hints(self) -> None:
        env = SqlDriftEnvironment()
        try:
            obs = env.reset(
                seed=42,
                scenario_id="01_correlated_subquery",
                learned_hints="CUSTOM",
            )
            assert obs.learned_hints == "CUSTOM"
        finally:
            env.close()

    def test_drift_scenario_does_not_leak_drift_card_before_drift(self) -> None:
        env = SqlDriftEnvironment()
        try:
            obs = env.reset(seed=7, scenario_id="07_drift_column_rename")
            assert "drift:column_rename" not in obs.learned_hints
        finally:
            env.close()

    def test_hints_bounded_to_800_chars(self) -> None:
        env = SqlDriftEnvironment()
        try:
            obs = env.reset(seed=1, scenario_id="01_correlated_subquery")
            assert len(obs.learned_hints) <= 800
        finally:
            env.close()

    def test_drift_card_reappears_after_changelog_ack(self) -> None:
        env = SqlDriftEnvironment()
        try:
            obs = env.reset(seed=7, scenario_id="07_drift_column_rename")
            env.step(
                SqlDriftAction(
                    tool=ToolName.RUN_QUERY,
                    payload=RunQueryPayload(sql=obs.baseline_sql),
                )
            )
            for _ in range(20):
                if env.state.drift_fired:
                    break
                env.step(SqlDriftAction(tool=ToolName.LIST_TABLES, payload=ListTablesPayload()))
            obs = env.step(
                SqlDriftAction(tool=ToolName.READ_CHANGELOG, payload=ReadChangelogPayload())
            )
            assert "drift:column_rename" in obs.learned_hints
        finally:
            env.close()


class TestTerminalAppend:
    def test_correct_fast_submission_appends_entry(self, tmp_path: Path) -> None:
        """Drift scenario where baseline raises post-drift → speedup = +∞.

        At CI baseline timings the agent and baseline wall clocks can
        overlap within noise, so we pick the deterministic ``+∞``
        post-drift-raise path to verify that persistence triggers on
        the r_correct == 1 terminal.
        """
        store = Store(directory=tmp_path)
        env = SqlDriftEnvironment(skill_store=store)
        try:
            obs = env.reset(seed=7, scenario_id="07_drift_column_rename")
            env.step(
                SqlDriftAction(
                    tool=ToolName.RUN_QUERY,
                    payload=RunQueryPayload(sql=obs.baseline_sql),
                )
            )
            for _ in range(20):
                if env.state.drift_fired:
                    break
                env.step(SqlDriftAction(tool=ToolName.LIST_TABLES, payload=ListTablesPayload()))
            gt = env._runtime.instance.gt_sql_postdrift
            obs = _submit(env, gt)
            assert obs.done is True
            assert obs.reward_components["r_correct"] == 1.0, obs.reward_components
            persisted = store.read_playbook()
            learned = [e for e in persisted if e.source == "learned"]
            assert len(learned) == 1
            assert learned[0].tag_set

        finally:
            env.close()

    def test_wrong_submission_does_not_append(self, tmp_path: Path) -> None:
        store = Store(directory=tmp_path)
        env = SqlDriftEnvironment(skill_store=store)
        try:
            env.reset(seed=42, scenario_id="01_correlated_subquery")
            _diagnostic(env)
            _submit(env, "SELECT 1 AS x")
            learned = [e for e in store.read_playbook() if e.source == "learned"]
            assert learned == []
        finally:
            env.close()

    def test_no_store_means_no_persistence(self) -> None:
        env = SqlDriftEnvironment()
        try:
            obs = env.reset(seed=42, scenario_id="03_cartesian_join")
            _diagnostic(env)
            _submit(env, obs.baseline_sql)
        finally:
            env.close()
