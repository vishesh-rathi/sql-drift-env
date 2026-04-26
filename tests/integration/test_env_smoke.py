"""P7 — end-to-end smoke tests for :class:`SqlDriftEnvironment`.

In-process `env.reset() → env.step()*` rollouts, no HTTP/WS. Covers:

- Every tool dispatches to the right handler and returns the right
  result / ToolError kind.
- The six-component reward envelope is always populated with the
  canonical keys and sums to ``observation.reward``.
- Public :attr:`env.state` never leaks private runtime fields (seed,
  DB handle, ground-truth hash, baseline SQL / runtime).
- Drift fires at the hybrid-scheduled step and resolves the post-drift
  ground-truth hash; the drift scenario then yields a positive reward
  for a correct post-drift submission.
- Correct-GT submission on a static scenario produces ``r_correct>=0.5``
  and terminates the episode with ``done=True``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from engine.profiler import QueryWatchdogEscalationError
from models import (
    REWARD_COMPONENT_KEYS,
    ConsultDBAPayload,
    DescribeTablePayload,
    ExplainQueryPayload,
    ListTablesPayload,
    ReadChangelogPayload,
    RunQueryPayload,
    SampleRowsPayload,
    SqlDriftAction,
    SubmitRewritePayload,
    ToolError,
    ToolErrorCode,
    ToolName,
)
from scenarios import get_spec
from server import SqlDriftEnvironment

# --- helpers ---------------------------------------------------------------


def _action(tool: ToolName, payload) -> SqlDriftAction:
    return SqlDriftAction(tool=tool, payload=payload)


@pytest.fixture()
def env():
    e = SqlDriftEnvironment()
    yield e
    e.close()


# --- basic lifecycle -------------------------------------------------------


class TestLifecycle:
    def test_reset_sets_step_zero_and_diagnose_phase(self, env):
        obs = env.reset(seed=42, scenario_id="01_correlated_subquery")
        assert obs.step == 0
        assert obs.phase.value == "diagnose"
        assert obs.done is False
        assert obs.reward is None
        assert obs.budget_steps_remaining == 25

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))

    def test_reset_is_repeatable(self, env):
        env.reset(seed=1, scenario_id="01_correlated_subquery")
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        obs = env.reset(seed=2, scenario_id="02_select_star_join")
        assert obs.step == 0
        assert env.state.scenario_id == "02_select_star_join"

    def test_drift_reset_hides_future_schema_change(self, env):
        obs = env.reset(seed=7, scenario_id="07_drift_column_rename")
        assert "Under drift" not in obs.schema_synopsis
        assert "account_id" not in obs.schema_synopsis

    def test_unknown_scenario_raises(self, env):
        with pytest.raises(KeyError):
            env.reset(seed=1, scenario_id="does_not_exist")

    @pytest.mark.parametrize("budget_steps", [0, -5])
    def test_invalid_budget_steps_rejected_at_reset(self, env, budget_steps):
        with pytest.raises(ValidationError, match="budget_steps"):
            env.reset(seed=1, scenario_id="01_correlated_subquery", budget_steps=budget_steps)

    def test_invalid_difficulty_rejected_at_reset(self, env):
        with pytest.raises(ValidationError, match="difficulty"):
            env.reset(seed=1, scenario_id="01_correlated_subquery", difficulty="impossible")

    def test_reset_difficulty_changes_fixture_scale(self, env):
        spec = get_spec("01_correlated_subquery")

        env.reset(seed=42, scenario_id=spec.scenario_id, difficulty="easy")
        easy_obs = env.step(
            _action(ToolName.RUN_QUERY, RunQueryPayload(sql="SELECT COUNT(*) AS n FROM users"))
        )
        easy_count = easy_obs.tool_result.rows[0][0]

        env.reset(seed=42, scenario_id=spec.scenario_id, difficulty="hard")
        hard_obs = env.step(
            _action(ToolName.RUN_QUERY, RunQueryPayload(sql="SELECT COUNT(*) AS n FROM users"))
        )
        hard_count = hard_obs.tool_result.rows[0][0]

        assert easy_count == max(1, spec.base_scale // 2)
        assert hard_count == spec.base_scale * 2

    def test_step_after_submit_raises_without_mutating_state(self, env, monkeypatch):
        obs0 = env.reset(seed=42, scenario_id="01_correlated_subquery")
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        env.step(_action(ToolName.SUBMIT_REWRITE, SubmitRewritePayload(sql=obs0.baseline_sql)))
        prior_steps = env.state.step_count
        monkeypatch.setattr(
            env,
            "_dispatch",
            lambda *args, **kwargs: pytest.fail("step() dispatched after terminal submit"),
        )
        with pytest.raises(ValueError, match="already finished"):
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert env.state.step_count == prior_steps


# --- state sanitization (no private fields in public state) ---------------


class TestStateLeak:
    PRIVATE_TOKENS = (
        "seed",
        "conn",
        "gt_result_hash",
        "baseline_runtime_ms",
        "baseline_sql_canonical",
        "baseline_postdrift_raises",
        "submitted_sql",
        "failed_query_hashes",
        "changelog_entries",
    )

    def test_state_json_leaks_nothing(self, env):
        env.reset(seed=42, scenario_id="07_drift_column_rename")
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        j = env.state.model_dump_json()
        parsed = json.loads(j)
        assert "scenario_id" in parsed  # public field present
        for tok in self.PRIVATE_TOKENS:
            assert tok not in j, f"private token {tok!r} leaked in state JSON: {j}"


# --- tool dispatch (8 tools) -----------------------------------------------


class TestToolDispatch:
    def test_list_tables(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert obs.tool_result.kind == "list_tables_result"
        assert set(obs.tool_result.tables) >= {"orders", "users"}

    def test_describe_table_ok(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(_action(ToolName.DESCRIBE_TABLE, DescribeTablePayload(table="users")))
        assert obs.tool_result.kind == "describe_table_result"
        assert any(c["name"] == "id" for c in obs.tool_result.columns)

    def test_describe_table_unknown(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(_action(ToolName.DESCRIBE_TABLE, DescribeTablePayload(table="nope")))
        assert isinstance(obs.tool_result, ToolError)
        assert obs.tool_result.code == ToolErrorCode.UNKNOWN_TABLE

    def test_sample_rows(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(_action(ToolName.SAMPLE_ROWS, SampleRowsPayload(table="users", limit=3)))
        assert obs.tool_result.kind == "sample_rows_result"
        assert len(obs.tool_result.rows) <= 3

    def test_run_query_ok(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(
            _action(ToolName.RUN_QUERY, RunQueryPayload(sql="SELECT COUNT(*) FROM users"))
        )
        assert obs.tool_result.kind == "run_query_result"
        assert obs.tool_result.row_count == 1

    def test_run_query_db_error(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(
            _action(
                ToolName.RUN_QUERY,
                RunQueryPayload(sql="SELECT * FROM does_not_exist"),
            )
        )
        assert isinstance(obs.tool_result, ToolError)
        assert obs.tool_result.code == ToolErrorCode.DB_ERROR

    def test_repeat_failing_query_gate_fires(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        bad = "SELECT * FROM does_not_exist"
        env.step(_action(ToolName.RUN_QUERY, RunQueryPayload(sql=bad)))
        obs = env.step(_action(ToolName.RUN_QUERY, RunQueryPayload(sql=bad)))
        # First attempt: just the malformed-tool penalty (-0.3) + step tax.
        # Second attempt: both penalties stack.
        assert obs.reward_components["r_gatekeepers"] <= -0.5

    def test_repeat_failing_query_detects_whitespace_variants(self, env):
        """Whitespace/case-only variants of a broken query count as the same repeat.

        Without SQL canonicalization the hash would differ for "select *
        from does_not_exist" vs "SELECT  *  FROM does_not_exist" and
        the repeat-failing-query gate would under-fire.
        """
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        env.step(
            _action(
                ToolName.RUN_QUERY,
                RunQueryPayload(sql="SELECT * FROM does_not_exist"),
            )
        )
        obs = env.step(
            _action(
                ToolName.RUN_QUERY,
                RunQueryPayload(sql="select  *   from   does_not_exist"),
            )
        )
        assert isinstance(obs.tool_result, ToolError)
        assert obs.tool_result.code == ToolErrorCode.DB_ERROR
        assert obs.reward_components["r_gatekeepers"] <= -0.5

    def test_repeat_failing_query_penalty_escalates_across_attempts(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        bad = "SELECT * FROM does_not_exist"
        first = env.step(_action(ToolName.RUN_QUERY, RunQueryPayload(sql=bad)))
        second = env.step(_action(ToolName.RUN_QUERY, RunQueryPayload(sql=bad)))
        third = env.step(_action(ToolName.RUN_QUERY, RunQueryPayload(sql=bad)))
        assert first.reward_components["r_gatekeepers"] > second.reward_components["r_gatekeepers"]
        assert second.reward_components["r_gatekeepers"] > third.reward_components["r_gatekeepers"]

    def test_explain_query(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(
            _action(
                ToolName.EXPLAIN_QUERY,
                ExplainQueryPayload(sql="SELECT 1"),
            )
        )
        assert obs.tool_result.kind == "explain_query_result"
        assert len(obs.tool_result.plan) > 0

    def test_read_changelog_empty_on_static(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(_action(ToolName.READ_CHANGELOG, ReadChangelogPayload()))
        assert obs.tool_result.kind == "read_changelog_result"
        assert obs.tool_result.entries == []

    def test_consult_dba_disabled_by_default(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(_action(ToolName.CONSULT_DBA, ConsultDBAPayload(question="?")))
        assert isinstance(obs.tool_result, ToolError)
        assert obs.tool_result.code == ToolErrorCode.INVALID_TOOL_ARGUMENT

    def test_consult_dba_enabled_consumes_tier(self, env):
        env.reset(
            seed=42,
            scenario_id="01_correlated_subquery",
            enable_dba_oracle=True,
        )
        obs = env.step(_action(ToolName.CONSULT_DBA, ConsultDBAPayload(question="?")))
        assert obs.tool_result.kind == "consult_dba_result"
        assert obs.tool_result.tier == 1
        assert env.state.consultations_used == 1
        # Consult penalty fires.
        assert obs.reward_components["r_consult_dba"] == -0.1


# --- reward envelope -------------------------------------------------------


class TestRewardEnvelope:
    def test_reward_components_always_has_six_canonical_keys(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert set(obs.reward_components.keys()) == set(REWARD_COMPONENT_KEYS)

    def test_reward_equals_component_sum(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert obs.reward == pytest.approx(sum(obs.reward_components.values()))

    def test_first_productive_diagnostic_is_rewarded_more_than_repeat(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        first = env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        second = env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert first.reward > 0.0
        assert first.reward_components["r_step_tax"] > second.reward_components["r_step_tax"]

    def test_read_changelog_after_drift_gets_positive_step_shaping(self, env):
        obs0 = env.reset(seed=7, scenario_id="07_drift_column_rename")
        env.step(_action(ToolName.RUN_QUERY, RunQueryPayload(sql=obs0.baseline_sql)))
        for _ in range(20):
            if env.state.drift_fired:
                break
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        obs = env.step(_action(ToolName.READ_CHANGELOG, ReadChangelogPayload()))
        assert obs.reward_components["r_step_tax"] > 0.0
        assert obs.reward > 0.0

    def test_correct_submission_static_terminates_with_positive_reward(self, env):
        obs0 = env.reset(seed=42, scenario_id="01_correlated_subquery")
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        obs = env.step(
            _action(ToolName.SUBMIT_REWRITE, SubmitRewritePayload(sql=obs0.baseline_sql))
        )
        assert obs.done is True
        assert obs.tool_result.accepted is True
        assert obs.tool_result.matches_ground_truth is True
        assert obs.reward > 0.0
        assert obs.reward_components["r_correct"] >= 0.5

    def test_submit_rewrite_uses_streaming_hash_path(self, env, monkeypatch):
        obs0 = env.reset(seed=42, scenario_id="01_correlated_subquery")
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        gt = obs0.baseline_sql
        gt_hash = env._runtime.gt_result_hash_predrift
        monkeypatch.setattr(
            "server.sql_drift_env_environment.execute_once_timed",
            lambda *args, **kwargs: pytest.fail("submit_rewrite should not materialize full rows"),
        )
        monkeypatch.setattr(
            "server.sql_drift_env_environment.execute_hash_timed",
            lambda conn, sql, *, timeout_s: (gt_hash, 1.0),
            raising=False,
        )
        obs = env.step(_action(ToolName.SUBMIT_REWRITE, SubmitRewritePayload(sql=gt)))
        assert obs.done is True
        assert obs.tool_result.accepted is True
        assert obs.tool_result.matches_ground_truth is True

    def test_wrong_submission_nets_negative(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        obs = env.step(
            _action(
                ToolName.SUBMIT_REWRITE,
                SubmitRewritePayload(sql="SELECT 1 AS x"),
            )
        )
        assert obs.done is True
        assert obs.tool_result.matches_ground_truth is False
        assert obs.reward < 0.0
        assert obs.reward_components["r_correct"] == -1.0

    def test_submit_before_diagnose_is_rejected(self, env):
        obs0 = env.reset(seed=42, scenario_id="01_correlated_subquery")
        obs = env.step(
            _action(ToolName.SUBMIT_REWRITE, SubmitRewritePayload(sql=obs0.baseline_sql))
        )
        assert isinstance(obs.tool_result, ToolError)
        assert obs.tool_result.code == ToolErrorCode.SUBMIT_BEFORE_DIAGNOSE
        assert obs.done is False
        assert env.state.submitted is False


# --- drift trigger ---------------------------------------------------------


class TestDriftTrigger:
    def test_drift_fires_and_resolves_post_drift_gt(self, env):
        obs0 = env.reset(seed=7, scenario_id="07_drift_column_rename")
        # Issue one RUN_QUERY to record first_run_query_step.
        env.step(_action(ToolName.RUN_QUERY, RunQueryPayload(sql=obs0.baseline_sql)))
        # Burn neutral steps until drift should fire.
        for _ in range(20):
            if env.state.drift_fired:
                break
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert env._runtime.drift_fired is True
        assert env.state.drift_fired is True
        assert env._runtime.gt_result_hash_postdrift is not None
        # Baseline should now raise on the renamed column.
        assert env._runtime.baseline_postdrift_raises is True

    def test_changelog_surfaces_after_drift(self, env):
        obs0 = env.reset(seed=7, scenario_id="07_drift_column_rename")
        env.step(
            _action(
                ToolName.RUN_QUERY,
                RunQueryPayload(sql=obs0.baseline_sql),
            )
        )
        for _ in range(20):
            if env.state.drift_fired:
                break
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        obs = env.step(_action(ToolName.READ_CHANGELOG, ReadChangelogPayload()))
        assert obs.tool_result.kind == "read_changelog_result"
        assert len(obs.tool_result.entries) == 1
        assert "renamed" in obs.tool_result.entries[0]
        assert obs.drift_acknowledged is True

    def test_correct_post_drift_submission_nets_highly_positive(self, env):
        obs0 = env.reset(seed=7, scenario_id="07_drift_column_rename")
        env.step(
            _action(
                ToolName.RUN_QUERY,
                RunQueryPayload(sql=obs0.baseline_sql),
            )
        )
        for _ in range(20):
            if env.state.drift_fired:
                break
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        gt_post = env._runtime.instance.gt_sql_postdrift
        obs = env.step(_action(ToolName.SUBMIT_REWRITE, SubmitRewritePayload(sql=gt_post)))
        assert obs.done is True
        assert obs.tool_result.matches_ground_truth is True
        assert obs.reward_components["r_correct"] == 1.0
        assert obs.reward_components["r_drift"] == 0.5
        assert obs.reward > 1.0

    def test_pre_drift_submission_after_drift_raises_tool_error(self, env):
        """Submitting the pre-drift baseline after drift fires is a DB error.

        The renamed column doesn't exist any more, so the submission
        doesn't terminate the episode — it just burns a step with a
        malformed-action penalty; the agent is free to try again with a
        drift-adapted rewrite.
        """
        obs0 = env.reset(seed=7, scenario_id="07_drift_column_rename")
        baseline = obs0.baseline_sql
        env.step(_action(ToolName.RUN_QUERY, RunQueryPayload(sql=baseline)))
        for _ in range(20):
            if env.state.drift_fired:
                break
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        obs = env.step(_action(ToolName.SUBMIT_REWRITE, SubmitRewritePayload(sql=baseline)))
        assert isinstance(obs.tool_result, ToolError)
        assert obs.tool_result.code == ToolErrorCode.DB_ERROR
        assert obs.done is False
        assert env.state.submitted is False
        # Malformed-action gate fires (and step tax).
        assert obs.reward_components["r_gatekeepers"] <= -0.3


# --- budget exhaustion -----------------------------------------------------


class TestBudgetExhaustion:
    def test_budget_exhaustion_terminates(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery", budget_steps=3)
        for _ in range(3):
            obs = env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert obs.done is True
        assert obs.budget_steps_remaining == 0

    def test_step_after_budget_exhaustion_raises_without_dispatch(self, env, monkeypatch):
        env.reset(seed=42, scenario_id="01_correlated_subquery", budget_steps=1)
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        prior_steps = env.state.step_count
        monkeypatch.setattr(
            env,
            "_dispatch",
            lambda *args, **kwargs: pytest.fail("step() dispatched after budget exhaustion"),
        )
        with pytest.raises(ValueError, match="already finished"):
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert env.state.step_count == prior_steps

    def test_watchdog_escalation_marks_episode_terminal(self, env, monkeypatch):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        monkeypatch.setattr(
            env,
            "_dispatch",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                QueryWatchdogEscalationError("worker did not stop")
            ),
        )
        with pytest.raises(QueryWatchdogEscalationError, match="did not stop"):
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert env.state.phase.value == "finalize"
        assert env.state.budget_steps_remaining == 0
        with pytest.raises(ValueError, match="already finished"):
            env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        env._runtime.connection_poisoned = False

    def test_step_count_increments(self, env):
        env.reset(seed=42, scenario_id="01_correlated_subquery")
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        env.step(_action(ToolName.LIST_TABLES, ListTablesPayload()))
        assert env.state.step_count == 2
