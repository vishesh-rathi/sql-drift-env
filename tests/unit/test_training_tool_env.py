"""Unit tests for :class:`training.tool_env.SqlDriftToolEnv`.

We do NOT spin up an HTTP server — instead we inject a tiny in-process
client via :func:`training.tool_env.set_client_factory` and verify:

* every public tool method routes to the right :class:`SqlDriftAction`
  and returns a readable string,
* reward bookkeeping (``self.reward``, ``self.episode_return``,
  ``self.terminal_reward``, ``self.done``) reflects the env,
* reset clears per-episode state,
* the TRL tool-surface invariants hold (each tool method has a
  typed signature + ``Args:`` docstring; private helpers start with
  an underscore so TRL does NOT expose them as tools).
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

from models import (
    ConsultDBAResult,
    DescribeTableResult,
    EpisodePhase,
    ExplainQueryResult,
    ListTablesResult,
    ReadChangelogResult,
    RunQueryResult,
    SampleRowsResult,
    SqlDriftAction,
    SqlDriftObservation,
    SubmitRewriteResult,
    ToolError,
    ToolErrorCode,
    ToolName,
)


class _FakeClient:
    """In-process stand-in for :class:`openenv.core.env_client.SyncEnvClient`."""

    def __init__(self) -> None:
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[SqlDriftAction] = []
        self._queue: list[SqlDriftObservation] = []
        self._reset_obs = SqlDriftObservation(
            step=0,
            phase=EpisodePhase.DIAGNOSE,
            budget_steps_remaining=25,
            baseline_sql="SELECT 1",
            schema_synopsis="t(x INT)",
        )
        self.closed = False

    # ------------------------------------------------------------------
    # EnvClient surface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> SqlDriftObservation:
        self.reset_calls.append({"seed": seed, **kwargs})
        return self._reset_obs

    def step(self, action: SqlDriftAction) -> SqlDriftObservation:
        self.step_calls.append(action)
        if not self._queue:
            raise AssertionError("test requested more steps than queued observations")
        return self._queue.pop(0)

    def close(self) -> None:
        self.closed = True

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def queue(self, obs: SqlDriftObservation) -> None:
        self._queue.append(obs)


def _make_observation(
    *,
    tool_result: Any,
    reward: float = 0.0,
    done: bool = False,
    drift_fired: bool = False,
    components: dict[str, float] | None = None,
) -> SqlDriftObservation:
    obs = SqlDriftObservation(
        step=1,
        phase=EpisodePhase.DIAGNOSE,
        budget_steps_remaining=20,
        tool_result=tool_result,
        drift_fired=drift_fired,
        reward_components=components or {},
    )
    obs.reward = reward
    obs.done = done
    return obs


@pytest.fixture()
def fake_client():
    from training import tool_env

    client = _FakeClient()
    tool_env.set_client_factory(lambda: client)
    try:
        yield client
    finally:
        tool_env.set_client_factory(None)


class TestToolEnvSurface:
    """TRL-contract invariants about the class's public methods."""

    def test_every_public_method_has_typed_args_docstring(self) -> None:
        from training.tool_env import SqlDriftToolEnv

        exposed = [
            name
            for name, _ in inspect.getmembers(SqlDriftToolEnv, inspect.isfunction)
            if not name.startswith("_") and name not in {"reset", "close"}
        ]
        assert exposed, "expected at least one tool method"
        for name in exposed:
            method = getattr(SqlDriftToolEnv, name)
            sig = inspect.signature(method)
            doc = (method.__doc__ or "").strip()
            assert doc, f"{name} needs a docstring (TRL tool schema extractor reads it)"
            # All non-self parameters must be annotated so TRL can
            # derive the tool-call JSON schema.
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                assert param.annotation is not inspect.Parameter.empty, (
                    f"{name}({pname}) must have a type annotation"
                )
            if len(sig.parameters) > 1:
                assert "Args:" in doc, f"{name} takes arguments but has no Args: block"


class TestDispatch:
    def test_constructor_uses_explicit_env_url(self, monkeypatch) -> None:
        from training.tool_env import SqlDriftToolEnv

        captured: dict[str, str] = {}

        class _ClientBuilder:
            def __init__(self, *, base_url: str) -> None:
                captured["base_url"] = base_url

            def sync(self) -> _FakeClient:
                return _FakeClient()

        monkeypatch.setattr("training.tool_env.SqlDriftEnv", _ClientBuilder)
        env = SqlDriftToolEnv(env_url="http://trainer:8123")
        try:
            assert captured["base_url"] == "http://trainer:8123"
        finally:
            env.close()

    def test_constructor_uses_env_url_from_environment(self, monkeypatch) -> None:
        import training.tool_env as tool_env

        captured: dict[str, str] = {}

        class _ClientBuilder:
            def __init__(self, *, base_url: str) -> None:
                captured["base_url"] = base_url

            def sync(self) -> _FakeClient:
                return _FakeClient()

        with monkeypatch.context() as m:
            m.setenv("SQL_DRIFT_ENV_URL", "http://from-env:8123")
            reloaded = importlib.reload(tool_env)
            m.setattr(reloaded, "SqlDriftEnv", _ClientBuilder)
            env = reloaded.SqlDriftToolEnv()
            try:
                assert captured["base_url"] == "http://from-env:8123"
            finally:
                env.close()
        importlib.reload(tool_env)

    def test_reset_stores_client_and_resets_bookkeeping(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        env = SqlDriftToolEnv()
        env.reward = 5.0
        env.episode_return = 7.0
        env.done = True

        prelude = env.reset(
            seed=42,
            scenario_id="07_drift_column_rename",
            budget_steps=25,
            enable_dba_oracle=True,
            difficulty="hard",
        )

        assert fake_client.reset_calls == [
            {
                "seed": 42,
                "scenario_id": "07_drift_column_rename",
                "budget_steps": 25,
                "enable_dba_oracle": True,
                "difficulty": "hard",
            }
        ]
        assert env.reward == 0.0
        assert env.episode_return == 0.0
        assert env.done is False
        assert "Baseline query" in prelude
        assert "Schema synopsis" in prelude

    def test_list_tables_routes_to_correct_action(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        fake_client.queue(
            _make_observation(
                tool_result=ListTablesResult(tables=["orders", "users"]),
                reward=-0.02,
            )
        )
        env = SqlDriftToolEnv()
        env.reset(seed=0)
        out = env.list_tables()
        assert "orders" in out and "users" in out
        assert fake_client.step_calls[0].tool is ToolName.LIST_TABLES
        assert env.reward == pytest.approx(-0.02)
        assert env.episode_return == pytest.approx(-0.02)
        assert env.done is False

    def test_describe_and_sample_and_explain_pass_through(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        fake_client.queue(
            _make_observation(
                tool_result=DescribeTableResult(
                    table="orders", columns=[{"name": "id", "type": "INT"}]
                ),
            )
        )
        fake_client.queue(
            _make_observation(
                tool_result=SampleRowsResult(table="orders", columns=["id"], rows=[[1], [2]])
            )
        )
        fake_client.queue(_make_observation(tool_result=ExplainQueryResult(plan="SEQ_SCAN orders")))
        env = SqlDriftToolEnv()
        env.reset(seed=0)
        assert "id: INT" in env.describe_table("orders")
        assert "1" in env.sample_rows("orders", limit=2)
        assert "SEQ_SCAN" in env.explain_query("SELECT id FROM orders")
        tools = [a.tool for a in fake_client.step_calls]
        assert tools == [
            ToolName.DESCRIBE_TABLE,
            ToolName.SAMPLE_ROWS,
            ToolName.EXPLAIN_QUERY,
        ]

    def test_submit_rewrite_sets_terminal_reward_and_done(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        fake_client.queue(
            _make_observation(
                tool_result=SubmitRewriteResult(
                    accepted=True, runtime_ms=12.5, matches_ground_truth=True
                ),
                reward=1.0,
                done=True,
            )
        )
        env = SqlDriftToolEnv()
        env.reset(seed=0)
        msg = env.submit_rewrite("SELECT 1")
        assert "matched ground truth" in msg
        assert env.done is True
        assert env.terminal_reward == pytest.approx(1.0)
        assert env.submitted is True

        # Any further dispatch MUST raise (TRL-sanctioned pattern).
        with pytest.raises(ValueError):
            env.list_tables()

    def test_run_query_tool_error_surfaces_as_string_not_exception(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        fake_client.queue(
            _make_observation(
                tool_result=ToolError(code=ToolErrorCode.DB_ERROR, message="boom"),
                reward=-0.02,
            )
        )
        env = SqlDriftToolEnv()
        env.reset(seed=0)
        out = env.run_query("SELECT bad")
        assert out.startswith("error[db_error]")
        assert env.done is False

    def test_read_changelog_and_consult_dba(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        fake_client.queue(
            _make_observation(
                tool_result=ReadChangelogResult(entries=["renamed user_id -> account_id"])
            )
        )
        fake_client.queue(
            _make_observation(tool_result=ConsultDBAResult(tier=1, hint="watch the join"))
        )
        env = SqlDriftToolEnv()
        env.reset(seed=0)
        assert "account_id" in env.read_changelog()
        assert "watch the join" in env.consult_dba("what changed?")

    def test_post_drift_hints_are_appended_to_tool_result(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        obs = _make_observation(
            tool_result=ReadChangelogResult(entries=["renamed user_id -> account_id"]),
            drift_fired=True,
        )
        obs.learned_hints = "- [drift:column_rename] update old identifiers"
        fake_client.queue(obs)
        env = SqlDriftToolEnv()
        env.reset(seed=0)
        out = env.read_changelog()
        assert "renamed user_id -> account_id" in out
        assert "Learned hints:" in out
        assert "drift:column_rename" in out

    def test_reward_components_are_copied(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        fake_client.queue(
            _make_observation(
                tool_result=ListTablesResult(tables=["t"]),
                reward=0.1,
                components={"r_correct": 0.5, "r_speedup": 0.0},
            )
        )
        env = SqlDriftToolEnv()
        env.reset(seed=0)
        env.list_tables()
        assert env.components["r_correct"] == 0.5


class TestRunQueryRowsRendered:
    def test_run_query_renders_table_with_runtime(self, fake_client) -> None:
        from training.tool_env import SqlDriftToolEnv

        fake_client.queue(
            _make_observation(
                tool_result=RunQueryResult(
                    columns=["a", "b"],
                    rows=[[1, 2], [3, 4]],
                    runtime_ms=0.5,
                    row_count=2,
                ),
                reward=-0.02,
            )
        )
        env = SqlDriftToolEnv()
        env.reset(seed=0)
        out = env.run_query("SELECT a, b FROM t")
        assert "a, b" in out
        assert "1 | 2" in out
        assert "3 | 4" in out
